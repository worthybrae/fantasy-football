"""Cross-year player-season similarity (stat twins) and board value-neighbors."""
import numpy as np
import pandas as pd
from scoring.ppr import compute_ppr_points

FEATURES = ["ppg", "games", "target_share", "carry_share",
            "yards_per_opp", "td_per_opp", "rec_pg"]
FEATURE_WEIGHTS = {"ppg": 2.0, "target_share": 2.0, "carry_share": 2.0,
                   "games": 1.0, "yards_per_opp": 1.0, "td_per_opp": 1.0,
                   "rec_pg": 1.0}
MIN_GAMES = 4
DECAY = 2.0

_STAT_COLS = ["targets", "carries", "receiving_yards", "rushing_yards",
              "receiving_tds", "rushing_tds", "receptions"]

def player_season_features(weekly: pd.DataFrame) -> pd.DataFrame:
    wk = weekly.copy()
    wk["ppr_points"] = compute_ppr_points(wk)
    for c in _STAT_COLS:
        wk[c] = (pd.to_numeric(wk[c], errors="coerce").fillna(0)
                 if c in wk.columns else 0.0)
    # Compute team totals (per season, per team)
    team_totals = wk.groupby(["season", "recent_team"])[["targets", "carries"]].sum()
    # For each player-season, sum targets/carries across distinct teams they played for
    player_teams = wk.groupby(["player_id", "season", "recent_team"]).size().reset_index(name="dummy").drop("dummy", axis=1)
    player_teams = player_teams.merge(team_totals.reset_index(), on=["season", "recent_team"])
    team_totals_by_player_season = player_teams.groupby(["player_id", "season"])[["targets", "carries"]].sum()
    team_totals_by_player_season = team_totals_by_player_season.rename(columns={"targets": "team_targets", "carries": "team_carries"})

    g = wk.groupby(["player_id", "season"]).agg(
        name=("player_display_name", "last"), position=("position", "last"),
        games=("week", "nunique"), points=("ppr_points", "sum"),
        targets=("targets", "sum"), carries=("carries", "sum"),
        rec_yards=("receiving_yards", "sum"), rush_yards=("rushing_yards", "sum"),
        rec_tds=("receiving_tds", "sum"), rush_tds=("rushing_tds", "sum"),
        receptions=("receptions", "sum")).reset_index()

    g = g.merge(team_totals_by_player_season, left_on=["player_id", "season"], right_index=True, how="left")
    opps = (g["targets"] + g["carries"]).replace(0, np.nan)
    g["ppg"] = g["points"] / g["games"]
    g["tds"] = g["rec_tds"] + g["rush_tds"]
    g["target_share"] = g["targets"] / g["team_targets"].replace(0, np.nan)
    g["carry_share"] = g["carries"] / g["team_carries"].replace(0, np.nan)
    g["yards_per_opp"] = (g["rec_yards"] + g["rush_yards"]) / opps
    g["td_per_opp"] = g["tds"] / opps
    g["rec_pg"] = g["receptions"] / g["games"]
    return g

def find_twins(weekly: pd.DataFrame, player_id: str, top_n: int = 5) -> dict | None:
    all_feats = player_season_features(weekly)
    feats = all_feats[all_feats["games"] >= MIN_GAMES]
    mine = feats[feats["player_id"] == player_id]
    if mine.empty:
        return None
    target = mine.sort_values("season").iloc[-1]
    pool = feats[feats["position"] == target["position"]].copy()
    zcols = []
    for f in FEATURES:
        col = pool[f].astype(float)
        std = col.std()
        z = f + "_z"
        if pd.isna(std) or std == 0:
            pool[z] = 0.0
        else:
            pool[z] = ((col - col.mean()) / std).fillna(0.0)
        zcols.append(z)
    w = np.array([FEATURE_WEIGHTS[f] for f in FEATURES])
    tvec = pool.loc[(pool["player_id"] == player_id)
                    & (pool["season"] == target["season"]), zcols
                    ].iloc[0].to_numpy(dtype=float)
    cand = pool[pool["player_id"] != player_id].copy()
    if cand.empty:
        return {"mode": "stat_twins", "target_season": int(target["season"]),
                "players": []}
    diffs = cand[zcols].to_numpy(dtype=float) - tvec
    cand["distance"] = np.sqrt(((diffs ** 2) * w).sum(axis=1) / w.sum())
    cand["similarity"] = (100 * np.exp(-cand["distance"] / DECAY)).round(1)
    nxt = all_feats[["player_id", "season", "ppg"]].copy()
    nxt["season"] = nxt["season"] - 1
    nxt = nxt.rename(columns={"ppg": "next_ppg"})
    cand = cand.merge(nxt, on=["player_id", "season"], how="left")
    cand = cand.sort_values("distance").head(top_n)
    players = [{"player_id": r["player_id"], "name": r["name"],
                "season": int(r["season"]),
                "similarity": float(r["similarity"]),
                "ppg": round(float(r["ppg"]), 1),
                "next_ppg": (None if pd.isna(r["next_ppg"])
                             else round(float(r["next_ppg"]), 1))}
               for _, r in cand.iterrows()]
    return {"mode": "stat_twins", "target_season": int(target["season"]),
            "players": players}

def value_neighbors(board: pd.DataFrame, player_id: str, top_n: int = 5) -> dict:
    me = board[board["player_id"] == player_id].iloc[0]
    pool = board[(board["position"] == me["position"])
                 & (board["player_id"] != player_id)].copy()
    pool["_d"] = (pool["vor"] - me["vor"]).abs()
    pool = pool.sort_values("_d").head(top_n)
    players = [{"player_id": r["player_id"], "name": r["name"], "season": None,
                "similarity": None, "ppg": None, "next_ppg": None,
                "rank": int(r["rank"]),
                "adp": (None if pd.isna(r["adp"]) else float(r["adp"]))}
               for _, r in pool.iterrows()]
    return {"mode": "value_neighbors", "players": players}
