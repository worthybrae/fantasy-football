"""Cross-year player-season similarity (stat twins) and board value-neighbors."""
import numpy as np
import pandas as pd
from scoring.ppr import compute_ppr_points, normalize_rules

FEATURES = ["ppg", "games", "target_share", "carry_share",
            "yards_per_opp", "td_per_opp", "rec_pg"]
FEATURE_WEIGHTS = {"ppg": 2.0, "target_share": 2.0, "carry_share": 2.0,
                   "games": 1.0, "yards_per_opp": 1.0, "td_per_opp": 1.0,
                   "rec_pg": 1.0}
MIN_GAMES = 4
DECAY = 2.0

_STAT_COLS = ["targets", "carries", "receiving_yards", "rushing_yards",
              "receiving_tds", "rushing_tds", "receptions"]

def player_season_features(weekly: pd.DataFrame,
                           rules: dict | None = None) -> pd.DataFrame:
    """Per player-season aggregate, scored under `rules` (None = full PPR).

    `rules` is the league's `settings.scoring`, threaded in the same way
    scoring/factors.py and scoring/player_history.py already take it. It is
    not decoration: `ppg` is one of the seven FEATURES the stat-twin distance
    is computed over AND the number `find_twins` reports as `next_ppg`, so in
    a half-PPR league an unscored frame picks different comparables and
    attaches a full-PPR forecast to them. `points` is what `pos_finish`
    ("finished WR4") ranks on, and `ppg` is what the board's `stats.ppg` --
    the fallback rung of `board.projections` -- reads.

    None keeps every existing caller byte-identical.
    """
    wk = weekly.copy()
    wk["ppr_points"] = compute_ppr_points(wk, normalize_rules(rules))
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

def _age_in_season(birth_date, season: int) -> int | None:
    """Full years old on Sept 1 (opening week) of the season year."""
    if birth_date is None or pd.isna(birth_date):
        return None
    bd = pd.Timestamp(birth_date)
    return season - bd.year - (1 if (bd.month, bd.day) > (9, 1) else 0)


def find_twins(weekly: pd.DataFrame, player_id: str, top_n: int = 5,
               players: pd.DataFrame | None = None, *,
               season_features: pd.DataFrame | None = None,
               rules: dict | None = None) -> dict | None:
    """`weekly` is used for exactly one thing -- `player_season_features` --
    so a caller that already has that frame can hand it over as
    `season_features` and `weekly` is then ignored entirely. That is not an
    optimisation detail of this function so much as a measured fact about
    its caller: scoring/profile.py computed the same 174,373-row aggregate
    here AND in `season_summaries` on every profile click, 0.211s each,
    and threw both away. See scoring/profile_cache.py. Passing nothing
    keeps the old behaviour byte for byte.

    `rules` is used ONLY on the path that computes the features here; a
    caller supplying `season_features` has already priced them and this
    argument is ignored, exactly as `weekly` is. Getting that wrong in the
    other direction would be silent: a half-PPR profile handing over a
    PPR-priced cached frame would match twins on PPR and say nothing."""
    all_feats = (player_season_features(weekly, rules) if season_features is None
                 else season_features)
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
    # A comp's value is the "what happened next" trend signal. No following
    # season in the data (retired, out of the league, or the season hasn't
    # been played yet -- which also drops every same-year comp) means no
    # signal, so the comp is excluded rather than shown with a dangling arrow.
    cand = cand[cand["next_ppg"].notna()]
    # Age matching: comps must have been exactly the target's age during
    # their comp season, so the trend signal tracks the age curve. Missing
    # birth dates (or no players table yet) degrade to the unfiltered pool
    # for the target, and exclude only the individual unknown candidates.
    target_age = None
    if players is not None and not players.empty and "birth_date" in players.columns:
        births = players.set_index("gsis_id")["birth_date"]
        target_age = _age_in_season(births.get(player_id), int(target["season"]))
        cand["age"] = [
            _age_in_season(births.get(pid), int(s))
            for pid, s in zip(cand["player_id"], cand["season"])]
        if target_age is not None:
            cand = cand[cand["age"] == target_age]
    else:
        cand["age"] = None
    cand = cand.sort_values("distance").head(top_n)
    comps = [{"player_id": r["player_id"], "name": r["name"],
              "season": int(r["season"]),
              "similarity": float(r["similarity"]),
              "ppg": round(float(r["ppg"]), 1),
              "next_ppg": (None if pd.isna(r["next_ppg"])
                           else round(float(r["next_ppg"]), 1)),
              "age": (None if r["age"] is None or pd.isna(r["age"])
                      else int(r["age"]))}
             for _, r in cand.iterrows()]
    return {"mode": "stat_twins", "target_season": int(target["season"]),
            "target_age": target_age, "players": comps}

def value_neighbors(board: pd.DataFrame, player_id: str, top_n: int = 5) -> dict:
    me = board[board["player_id"] == player_id].iloc[0]
    pool = board[(board["position"] == me["position"])
                 & (board["player_id"] != player_id)].copy()
    pool["_d"] = (pool["vor"] - me["vor"]).abs()
    pool = pool.sort_values("_d").head(top_n)
    players = [{"player_id": r["player_id"], "name": r["name"], "season": None,
                "similarity": None, "ppg": None, "next_ppg": None,
                "rank": int(r["rank"]),
                "market_rank": (None if pd.isna(r["market_rank"]) else float(r["market_rank"]))}
               for _, r in pool.iterrows()]
    return {"mode": "value_neighbors", "players": players}


# What "similar player" is scored on, and how much each part counts. Not the
# same list as FEATURES above and deliberately so: `find_twins` matches one
# SEASON against seasons that already happened, so a stat line is the whole
# of it. This one matches a PLAYER against the players he is being drafted
# beside, where the projection is the loudest single fact about him and his
# body is a real part of what makes two backs the same kind of back.
#
# The stat line still outweighs the rest put together. Height and weight are
# priced low on purpose: they separate a slot receiver from an X, which is
# worth something, but two players are not similar BECAUSE they are the same
# size, and a heavier weighting turns the card into a combine list.
SIMILAR_WEIGHTS = {
    "proj_points": 2.0,
    "ppg": 2.0,
    "target_share": 1.5,
    "carry_share": 1.5,
    "yards_per_opp": 1.0,
    "td_per_opp": 1.0,
    "rec_pg": 1.0,
    "games": 0.5,
    "age": 1.0,
    "height": 0.5,
    "weight": 0.5,
}

# What fraction of the total weight a pair has to share before a distance
# between them means anything. Two players compared on one feature alone
# would score 100 for being the same size, or for carrying the same
# projection, which is the failure this guards.
#
# 0.3 is where a rookie lands: he has no stat line, so he is compared on his
# projection, his age and his build -- four facts of the eleven, about a
# third of the weight -- and that is a real if thin comparison, which is why
# the line sits under it rather than over it. Projection alone (a rookie in a
# database with no bio columns yet) is 0.17 and drops out, which is the
# intended half of the same rule: one number in common is not a resemblance.
MIN_SHARED_WEIGHT = 0.3


def _latest_season_stats(season_features: pd.DataFrame | None) -> pd.DataFrame:
    """Each player's most recent season on file, one row per player.

    The most recent SEASON HE PLAYED, not last year specifically -- a back
    who missed all of last season is still the player his last real year
    describes, and the alternative (no stat line at all) drops him out of
    every comparison on the board.
    """
    cols = ["player_id", "ppg", "games", "target_share", "carry_share",
            "yards_per_opp", "td_per_opp", "rec_pg"]
    if season_features is None or season_features.empty:
        return pd.DataFrame(columns=cols)
    have = [c for c in cols if c in season_features.columns]
    latest = (season_features.sort_values("season")
              .drop_duplicates("player_id", keep="last"))
    return latest[have].copy()


def similar_players(board: pd.DataFrame, player_id: str, *,
                    season_features: pd.DataFrame | None = None,
                    players: pd.DataFrame | None = None,
                    season: int | None = None, top_n: int = 5) -> dict | None:
    """The players on THIS year's board most like this one, best first.

    A different question from `find_twins`, which asks what has happened
    before: this one asks who else is on the shelf. The pool is the board
    itself, filtered to his own position -- a tight end is not "like" a
    running back in any sense a drafter can use, however close their numbers
    land -- and the score is a percentage, so the rows can be ranked against
    each other rather than read as bare distances.

    Every part of `SIMILAR_WEIGHTS` a pair actually shares is used, and the
    distance is divided by the weight of exactly those parts. That is what
    lets a rookie (no stat line) and a veteran (no missing anything) sit in
    the same pool without the rookie either crashing the sort or matching
    everybody at 100: he is compared on what he has, and dropped when what
    he has is too little to mean anything (MIN_SHARED_WEIGHT).

    Returns None when the player is not on the board, or when nobody at his
    position is left to compare him against.
    """
    if board is None or board.empty or "player_id" not in board.columns:
        return None
    me = board[board["player_id"] == player_id]
    if me.empty:
        return None
    me = me.iloc[0]

    keep = ["player_id", "name", "position", "proj_points", "rank",
            "market_rank"]
    pool = board[board["position"] == me["position"]].copy()
    pool = pool[[c for c in keep if c in pool.columns]]
    pool = pool.merge(_latest_season_stats(season_features), on="player_id",
                      how="left")

    # Bio, when the players table carries it. `height`/`weight` arrived with
    # a later import (pipeline/sources.py) and a database refreshed before
    # that has neither column -- so they are merged if present and simply
    # absent from the distance if not, rather than being required.
    pool["age"] = None
    if players is not None and not players.empty and "gsis_id" in players.columns:
        bio = players.set_index("gsis_id")
        if "birth_date" in bio.columns and season is not None:
            births = bio["birth_date"]
            pool["age"] = [_age_in_season(births.get(pid), int(season))
                           for pid in pool["player_id"]]
        for col in ("height", "weight"):
            if col in bio.columns:
                pool[col] = [pd.to_numeric(bio[col].get(pid), errors="coerce")
                             for pid in pool["player_id"]]
        # Not a feature -- nobody is similar BECAUSE of his photograph. It
        # rides along because the card is a row of faces and the payload is
        # where the card gets them; fetching them separately would be a
        # second read of the same table for the same five rows.
        if "headshot" in bio.columns:
            pool["headshot"] = [bio["headshot"].get(pid)
                                for pid in pool["player_id"]]

    feats = [f for f in SIMILAR_WEIGHTS if f in pool.columns]
    if not feats:
        return None
    # z-scored over his own position, so a target share and a weight in
    # pounds can be added up at all. A feature with no spread (or one value)
    # contributes nothing rather than dividing by zero.
    z = pd.DataFrame(index=pool.index)
    for f in feats:
        col = pd.to_numeric(pool[f], errors="coerce").astype(float)
        std = col.std()
        z[f] = 0.0 if pd.isna(std) or std == 0 else (col - col.mean()) / std
        z.loc[col.isna(), f] = np.nan

    mine = pool.index[pool["player_id"] == player_id]
    if len(mine) == 0:
        return None
    tvec = z.loc[mine[0]].to_numpy(dtype=float)
    w = np.array([SIMILAR_WEIGHTS[f] for f in feats], dtype=float)
    total_w = w.sum()

    cand = pool[pool["player_id"] != player_id]
    if cand.empty:
        return {"season": None if season is None else int(season),
                "position": str(me["position"]), "players": []}
    mat = z.loc[cand.index].to_numpy(dtype=float)
    shared = ~np.isnan(mat) & ~np.isnan(tvec)
    shared_w = (shared * w).sum(axis=1)
    diffs = np.where(shared, mat - tvec, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.sqrt(((diffs ** 2) * w).sum(axis=1) / shared_w)
    out = cand.copy()
    out["distance"] = dist
    out["shared_weight"] = shared_w / total_w
    out = out[(out["shared_weight"] >= MIN_SHARED_WEIGHT)
              & out["distance"].notna()]
    # Same curve `find_twins` scores on, so 80% means the same closeness on
    # both cards of the same popup.
    out["similarity"] = (100 * np.exp(-out["distance"] / DECAY)).round(1)
    out = out.sort_values(["distance", "player_id"]).head(top_n)

    def _num(value, digits=1):
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)

    def _text(value):
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return None
        return str(value)

    rows = [{"player_id": r["player_id"], "name": r["name"],
             "headshot": _text(r.get("headshot")),
             "similarity": float(r["similarity"]),
             "proj_points": _num(r.get("proj_points")),
             "ppg": _num(r.get("ppg")),
             "age": (None if r.get("age") is None or pd.isna(r.get("age"))
                     else int(r["age"])),
             "height": _num(r.get("height"), 0),
             "weight": _num(r.get("weight"), 0),
             "rank": (None if pd.isna(r.get("rank")) else int(r["rank"])),
             "market_rank": _num(r.get("market_rank"))}
            for _, r in out.iterrows()]
    return {"season": None if season is None else int(season),
            "position": str(me["position"]), "players": rows}
