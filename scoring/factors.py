import pandas as pd
from scoring.ppr import compute_ppr_points
from scoring.config import RECENCY_WEIGHTS

def player_seasons(weekly: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    wk = weekly.copy()
    wk["ppr_points"] = compute_ppr_points(wk, rules)
    grp = wk.groupby(["player_id", "season"]).agg(
        points=("ppr_points", "sum"), games=("week", "nunique"),
        player_name=("player_display_name", "last"),
        position=("position", "last"), team=("recent_team", "last")).reset_index()
    grp["ppg"] = grp["points"] / grp["games"]
    return grp

def production_factor(weekly: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    ps = player_seasons(weekly, rules)
    ps["w"] = ps["season"].map(RECENCY_WEIGHTS).fillna(0.0)
    ps["wppg"] = ps["ppg"] * ps["w"]
    agg = ps.groupby("player_id").agg(wsum=("wppg", "sum"), wtot=("w", "sum")).reset_index()
    agg["production_raw"] = agg["wsum"] / agg["wtot"].replace(0, float("nan"))
    return agg[["player_id", "production_raw"]]

def durability_factor(weekly: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    ps = player_seasons(weekly, rules)
    first = ps.groupby("player_id")["season"].min()
    games = ps.groupby("player_id")["games"].sum()
    latest = ps["season"].max()
    possible = (latest - first + 1) * 17
    out = (games / possible).rename("durability_raw").reset_index()
    return out[["player_id", "durability_raw"]]

def role_factor(depth_charts: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    # Process depth charts if available
    dc = depth_charts.copy()
    if "formation" in dc.columns:
        dc = dc[dc["formation"] == "Offense"]
    if not dc.empty:
        if "week" in dc.columns and dc["week"].notna().any():
            dc = dc[dc["week"] == dc["week"].max()]
        dc["depth_rank_score"] = 1.0 / pd.to_numeric(dc["depth_team"], errors="coerce").clip(lower=1)
        dc = dc.rename(columns={"gsis_id": "player_id"})[["player_id", "depth_rank_score"]]
        dc = dc.groupby("player_id", as_index=False)["depth_rank_score"].max()
    else:
        # Create empty but mergeable dataframe for depth scores when data is absent
        dc = pd.DataFrame(columns=["player_id", "depth_rank_score"])

    # Always compute opportunity share from weekly data
    wk = weekly[weekly["season"] == weekly["season"].max()].copy()
    for c in ("targets", "carries"):
        wk[c] = pd.to_numeric(wk[c], errors="coerce").fillna(0) if c in wk.columns else 0.0
    wk["opps"] = wk["targets"] + wk["carries"]
    team_opps = wk.groupby("recent_team")["opps"].transform("sum")
    wk["opp_share"] = wk["opps"] / team_opps.replace(0, float("nan"))
    share = wk.groupby("player_id", as_index=False)["opp_share"].sum()

    # Outer merge preserves both signals; 50/50 blend handles missing data with skipna
    out = dc.merge(share, on="player_id", how="outer")
    # 50/50 blend; if one side missing, use the other alone
    out["role_raw"] = out[["depth_rank_score", "opp_share"]].mean(axis=1, skipna=True)
    return out[["player_id", "role_raw"]]

def environment_factor(schedules: pd.DataFrame) -> pd.DataFrame:
    sc = schedules.dropna(subset=["total_line", "spread_line"]).copy()
    home = pd.DataFrame({"team": sc["home_team"],
                         "implied": (sc["total_line"] + sc["spread_line"]) / 2})
    away = pd.DataFrame({"team": sc["away_team"],
                         "implied": (sc["total_line"] - sc["spread_line"]) / 2})
    both = pd.concat([home, away])
    if both.empty:
        return pd.DataFrame(columns=["team", "env_raw"])
    return both.groupby("team", as_index=False)["implied"].mean().rename(
        columns={"implied": "env_raw"})

def schedule_factor(weekly_prior: pd.DataFrame, schedules: pd.DataFrame,
                    rules: dict | None = None) -> pd.DataFrame:
    wk = weekly_prior.copy()
    wk["ppr_points"] = compute_ppr_points(wk, rules)
    weeks = wk.groupby("opponent_team")["week"].nunique().rename("def_games")
    allowed = wk.groupby(["opponent_team", "position"])["ppr_points"].sum().reset_index()
    allowed = allowed.merge(weeks, on="opponent_team")
    allowed["fpa_pg"] = allowed["ppr_points"] / allowed["def_games"]

    games = pd.concat([
        schedules.rename(columns={"home_team": "team", "away_team": "opp"})[["team", "opp"]],
        schedules.rename(columns={"away_team": "team", "home_team": "opp"})[["team", "opp"]]])
    merged = games.merge(allowed, left_on="opp", right_on="opponent_team")
    return merged.groupby(["team", "position"], as_index=False)["fpa_pg"].mean().rename(
        columns={"fpa_pg": "sos_raw"})

def bye_weeks(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules[schedules["week"] <= 18]
    playing = pd.concat([
        reg[["week", "home_team"]].rename(columns={"home_team": "team"}),
        reg[["week", "away_team"]].rename(columns={"away_team": "team"})])
    rows = []
    for team, grp in playing.groupby("team"):
        missing = sorted(set(range(1, 19)) - set(grp["week"]))
        rows.append({"team": team, "bye": missing[0] if missing else None})
    return pd.DataFrame(rows)

def normalize_within_position(df: pd.DataFrame, raw_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = (out.groupby("position")[raw_col].rank(pct=True) * 100)
    out[out_col] = out[out_col].fillna(50.0)
    return out
