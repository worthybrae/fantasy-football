"""What was knowable about a player on a given draft day.

Every column here answers "what did the room know when this pick was made",
which is the only thing a model of that pick may use. A feature for a 2022
pick may read 2016-2021 and the player's age in 2022, and nothing later --
the functions below take the draft season and filter on it rather than
leaving that discipline to their callers.

Keyed by `adp_match_key`, the same join key the rest of the pipeline uses,
so these attach to a choice pool without a second name normalizer.
"""
import numpy as np
import pandas as pd

from pipeline.db import read_table
from scoring.board import adp_match_key
from scoring.ppr import compute_ppr_points

COLUMNS = ["key", "age", "ppg_std", "missed_rate", "no_track_record",
           "prod_rank", "trend"]
GAMES = 17
# Drafts happen at the end of August, so age at 1 September of the draft
# year is the age the room would have said out loud.
DRAFT_MONTH_DAY = "-09-01"


def _slope(seasons: np.ndarray, values: np.ndarray) -> float:
    """Points-per-game change per season. Zero with fewer than two seasons,
    which is an absence of evidence rather than a flat trajectory -- callers
    that need to tell those apart read `no_track_record`."""
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(seasons, values, 1)[0])


def attributes_as_of(conn, season: int, rules: dict | None = None) -> pd.DataFrame:
    weekly = read_table(conn, "weekly")
    if weekly.empty:
        return pd.DataFrame(columns=COLUMNS)
    prior = weekly[weekly["season"] < season]
    if prior.empty:
        return pd.DataFrame(columns=COLUMNS)

    wk = prior.copy()
    wk["ppr_points"] = compute_ppr_points(wk, rules)
    per_season = wk.groupby(["player_id", "season"], as_index=False).agg(
        points=("ppr_points", "sum"), games=("week", "nunique"),
        name=("player_display_name", "last"), position=("position", "last"),
        team=("recent_team", "last"))
    per_season["ppg"] = per_season["points"] / per_season["games"]

    rows = []
    for player_id, grp in per_season.groupby("player_id"):
        grp = grp.sort_values("season")
        latest = grp.iloc[-1]
        key = adp_match_key(latest["name"], latest["position"], latest["team"])
        if key is None:
            continue
        ppg = grp["ppg"].to_numpy(dtype=float)
        rows.append({
            "player_id": player_id, "key": key,
            "ppg_std": float(ppg.std(ddof=0)) if len(ppg) > 1 else 0.0,
            "missed_rate": float(max(0.0, 1.0 - grp["games"].sum()
                                     / (GAMES * len(grp)))),
            "no_track_record": False,
            "last_ppg": float(latest["ppg"]),
            "trend": _slope(grp["season"].to_numpy(dtype=float), ppg),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=COLUMNS)

    # Age: `players.birth_date` is the same source the stat-twin matcher
    # already uses. Unknown stays NaN rather than becoming a guess -- the
    # feature builder centres within position and treats NaN as neutral.
    people = read_table(conn, "players")
    if people.empty or "birth_date" not in people.columns:
        out["age"] = np.nan
    else:
        born = people[["gsis_id", "birth_date"]].rename(columns={"gsis_id": "player_id"})
        out = out.merge(born, on="player_id", how="left")
        draft_day = pd.Timestamp(f"{season}{DRAFT_MONTH_DAY}")
        birth = pd.to_datetime(out["birth_date"], errors="coerce")
        out["age"] = (draft_day - birth).dt.days / 365.25

    # Dense, not ordinal: two players who actually tied on production must
    # tie on rank too. `prod_rank` feeds `hype` downstream (a claim about
    # how far the market sits ahead of production), and an invented rank
    # gap between equals would put a number on a distinction that doesn't
    # exist in the data.
    out["prod_rank"] = out["last_ppg"].rank(method="dense", ascending=False).astype(int)
    return out[COLUMNS]
