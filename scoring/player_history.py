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

COLUMNS = ["key", "age", "no_track_record", "prod_rank", "trend",
           "usage", "efficiency", "played_share", "peak_gap"]
# Drafts happen at the end of August, so age at 1 September of the draft
# year is the age the room would have said out loud.
DRAFT_MONTH_DAY = "-09-01"

# The three ways a player touches the ball. Summed into one "opportunities"
# count rather than kept apart, because a manager choosing between a running
# back and a wide receiver is comparing how often each gets the ball at all,
# not carries against targets. Per-position meaning comes from centring
# within position downstream, not from separate columns here.
_OPPORTUNITY_COLUMNS = ("carries", "targets", "attempts")


def assert_no_column_collision(frame: pd.DataFrame) -> None:
    """Fail before a left join silently renames both sides.

    `attributes_as_of` is merged onto pools that also carry board columns.
    pandas resolves a name held by both frames by suffixing them `_x`/`_y`,
    so the caller's later `frame["durability"]` raises a KeyError far from
    the cause -- or, worse, finds a column that happens to survive and reads
    the wrong numbers. This turns both into one error naming the column.
    """
    clash = sorted(set(frame.columns) & set(COLUMNS) - {"key"})
    if clash:
        raise ValueError(
            f"{clash} would collide with attributes_as_of on merge; rename "
            "the attribute rather than letting pandas suffix both sides")


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
    wk["opportunities"] = sum(
        pd.to_numeric(wk[col], errors="coerce").fillna(0)
        if col in wk.columns else 0.0
        for col in _OPPORTUNITY_COLUMNS)
    per_season = wk.groupby(["player_id", "season"], as_index=False).agg(
        points=("ppr_points", "sum"), games=("week", "nunique"),
        opportunities=("opportunities", "sum"),
        name=("player_display_name", "last"), position=("position", "last"),
        team=("recent_team", "last"))
    per_season["ppg"] = per_season["points"] / per_season["games"]

    # Games the player's team actually played, so a season cut short by
    # injury reads differently from one played in the 14-game era or ended
    # by a bye. Counted from the same weekly rows rather than hardcoded per
    # era, which keeps it right for partial seasons in the current table.
    team_games = wk.groupby(["recent_team", "season"])["week"].nunique()
    per_season["team_games"] = pd.Series(
        list(zip(per_season["team"], per_season["season"]))).map(
            team_games).to_numpy()

    rows = []
    for player_id, grp in per_season.groupby("player_id"):
        grp = grp.sort_values("season")
        latest = grp.iloc[-1]
        key = adp_match_key(latest["name"], latest["position"], latest["team"])
        if key is None:
            continue
        ppg = grp["ppg"].to_numpy(dtype=float)
        games = float(latest["games"])
        opportunities = float(latest["opportunities"])
        team_games = float(latest["team_games"])
        rows.append({
            "player_id": player_id, "key": key,
            "no_track_record": False,
            "last_ppg": float(latest["ppg"]),
            "trend": _slope(grp["season"].to_numpy(dtype=float), ppg),
            # Volume and rate, kept apart on purpose: a 20-touch grinder and
            # a 6-touch big-play threat can post the same points per game,
            # and managers demonstrably treat those as different players.
            "usage": opportunities / games if games else np.nan,
            "efficiency": (float(latest["points"]) / opportunities
                           if opportunities else np.nan),
            # Named for what it measures, not "durability": `board` already
            # has a `durability` column meaning a within-position percentile,
            # and a left join of two columns with one name silently suffixes
            # both to _x/_y rather than failing.
            "played_share": (games / team_games
                             if team_games and np.isfinite(team_games)
                             else np.nan),
            # How far last season fell short of the best he has ever played.
            # Distinct from `trend`, which is a slope: a player three years
            # past a career year and flat since has a trend near zero and a
            # large gap, and the market treats him as a bounce-back bet.
            "peak_gap": float(ppg.max() - latest["ppg"]),
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
