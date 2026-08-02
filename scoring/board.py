"""Assemble the draft board.

Pipeline: read the raw tables -> build the player universe (latest-season
weekly rows unioned with ADP-only entries for rookies/K/DST) -> compute raw
factors -> normalize each factor to 0-100 within position (missing -> 50)
-> force K/DST factors to neutral except environment -> composite score ->
VOR -> tiers -> rank by VOR descending.

Universe construction:
  1. Start from the latest season in `weekly`, restricted to fantasy-relevant
     positions (QB/RB/WR/TE/K; DST never appears in weekly since it is a
     team-level, not player-level, stat line).
  2. Any ADP row whose normalized (name, position) isn't already in the
     universe becomes a new row -- this is how rookies and K/DST enter the
     board. DST joins to its environment/schedule/bye via `team` rather than
     a name match.
  3. K/DST get every factor neutral (50) except `environment`, since the PPR
     scoring formula does not score kicking/defense stats, so production,
     durability, role and schedule computed from real data would be noise.

Real-data adapters (discovered smoke-testing against data/nfl.duckdb; the
task-4/5 factor functions were written and tested against synthetic tables
with the *old* nflverse schema):
  * depth_charts: the live depth-chart pull no longer has
    formation/depth_team/week columns -- it has `pos_rank`, a 1-indexed
    within-position depth rank, which is exactly what `role_factor` expects
    as `depth_team`. Renamed here so `role_factor` keeps working unmodified.
  * adp: kickers are labeled position "PK" instead of "K", and the LA Rams
    are coded team "LAR" instead of "LA" (every other team abbreviation
    already agrees with weekly/schedules/depth_charts). Both are normalized
    before joining so real kickers/Rams DST match their existing universe
    row instead of being added as spurious "new" (falsely rookie) players
    with no real environment score.
  * names: `_norm_name` also folds accents (e.g. "Piñeiro" -> "pineiro")
    because the ADP feed and nflverse weekly data disagree on
    transliteration for at least one real player, which otherwise produces
    two board rows for the same person.
"""
import re
import unicodedata
import pandas as pd
from pipeline.db import read_table
from scoring import factors
from scoring.composite import compute_composite, apply_vor, assign_tiers
from scoring.config import DEFAULT_WEIGHTS

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}
_NEUTRAL_FACTORS_FOR_KDST = ["production", "durability", "role", "schedule"]
_ADP_POSITION_ALIASES = {"PK": "K"}
_ADP_TEAM_ALIASES = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "SD": "LAC", "OAK": "LV", "STL": "LA"}

_BOARD_COLUMNS = [
    "player_id", "name", "position", "team", "bye", "production", "durability",
    "role", "environment", "schedule", "composite", "vor", "tier", "adp",
    "edge", "rookie", "drafted", "rank",
]


def _norm_name(name: str) -> str:
    # Fold accents first (e.g. "Piñeiro" -> "Pineiro") -- the ADP feed and
    # nflverse weekly data disagree on transliteration for some players, and
    # without this a single player ends up as two board rows.
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[.'\-]", "", s.lower())
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return " ".join(parts)


def _adapt_depth_charts(depth: pd.DataFrame) -> pd.DataFrame:
    """Map the live nflverse depth-chart schema onto what role_factor expects."""
    if depth.empty:
        return depth
    if "depth_team" not in depth.columns and "pos_rank" in depth.columns:
        depth = depth.rename(columns={"pos_rank": "depth_team"})
    if "depth_team" not in depth.columns:
        # Unknown schema variant -- fall back to "no depth data" rather than
        # let role_factor crash; opp-share still contributes to role_raw.
        return pd.DataFrame(columns=["gsis_id", "depth_team"])
    return depth


def _adapt_adp(adp: pd.DataFrame) -> pd.DataFrame:
    if adp.empty:
        return adp
    adp = adp.copy()
    adp["position"] = adp["position"].replace(_ADP_POSITION_ALIASES)
    adp["team"] = adp["team"].replace(_ADP_TEAM_ALIASES)
    return adp


def _build_universe(weekly: pd.DataFrame) -> pd.DataFrame:
    cols = ["player_id", "name", "position", "team"]
    if weekly.empty:
        return pd.DataFrame(columns=cols)
    ps = factors.player_seasons(weekly)
    latest = ps[ps["season"] == ps["season"].max()]
    latest = latest[latest["position"].isin(FANTASY_POSITIONS)]
    uni = latest[["player_id", "player_name", "position", "team"]].rename(
        columns={"player_name": "name"}).drop_duplicates("player_id")
    return uni[cols]


def _add_adp_only_players(uni: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    if adp.empty:
        return uni
    known = set(zip(uni["norm"], uni["position"]))
    unmatched = adp[~adp.apply(lambda r: (r["norm"], r["position"]) in known, axis=1)]
    if unmatched.empty:
        return uni
    extra = unmatched.assign(
        player_id="adp_" + unmatched["norm"].str.replace(" ", "_"),
        name=unmatched["adp_name"])[["player_id", "name", "position", "team", "norm"]]
    return pd.concat([uni, extra], ignore_index=True)


def _merge_adp(uni: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    if adp.empty:
        uni["adp"] = pd.NA
        return uni
    players_adp = adp[adp["position"] != "DST"][["norm", "position", "adp"]]
    uni = uni.merge(players_adp, on=["norm", "position"], how="left")
    dst_adp = adp[adp["position"] == "DST"][["team", "adp"]].rename(columns={"adp": "adp_dst"})
    uni = uni.merge(dst_adp, on="team", how="left")
    uni.loc[uni["position"] == "DST", "adp"] = uni["adp_dst"]
    return uni.drop(columns=["adp_dst"])


def build_board(conn, weights: dict | None = None) -> pd.DataFrame:
    weekly = read_table(conn, "weekly")
    depth = _adapt_depth_charts(read_table(conn, "depth_charts"))
    sched = read_table(conn, "schedules")
    adp = _adapt_adp(read_table(conn, "adp"))
    drafted = read_table(conn, "drafted")

    uni = _build_universe(weekly)
    uni["norm"] = uni["name"].map(_norm_name)
    if not adp.empty:
        adp = adp.copy()
        adp["norm"] = adp["adp_name"].map(_norm_name)
        uni = _add_adp_only_players(uni, adp)

    # -- raw factors --
    if weekly.empty:
        prod = pd.DataFrame(columns=["player_id", "production_raw"])
        dura = pd.DataFrame(columns=["player_id", "durability_raw"])
        role = pd.DataFrame(columns=["player_id", "role_raw"])
    else:
        prod = factors.production_factor(weekly)
        dura = factors.durability_factor(weekly)
        role = factors.role_factor(depth, weekly)
    for raw in (prod, dura, role):
        uni = uni.merge(raw, on="player_id", how="left")

    env = factors.environment_factor(sched) if not sched.empty else pd.DataFrame(columns=["team", "env_raw"])
    uni = uni.merge(env, on="team", how="left")

    prior = weekly[weekly["season"] == weekly["season"].max()] if not weekly.empty else weekly
    sos = factors.schedule_factor(prior, sched) if not (sched.empty or weekly.empty) \
        else pd.DataFrame(columns=["team", "position", "sos_raw"])
    uni = uni.merge(sos, on=["team", "position"], how="left")

    byes = factors.bye_weeks(sched) if not sched.empty else pd.DataFrame(columns=["team", "bye"])
    uni = uni.merge(byes, on="team", how="left")

    # -- normalize each factor to 0-100 within position (NaN -> 50) --
    for raw_col, col in [("production_raw", "production"), ("durability_raw", "durability"),
                         ("role_raw", "role"), ("env_raw", "environment"), ("sos_raw", "schedule")]:
        if raw_col not in uni.columns:
            uni[raw_col] = pd.NA
        uni = factors.normalize_within_position(uni, raw_col, col)

    uni["rookie"] = ~uni["player_id"].isin(set(weekly.get("player_id", pd.Series(dtype=str))))

    # K/DST: the PPR formula doesn't score kicking/defense, so real
    # production/durability/role/schedule signals would just be noise --
    # only environment (team implied points) is a meaningful factor for them.
    kdst_mask = uni["position"].isin(["K", "DST"])
    uni.loc[kdst_mask, _NEUTRAL_FACTORS_FOR_KDST] = 50.0

    uni = _merge_adp(uni, adp)

    # -- score --
    uni["composite"] = compute_composite(uni, weights or DEFAULT_WEIGHTS)
    uni = apply_vor(uni)
    uni = assign_tiers(uni)
    uni = uni.sort_values("vor", ascending=False).reset_index(drop=True)
    uni["rank"] = uni.index + 1
    adp_rank = uni["adp"].rank(method="first")
    uni["edge"] = adp_rank - uni["rank"]

    drafted_ids = set(drafted["player_id"]) if not drafted.empty else set()
    uni["drafted"] = uni["player_id"].isin(drafted_ids)

    return uni[_BOARD_COLUMNS]
