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
     `rookie` is also forced False for K/DST: the flag means "skill player
     with no NFL history," and K/DST enter from ADP by design, so it would
     otherwise be meaningless noise on every K/DST row.

Column order is not part of the API contract -- consumers must access
columns by name (Task 7 serializes rows to dicts keyed by column name).

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
from scoring import factors, league
from scoring.composite import compute_composite, apply_vor, assign_tiers
from scoring.config import DEFAULT_WEIGHTS, RECENCY_WEIGHTS
from scoring.market import add_market, select_format
from scoring.similarity import player_season_features

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}
_NEUTRAL_FACTORS_FOR_KDST = ["production", "durability", "role", "schedule"]
_ADP_POSITION_ALIASES = {"PK": "K"}
_ADP_TEAM_ALIASES = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "SD": "LAC", "OAK": "LV", "STL": "LA"}

_BOARD_COLUMNS = [
    "player_id", "name", "position", "team", "bye", "production", "durability",
    "role", "environment", "schedule", "composite", "vor", "tier", "market_rank",
    "market_spread", "market_sources", "espn_ppr_rank", "espn_id", "ffc_rank", "edge",
    "rookie", "drafted", "rank",
    "stats", "avail_pct", "ev", "ev_se",
]


def _norm_name(name: str) -> str:
    # Fold accents first (e.g. "Piñeiro" -> "Pineiro") -- the ADP feed and
    # nflverse weekly data disagree on transliteration for some players, and
    # without this a single player ends up as two board rows.
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[.'\-]", "", s.lower())
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return " ".join(parts)


def adp_match_key(name, position, team=None):
    """Join key between an ADP row and a draft pick, as one string.

    Players join on (position, normalized name). DSTs join on team alone,
    exactly the way `_merge_adp` already joins them on the live board: the
    ADP feed labels a defense with its nickname ("Ravens") while ESPN calls
    it "Ravens D/ST", so no name match is possible, and a team has exactly
    one defense. Team abbreviations go through `_ADP_TEAM_ALIASES` on both
    sides so LAR/WSH/JAC-style spellings meet nflverse's.

    Returns None for a DST with no team, which is unmatchable rather than
    matchable-against-anything. Also returns None for a pick with no
    position at all: a draft pick whose ESPN player id is missing from that
    season's directory arrives here with position/name/team all NA, and
    `NA == "DST"` is NA, whose truth value raises.
    """
    if position is None or (not isinstance(position, str) and pd.isna(position)):
        return None
    if position == "DST":
        if team is None or (not isinstance(team, str) and pd.isna(team)):
            return None
        abbrev = str(team).upper()
        return f"DST|{_ADP_TEAM_ALIASES.get(abbrev, abbrev)}"
    return f"{position}|{_norm_name(name)}"


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


def _current_teams(depth: pd.DataFrame) -> pd.DataFrame:
    """Latest known team per player from the newest depth-chart snapshot.

    Weekly stats say who a player *last played* for, so offseason trades and
    signings are invisible there until games are played. The current-season
    depth chart is the authority on `team`; build_board falls back to the
    ADP feed's team, then to the historical weekly team.
    """
    if depth.empty or not {"gsis_id", "team"}.issubset(depth.columns):
        return pd.DataFrame(columns=["player_id", "cur_team"])
    cur = depth
    if "dt" in cur.columns:
        cur = cur.sort_values("dt")
    cur = cur.drop_duplicates("gsis_id", keep="last").dropna(subset=["team"])
    return cur[["gsis_id", "team"]].rename(
        columns={"gsis_id": "player_id", "team": "cur_team"})


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


def _dedupe_adp(adp: pd.DataFrame) -> pd.DataFrame:
    """Keep the best-known (lowest) ADP row per join key.

    Player rows join on (norm, position); DST rows join on team only
    (matching `_merge_adp` and `_add_adp_only_players`). Without this, a
    duplicate ADP row for the same player/team would silently duplicate the
    corresponding board row.
    """
    if adp.empty:
        return adp
    dst = adp[adp["position"] == "DST"].sort_values("adp").drop_duplicates("team", keep="first")
    players = adp[adp["position"] != "DST"].sort_values("adp").drop_duplicates(
        ["norm", "position"], keep="first")
    return pd.concat([players, dst], ignore_index=True)


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


# Payload key -> weekly-table column for passing aggregates, which aren't
# part of player_season_features (that frame feeds twin matching in
# similarity.py and must not change).
_PASS_COLS = {
    "completions": "completions", "attempts": "attempts",
    "pass_yards": "passing_yards", "pass_tds": "passing_tds",
    "interceptions": "passing_interceptions",
}


def _latest_season_stats(weekly: pd.DataFrame) -> pd.DataFrame:
    """Per-player latest-season stat summary, one nested dict per row.

    Serialized like market_sources: the API's NaN->None pass turns a
    missing merge (rookie/K/DST with no weekly rows) into JSON null.
    """
    if weekly.empty:
        return pd.DataFrame(columns=["player_id", "stats"])
    feats = player_season_features(weekly)
    latest = feats[feats["season"] == feats["season"].max()].copy()

    wk = weekly[weekly["season"] == weekly["season"].max()].copy()
    for out, col in _PASS_COLS.items():
        wk[out] = (pd.to_numeric(wk[col], errors="coerce").fillna(0)
                   if col in wk.columns else 0.0)
    passing = wk.groupby("player_id", as_index=False)[list(_PASS_COLS)].sum()
    latest = latest.merge(passing, on="player_id", how="left")
    for out in _PASS_COLS:
        latest[out] = latest[out].fillna(0)

    latest["stats"] = latest.apply(lambda r: {
        "season": int(r["season"]), "games": int(r["games"]),
        "ppg": round(float(r["ppg"]), 1), "points": round(float(r["points"]), 1),
        "carries": int(r["carries"]), "rush_yards": int(r["rush_yards"]),
        "targets": int(r["targets"]), "receptions": int(r["receptions"]),
        "rec_yards": int(r["rec_yards"]), "tds": int(r["tds"]),
        "completions": int(r["completions"]), "attempts": int(r["attempts"]),
        "pass_yards": int(r["pass_yards"]), "pass_tds": int(r["pass_tds"]),
        "interceptions": int(r["interceptions"]),
    }, axis=1)
    return latest[["player_id", "stats"]]


def build_board(conn, weights: dict | None = None,
                settings: "league.LeagueSettings | None" = None) -> pd.DataFrame:
    # The weekly table reaches back to 2016 for profiles/stat twins, but the
    # board scores on the RECENCY_WEIGHTS window only: production would zero
    # out older seasons anyway, and durability counts "possible games" from a
    # player's first season in the data, so deep history would punish
    # veterans for decade-old injuries.
    settings = settings or league.load(conn)
    rules = settings.scoring
    # The consensus ADP is scoring-format-aware: FFC/FantasyPros/MFL/CBS each
    # publish a per-format feed, and the board reflects the LEAGUE's format so
    # both the recommendations and the +/- vs ADP are right for PPR, half-PPR
    # or standard. `fmt` is 'ppr' for an unknown/PPR league, which is today's
    # behavior. FFC (the `adp` table) is format-selected here, before it
    # becomes the board's `adp` column / `ffc_rank`; the other three sources
    # are selected inside add_market. ESPN stays PPR (espn_adp is PPR-only).
    fmt = league.scoring_format(settings)
    weekly = read_table(conn, "weekly")
    if not weekly.empty:
        weekly = weekly[weekly["season"].isin(RECENCY_WEIGHTS)]
    depth = _adapt_depth_charts(read_table(conn, "depth_charts"))
    sched = read_table(conn, "schedules")
    adp = _adapt_adp(select_format(read_table(conn, "adp"), fmt))
    drafted = read_table(conn, "drafted")
    espn = read_table(conn, "espn_adp")
    fp = read_table(conn, "fp_ecr")
    sleeper = read_table(conn, "sleeper_ids")
    mfl = read_table(conn, "mfl_adp")
    cbs = read_table(conn, "cbs_ranks")

    uni = _build_universe(weekly)
    uni["norm"] = uni["name"].map(_norm_name)
    if not adp.empty:
        adp = adp.copy()
        adp["norm"] = adp["adp_name"].map(_norm_name)
        adp = _dedupe_adp(adp)
        uni = _add_adp_only_players(uni, adp)

    # Override the historical team with the current one (depth chart first,
    # ADP second). Must happen before the environment/schedule/bye merges
    # below, which all join on `team`.
    uni = uni.merge(_current_teams(depth), on="player_id", how="left")
    if not adp.empty:
        adp_teams = adp[adp["position"] != "DST"][["norm", "position", "team"]].rename(
            columns={"team": "adp_team"})
        uni = uni.merge(adp_teams, on=["norm", "position"], how="left")
    else:
        uni["adp_team"] = pd.NA
    uni["team"] = uni["cur_team"].fillna(uni["adp_team"]).fillna(uni["team"])
    uni = uni.drop(columns=["cur_team", "adp_team"])

    # -- raw factors --
    if weekly.empty:
        prod = pd.DataFrame(columns=["player_id", "production_raw"])
        dura = pd.DataFrame(columns=["player_id", "durability_raw"])
        role = pd.DataFrame(columns=["player_id", "role_raw"])
    else:
        prod = factors.production_factor(weekly, rules)
        dura = factors.durability_factor(weekly, rules)
        role = factors.role_factor(depth, weekly)
    for raw in (prod, dura, role):
        uni = uni.merge(raw, on="player_id", how="left")

    env = factors.environment_factor(sched) if not sched.empty else pd.DataFrame(columns=["team", "env_raw"])
    uni = uni.merge(env, on="team", how="left")

    prior = weekly[weekly["season"] == weekly["season"].max()] if not weekly.empty else weekly
    sos = factors.schedule_factor(prior, sched, rules) if not (sched.empty or weekly.empty) \
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
    # `rookie` is also meaningless for them (they enter from ADP by design,
    # not because they lack NFL history), so force it False.
    kdst_mask = uni["position"].isin(["K", "DST"])
    uni.loc[kdst_mask, _NEUTRAL_FACTORS_FOR_KDST] = 50.0
    uni.loc[kdst_mask, "rookie"] = False

    uni = _merge_adp(uni, adp)

    # -- score --
    uni["composite"] = compute_composite(uni, weights or DEFAULT_WEIGHTS)
    uni = apply_vor(uni, settings.replacement_ranks)
    uni = assign_tiers(uni)
    uni = uni.sort_values("vor", ascending=False).reset_index(drop=True)
    uni["rank"] = uni.index + 1
    uni = add_market(uni, espn, fp, sleeper, mfl=mfl, cbs=cbs, fmt=fmt)
    # A player ESPN does not rank is not draftable, and the board is the
    # draftable pool -- the simulator builds from it, so anyone left here can
    # be assigned a pick. Retired and out-of-league players were reaching the
    # grid because two of the five sources still carried them: Tyreek Hill
    # (retired) held CBS 184 and FantasyPros 301 against ESPN's "not a
    # fantasy player", and the median sided with the stale pair.
    #
    # `add_market` sets the flag; the exemption for defenses lives with it.
    # This is the same rule the player table applied in the client, moved to
    # where it also governs what can be drafted, and corrected: the client
    # tested `espn_ppr_rank === null`, which a rank of 1899 passes.
    if "espn_unranked" in uni.columns:
        uni = uni[~uni["espn_unranked"]].drop(columns=["espn_unranked"])
    uni = uni.merge(_latest_season_stats(weekly), on="player_id", how="left")

    drafted_ids = set(drafted["player_id"]) if not drafted.empty else set()
    uni["drafted"] = uni["player_id"].isin(drafted_ids)

    # Sim output (Task 13): both tables are absent until run_sim's first
    # write, and even then only carry the most recent run (write_table fully
    # replaces the table each time) -- so a left-merge here is always at most
    # one row per player_id, and every column is null until a sim has run.
    # `drop_duplicates` is not defensive about run_sim, which does write one
    # row per player per run. It is about this frame: `_add_adp_only_players`
    # synthesizes `player_id = "adp_" + norm` with no position in the key, so
    # one normalized name at two positions in the ADP feed (neither matching
    # the weekly universe) produces two board rows carrying the same id. A
    # many-to-many merge would then turn 2 rows into 4.
    sim_avail = read_table(conn, "sim_survival")
    sim_ev = read_table(conn, "sim_results")
    if sim_avail.empty:
        uni["avail_pct"] = pd.NA
    else:
        avail = sim_avail[["player_id", "avail_pct"]].drop_duplicates("player_id")
        uni = uni.merge(avail, on="player_id", how="left")
    if sim_ev.empty:
        uni["ev"] = pd.NA
        uni["ev_se"] = pd.NA
    else:
        ev = sim_ev[["player_id", "ev", "se"]].rename(columns={"se": "ev_se"})
        uni = uni.merge(ev.drop_duplicates("player_id"), on="player_id", how="left")

    return uni[_BOARD_COLUMNS]
