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
  3. K/DST get every factor neutral (50) except `environment`, since a
     factor computed from stats the league does not score would be noise.
     A league that DOES price kicking (scoring/league.py maps ESPN's kicking
     stat ids) gets a kicker's production, durability and schedule back --
     see `_neutral_factors` for the argument factor by factor. DST is
     unconditional: nothing in this database can compute a defense.
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
import warnings
import numpy as np
import pandas as pd
from pipeline.db import read_table
from scoring import factors, league
from scoring.composite import compute_composite, apply_vor, assign_tiers
from scoring.config import DEFAULT_WEIGHTS, RECENCY_WEIGHTS
from scoring.market import add_market, select_format
from scoring.ppr import compute_ppr_points, normalize_rules, prices_kicking
from scoring.similarity import player_season_features

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}
_NEUTRAL_FACTORS_FOR_KDST = ["production", "durability", "role", "schedule"]
# The subset of those a kicker gets back once the league actually prices
# kicking -- see `_neutral_factors` for the argument, factor by factor.
_KICKER_FACTORS_THAT_BECOME_REAL = ["production", "durability", "schedule"]
_ADP_POSITION_ALIASES = {"PK": "K"}
_ADP_TEAM_ALIASES = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "SD": "LAC", "OAK": "LV", "STL": "LA"}

_BOARD_COLUMNS = [
    "player_id", "name", "position", "team", "bye", "production", "durability",
    "role", "environment", "schedule", "composite", "proj_points",
    # How much this league's scoring rules re-price ESPN's PPR-shaped season
    # projection for this player: 1.0 in a PPR league, ~0.83 for a
    # high-reception WR in half-PPR, ~0.67 in standard. On the board rather
    # than kept inside `projections()` because two other places show a number
    # derived from the same raw `espn_proj` and would otherwise quote it in
    # PPR while everything beside it is in the league's currency -- the
    # profile's `summary.proj_ppg` (scoring/profile.py) and the draft room's
    # trending icon (api/live.py `_board_cell`). One column, computed once,
    # so those three cannot disagree. It is also the honest way to SHOW the
    # estimate rather than bury it: see `projection_scale`.
    "proj_scale", "vor",
    "tier", "market_rank",
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


def _neutral_factors(position: str, rules: dict | None) -> list:
    """Which factors this position must have pinned to 50 under `rules`.

    DST: all four, always. There is no team-defense row in `weekly` to
    compute production, durability or schedule from, and no defensive
    scoring this app can express (scoring/league.py's `from_espn` gives the
    measurements). A defense's numbers here would be fabricated, not
    computed, so nothing changes for DST and nothing should.

    K under a league that prices no kicking: all four, unchanged. Every
    kicking column is unscored, so `production` is a recency-weighted
    average of zero and `schedule` is points allowed to kickers of zero.
    This is today's behaviour for the owner's live board and it is bit for
    bit what it was.

    K under a league that DOES price kicking -- three of the four come back:

      * production: a recency-weighted per-game average of the league's own
        kicking points. Real signal, and the one number that separates
        kickers.
      * schedule: prior-season kicking points allowed per game by each
        opponent. The profile's schedule card shows the same quantity
        (scoring/profile.py `weekly_difficulty`) and is released on the same
        condition, so the two cannot disagree.
      * durability: games played over games possible. This one was NEVER a
        scoring artefact -- `factors.durability_factor` counts weeks, and
        counts them identically whatever the rules say -- so a kicker's
        durability has been real data thrown away all along. It is released
        WITH the other two rather than unconditionally, deliberately: an
        unconditional release would move the live PPR board's K rows in the
        middle of the owner's draft, for a factor nobody asked about. The
        condition is the safety property, not the argument.

      * role stays neutral. `factors.role_factor` blends depth-chart rank
        with share of team targets+carries, and the second half is
        structurally zero for every kicker who ever played -- so half the
        blend is not "missing" (which the skipna mean would handle) but a
        real zero that drags every kicker's role toward the floor. That is
        the same kind of noise the neutral existed to suppress, and scoring
        kicking does not touch it.
    """
    if position == "DST":
        return _NEUTRAL_FACTORS_FOR_KDST
    if position == "K" and prices_kicking(rules):
        return [f for f in _NEUTRAL_FACTORS_FOR_KDST
                if f not in _KICKER_FACTORS_THAT_BECOME_REAL]
    return _NEUTRAL_FACTORS_FOR_KDST


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


GAMES = 17
# Floor for players with no projection and no stat history, per position, so
# a K or a rookie DST never lands as NaN inside the lineup optimizer.
POSITION_FLOOR = {"QB": 180.0, "RB": 80.0, "WR": 80.0, "TE": 60.0,
                  "K": 110.0, "DST": 100.0}


# A player needs at least this many full-PPR points per game in his latest
# season before the ratio of his league-scored to his PPR-scored production is
# trusted as a re-pricing factor. Below it the denominator is noise -- one
# garbage-time catch in one game -- and a ratio built on noise is worse than
# the position median, which is what such a player falls back to. One point a
# game is far under any player who affects a draft (the 249-row real board's
# lowest non-zero latest-season PPR ppg is a fraction of a point, from players
# with a single appearance).
_MIN_PPG_FOR_SCALE = 1.0


# Games a KICKER must have played before `projections()`'s second rung will
# extrapolate his per-game rate to a full season. More than half of a
# 17-game year: the rung's claim is "this is what he would do over a
# season", and a kicker who has not played half of one has not shown that.
#
# WHY ONLY KICKERS, AND WHY THIS EXISTS AT ALL. The rung is
# `stats.ppg * GAMES`, reached only by players ESPN publishes no season
# projection for. ESPN projects every skill player who matters, so for
# QB/RB/WR/TE that rung is a rare edge case. It projects almost no fringe
# kicker, so the moment kicking became scorable the rung went from
# unreachable for kickers (an unscored kicker's ppg is 0.0, which is falsy,
# so every one of them fell straight to POSITION_FLOOR) to the normal path
# for everyone outside ESPN's top twenty -- fed by whatever partial season
# they happened to play.
#
# That is not theoretical. On the real 249-row board it put Ben Sauls (THREE
# games, 8/8 FG) at 181.9 projected points and Spencer Shrader (five games)
# at 190.4, ahead of Brandon Aubrey's 171.6 off a full 17-game season. K's
# replacement rank is 3 (scoring/config.STREAMED_REPLACEMENT_RANK), so those
# two undraftable kickers became the baseline every kicker is valued
# against: the best real kicker in the league landed at exactly 0.0 VOR and
# the whole position dropped ~30 places. The guard puts a kicker with too
# small a sample back on POSITION_FLOOR -- which is precisely where every
# unprojected kicker sat before kicking was scorable, so this restores the
# old treatment for the players whose sample cannot support a new one, and
# only for them.
#
# NOT widened to every position, deliberately. The same extrapolation is
# just as questionable for a skill player -- Anthony Richardson reaches this
# rung on TWO games and is projected 18.7 points for a season -- but fixing
# that would move a real skill player's projection, rank and VOR on a board
# the owner is drafting from right now, which a kicker-and-defense change
# has no business doing. It is written up as a follow-up instead.
_MIN_GAMES_FOR_KICKER_PPG = 9


def projection_scale(board: pd.DataFrame, rules: dict | None = None) -> np.ndarray:
    """Per-row factor converting ESPN's PPR projection into league points.

    THE PROBLEM. `projections()`'s first and best rung is `espn_proj`, ESPN's
    own projected season total. It is computed under ESPN's scoring, and this
    pipeline asks for it from `.../leaguedefaults/3?view=kona_player_info`
    (pipeline/sources.py) -- league default 3 is ESPN's full-PPR default, which
    is also why the same payload's rank is read out of `draftRanksByRankType
    .PPR`. So it is a full-PPR number, always, for every league. It cannot be
    re-derived under other rules, because the feed stores only the total:
    `espn_adp` is (espn_id, espn_name, position, team, espn_adp,
    espn_ppr_rank, espn_proj) and carries no projected receptions, yards or
    touchdowns to re-price (verified against data/nfl.duckdb's actual schema).

    WHY IT CANNOT SIMPLY BE LEFT ALONE. The second rung, `stats.ppg * GAMES`,
    IS league-scored now. A board that priced some players (everyone ESPN
    projects -- essentially every player who matters) in PPR and the rest in
    half-PPR would be internally incoherent: `vor` differences `proj_points`
    ACROSS players, so mixing two currencies in one column is worse than the
    single wrong currency it replaced. Either both rungs move or neither does.

    WHAT THIS DOES INSTEAD. Re-price ESPN's total by the ratio the player's
    own most recent season implies:

        scale = (his latest-season points per game under THIS league's rules)
              / (his latest-season points per game under full PPR)

    and `proj_points = espn_proj * scale`. This is EXACT whenever ESPN's
    projected stat mix is a uniform scale-up or scale-down of the mix the
    player actually posted last season -- which is precisely the assumption
    the `stats.ppg * GAMES` rung already makes about the same player, so the
    two rungs now rest on one assumption instead of disagreeing. Worked
    example, a real one: Ja'Marr Chase's 2025 line (125 rec, 1412 rec yds,
    8 TD, 16 games) is 19.6 ppg PPR and 15.7 half-PPR, a scale of 0.798;
    ESPN's 336.4 becomes 268.5. Subtracting the reception delta from an exact
    projected reception count would have given 336.4 - 0.5 x (125 x 336.4/313.6)
    = 269.4 -- 0.3% away, and that alternative needs a number this database
    does not have. It also generalises: the ratio carries a league that scores
    passing touchdowns at 6 or yards at 0.05, which "subtract half a point per
    catch" cannot.

    WHAT IT IS NOT. It is not ESPN's opinion under this league's rules; ESPN
    publishes no such number through this endpoint. It is ESPN's opinion,
    converted using this player's own production mix. That estimate is
    surfaced, not buried: the factor rides onto the board as `proj_scale` for
    every consumer to see, and `build_board` warns once per non-PPR build.

    FALLBACKS, in order:
      * no usable ratio for a player (a rookie or anyone with no latest-season
        weekly rows, or under `_MIN_PPG_FOR_SCALE`) -> the median scale of the
        players at his position who do have one. Without this a rookie WR in a
        standard league would keep a full-PPR projection while every veteran
        WR was marked down 30%, which is a systematic bias in favour of
        exactly the players there is least reason to be confident about.
      * a position with no usable ratio at all -> 1.0. This is K and DST on
        every real board, and it stays true now that kicking is scorable,
        though for a sharper reason: the ratio's DENOMINATOR is full-PPR
        points per game, and full PPR scores no kicking at all, so a kicker's
        `ppr_ppg` is 0 and never clears `_MIN_PPG_FOR_SCALE` however well the
        league pays him. DST has no weekly rows to score either way. A
        conversion out of PPR is meaningless for a position PPR does not
        price, so 1.0 is the right answer for them, not a shrug.
      * `rules` that price exactly full PPR (including None) -> all ones, by
        the shortest possible path, so a PPR league is untouched.
    """
    n = len(board)
    ones = np.ones(n, dtype=float)
    rules = normalize_rules(rules)
    if rules is None or n == 0:
        return ones

    # `league_ppg`/`ppr_ppg` are put on the frame by `_latest_season_stats`,
    # the same way and at the same moment as `stats`, which `projections()`
    # already reads by this same intra-module column contract. A frame without
    # them (a bare fixture, or a board handed back after `_BOARD_COLUMNS` has
    # dropped them) cannot produce any ratio at all -- so say so rather than
    # silently returning ones, which is exactly the kind of quiet PPR fallback
    # this whole change exists to remove.
    if not {"league_ppg", "ppr_ppg"}.issubset(board.columns):
        warnings.warn(
            "projection_scale: this league does not score full PPR, but the "
            "frame carries no league_ppg/ppr_ppg columns, so ESPN's "
            "PPR-shaped season projection cannot be converted and is used "
            "as-is. Any proj_points taken from ESPN is priced in full PPR.",
            RuntimeWarning)
        return ones

    league_ppg = pd.to_numeric(board["league_ppg"], errors="coerce").to_numpy(dtype=float)
    ppr_ppg = pd.to_numeric(board["ppr_ppg"], errors="coerce").to_numpy(dtype=float)

    usable = np.isfinite(league_ppg) & np.isfinite(ppr_ppg) & (ppr_ppg >= _MIN_PPG_FOR_SCALE)
    scale = np.full(n, np.nan)
    scale[usable] = league_ppg[usable] / ppr_ppg[usable]

    # Position medians for everyone else. `np.nanmedian` over an all-NaN slice
    # warns and returns NaN; the emptiness is tested first so a K/DST position
    # (never any usable row) is a clean 1.0 rather than a RuntimeWarning.
    positions = board["position"].to_numpy()
    for pos in pd.unique(positions):
        at_pos = positions == pos
        known = at_pos & usable
        fill = float(np.median(scale[known])) if known.any() else 1.0
        gaps = at_pos & ~usable
        scale[gaps] = fill
    return scale


def projections(conn, board: pd.DataFrame) -> pd.Series:
    """Projected season points per player_id.

    Ladder: ESPN's own season projection re-priced into the league's scoring
    (`proj_scale`, see `projection_scale` for what that conversion is and is
    not), then the player's latest-season points per game -- already in the
    league's scoring, via `_latest_season_stats` -- scaled to a full season,
    then a per-position floor.

    The scale is read off the frame as a column rather than taken as an
    argument, the same intra-module contract `stats` already uses: both are
    merged onto `uni` by `build_board` a few lines before this runs. A frame
    without it (a bare fixture) scales by 1.0, which is what every PPR caller
    wants anyway.

    (The old docstring said "recency-weighted PPG". It never was: the fallback
    reads `stats.ppg`, which `_latest_season_stats` builds from the LATEST
    season alone. `career_summary` in scoring/profile.py is the
    recency-weighted one, and it feeds nothing here.)
    """
    espn = read_table(conn, "espn_adp")
    lookup = {}
    if not espn.empty and "espn_proj" in espn.columns:
        valid = espn.dropna(subset=["espn_proj"])
        valid = valid[valid["espn_proj"] > 0]
        teams = (valid["team"] if "team" in valid.columns
                 else pd.Series([None] * len(valid), index=valid.index))
        # DSTs key on team, not name: ESPN says "Ravens D/ST" and the board
        # says whatever the ADP feed's nickname is, so a name join never hit
        # and every defense fell through to POSITION_FLOOR.
        for (_, row), team in zip(valid.iterrows(), teams):
            key = adp_match_key(row["espn_name"], row["position"], team)
            if key is not None:
                lookup[key] = float(row["espn_proj"])

    # Positional, not `board["proj_scale"]` by label: `_add_adp_only_players`
    # can put two rows on the board with the same synthesized player_id, and
    # every other per-row array in this function is already built by iterating
    # `board.iterrows()` in row order (see build_board's own comment on why
    # `.map()` is unsafe here).
    scale = (pd.to_numeric(board["proj_scale"], errors="coerce")
             .fillna(1.0).to_numpy(dtype=float)
             if "proj_scale" in board.columns else np.ones(len(board)))

    values = []
    for i, (_, row) in enumerate(board.iterrows()):
        key = adp_match_key(row["name"], row["position"], row.get("team"))
        proj = lookup.get(key) if key is not None else None
        if proj is not None:
            proj = proj * scale[i]
            # Same rule the `espn_proj > 0` filter above applies to the raw
            # feed, applied again after conversion: a scale that lands a
            # projection at or below zero (a league that scores a player's
            # whole production mix at nothing) is not a projection, it is a
            # missing one, and it must fall through to the next rung rather
            # than pin a real player at 0 and outrank the position floor.
            if proj <= 0:
                proj = None
        if proj is None:
            stats = row.get("stats")
            ppg = stats.get("ppg") if isinstance(stats, dict) else None
            games = stats.get("games") if isinstance(stats, dict) else None
            if row["position"] == "K" and (games or 0) < _MIN_GAMES_FOR_KICKER_PPG:
                # Too little of a season to extrapolate -- fall to the floor,
                # which is where this kicker sat before kicking was scorable.
                # See _MIN_GAMES_FOR_KICKER_PPG for the two real kickers that
                # made this necessary and why it stops at kickers.
                ppg = None
            proj = float(ppg) * GAMES if ppg else None
        if proj is None or not np.isfinite(proj):
            proj = POSITION_FLOOR.get(row["position"], 80.0)
        values.append(proj)
    return pd.Series(values, index=board["player_id"].to_numpy(), dtype=float)


def _latest_season_stats(weekly: pd.DataFrame,
                         rules: dict | None = None) -> pd.DataFrame:
    """Per-player latest-season stat summary, one nested dict per row.

    Serialized like market_sources: the API's NaN->None pass turns a
    missing merge (rookie/K/DST with no weekly rows) into JSON null.

    `stats.ppg` and `stats.points` are scored under `rules` -- the league's
    `settings.scoring` -- because this is where `projections()` reads its
    fallback rung from, and because the same numbers are what the draft room
    shows as "last year" beside a projection. None is full PPR, unchanged.

    `league_ppg` and `ppr_ppg` ride out alongside: the same latest season in
    the league's scoring and in full PPR. `projection_scale` divides one by
    the other to convert ESPN's PPR-shaped projection. Both are the UNROUNDED
    per-game figures, not `stats.ppg`, which is rounded to one decimal for
    display -- dividing a rounded number by an unrounded one would put a
    ratio of 0.997 on a league that scores exactly full PPR. Both are dropped
    by `_BOARD_COLUMNS` and exist only for that one consumer.
    """
    if weekly.empty:
        return pd.DataFrame(columns=["player_id", "stats", "league_ppg", "ppr_ppg"])
    rules = normalize_rules(rules)
    feats = player_season_features(weekly, rules)
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

    latest["league_ppg"] = latest["ppg"]
    if rules is None:
        # Same rules, same numbers -- taken from the column already computed
        # rather than summed a second time, so the two are equal to the bit
        # and a PPR board's `proj_scale` is exactly 1.0 (see normalize_rules
        # in scoring/ppr.py for why the last bit matters here).
        latest["ppr_ppg"] = latest["ppg"]
    else:
        # One extra pass over the LATEST season only (`wk`, already sliced
        # above for the passing aggregates), not a second
        # player_season_features over the whole recency window: the
        # arithmetic is identical -- sum of weekly points over distinct weeks
        # played -- at a fraction of the work, and only a non-PPR league pays
        # it at all.
        ppr = wk.assign(_ppr=compute_ppr_points(wk)).groupby(
            "player_id", as_index=False)["_ppr"].sum()
        latest = latest.merge(ppr, on="player_id", how="left")
        latest["ppr_ppg"] = latest["_ppr"] / latest["games"]
    return latest[["player_id", "stats", "league_ppg", "ppr_ppg"]]


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

    # K/DST: factors computed from data the league does not score are noise,
    # not signal, so they are pinned neutral -- see `_neutral_factors` for
    # which ones, for which position, and why that now depends on `rules`.
    # `rookie` is meaningless for both regardless (they enter from ADP by
    # design, not because they lack NFL history), so force it False.
    for pos, neutral in (("K", _neutral_factors("K", rules)),
                         ("DST", _neutral_factors("DST", rules))):
        mask = uni["position"] == pos
        if neutral:
            uni.loc[mask, neutral] = 50.0
    uni.loc[uni["position"].isin(["K", "DST"]), "rookie"] = False

    uni = _merge_adp(uni, adp)

    # `stats` is merged before scoring, not after: projections() falls back
    # to this column's latest-season PPG when ESPN has no season projection
    # for a player, and it has to run before the ranking that depends on it.
    # add_market still runs after, because its `edge` column is
    # market_rank - rank and needs the rank this block assigns.
    #
    # `rules`, not full PPR: `stats.ppg`/`stats.points` are the league's own
    # points now, so the fallback rung prices a half-PPR league in half-PPR
    # points. `league_ppg`/`ppr_ppg` come along for `projection_scale` and are
    # dropped by `_BOARD_COLUMNS` at the end.
    uni = uni.merge(_latest_season_stats(weekly, rules), on="player_id", how="left")
    uni["proj_scale"] = projection_scale(uni, rules)

    # -- score --
    uni["composite"] = compute_composite(uni, weights or DEFAULT_WEIGHTS)
    # Projected season points, and VOR as a difference of them. NOT a
    # difference of composites: composite is a within-position percentile
    # (factors.normalize_within_position), so differencing it produces a
    # number with no cross-position meaning -- and this frame is then sorted
    # across positions. See the spec's section 2.
    proj = projections(conn, uni)
    # `.to_numpy()`, not `.map(proj)`: `_add_adp_only_players` synthesizes
    # `player_id` from the normalized name alone (no position), so two
    # ADP-only players who share a normalized name at different positions
    # (e.g. a name collision between an unmatched RB and an unmatched TE)
    # reach `uni` with the SAME id before any dedup runs. `.map()` against a
    # duplicate-valued index raises `InvalidIndexError: Reindexing only
    # valid with uniquely valued Index objects`, taking the whole board down
    # with it. `projections()` builds `proj`'s values by iterating
    # `board.iterrows()` in the same row order as `uni`, so assigning
    # positionally gives every row -- including both id-colliding rows --
    # its own correct value, without ever reindexing on the id at all.
    uni["proj_points"] = proj.to_numpy(dtype=float)
    # `proj_points` now follows this league's scoring on both rungs: the
    # fallback reads `stats.ppg`, scored under `rules` a few lines above, and
    # ESPN's own season projection is re-priced by `proj_scale`. What is left
    # is not a PPR-priced board -- it is one number in the ladder that is an
    # ESTIMATE rather than a measurement, and the warning says exactly that
    # much and no more. It fires on the scoring rules, not on `fmt`: `fmt` is
    # a three-way label derived from the reception value alone, so a league
    # that scores full-point receptions but 6-point passing touchdowns reads
    # as 'ppr' and still has every ESPN projection converted.
    #
    # Why an estimate is unavoidable here: ESPN publishes one number, computed
    # under its own PPR default (pipeline/sources.py fetches
    # `leaguedefaults/3`), and `espn_adp` stores only that total -- no
    # projected receptions, yards or touchdowns to re-price. See
    # `projection_scale` for the conversion, its error on a real player
    # (0.3%), and why leaving the rung alone would have been worse than
    # converting it.
    # `n_espn > 0` as well as non-PPR rules: a league whose rules differ from
    # full PPR only in ways that cannot move an ESPN projection converts
    # nothing, and warning that projections were "CONVERTED, not re-derived"
    # when every scale is exactly 1.0 is a false alarm. That is not
    # hypothetical -- it is the owner's own league the moment its kicking
    # rules are read: 20 rules instead of 14, every skill player's scale
    # still exactly 1.0, and no kicker convertible at all (see
    # `projection_scale`).
    n_espn = int((uni["proj_scale"] != 1.0).sum())
    if normalize_rules(rules) is not None and n_espn:
        warnings.warn(
            f"board: league scoring format is '{fmt}'. proj_points follows "
            "this league's rules, but ESPN's season projection -- the first "
            "rung of the ladder -- is published in full PPR only, so it is "
            f"CONVERTED, not re-derived: {n_espn} of {len(uni)} players carry "
            "a proj_scale != 1.0, the ratio of their latest season under this "
            "league's rules to the same season in full PPR. Exact only if a "
            "player's projected stat mix matches last season's mix.",
            RuntimeWarning)
    uni = apply_vor(uni, settings.replacement_ranks, column="proj_points")
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
