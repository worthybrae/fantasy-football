"""Cache the refresh-static frames `build_profile` derives from.

scoring/board_cache.py already removed the 1.6-1.9s board rebuild from every
profile click. What was left is `build_profile`'s own work, measured on a
copy of data/nfl.duckdb (174,373 weekly rows, 253,106 snap_counts rows,
416,885 depth_charts rows) with the board already warm, meaned over eight
players spanning every branch in scoring/profile.py:

    season_summaries    0.486s      read depth_charts   0.233s
    read weekly         0.247s      find_twins          0.173s
    read snap_counts    0.134s      _outlook            0.095s
    everything else     0.082s      -> ~1.45-1.75s per click

Broken down one level further (same database, same run), the three lines
that actually cost the money are:

  * `snaps["player"].map(_norm_name)` inside `season_summaries` -- 0.343s.
    A per-row Python call over all 253,106 snap rows, on every click, to
    produce a 22,394-row aggregate that depends on nothing per-player.
  * `player_season_features(weekly)` -- 0.211s, computed TWICE per click:
    once in `season_summaries` and again inside `find_twins`, from the same
    frame, discarded both times.
  * the raw reads of `weekly` + `snap_counts` + `depth_charts` -- 0.614s
    for 844,364 rows that get filtered down to one player, one team, and
    two small aggregates.

THE CHOICE MADE HERE, AND WHAT WAS REJECTED: the obvious fix is to cache
the five raw frames (`weekly`, `snap_counts`, `depth_charts`, `schedules`,
`players`) and hand out copies. That was measured and rejected: it saves
only the 0.614s of reads (leaving `season_summaries` and `find_twins`
recomputing the same aggregates at ~0.9s a click) and it costs 665 MB
resident -- `weekly` alone is 240 MB deep, `depth_charts` 275 MB,
`snap_counts` 144 MB. Caching the *derived* artifacts instead is both
faster and ~16x smaller, because the expensive part was never the read:

    season_features   19,521 x 22    6.0 MB   kills 2 x 0.211s
    snap_share        22,394 x 4     3.2 MB   kills 0.343s + 0.134s read
    prior_weekly      18,539 x 145  25.7 MB   the only large weekly slice
                                              anything still needs whole
    players           25,044 x 4     5.2 MB   find_twins' birth dates
    schedules            272 x 46    0.33 MB
                                    -------
                                     40.4 MB per cached database

THE REDESIGNED PLAYER CARD added four more, on exactly the same argument --
every one of them a whole-league computation with no per-player component,
and every one of them ruinous if repeated per click. Measured the same way,
on the same database:

    season_ranks     12,573 x 9     3.6 MB   positional rank by points per
                                             game + the volatility rank, for
                                             every qualified player-season
    comp_pool         6,784 x 9     2.0 MB   comparable seasons with the
                                             following season attached
    pfr_to_gsis       5,438 x 2     0.5 MB   the per-game snap crosswalk
    line_quality         32 x 10   ~0.0 MB   0.76s to build, the single most
                                             expensive line in this module
                                    -------
                                      6.1 MB more, 46.5 MB per database

`line_quality` is the one worth arguing about: it is 0.76s of a cold build
(1.54s before scoring/oline.py was made to read `snap_counts` once instead
of three times) against a warm profile budget of 0.15s. Left where it was
found -- called per request -- it would have made a profile click five times
slower than the click it is decorating.

WHAT THE CARD'S ADDITIONS COST IN TOTAL: a cold build goes from 1.18s to
1.89s, and a warm click from 0.153s to 0.163s (medians of 25 warm calls on
data/nfl.duckdb, board already built). The 10 ms is per-player work that
cannot be cached -- one filtered snap read (0.9 ms), the cohort's band
filter, and the biography lookup.

`weekly` and `depth_charts` are then not cached at all: the per-player rows
`season_summaries`/`game_log` need, and the per-team rows
`team_depth_chart`/`_outlook` need, are read with the filter pushed into
SQL (36 ms and 18 ms, against 247 ms and 233 ms for the whole tables).

WHICH CALLEES GENUINELY NEED A WHOLE FRAME -- established by reading each
one, then checked against the byte-for-byte payload comparison, not assumed:
  * `find_twins` (scoring/similarity.py) compares the target against every
    other player-season, so it needs every row -- but only ever through
    `player_season_features(weekly)`. Reading it line by line: `all_feats`
    is its ONLY use of `weekly`. So it needs the full *features* frame, and
    no weekly rows at all once that frame is handed to it.
  * `season_summaries` needs a league-wide frame for exactly one column:
    `pos_finish`, ranked over `feats` across every player. Its other four
    uses of `weekly` (team_by_season, the passing aggregates, ppg_std, the
    empty check) are all `weekly[weekly.player_id == player_id]`.
  * `factors.schedule_factor` (via `_outlook`) and `weekly_difficulty` both
    need every player's rows -- but only for the latest season, which is
    what both already slice down to on their own.
  * `game_log` needs a league-wide `season -> max(week)` map, to know how
    long each season ran before it can zero-fill the weeks the player
    missed. That map is 10 rows.
  * `team_depth_chart` needs one team's rows; `_outlook`'s depth lookup
    needs one player's rows. Neither looks outside those.
  * the card's four new frames all need every row and no per-player row:
    a positional rank is meaningless without the position, a cohort is
    meaningless without the pool, the snap crosswalk is built from the whole
    `snap_counts`, and a line rating ranks 32 teams against each other.

INVALIDATION: the key is `board_cache._db_key` + `board_cache._meta_key` +
the league's scoring rules. The first two are imported from
scoring/board_cache.py rather than re-implemented, so this cache cannot drift
away from the board's answer to "has a pipeline refresh happened".

The rules component is the newest and the one with teeth. `season_features`
is the ONLY frame here that depends on them -- `player_season_features` now
scores every weekly row under the league's `settings.scoring` -- and it is
what `season_summaries` ranks `pos_finish` on and what `find_twins` matches
and reports `next_ppg` from. Keyed on the database alone, one process serving
a PPR league and a half-PPR league (or one test doing both) would hand the
second league the first one's PPR-priced frame and show it PPR comparables
with a PPR forecast, silently, with every other number on the page correctly
half-PPR. `season_ranks` and `comp_pool` join it: the coefficient of
variation divides one rules-priced number by another, and a cohort's whole
content is "seasons that scored what this one scored". The rules-independent
frames (snap_share, prior_weekly, season_len, schedules, players,
pfr_to_gsis, line_quality) are rebuilt with them rather than split into a
second cache: the whole set is one dataclass, a second format is one extra
46 MB entry, and no league changes its scoring mid-draft. Every table read here (`weekly`, `snap_counts`, `depth_charts`,
`schedules`, `players`, and -- for their column lists only -- `player_news`
and `player_status`) is a UNIVERSAL_TABLES source (pipeline/db.py) that
`pipeline.refresh.main()` rewrites and then stamps into `meta` via
`record_freshness`, unconditionally, success or failure -- verified by
reading pipeline/refresh.py's job list against this exact set of table
names. Nothing here depends on a LEAGUE_TABLES table, so unlike the board
cache this one needs no `drafted` or `sim_*` component: a pick changes the
board row `build_profile` reads out of `cached_build_board`, and changes
nothing in these frames. That is why a pick does not throw away 40 MB of
correctly-cached aggregates.

WHAT IS DELIBERATELY NOT DEFENDED, same gap board_cache accepts and for the
same reason: a `write_table` onto `weekly`/`snap_counts`/`depth_charts`/
`schedules`/`players`/`player_news`/`player_status` that does not also touch
`meta` -- a test or a script writing directly -- will not invalidate this
cache, because `meta` is the only cheap change signal DuckDB exposes through
this driver. Call `profile_cache.clear()` after any such write. The two news
tables make that sharper than it was: CREATING one is the change that
matters (a database with no `player_news` serves an empty feed and caches
the fact), so a test that seeds a profile database, calls `build_profile`,
and only then writes `player_news` will keep serving `news: []` until it
clears.
"""
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.db import read_table
from scoring.board import _norm_name
# Imported, not re-derived. The brief for this change was explicit that a
# second, divergent "has the data changed" signal is worse than reusing the
# board's -- and it is: two keys that disagree by one table would show a
# profile assembled half from before a refresh and half from after.
from scoring.board_cache import _db_key, _meta_key
from scoring.config import CURRENT_SEASON
from scoring.oline import (LINE_QUALITY_COLUMNS, line_quality,
                           reconcile_pfr_to_gsis)
from scoring.ppr import compute_ppr_points, normalize_rules
from scoring.similarity import player_season_features

# How many games a player-season needs before it is ranked against the rest
# of its position, or used as a comparable.
#
# A JUDGEMENT CALL, stated rather than derived, and the one number every
# rank in the profile is sensitive to. Rank a whole position-season with no
# qualifier at all and 2025's running back pool is 151 players, most of whom
# played one or two games -- a 25-ppg one-week wonder then sits above every
# real starter and the rank stops meaning "how good was this back's year".
# Set it too high and the pool stops being the league.
#
# 8 is half a 17-game season: enough games that a per-game average is about
# the player rather than about one afternoon, low enough that a back who
# missed half the year with an injury is still ranked. Measured on
# data/nfl.duckdb, Jahmyr Gibbs' rank is completely insensitive to the exact
# threshold -- RB3 in 2025, RB2 in 2024, RB8 in 2023 at every cutoff from 0
# to 10 -- so what the number actually moves is the DENOMINATOR the rank is
# read against (2025: 151 backs unfiltered, 127 at 4 games, 97 at 8, 89 at
# 10). That is why the denominator is served next to the rank rather than
# left implicit: "RB3" is the same claim at every cutoff, "RB3 of 97" is not.
RANK_MIN_GAMES = 8

# Columns of the frames below, named once so the empty-database shape and
# the populated shape cannot drift apart.
_SEASON_RANK_COLUMNS = ["player_id", "season", "position", "pos_rank_ppg",
                        "pos_rank_ppg_n", "cv", "cv_rank", "cv_rank_n",
                        "cv_pos_median",
                        # Percentiles within (season, position), for the
                        # usage card's colour. See `season_rank_frame`.
                        "snap_share_pctl", "target_share_pctl", "carries_pg_pctl",
                        "targets_pg_pctl", "receptions_pg_pctl",
                        "yards_pg_pctl"]
_COMP_POOL_COLUMNS = ["player_id", "name", "season", "position", "games",
                      "ppg", "next_ppg", "change", "nfl_season"]

# One entry per physical database, at ~40 MB each. Unlike the board cache
# there is no weights component and no drafted component in the key, so the
# only way to hold more than one live entry is to serve several databases
# from one process (the API serves one; the test suite makes a fresh
# tmp_path database per test, which is what this bound is really sized for)
# or to sit across a pipeline refresh. 4 keeps a refresh's worth of history
# from pinning 200 MB while still never thrashing in normal use.
_MAX_ENTRIES = 4

_lock = threading.Lock()
_cache: "OrderedDict[tuple, ProfileFrames]" = OrderedDict()


@dataclass(frozen=True)
class ProfileFrames:
    """Everything `build_profile` needs that is the same for every player.

    `weekly_empty` is carried explicitly rather than inferred from
    `season_features.empty`. The two agree on every real database, but
    `build_profile` branches on "is `weekly` empty" to decide whether to
    look for stat twins at all, and inferring it from a groupby result
    would silently change that branch for a `weekly` whose rows all have a
    null player_id or season -- a shape no test builds today, which is
    exactly why it should not be quietly re-defined here.
    """
    weekly_empty: bool
    season_features: pd.DataFrame
    snap_share: pd.DataFrame | None
    prior_weekly: pd.DataFrame
    season_len: pd.Series
    schedules: pd.DataFrame
    players: pd.DataFrame
    # -- the four frames the redesigned player card added, every one of them
    # a cross-player computation that must not be redone per click. See
    # `season_rank_frame`, `comparable_pool`, `_pfr_crosswalk` and
    # `_line_quality` below for what each costs to build.
    season_ranks: pd.DataFrame
    comp_pool: pd.DataFrame
    pfr_to_gsis: pd.DataFrame
    line_quality: pd.DataFrame
    # Team targets per (season, week, team). Tiny -- about one row per team
    # per week -- and the denominator a per-game target share needs. The
    # season figure the card already carries averages a role away: a receiver
    # who saw a fifth of the targets in September and a tenth in December
    # reads the same as one who was steady all year.
    team_week_targets: pd.DataFrame
    draft_season: int
    # `snap_counts`' column names, carried rather than re-queried: the
    # per-game snap read has to know whether `game_type` exists before it can
    # put it in a WHERE clause, and an information_schema round trip for that
    # measured 1.1 ms -- on a request whose whole warm budget is 150.
    snap_columns: frozenset
    # `player_news` / `player_status` (pipeline/news.py), carried for exactly
    # the same reason and measured the same way: both reads have to know the
    # table is there and carries the columns they name before they can query
    # it, and asking information_schema twice cost MORE THAN THE QUERIES DID
    # -- 0.53 + 0.53 ms of schema lookup against 0.62 + 0.11 ms for the two
    # SELECTs (medians of 300, one session, data/nfl.duckdb + a real news
    # table). Carrying the two column lists here is what takes the whole
    # addition from ~2.6 ms a click to 1.6 ms.
    #
    # The tables are UNIVERSAL_TABLES that `pipeline.refresh.main()` writes
    # and then stamps into `meta` unconditionally (verified against its job
    # list, which runs player_status then player_news last of all), so
    # `_meta_key` invalidates this the moment a refresh creates them --
    # which is the case that matters, because data/nfl.duckdb has no such
    # tables until one runs.
    #
    # NOT cached: the rows. A profile serves ~8 headlines out of 2,346 and
    # one status row out of 249, both filtered in SQL. Caching the news table
    # would add ~1.2 MB to a 46 MB entry and `.copy()` it per click to hand
    # back eight rows.
    news_columns: frozenset
    status_columns: frozenset

    def copy(self) -> "ProfileFrames":
        """A private copy of every frame, for one request to do as it likes with.

        THE BUG THIS EXISTS FOR, which is not hypothetical: the very first
        thing `season_summaries` does with the features frame is
        `feats["pos_finish"] = ...`. Handing out the cached object would
        write a new column into the cache on the first click and keep it
        there for every later one. `find_twins` and the rest happen to copy
        before touching anything (verified by reading them), but relying on
        that means every future edit to any of six functions has to
        remember it. Copying costs 5.7 ms for all of these together --
        1.2 ms for season_features, 0.7 for snap_share, 3.1 for
        prior_weekly, 0.6 for players, 0.1 for schedules -- against the
        0.614s of reads this replaces, so the safe option is also nearly
        free. Copies of object columns duplicate pointers, not the strings
        behind them, so a copy is a few MB rather than another 40.
        """
        return ProfileFrames(
            weekly_empty=self.weekly_empty,
            season_features=self.season_features.copy(),
            snap_share=None if self.snap_share is None else self.snap_share.copy(),
            prior_weekly=self.prior_weekly.copy(),
            season_len=self.season_len.copy(),
            schedules=self.schedules.copy(),
            players=self.players.copy(),
            # +2.4 ms measured on data/nfl.duckdb (1.4 for season_ranks at
            # 12,573 rows, 0.8 for comp_pool at 8,842, 0.1 each for the two
            # small ones), on the same argument the five above are copied
            # for: a consumer that annotates a cached frame in place poisons
            # every later click, and no future edit should have to remember
            # which frames are safe to write to.
            team_week_targets=self.team_week_targets.copy(),
            season_ranks=self.season_ranks.copy(),
            comp_pool=self.comp_pool.copy(),
            pfr_to_gsis=self.pfr_to_gsis.copy(),
            line_quality=self.line_quality.copy(),
            draft_season=self.draft_season,
            snap_columns=self.snap_columns,
            news_columns=self.news_columns,
            status_columns=self.status_columns,
        )


def snap_share_by_season(snaps: pd.DataFrame | None) -> pd.DataFrame | None:
    """Mean offensive snap share per (normalized name, team, season).

    Lifted verbatim out of `season_summaries` so the cached value and the
    uncached path cannot drift; `season_summaries` still calls this when no
    precomputed frame is handed to it. Returns None -- not an empty frame --
    when there is no snap data, because `season_summaries` distinguishes
    "no snap table" (every snap_share is NaN) from "a snap table with no
    row for this player" (also NaN, but reached through the merge), and
    collapsing the two would change which branch runs.

    The `_norm_name` map is the single most expensive line in the whole
    profile: 0.343s for 253,106 rows, a Python-level call per row. It
    depends only on the snap table, so it belongs behind the cache.
    """
    if snaps is None or snaps.empty:
        return None
    s = snaps.copy()
    s["_norm_name"] = s["player"].map(_norm_name)
    return (s.groupby(["_norm_name", "team", "season"], as_index=False)
             ["offense_pct"].mean())


def _snap_share_by_player(weekly: pd.DataFrame, feats: pd.DataFrame,
                          share: pd.DataFrame | None,
                          q: pd.DataFrame) -> pd.Series:
    """Every qualified player-season's snap share, by the per-player rule.

    `season_summaries` resolves one player's team as the LAST `recent_team` of
    each of his seasons, normalises his name, and merges the snap frame on
    (name, team, season). This is that, for everyone, so the number the card
    prints and the pool it is coloured against are the same computation and
    cannot drift apart. Returns NaN where the snap table has no row -- which
    `rank(pct=True)` leaves out of the pool rather than ranking as a zero.
    """
    if share is None:
        return pd.Series(np.nan, index=q.index)
    team = (weekly.groupby(["player_id", "season"])["recent_team"].last()
            .rename("team").reset_index())
    keyed = (feats[["player_id", "season", "name"]].merge(team,
             on=["player_id", "season"], how="left"))
    keyed["_norm_name"] = keyed["name"].map(_norm_name)
    keyed = keyed.merge(share, on=["_norm_name", "team", "season"], how="left")
    lookup = keyed.set_index(["player_id", "season"])["offense_pct"]
    idx = pd.MultiIndex.from_arrays([q["player_id"], q["season"]])
    return pd.Series(lookup.reindex(idx).to_numpy(), index=q.index)


def season_rank_frame(weekly: pd.DataFrame, feats: pd.DataFrame,
                      rules: dict | None,
                      share: pd.DataFrame | None = None) -> pd.DataFrame:
    """Positional rank and volatility rank for every qualified player-season.

    Two ranks per (season, position) pool, both restricted to seasons of at
    least RANK_MIN_GAMES games:

      * `pos_rank_ppg` -- by points per game, best first. Distinct from
        `season_summaries`' existing `pos_finish`, which ranks on SEASON
        TOTAL points and is the "finished RB12" framing. Both belong on the
        card and they answer different questions: a back who missed six
        games can finish RB30 on totals and still have been the third-best
        back per game while he played. `pos_finish` is deliberately left
        exactly as it is, unqualified pool and all.

      * `cv_rank` -- by COEFFICIENT OF VARIATION, sigma over mean, steadiest
        first.

    WHY THE COEFFICIENT AND NOT THE STANDARD DEVIATION, which is the whole
    reason this column exists rather than a rank straight off the `ppg_std`
    the payload already carries: raw sigma is not a measure of consistency,
    it is a second measure of scoring. A 22-ppg back swings in bigger
    absolute points than an 8-ppg back because his good weeks are bigger,
    not because he is less reliable -- measured across the 2023-2025 running
    back pools of data/nfl.duckdb, sigma correlates with points per game at
    r = 0.85 / 0.81 / 0.87. Ranked on raw sigma, Jahmyr Gibbs' 2025 comes
    out 97th of 97, "the most volatile back in the league", which is the
    exact opposite of the truth: on the coefficient the same season is 28th
    of 95, well inside the steady half, against a position median of 0.767.
    DO NOT "SIMPLIFY" THIS BACK TO A SIGMA RANK.

    `cv_pos_median` travels with the rank for the same reason the
    denominators do: 0.627 is not a number anybody has intuitions about, and
    "0.627 against a position median of 0.767" is.

    Sigma is the SAMPLE standard deviation of the weekly points, identical
    to the `ppg_std` `season_summaries` already publishes (same rows, same
    `.std()`, same `rules`) -- so a card showing both cannot show a
    coefficient that fails to divide out.

    A season whose mean is zero or negative gets no coefficient at all
    (NaN, so it ranks nowhere and is not counted in `cv_rank_n`): sigma over
    a mean of nothing is not a volatility, and a NEGATIVE coefficient would
    sort as the steadiest season in the league. That is the whole difference
    between the 97 backs in 2025's ppg pool and the 95 in its volatility
    pool.
    """
    if feats.empty or weekly.empty:
        return pd.DataFrame(columns=_SEASON_RANK_COLUMNS)
    pts = compute_ppr_points(weekly, normalize_rules(rules))
    sd = (pd.DataFrame({"player_id": weekly["player_id"],
                        "season": weekly["season"], "_pts": pts})
          .groupby(["player_id", "season"])["_pts"].std()
          .rename("ppg_sd").reset_index())
    q = feats.loc[feats["games"] >= RANK_MIN_GAMES,
                  ["player_id", "season", "position", "ppg", "games",
                   "target_share", "targets", "carries", "receptions",
                   "rush_yards", "rec_yards"]].merge(
        sd, on=["player_id", "season"], how="left")
    if q.empty:
        return pd.DataFrame(columns=_SEASON_RANK_COLUMNS)

    by_pool = q.groupby(["season", "position"])
    q["pos_rank_ppg"] = by_pool["ppg"].rank(ascending=False, method="min")
    # `size`, not `count`: every row in the pool is a ranked row, and the
    # denominator has to be the pool the reader is being told about.
    q["pos_rank_ppg_n"] = by_pool["ppg"].transform("size")

    q["cv"] = q["ppg_sd"] / q["ppg"].where(q["ppg"] > 0)
    cv_pool = q.groupby(["season", "position"])["cv"]
    q["cv_rank"] = cv_pool.rank(ascending=True, method="min")
    # `count`, not `size`: the seasons with no coefficient are not ranked,
    # so counting them would advertise a denominator nobody occupies.
    q["cv_rank_n"] = cv_pool.transform("count")
    q["cv_pos_median"] = cv_pool.transform("median")

    # -- what the usage card is coloured against ---------------------------
    #
    # A share or a rate is meaningless as a level: 52% of snaps is a workhorse
    # back and a part-time receiver, and 5.9 carries a game is a committee
    # back and a busy slot receiver. The only honest reading is a place among
    # the same position in the same year, which is the pool every other rank
    # on this card already uses.
    #
    # `pct=True` rather than a place out of N: the card paints a five-step
    # ramp, and a percentile is what a ramp cuts. Higher is better for every
    # one of these -- more snaps, more targets, more yards -- so none of them
    # inverts the way `cv_rank` does.
    #
    # Snap share is ranked too, and the join that reaches it is the SAME one
    # `season_summaries` does per player -- same key, same team rule (the last
    # team of the season), same `share` frame, applied to everyone at once
    # rather than to one player. That is the whole reason it is done here: a
    # percentile computed off a second, similar-looking join is a colour that
    # can argue with the number it is painted behind.
    q["_snap_share"] = _snap_share_by_player(weekly, feats, share, q)
    q["_yards_pg"] = (q["rush_yards"] + q["rec_yards"]) / q["games"]
    for col, source in (("snap_share_pctl", q["_snap_share"]),
                        ("target_share_pctl", q["target_share"]),
                        ("carries_pg_pctl", q["carries"] / q["games"]),
                        ("targets_pg_pctl", q["targets"] / q["games"]),
                        ("receptions_pg_pctl", q["receptions"] / q["games"]),
                        ("yards_pg_pctl", q["_yards_pg"])):
        q[col] = (source.groupby([q["season"], q["position"]])
                  .rank(pct=True, ascending=True))
    return q[_SEASON_RANK_COLUMNS]


def _team_week_targets(weekly: pd.DataFrame) -> pd.DataFrame:
    """Team targets per (season, week, team) -- a per-game target share's
    denominator.

    Built here for the same reason every other frame is: it is a whole-league
    aggregate with no per-player component, and recomputing it on each profile
    click would walk 174k weekly rows to divide seventeen numbers.
    """
    cols = {"season", "week", "recent_team", "targets"}
    if weekly.empty or not cols.issubset(weekly.columns):
        return pd.DataFrame(columns=["season", "week", "recent_team", "team_targets"])
    t = weekly[["season", "week", "recent_team", "targets"]].copy()
    t["targets"] = pd.to_numeric(t["targets"], errors="coerce").fillna(0)
    return (t.groupby(["season", "week", "recent_team"], as_index=False)["targets"]
            .sum().rename(columns={"targets": "team_targets"}))


def comparable_pool(feats: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Every player-season that can serve as a comparable, with what happened next.

    A row survives only if BOTH it and the following season clear
    RANK_MIN_GAMES. The following season is the entire point -- a comparable
    with no next year carries no "and then what" -- and the symmetric
    qualifier is what stops a two-game injury year from being read as an
    11-point collapse. Measured on data/nfl.duckdb, that qualifier is the
    difference between Gibbs' 2025 cohort reading "median -4.2, 14 of 19
    declined" and "median -2.8, 13 of 19 declined": Saquon Barkley's 2019,
    ended after two games by a high ankle sprain, is the single row it
    removes.

    `nfl_season` is 1-based (a rookie year is 1), from `players.rookie_season`
    -- the same column scoring/oline.py's `_experience` counts service time
    off. A player the `players` table doesn't know, or a `players` with no
    `rookie_season` at all, gets NaN and is filtered out by any experience
    band rather than silently matching one.

    The filtering itself -- position, points-per-game band, experience band
    -- is per-request and lives in scoring/profile.py; this is only the pool
    it filters, which depends on nothing per-player and costs 0.05s to build
    against 8,842 surviving rows.
    """
    if feats.empty:
        return pd.DataFrame(columns=_COMP_POOL_COLUMNS)
    q = feats.loc[feats["games"] >= RANK_MIN_GAMES,
                  ["player_id", "name", "season", "position", "games", "ppg"]]
    nxt = q[["player_id", "season", "ppg"]].copy()
    # season - 1 so the row joins onto the season BEFORE it, which is the
    # same shift `find_twins` uses to attach `next_ppg`.
    nxt["season"] = nxt["season"] - 1
    pool = q.merge(nxt.rename(columns={"ppg": "next_ppg"}),
                   on=["player_id", "season"], how="inner")
    pool["change"] = pool["next_ppg"] - pool["ppg"]

    rookie = None
    if not players.empty and {"gsis_id", "rookie_season"}.issubset(players.columns):
        rookie = (players.drop_duplicates("gsis_id")
                  .set_index("gsis_id")["rookie_season"])
    pool["nfl_season"] = (pool["season"] - pool["player_id"].map(rookie) + 1
                          if rookie is not None else float("nan"))
    return pool[_COMP_POOL_COLUMNS]


# Every column oline.line_quality reaches for, by table. Checked before it
# is called at all, because unlike everything else in this module it is
# reached on databases it was never written for: it used to be imported by
# nothing but its own test, whose fixtures are all in the real nflverse
# shape, and the profile now calls it on every database the app has --
# including test fixtures whose snap_counts is (player, team, season,
# offense_pct) and whose depth_charts has no `pos_name`. Missing columns
# mean "this database cannot rate a line", which is an empty frame and a
# null on the card, not a KeyError out of a profile request.
_LINE_QUALITY_NEEDS = {
    "snap_counts": {"season", "week", "team", "player", "pfr_player_id",
                    "position", "game_type", "offense_snaps", "offense_pct"},
    "depth_charts": {"dt", "team", "pos_name", "pos_rank", "gsis_id",
                     "player_name"},
    "players": {"gsis_id", "display_name", "rookie_season"},
}


def _table_columns(conn, name: str) -> set[str]:
    """Column names of `name`, or an empty set if the table doesn't exist.

    Moved here from scoring/profile.py (which now imports it) rather than
    copied, so the two callers cannot drift: the same existence check
    `read_table` (pipeline/db.py) makes, one query later. A filtered read
    has to know a column is there before it can put it in a WHERE clause,
    and a table that predates a column must fall back rather than raise.
    """
    return {r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        [name]).fetchall()}


def _line_quality(conn, season: int, snaps: pd.DataFrame) -> pd.DataFrame:
    """`oline.line_quality` for the season being drafted, or an empty frame.

    COST, AND WHY IT IS BEHIND THIS CACHE AT ALL: 1.54s on data/nfl.duckdb.
    It reads `snap_counts` (253,106 rows) four times over and rebuilds the
    pfr->gsis crosswalk twice, because it was written as a standalone
    analysis nothing called per request. It is a whole-league, whole-season
    computation with no per-player component, so it belongs here and is
    built exactly once per (database, refresh, season). Left uncached it
    would have made a profile click ten times slower than the 0.15s it costs
    today -- the single largest thing added by the redesigned card.

    THE SEASON IS `CURRENT_SEASON`, NOT `settings.season`, which is the same
    correction scoring/draft_sim.py already documents at its own `season =
    CURRENT_SEASON`: `league.load` returns the newest IMPORTED season's
    rules (2025 for a league whose last completed draft was 2025), which is
    the right source for teams/starters/scoring and the wrong answer to
    "which season is being drafted". It matters here more than most places,
    because `line_quality` reads today's depth-chart snapshot against
    `season - 1`'s snap history: asked for 2025 it rates the 2026 depth
    charts sitting in this database against 2024's line, and Detroit comes
    out 22nd of 32 at 43.1 with 0.200 returning; asked for 2026 it rates
    them against 2025's, and Detroit is 21st of 32 at 44.4 with 0.600
    returning. The second is the one that describes the line that will
    block this season.

    CURRENT_SEASON is a module constant, so it is deliberately NOT part of
    the cache key -- the same argument scoring/board_cache.py makes for
    REPLACEMENT_RANK: a constant can only change by editing the source and
    restarting, and a restart empties this cache along with it.
    """
    for table, need in _LINE_QUALITY_NEEDS.items():
        if not need.issubset(_table_columns(conn, table)):
            return pd.DataFrame(columns=LINE_QUALITY_COLUMNS)
    return line_quality(conn, season, snaps=snaps)


def _pfr_crosswalk(conn, snaps: pd.DataFrame) -> pd.DataFrame:
    """pfr_player_id -> gsis_id for every player in `snap_counts`.

    `snap_counts` is the only per-GAME snap source and it keys on
    Pro-Football-Reference ids; the profile is keyed on gsis. The existing
    season aggregate (`snap_share_by_season`) sidesteps that by joining on
    normalized name + team + season, which is fine for a mean but is a
    three-column name join to hang seventeen individual games off.

    `oline.reconcile_pfr_to_gsis` is that same name join with the two things
    a bare one lacks -- a `rookie_season` plausibility filter and a hard drop
    of any id that still matches two people -- so it is reused rather than
    re-implemented, with `positions=None` because its default is this
    module's opposite subject: restricted to the five line spots it resolves
    849 of 5,890 ids and no skill players at all.

    0.12s against a `snaps` frame this module has already read (which is why
    it is handed over rather than read again), and 5,438 of 5,890 ids
    resolved -- 93.4% of the skill-position rows in the table. What does not
    resolve gets a null snap column, not a wrong player's.
    """
    return reconcile_pfr_to_gsis(conn, positions=None, snaps=snaps)


def _build(conn, rules: dict | None) -> ProfileFrames:
    weekly = read_table(conn, "weekly")
    snaps = read_table(conn, "snap_counts")

    if weekly.empty:
        # Never consumed: `build_profile` short-circuits both the summaries
        # and the twin search on an empty `weekly`, so nothing reads these.
        # Left as an empty frame rather than `player_season_features(weekly)`
        # because that call raises on a `read_table` miss (no columns at
        # all to compute ppr points from) -- and today's code never makes
        # it either, for the same reason.
        feats = pd.DataFrame()
        prior = weekly
        season_len = pd.Series(dtype="int64")
    else:
        feats = player_season_features(weekly, rules)
        # Sliced exactly the way build_profile and _outlook slice it today,
        # off the same full frame, so the rows AND their original index and
        # order are bit-identical to what the uncached code produced. A
        # `WHERE season = (SELECT max(season) ...)` would have been cheaper
        # to read but would have handed downstream groupby-sums a different
        # row order, and `weekly_difficulty`'s `pct` is a rank -- an exact
        # tie broken differently is a visible payload change.
        prior = weekly[weekly["season"] == weekly["season"].max()]
        season_len = weekly.groupby("season")["week"].max()

    players = read_table(conn, "players")
    # Built once and handed to both the frame the card reads and the frame
    # the card's colours are ranked in: two calls would be two chances for
    # a snap share and its own percentile to come from different numbers.
    share = snap_share_by_season(snaps)
    return ProfileFrames(
        weekly_empty=bool(weekly.empty),
        season_features=feats,
        snap_share=share,
        prior_weekly=prior,
        season_len=season_len,
        schedules=read_table(conn, "schedules"),
        players=players,
        season_ranks=season_rank_frame(weekly, feats, rules, share),
        comp_pool=comparable_pool(feats, players),
        pfr_to_gsis=_pfr_crosswalk(conn, snaps),
        line_quality=_line_quality(conn, CURRENT_SEASON, snaps),
        team_week_targets=_team_week_targets(weekly),
        draft_season=CURRENT_SEASON,
        snap_columns=frozenset(snaps.columns),
        news_columns=frozenset(_table_columns(conn, "player_news")),
        status_columns=frozenset(_table_columns(conn, "player_status")),
    )


def cached_profile_frames(conn, rules: dict | None = None) -> ProfileFrames:
    """The per-database, per-scoring-rules frames for one `build_profile` call.

    Costs one full read of `weekly` and `snap_counts` plus the derived
    aggregates on a miss (1.89s, of which 0.76 is the line rating) and
    6.2 ms on a hit. See the module docstring for the key, and for the one
    staleness gap it shares with scoring/board_cache.py.

    `rules` is the league's `settings.scoring`; None means full PPR.
    `normalize_rules` collapses a rules dict that IS full PPR (which the
    owner's ESPN-imported league is, listed in a different order -- see
    scoring/ppr.py) back onto None, so a PPR league shares one cache entry
    with every rules-less caller and gets the frame it always got, computed
    by the identical call.
    """
    rules = normalize_rules(rules)
    # Sorted items, not the dict: a dict is unhashable, and two dicts with
    # the same rules in a different order must not be two cache entries.
    rules_key = None if rules is None else tuple(sorted(rules.items()))
    key = (_db_key(conn), _meta_key(conn), rules_key)

    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit.copy()

    # Built outside the lock, exactly as board_cache builds boards outside
    # its own: two racing requests on a cold cache both do the work and the
    # later one wins the slot, which wastes a build but never serializes
    # concurrent profile clicks behind one mutex.
    frames = _build(conn, rules)

    with _lock:
        _cache[key] = frames
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)

    return frames.copy()


def clear() -> None:
    """Drop every cached frame set. Test hook, and the escape valve for the
    gap in the module docstring: after a test or a script rewrites
    `weekly`/`snap_counts`/`depth_charts`/`schedules`/`players`/
    `player_news`/`player_status` via `write_table` without also updating
    `meta`, call this so the next profile can't be assembled from the old
    contents."""
    with _lock:
        _cache.clear()
