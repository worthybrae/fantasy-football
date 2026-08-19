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
half-PPR. The four rules-independent frames (snap_share, prior_weekly,
season_len, schedules, players) are rebuilt with it rather than split into a
second cache: the whole set is one dataclass, a second format is one extra
40 MB entry, and no league changes its scoring mid-draft. Every table read here (`weekly`, `snap_counts`, `depth_charts`,
`schedules`, `players`) is a UNIVERSAL_TABLES source (pipeline/db.py) that
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
`schedules`/`players` that does not also touch `meta` -- a test or a script
writing directly -- will not invalidate this cache, because `meta` is the
only cheap change signal DuckDB exposes through this driver. Call
`profile_cache.clear()` after any such write.
"""
import threading
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd

from pipeline.db import read_table
from scoring.board import _norm_name
# Imported, not re-derived. The brief for this change was explicit that a
# second, divergent "has the data changed" signal is worse than reusing the
# board's -- and it is: two keys that disagree by one table would show a
# profile assembled half from before a refresh and half from after.
from scoring.board_cache import _db_key, _meta_key
from scoring.ppr import normalize_rules
from scoring.similarity import player_season_features

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

    return ProfileFrames(
        weekly_empty=bool(weekly.empty),
        season_features=feats,
        snap_share=snap_share_by_season(snaps),
        prior_weekly=prior,
        season_len=season_len,
        schedules=read_table(conn, "schedules"),
        players=read_table(conn, "players"),
    )


def cached_profile_frames(conn, rules: dict | None = None) -> ProfileFrames:
    """The per-database, per-scoring-rules frames for one `build_profile` call.

    Costs one full read of `weekly` and `snap_counts` on a miss (~1.0s,
    which is what a click used to cost every single time) and 5.7 ms on a
    hit. See the module docstring for the key, and for the one staleness
    gap it shares with scoring/board_cache.py.

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
    `weekly`/`snap_counts`/`depth_charts`/`schedules`/`players` via
    `write_table` without also updating `meta`, call this so the next
    profile can't be assembled from the old contents."""
    with _lock:
        _cache.clear()
