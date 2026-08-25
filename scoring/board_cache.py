"""Cache the built board so a click during a live draft doesn't re-run it.

Measured on data/nfl.duckdb (249 players): `build_board` alone costs ~1.6-1.9s
and `build_profile` -- which calls it just to read one row back out --
~3.1s. `GET /api/players` costs the same 1.6-1.9s per request. Every one of
those calls recomputes every factor, the composite, VOR, tiers, and a
five-source market consensus from scratch. `cached_build_board` below is the
fix: build once, serve the same frame to whichever caller asks next, as long
as nothing the board depends on has actually changed.

THE STALENESS FAILURE THIS KEY GUARDS AGAINST: the owner picks a player
during a live draft, someone else's board client posts to `/api/drafted`
(or the live-draft socket in api/live.py writes the `drafted` table directly
-- see below), and the next click on ANY other player must not come back
showing the just-picked player as still available. A cache keyed on
anything less than the live `drafted` table would show exactly that lie on
draft night, which is worse than the 1.6-1.9s it replaces. So every
dependency `build_board` (scoring/board.py) actually reads is enumerated
below and folded into the key; nothing is invalidated by a timer.

WHAT build_board READS (verified by reading scoring/board.py itself, not
inferred from the task brief, which warned its own list might be
incomplete):
  * the `weights` argument (five floats; the profile/players sliders)
  * `settings` -- either passed in, or `league.load(conn)` reads the
    `league` table when it isn't. `settings.replacement_ranks` (used by
    `apply_vor`) is a property of `settings.teams/starters/flex_slots` (from
    that table) AND of `REPLACEMENT_RANK`/`STREAMED_REPLACEMENT_RANK`/
    `FLEX_SHARES`, which are Python constants in scoring/config.py, loaded
    once at import time. A constant can only change between two calls to
    this cache by editing the source and restarting the process -- and a
    restart empties `_cache` (it's a plain module global) along with it, so
    no key component is needed for that half of replacement_ranks: the
    cache can never observe two different values of a constant within one
    process lifetime. The `league` table half is covered by `settings_key`.
  * `weekly`, `depth_charts`, `schedules`, `adp`, `espn_adp`, `fp_ecr`,
    `sleeper_ids`, `mfl_adp`, `cbs_ranks` -- every one of these is a
    UNIVERSAL_TABLES (pipeline/db.py) source that `pipeline.refresh.main()`
    (pipeline/refresh.py) rewrites and then stamps into `meta` via
    `record_freshness`, unconditionally, success or failure. Reading the
    (tiny, ~11-row) `meta` table and fingerprinting its
    (source, refreshed_at) pairs is a cheap, complete proxy for "did a
    pipeline refresh touch anything build_board reads from the universal
    tables" -- confirmed by reading pipeline/refresh.py's job list against
    this exact set of table names, not assumed.
  * `drafted` -- a LEAGUE_TABLES table, so `meta` never covers it. Changes
    on every pick, from two independent writers: `POST /api/drafted/{id}`
    in this file, AND the live-draft websocket handler in api/live.py,
    which inserts into `drafted` directly (confirmed by reading it) and
    which this task is not allowed to touch. An invalidate-on-write hook in
    api/main.py alone would therefore MISS every live-draft pick. Read the
    table fresh on every call instead (it's at most a few hundred rows,
    cheap next to a 1.6s rebuild) and key on the exact set of drafted
    player_ids -- not a row count, which can't tell "A swapped for B" from
    "no change", and not pick_no, which build_board never uses (only
    `.isin(drafted_ids)` does).
  * `sim_survival` / `sim_results` -- also LEAGUE_TABLES, written wholesale
    (write_table does CREATE OR REPLACE) by `run_sim` when a simulation
    finishes, not by pipeline.refresh, so `meta` doesn't cover these either.
    Not called out in the task brief's dependency list -- found by reading
    build_board itself, as instructed. Fingerprinted cheaply via each
    table's own `run_id` (both tables carry one, stamped once per run) plus
    `sim_results.created_at` and each table's row count, rather than
    hashing the full contents.

WHAT IS DELIBERATELY NOT DEFENDED: a table swapped out from under a live
`conn` by something other than pipeline.refresh, run_sim, or the two
`drafted` writers above (e.g. a test or a script calling `write_table`
directly on `weekly`/`adp`/etc. without also touching `meta`) will not
invalidate this cache, because nothing updates `meta` for it and there is no
cheap per-table change signal available (DuckDB tables expose no version
counter or updated-at through this driver). This is the one gap in "read
every table fresh" and it's accepted rather than papered over: closing it
completely would mean fingerprinting the full contents of `weekly` (millions
of cells) on every request, which defeats the point of caching at all. Call
`board_cache.clear()` after any such direct write in a test or a script.
"""
import threading
from collections import OrderedDict

import pandas as pd

from pipeline.db import read_table
from scoring import league
from scoring.board import build_board
from scoring.config import DEFAULT_WEIGHTS

# Fixed, alphabetical order so two dicts with the same values in a different
# key order (unlikely from the API, which always builds all five, but not
# guaranteed from a future direct caller) hash to the same cache key.
_WEIGHT_KEYS = ("durability", "environment", "production", "role", "schedule")

# The owner drags five continuous slider values in the UI; a cache keyed
# partly on those floats (see _weights_key) grows one entry per distinct
# drag position if left unbounded -- a slow memory leak that never shows up
# in a quick smoke test but does over a multi-hour draft-night session with
# the tab left open. Each cached frame is 249 rows x 27 columns, several of
# them (stats, market_sources) nested dicts -- on the order of a few hundred
# KB to low single-digit MB per entry, not free, so the bound is deliberate
# rather than generous. 16 comfortably holds the default board plus a
# handful of concurrent slider experiments and a couple of drafted-state
# generations (every pick changes `_drafted_key`, so a pick's-worth of
# now-dead entries ages out on its own as the draft proceeds) without ever
# growing further than that.
_MAX_ENTRIES = 16

_lock = threading.Lock()
_cache: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_inflight: "dict[tuple, threading.Event]" = {}


def get_or_build(cache, inflight, lock, key, build, max_entries):
    """LRU lookup with single-flight: one build per key at a time.

    Shared by this cache and scoring/profile_cache.py, which has the same
    shape and had the same hole. On a miss, the first caller builds; every
    other caller that misses the same key while that build is running WAITS
    for it and takes its result, rather than starting a build of its own.
    Different keys still build concurrently -- nothing here serializes
    /api/players load across weight profiles (see test_concurrent_requests).

    WHY THIS EXISTS. A hover tip over eight players in a freshly-connected
    draft is eight `build_profile` calls in the same second, every one a miss
    on the same key. Each miss built the whole board and the whole set of
    profile frames itself -- measured at 5.4s and 1.67 GB peak RSS for one
    build on data/nfl.duckdb -- so eight hovers were eight of those at once.
    On 2026-08-25 at 09:50:31Z that killed the production container: the
    Railway edge log shows 35 profile requests answered 502 ("connection
    closed unexpectedly" after 6-28s in flight, then "connection refused"
    until the restart 16s later), and every one of them reached the browser
    as "Could not load this player." With single-flight the same eight
    hovers cost one build and seven waits.

    A build that raises releases its waiters, and the next of them to wake
    becomes the builder (the loop re-checks the cache and the in-flight
    table), so one failed build never strands the rest. Returns the stored
    object; callers copy it, as they did before.
    """
    while True:
        with lock:
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                return hit
            pending = inflight.get(key)
            if pending is None:
                pending = inflight[key] = threading.Event()
                break
        pending.wait()
    try:
        built = build()
    except BaseException:
        with lock:
            inflight.pop(key, None)
        pending.set()
        raise
    with lock:
        cache[key] = built
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)
        inflight.pop(key, None)
    pending.set()
    return built


def _weights_key(weights: dict | None) -> tuple:
    w = weights or DEFAULT_WEIGHTS
    return tuple(float(w.get(k, DEFAULT_WEIGHTS[k])) for k in _WEIGHT_KEYS)


def _db_key(conn) -> tuple:
    """Identify which physical database `conn` points at.

    `conn.cursor()` -- what every api/main.py handler actually passes in --
    returns a NEW DuckDBPyConnection object on every call, sharing the
    underlying database but never `is`/`id()`-stable across requests, and
    DuckDBPyConnection has no writable __dict__ (verified: assigning any
    attribute to one raises AttributeError), so neither identity nor a
    stashed-on-the-object cache is available. `PRAGMA database_list`
    returns the engine's own (oid, name, file_path) for every attached
    database and is stable across cursors sharing one connection (verified
    against a real connection and its .cursor()) -- this is what actually
    tells two different `conn` arguments apart. Without this, two different
    test databases (or a real server conn vs. a scratch copy opened for
    measurement) that happen to produce identical weights/settings/drafted/
    meta/sim fingerprints -- e.g. two freshly-seeded, nothing-drafted-yet
    fixtures -- would collide on the same key and one test's board would be
    served to the other's request.
    """
    return tuple(conn.execute("PRAGMA database_list").fetchall())


def _drafted_key(conn) -> frozenset:
    drafted = read_table(conn, "drafted")
    if drafted.empty:
        return frozenset()
    return frozenset(drafted["player_id"])


def _meta_key(conn) -> tuple:
    meta = read_table(conn, "meta")
    if meta.empty:
        return ()
    return tuple(sorted(
        (str(r["source"]), str(r["refreshed_at"])) for _, r in meta.iterrows()))


def _sim_key(conn) -> tuple:
    avail = read_table(conn, "sim_survival")
    results = read_table(conn, "sim_results")
    avail_fp = None if avail.empty else (str(avail.iloc[0].get("run_id")), len(avail))
    results_fp = None if results.empty else (
        str(results.iloc[0].get("run_id")),
        str(results.iloc[0].get("created_at")),
        len(results))
    return (avail_fp, results_fp)


def cached_build_board(conn, weights: dict | None = None,
                       settings: "league.LeagueSettings | None" = None) -> pd.DataFrame:
    """Drop-in cached replacement for `scoring.board.build_board`.

    Returns a fresh `.copy()` on every call (hit or miss) so a caller that
    mutates its result -- none do today, but nothing stops a future one --
    can never corrupt what the next caller receives. Does not change what
    `build_board` returns and does not modify scoring/board.py; this only
    decides when NOT to call it again. See the module docstring for exactly
    what's in the key and why, and the "WHAT IS DELIBERATELY NOT DEFENDED"
    section for the one gap.
    """
    settings = settings or league.load(conn)
    weights = weights or DEFAULT_WEIGHTS
    key = (
        _db_key(conn),
        _weights_key(weights),
        league.to_json(settings),
        _drafted_key(conn),
        _meta_key(conn),
        _sim_key(conn),
    )

    # Miss: build outside the lock, but once per key -- a second request
    # arriving during the build waits for it instead of duplicating it. That
    # "rare race" was not rare: a hover across a column of players is a burst
    # of misses on one key. See get_or_build.
    board = get_or_build(_cache, _inflight, _lock, key,
                         lambda: build_board(conn, weights, settings),
                         _MAX_ENTRIES)
    return board.copy()


def clear() -> None:
    """Drop every cached board. Test hook, and the escape valve for the one
    gap this cache accepts (see the module docstring): after a test or a
    script writes `weekly`/`adp`/etc. directly via `write_table` without
    also updating `meta`, call this so the next `cached_build_board` call
    can't serve a board built from the old contents."""
    with _lock:
        _cache.clear()
