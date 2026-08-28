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
showing the just-picked player as still available. That answer must always
be live, which used to mean the drafted set was part of the KEY. It is not
any more -- it is read fresh and stamped onto the returned copy instead, for
the reason `_drafted_ids` gives -- and every OTHER dependency `build_board`
(scoring/board.py) reads is enumerated below and folded into the key.
Nothing is invalidated by a timer.

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
    which inserts into `drafted` directly (confirmed by reading it). An
    invalidate-on-write hook in api/main.py alone would therefore MISS every
    live-draft pick, so the table is read fresh on every call here (it is at
    most a few hundred rows). It is NOT in the key -- see `_drafted_ids` for
    why that was the single most expensive line in this file.
  * `sim_survival` / `sim_results` -- also LEAGUE_TABLES, written wholesale
    (write_table does CREATE OR REPLACE) by `run_sim` when a simulation
    finishes, not by pipeline.refresh, so `meta` doesn't cover these either.
    Not called out in the task brief's dependency list -- found by reading
    build_board itself, as instructed. Fingerprinted cheaply via each
    table's own `run_id` (both tables carry one, stamped once per run) plus
    `sim_results.created_at` and each table's row count, rather than
    hashing the full contents.

WHAT THE KEY IS KEYED ON, first component first: the IDENTITY of the
universal data behind the connection, not the connection's file. Every
league file is provisioned from one snapshot of the shared database
(pipeline/leagues), so two league files whose `meta` rows agree hold
byte-identical universal tables and one built board is the board for both
-- which matters because two hundred live rooms are two hundred files, and
keying on the file would build two hundred identical boards and hold two
hundred copies of them. So the identity is the `meta` fingerprint whenever
there is one, and the file path only when `meta` is empty, which is a test
fixture seeded with write_table and never stamped by record_freshness (see
`_identity_key`). The rest of the key is the weights, the settings and the
sim tables, each for the reason given below.

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
import contextlib
import hashlib
import os
import threading
import time
from collections import OrderedDict

import pandas as pd

from pipeline.db import read_table
from scoring import league
from scoring.board import build_board
from scoring.config import DEFAULT_WEIGHTS
from scoring.draft_sim import build_pool

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
# rather than generous. 64 holds the default board plus a handful of
# concurrent slider experiments and every league drafting at once on a
# busy evening -- each live room's league is its own settings key -- without
# ever growing further than that. (It used to have to hold several drafted-state
# generations too, because every pick minted a new key. It does not any
# more: a pick changes nothing in the key at all.)
_MAX_ENTRIES = 64

_lock = threading.Lock()
_cache: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_inflight: "dict[tuple, threading.Event]" = {}

# HOW MANY BOARD BUILDS MAY RUN AT ONCE, ACROSS EVERY KEY.
#
# Single-flight below bounds duplicate work on ONE key; it does nothing about
# eight DIFFERENT keys, which is what a box serving several live drafts is --
# one key per league, each with its own settings. A build peaks near 0.9 GB
# of RSS, so eight of them at once is not a slow request, it is the container
# being killed and every one of those drafts losing its socket (that is what
# happened on 2026-08-25; see get_or_build).
#
# Two, not one: one would serialize a second league's first board behind a
# first league's, and the machine has the headroom for a pair. Waiters queue
# on the semaphore rather than failing -- a build that starts a second late
# is a slow click, where a build that runs out of memory is an outage.
#
# Read through the module global on every acquire (`BUILD_SLOTS` inside
# get_or_build, not a captured local) so a test can swap in a Semaphore(1)
# and observe the ordering.
BUILD_SLOTS = threading.Semaphore(2)

# THE PERMIT IS PER THREAD, NOT PER BUILD, AND THAT IS LOAD-BEARING.
#
# A build can start another build. `scoring/profile_cache.cached_profile`
# goes through `get_or_build` and the `build_profile` inside it calls
# `cached_build_board` and `cached_profile_frames`, each of which goes
# through `get_or_build` again. With one permit taken per ENTRY, two cold
# profile requests for different players -- two rooms under live settings
# right after a refresh, which is exactly when everything is cold -- took
# both permits, then each waited for a third that could never be released.
# Both threads stuck, `BUILD_SLOTS._value` pinned at 0, and the semaphore
# written to protect the container instead killed the process. Reproduced in
# tests/test_board_cache.py.
#
# So a thread that already holds the permit re-enters free. THE MEMORY BOUND
# IS UNCHANGED, which is the only reason this is safe: `build_profile` calls
# the board and the frames one after the other, never concurrently, so a
# thread inside a nested build still has exactly one build's worth of frames
# in flight. Two permits still means at most two builds running at once; it
# now means two THREADS building rather than two ENTRIES into this function,
# which is what the bound was always trying to say.
_permit_depth = threading.local()


@contextlib.contextmanager
def _build_permit():
    """One BUILD_SLOTS permit per thread, however deep the builds nest.

    The semaphore object is captured on the way in and released on the way
    out, so a test that swaps `BUILD_SLOTS` mid-build cannot release a permit
    into a semaphore it never took one from. Depth is only counted after a
    successful acquire, so an interrupted acquire cannot leave a thread
    believing it holds a permit it does not.
    """
    depth = getattr(_permit_depth, "n", 0)
    slots = BUILD_SLOTS if depth == 0 else None
    if slots is not None:
        slots.acquire()
    _permit_depth.n = depth + 1
    try:
        yield
    finally:
        _permit_depth.n = depth
        if slots is not None:
            slots.release()


def get_or_build(cache, inflight, lock, key, build, max_entries):
    """LRU lookup with single-flight: one build per key at a time.

    Shared by this cache and scoring/profile_cache.py, which has the same
    shape and had the same hole. On a miss, the first caller builds; every
    other caller that misses the same key while that build is running WAITS
    for it and takes its result, rather than starting a build of its own.
    Different keys still build concurrently, up to BUILD_SLOTS of them at a
    time -- see that constant for why the bound exists and why it is not 1.

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

    THE SLOT IS HELD AROUND `build()` AND NOTHING ELSE -- not around the wait
    above it. A thread waiting on somebody else's build is holding no memory
    of its own and must not hold a slot either; if it did, two waiters could
    fill both slots and nobody could build the thing they are waiting for.

    And it is held ONCE PER THREAD however deep the builds nest -- a build
    that starts another build (a profile needs a board and a set of frames)
    would otherwise deadlock against itself. See `_build_permit`.
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
        with _build_permit():
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


def _drafted_ids(conn) -> set:
    """Who has been taken, read fresh, every call.

    THIS USED TO BE THE CACHE KEY, and taking it out is the whole point of
    the change. `build_board` touches the drafted set in exactly ONE place
    (`uni["drafted"] = uni["player_id"].isin(drafted_ids)`, the last thing it
    does before the sim merge) -- one boolean column, joined on the id the
    frame is already keyed by. Keying the cache on it meant every pick of a
    live draft threw away a board still correct in all 36 of its other
    columns, and paid 1.6s and ~0.9 GB to compute those again -- a dozen-plus
    times an hour, on the one evening of the year when the tool is under
    load and a stall is most expensive.

    So the board is cached without it and the column is stamped on afterwards
    (see `cached_build_board`). The freshness guarantee the key was there to
    provide is unchanged: this reads the table on EVERY call, hit or miss, so
    a pick made by `POST /api/drafted/{id}` or by the live-draft socket in
    api/live.py is visible to the very next request either way.

    A set rather than a frozenset now that nothing hashes it -- `.isin` takes
    either, and this says plainly that it is no longer a key."""
    drafted = read_table(conn, "drafted", columns=["player_id"])
    if drafted.empty:
        return set()
    return set(drafted["player_id"])


def _meta_key(conn) -> tuple:
    meta = read_table(conn, "meta")
    if meta.empty:
        return ()
    return tuple(sorted(
        (str(r["source"]), str(r["refreshed_at"])) for _, r in meta.iterrows()))


def _identity_key(conn) -> tuple:
    """What the universal data behind `conn` IS, for the cache key.

    Every league file is provisioned from the same snapshot of the shared
    database (pipeline/leagues), so two league files whose `meta` rows
    agree hold byte-identical universal tables, and a board built from one
    is the board for the other: with two hundred rooms in two hundred
    files, keying on the file would build two hundred identical boards. So
    the identity is the `meta` fingerprint whenever there is one. A
    database with no `meta` at all -- a test fixture seeded with
    write_table and never stamped by record_freshness -- falls back to the
    file, which is the collision `_db_key` was written for.
    """
    meta = _meta_key(conn)
    return ("meta", meta) if meta else ("path", _db_key(conn))


def board_fingerprint(board: pd.DataFrame) -> str:
    """Identity of the draftable set, order-independent.

    A session caches pool indices. If the board is rebuilt underneath it,
    those indices point at different players and every recommendation is
    silently about the wrong person. Row order is an artifact of assembly,
    not a change in who is draftable, so it is sorted out.
    """
    ids = sorted(str(p) for p in board["player_id"])
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def _sim_key(conn) -> tuple:
    avail = read_table(conn, "sim_survival")
    results = read_table(conn, "sim_results")
    avail_fp = None if avail.empty else (str(avail.iloc[0].get("run_id")), len(avail))
    results_fp = None if results.empty else (
        str(results.iloc[0].get("run_id")),
        str(results.iloc[0].get("created_at")),
        len(results))
    return (avail_fp, results_fp)


def board_key(conn, weights: dict | None = None,
              settings: "league.LeagueSettings | None" = None) -> tuple:
    """The key `cached_build_board` looks a board up under.

    Lifted out of that function so a second caller can ask what a board
    WOULD be without building one. `api/main.py` uses it for the ETag on
    /api/players: two requests whose keys agree get the identical frame out
    of this cache, so they get identical bytes, so the second one can be
    answered 304 without building anything at all. Keeping the two in one
    function is the point -- an ETag derived from a key that had drifted
    from the cache's own would serve one board's bytes under another's
    name.
    """
    settings = settings or league.load(conn)
    weights = weights or DEFAULT_WEIGHTS
    return (
        _identity_key(conn),
        _weights_key(weights),
        league.to_json(settings),
        _sim_key(conn),
    )


def board_answer_key(conn, weights: dict | None = None,
                     settings: "league.LeagueSettings | None" = None) -> tuple:
    """Everything a served board is a function of, in one tuple.

    `board_key` plus the drafted set. The drafted set is deliberately NOT in
    the cache key -- see `_drafted_ids` for why, it is one boolean column and
    keying on it threw away a good board on every pick -- but it IS in the
    answer, so anything validating a response has to carry it. Read fresh
    here, exactly as `cached_build_board` reads it, so a pick made a moment
    ago cannot be revalidated away.
    """
    return (board_key(conn, weights, settings),
            tuple(sorted(str(pid) for pid in _drafted_ids(conn))))


def cached_build_board(conn, weights: dict | None = None,
                       settings: "league.LeagueSettings | None" = None) -> pd.DataFrame:
    """Drop-in cached replacement for `scoring.board.build_board`.

    Returns a fresh `.copy()` on every call (hit or miss) so a caller that
    mutates its result -- none do today, but nothing stops a future one --
    can never corrupt what the next caller receives. Returns what
    `build_board` returns, column for column and value for value. See the
    module docstring for exactly what's in the key and why, and the "WHAT IS
    DELIBERATELY NOT DEFENDED" section for the one gap.

    `drafted` is the one column NOT taken from the cached frame: it is
    stamped on afterwards from a fresh read of the table, which is what lets
    a whole draft's worth of picks share one cached board. `build_board`
    still sets the column itself, and that stays -- a direct caller must get
    a complete board -- it is simply overwritten here with an answer that
    cannot be stale. See `_drafted_ids`.
    """
    settings = settings or league.load(conn)
    weights = weights or DEFAULT_WEIGHTS
    key = board_key(conn, weights, settings)

    # Miss: build outside the lock, but once per key -- a second request
    # arriving during the build waits for it instead of duplicating it. That
    # "rare race" was not rare: a hover across a column of players is a burst
    # of misses on one key. See get_or_build.
    board = get_or_build(_cache, _inflight, _lock, key,
                         lambda: build_board(conn, weights, settings),
                         _MAX_ENTRIES)
    board = board.copy()
    board["drafted"] = board["player_id"].isin(_drafted_ids(conn))
    return board


# HOW MANY FINISHED PROFILES `warm` PRE-BUILDS, and the environment variable
# that turns it down or off.
#
# The three caches below `warm` are all league-wide: one board, one set of
# frames, one game-points table, and every player shares them. What none of
# them covers is the per-player half of a profile, which is 239 ms a player
# with all three warm (see scoring/profile_cache.cached_profile for the
# section-by-section measurement). So the first person to open ANY card after
# a deploy still paid that, and on a shared vCPU it is multiples of it --
# which is exactly the "right after a deploy" window the edge log showed at
# p50 1.5s.
#
# 40 is the read of what a room actually opens: the board is ~250 players and
# a drafter looks at the top of it, ESPN's order being the one the room is
# sorted by when it loads. 40 profiles is ~10s of background CPU on
# data/nfl.duckdb and ~7 MB resident (a payload is ~165 KB), against a
# `_PAYLOAD_MAX_ENTRIES` of 256 -- so this fills a sixth of that cache and
# evicts nothing.
#
# THIS WARMS THE STORED LEAGUE, AND ONLY THAT. At boot there is no other --
# a live room's settings arrive with a connect, minutes or hours later, and
# are a different cache key. `warm_profiles_for` below is the same pass for
# one of those, which api/live.py fires once a room's first ranking lands.
#
# `WARM_PROFILES=0` disables both, which is the escape hatch if a container
# turns out to be too small to spend the CPU: nothing else changes, the first
# click just pays what it paid before.
WARM_PROFILES_ENV = "WARM_PROFILES"
DEFAULT_WARM_PROFILES = 40


def _warm_profile_count() -> int:
    """`WARM_PROFILES` as a count, defaulting to DEFAULT_WARM_PROFILES.

    Unset, empty and unparseable all mean the default rather than zero: this
    runs on a boot thread nobody is watching, and a typo in a platform
    variable silently turning a performance feature off is the failure that
    takes months to notice. A negative is clamped to zero, which is the one
    thing it can sensibly mean."""
    raw = os.environ.get(WARM_PROFILES_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_WARM_PROFILES
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        print(f"warm: {WARM_PROFILES_ENV}={raw!r} is not a number; "
              f"warming {DEFAULT_WARM_PROFILES} profiles", flush=True)
        return DEFAULT_WARM_PROFILES


def _warm_order(board: pd.DataFrame, n: int) -> list:
    """The first `n` player_ids to pre-build, ESPN's ranking first.

    ESPN's rank rather than the board's own: this is a guess at what somebody
    will click, and the room's list arrives in the market's order, not in the
    order this app has re-ranked it into. A player ESPN does not rank carries
    NaN there and is skipped rather than sorted to one end -- he is by
    definition not near the top of anybody's list. If the column is missing
    (a fixture, an unrefreshed database) or every value is NaN, the board's
    own order is a perfectly good second answer; the point is to warm
    SOMETHING, not to warm the ideal set."""
    order = board
    if "espn_ppr_rank" in board.columns:
        ranked = board[board["espn_ppr_rank"].notna()]
        if not ranked.empty:
            order = ranked.sort_values("espn_ppr_rank")
    return [str(p) for p in order["player_id"].head(n)]


def _warm_profiles(conn, settings, n: int, weights: dict | None = None) -> tuple:
    """Pre-build up to `n` finished profiles. Returns (built, failed).

    ONE AT A TIME, ON THIS THREAD. `get_or_build` holds one of the two
    BUILD_SLOTS around each build, so a sequential loop here occupies at most
    one of them and a request arriving mid-warm can always still build its
    own. Fanning this out across threads would take both, and the thing it
    would be starving is the person who is waiting.

    A player that will not build does not stop the ones after him -- same
    argument as `warm`'s own per-builder try -- but the failures are counted
    rather than printed, because forty log lines about a boot-time
    optimisation nobody asked for is not a report, it is noise.
    """
    from scoring.profile_cache import cached_profile

    built = failed = 0
    for player_id in _warm_order(cached_build_board(conn, weights, settings), n):
        try:
            cached_profile(conn, player_id, weights, settings)
            built += 1
        except Exception:  # noqa: BLE001 -- see the docstring
            failed += 1
    return built, failed


def warm_profiles_for(conn, settings, weights: dict | None = None,
                      n: int | None = None) -> int:
    """The same pass as `warm`'s fourth, under whatever league a caller has.

    WHY THIS IS SEPARATE FROM `warm`. That function warms the STORED league
    -- the `league` table row -- because at boot that is the only league
    there is. A connected draft is priced under the settings ESPN reports
    live (see `_league_settings` in api/main.py), which is a different cache
    key, so every one of the forty payloads `warm` built is a miss for the
    one person in the building who is actually drafting. api/live.py calls
    this with the room's own settings once its first ranking lands, which is
    the moment the board under those settings exists and the reader is about
    to start opening cards.

    `n` defaults to the environment at CALL time rather than at import, so
    `WARM_PROFILES=0` turns this off along with the boot pass and a
    deployment has one switch, not two.

    NEVER RAISES. It runs on a daemon thread beside a live draft; the only
    reasonable answer to "the profiles would not build" is a line in the log
    and a cache that stays cold, which is the state the next request already
    knows how to handle. Returns how many it built, which is what a caller
    or a test would have to assert on anyway.
    """
    n = _warm_profile_count() if n is None else max(0, int(n))
    if not n:
        return 0
    started = time.perf_counter()
    try:
        built, failed = _warm_profiles(conn, settings, n, weights)
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        print(f"warm: room profiles failed, leaving them cold: {exc!r}",
              flush=True)
        return 0
    print(f"warm: {built} room profiles in {time.perf_counter() - started:.1f}s"
          + (f" ({failed} failed)" if failed else ""), flush=True)
    return built


def warm(conn) -> list:
    """Build every cached frame now, on a thread nobody is waiting on.

    NOTHING USED TO DO THIS, and the cost landed on a person. `_meta_key`
    changes the moment a refresh finishes, which retires every cached board
    at once -- so the first reader after a refresh (and the first reader
    after a deploy, when the cache is empty by definition) paid the full cold
    build: 1.6s for the board, another 1.9s for the profile frames, on a
    request they made by clicking something. A refresh already knows it has
    just invalidated everything; it may as well pay for the rebuild itself.

    Failure here is not the caller's problem. This runs behind a refresh loop
    and at boot, where the only reasonable answer to "the board would not
    build" is a line in the log and a cache that stays empty -- exactly the
    state the next request already knows how to handle. So each builder is
    tried on its own and an exception from one does not stop the next.

    The three caches are imported INSIDE the function on purpose:
    scoring/profile_cache.py imports `get_or_build` and the two key helpers
    from this module, so an import at the top of this file would be a cycle.

    Returns the names that warmed, which is what a caller would have to
    assert on anyway -- including "profiles" for the per-player pass at the
    bottom, so that a deployment which has turned it off, or a database that
    could not give it a board to work from, is visible rather than merely
    slow.
    """
    from scoring.game_points import cached_game_points
    from scoring.profile_cache import cached_profile_frames

    try:
        # The whole settings object, not just its scoring rules: the board
        # and the frames need the rules, and the profile pass at the bottom
        # needs the same LEAGUE the board was built under or it would be
        # priming a key nothing will ever ask for.
        settings = league.load(conn)
        rules = settings.scoring
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        print(f"warm: league settings unreadable ({exc!r}); warming full PPR",
              flush=True)
        settings, rules = None, None

    warmed = []
    for name, build in (
            ("board", lambda: cached_build_board(conn)),
            ("profile_frames", lambda: cached_profile_frames(conn, rules)),
            ("game_points", lambda: cached_game_points(conn, rules))):
        try:
            build()
            warmed.append(name)
        except Exception as exc:  # noqa: BLE001 -- see the docstring
            print(f"warm: {name} failed, leaving it cold: {exc!r}", flush=True)

    # LAST, AND ONLY IF THE THREE ABOVE IT ARE IN HAND. Every profile this
    # builds reads the board and the frames; if either is cold, each one
    # would build its own (1.7s and 2.1s), and forty of those on a boot
    # thread is exactly the stampede the rest of this module exists to
    # prevent. `settings` being None means `league.load` raised, in which
    # case there is no stored league to warm under and the board pass has
    # already failed for the same reason.
    #
    # SILENT WHEN IT DOES NOT RUN. A skip is not an incident: it is a
    # correctly cold cache, and the next request handles that.
    n = _warm_profile_count()
    if n and settings is not None and {"board", "profile_frames"} <= set(warmed):
        started = time.perf_counter()
        built, failed = _warm_profiles(conn, settings, n)
        print(f"warm: {built} profiles in {time.perf_counter() - started:.1f}s"
              + (f" ({failed} failed)" if failed else ""), flush=True)
        warmed.append("profiles")
    return warmed


# The simulation pool beside the board: build_pool re-reads `weekly` whole
# (2.2-3.6 s, most of a gigabyte in flight) and depends only on the
# universal data, the board's player set and the league's shape -- so with
# the same identity, fingerprint and settings it is the same pool for every
# league. Smaller than the board's cache: a pool is a few numpy arrays, and
# the settings shapes drafting at once on one evening are few.
_POOL_MAX_ENTRIES = 16
_pool_cache: "OrderedDict[tuple, object]" = OrderedDict()
_pool_inflight: dict = {}


def cached_build_pool(conn, board: pd.DataFrame, settings,
                      weights: dict | None = None):
    """Cached `scoring.draft_sim.build_pool`, keyed on what the pool is
    actually a function of: the universal data (`_identity_key`), the set
    of players on `board` (`board_fingerprint`), the league's settings, the
    weights the board was built with (`build_pool` reads the board's own
    composite columns) and the sim tables (`_sim_key`), the same two the
    board's key carries. `build_session` passes no weights: the defaults.

    Returns the cached SimPool ITSELF, not a copy: its arrays are read by
    `survival` and `rank_available` and written by nothing, and a copy per
    connect would be the memory this cache exists to stop.
    """
    key = (_identity_key(conn), board_fingerprint(board),
           league.to_json(settings), _weights_key(weights), _sim_key(conn))
    return get_or_build(_pool_cache, _pool_inflight, _lock, key,
                        lambda: build_pool(conn, board, settings),
                        _POOL_MAX_ENTRIES)


def clear() -> None:
    """Drop every cached board and pool. Test hook, and the escape valve for the one
    gap this cache accepts (see the module docstring): after a test or a
    script writes `weekly`/`adp`/etc. directly via `write_table` without
    also updating `meta`, call this so the next `cached_build_board` call
    can't serve a board built from the old contents.

    Drops scoring/profile_cache.py's finished payloads too. A profile is
    built out of a board, so a board that has to be thrown away takes every
    payload assembled from one with it -- and that cache's own key carries
    this module's `_identity_key`, so a test that gets past THIS escape
    valve would get past that one by the same route. Imported inside the
    function because profile_cache imports this module."""
    from scoring import profile_cache

    with _lock:
        _cache.clear()
        _pool_cache.clear()
    profile_cache.clear_payloads()
