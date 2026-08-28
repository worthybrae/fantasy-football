"""What the board cache costs on draft night, and what it costs at boot.

Three properties, each of which was a production failure or the direct cause
of one:

  * A PICK IS NOT A REBUILD. The drafted set used to be part of the cache
    key, so every pick of a live draft threw away a board that was correct in
    all but one boolean column and paid 1.6s and ~0.9 GB to compute it again.
  * BUILDS ARE BOUNDED. Single-flight bounds duplicate work on ONE key and
    does nothing about eight different ones, which is what a box serving
    several leagues is. Eight at once killed the container on 2026-08-25.
  * THE CACHES WARM THEMSELVES. `_meta_key` moves the moment a refresh ends,
    so without this the first person to click after a refresh -- or after a
    deploy -- paid the whole cold build.

The endpoint-level behaviour these sit under (a profile reflecting a pick
made after the first call, a connected session not rebuilding per request)
is in tests/test_api.py and stays there.
"""
import threading

import pandas as pd
import pytest

from pipeline.db import get_conn, write_table
from scoring import board_cache as bc
from tests.test_board import _seed


@pytest.fixture(autouse=True)
def _clean_caches():
    """Every cache keys on the database file and each test gets its own, but
    a leftover entry from a previous test is exactly the kind of thing that
    makes a build counter lie."""
    from scoring import game_points, profile_cache
    bc.clear(); profile_cache.clear(); game_points.clear()
    yield
    bc.clear(); profile_cache.clear(); game_points.clear()


def _seed_full(tmp_path):
    """tests/test_board's fixture plus the tables the profile frames read, so
    all three of `warm`'s builders have something to build from."""
    conn = _seed(tmp_path)
    write_table(conn, "snap_counts", pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))
    write_table(conn, "meta", pd.DataFrame([
        {"source": "weekly", "ok": True, "rows": 17,
         "refreshed_at": pd.Timestamp("2026-08-01 12:00:00")}]))
    return conn


def _counting_build(monkeypatch):
    """Wrap `build_board` where board_cache calls it, and count."""
    builds = []
    real = bc.build_board

    def counted(conn, weights=None, settings=None):
        builds.append(1)
        return real(conn, weights, settings)

    monkeypatch.setattr(bc, "build_board", counted)
    return builds


# ---------------------------------------------------------------------------
# A pick is not a rebuild
# ---------------------------------------------------------------------------

def test_a_pick_reuses_the_cached_board(tmp_path, monkeypatch):
    """The whole point of taking `drafted` out of the key. `build_board`
    touches that table in exactly one place -- one boolean column joined on
    the id the frame is already keyed by -- so a pick invalidates one column,
    not thirty-six."""
    conn = _seed_full(tmp_path)
    builds = _counting_build(monkeypatch)

    before = bc.cached_build_board(conn)
    assert not before["drafted"].any()

    write_table(conn, "drafted", pd.DataFrame([{"player_id": "p1",
                                                "pick_no": 1}]))
    after = bc.cached_build_board(conn)

    assert builds == [1], "a pick rebuilt the board"
    assert after.loc[after["player_id"] == "p1", "drafted"].tolist() == [True]
    assert after.loc[after["player_id"] != "p1", "drafted"].tolist() == \
        [False] * (len(after) - 1)


def test_un_drafting_a_player_is_visible_on_the_next_call(tmp_path):
    """The other direction, which a cache that merely ORed picks together
    would get wrong: `DELETE /api/drafted/{id}` is a real endpoint and an
    undo is a real thing to do in a draft room."""
    conn = _seed_full(tmp_path)
    write_table(conn, "drafted", pd.DataFrame([{"player_id": "p1",
                                                "pick_no": 1}]))
    assert bc.cached_build_board(conn)["drafted"].any()

    write_table(conn, "drafted", pd.DataFrame(columns=["player_id", "pick_no"]))
    assert not bc.cached_build_board(conn)["drafted"].any()


def test_the_cached_frame_is_never_mutated_by_the_stamp(tmp_path):
    """The `drafted` column is written onto the caller's COPY. Writing it
    onto the stored frame would mean the next caller inherits whoever asked
    first -- the same bug the `.copy()` has always been there to prevent, in
    the one place that now writes to the frame."""
    conn = _seed_full(tmp_path)
    first = bc.cached_build_board(conn)
    write_table(conn, "drafted", pd.DataFrame([{"player_id": "p1",
                                                "pick_no": 1}]))
    bc.cached_build_board(conn)

    assert not first["drafted"].any(), "an earlier caller's frame was mutated"


# ---------------------------------------------------------------------------
# Bounded concurrency
# ---------------------------------------------------------------------------

class _BlockingBuild:
    """A build that reports when it starts and finishes when told to."""

    def __init__(self):
        self.entered = threading.Semaphore(0)
        self.release = threading.Event()
        self.count = 0
        self._lock = threading.Lock()

    def __call__(self, conn, weights=None, settings=None):
        with self._lock:
            self.count += 1
        self.entered.release()
        assert self.release.wait(5), "build was never released"
        return pd.DataFrame({"player_id": ["p1"], "drafted": [False]})


def test_only_as_many_boards_build_as_there_are_slots(tmp_path, monkeypatch):
    """Different keys still build concurrently -- up to BUILD_SLOTS of them.
    Eight at once is eight ~0.9 GB frames and a container that gets killed,
    which on 2026-08-25 took 35 requests and every open draft socket with
    it. With one slot here, the second build cannot start until the first
    lets go."""
    conn = _seed_full(tmp_path)
    build = _BlockingBuild()
    monkeypatch.setattr(bc, "build_board", build)
    monkeypatch.setattr(bc, "BUILD_SLOTS", threading.Semaphore(1))

    def ask(production):
        # Its own cursor, exactly as a request handler gets one: a DuckDB
        # connection object is not for two threads to call at once, and
        # `_db_key` reads through a cursor to the same database anyway.
        bc.cached_build_board(conn.cursor(),
                              weights={"production": production,
                                       "durability": 1.0, "role": 1.0,
                                       "environment": 1.0, "schedule": 1.0})

    first = threading.Thread(target=ask, args=(1.0,), daemon=True)
    first.start()
    assert build.entered.acquire(timeout=5), "the first build never started"

    second = threading.Thread(target=ask, args=(2.0,), daemon=True)
    second.start()
    assert not build.entered.acquire(timeout=0.3), \
        "a second build started while the only slot was held"

    build.release.set()
    assert build.entered.acquire(timeout=5), \
        "the second build never started after the slot was freed"
    first.join(timeout=5); second.join(timeout=5)
    assert build.count == 2


def test_a_waiter_does_not_hold_a_slot(tmp_path, monkeypatch):
    """The deadlock the acquire has to be narrow to avoid: a caller waiting
    on somebody else's build holds no memory and must hold no slot. If it
    took one, two waiters could fill every slot and nobody could build the
    thing they were waiting for."""
    conn = _seed_full(tmp_path)
    build = _BlockingBuild()
    monkeypatch.setattr(bc, "build_board", build)
    monkeypatch.setattr(bc, "BUILD_SLOTS", threading.Semaphore(1))

    done = []

    def ask():
        bc.cached_build_board(conn.cursor())
        done.append(1)

    first = threading.Thread(target=ask, daemon=True)
    first.start()
    assert build.entered.acquire(timeout=5)
    # Same key, so this one waits on the in-flight build rather than starting
    # its own -- and must not be sitting on the only slot while it does.
    second = threading.Thread(target=ask, daemon=True)
    second.start()

    build.release.set()
    first.join(timeout=5); second.join(timeout=5)
    assert done == [1, 1]
    assert build.count == 1


def test_a_failed_build_gives_its_slot_back(tmp_path, monkeypatch):
    """A build that raises already releases its waiters. It has to release
    its slot too, or a handful of failures leaves a process that can never
    build anything again."""
    conn = _seed_full(tmp_path)
    slots = threading.Semaphore(1)
    monkeypatch.setattr(bc, "BUILD_SLOTS", slots)

    def boom(*a, **k):
        raise RuntimeError("nflverse is having a day")

    monkeypatch.setattr(bc, "build_board", boom)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            bc.cached_build_board(conn)

    assert slots.acquire(blocking=False), "the slot was never given back"
    slots.release()


# A build that starts another build must not deadlock against the bound.
# Bounded at 15s: the fixture's builds are under a second each, so a pass is
# instant and a REGRESSION fails in fifteen seconds rather than hanging a CI
# run. Daemon threads for the same reason -- a regression leaves them stuck
# for good, and non-daemon threads would keep the interpreter from exiting
# after the assertion had already said what was wrong.
_DEADLOCK_TIMEOUT_S = 15


def test_two_concurrent_cold_profile_builds_do_not_deadlock(tmp_path):
    """THE BUG THIS EXISTS FOR, which took the whole process down.

    `cached_profile` builds inside a BUILD_SLOTS permit, and the
    `build_profile` inside it calls `cached_build_board` and
    `cached_profile_frames`, each of which wanted a permit of its own. Two
    cold profile requests for different players -- two rooms under live
    settings right after a refresh, when everything is cold by definition --
    took both permits and then each waited for a third that could never be
    released. Both threads stuck, `BUILD_SLOTS._value` pinned at 0, and
    nothing in the process could ever build anything again.

    Two connections, as two request threads would have; two different
    players, so single-flight cannot quietly turn this into one build and
    one waiter."""
    from tests.test_profile import _seed_two_players

    conn = _seed_two_players(tmp_path)
    bc.clear()
    from scoring import profile_cache
    profile_cache.clear()

    done, errors = [], []
    conns = [conn.cursor(), conn.cursor()]

    def run(i, pid):
        try:
            profile_cache.cached_profile(conns[i], pid)
            done.append(pid)
        except BaseException as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=run, args=(0, "p1"), daemon=True),
               threading.Thread(target=run, args=(1, "p2"), daemon=True)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_DEADLOCK_TIMEOUT_S)

    alive = [t for t in threads if t.is_alive()]
    assert not alive, (
        f"{len(alive)} thread(s) still stuck after {_DEADLOCK_TIMEOUT_S}s; "
        f"finished={done} errors={errors}; "
        f"BUILD_SLOTS._value={bc.BUILD_SLOTS._value}")
    assert not errors, errors
    assert sorted(done) == ["p1", "p2"]


def test_a_nested_build_takes_no_second_permit(tmp_path, monkeypatch):
    """The same property stated directly, and the one a single permit makes
    unmistakable: with BUILD_SLOTS at 1, a build that starts another build
    still finishes. Before the permit was made re-entrant this hung on the
    first nested call."""
    from scoring import profile_cache

    conn = _seed_full(tmp_path)
    bc.clear()
    profile_cache.clear()
    monkeypatch.setattr(bc, "BUILD_SLOTS", threading.Semaphore(1))

    done = []
    t = threading.Thread(
        target=lambda: done.append(profile_cache.cached_profile(conn, "p1")),
        daemon=True)
    t.start()
    t.join(timeout=_DEADLOCK_TIMEOUT_S)

    assert not t.is_alive(), "a nested build waited for a permit it held"
    assert done and done[0]["header"]["player_id"] == "p1"


def test_the_permit_is_given_back_after_a_nested_build(tmp_path, monkeypatch):
    """Re-entrancy must not leak. The outermost frame releases exactly one
    permit however many times the thread re-entered, and a thread that has
    finished holds none."""
    from scoring import profile_cache

    conn = _seed_full(tmp_path)
    bc.clear()
    profile_cache.clear()
    # BOUNDED, which is the whole assertion: it raises the moment a permit is
    # released more times than it was taken, so a re-entrant frame that gave
    # one back on the way out fails the build it was in rather than silently
    # growing the bound this semaphore exists to enforce.
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(bc, "BUILD_SLOTS", slots)

    profile_cache.cached_profile(conn, "p1")

    assert slots.acquire(blocking=False), "the permit was never given back"
    slots.release()
    assert slots._value == 1


def test_a_failed_nested_build_gives_the_permit_back(tmp_path, monkeypatch):
    """Same as `test_a_failed_build_gives_its_slot_back`, one level down: an
    exception out of the innermost build must unwind every frame's
    bookkeeping, not just the one that raised."""
    from scoring import profile_cache

    conn = _seed_full(tmp_path)
    bc.clear()
    profile_cache.clear()
    slots = threading.Semaphore(1)
    monkeypatch.setattr(bc, "BUILD_SLOTS", slots)

    def boom(*a, **k):
        raise RuntimeError("nflverse is having a day")

    monkeypatch.setattr(bc, "build_board", boom)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            profile_cache.cached_profile(conn, "p1")

    assert slots.acquire(blocking=False), "the permit was never given back"
    slots.release()


# ---------------------------------------------------------------------------
# Warming
# ---------------------------------------------------------------------------

def test_warm_builds_all_three_caches(tmp_path, monkeypatch):
    """After it runs, the board, the profile frames and the per-game points
    are all in memory -- so the first person to click after a refresh gets a
    hit rather than the 1.6s build the refresh just made necessary."""
    from scoring import game_points, profile_cache
    conn = _seed_full(tmp_path)
    builds = _counting_build(monkeypatch)

    # The environment decides how many profiles the fourth pass pre-builds;
    # pinned here so this test says the same thing on a box that has turned
    # it off. `_seed_full`'s board has one ESPN-ranked player, so the pass
    # warms one and the board build below is still a hit.
    monkeypatch.delenv(bc.WARM_PROFILES_ENV, raising=False)

    assert bc.warm(conn) == ["board", "profile_frames", "game_points",
                             "profiles"]
    assert builds == [1]
    assert len(bc._cache) == 1
    assert len(profile_cache._cache) == 1
    assert len(game_points._cache) == 1

    # ...and the request that follows is served from what it built.
    bc.cached_build_board(conn)
    assert builds == [1]


def _seed_ranked_board(tmp_path):
    """`_seed_full` with four ESPN-ranked players and one ESPN does not rank.

    The ESPN ranks are deliberately NOT the ADP order (the ADP favourite is
    ESPN's fourth), so a test that asserts on the warmed set is asserting
    that ESPN's ranking was the one used and not merely that something in
    board order came out."""
    conn = _seed_full(tmp_path)
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "Second Guy", "position": "RB", "team": "GB", "adp": 12.0},
        {"adp_name": "Third Guy", "position": "TE", "team": "SF", "adp": 30.0},
        {"adp_name": "Fourth Guy", "position": "WR", "team": "KC", "adp": 45.0},
        {"adp_name": "Rookie Guy", "position": "WR", "team": "GB", "adp": 90.0}]))
    write_table(conn, "espn_adp", pd.DataFrame([
        {"espn_id": 501, "espn_name": "Amon-Ra St Brown", "position": "WR",
         "espn_adp": 1.0, "espn_ppr_rank": 4},
        {"espn_id": 502, "espn_name": "Second Guy", "position": "RB",
         "espn_adp": 2.0, "espn_ppr_rank": 1},
        {"espn_id": 503, "espn_name": "Third Guy", "position": "TE",
         "espn_adp": 3.0, "espn_ppr_rank": 2},
        {"espn_id": 504, "espn_name": "Fourth Guy", "position": "WR",
         "espn_adp": 4.0, "espn_ppr_rank": 3}]))
    return conn


def _warmed_ids(profile_cache):
    """The player_ids in the payload cache. `cached_profile` puts the player
    last in its key tuple."""
    return [key[-1] for key in profile_cache._payload_cache]


def test_warm_pre_builds_the_top_profiles_in_espn_order(tmp_path, monkeypatch):
    """The fourth pass. Board, frames and game points are league-wide and
    every player shares them; what none of them covers is the 239 ms of
    per-player work a card still costs with all three warm, so the first
    person to open ANY card after a deploy paid it. Three asked for, three
    built -- ESPN's top three, not the ADP order and not the player ESPN
    does not rank at all."""
    from scoring import profile_cache
    conn = _seed_ranked_board(tmp_path)
    monkeypatch.setenv(bc.WARM_PROFILES_ENV, "3")

    assert "profiles" in bc.warm(conn)

    assert _warmed_ids(profile_cache) == ["adp_second_guy", "adp_third_guy",
                                          "adp_fourth_guy"]


def test_warm_profiles_serves_the_request_that_follows(tmp_path, monkeypatch):
    """What the pass is FOR: the click after the deploy is a lookup. Counted
    through `build_profile` itself, because a cache that holds the right
    number of entries under the wrong key would pass the test above."""
    from scoring import profile as profile_mod
    from scoring import profile_cache
    conn = _seed_ranked_board(tmp_path)
    monkeypatch.setenv(bc.WARM_PROFILES_ENV, "3")
    bc.warm(conn)

    def boom(*a, **k):
        raise AssertionError("a warmed profile was rebuilt")

    monkeypatch.setattr(profile_mod, "build_profile", boom)
    assert profile_cache.cached_profile(conn, "adp_second_guy") is not None


def test_warm_profiles_can_be_turned_off(tmp_path, monkeypatch):
    """The escape hatch for a container too small to spend the CPU. Nothing
    else about `warm` changes, and the absence is in the returned names
    rather than only in a log nobody reads."""
    from scoring import profile_cache
    conn = _seed_ranked_board(tmp_path)
    monkeypatch.setenv(bc.WARM_PROFILES_ENV, "0")

    assert bc.warm(conn) == ["board", "profile_frames", "game_points"]
    assert not profile_cache._payload_cache


def test_warm_skips_the_profiles_when_the_board_would_not_build(tmp_path, monkeypatch):
    """Every profile this pass builds reads the board and the frames. With
    either cold, each of the forty would build its own -- 1.7s and 2.1s --
    which is the stampede the rest of this module exists to prevent. A skip
    is not an incident: it is a correctly cold cache."""
    from scoring import profile_cache
    conn = _seed_ranked_board(tmp_path)
    monkeypatch.setenv(bc.WARM_PROFILES_ENV, "3")

    def boom(*a, **k):
        raise RuntimeError("no")

    monkeypatch.setattr(bc, "build_board", boom)
    warmed = bc.warm(conn)

    assert "profiles" not in warmed
    assert not profile_cache._payload_cache


def test_warm_profiles_falls_back_to_board_order_without_espn_ranks(tmp_path, monkeypatch):
    """An unrefreshed database, or a fixture, has no ESPN ranking to sort by.
    The point is to warm SOMETHING rather than the ideal set, so the board's
    own order is a perfectly good second answer -- warming nothing at all
    would be the wrong reading of a missing column."""
    from scoring import profile_cache
    conn = _seed_ranked_board(tmp_path)
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp", "espn_ppr_rank"]))
    monkeypatch.setenv(bc.WARM_PROFILES_ENV, "2")

    assert "profiles" in bc.warm(conn)
    assert len(profile_cache._payload_cache) == 2


def test_warm_profile_count_reads_the_environment(monkeypatch):
    """Unset, empty and unparseable all mean the default, never zero: this
    runs on a boot thread nobody is watching, and a typo in a platform
    variable silently turning a performance feature off is the failure that
    takes months to notice."""
    monkeypatch.delenv(bc.WARM_PROFILES_ENV, raising=False)
    assert bc._warm_profile_count() == bc.DEFAULT_WARM_PROFILES
    monkeypatch.setenv(bc.WARM_PROFILES_ENV, "  ")
    assert bc._warm_profile_count() == bc.DEFAULT_WARM_PROFILES
    monkeypatch.setenv(bc.WARM_PROFILES_ENV, "not-a-number")
    assert bc._warm_profile_count() == bc.DEFAULT_WARM_PROFILES
    monkeypatch.setenv(bc.WARM_PROFILES_ENV, "12")
    assert bc._warm_profile_count() == 12
    monkeypatch.setenv(bc.WARM_PROFILES_ENV, "0")
    assert bc._warm_profile_count() == 0
    # A negative can only sensibly mean off.
    monkeypatch.setenv(bc.WARM_PROFILES_ENV, "-5")
    assert bc._warm_profile_count() == 0


def test_warm_carries_on_past_a_builder_that_raises(tmp_path, monkeypatch):
    """It runs behind a refresh loop and at boot, where the only sensible
    answer to "the board would not build" is a log line and a cache that
    stays cold -- which is the state the next request already handles. It is
    not a reason to skip the other two, and never a reason to take down the
    thread it is on."""
    from scoring import game_points
    conn = _seed_full(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("no")

    monkeypatch.setattr(bc, "build_board", boom)
    assert bc.warm(conn) == ["profile_frames", "game_points"]
    assert len(bc._cache) == 0
    assert len(game_points._cache) == 1


def test_warm_on_an_empty_database_says_so_rather_than_raising(tmp_path):
    """A volume that has never been refreshed. Whatever each builder makes of
    a database with no tables in it, `warm` must come back."""
    conn = get_conn(str(tmp_path / "bare.duckdb"))
    assert isinstance(bc.warm(conn), list)
