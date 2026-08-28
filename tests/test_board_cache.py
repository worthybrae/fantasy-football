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

    assert bc.warm(conn) == ["board", "profile_frames", "game_points"]
    assert builds == [1]
    assert len(bc._cache) == 1
    assert len(profile_cache._cache) == 1
    assert len(game_points._cache) == 1

    # ...and the request that follows is served from what it built.
    bc.cached_build_board(conn)
    assert builds == [1]


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
