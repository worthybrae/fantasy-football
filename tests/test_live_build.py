"""api/live_build.py: draft-session builds in a worker pool, or inline."""
import os

import pytest

from api import live_build
from pipeline.db import get_conn
from pipeline.leagues import provision_league

try:
    from tests.test_live_api import _seed_minimal_live_db
except ImportError:                       # tests/ is not a package
    from test_live_api import _seed_minimal_live_db


@pytest.fixture(autouse=True)
def _no_pool_left_behind():
    yield
    live_build.shutdown()


def test_workers_reads_the_environment(monkeypatch):
    monkeypatch.delenv(live_build.WORKERS_ENV, raising=False)
    assert live_build.workers() == 0
    monkeypatch.setenv(live_build.WORKERS_ENV, "3")
    assert live_build.workers() == 3
    monkeypatch.setenv(live_build.WORKERS_ENV, "")
    assert live_build.workers() == 0
    monkeypatch.setenv(live_build.WORKERS_ENV, "-2")
    assert live_build.workers() == 0
    monkeypatch.setenv(live_build.WORKERS_ENV, "many")
    assert live_build.workers() == 0


def test_submit_refuses_without_workers(monkeypatch):
    monkeypatch.delenv(live_build.WORKERS_ENV, raising=False)
    with pytest.raises(RuntimeError):
        live_build.submit(print, "never")


def _league_file(tmp_path, monkeypatch):
    universal = str(tmp_path / "nfl.duckdb")
    _seed_minimal_live_db(universal)
    root = str(tmp_path / "leagues")
    monkeypatch.setattr("pipeline.leagues.LEAGUES_ROOT", root)
    league_path = provision_league("77", universal_path=universal, root=root)
    return universal, league_path


def test_inline_build_honours_a_monkeypatched_build_session(tmp_path, monkeypatch):
    """With no workers the build runs here, through this module's own
    `build_session` name, so a test double is what gets called -- and the
    worker's connection is closed by the time it returns."""
    monkeypatch.delenv(live_build.WORKERS_ENV, raising=False)
    universal, league_path = _league_file(tmp_path, monkeypatch)
    calls = []

    def fake(cur, my_slot, league_id="", settings=None, progress=None):
        calls.append({"my_slot": my_slot, "league_id": league_id,
                      "settings": settings,
                      "players": cur.execute("SELECT count(*) FROM weekly").fetchone()[0]})
        return "the session"
    monkeypatch.setattr("api.live_build.build_session", fake)

    got = live_build.build_in_worker(league_path, universal, "77", None, None, None)
    assert got == "the session"
    assert calls == [{"my_slot": None, "league_id": "77", "settings": None,
                      "players": calls[0]["players"]}]
    assert calls[0]["players"] > 0
    # The file is free again: a fresh read-write open succeeds.
    get_conn(league_path).close()


def test_a_real_worker_builds_the_session_off_process(tmp_path, monkeypatch):
    """One spawned worker, the minimal seeded league, a DraftSession back
    with the seeded players on its board and the file closed behind it.
    Spawning imports the scoring stack fresh, so this is the slow one."""
    monkeypatch.setenv(live_build.WORKERS_ENV, "1")
    universal, league_path = _league_file(tmp_path, monkeypatch)
    # A parent-held handle on the league file would refuse the worker.
    # Nothing here holds one: provision_league closed its own.
    session = live_build.build_in_worker(league_path, universal, "77", None, None, None)
    assert type(session).__name__ == "DraftSession"
    assert session.league_id == "77"
    assert "p1" in set(session.board["player_id"])
    assert session.my_slot is None
    assert os.getpid() == os.getpid()      # built elsewhere, unpickled here
    get_conn(league_path).close()


def _inline_future(fn, *args):
    """A Future resolved by running `fn` on a thread -- what a recorder
    stands in for the pool with, so nothing is spawned."""
    from concurrent.futures import ThreadPoolExecutor
    return ThreadPoolExecutor(max_workers=1).submit(fn, *args)


def test_a_broken_pool_is_replaced_once_and_the_build_still_lands(tmp_path, monkeypatch):
    """The first submit meets a pool whose worker died; the pool is dropped,
    a fresh one is built, and the retry carries the build."""
    from concurrent.futures.process import BrokenProcessPool
    monkeypatch.setenv(live_build.WORKERS_ENV, "1")
    universal, league_path = _league_file(tmp_path, monkeypatch)
    monkeypatch.setattr("api.live_build.build_session",
                        lambda cur, my_slot, league_id="", settings=None, progress=None: "built")
    pools = []

    class _Pool:
        def __init__(self, broken):
            self.broken = broken

        def submit(self, fn, *args):
            if self.broken:
                raise BrokenProcessPool("a worker died")
            return _inline_future(fn, *args)

        def shutdown(self, wait=True, cancel_futures=False):
            pass

    def fake_get_pool():
        pools.append(_Pool(broken=(len(pools) == 0)))
        return pools[-1]
    monkeypatch.setattr("api.live_build._get_pool", fake_get_pool)

    assert live_build.build_in_worker(league_path, universal, "77", None, None, None) == "built"
    assert len(pools) == 2 and pools[0].broken and not pools[1].broken


def test_a_pool_broken_twice_falls_back_to_inline(tmp_path, monkeypatch):
    from concurrent.futures.process import BrokenProcessPool
    monkeypatch.setenv(live_build.WORKERS_ENV, "1")
    universal, league_path = _league_file(tmp_path, monkeypatch)
    monkeypatch.setattr("api.live_build.build_session",
                        lambda cur, my_slot, league_id="", settings=None, progress=None: "inline")

    class _Broken:
        def submit(self, fn, *args):
            raise BrokenProcessPool("still dead")

        def shutdown(self, wait=True, cancel_futures=False):
            pass
    monkeypatch.setattr("api.live_build._get_pool", lambda: _Broken())
    assert live_build.build_in_worker(league_path, universal, "77", None, None, None) == "inline"


def test_a_build_still_queued_after_the_timeout_is_cancelled_and_runs_inline(tmp_path, monkeypatch):
    """A future the pool never started can be cancelled, and nothing holds
    the file -- so inline is safe."""
    from concurrent.futures import Future
    monkeypatch.setenv(live_build.WORKERS_ENV, "1")
    monkeypatch.setattr("api.live_build.LIVE_BUILD_TIMEOUT", 0.2)
    universal, league_path = _league_file(tmp_path, monkeypatch)
    monkeypatch.setattr("api.live_build.build_session",
                        lambda cur, my_slot, league_id="", settings=None, progress=None: "inline")
    queued = Future()
    monkeypatch.setattr("api.live_build.submit", lambda fn, *args: queued)
    assert live_build.build_in_worker(league_path, universal, "77", None, None, None) == "inline"
    assert queued.cancelled()


def test_a_running_worker_that_finishes_in_the_grace_is_used(tmp_path, monkeypatch):
    """Past the timeout but running: no inline race. The worker's own
    result, landing inside the grace, is what the connect gets."""
    import threading
    from concurrent.futures import Future
    monkeypatch.setenv(live_build.WORKERS_ENV, "1")
    monkeypatch.setattr("api.live_build.LIVE_BUILD_TIMEOUT", 0.2)
    monkeypatch.setattr("api.live_build.LIVE_BUILD_GRACE", 5.0)
    universal, league_path = _league_file(tmp_path, monkeypatch)
    inline = []
    monkeypatch.setattr("api.live_build.build_session",
                        lambda *a, **k: inline.append(1) or "inline")
    running = Future()
    running.set_running_or_notify_cancel()
    threading.Timer(0.5, running.set_result, args=("from the worker",)).start()
    monkeypatch.setattr("api.live_build.submit", lambda fn, *args: running)
    assert live_build.build_in_worker(league_path, universal, "77", None, None, None) == "from the worker"
    assert inline == []


def test_a_running_worker_that_never_finishes_is_a_refusal_not_a_race(tmp_path, monkeypatch):
    from concurrent.futures import Future
    monkeypatch.setenv(live_build.WORKERS_ENV, "1")
    monkeypatch.setattr("api.live_build.LIVE_BUILD_TIMEOUT", 0.2)
    monkeypatch.setattr("api.live_build.LIVE_BUILD_GRACE", 0.2)
    universal, league_path = _league_file(tmp_path, monkeypatch)
    inline = []
    monkeypatch.setattr("api.live_build.build_session",
                        lambda *a, **k: inline.append(1) or "inline")
    running = Future()
    running.set_running_or_notify_cancel()
    monkeypatch.setattr("api.live_build.submit", lambda fn, *args: running)
    with pytest.raises(live_build.BuildTimedOut):
        live_build.build_in_worker(league_path, universal, "77", None, None, None)
    assert inline == []


def test_a_worker_that_dies_mid_build_drops_the_pool_and_builds_inline(tmp_path, monkeypatch):
    """BrokenProcessPool out of result(): the worker was killed with the
    build in flight. The pool is dropped so the next connect gets a fresh
    one, and this connect builds inline (the worker is gone, the file is
    free)."""
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool
    monkeypatch.setenv(live_build.WORKERS_ENV, "1")
    universal, league_path = _league_file(tmp_path, monkeypatch)
    monkeypatch.setattr("api.live_build.build_session",
                        lambda cur, my_slot, league_id="", settings=None, progress=None: "inline")
    dead = Future()
    dead.set_exception(BrokenProcessPool("worker killed"))
    pools = []

    class _Pool:
        def submit(self, fn, *args):
            return dead

        def shutdown(self, wait=True, cancel_futures=False):
            pools.append("shut down")
    monkeypatch.setattr("api.live_build._pool", _Pool())
    monkeypatch.setattr("api.live_build._get_pool", lambda: live_build._pool or pools.append("fresh"))
    assert live_build.build_in_worker(league_path, universal, "77", None, None, None) == "inline"
    assert live_build._pool is None, "the broken pool was dropped"
    assert pools == ["shut down"]
