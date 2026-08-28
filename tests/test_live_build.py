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
