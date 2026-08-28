"""The background work a deployed instance does to itself.

A local checkout is refreshed by a person typing `make refresh`. A deployed
one has nobody to type it: the volume starts empty, and an empty volume is a
site that renders its own furniture around no data at all -- "no mock draft
is running", an empty archive, blank benefit panels. So the server refreshes
itself, and these tests pin the two properties that make that safe rather
than merely automatic.
"""
import datetime as dt

import pandas as pd

from api import jobs


class _Conn:
    """Just enough connection to answer the freshness question."""

    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return self

    def close(self):
        pass


def _meta(*times):
    return pd.DataFrame({"source": [f"s{i}" for i in range(len(times))],
                         "ok": [True] * len(times),
                         "rows": [1] * len(times),
                         "refreshed_at": list(times)})


NOW = dt.datetime(2026, 8, 25, 12, 0, 0)


def test_an_empty_database_is_always_refreshed(monkeypatch):
    """The case this exists for: a volume with nothing on it. There is no
    freshness to compare, and the answer is not "wait for the schedule"."""
    monkeypatch.setattr(jobs, "_read_meta", lambda conn: pd.DataFrame())
    assert jobs.should_refresh(_Conn([]), max_age_hours=24, now=NOW) is True


def test_fresh_data_is_left_alone(monkeypatch):
    """A redeploy is not a reason to re-pull seventeen sources. `player_news`
    alone is ~223 Google News queries, and pushing a CSS fix should not cost
    them -- nor should it swap every table under whoever is reading the site
    at the time."""
    monkeypatch.setattr(jobs, "_read_meta",
                        lambda conn: _meta(NOW - dt.timedelta(hours=2)))
    assert jobs.should_refresh(_Conn([]), max_age_hours=24, now=NOW) is False


def test_stale_data_is_refreshed(monkeypatch):
    monkeypatch.setattr(jobs, "_read_meta",
                        lambda conn: _meta(NOW - dt.timedelta(hours=30)))
    assert jobs.should_refresh(_Conn([]), max_age_hours=24, now=NOW) is True


def test_the_oldest_source_decides(monkeypatch):
    """Not the newest. One source that refreshed an hour ago does not make a
    database current when another has not moved in a week -- and `refresh`
    carries on past a source that fails, so a partial run is the ordinary
    way to arrive in exactly that state."""
    monkeypatch.setattr(jobs, "_read_meta", lambda conn: _meta(
        NOW - dt.timedelta(hours=1), NOW - dt.timedelta(days=7)))
    assert jobs.should_refresh(_Conn([]), max_age_hours=24, now=NOW) is True


def test_a_null_timestamp_counts_as_never(monkeypatch):
    """`meta` carries a row per source whether or not it succeeded, so a
    source that has never completed has a row and no time in it. Treating
    that as fresh would leave a database permanently half-built."""
    monkeypatch.setattr(jobs, "_read_meta", lambda conn: _meta(None))
    assert jobs.should_refresh(_Conn([]), max_age_hours=24, now=NOW) is True


def test_nothing_starts_unless_it_is_switched_on(monkeypatch):
    """Off by default, so importing the app in a test, a script or a local
    dev server never begins pulling from the network. The deployment opts
    in; nothing else does."""
    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == []


def test_the_switch_starts_it(monkeypatch):
    monkeypatch.setenv(jobs.REFRESH_ENV, "1")
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == ["refresh"]


def test_a_refresh_that_throws_does_not_kill_the_loop(monkeypatch):
    """The loop is the only thing that will ever refresh this deployment. A
    source timing out, ESPN 500ing, a pandas error on a malformed feed --
    none of them may end it, or the site quietly stops updating and the only
    symptom is data that gets older."""
    monkeypatch.setattr(jobs, "_read_meta", lambda conn: pd.DataFrame())
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("nflverse is having a day")

    slept = []
    jobs.refresh_once(_Conn([]), max_age_hours=24, run=boom,
                      on_error=lambda exc: slept.append(exc))
    assert calls == [1]
    assert len(slept) == 1


def test_a_refresh_it_decides_against_does_not_run(monkeypatch):
    monkeypatch.setattr(jobs, "_read_meta",
                        lambda conn: _meta(NOW - dt.timedelta(minutes=5)))
    calls = []
    jobs.refresh_once(_Conn([]), max_age_hours=24, run=lambda: calls.append(1),
                      now=NOW)
    assert calls == []


def test_the_farm_is_its_own_switch(monkeypatch):
    """Separate from the refresh, because they fail for unrelated reasons: a
    stats feed being down is no reason to stop playing mock drafts, and an
    expired ESPN login is no reason to stop pulling ADP."""
    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    monkeypatch.setenv(jobs.FARM_ENV, "1")
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == ["farm-1"]


def test_both_switches_start_both(monkeypatch):
    monkeypatch.setenv(jobs.REFRESH_ENV, "1")
    monkeypatch.setenv(jobs.FARM_ENV, "1")
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == ["refresh", "farm-1"]


def test_a_farm_pass_that_throws_does_not_kill_the_loop():
    """A room torn down between the directory listing it and the join, ESPN
    500ing, a websocket dropped mid-draft -- all ordinary, all reasons to try
    another room rather than to stop farming until somebody redeploys."""
    seen = []

    def boom(batch):
        raise RuntimeError("room 12345 vanished")

    assert jobs.farm_once(run=boom, on_error=seen.append) is False
    assert len(seen) == 1


def test_an_expired_espn_login_backs_off_rather_than_exiting():
    """The one failure waiting cannot fix. It still must not end the thread:
    a thread that has exited looks exactly like a thread that is idle, and
    the corpus would stop growing with nothing in the logs to say why. Long
    backoff instead, so a re-uploaded state file is picked up without a
    redeploy."""
    def no_login(batch):
        raise RuntimeError("No saved ESPN login at data/espn_state.json")

    assert jobs.farm_once(run=no_login, on_error=lambda e: None) is False


def test_a_blank_switch_is_off(monkeypatch):
    """A platform writes the empty string for a variable somebody added and
    left blank, and `bool("")` in the wrong place is how a farm starts on an
    instance that was never meant to run one."""
    monkeypatch.setenv(jobs.REFRESH_ENV, "")
    monkeypatch.setenv(jobs.FARM_ENV, "  ")
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == []


def test_the_farm_gets_its_login_written_before_it_starts(monkeypatch):
    """The variable is read at start-up, not inside the loop: a login that
    only arrived after the loop had given up would sit unused until the next
    redeploy. See api/seed_state.py for why it is a variable at all."""
    import base64
    import gzip
    import json

    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    monkeypatch.setenv(jobs.FARM_ENV, "1")
    written = []
    monkeypatch.setattr("api.seed_state.write_state",
                        lambda *a, **k: written.append(1) or True)
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: None)
    assert written == [1]
    # Keep the encoders referenced so the import list matches the real path.
    assert base64 and gzip and json


def test_no_farm_means_no_login_is_written(monkeypatch):
    """An instance not farming has no business writing an ESPN session to
    its disk, even if the variable is present in the environment."""
    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    monkeypatch.delenv(jobs.FARM_ENV, raising=False)
    written = []
    monkeypatch.setattr("api.seed_state.write_state",
                        lambda *a, **k: written.append(1) or True)
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: None)
    assert written == []


def test_six_farms_means_six_processes(monkeypatch):
    """Parity with a developer's laptop, which runs several `mock_farm`
    processes in several terminals. `farm(n)` plays its n drafts one after
    another, so being in six drafts at once means six processes -- not a
    bigger n. `farm_claims` keeps them out of each other's rooms."""
    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    monkeypatch.setenv(jobs.FARM_ENV, "1")
    monkeypatch.setenv(jobs.FARM_CONCURRENCY_ENV, "6")
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == [f"farm-{i}" for i in range(1, 7)]


def test_concurrency_defaults_to_one(monkeypatch):
    """Switching the farm on must not silently multiply this deployment's
    traffic to ESPN. One unless asked."""
    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    monkeypatch.delenv(jobs.FARM_CONCURRENCY_ENV, raising=False)
    monkeypatch.setenv(jobs.FARM_ENV, "1")
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == ["farm-1"]


def test_zero_still_runs_one(monkeypatch):
    """A zero would read in a settings page as "farming" and behave as "not
    farming" -- the worst combination, because nothing looks wrong."""
    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    monkeypatch.setenv(jobs.FARM_ENV, "1")
    monkeypatch.setenv(jobs.FARM_CONCURRENCY_ENV, "0")
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == ["farm-1"]


def test_an_extra_digit_is_clamped(monkeypatch):
    """60 instead of 6. The cap is about ESPN, not about this machine: one
    address holding sixty seats in a public mock lobby is the shape of thing
    that gets an account looked at."""
    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    monkeypatch.setenv(jobs.FARM_ENV, "1")
    monkeypatch.setenv(jobs.FARM_CONCURRENCY_ENV, "60")
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert len(started) == jobs.MAX_FARM_CONCURRENCY


def test_the_farm_waits_for_data(monkeypatch):
    """MEASURED IN PRODUCTION, on the first deploy. A fresh volume has an
    empty nfl.duckdb; the farm snapshots it before the refresh has written
    anything and reports `board built: 0 pool players, 0 selectable`. The
    snapshot is taken once per process, so it never recovers -- it keeps
    joining real ESPN rooms it cannot pick in and letting ESPN autodraft the
    seat."""
    import threading

    monkeypatch.setenv(jobs.REFRESH_ENV, "1")
    monkeypatch.setenv(jobs.FARM_ENV, "1")
    monkeypatch.setattr("api.seed_state.write_state", lambda *a, **k: True)

    threads = {}
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: threads.setdefault(name, fn))

    farmed = []
    monkeypatch.setattr(jobs, "farm_once",
                        lambda **k: farmed.append(1) or True)
    runner = threading.Thread(target=threads["farm-1"], daemon=True)
    runner.start()
    runner.join(timeout=0.3)

    # Still blocked: no refresh has completed, so there is nothing to farm
    # against and the loop has not called farm_once even once.
    assert farmed == []


def test_no_refresh_configured_means_no_waiting(monkeypatch):
    """An instance whose data arrived some other way -- an upload, a restored
    volume -- has nothing to wait for, and blocking forever on a refresh that
    was never switched on would be a farm that silently never runs."""
    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    monkeypatch.setenv(jobs.FARM_ENV, "1")
    monkeypatch.setattr("api.seed_state.write_state", lambda *a, **k: True)

    threads = {}
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: threads.setdefault(name, fn))

    calls = []

    def once(**k):
        calls.append(1)
        raise SystemExit  # break the endless loop after one pass

    monkeypatch.setattr(jobs, "farm_once", once)
    try:
        threads["farm-1"]()
    except SystemExit:
        pass
    assert calls == [1]


def test_a_failed_refresh_still_releases_the_farm(monkeypatch):
    """Holding the farm back forever because one source 500'd would be a
    worse failure than farming on yesterday's board -- the volume still has
    whatever the last successful refresh wrote."""
    import threading

    ready = threading.Event()
    monkeypatch.setattr(jobs, "should_refresh", lambda *a, **k: True)
    monkeypatch.setattr(jobs, "time", type("T", (), {"sleep": staticmethod(
        lambda s: (_ for _ in ()).throw(SystemExit))})())

    def boom():
        raise RuntimeError("nflverse is having a day")

    try:
        jobs._refresh_loop(_Conn([]), 24, ready, run=boom)
    except SystemExit:
        pass
    assert ready.is_set()


def test_a_database_with_data_warms_its_caches_at_boot(monkeypatch):
    """A cold process has an empty board cache, so the FIRST reader after a
    deploy pays 1.6s for the board and 1.9s more for the profile frames --
    on a click. Default on, unlike the other two switches: it costs one
    build of data the instance already holds and nobody waits for it.

    (`WARM_ON_BOOT` is deleted rather than set, because the default is the
    thing being tested -- tests/conftest.py turns it off for the suite, so
    without this the fixture's answer would be what got asserted.)"""
    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    monkeypatch.delenv(jobs.FARM_ENV, raising=False)
    monkeypatch.delenv(jobs.WARM_ON_BOOT_ENV, raising=False)
    monkeypatch.setattr(jobs, "_read_meta", lambda conn: _meta(NOW))
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == ["warm-caches"]


def test_the_warm_switch_turns_it_off(monkeypatch):
    """What tests/conftest.py uses, and what a deployment would use to stop
    a boot-time build without a redeploy. A switch that is on by default
    still has to be switchable, or the suite has a real board build starting
    behind every app it creates."""
    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    monkeypatch.delenv(jobs.FARM_ENV, raising=False)
    monkeypatch.setenv(jobs.WARM_ON_BOOT_ENV, "0")
    monkeypatch.setattr(jobs, "_read_meta", lambda conn: _meta(NOW))
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == []


def test_the_suite_itself_has_the_warm_switched_off():
    """The fixture in tests/conftest.py is what keeps a background build out
    of every other test in this suite, so it is worth one assertion that it
    is actually in force rather than silently unregistered."""
    assert jobs._on(jobs.WARM_ON_BOOT_ENV, default=True) is False


def test_warming_never_raises_out_of_its_thread():
    """It is a daemon thread nobody is waiting on. `board_cache.warm` catches
    what its three builders throw, but the lines around it can throw too --
    `conn.cursor()` on a connection a test tore down while the thread was
    starting is the one that actually happens -- and an exception escaping
    here is a traceback about work nobody asked for."""
    class Closed:
        def cursor(self):
            raise RuntimeError("Connection Error: connection already closed")

    jobs._warm_caches(Closed())  # no raise


def test_an_empty_volume_is_not_warmed(monkeypatch):
    """Nothing to build from: no `meta` row means no refresh has ever
    finished, so warming would cache an empty board under a key the first
    real refresh immediately retires. That refresh warms it on the way out
    (pipeline/refresh.main)."""
    monkeypatch.delenv(jobs.REFRESH_ENV, raising=False)
    monkeypatch.delenv(jobs.FARM_ENV, raising=False)
    monkeypatch.setattr(jobs, "_read_meta", lambda conn: pd.DataFrame())
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == []


def test_a_connection_that_cannot_answer_is_not_warmed():
    """"Cannot tell" and "nothing there" get the same answer, because they
    lead to the same decision and the cost of being wrong is one cold
    request rather than a failed boot. The fake connections in this file are
    exactly that case, and so is a database mid-creation."""
    assert jobs._has_data(_Conn([])) is False
