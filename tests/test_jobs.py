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
    assert started == ["farm"]


def test_both_switches_start_both(monkeypatch):
    monkeypatch.setenv(jobs.REFRESH_ENV, "1")
    monkeypatch.setenv(jobs.FARM_ENV, "1")
    started = []
    jobs.start_jobs(_Conn([]), spawn=lambda name, fn: started.append(name))
    assert started == ["refresh", "farm"]


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
