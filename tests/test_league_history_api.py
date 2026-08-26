"""The league-history routes: who may ask, what starts an import, what the
page reads while it runs and after."""
import json
import tempfile
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import drafts as drafts_api
from api import league_history as lh
from pipeline import espn_drafts as drafts
from scoring.config import CURRENT_SEASON
from tests.test_league_activity import (DRAFT_PAYLOAD, MATCHUP_PAYLOAD, ROSTER_PAYLOAD,
                                        TEAM_PAYLOAD, TXN_PAYLOAD, SWID_A)

LEAGUE = "53929318"


def _entries_payload(*leagues):
    """What espn_drafts.league_entries reads: the account's team list.

    seasonId matches CURRENT_SEASON, not a fixed year -- league_entries
    filters entries to the season owned_league asks for
    (str(CURRENT_SEASON)), and a mismatched year here would filter every
    entry out before the league_id check ever runs.
    """
    return json.dumps({"preferences": [
        {"typeId": 9, "metaData": {"entry": {
            "gameId": 1, "seasonId": CURRENT_SEASON, "entryId": 4,
            "groups": [{"groupId": int(lid), "groupName": name}],
            "entryMetadata": {"teamName": "Mine"}}}}
        for lid, name in leagues]})


def _fetch(entries, calls=None):
    """One fake for both the account list and the season/week reads."""
    served = {
        "view=mTeam": dict(TEAM_PAYLOAD, status=dict(TEAM_PAYLOAD["status"], finalScoringPeriod=1,
                                                     latestScoringPeriod=1, previousSeasons=[])),
        "view=mMatchupScore": MATCHUP_PAYLOAD, "view=mDraftDetail": DRAFT_PAYLOAD,
        "view=mTransactions2": TXN_PAYLOAD, "view=mRoster": ROSTER_PAYLOAD,
        "view=mSettings": {"settings": {"rosterSettings": {"lineupSlotCounts": {"0": 1, "20": 5}},
                                        "acquisitionSettings": {}}},
    }

    def fetch(url, cookies, headers=None):
        if calls is not None:
            calls.append(url)
        # fan_url puts showFantasyEntries=true in the query -- the real
        # marker of an account-list read, unlike a guess at the host name.
        if "showFantasyEntries" in url:
            return 200, entries
        for key, payload in served.items():
            if key in url:
                return 200, json.dumps(payload)
        return 404, "not found"
    return fetch


@pytest.fixture
def owner(monkeypatch, tmp_path):
    """This machine's saved login, owning league 53929318 as member A."""
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: (SWID_A, "s2"))
    monkeypatch.setattr(drafts_api, "_CACHE", {})
    monkeypatch.setattr("pipeline.leagues.LEAGUES_ROOT", str(tmp_path / "leagues"))
    # LEAGUE ("53929318") is also this deployment's configured
    # DEFAULT_LEAGUE_ID -- league_db_path/provision_league special-case that
    # exact id to pipeline.db.DEFAULT_PATH (the real data/nfl.duckdb)
    # regardless of `root`, which would route these tests at the real
    # database out from under the LEAGUES_ROOT patch above. Move the
    # sentinel out of the way so every league here resolves under the
    # isolated root like any other league would.
    monkeypatch.setattr("pipeline.leagues.DEFAULT_LEAGUE_ID", "0")
    lh._JOBS.clear()
    lh._ANSWERS.clear()
    return tmp_path


def _client(fetch, runner=None):
    app = FastAPI()
    # A fresh universal database per client, under the system temp dir --
    # never data/nfl.duckdb, which another session may hold the lock on.
    universal_path = str(Path(tempfile.mkdtemp()) / "universal.duckdb")
    lh.register_league_history_routes(app, fetch=fetch, runner=runner,
                                      universal_path=universal_path)
    return TestClient(app, client=("127.0.0.1", 50000))


def _sync(fn):
    """A runner that runs the job on the request thread."""
    fn()


def test_a_visitor_with_no_session_is_refused(monkeypatch, owner):
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    resp = _client(_fetch(_entries_payload((LEAGUE, "Mine")))).get(f"/api/leagues/{LEAGUE}/history")
    assert resp.status_code == 403


def test_a_league_not_on_the_account_is_refused(owner):
    resp = _client(_fetch(_entries_payload(("999", "Other")))).get(f"/api/leagues/{LEAGUE}/history")
    assert resp.status_code == 403


def test_a_mock_league_is_refused(owner):
    resp = _client(_fetch(_entries_payload(("777", "Pro 8-Team Mock")))).post("/api/leagues/777/history")
    assert resp.status_code == 400


def test_first_visit_is_404_then_the_import_fills_it(owner):
    calls = []
    client = _client(_fetch(_entries_payload((LEAGUE, "Mine")), calls), runner=_sync)
    assert client.get(f"/api/leagues/{LEAGUE}/history").status_code == 404
    assert client.get(f"/api/leagues/{LEAGUE}/history/progress").json()["phase"] == "idle"
    started = client.post(f"/api/leagues/{LEAGUE}/history")
    assert started.status_code == 202 and started.json()["status"] == "running"
    progress = client.get(f"/api/leagues/{LEAGUE}/history/progress").json()
    assert progress["phase"] == "done" and progress["seasons_done"] == [CURRENT_SEASON]
    body = client.get(f"/api/leagues/{LEAGUE}/history").json()
    assert body["seasons"] and body["members"]
    assert body["members"][0]["display_name"] in {"alpha99", "bravo", "charlie"}
    # Every week read once, the season views once.
    assert sum("scoringPeriodId=" in u for u in calls) == 2
    prof = client.get(f"/api/leagues/{LEAGUE}/managers/{SWID_A}").json()
    assert prof["member_id"] == SWID_A and "finishes" in prof
    assert client.get(f"/api/leagues/{LEAGUE}/managers/{{NOBODY}}").status_code == 404


def test_a_second_post_while_running_does_not_start_another(owner):
    holds = []

    def runner(fn):
        holds.append(fn)      # never run: the job is "in flight"
    client = _client(_fetch(_entries_payload((LEAGUE, "Mine"))), runner=runner)
    assert client.post(f"/api/leagues/{LEAGUE}/history").status_code == 202
    assert client.post(f"/api/leagues/{LEAGUE}/history").status_code == 202
    assert len(holds) == 1


def test_a_fresh_league_answers_200_without_fetching(owner):
    calls = []
    client = _client(_fetch(_entries_payload((LEAGUE, "Mine")), calls), runner=_sync)
    client.post(f"/api/leagues/{LEAGUE}/history")
    before = len(calls)
    assert client.post(f"/api/leagues/{LEAGUE}/history").json()["status"] == "fresh"
    # Nothing at all: owned_league's ownership check is the same _cached
    # call /api/espn/drafts uses, so the account list from the first POST
    # is still good, and "fresh" itself costs no ESPN round trip either.
    assert len(calls) == before


def test_two_concurrent_first_time_posts_start_only_one_job(owner):
    """Real threads, both racing to start the SAME never-imported league's
    first import -- not the sequential double-call above, which never
    exercises two requests actually overlapping. start_history's running
    check and its registration in _JOBS must be one atomic step under
    _LOCK, or two requests that both land before either has registered can
    each decide "nothing running yet" and each start their own job against
    the same league file.
    """
    started = []
    entered = threading.Event()
    release = threading.Event()

    def runner(fn):
        # Whichever request wins the race reaches here and parks, so the
        # test can look for a second job trying to reach it too before
        # letting the winner's request finish.
        started.append(fn)
        entered.set()
        release.wait(timeout=5)

    client = _client(_fetch(_entries_payload((LEAGUE, "Mine"))), runner=runner)
    results = []

    def post():
        results.append(client.post(f"/api/leagues/{LEAGUE}/history").status_code)

    t1 = threading.Thread(target=post)
    t2 = threading.Thread(target=post)
    t1.start()
    t2.start()
    assert entered.wait(timeout=5), "neither POST ever reached a job"
    # A real window for a second, wrongly-started job to also reach the
    # runner, before letting the one that did proceed.
    time.sleep(0.2)
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert sorted(results) == [202, 202]
    assert len(started) == 1


def test_a_failure_after_the_walk_is_recorded_and_the_next_post_retries(owner, monkeypatch):
    """_rebuild_standings raising after import_activity already succeeded
    must not leave progress reading "done" -- otherwise every later POST
    would answer "fresh" while the GET routes 500 against a league whose
    league_standings table was never written."""
    def boom(conn):
        raise RuntimeError("standings blew up")
    monkeypatch.setattr(lh, "_rebuild_standings", boom)
    client = _client(_fetch(_entries_payload((LEAGUE, "Mine"))), runner=_sync)

    started = client.post(f"/api/leagues/{LEAGUE}/history")
    assert started.status_code == 202
    progress = client.get(f"/api/leagues/{LEAGUE}/history/progress").json()
    assert progress["phase"] == "failed"
    assert "standings blew up" in progress["error"]

    # A failed job is not "running" (the retry below must be allowed to
    # start) and its league must not be read as "fresh" either -- the walk
    # never finished cleanly, whatever import_activity itself stored.
    retry = client.post(f"/api/leagues/{LEAGUE}/history")
    assert retry.status_code == 202 and retry.json()["status"] == "running"


def test_a_failure_before_the_job_starts_releases_the_reservation(owner, monkeypatch):
    """_imported (or _fresh) raising after the slot is reserved -- a DB read
    error, say -- must not leave the placeholder "running" Progress behind:
    nothing would ever advance it, and every later POST would answer 202
    running forever against a job that never actually started."""
    real_imported = lh._imported
    should_boom = {"on": True}

    def maybe_boom(league_id):
        if should_boom["on"]:
            raise RuntimeError("disk on fire")
        return real_imported(league_id)
    monkeypatch.setattr(lh, "_imported", maybe_boom)
    client = _client(_fetch(_entries_payload((LEAGUE, "Mine"))), runner=_sync)

    with pytest.raises(RuntimeError):
        client.post(f"/api/leagues/{LEAGUE}/history")

    # The reservation must not have leaked: progress reads idle, not a
    # "running" placeholder nothing will ever advance.
    assert client.get(f"/api/leagues/{LEAGUE}/history/progress").json()["phase"] == "idle"

    should_boom["on"] = False
    retry = client.post(f"/api/leagues/{LEAGUE}/history")
    assert retry.status_code == 202 and retry.json()["status"] == "running"
