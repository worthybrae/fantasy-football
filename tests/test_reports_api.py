import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pipeline.db import get_conn, read_table, write_table
from tests.test_league_report import seed_league


@pytest.fixture
def league_root(tmp_path, monkeypatch):
    """A leagues root with one imported league, 424242, and billing off."""
    root = tmp_path / "leagues"
    root.mkdir()
    seed_league(str(root / "424242.duckdb")).close()
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return str(root)


def _app(tmp_path, league_root, session=None, entries=None, spawn=None):
    from api import reports
    # A bare app with only the report routes registered against the fixture
    # root -- create_app() would register its own set against the real root
    # and the two would clash, and create_app is already exercised by
    # tests/test_api.py.
    from fastapi import FastAPI
    bare = FastAPI()
    reports.register_report_routes(
        bare, root=league_root, spawn=spawn or (lambda name, fn: fn()),
        session_for=(lambda request: session),
        entries_for=(lambda sess: entries or []))
    return TestClient(bare)


def test_store_and_load_round_trip(tmp_path):
    from api import reports
    conn = get_conn(str(tmp_path / "t.duckdb"))
    assert reports.list_reports(conn) == []
    assert reports.load_report(conn, 2024) is None
    reports.store_report(conn, {"season": 2024, "status": "ready", "model": "m", "x": 1})
    reports.store_report(conn, {"season": 2024, "status": "ready", "model": "m", "x": 2})
    reports.store_report(conn, {"season": 2023, "status": "numbers_only", "model": None})
    got = reports.load_report(conn, 2024)
    assert got["x"] == 2 and got["generated_at"]
    rows = reports.list_reports(conn)
    assert [r["season"] for r in rows] == [2024, 2023]
    assert set(rows[0]) == {"season", "generated_at", "status"}


def test_build_report_stores_numbers_only_without_a_client(league_root):
    from api import reports
    payload = reports.build_report("424242", 2024, client=None, root=league_root)
    assert payload["status"] == "numbers_only" and payload["model"] is None
    conn = get_conn(f"{league_root}/424242.duckdb")
    assert reports.load_report(conn, 2024)["report_cards"][0]["grade"] == "A"


def test_build_report_uses_the_writer(league_root):
    from api import reports
    from tests.test_blurbs import FakeClient, _Response
    managers = [f"m{i}" for i in range(1, 9)]
    answer = json.dumps({"intro": "Hi", "cards": [
        {"manager": m, "nickname": "N", "blurb": "B"} for m in managers],
        "rankings": [{"manager": m, "line": "L"} for m in managers]})
    payload = reports.build_report("424242", 2024, client=FakeClient([_Response(answer)]),
                                   root=league_root)
    assert payload["status"] == "ready" and payload["model"] == "claude-haiku-4-5"
    assert payload["intro"] == "Hi" and payload["report_cards"][0]["nickname"] == "N"


def test_build_report_stores_a_failure(league_root):
    from api import reports
    payload = reports.build_report("424242", 2019, root=league_root)
    assert payload["status"] == "failed" and "no graded picks" in payload["reason"]


def test_spawn_build_is_single_flight(league_root):
    from api import reports
    held = []
    def spawn(name, fn):
        held.append(fn)          # do not run yet
    assert reports.spawn_build("424242", 2024, spawn=spawn, root=league_root)
    assert reports.building("424242", 2024)
    assert not reports.spawn_build("424242", 2024, spawn=spawn, root=league_root)
    held[0]()
    assert not reports.building("424242", 2024)


def test_get_reports_lists_and_serves(tmp_path, league_root):
    from api import reports
    reports.build_report("424242", 2024, root=league_root)
    client = _app(tmp_path, league_root)
    assert client.get("/api/leagues/999/reports").json() == []
    assert client.get("/api/leagues/999/report/2024").status_code == 404
    rows = client.get("/api/leagues/424242/reports").json()
    assert rows[0]["season"] == 2024 and rows[0]["status"] == "numbers_only"
    resp = client.get("/api/leagues/424242/report/2024")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "public, max-age=300"
    assert resp.json()["league_name"] == "Test League"
    assert client.get("/api/leagues/424242/report/2019").status_code == 404


def test_post_requires_a_session_that_owns_the_league(tmp_path, league_root):
    client = _app(tmp_path, league_root, session=None)
    assert client.post("/api/leagues/424242/report/2024").status_code == 403

    class Sess:
        swid = "{X}"
        cookies = {"SWID": "{X}", "espn_s2": "s"}
    client = _app(tmp_path, league_root, session=Sess(), entries=[{"league_id": "1"}])
    assert client.post("/api/leagues/424242/report/2024").status_code == 403


def test_post_refuses_a_mock_league(tmp_path, league_root, monkeypatch):
    from api import billing
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: True)

    class Sess:
        swid = "{X}"
        cookies = {}
    client = _app(tmp_path, league_root, session=Sess(), entries=[{"league_id": "424242"}])
    assert client.post("/api/leagues/424242/report/2024").status_code == 400


def test_post_builds_and_answers_202(tmp_path, league_root, monkeypatch):
    from api import billing, reports
    from pipeline import league_history
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: False)
    imported = []
    monkeypatch.setattr(reports, "import_history",
                        lambda league_id, cookies, **kw: imported.append(league_id))

    class Sess:
        swid = "{X}"
        cookies = {"SWID": "{X}", "espn_s2": "s"}
    client = _app(tmp_path, league_root, session=Sess(), entries=[{"league_id": "424242"}])
    resp = client.post("/api/leagues/424242/report/2024")
    assert resp.status_code == 202 and resp.json() == {"status": "building"}
    # The synchronous spawn ran the job: history was refreshed, report stored.
    assert imported == ["424242"]
    assert client.get("/api/leagues/424242/report/2024").status_code == 200


def test_post_refreshes_history_when_the_season_is_missing_even_if_fresh(
        tmp_path, league_root, monkeypatch):
    from api import billing, reports
    from pipeline.db import record_freshness
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: False)
    # The whole-league freshness stamp a PRE-draft import leaves: fresh, but
    # with no picks at all for the season that just finished drafting.
    conn = get_conn(f"{league_root}/424242.duckdb")
    record_freshness(conn, "league", True, 1)
    conn.close()
    imported = []
    monkeypatch.setattr(reports, "import_history",
                        lambda league_id, cookies, **kw: imported.append(league_id))

    class Sess:
        swid = "{X}"
        cookies = {"SWID": "{X}", "espn_s2": "s"}
    client = _app(tmp_path, league_root, session=Sess(), entries=[{"league_id": "424242"}])
    resp = client.post("/api/leagues/424242/report/2026")
    assert resp.status_code == 202
    assert imported == ["424242"]


def test_post_skips_the_refresh_when_fresh_and_the_season_is_on_file(
        tmp_path, league_root, monkeypatch):
    from api import billing, reports
    from pipeline.db import record_freshness
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: False)
    conn = get_conn(f"{league_root}/424242.duckdb")
    record_freshness(conn, "league", True, 1)
    conn.close()
    imported = []
    monkeypatch.setattr(reports, "import_history",
                        lambda league_id, cookies, **kw: imported.append(league_id))

    class Sess:
        swid = "{X}"
        cookies = {"SWID": "{X}", "espn_s2": "s"}
    client = _app(tmp_path, league_root, session=Sess(), entries=[{"league_id": "424242"}])
    resp = client.post("/api/leagues/424242/report/2024")
    assert resp.status_code == 202
    assert imported == []


def test_post_while_building_does_not_start_another(tmp_path, league_root, monkeypatch):
    from api import billing, reports
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: False)
    held = []

    class Sess:
        swid = "{X}"
        cookies = {}
    client = _app(tmp_path, league_root, session=Sess(), entries=[{"league_id": "424242"}],
                  spawn=lambda name, fn: held.append(fn))
    assert client.post("/api/leagues/424242/report/2024").status_code == 202
    assert client.post("/api/leagues/424242/report/2024").status_code == 202
    assert len(held) == 1
    held[0]()


def test_cookies_for_connect_never_raises(monkeypatch):
    """This runs on the connect path -- see `cookies_for_connect`'s own
    docstring -- so a down credential store must come back as "no cookies",
    never as the 503 `custody_for` raises for every other caller."""
    from fastapi import HTTPException
    from starlette.requests import Request
    from api import reports

    def _down(request, store=None):
        raise HTTPException(status_code=503, detail="the credential store is down")

    monkeypatch.setattr("api.drafts.session_for", _down)
    request = Request({"type": "http", "headers": []})
    assert reports.cookies_for_connect(request, "{X}", None) is None
    # The body carried its own credential, so the (raising) session lookup
    # is never even reached.
    assert reports.cookies_for_connect(request, "{X}", "s2") == {
        "SWID": "{X}", "espn_s2": "s2"}


# -- a writer THIS process is holding --------------------------------------
#
# DuckDB caches one database instance per file per process, and every
# connection to it must agree on the configuration -- so `read_only=True`
# against a file this process already has open read-write is refused
# outright ("Can't open a connection to same database file with a different
# configuration than existing connections"), with no "lock" in the message.
# That is not a corner case: `api/main.py` holds the default league's file
# for the life of the server, and `api/live.py` holds a live league's own
# `league_conn` for the whole draft. Every read below went 500 (or silently
# False) before `_open` learned to fall back to a shared handle.


@pytest.fixture
def held_writer(league_root):
    """League 424242's own file, open read-write in THIS process."""
    conn = get_conn(f"{league_root}/424242.duckdb")
    yield conn
    conn.close()


def test_get_reports_serves_while_this_process_holds_the_writer(
        tmp_path, league_root, held_writer):
    from api import reports
    reports.build_report("424242", 2024, root=league_root)
    client = _app(tmp_path, league_root)
    index = client.get("/api/leagues/424242/reports")
    assert index.status_code == 200 and index.json()[0]["season"] == 2024
    one = client.get("/api/leagues/424242/report/2024")
    assert one.status_code == 200 and one.json()["league_name"] == "Test League"
    # The fallback must not turn a public read into a write: a league with no
    # file still answers empty, and no file appears for it.
    assert client.get("/api/leagues/999/reports").json() == []
    assert not (tmp_path / "leagues" / "999.duckdb").exists()


def test_the_shared_fallback_writes_no_schema_on_a_public_read(tmp_path, league_root):
    """The fallback is a plain `duckdb.connect`, never `get_conn`.

    `get_conn` runs DDL -- meta, drafted, draft_order, and an ALTER on any
    older `drafted` -- which is a WRITE, on the one path that is public,
    unauthenticated and supposed to be read-only. A league file that has
    none of those tables must still have none after a GET."""
    import duckdb
    path = f"{league_root}/515151.duckdb"
    bare = duckdb.connect(path)
    bare.execute("CREATE TABLE league_reports (season BIGINT, generated_at TIMESTAMP, "
                 "model VARCHAR, status VARCHAR, payload_json VARCHAR)")
    try:
        client = _app(tmp_path, league_root)
        assert client.get("/api/leagues/515151/reports").json() == []
        tables = {r[0] for r in bare.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        assert tables == {"league_reports"}
    finally:
        bare.close()


def test_on_draft_complete_builds_while_this_process_holds_the_writer(
        league_root, held_writer, monkeypatch):
    """The room holds `league_conn` for the whole draft, so the hook's own
    `open_read` for `names_for` runs against a file this process already has
    open read-write -- it used to raise, be swallowed, and return False."""
    from api import billing, reports
    from tests.test_league_report import _settings
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: False)
    jobs = []

    class Sess:
        settings = _settings(2026)
        settings_from_espn = True
        board_by_id = {"x": {"player_id": "x", "name": "X", "position": "RB",
                             "market_rank": 20.0}}
        team_slots = {}
    assert reports.on_draft_complete(
        "424242", 2026, "{X}", Sess(), [("x", 32)],
        spawn=lambda name, fn: jobs.append(name) or fn(), root=league_root) is True
    assert len(jobs) == 1
    assert reports.load_report(held_writer, 2026) is not None


def test_on_draft_complete_does_not_trust_a_pick_order_espn_did_not_send(
        league_root, monkeypatch):
    """`session.settings_from_espn` is False whenever the connect's ESPN
    settings fetch failed and build_session fell back to the database's own
    `league` row -- whose pick order is last season's. Every other consumer
    gates on that flag; this hook has to as well, or the report names the
    wrong manager for every pick and sounds certain about it."""
    import dataclasses
    from api import billing, reports
    from tests.test_league_report import _settings
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: False)
    captured = {}
    monkeypatch.setattr(reports, "spawn_build",
                        lambda lid, season, picks=None, **kw: captured.update(picks=picks) or True)

    class Sess:
        # Last season's order, reversed: slot 1 would resolve to team 8.
        settings = dataclasses.replace(_settings(2026), pick_order=(8, 7, 6, 5, 4, 3, 2, 1))
        settings_from_espn = False
        board_by_id = {"x": {"player_id": "x", "name": "X", "position": "RB",
                             "market_rank": 20.0}}
        team_slots = {1: "Piss Floor"}
    assert reports.on_draft_complete("424242", 2026, "{X}", Sess(), [("x", 1)],
                                     root=league_root) is True
    row = captured["picks"].iloc[0]
    assert row["manager"] == "Piss Floor" and row["team_id"] is None


def test_spawn_draft_complete_runs_the_hook_off_the_callers_thread():
    """The socket read thread calls this, not `on_draft_complete` -- so the
    entitlement read, the DuckDB open and `is_free_draft`'s possible HTTP
    call all happen somewhere else. Default `spawn` is a daemon thread."""
    import threading
    from api import reports
    seen = []
    monkey = lambda *a, **kw: seen.append((threading.current_thread().name, a))  # noqa: E731
    real, reports.on_draft_complete = reports.on_draft_complete, monkey
    try:
        reports.spawn_draft_complete("424242", 2026, "{X}", object(), [("x", 1)])
        for thread in threading.enumerate():
            if thread.name.startswith("job-draft-complete-"):
                thread.join(timeout=5)
    finally:
        reports.on_draft_complete = real
    assert len(seen) == 1
    name, args = seen[0]
    assert name == "job-draft-complete-424242-2026"
    assert name != threading.current_thread().name
    assert args[:3] == ("424242", 2026, "{X}")


def test_the_history_refresh_check_runs_while_this_process_holds_the_writer(
        tmp_path, league_root, held_writer, monkeypatch):
    """`refresh_history` opens the file to ask whether the import can be
    skipped. Under a held writer that open used to raise, the whole check
    was swallowed by its own `except`, and the import never ran at all."""
    from api import billing, reports
    from pipeline.db import record_freshness
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: False)
    imported = []
    monkeypatch.setattr(reports, "import_history",
                        lambda league_id, cookies, **kw: imported.append(league_id))

    class Sess:
        swid = "{X}"
        cookies = {"SWID": "{X}", "espn_s2": "s"}
    client = _app(tmp_path, league_root, session=Sess(), entries=[{"league_id": "424242"}])
    assert client.post("/api/leagues/424242/report/2024").status_code == 202
    assert imported == ["424242"]
    # The other half of the check, through the same held writer: fresh, with
    # the season already on file, and the import is skipped.
    record_freshness(held_writer, "league", True, 1)
    assert client.post("/api/leagues/424242/report/2024").status_code == 202
    assert imported == ["424242"]
