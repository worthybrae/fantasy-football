"""Many live sessions in one process, keyed by the `espn_live` cookie.

tests/test_live_api.py pins every connect to the default session so its
`state` dict keeps meaning what it always meant. This file is the other
half: real, distinct sids, and the isolation between them.
"""
import types

import pytest
from fastapi.testclient import TestClient

from api import live
from api.live import DEFAULT_SID, SID_COOKIE, SID_MAX_AGE
from api.main import create_app

try:
    from tests.test_live_api import _seed_minimal_live_db
except ImportError:                       # tests/ is not a package
    from test_live_api import _seed_minimal_live_db


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """No network, no real league files, no ESPN lobby lookups."""
    monkeypatch.setattr("pipeline.leagues.LEAGUES_ROOT",
                        str(tmp_path / "leagues_root"))
    monkeypatch.setattr("api.live.fetch_team_slots", lambda *a, **k: {})
    monkeypatch.setattr("api.live.fetch_league_settings", lambda *a, **k: None)
    monkeypatch.setattr("api.live.billing.is_free_draft", lambda league_id: True)


def _fake_socket(seen, stops):
    """A run_socket_listener that opens, publishes a live handle, and stays
    connected until its stop_event is set."""
    def fake(listener, league_id, team_id, swid, token, on_change=None,
             stop_event=None, on_activity=None, on_socket=None):
        seen.append({"league_id": league_id, "team_id": team_id,
                     "token": token})
        stops.append(stop_event)
        if on_socket is not None:
            class _Handle:
                def alive(self):
                    return True

                def send(self, text):
                    pass
            on_socket(_Handle())
        if stop_event is not None:
            stop_event.wait(timeout=10)
    return fake


def _connect(client, league_id, team_id="2"):
    resp = client.post("/api/live/connect-token", json={
        "leagueId": league_id, "teamId": team_id, "swid": "{X}",
        "token": f"tok-{league_id}", "season": "2026"})
    assert resp.status_code == 200, resp.text
    return resp


def _app(tmp_path, monkeypatch):
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    seen, stops = [], []
    monkeypatch.setattr("api.live.run_socket_listener", _fake_socket(seen, stops))
    return create_app(path), seen, stops


def _stop_all(clients):
    for c in clients:
        try:
            c.post("/api/live/stop")
        except Exception:            # noqa: BLE001 -- teardown only
            pass


def test_two_cookies_are_two_rooms_and_stopping_one_leaves_the_other(
        tmp_path, monkeypatch):
    app, seen, stops = _app(tmp_path, monkeypatch)
    a, b = TestClient(app), TestClient(app)
    try:
        _connect(a, "1")
        _connect(b, "2")
        assert a.cookies.get(SID_COOKIE) != b.cookies.get(SID_COOKIE)

        sa, sb = a.get("/api/live/state").json(), b.get("/api/live/state").json()
        assert sa["active"] is True and sb["active"] is True
        assert sa["league_id"] == "1"
        assert sb["league_id"] == "2"
        assert [s["league_id"] for s in seen] == ["1", "2"]
        assert app.state.live_registry.active_count() == 2

        assert a.post("/api/live/stop").json()["active"] is False
        assert a.get("/api/live/state").json()["active"] is False
        # b's socket was never asked to stop; only a's was.
        assert stops[0].is_set() and not stops[1].is_set()
        assert b.get("/api/live/state").json()["active"] is True
        assert app.state.live_registry.active_count() == 1
    finally:
        _stop_all([a, b])


def test_a_second_connect_on_the_same_cookie_supersedes_only_itself(
        tmp_path, monkeypatch):
    app, seen, stops = _app(tmp_path, monkeypatch)
    a = TestClient(app)
    try:
        _connect(a, "1")
        sid = a.cookies.get(SID_COOKIE)
        _connect(a, "2")
        assert a.cookies.get(SID_COOKIE) == sid, "a reconnect keeps its room"
        assert stops[0].is_set(), "the first listener was stopped first"
        assert app.state.live_registry.active_count() == 1
        assert a.get("/api/live/state").json()["league_id"] == "2"
    finally:
        _stop_all([a])


def test_no_cookie_sees_no_session_even_while_another_runs(tmp_path, monkeypatch):
    app, seen, stops = _app(tmp_path, monkeypatch)
    a, stranger = TestClient(app), TestClient(app)
    try:
        _connect(a, "1")
        assert stranger.cookies.get(SID_COOKIE) is None
        body = stranger.get("/api/live/state").json()
        assert body["active"] is False
        assert body["token_received"] is False
        assert stranger.get("/api/live/connect-progress").json()["phase"] == "idle"
        assert stranger.get("/api/live/board").json() == {"active": False}
        resp = stranger.post("/api/live/select", json={"player_id": "p1"})
        assert resp.status_code == 409
        # A cookie for a room this process never held is the same stranger.
        ghost = TestClient(app, cookies={SID_COOKIE: "never-minted"})
        assert ghost.get("/api/live/state").json()["active"] is False
        assert ghost.get("/api/live/board").status_code == 404
        assert ghost.post("/api/live/select", json={"player_id": "p1"}).status_code == 409
        assert ghost.post("/api/live/stop").status_code == 409
    finally:
        _stop_all([a])


def test_connect_token_hands_the_browser_its_room_cookie(tmp_path, monkeypatch):
    app, seen, stops = _app(tmp_path, monkeypatch)
    a = TestClient(app)
    try:
        resp = _connect(a, "1")
        header = resp.headers["set-cookie"]
        assert f"{SID_COOKIE}=" in header
        assert "HttpOnly" in header
        assert f"Max-Age={SID_MAX_AGE}" in header and SID_MAX_AGE == 43200
        assert "SameSite=lax" in header
        assert a.cookies.get(SID_COOKIE) != DEFAULT_SID
    finally:
        _stop_all([a])


def test_live_settings_follow_the_cookie(tmp_path, monkeypatch):
    app, seen, stops = _app(tmp_path, monkeypatch)
    a = TestClient(app)
    try:
        _connect(a, "1")
        sid = a.cookies.get(SID_COOKIE)
        mine = types.SimpleNamespace(app=app, cookies={SID_COOKIE: sid})
        stranger = types.SimpleNamespace(app=app, cookies={})
        settings = live.live_settings(mine)
        assert settings is not None
        assert settings.teams == a.get("/api/live/state").json()["settings"]["teams"]
        assert live.live_settings(stranger) is None
        # The accessor api/main.py holds answers the same two questions.
        assert app.state.live_settings(mine) is settings
        assert app.state.live_settings(stranger) is None
    finally:
        _stop_all([a])
