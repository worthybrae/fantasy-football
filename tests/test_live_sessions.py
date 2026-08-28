"""Many live sessions in one process, keyed by the `espn_live` cookie.

tests/test_live_api.py switches the default room on and pins every connect
to it so its `state` dict keeps meaning what it always meant. This file is
the other half: the default room OFF (production's setting), real and
distinct sids, and the isolation between them.

Clients speak https: the room cookie is `Secure` under the same rule the
custody cookie follows, and a client on plain http would drop it.
"""
import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

from api import live
from api.live import DEFAULT_ROOM_ENV, DEFAULT_SID, SID_COOKIE, SID_MAX_AGE

try:
    from tests.test_live_api import _seed_minimal_live_db
except ImportError:                       # tests/ is not a package
    from test_live_api import _seed_minimal_live_db

HTTPS = "https://testserver"


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """No network, no real league files, no ESPN lobby lookups, and no
    default room -- exactly what a deployment runs with."""
    monkeypatch.delenv(DEFAULT_ROOM_ENV, raising=False)
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
    # Imported here, not at the top: api.main loads `.env` at import unless
    # a test is already running, and a collection-time import would switch
    # billing on (the owner's Stripe key) for every test after this file.
    from api.main import create_app
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    seen, stops = [], []
    monkeypatch.setattr("api.live.run_socket_listener", _fake_socket(seen, stops))
    return create_app(path), seen, stops


def _client(app):
    return TestClient(app, base_url=HTTPS)


def _stop_all(clients):
    for c in clients:
        try:
            c.post("/api/live/stop")
        except Exception:            # noqa: BLE001 -- teardown only
            pass


def test_two_cookies_are_two_rooms_and_stopping_one_leaves_the_other(
        tmp_path, monkeypatch):
    app, seen, stops = _app(tmp_path, monkeypatch)
    a, b = _client(app), _client(app)
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
    a = _client(app)
    try:
        _connect(a, "1")
        sid = a.cookies.get(SID_COOKIE)
        resp = _connect(a, "2")
        assert a.cookies.get(SID_COOKIE) == sid, "a reconnect keeps its room"
        # ...and renews it for the full twelve hours.
        assert f"Max-Age={SID_MAX_AGE}" in resp.headers["set-cookie"]
        assert stops[0].is_set(), "the first listener was stopped first"
        assert app.state.live_registry.active_count() == 1
        assert a.get("/api/live/state").json()["league_id"] == "2"
    finally:
        _stop_all([a])


def test_no_cookie_is_no_session_even_while_another_runs(tmp_path, monkeypatch):
    """With the default room off, a cookieless or unknown-cookie request has
    no session: nothing to read, nothing to stop, and nothing it can delete."""
    app, seen, stops = _app(tmp_path, monkeypatch)
    a, stranger = _client(app), _client(app)
    try:
        _connect(a, "1")
        assert stranger.cookies.get(SID_COOKIE) is None
        body = stranger.get("/api/live/state").json()
        assert body["active"] is False
        assert body["token_received"] is False
        assert stranger.get("/api/live/connect-progress").json()["phase"] == "idle"
        assert stranger.get("/api/live/board").status_code == 404
        assert stranger.post("/api/live/select", json={"player_id": "p1"}).status_code == 409
        assert stranger.post("/api/live/autodraft", json={"on": True}).status_code == 409
        assert stranger.post("/api/live/stop").status_code == 409
        assert stranger.post("/api/live/start?my_slot=1").status_code == 409
        # a's room is untouched by all of that.
        assert a.get("/api/live/state").json()["active"] is True

        for bad in ("never-minted", DEFAULT_SID, "../../etc", "x" * 200, "a b"):
            ghost = TestClient(app, base_url=HTTPS, cookies={SID_COOKIE: bad})
            assert ghost.get("/api/live/state").json()["active"] is False, bad
            assert ghost.get("/api/live/board").status_code == 404, bad
            assert ghost.post("/api/live/stop").status_code == 409, bad
    finally:
        _stop_all([a])


def test_the_default_room_only_exists_when_switched_on(tmp_path, monkeypatch):
    """Cookieless requests reach DEFAULT_SID only under LIVE_DEFAULT_ROOM,
    and a cookieless connect then adopts that room rather than minting."""
    app, seen, stops = _app(tmp_path, monkeypatch)
    plain = _client(app)
    try:
        assert live.sid_for(types.SimpleNamespace(cookies={})) is None
        monkeypatch.setenv(DEFAULT_ROOM_ENV, "1")
        assert live.sid_for(types.SimpleNamespace(cookies={})) == DEFAULT_SID
        assert plain.get("/api/live/board").json() == {"active": False}
        resp = _connect(plain, "1")
        assert f"{SID_COOKIE}={DEFAULT_SID}" in resp.headers["set-cookie"]
        assert app.state.live_registry.get(DEFAULT_SID).state["session"] is not None
    finally:
        _stop_all([plain])


def test_connect_token_hands_the_browser_its_room_cookie(tmp_path, monkeypatch):
    app, seen, stops = _app(tmp_path, monkeypatch)
    a = _client(app)
    try:
        resp = _connect(a, "1")
        header = resp.headers["set-cookie"]
        assert f"{SID_COOKIE}=" in header
        assert "HttpOnly" in header
        assert "Secure" in header
        assert f"Max-Age={SID_MAX_AGE}" in header and SID_MAX_AGE == 43200
        assert "SameSite=lax" in header
        assert a.cookies.get(SID_COOKIE) != DEFAULT_SID
    finally:
        _stop_all([a])


def test_the_session_endpoint_mints_once_and_renews_after(tmp_path, monkeypatch):
    app, seen, stops = _app(tmp_path, monkeypatch)
    a = _client(app)
    first = a.post("/api/live/session")
    assert first.json() == {"sid_set": True}
    sid = a.cookies.get(SID_COOKIE)
    assert sid and sid != DEFAULT_SID
    assert f"Max-Age={SID_MAX_AGE}" in first.headers["set-cookie"]
    second = a.post("/api/live/session")
    assert second.json() == {"sid_set": False}
    assert a.cookies.get(SID_COOKIE) == sid
    assert f"Max-Age={SID_MAX_AGE}" in second.headers["set-cookie"]
    # No room was created for an idle cookie.
    assert app.state.live_registry.get(sid) is None
    assert a.get("/api/live/state").json()["active"] is False


def test_progress_polled_during_a_connect_belongs_to_the_polling_client(
        tmp_path, monkeypatch):
    """The page asks for its cookie, then starts the connect POST and the
    progress poll together. The poll must read its own connect's stages
    from the first read, while the POST is still blocked -- and never
    another client's -- with real minted sids."""
    app, seen, stops = _app(tmp_path, monkeypatch)
    a, other = _client(app), _client(app)
    try:
        _connect(other, "2")             # somebody else's finished record
        assert a.post("/api/live/session").json() == {"sid_set": True}

        # Hold the connect inside build_session so the poll runs mid-connect.
        real_build = live.build_session
        entered, release = threading.Event(), threading.Event()

        def slow_build(*args, **kwargs):
            entered.set()
            release.wait(timeout=10)
            return real_build(*args, **kwargs)
        monkeypatch.setattr("api.live.build_session", slow_build)

        result = {}

        def connect():
            result["resp"] = a.post("/api/live/connect-token", json={
                "leagueId": "1", "teamId": "2", "swid": "{X}",
                "token": "tok-1", "season": "2026"})
        t = threading.Thread(target=connect)
        t.start()
        assert entered.wait(timeout=10)
        mid = a.get("/api/live/connect-progress").json()
        assert mid["phase"] == "connecting"
        assert mid["facts"]["league_id"] == "1"
        # Mid-build: the stages before the board are done, the rest wait.
        status = {st["key"]: st["status"] for st in mid["stages"]}
        assert status["token"] == "ok"
        assert "pending" in status.values()
        release.set()
        t.join(timeout=30)
        assert result["resp"].status_code == 200, result["resp"].text
        done = a.get("/api/live/connect-progress").json()
        assert done["facts"]["league_id"] == "1"
        assert other.get("/api/live/connect-progress").json()["facts"]["league_id"] == "2"
    finally:
        release.set() if "release" in dir() else None
        _stop_all([a, other])


def test_a_new_cookie_taking_the_same_seat_evicts_the_old_room(tmp_path, monkeypatch):
    """A browser whose cookie expired mid-draft comes back under a new sid
    for the same (league, team). One seat, one socket: the old room is
    stopped, its record deleted, and it is gone from the registry."""
    app, seen, stops = _app(tmp_path, monkeypatch)
    a, b = _client(app), _client(app)
    try:
        _connect(a, "1", team_id="2")
        old_sid = a.cookies.get(SID_COOKIE)
        assert app.state.live_registry.active_count() == 1
        _connect(b, "1", team_id="2")
        assert stops[0].is_set(), "the old seat's listener was stopped"
        assert app.state.live_registry.active_count() == 1
        assert app.state.live_registry.get(old_sid) is None
        assert a.get("/api/live/state").json()["active"] is False
        assert b.get("/api/live/state").json()["active"] is True
        # A different seat in the same league is a leaguemate, not a
        # duplicate: it stays.
        c = _client(app)
        _connect(c, "1", team_id="3")
        assert app.state.live_registry.active_count() == 2
        _stop_all([c])
    finally:
        _stop_all([a, b])


def test_a_seat_whose_listener_will_not_stop_refuses_the_new_room(tmp_path, monkeypatch):
    """If the old room's listener does not exit in time its thread may still
    be using its league connection, so the new connect must fail rather
    than drop the room and reopen the same single-writer file underneath
    a live thread. Same 503 as the same-cookie supersede path."""
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    release = threading.Event()
    seen = []

    def stubborn(listener, league_id, team_id, swid, token, on_change=None,
                 stop_event=None, on_activity=None, on_socket=None):
        seen.append(league_id)
        release.wait(timeout=30)          # ignores stop_event on purpose
    monkeypatch.setattr("api.live.run_socket_listener", stubborn)
    monkeypatch.setattr("api.live.LISTENER_STOP_TIMEOUT", 0.2)
    from api.main import create_app
    app = create_app(path)
    a, b = _client(app), _client(app)
    try:
        _connect(a, "1", team_id="2")
        old_sid = a.cookies.get(SID_COOKIE)
        # The page's order: the room cookie first, then the connect. A cookie
        # minted by a connect that then fails does not reach the browser (an
        # HTTPException's response is built fresh), which is one more reason
        # the client asks for it up front.
        assert b.post("/api/live/session").json() == {"sid_set": True}
        resp = b.post("/api/live/connect-token", json={
            "leagueId": "1", "teamId": "2", "swid": "{X}",
            "token": "tok-1", "season": "2026"})
        assert resp.status_code == 503, resp.text
        assert "did not stop" in resp.json()["detail"]
        assert app.state.live_registry.get(old_sid) is not None
        assert app.state.live_registry.get(old_sid).state["listener"] is not None
        assert a.get("/api/live/state").json()["active"] is True
        assert len(seen) == 1, "no second socket was opened for the seat"
        progress = b.get("/api/live/connect-progress").json()
        assert progress["phase"] == "failed"
    finally:
        release.set()
        _stop_all([a, b])


def test_live_settings_follow_the_cookie(tmp_path, monkeypatch):
    app, seen, stops = _app(tmp_path, monkeypatch)
    a = _client(app)
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


def test_sid_activity_is_stamped_by_routes_and_frames(tmp_path, monkeypatch):
    app, seen, stops = _app(tmp_path, monkeypatch)
    a = _client(app)
    try:
        _connect(a, "1")
        room = app.state.live_registry.get(a.cookies.get(SID_COOKIE))
        before = room.last_activity
        time.sleep(0.01)
        a.get("/api/live/state")
        assert room.last_activity > before
    finally:
        _stop_all([a])


def test_the_load_budget_for_rollouts():
    assert live.rollouts_for_load(10) == 400
    assert live.rollouts_for_load(50) == 400
    assert live.rollouts_for_load(51) == 150
    assert live.RECOMPUTE_SLOTS._value >= 2


def test_the_reaper_drops_idle_rooms_and_keeps_fresh_ones(tmp_path, monkeypatch):
    """Four hours of silence retires a room -- listener stopped, record
    gone, dropped from the registry -- while a room touched a minute ago
    stays. An idle room with nothing in it (what a failed connect leaves
    behind) goes the same way."""
    app, seen, stops = _app(tmp_path, monkeypatch)
    a, b = _client(app), _client(app)
    try:
        _connect(a, "1", team_id="2")
        _connect(b, "2", team_id="2")
        registry = app.state.live_registry
        old_sid, fresh_sid = a.cookies.get(SID_COOKIE), b.cookies.get(SID_COOKIE)
        # ...and an empty room from a cookie that never connected.
        ghost = registry.get_or_create("ghost-room")
        now = time.monotonic()
        registry.get(old_sid).last_activity = now - 4 * 3600
        ghost.last_activity = now - 4 * 3600
        registry.get(fresh_sid).last_activity = now - 60

        gone = registry.run_once(idle=3 * 3600, now=now)
        assert sorted(gone) == sorted([old_sid, "ghost-room"])
        assert stops[0].is_set() and not stops[1].is_set()
        assert registry.get(old_sid) is None
        assert registry.get("ghost-room") is None
        assert registry.get(fresh_sid) is not None
        assert registry.active_count() == 1
        assert a.get("/api/live/state").json()["active"] is False
        assert b.get("/api/live/state").json()["active"] is True
        # The evicted room's record went with it; the fresh one's stands.
        from api.live_records import record_store
        assert record_store(str(tmp_path / "live.duckdb")).load(old_sid) is None
        assert record_store(str(tmp_path / "live.duckdb")).load(fresh_sid) is not None
    finally:
        _stop_all([a, b])


def test_the_reaper_leaves_a_room_whose_listener_will_not_stop(tmp_path, monkeypatch):
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    release = threading.Event()

    def stubborn(listener, league_id, team_id, swid, token, on_change=None,
                 stop_event=None, on_activity=None, on_socket=None):
        release.wait(timeout=30)
    monkeypatch.setattr("api.live.run_socket_listener", stubborn)
    monkeypatch.setattr("api.live.LISTENER_STOP_TIMEOUT", 0.2)
    from api.main import create_app
    app = create_app(path)
    a = _client(app)
    try:
        _connect(a, "1", team_id="2")
        registry = app.state.live_registry
        sid = a.cookies.get(SID_COOKIE)
        now = time.monotonic()
        registry.get(sid).last_activity = now - 4 * 3600
        assert registry.run_once(idle=3 * 3600, now=now) == []
        assert registry.get(sid) is not None
        assert registry.get(sid).state["listener"] is not None
    finally:
        release.set()
        _stop_all([a])


def test_connect_and_frames_stamp_activity(tmp_path, monkeypatch):
    """A room mid-build is not idle: the connect stamps it, and so does
    every socket frame after."""
    app, seen, stops = _app(tmp_path, monkeypatch)
    a = _client(app)
    try:
        assert a.post("/api/live/session").json()["sid_set"] is True
        room = app.state.live_registry.get_or_create(a.cookies.get(SID_COOKIE))
        room.last_activity = time.monotonic() - 4 * 3600
        _connect(a, "1", team_id="2")
        assert time.monotonic() - room.last_activity < 60
    finally:
        _stop_all([a])


def test_the_fake_socket_switch_routes_the_pump_to_the_replay(tmp_path, monkeypatch):
    """LIVE_FAKE_SOCKET on: the pump calls api.live_fake_socket's runner
    with run_socket_listener's signature, and never the real one."""
    import sys
    import types as _types
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    real_calls, fake_calls = [], []

    def real(*a, **k):
        real_calls.append(a)

    def fake(listener, league_id, team_id, swid, token, on_change=None,
             stop_event=None, on_activity=None, on_socket=None):
        fake_calls.append({"league_id": league_id, "team_id": team_id,
                           "swid": swid, "token": token,
                           "hooks": (on_change, on_activity, on_socket)})
        if stop_event is not None:
            stop_event.wait(timeout=10)
    monkeypatch.setattr("api.live.run_socket_listener", real)
    monkeypatch.setitem(sys.modules, "api.live_fake_socket",
                        _types.SimpleNamespace(run_fake_socket_listener=fake))
    monkeypatch.setenv(live.FAKE_SOCKET_ENV, "1")
    from api.main import create_app
    a = _client(create_app(path))
    try:
        _connect(a, "1", team_id="2")
        assert real_calls == []
        assert len(fake_calls) == 1
        assert fake_calls[0]["league_id"] == "1" and fake_calls[0]["token"] == "tok-1"
        assert all(h is not None for h in fake_calls[0]["hooks"])
    finally:
        _stop_all([a])
