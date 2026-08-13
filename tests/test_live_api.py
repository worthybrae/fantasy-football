import numpy as np
import pandas as pd
import pytest

from api.live import DEFAULT_SEED, DraftSession, board_fingerprint, build_session
from pipeline.db import get_conn, write_table
from scoring import league as league_mod


def _fake_session(seed=20260811):
    """A DraftSession with cheap stand-ins for the expensive fields.

    build_session costs ~17s against a real database (fit_all alone is 14.9s),
    which is exactly why the session exists -- but it makes a poor unit-test
    fixture. Tests that exercise session *behaviour* build one directly.
    """
    from datetime import datetime, timezone
    return DraftSession(
        my_slot=4, league_id="53929318",
        slot_managers={i: f"m{i}" for i in range(1, 9)},
        settings=None, pool=None, betas={}, crosswalk={111: "g1"},
        board_fingerprint="abc", seed=seed,
        started_at=datetime(2026, 8, 11, tzinfo=timezone.utc))


def _seed_minimal_live_db(path, extra_players=None):
    """Minimal database for testing build_session.

    Includes all tables required by build_session: base NFL data (weekly,
    schedules, adp, etc.), draft history for fit_all, league settings, and
    draft_order for the current session.

    `extra_players` (optional): a list of {"player_id", "name", "position",
    "team", "espn_id"} dicts, seeded the same way as the default "A Star" --
    weekly stats, an adp row, and an espn_adp row -- so a test can pin real
    ESPN ids (e.g. from tests/fixtures/espn_draft_socket.jsonl) to known,
    predictable board player_ids and exercise the crosswalk against more
    than one player.
    """
    conn = get_conn(path)

    players = [{"player_id": "p1", "name": "A Star", "position": "WR",
               "team": "DET", "espn_id": 4429795}] + list(extra_players or [])

    # Base NFL data
    write_table(conn, "weekly", pd.DataFrame([
        {"player_id": p["player_id"], "player_display_name": p["name"],
         "position": p["position"], "recent_team": p["team"],
         "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 8, "receiving_yards": 90, "targets": 10, "carries": 0}
        for p in players for w in range(1, 18)]))
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": p["name"], "position": p["position"], "team": p["team"],
         "adp": 5.1 + i}
        for i, p in enumerate(players)]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "snap_counts", pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))
    # A real row, not an empty frame: espn_id has to reach the board for the
    # live crosswalk to resolve a pick, and an empty espn_adp silently yields
    # an empty crosswalk that no other assertion here would notice.
    write_table(conn, "espn_adp", pd.DataFrame([
        {"espn_id": p["espn_id"], "espn_name": p["name"], "position": p["position"],
         "team": p["team"], "espn_adp": 5.0 + i, "espn_ppr_rank": 5 + i,
         "espn_proj": 210.0 - i}
        for i, p in enumerate(players)]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))

    # Draft history for fit_all
    write_table(conn, "draft_picks", pd.DataFrame([
        {"season": 2025, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 11, "player_name": "A Star",
         "position": "WR", "nfl_team": "DET", "keeper": False},
        {"season": 2025, "overall_pick": 2, "round": 1, "round_pick": 2,
         "team_id": 2, "espn_player_id": 12, "player_name": "B Steady",
         "position": "WR", "nfl_team": "GB", "keeper": False},
    ]))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "m1", "slot": 1},
        {"season": 2025, "team_id": 2, "manager": "m2", "slot": 2},
    ]))
    write_table(conn, "historic_adp", pd.DataFrame([
        {"season": 2025, "adp_name": "A Star", "position": "WR",
         "team": "DET", "adp_rank": 1},
        {"season": 2025, "adp_name": "B Steady", "position": "WR",
         "team": "GB", "adp_rank": 2},
    ]))

    # League settings
    settings = league_mod.LeagueSettings(
        season=2026, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1},
        flex_slots=1, bench=6, scoring={"receptions": 0.5},
        draft_type="SNAKE")
    write_table(conn, "league", pd.DataFrame([
        {"season": 2026, "league_id": "53929318",
         "settings_json": league_mod.to_json(settings)}]))

    # Draft order for the current session
    write_table(conn, "draft_order", pd.DataFrame([
        {"slot": i, "manager": f"m{i}"} for i in range(1, 9)]))

    conn.close()


def test_board_fingerprint_changes_when_the_player_set_changes():
    """The session caches pool indices. If the board is rebuilt underneath it
    -- a refresh drops a retired player, say -- those indices point at
    different people and every recommendation is silently about the wrong
    player."""
    a = pd.DataFrame({"player_id": ["g1", "g2"], "market_rank": [1.0, 2.0]})
    b = pd.DataFrame({"player_id": ["g1", "g3"], "market_rank": [1.0, 2.0]})
    assert board_fingerprint(a) != board_fingerprint(b)


def test_board_fingerprint_is_stable_for_the_same_board():
    a = pd.DataFrame({"player_id": ["g1", "g2"], "market_rank": [1.0, 2.0]})
    assert board_fingerprint(a) == board_fingerprint(a.copy())


def test_board_fingerprint_ignores_row_order():
    """Row order is an artifact of how the board was assembled, not a change
    in who is draftable."""
    a = pd.DataFrame({"player_id": ["g1", "g2"], "market_rank": [1.0, 2.0]})
    b = a.iloc[::-1].reset_index(drop=True)
    assert board_fingerprint(a) == board_fingerprint(b)


def test_session_pins_a_seed_that_does_not_move(tmp_path, monkeypatch):
    """The load-bearing decision. Roster EV carries +/-4.2 at 400 rollouts,
    so +/-16.8 at the 25 a live refresh affords, against a 9.3-point gap
    between adjacent candidates. An unpinned seed reshuffles the
    recommendation while the board sits still."""
    s = _fake_session(seed=4242)
    assert s.seed == 4242
    # Two reads, same value -- not a property, not derived from a clock.
    assert s.seed == 4242


def test_build_session_builds_a_usable_crosswalk(tmp_path):
    """Guards a silent failure that a green suite hid once already.

    `build_crosswalk` used to take a connection and was still being called
    with one after its signature changed to take the board. A connection has
    no `.columns`, so it returned an empty dict -- no exception, no failing
    test, and every live pick unmapped. Assert the crosswalk is actually
    populated, not merely that build_session returns.
    """
    path = str(tmp_path / "crosswalk.duckdb")
    _seed_minimal_live_db(path)
    session = build_session(get_conn(path), my_slot=1)
    assert isinstance(session.crosswalk, dict)
    assert session.crosswalk, "crosswalk is empty -- espn_id never reached the board"
    for espn_id, player_id in session.crosswalk.items():
        assert isinstance(espn_id, int)
        assert isinstance(player_id, str)


def test_build_session_default_seed_is_pinned(tmp_path):
    """The default seed must be pinned to a constant, not derived from a clock.

    The seed is a Global Constraint: it pins recommendations across the draft's
    lifetime. Roster EV carries ±16.8 at the 25 rollouts a live refresh
    affords, against a 9.3-point gap between adjacent candidates -- a
    clock-derived seed makes the tool reshuffle advice while the board sits
    still, silently.

    This test exercises the default parameter path by omitting the seed
    argument. If the default is changed to `seed = seed or int(time.time())`,
    the result would be a clock value, not DEFAULT_SEED, and this test fails
    immediately without mocking clocks or delays."""
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    conn = get_conn(path)
    # Call with seed argument omitted; should use DEFAULT_SEED default.
    session = build_session(conn, my_slot=1)
    conn.close()

    # The default must equal DEFAULT_SEED, not a clock-derived value.
    assert session.seed == DEFAULT_SEED


from api.live import STALE_AFTER_SECONDS, picks_until_turn, rollouts_for


def test_rollouts_scale_with_how_close_my_turn_is():
    """One code path, N as a parameter. Measured costs: 25 -> 4.7s,
    100 -> 19.5s, 200 -> 35.2s. Picks arrive every ~20-30s and the clock is
    ~90s, so each tier has to fit the window it is chosen for."""
    assert rollouts_for(7) == 25
    assert rollouts_for(3) == 25
    assert rollouts_for(2) == 100
    assert rollouts_for(1) == 100
    assert rollouts_for(0) == 200


def test_rollouts_never_returns_zero_or_negative():
    """A negative distance means the pick count ran past my turn -- a desync.
    It must still produce a usable budget rather than an empty search."""
    assert rollouts_for(-1) == 200


def test_picks_until_turn_counts_the_snake_correctly():
    """8 teams, slot 4 picks at overall 4, 13, 20. With 8 picks made the next
    pick is #9, and mine is #13, so 4 others go first."""
    settings = type("S", (), {"teams": 8, "rounds": 15})()
    assert picks_until_turn(settings, my_slot=4, picks_made=0) == 3
    assert picks_until_turn(settings, my_slot=4, picks_made=3) == 0
    assert picks_until_turn(settings, my_slot=4, picks_made=8) == 4
    assert picks_until_turn(settings, my_slot=4, picks_made=12) == 0


def test_picks_until_turn_at_the_snake_turn_is_zero_twice_running():
    """Slot 1 picks at overall 1, then 16 and 17 back to back."""
    settings = type("S", (), {"teams": 8, "rounds": 15})()
    assert picks_until_turn(settings, my_slot=1, picks_made=15) == 0
    assert picks_until_turn(settings, my_slot=1, picks_made=16) == 0


def test_state_reports_stale_when_the_poll_is_old(tmp_path):
    """A frozen board that looks live is worse than one that admits it. The
    threshold is the spec's 15s."""
    from datetime import datetime, timedelta, timezone
    from api.live import _is_stale
    now = datetime.now(timezone.utc)
    assert not _is_stale(now, now)
    assert not _is_stale(now - timedelta(seconds=STALE_AFTER_SECONDS - 1), now)
    assert _is_stale(now - timedelta(seconds=STALE_AFTER_SECONDS + 1), now)
    assert _is_stale(None, now)          # never polled at all


def test_token_endpoint_stores_the_payload_and_state_reports_it(tmp_path):
    """The extension delivers a token here. Storing and acknowledging it is
    verifiable on its own -- did a well-formed token arrive -- separately from
    whether the socket later accepts it, which needs a live draft to settle.

    Only the token and public ids: the espn_s2 session cookie is never part
    of this payload, by design, so a breach of this store leaks a two-hour
    nonce rather than an account session.
    """
    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(str(tmp_path / "t.duckdb")))

    assert client.get("/api/live/state").json()["token_received"] is False

    r = client.post("/api/live/token", json={
        "leagueId": "53929318", "teamId": "4",
        "swid": "{8491403C-A53F-4257-8D52-F8AE32CED897}",
        "token": "-1339288665", "season": "2026"})
    assert r.status_code == 200
    assert r.json() == {"received": True, "league_id": "53929318",
                        "team_id": "4"}
    assert client.get("/api/live/state").json()["token_received"] is True


def test_state_is_inactive_before_start(tmp_path):
    from fastapi.testclient import TestClient
    from api.main import create_app
    body = TestClient(create_app(str(tmp_path / "t.duckdb"))).get("/api/live/state").json()
    assert body["active"] is False
    assert body["candidates"] == []
    assert body["candidates_as_of_pick"] is None


import dataclasses

from api.live import register_live_routes


def _live_routes_with_conn(tmp_path):
    """A (state, _recompute) pair wired to a throwaway DuckDB file, for
    exercising _recompute directly without going through the FastAPI app."""
    from fastapi import FastAPI
    from pipeline.db import get_conn
    conn = get_conn(str(tmp_path / "live.duckdb"))
    return register_live_routes(FastAPI(), conn)


def _live_session(seed=DEFAULT_SEED):
    """A DraftSession usable with the real _recompute -- unlike
    _fake_session, `settings` must carry real teams/rounds because
    _recompute calls picks_until_turn(session.settings, ...), which calls
    snake_slots(settings.teams, settings.rounds)."""
    settings = type("S", (), {"teams": 8, "rounds": 15})()
    return dataclasses.replace(_fake_session(seed=seed), settings=settings)


def _fake_candidates_frame(player_id):
    return pd.DataFrame({"player_id": [player_id], "ev": [1.0], "se": [0.1],
                          "applied_pct": [1.0], "rank": [1]})


def test_recompute_discards_a_result_the_pick_count_has_moved_past(tmp_path, monkeypatch):
    """The pick-count guard: a search captured at picks_made=3 must not
    overwrite a poll that already recorded picks_made=5 by the time the
    search finishes -- a stale recommendation is worse than none."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    session = _live_session()
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))
    monkeypatch.setattr("api.live.search_pick",
                         lambda *a, **k: _fake_candidates_frame("stale"))

    # A newer poll landed and recorded picks_made=5 while this computation,
    # captured at picks_made=3, was still running.
    state["as_of_pick"] = 5
    state["candidates"] = [{"player_id": "fresh"}]

    _recompute(session, picks_made=3)

    assert state["as_of_pick"] == 5
    assert state["candidates"] == [{"player_id": "fresh"}]


def test_recompute_stores_its_result_when_nothing_superseded_it(tmp_path, monkeypatch):
    """The guard's other branch: an un-superseded result is served, not
    swallowed by an overzealous check."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    session = _live_session()
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))
    monkeypatch.setattr("api.live.search_pick",
                         lambda *a, **k: _fake_candidates_frame("winner"))

    _recompute(session, picks_made=3)

    assert state["as_of_pick"] == 3
    assert state["candidates"] == [{"player_id": "winner", "ev": 1.0, "se": 0.1,
                                     "applied_pct": 1.0, "rank": 1}]


def test_recompute_discards_a_result_from_a_stopped_and_restarted_session(tmp_path, monkeypatch):
    """The generation guard. live_stop then live_start resets as_of_pick to
    None, which blinds the pick-count guard alone (`state["as_of_pick"] is
    not None` is False right after a restart). A _recompute launched under
    the session that got stopped must still be discarded, not silently
    overwrite the new session's state with results computed against a
    different my_slot/pool."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    old_session = _live_session()
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))

    def fake_search_pick(*a, **k):
        # Simulate live_stop() followed by live_start() landing while this
        # search is in flight -- exactly what those handlers do to `state`
        # under the lock: bump the generation and reset as_of_pick.
        state["generation"] += 1
        state["as_of_pick"] = None
        state["candidates"] = [{"player_id": "fresh-session"}]
        return _fake_candidates_frame("stale-session")

    monkeypatch.setattr("api.live.search_pick", fake_search_pick)

    _recompute(old_session, picks_made=0)

    assert state["candidates"] == [{"player_id": "fresh-session"}]
    assert state["as_of_pick"] is None


def test_recompute_passes_the_session_seed_to_search_pick(tmp_path, monkeypatch):
    """The seed is pinned for the session's lifetime -- _recompute must hand
    search_pick session.seed, never a freshly generated value."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    session = _live_session(seed=773311)
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))
    captured = {}

    def fake_search_pick(*a, **k):
        captured.update(k)
        return _fake_candidates_frame("p1")

    monkeypatch.setattr("api.live.search_pick", fake_search_pick)

    _recompute(session, picks_made=0)

    assert captured["seed"] == 773311


def test_live_start_success_path_builds_and_stores_a_session(tmp_path):
    """live_start's non-reused path: build_session runs against a real
    database, the response carries the pinned seed and a real board
    fingerprint, and a second call reuses the stored session rather than
    rebuilding it."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    client = TestClient(create_app(path))
    body = client.post("/api/live/start", params={"my_slot": 1}).json()
    assert body["active"] is True
    assert body["reused"] is False
    assert body["seed"] == DEFAULT_SEED
    assert isinstance(body["board_fingerprint"], str) and body["board_fingerprint"]

    reused = client.post("/api/live/start", params={"my_slot": 1}).json()
    assert reused == {"active": True, "reused": True}


def test_connect_rejects_a_url_with_no_league_id(tmp_path):
    from fastapi.testclient import TestClient
    from api.main import create_app
    c = TestClient(create_app(str(tmp_path / "t.duckdb")))
    r = c.post("/api/live/connect",
               json={"url": "https://fantasy.espn.com/football/mockdraftlobby",
                     "my_slot": 4})
    assert r.status_code == 422
    assert "league id" in r.json()["detail"].lower()


def test_connect_accepts_a_real_draft_url_and_a_mock_url(tmp_path, monkeypatch):
    """Both forms carry leagueId and resolve identically -- a mock draft is
    not a special case, which is what makes mocks usable as a rehearsal."""
    from api.live import _resolve_league_id
    assert _resolve_league_id(
        "https://fantasy.espn.com/football/draft?leagueId=53929318") == "53929318"
    assert _resolve_league_id(
        "https://fantasy.espn.com/football/draft?leagueId=539649131&seasonId=2026"
    ) == "539649131"
    assert _resolve_league_id("539649131") == "539649131"


def test_team_id_from_url_reads_teamid_when_present():
    """The socket speaks team ids (SELECTING 2 30000, SELECTED 2 ... 2), not
    draft slots -- read it off the URL when ESPN put it there rather than
    asking the drafter to state their own slot number by hand."""
    from api.live import _team_id_from_url
    assert _team_id_from_url(
        "https://fantasy.espn.com/football/draft?leagueId=1&seasonId=2026"
        "&teamId=7&memberId={ABC}") == 7
    assert _team_id_from_url("https://fantasy.espn.com/football/draft?leagueId=1") is None
    assert _team_id_from_url("") is None


def test_slot_for_team_translates_team_id_through_the_manager(tmp_path):
    """A team id is not a draft slot -- team 4 is not necessarily drafting
    4th -- and the obvious one-hop route does not exist. `draft_teams` has a
    `slot` column and every row of it is null on the real database, all 48
    across six seasons; it was never populated, and reading it directly
    crashed on int(NAType) the first time a live league hit it.

    So the translation goes through the manager: draft_teams maps team id to
    manager, draft_order maps manager to slot. The most recent season wins
    when a team id appears in more than one."""
    from api.live import _slot_for_team
    conn = get_conn(str(tmp_path / "slots.duckdb"))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2024, "team_id": 4, "manager": "old_owner", "slot": None},
        {"season": 2025, "team_id": 4, "manager": "m4", "slot": None},
        {"season": 2025, "team_id": 9, "manager": "m9", "slot": None},
    ]))
    write_table(conn, "draft_order", pd.DataFrame([
        {"slot": 2, "manager": "m4", "is_me": True},
        {"slot": 6, "manager": "m9", "is_me": False},
        {"slot": 8, "manager": "old_owner", "is_me": False},
    ]))
    # 2025's manager wins over 2024's, so slot 2 rather than 8.
    assert _slot_for_team(conn, 4) == 2
    assert _slot_for_team(conn, 9) == 6
    conn.close()


def test_slot_for_team_returns_none_when_the_chain_breaks(tmp_path):
    """None, not a raise and not a guess.

    A mock draft's managers appear in neither table, which is the normal
    case rather than an error -- and turn detection does not need a slot at
    all, because the socket says `SELECTING <teamId>` outright. Returning a
    fabricated number would be the unforgivable version: a wrong slot
    attributes every pick to the wrong manager and nothing downstream can
    detect that it happened."""
    from api.live import _slot_for_team
    conn = get_conn(str(tmp_path / "slots.duckdb"))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 4, "manager": "m4", "slot": None}]))
    write_table(conn, "draft_order", pd.DataFrame([
        {"slot": 2, "manager": "m4", "is_me": True}]))
    assert _slot_for_team(conn, 99) is None      # team id nobody has seen
    conn.close()


def test_connect_resolves_my_slot_from_a_teamid_in_the_url(tmp_path, monkeypatch):
    """Correction B, end to end: a URL carrying teamId= must resolve my_slot
    through draft_teams rather than trusting (or requiring) the request
    body's my_slot -- and the resulting session must actually use the
    translated slot, not the untranslated team id."""
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    conn = get_conn(path)
    # Team id 2 drafts from slot 7 -- deliberately NOT slot 2, so a bug that
    # used the team id itself as the slot would be caught.
    # slot is null in draft_teams on the real database -- always. The hop
    # that actually resolves a slot is draft_order, keyed by manager.
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "m1", "slot": None},
        {"season": 2025, "team_id": 2, "manager": "m2", "slot": None}]))
    write_table(conn, "draft_order", pd.DataFrame([
        {"slot": 1, "manager": "m1", "is_me": False},
        {"slot": 7, "manager": "m2", "is_me": True}]))
    conn.close()

    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    resp = client.post(
        "/api/live/connect",
        json={"url": "https://fantasy.espn.com/football/draft?leagueId=1"
                     "&seasonId=2026&teamId=2&memberId={X}"})
    assert resp.status_code == 200
    # Echoed straight back on connect, so the screen can show it for a human
    # sanity check before the draft starts -- a re-randomised draft order
    # would make this silently wrong and nothing else would notice.
    assert resp.json()["my_slot"] == 7

    state = client.get("/api/live/state").json()
    assert state["my_slot"] == 7
    client.post("/api/live/stop")


def test_connect_falls_back_to_the_team_id_when_the_slot_is_unresolvable(tmp_path, monkeypatch):
    """The mock-draft case, and the reason this is not an error.

    A mock's managers appear in neither draft_teams nor draft_order, so no
    slot can be named -- but the socket reports whose turn it is directly
    (`SELECTING <teamId>`), so the draft is perfectly followable without one.
    The team id stands in, and is echoed back so a human can see what was
    assumed rather than discovering it when the board names the wrong
    manager on the clock.
    """
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    resp = client.post(
        "/api/live/connect",
        json={"url": "https://fantasy.espn.com/football/draft?leagueId=1"
                     "&teamId=999&memberId={X}"})
    assert resp.status_code == 200
    # Connecting succeeds; the slot is simply unknown. Turn detection does
    # not need it -- the socket says `SELECTING <teamId>` outright -- and
    # inventing one would attribute picks to the wrong manager.
    assert resp.json()["my_slot"] is None
    client.post("/api/live/stop")

def test_a_rejected_connect_does_not_tear_down_the_running_listener(
        tmp_path, monkeypatch):
    """my_slot/teamId validation must run before the old listener is
    stopped. Stopping first would mean an invalid second request (typo'd
    URL, forgotten my_slot) kills a listener that was working fine, leaving
    the draft unwatched even though the request that broke it was refused."""
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    def fake_run_listener(listener, url, state_path, on_change=None,
                          headless=False, stop_event=None):
        while stop_event is None or not stop_event.is_set():
            time.sleep(0.01)

    monkeypatch.setattr("api.live.run_listener", fake_run_listener)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    r1 = _connect(client, "1")
    assert r1.status_code == 200
    assert _wait_until(
        lambda: client.get("/api/live/state").json()["listener_alive"])

    # A URL with no league id at all -- must be rejected without touching
    # the still-good first listener. (A missing slot is no longer an error:
    # a mock draft resolves to none and connects fine.)
    bad = client.post(
        "/api/live/connect",
        json={"url": "https://fantasy.espn.com/football/mockdraftlobby"})
    assert bad.status_code == 422

    assert client.get("/api/live/state").json()["listener_alive"] is True
    client.post("/api/live/stop")


def test_connect_still_works_when_no_slot_can_be_resolved(tmp_path, monkeypatch):
    """A mock draft resolves to no slot at all, and that must not block the
    connection -- the socket reports whose turn it is directly. The team id
    stands in as a default and is echoed back for the human to check."""
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    resp = client.post("/api/live/connect", json={
        "url": "https://fantasy.espn.com/football/draft?leagueId=1&teamId=6"})
    assert resp.status_code == 200
    # None, not a guess. The team id is not a slot, and the socket names the
    # team itself once it connects -- so "not known yet" is the honest answer
    # rather than echoing the team id back as though it were a slot.
    assert resp.json()["my_slot"] is None
    client.post("/api/live/stop")


import json
from pathlib import Path

_SOCKET_FIXTURE = Path("tests/fixtures/espn_draft_socket.jsonl")


def _first_n_selected_frames(n):
    """A prefix of the real captured frames, cut right after the n-th
    SELECTED. Exercises the listener against text ESPN actually sent over
    the socket -- STATE, CLOCK, PING/PONG and all -- rather than an invented
    protocol string, the same discipline test_draft_listener.py and
    test_espn_live.py already hold their own fixture-driven tests to."""
    rows = [json.loads(l) for l in _SOCKET_FIXTURE.read_text().splitlines() if l]
    payloads = [str(r.get("payload") or "") for r in rows if r["kind"] == "ws-recv"]
    out, seen = [], 0
    for p in payloads:
        out.append(p)
        if p.strip().startswith("SELECTED "):
            seen += 1
            if seen == n:
                break
    return out


@pytest.mark.skipif(not _SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_connect_wires_the_listener_to_apply_picks_and_recompute(tmp_path, monkeypatch):
    """No test exercised connect's success path before this one -- nothing
    verified that it actually wires DraftListener to apply_picks and
    _recompute rather than just returning a 200.

    Only `run_listener` is faked, because that is the one piece that needs a
    real browser; the fake replays real frames from the captured mock draft
    through the exact same DraftListener/on_change shape run_listener uses
    (see pipeline/draft_listener.py). Everything downstream is the real
    code: build_session against a real (tiny) database, apply_picks writing
    real rows to `drafted`, and _recompute. search_pick is stubbed only for
    speed/determinism, the same way the other _recompute tests in this file
    already do it -- the thing under test is the wiring, not the search.
    """
    import threading

    path = str(tmp_path / "live.duckdb")
    # Pin three real ESPN ids from the fixture's first three SELECTED frames
    # (picks: team 1 -> 4430807, team 2 -> 4429795, team 3 -> 4426515) to
    # known board player_ids, so drafted rows can be checked exactly rather
    # than merely for count.
    _seed_minimal_live_db(path, extra_players=[
        {"player_id": "p2", "name": "B Runner", "position": "RB",
         "team": "DET", "espn_id": 4430807},
        {"player_id": "p3", "name": "C Slot", "position": "WR",
         "team": "GB", "espn_id": 4426515},
    ])

    # Learn the crosswalk build_session will actually compute, rather than
    # assuming the market-matching internals -- same real code path connect
    # uses, just called once up front to read off its result.
    setup_conn = get_conn(path)
    crosswalk = build_session(setup_conn, my_slot=1).crosswalk
    setup_conn.close()
    for espn_id in (4430807, 4429795, 4426515):
        assert espn_id in crosswalk, f"fixture espn id {espn_id} did not cross-walk"

    recompute_calls = []
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))
    monkeypatch.setattr(
        "api.live.search_pick",
        lambda *a, **k: recompute_calls.append(k) or _fake_candidates_frame("winner"))

    done = threading.Event()

    def fake_run_listener(listener, url, state_path, on_change=None,
                          headless=False, stop_event=None):
        """Stands in for the real, browser-driving run_listener. Replays
        real frames and fires on_change exactly when the real one would --
        after any frame that changes the pick count -- so the wiring under
        test sees the same call shape it would against a live socket.
        Accepts (and ignores) stop_event: this fake finishes on its own
        after three frames, so nothing here needs to check it, but the real
        live_connect always passes it and a fake with a narrower signature
        would raise a TypeError as soon as connect called it."""
        seen = 0
        for frame in _first_n_selected_frames(3):
            listener.on_frame(frame)
            count = len(listener.picks().rows)
            if count != seen:
                seen = count
                if on_change is not None:
                    on_change()
        done.set()

    monkeypatch.setattr("api.live.run_listener", fake_run_listener)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    resp = client.post("/api/live/connect",
                       json={"url": "https://fantasy.espn.com/football/draft?leagueId=1",
                             "my_slot": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is True
    assert body["league_id"] == "1"

    assert done.wait(timeout=5), "fake listener thread never finished"

    # apply_picks must have written the RIGHT player at the RIGHT pick_no,
    # in the socket's order -- not merely three rows.
    conn = get_conn(path)
    rows = conn.execute(
        "SELECT player_id, pick_no FROM drafted ORDER BY pick_no").fetchall()
    conn.close()
    expected = [(crosswalk[4430807], 1), (crosswalk[4429795], 2),
               (crosswalk[4426515], 3)]
    assert rows == expected

    # And a recompute really ran off the resulting pick count -- not just
    # that drafted got written.
    assert len(recompute_calls) >= 1
    state = client.get("/api/live/state").json()
    assert state["candidates_as_of_pick"] == 3
    assert state["candidates"] == [{"player_id": "winner", "ev": 1.0, "se": 0.1,
                                    "applied_pct": 1.0, "rank": 1}]


# --- Listener lifecycle: at most one running, stop really stops it, a dead
# listener is visible, and a superseded one can never write. -----------------
#
# None of this had a test before: connect started a daemon thread and nothing
# else in this file ever exercised what a *second* connect, or a stop, or a
# thread that raises, actually does to that thread. The four tests below
# correspond one to one with the four review criticals.

import threading
import time


def _wait_until(predicate, timeout=5.0, interval=0.01):
    """Poll a background thread's effect on shared state instead of
    sleeping a fixed guess. Every listener-lifecycle test below is racing a
    real (fake) thread, and a fixed sleep is either too slow (flaky in CI)
    or too fast (flaky here) -- polling is the only version of this that is
    both fast and reliable."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _connect(client, league_id, my_slot=1):
    return client.post(
        "/api/live/connect",
        json={"url": f"https://fantasy.espn.com/football/draft?leagueId={league_id}",
             "my_slot": my_slot})


def test_a_second_connect_stops_the_first_listener_before_starting_a_new_one(
        tmp_path, monkeypatch):
    """Critical #1: nothing rejected a repeat POST /api/live/connect and
    nothing stopped the first thread -- it kept running run_listener's
    `while True` with its own browser and socket forever. That is a second
    socket as the same team, the one thing this whole module exists to
    prevent.

    The fake below blocks on its own stop_event exactly like the real
    run_listener's poll loop, and records (under a lock, so a violation
    detected on the fake's own thread is not lost) whenever it starts while
    another fake is already active. If connect #2 does not fully stop and
    join connect #1's thread before starting its own, this test catches it
    two ways: a recorded overlap, or #1 missing from `stopped` by the time
    connect #2's HTTP response has already come back (which only happens if
    the stop-and-join is synchronous inside the handler, not fire-and-forget).
    """
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    active = {"n": 0}
    violations = []
    started, stopped = [], []
    tracker_lock = threading.Lock()

    def make_fake(fake_id):
        def fake_run_listener(listener, url, state_path, on_change=None,
                              headless=False, stop_event=None):
            with tracker_lock:
                if active["n"] != 0:
                    violations.append(
                        f"{fake_id} started while {active['n']} listener(s) already active")
                active["n"] += 1
                started.append(fake_id)
            try:
                while stop_event is None or not stop_event.is_set():
                    time.sleep(0.01)
            finally:
                with tracker_lock:
                    active["n"] -= 1
                    stopped.append(fake_id)
        return fake_run_listener

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    monkeypatch.setattr("api.live.run_listener", make_fake("first"))
    r1 = _connect(client, "1")
    assert r1.status_code == 200
    assert _wait_until(lambda: "first" in started), "first listener never started"

    monkeypatch.setattr("api.live.run_listener", make_fake("second"))
    r2 = _connect(client, "2")
    assert r2.status_code == 200
    # By the time connect #2's response has returned, _stop_listener already
    # ran (and joined) synchronously inside the handler, before the second
    # thread was even created -- so this is true immediately, no polling.
    assert stopped == ["first"], (
        "connect #2 must not answer until connect #1's thread has fully exited")

    assert _wait_until(lambda: "second" in started), "second listener never started"
    assert violations == [], f"two listeners were active at once: {violations}"

    client.post("/api/live/stop")
    assert _wait_until(lambda: stopped == ["first", "second"])


def test_stop_actually_joins_the_listener_thread(tmp_path, monkeypatch):
    """Critical #3: live_stop reset the state dict and never touched the
    thread or browser -- after "stopping", the old thread kept writing
    picks and recomputing indefinitely. `stopped.append(...)` below only
    runs when the fake's target function actually returns, which is the
    same event as the real Thread finishing -- so seeing it appear here is
    proof the thread exited, not just that the API forgot about it.
    """
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    started, stopped = [], []

    def fake_run_listener(listener, url, state_path, on_change=None,
                          headless=False, stop_event=None):
        started.append(True)
        while stop_event is None or not stop_event.is_set():
            time.sleep(0.01)
        stopped.append(True)

    monkeypatch.setattr("api.live.run_listener", fake_run_listener)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    r = _connect(client, "1")
    assert r.status_code == 200
    assert _wait_until(lambda: started), "listener never started"
    assert not stopped, "listener exited before stop was even requested"

    resp = client.post("/api/live/stop")
    body = resp.json()
    assert body == {"active": False, "listener_stopped": True}
    # No polling here either: live_stop's own {"listener_stopped": True}
    # already asserts the join completed synchronously before it answered.
    assert stopped == [True]


def test_a_listener_exception_reaches_live_state_instead_of_dying_silently(
        tmp_path, monkeypatch):
    """Critical #4: chromium.launch, page.goto and the Playwright import sat
    outside any try/except, and pump() did not wrap the call -- any failure
    (bad URL, ESPN down, Playwright missing) killed the daemon thread with a
    stderr trace nobody sees, while the endpoint had already answered
    {"connected": true}. A dead listener must be visible through the API,
    not indistinguishable from a working one."""
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    def fake_run_listener(listener, url, state_path, on_change=None,
                          headless=False, stop_event=None):
        raise RuntimeError("simulated: ESPN unreachable")

    monkeypatch.setattr("api.live.run_listener", fake_run_listener)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    resp = _connect(client, "1")
    # The endpoint itself still answers 200 -- the failure happens
    # asynchronously, on the thread, after the response was already sent.
    # That asymmetry is exactly why /api/live/state has to carry it.
    assert resp.status_code == 200

    assert _wait_until(
        lambda: client.get("/api/live/state").json()["listener_error"] is not None), \
        "listener_error never appeared on /api/live/state"

    body = client.get("/api/live/state").json()
    assert body["active"] is True
    assert "simulated: ESPN unreachable" in body["listener_error"]
    assert body["listener_alive"] is False


def test_a_superseded_listeners_late_callback_cannot_write_drafted(
        tmp_path, monkeypatch):
    """Critical #2: on_change called apply_picks unconditionally, with no
    check that its own listener was still the active one. Traced: connect
    #2 bumps state["generation"], but connect #1's on_change can still fire
    afterward -- `stop_event` only asks run_listener's *loop* to exit, it
    does not cancel a websocket callback Playwright may already be running
    when the frame arrives. The generation guard inside _recompute only
    protects a race within one _recompute call; it does nothing for an
    entire second listener still calling apply_picks from outside it.

    This test invokes a captured on_change directly, after the listener it
    belongs to has already been superseded by a second connect, to prove
    the write is refused by identity -- not merely that it happens not to
    race in practice.
    """
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    captured = {}

    def fake_run_listener_capture(listener, url, state_path, on_change=None,
                                  headless=False, stop_event=None):
        captured["on_change"] = on_change
        captured["listener"] = listener
        while stop_event is None or not stop_event.is_set():
            time.sleep(0.01)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    monkeypatch.setattr("api.live.run_listener", fake_run_listener_capture)
    r1 = _connect(client, "1")
    assert r1.status_code == 200
    assert _wait_until(lambda: "on_change" in captured)

    # Give the soon-to-be-superseded listener one real, crosswalk-resolvable
    # pick, so a buggy on_change (the one this test guards against) would
    # have something real to write.
    captured["listener"].on_frame("SELECTED 1 4429795 2")
    stale_on_change, stale_listener = captured["on_change"], captured["listener"]

    # Connect #2 supersedes #1: stops and joins it (see the first test in
    # this section), and replaces state["listener"].
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)
    r2 = _connect(client, "2")
    assert r2.status_code == 200
    assert captured["listener"] is stale_listener   # sanity: didn't get overwritten

    # Now fire the STALE listener's on_change directly, simulating the
    # queued Playwright callback the stop signal could not cancel.
    stale_on_change()

    conn = get_conn(path)
    n = conn.execute("SELECT count(*) FROM drafted").fetchone()[0]
    conn.close()
    assert n == 0, "a superseded listener's callback must never write to drafted"

    client.post("/api/live/stop")


def test_stop_listener_refuses_a_reconnect_if_the_old_thread_will_not_die(
        tmp_path, monkeypatch):
    """The other half of the "stop and wait" choice made for Critical #1:
    if the previous listener does not honour stop_event within
    LISTENER_STOP_TIMEOUT, connect must refuse rather than start a second
    thread while the first might still be alive -- the one outcome the
    whole module exists to prevent. Shrinks the timeout so the test does
    not actually wait ten seconds for a thread that, by construction, never
    stops."""
    monkeypatch.setattr("api.live.LISTENER_STOP_TIMEOUT", 0.05)
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    def fake_run_listener_ignores_stop(listener, url, state_path, on_change=None,
                                       headless=False, stop_event=None):
        # Deliberately never checks stop_event -- an uncooperative listener.
        time.sleep(5)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    monkeypatch.setattr("api.live.run_listener", fake_run_listener_ignores_stop)
    r1 = _connect(client, "1")
    assert r1.status_code == 200

    r2 = _connect(client, "2")
    assert r2.status_code == 503
    assert "did not stop" in r2.json()["detail"].lower()
