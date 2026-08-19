import numpy as np
import pandas as pd
import pytest

from api.live import DEFAULT_SEED, DraftSession, board_fingerprint, build_session
from pipeline.db import get_conn, write_table
from scoring import league as league_mod


@pytest.fixture(autouse=True)
def _isolated_leagues_root(tmp_path, monkeypatch):
    """Every /api/live/connect in this file that names a real (non-default)
    league id -- which is every one of them, since parse_league_id only ever
    reads a URL's digits and never the __default__ sentinel -- now provisions
    a per-league file under pipeline.leagues.LEAGUES_ROOT. Autouse and
    file-wide so no test here, including ones written before per-league
    routing existed and mentioning nothing about leagues, can create a real
    file under the repo's data/leagues/. Returns the root so a test that
    needs to pre-seed a league's own file (see league_db_path) can find it.
    """
    root = str(tmp_path / "leagues_root")
    monkeypatch.setattr("pipeline.leagues.LEAGUES_ROOT", root)
    return root


@pytest.fixture(autouse=True)
def _stub_team_slots_fetch(monkeypatch):
    """Keep every connect in this file hermetic. On connect, api.live now
    fetches ESPN's real team names AND the league's real roster/scoring for the
    session -- both network calls. Both are best-effort (team names -> {},
    settings -> None, so a live draft still connects offline), but a real GET
    in a unit test is slow and non-deterministic. Stub both file-wide: team
    slots to {} (the board test attaches team_slots to its session directly),
    and league settings to None so build_session falls back to the seeded
    database's own settings, which every existing test already relies on.
    """
    monkeypatch.setattr("api.live.fetch_team_slots", lambda *a, **k: {})
    monkeypatch.setattr("api.live.fetch_league_settings", lambda *a, **k: None)


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
    """One code path, N as a parameter. Budgets were cut (~0.19s/rollout) so a
    recompute lands in ~2-8s instead of up to ~35s -- a fast mock blew several
    picks past our turn before the old on-the-clock search (200 -> ~35s)
    finished. FAR is smallest (runs on every opponent pick, coarse is fine),
    NOW largest (our actual decision) but still inside a real clock."""
    assert rollouts_for(7) == 12
    assert rollouts_for(3) == 12
    assert rollouts_for(2) == 25
    assert rollouts_for(1) == 25
    assert rollouts_for(0) == 40


def test_rollouts_never_returns_zero_or_negative():
    """A negative distance means the pick count ran past my turn -- a desync.
    It must still produce a usable budget rather than an empty search."""
    assert rollouts_for(-1) == 40


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


def test_state_is_inactive_before_start(tmp_path):
    from fastapi.testclient import TestClient
    from api.main import create_app
    body = TestClient(create_app(str(tmp_path / "t.duckdb"))).get("/api/live/state").json()
    assert body["active"] is False
    assert body["candidates"] == []
    assert body["candidates_as_of_pick"] is None
    # No bookmarklet click yet, so the connect screen keeps showing the
    # install guide rather than following a board that does not exist.
    assert body["token_received"] is False


import dataclasses

from api.live import register_live_routes


def _live_routes_with_conn(tmp_path):
    """A (state, _recompute) pair wired to a throwaway DuckDB file, for
    exercising _recompute directly without going through the FastAPI app."""
    from fastapi import FastAPI
    from pipeline.db import get_conn
    path = str(tmp_path / "live.duckdb")
    conn = get_conn(path)
    return register_live_routes(FastAPI(), conn, path)


def _live_session(seed=DEFAULT_SEED):
    """A DraftSession usable with the real _recompute -- unlike
    _fake_session, `settings` must carry real teams/rounds because
    _recompute calls picks_until_turn(session.settings, ...), which calls
    snake_slots(settings.teams, settings.rounds)."""
    settings = type("S", (), {"teams": 8, "rounds": 15})()
    return dataclasses.replace(_fake_session(seed=seed), settings=settings)


def _fake_candidates_frame(player_id):
    """A stand-in for rank_available's return shape -- gain_now, not EV."""
    return pd.DataFrame({"player_id": [player_id], "position": ["WR"],
                          "proj_points": [200.0], "vor_points": [50.0],
                          "gain_now": [12.5], "survive_pct": [80.0],
                          "fills": ["WR1"], "rank": [1]})


def _fake_survival_frame(*a, **k):
    """A stand-in for survival()'s return shape, for tests that mock the
    ranking step entirely (its own contents never reach a mocked
    rank_available)."""
    return pd.DataFrame({"player_id": [], "avail_pct": []})


def test_state_candidates_carry_gain_now(tmp_path):
    """The recommendation is gain_now, not simulated end-of-draft EV.

    EV's standard error was larger than the spread between good candidates,
    so the top slot moved with the sampling seed. gain_now is deterministic
    given the survival estimate -- and that determinism is exactly what
    ought to be under test, so this drives the real engine (real pool, real
    settings, real survival() and rank_available()) through a real
    /api/live/state GET rather than mocking the ranking step, the way the
    recompute-guard tests below do.

    `register_live_routes` gives `(state, _recompute)` for exercising
    _recompute directly (see `_live_routes_with_conn`), but never hands back
    the FastAPI app it mounted routes on, so there is no existing fixture
    that also lets a test hit the HTTP layer. Building the app and its
    TestClient inline here, the same two lines `_live_routes_with_conn`
    already uses plus wrapping them in a TestClient, isn't a new fixture
    style -- it's the same pieces already used to reach state/_recompute.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path, extra_players=[
        {"player_id": "p2", "name": "B Runner", "position": "RB",
         "team": "DET", "espn_id": 4430807},
        {"player_id": "p3", "name": "C Slot", "position": "WR",
         "team": "GB", "espn_id": 4426515},
    ])
    conn = get_conn(path)
    app = FastAPI()
    state, _recompute = register_live_routes(app, conn, path)
    client = TestClient(app)

    session = build_session(conn, my_slot=1)
    state["session"] = session
    _recompute(session, picks_made=0)

    body = client.get("/api/live/state").json()
    assert body["candidates"], "no recommendation produced"
    row = body["candidates"][0]
    assert set(row) >= {"player_id", "position", "proj_points", "vor_points",
                        "gain_now", "survive_pct", "fills", "rank"}
    assert "ev" not in row
    gains = [c["gain_now"] for c in body["candidates"]]
    assert gains == sorted(gains, reverse=True)


def test_recompute_discards_a_result_the_pick_count_has_moved_past(tmp_path, monkeypatch):
    """The pick-count guard: a search captured at picks_made=3 must not
    overwrite a poll that already recorded picks_made=5 by the time the
    search finishes -- a stale recommendation is worse than none."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    session = _live_session()
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))
    monkeypatch.setattr("api.live.survival", _fake_survival_frame)
    monkeypatch.setattr("api.live.rank_available",
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
    monkeypatch.setattr("api.live.survival", _fake_survival_frame)
    monkeypatch.setattr("api.live.rank_available",
                         lambda *a, **k: _fake_candidates_frame("winner"))

    _recompute(session, picks_made=3)

    assert state["as_of_pick"] == 3
    assert state["candidates"] == [{"player_id": "winner", "position": "WR",
                                     "proj_points": 200.0, "vor_points": 50.0,
                                     "gain_now": 12.5, "survive_pct": 80.0,
                                     "fills": "WR1", "rank": 1}]


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
    monkeypatch.setattr("api.live.survival", _fake_survival_frame)

    def fake_rank_available(*a, **k):
        # Simulate live_stop() followed by live_start() landing while this
        # ranking is in flight -- exactly what those handlers do to `state`
        # under the lock: bump the generation and reset as_of_pick.
        state["generation"] += 1
        state["as_of_pick"] = None
        state["candidates"] = [{"player_id": "fresh-session"}]
        return _fake_candidates_frame("stale-session")

    monkeypatch.setattr("api.live.rank_available", fake_rank_available)

    _recompute(old_session, picks_made=0)

    assert state["candidates"] == [{"player_id": "fresh-session"}]
    assert state["as_of_pick"] is None


def test_recompute_passes_the_session_seed_to_survival(tmp_path, monkeypatch):
    """The seed is pinned for the session's lifetime -- _recompute must hand
    survival session.seed, never a freshly generated value."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    session = _live_session(seed=773311)
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))
    monkeypatch.setattr("api.live.rank_available",
                         lambda *a, **k: _fake_candidates_frame("p1"))
    captured = {}

    def fake_survival(*a, **k):
        captured.update(k)
        return pd.DataFrame({"player_id": [], "avail_pct": []})

    monkeypatch.setattr("api.live.survival", fake_survival)

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


def test_connect_resolves_my_slot_from_a_teamid_in_the_url(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """Correction B, end to end: a URL carrying teamId= must resolve my_slot
    through draft_teams rather than trusting (or requiring) the request
    body's my_slot -- and the resulting session must actually use the
    translated slot, not the untranslated team id."""
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    # leagueId=1 in the URL below names a real, non-default league, so the
    # connect builds against ITS OWN file -- draft_teams and draft_order are
    # LEAGUE_TABLES (pipeline/db.py), which live there, not on `path`.
    # provision_league first (copies the universal tables `path` seeded)
    # so the league-specific write below lands on top of that, not wiped by
    # it -- the same idempotent-reconnect guarantee live_connect itself
    # relies on.
    from pipeline.leagues import provision_league
    lg_path = provision_league("1", universal_path=path, root=_isolated_leagues_root)
    conn = get_conn(lg_path)
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
def test_connect_wires_the_listener_to_apply_picks_and_recompute(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """No test exercised connect's success path before this one -- nothing
    verified that it actually wires DraftListener to apply_picks and
    _recompute rather than just returning a 200.

    Only `run_listener` is faked, because that is the one piece that needs a
    real browser; the fake replays real frames from the captured mock draft
    through the exact same DraftListener/on_change shape run_listener uses
    (see pipeline/draft_listener.py). Everything downstream is the real
    code: build_session against a real (tiny) database, apply_picks writing
    real rows to `drafted`, and _recompute. survival/rank_available are
    stubbed only for speed/determinism, the same way the other _recompute
    tests in this file already do it -- the thing under test is the wiring,
    not the ranking.
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

    # leagueId=1 in the URL below is a real, non-default league: the
    # connect below builds against its own file, not `path`. draft_teams and
    # draft_order are LEAGUE_TABLES (pipeline/db.py) and so are not among the
    # universal tables provision_league copies -- on_change's my_slot
    # resolution (via listener.my_team_id -> _slot_for_team) needs them on
    # THAT file, the same as `path` already has them, or my_slot never
    # resolves and _recompute skips every call.
    from pipeline.db import read_table
    from pipeline.leagues import provision_league
    lg_path = provision_league("1", universal_path=path, root=_isolated_leagues_root)
    lg_conn, src_conn = get_conn(lg_path), get_conn(path)
    for table in ("draft_teams", "draft_order"):
        write_table(lg_conn, table, read_table(src_conn, table))
    lg_conn.close()
    src_conn.close()

    recompute_calls = []
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))
    monkeypatch.setattr("api.live.survival", _fake_survival_frame)
    monkeypatch.setattr(
        "api.live.rank_available",
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
    # in the socket's order -- not merely three rows. leagueId=1 is a real,
    # non-default league, so the write landed on its own file, not `path`.
    from pipeline.leagues import league_db_path
    lg_path = league_db_path("1", root=_isolated_leagues_root)
    conn = get_conn(lg_path)
    rows = conn.execute(
        "SELECT player_id, pick_no FROM drafted ORDER BY pick_no").fetchall()
    conn.close()
    expected = [(crosswalk[4430807], 1), (crosswalk[4429795], 2),
               (crosswalk[4426515], 3)]
    assert rows == expected

    # And a recompute really ran off the resulting pick count -- not just
    # that drafted got written. It runs on the background worker now, so poll
    # for its result rather than reading it synchronously.
    assert _wait_until(
        lambda: client.get("/api/live/state").json()["candidates_as_of_pick"] == 3), \
        "recompute worker never produced candidates for pick 3"
    assert len(recompute_calls) >= 1
    state = client.get("/api/live/state").json()
    assert state["candidates"] == [{"player_id": "winner", "position": "WR",
                                    "proj_points": 200.0, "vor_points": 50.0,
                                    "gain_now": 12.5, "survive_pct": 80.0,
                                    "fills": "WR1", "rank": 1}]


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
        tmp_path, monkeypatch, _isolated_leagues_root):
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

    # The stale listener belongs to league "1" -- a real, non-default
    # league -- so a write that slipped past the identity guard would land
    # on ITS OWN file, not `path`. Checking `path` here would pass trivially
    # (nothing was ever routed there for this league), which would no longer
    # pin the guard this test exists for.
    from pipeline.leagues import league_db_path
    lg_path = league_db_path("1", root=_isolated_leagues_root)
    conn = get_conn(lg_path)
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


def test_connect_with_a_league_id_isolates_its_drafted_table(tmp_path, monkeypatch):
    """A connect naming a real league builds against that league's own file,
    so marking a pick there does not touch the default database."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    from pipeline.db import get_conn, read_table

    default_path = str(tmp_path / "default.duckdb")
    _seed_minimal_live_db(default_path)
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)
    monkeypatch.setattr("pipeline.leagues.LEAGUES_ROOT", str(tmp_path / "lg"))

    client = TestClient(create_app(default_path))
    r = client.post("/api/live/connect", json={
        "url": "https://fantasy.espn.com/football/draft?leagueId=777&teamId=2"})
    assert r.status_code == 200
    client.post("/api/live/stop")

    # The default database's drafted table is untouched by league 777's session.
    assert read_table(get_conn(default_path), "drafted").empty
    # And this isn't vacuously true because nothing was ever written anywhere
    # (run_listener is a no-op above, so no pick lands in *any* drafted
    # table during this test) -- pin that the session genuinely routed to
    # league 777's own file, provisioned separately from the default one.
    assert (tmp_path / "lg" / "777.duckdb").exists()


def test_connect_to_the_configured_default_league_uses_the_apps_own_database(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The existing user's own league -- DEFAULT_LEAGUE_ID, "53929318" unless
    overridden -- must resolve to THIS app's own database (`db_path`, a tmp
    file standing in for the real data/nfl.duckdb here), not cold-start a
    fresh per-league file. league_db_path resolves DEFAULT_LEAGUE_ID to
    pipeline.db's own DEFAULT_PATH -- a fixed string, not necessarily this
    app's db_path -- so live_connect has to special-case this id itself,
    before ever calling provision_league, or a test (or a deployment) whose
    db_path differs from the literal "data/nfl.duckdb" would silently open
    the wrong file (or a second, conflicting connection to the one `conn`
    already holds, if the two happen to be the same file).

    This is the regression review caught: without the special case, EVERY
    connect -- including one naming the project's own real league --
    cold-starts, discarding the 712 draft_picks and fitted managers already
    in the database and losing my_slot resolution (draft_teams/draft_order
    would be empty on the fresh file too).
    """
    from pipeline.db import read_table
    from pipeline.leagues import DEFAULT_LEAGUE_ID

    path = str(tmp_path / "default.duckdb")
    _seed_minimal_live_db(path)
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    resp = client.post("/api/live/connect", json={
        "url": f"https://fantasy.espn.com/football/draft?leagueId={DEFAULT_LEAGUE_ID}"
               "&teamId=2"})
    assert resp.status_code == 200
    # my_slot resolves at all -- draft_teams/draft_order were only ever
    # seeded on `path`, so this fails if the session built against anything
    # else. team_id 2 -> m2 -> slot 2 in _seed_minimal_live_db.
    assert resp.json()["my_slot"] == 2
    client.post("/api/live/stop")

    # No per-league file was created for the configured default league.
    assert not (Path(_isolated_leagues_root) / f"{DEFAULT_LEAGUE_ID}.duckdb").exists()
    # And the pick history/board this session actually built against is
    # `path` itself -- proven by reading draft_picks (seeded only on `path`)
    # straight back off it, undisturbed.
    assert not read_table(get_conn(path), "draft_picks").empty


def test_stop_does_not_close_a_league_conn_the_listener_thread_is_still_using(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """Review finding #2: a stop that cannot confirm the listener thread has
    actually exited must not close that session's per-league connection --
    the thread might be using it at that exact moment, and closing a live
    DuckDB connection out from under an in-flight use is worse than leaving
    it open (a leaked connection is recoverable on restart; a use-after-close
    is not). Pre-Task-5 this had no window at all: the shared `conn` was
    immortal, never closed by anything.

    Verified directly by spying on close(), rather than inferring it from a
    file-lock conflict: DuckDB, at least in the version this project pins,
    allows more than one connection to the same file within one process
    (confirmed empirically -- opening a second one did not raise), so a lock
    error is not a reliable signal here. What actually matters is simpler
    and more direct: was .close() ever called on the connection api.live
    itself opened.
    """
    monkeypatch.setattr("api.live.LISTENER_STOP_TIMEOUT", 0.05)
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    from pipeline.db import get_conn as real_get_conn
    closed = []

    class _SpyConn:
        """Delegates everything to the real connection except close(),
        which is recorded rather than merely trusted to have (not) run."""
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def close(self):
            closed.append(True)
            self._inner.close()

    monkeypatch.setattr("api.live.get_conn", lambda p: _SpyConn(real_get_conn(p)))

    def fake_run_listener_ignores_stop(listener, url, state_path, on_change=None,
                                       headless=False, stop_event=None):
        # Deliberately never checks stop_event -- an uncooperative listener,
        # same as test_stop_listener_refuses_a_reconnect_if_the_old_thread_will_not_die.
        time.sleep(5)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    monkeypatch.setattr("api.live.run_listener", fake_run_listener_ignores_stop)
    r1 = _connect(client, "1")
    assert r1.status_code == 200

    stop_resp = client.post("/api/live/stop").json()
    assert stop_resp["listener_stopped"] is False, (
        "test setup: the listener thread must still be alive for this to "
        "pin anything -- if it isn't, LISTENER_STOP_TIMEOUT or the fake "
        "above needs adjusting")

    assert closed == [], (
        "live_stop must not close a per-league connection while its "
        "listener thread is still alive")


def test_state_query_and_stop_close_are_serialized_by_the_lock(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """Review finding: /api/live/state used to snapshot league_conn under
    `lock`, then release the lock before running the picks-made count on it
    -- a concurrent /api/live/stop, on a different HTTP thread (Starlette
    runs each sync handler on its own threadpool thread), could close that
    exact connection in the gap between snapshot and query. The fix moves
    the query inside the same `with lock:` block live_stop/_stop_listener
    take to close the connection, so the two now serialize instead of
    racing.

    Constructed as a real two-thread race, not just reasoned about: get_conn
    is wrapped so cursor() on the per-league connection blocks mid-call
    until a concurrent /api/live/stop has had a real chance to run. With the
    fix, that stop cannot actually close anything until the blocked
    cursor()/count query's whole critical section (which now includes the
    close-gating lock) has released it -- so the close can never land before
    the query finishes, and /api/live/state must return a clean 200. Without
    the fix, the stop's close is free to run while the query is still
    parked in cursor(), so resuming afterward calls cursor() on an
    already-closed connection.
    """
    from pipeline.db import get_conn as real_get_conn

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)

    entered_cursor = threading.Event()
    release_cursor = threading.Event()

    class _SlowSpyConn:
        """Delegates to the real connection, except cursor() blocks until
        told to proceed -- the window the concurrent stop races into."""
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def cursor(self):
            entered_cursor.set()
            release_cursor.wait(timeout=5)
            return self._inner.cursor()

    monkeypatch.setattr("api.live.get_conn", lambda p: _SlowSpyConn(real_get_conn(p)))

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    r = _connect(client, "1")
    assert r.status_code == 200

    results = {}

    def do_state():
        try:
            results["state"] = client.get("/api/live/state")
        except Exception as exc:                          # noqa: BLE001
            results["state_exc"] = exc

    t_state = threading.Thread(target=do_state)
    t_state.start()
    assert entered_cursor.wait(timeout=5), "state handler never reached cursor()"

    def do_stop():
        results["stop"] = client.post("/api/live/stop")

    t_stop = threading.Thread(target=do_stop)
    t_stop.start()
    # A real window for the race to land in, before letting the blocked
    # query proceed: with the fix, the stop thread spends this whole time
    # blocked trying to acquire `lock` (the state handler is still holding
    # it, parked in cursor()) -- it cannot have closed anything yet no
    # matter how long this sleep is.
    time.sleep(0.2)
    release_cursor.set()
    t_state.join(timeout=5)
    t_stop.join(timeout=5)

    assert "state_exc" not in results, (
        f"/api/live/state raised against a connection a concurrent stop "
        f"closed underneath it: {results.get('state_exc')!r}")
    assert results["state"].status_code == 200


# --- Bookmarklet path: /api/live/connect-token opens the socket directly ------
#
# The bookmarklet mints ESPN's per-draft token in the user's own browser and
# delivers only that token plus the public ids. This endpoint opens the socket
# from it with no browser window on this machine (run_socket_listener), so a
# stranger's league with no saved login here still produces a live board. Only
# run_socket_listener is faked -- the one piece that needs a real ESPN socket;
# everything downstream (build_session, apply_picks, _recompute) is real code,
# the same discipline the browser-path wiring test above holds.

@pytest.mark.skipif(not _SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_connect_token_resolves_slot_and_wires_socket_picks(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """End to end for the bookmarklet path: a delivered token must resolve
    my_slot from its team id up front (the socket path knows the team from the
    start, unlike the browser JOIN), open the socket via run_socket_listener,
    and wire the replayed frames through apply_picks and _recompute exactly as
    the browser path does -- plus mark token_received so the connect screen
    knows the click landed."""
    import threading

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path, extra_players=[
        {"player_id": "p2", "name": "B Runner", "position": "RB",
         "team": "DET", "espn_id": 4430807},
        {"player_id": "p3", "name": "C Slot", "position": "WR",
         "team": "GB", "espn_id": 4426515},
    ])

    setup_conn = get_conn(path)
    crosswalk = build_session(setup_conn, my_slot=1).crosswalk
    setup_conn.close()
    for espn_id in (4430807, 4429795, 4426515):
        assert espn_id in crosswalk, f"fixture espn id {espn_id} did not cross-walk"

    # leagueId=1 is a real, non-default league: connect-token builds against
    # its own file. draft_teams/draft_order are LEAGUE_TABLES, so they must be
    # seeded on THAT file for team 2 -> slot 7 to resolve. Team 2 drafts slot
    # 7 (deliberately not 2), so a bug using the team id as the slot is caught.
    from pipeline.leagues import provision_league
    lg_path = provision_league("1", universal_path=path, root=_isolated_leagues_root)
    lg_conn = get_conn(lg_path)
    write_table(lg_conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "m1", "slot": None},
        {"season": 2025, "team_id": 2, "manager": "m2", "slot": None}]))
    write_table(lg_conn, "draft_order", pd.DataFrame([
        {"slot": 1, "manager": "m1", "is_me": False},
        {"slot": 7, "manager": "m2", "is_me": True}]))
    lg_conn.close()

    recompute_calls = []
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))
    monkeypatch.setattr("api.live.survival", _fake_survival_frame)
    monkeypatch.setattr(
        "api.live.rank_available",
        lambda *a, **k: recompute_calls.append(k) or _fake_candidates_frame("winner"))

    done = threading.Event()
    seen_args = {}

    def fake_run_socket_listener(listener, league_id, team_id, swid, token,
                                 on_change=None, stop_event=None,
                                 on_activity=None):
        """Stands in for the real, socket-opening run_socket_listener. Records
        the args it was called with (so the test can prove the token and ids
        reached it) and replays real captured frames, firing on_activity on
        every frame and on_change when the pick count changes -- the same
        contract the real run_socket_listener holds."""
        seen_args.update(league_id=league_id, team_id=team_id, swid=swid,
                         token=token)
        prev = 0
        for frame in _first_n_selected_frames(3):
            listener.on_frame(frame)
            if on_activity is not None:
                on_activity()
            count = len(listener.picks().rows)
            if count != prev:
                prev = count
                if on_change is not None:
                    on_change()
        done.set()

    monkeypatch.setattr("api.live.run_socket_listener", fake_run_socket_listener)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    resp = client.post("/api/live/connect-token", json={
        "leagueId": "1", "teamId": "2", "swid": "{X}",
        "token": "1953383334", "season": "2026"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["connected"] is True
    assert body["league_id"] == "1"
    # Resolved up front from the team id -- no need to wait for the socket.
    assert body["my_slot"] == 7

    assert done.wait(timeout=5), "fake socket listener thread never finished"

    # The token and public ids actually reached run_socket_listener.
    assert seen_args == {"league_id": "1", "team_id": "2", "swid": "{X}",
                         "token": "1953383334"}

    from pipeline.leagues import league_db_path
    conn = get_conn(league_db_path("1", root=_isolated_leagues_root))
    rows = conn.execute(
        "SELECT player_id, pick_no FROM drafted ORDER BY pick_no").fetchall()
    conn.close()
    assert rows == [(crosswalk[4430807], 1), (crosswalk[4429795], 2),
                    (crosswalk[4426515], 3)]

    # survival/rank_available now run on the background recompute worker,
    # not inline in the frame callback, so the result arrives shortly AFTER
    # the listener finishes -- poll for it rather than reading it
    # synchronously.
    assert _wait_until(
        lambda: client.get("/api/live/state").json()["candidates_as_of_pick"] == 3), \
        "recompute worker never produced candidates for pick 3"
    assert len(recompute_calls) >= 1
    state = client.get("/api/live/state").json()
    # The click landed: token_received flips true so the connect screen can
    # stop showing the install guide and follow the board.
    assert state["token_received"] is True
    # on_activity stamped last_poll_at, so the board is not falsely stale --
    # the "no successful update" banner would otherwise show forever, since
    # nothing else ever set last_poll_at.
    assert state["last_poll_at"] is not None
    assert state["stale"] is False
    client.post("/api/live/stop")


def test_connect_token_rejects_missing_or_nonnumeric_fields(tmp_path, monkeypatch):
    """Bad requests are refused up front -- before any listener teardown --
    so a malformed bookmarklet POST can never stop a listener that was
    working. A missing token and a non-numeric team id are the two things the
    socket path cannot proceed without: it has no browser JOIN to learn the
    team from, and _slot_for_team compares against integer team ids."""
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    # Fail loudly if the endpoint ever reaches the listener on a bad request.
    monkeypatch.setattr("api.live.run_socket_listener",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("listener started on an invalid request")))

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    missing_token = client.post("/api/live/connect-token", json={
        "leagueId": "1", "teamId": "2", "swid": "{X}", "token": "",
        "season": "2026"})
    assert missing_token.status_code == 422

    nonnumeric_team = client.post("/api/live/connect-token", json={
        "leagueId": "1", "teamId": "notanumber", "swid": "{X}",
        "token": "99", "season": "2026"})
    assert nonnumeric_team.status_code == 422

    # Nothing was started, so nothing is active.
    assert client.get("/api/live/state").json()["active"] is False


def test_slot_from_socket_derives_slot_for_a_mock():
    """A mock's managers are in no history, so _slot_for_team returns None and
    recommendations would stay blank forever. _slot_from_socket recovers the
    slot from the socket alone: in round 1 the overall pick order is the slot
    order, so my team's position on the clock -- or its first-round pick --
    names its slot, and a reconnect replay must not shift it."""
    from api.live import _slot_from_socket
    from pipeline.draft_listener import DraftListener

    L = DraftListener({})
    L.on_frame("TOKEN 1:100:77:{SWID}:12345")     # names my team as 77
    assert L.my_team_id == 77

    # Two picks in, my team unseen in round 1 -> not known yet.
    L.on_frame("SELECTED 40 1001 2")              # slot 1
    L.on_frame("SELECTED 41 1002 2")              # slot 2
    assert _slot_from_socket(L, 8) is None

    # My team on the clock at pick 3 -> slot 3, before it even picks.
    L.on_frame("SELECTING 77 30000")
    assert _slot_from_socket(L, 8) == 3

    # It picks; still slot 3 by position.
    L.on_frame("SELECTED 77 1003 2")
    assert _slot_from_socket(L, 8) == 3

    # ESPN replays the whole first round on a reconnect JOIN -- the duplicate
    # SELECTED frames must not move the slot.
    for f in ("SELECTED 40 1001 2", "SELECTED 41 1002 2", "SELECTED 77 1003 2"):
        L.on_frame(f)
    assert _slot_from_socket(L, 8) == 3


# --- GET /api/live/board: the full draft-board grid ---------------------------


def test_board_is_inactive_before_a_session(tmp_path):
    """No session -> the endpoint says so and nothing else, the same shape the
    front end already keys the whole board screen off."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    body = TestClient(create_app(str(tmp_path / "t.duckdb"))).get(
        "/api/live/board").json()
    assert body == {"active": False}


def test_board_returns_the_grid_with_snake_positions_and_stats(tmp_path):
    """The whole contract in one pass: columns named (with a placeholder for a
    slot the fetch never learned), is_me on the right column, the snake sending
    an overall-9 pick to the reversed second round's slot 8, player stats
    joined off the board, a crosswalk miss still emitting a cell, and value =
    overall - market_rank with the right sign."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    path = str(tmp_path / "board.duckdb")
    _seed_minimal_live_db(path, extra_players=[
        {"player_id": "p2", "name": "B Runner", "position": "RB",
         "team": "DET", "espn_id": 222},
        {"player_id": "p3", "name": "C Slot", "position": "WR",
         "team": "GB", "espn_id": 333},
    ])

    conn = get_conn(path)
    app = FastAPI()
    state, _recompute = register_live_routes(app, conn, path)

    # A real session -- real board with per-player stats -- with my_slot=7 and
    # team names attached the way _attach_team_slots would on connect, except
    # slot 8 is deliberately omitted so the endpoint's "Team 8" fallback shows.
    session = build_session(conn, my_slot=7, league_id="53929318")
    team_slots = {i: f"Franchise {i}" for i in range(1, 8)}      # 1..7, no 8
    state["session"] = dataclasses.replace(session, team_slots=team_slots)

    # Seed drafted directly: a round-1 pair, a crosswalk miss, and -- the
    # load-bearing case -- the star at overall 9, which the snake sends to slot
    # 8 in the reversed second round (drafted far below rank, so a steal: >0).
    conn.execute("DELETE FROM drafted")
    for pid, overall in [("p2", 1), ("p3", 2), ("ghost_id", 3), ("p1", 9)]:
        conn.execute("INSERT INTO drafted VALUES (?, ?)", [pid, overall])

    body = TestClient(app).get("/api/live/board").json()

    assert body["active"] is True
    assert body["teams"] == 8
    assert body["rounds"] == 16
    assert body["my_slot"] == 7
    assert body["picks_made"] == 4
    # 4 picks made -> pick #5 is on the clock: slot 5, still round 1.
    assert body["on_the_clock"] == 5

    cols = body["columns"]
    assert [c["slot"] for c in cols] == list(range(1, 9))
    assert cols[6]["is_me"] is True                      # slot 7 == my_slot
    assert sum(c["is_me"] for c in cols) == 1            # and only that one
    assert cols[6]["team_name"] == "Franchise 7"
    assert cols[7]["team_name"] == "Team 8"              # the omitted-slot fallback

    cells = {c["overall"]: c for c in body["cells"]}
    assert set(cells) == {1, 2, 3, 9}
    # Round 1 ascends by slot...
    assert (cells[1]["round"], cells[1]["slot"]) == (1, 1)
    assert (cells[2]["round"], cells[2]["slot"]) == (1, 2)
    # ...and overall 9 lands in the REVERSED second round: slot 8.
    assert (cells[9]["round"], cells[9]["slot"]) == (2, 8)

    # Player stats joined from the board.
    assert cells[1]["player"]["name"] == "B Runner"
    assert cells[1]["player"]["position"] == "RB"
    assert cells[9]["player"]["name"] == "A Star"
    assert cells[9]["player"]["last_ppg"] is not None    # p1 has weekly rows
    # The icon inputs: ESPN's own PPR rank (hype/lame vs market_rank) and this
    # year's projected ppg (trending vs last_ppg) = espn_proj / 17.
    assert cells[9]["player"]["espn_ppr_rank"] is not None
    assert cells[9]["player"]["proj_ppg"] == round(210.0 / 17, 1)

    # A crosswalk miss (an id the board does not carry) still emits a cell --
    # id as the name, everything else null -- so the grid never drops a pick.
    ghost = cells[3]["player"]
    assert ghost["name"] == "ghost_id"
    assert ghost["position"] is None
    assert ghost["market_rank"] is None
    assert ghost["value"] is None

    # value = overall - market_rank, and the star taken at 9 is a steal (>0).
    star = cells[9]["player"]
    assert star["market_rank"] is not None
    assert star["value"] == 9 - star["market_rank"]
    assert star["value"] > 0
    conn.close()


def test_build_session_uses_passed_settings_over_the_database(tmp_path):
    """When the connect flow fetched the league's real roster from ESPN, the
    session must draft for THAT roster, not the database's own -- so build_session
    with an explicit settings arg overrides league.load(conn)."""
    import dataclasses as _dc
    from scoring.league import default_settings
    path = str(tmp_path / "s.duckdb")
    _seed_minimal_live_db(path)
    conn = get_conn(path)
    # bench 8 instead of the default 5 -> 18 rounds, a value the seeded db's
    # own league row does not carry, so it can only come from the passed arg.
    custom = _dc.replace(default_settings(), bench=8)
    session = build_session(conn, my_slot=1, settings=custom)
    assert session.settings.bench == 8
    assert session.settings.rounds == custom.rounds == 18
    conn.close()


def test_unmapped_picks_surface_in_state(tmp_path, monkeypatch, _isolated_leagues_root):
    """A pick the crosswalk can't resolve (a D/ST whose id doesn't map, say)
    must show up in /api/live/state.unmapped_picks rather than vanishing --
    that visibility is how a wrong id scheme gets caught. The LivePicks.unmapped
    list was being computed and discarded; now it is stored."""
    path = str(tmp_path / "u.duckdb")
    _seed_minimal_live_db(path)
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))
    monkeypatch.setattr("api.live.survival", _fake_survival_frame)
    monkeypatch.setattr("api.live.rank_available",
                        lambda *a, **k: _fake_candidates_frame("x"))

    def fake_socket(listener, league_id, team_id, swid, token,
                    on_change=None, stop_event=None, on_activity=None):
        # An ESPN id no board row carries -> picks_from_events routes it to
        # `unmapped`, not `drafted`.
        listener.on_frame("SELECTED 1 99999999 2")
        if on_change is not None:
            on_change()

    monkeypatch.setattr("api.live.run_socket_listener", fake_socket)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    resp = client.post("/api/live/connect-token", json={
        "leagueId": "1", "teamId": "2", "swid": "{X}", "token": "t",
        "season": "2026"})
    assert resp.status_code == 200
    assert _wait_until(
        lambda: len(client.get("/api/live/state").json()["unmapped_picks"]) >= 1), \
        "the unmapped pick never surfaced in state"
    um = client.get("/api/live/state").json()["unmapped_picks"]
    assert um[0]["espn_player_id"] == 99999999
    client.post("/api/live/stop")
