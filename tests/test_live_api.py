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


def _seed_minimal_live_db(path):
    """Minimal database for testing build_session.

    Includes all tables required by build_session: base NFL data (weekly,
    schedules, adp, etc.), draft history for fit_all, league settings, and
    draft_order for the current session.
    """
    conn = get_conn(path)

    # Base NFL data
    write_table(conn, "weekly", pd.DataFrame([
        {"player_id": "p1", "player_display_name": "A Star", "position": "WR",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 8, "receiving_yards": 90, "targets": 10, "carries": 0}
        for w in range(1, 18)]))
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "A Star", "position": "WR", "team": "DET", "adp": 5.1}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "snap_counts", pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))
    # A real row, not an empty frame: espn_id has to reach the board for the
    # live crosswalk to resolve a pick, and an empty espn_adp silently yields
    # an empty crosswalk that no other assertion here would notice.
    write_table(conn, "espn_adp", pd.DataFrame([
        {"espn_id": 4429795, "espn_name": "A Star", "position": "WR",
         "team": "DET", "espn_adp": 5.0, "espn_ppr_rank": 5, "espn_proj": 210.0}]))
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
