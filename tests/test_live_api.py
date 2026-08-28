import pathlib
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
def _connects_land_in_the_default_session(monkeypatch):
    """Every connect in this file lands in the DEFAULT session.

    api.live keys live sessions by the `espn_live` cookie and mints a fresh
    sid for a connect that has none, so a TestClient's connect would
    otherwise start a room the `state` dict this file unpacks from
    register_live_routes cannot see. Pinning the mint to DEFAULT_SID keeps
    the cookie (the client carries it on every later request) and the
    unpacked dict pointing at the same room. tests/test_live_sessions.py is
    where real, distinct sids are exercised.
    """
    from api.live import DEFAULT_ROOM_ENV, DEFAULT_SID
    monkeypatch.setenv(DEFAULT_ROOM_ENV, "1")
    monkeypatch.setattr("api.live._mint_sid", lambda: DEFAULT_SID)


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


def test_build_session_survives_a_duplicate_player_id_from_the_adp_feed(tmp_path):
    """The whole-branch consequence of the build_pool duplicate-id crash.

    build_session calls build_board and then build_pool. Task 1 hardened
    build_board against a board row pair sharing a synthesized
    `player_id` (`_add_adp_only_players` keys it on the normalized name
    alone, so one name at two positions collides); build_pool, three lines
    later, still did `board.set_index("player_id")` + `.map()` and raised
    `InvalidIndexError`. So an ADP-feed name collision took the whole
    session build down and no live draft could start -- the Critical was
    fixed at one of two sites on the same call chain.
    """
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    conn = get_conn(path)
    # Neither row matches the weekly universe, so both reach the board as
    # ADP-only players under the one synthesized id `adp_dup_guy`.
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "A Star", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "Dup Guy", "position": "RB", "team": "SF", "adp": 6.0},
        {"adp_name": "Dup Guy", "position": "TE", "team": "GB", "adp": 7.0},
    ]))

    session = build_session(conn, my_slot=1)

    ids = list(session.pool.player_id)
    assert ids.count("adp_dup_guy") == 2, \
        "the collision never reached the pool -- this no longer tests anything"
    # Each colliding row keeps its own position and its own projection.
    dup = [(pos, pts) for pid, pos, pts
           in zip(ids, session.pool.position, session.pool.points)
           if pid == "adp_dup_guy"]
    assert sorted(p for p, _ in dup) == ["RB", "TE"]


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


from api.live import STALE_AFTER_SECONDS, picks_until_turn


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
    # Present with a null value, not omitted -- same convention as every
    # other key on this branch. No room, no report card to link to.
    assert body["report_url"] is None


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


def _settings(teams=8, rounds=15):
    from scoring.league import LeagueSettings
    return LeagueSettings(season=2026, teams=teams,
                          starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
                          flex_slots=2, bench=rounds - 10, scoring={}, draft_type="SNAKE")


def _live_session(seed=DEFAULT_SEED, my_slot=4):
    """A DraftSession usable with the real _recompute: real settings (the
    plan needs teams, rounds and the roster shape), a small board and a
    pool aligned with it. Nothing else is expensive."""
    import types
    board = pd.DataFrame([
        {"player_id": "gibbs", "name": "J Gibbs", "position": "RB", "proj_points": 300.0,
         "espn_rank": 5.0, "espn_pos_rank": 1.0, "espn_adp": 5.5, "market_rank": 4.0,
         "bye": 5, "career_games_pg": 16.0},
        {"player_id": "rb2", "name": "B Back", "position": "RB", "proj_points": 250.0,
         "espn_rank": 8.0, "espn_pos_rank": 2.0, "espn_adp": 9.0, "market_rank": 9.0,
         "bye": 7, "career_games_pg": 15.0},
        {"player_id": "wr1", "name": "W Out", "position": "WR", "proj_points": 280.0,
         "espn_rank": 6.0, "espn_pos_rank": 1.0, "espn_adp": 6.5, "market_rank": 6.0,
         "bye": 5, "career_games_pg": 16.5},
        {"player_id": "wr2", "name": "S Lot", "position": "WR", "proj_points": 240.0,
         "espn_rank": 9.0, "espn_pos_rank": 2.0, "espn_adp": 10.0, "market_rank": 10.0,
         "bye": 9, "career_games_pg": 12.0},
        {"player_id": "te1", "name": "T End", "position": "TE", "proj_points": 200.0,
         "espn_rank": np.nan, "espn_pos_rank": np.nan, "espn_adp": np.nan, "market_rank": 30.0,
         "bye": 11, "career_games_pg": 14.0},
    ])
    pool = types.SimpleNamespace(
        player_id=board["player_id"].to_numpy(dtype=object),
        position=board["position"].to_numpy(dtype=object),
        points=board["proj_points"].to_numpy(dtype=float))
    return dataclasses.replace(_fake_session(seed=seed), my_slot=my_slot,
                               settings=_settings(), board=board, pool=pool)


def _fake_candidate_rows(player_id):
    """A stand-in for the ranked rows the recompute stores: exactly the
    room's candidate contract, one row."""
    return [{"player_id": player_id, "position": "WR", "proj_points": 200.0,
             "espn_rank": 1, "espn_pos_rank": 1, "espn_adp": 1.5,
             "market_rank": 2, "lasts_pct": 80.0, "lasts_at_pick": 9,
             "edge_pts": 12.5, "need": "starter", "favourite": False,
             "rank": 1}]


def _fake_rank_and_plan(player_id):
    """A `rank_and_plan` double: the one row, no plan, no turns."""
    return lambda *a, **k: (_fake_candidate_rows(player_id), [], [])


def _fake_availability(table, player_ids, k, n, espn_adp=None, market_rank=None,
                       positions=None, **_):
    """Everybody is still there: what a corpus with nothing in it answers."""
    return np.ones(len(list(player_ids)), dtype=float)


CANDIDATE_KEYS = {"player_id", "position", "proj_points", "espn_rank",
                  "espn_pos_rank", "espn_adp", "market_rank", "lasts_pct",
                  "lasts_at_pick", "edge_pts", "need", "favourite", "rank"}


def test_state_candidates_carry_the_rooms_contract_in_espn_order(tmp_path):
    """Real engine end to end: a seeded database, a real build_session, a
    real _recompute (the availability table falls back to its ADP curve for
    players the corpus has never seen) and a real GET. Every row carries
    exactly the keys the room reads, in ESPN order with the unranked last,
    and the plan the same recompute drew rides beside them."""
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
    assert body["candidates"], "no ranking produced"
    for row in body["candidates"]:
        assert set(row) == CANDIDATE_KEYS
        assert row["lasts_pct"] is None or 0.0 <= row["lasts_pct"] <= 100.0
        assert row["need"] in {"starter", "flex", "deferred", "bench", "capped"}
        assert row["favourite"] is False
    assert [c["rank"] for c in body["candidates"]] == list(range(1, len(body["candidates"]) + 1))
    ranked = [c["espn_rank"] for c in body["candidates"] if c["espn_rank"] is not None]
    assert ranked == sorted(ranked)
    unranked_at = [i for i, c in enumerate(body["candidates"]) if c["espn_rank"] is None]
    assert unranked_at == list(range(len(body["candidates"]) - len(unranked_at), len(body["candidates"])))
    assert "horizon_pick" not in body and "horizon_is_end_of_draft" not in body
    assert isinstance(body["plan"], list)
    for turn in body["plan"]:
        assert set(turn) == {"pick_no", "round", "target", "alternates"}
    assert body["candidates_as_of_pick"] == 0


def test_state_measures_lasts_against_my_next_turn(tmp_path, monkeypatch):
    """Slot 2 of an 8-team snake with nothing drafted: my next turn is pick
    2, and that is the pick every row's `lasts_at_pick` names. The
    availability answer is what `availability_at` said, as a percentage."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    conn = get_conn(path)
    app = FastAPI()
    state, _recompute = register_live_routes(app, conn, path)
    client = TestClient(app)
    asked = []

    def availability(table, ids, k, n, espn_adp=None, market_rank=None,
                     positions=None, **_):
        asked.append((k, n))
        return np.full(len(list(ids)), 0.4)
    monkeypatch.setattr("api.live.availability_at", availability)
    monkeypatch.setattr("api.live.build_plan", lambda **k: [])
    session = build_session(conn, my_slot=2)
    state["session"] = session
    _recompute(session, picks_made=0)
    body = client.get("/api/live/state").json()
    assert asked[0] == (0, 2)
    assert all(c["lasts_at_pick"] == 2 and c["lasts_pct"] == 40.0 for c in body["candidates"])


def test_after_my_last_turn_there_is_nothing_to_last_to(tmp_path):
    """The seeded league is 8 teams x 16 rounds, so slot 2's last turn is
    pick 127. With 127 picks made there is no turn of mine left: no
    `lasts_pct`, no `lasts_at_pick`, no edge, and an empty plan. The ghost
    rows carry ids not in the pool, which `_drafted_state` keeps as None
    entries -- they still consume their turns, which is all this needs."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from scoring.draft_sim import snake_slots

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    conn = get_conn(path)
    app = FastAPI()
    state, _recompute = register_live_routes(app, conn, path)
    client = TestClient(app)
    session = build_session(conn, my_slot=2)
    slots = snake_slots(session.settings.teams, session.settings.rounds)
    assert len(slots) == 128 and slots[126] == 2, "fixture drifted"
    write_table(conn, "drafted", pd.DataFrame(
        [{"player_id": f"ghost{i}", "pick_no": i} for i in range(1, 128)]))
    state["session"] = session
    _recompute(session, picks_made=127)
    body = client.get("/api/live/state").json()
    assert body["candidates"], "the pool is still on the board"
    assert all(c["lasts_pct"] is None and c["lasts_at_pick"] is None
               and c["edge_pts"] is None for c in body["candidates"])
    assert body["plan"] == []


def test_state_carries_the_plan_key_even_with_no_session(tmp_path):
    """Same "present with a null/empty value, never omitted" convention
    listener_alive and socket_alive already follow on the inactive branch."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    path = str(tmp_path / "live.duckdb")
    conn = get_conn(path)
    app = FastAPI()
    register_live_routes(app, conn, path)
    body = TestClient(app).get("/api/live/state").json()
    assert body["active"] is False
    assert body["plan"] == []
    assert body["candidates"] == []
    assert "horizon_pick" not in body


def test_candidates_are_empty_until_a_ranking_lands_and_then_served_verbatim(tmp_path):
    """No stand-in list. Before the first recompute the room gets [] (and
    draws its ranking state); a ranking that exists is served exactly as
    stored, with the pick it was measured at, never swapped for anything."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    conn = get_conn(path)
    app = FastAPI()
    state, _recompute = register_live_routes(app, conn, path)
    client = TestClient(app)

    session = build_session(conn, my_slot=1)
    state["session"] = session
    body = client.get("/api/live/state").json()
    assert body["my_slot"] == 1
    assert body["candidates"] == [] and body["candidates_as_of_pick"] is None
    assert body["plan"] == []

    state["candidates"] = _fake_candidate_rows("p1")
    state["as_of_pick"] = 7
    state["plan"] = [{"pick_no": 9, "round": 2, "target": None, "alternates": []}]
    body = client.get("/api/live/state").json()
    assert body["candidates"] == _fake_candidate_rows("p1")
    assert body["candidates_as_of_pick"] == 7
    assert body["plan"][0]["pick_no"] == 9


def test_state_carries_the_pick_clock_league_settings_and_my_roster(tmp_path):
    """Task 7b: the rail's three missing feeds. `/api/live/state` must serve
    `ms_remaining` straight off the listener, `settings` off the session's
    real LeagueSettings (not the frontend's old hardcoded 8/15 constants),
    and `my_roster` -- this slot's own picks, replayed with _seed_rosters the
    same way _recompute already does, mapped back to board rows.

    Pick #1 in an 8-team snake belongs to slot 1: p1 goes to me, p2 to slot
    2 (not mine), so my_roster must carry exactly p1's board row rather than
    every drafted player.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pipeline.draft_listener import DraftListener

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path, extra_players=[
        {"player_id": "p2", "name": "B Runner", "position": "RB",
         "team": "DET", "espn_id": 4430807},
    ])
    conn = get_conn(path)
    app = FastAPI()
    state, _recompute = register_live_routes(app, conn, path)
    client = TestClient(app)

    session = build_session(conn, my_slot=1)
    state["session"] = session
    # A listener with no socket/thread wired up -- live_state only reads its
    # ms_remaining attribute, the same field pipeline/draft_listener.py sets
    # from CLOCK/SELECTING frames.
    state["listener"] = DraftListener(session.crosswalk)
    state["listener"].ms_remaining = 23000

    write_table(conn, "drafted", pd.DataFrame([
        {"player_id": "p1", "pick_no": 1},
        {"player_id": "p2", "pick_no": 2},
    ]))

    body = client.get("/api/live/state").json()

    assert body["ms_remaining"] == 23000
    assert body["settings"] == {
        "teams": session.settings.teams,
        "rounds": session.settings.rounds,
        "starters": dict(session.settings.starters),
        "flex_slots": session.settings.flex_slots,
        "bench": session.settings.bench,
        "scoring_format": "half",       # scoring={"receptions": 0.5}
    }
    assert body["my_roster"] == [{
        "player_id": "p1", "name": "A Star", "position": "WR",
        "proj_points": session.board_by_id["p1"]["proj_points"],
        # What the rail actually prints: ESPN's week-1 projection, already in
        # this league's scoring. None on a board built without one -- a dash
        # on the rail, never a zero.
        "wk1_points": (None if pd.isna(session.board_by_id["p1"].get("proj_wk1"))
                       else float(session.board_by_id["p1"].get("proj_wk1"))),
    }]


def test_state_inactive_carries_null_clock_settings_and_empty_roster(tmp_path):
    """The client should never have to branch on whether these keys exist
    (see api/live.py's live_state docstring/comments) -- the inactive
    response carries the same three keys this task adds, with null/empty
    values rather than omitting them."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    body = TestClient(create_app(str(tmp_path / "t.duckdb"))).get("/api/live/state").json()
    assert body["ms_remaining"] is None
    assert body["settings"] == {
        "teams": None, "rounds": None, "starters": {},
        "flex_slots": None, "bench": None, "scoring_format": None,
    }
    assert body["my_roster"] == []


def test_my_roster_replays_picks_in_order_including_a_pool_miss(tmp_path):
    """Direct unit test of the _my_roster helper (not the HTTP layer): slot
    8 of an 8-team snake picks twice in a row, at the turn (overall picks 8
    and 9) -- p2 first, then p3 -- and picks 1-7 (someone else's turns) are
    unattributable (`_seed_rosters` documents a None taken_order entry as
    "drafted, then dropped off the board" -- still consumes a turn, adds
    nothing to any roster). If my_roster silently re-sorted by anything
    other than pick order (player_id, board rank, ...) this would catch it,
    since p2/p3 are deliberately NOT in alphabetical or board order.
    """
    from api.live import _my_roster

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path, extra_players=[
        {"player_id": "p2", "name": "B Runner", "position": "RB",
         "team": "DET", "espn_id": 4430807},
        {"player_id": "p3", "name": "C Slot", "position": "WR",
         "team": "GB", "espn_id": 4426515},
    ])
    conn = get_conn(path)
    session = build_session(conn, my_slot=8)
    pool_index = {pid: i for i, pid in enumerate(session.pool.player_id)}

    taken_order = [None] * 7 + [pool_index["p2"], pool_index["p3"]]
    roster = _my_roster(session, taken_order)

    assert [r["player_id"] for r in roster] == ["p2", "p3"]
    assert roster[0]["position"] == "RB"
    assert roster[1]["position"] == "WR"


def test_my_roster_is_empty_when_my_slot_is_not_known_yet(tmp_path):
    """Honest, not a guess: with no my_slot the socket hasn't named our team
    (see DraftSession.my_slot's own docstring), so there is genuinely no
    roster to attribute anything to."""
    from api.live import _my_roster
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    session = build_session(get_conn(path), my_slot=None)
    assert _my_roster(session, [0]) == []


def _no_roster(*a, **k):
    return ({slot: {"counts": {}, "indices": []} for slot in range(1, 9)}, [])


def test_recompute_discards_a_result_the_pick_count_has_moved_past(tmp_path, monkeypatch):
    """The pick-count guard: a ranking captured at picks_made=3 must not
    overwrite a poll that already recorded picks_made=5 by the time the
    ranking finishes -- a stale recommendation is worse than none."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    session = _live_session()
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (np.zeros(5, bool), []))
    monkeypatch.setattr("api.live._seed_rosters", _no_roster)
    monkeypatch.setattr("api.live.availability_at", _fake_availability)
    monkeypatch.setattr("api.live.build_plan", lambda **k: [])
    state["as_of_pick"] = 5
    state["candidates"] = [{"player_id": "fresh"}]
    _recompute(session, picks_made=3)
    assert state["as_of_pick"] == 5
    assert state["candidates"] == [{"player_id": "fresh"}]


def test_recompute_plans_exactly_my_remaining_turns(tmp_path, monkeypatch):
    """`build_plan` is handed my turns as overall pick numbers, conditioned
    on the picks made. Slot 4 of an 8-team, 15-round snake picks at 4, 13,
    20, ... . With 3 picks made the next one is mine at pick 4, so all 15
    remain; with 4 made it has been used and 14 do -- and `need_kind` gets
    that same count, which is the whole of the deferral rule."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    session = _live_session()
    seen = {}
    monkeypatch.setattr("api.live._drafted_state",
                        lambda cur, pool: (np.zeros(5, bool), [None] * state["_made"]))
    monkeypatch.setattr("api.live._seed_rosters", _no_roster)
    monkeypatch.setattr("api.live.availability_at", _fake_availability)

    def capture(**kwargs):
        seen.update(kwargs)
        return []
    monkeypatch.setattr("api.live.build_plan", capture)
    state["_made"] = 3          # pick 4 is on the clock and it is mine
    _recompute(session, picks_made=3)
    assert seen["turns"][0] == 4 and len(seen["turns"]) == 15
    assert seen["picks_made"] == 3
    state["_made"] = 4          # my pick 4 has been made
    _recompute(session, picks_made=4)
    assert seen["turns"][0] == 13 and len(seen["turns"]) == 14


def test_recompute_stores_its_rows_and_plan_when_nothing_superseded_it(tmp_path, monkeypatch):
    """The guard's other branch: an un-superseded result is served, not
    swallowed by an overzealous check -- rows in ESPN order with the
    unranked tight end last, and the plan `build_plan` returned."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    session = _live_session()
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (np.zeros(5, bool), [None] * 3))
    monkeypatch.setattr("api.live._seed_rosters", _no_roster)
    monkeypatch.setattr("api.live.availability_at", _fake_availability)
    plan = [{"pick_no": 4, "round": 1, "target": {"player_id": "gibbs", "lasts_pct": 100.0,
                                                   "edge_pts": 50.0, "pros": [], "cons": []},
             "alternates": []}]
    monkeypatch.setattr("api.live.build_plan", lambda **k: plan)
    _recompute(session, picks_made=3)
    assert state["as_of_pick"] == 3
    assert state["plan"] == plan
    rows = state["candidates"]
    assert [r["player_id"] for r in rows] == ["gibbs", "wr1", "rb2", "wr2", "te1"]
    assert [r["rank"] for r in rows] == [1, 2, 3, 4, 5]
    assert set(rows[0]) == CANDIDATE_KEYS
    # Pick 4 is mine and on the clock, so "lasts" is measured to my turn
    # after it -- pick 13 -- which is what waiting would mean.
    assert rows[0]["lasts_pct"] == 100.0 and rows[0]["lasts_at_pick"] == 13
    assert rows[-1]["espn_rank"] is None and rows[-1]["market_rank"] == 30


def test_recompute_discards_a_result_from_a_stopped_and_restarted_session(tmp_path, monkeypatch):
    """The generation guard. live_stop then live_start resets as_of_pick to
    None, which blinds the pick-count guard alone. A _recompute launched
    under the session that got stopped must still be discarded, not
    silently overwrite the new session's state."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    old_session = _live_session()
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (np.zeros(5, bool), []))
    monkeypatch.setattr("api.live._seed_rosters", _no_roster)
    monkeypatch.setattr("api.live.availability_at", _fake_availability)

    def restart_underneath(**kwargs):
        state["generation"] += 1
        state["as_of_pick"] = None
        state["candidates"] = [{"player_id": "fresh-session"}]
        return []
    monkeypatch.setattr("api.live.build_plan", restart_underneath)
    _recompute(old_session, picks_made=0)
    assert state["candidates"] == [{"player_id": "fresh-session"}]
    assert state["as_of_pick"] is None


def test_a_player_who_will_not_last_is_not_the_target_at_my_next_turn(tmp_path, monkeypatch):
    """The Gibbs case. Pick 6 of an 8-team snake is slot 6's; four picks
    are in and somebody else is on the clock at 5. Gibbs lasts to my turn
    2% of the time, everybody else 90%: the real plan neither targets him
    nor lists him as an alternate at that turn (eligibility is P >= 0.50),
    and his row says 2.0% at pick 6."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    session = _live_session(my_slot=6)
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (np.zeros(5, bool), [None] * 4))
    monkeypatch.setattr("api.live._seed_rosters", _no_roster)

    def availability(table, ids, k, n, espn_adp=None, market_rank=None,
                     positions=None, **_):
        return np.array([0.02 if pid == "gibbs" else 0.9 for pid in ids])
    # Both seams: the rows read api.live's name, the plan reads its own.
    monkeypatch.setattr("api.live.availability_at", availability)
    monkeypatch.setattr("scoring.plan.availability_at", availability)
    _recompute(session, picks_made=4)
    gibbs = next(r for r in state["candidates"] if r["player_id"] == "gibbs")
    assert gibbs["lasts_pct"] == 2.0 and gibbs["lasts_at_pick"] == 6
    first = state["plan"][0]
    assert first["pick_no"] == 6
    assert first["target"] is not None and first["target"]["player_id"] != "gibbs"
    assert all(a["player_id"] != "gibbs" for a in first["alternates"])


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


def test_slot_from_pick_order_resolves_directly_from_espns_own_order():
    """Defect 1 (post-merge fix): ESPN's draftSettings.pickOrder names the
    slot exactly, with no imported history and no draft round needed --
    pickOrder[k] (0-based) is slot k+1's team id."""
    from api.live import _slot_from_pick_order
    settings = league_mod.LeagueSettings(
        season=2026, teams=8, starters={}, flex_slots=0, bench=0, scoring={},
        draft_type="SNAKE", pick_order=(30, 12, 45, 7, 88, 3, 61, 19))
    assert _slot_from_pick_order(settings, 45) == 3
    assert _slot_from_pick_order(settings, 30) == 1
    assert _slot_from_pick_order(settings, 19) == 8


def test_slot_from_pick_order_falls_through_rather_than_raising_or_guessing():
    """A team id ESPN's own pickOrder does not carry -- tuple.index would
    raise ValueError -- must fall through to None (so the caller tries
    _slot_for_team / _slot_from_socket next) rather than raising into a
    connect or fabricating a slot. Same for no settings at all (the ESPN
    fetch failed) and no team_id yet."""
    from api.live import _slot_from_pick_order
    settings = league_mod.LeagueSettings(
        season=2026, teams=4, starters={}, flex_slots=0, bench=0, scoring={},
        draft_type="SNAKE", pick_order=(30, 12, 45, 7))
    assert _slot_from_pick_order(settings, 999) is None
    assert _slot_from_pick_order(settings, None) is None
    assert _slot_from_pick_order(None, 45) is None
    # No pickOrder published at all (the ordinary case for a league whose
    # settings fetch failed, or from_espn's default of an empty tuple).
    no_order = league_mod.LeagueSettings(
        season=2026, teams=4, starters={}, flex_slots=0, bench=0, scoring={},
        draft_type="SNAKE")
    assert _slot_from_pick_order(no_order, 30) is None


def test_connect_resolves_my_slot_from_espns_pick_order_with_no_frames(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """Defect 1, end to end: the room showed an empty pool at pick 1 of a
    real mock draft because my_slot stayed None until the user's own team
    had picked (or come on the clock) in round 1 -- _slot_for_team has no
    history for a mock's strangers, and _slot_from_socket needs a round-1
    frame that had not arrived yet. ESPN's own pickOrder answers immediately.

    run_listener is stubbed to a true no-op below -- literally zero socket
    frames are ever delivered -- and my_slot still resolves, straight out of
    /api/live/connect's own response, because _provision_and_build now tries
    _slot_from_pick_order before _slot_for_team and before build_session
    even runs. Nothing here seeds draft_teams/draft_order on the provisioned
    league file (provision_league does not copy those -- they are
    LEAGUE_TABLES, see pipeline/db.py), so _slot_for_team is guaranteed to
    fail here too -- this is the mock-draft case exactly.
    """
    from pipeline.espn_league import parse_settings
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)

    espn_payload = {
        "settings": {
            "size": 8,
            "rosterSettings": {"lineupSlotCounts": {
                "0": 1, "2": 2, "4": 2, "6": 1, "16": 1, "17": 1,
                "20": 5, "21": 1, "23": 2}},
            "scoringSettings": {"scoringItems": [
                {"statId": 53, "points": 1.0}]},
            "draftSettings": {"type": "SNAKE",
                              "pickOrder": [3, 7, 1, 2, 4, 5, 6, 8]},
        }}
    raw = parse_settings(espn_payload, 2026)
    monkeypatch.setattr("api.live.fetch_league_settings", lambda *a, **k: raw)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))

    # teamId=2 sits at pickOrder index 3 (0-based) -> slot 4.
    resp = client.post(
        "/api/live/connect",
        json={"url": "https://fantasy.espn.com/football/draft?leagueId=1"
                     "&teamId=2&memberId={X}"})
    assert resp.status_code == 200
    assert resp.json()["my_slot"] == 4
    # Confirmed known before any frame -- the state endpoint agrees with
    # connect's own response, and run_listener never delivered a single one.
    assert client.get("/api/live/state").json()["my_slot"] == 4
    client.post("/api/live/stop")


def _espn_settings_with_pick_order(order=(3, 7, 1, 2, 4, 5, 6, 8)):
    """ESPN's raw league-settings payload, parsed, carrying `order` as its
    draft pickOrder -- the shape `fetch_league_settings` returns on a
    successful fetch, i.e. what _league_settings_from_espn is handed (it is
    league_mod.from_espn that turns this dict into a LeagueSettings)."""
    from pipeline.espn_league import parse_settings
    return parse_settings({
        "settings": {
            "size": 8,
            "rosterSettings": {"lineupSlotCounts": {
                "0": 1, "2": 2, "4": 2, "6": 1, "16": 1, "17": 1,
                "20": 5, "21": 1, "23": 2}},
            "scoringSettings": {"scoringItems": [{"statId": 53, "points": 1.0}]},
            "draftSettings": {"type": "SNAKE", "pickOrder": list(order)},
        }}, 2026)


def test_state_is_never_empty_once_connect_has_resolved_my_slot(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The owner's original complaint, reachable again after two fixes that
    cancelled out. Reproduced exactly as written here before the fix.

    Resolving my_slot at connect (from ESPN's pickOrder) removed the only
    condition under which live_state's vor fallback could run, and nothing
    requested a recompute at launch -- request_recompute fires from on_change
    (a pick landed) and from on_activity gated on _resolve_slot having JUST
    returned True, which cannot happen for a session that already has a slot.
    So `state["candidates"]` stayed at the [] _launch_listener initialises it
    to until the first pick landed, and the room renders an empty list as the
    literal string "No candidates yet.".

    Before the fix, against this exact setup:

        connect (url has teamId=2): 200  my_slot = 4
        GET /api/live/state       : my_slot=4 picks_made=0 len(candidates)=0
        same seed, url WITHOUT teamId: my_slot=None len(candidates)=1

    Both branches are asserted below, in one test, because the defect was
    precisely that the two disagreed. run_listener is a no-op: not one frame
    is ever delivered, so nothing but the connect itself can fill the board.
    """
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)
    monkeypatch.setattr("api.live.fetch_league_settings",
                        lambda *a, **k: _espn_settings_with_pick_order())

    from fastapi.testclient import TestClient
    from api.main import create_app

    base = "https://fantasy.espn.com/football/draft?leagueId=1&memberId={X}"
    for label, url, expect_slot in (("with teamId", base + "&teamId=2", 4),
                                    ("without teamId", base, None)):
        path = str(tmp_path / f"live_{expect_slot}.duckdb")
        _seed_minimal_live_db(path)
        client = TestClient(create_app(path))
        assert client.post("/api/live/connect",
                           json={"url": url}).json()["my_slot"] == expect_slot
        body = client.get("/api/live/state").json()
        assert body["my_slot"] == expect_slot
        assert body["picks_made"] == 0
        if expect_slot is not None:
            # A ranking is requested at launch the moment the slot is known.
            assert _wait_until(
                lambda: client.get("/api/live/state").json()["candidates"]), (
                f"{label}: empty board at pick 0 -- this is the defect")
            assert client.get("/api/live/state").json()["candidates_as_of_pick"] == 0
        else:
            # No slot, no turns: the board is still ranked in ESPN's order,
            # with nothing to "last" to and no plan, until the socket names
            # our team (see _resolve_slot).
            assert _wait_until(
                lambda: client.get("/api/live/state").json()["candidates"]), (
                f"{label}: empty board at pick 0 with no slot")
            body = client.get("/api/live/state").json()
            assert all(c["lasts_pct"] is None and c["lasts_at_pick"] is None
                       and c["edge_pts"] is None and c["need"] is None
                       for c in body["candidates"])
            assert body["plan"] == []
            assert [c["rank"] for c in body["candidates"]] == list(range(1, len(body["candidates"]) + 1))
        client.post("/api/live/stop")


def test_connect_ranks_once_at_launch_instead_of_waiting_for_a_pick(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The vor fallback keeps the board from being EMPTY; this is what makes
    a real, slot-aware ranking actually arrive.

    Nothing used to request one until a pick landed, so connecting at pick 0
    -- or reconnecting mid-draft while already on the clock -- served the
    pool with no ranking at all for the whole of that turn.
    A full recompute measures 0.97-1.49s against the real 249-player pool at
    400 rollouts, so it cannot be done inline on the connect; it is requested
    on the worker, once, before the listener thread starts.

    The signal asserted is a non-None lasts_pct, which only a recompute
    produces (the list is [] until one lands). The real availability table
    and plan run here -- the seeded pool is one player, so the "expensive"
    part is trivial -- because a mocked ranking could not tell the two
    payloads apart.
    """
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)
    monkeypatch.setattr("api.live.fetch_league_settings",
                        lambda *a, **k: _espn_settings_with_pick_order())

    from fastapi.testclient import TestClient
    from api.main import create_app

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    client = TestClient(create_app(path))
    assert client.post(
        "/api/live/connect",
        json={"url": "https://fantasy.espn.com/football/draft?leagueId=1"
                     "&teamId=2&memberId={X}"}).json()["my_slot"] == 4

    assert _wait_until(
        lambda: any(c["lasts_pct"] is not None for c in
                    client.get("/api/live/state").json()["candidates"])), \
        "no ranking was ever requested -- the board waits for the first pick"
    body = client.get("/api/live/state").json()
    assert body["candidates_as_of_pick"] == 0
    assert body["recompute_error"] is None
    client.post("/api/live/stop")


def test_resolve_slot_never_trusts_the_databases_own_pick_order(
        tmp_path, monkeypatch):
    """A stale pick order is a CONFIDENT wrong answer, and nothing
    downstream can detect one (see _slot_for_team's docstring).

    The `league` table holds one row per season, each with that season's real
    pick order, and league_mod.load() takes the newest -- so on the default
    league's own database that is LAST season's shuffle. ESPN team ids are
    stable across seasons, so `pick_order.index(team_id) + 1` answers rather
    than falling through, and it used to win outright over _slot_for_team,
    which reads the CURRENT manually-configured draft_order and is right.

    Trigger, all three parts reachable together and reproduced here: the
    default league (so `load()` sees a real `league` table at all -- a
    provisioned league file has none), a failed ESPN settings fetch (so
    build_session falls back to that table), and no teamId= in the pasted url
    (so the slot is resolved later, from sess.settings, rather than at
    connect from the fetch's own return value). That last one is the natural
    waiting-room url ConnectBody's own docstring names.

    The numbers below are this deployment's real ones: data/nfl.duckdb's
    newest `league` row is 2025 with pick_order [7, 5, 2, 8, 4, 1, 6, 3], and
    against it team 2 resolves to slot 3 while draft_order says slot 2.
    """
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    # Last season's real order, on the default league's own database.
    conn = get_conn(path)
    stale = league_mod.LeagueSettings(
        season=2025, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1},
        flex_slots=1, bench=6, scoring={"receptions": 0.5},
        draft_type="SNAKE", pick_order=(7, 5, 2, 8, 4, 1, 6, 3))
    write_table(conn, "league", pd.DataFrame([
        {"season": 2025, "league_id": "53929318",
         "settings_json": league_mod.to_json(stale)}]))
    conn.close()

    # The autouse fixture already stubs fetch_league_settings to None, which
    # IS the failed-fetch case -- stated again here because it is a
    # precondition of the defect, not incidental.
    monkeypatch.setattr("api.live.fetch_league_settings", lambda *a, **k: None)

    tick = threading.Event()

    def fake_run_listener(listener, url, state_path, on_change=None,
                          headless=False, stop_event=None):
        # The socket names our team; no pick has landed, which is the
        # waiting-room state _resolve_slot exists for.
        listener.my_team_id = 2
        while stop_event is None or not stop_event.is_set():
            if tick.wait(timeout=0.01):
                tick.clear()
                on_change()

    monkeypatch.setattr("api.live.run_listener", fake_run_listener)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    # The default league id, so the session builds against `path` itself and
    # league_mod.load() actually sees the stale row above.
    resp = client.post(
        "/api/live/connect",
        json={"url": "https://fantasy.espn.com/football/draft?leagueId=53929318"})
    assert resp.status_code == 200
    assert resp.json()["my_slot"] is None      # no teamId= in the url

    tick.set()
    assert _wait_until(
        lambda: client.get("/api/live/state").json()["my_slot"] is not None), \
        "_resolve_slot never ran"
    # 2, from draft_teams -> draft_order. NOT 3, which is where team 2 sits
    # in the 2025 pick order sitting in the database.
    assert client.get("/api/live/state").json()["my_slot"] == 2
    client.post("/api/live/stop")


def test_build_session_records_where_its_settings_came_from(tmp_path):
    """The provenance flag the gate above depends on. Nothing else can tell
    the two sources apart after build_session has run: `settings` is a plain
    LeagueSettings either way."""
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    conn = get_conn(path)

    assert build_session(conn, my_slot=1).settings_from_espn is False
    fetched = league_mod.from_espn(_espn_settings_with_pick_order())
    session = build_session(conn, my_slot=1, settings=fetched)
    assert session.settings_from_espn is True
    assert session.settings.pick_order == (3, 7, 1, 2, 4, 5, 6, 8)
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
    real rows to `drafted`, and _recompute. The ranking step is
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
    monkeypatch.setattr("api.live.rank_and_plan", lambda *a, **k: recompute_calls.append(k) or (_fake_candidate_rows("winner"), [], []))

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
    assert state["candidates"] == _fake_candidate_rows("winner")


# --- Listener lifecycle: at most one running, stop really stops it, a dead
# listener is visible, and a superseded one can never write. -----------------
#
# None of this had a test before: connect started a daemon thread and nothing
# else in this file ever exercised what a *second* connect, or a stop, or a
# thread that raises, actually does to that thread. The four tests below
# correspond one to one with the four review criticals.

import threading
import re
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


def test_a_failing_recompute_is_reported_and_does_not_kill_the_worker(
        tmp_path, monkeypatch):
    """A dead recompute worker used to be invisible and permanent.

    `_recompute` ran unguarded inside recompute_worker's loop, which is the
    thread's whole body -- one exception returned from the worker and
    nothing ranked again for the rest of the draft. Candidates and
    as_of_pick froze at whatever pick they last reached, and nothing
    reported it: `listener_alive` tracks the LISTENER thread, which stays
    perfectly healthy while this happens.

    The failure driven here is a real one: `_drafted_state` raises
    ValueError on a drafted row with a null `pick_no`, which live_state
    already wraps in try/except ValueError for exactly that reason. Both
    halves asserted -- the failure reaches /api/live/state as
    `recompute_error`, and the worker is still running afterwards, so the
    next request succeeds and clears it.
    """
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)

    # my_slot has to resolve or _recompute returns before it can fail at
    # all (survival needs a real slot -- see its own docstring).
    monkeypatch.setattr("api.live._slot_for_team", lambda cur, team_id: 1)

    failing = {"on": True}

    def flaky_drafted_state(cur, pool):
        if failing["on"]:
            raise ValueError("2 drafted rows have no pick_no")
        # A real pool-aligned mask, not a bare set(): live_state now feeds
        # this same value to available_by_vor for its fallback list (see
        # test_the_vor_fallback_fires_on_an_empty_list_not_on_an_unknown_slot),
        # and `~np.asarray(set())` is a TypeError. Nothing about this test
        # wanted a fake shape here -- it only ever needed "no picks yet".
        return (np.zeros(len(pool.player_id), dtype=bool), [])

    monkeypatch.setattr("api.live._drafted_state", flaky_drafted_state)
    monkeypatch.setattr("api.live.rank_and_plan", _fake_rank_and_plan("p1"))

    tick = threading.Event()

    def fake_run_listener(listener, url, state_path, on_change=None,
                          headless=False, stop_event=None):
        # my_team_id is what on_activity's _resolve_slot needs to fire; the
        # rest is a pick pump the test drives one beat at a time.
        listener.my_team_id = 1
        while stop_event is None or not stop_event.is_set():
            if tick.wait(timeout=0.01):
                tick.clear()
                on_change()

    monkeypatch.setattr("api.live.run_listener", fake_run_listener)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    assert _connect(client, "1").status_code == 200

    tick.set()
    assert _wait_until(
        lambda: client.get("/api/live/state").json()["recompute_error"] is not None), \
        "a failing recompute never reached /api/live/state"
    body = client.get("/api/live/state").json()
    assert "ValueError" in body["recompute_error"]
    assert "no pick_no" in body["recompute_error"]
    # The listener is untouched -- this failure is invisible in every field
    # that existed before.
    assert body["listener_alive"] is True
    assert body["listener_error"] is None
    # Nothing was ever ranked, so no horizon was ever measured. (Not
    # `candidates_as_of_pick is None`, which this used to assert: live_state
    # now serves the vor-ranked pool whenever the ranked list is empty, and
    # stamps it with the current pick count. That is the point of the
    # fallback -- a broken recompute worker must not leave the room looking
    # at "No candidates yet." -- so as_of_pick no longer distinguishes "a
    # ranking landed" from "a ranking never happened". The horizon does:
    # only _recompute writes it, and the fallback explicitly nulls it.)
    assert body["plan"] == []
    assert body["candidates"] == []

    failing["on"] = False
    tick.set()
    assert _wait_until(
        lambda: any(c["lasts_pct"] is not None for c in
                    client.get("/api/live/state").json()["candidates"])), \
        "the worker died on the first failure instead of surviving it"
    assert client.get("/api/live/state").json()["recompute_error"] is None

    client.post("/api/live/stop")


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

def _connect_token_setup(tmp_path, monkeypatch, *, league_id="1", team_id="2",
                         slot=7, extra_players=None):
    """Shared setup for a bare /api/live/connect-token POST: a minimal live
    database, the named league provisioned with `team_id` resolving to `slot`
    up front (same two-hop draft_teams -> draft_order translation
    test_connect_token_resolves_slot_and_wires_socket_picks below exercises),
    and run_socket_listener faked so no real ESPN socket ever opens.

    `is_free_draft` is forced False: these two tests are about the
    history-import and report-card hooks, both gated on a real (non-mock)
    league id, and the real check can make a network call this suite must
    never depend on.

    Returns (client, body) -- the caller POSTs `body` itself so it can assert
    on the response.
    """
    from pipeline.leagues import provision_league, LEAGUES_ROOT

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path, extra_players=extra_players)

    lg_path = provision_league(league_id, universal_path=path, root=LEAGUES_ROOT)
    lg_conn = get_conn(lg_path)
    write_table(lg_conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "m1", "slot": None},
        {"season": 2025, "team_id": int(team_id), "manager": "m2", "slot": None}]))
    write_table(lg_conn, "draft_order", pd.DataFrame([
        {"slot": 1, "manager": "m1", "is_me": False},
        {"slot": slot, "manager": "m2", "is_me": True}]))
    lg_conn.close()

    monkeypatch.setattr("api.live.billing.is_free_draft", lambda lid: False)
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))
    monkeypatch.setattr("api.live.rank_and_plan", _fake_rank_and_plan("winner"))

    def fake_run_socket_listener(listener, league_id, team_id, swid, token,
                                 on_change=None, stop_event=None,
                                 on_activity=None, on_socket=None):
        # No real socket: the listener thread just parks until stopped. The
        # connect-token hooks under test run on the request thread, before
        # this background thread is even started.
        if stop_event is not None:
            stop_event.wait(timeout=5)

    monkeypatch.setattr("api.live.run_socket_listener", fake_run_socket_listener)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    body = {"leagueId": league_id, "teamId": team_id, "swid": "{X}",
           "token": "1953383334", "season": "2026"}
    return client, body


def _connect_token_with_picks(tmp_path, monkeypatch, *, teams=2, rounds=1,
                              league_id="1", team_id="2", slot=2):
    """Connects a `teams`-team, `rounds`-round league over connect-token, and
    hands back a `feed` the caller drives by hand: one (player_id, pick_no)
    pair per pick, in order. `pick_no` is documentary only -- the real
    pick number a SELECTED frame earns is its 1-based position in the
    stream (see picks_from_events) -- so the pairs must already be in play
    order.

    `rounds` is a computed property (starters + flex_slots + bench: see
    scoring/league.py), so a small league is built by giving it one shallow
    starters entry rather than by setting `rounds` directly. The league
    settings live on the PROVISIONED league file, not the universal one --
    `league` is a LEAGUE_TABLE, so provision_league never copies it, and
    build_session falls back to it via league_mod.load reading that file's
    own (here: freshly written) `league` table.

    `feed` builds a real "SELECTED <team> <espn_id> <slot>" frame per pick,
    the espn id coming from the reverse of the session's own crosswalk, and
    folds each one through the real DraftListener.on_frame + on_change --
    the same path production runs, with only the socket itself faked.
    """
    from pipeline.leagues import provision_league, LEAGUES_ROOT

    path = str(tmp_path / "live.duckdb")
    extra_players = [{"player_id": "p2", "name": "B Runner", "position": "WR",
                      "team": "GB", "espn_id": 4430807}]
    _seed_minimal_live_db(path, extra_players=extra_players)

    setup_conn = get_conn(path)
    crosswalk = build_session(setup_conn, my_slot=1).crosswalk
    setup_conn.close()
    reverse_crosswalk = {pid: espn_id for espn_id, pid in crosswalk.items()}

    lg_path = provision_league(league_id, universal_path=path, root=LEAGUES_ROOT)
    lg_conn = get_conn(lg_path)
    settings = league_mod.LeagueSettings(
        season=2026, teams=teams, starters={"WR": rounds}, flex_slots=0,
        bench=0, scoring={"receptions": 0.5}, draft_type="SNAKE")
    write_table(lg_conn, "league", pd.DataFrame([
        {"season": 2026, "league_id": league_id,
         "settings_json": league_mod.to_json(settings)}]))
    write_table(lg_conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": i, "manager": f"m{i}", "slot": None}
        for i in range(1, teams + 1)]))
    write_table(lg_conn, "draft_order", pd.DataFrame([
        {"slot": i, "manager": f"m{i}", "is_me": i == slot}
        for i in range(1, teams + 1)]))
    lg_conn.close()

    monkeypatch.setattr("api.live.billing.is_free_draft", lambda lid: False)
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (set(), []))
    monkeypatch.setattr("api.live.rank_and_plan", _fake_rank_and_plan("winner"))

    holder = {}
    ready = threading.Event()

    def fake_run_socket_listener(listener, league_id, team_id, swid, token,
                                 on_change=None, stop_event=None,
                                 on_activity=None, on_socket=None):
        holder["listener"] = listener
        holder["on_change"] = on_change
        ready.set()
        if stop_event is not None:
            stop_event.wait(timeout=5)

    monkeypatch.setattr("api.live.run_socket_listener", fake_run_socket_listener)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    body = {"leagueId": league_id, "teamId": team_id, "swid": "{X}",
           "token": "1953383334", "season": "2026"}
    resp = client.post("/api/live/connect-token", json=body)
    assert resp.status_code == 200
    assert ready.wait(timeout=5), "fake socket listener never started"

    def feed(picks):
        listener = holder["listener"]
        on_change = holder["on_change"]
        for player_id, _pick_no in picks:
            espn_id = reverse_crosswalk[player_id]
            listener.on_frame(f"SELECTED 1 {espn_id} 4")
            on_change()

    return client, body, feed


def test_connect_token_spawns_a_history_import(tmp_path, monkeypatch):
    """The connect request is the one moment the account's cookies are in
    hand, so that is when the league's history is fetched -- not at the end
    of the draft, which has no request and no credential."""
    from api import live, reports
    from pipeline import league_history
    spawned = []
    monkeypatch.setattr(league_history, "spawn_import_if_stale",
                        lambda league_id, cookies, **kw: spawned.append((league_id, cookies)) or True)
    monkeypatch.setattr(reports, "cookies_for_connect",
                        lambda request, swid, espn_s2, store=None: {"SWID": swid, "espn_s2": "s2"})
    client, body = _connect_token_setup(tmp_path, monkeypatch)   # the existing test's setup
    resp = client.post("/api/live/connect-token", json=body)
    assert resp.status_code == 200
    assert spawned == [(body["leagueId"], {"SWID": body["swid"], "espn_s2": "s2"})]


def test_state_carries_the_rooms_report_url(tmp_path, monkeypatch):
    """The room is the only way into an auto-built report: by the time it
    exists, `GET /api/espn/drafts` has dropped the finished draft and the
    dashboard card with the other link is gone. The URL comes off the
    connect token, which is the only thing that knows both the league and
    the season."""
    from api import reports
    monkeypatch.setattr(reports, "spawn_draft_complete", lambda *a, **kw: True)
    client, body, feed = _connect_token_with_picks(tmp_path, monkeypatch, teams=2, rounds=1)
    # These fixtures stub `_drafted_state` with a shape the state endpoint's
    # empty-list fallback cannot mask off; stub the fallback itself, which
    # this test says nothing about either way.
    mid = client.get("/api/live/state").json()
    assert mid["report_url"] == "/leagues/1/report/2026"
    assert mid["on_the_clock"] is not None
    feed([("p1", 1), ("p2", 2)])
    # Draft over: `on_the_clock` is null, which is what the rail renders
    # "Draft complete" (and now the link) on.
    done = client.get("/api/live/state").json()
    assert done["on_the_clock"] is None
    assert done["report_url"] == "/leagues/1/report/2026"


def test_a_mock_rooms_state_carries_no_report_url(tmp_path, monkeypatch):
    """A mock draft never builds a report card (`on_draft_complete` refuses
    one), so its room offers no link to a page that would only say so."""
    client, body = _connect_token_setup(tmp_path, monkeypatch)
    monkeypatch.setattr("api.live.billing.is_free_draft", lambda lid: True)
    assert client.post("/api/live/connect-token", json=body).status_code == 200
    assert client.get("/api/live/state").json()["report_url"] is None


def test_last_pick_calls_on_draft_complete(tmp_path, monkeypatch):
    """The `made >= total_picks` branch hands the room to the report job with
    what it needs: league, season, the connecting SWID, the session and the
    drafted rows -- once, and OFF this thread.

    `spawn_draft_complete`, not `on_draft_complete`: this runs on the socket
    read thread, whose own comment two lines below says it must return fast,
    and the hook is not fast (an entitlement read, a DuckDB open, two table
    reads, a frame build, and `billing.is_free_draft`'s cold-cache branch,
    which can make an outbound HTTP call)."""
    from api import live, reports
    calls = []
    monkeypatch.setattr(reports, "spawn_draft_complete",
                        lambda league_id, season, swid, session, drafted, **kw: calls.append(
                            (league_id, season, swid, len(drafted))) or True)
    # Drive a synthetic 2-team, 1-round session to its last pick (see
    # _connect_token_with_picks: two real SELECTED frames through the real
    # DraftListener/on_change/apply_picks path, no captured fixture needed
    # for a draft this small), then assert.
    client, body, feed = _connect_token_with_picks(tmp_path, monkeypatch, teams=2, rounds=1)
    feed([("p1", 1), ("p2", 2)])
    assert calls == [(body["leagueId"], body["season"], body["swid"], 2)]


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
    monkeypatch.setattr("api.live.rank_and_plan", lambda *a, **k: recompute_calls.append(k) or (_fake_candidate_rows("winner"), [], []))

    done = threading.Event()
    seen_args = {}

    def fake_run_socket_listener(listener, league_id, team_id, swid, token,
                                 on_change=None, stop_event=None,
                                 on_activity=None, on_socket=None):
        """Stands in for the real, socket-opening run_socket_listener. Records
        the args it was called with (so the test can prove the token and ids
        reached it) and replays real captured frames, firing on_activity on
        every frame and on_change when the pick count changes -- the same
        contract the real run_socket_listener holds. Accepts (and ignores)
        on_socket: live_connect_token now always passes it, and a fake with a
        narrower signature raises a TypeError the pump() thread swallows into
        listener_error rather than ever finishing (see run_fn's real caller)."""
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
    # NINE picks made, not the four rows this table holds. `drafted` is a
    # sparse record of a dense sequence -- picks 4-8 have no row here, exactly
    # as a pick the crosswalk could not resolve has none in a live draft (see
    # api/live.PICKS_MADE_SQL) -- and the highest pick number is what says how
    # far the draft has got. Reading the row count instead made this endpoint
    # contradict itself: it placed p1's cell at overall 9, in round 2, while
    # simultaneously reporting a draft only four picks old and still in round
    # 1. It cannot be both.
    assert body["picks_made"] == 9
    # 9 picks made -> pick #10 is on the clock: the reversed second round's
    # second pick, slot 7.
    assert body["on_the_clock"] == 7

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
    # year's projected ppg (trending vs last_ppg).
    #
    # proj_ppg is espn_proj RE-PRICED into this league's scoring, not the raw
    # feed number. This fixture's league scores `{"receptions": 0.5}` and
    # nothing else, and A Star's weekly line is 8 rec / 90 rec yds a game:
    # 8x1.0 + 90x0.1 = 17.0 ppg in full PPR against 8x0.5 = 4.0 ppg here, a
    # proj_scale of 4/17 = 0.2353. ESPN's 210.0 becomes 49.4, or 2.9 a game.
    # The raw 210/17 = 12.4 would put a full-PPR projection beside a last_ppg
    # of 4.0 and read as a 3x breakout that is entirely the scoring rules.
    # See scoring/board.projection_scale.
    assert cells[9]["player"]["espn_ppr_rank"] is not None
    assert cells[9]["player"]["last_ppg"] == 4.0
    assert cells[9]["player"]["proj_ppg"] == round(210.0 * (4.0 / 17.0) / 17, 1)
    assert cells[9]["player"]["proj_ppg"] == 2.9

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
    # (taken, taken_order) shaped like _drafted_state's real "nothing
    # drafted yet" return (see its own docstring) -- a bare `set()` used to
    # stand in here and went unexercised (my_slot stays None throughout this
    # test, so _recompute's own call to _drafted_state was always skipped),
    # but /api/live/state's Defect 2 fallback (scoring.gain.available_by_vor)
    # now calls _drafted_state directly whenever my_slot is None, and it
    # needs the real boolean-array shape, not a set.
    monkeypatch.setattr(
        "api.live._drafted_state",
        lambda cur, pool: (np.zeros(len(pool.player_id), dtype=bool), []))
    monkeypatch.setattr("api.live.rank_and_plan", _fake_rank_and_plan("x"))

    def fake_socket(listener, league_id, team_id, swid, token,
                    on_change=None, stop_event=None, on_activity=None,
                    on_socket=None):
        # An ESPN id no board row carries -> picks_from_events routes it to
        # `unmapped`, not `drafted`. on_socket: see fake_run_socket_listener's
        # comment above -- live_connect_token always passes it now.
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


# --- POST /api/live/select: makes the pick ------------------------------
#
# These build the live app directly off register_live_routes (the same
# pieces _live_routes_with_conn already uses), rather than through a real
# /api/live/connect-token, so a session with a known board/socket/listener
# can be assembled without a network fetch or a real ESPN handshake. The
# fake socket's `sent` list and `on_send` hook stand in for
# pipeline.draft_socket.SocketHandle: a real SocketHandle.send writes to a
# live websocket, which these tests must never open.

from pipeline.draft_listener import DraftListener


class _FakeSocket:
    """Stands in for pipeline.draft_socket.SocketHandle. `send` records the
    text and, if wired via `on_send`, calls the hook synchronously -- the
    same turn order a real send-then-ESPN-answers round trip has, letting a
    test answer its own SELECT before live_select's poll loop ever runs
    rather than depending on real wall-clock time to pass."""

    def __init__(self, alive: bool = True):
        self.sent = []
        self._alive = alive
        self._on_send = None
        self.send_error = None

    def on_send(self, fn):
        self._on_send = fn

    def alive(self) -> bool:
        return self._alive

    def detach(self) -> None:
        """What run_socket_listener's `finally` does on every drop, before it
        reconnects -- SocketHandle.detach clears the socket and alive() goes
        false while the listener thread stays perfectly healthy."""
        self._alive = False

    def send(self, text: str) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(text)
        if self._on_send is not None:
            self._on_send(text)


def _select_board():
    """One ordinary player (a real espn_id), one D/ST (espn_id null, resolved
    instead through ESPN_PRO_TEAM_BY_ABBREV/_dst_espn_id), and one player with
    neither -- a genuine crosswalk gap -- so both of _espn_id_for's success
    paths and its failure path are all reachable off one board."""
    return pd.DataFrame([
        {"player_id": "00-0039139", "name": "Test Player", "position": "WR",
         "team": "DET", "espn_id": 4429795},
        {"player_id": "dst_bal", "name": "Ravens D/ST", "position": "DST",
         "team": "BAL", "espn_id": np.nan},
        {"player_id": "no_crosswalk", "name": "Ghost Player", "position": "WR",
         "team": "ZZZ", "espn_id": np.nan},
    ])


def _select_session(my_slot):
    """A DraftSession with a real (small) board/board_by_id and real
    teams/rounds, so picks_until_turn and _espn_id_for both have what they
    need. teams=8 matches snake_slots' round-1 order: slot 1 is the very
    first overall pick, so my_slot=1 is on the clock at picks_made=0 and any
    other slot is not."""
    board = _select_board()
    settings = type("S", (), {"teams": 8, "rounds": 15})()
    return dataclasses.replace(
        _fake_session(), my_slot=my_slot, settings=settings, board=board,
        board_by_id={str(r["player_id"]): r
                     for r in board.to_dict(orient="records")})


def _live_app(tmp_path):
    """A (client, state) pair wired to a throwaway DuckDB file -- the same
    register_live_routes plumbing _live_routes_with_conn uses, plus the
    TestClient test_state_candidates_carry_the_rooms_contract_in_espn_order already established is
    fine to build inline. `state["league_conn"]` is pinned to this test's own
    connection so _drafted_count/_mark_drafted (which only ever see `state`,
    matching the brief's helper signatures) can reach the same `drafted`
    table live_select reads -- production only ever populates league_conn
    this way for a non-default league, but live_select reads
    `state["league_conn"] or conn` either way, so this is a legitimate stand-
    in for "the connection this session's data lives on," not a shortcut
    around it.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    path = str(tmp_path / "live.duckdb")
    conn = get_conn(path)
    app = FastAPI()
    state, _recompute = register_live_routes(app, conn, path)
    state["league_conn"] = conn
    return TestClient(app), state


def _drafted_count(state) -> int:
    cur = state["league_conn"].cursor()
    try:
        return cur.execute("SELECT count(*) FROM drafted").fetchone()[0]
    finally:
        cur.close()


def _mark_drafted(state, player_id: str, pick_no: int = 1) -> None:
    state["league_conn"].execute(
        "INSERT INTO drafted VALUES (?, ?)", [player_id, pick_no])


@pytest.fixture
def live_app_on_socket(tmp_path):
    """A connected socket, but my_slot (4) is not on the clock at
    picks_made=0 -- slot 1 is."""
    client, state = _live_app(tmp_path)
    session = _select_session(my_slot=4)
    ws = _FakeSocket()
    state["session"] = session
    state["socket"] = ws
    state["listener"] = DraftListener(session.crosswalk)
    return client, state, ws


@pytest.fixture
def live_app_on_clock(tmp_path):
    """my_slot=1 -- on the clock at picks_made=0, socket connected."""
    client, state = _live_app(tmp_path)
    session = _select_session(my_slot=1)
    ws = _FakeSocket()
    listener = DraftListener(session.crosswalk)
    state["session"] = session
    state["socket"] = ws
    state["listener"] = listener
    return client, state, ws, listener


@pytest.fixture
def live_app_on_clock_no_socket(tmp_path):
    """On the clock, but no socket session is running -- state["socket"] is
    None, the same as after _stop_listener or before any connect-token."""
    client, state = _live_app(tmp_path)
    session = _select_session(my_slot=1)
    state["session"] = session
    state["socket"] = None
    state["listener"] = DraftListener(session.crosswalk)
    return client, state


def test_select_off_turn_is_rejected_and_sends_nothing(live_app_on_socket):
    """A SELECT off-turn is at best ignored and at worst a misdraft."""
    client, state, ws = live_app_on_socket   # my_slot is NOT on the clock
    body = client.post("/api/live/select", json={"player_id": "p1"})
    assert body.status_code == 409
    assert ws.sent == []


def test_select_sends_the_espn_id_and_waits_for_confirmation(live_app_on_clock):
    client, state, ws, listener = live_app_on_clock
    ws.on_send(lambda text: listener.on_frame("SELECTED 30 4429795 3 {SWID}\n"))
    res = client.post("/api/live/select", json={"player_id": "00-0039139"})
    assert res.status_code == 200
    assert ws.sent == ["SELECT 4429795\n"]
    body = res.json()
    assert body["espn_id"] == 4429795
    assert body["player_id"] == "00-0039139"
    assert body["pick_no"] == 1


def test_select_resolves_a_defense_to_its_negative_espn_id(live_app_on_clock):
    """Board rows for D/ST carry espn_id None; the real id is -(16000+team)."""
    client, state, ws, listener = live_app_on_clock
    ws.on_send(lambda text: listener.on_frame("SELECTED 30 -16033 3 {SWID}\n"))
    res = client.post("/api/live/select", json={"player_id": "dst_bal"})
    assert res.status_code == 200
    assert ws.sent == ["SELECT -16033\n"]
    assert res.json()["espn_id"] == -16033


def test_select_times_out_without_confirmation_and_writes_nothing(
        live_app_on_clock, monkeypatch):
    """The one case the tool cannot resolve. The pick may or may not have
    landed, so the board must not claim either way. SELECT_TIMEOUT_SECONDS/
    SELECT_POLL_SECONDS are shrunk so this test doesn't burn the real 8s."""
    monkeypatch.setattr("api.live.SELECT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("api.live.SELECT_POLL_SECONDS", 0.01)
    client, state, ws, listener = live_app_on_clock
    before = _drafted_count(state)
    res = client.post("/api/live/select", json={"player_id": "00-0039139"})
    assert res.status_code == 504
    assert ws.sent == ["SELECT 4429795\n"]
    assert _drafted_count(state) == before


def test_select_with_no_socket_is_503(live_app_on_clock_no_socket):
    client, state = live_app_on_clock_no_socket
    assert client.post("/api/live/select",
                       json={"player_id": "00-0039139"}).status_code == 503


def test_select_an_already_drafted_player_is_409(live_app_on_clock):
    client, state, ws, listener = live_app_on_clock
    _mark_drafted(state, "00-0039139")
    assert client.post("/api/live/select",
                       json={"player_id": "00-0039139"}).status_code == 409
    assert ws.sent == []


def test_select_send_failure_via_connection_closed_is_503_not_500(live_app_on_clock):
    """SocketHandle.send raises the builtin ConnectionError only when nothing
    is attached. In the race where send grabs the socket reference just
    before the listener thread detaches and closes it, the real
    websockets.sync.client connection raises websockets.exceptions.
    ConnectionClosed instead -- an exception whose MRO (ConnectionClosed,
    WebSocketException, Exception, BaseException, object) does NOT pass
    through ConnectionError. An endpoint that only caught
    (ConnectionError, OSError) would let this leak into an unhandled 500 on
    exactly the failure -- a dropped socket -- it exists to report as a
    clean 503."""
    from websockets.exceptions import ConnectionClosed

    client, state, ws, listener = live_app_on_clock
    ws.send_error = ConnectionClosed(None, None)
    res = client.post("/api/live/select", json={"player_id": "00-0039139"})
    assert res.status_code == 503
    assert ws.sent == []


def test_select_after_the_draft_is_over_is_409_and_sends_nothing(live_app_on_clock):
    """picks_until_turn falls through to 0 ("my turn") once picks_made
    reaches the end of the slot list -- that is "no turns left," not "my
    turn again." Nothing else in the endpoint rules a finished draft out on
    its own: session stays non-None, the socket can still be alive, and a
    real board carries far more players than teams*rounds picks, so
    "00-0039139" (never drafted) still clears the `already` check. Fill every
    one of the session's 8*15 = 120 slots with filler picks and confirm a
    request for that undrafted board player is refused rather than sent."""
    client, state, ws, listener = live_app_on_clock
    total = 8 * 15   # session teams=8, rounds=15 -- see _select_session
    for i in range(total):
        _mark_drafted(state, f"filler-{i}", pick_no=i + 1)
    res = client.post("/api/live/select", json={"player_id": "00-0039139"})
    assert res.status_code == 409
    assert ws.sent == []


def test_select_unknown_player_id_is_400(live_app_on_clock):
    """A player_id the board doesn't carry at all -- a typo, a stale id from
    a rebuilt board -- must be refused before anything is sent, not silently
    resolved to nothing."""
    client, state, ws, listener = live_app_on_clock
    res = client.post("/api/live/select", json={"player_id": "not-on-the-board"})
    assert res.status_code == 400
    assert ws.sent == []


def test_select_with_a_crosswalk_gap_is_400_and_sends_nothing(live_app_on_clock):
    """The safety-relevant 400 the brief calls out: a board row that is
    neither mapped to a real espn_id nor a D/ST _espn_id_for can reconstruct
    one for. This must be refused by name, not sent with a guessed id."""
    client, state, ws, listener = live_app_on_clock
    res = client.post("/api/live/select", json={"player_id": "no_crosswalk"})
    assert res.status_code == 400
    assert ws.sent == []


def test_select_a_player_espn_already_confirmed_is_409_not_a_fake_pick_number(
        live_app_on_clock):
    """A confirmation that never happened.

    `listener.selected_espn_ids` accumulates for the whole session and is
    never cleared -- ESPN replays the draft so far on every JOIN, so it holds
    every SELECTED this socket has ever seen. The wait loop only tests
    membership, so an id already in the set satisfied it on the FIRST
    iteration: HTTP 200 with `pick_no: picks_made + 1`, a number belonging to
    somebody else, for a pick this request never made. The dialog closed
    saying it landed and the user walked away without a player.

    Reachable whenever `drafted` and `selected_espn_ids` disagree, which is
    exactly the unmapped-pick case the rest of api/live.py acknowledges is
    real: ESPN confirmed a player the crosswalk could not resolve, so he is
    in the set and never reached `drafted`. Reproduced here with a listener
    whose crosswalk is empty, so the SELECTED frame lands in the set and
    picks() resolves nothing.
    """
    client, state, ws, listener = live_app_on_clock
    listener.crosswalk = {}
    listener.on_frame("SELECTED 30 4429795 3 {SWID}\n")
    assert 4429795 in listener.selected_espn_ids
    assert _drafted_count(state) == 0     # the crosswalk never resolved him

    res = client.post("/api/live/select", json={"player_id": "00-0039139"})

    assert res.status_code == 409
    assert "already confirmed" in res.json()["detail"]
    # Guarded BEFORE the send: nothing reached ESPN, so this cannot double
    # up on a pick some other team is mid-way through making.
    assert ws.sent == []


def test_state_reports_socket_alive(live_app_on_clock):
    """Spec section 6: the draft buttons disable while the socket is down and
    re-enable when the handle reports alive again. `state["socket"]` already
    gates POST /api/live/select with a 503, but /api/live/state carried no
    equivalent field, so the room derived `isMyTurn` from `on_the_clock ===
    my_slot` alone. During a run_socket_listener reconnect the handle is
    detached while the listener thread is alive and `stale` has not tripped,
    so the buttons stayed enabled, the pill still read ESPN LIVE, and every
    click 503'd.
    """
    client, state, ws, listener = live_app_on_clock
    # _select_session carries a teams/rounds-only settings stand-in and no
    # pool, which is all live_select needs; /api/live/state additionally
    # serves the league shape and replays the drafted rows for my_roster, so
    # give it a real LeagueSettings and an empty pool (`drafted` is empty in
    # this fixture, so nothing is replayed).
    state["session"] = dataclasses.replace(
        state["session"],
        pool=type("P", (), {"player_id": np.array([])})(),
        settings=league_mod.LeagueSettings(
            season=2026, teams=8,
            starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
            flex_slots=1, bench=6, scoring={"receptions": 1.0},
            draft_type="SNAKE"))
    assert client.get("/api/live/state").json()["socket_alive"] is True

    # Exactly what run_socket_listener does in its `finally` on a drop.
    ws.detach()
    body = client.get("/api/live/state").json()
    assert body["socket_alive"] is False
    # The listener itself is untouched -- this is the state the room could
    # not see before, not a dead listener.
    assert body["listener_error"] is None


def test_state_reports_socket_alive_false_with_no_session(tmp_path):
    """Present with a false value on the inactive branch too, never omitted
    -- the same convention listener_alive follows."""
    client, state = _live_app(tmp_path)
    body = client.get("/api/live/state").json()
    assert body["active"] is False
    assert body["socket_alive"] is False
    assert body["recompute_error"] is None


def test_state_reports_draft_started_from_the_listener(live_app_on_clock):
    """Defect 4 (post-merge fix): DraftListener.started, set once by the
    socket's own STATE frame (pipeline/draft_listener.py), reaches
    /api/live/state as draft_started -- so the rail can tell "the draft has
    not started yet" apart from "started, waiting on someone else's pick,"
    which used to render as the identical 'Waiting on the room' heading."""
    client, state, ws, listener = live_app_on_clock
    # live_app_on_clock's session carries no real pool (_fake_session's
    # pool=None) -- fine for /api/live/select, which this fixture exists
    # for, but /api/live/state's my_slot-known branch reads session.pool for
    # my_roster. Same trivial-pool/real-settings patch
    # test_state_reports_socket_alive already applies for the same reason.
    state["session"] = dataclasses.replace(
        state["session"],
        pool=type("P", (), {"player_id": np.array([])})(),
        settings=league_mod.LeagueSettings(
            season=2026, teams=8,
            starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
            flex_slots=1, bench=6, scoring={"receptions": 1.0},
            draft_type="SNAKE"))

    assert client.get("/api/live/state").json()["draft_started"] is False
    listener.on_frame("STATE 1\n")
    assert client.get("/api/live/state").json()["draft_started"] is True


def test_state_draft_started_false_with_no_listener_at_all(tmp_path):
    """A session with no listener wired up at all (state["listener"] stays
    None, exactly _launch_listener's own pre-connect state, or a session
    built via /api/live/start with no socket) reads draft_started False
    rather than raising -- same "present with a false value, never omitted"
    convention as listener_alive/socket_alive above."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    conn = get_conn(path)
    app = FastAPI()
    state, _recompute = register_live_routes(app, conn, path)
    client = TestClient(app)

    state["session"] = build_session(conn, my_slot=None)
    state["listener"] = None
    body = client.get("/api/live/state").json()
    assert body["draft_started"] is False


def test_state_inactive_reports_draft_started_false(tmp_path):
    """Present on the inactive (no session at all) branch too, never
    omitted -- same convention as listener_alive/socket_alive."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    body = TestClient(create_app(str(tmp_path / "t.duckdb"))).get("/api/live/state").json()
    assert body["draft_started"] is False


# --- The connect screen's progress record -------------------------------------
#
# GET /api/live/connect-progress, written stage by stage as the connect runs.
# Its whole reason for existing is that a connect is not fast: measured
# against the real database, 32-35s against the owner's own league (fit_all
# 27.5-30.7s of it) and 8.6s against a fresh mock. All of that used to happen
# behind one spinner. Every assertion below is on a value the connect actually
# discovered -- there is nothing on this endpoint that is not measured or read.

def test_connect_progress_is_idle_before_any_connect(tmp_path):
    """Present with an empty value, never a 404 -- the same convention
    /api/live/state's inactive branch follows, so the screen never has to
    branch on whether the key exists."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    body = TestClient(create_app(str(tmp_path / "t.duckdb"))).get(
        "/api/live/connect-progress").json()
    assert body == {"phase": "idle", "stages": [], "facts": {},
                    "error": None, "elapsed_ms": 0}


def test_connect_records_every_stage_with_the_value_it_discovered(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The screen's whole claim: each line names real work and carries what
    that step found. Asserted against the values, not just the statuses --
    a stage that reports "done" with nothing to show is the spinner this
    replaces."""
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)
    monkeypatch.setattr("api.live.fetch_league_settings",
                        lambda *a, **k: _espn_settings_with_pick_order())
    monkeypatch.setattr("api.live.fetch_team_slots",
                        lambda *a, **k: {1: "Alpha", 2: "Bravo", 3: "Charlie",
                                         4: "Delta", 5: "Echo", 6: "Fox",
                                         7: "Golf", 8: "Hotel"})

    from fastapi.testclient import TestClient
    from api.main import create_app
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    client = TestClient(create_app(path))

    assert client.post(
        "/api/live/connect",
        json={"url": "https://fantasy.espn.com/football/draft?leagueId=1"
                     "&teamId=2&memberId={X}"}).status_code == 200

    body = client.get("/api/live/connect-progress").json()
    stages = {s["key"]: s for s in body["stages"]}
    # No `reset` row: nothing was listening, so there was nothing to stop and
    # the connect must not draw a step it did not take. No `socket` row
    # either -- this is the browser-observer path, which has no handle of its
    # own (see state["socket"]).
    assert [s["key"] for s in body["stages"]] == [
        "token", "settings", "league", "slot", "board", "pool", "history",
        "teams", "ranking"]

    # ESPN's own settings, read off the fetch this connect actually made:
    # 8 teams, receptions 1.0 -> PPR, and 8 starters + 2 flex + 5 bench.
    assert stages["settings"]["value"] == "8 teams · PPR · 15 rounds"
    # teamId=2 sits at index 3 of pickOrder [3,7,1,2,4,5,6,8] -> slot 4.
    assert stages["slot"]["value"] == "you pick 4th of 8"
    assert re.fullmatch(r"\d+ players", stages["board"]["value"])
    assert stages["teams"]["value"] == "8 of 8"
    assert stages["league"]["value"] == "new file · seeded from the shared database"
    assert all(s["status"] == "ok" for s in body["stages"]
               if s["key"] != "ranking"), body["stages"]
    # Every finished stage carries its own real duration.
    assert all(isinstance(s["ms"], int) for s in body["stages"]
               if s["status"] == "ok")

    # The facts the handoff screen draws, all of them read rather than assumed.
    facts = body["facts"]
    assert facts["teams"] == 8 and facts["rounds"] == 15
    assert facts["scoring_format"] == "ppr"
    assert facts["settings_source"] == "espn"
    assert facts["my_slot"] == 4
    assert facts["my_team"] == "Delta"
    assert facts["players"] == int(stages["board"]["value"].split()[0])

    # The ranking lands on the worker, after the connect returned -- so the
    # record keeps being written after the POST is over, which is the point.
    assert _wait_until(
        lambda: client.get("/api/live/connect-progress")
        .json()["phase"] == "ready"), "the connect never settled"
    final = client.get("/api/live/connect-progress").json()
    assert re.fullmatch(r"\d+ ranked", final["stages"][-1]["value"])
    assert final["elapsed_ms"] > 0
    client.post("/api/live/stop")


def test_connect_progress_surfaces_the_silent_settings_fallback(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """_league_settings_from_espn returns None on ANY failure -- no network,
    an unpublished league, a mock that has already been torn down (verified:
    the owner's own mock id 1132152457 now 404s) -- and build_session then
    quietly uses the database's own roster. That roster decides the round
    count and every replacement level, so a draft shaped differently is
    wrong everywhere downstream and nothing told the owner.

    Now it is a warned stage, and the fallback's actual shape is published
    with it, so the screen can say which roster it is about to draft for."""
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)
    monkeypatch.setattr("api.live.fetch_league_settings", lambda *a, **k: None)

    from fastapi.testclient import TestClient
    from api.main import create_app
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    client = TestClient(create_app(path))
    assert _connect(client, "1").status_code == 200

    body = client.get("/api/live/connect-progress").json()
    stages = {s["key"]: s for s in body["stages"]}
    assert stages["settings"]["status"] == "warn"
    assert stages["settings"]["value"] == "unavailable"
    # And WHICH fallback, which is the part that was invisible: league "1" is
    # provisioned fresh, `league` is a LEAGUE_TABLE that provisioning does not
    # copy, so this is not the owner's saved roster at all -- it is the
    # built-in cold-start default (8 teams, full PPR, 15 rounds), a league
    # they have never seen. The seeded database's own half-PPR settings live
    # on the shared file and never reach this session.
    assert body["facts"]["settings_source"] == "default"
    assert body["facts"]["teams"] == 8
    assert body["facts"]["scoring_format"] == "ppr"
    assert body["facts"]["rounds"] == 15
    # Warned, not failed: the connect carried on and the room works.
    assert body["phase"] in ("connecting", "ready")
    assert body["error"] is None
    client.post("/api/live/stop")


def test_a_rejected_connect_names_the_stage_it_died_on(tmp_path):
    """A connect that fails must still surface a real error -- and now it
    also says how far it got. The url below carries no league id, which is
    refused before any listener is touched (see live_connect)."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(str(tmp_path / "t.duckdb")))
    assert client.post("/api/live/connect",
                       json={"url": "https://example.com/nothing"}).status_code == 422

    body = client.get("/api/live/connect-progress").json()
    assert body["phase"] == "failed"
    assert body["error"]["stage"] == "token"
    assert "league id" in body["error"]["detail"]
    # Nothing after the failure claims to have run.
    assert all(s["status"] == "pending" for s in body["stages"]
               if s["key"] != "token")


def test_an_expired_token_fails_the_socket_stage_after_the_connect_returned(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The failure the connect screen exists to name. run_socket_listener
    raises after MAX_EMPTY_RECONNECTS frameless attempts -- the signature of
    an expired draft token -- and it happens on the listener thread, AFTER
    the endpoint has already answered 200. Before this, the only place that
    surfaced was a red pill inside a draft room the user had already been
    handed."""
    def boom(*a, **k):
        raise RuntimeError("could not reconnect to the draft socket -- the "
                           "draft token may have expired")

    monkeypatch.setattr("api.live.run_socket_listener", boom)
    from fastapi.testclient import TestClient
    from api.main import create_app
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    client = TestClient(create_app(path))

    assert client.post("/api/live/connect-token", json={
        "leagueId": "1", "teamId": "2", "swid": "{X}",
        "token": "99", "season": "2026"}).status_code == 200

    assert _wait_until(
        lambda: client.get("/api/live/connect-progress")
        .json()["phase"] == "failed"), "the socket failure never surfaced"
    body = client.get("/api/live/connect-progress").json()
    assert body["error"]["stage"] == "socket"
    assert "token may have expired" in body["error"]["detail"]
    assert "click the ESPN Draft Assist bookmark again" in body["error"]["hint"]
    # Everything it did manage is still on screen, with its values.
    stages = {s["key"]: s for s in body["stages"]}
    assert stages["board"]["status"] == "ok"
    assert stages["token"]["value"] == "team 2 · season 2026"
    client.post("/api/live/stop")


def test_progress_is_readable_while_the_connect_is_still_blocked(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """THE ARCHITECTURAL CLAIM, tested rather than assumed: the connect
    endpoint keeps its synchronous contract (and every existing test of it),
    and the progress is carried by a second request served concurrently.

    Both endpoints are plain `def` handlers, so Starlette runs them in the
    threadpool and one blocking connect does not stop the other from
    answering. Verified separately against a real uvicorn (a 5s blocking sync
    POST, GETs answering in 2-18ms throughout); this pins the same property
    in-process, where the settings fetch is slowed to hold the connect open
    long enough to poll it.
    """
    import threading
    started = threading.Event()
    release = threading.Event()

    def slow_settings(*a, **k):
        started.set()
        release.wait(timeout=5)
        return None

    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)
    monkeypatch.setattr("api.live.fetch_league_settings", slow_settings)

    from fastapi.testclient import TestClient
    from api.main import create_app
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    client = TestClient(create_app(path))

    done = threading.Event()
    thread = threading.Thread(target=lambda: (_connect(client, "1"), done.set()))
    thread.start()
    try:
        assert started.wait(timeout=5), "the connect never reached the fetch"
        # The POST is still blocked here -- and the record already reads.
        body = client.get("/api/live/connect-progress").json()
        assert body["phase"] == "connecting"
        assert not done.is_set(), "the connect finished before it was polled"
        stages = {s["key"]: s for s in body["stages"]}
        assert stages["token"]["status"] == "ok"
        assert stages["settings"]["status"] == "running"
        assert stages["board"]["status"] == "pending"
        assert body["elapsed_ms"] > 0
    finally:
        release.set()
        thread.join(timeout=30)
    assert done.is_set()
    client.post("/api/live/stop")


# --- Joining a draft already in progress ------------------------------------
#
# ESPN replays every prior SELECTED frame the moment a socket JOINs a draft
# that has already started (see pipeline/draft_socket.run_socket_listener and
# espn_live.picks_from_events, whose dedup exists for exactly that replay).
# These drive that replay end to end -- the real run_socket_listener with only
# `_connect` faked, so the frames go through the same recv loop, the same
# on_activity/on_change callbacks, the same apply_picks and the same
# /api/live/state a real connect uses.

def _capture_frames_through(n_selected):
    """Every ws-recv payload of the real capture, cut right after the n-th
    SELECTED -- i.e. what a socket that JOINs after n picks have been made
    has to arrive already knowing."""
    rows = [json.loads(l) for l in _SOCKET_FIXTURE.read_text().splitlines() if l]
    payloads = [str(r.get("payload") or "") for r in rows if r["kind"] == "ws-recv"]
    out, seen = [], 0
    for p in payloads:
        out.append(p)
        if p.strip().startswith("SELECTED "):
            seen += 1
            if seen == n_selected:
                break
    return out


class _ReplaySocket:
    """A socket that hands over a JOIN's whole replay burst back to back and
    then goes quiet, which is what `_connect` returns to run_socket_listener
    on a mid-draft connect. TimeoutError is the real websockets `recv`
    contract for "nothing waiting", which the recv loop treats as a poll tick
    rather than a drop, so the listener stays connected instead of
    reconnecting into a second replay."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent = []

    def recv(self, timeout=None):
        if self._frames:
            return self._frames.pop(0)
        raise TimeoutError

    def send(self, text):
        self.sent.append(text)

    def close(self):
        pass


# `_seed_minimal_live_db`'s own default player already carries this id, which
# is the capture's overall pick 2. Seeding a second board row for it would
# make the crosswalk (a dict keyed by espn_id) silently keep only one of the
# two, so the replay below reuses that row instead of adding a duplicate.
_SEEDED_ESPN_ID = 4429795


def _seed_capture_league(path, root, espn_ids, positions=None):
    """A board carrying one player per given ESPN id, plus league tables that
    put team 2 (the capture's own team -- see its TOKEN frame) in slot 2.

    `espn_ids` is the ordered list of ids the replay will SELECT; an id passed
    as None gets a synthetic espn_id no frame ever names, which is a player
    who is on the board but out of the crosswalk's reach for this draft --
    exactly the shape of a real pick the crosswalk cannot resolve.

    Returns the session's REAL crosswalk (espn id -> board player_id), read
    off the built board rather than assumed from the seed, so the assertions
    below name the player the running code actually resolved.
    """
    positions = positions or ["RB", "WR", "WR", "RB", "WR", "RB", "TE", "QB"]
    extra = [{"player_id": f"x{i}", "name": f"Player {i}",
              "position": positions[i % len(positions)],
              "team": "DET" if i % 2 else "GB",
              "espn_id": e if e is not None else 900000 + i}
             for i, e in enumerate(espn_ids) if e != _SEEDED_ESPN_ID]
    _seed_minimal_live_db(path, extra_players=extra)

    from pipeline.leagues import provision_league
    lg = get_conn(provision_league("1", universal_path=path, root=root))
    write_table(lg, "draft_teams", pd.DataFrame(
        [{"season": 2025, "team_id": t, "manager": f"m{t}", "slot": None}
         for t in range(1, 9)]))
    write_table(lg, "draft_order", pd.DataFrame(
        [{"slot": t, "manager": f"m{t}", "is_me": t == 2} for t in range(1, 9)]))
    lg.close()

    setup = get_conn(path)
    crosswalk = build_session(setup, my_slot=2).crosswalk
    setup.close()
    for e in espn_ids:
        if e is not None:
            assert e in crosswalk, f"fixture espn id {e} did not cross-walk"
    return crosswalk


def _join_mid_draft(monkeypatch, tmp_path, root, n_picks, hole_at=None,
                    make_socket=None):
    """Connect to a draft `n_picks` in, and return (client, by_espn_id).

    `hole_at` is a 1-based overall pick whose player is deliberately absent
    from the board's crosswalk.

    `make_socket(frames) -> socket` swaps in a different stand-in for the
    connection `_connect` returns; the default is `_ReplaySocket`, which
    delivers the burst and then goes quiet. The autodraft tests at the end of
    this file pass one that can also ANSWER a send, which a JOIN replay alone
    never has to do.
    """
    from pipeline import draft_socket

    frames = _capture_frames_through(n_picks)
    selected = [f.strip() for f in frames if f.strip().startswith("SELECTED ")]
    ids = [int(f.split()[2]) for f in selected]
    seeded = list(ids)
    if hole_at is not None:
        seeded[hole_at - 1] = None

    path = str(tmp_path / "live.duckdb")
    by_espn = _seed_capture_league(path, root, seeded)

    make_socket = make_socket or _ReplaySocket
    monkeypatch.setattr(draft_socket, "_connect",
                        lambda url, cookie: make_socket(frames))
    # The ranking is not what these assert; keep it cheap and deterministic so
    # the recompute worker cannot outrun the listener's own writes.
    monkeypatch.setattr("api.live.rank_and_plan", _fake_rank_and_plan("x0"))

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    resp = client.post("/api/live/connect-token", json={
        "leagueId": "1", "teamId": "2", "swid": "{X}",
        "token": "1953383334", "season": "2026"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["my_slot"] == 2
    return client, by_espn, ids


@pytest.mark.skipif(not _SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_joining_mid_draft_shows_the_owners_existing_roster(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The owner's report: "if i join a draft halfway through it doesnt show
    my previous picks".

    The capture is an 8-team snake whose own team is 2 (its TOKEN frame), so
    overall picks 2, 15 and 18 are the owner's. Joining after 23 picks must
    show all three, in pick order, and count 23 picks made.
    """
    client, by_espn, ids = _join_mid_draft(
        monkeypatch, tmp_path, _isolated_leagues_root, 23)
    try:
        assert _wait_until(
            lambda: client.get("/api/live/state").json()["picks_made"] == 23), \
            "the JOIN replay never finished landing in `drafted`"
        body = client.get("/api/live/state").json()
        assert body["unmapped_picks"] == []
        assert [p["player_id"] for p in body["my_roster"]] == [
            by_espn[ids[1]], by_espn[ids[14]], by_espn[ids[17]]]
        # Pick 24 is the last of round 3 in an 8-team snake -> slot 8.
        assert body["on_the_clock"] == 8
        cells = client.get("/api/live/board").json()["cells"]
        assert sorted(c["overall"] for c in cells) == list(range(1, 24))
    finally:
        client.post("/api/live/stop")


@pytest.mark.skipif(not _SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_a_pick_the_crosswalk_cannot_map_does_not_shift_every_later_pick(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """THE DEFECT behind the report, and the reason it showed up on a
    mid-draft join rather than a draft watched from pick 1.

    A pick whose player the crosswalk cannot reach is reported in
    `unmapped_picks` and never written to `drafted` -- so `drafted.pick_no`
    has a HOLE in it. `_drafted_state` used to build `taken_order` from the
    ROWS of that table, one entry per row, and `_seed_rosters` walks it by
    position: entry i is the pick that was made i-th. One missing row
    therefore pulled every later pick one slot to the left, and every roster
    from that point on belonged to the wrong team.

    Watching from pick 1 the damage is one cell at a time and self-evident.
    On a mid-draft JOIN the whole replay lands at once, so a single unmapped
    pick in round 1 silently re-attributes every round after it -- the
    owner's own picks land in a neighbour's column and MY ROSTER shows
    players they never drafted.

    Same capture and same slot as the test above, with overall pick 1's
    player off the board. The owner's picks (2, 15, 18) must still be the
    owner's; the count of picks made must still be 23, not 22, or the room
    names the wrong slot on the clock.
    """
    client, by_espn, ids = _join_mid_draft(
        monkeypatch, tmp_path, _isolated_leagues_root, 23, hole_at=1)
    try:
        assert _wait_until(
            lambda: client.get("/api/live/state").json()["unmapped_picks"]), \
            "the unmapped pick was never surfaced"
        assert _wait_until(
            lambda: client.get("/api/live/state").json()["picks_made"] == 23), \
            "picks_made counted rows in `drafted`, not picks actually made"
        body = client.get("/api/live/state").json()
        # Honestly reported rather than silently dropped -- this player IS
        # gone and the board cannot know which of its rows he is.
        assert body["unmapped_picks"] == [
            {"espn_player_id": ids[0], "overall_pick": 1}]
        # The whole point: attribution is by overall pick number, so the hole
        # at pick 1 costs pick 1 and nothing else.
        assert [p["player_id"] for p in body["my_roster"]] == [
            by_espn[ids[1]], by_espn[ids[14]], by_espn[ids[17]]]
        assert body["on_the_clock"] == 8
        cells = client.get("/api/live/board").json()["cells"]
        # Every pick but the unmapped one, each in its own true position.
        assert sorted(c["overall"] for c in cells) == list(range(2, 24))
    finally:
        client.post("/api/live/stop")


# --- ESPN's autodraft: the state, and the way back out of it ---------------
#
# Miss a pick and ESPN flips your team onto autodraft and starts making the
# picks itself. Nothing in its UI tells this tool that happened; the socket
# does, and these drive both directions of it.
#
# The frames are real. tests/fixtures/espn_autodraft.jsonl is three records
# lifted VERBATIM out of data/draft_room_trace.jsonl -- lines 1112, 1186 and
# 1188: ESPN flipping team 2 on the moment its clock ran out, the real ESPN
# client turning it back off, and ESPN's echo two frames later. That trace is
# not git-tracked, which is why the three frames live in a fixture instead;
# they are the same captured session as tests/fixtures/espn_draft_socket.jsonl
# (identical socket URL, and every record of that fixture appears in the
# trace), so replaying the two together is one continuous draft, not a splice
# of two.
#
# Both tests run the REAL run_socket_listener with only `_connect` faked, the
# same as the mid-draft-join tests above: the frames go through the same recv
# loop, the same DraftListener, the same POST /api/live/autodraft and the same
# /api/live/state a live draft night uses.

_AUTODRAFT_FIXTURE = Path("tests/fixtures/espn_autodraft.jsonl")


def _autodraft_frames():
    """(espn_flips_us_on, what_the_client_sends, espn_confirms_off) -- the
    three captured payloads in the order they happened. The kinds are
    asserted rather than assumed: the middle one is the only ws-send of the
    three, and reading it as an inbound frame would quietly turn this whole
    section into a test of nothing."""
    rows = [json.loads(l) for l in _AUTODRAFT_FIXTURE.read_text().splitlines() if l]
    kinds = [r["kind"] for r in rows]
    assert kinds == ["ws-recv", "ws-send", "ws-recv"], kinds
    return tuple(str(r.get("payload") or "") for r in rows)


class _AutodraftSocket:
    """A `_ReplaySocket` that can also answer back.

    Delivers the JOIN replay burst and then goes quiet (TimeoutError, the
    real `websockets` contract for "nothing waiting"), exactly like
    _ReplaySocket. Two additions:

      - `flip()`, called from the test, queues ESPN's unprompted `AUTODRAFT 2
        true` -- the frame a missed pick produces, arriving with nothing sent
        to provoke it.
      - a send is echoed ONLY when it matches the capture's own outbound
        frame byte for byte. That is deliberate and is what makes the round
        trip a real pin on the wire format: an endpoint that sent
        `AUTODRAFT 0`, or included the team id ESPN's inbound frames carry,
        would get no echo and time out rather than quietly passing.

    Locked because `send` runs on whatever thread FastAPI hands the request
    while `recv` runs on the listener thread -- the same split
    pipeline.draft_socket.SocketHandle exists for.
    """

    def __init__(self, frames, flip_frame, outbound, echo_frame):
        self._lock = threading.Lock()
        self._frames = list(frames)
        self._queued = []
        self._flip_frame = flip_frame
        self._outbound = outbound
        self._echo_frame = echo_frame
        self.sent = []

    def flip(self):
        """ESPN putting us on autodraft with nothing sent to ask for it."""
        with self._lock:
            self._queued.append(self._flip_frame)

    def recv(self, timeout=None):
        with self._lock:
            if self._frames:
                return self._frames.pop(0)
            if self._queued:
                return self._queued.pop(0)
        raise TimeoutError

    def send(self, text):
        with self._lock:
            self.sent.append(text)
            if text == self._outbound:
                self._queued.append(self._echo_frame)

    def close(self):
        pass


def _autodraft_of(client):
    return client.get("/api/live/state").json()["autodraft"]


def _make_state_readable(state):
    """live_app_on_clock's session is built for /api/live/select, which needs
    only teams/rounds and a board -- /api/live/state additionally serves the
    league shape and replays the drafted rows for my_roster, so it needs a
    real LeagueSettings and a pool. Same trivial-pool/real-settings patch
    test_state_reports_socket_alive and test_state_reports_draft_started_from
    _the_listener each already apply inline, for the same reason (`drafted` is
    empty in this fixture, so nothing is replayed)."""
    state["session"] = dataclasses.replace(
        state["session"],
        pool=type("P", (), {"player_id": np.array([])})(),
        settings=league_mod.LeagueSettings(
            season=2026, teams=8,
            starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
            flex_slots=1, bench=6, scoring={"receptions": 1.0},
            draft_type="SNAKE"))


@pytest.mark.skipif(not _SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_espn_flipping_us_onto_autodraft_is_seen_and_can_be_turned_back_off(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The whole round trip, in the order draft night produces it.

    1. Connect mid-draft. The capture's own first frame is `AUTODRAFT 2
       false`, so the room starts out correctly saying autodraft is off.
    2. ESPN flips us on, unprompted (the captured frame from the moment team
       2's clock ran out). /api/live/state has to notice with nothing having
       asked it to.
    3. POST /api/live/autodraft {"on": false} sends the capture's own
       outbound frame and returns only once ESPN's echo names OUR team --
       and the state then reads off that echo, not off the request.
    """
    flip_on, outbound, echo_off = _autodraft_frames()

    sockets = []

    def make_socket(frames):
        sock = _AutodraftSocket(frames, flip_on, outbound, echo_off)
        sockets.append(sock)
        return sock

    client, _by_espn, _ids = _join_mid_draft(
        monkeypatch, tmp_path, _isolated_leagues_root, 3,
        make_socket=make_socket)
    try:
        # Off to start with, and known to be off -- not merely "not yet
        # heard of". The capture opens with ESPN stating it.
        assert _wait_until(lambda: _autodraft_of(client) is False), \
            "the JOIN replay's own AUTODRAFT frame never reached the state"

        sockets[0].flip()
        assert _wait_until(lambda: _autodraft_of(client) is True), \
            "ESPN put us on autodraft and the room never noticed"

        res = client.post("/api/live/autodraft", json={"on": False})
        assert res.status_code == 200, res.text
        assert res.json() == {"autodraft": False, "changed": True}
        # Byte for byte the frame ESPN's own client sent (trace line 1186):
        # no team id, because the socket already identifies the team.
        assert sockets[0].sent == [outbound]
        # And the state follows ESPN's echo, which is the only reason the
        # call above returned at all.
        assert _autodraft_of(client) is False
    finally:
        client.post("/api/live/stop")


@pytest.mark.skipif(not _SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_joining_a_draft_while_already_on_autodraft_learns_it_from_the_replay(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """Connecting mid-draft to a team ESPN is ALREADY drafting for.

    This is the case a naive implementation gets wrong in the most dangerous
    direction -- silently reporting "autodraft off" for a session whose picks
    ESPN is making. Two things have to hold for it to work, and both are
    properties of the real capture rather than of the code:

      - ESPN states autodraft in the JOIN replay at all (it does: the
        capture's very first frame, before INIT, before anything).
      - it states it BEFORE the TOKEN frame that names our own team (it does:
        trace line 503 vs line 511). So the listener cannot decide "is this
        me?" as the frame arrives; it has to keep the flag per team id and
        resolve ours later.

    The replay here is the capture's own, with its leading `AUTODRAFT 2
    false` swapped for the captured `AUTODRAFT 2 true` -- the same frame, the
    same team, from the same session, moved to the position a draft already
    on autodraft would put it in. Nothing else about the burst changes: the
    SELECTED frames, and so the picks and the crosswalk, are untouched.
    """
    flip_on, outbound, echo_off = _autodraft_frames()
    off_frame = echo_off      # `AUTODRAFT 2 false`, what the replay carries

    swapped = []

    def make_socket(frames):
        replay = list(frames)
        assert replay[0] == off_frame, \
            f"the capture no longer opens with {off_frame!r}: {replay[0]!r}"
        replay[0] = flip_on
        swapped.append(True)
        return _AutodraftSocket(replay, flip_on, outbound, echo_off)

    client, _by_espn, _ids = _join_mid_draft(
        monkeypatch, tmp_path, _isolated_leagues_root, 3,
        make_socket=make_socket)
    try:
        assert swapped, "the replay socket was never built"
        assert _wait_until(lambda: _autodraft_of(client) is True), \
            "a connect that JOINed an already-autodrafting team reported " \
            f"autodraft={_autodraft_of(client)!r}"
        # Not a fluke of ordering: the picks landed too, so this is the real
        # replay path and not a listener that stopped at the first frame.
        assert _wait_until(
            lambda: client.get("/api/live/state").json()["picks_made"] == 3)
    finally:
        client.post("/api/live/stop")


# --- POST /api/live/autodraft: the guards ---------------------------------
#
# Same fixtures the SELECT tests use (see _live_app / live_app_on_clock), for
# the failure states that need no capture: they are about this session's own
# shape, not about anything ESPN sent.


def test_autodraft_with_no_session_is_409(tmp_path):
    client, _state = _live_app(tmp_path)
    res = client.post("/api/live/autodraft", json={"on": False})
    assert res.status_code == 409
    assert "no live draft session" in res.json()["detail"]


def test_autodraft_with_no_socket_is_503(live_app_on_clock_no_socket):
    """The browser-observer path (/api/live/connect) publishes no
    SocketHandle at all, and a run_socket_listener reconnect detaches the one
    it has -- neither can send, and both must say so rather than time out."""
    client, state = live_app_on_clock_no_socket
    state["listener"].on_frame("TOKEN 1:1:30:{X}:1\n")
    res = client.post("/api/live/autodraft", json={"on": False})
    assert res.status_code == 503


def test_autodraft_before_the_socket_names_our_team_is_refused(live_app_on_clock):
    """ESPN broadcasts AUTODRAFT for every team in the room, so without our
    own team id there is nothing to match an echo against -- confirming on a
    neighbour's frame would report a change that never happened to us.
    Refused rather than guessed."""
    client, _state, ws, listener = live_app_on_clock
    assert listener.my_team_id is None
    res = client.post("/api/live/autodraft", json={"on": True})
    assert res.status_code == 409
    assert "ESPN team" in res.json()["detail"]
    assert ws.sent == []


def test_autodraft_returns_only_on_espns_echo_for_our_own_team(live_app_on_clock):
    """A neighbour flipping must not confirm our request. Team 3's frame is
    fed first and has to be ignored; only team 30's -- ours, per the TOKEN
    below -- answers the request."""
    client, state, ws, listener = live_app_on_clock
    _make_state_readable(state)
    listener.on_frame("TOKEN 1:1:30:{X}:1\n")
    ws.on_send(lambda text: (listener.on_frame("AUTODRAFT 3 true\n"),
                             listener.on_frame("AUTODRAFT 30 true\n")))
    res = client.post("/api/live/autodraft", json={"on": True})
    assert res.status_code == 200
    assert ws.sent == ["AUTODRAFT true\n"]
    assert res.json() == {"autodraft": True, "changed": True}
    assert client.get("/api/live/state").json()["autodraft"] is True


def test_autodraft_times_out_without_confirmation_and_changes_nothing(
        live_app_on_clock, monkeypatch):
    """No echo means no claim. The endpoint must not report success, and
    nothing may write the flag on the request's behalf -- only ESPN's own
    frame ever sets it."""
    import api.live
    monkeypatch.setattr(api.live, "AUTODRAFT_TIMEOUT_SECONDS", 0.2)
    client, state, ws, listener = live_app_on_clock
    _make_state_readable(state)
    listener.on_frame("TOKEN 1:1:30:{X}:1\n")
    listener.on_frame("AUTODRAFT 30 true\n")
    res = client.post("/api/live/autodraft", json={"on": False})
    assert res.status_code == 504
    assert "still on" in res.json()["detail"]
    assert ws.sent == ["AUTODRAFT false\n"]
    # Unchanged, and still ESPN's last word rather than the request's.
    assert listener.my_autodraft is True
    assert client.get("/api/live/state").json()["autodraft"] is True


def test_autodraft_already_in_the_requested_state_sends_nothing(live_app_on_clock):
    """ESPN has not been observed re-broadcasting a flag that did not change,
    so a redundant send would sit out the whole timeout and then report
    failure for a state that is already exactly what was asked for. The
    honest answer is the state itself, and `changed: false` to say no command
    was sent."""
    client, _state, ws, listener = live_app_on_clock
    listener.on_frame("TOKEN 1:1:30:{X}:1\n")
    listener.on_frame("AUTODRAFT 30 true\n")
    res = client.post("/api/live/autodraft", json={"on": True})
    assert res.status_code == 200
    assert res.json() == {"autodraft": True, "changed": False}
    assert ws.sent == []


def test_autodraft_send_failure_via_connection_closed_is_503_not_500(
        live_app_on_clock):
    """The same race live_select documents: this request reads the socket
    reference just before the listener thread detaches and closes it, and the
    real connection underneath raises ConnectionClosed -- which is NOT a
    subclass of ConnectionError. Catching only the builtin would turn a
    dropped socket into a 500."""
    from websockets.exceptions import ConnectionClosed

    client, _state, ws, listener = live_app_on_clock
    listener.on_frame("TOKEN 1:1:30:{X}:1\n")
    ws.send_error = ConnectionClosed(None, None)
    res = client.post("/api/live/autodraft", json={"on": True})
    assert res.status_code == 503
    assert "could not reach ESPN" in res.json()["detail"]


def test_state_reports_autodraft_null_when_espn_has_not_said(live_app_on_clock):
    """Null, not false. "We have not been told" and "ESPN says it is off" are
    different facts, and defaulting the first to the second would hide
    exactly the state this feature exists to surface."""
    client, state, _ws, _listener = live_app_on_clock
    _make_state_readable(state)
    assert client.get("/api/live/state").json()["autodraft"] is None


def test_inactive_state_carries_the_autodraft_key(tmp_path):
    """Present with a null value, never omitted -- the same convention
    listener_alive/socket_alive already follow on this branch."""
    client, _state = _live_app(tmp_path)
    body = client.get("/api/live/state").json()
    assert body["active"] is False
    assert "autodraft" in body and body["autodraft"] is None


# --- Surviving a restart: the saved session record ---------------------------
#
# An API restart mid-draft used to lose the whole live session -- the socket,
# the listener, the resolved slot and the connect parameters -- and the only
# way back was to go to the ESPN tab and click the bookmarklet again, on a
# thirty-second clock. `drafted` was always durable (it is written to the
# league's own file); what was lost was the session's identity. These pin the
# record that closes that gap, and, just as importantly, what it does NOT
# hold and when it is deleted.

def _record_path(path):
    from api.live import session_record_path
    return session_record_path(path)


def test_the_session_record_round_trips_and_is_readable_only_by_its_owner(tmp_path):
    """The record carries the five connect fields plus the resolved slot,
    and nothing wider: no espn_s2 (it never reaches this process at all --
    see TokenBody's docstring), and mode 0600, because a file holding a
    token that can send SELECT on somebody's live draft must not be readable
    by other accounts on the machine."""
    import os
    from api.live import load_session_record, save_session_record

    db = str(tmp_path / "nfl.duckdb")
    written = save_session_record(db, league_id="1", team_id="2", season="2026",
                                  swid="{X}", token="1953383334", my_slot=7)
    assert written == _record_path(db)
    assert oct(os.stat(written).st_mode & 0o777) == oct(0o600)

    record = load_session_record(db)
    assert record["league_id"] == "1"
    assert record["team_id"] == "2"
    assert record["season"] == "2026"
    assert record["swid"] == "{X}"
    assert record["token"] == "1953383334"
    assert record["my_slot"] == 7
    # Exactly these keys. A record that grows a field is a decision to widen
    # what sits on disk, and it should have to change this line to do it.
    assert set(record) == {"version", "saved_at", "league_id", "team_id",
                           "season", "swid", "token", "my_slot"}
    assert "espn_s2" not in json.dumps(record)


def test_the_record_path_follows_the_database_the_app_was_opened_with(tmp_path):
    """DRAFT_DB_PATH is how a second instance runs against a copy (the whole
    test suite does it). A fixed record path would have a scratch server
    restore the real draft, or delete its token."""
    from api.live import session_record_path
    a = session_record_path(str(tmp_path / "real.duckdb"))
    b = session_record_path(str(tmp_path / "scratch.duckdb"))
    assert a != b
    assert a.startswith(str(tmp_path / "real.duckdb"))


def test_a_record_too_old_to_carry_a_live_token_is_ignored_and_deleted(tmp_path):
    """ESPN's draftSecurity token dies with its draft. A record older than
    the age bound describes a draft that is over, so restoring it would
    spend up to 34.8s rebuilding a board for a token ESPN will refuse -- and
    leaving it on disk would leave the token there for ever. Deleting on
    read is the only automatic bound on that, for a draft that was never
    stopped cleanly."""
    import os
    from datetime import timedelta
    from api.live import (SESSION_RECORD_MAX_AGE_SECONDS, load_session_record,
                          save_session_record)
    from datetime import datetime, timezone

    db = str(tmp_path / "nfl.duckdb")
    save_session_record(db, league_id="1", team_id="2", season="2026",
                        swid="{X}", token="tok", my_slot=7)
    just_inside = (datetime.now(timezone.utc)
                   + timedelta(seconds=SESSION_RECORD_MAX_AGE_SECONDS - 60))
    assert load_session_record(db, now=just_inside) is not None
    assert os.path.exists(_record_path(db))

    past = (datetime.now(timezone.utc)
            + timedelta(seconds=SESSION_RECORD_MAX_AGE_SECONDS + 60))
    assert load_session_record(db, now=past) is None
    assert not os.path.exists(_record_path(db)), \
        "an expired token was left on disk"


def test_a_damaged_or_incomplete_record_is_no_session_rather_than_a_crash(tmp_path):
    """A restore is a convenience; the API has to come up either way. Every
    unusable record reads as "no saved session", never an exception out of
    create_app."""
    from api.live import load_session_record, save_session_record

    db = str(tmp_path / "nfl.duckdb")
    assert load_session_record(db) is None            # nothing saved at all

    Path(_record_path(db)).write_text("{not json")
    assert load_session_record(db) is None

    Path(_record_path(db)).write_text(json.dumps(
        {"version": 999, "league_id": "1", "team_id": "2", "swid": "{X}",
         "token": "t", "saved_at": "2026-08-19T00:00:00+00:00"}))
    assert load_session_record(db) is None            # a version we don't know

    save_session_record(db, league_id="1", team_id="notanumber", season="2026",
                        swid="{X}", token="t")
    assert load_session_record(db) is None, \
        "a non-numeric team id must be refused here, as it is on connect"


def _seed_league_one_with_slot_seven(path, leagues_root):
    """League 1's own file, with team 2 drafting from slot 7 (deliberately
    not slot 2). draft_teams/draft_order are LEAGUE_TABLES, so they have to
    live on the league's file, not on the shared one."""
    from pipeline.leagues import provision_league
    lg_path = provision_league("1", universal_path=path, root=leagues_root)
    lg = get_conn(lg_path)
    write_table(lg, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "m1", "slot": None},
        {"season": 2025, "team_id": 2, "manager": "m2", "slot": None}]))
    write_table(lg, "draft_order", pd.DataFrame([
        {"slot": 1, "manager": "m1", "is_me": False},
        {"slot": 7, "manager": "m2", "is_me": True}]))
    lg.close()
    return lg_path


def _fake_socket(seen, stops, frames=()):
    """A run_socket_listener that opens, replays `frames`, and then STAYS
    connected until its stop_event is set -- the same shape the real one
    has, which is what makes listener_alive/socket_alive mean anything in
    these tests. Records the ids and token it was handed, so a restore can
    be checked to have reconnected to the same draft with the same token.
    """
    def fake(listener, league_id, team_id, swid, token, on_change=None,
             stop_event=None, on_activity=None, on_socket=None):
        seen.append({"league_id": league_id, "team_id": team_id,
                     "swid": swid, "token": token})
        stops.append(stop_event)
        if on_socket is not None:
            class _Handle:
                def alive(self):
                    return True

                def send(self, text):
                    pass
            on_socket(_Handle())
        prev = 0
        for frame in frames:
            listener.on_frame(frame)
            if on_activity is not None:
                on_activity()
            count = len(listener.picks().rows)
            if count != prev:
                prev = count
                if on_change is not None:
                    on_change()
        if stop_event is not None:
            stop_event.wait(timeout=10)
    return fake


@pytest.mark.skipif(not _SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_connect_token_saves_the_session_and_stop_deletes_it(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The bookmarklet connect is the only path that has a token to save,
    and an explicit stop is the user saying the draft is over for them -- so
    the token must not outlive it on disk."""
    import os
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    _seed_league_one_with_slot_seven(path, _isolated_leagues_root)

    seen, stops = [], []
    monkeypatch.setattr("api.live.run_socket_listener", _fake_socket(seen, stops))

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    try:
        resp = client.post("/api/live/connect-token", json={
            "leagueId": "1", "teamId": "2", "swid": "{X}",
            "token": "1953383334", "season": "2026"})
        assert resp.status_code == 200

        from api.live import load_session_record
        record = load_session_record(path)
        assert record is not None
        assert (record["league_id"], record["team_id"], record["swid"],
                record["token"], record["season"]) == \
            ("1", "2", "{X}", "1953383334", "2026")
        # The slot connect resolved, saved with it -- so a restart does not
        # depend on relearning it from a socket that may be dead.
        assert record["my_slot"] == 7
    finally:
        client.post("/api/live/stop")
        for ev in stops:
            if ev is not None:
                ev.set()
    assert not os.path.exists(_record_path(path)), \
        "stopping the session left the draft token on disk"


@pytest.mark.skipif(not _SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_a_second_app_on_the_same_database_restores_the_session_by_itself(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The whole point. A new app object over the same database -- what a
    restarted process gets -- must come back to the same league, team, slot
    and picks with nobody clicking anything, and must reopen the socket with
    the same token.

    This is the in-process version; the real one (uvicorn started, killed
    and started again) is in the task report, because a test that constructs
    a second app is not by itself proof that a process restart works.
    """
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path, extra_players=[
        {"player_id": "p2", "name": "B Runner", "position": "RB",
         "team": "DET", "espn_id": 4430807},
        {"player_id": "p3", "name": "C Slot", "position": "WR",
         "team": "GB", "espn_id": 4426515},
    ])
    _seed_league_one_with_slot_seven(path, _isolated_leagues_root)

    seen, stops = [], []
    monkeypatch.setattr("api.live.run_socket_listener",
                        _fake_socket(seen, stops, _first_n_selected_frames(3)))

    from fastapi.testclient import TestClient
    from api.main import create_app
    first = TestClient(create_app(path))
    try:
        assert first.post("/api/live/connect-token", json={
            "leagueId": "1", "teamId": "2", "swid": "{X}",
            "token": "1953383334", "season": "2026"}).status_code == 200
        assert _wait_until(
            lambda: first.get("/api/live/state").json()["picks_made"] == 3)
        before = first.get("/api/live/state").json()
        assert before["my_slot"] == 7
    finally:
        # The "restart": the old app object is simply abandoned, exactly as
        # a killed process abandons its in-memory state. Its listener thread
        # is asked to stop the way a kill would end it -- without going
        # through /api/live/stop, which would delete the record on purpose.
        for ev in stops:
            if ev is not None:
                ev.set()
        seen.clear()

    # The restored socket replays NOTHING. Whatever picks_made reads after
    # the restore therefore came out of the league's own `drafted` table and
    # not out of a convenient replay -- which is the half of this that was
    # already durable, and has to be shown to still be.
    monkeypatch.setattr("api.live.run_socket_listener", _fake_socket(seen, stops))
    second = TestClient(create_app(path))
    try:
        assert _wait_until(
            lambda: second.get("/api/live/state").json()["active"], timeout=30), \
            "the restore never produced a session"
        after = second.get("/api/live/state").json()
        assert after["my_slot"] == 7
        assert after["picks_made"] == 3          # still in the league's file
        assert after["restoring"] is False
        assert after["restore_error"] is None
        assert after["token_received"] is True
        # Reconnected to the same draft, as the same team, with the same
        # token -- the four fields that exist nowhere else.
        assert _wait_until(lambda: len(seen) == 1)
        assert seen[0] == {"league_id": "1", "team_id": "2", "swid": "{X}",
                           "token": "1953383334"}
    finally:
        second.post("/api/live/stop")
        for ev in stops:
            if ev is not None:
                ev.set()


def test_a_restore_against_an_expired_token_says_to_click_the_bookmark_again(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The case most likely to actually happen: ESPN's token is minted per
    draft and does not outlive it, so a restart after the draft ended (or a
    record for a draft since cancelled) meets a token ESPN refuses.

    That must be bounded and loud. The REAL run_socket_listener runs here --
    only the socket underneath it is stubbed to fail the handshake -- so
    this pins the actual give-up rule (MAX_EMPTY_RECONNECTS frameless
    attempts) and the actual sentence the user is shown, not a stand-in for
    them.
    """
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    _seed_league_one_with_slot_seven(path, _isolated_leagues_root)

    from api.live import save_session_record
    save_session_record(path, league_id="1", team_id="2", season="2026",
                        swid="{X}", token="expired", my_slot=7)

    attempts = []

    def refuse(url, cookie_header):
        attempts.append(url)
        raise OSError("HTTP 401")

    monkeypatch.setattr("pipeline.draft_socket._connect", refuse)
    # The real backoff is 2s between attempts; the rule under test is the
    # count, not the wall clock, so the wait is shortened rather than the
    # test being made to take ten seconds.
    monkeypatch.setattr("pipeline.draft_socket.RECONNECT_BACKOFF_SECONDS", 0.01)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    try:
        assert _wait_until(
            lambda: client.get("/api/live/state").json()["active"], timeout=30), \
            "the restore never rebuilt the session"
        # It gives up rather than reconnecting for ever...
        assert _wait_until(
            lambda: client.get("/api/live/state").json()["listener_error"],
            timeout=15), "an expired token never surfaced an error"
        body = client.get("/api/live/state").json()
        from pipeline.draft_socket import MAX_EMPTY_RECONNECTS
        assert len(attempts) == MAX_EMPTY_RECONNECTS
        # ...and says the one thing the owner can act on.
        assert "token may have expired" in body["listener_error"]
        assert "click the ESPN Draft Assist bookmark" in body["listener_error"]
        assert body["socket_alive"] is False
        assert _wait_until(
            lambda: client.get("/api/live/state").json()["listener_alive"] is False)
        # The board is still there. A dead socket costs the live feed, not
        # the draft: the picks and the slot came back regardless, which is
        # why the restore builds the session before it tries the socket.
        assert body["my_slot"] == 7
        # The failed connect stage carries the same instruction for the
        # connect screen, with the hint a live connect would have given.
        prog = client.get("/api/live/connect-progress").json()
        assert prog["phase"] == "failed"
        assert prog["error"]["stage"] == "socket"
        assert "bookmark" in prog["error"]["hint"]
    finally:
        client.post("/api/live/stop")


@pytest.mark.skipif(not _SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_the_browser_observer_path_clears_a_saved_token(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The record must describe the CURRENT session or nothing at all. The
    browser path holds no token of its own, so a connect through it has to
    take the old one out rather than leave a restart reconnecting to a
    session that is no longer live."""
    import os
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    _seed_league_one_with_slot_seven(path, _isolated_leagues_root)

    from api.live import save_session_record
    save_session_record(path, league_id="1", team_id="2", season="2026",
                        swid="{X}", token="tok", my_slot=7)
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    try:
        assert client.post("/api/live/connect", json={
            "url": "https://fantasy.espn.com/football/draft?leagueId=1"
                   "&seasonId=2026&teamId=2"}).status_code == 200
        assert not os.path.exists(_record_path(path))
    finally:
        client.post("/api/live/stop")


def test_the_saved_slot_is_a_fallback_and_never_overrides_a_live_one(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """Precedence, both ways round. ESPN's pick order and the league's own
    draft_order are current; a saved slot may be a draft order ago. So the
    live answer wins whenever there is one -- and the saved one is used only
    where nothing else can answer at all, which is the mock-draft case the
    slot was originally read off the socket for."""
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    _seed_league_one_with_slot_seven(path, _isolated_leagues_root)

    from api.live import save_session_record
    save_session_record(path, league_id="1", team_id="2", season="2026",
                        swid="{X}", token="tok", my_slot=6)
    seen, stops = [], []
    monkeypatch.setattr("api.live.run_socket_listener", _fake_socket(seen, stops))

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    try:
        assert _wait_until(
            lambda: client.get("/api/live/state").json()["active"], timeout=30)
        # draft_order says slot 7 for team 2 and it is read fresh, so the
        # stale 6 in the record loses.
        assert client.get("/api/live/state").json()["my_slot"] == 7
    finally:
        client.post("/api/live/stop")
        for ev in stops:
            if ev is not None:
                ev.set()

    # Now the case nothing else can answer: no draft history for this league
    # at all, which is exactly a mock draft.
    from pipeline.leagues import league_db_path
    lg = get_conn(league_db_path("1", root=_isolated_leagues_root))
    write_table(lg, "draft_teams", pd.DataFrame(
        columns=["season", "team_id", "manager", "slot"]))
    write_table(lg, "draft_order", pd.DataFrame(columns=["slot", "manager"]))
    lg.close()
    save_session_record(path, league_id="1", team_id="2", season="2026",
                        swid="{X}", token="tok", my_slot=6)
    seen2, stops2 = [], []
    monkeypatch.setattr("api.live.run_socket_listener", _fake_socket(seen2, stops2))
    client2 = TestClient(create_app(path))
    try:
        assert _wait_until(
            lambda: client2.get("/api/live/state").json()["active"], timeout=30)
        assert client2.get("/api/live/state").json()["my_slot"] == 6, \
            "with nothing live to resolve from, the saved slot is the only " \
            "thing that knows which column is ours"
    finally:
        client2.post("/api/live/stop")
        for ev in stops2:
            if ev is not None:
                ev.set()


def test_state_says_restoring_rather_than_looking_like_no_draft(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """The honest cost of rebuilding in the background: for the 4.1-34.8s a
    build takes, /api/live/state is on its inactive branch. Without this key
    that is indistinguishable from "no draft is running" -- the room would
    tell the owner nothing is happening at the moment something is."""
    import threading
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    _seed_league_one_with_slot_seven(path, _isolated_leagues_root)

    from api.live import save_session_record
    save_session_record(path, league_id="1", team_id="2", season="2026",
                        swid="{X}", token="tok", my_slot=7)

    # Hold the restore inside the board build, where a real one spends most
    # of its time, and look at what a poll sees while it is in there.
    inside, release = threading.Event(), threading.Event()

    def slow_build(*a, **k):
        # `build_session` here is this module's own import, bound at import
        # time, so monkeypatching api.live's name does not recurse into it.
        inside.set()
        release.wait(timeout=10)
        return build_session(*a, **k)

    seen, stops = [], []
    monkeypatch.setattr("api.live.build_session", slow_build)
    monkeypatch.setattr("api.live.run_socket_listener", _fake_socket(seen, stops))

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    try:
        assert inside.wait(timeout=10), "the restore never started building"
        body = client.get("/api/live/state").json()
        assert body["active"] is False
        assert body["restoring"] is True
        assert body["restore_error"] is None
        release.set()
        assert _wait_until(
            lambda: client.get("/api/live/state").json()["active"], timeout=30)
        assert client.get("/api/live/state").json()["restoring"] is False
    finally:
        release.set()
        client.post("/api/live/stop")
        for ev in stops:
            if ev is not None:
                ev.set()


def test_a_restore_that_cannot_rebuild_says_so_instead_of_saying_no_draft(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """A restore that dies before it has a listener has no listener_error to
    hang off, and reporting nothing would be indistinguishable from "there
    was no draft" -- the exact silence this whole change exists to remove."""
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    _seed_league_one_with_slot_seven(path, _isolated_leagues_root)

    from api.live import save_session_record
    save_session_record(path, league_id="1", team_id="2", season="2026",
                        swid="{X}", token="tok", my_slot=7)

    def boom(*a, **k):
        raise RuntimeError("the board would not build")

    monkeypatch.setattr("api.live.build_session", boom)
    monkeypatch.setattr("api.live.run_socket_listener",
                        lambda *a, **k: pytest.fail("a failed restore opened a socket"))

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    assert _wait_until(
        lambda: client.get("/api/live/state").json()["restore_error"], timeout=30)
    body = client.get("/api/live/state").json()
    assert body["active"] is False
    assert body["restoring"] is False
    assert "the board would not build" in body["restore_error"]


@pytest.mark.skipif(not _SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_the_last_pick_of_the_draft_deletes_the_saved_token(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """A token is worthless once its draft ends -- ESPN refuses it -- so it
    goes the moment that becomes true, rather than waiting out the age
    bound. A one-round, three-team league so the fixture's three real picks
    finish it."""
    import os
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path, extra_players=[
        {"player_id": "p2", "name": "B Runner", "position": "RB",
         "team": "DET", "espn_id": 4430807},
        {"player_id": "p3", "name": "C Slot", "position": "WR",
         "team": "GB", "espn_id": 4426515},
    ])
    lg_path = _seed_league_one_with_slot_seven(path, _isolated_leagues_root)
    tiny = league_mod.LeagueSettings(
        season=2026, teams=3, starters={"WR": 1}, flex_slots=0, bench=0,
        scoring={"receptions": 0.5}, draft_type="SNAKE")
    lg = get_conn(lg_path)
    write_table(lg, "league", pd.DataFrame([
        {"season": 2026, "league_id": "1",
         "settings_json": league_mod.to_json(tiny)}]))
    write_table(lg, "draft_order", pd.DataFrame([
        {"slot": i, "manager": f"m{i}", "is_me": i == 1} for i in (1, 2, 3)]))
    lg.close()

    seen, stops = [], []
    monkeypatch.setattr("api.live.run_socket_listener",
                        _fake_socket(seen, stops, _first_n_selected_frames(3)))

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    try:
        assert client.post("/api/live/connect-token", json={
            "leagueId": "1", "teamId": "2", "swid": "{X}",
            "token": "tok", "season": "2026"}).status_code == 200
        assert _wait_until(
            lambda: client.get("/api/live/state").json()["picks_made"] == 3)
        assert _wait_until(lambda: not os.path.exists(_record_path(path))), \
            "the token outlived the draft it was minted for"
    finally:
        client.post("/api/live/stop")
        for ev in stops:
            if ev is not None:
                ev.set()


def test_stopping_during_a_restore_keeps_the_session_stopped(
        tmp_path, monkeypatch, _isolated_leagues_root):
    """POST /api/live/stop while the rebuild is still running has to win. It
    bumps `generation` and nothing else -- no connect sequence moves -- so
    without the generation half of the guard the restore would finish a few
    seconds later and bring a session back up underneath the user who had
    just stopped it."""
    import threading
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    _seed_league_one_with_slot_seven(path, _isolated_leagues_root)

    from api.live import save_session_record
    save_session_record(path, league_id="1", team_id="2", season="2026",
                        swid="{X}", token="tok", my_slot=7)

    inside, release = threading.Event(), threading.Event()

    def slow_build(*a, **k):
        # `build_session` here is this module's own import, bound at import
        # time, so monkeypatching api.live's name does not recurse into it.
        inside.set()
        release.wait(timeout=10)
        return build_session(*a, **k)

    monkeypatch.setattr("api.live.build_session", slow_build)
    monkeypatch.setattr("api.live.run_socket_listener",
                        lambda *a, **k: pytest.fail(
                            "a stopped restore opened a socket anyway"))

    from fastapi.testclient import TestClient
    from api.main import create_app
    client = TestClient(create_app(path))
    assert inside.wait(timeout=10), "the restore never started building"
    assert client.post("/api/live/stop").status_code == 200
    release.set()
    # Give the restore every chance to finish and register; it must not.
    assert not _wait_until(
        lambda: client.get("/api/live/state").json()["active"], timeout=5), \
        "the restore re-registered a session the user had stopped"
    body = client.get("/api/live/state").json()
    assert body["active"] is False
    assert body["restoring"] is False


def test_the_app_publishes_the_running_sessions_league_settings(tmp_path):
    """`app.state.live_settings` is the seam api/main.py's board and profile
    endpoints read, and this is the other end of it.

    Without it those two endpoints price everything on the STORED `league`
    row -- the newest imported season's, which on the owner's database is 2025
    and scores no kicking -- while the room beside them ranks on the roster
    and scoring this connect fetched from ESPN. An accessor rather than the
    `state` dict itself so the lock discipline stays inside this module; it
    returns the session's own frozen LeagueSettings, so a caller cannot
    observe a half-replaced session (they are swapped whole, by
    dataclasses.replace, never mutated).
    """
    from fastapi import FastAPI
    import dataclasses as _dc
    from scoring.league import default_settings

    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    conn = get_conn(path)
    app = FastAPI()
    state, _recompute = register_live_routes(app, conn, path)

    # No session at all -- api/main.py falls back to league.load for this.
    assert app.state.live_settings() is None

    custom = _dc.replace(default_settings(), bench=8)
    state["session"] = build_session(conn, my_slot=1, settings=custom)
    assert app.state.live_settings() is custom

    # And a session that ended stops answering, rather than leaving the last
    # draft's league pricing every profile opened after it.
    state["session"] = None
    assert app.state.live_settings() is None
    conn.close()


# ---------------------------------------------------------------------------
# naming a player the board does not carry (api.live.identify_players)
# ---------------------------------------------------------------------------


def _identity_db():
    """The two identity tables and nothing else, in memory."""
    import duckdb
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE players (gsis_id VARCHAR, display_name VARCHAR, "
                 "headshot VARCHAR)")
    conn.execute("INSERT INTO players VALUES ('00-1', 'Evan Engram', "
                 "'http://x/e.png'), ('00-2', 'No Weekly Rows', NULL)")
    conn.execute("CREATE TABLE weekly (player_id VARCHAR, player_display_name "
                 "VARCHAR, position VARCHAR, recent_team VARCHAR, season "
                 "INTEGER, week INTEGER)")
    conn.execute("INSERT INTO weekly VALUES "
                 "('00-1', 'Evan Engram', 'TE', 'JAX', 2024, 17), "
                 "('00-1', 'Evan Engram', 'TE', 'DEN', 2025, 3), "
                 "('00-3', 'Roster Table Missed Him', 'WR', 'NYJ', 2025, 1)")
    return conn


def test_a_player_off_the_board_is_named_from_the_identity_tables():
    from api.live import clear_identity_cache, identify_players
    clear_identity_cache()

    found = identify_players(_identity_db(), ["00-1", "00-2", "00-3", "nobody"])

    # The team is the one he last played for, not the first row in the table.
    assert found["00-1"] == {"name": "Evan Engram", "headshot": "http://x/e.png",
                             "position": "TE", "team": "DEN"}
    # A name and a face, with no weekly rows to place him: half an identity
    # beats an id.
    assert found["00-2"]["name"] == "No Weekly Rows"
    assert found["00-2"]["position"] is None
    # And the other way round -- nflverse adds its roster table on its own
    # schedule, so `weekly` alone has to be enough to name somebody.
    assert found["00-3"]["name"] == "Roster Table Missed Him"
    assert found["00-3"]["team"] == "NYJ"
    # Nothing invented for an id neither table knows.
    assert "nobody" not in found


def test_an_id_nothing_knows_is_not_re_queried_every_poll():
    """/api/live/board is polled every couple of seconds for a whole draft."""
    from api.live import clear_identity_cache, identify_players
    clear_identity_cache()
    inner = _identity_db()
    calls = {"n": 0}

    class Counting:
        """DuckDB's own connection will not take a patched `execute` (the
        attribute is read-only), so the count is kept out here."""

        def execute(self, *args, **kwargs):
            calls["n"] += 1
            return inner.execute(*args, **kwargs)

    conn = Counting()
    identify_players(conn, ["00-1", "nobody"])
    identify_players(conn, ["00-1", "nobody"])

    assert calls["n"] == 1


def test_a_broken_identity_lookup_costs_the_names_and_nothing_else():
    from api.live import clear_identity_cache, identify_players

    class Broken:
        def execute(self, *args, **kwargs):
            raise RuntimeError("no such table: players")

    clear_identity_cache()
    assert identify_players(Broken(), ["00-1"]) == {}


def test_an_off_board_cell_carries_the_name_and_no_ranks():
    from api.live import _board_cell
    from scoring.draft_sim import snake_slots

    cell = _board_cell("00-1", 5, 8, snake_slots(8, 16), {},
                       {"00-1": {"name": "Evan Engram", "position": "TE",
                                 "team": "DEN", "headshot": "http://x/e.png"}})

    assert cell["player"]["name"] == "Evan Engram"
    assert cell["player"]["position"] == "TE"
    assert cell["player"]["overall_rank"] is None
    assert cell["player"]["market_rank"] is None
    assert cell["player"]["value"] is None


def test_a_cell_for_a_player_nothing_can_name_still_draws():
    from api.live import _board_cell
    from scoring.draft_sim import snake_slots

    cell = _board_cell("who-is-this", 5, 8, snake_slots(8, 16), {}, {})

    assert cell["player"]["name"] == "who-is-this"
    assert cell["player"]["headshot"] is None


def test_on_the_clock_the_first_turn_is_target_nows_and_argues_about_my_next_turn(
        tmp_path, monkeypatch):
    """Spec section 4: on the clock the cards are `target_now`'s, and their
    "still there" line names the pick I would be waiting for -- the same
    one the candidate row's `lasts_at_pick` names -- not this pick, where
    everybody is 100% there and the line says nothing."""
    state, _recompute = _live_routes_with_conn(tmp_path)
    session = _live_session()                     # slot 4: picks 4, 13, ...
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (np.zeros(5, bool), [None] * 3))
    monkeypatch.setattr("api.live._seed_rosters", _no_roster)

    def availability(table, ids, k, n, espn_adp=None, market_rank=None,
                     positions=None, **_):
        return np.full(len(list(ids)), 0.6)
    monkeypatch.setattr("api.live.availability_at", availability)
    monkeypatch.setattr("scoring.plan.availability_at", availability)
    _recompute(session, picks_made=3)
    first = state["plan"][0]
    assert first["pick_no"] == 4 and first["round"] == 1
    assert first["target"] is not None
    assert any("pick 13" in pro for pro in first["target"]["pros"]), first["target"]["pros"]
    assert not any("pick 4" in pro for pro in first["target"]["pros"])
    row = next(r for r in state["candidates"] if r["player_id"] == first["target"]["player_id"])
    assert row["lasts_at_pick"] == 13
    assert first["target"]["lasts_pct"] == row["lasts_pct"] == 60.0


def test_a_turn_nobody_clears_is_serialised_with_a_null_target(tmp_path, monkeypatch):
    """When no available player is likely enough to be there at a turn,
    the plan says so with `target: null` -- serialised as JSON null, which
    the page draws as "no clear target" -- rather than inventing a name."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pipeline.db import get_conn
    path = str(tmp_path / "live.duckdb")
    conn = get_conn(path)
    app = FastAPI()
    state, _recompute = register_live_routes(app, conn, path)
    client = TestClient(app)
    session = _live_session()                     # not on the clock at pick 1
    state["session"] = session
    monkeypatch.setattr("api.live._drafted_state", lambda cur, pool: (np.zeros(5, bool), []))
    monkeypatch.setattr("api.live._seed_rosters", _no_roster)

    def nobody_lasts(table, ids, k, n, espn_adp=None, market_rank=None,
                     positions=None, **_):
        return np.full(len(list(ids)), 0.1)
    monkeypatch.setattr("api.live.availability_at", nobody_lasts)
    monkeypatch.setattr("scoring.plan.availability_at", nobody_lasts)
    _recompute(session, picks_made=0)
    body = client.get("/api/live/state").json()
    assert body["plan"], "there are turns to plan"
    assert body["plan"][0]["pick_no"] == 4
    assert body["plan"][0]["target"] is None
    assert body["plan"][0]["alternates"] == []
    assert '"target": null' in client.get("/api/live/state").text.replace(" ", "").replace('"target":null', '"target": null')


def _connected_room(tmp_path, monkeypatch, extra_players=None):
    """A room connected through the token path with a socket that stays
    open, for tests about what the connect resolves."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path, extra_players=extra_players)
    seen, stops = [], []
    monkeypatch.setattr("api.live.run_socket_listener", _fake_socket(seen, stops))
    client = TestClient(create_app(path))
    resp = client.post("/api/live/connect-token", json={
        "leagueId": "1", "teamId": "2", "swid": "{X}",
        "token": "tok", "season": "2026"})
    assert resp.status_code == 200, resp.text
    return client


def test_favourites_resolved_at_connect_star_the_row(tmp_path, monkeypatch):
    monkeypatch.setattr("api.live._account_ids_for", lambda swid: ["acct-1"] if swid else [])
    asked = []
    monkeypatch.setattr("api.live.billing.favorites",
                        lambda ids: asked.append(list(ids)) or ["p1"], raising=False)
    client = _connected_room(tmp_path, monkeypatch, extra_players=[
        {"player_id": "p2", "name": "B Runner", "position": "RB",
         "team": "DET", "espn_id": 4430807}])
    try:
        assert asked == [["acct-1"]]
        assert _wait_until(lambda: client.get("/api/live/state").json()["candidates"])
        rows = {c["player_id"]: c for c in client.get("/api/live/state").json()["candidates"]}
        assert rows["p1"]["favourite"] is True and rows["p2"]["favourite"] is False
    finally:
        client.post("/api/live/stop")


def test_a_star_added_mid_draft_reaches_the_room_on_the_next_read(tmp_path, monkeypatch):
    """The room re-reads the account's stars every FAVOURITES_REFRESH_SECONDS;
    a PUT in between shows up on the next poll after that, with a fresh
    ranking behind it."""
    from api import live as live_mod
    monkeypatch.setattr("api.live._account_ids_for", lambda swid: ["acct-1"])
    stars = {"now": ["p1"]}
    monkeypatch.setattr("api.live.billing.favorites", lambda ids: list(stars["now"]), raising=False)
    client = _connected_room(tmp_path, monkeypatch, extra_players=[
        {"player_id": "p2", "name": "B Runner", "position": "RB",
         "team": "DET", "espn_id": 4430807}])
    try:
        assert _wait_until(lambda: client.get("/api/live/state").json()["candidates"])
        stars["now"] = ["p2"]
        # Not due yet: the poll keeps the stars it has.
        rows = {c["player_id"]: c for c in client.get("/api/live/state").json()["candidates"]}
        assert rows["p1"]["favourite"] is True
        # Make the read due, then poll: the room re-reads and re-ranks.
        room = client.app.state.live_registry.get(live_mod.DEFAULT_SID)
        with room.lock:
            room.state["favourites_at"] = 0.0
        client.get("/api/live/state")
        assert _wait_until(lambda: {c["player_id"]: c["favourite"] for c in
                                    client.get("/api/live/state").json()["candidates"]}
                           == {"p1": False, "p2": True})
        assert room.state["session"].favourites == frozenset({"p2"})
    finally:
        client.post("/api/live/stop")


def test_a_room_with_no_account_or_a_broken_store_has_no_stars_and_still_works(
        tmp_path, monkeypatch):
    from api import billing
    monkeypatch.setattr("api.live._account_ids_for", lambda swid: ["acct-1"])
    monkeypatch.setattr(billing, "favorites", lambda ids: (_ for _ in ()).throw(RuntimeError("store down")),
                        raising=False)
    client = _connected_room(tmp_path, monkeypatch)
    try:
        assert _wait_until(lambda: client.get("/api/live/state").json()["candidates"])
        assert all(c["favourite"] is False for c in client.get("/api/live/state").json()["candidates"])
    finally:
        client.post("/api/live/stop")
    # And a billing module with no `favorites` at all.
    monkeypatch.delattr(billing, "favorites", raising=False)
    from api import live as live_mod
    assert live_mod._favourites_for(["acct-1"]) == frozenset()
    monkeypatch.setattr("api.live._account_ids_for", lambda swid: [])
    assert live_mod._favourites_for([]) == frozenset()


def test_a_connect_with_history_fits_nobody(tmp_path, monkeypatch, _isolated_leagues_root):
    """The room's plan is a rule over counted availability, not a model of
    each opponent: a league with imported history is connected without
    `fit_all` (up to 30 s) or the cold-start opponent, and the history
    stage reports the managers the import already holds."""
    import scoring.draft_model as draft_model
    path = str(tmp_path / "live.duckdb")
    _seed_minimal_live_db(path)
    lg_path = _seed_league_one_with_slot_seven(path, _isolated_leagues_root)
    monkeypatch.setattr(draft_model, "fit_all",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fit_all ran")))
    conn = get_conn(lg_path)
    try:
        session = build_session(conn, 7, league_id="1")
    finally:
        conn.close()
    assert session.managers == 2
    assert session.betas == {} and session.nested is None
