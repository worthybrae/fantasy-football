import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient
from pipeline.db import get_conn, write_table
from api.main import create_app

def _seed(path):
    conn = get_conn(path)
    weekly = pd.DataFrame([
        {"player_id": "p1", "player_display_name": "A Star", "position": "WR",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 8, "receiving_yards": 90, "targets": 10, "carries": 0}
        for w in range(1, 18)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "A Star", "position": "WR", "team": "DET", "adp": 5.1}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "snap_counts", pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp", "espn_ppr_rank"]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))
    conn.close()

def _seed_draft_history(path):
    """Minimal imported league: one season, two managers, matched ADP.

    run_sim refuses to run with no fitted opponent models -- with none, every
    opponent draws uniformly over the whole pool and the output is
    confidently wrong rather than merely rough. Any test that runs a real sim
    therefore needs enough history for fit_all to produce a per-manager fit.
    """
    conn = get_conn(path)
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
    conn.close()


def _client(tmp_path, draft_history=False):
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    if draft_history:
        _seed_draft_history(path)
    return TestClient(create_app(path))

def _weekly_rows(pid, name, team, opp, weeks_by_season, receptions, yards, targets):
    return [
        {"player_id": pid, "player_display_name": name, "position": "WR",
         "recent_team": team, "opponent_team": opp, "season": season, "week": w,
         "receptions": receptions, "receiving_yards": yards, "targets": targets, "carries": 0}
        for season, n_weeks in weeks_by_season.items()
        for w in range(1, n_weeks + 1)
    ]

def _seed_two_players(path):
    """Two WR players engineered so production and durability disagree on who
    ranks first: p1 has much higher per-game production but far fewer career
    games (low durability); p2 has modest production but a full three-season
    workload (high durability). Both must land in the 2025 (latest) season so
    they're included in the board universe."""
    conn = get_conn(path)
    weekly = pd.DataFrame(
        _weekly_rows("p1", "A Star", "DET", "GB",
                     {2023: 4, 2024: 4, 2025: 17}, receptions=8, yards=90, targets=10) +
        _weekly_rows("p2", "B Steady", "GB", "DET",
                     {2023: 17, 2024: 17, 2025: 17}, receptions=2, yards=15, targets=4)
    )
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "A Star", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "B Steady", "position": "WR", "team": "GB", "adp": 20.0}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    conn.close()

def test_players_endpoint(tmp_path):
    c = _client(tmp_path)
    body = c.get("/api/players").json()
    row = body["players"][0]
    assert row["name"] == "A Star"
    assert row["market_rank"] == 1.0
    assert "adp" not in row
    assert "espn_ppr_rank" in row  # None here (empty espn_adp seed), but key must be present

def test_players_custom_weights(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/players", params={"w_production": 1.0, "w_role": 0.0,
                                      "w_environment": 0.0, "w_schedule": 0.0,
                                      "w_durability": 0.0})
    assert r.status_code == 200

def test_players_custom_weights_change_output(tmp_path):
    """Different weight profiles must actually change the board -- a
    regression here would mean the weight query params are silently ignored.
    p1 has high per-game production but low career durability; p2 is the
    opposite, so weighting one factor at 100% flips who ranks first."""
    path = str(tmp_path / "two.duckdb")
    _seed_two_players(path)
    c = TestClient(create_app(path))

    production_only = c.get("/api/players", params={
        "w_production": 1.0, "w_role": 0.0, "w_environment": 0.0,
        "w_schedule": 0.0, "w_durability": 0.0}).json()["players"]
    durability_only = c.get("/api/players", params={
        "w_production": 0.0, "w_role": 0.0, "w_environment": 0.0,
        "w_schedule": 0.0, "w_durability": 1.0}).json()["players"]

    assert production_only[0]["player_id"] == "p1"
    assert durability_only[0]["player_id"] == "p2"
    composite_by_id_prod = {p["player_id"]: p["composite"] for p in production_only}
    composite_by_id_dura = {p["player_id"]: p["composite"] for p in durability_only}
    assert composite_by_id_prod["p1"] != composite_by_id_dura["p1"]

def test_players_all_zero_weights_returns_422(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/players", params={"w_production": 0.0, "w_role": 0.0,
                                      "w_environment": 0.0, "w_schedule": 0.0,
                                      "w_durability": 0.0})
    assert r.status_code == 422
    assert "detail" in r.json()

def test_players_negative_weight_returns_422(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/players", params={"w_production": -1.0, "w_role": 0.25,
                                      "w_environment": 0.2, "w_schedule": 0.1,
                                      "w_durability": 0.1})
    assert r.status_code == 422

def test_drafted_roundtrip(tmp_path):
    # Task 13 adds pick_no to the POST response (see test_drafted_records_
    # pick_order). The first draft in a fresh database is deterministically
    # pick_no 1, so full-dict equality -- which also catches any unexpected
    # extra key -- still holds and is restored rather than weakened to
    # field-level checks.
    c = _client(tmp_path)
    pid = c.get("/api/players").json()["players"][0]["player_id"]
    assert c.post(f"/api/drafted/{pid}").json() == {"drafted": True, "pick_no": 1}
    assert any(p["drafted"] for p in c.get("/api/players").json()["players"])
    assert c.delete(f"/api/drafted/{pid}").json() == {"drafted": False}

def test_meta(tmp_path):
    c = _client(tmp_path)
    assert "sources" in c.get("/api/meta").json()

def test_concurrent_requests(tmp_path):
    """Verify per-request cursors handle concurrent requests without thread-safety issues."""
    c = _client(tmp_path)
    pid = c.get("/api/players").json()["players"][0]["player_id"]

    def get_players():
        r = c.get("/api/players")
        assert r.status_code == 200
        return r

    def post_drafted():
        r = c.post(f"/api/drafted/{pid}")
        assert r.status_code == 200
        return r

    # Fire 8 parallel GET requests + 1 POST request
    with ThreadPoolExecutor(max_workers=9) as executor:
        futures = []
        for _ in range(8):
            futures.append(executor.submit(get_players))
        futures.append(executor.submit(post_drafted))

        # Verify all complete successfully with no exceptions
        for future in as_completed(futures):
            result = future.result()
            assert result.status_code == 200

def test_profile_endpoint(tmp_path):
    c = _client(tmp_path)
    pid = c.get("/api/players").json()["players"][0]["player_id"]
    p = c.get(f"/api/players/{pid}/profile")
    assert p.status_code == 200
    body = p.json()
    assert body["header"]["player_id"] == pid
    assert {"factors", "seasons", "game_log", "outlook", "similar"} <= set(body)

def test_profile_404(tmp_path):
    assert _client(tmp_path).get("/api/players/nope/profile").status_code == 404

def test_profile_custom_weights_change_composite(tmp_path):
    """The profile endpoint must honor the same w_* sliders the board uses --
    a regression here would mean the drawer's rank/VOR/composite can
    contradict the clicked row. Reuses the p1/p2 production-vs-durability
    seed from test_players_custom_weights_change_output."""
    path = str(tmp_path / "two.duckdb")
    _seed_two_players(path)
    c = TestClient(create_app(path))

    default_body = c.get("/api/players/p1/profile").json()
    custom_body = c.get("/api/players/p1/profile", params={
        "w_production": 0.0, "w_role": 0.0, "w_environment": 0.0,
        "w_schedule": 0.0, "w_durability": 1.0}).json()

    assert default_body["header"]["composite"] != custom_body["header"]["composite"]

def test_profile_all_zero_weights_returns_422(tmp_path):
    c = _client(tmp_path)
    pid = c.get("/api/players").json()["players"][0]["player_id"]
    r = c.get(f"/api/players/{pid}/profile", params={
        "w_production": 0.0, "w_role": 0.0, "w_environment": 0.0,
        "w_schedule": 0.0, "w_durability": 0.0})
    assert r.status_code == 422
    assert "detail" in r.json()

def test_league_endpoint_reports_fallback_when_not_imported(tmp_path):
    client = _client(tmp_path)
    body = client.get("/api/league").json()
    assert body["derived"] is False
    assert body["teams"] == 8
    assert body["rounds"] == 15

def test_league_endpoint_reports_derived_settings(tmp_path):
    from scoring import league
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    settings = league.LeagueSettings(
        season=2026, teams=10,
        starters={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1},
        flex_slots=1, bench=6, scoring={"receptions": 0.5},
        draft_type="SNAKE", unmapped_scoring=("101",))
    write_table(conn, "league", pd.DataFrame(
        [{"season": 2026, "settings_json": league.to_json(settings)}]))
    conn.close()
    body = TestClient(create_app(path)).get("/api/league").json()
    assert body["derived"] is True
    assert body["teams"] == 10
    assert body["unmapped_scoring"] == ["101"]

def test_managers_endpoint_groups_coefficients(tmp_path):
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "manager_profiles", pd.DataFrame([
        {"manager": "worthy", "feature": "reach", "value": -1.2,
         "pooled_value": -0.4, "n_picks": 105, "heldout_gain": 0.08,
         "uses_personal": True, "summary": "sticks to market order"},
        {"manager": "worthy", "feature": "run", "value": 0.6,
         "pooled_value": 0.1, "n_picks": 105, "heldout_gain": 0.08,
         "uses_personal": True, "summary": "sticks to market order"},
    ]))
    conn.close()
    body = TestClient(create_app(path)).get("/api/managers").json()
    assert len(body["managers"]) == 1
    entry = body["managers"][0]
    assert entry["manager"] == "worthy"
    assert entry["n_picks"] == 105
    assert entry["uses_personal"] is True
    assert {c["feature"] for c in entry["coefficients"]} == {"reach", "run"}

def test_managers_endpoint_null_heldout_gain_serializes_as_json_null(tmp_path):
    """heldout_gain is written as None (-> NaN on the DuckDB/pandas round trip)
    whenever a manager has too few picks or too few seasons for a holdout
    split (see draft_model._heldout_gain / write_profiles). FastAPI's JSON
    encoder raises on a bare NaN, so this must come back as JSON null, not
    crash the endpoint."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "manager_profiles", pd.DataFrame([
        {"manager": "rookie", "feature": "reach", "value": 0.0,
         "pooled_value": 0.0, "n_picks": 3, "heldout_gain": None,
         "uses_personal": False, "summary": "league average, not enough signal"},
    ]))
    conn.close()
    r = TestClient(create_app(path)).get("/api/managers")
    assert r.status_code == 200
    entry = r.json()["managers"][0]
    assert entry["heldout_gain"] is None

def test_draft_order_round_trip(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/draft-order").json()["source"] == "none"
    payload = {"order": [{"slot": 1, "manager": "worthy"},
                         {"slot": 2, "manager": "dan"}], "my_slot": 2}
    assert client.put("/api/draft-order", json=payload).status_code == 200
    body = client.get("/api/draft-order").json()
    assert body["source"] == "manual"
    assert body["my_slot"] == 2
    assert body["order"] == [{"slot": 1, "manager": "worthy"},
                             {"slot": 2, "manager": "dan"}]

def test_draft_order_seeds_from_imported_teams(tmp_path):
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "old", "slot": 1},
        {"season": 2026, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2026, "team_id": 2, "manager": "dan", "slot": 2},
    ]))
    conn.close()
    body = TestClient(create_app(path)).get("/api/draft-order").json()
    assert body["source"] == "espn"
    assert [e["manager"] for e in body["order"]] == ["worthy", "dan"]

def test_draft_order_espn_path_skips_null_slot(tmp_path):
    """draftDayPickOrder can be missing on the ESPN side (parse_draft_teams
    writes slot=None for it). The endpoint must drop those rows instead of
    crashing on int(None)."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2026, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2026, "team_id": 2, "manager": "ghost", "slot": None},
        {"season": 2026, "team_id": 3, "manager": "dan", "slot": 2},
    ]))
    conn.close()
    r = TestClient(create_app(path)).get("/api/draft-order")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "espn"
    assert [e["manager"] for e in body["order"]] == ["worthy", "dan"]

def test_draft_order_rejects_duplicate_slots(tmp_path):
    client = _client(tmp_path)
    payload = {"order": [{"slot": 1, "manager": "worthy"},
                         {"slot": 1, "manager": "dan"}], "my_slot": 1}
    r = client.put("/api/draft-order", json=payload)
    assert r.status_code == 422

def test_draft_order_rejects_my_slot_not_in_order(tmp_path):
    client = _client(tmp_path)
    payload = {"order": [{"slot": 1, "manager": "worthy"},
                         {"slot": 2, "manager": "dan"}], "my_slot": 99}
    r = client.put("/api/draft-order", json=payload)
    assert r.status_code == 422

def test_draft_order_accepts_string_my_slot(tmp_path):
    """A frontend <select> (Task 14) can plausibly submit my_slot as a JSON
    string ("2") rather than a number. The membership check already casts
    with int(my_slot), but the row-build comparison must use the same
    normalized value -- otherwise is_me is False for every row, the PUT
    still returns 200, and the next GET silently reports my_slot: null."""
    client = _client(tmp_path)
    payload = {"order": [{"slot": 1, "manager": "worthy"},
                         {"slot": 2, "manager": "dan"}], "my_slot": "2"}
    assert client.put("/api/draft-order", json=payload).status_code == 200
    body = client.get("/api/draft-order").json()
    assert body["my_slot"] == 2

def test_sim_endpoint_starts_and_completes(tmp_path):
    import time
    client = _client(tmp_path, draft_history=True)
    client.put("/api/draft-order", json={
        "order": [{"slot": s, "manager": f"m{s}"} for s in range(1, 9)],
        "my_slot": 1})
    started = client.post("/api/sim", json={"my_slot": 1, "rollouts": 3})
    assert started.status_code == 200
    run_id = started.json()["run_id"]
    for _ in range(200):
        status = client.get(f"/api/sim/{run_id}").json()
        if status["status"] != "running":
            break
        time.sleep(0.05)
    assert status["status"] == "done", status.get("detail")

def test_sim_status_for_unknown_run_is_404(tmp_path):
    assert _client(tmp_path).get("/api/sim/nope").status_code == 404

def test_sim_refuses_to_run_without_fitted_manager_models(tmp_path):
    """With no imported draft history, fit_all returns nothing and every
    opponent's beta is zeros -- a uniform draw over the whole pool. Measured
    on a 500-player pool, the consensus number one comes back 100% likely to
    still be available at slot 8 and the top ten average 98.75%. The run has
    to fail loudly rather than write those numbers onto the board."""
    import time
    client = _client(tmp_path)                 # deliberately no draft history
    client.put("/api/draft-order", json={
        "order": [{"slot": s, "manager": f"m{s}"} for s in range(1, 9)],
        "my_slot": 1})
    run_id = client.post("/api/sim", json={"my_slot": 1, "rollouts": 3}).json()["run_id"]
    for _ in range(200):
        status = client.get(f"/api/sim/{run_id}").json()
        if status["status"] != "running":
            break
        time.sleep(0.05)
    assert status["status"] == "error"
    assert "fit-managers" in status["detail"]
    # Nothing written, so the board keeps showing blank sim columns rather
    # than a confident set of fabricated ones.
    row = client.get("/api/players").json()["players"][0]
    assert row["avail_pct"] is None and row["ev"] is None

def test_sim_latest_is_null_before_any_run_and_carries_provenance_after(tmp_path):
    """The board merges sim_results into every request unconditionally, so a
    reloaded page shows whatever the last run produced. /api/sim/latest is
    how the header can say which slot, which pick, and how old."""
    import time
    client = _client(tmp_path, draft_history=True)
    assert client.get("/api/sim/latest").json() == {"run": None}

    client.put("/api/draft-order", json={
        "order": [{"slot": s, "manager": f"m{s}"} for s in range(1, 9)],
        "my_slot": 3})
    run_id = client.post("/api/sim", json={"my_slot": 3, "rollouts": 3}).json()["run_id"]
    for _ in range(200):
        if client.get(f"/api/sim/{run_id}").json()["status"] != "running":
            break
        time.sleep(0.05)

    run = client.get("/api/sim/latest").json()["run"]
    assert run is not None
    assert run["my_slot"] == 3
    assert run["pick_no"] == 3          # slot 3, nothing drafted -> overall 3
    assert run["created_at"]

def test_model_endpoint_reports_unfitted_before_anything_is_imported(tmp_path):
    body = _client(tmp_path).get("/api/model").json()
    assert body == {"fitted": False, "n_managers": 0, "backtest": None}

def test_model_endpoint_surfaces_the_persisted_backtest(tmp_path):
    """Spec Part 3: if the model does not beat ADP-only, the board must not
    present simulator output as authoritative. The backtest was printed to
    stdout by `make fit-managers` and then dropped, so nothing downstream
    could ever know."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "manager_profiles", pd.DataFrame([
        {"manager": "worthy", "feature": "reach", "value": 1.0,
         "pooled_value": 0.0, "n_picks": 40, "heldout_gain": 0.1,
         "uses_personal": True, "summary": "reaches"}]))
    write_table(conn, "model_backtest", pd.DataFrame([
        {"holdout_season": 2025, "top1": 0.1, "top5": 0.3, "logloss": 4.0,
         "adp_top1": 0.2, "adp_logloss": 3.0, "beats_adp": False}]))
    conn.close()

    body = TestClient(create_app(path)).get("/api/model").json()
    assert body["fitted"] is True
    assert body["n_managers"] == 1
    assert body["backtest"]["beats_adp"] is False
    assert body["backtest"]["holdout_season"] == 2025

def test_model_endpoint_serializes_an_infinite_logloss_as_null(tmp_path):
    """backtest() genuinely returns inf when there is nothing to score, and
    JSON has no infinity -- FastAPI's encoder rejects it outright."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "model_backtest", pd.DataFrame([
        {"holdout_season": None, "top1": 0.0, "top5": 0.0,
         "logloss": float("inf"), "adp_top1": 0.0,
         "adp_logloss": float("inf"), "beats_adp": False}]))
    conn.close()

    r = TestClient(create_app(path)).get("/api/model")
    assert r.status_code == 200
    assert r.json()["backtest"]["logloss"] is None
    assert r.json()["backtest"]["holdout_season"] is None

def test_players_expose_sim_columns_as_null_without_a_sim(tmp_path):
    row = _client(tmp_path).get("/api/players").json()["players"][0]
    assert row["avail_pct"] is None
    assert row["ev"] is None
    assert row["ev_se"] is None

def test_drafted_records_pick_order(tmp_path):
    client = _client(tmp_path)
    assert client.post("/api/drafted/p1").json()["pick_no"] == 1
    assert client.post("/api/drafted/p2").json()["pick_no"] == 2

def test_redrafting_an_already_drafted_player_reports_its_original_pick_no(tmp_path):
    """INSERT OR IGNORE silently no-ops for an already-drafted player, so the
    stored pick_no doesn't change on a re-POST -- the response must not lie
    about that by returning the freshly-computed next_pick anyway. Task 14's
    rail reads pick order to derive whose turn it is, so a response that
    disagrees with the stored row is a real bug, not a cosmetic one."""
    client = _client(tmp_path)
    assert client.post("/api/drafted/p1").json()["pick_no"] == 1
    assert client.post("/api/drafted/p2").json()["pick_no"] == 2
    # Re-drafting p1 must still report 1, not 3 (the next free pick_no).
    assert client.post("/api/drafted/p1").json() == {"drafted": True, "pick_no": 1}

def test_concurrent_players_requests_during_running_sim(tmp_path):
    """Task 13 correctness point 4: the sim worker thread does long-running
    writes (sim_results, sim_survival, manager_profiles -- all via its own
    conn.cursor()) while request handlers keep reading/writing through their
    own cursors on the same shared DuckDB connection. Fire a burst of
    /api/players reads immediately after kicking off a sim and confirm none
    of them surface a thread-safety exception (mirrors the existing
    test_concurrent_requests pattern, but overlapping a real sim run)."""
    client = _client(tmp_path, draft_history=True)
    client.put("/api/draft-order", json={
        "order": [{"slot": s, "manager": f"m{s}"} for s in range(1, 9)],
        "my_slot": 1})
    started = client.post("/api/sim", json={"my_slot": 1, "rollouts": 15})
    assert started.status_code == 200
    run_id = started.json()["run_id"]

    def get_players():
        r = client.get("/api/players")
        assert r.status_code == 200
        return r

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_players) for _ in range(16)]
        for future in as_completed(futures):
            future.result()

    for _ in range(200):
        status = client.get(f"/api/sim/{run_id}").json()
        if status["status"] != "running":
            break
        import time
        time.sleep(0.05)
    assert status["status"] == "done", status.get("detail")
