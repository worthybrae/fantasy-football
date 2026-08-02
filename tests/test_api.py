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
    conn.close()

def _client(tmp_path):
    path = str(tmp_path / "t.duckdb")
    _seed(path)
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
    assert body["players"][0]["name"] == "A Star"
    assert body["players"][0]["adp"] == 5.1

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
    c = _client(tmp_path)
    pid = c.get("/api/players").json()["players"][0]["player_id"]
    assert c.post(f"/api/drafted/{pid}").json() == {"drafted": True}
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
