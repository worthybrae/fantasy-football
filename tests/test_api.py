import json
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

def test_managers_endpoint_marks_every_non_dummy_coefficient_shown(tmp_path):
    """`shown` comes from the model, so a feature added there reaches the
    rail's card without a second, hand-kept list in the web client going
    stale (which is exactly what happened when the model grew `age`,
    `hype`, `trend` and `no_track_record`). Position dummies stay off: one
    bar for `pos_RB` says nothing without the others beside it."""
    from scoring.draft_model import FEATURE_NAMES
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "manager_profiles", pd.DataFrame([
        {"manager": "worthy", "feature": f, "value": 0.1, "pooled_value": 0.0,
         "n_picks": 105, "heldout_gain": 0.08, "uses_personal": True,
         "summary": "s"} for f in FEATURE_NAMES]))
    conn.close()
    body = TestClient(create_app(path)).get("/api/managers").json()
    shown = {c["feature"] for c in body["managers"][0]["coefficients"] if c["shown"]}
    assert shown == {f for f in FEATURE_NAMES if not f.startswith("pos_")}
    assert {"age", "hype", "trend", "no_track_record"} <= shown

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

def test_managers_history_endpoint_empty_without_import(tmp_path):
    assert _client(tmp_path).get("/api/managers/history").json() == {"managers": []}


def test_managers_history_endpoint_shapes_real_picks(tmp_path):
    """Two managers, two seasons, exercising every shape in the payload:
    first-rounders sorted most-recent-first, a keeper flagged, a pick whose
    espn_player_id had no match in that season's ESPN player directory
    (player_name/position/nfl_team come back JSON null rather than a
    literal-"null" string), and round-bucketed position counts spanning
    early (<=3) and late (9+)."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2024, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2024, "team_id": 2, "manager": "dan", "slot": 2},
        {"season": 2025, "team_id": 1, "manager": "worthy", "slot": 1},
    ]))
    write_table(conn, "draft_picks", pd.DataFrame([
        {"season": 2024, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 11, "player_name": "Bijan Robinson",
         "position": "RB", "nfl_team": "ATL", "keeper": False},
        {"season": 2024, "overall_pick": 2, "round": 1, "round_pick": 2,
         "team_id": 2, "espn_player_id": 99, "player_name": None,
         "position": None, "nfl_team": None, "keeper": False},
        {"season": 2024, "overall_pick": 9, "round": 2, "round_pick": 1,
         "team_id": 1, "espn_player_id": 12, "player_name": "Breece Hall",
         "position": "RB", "nfl_team": "NYJ", "keeper": False},
        {"season": 2024, "overall_pick": 73, "round": 10, "round_pick": 1,
         "team_id": 1, "espn_player_id": 13, "player_name": "Late WR",
         "position": "WR", "nfl_team": "GB", "keeper": False},
        {"season": 2025, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 14, "player_name": "Ja'Marr Chase",
         "position": "WR", "nfl_team": "CIN", "keeper": True},
    ]))
    conn.close()

    body = TestClient(create_app(path)).get("/api/managers/history").json()
    by_name = {m["manager"]: m for m in body["managers"]}
    assert set(by_name) == {"worthy", "dan"}

    worthy = by_name["worthy"]
    assert worthy["seasons"] == 2
    assert worthy["total_picks"] == 4
    assert [fr["season"] for fr in worthy["first_rounders"]] == [2025, 2024]
    assert worthy["first_rounders"][0] == {
        "season": 2025, "player_name": "Ja'Marr Chase", "position": "WR",
        "nfl_team": "CIN", "keeper": True}
    assert worthy["shape"]["early"]["RB"] == 2   # round 1 + round 2
    assert worthy["shape"]["early"]["WR"] == 1   # 2025's round-1 keeper
    assert worthy["shape"]["late"]["WR"] == 1    # round 10
    assert worthy["shape"]["mid"]["RB"] == 0

    dan = by_name["dan"]
    assert dan["seasons"] == 1
    assert dan["total_picks"] == 1
    assert dan["first_rounders"] == [
        {"season": 2024, "player_name": None, "position": None,
         "nfl_team": None, "keeper": False}]
    # No manager_tendencies table written -- the block is omitted rather than
    # rendered as a row of blanks. See _tendency_payload.
    assert worthy["tendencies"] is None
    assert dan["tendencies"] is None


def _seed_tendencies_table(path):
    """A manager_tendencies table in the long shape draft_model writes."""
    conn = get_conn(path)
    write_table(conn, "manager_tendencies", pd.DataFrame([
        {"manager": "worthy", "metric": "first_pick", "key": "WR",
         "value": 1.0, "n": 2},
        {"manager": "worthy", "metric": "first_pick", "key": "RB",
         "value": 4.0, "n": 2},
        {"manager": "worthy", "metric": "reach_overall", "key": None,
         "value": 4.25, "n": 88},
        {"manager": "worthy", "metric": "reach_bucket", "key": "late",
         "value": -3.5, "n": 30},
        {"manager": "worthy", "metric": "reach_bucket", "key": "early",
         "value": 9.0, "n": 18},
        {"manager": "worthy", "metric": "reach_bucket", "key": "mid",
         "value": 2.0, "n": 40},
        {"manager": "worthy", "metric": "reach_position", "key": "TE",
         "value": 14.0, "n": 9},
        {"manager": "worthy", "metric": "reach_position", "key": "K",
         "value": 58.0, "n": 6},
        {"manager": "worthy", "metric": "first_at_position", "key": "K",
         "value": 14.5, "n": 6},
        {"manager": "worthy", "metric": "first_at_position", "key": "QB",
         "value": 6.0, "n": 6},
    ]))
    conn.close()


def test_managers_history_endpoint_serves_measured_tendencies(tmp_path):
    """The facts the card leads with. Ordering is the endpoint's job, not the
    client's: openers most-used first, reach buckets in early->late draft
    order rather than table order, reach positions strongest first, and the
    first-at-position list earliest round first."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2024, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2025, "team_id": 1, "manager": "worthy", "slot": 1},
    ]))
    write_table(conn, "draft_picks", pd.DataFrame([
        {"season": 2024, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 11, "player_name": "Bijan Robinson",
         "position": "RB", "nfl_team": "ATL", "keeper": False},
        {"season": 2025, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 14, "player_name": "Ja'Marr Chase",
         "position": "WR", "nfl_team": "CIN", "keeper": False},
    ]))
    conn.close()
    _seed_tendencies_table(path)

    body = TestClient(create_app(path)).get("/api/managers/history").json()
    t = body["managers"][0]["tendencies"]

    assert t["first_pick"] == [{"position": "RB", "drafts": 4, "of": 2},
                               {"position": "WR", "drafts": 1, "of": 2}]
    assert t["reach"] == {"mean_gap": 4.25, "n": 88}
    assert [b["bucket"] for b in t["reach_by_bucket"]] == ["early", "mid", "late"]
    assert t["reach_by_bucket"][2]["mean_gap"] == -3.5
    assert [p["position"] for p in t["reach_by_position"]] == ["K", "TE"]
    assert [f["position"] for f in t["first_at_position"]] == ["QB", "K"]
    assert t["first_at_position"][0] == {"position": "QB", "mean_round": 6.0,
                                         "drafts": 6}


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
    could ever know.

    `seasons` (not `holdout_season`, which leave-one-season-out has no
    single value for) is stored as a JSON string -- see
    `draft_model.write_backtest` -- so the fixture row writes it that way
    too, matching what `write_backtest` actually persists rather than what
    `backtest()` returns in memory.
    """
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "manager_profiles", pd.DataFrame([
        {"manager": "worthy", "feature": "reach", "value": 1.0,
         "pooled_value": 0.0, "n_picks": 40, "heldout_gain": 0.1,
         "uses_personal": True, "summary": "reaches"}]))
    write_table(conn, "model_backtest", pd.DataFrame([
        {"seasons": json.dumps([2023, 2024, 2025]), "top1": 0.1, "top5": 0.3,
         "logloss": 4.0, "adp_top1": 0.2, "adp_logloss": 3.0,
         "beats_adp": False}]))
    conn.close()

    body = TestClient(create_app(path)).get("/api/model").json()
    assert body["fitted"] is True
    assert body["n_managers"] == 1
    assert body["backtest"]["beats_adp"] is False
    assert body["backtest"]["seasons"] == [2023, 2024, 2025]

def test_model_endpoint_beats_adp_is_null_when_the_column_is_missing(tmp_path):
    """A `model_backtest` row written before `beats_adp` existed made no
    comparison. `bool(None)` is False, and the rail renders False as the
    positive claim "the fitted model does not beat the ADP baseline" -- an
    accusation about a comparison nobody ran. Null, like the numbers beside
    it, so the rail can stay quiet."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "model_backtest", pd.DataFrame([
        {"seasons": json.dumps([2025]), "top1": 0.2, "top5": 0.5,
         "logloss": 3.0}]))
    conn.close()

    body = TestClient(create_app(path)).get("/api/model").json()
    assert body["backtest"]["beats_adp"] is None
    assert body["backtest"]["adp_top1"] is None

def test_model_endpoint_serializes_an_infinite_logloss_as_null(tmp_path):
    """backtest() genuinely returns inf when there is nothing to score, and
    JSON has no infinity -- FastAPI's encoder rejects it outright. `seasons`
    is `[]` in that case (see `backtest`'s empty-observations branch), not
    `None` -- LOSO always returns a list, just an empty one when there is
    nothing to rotate through."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "model_backtest", pd.DataFrame([
        {"seasons": json.dumps([]), "top1": 0.0, "top5": 0.0,
         "logloss": float("inf"), "adp_top1": 0.0,
         "adp_logloss": float("inf"), "beats_adp": False}]))
    conn.close()

    r = TestClient(create_app(path)).get("/api/model")
    assert r.status_code == 200
    assert r.json()["backtest"]["logloss"] is None
    assert r.json()["backtest"]["seasons"] == []

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


def test_sim_board_is_empty_without_a_run(tmp_path):
    body = _client(tmp_path).get("/api/sim/board").json()
    assert body["run"] is None
    assert body["cells"] == []
    assert body["teams"] == 8
    assert body["rounds"] == 15


def test_sim_board_serves_cells_with_player_details(tmp_path):
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "sim_results", pd.DataFrame(
        [{"run_id": "r1", "player_id": "p1", "ev": 1.0, "se": 0.1, "rank": 1,
          "applied_pct": 1.0, "my_slot": 4, "pick_no": 0,
          "created_at": "2026-08-09 12:00:00"}]))
    write_table(conn, "sim_board", pd.DataFrame([
        {"run_id": "r1", "overall_pick": 1, "round": 1, "round_pick": 1,
         "slot": 1, "alt_rank": 0, "player_id": "p1", "prob": 0.42,
         "certain": False},
        {"run_id": "r1", "overall_pick": 1, "round": 1, "round_pick": 1,
         "slot": 1, "alt_rank": 1, "player_id": "nope", "prob": 0.2,
         "certain": False},
    ]))
    conn.close()
    body = TestClient(create_app(path)).get("/api/sim/board").json()
    assert body["run"]["my_slot"] == 4
    primary = [c for c in body["cells"] if c["alt_rank"] == 0][0]
    assert primary["name"] == "A Star"
    assert primary["position"] == "WR"
    assert primary["prob"] == 0.42
    # `market_spread` rides along so the grid can mark a contested placement.
    # Present on every cell, null included -- the client tests it against a
    # threshold, and a missing key would read as 0 (agreement) rather than
    # "unknown", which is the wrong way for that mistake to fail.
    assert "market_spread" in primary
    # A cell naming a player the board no longer carries still renders,
    # with the id standing in for the name rather than vanishing. It has no
    # board row to take a spread from, so it reports none rather than
    # inheriting the previous row's.
    alt = [c for c in body["cells"] if c["alt_rank"] == 1][0]
    assert alt["name"] == "nope"
    assert alt["position"] is None
    assert alt["market_spread"] is None


def _sim_board_cells(run_id="r1"):
    return pd.DataFrame([
        {"run_id": run_id, "overall_pick": 1, "round": 1, "round_pick": 1,
         "slot": 1, "alt_rank": 0, "player_id": "p1", "prob": 0.42,
         "certain": False}])


def test_sim_board_survives_a_run_that_predates_the_provenance_columns(tmp_path):
    """`sim_results` was created without my_slot/pick_no/created_at -- they
    arrived eleven commits later. /api/sim/latest guards for exactly that
    (`"created_at" not in res.columns`); the board handler must too, or the
    whole page 500s for anyone whose last run predates the migration."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "sim_results", pd.DataFrame(
        [{"run_id": "r1", "player_id": "p1", "ev": 1.0, "se": 0.1, "rank": 1,
          "applied_pct": 1.0}]))
    write_table(conn, "sim_board", _sim_board_cells())
    conn.close()
    res = TestClient(create_app(path)).get("/api/sim/board")
    assert res.status_code == 200
    body = res.json()
    # Same answer /api/sim/latest gives for the same table: no provenance on
    # record, so no run to describe -- and no column marked "(you)".
    assert body["run"] is None
    assert [c["name"] for c in body["cells"]] == ["A Star"]


def _seed_adp_only(path, rows, keep_a_star=True):
    """Replace the ADP feed with entries that match nothing in the weekly
    universe, so `_add_adp_only_players` puts them on the board with a
    synthetic `"adp_" + norm` id."""
    conn = get_conn(path)
    head = [{"adp_name": "A Star", "position": "WR", "team": "DET",
             "adp": 5.1}] if keep_a_star else []
    write_table(conn, "adp", pd.DataFrame(head + rows))
    conn.close()


def test_sim_board_survives_a_duplicate_player_id_on_the_board(tmp_path):
    """A duplicated id makes `names.loc[pid]` a DataFrame, so `row["name"]`
    is a Series and the response encoder blows up -- one bad board row would
    otherwise kill all 120 cells."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    # One normalized name at two positions, neither in the weekly universe.
    # `_add_adp_only_players` keys the synthetic id on the name alone, so both
    # rows land on the board sharing `adp_dup_guy` -- the duplication
    # `build_board`'s own drop_duplicates comment documents.
    _seed_adp_only(path, [
        {"adp_name": "Dup Guy", "position": "RB", "team": "SF", "adp": 6.0},
        {"adp_name": "Dup Guy", "position": "TE", "team": "GB", "adp": 7.0},
    ])
    conn = get_conn(path)
    write_table(conn, "sim_results", pd.DataFrame(
        [{"run_id": "r1", "player_id": "p1", "ev": 1.0, "se": 0.1, "rank": 1,
          "applied_pct": 1.0, "my_slot": 4, "pick_no": 0,
          "created_at": "2026-08-09 12:00:00"}]))
    cells = _sim_board_cells()
    cells.loc[len(cells)] = {"run_id": "r1", "overall_pick": 2, "round": 1,
                             "round_pick": 2, "slot": 2, "alt_rank": 0,
                             "player_id": "adp_dup_guy", "prob": 0.3,
                             "certain": False}
    write_table(conn, "sim_board", cells)
    conn.close()
    res = TestClient(create_app(path)).get("/api/sim/board")
    assert res.status_code == 200
    dup = [c for c in res.json()["cells"] if c["player_id"] == "adp_dup_guy"][0]
    assert dup["name"] == "Dup Guy"
    # One of the two duplicated rows wins; either is a real board row, and
    # the cell must carry a plain string rather than a Series.
    assert dup["position"] in ("RB", "TE")


def test_sim_board_serializes_a_missing_team_as_null(tmp_path):
    """`SimBoardCell.team` is `string | null`, and pandas has two spellings
    of missing that break it: np.nan makes json.dumps raise ("Out of range
    float values are not JSON compliant") -> HTTP 500, and pd.NA survives
    FastAPI's encoder as `{}`, which React then refuses to render as a child
    (`DraftGrid` does `{primary.team ?? '—'}`) -- no error boundary, so
    /draft-board white-screens. /api/players carries an explicit comment
    about exactly this hazard; this handler needs the same normalization.

    Reached the way real data reaches it: an ADP feed that publishes no team
    at all comes back out of DuckDB as a nullable Int32 column, and
    build_board's `cur_team.fillna(adp_team).fillna(team)` chain then has
    nothing to fall back on for an ADP-only player."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    _seed_adp_only(path, [
        {"adp_name": "No Team Guy", "position": "RB", "team": None,
         "adp": 6.0}], keep_a_star=False)
    conn = get_conn(path)
    write_table(conn, "sim_results", pd.DataFrame(
        [{"run_id": "r1", "player_id": "p1", "ev": 1.0, "se": 0.1, "rank": 1,
          "applied_pct": 1.0, "my_slot": 4, "pick_no": 0,
          "created_at": "2026-08-09 12:00:00"}]))
    write_table(conn, "sim_board", pd.DataFrame([
        {"run_id": "r1", "overall_pick": 1, "round": 1, "round_pick": 1,
         "slot": 1, "alt_rank": 0, "player_id": "adp_no_team_guy",
         "prob": 0.42, "certain": False}]))
    conn.close()
    res = TestClient(create_app(path)).get("/api/sim/board")
    assert res.status_code == 200
    cell = res.json()["cells"][0]
    assert cell["name"] == "No Team Guy"
    assert cell["team"] is None


def test_draft_order_seeds_slots_when_espn_has_not_published_an_order(tmp_path):
    # ESPN leaves draftDayPickOrder null until it publishes a draft order,
    # which is the normal state for the season you are preparing for. An
    # empty order left the rail with no rows to edit and no way to enter one.
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2026, "team_id": 7, "manager": "dan", "slot": None},
        {"season": 2026, "team_id": 3, "manager": "worthy", "slot": None},
    ]))
    conn.close()
    body = TestClient(create_app(path)).get("/api/draft-order").json()
    assert body["source"] == "unpublished"
    assert body["order"] == [{"slot": 1, "manager": "worthy"},
                             {"slot": 2, "manager": "dan"}]
