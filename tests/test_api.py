import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient
from pipeline.db import get_conn, record_freshness, write_table
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

    Not strictly required since cold-start: a no-history league now runs on
    the market prior. But a fixture WITH history exercises the per-manager fit
    path -- the personal-vs-pooled gating, the reach warning -- that a
    cold-start league never reaches, so tests wanting that path seed this.
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
    opposite, so weighting one factor at 100% flips who has the higher
    composite.

    This used to also assert the flip in board ORDER (`players[0]`). Task 1
    moved the board's sort key from `composite`-derived `vor` to
    `proj_points`-derived `vor`, and `proj_points` (scoring.board.projections)
    is not a function of these weights at all -- so order no longer moves
    with them, by design (same reasoning as
    test_board_uses_league_settings_when_present in test_board.py). Weights
    still drive `composite` exactly as before; that's what this now checks.
    """
    path = str(tmp_path / "two.duckdb")
    _seed_two_players(path)
    c = TestClient(create_app(path))

    production_only = c.get("/api/players", params={
        "w_production": 1.0, "w_role": 0.0, "w_environment": 0.0,
        "w_schedule": 0.0, "w_durability": 0.0}).json()["players"]
    durability_only = c.get("/api/players", params={
        "w_production": 0.0, "w_role": 0.0, "w_environment": 0.0,
        "w_schedule": 0.0, "w_durability": 1.0}).json()["players"]

    composite_by_id_prod = {p["player_id"]: p["composite"] for p in production_only}
    composite_by_id_dura = {p["player_id"]: p["composite"] for p in durability_only}
    # production-only: p1 (high per-game production) leads on composite.
    assert composite_by_id_prod["p1"] > composite_by_id_prod["p2"]
    # durability-only: p2 (full 3-season workload) leads on composite --
    # the flip this test exists to catch.
    assert composite_by_id_dura["p2"] > composite_by_id_dura["p1"]
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

def test_profile_endpoint_carries_the_news_feed_and_the_injury_status(tmp_path):
    """The two tables pipeline/news.py writes reach the client, over HTTP,
    with the attribution marker intact and the injury status one lookup from
    the header rather than buried in the feed.

    `player_news`/`player_status` are written straight through write_table
    here without touching `meta`, which is the staleness gap
    scoring/profile_cache.py documents -- so the caches are cleared before
    the app is built, exactly as that module's docstring instructs.
    """
    from scoring import board_cache, profile_cache
    from pipeline.news import ATTR_EXACT, ATTR_NAME
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    now = pd.Timestamp("2026-08-19 12:00:00")
    write_table(conn, "player_news", pd.DataFrame([
        {"player_id": "p1", "name": "A Star", "headline": "id-tagged, older",
         "url": "https://espn.com/x", "published_at": now - pd.Timedelta(days=1),
         "source": "ESPN", "attribution": ATTR_EXACT, "fetched_at": now},
        {"player_id": "p1", "name": "A Star", "headline": "name-matched, newer",
         "url": "https://news.google.com/y", "published_at": now,
         "source": "Yahoo Sports", "attribution": ATTR_NAME, "fetched_at": now}]))
    write_table(conn, "player_status", pd.DataFrame([
        {"player_id": "p1", "name": "A Star", "position": "WR", "team": "DET",
         "sleeper_id": "4035", "injury_status": "Questionable",
         "injury_body_part": "Hamstring", "injury_notes": None,
         "practice_participation": None, "depth_chart_position": "LWR",
         "depth_chart_order": 1, "news_updated": now, "fetched_at": now}]))
    conn.close()
    board_cache.clear(); profile_cache.clear()

    body = TestClient(create_app(path)).get("/api/players/p1/profile").json()
    assert body["status"]["injury_status"] == "Questionable"
    assert body["status"]["source"] == "sleeper"
    # Newest first, and the marker survives serialization rather than being
    # flattened into one undifferentiated feed.
    assert [i["headline"] for i in body["news"]] == [
        "name-matched, newer", "id-tagged, older"]
    assert [i["attribution"] for i in body["news"]] == [ATTR_NAME, ATTR_EXACT]


def test_profile_endpoint_without_a_refresh_serves_an_empty_feed(tmp_path):
    """A database that has never run `make refresh` has no `player_news` and
    no `player_status` table at all -- which is what data/nfl.duckdb looked
    like when this was wired. Empty list and null, HTTP 200, no error."""
    c = _client(tmp_path)
    body = c.get("/api/players/p1/profile").json()
    assert body["news"] == [] and body["status"] is None


def test_profile_404(tmp_path):
    assert _client(tmp_path).get("/api/players/nope/profile").status_code == 404

def test_profile_endpoint_reflects_a_pick_made_after_first_call(tmp_path):
    """The scenario /api/players and /api/players/{id}/profile now share a
    cache (scoring/board_cache.py) for: the owner opens a profile (building
    and caching the board), then drafts a player through POST
    /api/drafted/{id} -- the next profile click, for that same player or any
    other, must reflect the pick rather than replaying the board that was
    cached before it. A cache keyed on anything less than a fresh read of
    the `drafted` table would show this player as still available."""
    c = _client(tmp_path)
    pid = c.get("/api/players").json()["players"][0]["player_id"]

    before = c.get(f"/api/players/{pid}/profile").json()
    assert before["header"]["drafted"] is False

    assert c.post(f"/api/drafted/{pid}").status_code == 200

    after = c.get(f"/api/players/{pid}/profile").json()
    assert after["header"]["drafted"] is True

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

def test_sim_runs_a_league_with_no_history_via_the_market_prior(tmp_path):
    """The cold-start contract, at the API level. A brand-new league has no
    imported draft history, so there are no per-manager fits -- but the sim no
    longer refuses. `fit_all` returns a market-following prior (cold_start_fits)
    and every opponent inherits it, so the run completes and writes real sim
    columns onto the board.

    This replaced a test asserting the opposite (refuse with "fit-managers"),
    which was correct only while a no-history league had no opponent model at
    all. It does now, and that is the whole point of the cold-start prior: the
    tool works for a first-time user instead of only the one league whose
    history was baked into the database.
    """
    import time
    client = _client(tmp_path)                 # deliberately no draft history
    client.put("/api/draft-order", json={
        "order": [{"slot": s, "manager": f"m{s}"} for s in range(1, 9)],
        "my_slot": 1})
    run_id = client.post("/api/sim", json={"my_slot": 1, "rollouts": 3}).json()["run_id"]
    status = {"status": "running"}
    for _ in range(400):
        status = client.get(f"/api/sim/{run_id}").json()
        if status["status"] != "running":
            break
        time.sleep(0.05)
    assert status["status"] == "done", status
    # The board now carries real sim output rather than blanks -- an actual
    # recommendation for a league the tool has never seen before.
    players = client.get("/api/players").json()["players"]
    assert any(p["ev"] is not None for p in players)

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


def test_landing_status_reports_a_ready_machine(tmp_path):
    """The readiness strip's whole job is answering "will this work tonight",
    so every fact it shows has to come back on a machine that IS ready --
    sources refreshed, history imported, managers fitted, a sim on record."""
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    _seed_draft_history(path)
    conn = get_conn(path)
    record_freshness(conn, "adp", True, 300)
    record_freshness(conn, "schedules", False, 0)
    write_table(conn, "manager_profiles", pd.DataFrame([
        {"manager": "m1", "feature": "reach", "value": 0.1, "pooled_value": 0.2,
         "n_picks": 40, "heldout_gain": 0.03, "uses_personal": True,
         "summary": "reaches"},
        {"manager": "m2", "feature": "reach", "value": 0.1, "pooled_value": 0.2,
         "n_picks": 12, "heldout_gain": None, "uses_personal": False,
         "summary": "league average"},
    ]))
    write_table(conn, "sim_results", pd.DataFrame([
        {"run_id": "r1", "player_id": "p1", "ev": 1.0, "se": 0.1,
         "applied_pct": 0.5, "rank": 1, "my_slot": 4, "pick_no": 4,
         "created_at": "2026-08-15 10:00:00"}]))
    conn.close()

    body = TestClient(create_app(path)).get("/api/landing/status").json()

    assert {s["source"] for s in body["sources"]} == {"adp", "schedules"}
    assert [s["ok"] for s in body["sources"] if s["source"] == "adp"] == [True]
    assert body["league"]["teams"] == 8
    assert body["league"]["rounds"] > 0
    assert body["history"] == {"picks": 2, "seasons": [2025], "teams": 2}
    # Fitted counts managers with a profile at all; personal counts the subset
    # whose own model beat the pooled one -- the rail's `uses_personal` gate.
    assert body["managers"] == {"fitted": 2, "personal": 1}
    assert body["sim"]["my_slot"] == 4
    assert body["sim"]["run_id"] == "r1"


def test_landing_status_on_a_machine_with_nothing_done_yet(tmp_path):
    """The states a first-time user actually sees. These are the ones worth
    pinning: a strip that reports "ready" on an empty database would be worse
    than no strip at all."""
    body = _client(tmp_path).get("/api/landing/status").json()
    assert body["sim"] is None
    assert body["history"] == {"picks": 0, "seasons": [], "teams": 0}
    assert body["managers"] == {"fitted": 0, "personal": 0}
    # No ESPN import, so the league is the built-in default rather than a
    # derived one -- the strip says so instead of implying a configured league.
    assert body["league"]["derived"] is False


def test_landing_status_does_not_build_the_board(tmp_path, monkeypatch):
    """The split between /api/landing/status and /api/landing/preview only
    buys anything if status stays off the 3.5s board build. Enforced here
    rather than left as an intention someone later 'simplifies' away."""
    monkeypatch.setattr("api.main.build_board", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("landing status must not build the board")))
    assert _client(tmp_path).get("/api/landing/status").status_code == 200


def test_landing_preview_slices_the_board_in_rank_order(tmp_path):
    path = str(tmp_path / "t.duckdb")
    _seed_two_players(path)
    client = TestClient(create_app(path))

    body = client.get("/api/landing/preview?limit=1").json()
    assert body["pool"] == len(client.get("/api/players").json()["players"])
    assert len(body["players"]) == 1
    row = body["players"][0]
    assert row["rank"] == 1
    # Slim on purpose: the preview renders eight fields, and shipping the
    # board's full 27 would put 185KB on a landing page for 12 rows.
    assert set(row) == {"rank", "name", "position", "team", "tier", "vor",
                        "market_rank", "edge"}


def test_landing_preview_clamps_an_out_of_range_limit(tmp_path):
    """Clamped, not rejected. A bad limit is never worth a landing page that
    renders an error where the board preview should be."""
    client = _client(tmp_path)
    assert len(client.get("/api/landing/preview?limit=0").json()["players"]) == 1
    assert client.get("/api/landing/preview?limit=999").status_code == 200


# ---------------------------------------------------------------------------
# The league these endpoints price in: the live session's when a draft is
# connected, the stored `league` row when one is not.
# ---------------------------------------------------------------------------
#
# WHY A KICKER IS THE PROBE. The two leagues differ in the ONE way that is
# visible end to end without arithmetic: the owner's stored row is the 2025
# import and prices no kicking (6 of the 11 kicking stat ids scoring/league.py
# can map sit in its `unmapped_scoring`, written before the map understood
# them), so a kicker's history is blanked, his `stats.ppg` is 0.0 and his
# chart is empty. A league fetched live from ESPN at connect prices kicking,
# and all three fill in. Any other difference between two leagues -- half-PPR
# against PPR, say -- moves numbers by fractions; this one moves a whole card
# between "empty" and "populated", which is what the owner actually saw.

def _kicker_app(tmp_path, league_row=None):
    """The kicker fixture from tests/test_profile.py, behind a real app.

    Reused rather than re-seeded so the endpoint tests and the `build_profile`
    tests are looking at the same kicker: 10 weeks of 2 field goals from 30-39
    and 3 extra points, worth 9.0 a game under the owner's real ESPN kicking
    values and 0.0 under full PPR.

    The fixture's own connection is CLOSED before the app opens the file --
    DuckDB is single-writer per file and `create_app` opens its own.
    """
    from tests.test_profile import _seed_kicker_fixture
    from scoring import board_cache, game_points, profile_cache
    conn = _seed_kicker_fixture(tmp_path)
    if league_row is not None:
        from scoring import league as league_mod
        write_table(conn, "league", pd.DataFrame([
            {"season": league_row.season,
             "settings_json": league_mod.to_json(league_row)}]))
    conn.close()
    # Every one of these keys on the database file, and tmp_path is unique per
    # test -- cleared anyway because the fixture writes `weekly` without
    # stamping `meta`, the one staleness gap all three caches document.
    board_cache.clear()
    profile_cache.clear()
    game_points.clear()
    return TestClient(create_app(str(tmp_path / "kick.duckdb")))


def _kicker_league():
    from tests.test_profile import _kicker_rules, _kicker_settings
    return _kicker_settings(_kicker_rules())


def test_profile_and_board_price_a_kicker_by_the_live_sessions_league(tmp_path):
    """The defect the owner reported: "for the league shouldn't it be dynamic
    based on when i enter in url?"

    api/live.py's connect builds the draft room against the roster and scoring
    it fetches from ESPN with the ids the bookmarklet supplied. These two
    endpoints read the stored `league` row instead, so the room and the card
    beside it were two different leagues -- the kicker the room was ranking on
    real points had an empty profile and a 0.0 season on the board.
    """
    client = _kicker_app(tmp_path)

    # No session: exactly what it did before -- there is no league row here,
    # so `league.load` returns the full-PPR default, which prices no kicking.
    assert client.app.state.live_settings() is None
    before = client.get("/api/players/k1/profile").json()
    assert before["seasons"] == []
    assert before["game_log"] == []
    board_before = {p["player_id"]: p
                    for p in client.get("/api/players").json()["players"]}
    assert board_before["k1"]["stats"]["ppg"] == 0.0
    assert board_before["k1"]["game_points"] is None

    # A live session, installed exactly where api/live.py installs it.
    client.app.state.live_settings = lambda request=None: _kicker_league()

    after = client.get("/api/players/k1/profile").json()
    assert [s["season"] for s in after["seasons"]] == [2025]
    assert after["seasons"][0]["ppg"] == 9.0
    assert len(after["game_log"]) == 10
    board_after = {p["player_id"]: p
                   for p in client.get("/api/players").json()["players"]}
    assert board_after["k1"]["stats"]["ppg"] == 9.0
    # ...and the chart the available table draws fills in with it, since it is
    # priced by the same settings the board is.
    assert board_after["k1"]["game_points"] == [9.0] * 10

    # The skill player beside him is untouched: nothing in this league's
    # kicking rules changes what a receiver scores.
    assert board_after["p1"]["stats"]["ppg"] == board_before["p1"]["stats"]["ppg"]
    assert board_after["p1"]["game_points"] == board_before["p1"]["game_points"]


def test_with_no_session_the_endpoints_still_read_the_stored_league_row(tmp_path):
    """The fallback, which is the normal case: opening a profile with no draft
    connected must behave exactly as it did -- `league.load(conn)`, the stored
    row, not the built-in default and not the last session's settings."""
    client = _kicker_app(tmp_path, league_row=_kicker_league())

    assert client.app.state.live_settings() is None
    body = client.get("/api/players/k1/profile").json()
    assert body["seasons"][0]["ppg"] == 9.0
    row = next(p for p in client.get("/api/players").json()["players"]
               if p["player_id"] == "k1")
    assert row["stats"]["ppg"] == 9.0
    assert row["game_points"] == [9.0] * 10


def test_a_live_session_does_not_serve_its_board_to_a_later_request_without_one(tmp_path):
    """The cache-key half of the same change. Both boards are in one process
    and one `_cache`, keyed partly on `league.to_json(settings)` -- so the
    league that asked is the league that gets served, in both directions and
    however many times the session appears and disappears."""
    client = _kicker_app(tmp_path)

    def ppg():
        return next(p for p in client.get("/api/players").json()["players"]
                    if p["player_id"] == "k1")["stats"]["ppg"]

    assert ppg() == 0.0
    client.app.state.live_settings = lambda request=None: _kicker_league()
    assert ppg() == 9.0
    client.app.state.live_settings = lambda request=None: None
    assert ppg() == 0.0
    client.app.state.live_settings = lambda request=None: _kicker_league()
    assert ppg() == 9.0


def test_a_connected_session_does_not_rebuild_the_board_on_every_request(tmp_path):
    """...and the same key must not THRASH: a session's settings object is
    fixed for its whole life, so every request during one draft has to hit the
    same entry. Keyed on anything that varies per request -- a timestamp, a
    fresh dataclass -- this would rebuild the 1.6-1.9s board on every click of
    a draft night, which is worse than the bug it fixed."""
    import scoring.board_cache as bc
    client = _kicker_app(tmp_path)
    settings = _kicker_league()
    client.app.state.live_settings = lambda request=None: settings

    builds = []
    real = bc.build_board
    bc.build_board = lambda *a, **k: (builds.append(1), real(*a, **k))[1]
    try:
        for _ in range(4):
            client.get("/api/players")
        client.get("/api/players/k1/profile")
    finally:
        bc.build_board = real
    # One build for all five requests the session makes.
    assert builds == [1]

    # THE LANDING PREVIEW IS THE ONE THAT IS ALLOWED TO DIFFER, and it did
    # not used to be: it passed the caller's own session settings, so this
    # test counted six requests and one build. It is now deliberately
    # anonymous -- `_league_settings(cur, None)`, because the answer carries
    # `Cache-Control: public` and is shared with everybody, so it must not
    # vary with whoever happens to be in a draft room. That is a second
    # league, so it is a second entry, built once and then cached like any
    # other.
    builds.clear()
    bc.build_board = lambda *a, **k: (builds.append(1), real(*a, **k))[1]
    try:
        client.get("/api/landing/preview")
        client.get("/api/landing/preview")
        client.get("/api/players")
    finally:
        bc.build_board = real
    assert builds == [1]


# ---------------------------------------------------------------------------
# `game_points`: last completed season, one bar per week, for the available
# table's inline chart (web/src/components/draft/AvailableList.tsx).
# ---------------------------------------------------------------------------

def _seed_game_points(path):
    """Three players covering the three shapes the chart has to draw: a full
    season, a season with holes in it, and a player with no rows at all."""
    conn = get_conn(path)
    weekly = pd.DataFrame(
        [{"player_id": "p1", "player_display_name": "A Star", "position": "WR",
          "recent_team": "DET", "opponent_team": "GB", "season": 2025,
          "week": w, "receptions": 8, "receiving_yards": 90, "targets": 10,
          "carries": 0}
         for w in range(1, 18)]
        + [{"player_id": "p2", "player_display_name": "B Gap", "position": "RB",
            "recent_team": "GB", "opponent_team": "DET", "season": 2025,
            "week": w, "receptions": 3, "receiving_yards": 20, "targets": 4,
            "carries": 5}
           for w in (1, 3, 17)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "A Star", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "B Gap", "position": "RB", "team": "GB", "adp": 40.0},
        # No weekly rows anywhere: the rookie/defense case.
        {"adp_name": "C Rookie", "position": "WR", "team": "SF", "adp": 90.0}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "snap_counts", pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))
    conn.close()
    from scoring import board_cache, game_points
    board_cache.clear()
    game_points.clear()


def test_players_carry_last_seasons_points_week_by_week(tmp_path):
    """Indexed by WEEK, not packed by game, and null for a week with no row.

    The chart draws every player against the same slots, so a back who played
    three games reads as three bars in an otherwise empty season rather than
    as a full one with short bars -- and a bye or an injury reads as a gap
    rather than as a nought-point game, which is a different claim.
    """
    path = str(tmp_path / "g.duckdb")
    _seed_game_points(path)
    rows = {p["name"]: p for p in
            TestClient(create_app(path)).get("/api/players").json()["players"]}

    # 8 receptions + 90 yards = 8 x 1.0 + 90 x 0.1 = 17.0, every week of 17.
    assert rows["A Star"]["game_points"] == [17.0] * 17
    # 3 + 2.0 = 5.0, in weeks 1, 3 and 17 of the same 17-week season.
    gap = rows["B Gap"]["game_points"]
    assert len(gap) == 17
    assert gap == [5.0, None, 5.0] + [None] * 13 + [5.0]
    # A player with no rows in that season gets JSON null -- an explicit "no
    # games last season" the table renders as an empty state. Never [] or
    # [0, 0, ...], both of which read as "played and did not score".
    assert rows["C Rookie"]["game_points"] is None


def test_the_per_game_chart_is_built_once_per_refresh_not_once_per_pick(tmp_path):
    """The whole reason it is not a `build_board` column: these arrays depend
    on nothing a pick can change, so they are keyed like
    scoring/profile_cache.py is (database + `meta` + the league's rules) and
    survive one.

    NEITHER CACHE NOTICES A PICK ANY MORE. This used to assert two board
    builds, because the board cache was keyed on the `drafted` table and a
    pick threw the whole board away -- a dozen-plus times an hour on draft
    night, at 1.6s and ~0.9 GB each. The drafted set is now stamped onto the
    cached frame instead of keying it (see `board_cache._drafted_ids`), so
    the pick below costs one column rather than one build. What the pick must
    still do -- show up, immediately, on the very next request -- is pinned
    by `test_a_pick_reuses_the_cached_board` in tests/test_board_cache.py and
    by `test_profile_endpoint_reflects_a_pick_made_after_first_call`, in this
    file, above."""
    import scoring.board_cache as bc
    import scoring.game_points as gp
    path = str(tmp_path / "g.duckdb")
    _seed_game_points(path)
    client = TestClient(create_app(path))

    boards, games = [], []
    real_board, real_games = bc.build_board, gp._build
    bc.build_board = lambda *a, **k: (boards.append(1), real_board(*a, **k))[1]
    gp._build = lambda *a, **k: (games.append(1), real_games(*a, **k))[1]
    try:
        client.get("/api/players")
        client.get("/api/players")
        client.post("/api/drafted/p1")
        client.get("/api/players")
    finally:
        bc.build_board, gp._build = real_board, real_games

    # The pick costs one column on a cached frame, not a rebuild...
    assert len(boards) == 1
    # ...and this is untouched either way.
    assert len(games) == 1


# ---------------------------------------------------------------------------
# Cache headers, at the level of the assembled app
# ---------------------------------------------------------------------------


def test_the_cache_middleware_is_wired_into_create_app(tmp_path):
    """NOTHING ELSE PINS THE INSTALL. `api/http_cache.py` has its own tests,
    but they build two-route apps; if the two `install` calls fell out of
    `create_app` in a merge, every one of them would still pass and the
    deployed app would answer every endpoint with no policy at all -- which,
    behind a cache-everything rule, is how one reader's draft ends up served
    to the next.

    Two endpoints, one for each half. `/api/live/state` names no policy of
    its own, so it can only be `private, no-store` if the default middleware
    is on the app. `/api/lobby` opts in, so it can only say `s-maxage=15` if
    the route was reached and its header survived to the socket."""
    from api import lobby
    lobby.clear_cache()
    try:
        # Primed so the route reads the module cache rather than ESPN.
        lobby._lobby_summary(fetch=lambda season=None: [], now_ms=1_000.0)
        client = _client(tmp_path)

        assert client.get("/api/live/state").headers["Cache-Control"] == "private, no-store"
        assert "s-maxage=15" in client.get("/api/lobby").headers["Cache-Control"]
    finally:
        lobby.clear_cache()


def test_a_public_landing_answer_does_not_vary_with_a_cookie(tmp_path):
    """THE ONE MISTAKE A LATER HEADER EDIT CANNOT UNDO. `public` tells a
    shared cache it may hand this response to the next visitor. If the
    handler behind it reads a cookie -- `espn_live` names one browser's draft
    room, `espn_custody` its stored ESPN session -- then what the cache
    stores is one person's answer and what it serves is everybody's.

    These two endpoints are the ones close enough to reach: the preview
    prices its board through `_league_settings`, which HAS a live-session
    path, and the status strip reads the same league. Both are pinned here
    rather than in the handlers, because the property has to survive whoever
    edits them next -- it is about the header, not about today's code.
    """
    import uuid
    # OVER HTTPS, because `espn_custody` is one of the two cookies and
    # `CredentialTransportGuard` refuses any request carrying it over
    # plaintext -- correctly, and on every path, so a plain-http client here
    # would be testing the guard instead of the header.
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    client = TestClient(create_app(path), base_url="https://testserver")
    cookies = {"Cookie": f"espn_live={uuid.uuid4().hex}; "
                         f"espn_custody={uuid.uuid4().hex}"}

    for path in ("/api/landing/status", "/api/landing/preview"):
        anonymous = client.get(path)
        carrying = client.get(path, headers=cookies)

        assert anonymous.status_code == 200, path
        assert carrying.status_code == 200, path
        assert "public" in anonymous.headers["Cache-Control"], path
        # Same policy for both, or the cache would key one of them differently
        # and the question of which body it stored would be a coin toss.
        assert anonymous.headers["Cache-Control"] == carrying.headers["Cache-Control"], path
        assert anonymous.json() == carrying.json(), path
