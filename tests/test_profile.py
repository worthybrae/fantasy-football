import pandas as pd
from pipeline.db import get_conn, write_table
from scoring.profile import season_summaries, game_log, build_profile, _stat_line

def _weekly_rows():
    return pd.DataFrame(
        [{"player_id": "p1", "player_display_name": "Amon-Ra St. Brown",
          "position": "WR", "recent_team": "DET", "opponent_team": "GB",
          "season": 2025, "week": w, "receptions": 6, "receiving_yards": 80,
          "receiving_tds": 1, "targets": 9, "carries": 0}
         for w in range(1, 11)])

def test_season_summaries_math_and_snap_join():
    snaps = pd.DataFrame([{"player": "Amon-Ra St Brown", "team": "DET",
                           "season": 2025, "offense_pct": 0.9}])
    s = season_summaries(_weekly_rows(), snaps, "p1")[0]
    assert s["season"] == 2025 and s["games"] == 10
    assert s["ppg"] == 20.0            # 6 + 8 + 6 = 20 per game
    assert s["snap_share"] == 0.9      # matched despite punctuation
    assert s["target_share"] == 1.0

def test_game_log_line_and_order():
    rows = game_log(_weekly_rows(), "p1")
    assert rows[0]["week"] == 10       # newest first
    assert rows[0]["stat_line"] == "9 tgt, 6 rec, 80 yds, 1 TD"
    assert rows[0]["ppr_points"] == 20.0

def _seed(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "weekly", _weekly_rows())
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET",
         "adp": 5.1},
        {"adp_name": "Rookie Guy", "position": "WR", "team": "GB", "adp": 90.0}]))
    write_table(conn, "depth_charts",
                pd.DataFrame(columns=["gsis_id", "depth_team", "formation",
                                      "week", "position"]))
    write_table(conn, "snap_counts",
                pd.DataFrame(columns=["player", "team", "season", "offense_pct"]))
    return conn

def test_build_profile_shape(tmp_path):
    p = build_profile(_seed(tmp_path), "p1")
    assert p["header"]["name"] == "Amon-Ra St. Brown"
    assert set(p["factors"]) == {"production", "durability", "role",
                                 "environment", "schedule"}
    assert p["seasons"][0]["games"] == 10
    assert len(p["game_log"]) == 10
    assert p["outlook"]["implied_points"] == 27.0
    assert p["outlook"]["bye"] is None or isinstance(p["outlook"]["bye"], int)

def test_build_profile_unknown_and_rookie(tmp_path):
    from scoring.board import build_board
    conn = _seed(tmp_path)
    assert build_profile(conn, "nope") is None
    # rookie (ADP-only): empty history, value-neighbors fallback
    board = build_board(conn)
    rk = board[board["name"] == "Rookie Guy"].iloc[0]["player_id"]
    prof = build_profile(conn, rk)
    assert prof["seasons"] == [] and prof["game_log"] == []
    assert prof["similar"]["mode"] == "value_neighbors"

def test_no_nan_anywhere(tmp_path):
    import math, json
    p = build_profile(_seed(tmp_path), "p1")
    json.dumps(p, allow_nan=False)  # raises if any NaN survived scrubbing

def test_scrubs_pandas_na_from_empty_adp(tmp_path):
    import json
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "weekly", _weekly_rows())
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    # empty adp -> uni["adp"] = pd.NA for every row (scoring/board.py), which
    # is not a float NaN and slips past isinstance(v, (np.floating, float))
    write_table(conn, "adp",
                pd.DataFrame(columns=["adp_name", "position", "team", "adp"]))
    write_table(conn, "depth_charts",
                pd.DataFrame(columns=["gsis_id", "depth_team", "formation",
                                      "week", "position"]))
    write_table(conn, "snap_counts",
                pd.DataFrame(columns=["player", "team", "season", "offense_pct"]))
    p = build_profile(conn, "p1")
    json.dumps(p, allow_nan=False)  # must not raise
    assert p["header"]["adp"] is None

def _twin_weekly_rows():
    # p1: the profiled player, 2025 (latest season).
    p1 = [{"player_id": "p1", "player_display_name": "Star One", "position": "WR",
           "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
           "receptions": 6, "receiving_yards": 80, "receiving_tds": 1, "targets": 9,
           "carries": 0} for w in range(1, 11)]
    # p2: an on-board twin -- also present in 2025 (the latest season), so it
    # enters the board universe and gets a real rank.
    p2 = [{"player_id": "p2", "player_display_name": "On Board Twin", "position": "WR",
           "recent_team": "GB", "opponent_team": "DET", "season": 2025, "week": w,
           "receptions": 6, "receiving_yards": 78, "receiving_tds": 1, "targets": 9,
           "carries": 0} for w in range(1, 11)]
    # p3: an off-board twin -- present only in 2023 (not the latest season)
    # and absent from ADP, so it never enters the board universe even though
    # its stat-similarity to p1 qualifies it as a twin candidate.
    p3 = [{"player_id": "p3", "player_display_name": "Off Board Twin", "position": "WR",
           "recent_team": "GB", "opponent_team": "DET", "season": 2023, "week": w,
           "receptions": 6, "receiving_yards": 79, "receiving_tds": 1, "targets": 9,
           "carries": 0} for w in range(1, 11)]
    return pd.DataFrame(p1 + p2 + p3)

def test_build_profile_enriches_stat_twins_on_and_off_board(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "weekly", _twin_weekly_rows())
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    # p3 is deliberately absent here -- it must not enter the board universe
    # via an ADP match either.
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Star One", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "On Board Twin", "position": "WR", "team": "GB", "adp": 20.0}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "snap_counts", pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))

    prof = build_profile(conn, "p1")
    assert prof["similar"]["mode"] == "stat_twins"
    twins = {p["player_id"]: p for p in prof["similar"]["players"]}
    assert "p2" in twins and "p3" in twins

    assert isinstance(twins["p2"]["rank"], int)

    assert twins["p3"]["rank"] is None
    assert twins["p3"]["adp"] is None

def test_stat_line_qb_format_with_rush():
    row = {"completions": 18, "attempts": 25, "passing_yards": 245,
           "passing_tds": 2, "passing_interceptions": 1,
           "carries": 5, "rushing_yards": 32}
    assert _stat_line(row, "QB") == "18/25, 245 yds, 2 TD, 1 INT · 5 car, 32 yds"

def test_stat_line_qb_format_no_rush():
    row = {"completions": 20, "attempts": 30, "passing_yards": 300,
           "passing_tds": 3, "passing_interceptions": 0, "carries": 0}
    assert _stat_line(row, "QB") == "20/30, 300 yds, 3 TD, 0 INT"

def test_stat_line_rb_format_with_receiving():
    row = {"carries": 20, "rushing_yards": 110, "rushing_tds": 1,
           "targets": 3, "receptions": 2, "receiving_yards": 15}
    assert _stat_line(row, "RB") == "20 car, 110 yds, 1 TD · 2 rec, 15 yds"

def test_stat_line_rb_format_no_targets():
    row = {"carries": 15, "rushing_yards": 60, "rushing_tds": 0, "targets": 0}
    assert _stat_line(row, "RB") == "15 car, 60 yds, 0 TD"

def _kicker_weekly_rows():
    return pd.DataFrame([
        {"player_id": "k1", "player_display_name": "Foot Guy", "position": "K",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 0, "receiving_yards": 0, "receiving_tds": 0, "targets": 0,
         "carries": 0}
        for w in range(1, 11)])

def test_build_profile_kicker_has_no_history(tmp_path):
    conn = get_conn(str(tmp_path / "k.duckdb"))
    write_table(conn, "weekly", _kicker_weekly_rows())
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Foot Guy", "position": "K", "team": "DET", "adp": 150.0}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "snap_counts", pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))

    prof = build_profile(conn, "k1")
    assert prof["header"]["position"] == "K"
    # Kickers have real weekly rows (unlike rookies), but the PPR formula
    # doesn't score kicking stats, so every one nets 0 -- an all-zero
    # "history" is factually misleading and must collapse instead.
    assert prof["seasons"] == []
    assert prof["game_log"] == []
