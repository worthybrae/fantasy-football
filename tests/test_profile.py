import pandas as pd
from pipeline.db import get_conn, write_table
from scoring.profile import season_summaries, game_log, build_profile

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
