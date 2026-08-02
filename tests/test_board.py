import pandas as pd
from pipeline.db import get_conn, write_table
from scoring.board import build_board, _norm_name

def _seed(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    weekly = pd.DataFrame([
        {"player_id": "p1", "player_display_name": "Amon-Ra St. Brown", "position": "WR",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 8, "receiving_yards": 90, "targets": 10, "carries": 0}
        for w in range(1, 18)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "Rookie Guy", "position": "WR", "team": "GB", "adp": 90.0}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    return conn

def test_norm_name():
    assert _norm_name("Amon-Ra St. Brown") == _norm_name("Amon-Ra St Brown")
    assert _norm_name("Odell Beckham Jr.") == _norm_name("odell beckham")

def test_norm_name_folds_accents():
    # ADP feed and nflverse weekly data disagree on transliteration for some
    # players (e.g. real-world "Eddy Piñeiro" vs "Eddy Pineiro"); without
    # accent folding these would join as two different people.
    assert _norm_name("Eddy Piñeiro") == _norm_name("Eddy Pineiro")

def test_board_shape_and_join(tmp_path):
    board = build_board(_seed(tmp_path))
    star = board[board["player_id"] == "p1"].iloc[0]
    assert star["adp"] == 5.1               # ADP joined despite punctuation differences
    assert not star["rookie"]
    rook = board[board["name"] == "Rookie Guy"].iloc[0]
    assert rook["rookie"] and rook["production"] == 50.0
    assert list(board["rank"]) == sorted(board["rank"].tolist())
    for col in ["vor", "tier", "composite", "edge", "drafted", "bye"]:
        assert col in board.columns

def test_board_column_contract(tmp_path):
    board = build_board(_seed(tmp_path))
    expected = ["player_id", "name", "position", "team", "bye", "production",
                "durability", "role", "environment", "schedule", "composite",
                "vor", "tier", "adp", "edge", "rookie", "drafted", "rank"]
    assert list(board.columns) == expected
    assert board["rank"].tolist() == list(range(1, len(board) + 1))

def test_empty_database_does_not_crash(tmp_path):
    conn = get_conn(str(tmp_path / "empty.duckdb"))
    board = build_board(conn)
    assert board.empty
    for col in ["player_id", "name", "position", "team", "bye", "production",
                "durability", "role", "environment", "schedule", "composite",
                "vor", "tier", "adp", "edge", "rookie", "drafted", "rank"]:
        assert col in board.columns

def test_adp_position_alias_pk_matches_k(tmp_path):
    # Real-world ADP feed labels kickers "PK" while weekly/factors use "K";
    # without normalizing, an established kicker would be duplicated as a
    # "new" (falsely rookie) player instead of joining the existing weekly row.
    conn = get_conn(str(tmp_path / "t.duckdb"))
    weekly = pd.DataFrame([
        {"player_id": "k1", "player_display_name": "Some Kicker", "position": "K",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 0, "receiving_yards": 0, "targets": 0, "carries": 0}
        for w in range(1, 18)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 45.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Some Kicker", "position": "PK", "team": "DET", "adp": 150.0},
    ]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    board = build_board(conn)
    kickers = board[board["player_id"] == "k1"]
    assert len(kickers) == 1
    assert kickers.iloc[0]["adp"] == 150.0
    assert not kickers.iloc[0]["rookie"]
    # K factors are forced neutral except environment
    assert kickers.iloc[0]["production"] == 50.0
    assert kickers.iloc[0]["durability"] == 50.0
    assert kickers.iloc[0]["role"] == 50.0
    assert kickers.iloc[0]["schedule"] == 50.0

def test_adp_team_alias_lar_matches_la(tmp_path):
    # Real-world ADP feed codes the Rams "LAR" while schedules/weekly use "LA";
    # without normalizing, a Rams DST would fail to pick up its environment score.
    conn = get_conn(str(tmp_path / "t.duckdb"))
    weekly = pd.DataFrame([
        {"player_id": "p1", "player_display_name": "Some WR", "position": "WR",
         "recent_team": "SF", "opponent_team": "LA", "season": 2025, "week": w,
         "receptions": 5, "receiving_yards": 50, "targets": 8, "carries": 0}
        for w in range(1, 18)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "LA", "away_team": "SF", "week": 1,
         "total_line": 50.0, "spread_line": 2.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "LA Rams Defense", "position": "DST", "team": "LAR", "adp": 130.0},
    ]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    board = build_board(conn)
    dst = board[board["position"] == "DST"].iloc[0]
    assert dst["adp"] == 130.0
    assert dst["team"] == "LA"
    assert dst["environment"] != 50.0  # real environment score, not the neutral default

def test_accented_name_joins_to_single_player(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    weekly = pd.DataFrame([
        {"player_id": "k1", "player_display_name": "Eddy Pineiro", "position": "K",
         "recent_team": "SF", "opponent_team": "LA", "season": 2025, "week": w,
         "receptions": 0, "receiving_yards": 0, "targets": 0, "carries": 0}
        for w in range(1, 18)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "LA", "away_team": "SF", "week": 1,
         "total_line": 45.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Eddy Piñeiro", "position": "PK", "team": "SF", "adp": 156.8},
    ]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    board = build_board(conn)
    matches = board[board["player_id"] == "k1"]
    assert len(matches) == 1
    assert matches.iloc[0]["adp"] == 156.8

def test_depth_charts_real_schema_pos_rank(tmp_path):
    # Real nflverse depth_charts pull has pos_rank/gsis_id, not the old
    # depth_team/formation/week columns; role_factor should still work
    # via the pos_rank -> depth_team adapter in build_board.
    conn = get_conn(str(tmp_path / "t.duckdb"))
    weekly = pd.DataFrame([
        {"player_id": "p1", "player_display_name": "Some WR", "position": "WR",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 5, "receiving_yards": 50, "targets": 8, "carries": 0}
        for w in range(1, 18)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 45.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame(columns=["adp_name", "position", "team", "adp"]))
    write_table(conn, "depth_charts", pd.DataFrame([
        {"team": "DET", "gsis_id": "p1", "pos_abb": "WR", "pos_slot": 1, "pos_rank": 1},
    ]))
    board = build_board(conn)  # must not raise KeyError('depth_team')
    row = board[board["player_id"] == "p1"].iloc[0]
    assert row["role"] > 50.0  # starter (rank 1) beats the neutral default
