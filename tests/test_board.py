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
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp", "espn_ppr_rank"]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))
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
    # single-source seed (FFC only): star's FFC adp (5.1) is lowest -> rank 1
    assert star["market_rank"] == 1.0
    assert star["market_sources"]["ffc"] == 1.0
    assert not star["rookie"]
    rook = board[board["name"] == "Rookie Guy"].iloc[0]
    assert rook["rookie"] and rook["production"] == 50.0
    assert list(board["rank"]) == sorted(board["rank"].tolist())
    for col in ["vor", "tier", "composite", "edge", "drafted", "bye",
                "market_rank", "market_spread", "market_sources"]:
        assert col in board.columns
    assert "adp" not in board.columns

def test_board_column_contract(tmp_path):
    board = build_board(_seed(tmp_path))
    expected = ["player_id", "name", "position", "team", "bye", "production",
                "durability", "role", "environment", "schedule", "composite",
                "vor", "tier", "market_rank", "market_spread", "market_sources",
                "edge", "rookie", "drafted", "rank"]
    assert list(board.columns) == expected
    assert board["rank"].tolist() == list(range(1, len(board) + 1))

def test_empty_database_does_not_crash(tmp_path):
    conn = get_conn(str(tmp_path / "empty.duckdb"))
    board = build_board(conn)
    assert board.empty
    for col in ["player_id", "name", "position", "team", "bye", "production",
                "durability", "role", "environment", "schedule", "composite",
                "vor", "tier", "market_rank", "market_spread", "market_sources",
                "edge", "rookie", "drafted", "rank"]:
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
    # market_sources.ffc is not None -> the PK/K alias let the FFC adp join
    assert kickers.iloc[0]["market_sources"]["ffc"] is not None
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
    # market_sources.ffc is not None -> the LAR/LA alias let the FFC adp join
    assert dst["market_sources"]["ffc"] is not None
    assert dst["team"] == "LA"
    assert dst["environment"] != 50.0  # real environment score, not the neutral default
    assert dst["rookie"] == False  # noqa: E712 -- rookie is meaningless noise for DST
    assert dst["production"] == 50.0
    assert dst["durability"] == 50.0
    assert dst["role"] == 50.0
    assert dst["schedule"] == 50.0

def test_kdst_never_flagged_rookie_even_when_adp_only(tmp_path):
    # `rookie` means "skill player with no NFL history." K/DST enter the
    # universe purely from ADP by design (never from weekly), so the naive
    # "no rows in weekly" rule would mark every single one rookie=True --
    # meaningless noise on the badge. Confirm the K/DST override applies
    # even when there is no weekly row to match at all (unlike the PK/LAR
    # alias tests above, where the K happens to also exist in weekly).
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
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Brand New Kicker", "position": "PK", "team": "DET", "adp": 160.0},
        {"adp_name": "Detroit Defense", "position": "DST", "team": "DET", "adp": 130.8},
    ]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    board = build_board(conn)
    k_row = board[board["name"] == "Brand New Kicker"].iloc[0]
    dst_row = board[board["position"] == "DST"].iloc[0]
    assert k_row["rookie"] == False  # noqa: E712
    assert dst_row["rookie"] == False  # noqa: E712

def test_adp_dedupe_keeps_lowest_adp(tmp_path):
    # A duplicate ADP row for the same normalized (name, position) must not
    # silently duplicate the corresponding board row; the best-known (lowest)
    # ADP value wins. A third, ADP-only "Marker" player at adp=16 (between the
    # two dupe candidates) makes the winning value observable via market_rank
    # ordering, since raw adp is no longer exposed on the board.
    conn = get_conn(str(tmp_path / "t.duckdb"))
    weekly = pd.DataFrame([
        {"player_id": "p1", "player_display_name": "Dupe Player", "position": "WR",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 5, "receiving_yards": 50, "targets": 8, "carries": 0}
        for w in range(1, 18)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 45.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Dupe Player", "position": "WR", "team": "DET", "adp": 12.0},
        {"adp_name": "Dupe Player", "position": "WR", "team": "DET", "adp": 20.0},
        {"adp_name": "Marker Player", "position": "WR", "team": "GB", "adp": 16.0},
    ]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    board = build_board(conn)
    matches = board[board["player_id"] == "p1"]
    assert len(matches) == 1
    marker = board[board["name"] == "Marker Player"].iloc[0]
    # If dedupe kept 12.0 (not 20.0), Dupe Player ranks ahead of Marker (16.0)
    assert matches.iloc[0]["market_rank"] < marker["market_rank"]

def test_adp_dedupe_dst_keeps_lowest_adp(tmp_path):
    # Same guarantee for DST, which joins on team rather than (norm, position).
    # A third, ADP-only "Marker" player at adp=115 (between the two dupe
    # candidates) makes the winning value observable via market_rank ordering.
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
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Detroit Defense", "position": "DST", "team": "DET", "adp": 100.0},
        {"adp_name": "Detroit D/ST", "position": "DST", "team": "DET", "adp": 130.8},
        {"adp_name": "Marker Player", "position": "WR", "team": "GB", "adp": 115.0},
    ]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    board = build_board(conn)
    dst_rows = board[board["position"] == "DST"]
    assert len(dst_rows) == 1
    marker = board[board["name"] == "Marker Player"].iloc[0]
    # If dedupe kept 100.0 (not 130.8), Detroit DST ranks ahead of Marker (115.0)
    assert dst_rows.iloc[0]["market_rank"] < marker["market_rank"]

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
    # market_sources.ffc is not None -> accent folding let the FFC adp join
    assert matches.iloc[0]["market_sources"]["ffc"] is not None

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
