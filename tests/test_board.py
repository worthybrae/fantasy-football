import pandas as pd
import pytest
from pipeline.db import get_conn, write_table
from scoring.board import build_board, _norm_name

def _seed(tmp_path, include_qb=False):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    rows = [
        {"player_id": "p1", "player_display_name": "Amon-Ra St. Brown", "position": "WR",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 8, "receiving_yards": 90, "targets": 10, "carries": 0}
        for w in range(1, 18)]
    if include_qb:
        # Exercises _PASS_COLS' real-column (pd.to_numeric) branch in
        # _latest_season_stats -- the plain _seed() fixture above has no
        # passing columns at all, so that branch only runs when this flag
        # is set.
        rows += [
            {"player_id": "q1", "player_display_name": "Some QB", "position": "QB",
             "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
             "completions": 20, "attempts": 30, "passing_yards": 250,
             "passing_tds": 2, "passing_interceptions": 1, "carries": 0}
            for w in range(1, 4)]
    weekly = pd.DataFrame(rows)
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "Rookie Guy", "position": "WR", "team": "GB", "adp": 90.0}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "espn_adp", pd.DataFrame([
        {"espn_id": 501, "espn_name": "Amon-Ra St Brown", "position": "WR",
         "espn_adp": 1.0, "espn_ppr_rank": 2}]))
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
    # star: FFC adp (5.1) is lowest -> FFC rank 1; ESPN PPR rank 2 is a
    # consensus input too (the PPR-native ESPN component) -> mean 1.5
    assert star["market_rank"] == 1.5
    assert star["market_sources"]["ffc"] == 1.0
    assert star["market_sources"]["espn"] == 2.0
    assert not star["rookie"]
    # espn_ppr_rank still flows through the name-fallback join path (no
    # sleeper crosswalk in this fixture) as its own column.
    assert star["espn_ppr_rank"] == 2.0
    rook = board[board["name"] == "Rookie Guy"].iloc[0]
    assert pd.isna(rook["espn_ppr_rank"])  # no ESPN row for this player
    assert rook["rookie"] and rook["production"] == 50.0
    assert list(board["rank"]) == sorted(board["rank"].tolist())
    for col in ["vor", "tier", "composite", "edge", "drafted", "bye",
                "market_rank", "market_spread", "market_sources", "espn_ppr_rank", "stats"]:
        assert col in board.columns
    assert "adp" not in board.columns

def test_board_column_contract(tmp_path):
    # Task 13 deliberately extends the contract with exactly avail_pct, ev,
    # ev_se (sim output) -- see test_board_sim_columns_are_null_without_a_sim
    # for the null-by-default guarantee this list alone doesn't cover.
    # Task 9 adds `ffc_rank`: `draft_sim.build_pool` ranks the simulator's
    # pool on it so the simulator and the fit read one source (see
    # `scoring.market._DROP_COLS`), which means it has to survive onto the
    # board rather than being dropped with the other consensus inputs.
    board = build_board(_seed(tmp_path))
    expected = ["player_id", "name", "position", "team", "bye", "production",
                "durability", "role", "environment", "schedule", "composite",
                # Task 1: proj_points, projected season points, is what `vor`
                # now differences (cross-position comparable, unlike the
                # within-position composite percentile it replaced).
                "proj_points",
                # The scoring-format work adds proj_scale: how much this
                # league's rules re-price ESPN's PPR-only season projection
                # for this player (1.0 in a PPR league). It is on the board,
                # not private to projections(), because scoring/profile.py's
                # `summary.proj_ppg` and api/live.py's trending icon derive
                # numbers from the same raw espn_proj and must use the same
                # factor. See scoring/board.projection_scale.
                "proj_scale",
                "vor", "tier", "market_rank", "market_spread", "market_sources",
                # espn_id rides onto the board so a live draft pick, which
                # arrives as an ESPN player id and nothing else, resolves by
                # exact lookup instead of a name match under a 30s clock.
                "espn_ppr_rank", "espn_id", "ffc_rank", "edge", "rookie",
                "drafted", "rank",
                "stats", "avail_pct", "ev", "ev_se"]
    assert list(board.columns) == expected
    assert board["rank"].tolist() == list(range(1, len(board) + 1))

def test_empty_database_does_not_crash(tmp_path):
    conn = get_conn(str(tmp_path / "empty.duckdb"))
    board = build_board(conn)
    assert board.empty
    for col in ["player_id", "name", "position", "team", "bye", "production",
                "durability", "role", "environment", "schedule", "composite",
                "vor", "tier", "market_rank", "market_spread", "market_sources",
                "espn_ppr_rank", "edge", "rookie", "drafted", "rank", "stats"]:
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

def test_traded_player_gets_current_team_from_latest_depth_chart(tmp_path):
    # A player's team must come from the newest depth-chart snapshot, not
    # from last season's weekly stats -- offseason trades (e.g. a DET player
    # moving to HOU) are invisible to weekly data until games are played.
    conn = get_conn(str(tmp_path / "t.duckdb"))
    weekly = pd.DataFrame([
        {"player_id": "p1", "player_display_name": "Traded Back", "position": "RB",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 3, "receiving_yards": 20, "targets": 4, "carries": 15}
        for w in range(1, 18)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 45.0, "spread_line": 3.0},
        {"home_team": "HOU", "away_team": "TEN", "week": 1,
         "total_line": 48.0, "spread_line": 6.0}]))
    write_table(conn, "adp", pd.DataFrame(
        [{"adp_name": "Traded Back", "position": "RB", "team": "HOU", "adp": 60.0}]))
    write_table(conn, "depth_charts", pd.DataFrame([
        {"dt": "2026-07-01T09:00:00Z", "team": "DET", "gsis_id": "p1",
         "pos_abb": "RB", "pos_slot": 1, "pos_rank": 1},
        {"dt": "2026-08-02T09:00:00Z", "team": "HOU", "gsis_id": "p1",
         "pos_abb": "RB", "pos_slot": 1, "pos_rank": 1},
    ]))
    row = build_board(conn).set_index("player_id").loc["p1"]
    assert row["team"] == "HOU"  # newest snapshot wins over both weekly and older snapshots

def test_traded_player_falls_back_to_adp_team_without_depth_chart(tmp_path):
    # If the player has no depth-chart row, the ADP feed's team is still
    # more current than last season's weekly stats.
    conn = get_conn(str(tmp_path / "t.duckdb"))
    weekly = pd.DataFrame([
        {"player_id": "p1", "player_display_name": "Traded Back", "position": "RB",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 3, "receiving_yards": 20, "targets": 4, "carries": 15}
        for w in range(1, 18)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "HOU", "away_team": "TEN", "week": 1,
         "total_line": 48.0, "spread_line": 6.0}]))
    write_table(conn, "adp", pd.DataFrame(
        [{"adp_name": "Traded Back", "position": "RB", "team": "HOU", "adp": 60.0}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["dt", "team", "gsis_id", "pos_abb", "pos_slot", "pos_rank"]))
    row = build_board(conn).set_index("player_id").loc["p1"]
    assert row["team"] == "HOU"

def test_old_seasons_do_not_affect_board_factors(tmp_path):
    # HISTORY_SEASONS now reaches back to 2016 for profiles/stat twins, but
    # the board must score on the RECENCY_WEIGHTS window only -- without the
    # filter, durability_factor would count a decade of "possible games" and
    # production would see seasons the recency weights zero out anyway.
    # Two RBs, else normalize_within_position collapses any raw change to the
    # same percentile and the assertion can't see the difference.
    recent = pd.DataFrame([
        {"player_id": pid, "player_display_name": name, "position": "RB",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 3, "receiving_yards": 20, "targets": 4, "carries": 15}
        for pid, name in [("p1", "Old Vet"), ("p2", "Steady Guy")]
        for w in range(1, 18)])
    ancient = pd.DataFrame([
        {"player_id": "p1", "player_display_name": "Old Vet", "position": "RB",
         "recent_team": "CHI", "opponent_team": "MIN", "season": 2016, "week": w,
         "receptions": 10, "receiving_yards": 150, "targets": 12, "carries": 25}
        for w in range(1, 18)])
    boards = {}
    for label, weekly in [("recent_only", recent), ("with_ancient", pd.concat([recent, ancient]))]:
        conn = get_conn(str(tmp_path / f"{label}.duckdb"))
        write_table(conn, "weekly", weekly)
        write_table(conn, "schedules", pd.DataFrame([
            {"home_team": "DET", "away_team": "GB", "week": 1,
             "total_line": 45.0, "spread_line": 3.0}]))
        write_table(conn, "adp", pd.DataFrame(
            [{"adp_name": "Old Vet", "position": "RB", "team": "DET", "adp": 60.0},
             {"adp_name": "Steady Guy", "position": "RB", "team": "DET", "adp": 61.0}]))
        write_table(conn, "depth_charts", pd.DataFrame(
            columns=["dt", "team", "gsis_id", "pos_abb", "pos_slot", "pos_rank"]))
        boards[label] = build_board(conn).set_index("player_id").loc["p1"]
    for col in ["production", "durability", "role", "schedule", "composite", "vor"]:
        assert boards["with_ancient"][col] == boards["recent_only"][col], col

def test_board_stats_summary(tmp_path):
    board = build_board(_seed(tmp_path))
    s = board[board["player_id"] == "p1"].iloc[0]["stats"]
    assert s["season"] == 2025 and s["games"] == 17
    assert s["receptions"] == 8 * 17
    assert s["rec_yards"] == 90 * 17
    assert s["targets"] == 10 * 17
    # PPR: 8 rec + 9.0 rec-yd pts = 17.0 per game
    assert s["ppg"] == 17.0
    assert s["points"] == 17.0 * 17
    # no passing columns in this fixture -> zero-filled, not missing
    assert s["pass_yards"] == 0 and s["attempts"] == 0
    # ADP-only player has no weekly rows -> no stats dict at all
    rook = board[board["name"] == "Rookie Guy"].iloc[0]
    assert not isinstance(rook["stats"], dict)

def test_board_stats_summary_passing_cols(tmp_path):
    # test_board_stats_summary above only exercises the "no passing columns
    # present" zero-fill path; this covers the real-column pd.to_numeric
    # aggregation branch in _latest_season_stats/_PASS_COLS.
    board = build_board(_seed(tmp_path, include_qb=True))
    qb = board[board["player_id"] == "q1"].iloc[0]["stats"]
    weeks = 3
    assert qb["games"] == weeks
    assert qb["completions"] == 20 * weeks
    assert qb["attempts"] == 30 * weeks
    assert qb["pass_yards"] == 250 * weeks
    assert qb["pass_tds"] == 2 * weeks
    assert qb["interceptions"] == 1 * weeks

def test_board_vor_is_in_projected_points(tmp_path):
    """`vor`'s exact arithmetic: it differences `proj_points` at the
    position's replacement rank, not `composite`.

    This runs on `_seed`, the file's real fixture, which only ever produces
    WR rows (one real, one ADP-only) -- so it can confirm the formula
    (`vor == proj_points - replacement`) but, with a single position present,
    it CANNOT catch a cross-position ranking regression: sorting one
    position's rows by any monotonic function of itself is trivially sorted.
    `test_board_ranks_by_points_not_by_composite_percentile_across_positions`
    below is the fixture built specifically to catch that (the actual Mark
    Andrews case) -- see its docstring.
    """
    conn = _seed(tmp_path)
    board = build_board(conn)
    assert "proj_points" in board.columns
    # vor is a points difference, so it moves on the same scale as proj_points
    wrs = board[board["position"] == "WR"]
    assert (wrs["vor"] - (wrs["proj_points"] - wrs["proj_points"].nlargest(
        24).iloc[-1])).abs().max() < 1e-6
    # and the board is sorted by it
    assert board["vor"].is_monotonic_decreasing

def test_board_ranks_by_points_not_by_composite_percentile_across_positions(tmp_path):
    """Regression for the Mark Andrews case: with VOR built from composite (a
    within-position percentile), a TE at ADP 149 ranked 13th overall because
    being far above TE10 in percentile scored the same as being far above
    RB22 in percentile -- comparing two numbers with no shared unit.

    `_seed` only ever produces one position (see the test above), which
    can't exercise this: a within-position ranking sorted by any monotonic
    function of itself is trivially sorted, whether that function is
    `composite` or `proj_points`. This builds a dedicated two-position (RB,
    TE), two-player-per-position fixture where the two facts are engineered
    to point opposite ways:

      - `te_star` OUT-RANKS `rb_good` on `composite`: both dominate their own
        2-player position group on production/role/schedule, but `rb_good`'s
        team is deliberately given a low schedules-implied environment score
        against `rb_scrub`'s high one (`environment_factor` is purely a
        team-level number, decoupled from either player's own production),
        which is the one factor `rb_good` loses and `te_star` wins -- pulling
        `rb_good`'s composite (0.20 weight on environment) below `te_star`'s.
      - `rb_good` OUT-SCORES `te_star` on `proj_points`: real rushing volume
        (20 carries/120 yards per game) outproduces `te_star`'s modest real
        receiving line (3 rec/30 yards per game) in points, independent of
        either player's within-position percentile.

    So if `apply_vor` (or `build_board`'s call into it) ever reverts to
    `column="composite"`, `te_star` -- ranked below `rb_good` today -- would
    rank above him again, and this test would catch it. Verified against the
    reverted code path directly, not just asserted: differencing this
    fixture's own `composite` column the way the old code did gives
    vor(te_star) = 91.25 - 58.75 = 32.5 > vor(rb_good) = 81.25 - 68.75 =
    12.5, i.e. exactly the wrong order this test guards against.
    """
    conn = get_conn(str(tmp_path / "cross_position.duckdb"))
    rows = []
    for w in range(1, 18):
        rows += [
            {"player_id": "te_star", "player_display_name": "Star TE", "position": "TE",
             "recent_team": "KC", "opponent_team": "LV", "season": 2025, "week": w,
             "receptions": 3, "receiving_yards": 30, "targets": 4, "carries": 0},
            {"player_id": "te_scrub", "player_display_name": "Scrub TE", "position": "TE",
             "recent_team": "LV", "opponent_team": "KC", "season": 2025, "week": w,
             "receptions": 0, "receiving_yards": 0, "targets": 1, "carries": 0},
            {"player_id": "rb_good", "player_display_name": "Good RB", "position": "RB",
             "recent_team": "DAL", "opponent_team": "NYG", "season": 2025, "week": w,
             "receptions": 0, "receiving_yards": 0, "targets": 0, "carries": 20,
             "rushing_yards": 120},
            {"player_id": "rb_scrub", "player_display_name": "Scrub RB", "position": "RB",
             "recent_team": "NYG", "opponent_team": "DAL", "season": 2025, "week": w,
             "receptions": 0, "receiving_yards": 0, "targets": 0, "carries": 1,
             "rushing_yards": 2},
        ]
    write_table(conn, "weekly", pd.DataFrame(rows))
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "KC", "away_team": "LV", "week": 1,
         "total_line": 48.0, "spread_line": 3.0},
        # NYG home (not DAL): implied points = (total_line +/- spread) / 2,
        # so the home team gets the higher number. rb_good's team (DAL) must
        # land BELOW rb_scrub's team (NYG) on environment for the inversion
        # above to hold -- this is what makes that happen.
        {"home_team": "NYG", "away_team": "DAL", "week": 1,
         "total_line": 44.0, "spread_line": 6.0},
    ]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Star TE", "position": "TE", "team": "KC", "adp": 40.0},
        {"adp_name": "Scrub TE", "position": "TE", "team": "LV", "adp": 200.0},
        {"adp_name": "Good RB", "position": "RB", "team": "DAL", "adp": 20.0},
        {"adp_name": "Scrub RB", "position": "RB", "team": "NYG", "adp": 220.0},
    ]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp", "espn_ppr_rank"]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))

    board = build_board(conn).set_index("player_id")
    assert board.loc["te_star", "composite"] > board.loc["rb_good", "composite"]
    assert board.loc["rb_good", "proj_points"] > board.loc["te_star", "proj_points"]
    # The regression: vor, and therefore rank, must follow proj_points, not
    # the composite ordering above.
    assert board.loc["rb_good", "vor"] > board.loc["te_star", "vor"]
    assert board.loc["rb_good", "rank"] < board.loc["te_star", "rank"]

def test_board_uses_league_settings_when_present(tmp_path):
    from pipeline.db import read_table, write_table
    from scoring import league
    conn = _seed(tmp_path)
    # The brief's own fixture (_seed) has exactly one real weekly WR, so
    # every alternative rank/points setting collapses to the same result
    # (normalize_within_position's percentile rank flattens a lone real
    # value to 100 regardless of the scoring rules, and the WR replacement
    # index clamps to the same "last of 2" player whether the configured
    # rank is 16 or 24). A second real WR, engineered so its production
    # ranks *below* the seed's star under full PPR but *above* it under the
    # league's half-PPR scoring below, makes the derived-settings path
    # actually observable instead of merely "doesn't crash":
    #   Full PPR (rec=1.0, yds=0.1): p1 = 8+9.0=17.0/gm; p2 = 2+14.0=16.0/gm -> p1 ahead.
    #   Half PPR (rec=0.5, yds=0.1): p1 = 4+9.0=13.0/gm; p2 = 1+14.0=15.0/gm -> p2 ahead.
    extra = pd.DataFrame(
        [{"player_id": "p2", "player_display_name": "Other WR", "position": "WR",
          "recent_team": "GB", "opponent_team": "DET", "season": 2025, "week": w,
          "receptions": 2, "receiving_yards": 140, "targets": 6, "carries": 0}
         for w in range(1, 18)])
    write_table(conn, "weekly", pd.concat([read_table(conn, "weekly"), extra], ignore_index=True))
    settings = league.LeagueSettings(
        season=2026, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=0, bench=5,
        scoring={"receptions": 0.5, "receiving_yards": 0.1},
        draft_type="SNAKE")
    write_table(conn, "league", pd.DataFrame(
        [{"season": 2026, "settings_json": league.to_json(settings)}]))
    board = build_board(conn)
    star = board[board["player_id"] == "p1"].iloc[0]
    other = board[board["player_id"] == "p2"].iloc[0]
    # Under the league's half-PPR scoring, "Other WR" (fewer catches, more
    # yards) out-produces "Star WR" -- the reverse of the full-PPR default
    # computed in the comment above (under DEFAULT_RULES, p1 leads 17.0 to
    # 16.0/gm). This flip can only happen if settings.scoring, not the
    # hardcoded DEFAULT_RULES, drove production_factor -- and it propagates
    # through composite.
    assert other["production"] > star["production"]
    assert other["composite"] > star["composite"]
    # `vor` follows the flip too now. It differences `proj_points`, and both
    # rungs of `projections()` are priced in this league's scoring: this
    # fixture's espn_adp carries no `espn_proj` column at all, so every row
    # falls to `stats.ppg * GAMES`, and `stats.ppg` is the league's points.
    #   p1: 8 x 0.5 + 90 x 0.1 = 13.0/gm -> 221.0
    #   p2: 2 x 0.5 + 140 x 0.1 = 15.0/gm -> 255.0
    # Under full PPR it is 17.0 -> 289.0 against 16.0 -> 272.0, the other way
    # round. This assertion USED to say the opposite, and carried a paragraph
    # explaining that `projections()` priced every league in fixed full PPR
    # so the star kept the higher vor while losing on production -- a board
    # that contradicted itself. That gap is what this change closed.
    assert star["proj_points"] == 13.0 * 17
    assert other["proj_points"] == 15.0 * 17
    assert other["vor"] > star["vor"]
    assert other["rank"] < star["rank"]

def test_board_without_league_table_is_unchanged(tmp_path):
    # Same fixture, no `league` table -> the pre-existing expectations hold.
    board = build_board(_seed(tmp_path))
    star = board[board["player_id"] == "p1"].iloc[0]
    assert star["market_rank"] == 1.5

def test_board_sim_columns_are_null_without_a_sim(tmp_path):
    board = build_board(_seed(tmp_path))
    for col in ("avail_pct", "ev", "ev_se"):
        assert col in board.columns
        assert board[col].isna().all()

def test_board_merges_sim_results_when_present(tmp_path):
    from pipeline.db import write_table
    conn = _seed(tmp_path)
    write_table(conn, "sim_survival", pd.DataFrame(
        [{"run_id": "r1", "player_id": "p1", "avail_pct": 0.42}]))
    write_table(conn, "sim_results", pd.DataFrame(
        [{"run_id": "r1", "player_id": "p1", "ev": 1580.5, "se": 4.2,
          "rank": 1, "my_slot": 4}]))
    board = build_board(conn)
    row = board[board["player_id"] == "p1"].iloc[0]
    assert row["avail_pct"] == 0.42
    assert row["ev"] == 1580.5
    assert row["ev_se"] == 4.2

def test_board_sim_columns_null_for_player_outside_sim_population(tmp_path):
    """A sim only ever covers the pool it was run against (e.g. candidates
    considered at one pick) -- a player absent from sim_survival/sim_results
    entirely (here "Rookie Guy", the seed's ADP-only player) must land as a
    left-merge miss (NaN/null), not error or silently inherit another row's
    value, once a sim has run for someone else."""
    from pipeline.db import write_table
    conn = _seed(tmp_path)
    write_table(conn, "sim_survival", pd.DataFrame(
        [{"run_id": "r1", "player_id": "p1", "avail_pct": 0.42}]))
    write_table(conn, "sim_results", pd.DataFrame(
        [{"run_id": "r1", "player_id": "p1", "ev": 1580.5, "se": 4.2,
          "rank": 1, "my_slot": 4}]))
    board = build_board(conn)
    row = board[board["name"] == "Rookie Guy"].iloc[0]
    assert pd.isna(row["avail_pct"])
    assert pd.isna(row["ev"])
    assert pd.isna(row["ev_se"])

def test_board_sim_merge_does_not_multiply_rows_on_a_duplicate_player_id(tmp_path):
    """The merge has to be safe structurally, not just because run_sim
    happens to write one row per player.

    `_add_adp_only_players` synthesizes `player_id = "adp_" + norm` with no
    position in the key, so one normalized name at two positions in the ADP
    feed (neither matching the weekly universe) already yields two board rows
    with the same id. A many-to-many merge against a sim table would then
    turn 2 rows into 4, silently duplicating players on the board.
    """
    from pipeline.db import write_table
    conn = _seed(tmp_path)
    before = len(build_board(conn))
    write_table(conn, "sim_survival", pd.DataFrame([
        {"run_id": "r1", "player_id": "p1", "avail_pct": 0.42},
        {"run_id": "r1", "player_id": "p1", "avail_pct": 0.11}]))
    write_table(conn, "sim_results", pd.DataFrame([
        {"run_id": "r1", "player_id": "p1", "ev": 1580.5, "se": 4.2,
         "rank": 1, "my_slot": 4},
        {"run_id": "r1", "player_id": "p1", "ev": 1.0, "se": 0.1,
         "rank": 2, "my_slot": 4}]))

    board = build_board(conn)
    assert len(board) == before
    assert board["player_id"].tolist().count("p1") == 1


def test_adp_match_key_returns_none_for_a_missing_position():
    # A draft pick whose ESPN player id is absent from that season's directory
    # arrives with position/name/team all NA. `NA == "DST"` is NA, whose truth
    # value raises -- validate_import crashed on a real import because of it.
    from scoring.board import adp_match_key
    assert adp_match_key(pd.NA, pd.NA, pd.NA) is None
    assert adp_match_key(None, None) is None
    assert adp_match_key("Justin Jefferson", "WR") is not None


# -- scoring-format-aware consensus, end to end ----------------------------

def _settings_std():
    """A standard-scoring (receptions 0) LeagueSettings for build_board --
    scoring_format reads it as 'std', so the consensus must select each
    source's 'std' rows (falling back to 'ppr' where a source has none)."""
    from dataclasses import replace
    from scoring import league
    base = league.default_settings()
    return replace(base, scoring={**base.scoring, "receptions": 0.0})


def test_board_market_rank_unchanged_none_vs_explicit_ppr_settings(tmp_path):
    # Passing an explicit PPR league must not move any player's market_rank
    # off what settings=None (default PPR) produces. Row ORDER can differ
    # (VOR depends on the scoring rules, and the ESPN-parsed league scores a
    # hair differently than DEFAULT_RULES), so compare per player_id, not
    # positionally -- market_rank is consensus-ADP, independent of scoring.
    from tests.test_league import _settings
    conn = _seed(tmp_path)
    a = build_board(conn).set_index("player_id")["market_rank"]
    b = build_board(conn, settings=_settings()).set_index("player_id")["market_rank"]
    assert a.sort_index().round(3).equals(b.sort_index().round(3))


def test_board_ppr_is_byte_identical_when_sources_carry_a_format_column(tmp_path):
    # The most important invariant: for a PPR league the format machinery must
    # reproduce today's board exactly, even once the source tables carry a
    # `format` column (and extra non-PPR rows). Baseline board has no `format`
    # column anywhere; the second seeds the FFC/adp table with format-tagged
    # PPR rows identical to the baseline PLUS a decoy std row a PPR league must
    # drop. The two boards must match on every consensus column.
    plain = _seed(tmp_path)
    fmt = _seed(tmp_path / "fmt")
    write_table(fmt, "adp", pd.DataFrame([
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET", "adp": 5.1, "format": "ppr"},
        {"adp_name": "Rookie Guy", "position": "WR", "team": "GB", "adp": 90.0, "format": "ppr"},
        # decoy: a standard-format row the PPR build must ignore.
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET", "adp": 40.0, "format": "std"},
    ]))
    a = build_board(plain).set_index("player_id")
    b = build_board(fmt).set_index("player_id")
    assert list(a.index) == list(b.index)
    assert a["market_rank"].tolist() == b["market_rank"].tolist()
    assert a["ffc_rank"].tolist() == b["ffc_rank"].tolist()
    assert list(a["market_sources"]) == list(b["market_sources"])


def test_board_consensus_follows_the_league_scoring_format(tmp_path):
    # A source that publishes per-format rows must vote its format's row. MFL
    # ranks the star 1 in PPR and 25 in standard; the PPR board's consensus
    # must land on 1, the standard board's on 25 -- moving market_rank with it.
    conn = _seed(tmp_path)
    write_table(conn, "mfl_adp", pd.DataFrame([
        {"mfl_name": "Amon-Ra St Brown", "position": "WR", "mfl_rank": 1, "format": "ppr"},
        {"mfl_name": "Amon-Ra St Brown", "position": "WR", "mfl_rank": 25, "format": "std"},
    ]))
    ppr = build_board(conn).set_index("player_id")            # settings=None -> ppr
    std = build_board(conn, settings=_settings_std()).set_index("player_id")
    # star: ffc 1, espn 2, mfl 1 -> median([1,1,2]) = 1.0 in PPR;
    #       ffc 1, espn 2, mfl 25 -> median([1,2,25]) = 2.0 in standard.
    assert ppr.loc["p1", "market_sources"]["mfl"] == 1.0
    assert std.loc["p1", "market_sources"]["mfl"] == 25.0
    assert ppr.loc["p1", "market_rank"] == 1.0
    assert std.loc["p1", "market_rank"] == 2.0


# -- scoring-format awareness of proj_points (scoring.board.projection_scale) --
#
# The board's own `production`/`composite` have followed `settings.scoring`
# since the league-settings work, but `proj_points` -- which is what `vor`,
# and therefore the whole ranking, differences -- did not: its fallback rung
# read a fixed full-PPR ppg and its ESPN rung was ESPN's own PPR-only season
# number. These tests pin BOTH rungs to the league's rules, and pin the one
# thing that must not move: a full-PPR league.

def _half_ppr(**overrides):
    """DEFAULT_RULES with half-point receptions -- a complete rule set, not a
    two-key sketch, so the only thing separating it from full PPR is the one
    value the format is named for."""
    from scoring import league
    from scoring.ppr import DEFAULT_RULES
    scoring = {**DEFAULT_RULES, "receptions": 0.5}
    scoring.update(overrides)
    return league.LeagueSettings(
        season=2026, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=2, bench=5, scoring=scoring, draft_type="SNAKE")


def _seed_with_espn_projection(tmp_path):
    """`_seed` plus a real espn_proj for p1 and a second WR ESPN never
    projects, so one row exercises the ESPN rung and one the ppg fallback."""
    from pipeline.db import read_table
    conn = _seed(tmp_path)
    extra = pd.DataFrame(
        [{"player_id": "p2", "player_display_name": "Other WR", "position": "WR",
          "recent_team": "GB", "opponent_team": "DET", "season": 2025, "week": w,
          "receptions": 2, "receiving_yards": 140, "targets": 6, "carries": 0}
         for w in range(1, 18)])
    write_table(conn, "weekly", pd.concat([read_table(conn, "weekly"), extra],
                                          ignore_index=True))
    write_table(conn, "espn_adp", pd.DataFrame([
        {"espn_id": 501, "espn_name": "Amon-Ra St Brown", "position": "WR",
         "espn_adp": 1.0, "espn_ppr_rank": 2, "espn_proj": 280.0}]))
    return conn


def test_projections_price_both_rungs_in_the_leagues_scoring(tmp_path):
    """ESPN's rung is CONVERTED, the ppg rung is scored directly, and the two
    end up on one scale.

    p1 (8 rec, 90 rec yds a game) is 17.0 ppg in full PPR and 13.0 in
    half-PPR, so his proj_scale is 13/17 = 0.7647 and ESPN's 280.0 becomes
    214.1. p2 (2 rec, 140 rec yds) has no ESPN row at all and falls to
    `stats.ppg * GAMES` = 15.0 x 17 = 255.0.

    The flip is the point: in full PPR p1 leads on projected points
    (280.0 > 16.0 x 17 = 272.0) and in half-PPR p2 does (255.0 > 214.1),
    because p1's edge was 6 catches a game and half of it just stopped
    counting. Before this change p1 kept 280.0 in every league -- ESPN's
    number, in PPR, next to a fallback rung that was also in PPR.
    """
    conn = _seed_with_espn_projection(tmp_path)
    ppr = build_board(conn).set_index("player_id")
    assert ppr.loc["p1", "proj_scale"] == 1.0
    assert ppr.loc["p1", "proj_points"] == 280.0            # ESPN's own number
    assert ppr.loc["p2", "proj_points"] == 16.0 * 17        # 272.0, full-PPR ppg
    assert ppr.loc["p1", "vor"] > ppr.loc["p2", "vor"]

    half = build_board(conn, settings=_half_ppr()).set_index("player_id")
    assert half.loc["p1", "proj_scale"] == pytest.approx(13.0 / 17.0)
    assert half.loc["p1", "proj_points"] == pytest.approx(280.0 * 13.0 / 17.0)
    assert half.loc["p2", "proj_scale"] == pytest.approx(15.0 / 16.0)
    assert half.loc["p2", "proj_points"] == 15.0 * 17       # 255.0, half-PPR ppg
    assert half.loc["p2", "vor"] > half.loc["p1", "vor"]
    assert half.loc["p2", "rank"] < half.loc["p1", "rank"]


def test_projections_are_unchanged_when_the_leagues_rules_are_ppr_in_another_order(tmp_path):
    """The ULP guard, and it is not hypothetical.

    `league.from_espn` builds `scoring` in ESPN's `scoring_items` order, which
    is not DEFAULT_RULES' order -- the owner's own imported league lists the
    same 14 rules starting at `rushing_tds` instead of `passing_yards`.
    `compute_ppr_points` accumulates term by term in dict order, so the same
    rules summed in a different order differ in the last bits of a float:
    measured on data/nfl.duckdb, 478 of 54,473 weekly rows from 2023 on come
    out different, by up to 7.1e-15.

    That is invisible in a points column and very visible here, because
    `projection_scale` DIVIDES two such sums. Without `normalize_rules`
    (scoring/ppr.py) collapsing a reordered-but-identical rule set back onto
    None, the scale lands at 0.999999999999999x instead of 1.0, every
    proj_points shifts in its last bits, and two players who should have tied
    on vor can swap places -- a re-ranked board for a league whose scoring did
    not change at all. Exactly, not approximately, is the assertion.
    """
    from scoring import league
    from scoring.ppr import DEFAULT_RULES
    conn = _seed_with_espn_projection(tmp_path)
    reordered = {k: DEFAULT_RULES[k] for k in reversed(list(DEFAULT_RULES))}
    assert list(reordered) != list(DEFAULT_RULES) and reordered == DEFAULT_RULES
    settings = league.LeagueSettings(
        season=2026, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=2, bench=5, scoring=reordered, draft_type="SNAKE")

    board = build_board(conn, settings=settings).set_index("player_id")
    assert (board["proj_scale"] == 1.0).all()
    assert board.loc["p1", "proj_points"] == 280.0
    assert board.loc["p2", "proj_points"] == 16.0 * 17


def test_projection_scale_leaves_kickers_and_defenses_alone(tmp_path):
    """K and DST never move between scoring formats, and 1.0 says so.

    nflverse weekly rows carry no kicking or defensive scoring, so both
    positions total zero points under every rule set -- there is no ratio to
    take. Falling back to a cross-position median would mark a kicker down by
    a WR's reception share, which is wrong in the arithmetic and wrong in the
    football: a half-PPR league does not change what a kicker scores.
    """
    conn = _seed_with_espn_projection(tmp_path)
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "Other WR", "position": "WR", "team": "GB", "adp": 6.0},
        {"adp_name": "Some Kicker", "position": "PK", "team": "DET", "adp": 150.0},
        {"adp_name": "Lions", "position": "DST", "team": "DET", "adp": 160.0}]))
    board = build_board(conn, settings=_half_ppr()).set_index("name")
    assert board.loc["Some Kicker", "proj_scale"] == 1.0
    assert board.loc["Lions", "proj_scale"] == 1.0
    assert board.loc["Amon-Ra St. Brown", "proj_scale"] < 1.0


def test_projection_scale_falls_back_to_the_position_median(tmp_path):
    """A player with no latest-season rows takes his position's median scale.

    "Rookie Guy" enters from the ADP feed with no weekly history, so there is
    no ratio of his own to take. Left at 1.0 he would keep a full-PPR
    projection while every WR with a track record was marked down, which
    systematically favours precisely the players there is least reason to be
    confident about. The two real WRs here scale by 13/17 and 15/16, so the
    median of the two is what he gets.
    """
    import numpy as np
    conn = _seed_with_espn_projection(tmp_path)
    board = build_board(conn, settings=_half_ppr()).set_index("name")
    expected = float(np.median([13.0 / 17.0, 15.0 / 16.0]))
    assert board.loc["Rookie Guy", "proj_scale"] == pytest.approx(expected)
    assert board.loc["Rookie Guy", "proj_scale"] < 1.0


def test_projection_scale_warns_when_it_cannot_convert(tmp_path):
    """Silence is the failure mode this whole change exists to remove.

    A frame with no league_ppg/ppr_ppg columns (a bare fixture, or a board
    already narrowed to `_BOARD_COLUMNS`) cannot produce a ratio for anybody.
    Returning ones without a word would leave a non-PPR league ranked on
    ESPN's PPR projections and nothing on the page would say so.
    """
    import warnings as w
    from scoring.board import projection_scale
    bare = pd.DataFrame([{"name": "X", "position": "WR", "stats": {"ppg": 10.0}}])
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        scale = projection_scale(bare, _half_ppr().scoring)
    assert list(scale) == [1.0]
    assert any(issubclass(c.category, RuntimeWarning)
               and "ppr_ppg" in str(c.message) for c in caught)
    # ...and no warning at all for a league that really is full PPR.
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        assert list(projection_scale(bare, None)) == [1.0]
    assert not caught


def test_build_board_warns_that_the_espn_rung_is_converted_not_re_derived(tmp_path):
    """The warning had to change, not go away.

    It used to say proj_points was priced in fixed full PPR regardless of the
    league -- a description of a bug that is now fixed, which would have been
    worse than no warning. What is left is narrower and still true: ESPN
    publishes one season projection, in PPR, and `espn_adp` stores only that
    total, so converting it is an estimate rather than a re-derivation.
    """
    import warnings as w
    conn = _seed_with_espn_projection(tmp_path)
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        build_board(conn, settings=_half_ppr())
    messages = [str(c.message) for c in caught
                if issubclass(c.category, RuntimeWarning)]
    assert any("CONVERTED, not re-derived" in m for m in messages)
    assert not any("priced in fixed full PPR" in m for m in messages)

    # A full-PPR league says nothing, because there is nothing to say.
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        build_board(conn)
    assert not [c for c in caught if issubclass(c.category, RuntimeWarning)
                and "proj_points" in str(c.message)]


def test_a_kickers_factors_stay_neutral_until_the_league_prices_kicking():
    """`_neutral_factors` is the one place that decides this, and the
    condition is the league's rules rather than the position.

    Production, durability and schedule become real signal for a kicker the
    moment his kicks are worth points; `role` never does, because
    `factors.role_factor` blends depth-chart rank with share of team
    targets+carries and the second half is structurally zero for every
    kicker. A defense's release is decided by a different input entirely --
    see the DST test below -- and never by the skill-position `rules`.
    """
    from scoring.board import _neutral_factors, _NEUTRAL_FACTORS_FOR_KDST
    from scoring.ppr import DEFAULT_RULES
    kicking = {**DEFAULT_RULES, "fg_made_30_39": 3.0, "pat_made": 1.0}

    assert _neutral_factors("K", None) == _NEUTRAL_FACTORS_FOR_KDST
    assert _neutral_factors("K", DEFAULT_RULES) == _NEUTRAL_FACTORS_FOR_KDST
    assert _neutral_factors("K", kicking) == ["role"]

    # A league's kicking rules say nothing about its defenses, and with no
    # `dst_weekly` on hand a defense is exactly as neutral as it always was.
    assert _neutral_factors("DST", None) == _NEUTRAL_FACTORS_FOR_KDST
    assert _neutral_factors("DST", kicking) == _NEUTRAL_FACTORS_FOR_KDST


def test_a_defense_gets_production_back_only_once_there_is_a_history():
    """Two conditions, both required, and only ONE factor released.

    `durability` is deliberately still pinned even though `dst_weekly` could
    compute it: a defense plays every game its team plays, so games/possible
    is 17/17 for all 32 and the factor is a constant. `role` has nothing to
    compute from. `schedule` is left for separate work -- see
    `_neutral_factors`.
    """
    from scoring.board import (_neutral_factors, _NEUTRAL_FACTORS_FOR_KDST,
                               _DST_FACTORS_THAT_BECOME_REAL)
    assert _DST_FACTORS_THAT_BECOME_REAL == ["production"]
    # History present and the league prices defense (None = ESPN's defaults).
    assert _neutral_factors("DST", None, None, True) == \
        ["durability", "role", "schedule"]
    # History present but this league scores no defense at all.
    assert _neutral_factors("DST", None, {}, True) == _NEUTRAL_FACTORS_FOR_KDST
    # League prices defense but there is no history to compute from.
    assert _neutral_factors("DST", None, None, False) == _NEUTRAL_FACTORS_FOR_KDST


def test_dst_team_history_scores_a_defense_under_the_leagues_own_rules():
    """Keyed on `team`, not `player_id`: defenses reach the board from the ADP
    feed with a synthesized id, and join everything else on team already."""
    import pandas as pd
    from scoring.board import dst_team_history
    # Two teams, two seasons, one stat: 3 sacks a week at 1 point each.
    rows = [{"season": season, "week": week, "espn_id": espn_id, "team": team,
             "applied_total": 6.0, "stat_99": 3.0}
            for season, weight in ((2024, 0.3), (2025, 0.5))
            for team, espn_id in (("DEN", -16007), ("HOU", -16034))
            for week in (1, 2)]
    hist = dst_team_history(pd.DataFrame(rows), {"99": 2.0})
    assert sorted(hist["team"]) == ["DEN", "HOU"]
    row = hist[hist["team"] == "DEN"].iloc[0]
    # 6 points a week under this league's rules, every season -> 6.0 ppg.
    # approx, not ==: the recency weighting divides by 0.3 + 0.5, exactly as
    # `factors.production_factor` does for a running back.
    assert row["dst_production_raw"] == pytest.approx(6.0)
    assert row["dst_stats"]["season"] == 2025
    assert row["dst_stats"]["games"] == 2
    assert row["dst_stats"]["points"] == 12.0
    # The board's `stats` dict has one key set for every position.
    assert row["dst_stats"]["receptions"] == 0
    # ESPN scored the same weeks at 6.0 too, so the conversion factor is 1.0.
    assert row["dst_league_ppg"] == 6.0 and row["dst_espn_ppg"] == 6.0
    # A league that scores no defense gets zeros, not ESPN's defaults.
    zero = dst_team_history(pd.DataFrame(rows), {})
    assert zero[zero["team"] == "DEN"].iloc[0]["dst_production_raw"] == 0.0
    # No data at all is not an error, it is "nothing to release".
    assert dst_team_history(pd.DataFrame()).empty


def test_a_kicker_needs_most_of_a_season_before_his_ppg_is_extrapolated():
    """The regression that scoring kickers introduced, pinned.

    `projections()`'s second rung multiplies a player's per-game points by a
    full season. ESPN publishes no projection for a fringe kicker, so that
    rung became the normal path for them -- and on the real 249-row board it
    put Ben Sauls (three games) and Spencer Shrader (five) ahead of Brandon
    Aubrey's full 17-game season. K's replacement rank is 3, so those two
    undraftable kickers became the baseline every kicker was priced against
    and the whole position fell about thirty places.

    Below the threshold a kicker falls to POSITION_FLOOR, which is exactly
    where every unprojected kicker sat before kicking was scorable.
    """
    import pandas as pd
    from scoring.board import (projections, POSITION_FLOOR, GAMES,
                               _MIN_GAMES_FOR_KICKER_PPG)

    class _NoEspn:
        def execute(self, *a, **k):
            raise AssertionError("should not be reached")

    def _board(games, ppg):
        return pd.DataFrame([{
            "player_id": "k1", "name": "Kick Guy", "position": "K",
            "team": "DAL", "proj_scale": 1.0,
            "stats": {"season": 2025, "games": games, "ppg": ppg}}])

    from unittest.mock import patch
    with patch("scoring.board.read_table", return_value=pd.DataFrame()):
        short = _board(_MIN_GAMES_FOR_KICKER_PPG - 1, 11.2)
        assert projections(_NoEspn(), short).iloc[0] == POSITION_FLOOR["K"]
        full = _board(_MIN_GAMES_FOR_KICKER_PPG, 11.2)
        assert projections(_NoEspn(), full).iloc[0] == 11.2 * GAMES
        # A skill player on the same tiny sample is deliberately NOT guarded:
        # changing him would move a board the owner is drafting from.
        wr = short.assign(position="WR", name="WR Guy")
        assert projections(_NoEspn(), wr).iloc[0] == 11.2 * GAMES


def _seed_with_two_defenses(tmp_path, with_dst_weekly: bool):
    """The same board twice, differing only in whether `dst_weekly` exists.

    Two defenses that ESPN ranks identically but that played very differently,
    so anything that moves has to have come from the new table.
    """
    conn = _seed(tmp_path)
    adp = pd.DataFrame([
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "Rookie Guy", "position": "WR", "team": "GB", "adp": 90.0},
        {"adp_name": "Detroit Defense", "position": "DST", "team": "DET", "adp": 100.0},
        {"adp_name": "Green Bay Defense", "position": "DST", "team": "GB", "adp": 101.0}])
    write_table(conn, "adp", adp)
    if with_dst_weekly:
        rows = []
        for team, espn_id, sacks in (("DET", -16008, 5.0), ("GB", -16009, 1.0)):
            for season in (2024, 2025):
                for week in range(1, 18):
                    rows.append({"season": season, "week": week, "team": team,
                                 "espn_id": espn_id, "applied_total": sacks,
                                 "stat_99": sacks})
        write_table(conn, "dst_weekly", pd.DataFrame(rows))
    return conn


def test_a_board_with_no_dst_weekly_is_the_board_it_always_was(tmp_path):
    """The no-op condition, stated as a test rather than as an intention:
    every database is in this state until a refresh has run the new job."""
    before = build_board(_seed_with_two_defenses(tmp_path / "a", False))
    dst = before[before["position"] == "DST"]
    assert len(dst) == 2
    assert set(dst["production"]) == {50.0}
    assert dst["stats"].isna().all()


def test_dst_weekly_moves_defenses_and_nothing_else(tmp_path):
    """The guarantee the live board rests on, checked the only way that
    counts: build both boards and diff them, exactly.

    Defenses gain a real `production` spread and a real `stats` line. Every
    other position keeps every value it had, to the bit.
    """
    before = build_board(_seed_with_two_defenses(tmp_path / "a", False))
    after = build_board(_seed_with_two_defenses(tmp_path / "b", True))

    dst = after[after["position"] == "DST"].set_index("team")
    assert set(dst["production"]) != {50.0}
    assert dst.loc["DET", "production"] > dst.loc["GB", "production"]
    assert dst.loc["DET", "stats"]["points"] == 85.0     # 17 weeks x 5
    assert dst.loc["GB", "stats"]["points"] == 17.0
    # ESPN scored these weeks identically, so nothing is converted.
    assert set(dst["proj_scale"]) == {1.0}

    skill = ["QB", "RB", "WR", "TE", "K"]
    keep = [c for c in before.columns if c not in ("rank", "edge")]
    a = before[before["position"].isin(skill)].sort_values("player_id").reset_index(drop=True)
    b = after[after["position"].isin(skill)].sort_values("player_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keep], b[keep], check_exact=True)
    # `rank` and `edge` are the only two columns that CAN move, because both
    # are positions in a single cross-position ordering: a defense that
    # re-sorts past a skill player pushes his rank number down by one. The
    # order among skill players themselves is untouched.
    order_before = before[before["position"].isin(skill)].sort_values("rank")["player_id"].tolist()
    order_after = after[after["position"].isin(skill)].sort_values("rank")["player_id"].tolist()
    assert order_before == order_after
