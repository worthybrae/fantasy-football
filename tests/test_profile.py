import pandas as pd
from pipeline.db import get_conn, record_freshness, write_table
from scoring.profile import season_summaries, game_log, build_profile, _stat_line

def _weekly_rows():
    return pd.DataFrame(
        [{"player_id": "p1", "player_display_name": "Amon-Ra St. Brown",
          "position": "WR", "recent_team": "DET", "opponent_team": "GB",
          "season": 2025, "week": w, "receptions": 6, "receiving_yards": 80,
          "receiving_tds": 1, "targets": 9, "carries": 0}
         for w in range(1, 11)])

def _qb_weekly_rows():
    return pd.DataFrame(
        [{"player_id": "q1", "player_display_name": "QB Guy",
          "position": "QB", "recent_team": "DET", "opponent_team": "GB",
          "season": 2025, "week": w, "completions": 20, "attempts": 30,
          "passing_yards": 250, "passing_tds": 2, "passing_interceptions": 1,
          "carries": 4, "rushing_yards": 20, "rushing_tds": 0,
          "targets": 0, "receptions": 0, "receiving_yards": 0,
          "receiving_tds": 0}
         for w in range(1, 6)])

def test_season_summaries_math_and_snap_join():
    snaps = pd.DataFrame([{"player": "Amon-Ra St Brown", "team": "DET",
                           "season": 2025, "offense_pct": 0.9}])
    s = season_summaries(_weekly_rows(), snaps, "p1")[0]
    assert s["season"] == 2025 and s["games"] == 10
    assert s["ppg"] == 20.0            # 6 + 8 + 6 = 20 per game
    assert s["snap_share"] == 0.9      # matched despite punctuation
    assert s["target_share"] == 1.0

def test_season_summaries_passing_aggregates():
    s = season_summaries(_qb_weekly_rows(), pd.DataFrame(), "q1")[0]
    assert s["completions"] == 100 and s["attempts"] == 150
    assert s["pass_yards"] == 1250 and s["pass_tds"] == 10
    assert s["interceptions"] == 5

def test_season_summaries_passing_zero_filled_without_columns():
    # WR fixture has no passing columns at all -- fields must still exist
    s = season_summaries(_weekly_rows(), pd.DataFrame(), "p1")[0]
    assert s["pass_yards"] == 0 and s["attempts"] == 0

def test_game_log_line_and_order():
    rows = game_log(_weekly_rows(), "p1")
    assert rows[0]["week"] == 10       # newest first
    assert rows[0]["stat_line"] == "9 tgt, 6 rec, 80 yds, 1 TD"
    assert rows[0]["ppr_points"] == 20.0

def test_game_log_includes_structured_stats():
    rows = game_log(_weekly_rows(), "p1")
    s = rows[0]["stats"]
    assert s["targets"] == 9 and s["receptions"] == 6
    assert s["rec_yards"] == 80 and s["rec_tds"] == 1
    # columns absent from the weekly frame zero-fill
    assert s["pass_yards"] == 0 and s["completions"] == 0 and s["carries"] == 0
    assert set(s) == {"completions", "attempts", "pass_yards", "pass_tds",
                      "interceptions", "carries", "rush_yards", "rush_tds",
                      "targets", "receptions", "rec_yards", "rec_tds"}

def test_game_log_structured_passing_stats_nonzero():
    s = game_log(_qb_weekly_rows(), "q1")[0]["stats"]
    assert s["pass_yards"] == 250 and s["completions"] == 20
    assert s["attempts"] == 30 and s["pass_tds"] == 2
    assert s["interceptions"] == 1

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
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp", "espn_ppr_rank"]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))
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
    # is not a float NaN and slips past isinstance(v, (np.floating, float));
    # add_market then folds that into market_rank/market_sources -- with
    # every source table also empty, market_rank stays NaN and the nested
    # market_sources dict is all-None, both of which must scrub cleanly too.
    write_table(conn, "adp",
                pd.DataFrame(columns=["adp_name", "position", "team", "adp"]))
    write_table(conn, "depth_charts",
                pd.DataFrame(columns=["gsis_id", "depth_team", "formation",
                                      "week", "position"]))
    write_table(conn, "snap_counts",
                pd.DataFrame(columns=["player", "team", "season", "offense_pct"]))
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp", "espn_ppr_rank"]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))
    p = build_profile(conn, "p1")
    json.dumps(p, allow_nan=False)  # must not raise
    assert "adp" not in p["header"]
    assert p["header"]["market_rank"] is None
    assert p["header"]["market_sources"] == {"ffc": None, "espn": None, "fp": None,
                                             "mfl": None, "cbs": None, "fp_tier": None}

def _twin_weekly_rows():
    # p1: the profiled player, 2025 (latest season).
    p1 = [{"player_id": "p1", "player_display_name": "Star One", "position": "WR",
           "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
           "receptions": 6, "receiving_yards": 80, "receiving_tds": 1, "targets": 9,
           "carries": 0} for w in range(1, 11)]
    # p2: an on-board twin -- present in 2025 (the latest season), so it
    # enters the board universe and gets a real rank. Its *comp* season is
    # 2024, since comps must have a following season on file to survive the
    # next-season filter (2025 comps never can -- 2026 hasn't been played).
    p2 = [{"player_id": "p2", "player_display_name": "On Board Twin", "position": "WR",
           "recent_team": "GB", "opponent_team": "DET", "season": season, "week": w,
           "receptions": 6, "receiving_yards": 78, "receiving_tds": 1, "targets": 9,
           "carries": 0} for season in (2024, 2025) for w in range(1, 11)]
    # p3: an off-board twin -- present only in 2023/2024 (not the latest
    # season) and absent from ADP, so it never enters the board universe even
    # though its stat-similarity to p1 qualifies it as a twin candidate.
    p3 = [{"player_id": "p3", "player_display_name": "Off Board Twin", "position": "WR",
           "recent_team": "GB", "opponent_team": "DET", "season": season, "week": w,
           "receptions": 6, "receiving_yards": 79, "receiving_tds": 1, "targets": 9,
           "carries": 0} for season in (2023, 2024) for w in range(1, 11)]
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
    assert "market_rank" in twins["p2"]

    assert twins["p3"]["rank"] is None
    assert twins["p3"]["market_rank"] is None

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

def test_game_log_zero_fills_missed_weeks():
    # p1 misses week 2 while the league (p2) plays through week 4: the log
    # must still show week 2 as a zeroed dnp row, and cover weeks 3-4 too.
    rows = pd.DataFrame(
        [{"player_id": "p1", "player_display_name": "Fragile Guy",
          "position": "WR", "recent_team": "DET", "opponent_team": "GB",
          "season": 2025, "week": w, "receptions": 6, "receiving_yards": 80,
          "receiving_tds": 0, "targets": 9, "carries": 0}
         for w in (1, 3)]
        + [{"player_id": "p2", "player_display_name": "Iron Man",
            "position": "WR", "recent_team": "GB", "opponent_team": "DET",
            "season": 2025, "week": w, "receptions": 4, "receiving_yards": 40,
            "receiving_tds": 0, "targets": 6, "carries": 0}
           for w in range(1, 5)])
    logs = game_log(rows, "p1")
    assert [r["week"] for r in logs] == [4, 3, 2, 1]   # newest first, no gaps
    missed = next(r for r in logs if r["week"] == 2)
    assert missed["dnp"] is True
    assert missed["ppr_points"] == 0.0 and missed["opponent"] is None
    assert missed["stats"]["receptions"] == 0 and missed["stats"]["targets"] == 0
    played = next(r for r in logs if r["week"] == 3)
    assert played["dnp"] is False and played["ppr_points"] > 0

def test_season_summaries_ppg_std():
    # points per game: 8, 12, 16, 20 -> mean 14, sample var 80/3
    rows = pd.DataFrame(
        [{"player_id": "p1", "player_display_name": "Boom Bust", "position": "WR",
          "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
          "receptions": r, "receiving_yards": r * 10, "receiving_tds": 0,
          "targets": r + 2, "carries": 0}
         for w, r in [(1, 4), (2, 6), (3, 8), (4, 10)]])
    s = season_summaries(rows, pd.DataFrame(), "p1")[0]
    assert abs(s["ppg_std"] - round((80 / 3) ** 0.5, 2)) < 1e-9

def test_season_summaries_ppg_std_single_game_is_none():
    rows = pd.DataFrame(
        [{"player_id": "p1", "player_display_name": "One Game", "position": "WR",
          "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": 1,
          "receptions": 5, "receiving_yards": 50, "receiving_tds": 0,
          "targets": 7, "carries": 0}])
    s = season_summaries(rows, pd.DataFrame(), "p1")[0]
    assert s["ppg_std"] is None

def test_build_profile_passes_players_for_age_matching(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "weekly", _twin_weekly_rows())
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Star One", "position": "WR", "team": "DET", "adp": 5.1}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "snap_counts", pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))
    # p1 is 25 in 2025; p3 was 25 during its 2023 comp season; p2 was 30 in
    # 2024 -> only the same-age comp survives.
    write_table(conn, "players", pd.DataFrame([
        {"gsis_id": "p1", "display_name": "Star One", "birth_date": "2000-01-01", "rookie_season": 2022},
        {"gsis_id": "p2", "display_name": "On Board Twin", "birth_date": "1994-01-01", "rookie_season": 2016},
        {"gsis_id": "p3", "display_name": "Off Board Twin", "birth_date": "1998-01-01", "rookie_season": 2020},
    ]))
    prof = build_profile(conn, "p1")
    assert prof["similar"]["mode"] == "stat_twins"
    assert prof["similar"]["target_age"] == 25
    ids = {p["player_id"] for p in prof["similar"]["players"]}
    assert ids == {"p3"} and prof["similar"]["players"][0]["age"] == 25

def test_season_summaries_position_finish_rank():
    # p1 out-scores p2 in 2025 (both WR); the QB must not affect WR ranks.
    mk = lambda pid, name, pos, rec: [
        {"player_id": pid, "player_display_name": name, "position": pos,
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": rec, "receiving_yards": rec * 10, "receiving_tds": 0,
         "targets": rec + 2, "carries": 0}
        for w in range(1, 5)]
    rows = pd.DataFrame(mk("p1", "Better WR", "WR", 8)
                        + mk("p2", "Worse WR", "WR", 3)
                        + mk("q1", "Some QB", "QB", 9))
    assert season_summaries(rows, pd.DataFrame(), "p1")[0]["pos_finish"] == 1
    assert season_summaries(rows, pd.DataFrame(), "p2")[0]["pos_finish"] == 2
    assert season_summaries(rows, pd.DataFrame(), "q1")[0]["pos_finish"] == 1

def test_career_summary_weighted_math():
    from scoring.profile import career_summary
    # 2025 w=0.5, 2024 w=0.3 -> renormalized 0.625/0.375
    seasons = [
        {"season": 2025, "games": 10, "ppg": 20.0, "completions": 0, "attempts": 0,
         "pass_yards": 0, "pass_tds": 0, "interceptions": 0, "carries": 100,
         "rush_yards": 500, "targets": 30, "receptions": 20, "rec_yards": 150, "tds": 5},
        {"season": 2024, "games": 5, "ppg": 12.0, "completions": 0, "attempts": 0,
         "pass_yards": 0, "pass_tds": 0, "interceptions": 0, "carries": 25,
         "rush_yards": 100, "targets": 10, "receptions": 5, "rec_yards": 40, "tds": 1},
        {"season": 2016, "games": 17, "ppg": 99.0, "completions": 0, "attempts": 0,
         "pass_yards": 0, "pass_tds": 0, "interceptions": 0, "carries": 400,
         "rush_yards": 2000, "targets": 90, "receptions": 80, "rec_yards": 700, "tds": 20},
    ]
    s = career_summary(seasons)
    assert abs(s["w_ppg"] - 17.0) < 1e-9        # 20*.625 + 12*.375 = 17.0
    assert abs(s["w_stats"]["carries"] - 8.1) < 1e-9   # 10*.625 + 5*.375 = 8.125 -> 8.1
    # 2016 is outside the recency window and must not leak in

def test_career_summary_empty_window():
    from scoring.profile import career_summary
    s = career_summary([])
    assert s["w_ppg"] is None and s["w_stats"] == {}

def test_build_profile_summary_with_espn_projection(tmp_path):
    conn = _seed(tmp_path)
    conn.execute("DROP TABLE espn_adp")
    write_table(conn, "espn_adp", pd.DataFrame([
        {"espn_id": 99, "espn_name": "Amon-Ra St. Brown", "position": "WR",
         "espn_adp": 5.0, "espn_ppr_rank": 4, "espn_proj": 340.0}]))
    write_table(conn, "sleeper_ids", pd.DataFrame([
        {"gsis_id": "p1", "espn_id": 99, "sleeper_name": "Amon-Ra St. Brown",
         "position": "WR", "team": "DET"}]))
    p = build_profile(conn, "p1")
    s = p["summary"]
    assert s["proj_ppg"] == 20.0                # 340 / 17
    assert s["w_ppg"] == 20.0                   # single 2025 season at 20 ppg
    assert s["proj_delta"] == 0.0
    assert s["w_stats"]["receptions"] == 6.0

def test_build_profile_summary_degrades_without_proj_column(tmp_path):
    # Old espn_adp schema (pre-refresh) has no espn_proj column at all.
    p = build_profile(_seed(tmp_path), "p1")
    assert p["summary"]["proj_ppg"] is None and p["summary"]["proj_delta"] is None
    assert p["summary"]["w_ppg"] == 20.0

def _depth_fixture():
    rows = []
    for dt, rb1 in [("2026-08-01T09:00:00Z", "Old Starter"), ("2026-08-02T09:00:00Z", "New Starter")]:
        rows += [
            {"dt": dt, "team": "HOU", "gsis_id": "qb1", "player_name": "Some QB",
             "pos_abb": "QB", "pos_rank": 1},
            {"dt": dt, "team": "HOU", "gsis_id": "rb1", "player_name": rb1,
             "pos_abb": "RB", "pos_rank": 1},
            {"dt": dt, "team": "HOU", "gsis_id": "rb2", "player_name": "Me Backup",
             "pos_abb": "RB", "pos_rank": 2},
            {"dt": dt, "team": "DAL", "gsis_id": "x1", "player_name": "Other Team Guy",
             "pos_abb": "RB", "pos_rank": 1},
        ]
    return pd.DataFrame(rows)

def test_team_depth_chart_latest_snapshot_grouped_and_flagged():
    from scoring.profile import team_depth_chart
    out = team_depth_chart(_depth_fixture(), "HOU", "rb2")
    by_pos = {g["position"]: g["players"] for g in out}
    assert [p["name"] for p in by_pos["RB"]] == ["New Starter", "Me Backup"]  # latest dt wins
    assert by_pos["RB"][1]["is_me"] is True and by_pos["RB"][0]["is_me"] is False
    assert by_pos["QB"][0]["rank"] == 1
    assert all(g["position"] in ("QB", "RB", "WR", "TE") for g in out)

def test_team_depth_chart_old_schema_degrades_empty():
    from scoring.profile import team_depth_chart
    old = pd.DataFrame(columns=["gsis_id", "depth_team", "formation", "week", "position"])
    assert team_depth_chart(old, "HOU", "p1") == []

def test_weekly_difficulty_rows_and_percentiles():
    from scoring.profile import weekly_difficulty
    # Prior season: WRs score 30 pts/wk vs SOFT, 10 pts/wk vs TUFF (2 games each).
    prior = pd.DataFrame(
        [{"player_id": "w1", "player_display_name": "WR One", "position": "WR",
          "recent_team": "AAA", "opponent_team": opp, "season": 2025, "week": w,
          "receptions": rec, "receiving_yards": rec * 10, "targets": rec, "carries": 0}
         for opp, rec in [("SOFT", 15), ("TUFF", 5)] for w in (1, 2)])
    sched = pd.DataFrame([
        {"home_team": "HOU", "away_team": "SOFT", "week": 1, "total_line": 45.0, "spread_line": 3.0},
        {"home_team": "TUFF", "away_team": "HOU", "week": 2, "total_line": 45.0, "spread_line": 3.0},
        # no HOU game week 3 -> bye row
    ])
    out = weekly_difficulty(sched, prior, "HOU", "WR")
    assert len(out) == 18
    wk1, wk2, wk3 = out[0], out[1], out[2]
    assert wk1["opponent"] == "SOFT" and wk1["home"] is True
    assert wk2["opponent"] == "TUFF" and wk2["home"] is False
    assert wk3["opponent"] is None
    assert wk1["fpa_pg"] == 30.0 and wk2["fpa_pg"] == 10.0
    assert wk1["pct"] > wk2["pct"]          # high FPA = soft = high percentile
    assert weekly_difficulty(sched, prior, "HOU", "K") == []


# -- board cache invalidation (scoring/board_cache.py) --------------------
#
# build_profile used to call scoring.board.build_board directly, rebuilding
# the whole 249-row board (every factor, the composite, VOR, tiers, a
# five-source market consensus) on every single call just to read one row
# back out -- measured on data/nfl.duckdb at ~3.1s end to end. It now goes
# through cached_build_board (scoring/board_cache.py), which reuses a
# previously-built board when nothing it depends on has changed. A test that
# only timed the second call would prove the cache exists, not that it's
# correct -- these instead pin the two failure modes a wrong cache key would
# cause: serving one weight profile's board under another's, and serving a
# board built before a pick as though it were built after one.

def _two_player_weekly_rows():
    """p1: high per-game production, a short career (4/4/17 games across
    three seasons). p2: modest production, a full three-season workload.
    Production-only weights must favor p1; durability-only weights must
    favor p2 -- the same disagreement test_api.py's
    test_players_custom_weights_change_output uses to prove weights aren't
    silently ignored, reused here to prove a wrong weights key in the CACHE
    doesn't silently ignore them either."""
    def rows(pid, name, team, opp, weeks_by_season, receptions, yards, targets):
        return [
            {"player_id": pid, "player_display_name": name, "position": "WR",
             "recent_team": team, "opponent_team": opp, "season": season, "week": w,
             "receptions": receptions, "receiving_yards": yards, "targets": targets,
             "carries": 0}
            for season, n_weeks in weeks_by_season.items()
            for w in range(1, n_weeks + 1)
        ]
    return pd.DataFrame(
        rows("p1", "A Star", "DET", "GB", {2023: 4, 2024: 4, 2025: 17},
             receptions=8, yards=90, targets=10)
        + rows("p2", "B Steady", "GB", "DET", {2023: 17, 2024: 17, 2025: 17},
               receptions=2, yards=15, targets=4))


def _seed_two_players(tmp_path):
    conn = get_conn(str(tmp_path / "two.duckdb"))
    write_table(conn, "weekly", _two_player_weekly_rows())
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "A Star", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "B Steady", "position": "WR", "team": "GB", "adp": 20.0}]))
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
    return conn


def test_build_profile_cache_does_not_serve_one_weight_profile_under_another(tmp_path):
    conn = _seed_two_players(tmp_path)
    durability_only = {"production": 0.0, "role": 0.0, "environment": 0.0,
                       "schedule": 0.0, "durability": 1.0}

    default_first = build_profile(conn, "p1")                       # populates the cache
    custom = build_profile(conn, "p1", durability_only)             # a different key
    default_second = build_profile(conn, "p1")                      # must not read `custom`'s entry

    assert default_first["header"]["composite"] == default_second["header"]["composite"]
    assert default_first["header"]["composite"] != custom["header"]["composite"]


def test_build_profile_reflects_a_pick_made_after_first_call(tmp_path):
    """The scenario the task exists for: the owner opens a player's profile
    (the board gets built and cached), somebody -- the API's own
    POST /api/drafted, or (during a real live draft) the websocket handler
    in api/live.py, which writes the `drafted` table directly and outside
    this cache's control -- drafts a player, and the very next profile
    click must not still show that player as available."""
    conn = _seed(tmp_path)
    before = build_profile(conn, "p1")
    assert before["header"]["drafted"] is False

    # Mirrors the INSERT api/main.py's POST /api/drafted/{id} and api/live.py
    # both use -- neither goes through this cache, so the cache has to
    # notice on its own (see _drafted_key in scoring/board_cache.py).
    conn.execute("INSERT INTO drafted VALUES (?, ?)", ["p1", 1])

    after = build_profile(conn, "p1")
    assert after["header"]["drafted"] is True


def test_build_profile_reflects_a_pipeline_refresh(tmp_path):
    """`weekly` is a UNIVERSAL_TABLES source pipeline.refresh rewrites and
    stamps into `meta` via record_freshness -- not something a pick or a
    weight slider touches. Before any refresh, a player absent from `weekly`
    at seed time has no profile at all; after `weekly` is rewritten with
    that player added (a `write_table` + `record_freshness` pair, exactly
    what pipeline.refresh.main() does for every source it touches), the
    cache must rebuild rather than keep serving the pre-refresh universe."""
    conn = _seed(tmp_path)
    assert build_profile(conn, "p2") is None          # not in the seeded universe yet

    refreshed = pd.concat([_weekly_rows(), pd.DataFrame(
        [{"player_id": "p2", "player_display_name": "Fresh Import", "position": "WR",
          "recent_team": "GB", "opponent_team": "DET", "season": 2025, "week": w,
          "receptions": 5, "receiving_yards": 60, "receiving_tds": 0, "targets": 8,
          "carries": 0} for w in range(1, 11)])], ignore_index=True)
    write_table(conn, "weekly", refreshed)
    record_freshness(conn, "weekly", True, len(refreshed))

    after = build_profile(conn, "p2")
    assert after is not None
    assert after["header"]["name"] == "Fresh Import"


def test_cached_build_board_lru_is_bounded(tmp_path):
    """The owner drags five continuous slider values in the UI; a cache
    keyed partly on those floats grows one entry per distinct drag position
    if nothing bounds it -- a slow leak over a multi-hour draft-night
    session. Push far more distinct weight combinations through the cache
    than its stated bound (scoring.board_cache._MAX_ENTRIES) and confirm the
    LRU actually evicts rather than just documenting that it should."""
    from scoring.board_cache import _MAX_ENTRIES, cached_build_board, _cache
    conn = _seed(tmp_path)
    for i in range(_MAX_ENTRIES + 10):
        weights = {"production": 0.2 + i * 1e-4, "role": 0.2, "environment": 0.2,
                   "schedule": 0.2, "durability": 0.2 - i * 1e-4}
        cached_build_board(conn, weights)
    assert len(_cache) <= _MAX_ENTRIES
