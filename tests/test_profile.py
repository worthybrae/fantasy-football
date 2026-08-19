import dataclasses

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
    # ...including the kicking keys, which is the point of them being on
    # every row rather than only a kicker's: the client indexes a uniform
    # shape. A receiver's game carries them at zero, exactly as he has
    # always carried `pass_yards` at zero.
    assert s["fg_made"] == 0 and s["fg_att"] == 0 and s["fg_long"] == 0
    assert s["pat_made"] == 0 and s["pat_att"] == 0
    # The original twelve keys are still exactly the original twelve; the
    # kicking keys are additive. Asserted as two sets rather than one so a
    # future change that DROPS one of the twelve cannot be hidden by a
    # change that adds another.
    assert {"completions", "attempts", "pass_yards", "pass_tds",
            "interceptions", "carries", "rush_yards", "rush_tds",
            "targets", "receptions", "rec_yards", "rec_tds"}.issubset(s)
    assert set(s) == {"completions", "attempts", "pass_yards", "pass_tds",
                      "interceptions", "carries", "rush_yards", "rush_tds",
                      "targets", "receptions", "rec_yards", "rec_tds",
                      "fg_made", "fg_att", "fg_long", "pat_made", "pat_att"}

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


# -- profile frame cache + filtered reads (scoring/profile_cache.py) -------
#
# With the board cached, what was left of a profile click was build_profile's
# own reading: `weekly` (174,373 rows), `snap_counts` (253,106) and
# `depth_charts` (416,885) pulled whole on every click and then filtered down
# to one player in pandas, plus `player_season_features` recomputed twice and
# a 253k-row `_norm_name` map recomputed once, per click. That work now comes
# from cached_profile_frames, and the two genuinely per-player reads push
# their filter into SQL. Timing the second call would prove the cache exists,
# not that it's right; these pin the four ways it could be wrong instead --
# a cached frame annotated by whoever used it first, a SQL filter that
# doesn't select what the pandas filter did, a schema the filter can't
# express, and the one aggregate that is NOT per-player quietly becoming so.

def _seed_with_real_depth_schema(tmp_path):
    """Like _seed_two_players but with a depth_charts in the live nflverse
    shape (dt/team/pos_abb/pos_rank/gsis_id), which is the one _depth_slice
    can actually push a WHERE clause at, and a snap_counts with rows in it."""
    conn = get_conn(str(tmp_path / "depth.duckdb"))
    write_table(conn, "weekly", _two_player_weekly_rows())
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "A Star", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "B Steady", "position": "WR", "team": "GB", "adp": 20.0}]))
    write_table(conn, "depth_charts", pd.DataFrame([
        # two snapshots, so the "latest dt within the team" narrowing is live
        {"dt": dt, "team": team, "player_name": name, "gsis_id": pid,
         "pos_abb": "WR", "pos_rank": rank}
        for dt in ("2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z")
        for team, name, pid, rank in (("DET", "A Star", "p1", 1),
                                      ("DET", "Someone Else", "p3", 2),
                                      ("GB", "B Steady", "p2", 1))]))
    write_table(conn, "snap_counts", pd.DataFrame([
        {"player": "A Star", "team": "DET", "season": s, "offense_pct": 0.8}
        for s in (2023, 2024, 2025)] + [
        {"player": "B Steady", "team": "GB", "season": s, "offense_pct": 0.5}
        for s in (2023, 2024, 2025)]))
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp", "espn_ppr_rank"]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))
    return conn


def test_profile_frame_cache_is_not_annotated_by_the_player_who_used_it_first(tmp_path):
    """The trap in handing the same frame to two requests: the first line of
    season_summaries used to be `feats["pos_finish"] = ...`, which would
    write a column straight into the cached aggregate and leave it there for
    every later click. Draft one player's profile, then another's, then the
    first again -- the first player's payload must be identical to itself,
    and the cached frame must come back out unmarked."""
    from scoring import profile_cache
    conn = _seed_with_real_depth_schema(tmp_path)
    profile_cache.clear()

    first = build_profile(conn, "p1")
    build_profile(conn, "p2")
    again = build_profile(conn, "p1")

    assert first == again
    assert len(profile_cache._cache) == 1
    frames = next(iter(profile_cache._cache.values()))
    assert "pos_finish" not in frames.season_features.columns
    assert "_norm_name" not in frames.prior_weekly.columns


def test_season_summaries_does_not_annotate_the_features_frame_it_is_given(tmp_path):
    """The other half of the same defence, one layer down. cached_profile_
    frames hands out a copy, so the cache survives even a mutating
    season_summaries -- which means the cache test above cannot see this
    regression at all. Assert it here instead: a features frame passed in
    must come back out with the columns it went in with, or the belt is
    doing all the work and the braces are decorative."""
    from scoring.similarity import player_season_features
    weekly = _two_player_weekly_rows()
    feats = player_season_features(weekly)
    before = list(feats.columns)

    season_summaries(weekly[weekly["player_id"] == "p1"], None, "p1",
                     season_features=feats, snap_share=None)

    assert list(feats.columns) == before


def test_filtered_reads_select_exactly_what_a_full_read_and_filter_did(tmp_path):
    """`_player_weekly` and `_depth_slice` replace a whole-table read plus a
    pandas mask. Both downstream consumers are order-sensitive --
    `groupby("season")["recent_team"].last()` and `wk["position"].iloc[-1]`
    in one, `sort_values("pos_rank").drop_duplicates(...)` in the other --
    so this compares values, ROW ORDER and dtypes, not just contents."""
    from pipeline.db import read_table
    from scoring.profile import _player_weekly, _depth_slice
    conn = _seed_with_real_depth_schema(tmp_path)
    weekly, depth = read_table(conn, "weekly"), read_table(conn, "depth_charts")

    for pid, team in (("p1", "DET"), ("p2", "GB"), ("nobody", "SEA")):
        pd.testing.assert_frame_equal(
            _player_weekly(conn, pid),
            weekly[weekly["player_id"] == pid].reset_index(drop=True))
        pd.testing.assert_frame_equal(
            _depth_slice(conn, team, pid),
            depth[(depth["team"] == team) | (depth["gsis_id"] == pid)]
            .reset_index(drop=True))


def test_depth_slice_falls_back_when_the_schema_has_no_team_column(tmp_path):
    """_seed writes depth_charts with gsis_id/depth_team and no `team` at
    all -- an older schema variant the WHERE clause cannot be written
    against. Falling back to the full read keeps `_outlook`'s depth-slot
    lookup working; raising, or returning nothing, would silently drop it."""
    from pipeline.db import read_table
    from scoring.profile import _depth_slice
    conn = _seed(tmp_path)
    pd.testing.assert_frame_equal(_depth_slice(conn, "DET", "p1"),
                                  read_table(conn, "depth_charts"))


def test_game_log_still_zero_fills_weeks_the_player_missed(tmp_path):
    """build_profile now hands game_log only this player's weekly rows, so
    the league-wide `season -> last week` map has to be passed in beside
    them. Derived from the player's own rows instead, a season he left early
    would simply end early and the games he missed would vanish from the log.
    p1 played 4 games in 2023; p2 played all 17, so 2023 ran 17 weeks and
    p1's log must show 13 of them as dnp."""
    conn = _seed_with_real_depth_schema(tmp_path)
    log = build_profile(conn, "p1")["game_log"]
    y2023 = [r for r in log if r["season"] == 2023]
    assert len(y2023) == 17
    assert sum(r["dnp"] for r in y2023) == 13
    assert [r["week"] for r in y2023] == list(range(17, 0, -1))


def test_season_summaries_honours_an_explicit_empty_snap_share(tmp_path):
    """`snap_share=None` means "there is no snap data" and `snap_share`
    unset means "work it out from `snaps`" -- two different answers, which
    is why the default is a sentinel and not None. Getting that backwards
    would either crash on an empty table or silently drop the snap column."""
    snaps = pd.DataFrame([{"player": "Amon-Ra St. Brown", "team": "DET",
                           "season": 2025, "offense_pct": 0.9}])
    from_snaps = season_summaries(_weekly_rows(), snaps, "p1")[0]
    explicit_none = season_summaries(_weekly_rows(), snaps, "p1", snap_share=None)[0]
    assert from_snaps["snap_share"] == 0.9
    assert explicit_none["snap_share"] is None


def test_profile_frame_cache_is_bounded(tmp_path):
    """~40 MB per entry (season_features 6.0, snap_share 3.2, prior_weekly
    25.7, players 5.2, schedules 0.33 on data/nfl.duckdb). One process
    serving a database per league -- or a test session making a fresh one
    per test -- must not accumulate them without limit."""
    from scoring.profile_cache import _MAX_ENTRIES, cached_profile_frames, _cache, clear
    clear()
    for i in range(_MAX_ENTRIES + 5):
        conn = _seed(tmp_path / f"db{i}")
        cached_profile_frames(conn)
    assert len(_cache) <= _MAX_ENTRIES


# -- scoring-format awareness of the whole profile ---------------------------
#
# `build_profile` never received the league's settings at all before this, so
# every number it derived was priced in full PPR whatever the league scored:
# ppg, the week-to-week volatility beside it, the game log, the stat twins AND
# the `next_ppg` the owner reads as a forecast, the points-allowed schedule,
# and `proj_ppg`/`proj_delta` (ESPN's PPR season number differenced against a
# PPR career average). All of it sat next to a header row -- the board row --
# that had already been priced under the league's real rules.

def _fmt_settings(reception_points):
    """A COMPLETE rule set with one value moved, so the only thing separating
    the three leagues under test is the reception value the format is named
    for. A two-key sketch would also change every number, but for the boring
    reason that it stopped scoring touchdowns."""
    from scoring import league
    from scoring.ppr import DEFAULT_RULES
    return league.LeagueSettings(
        season=2026, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=2, bench=5,
        scoring={**DEFAULT_RULES, "receptions": reception_points},
        draft_type="SNAKE")


def _seed_format_fixture(tmp_path):
    """One receiving-heavy WR with an uneven week-to-week line, plus a twin.

    p1 (2025, 10 games) alternates 4 and 8 catches on a flat 80 yards and one
    touchdown, so his per-game points are 18/22 in full PPR, 16/18 in half and
    a flat 14 in standard:
        ppg      20.0   17.0   14.0
        ppg_std   2.11   1.05    0.0
    The volatility collapsing to zero is the whole point: in a standard league
    the only thing that varied about his weeks was the catch count, and catches
    stopped being points.

    t1 is a stat twin with a 2023 season (the comp) and a 2024 season (the
    "what happened next"), so `next_ppg` exists and moves with the rules too:
        2024 ppg  12.0    9.0    6.0
    """
    conn = get_conn(str(tmp_path / "fmt.duckdb"))
    weekly = pd.DataFrame(
        [{"player_id": "p1", "player_display_name": "Amon-Ra St. Brown",
          "position": "WR", "recent_team": "DET", "opponent_team": "GB",
          "season": 2025, "week": w, "receptions": 4 if w % 2 else 8,
          "receiving_yards": 80, "receiving_tds": 1, "targets": 10, "carries": 0}
         for w in range(1, 11)]
        + [{"player_id": "t1", "player_display_name": "Twin Guy",
            "position": "WR", "recent_team": "GB", "opponent_team": "DET",
            "season": 2023, "week": w, "receptions": 3, "receiving_yards": 50,
            "receiving_tds": 0, "targets": 5, "carries": 0}
           for w in range(1, 11)]
        + [{"player_id": "t1", "player_display_name": "Twin Guy",
            "position": "WR", "recent_team": "GB", "opponent_team": "DET",
            "season": 2024, "week": w, "receptions": 6, "receiving_yards": 60,
            "receiving_tds": 0, "targets": 8, "carries": 0}
           for w in range(1, 11)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET",
         "adp": 5.1}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "snap_counts", pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))
    write_table(conn, "espn_adp", pd.DataFrame([
        {"espn_id": 501, "espn_name": "Amon-Ra St Brown", "position": "WR",
         "espn_adp": 1.0, "espn_ppr_rank": 2, "espn_proj": 340.0}]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))
    return conn


def _profiles_by_format(conn):
    from scoring import board_cache, profile_cache
    out = {}
    for tag, rec in (("ppr", 1.0), ("half", 0.5), ("std", 0.0)):
        board_cache.clear()
        profile_cache.clear()
        out[tag] = build_profile(conn, "p1", None, _fmt_settings(rec))
    return out


def test_profile_per_game_points_and_volatility_follow_the_leagues_scoring(tmp_path):
    p = _profiles_by_format(_seed_format_fixture(tmp_path))
    assert [p[f]["seasons"][0]["ppg"] for f in ("ppr", "half", "std")] == [20.0, 17.0, 14.0]
    # Sample std of five 18s and five 22s is sqrt(40/9) = 2.108; halving the
    # reception value halves the spread, and removing it entirely leaves ten
    # identical weeks.
    assert p["ppr"]["seasons"][0]["ppg_std"] == 2.11
    assert p["half"]["seasons"][0]["ppg_std"] == 1.05
    assert p["std"]["seasons"][0]["ppg_std"] == 0.0
    # The header (the board row) already agreed with the league; now the body
    # of the page does too.
    for f in ("ppr", "half", "std"):
        assert p[f]["header"]["stats"]["ppg"] == p[f]["seasons"][0]["ppg"]


def test_profile_game_log_points_follow_the_leagues_scoring(tmp_path):
    p = _profiles_by_format(_seed_format_fixture(tmp_path))
    # Week 10 is an 8-catch week: 8 + 8.0 + 6 = 22.0 PPR, 18.0 half, 14.0 std.
    for f, expected in (("ppr", 22.0), ("half", 18.0), ("std", 14.0)):
        top = p[f]["game_log"][0]
        assert top["week"] == 10
        assert top["ppr_points"] == expected
        # The stat line itself is counts, not points -- it must NOT move.
        assert top["stat_line"] == "10 tgt, 8 rec, 80 yds, 1 TD"


def test_stat_twins_are_matched_and_forecast_in_the_leagues_scoring(tmp_path):
    """The comp's `next_ppg` is the number read as "what happened to players
    like him", and it was full PPR in every league. Both the ppg shown for
    the comp season and the following season's ppg move here."""
    p = _profiles_by_format(_seed_format_fixture(tmp_path))
    for f, comp_ppg, next_ppg in (("ppr", 8.0, 12.0), ("half", 6.5, 9.0),
                                  ("std", 5.0, 6.0)):
        similar = p[f]["similar"]
        assert similar["mode"] == "stat_twins"
        twin = similar["players"][0]
        assert twin["name"] == "Twin Guy" and twin["season"] == 2023
        assert twin["ppg"] == comp_ppg
        assert twin["next_ppg"] == next_ppg


def test_profile_projection_is_converted_out_of_espns_ppr(tmp_path):
    """`proj_ppg` and `proj_delta` were the worst of the lot: ESPN's PPR-only
    season number divided by 17 and then differenced against a career average
    that this change makes league-scored. Subtracting one currency from
    another produced a "projected improvement" that was nothing but the
    missing reception points.

    p1's scale is his own latest season: 17.0/20.0 in half, 14.0/20.0 in
    standard, so ESPN's 340.0 becomes 289.0 and 238.0.
    """
    p = _profiles_by_format(_seed_format_fixture(tmp_path))
    assert p["ppr"]["summary"]["proj_ppg"] == round(340.0 / 17, 1)
    assert p["half"]["summary"]["proj_ppg"] == round(340.0 * (17.0 / 20.0) / 17, 1)
    assert p["std"]["summary"]["proj_ppg"] == round(340.0 * (14.0 / 20.0) / 17, 1)
    # w_ppg is the league's points too, so the delta stays a like-for-like
    # comparison in all three.
    for f in ("ppr", "half", "std"):
        s = p[f]["summary"]
        assert s["proj_delta"] == round(s["proj_ppg"] - s["w_ppg"], 1)


def test_profile_schedule_difficulty_follows_the_leagues_scoring(tmp_path):
    """Points ALLOWED is as format-dependent as points scored, and it sits on
    the same page as the board's `schedule` factor, which has followed the
    league's rules since the settings work."""
    p = _profiles_by_format(_seed_format_fixture(tmp_path))
    fpa = [r["fpa_pg"] for f in ("ppr", "half", "std")
           for r in p[f]["schedule"] if r["week"] == 1]
    assert fpa[0] > fpa[1] > fpa[2]
    assert p["ppr"]["outlook"]["sos_raw"] > p["half"]["outlook"]["sos_raw"]
    assert p["half"]["outlook"]["sos_raw"] > p["std"]["outlook"]["sos_raw"]


def test_profile_frame_cache_does_not_serve_a_ppr_frame_to_a_half_ppr_league(tmp_path):
    """THE CACHE TRAP, on the profile side.

    scoring/profile_cache.py caches `season_features` -- the frame the stat
    twins are matched on and `next_ppg` is read from -- keyed only on the
    database and the pipeline-refresh fingerprint. Now that the frame is
    priced under the league's rules, a key that does not carry them would
    hand the second league whichever format asked first, silently, with every
    other number on the page correctly its own. The order matters: PPR first
    then half must give half's answer, and half first then PPR must give
    PPR's.
    """
    from scoring import board_cache, profile_cache
    conn = _seed_format_fixture(tmp_path)

    profile_cache.clear(); board_cache.clear()
    first_ppr = build_profile(conn, "p1", None, _fmt_settings(1.0))
    board_cache.clear()
    then_half = build_profile(conn, "p1", None, _fmt_settings(0.5))
    assert first_ppr["seasons"][0]["ppg"] == 20.0
    assert then_half["seasons"][0]["ppg"] == 17.0
    assert then_half["similar"]["players"][0]["next_ppg"] == 9.0

    profile_cache.clear(); board_cache.clear()
    first_half = build_profile(conn, "p1", None, _fmt_settings(0.5))
    board_cache.clear()
    then_ppr = build_profile(conn, "p1", None, _fmt_settings(1.0))
    assert first_half["seasons"][0]["ppg"] == 17.0
    assert then_ppr["seasons"][0]["ppg"] == 20.0
    assert then_ppr["similar"]["players"][0]["next_ppg"] == 12.0
    # Two entries, not one: the rules are part of the key.
    assert len(profile_cache._cache) == 2


def test_profile_frame_cache_treats_reordered_ppr_rules_as_one_entry(tmp_path):
    """The other half of the key's contract. `league.from_espn` emits the same
    14 PPR rules in ESPN's order, not DEFAULT_RULES'; two dicts that price
    identically must share one 40 MB entry and, more importantly, produce the
    identical frame rather than one differing in the last bits of a float."""
    from scoring import board_cache, profile_cache
    from scoring.ppr import DEFAULT_RULES
    conn = _seed_format_fixture(tmp_path)
    reordered = {k: DEFAULT_RULES[k] for k in reversed(list(DEFAULT_RULES))}

    profile_cache.clear(); board_cache.clear()
    build_profile(conn, "p1", None, _fmt_settings(1.0))
    board_cache.clear()
    build_profile(conn, "p1", None,
                  dataclasses.replace(_fmt_settings(1.0), scoring=reordered))
    assert len(profile_cache._cache) == 1


def test_board_cache_does_not_serve_a_ppr_board_to_a_half_ppr_league(tmp_path):
    """THE CACHE TRAP, on the board side.

    scoring/board_cache.py keys on `league.to_json(settings)`, which carries
    the whole `scoring` dict -- so a format change already invalidates it.
    That was verified by reading the key, and this pins it: without a scoring
    component a half-PPR league would be served the PPR league's proj_points,
    proj_scale and rank, which is the one number the whole board sorts on.
    """
    from scoring import board_cache
    conn = _seed_format_fixture(tmp_path)
    board_cache.clear()
    ppr = board_cache.cached_build_board(conn, None, _fmt_settings(1.0))
    half = board_cache.cached_build_board(conn, None, _fmt_settings(0.5))
    again = board_cache.cached_build_board(conn, None, _fmt_settings(1.0))

    p_ppr = ppr.set_index("player_id").loc["p1"]
    p_half = half.set_index("player_id").loc["p1"]
    assert p_ppr["proj_scale"] == 1.0
    assert p_half["proj_scale"] == 17.0 / 20.0
    assert p_ppr["proj_points"] == 340.0
    assert p_half["proj_points"] == 340.0 * (17.0 / 20.0)
    # And the PPR board comes back unchanged after the half-PPR build, so the
    # second call did not overwrite the first league's entry.
    pd.testing.assert_series_equal(p_ppr, again.set_index("player_id").loc["p1"])
    assert len(board_cache._cache) == 2


# ---------------------------------------------------------------------------
# Kickers: the history that was deliberately blanked, and when it stays that way
# ---------------------------------------------------------------------------

def _kicker_rules():
    """The owner's real kicking values, through the real ESPN map."""
    from pipeline.espn_league import parse_settings
    from scoring import league
    return league.from_espn(parse_settings({"settings": {
        "size": 8,
        "rosterSettings": {"lineupSlotCounts": {
            "0": 1, "2": 2, "4": 2, "6": 1, "16": 1, "17": 1, "20": 5, "23": 2}},
        "scoringSettings": {"scoringItems": [
            {"statId": 3, "points": 0.04}, {"statId": 4, "points": 4.0},
            {"statId": 19, "points": 2.0}, {"statId": 20, "points": -2.0},
            {"statId": 24, "points": 0.1}, {"statId": 25, "points": 6.0},
            {"statId": 26, "points": 2.0}, {"statId": 42, "points": 0.1},
            {"statId": 43, "points": 6.0}, {"statId": 44, "points": 2.0},
            {"statId": 53, "points": 1.0}, {"statId": 72, "points": -2.0},
            {"statId": 77, "points": 4.0}, {"statId": 80, "points": 3.0},
            {"statId": 85, "points": -1.0}, {"statId": 86, "points": 1.0},
            {"statId": 198, "points": 5.0}, {"statId": 201, "points": 6.0}]},
        "draftSettings": {"type": "SNAKE", "pickOrder": [1, 2, 3, 4, 5, 6, 7, 8]},
    }}, 2025)).scoring


def _kicker_settings(scoring):
    from scoring import league
    return league.LeagueSettings(
        season=2025, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=2, bench=5, scoring=scoring, draft_type="SNAKE")


def _seed_kicker_fixture(tmp_path):
    """One kicker with ten identical real-shaped weeks, plus a WR to rank against.

    Each of k1's weeks is 2 field goals from 30-39 and 3 extra points, which
    under the owner's scoring is 2 x 3 + 3 x 1 = 9.0 points a game, 90.0 for
    the season -- and 0.0 under full PPR, which scores no kicking at all.
    """
    conn = get_conn(str(tmp_path / "kick.duckdb"))
    weekly = pd.DataFrame(
        [{"player_id": "k1", "player_display_name": "Kick Guy",
          "position": "K", "recent_team": "DAL", "opponent_team": "PHI",
          "season": 2025, "week": w, "fg_made": 2, "fg_att": 2, "fg_long": 38,
          "fg_made_30_39": 2, "fg_missed": 0, "fg_blocked": 0,
          "pat_made": 3, "pat_att": 3, "pat_missed": 0, "pat_blocked": 0,
          "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
          "targets": 0, "carries": 0}
         for w in range(1, 11)]
        + [{"player_id": "p1", "player_display_name": "Amon-Ra St. Brown",
            "position": "WR", "recent_team": "DET", "opponent_team": "GB",
            "season": 2025, "week": w, "receptions": 6, "receiving_yards": 80,
            "receiving_tds": 1, "targets": 9, "carries": 0, "fg_made": 0,
            "fg_att": 0, "fg_long": 0, "pat_made": 0, "pat_att": 0,
            "fg_missed": 0, "fg_blocked": 0, "fg_made_30_39": 0,
            "pat_missed": 0, "pat_blocked": 0}
           for w in range(1, 11)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DAL", "away_team": "PHI", "week": 1,
         "total_line": 47.0, "spread_line": 2.0},
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Kick Guy", "position": "PK", "team": "DAL", "adp": 140.0},
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET",
         "adp": 5.1}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "snap_counts", pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp",
                 "espn_ppr_rank", "espn_proj"]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave",
                 "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))
    return conn


def _kicker_profile(conn, scoring):
    from scoring import board_cache, profile_cache
    board_cache.clear()
    profile_cache.clear()
    return build_profile(conn, "k1", None, _kicker_settings(scoring))


def test_a_kickers_history_is_still_blanked_when_the_league_prices_no_kicking(tmp_path):
    """The original reasoning, preserved rather than deleted.

    Under full PPR every one of this kicker's weeks is worth 0.0, so a
    history would be ten rows of zero -- misleading, not informative, which
    is exactly why scoring/profile.py collapsed it in the first place. That
    is still the right answer, and it is now reached by asking the RULES
    rather than by testing the position.
    """
    from scoring.ppr import DEFAULT_RULES
    p = _kicker_profile(_seed_kicker_fixture(tmp_path), dict(DEFAULT_RULES))
    assert p["header"]["position"] == "K"
    assert p["seasons"] == []
    assert p["game_log"] == []


def test_a_kickers_history_is_real_once_the_league_prices_kicking(tmp_path):
    """Ten weeks of 2 FG from 30-39 (3 points each) and 3 PATs (1 each).

    9.0 a game, 90.0 for the season, and every week identical so the
    volatility is exactly 0.0.
    """
    p = _kicker_profile(_seed_kicker_fixture(tmp_path), _kicker_rules())
    assert len(p["seasons"]) == 1
    season = p["seasons"][0]
    assert season["season"] == 2025 and season["games"] == 10
    assert season["ppg"] == 9.0
    assert season["ppg_std"] == 0.0
    # He is the only kicker in the fixture, so he finished K1.
    assert season["pos_finish"] == 1
    # The kicking counts behind those points ride on the season row.
    assert season["fg_made"] == 20 and season["fg_att"] == 20
    assert season["pat_made"] == 30 and season["pat_att"] == 30
    assert season["fg_long"] == 38          # a season maximum, not a sum

    assert len(p["game_log"]) == 10
    top = p["game_log"][0]
    assert top["ppr_points"] == 9.0
    # The line that used to read "0 tgt, 0 rec, 0 yds, 0 TD" for every kicker.
    assert top["stat_line"] == "2/2 FG, long 38 · 3/3 XP"
    assert top["stats"]["fg_made"] == 2 and top["stats"]["pat_made"] == 3
    # And the recency-weighted career summary now has something to average.
    assert p["summary"]["w_ppg"] == 9.0


def test_a_kickers_stat_line_survives_a_week_with_no_field_goal(tmp_path):
    """`fg_long` is a maximum, so a kick-less week has none -- the line must
    not claim "long 0"."""
    row = {"fg_made": 0, "fg_att": 0, "fg_long": 0, "pat_made": 4, "pat_att": 4}
    assert _stat_line(row, "K") == "0/0 FG · 4/4 XP"


def test_a_defenses_profile_is_unchanged_because_nothing_can_score_one(tmp_path):
    """DST has no weekly rows at all, so there is nothing to price however
    the league scores. This pins that the kicking work did not quietly
    invent a defensive history."""
    conn = _seed_kicker_fixture(tmp_path)
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Kick Guy", "position": "PK", "team": "DAL", "adp": 140.0},
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET",
         "adp": 5.1},
        {"adp_name": "Cowboys", "position": "DST", "team": "DAL", "adp": 150.0}]))
    from scoring import board_cache, profile_cache
    from scoring.board import build_board
    board_cache.clear()
    profile_cache.clear()
    settings = _kicker_settings(_kicker_rules())
    board = build_board(conn, settings=settings)
    dst = board[board["position"] == "DST"].iloc[0]
    profile = build_profile(conn, dst["player_id"], None, settings)
    assert profile["seasons"] == [] and profile["game_log"] == []
    assert profile["similar"]["mode"] == "value_neighbors"
    # Every factor but environment stays neutral for a defense, whatever the
    # league scores.
    for factor in ("production", "durability", "role", "schedule"):
        assert profile["factors"][factor] == 50.0
