import dataclasses

import pandas as pd
from pipeline.db import get_conn, record_freshness, write_table
from scoring.profile import (season_summaries, game_log, build_profile, _stat_line,
                            _proj_pos_finish)

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


# ===========================================================================
# The redesigned player card's seven additions
# ===========================================================================
#
# Everything below is ADDITIVE to the payload: no pre-existing key changes
# value or disappears. That was checked the only way it can be checked --
# against the real database, before and after, key by key, for a skill
# player, a kicker and a defense (see
# .superpowers/sdd/profile-data-report.md). What these tests pin instead is
# the part a payload diff cannot see: that each new number is the number it
# claims to be, and that the ones which rank a whole position-season are
# computed once per database rather than once per click.

def _card_weekly_rows():
    """Two receivers whose RAW volatility and SCALE-ADJUSTED volatility
    disagree, plus a third season nobody can rank.

    big   40.0 ppg, weeks alternating 32 and 48 -> sigma 8.55, cv 0.214
    small 10.0 ppg, weeks alternating  6 and 14 -> sigma 4.28, cv 0.428

    `big` swings twice as far in absolute points and half as far relative to
    what he scores. Rank the two on sigma and `big` is the more volatile of
    the two; rank them on the coefficient and he is the steadier -- which is
    the whole reason scoring/profile_cache.season_rank_frame divides. On
    data/nfl.duckdb the same disagreement puts Jahmyr Gibbs' 2025 at 97th of
    97 backs by sigma and 28th of 95 by coefficient.

    `short` plays 4 games -- under RANK_MIN_GAMES -- so his season is ranked
    nowhere and must come back null rather than 1st of a pool of one.

    Each carries a 2024 season as well, so 2025 has a preceding year that
    CAN enter a cohort (a comparable needs a following season on record).
    """
    def rows(pid, name, season, per_week):
        # `big` alternates his opponent as well as his catch count, so GB
        # concedes only his 48s and CHI only his 32s -- three teams end up in
        # the points-allowed table with three different figures, which is
        # what a schedule RANK needs to be a rank of anything.
        return [{"player_id": pid, "player_display_name": name, "position": "WR",
                 "recent_team": "DET" if pid == "big" else "GB",
                 "opponent_team": (("GB" if w % 2 else "CHI") if pid == "big"
                                   else "DET"),
                 "season": season, "week": w, "receptions": r,
                 "receiving_yards": 0, "receiving_tds": 0, "targets": r + 2,
                 "carries": 0}
                for w, r in enumerate(per_week, start=1)]
    alt = lambda a, b, n: [a if i % 2 else b for i in range(n)]
    return pd.DataFrame(
        rows("big", "Big Swing", 2025, alt(32, 48, 8))
        + rows("big", "Big Swing", 2024, [38] * 8)          # flat 38.0 ppg
        + rows("small", "Small Swing", 2025, alt(6, 14, 8))
        + rows("small", "Small Swing", 2024, [12] * 8)      # flat 12.0 ppg
        + rows("short", "Short Season", 2025, [30] * 4)     # under the qualifier
        + rows("short", "Short Season", 2024, [30] * 4))


def _seed_card_fixture(tmp_path, *, snaps=None, depth=None, players=None):
    from scoring import board_cache, profile_cache
    board_cache.clear()
    profile_cache.clear()
    conn = get_conn(str(tmp_path / "card.duckdb"))
    write_table(conn, "weekly", _card_weekly_rows())
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0},
        {"home_team": "CHI", "away_team": "DET", "week": 2,
         "total_line": 48.0, "spread_line": -1.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Big Swing", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "Small Swing", "position": "WR", "team": "GB", "adp": 20.0},
        {"adp_name": "Short Season", "position": "WR", "team": "GB", "adp": 30.0}]))
    write_table(conn, "depth_charts", depth if depth is not None else pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "snap_counts", snaps if snaps is not None else pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))
    write_table(conn, "players", players if players is not None else pd.DataFrame([
        {"gsis_id": "big", "display_name": "Big Swing",
         "birth_date": "2000-03-20", "rookie_season": 2023},
        {"gsis_id": "small", "display_name": "Small Swing",
         "birth_date": "1996-10-15", "rookie_season": 2019},
        {"gsis_id": "short", "display_name": "Short Season",
         "birth_date": "1999-01-01", "rookie_season": 2022}]))
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp", "espn_ppr_rank"]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave",
                 "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))
    return conn


def _season(profile, season):
    return next(s for s in profile["seasons"] if s["season"] == season)


# -- 2. positional rank by points per game ----------------------------------

def test_positional_rank_by_ppg_is_distinct_from_the_total_points_finish(tmp_path):
    """`pos_finish` ranks on SEASON TOTAL points over every player-season;
    this ranks on POINTS PER GAME over the qualified ones. They are different
    questions and the fixture separates them: `short` scores 30 a game, more
    than either qualifier, but only for four games -- so he finishes last on
    totals and would be first on per-game if he were ranked at all."""
    conn = _seed_card_fixture(tmp_path)
    big = _season(build_profile(conn, "big"), 2025)
    small = _season(build_profile(conn, "small"), 2025)

    assert big["ppg"] == 40.0 and small["ppg"] == 10.0
    assert (big["pos_rank_ppg"], big["pos_rank_ppg_n"]) == (1, 2)
    assert (small["pos_rank_ppg"], small["pos_rank_ppg_n"]) == (2, 2)
    # The denominator is the qualified pool, not the whole position: `short`
    # is a third WR in this season and is not in it.
    #
    # And the two rankings genuinely disagree, which is why both are served:
    # on SEASON TOTALS `short`'s four 30-point games (120) beat `small`'s
    # eight 10-point ones (80), so `small` finishes 3rd of the three -- while
    # per game he is 2nd of the two anyone can rank.
    short = _season(build_profile(conn, "short"), 2025)
    assert (big["pos_finish"], short["pos_finish"], small["pos_finish"]) == (1, 2, 3)


def test_a_season_under_the_games_qualifier_is_ranked_nowhere(tmp_path):
    """Four games is not a season you can rank a per-game average from, and
    the honest answer is no rank at all. Null, not 1st-of-1: a rank nobody
    computed must not arrive looking like a computed one."""
    conn = _seed_card_fixture(tmp_path)
    short = _season(build_profile(conn, "short"), 2025)
    assert short["ppg"] == 30.0        # the average is still shown
    for key in ("pos_rank_ppg", "pos_rank_ppg_n", "cv", "cv_rank",
                "cv_rank_n", "cv_pos_median"):
        assert short[key] is None, key
    # ...and he is absent from the denominator the ranked players are given.
    assert _season(build_profile(conn, "big"), 2025)["pos_rank_ppg_n"] == 2


# -- 3. volatility, scale-adjusted ------------------------------------------

def test_volatility_ranks_on_the_coefficient_not_the_raw_deviation(tmp_path):
    """THE REGRESSION THIS TEST EXISTS FOR.

    Ranking on raw sigma re-ranks by scoring: a bigger scorer swings in
    bigger absolute points because his good weeks are bigger, not because he
    is less reliable (measured across the 2023-2025 running back pools of
    data/nfl.duckdb, sigma correlates with points per game at r = 0.85 /
    0.81 / 0.87). On raw sigma Jahmyr Gibbs' 2025 comes out 97th of 97 --
    "the most volatile back in the league" -- and on the coefficient the same
    season is 28th of 95.

    Here `big` has DOUBLE `small`'s sigma and HALF his coefficient, so the
    two orderings are exact opposites and no amount of luck can pass this
    with a sigma rank.
    """
    conn = _seed_card_fixture(tmp_path)
    big = _season(build_profile(conn, "big"), 2025)
    small = _season(build_profile(conn, "small"), 2025)

    # Raw sigma -- still published, unchanged, and still says the opposite.
    assert big["ppg_std"] > small["ppg_std"]
    assert (big["ppg_std"], small["ppg_std"]) == (8.55, 4.28)

    # The coefficient, and the rank built on it.
    assert big["cv"] == 0.214 and small["cv"] == 0.428
    assert (big["cv_rank"], big["cv_rank_n"]) == (1, 2)
    assert (small["cv_rank"], small["cv_rank_n"]) == (2, 2)

    # The position median travels with the rank, because 0.214 is not a
    # number anyone has intuitions about and "0.214 against 0.321" is.
    assert big["cv_pos_median"] == small["cv_pos_median"] == 0.321


def test_a_season_with_no_positive_mean_gets_no_coefficient(tmp_path):
    """sigma over a mean of nothing is not a volatility, and a NEGATIVE
    coefficient would sort as the steadiest season in the league. Such a
    season is excluded from the volatility pool entirely -- which is why the
    two denominators can differ (on data/nfl.duckdb, 2025's running backs are
    97 by points per game and 95 by coefficient)."""
    from scoring.profile_cache import season_rank_frame
    from scoring.similarity import player_season_features
    weekly = pd.concat([_card_weekly_rows(), pd.DataFrame(
        [{"player_id": "sunk", "player_display_name": "Net Negative",
          "position": "WR", "recent_team": "GB", "opponent_team": "DET",
          "season": 2025, "week": w, "receptions": 0, "receiving_yards": 0,
          "receiving_tds": 0, "targets": 1, "carries": 0,
          "receiving_fumbles_lost": 1 if w % 2 else 0}
         for w in range(1, 9)])], ignore_index=True)
    ranks = season_rank_frame(weekly, player_season_features(weekly), None)
    sunk = ranks[(ranks["player_id"] == "sunk") & (ranks["season"] == 2025)].iloc[0]

    assert sunk["pos_rank_ppg"] == 3 and sunk["pos_rank_ppg_n"] == 3
    assert pd.isna(sunk["cv"]) and pd.isna(sunk["cv_rank"])
    assert sunk["cv_rank_n"] == 2          # counted out of the volatility pool


# -- 4. comparable cohort ---------------------------------------------------

def test_comparable_cohort_is_a_distribution_not_a_top_five(tmp_path):
    """`similar` answers "who does he resemble" with five names; this answers
    "what does a season like this usually do next", which needs every
    qualifying season and the shape they make.

    `big`'s target season is 2025 at 40.0 ppg in his 3rd year. `small`'s 2024
    (12.0 ppg, 6th year) is outside both bands. `big`'s own 2024 (38.0 ppg,
    2nd year, followed by 40.0) is inside both -- 2.0 ppg apart and one year
    apart -- so it is the cohort, and its change is +2.0.
    """
    conn = _seed_card_fixture(tmp_path)
    cohort = build_profile(conn, "big")["cohort"]

    assert cohort["season"] == 2025 and cohort["ppg"] == 40.0
    assert cohort["nfl_season"] == 3
    # The band is served, not just applied: a card that says "n comparable
    # seasons" without saying what made them comparable cannot be read.
    assert cohort["ppg_band"] == 3.0 and cohort["exp_band"] == 1
    assert cohort["min_games"] == 8

    assert cohort["n"] == 1
    assert cohort["declined"] == 0 and cohort["improved"] == 1
    assert cohort["median_change"] == 2.0
    comp = cohort["players"][0]
    assert comp["player_id"] == "big" and comp["season"] == 2024
    assert comp["ppg"] == 38.0 and comp["next_ppg"] == 40.0
    # The CHANGE is served per comparable, not just the raw next season --
    # the distribution the card draws is of movement, which is one column.
    assert comp["change"] == 2.0


def test_a_comparable_needs_a_following_season_on_record(tmp_path):
    """A comp with no next year carries no "and then what", so it is not a
    comp. Every 2025 season in this fixture is therefore ineligible (2026 has
    not been played) -- including `big`'s own, which is why he can never turn
    up in his own cohort here."""
    conn = _seed_card_fixture(tmp_path)
    cohort = build_profile(conn, "big")["cohort"]
    assert all(c["season"] < 2025 for c in cohort["players"])
    assert not any(c["player_id"] == "big" and c["season"] == 2025
                   for c in cohort["players"])


def test_the_ppg_band_actually_excludes_a_season_outside_it(tmp_path):
    """Without a band every season of the position would be a "comparable".
    `small`'s 2024 is 12.0 ppg against `big`'s 40.0 -- 28 points outside a
    3-point band -- and must not appear whatever else is true of it."""
    conn = _seed_card_fixture(tmp_path)
    ids = {c["player_id"] for c in build_profile(conn, "big")["cohort"]["players"]}
    assert "small" not in ids
    # ...and read from the other side, `small`'s own cohort excludes `big`.
    ids = {c["player_id"] for c in build_profile(conn, "small")["cohort"]["players"]}
    assert "big" not in ids


def test_a_player_with_no_rookie_season_keeps_the_ppg_band_and_says_so(tmp_path):
    """Missing biography drops the experience band rather than matching one.
    A wider cohort is fine; a wider cohort presented as a tight one is not,
    so `exp_band` comes back null to say what happened."""
    conn = _seed_card_fixture(tmp_path, players=pd.DataFrame([
        {"gsis_id": "big", "display_name": "Big Swing",
         "birth_date": None, "rookie_season": None}]))
    cohort = build_profile(conn, "big")["cohort"]
    assert cohort["nfl_season"] is None and cohort["exp_band"] is None
    assert cohort["ppg_band"] == 3.0


# -- 5. schedule as a rank --------------------------------------------------

def test_schedule_serves_a_league_rank_and_rank_one_is_the_softest(tmp_path):
    """The owner asked for a rank in place of a percentile, in those words:
    "41st percentile" is a number you have to convert before you can use it.

    Note the DIRECTION, which is the opposite of a difficulty rank -- a high
    `fpa_pg` is a GOOD matchup, so rank 1 goes to the defence that gave up
    the most. Detroit's receivers hung 48.0 a game on GB and 32.0 on CHI, so
    from a Detroit receiver's chair GB is the softer of the two.
    """
    conn = _seed_card_fixture(tmp_path)
    sched = build_profile(conn, "big")["schedule"]
    wk1 = next(r for r in sched if r["week"] == 1)
    wk2 = next(r for r in sched if r["week"] == 2)

    assert (wk1["opponent"], wk1["fpa_pg"], wk1["rank"]) == ("GB", 48.0, 1)
    assert (wk2["opponent"], wk2["fpa_pg"], wk2["rank"]) == ("CHI", 32.0, 2)
    # Three defences in the table, so the denominator is 3 -- and DET, which
    # this receiver never faces, is the 3rd of them.
    assert wk1["rank_n"] == 3 and wk2["rank_n"] == 3
    # `pct` is untouched -- it is what the existing strip renders.
    assert wk1["pct"] > wk2["pct"]
    # A bye keeps its nulls on the new keys as well as the old ones.
    bye = next(r for r in sched if r["opponent"] is None)
    assert bye["rank"] is None and bye["rank_n"] is None


# -- 6. per-game snap share -------------------------------------------------

def _card_snap_counts():
    """Real-schema snap_counts: pfr ids, per-game rows, playoffs included.

    `big` climbs from 50% to 90% across 2025 -- the story a season mean of
    70% cannot tell. Week 4 is deliberately missing (he did not play) and
    there is a week-19 playoff row that must NOT reach the log, whose rows
    come from the REG-only `weekly` table.
    """
    rows = [{"season": 2025, "week": w, "game_type": "REG", "player": "Big Swing",
             "pfr_player_id": "BigSw00", "position": "WR", "team": "DET",
             "opponent": "GB", "offense_snaps": 40, "offense_pct": pct}
            for w, pct in ((1, 0.50), (2, 0.60), (3, 0.70), (5, 0.80),
                           (6, 0.90), (7, 0.90), (8, 0.90))]
    rows.append({"season": 2025, "week": 19, "game_type": "DIV",
                 "player": "Big Swing", "pfr_player_id": "BigSw00",
                 "position": "WR", "team": "DET", "opponent": "GB",
                 "offense_snaps": 40, "offense_pct": 0.99})
    # An offensive line, so line_quality has something to rate.
    for i, (pfr, name) in enumerate(
            [("LineA00", "Line A"), ("LineB00", "Line B"), ("LineC00", "Line C"),
             ("LineD00", "Line D"), ("LineE00", "Line E")]):
        for season in (2024, 2025):
            for w in range(1, 9):
                rows.append({"season": season, "week": w, "game_type": "REG",
                             "player": name, "pfr_player_id": pfr, "position": "T",
                             "team": "DET", "opponent": "GB",
                             "offense_snaps": 60 - i, "offense_pct": 1.0})
    return pd.DataFrame(rows)


def _card_players_with_line():
    return pd.DataFrame(
        [{"gsis_id": "big", "display_name": "Big Swing",
          "birth_date": "2000-03-20", "rookie_season": 2023},
         {"gsis_id": "small", "display_name": "Small Swing",
          "birth_date": "1996-10-15", "rookie_season": 2019},
         {"gsis_id": "short", "display_name": "Short Season",
          "birth_date": "1999-01-01", "rookie_season": 2022}]
        + [{"gsis_id": f"ol{i}", "display_name": n,
            "birth_date": "1995-01-01", "rookie_season": 2018}
           for i, n in enumerate(["Line A", "Line B", "Line C", "Line D", "Line E"])])


def _card_depth_charts():
    return pd.DataFrame([
        {"dt": "2026-08-09T07:49:05Z", "team": "DET", "gsis_id": f"ol{i}",
         "player_name": n, "pos_name": slot, "pos_abb": "T", "pos_rank": 1}
        for i, (n, slot) in enumerate(zip(
            ["Line A", "Line B", "Line C", "Line D", "Line E"],
            ["Left Tackle", "Left Guard", "Center", "Right Guard", "Right Tackle"]))])


def test_per_game_snap_share_rides_the_log_and_is_regular_season_only(tmp_path):
    """A season mean of 70% and a role that climbed 50 -> 90 are the same
    number and different facts; the card wants the second. Jahmyr Gibbs' 2025
    on data/nfl.duckdb reads 66, 56, 69, 62, 52, 69, 56, 66, 50, 73, 74, 70,
    69, 81, 86, 69, 71 -- verified against `snap_counts` game by game.

    REG only: these hang off `game_log`, whose rows come from the REG-only
    `weekly` table, so a week-19 playoff row would either land on a week the
    log does not have or collide with an 18-week era's week 19.
    """
    conn = _seed_card_fixture(tmp_path, snaps=_card_snap_counts(),
                              depth=_card_depth_charts(),
                              players=_card_players_with_line())
    log = build_profile(conn, "big")["game_log"]
    by_week = {r["week"]: r for r in log if r["season"] == 2025}

    assert [by_week[w]["snap_pct"] for w in (1, 2, 3)] == [0.5, 0.6, 0.7]
    assert [by_week[w]["snap_pct"] for w in (6, 7, 8)] == [0.9, 0.9, 0.9]
    # Week 4 he played and `snap_counts` has no row for him -- the snap table
    # is a separate scrape from the stat table and does miss games. Null, not
    # zero: "we don't know" is not "he was never on the field".
    assert by_week[4]["dnp"] is False and by_week[4]["snap_pct"] is None
    assert max(by_week) == 8               # no week 19 leaked in from the playoffs

    # A week he did not play at all keeps its null too -- `short` played four
    # of the season's eight weeks.
    short = {r["week"]: r for r in build_profile(conn, "short")["game_log"]}
    assert short[8]["dnp"] is True and short[8]["snap_pct"] is None


def test_snap_share_is_keyed_through_the_crosswalk_so_an_unmatched_id_is_null(tmp_path):
    """The season aggregate can afford a name+team+season join because it is
    averaging; hanging seventeen individual games off a name join cannot.
    `oline.reconcile_pfr_to_gsis` is reused for the id resolution (with its
    position filter lifted -- restricted to the five line spots it resolves
    849 of data/nfl.duckdb's 5,890 pfr ids and no skill players at all), and
    what it cannot resolve gets a null rather than somebody else's snaps."""
    snaps = _card_snap_counts()
    snaps.loc[snaps["pfr_player_id"] == "BigSw00", "player"] = "Someone Else"
    conn = _seed_card_fixture(tmp_path, snaps=snaps, depth=_card_depth_charts(),
                              players=_card_players_with_line())
    log = build_profile(conn, "big")["game_log"]
    assert all(r["snap_pct"] is None for r in log)


# -- 7. team offensive line -------------------------------------------------

def test_team_line_quality_serves_the_rank_and_its_four_components(tmp_path):
    """scoring/oline.py had been built, tested and imported by nothing but
    its own test file since it was written. The composite alone is an opaque
    0-100, so the four parts ride with it: on data/nfl.duckdb for 2026,
    Detroit is 21st of 32 at 44.4 -- not because the line is falling apart
    (continuity 0.815) but because its starters have missed a quarter of
    their careers between them (availability 0.748)."""
    conn = _seed_card_fixture(tmp_path, snaps=_card_snap_counts(),
                              depth=_card_depth_charts(),
                              players=_card_players_with_line())
    oline = build_profile(conn, "big")["oline"]

    from scoring.config import CURRENT_SEASON
    assert oline["season"] == CURRENT_SEASON
    assert oline["team"] == "DET"
    assert oline["rank"] == 1 and oline["teams"] == 1   # the only rated line here
    assert 0.0 <= oline["line_quality"] <= 100.0
    # Five linemen took every snap in 2025 and all five are still penciled in.
    assert oline["continuity"] == 1.0
    assert oline["returning"] == 1.0
    assert oline["availability"] == 1.0
    assert oline["experience"] == 8.0                   # 2026 - 2018


def test_a_defense_gets_no_offensive_line_because_it_would_mean_nothing(tmp_path):
    """A defense's own team's offensive line is a fact about the eleven
    players who leave the field when it comes on. A plausible-looking number
    that means nothing is worse than a blank."""
    from scoring.profile import team_line_quality
    from scoring.oline import line_quality
    conn = _seed_card_fixture(tmp_path, snaps=_card_snap_counts(),
                              depth=_card_depth_charts(),
                              players=_card_players_with_line())
    lq = line_quality(conn, 2026)
    assert team_line_quality(lq, 2026, "DET", "DST") is None
    assert team_line_quality(lq, 2026, "DET", "K") is not None
    # An unrated team, and a player with no team at all, degrade the same way.
    assert team_line_quality(lq, 2026, "SEA", "WR") is None
    assert team_line_quality(lq, 2026, None, "WR") is None


# -- 1. age and NFL season --------------------------------------------------

def test_age_is_as_of_opening_week_and_the_nfl_season_is_one_based(tmp_path):
    """AGE IS AS OF SEPTEMBER 1 of the season year -- opening week -- which is
    not a convention invented for this card: it is
    `similarity._age_in_season`, which the stat-twin block has always matched
    comparables on, imported rather than re-implemented so the two halves of
    the page cannot disagree.

    `big` is born 2000-03-20 (before September 1, so he turns over inside the
    off-season) and `small` 1996-10-15 (after it, so he is still the younger
    number all season). Rookie seasons 2023 and 2019, and `nfl_season` counts
    a rookie year as the 1st, not the 0th.
    """
    from scoring.config import CURRENT_SEASON
    conn = _seed_card_fixture(tmp_path)
    big = build_profile(conn, "big")
    small = build_profile(conn, "small")

    assert big["bio"]["season"] == CURRENT_SEASON       # the season being drafted
    assert big["bio"]["birth_date"] == "2000-03-20"
    assert big["bio"]["rookie_season"] == 2023
    assert big["bio"]["age"] == 26 and big["bio"]["nfl_season"] == 4
    # Born after September 1: 2026 - 1996 - 1.
    assert small["bio"]["age"] == 29 and small["bio"]["nfl_season"] == 8

    # And every season row carries the same two facts as of ITS year, which
    # is what makes a history table readable -- 16 ppg at 21 and 16 ppg at 30
    # are not the same season.
    assert _season(big, 2025)["age"] == 25 and _season(big, 2025)["nfl_season"] == 3
    assert _season(big, 2024)["age"] == 24 and _season(big, 2024)["nfl_season"] == 2


def test_a_player_the_players_table_has_never_heard_of_gets_nulls_not_zeros(tmp_path):
    """Every defense, and any player nflverse has no biography for. "Age 0"
    is a claim; "age unknown" is the truth."""
    conn = _seed_card_fixture(tmp_path, players=pd.DataFrame(
        columns=["gsis_id", "display_name", "birth_date", "rookie_season"]))
    bio = build_profile(conn, "big")["bio"]
    assert bio["age"] is None and bio["nfl_season"] is None
    assert bio["birth_date"] is None and bio["rookie_season"] is None
    assert _season(build_profile(conn, "big"), 2025)["age"] is None


# -- K and DST degrade honestly ---------------------------------------------

def test_a_defense_reaches_none_of_the_new_code_and_raises_in_none_of_it(tmp_path):
    """A defense has no weekly rows at all, so most of the card is genuinely
    unavailable for one. Nulls and empty lists rather than zeros, and nothing
    that raises on the way there."""
    import json
    conn = _seed_kicker_fixture(tmp_path)
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Kick Guy", "position": "PK", "team": "DAL", "adp": 140.0},
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "Cowboys", "position": "DST", "team": "DAL", "adp": 150.0}]))
    from scoring import board_cache, profile_cache
    from scoring.board import build_board
    board_cache.clear(); profile_cache.clear()
    settings = _kicker_settings(_kicker_rules())
    dst_id = build_board(conn, settings=settings).pipe(
        lambda b: b[b["position"] == "DST"].iloc[0]["player_id"])

    p = build_profile(conn, dst_id, None, settings)
    json.dumps(p, allow_nan=False)          # nothing unserialisable slipped in
    assert p["cohort"] is None and p["oline"] is None
    assert p["bio"] == {"season": 2026, "birth_date": None, "rookie_season": None,
                        "age": None, "nfl_season": None}
    assert p["seasons"] == [] and p["game_log"] == [] and p["schedule"] == []


def test_a_kicker_gets_every_new_number_a_skill_player_gets(tmp_path):
    """Kickers score for real now, so their card is a real card: every rank,
    the cohort and the coefficient, all priced on the league's kicking
    points. (On data/nfl.duckdb, with the kicking rules restored, Brandon
    Aubrey's 2025 comes back K4 of 33 by points per game, 13th of 33 on the
    coefficient, and a cohort of 41 seasons with a median change of -0.6.)

    A second kicker with two seasons is added here rather than to the shared
    fixture, because a comparable needs a FOLLOWING season on record and the
    shared fixture has one season in total -- so without him there is no pool
    for anybody, and `cohort` is hidden the same way it is for a rookie.
    """
    conn = _seed_kicker_fixture(tmp_path)
    existing = conn.execute("SELECT * FROM weekly").df()
    k2 = pd.DataFrame(
        [{**dict.fromkeys(existing.columns, 0), "player_id": "k2",
          "player_display_name": "Other Boot", "position": "K",
          "recent_team": "PHI", "opponent_team": "DAL", "season": season,
          "week": w, "fg_made": fg, "fg_att": fg, "fg_long": 38,
          "fg_made_30_39": fg, "pat_made": 2, "pat_att": 2}
         for season, fg in ((2024, 2), (2025, 1)) for w in range(1, 11)])
    write_table(conn, "weekly", pd.concat([existing, k2], ignore_index=True))

    p = _kicker_profile(conn, _kicker_rules())
    season = p["seasons"][0]

    # k1 is 9.0 a game, k2 is 5.0 in 2025 -- so K1 of the two who qualify.
    assert season["pos_rank_ppg"] == 1 and season["pos_rank_ppg_n"] == 2
    # Ten identical weeks: sigma 0, so the coefficient is 0 and he is the
    # steadiest of them.
    assert season["cv"] == 0.0 and season["cv_rank"] == 1

    # k2's 2024 (8.0 a game, followed by 5.0) is inside the 3-point band
    # around k1's 9.0, so the cohort is real and its change is negative.
    cohort = p["cohort"]
    assert cohort["position"] == "K" and cohort["n"] == 1
    assert cohort["players"][0]["change"] == -3.0
    assert cohort["declined"] == 1

    assert p["bio"]["season"] == 2026
    # And a kicker's schedule is a rank now too -- two defences conceded
    # kicking points in this fixture, so the denominator is 2.
    assert p["schedule"] and p["schedule"][0]["rank_n"] == 2


def test_a_kicker_whose_league_prices_no_kicking_still_gets_no_ranks(tmp_path):
    """The blanking branch is on the RULES, not the position, and everything
    added here sits downstream of it: no history, so nothing to rank."""
    from scoring.ppr import DEFAULT_RULES
    p = _kicker_profile(_seed_kicker_fixture(tmp_path), dict(DEFAULT_RULES))
    assert p["seasons"] == [] and p["cohort"] is None
    assert p["bio"]["age"] is None          # no players table in that fixture


# -- caching: cross-player work must not be redone per click ----------------

def test_the_line_rating_is_built_once_per_database_not_once_per_click(tmp_path):
    """`oline.line_quality` costs 0.76s on data/nfl.duckdb -- it reads
    `snap_counts` whole and rebuilds a name crosswalk -- against a warm
    profile budget of 0.15s. It is a whole-league, whole-season computation
    with no per-player component, so it belongs behind the frame cache and is
    built exactly once per (database, refresh, scoring format). Counting
    calls proves that; timing the second click would only prove a cache
    exists."""
    from scoring import profile_cache
    conn = _seed_card_fixture(tmp_path, snaps=_card_snap_counts(),
                              depth=_card_depth_charts(),
                              players=_card_players_with_line())
    calls = []
    real = profile_cache.line_quality
    profile_cache.line_quality = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        # `big` is the only one of the three on a team with a rated line, so
        # he is the one whose payload proves the frame arrived at all.
        assert build_profile(conn, "big")["oline"] is not None
        for pid in ("small", "short", "big"):
            build_profile(conn, pid)
    finally:
        profile_cache.line_quality = real
    assert calls == [1]


def test_the_position_rank_frames_are_built_once_per_database_too(tmp_path):
    """Same argument, same place: ranking every player-season of every
    position, and assembling the comparable pool, are cross-player
    computations. Four clicks, one build."""
    from scoring import profile_cache
    conn = _seed_card_fixture(tmp_path)
    ranks, pools = [], []
    real_r, real_p = profile_cache.season_rank_frame, profile_cache.comparable_pool
    profile_cache.season_rank_frame = lambda *a, **k: (ranks.append(1), real_r(*a, **k))[1]
    profile_cache.comparable_pool = lambda *a, **k: (pools.append(1), real_p(*a, **k))[1]
    try:
        for pid in ("big", "small", "short", "big"):
            build_profile(conn, pid)
    finally:
        profile_cache.season_rank_frame = real_r
        profile_cache.comparable_pool = real_p
    assert ranks == [1] and pools == [1]


def test_the_rank_frame_is_priced_by_the_leagues_scoring_not_always_ppr(tmp_path):
    """The coefficient divides one rules-priced number by another, so a
    cached frame handed to the wrong league would put a PPR volatility beside
    a standard-league average. profile_cache keys on the rules; this pins
    that the numbers actually move with them."""
    conn = _seed_format_fixture(tmp_path)
    p = _profiles_by_format(conn)
    # ppg 20.0/17.0/14.0 with sigma 2.11/1.05/0.0 -> the coefficient shrinks
    # to nothing as receptions stop being points.
    assert _season(p["ppr"], 2025)["cv"] == round(2.108185106778921 / 20.0, 3)
    assert _season(p["half"], 2025)["cv"] == round(1.0540925533894598 / 17.0, 3)
    assert _season(p["std"], 2025)["cv"] == 0.0


# -- news + injury status (pipeline/news.py -> the profile payload) ----------
#
# `player_news` and `player_status` were written by pipeline/news.py and read
# by nobody. These pin the three things that could quietly go wrong on the
# way to the card: the attribution marker being flattened away, the feed
# arriving in an order nobody chose, and a player with no news or no injury
# being given one.

def _news_rows(player_id="p1", name="Amon-Ra St. Brown"):
    """One player's feed, mixing both attributions, deliberately out of order.

    Two of these share a `published_at` to the second -- which is not a
    contrived case: Google sends batches with identical pubDates, and 114 of
    the 2,192 (player, published_at) groups in the real table hold more than
    one row.
    """
    from pipeline.news import ATTR_EXACT, ATTR_NAME
    ts = pd.Timestamp("2026-08-19 12:00:00")
    return pd.DataFrame([
        {"player_id": player_id, "name": name, "headline": "oldest name item",
         "url": "https://news.google.com/a", "published_at": ts - pd.Timedelta(days=3),
         "source": "Yahoo Sports", "attribution": ATTR_NAME, "fetched_at": ts},
        {"player_id": player_id, "name": name, "headline": "newest of all",
         "url": "https://espn.com/newest", "published_at": ts,
         "source": "ESPN", "attribution": ATTR_EXACT, "fetched_at": ts},
        {"player_id": player_id, "name": name, "headline": "tied, name-matched",
         "url": "https://news.google.com/tied-b", "published_at": ts - pd.Timedelta(days=1),
         "source": "CBS Sports", "attribution": ATTR_NAME, "fetched_at": ts},
        {"player_id": player_id, "name": name, "headline": "tied, id-tagged",
         "url": "https://espn.com/tied-a", "published_at": ts - pd.Timedelta(days=1),
         "source": "ESPN", "attribution": ATTR_EXACT, "fetched_at": ts},
    ])


def _status_row(player_id="p1", name="Amon-Ra St. Brown", position="WR",
                team="DET", **over):
    row = {"player_id": player_id, "name": name, "position": position,
           "team": team, "sleeper_id": "4035",
           "injury_status": None, "injury_body_part": None,
           "injury_notes": None, "practice_participation": None,
           "depth_chart_position": "LWR", "depth_chart_order": 1,
           "news_updated": pd.Timestamp("2026-08-18 09:30:00"),
           "fetched_at": pd.Timestamp("2026-08-19 12:00:00")}
    row.update(over)
    return pd.DataFrame([row])


def test_player_news_is_newest_first_and_breaks_a_tie_toward_the_exact_item(tmp_path):
    """Recency is the primary key and attribution only the tiebreak, on
    purpose: the marker says how sure we are that an item is about this
    player, not how much it matters. On the real table not one of the 114
    ties mixes the two attributions, so this fixture manufactures the case
    the rule exists for."""
    from scoring.profile import player_news
    conn = get_conn(str(tmp_path / "news.duckdb"))
    write_table(conn, "player_news", _news_rows())

    items = player_news(conn, "p1")
    assert [i["headline"] for i in items] == [
        "newest of all",            # 08-19
        "tied, id-tagged",          # 08-18, exact wins the tie
        "tied, name-matched",       # 08-18
        "oldest name item"]         # 08-16
    # Sorted, not merely stable: reading them back in a different physical
    # order must not change the answer.
    write_table(conn, "player_news", _news_rows().iloc[::-1].reset_index(drop=True))
    assert [i["headline"] for i in player_news(conn, "p1")] == [
        "newest of all", "tied, id-tagged", "tied, name-matched",
        "oldest name item"]


def test_player_news_keeps_the_two_attributions_apart(tmp_path):
    """THE ONE THING THAT MUST NOT BE FLATTENED. An ESPN athlete-id match is
    a fact; a name+team search hit is a ~93-96% guess. The card presents them
    differently, so the marker travels verbatim -- not as a boolean, not
    dropped, and not translated into some third vocabulary of this file's
    own invention."""
    from pipeline.news import ATTR_EXACT, ATTR_NAME
    from scoring.profile import player_news
    conn = get_conn(str(tmp_path / "news.duckdb"))
    write_table(conn, "player_news", _news_rows())

    items = player_news(conn, "p1")
    assert {i["attribution"] for i in items} == {ATTR_EXACT, ATTR_NAME}
    exact = [i for i in items if i["attribution"] == ATTR_EXACT]
    assert {i["headline"] for i in exact} == {"newest of all", "tied, id-tagged"}
    # And the rest of the item is there to be rendered, not just the marker.
    assert exact[0]["source"] == "ESPN"
    assert exact[0]["url"] == "https://espn.com/newest"
    assert exact[0]["published_at"] == "2026-08-19T12:00:00"


def test_player_news_caps_the_feed(tmp_path):
    """The cap is a real trim, not decoration: the stored feed runs to 14
    items for some players and the median is 10. See NEWS_ITEMS for the
    bytes it is protecting."""
    from scoring.profile import NEWS_ITEMS, player_news
    from pipeline.news import ATTR_NAME
    conn = get_conn(str(tmp_path / "news.duckdb"))
    many = pd.DataFrame([
        {"player_id": "p1", "name": "Amon-Ra St. Brown",
         "headline": f"item {i}", "url": f"https://news.google.com/{i}",
         "published_at": pd.Timestamp("2026-08-19 12:00:00") - pd.Timedelta(hours=i),
         "source": "Yahoo Sports", "attribution": ATTR_NAME,
         "fetched_at": pd.Timestamp("2026-08-19 12:00:00")}
        for i in range(30)])
    write_table(conn, "player_news", many)

    items = player_news(conn, "p1")
    assert len(items) == NEWS_ITEMS
    # ...and it keeps the NEWEST NEWS_ITEMS, not the first NEWS_ITEMS rows.
    assert [i["headline"] for i in items] == [f"item {i}" for i in range(NEWS_ITEMS)]
    assert len(player_news(conn, "p1", limit=3)) == 3


def test_player_news_is_empty_rather_than_invented(tmp_path):
    """Three ways to have no news, all of them an empty list. The first is
    not hypothetical: data/nfl.duckdb had no `player_news` table at all when
    this was wired, because the table only exists after a `make refresh`."""
    from scoring.profile import player_news
    conn = get_conn(str(tmp_path / "news.duckdb"))
    assert player_news(conn, "p1") == []              # no table

    write_table(conn, "player_news", _news_rows())
    assert player_news(conn, "nobody") == []          # no rows for him

    write_table(conn, "player_news", _news_rows().drop(columns=["attribution"]))
    assert player_news(conn, "p1") == []              # no attribution column


def test_player_status_carries_the_injury_and_the_depth_chart(tmp_path):
    from scoring.profile import player_status
    conn = get_conn(str(tmp_path / "st.duckdb"))
    write_table(conn, "player_status", _status_row(
        injury_status="Questionable", injury_body_part="Knee - ACL",
        injury_notes="Surgery"))

    s = player_status(conn, "p1")
    assert s["injury_status"] == "Questionable"
    assert s["injury_body_part"] == "Knee - ACL"
    assert s["injury_notes"] == "Surgery"
    assert s["depth_chart_position"] == "LWR" and s["depth_chart_order"] == 1
    assert s["news_updated"] == "2026-08-18T09:30:00"
    assert s["source"] == "sleeper"
    # Measured null for all 3,186 active rostered players in the live
    # payload, so it is deliberately not served. See `player_status`.
    assert "practice_participation" not in s


def test_player_status_never_says_healthy(tmp_path):
    """Sleeper publishes Questionable / IR / PUP / Sus / Out / Doubtful /
    NA / DNR / COV, and null. There is no "Healthy" in the vocabulary
    (measured across all 12,221 players), so a player with nothing wrong
    with him gets a null and the card decides what to draw. Inventing the
    word here would be this payload claiming something Sleeper did not."""
    from scoring.profile import player_status
    conn = get_conn(str(tmp_path / "st.duckdb"))
    write_table(conn, "player_status", _status_row())

    s = player_status(conn, "p1")
    assert s is not None                    # we DID look him up
    assert s["injury_status"] is None       # and Sleeper said nothing
    assert s["injury_body_part"] is None and s["injury_notes"] is None


def test_player_status_is_none_when_sleeper_does_not_carry_him(tmp_path):
    """parse_sleeper_status reindexes onto the whole board, so every board
    player HAS a row -- a null `sleeper_id` is what "we looked and Sleeper
    has never heard of him" looks like. That is all 26 defenses and 1 of 24
    kickers. None, rather than a dict of nulls the client has to inspect
    field by field."""
    from scoring.profile import player_status
    conn = get_conn(str(tmp_path / "st.duckdb"))
    assert player_status(conn, "p1") is None          # no table at all

    write_table(conn, "player_status", pd.concat([
        _status_row(),
        _status_row("d1", "Cowboys", "DST", "DAL", sleeper_id=None,
                    depth_chart_position=None, depth_chart_order=None,
                    news_updated=None)], ignore_index=True))
    assert player_status(conn, "d1") is None
    assert player_status(conn, "p1") is not None
    assert player_status(conn, "nobody") is None      # no row


def _seed_with_news(tmp_path):
    from scoring import board_cache, profile_cache
    conn = _seed(tmp_path)
    write_table(conn, "player_news", _news_rows())
    write_table(conn, "player_status", _status_row(injury_status="Questionable",
                                                   injury_body_part="Hamstring"))
    # The tables were written straight through write_table without touching
    # `meta`, which is the one staleness gap scoring/profile_cache.py
    # documents -- and it bites harder for these two than for the rest,
    # because CREATING the table is the change that matters.
    board_cache.clear(); profile_cache.clear()
    return conn


def test_build_profile_serves_the_feed_and_the_status_beside_the_header(tmp_path):
    """The wiring, end to end. `status` is a sibling of `header` rather than
    something nested in the feed: a manager wants to see Questionable before
    he spends the pick, in one lookup from the component that draws the
    name."""
    p = build_profile(_seed_with_news(tmp_path), "p1")
    assert list(p)[:2] == ["header", "status"]
    assert p["status"]["injury_status"] == "Questionable"
    assert p["status"]["injury_body_part"] == "Hamstring"
    assert [i["headline"] for i in p["news"]][0] == "newest of all"
    assert len(p["news"]) == 4


def test_news_and_status_are_additive_to_a_payload_that_does_not_move(tmp_path):
    """Every pre-existing key, value for value, against the same fixture
    without the two tables. (Checked the same way against data/nfl.duckdb
    for all 249 board players: `news` and `status` added, nothing else
    changed -- see .superpowers/sdd/payload-wiring-report.md.)"""
    from scoring import board_cache, profile_cache
    board_cache.clear(); profile_cache.clear()
    without = build_profile(_seed(tmp_path / "a"), "p1")
    with_news = build_profile(_seed_with_news(tmp_path / "b"), "p1")

    # The two keys are always PRESENT -- an empty feed is `news: []`, not a
    # missing key, so a client never has to branch on whether the pipeline
    # has run. What changes is only their contents.
    assert set(with_news) == set(without)
    assert without["news"] == [] and without["status"] is None
    assert with_news["news"] and with_news["status"] is not None
    for key in without:
        if key in ("news", "status"):
            continue
        assert with_news[key] == without[key], key


def test_the_new_keys_survive_json_with_no_nan_and_no_datetimes(tmp_path):
    """`_scrub` passes a datetime straight through -- it is a scalar and it
    is not NaN -- so `json.dumps(..., allow_nan=False)` would raise
    TypeError on one. Every timestamp is stringified on the way out
    (`_ts`), which is also how `player_bio` has always handled a birth
    date."""
    import json
    p = build_profile(_seed_with_news(tmp_path), "p1")
    json.dumps(p, allow_nan=False)
    assert all(isinstance(i["published_at"], str) for i in p["news"])
    assert isinstance(p["status"]["news_updated"], str)


def test_a_kicker_gets_a_feed_and_a_defense_gets_an_empty_one(tmp_path):
    """Neither may raise, and neither may be given something it has not got.

    A kicker is a person somebody writes about: on the real table all 24 of
    them have a feed and 23 of 24 match Sleeper. A defense has neither --
    pipeline/news.py runs no query for a DST at all, because "Denver
    Defense" as a search phrase returns whatever the newspaper wrote about
    the Broncos, and Sleeper has no such player to match.
    """
    import json
    from scoring import board_cache, profile_cache
    from scoring.board import build_board
    conn = _seed_kicker_fixture(tmp_path)
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Kick Guy", "position": "PK", "team": "DAL", "adp": 140.0},
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "Cowboys", "position": "DST", "team": "DAL", "adp": 150.0}]))
    write_table(conn, "player_news", _news_rows("k1", "Kick Guy"))
    write_table(conn, "player_status", _status_row(
        "k1", "Kick Guy", "K", "DAL", depth_chart_position="K"))
    board_cache.clear(); profile_cache.clear()
    settings = _kicker_settings(_kicker_rules())
    dst_id = build_board(conn, settings=settings).pipe(
        lambda b: b[b["position"] == "DST"].iloc[0]["player_id"])

    kicker = build_profile(conn, "k1", None, settings)
    assert len(kicker["news"]) == 4 and kicker["status"]["depth_chart_position"] == "K"

    defense = build_profile(conn, dst_id, None, settings)
    json.dumps(defense, allow_nan=False)
    assert defense["news"] == []            # no feed exists for a team defense
    assert defense["status"] is None        # and Sleeper has no such player


def test_the_column_lists_for_the_two_news_tables_are_cached_not_re_queried(tmp_path):
    """Both reads have to know the table is there before they can query it,
    and that information_schema round trip measured 0.53 + 0.53 ms against
    0.62 + 0.11 ms for the queries themselves -- the schema lookup cost more
    than the reads did. They ride in ProfileFrames beside `snap_columns`,
    which is there for the identical reason, and carrying them is what takes
    the whole addition from ~2.6 ms a click to 1.6 ms."""
    from scoring import profile_cache
    conn = _seed_with_news(tmp_path)
    build_profile(conn, "p1")

    frames = next(iter(profile_cache._cache.values()))
    assert "attribution" in frames.news_columns
    assert "injury_status" in frames.status_columns
    # A database with neither table carries the empty sets, which is what
    # makes `news: []` / `status: None` the answer without a query.
    profile_cache.clear()
    plain = _seed(tmp_path / "plain")
    build_profile(plain, "p1")
    frames = next(iter(profile_cache._cache.values()))
    assert frames.news_columns == frozenset()
    assert frames.status_columns == frozenset()


def test_an_all_null_status_table_is_none_rather_than_a_dict_of_nans(tmp_path):
    """The shape a DST-only board, or a failed Sleeper fetch, actually
    produces: pandas gives an all-null column no dtype to work with, DuckDB
    stores it as DOUBLE, and it comes back as float nan -- not None. An
    `is None` guard would sail past that and put nans in the payload, where
    `json.dumps(allow_nan=False)` raises rather than showing anything."""
    import json
    from scoring.profile import player_status
    conn = get_conn(str(tmp_path / "st.duckdb"))
    write_table(conn, "player_status", pd.DataFrame([
        {"player_id": "d1", "name": "Cowboys", "position": "DST", "team": "DAL",
         "sleeper_id": None, "injury_status": None, "injury_body_part": None,
         "injury_notes": None, "practice_participation": None,
         "depth_chart_position": None, "depth_chart_order": None,
         "news_updated": None, "fetched_at": None}]))
    assert conn.execute("SELECT typeof(sleeper_id) FROM player_status").fetchone()[0] != "VARCHAR"
    assert player_status(conn, "d1") is None


def test_a_feed_with_no_publication_names_serves_nulls_not_nans(tmp_path):
    """Same failure one table over: `source` is optional in the RSS -- Google
    sends it, a hand-written row need not -- and an all-null column arrives
    as float nan. The item must carry `source: null` and stay serialisable."""
    import json
    from pipeline.news import ATTR_NAME
    from scoring.profile import player_news
    conn = get_conn(str(tmp_path / "news.duckdb"))
    write_table(conn, "player_news", pd.DataFrame([
        {"player_id": "p1", "name": "A Star", "headline": "no publication",
         "url": "https://news.google.com/z",
         "published_at": pd.Timestamp("2026-08-19 12:00:00"), "source": None,
         "attribution": ATTR_NAME,
         "fetched_at": pd.Timestamp("2026-08-19 12:00:00")}]))

    items = player_news(conn, "p1")
    assert items[0]["source"] is None
    json.dumps(items, allow_nan=False)


def test_proj_pos_finish_ranks_within_the_position_best_first():
    """The projection as a place, so it can sit on the same ladder as a
    played season's finish rather than beside it in another unit."""
    board = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "position": ["RB", "RB", "WR"],
        "proj_points": [300.0, 250.0, 280.0],
    })
    assert _proj_pos_finish(board, "a") == 1
    assert _proj_pos_finish(board, "b") == 2
    # Ranked within his OWN position: the best receiver is WR1, not WR2
    # because two backs outscore him.
    assert _proj_pos_finish(board, "c") == 1


def test_proj_pos_finish_is_none_without_a_projection():
    """A player the board could not price gets no rank at all. A finish
    invented for a player with nothing to rank would be the one number on the
    card that came from nowhere, and it would sort him against players the
    same column is measuring honestly."""
    board = pd.DataFrame({
        "player_id": ["a", "b"],
        "position": ["RB", "RB"],
        "proj_points": [300.0, None],
    })
    assert _proj_pos_finish(board, "b") is None
    # The unpriced player is not in the pool, so he does not push anyone down.
    assert _proj_pos_finish(board, "a") == 1


def test_proj_pos_finish_ties_share_the_better_place():
    """`method="min"`, which is what a finish means everywhere else here: two
    identical projections are both RB1, not both RB1.5."""
    board = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "position": ["RB", "RB", "RB"],
        "proj_points": [300.0, 300.0, 200.0],
    })
    assert _proj_pos_finish(board, "a") == 1
    assert _proj_pos_finish(board, "b") == 1
    assert _proj_pos_finish(board, "c") == 3
