import pandas as pd
from scoring import factors

def _weekly(rows):
    base = {"player_id": "p1", "player_display_name": "A Player", "position": "WR",
            "recent_team": "MIN", "opponent_team": "GB", "season": 2025, "week": 1,
            "receptions": 5, "receiving_yards": 50, "targets": 8, "carries": 0}
    return pd.DataFrame([{**base, **r} for r in rows])

def test_production_recency_weighting():
    # 10 ppg in 2025 (weight .5), 20 ppg in 2024 (weight .3) -> (10*.5+20*.3)/.8 = 13.75
    wk = _weekly([
        {"season": 2025, "receptions": 10, "receiving_yards": 0},   # 10 pts, 1 game
        {"season": 2024, "receptions": 20, "receiving_yards": 0},   # 20 pts, 1 game
    ])
    prod = factors.production_factor(wk)
    assert abs(prod.iloc[0]["production_raw"] - 13.75) < 1e-6

def test_environment_implied_points():
    sched = pd.DataFrame([{"home_team": "MIN", "away_team": "GB",
                           "total_line": 50.0, "spread_line": 4.0, "week": 1}])
    env = factors.environment_factor(sched)
    env = env.set_index("team")["env_raw"]
    assert env["MIN"] == 27.0 and env["GB"] == 23.0

def test_bye_weeks():
    rows = [{"home_team": "MIN", "away_team": "GB", "week": w} for w in range(1, 19) if w != 7]
    byes = factors.bye_weeks(pd.DataFrame(rows)).set_index("team")["bye"]
    assert byes["MIN"] == 7

def test_normalize_within_position_neutral_fill():
    df = pd.DataFrame({"position": ["WR", "WR", "WR"], "x": [1.0, 3.0, None]})
    out = factors.normalize_within_position(df, "x", "x_n")
    assert out["x_n"].iloc[2] == 50.0
    assert out["x_n"].iloc[1] > out["x_n"].iloc[0]

def test_schedule_factor_points_allowed():
    # GB allowed 30 WR ppg in 2025; MIN plays GB twice in 2026
    wk25 = _weekly([{"receptions": 30, "receiving_yards": 0, "opponent_team": "GB"}])
    sched26 = pd.DataFrame([
        {"home_team": "MIN", "away_team": "GB", "week": 1},
        {"home_team": "GB", "away_team": "MIN", "week": 8},
    ])
    sos = factors.schedule_factor(wk25, sched26)
    row = sos[(sos["team"] == "MIN") & (sos["position"] == "WR")]
    assert row.iloc[0]["sos_raw"] == 30.0

def test_player_seasons():
    # 2 weeks with 10 pts each = 20 pts total, 2 games, ppg = 10
    wk = _weekly([
        {"week": 1, "receptions": 10, "receiving_yards": 0},
        {"week": 2, "receptions": 10, "receiving_yards": 0},
    ])
    ps = factors.player_seasons(wk)
    assert len(ps) == 1
    assert ps.iloc[0]["games"] == 2
    assert ps.iloc[0]["ppg"] == 10.0
    assert ps.iloc[0]["player_id"] == "p1"

def test_durability_factor():
    # 17 games in 1 season (2025) = 1.0 durability (all games available)
    wk = _weekly([{"week": w, "receptions": 1, "receiving_yards": 0} for w in range(1, 18)])
    dur = factors.durability_factor(wk)
    assert len(dur) == 1
    assert dur.iloc[0]["durability_raw"] == 1.0

def test_role_factor_with_depth_charts():
    # Starter (depth_team=1) should score higher than backup (depth_team=2)
    dc = pd.DataFrame([
        {"gsis_id": "p1", "depth_team": 1},  # starter
        {"gsis_id": "p2", "depth_team": 2},  # backup
    ])
    wk = _weekly([{"player_id": "p1", "targets": 8, "carries": 2}])
    wk = pd.concat([wk, pd.DataFrame([{"player_id": "p2", "player_display_name": "B Player",
                                      "position": "WR", "recent_team": "MIN", "opponent_team": "GB",
                                      "season": 2025, "week": 1, "receptions": 0, "receiving_yards": 0,
                                      "targets": 2, "carries": 0}])], ignore_index=True)
    rf = factors.role_factor(dc, wk)
    p1_role = rf[rf["player_id"] == "p1"].iloc[0]["role_raw"]
    p2_role = rf[rf["player_id"] == "p2"].iloc[0]["role_raw"]
    assert p1_role > p2_role

def test_role_factor_empty_depth_charts_returns_opp_share():
    # Empty depth_charts should still return opp_share signal, not empty result
    dc = pd.DataFrame(columns=["gsis_id", "depth_team"])  # empty
    wk = _weekly([{"player_id": "p1", "targets": 10, "carries": 0}])
    rf = factors.role_factor(dc, wk)
    assert len(rf) > 0  # must not be empty
    assert rf.iloc[0]["player_id"] == "p1"
    assert 0 < rf.iloc[0]["role_raw"] <= 1.0

def test_production_factor_zero_weight_no_crash():
    # Player with season not in RECENCY_WEIGHTS should not crash on division
    wk = _weekly([
        {"season": 2022, "receptions": 10, "receiving_yards": 0},  # 2022 not in weights
    ])
    prod = factors.production_factor(wk)
    # Should handle gracefully: either NaN or skip the season
    assert len(prod) == 1

def test_role_factor_zero_team_opportunities_no_crash():
    # Team with no targets/carries should not crash on division
    dc = pd.DataFrame([{"gsis_id": "p1", "depth_team": 1}])
    wk = _weekly([{"player_id": "p1", "targets": 0, "carries": 0}])
    rf = factors.role_factor(dc, wk)
    # Should handle gracefully: depth score alone used via skipna
    assert len(rf) == 1
