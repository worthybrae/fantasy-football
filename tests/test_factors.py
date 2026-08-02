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
