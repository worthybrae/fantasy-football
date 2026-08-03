import pandas as pd
from scoring.ppr import compute_ppr_points

def test_receiver_line():
    # 8 rec, 100 yds, 1 TD = 8 + 10 + 6 = 24.0
    df = pd.DataFrame([{"receptions": 8, "receiving_yards": 100, "receiving_tds": 1}])
    assert compute_ppr_points(df).iloc[0] == 24.0

def test_qb_line():
    # 300 pass yds, 3 TD, 1 INT, 20 rush yds = 12 + 12 - 2 + 2 = 24.0
    # The nflverse stats_player_week schema (2016-2025 verified) names the
    # column `passing_interceptions`; a rule keyed "interceptions" silently
    # never applies because the scorer zero-fills missing columns.
    df = pd.DataFrame([{"passing_yards": 300, "passing_tds": 3,
                        "passing_interceptions": 1, "rushing_yards": 20}])
    assert compute_ppr_points(df).iloc[0] == 24.0

def test_fumbles_and_2pt():
    df = pd.DataFrame([{"rushing_fumbles_lost": 1, "sack_fumbles_lost": 1,
                        "rushing_2pt_conversions": 1}])
    assert compute_ppr_points(df).iloc[0] == -2.0

def test_missing_columns_are_zero():
    df = pd.DataFrame([{"receptions": 5}])
    assert compute_ppr_points(df).iloc[0] == 5.0
