import numpy as np
import pandas as pd
from scoring.market import add_market

def _board():
    return pd.DataFrame({
        "player_id": ["g1", "g2", "g3"],
        "name": ["Jahmyr Gibbs", "Amon-Ra St. Brown", "Ravens DST"],
        "position": ["RB", "WR", "DST"], "team": ["DET", "DET", "BAL"],
        "rank": [1, 2, 3], "adp": [1.6, 5.1, np.nan]})

def _espn():
    return pd.DataFrame({"espn_id": [101, 102], "espn_name": ["Jahmyr Gibbs", "Amon-Ra St Brown"],
                         "position": ["RB", "WR"], "espn_adp": [1.77, 6.0],
                         "espn_ppr_rank": [1, 4]})

def _sleeper():
    return pd.DataFrame({"gsis_id": ["g1"], "espn_id": [101],
                         "sleeper_name": ["Jahmyr Gibbs"], "position": ["RB"], "team": ["DET"]})

def _fp():
    return pd.DataFrame({"fp_name": ["Amon-Ra St. Brown", "Baltimore Ravens"],
                         "team": ["DET", "BAL"], "position": ["WR", "DST"],
                         "rank_ecr": [3, 40], "rank_ave": [3.0, 41.0],
                         "rank_std": [1.0, 5.0], "fp_tier": [1, 5]})

def test_consensus_all_paths():
    out = add_market(_board(), _espn(), _fp(), _sleeper())
    g1 = out[out["player_id"] == "g1"].iloc[0]   # FFC rank 1, ESPN rank 1 (via crosswalk)
    assert g1["market_rank"] == 1.0 and g1["market_sources"]["espn"] == 1.0
    g2 = out[out["player_id"] == "g2"].iloc[0]   # FFC 2, ESPN 2 (name fallback), FP 3
    assert g2["market_rank"] == round((2 + 2 + 3) / 3, 1)
    assert g2["market_spread"] == 1.0
    assert g2["market_sources"]["fp_tier"] == 1
    dst = out[out["player_id"] == "g3"].iloc[0]  # FP only, joined by team
    assert dst["market_sources"]["fp"] == 40.0 and dst["market_rank"] == 40.0
    assert pd.isna(dst["market_spread"])         # single source
    assert "adp" not in out.columns

def test_edge_uses_market_rank():
    out = add_market(_board(), _espn(), _fp(), _sleeper())
    g2 = out[out["player_id"] == "g2"].iloc[0]
    assert g2["edge"] == g2["market_rank"] - g2["rank"]

def test_all_sources_empty():
    empty = pd.DataFrame()
    out = add_market(_board(), empty, empty, empty)
    r = out.iloc[0]
    assert r["market_rank"] == 1.0              # FFC alone still ranks
    assert out[out["player_id"] == "g3"].iloc[0]["market_sources"]["ffc"] is None or \
           pd.isna(out[out["player_id"] == "g3"].iloc[0]["market_sources"]["ffc"])

def test_no_sources_at_all():
    board = _board().assign(adp=np.nan)
    empty = pd.DataFrame()
    out = add_market(board, empty, empty, empty)
    assert out["market_rank"].isna().all()
