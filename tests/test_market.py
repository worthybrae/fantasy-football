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

def test_duplicate_espn_id_lowest_rank_wins():
    """Regression: duplicate espn_id should not crash; lowest rank wins."""
    board = _board()
    sleeper = _sleeper()
    # Two ESPN rows for same espn_id (101), different ADP → different rank
    espn = pd.DataFrame({
        "espn_id": [101, 101, 102],
        "espn_name": ["Jahmyr Gibbs", "Jahmyr Gibbs (ALT)", "Amon-Ra St Brown"],
        "position": ["RB", "RB", "WR"],
        "espn_adp": [1.77, 2.5, 6.0],
        "espn_ppr_rank": [1, 2, 4]
    })
    fp = _fp()
    out = add_market(board, espn, fp, sleeper)
    g1 = out[out["player_id"] == "g1"].iloc[0]
    # Should use rank from lowest espn_adp (1.77 → rank 1), not second one
    assert g1["market_sources"]["espn"] == 1.0

def test_duplicate_gsis_id_in_crosswalk_lowest_rank_wins():
    """Regression: a junk crosswalk row mapping two different espn_ids to the
    same gsis_id (seen live: 00-0029981 "Duplicate Player") must not crash
    board building via a non-unique index in `.map()`; lowest espn_rank
    wins for that player."""
    board = _board()
    espn = pd.DataFrame({
        "espn_id": [101, 999],
        "espn_name": ["Jahmyr Gibbs", "Duplicate Player"],
        "position": ["RB", "RB"],
        "espn_adp": [1.77, 0.5],   # the duplicate has the better (lower) ADP
        "espn_ppr_rank": [1, 1],
    })
    # Crosswalk maps BOTH espn_ids to the same gsis_id "g1".
    sleeper = pd.DataFrame({
        "gsis_id": ["g1", "g1"], "espn_id": [101, 999],
        "sleeper_name": ["Jahmyr Gibbs", "Duplicate Player"],
        "position": ["RB", "RB"], "team": ["DET", "DET"],
    })
    fp = _fp()
    out = add_market(board, espn, fp, sleeper)  # must not raise
    g1 = out[out["player_id"] == "g1"].iloc[0]
    assert g1["market_sources"]["espn"] == 1.0

def test_duplicate_fp_name_position_lowest_rank_wins():
    """Regression: duplicate (fp_name, position) should not crash; lowest rank wins."""
    board = _board()
    sleeper = _sleeper()
    espn = _espn()
    # Two FP rows for Amon-Ra St. Brown WR, different rank_ecr
    fp = pd.DataFrame({
        "fp_name": ["Amon-Ra St. Brown", "Amon-Ra St. Brown", "Baltimore Ravens"],
        "team": ["DET", "DET", "BAL"],
        "position": ["WR", "WR", "DST"],
        "rank_ecr": [3, 5, 40],
        "rank_ave": [3.0, 5.0, 41.0],
        "rank_std": [1.0, 1.5, 5.0],
        "fp_tier": [1, 2, 5]
    })
    out = add_market(board, espn, fp, sleeper)
    g2 = out[out["player_id"] == "g2"].iloc[0]
    # Should use lowest rank (3), not second one (5); should use tier 1
    assert g2["market_sources"]["fp"] == 3.0
    assert g2["market_sources"]["fp_tier"] == 1
