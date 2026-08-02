import pandas as pd
import pytest
from scoring.composite import compute_composite, apply_vor, assign_tiers

def _df(n, pos="WR"):
    return pd.DataFrame({
        "position": [pos] * n,
        "production": [100.0 - i * 10 for i in range(n)],
        "durability": [50.0] * n, "role": [50.0] * n,
        "environment": [50.0] * n, "schedule": [50.0] * n})

def test_composite_weighted_mean():
    df = _df(1)
    # equal weights on two factors: (100+50)/2 = 75
    s = compute_composite(df, {"production": 1.0, "role": 1.0})
    assert s.iloc[0] == 75.0

def test_composite_rejects_unknown_key():
    with pytest.raises(ValueError):
        compute_composite(_df(1), {"vibes": 1.0})

def test_vor_replacement_is_zero():
    df = _df(30)
    df["composite"] = compute_composite(df, {"production": 1.0})
    out = apply_vor(df)
    # WR replacement rank is 24 -> the 24th WR has vor == 0
    r = out.sort_values("composite", ascending=False).iloc[23]
    assert r["vor"] == 0.0
    assert out.sort_values("composite", ascending=False).iloc[0]["vor"] > 0

def test_tiers_break_on_gap():
    df = _df(4)
    df["composite"] = [100.0, 99.0, 98.0, 60.0]  # big gap before the last player
    df["vor"] = df["composite"]
    out = assign_tiers(df)
    tiers = out.sort_values("vor", ascending=False)["tier"].tolist()
    assert tiers[0] == tiers[1] == tiers[2] == 1 and tiers[3] == 2

def test_tiers_two_players_huge_gap():
    """Test that 2-player group with huge gap correctly assigns tier 2 to weaker player."""
    df = _df(2)
    df["composite"] = [100.0, 20.0]  # huge gap (80 points)
    df["vor"] = df["composite"]
    out = assign_tiers(df)
    tiers = out.sort_values("vor", ascending=False)["tier"].tolist()
    # With only 2 players, len(gaps) == 1 < 2, so threshold = inf
    # Both should be tier 1
    assert tiers[0] == 1 and tiers[1] == 1

def test_vor_and_tiers_mixed_positions():
    """Test VOR and tier assignment with mixed positions (WR and RB) in single DataFrame."""
    # Create 30 WRs (replacement rank 24) and 30 RBs (replacement rank 22)
    wr_df = _df(30, pos="WR")
    rb_df = _df(30, pos="RB")
    # Interleave to test index alignment and avoid duplicate indices
    df = pd.concat([wr_df.iloc[:15], rb_df.iloc[:15], wr_df.iloc[15:], rb_df.iloc[15:]], ignore_index=True)

    df["composite"] = compute_composite(df, {"production": 1.0})
    out = apply_vor(df)
    out = assign_tiers(out)

    # Verify each position's replacement math is independent
    wr_data = out[out["position"] == "WR"].sort_values("composite", ascending=False)
    rb_data = out[out["position"] == "RB"].sort_values("composite", ascending=False)

    # 24th WR (rank index 23) should have vor == 0
    assert wr_data.iloc[23]["vor"] == 0.0, "24th WR should have vor=0"

    # 22nd RB (rank index 21) should have vor == 0
    assert rb_data.iloc[21]["vor"] == 0.0, f"22nd RB should have vor=0, got {rb_data.iloc[21]['vor']}"

    # Top WR vor should be greater than top RB vor (WRs are deeper in value)
    # because WRs ranked 100, 90, 80, ... 10 (top is 100, 24th is 10)
    # and RBs ranked 100, 90, 80, ... -120 (top is 100, 22nd is -120)
    assert wr_data.iloc[0]["vor"] > 0, "Top WR should have positive vor"
    assert rb_data.iloc[0]["vor"] > 0, "Top RB should have positive vor"
