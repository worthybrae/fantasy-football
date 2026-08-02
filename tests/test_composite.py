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
