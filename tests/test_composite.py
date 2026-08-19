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
    """Test that 2-player group never splits tiers (designed behavior).

    Tier breaks require at least 3 players (2 real gaps) because mean+std
    is undefined for a single gap. Groups with ≤2 players remain tier 1 by design,
    regardless of gap size.
    """
    df = _df(2)
    df["composite"] = [100.0, 20.0]  # huge gap (80 points)
    df["vor"] = df["composite"]
    out = assign_tiers(df)
    tiers = out.sort_values("vor", ascending=False)["tier"].tolist()
    # With only 2 players, len(gaps) == 1 < 2, so threshold = inf
    # Both should be tier 1 by design
    assert tiers[0] == 1 and tiers[1] == 1

def test_vor_and_tiers_mixed_positions():
    """Test VOR and tier assignment with mixed positions (WR and RB) in single DataFrame.

    Verifies: (a) each position's replacement math is independent,
    (b) no cross-contamination of VOR values across positions via index misalignment.
    """
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

    # Cross-contamination guard: compute WR-only frame separately and verify
    # top WR's VOR matches exactly (no index misalignment)
    wr_only_df = _df(30, pos="WR")
    wr_only_df["composite"] = compute_composite(wr_only_df, {"production": 1.0})
    wr_only_out = apply_vor(wr_only_df)
    wr_only_sorted = wr_only_out.sort_values("composite", ascending=False)

    top_wr_vor_mixed = wr_data.iloc[0]["vor"]
    top_wr_vor_alone = wr_only_sorted.iloc[0]["vor"]
    assert top_wr_vor_mixed == top_wr_vor_alone, \
        f"Top WR VOR mismatch: mixed={top_wr_vor_mixed}, alone={top_wr_vor_alone} (index misalignment check)"

    # Sanity check: both should be positive
    assert top_wr_vor_mixed > 0, "Top WR should have positive vor"
    assert rb_data.iloc[0]["vor"] > 0, "Top RB should have positive vor"

def test_apply_vor_accepts_custom_replacement_ranks():
    import pandas as pd
    from scoring.composite import apply_vor
    df = pd.DataFrame({"position": ["RB"] * 5,
                       "composite": [90.0, 80.0, 70.0, 60.0, 50.0]})
    # Replacement at RB2 (index 1) -> the 80.0 player is the baseline.
    out = apply_vor(df, {"RB": 2})
    assert out.sort_values("composite", ascending=False)["vor"].tolist() == [
        10.0, 0.0, -10.0, -20.0, -30.0]

def test_apply_vor_differences_the_named_column():
    """VOR must be expressible in projected points, not just composite.

    The board ranks across positions on this number, and a difference of
    two within-position percentiles has no cross-position meaning -- that
    is what put a TE at ADP 149 thirteenth overall.
    """
    df = pd.DataFrame({
        "player_id": ["a", "b", "c", "d"],
        "position": ["RB", "RB", "TE", "TE"],
        "composite": [90.0, 50.0, 90.0, 50.0],
        "proj_points": [280.0, 150.0, 140.0, 100.0],
    })
    out = apply_vor(df, {"RB": 2, "TE": 2}, column="proj_points")
    assert list(out["vor"]) == [130.0, 0.0, 40.0, 0.0]


def test_apply_vor_still_defaults_to_composite():
    df = pd.DataFrame({
        "player_id": ["a", "b"],
        "position": ["RB", "RB"],
        "composite": [90.0, 50.0],
        "proj_points": [280.0, 150.0],
    })
    out = apply_vor(df, {"RB": 2})
    assert list(out["vor"]) == [40.0, 0.0]


def test_an_empty_replacement_ranks_dict_does_not_revert_to_the_hardcoded_ranks():
    """`replacement_ranks or REPLACEMENT_RANK` treats {} as "unspecified", so
    a league that derives no per-position ranks would silently be scored
    against this repo's hardcoded 8-team ones. {} must fall through to
    apply_vor's own per-position default (9) instead."""
    df = _df(25)
    df["composite"] = df["production"]
    ordered = lambda out: out.sort_values("composite", ascending=False)

    default = ordered(apply_vor(df))
    empty = ordered(apply_vor(df, {}))

    assert default.iloc[23]["vor"] == 0.0        # REPLACEMENT_RANK["WR"] == 24
    assert empty.iloc[8]["vor"] == 0.0           # apply_vor's own default, 9
    assert empty.iloc[23]["vor"] != 0.0
