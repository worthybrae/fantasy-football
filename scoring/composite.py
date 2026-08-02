import pandas as pd
from scoring.config import DEFAULT_WEIGHTS, REPLACEMENT_RANK

FACTORS = list(DEFAULT_WEIGHTS)

def compute_composite(df: pd.DataFrame, weights: dict) -> pd.Series:
    unknown = set(weights) - set(FACTORS)
    if unknown:
        raise ValueError(f"unknown weight keys: {unknown}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    score = pd.Series(0.0, index=df.index)
    for name, w in weights.items():
        score = score + df[name] * (w / total)
    return score

def apply_vor(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vor"] = 0.0
    for pos, grp in out.groupby("position"):
        ranked = grp.sort_values("composite", ascending=False)
        idx = min(REPLACEMENT_RANK.get(pos, 9), len(ranked)) - 1
        replacement = ranked.iloc[idx]["composite"]
        out.loc[grp.index, "vor"] = grp["composite"] - replacement
    return out

def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Assign tiers within each position based on VOR gaps.

    Tiers break where the gap to the next player exceeds mean + std of all gaps
    in that position. Tier breaks require at least 3 players (2 real consecutive
    gaps) because you cannot compute meaningful mean + std from a single gap.
    Groups with ≤2 players remain tier 1 by design.

    Args:
        df: DataFrame with 'position' and 'vor' columns

    Returns:
        DataFrame copy with added 'tier' column (1 = best, incrementing downward)
    """
    out = df.copy()
    out["tier"] = 1
    for pos, grp in out.groupby("position"):
        ranked = grp.sort_values("vor", ascending=False)
        drops = -ranked["vor"].diff().fillna(0)  # positive gaps between consecutive players
        gaps = -ranked["vor"].diff().dropna()  # N-1 real gaps (excluding synthetic leading 0)
        if len(gaps) >= 2 and gaps.std() > 0:
            threshold = gaps.mean() + gaps.std()
        else:
            threshold = float("inf")
        tier = (drops > threshold).cumsum() + 1
        out.loc[ranked.index, "tier"] = tier
    return out
