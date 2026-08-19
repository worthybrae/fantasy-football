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

def apply_vor(df: pd.DataFrame, replacement_ranks: dict | None = None,
              column: str = "composite") -> pd.DataFrame:
    """Value over replacement, as a difference within `column`.

    `column` exists because the two callers want different units. The board
    ranks across positions and must difference `proj_points` -- a difference
    of two within-position percentiles (which is what `composite` is, see
    factors.normalize_within_position) has no cross-position meaning, and
    ranking on one put a TE at ADP 149 thirteenth overall. Callers that only
    compare within a position can keep the composite default.
    """
    # `replacement_ranks or REPLACEMENT_RANK` would be wrong: an empty dict is
    # falsy, so a league that genuinely derives no replacement ranks would
    # silently revert to this repo's hardcoded 8-team ones. None means "not
    # specified"; {} means "no per-position ranks", which falls through to the
    # per-position default below.
    ranks = REPLACEMENT_RANK if replacement_ranks is None else replacement_ranks
    out = df.copy()
    out["vor"] = 0.0
    for pos, grp in out.groupby("position"):
        ranked = grp.sort_values(column, ascending=False)
        idx = min(ranks.get(pos, 9), len(ranked)) - 1
        replacement = ranked.iloc[idx][column]
        out.loc[grp.index, "vor"] = grp[column] - replacement
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
