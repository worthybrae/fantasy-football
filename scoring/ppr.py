import pandas as pd

DEFAULT_RULES = {
    "passing_yards": 0.04, "passing_tds": 4.0, "passing_interceptions": -2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receptions": 1.0, "receiving_yards": 0.1, "receiving_tds": 6.0,
    "sack_fumbles_lost": -2.0, "rushing_fumbles_lost": -2.0, "receiving_fumbles_lost": -2.0,
    "passing_2pt_conversions": 2.0, "rushing_2pt_conversions": 2.0, "receiving_2pt_conversions": 2.0,
}
_RULES = DEFAULT_RULES  # back-compat for existing imports

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0)
    return pd.Series(0.0, index=df.index)

def compute_ppr_points(df: pd.DataFrame, rules: dict | None = None) -> pd.Series:
    # `rules or DEFAULT_RULES` would be wrong: an empty dict is falsy, and an
    # empty dict is exactly what league.from_espn produces when none of a
    # league's scoring items map to an nflverse column. Silently scoring that
    # league on standard PPR is worse than scoring it at zero, which at least
    # shows up. None means "not specified"; {} means "nothing scores".
    rules = DEFAULT_RULES if rules is None else rules
    total = pd.Series(0.0, index=df.index)
    for col, pts in rules.items():
        total = total + _col(df, col) * pts
    return total
