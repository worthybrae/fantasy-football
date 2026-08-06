import pandas as pd

_RULES = {
    "passing_yards": 0.04, "passing_tds": 4.0, "passing_interceptions": -2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receptions": 1.0, "receiving_yards": 0.1, "receiving_tds": 6.0,
    "sack_fumbles_lost": -2.0, "rushing_fumbles_lost": -2.0, "receiving_fumbles_lost": -2.0,
    "passing_2pt_conversions": 2.0, "rushing_2pt_conversions": 2.0, "receiving_2pt_conversions": 2.0,
}

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0)
    return pd.Series(0.0, index=df.index)

def compute_ppr_points(df: pd.DataFrame) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for col, pts in _RULES.items():
        total = total + _col(df, col) * pts
    return total
