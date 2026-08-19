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

def normalize_rules(rules: dict | None) -> dict | None:
    """`None` when `rules` prices exactly the built-in full-PPR ruleset.

    Every function that gained a `rules` argument in the scoring-format work
    calls this first, so a full-PPR league takes the SAME code path the
    rules-less version took -- `compute_ppr_points(df, None)`, term for term,
    in DEFAULT_RULES' own insertion order.

    THE BUG THIS PREVENTS, which is not hypothetical: `LeagueSettings.scoring`
    built by `league.from_espn` lists the same 14 rules the owner's real PPR
    league scores, in ESPN's `scoring_items` order rather than DEFAULT_RULES'
    (verified against data/nfl.duckdb -- `rushing_tds` first, `passing_yards`
    twelfth). `compute_ppr_points` accumulates `total = total + col * pts` in
    dict order, so the same 14 terms summed in a different order differ in the
    last bits of a float. That is invisible in a points column and NOT
    invisible downstream: `projections()` now divides two such sums to price
    ESPN's projection, and 0.9999999999999998 instead of 1.0 shifts every
    `proj_points`, which shifts `vor`, which can reorder two players who
    should have tied -- a re-ranked board for a league whose scoring did not
    change at all.

    Dicts compare by content, not order, so the reordered-but-identical case
    collapses here and the owner's PPR league is bit-for-bit what it was.
    `{}` is NOT normalized: an empty dict means "nothing scores" (see
    `compute_ppr_points`), which is a real, different answer from full PPR.
    """
    return None if rules is None or rules == DEFAULT_RULES else rules


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
