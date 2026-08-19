import pandas as pd
from scoring.ppr import compute_ppr_points

def test_receiver_line():
    # 8 rec, 100 yds, 1 TD = 8 + 10 + 6 = 24.0
    df = pd.DataFrame([{"receptions": 8, "receiving_yards": 100, "receiving_tds": 1}])
    assert compute_ppr_points(df).iloc[0] == 24.0

def test_qb_line():
    # 300 pass yds, 3 TD, 1 INT, 20 rush yds = 12 + 12 - 2 + 2 = 24.0
    # The nflverse stats_player_week schema (2016-2025 verified) names the
    # column `passing_interceptions`; a rule keyed "interceptions" silently
    # never applies because the scorer zero-fills missing columns.
    df = pd.DataFrame([{"passing_yards": 300, "passing_tds": 3,
                        "passing_interceptions": 1, "rushing_yards": 20}])
    assert compute_ppr_points(df).iloc[0] == 24.0

def test_fumbles_and_2pt():
    df = pd.DataFrame([{"rushing_fumbles_lost": 1, "sack_fumbles_lost": 1,
                        "rushing_2pt_conversions": 1}])
    assert compute_ppr_points(df).iloc[0] == -2.0

def test_missing_columns_are_zero():
    df = pd.DataFrame([{"receptions": 5}])
    assert compute_ppr_points(df).iloc[0] == 5.0

def test_custom_rules_override_defaults():
    from scoring.ppr import compute_ppr_points
    import pandas as pd
    df = pd.DataFrame([{"receptions": 5, "receiving_yards": 100}])
    assert compute_ppr_points(df).iloc[0] == 15.0            # 5*1.0 + 100*0.1
    half = compute_ppr_points(df, {"receptions": 0.5, "receiving_yards": 0.1})
    assert half.iloc[0] == 12.5                              # 5*0.5 + 100*0.1

def test_an_empty_rules_dict_scores_nothing_rather_than_reverting_to_ppr():
    """`rules or DEFAULT_RULES` treats {} as "unspecified".

    league.from_espn genuinely produces `scoring == {}` when none of a
    league's ESPN scoring items map to an nflverse column, and quietly
    scoring that league on standard PPR is worse than scoring it at zero:
    zero is visibly wrong, a plausible-looking standard-PPR board is not.
    """
    df = pd.DataFrame([{"receptions": 8, "receiving_yards": 100,
                        "receiving_tds": 1}])
    assert compute_ppr_points(df, {}).iloc[0] == 0.0
    assert compute_ppr_points(df, None).iloc[0] == 24.0


def test_normalize_rules_collapses_a_reordered_ppr_rule_set():
    """`league.from_espn` builds `scoring` in ESPN's `scoring_items` order --
    the owner's own imported league starts at `rushing_tds`, not
    `passing_yards` -- and `compute_ppr_points` accumulates term by term in
    dict order, so the same 14 rules summed in a different order differ in
    the last bits of a float. Measured on data/nfl.duckdb, 478 of the 54,473
    weekly rows from 2023 on come out different, by up to 7.1e-15.

    Harmless in a points column; not harmless in `board.projection_scale`,
    which divides two such sums and would land at 0.999999999999999x instead
    of 1.0 -- shifting every proj_points, and with it vor and the board's
    order, for a league whose scoring did not change at all.
    """
    from scoring.ppr import DEFAULT_RULES, normalize_rules
    reordered = {k: DEFAULT_RULES[k] for k in reversed(list(DEFAULT_RULES))}
    assert list(reordered) != list(DEFAULT_RULES)
    assert normalize_rules(None) is None
    assert normalize_rules(DEFAULT_RULES) is None
    assert normalize_rules(reordered) is None

    # A real rule change is NOT collapsed, and neither is {} -- "nothing
    # scores" stays a different answer from full PPR (see the test above).
    half = {**DEFAULT_RULES, "receptions": 0.5}
    assert normalize_rules(half) == half
    assert normalize_rules({}) == {}


def test_the_reordering_hazard_normalize_rules_exists_for_is_real():
    """Not a theoretical float argument: build the same season two ways.

    Sixteen weeks of a receiving line summed under DEFAULT_RULES and under
    the same rules reversed. If these were guaranteed bit-identical
    `normalize_rules` would be pointless -- and on the real weekly table they
    are not (478 rows out of 54,473). This pins the mechanism rather than the
    exact rows: the two rule sets score identically to within a rounding
    error, which is precisely why a ratio of them must not be trusted to be
    exactly 1.0.
    """
    from scoring.ppr import DEFAULT_RULES, compute_ppr_points, normalize_rules
    reordered = {k: DEFAULT_RULES[k] for k in reversed(list(DEFAULT_RULES))}
    df = pd.DataFrame([{"receptions": 7, "receiving_yards": 83.3,
                        "receiving_tds": 1, "rushing_yards": 12.7,
                        "passing_yards": 0.0}] * 16)
    a = compute_ppr_points(df, None).sum()
    b = compute_ppr_points(df, reordered).sum()
    assert abs(a - b) < 1e-9
    # ...and the normalized call is the identical call, so it cannot differ.
    assert compute_ppr_points(df, normalize_rules(reordered)).sum() == a
