import pandas as pd

# The skill-position full-PPR ruleset, and ONLY that.
#
# DO NOT ADD KICKING (or defensive) TERMS HERE. This dict is not just a
# default -- `normalize_rules` below tests league rules against it by
# equality, and the owner's real ESPN league parses to exactly these 14
# entries (verified against data/nfl.duckdb's `league` table: same 14 keys,
# same values, different insertion order). Adding a 15th entry here would
# make that equality fail, which would push a league that scores precisely
# full PPR off the `rules is None` fast path and onto a summation in ESPN's
# key order -- the float-ordering bug `normalize_rules` exists to prevent,
# re-introduced for the one league it was written for. Kicking reaches the
# rules through `scoring.league.ESPN_STAT_COLUMNS` instead, which is where
# a league's ACTUAL kicking point values live.
DEFAULT_RULES = {
    "passing_yards": 0.04, "passing_tds": 4.0, "passing_interceptions": -2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receptions": 1.0, "receiving_yards": 0.1, "receiving_tds": 6.0,
    "sack_fumbles_lost": -2.0, "rushing_fumbles_lost": -2.0, "receiving_fumbles_lost": -2.0,
    "passing_2pt_conversions": 2.0, "rushing_2pt_conversions": 2.0, "receiving_2pt_conversions": 2.0,
}
_RULES = DEFAULT_RULES  # back-compat for existing imports

# Every nflverse `weekly` column a kicking scoring rule can name -- the
# right-hand side of every kicking entry in `scoring.league.ESPN_STAT_COLUMNS`,
# and nothing else.
#
# It answers exactly one question -- "does this league price kicking at
# all?" -- which two consumers need and MUST answer the same way, or they
# contradict each other on the same screen: scoring/profile.py decides
# whether a kicker's season history is real or a column of zeros, and
# scoring/board.py decides whether a kicker's production/durability/schedule
# factors are signal or noise. One predicate, one answer.
KICKING_COLUMNS = frozenset({
    "fg_made", "fg_att", "fg_missed", "fg_blocked",
    "fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
    "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
    "pat_made", "pat_att", "pat_missed", "pat_blocked",
})


def prices_kicking(rules: dict | None) -> bool:
    """True when `rules` gives any kicking stat a non-zero point value.

    `None` is full PPR (DEFAULT_RULES), which scores no kicking at all --
    so the answer for every caller that has not been handed a league's real
    ESPN scoring is False, exactly as it was before kicking was mappable.

    Non-zero, not merely present: ESPN emits scoring items it does not
    actually score (the owner's league carries statId 121 and 122 -- two
    points-allowed tiers -- at 0.0 points), and a rule worth zero points
    produces the same column of zeros the old kicker-blanking existed to
    hide. "Present but worth nothing" is not "priced".
    """
    if not rules:
        return False
    return any(col in KICKING_COLUMNS and pts for col, pts in rules.items())

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

    THE SECOND HALF OF THE SAME BUG, found when kicking became mappable.
    Once `scoring.league.ESPN_STAT_COLUMNS` understands ESPN's kicking ids,
    the owner's league stops parsing to 14 rules and starts parsing to 20 --
    the same 14 plus six kicking terms. That is no longer `== DEFAULT_RULES`,
    so the whole league would have fallen off the fast path above and back
    onto a sum in ESPN's key order: precisely the float-ordering shift
    documented above, re-introduced for the one league it was written for,
    by a change that was supposed to touch nothing but kickers.

    So the equality test is applied to the CORE ruleset -- the keys
    DEFAULT_RULES prices -- and a league whose core is exactly full PPR keeps
    DEFAULT_RULES' own key order, with its extra rules appended after. Every
    skill player then accumulates the same 14 terms in the same order as
    before, and the extra terms add `0 * points` (a kicking column is zero
    for anyone who is not a kicker), and `x + 0.0 == x` exactly. Verified,
    not assumed: the real 249-row board is byte-identical on every
    pre-existing column for QB/RB/WR/TE across this change.

    Deliberately NOT canonicalised for every league. Re-ordering a half-PPR
    league's rules would fix the same latent instability there (its board
    currently depends on the order ESPN happens to list its scoring items
    in), but it would also move that board's last bits today, for no benefit
    today. This does the minimum that keeps a real league still.
    """
    if rules is None or rules == DEFAULT_RULES:
        return None
    core = {k: v for k, v in rules.items() if k in DEFAULT_RULES}
    if core == DEFAULT_RULES:
        extra = {k: v for k, v in rules.items() if k not in DEFAULT_RULES}
        return {**DEFAULT_RULES, **extra}
    return rules


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
