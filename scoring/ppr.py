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
    actually score (the owner's league carries statId 123, 124 and 125 --
    three points-allowed tiers -- at `points` 0.0), and a rule worth zero
    points produces the same column of zeros the old kicker-blanking existed
    to hide. "Present but worth nothing" is not "priced".

    (This example USED to name 121 and 122. Re-read against the owner's live
    2025 settings, those two ids are not in his league's scoring items at
    all; 123/124/125 are the ones sitting at 0.0. The point stands, the ids
    were wrong. And note what "0.0" means for those three -- see `from_espn`
    in scoring/league.py: their real value is in `pointsOverrides["16"]`,
    which is a different thing from "worth nothing" and is why this
    predicate is about KICKING rules, not about scoring items in general.)
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


# ---------------------------------------------------------------------------
# Team defense
# ---------------------------------------------------------------------------
#
# WHY THIS IS KEYED BY ESPN STAT ID AND NOT BY AN nflverse COLUMN, which is
# the whole reason defenses are scorable at all now.
#
# `ESPN_STAT_COLUMNS` (scoring/league.py) has to IDENTIFY every id it maps: it
# re-derives the stat from nflverse's `weekly` table, so naming the wrong
# column silently mis-scores a whole position. That is a real risk and it is
# why so many ids are deliberately left unmapped there.
#
# Defenses do not go through that step at all. ESPN serves the D/ST stat line
# ITSELF -- `pipeline/sources.fetch_espn_dst` reads it from the same
# `leaguedefaults/3?view=kona_player_info` endpoint `fetch_espn_adp` already
# uses -- keyed by the SAME stat ids the league's own `scoringItems` are keyed
# by. So the value and the point value arrive from ESPN together, and scoring
# is ESPN's own arithmetic rather than our reconstruction of it. Nothing has
# to be identified for the number to be right.
#
# VERIFIED, not assumed. Summing `stat x points` over the ids a league gives a
# slot-16 `pointsOverrides` entry reproduces ESPN's own published
# `appliedTotal` EXACTLY -- 0 mismatches in 3744 defense-weeks, being all 32
# defenses x every week of 2018, 2020, 2022, 2023, 2024 and 2025, checked
# against each season's own scoring settings.
#
# THE TIER PROBLEM IS GONE, and this is why. Points allowed and yards allowed
# are scored by ESPN as nine-way step functions, which `column x points`
# cannot express -- IF you have to compute the bucket. You do not: ESPN emits
# the bucket ALREADY DECIDED, as nine one-hot ids per game. Measured over 544
# defense-weeks of 2025, exactly one id of each family is 1.0 in every single
# game (0 violations), and the band each one covers is exact:
#
#   points allowed   89: 0    90: 1-6    91: 7-13   92: 14-17  121: 18-21
#                   122: 22-27  123: 28-34  124: 35-45  125: 46+
#   yards allowed   128: <100  129: 100-199  130: 200-299  131: 300-349
#                   132: 350-399  133: 400-449  134: 450-499  135: 500-549
#                   136: 550+
#
# (Boundaries read off ESPN's own per-game `120` points-allowed and `127`
# yards-allowed totals; the yards bands agree with the ones scoring/league.py
# had already derived from nflverse.) A step function over a per-game team
# aggregate, handed over pre-bucketed, IS a `column x points` rule.
#
# 187-196 is a SECOND, identical points-allowed family (187 duplicates 120,
# and 188-196 are one-hot with the same bands as 89-125, verified row for row
# over all 544 defense-weeks). A league that priced both would double-count --
# and so would ESPN, in ESPN's own scoring. Mirroring ESPN is the correct
# behaviour here, exactly as it already is for kicking's 74 vs 198/201 split.
DEFAULT_DST_RULES = {
    "89": 5.0, "90": 4.0, "91": 3.0, "92": 1.0,          # points allowed 0 / 1-6 / 7-13 / 14-17
    "93": 6.0,                                            # blocked-kick-return TD
    "95": 2.0, "96": 2.0, "97": 2.0, "98": 2.0, "99": 1.0,  # INT / fumble rec / blocked kick / safety / sack
    "101": 6.0, "102": 6.0, "103": 6.0, "104": 6.0,       # KR / PR / INT-return / fumble-return TD
    "123": -1.0, "124": -3.0, "125": -5.0,                # points allowed 28-34 / 35-45 / 46+
    "128": 5.0, "129": 3.0, "130": 2.0,                   # yards allowed <100 / 100-199 / 200-299
    "132": -1.0, "133": -3.0, "134": -5.0,                # yards allowed 350-399 / 400-449 / 450-499
    "135": -6.0, "136": -7.0,                             # yards allowed 500-549 / 550+
    "206": 2.0,                                           # defensive 2pt return
    "209": 1.0,                                           # unidentified -- see below
}
# DEFAULT_DST_RULES is ESPN's leaguedefaults/3 D/ST scoring, read live from
# `.../seasons/2025/segments/0/leaguedefaults/3?view=mSettings` -- the SAME
# default league this pipeline already takes `espn_proj` and the PPR rank from
# (pipeline/sources.ESPN_URL). It is the right default for the same reason
# DEFAULT_RULES is: it is what an ESPN league scores before anyone changes it.
# It is not this deployment's league guessed at -- though it happens to be
# exactly that: all 46 of the owner's league's scoring items are identical to
# leaguedefaults/3's, value for value and override for override (diffed
# programmatically, empty diff).
#
# It is deliberately NOT merged into DEFAULT_RULES. That dict is compared by
# equality in `normalize_rules` to keep a full-PPR league on the fast path;
# adding a 15th key would push the owner's league off it and move every
# skill-position number in the last bits of a float. Same reason kicking is
# not in there.
#
# THE COMMENTS ABOVE NAME THE IDS, THE SCORING DOES NOT DEPEND ON THE NAMES.
# Each name was established per-week against this database's `weekly` table,
# over all 544 defense-weeks of 2025, and only where the agreement was near
# total: 95 = interceptions (273/273 non-zero weeks agree, season 380 = 380),
# 97 = blocked kicks (43/43, 44 = 44), 103 = interception-return TD (28/28,
# 28 = 28), 206 = defensive 2pt return (2/2, exact on all 544 rows),
# 96 = opponent fumble recoveries (194/198, 242 vs 246), 106... is not priced,
# 99 = sacks (466/473, 1285 vs 1278 -- nflverse carries half-sacks),
# 98 = safeties (season total 12 = 12; the six weekly disagreements are three
# games where ESPN credits the defense and nflverse credits no player),
# 104 = fumble-return TD (16/19, 18 vs 19). 93/101/102 are the three
# special-teams return TDs and sum to 26 against nflverse's 27
# `special_teams_tds`, which is why they are named as a group and not
# individually pinned.
#
# `209` IS NOT IDENTIFIED AND IS NOT GUESSED. No defense carries a non-zero
# value for it in any season checked -- it never appears in a D/ST stat line
# at all. It is priced here because ESPN prices it, keyed by ESPN's own id: if
# ESPN ever emits it, we score it exactly as ESPN would. That is the
# difference between this map and ESPN_STAT_COLUMNS -- there is no column to
# choose, so there is no guess to make, and an unnamed id costs nothing.
#
# 94, 105, 205 are ROLLUPS ESPN publishes alongside the components (94 = 103 +
# 104, 105 = 93 + 94 + 101 + 102, 205 = 206, all verified row for row over 544
# defense-weeks). No league in evidence prices them, and they are absent here
# because they are absent from ESPN's scoring items -- not because we chose to
# drop them.


def dst_stat_column(stat_id) -> str:
    """The `dst_weekly` column holding ESPN stat `stat_id`.

    One function so the writer (pipeline/sources.parse_espn_dst) and the
    reader (`compute_dst_points`) cannot drift. A bare id would be a terrible
    column name -- `89` is not an identifier and DuckDB would need quoting
    everywhere.
    """
    return f"stat_{stat_id}"


def prices_defense(dst_rules: dict | None) -> bool:
    """True when `dst_rules` gives any defensive stat a non-zero point value.

    `None` is "not specified", which means DEFAULT_DST_RULES (ESPN's own
    default D/ST scoring) -- so it is True, unlike `prices_kicking(None)`.
    The asymmetry is not an oversight: full PPR genuinely scores no kicking,
    so "unspecified" there really does mean "kicking is worth nothing", while
    every ESPN league ever created scores defenses out of the box.

    `{}` is `league.from_espn` finding no slot-16 override anywhere -- a
    league that truly does not score defense (or does not roster one). That
    is False, and it is a different answer from `None`, exactly as it is in
    `compute_ppr_points`.
    """
    if dst_rules is None:
        return True
    return any(bool(pts) for pts in dst_rules.values())


def compute_dst_points(df: pd.DataFrame, dst_rules: dict | None = None) -> pd.Series:
    """Score defense rows from `dst_weekly` under a league's own D/ST rules.

    Same contract as `compute_ppr_points`, one level down: `None` means "not
    specified" and falls back to DEFAULT_DST_RULES; `{}` means "nothing
    scores" and returns zeros. A rule naming a stat this season's payload
    never carried contributes a column of zeros rather than raising -- ESPN
    omits an id from a stat line entirely when it is zero for that game, so a
    missing column is normal, not a schema problem.
    """
    dst_rules = DEFAULT_DST_RULES if dst_rules is None else dst_rules
    total = pd.Series(0.0, index=df.index)
    for stat_id, pts in dst_rules.items():
        total = total + _col(df, dst_stat_column(stat_id)) * pts
    return total
