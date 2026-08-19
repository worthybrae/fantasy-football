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


# ---------------------------------------------------------------------------
# Kicking: pinned against REAL weekly rows, not a synthetic line.
# ---------------------------------------------------------------------------

# Brandon Aubrey's and Joshua Karty's actual 2025 weeks, copied verbatim out
# of data/nfl.duckdb's `weekly` table. Regenerate with:
#
#   SELECT week, fg_made_0_19, fg_made_20_29, fg_made_30_39, fg_made_40_49,
#          fg_made_50_59, fg_made_60_, fg_missed, fg_blocked,
#          pat_made, pat_missed, pat_blocked
#   FROM weekly WHERE season = 2025 AND position = 'K'
#     AND player_display_name = '<name>' ORDER BY week
#
# Embedded rather than read from the database because no test in this suite
# depends on data/nfl.duckdb existing -- but these ARE the real rows, and the
# point of the pair is that the arithmetic below is checkable against a
# published third-party number (see the assertions).
#
# Column order: week, <40 in three bands, 40-49, 50-59, 60+, missed, blocked,
# PAT made, PAT missed, PAT blocked.
_AUBREY_2025 = [
    (1, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0),
    (2, 0, 0, 0, 2, 1, 1, 0, 0, 4, 0, 0),
    (3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
    (4, 0, 1, 0, 0, 0, 0, 0, 0, 5, 0, 0),
    (5, 0, 0, 1, 0, 0, 0, 0, 0, 4, 1, 0),
    (6, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0),
    (7, 0, 1, 0, 1, 0, 1, 0, 0, 5, 0, 0),
    (8, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0),
    (9, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0),
    (11, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0),
    (12, 0, 0, 0, 1, 0, 0, 1, 0, 3, 0, 0),
    (13, 0, 1, 1, 1, 0, 0, 0, 0, 2, 0, 0),
    (14, 0, 1, 0, 1, 2, 1, 0, 0, 1, 0, 0),
    (15, 0, 1, 1, 2, 0, 0, 2, 0, 2, 0, 0),
    (16, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0),
    (17, 0, 0, 0, 1, 2, 0, 1, 0, 3, 0, 0),
    (18, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0),
]
# Karty is the second kicker on purpose: two blocked field goals AND two
# blocked extra points in eight games, which is what makes him able to fail
# the `85 -> [fg_missed, fg_blocked]` fan-out that Aubrey (no blocks at all)
# cannot.
_KARTY_2025 = [
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0),
    (2, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 1),
    (3, 0, 1, 1, 1, 1, 0, 0, 2, 2, 0, 0),
    (4, 0, 0, 1, 1, 0, 0, 0, 0, 3, 0, 0),
    (5, 0, 0, 0, 1, 0, 0, 1, 0, 2, 0, 1),
    (6, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0),
    (7, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0),
    (9, 0, 0, 0, 0, 0, 0, 1, 0, 4, 1, 0),
]
_KICK_COLUMNS = ["week", "fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
                 "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
                 "fg_missed", "fg_blocked", "pat_made", "pat_missed",
                 "pat_blocked"]


def _kicker_weeks(rows, player_id, name):
    df = pd.DataFrame(rows, columns=_KICK_COLUMNS)
    df.insert(0, "player_id", player_id)
    df.insert(1, "player_display_name", name)
    df["position"] = "K"
    df["season"] = 2025
    df["recent_team"] = "DAL"
    df["opponent_team"] = "PHI"
    return df


def _owner_league_rules():
    """The owner's REAL ESPN scoring, parsed through the real code path.

    Point values are the ones his league publishes: FG under 40 is 3, 40-49
    is 4, 50-59 is 5, 60+ is 6, a miss is -1 and a PAT is 1 (statIds 80, 77,
    198, 201, 85, 86 -- see scoring/league.ESPN_STAT_COLUMNS for how each id
    was established). Built through `from_espn` rather than written out by
    hand so this test fails if the MAP breaks, not only if the arithmetic
    does.
    """
    from pipeline.espn_league import parse_settings
    from scoring import league
    payload = {"settings": {
        "size": 8,
        "rosterSettings": {"lineupSlotCounts": {
            "0": 1, "2": 2, "4": 2, "6": 1, "16": 1, "17": 1, "20": 5, "23": 2}},
        "scoringSettings": {"scoringItems": [
            {"statId": 3, "points": 0.04}, {"statId": 4, "points": 4.0},
            {"statId": 19, "points": 2.0}, {"statId": 20, "points": -2.0},
            {"statId": 24, "points": 0.1}, {"statId": 25, "points": 6.0},
            {"statId": 26, "points": 2.0}, {"statId": 42, "points": 0.1},
            {"statId": 43, "points": 6.0}, {"statId": 44, "points": 2.0},
            {"statId": 53, "points": 1.0}, {"statId": 72, "points": -2.0},
            {"statId": 77, "points": 4.0}, {"statId": 80, "points": 3.0},
            {"statId": 85, "points": -1.0}, {"statId": 86, "points": 1.0},
            {"statId": 198, "points": 5.0}, {"statId": 201, "points": 6.0},
        ]},
        "draftSettings": {"type": "SNAKE", "pickOrder": [1, 2, 3, 4, 5, 6, 7, 8]},
    }}
    return league.from_espn(parse_settings(payload, 2025)).scoring


def test_espn_kicking_ids_reach_the_weekly_columns_they_score():
    rules = _owner_league_rules()
    # 80 fans out to the three sub-40 bands, 85 and 88 to missed+blocked.
    assert rules["fg_made_0_19"] == 3.0
    assert rules["fg_made_20_29"] == 3.0
    assert rules["fg_made_30_39"] == 3.0
    assert rules["fg_made_40_49"] == 4.0
    assert rules["fg_made_50_59"] == 5.0
    assert rules["fg_made_60_"] == 6.0
    assert rules["fg_missed"] == -1.0
    assert rules["fg_blocked"] == -1.0
    assert rules["pat_made"] == 1.0
    # ...and the skill-position half is untouched by any of it.
    from scoring.ppr import DEFAULT_RULES
    assert {k: v for k, v in rules.items() if k in DEFAULT_RULES} == DEFAULT_RULES


def test_a_real_kicker_season_scores_what_espn_says_it_scores():
    """Brandon Aubrey's real 2025, hand-checked against ESPN's own total.

    Season totals from the rows above:
        FG under 40   0 + 10 + 5 = 15   x  3 =  45
        FG 40-49                  10   x  4 =  40
        FG 50-59                   8   x  5 =  40
        FG 60+                     3   x  6 =  18
        FG missed                  6   x -1 =  -6
        PAT made                  47   x  1 =  47
                                            ------
                                             184.0

    ESPN publishes Aubrey's 2025 fantasy total for this league's scoring as
    184.6. The missing 0.6 is not a rounding error and not a gap in the
    kicking map: Aubrey ran once for 6 yards in week 2, and this league
    scores rushing yards at 0.1 (statId 24). 184.0 + 0.6 = 184.6, exactly.
    That last 0.6 is why this test asserts the kicking arithmetic on its
    own -- the fixture carries no rushing column -- and the whole-season
    reconciliation is recorded here rather than being silently absorbed.
    """
    df = _kicker_weeks(_AUBREY_2025, "aubrey", "Brandon Aubrey")
    rules = _owner_league_rules()
    # The bands really are what the arithmetic above claims.
    assert int(df[["fg_made_0_19", "fg_made_20_29", "fg_made_30_39"]].sum().sum()) == 15
    assert int(df["fg_made_40_49"].sum()) == 10
    assert int(df["fg_made_50_59"].sum()) == 8
    assert int(df["fg_made_60_"].sum()) == 3
    assert int(df["fg_missed"].sum()) == 6
    assert int(df["pat_made"].sum()) == 47

    assert compute_ppr_points(df, rules).sum() == 184.0
    # Weekly, not just in aggregate: week 2 was 2 from 40-49, 1 from 50-59,
    # 1 from 60+ and 4 extra points = 8 + 5 + 6 + 4 = 23.
    week2 = df[df["week"] == 2]
    assert compute_ppr_points(week2, rules).iloc[0] == 23.0


def test_a_blocked_kick_scores_as_a_miss_because_espn_counts_it_as_one():
    """Joshua Karty's real 2025 -- the case Aubrey cannot test.

        FG under 40    0 + 3 + 3 = 6   x  3 =  18
        FG 40-49                   3   x  4 =  12
        FG 50-59                   1   x  5 =   5
        FG missed                  3   x -1 =  -3
        FG blocked                 2   x -1 =  -2
        PAT made                  23   x  1 =  23
                                             -----
                                              53.0

    ESPN publishes 53.0 for this league. It only reconciles because the two
    blocked field goals are charged as misses: nflverse keeps `fg_missed`
    (3) and `fg_blocked` (2) in separate columns and ESPN reports a single
    "FG missed" of 5. Score `fg_missed` alone and this season comes out at
    55.0 -- two points of credit for kicks that were blocked. His two
    blocked PATs are NOT charged, because this league prices statId 88 at
    nothing.
    """
    df = _kicker_weeks(_KARTY_2025, "karty", "Joshua Karty")
    rules = _owner_league_rules()
    assert int(df["fg_missed"].sum()) == 3
    assert int(df["fg_blocked"].sum()) == 2
    assert int(df["pat_blocked"].sum()) == 2

    assert compute_ppr_points(df, rules).sum() == 53.0
    # The failure this guards against, stated as the number it would produce.
    without_blocked = {k: v for k, v in rules.items() if k != "fg_blocked"}
    assert compute_ppr_points(df, without_blocked).sum() == 55.0


def test_a_kicker_scores_nothing_under_a_league_that_prices_no_kicking():
    """The premise the old kicker-blanking rested on, kept true.

    Full PPR scores no kicking, so Aubrey's whole real season is 0.0 points
    -- which is exactly why scoring/profile.py still blanks a kicker's
    history for a league whose rules do not price kicking.
    """
    df = _kicker_weeks(_AUBREY_2025, "aubrey", "Brandon Aubrey")
    assert compute_ppr_points(df, None).sum() == 0.0


def test_prices_kicking_asks_whether_the_league_pays_for_kicking():
    from scoring.ppr import DEFAULT_RULES, prices_kicking
    assert prices_kicking(None) is False
    assert prices_kicking({}) is False
    assert prices_kicking(DEFAULT_RULES) is False
    assert prices_kicking(_owner_league_rules()) is True
    # Present but worth nothing is not priced: ESPN really does emit scoring
    # items at 0.0 points (the owner's league carries two points-allowed
    # tiers that way), and a zero-valued rule produces the same column of
    # zeros the blanking exists to hide.
    assert prices_kicking({**DEFAULT_RULES, "fg_made": 0.0}) is False
    assert prices_kicking({**DEFAULT_RULES, "fg_made": 3.0}) is True


def test_kicking_rules_do_not_disturb_a_full_ppr_skill_players_points():
    """The guarantee the owner's live draft rests on.

    His league parses to 14 rules today and to 20 once the kicking ids are
    understood. `normalize_rules` must keep the skill half summing in
    DEFAULT_RULES' own order, or every skill player's points move in their
    last bits -- which is the documented float-ordering bug above, and it
    would reach `projection_scale` and re-rank the board.
    """
    from scoring.ppr import DEFAULT_RULES, normalize_rules
    with_kicking = _owner_league_rules()
    assert len(with_kicking) > len(DEFAULT_RULES)
    assert normalize_rules(with_kicking) is not None      # it IS a real change
    # ...but the core sums in the canonical order, bit for bit.
    assert list(normalize_rules(with_kicking))[:len(DEFAULT_RULES)] == list(DEFAULT_RULES)

    receiver = pd.DataFrame([{"receptions": 6, "receiving_yards": 83,
                              "receiving_tds": 1, "carries": 2,
                              "rushing_yards": 11}] * 17)
    baseline = compute_ppr_points(receiver, None)
    with_k = compute_ppr_points(receiver, normalize_rules(with_kicking))
    # Bit-identical, not approximately equal.
    assert list(with_k) == list(baseline)
    assert with_k.sum() == baseline.sum()
