import pandas as pd
from pipeline.db import get_conn, write_table
from scoring import league
from scoring.config import LEAGUE_TEAMS, REPLACEMENT_RANK

# 8 teams, QB/2RB/2WR/TE/2FLEX/K/DST + 5 bench, full PPR.
ESPN_SETTINGS = {
    "settings": {
        "size": 8,
        "rosterSettings": {"lineupSlotCounts": {
            "0": 1, "2": 2, "4": 2, "6": 1, "16": 1, "17": 1,
            "20": 5, "21": 1, "23": 2,
        }},
        "scoringSettings": {"scoringItems": [
            {"statId": 3, "points": 0.04}, {"statId": 4, "points": 4.0},
            {"statId": 20, "points": -2.0}, {"statId": 24, "points": 0.1},
            {"statId": 25, "points": 6.0}, {"statId": 53, "points": 1.0},
            {"statId": 42, "points": 0.1}, {"statId": 43, "points": 6.0},
            {"statId": 72, "points": -2.0}, {"statId": 101, "points": 6.0},
        ]},
        "draftSettings": {"type": "SNAKE", "pickOrder": [3, 7, 1, 2, 4, 5, 6, 8]},
    }
}

def _settings():
    from pipeline.espn_league import parse_settings
    return league.from_espn(parse_settings(ESPN_SETTINGS, 2026))

def test_starters_and_rounds():
    s = _settings()
    assert s.teams == 8
    assert s.starters == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}
    assert s.flex_slots == 2
    assert s.bench == 5
    # 8 starting slots + 2 flex + 5 bench; IR is not drafted.
    assert s.rounds == 15

def test_replacement_ranks_match_todays_hardcoded_values():
    assert _settings().replacement_ranks == REPLACEMENT_RANK

def test_scoring_maps_espn_stat_ids_to_nflverse_columns():
    s = _settings()
    assert s.scoring["receptions"] == 1.0
    assert s.scoring["passing_yards"] == 0.04
    # ESPN has one "fumble lost" stat; nflverse splits it three ways.
    assert s.scoring["sack_fumbles_lost"] == -2.0
    assert s.scoring["rushing_fumbles_lost"] == -2.0
    assert s.scoring["receiving_fumbles_lost"] == -2.0

def test_unmapped_scoring_items_are_reported_not_dropped_silently():
    # 101 is a kickoff-return TD. The reason it stays unmapped is NOT that
    # nflverse lacks it, which is what this comment used to say: nflverse
    # carries it in `special_teams_tds`, but that one column also covers the
    # punt-return TD (statId 102), so pricing both would score every return
    # touchdown twice. See scoring/league.ESPN_STAT_COLUMNS.
    assert "101" in _settings().unmapped_scoring

def test_default_settings_reproduce_config_constants():
    s = league.default_settings()
    assert s.teams == LEAGUE_TEAMS
    assert s.replacement_ranks == REPLACEMENT_RANK
    assert s.rounds == 15

def test_load_returns_defaults_when_no_league_table(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    assert league.load(conn) == league.default_settings()

def test_load_reads_newest_season_from_league_table(tmp_path):
    from pipeline.espn_league import parse_settings
    conn = get_conn(str(tmp_path / "t.duckdb"))
    older = league.from_espn(parse_settings(ESPN_SETTINGS, 2025))
    rows = pd.DataFrame([
        {"season": 2025, "settings_json": league.to_json(older)},
        {"season": 2026, "settings_json": league.to_json(_settings())},
    ])
    write_table(conn, "league", rows)
    assert league.load(conn).season == 2026


def _with_receptions(pts):
    """A LeagueSettings identical to the default except for the reception
    point value, which is the only thing scoring_format reads."""
    from dataclasses import replace
    base = league.default_settings()
    return replace(base, scoring={**base.scoring, "receptions": pts})


def test_scoring_format_maps_reception_points_to_the_three_tokens():
    # Boundaries: 0.75 and 0.25 are the cutoffs, inclusive at the bottom of
    # each band. The real PPR league scores receptions 1.0.
    assert league.scoring_format(_with_receptions(1.0)) == "ppr"
    assert league.scoring_format(_with_receptions(0.75)) == "ppr"
    assert league.scoring_format(_with_receptions(0.74)) == "half"
    assert league.scoring_format(_with_receptions(0.5)) == "half"
    assert league.scoring_format(_with_receptions(0.25)) == "half"
    assert league.scoring_format(_with_receptions(0.24)) == "std"
    assert league.scoring_format(_with_receptions(0.0)) == "std"
    # The real ESPN-parsed PPR league (receptions 1.0) reads 'ppr'.
    assert league.scoring_format(_settings()) == "ppr"


def test_scoring_format_defaults_to_ppr_for_unknown_settings():
    # None (no ESPN import) and an empty scoring dict both mean "assume PPR",
    # which is the format the board has always read every source as.
    from dataclasses import replace
    assert league.scoring_format(None) == "ppr"
    assert league.scoring_format(replace(league.default_settings(), scoring={})) == "ppr"
    # default_settings is full PPR (DEFAULT_RULES scores receptions 1.0).
    assert league.scoring_format(league.default_settings()) == "ppr"


# 8 teams, PPR, plus the kicking half of the owner's real ESPN scoring and a
# representative slice of the team-defense items -- including the
# `pointsOverrides` shape ESPN really uses for them, which is what a naive
# defensive mapping would silently price at zero.
ESPN_SETTINGS_WITH_KICKING = {
    "settings": {
        "size": 8,
        "rosterSettings": {"lineupSlotCounts": {
            "0": 1, "2": 2, "4": 2, "6": 1, "16": 1, "17": 1,
            "20": 5, "21": 1, "23": 2,
        }},
        "scoringSettings": {"scoringItems": [
            {"statId": 3, "points": 0.04}, {"statId": 4, "points": 4.0},
            {"statId": 19, "points": 2.0}, {"statId": 20, "points": -2.0},
            {"statId": 24, "points": 0.1}, {"statId": 25, "points": 6.0},
            {"statId": 26, "points": 2.0}, {"statId": 42, "points": 0.1},
            {"statId": 43, "points": 6.0}, {"statId": 44, "points": 2.0},
            {"statId": 53, "points": 1.0}, {"statId": 72, "points": -2.0},
            # kicking
            {"statId": 77, "points": 4.0}, {"statId": 80, "points": 3.0},
            {"statId": 85, "points": -1.0}, {"statId": 86, "points": 1.0},
            {"statId": 198, "points": 5.0}, {"statId": 201, "points": 6.0},
            # team defense, exactly as ESPN publishes it
            {"statId": 99, "points": 0.0, "pointsOverrides": {"16": 1.0}},
            {"statId": 95, "points": 0.0, "pointsOverrides": {"16": 2.0}},
            {"statId": 89, "points": 0.0, "pointsOverrides": {"16": 5.0}},
            {"statId": 128, "points": 0.0, "pointsOverrides": {"16": 5.0}},
            {"statId": 101, "points": 6.0, "pointsOverrides": {"16": 6.0}},
            {"statId": 102, "points": 6.0, "pointsOverrides": {"16": 6.0}},
        ]},
        "draftSettings": {"type": "SNAKE", "pickOrder": [3, 7, 1, 2, 4, 5, 6, 8]},
    }
}


def _kicking_settings():
    from pipeline.espn_league import parse_settings
    return league.from_espn(parse_settings(ESPN_SETTINGS_WITH_KICKING, 2026))


def test_espn_kicking_stat_ids_reach_the_weekly_columns_they_score():
    """Each id's meaning was established by joining ESPN's own 2025 actuals
    to this database's weekly table for 40 kickers (see
    scoring/league.ESPN_STAT_COLUMNS); this pins the resulting map."""
    s = _kicking_settings().scoring
    assert s["fg_made_40_49"] == 4.0                       # 77
    assert s["fg_made_50_59"] == 5.0                       # 198
    assert s["fg_made_60_"] == 6.0                         # 201
    assert s["pat_made"] == 1.0                            # 86
    # 80 is a single ESPN item covering everything under 40 yards; nflverse
    # splits it three ways, so it fans out and each band gets the same value.
    assert (s["fg_made_0_19"], s["fg_made_20_29"], s["fg_made_30_39"]) == (3.0, 3.0, 3.0)
    # ESPN counts a blocked field goal as a missed one; nflverse does not.
    assert s["fg_missed"] == -1.0 and s["fg_blocked"] == -1.0


def test_reading_kicking_leaves_the_skill_positions_scoring_untouched():
    """The guarantee the live board rests on: this league is still exactly
    full PPR everywhere DEFAULT_RULES has an opinion."""
    from scoring.ppr import DEFAULT_RULES
    s = _kicking_settings().scoring
    assert {k: v for k, v in s.items() if k in DEFAULT_RULES} == DEFAULT_RULES


def test_team_defense_items_are_read_from_the_slot_16_override_not_points():
    """The failure mode this pins: every defensive item ESPN publishes carries
    `points: 0.0`, with the real value in `pointsOverrides["16"]`. A parser
    that reads `points` -- which this one did -- prices a whole position at
    nothing and says so nowhere.

    `dst_scoring` is keyed by ESPN stat id, not by an nflverse column, because
    the stat values come from ESPN too (pipeline/sources.fetch_espn_dst). No
    column has to be identified, so no column can be identified wrong; see
    scoring/ppr.DEFAULT_DST_RULES for the 3744-defense-week reconciliation
    against ESPN's own scored totals.
    """
    s = _kicking_settings()
    assert s.dst_scoring == {"99": 1.0, "95": 2.0, "89": 5.0, "128": 5.0,
                             "101": 6.0, "102": 6.0}
    # Not one of them leaked into the skill-position rules, and nothing was
    # priced at zero by reading `points` off an override item.
    assert "special_teams_tds" not in s.scoring
    assert "def_sacks" not in s.scoring and "def_interceptions" not in s.scoring
    assert all(v != 0.0 for v in s.scoring.values())


def test_an_id_that_also_scores_for_a_skill_player_stays_unmapped():
    """An id leaves `unmapped_scoring` only when it is FULLY accounted for.

    99/95/89/128 score for the defense slot and nothing else (`points` 0.0),
    so once `dst_scoring` prices them there is nothing left to report.

    101 and 102 are the kickoff- and punt-return touchdown and are worth 6
    points to ANY player -- a wide receiver who returns a kick scores them.
    That half is still unmappable, because nflverse carries one
    `special_teams_tds` column covering both, so pricing the pair would score
    every return touchdown twice. Being scored for defenses does not make
    them scored for everybody, and reporting otherwise would be the quiet
    kind of wrong.
    """
    s = _kicking_settings()
    for stat_id in ("99", "95", "89", "128"):
        assert stat_id not in s.unmapped_scoring
        assert stat_id in s.dst_scoring
    for stat_id in ("101", "102"):
        assert stat_id in s.unmapped_scoring
        assert stat_id in s.dst_scoring


def test_dst_scoring_absent_from_a_stored_row_is_none_not_empty():
    """`None` means "this row predates D/ST parsing" and resolves to ESPN's
    default D/ST scoring; `{}` means "this league scores no defense". Every
    `league` row on this deployment's database is the first case, so
    defaulting the missing key to `{}` would silently zero every defense.
    """
    import json
    s = _kicking_settings()
    blob = json.loads(league.to_json(s))
    assert blob["dst_scoring"] == s.dst_scoring
    blob.pop("dst_scoring")
    assert league.from_json(json.dumps(blob)).dst_scoring is None
    assert league.default_settings().dst_scoring is None
    # A league with no defensive item anywhere really does score no defense.
    assert _settings().dst_scoring == {}
