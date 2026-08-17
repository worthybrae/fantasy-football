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
    # 101 is a kick-return TD, which nflverse weekly player stats do not carry.
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
