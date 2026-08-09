import pandas as pd
from pipeline.espn_league import (
    parse_draft_picks, parse_draft_teams, parse_player_directory,
)

DRAFT_PAYLOAD = {
    "draftDetail": {
        "drafted": True,
        "picks": [
            {"overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1,
             "teamId": 3, "playerId": 4046537, "keeper": False},
            {"overallPickNumber": 2, "roundId": 1, "roundPickNumber": 2,
             "teamId": 7, "playerId": 3117251, "keeper": False},
            {"overallPickNumber": 9, "roundId": 2, "roundPickNumber": 1,
             "teamId": 7, "playerId": 2977187, "keeper": True},
        ],
    }
}

TEAM_PAYLOAD = {
    "teams": [
        {"id": 3, "name": "Team Alpha", "owners": ["{AAA}"], "draftDayPickOrder": 1},
        {"id": 7, "name": "Team Bravo", "owners": ["{BBB}"], "draftDayPickOrder": 2},
    ],
    "members": [
        {"id": "{AAA}", "displayName": "worthy"},
        {"id": "{BBB}", "displayName": "dan"},
    ],
}

def test_parse_draft_picks():
    df = parse_draft_picks(DRAFT_PAYLOAD, 2025)
    assert list(df.columns) == ["season", "overall_pick", "round", "round_pick",
                                "team_id", "espn_player_id", "keeper"]
    assert len(df) == 3
    assert df.iloc[0]["season"] == 2025
    assert df.iloc[0]["espn_player_id"] == 4046537
    assert df.iloc[2]["keeper"]

def test_parse_draft_teams_uses_member_display_name():
    df = parse_draft_teams(TEAM_PAYLOAD, 2025)
    assert df.set_index("team_id")["manager"].to_dict() == {3: "worthy", 7: "dan"}
    assert df.set_index("team_id")["slot"].to_dict() == {3: 1, 7: 2}

def test_parse_draft_teams_falls_back_to_team_name():
    payload = {"teams": [{"id": 3, "name": "Orphan", "owners": [], "draftDayPickOrder": 1}],
               "members": []}
    assert parse_draft_teams(payload, 2025).iloc[0]["manager"] == "Orphan"

def test_parse_player_directory_maps_positions_and_skips_unknown():
    payload = [
        {"id": 4046537, "fullName": "Justin Jefferson",
         "defaultPositionId": 3, "proTeamId": 16},
        {"id": 999, "fullName": "Some Punter", "defaultPositionId": 7, "proTeamId": 16},
        {"id": 16018, "fullName": "Vikings D/ST",
         "defaultPositionId": 16, "proTeamId": 16},
    ]
    df = parse_player_directory(payload)
    assert list(df.columns) == ["espn_player_id", "player_name", "position", "nfl_team"]
    assert df.set_index("espn_player_id")["position"].to_dict() == {4046537: "WR", 16018: "DST"}
    assert df.set_index("espn_player_id")["nfl_team"].to_dict() == {4046537: "MIN", 16018: "MIN"}

def test_parse_settings_extracts_pick_order_and_draft_type():
    from pipeline.espn_league import parse_settings
    from tests.test_league import ESPN_SETTINGS
    out = parse_settings(ESPN_SETTINGS, 2026)
    assert out["season"] == 2026
    assert out["teams"] == 8
    assert out["draft_type"] == "SNAKE"
    assert out["pick_order"] == [3, 7, 1, 2, 4, 5, 6, 8]
    assert out["lineup_slots"]["23"] == 2
