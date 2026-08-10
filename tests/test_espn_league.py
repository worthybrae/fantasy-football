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

import pytest
from pipeline.db import get_conn, read_table, write_table
from pipeline.espn_league import (
    parse_league_id, season_url, history_url, fetch_season, import_seasons,
    validate_import,
)
from tests.test_league import ESPN_SETTINGS

def test_parse_league_id_from_url_and_bare_id():
    assert parse_league_id("https://fantasy.espn.com/football/league?leagueId=123456") == "123456"
    assert parse_league_id("123456") == "123456"

def test_season_url_uses_the_dated_path_for_every_season():
    # Was leagueHistory for prior seasons. Real leagues that keep the same id
    # 404 on leagueHistory and answer on the dated path, so that is now the
    # primary for every year and leagueHistory is the fallback.
    assert "seasons/2024/segments/0/leagues/99" in season_url("99", 2024, current_season=2026)
    assert "seasons/2026/segments/0/leagues/99" in season_url("99", 2026, current_season=2026)

def test_history_url_is_the_documented_fallback_form():
    assert "leagueHistory/99" in history_url("99", 2024)
    assert "seasonId=2024" in history_url("99", 2024)

def test_fetch_season_falls_back_to_league_history_when_the_dated_path_404s():
    seen = []
    def fetch(url):
        seen.append(url)
        if "leagueHistory" not in url:
            raise FileNotFoundError(url)
        return {"draftDetail": {"drafted": True}}
    assert fetch_season(fetch, "99", 2024, 2026) == {"draftDetail": {"drafted": True}}
    assert len(seen) == 2 and "leagueHistory" in seen[1]

def test_fetch_season_raises_only_when_both_forms_are_missing():
    def fetch(url):
        raise FileNotFoundError(url)
    with pytest.raises(FileNotFoundError):
        fetch_season(fetch, "99", 2024, 2026)

def test_parse_draft_picks_drops_espns_placeholder_picks():
    # ESPN pads an undrafted board with playerId -1 in every slot, and even a
    # completed draft carries a few for rounds nobody filled.
    payload = {"draftDetail": {"drafted": True, "picks": [
        {"overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1,
         "teamId": 3, "playerId": 4046537, "keeper": False},
        {"overallPickNumber": 2, "roundId": 1, "roundPickNumber": 2,
         "teamId": 7, "playerId": -1, "keeper": False},
        {"overallPickNumber": 3, "roundId": 1, "roundPickNumber": 3,
         "teamId": 1, "playerId": None, "keeper": False},
    ]}}
    df = parse_draft_picks(payload, 2025)
    assert df["espn_player_id"].tolist() == [4046537]

def test_import_seasons_skips_a_season_whose_draft_has_not_happened(tmp_path):
    # The upcoming season returns a full board with every playerId -1 and
    # drafted=False. Importing it would teach the model picks nobody made.
    upcoming = {"draftDetail": {"drafted": False, "picks": [
        {"overallPickNumber": i, "roundId": 1, "roundPickNumber": i,
         "teamId": 3, "playerId": -1, "keeper": False} for i in range(1, 9)]}}
    def fetch(url):
        if "2026" in url:
            return {**upcoming, **TEAM_PAYLOAD, **ESPN_SETTINGS}
        if "2025" not in url:
            raise FileNotFoundError(url)
        if "/players?" in url:
            return [{"id": 4046537, "fullName": "Justin Jefferson",
                     "defaultPositionId": 3, "proTeamId": 16}]
        return {**DRAFT_PAYLOAD, **TEAM_PAYLOAD, **ESPN_SETTINGS}
    conn = get_conn(str(tmp_path / "t.duckdb"))
    summary = import_seasons(conn, "99", current_season=2026, fetch=fetch)
    assert summary["seasons"] == [2025]
    assert set(read_table(conn, "draft_picks")["season"]) == {2025}

def _fake_fetch(seasons):
    """Serve league + player-directory payloads for `seasons`, 404 otherwise."""
    def fetch(url):
        year = next((s for s in seasons if str(s) in url), None)
        if year is None:
            raise FileNotFoundError(url)
        if "/players?" in url:
            return [{"id": 4046537, "fullName": "Justin Jefferson",
                     "defaultPositionId": 3, "proTeamId": 16},
                    {"id": 3117251, "fullName": "Saquon Barkley",
                     "defaultPositionId": 2, "proTeamId": 21}]
        return {**DRAFT_PAYLOAD, **TEAM_PAYLOAD, **ESPN_SETTINGS}
    return fetch

def test_import_seasons_walks_back_and_writes_tables(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    summary = import_seasons(conn, "99", current_season=2026,
                             fetch=_fake_fetch([2025, 2024]))
    assert summary["seasons"] == [2025, 2024]
    picks = read_table(conn, "draft_picks")
    assert len(picks) == 6                     # 3 picks x 2 seasons
    # One of the 3 picks per season (a keeper) has no matching entry in the
    # fake player directory below, so its player_name is NaN by design --
    # dropna before comparing rather than asserting every raw pick resolves
    # a name (parse_draft_picks intentionally keeps unmatched picks, per
    # Task 1's own test).
    assert set(picks["player_name"].dropna()) == {"Justin Jefferson", "Saquon Barkley"}
    # The keeper pick (playerId 2977187) has no directory entry, once per
    # imported season -- assert the null path explicitly rather than only
    # excluding it, since those null rows flow downstream into
    # build_observations and validate_import.
    assert picks["player_name"].isna().sum() == 2
    assert not read_table(conn, "league").empty
    assert not read_table(conn, "draft_teams").empty

def test_import_seasons_stops_after_two_consecutive_missing_seasons(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    # 2025 present, 2024 and 2023 missing -> walk halts, 2022 never requested.
    summary = import_seasons(conn, "99", current_season=2026,
                             fetch=_fake_fetch([2025, 2022]))
    assert summary["seasons"] == [2025]

def test_import_rejects_non_snake_draft(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    auction = {**ESPN_SETTINGS}
    auction["settings"] = {**ESPN_SETTINGS["settings"],
                           "draftSettings": {"type": "AUCTION", "pickOrder": []}}
    def fetch(url):
        if "2025" not in url:
            raise FileNotFoundError(url)
        if "/players?" in url:
            return []
        return {**DRAFT_PAYLOAD, **TEAM_PAYLOAD, **auction}
    with pytest.raises(ValueError, match="AUCTION"):
        import_seasons(conn, "99", current_season=2026, fetch=fetch)

def test_import_seasons_propagates_non_missing_errors(tmp_path):
    # A persistent auth failure (RuntimeError from EspnClient.get_json on a
    # non-401/403-recoverable, non-404 status) is not "this season doesn't
    # exist" -- it must propagate with its real cause, not be swallowed as a
    # miss and reported as the generic "no drafted seasons found".
    conn = get_conn(str(tmp_path / "t.duckdb"))
    def fetch(url):
        raise RuntimeError("401 for " + url)
    with pytest.raises(RuntimeError, match="401"):
        import_seasons(conn, "99", current_season=2026, fetch=fetch)

def test_validate_import_reports_pick_count_and_adp_match_rate(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    import_seasons(conn, "99", current_season=2026, fetch=_fake_fetch([2025]))
    lines = "\n".join(validate_import(conn))
    assert "2025" in lines
    assert "ADP match" in lines

def test_validate_import_matches_kickers_and_defenses(tmp_path):
    """The report is the spec's honesty mechanism, so it has to score the
    same join the fitting does.

    A K pick and a DST pick can never match by (normalized name, position):
    the ADP feed says "PK" and "Ravens" where ESPN says "K" and
    "Ravens D/ST". Before the fix this printed "ADP match 33%" plus the
    low-match warning for three perfectly good picks, blaming the data for a
    code bug.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "draft_picks", pd.DataFrame([
        {"season": 2025, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 11, "player_name": "Real Receiver",
         "position": "WR", "nfl_team": "DET", "keeper": False},
        {"season": 2025, "overall_pick": 2, "round": 1, "round_pick": 2,
         "team_id": 2, "espn_player_id": 12, "player_name": "Boot Leg",
         "position": "K", "nfl_team": "GB", "keeper": False},
        {"season": 2025, "overall_pick": 3, "round": 2, "round_pick": 1,
         "team_id": 2, "espn_player_id": 13, "player_name": "Ravens D/ST",
         "position": "DST", "nfl_team": "BAL", "keeper": False},
    ]))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2025, "team_id": 2, "manager": "dan", "slot": 2},
    ]))
    write_table(conn, "historic_adp", pd.DataFrame([
        {"season": 2025, "adp_name": "Real Receiver", "position": "WR",
         "team": "DET", "adp_rank": 1},
        {"season": 2025, "adp_name": "Boot Leg", "position": "K",
         "team": "GB", "adp_rank": 2},
        {"season": 2025, "adp_name": "Ravens", "position": "DST",
         "team": "BAL", "adp_rank": 3},
    ]))
    lines = "\n".join(validate_import(conn))
    assert "ADP match 100%" in lines
    assert "low ADP match" not in lines


def _small_payload(season, n_teams, n_picks):
    """League with a 1-round roster (QB only) so `expected = teams * rounds`
    can be made to exactly equal a controlled pick count per season."""
    picks = [{"overallPickNumber": i + 1, "roundId": 1, "roundPickNumber": i + 1,
              "teamId": (i % n_teams) + 1, "playerId": 9000 + i, "keeper": False}
             for i in range(n_picks)]
    settings = {"settings": {
        "size": n_teams,
        "rosterSettings": {"lineupSlotCounts": {"0": 1}},
        "scoringSettings": {"scoringItems": []},
        "draftSettings": {"type": "SNAKE", "pickOrder": list(range(1, n_teams + 1))},
    }}
    teams_payload = {
        "teams": [{"id": i + 1, "name": f"Team {i + 1}", "owners": [f"{{U{i}}}"],
                   "draftDayPickOrder": i + 1} for i in range(n_teams)],
        "members": [{"id": f"{{U{i}}}", "displayName": f"user{i}"} for i in range(n_teams)],
    }
    return {"draftDetail": {"drafted": True, "picks": picks}, **teams_payload, **settings}

def test_validate_import_uses_each_seasons_own_settings_for_expected_count(tmp_path):
    # 2025 is an 8->2-team league with 2 picks (expected = 2 teams * 1 round);
    # 2024 was a 3-team league with 3 picks (expected = 3 teams * 1 round).
    # Both seasons' actual pick counts exactly match THEIR OWN settings, so
    # neither should get a "<-- expected N" mismatch flag. The newest
    # season's settings (2025, 2 teams) must not leak into 2024's check.
    conn = get_conn(str(tmp_path / "t.duckdb"))
    payloads = {2025: _small_payload(2025, 2, 2), 2024: _small_payload(2024, 3, 3)}
    def fetch(url):
        year = next((s for s in payloads if str(s) in url), None)
        if year is None:
            raise FileNotFoundError(url)
        if "/players?" in url:
            return []
        return payloads[year]
    import_seasons(conn, "99", current_season=2026, fetch=fetch)
    lines = "\n".join(validate_import(conn))
    assert "expected" not in lines


def test_historic_espn_ranks_by_ppr_rank_and_records_adp_usability(tmp_path):
    """Ranks come from espn_ppr_rank, which survives seasons where ESPN's
    espn_adp column resets to a constant."""
    import pandas as pd
    from pipeline.db import get_conn, read_table, write_table
    from pipeline.import_league import build_historic_espn

    frames = {
        2025: pd.DataFrame([  # corrupt ADP, sane ranks
            {"espn_name": "Saquon Barkley", "position": "RB",
             "espn_adp": 170.0, "espn_ppr_rank": 4},
            {"espn_name": "Jahmyr Gibbs", "position": "RB",
             "espn_adp": 170.0, "espn_ppr_rank": 5},
        ]),
        2024: pd.DataFrame([
            {"espn_name": "CeeDee Lamb", "position": "WR",
             "espn_adp": 6.3, "espn_ppr_rank": 1},
            {"espn_name": "Alvin Kamara", "position": "RB",
             "espn_adp": 8.3, "espn_ppr_rank": 2},
        ]),
    }
    out = build_historic_espn(frames)
    assert list(out.columns) == ["season", "espn_name", "position",
                                 "espn_rank", "adp_usable"]
    y25 = out[out["season"] == 2025].sort_values("espn_rank")
    assert y25["espn_name"].tolist() == ["Saquon Barkley", "Jahmyr Gibbs"]
    assert y25["espn_rank"].tolist() == [1, 2]      # dense, not raw ppr_rank
    assert not y25["adp_usable"].any()
    assert out[out["season"] == 2024]["adp_usable"].all()
