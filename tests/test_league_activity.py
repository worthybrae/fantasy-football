"""A league's activity out of ESPN's payloads: who is in it, who played
whom, every transaction, every week's lineup. The fixtures are trimmed
copies of real answers (league 53929318, 2025), two teams each, so a
parser is tested against the shape ESPN actually sends."""
import json

import duckdb
import pandas as pd
import pytest

from pipeline import league_activity as act
from pipeline.db import read_table, write_table

SWID_A = "{AAAAAAAA-0000-0000-0000-000000000001}"
SWID_B = "{BBBBBBBB-0000-0000-0000-000000000002}"
SWID_C = "{CCCCCCCC-0000-0000-0000-000000000003}"   # co-owner, no team

TEAM_PAYLOAD = {
    "seasonId": 2025,
    "members": [
        {"id": SWID_A, "displayName": "alpha99", "firstName": "Al", "lastName": "Pha"},
        {"id": SWID_B, "displayName": "bravo", "firstName": "Bea", "lastName": "Vo"},
        {"id": SWID_C, "displayName": "charlie", "firstName": "Cy", "lastName": "Lee"},
    ],
    "teams": [
        {"id": 1, "name": "Team One", "primaryOwner": SWID_A, "owners": [SWID_A, SWID_C],
         "draftDayProjectedRank": 3, "playoffSeed": 2, "rankCalculatedFinal": 1,
         "record": {"overall": {"wins": 9, "losses": 5, "ties": 0,
                                "pointsFor": 1892.4, "pointsAgainst": 1700.1}},
         "transactionCounter": {"acquisitions": 30, "drops": 31, "trades": 2,
                                "moveToActive": 65, "moveToIR": 3,
                                "matchupAcquisitionTotals": {"2": 2}}},
        {"id": 2, "name": "Team Two", "primaryOwner": SWID_B, "owners": [SWID_B],
         "draftDayProjectedRank": 7, "playoffSeed": 8, "rankCalculatedFinal": 8,
         "record": {"overall": {"wins": 4, "losses": 10, "ties": 0,
                                "pointsFor": 1600.0, "pointsAgainst": 1850.2}},
         "transactionCounter": {"acquisitions": 15, "drops": 15, "trades": 0,
                                "moveToActive": 40, "moveToIR": 0}},
    ],
    "status": {"currentMatchupPeriod": 16, "finalScoringPeriod": 17,
               "latestScoringPeriod": 19, "previousSeasons": [2023, 2024]},
}

MATCHUP_PAYLOAD = {
    "seasonId": 2025,
    "schedule": [
        {"id": 1, "matchupPeriodId": 1, "playoffTierType": "NONE", "winner": "HOME",
         "home": {"teamId": 1, "totalPoints": 117.68, "pointsByScoringPeriod": {"1": 117.68}},
         "away": {"teamId": 2, "totalPoints": 99.1, "pointsByScoringPeriod": {"1": 99.1}}},
        {"id": 61, "matchupPeriodId": 16, "playoffTierType": "WINNERS_BRACKET", "winner": "AWAY",
         "home": {"teamId": 2, "totalPoints": 306.26,
                  "pointsByScoringPeriod": {"16": 138.84, "17": 167.42}},
         "away": {"teamId": 1, "totalPoints": 337.84,
                  "pointsByScoringPeriod": {"16": 175.0, "17": 162.84}}},
        # A bye: one side only.
        {"id": 62, "matchupPeriodId": 15, "playoffTierType": "WINNERS_BRACKET",
         "winner": "UNDECIDED",
         "home": {"teamId": 1, "totalPoints": 0.0, "pointsByScoringPeriod": {}}},
    ],
}

TXN_PAYLOAD = {
    "seasonId": 2025,
    "transactions": [
        {"id": "t-waiver-won", "type": "WAIVER", "status": "EXECUTED",
         "executionType": "PROCESS", "bidAmount": 0, "teamId": 1,
         "memberId": "NightlyLeagueUpdateTaskProcessor",
         "proposedDate": 1761725060311, "scoringPeriodId": 9,
         "relatedTransactionId": "t-claim",
         "items": [{"type": "ADD", "playerId": 3121422, "fromTeamId": 0, "toTeamId": 1,
                    "fromLineupSlotId": -1, "toLineupSlotId": 20},
                   {"type": "DROP", "playerId": 3916148, "fromTeamId": 1, "toTeamId": 0,
                    "fromLineupSlotId": 20, "toLineupSlotId": -1}]},
        {"id": "t-lineup", "type": "FUTURE_ROSTER", "status": "EXECUTED",
         "executionType": "EXECUTE", "bidAmount": 0, "teamId": 2, "memberId": SWID_B,
         "proposedDate": 1761017954293, "scoringPeriodId": 9,
         "items": [{"type": "LINEUP", "playerId": 4429025, "fromTeamId": 0, "toTeamId": 0,
                    "fromLineupSlotId": 20, "toLineupSlotId": 4}]},
        {"id": "t-decline", "type": "TRADE_DECLINE", "status": "EXECUTED",
         "executionType": "EXECUTE", "bidAmount": 0, "teamId": 1, "memberId": SWID_A,
         "proposedDate": 1761000000000, "scoringPeriodId": 9},   # no items
    ],
}

def _stat(period, source, total):
    return {"scoringPeriodId": period, "statSourceId": source, "statSplitTypeId": 1,
            "appliedTotal": total}

ROSTER_PAYLOAD = {
    "seasonId": 2025, "scoringPeriodId": 3,
    "teams": [
        {"id": 1, "roster": {"entries": [
            {"playerId": 4241389, "lineupSlotId": 4, "acquisitionType": "DRAFT",
             "injuryStatus": "NORMAL",
             "playerPoolEntry": {"player": {
                 "fullName": "CeeDee Lamb", "defaultPositionId": 3, "proTeamId": 6,
                 "injuryStatus": "ACTIVE",
                 "stats": [_stat(0, 0, 200.9), _stat(3, 1, 20.0), _stat(3, 0, 18.1)]}}},
            {"playerId": -16024, "lineupSlotId": 16, "acquisitionType": "ADD",
             "injuryStatus": "NORMAL",
             "playerPoolEntry": {"player": {
                 "fullName": "Chargers D/ST", "defaultPositionId": 16, "proTeamId": 24,
                 "injuryStatus": "ACTIVE", "stats": [_stat(3, 1, 6.0)]}}},   # no actual row
        ]}},
        {"id": 2, "roster": {"entries": [
            {"playerId": 3128429, "lineupSlotId": 20, "acquisitionType": "DRAFT",
             "injuryStatus": "OUT",
             "playerPoolEntry": {"player": {
                 "fullName": "Some Back", "defaultPositionId": 2, "proTeamId": 1,
                 "injuryStatus": "OUT", "stats": [_stat(3, 0, 0.0), _stat(3, 1, 11.5)]}}},
        ]}},
    ],
}

DRAFT_PAYLOAD = {
    "seasonId": 2025,
    "draftDetail": {"drafted": True, "inProgress": False, "picks": [
        {"overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1, "teamId": 1,
         "memberId": SWID_A, "playerId": 4362628, "autoDraftTypeId": 0, "keeper": False},
        {"overallPickNumber": 2, "roundId": 1, "roundPickNumber": 2, "teamId": 2,
         "memberId": SWID_B, "playerId": 4241389, "autoDraftTypeId": 1, "keeper": False},
        {"overallPickNumber": 3, "roundId": 2, "roundPickNumber": 1, "teamId": 2,
         "memberId": SWID_B, "playerId": -1, "autoDraftTypeId": 0, "keeper": False},
    ]},
}


def test_members_are_one_row_per_member_with_their_team_and_counters():
    df = act.parse_members(TEAM_PAYLOAD, 2025)
    assert list(df.columns) == act.MEMBER_COLUMNS
    a = df[df.member_id == SWID_A].iloc[0]
    assert (a.display_name, a.first_name, a.last_name) == ("alpha99", "Al", "Pha")
    assert a.team_id == 1 and a.draft_day_rank == 3
    assert (a.acquisitions, a.drops, a.trades, a.lineup_moves, a.ir_moves) == (30, 31, 2, 65, 3)
    # A co-owner owns no team as its primary, so no team and no counters.
    c = df[df.member_id == SWID_C].iloc[0]
    assert pd.isna(c.team_id) and pd.isna(c.acquisitions)
    assert (df.season == 2025).all()


def test_primary_owners_maps_team_to_member():
    assert act.primary_owners(TEAM_PAYLOAD) == {1: SWID_A, 2: SWID_B}


def test_matchups_keep_tier_winner_points_and_a_bye():
    df = act.parse_matchups(MATCHUP_PAYLOAD, 2025)
    assert list(df.columns) == act.MATCHUP_COLUMNS
    wk1 = df[df.matchup_period == 1].iloc[0]
    assert (wk1.tier, wk1.winner) == ("NONE", "HOME")
    assert (wk1.home_team_id, wk1.away_team_id) == (1, 2)
    assert (wk1.home_points, wk1.away_points) == (117.68, 99.1)
    assert json.loads(wk1.home_points_by_week_json) == {"1": 117.68}
    final = df[df.matchup_period == 16].iloc[0]
    assert final.tier == "WINNERS_BRACKET" and final.winner == "AWAY"
    assert json.loads(final.away_points_by_week_json) == {"16": 175.0, "17": 162.84}
    bye = df[df.matchup_period == 15].iloc[0]
    assert pd.isna(bye.away_team_id) and pd.isna(bye.away_points)


def test_transactions_resolve_the_member_and_keep_items_verbatim():
    owners = act.primary_owners(TEAM_PAYLOAD)
    df = act.parse_transactions(TXN_PAYLOAD, 2025, 9, owners)
    assert list(df.columns) == act.TRANSACTION_COLUMNS
    assert len(df) == 3
    won = df[df.txn_id == "t-waiver-won"].iloc[0]
    # ESPN's processor is not a member; the team's primary owner is.
    assert won.member_id == SWID_A and won.team_id == 1
    assert (won.type, won.status, won.execution_type) == ("WAIVER", "EXECUTED", "PROCESS")
    assert won.related_txn_id == "t-claim"
    assert [i["type"] for i in json.loads(won.items_json)] == ["ADD", "DROP"]
    lineup = df[df.txn_id == "t-lineup"].iloc[0]
    assert lineup.member_id == SWID_B      # a real SWID is kept as is
    assert lineup.proposed_at == pd.Timestamp(1761017954293, unit="ms", tz="UTC")
    decline = df[df.txn_id == "t-decline"].iloc[0]
    assert json.loads(decline.items_json) == []
    assert (df.week == 9).all() and (df.season == 2025).all()


def test_lineups_take_the_weeks_actual_and_projected_rows():
    df = act.parse_lineups(ROSTER_PAYLOAD, 2025, 3)
    assert list(df.columns) == act.LINEUP_COLUMNS
    lamb = df[df.player_id == 4241389].iloc[0]
    assert (lamb.team_id, lamb.player_name, lamb.position) == (1, "CeeDee Lamb", "WR")
    assert lamb.lineup_slot == 4 and lamb.acquisition_type == "DRAFT"
    # The week's rows, not the season total (period 0).
    assert (lamb.actual_points, lamb.projected_points) == (18.1, 20.0)
    dst = df[df.player_id == -16024].iloc[0]
    assert dst.position == "DST" and pd.isna(dst.actual_points) and dst.projected_points == 6.0
    out = df[df.player_id == 3128429].iloc[0]
    assert out.injury_status == "OUT" and out.lineup_slot == 20
    assert (df.week == 3).all()


def test_draft_flags_mark_autodrafts_and_skip_padding():
    owners = act.primary_owners(TEAM_PAYLOAD)
    df = act.parse_draft_flags(DRAFT_PAYLOAD, 2025, owners)
    assert list(df.columns) == act.DRAFT_FLAG_COLUMNS
    assert list(df.overall_pick) == [1, 2]          # playerId -1 is padding
    assert list(df.autodraft) == [False, True]
    assert list(df.member_id) == [SWID_A, SWID_B]
    assert list(df["round"]) == [1, 1]


def test_the_new_tables_are_registered_as_league_tables():
    from pipeline.db import LEAGUE_TABLES
    for name in ("league_members", "league_matchups", "league_transactions",
                 "league_lineups", "league_draft_flags", "league_raw"):
        assert name in LEAGUE_TABLES


def _fetch_json(served, calls):
    """`fetch_json(url) -> dict`, the shape league_history.json_fetch makes.
    `served` maps a (season, view, week) key to a payload; anything else 404s."""
    def fetch(url):
        calls.append(url)
        for (season, view, week), payload in served.items():
            if (f"/seasons/{season}/" in url and f"view={view}" in url
                    and ((week is None and "scoringPeriodId" not in url)
                         or (week is not None and f"scoringPeriodId={week}" in url))):
                return payload
        raise FileNotFoundError(url)
    return fetch


def _served(season, weeks=(1, 2), final=2):
    status = dict(TEAM_PAYLOAD["status"], finalScoringPeriod=final,
                  latestScoringPeriod=final, currentMatchupPeriod=final + 1)
    served = {
        (season, "mTeam", None): dict(TEAM_PAYLOAD, seasonId=season, status=status),
        (season, "mMatchupScore", None): dict(MATCHUP_PAYLOAD, seasonId=season),
        (season, "mDraftDetail", None): dict(DRAFT_PAYLOAD, seasonId=season),
    }
    for w in weeks:
        served[(season, "mTransactions2", w)] = dict(TXN_PAYLOAD, seasonId=season)
        served[(season, "mRoster", w)] = dict(ROSTER_PAYLOAD, seasonId=season, scoringPeriodId=w)
    return served


def test_season_url_names_the_view_and_the_week():
    assert act.season_url("53929318", 2025, "mTeam").endswith(
        "/seasons/2025/segments/0/leagues/53929318?view=mTeam")
    assert act.season_url("53929318", 2025, "mRoster", week=3).endswith(
        "?view=mRoster&scoringPeriodId=3")


def test_the_walk_stores_raw_answers_and_builds_every_table(tmp_path):
    conn = duckdb.connect(str(tmp_path / "league.duckdb"))
    calls = []
    summary = act.import_activity(conn, "53929318", _fetch_json(_served(2024), calls),
                                  seasons=[2024], current_season=2025)
    raw = read_table(conn, "league_raw")
    assert set(zip(raw.season, raw.view, raw.week)) == {
        (2024, "mTeam", 0), (2024, "mMatchupScore", 0), (2024, "mDraftDetail", 0),
        (2024, "mTransactions2", 1), (2024, "mTransactions2", 2),
        (2024, "mRoster", 1), (2024, "mRoster", 2)}
    assert len(read_table(conn, "league_members")) == 3
    assert len(read_table(conn, "league_matchups")) == 3
    # Two weeks of the same three transactions: de-duplicated on the id.
    assert len(read_table(conn, "league_transactions")) == 3
    assert len(read_table(conn, "league_lineups")) == 6
    assert len(read_table(conn, "league_draft_flags")) == 2
    assert summary == {"seasons": [2024], "requests": 7, "weeks": {2024: 2}}


def test_a_finished_season_is_never_fetched_twice(tmp_path):
    conn = duckdb.connect(str(tmp_path / "league.duckdb"))
    served = _served(2024)
    act.import_activity(conn, "53929318", _fetch_json(served, []), seasons=[2024],
                        current_season=2025)
    calls = []
    summary = act.import_activity(conn, "53929318", _fetch_json(served, calls), seasons=[2024],
                                  current_season=2025)
    assert calls == [] and summary["requests"] == 0


def test_the_current_season_refetches_only_what_moved(tmp_path):
    conn = duckdb.connect(str(tmp_path / "league.duckdb"))
    # First pass: the season is two weeks in.
    served = _served(2025, weeks=(1, 2), final=17)
    served[(2025, "mTeam", None)]["status"].update(latestScoringPeriod=2, currentMatchupPeriod=2)
    act.import_activity(conn, "53929318", _fetch_json(served, []), seasons=[2025],
                        current_season=2025)
    # Second pass a day later: week 3 exists now. Week 1 is over and stays;
    # week 2 (the latest last time) and week 3 are read; the season views
    # are read again because the standings moved.
    served = _served(2025, weeks=(1, 2, 3), final=17)
    served[(2025, "mTeam", None)]["status"].update(latestScoringPeriod=3, currentMatchupPeriod=3)
    calls = []
    act.import_activity(conn, "53929318", _fetch_json(served, calls), seasons=[2025],
                        current_season=2025, max_age_hours=0)
    weeks = sorted(int(u.split("scoringPeriodId=")[1]) for u in calls if "scoringPeriodId=" in u)
    assert weeks == [2, 2, 3, 3]
    assert len(read_table(conn, "league_lineups")) == 9      # three weeks, three rows each


def test_progress_publishes_the_shape_up_front_and_ticks(tmp_path):
    conn = duckdb.connect(str(tmp_path / "league.duckdb"))
    progress = act.Progress("53929318")
    snap = progress.snapshot()
    assert snap["phase"] == "idle"
    act.import_activity(conn, "53929318", _fetch_json(_served(2024), []), seasons=[2024],
                        current_season=2025, progress=progress)
    snap = progress.snapshot()
    assert snap["phase"] == "done" and snap["seasons_done"] == [2024]
    assert snap["stages"] == [{"season": 2024, "label": "2024 · done", "done": True,
                               "week": 2, "weeks": 2}]


def test_progress_records_a_failure_and_the_walk_raises(tmp_path):
    conn = duckdb.connect(str(tmp_path / "league.duckdb"))
    progress = act.Progress("53929318")

    def boom(url):
        raise RuntimeError("ESPN answered 500")
    with pytest.raises(RuntimeError):
        act.import_activity(conn, "53929318", boom, seasons=[2024], current_season=2025,
                            progress=progress)
    snap = progress.snapshot()
    assert snap["phase"] == "failed" and "500" in snap["error"]
