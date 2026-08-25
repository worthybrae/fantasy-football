# League History and Manager Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import every season of a league's ESPN history (members, matchups, transactions, weekly lineups) into the league's own database on first visit, and turn it into a profile of every manager on the league page.

**Architecture:** Pure parsers over ESPN payloads write per-league DuckDB tables; a week-by-week walk with a progress record fetches them; `scoring/manager_profile.py` computes facts from the tables with no network; a small FastAPI module starts the import and serves the facts; two React pages render them. Tasks 1-6 are new files and run on `main` today; tasks 7-10 wire into the `worktree-league-report` branch's `import_history`, `league_standings` and `team_profiles` after that branch merges.

**Tech Stack:** Python 3.10, pandas, DuckDB, FastAPI, httpx (through `pipeline/espn_drafts.http_fetch`), pytest; React 19, TypeScript, react-router 7, Vite, Playwright (via `.venv/bin/python`).

**Spec:** `docs/superpowers/specs/2026-08-25-league-history-design.md`

## Global Constraints

- The manager key is the ESPN member id (SWID, e.g. `{8491403C-A53F-...}`) everywhere; team ids and names are season-scoped.
- New tables are per-league and listed in `LEAGUE_TABLES` (`pipeline/db.py`); `tests/test_leagues.py` asserts that set exactly.
- One writer per DuckDB file: every writer takes its own connection, writes, closes; a locked write is retried once after a short wait.
- Fetch shape is `fetch(url, cookies, headers) -> (status, text)` (`pipeline/espn_drafts.http_fetch`); the season/week walk uses the branch's `json_fetch(raw, cookies)` adapter which raises `FileNotFoundError` on 404.
- Every profile fact carries its sample size as `n`; the page states no adjective off fewer than five observations.
- No network in `scoring/`; no LLM anywhere in this plan.
- Routes require a custody session whose league list contains the league (403 otherwise); mock league ids (name contains "mock") answer 400.
- Persisted prose (code comments, commit messages, docs) is normal prose.
- ESPN base: `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl`; season path `/seasons/{yr}/segments/0/leagues/{id}`.
- Lineup slot ids: starters are `ESPN_SLOT_POSITIONS` (0 QB, 2 RB, 4 WR, 6 TE, 16 DST, 17 K) plus `ESPN_FLEX_SLOT` 23; bench is `ESPN_BENCH_SLOT` 20; IR is `ESPN_IR_SLOT` 21. Player positions come from `ESPN_POSITION_BY_ID` (`pipeline/espn_league.py`).

---

## File map

| File | Responsibility |
|---|---|
| `pipeline/league_activity.py` (new) | Parsers: ESPN payload dict -> DataFrame for members, matchups, transactions, lineups, draft flags. The raw store (`league_raw`). The week walk `import_activity` and its `Progress` record. |
| `pipeline/db.py` (modify `LEAGUE_TABLES`) | Register the six new per-league tables. |
| `scoring/manager_profile.py` (new) | Facts: `profile(conn, member_id)`, `league_overview(conn)`, and the helpers each section is built from. |
| `api/league_access.py` (new) | `owned_league(request, store, league_id) -> Session`, shared by this API and the reports API. |
| `api/league_history.py` (new) | Routes: start import, progress, overview, manager profile. In-process job registry. |
| `api/main.py` (modify, task 7) | Register the routes before `register_spa`. |
| `pipeline/league_history.py` (modify, task 7, branch file) | Call `import_activity` after `import_seasons`. |
| `web/src/api.ts` (append) | Types and fetchers for the four endpoints. |
| `web/src/pages/LeaguePage.tsx` (modify) | Import state, seasons strip, manager grid. |
| `web/src/pages/ManagerPage.tsx` (new) | The profile page. |
| `web/src/pages/league.css` (modify) | Styles for both pages (`lg-` prefix). |
| `web/src/App.tsx` (modify) | The manager route. |
| `tests/test_league_activity.py` (new) | Parsers, raw store, walk, progress. |
| `tests/test_manager_profile.py` (new) | Every fact on a hand-built league. |
| `tests/test_league_history_api.py` (new) | Routes, ownership, job lifecycle. |
| `tests/test_leagues.py` (modify) | `LEAGUE_TABLES` assertion. |

---

### Task 1: Parsers and the tables they fill

**Files:**
- Create: `pipeline/league_activity.py`
- Modify: `pipeline/db.py:17-21` (`LEAGUE_TABLES`)
- Modify: `tests/test_leagues.py:19-22`
- Test: `tests/test_league_activity.py`

**Interfaces:**
- Consumes: `pipeline.espn_league.ESPN_POSITION_BY_ID`, `ESPN_BENCH_SLOT`, `ESPN_IR_SLOT`.
- Produces: `parse_members(payload, season) -> DataFrame` (MEMBER_COLUMNS), `parse_matchups(payload, season) -> DataFrame` (MATCHUP_COLUMNS), `parse_transactions(payload, season, week, owners) -> DataFrame` (TRANSACTION_COLUMNS), `parse_lineups(payload, season, week) -> DataFrame` (LINEUP_COLUMNS), `parse_draft_flags(payload, season, owners) -> DataFrame` (DRAFT_FLAG_COLUMNS), `primary_owners(payload) -> dict[int, str]`. The column-name constants are the table schemas.

- [ ] **Step 1: Write the failing parser tests with trimmed fixtures**

Create `tests/test_league_activity.py`:

```python
"""A league's activity out of ESPN's payloads: who is in it, who played
whom, every transaction, every week's lineup. The fixtures are trimmed
copies of real answers (league 53929318, 2025), two teams each, so a
parser is tested against the shape ESPN actually sends."""
import json

import pandas as pd
import pytest

from pipeline import league_activity as act

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_league_activity.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.league_activity'`

- [ ] **Step 3: Write the parsers**

Create `pipeline/league_activity.py`:

```python
"""A league's activity, out of ESPN's own payloads.

The draft importer (`pipeline/espn_league.py`) keeps who picked whom. This
module keeps everything else a league does in a season: who is in it,
who played whom and with what result, every transaction the league
processed, and every week's lineup. Each parser is a pure function from
one ESPN answer to one DataFrame, so it can be tested against a copy of a
real answer and re-run over stored payloads when it changes.

ESPN's shapes, as probed on 2026-08-25 (league 53929318):

  * `mTeam`: `members[]` carry the SWID (`id`) and names; `teams[]` carry
    `primaryOwner`, the record, the playoff seed, the final rank, ESPN's
    draft-day projected rank, and `transactionCounter` with the season's
    activity totals.
  * `mMatchupScore`: `schedule[]`, one entry per matchup, `playoffTierType`
    naming the bracket, `winner` HOME/AWAY/TIE/UNDECIDED, and each side's
    points in total and by scoring period. A bye has one side only.
  * `mTransactions2&scoringPeriodId=N`: `transactions[]` for that week.
    `memberId` is the acting SWID, or ESPN's processor name for a waiver
    run; `items[]` carries the players and slots moved, and is absent on
    a TRADE_DECLINE.
  * `mRoster&scoringPeriodId=N`: every team's roster that week, each entry
    with its slot and the player's `stats[]`, of which the rows with this
    period and `statSourceId` 0 (actual) / 1 (projected) are the week's.
  * `mDraftDetail`: every pick with `memberId` and `autoDraftTypeId`
    (non-zero means the computer picked); padding rows have playerId -1.
"""
from __future__ import annotations

import json

import pandas as pd

from pipeline.espn_league import ESPN_POSITION_BY_ID

MEMBER_COLUMNS = ["season", "member_id", "display_name", "first_name", "last_name",
                  "team_id", "draft_day_rank", "acquisitions", "drops", "trades",
                  "lineup_moves", "ir_moves"]
MATCHUP_COLUMNS = ["season", "matchup_period", "tier", "home_team_id", "away_team_id",
                   "home_points", "away_points", "winner",
                   "home_points_by_week_json", "away_points_by_week_json"]
TRANSACTION_COLUMNS = ["season", "week", "txn_id", "related_txn_id", "team_id",
                       "member_id", "type", "status", "execution_type", "bid_amount",
                       "proposed_at", "items_json"]
LINEUP_COLUMNS = ["season", "week", "team_id", "player_id", "player_name", "position",
                  "lineup_slot", "actual_points", "projected_points",
                  "acquisition_type", "injury_status"]
DRAFT_FLAG_COLUMNS = ["season", "overall_pick", "round", "team_id", "member_id",
                      "player_id", "autodraft", "keeper"]


def _int(value):
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _float(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _is_swid(value) -> bool:
    return isinstance(value, str) and value.startswith("{") and value.endswith("}")


def primary_owners(payload: dict) -> dict:
    """team id -> the SWID of the team's primary owner."""
    out = {}
    for team in payload.get("teams") or []:
        owner = team.get("primaryOwner")
        if _is_swid(owner) and team.get("id") is not None:
            out[int(team["id"])] = owner
    return out


def parse_members(payload: dict, season: int) -> pd.DataFrame:
    """One row per member: names, the team they own this season, and the
    season's activity counters for that team. A member with no team as
    its primary owner (a co-owner) keeps the row and leaves the rest NULL."""
    by_owner = {}
    for team in payload.get("teams") or []:
        counter = team.get("transactionCounter") or {}
        by_owner[team.get("primaryOwner")] = {
            "team_id": _int(team.get("id")),
            "draft_day_rank": _int(team.get("draftDayProjectedRank")),
            "acquisitions": _int(counter.get("acquisitions")),
            "drops": _int(counter.get("drops")),
            "trades": _int(counter.get("trades")),
            "lineup_moves": _int(counter.get("moveToActive")),
            "ir_moves": _int(counter.get("moveToIR")),
        }
    rows = []
    for member in payload.get("members") or []:
        swid = member.get("id")
        if not _is_swid(swid):
            continue
        row = {"season": season, "member_id": swid,
               "display_name": member.get("displayName"),
               "first_name": member.get("firstName"),
               "last_name": member.get("lastName")}
        row.update(by_owner.get(swid) or {k: None for k in MEMBER_COLUMNS[5:]})
        rows.append(row)
    df = pd.DataFrame(rows, columns=MEMBER_COLUMNS)
    for col in MEMBER_COLUMNS[5:]:
        df[col] = df[col].astype("Int64")
    return df


def parse_matchups(payload: dict, season: int) -> pd.DataFrame:
    rows = []
    for m in payload.get("schedule") or []:
        home = m.get("home") or {}
        away = m.get("away") or {}
        rows.append({
            "season": season,
            "matchup_period": _int(m.get("matchupPeriodId")),
            "tier": m.get("playoffTierType") or "NONE",
            "home_team_id": _int(home.get("teamId")),
            "away_team_id": _int(away.get("teamId")),
            "home_points": _float(home.get("totalPoints")) if home else None,
            "away_points": _float(away.get("totalPoints")) if away else None,
            "winner": m.get("winner") or "UNDECIDED",
            "home_points_by_week_json": json.dumps(home.get("pointsByScoringPeriod") or {}),
            "away_points_by_week_json": json.dumps(away.get("pointsByScoringPeriod") or {}),
        })
    df = pd.DataFrame(rows, columns=MATCHUP_COLUMNS)
    for col in ("matchup_period", "home_team_id", "away_team_id"):
        df[col] = df[col].astype("Int64")
    for col in ("home_points", "away_points"):
        df[col] = df[col].astype("Float64")
    return df


def parse_transactions(payload: dict, season: int, week: int, owners: dict) -> pd.DataFrame:
    """One row per transaction. `member_id` is the acting SWID when ESPN
    names one, else the team's primary owner (`owners`, from
    `primary_owners`) -- a waiver processed overnight is still that
    team's claim."""
    rows = []
    for t in payload.get("transactions") or []:
        team_id = _int(t.get("teamId"))
        actor = t.get("memberId")
        member = actor if _is_swid(actor) else owners.get(team_id)
        proposed = _int(t.get("proposedDate"))
        rows.append({
            "season": season, "week": week,
            "txn_id": t.get("id"), "related_txn_id": t.get("relatedTransactionId"),
            "team_id": team_id, "member_id": member,
            "type": t.get("type"), "status": t.get("status"),
            "execution_type": t.get("executionType"),
            "bid_amount": _float(t.get("bidAmount")),
            "proposed_at": (pd.Timestamp(proposed, unit="ms", tz="UTC")
                            if proposed is not None else pd.NaT),
            "items_json": json.dumps(t.get("items") or []),
        })
    df = pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)
    df["team_id"] = df["team_id"].astype("Int64")
    return df


def _week_points(stats: list, week: int, source: int):
    for row in stats or []:
        if (_int(row.get("scoringPeriodId")) == week
                and _int(row.get("statSourceId")) == source):
            return _float(row.get("appliedTotal"))
    return None


def parse_lineups(payload: dict, season: int, week: int) -> pd.DataFrame:
    rows = []
    for team in payload.get("teams") or []:
        team_id = _int(team.get("id"))
        for entry in ((team.get("roster") or {}).get("entries")) or []:
            player = ((entry.get("playerPoolEntry") or {}).get("player")) or {}
            stats = player.get("stats") or []
            rows.append({
                "season": season, "week": week, "team_id": team_id,
                "player_id": _int(entry.get("playerId")),
                "player_name": player.get("fullName"),
                "position": ESPN_POSITION_BY_ID.get(_int(player.get("defaultPositionId"))),
                "lineup_slot": _int(entry.get("lineupSlotId")),
                "actual_points": _week_points(stats, week, 0),
                "projected_points": _week_points(stats, week, 1),
                "acquisition_type": entry.get("acquisitionType"),
                "injury_status": entry.get("injuryStatus") or player.get("injuryStatus"),
            })
    df = pd.DataFrame(rows, columns=LINEUP_COLUMNS)
    for col in ("team_id", "player_id", "lineup_slot"):
        df[col] = df[col].astype("Int64")
    for col in ("actual_points", "projected_points"):
        df[col] = df[col].astype("Float64")
    return df


def parse_draft_flags(payload: dict, season: int, owners: dict) -> pd.DataFrame:
    """Who made each pick, and whether a person did. Padding rows (playerId
    -1, the slots of a draft not yet made) are dropped, the same rule
    `espn_league._is_real_pick` applies with a looser id test: a flag row
    only says who was on the clock, so a team D/ST id is fine here."""
    rows = []
    detail = payload.get("draftDetail") or {}
    for p in detail.get("picks") or []:
        pid = _int(p.get("playerId"))
        if pid is None or pid == -1:
            continue
        team_id = _int(p.get("teamId"))
        actor = p.get("memberId")
        rows.append({
            "season": season,
            "overall_pick": _int(p.get("overallPickNumber")),
            "round": _int(p.get("roundId")),
            "team_id": team_id,
            "member_id": actor if _is_swid(actor) else owners.get(team_id),
            "player_id": pid,
            "autodraft": bool(_int(p.get("autoDraftTypeId")) or 0),
            "keeper": bool(p.get("keeper")),
        })
    return pd.DataFrame(rows, columns=DRAFT_FLAG_COLUMNS)
```

- [ ] **Step 4: Register the tables**

In `pipeline/db.py`, extend `LEAGUE_TABLES`:

```python
LEAGUE_TABLES = frozenset({
    "league", "draft_picks", "draft_teams", "draft_order",
    "manager_profiles", "manager_tendencies", "drafted",
    "sim_board", "sim_results", "sim_survival",
    # A league's own history beyond the draft (pipeline/league_activity.py):
    # who is in it, who played whom, every transaction, every week's
    # lineup, the raw ESPN answers those were parsed from. One league's
    # business and nobody else's.
    "league_members", "league_matchups", "league_transactions",
    "league_lineups", "league_draft_flags", "league_raw",
})
```

In `tests/test_leagues.py`, the exact-set assertion gains the same six names (keep whatever the league-report branch has added when this is rebased; the set is the union).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_league_activity.py tests/test_leagues.py -q -p no:cacheprovider`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline/league_activity.py pipeline/db.py tests/test_league_activity.py tests/test_leagues.py
git commit -m "A league's members, matchups, transactions and lineups parsed out of ESPN's payloads"
```

---

### Task 2: The raw store, the week walk, and progress

**Files:**
- Modify: `pipeline/league_activity.py` (append)
- Test: `tests/test_league_activity.py` (append)

**Interfaces:**
- Consumes: Task 1 parsers; `pipeline.db.write_table`, `read_table`.
- Produces: `class Progress` with `.start(seasons)`, `.tick(season, week, total)`, `.finish(season)`, `.fail(error)`, `.done()`, `.snapshot() -> dict`; `import_activity(conn, league_id, fetch_json, seasons, current_season, progress=None, pause=0.0) -> dict` where `fetch_json(url) -> dict` raises `FileNotFoundError` on 404; `rebuild_tables(conn)`; `season_url(league_id, season, view, week=None) -> str`; `fresh_weeks(conn, season, latest) -> set[int]`.

- [ ] **Step 1: Write the failing walk tests**

Append to `tests/test_league_activity.py`:

```python
import duckdb

from pipeline.db import read_table, write_table


def _fetch_json(served, calls):
    """`fetch_json(url) -> dict`, the shape league_history.json_fetch makes.
    `served` maps a (season, view, week) key to a payload; anything else 404s."""
    def fetch(url):
        calls.append(url)
        for (season, view, week), payload in served.items():
            if (f"/seasons/{season}/" in url and f"view={view}" in url
                    and (week is None and "scoringPeriodId" not in url
                         or week is not None and f"scoringPeriodId={week}" in url)):
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_league_activity.py -q -p no:cacheprovider -k "walk or season_url or fetched_twice or moved or progress"`
Expected: FAIL with `AttributeError: module 'pipeline.league_activity' has no attribute 'season_url'`

- [ ] **Step 3: Write the store, the walk and the progress record**

Append to `pipeline/league_activity.py`:

```python
import threading
import time

from pipeline.db import read_table, write_table

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
SEASON_VIEWS = ("mTeam", "mMatchupScore", "mDraftDetail")
WEEK_VIEWS = ("mTransactions2", "mRoster")
RAW_COLUMNS = ["season", "view", "week", "fetched_at", "payload_json"]

# How old the current season's stored answer may be before it is read
# again. A day: standings and transactions move weekly, and a page that
# re-read ESPN on every visit would spend forty requests to learn nothing.
CURRENT_MAX_AGE_HOURS = 24.0


def season_url(league_id: str, season: int, view: str, week: int | None = None) -> str:
    url = f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}?view={view}"
    return url if week is None else f"{url}&scoringPeriodId={week}"


class Progress:
    """What an import is doing, for a page to draw.

    Published up front as one stage per season, then ticked week by week
    inside the season being read, so the page can show the whole shape
    before the first answer lands. Thread-safe: the walk writes on its
    thread and the API reads on request threads."""

    def __init__(self, league_id: str):
        self.league_id = league_id
        self._lock = threading.Lock()
        self._phase = "idle"
        self._stages = []
        self._done = []
        self._error = None
        self._started = None

    def start(self, seasons: list) -> None:
        with self._lock:
            self._phase = "running"
            self._started = time.time()
            self._error = None
            self._done = []
            self._stages = [{"season": s, "label": f"{s} · waiting", "done": False,
                             "week": 0, "weeks": None} for s in seasons]

    def _stage(self, season):
        return next((s for s in self._stages if s["season"] == season), None)

    def tick(self, season: int, week: int, total: int) -> None:
        with self._lock:
            stage = self._stage(season)
            if stage is None:
                return
            stage.update(week=week, weeks=total, label=f"{season} · week {week} of {total}")

    def finish(self, season: int) -> None:
        with self._lock:
            stage = self._stage(season)
            if stage is not None:
                stage.update(done=True, label=f"{season} · done")
            self._done.append(season)

    def fail(self, error: str) -> None:
        with self._lock:
            self._phase = "failed"
            self._error = error

    def done(self) -> None:
        with self._lock:
            self._phase = "done"

    def snapshot(self) -> dict:
        with self._lock:
            return {"league_id": self.league_id, "phase": self._phase,
                    "started_at": self._started, "error": self._error,
                    "seasons_done": list(self._done),
                    "stages": [dict(s) for s in self._stages]}


def _raw(conn) -> pd.DataFrame:
    return read_table(conn, "league_raw")


def _store_raw(conn, season: int, view: str, week: int, payload: dict) -> None:
    existing = _raw(conn)
    row = pd.DataFrame([{"season": season, "view": view, "week": week,
                         "fetched_at": pd.Timestamp.now(tz="UTC"),
                         "payload_json": json.dumps(payload)}], columns=RAW_COLUMNS)
    if not existing.empty:
        keep = ~((existing.season == season) & (existing.view == view) & (existing.week == week))
        row = pd.concat([existing[keep], row], ignore_index=True)
    write_table(conn, "league_raw", row)


def _have(raw: pd.DataFrame, season: int, view: str, week: int) -> bool:
    if raw.empty:
        return False
    hit = raw[(raw.season == season) & (raw.view == view) & (raw.week == week)]
    return not hit.empty


def _stale(raw: pd.DataFrame, season: int, max_age_hours: float) -> bool:
    if raw.empty or raw[raw.season == season].empty:
        return True
    newest = pd.Timestamp(raw[raw.season == season].fetched_at.max())
    if newest.tzinfo is None:
        newest = newest.tz_localize("UTC")
    return (pd.Timestamp.now(tz="UTC") - newest).total_seconds() > max_age_hours * 3600


def import_activity(conn, league_id: str, fetch_json, seasons: list, current_season: int,
                    progress: Progress | None = None, pause: float = 0.0,
                    max_age_hours: float = CURRENT_MAX_AGE_HOURS) -> dict:
    """Walk the seasons and their weeks, keep every answer, rebuild the tables.

    `seasons` is what `import_seasons` found drafted, newest first. A
    finished season already on file is skipped whole. The current season
    is read again when its newest stored answer is older than
    `max_age_hours`; within it, a week whose games are over (before the
    latest scoring period) and already stored is not read again.

    Raises whatever `fetch_json` raises other than FileNotFoundError, after
    recording the failure on `progress`. A 404 on a week is treated as an
    empty week.
    """
    progress = progress or Progress(league_id)
    progress.start(list(seasons))
    requests = 0
    weeks_read = {}
    try:
        for season in seasons:
            raw = _raw(conn)
            finished = season < current_season
            if finished and all(_have(raw, season, v, 0) for v in SEASON_VIEWS):
                progress.finish(season)
                continue
            if not finished and not _stale(raw, season, max_age_hours):
                progress.finish(season)
                continue
            payloads = {}
            for view in SEASON_VIEWS:
                payloads[view] = fetch_json(season_url(league_id, season, view))
                requests += 1
                _store_raw(conn, season, view, 0, payloads[view])
                if pause:
                    time.sleep(pause)
            status = payloads["mTeam"].get("status") or {}
            final = int(status.get("finalScoringPeriod") or 17)
            latest = int(status.get("latestScoringPeriod") or final)
            last_week = final if finished else min(final, max(latest, 1))
            raw = _raw(conn)
            weeks_read[season] = 0
            for week in range(1, last_week + 1):
                progress.tick(season, week, last_week)
                # Over and stored: nothing can have changed.
                settled = finished or week < latest
                if settled and all(_have(raw, season, v, week) for v in WEEK_VIEWS):
                    continue
                for view in WEEK_VIEWS:
                    try:
                        payload = fetch_json(season_url(league_id, season, view, week=week))
                    except FileNotFoundError:
                        payload = {}
                    requests += 1
                    _store_raw(conn, season, view, week, payload)
                    if pause:
                        time.sleep(pause)
                weeks_read[season] += 1
            progress.finish(season)
        rebuild_tables(conn)
    except Exception as exc:      # noqa: BLE001 -- recorded, then re-raised
        progress.fail(f"{type(exc).__name__}: {exc}")
        raise
    progress.done()
    return {"seasons": list(seasons), "requests": requests, "weeks": weeks_read}


def rebuild_tables(conn) -> None:
    """Every activity table from `league_raw`, whole. Idempotent, so a walk
    that stopped halfway leaves tables that agree with what is stored."""
    raw = _raw(conn)
    members, matchups, txns, lineups, flags = [], [], [], [], []
    if not raw.empty:
        for season in sorted(raw.season.unique()):
            rows = raw[raw.season == season]

            def payload(view, week=0):
                hit = rows[(rows.view == view) & (rows.week == week)]
                return json.loads(hit.iloc[0].payload_json) if not hit.empty else {}

            team = payload("mTeam")
            owners = primary_owners(team)
            members.append(parse_members(team, int(season)))
            matchups.append(parse_matchups(payload("mMatchupScore"), int(season)))
            flags.append(parse_draft_flags(payload("mDraftDetail"), int(season), owners))
            for week in sorted(rows[rows.week > 0].week.unique()):
                txns.append(parse_transactions(payload("mTransactions2", int(week)),
                                               int(season), int(week), owners))
                lineups.append(parse_lineups(payload("mRoster", int(week)),
                                             int(season), int(week)))

    def concat(frames, columns):
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

    write_table(conn, "league_members", concat(members, MEMBER_COLUMNS))
    write_table(conn, "league_matchups", concat(matchups, MATCHUP_COLUMNS))
    # The same transaction is answered under every week it touched; keep
    # the first sighting.
    all_txns = concat(txns, TRANSACTION_COLUMNS)
    if not all_txns.empty:
        all_txns = all_txns.drop_duplicates("txn_id", keep="first").reset_index(drop=True)
    write_table(conn, "league_transactions", all_txns)
    write_table(conn, "league_lineups", concat(lineups, LINEUP_COLUMNS))
    write_table(conn, "league_draft_flags", concat(flags, DRAFT_FLAG_COLUMNS))
```

Note for the implementer: `write_table` on an empty frame with only column names creates a table with `object` columns; that is acceptable for reads through `read_table`, which every consumer in this plan uses.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_league_activity.py -q -p no:cacheprovider`
Expected: all PASS. If `test_the_current_season_refetches_only_what_moved` sees `[1, 1, 2, 2, 3, 3]`, the `settled` rule is wrong: week 1 must be skipped because `1 < latest (3)` and both views are stored.

- [ ] **Step 5: Commit**

```bash
git add pipeline/league_activity.py tests/test_league_activity.py
git commit -m "The week walk keeps every ESPN answer, rebuilds the tables from them, and reports progress"
```

---

### Task 3: Facts from finishes and matchups

**Files:**
- Create: `scoring/manager_profile.py`
- Test: `tests/test_manager_profile.py`

**Interfaces:**
- Consumes: tables `league_members`, `league_standings` (branch's columns: `season, team_id, manager, team_name, wins, losses, ties, points_for, points_against, playoff_seed, final_rank`), `league_matchups`.
- Produces: `members(conn) -> DataFrame`, `seasons_of(conn, member_id) -> list[dict]`, `finishes(conn, member_id) -> dict`, `playoffs(conn, member_id) -> dict`, `head_to_head(conn) -> dict[(a, b)] -> dict`, `luck(conn, member_id) -> dict`, `seasons_strip(conn) -> list[dict]`. Every dict carries `n`.

- [ ] **Step 1: Write the failing tests on a hand-built league**

Create `tests/test_manager_profile.py`:

```python
"""Every fact on a manager's profile, on a league small enough to check by
hand: four teams, two seasons, known answers."""
import json

import duckdb
import pandas as pd
import pytest

from pipeline.db import write_table
from scoring import manager_profile as mp

A, B, C, D = ("{A}", "{B}", "{C}", "{D}")


def _members(season, counters=None):
    rows = []
    for i, swid in enumerate((A, B, C, D), start=1):
        row = {"season": season, "member_id": swid, "display_name": swid.strip("{}").lower(),
               "first_name": None, "last_name": None, "team_id": i,
               "draft_day_rank": i, "acquisitions": 10 * i, "drops": 10 * i,
               "trades": 0, "lineup_moves": 20, "ir_moves": 0}
        row.update((counters or {}).get(swid, {}))
        rows.append(row)
    return rows


def _standings(season, finals, points):
    """finals: team_id -> final_rank; points: team_id -> points_for."""
    rows = []
    for tid in (1, 2, 3, 4):
        rows.append({"season": season, "team_id": tid, "manager": f"m{tid}",
                     "team_name": f"Team {tid}", "wins": 0, "losses": 0, "ties": 0,
                     "points_for": points[tid], "points_against": 0.0,
                     "playoff_seed": finals[tid] if finals[tid] <= 2 else None,
                     "final_rank": finals[tid]})
    return rows


def _matchup(season, period, home, away, hp, ap, tier="NONE"):
    winner = "HOME" if hp > ap else "AWAY" if ap > hp else "TIE"
    return {"season": season, "matchup_period": period, "tier": tier,
            "home_team_id": home, "away_team_id": away, "home_points": hp,
            "away_points": ap, "winner": winner,
            "home_points_by_week_json": json.dumps({str(period): hp}),
            "away_points_by_week_json": json.dumps({str(period): ap})}


@pytest.fixture
def league(tmp_path):
    """2024: A wins the title, D last. 2025: B wins, A last.

    Regular season is three weeks of round-robin; week 4 is the final
    between the top two. A's 2024 regular season: beat B, beat C, beat D
    (3-0) while scoring the most every week -- all-play 3-0-... so luck 0.
    D's 2024: lost every game but out-scored one team each week (all-play
    1 of 3 each week => expected 1 win over 3 games) -- luck -1.0."""
    conn = duckdb.connect(str(tmp_path / "league.duckdb"))
    write_table(conn, "league_members", pd.DataFrame(_members(2024) + _members(2025)))
    write_table(conn, "league_standings", pd.DataFrame(
        _standings(2024, {1: 1, 2: 2, 3: 3, 4: 4}, {1: 400.0, 2: 350.0, 3: 300.0, 4: 250.0})
        + _standings(2025, {1: 4, 2: 1, 3: 2, 4: 3}, {1: 250.0, 2: 400.0, 3: 350.0, 4: 300.0})))
    m = [
        # 2024 regular season. Scores: A 130, B 120, C 110, D 100 each week,
        # except week 3 where D (105) beats C (95).
        _matchup(2024, 1, 1, 2, 130, 120), _matchup(2024, 1, 3, 4, 110, 100),
        _matchup(2024, 2, 1, 3, 130, 110), _matchup(2024, 2, 2, 4, 120, 100),
        _matchup(2024, 3, 1, 4, 130, 100), _matchup(2024, 3, 2, 3, 120, 95),
        _matchup(2024, 4, 1, 2, 140, 120, tier="WINNERS_BRACKET"),
        _matchup(2024, 4, 3, 4, 100, 90, tier="LOSERS_CONSOLATION_LADDER"),
        # 2025: B dominates, A collapses.
        _matchup(2025, 1, 2, 1, 130, 100), _matchup(2025, 1, 3, 4, 120, 110),
        _matchup(2025, 2, 2, 3, 130, 120), _matchup(2025, 2, 1, 4, 100, 110),
        _matchup(2025, 3, 2, 4, 130, 110), _matchup(2025, 3, 1, 3, 100, 120),
        _matchup(2025, 4, 2, 3, 140, 120, tier="WINNERS_BRACKET"),
    ]
    write_table(conn, "league_matchups", pd.DataFrame(m))
    return conn


def test_members_resolves_every_member_once_with_seasons(league):
    df = mp.members(league)
    assert sorted(df.member_id) == sorted([A, B, C, D])
    assert df[df.member_id == A].iloc[0].seasons == [2024, 2025]


def test_finishes_read_the_standings_through_the_members_table(league):
    f = mp.finishes(league, A)
    assert f["n"] == 2
    assert [s["season"] for s in f["seasons"]] == [2024, 2025]
    assert f["seasons"][0]["final_rank"] == 1 and f["seasons"][1]["final_rank"] == 4
    # draft_day_rank 1 both years: projected 1st, finished 1st then 4th.
    assert f["seasons"][0]["outperformance"] == 0 and f["seasons"][1]["outperformance"] == -3
    assert f["titles"] == [2024] and f["avg_finish"] == 2.5
    assert f["points_per_season"] == 325.0


def test_playoffs_come_from_the_bracket_not_the_seed(league):
    p = mp.playoffs(league, A)
    assert p["appearances"] == [2024] and p["bracket_wins"] == 1 and p["bracket_losses"] == 0
    assert p["titles"] == [2024]
    d = mp.playoffs(league, D)
    assert d["appearances"] == [] and d["consolation"] == {"wins": 0, "losses": 1}
    assert d["last_place"] == [2024]
    assert mp.playoffs(league, B)["bracket_losses"] == 1


def test_head_to_head_counts_every_meeting_both_ways(league):
    h2h = mp.head_to_head(league)
    ab = h2h[(A, B)]
    # 2024 wk1 A beat B, wk4 final A beat B; 2025 wk1 B beat A.
    assert (ab["wins"], ab["losses"], ab["games"]) == (2, 1, 3)
    assert ab["playoff_games"] == 1
    assert ab["margin"] == pytest.approx((130 - 120) + (140 - 120) + (100 - 130))
    ba = h2h[(B, A)]
    assert (ba["wins"], ba["losses"]) == (1, 2) and ba["margin"] == -ab["margin"]


def test_luck_is_actual_wins_minus_all_play_expectation(league):
    a = mp.luck(league, A)
    s24 = next(s for s in a["seasons"] if s["season"] == 2024)
    assert s24["actual_wins"] == 3 and s24["expected_wins"] == pytest.approx(3.0)
    assert s24["luck"] == pytest.approx(0.0)
    d = mp.luck(league, D)
    s24 = next(s for s in d["seasons"] if s["season"] == 2024)
    # D lost all three but out-scored C in week 3: all-play 1 of 9.
    assert s24["actual_wins"] == 0 and s24["expected_wins"] == pytest.approx(1 / 3)
    assert s24["luck"] == pytest.approx(-1 / 3)
    assert d["n"] == 6      # regular-season games over two seasons


def test_seasons_strip_names_the_champion_and_the_last_place(league):
    strip = mp.seasons_strip(league)
    assert [s["season"] for s in strip] == [2024, 2025]
    assert strip[0]["champion"]["member_id"] == A and strip[0]["last"]["member_id"] == D
    assert strip[0]["top_scorer"]["member_id"] == A
    assert strip[1]["champion"]["member_id"] == B and strip[1]["last"]["member_id"] == A
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_manager_profile.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'scoring.manager_profile'`

- [ ] **Step 3: Write the first half of the module**

Create `scoring/manager_profile.py`:

```python
"""A manager's profile, from the league's own history.

Pure functions over a league database. No network, no model. Every
section returns plain dicts and lists so the API can serialise them as
they are, and every section carries `n` -- the number of observations it
rests on -- so the page can decline to put an adjective on three data
points.

The manager key is the ESPN member id (SWID). Teams are season-scoped and
resolve through `league_members`, which is also how `league_standings`
(keyed on team) is read here.
"""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from pipeline.db import read_table

PLAYOFF_TIERS = {"WINNERS_BRACKET"}
CONSOLATION_TIERS = {"WINNERS_CONSOLATION_LADDER", "LOSERS_CONSOLATION_LADDER"}


def _num(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _int(value):
    if value is None or pd.isna(value):
        return None
    return int(value)


def members(conn) -> pd.DataFrame:
    """One row per member across seasons: `member_id`, `display_name` (the
    newest season's), `seasons` (ascending), `teams` ({season: team_id})."""
    m = read_table(conn, "league_members")
    if m.empty:
        return pd.DataFrame(columns=["member_id", "display_name", "seasons", "teams"])
    rows = []
    for swid, g in m.sort_values("season").groupby("member_id"):
        played = g[g.team_id.notna()]
        rows.append({
            "member_id": swid,
            "display_name": g.iloc[-1].display_name,
            "first_name": g.iloc[-1].first_name,
            "seasons": [int(s) for s in played.season],
            "teams": {int(r.season): int(r.team_id) for r in played.itertuples()},
        })
    return pd.DataFrame(rows)


def _team_of(conn) -> dict:
    """(season, team_id) -> member_id, for joining team-keyed tables."""
    m = read_table(conn, "league_members")
    if m.empty:
        return {}
    return {(int(r.season), int(r.team_id)): r.member_id
            for r in m[m.team_id.notna()].itertuples()}


def _teams_for(conn, member_id: str) -> dict:
    """season -> team_id for one member."""
    return {season: team for (season, team), swid in _team_of(conn).items()
            if swid == member_id}


def finishes(conn, member_id: str) -> dict:
    mine = _teams_for(conn, member_id)
    st = read_table(conn, "league_standings")
    mem = read_table(conn, "league_members")
    seasons = []
    for season in sorted(mine):
        row = st[(st.season == season) & (st.team_id == mine[season])]
        me = mem[(mem.season == season) & (mem.member_id == member_id)]
        if row.empty:
            continue
        r = row.iloc[0]
        projected = _int(me.iloc[0].draft_day_rank) if not me.empty else None
        final = _int(r.final_rank)
        seasons.append({
            "season": int(season), "team_name": r.team_name,
            "wins": _int(r.wins), "losses": _int(r.losses), "ties": _int(r.ties),
            "points_for": _num(r.points_for), "points_against": _num(r.points_against),
            "playoff_seed": _int(r.playoff_seed), "final_rank": final,
            "draft_day_rank": projected,
            "outperformance": (projected - final) if projected is not None and final is not None else None,
        })
    done = [s for s in seasons if s["final_rank"] is not None]
    games = sum((s["wins"] or 0) + (s["losses"] or 0) + (s["ties"] or 0) for s in done)
    wins = sum(s["wins"] or 0 for s in done)
    points = [s["points_for"] for s in done if s["points_for"] is not None]
    return {
        "n": len(done),
        "seasons": seasons,
        "titles": [s["season"] for s in done if s["final_rank"] == 1],
        "avg_finish": round(sum(s["final_rank"] for s in done) / len(done), 2) if done else None,
        "win_pct": round(wins / games, 3) if games else None,
        "points_per_season": round(sum(points) / len(points), 1) if points else None,
        "outperformance": round(sum(s["outperformance"] for s in done
                                    if s["outperformance"] is not None), 1) if done else None,
    }


def _matchups(conn) -> pd.DataFrame:
    m = read_table(conn, "league_matchups")
    return m if not m.empty else pd.DataFrame(columns=[
        "season", "matchup_period", "tier", "home_team_id", "away_team_id",
        "home_points", "away_points", "winner"])


def _sides(conn):
    """Every matchup as two rows, one per side: member, opponent, points,
    opponent points, won/lost/tied, tier. Byes are dropped."""
    owner = _team_of(conn)
    rows = []
    for r in _matchups(conn).itertuples():
        if pd.isna(r.away_team_id) or pd.isna(r.home_team_id):
            continue
        season = int(r.season)
        home = owner.get((season, int(r.home_team_id)))
        away = owner.get((season, int(r.away_team_id)))
        if home is None or away is None:
            continue
        hp, ap = _num(r.home_points) or 0.0, _num(r.away_points) or 0.0
        for me, them, pf, pa in ((home, away, hp, ap), (away, home, ap, hp)):
            result = "T" if r.winner == "TIE" else ("W" if pf > pa else "L")
            if r.winner == "UNDECIDED":
                continue
            rows.append({"season": season, "period": int(r.matchup_period), "tier": r.tier,
                         "member_id": me, "opponent_id": them, "points": pf,
                         "against": pa, "result": result})
    return pd.DataFrame(rows, columns=["season", "period", "tier", "member_id",
                                       "opponent_id", "points", "against", "result"])


def playoffs(conn, member_id: str) -> dict:
    sides = _sides(conn)
    mine = sides[sides.member_id == member_id]
    bracket = mine[mine.tier.isin(PLAYOFF_TIERS)]
    consolation = mine[mine.tier.isin(CONSOLATION_TIERS)]
    st = read_table(conn, "league_standings")
    teams = _teams_for(conn, member_id)
    last, titles = [], []
    for season, team in teams.items():
        row = st[(st.season == season) & (st.team_id == team)]
        if row.empty or pd.isna(row.iloc[0].final_rank):
            continue
        size = int(st[st.season == season].team_id.nunique())
        if int(row.iloc[0].final_rank) == size:
            last.append(int(season))
        if int(row.iloc[0].final_rank) == 1:
            titles.append(int(season))
    return {
        "n": int(len(bracket) + len(consolation)),
        "appearances": sorted(int(s) for s in bracket.season.unique()),
        "bracket_wins": int((bracket.result == "W").sum()),
        "bracket_losses": int((bracket.result == "L").sum()),
        "consolation": {"wins": int((consolation.result == "W").sum()),
                        "losses": int((consolation.result == "L").sum())},
        "titles": sorted(titles),
        "last_place": sorted(last),
    }


def head_to_head(conn) -> dict:
    """(member, opponent) -> games, wins, losses, ties, margin, playoff_games."""
    out = defaultdict(lambda: {"games": 0, "wins": 0, "losses": 0, "ties": 0,
                               "margin": 0.0, "playoff_games": 0})
    for r in _sides(conn).itertuples():
        rec = out[(r.member_id, r.opponent_id)]
        rec["games"] += 1
        rec["wins"] += r.result == "W"
        rec["losses"] += r.result == "L"
        rec["ties"] += r.result == "T"
        rec["margin"] += r.points - r.against
        rec["playoff_games"] += r.tier in PLAYOFF_TIERS
    for rec in out.values():
        rec["margin"] = round(rec["margin"], 2)
        rec["n"] = rec["games"]
    return dict(out)


def luck(conn, member_id: str) -> dict:
    """All-play: each regular-season week, my points against every other
    team's that week. Expected wins is the all-play win fraction times
    games played; luck is actual minus expected."""
    sides = _sides(conn)
    regular = sides[sides.tier == "NONE"]
    seasons = []
    total_n = 0
    for season, g in regular.groupby("season"):
        mine = g[g.member_id == member_id]
        if mine.empty:
            continue
        allplay_w = allplay_n = 0
        for r in mine.itertuples():
            others = g[(g.period == r.period) & (g.member_id != member_id)]
            allplay_w += int((others.points < r.points).sum()) + 0.5 * int((others.points == r.points).sum())
            allplay_n += len(others)
        games = len(mine)
        actual = int((mine.result == "W").sum()) + 0.5 * int((mine.result == "T").sum())
        expected = games * (allplay_w / allplay_n) if allplay_n else 0.0
        seasons.append({"season": int(season), "games": games, "actual_wins": actual,
                        "expected_wins": round(expected, 2), "luck": round(actual - expected, 2)})
        total_n += games
    return {"n": total_n, "seasons": seasons,
            "luck": round(sum(s["luck"] for s in seasons), 2) if seasons else None}


def _name(conn, member_id: str) -> str:
    m = members(conn)
    hit = m[m.member_id == member_id]
    return str(hit.iloc[0].display_name) if not hit.empty else member_id


def seasons_strip(conn) -> list:
    """Per season: champion, runner-up, last place, top scorer."""
    st = read_table(conn, "league_standings")
    owner = _team_of(conn)
    out = []
    if st.empty:
        return out
    for season, g in st.sort_values("season").groupby("season"):
        season = int(season)

        def who(team_id):
            swid = owner.get((season, int(team_id)))
            return {"member_id": swid, "display_name": _name(conn, swid) if swid else None}

        ranked = g[g.final_rank.notna()].sort_values("final_rank")
        scored = g[g.points_for.notna()].sort_values("points_for", ascending=False)
        out.append({
            "season": season,
            "teams": int(g.team_id.nunique()),
            "complete": not ranked.empty,
            "champion": who(ranked.iloc[0].team_id) if not ranked.empty else None,
            "runner_up": who(ranked.iloc[1].team_id) if len(ranked) > 1 else None,
            "last": who(ranked.iloc[-1].team_id) if not ranked.empty else None,
            "top_scorer": who(scored.iloc[0].team_id) if not scored.empty else None,
        })
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_manager_profile.py -q -p no:cacheprovider`
Expected: all PASS. If `test_luck` fails on D's expected wins, check `_sides`: the `others` frame must contain the three other teams' rows for the period (six rows exist per period, one per side; filter `member_id != me` leaves three, not five -- each team appears once as `member_id`).

- [ ] **Step 5: Commit**

```bash
git add scoring/manager_profile.py tests/test_manager_profile.py
git commit -m "A manager's finishes, playoffs, head-to-head and luck, from the league's own tables"
```

---

### Task 4: Facts from transactions

**Files:**
- Modify: `scoring/manager_profile.py` (append)
- Test: `tests/test_manager_profile.py` (append)

**Interfaces:**
- Consumes: `league_transactions`, `league_lineups` (for trade balance and player names), `league_members`.
- Produces: `waivers(conn, member_id) -> dict`, `trades(conn, member_id) -> dict`, `player_names(conn) -> dict[int, str]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_manager_profile.py`:

```python
def _txn(season, week, tid, team, member, type_, status, execution, items, bid=0.0,
         related=None, when="2025-10-01T12:00:00Z"):
    return {"season": season, "week": week, "txn_id": tid, "related_txn_id": related,
            "team_id": team, "member_id": member, "type": type_, "status": status,
            "execution_type": execution, "bid_amount": bid,
            "proposed_at": pd.Timestamp(when), "items_json": json.dumps(items)}


def _add(pid, to_team):
    return {"type": "ADD", "playerId": pid, "fromTeamId": 0, "toTeamId": to_team,
            "fromLineupSlotId": -1, "toLineupSlotId": 20}


def _drop(pid, from_team):
    return {"type": "DROP", "playerId": pid, "fromTeamId": from_team, "toTeamId": 0,
            "fromLineupSlotId": 20, "toLineupSlotId": -1}


def _trade_item(pid, from_team, to_team):
    return {"type": "TRADE", "playerId": pid, "fromTeamId": from_team, "toTeamId": to_team,
            "fromLineupSlotId": 20, "toLineupSlotId": 20}


def _lineup(season, week, team, pid, name, pos, slot, actual, projected=None):
    return {"season": season, "week": week, "team_id": team, "player_id": pid,
            "player_name": name, "position": pos, "lineup_slot": slot,
            "actual_points": actual, "projected_points": projected,
            "acquisition_type": "DRAFT", "injury_status": "ACTIVE"}


@pytest.fixture
def activity(league):
    """On top of `league`: 2025 transactions and lineups.

    A (team 1) claims player 100 in week 2 and wins it; claims 101 in week
    3 and loses it to B (same process run, B's executed, A's failed); adds
    102 as a free agent on a Tuesday; cancels a claim on 103.
    A proposes a trade to B in week 3 (A sends 200, gets 300); B accepts.
    After the trade, 300 scores 20 and 25 for A in weeks 4-5; 200 scores
    10 and 10 for B. Balance for A: (20+25) - (10+10) = +25.
    B proposes a trade to A in week 4; A declines."""
    conn = league
    t = [
        _txn(2025, 2, "w1", 1, A, "WAIVER", "PENDING", "EXECUTE", [_add(100, 1)],
             when="2025-09-15T18:00:00Z"),
        _txn(2025, 2, "w1p", 1, A, "WAIVER", "EXECUTED", "PROCESS", [_add(100, 1)],
             related="w1", when="2025-09-17T07:00:00Z"),
        _txn(2025, 3, "w2", 1, A, "WAIVER", "PENDING", "EXECUTE", [_add(101, 1)]),
        _txn(2025, 3, "w2p", 1, A, "WAIVER", "FAILED_INVALIDPLAYERSOURCE", "PROCESS",
             [_add(101, 1)], related="w2", when="2025-09-24T07:00:00Z"),
        _txn(2025, 3, "w2b", 2, B, "WAIVER", "PENDING", "EXECUTE", [_add(101, 2)]),
        _txn(2025, 3, "w2bp", 2, B, "WAIVER", "EXECUTED", "PROCESS", [_add(101, 2)],
             related="w2b", when="2025-09-24T07:00:00Z"),
        _txn(2025, 3, "fa1", 1, A, "FREEAGENT", "EXECUTED", "EXECUTE",
             [_add(102, 1), _drop(50, 1)], when="2025-09-23T15:00:00Z"),   # a Tuesday
        _txn(2025, 3, "wc", 1, A, "WAIVER", "PENDING", "EXECUTE", [_add(103, 1)]),
        _txn(2025, 3, "wcc", 1, A, "WAIVER", "CANCELED", "CANCEL", [_add(103, 1)], related="wc"),
        _txn(2025, 3, "tp1", 1, A, "TRADE_PROPOSAL", "PENDING", "EXECUTE",
             [_trade_item(200, 1, 2), _trade_item(300, 2, 1)]),
        _txn(2025, 3, "ta1", 2, B, "TRADE_ACCEPT", "EXECUTED", "PROCESS",
             [_trade_item(200, 1, 2), _trade_item(300, 2, 1)], related="tp1",
             when="2025-09-25T10:00:00Z"),
        _txn(2025, 4, "tp2", 2, B, "TRADE_PROPOSAL", "PENDING", "EXECUTE",
             [_trade_item(201, 2, 1), _trade_item(301, 1, 2)]),
        _txn(2025, 4, "td2", 1, A, "TRADE_DECLINE", "EXECUTED", "EXECUTE", [], related="tp2"),
        _txn(2025, 3, "lm", 1, A, "FUTURE_ROSTER", "EXECUTED", "EXECUTE",
             [{"type": "LINEUP", "playerId": 300, "fromLineupSlotId": 20, "toLineupSlotId": 2}]),
    ]
    write_table(conn, "league_transactions", pd.DataFrame(t))
    lu = [
        _lineup(2025, 3, 1, 200, "Sent Back", "RB", 2, 12.0),
        _lineup(2025, 3, 2, 300, "Got Back", "RB", 2, 15.0),
        _lineup(2025, 4, 1, 300, "Got Back", "RB", 2, 20.0),
        _lineup(2025, 5, 1, 300, "Got Back", "RB", 2, 25.0),
        _lineup(2025, 4, 2, 200, "Sent Back", "RB", 2, 10.0),
        _lineup(2025, 5, 2, 200, "Sent Back", "RB", 2, 10.0),
    ]
    write_table(conn, "league_lineups", pd.DataFrame(lu))
    return conn


def test_waivers_count_claims_wins_losses_and_free_agents(activity):
    w = mp.waivers(activity, A)
    assert w["n"] == 3                      # claims made: 100, 101, 103
    assert w["claims"] == 3 and w["won"] == 1 and w["lost"] == 1 and w["canceled"] == 1
    assert w["free_agent_adds"] == 1 and w["drops"] == 1
    assert w["adds_by_weekday"]["Tue"] == 1
    assert w["busiest_week"] == {"season": 2025, "week": 3, "moves": 2}   # won/lost/fa: week 3 has fa + lost
    assert "bids" not in w                  # no FAAB this league
    b = mp.waivers(activity, B)
    assert b["won"] == 1 and b["lost"] == 0


def test_trades_count_proposals_outcomes_partners_and_balance(activity):
    t = mp.trades(activity, A)
    assert t["proposed"] == 1 and t["received"] == 1
    assert t["accepted"] == 1 and t["declined_by_me"] == 1 and t["declined_by_them"] == 0
    assert t["vetoed"] == 0
    assert t["partners"] == [{"member_id": B, "display_name": "b", "trades": 1}]
    assert t["n"] == 1                      # executed trades the balance rests on
    assert t["balance"] == pytest.approx(25.0)
    assert t["ledger"][0]["sent"] == [{"player_id": 200, "name": "Sent Back"}]
    assert t["ledger"][0]["received"] == [{"player_id": 300, "name": "Got Back"}]
    assert t["ledger"][0]["balance"] == pytest.approx(25.0)
    b = mp.trades(activity, B)
    assert b["balance"] == pytest.approx(-25.0) and b["declined_by_them"] == 1


def test_player_names_come_from_any_week_a_player_was_rostered(activity):
    names = mp.player_names(activity)
    assert names[300] == "Got Back" and 999 not in names
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_manager_profile.py -q -p no:cacheprovider -k "waivers or trades or player_names"`
Expected: FAIL with `AttributeError: module 'scoring.manager_profile' has no attribute 'waivers'`

- [ ] **Step 3: Write the transaction facts**

Append to `scoring/manager_profile.py`:

```python
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _txns(conn) -> pd.DataFrame:
    t = read_table(conn, "league_transactions")
    if t.empty:
        return pd.DataFrame(columns=["season", "week", "txn_id", "related_txn_id", "team_id",
                                     "member_id", "type", "status", "execution_type",
                                     "bid_amount", "proposed_at", "items_json"])
    t = t.copy()
    t["items"] = t.items_json.map(lambda s: json.loads(s) if isinstance(s, str) else [])
    return t


def player_names(conn) -> dict:
    """player id -> the name the roster gave it, from any week it was rostered."""
    lu = read_table(conn, "league_lineups")
    if lu.empty:
        return {}
    named = lu[lu.player_name.notna()].drop_duplicates("player_id", keep="last")
    return {int(r.player_id): str(r.player_name) for r in named.itertuples()}


def _faab(conn) -> set:
    """Seasons whose settings used an acquisition budget. Read from the raw
    mSettings answer when it is stored; absent otherwise."""
    raw = read_table(conn, "league_raw")
    out = set()
    if raw.empty:
        return out
    for r in raw[raw.view == "mSettings"].itertuples():
        settings = (json.loads(r.payload_json).get("settings") or {}).get("acquisitionSettings") or {}
        if settings.get("isUsingAcquisitionBudget"):
            out.add(int(r.season))
    return out


def waivers(conn, member_id: str) -> dict:
    t = _txns(conn)
    mine = t[t.member_id == member_id]
    claims = mine[(mine.type == "WAIVER") & (mine.execution_type == "EXECUTE")]
    outcomes = mine[(mine.type == "WAIVER") & (mine.execution_type != "EXECUTE")]
    won = outcomes[outcomes.status == "EXECUTED"]
    lost = outcomes[outcomes.status.str.startswith("FAILED", na=False)]
    canceled = outcomes[outcomes.status == "CANCELED"]
    fa = mine[(mine.type == "FREEAGENT") & (mine.status == "EXECUTED")]
    drops = sum(1 for items in pd.concat([fa["items"], won["items"]]) if not len(items) == 0
                for i in items if i.get("type") == "DROP")
    by_day = {d: 0 for d in WEEKDAYS}
    for ts in pd.concat([fa.proposed_at, claims.proposed_at]):
        if pd.notna(ts):
            by_day[WEEKDAYS[pd.Timestamp(ts).dayofweek]] += 1
    moves = pd.concat([fa, won, lost]).groupby(["season", "week"]).size()
    busiest = None
    if len(moves):
        (season, week), count = moves.idxmax(), int(moves.max())
        busiest = {"season": int(season), "week": int(week), "moves": count}
    out = {
        "n": int(len(claims)),
        "claims": int(len(claims)), "won": int(len(won)), "lost": int(len(lost)),
        "canceled": int(len(canceled)),
        "win_rate": round(len(won) / (len(won) + len(lost)), 3) if (len(won) + len(lost)) else None,
        "free_agent_adds": int(len(fa)), "drops": int(drops),
        "adds_by_weekday": by_day, "busiest_week": busiest,
        "seasons": sorted(int(s) for s in mine.season.unique()),
    }
    faab = _faab(conn)
    bids = claims[claims.season.isin(faab)] if faab else claims.iloc[0:0]
    if len(bids):
        out["bids"] = {"n": int(len(bids)), "mean": round(float(bids.bid_amount.mean()), 1),
                       "max": float(bids.bid_amount.max()),
                       "spent": float(won[won.season.isin(faab)].bid_amount.sum())}
    return out


def _rest_of_season_points(lu: pd.DataFrame, season: int, after_week: int,
                           player_id: int, team_id: int) -> float:
    rows = lu[(lu.season == season) & (lu.week > after_week) & (lu.player_id == player_id)
              & (lu.team_id == team_id) & (lu.actual_points.notna())]
    return float(rows.actual_points.sum())


def trades(conn, member_id: str) -> dict:
    t = _txns(conn)
    teams = _teams_for(conn, member_id)
    lu = read_table(conn, "league_lineups")
    names = player_names(conn)
    display = {r.member_id: r.display_name for r in members(conn).itertuples()}
    owner = _team_of(conn)
    proposals = t[t.type == "TRADE_PROPOSAL"]

    def counterpart(items, season):
        for i in items:
            for key in ("toTeamId", "fromTeamId"):
                tid = i.get(key)
                if tid and tid != teams.get(season):
                    return owner.get((season, int(tid)))
        return None

    def involves(row):
        return any(teams.get(int(row.season)) in (i.get("fromTeamId"), i.get("toTeamId"))
                   for i in row["items"])

    mine_p = proposals[proposals.member_id == member_id]
    received = proposals[(proposals.member_id != member_id) & proposals.apply(involves, axis=1)] \
        if len(proposals) else proposals
    accepts = t[(t.type == "TRADE_ACCEPT")]
    declines = t[t.type == "TRADE_DECLINE"]
    vetoes = t[t.type == "TRADE_VETO"]
    my_ids = set(mine_p.txn_id)
    their_ids = set(received.txn_id)
    executed = accepts[accepts.related_txn_id.isin(my_ids | their_ids)
                       & accepts.status.isin(["EXECUTED", None]) | accepts.related_txn_id.isin(my_ids | their_ids) & accepts.status.isna()]
    partners = defaultdict(int)
    ledger = []
    for r in executed.itertuples():
        season = int(r.season)
        my_team = teams.get(season)
        items = r.items
        other = counterpart(items, season)
        partners[other] += 1
        sent = [i for i in items if i.get("fromTeamId") == my_team]
        got = [i for i in items if i.get("toTeamId") == my_team]
        other_team = next((i.get("toTeamId") for i in sent), None)
        week = int(r.week)
        balance = (sum(_rest_of_season_points(lu, season, week, int(i["playerId"]), my_team) for i in got)
                   - sum(_rest_of_season_points(lu, season, week, int(i["playerId"]), other_team) for i in sent))
        ledger.append({
            "season": season, "week": week, "with": other,
            "with_name": display.get(other, other),
            "sent": [{"player_id": int(i["playerId"]), "name": names.get(int(i["playerId"]), str(i["playerId"]))} for i in sent],
            "received": [{"player_id": int(i["playerId"]), "name": names.get(int(i["playerId"]), str(i["playerId"]))} for i in got],
            "balance": round(balance, 1),
        })
    ledger.sort(key=lambda x: (x["season"], x["week"]), reverse=True)
    return {
        "n": len(ledger),
        "proposed": int(len(mine_p)), "received": int(len(received)),
        "accepted": len(ledger),
        "declined_by_me": int(declines[declines.member_id == member_id].related_txn_id.isin(their_ids).sum()),
        "declined_by_them": int(declines[declines.member_id != member_id].related_txn_id.isin(my_ids).sum()),
        "vetoed": int(vetoes.related_txn_id.isin(my_ids | their_ids).sum()),
        "partners": [{"member_id": m, "display_name": display.get(m, m), "trades": c}
                     for m, c in sorted(partners.items(), key=lambda kv: -kv[1])],
        "balance": round(sum(l["balance"] for l in ledger), 1) if ledger else None,
        "ledger": ledger,
    }
```

Note for the implementer: ESPN's `TRADE_ACCEPT` rows carry `status` `None` in some seasons and `EXECUTED` in others (both observed); the `executed` filter above must accept both -- simplify it to `accepts[accepts.related_txn_id.isin(my_ids | their_ids) & (accepts.status.isna() | (accepts.status == "EXECUTED"))]` if the one-liner in the sketch does not read cleanly.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_manager_profile.py -q -p no:cacheprovider`
Expected: all PASS. `test_waivers`' `busiest_week` counts fa + lost in week 3 (2 moves) against won in week 2 (1 move).

- [ ] **Step 5: Commit**

```bash
git add scoring/manager_profile.py tests/test_manager_profile.py
git commit -m "A manager's waiver and trade record, with each trade's rest-of-season balance"
```

---

### Task 5: Facts from lineups, the draft chapter's flags, and the overview

**Files:**
- Modify: `scoring/manager_profile.py` (append)
- Test: `tests/test_manager_profile.py` (append)

**Interfaces:**
- Consumes: `league_lineups`, `league_members.lineup_moves`, `league_draft_flags`, `league_raw` (mSettings for `lineupSlotCounts`).
- Produces: `lineups(conn, member_id) -> dict`, `draft_flags(conn, member_id) -> dict`, `optimal_points(entries, slot_counts) -> float`, `defining_line(conn, member_id, grid) -> str`, `league_overview(conn) -> dict`, `profile(conn, member_id) -> dict`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_manager_profile.py`:

```python
SLOTS = {"0": 1, "2": 1, "4": 1, "23": 1, "20": 3}     # QB, RB, WR, FLEX, 3 bench


def test_optimal_points_fills_flex_with_the_best_leftover():
    entries = [("QB", 20.0), ("RB", 15.0), ("RB", 12.0), ("WR", 10.0), ("WR", 18.0), ("TE", 9.0)]
    # QB 20 + RB 15 + WR 18 + FLEX best of {RB 12, WR 10, TE 9} = 12 -> 65
    assert mp.optimal_points(entries, SLOTS) == pytest.approx(65.0)


@pytest.fixture
def lineups_league(league):
    conn = league
    raw = pd.DataFrame([{"season": 2025, "view": "mSettings", "week": 0,
                         "fetched_at": pd.Timestamp.now(tz="UTC"),
                         "payload_json": json.dumps({"settings": {
                             "rosterSettings": {"lineupSlotCounts": SLOTS},
                             "acquisitionSettings": {"isUsingAcquisitionBudget": False}}})}])
    write_table(conn, "league_raw", raw)
    # A, 2025 week 1: started QB 20, RB 8, WR 18, FLEX RB 5; benched RB 15 (OUT starter RB 8).
    # Optimal: 20 + 15 + 18 + 8 = 61; started 51; left 10.
    rows = [
        _lineup(2025, 1, 1, 1, "Q", "QB", 0, 20.0), _lineup(2025, 1, 1, 2, "R1", "RB", 2, 8.0),
        _lineup(2025, 1, 1, 3, "W", "WR", 4, 18.0), _lineup(2025, 1, 1, 4, "R2", "RB", 23, 5.0),
        _lineup(2025, 1, 1, 5, "R3", "RB", 20, 15.0),
        # week 2: optimal lineup started (hit). QB 10, RB 10, WR 10, FLEX 10, bench 1.
        _lineup(2025, 2, 1, 1, "Q", "QB", 0, 10.0), _lineup(2025, 2, 1, 2, "R1", "RB", 2, 10.0),
        _lineup(2025, 2, 1, 3, "W", "WR", 4, 10.0), _lineup(2025, 2, 1, 4, "R2", "RB", 23, 10.0),
        _lineup(2025, 2, 1, 5, "R3", "RB", 20, 1.0),
    ]
    df = pd.DataFrame(rows)
    df.loc[(df.week == 1) & (df.player_id == 2), "injury_status"] = "OUT"
    write_table(conn, "league_lineups", df)
    flags = pd.DataFrame([
        {"season": 2025, "overall_pick": 1, "round": 1, "team_id": 1, "member_id": A,
         "player_id": 1, "autodraft": False, "keeper": False},
        {"season": 2025, "overall_pick": 5, "round": 2, "team_id": 1, "member_id": A,
         "player_id": 2, "autodraft": True, "keeper": False},
    ])
    write_table(conn, "league_draft_flags", flags)
    return conn


def test_lineups_measure_points_left_hit_rate_and_bad_starts(lineups_league):
    l = mp.lineups(lineups_league, A)
    assert l["n"] == 2
    weeks = {w["week"]: w for w in l["weeks"]}
    assert weeks[1]["started"] == 51.0 and weeks[1]["optimal"] == 61.0 and weeks[1]["left"] == 10.0
    assert weeks[2]["left"] == 0.0
    assert l["bench_points_left"] == 5.0 and l["hit_rate"] == 0.5
    assert l["started_out"] == 1
    assert l["worst_week"] == {"season": 2025, "week": 1, "left": 10.0}
    assert l["moves_per_week"] is None or l["moves_per_week"] > 0


def test_draft_flags_give_an_autodraft_rate(lineups_league):
    d = mp.draft_flags(lineups_league, A)
    assert d["n"] == 2 and d["autodraft_rate"] == 0.5 and d["autodrafts"] == 1


def test_profile_assembles_every_section_with_names(lineups_league):
    p = mp.profile(lineups_league, A)
    assert p["member_id"] == A and p["display_name"] == "a"
    for key in ("finishes", "playoffs", "head_to_head", "luck", "waivers", "trades",
                "lineups", "draft_flags"):
        assert key in p
    assert p["head_to_head"][0]["opponent"]["display_name"] in {"b", "c", "d"}
    assert mp.profile(lineups_league, "{NOBODY}") is None


def test_overview_has_the_strip_and_a_grid_with_a_defining_line(lineups_league):
    o = mp.league_overview(lineups_league)
    assert [s["season"] for s in o["seasons"]] == [2024, 2025]
    grid = {m["member_id"]: m for m in o["members"]}
    assert grid[A]["titles"] == 1 and grid[A]["seasons"] == 2
    assert isinstance(grid[A]["defining_line"], str) and grid[A]["defining_line"]
    # D has no lineups, no trades: the line falls back to the record.
    assert "finish" in grid[D]["defining_line"] or "-" in grid[D]["defining_line"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_manager_profile.py -q -p no:cacheprovider -k "optimal or lineups or draft_flags or profile or overview"`
Expected: FAIL with `AttributeError: ... 'optimal_points'`

- [ ] **Step 3: Write the lineup facts, the flags, the overview and `profile`**

Append to `scoring/manager_profile.py`:

```python
from pipeline.espn_league import (ESPN_BENCH_SLOT, ESPN_FLEX_SLOT, ESPN_IR_SLOT,
                                  ESPN_SLOT_POSITIONS)

FLEX_POSITIONS = {"RB", "WR", "TE"}


def _slot_counts(conn, season: int) -> dict:
    raw = read_table(conn, "league_raw")
    if raw.empty:
        return {}
    hit = raw[(raw.view == "mSettings") & (raw.season == season)]
    if hit.empty:
        return {}
    settings = json.loads(hit.iloc[0].payload_json).get("settings") or {}
    return (settings.get("rosterSettings") or {}).get("lineupSlotCounts") or {}


def optimal_points(entries: list, slot_counts: dict) -> float:
    """The most a roster could have scored that week under the league's
    starting slots. `entries` is [(position, points)]. Fixed slots take
    their position's best scorers first; FLEX takes the best of what is
    left among RB/WR/TE. Greedy is exact here because every fixed slot
    admits one position and FLEX is filled last from the remainder."""
    pool = sorted(((pts, pos) for pos, pts in entries if pts is not None), reverse=True)
    total = 0.0
    for slot_id, count in slot_counts.items():
        pos = ESPN_SLOT_POSITIONS.get(int(slot_id))
        if pos is None:
            continue
        for _ in range(int(count)):
            pick = next((e for e in pool if e[1] == pos), None)
            if pick is None:
                break
            pool.remove(pick)
            total += pick[0]
    for _ in range(int(slot_counts.get(str(ESPN_FLEX_SLOT), 0))):
        pick = next((e for e in pool if e[1] in FLEX_POSITIONS), None)
        if pick is None:
            break
        pool.remove(pick)
        total += pick[0]
    return round(total, 2)


def lineups(conn, member_id: str) -> dict:
    lu = read_table(conn, "league_lineups")
    teams = _teams_for(conn, member_id)
    mem = read_table(conn, "league_members")
    weeks = []
    started_out = 0
    for season, team in teams.items():
        slots = _slot_counts(conn, season)
        rows = lu[(lu.season == season) & (lu.team_id == team)] if not lu.empty else lu
        if rows.empty or not slots:
            continue
        for week, g in rows.groupby("week"):
            scored = g[g.actual_points.notna()]
            if scored.empty:
                continue
            starters = scored[~scored.lineup_slot.isin([ESPN_BENCH_SLOT, ESPN_IR_SLOT])]
            started = float(starters.actual_points.sum())
            optimal = optimal_points([(r.position, float(r.actual_points))
                                      for r in scored.itertuples()], slots)
            started_out += int(starters.injury_status.isin(["OUT", "IR", "SUSPENSION"]).sum())
            weeks.append({"season": int(season), "week": int(week), "started": round(started, 2),
                          "optimal": optimal, "left": round(max(0.0, optimal - started), 2)})
    n = len(weeks)
    moves = mem[(mem.member_id == member_id) & mem.lineup_moves.notna()]
    moves_per_week = None
    if not moves.empty and n:
        seasons = {w["season"] for w in weeks}
        per_season = {int(r.season): int(r.lineup_moves) for r in moves.itertuples()}
        counted = [per_season[s] for s in seasons if s in per_season]
        weeks_in = sum(1 for w in weeks if w["season"] in per_season)
        moves_per_week = round(sum(counted) / weeks_in, 1) if weeks_in else None
    worst = max(weeks, key=lambda w: w["left"]) if weeks else None
    return {
        "n": n,
        "weeks": weeks,
        "bench_points_left": round(sum(w["left"] for w in weeks) / n, 1) if n else None,
        "hit_rate": round(sum(1 for w in weeks if w["left"] <= 0.5) / n, 3) if n else None,
        "started_out": started_out,
        "worst_week": {"season": worst["season"], "week": worst["week"], "left": worst["left"]} if worst else None,
        "moves_per_week": moves_per_week,
    }


def draft_flags(conn, member_id: str) -> dict:
    f = read_table(conn, "league_draft_flags")
    mine = f[f.member_id == member_id] if not f.empty else f
    n = int(len(mine))
    auto = int(mine.autodraft.sum()) if n else 0
    return {"n": n, "autodrafts": auto, "autodraft_rate": round(auto / n, 3) if n else None,
            "keepers": int(mine.keeper.sum()) if n else 0,
            "seasons": sorted(int(s) for s in mine.season.unique()) if n else []}


# The rules a defining line is chosen by, most extreme rank first. Each is
# (label, section, key, higher_is_it, minimum n). The member holding the
# league's top (or bottom) value on a fact with enough behind it gets that
# line; a member with no extreme gets their record.
DEFINING_RULES = [
    ("wins more than anyone", "finishes", "win_pct", True, 2),
    ("outperforms the draft board", "finishes", "outperformance", True, 2),
    ("trades more than anyone", "trades", "accepted", True, 2),
    ("works the wire harder than anyone", "waivers", "claims", True, 5),
    ("leaves the most points on the bench", "lineups", "bench_points_left", True, 5),
    ("sets the best lineups", "lineups", "hit_rate", True, 5),
    ("the luckiest record in the league", "luck", "luck", True, 5),
    ("the unluckiest record in the league", "luck", "luck", False, 5),
    ("lets the computer draft", "draft_flags", "autodraft_rate", True, 5),
]


def defining_line(member_id: str, facts_by_member: dict) -> str:
    mine = facts_by_member[member_id]
    for label, section, key, higher, minimum in DEFINING_RULES:
        values = {m: f[section].get(key) for m, f in facts_by_member.items()
                  if f[section].get("n", 0) >= minimum and f[section].get(key) is not None}
        if len(values) < 2 or member_id not in values:
            continue
        best = max(values, key=values.get) if higher else min(values, key=values.get)
        if best == member_id and values[best] != sorted(values.values())[len(values) // 2] or (
                best == member_id and len(set(values.values())) > 1):
            return label
    fin = mine["finishes"]
    if fin["n"]:
        last = fin["seasons"][-1]
        return f"{last['wins']}-{last['losses']} last season, avg finish {fin['avg_finish']}"
    return "first season in the league"


def _facts(conn, member_id: str) -> dict:
    return {"finishes": finishes(conn, member_id), "playoffs": playoffs(conn, member_id),
            "luck": luck(conn, member_id), "waivers": waivers(conn, member_id),
            "trades": trades(conn, member_id), "lineups": lineups(conn, member_id),
            "draft_flags": draft_flags(conn, member_id)}


def league_overview(conn) -> dict:
    m = members(conn)
    facts = {r.member_id: _facts(conn, r.member_id) for r in m.itertuples()}
    grid = []
    for r in m.itertuples():
        f = facts[r.member_id]["finishes"]
        grid.append({
            "member_id": r.member_id, "display_name": r.display_name,
            "seasons": len(r.seasons), "titles": len(f["titles"]),
            "avg_finish": f["avg_finish"], "win_pct": f["win_pct"],
            "defining_line": defining_line(r.member_id, facts),
        })
    grid.sort(key=lambda g: (g["avg_finish"] is None, g["avg_finish"] or 0))
    return {"seasons": seasons_strip(conn), "members": grid}


def profile(conn, member_id: str) -> dict | None:
    m = members(conn)
    hit = m[m.member_id == member_id]
    if hit.empty:
        return None
    me = hit.iloc[0]
    display = {r.member_id: r.display_name for r in m.itertuples()}
    h2h = [{"opponent": {"member_id": b, "display_name": display.get(b, b)}, **rec}
           for (a, b), rec in head_to_head(conn).items() if a == member_id]
    h2h.sort(key=lambda x: -x["games"])
    return {
        "member_id": member_id, "display_name": me.display_name,
        "first_name": me.first_name, "seasons": me.seasons,
        "head_to_head": h2h,
        **_facts(conn, member_id),
    }
```

Note for the implementer: the condition in `defining_line`'s loop is deliberately conservative -- the member must hold the extreme AND the values must not all be equal. Rewrite it as two plain lines (`if best != member_id: continue` / `if len(set(values.values())) < 2: continue` / `return label`) rather than keeping the compound expression.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_manager_profile.py -q -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scoring/manager_profile.py tests/test_manager_profile.py
git commit -m "Lineup discipline, autodraft rate, the league overview and the assembled profile"
```

---

### Task 6: The API and the ownership check

**Files:**
- Create: `api/league_access.py`, `api/league_history.py`
- Test: `tests/test_league_history_api.py`

**Interfaces:**
- Consumes: `api.drafts.session_for(request, store) -> Session | None` (`Session.swid`, `.espn_s2`, `.cookies` -- see note), `pipeline.espn_drafts.league_entries(swid, cookies, season, fetch) -> list[dict]` with `league_id` and `name` keys, `pipeline.espn_drafts.cookies_for(swid, espn_s2)`, `pipeline.leagues.league_db_path`, Task 2 `Progress` and `import_activity`, Task 5 `league_overview`/`profile`.
- Produces: `owned_league(request, store, league_id, fetch=None) -> Session` raising `HTTPException(403|400)`; `register_league_history_routes(app, store=None, fetch=None, runner=None)`; module registry `_JOBS: dict[league_id, Progress]`.

Note: `Session` in `api/drafts.py` has `swid` and `espn_s2`; cookies are built with `cookies_for`. Until task 7 wires `import_history`, the route calls `import_activity` directly against the league's file with seasons read from the current season's `mTeam` `status.previousSeasons` (plus the current season) -- the `seasons_from_status` helper below. Task 7 replaces that call with `import_history`.

- [ ] **Step 1: Write the failing API tests**

Create `tests/test_league_history_api.py`:

```python
"""The league-history routes: who may ask, what starts an import, what the
page reads while it runs and after."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import drafts as drafts_api
from api import league_history as lh
from pipeline import espn_drafts as drafts
from tests.test_league_activity import (DRAFT_PAYLOAD, MATCHUP_PAYLOAD, ROSTER_PAYLOAD,
                                        TEAM_PAYLOAD, TXN_PAYLOAD, SWID_A)

LEAGUE = "53929318"


def _entries_payload(*leagues):
    """What espn_drafts.league_entries reads: the account's team list."""
    return json.dumps({"preferences": [
        {"typeId": 9, "metaData": {"entry": {
            "gameId": 1, "seasonId": 2025, "entryId": 4, "groups": [{"groupId": int(lid), "groupName": name}],
            "entryMetadata": {"teamName": "Mine"}}}}
        for lid, name in leagues]})


def _fetch(entries, calls=None):
    """One fake for both the account list and the season/week reads."""
    served = {
        "view=mTeam": dict(TEAM_PAYLOAD, status=dict(TEAM_PAYLOAD["status"], finalScoringPeriod=1,
                                                     latestScoringPeriod=1, previousSeasons=[])),
        "view=mMatchupScore": MATCHUP_PAYLOAD, "view=mDraftDetail": DRAFT_PAYLOAD,
        "view=mTransactions2": TXN_PAYLOAD, "view=mRoster": ROSTER_PAYLOAD,
        "view=mSettings": {"settings": {"rosterSettings": {"lineupSlotCounts": {"0": 1, "20": 5}},
                                        "acquisitionSettings": {}}},
    }

    def fetch(url, cookies, headers=None):
        if calls is not None:
            calls.append(url)
        if "fan/" in url or "chui" in url or "/fans/" in url:
            return 200, entries
        for key, payload in served.items():
            if key in url:
                return 200, json.dumps(payload)
        return 404, "not found"
    return fetch


@pytest.fixture
def owner(monkeypatch, tmp_path):
    """This machine's saved login, owning league 53929318 as member A."""
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: (SWID_A, "s2"))
    monkeypatch.setattr(drafts_api, "_CACHE", {})
    monkeypatch.setattr("pipeline.leagues.LEAGUES_ROOT", str(tmp_path / "leagues"))
    lh._JOBS.clear()
    return tmp_path


def _client(fetch, runner=None):
    app = FastAPI()
    lh.register_league_history_routes(app, fetch=fetch, runner=runner)
    return TestClient(app, client=("127.0.0.1", 50000))


def _sync(fn):
    """A runner that runs the job on the request thread."""
    fn()


def test_a_visitor_with_no_session_is_refused(monkeypatch, owner):
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    resp = _client(_fetch(_entries_payload((LEAGUE, "Mine")))).get(f"/api/leagues/{LEAGUE}/history")
    assert resp.status_code == 403


def test_a_league_not_on_the_account_is_refused(owner):
    resp = _client(_fetch(_entries_payload(("999", "Other")))).get(f"/api/leagues/{LEAGUE}/history")
    assert resp.status_code == 403


def test_a_mock_league_is_refused(owner):
    resp = _client(_fetch(_entries_payload(("777", "Pro 8-Team Mock")))).post("/api/leagues/777/history")
    assert resp.status_code == 400


def test_first_visit_is_404_then_the_import_fills_it(owner):
    calls = []
    client = _client(_fetch(_entries_payload((LEAGUE, "Mine")), calls), runner=_sync)
    assert client.get(f"/api/leagues/{LEAGUE}/history").status_code == 404
    assert client.get(f"/api/leagues/{LEAGUE}/history/progress").json()["phase"] == "idle"
    started = client.post(f"/api/leagues/{LEAGUE}/history")
    assert started.status_code == 202 and started.json()["status"] == "running"
    progress = client.get(f"/api/leagues/{LEAGUE}/history/progress").json()
    assert progress["phase"] == "done" and progress["seasons_done"] == [2025]
    body = client.get(f"/api/leagues/{LEAGUE}/history").json()
    assert body["seasons"] and body["members"]
    assert body["members"][0]["display_name"] in {"alpha99", "bravo", "charlie"}
    # Every week read once, the season views once.
    assert sum("scoringPeriodId=" in u for u in calls) == 2
    prof = client.get(f"/api/leagues/{LEAGUE}/managers/{SWID_A}").json()
    assert prof["member_id"] == SWID_A and "finishes" in prof
    assert client.get(f"/api/leagues/{LEAGUE}/managers/{{NOBODY}}").status_code == 404


def test_a_second_post_while_running_does_not_start_another(owner):
    holds = []

    def runner(fn):
        holds.append(fn)      # never run: the job is "in flight"
    client = _client(_fetch(_entries_payload((LEAGUE, "Mine"))), runner=runner)
    assert client.post(f"/api/leagues/{LEAGUE}/history").status_code == 202
    assert client.post(f"/api/leagues/{LEAGUE}/history").status_code == 202
    assert len(holds) == 1


def test_a_fresh_league_answers_200_without_fetching(owner):
    calls = []
    client = _client(_fetch(_entries_payload((LEAGUE, "Mine")), calls), runner=_sync)
    client.post(f"/api/leagues/{LEAGUE}/history")
    before = len(calls)
    assert client.post(f"/api/leagues/{LEAGUE}/history").json()["status"] == "fresh"
    assert len(calls) == before + 1     # the account list, nothing else
```

Note for the implementer: the URL `league_entries` fetches is in `pipeline/espn_drafts.py` (`_entries_url` or the constant beside `league_entries`); match the fake's `"fan/" in url` test to it, and match `_entries_payload` to what `league_entries` parses (read that function first; the sketch above follows `tests/test_drafts_api.py::_entry`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_league_history_api.py -q -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.league_history'`

- [ ] **Step 3: Write the ownership check**

Create `api/league_access.py`:

```python
"""Who may read a league's history: an account that holds the league.

One check, shared by every per-league route (the history routes here and
the report routes when they land), so "does this session own this league"
is answered one way. The league list is the same one `GET /api/espn/drafts`
serves, read through the same `league_entries` call and the same cache.
"""
from __future__ import annotations

from fastapi import HTTPException, Request

from api.drafts import CURRENT_SEASON, Session, _cached, session_for
from pipeline import espn_drafts as drafts


def is_mock(name: str | None) -> bool:
    return "mock" in (name or "").lower()


def owned_league(request: Request, store, league_id: str, fetch=None) -> Session:
    """The session behind this request, if it holds `league_id`.

    403 with no session or a league the account does not hold. 400 for a
    mock room: a mock has no history worth a page.
    """
    session = session_for(request, store)
    if session is None:
        raise HTTPException(status_code=403, detail="Sign in to read a league.")
    try:
        rows = _cached((session.swid, "entries", str(CURRENT_SEASON)),
                       lambda: drafts.league_entries(session.swid, session.cookies,
                                                     str(CURRENT_SEASON), fetch=fetch))
    except drafts.SessionExpired:
        raise HTTPException(status_code=403, detail="ESPN no longer accepts this session.")
    mine = next((r for r in rows if str(r.get("league_id")) == str(league_id)), None)
    if mine is None:
        raise HTTPException(status_code=403, detail="That league is not on this account.")
    if is_mock(mine.get("name")):
        raise HTTPException(status_code=400, detail="A mock room has no history.")
    return session
```

Note for the implementer: `Session` in `api/drafts.py` exposes `swid` and `espn_s2`; if it has no `cookies` property, add one there (`return cookies_for(self.swid, self.espn_s2)`) -- a one-line, backwards-compatible addition.

- [ ] **Step 4: Write the routes**

Create `api/league_history.py`:

```python
"""A league's history, imported on first visit and served as manager profiles.

    POST /api/leagues/{id}/history            start the import (202), or 200 "fresh"
    GET  /api/leagues/{id}/history/progress   the stage list while it runs
    GET  /api/leagues/{id}/history            seasons strip + manager grid (404 until imported)
    GET  /api/leagues/{id}/managers/{member}  one manager's profile

Every route goes through `owned_league`. The import runs on a daemon
thread with the session's own ESPN cookies, one flight per league; the
page polls progress every couple of seconds, which for a job that changes
state every few seconds is the whole of what a stream would buy.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

from fastapi import HTTPException, Request

from api.league_access import owned_league
from pipeline import espn_drafts as drafts
from pipeline.db import DEFAULT_PATH, get_conn, read_table
from pipeline.league_activity import Progress, import_activity, season_url
from pipeline.leagues import league_db_path, provision_league
from scoring import manager_profile
from scoring.config import CURRENT_SEASON

# league_id -> Progress of the newest import in this process.
_JOBS: dict = {}
_LOCK = threading.Lock()
# The overview and profiles are cheap but not free (a few table scans);
# five minutes per league, dropped when an import finishes.
CACHE_SECONDS = 300.0
_ANSWERS: dict = {}
# Seconds between ESPN requests in the walk. Enough to be a polite reader,
# short enough that six seasons take a couple of minutes.
PAUSE_SECONDS = 0.15


def _leagues_root():
    from pipeline import leagues
    return leagues.LEAGUES_ROOT


def _path(league_id: str) -> str:
    return league_db_path(league_id, root=_leagues_root())


def _imported(league_id: str) -> bool:
    path = _path(league_id)
    if not Path(path).exists():
        return False
    conn = get_conn(path)
    try:
        return not read_table(conn, "league_members").empty
    finally:
        conn.close()


def seasons_from_status(fetch_json, league_id: str, current_season: int) -> list:
    """The seasons to walk, newest first: the current one and everything
    ESPN lists in `status.previousSeasons`."""
    team = fetch_json(season_url(league_id, current_season, "mTeam"))
    previous = [int(s) for s in ((team.get("status") or {}).get("previousSeasons") or [])]
    return [current_season] + sorted(previous, reverse=True)


def run_import(league_id: str, cookies: dict, progress: Progress, fetch=None,
               universal_path: str = DEFAULT_PATH) -> None:
    """The job. Until `import_history` carries the week walk (see the
    league-report branch), this walks activity on its own."""
    from pipeline.league_history import json_fetch
    raw = fetch if fetch is not None else drafts.http_fetch()
    fetch_json = json_fetch(raw, cookies)
    path = provision_league(league_id, universal_path, root=_leagues_root())
    conn = get_conn(path)
    try:
        seasons = seasons_from_status(fetch_json, league_id, CURRENT_SEASON)
        import_activity(conn, league_id, fetch_json, seasons, CURRENT_SEASON,
                        progress=progress, pause=PAUSE_SECONDS)
    finally:
        conn.close()
        with _LOCK:
            _ANSWERS.pop(league_id, None)


def _answer(league_id: str, key: str, build):
    now = time.monotonic()
    with _LOCK:
        hit = _ANSWERS.get(league_id, {}).get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    value = build()
    with _LOCK:
        _ANSWERS.setdefault(league_id, {})[key] = (now + CACHE_SECONDS, value)
    return value


def register_league_history_routes(app, store=None, fetch=None, runner=None) -> None:
    """`fetch` and `runner` are seams for tests: the ESPN fetch, and how the
    job is started (default: a daemon thread)."""

    def start(fn):
        if runner is not None:
            runner(fn)
        else:
            threading.Thread(target=fn, name="league-history", daemon=True).start()

    @app.post("/api/leagues/{league_id}/history")
    def start_history(league_id: str, request: Request):
        session = owned_league(request, store, league_id, fetch=fetch)
        with _LOCK:
            running = _JOBS.get(league_id)
            if running is not None and running.snapshot()["phase"] == "running":
                return _status(202, {"status": "running"})
        if _imported(league_id) and _fresh(league_id):
            return {"status": "fresh"}
        progress = Progress(league_id)
        progress.start([])
        with _LOCK:
            _JOBS[league_id] = progress
        cookies = drafts.cookies_for(session.swid, session.espn_s2)

        def job():
            try:
                run_import(league_id, cookies, progress, fetch=fetch)
            except Exception:      # noqa: BLE001 -- recorded on progress
                traceback.print_exc()
        start(job)
        return _status(202, {"status": "running"})

    @app.get("/api/leagues/{league_id}/history/progress")
    def history_progress(league_id: str, request: Request):
        owned_league(request, store, league_id, fetch=fetch)
        with _LOCK:
            progress = _JOBS.get(league_id)
        return progress.snapshot() if progress is not None else {"phase": "idle"}

    @app.get("/api/leagues/{league_id}/history")
    def history(league_id: str, request: Request):
        owned_league(request, store, league_id, fetch=fetch)
        if not _imported(league_id):
            raise HTTPException(status_code=404, detail="No history imported yet.")

        def build():
            conn = get_conn(_path(league_id))
            try:
                overview = manager_profile.league_overview(conn)
                raw = read_table(conn, "league_raw")
            finally:
                conn.close()
            newest = None if raw.empty else str(raw.fetched_at.max())
            return dict(overview, league_id=league_id, imported_at=newest)
        return _answer(league_id, "overview", build)

    @app.get("/api/leagues/{league_id}/managers/{member_id}")
    def manager(league_id: str, member_id: str, request: Request):
        owned_league(request, store, league_id, fetch=fetch)
        if not _imported(league_id):
            raise HTTPException(status_code=404, detail="No history imported yet.")

        def build():
            conn = get_conn(_path(league_id))
            try:
                return manager_profile.profile(conn, member_id)
            finally:
                conn.close()
        body = _answer(league_id, f"manager:{member_id}", build)
        if body is None:
            raise HTTPException(status_code=404, detail="No such manager in this league.")
        return body


def _fresh(league_id: str) -> bool:
    """Newest raw answer for the current season within a day."""
    from pipeline.league_activity import CURRENT_MAX_AGE_HOURS, _raw, _stale
    conn = get_conn(_path(league_id))
    try:
        return not _stale(_raw(conn), CURRENT_SEASON, CURRENT_MAX_AGE_HOURS)
    finally:
        conn.close()


def _status(code: int, body: dict):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=code, content=body)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_league_history_api.py tests/test_drafts_api.py -q -p no:cacheprovider`
Expected: all PASS. If `test_first_visit` fails with the fake's account list, read `league_entries` in `pipeline/espn_drafts.py` and adjust `_entries_payload` and the fake's URL match to it -- the test's shape follows that function, not the other way round.

- [ ] **Step 6: Commit**

```bash
git add api/league_access.py api/league_history.py tests/test_league_history_api.py api/drafts.py
git commit -m "Routes that import a league's history on first visit and serve the manager profiles"
```

---

### Task 7: Wire into the league-report branch (after it merges)

**Files:**
- Modify: `pipeline/league_history.py` (`import_history`)
- Modify: `api/league_history.py` (`run_import`)
- Modify: `api/main.py` (between `register_seo_routes` and `register_spa`)
- Modify: `scoring/manager_profile.py` (`profile`: draft chapter)
- Modify: `pipeline/db.py`, `tests/test_leagues.py` (merge the two branches' additions to `LEAGUE_TABLES`)
- Test: `tests/test_league_history.py` (branch's file, append), `tests/test_league_history_api.py`

**Interfaces:**
- Consumes: branch's `import_history(...) -> summary` with `summary["seasons"]`, `json_fetch`; branch's `scoring.league_report.team_profiles(conn) -> list[dict]` keyed by `manager` (display name).
- Produces: `import_history` returns the summary with an `activity` key (`import_activity`'s dict); `profile()["draft"]` is the matching `team_profiles` record or `None`.

- [ ] **Step 1: Rebase and resolve `LEAGUE_TABLES`**

```bash
git fetch origin && git rebase origin/main
```

Resolve `pipeline/db.py` so `LEAGUE_TABLES` holds the union: the branch's `league_standings`, `league_reports` and this plan's six. Update the exact-set assertion in `tests/test_leagues.py` to the same union. Run `.venv/bin/python -m pytest tests/test_leagues.py -q -p no:cacheprovider` -- PASS.

- [ ] **Step 2: Write the failing test for the combined import**

Append to `tests/test_league_history.py` (the branch's file; reuse its `_raw_fetch` and `SEASON_PAYLOAD`):

```python
def test_import_history_also_walks_the_seasons_activity(tmp_path, monkeypatch):
    from pipeline import league_activity as act
    from pipeline.db import read_table
    from pipeline.leagues import league_db_path
    seen = []

    def fake_activity(conn, league_id, fetch_json, seasons, current_season, **kw):
        seen.append(list(seasons))
        return {"seasons": list(seasons), "requests": 0, "weeks": {}}
    monkeypatch.setattr(act, "import_activity", fake_activity)
    from pipeline import league_history as lh
    monkeypatch.setattr(lh, "import_activity", fake_activity)
    summary = lh.import_history("4242", {"SWID": "{X}", "espn_s2": "s"},
                                fetch=_raw_fetch([2025, 2024]), current_season=2025,
                                universal_path=str(tmp_path / "u.duckdb"),
                                root=str(tmp_path / "leagues"), adp_fetch=lambda s: None)
    assert seen == [summary["seasons"]]
    assert summary["activity"]["seasons"] == summary["seasons"]
```

Run: `.venv/bin/python -m pytest tests/test_league_history.py -q -p no:cacheprovider -k activity` -- FAIL (no `activity` key).

- [ ] **Step 3: Call the walk from `import_history`**

In `pipeline/league_history.py`, import `from pipeline.league_activity import import_activity` and, inside `import_history` after the `historic_adp` write and before `record_freshness(conn, "league", ...)`:

```python
        # Everything a season does after the draft -- see league_activity.
        # Same connection, same fetch, same walk from the caller's point of
        # view. `progress` is the page's stage list when a visit started
        # this; None from the connect path, which nobody is watching.
        summary["activity"] = import_activity(
            conn, league_id, json_fetch(raw, cookies), summary["seasons"],
            current_season, progress=progress, pause=pause)
```

Add `progress=None, pause=0.0` to `import_history`'s signature. Run the test -- PASS.

- [ ] **Step 4: `run_import` uses `import_history`**

Replace the body of `run_import` in `api/league_history.py`:

```python
def run_import(league_id: str, cookies: dict, progress: Progress, fetch=None,
               universal_path: str = DEFAULT_PATH) -> None:
    from pipeline.league_history import import_history
    try:
        import_history(league_id, cookies, fetch=fetch, universal_path=universal_path,
                       root=_leagues_root(), progress=progress, pause=PAUSE_SECONDS)
    except Exception as exc:      # noqa: BLE001
        if progress.snapshot()["phase"] != "failed":
            progress.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        with _LOCK:
            _ANSWERS.pop(league_id, None)
```

Delete `seasons_from_status`. Update `tests/test_league_history_api.py`'s fake so the branch's `import_seasons` is satisfied too (it fetches the season payload with `view=mDraftDetail&view=mTeam&view=mSettings` and the players directory `/players?`; serve the merged `TEAM_PAYLOAD | DRAFT_PAYLOAD | settings` for the former and `[]` for the latter, following `tests/test_league_history.py::_raw_fetch`). Run: `.venv/bin/python -m pytest tests/test_league_history_api.py -q -p no:cacheprovider` -- PASS.

- [ ] **Step 5: Register the routes**

In `api/main.py`, after `register_seo_routes(app, conn)` and before `register_spa(app)`:

```python
    # A league's history and manager profiles, imported on first visit
    # with the visitor's own ESPN session (api/league_history.py). Before
    # the SPA catch-all, like everything under /api.
    from api.league_history import register_league_history_routes
    register_league_history_routes(app)
```

- [ ] **Step 6: The draft chapter from `team_profiles`**

In `scoring/manager_profile.py`, extend `profile`:

```python
def _draft_chapter(conn, display_name: str):
    """The league-report branch's draft habits for this manager, matched
    on the display name both tables carry. None when the branch's tables
    are not there yet or the manager has no graded picks."""
    try:
        from scoring.league_report import team_profiles
    except ImportError:
        return None
    try:
        return next((p for p in team_profiles(conn) if p.get("manager") == display_name), None)
    except Exception:      # noqa: BLE001 -- no picks, no ADP: no chapter
        return None
```

and add `"draft": _draft_chapter(conn, me.display_name),` to the returned dict. Append to `tests/test_manager_profile.py`:

```python
def test_the_draft_chapter_is_the_report_branch_s_record_or_none(lineups_league, monkeypatch):
    import scoring.league_report as lr
    monkeypatch.setattr(lr, "team_profiles", lambda conn: [{"manager": "a", "mean_value": 2.5}])
    assert mp.profile(lineups_league, A)["draft"] == {"manager": "a", "mean_value": 2.5}
    monkeypatch.setattr(lr, "team_profiles", lambda conn: [])
    assert mp.profile(lineups_league, A)["draft"] is None
```

Run: `.venv/bin/python -m pytest tests/test_manager_profile.py -q -p no:cacheprovider` -- PASS.

- [ ] **Step 7: Run the whole Python suite and commit**

Run: `.venv/bin/python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_live_api.py`
Expected: PASS except the two pre-existing failures noted in this session's history (`test_api.py::test_a_connected_session_does_not_rebuild_the_board_on_every_request`, and the flaky `test_demo_live.py::test_a_stale_answer_is_served_rather_than_rebuilt`).

```bash
git add pipeline/league_history.py api/league_history.py api/main.py scoring/manager_profile.py pipeline/db.py tests
git commit -m "The history import walks a season's activity too, the routes are registered, and a profile carries its draft chapter"
```

---

### Task 8: The web client and the league page

**Files:**
- Modify: `web/src/api.ts` (append)
- Modify: `web/src/pages/LeaguePage.tsx`
- Modify: `web/src/pages/league.css` (append)

**Interfaces:**
- Consumes: the four endpoints of Task 6.
- Produces: `startLeagueHistory(id)`, `fetchLeagueHistoryProgress(id)`, `fetchLeagueHistory(id)`, `fetchManagerProfile(id, memberId)`; interfaces `HistoryProgress`, `LeagueHistory`, `ManagerProfile`.

- [ ] **Step 1: Append the client**

Append to `web/src/api.ts`:

```ts
/** -- league history (api/league_history.py) ------------------------------ */

export interface HistoryStage {
  season: number
  label: string
  done: boolean
  week: number
  weeks: number | null
}

export interface HistoryProgress {
  phase: 'idle' | 'running' | 'done' | 'failed'
  league_id?: string
  started_at?: number | null
  error?: string | null
  seasons_done?: number[]
  stages?: HistoryStage[]
}

export interface Named {
  member_id: string | null
  display_name: string | null
}

export interface SeasonSummary {
  season: number
  teams: number
  complete: boolean
  champion: Named | null
  runner_up: Named | null
  last: Named | null
  top_scorer: Named | null
}

export interface MemberCard {
  member_id: string
  display_name: string
  seasons: number
  titles: number
  avg_finish: number | null
  win_pct: number | null
  defining_line: string
}

export interface LeagueHistory {
  league_id: string
  imported_at: string | null
  seasons: SeasonSummary[]
  members: MemberCard[]
}

export interface ManagerProfile {
  member_id: string
  display_name: string
  first_name: string | null
  seasons: number[]
  finishes: {
    n: number
    seasons: { season: number; team_name: string | null; wins: number | null; losses: number | null;
               ties: number | null; points_for: number | null; points_against: number | null;
               playoff_seed: number | null; final_rank: number | null; draft_day_rank: number | null;
               outperformance: number | null }[]
    titles: number[]; avg_finish: number | null; win_pct: number | null
    points_per_season: number | null; outperformance: number | null
  }
  playoffs: { n: number; appearances: number[]; bracket_wins: number; bracket_losses: number;
              consolation: { wins: number; losses: number }; titles: number[]; last_place: number[] }
  head_to_head: { opponent: Named; games: number; wins: number; losses: number; ties: number;
                  margin: number; playoff_games: number; n: number }[]
  luck: { n: number; luck: number | null;
          seasons: { season: number; games: number; actual_wins: number; expected_wins: number; luck: number }[] }
  waivers: { n: number; claims: number; won: number; lost: number; canceled: number; win_rate: number | null;
             free_agent_adds: number; drops: number; adds_by_weekday: Record<string, number>;
             busiest_week: { season: number; week: number; moves: number } | null; seasons: number[];
             bids?: { n: number; mean: number; max: number; spent: number } }
  trades: { n: number; proposed: number; received: number; accepted: number; declined_by_me: number;
            declined_by_them: number; vetoed: number; partners: { member_id: string; display_name: string; trades: number }[];
            balance: number | null;
            ledger: { season: number; week: number; with: string; with_name: string;
                      sent: { player_id: number; name: string }[]; received: { player_id: number; name: string }[];
                      balance: number }[] }
  lineups: { n: number; weeks: { season: number; week: number; started: number; optimal: number; left: number }[];
             bench_points_left: number | null; hit_rate: number | null; started_out: number;
             worst_week: { season: number; week: number; left: number } | null; moves_per_week: number | null }
  draft_flags: { n: number; autodrafts: number; autodraft_rate: number | null; keepers: number; seasons: number[] }
  /** The league-report branch's draft habits, when its tables exist. */
  draft: Record<string, unknown> | null
}

export async function startLeagueHistory(leagueId: string): Promise<'running' | 'fresh'> {
  const res = await fetch(`/api/leagues/${encodeURIComponent(leagueId)}/history`, { method: 'POST' })
  if (!res.ok) throw new Error(await detailText(res))
  return (await res.json()).status
}

export async function fetchLeagueHistoryProgress(leagueId: string): Promise<HistoryProgress> {
  const res = await fetch(`/api/leagues/${encodeURIComponent(leagueId)}/history/progress`)
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}

/** Null on 404: nothing imported yet, which is a state, not an error. */
export async function fetchLeagueHistory(leagueId: string): Promise<LeagueHistory | null> {
  const res = await fetch(`/api/leagues/${encodeURIComponent(leagueId)}/history`)
  if (res.status === 404) return null
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}

export async function fetchManagerProfile(leagueId: string, memberId: string): Promise<ManagerProfile> {
  const res = await fetch(
    `/api/leagues/${encodeURIComponent(leagueId)}/managers/${encodeURIComponent(memberId)}`)
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}
```

- [ ] **Step 2: The league page's history section**

In `web/src/pages/LeaguePage.tsx`, replace the `lg-soon` placeholder section with a `<LeagueHistorySection leagueId={league.league_id} me={...} />` and add the component in the same file:

```tsx
import { fetchLeagueHistory, fetchLeagueHistoryProgress, startLeagueHistory,
         type HistoryProgress, type LeagueHistory } from '../api'

// THE LEAGUE'S HISTORY, IMPORTED ON THE FIRST VISIT. The page asks for the
// overview; a 404 means nobody has read this league yet, so it starts the
// import with the visitor's own session and draws the stage list while
// ESPN is read -- one row per season, the current one ticking week by
// week -- then the overview the moment the job says done.
const POLL_MS = 2000

function LeagueHistorySection({ leagueId, me }: { leagueId: string; me: string | null }) {
  const [history, setHistory] = useState<LeagueHistory | null | undefined>(undefined)
  const [progress, setProgress] = useState<HistoryProgress | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const body = await fetchLeagueHistory(leagueId)
      setHistory(body)
      return body
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setHistory(null)
      return null
    }
  }, [leagueId])

  const start = useCallback(async () => {
    setError(null)
    try {
      const status = await startLeagueHistory(leagueId)
      if (status === 'fresh') { await load(); return }
      setProgress({ phase: 'running', stages: [] })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [leagueId, load])

  useEffect(() => {
    let cancelled = false
    load().then((body) => { if (!cancelled && body === null) start() })
    return () => { cancelled = true }
  }, [load, start])

  // Poll while running; on done, fetch the overview and stop.
  useEffect(() => {
    if (progress?.phase !== 'running') return
    let cancelled = false
    const id = setInterval(async () => {
      try {
        const next = await fetchLeagueHistoryProgress(leagueId)
        if (cancelled) return
        setProgress(next)
        if (next.phase === 'done') { clearInterval(id); await load() }
        if (next.phase === 'failed') clearInterval(id)
      } catch { /* the next tick asks again */ }
    }, POLL_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [progress?.phase, leagueId, load])

  if (history === undefined && progress === null && error === null) {
    return <p className="mk-loading">Reading the league…</p>
  }
  if (history) return <History history={history} leagueId={leagueId} me={me} />
  if (progress?.phase === 'failed' || error) {
    return (
      <section className="lg-import">
        <h2 className="mk-h2">The history could not be read.</h2>
        <p className="mk-sub">{progress?.error ?? error}</p>
        <button type="button" className="db-join" onClick={start}>Try again</button>
      </section>
    )
  }
  return (
    <section className="lg-import" aria-live="polite">
      <h2 className="mk-h2">Reading this league's history from ESPN…</h2>
      <p className="mk-sub">Every season, every week: standings, matchups, waivers, trades, lineups. A minute or two.</p>
      <ul className="lg-stages mono">
        {(progress?.stages ?? []).map((s) => (
          <li key={s.season} className={`lg-stage${s.done ? ' is-done' : s.week > 0 ? ' is-live' : ''}`}>
            <span className="lg-stage-dot" aria-hidden="true" />{s.label}
          </li>
        ))}
      </ul>
    </section>
  )
}

function History({ history, leagueId, me }: { history: LeagueHistory; leagueId: string; me: string | null }) {
  return (
    <>
      <section className="lg-seasons">
        <h2 className="mk-h2">Seasons</h2>
        <ol className="lg-strip">
          {history.seasons.map((s) => (
            <li className={`lg-season${s.complete ? '' : ' is-open'}`} key={s.season}>
              <span className="mono lg-season-year">{s.season}</span>
              <Row label="Champion" who={s.champion} leagueId={leagueId} crown />
              <Row label="Last" who={s.last} leagueId={leagueId} />
              <Row label="Top scorer" who={s.top_scorer} leagueId={leagueId} />
            </li>
          ))}
        </ol>
      </section>
      <section className="lg-managers">
        <h2 className="mk-h2">Managers</h2>
        <ul className="lg-grid">
          {history.members.map((m) => (
            <li className={`lg-manager${m.member_id === me ? ' is-me' : ''}`} key={m.member_id}>
              <Link to={`/league/${encodeURIComponent(leagueId)}/manager/${encodeURIComponent(m.member_id)}`} className="lg-manager-name">
                {m.display_name}
              </Link>
              <p className="lg-manager-line">{m.defining_line}</p>
              <p className="mono lg-manager-figs">
                {m.seasons} {m.seasons === 1 ? 'season' : 'seasons'}
                {m.titles > 0 && ` · ${m.titles} ${m.titles === 1 ? 'title' : 'titles'}`}
                {m.avg_finish !== null && ` · avg finish ${m.avg_finish}`}
                {m.win_pct !== null && ` · ${Math.round(m.win_pct * 100)}% wins`}
              </p>
            </li>
          ))}
        </ul>
      </section>
    </>
  )
}

function Row({ label, who, leagueId, crown = false }: {
  label: string; who: { member_id: string | null; display_name: string | null } | null
  leagueId: string; crown?: boolean
}) {
  if (!who || !who.member_id) return <span className="lg-season-row is-empty">{label}: —</span>
  return (
    <span className="lg-season-row">
      <span className="lg-season-label">{crown ? '♛ ' : ''}{label}</span>
      <Link to={`/league/${encodeURIComponent(leagueId)}/manager/${encodeURIComponent(who.member_id)}`}>
        {who.display_name}
      </Link>
    </span>
  )
}
```

`me` is the visiting member's SWID; the league list does not carry it, so pass `null` for now and mark nothing (the highlight lands when `/api/espn/drafts` adds `member_id`, out of this plan's scope).

- [ ] **Step 3: Styles**

Append to `web/src/pages/league.css`:

```css
/* -- history: importing ------------------------------------------------ */
.lg-import { margin-top: 28px; }
.lg-stages { list-style: none; margin: 14px 0 0; padding: 0; display: grid; gap: 6px; font-size: var(--text-sm); color: var(--text-3); }
.lg-stage { display: flex; align-items: center; gap: 10px; }
.lg-stage-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--border); }
.lg-stage.is-live { color: var(--text-1); }
.lg-stage.is-live .lg-stage-dot { background: var(--accent); animation: db-dot 1.8s ease-in-out infinite; }
.lg-stage.is-done { color: var(--text-2); }
.lg-stage.is-done .lg-stage-dot { background: var(--ok); }

/* -- history: seasons strip -------------------------------------------- */
.lg-seasons { margin-top: 28px; }
.lg-strip { list-style: none; margin: 14px 0 0; padding: 0; display: grid; grid-auto-flow: column; grid-auto-columns: minmax(180px, 1fr); gap: 12px; overflow-x: auto; }
.lg-season { padding: 14px 16px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-1); display: grid; gap: 6px; }
.lg-season.is-open { border-style: dashed; }
.lg-season-year { font-size: var(--text-md); font-weight: 700; color: var(--text-1); }
.lg-season-row { display: flex; justify-content: space-between; gap: 8px; font-size: var(--text-sm); }
.lg-season-row.is-empty { color: var(--text-3); }
.lg-season-label { color: var(--text-3); }
.lg-season-row a { color: var(--text-1); text-decoration: none; }
.lg-season-row a:hover { color: var(--accent); }

/* -- history: manager grid --------------------------------------------- */
.lg-managers { margin-top: 28px; }
.lg-grid { list-style: none; margin: 14px 0 0; padding: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
.lg-manager { padding: 16px 18px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-1); display: grid; gap: 6px; }
.lg-manager.is-me { border-color: var(--accent-border); }
.lg-manager-name { font-size: var(--text-md); font-weight: 700; color: var(--text-1); text-decoration: none; }
.lg-manager-name:hover { color: var(--accent); }
.lg-manager-line { margin: 0; color: var(--text-2); font-size: var(--text-sm); }
.lg-manager-figs { margin: 0; color: var(--text-3); font-size: var(--text-xs); }
```

- [ ] **Step 4: Typecheck and lint**

Run: `cd web && npx tsc -b && npm run -s lint`
Expected: `tsc` clean; lint reports 0 errors (the repo's existing warnings are fine).

- [ ] **Step 5: Commit**

```bash
git add web/src/api.ts web/src/pages/LeaguePage.tsx web/src/pages/league.css
git commit -m "The league page reads its history on first visit and shows the seasons and the managers"
```

---

### Task 9: The manager page

**Files:**
- Create: `web/src/pages/ManagerPage.tsx`
- Modify: `web/src/App.tsx` (route), `web/src/pages/league.css` (append)

**Interfaces:**
- Consumes: `fetchManagerProfile`, `ManagerProfile` (Task 8); `PlayerOverlay` + `OverlayTarget` from `web/src/components/draft/PlayerOverlay.tsx` for player names.

- [ ] **Step 1: The route**

In `web/src/App.tsx`, after the `/league/:leagueId` route:

```tsx
        {/* One manager in that league, over every season. */}
        <Route path="/league/:leagueId/manager/:memberId" element={<ManagerPage />} />
```

and `import ManagerPage from './pages/ManagerPage'`.

- [ ] **Step 2: The page**

Create `web/src/pages/ManagerPage.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { fetchManagerProfile, type ManagerProfile } from '../api'
import { Logo } from '../components/Logo'
import PlayerOverlay, { type OverlayTarget } from '../components/draft/PlayerOverlay'
import { useDocumentMeta } from '../lib/documentMeta'
import '../landing.css'
import '../market.css'
import './league.css'

// ONE MANAGER, OVER EVERY SEASON. /league/:leagueId/manager/:memberId.
//
// The sections mirror the facts document (scoring/manager_profile.py):
// finishes, playoffs, head-to-head, luck, draft, waivers, trades, lineups.
// Each carries the number of observations it rests on, and a figure that
// rests on fewer than five is shown with that number and no adjective --
// "63% waiver win rate (n=3)" is a fact; "wins the wire" off three claims
// is not.
//
// Player names open the same profile popup the draft room and the archive
// use (PlayerOverlay -> PlayerProfile); this page describes no player
// itself.
const SMALL = 5

function pct(v: number | null): string { return v === null ? '—' : `${Math.round(v * 100)}%` }
function num(v: number | null, digits = 1): string { return v === null ? '—' : v.toFixed(digits) }
function signed(v: number | null): string { return v === null ? '—' : (v > 0 ? `+${v}` : `${v}`) }

function Fig({ label, value, n }: { label: string; value: string; n?: number }) {
  return (
    <div className="lg-fig">
      <span className="lg-fig-value mono">{value}</span>
      <span className="lg-fig-label">{label}{n !== undefined && n < SMALL && <span className="lg-n"> n={n}</span>}</span>
    </div>
  )
}

export default function ManagerPage() {
  const { leagueId = '', memberId = '' } = useParams()
  const [p, setP] = useState<ManagerProfile | null | undefined>(undefined)
  const [error, setError] = useState<string | null>(null)
  const [target, setTarget] = useState<OverlayTarget | null>(null)

  useDocumentMeta({ title: `${p?.display_name ?? 'Manager'} – ESPN Draft Assist`, noindex: true })

  useEffect(() => {
    let cancelled = false
    fetchManagerProfile(leagueId, memberId)
      .then((body) => { if (!cancelled) setP(body) })
      .catch((e) => { if (!cancelled) { setError(e instanceof Error ? e.message : String(e)); setP(null) } })
    return () => { cancelled = true }
  }, [leagueId, memberId])

  const openPlayer = (id: number) => setTarget({ playerId: String(id), seed: null })

  return (
    <div className="mk-page">
      <header className="draft-topbar">
        <Link to="/" className="draft-topbar-title"><Logo /> ESPN Draft Assist</Link>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <nav className="draft-topbar-tabs" aria-label="Views">
          <Link className="draft-tab is-active" to="/" aria-current="page">Home</Link>
          <Link className="draft-tab" to="/archive">Data</Link>
          <Link className="draft-tab" to="/live">Live</Link>
        </nav>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <span className="draft-topbar-league">{p ? `${p.seasons.length} seasons` : 'Manager'}</span>
        <span className="draft-topbar-spacer" />
        <Link className="draft-topbar-link" to={`/league/${encodeURIComponent(leagueId)}`}>← Back to the league</Link>
      </header>

      <main className="mk lg">
        {p === undefined ? (
          <p className="mk-loading">Reading the manager…</p>
        ) : p === null ? (
          <section className="mk-panel mk-gate">
            <h2 className="mk-h2">No profile here.</h2>
            <p className="mk-gate-copy">{error ?? 'That manager is not in this league.'}</p>
          </section>
        ) : (
          <>
            <header className="mk-mast">
              <div className="mk-mast-say">
                <h1 className="mk-title">{p.display_name}</h1>
                <p className="mk-lede">
                  {p.seasons.length} {p.seasons.length === 1 ? 'season' : 'seasons'} · {p.seasons[0]}–{p.seasons[p.seasons.length - 1]}
                  {p.finishes.titles.length > 0 && ` · ${p.finishes.titles.length} ${p.finishes.titles.length === 1 ? 'title' : 'titles'} (${p.finishes.titles.join(', ')})`}
                </p>
              </div>
              <div className="lg-figs">
                <Fig label="win rate" value={pct(p.finishes.win_pct)} n={p.finishes.n} />
                <Fig label="avg finish" value={num(p.finishes.avg_finish, 1)} n={p.finishes.n} />
                <Fig label="vs draft-day projection" value={signed(p.finishes.outperformance)} n={p.finishes.n} />
              </div>
            </header>

            <section className="lg-sec">
              <h2 className="mk-h2">Finishes</h2>
              <table className="lg-table mono">
                <thead><tr><th>Season</th><th>Team</th><th>Record</th><th>Points</th><th>Seed</th><th>Finish</th><th>Projected</th></tr></thead>
                <tbody>
                  {p.finishes.seasons.map((s) => (
                    <tr key={s.season} className={s.final_rank === 1 ? 'is-title' : ''}>
                      <td>{s.season}</td><td className="lg-td-text">{s.team_name}</td>
                      <td>{s.wins ?? '—'}-{s.losses ?? '—'}{s.ties ? `-${s.ties}` : ''}</td>
                      <td>{num(s.points_for, 0)}</td><td>{s.playoff_seed ?? '—'}</td>
                      <td>{s.final_rank ?? '—'}</td><td>{s.draft_day_rank ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Playoffs</h2>
              <div className="lg-figs">
                <Fig label="appearances" value={String(p.playoffs.appearances.length)} />
                <Fig label="bracket record" value={`${p.playoffs.bracket_wins}-${p.playoffs.bracket_losses}`} n={p.playoffs.n} />
                <Fig label="titles" value={String(p.playoffs.titles.length)} />
                <Fig label="last place" value={p.playoffs.last_place.length ? p.playoffs.last_place.join(', ') : 'never'} />
              </div>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Head-to-head</h2>
              <table className="lg-table mono">
                <thead><tr><th>Opponent</th><th>Games</th><th>Record</th><th>Margin</th><th>Playoff</th></tr></thead>
                <tbody>
                  {p.head_to_head.map((h) => (
                    <tr key={h.opponent.member_id ?? ''}>
                      <td className="lg-td-text">
                        <Link to={`/league/${encodeURIComponent(leagueId)}/manager/${encodeURIComponent(h.opponent.member_id ?? '')}`}>{h.opponent.display_name}</Link>
                      </td>
                      <td>{h.games}</td><td>{h.wins}-{h.losses}{h.ties ? `-${h.ties}` : ''}</td>
                      <td className={h.margin > 0 ? 'is-up' : h.margin < 0 ? 'is-down' : ''}>{signed(Math.round(h.margin))}</td>
                      <td>{h.playoff_games || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Luck</h2>
              <p className="mk-sub">Actual wins against the all-play expectation: your score against every team's, every week.</p>
              <div className="lg-figs">
                <Fig label="career luck (wins)" value={signed(p.luck.luck)} n={p.luck.n} />
                {p.luck.seasons.map((s) => (
                  <Fig key={s.season} label={`${s.season}: ${s.actual_wins} won, ${s.expected_wins} expected`} value={signed(s.luck)} />
                ))}
              </div>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Draft</h2>
              <div className="lg-figs">
                <Fig label="autodraft rate" value={pct(p.draft_flags.autodraft_rate)} n={p.draft_flags.n} />
                {p.draft && typeof p.draft.mean_value === 'number' && (
                  <Fig label="value per pick vs ADP" value={signed(p.draft.mean_value as number)} />
                )}
                {p.draft && typeof p.draft.steal_rate === 'number' && (
                  <Fig label="steal rate" value={pct(p.draft.steal_rate as number)} />
                )}
                {p.draft && typeof p.draft.reach_rate === 'number' && (
                  <Fig label="reach rate" value={pct(p.draft.reach_rate as number)} />
                )}
              </div>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Waivers</h2>
              <div className="lg-figs">
                <Fig label="claims" value={String(p.waivers.claims)} />
                <Fig label="won" value={pct(p.waivers.win_rate)} n={p.waivers.won + p.waivers.lost} />
                <Fig label="free-agent adds" value={String(p.waivers.free_agent_adds)} />
                <Fig label="drops" value={String(p.waivers.drops)} />
                {p.waivers.bids && <Fig label="avg bid" value={`$${p.waivers.bids.mean}`} n={p.waivers.bids.n} />}
                {p.waivers.busiest_week && (
                  <Fig label="busiest week" value={`${p.waivers.busiest_week.season} wk ${p.waivers.busiest_week.week}`} />
                )}
              </div>
              <p className="mk-sub mono">
                {Object.entries(p.waivers.adds_by_weekday).map(([d, n]) => `${d} ${n}`).join(' · ')}
              </p>
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Trades</h2>
              <div className="lg-figs">
                <Fig label="proposed" value={String(p.trades.proposed)} />
                <Fig label="received" value={String(p.trades.received)} />
                <Fig label="done" value={String(p.trades.accepted)} />
                <Fig label="declined (by me / by them)" value={`${p.trades.declined_by_me} / ${p.trades.declined_by_them}`} />
                <Fig label="rest-of-season balance" value={signed(p.trades.balance)} n={p.trades.n} />
              </div>
              {p.trades.ledger.length > 0 && (
                <ul className="lg-ledger">
                  {p.trades.ledger.map((t, i) => (
                    <li key={i} className="lg-trade">
                      <span className="mono lg-trade-when">{t.season} wk {t.week}</span>
                      <span className="lg-trade-body">
                        sent {t.sent.map((x) => <button key={x.player_id} type="button" className="lg-player" onClick={() => openPlayer(x.player_id)}>{x.name}</button>)}
                        {' '}to <Link to={`/league/${encodeURIComponent(leagueId)}/manager/${encodeURIComponent(t.with)}`}>{t.with_name}</Link>
                        {' '}for {t.received.map((x) => <button key={x.player_id} type="button" className="lg-player" onClick={() => openPlayer(x.player_id)}>{x.name}</button>)}
                      </span>
                      <span className={`mono lg-trade-balance${t.balance > 0 ? ' is-up' : t.balance < 0 ? ' is-down' : ''}`}>{signed(t.balance)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className="lg-sec">
              <h2 className="mk-h2">Lineups</h2>
              <div className="lg-figs">
                <Fig label="bench points left / week" value={num(p.lineups.bench_points_left)} n={p.lineups.n} />
                <Fig label="optimal lineup rate" value={pct(p.lineups.hit_rate)} n={p.lineups.n} />
                <Fig label="started an OUT player" value={String(p.lineups.started_out)} />
                <Fig label="lineup moves / week" value={num(p.lineups.moves_per_week)} />
                {p.lineups.worst_week && (
                  <Fig label="worst week" value={`${p.lineups.worst_week.season} wk ${p.lineups.worst_week.week}: ${p.lineups.worst_week.left} left`} />
                )}
              </div>
            </section>
          </>
        )}
      </main>
      {target && <PlayerOverlay target={target} onClose={() => setTarget(null)} />}
    </div>
  )
}
```

Note for the implementer: check `PlayerOverlayProps` in `web/src/components/draft/PlayerOverlay.tsx` for required props (`onSelectPlayer` may be required; pass `() => {}` if so) and for `OverlayTarget`'s `playerId` type (string in the archive's use; keep that).

- [ ] **Step 3: Styles**

Append to `web/src/pages/league.css`:

```css
/* -- manager page -------------------------------------------------------- */
.lg-sec { margin-top: 28px; }
.lg-figs { display: flex; flex-wrap: wrap; gap: 18px 28px; margin-top: 12px; }
.lg-fig { display: grid; gap: 2px; }
.lg-fig-value { font-size: 22px; font-weight: 700; color: var(--text-1); }
.lg-fig-label { font-size: var(--text-xs); color: var(--text-3); text-transform: uppercase; letter-spacing: 0.06em; }
.lg-n { color: var(--text-3); text-transform: none; letter-spacing: 0; }
.lg-table { margin-top: 12px; border-collapse: collapse; font-size: var(--text-sm); }
.lg-table th, .lg-table td { padding: 6px 12px 6px 0; text-align: right; border-bottom: 1px solid var(--border); }
.lg-table th { color: var(--text-3); font-weight: 600; font-size: var(--text-xs); text-transform: uppercase; letter-spacing: 0.06em; }
.lg-table .lg-td-text { text-align: left; font-family: var(--font-sans); }
.lg-table .lg-td-text a { color: var(--text-1); text-decoration: none; }
.lg-table .lg-td-text a:hover { color: var(--accent); }
.lg-table tr.is-title td:first-child::after { content: ' ♛'; color: var(--accent); }
.is-up { color: var(--ok); }
.is-down { color: var(--danger, #e5484d); }
.lg-ledger { list-style: none; margin: 12px 0 0; padding: 0; display: grid; gap: 8px; }
.lg-trade { display: grid; grid-template-columns: 110px 1fr 70px; gap: 12px; align-items: baseline; font-size: var(--text-sm); }
.lg-trade-when { color: var(--text-3); }
.lg-trade-balance { text-align: right; }
.lg-player { border: 0; padding: 0; margin: 0 2px; background: none; font: inherit; color: var(--text-1); text-decoration: underline dotted; cursor: pointer; }
.lg-player:hover { color: var(--accent); }
```

- [ ] **Step 4: Typecheck, lint, commit**

Run: `cd web && npx tsc -b && npm run -s lint` -- clean.

```bash
git add web/src/pages/ManagerPage.tsx web/src/pages/league.css web/src/App.tsx
git commit -m "The manager page: finishes, playoffs, head-to-head, luck, draft, waivers, trades, lineups"
```

---

### Task 10: Verify end to end, then merge

**Files:** none new.

- [ ] **Step 1: Start the servers**

```bash
DEMO_WARM=0 .venv/bin/uvicorn api.main:app --port 8000 &
(cd web && npm run -s dev) &
```

Check for another session's `pytest` holding the DuckDB lock first (`lsof data/nfl.duckdb`); wait for it.

- [ ] **Step 2: Drive the real league as the owner**

Write and run a Playwright script with `.venv/bin/python` (chromium is installed there): open `http://localhost:5173/league/53929318`, assert the `.lg-import` stage list appears with one row per season and the current one ticking, wait (up to four minutes) for `.lg-strip` and `.lg-grid`, screenshot, then click the first `.lg-manager-name`, wait for `.lg-sec` sections (eight of them), assert no console errors, screenshot. Look at both screenshots. Confirm on the manager page that the head-to-head grid has seven opponents, the finishes table has one row per season, and the trades ledger names players.

- [ ] **Step 3: Check the numbers against ESPN**

Pick one manager. Compare the finishes table with ESPN's league history page for two seasons (record, final rank) and the trade count with ESPN's transaction counter (`league_members.trades` for that season). They must agree; a disagreement is a parser bug to fix here, not a note.

- [ ] **Step 4: Full suites**

```bash
.venv/bin/python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_live_api.py
cd web && npx tsc -b && npm run -s lint
```

- [ ] **Step 5: Stop the servers, merge**

Kill the two servers. The repo's convention is a local merge to `main` without a PR:

```bash
git checkout main && git merge --no-ff <branch> && git push origin main
```

Commit message for the merge: "League history and manager profiles".

---

## Self-review

**Spec coverage.** Tables (task 1, 2); import with freshness and progress (2); finishes, playoffs, head-to-head, luck, seasons strip (3); waivers with FAAB only when used, trades with balance and partners (4); lineups with optimal lineup, hit rate, bad starts, moves per week; autodraft rate; defining line; overview; profile (5); API with ownership, 403/400, job registry, cache invalidation, progress polling (6); wiring into `import_history`, registration, draft chapter from `team_profiles` (7); client, league page importing/imported states, seasons strip, manager grid (8); manager page with every section and the player popup (9); Playwright and merge (10). Head-to-head grid on the manager page is a table rather than a coloured matrix; the spec's "grid coloured by margin" is met by the margin column's colour.

**Gaps accepted.** `me` highlighting in the manager grid needs the account's member id from `/api/espn/drafts`, which is not in scope; noted inline in task 8. Trade `received` counts proposals whose items name my team; ESPN's `TRADE_PROPOSAL` rows are visible to every member, so this holds.

**Type consistency.** `Progress.snapshot()` keys match `HistoryProgress`; `league_overview` keys match `LeagueHistory`; `profile` keys match `ManagerProfile`; `owned_league(request, store, league_id, fetch)` is called the same way in every route; `import_activity(conn, league_id, fetch_json, seasons, current_season, progress, pause, max_age_hours)` is called with keywords everywhere.
