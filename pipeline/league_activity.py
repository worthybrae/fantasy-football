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
