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
import threading
import time

import pandas as pd

from pipeline.db import read_table, write_table
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

    def absent(self, season: int) -> None:
        """A season ESPN 404s outright, most often the current season
        unioned in by `import_history` before its league has rolled over to
        it. Marks the stage done with its own label rather than raising --
        there is nothing this season to read, not a failure of the walk."""
        with self._lock:
            stage = self._stage(season)
            if stage is not None:
                stage.update(done=True, label=f"{season} · not on ESPN")
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


def _stored_latest(raw: pd.DataFrame, season: int) -> int:
    """The `latestScoringPeriod` already on file for this season, from
    before this walk's own fetch, or 0 if nothing is stored yet. A week
    strictly behind that mark was already over the last time we looked, so
    its box score cannot have changed since; a week that WAS the latest
    (or newer) earns one more read even after the season has since moved
    on, because its stats may only have settled since then."""
    if raw.empty:
        return 0
    hit = raw[(raw.season == season) & (raw.view == "mTeam") & (raw.week == 0)]
    if hit.empty:
        return 0
    status = (json.loads(hit.iloc[0].payload_json).get("status")) or {}
    return int(status.get("latestScoringPeriod") or 0)


def import_activity(conn, league_id: str, fetch_json, seasons: list, current_season: int,
                    progress: Progress | None = None, pause: float = 0.0,
                    max_age_hours: float = CURRENT_MAX_AGE_HOURS) -> dict:
    """Walk the seasons and their weeks, keep every answer, rebuild the tables.

    `seasons` is what `import_seasons` found drafted, newest first. A
    finished season already on file is skipped whole. The current season
    is read again when its newest stored answer is older than
    `max_age_hours`; within it, a week whose games were already over as of
    the *previously* stored standings, and is already stored, is not read
    again -- a week that was still the latest at last check gets one more
    read even once the season has moved past it, since its stats may have
    only settled since then.

    Raises whatever `fetch_json` raises other than FileNotFoundError, after
    recording the failure on `progress`. A 404 on a week is treated as an
    empty week; a 404 on a season's own views (a league ESPN has not
    rolled over to that season yet -- the current season, most often, when
    `import_history` unions it in whether or not it has been drafted) skips
    that season outright rather than failing the whole walk, and is
    recorded on `progress` via `Progress.absent`.
    """
    progress = progress or Progress(league_id)
    progress.start(list(seasons))
    requests = 0
    weeks_read = {}
    try:
        for season in seasons:
            raw = _raw(conn)
            old_latest = _stored_latest(raw, season)
            finished = season < current_season
            if finished and all(_have(raw, season, v, 0) for v in SEASON_VIEWS):
                progress.finish(season)
                continue
            if not finished and not _stale(raw, season, max_age_hours):
                progress.finish(season)
                continue
            payloads = {}
            try:
                for view in SEASON_VIEWS:
                    payloads[view] = fetch_json(season_url(league_id, season, view))
                    requests += 1
                    _store_raw(conn, season, view, 0, payloads[view])
                    if pause:
                        time.sleep(pause)
            except FileNotFoundError:
                # The season itself does not exist on ESPN yet (or ever
                # will) -- nothing to store, and not a reason to fail the
                # rest of the walk. Whatever raw rows an earlier walk left
                # for this season, if any, are untouched.
                progress.absent(season)
                continue
            status = payloads["mTeam"].get("status") or {}
            final = int(status.get("finalScoringPeriod") or 17)
            latest = int(status.get("latestScoringPeriod") or final)
            last_week = final if finished else min(final, max(latest, 1))
            raw = _raw(conn)
            weeks_read[season] = 0
            for week in range(1, last_week + 1):
                progress.tick(season, week, last_week)
                # Over and stored, as of what standings last said: nothing
                # can have changed. Judged against `old_latest`, not the
                # freshly refetched `latest` above, so the week that was
                # in progress last time still gets one more read now.
                settled = finished or week < old_latest
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
