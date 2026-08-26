"""League reports: build, store, serve.

A report is one JSON document per league-season -- power rankings, a
report card per team, a profile per manager -- computed by
`scoring/league_report.py`, written up by `scoring/blurbs.py`, and stored
in the league's own database in `league_reports`. Stored, never rebuilt on
read: a shared link costs nothing after the first build and every reader
sees the same blurbs.

Two ways in. The live listener calls `on_draft_complete` when the last pick
of a real draft lands (`api/live.py`). The owner calls
`POST /api/leagues/{id}/report/{season}` to build a past season or rebuild
this one; that path refreshes the league's history with the session's
cookies first. Both run the build on a daemon thread, one flight per
league-season, because a build takes seconds and the request must not.

Reading is public. Building is the owner's, and gated by billing exactly
as the draft itself is.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
import time
import traceback
from pathlib import Path

import duckdb
from fastapi import HTTPException, Request, Response

from api import billing
from pipeline import espn_drafts
from pipeline.db import get_conn
from pipeline.league_history import import_history, is_fresh
from pipeline.leagues import LEAGUES_ROOT, league_db_path
from scoring import blurbs, league_report
from scoring.config import CURRENT_SEASON

REPORTS_DDL = """CREATE TABLE IF NOT EXISTS league_reports (
    season BIGINT, generated_at TIMESTAMP, model VARCHAR, status VARCHAR,
    payload_json VARCHAR)"""

LOCK_TRIES = 4
LOCK_RETRY_SECONDS = 0.15
CACHE_HEADER = "public, max-age=300"

_lock = threading.Lock()
_inflight: set = set()


# -- storage -------------------------------------------------------------


def store_report(conn, payload: dict) -> None:
    """Replace this season's row with `payload`. One row per season."""
    conn.execute(REPORTS_DDL)
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    payload = dict(payload)
    payload["generated_at"] = now.isoformat(timespec="seconds") + "Z"
    conn.execute("DELETE FROM league_reports WHERE season = ?", [int(payload["season"])])
    conn.execute("INSERT INTO league_reports VALUES (?, ?, ?, ?, ?)",
                 [int(payload["season"]), now, payload.get("model"),
                  payload.get("status"), json.dumps(payload)])


def _has_table(conn, table: str) -> bool:
    return bool(conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0])


def load_report(conn, season: int) -> dict | None:
    """The full stored document for one season, or None."""
    if not _has_table(conn, "league_reports"):
        return None
    row = conn.execute("SELECT payload_json FROM league_reports WHERE season = ?",
                       [int(season)]).fetchone()
    return json.loads(row[0]) if row else None


def list_reports(conn) -> list:
    """Every stored season, newest first, without the heavy payload."""
    if not _has_table(conn, "league_reports"):
        return []
    rows = conn.execute(
        "SELECT season, generated_at, status FROM league_reports ORDER BY season DESC"
    ).fetchall()
    return [{"season": int(s), "generated_at": g.isoformat(timespec="seconds") + "Z",
             "status": st} for s, g, st in rows]


def _open(path: str, read_only: bool):
    """Open a league's database, retrying a lock a few times before giving up.

    Same shape as `api/mocks.py`'s corpus reader: DuckDB is single-writer per
    file, and a build racing a live draft's own connection to the same file
    is unlucky rather than doomed.
    """
    last = None
    for attempt in range(LOCK_TRIES):
        try:
            return duckdb.connect(path, read_only=True) if read_only else get_conn(path)
        except Exception as exc:                # noqa: BLE001
            if "lock" not in str(exc).lower():
                raise
            last = exc
            if attempt + 1 < LOCK_TRIES:
                time.sleep(LOCK_RETRY_SECONDS)
    raise HTTPException(status_code=503,
                        detail=f"this league's database is busy -- try again in a moment ({last})")


def _season_on_file(conn, season: int) -> bool:
    """Whether `draft_picks` already has a row for this season.

    `is_fresh` is a WHOLE-LEAGUE stamp: the import that sets it runs at room
    connect, before a single pick lands, so a fresh league can still have
    zero rows for the season a draft just finished. Checked alongside
    `is_fresh` rather than instead of it -- see `refresh_history` below.
    """
    if not _has_table(conn, "draft_picks"):
        return False
    row = conn.execute("SELECT count(*) FROM draft_picks WHERE season = ?",
                       [int(season)]).fetchone()
    return bool(row and row[0])


def open_read(league_id: str, root: str | None = None):
    """A read-only connection to a league's file, or None when there is none."""
    path = league_db_path(league_id, root=root or LEAGUES_ROOT)
    if not Path(path).exists():
        return None
    return _open(path, read_only=True)


# -- the job ---------------------------------------------------------------


def build_report(league_id: str, season: int, picks=None, client=None,
                 root: str | None = None) -> dict:
    """Compute, write, store. Returns the stored payload.

    The league's connection is held for the reads, released for the model
    call, and reopened for the one write, so the live listener writing
    `drafted` to the same file is never blocked for the seconds the writer
    takes.
    """
    path = league_db_path(league_id, root=root or LEAGUES_ROOT)
    conn = _open(path, read_only=False)
    try:
        facts = league_report.build_facts(conn, league_id, int(season), picks=picks)
    finally:
        conn.close()
    prose = None
    if facts["status"] != "failed":
        prose = blurbs.write_blurbs(client if client is not None else blurbs.default_client(),
                                    facts)
    payload = league_report.merge_prose(facts, prose, blurbs.MODEL)
    conn = _open(path, read_only=False)
    try:
        store_report(conn, payload)
        return load_report(conn, int(season))
    finally:
        conn.close()


def building(league_id: str, season: int) -> bool:
    with _lock:
        return (str(league_id), int(season)) in _inflight


def _spawn(name: str, fn) -> None:
    threading.Thread(target=fn, name=f"job-{name}", daemon=True).start()


def spawn_build(league_id: str, season: int, picks=None, client=None, spawn=None,
                before=None, root: str | None = None) -> bool:
    """Start one build for this league-season unless one is running.

    `before` runs first, on the same thread -- the POST route's history
    refresh -- so a build never races the import writing to the same file.
    """
    key = (str(league_id), int(season))
    with _lock:
        if key in _inflight:
            return False
        _inflight.add(key)

    def run():
        try:
            if before is not None:
                before()
            build_report(league_id, season, picks=picks, client=client, root=root)
            print(f"league {league_id}: report for {season} stored")
        except Exception:      # noqa: BLE001 -- a background job
            print(f"league {league_id}: report for {season} failed")
            traceback.print_exc()
        finally:
            with _lock:
                _inflight.discard(key)
    (spawn or _spawn)(f"report-{league_id}-{season}", run)
    return True


# -- the live room's hooks --------------------------------------------------


def cookies_for_connect(request: Request, swid: str, espn_s2: str | None, store=None) -> dict | None:
    """The ESPN cookies a connect request can act with, or None.

    The bookmarklet may send `espn_s2` in the body; a returning browser
    carries the custody cookie instead; the machine owner has a saved
    login. Same order `api/drafts.session_for` uses.
    """
    if espn_s2:
        return espn_drafts.cookies_for(swid, espn_s2)
    from api.drafts import session_for
    session = session_for(request, store)
    return session.cookies if session is not None else None


def on_draft_complete(league_id: str, season, swid: str | None, session, drafted: list,
                      store=None, spawn=None, root: str | None = None) -> bool:
    """The last pick landed: build this season's report if the room may.

    Free instances build every real draft; a paid instance builds only a
    draft somebody paid for, checked on the connecting account's SWID
    because there is no request on the listener thread. Mocks never build.
    Never raises.
    """
    try:
        if billing.is_free_draft(league_id):
            return False
        year = int(season or CURRENT_SEASON)
        if billing.enabled():
            if not swid:
                return False
            from api.custody import _store
            if not billing.entitled(_store(store).account_ids(swid), league_id, year):
                return False
        conn = open_read(league_id, root=root)
        try:
            names = league_report.names_for(conn) if conn is not None else {}
        finally:
            if conn is not None:
                conn.close()
        picks = league_report.live_picks(
            drafted, getattr(session, "board_by_id", {}) or {}, session.settings, names,
            getattr(session, "team_slots", {}) or {}, year)
        return spawn_build(league_id, year, picks=picks, spawn=spawn, root=root)
    except Exception:      # noqa: BLE001
        traceback.print_exc()
        return False


# -- routes ------------------------------------------------------------------


def register_report_routes(app, store=None, fetch=None, spawn=None, root: str | None = None,
                           session_for=None, entries_for=None):
    """`GET /api/leagues/{id}/reports`, `GET .../report/{season}`,
    `POST .../report/{season}`. `session_for` and `entries_for` are injectable
    for tests; the defaults read custody and the account's ESPN league list."""
    from api.drafts import session_for as _default_session_for

    def _session(request):
        if session_for is not None:
            return session_for(request)
        return _default_session_for(request, store)

    def _entries(session):
        if entries_for is not None:
            return entries_for(session)
        return espn_drafts.league_entries(session.swid, session.cookies, None, fetch=fetch)

    @app.get("/api/leagues/{league_id}/reports")
    def reports_index(league_id: str):
        conn = open_read(league_id, root=root)
        if conn is None:
            return []
        try:
            return list_reports(conn)
        finally:
            conn.close()

    @app.get("/api/leagues/{league_id}/report/{season}")
    def report(league_id: str, season: int, response: Response):
        conn = open_read(league_id, root=root)
        if conn is None:
            raise HTTPException(status_code=404, detail="No report for this league.")
        try:
            payload = load_report(conn, season)
        finally:
            conn.close()
        if payload is None:
            raise HTTPException(status_code=404, detail="No report for this season.")
        response.headers["Cache-Control"] = CACHE_HEADER
        return payload

    @app.post("/api/leagues/{league_id}/report/{season}", status_code=202)
    def build(league_id: str, season: int, request: Request):
        session = _session(request)
        if session is None:
            raise HTTPException(status_code=403, detail="Connect your ESPN account first.")
        if not any(str(e.get("league_id")) == str(league_id) for e in _entries(session)):
            raise HTTPException(status_code=403, detail="That league is not one of yours.")
        if billing.is_free_draft(league_id):
            raise HTTPException(status_code=400, detail="Mock drafts have no report card.")
        billing.require_paid(request, league_id, season, store)
        cookies = session.cookies

        def refresh_history():
            """Refresh this league's history before building -- unless it is
            already fresh AND this season's picks are already on file.

            BOTH, not `is_fresh` alone. `is_fresh` is a whole-league stamp
            set by the import that runs at room connect, before a single
            pick lands -- so a league can be "fresh" while the season that
            just finished drafting has zero rows in `draft_picks`. Skipping
            on the stamp alone would leave that build running against an
            empty season and stored as `status: "failed"` for up to
            FRESH_DAYS, while the caller was told 202 "building".
            """
            try:
                path = league_db_path(league_id, root=root or LEAGUES_ROOT)
                if Path(path).exists():
                    existing = _open(path, read_only=True)
                    try:
                        if is_fresh(existing) and _season_on_file(existing, season):
                            return
                    finally:
                        existing.close()
                import_history(league_id, cookies, current_season=CURRENT_SEASON,
                               root=root or LEAGUES_ROOT)
            except Exception as exc:      # noqa: BLE001 -- the build runs on what is there
                print(f"league {league_id}: history refresh failed: {exc}")
        spawn_build(league_id, season, spawn=spawn, before=refresh_history, root=root)
        return {"status": "building"}
