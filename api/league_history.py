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

import pandas as pd
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from api.league_access import owned_league
from pipeline import espn_drafts as drafts
from pipeline import leagues
from pipeline.db import DEFAULT_PATH, get_conn, read_table, write_table
from pipeline.espn_league import STANDINGS_COLUMNS, parse_standings
from pipeline.league_activity import (CURRENT_MAX_AGE_HOURS, Progress, _raw, _stale,
                                      import_activity, season_url)
from pipeline.league_history import json_fetch
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
    # Read now, not bound as a default argument -- pipeline.leagues.LEAGUES_ROOT
    # is what tests (and a deployment with a non-default data dir) patch, and
    # a frozen default would stop seeing that patch after import time.
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


def _rebuild_standings(conn) -> None:
    """`league_standings`, from the `mTeam` answers `import_activity` just
    stored in `league_raw`. No new fetch: the season view already carries
    each team's record, points and final rank.

    `import_activity` alone never writes this table -- it is Task 7's
    `import_history` (via `pipeline.espn_league.import_seasons`) that
    builds it properly, alongside draft validation this walk does not do.
    Until that lands, `scoring/manager_profile.py` reads `league_standings`
    directly with no guard for the table being altogether absent, so a
    freshly imported league would 500 every read rather than show a page
    with no titles or finishes yet. This keeps the table present -- one row
    per team ESPN has answered for -- without duplicating the draft walk.
    """
    raw = read_table(conn, "league_raw")
    frames = []
    if not raw.empty:
        season_views = raw[(raw.view == "mTeam") & (raw.week == 0)]
        for row in season_views.itertuples():
            frames.append(parse_standings(json.loads(row.payload_json), int(row.season)))
    standings = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=STANDINGS_COLUMNS))
    write_table(conn, "league_standings", standings)


def run_import(league_id: str, cookies: dict, progress: Progress, fetch=None,
               universal_path: str = DEFAULT_PATH) -> None:
    """The job. Until `import_history` carries the week walk (see the
    league-report branch), this walks activity on its own."""
    raw = fetch if fetch is not None else drafts.http_fetch()
    fetch_json = json_fetch(raw, cookies)
    path = provision_league(league_id, universal_path, root=_leagues_root())
    conn = get_conn(path)
    try:
        seasons = seasons_from_status(fetch_json, league_id, CURRENT_SEASON)
        import_activity(conn, league_id, fetch_json, seasons, CURRENT_SEASON,
                        progress=progress, pause=PAUSE_SECONDS)
        _rebuild_standings(conn)
    except Exception as exc:      # noqa: BLE001 -- import_activity already
        # records its own failures on `progress` before re-raising; this
        # also catches anything AFTER it (right now, only
        # _rebuild_standings), so a failure there cannot leave `progress`
        # reading "done" for a league whose file never got a usable
        # league_standings table -- the same pattern import_activity uses,
        # not doubled up if it already recorded one.
        if progress.snapshot()["phase"] != "failed":
            progress.fail(f"{type(exc).__name__}: {exc}")
        raise
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


def _release_reservation(league_id: str, progress: Progress) -> None:
    """Undo start_history's reservation of `_JOBS[league_id]`, but only if
    it is still ours. Used on the "fresh, no job needed" path and on any
    failure between reserving the slot and the job actually starting --
    both cases need the same identity-guarded release, so it lives once
    here rather than twice inline."""
    with _LOCK:
        if _JOBS.get(league_id) is progress:
            del _JOBS[league_id]


def register_league_history_routes(app, store=None, fetch=None, runner=None,
                                   universal_path: str = DEFAULT_PATH) -> None:
    """`fetch` and `runner` are seams for tests: the ESPN fetch, and how the
    job is started (default: a daemon thread). `universal_path` is where a
    newly provisioned league copies its universal tables from -- a seam so
    tests never provision against the real data/nfl.duckdb."""

    def start(fn):
        if runner is not None:
            runner(fn)
        else:
            threading.Thread(target=fn, name="league-history", daemon=True).start()

    @app.post("/api/leagues/{league_id}/history")
    def start_history(league_id: str, request: Request):
        session = owned_league(request, store, league_id, fetch=fetch)
        # Check-and-reserve is one atomic step under _LOCK: a job is only
        # ever recorded in _JOBS while the lock is held, so two requests
        # that both arrive before anything is registered cannot each see
        # "nothing running yet" and both start their own import against the
        # same league file. _imported/_fresh (file reads) run AFTER the
        # reservation, deliberately outside the lock -- they no longer need
        # it, since the reservation already claims the slot.
        #
        # Lock order: _LOCK, then (inside snapshot()/start() below) a
        # Progress's own internal lock -- never the reverse, so this can
        # never deadlock against anything that reads a Progress under its
        # own lock first.
        with _LOCK:
            running = _JOBS.get(league_id)
            status = running.snapshot()["phase"] if running is not None else None
            if status == "running":
                return _status(202, {"status": "running"})
            # A job that failed does not block a retry -- and the fresh
            # check just below must not fire for it either: the walk never
            # finished cleanly, whatever import_activity itself managed to
            # store before that.
            retryable = status == "failed"
            progress = Progress(league_id)
            progress.start([])
            _JOBS[league_id] = progress
        # Everything from here down can still raise before the job is
        # genuinely under way -- a DB read inside _imported/_fresh, cookie
        # construction, the thread itself failing to start -- and none of
        # that may leave the reservation above behind: an orphaned
        # "running" placeholder is a job nothing will ever advance, and
        # every later POST would answer 202 running forever.
        try:
            if not retryable and _imported(league_id) and _fresh(league_id):
                _release_reservation(league_id, progress)
                return {"status": "fresh"}
            cookies = drafts.cookies_for(session.swid, session.espn_s2)

            def job():
                try:
                    run_import(league_id, cookies, progress, fetch=fetch,
                              universal_path=universal_path)
                except Exception:      # noqa: BLE001 -- run_import has
                    traceback.print_exc()  # already recorded the failure
                    # on `progress`; this only puts a trace in the server's
                    # own log.
            start(job)
        except BaseException:
            _release_reservation(league_id, progress)
            raise
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
    conn = get_conn(_path(league_id))
    try:
        return not _stale(_raw(conn), CURRENT_SEASON, CURRENT_MAX_AGE_HOURS)
    finally:
        conn.close()


def _status(code: int, body: dict):
    return JSONResponse(status_code=code, content=body)
