"""Import one league's ESPN history into that league's own database file.

`pipeline/import_league.py` is the command-line importer for the machine
owner's league: it logs in with Playwright and writes to the default
database. This module is the same import for any league an account has
connected through the site -- the account's own ESPN cookies do the
authentication, and the rows land in `data/leagues/<id>.duckdb` (the
default league still resolves to the shared file, see
`pipeline.leagues.league_db_path`).

It runs in the background when an account connects to a real league's
room, so that by the time the draft ends the history a report card needs
is already on disk and the draft-end job needs no network call and no
credential.
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path

import pandas as pd

from pipeline.db import DEFAULT_PATH, get_conn, read_table, record_freshness, write_table
from pipeline.espn_drafts import KONA, http_fetch
from pipeline.espn_league import PLAYERS_FILTER, import_seasons
from pipeline.league_activity import import_activity
from pipeline import leagues
from pipeline.leagues import league_db_path, provision_league
from scoring.config import CURRENT_SEASON

FRESH_DAYS = 7
# Seconds between ESPN requests in the walk. Enough to be a polite reader,
# short enough that six seasons still take a couple of minutes.
PAUSE_SECONDS = 0.15


def json_fetch(fetch, cookies: dict):
    """The one-argument JSON fetch `import_seasons` wants, over a cookie fetch.

    `fetch(url, cookies, headers) -> (status, text)` is the shape of
    `espn_drafts.http_fetch`. 404 becomes FileNotFoundError because that is
    what the season walk reads as "no such season"; 401/403 becomes
    PermissionError so an expired session is a named failure rather than
    fifteen silent misses.
    """
    def get(url: str):
        headers = dict(KONA)
        if "/players?" in url:
            headers["X-Fantasy-Filter"] = PLAYERS_FILTER
        status, text = fetch(url, cookies, headers)
        if status == 404:
            raise FileNotFoundError(url)
        if status in (401, 403):
            raise PermissionError(f"{status} for {url}")
        if status >= 400:
            raise RuntimeError(f"{status} for {url}")
        return json.loads(text)
    return get


def is_fresh(conn, max_age_days: int = FRESH_DAYS) -> bool:
    meta = read_table(conn, "meta")
    if meta.empty:
        return False
    row = meta[meta["source"] == "league"]
    if row.empty or not bool(row.iloc[0]["ok"]):
        return False
    age = pd.Timestamp.now() - pd.Timestamp(row.iloc[0]["refreshed_at"])
    return age.days < max_age_days


def import_history(league_id: str, cookies: dict, fetch=None,
                   current_season: int = CURRENT_SEASON,
                   universal_path: str = DEFAULT_PATH, root: str | None = None,
                   adp_fetch=None, progress=None, pause: float = PAUSE_SECONDS) -> dict:
    """Import drafted seasons, standings, historic ADP and every season's
    activity for one league.

    Returns `import_seasons`'s summary, with an `activity` key added
    (`import_activity`'s own summary). Raises what `import_seasons` raises
    (no drafted seasons, an auction league, a dead session); the caller
    decides whether that is a log line or an error.

    `root=None` reads `leagues.LEAGUES_ROOT` HERE rather than binding it as
    a default argument, which would freeze it at import time -- the same
    call-time read `api/live.py` makes, and the reason is the same: the test
    suite's `_isolated_leagues_root` patches that module attribute, and a
    frozen default would write a real file under the repo's data/leagues/.
    """
    root = leagues.LEAGUES_ROOT if root is None else root
    # Imported here, not at module scope: this module is reachable from
    # `api/live.py`'s own imports, and `pipeline.import_league` pulls in
    # `pipeline.sources` and with it `nfl_data_py` -- seconds of import cost
    # on every API boot, for a function only the history import ever calls.
    from pipeline.import_league import historic_adp_frames
    raw = fetch if fetch is not None else http_fetch()
    path = provision_league(league_id, universal_path, root=root)
    conn = get_conn(path)
    try:
        summary = import_seasons(conn, league_id, current_season, json_fetch(raw, cookies))
        existing = read_table(conn, "historic_adp")
        have = set(existing["season"]) if not existing.empty else set()
        frames = [existing] if not existing.empty else []
        frames.extend(historic_adp_frames(summary["seasons"], fetch_adp=adp_fetch, skip=have))
        if frames:
            rows = pd.concat(frames, ignore_index=True)
            write_table(conn, "historic_adp", rows)
            record_freshness(conn, "historic_adp", True, len(rows))
        # import_seasons walks only DRAFTED seasons; ESPN's own
        # status.previousSeasons can list more than that. The spec wants
        # every season the league has, so the activity walk below also
        # covers the current season even when import_seasons did not find
        # it drafted yet -- a season whose draft has not happened still has
        # members, standings-in-progress and settings worth reading.
        seasons = sorted(set(summary["seasons"]) | {current_season}, reverse=True)
        # Everything a season does after the draft -- see league_activity.
        # Same connection, same fetch, same walk from the caller's point of
        # view. `progress` is the page's stage list when a visit started
        # this; None from the connect path, which nobody is watching.
        summary["activity"] = import_activity(
            conn, league_id, json_fetch(raw, cookies), seasons,
            current_season, progress=progress, pause=pause)
        record_freshness(conn, "league", True, len(summary["seasons"]))
        return summary
    finally:
        conn.close()


def _spawn(name: str, fn) -> None:
    threading.Thread(target=fn, name=f"job-{name}", daemon=True).start()


def spawn_import_if_stale(league_id: str, cookies: dict, spawn=None, **kwargs) -> bool:
    """Start a background import unless this league was imported recently.

    Never raises: this is called from a request path that must not fail
    because a side job could not start. Returns whether a job started.
    """
    root = kwargs.get("root") or leagues.LEAGUES_ROOT
    try:
        path = league_db_path(league_id, root=root)
        if Path(path).exists():
            conn = get_conn(path)
            try:
                if is_fresh(conn):
                    return False
            finally:
                conn.close()
    except Exception:      # noqa: BLE001
        traceback.print_exc()
        return False

    def run():
        try:
            summary = import_history(league_id, cookies, **kwargs)
            print(f"league {league_id}: imported {summary['picks']} picks "
                  f"across {len(summary['seasons'])} seasons")
        except Exception as exc:      # noqa: BLE001 -- a background job
            print(f"league {league_id}: history import failed: {exc}")
    (spawn or _spawn)(f"history-{league_id}", run)
    return True
