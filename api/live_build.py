"""Building a draft session off the web process's CPU.

A connect costs 8-35 s of CPU -- the board (1.5-1.9 s), the simulation pool
(2.2-3.6 s) and, for a league with imported history, `fit_all` (up to
30 s). Done on the request thread under the GIL, every other request in
the process waits behind it: with a dozen rooms connecting at eight
o'clock, a poll of /api/live/state that should take 5 ms takes seconds.

So when `LIVE_BUILD_WORKERS` is above zero the build runs in a child
process instead. A spawn-context pool, not fork: the parent holds DuckDB
connections and a websocket or two hundred, and forking a process with
live database handles and threads is how you get a child that deadlocks
on a lock its parent's thread was holding. Spawned workers import this
module fresh, which is why `_build_job` and everything it calls has to
be reachable from the module top level.

WHAT THE WORKER MAY TOUCH. One league file, and only after the parent has
provisioned it and closed its own handle: DuckDB's lock is per process,
so the worker cannot open a file the parent still has open, and it cannot
open the shared universal database at all (the parent holds that one for
its whole life). api/live.py's `_provision_and_build` keeps that order --
provision, submit, wait, then open the parent's own connection -- and
falls back to building inline when another room in the parent already
holds the league file (two leaguemates share one), because that file is
not available to a second process either.

With the variable unset or zero everything runs inline, exactly as before
this module existed, and a test that monkeypatches `build_session` here is
honoured because the inline path calls through this module's own name.
"""
from __future__ import annotations

import multiprocessing
import os
import threading
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool

WORKERS_ENV = "LIVE_BUILD_WORKERS"

# DuckDB's defaults are sized for one database per machine: a thread per
# core and most of the RAM. A deployment holds one connection per live
# room, so each gets a slice instead. Two threads is enough for the small
# per-pick queries a room runs; 256 MB is well above what one league file's
# working set needs and low enough that two hundred of them do not add up
# to the container.
CONN_THREADS = 2
CONN_MEMORY_LIMIT = "256MB"

# How long a connect waits for its worker. Well past the 35 s the slowest
# measured build takes, and short enough that a worker that has hung does
# not hold a request thread for the rest of the evening: on expiry the
# future is cancelled and that one connect builds inline instead.
LIVE_BUILD_TIMEOUT = 120.0

_pool_lock = threading.Lock()
_pool: ProcessPoolExecutor | None = None


def workers() -> int:
    """How many build workers this process runs: the variable, or none."""
    raw = (os.environ.get(WORKERS_ENV) or "").strip()
    try:
        return max(0, int(raw)) if raw else 0
    except ValueError:
        return 0


def apply_conn_limits(conn) -> None:
    """Cap one per-league DuckDB connection's threads and memory."""
    conn.execute(f"SET threads={CONN_THREADS}")
    conn.execute(f"SET memory_limit='{CONN_MEMORY_LIMIT}'")


def _get_pool() -> ProcessPoolExecutor:
    """The pool, created on first use with the size the environment names.

    Lazy so importing this module -- which every test does through api.live
    -- never spawns a process, and so the size is read when the first build
    is asked for rather than at import.
    """
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(
                max_workers=workers(),
                mp_context=multiprocessing.get_context("spawn"))
        return _pool


def submit(fn, *args) -> Future:
    """Run `fn(*args)` in a worker. Requires `workers() > 0`.

    A pool whose worker has died (killed by the kernel for memory, most
    likely) refuses every later submit with BrokenProcessPool, forever.
    One such refusal drops the pool and builds a fresh one for a single
    retry; a second refusal propagates, and the caller falls back to
    building inline for that connect.
    """
    if workers() <= 0:
        raise RuntimeError(f"{WORKERS_ENV} is not set; nothing to submit to")
    try:
        return _get_pool().submit(fn, *args)
    except BrokenProcessPool:
        print("live_build: worker pool is broken; replacing it")
        shutdown()
        return _get_pool().submit(fn, *args)


def shutdown() -> None:
    """Stop the pool without waiting. Called from the app's shutdown hook
    and when a broken pool is replaced; a process exit reaps the workers
    anyway."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


def build_session(*args, **kwargs):
    """`api.live.build_session`, reached by name at call time.

    An indirection rather than an import: api.live imports this module, so
    importing it back at the top would be a cycle, and a test that
    monkeypatches `api.live_build.build_session` needs a name of this
    module's own to replace.
    """
    from api import live
    return live.build_session(*args, **kwargs)


def _build_job(league_path: str, league_id: str, settings_json: str | None,
               team_id: int | None, season) -> object:
    """The worker's whole job: open the league file, work out the slot,
    build the session, close the file, hand the session back.

    Runs in a worker process when there is a pool, and in the caller's
    thread otherwise. No progress object: one cannot be pickled across a
    process boundary, and the parent reports the stages from the session
    it gets back. Returns the DraftSession; `season` is accepted for
    symmetry with the connect and unused here, since build_session reads
    the season off the league's own settings.
    """
    from api import live
    from pipeline.db import get_conn
    from scoring import league as league_mod

    settings = league_mod.from_json(settings_json) if settings_json else None
    conn = get_conn(league_path)
    try:
        apply_conn_limits(conn)
        cur = conn.cursor()
        try:
            my_slot = None
            if team_id is not None:
                my_slot = (live._slot_from_pick_order(settings, team_id)
                           or live._slot_for_team(cur, team_id))
            return build_session(cur, my_slot, league_id=league_id,
                                 settings=settings)
        finally:
            cur.close()
    finally:
        conn.close()


def build_in_worker(league_path: str, universal_path: str, league_id: str,
                    settings_json: str | None, team_id: int | None,
                    season) -> object:
    """Build one league's session and return it (a DraftSession).

    The league file at `league_path` must already be provisioned from
    `universal_path` by the caller, and the caller must not hold a
    connection to it while this runs -- see the module docstring for why
    the worker, not the caller, does the opening. `universal_path` is the
    shared database the league file was seeded from; the worker never
    opens it (the parent holds it for its whole life) and it is carried
    here only so the provisioning contract is visible at the call.

    In a worker process when `workers() > 0`, inline otherwise -- and
    inline for THIS call, with a line in the log, when the pool is broken
    twice over or the worker does not answer within LIVE_BUILD_TIMEOUT.
    A connect that falls back is slower, not failed.
    """
    args = (league_path, league_id, settings_json, team_id, season)
    if workers() <= 0:
        return _build_job(*args)
    try:
        future = submit(_build_job, *args)
    except BrokenProcessPool as exc:
        print(f"live_build: worker pool broken twice ({exc}); building "
              f"league {league_id} inline")
        return _build_job(*args)
    try:
        return future.result(timeout=LIVE_BUILD_TIMEOUT)
    except TimeoutError:
        future.cancel()
        print(f"live_build: worker did not build league {league_id} within "
              f"{LIVE_BUILD_TIMEOUT:.0f}s; building it inline")
        return _build_job(*args)
