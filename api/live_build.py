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

WHAT THE WORKER MAY TOUCH. One league file, which it provisions for itself
from the read-only universal snapshot (pipeline/leagues.snapshot_universal)
when the file does not exist yet, then opens, builds against and closes.
DuckDB's lock is per process, so the worker cannot open a file the parent
has open, and it never opens the live universal database at all (the
parent holds that one for its whole life; the snapshot is the worker's
copy of it). api/live.py's `_provision_and_build` keeps the order that
makes this safe -- register the room as a holder of the file, submit,
wait, then open the parent's own connection -- and builds inline instead
when another room in the parent already holds the file (two leaguemates
share one), because that file is not available to a second process.

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

from pipeline.db import (WORKER_CONN_MEMORY_LIMIT, WORKER_CONN_THREADS,
                         apply_worker_conn_limits)

WORKERS_ENV = "LIVE_BUILD_WORKERS"

# The worker's build connection gets the WORKER limits from pipeline/db.py;
# the parent's own per-room connection gets the smaller PARENT ones there.
# Kept as names here for the callers and tests that read them.
CONN_THREADS = WORKER_CONN_THREADS
CONN_MEMORY_LIMIT = WORKER_CONN_MEMORY_LIMIT

# How long a connect waits for its worker. Well past the 35 s the slowest
# measured build takes, and short enough that a worker that has hung does
# not hold a request thread for the rest of the evening.
LIVE_BUILD_TIMEOUT = 120.0
# How much longer a connect waits for a worker that would not cancel: a
# running build cannot be cancelled, and while it runs the worker has the
# league file open, so building inline would race it for the lock. Past
# this the connect is refused (BuildTimedOut) rather than raced.
LIVE_BUILD_GRACE = 30.0


class BuildTimedOut(RuntimeError):
    """A worker did not finish within LIVE_BUILD_TIMEOUT + LIVE_BUILD_GRACE
    and could not be cancelled. The caller must not open the file.

    The build itself is NOT over: the worker is still running and still has
    the league file open, for as long as it takes. That is what
    `build_in_worker`'s `abandoned` hook is for -- see it, and api/live.py's
    `_build_off_process`, for who keeps the file spoken for meanwhile."""

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
    """Cap a WORKER's build connection (see pipeline/db.py for the sizes)."""
    apply_worker_conn_limits(conn)


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


def _build_job(league_path: str, snapshot_path: str, league_id: str,
               settings_json: str | None, team_id: int | None, season) -> object:
    """The worker's whole job: provision the league file from the snapshot
    if it does not exist yet, open it, work out the slot, build the
    session, close the file, hand the session back.

    Runs in a worker process when there is a pool, and in the caller's
    thread otherwise. No progress object: one cannot be pickled across a
    process boundary, and the parent reports the stages from the session
    it gets back. Returns the DraftSession; `season` is accepted for
    symmetry with the connect and unused here, since build_session reads
    the season off the league's own settings.
    """
    from api import live
    from pipeline.db import get_conn
    from pipeline.leagues import provision_league
    from scoring import league as league_mod

    settings = league_mod.from_json(settings_json) if settings_json else None
    provision_league(league_id, universal_path=snapshot_path,
                     root=os.path.dirname(league_path), snapshot=snapshot_path)
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
                    season, abandoned=None) -> object:
    """Build one league's session and return it (a DraftSession).

    `universal_path` is the READ-ONLY SNAPSHOT of the universal database
    (pipeline/leagues.snapshot_universal), not the live file: the worker
    provisions the league file from it when the file does not exist yet,
    and it never opens the live file, which the parent holds. The caller
    must not hold a connection to `league_path` while this runs -- see the
    module docstring for why the worker, not the caller, does the opening.

    In a worker process when `workers() > 0`, inline otherwise -- and
    inline for THIS call, with a line in the log, when the pool is broken
    twice over or the worker does not answer within LIVE_BUILD_TIMEOUT.
    A connect that falls back is slower, not failed.

    `abandoned(future)` is called on the one path this function gives up on
    a worker that is still running (BuildTimedOut). The build goes on with
    nobody waiting for it, and until it ends the worker still has the
    league file open -- so the caller is handed the future and can keep the
    file spoken for until it completes. Without that, the retry the 503
    invites finds the file unclaimed and hands it to a SECOND worker, which
    is a lock error rather than a slow connect.
    """
    args = (league_path, universal_path, league_id, settings_json, team_id,
            season)
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
    except BrokenProcessPool as exc:
        # The worker died mid-build (the kernel's memory killer, most
        # likely). The pool is finished with; the next connect gets a new
        # one, and this one builds inline -- the file is free, the worker
        # is gone.
        print(f"live_build: worker died building league {league_id} "
              f"({exc}); replacing the pool and building inline")
        shutdown()
        return _build_job(*args)
    except TimeoutError:
        pass
    if future.cancel():
        # Never started: nothing holds the file, so inline is safe.
        print(f"live_build: league {league_id} waited {LIVE_BUILD_TIMEOUT:.0f}s "
              "for a worker; building it inline")
        return _build_job(*args)
    # Running, and a running worker cannot be cancelled -- it has the file
    # open, so this process must NOT open it. Give it a little longer.
    print(f"live_build: worker still building league {league_id} after "
          f"{LIVE_BUILD_TIMEOUT:.0f}s; waiting {LIVE_BUILD_GRACE:.0f}s more")
    try:
        failure = future.exception(timeout=LIVE_BUILD_GRACE)
    except TimeoutError:
        if abandoned is not None:
            # Before the raise, so the file is claimed by the time the
            # caller's own claim comes off in its failure handler.
            abandoned(future)
        raise BuildTimedOut(
            f"the build for league {league_id} timed out after "
            f"{LIVE_BUILD_TIMEOUT + LIVE_BUILD_GRACE:.0f}s") from None
    # `exception()` RETURNS the worker's failure rather than raising it, so
    # a worker that died during the grace shows up here as a value.
    if isinstance(failure, BrokenProcessPool):
        print(f"live_build: worker died building league {league_id} "
              f"({failure}); replacing the pool and building inline")
        shutdown()
        return _build_job(*args)
    if failure is not None:
        raise failure               # the build's own error, as the inline path would
    return future.result()          # finished in the grace
