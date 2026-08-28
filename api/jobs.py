"""Work a deployed instance does to itself, in the background.

WHY THE SERVER REFRESHES ITSELF AT ALL. A local checkout is refreshed by a
person typing `make refresh` before draft night. A deployment has nobody to
type it, and its volume starts empty -- which is not a crash but something
worse-looking: the site comes up and renders all its own furniture around no
data whatsoever. "No mock draft is running", an empty archive, blank panels
where the figures go.

WHY IN THIS PROCESS RATHER THAN A SECOND SERVICE, which is where anybody
would put it first. Two independent reasons, and either alone would be
enough:

  * DuckDB takes a single-writer lock per FILE, per PROCESS. A second
    connection inside one process shares the cached instance and is fine; a
    second process gets `Could not set lock on file`. The Makefile already
    documents this happening between the API and `make fit-prior`.
  * A platform volume attaches to one service. There is no arrangement in
    which a worker service and the web service both hold `/data`.

So the jobs are threads here, sharing this process's DuckDB instance, and
they are daemon threads: a deploy that needs to stop is not held open by a
refresh half way through 223 news queries.

EVERYTHING IS OFF UNLESS SWITCHED ON. Importing this app -- in a test, in a
script, on a laptop running `make up` -- must never start pulling from the
network. The deployment opts in with an environment variable; nothing else
does.
"""
from __future__ import annotations

import datetime as dt
import os
import threading
import time

import pandas as pd

from pipeline.db import read_table

# Whether this instance refreshes itself at all.
REFRESH_ENV = "RUN_REFRESH_ON_BOOT"

# Whether this instance plays mock drafts to grow the corpus. Separate switch
# from the refresh: they fail for completely different reasons (a stats feed
# being down vs an expired ESPN login) and one being off is not a reason for
# the other to be.
FARM_ENV = "RUN_FARM"

# How many drafts one pass of the farm asks for before the loop comes back
# round. Not a limit on the total -- the loop is endless -- but a bound on
# how long a single call sits inside `mock_farm.farm` without the supervisor
# above it getting a look in. Each draft is 20-40 minutes of wall clock.
FARM_BATCH_ENV = "FARM_DRAFTS_PER_PASS"
DEFAULT_FARM_BATCH = 1

# HOW MANY DRAFTS AT ONCE. Not the same question as the batch above: `farm(n)`
# plays its n drafts one after another ("ONE SESSION PER PROCESS"), so
# concurrency comes from running several processes, exactly as a developer
# does locally with several terminals.
#
# `pipeline.farm_claims` is what makes that safe -- an atomic O_CREAT|O_EXCL
# claim per room, so two farms never sit in the same draft -- and the account
# side is settled: one ESPN account can hold seats in several drafts at once
# (see mock_farm.farm's own docstring, which reports it confirmed).
#
# Defaults to one, so switching the farm on does not silently multiply this
# deployment's traffic to ESPN. Raise it to match what the corpus needs.
FARM_CONCURRENCY_ENV = "FARM_CONCURRENCY"
DEFAULT_FARM_CONCURRENCY = 1

# The ceiling, and it is about ESPN rather than about this machine. Each farm
# is a ~60 MB process, so the memory is cheap; what is not cheap is a single
# datacenter IP holding a dozen seats in a public mock lobby, which is the
# shape of behaviour that gets an account looked at.
MAX_FARM_CONCURRENCY = 8

# How long the farm waits after a pass that recorded nothing. ESPN's mock
# lobby is empty at 4am and full at 8pm, so a run that finds no joinable room
# is an ordinary outcome and not an error -- it just should not be retried in
# a tight loop.
FARM_IDLE_SECONDS = 600.0

# How old the oldest source may be before a boot decides to refresh. A day,
# because that is roughly the cadence the sources themselves move at: ADP
# drifts daily in August, and nflverse's weekly file lands once a week.
MAX_AGE_ENV = "REFRESH_MAX_AGE_HOURS"
DEFAULT_MAX_AGE_HOURS = 24.0

# How long the loop waits before asking again. Not the refresh interval --
# `should_refresh` decides that. This is only how often the question is put,
# and it is deliberately much shorter than the age threshold so that a
# container which happens to boot an hour before staleness does not sit out
# the whole next day.
POLL_SECONDS = 3600.0

# How long to wait after a refresh raises before trying again. Long enough
# not to hammer a source that is down, short enough that a transient failure
# does not cost a day of freshness.
ERROR_BACKOFF_SECONDS = 900.0


def _read_meta(conn) -> pd.DataFrame:
    """The per-source freshness table. Its own function so the tests can
    answer the freshness question without a database."""
    cur = conn.cursor()
    try:
        return read_table(cur, "meta")
    finally:
        cur.close()


def should_refresh(conn, max_age_hours: float, now: dt.datetime | None = None) -> bool:
    """Whether this database is old enough to be worth rebuilding.

    THE OLDEST SOURCE DECIDES, not the newest. `pipeline.refresh` carries on
    past a source that fails, so "some tables current, one a week old" is an
    ordinary state rather than a corrupt one -- and reading the newest would
    call that database fresh forever.

    A row with no timestamp is a source that has a `meta` row and has never
    completed. That counts as never refreshed, for the same reason: treating
    it as current leaves a half-built database that never finishes.
    """
    meta = _read_meta(conn)
    if meta.empty or "refreshed_at" not in meta.columns:
        return True
    stamps = pd.to_datetime(meta["refreshed_at"], errors="coerce")
    if stamps.isna().any():
        return True
    oldest = stamps.min()
    if pd.isna(oldest):
        return True
    when = pd.Timestamp(now or dt.datetime.now())
    return (when - oldest) > pd.Timedelta(hours=max_age_hours)


def refresh_once(conn, max_age_hours: float, run=None, now=None,
                 on_error=None) -> bool:
    """One decision and, if it says so, one refresh. Returns whether it ran.

    A REFRESH THAT THROWS MUST NOT END THE LOOP that calls this. This loop is
    the only thing that will ever refresh a deployed instance; if a source
    timing out could kill it, the site would quietly stop updating and the
    only symptom would be data that slowly gets older. So the exception is
    caught here, reported, and the caller carries on.
    """
    if not should_refresh(conn, max_age_hours, now):
        return False
    runner = run if run is not None else _run_refresh
    try:
        runner()
    except Exception as exc:      # noqa: BLE001 -- see the docstring: losing
        if on_error is not None:  # a refresh is a bad day, losing the loop
            on_error(exc)         # is a bad month
        else:
            print(f"refresh failed, will retry: {exc!r}", flush=True)
        return False
    return True


def _run_refresh() -> None:
    """`make refresh`, in this process.

    Imported inside the function, not at module scope: `pipeline.refresh`
    pulls in every source module and nfl_data_py behind them, which is real
    import cost on a path that a deployment with the switch off never takes.

    It opens its own DuckDB connection. That is safe here and only here --
    same process, so DuckDB hands back the cached instance rather than
    fighting the app's own connection for the file lock.
    """
    from pipeline import refresh
    refresh.main()

    # FOLD THE WRITES INTO THE FILE. DuckDB appends a commit to `<name>.wal`
    # and only merges it at a checkpoint, which a connection held for the
    # life of the process may not reach for hours. Everything that reads this
    # database as a FILE rather than through this connection sees only what
    # has been folded in -- and the farm is exactly that reader: it copies
    # the file when it cannot open it (`mock_farm.open_board_db`).
    #
    # That copy now brings the WAL along, so this is belt and braces rather
    # than the fix. It is still worth doing: it bounds how far the file can
    # lag behind reality, which matters for anything that copies the volume
    # -- a backup, a download, a support request asking for the database.
    #
    # Failure here is not a failed refresh. The data is committed either way;
    # an un-checkpointed database is merely one whose recent history is in
    # the sidecar.
    try:
        from pipeline.db import get_conn
        get_conn().execute("CHECKPOINT")
    except Exception as exc:      # noqa: BLE001 -- see above
        print(f"refresh: could not checkpoint ({exc!r}) -- "
              f"data is committed, the file just lags its WAL", flush=True)


def _refresh_loop(conn, max_age_hours: float, ready=None) -> None:
    while True:
        failed = []
        refresh_once(conn, max_age_hours, on_error=failed.append)
        # THE FARM IS WAITING ON THIS. Set after the first pass whatever the
        # outcome: a refresh that failed still leaves whatever data was
        # already on the volume, and holding the farm back forever because
        # one source 500'd would be a worse failure than farming on
        # yesterday's board. Set once; `Event.set` on an already-set event is
        # a no-op.
        if ready is not None:
            ready.set()
        time.sleep(ERROR_BACKOFF_SECONDS if failed else POLL_SECONDS)


def _run_farm(batch: int) -> None:
    """`make farm-mocks`, as a CHILD PROCESS. Not a call into `mock_farm`.

    THIS IS THE ONE JOB THAT CANNOT SHARE THIS PROCESS, and the reason is the
    opposite of the reason it cannot be a separate service.

    DuckDB caches one database instance per file per process, and every
    connection to that file must then agree on the configuration. This app
    holds `nfl.duckdb` read-write for its whole life; the farm wants the same
    file `read_only=True` (`mock_farm.open_board_db`) and the corpus
    read-write while `api/market` reads it read-only. In one process both are
    an outright refusal:

        ConnectionException: Can't open a connection to same database file
        with a different configuration than existing connections

    `open_board_db` already solves the cross-process version of this -- it
    catches the lock error and reads a snapshot copy, which is exactly how
    several farms run alongside `make up` on a developer's machine -- but its
    guard tests for the word "lock", and the same-process message does not
    contain it. Rather than widen that guard, this runs the farm the way it
    is already proven to run: its own process, against the same volume.

    A separate SERVICE is still impossible (a platform volume attaches to one
    service). A separate process inside this container is free.

    NO SEED. The local Makefile passes one so a person can reproduce a run; a
    server looping forever wants a different room each time, and `mock_farm`
    treats seed 0 as "pick for yourself".
    """
    import subprocess
    import sys

    # `check=True` so a non-zero exit reaches the supervisor as an exception
    # and takes the backoff, rather than looking like a completed pass and
    # spinning. Output is inherited so the farm's own log -- which is already
    # credential-scrubbed by `pipeline.redact` -- lands in the platform's.
    subprocess.run([sys.executable, "-u", "-m", "pipeline.mock_farm",
                    str(batch)], check=True)


def farm_once(run=None, batch: int = DEFAULT_FARM_BATCH, on_error=None) -> bool:
    """One pass of the farm. Returns whether it completed without raising.

    THE SAME RULE AS THE REFRESH: an exception must not end the loop. The
    farm's failures are mostly environmental and mostly temporary -- a room
    torn down between the directory listing it and the join, ESPN 500ing, a
    websocket dropped mid-draft -- and every one of them is a reason to try
    another room rather than to stop farming until somebody redeploys.

    THE ONE FAILURE THAT IS NOT TEMPORARY is an expired ESPN login, which
    surfaces as a RuntimeError from `load_cookies` naming the state file.
    The loop cannot fix that by waiting, but it also must not exit: exiting
    would take the thread down silently, and a thread that is gone looks
    exactly like a thread that is idle. It backs off long instead, so a
    re-uploaded state file is picked up without a redeploy.
    """
    runner = run if run is not None else _run_farm
    try:
        runner(batch)
    except Exception as exc:      # noqa: BLE001 -- see the docstring
        if on_error is not None:
            on_error(exc)
        else:
            print(f"farm pass failed, will retry: {exc!r}", flush=True)
        return False
    return True


def _farm_loop(batch: int, ready=None) -> None:
    """Farm forever, but not before there is a board to farm with.

    THE RACE THIS EXISTS FOR, measured in production on the first deploy.
    A fresh volume has an empty `nfl.duckdb`. The farm starts, finds the
    file locked by this very process, and takes the snapshot copy that
    `mock_farm.open_board_db` is designed to fall back to -- except the file
    it copies has nothing in it yet, because the refresh is still running.
    The log line is exact:

        board built: 0 pool players, 0 selectable

    and the snapshot is taken ONCE PER PROCESS, so that farm never recovers.
    It goes on joining real ESPN rooms it cannot pick in, letting ESPN
    autodraft the seat, for as long as the container lives.

    Waiting costs one refresh at the start of a deployment's life. Not
    waiting costs every room joined before the data lands.
    """
    if ready is not None:
        ready.wait()
    while True:
        ok = farm_once(batch=batch)
        time.sleep(FARM_IDLE_SECONDS if not ok else 0)


def _spawn(name: str, fn) -> None:
    threading.Thread(target=fn, name=f"job-{name}", daemon=True).start()


def start_jobs(conn, spawn=None) -> list:
    """Start whatever this environment has switched on. Returns their names.

    CALLED AFTER THE APP IS BUILT, NOT BEFORE, and it never blocks: the first
    refresh takes minutes (`player_news` is ~223 queries), and a healthcheck
    that waited for it would fail the deploy and roll back a working image.
    The site comes up honestly empty and fills in behind the reader.
    """
    started = []
    launch = spawn if spawn is not None else _spawn
    # The farm's ESPN login, if it was handed over as a variable. Before the
    # farm starts and not inside it: a login that arrives after the loop has
    # already given up would sit unused until the next redeploy.
    if _on(FARM_ENV):
        from api.seed_state import write_state
        write_state()

    # The gate between the two jobs. Set when there is data worth farming
    # against -- see `_farm_loop` for the production failure that made this
    # necessary. With no refresh configured it is open from the start: this
    # instance's data came from somewhere else and is not this loop's to
    # wait for.
    ready = threading.Event()
    if not _on(REFRESH_ENV):
        ready.set()

    if _on(REFRESH_ENV):
        max_age = _hours(MAX_AGE_ENV, DEFAULT_MAX_AGE_HOURS)
        launch("refresh", lambda: _refresh_loop(conn, max_age, ready))
        started.append("refresh")
    if _on(FARM_ENV):
        batch = int(_hours(FARM_BATCH_ENV, DEFAULT_FARM_BATCH))
        for i in range(_concurrency()):
            # Numbered, because the whole point is that there are several and
            # a log line has to say which one is speaking.
            name = f"farm-{i + 1}"
            launch(name, lambda b=batch: _farm_loop(b, ready))
            started.append(name)

    # LAST, and the one job with no environment switch in front of it: it
    # costs a single build of data this instance already holds, and nobody
    # is waiting on it. A cold process has an empty board cache
    # (scoring/board_cache.py), so the FIRST reader after a deploy pays 1.6s
    # for the board and 1.9s more for the profile frames, on a click. Doing
    # it here moves that onto a thread nobody is watching.
    #
    # Only when there is something to build FROM. A volume that has never
    # been refreshed has no `meta` rows and no `weekly` either, so warming it
    # would cache an empty board under a key that the first real refresh
    # immediately retires -- work for nothing, and a misleading log line. The
    # refresh loop above will warm it when it finishes (see
    # pipeline/refresh.main).
    if _has_data(conn):
        launch("warm-caches", lambda: _warm_caches(conn))
        started.append("warm-caches")
    return started


def _warm_caches(conn) -> None:
    """`board_cache.warm`, on its own cursor.

    A cursor rather than the connection itself, for the reason every request
    handler in api/main.py takes one: this runs alongside the refresh loop,
    which is writing. The cache key sees through it either way -- `_db_key`
    is `PRAGMA database_list`, which a cursor answers identically to the
    connection it came from -- so the frames warmed here are the same entries
    the request threads go on to hit.

    Imported inside the function: importing the scoring stack costs real time
    at start-up, and `start_jobs` is called while the app is coming up.
    """
    from scoring import board_cache
    cur = conn.cursor()
    try:
        board_cache.warm(cur)
    finally:
        cur.close()


def _has_data(conn) -> bool:
    """Whether this database has been refreshed at all -- any `meta` row.

    Anything that goes wrong reading it is a no: the fake connections in
    tests/test_jobs.py cannot answer this question at all, and neither can a
    database mid-creation. "Cannot tell" and "nothing there" get the same
    answer because they lead to the same decision, and the cost of being
    wrong is one cold request rather than a failed boot.
    """
    try:
        return not _read_meta(conn).empty
    except Exception:  # noqa: BLE001 -- see the docstring
        return False


def _concurrency() -> int:
    """How many farms to run, clamped.

    Clamped at both ends rather than trusted: a zero would switch the farm on
    and then run none of it, which reads in a settings page as "farming" and
    behaves as "not farming"; and an accidental extra digit would put this
    deployment in ninety mock drafts at once from one address.
    """
    want = int(_hours(FARM_CONCURRENCY_ENV, DEFAULT_FARM_CONCURRENCY))
    return max(1, min(want, MAX_FARM_CONCURRENCY))


def _on(name: str) -> bool:
    """A switch is on for "1", "true", "yes" -- and off for anything else,
    including the empty string a platform writes for a variable somebody
    added and left blank."""
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _hours(name: str, fallback: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        value = float(raw)
    except ValueError:
        return fallback
    # A zero or negative threshold would mean "refresh on every poll", which
    # is a way to spend a day of API quota by typing 0 into a settings box.
    return value if value > 0 else fallback
