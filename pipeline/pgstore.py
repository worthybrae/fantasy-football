"""One Postgres pool for every store that has a Postgres backend.

WHY THIS EXISTS. Three pieces of per-user state -- ESPN credential custody,
Stripe entitlements, and the live-draft session record -- were each a DuckDB
file (or, for the session record, a JSON file) next to the draft database.
That is exactly right for one process on one laptop and wrong for several
processes on several machines: DuckDB takes a single-writer lock per file, so
a second worker does not get a slower store, it gets an unusable one.

SELECTED BY THE ENVIRONMENT, `SUPABASE_DB_URL` first. With a Postgres DSN
there -- or a project URL and a password, from which one is composed -- those
stores read and write Postgres. Absent, or set to something that is not a
Postgres URL -- a checkout, the test suite, the owner's own machine --
`enabled()` is False and every store keeps the file it has always used. That
is the property worth protecting: nothing in this repository should need a
network to run its tests, and no laptop should need a database server to open
a draft.

WHY A POOL RATHER THAN A CONNECTION. The opposite of the DuckDB problem: a
Supabase session pooler will happily hand out connections but there is a
ceiling on them, and a FastAPI worker that opened one per request would spend
most of a draft night in TLS handshakes. `psycopg_pool` keeps a handful open
and hands them round. `autocommit` because every store here writes single
statements that stand alone -- an implicit transaction left open across a
request is a lock held for as long as the slowest handler.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlsplit

DSN_ENV = "SUPABASE_DB_URL"

# THE OTHER THREE, WHICH EXIST BECAUSE OF WHAT SUPABASE PUTS ON SCREEN. A
# Supabase project page offers a "Project URL" (`https://<ref>.supabase.co`,
# the PostgREST endpoint) far more prominently than it offers a connection
# string, and a database password that is shown once at creation and never
# again next to a DSN. Somebody wiring up a deployment reaches for the two
# things the dashboard hands them. So either spelling works: a real DSN in
# `SUPABASE_DB_URL`, or the project URL plus the password, from which the
# pooler DSN is composed below.
PROJECT_URL_ENV = "SUPABASE_URL"
PASSWORD_ENV = "SUPABASE_DB_PASSWORD"
REGION_ENV = "SUPABASE_DB_REGION"

# Where the project's pooler lives. Not derivable from the project URL --
# `<ref>.supabase.co` resolves the same wherever the database sits -- so it
# has to be told, and this is the region the owner's project is in.
DEFAULT_REGION = "us-west-2"

# psycopg accepts both spellings and so does everything else; `postgres://`
# is what several hosts still print.
POSTGRES_SCHEMES = ("postgresql://", "postgres://")

log = logging.getLogger(__name__)

_lock = threading.Lock()
_pool = None

# Said once per process, keyed by a short tag rather than by the value: these
# lines are read by whoever is looking at a boot log wondering why a store is
# in the wrong backend, and one line is the whole point. NOT guarded by
# `_lock`: `pool()` calls `dsn()` while holding it, and a second acquire of a
# plain Lock is a deadlock. Two threads racing here print the line twice,
# which costs nothing.
_said: set = set()

# Called when the pool is dropped. Every store that creates its schema once
# per process keeps a "already done" flag, and a flag that outlives the pool
# it was set against is a store that never creates its tables again -- which
# is wrong for a test that closes the pool between cases, and wrong for a
# process that reconnects to a database somebody has since rebuilt.
_on_close: list = []


class StoreError(Exception):
    """A store could not be reached or refused the operation.

    The one exception type a caller has to catch, whichever backend is in
    play: the DuckDB backends wrap their own driver errors into it, so an API
    handler answers 503 without knowing or caring which store it is talking
    to.
    """


def _say_once(tag: str, level: int, message: str) -> None:
    if tag not in _said:
        _said.add(tag)
        log.log(level, message)


def _project_ref(url: str) -> str | None:
    """The `<ref>` out of `https://<ref>.supabase.co`.

    Tolerant about the scheme and a trailing path, because what gets pasted
    into a host's variable editor is whatever was on the clipboard: with the
    scheme, without it, with the trailing slash the dashboard's copy button
    includes.
    """
    url = url.strip().rstrip("/")
    if not url:
        return None
    host = urlsplit(url).netloc or url.split("/", 1)[0]
    host = host.rpartition("@")[2].partition(":")[0]      # no creds, no port
    ref = host.split(".")[0].strip()
    return ref or None


def _composed() -> str | None:
    """The session-pooler DSN built from a project URL and a password.

    The port is 5432, the session pooler, for the reason the README gives:
    psycopg's prepared statements do not survive the 6543 transaction pooler.
    The password is URL-encoded because Supabase generates passwords with
    punctuation in them, and an unescaped `@` or `#` in a DSN does not fail
    loudly -- it silently redraws where the host starts, or truncates the
    password at the fragment.
    """
    password = os.environ.get(PASSWORD_ENV, "")
    if not password:
        return None
    # Either variable may hold the project URL: `SUPABASE_URL` because that is
    # its name, and `SUPABASE_DB_URL` because that is where it has actually
    # been put. The purpose-named one wins if both are set.
    ref = (_project_ref(os.environ.get(PROJECT_URL_ENV, ""))
           or _project_ref(os.environ.get(DSN_ENV, "")))
    if ref is None:
        return None
    region = os.environ.get(REGION_ENV, "").strip() or DEFAULT_REGION
    host = f"aws-0-{region}.pooler.supabase.com"
    # The host, and nothing else. The password is in the string being built.
    _say_once("composed", logging.INFO, f"composed a Supabase DSN for {host}")
    return (f"postgresql://postgres.{ref}:{quote_plus(password)}"
            f"@{host}:5432/postgres?sslmode=require")


def dsn() -> str | None:
    """The DSN to connect with, or None. Read fresh every time rather than
    cached, so a test can set and unset the variables without a module reload.

    ONLY EVER A POSTGRES URL. `SUPABASE_DB_URL` was once handed straight to
    psycopg, which meant that setting it to the project URL -- the thing the
    Supabase dashboard shows first, and an honest reading of the name -- got
    every custody, billing and live-session call a driver error at the far end
    of a boot rather than an answer here. A value that is not a
    `postgresql://` or `postgres://` URL is therefore not a DSN: it is said
    once and treated as unset, and if it looks like a project URL it is
    offered to the composition above instead. Never raised on, because the
    fallback for "no Postgres" is the DuckDB file every store already has, and
    an exception here would take down a process that would otherwise run.
    """
    configured = os.environ.get(DSN_ENV, "").strip()
    if configured.startswith(POSTGRES_SCHEMES):
        return configured
    if configured:
        # The value is not repeated. It is a URL rather than a secret today,
        # but this variable is the one people paste passwords into.
        _say_once("not-a-dsn", logging.WARNING,
                  f"{DSN_ENV} is not a postgres:// URL; ignoring")
    return _composed()


def enabled() -> bool:
    return dsn() is not None


def pool():
    """The process-wide pool, opened on first use.

    Lazy, and the driver is imported lazily with it: a process that never
    touches Postgres never opens a socket and never pays for the import.
    """
    global _pool
    with _lock:
        if _pool is None:
            value = dsn()
            if value is None:
                raise StoreError(
                    f"no Postgres DSN: set {DSN_ENV} to a postgres:// URL, "
                    f"or {PROJECT_URL_ENV} with {PASSWORD_ENV}")
            from psycopg_pool import ConnectionPool
            # `timeout` is how long a caller waits for a free connection,
            # not how long a query may run. Five seconds because the caller is
            # usually a request that has a user waiting on it: a pool with
            # nothing free is a store that is already failing, and taking ten
            # seconds to say so just holds a worker open for twice as long.
            _pool = ConnectionPool(value, min_size=1, max_size=8, open=True,
                                   timeout=5, kwargs={"autocommit": True})
        return _pool


def create_schema(statements, label: str) -> None:
    """Run each DDL statement once, and once more if it lost a race.

    `CREATE TABLE IF NOT EXISTS` IS NOT ATOMIC IN POSTGRES. Two workers
    booting together both find the table missing and both create it; the
    loser does not get the "already there, nothing to do" the statement
    asks for, it gets a UniqueViolation on the system catalogue (two rows
    for one type name) or a DuplicateTable. That was one 503 on a cold
    deploy, healed by the next request -- a self-inflicted error page every
    time the platform started two workers at once.

    Retried once, per statement, because by the time the loser tries again
    the winner has committed and IF NOT EXISTS is finally true. A second
    failure is a real one. `label` names the store in the error a caller
    would see, and the driver's message is left out for the reason every
    other store here leaves it out: it carries the DSN.
    """
    import psycopg

    raced = (psycopg.errors.UniqueViolation, psycopg.errors.DuplicateTable,
             psycopg.errors.DuplicateObject)
    for statement in statements:
        for attempt in (1, 2):
            try:
                with pool().connection() as conn:
                    conn.execute(statement)
                break
            except raced as exc:
                if attempt == 2:
                    raise StoreError(f"the {label} is not usable "
                                     f"({type(exc).__name__})") from None
            except psycopg.Error as exc:
                raise StoreError(f"the {label} is not usable "
                                 f"({type(exc).__name__})") from None


def on_close(callback) -> None:
    """Run `callback` the next time (and every time) the pool is dropped.

    The one thing this is for: schema flags. See `_on_close`.
    """
    with _lock:
        if callback not in _on_close:
            _on_close.append(callback)


# -- the timestamp contract --------------------------------------------------
#
# Every store here is written in naive UTC, because that is what a DuckDB
# TIMESTAMP column holds and what the reaper's arithmetic assumes. Postgres
# has a real TIMESTAMPTZ and should use it -- a naive value sent to one is
# read in whatever the session's TimeZone happens to be, which is an expiry
# that is wrong by an offset nobody wrote down. So the zone is attached on the
# way in and taken off again on the way out, in one place, and no caller can
# tell which store answered.


def to_pg(value):
    """Naive UTC becomes aware UTC, on its way into a TIMESTAMPTZ column."""
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def from_pg(value):
    """And back to naive UTC, which is what every caller here expects."""
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def close() -> None:
    """Drop the pool. For tests, and for a process that wants to release its
    connections without exiting."""
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None
        callbacks = list(_on_close)
    # Outside the lock: a callback is another module's teardown and may well
    # ask this one whether it is enabled.
    for callback in callbacks:
        try:
            callback()
        except Exception:      # noqa: BLE001 -- these are schema flags being
            # reset. One store failing to forget its own is not a reason for
            # the next store never to hear that the pool went away, and this
            # runs on a shutdown path where there is nobody left to tell.
            pass
