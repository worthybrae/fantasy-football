"""One Postgres pool for every store that has a Postgres backend.

WHY THIS EXISTS. Three pieces of per-user state -- ESPN credential custody,
Stripe entitlements, and the live-draft session record -- were each a DuckDB
file (or, for the session record, a JSON file) next to the draft database.
That is exactly right for one process on one laptop and wrong for several
processes on several machines: DuckDB takes a single-writer lock per file, so
a second worker does not get a slower store, it gets an unusable one.

SELECTED BY ONE VARIABLE, `SUPABASE_DB_URL`. With it set, those stores read
and write Postgres. Absent -- a checkout, the test suite, the owner's own
machine -- `enabled()` is False and every store keeps the file it has always
used. That is the property worth protecting: nothing in this repository
should need a network to run its tests, and no laptop should need a database
server to open a draft.

WHY A POOL RATHER THAN A CONNECTION. The opposite of the DuckDB problem: a
Supabase session pooler will happily hand out connections but there is a
ceiling on them, and a FastAPI worker that opened one per request would spend
most of a draft night in TLS handshakes. `psycopg_pool` keeps a handful open
and hands them round. `autocommit` because every store here writes single
statements that stand alone -- an implicit transaction left open across a
request is a lock held for as long as the slowest handler.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone

DSN_ENV = "SUPABASE_DB_URL"

_lock = threading.Lock()
_pool = None


class StoreError(Exception):
    """A store could not be reached or refused the operation.

    The one exception type a caller has to catch, whichever backend is in
    play: the DuckDB backends wrap their own driver errors into it, so an API
    handler answers 503 without knowing or caring which store it is talking
    to.
    """


def dsn() -> str | None:
    """The configured DSN, or None. Read fresh every time rather than cached,
    so a test can set and unset the variable without a module reload."""
    value = os.environ.get(DSN_ENV, "").strip()
    return value or None


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
                raise StoreError(f"{DSN_ENV} is not set")
            from psycopg_pool import ConnectionPool
            _pool = ConnectionPool(value, min_size=1, max_size=8, open=True,
                                   timeout=10, kwargs={"autocommit": True})
        return _pool


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
