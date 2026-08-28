"""Money, and the single thing it buys.

WHAT IS SOLD. One real league's draft, for one season, for $9.99. Mock drafts
are free and stay free -- that is the promise the landing page already makes
("Free in mock drafts, $9.99 when you draft for real"), and it is also the
only honest split: a mock is where somebody finds out whether this tool helps
them, and charging for the trial would be charging for the sales pitch.

WHO IS BUYING, which is the hard part and the reason this module is shaped
the way it is. This product has no accounts. It has never asked for an email
or a password, and `pipeline/credentials.py` goes to some length to make sure
the ESPN sessions it holds cannot be enumerated even by somebody standing
over the database file. Stripe, meanwhile, needs a customer, and a customer
needs an email.

The join is the HMAC that credential store already computes. An entitlement
row is keyed by `store.account_id(swid)` -- the same keyed hash that indexes
the credential table, so:

  * the row says nothing about who the person is. Without the custody key it
    cannot even be tested for membership, exactly like the credential rows;
  * the email lives on Stripe's side, where a receipt can be sent from, and
    never on ours;
  * an account that connects from a second browser is the same buyer, because
    the id is derived from the ESPN account rather than from the cookie.

WHY A SEPARATE DATABASE FILE from custody. The credential store's whole claim
is that a stolen copy of it is inert. Purchases are a different kind of
secret with a different disclosure story (they are in Stripe too, under a
real name and email), and mixing a table of Stripe ids into that file would
weaken a sentence worth keeping true. Nothing here can decrypt anything
there; it only borrows the id.

BILLING OFF IS THE DEFAULT AND MUST STAY WORKING. With no `STRIPE_SECRET_KEY`
in the environment -- a checkout on somebody's laptop, the test suite, the
owner's own machine -- `enabled()` is False, every draft is free, and the
gate below is a no-op. Turning payment on is one variable on one deployment,
not a code path anybody else has to think about.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from urllib.parse import urlparse

import duckdb
from fastapi import HTTPException, Request
from pydantic import BaseModel

from api.custody import custody_for, _store as _custody_store
from pipeline import pgstore
from scoring.config import CURRENT_SEASON

# -- configuration -----------------------------------------------------------

# The key. `rk_` (a restricted key) is what belongs here rather than `sk_`:
# this integration creates Checkout Sessions and reads nothing else, so a key
# scoped to exactly that is a key worth far less to whoever steals it.
KEY_ENV = "STRIPE_SECRET_KEY"

# The Price to sell, made in the Stripe Dashboard rather than by this code.
# A price is a business decision with tax and reporting attached; a program
# that creates one on boot is a program that can quietly create a second one.
PRICE_ENV = "STRIPE_PRICE_ID"

# The webhook signing secret. Without it the webhook refuses every request:
# an unverified webhook is an endpoint that grants paid access to anybody who
# can POST JSON at it.
WEBHOOK_SECRET_ENV = "STRIPE_WEBHOOK_SECRET"

# Where Stripe sends the buyer back to. Needed because the app sits behind a
# proxy that terminates TLS, so the request's own base URL says `http://` and
# Stripe rejects a plaintext return URL in live mode.
BASE_URL_ENV = "PUBLIC_BASE_URL"

# Entitlements and the webhook's own event log. Relative by default, which
# puts it on the volume: the Dockerfile mounts that at `/app/data` precisely
# so paths like this one need no environment variable to survive a redeploy.
DB_PATH_ENV = "BILLING_DB_PATH"
DEFAULT_DB_PATH = "data/billing.duckdb"

# Pinned, not floating. A Stripe API version changes response shapes, and a
# deployment that silently follows the newest one is a deployment whose
# webhook can start failing on a day nobody touched it.
API_VERSION = "2026-07-29.dahlia"

# Tags the sessions this integration creates, so the Dashboard can tell them
# apart from any other flow that ever gets built. The suffix is fixed rather
# than generated per call: it identifies the INTEGRATION, and a new one per
# session would make the grouping useless.
INTEGRATION_ID = "draftassist-hnwqkzrb"

# THE ONE STORAGE FAILURE TYPE, whichever backend is underneath, and the same
# class `pipeline/credentials.py` re-exports. Both adapters below wrap their
# driver's errors into it, so the routes can answer 503 without knowing which
# store they were talking to.
StoreError = pgstore.StoreError

_lock = threading.Lock()
_store = None
# Whether the corpus seed has been CLAIMED in this process. Separate from
# `_store` because the seed happens OUTSIDE `_lock` -- see `_db`.
_seeded = False
# Set when the claimed seed has finished (or given up). `_db` does not wait on
# it -- that is the whole point -- but `reset_for_tests` does, so a suite that
# swaps the store out from under a running seed does not race it, and so does
# any test that wants to assert on what the seed wrote.
_seed_done = threading.Event()
_seed_done.set()
# How long `reset_for_tests` gives an in-flight seed to finish. Generous
# enough that it is never the reason a suite fails, short enough that a seed
# wedged on a store that has gone away does not hang the run.
_SEED_WAIT_SECONDS = 30

# Mock league ids already known to the table, so the hot path (a connect
# asking "is this free") is a set lookup rather than a query.
_known_mocks: set = set()


def enabled() -> bool:
    """Whether this instance sells anything at all.

    Read from the environment on every call rather than captured at import:
    the test suite turns billing on and off around individual tests, and a
    value frozen at import time would make that impossible without reloading
    the module.
    """
    return bool(os.environ.get(KEY_ENV, "").strip())


def _price_id() -> str:
    price = os.environ.get(PRICE_ENV, "").strip()
    if not price:
        raise HTTPException(
            status_code=503,
            detail=f"billing is switched on but {PRICE_ENV} is not set")
    return price


# -- the store ---------------------------------------------------------------


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS entitlement (
        account_id VARCHAR NOT NULL,
        league_id VARCHAR NOT NULL,
        season INTEGER NOT NULL,
        granted_at TIMESTAMP NOT NULL,
        revoked_at TIMESTAMP,
        checkout_session VARCHAR,
        payment_intent VARCHAR,
        PRIMARY KEY (account_id, league_id, season))""",
    # WEBHOOK IDEMPOTENCY. Stripe retries on any non-2xx and can deliver the
    # same event twice on its own; every handler below is gated on this table
    # so a replay is a no-op rather than a second grant (or, for a refund, a
    # second revoke over a fresh purchase).
    """CREATE TABLE IF NOT EXISTS billing_event (
        event_id VARCHAR PRIMARY KEY,
        type VARCHAR,
        seen_at TIMESTAMP NOT NULL)""",
    # Rooms we seated somebody in ourselves (see `note_mock_room`). Here
    # rather than anywhere else because its only job is to answer a billing
    # question: is this draft one of the free ones.
    """CREATE TABLE IF NOT EXISTS mock_room (
        league_id VARCHAR PRIMARY KEY,
        seen_at TIMESTAMP NOT NULL)""",
    # The players an account has starred. HERE, in the entitlement store,
    # rather than in a fourth database, because it is keyed by exactly the
    # same thing an entitlement is -- `store.account_id(swid)` -- and a
    # preference that could not be read in the same round trip as the
    # purchase would be a second connection for no gain. It says as little
    # about the person as the entitlement row does: an account id nobody can
    # test for membership without the custody key, and a list of public
    # player ids.
    #
    # `ord` is the order the user put them in, not a rank we computed: the
    # picker is an ordered list and the draft plan reads it that way, so the
    # position has to survive the round trip. Zero-based, dense.
    """CREATE TABLE IF NOT EXISTS favorite_player (
        account_id VARCHAR NOT NULL,
        player_id VARCHAR NOT NULL,
        ord INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL,
        PRIMARY KEY (account_id, player_id))""",
)

# The same four tables in Postgres spellings, derived rather than written out
# again so that adding a column cannot leave the two backends holding
# different tables. TIMESTAMPTZ for the same reason pipeline/pgstore.to_pg
# exists: these are audit columns, and one that is silently wrong by a session
# offset is worse than no column at all.
_PG_SCHEMA = tuple(
    statement.replace(" VARCHAR", " TEXT").replace(" TIMESTAMP", " TIMESTAMPTZ")
    for statement in _SCHEMA)


class _Duck:
    """The billing file. One connection for the process, as DuckDB requires:
    it takes a single-writer lock per file, and two connections from two
    threads are two chances to meet it."""

    def __init__(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = duckdb.connect(path)
        for statement in _SCHEMA:
            self._conn.execute(statement)

    # EVERY STATEMENT GOES THROUGH A CURSOR, not through `self._conn`.
    #
    # A DuckDB connection holds the result of the last statement run on it.
    # Two threads sharing one connection therefore share one result slot: A
    # executes, B executes, A fetches -- and A gets B's rows, or none.
    # Measured on this class before the change: four threads reading two
    # accounts, seven empty answers in six thousand reads. Empty is the
    # dangerous shape, because every read here means "nothing stored" by that
    # answer -- an account with favourites is shown the onboarding picker, and
    # a paying customer could be told they have not paid.
    #
    # `cursor()` hands out a separate connection to the same database with its
    # own result slot, which is what `api/main.py` takes one of per request
    # and what `transaction` below already did. The connection stays a single
    # one per file because that part IS required: DuckDB takes a per-file
    # write lock per process.

    def execute(self, sql: str, params=()) -> list:
        cursor = self._conn.cursor()
        try:
            return cursor.execute(sql, list(params)).fetchall()
        except duckdb.Error as exc:
            raise StoreError(f"the entitlement store refused a statement "
                             f"({type(exc).__name__})") from None
        finally:
            cursor.close()

    def executemany(self, sql: str, rows) -> None:
        cursor = self._conn.cursor()
        try:
            cursor.executemany(sql, [list(row) for row in rows])
        except duckdb.Error as exc:
            raise StoreError(f"the entitlement store refused a statement "
                             f"({type(exc).__name__})") from None
        finally:
            cursor.close()

    def transaction(self, statements) -> None:
        """Several `(sql, params)` pairs, all of them or none.

        On its own cursor like everything else here, and for a second reason
        on top of the shared result slot above: a transaction is state on the
        connection too, so two threads opening one on `self._conn` would be a
        `BEGIN` inside a `BEGIN` -- which DuckDB refuses -- and a rollback by
        either would discard the other's work.

        The `finally` is what actually makes this all-or-nothing, whatever
        goes wrong: closing a cursor with an open transaction discards it, so
        even a failure this `except` does not name -- a bad parameter type,
        an interrupt -- cannot leave half a replace behind. The explicit
        ROLLBACK is there to release the write lock at the moment of the
        failure rather than at the close.
        """
        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")
            for sql, params in statements:
                cursor.execute(sql, list(params))
            cursor.execute("COMMIT")
        except duckdb.Error as exc:
            try:
                cursor.execute("ROLLBACK")
            except duckdb.Error:      # already rolled back by the failure
                pass
            raise StoreError(f"the entitlement store refused a statement "
                             f"({type(exc).__name__})") from None
        finally:
            cursor.close()

    def close(self) -> None:
        self._conn.close()


class _Pg:
    """The same four tables in Postgres, shared by every process with the DSN.

    Which is the entire point of this backend: the file above is correct for
    one process and unusable for two, and an entitlement that only one worker
    can read is a customer who paid and still gets a 402.
    """

    def __init__(self):
        # Imported here rather than at module scope so a checkout with no DSN
        # never needs the driver at all.
        import psycopg

        self._error = psycopg.Error
        self._ensure_schema()

    def execute(self, sql: str, params=()) -> list:
        # `?` becomes `%s`. The SQL below is written once for both backends,
        # and no statement in this module contains a literal question mark.
        try:
            with pgstore.pool().connection() as conn:
                cursor = conn.execute(
                    sql.replace("?", "%s"),
                    [pgstore.to_pg(p) for p in params] or None)
                if cursor.description is None:
                    return []
                return [tuple(pgstore.from_pg(value) for value in row)
                        for row in cursor.fetchall()]
        except self._error as exc:
            raise StoreError(f"the entitlement store is not usable "
                             f"({type(exc).__name__})") from None

    def executemany(self, sql: str, rows) -> None:
        try:
            with pgstore.pool().connection() as conn:
                conn.cursor().executemany(
                    sql.replace("?", "%s"),
                    [[pgstore.to_pg(p) for p in row] for row in rows])
        except self._error as exc:
            raise StoreError(f"the entitlement store is not usable "
                             f"({type(exc).__name__})") from None

    def transaction(self, statements) -> None:
        """Several `(sql, params)` pairs, all of them or none.

        One pooled connection for the lot: psycopg's context manager opens a
        transaction on entry, commits it on a clean exit and rolls it back on
        any exception, so "all or none" needs no BEGIN of our own here.
        """
        try:
            with pgstore.pool().connection() as conn:
                for sql, params in statements:
                    conn.execute(sql.replace("?", "%s"),
                                 [pgstore.to_pg(p) for p in params] or None)
        except self._error as exc:
            raise StoreError(f"the entitlement store is not usable "
                             f"({type(exc).__name__})") from None

    def _ensure_schema(self) -> None:
        """Create the tables once per process, not once per adapter.

        `reset_for_tests` builds a fresh adapter, and on Postgres the tables it
        would create are already there -- four round trips to say so.
        """
        global _PG_READY
        if _PG_READY:
            return
        with _PG_LOCK:
            if _PG_READY:
                return
            # Through pgstore rather than `self.execute`, for the race two
            # workers booting together run into. See pgstore.create_schema.
            pgstore.create_schema(_PG_SCHEMA, "entitlement store")
            _PG_READY = True

    def close(self) -> None:
        """The pool outlives any one store. See pipeline/pgstore.close()."""


_PG_READY = False
_PG_LOCK = threading.Lock()


def _forget_pg_schema() -> None:
    """The tables have to be created again against the next pool."""
    global _PG_READY
    with _PG_LOCK:
        _PG_READY = False


pgstore.on_close(_forget_pg_schema)


def _db():
    """The entitlement store, built once for the process.

    Two backends behind the same three methods, chosen by SUPABASE_DB_URL like
    every other store in this project. Nothing below this function knows which
    one it got: both take `?` placeholders and naive UTC datetimes, and both
    answer with a list of tuples.
    """
    global _store, _seeded
    with _lock:
        if _store is None:
            _store = (_Pg() if pgstore.enabled()
                      else _Duck(os.environ.get(DB_PATH_ENV) or DEFAULT_DB_PATH))
        store = _store
        seed = not _seeded
        _seeded = True
    if seed:
        # ON A THREAD, AND NOT ON THE CALLER. This used to run inline on
        # whichever request happened to be the first in the process to ask the
        # store anything, and on 2026-08-27 that request was a
        # `GET /api/account/favorites` on a freshly deployed container: the
        # Railway edge log has it at 24,026 ms, against 85 ms for the two
        # either side of it. The seed is boot work whose size is set by how
        # far the corpus has run ahead of the table -- it is not this request's
        # work in any sense, and no request should ever be able to discover
        # how big it is.
        #
        # Nothing is lost by not waiting. The seed was ALREADY racing every
        # other caller: `_seeded` is claimed under `_lock` and the work runs
        # outside it, so every concurrent `is_free_draft` already ran against
        # a table the seed had not finished filling. This only stops the one
        # caller that drew the short straw from paying for it too, and that
        # caller's fallback is the same one all the others have -- the lobby
        # directory, and a tie that goes to the drafter.
        _seed_done.clear()
        threading.Thread(target=_seed_mocks_from_corpus, args=(store,),
                         name="billing-seed-mocks", daemon=True).start()
    return store


def _seed_mocks_from_corpus(store) -> None:
    """Every mock draft ever recorded is a mock room, and the corpus knows.

    THE GAP THIS CLOSES. The two live tests -- rooms we seated somebody in,
    and rooms ESPN's directory currently lists -- miss the same case: a mock
    that has already closed. ESPN drops a room from the directory the moment
    it fills or starts, and the farm's own rooms never touch mock-join at
    all, so a reader connecting to one of those would be asked to pay for a
    free draft. That is the failure this module's free/paid rule is written
    to avoid.

    `draft_log` has the answer and has had it all along: every row with
    source `mock` is a mock, with its league id, going back to the first
    draft ever recorded. Read once per process, on a thread `_db` starts the
    first time anything asks the store a question -- see there for why it is
    not the asking caller that pays for it.

    Best-effort and read-only. The farm writes that file as drafts finish,
    and a lock held by it is not a reason to fail a request -- the seed is an
    improvement on two tests that still work without it.

    ONLY WHAT IS MISSING GOES IN. Every row already in `mock_room` is one
    this process wrote on a previous boot, and re-inserting all 854 of them
    on every restart is 854 round trips to say nothing. One SELECT names
    what is there, and the insert carries the difference -- usually none.
    When it is not none, it goes in a handful of statements rather than one
    per row: see `_insert_mock_rooms`.
    """
    try:
        try:
            from pipeline import draft_log as dl
            corpus = duckdb.connect(dl.CORPUS_PATH, read_only=True)
        except Exception:  # noqa: BLE001 -- no corpus yet, or the farm has it
            return
        try:
            rows = corpus.execute(
                "SELECT DISTINCT league_id FROM draft_log "
                "WHERE source = ? AND league_id IS NOT NULL",
                [dl.SOURCE_MOCK]).fetchall()
            if rows:
                have = {str(r[0]) for r in store.execute(
                    "SELECT league_id FROM mock_room")}
                missing = sorted({str(r[0]) for r in rows} - have)
                _insert_mock_rooms(store, missing)
        except Exception:  # noqa: BLE001 -- an older corpus without the
            pass           # column, a partial write: the live tests remain
        finally:
            corpus.close()
    finally:
        _seed_done.set()


# How many rooms go into one INSERT. The bound is Postgres's 65535 bind
# parameters and two per row; 500 leaves three orders of magnitude of headroom
# and still turns the whole backlog into a handful of statements.
_SEED_CHUNK = 500


def _insert_mock_rooms(store, league_ids) -> None:
    """The missing rooms, in as few statements as the backend will take.

    NOT `executemany`, WHICH IS WHERE THE 24 SECONDS WENT. Both adapters
    implement it the honest way -- one execution per row -- so 854 missing
    rooms was 854 statements, each its own auto-committed transaction and so
    its own fsync on Railway's network volume (or its own round trip to a
    Postgres in another region). Measured on this machine's local NVMe, where
    an fsync is at its cheapest: 854 rows cost 1.100 s row-at-a-time and
    0.008 s as one statement, and the same seed inside a booting app took
    4.3 s. A volume whose fsync is the ~28 ms a network disk charges puts
    that same 854 rows at ~24 s, which is what the edge log recorded.

    `set_favorites` already writes its list this way and says why in the same
    words: one statement with every row's values in it, so the cost is a
    property of the backend's latency and not of the length of the list.

    DEDUPLICATED FIRST, which `executemany` did not have to be. ESPN's mock
    directory can list the same room twice in one read, and one statement
    holding a key twice is a conflict with its own row rather than with a
    stored one -- `ON CONFLICT DO NOTHING` covers that on both backends, but
    only the version of the rule that stays true is worth relying on.
    """
    rooms = list(dict.fromkeys(str(i) for i in league_ids))
    now = _now()
    for start in range(0, len(rooms), _SEED_CHUNK):
        chunk = rooms[start:start + _SEED_CHUNK]
        params: list = []
        for league_id in chunk:
            params.extend([league_id, now])
        values = ", ".join("(?, ?)" for _ in chunk)
        store.execute(
            f"INSERT INTO mock_room VALUES {values} ON CONFLICT DO NOTHING",
            params)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def reset_for_tests(path: str | None = None) -> None:
    """Drop the cached store so a test can point at its own file.

    Waits for any seed still running first. `_db` starts that on a thread now,
    so without this a test that finished while the seed was mid-insert would
    close the connection under it -- swallowed, but as a write that silently
    did not happen and as a test whose neighbour's rows depend on timing.
    """
    global _store, _seeded
    _seed_done.wait(_SEED_WAIT_SECONDS)
    with _lock:
        if _store is not None:
            _store.close()
        _store = None
        _seeded = False
        _seed_done.set()
        _known_mocks.clear()
    if path is not None:
        os.environ[DB_PATH_ENV] = path


def _unavailable(exc) -> HTTPException:
    """The entitlement store could not answer, as a refusal a route can raise.

    503 rather than the 500 an unhandled StoreError would become. The code is
    fine and the request was fine -- the store is unreachable -- and the three
    callers all want the same thing from that distinction. The webhook wants it
    most: Stripe retries on any non-2xx, so a database that is briefly away
    delays a fulfilment instead of dropping it.

    The type name and nothing else. A psycopg error stringifies to whatever the
    server said, which can include the DSN.
    """
    return HTTPException(
        status_code=503,
        detail=f"the entitlement store is not usable ({type(exc).__name__})")


# -- which drafts are free ---------------------------------------------------


def note_mock_room(league_id) -> None:
    """Remember that we put somebody in this ESPN mock room.

    Called by the mock-join endpoint, which is the one place a seat in a mock
    is taken through this server. It is what lets the gate below tell a mock
    from a real league WITHOUT another call to ESPN on the connect path -- and
    a wrong answer there charges somebody for a free thing, so the test has to
    be one we can make from data already in hand.

    Best-effort. A draft that fails to be recorded here is one the lobby test
    below usually still catches, and a billing table that cannot be written is
    not a reason to refuse somebody the seat they just paid nothing for.
    """
    if not league_id or not enabled():
        return
    try:
        _db().execute(
            "INSERT INTO mock_room VALUES (?, ?) ON CONFLICT DO NOTHING",
            [str(league_id), _now()])
    except Exception:      # noqa: BLE001 -- see the docstring
        pass


def note_mock_rooms(league_ids) -> None:
    """Remember every room ESPN's mock lobby has shown us.

    Called from the lobby's own refresh, which runs every twelve seconds
    while anybody is on the site. It is what makes the gate below correct
    rather than merely usually-correct: the directory lists the rooms open
    right now, this table accumulates every one it has EVER listed, and it
    lives on the volume, so knowledge survives a redeploy.

    Without it a mock somebody joined on ESPN's own site an hour ago -- long
    gone from the directory by the time they connect -- would look exactly
    like a real league and get charged for.
    """
    # Nothing to learn for an instance that sells nothing: without a Stripe
    # key every draft is free, `is_free_draft` is never consulted, and this
    # would be a database file created on every laptop and in every test run
    # to answer a question nobody asks there.
    if not enabled():
        return
    fresh = [str(i) for i in league_ids if i is not None]
    if not fresh:
        return
    with _lock:
        unseen = [i for i in fresh if i not in _known_mocks]
        _known_mocks.update(unseen)
    if not unseen:
        return
    try:
        # One statement, not one per room, for the reason `_insert_mock_rooms`
        # gives: this is reachable from a connect (`is_free_draft` reads the
        # directory when the lobby cache has gone cold), and a full directory
        # is enough rooms for a row-at-a-time write to be felt there.
        _insert_mock_rooms(_db(), unseen)
    except Exception:      # noqa: BLE001 -- see note_mock_room
        pass


def is_free_draft(league_id) -> bool:
    """Whether this league id is a mock room, and so free.

    THREE ANSWERS, IN ORDER OF WHAT THEY COST:

      1. The table. Either we seated somebody there ourselves, or ESPN's
         mock-lobby directory has listed it at some point while this
         deployment has been running (`note_mock_rooms`). No I/O.
      2. The directory in memory, which the lobby watcher keeps seconds
         fresh whenever anybody is on the site. No I/O.
      3. One read of the directory, if the cache has gone cold. This is the
         only case that costs an upstream call, and it happens on a connect,
         which already makes several.

    Then: not a mock. A real league id has never appeared in ESPN's mock
    directory and never will, so this is the answer that makes anything
    chargeable at all.

    THE ONE EXCEPTION IS ESPN BEING UNREADABLE. If the directory cannot be
    read at all we cannot tell the two apart, and the tie goes to the
    drafter: blocking somebody out of a free mock draft on draft night
    because a third party is down is a worse failure than a missed $9.99.
    """
    if not league_id:
        return True
    league_id = str(league_id)
    with _lock:
        if league_id in _known_mocks:
            return True
    try:
        rows = _db().execute(
            "SELECT 1 FROM mock_room WHERE league_id = ?", [league_id])
        if rows:
            with _lock:
                _known_mocks.add(league_id)
            return True
    except Exception:      # noqa: BLE001 -- see below
        # THE TIE GOES TO THE DRAFTER, unchanged, and now with a second way to
        # get here. It used to mean a corrupt or locked billing file. On
        # Postgres it also means the network: a pool with nothing free after
        # five seconds, or a database that is refusing connections, both
        # arrive as StoreError. Charging somebody for a free mock because a
        # third party is having a bad minute is a worse failure than a missed
        # $9.99, so the answer stays "free".
        return True
    try:
        from api import lobby
        rows = lobby.cached_rows()
        if rows is None:
            # Cold cache. Read the directory once; this also teaches the
            # table every room currently open, so the next unknown id is
            # usually answered without any of this.
            rows = lobby._cached_rows()
    except Exception:      # noqa: BLE001
        rows = None
    if rows is None:
        return True
    if any(str(r.get("leagueId")) == league_id for r in rows):
        note_mock_room(league_id)
        return True
    return False


# -- entitlements ------------------------------------------------------------


def is_local_request(request: Request) -> bool:
    """Is this request genuinely from the machine the server runs on.

    WHY THIS QUESTION IS ASKED AT ALL. `pipeline/espn_drafts.saved_session`
    is the owner's own ESPN login, kept in a file, and it is how a local
    checkout reaches its leagues without ever connecting anything (the
    dashboard reports `source: "local"`). There is no cookie in that flow, so
    `custody_for` has nothing to resolve and a purchase has nothing to attach
    to -- which is a 401 on the one machine where the person IS the account.

    WHY IT MUST BE THIS NARROW. That same file exists on the deployment: it
    is the FARM's ESPN login, uploaded as `FARM_ESPN_STATE_B64`. If billing
    accepted it as an identity there, every visitor on earth would resolve to
    the same account -- one person's $9.99 would unlock the product for
    everybody, and anybody could spend what somebody else bought. So the
    local login is honoured only where it means what it says.

    SHARED WITH `api/drafts.session_for`, which is the same question with a
    worse answer when it goes wrong: there, honouring the farm's login for a
    stranger served every visitor the owner's league list, so the landing
    page took them all for the owner and drew the dashboard instead of the
    introduction.

    TWO CONDITIONS, because either alone can be arranged. A loopback client
    address is not proof on its own: a proxy running beside the app can
    present one. A request that came through any proxy carries forwarding
    headers, and a direct one does not. Both together are as close to "this
    request did not come off a network" as the app can get.
    """
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if host not in ("127.0.0.1", "::1", "localhost"):
        return False
    return not any(h in request.headers for h in
                   ("x-forwarded-for", "x-forwarded-proto", "forwarded"))


def _local_swid(request: Request):
    """The owner's own ESPN account, on the owner's own machine. Else None."""
    if not is_local_request(request):
        return None
    try:
        from pipeline import espn_drafts
        saved = espn_drafts.saved_session()
    except Exception:      # noqa: BLE001 -- no file, unreadable file
        return None
    return saved[0] if saved else None


def _account_ids(request: Request, store=None) -> list:
    """Every id this browser's ESPN account could hold rows under.

    A list, not a value, because the custody key can be rotated: a row written
    under version 1 keeps its version-1 id forever and the newest key computes
    a different one. `CredentialStore.account_ids` yields both, newest first,
    which is the same lazy-rotation rule the credential rows themselves
    follow.

    Empty when this browser holds no stored ESPN session at all. That is not
    an error here -- it is the ordinary state of somebody who has not
    connected -- and every caller treats it as "not entitled".
    """
    resolved = custody_for(request, store)
    # The connected path: a cookie this browser was given, resolving to a
    # stored ESPN session. Every visitor to a deployment is this.
    swid = resolved.swid if resolved is not None else _local_swid(request)
    if swid is None:
        return []
    return _custody_store(store).account_ids(swid)


def entitled(account_ids: list, league_id, season: int) -> bool:
    """Has one of these ids paid for this draft."""
    if not account_ids:
        return False
    marks = ", ".join("?" for _ in account_ids)
    rows = _db().execute(
        f"""SELECT 1 FROM entitlement
             WHERE account_id IN ({marks})
               AND league_id = ? AND season = ? AND revoked_at IS NULL""",
        [*account_ids, str(league_id), int(season)])
    return len(rows) > 0


def grant(account_id: str, league_id, season: int, checkout_session=None,
          payment_intent=None) -> None:
    """Record a paid draft. Idempotent on (account, league, season)."""
    _db().execute(
        """INSERT INTO entitlement
             (account_id, league_id, season, granted_at, revoked_at,
              checkout_session, payment_intent)
           VALUES (?, ?, ?, ?, NULL, ?, ?)
           ON CONFLICT (account_id, league_id, season) DO UPDATE SET
             granted_at = excluded.granted_at,
             revoked_at = NULL,
             checkout_session = excluded.checkout_session,
             payment_intent = excluded.payment_intent""",
        [account_id, str(league_id), int(season), _now(),
         checkout_session, payment_intent])


def revoke(payment_intent: str) -> int:
    """Withdraw whatever this payment bought. Returns rows affected.

    Keyed on the payment rather than on the account, because a refund names a
    charge and nothing else: the person refunding in the Stripe Dashboard is
    not going to be holding an HMAC.
    """
    if not payment_intent:
        return 0
    before = _db().execute(
        "SELECT count(*) FROM entitlement "
        "WHERE payment_intent = ? AND revoked_at IS NULL",
        [payment_intent])[0][0]
    _db().execute(
        "UPDATE entitlement SET revoked_at = ? "
        "WHERE payment_intent = ? AND revoked_at IS NULL",
        [_now(), payment_intent])
    return int(before)


def _first_time(event_id: str, kind: str) -> bool:
    """Whether this Stripe event has been acted on before.

    The INSERT is the claim: `ON CONFLICT DO NOTHING` plus a row count means
    two deliveries racing each other cannot both proceed, which a
    SELECT-then-INSERT would allow.
    """
    inserted = _db().execute(
        "INSERT INTO billing_event (event_id, type, seen_at) VALUES (?, ?, ?) "
        "ON CONFLICT (event_id) DO NOTHING RETURNING event_id",
        [event_id, kind, _now()])
    # RETURNING, rather than the row count an INSERT answers with, because the
    # two backends count differently and only one of them can be asked that
    # way. A row comes back only when this statement actually wrote one, so an
    # empty result IS the duplicate. Counting the table afterwards instead
    # would always find the row -- this call just put it there -- and report
    # every event as new, which is the bug this shape exists to avoid.
    return len(inserted) == 1


# -- favourites --------------------------------------------------------------
#
# The only thing in this module that is not about money. It lives here because
# it is keyed by the same account id and stored in the same file, and a second
# store for one small table would be a second connection, a second set of
# schema statements and a second thing to get wrong. Nothing below charges for
# anything or reads a Stripe object.


def favorites(account_ids: list) -> list:
    """The players this account starred, in the order it starred them.

    A LIST OF IDS, not a set, because the order is the preference: the picker
    is an ordered list and `scoring/plan.py` reads the first names first.

    Takes every id version for the same reason `entitled` does -- a rotated
    custody key computes a different id for the same person, and rows keep the
    id they were written under. THE NEWEST VERSION THAT HAS ROWS WINS, rather
    than a union of all of them: a merge could hand back more than the
    twenty-five the endpoint accepts, in an order that is nobody's choice, and
    a list saved after a rotation is by definition the more recent answer.
    """
    if not account_ids:
        return []
    marks = ", ".join("?" for _ in account_ids)
    rows = _db().execute(
        f"""SELECT account_id, player_id FROM favorite_player
             WHERE account_id IN ({marks}) ORDER BY ord""",
        [str(account_id) for account_id in account_ids])
    if not rows:
        return []
    chosen: dict = {}
    for account_id, player_id in rows:
        chosen.setdefault(str(account_id), []).append(str(player_id))
    for account_id in account_ids:
        found = chosen.get(str(account_id))
        if found:
            return found
    return []


def set_favorites(account_ids: list, player_ids) -> list:
    """Replace this account's whole list. Returns what was stored.

    TAKES THE SAME LIST `favorites` DOES, newest first, and not just the id it
    is about to write under. It DELETES UNDER EVERY VERSION and inserts under
    the newest, because the read has to be able to say "nothing" afterwards.
    Delete only under the newest and a rotated account that clears its list
    finds the old key's rows again on the next read -- `favorites` falls
    through to the version that still has rows -- so the user's clear silently
    reverts to whatever they chose before the rotation. Latent today (nothing
    clears a list yet: the endpoint's floor is five) and a bug the moment
    anything does.

    ONE TRANSACTION, and that is the entire reason `transaction` exists on the
    adapters. The write is a replace, so it is a DELETE and an INSERT; if the
    delete committed and the insert did not, somebody who edited their list
    would be left with no list at all -- and the Dashboard would then show
    them the onboarding panel again as if they had never chosen.

    The insert is ONE statement with every row's values in it rather than a
    row at a time, so a save is two round trips whatever the backend and
    however long the list.

    No validation here: 5-25, and that every id names a player, are properties
    of the REQUEST and are checked where the board is in hand (api/account.py).
    A caller with a legitimate reason to store something else -- a test, a
    migration -- should not have to argue with this function.
    """
    accounts = [str(account_id) for account_id in account_ids]
    if not accounts:
        raise ValueError("set_favorites needs an account id to write under")
    ids = [str(player_id) for player_id in player_ids]
    marks = ", ".join("?" for _ in accounts)
    statements = [
        (f"DELETE FROM favorite_player WHERE account_id IN ({marks})",
         accounts)]
    if ids:
        now = _now()
        params: list = []
        for position, player_id in enumerate(ids):
            params.extend([accounts[0], player_id, position, now])
        values = ", ".join("(?, ?, ?, ?)" for _ in ids)
        statements.append((
            f"""INSERT INTO favorite_player
                  (account_id, player_id, ord, created_at)
                VALUES {values}""",
            params))
    _db().transaction(statements)
    return ids


# -- the gate ----------------------------------------------------------------


def require_paid(request: Request, league_id, season, store=None) -> None:
    """Refuse to open a real draft nobody has paid for.

    402, which is the one status code that means exactly this, and the body
    names the league so the page can offer checkout for the right draft rather
    than for whatever it last had in a state variable.

    Free, always, when: billing is off on this instance, the room is a mock,
    or this account has already bought this draft.
    """
    if not enabled():
        return
    if is_free_draft(league_id):
        return
    season = int(season or CURRENT_SEASON)
    if entitled(_account_ids(request, store), league_id, season):
        return
    raise HTTPException(
        status_code=402,
        detail={"error": "payment_required",
                "league_id": str(league_id),
                "season": season,
                "message": "Mock drafts are free. A real league's draft is "
                           "$9.99 for the season."})


# -- Stripe ------------------------------------------------------------------


def _client():
    """The Stripe client, built per call from the environment.

    Imported inside the function so a checkout without the `stripe` package
    installed -- every local one, until somebody re-runs `make setup` -- fails
    only if it actually tries to sell something.

    A `StripeClient` instance rather than the module-level `stripe.api_key`,
    which is deprecated in every current SDK.
    """
    import stripe
    return stripe.StripeClient(api_key=os.environ[KEY_ENV],
                               stripe_version=API_VERSION)


def _return_to(raw) -> str:
    """A path on this site to send the buyer back to.

    Validated rather than trusted: this value ends up in a URL Stripe will
    redirect a browser to, and an unchecked one is an open redirect with a
    payment page in front of it. A path, starting with a single slash, and
    nothing else -- no scheme, no host, no protocol-relative `//evil.com`.
    """
    path = (raw or "/").strip()
    if not path.startswith("/") or path.startswith("//"):
        return "/"
    if urlparse(path).scheme or urlparse(path).netloc:
        return "/"
    return path


def _base_url(request: Request) -> str:
    """Where this site lives, for the two URLs Stripe sends the buyer back to.

    The environment first, because behind a TLS-terminating proxy the
    request's own base URL is `http://` and Stripe refuses a plaintext return
    URL in live mode.
    """
    configured = os.environ.get(BASE_URL_ENV, "").strip()
    if configured:
        return configured.rstrip("/")
    return str(request.base_url).rstrip("/")


class CheckoutBody(BaseModel):
    leagueId: str
    season: str | None = None
    # Where to land after paying. A path on this site; see `_return_to`.
    returnTo: str | None = None


def register_billing_routes(app, store=None):
    """Mount the three billing endpoints.

    They are mounted whether or not billing is switched on. `/status` has a
    truthful answer either way ("nothing is for sale here"), and the other two
    refusing with a 503 is a better failure than a 404 that looks like a
    routing bug.
    """

    @app.get("/api/billing/status")
    def billing_status(request: Request, leagueId: str | None = None,
                       season: str | None = None):
        """What this browser owes for this draft, if anything.

        The page calls it before offering a connect button, so it answers for
        an unconnected visitor too rather than 401ing: "you are not entitled"
        is the honest answer to "may I draft", and being signed out is one
        reason for it.
        """
        if not enabled():
            return {"enabled": False, "required": False, "entitled": True}
        year = int(season or CURRENT_SEASON)
        if leagueId is None:
            return {"enabled": True, "required": False, "entitled": False}
        if is_free_draft(leagueId):
            return {"enabled": True, "required": False, "entitled": True,
                    "reason": "mock"}
        try:
            paid = entitled(_account_ids(request, store), leagueId, year)
        except StoreError as exc:
            raise _unavailable(exc) from None
        return {"enabled": True, "required": not paid, "entitled": paid,
                "league_id": str(leagueId), "season": year}

    @app.post("/api/billing/checkout")
    def billing_checkout(body: CheckoutBody, request: Request):
        """Start a Checkout Session for one real draft.

        REQUIRES A CONNECTED ESPN ACCOUNT, and not for revenue reasons: the
        entitlement has to be keyed to something, and the connected account is
        the only identity this product has. Buying first and connecting
        afterwards would leave a payment with nothing to attach to.
        """
        if not enabled():
            raise HTTPException(status_code=503,
                                detail="this instance does not sell anything")
        ids = _account_ids(request, store)
        if not ids:
            raise HTTPException(
                status_code=401,
                detail="Connect your ESPN account first. The purchase is "
                       "attached to it.")
        year = int(body.season or CURRENT_SEASON)
        if is_free_draft(body.leagueId):
            raise HTTPException(status_code=400,
                                detail="Mock drafts are free.")
        try:
            already_paid = entitled(ids, body.leagueId, year)
        except StoreError as exc:
            raise _unavailable(exc) from None
        if already_paid:
            raise HTTPException(status_code=409,
                                detail="This draft is already paid for.")
        base = _base_url(request)
        back = _return_to(body.returnTo)
        joiner = "&" if "?" in back else "?"
        try:
            session = _client().v1.checkout.sessions.create(
                params={
                    "mode": "payment",
                    "line_items": [{"price": _price_id(), "quantity": 1}],
                    # The id the webhook grants against. `client_reference_id`
                    # rather than metadata alone because Stripe shows it on the
                    # payment in the Dashboard, which is what a refund needs to
                    # be traceable back to a row.
                    "client_reference_id": ids[0],
                    "metadata": {"account_id": ids[0],
                                 "league_id": str(body.leagueId),
                                 "season": str(year)},
                    "success_url": f"{base}{back}{joiner}paid=1",
                    "cancel_url": f"{base}{back}",
                    "integration_identifier": INTEGRATION_ID,
                },
                # One purchase per account per draft, even if the button is
                # double-clicked or the page is reloaded mid-redirect.
                options={"idempotency_key":
                         f"draft:{ids[0]}:{body.leagueId}:{year}"},
            )
        except Exception as exc:      # noqa: BLE001 -- Stripe refused, and the
            # page has to say something other than "500". The message is
            # Stripe's own, which for a card or configuration problem is more
            # useful than anything invented here.
            raise HTTPException(status_code=502,
                                detail=f"Stripe would not open a checkout: "
                                       f"{exc}") from None
        return {"url": session.url}

    @app.post("/api/billing/webhook")
    async def billing_webhook(request: Request):
        """Stripe's own report of what happened, which is what grants access.

        FULFILMENT LIVES HERE AND NOT ON THE SUCCESS PAGE. A buyer whose
        browser dies during the redirect has still paid, and a success page
        that grants access is a success page anybody can visit.

        The RAW body, not the parsed one: the signature is over the exact
        bytes Stripe sent, and any reserialisation invalidates it.
        """
        secret = os.environ.get(WEBHOOK_SECRET_ENV, "").strip()
        if not secret:
            # An unverified webhook grants paid access to anybody who can POST
            # JSON here. Refusing outright is the only safe unconfigured state.
            raise HTTPException(status_code=503,
                                detail=f"{WEBHOOK_SECRET_ENV} is not set")
        import stripe
        payload = await request.body()
        try:
            event = stripe.Webhook.construct_event(
                payload, request.headers.get("stripe-signature", ""), secret)
        except Exception:      # noqa: BLE001 -- a bad signature is not an
            # error to explain in detail to whoever sent it.
            raise HTTPException(status_code=400,
                                detail="bad signature") from None
        try:
            return _handle_event(event)
        except StoreError as exc:
            # Stripe retries on any non-2xx, so this is a fulfilment that
            # happens late rather than one that never happens. A 500 would say
            # the same thing to Stripe and a different thing to whoever reads
            # the logs.
            raise _unavailable(exc) from None

    return app


def _handle_event(event) -> dict:
    """Act on one verified Stripe event. Separate from the route so a test can
    exercise the decisions without minting a signature."""
    kind = event["type"]
    obj = event["data"]["object"]

    # Both of these mean "a Checkout Session finished". The second is the
    # delayed-notification case, where the money lands hours after the buyer
    # left the page -- and where the FIRST event arrives unpaid, which is why
    # neither is trusted without re-reading `payment_status`.
    if kind in ("checkout.session.completed",
                "checkout.session.async_payment_succeeded"):
        if obj.get("payment_status") == "unpaid":
            return {"ignored": "unpaid"}
        meta = obj.get("metadata") or {}
        account_id = meta.get("account_id") or obj.get("client_reference_id")
        league_id = meta.get("league_id")
        season = meta.get("season")
        if not (account_id and league_id and season):
            # Nothing to grant against. Answered 200 rather than 400 so Stripe
            # stops retrying an event that will never be actionable -- a
            # payment made through some other flow, a Payment Link, a test.
            return {"ignored": "no metadata"}
        if not _first_time(event["id"], kind):
            return {"ignored": "duplicate"}
        grant(account_id, league_id, int(season),
              checkout_session=obj.get("id"),
              payment_intent=obj.get("payment_intent"))
        return {"granted": True}

    # A refund takes the draft back. Partial refunds included: this product
    # is one indivisible thing at one price, so half of it is not half a
    # draft.
    if kind == "charge.refunded":
        if not _first_time(event["id"], kind):
            return {"ignored": "duplicate"}
        return {"revoked": revoke(obj.get("payment_intent"))}

    if kind == "checkout.session.async_payment_failed":
        # Nothing was granted (the completed event for these arrives unpaid
        # and is ignored above), so there is nothing to undo. Logged as seen
        # so the event table shows what actually arrived.
        _first_time(event["id"], kind)
        return {"ignored": "payment failed"}

    return {"ignored": kind}
