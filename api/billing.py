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

_lock = threading.Lock()
_conn = None

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


def _db():
    """The entitlement database, opened once for the process.

    One connection, guarded by a lock, for the same reason everything else in
    this project holds one: DuckDB takes a single-writer lock per file, and
    two connections from two threads are two chances to meet it.
    """
    global _conn
    with _lock:
        if _conn is None:
            path = os.environ.get(DB_PATH_ENV) or DEFAULT_DB_PATH
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = duckdb.connect(path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entitlement (
                    account_id VARCHAR NOT NULL,
                    league_id VARCHAR NOT NULL,
                    season INTEGER NOT NULL,
                    granted_at TIMESTAMP NOT NULL,
                    revoked_at TIMESTAMP,
                    checkout_session VARCHAR,
                    payment_intent VARCHAR,
                    PRIMARY KEY (account_id, league_id, season))""")
            # WEBHOOK IDEMPOTENCY. Stripe retries on any non-2xx and can
            # deliver the same event twice on its own; every handler below is
            # gated on this table so a replay is a no-op rather than a second
            # grant (or, for a refund, a second revoke over a fresh purchase).
            conn.execute("""
                CREATE TABLE IF NOT EXISTS billing_event (
                    event_id VARCHAR PRIMARY KEY,
                    type VARCHAR,
                    seen_at TIMESTAMP NOT NULL)""")
            # Rooms we seated somebody in ourselves (see `note_mock_room`).
            # Here rather than anywhere else because its only job is to answer
            # a billing question: is this draft one of the free ones.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mock_room (
                    league_id VARCHAR PRIMARY KEY,
                    seen_at TIMESTAMP NOT NULL)""")
            _conn = conn
            _seed_mocks_from_corpus(conn)
    return _conn


def _seed_mocks_from_corpus(conn) -> None:
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
    draft ever recorded. Read once per process, at the moment the tables are
    created.

    Best-effort and read-only. The farm writes that file as drafts finish,
    and a lock held by it is not a reason to fail a request -- the seed is an
    improvement on two tests that still work without it.
    """
    try:
        from pipeline import draft_log as dl
        corpus = duckdb.connect(dl.CORPUS_PATH, read_only=True)
    except Exception:      # noqa: BLE001 -- no corpus yet, or the farm has it
        return
    try:
        rows = corpus.execute(
            "SELECT DISTINCT league_id FROM draft_log "
            "WHERE source = ? AND league_id IS NOT NULL",
            [dl.SOURCE_MOCK]).fetchall()
        if rows:
            conn.executemany(
                "INSERT INTO mock_room VALUES (?, ?) ON CONFLICT DO NOTHING",
                [[str(r[0]), _now()] for r in rows])
    except Exception:      # noqa: BLE001 -- an older corpus without the
        pass               # column, a partial write: the live tests remain
    finally:
        corpus.close()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def reset_for_tests(path: str | None = None) -> None:
    """Drop the cached connection so a test can point at its own file."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None
        _known_mocks.clear()
    if path is not None:
        os.environ[DB_PATH_ENV] = path


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
        now = _now()
        _db().executemany(
            "INSERT INTO mock_room VALUES (?, ?) ON CONFLICT DO NOTHING",
            [[i, now] for i in unseen])
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
        row = _db().execute(
            "SELECT 1 FROM mock_room WHERE league_id = ?", [league_id]).fetchone()
        if row is not None:
            with _lock:
                _known_mocks.add(league_id)
            return True
    except Exception:      # noqa: BLE001 -- an unreadable billing database is
        return True        # not a reason to charge anybody
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
    row = _db().execute(
        f"""SELECT 1 FROM entitlement
             WHERE account_id IN ({marks})
               AND league_id = ? AND season = ? AND revoked_at IS NULL""",
        [*account_ids, str(league_id), int(season)]).fetchone()
    return row is not None


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
        [payment_intent]).fetchone()[0]
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
        "INSERT INTO billing_event VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
        [event_id, kind, _now()]).fetchall()
    # DuckDB answers an INSERT with the number of rows it actually wrote, so
    # zero IS the duplicate. Counting the table afterwards instead would
    # always find the row -- this call just put it there -- and report every
    # event as new, which is the bug this shape exists to avoid.
    return bool(inserted and inserted[0][0])


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
        paid = entitled(_account_ids(request, store), leagueId, year)
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
        if entitled(ids, body.leagueId, year):
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
        return _handle_event(event)

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
