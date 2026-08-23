"""Custody of other people's ESPN sessions.

WHAT IS ACTUALLY BEING HELD, because it decides everything below. An
`espn_s2` + `SWID` pair is a full ESPN account session. It is not scoped to
fantasy, it cannot be downgraded to read-only, and ESPN publishes no
per-application revocation -- a user's only remedy after a leak is to log out
of ESPN everywhere. It stays valid for months, so "the tokens would have
expired by the time anyone read the dump" is not available as a comfort here.

AND WE CANNOT TELL THEM. `fan.api.espn.com/apis/v2/fans/{SWID}` returns no
email and no name (verified against a real account: only `dmaId`, an account
creation date, and insider/premium flags). There is no way to contact a user,
ever. That single fact is why this module is shaped the way it is:

  - a leak cannot be followed by a notification, so it has to be survivable
    without one -- hence "a stolen database file is inert" as the bar;
  - a user who clears cookies cannot be recovered and cannot be asked, so a
    clock (`expires_at`) is the ONLY thing that ever removes their row.

THE ONE PROPERTY THIS MODULE EXISTS TO PROVIDE: someone who walks off with
the database file and a checkout of this repository, and does NOT have the
environment, learns nothing. Not "the tokens are encrypted but the user list
is readable" -- there is no readable user list either, because the SWIDs live
inside the encrypted blob and the row keys are HMACs of them. Absent the key,
an attacker holding a SWID cannot even confirm whether that person is in the
table. `tests/test_credentials.py` asserts this against the real bytes on
disk, not against the query results.

SWID IS AN IDENTIFIER, NOT A SECRET. It rides in the invite POST's query
string, in the draft socket's JOIN url, and in anything either gets pasted
into (`pipeline/redact.py`'s docstring is a catalogue of the places it
leaks). So a server that hands back a stored session to whoever presents a
SWID is an account-takeover endpoint with extra steps. Here, SWID is the
USERNAME -- a row key and nothing more -- and the 256-bit `httpOnly` cookie
minted by `connect` is the PASSWORD. Nothing in this module accepts a SWID as
authorization; `resolve` takes a cookie and only a cookie.

TWO TABLES, NOT ONE. `espn_credential` is per ESPN user; `espn_session` is per
browser. Collapsing them would mean a user connecting from a second browser
silently logs the first one out -- and the first browser is frequently the one
with the draft open.

WHY `cryptography` (the first third-party dependency this project has added
for anything but data). Fernet is AES-128-CBC plus an HMAC tag with a
versioned wire format and a maintained implementation. The alternative was
hand-rolling an AEAD around `hashlib`, on the one code path in this repository
where getting it subtly wrong is unrecoverable for people we cannot contact.
That is not a close call.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import duckdb

from pipeline import redact

# The environment variable holding the key material. NEVER a column, never a
# file beside the database, never a default baked into this file: the whole
# inert-dump property is exactly the statement "the key is not in the thing
# that gets stolen". A default value here would quietly turn every deployment
# that forgot to set it into a plaintext store that still passes its tests.
KEYS_ENV = "ESPN_CUSTODY_KEYS"

# Where the two tables live. Its own file rather than a couple of extra tables
# in `data/nfl.duckdb`, for two reasons that both point the same way: the main
# database is copied around freely (per-league provisioning seeds new files
# from it, see pipeline/leagues.provision_league) and is the thing anyone
# debugging this project opens first. Neither is a place stored account
# sessions should be able to ride along in by accident.
DB_PATH_ENV = "ESPN_CUSTODY_DB_PATH"
DEFAULT_DB_PATH = "data/custody.duckdb"

# How long a credential survives without being used. Sliding: every successful
# `resolve` pushes it out again, so this is an IDLE timeout, not a hard
# lifetime, and an active user is never logged out by it.
#
# This is the only reaper there is. A user who clears their cookies is gone --
# no email, no recovery, no way to ask them whether they still want us holding
# their session -- so without a clock the table would accumulate live account
# sessions belonging to people who cannot be reached, forever. Thirty days is
# the spec's number: long enough that a draft season does not require
# reconnecting, short enough that the table reflects people who are still here.
TTL_DAYS_ENV = "ESPN_CUSTODY_TTL_DAYS"
DEFAULT_TTL_DAYS = 30

# The name of the browser cookie. Also registered in pipeline/redact.py's
# parameter-name rule, so a Cookie header or a URL that ends up in a log or an
# exception loses the value rather than printing the password to every stored
# session it names.
COOKIE_NAME = "espn_custody"

# The dev opt-out for the plain-HTTP refusal, deliberately NOT sharing a name
# with anything else that makes local development work. A single `DEV=1` that
# both points the frontend at a local API and disables a credential-transport
# check is how "it was on in staging, and staging's config was copied to
# production" happens. This switch does one thing, is named after the one
# thing, and appears in no other module.
ALLOW_PLAINTEXT_ENV = "ESPN_CUSTODY_ALLOW_PLAINTEXT_HTTP"

# Whether to believe `X-Forwarded-Proto` when deciding the request was TLS.
# Off by default because that header is just a header: a client can send it,
# so trusting it unconditionally would make the plain-HTTP refusal a formality
# any attacker could opt out of. A deployment that really does terminate TLS
# at a proxy sets this, having satisfied itself that the proxy overwrites the
# header rather than passing the client's through.
TRUST_FORWARDED_PROTO_ENV = "ESPN_CUSTODY_TRUST_FORWARDED_PROTO"


class CustodyUnavailable(RuntimeError):
    """No usable key material, so credentials can be neither stored nor read.

    Raised rather than falling back to anything: the only fallbacks available
    are "store it in the clear" and "pretend the user is not connected", and
    the first is the failure this whole module exists to prevent while the
    second would silently drop a working session on a misconfigured restart.
    A deployment missing its key is broken and should say so.
    """


class InsecureTransport(RuntimeError):
    """The request carrying credentials was not TLS and the opt-out is unset."""


@dataclass(frozen=True)
class CustodyKey:
    """One key version: what encrypts blobs, and what keys the row HMACs."""

    version: int
    fernet: object
    # Derived from the same secret rather than being a second configured
    # value, because two secrets to rotate in step is two chances to rotate
    # only one. Domain-separated (see _derive_index_key) so the value that
    # indexes rows is not literally the value that decrypts them -- an
    # attacker who somehow learns the index key still cannot read a blob.
    index_key: bytes


@dataclass(frozen=True)
class MintedSession:
    """What `connect` hands back. `cookie` is the ONLY time the raw session
    value exists outside the browser -- it is never stored, so it cannot be
    re-read later, only replaced."""

    cookie: str
    credential_id: str
    session_id: str
    expires_at: datetime


@dataclass(frozen=True)
class ResolvedCredential:
    """A decrypted session, in memory, for the length of one request.

    `credential_id` rides along so a caller that gets a 401 from ESPN can
    delete the row it just used without having to hold the SWID to look it up
    again (see `forget_if_unauthorized`).
    """

    credential_id: str
    session_id: str
    swid: str
    espn_s2: str
    expires_at: datetime


def _derive_index_key(raw: bytes) -> bytes:
    """The HMAC key for row ids, from the same secret Fernet uses.

    Domain separation via a fixed label: `HMAC(secret, label)` cannot be
    confused with `HMAC(secret, swid)` for any swid, so the index key and the
    row ids computed under it are cryptographically independent of the
    encryption key even though one secret configures both.
    """
    return hmac.new(raw, b"espn-custody-index-v1", sha256).digest()


def _parse_keys(spec: str) -> dict:
    """`"1:<key>,2:<key>"` (or a bare key, meaning version 1) -> versions.

    Multiple versions coexist ON PURPOSE and that is the whole rotation
    story: a row records the version it was written under, lookups try every
    known version, and new writes use the highest. So adding a key is a
    deploy, not a flag day, and removing the old one later is what actually
    completes the rotation.
    """
    from cryptography.fernet import Fernet

    keys = {}
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        version, _, material = chunk.partition(":")
        if not material:
            # A bare key with no version prefix. Accepted because the first
            # deployment has exactly one key and should not have to learn the
            # rotation syntax to set it; it is version 1, so adding "2:..."
            # later is a pure addition.
            version, material = "1", version
        try:
            number = int(version)
        except ValueError:
            raise CustodyUnavailable(
                f"{KEYS_ENV} has a non-numeric key version {version!r}")
        try:
            fernet = Fernet(material.strip())
        except (ValueError, TypeError) as exc:
            # Deliberately does NOT include the offending value: this string
            # reaches a log, and the offending value is a key.
            raise CustodyUnavailable(
                f"{KEYS_ENV} version {number} is not a valid Fernet key "
                "(32 url-safe base64 bytes, as `Fernet.generate_key()` "
                "prints)") from exc
        keys[number] = CustodyKey(
            version=number, fernet=fernet,
            index_key=_derive_index_key(base64.urlsafe_b64decode(material.strip())))
    if not keys:
        raise CustodyUnavailable(
            f"{KEYS_ENV} is not set. Generate one with "
            "`python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\"` and set it in the "
            "environment -- never in the database, and never in the repo.")
    return keys


def generate_key() -> str:
    """A fresh key, for an operator setting up a deployment or rotating one.

    Here rather than in a script so there is one place that says what shape
    the value has, and so the test suite can mint throwaway keys the same way
    production mints real ones.
    """
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


def _utc(value: datetime | None) -> datetime:
    """Naive UTC, which is what the TIMESTAMP columns hold.

    DuckDB's TIMESTAMP is naive, so an aware datetime going in and a naive one
    coming out would compare wrongly by exactly the local offset -- which for
    a reaper means credentials living hours longer or dying hours early, on a
    machine whose timezone nobody wrote down. Everything crosses this
    function, so the ambiguity exists in one place instead of at every call.
    """
    if value is None:
        value = datetime.now(timezone.utc)
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def is_secure_transport(scheme: str, forwarded_proto: str | None = None,
                        env=None) -> bool:
    """Whether a request carrying credentials may be honoured.

    Kept here, framework-free, rather than in the API layer: the rule is a
    property of the credential (it must not cross a plaintext wire), not of
    FastAPI, and a pure function is testable without standing up a server.
    """
    env = os.environ if env is None else env
    if (scheme or "").lower() == "https":
        return True
    if env.get(TRUST_FORWARDED_PROTO_ENV) and (
            forwarded_proto or "").split(",")[0].strip().lower() == "https":
        return True
    return bool(env.get(ALLOW_PLAINTEXT_ENV))


def require_secure_transport(scheme: str, forwarded_proto: str | None = None,
                             env=None) -> None:
    """`is_secure_transport`, as a refusal.

    FAILS CLOSED, and the reason it is worth a hard refusal rather than a
    warning is specific: the session cookie is set `Secure`, and a `Secure`
    cookie on a plaintext connection is simply DROPPED by the browser. So the
    degraded mode is not "slightly less safe", it is "the credential was
    transmitted in the clear and the user appears not to be connected" -- a
    failure that looks like a bug, gets retried, and transmits it again.
    """
    if not is_secure_transport(scheme, forwarded_proto, env=env):
        raise InsecureTransport(
            "This endpoint handles ESPN account credentials and refuses to do "
            "so over plain HTTP. Use HTTPS, or set "
            f"{ALLOW_PLAINTEXT_ENV}=1 for local development only.")


# One DuckDB connection per file, reused. DuckDB takes a single-writer lock on
# the file, so opening a fresh connection per call would make two overlapping
# requests a hard failure rather than a queue. A single module-wide lock around
# every statement is enough serialisation for a table whose entire write
# traffic is "somebody clicked connect".
_CONNS: dict = {}
_CONN_LOCK = threading.RLock()

_SCHEMA = (
    # `id` is HMAC(swid, index_key), hex. The raw SWID appears in no column at
    # all -- it is inside `blob` -- so a dump cannot be joined against anything
    # ESPN knows, and an attacker holding a SWID cannot test it for membership
    # without the key.
    """CREATE TABLE IF NOT EXISTS espn_credential (
        id VARCHAR PRIMARY KEY,
        blob VARCHAR NOT NULL,
        key_version INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL,
        last_used_at TIMESTAMP NOT NULL,
        expires_at TIMESTAMP NOT NULL)""",
    # `id` is HMAC(cookie_value, index_key), hex, for the same reason: the
    # value the browser holds is a password, and a stolen dump full of
    # passwords would let the thief log in as every user rather than merely
    # read about them.
    #
    # NO FOREIGN KEY, even though `credential_id` is one. DuckDB does not
    # implement ON DELETE CASCADE, so the constraint would only be able to
    # REFUSE a credential delete that still had sessions -- turning "disconnect
    # everywhere" into an error in exactly the case it exists for. The cascade
    # is therefore done in code (see `_delete_credential`), sessions first, so
    # that a crash mid-delete leaves a credential nobody can reach rather than
    # a session pointing at nothing.
    """CREATE TABLE IF NOT EXISTS espn_session (
        id VARCHAR PRIMARY KEY,
        credential_id VARCHAR NOT NULL,
        key_version INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL,
        last_used_at TIMESTAMP NOT NULL,
        expires_at TIMESTAMP NOT NULL)""",
)


def _connect(path: str):
    with _CONN_LOCK:
        conn = _CONNS.get(path)
        if conn is None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            conn = duckdb.connect(path)
            for statement in _SCHEMA:
                conn.execute(statement)
            _CONNS[path] = conn
        return conn


def close_all() -> None:
    """Drop every cached connection. For tests, and for a process that wants
    to release the file lock without exiting."""
    with _CONN_LOCK:
        for conn in _CONNS.values():
            try:
                conn.close()
            except Exception:      # noqa: BLE001 -- closing a dead handle
                pass
        _CONNS.clear()


class CredentialStore:
    """The custody tables, and the only code that reads or writes them.

    Constructed with explicit `path`/`keys` in tests and from the environment
    in production (`default_store`). Explicit-by-default rather than a module
    of functions reading `os.environ` on every call, because a test that has
    to mutate process-wide environment to exercise key rotation is a test that
    can leak that state into the next one -- and these particular tests are
    the deliverable.
    """

    def __init__(self, path: str | None = None, keys: str | None = None,
                 ttl_days: float | None = None, out=print):
        self.path = path or os.environ.get(DB_PATH_ENV) or DEFAULT_DB_PATH
        self._keys_spec = keys
        self._ttl_days = ttl_days
        # Every line this module prints goes through the scrubber, once, here
        # -- rather than at each call site, which is the argument
        # pipeline/redact.py makes at length: call sites accumulate and one of
        # them forgets. Nothing below has to know it is being redacted.
        self._out = redact.redacting(out)

    # -- keys ---------------------------------------------------------------

    @property
    def keys(self) -> dict:
        """version -> CustodyKey, read fresh from the environment each time.

        Not cached: the cost is parsing a short string, and the benefit is
        that adding a key version is a restart of the process rather than a
        subtlety about when a cache was filled. Raises rather than returning
        {} -- see CustodyUnavailable.
        """
        spec = self._keys_spec
        if spec is None:
            spec = os.environ.get(KEYS_ENV, "")
        return _parse_keys(spec)

    @property
    def current_key(self) -> CustodyKey:
        """The highest version, which is what new rows are written under."""
        keys = self.keys
        return keys[max(keys)]

    @property
    def ttl(self) -> timedelta:
        days = self._ttl_days
        if days is None:
            try:
                days = float(os.environ.get(TTL_DAYS_ENV, "") or DEFAULT_TTL_DAYS)
            except ValueError:
                days = DEFAULT_TTL_DAYS
        return timedelta(days=days)

    # -- identifiers --------------------------------------------------------

    def _row_id(self, value: str, key: CustodyKey) -> str:
        """HMAC, not a plain hash. A plain SHA-256 of a SWID would be
        enumerable: SWIDs are public GUIDs that appear in URLs, so an attacker
        with a dump and a list of candidate SWIDs could hash each one and
        learn exactly which of them we hold. Keying the hash with a secret the
        dump does not contain removes that entirely -- membership cannot be
        tested at all without the key."""
        return hmac.new(key.index_key, str(value).encode("utf-8"),
                        sha256).hexdigest()

    def _candidate_ids(self, value: str):
        """(key, id) for every known key version, newest first.

        This is what makes rotation lazy instead of a migration: a row written
        under version 1 keeps its version-1 id forever, and lookups simply try
        version 2's id first and version 1's second. Adding a key breaks
        nothing and requires no rewrite pass.
        """
        keys = self.keys
        for version in sorted(keys, reverse=True):
            key = keys[version]
            yield key, self._row_id(value, key)

    # -- writing ------------------------------------------------------------

    def connect(self, swid: str, espn_s2: str, now=None) -> MintedSession:
        """Take custody of one ESPN session and mint a browser session for it.

        Upsert, not insert: reconnecting from the same browser must not leave
        the previous credential row behind holding an older `espn_s2`, and
        must not disturb the OTHER browsers this user has connected from.
        """
        if not swid or not espn_s2:
            # Storing half a session buys nothing and costs the same custody
            # obligations, so it is refused rather than half-written.
            raise ValueError("both swid and espn_s2 are required")
        now = _utc(now)
        key = self.current_key
        conn = _connect(self.path)
        blob = key.fernet.encrypt(
            json.dumps({"swid": str(swid), "espn_s2": str(espn_s2)},
                       separators=(",", ":")).encode("utf-8")).decode("ascii")
        credential_id = self._row_id(swid, key)
        expires_at = now + self.ttl

        with _CONN_LOCK:
            # An existing row for this user, possibly under an older key
            # version and therefore under a DIFFERENT id.
            previous = None
            for candidate_key, candidate_id in self._candidate_ids(swid):
                row = conn.execute(
                    "SELECT id, created_at FROM espn_credential WHERE id = ?",
                    [candidate_id]).fetchone()
                if row:
                    previous = row
                    break
            created_at = previous[1] if previous else now
            conn.execute("DELETE FROM espn_credential WHERE id = ?",
                         [credential_id])
            conn.execute(
                "INSERT INTO espn_credential VALUES (?, ?, ?, ?, ?, ?)",
                [credential_id, blob, key.version, created_at, now, expires_at])
            if previous and previous[0] != credential_id:
                # Rotation caught up with this user: their row moves to the new
                # key version, so their id changes. Their OTHER browsers are
                # repointed rather than orphaned -- the two-table split exists
                # precisely so a second browser is not evicted, and a key
                # rotation is not a reason to start evicting them.
                conn.execute(
                    "UPDATE espn_session SET credential_id = ? "
                    "WHERE credential_id = ?", [credential_id, previous[0]])
                conn.execute("DELETE FROM espn_credential WHERE id = ?",
                             [previous[0]])

            # 256 bits from the OS CSPRNG. This value is the password to a
            # stored ESPN account session, so it is generated the same way a
            # password reset token would be and is never derived from anything
            # guessable (no user id, no timestamp, no counter).
            cookie = secrets.token_urlsafe(32)
            session_id = self._row_id(cookie, key)
            conn.execute("DELETE FROM espn_session WHERE id = ?", [session_id])
            conn.execute(
                "INSERT INTO espn_session VALUES (?, ?, ?, ?, ?, ?)",
                [session_id, credential_id, key.version, now, now, expires_at])
        return MintedSession(cookie=cookie, credential_id=credential_id,
                             session_id=session_id, expires_at=expires_at)

    # -- reading ------------------------------------------------------------

    def resolve(self, cookie: str, now=None) -> ResolvedCredential | None:
        """The stored ESPN session this browser is authorised for, or None.

        THE COOKIE IS THE ONLY INPUT. There is deliberately no `resolve_swid`
        beside this: a lookup keyed on SWID would be an endpoint that hands a
        stranger's account session to anyone who read their SWID out of a
        draft URL, which is the single failure this design is built around.

        Every refusal returns None rather than distinguishing "no such
        session" from "expired" from "credential gone": the caller's response
        is the same in all three (ask the user to reconnect), and a caller
        that cannot tell them apart cannot accidentally turn them into an
        oracle.
        """
        if not cookie:
            return None
        # Key material first, deliberately: `_connect` CREATES the database
        # file, and a deployment with no key must not leave a fresh, empty
        # custody database behind as evidence that it tried.
        self.keys
        now = _utc(now)
        conn = _connect(self.path)
        with _CONN_LOCK:
            # Reap first, so an expired row cannot be resolved even by a
            # deployment that never calls `reap` on a schedule. The TTL is the
            # only thing that ever removes an abandoned credential; making it
            # depend on a cron somebody remembered to write would mean it does
            # not really exist.
            self._reap(conn, now)
            found = None
            for _key, session_id in self._candidate_ids(cookie):
                row = conn.execute(
                    "SELECT id, credential_id FROM espn_session WHERE id = ?",
                    [session_id]).fetchone()
                if row:
                    found = row
                    break
            if not found:
                return None
            session_id, credential_id = found
            row = conn.execute(
                "SELECT blob, key_version, expires_at FROM espn_credential "
                "WHERE id = ?", [credential_id]).fetchone()
            if not row:
                # A session whose credential is gone: "disconnect everywhere"
                # from another browser, a 401, or the reaper. The session row
                # is cleaned up here rather than left to rot, since it can
                # never again resolve to anything.
                conn.execute("DELETE FROM espn_session WHERE id = ?",
                             [session_id])
                return None
            blob, key_version, expires_at = row
            key = self.keys.get(int(key_version))
            if key is None:
                # The key this row was written under has been retired. Not an
                # error: retiring a key is how a rotation is COMPLETED, and
                # the rows still holding it become unreadable by design. They
                # are left in place for the reaper rather than deleted here,
                # so that a deployment which merely misconfigured its key list
                # does not destroy every credential on the first request.
                self._out(f"custody: no key version {key_version}; "
                          "credential unreadable until it is restored")
                return None
            try:
                payload = json.loads(key.fernet.decrypt(blob.encode("ascii")))
            except Exception:      # noqa: BLE001 -- see below
                # InvalidToken, or a blob that is not JSON. Both mean the row
                # is not usable, and neither is worth an exception reaching a
                # request handler -- but the row is NOT deleted on a decrypt
                # failure, because the likeliest cause is a wrong key rather
                # than a corrupt row, and reacting to a misconfigured
                # environment by wiping everyone's credentials is worse than
                # any outage.
                self._out("custody: a stored credential did not decrypt under "
                          f"key version {key_version}")
                return None
            fresh = now + self.ttl
            # Sliding expiry: touching both rows on every use is what makes
            # the TTL an IDLE timeout. An active user is never logged out by
            # it, and a user who stopped coming is removed without anyone
            # having to notice they stopped.
            conn.execute(
                "UPDATE espn_session SET last_used_at = ?, expires_at = ? "
                "WHERE id = ?", [now, fresh, session_id])
            conn.execute(
                "UPDATE espn_credential SET last_used_at = ?, expires_at = ? "
                "WHERE id = ?", [now, fresh, credential_id])
        return ResolvedCredential(
            credential_id=credential_id, session_id=session_id,
            swid=str(payload.get("swid") or ""),
            espn_s2=str(payload.get("espn_s2") or ""),
            expires_at=fresh)

    # -- deleting -----------------------------------------------------------

    def disconnect(self, cookie: str) -> bool:
        """Log THIS browser out. The credential stays: the user's other
        browsers, and any draft one of them has open, are none of this
        browser's business."""
        if not cookie:
            return False
        conn = _connect(self.path)
        with _CONN_LOCK:
            for _key, session_id in self._candidate_ids(cookie):
                if conn.execute(
                        "SELECT 1 FROM espn_session WHERE id = ?",
                        [session_id]).fetchone():
                    conn.execute("DELETE FROM espn_session WHERE id = ?",
                                 [session_id])
                    return True
        return False

    def disconnect_everywhere(self, cookie: str) -> bool:
        """Delete the stored ESPN session and log out every browser holding it.

        Authorised by the cookie like everything else -- a user can only do
        this to themselves, and only from a browser they have already proved
        they hold a session for.
        """
        resolved = self.resolve(cookie)
        if resolved is None:
            return False
        return self.forget_credential(resolved.credential_id)

    def forget_credential(self, credential_id: str) -> bool:
        """Delete one credential and every session that pointed at it."""
        conn = _connect(self.path)
        with _CONN_LOCK:
            return self._delete_credential(conn, credential_id) >= 0

    @staticmethod
    def _delete_credential(conn, credential_id: str) -> int:
        """Returns how many session rows went with it, or -1 if there was no
        such credential -- so a caller can tell "deleted, no browsers attached"
        from "there was nothing there", which `disconnect_everywhere` and the
        reaper report differently."""
        existing = conn.execute("SELECT 1 FROM espn_credential WHERE id = ?",
                                [credential_id]).fetchone()
        sessions = int(conn.execute(
            "SELECT count(*) FROM espn_session WHERE credential_id = ?",
            [credential_id]).fetchone()[0])
        # Sessions FIRST. DuckDB has no ON DELETE CASCADE (see _SCHEMA), so
        # this ordering is the cascade, and it is the safe half to do first:
        # an interrupted delete leaves a credential with no way to reach it,
        # whereas the other order leaves live cookies pointing at a row that
        # is about to vanish.
        conn.execute("DELETE FROM espn_session WHERE credential_id = ?",
                     [credential_id])
        conn.execute("DELETE FROM espn_credential WHERE id = ?",
                     [credential_id])
        return sessions if existing else -1

    def forget_if_unauthorized(self, credential_id: str, exc_or_status) -> bool:
        """Delete the credential if ESPN just said 401. Returns whether it did.

        WHY DELETE RATHER THAN RETRY. A 401 means ESPN has already invalidated
        this session -- the user logged out, changed their password, or ESPN
        expired it. There is no refresh flow to run and no second attempt that
        can succeed, so retrying only means asking ESPN to reject it again.
        What keeping the row WOULD achieve is holding a known-dead account
        session on disk, which is pure liability with no remaining upside.

        Duck-typed on `response.status_code` in the same style as
        `redact.redacted_error`, so it takes an httpx `HTTPStatusError`, any
        exception shaped like one, or a bare status code -- without this
        module importing an HTTP client it otherwise has no use for.
        """
        status = exc_or_status
        if not isinstance(status, int):
            status = getattr(getattr(exc_or_status, "response", None),
                             "status_code", None)
        if status != 401:
            return False
        deleted = self.forget_credential(credential_id)
        if deleted:
            # The exception is worth printing -- which ESPN endpoint refused
            # tells you whether the session died or the league went private --
            # and it is also the single most dangerous string in this module.
            # An httpx HTTPStatusError stringifies to a sentence wrapped
            # around the whole request URL, and an ESPN request made on a
            # stored credential's behalf carries `espn_s2=` and the SWID in
            # exactly that URL. So it goes through `redacted_error` (status
            # first, URL scrubbed) AND through `self._out`, which is the
            # module-wide scrubber. Belt and braces on purpose: redaction is
            # idempotent (see pipeline/redact.py), so the second pass costs
            # nothing and covers a future caller who logs the raw exception.
            detail = ("" if isinstance(exc_or_status, int)
                      else f" -- {redact.redacted_error(exc_or_status)}")
            self._out("custody: ESPN rejected a stored session (401); "
                      f"the credential has been deleted{detail}")
        return deleted

    # -- the reaper ---------------------------------------------------------

    def reap(self, now=None) -> dict:
        """Delete everything past its expiry. Returns what went.

        Callable on a schedule, but not DEPENDENT on one -- `resolve` runs it
        too. See that method for why.
        """
        conn = _connect(self.path)
        with _CONN_LOCK:
            return self._reap(conn, _utc(now))

    @staticmethod
    def _reap(conn, now: datetime) -> dict:
        expired = [r[0] for r in conn.execute(
            "SELECT id FROM espn_credential WHERE expires_at <= ?",
            [now]).fetchall()]
        # Counted, not inferred. A reaped credential takes its browsers with
        # it, and a caller watching these numbers for "is the store growing
        # without bound" needs the total that actually went, not the subset
        # that happened to time out on its own clock.
        gone = 0
        for credential_id in expired:
            gone += max(CredentialStore._delete_credential(conn, credential_id), 0)
        # Sessions expire on their own clock as well as with their credential:
        # a browser that has not come back in the TTL is logged out even if
        # another browser has been keeping the credential alive.
        gone += int(conn.execute(
            "SELECT count(*) FROM espn_session WHERE expires_at <= ?",
            [now]).fetchone()[0])
        conn.execute("DELETE FROM espn_session WHERE expires_at <= ?", [now])
        return {"credentials": len(expired), "sessions": gone}

    # -- introspection, for operators and tests -----------------------------

    def counts(self) -> dict:
        """How many rows there are. Deliberately the ONLY thing this module
        will tell a caller about the table as a whole: there is no "list the
        users" call, because there is no way to write one that does not
        decrypt every credential to produce its answer."""
        conn = _connect(self.path)
        with _CONN_LOCK:
            return {
                "credentials": int(conn.execute(
                    "SELECT count(*) FROM espn_credential").fetchone()[0]),
                "sessions": int(conn.execute(
                    "SELECT count(*) FROM espn_session").fetchone()[0]),
            }


_DEFAULT = None
_DEFAULT_LOCK = threading.Lock()


def default_store() -> CredentialStore:
    """The process-wide store, configured from the environment.

    Lazy on purpose: constructing it opens (and creates) the database file and
    takes DuckDB's lock on it, and a process that never touches a credential
    -- the whole test suite, `make sim`, the farm -- should not be doing
    either.
    """
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = CredentialStore()
        return _DEFAULT


def reset_default_store() -> None:
    """Forget the process-wide store, so the next call re-reads the
    environment. For tests, and for nothing else."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        _DEFAULT = None
