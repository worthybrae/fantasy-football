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
from pipeline.espn_identity import OwnershipUnproven, canonical_swid, verify_account

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
# In its OWN directory, which is created 0700, because DuckDB will not accept a
# pre-created empty file (it refuses anything on that path that is not already
# a valid database) and therefore creates the file itself at the process umask
# -- 0644 on a default machine. A file that is briefly world-readable inside a
# directory nobody else can traverse is not readable by anybody; the same file
# directly in `data/` would be. `_lock_down` then tightens the file itself, so
# both locks are on.
DEFAULT_DB_PATH = "data/custody/custody.duckdb"

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
    # Whether this connect actually rewrote the stored ESPN session, or merely
    # attached a new browser to the one that was already held. False is the
    # ordinary answer for a second browser with no cookie -- see `connect`,
    # lock 2 -- and it is worth returning rather than inferring, because it is
    # the difference between "your session was updated" and "this browser is
    # now signed in to the session you already gave us".
    replaced: bool = True


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
        if number < 1:
            # Versions are compared with max() to pick the writer and sorted
            # to order lookups, so a zero or negative version sorts below a
            # real one and would quietly never be written under. It is always
            # a typo; there is no deployment that wants it.
            raise CustodyUnavailable(
                f"{KEYS_ENV} has key version {number}; versions start at 1")
        if number in keys:
            # THE EXPENSIVE TYPO. `"1:old,1:new"` parses cleanly and the
            # second value silently wins, so every row already written under
            # the first is permanently unreadable -- and the failure does not
            # appear until a user comes back, by which time the old key may
            # have been rotated out of wherever it was kept. Refusing to start
            # is enormously cheaper than discovering this later.
            raise CustodyUnavailable(
                f"{KEYS_ENV} lists key version {number} twice; one version is "
                "one key, and the second would make the first version's rows "
                "unreadable")
        try:
            fernet = Fernet(material.strip())
        except (ValueError, TypeError) as exc:
            # Deliberately does NOT include the offending value: this string
            # reaches a log, and the offending value is a key.
            raise CustodyUnavailable(
                f"{KEYS_ENV} version {number} is not a valid Fernet key "
                "(32 url-safe base64 bytes, as `Fernet.generate_key()` "
                "prints)") from exc
        raw = base64.urlsafe_b64decode(material.strip())
        if not any(raw):
            # An all-zero key is what a placeholder, a truncated secret store
            # read, or a `base64 < /dev/zero` copy-paste produces. It is a
            # valid Fernet key and encrypts perfectly well, which is the
            # problem: the ciphertext is decryptable by anyone who guesses the
            # single most guessable key there is.
            raise CustodyUnavailable(
                f"{KEYS_ENV} version {number} is all zero bytes, which is a "
                "placeholder rather than a key")
        keys[number] = CustodyKey(
            version=number, fernet=fernet,
            index_key=_derive_index_key(raw))
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


# The only strings that turn a protection OFF. Everything else -- including
# the empty string, "0", "false", "no", "off", and anything misspelled -- means
# the protection stays ON.
#
# WHY THIS IS NOT A STYLE PREFERENCE. `bool(os.environ.get(NAME))` is true for
# every non-empty string, so an operator who writes `..._ALLOW_PLAINTEXT_HTTP=0`
# meaning "off" gets the opposite of what they wrote: plain HTTP accepted, and
# the session cookie shipped WITHOUT its `Secure` flag. The same bug on
# `..._TRUST_FORWARDED_PROTO=false` makes the server believe a CLIENT-supplied
# `X-Forwarded-Proto: https` header and skip the TLS refusal entirely. Both
# variables exist to weaken a protection deliberately, so both are parsed so
# that only a deliberate, affirmative value can do it.
_AFFIRMATIVE = frozenset({"1", "true", "yes", "on", "y", "t"})


def _env_flag(env, name: str) -> bool:
    return str(env.get(name, "")).strip().lower() in _AFFIRMATIVE


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
    if _env_flag(env, TRUST_FORWARDED_PROTO_ENV) and (
            forwarded_proto or "").split(",")[0].strip().lower() == "https":
        return True
    return _env_flag(env, ALLOW_PLAINTEXT_ENV)


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


def _lock_down(path: str) -> None:
    """Owner-only on the database and its write-ahead log.

    `api/live.py`'s `save_session_record` has used mode 0600 since it was
    written, for a per-draft nonce that expires in two hours. This file holds
    full ESPN account sessions that stay valid for months, so it does not get
    to be the more readable of the two.

    DuckDB creates both files itself, offers no mode argument, and REFUSES to
    open a path that already exists and is not a valid database -- so
    pre-creating the file at 0600 (the trick `save_session_record` uses) is
    not available here. What closes the window instead is the directory: it is
    created 0700, so the file's brief 0644 existence is inside something no
    other user can traverse. This then tightens the file and the WAL, and is
    called again after every write because the WAL appears on the first one.

    Best effort: a chmod that fails (a filesystem with no modes, a path owned
    by somebody else) must not take down a connect. The value the mode
    protects is already encrypted; this is the second lock on the same door.
    """
    for candidate in (path, path + ".wal"):
        try:
            os.chmod(candidate, 0o600)
        except OSError:
            pass
    try:
        os.chmod(Path(path).parent, 0o700)
    except OSError:
        pass


def _connect(path: str):
    """The one connection to the custody file, opened on first use.

    Every DuckDB failure becomes `CustodyUnavailable`. The one that actually
    happens is the file lock: DuckDB allows a single writer per file, so a
    second uvicorn worker, a second replica, or a developer with a REPL open
    makes every custody request in every other process raise `IOException`.
    Uncaught, that is a 500 raised AFTER the credential has already been read
    off the wire. As `CustodyUnavailable` it is a clean, logged 503 that says
    what is wrong, and the credential is not stored by a process that could
    not have stored it anyway.
    """
    with _CONN_LOCK:
        conn = _CONNS.get(path)
        if conn is None:
            try:
                # 0700 on the directory, which is what actually protects the
                # file during the moment DuckDB creates it. `mode` applies to
                # the leaf directory and only when this call creates it, so a
                # path the operator already made keeps whatever they chose --
                # `_lock_down` still tightens the files inside it.
                Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                conn = duckdb.connect(path)
                for statement in _SCHEMA:
                    conn.execute(statement)
            except duckdb.Error as exc:
                raise CustodyUnavailable(
                    f"the credential store at {path} is not usable "
                    f"({type(exc).__name__}). Another process may hold its "
                    "write lock -- this store supports one writer.") from None
            except OSError as exc:
                raise CustodyUnavailable(
                    f"the credential store at {path} could not be opened "
                    f"({exc.strerror})") from None
            _lock_down(path)
            _CONNS[path] = conn
        return conn


# Every table the store owns, in the order a rebuild must create them. Kept
# beside _SCHEMA rather than derived from it so that adding a table forces a
# decision about whether `_compact` should carry it across.
_TABLES = ("espn_credential", "espn_session")


def _compact(path: str):
    """Rewrite the store from its live rows, so deleted ones cannot be read
    back out of the file. Returns the connection to use afterwards.

    WHY A CHECKPOINT IS NOT ENOUGH, measured rather than assumed. DuckDB is
    copy-on-write: a checkpoint writes the surviving rows to NEW blocks and
    marks the old ones free, but free blocks keep their bytes until something
    reuses them. Deleting one credential out of five and checkpointing leaves
    all five ciphertexts in the file. (The single-row case happens to come out
    clean, which is exactly how a weak test can pass over this.)

    Nothing in the storage engine erases one row's bytes, so the store is
    rebuilt instead: a fresh database containing only the live rows, swapped
    into place with `os.replace`. What is left afterwards contains what the
    tables contain and nothing else.

    AFFORDABLE BECAUSE OF WHAT THIS TABLE IS. One row per person who has
    connected an account, a few hundred bytes each; a rebuild measured about
    10ms. And it runs only when a credential is actually removed or replaced,
    which is a person clicking disconnect, a 401, or the reaper -- never on a
    read. If this table ever grows to a size where that is the wrong trade,
    the thing to change is the storage, not this guarantee.

    ATOMIC, so a crash cannot lose the store: the rebuild is written to a
    separate file and `os.replace` swaps it in one step. A crash before the
    swap leaves the original intact and a stale `.compact` file that the next
    attempt removes; a crash after it leaves the complete new one.

    THE HANDLE YOU HELD IS DEAD AFTERWARDS. This closes the connection and
    opens a new one against the new file, so any caller holding a connection
    across a mutation must ask `_connect` for it again -- which is why every
    method here fetches one at the top rather than caching it.

    Best effort about FAILING: if anything goes wrong the original file is
    left exactly as it was and the caller carries on. A store that could not
    be compacted is a store with recoverable deleted rows in it, which is
    worse than nothing -- but losing the live credentials of everyone using
    the product would be worse still.
    """
    with _CONN_LOCK:
        conn = _CONNS.get(path)
        if conn is None:
            return _connect(path)
        if "'" in path:
            # ATTACH takes no bind parameters, so a quote in the path could
            # end the string literal. No deployment has one; refusing beats
            # building SQL out of it.
            return conn
        tmp = path + ".compact"
        try:
            for stale in (tmp, tmp + ".wal"):
                if os.path.exists(stale):
                    os.unlink(stale)
            conn.execute(f"ATTACH '{tmp}' AS compacted")
            for statement in _SCHEMA:
                conn.execute(statement.replace(
                    "CREATE TABLE IF NOT EXISTS ",
                    "CREATE TABLE IF NOT EXISTS compacted."))
            for table in _TABLES:
                conn.execute(
                    f"INSERT INTO compacted.{table} SELECT * FROM {table}")
            conn.execute("DETACH compacted")
            # Closed before the swap so DuckDB folds and removes its own WAL
            # first -- replacing the file underneath an open handle would
            # leave a connection reading a file that no longer exists.
            conn.close()
            _CONNS.pop(path, None)
            if os.path.exists(path + ".wal"):
                os.unlink(path + ".wal")
            os.replace(tmp, path)
        except (duckdb.Error, OSError):
            for stale in (tmp, tmp + ".wal"):
                try:
                    os.unlink(stale)
                except OSError:
                    pass
        return _connect(path)


def _flush(conn, path: str) -> None:
    """Force the write-ahead log into the database file, then forget it.

    WHY A DELETE IS NOT A DELETE WITHOUT THIS. DuckDB writes changes to a WAL
    and only folds it into the main file when it grows past its auto-checkpoint
    threshold, which defaults to 16 MB -- tens of thousands of operations at
    these row sizes. Until then the WAL still holds the ORIGINAL INSERT of
    every row, including every credential a user has explicitly deleted. So a
    long-running API would keep a recoverable copy of every ESPN session it
    had ever been given, and "disconnect everywhere" would be a lie told to
    the one user who cared enough to ask.

    Measured on this schema: after `disconnect_everywhere` the ciphertext is
    still present in the files on disk; after a `CHECKPOINT` it is gone.

    Called after writes as well as deletes, so the WAL cannot sit there
    accumulating live credentials in the first place.
    """
    try:
        conn.execute("CHECKPOINT")
    except duckdb.Error:
        # A checkpoint can be refused while another cursor holds a
        # transaction. Not fatal and not worth failing a user's disconnect
        # over: the next mutation checkpoints again, and the row is already
        # gone from the table either way.
        pass
    _lock_down(path)


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
                 ttl_days: float | None = None, out=print, verifier=None):
        self.path = path or os.environ.get(DB_PATH_ENV) or DEFAULT_DB_PATH
        self._keys_spec = keys
        self._ttl_days = ttl_days
        # How this store proves that whoever sent an espn_s2 owns the account
        # they claim (see `pipeline/espn_identity.py`, and `connect` below for
        # why a write path needs authorization at all). Injected at
        # CONSTRUCTION rather than passed per call, so that no call site can
        # skip it by forgetting an argument; tests substitute a fake one and
        # therefore never touch the network.
        self._verifier = verifier or verify_account
        # Key versions whose absence has already been reported, so a retired
        # key logs once rather than on every request from every browser that
        # still holds a cookie minted under it.
        self._warned_versions: set = set()
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

    def account_id(self, swid: str) -> str:
        """The id anything per-account keys its own rows on.

        PUBLIC ON PURPOSE, and narrow on purpose. `api/billing.py` needs a
        stable handle for an ESPN account so a purchase can be attached to
        one, and the choice is between inventing a second identifier (another
        table, another thing to keep in step, another readable list of who
        uses this) or lending out the one already computed here.

        What it hands over is an HMAC under the current key. It reveals no
        SWID, cannot be reversed, and cannot be tested for membership by
        anybody without the key -- so a table keyed by it inherits the
        property this module exists to provide, without that table having to
        hold any ESPN material at all.
        """
        return self._row_id(swid, self.current_key)

    def account_ids(self, swid: str) -> list:
        """Every id this account could have rows under, newest key first.

        The plural is key rotation, and callers must read across all of them
        for the same reason `resolve` does: a row written under version 1
        keeps its version-1 id forever. Reading the list and writing
        `account_id` above is the same lazy-rotation bargain the credential
        rows make -- nothing is rewritten, and nothing is orphaned.
        """
        return [row_id for _key, row_id in self._candidate_ids(swid)]

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

    def connect(self, swid: str, espn_s2: str, now=None,
                cookie: str | None = None) -> MintedSession:
        """Take custody of one ESPN session and mint a browser session for it.

        THE ROW IS KEYED ON THE SESSION, NOT ON THE ACCOUNT, and that single
        choice is what makes this write path safe without asking ESPN
        anything.

        WHAT IT IS DEFENDING AGAINST, because the danger is real and has not
        gone away: SWID is public. It rides in the invite POST's query string
        and the draft socket's JOIN url, so a row keyed on a CLAIMED SWID lets
        a stranger post their own `espn_s2` under a victim's id and replace
        what is stored. The victim is not merely logged out -- their browser
        keeps a valid cookie that now resolves to the ATTACKER'S ESPN account,
        so the next feature to act on a resolved credential acts on the wrong
        account while looking perfectly healthy.

        THE OLD DEFENCE, AND WHY IT IS GONE. It was to make ESPN name the
        owner (`pipeline/espn_identity.py`) and key the row on ESPN'S ANSWER
        rather than the caller's claim. That was the right defence for a row
        keyed by name, and it fails closed -- which is exactly what it now
        does on every request, so it is worth stating plainly why rather than
        leaving a mystery 403 behind: measured against a real account,
        `fan.api.espn.com/apis/v2/fans/{SWID}` returns the same profile with
        NO cookies at all. It is a public endpoint. It cannot prove ownership
        of anything, `verify_account` refuses rather than pretend it can, and
        no other ESPN endpoint was found that distinguishes -- a private
        league proves membership in a league, not identity, and it answers 404
        for anyone whose leagues are public.

        So the name stops being a key. `HMAC(espn_s2)` is one: the id is
        derived from the SECRET, which only the browser holding it can
        present. There is no name left to squat on, so the attack above has
        nowhere to start -- a caller can only ever write the row their own
        session hashes to, and a stranger armed with a victim's SWID hashes to
        a row that has nothing to do with them.

        WHAT THIS NARROWS, STATED SO NOBODY REDISCOVERS IT AS A BUG. ESPN
        reissues `espn_s2` per sign-in, so a second browser gets its own row
        instead of attaching to an existing one, and "disconnect everywhere"
        means every browser holding THIS stored session rather than every
        browser this person has ever used. Grouping rows by the account inside
        the blob would restore the old meaning and rebuild exactly the
        name-keyed structure this change exists to remove.

        PRESENTING THE SECRET IS THE AUTHORIZATION. A connect carrying an
        `espn_s2` that is already stored is, necessarily, from somebody who
        holds that session, so it refreshes the row and mints a new browser
        cookie. The cookie is still read, for the one thing it says that the
        secret does not: a browser that already resolves to this credential is
        attaching rather than re-storing, and rewriting an identical blob
        would be work for nothing.
        """
        if not swid or not espn_s2:
            # Storing half a session buys nothing and costs the same custody
            # obligations, so it is refused rather than half-written.
            raise ValueError("both swid and espn_s2 are required")
        # The account id is still normalised and still stored -- it is what
        # every ESPN call made on this credential's behalf has to send. It is
        # simply no longer what the row is FILED under.
        owner = canonical_swid(swid)
        if not owner:
            raise ValueError("swid is not an ESPN account id")

        now = _utc(now)
        key = self.current_key
        expires_at = now + self.ttl

        # Resolved BEFORE the lock and before the store is touched, because
        # `resolve` is itself a full read path (it reaps, it slides expiry)
        # and running it inside this method's critical section would nest two
        # different jobs in one transaction for no reason.
        holder = self.resolve(cookie, now=now) if cookie else None
        conn = _connect(self.path)

        with _CONN_LOCK:
            # An existing row for this SESSION, possibly under an older key
            # version and therefore under a DIFFERENT id.
            previous = None
            for _candidate_key, candidate_id in self._candidate_ids(espn_s2):
                row = conn.execute(
                    "SELECT id, created_at FROM espn_credential WHERE id = ?",
                    [candidate_id]).fetchone()
                if row:
                    previous = row
                    break

            # THE SAME SECRET UNDER THE SAME KEY IS THE SAME ROW, and there is
            # nothing to write: the blob would be byte-identical, and
            # rewriting it would churn ciphertext (and leave the old copy in a
            # freed block) to store what is already there. This is the
            # ordinary case of somebody clicking the bookmarklet twice, or a
            # second browser sharing one ESPN session.
            #
            # A row found under a DIFFERENT id is the same secret under an
            # older key version -- rotation catching up -- which does need
            # writing, below.
            fresh_id = self._row_id(espn_s2, key)
            attached = previous is not None and previous[0] == fresh_id

            dropped: set = set()
            if attached:
                # Attach only. The stored secret, its key version and its
                # created_at are all left exactly as they were.
                credential_id = previous[0]
                replaced = False
            else:
                credential_id = fresh_id
                blob = key.fernet.encrypt(
                    json.dumps({"swid": owner, "espn_s2": str(espn_s2)},
                               separators=(",", ":")).encode("utf-8")
                ).decode("ascii")
                created_at = previous[1] if previous else now
                conn.execute("DELETE FROM espn_credential WHERE id = ?",
                             [credential_id])
                conn.execute(
                    "INSERT INTO espn_credential VALUES (?, ?, ?, ?, ?, ?)",
                    [credential_id, blob, key.version, created_at, now,
                     expires_at])
                # Rows this connect supersedes, both of which have to go rather
                # than linger: a stored `espn_s2` is a LIVE ESPN session for as
                # long as ESPN honours it, so leaving one behind is leaving a
                # working credential on disk that nobody is using and nobody
                # will notice.
                #
                #   `previous` -- the same secret under an older key version.
                #   Rotation caught up with this user, so the row moves and
                #   takes its browsers with it rather than evicting them.
                #
                #   `holder` -- the credential the CALLER'S COOKIE resolves to,
                #   when ESPN has reissued their `espn_s2` and this is
                #   therefore a different row. Superseding it is authorised by
                #   that cookie, which is the password to the very row being
                #   dropped; without this the old session would sit encrypted
                #   on disk until the reaper reached it a month later.
                dropped = {row for row in
                           (previous[0] if previous else None,
                            holder.credential_id if holder else None)
                           if row and row != credential_id}
                for stale in dropped:
                    conn.execute(
                        "UPDATE espn_session SET credential_id = ? "
                        "WHERE credential_id = ?", [credential_id, stale])
                    conn.execute("DELETE FROM espn_credential WHERE id = ?",
                                 [stale])
                replaced = True

            # 256 bits from the OS CSPRNG. This value is the password to a
            # stored ESPN account session, so it is generated the same way a
            # password reset token would be and is never derived from anything
            # guessable (no user id, no timestamp, no counter).
            minted = secrets.token_urlsafe(32)
            session_id = self._row_id(minted, key)
            conn.execute("DELETE FROM espn_session WHERE id = ?", [session_id])
            conn.execute(
                "INSERT INTO espn_session VALUES (?, ?, ?, ?, ?, ?)",
                [session_id, credential_id, key.version, now, now, expires_at])
            # The credential is alive again whether or not its secret changed,
            # so its clock restarts either way -- otherwise a user who keeps
            # connecting new browsers would still be reaped on the old one's
            # schedule.
            conn.execute(
                "UPDATE espn_credential SET last_used_at = ?, expires_at = ? "
                "WHERE id = ?", [now, expires_at, credential_id])
            # Fold the write-ahead log into the file now rather than at
            # DuckDB's 16 MB threshold, so a live credential is not left
            # sitting in a WAL that nothing will fold in for months. See
            # `_flush`.
            _flush(conn, self.path)
        if dropped:
            # A superseded credential leaves its ciphertext in a freed block,
            # which is the same residue a delete leaves and the same problem:
            # ESPN reissues `espn_s2`, so the old one is still a live session
            # for as long as ESPN honours it. Keyed on whether a row was
            # actually dropped rather than on `replaced`, because a first-ever
            # connect also writes a row and has nothing to compact away.
            # Outside the lock because `_compact` takes it again and does its
            # own bookkeeping.
            _compact(self.path)
        return MintedSession(cookie=minted, credential_id=credential_id,
                             session_id=session_id, expires_at=expires_at,
                             replaced=replaced)

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
            if self._reap(conn, now)["credentials"]:
                # `_compact` replaces the file, so the handle taken above is
                # dead from here on -- everything below uses the new one.
                conn = _compact(self.path)
            found = None
            for _key, session_id in self._candidate_ids(cookie):
                row = conn.execute(
                    "SELECT id, credential_id FROM espn_session WHERE id = ?",
                    [session_id]).fetchone()
                if row:
                    found = row
                    break
            if not found:
                # A miss is almost always an ordinary wrong or expired cookie
                # and says nothing. But it is ALSO what a retired key looks
                # like: session ids are HMAC-ed under the key too, so a cookie
                # minted under a version that is no longer configured cannot
                # be found at all, and this returns None long before the
                # `key_version` check further down could report anything. That
                # is a whole deployment's users appearing to be logged out
                # with not one line explaining why, so the rows are asked
                # directly which versions they were written under.
                self._warn_about_retired_keys(conn)
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
                _flush(conn, self.path)
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
                    # Without this the deleted row stays fully recoverable in
                    # the write-ahead log, possibly for months. See `_flush`.
                    _flush(conn, self.path)
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
            gone = self._delete_credential(conn, credential_id) >= 0
            # THE DELETION THAT HAS TO BE REAL. This is the call behind
            # "disconnect everywhere" and behind a 401, so a copy left
            # recoverable anywhere in the file would make both of them a lie
            # told to the one user who cared enough to ask. `_flush` gets it
            # out of the write-ahead log; `_compact` gets it out of the blocks
            # the checkpoint merely marked free.
            _flush(conn, self.path)
            if gone:
                _compact(self.path)
            return gone

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
            reaped = self._reap(conn, _utc(now))
            _flush(conn, self.path)
            if reaped["credentials"]:
                _compact(self.path)
            return reaped

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

    def _warn_about_retired_keys(self, conn) -> None:
        """Say once, per version, that rows exist under a key we do not hold.

        Once, because the alternative is a line per request per browser for
        every user of a deployment whose key list is wrong -- which is a log
        nobody reads, on exactly the day somebody needs to.
        """
        known = set(self.keys)
        try:
            rows = conn.execute(
                "SELECT DISTINCT key_version FROM espn_session").fetchall()
        except duckdb.Error:
            return
        for (version,) in rows:
            version = int(version)
            if version in known or version in self._warned_versions:
                continue
            self._warned_versions.add(version)
            self._out(
                f"custody: sessions exist under key version {version}, which "
                f"is not in {KEYS_ENV}. Every browser holding one appears "
                "logged out until that key is restored.")

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
