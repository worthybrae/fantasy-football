"""Where a live draft's session record is kept, once there is more than one.

WHAT A RECORD IS, because it decides everything below. It is an ESPN
draftSecurity token plus the SWID it was minted for -- together, enough to
rejoin somebody's draft. `api/live.py` writes one so that a deployment which
restarts mid-draft can pick the socket back up without the owner touching
anything, and its docstring is worth reading for why that is worth the
exposure at all.

THE PROBLEM THIS SOLVES. There is exactly ONE of those files per deployment,
named after the draft database, and it is the whole state: the second person
to connect overwrites the first, and a restart then restores the wrong draft
or none at all. That is fine for one person on one laptop and wrong for a
deployment several people are drafting through, which is the direction this
is going.

So a record gets a session id and there can be many. Two backends, chosen the
same way every other store here chooses: files while SUPABASE_DB_URL is
unset, Postgres once it is set, because a record written by one worker is
useless if the worker that restarts cannot read it.

WHAT DOES NOT CHANGE. The file backend is as readable as the file it
replaces and no more: mode 0600, in a 0700 directory, written whole via
os.replace. The twelve-hour rule is the same rule, and it still DELETES
rather than merely ignoring -- it is the only automatic bound on how long a
dead token stays anywhere.

WHAT IMPROVES ON POSTGRES. A shared database is a worse place for a record
than a 0600 file is, so the record does not go in as plaintext: the WHOLE
document rides under the custody key, the same Fernet key
`pipeline/credentials.py` encrypts ESPN sessions with. The token is the part
that rejoins a draft, but the rest of the row is a SWID, a league id and a
team id -- who drafts here, for whom, in which league -- and a table listing
that for every reader is exactly the thing this project does not keep
anywhere else. So the row is a session id, one ciphertext, and a timestamp
the age rule can read without a key.

Without a configured key the document is stored as it is, which is the
exposure today's file already has -- but a deployment holding other people's
sessions has that key by definition. Rows written under the earlier shape
(the token alone encrypted, the rest in the clear) are still read, and every
configured key version is tried, so neither a deploy nor a key rotation
loses a draft that is running.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone

from pipeline import credentials as cred
from pipeline import pgstore

# The record for a deployment that had only ever written one. `load_all`
# surfaces the legacy file under this id so a restart that lands mid-draft on
# the day this ships still finds the draft it was running.
LEGACY_SID = "__default__"

# A session id becomes a filename. Whatever ends up minting these upstream is
# handling identifiers that came off a request, and a `../` in one writes an
# ESPN token wherever the caller likes -- so the shape is refused rather than
# sanitised, because sanitising is where the interesting bugs live.
_SID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def _live():
    """`api.live`, imported on use rather than at the top of this file.

    That module owns the record format, the twelve-hour rule and the legacy
    path, and it is the module that will import THIS one once the store is
    wired in. Importing it here at module scope would be the cycle.
    """
    from api import live

    return live


def _checked(sid: str) -> str:
    if not isinstance(sid, str) or not _SID.match(sid) or sid in (".", ".."):
        raise ValueError(f"not a usable session id: {sid!r}")
    return sid


def _wellformed(record) -> bool:
    """The same up-front rejections `api.live.load_session_record` makes.

    A record that fails any of these cannot open a socket, so returning it
    would only move the failure somewhere with less context. The team id is
    checked as an integer because `_slot_for_team` compares integers.
    """
    live = _live()
    if not isinstance(record, dict):
        return False
    if record.get("version") != live.SESSION_RECORD_VERSION:
        return False
    if not all(record.get(k) for k in ("league_id", "team_id", "swid", "token")):
        return False
    try:
        int(record["team_id"])
    except (TypeError, ValueError):
        return False
    return True


def _too_old(record, now=None) -> bool:
    """Whether this record describes a draft that is over.

    A record with no usable timestamp counts as too old, deliberately: the age
    rule could never fire for it otherwise, which means the token would sit
    wherever it is stored forever. Same judgement `load_session_record` makes.
    """
    try:
        saved_at = datetime.fromisoformat(record.get("saved_at") or "")
    except (TypeError, ValueError):
        return True
    if saved_at.tzinfo is None:
        return True
    now = now or datetime.now(timezone.utc)
    age = (now - saved_at).total_seconds()
    return age > _live().SESSION_RECORD_MAX_AGE_SECONDS


class SessionRecordStore:
    """One record per live draft, keyed by a session id.

    Four methods and no update-in-place: a record is small, written whole, and
    replaced whole. `load` is the only one with a clock in it, because the age
    rule has to fire on a deployment that never calls anything else.
    """

    def save(self, sid: str, record: dict) -> None:
        raise NotImplementedError

    def load(self, sid: str, now=None) -> dict | None:
        raise NotImplementedError

    def load_all(self, now=None) -> dict:
        raise NotImplementedError

    def delete(self, sid: str) -> None:
        raise NotImplementedError


class _FileRecords(SessionRecordStore):
    """A directory of records beside the draft database.

    Derived from the database path for the same reason the single file was
    (see `api.live.session_record_path`): DRAFT_DB_PATH is how a second
    instance runs against a copy, and a fixed location would have one of them
    restoring the other's draft.
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.dir = self.db_path + ".live-sessions"

    def _path(self, sid: str) -> str:
        return os.path.join(self.dir, _checked(sid) + ".json")

    def save(self, sid: str, record: dict) -> None:
        path = self._path(sid)
        # 0700, which is what actually protects a record during the moment it
        # is being created. The file mode below is the second lock on the
        # same door.
        os.makedirs(self.dir, exist_ok=True, mode=0o700)
        tmp = f"{path}.tmp"
        # Created 0600 at open() time rather than chmod-ed afterwards: a chmod
        # leaves a window in which the token is world-readable. Written to a
        # temporary file and os.replace-d so a concurrent reader sees a whole
        # record or nothing -- a torn read would lose a live draft.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh)
        os.replace(tmp, path)

    def _read(self, path: str):
        try:
            with open(path) as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            # Unreadable, or not JSON. Left in place rather than deleted: this
            # is somebody's token, and a transient read failure must not be
            # the reason it is thrown away.
            return None

    def load(self, sid: str, now=None) -> dict | None:
        path = self._path(sid)
        record = self._read(path)
        if record is None and sid == LEGACY_SID:
            # The record the old single-file code wrote, complete with its own
            # age rule and its own delete.
            return _live().load_session_record(self.db_path, now)
        if record is None or not _wellformed(record):
            return None
        if _too_old(record, now):
            self.delete(sid)
            return None
        return record

    def load_all(self, now=None) -> dict:
        found = {}
        try:
            names = sorted(os.listdir(self.dir))
        except OSError:
            names = []
        for name in names:
            if not name.endswith(".json"):
                continue
            sid = name[: -len(".json")]
            if not _SID.match(sid):
                # Nothing this class writes, so somebody else put it there.
                # Skipped rather than refused: a stray file in the directory
                # must not be what stops a restore.
                continue
            record = self.load(sid, now)
            if record is not None:
                found[sid] = record
        if LEGACY_SID not in found:
            legacy = _live().load_session_record(self.db_path, now)
            if legacy is not None:
                found[LEGACY_SID] = legacy
        return found

    def delete(self, sid: str) -> None:
        try:
            os.unlink(self._path(sid))
        except FileNotFoundError:
            pass
        if sid == LEGACY_SID:
            # Both places, or the token stays on disk and the next restore
            # picks it straight back up -- a delete that reports success and
            # changes nothing.
            _live().clear_session_record(self.db_path)


_PG_SCHEMA = (
    # JSONB rather than a column per field: this record's shape belongs to
    # api/live.py and has changed before (`my_slot` arrived after the rest).
    # A version number inside the document is how that is already handled.
    """CREATE TABLE IF NOT EXISTS live_session (
        sid TEXT PRIMARY KEY,
        record JSONB NOT NULL,
        saved_at TIMESTAMPTZ NOT NULL)""",
)

# THE WHOLE DOCUMENT, encrypted, as the only key in the stored object. A row
# is then `{"blob": "<fernet>"}` and says nothing else about who is drafting.
_BLOB = "blob"

# The earlier shape: `token` lifted out of the document and stored encrypted
# under this key, with league id, team id and swid beside it in the clear.
# Still READ -- a deploy must not lose the drafts that are running -- and
# never written. Named differently from `token` so that a record written with
# a key and read without one is obviously missing its token rather than
# silently carrying ciphertext where the socket expects a nonce.
_TOKEN_BLOB = "token_blob"

_PG_READY = False
_PG_LOCK = threading.Lock()


def _write_key():
    """The custody key a new record rides under, or None if there is not one.

    None is the same exposure the file has today, which is the honest floor:
    a deployment with no custody key is not holding anybody else's ESPN
    sessions either. A deployment that is has this key by definition.

    The NEWEST configured version, exactly like every other write under
    custody: adding a key version changes what new rows are written under
    and nothing else.
    """
    try:
        return cred.default_store().current_key.fernet
    except cred.CustodyUnavailable:
        return None


def _read_keys() -> list:
    """Every configured custody key, newest version first.

    ROTATION IS WHY THIS IS A LIST, and why it is not the current key alone.
    Adding a key version is not a migration anywhere else in this project
    (see `pipeline/credentials.py` and its `_candidate_ids`): a row written
    under version 1 keeps version 1 forever and lookups try every version.
    A record here carries no version of its own -- it is one blob inside a
    JSONB document -- so the keys are simply tried newest first. That costs
    one failed decrypt per older version on a rotation day and nothing on
    any other day.

    What it buys is the whole point: read under the current key alone, a
    rotation made every live draft's token undecryptable, `load` answered
    None for a record that was perfectly good, and the restore brought the
    room back with no socket in it -- mid-draft, on the one morning
    somebody was rotating keys.
    """
    try:
        keys = cred.default_store().keys
    except cred.CustodyUnavailable:
        return []
    return [keys[version].fernet for version in sorted(keys, reverse=True)]


def _decrypt(blob):
    """`blob` under whichever configured key wrote it, or None for none of
    them. Bytes out; the callers decode what they know they wrote."""
    try:
        raw = str(blob).encode("ascii")
    except (UnicodeEncodeError, AttributeError):
        return None
    for key in _read_keys():
        try:
            return key.decrypt(raw)
        except Exception:      # noqa: BLE001 -- InvalidToken (this key did
            continue           # not write it), or not a Fernet token at all
    return None


class _PgRecords(SessionRecordStore):
    """The same records in Postgres, so any worker can restore any draft."""

    def __init__(self):
        # Imported here rather than at module scope so a checkout with no DSN
        # never needs the driver at all.
        import psycopg

        self._error = psycopg.Error
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the table once per process, not once per store.

        `record_store` caches, so in a server this runs once either way -- but
        the DDL is three round trips to a database in another region, and
        "the caller might construct one per request" is exactly the assumption
        that made this worth guarding rather than trusting.
        """
        global _PG_READY
        if _PG_READY:
            return
        with _PG_LOCK:
            if _PG_READY:
                return
            # Through pgstore rather than `self._run`, for the race two
            # workers booting together run into. See pgstore.create_schema.
            pgstore.create_schema(_PG_SCHEMA, "live-session store")
            _PG_READY = True

    def _run(self, sql: str, params=()) -> list:
        try:
            with pgstore.pool().connection() as conn:
                cursor = conn.execute(sql.replace("?", "%s"),
                                      [pgstore.to_pg(p) for p in params] or None)
                if cursor.description is None:
                    return []
                return [tuple(pgstore.from_pg(value) for value in row)
                        for row in cursor.fetchall()]
        except self._error as exc:
            raise pgstore.StoreError(
                "the live-session store is not usable "
                f"({type(exc).__name__})") from None

    def save(self, sid: str, record: dict) -> None:
        from psycopg.types.json import Jsonb

        body = dict(record)
        key = _write_key()
        if key is not None:
            # One ciphertext, not a document with one encrypted field: see
            # the module docstring for what the other fields are worth to
            # somebody reading this table.
            body = {_BLOB: key.encrypt(
                json.dumps(body).encode("utf-8")).decode("ascii")}
        saved_at = datetime.now(timezone.utc)
        self._run(
            "INSERT INTO live_session (sid, record, saved_at) VALUES (?, ?, ?) "
            "ON CONFLICT (sid) DO UPDATE SET record = excluded.record, "
            "saved_at = excluded.saved_at",
            [_checked(sid), Jsonb(body), saved_at])

    def _decrypted(self, body):
        """The record as its caller expects it, or None if the token is gone.

        A record whose token cannot be decrypted is NOT deleted, for the same
        reason `pipeline/credentials.py` does not delete on a failed decrypt:
        the likeliest cause is a key list that is temporarily wrong, and
        reacting to a misconfigured environment by destroying live drafts is
        worse than any outage.

        Every configured key version is tried (see `_read_keys`), so a
        record written before a rotation still opens after one -- and both
        stored shapes are read: the whole-document blob this writes now, and
        the token-only blob rows written before it did.
        """
        if not isinstance(body, dict):
            return body
        if _BLOB in body:
            raw = _decrypt(body[_BLOB])
            if raw is None:
                return None
            try:
                record = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
            return record if isinstance(record, dict) else None
        if _TOKEN_BLOB not in body:
            return body                 # no key was configured when it was
        body = dict(body)               # written; stored as it came in
        token = _decrypt(body.pop(_TOKEN_BLOB))
        if token is None:
            return None
        body["token"] = token.decode("utf-8")
        return body

    def load(self, sid: str, now=None) -> dict | None:
        rows = self._run("SELECT record FROM live_session WHERE sid = ?",
                         [_checked(sid)])
        if not rows:
            return None
        record = self._decrypted(rows[0][0])
        if record is None or not _wellformed(record):
            return None
        if _too_old(record, now):
            self.delete(sid)
            return None
        return record

    def load_all(self, now=None) -> dict:
        found = {}
        for sid, body in self._run("SELECT sid, record FROM live_session"):
            record = self._decrypted(body)
            if record is None or not _wellformed(record):
                continue
            if _too_old(record, now):
                self.delete(sid)
                continue
            found[sid] = record
        return found

    def delete(self, sid: str) -> None:
        self._run("DELETE FROM live_session WHERE sid = ?", [_checked(sid)])


# One store per database path, because callers build one per request and a
# store is not free to build: the Postgres one opens a pool connection to
# create its table, and neither backend holds anything a caller would want a
# private copy of. Keyed on None when Postgres is in play, since the database
# path means nothing there and every path would otherwise get its own.
_STORES: dict = {}
_STORES_LOCK = threading.Lock()


def _forget_stores() -> None:
    """Drop the cached stores and the schema flag with the pool they were
    built against."""
    global _PG_READY
    with _PG_LOCK:
        _PG_READY = False
    with _STORES_LOCK:
        _STORES.clear()


pgstore.on_close(_forget_stores)


def record_store(db_path: str) -> SessionRecordStore:
    """The store this process should use. See the module docstring."""
    on_postgres = pgstore.enabled()
    key = None if on_postgres else str(db_path)
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            store = _PgRecords() if on_postgres else _FileRecords(str(db_path))
            _STORES[key] = store
        return store
