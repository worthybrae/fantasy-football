"""api.live_records: where a live draft's session record is kept.

WHAT THE RECORD IS, because it decides how hard these tests push. It holds an
ESPN draftSecurity token and a SWID, which together are enough to rejoin
somebody's draft. api/live.py has kept exactly one of them, in one file next
to the database, since the feature was written -- which is right for one
draft on one machine and wrong the moment two people are drafting through the
same deployment. These are the tests for keeping several, and for not making
the token any more readable than it already was on the way.

The file backend is what runs here. The Postgres one is selected by
SUPABASE_DB_URL and is covered by the stores' own pg tests -- the suite must
not need a network.
"""
import json
import os
import stat
from datetime import datetime, timedelta, timezone

import pytest

from api import live, live_records
from pipeline import credentials as cred


def _record(now=None, token="tok-abc", league="777", team="3"):
    return {
        "version": live.SESSION_RECORD_VERSION,
        "saved_at": (now or datetime.now(timezone.utc)).isoformat(),
        "league_id": league,
        "team_id": team,
        "season": "2026",
        "swid": "{SOME-SWID}",
        "token": token,
        "my_slot": 4,
    }


@pytest.fixture
def store(tmp_path):
    return live_records.record_store(str(tmp_path / "draft.duckdb"))


class _FakePgRecords(live_records._PgRecords):
    """The Postgres backend with its table in a dictionary.

    WHY THIS IS HERE AND NOT ONLY IN tests/test_live_records_pg.py: that
    file needs a real database and is skipped without one, and this suite
    must pass with no network. What these tests exercise is not the driver
    but the ENCODING -- what this class puts in the `record` column and
    what it can read back out of it -- which is its own code either way.
    `_run` is the whole seam: every statement the class issues goes through
    it, so replacing it with a dict is replacing Postgres and nothing else.
    """

    def __init__(self):
        self.rows: dict = {}

    def _run(self, sql: str, params=()):
        if sql.startswith("INSERT"):
            sid, body, _saved_at = params
            self.rows[sid] = getattr(body, "obj", body)   # unwrap Jsonb
            return []
        if sql.startswith("SELECT record"):
            row = self.rows.get(params[0])
            return [] if row is None else [(row,)]
        if sql.startswith("SELECT sid, record"):
            return list(self.rows.items())
        if sql.startswith("DELETE"):
            self.rows.pop(params[0], None)
            return []
        raise AssertionError(f"unexpected statement: {sql}")


@pytest.fixture
def keys(monkeypatch):
    """Configure the custody key list for one test and rebuild the store."""
    def use(spec: str) -> None:
        monkeypatch.setenv(cred.KEYS_ENV, spec)
        cred.reset_default_store()
    yield use
    cred.reset_default_store()


def test_a_stored_record_says_nothing_about_who_is_drafting(keys):
    """The token is what rejoins a draft, and it was the only field
    encrypted. The rest of the row is a SWID, a league id and a team id --
    who drafts here, for whom, in which league -- and a shared table listing
    that for every reader is the thing this project keeps nowhere else. The
    whole document goes in as one ciphertext."""
    keys(f"1:{cred.generate_key()}")
    store = _FakePgRecords()
    store.save("draft-a", _record(token="a-real-looking-token"))

    stored = store.rows["draft-a"]
    assert set(stored) == {"blob"}
    for secret in ("a-real-looking-token", "{SOME-SWID}", "777"):
        assert secret not in str(stored)
    # And the deployment that wrote it reads it back whole.
    got = store.load("draft-a")
    assert got["token"] == "a-real-looking-token"
    assert got["swid"] == "{SOME-SWID}" and got["league_id"] == "777"
    assert got["my_slot"] == 4


def test_the_earlier_row_shape_is_still_read(keys):
    """A deploy must not lose the drafts that are running. Rows written
    before the whole document was encrypted carry the token alone under
    `token_blob`, with the rest beside it in the clear."""
    key = cred.generate_key()
    keys(f"1:{key}")
    store = _FakePgRecords()
    fernet = live_records._write_key()
    old = _record()
    old.pop("token")
    old["token_blob"] = fernet.encrypt(b"tok-abc").decode("ascii")
    store.rows["draft-old"] = old

    got = store.load("draft-old")
    assert got["token"] == "tok-abc"
    assert got["league_id"] == "777"
    assert store.load_all()["draft-old"]["token"] == "tok-abc"


def test_without_a_custody_key_the_document_is_stored_as_it_came_in(keys):
    """The honest floor, and the same one the file backend has: a
    deployment with no custody key is not holding anybody else's ESPN
    sessions either. It must still work."""
    keys("")
    store = _FakePgRecords()
    store.save("draft-a", _record(token="tok-plain"))
    assert store.rows["draft-a"]["token"] == "tok-plain"
    assert store.load("draft-a")["token"] == "tok-plain"


def test_a_record_written_before_a_key_rotation_still_opens_after_one(keys):
    """Adding a key version is not a migration anywhere else in this project
    (see pipeline/credentials.py), and it must not be one here. Read under
    the newest key alone, a rotation turned every live draft's token into
    something that would not decrypt: `load` answered None for a perfectly
    good record and the restart brought the room back with no socket in
    it, on the one morning somebody was rotating keys."""
    one, two = cred.generate_key(), cred.generate_key()
    keys(f"1:{one}")
    store = _FakePgRecords()
    store.save("draft-a", _record(token="tok-before"))

    keys(f"1:{one},2:{two}")                    # the rotation
    assert store.load("draft-a")["token"] == "tok-before"
    store.save("draft-b", _record(token="tok-after"))
    assert store.load("draft-b")["token"] == "tok-after"
    assert store.load("draft-a")["token"] == "tok-before"

    # And once version 1 is retired, its record is unreadable -- but still
    # there, because destroying a live draft over a key list that might
    # simply be wrong today is worse than any outage.
    keys(f"2:{two}")
    assert store.load("draft-a") is None
    assert "draft-a" in store.rows
    assert store.load("draft-b")["token"] == "tok-after"


def test_a_saved_record_comes_back_and_a_deleted_one_does_not(store):
    store.save("draft-a", _record())

    got = store.load("draft-a")
    assert got is not None
    assert got["token"] == "tok-abc"
    assert got["league_id"] == "777"

    store.delete("draft-a")
    assert store.load("draft-a") is None


def test_two_drafts_do_not_overwrite_each_other(store):
    """The whole reason this exists: one file per deployment meant the second
    person to connect took the first one's session with them."""
    store.save("draft-a", _record(token="tok-a", league="111"))
    store.save("draft-b", _record(token="tok-b", league="222"))

    assert store.load("draft-a")["token"] == "tok-a"
    assert store.load("draft-b")["token"] == "tok-b"
    assert set(store.load_all()) == {"draft-a", "draft-b"}


def test_the_token_is_never_more_readable_than_the_file_it_replaced(tmp_path, store):
    """api/live.py has written that file 0600 since it was written, for a
    token that is live for a couple of hours. A directory of them does not get
    to be the more readable of the two."""
    store.save("draft-a", _record())
    directory = tmp_path / "draft.duckdb.live-sessions"
    written = directory / "draft-a.json"

    assert stat.S_IMODE(written.stat().st_mode) == 0o600
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_a_stale_record_is_dropped_and_deleted(tmp_path, store):
    """Thirteen hours. The token is a two-hour nonce, so a record this old
    describes a draft that is over -- and the only automatic bound on how long
    a dead token sits on disk is this rule firing. Ignoring it would leave the
    file there forever."""
    store.save("draft-a", _record(now=datetime.now(timezone.utc) - timedelta(hours=13)))

    assert store.load("draft-a") is None
    assert not (tmp_path / "draft.duckdb.live-sessions" / "draft-a.json").exists()


def test_a_record_that_is_not_usable_is_not_returned(store):
    """The same up-front rejections api/live.load_session_record makes: a
    version this build does not know, and a record missing something the
    socket cannot open without."""
    unknown_version = _record()
    unknown_version["version"] = live.SESSION_RECORD_VERSION + 1
    store.save("odd-version", unknown_version)

    no_token = _record()
    no_token["token"] = ""
    store.save("no-token", no_token)

    not_a_team = _record(team="not-a-number")
    store.save("bad-team", not_a_team)

    assert store.load("odd-version") is None
    assert store.load("no-token") is None
    assert store.load("bad-team") is None
    assert store.load_all() == {}


def test_the_record_written_by_the_old_single_file_path_is_still_found(tmp_path, store):
    """A deployment that restarts mid-draft after this ships has the record the
    OLD code wrote, in the old place. Missing it would lose exactly the draft
    this feature exists to restore."""
    db_path = str(tmp_path / "draft.duckdb")
    live.save_session_record(db_path, "888", 3, 2026, "{SWID}", "legacy-token")

    everything = store.load_all()
    assert set(everything) == {"__default__"}
    assert everything["__default__"]["token"] == "legacy-token"
    assert store.load("__default__")["league_id"] == "888"


def test_deleting_the_legacy_record_really_removes_the_file(tmp_path, store):
    """Otherwise the token stays on disk and every restore picks it up again --
    a delete that reports success and changes nothing."""
    db_path = str(tmp_path / "draft.duckdb")
    live.save_session_record(db_path, "888", 3, 2026, "{SWID}", "legacy-token")

    store.delete("__default__")

    assert not os.path.exists(live.session_record_path(db_path))
    assert store.load_all() == {}


def test_a_stale_legacy_record_is_dropped_too(tmp_path, store):
    db_path = str(tmp_path / "draft.duckdb")
    path = live.save_session_record(db_path, "888", 3, 2026, "{SWID}", "old-token")
    body = json.loads(open(path).read())
    body["saved_at"] = (datetime.now(timezone.utc) - timedelta(hours=13)).isoformat()
    open(path, "w").write(json.dumps(body))

    assert store.load_all() == {}
    assert not os.path.exists(path)


def test_a_session_id_cannot_reach_out_of_its_own_directory(store):
    """The sid becomes a filename, and the caller upstream of this is a draft
    identifier that came off a request. A traversal here writes a token
    wherever the attacker likes, so it is refused rather than sanitised."""
    for bad in ("../escape", "nested/sid", ".", "..", ""):
        with pytest.raises(ValueError):
            store.save(bad, _record())
        with pytest.raises(ValueError):
            store.load(bad)
