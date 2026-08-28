"""Live-draft session records against a real Postgres.

Skipped unless SUPABASE_TEST_DB_URL is set. tests/test_live_records.py covers
the file backend and is the one that has to pass with no network; this file
covers the two things only the Postgres backend does -- a JSONB round trip,
and keeping the ESPN token out of a shared database in plaintext.

Run it against a database you are willing to write to:

    SUPABASE_TEST_DB_URL=postgresql://... .venv/bin/pytest tests/test_live_records_pg.py

SUPABASE_TEST_DB_URL rather than SUPABASE_DB_URL, because tests/conftest.py
clears that one for every test: it is the only thing choosing between a DuckDB
file and a shared Postgres, so a developer with it exported would otherwise
send the whole suite at a deployment's database. The fixture below sets it
from this variable, for the length of one test.
"""
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from api import live, live_records
from pipeline import credentials as cred
from pipeline import pgstore

DSN_ENV = "SUPABASE_TEST_DB_URL"
DSN = os.environ.get(DSN_ENV, "").strip()

pytestmark = pytest.mark.skipif(not DSN, reason=f"needs {DSN_ENV}")


@pytest.fixture(autouse=True)
def _postgres(monkeypatch):
    """Point every store at the test database, over conftest's delenv."""
    monkeypatch.setenv(pgstore.DSN_ENV, DSN)


def _record(hours_old=0, token="tok-abc"):
    saved = datetime.now(timezone.utc) - timedelta(hours=hours_old)
    return {
        "version": live.SESSION_RECORD_VERSION,
        "saved_at": saved.isoformat(),
        "league_id": "777",
        "team_id": "3",
        "season": "2026",
        "swid": "{SOME-SWID}",
        "token": token,
        "my_slot": 4,
    }


@pytest.fixture
def store(monkeypatch):
    """A store with a custody key configured, which is the state any
    deployment holding other people's sessions is in."""
    monkeypatch.setenv(cred.KEYS_ENV, f"1:{cred.generate_key()}")
    cred.reset_default_store()
    yield live_records.record_store("unused-by-the-postgres-backend")
    cred.reset_default_store()


def test_a_record_round_trips_through_postgres(store):
    sid = "test-" + uuid.uuid4().hex
    try:
        store.save(sid, _record())
        got = store.load(sid)
        assert got is not None
        assert got["token"] == "tok-abc"
        assert got["my_slot"] == 4
        assert store.load_all()[sid]["league_id"] == "777"
    finally:
        store.delete(sid)
    assert store.load(sid) is None


def test_the_token_is_not_in_the_shared_database_in_plaintext(store):
    """A 0600 file on one box and a table every worker can read are not the
    same exposure, and this token is enough to rejoin somebody's draft. With a
    custody key configured it goes in encrypted under that key -- the same one
    pipeline/credentials.py holds ESPN sessions under."""
    sid = "test-" + uuid.uuid4().hex
    try:
        store.save(sid, _record(token="a-real-looking-token"))
        stored = store._run("SELECT record FROM live_session WHERE sid = ?",
                            [sid])[0][0]

        assert "token" not in stored
        assert "a-real-looking-token" not in str(stored)
        # And it is still readable by the deployment that wrote it.
        assert store.load(sid)["token"] == "a-real-looking-token"
    finally:
        store.delete(sid)


def test_a_stale_record_is_deleted_rather_than_left_in_the_table(store):
    """The twelve-hour rule is the only automatic bound on how long a dead
    token stays anywhere, and a row nobody deletes is a token nobody deletes."""
    sid = "test-" + uuid.uuid4().hex
    try:
        store.save(sid, _record(hours_old=13))
        assert store.load(sid) is None
        assert store._run(
            "SELECT count(*) FROM live_session WHERE sid = ?", [sid])[0][0] == 0
    finally:
        store.delete(sid)
