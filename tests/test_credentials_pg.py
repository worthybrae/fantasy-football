"""The credential store against a real Postgres.

Skipped unless SUPABASE_TEST_DB_URL is set, which is the whole point: the
suite in tests/test_credentials.py is the one that has to pass on a laptop
with no network, and it exercises the DuckDB backend. This file exists because
the two backends can only differ in ways a fake would agree with --
placeholder syntax, a TIMESTAMPTZ column handing back an aware datetime where
every comparison in the module expects a naive one -- so it runs the round
trip that matters against the database a deployment actually uses.

Run it against a database you are willing to write to:

    SUPABASE_TEST_DB_URL=postgresql://... .venv/bin/pytest tests/test_credentials_pg.py

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

from pipeline import credentials as cred
from pipeline import pgstore

DSN_ENV = "SUPABASE_TEST_DB_URL"
DSN = os.environ.get(DSN_ENV, "").strip()

pytestmark = pytest.mark.skipif(not DSN, reason=f"needs {DSN_ENV}")


@pytest.fixture(autouse=True)
def _postgres(monkeypatch):
    """Point every store at the test database, over conftest's delenv."""
    monkeypatch.setenv(pgstore.DSN_ENV, DSN)


def _store():
    return cred.CredentialStore(dsn=pgstore.dsn(), keys=f"1:{cred.generate_key()}",
                                out=lambda *a: None, verifier=lambda swid, s2: True)


def test_round_trip_on_postgres():
    store = _store()
    swid = "{" + str(uuid.uuid4()).upper() + "}"
    minted = store.connect(swid, "s2-" + uuid.uuid4().hex)
    try:
        got = store.resolve(minted.cookie)
        assert got is not None and got.swid == swid
    finally:
        store.forget_credential(minted.credential_id)
    assert store.resolve(minted.cookie) is None


def test_the_reaper_agrees_with_postgres_about_what_time_it_is():
    """The expiry columns are TIMESTAMPTZ and every datetime this module makes
    is naive UTC, so the two have to be reconciled somewhere. If they are not,
    Postgres reads a naive value in whatever the session's TimeZone happens to
    be and a fresh credential is already hours expired -- which is the reaper
    deleting live sessions, silently, on one deployment and not another.

    A one-minute TTL is what makes that visible: an offset of any size at all
    puts the row on the wrong side of both comparisons below.
    """
    store = cred.CredentialStore(
        dsn=pgstore.dsn(), keys=f"1:{cred.generate_key()}",
        ttl_days=1 / (24 * 60), out=lambda *a: None,
        verifier=lambda swid, s2: True)
    minted = store.connect("{" + str(uuid.uuid4()).upper() + "}",
                           "s2-" + uuid.uuid4().hex)
    try:
        store.reap()
        assert store.resolve(minted.cookie) is not None
        store.reap(now=datetime.now(timezone.utc) + timedelta(minutes=2))
        assert store.resolve(minted.cookie) is None
    finally:
        store.forget_credential(minted.credential_id)
