"""The credential store against a real Postgres.

Skipped unless SUPABASE_DB_URL is set, which is the whole point: the suite in
tests/test_credentials.py is the one that has to pass on a laptop with no
network, and it exercises the DuckDB backend. This file exists because the
two backends can only differ in ways a fake would agree with -- placeholder
syntax, a TIMESTAMPTZ column handing back an aware datetime where every
comparison in the module expects a naive one -- so it runs the round trip
that matters against the database a deployment actually uses.
"""
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from pipeline import credentials as cred
from pipeline import pgstore

pytestmark = pytest.mark.skipif(not os.environ.get(pgstore.DSN_ENV),
                                reason="needs SUPABASE_DB_URL")


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
