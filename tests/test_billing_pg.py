"""Entitlements against a real Postgres.

Skipped unless SUPABASE_TEST_DB_URL is set. tests/test_billing.py is the suite
that has to pass on a laptop with no network and it exercises the DuckDB
backend; this file exists for the handful of things the two backends can
disagree about and a fake would not -- `?` versus `%s`, and whether an INSERT
that hit a conflict can be told apart from one that wrote a row.

Run it against a database you are willing to write to:

    SUPABASE_TEST_DB_URL=postgresql://... .venv/bin/pytest tests/test_billing_pg.py

SUPABASE_TEST_DB_URL rather than SUPABASE_DB_URL, because tests/conftest.py
clears that one for every test: it is the only thing choosing between a DuckDB
file and a shared Postgres, so a developer with it exported would otherwise
send the whole suite at a deployment's database. The fixture below sets it
from this variable, for the length of one test.
"""
import os
import uuid

import pytest

from api import billing
from pipeline import pgstore

DSN_ENV = "SUPABASE_TEST_DB_URL"
DSN = os.environ.get(DSN_ENV, "").strip()

pytestmark = pytest.mark.skipif(not DSN, reason=f"needs {DSN_ENV}")


@pytest.fixture(autouse=True)
def _postgres(monkeypatch):
    """Point every store at the test database, over conftest's delenv."""
    monkeypatch.setenv(pgstore.DSN_ENV, DSN)


def test_a_granted_draft_is_entitled_and_a_refund_takes_it_away():
    account = "pg-test-" + uuid.uuid4().hex
    intent = "pi_" + uuid.uuid4().hex
    league = str(uuid.uuid4().int % 10**9)
    billing.reset_for_tests(None)
    try:
        assert billing.entitled([account], league, 2026) is False

        billing.grant(account, league, 2026, checkout_session="cs_test",
                      payment_intent=intent)
        assert billing.entitled([account], league, 2026) is True
        # The same three ways to be wrong about what was bought that the
        # DuckDB suite checks: another league, another season, another buyer.
        assert billing.entitled([account], "0", 2026) is False
        assert billing.entitled([account], league, 2025) is False
        assert billing.entitled(["somebody-else"], league, 2026) is False

        assert billing.revoke(intent) == 1
        assert billing.entitled([account], league, 2026) is False
    finally:
        billing._db().execute("DELETE FROM entitlement WHERE account_id = ?",
                              [account])
        billing.reset_for_tests(None)


def test_a_replayed_stripe_event_is_only_acted_on_once():
    """The idempotency claim, which is the one statement whose shape had to
    change for Postgres: DuckDB answers an INSERT with a row count and
    Postgres does not, so both are now asked with RETURNING instead."""
    event = "evt_" + uuid.uuid4().hex
    billing.reset_for_tests(None)
    try:
        assert billing._first_time(event, "checkout.session.completed") is True
        assert billing._first_time(event, "checkout.session.completed") is False
    finally:
        billing._db().execute("DELETE FROM billing_event WHERE event_id = ?",
                              [event])
        billing.reset_for_tests(None)
