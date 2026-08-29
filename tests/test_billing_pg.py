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


def test_a_favourites_list_round_trips_and_replaces_in_one_transaction():
    """Favourites over the other backend.

    Here rather than only in `tests/test_account_api.py` because the store's
    two halves are exactly where the backends can disagree and a fake would
    not: the save is ONE multi-row `INSERT ... VALUES (?,?,?,?), (?,?,?,?) ...`
    whose placeholders have to survive the `?` -> `%s` rewrite, and the
    replace is a DELETE and that INSERT inside one psycopg transaction rather
    than the explicit BEGIN/COMMIT DuckDB needs.
    """
    account = "pg-test-" + uuid.uuid4().hex
    older = "pg-test-" + uuid.uuid4().hex
    six = ["p7", "p3", "p19", "p11", "p2", "p25"]
    billing.reset_for_tests(None)
    try:
        assert billing.favorites([account]) == []

        assert billing.set_favorites([account], six) == six
        # In the order it was saved, not the order the rows came back.
        assert billing.favorites([account]) == six

        five = ["p1", "p4", "p5", "p6", "p8"]
        billing.set_favorites([account], five)
        assert billing.favorites([account]) == five

        # And the rotation rule: the newest id that has rows is the answer.
        billing.set_favorites([older], six)
        assert billing.favorites([account, older]) == five
        assert billing.favorites([older]) == six
    finally:
        for name in (account, older):
            billing._db().execute(
                "DELETE FROM favorite_player WHERE account_id = ?", [name])
        billing.reset_for_tests(None)


def test_a_founder_seat_is_claimed_once_and_survives_a_key_rotation(monkeypatch):
    """The founder claim over the other backend.

    Here rather than only in tests/test_billing.py because this is the one
    statement in the module that counts and inserts in the same breath --
    `INSERT INTO founder ... SELECT ?, (SELECT count(*) FROM founder) + 1, ?
    WHERE (SELECT count(*) FROM founder) < ?` -- and a placeholder in a SELECT
    list is exactly where the `?` -> `%s` rewrite and psycopg's parameter
    typing can part company with DuckDB. A fake would not notice.

    A LIMIT NOBODY CAN REACH, because this table is shared: the database may
    already hold rows from another run, and asserting "ordinal 1" against it
    would be asserting on somebody else's history. What is asserted is what
    the shape of the statement decides -- one seat per account however many
    times it asks, and the older id's seat found from the newer one.
    """
    new = "pg-test-" + uuid.uuid4().hex
    old = "pg-test-" + uuid.uuid4().hex
    monkeypatch.setenv(billing.FOUNDERS_LIMIT_ENV, str(10 ** 9))
    billing.reset_for_tests(None)
    try:
        assert billing.is_founder([new]) is False

        ordinal = billing.claim_founder([old])
        assert ordinal is not None and ordinal >= 1
        # Idempotent, because every caller is a route that runs on every page
        # load...
        assert billing.claim_founder([old]) == ordinal
        # ...and a rotated key finds the same row rather than spending a
        # second seat on the same person.
        assert billing.claim_founder([new, old]) == ordinal
        assert billing.is_founder([new, old]) is True
        assert billing.founder_ordinal([new]) is None
    finally:
        for name in (new, old):
            billing._db().execute("DELETE FROM founder WHERE account_id = ?",
                                  [name])
        billing.reset_for_tests(None)


def test_a_full_list_hands_out_nothing_here_either(monkeypatch):
    """The `WHERE` that stops the hundred and first, over the backend where it
    is a different planner's problem.

    The limit is set to whatever the table already holds rather than to zero:
    zero is refused in Python before the statement is sent, which would leave
    the guard itself untested.
    """
    seed = "pg-test-" + uuid.uuid4().hex
    latecomer = "pg-test-" + uuid.uuid4().hex
    monkeypatch.setenv(billing.FOUNDERS_LIMIT_ENV, str(10 ** 9))
    billing.reset_for_tests(None)
    try:
        assert billing.claim_founder([seed]) is not None
        full = billing.founders_taken()
        monkeypatch.setenv(billing.FOUNDERS_LIMIT_ENV, str(full))

        assert billing.claim_founder([latecomer]) is None
        assert billing.founders_taken() == full
        assert billing.founders_left() == 0
    finally:
        for name in (seed, latecomer):
            billing._db().execute("DELETE FROM founder WHERE account_id = ?",
                                  [name])
        billing.reset_for_tests(None)
