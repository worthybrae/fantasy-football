"""The Postgres pool helper, and specifically its off switch.

Every store that grew a Postgres backend asks `pgstore.enabled()` first, so
the property worth testing here is the one the whole test suite depends on:
with no SUPABASE_DB_URL in the environment there is no pool, no import of a
driver, and no network.
"""
import pytest

from pipeline import pgstore


def test_disabled_without_the_variable(monkeypatch):
    monkeypatch.delenv(pgstore.DSN_ENV, raising=False)
    assert pgstore.enabled() is False
    assert pgstore.dsn() is None


def test_enabled_with_the_variable(monkeypatch):
    monkeypatch.setenv(pgstore.DSN_ENV, "postgresql://u:p@h:5432/db")
    assert pgstore.enabled() is True
    assert pgstore.dsn() == "postgresql://u:p@h:5432/db"


def test_pool_refuses_when_disabled(monkeypatch):
    monkeypatch.delenv(pgstore.DSN_ENV, raising=False)
    pgstore.close()
    with pytest.raises(pgstore.StoreError):
        pgstore.pool()
