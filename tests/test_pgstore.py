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


def test_closing_the_pool_tells_the_stores_that_cached_something_about_it():
    """Every store that creates its schema once per process keeps a flag
    saying it has. A flag that outlives the pool it was set against is a store
    that never creates its tables again -- so `close` is what resets them,
    through this registry."""
    fired = []

    def note():
        fired.append(1)

    try:
        pgstore.on_close(note)
        pgstore.close()
        assert len(fired) == 1

        # Registered once, however many times it is offered -- a module that
        # registers at import must not accumulate a callback per reload.
        pgstore.on_close(note)
        pgstore.close()
        assert len(fired) == 2
    finally:
        # The registry is process-wide and has no unregister: a test that left
        # its callback in it would fire on every later close in the run.
        pgstore._on_close.remove(note)
