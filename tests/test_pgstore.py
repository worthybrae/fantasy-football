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


def test_schema_creation_survives_two_workers_creating_it_at_once(monkeypatch):
    """`CREATE TABLE IF NOT EXISTS` is not atomic in Postgres: two workers
    booting together both find the table missing, and the loser gets a
    duplicate-key error out of the system catalogue rather than the "already
    there" the statement asks for. That was one 503 per cold deploy, healed
    by the next request. The loser tries once more, by which time the
    winner has committed."""
    import contextlib

    import psycopg

    class _Conn:
        def __init__(self):
            self.ran = []
            self.raised = 0

        def execute(self, sql):
            self.ran.append(sql)
            if len(self.ran) == 1:          # the first statement loses
                self.raised += 1
                raise psycopg.errors.UniqueViolation("duplicate key")

    conn = _Conn()

    class _Pool:
        def connection(self):
            return contextlib.nullcontext(conn)

    monkeypatch.setattr(pgstore, "pool", _Pool)
    pgstore.create_schema(("CREATE TABLE IF NOT EXISTS x (a INT)",), "test store")
    assert conn.raised == 1
    assert len(conn.ran) == 2, "the losing statement was not retried"


def test_schema_creation_gives_up_after_one_retry(monkeypatch):
    """A statement that fails the same way twice is not a race any more."""
    import contextlib

    import psycopg

    class _Conn:
        def __init__(self):
            self.ran = []

        def execute(self, sql):
            self.ran.append(sql)
            raise psycopg.errors.DuplicateTable("relation exists")

    conn = _Conn()

    class _Pool:
        def connection(self):
            return contextlib.nullcontext(conn)

    monkeypatch.setattr(pgstore, "pool", _Pool)
    with pytest.raises(pgstore.StoreError) as caught:
        pgstore.create_schema(("CREATE TABLE IF NOT EXISTS x (a INT)",),
                              "test store")
    assert "test store" in str(caught.value)
    assert len(conn.ran) == 2


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
