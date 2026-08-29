"""The Postgres pool helper, and specifically its off switch.

Every store that grew a Postgres backend asks `pgstore.enabled()` first, so
the property worth testing here is the one the whole test suite depends on:
with no SUPABASE_DB_URL in the environment there is no pool, no import of a
driver, and no network.

AND WHAT COUNTS AS A DSN. The variable is named for a database URL and holds
whatever somebody put in it -- the case that started this file's second half
was the project URL, `https://<ref>.supabase.co`, set on the deployment
because that is the value the Supabase dashboard shows first. Handed to
psycopg it fails every custody, billing and live-session call at the far end
of a boot, so it is refused here instead, and turned into a real DSN where
the password to go with it is also set.
"""
import logging

import pytest

from pipeline import pgstore


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """None of the four variables, and nothing said yet.

    `tests/conftest.py` scrubs these for the whole suite; this repeats it per
    test because these tests are the ones that set them. `_said` is process-
    wide and each warning fires once, so a test that expected a line and ran
    second would not see one.
    """
    for name in (pgstore.DSN_ENV, pgstore.PROJECT_URL_ENV,
                 pgstore.PASSWORD_ENV, pgstore.REGION_ENV):
        monkeypatch.delenv(name, raising=False)
    pgstore._said.clear()
    yield
    pgstore._said.clear()


def test_disabled_without_the_variable(monkeypatch):
    monkeypatch.delenv(pgstore.DSN_ENV, raising=False)
    assert pgstore.enabled() is False
    assert pgstore.dsn() is None


def test_enabled_with_the_variable(monkeypatch):
    monkeypatch.setenv(pgstore.DSN_ENV, "postgresql://u:p@h:5432/db")
    assert pgstore.enabled() is True
    assert pgstore.dsn() == "postgresql://u:p@h:5432/db"


def test_the_other_postgres_spelling_is_a_dsn_too(monkeypatch):
    """`postgres://` is what several hosts print, psycopg takes both."""
    monkeypatch.setenv(pgstore.DSN_ENV, "postgres://u:p@h:5432/db")
    assert pgstore.dsn() == "postgres://u:p@h:5432/db"


def test_the_project_url_is_not_a_dsn(monkeypatch, caplog):
    """THE BUG THIS FILE EXISTS FOR. `https://<ref>.supabase.co` is the
    PostgREST endpoint, not a connection string. Set alone it is ignored --
    one line in the log, never an exception, and every store stays on the
    DuckDB file it would have used with nothing set at all."""
    monkeypatch.setenv(pgstore.DSN_ENV, "https://abcdefghijk.supabase.co")
    with caplog.at_level(logging.WARNING, logger="pipeline.pgstore"):
        assert pgstore.dsn() is None
        assert pgstore.enabled() is False
    lines = [r.getMessage() for r in caplog.records
             if r.levelno >= logging.WARNING]
    assert lines == ["SUPABASE_DB_URL is not a postgres:// URL; ignoring"], (
        "one warning, once")

    # And once per process, however many stores ask.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pipeline.pgstore"):
        assert pgstore.dsn() is None
    assert caplog.records == []


def test_composed_from_the_project_url_and_a_password(monkeypatch):
    """The two values the dashboard hands you, in the two variables named for
    them, become the session-pooler DSN -- port 5432, sslmode=require."""
    monkeypatch.setenv(pgstore.PROJECT_URL_ENV, "https://abcdefghijk.supabase.co")
    monkeypatch.setenv(pgstore.PASSWORD_ENV, "hunter2")
    assert pgstore.dsn() == (
        "postgresql://postgres.abcdefghijk:hunter2"
        "@aws-0-us-west-2.pooler.supabase.com:5432/postgres?sslmode=require")
    assert pgstore.enabled() is True


def test_the_region_can_be_overridden(monkeypatch):
    """`<ref>.supabase.co` resolves the same wherever the database sits, so
    the pooler's region cannot be read off the project URL."""
    monkeypatch.setenv(pgstore.PROJECT_URL_ENV, "https://abcdefghijk.supabase.co")
    monkeypatch.setenv(pgstore.PASSWORD_ENV, "hunter2")
    monkeypatch.setenv(pgstore.REGION_ENV, "eu-central-1")
    assert "@aws-0-eu-central-1.pooler.supabase.com:5432/" in pgstore.dsn()


def test_the_password_is_url_encoded(monkeypatch):
    """Supabase generates passwords with punctuation in them. An unescaped
    `@` does not fail loudly, it redraws where the host starts; a `#` cuts
    the password short at the fragment."""
    monkeypatch.setenv(pgstore.PROJECT_URL_ENV, "https://abcdefghijk.supabase.co")
    monkeypatch.setenv(pgstore.PASSWORD_ENV, "p@ss#word/x")
    value = pgstore.dsn()
    assert value == (
        "postgresql://postgres.abcdefghijk:p%40ss%23word%2Fx"
        "@aws-0-us-west-2.pooler.supabase.com:5432/postgres?sslmode=require")
    # One `@`, and it is the one that separates the credentials from the host.
    assert value.count("@") == 1


def test_the_misfiled_project_url_composes_too(monkeypatch, caplog):
    """The variable that held the wrong value is still the project URL, so
    adding the password is enough to make the deployment work -- no rename,
    which is the fix somebody makes at 2am."""
    monkeypatch.setenv(pgstore.DSN_ENV, "https://abcdefghijk.supabase.co")
    monkeypatch.setenv(pgstore.PASSWORD_ENV, "hunter2")
    with caplog.at_level(logging.WARNING, logger="pipeline.pgstore"):
        assert pgstore.dsn() == (
            "postgresql://postgres.abcdefghijk:hunter2"
            "@aws-0-us-west-2.pooler.supabase.com:5432/postgres"
            "?sslmode=require")
    # Still said, because the variable still holds the wrong kind of value.
    assert any("not a postgres:// URL" in r.getMessage()
               for r in caplog.records)


def test_nothing_is_composed_without_the_password(monkeypatch):
    """Half the pair is not a DSN. A project URL on its own leaves every
    store where it was."""
    monkeypatch.setenv(pgstore.PROJECT_URL_ENV, "https://abcdefghijk.supabase.co")
    assert pgstore.dsn() is None
    assert pgstore.enabled() is False


def test_a_real_dsn_beats_a_composable_pair(monkeypatch):
    """Somebody who wrote the connection string out gets the connection
    string they wrote, not one assembled around it."""
    monkeypatch.setenv(pgstore.DSN_ENV, "postgresql://u:p@h:5432/db")
    monkeypatch.setenv(pgstore.PROJECT_URL_ENV, "https://abcdefghijk.supabase.co")
    monkeypatch.setenv(pgstore.PASSWORD_ENV, "hunter2")
    assert pgstore.dsn() == "postgresql://u:p@h:5432/db"


def test_the_password_never_reaches_the_log(monkeypatch, caplog):
    monkeypatch.setenv(pgstore.PROJECT_URL_ENV, "https://abcdefghijk.supabase.co")
    monkeypatch.setenv(pgstore.PASSWORD_ENV, "hunter2")
    with caplog.at_level(logging.DEBUG, logger="pipeline.pgstore"):
        pgstore.dsn()
    assert caplog.records, "the composition says so once"
    assert not any("hunter2" in r.getMessage() for r in caplog.records)


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
