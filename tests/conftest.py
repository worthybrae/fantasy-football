"""Suite-wide fixtures.

THE SEO WARM-UP STAYS OFF UNDER PYTEST. `api/seo.register_seo_routes`
starts a daemon thread that builds the ADP aggregation on registration,
so the first crawler after a deploy does not pay for it. Every
`create_app()` in this suite would start one too -- against the REAL
corpus at `data/draft_corpus.duckdb` and whatever fixture board the test
handed in -- and, because the real corpus holds ids no fixture roster
knows, each thread ends in `cached_build_board` on the request-side
connection. That build ran concurrently with tests that count board
builds (`tests/test_demo_live.py`'s stale-answer test failed only in a
full run, never alone). `tests/test_seo.py` covers the warm-up
deliberately with its own fixture; nothing else wants it.
"""
import pytest


@pytest.fixture(autouse=True, scope="session")
def _seo_warm_off():
    from api import seo
    before = seo.WARM_ON_REGISTER
    seo.WARM_ON_REGISTER = False
    yield
    seo.WARM_ON_REGISTER = before


@pytest.fixture(autouse=True)
def _no_shared_postgres(monkeypatch):
    """No test reaches a shared Postgres unless it switches one on itself.

    SUPABASE_DB_URL is the only thing that chooses between a DuckDB file and
    Postgres, in every store that has both -- custody, billing, live-session
    records. So a developer with that variable exported in their shell, which
    is exactly what they need to run the `*_pg.py` files or a local server
    against Supabase, would otherwise point this entire suite at whatever
    database a deployment uses. `tests/test_billing.py` would then grant and
    revoke entitlements in it.

    The `*_pg.py` files read SUPABASE_TEST_DB_URL instead and set this one
    themselves, in their own autouse fixture, which runs after this.
    """
    from pipeline import pgstore
    monkeypatch.delenv(pgstore.DSN_ENV, raising=False)


@pytest.fixture(autouse=True)
def _no_history_import(request):
    """History import stays off under pytest, except where a test wants to
    watch it. `api/live.py` now calls `league_history.spawn_import_if_stale`
    on real (non-mock) connects, so every connect `tests/test_live_api.py`
    drives does too -- most against a provisioned-but-empty per-league file,
    which is exactly the "stale" case that spawns a background thread. That
    thread's `import_history` then makes a real ESPN call with no real
    cookies and fails, printing a traceback that has nothing to do with
    whatever the test itself is checking. Harmless, but noisy across a full
    run; `tests/test_live_api.py`'s two connect-token history tests override
    this via monkeypatch to watch the call instead of skip it.

    Function-scoped, not session-scoped like `_seo_warm_off` above, and
    excluding `tests/test_league_history.py` by name: that file calls
    `spawn_import_if_stale` directly to test ITS OWN real behaviour, and a
    session-wide no-op would silently break every assertion there instead of
    protecting it from anything.
    """
    if "test_league_history" in request.module.__name__:
        yield
        return
    from pipeline import league_history
    before = league_history.spawn_import_if_stale
    league_history.spawn_import_if_stale = lambda *a, **k: False
    yield
    league_history.spawn_import_if_stale = before
