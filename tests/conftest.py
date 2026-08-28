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

AND NEITHER OF THE OTHER TWO WARM-UPS, for the same reason and with the
same shape -- see `_boot_warm_off`.
"""
import os

import pytest

# AT IMPORT, before any fixture: a `.env` the developer keeps beside the
# repository, or a DSN exported in their shell, must not reach a single
# test's collection -- `api/main.py` guards its own loader against pytest,
# but the shell's variable needs no loader. `_no_shared_postgres` below
# repeats this per test, for a test that sets it and forgets.
os.environ.pop("SUPABASE_DB_URL", None)

# AND THE WARM-UPS, AT IMPORT, which the two session fixtures below cannot
# reach. `api/main.py` ends with `app = create_app()` at module scope, so
# importing it -- which tests/test_api.py does at COLLECTION time, before any
# fixture has run -- builds a whole app and starts its background threads
# against the real `data/nfl.duckdb` and the real corpus. Those fixtures then
# turn the flags off for the apps the tests themselves build, and that one is
# already running.
#
# It was survivable while each of those threads did one build and exited. The
# ADP warm-up now renews itself on a timer (api/seo.py, so /adp is never the
# request that finds the aggregate cold), which in a five-minute run means a
# second board build landing in the middle of somebody's test. These three
# variables are what the modules read as they are imported, so they have to
# be set before the import, not after it.
for _switch in ("SEO_WARM", "DEMO_WARM", "WARM_ON_BOOT"):
    os.environ[_switch] = "0"


@pytest.fixture(autouse=True, scope="session")
def _seo_warm_off():
    from api import seo
    before = seo.WARM_ON_REGISTER
    seo.WARM_ON_REGISTER = False
    yield
    seo.WARM_ON_REGISTER = before


@pytest.fixture(autouse=True, scope="session")
def _boot_warm_off():
    """The other two background warm-ups stay off under pytest as well.

    Both do the same thing the SEO one does -- start a daemon thread that
    builds a real board while the test that created the app goes on to count
    board builds -- and both are on by default in a deployment, which is
    where they belong:

      * `api/jobs.start_jobs` warms the board, profile and game-point caches
        when the database it is handed has been refreshed. Switched with an
        environment variable rather than a module flag because that is what
        every other job in that file is switched with, and because a
        deployment may want to turn it off without a redeploy.
      * `api/demo.register_demo_routes` rebuilds the landing demo. It already
        reads DEMO_WARM, but at IMPORT time, so the flag is what a fixture
        can move; that is also how tests/test_demo_live.py turns it back ON
        for the two tests that watch it, and this default is what lets them.

    Session-scoped and autouse, exactly like `_seo_warm_off`: the point is
    that no test has to know these threads exist.
    """
    from api import demo, jobs
    before_env = os.environ.get(jobs.WARM_ON_BOOT_ENV)
    before_demo = demo.WARM_ON_REGISTER
    os.environ[jobs.WARM_ON_BOOT_ENV] = "0"
    demo.WARM_ON_REGISTER = False
    yield
    demo.WARM_ON_REGISTER = before_demo
    if before_env is None:
        os.environ.pop(jobs.WARM_ON_BOOT_ENV, None)
    else:
        os.environ[jobs.WARM_ON_BOOT_ENV] = before_env


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
