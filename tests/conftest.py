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
