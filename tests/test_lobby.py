"""api.lobby: the cached, cookie-free ESPN mock-lobby summary behind
GET /api/lobby.

Every test here stubs the upstream fetch (`api.lobby._lobby_summary`'s
`fetch` argument) rather than touching the network -- see that function's
docstring for why the module is written so a test can do that without
monkeypatching a private module attribute. `_reset_lobby_cache` is autouse
so the module-level cache dict never leaks a payload or a call count from
one test into the next.
"""
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import lobby


@pytest.fixture(autouse=True)
def _reset_lobby_cache():
    lobby.clear_cache()
    yield
    lobby.clear_cache()


def _row(league_size=8, rank_type="PPR", teams_joined=3, full=False,
         draft_date=None, extra_stat_ids=(53,)):
    return {
        "leagueSize": league_size,
        "rankType": rank_type,
        "teamsJoined": teams_joined,
        "full": full,
        "draftDate": draft_date,
        "draftType": "SNAKE",
        "scoringItemStatIds": list(extra_stat_ids),
    }


def _counting_fetch(rows):
    """A fetch stub that returns `rows` every time and remembers how many
    times it was called, so a test can assert the cache actually short-
    circuited the second call rather than merely returning the same data
    twice by coincidence."""
    calls = {"n": 0}

    def fetch(season):
        calls["n"] += 1
        return rows

    return fetch, calls


def test_summarize_counts_total_open_and_joinable_and_lists_soonest_first():
    now_ms = 1_000_000_000.0
    rows = [
        _row(teams_joined=1, draft_date=now_ms + 30_000),   # joinable, soon
        _row(teams_joined=6, draft_date=now_ms + 10_000),   # joinable, soonest
        _row(full=True, draft_date=now_ms + 5_000),         # full -> not joinable
        _row(draft_date=now_ms - 5_000),                    # already started
        _row(draft_date=None),                              # no draftDate at all
    ]

    result = lobby._summarize(rows, now_ms)

    assert result["available"] is True
    assert result["total_open"] == 5
    assert result["joinable"] == 2
    # Soonest-starting joinable room first.
    assert [r["starts_in_seconds"] for r in result["upcoming"]] == [10, 30]
    assert result["upcoming"][0]["league_size"] == 8
    assert result["upcoming"][0]["scoring"] == "PPR"
    assert result["upcoming"][0]["teams_joined"] == 6


def test_summarize_caps_the_upcoming_list_at_the_limit():
    now_ms = 1_000_000_000.0
    rows = [_row(draft_date=now_ms + i * 1_000) for i in range(1, 20)]

    result = lobby._summarize(rows, now_ms)

    assert result["joinable"] == 19
    assert len(result["upcoming"]) == lobby._UPCOMING_LIMIT


def test_lobby_summary_serves_a_cache_hit_without_calling_fetch_again():
    """The whole point of the cache: a thousand visitors polling within the
    TTL must cost ESPN one upstream call, not one per visitor."""
    fetch, calls = _counting_fetch([_row(draft_date=2_000_000_000_000.0)])

    first = lobby._lobby_summary(fetch=fetch, now_ms=1_000_000_000_000.0)
    second = lobby._lobby_summary(fetch=fetch, now_ms=1_000_000_000_000.0)

    assert calls["n"] == 1
    assert first == second


def test_lobby_summary_refetches_once_the_ttl_has_elapsed(monkeypatch):
    fetch, calls = _counting_fetch([_row(draft_date=2_000_000_000_000.0)])

    clock = {"t": 0.0}
    monkeypatch.setattr(lobby.time, "monotonic", lambda: clock["t"])

    lobby._lobby_summary(fetch=fetch, now_ms=1_000_000_000_000.0)
    assert calls["n"] == 1

    # Still within the TTL window -- must stay a cache hit.
    clock["t"] = lobby._CACHE_TTL_SECONDS - 1
    lobby._lobby_summary(fetch=fetch, now_ms=1_000_000_000_000.0)
    assert calls["n"] == 1

    # Past the TTL -- must fetch again.
    clock["t"] = lobby._CACHE_TTL_SECONDS + 1
    lobby._lobby_summary(fetch=fetch, now_ms=1_000_000_000_000.0)
    assert calls["n"] == 2


def test_lobby_summary_upstream_failure_yields_the_unavailable_payload():
    def fetch(season):
        raise RuntimeError("ESPN is down")

    result = lobby._lobby_summary(fetch=fetch)

    assert result == lobby._UNAVAILABLE
    assert result["available"] is False


def test_lobby_summary_bad_shape_from_upstream_is_also_unavailable():
    """A `fetch` that hands back something that isn't a list of row dicts
    (here, a bare dict -- iterating it yields its keys, which have no
    `.get`) must land on the same graceful `_UNAVAILABLE` outcome as a
    network failure, not an exception escaping the route."""
    def fetch(season):
        return {"not": "a list"}

    result = lobby._lobby_summary(fetch=fetch)

    assert result == lobby._UNAVAILABLE


def test_fetch_lobby_rows_rejects_a_non_list_body(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"unexpected": "object"}

    monkeypatch.setattr(
        "httpx.get", lambda *a, **k: _FakeResponse())

    with pytest.raises(ValueError):
        lobby._fetch_lobby_rows(season=2026)


def test_lobby_route_returns_the_cached_summary():
    fetch, calls = _counting_fetch([_row(draft_date=2_000_000_000_000.0,
                                          teams_joined=4)])
    lobby._lobby_summary(fetch=fetch, now_ms=1_000_000_000_000.0)
    assert calls["n"] == 1

    app = FastAPI()
    lobby.register_lobby_routes(app)
    client = TestClient(app)

    response = client.get("/api/lobby")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["joinable"] == 1
    # Route reads through `_lobby_summary` with no `fetch` override, so it
    # must have served the cache primed above rather than reaching ESPN.
    assert calls["n"] == 1
