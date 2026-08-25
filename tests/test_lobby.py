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


# -- the joinable rooms themselves (GET /api/lobby/rooms) --------------------
#
# The summary above answers "is anything happening" for the marketing widget.
# These answer "which room do I join", which needs the one field the summary
# has no use for: the league id.


def _room(league_id, **kwargs):
    row = _row(**kwargs)
    row["leagueId"] = league_id
    return row


def test_rooms_lists_the_joinable_ones_soonest_first_with_their_ids():
    now_ms = 1_000_000_000.0
    rows = [
        _room(11, teams_joined=1, draft_date=now_ms + 30_000),
        _room(22, teams_joined=6, draft_date=now_ms + 10_000),
        _room(33, full=True, draft_date=now_ms + 5_000),      # full
        _room(44, draft_date=now_ms - 5_000),                 # already started
        _room(55, draft_date=None),                           # no date at all
    ]
    result = lobby._rooms(rows, now_ms)

    assert [r["league_id"] for r in result] == ["22", "11"]
    assert [r["starts_in_seconds"] for r in result] == [10, 30]
    assert result[0]["teams_joined"] == 6 and result[0]["league_size"] == 8
    assert result[0]["scoring"] == "PPR" and result[0]["draft_type"] == "SNAKE"


def test_an_auction_is_never_offered():
    """ESPN runs auctions beside snakes. An auction has no pick order, no seat
    that owns pick 23, and nothing in the draft room is written for bidding --
    so a room this tool cannot help anybody draft in is not offered at all."""
    now_ms = 1_000_000_000.0
    snake = _room(11, draft_date=now_ms + 30_000)
    auction = _room(22, draft_date=now_ms + 10_000)
    auction["draftType"] = "AUCTION"

    assert [r["league_id"] for r in lobby._rooms([snake, auction], now_ms)] == ["11"]
    # And the landing widget's count says the same thing, so "12 rooms open"
    # never includes rooms nobody here can use.
    assert lobby._summarize([snake, auction], now_ms)["joinable"] == 1


def test_a_room_already_drafting_is_not_offered_even_with_a_future_date():
    """`draftInProgress` and a `draftDate` that has not passed disagree only
    on a row ESPN has not caught up with. A seat in a room that is already
    picking is not a seat anybody can use."""
    now_ms = 1_000_000_000.0
    rows = [_room(11, draft_date=now_ms + 30_000)]
    rows[0]["draftInProgress"] = True

    assert lobby._rooms(rows, now_ms) == []


def test_the_room_list_and_the_summary_share_one_upstream_call():
    """Both are the same directory read. A page showing the widget and the
    room list must not cost ESPN two requests."""
    fetch, calls = _counting_fetch([_room(11, draft_date=2_000_000_000_000.0)])

    lobby._lobby_summary(fetch=fetch, now_ms=1_000_000_000_000.0)
    body = lobby._lobby_rooms(fetch=fetch, now_ms=1_000_000_000_000.0)

    assert calls["n"] == 1
    assert [r["league_id"] for r in body["rooms"]] == ["11"]
    assert body["available"] is True


def test_the_room_list_says_how_many_it_left_out():
    """A busy lobby holds more rooms than one answer carries. A page filtering
    over 120 of 137 has to be able to say which it is."""
    now_ms = 1_000_000_000.0
    rows = [_room(n, draft_date=now_ms + 10_000 + n) for n in range(140)]

    body = lobby._lobby_rooms(fetch=lambda season: rows, now_ms=now_ms)

    assert body["total"] == 140
    assert len(body["rooms"]) == lobby._ROOMS_LIMIT


def test_rooms_upstream_failure_is_an_empty_list_not_an_error():
    def fetch(season):
        raise RuntimeError("ESPN is down")

    body = lobby._lobby_rooms(fetch=fetch)
    assert body == {"available": False, "total": 0, "rooms": []}


def test_the_rooms_route_serves_the_cached_read():
    fetch, calls = _counting_fetch([_room(11, draft_date=2_000_000_000_000.0)])
    lobby._lobby_summary(fetch=fetch, now_ms=1_000_000_000_000.0)

    app = FastAPI()
    lobby.register_lobby_routes(app)
    response = TestClient(app).get("/api/lobby/rooms")

    assert response.status_code == 200
    assert [r["league_id"] for r in response.json()["rooms"]] == ["11"]
    assert calls["n"] == 1


# -- watching the seats ------------------------------------------------------
#
# The count in a room card is the one number on that page that moves while
# somebody reads it. These cover what the module records about that movement:
# only real reads teach it anything, a first sighting is not a fill, and a
# failed read is not an emptying.


def test_a_room_first_seen_is_not_reported_as_filling():
    lobby._track_seats([_room(11, teams_joined=6)], now=100.0)

    facts = lobby._seat_facts("11", now=100.0)

    # Six seats were taken before anybody here was looking, so there is no
    # change to report -- `None`, which the page draws differently from "just
    # changed".
    assert facts["seats_changed_seconds"] is None
    assert facts["seats_delta"] == 0
    assert facts["watched_seconds"] == 0


def test_a_seat_taken_between_reads_is_tracked_with_its_delta():
    lobby._track_seats([_room(11, teams_joined=4)], now=100.0)
    lobby._track_seats([_room(11, teams_joined=6)], now=130.0)

    facts = lobby._seat_facts("11", now=145.0)

    assert facts["seats_delta"] == 2
    assert facts["seats_changed_seconds"] == 15
    assert facts["watched_seconds"] == 45


def test_a_read_that_finds_no_movement_leaves_the_last_change_where_it_was():
    lobby._track_seats([_room(11, teams_joined=4)], now=100.0)
    lobby._track_seats([_room(11, teams_joined=5)], now=110.0)
    lobby._track_seats([_room(11, teams_joined=5)], now=170.0)

    facts = lobby._seat_facts("11", now=170.0)

    # The clock since the change keeps running; the change itself does not
    # get re-stamped by a read that saw the same number.
    assert facts["seats_delta"] == 1
    assert facts["seats_changed_seconds"] == 60


def test_only_a_fresh_read_records_a_change():
    """A cache hit must not age or invent history: a thousand visitors reading
    one cached answer saw one read of ESPN between them."""
    rows = {"at": [_room(11, teams_joined=4, draft_date=2_000_000_000_000.0)]}

    def fetch(season):
        return rows["at"]

    lobby._lobby_rooms(fetch=fetch, now_ms=1_000_000_000_000.0)
    # ESPN's answer changes, but the TTL has not elapsed, so nothing re-reads.
    rows["at"] = [_room(11, teams_joined=9, draft_date=2_000_000_000_000.0)]
    body = lobby._lobby_rooms(fetch=fetch, now_ms=1_000_000_000_000.0)

    room = body["rooms"][0]
    assert room["teams_joined"] == 4
    assert room["seats_delta"] == 0
    assert room["seats_changed_seconds"] is None


def test_an_unreadable_lobby_does_not_wipe_what_was_tracked():
    lobby._track_seats([_room(11, teams_joined=4)], now=100.0)
    lobby._track_seats([_room(11, teams_joined=7)], now=110.0)

    def fetch(season):
        raise RuntimeError("ESPN is down")

    assert lobby._refresh_rows(fetch=fetch) is None
    facts = lobby._seat_facts("11", now=110.0)

    # An ESPN blip is not the room emptying: the history it had is still the
    # last thing ESPN actually said.
    assert facts["seats_delta"] == 3
    assert facts["seats_changed_seconds"] == 0


def test_the_room_list_carries_each_room_s_seat_history():
    now_ms = 1_000_000_000.0
    lobby._track_seats([_room(11, teams_joined=3)], now=100.0)
    lobby._track_seats([_room(11, teams_joined=4)], now=140.0)

    rooms = lobby._rooms([_room(11, teams_joined=4, draft_date=now_ms + 30_000)],
                         now_ms)

    assert rooms[0]["seats_delta"] == 1
    assert rooms[0]["seats_changed_seconds"] is not None


def test_a_room_the_lobby_has_stopped_listing_is_forgotten():
    lobby._track_seats([_room(11, teams_joined=3)], now=100.0)
    lobby._track_seats([_room(22, teams_joined=3)],
                       now=100.0 + lobby._SEAT_FORGET_SECONDS + 1)

    # Room 11 has not been in the directory for a quarter of an hour: it
    # started, or it filled. The lobby turns over completely every few
    # minutes, so a tracker that never forgets is a leak.
    assert lobby._seat_facts("11")["watched_seconds"] == 0
    assert lobby._seats.keys() == {"22"}


def test_the_watcher_reads_nothing_while_nobody_is_looking(monkeypatch):
    calls = {"n": 0}

    def fetch(season):
        calls["n"] += 1
        return [_room(11, draft_date=2_000_000_000_000.0)]

    monkeypatch.setattr("api.lobby._fetch_lobby_rows", fetch)

    assert lobby._watch_once() is False
    assert calls["n"] == 0

    lobby._note_demand()
    assert lobby._watch_once() is True
    assert calls["n"] == 1


def test_the_watcher_stops_once_the_demand_window_has_passed(monkeypatch):
    monkeypatch.setattr("api.lobby._fetch_lobby_rows",
                        lambda season: [_room(11, draft_date=2e12)])
    lobby._note_demand()
    assert lobby._wanted() is True

    # Every tab is closed: nothing has asked for four minutes.
    import time as _time
    lobby._note_demand(now=_time.monotonic() - lobby._DEMAND_WINDOW_SECONDS - 1)

    assert lobby._wanted() is False
    assert lobby._watch_once() is False


def test_a_request_to_either_route_keeps_the_watcher_awake():
    lobby._lobby_summary(fetch=_counting_fetch([])[0],
                         now_ms=1_000_000_000_000.0)
    app = FastAPI()
    lobby.register_lobby_routes(app)
    client = TestClient(app)

    assert lobby._wanted() is False
    client.get("/api/lobby/rooms")
    assert lobby._wanted() is True
