"""The two routes that let a session join a draft, and the rule about whose."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.drafts as drafts_api
from pipeline import espn_drafts as drafts


@pytest.fixture(autouse=True)
def _clear_cache():
    """The draft list is memoised per identity for two minutes. Tests that
    change what ESPN says must not read the previous test's answer."""
    drafts_api._CACHE.clear()
    yield
    drafts_api._CACHE.clear()


def _fetcher(routes):
    def fetch(url, cookies, headers=None):
        for key, answer in routes.items():
            if key in url:
                return answer
        return 404, ""
    return fetch


def _entry(league_id, *, name=None, team_id=4, complete=False,
           date_ms=1_800_000_000_000):
    return {"typeId": 9, "metaData": {"entry": {
        "gameId": 1, "seasonId": 2026, "entryId": team_id,
        "entryMetadata": {"teamName": "My team", "draftComplete": complete},
        "groups": [{"groupId": league_id, "groupSize": 12,
                    "groupName": name or f"League {league_id}",
                    "draftDate": date_ms, "draftStatus": 1,
                    "draftTypeName": "Snake"}]}}}


def _profile(*entries):
    return json.dumps({"preferences": list(entries)})


def _client(fetch, store=None, client=("127.0.0.1", 50000)):
    app = FastAPI()
    drafts_api.register_draft_routes(app, store=store, fetch=fetch)
    # Loopback by default: the local-login tests below are about the owner's
    # own machine, and `session_for` only reaches for the saved login from
    # there (see `test_the_saved_login_is_invisible_off_the_machine`).
    return TestClient(app, client=client)


def _local(monkeypatch, swid="{ABC}", espn_s2="s2value"):
    """Stand in for this machine's own saved ESPN login."""
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: (swid, espn_s2))


def test_a_visitor_with_no_session_gets_an_answer_rather_than_a_401(monkeypatch):
    """This is the probe the connect screen runs on load, and the ordinary
    case is somebody who has connected nothing. A 401 here would be an error
    in every console and a retry in every client."""
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    body = _client(_fetcher({})).get("/api/espn/drafts").json()
    assert body == {"connected": False, "leagues": []}


def test_the_local_login_serves_the_owners_upcoming_drafts(monkeypatch):
    _local(monkeypatch)
    fetch = _fetcher({"fan.api": (200, _profile(
        _entry("111", name="Home league", team_id=6),
        _entry("222", name="Done", complete=True)))})
    body = _client(fetch).get("/api/espn/drafts?season=2026").json()
    assert body["connected"] is True and body["source"] == "local"
    assert [lg["name"] for lg in body["leagues"]] == ["Home league"]
    # ISO with a Z, so `new Date(...)` in the browser needs no date library.
    assert body["leagues"][0]["draft_at"].endswith("Z")
    # And the team, without which the row could not be a join button.
    assert body["leagues"][0]["team_id"] == "6"


def test_the_saved_login_is_invisible_off_the_machine(monkeypatch):
    """THE ONE THAT MATTERS. On the deployment `data/espn_state.json` is the
    FARM's login (`FARM_ESPN_STATE_B64`), and for a while any cookieless
    request was answered with it: every visitor was `connected`, saw the
    owner's leagues, and the landing page drew the dashboard for them
    instead of the introduction. A request that arrived through a proxy
    (forwarding headers) or from any address but loopback is a stranger."""
    _local(monkeypatch)
    fetch = _fetcher({"/apis/v2/profile": (200, _profile(_entry("1")))})
    proxied = _client(fetch).get("/api/espn/drafts",
                                 headers={"x-forwarded-for": "203.0.113.7"})
    assert proxied.status_code == 200
    assert proxied.json() == {"connected": False, "leagues": []}
    remote = _client(fetch, client=("203.0.113.7", 40000)).get("/api/espn/drafts")
    assert remote.status_code == 200
    assert remote.json() == {"connected": False, "leagues": []}


def test_the_league_list_is_reused_rather_than_refetched(monkeypatch):
    """One profile call plus one per league is a dozen round trips to ESPN for
    a page that gets refreshed. Twice in a row must be one round."""
    _local(monkeypatch)
    calls = []
    inner = _fetcher({"fan.api": (200, _profile(_entry("111", name="Home")))})

    def counting(url, cookies, headers=None):
        calls.append(url)
        return inner(url, cookies, headers)

    client = _client(counting)
    client.get("/api/espn/drafts?season=2026")
    first = len(calls)
    client.get("/api/espn/drafts?season=2026")
    assert len(calls) == first


def test_an_expired_local_session_reads_as_signed_out(monkeypatch):
    """ESPN rejecting the session is not an error to show, it is a state: the
    answer is "not connected", which puts the user back on the bookmarklet in
    one round trip instead of retrying a dead login."""
    _local(monkeypatch)
    body = _client(_fetcher({"fan.api": (401, "")})).get("/api/espn/drafts").json()
    assert body["connected"] is False and body["expired"] is True


def test_the_token_is_minted_for_the_local_login(monkeypatch):
    _local(monkeypatch)
    fetch = _fetcher({"draftSecurity": (200, "8675309")})
    body = _client(fetch).post("/api/espn/draft-token", json={
        "leagueId": "111", "teamId": "4", "season": "2026"}).json()
    assert body["token"] == "8675309"
    # The swid rides back because the connect it feeds needs it, and it is the
    # SESSION's, never one the caller supplied.
    assert body["swid"] == "{ABC}"


def test_minting_needs_a_session_at_all(monkeypatch):
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    response = _client(_fetcher({})).post("/api/espn/draft-token", json={
        "leagueId": "111", "teamId": "4"})
    assert response.status_code == 401


def test_a_closed_draft_is_the_upstream_refusing_not_a_bad_request(monkeypatch):
    """ESPN says no for reasons that are not the session -- the draft is not
    open, the team is not ours. Blaming the caller's request would send a
    reader looking for a bug in their own league id."""
    _local(monkeypatch)
    fetch = _fetcher({"draftSecurity": (404, "")})
    response = _client(fetch).post("/api/espn/draft-token", json={
        "leagueId": "111", "teamId": "4"})
    assert response.status_code == 502


class _NoSession:
    """A store that resolves nothing -- an expired or forged custody cookie."""

    def resolve(self, cookie, now=None):
        return None


def test_a_cookie_that_does_not_resolve_never_falls_back_to_the_owner(monkeypatch):
    """THE BUG THIS EXISTS TO PREVENT. A visitor whose custody cookie has
    expired must read as signed out -- if the fallback ran on a FAILED resolve
    rather than only on no cookie at all, that visitor would be served the
    machine owner's leagues and could join the owner's drafts."""
    _local(monkeypatch)
    # The plaintext allowance is what the guard below is normally refusing on;
    # granted, the request reaches the resolve and proves the real point.
    monkeypatch.setenv("ESPN_CUSTODY_ALLOW_PLAINTEXT_HTTP", "1")
    client = _client(_fetcher({"fan.api": (200, _profile(_entry("111")))}),
                     store=_NoSession())
    client.cookies.set(drafts_api.cred.COOKIE_NAME, "not-a-real-session")
    assert client.get("/api/espn/drafts").json() == {"connected": False,
                                                     "leagues": []}


def test_a_cookie_on_a_plaintext_wire_is_refused_before_any_of_this(monkeypatch):
    """A cookie on the wire is a credential on the wire. The refusal is the
    custody guard's, reached through these routes because they resolve through
    it -- so the rule cannot be forgotten by a route that reads a session."""
    _local(monkeypatch)
    monkeypatch.delenv("ESPN_CUSTODY_ALLOW_PLAINTEXT_HTTP", raising=False)
    client = _client(_fetcher({}), store=_NoSession())
    client.cookies.set(drafts_api.cred.COOKIE_NAME, "anything")
    response = client.get("/api/espn/drafts")
    assert response.status_code >= 400
    assert "plain HTTP" in response.json()["detail"]


# -- taking a seat in an ESPN mock room --------------------------------------
#
# The lobby listing is public and cookie-free (api/lobby.py). This half is not:
# a seat is taken as somebody, and the token that follows drives their draft.


def _poster(answer, seen=None):
    """Stand in for ESPN's invite POST. `answer` is the body it returns."""
    def post(url, payload):
        if seen is not None:
            seen.append((url, payload))
        return answer
    return post


def test_joining_a_mock_takes_the_seat_then_mints_for_it(monkeypatch):
    """Two ESPN calls, in order, and the team id comes from the FIRST one --
    ESPN assigns the seat, we never pick it."""
    _local(monkeypatch)
    seen = []
    # A signed integer, like the real one (see espn_drafts.mint_draft_token).
    fetch = _fetcher({"draftSecurity": (200, "-4429")})
    app = FastAPI()
    drafts_api.register_draft_routes(app, fetch=fetch,
                                     post=_poster('[{"teamId": 5}]', seen))
    body = TestClient(app, client=("127.0.0.1", 50000)).post("/api/espn/mock-join",
                                json={"leagueId": "999"}).json()

    assert body["teamId"] == "5"
    assert body["token"] == "-4429"
    assert body["leagueId"] == "999"
    # The invite POST really was ESPN's invite POST, with the sentinel that
    # means "any open seat".
    assert "invites" in seen[0][0] and seen[0][1] == [{"teamId": -1}]


def test_joining_a_mock_needs_a_session(monkeypatch):
    """The listing is public; the seat is not. A visitor with nothing
    connected gets a 401 rather than a room in somebody else's name."""
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    app = FastAPI()
    drafts_api.register_draft_routes(app, fetch=_fetcher({}),
                                     post=_poster('[{"teamId": 5}]'))
    assert TestClient(app, client=("127.0.0.1", 50000)).post("/api/espn/mock-join",
                                json={"leagueId": "999"}).status_code == 401


def test_a_room_that_filled_first_is_the_upstream_refusing(monkeypatch):
    """The lobby moves in seconds: a room listed a moment ago can be full by
    the time somebody clicks it. That is ESPN saying no, not a bad request,
    and the reader is told so the list can be re-read."""
    _local(monkeypatch)

    def post(url, payload):
        return '{"messages": ["League is full"]}'

    app = FastAPI()
    drafts_api.register_draft_routes(app, fetch=_fetcher({}), post=post)
    response = TestClient(app, client=("127.0.0.1", 50000)).post("/api/espn/mock-join",
                                    json={"leagueId": "999"})

    assert response.status_code == 502
    assert "999" in response.json()["detail"]


# -- the waiting room: which seats are open, and taking a chosen one --------
#
# ESPN honours a specific `teamId` in the invite, not only the `-1` ("any open
# seat") the farm has always sent -- probed live 2026-08-24 against an open
# room: asked for seat 4 while seat 2 was also open, and ESPN assigned 4.


def _room_body(open_seats=(4,), teams=4, in_progress=False,
               date_ms=1_787_608_890_000, mine=None):
    return json.dumps({
        "draftDetail": {"inProgress": in_progress, "drafted": False,
                        "picks": [{"id": n} for n in range(teams * 16)]},
        "settings": {"draftSettings": {
            "date": date_ms, "type": "SNAKE", "timePerSelection": 30,
            "pickOrder": list(range(1, teams + 1)),
        }},
        # ESPN pre-populates the whole pick list before a room drafts, which
        # is where the round count comes from.
        "teams": [{
            "id": n,
            "name": f"Team {n}",
            "owners": ([] if n in open_seats
                       else ([mine] if mine and n == 1 else [f"{{OWNER-{n}}}"])),
        } for n in range(1, teams + 1)],
    })


def test_the_room_reports_every_seat_and_which_one_is_yours(monkeypatch):
    _local(monkeypatch, swid="{ABC}")
    fetch = _fetcher({"leagues/999": (200, _room_body(open_seats=(3, 4), mine="{ABC}"))})
    body = _client(fetch).get("/api/espn/mock-room/999").json()

    assert [s["team_id"] for s in body["seats"]] == ["1", "2", "3", "4"]
    assert [s["taken"] for s in body["seats"]] == [True, True, False, False]
    # The seat this session owns, marked on the seat and named once at the top
    # so the page does not have to hunt for it.
    assert body["my_team_id"] == "1"
    assert body["seats"][0]["mine"] is True
    # What the countdown counts to, and the clock the room will run on.
    assert body["draft_at"].endswith("Z")
    assert body["clock_seconds"] == 30
    assert body["in_progress"] is False
    # The draft order, so each seat can say which picks it gets.
    assert [s["slot"] for s in body["seats"]] == [1, 2, 3, 4]
    # Rounds come from the pre-populated pick list, not from a roster count
    # that includes an IR slot nobody drafts into.
    assert body["rounds"] == 16


def test_a_room_read_needs_no_session(monkeypatch):
    """The directory is public and so is a room. A visitor with no session
    still gets the seats -- they simply own none of them."""
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    fetch = _fetcher({"leagues/999": (200, _room_body())})
    body = _client(fetch).get("/api/espn/mock-room/999").json()

    assert body["my_team_id"] is None
    assert all(s["mine"] is False for s in body["seats"])


def test_joining_a_chosen_seat_asks_espn_for_that_seat(monkeypatch):
    _local(monkeypatch)
    seen = []
    fetch = _fetcher({"draftSecurity": (200, "-4429")})
    app = FastAPI()
    drafts_api.register_draft_routes(app, fetch=fetch,
                                     post=_poster('[{"teamId": 7}]', seen))
    body = TestClient(app, client=("127.0.0.1", 50000)).post("/api/espn/mock-join",
                                json={"leagueId": "999", "teamId": "7"}).json()

    assert seen[0][1] == [{"teamId": 7}]
    assert body["teamId"] == "7"


def test_no_chosen_seat_still_means_any_open_seat(monkeypatch):
    """The sentinel is what the farm sends and what a reader who does not care
    which seat they get should still send."""
    _local(monkeypatch)
    seen = []
    fetch = _fetcher({"draftSecurity": (200, "-4429")})
    app = FastAPI()
    drafts_api.register_draft_routes(app, fetch=fetch,
                                     post=_poster('[{"teamId": 2}]', seen))
    TestClient(app, client=("127.0.0.1", 50000)).post("/api/espn/mock-join", json={"leagueId": "999"})

    assert seen[0][1] == [{"teamId": -1}]


# -- rooms in progress: how far along each one is ---------------------------
#
# The Home page's mock cards say "round 3 · pick 21" for a room you are
# drafting in. That comes from the same public room read the waiting room
# uses, counted rather than listed, and asked for in one request for every
# room on the page.


def _drafting_body(teams=8, rounds=16, made=21):
    picks = []
    for n in range(teams * rounds):
        # ESPN pre-populates every slot; the ones made carry a player id and
        # the rest the -1 placeholder (pipeline/espn_league._is_real_pick).
        picks.append({"id": n, "playerId": 1000 + n if n < made else -1})
    return json.dumps({
        "draftDetail": {"inProgress": True, "drafted": False, "picks": picks},
        "settings": {"draftSettings": {
            "date": 1_787_608_890_000, "type": "SNAKE", "timePerSelection": 30,
            "pickOrder": list(range(1, teams + 1)),
        }},
        "teams": [{"id": n, "name": f"Team {n}", "owners": [f"{{OWNER-{n}}}"]}
                  for n in range(1, teams + 1)],
    })


def test_progress_reads_each_room_s_shape_and_state(monkeypatch):
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    drafts_api._PROGRESS.clear()
    fetch = _fetcher({
        "leagues/111": (200, _drafting_body(teams=8, rounds=16, made=21)),
        "leagues/222": (200, _drafting_body(teams=10, rounds=15, made=0)),
    })
    body = _client(fetch).get("/api/espn/rooms/progress?ids=111,222").json()

    # ESPN's read knows the room's shape and state, but names no player
    # until the draft is over (see the farm test below), so the count and
    # the round are unknown from here rather than "0, round 1".
    assert body["rooms"]["111"] == {
        "picks_made": None, "picks_total": 128, "teams": 8, "rounds": 16,
        "round": None, "pick_in_round": None, "in_progress": True, "drafted": False,
    }
    assert body["rooms"]["222"]["picks_total"] == 150
    assert body["rooms"]["222"]["teams"] == 10


def test_progress_skips_a_room_espn_will_not_read(monkeypatch):
    """One dead room must not cost the page the other ten."""
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    drafts_api._PROGRESS.clear()
    fetch = _fetcher({"leagues/111": (200, _drafting_body(made=5))})
    body = _client(fetch).get("/api/espn/rooms/progress?ids=111,999").json()
    assert set(body["rooms"]) == {"111"}


def test_progress_asks_nothing_for_no_ids(monkeypatch):
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    calls = []

    def fetch(url, cookies, headers=None):
        calls.append(url)
        return 200, _drafting_body()
    assert _client(fetch).get("/api/espn/rooms/progress?ids=").json() == {"rooms": {}}
    assert _client(fetch).get("/api/espn/rooms/progress").json() == {"rooms": {}}
    assert calls == []


def test_progress_is_shared_between_readers_for_a_few_seconds(monkeypatch):
    """Ten dashboards open on the same eleven rooms are eleven reads of ESPN
    per window, not a hundred and ten."""
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    drafts_api._PROGRESS.clear()
    calls = []

    def fetch(url, cookies, headers=None):
        calls.append(url)
        return 200, _drafting_body(made=5)
    client = _client(fetch)
    client.get("/api/espn/rooms/progress?ids=111")
    client.get("/api/espn/rooms/progress?ids=111")
    assert len(calls) == 1


def test_progress_prefers_the_farm_s_own_count_of_a_room_it_sits_in(
        monkeypatch, tmp_path):
    """ESPN's public room read lists every slot but names no player until the
    draft is over -- probed live: six rooms mid-draft, 0 picks each by that
    read while the farm's files held 12 to 103. The farm hears the picks on
    the socket and writes them as they land, so a room it sits in is
    counted from its file, and the read is only for rooms it does not."""
    import api.demo as demo
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    monkeypatch.setattr(demo, "FARM_DIR", str(tmp_path))
    drafts_api._PROGRESS.clear()
    (tmp_path / "111.json").write_text(json.dumps({
        "league_id": "111", "teams": 8, "rounds": 16,
        "picks": [{"pick_no": n + 1} for n in range(37)],
    }))
    fetch = _fetcher({"leagues/111": (200, _drafting_body(made=0)),
                      "leagues/222": (200, _drafting_body(made=0))})
    body = _client(fetch).get("/api/espn/rooms/progress?ids=111,222").json()
    assert body["rooms"]["111"]["picks_made"] == 37
    assert body["rooms"]["111"]["round"] == 5
    assert body["rooms"]["111"]["pick_in_round"] == 6
    assert body["rooms"]["111"]["in_progress"] is True
    # Not one the farm is in: ESPN's read, which cannot count mid-draft, so
    # the round is honestly unknown rather than "round 1, pick 1".
    assert body["rooms"]["222"]["picks_made"] is None
    assert body["rooms"]["222"]["round"] is None
    assert body["rooms"]["222"]["in_progress"] is True


def test_progress_ignores_a_farm_file_that_has_gone_quiet(monkeypatch, tmp_path):
    import os
    import time as _time
    import api.demo as demo
    monkeypatch.setattr(drafts, "saved_session", lambda *a, **k: None)
    monkeypatch.setattr(demo, "FARM_DIR", str(tmp_path))
    drafts_api._PROGRESS.clear()
    path = tmp_path / "111.json"
    path.write_text(json.dumps({"league_id": "111", "teams": 8, "rounds": 16,
                                "picks": [{"pick_no": 1}]}))
    ago = _time.time() - demo.STALE_SECONDS - 1
    os.utime(path, (ago, ago))
    fetch = _fetcher({"leagues/111": (200, _drafting_body(made=0))})
    body = _client(fetch).get("/api/espn/rooms/progress?ids=111").json()
    assert body["rooms"]["111"]["picks_made"] is None
