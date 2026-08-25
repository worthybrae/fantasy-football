"""GET /api/lobby: a live read of ESPN's public mock-draft lobby, cached and
reduced to what the landing page's proof-of-life widget needs.

WHY THIS EXISTS. The landing page currently claims "Try it in a mock draft"
as flat marketing copy. ESPN's own mock-draft lobby directory is public --
no cookies required, confirmed by calling it with none at all (see
`docs/superpowers/reference/espn-mock-lobby-endpoints.md`) -- so the page
can show real numbers instead of a claim: how many rooms are open right
now, and when the next one starts.

`pipeline.espn_mock_lobby` already builds this same directory URL for the
farm's room-picking loop, and this module reuses `lobby_url` from it. It
does NOT reuse that module's `list_mock_leagues` / `http_fetch`, though,
because that call chain goes through `pipeline.draft_socket.http_fetch`,
which ALWAYS attaches the saved `espn_s2`/`SWID` cookies -- its whole job is
authenticating the farm's join requests. Reusing it here would turn a page
any anonymous visitor can load into something that sends the owner's own
ESPN login upstream on every visitor's behalf, silently. This module makes
its own plain, cookie-free request instead (`_fetch_lobby_rows` below) and
never imports `draft_socket` at all.

CACHING: A THOUSAND VISITORS MUST COST ESPN ONE REQUEST A MINUTE. The
landing page is public, and every visitor's browser polls it independently
(see `web/src/components/LobbyStrip.tsx`). With no cache in front of the
upstream call, a thousand simultaneous visitors would mean a thousand
requests to ESPN per poll interval. `_CACHE_TTL_SECONDS` bounds that to one
upstream call per TTL window no matter how many people are looking at the
page at once. A plain module-level dict guarded by a lock is enough --
there is exactly one lobby summary, shared by every caller, with no
per-visitor variation to key on.

WATCHING THE SEATS. A room's seat count is the one field on this page that
moves while somebody looks at it: rooms fill in the last minutes before they
start, and a count read ninety seconds ago is a different room from the one
on screen. So the directory is re-read on a timer (`_watch_lobby` below) for
as long as anybody is actually looking, and each read is DIFFED against the
last one -- `_track_seats` remembers, per room, what the count was, when it
last moved and by how much. That is what lets the page say "two seats went in
the last minute" instead of redrawing the same number and hoping the reader
notices it changed.

The watcher costs ESPN one request per `_WATCH_SECONDS` while the page is
open and nothing at all when it is not: `_note_demand` stamps every request
to this module's routes, and the loop sleeps through any window where nobody
has asked in `_DEMAND_WINDOW_SECONDS`. It is the same cached directory read
every visitor already shares, on a clock instead of on a request.

FAILURE IS SILENT, ON PURPOSE. ESPN being unreachable, slow, or returning
something other than the expected JSON array must not turn into a 500 for
every visitor of the landing page. `_lobby_summary` catches everything from
the fetch and the parse in one `except Exception` and hands back
`_UNAVAILABLE` -- a well-formed payload with `available: False` -- instead
of letting the exception propagate. The widget reads that flag and renders
nothing (see LobbyStrip.tsx's own docstring): a broken box on marketing
copy is worse than no box, so the failure has to reach the frontend as
"nothing to show", never as an error.
"""
from __future__ import annotations

import json
import os
import threading
import time

from pipeline.espn_mock_lobby import lobby_url, room_url
from scoring.config import CURRENT_SEASON

# One upstream call per this many seconds, no matter how many visitors poll
# /api/lobby in between -- see the module docstring's CACHING section.
#
# 20s, down from the 60s that matched the marketing widget's slowest poll:
# the room list is a wall of seat counts now, and a count up to a minute old
# is one a reader can watch be wrong (a room shows 4/10, they open it, ESPN
# says 7/10). This is the FLOOR, not the cadence -- while anybody is looking,
# `_watch_lobby` refreshes on its own clock well inside this window, so what
# the TTL actually bounds is how stale an unwatched lobby can get before the
# next visitor's first request pays for a read.
_CACHE_TTL_SECONDS = 20.0

# How often the watcher re-reads the directory while somebody is looking.
# 12s is under the page's own poll (MockLobby.tsx polls at 12s too), so a
# visitor's request nearly always lands on a directory that was refreshed
# behind them rather than paying for the fetch itself.
_WATCH_SECONDS = 12.0

# How long after the last request to this module the watcher keeps reading.
# Four minutes covers a reader who is scrolling, filtering, or reading one
# room's card without clicking anything -- their browser is polling anyway,
# which re-stamps demand -- and stops the loop within minutes of the last
# tab closing. An idle server makes no requests to ESPN at all.
_DEMAND_WINDOW_SECONDS = 240.0

# A room the directory has stopped listing (it started, or it filled) is
# dropped from the seat tracker this long after it was last seen. The lobby
# turns over completely every few minutes, so without this the tracker would
# grow by a few hundred dead ids an hour for the life of the process.
_SEAT_FORGET_SECONDS = 900.0

# The watcher is started by `register_lobby_routes`, which every test that
# builds the app also calls -- and a background thread making real requests
# to ESPN during a test run is not something a test should have to know to
# turn off. So it is off under pytest, and off anywhere `LOBBY_WATCH=0` says
# so. `api.demo`'s `WARM_ON_REGISTER` gates its own warm thread the same way.
WATCH_ON_REGISTER = (os.environ.get("LOBBY_WATCH", "1") != "0"
                     and "PYTEST_CURRENT_TEST" not in os.environ)

# Headers a plain, anonymous GET of the lobby directory answered with today
# (see the reference doc) -- no cookie header, ever. `x-fantasy-source` and
# `origin` are the two ESPN appears to actually check; the rest match a real
# browser so the request doesn't stand out as a bot on its face.
_LOBBY_HEADERS = {
    "accept": "application/json",
    "x-fantasy-source": "kona",
    "origin": "https://fantasy.espn.com",
    "referer": "https://fantasy.espn.com/",
    "user-agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
}

# How many upcoming rooms the widget lists beneath its headline number.
_UPCOMING_LIMIT = 8

# The payload /api/lobby serves whenever ESPN can't be read, or answers with
# something this module doesn't recognise. `available: False` is the one
# field the widget checks before rendering anything else -- see the module
# docstring's FAILURE IS SILENT section.
_UNAVAILABLE = {"available": False, "total_open": 0, "joinable": 0,
                "upcoming": []}

# The joinable-room list served to a signed-in reader. Longer than the
# widget's eight, and long enough to FILTER over rather than merely read: a
# busy hour's lobby holds well over a hundred joinable rooms, and a page that
# served the soonest 24 could not answer "12-team PPR only" without asking
# ESPN again for every combination somebody might pick. 120 rooms is about 15
# KB on the wire and covers every measured hour of the lobby; the payload says
# how many were joinable in total so a page that is looking at a truncated set
# can say so rather than implying it has them all.
_ROOMS_LIMIT = 120

# The answer when ESPN cannot be read. Same silence rule as `_UNAVAILABLE`
# above: a dashboard section that cannot list rooms draws nothing.
_NO_ROOMS = {"available": False, "total": 0, "rooms": []}

_cache_lock = threading.Lock()
# THE DIRECTORY ROWS, not a payload built from them. Two endpoints now read
# this one directory -- the widget's summary and the joinable-room list -- and
# caching each one's finished payload separately would have made a page that
# shows both cost ESPN two requests for the same JSON. `rows` is None for a
# read that failed, which is cached exactly like a successful one so an ESPN
# outage is not retried once per visitor (see the module docstring).
# `expires_at` is a `time.monotonic()` deadline so a system clock change
# mid-process can't extend or shrink the window.
_cache = {"rows": None, "cached": False, "expires_at": 0.0}

# WHAT EACH ROOM'S SEAT COUNT WAS LAST TIME, keyed by league id:
#
#     {"teams": 6, "seen": <monotonic>, "changed_at": <monotonic> | None,
#      "delta": +1, "since": <monotonic of first sighting>}
#
# `changed_at` is None until a room has been seen to MOVE. That is a
# different fact from "changed 0 seconds ago" and the page renders it
# differently: a room first seen at 6/10 has not filled six seats in front of
# anybody, it was simply already there.
_seats_lock = threading.Lock()
_seats: dict[str, dict] = {}

# When somebody last asked this module for anything, as a `time.monotonic()`
# stamp. The watcher reads it to decide whether the lobby is worth re-reading
# -- see `_watch_lobby`. Guarded by `_cache_lock`, which every path through
# this module already takes.
_demand = {"at": 0.0}
_watching = {"started": False}


def _fetch_lobby_rows(season: int = CURRENT_SEASON):
    """One anonymous GET of ESPN's mock-lobby directory. No cookies, ever --
    see the module docstring for why that's load-bearing, not incidental.

    httpx is already a project dependency (see `pipeline.draft_socket.
    http_fetch`, which hits the very same host); nothing new is added here
    to make one unauthenticated GET.
    """
    import httpx

    response = httpx.get(lobby_url(season), headers=_LOBBY_HEADERS,
                         timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(
            f"ESPN's mock lobby returned {type(payload).__name__}, not a "
            "list of rooms")
    return payload


def _starts_in(row: dict, now_ms: float):
    """Seconds until this room starts picking, or None if it has no date.

    Clamped at 0 rather than left negative -- a row can slip from "future" to
    "already started" between the read and the response, and a countdown
    should never show a negative number for that.
    """
    draft_date = row.get("draftDate")
    if draft_date is None:
        return None
    return max(0, int((float(draft_date) - now_ms) / 1000.0))


# The only draft this tool can sit in. ESPN's lobby runs auctions alongside
# snakes, and an auction is a different game: there is no pick order, no seat
# that owns pick 23, and nothing in the draft room -- the board, the clock
# panel, `gain_now`, the survival model -- is written for bidding. A room we
# cannot help somebody draft in has no business being offered to them, so it
# is filtered out at the source rather than hidden in the page: the count, the
# landing widget and the room list then all mean the same thing.
_DRAFT_TYPE = "SNAKE"


def _joinable(rows: list, now_ms: float) -> list:
    """The rooms a person could actually take a seat in, soonest first.

    Snake, not full, not already picking, and a draft still ahead of now. The
    `draftInProgress` clause is not redundant with the date one: ESPN leaves a
    started room in the directory with its original `draftDate`, and for the
    seconds either side of that boundary the two fields disagree. A seat in a
    room that is already drafting is not a seat anybody can use, so the flag
    wins whenever it is set.
    """
    keep = [row for row in rows
            if row.get("draftType") == _DRAFT_TYPE
            and not row.get("full")
            and not row.get("draftInProgress")
            and row.get("draftDate") is not None
            and float(row["draftDate"]) > now_ms]
    keep.sort(key=lambda row: float(row["draftDate"]))
    return keep


def _track_seats(rows: list, now: float | None = None) -> None:
    """Diff this read's seat counts against the previous one.

    Called on every SUCCESSFUL fresh read of the directory and nowhere else,
    so the record is a history of what ESPN actually said rather than of when
    somebody happened to look. A cache hit changes nothing here, which is
    what keeps "changed 4 seconds ago" true no matter how many visitors read
    that same cached answer in between.

    A room seen for the first time is recorded WITHOUT a change: it arrived
    at whatever count it arrived at, and calling that a fill would put a
    "+6" on every room in the list the moment the process starts.

    Rows with no `leagueId` are skipped rather than collapsed under one key.
    The summary endpoint's rows have no id at all (the widget never needed
    one), and tracking them all as the same room would have every one of
    them reporting the last one's count.
    """
    at = time.monotonic() if now is None else now
    with _seats_lock:
        for row in rows:
            league_id = row.get("leagueId")
            if league_id is None:
                continue
            key = str(league_id)
            teams = int(row.get("teamsJoined") or 0)
            seen = _seats.get(key)
            if seen is None:
                _seats[key] = {"teams": teams, "seen": at, "since": at,
                               "changed_at": None, "delta": 0}
                continue
            if teams != seen["teams"]:
                seen["delta"] = teams - seen["teams"]
                seen["teams"] = teams
                seen["changed_at"] = at
            seen["seen"] = at
        # The lobby turns over completely every few minutes; without this the
        # tracker would keep every id the process ever saw.
        for key in [k for k, v in _seats.items()
                    if at - v["seen"] > _SEAT_FORGET_SECONDS]:
            del _seats[key]


def _seat_facts(league_id: str, now: float | None = None) -> dict:
    """What we know about this room's seat count moving, for the payload.

    `seats_changed_seconds` is None for a room that has not been seen to move
    -- which is NOT the same as zero, and the page draws it differently: a
    room first read at 6/10 did not fill six seats while anybody watched.

    `watched_seconds` is how long this room has been under observation, and
    it is what makes silence readable: "no change" over eight seconds is
    nothing, over eight minutes it is a room nobody is joining.
    """
    at = time.monotonic() if now is None else now
    with _seats_lock:
        seen = _seats.get(league_id)
        if seen is None:
            return {"seats_delta": 0, "seats_changed_seconds": None,
                    "watched_seconds": 0}
        changed_at = seen["changed_at"]
        return {
            "seats_delta": int(seen["delta"]),
            "seats_changed_seconds": (None if changed_at is None
                                      else max(0, int(at - changed_at))),
            "watched_seconds": max(0, int(at - seen["since"])),
        }


def _rooms(rows: list, now_ms: float) -> list:
    """The joinable rooms, as the dashboard's list renders them.

    Everything `_room_summary` carries plus the one field it has no use for
    and this cannot do without: `leagueId`, which is what a join POSTs to (see
    `api/drafts.py`'s mock-join). A string on the wire, because every other id
    this API serves is one and a browser that concatenates them should not
    have to remember which kind it got.
    """
    return [{
        "league_id": str(row.get("leagueId")),
        "league_size": row.get("leagueSize"),
        "teams_joined": int(row.get("teamsJoined") or 0),
        "scoring": row.get("rankType"),
        "draft_type": row.get("draftType"),
        # PRO / EXPERT / BEGINNER, or absent. Shown rather than filtered on:
        # the farm ranks by it because it is building a corpus, and a person
        # choosing a room is entitled to a different preference.
        "experience": row.get("experienceType"),
        "starts_in_seconds": _starts_in(row, now_ms),
        # What this room's seat count has DONE since we started watching it
        # (`_track_seats`). The count alone cannot say whether a room at 4/10
        # is filling or has been sitting at four for ten minutes, and that is
        # the difference between a room worth joining and a dead one.
        **_seat_facts(str(row.get("leagueId"))),
    } for row in _joinable(rows, now_ms)[:_ROOMS_LIMIT]]


def _room_summary(row: dict, now_ms: float) -> dict:
    """One directory row, reduced to what the widget shows about it.

    `starts_in_seconds` is clamped at 0 rather than left negative -- a row
    can slip from "future" to "already started" between the summary being
    built and the response being read, and a widget counting down should
    never show a negative number for that.
    """
    return {
        "league_size": row.get("leagueSize"),
        "scoring": row.get("rankType"),
        "teams_joined": int(row.get("teamsJoined") or 0),
        "starts_in_seconds": _starts_in(row, now_ms),
    }


def _summarize(rows: list, now_ms: float) -> dict:
    """Reduce the raw directory to the summary /api/lobby serves.

    `total_open` is every row ESPN's directory lists -- the whole point of
    the headline number is "this many rooms exist right now", not a
    filtered subset. `joinable` is `_joinable`'s definition, shared with the
    room list this page also serves so the count and the list can never
    disagree about what a joinable room is.
    """
    joinable_rows = _joinable(rows, now_ms)
    return {
        "available": True,
        "total_open": len(rows),
        "joinable": len(joinable_rows),
        "upcoming": [_room_summary(row, now_ms)
                     for row in joinable_rows[:_UPCOMING_LIMIT]],
    }


def _lobby_summary(fetch=None, season: int = CURRENT_SEASON,
                    now_ms: float | None = None) -> dict:
    """The cached, always-safe summary /api/lobby serves.

    `fetch` and `now_ms` are the two testable seams. `fetch` defaults to
    None here -- deliberately NOT bound to `_fetch_lobby_rows` as a default
    argument value, which would freeze the reference at import time and
    make monkeypatching the module-level name invisible to this function.
    Leaving it None and looking the name up inside the body means a test's
    `monkeypatch.setattr("api.lobby._fetch_lobby_rows", stub)` is honoured
    whether this is called directly or through the registered route.
    `now_ms` pins "now" so a `starts_in_seconds` assertion isn't racing the
    wall clock.

    Every failure -- a network error, a non-2xx response, a body that isn't
    the JSON array `_fetch_lobby_rows` expects -- is caught here and turned
    into `_UNAVAILABLE` rather than propagating: see the module docstring's
    FAILURE IS SILENT section. A cache hit short-circuits before any of
    that runs, so a live ESPN outage can't even be retried more than once
    per TTL window.
    """
    rows = _cached_rows(fetch, season)
    if rows is None:
        return dict(_UNAVAILABLE)
    return _summarize(rows, time.time() * 1000 if now_ms is None else now_ms)


def _lobby_rooms(fetch=None, season: int = CURRENT_SEASON,
                 now_ms: float | None = None) -> dict:
    """The joinable rooms, from the same cached directory read.

    Cookie-free like everything else in this module: the directory is public,
    and listing what is open must not send anybody's ESPN session upstream.
    TAKING a seat does need one, and that lives in `api/drafts.py` where the
    session plumbing already is.

    An unreadable lobby is `{"available": False, "rooms": []}` rather than an
    error, for the same reason the summary has `_UNAVAILABLE`: a dashboard
    section that cannot list rooms should draw nothing, not a red box.
    """
    rows = _cached_rows(fetch, season)
    if rows is None:
        return dict(_NO_ROOMS)
    at = time.time() * 1000 if now_ms is None else now_ms
    return {"available": True,
            # Everything joinable, counted before the cap: `total` is what the
            # lobby holds, `rooms` is what this answer carries.
            "total": len(_joinable(rows, at)),
            "rooms": _rooms(rows, at)}


def _cached_rows(fetch=None, season: int = CURRENT_SEASON):
    """The directory rows, memoised for `_CACHE_TTL_SECONDS`. None if ESPN
    could not be read, which is cached too -- see `_cache`.

    The SHAPE is checked here rather than left to each caller: a body that is
    not a list of row dicts is a failed read, and catching that once at the
    cache boundary is what lets `_summarize` and `_rooms` be plain functions
    over well-formed rows instead of two copies of the same defence.
    """
    now = time.monotonic()
    with _cache_lock:
        if _cache["cached"] and now < _cache["expires_at"]:
            return _cache["rows"]
    return _refresh_rows(fetch, season)


def _refresh_rows(fetch=None, season: int = CURRENT_SEASON):
    """Read the directory now, whatever the cache says, and record what moved.

    Split out of `_cached_rows` for the watcher (`_watch_once`), which is not
    asking for the rows -- it is asking ESPN what changed. Going through the
    cache would have made the watcher a no-op on every tick inside the TTL,
    which is every tick.
    """
    fetch = fetch or _fetch_lobby_rows
    try:
        rows = fetch(season)
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            rows = None
    except Exception:      # noqa: BLE001 -- see FAILURE IS SILENT above
        rows = None

    # Only a good read teaches us anything. A failed one leaves every room's
    # history where it was, so an ESPN blip does not read as the whole lobby
    # emptying and refilling.
    if rows is not None:
        _track_seats(rows)

    with _cache_lock:
        _cache["rows"] = rows
        _cache["cached"] = True
        _cache["expires_at"] = time.monotonic() + _CACHE_TTL_SECONDS
    return rows


def _note_demand(now: float | None = None) -> None:
    """Somebody asked. The watcher reads this to decide whether the lobby is
    worth re-reading -- see `_watch_once`."""
    with _cache_lock:
        _demand["at"] = time.monotonic() if now is None else now


def _wanted(now: float | None = None) -> bool:
    """Has anybody asked recently enough to be worth reading ESPN for?"""
    at = time.monotonic() if now is None else now
    with _cache_lock:
        return at - _demand["at"] <= _DEMAND_WINDOW_SECONDS


def _watch_once(season: int = CURRENT_SEASON) -> bool:
    """One tick of the watcher: a forced read while anybody is looking.

    Returns whether it read. Separate from the loop so a test can tick it by
    hand instead of sleeping.
    """
    if not _wanted():
        return False
    _refresh_rows(season=season)
    return True


def _watch_lobby(season: int = CURRENT_SEASON, stop: threading.Event | None = None,
                 tick: float | None = None) -> None:
    """Re-read the lobby every `_WATCH_SECONDS` for as long as anybody is
    looking. Sleeps FIRST: `register_lobby_routes` has just been called, and
    the first visitor's own request is a better trigger for the first read
    than a thread racing the app's own startup.
    """
    interval = _WATCH_SECONDS if tick is None else tick
    while True:
        if stop is not None:
            if stop.wait(interval):
                return
        else:
            time.sleep(interval)
        try:
            _watch_once(season)
        except Exception:      # noqa: BLE001
            # `_refresh_rows` swallows its own failures; this is the guard
            # against anything else killing the thread for the life of the
            # process, which would silently stop the whole page updating.
            pass


def start_watching(season: int = CURRENT_SEASON) -> None:
    """Start the watcher once per process. Idempotent -- a second call is a
    no-op, so registering the routes twice (tests do) cannot start two."""
    with _cache_lock:
        if _watching["started"]:
            return
        _watching["started"] = True
    threading.Thread(target=_watch_lobby, args=(season,),
                     name="lobby-watch", daemon=True).start()


def fetch_room(league_id, season: int = CURRENT_SEASON, fetch=None) -> dict:
    """One room: its seats, its draft settings and whether it has started.

    COOKIE-FREE, like everything else in this module, and for the same reason
    (see the docstring): a room read is public -- probed live -- and a page any
    visitor can open must not send the owner's ESPN login upstream to build it.

    NOT CACHED. The directory above is a headline number a thousand visitors
    share; this is one room that one reader is watching fill up, and a seat
    taken four seconds ago is exactly what they are looking for.

    `fetch(url) -> body` is injected for tests, and shaped like the drafts
    module's own `fetch(url, cookies, headers)` seam so `api/drafts.py` can
    hand its test double straight through. Raises on anything that is not the
    JSON object this expects: an unreadable room is the caller's to report --
    unlike the summary, there is no widget here to silently draw nothing.
    """
    if fetch is not None:
        status, text = fetch(room_url(league_id, int(season)), None, _LOBBY_HEADERS)
        if int(status) >= 400:
            raise ValueError(f"ESPN answered {status} for room {league_id}")
        payload = json.loads(text) if isinstance(text, (str, bytes)) else text
    else:
        import httpx
        response = httpx.get(room_url(league_id, int(season)),
                             headers=_LOBBY_HEADERS, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(
            f"ESPN's room {league_id} returned {type(payload).__name__}, not an "
            "object")
    return payload


def clear_cache() -> None:
    """Drop the cached directory read. Test hook only -- nothing in this
    module's own request path needs to evict early; the TTL is the only
    invalidation this cache ever needs in production."""
    with _cache_lock:
        _cache["rows"] = None
        _cache["cached"] = False
        _cache["expires_at"] = 0.0
        _demand["at"] = 0.0
    with _seats_lock:
        _seats.clear()


def register_lobby_routes(app):
    """Mount `GET /api/lobby`."""

    @app.get("/api/lobby")
    def lobby():
        _note_demand()
        return _lobby_summary()

    @app.get("/api/lobby/rooms")
    def rooms():
        """The open mock rooms a signed-in reader can take a seat in.

        Public, like the summary above -- the directory is. The seat itself is
        `POST /api/espn/mock-join`, which needs the session this does not.
        """
        _note_demand()
        return _lobby_rooms()

    # LAST, and gated (see WATCH_ON_REGISTER): the routes above must be
    # mounted whether or not this process is one that watches.
    if WATCH_ON_REGISTER:
        start_watching()

    return app
