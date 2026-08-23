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

import threading
import time

from pipeline.espn_mock_lobby import lobby_url
from scoring.config import CURRENT_SEASON

# One upstream call per this many seconds, no matter how many visitors poll
# /api/lobby in between -- see the module docstring's CACHING section. 60s
# matches the widget's own slowest poll interval (LobbyStrip.tsx): there is
# no benefit to caching tighter than the fastest anything will ever ask.
_CACHE_TTL_SECONDS = 60.0

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

_cache_lock = threading.Lock()
# `payload` is None until the first successful-or-failed fetch populates it;
# `expires_at` is a `time.monotonic()` deadline so a system clock change
# mid-process can't extend or shrink the window.
_cache = {"payload": None, "expires_at": 0.0}


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


def _room_summary(row: dict, now_ms: float) -> dict:
    """One directory row, reduced to what the widget shows about it.

    `starts_in_seconds` is clamped at 0 rather than left negative -- a row
    can slip from "future" to "already started" between the summary being
    built and the response being read, and a widget counting down should
    never show a negative number for that.
    """
    draft_date = row.get("draftDate")
    starts_in = None
    if draft_date is not None:
        starts_in = max(0, int((float(draft_date) - now_ms) / 1000.0))
    return {
        "league_size": row.get("leagueSize"),
        "scoring": row.get("rankType"),
        "teams_joined": int(row.get("teamsJoined") or 0),
        "starts_in_seconds": starts_in,
    }


def _summarize(rows: list, now_ms: float) -> dict:
    """Reduce the raw directory to the summary /api/lobby serves.

    `total_open` is every row ESPN's directory lists -- the whole point of
    the headline number is "this many rooms exist right now", not a
    filtered subset. `joinable` is exactly the spec's definition: not full,
    and its draft hasn't started yet. A room already `draftInProgress` with
    a stale `draftDate` in the past is excluded by the draft-in-the-future
    clause alone; a genuinely joinable room's `draftDate` is, by
    construction, still ahead of `now_ms`.
    """
    joinable_rows = [
        row for row in rows
        if not row.get("full")
        and row.get("draftDate") is not None
        and float(row["draftDate"]) > now_ms
    ]
    joinable_rows.sort(key=lambda row: float(row["draftDate"]))
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
    now = time.monotonic()
    with _cache_lock:
        cached = _cache["payload"]
        if cached is not None and now < _cache["expires_at"]:
            return cached

    fetch = fetch or _fetch_lobby_rows
    try:
        rows = fetch(season)
        effective_now_ms = time.time() * 1000 if now_ms is None else now_ms
        payload = _summarize(rows, effective_now_ms)
    except Exception:
        payload = dict(_UNAVAILABLE)

    with _cache_lock:
        _cache["payload"] = payload
        _cache["expires_at"] = time.monotonic() + _CACHE_TTL_SECONDS
    return payload


def clear_cache() -> None:
    """Drop the cached summary. Test hook only -- nothing in this module's
    own request path needs to evict early; the TTL is the only invalidation
    this cache ever needs in production."""
    with _cache_lock:
        _cache["payload"] = None
        _cache["expires_at"] = 0.0


def register_lobby_routes(app):
    """Mount `GET /api/lobby`."""

    @app.get("/api/lobby")
    def lobby():
        return _lobby_summary()

    return app
