"""Joining a draft without a bookmarklet, and finding one to join.

WHAT THIS ADDS, IN ONE SENTENCE: with an ESPN session in hand the server can
mint the draft token itself, so the flow becomes "here are your leagues and
when they draft, press join" instead of "find the bookmark, go to the ESPN
tab, click it, and click it again when the token expires".

WHERE THE SESSION COMES FROM, IN PRIORITY ORDER, AND WHY THE ORDER IS FIXED:

  1. The custody cookie. Somebody else's browser, holding a credential this
     deployment took custody of (`api/custody.py`). Authorised by the cookie
     and nothing else -- never by a SWID in a body, which is the account
     takeover `pipeline/espn_identity.py`'s docstring exists to prevent.
  2. The machine owner's own saved login (`data/espn_state.json`), and ONLY
     when the request presented no cookie at all AND came off the machine
     itself (`api/billing.is_local_request`: loopback, no forwarding headers).
     That file is one login, on one machine, put there by the owner running
     the importer's sign-in flow -- except on the deployment, where the same
     path holds the FARM's login (`FARM_ESPN_STATE_B64`). A request that
     reached a container through Railway's edge is never the machine's own,
     so the file is invisible to it, and a stranger is a stranger.

Getting that order backwards would be the whole bug: a visitor whose cookie
had expired would silently be served the OWNER'S leagues and could join the
owner's drafts. So the fallback is reached only when there is nothing to
resolve, never when a resolve fails.

WHY THE ANSWERS ARE CACHED. The draft list is one profile call plus one call
per league -- a dozen HTTP round trips to ESPN for a page the connect screen
polls. The cache is per session identity, short, and in memory: it exists to
keep a page refresh from being a dozen requests, not to serve stale drafts.

WHAT A REJECTED SESSION MEANS. ESPN answering 401/403 is not a transport
failure to retry, it is a credential to delete -- the user has signed out of
ESPN and cannot be told (`pipeline/credentials.py`). So a `SessionExpired`
from any of these routes drops the stored row, clears the browser's cookie,
and answers "not connected", which puts the user back on the bookmarklet path
in one round trip instead of leaving them retrying a dead login.
"""
from __future__ import annotations

import threading
import time

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel

from api.custody import (_store, abandon_custody, custody_for,
                         establish_custody, require_secure)
from pipeline import credentials as cred
from pipeline import espn_drafts as drafts
from pipeline import espn_mock_lobby as lobby

from api import billing
from api import lobby as lobby_rooms
from scoring.config import CURRENT_SEASON

# How long a draft list is reused. Long enough that opening the connect screen
# twice is one round of ESPN calls, short enough that a commissioner moving a
# draft time is visible within a coffee break. Draft times move in hours; this
# is not a live clock and must not be treated as one.
CACHE_SECONDS = 120.0

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()

# How long one room's progress -- picks made, round, whose turn -- stands
# before ESPN is asked again. The Home page polls this for every room its
# reader is drafting in, on the lobby's own 12s clock, and a farm sitting in
# eleven rooms at once is eleven public reads a window; this keeps that at
# eleven whether one dashboard is open or ten. Under the pick clock (30s in
# every lobby room observed), so a card is never more than one pick behind.
PROGRESS_SECONDS = 8.0
# How many rooms one request may ask after. Generous for a person (nobody
# drafts in forty rooms), a bound for a URL somebody types.
PROGRESS_MAX_IDS = 40
_PROGRESS: dict = {}


def _cached(key, build):
    """`build()`, memoised per key for CACHE_SECONDS.

    The lock covers the dict, not the call: two visitors arriving together on
    a cold cache both fetch, which costs a duplicate round trip and keeps a
    slow ESPN from serialising every request behind one lock.
    """
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    value = build()
    with _CACHE_LOCK:
        _CACHE[key] = (now + CACHE_SECONDS, value)
    return value


def _forget(key) -> None:
    with _CACHE_LOCK:
        _CACHE.pop(key, None)


class Session:
    """A resolved ESPN session and where it came from.

    `source` is served to the browser because the two sources mean different
    things to a user: "connected" is an account this browser holds and can
    disconnect, and "local" is this machine's own saved login, which no
    browser control can revoke -- the remedy there is to delete the state file
    or sign out of ESPN.
    """

    def __init__(self, swid: str, espn_s2: str, source: str, resolved=None):
        self.swid = swid
        self.espn_s2 = espn_s2
        self.source = source
        self.resolved = resolved

    @property
    def cookies(self) -> dict:
        return drafts.cookies_for(self.swid, self.espn_s2)

    @property
    def key(self) -> str:
        # Keyed on the identity, not on the cookie: the same account open in
        # two browsers is one league list, and a key made of the password
        # would put a credential in a process-wide dict.
        return f"{self.source}:{drafts.canonical_swid(self.swid)}"


def session_for(request: Request, store=None) -> Session | None:
    """The ESPN session this request may act as, or None.

    See the module docstring for why the cookie is checked first and the local
    login only in its total absence.
    """
    if request.cookies.get(cred.COOKIE_NAME):
        resolved = custody_for(request, store)
        if resolved is None:
            return None
        return Session(resolved.swid, resolved.espn_s2, "connected", resolved)
    # Off the network, the file is not this request's login. On the
    # deployment it is the farm's, and every visitor was being served the
    # owner's leagues -- connected, dashboard, no introduction -- for want
    # of this line.
    if not billing.is_local_request(request):
        return None
    local = drafts.saved_session()
    if local is None:
        return None
    return Session(local[0], local[1], "local")


def _drop(session: Session, response: Response, store=None) -> None:
    """Forget a session ESPN has just rejected.

    Both halves matter. The stored row goes because it cannot be repaired and
    nobody can be asked to repair it; the browser's cookie goes because a
    cookie naming a row that no longer exists would be sent on every request
    forever, and each of those requests would look like an authorisation
    attempt in a log.
    """
    _forget(session.key)
    if session.source != "connected" or session.resolved is None:
        # A local login is a file on this machine, not a row we may delete.
        # The cache entry goes so the next request re-reads it (the owner may
        # have signed in again since), and the rest is the owner's to fix.
        return
    try:
        # The store's own name for this case, so the deletion, its logging and
        # its redaction are the ones `pipeline/credentials.py` already tests
        # rather than a second implementation here. It duck-types on a status,
        # which is why 401 is passed rather than the exception -- a
        # `SessionExpired` carries no `response` for it to read.
        _store(store).forget_if_unauthorized(session.resolved.credential_id, 401)
    except Exception:      # noqa: BLE001 -- a failed cleanup must not mask
        pass               # the real answer, which is "you are signed out"
    response.delete_cookie(cred.COOKIE_NAME, path="/")


def _int_or_none(value):
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _epoch_ms_iso(value) -> str | None:
    """ESPN's epoch milliseconds as ISO 8601 with the Z `new Date(...)` wants
    -- the same shape `_row` serves a league's draft date in, so the browser
    parses one kind of timestamp from this API and not two."""
    ms = _int_or_none(value)
    if ms is None:
        return None
    from datetime import datetime, timezone
    return (datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
            .isoformat().replace("+00:00", "Z"))


def _seconds_until(value) -> float | None:
    """Seconds from now until an epoch-ms moment, negative once it has passed.

    Served alongside the timestamp rather than instead of it: the countdown
    reads this (no clock-skew argument between a browser and this server), and
    anything that has to name the day reads the timestamp.
    """
    ms = _int_or_none(value)
    return None if ms is None else (ms / 1000.0) - time.time()


def _row(draft: dict) -> dict:
    """One league, as JSON the browser can render without a date library."""
    at = draft.get("draft_at")
    return {
        "league_id": draft["league_id"],
        # The account's own team there. It is what makes the row a BUTTON
        # rather than a listing: a join needs a team, and this is the only
        # place ESPN names the one this account owns.
        "team_id": draft.get("team_id"),
        "team_name": draft.get("team_name"),
        "season": draft.get("season"),
        "name": draft.get("name"),
        "teams": draft.get("teams"),
        "draft_type": draft.get("draft_type"),
        # ISO 8601 with the Z, which is what `new Date(...)` wants. None for a
        # league whose commissioner has not scheduled one -- the browser shows
        # "no date set" rather than a countdown to nothing.
        "draft_at": None if at is None else at.isoformat().replace("+00:00", "Z"),
        "live": bool(draft.get("live")),
    }


class JoinBody(BaseModel):
    leagueId: str
    teamId: str
    season: str | None = None


class MockJoinBody(BaseModel):
    """One room from ESPN's mock lobby, and optionally one seat in it.

    `GET /api/lobby/rooms` is where the id comes from -- a public, cookie-free
    listing (api/lobby.py). Only this half needs to know who is asking.

    `teamId` is the seat the reader picked in the waiting room. Absent means
    "any open seat" (ESPN's own `-1` sentinel, what the farm sends), which is
    the right answer for anybody who does not care where they sit.
    """

    leagueId: str
    season: str | None = None
    teamId: str | None = None


class ConnectBody(BaseModel):
    """An ESPN account session, delivered by the bookmarklet.

    `espn_s2` IS the account. It arrives here because the bookmarklet runs on
    ESPN's own origin and can read it (measured: not HttpOnly, whatever the
    old comments in this repo claimed) -- and because `sameSite=Lax` means
    nothing on the helper's own origin ever could, which is why there is a
    bookmarklet rather than a button.
    """

    swid: str
    espn_s2: str


def register_draft_routes(app, store=None, fetch=None, post=None):
    """`GET /api/espn/drafts`, `POST /api/espn/draft-token`, and the mock-room
    join that reuses both.

    `fetch` (reads) and `post` (the one write ESPN takes: an invite) are
    injectable for the same reason everything else in this subsystem is: the
    whole of this logic is testable with no network, no ESPN and no real
    credential.
    """

    @app.get("/api/espn/drafts")
    def upcoming(request: Request, response: Response, season: str | None = None):
        """This account's leagues that have not drafted yet, soonest first.

        Answers 200 with `connected: false` for a visitor with no session
        rather than 401, for the reason `custody_status` gives: this is the
        probe the connect screen runs on load, and the ordinary case is a
        stranger.
        """
        session = session_for(request, store)
        if session is None:
            return {"connected": False, "leagues": []}
        year = season or str(CURRENT_SEASON)
        try:
            rows = _cached(
                (session.key, year),
                lambda: drafts.upcoming_drafts(session.swid, session.cookies,
                                               year, fetch=fetch))
        except drafts.SessionExpired as exc:
            _drop(session, response, store)
            return {"connected": False, "leagues": [],
                    "expired": True, "detail": str(exc)}
        return {"connected": True, "source": session.source,
                "season": int(year), "leagues": [_row(r) for r in rows]}

    @app.post("/api/espn/connect")
    def connect_account(body: ConnectBody, request: Request, response: Response):
        """Take custody of an ESPN session and hand this browser its key.

        SEPARATE FROM JOINING A DRAFT, deliberately. `/api/live/connect-token`
        can also carry an `espn_s2`, because the bookmarklet used to have
        exactly one moment to hand one over -- the click that joined a draft.
        It does not any more: a click on any ESPN page can connect the
        account, and the leagues to join are then discovered rather than
        supplied. One thing per route, and this one stores nothing about a
        draft at all.

        Everything that makes this safe is `establish_custody`'s and stays
        there: the TLS refusal, the proof that the session really belongs to
        the account it names (`pipeline/espn_identity.py` -- a SWID is public,
        so a request may not simply assert one), the encrypted row, and the
        `httpOnly` cookie that is the only thing able to read it back.
        """
        minted = establish_custody(request, response, body.swid, body.espn_s2,
                                   store=store)
        # The list this browser can now see. Built here rather than left to a
        # follow-up request so the connect answers the question the user
        # actually asked -- "what can I do now" -- in one round trip.
        session = Session(body.swid, body.espn_s2, "connected")
        try:
            rows = drafts.upcoming_drafts(body.swid, session.cookies,
                                          str(CURRENT_SEASON), fetch=fetch)
        except drafts.SessionExpired:
            # Verified a moment ago by `establish_custody` and rejected now:
            # nothing to do but say so. The row it just wrote is dropped, so a
            # dead session is never left behind by the request that took it.
            abandon_custody(minted)
            response.delete_cookie(cred.COOKIE_NAME, path="/")
            raise HTTPException(
                status_code=401,
                detail="ESPN rejected that session. Sign in to ESPN and "
                       "click the bookmarklet again.") from None
        except Exception:      # noqa: BLE001 -- the connect SUCCEEDED; a
            rows = []          # list we could not build is not a failure of it
        _CACHE.pop((session.key, str(CURRENT_SEASON)), None)
        return {"connected": True, "source": "connected",
                "season": CURRENT_SEASON, "leagues": [_row(r) for r in rows]}

    @app.post("/api/espn/draft-token")
    def draft_token(body: JoinBody, request: Request, response: Response):
        """Mint the token for one team, the way the bookmarklet would.

        The token is RETURNED rather than acted on, so the browser hands it to
        `/api/live/connect-token` exactly as the bookmarklet's popup does
        today. That keeps one connect path in this codebase instead of two
        that have to be kept in step -- the difference between the bookmarklet
        and this is only where the token came from.

        The transport guard applies even though no credential is in the
        REQUEST: the response is minted with one, and a token that can drive
        somebody's draft has no business crossing a plaintext wire either.
        """
        session = session_for(request, store)
        if session is None:
            raise HTTPException(
                status_code=401,
                detail="No ESPN session. Use the bookmarklet, or connect an "
                       "account first.")
        if session.source == "connected":
            require_secure(request)
        try:
            token = drafts.mint_draft_token(
                body.leagueId, body.teamId, body.season or str(CURRENT_SEASON),
                session.cookies, fetch=fetch)
        except drafts.SessionExpired as exc:
            _drop(session, response, store)
            raise HTTPException(status_code=401, detail=str(exc)) from None
        except RuntimeError as exc:
            # ESPN said no for a reason that is not the session: the draft is
            # not open, the team is not ours, ESPN is down. A 502 says "the
            # upstream refused" rather than blaming the caller's request.
            raise HTTPException(status_code=502, detail=str(exc)) from None
        return {"token": token, "swid": session.swid,
                "leagueId": body.leagueId, "teamId": body.teamId,
                "season": int(body.season or CURRENT_SEASON)}

    @app.get("/api/espn/mock-room/{league_id}")
    def mock_room(league_id: str, request: Request, season: str | None = None):
        """One mock room's seats, its clock, and whether it has started.

        WHAT THE WAITING ROOM IS DRAWN FROM. Every field here comes from a
        single public ESPN read (`lobby.room_url`, three views in one request)
        -- who is in each seat, the draft's own start time and pick clock, the
        pick order that decides which picks a seat gets, and whether picking
        has begun.

        NO SESSION REQUIRED, and no session sent upstream. The room is public;
        the only thing this needs a session FOR is saying which of those seats
        is yours, which is decided here by comparing SWIDs rather than by
        asking ESPN as you. A visitor with no session gets the same room with
        no seat marked.

        Not cached: the whole point of this endpoint is that a seat was taken
        four seconds ago, and the page polls it while somebody decides.
        """
        session = session_for(request, store)
        year = season or str(CURRENT_SEASON)
        try:
            payload = lobby_rooms.fetch_room(league_id, int(year), fetch=fetch)
        except Exception as exc:      # noqa: BLE001 -- ESPN unreachable, or a
            # room that has been torn down since the directory listed it.
            raise HTTPException(
                status_code=502,
                detail=f"Could not read room {league_id}: {exc}") from None
        settings = (payload.get("settings") or {}).get("draftSettings") or {}
        detail = payload.get("draftDetail") or {}
        seats = lobby.seats(payload, session.swid if session else None)
        mine = next((s["team_id"] for s in seats if s["mine"]), None)
        at = _epoch_ms_iso(settings.get("date"))
        return {
            "league_id": str(league_id),
            "teams": len(seats),
            "draft_type": settings.get("type"),
            # Seconds per pick, which is the other clock this room runs and the
            # one a reader is deciding whether they can keep up with.
            "clock_seconds": _int_or_none(settings.get("timePerSelection")),
            "draft_at": at,
            "starts_in_seconds": _seconds_until(settings.get("date")),
            # HOW MANY ROUNDS, from the pick list ESPN pre-populates before a
            # ball is drafted: `draftDetail.picks` is already the full 160 (or
            # 128, or 320) slots of the room, so rounds is that over the seat
            # count. NOT from `rosterSettings.lineupSlotCounts`, which sums to
            # 17 on a 16-round room because it counts an IR slot nobody drafts
            # into -- measured on a live room, and a board drawn with a
            # seventeenth round would be inventing a pick.
            "rounds": (len(detail.get("picks") or []) // len(seats)) if seats else None,
            "in_progress": bool(detail.get("inProgress")),
            "drafted": bool(detail.get("drafted")),
            "my_team_id": mine,
            "seats": seats,
        }

    @app.get("/api/espn/rooms/progress")
    def rooms_progress(ids: str = ""):
        """How far along each of these rooms is: picks made, round, pick.

        THE HOME PAGE'S LIVE CARDS, in one request. A room you are drafting
        in says "round 3 · pick 6" rather than a countdown, and that number
        moves every thirty seconds, so the page asks for all of its rooms
        together on the poll it already runs for the lobby.

        TWO SOURCES, AND THE FARM'S COMES FIRST. ESPN's public room read
        (`fetch_room`, the one the waiting room draws) lists every slot but
        names no player until the draft is OVER -- probed live against six
        rooms mid-draft: 0 picks each by that read, while the farm's files
        held 12 to 103. The picks travel on the draft socket, and the farm is
        on it: a room it sits in is counted from the file it writes on every
        pick (`mock_farm`, the same file the landing page's hero reads). A
        room the farm is not in gets ESPN's read for its shape and state, and
        an honest `None` for the count rather than "round 1, pick 1".

        No session, none sent upstream -- the read is public. A room ESPN
        will not read is left out rather than failing the request: one
        torn-down room must not blank ten live cards.
        """
        import os
        from api import demo as hero
        wanted = [i.strip() for i in ids.split(",") if i.strip()][:PROGRESS_MAX_IDS]
        out = {}
        for league_id in wanted:
            now = time.monotonic()
            with _CACHE_LOCK:
                hit = _PROGRESS.get(league_id)
            if hit is not None and hit[0] > now:
                if hit[1] is not None:
                    out[league_id] = hit[1]
                continue
            row = None
            path = os.path.join(hero.FARM_DIR, f"{league_id}.json")
            try:
                fresh = time.time() - os.stat(path).st_mtime <= hero.STALE_SECONDS
            except OSError:
                fresh = False
            record = hero._record(path) if fresh else None
            if record and int(record.get("teams") or 0):
                teams = int(record["teams"])
                rounds = int(record.get("rounds") or 0) or None
                made = len(record.get("picks") or [])
                total = teams * rounds if rounds else None
                row = {
                    "picks_made": made,
                    "picks_total": total,
                    "teams": teams,
                    "rounds": rounds,
                    # The pick that is UP, in round terms: 37 made means the
                    # 38th is on the clock, the sixth of round five.
                    "round": made // teams + 1,
                    "pick_in_round": made % teams + 1,
                    "in_progress": total is None or made < total,
                    "drafted": total is not None and made >= total,
                }
            else:
                try:
                    payload = lobby_rooms.fetch_room(league_id, CURRENT_SEASON, fetch=fetch)
                    detail = payload.get("draftDetail") or {}
                    picks = detail.get("picks") or []
                    teams = len(payload.get("teams") or [])
                    row = {
                        "picks_made": None,
                        "picks_total": len(picks) or None,
                        "teams": teams,
                        "rounds": (len(picks) // teams) if teams else None,
                        "round": None,
                        "pick_in_round": None,
                        "in_progress": bool(detail.get("inProgress")),
                        "drafted": bool(detail.get("drafted")),
                    }
                except Exception:      # noqa: BLE001 -- unreadable is "no answer"
                    row = None
            with _CACHE_LOCK:
                _PROGRESS[league_id] = (now + PROGRESS_SECONDS, row)
            if row is not None:
                out[league_id] = row
        return {"rooms": out}

    @app.post("/api/espn/mock-join")
    def mock_join(body: MockJoinBody, request: Request, response: Response):
        """Take a seat in an open ESPN mock room, then mint for that seat.

        TWO ESPN CALLS, IN ORDER, AND THE SEAT COMES FIRST. `lobby.join` POSTs
        the invite with `teamId: -1` ("any open seat") and ESPN answers with
        the team it assigned; that number is READ, never assumed. Minting for
        a guessed team would produce a socket URL for somebody else's seat,
        and the failure would surface as a draft in which the user's picks
        never land rather than as an error here.

        What comes back is the same five values `/api/espn/draft-token`
        returns, so the browser hands them to `/api/live/connect-token` exactly
        as it does for a real league. A mock draft and a real one differ in
        where the seat came from and in nothing after it.

        THE LOBBY MOVES IN SECONDS. Rooms fill between the listing being read
        and a row being clicked, and ESPN answers that with a refusal rather
        than a seat. That is a 502 with ESPN's own reason attached -- the
        upstream said no -- and the page re-reads the list rather than
        retrying a room that is gone.
        """
        session = session_for(request, store)
        if session is None:
            raise HTTPException(
                status_code=401,
                detail="No ESPN session. Connect an account first.")
        if session.source == "connected":
            require_secure(request)
        year = body.season or str(CURRENT_SEASON)
        poster = post or lobby.http_poster(session.cookies)
        try:
            team_id = lobby.join(poster, body.leagueId, session.swid, int(year),
                                 team_id=int(body.teamId) if body.teamId else None)
        except drafts.SessionExpired as exc:
            _drop(session, response, store)
            raise HTTPException(status_code=401, detail=str(exc)) from None
        except Exception as exc:      # noqa: BLE001 -- ESPN refused the seat:
            # the room filled, it started, or the write host said no. Every one
            # of those is the upstream refusing rather than a bad request, and
            # every one of them is fixed by reading the lobby again.
            raise HTTPException(
                status_code=502,
                detail=f"ESPN would not seat you in room {body.leagueId}: "
                       f"{exc}") from None
        # This room is a mock, and the gate in api/billing.py has to know it
        # without asking ESPN on the connect path -- a wrong answer there
        # charges somebody for a free draft. Recorded after the seat rather
        # than before it: a join ESPN refused is not a room anybody entered.
        billing.note_mock_room(body.leagueId)
        try:
            token = drafts.mint_draft_token(body.leagueId, team_id, year,
                                            session.cookies, fetch=fetch)
        except drafts.SessionExpired as exc:
            _drop(session, response, store)
            raise HTTPException(status_code=401, detail=str(exc)) from None
        except RuntimeError as exc:
            # The seat is taken and the token is not. Said plainly, because the
            # user IS in that room now -- retrying the join would take a second
            # seat in it.
            raise HTTPException(
                status_code=502,
                detail=f"Took seat {team_id} in room {body.leagueId} but ESPN "
                       f"would not mint the draft token: {exc}") from None
        return {"token": token, "swid": session.swid,
                "leagueId": body.leagueId, "teamId": str(team_id),
                "season": int(year), "mock": True}
