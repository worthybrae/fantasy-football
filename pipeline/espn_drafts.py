"""What a stored ESPN session can do that a bookmarklet cannot.

THE BOOKMARKLET'S WHOLE TRICK IS ONE HTTP GET. It runs on ESPN's own origin so
the browser will attach `espn_s2` to a request for
`.../teams/{team}/draftSecurity`, which mints the per-draft token the socket
opens with. Nothing about that GET needs a browser -- it needs the cookie. So
a process holding a stored `espn_s2` can mint the same token itself, and the
drafter never has to find a bookmark, click it, or click it AGAIN when the
token expires mid-draft (see `api/live.py`'s "click the bookmark again" copy,
which is the failure this removes).

AND THE SAME COOKIE ANSWERS A QUESTION NOTHING ELSE CAN: which leagues is this
person in, and when do they draft. `fan.api.espn.com` knows the first, the
league's own `mSettings` view knows the second, and neither is reachable
without an account session.

EVERY FUNCTION HERE TAKES ITS COOKIES PER CALL, for the reason
`pipeline/espn_identity.py` spells out at length: this module is asked about a
DIFFERENT person's session on every request, and a fetcher closing over the
owner's own login would answer for the owner every time.

FAILING ON AN EXPIRED SESSION IS A FEATURE, NOT AN ERROR PATH. ESPN answers
401/403 for a session the user has logged out of, and this module raises
`SessionExpired` for exactly those two. The caller's job is then to DELETE the
stored row rather than retry it: a credential ESPN has just rejected is not
going to start working, and the user cannot be told (see
`pipeline/credentials.py` on why there is no way to contact anybody).

SHAPES WE DO NOT OWN. The fan profile's layout is ESPN's, undocumented, and
has moved before. `league_ids` therefore walks it tolerantly and returns []
when it finds nothing it recognises, rather than raising: "we could not read
your league list" degrades to the bookmarklet, while an exception out of a
background scan would take a page down over somebody else's JSON.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from pipeline.espn_identity import canonical_swid
from pipeline.espn_league import BASE, STATE_PATH

FAN_BASE = "https://fan.api.espn.com/apis/v2/fans"

# ESPN 403s a draftSecurity request without it. Sent on every call here rather
# than only that one: the same header rides on the bookmarklet's fetch and on
# ESPN's own front end, so a request carrying it looks like the traffic these
# endpoints expect.
KONA = {"x-fantasy-source": "kona"}

# The fantasy football game. ESPN keys a fan's preferences by game, and this
# tool is football-only (`ffl` is in BASE itself).
FFL_GAME_ABBREV = "ffl"


class SessionExpired(RuntimeError):
    """ESPN rejected the session -- 401 or 403 rather than a transport fault.

    Separate from every other failure because the correct response is
    different in kind: a network blip is worth a retry and this is worth a
    DELETE. Carrying the distinction in the type keeps that decision at the
    caller, where the credential store is, instead of in a status code
    somebody has to remember to check.
    """


def http_fetch():
    """A `fetch(url, cookies, headers) -> (status, text)` callable hitting ESPN.

    Mirrors `espn_identity.http_fetch`, with headers as a third argument
    because two of the three endpoints here refuse the request without one.
    Never raises for a status: a 401 is data (an expired session is an
    ordinary state), so the status comes back and the caller decides.
    """
    import httpx

    def fetch(url: str, cookies: dict | None, headers: dict | None = None):
        response = httpx.get(url, cookies=cookies or {}, timeout=10.0,
                             headers={"User-Agent": "Mozilla/5.0",
                                      **(headers or {})})
        return response.status_code, response.text
    return fetch


def cookies_for(swid: str, espn_s2: str) -> dict:
    """The two-cookie jar ESPN wants, from a resolved credential.

    SWID is sent in its braced form because that is how ESPN sets it and how
    every one of its own endpoints reads it back; `canonical_swid` is the same
    normalizer the identity check uses, so a stored session and a live request
    cannot disagree about whether the braces are part of the value.
    """
    return {"SWID": canonical_swid(swid), "espn_s2": espn_s2}


def _body(status: int, text: str, what: str):
    """The parsed JSON body, or the right exception for what came back."""
    if status in (401, 403):
        raise SessionExpired(
            f"ESPN rejected the session while reading {what} ({status}). "
            "The account has been signed out of ESPN, or the session expired.")
    if status >= 400:
        raise RuntimeError(f"ESPN returned {status} for {what}.")
    try:
        return json.loads(text or "")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"ESPN returned unreadable {what}.") from exc


# -- where a session comes from ----------------------------------------------

def saved_session(state_path: str = STATE_PATH) -> tuple | None:
    """The machine owner's own ESPN login, out of Playwright's storage state.

    THE LOCAL PATH, and the reason this feature works today without anybody
    handing us a credential over a wire. `pipeline/espn_league.py`'s login flow
    already writes a browser context to `data/espn_state.json` -- cookie jar
    included, HttpOnly and all -- because the importer needs it. So on the
    machine that owns that file the session is ALREADY here, and asking the
    user to also paste it through an HTTPS endpoint into an encrypted table
    would be ceremony around a credential sitting in plaintext next to it.

    This is NOT a multi-user path and must never become one: the file is one
    login, the owner's. `api/drafts.py` reaches for it only when the request
    carries no custody cookie, which keeps "somebody else's browser" and "this
    machine's own login" from ever being confused for each other.

    Returns `(swid, espn_s2)` or None -- None for no file, unreadable JSON, or
    a jar missing either cookie, all of which mean the same thing to a caller:
    there is no local login, offer the bookmarklet.
    """
    try:
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        return None
    jar = {}
    for cookie in (state.get("cookies") or []):
        if isinstance(cookie, dict) and cookie.get("name") in ("SWID", "espn_s2"):
            jar[cookie["name"]] = cookie.get("value")
    if not jar.get("SWID") or not jar.get("espn_s2"):
        return None
    return canonical_swid(jar["SWID"]), jar["espn_s2"]


# -- the token the socket opens with -----------------------------------------

def draft_token_url(league_id, team_id, season) -> str:
    return (f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}"
            f"/teams/{team_id}/draftSecurity")


def mint_draft_token(league_id, team_id, season, cookies: dict,
                     fetch=None) -> str:
    """The per-draft token for one team, minted server-side.

    THE SAME CALL THE BOOKMARKLET MAKES, with the cookie supplied rather than
    attached by a browser. The response is a bare integer as text, which is
    why this strips and validates rather than parsing JSON -- an HTML error
    page would otherwise sail through as a "token" and fail much later, at the
    socket, as something unrecognisable.

    MINTED AT JOIN TIME, NEVER STORED. It is scoped to one team in one draft
    and it expires; the thing worth keeping is the session that can mint
    another (see the module docstring).
    """
    fetch = fetch or http_fetch()
    status, text = fetch(draft_token_url(league_id, team_id, season),
                         cookies, KONA)
    if status in (401, 403):
        raise SessionExpired(
            f"ESPN rejected the session while minting a draft token ({status}).")
    if status >= 400:
        raise RuntimeError(
            f"ESPN returned {status} minting a draft token. Is the draft open?")
    token = (text or "").strip()
    # A SIGNED integer. Measured, not assumed: a real mock draft answered
    # `-1872384467`, and an `isdigit()` check -- which is what this started as
    # -- rejected a perfectly good token as "the draft is not open". The check
    # exists to catch an HTML error page arriving as a 200, so it needs to be
    # exactly as strict as that and no stricter.
    try:
        int(token)
    except ValueError:
        raise RuntimeError(
            "ESPN did not return a draft token. The draft may not be open "
            "yet, or the session is not on this team.") from None
    return token


# -- which leagues, and when they draft --------------------------------------

def fan_url(swid: str) -> str:
    """The account profile, with the preferences that name a user's leagues.

    `?featureFlags=...` is what makes ESPN include them: the bare profile is
    the account row this project already reads for identity verification
    (`espn_identity.fan_url`), and it carries no league list at all.
    """
    from urllib.parse import quote
    return (f"{FAN_BASE}/{quote(canonical_swid(swid), safe='')}"
            "?featureFlags=expandAthlete&featureFlags=isolateEvents"
            "&showAirings=buy,live,replay&showFantasyEntries=true")


def _entries(payload):
    """Every fantasy `metaData.entry` in a fan profile, at any depth.

    Deliberately structural rather than positional. ESPN's preferences are a
    list of typed objects and the fantasy ones have moved inside it before;
    what has stayed put is that a fantasy entry is a dict carrying an
    `entryId` and a `groups` list. Walking for that shape survives a
    reshuffle that an index-based read would not.
    """
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "entryId" in node and isinstance(node.get("groups"), list):
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found


# ESPN's own game ids and draft states, named where they are read rather than
# left as magic numbers in a condition. Measured against a real profile:
# football is game 1; a draft that has finished reports status 2 and sets
# `entryMetadata.draftComplete`, and one in progress reports 3.
FFL_GAME_ID = 1
DRAFT_DONE = 2
DRAFT_LIVE = 3


def _profile(swid: str, cookies: dict, fetch=None):
    fetch = fetch or http_fetch()
    status, text = fetch(fan_url(swid), cookies, KONA)
    return _body(status, text, "your ESPN profile")


def league_entries(swid: str, cookies: dict, season=None, fetch=None) -> list:
    """This account's fantasy-football teams, one row each.

    ONE HTTP CALL FOR THE WHOLE LIST, and this is the discovery that shaped
    the module: the fan profile was expected to carry league ids and nothing
    else, so the first version fetched each league's `mSettings` view for its
    draft date -- a dozen round trips for a page the connect screen polls.
    The profile already has all of it: the league's id, name and size, the
    draft's date, type and state, and -- the one nothing else could supply --
    `entryId`, which is the TEAM id this account owns in that league. Without
    it the join button would have nothing to join as.

    `season` filters to one year, because a profile carries every season the
    account has played and a 2019 league is not a draft anybody is waiting
    for. None means every season, which is what the tests read.

    Returns [] rather than raising when the profile holds nothing we
    recognise -- see the module docstring on shapes we do not own.
    """
    rows = []
    for entry in _entries(_profile(swid, cookies, fetch=fetch)):
        game = entry.get("gameId")
        abbrev = str(entry.get("gameAbbrev") or "").lower()
        if game is not None and game != FFL_GAME_ID:
            continue
        if abbrev and abbrev != FFL_GAME_ABBREV:
            continue
        year = entry.get("seasonId")
        if season is not None and year is not None and int(year) != int(season):
            continue
        for group in entry.get("groups") or []:
            if not isinstance(group, dict) or group.get("groupId") is None:
                continue
            meta = entry.get("entryMetadata") or {}
            status = group.get("draftStatus")
            rows.append({
                "league_id": str(group["groupId"]),
                # The account's own team in that league. `entryId` is what
                # ESPN calls it here and `teamId` is what it calls the same
                # number in every url; the row uses the url's word, because
                # that is what the join hands back to the connect.
                "team_id": str(entry.get("entryId")),
                "season": int(year) if year is not None else None,
                "name": group.get("groupName"),
                "team_name": meta.get("teamName"),
                "teams": group.get("groupSize"),
                "draft_type": group.get("draftTypeName") or group.get("draftType"),
                "draft_at": _epoch_ms(group.get("draftDate")),
                # Two sources for one fact, and either is enough: the group's
                # status is about the DRAFT and the entry's flag is about this
                # team's part in it, and a profile mid-migration has been seen
                # to carry one without the other.
                "drafted": bool(meta.get("draftComplete")) or status == DRAFT_DONE,
                "live": status == DRAFT_LIVE,
            })
    return rows


def league_ids(swid: str, cookies: dict, fetch=None) -> list:
    """Just the league ids, in profile order, deduplicated.

    Kept as its own function because it is the question `league_draft` and any
    future per-league work start from, and because it is the smallest thing a
    caller can ask that still proves the session works.
    """
    seen, ids = set(), []
    for row in league_entries(swid, cookies, fetch=fetch):
        if row["league_id"] not in seen:
            seen.add(row["league_id"])
            ids.append(row["league_id"])
    return ids


def _epoch_ms(value):
    """ESPN's millisecond timestamps, as an aware UTC datetime or None."""
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return None
    if ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def league_draft(league_id, season, cookies: dict, fetch=None) -> dict:
    """One league's name, size, and when it drafts.

    Reads the same `mSettings` view `pipeline/espn_teams.py` already fetches
    for scoring and roster rules, so this adds an endpoint's worth of nothing:
    `settings.draftSettings` carries `date` (epoch ms) beside the `type` and
    `pickOrder` that module already parses.

    `drafted` is ESPN's own flag for a draft that has finished. It is served
    rather than inferred from the clock: a draft can be over long before its
    scheduled time (everyone autopicked) and can start late, and a list that
    called a finished draft "upcoming" would send a user into a dead room.

    `pick_order` is the seating plan, and it is the only thing here the fan
    profile cannot supply: `entryId` names the account's TEAM and a team id is
    not a seat. Empty until the commissioner sets the order -- ESPN publishes
    the field the moment it exists and omits it entirely before -- so an empty
    list means "not decided yet", never "you are drafting first".
    """
    fetch = fetch or http_fetch()
    url = (f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}"
           "?view=mSettings")
    payload = _body(*fetch(url, cookies, KONA), f"league {league_id}")
    settings = (payload.get("settings") or {}) if isinstance(payload, dict) else {}
    draft = settings.get("draftSettings") or {}
    return {
        "league_id": str(league_id),
        "season": int(season),
        "name": settings.get("name"),
        "teams": settings.get("size"),
        "draft_type": draft.get("type"),
        "draft_at": _epoch_ms(draft.get("date")),
        "drafted": bool(draft.get("drafted")),
        # Team ids in seat order. Read here rather than in a caller so the one
        # place that knows this view's shape stays the one place that parses
        # it -- see `slot_in_pick_order` for what the order MEANS.
        "pick_order": list(draft.get("pickOrder") or ()),
    }


def slot_in_pick_order(pick_order, team_id) -> int | None:
    """Which seat a team drafts from, out of ESPN's own published order.

    ONE PLACE KNOWS THAT `pickOrder[k]` IS SLOT k+1. That fact is read by the
    live room (`api/live._slot_from_pick_order`, which calls this), by the
    board's column names (`pipeline/espn_teams.fetch_team_slots`) and now by
    the dashboard's plan -- and a second copy of it that drifted by one would
    hand somebody a plan for the seat next to theirs, which is a wrong answer
    nothing downstream can detect.

    TOLERANT ABOUT THE TYPE, STRICT ABOUT THE ANSWER. A team id arrives as an
    int from a settings payload and as a string from the fan profile
    (`league_entries` serves ESPN's `entryId` as text, because that is what a
    url wants), so both are compared as integers. Anything that is not a
    number -- on either side -- is skipped rather than guessed at.

    None whenever it cannot answer: no order published, no team, or a team the
    order does not carry. Never a guess; the caller says "not known yet",
    which is a true sentence, and a fabricated seat is not.
    """
    if not pick_order or team_id is None:
        return None
    try:
        wanted = int(team_id)
    except (TypeError, ValueError):
        return None
    for slot, entry in enumerate(pick_order, start=1):
        try:
            if int(entry) == wanted:
                return slot
        except (TypeError, ValueError):
            continue
    return None


def upcoming_drafts(swid: str, cookies: dict, season=None, fetch=None,
                    now: datetime | None = None) -> list:
    """Every league this account has that has not drafted yet, soonest first.

    A draft that has STARTED is kept and flagged `live`, because that is the
    one a user most needs to reach -- a room with a clock running is not
    "upcoming" but it is certainly not something to hide. Only a finished
    draft is dropped, on ESPN's own flag rather than on the clock: a draft can
    end long before its scheduled time (everybody autopicked) and can start
    late, and a list that called a finished draft upcoming would send someone
    into a dead room.

    A league with no scheduled date sorts last rather than being dropped: "in
    a league, no date set" is a true and useful thing to show a drafter in
    August, and it is exactly the league worth going to check.
    """
    now = now or datetime.now(timezone.utc)
    far = datetime.max.replace(tzinfo=timezone.utc)
    rows = [r for r in league_entries(swid, cookies, season, fetch=fetch)
            if not r["drafted"]]
    for row in rows:
        at = row["draft_at"]
        # Past its hour and not finished: the room is open, or the league let
        # the time slide. Either way a countdown reading "-3 days" is nonsense
        # and "live" is the honest label.
        if at is not None and at < now:
            row["live"] = True
    rows.sort(key=lambda r: (r["draft_at"] is None, r["draft_at"] or far))
    return rows
