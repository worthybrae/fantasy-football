"""ESPN's public mock-draft lobby: what rooms are open, and how to take a seat.

WHY A LOBBY CLIENT EXISTS AT ALL. `pipeline.mock_backfill` harvests mock
drafts that were already played by hand and left on disk -- a fixed pile that
does not grow on its own. The pick model's cold-start prior is fitted on
about 700 picks from one league of eight people, and E006 measured that a
corpus that small cannot resolve an effect smaller than roughly r = 0.4. The
only answer is more drafts, and ESPN gives them away: its mock lobby opens a
new room every few minutes, all night, for free. This module is the half of
the farm that finds a room and gets into it; `pipeline.mock_farm` is the half
that plays it and writes it down.

EVERY ENDPOINT HERE WAS PROBED LIVE before it was written, and the findings
are recorded in this plan's `espn-lobby-intel.md`. Two of them cost a round
of guessing each and are worth restating where the code is:

  - The directory path segment is the subType NAME, `MOCKDRAFT_LOBBY`, not
    the numeric id (4). The numeric form does not answer.
  - The invite POST requires `join=true` in the query string. Without it
    ESPN returns HTTP 400 with an HTML body and no explanation of what was
    wrong. The body is also a JSON ARRAY, `[{"teamId": -1}]`, not an object,
    and `-1` means "any open seat".

ROOM SELECTION IS A POLICY, NOT A PREFERENCE. `pick_room` filters to 8-team
PPR snake rooms and nothing else. That is the owner's explicit instruction
(2026-08-23) and it is load-bearing for the corpus: `scoring.config.
LEAGUE_TEAMS` is 8, every draft Task 1 harvested is 8x16, and a prior fitted
over a mixture of 8-, 12- and 20-team rooms is averaging over league shapes
this tool is never used on. Team count changes who is on the clock at pick
k, which changes every reach/fall feature the model reads; it is not a
nuisance parameter.

WHAT THE RANKING IS ACTUALLY FOR. Survivors are ordered by `teamsJoined`
descending before anything else. A room at 6/8 is six humans waiting for a
draft to fill, and their picks are the signal the corpus wants. A room at 0/8
fills with ESPN's own autodraft engine and teaches us nothing but ADP read
back to us. Joining the fuller room is also the only decent thing to do with
a seat that a real person might otherwise be waiting on -- the measured
supply is a new 8-team PPR room every ~5 minutes, so there is no scarcity
argument for squatting an empty one.

WHY THE HUMAN FLOOR IS 1 AND NOT 4. `teamsJoined` IS the human count --
verified against the league's own `?view=mTeam`, which showed exactly one
seat with an `owners` entry for a directory row reading `teamsJoined: 1`. So
a floor on it is a floor on people, and the obvious move is to set it high.
The lobby says otherwise. Measured at 09:45 local over 25 8-team PPR snake
rooms: eleven were at 8/8 and eleven more at 0/8, and of the TEN still
joinable, four had one person, three had two, two had three and NONE had
four. A floor of 4 would have joined nothing at all, all morning.

The distribution is the explanation, and it is not "there are no people".
Eleven full rooms is people drafting in numbers; they are simply unreachable,
because a room that has filled cannot be joined. Waiting for an open room to
ACCUMULATE humans and then taking a seat is self-defeating -- by the time it
has them it is full. So the floor is set where it still means something and
costs nothing: never take a room with nobody in it while a room with somebody
in it exists, and if nothing at all has a person, wait rather than fill an
empty room with our own bot. Everything past that is decided at fit time,
where a pick wrongly excluded can be included again by changing a query and a
room we declined to join is gone forever.
"""
import json
import time
from urllib.parse import quote

from pipeline.draft_socket import DRAFT_SECURITY_HEADERS
from pipeline.espn_identity import canonical_swid
from pipeline.espn_league import BASE

# The write host is a DIFFERENT hostname from `espn_league.BASE`'s read host,
# and that is not cosmetic: `lm-api-reads` answers the invite POST with an
# error. Both were probed; only this one accepts the join.
WRITES_BASE = "https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl"

# The lobby's own subType, by name. See the module docstring: the numeric id
# (4) does not work in this path position.
MOCK_SUBTYPE = "MOCKDRAFT_LOBBY"

# The room shape the corpus is being built out of. See the module docstring
# for why this is a policy rather than a default -- changing either number
# means the drafts recorded after the change are not comparable with the ones
# recorded before it.
FARM_LEAGUE_SIZE = 8
FARM_DRAFT_TYPE = "SNAKE"
FARM_RANK_TYPE = "PPR"
# ESPN stat id 53 is receptions. A room's `rankType` says "PPR" and its
# `scoringItemStatIds` says whether receptions are actually scored, and both
# are checked rather than either alone: they agreed in every one of the 234
# rows observed, which is exactly why disagreement would be worth hearing
# about, and a room whose name says PPR while its scoring omits receptions
# would fill the corpus with standard-scoring behaviour under a PPR label.
RECEPTION_STAT_ID = 53

# How far ahead of `draftDate` a room has to be to be worth joining, and how
# far ahead is too far.
#
# The floor: joining, fetching the room's settings and opening the socket is
# a handful of seconds, and a room whose clock starts while that is still in
# flight costs the first pick or two. `draftAvailableDate` is ~90s before
# `draftDate` (measured), so a 60s floor also lands inside the window where
# the room is actually enterable rather than merely listed.
#
# The ceiling: a seat held for half an hour before the draft starts is a seat
# a person cannot use, and it is dead time for a loop whose whole job is
# volume. At the measured supply (a new 8-team PPR room every ~5 minutes) a
# 15-minute window always holds several rooms, so nothing is given up.
MIN_LEAD_SECONDS = 60
MAX_LEAD_SECONDS = 900

# PRO and EXPERT rooms before BEGINNER, as the second sort key. Not a claim
# that beginners draft badly -- it is that the tool is used against the
# league the owner actually plays in, and a corpus weighted toward rooms of
# people who have drafted before is a closer population to that. Ranked
# lowest-first (see `_rank_key`), so smaller sorts earlier; an
# experienceType ESPN has never been observed sending falls between the two
# known tiers rather than being dropped.
EXPERIENCE_ORDER = {"EXPERT": 0, "PRO": 0, "BEGINNER": 2}
_UNKNOWN_EXPERIENCE = 1

# The fewest people a room must already hold before we will take a seat in
# it. See the module docstring for the measurement behind the value: 1 is the
# largest floor that does not starve the run, and it buys the one thing worth
# buying -- an empty room is never taken while an occupied one is on offer.
# Configurable because the right value is a function of the time of day (ESPN
# recommends 12:00, 17:00 and 20:00 local and windows the lobby 12:00-22:00),
# and an evening run can afford to be pickier than a 5am one.
MIN_TEAMS_JOINED = 1


def lobby_url(season: int) -> str:
    """The mock-draft directory for a season."""
    return f"{BASE}/seasons/{int(season)}/leaguedirectory/{MOCK_SUBTYPE}"


def invite_url(league_id, swid: str, season: int) -> str:
    """The invite POST that takes a seat in a room.

    `join=true` is required (see the module docstring). `swid` is
    percent-encoded because the cookie's own value carries braces --
    `{8491403C-...}` -- which are not legal in a query string unescaped;
    `quote` with an empty `safe` escapes them rather than passing them
    through, which is the difference between a 201 and an argument ESPN
    silently reads as something else.
    """
    return (f"{WRITES_BASE}/seasons/{int(season)}/segments/0/leagues/"
            f"{league_id}/invites?memberId={quote(str(swid), safe='')}"
            "&join=true")


def _as_json(body):
    """A parsed body, whether the injected transport handed back text or
    already-decoded JSON.

    Same seam and the same tolerance as `draft_socket.draft_security_token`:
    `http_fetch` returns `response.text`, a test hands back a list or dict
    directly, and neither caller should have to care which.
    """
    if isinstance(body, (list, dict)):
        return body
    return json.loads(body)


def list_mock_leagues(fetch, season: int) -> list:
    """Every room ESPN's mock lobby is currently advertising.

    `fetch(url) -> body` is injected -- the same testable seam
    `draft_socket.draft_security_token` uses, so nothing in this module
    imports httpx or touches the network in a test.

    Returns the rows verbatim. Filtering belongs to `pick_room`, which is
    where the policy lives; a caller that wants to count what the lobby is
    serving (how many 12-team rooms, how many already drafting) needs the
    unfiltered list and there is no reason to make it fetch twice.

    Raises rather than returning [] when the body is not a JSON array. An
    empty lobby and an expired `espn_s2` are different situations with the
    same shape from a caller's point of view, and only one of them is fixed
    by waiting -- so the one that is not must be loud.
    """
    payload = _as_json(fetch(lobby_url(season)))
    if not isinstance(payload, list):
        raise ValueError(
            f"ESPN's mock lobby returned {type(payload).__name__}, not a "
            "list of rooms -- the espn_s2/SWID cookies may have expired, or "
            f"ESPN has changed this endpoint (got {str(payload)[:200]!r})")
    return payload


def is_farmable(row: dict, now_ms: float,
                min_lead_seconds: float = MIN_LEAD_SECONDS,
                max_lead_seconds: float = MAX_LEAD_SECONDS,
                min_teams_joined: int = MIN_TEAMS_JOINED) -> bool:
    """Whether one directory row is a room this farm should play.

    Every clause is required; see the module docstring for the reasoning
    behind the shape filters and the two lead-time bounds. `.get` throughout
    rather than `[]`: a row missing a field it has always carried is a room
    we know less about than the policy requires, which is a reason to skip
    it, not to raise out of a filter.
    """
    if row.get("leagueSize") != FARM_LEAGUE_SIZE:
        return False
    if row.get("draftType") != FARM_DRAFT_TYPE:
        return False
    if row.get("rankType") != FARM_RANK_TYPE:
        return False
    if RECEPTION_STAT_ID not in (row.get("scoringItemStatIds") or []):
        return False
    if row.get("full"):
        return False
    if row.get("draftInProgress"):
        return False
    # SKIPPED, not merely ranked below. A room with nobody in it is eight
    # ESPN autodrafters plus us, and its 128 picks are ADP read back to us
    # under a label saying "mock draft" -- worse than no draft, because a
    # later reader cannot tell it from a room of people. A missing
    # `teamsJoined` reads as 0 for the same reason every other `.get` here
    # does: a row that will not say is a room we know less about than the
    # policy requires.
    if int(row.get("teamsJoined") or 0) < int(min_teams_joined):
        return False
    draft_date = row.get("draftDate")
    if not draft_date:
        return False
    lead = (float(draft_date) - float(now_ms)) / 1000.0
    return min_lead_seconds <= lead <= max_lead_seconds


def _rank_key(row: dict) -> tuple:
    """Sort key for a survivor, lowest first.

    `-teamsJoined` first, so the fullest room wins -- the single most
    important ordering key, and the reason is in the module docstring: a full
    room is humans drafting, an empty one is ESPN's autodraft engine
    reflecting ADP back at us. Then experience tier, then soonest start so
    the loop is idle for as little as possible.
    """
    return (
        -int(row.get("teamsJoined") or 0),
        EXPERIENCE_ORDER.get(row.get("experienceType"), _UNKNOWN_EXPERIENCE),
        float(row.get("draftDate") or 0),
    )


def rank_rooms(rows, now_ms: float | None = None, exclude=(),
               min_lead_seconds: float = MIN_LEAD_SECONDS,
               max_lead_seconds: float = MAX_LEAD_SECONDS,
               min_teams_joined: int = MIN_TEAMS_JOINED) -> list:
    """The farmable rooms among `rows`, best first.

    `exclude` is the set of league ids this process has already tried and
    should not try again -- a room whose join failed, or one already played.
    ESPN keeps a room in the directory after we have taken a seat in it, so
    without this the loop would rank its own room top (its `teamsJoined` just
    went up by one) and try to join it a second time.
    """
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    skip = {str(x) for x in exclude}
    keep = [r for r in rows
            if str(r.get("leagueId")) not in skip
            and is_farmable(r, now_ms, min_lead_seconds, max_lead_seconds,
                            min_teams_joined)]
    return sorted(keep, key=_rank_key)


def pick_room(rows, now_ms: float | None = None, exclude=(),
              min_lead_seconds: float = MIN_LEAD_SECONDS,
              max_lead_seconds: float = MAX_LEAD_SECONDS,
              min_teams_joined: int = MIN_TEAMS_JOINED) -> dict | None:
    """The single best room to join right now, or None if the lobby has
    nothing that fits. None is an ordinary outcome -- the lobby serves rooms
    in batches, so a poll landing between batches sees no room inside the
    lead-time window -- and the caller's answer is to wait and poll again,
    not to relax the filter.
    """
    ranked = rank_rooms(rows, now_ms, exclude, min_lead_seconds,
                        max_lead_seconds, min_teams_joined)
    return ranked[0] if ranked else None


def lobby_report(rows, now_ms: float | None = None, exclude=(),
                 min_lead_seconds: float = MIN_LEAD_SECONDS,
                 max_lead_seconds: float = MAX_LEAD_SECONDS) -> dict:
    """What the lobby is offering, ignoring the human floor. For the LOG.

    `pick_room` returning None is two very different situations wearing the
    same face: a poll that landed between batches and saw no 8-team PPR room
    at all, or a lobby full of them with nobody sitting in any. An unattended
    run that only ever prints "nothing fits" cannot tell whether the floor is
    starving it, and that is precisely the number somebody reading the log in
    the morning needs.

    So this counts the survivors of every filter EXCEPT the human floor, and
    reports the best `teamsJoined` among them. `best` is None when there were
    no shape-and-timing survivors to have a best of -- which is the "between
    batches" case, said in the one way that distinguishes it.
    """
    open_rooms = rank_rooms(rows, now_ms, exclude, min_lead_seconds,
                            max_lead_seconds, min_teams_joined=0)
    joined = [int(r.get("teamsJoined") or 0) for r in open_rooms]
    return {"rows": len(rows), "open": len(open_rooms),
            "best": max(joined) if joined else None,
            "size": FARM_LEAGUE_SIZE}


def room_url(league_id, season: int) -> str:
    """One room's seats, settings and draft state.

    PROBED LIVE (2026-08-24) and public: this answers 200 with no cookies at
    all, the same as the directory. So a reader can be shown who is in a room
    and when it starts before they have any session -- and nobody's ESPN login
    is sent upstream to find that out.

    Three views, because the waiting room needs all three and one request is
    cheaper than three: `mTeam` for the seats and who owns them, `mSettings`
    for the draft's date, type, clock and pick order, `mDraftDetail` for
    whether picking has started.
    """
    return (f"{BASE}/seasons/{int(season)}/segments/0/leagues/{league_id}"
            "?view=mTeam&view=mSettings&view=mDraftDetail")


def seats(payload: dict, swid: str | None = None) -> list:
    """Every seat in a room, in draft order, with its owner resolved.

    A seat with an empty `owners` is OPEN -- verified against the directory's
    own `teamsJoined`, which is what the farm's human floor is built on (see
    this module's docstring). `mine` compares against the caller's SWID
    through `canonical_swid`, the same normalizer every other comparison in
    this codebase uses, so a stored session and a live response cannot
    disagree about whether the braces are part of the value.

    `slot` is the seat's position in `pickOrder`, which is the draft order --
    not the team id, which only happens to match it in a fresh mock room. The
    two are different fields and a room that reorders is a room where every
    pick number this page prints would be wrong.
    """
    order = ((payload.get("settings") or {}).get("draftSettings") or {}).get("pickOrder") or []
    slot_of = {int(team): i + 1 for i, team in enumerate(order)}
    mine = canonical_swid(swid) if swid else None
    out = []
    for team in payload.get("teams") or []:
        owners = team.get("owners") or []
        team_id = int(team.get("id"))
        out.append({
            "team_id": str(team_id),
            # Ordered by this, not by id: see above.
            "slot": slot_of.get(team_id, team_id),
            "name": team.get("name") or f"Team {team_id}",
            "taken": bool(owners),
            "mine": bool(mine and any(canonical_swid(o) == mine for o in owners)),
        })
    out.sort(key=lambda seat: seat["slot"])
    return out


def join(post, league_id, swid: str, season: int, team_id: int | None = None) -> int:
    """Take a seat in a room. Returns the team id ESPN assigned.

    `post(url, payload) -> body` is injected, mirroring the `fetch` seam
    above. The payload is a JSON array holding one object with `teamId: -1`
    ("any open seat") -- both the array wrapper and the sentinel were probed
    live; neither is a guess.

    `team_id` asks for ONE PARTICULAR SEAT; None sends the `-1` sentinel and
    takes whatever is open, which is what the farm wants and what a reader who
    does not care should send. That ESPN honours a specific id was probed live
    (2026-08-24): a room with seats 2, 3 and 4 open was asked for 4 -- not the
    first one it would have handed out -- and answered `{"teamId": 4}`. A seat
    somebody else took between the read and the POST is refused by ESPN rather
    than silently swapped, which is the behaviour the waiting room needs.

    The 201 response carries the assignment as `[{"isDeleted": false,
    "teamId": 7}]`, and that value is READ, never assumed. A wrong team id
    mints a socket URL for somebody else's seat, and the failure would show
    up as a draft in which our picks never land rather than as an error here.
    A response that does not carry one raises, for the same reason.
    """
    body = _as_json(post(invite_url(league_id, swid, season),
                         [{"teamId": -1 if team_id is None else int(team_id)}]))
    rows = body if isinstance(body, list) else [body]
    for row in rows:
        if isinstance(row, dict) and row.get("teamId") is not None:
            team_id = int(row["teamId"])
            if team_id > 0:
                return team_id
    raise ValueError(
        f"ESPN accepted the join for league {league_id} but did not say "
        f"which seat: {str(body)[:200]!r}")


def http_poster(cookies: dict):
    """A `post(url, payload) -> body_text` callable hitting ESPN over httpx.

    The counterpart to `draft_socket.http_fetch`, and deliberately the same
    shape: headers are that module's already-verified `DRAFT_SECURITY_HEADERS`
    plus the `content-type: application/json` this endpoint needs, cookies are
    the saved `espn_s2`/`SWID`, and the return value is the raw text so the
    parsing stays in `join` where a test can reach it.

    httpx is already a project dependency; nothing new is required to make
    one POST with two cookies.
    """
    import httpx

    headers = dict(DRAFT_SECURITY_HEADERS)
    headers["content-type"] = "application/json"

    def post(url: str, payload):
        response = httpx.post(url, cookies=cookies, headers=headers,
                              json=payload, timeout=15.0)
        response.raise_for_status()
        return response.text
    return post
