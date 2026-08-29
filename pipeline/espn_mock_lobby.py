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

ROOM SELECTION IS A POLICY, NOT A PREFERENCE, and the policy is now a LIST
of shapes rather than one. `pick_room` filters to snake rooms whose
`(leagueSize, format)` is one of `FARM_SHAPES` -- 8-, 10- and 12-team PPR
plus 10- and 12-team standard by default, `FARM_SHAPES` in the environment
to change it.

The original policy was 8-team PPR and nothing else (the owner's
instruction, 2026-08-23), for a reason that has not gone away: team count
changes who is on the clock at pick k and scoring changes who is worth
taking there, so a corpus that averages over shapes is answering a question
nobody asked. What changed is that the corpus is no longer one number per
player -- `scoring.availability` now counts each shape separately and only
uses a shape's own counts once it holds MIN_SHAPE_DRAFTS of them (60),
falling back to the pooled counts until then. So a 12-team room is no longer
contamination; it is the beginning of the table a 12-team league reads.

Which shape gets joined is decided by what the corpus is SHORTEST of, not by
preference order: `rank_rooms` takes the per-shape draft counts, puts the
least-recorded shape that has a joinable room first, and ranks within that
shape exactly as it always did (fullest room, then experience, then soonest
start). Ties go to the order the shapes are listed in, so two farm processes
polling the same lobby still agree.

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
import os
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

FARM_DRAFT_TYPE = "SNAKE"

# ESPN stat id 53 is receptions. A room's `rankType` NAMES a format and its
# `scoringItemStatIds` says whether receptions are actually scored, and both
# are read rather than either alone: they agreed in every one of the 234 rows
# observed, which is exactly why disagreement would be worth hearing about. A
# room whose name says PPR while its scoring omits receptions would fill the
# corpus with standard-scoring behaviour under a PPR label -- and now that
# standard rooms are farmed too, the mirror image (a STANDARD room that does
# score receptions) would do the same thing in the other direction. Either
# way the row is skipped rather than guessed at.
RECEPTION_STAT_ID = 53

# ESPN's word for a format, in the directory row, mapped to this project's
# (`scoring.league.scoring_format`). Only PPR and STANDARD have been observed
# in the wild; the half-PPR spellings are here because the directory has a
# field for the format and this is the list of what it could say, and a room
# ESPN labels in a way we do not recognise is skipped rather than assumed.
#
# HALF CANNOT BE CONFIRMED THE WAY THE OTHER TWO CAN: the directory publishes
# stat IDS, not point values, so "receptions are scored" is all it can tell
# us and 0.5 looks exactly like 1.0 from here. The room's own settings are
# read before a single pick is recorded (`mock_farm.play_draft`), and THAT is
# what the draft is filed under -- so a room mislabelled in the directory
# costs one join, not a wrong row in the corpus.
_ROW_FORMATS = {"PPR": "ppr", "HALF_PPR": "half", "HALF": "half",
                "STANDARD": "std", "STD": "std", "NON_PPR": "std"}

# The shapes the farm is building the corpus out of, as `(teams, format)`.
# The default is the owner's list (spec 2026-08-29 section 2): the three PPR
# sizes people actually play, plus standard at 10 and 12 -- 8-team standard
# is left out because it is the rarest room in the lobby and the corpus
# already holds 854 8-team PPR drafts.
#
# Read from the environment ONCE, at import, and parsed strictly: a typo in
# `FARM_SHAPES` should stop the process before it takes a seat in anybody's
# room, not silently narrow the rotation to whatever parsed.
FARM_SHAPES_ENV = "FARM_SHAPES"
DEFAULT_FARM_SHAPES = "8:ppr,10:ppr,12:ppr,10:std,12:std"


def parse_shapes(text: str) -> tuple:
    """`"8:ppr,10:std"` -> `((8, "ppr"), (10, "std"))`, in the order given.

    The order is kept because it is the tie-break in `rank_rooms`: two shapes
    the corpus holds equally little of are joined in the order they were
    listed, which keeps two farm processes reading the same lobby in
    agreement about what to take.

    Raises on anything it cannot read -- an unknown format word, a
    non-numeric size, an entry with no colon in it. A farm that quietly
    dropped the half of `FARM_SHAPES` it could not parse would spend the
    night recording a corpus nobody asked for.
    """
    shapes = []
    for entry in str(text or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        size, _, fmt = entry.partition(":")
        fmt = fmt.strip().lower()
        if fmt not in ("ppr", "half", "std"):
            raise ValueError(
                f"{FARM_SHAPES_ENV}: {entry!r} does not name a scoring "
                "format -- expected teams:ppr, teams:half or teams:std")
        try:
            teams = int(size.strip())
        except ValueError:
            raise ValueError(
                f"{FARM_SHAPES_ENV}: {entry!r} does not start with a team "
                "count") from None
        shape = (teams, fmt)
        if shape not in shapes:
            shapes.append(shape)
    if not shapes:
        raise ValueError(f"{FARM_SHAPES_ENV} is empty -- the farm would have "
                         "no room it is allowed to join")
    return tuple(shapes)


FARM_SHAPES = parse_shapes(os.environ.get(FARM_SHAPES_ENV)
                           or DEFAULT_FARM_SHAPES)


def row_format(row: dict) -> str | None:
    """The scoring format of one directory row, or None if it will not say.

    Two independent signals, both required to agree -- see
    `RECEPTION_STAT_ID` above. None means "this room is not one of the
    formats we can name", which every caller treats as a room to skip: a
    corpus row filed under a guessed format is worse than a room not played.
    """
    fmt = _ROW_FORMATS.get(str(row.get("rankType") or "").strip().upper())
    if fmt is None:
        return None
    scored = RECEPTION_STAT_ID in (row.get("scoringItemStatIds") or [])
    if scored != (fmt in ("ppr", "half")):
        return None
    return fmt


def shape_of(row: dict) -> tuple | None:
    """`(leagueSize, format)` for a directory row, or None when either half
    is missing or unreadable."""
    size = row.get("leagueSize")
    fmt = row_format(row)
    if size is None or fmt is None:
        return None
    try:
        return (int(size), fmt)
    except (TypeError, ValueError):
        return None


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
                min_teams_joined: int = MIN_TEAMS_JOINED,
                shapes=None) -> bool:
    """Whether one directory row is a room this farm should play.

    Every clause is required; see the module docstring for the reasoning
    behind the shape filter and the two lead-time bounds. `.get` throughout
    rather than `[]`: a row missing a field it has always carried is a room
    we know less about than the policy requires, which is a reason to skip
    it, not to raise out of a filter.

    `shapes` is the allowed `(teams, format)` list, `FARM_SHAPES` by default.
    A room of ANY other shape is skipped exactly as a full one is -- which
    shape to prefer among the allowed ones is a ranking question, not this
    one (see `rank_rooms`).
    """
    if shape_of(row) not in (FARM_SHAPES if shapes is None else tuple(shapes)):
        return False
    if row.get("draftType") != FARM_DRAFT_TYPE:
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
               min_teams_joined: int = MIN_TEAMS_JOINED,
               shapes=None, counts=None) -> list:
    """The farmable rooms among `rows`, best first.

    `exclude` is the set of league ids this process has already tried and
    should not try again -- a room whose join failed, or one already played.
    ESPN keeps a room in the directory after we have taken a seat in it, so
    without this the loop would rank its own room top (its `teamsJoined` just
    went up by one) and try to join it a second time.

    `counts` is `{(teams, format): drafts already recorded}` -- what
    `pipeline.draft_log.shape_counts` reads out of the corpus. WITH IT the
    ranking is a rotation: the shape the corpus is shortest of comes first,
    and the fullest-room ordering decides only within a shape. WITHOUT it
    (None) every room is ranked purely on how full it is, which is what a
    caller asking "what could I join right now" wants and what every reader
    of this function meant before shapes existed.

    The whole list is returned in that order rather than the best of the best
    shape, so a caller whose first choice is claimed by another process (see
    `mock_farm.farm`) falls to the next room down and only then to the next
    shape -- one poll, not two.
    """
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    skip = {str(x) for x in exclude}
    allowed = FARM_SHAPES if shapes is None else tuple(shapes)
    keep = [r for r in rows
            if str(r.get("leagueId")) not in skip
            and is_farmable(r, now_ms, min_lead_seconds, max_lead_seconds,
                            min_teams_joined, allowed)]
    if counts is None:
        return sorted(keep, key=_rank_key)
    return sorted(keep, key=lambda row: (_shape_key(shape_of(row), counts,
                                                    allowed), _rank_key(row)))


def _shape_key(shape, counts, allowed) -> tuple:
    """Sort key for a shape, lowest first: fewest recorded drafts, then the
    order `FARM_SHAPES` lists it in.

    The list order is a real tie-break, not decoration. Every shape starts at
    zero recorded drafts, and two farm processes that broke that tie on
    anything unstable (dict order, room id) would rank differently, both join,
    and put two bot seats in one room -- the exact failure `farm_claims`
    exists to prevent, arrived at from the other side.
    """
    return (int(counts.get(shape, 0)),
            allowed.index(shape) if shape in allowed else len(allowed))


def pick_room(rows, now_ms: float | None = None, exclude=(),
              min_lead_seconds: float = MIN_LEAD_SECONDS,
              max_lead_seconds: float = MAX_LEAD_SECONDS,
              min_teams_joined: int = MIN_TEAMS_JOINED,
              shapes=None, counts=None) -> dict | None:
    """The single best room to join right now, or None if the lobby has
    nothing that fits. None is an ordinary outcome -- the lobby serves rooms
    in batches, so a poll landing between batches sees no room inside the
    lead-time window -- and the caller's answer is to wait and poll again,
    not to relax the filter.
    """
    ranked = rank_rooms(rows, now_ms, exclude, min_lead_seconds,
                        max_lead_seconds, min_teams_joined, shapes, counts)
    return ranked[0] if ranked else None


def lobby_report(rows, now_ms: float | None = None, exclude=(),
                 min_lead_seconds: float = MIN_LEAD_SECONDS,
                 max_lead_seconds: float = MAX_LEAD_SECONDS,
                 shapes=None) -> dict:
    """What the lobby is offering, ignoring the human floor. For the LOG.

    `pick_room` returning None is two very different situations wearing the
    same face: a poll that landed between batches and saw no room of any
    farmed shape at all, or a lobby full of them with nobody sitting in any.
    An unattended run that only ever prints "nothing fits" cannot tell
    whether the floor is starving it, and that is precisely the number
    somebody reading the log in the morning needs.

    So this counts the survivors of every filter EXCEPT the human floor, and
    reports the best `teamsJoined` among them. `best` is None when there were
    no shape-and-timing survivors to have a best of -- which is the "between
    batches" case, said in the one way that distinguishes it. `size` is the
    seat count of THAT room rather than a constant, now that the rooms in
    this list are not all the same size, so "the fullest holds 3/12" names a
    real room.

    `by_shape` is how many joinable rooms each allowed shape has, in the
    order the shapes are listed -- the other half of the morning's question,
    which is no longer "is anybody in there" but "is the lobby even serving
    the shape the corpus is short of".
    """
    allowed = FARM_SHAPES if shapes is None else tuple(shapes)
    open_rooms = rank_rooms(rows, now_ms, exclude, min_lead_seconds,
                            max_lead_seconds, min_teams_joined=0,
                            shapes=allowed)
    joined = [int(r.get("teamsJoined") or 0) for r in open_rooms]
    fullest = max(open_rooms, key=lambda r: int(r.get("teamsJoined") or 0),
                  default=None)
    by_shape = {shape: 0 for shape in allowed}
    for room in open_rooms:
        shape = shape_of(room)
        if shape in by_shape:
            by_shape[shape] += 1
    return {"rows": len(rows), "open": len(open_rooms),
            "best": max(joined) if joined else None,
            "size": None if fullest is None else int(fullest["leagueSize"]),
            "by_shape": by_shape}


def shape_line(counts, report: dict | None = None, shapes=None) -> str:
    """One line naming every farmed shape, what the corpus holds of it, and
    how many rooms the lobby is offering. For the LOG, once per pass.

    `8:ppr 854 recorded/2 open  10:ppr 0/1  ...` -- the whole rotation on one
    line, in the order the tie-break uses, so a morning reader can see both
    why the farm chose what it chose and whether the shape it wants exists in
    the lobby at all.
    """
    allowed = FARM_SHAPES if shapes is None else tuple(shapes)
    by_shape = (report or {}).get("by_shape") or {}
    parts = []
    for shape in allowed:
        teams, fmt = shape
        parts.append(f"{teams}:{fmt} {int(counts.get(shape, 0))} recorded/"
                     f"{int(by_shape.get(shape, 0))} open")
    return "shapes -- " + ", ".join(parts)


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
