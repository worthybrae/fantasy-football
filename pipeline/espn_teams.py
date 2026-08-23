"""Real ESPN team names, mapped to draft slots.

The board (scoring/board.py) is one axis of the live draft grid -- the
players. The other axis is who owns each column, and that comes from here.
ESPN's public league view carries both the team names and the draft-slot
order, so a single unauthenticated GET names every column of the board
before a pick lands:

    GET {BASE}/seasons/{season}/segments/0/leagues/{leagueId}
        ?view=mTeam&view=mSettings

`resp["settings"]["draftSettings"]["pickOrder"]` is a list of team ids in
draft-slot order -- pickOrder[0] is slot 1's team id, pickOrder[1] is slot
2's, and so on -- and `resp["teams"]` maps each id to a display name. Joined,
they give slot -> team name.

Pure translation over the JSON, driven by an injected `fetch` the same way
`pipeline.draft_socket.draft_security_token` and
`pipeline.espn_league.import_seasons` already are -- so it is testable against
a literal fixture with no network. Unlike those, this one is best-effort by
design: the board is perfectly usable with "Team 3" placeholders, so any
failure at all (network down, missing pickOrder, a body that will not parse)
returns {} rather than raising into a /api/live/connect that would otherwise
have succeeded.
"""
import json

from pipeline.espn_league import BASE

# The same header the draft-socket fetch sends, and for the same reason (see
# pipeline.draft_socket.DRAFT_SECURITY_HEADERS): without
# `x-fantasy-source: kona` ESPN has been observed to silently degrade a
# response rather than reject it. NO cookies here -- this endpoint answers 200
# unauthenticated, which is the whole reason team names can be shown for a
# stranger's league with no saved login on this machine.
TEAM_VIEW_HEADERS = {
    "accept": "application/json",
    "x-fantasy-source": "kona",
    "origin": "https://fantasy.espn.com",
    "referer": "https://fantasy.espn.com/",
}


def _team_view_url(league_id, season) -> str:
    return (f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}"
            f"?view=mTeam&view=mSettings")


def _team_name(team: dict) -> str:
    """ESPN's own display name for a team, however it stored it.

    `name` is the full name a league sets ("The Hog Crankers"); some teams
    leave it empty and carry a location+nickname pair instead, and a bare team
    has only its `abbrev`. Tried in that order so every team gets the best
    label it actually has, with a "Team {id}" backstop so a slot never maps to
    an empty string.
    """
    name = (team.get("name") or "").strip()
    if name:
        return name
    pair = f"{team.get('location') or ''} {team.get('nickname') or ''}".strip()
    if pair:
        return pair
    return (team.get("abbrev") or "").strip() or f"Team {team.get('id')}"


def fetch_team_slots(fetch, league_id, season) -> dict:
    """slot (1-based) -> team display name, from ESPN's public league view.

    `fetch(url) -> body` is injected (see this module's docstring), so tests
    drive it from a fixture with no network. The body is JSON -- accepted
    either as the raw text/bytes a real HTTP client returns or as an
    already-parsed dict, the same both-readings tolerance
    `draft_socket.draft_security_token` allows for its integer body.

    Best-effort: returns {} on ANY failure -- a fetch that raises, a body that
    is not JSON, an absent `pickOrder` or `teams` -- so a connect that could
    not reach ESPN (or hit a league whose settings do not expose the pick
    order yet) still launches, with the board endpoint falling back to
    "Team {slot}" placeholders per column.
    """
    try:
        body = fetch(_team_view_url(league_id, season))
        payload = body if isinstance(body, dict) else json.loads(body)
        teams = payload.get("teams") or []
        pick_order = (((payload.get("settings") or {})
                       .get("draftSettings") or {}).get("pickOrder")) or []
        if not teams or not pick_order:
            return {}
        names = {t.get("id"): _team_name(t) for t in teams}
        # pickOrder is slot order: its k-th entry (1-based) is slot k's team
        # id. A team id in pickOrder that is missing from `teams` is skipped
        # rather than mapped to a placeholder here -- the board endpoint's own
        # "Team {slot}" fallback covers the gap, and inventing a name here
        # would hide that ESPN's two lists disagreed.
        return {slot: names[team_id]
                for slot, team_id in enumerate(pick_order, start=1)
                if team_id in names}
    except Exception:      # noqa: BLE001 -- best-effort by design; see docstring
        return {}


def fetch_league_settings(fetch, league_id, season) -> dict | None:
    """The league's real roster + scoring config, shaped for
    `scoring.league.from_espn`, or None.

    Same public mTeam&mSettings view fetch_team_slots already reads, so the two
    can share one GET. `settings.rosterSettings.lineupSlotCounts` is the roster
    (how many of each lineup slot, incl. FLEX and bench) -- the thing that
    decides how many rounds the draft is and what each team is drafting FOR;
    `settings.scoringSettings.scoringItems` is the scoring; `draftSettings`
    carries the snake type and pick order.

    Best-effort, like the rest of this module: returns None on any failure or
    an absent lineup config, so a connect that could not reach ESPN (or a mock
    whose settings are not published) falls back to the caller's default
    settings rather than failing. None, not {} -- the caller distinguishes
    "use the default" from a real (possibly empty-scoring) config.
    """
    try:
        body = fetch(_team_view_url(league_id, season))
        payload = body if isinstance(body, dict) else json.loads(body)
        settings = payload.get("settings") or {}
        lineup_slots = ((settings.get("rosterSettings") or {})
                        .get("lineupSlotCounts")) or {}
        n_teams = len(payload.get("teams") or []) or settings.get("size")
        if not lineup_slots or not n_teams:
            return None
        draft = settings.get("draftSettings") or {}
        return {
            "season": int(season),
            "teams": int(n_teams),
            "lineup_slots": {str(k): int(v) for k, v in lineup_slots.items()},
            "scoring_items": (settings.get("scoringSettings") or {})
                             .get("scoringItems") or [],
            "draft_type": draft.get("type") or "SNAKE",
            "pick_order": list(draft.get("pickOrder") or ()),
        }
    except Exception:      # noqa: BLE001 -- best-effort; caller falls back
        return None


def http_fetch():
    """A `fetch(url) -> body_text` callable hitting ESPN directly over httpx.

    Mirrors `pipeline.draft_socket.http_fetch`, but with NO cookies: this
    endpoint answers 200 unauthenticated (verified). httpx is already a
    project dependency, so nothing new is needed to make one GET request.
    """
    import httpx

    def fetch(url: str):
        response = httpx.get(url, headers=TEAM_VIEW_HEADERS, timeout=10.0)
        response.raise_for_status()
        return response.text
    return fetch


# The owner view, deliberately its own URL rather than a slice of
# `_team_view_url`'s combined mTeam+mSettings body. The combined view is
# fetched once per room by `fetch_league_settings` at join time, which is
# minutes before the draft opens; the owner census below is taken at a
# different moment on purpose (see `fetch_team_owners`), so sharing one GET
# would mean sharing the wrong instant.
def _owner_view_url(league_id, season) -> str:
    return (f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}"
            "?view=mTeam")


def fetch_team_owners(fetch, league_id, season) -> dict:
    """ESPN team id -> whether a real person is sitting in that seat.

    THE ONE FACT ESPN WILL NOT SELL YOU TWICE. A mock league 404s the moment
    its draft ends -- verified -- so this is readable only WHILE the room is
    live, and every caller has to take it then and carry it rather than
    reaching back for it afterwards.

    A team's `owners` array is the whole signal: non-empty means a member
    GUID is attached to the seat, empty means the seat is one of the
    computer entries ESPN pads a thin room out with (`abbrev` like `TM1`).
    Confirmed against live room 451008377: 8 teams, exactly one with an
    owner, matching the lobby directory's own `teamsJoined: 1`.

    `fetch(url) -> body` is injected, the same seam as the rest of this
    module, and the caller must pass an AUTHENTICATED fetch: unlike the
    public team-name view, a mock room's membership is not served to a
    stranger.

    Best-effort like its neighbours: {} on any failure at all. An empty map
    is "we do not know", which downstream leaves every seat's initial
    autodraft state NULL -- the honest answer, and the one the corpus stored
    before this existed.
    """
    try:
        body = fetch(_owner_view_url(league_id, season))
        payload = body if isinstance(body, dict) else json.loads(body)
        teams = payload.get("teams") or []
        owners = {}
        for team in teams:
            team_id = team.get("id")
            if team_id is None:
                continue
            owners[int(team_id)] = bool(team.get("owners") or [])
        return owners
    except Exception:      # noqa: BLE001 -- best-effort; see docstring
        return {}
