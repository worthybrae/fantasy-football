"""fetch_team_slots: ESPN's public league view -> slot -> team name.

Driven entirely through the injected `fetch` seam, so no test here touches
the network -- the same discipline pipeline/espn_league.py's parser tests and
pipeline/draft_socket.py's token test already hold.
"""
import json

from pipeline.espn_teams import fetch_team_slots


def _league_payload():
    """A realistic slice of ESPN's mTeam+mSettings view: teams keyed by id and
    a draftSettings.pickOrder giving the slot order. pickOrder [7,5,2,8] means
    slot 1 is team 7, slot 2 is team 5, and so on -- deliberately NOT ascending
    so a bug that used the team id (or the list order) as the slot would show.
    """
    return {
        "teams": [
            {"id": 2, "name": "The Hog Crankers", "abbrev": "HOG"},
            {"id": 5, "name": "", "location": "Motor City", "nickname": "Maulers"},
            {"id": 7, "name": "", "location": "", "nickname": "", "abbrev": "ZZZ"},
            {"id": 8, "name": "Sunday Scaries", "abbrev": "SCR"},
        ],
        "settings": {"draftSettings": {"pickOrder": [7, 5, 2, 8]}},
    }


def test_fetch_team_slots_maps_pick_order_to_names():
    payload = _league_payload()
    fetch = lambda url: json.dumps(payload)      # noqa: E731 -- one-line fake
    slots = fetch_team_slots(fetch, league_id="99", season=2026)
    assert slots == {
        1: "ZZZ",                # team 7: name/location/nickname all empty -> abbrev
        2: "Motor City Maulers",  # team 5: location + nickname
        3: "The Hog Crankers",    # team 2: full name wins
        4: "Sunday Scaries",      # team 8
    }


def test_fetch_team_slots_accepts_an_already_parsed_dict_body():
    """A fetch may hand back parsed JSON rather than raw text -- both readings
    are accepted, the same tolerance draft_security_token allows."""
    slots = fetch_team_slots(lambda url: _league_payload(),
                             league_id="99", season=2026)
    assert slots[3] == "The Hog Crankers"


def test_fetch_team_slots_hits_the_public_view_url():
    """The one GET must target the season/league mTeam+mSettings view."""
    seen = {}

    def fetch(url):
        seen["url"] = url
        return json.dumps(_league_payload())

    fetch_team_slots(fetch, league_id="123456", season=2026)
    assert "/seasons/2026/segments/0/leagues/123456" in seen["url"]
    assert "view=mTeam" in seen["url"] and "view=mSettings" in seen["url"]


def test_fetch_team_slots_returns_empty_when_the_fetch_raises():
    """Best-effort: a network failure must never raise into connect."""
    def fetch(url):
        raise RuntimeError("ESPN unreachable")

    assert fetch_team_slots(fetch, league_id="99", season=2026) == {}


def test_fetch_team_slots_returns_empty_without_a_pick_order():
    """A league whose settings do not (yet) expose the pick order yields no
    slot map -- the board falls back to placeholders rather than guessing."""
    payload = {"teams": [{"id": 1, "name": "Lonely"}], "settings": {}}
    assert fetch_team_slots(lambda url: json.dumps(payload),
                            league_id="99", season=2026) == {}


def test_fetch_team_slots_returns_empty_on_unparseable_body():
    """An HTML error page (or any non-JSON body) is a failure, handled the
    same best-effort way as a raised fetch."""
    assert fetch_team_slots(lambda url: "<html>nope</html>",
                            league_id="99", season=2026) == {}


def test_fetch_team_slots_skips_a_pick_order_id_absent_from_teams():
    """If pickOrder names a team id `teams` does not carry, that slot is left
    out rather than mapped to a made-up name -- the board endpoint's own
    "Team {slot}" fallback covers the gap."""
    payload = {
        "teams": [{"id": 1, "name": "Only One"}],
        "settings": {"draftSettings": {"pickOrder": [1, 2]}},
    }
    slots = fetch_team_slots(lambda url: json.dumps(payload),
                             league_id="99", season=2026)
    assert slots == {1: "Only One"}
