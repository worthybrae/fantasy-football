"""fetch_team_slots: ESPN's public league view -> slot -> team name.

Driven entirely through the injected `fetch` seam, so no test here touches
the network -- the same discipline pipeline/espn_league.py's parser tests and
pipeline/draft_socket.py's token test already hold.
"""
import json

from pipeline.espn_teams import fetch_team_owners, fetch_team_slots


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


def test_fetch_league_settings_parses_the_roster():
    """The league's real roster (lineupSlotCounts) becomes the from_espn shape,
    so the session drafts for the actual roster -- 1QB/2RB/2WR/1TE/1DST/1K +
    2 FLEX + 5 bench = 15 rounds. Best-effort: any failure or an absent lineup
    config returns None so the caller falls back to the database's settings."""
    import json
    from pipeline.espn_teams import fetch_league_settings
    from scoring import league as lm

    body = json.dumps({
        "teams": [{"id": i} for i in range(1, 9)],
        "settings": {
            "rosterSettings": {"lineupSlotCounts": {
                "0": 1, "2": 2, "4": 2, "6": 1, "16": 1, "17": 1,
                "20": 5, "23": 2}},
            "scoringSettings": {"scoringItems": [{"statId": 53, "points": 1.0}]},
            "draftSettings": {"type": "SNAKE", "pickOrder": [3, 1, 2]},
        },
    })
    raw = fetch_league_settings(lambda url: body, "1", 2026)
    assert raw["teams"] == 8
    assert raw["draft_type"] == "SNAKE"
    s = lm.from_espn(raw)
    assert s.rounds == 15
    assert s.starters == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "DST": 1, "K": 1}
    assert (s.flex_slots, s.bench) == (2, 5)

    # Failure paths -> None (caller falls back), never a raise.
    assert fetch_league_settings(
        lambda url: (_ for _ in ()).throw(ValueError()), "1", 2026) is None
    assert fetch_league_settings(lambda url: "{}", "1", 2026) is None


# ---------------------------------------------------------------------------
# fetch_team_owners: which seats hold a person. Read while the room is LIVE --
# a mock league 404s the moment its draft ends, so there is no second chance.
# ---------------------------------------------------------------------------


def _mock_room_payload():
    """A live 8-team mock room's `?view=mTeam` body, in the shape verified
    against room 451008377: one seat with an owner and seven of ESPN's own
    computer entries, whose `abbrev` is `TM<n>` and whose `owners` is []."""
    teams = [{"id": n, "abbrev": f"TM{n}", "owners": []} for n in range(1, 9)]
    teams[3]["owners"] = ["{8491403C-A53F-4257-8D52-F8AE32CED897}"]
    return {"teams": teams}


def test_a_seat_with_an_owner_is_a_person_and_an_empty_one_is_not():
    owners = fetch_team_owners(lambda url: json.dumps(_mock_room_payload()),
                               league_id="451008377", season=2026)
    assert owners == {1: False, 2: False, 3: False, 4: True,
                      5: False, 6: False, 7: False, 8: False}
    # Which is the directory's own `teamsJoined` for that room, independently
    # measured: 1.
    assert sum(owners.values()) == 1


def test_the_owner_view_is_its_own_url():
    """Not a slice of the combined mTeam+mSettings view. That one is fetched
    at join time, minutes before the room opens; this census is taken as the
    draft starts, and sharing a GET would mean sharing the wrong instant."""
    seen = {}

    def fetch(url):
        seen["url"] = url
        return _mock_room_payload()

    fetch_team_owners(fetch, league_id="451008377", season=2026)
    assert seen["url"].endswith(
        "/seasons/2026/segments/0/leagues/451008377?view=mTeam")


def test_the_owner_census_is_empty_rather_than_wrong_when_espn_will_not_say():
    """Best-effort like the rest of this module. {} means "we do not know",
    which leaves every seat's autodraft state NULL downstream -- the honest
    answer, and the one the corpus stored before this existed."""
    def raises(url):
        raise RuntimeError("404 -- the draft is over")

    assert fetch_team_owners(raises, league_id="9", season=2026) == {}
    assert fetch_team_owners(lambda url: "not json", "9", 2026) == {}
    assert fetch_team_owners(lambda url: {}, "9", 2026) == {}
    # A team row with no id is skipped rather than keyed on None.
    assert fetch_team_owners(lambda url: {"teams": [{"owners": ["x"]}]},
                             "9", 2026) == {}
