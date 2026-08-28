"""The landing page's live room: what wakes it, and the clock it draws.

These cover the two things that make the page look live rather than
periodic -- the identity that decides when a reader is woken and when a
cached board dies, and the pick clock the countdown is drawn from -- plus
the rule that neither is ever guessed at.
"""
import json
import os
import time
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.demo as demo


@pytest.fixture
def farm(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "FARM_DIR", str(tmp_path / "live"))
    monkeypatch.setattr(demo, "REPLAY_DIR", str(tmp_path / "archive"))
    (tmp_path / "live").mkdir()
    demo._CACHE.clear()
    demo._REPLAY.clear()
    demo._CLOCK.clear()
    demo._SHOWING.clear()
    yield tmp_path / "live"
    demo._CACHE.clear()
    demo._REPLAY.clear()
    demo._CLOCK.clear()
    demo._SHOWING.clear()


def _room(league_id="777", picks=3, teams=8, rounds=16, clock=90.0,
          written_at=None):
    return {
        "league_id": league_id,
        "season": 2026,
        "teams": teams,
        "rounds": rounds,
        "my_slot": 1,
        "started_at": "2026-08-24T18:00:00",
        "human_seats": 5,
        "seats": {},
        "picks": [{"pick_no": n + 1, "slot": (n % teams) + 1,
                   "player_id": f"00-000{n}", "autodrafted": False,
                   "clock_seconds": clock, "seconds_to_pick": 12.0}
                  for n in range(picks)],
        "written_at": time.time() if written_at is None else written_at,
    }


def _write(directory, room):
    path = directory / f"{room['league_id']}.json"
    path.write_text(json.dumps(room))
    return path


def test_identity_moves_with_the_pick(farm):
    _write(farm, _room(picks=3))
    before = demo._identity(time.time())
    _write(farm, _room(picks=4))
    assert demo._identity(time.time()) != before


def test_identity_ignores_a_room_the_page_is_not_showing(farm):
    """The farm sits in several rooms. A pick in one nobody is watching moves
    every file's mtime and must not wake a single reader."""
    _write(farm, _room(league_id="111", picks=20))
    _write(farm, _room(league_id="222", picks=2))
    shown = demo._identity(time.time())
    assert shown[0] in {"111", "222"}
    other = "222" if shown[0] == "111" else "111"
    _write(farm, _room(league_id=other, picks=9))
    assert demo._identity(time.time()) == shown


def test_the_shown_room_is_not_stolen_by_one_that_overtakes_it(farm):
    """The farm sits in several rooms at once and they run neck and neck. The
    page must not switch to whichever one happens to have made the latest
    pick -- that is a slideshow of strangers, not a draft."""
    _write(farm, _room(league_id="111", picks=20))
    _write(farm, _room(league_id="222", picks=18))
    assert demo._identity(time.time())[0] == "111"
    _write(farm, _room(league_id="222", picks=21))
    assert demo._identity(time.time())[0] == "111"
    _write(farm, _room(league_id="222", picks=30))
    assert demo._identity(time.time())[0] == "111"


def test_the_shown_room_is_left_when_it_reaches_the_handoff_round(farm):
    """Stay through round nine; the moment the room enters round ten, move
    to the best of the others."""
    teams = 8
    _write(farm, _room(league_id="111", picks=teams * 7, teams=teams))
    _write(farm, _room(league_id="222", picks=20, teams=teams))
    assert demo._identity(time.time())[0] == "111"
    # Last pick of round nine: still round nine, still ours -- though a
    # fresh choice would not have picked a room this deep.
    _write(farm, _room(league_id="111", picks=teams * 9 - 1, teams=teams))
    assert demo._identity(time.time())[0] == "111"
    # Round ten begins.
    _write(farm, _room(league_id="111", picks=teams * 9, teams=teams))
    assert demo._identity(time.time())[0] == "222"
    # And it does not come back, even though it has more picks than 222.
    _write(farm, _room(league_id="111", picks=teams * 9 + 1, teams=teams))
    assert demo._identity(time.time())[0] == "222"


def test_the_shown_room_is_shown_live_past_the_interesting_rounds(farm):
    """A room the page has stuck with is shown where it is, not rewound to
    round three the moment it crosses into round nine."""
    teams = 8
    _write(farm, _room(league_id="111", picks=teams * 8 + 3, teams=teams))
    demo._SHOWING["league_id"] = "111"
    assert demo._identity(time.time()) == ("111", teams * 8 + 3)


def test_a_room_that_finished_is_left(farm):
    _write(farm, _room(league_id="111", picks=20, teams=8, rounds=3))
    _write(farm, _room(league_id="222", picks=10))
    demo._SHOWING["league_id"] = "111"
    _write(farm, _room(league_id="111", picks=24, teams=8, rounds=3))
    assert demo._identity(time.time())[0] == "222"


def test_a_room_that_went_quiet_is_left(farm):
    path = _write(farm, _room(league_id="111", picks=20))
    _write(farm, _room(league_id="222", picks=10))
    assert demo._identity(time.time())[0] == "111"
    ago = time.time() - demo.STALE_SECONDS - 1
    os.utime(path, (ago, ago))
    assert demo._identity(time.time())[0] == "222"


def test_a_room_is_kept_for_its_first_ten_picks_however_the_others_run(farm):
    """A room past the handoff round has nothing to recommend it but its pick
    count, and a farm sitting in several of those at once has a new "furthest
    along" after nearly every pick in the building -- which is a hero that
    flips drafts on most polls. The page picks one and keeps it for ten of its
    own picks before it will look at the others again."""
    teams = 8
    _write(farm, _room(league_id="111", picks=teams * 11, teams=teams))
    _write(farm, _room(league_id="222", picks=teams * 10, teams=teams))
    assert demo._identity(time.time())[0] == "111"
    # 222 runs away with it, three picks to every one of 111's. Without a
    # floor under the choice that is a different room on screen every time.
    for extra in range(1, demo.STICKY_PICKS):
        _write(farm, _room(league_id="222", picks=teams * 10 + extra * 3,
                           teams=teams))
        _write(farm, _room(league_id="111", picks=teams * 11 + extra,
                           teams=teams))
        assert demo._identity(time.time())[0] == "111"
    # The tenth pick since it was chosen spends the floor, and the room with
    # the most picks takes over.
    _write(farm, _room(league_id="111", picks=teams * 11 + demo.STICKY_PICKS,
                       teams=teams))
    assert demo._identity(time.time())[0] == "222"


def test_a_held_room_that_goes_quiet_is_left_at_once(farm):
    """The floor is spent in picks, but a room whose file has stopped being
    written is over -- and being over does not wait for ten of them."""
    teams = 8
    path = _write(farm, _room(league_id="111", picks=teams * 11, teams=teams))
    _write(farm, _room(league_id="222", picks=teams * 10, teams=teams))
    assert demo._identity(time.time())[0] == "111"
    ago = time.time() - demo.STALE_SECONDS - 1
    os.utime(path, (ago, ago))
    assert demo._identity(time.time())[0] == "222"


def test_a_held_room_that_ends_is_left_at_once(farm):
    """Nor does a room that has drafted its last pick: there is nothing left
    to hold on to, four picks into the floor or not."""
    teams = 8
    _write(farm, _room(league_id="111", picks=20, teams=teams, rounds=3))
    _write(farm, _room(league_id="222", picks=10, teams=teams))
    assert demo._identity(time.time())[0] == "111"
    _write(farm, _room(league_id="111", picks=24, teams=teams, rounds=3))
    assert demo._identity(time.time())[0] == "222"


def test_dir_revision_notices_any_write(farm):
    _write(farm, _room(picks=1))
    before = demo._dir_revision(time.time())
    _write(farm, _room(league_id="999", picks=1))
    assert demo._dir_revision(time.time()) != before


def test_clock_is_the_length_espn_stated_and_the_last_pick_s_moment(farm):
    landed = time.time() - 12.0
    room = _room(clock=90.0, written_at=landed)
    length, opened = demo._clock(room, room["picks"], True)
    assert length == 90.0
    assert opened == pytest.approx(landed)


def test_the_corpus_answers_when_the_archive_has_no_timings(farm, monkeypatch):
    """Where the archive ends up after enough rooms published by processes
    that never recorded a clock: nothing timed on file, and the question goes
    to the drafts that are."""
    monkeypatch.setattr(demo, "_corpus_clock", lambda now=None: 30.0)
    _write(farm, _room(picks=30, clock=None))
    for_archive = demo._record(str(farm / "777.json"))
    for pick in for_archive["picks"]:
        pick["clock_seconds"] = None
    demo._archive(for_archive)
    demo._REPLAY.clear()
    assert demo._observed_clock() == 30.0


def test_no_clock_anywhere_is_no_countdown(farm, monkeypatch):
    """Nothing published by the room and nothing on file to borrow. Thirty or
    ninety typed in here would be a timer that disagrees with the room it
    claims to be drawn from, so there is no timer."""
    monkeypatch.setattr(demo, "_corpus_clock", lambda now=None: None)
    assert demo._observed_clock() is None
    room = _room(clock=None)
    for pick in room["picks"]:
        pick["clock_seconds"] = None
    assert demo._clock(room, room["picks"], True) == (None, None)


def test_a_rewound_room_has_no_clock(farm):
    """`_moment` shows a late draft at the end of its third round. The seconds
    since those picks are hours, and a countdown over them would be fiction."""
    room = _room(picks=3)
    assert demo._clock(room, room["picks"], False) == (None, None)


def _app(warm=False):
    """The demo routes on a bare app.

    Cold by default: warming spawns a background build, and a test that has
    just arranged the cache by hand should not be racing a thread that
    replaces it."""
    app = FastAPI()
    with mock.patch.object(demo, "WARM_ON_REGISTER", warm):
        demo.register_demo_routes(app, conn=None)
    return app


@pytest.fixture
def brief(monkeypatch):
    """A stream that ends on its own, so a test can read all of it.

    In a server, a stream ends when the reader goes away -- and the test
    client never does: it drives the app in-process and its request reports
    itself connected forever. So the OTHER exit is used, the one that exists
    because a tab left open for a week should not pin a generator for a week,
    wound down from fifteen minutes to a fraction of a second."""
    monkeypatch.setattr(demo, "EVENT_TICK", 0.02)
    monkeypatch.setattr(demo, "EVENT_MAX_SECONDS", 0.1)


def _events(body: str) -> list:
    return [json.loads(line[len("data: "):]) for line in body.splitlines()
            if line.startswith("data: ")]


def test_the_stream_says_the_room_moved(farm, brief):
    _write(farm, _room(picks=3))
    with TestClient(_app()) as client:
        response = client.get("/api/demo/events")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert _events(response.text)[0] == {"league": "777", "pick": 3}


def test_the_stream_repeats_nothing_while_the_room_is_still(farm, brief):
    """Every tick looks; only a change speaks. A stream that re-sent the same
    line every second would be a poll with extra steps."""
    _write(farm, _room(picks=3))
    with TestClient(_app()) as client:
        response = client.get("/api/demo/events")
    assert len(_events(response.text)) == 1


def test_the_stream_carries_no_board(farm, brief):
    """It is a signal, not a payload -- the board comes from the cached
    endpoint, once, however many readers were woken."""
    _write(farm, _room(picks=3))
    with TestClient(_app()) as client:
        response = client.get("/api/demo/events")
    assert set(_events(response.text)[0]) == {"league", "pick"}


# -- the gap between rooms -------------------------------------------------
#
# The farm deletes a room's file the moment its draft ends, so there is a
# stretch of every hour with nothing drafting. These cover what the page
# shows then, and the rule that it never pretends.


def test_a_live_room_is_archived_on_the_way_past(farm):
    _write(farm, _room(picks=40))
    demo._archive(demo._record(str(farm / "777.json")))
    assert (farm.parent / "archive" / "777.json").exists()


def test_a_room_too_early_to_replay_is_not_archived(farm):
    """Two rounds in is the floor: a replay that opens on an empty board is
    the same nothing the archive exists to avoid."""
    _write(farm, _room(picks=3))
    demo._archive(demo._record(str(farm / "777.json")))
    assert not (farm.parent / "archive").exists()


def test_the_archive_plays_when_nothing_is_drafting(farm):
    _write(farm, _room(picks=60))
    demo._archive(demo._record(str(farm / "777.json")))
    (farm / "777.json").unlink()            # the draft ended; the farm swept
    assert demo._identity(time.time())[0] == "replay"


def test_a_replay_advances_on_the_clock_with_no_file_changing(farm):
    """Which is why the event stream cannot watch mtimes alone."""
    _write(farm, _room(picks=60))
    demo._archive(demo._record(str(farm / "777.json")))
    (farm / "777.json").unlink()
    now = time.time()
    assert demo._identity(now) != demo._identity(now + 120)


def test_a_live_room_always_wins(farm):
    """The archive is the fallback, never the preference -- a draft happening
    now is the stronger claim and the page must not settle for less."""
    _write(farm, _room(league_id="555", picks=60))
    demo._archive(demo._record(str(farm / "555.json")))
    _write(farm, _room(league_id="666", picks=20))
    assert demo._identity(time.time())[0] != "replay"


def test_a_replayed_pick_is_clamped_to_a_watchable_pace():
    assert demo._dwell({"seconds_to_pick": 1.0}) == demo.REPLAY_MIN_SECONDS
    assert demo._dwell({"seconds_to_pick": 89.0}) == demo.REPLAY_MAX_SECONDS
    assert demo._dwell({"seconds_to_pick": 20.0}) == 20.0
    # Never timed at all -- the farm connected mid-turn. Still moves.
    assert demo.REPLAY_MIN_SECONDS <= demo._dwell({}) <= demo.REPLAY_MAX_SECONDS


def test_nothing_at_all_is_said_plainly(farm):
    """No farm and no archive -- a fresh install, or a hosted box the farm
    does not run on. The page says so rather than inventing a draft, which is
    the one lie a reader could check."""
    assert demo._replayed(time.time()) == (None, None)
    with TestClient(_app()) as client:
        body = client.get("/api/demo/live").json()
    assert body["live"] is False
    assert body["mode"] == "none"


# -- nobody waits for a build ----------------------------------------------
#
# A build is 2.8-4.0s warm and 7.6s cold. The rule is that no reader is ever
# on the other end of one: the answer that exists is served, and the rebuild
# happens behind them.


def test_a_stale_answer_is_served_rather_than_rebuilt(farm):
    """The reader gets the board immediately even though the room has moved
    on -- and the request does not build. The refresh runs behind it."""
    _write(farm, _room(picks=3))
    app = _app()
    standing = {"live": True, "mode": "live", "picks_made": 3}
    demo._CACHE["live"] = (("777", 1), time.monotonic() + 60, standing,
                           time.monotonic() - 3600)
    started = []

    def explode(*args, **kwargs):
        started.append(True)
        raise AssertionError("a request thread must never build")

    with mock.patch.object(demo, "_build", explode), \
            mock.patch.object(demo, "refresh_behind", create=True):
        with TestClient(app) as client:
            body = client.get("/api/demo/live").json()
    assert body == standing
    assert not started


def test_a_restart_starts_from_the_last_answer(tmp_path, monkeypatch):
    """What the previous process built is on disk, and goes straight back into
    the cache -- so the first visitor after a restart is not the one who pays
    for a cold build."""
    path = tmp_path / "demo-last.json"
    path.write_text(json.dumps({"live": True, "mode": "live",
                                "picks_made": 40, "server_now": time.time()}))
    monkeypatch.setattr(demo, "DEMO_LAST", str(path))
    restored = demo._restore(time.time())
    assert restored["picks_made"] == 40
    # Downgraded on the way in: that room was live when it was written and is
    # not live now, and the pill above it must not say it is.
    assert restored["mode"] == "replay"


def test_a_restart_keeps_showing_the_room_it_was_showing(farm, tmp_path,
                                                         monkeypatch):
    """The last answer names its room, and a restarted server picks that room
    back up rather than starting the choice over -- which, with two rooms
    neck and neck, would be one more jump for whoever is watching."""
    _write(farm, _room(league_id="111", picks=20))
    _write(farm, _room(league_id="222", picks=19))
    path = tmp_path / "demo-last.json"
    path.write_text(json.dumps({"live": True, "mode": "live", "league_id": "222",
                                "picks_made": 19, "server_now": time.time()}))
    monkeypatch.setattr(demo, "DEMO_LAST", str(path))
    monkeypatch.setattr(demo, "WARM_ON_REGISTER", True)
    with mock.patch.object(demo.threading, "Thread"):
        demo.register_demo_routes(FastAPI(), conn=mock.Mock())
    assert demo._identity(time.time())[0] == "222"


def test_last_night_s_answer_is_not_restored(tmp_path, monkeypatch):
    path = tmp_path / "demo-last.json"
    path.write_text(json.dumps({"live": True, "mode": "live",
                                "server_now": time.time() - demo.LAST_MAX_AGE - 1}))
    monkeypatch.setattr(demo, "DEMO_LAST", str(path))
    assert demo._restore(time.time()) is None


def test_a_room_that_never_published_a_clock_borrows_the_one_on_file(farm):
    """Most live rooms are silent about their clock until the farm's
    processes turn over. The archive answers for them -- but only when every
    room on file says the same thing."""
    _write(farm, _room(picks=30, clock=30.0))
    demo._archive(demo._record(str(farm / "777.json")))
    demo._REPLAY.clear()
    silent = _room(league_id="888", picks=10, clock=None)
    for pick in silent["picks"]:
        pick["clock_seconds"] = None
    length, opened = demo._clock(silent, silent["picks"], True)
    assert length == 30.0
    assert opened == pytest.approx(silent["written_at"])


def test_rooms_that_disagree_buy_no_countdown_at_all(farm):
    _write(farm, _room(league_id="111", picks=30, clock=30.0))
    demo._archive(demo._record(str(farm / "111.json")))
    _write(farm, _room(league_id="222", picks=30, clock=90.0))
    demo._archive(demo._record(str(farm / "222.json")))
    demo._REPLAY.clear()
    assert demo._observed_clock() is None
    silent = _room(league_id="888", picks=10)
    for pick in silent["picks"]:
        pick["clock_seconds"] = None
    assert demo._clock(silent, silent["picks"], True) == (None, None)


def test_the_roster_shows_every_slot_the_seat_will_fill(farm):
    """Ten starter slots in a sixteen-round room read as a full roster at
    5/10. The bench is part of the roster whether or not anybody is on it."""
    settings = {"starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1,
                             "DST": 1}, "flex_slots": 2, "rounds": 16}

    class _Board:
        def itertuples(self):
            return iter(())

    slots = demo._roster_payload([], 1, _Board(), settings)
    assert len(slots) == 16
    assert [s["slot"] for s in slots][-6:] == ["BN1", "BN2", "BN3", "BN4",
                                               "BN5", "BN6"]


def test_flex_reads_with_the_group_it_is_filled_from():
    """The live room's rail puts FLEX between WR and TE (SLOT_DISPLAY_ORDER in
    DraftRoom.tsx). This page draws the same panel, and was leaving FLEX after
    DST, where the build loop happens to append it."""
    settings = {"starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1,
                             "DST": 1}, "flex_slots": 2, "rounds": 16}

    class _Board:
        def itertuples(self):
            return iter(())

    slots = demo._roster_payload([], 1, _Board(), settings)
    assert [s["slot"] for s in slots][:10] == [
        "QB", "RB1", "RB2", "WR1", "WR2", "FLEX1", "FLEX2", "TE", "K", "DST"]


def test_flex_is_still_filled_from_the_leftovers_not_first():
    """Display order moved; the ASSIGNMENT rule did not. A dedicated slot is
    always filled before a FLEX one, so two receivers fill WR1/WR2 and the
    third is the one that lands in FLEX1."""
    settings = {"starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1,
                             "DST": 1}, "flex_slots": 2, "rounds": 16}

    class _Row:
        def __init__(self, pid, name):
            self.player_id, self.name = pid, name
            self.position, self.team, self.bye, self.proj_points = "WR", "SF", 9, 200.0

    class _Board:
        def itertuples(self):
            return iter((_Row("w1", "First"), _Row("w2", "Second"),
                         _Row("w3", "Third")))

    picks = [{"pick_no": n + 1, "slot": 1, "player_id": pid}
             for n, pid in enumerate(("w1", "w2", "w3"))]
    slots = demo._roster_payload(picks, 1, _Board(), settings)
    named = {s["slot"]: (s["player"] or {}).get("name") for s in slots}
    assert named["WR1"] == "First" and named["WR2"] == "Second"
    assert named["FLEX1"] == "Third"


def test_a_locked_corpus_costs_a_minute_not_an_hour(monkeypatch):
    """The farm holds a write lock while it records a finished draft. A read
    that lands in that second must not take the countdown away until the next
    hour."""
    demo._CLOCK.clear()
    demo._CLOCK.update({"at": time.time() - 120, "value": None})
    calls = []

    def _fake_connect(*args, **kwargs):
        calls.append(True)
        raise OSError("locked")

    monkeypatch.setattr("duckdb.connect", _fake_connect)
    assert demo._corpus_clock() is None
    assert calls, "a failed answer older than the retry window must be retried"
    demo._CLOCK.clear()


def test_every_pick_carries_the_whole_board_player_contract():
    """The room's own components read a landed pick as `BoardPlayer`, whose
    numeric fields are `number | null` and are guarded on `!== null`. A field
    this payload leaves OUT arrives as `undefined`, walks past that guard, and
    throws at `.toFixed` -- inside a click handler, where the only visible
    symptom is that clicking the pick does nothing at all. Every key, always,
    null where there is nothing to say."""
    class _Row:
        player_id = "00-0001"
        name = "Somebody"
        headshot = None
        position = "RB"
        team = "SF"
        bye = 9
        rank = 12
        tier = 2
        market_rank = 20
        espn_ppr_rank = 18
        vor = 33.0
        proj_points = 170.0

    class _Board:
        def itertuples(self):
            return iter((_Row(),))

    picks = [{"pick_no": 1, "slot": 1, "player_id": "00-0001"}]
    payload = demo._board_payload({"teams": 8, "rounds": 16}, picks, _Board(), 2)
    player = payload["cells"][0]["player"]
    # api.ts's `BoardPlayer`, field for field.
    assert set(player) == {
        "player_id", "name", "headshot", "position", "team", "bye",
        "overall_rank", "tier", "market_rank", "espn_ppr_rank", "vor",
        "last_ppg", "last_points", "proj_ppg", "value",
    }
    # A season projection, printed per game on the room's own 17-game
    # denominator.
    assert player["proj_ppg"] == 10.0
    assert player["vor"] == 33.0
    assert player["last_ppg"] is None and player["last_points"] is None


def test_ranked_rows_carry_the_whole_live_candidate_contract(monkeypatch):
    """Same failure mode `test_every_pick_carries_the_whole_board_player_contract`
    documents, one payload over: the room's own components read these rows
    as `LiveCandidate`, and a key left out arrives as `undefined` past every
    `!== null` guard. The landing room runs the room's own `rank_and_plan`
    and `target_now`, so its rows carry exactly the room's candidate keys
    plus the join-table extras this page serves itself."""
    import numpy as np
    import pandas as pd
    import scoring.league as league
    from api import live
    from scoring.league import LeagueSettings

    class _Pool:
        player_id = np.array(["00-0001", "00-0002"], dtype=object)
        position = np.array(["RB", "RB"], dtype=object)
        points = np.array([300.0, 250.0])

    board = pd.DataFrame([
        {"player_id": "00-0001", "name": "Somebody", "team": "SF", "bye": 9,
         "rank": 1, "market_rank": 2, "position": "RB", "proj_points": 300.0,
         "espn_rank": 1.0, "espn_pos_rank": 1.0, "espn_adp": 1.5, "career_games_pg": 16.0},
        {"player_id": "00-0002", "name": "Other", "team": "DET", "bye": 5,
         "rank": 2, "market_rank": 3, "position": "RB", "proj_points": 250.0,
         "espn_rank": 2.0, "espn_pos_rank": 2.0, "espn_adp": 2.5, "career_games_pg": 15.0},
    ])
    monkeypatch.setattr(demo, "_cached_pool", lambda conn, board, s: _Pool())
    monkeypatch.setattr(league, "load", lambda conn: LeagueSettings(
        season=2026, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=2, bench=5, scoring={}, draft_type="SNAKE"))
    # Everybody lasts: what an empty corpus's fallback says about players
    # nobody has ranked, and enough to see the shape of the rows.
    monkeypatch.setattr(live, "availability_at",
                        lambda table, ids, k, n, *a, **kw: np.ones(len(list(ids))))

    rows = demo._ranked(conn=None, board=board, picks=[], slot=1, limit=3)
    assert len(rows) == 2, "both board rows should have come through"
    assert set(rows[0]) == {
        "player_id", "position", "proj_points", "espn_rank", "espn_pos_rank",
        "espn_adp", "market_rank", "lasts_pct", "lasts_at_pick", "edge_pts",
        "need", "favourite", "rank",
        "name", "team", "bye", "adp", "vs_adp", "board_rank",
    }
    # Seat 1 with nothing drafted is on the clock at pick 1; "lasts" is
    # measured to its next turn, pick 16, and the better back leads.
    assert rows[0]["player_id"] == "00-0001"
    assert rows[0]["lasts_at_pick"] == 16 and rows[0]["lasts_pct"] == 100.0
    assert rows[0]["edge_pts"] == pytest.approx(50.0)
    assert rows[0]["name"] == "Somebody" and rows[0]["adp"] == 2 and rows[0]["board_rank"] == 1


def test_a_pick_that_fell_past_its_adp_is_a_positive_value():
    """Same sign as `api/live.py`: overall minus market rank, so a player the
    market had at 21 who went 31st is +10 -- a steal, drawn green with an up
    arrow. For a while this was written the other way round, and every arrow
    on the demo rail pointed the wrong way; the ticker's tooltip, which prints
    both numbers beside the verdict, is what made it visible."""
    import pandas as pd
    board = pd.DataFrame([{"player_id": "p", "name": "George Pickens", "position": "WR",
                           "team": "DAL", "market_rank": 21.0, "proj_points": 200.0}])
    picks = [{"player_id": "p", "pick_no": 31, "slot": 2}]
    payload = demo._board_payload({"teams": 8}, picks, board, None)
    cell = payload["cells"][0]
    assert cell["overall"] == 31 and cell["player"]["market_rank"] == 21
    assert cell["player"]["value"] == 10
