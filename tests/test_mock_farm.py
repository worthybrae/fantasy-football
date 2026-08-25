"""pipeline.mock_farm / pipeline.espn_mock_lobby: the loop that joins real
ESPN mock drafts and writes them into the cross-league corpus.

ENTIRELY OFFLINE, and that is a requirement rather than a convenience. Every
network-facing seam in both modules is an injected callable (`fetch`, `post`)
or a socket the caller owns, so what is tested here is the same code the live
farm runs, driven from two fixtures instead of ESPN:

  - `data/draft_room_trace.jsonl`, a real recorded 8-team ESPN mock: 128
    SELECTED frames, team ids 1..8, and AUTODRAFT frames flipping in both
    directions mid-draft. Replayed through the real `DraftListener`, so the
    record-building path is exercised against frames ESPN actually sent
    rather than against frames a test author imagined.
  - a hand-built pool for the pick policy, small enough that the right answer
    can be worked out by hand and seeded so both branches of the
    epsilon-greedy coin are pinned rather than sampled.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import draft_log as dl
from pipeline import espn_mock_lobby as lobby
from pipeline import mock_farm as mf
from pipeline import redact
from pipeline.draft_listener import DraftListener
from pipeline.espn_live import picks_from_events
from scoring import league
from scoring.draft_sim import SimPool, _roster_cap, snake_slots

# Resolved from this file rather than the working directory: pytest can be
# invoked from anywhere, and a relative path would skip the whole trace
# suite silently rather than fail.
TRACE = Path(__file__).resolve().parents[1] / "data" / "draft_room_trace.jsonl"

# What the capture holds, counted from the file itself once and pinned here
# so a change in the parsing silently producing different numbers fails the
# test rather than passing quietly.
TRACE_PICKS = 128
TRACE_TEAMS = 8
TRACE_ROUNDS = 16
TRACE_AUTODRAFTED = 30       # picks ESPN made for a team whose clock expired
TRACE_NOT_AUTODRAFTED = 13   # picks with an explicit AUTODRAFT <team> false
TRACE_AUTODRAFT_UNKNOWN = 85  # teams ESPN had not sent any AUTODRAFT for yet

# The room's own owner census, as `?view=mTeam` would report it -- four seats
# with a person, four of ESPN's computer entries. Invented (the capture is a
# socket trace and carries no REST body), but invented to match the shape the
# lobby actually serves: the live measurement at 09:45 found no joinable
# 8-team room with more than three people in it.
TRACE_OWNERS = {1: True, 2: True, 3: True, 4: False,
                5: False, 6: False, 7: True, 8: False}
# What the state machine makes of the capture once those four computer seats
# start out autodrafting. Counted from the fold once and pinned, so a change
# in the state machine fails here rather than passing quietly.
TRACE_AUTO_WITH_OWNERS = 94
TRACE_HUMAN_WITH_OWNERS = 34

# Every one of the capture's 128 turns opened with `SELECTING <team> 30000`.
# A real value off the wire, not a default: `clock_seconds` is stored per
# pick precisely because a room can run a different clock, and "took 25
# seconds" means opposite things on a 30-second clock and a 90-second one.
TRACE_CLOCK_SECONDS = 30.0
# What the capture's first three picks measure when the replay clock advances
# one second per frame (see `_ticking_clock`): a pick's duration is then
# exactly the number of frames ESPN sent between the SELECTING that opened
# its turn and the SELECTED that closed it. Counted from the file once and
# pinned here, the same way TRACE_PICKS is, so a change in the fold fails
# here rather than passing quietly.
TRACE_FIRST_SECONDS = [1.0, 3.0, 4.0]
TRACE_SLOWEST_SECONDS = 10.0


@pytest.fixture(autouse=True)
def _isolated_live_dir(tmp_path, monkeypatch):
    """Never the real `data/farm-live`.

    Autouse and file-wide, for the same reason `_stub_farm_deps` redirects the
    claim directory: `play_draft` now publishes a live file after every pick
    and `farm` deletes one in its `finally`, so several tests here that were
    written long before either existed would otherwise write into -- and sweep
    -- the directory a farm process on this machine may be publishing a real
    draft through, making that draft vanish off the page mid-round.
    """
    monkeypatch.setattr(mf, "LIVE_DIR", str(tmp_path / "farm-live"))
    return tmp_path / "farm-live"


def _trace_listener(clock=None) -> DraftListener:
    """The real recorded draft, replayed through the real listener.

    Frames are filtered to the fantasydraft host: the capture also holds
    ESPN's unrelated bamgrid connection, whose frames are JSON blobs the
    draft protocol knows nothing about.

    `clock` replaces the listener's own `time.monotonic`. The capture carries
    no timestamps of its own -- it is a list of frames, nothing more -- so a
    replay through the real clock finishes in milliseconds and every pick
    "takes" a few microseconds. Handing in a scripted clock is what makes the
    durations assertable: see `_ticking_clock`.
    """
    if not TRACE.exists():
        pytest.skip(f"{TRACE} is not present in this checkout")
    listener = (DraftListener({}) if clock is None
                else DraftListener({}, clock=clock))
    for line in TRACE.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") != "ws-recv" or "fantasydraft" not in (row.get("url") or ""):
            continue
        listener.on_frame(row.get("payload") or "")
    return listener


# ---------------------------------------------------------------------------
# The timeline: pick numbering, slot attribution, autodraft flags.
# ---------------------------------------------------------------------------


def test_the_recorded_draft_replays_as_128_picks():
    timeline = mf.draft_timeline(_trace_listener().events)
    assert len(timeline) == TRACE_PICKS
    assert [f.pick_no for f in timeline] == list(range(1, TRACE_PICKS + 1))


def test_pick_numbering_agrees_with_picks_from_events():
    """The corpus stores `pick_no`, and `pick_no` is what attributes a pick to
    a slot. `draft_timeline` reproduces `espn_live.picks_from_events`'
    numbering rules (every SELECTED consumes a number; a repeated player id is
    a reconnect replay and consumes none) rather than approximating them --
    this is the assertion that keeps the two from drifting."""
    events = _trace_listener().events
    live = picks_from_events(events, {})
    theirs = sorted([int(r) for r in live.rows["pick_no"]]
                    + [u["overall_pick"] for u in live.unmapped])
    assert [f.pick_no for f in mf.draft_timeline(events)] == theirs


def test_every_slot_belongs_to_exactly_one_team():
    """The team-count check. ESPN's SELECTED frames carry the team id, so an
    assumed snake that is the wrong width shows up as one slot drafted by two
    different teams."""
    timeline = mf.draft_timeline(_trace_listener().events)
    mapping = mf.slot_team_map(timeline, snake_slots(TRACE_TEAMS, TRACE_ROUNDS))
    assert mapping == {slot: slot for slot in range(1, TRACE_TEAMS + 1)}


def test_a_wrong_team_count_is_caught_rather_than_recorded():
    timeline = mf.draft_timeline(_trace_listener().events)
    with pytest.raises(ValueError, match="does not match the room"):
        mf.slot_team_map(timeline, snake_slots(10, 16))


def test_our_slot_comes_off_the_socket():
    timeline = mf.draft_timeline(_trace_listener().events)
    slots = snake_slots(TRACE_TEAMS, TRACE_ROUNDS)
    for team in range(1, TRACE_TEAMS + 1):
        assert mf.slot_for_team(timeline, slots, team) == team


def test_slot_is_none_until_the_socket_can_answer():
    """A guessed slot attributes every pick of the draft to the wrong seat, so
    "not known yet" has to be expressible."""
    slots = snake_slots(TRACE_TEAMS, TRACE_ROUNDS)
    assert mf.slot_for_team([], slots, 3) is None
    # On the clock with no pick yet: the slot of the pick about to happen.
    assert mf.slot_for_team([], slots, 3, on_the_clock=3) == 1
    assert mf.slot_for_team([], slots, 3, on_the_clock=4) is None


def test_autodraft_flags_match_the_capture():
    """Three-valued on purpose: True, False and "ESPN has not said" are
    different facts and the corpus stores which one it has."""
    timeline = mf.draft_timeline(_trace_listener().events)
    flags = [f.autodrafted for f in timeline]
    assert flags.count(True) == TRACE_AUTODRAFTED
    assert flags.count(False) == TRACE_NOT_AUTODRAFTED
    assert flags.count(None) == TRACE_AUTODRAFT_UNKNOWN


def test_the_owner_census_decides_a_seat_nobody_ever_sent_a_frame_about():
    """The whole point of the mTeam read. ESPN sends NO AUTODRAFT frame for a
    seat that was never human, so before the census those picks were NULL --
    0 True / 32 False / 96 NULL per draft, useless -- and the fit kept them."""
    events = _trace_listener().events
    blind = mf.draft_timeline(events)
    assert (sum(1 for f in blind if f.autodrafted is None)
            == TRACE_AUTODRAFT_UNKNOWN)

    timeline = mf.draft_timeline(events, TRACE_OWNERS)
    flags = [f.autodrafted for f in timeline]
    assert flags.count(None) == 0, "every pick's seat is now accounted for"
    assert flags.count(True) == TRACE_AUTO_WITH_OWNERS
    assert flags.count(False) == TRACE_HUMAN_WITH_OWNERS
    # Team 4 is one of ESPN's computer seats and ESPN never says so on the
    # socket. All sixteen of its picks are the engine's, from pick one.
    assert all(f.autodrafted is True
               for f in timeline if f.team_id == 4)


def test_a_seat_flips_both_ways_and_each_pick_takes_its_own_state():
    """The case that matters most: somebody drafts a few rounds and wanders
    off, then comes back. The capture has it -- team 2 is human, ESPN takes
    over after pick 62, the client asks for it back after pick 77, and ESPN
    takes over again after 126 -- and the flag has to follow, pick by pick,
    rather than being the room's state at the end."""
    timeline = mf.draft_timeline(_trace_listener().events, TRACE_OWNERS)
    team_2 = [(f.pick_no, f.autodrafted) for f in timeline if f.team_id == 2]
    assert team_2 == [
        (2, False), (15, False), (18, False), (31, False), (34, False),
        (47, False), (50, False),
        (63, True), (66, True),                       # ESPN took over
        (79, False), (82, False), (95, False), (98, False),
        (111, False), (114, False),                   # and gave it back
        (127, True)]                                  # and took it again
    # The listener's own dict is a LATEST-VALUE snapshot and says only what
    # the room looked like when the socket closed. That is why the timeline
    # folds the retained event list instead of reading it.
    assert _trace_listener().autodraft_by_team[2] is True


def test_a_frame_still_beats_the_census_it_contradicts():
    """The census is the INITIAL state, not an override. A seat ESPN listed
    as a computer that then sends `AUTODRAFT <team> false` is a person who
    arrived late, and the frame is the newer fact."""
    listener = DraftListener({})
    for frame in ["SELECTED 1 100 2\n", "AUTODRAFT 1 false\n",
                  "SELECTED 1 200 4\n"]:
        listener.on_frame(frame)
    timeline = mf.draft_timeline(listener.events, {1: False})
    assert [f.autodrafted for f in timeline] == [True, False]


def test_the_head_count_leaves_our_own_seat_out():
    """We always have an owner and we are a bot. Counting ourselves would
    make every room in the corpus look one person more human than it was."""
    assert mf.human_seats(TRACE_OWNERS, 2) == 3      # 4 owned seats, minus us
    assert mf.human_seats(TRACE_OWNERS, 5) == 4      # our seat was unowned
    # None, not 0: "the mTeam read failed" and "the room was all computers"
    # are different facts about a draft and only one is a reason to distrust
    # its picks.
    assert mf.human_seats({}, 2) is None
    assert mf.human_seats(None, 2) is None
    assert mf.human_seats({1: False, 2: False}, 9) == 0


def test_a_replayed_pick_does_not_consume_a_second_pick_number():
    """ESPN re-sends every prior SELECTED on a (re)JOIN, and this loop
    reconnects whenever ESPN drops the socket. Counting a replay twice would
    double the draft and shift every slot."""
    listener = DraftListener({})
    for frame in ["SELECTED 1 100 2\n", "SELECTED 2 200 4\n",
                  "SELECTED 1 100 2\n", "SELECTED 3 300 0\n"]:
        listener.on_frame(frame)
    timeline = mf.draft_timeline(listener.events)
    assert [(f.pick_no, f.espn_id) for f in timeline] == [
        (1, 100), (2, 200), (3, 300)]


def test_a_frame_with_no_player_id_still_consumes_its_pick_number():
    """Dropping it would leave a drafted player on the board; renumbering
    around it would hand every later pick to the wrong team."""
    listener = DraftListener({})
    for frame in ["SELECTED 1 100 2\n", "SELECTED 2\n", "SELECTED 3 300 0\n"]:
        listener.on_frame(frame)
    timeline = mf.draft_timeline(listener.events)
    assert [(f.pick_no, f.espn_id) for f in timeline] == [
        (1, 100), (2, None), (3, 300)]


def test_an_unrecognised_autodraft_value_leaves_the_last_known_flag():
    """Mapping an unknown value to False would report "autodraft is off" for a
    team ESPN may be picking for -- the exact failure the flag exists to
    prevent."""
    listener = DraftListener({})
    for frame in ["AUTODRAFT 1 true\n", "AUTODRAFT 1 maybe\n",
                  "SELECTED 1 100 2\n"]:
        listener.on_frame(frame)
    assert mf.draft_timeline(listener.events)[0].autodrafted is True


# ---------------------------------------------------------------------------
# How long each pick took: the most behavioral signal in the stream.
# ---------------------------------------------------------------------------


def _ticking_clock(step=1.0, start=0.0):
    """A clock that advances `step` on every reading.

    One reading per frame (`DraftListener.on_frame` stamps exactly once), so
    a duration measured through this clock is the frame COUNT between two
    frames -- deterministic, and derived from the capture rather than from
    wall-clock timing that would make the assertions flaky.
    """
    state = {"t": start - step}

    def clock():
        state["t"] += step
        return state["t"]

    return clock


def _listener_at(times, frames) -> DraftListener:
    """Replay `frames` with one scripted arrival time each.

    Used where the exact seconds matter more than the frame count -- a
    thirty-second autodraft expiry, a reconnect that arrives out of order.
    """
    ticks = iter(times)
    listener = DraftListener({}, clock=lambda: next(ticks))
    for frame in frames:
        listener.on_frame(frame)
    return listener


def test_the_recorded_draft_times_every_one_of_its_128_picks():
    """The capture is 128 clean SELECTING/SELECTED pairs, so nothing in it
    should come back untimed. `seconds_to_pick` is the thing this whole
    change exists to capture and a fold that quietly produced Nones would
    still pass every other test in this file."""
    timeline = mf.draft_timeline(_trace_listener(_ticking_clock()).events)
    assert len(timeline) == TRACE_PICKS
    assert all(f.seconds_to_pick is not None for f in timeline)
    assert [f.seconds_to_pick for f in timeline[:3]] == TRACE_FIRST_SECONDS
    assert max(f.seconds_to_pick for f in timeline) == TRACE_SLOWEST_SECONDS
    # Read off the wire, one per turn, never assumed.
    assert {f.clock_seconds for f in timeline} == {TRACE_CLOCK_SECONDS}


def test_a_pick_whose_turn_we_never_saw_open_is_untimed_not_instant():
    """Joining mid-turn is the real case: the farm's socket can attach after
    ESPN has already put somebody on the clock. Zero would say "he clicked
    the top of the list the moment it was his turn", which is a real
    behaviour this column exists to spot -- so the two must not collide."""
    listener = _listener_at([100.0, 104.0, 110.0],
                            ["SELECTED 1 100 2\n",         # joined mid-turn
                             "SELECTING 2 30000\n",
                             "SELECTED 2 200 4\n"])
    timeline = mf.draft_timeline(listener.events)
    assert timeline[0].seconds_to_pick is None
    assert timeline[0].clock_seconds is None
    assert timeline[1].seconds_to_pick == 6.0


def test_a_reconnect_replay_neither_duplicates_a_timing_nor_reverses_one():
    """ESPN re-sends the whole draft on every (re)JOIN and this loop
    reconnects whenever the socket drops. The replayed SELECTED frames arrive
    seconds after the picks they describe, so a fold that timed them would
    report durations that never happened -- and, once the replay is followed
    by the live turn, a negative one."""
    listener = _listener_at(
        [0.0, 3.0, 3.5, 9.0,          # the live draft: two real picks
         20.0, 20.5, 21.0,            # the reconnect replays both, then the
         27.0],                       # team already on the clock picks
        ["SELECTING 1 30000\n", "SELECTED 1 100 2\n",
         "SELECTING 2 30000\n", "SELECTED 2 200 4\n",
         "SELECTED 1 100 2\n", "SELECTED 2 200 4\n", "SELECTING 3 30000\n",
         "SELECTED 3 300 6\n"])
    timeline = mf.draft_timeline(listener.events)
    assert [f.espn_id for f in timeline] == [100, 200, 300]
    assert [f.seconds_to_pick for f in timeline] == [3.0, 5.5, 6.0]
    assert all(f.seconds_to_pick >= 0 for f in timeline)


def test_a_repeated_selecting_for_the_team_on_the_clock_keeps_the_first_start():
    """A reconnect mid-turn replays the SELECTING for whoever is picking.
    Taking the later stamp would shorten a real forty-second deliberation to
    the age of the reconnect, which is precisely backwards: the interesting
    picks are the slow ones."""
    listener = _listener_at([0.0, 25.0, 40.0],
                            ["SELECTING 1 30000\n", "SELECTING 1 30000\n",
                             "SELECTED 1 100 2\n"])
    assert mf.draft_timeline(listener.events)[0].seconds_to_pick == 40.0


def test_an_autodraft_pick_lands_at_the_clocks_expiry():
    """Not special-cased, and that is the point: ESPN makes the pick the
    instant the clock runs out, so an autodrafted pick falls out of the same
    arithmetic at seconds_to_pick == clock_seconds. That equality is how a
    model tells "the engine picked" from "a person picked fast" without
    having to trust the flag."""
    listener = _listener_at([0.0, 30.0, 30.0],
                            ["SELECTING 1 30000\n", "AUTODRAFT 1 true\n",
                             "SELECTED 1 100 2\n"])
    frame = mf.draft_timeline(listener.events)[0]
    assert frame.autodrafted is True
    assert frame.seconds_to_pick == frame.clock_seconds == 30.0


def test_a_turn_that_never_produced_a_pick_does_not_time_the_next_one():
    """The room has one clock, so one turn is open at a time. A stale turn
    left open would attach its start to that team's NEXT pick, minutes later,
    and record a deliberation nobody had."""
    listener = _listener_at([0.0, 100.0, 103.0, 200.0, 204.0],
                            ["SELECTING 1 30000\n",   # never picks
                             "SELECTING 2 30000\n", "SELECTED 2 200 4\n",
                             "SELECTING 1 30000\n", "SELECTED 1 100 2\n"])
    timeline = mf.draft_timeline(listener.events)
    assert [f.seconds_to_pick for f in timeline] == [3.0, 4.0]


def test_a_clock_length_the_frame_omitted_is_null_not_thirty():
    """30000 in every turn of the capture, and still read rather than
    assumed. A room on a different clock is the case where an assumed 30
    would silently mislabel every pick in it."""
    listener = _listener_at([0.0, 4.0],
                            ["SELECTING 1\n", "SELECTED 1 100 2\n"])
    frame = mf.draft_timeline(listener.events)[0]
    assert frame.seconds_to_pick == 4.0
    assert frame.clock_seconds is None


def test_an_event_list_with_no_timestamps_produces_no_durations():
    """`parse_frame` stamps nothing -- only the listener does. Anything that
    folds bare parsed events (a fixture, another test) must come back with
    Nones rather than raising or inventing zeros."""
    from pipeline.espn_live import parse_frame

    events = [parse_frame(f) for f in ["SELECTING 1 30000", "SELECTED 1 100 2"]]
    frame = mf.draft_timeline(events)[0]
    assert frame.seconds_to_pick is None
    # The clock LENGTH is still known: it came off the frame, not the clock.
    assert frame.clock_seconds == 30.0


# ---------------------------------------------------------------------------
# Building the corpus record.
# ---------------------------------------------------------------------------


def _tool_for(timeline) -> mf.Tool:
    """A Tool that can name every player in `timeline`.

    Deliberately synthetic: the point of these tests is the record SHAPE --
    slots, owner keys, autodraft flags, pick numbers -- and building a real
    board would drag DuckDB and 250 real players into a test about column
    layout. `build_tool` itself is exercised by the live run, not here.
    """
    espn_ids = [f.espn_id for f in timeline if f.espn_id is not None]
    crosswalk = {espn_id: f"p{espn_id}" for espn_id in espn_ids}
    pool_df = pd.DataFrame({
        "player_id": [f"p{e}" for e in espn_ids],
        "position": ["WR"] * len(espn_ids),
        "adp_rank": [float(i + 1) for i in range(len(espn_ids))],
        "proj_points": [200.0 - i for i in range(len(espn_ids))],
        "team": ["DET"] * len(espn_ids),
    })
    return mf.Tool(board=pd.DataFrame(), pool=None, pool_df=pool_df,
                   crosswalk=crosswalk, espn_by_index=[],
                   index_by_player={pid: i for i, pid
                                    in enumerate(pool_df["player_id"])},
                   sendable=np.ones(len(espn_ids), dtype=bool))


def _mock_settings():
    """8 teams x 16 rounds, the shape ESPN's mock lobby actually serves
    (verified live: QB/2RB/2WR/TE/FLEX/K/DST plus 7 bench)."""
    base = league.default_settings()
    import dataclasses
    return dataclasses.replace(base, bench=base.bench + 1)


def test_the_record_carries_every_pick_at_its_own_slot():
    timeline = mf.draft_timeline(_trace_listener().events)
    tool = _tool_for(timeline)
    record, dropped = mf.build_record(
        timeline, tool, _mock_settings(), league_id="999", season=2026,
        my_slot=4, started_at=None, teams=TRACE_TEAMS, rounds=TRACE_ROUNDS)

    assert dropped == 0
    assert len(record.picks) == TRACE_PICKS
    assert record.teams == TRACE_TEAMS and record.rounds == TRACE_ROUNDS
    assert record.my_slot == 4
    assert record.source == dl.SOURCE_MOCK
    expected_slots = snake_slots(TRACE_TEAMS, TRACE_ROUNDS)
    assert list(record.picks["slot"]) == expected_slots
    assert list(record.picks["round"]) == [
        (i // TRACE_TEAMS) + 1 for i in range(TRACE_PICKS)]
    # Every seat anonymous, our own included -- our picks are a bot's policy,
    # not a person's tendencies, and `my_slot` is how a reader excludes them.
    assert record.picks["is_anonymous"].all()
    assert set(record.picks["owner_key"]) == {
        dl.anonymous_key(record.resolved_id(), slot)
        for slot in range(1, TRACE_TEAMS + 1)}


def test_the_record_carries_the_autodraft_flag_the_capture_shows():
    timeline = mf.draft_timeline(_trace_listener().events)
    record, _ = mf.build_record(
        timeline, _tool_for(timeline), _mock_settings(), league_id="999",
        season=2026, my_slot=1, started_at=None, teams=TRACE_TEAMS,
        rounds=TRACE_ROUNDS)
    flags = list(record.picks["autodrafted"])
    assert flags.count(True) == TRACE_AUTODRAFTED
    assert flags.count(False) == TRACE_NOT_AUTODRAFTED
    assert sum(1 for f in flags if f is None) == TRACE_AUTODRAFT_UNKNOWN


def test_the_record_round_trips_through_the_corpus(tmp_path):
    """Written and read back rather than merely constructed: `autodrafted` is
    a column Task 1 added by ALTER, and a record that builds fine but does not
    store is exactly the failure that would not show up until a refit."""
    timeline = mf.draft_timeline(_trace_listener().events)
    record, _ = mf.build_record(
        timeline, _tool_for(timeline), _mock_settings(), league_id="999",
        season=2026, my_slot=6, started_at=None, teams=TRACE_TEAMS,
        rounds=TRACE_ROUNDS)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    try:
        draft_id = dl.record(corpus, record)
        head = corpus.execute(
            "SELECT teams, rounds, my_slot, source FROM draft_log "
            "WHERE draft_id = ?", [draft_id]).fetchone()
        assert head == (TRACE_TEAMS, TRACE_ROUNDS, 6, dl.SOURCE_MOCK)
        stored = corpus.execute(
            "SELECT count(*), count(autodrafted), "
            "       sum(CASE WHEN autodrafted THEN 1 ELSE 0 END) "
            "FROM draft_log_pick WHERE draft_id = ?", [draft_id]).fetchone()
        assert stored[0] == TRACE_PICKS
        assert stored[1] == TRACE_AUTODRAFTED + TRACE_NOT_AUTODRAFTED
        assert stored[2] == TRACE_AUTODRAFTED
        # Recording the same draft twice is one draft, not two.
        dl.record(corpus, record)
        assert corpus.execute(
            "SELECT count(*) FROM draft_log_pick WHERE draft_id = ?",
            [draft_id]).fetchone()[0] == TRACE_PICKS
    finally:
        corpus.close()


def test_the_owner_census_reaches_the_corpus_on_every_pick(tmp_path):
    """`had_owner` and `human_seats` are two more columns added by ALTER, and
    a record that builds fine but does not STORE is exactly the failure that
    would not surface until a refit read NULLs it did not expect."""
    timeline = mf.draft_timeline(_trace_listener().events, TRACE_OWNERS)
    record, _ = mf.build_record(
        timeline, _tool_for(timeline), _mock_settings(), league_id="999",
        season=2026, my_slot=2, started_at=None, teams=TRACE_TEAMS,
        rounds=TRACE_ROUNDS, owners=TRACE_OWNERS, my_team_id=2)
    assert record.human_seats == 3            # four owned seats, ours not one

    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    try:
        draft_id = dl.record(corpus, record)
        assert corpus.execute(
            "SELECT human_seats FROM draft_log WHERE draft_id = ?",
            [draft_id]).fetchone()[0] == 3
        rows = corpus.execute(
            "SELECT count(*), "
            "  sum(CASE WHEN had_owner THEN 1 ELSE 0 END), "
            "  sum(CASE WHEN autodrafted THEN 1 ELSE 0 END), "
            "  count(*) FILTER (WHERE autodrafted IS NULL) "
            "FROM draft_log_pick WHERE draft_id = ?", [draft_id]).fetchone()
        # Four owned seats x 16 rounds, and not one unknown pick left.
        assert rows == (TRACE_PICKS, 64, TRACE_AUTO_WITH_OWNERS, 0)
        # The subset claim `fit_prior` leans on: an unowned seat is ALWAYS
        # autodrafted, so excluding `autodrafted IS TRUE` already excludes
        # every computer seat and no second filter is needed.
        assert corpus.execute(
            "SELECT count(*) FROM draft_log_pick WHERE draft_id = ? "
            "AND had_owner IS FALSE AND NOT autodrafted",
            [draft_id]).fetchone()[0] == 0
    finally:
        corpus.close()


def test_a_draft_with_no_census_stores_nulls_rather_than_guesses(tmp_path):
    """The mTeam read is best-effort and a room can refuse it. NULL is the
    honest answer for that draft -- and `human_seats` is NULL too, so a later
    query can tell "nobody was in the room" from "nobody asked"."""
    timeline = mf.draft_timeline(_trace_listener().events)
    record, _ = mf.build_record(
        timeline, _tool_for(timeline), _mock_settings(), league_id="999",
        season=2026, my_slot=2, started_at=None, teams=TRACE_TEAMS,
        rounds=TRACE_ROUNDS, owners={}, my_team_id=2)
    assert record.human_seats is None
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    try:
        draft_id = dl.record(corpus, record)
        assert corpus.execute(
            "SELECT count(*) FROM draft_log_pick "
            "WHERE draft_id = ? AND had_owner IS NULL",
            [draft_id]).fetchone()[0] == TRACE_PICKS
        assert corpus.execute(
            "SELECT human_seats FROM draft_log WHERE draft_id = ?",
            [draft_id]).fetchone()[0] is None
    finally:
        corpus.close()


def test_a_pick_with_no_pool_row_is_dropped_and_counted():
    """The same discipline as mock_backfill: never written half-filled, never
    silently lost -- a silent drop looks like a smaller corpus rather than
    like the join failure it is."""
    timeline = mf.draft_timeline(_trace_listener().events)
    tool = _tool_for(timeline)
    tool.pool_df = tool.pool_df.iloc[1:].reset_index(drop=True)
    record, dropped = mf.build_record(
        timeline, tool, _mock_settings(), league_id="999", season=2026,
        my_slot=1, started_at=None, teams=TRACE_TEAMS, rounds=TRACE_ROUNDS)
    assert dropped == 1
    assert len(record.picks) == TRACE_PICKS - 1
    # The surviving picks keep their own numbers: a gap stays a gap.
    assert 1 not in set(record.picks["pick_no"])
    assert list(record.picks["pick_no"]) == list(range(2, TRACE_PICKS + 1))


def test_taken_order_keeps_a_place_for_an_unresolvable_pick():
    """A pick nobody can name still consumed a turn; leaving it out of
    `taken_order` would hand every later pick to the wrong slot."""
    listener = DraftListener({})
    for frame in ["SELECTED 1 100 2\n", "SELECTED 2 999 4\n",
                  "SELECTED 3 300 0\n"]:
        listener.on_frame(frame)
    timeline = mf.draft_timeline(listener.events)
    tool = _tool_for([f for f in timeline if f.espn_id != 999])
    assert mf.taken_order_from(timeline, tool) == [0, None, 1]


def test_the_pick_timings_reach_the_corpus(tmp_path):
    """Built AND stored. `seconds_to_pick` and `clock_seconds` are two more
    columns added by ALTER, and a record that builds fine but does not store
    is the failure that would not surface until a refit found a column of
    NULLs -- which is exactly what every draft farmed before this landed
    already has, permanently."""
    timeline = mf.draft_timeline(_trace_listener(_ticking_clock()).events)
    record, _ = mf.build_record(
        timeline, _tool_for(timeline), _mock_settings(), league_id="999",
        season=2026, my_slot=3, started_at=None, teams=TRACE_TEAMS,
        rounds=TRACE_ROUNDS)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    try:
        draft_id = dl.record(corpus, record)
        stored = corpus.execute(
            "SELECT count(seconds_to_pick), min(seconds_to_pick), "
            "       max(seconds_to_pick), count(DISTINCT clock_seconds), "
            "       max(clock_seconds) "
            "FROM draft_log_pick WHERE draft_id = ?", [draft_id]).fetchone()
        assert stored == (TRACE_PICKS, TRACE_FIRST_SECONDS[0],
                          TRACE_SLOWEST_SECONDS, 1, TRACE_CLOCK_SECONDS)
    finally:
        corpus.close()


def test_an_untimed_pick_stores_null_rather_than_zero(tmp_path):
    """The one thing this column must never do is make "we could not time
    this pick" look like "he picked instantly"."""
    listener = _listener_at([100.0], ["SELECTED 1 100 2\n"])
    timeline = mf.draft_timeline(listener.events)
    record, _ = mf.build_record(
        timeline, _tool_for(timeline), _mock_settings(), league_id="999",
        season=2026, my_slot=1, started_at=None, teams=1, rounds=1)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    try:
        draft_id = dl.record(corpus, record)
        assert corpus.execute(
            "SELECT seconds_to_pick, clock_seconds FROM draft_log_pick "
            "WHERE draft_id = ?", [draft_id]).fetchone() == (None, None)
    finally:
        corpus.close()


# ---------------------------------------------------------------------------
# The board ESPN puts on the drafters' screens.
# ---------------------------------------------------------------------------


def _board_db(rows):
    """An in-memory database holding nothing but an `espn_adp` table."""
    import duckdb

    conn = duckdb.connect()
    conn.execute("CREATE TABLE espn_adp (espn_id BIGINT, espn_name VARCHAR, "
                 "position VARCHAR, team VARCHAR, espn_adp DOUBLE, "
                 "espn_ppr_rank BIGINT, espn_proj DOUBLE)")
    for row in rows:
        conn.execute("INSERT INTO espn_adp VALUES (?, ?, ?, ?, ?, ?, ?)", row)
    return conn


def test_the_pool_records_the_list_the_room_was_reading():
    """ESPN's rank, ESPN's projection and the bye, per player, per draft.

    The rank and the bye ride on the board already; the PROJECTION does not
    -- `_BOARD_COLUMNS` drops it, because the board's own `proj_points` is
    that number re-priced into the league's scoring. So it is joined from
    `espn_adp` on ESPN's own id, which is why this test carries a database at
    all.
    """
    board = pd.DataFrame({
        "player_id": ["p1", "p2"], "espn_ppr_rank": [1.0, 12.0],
        "bye": [7.0, 9.0], "espn_id": [4001.0, 4002.0]})
    conn = _board_db([(4001, "A", "RB", "DET", 1.2, 1, 310.5),
                      (4002, "B", "WR", "GB", 12.4, 12, 240.0)])
    try:
        got = mf.espn_pool_columns(board, conn)
    finally:
        conn.close()
    assert list(got["player_id"]) == ["p1", "p2"]
    assert list(got["espn_rank"]) == [1.0, 12.0]
    assert list(got["espn_proj"]) == [310.5, 240.0]
    assert list(got["bye"]) == [7.0, 9.0]


def test_a_player_espn_does_not_rank_gets_nulls_not_a_guess():
    """ESPN publishes 500 players and a draft pool is larger. An unranked
    player is a real and expected NULL -- filling him in with a sentinel
    would put a fake rank into the column the model reads first."""
    board = pd.DataFrame({
        "player_id": ["p1", "p9"], "espn_ppr_rank": [1.0, np.nan],
        "bye": [7.0, np.nan], "espn_id": [4001.0, np.nan]})
    conn = _board_db([(4001, "A", "RB", "DET", 1.2, 1, 310.5)])
    try:
        got = mf.espn_pool_columns(board, conn)
    finally:
        conn.close()
    assert got.loc[1, "player_id"] == "p9"
    assert pd.isna(got.loc[1, "espn_rank"])
    assert pd.isna(got.loc[1, "espn_proj"])
    assert pd.isna(got.loc[1, "bye"])


def test_a_database_with_no_espn_table_still_answers_with_the_column():
    """A fixture, or a snapshot taken before the ESPN job ran. The frame's
    SHAPE must not depend on what the database happened to hold, or a caller
    counting coverage gets a KeyError instead of a zero."""
    import duckdb

    conn = duckdb.connect()
    try:
        got = mf.espn_pool_columns(
            pd.DataFrame({"player_id": ["p1"], "espn_ppr_rank": [3.0],
                          "bye": [11.0], "espn_id": [4001.0]}), conn)
    finally:
        conn.close()
    assert list(got.columns) == ["player_id", "espn_rank", "espn_proj", "bye"]
    assert got["espn_rank"].iloc[0] == 3.0
    assert pd.isna(got["espn_proj"].iloc[0])


def test_the_pool_snapshot_round_trips_the_espn_columns(tmp_path):
    """Stored, not merely computed -- the same ALTER-added-column risk the
    pick timings carry."""
    pool = pd.DataFrame({"player_id": ["p1", "p2"], "position": ["RB", "WR"],
                         "team": ["DET", "GB"], "adp_rank": [1.0, 2.0],
                         "proj_points": [250.0, 240.0],
                         "espn_rank": [1.0, 12.0],
                         "espn_proj": [310.5, 240.0], "bye": [7, 9]})
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    try:
        draft_id = dl.record(corpus, dl.DraftRecord(
            source=dl.SOURCE_MOCK, season=2026, started_at="t", pool=pool))
        got = corpus.execute(
            "SELECT espn_rank, espn_proj, bye FROM draft_log_pool "
            "WHERE draft_id = ? ORDER BY player_id", [draft_id]).fetchall()
        assert got == [(1.0, 310.5, 7), (12.0, 240.0, 9)]
    finally:
        corpus.close()


# ---------------------------------------------------------------------------
# The epsilon-greedy pick policy.
# ---------------------------------------------------------------------------


def _fake_pool(n: int = 6) -> SimPool:
    """A tiny pool whose right answer can be worked out by hand: player 0 is
    the best QB, players 1..n-1 are receivers in descending value."""
    points = np.array([300.0 - 10.0 * i for i in range(n)])
    position = np.array(["QB"] + ["WR"] * (n - 1))
    ranks = np.arange(1, n + 1, dtype=float)
    return SimPool(
        player_id=np.array([f"p{i}" for i in range(n)]),
        norm=np.array([f"p{i}" for i in range(n)]),
        position=position, adp_rank=ranks, points=points,
        availability=np.full(n, 90.0), vor=points - 100.0,
        market_rank=ranks, age=np.full(n, 25.0),
        no_track_record=np.zeros(n), hype=np.zeros(n), trend=np.zeros(n))


def _empty_roster():
    return {"counts": {}, "indices": []}


# Pinned from `numpy.random.default_rng(seed).random()`: seed 3 draws 0.0856
# (below epsilon, the explore branch) and seed 0 draws 0.6370 (above it, the
# exploit branch). Both branches are exercised from a real generator rather
# than by overriding epsilon, because what is being tested is the branch the
# coin actually selects.
EXPLORE_SEED = 3
EXPLOIT_SEED = 0


def test_the_greedy_branch_takes_the_tools_own_top_candidate():
    pool = _fake_pool()
    settings = _mock_settings()
    available = np.arange(len(pool.player_id))
    idx = mf.choose_index(pool, available, _empty_roster(), settings,
                          _roster_cap(settings), gap=None, turns_left=None,
                          rng=np.random.default_rng(EXPLOIT_SEED))
    assert idx == 0          # the highest-value player on the board


def test_the_explore_branch_takes_someone_else():
    """Exploration is the point: a bot that always played its own top
    candidate would write a corpus that is a deterministic function of our own
    ranking."""
    pool = _fake_pool()
    settings = _mock_settings()
    available = np.arange(len(pool.player_id))
    idx = mf.choose_index(pool, available, _empty_roster(), settings,
                          _roster_cap(settings), gap=None, turns_left=None,
                          rng=np.random.default_rng(EXPLORE_SEED))
    assert idx in set(range(len(pool.player_id)))
    assert idx != 0


def test_the_explore_branch_stays_inside_the_roster_caps():
    """Uniform over CAP-LEGAL players, not over the board: drafting a fourth
    quarterback is not exploration, it is a roster the rules do not allow."""
    pool = _fake_pool()
    settings = _mock_settings()
    caps = _roster_cap(settings)
    roster = {"counts": {"QB": caps["QB"]}, "indices": []}
    available = np.arange(len(pool.player_id))
    for seed in range(40):
        rng = np.random.default_rng(seed)
        if float(np.random.default_rng(seed).random()) >= mf.EPSILON:
            continue
        idx = mf.choose_index(pool, available, roster, settings, caps,
                              gap=None, turns_left=None, rng=rng)
        assert pool.position[idx] != "QB"


def test_the_coin_is_flipped_even_when_there_is_nothing_to_pick():
    """A variable number of draws per pick would make the whole draft's random
    stream depend on which branch each earlier pick took, and no run could be
    reproduced from its seed."""
    pool = _fake_pool()
    settings = _mock_settings()
    rng = np.random.default_rng(EXPLOIT_SEED)
    assert mf.choose_index(pool, np.array([], dtype=int), _empty_roster(),
                           settings, _roster_cap(settings), gap=None,
                           turns_left=None, rng=rng) is None
    # One value consumed, so the next draw matches a fresh generator's second.
    fresh = np.random.default_rng(EXPLOIT_SEED)
    fresh.random()
    assert rng.random() == fresh.random()


def test_the_policy_only_ever_offers_players_it_was_given():
    """`available` is pre-filtered to players that can actually be SELECTed
    (a board row with no ESPN id can be ranked but not picked). The policy
    must not reach outside it."""
    pool = _fake_pool()
    settings = _mock_settings()
    available = np.array([3, 4, 5])
    for seed in range(20):
        idx = mf.choose_index(pool, available, _empty_roster(), settings,
                              _roster_cap(settings), gap=None, turns_left=None,
                              rng=np.random.default_rng(seed))
        assert idx in set(available.tolist())


# ---------------------------------------------------------------------------
# The lobby: which rooms the farm is allowed to join, and in what order.
# ---------------------------------------------------------------------------


NOW_MS = 1_700_000_000_000


def _room(**overrides) -> dict:
    """A directory row that passes every clause of the policy, so a test can
    break exactly one thing at a time."""
    row = {"leagueId": 1, "leagueSize": 8, "draftType": "SNAKE",
           "rankType": "PPR", "scoringItemStatIds": [53, 42],
           "full": False, "draftInProgress": False,
           "teamsJoined": 3, "experienceType": "PRO",
           "draftDate": NOW_MS + 300_000,
           "draftAvailableDate": NOW_MS + 210_000}
    row.update(overrides)
    return row


def test_the_policy_accepts_an_eight_team_ppr_snake_room():
    assert lobby.is_farmable(_room(), NOW_MS) is True


@pytest.mark.parametrize("override", [
    {"leagueSize": 10},                       # wrong league shape
    {"draftType": "AUCTION"},
    {"rankType": "STANDARD"},
    {"scoringItemStatIds": [42]},             # PPR by name, no receptions
    {"full": True},
    {"draftInProgress": True},
    {"draftDate": NOW_MS - 1},                # already started
    {"draftDate": NOW_MS + 5_000},            # too soon to join in time
    {"draftDate": NOW_MS + 3_600_000},        # too far off to hold a seat
    {"draftDate": None},
])
def test_the_policy_rejects_everything_else(override):
    assert lobby.is_farmable(_room(**override), NOW_MS) is False


def test_receptions_are_required_even_when_the_name_says_ppr():
    """`rankType` and stat id 53 agreed in every row observed, which is
    exactly why disagreement is worth hearing about: a room labelled PPR whose
    scoring omits receptions would fill the corpus with standard-scoring
    behaviour."""
    assert lobby.is_farmable(
        _room(rankType="PPR", scoringItemStatIds=[]), NOW_MS) is False


def test_the_fullest_room_wins():
    """The single most important ordering key: a room at 6/8 is six humans
    waiting for a draft, a room at 0/8 fills with ESPN's autodraft engine."""
    rooms = [_room(leagueId=1, teamsJoined=1),
             _room(leagueId=2, teamsJoined=6),
             _room(leagueId=3, teamsJoined=2)]
    assert [r["leagueId"] for r in lobby.rank_rooms(rooms, NOW_MS)] == [2, 3, 1]


def test_an_empty_room_is_skipped_rather_than_ranked_last():
    """Not a preference. A room with nobody in it is eight ESPN autodrafters
    plus us, and its 128 picks are ADP read back to us under a label saying
    "mock draft" -- which is worse than no draft, because a later reader
    cannot tell it apart from a room of people."""
    rooms = [_room(leagueId=1, teamsJoined=0),
             _room(leagueId=2, teamsJoined=0)]
    assert lobby.rank_rooms(rooms, NOW_MS) == []
    assert lobby.pick_room(rooms, NOW_MS) is None
    assert lobby.is_farmable(_room(teamsJoined=0), NOW_MS) is False
    assert lobby.is_farmable(_room(teamsJoined=1), NOW_MS) is True


def test_the_human_floor_is_a_knob_and_defaults_to_one():
    """Measured at 09:45 over 25 8-team PPR snake rooms: of the ten still
    joinable, four had one person, three had two, two had three and NONE had
    four. A hard floor of 4 would have joined nothing at all, all morning --
    so the default is the largest floor that does not starve the run, and the
    rest of the filtering happens at fit time where it is reversible."""
    assert lobby.MIN_TEAMS_JOINED == 1
    rooms = [_room(leagueId=1, teamsJoined=1),
             _room(leagueId=2, teamsJoined=3)]
    assert len(lobby.rank_rooms(rooms, NOW_MS)) == 2
    assert len(lobby.rank_rooms(rooms, NOW_MS, min_teams_joined=3)) == 1
    assert lobby.rank_rooms(rooms, NOW_MS, min_teams_joined=4) == []
    # And zero means "take anything", which is what the report below needs.
    assert len(lobby.rank_rooms([_room(teamsJoined=0)], NOW_MS,
                                min_teams_joined=0)) == 1


def test_the_lobby_report_says_whether_the_floor_is_the_problem():
    """`pick_room` returning None is two different situations wearing one
    face -- a poll between batches, or a lobby full of empty rooms -- and an
    unattended log that cannot tell them apart cannot say whether the floor
    is starving the run."""
    empty = [_room(leagueId=1, teamsJoined=0),
             _room(leagueId=2, teamsJoined=0),
             _room(leagueId=3, leagueSize=12)]      # wrong shape entirely
    report = lobby.lobby_report(empty, NOW_MS)
    assert report == {"rows": 3, "open": 2, "best": 0, "size": 8}
    # Nothing of the right shape at all: `best` is None rather than 0, which
    # is the distinction the log turns into two different sentences.
    nothing = lobby.lobby_report([_room(leagueSize=12)], NOW_MS)
    assert nothing["open"] == 0 and nothing["best"] is None


def test_experience_breaks_a_tie_before_start_time():
    rooms = [_room(leagueId=1, experienceType="BEGINNER",
                   draftDate=NOW_MS + 120_000),
             _room(leagueId=2, experienceType="PRO",
                   draftDate=NOW_MS + 400_000),
             _room(leagueId=3, experienceType="EXPERT",
                   draftDate=NOW_MS + 300_000)]
    assert [r["leagueId"] for r in lobby.rank_rooms(rooms, NOW_MS)] == [3, 2, 1]


def test_a_room_already_played_is_not_ranked_again():
    """ESPN keeps a room in the directory after we take a seat in it -- with
    `teamsJoined` one higher, which would rank our own room top."""
    rooms = [_room(leagueId=7, teamsJoined=6), _room(leagueId=8, teamsJoined=1)]
    ranked = lobby.rank_rooms(rooms, NOW_MS, exclude={"7"})
    assert [r["leagueId"] for r in ranked] == [8]


def test_an_empty_lobby_is_an_ordinary_answer():
    assert lobby.pick_room([], NOW_MS) is None
    assert lobby.pick_room([_room(leagueSize=12)], NOW_MS) is None


def test_the_lobby_listing_refuses_a_body_that_is_not_a_list():
    """An empty lobby and an expired cookie look the same to a caller, and
    only one of them is fixed by waiting."""
    with pytest.raises(ValueError, match="expired"):
        lobby.list_mock_leagues(lambda url: '{"messages": ["unauthorized"]}',
                                2026)


def test_the_lobby_listing_returns_the_rows_verbatim():
    seen = {}

    def fetch(url):
        seen["url"] = url
        return json.dumps([_room(leagueId=5)])

    rows = lobby.list_mock_leagues(fetch, 2026)
    assert [r["leagueId"] for r in rows] == [5]
    assert seen["url"].endswith("/seasons/2026/leaguedirectory/MOCKDRAFT_LOBBY")


def test_the_invite_url_carries_join_true_and_an_escaped_swid():
    """`join=true` is required -- without it ESPN answers 400 with an HTML
    body -- and the SWID cookie's braces are not legal in a query string."""
    url = lobby.invite_url(42, "{ABC-DEF}", 2026)
    assert "join=true" in url
    assert "memberId=%7BABC-DEF%7D" in url
    assert url.startswith(lobby.WRITES_BASE)
    assert "/leagues/42/invites" in url


def test_join_reads_the_seat_espn_assigned():
    """Read, never assumed: a wrong team id mints a socket URL for somebody
    else's seat, and the failure shows up as picks that never land."""
    sent = {}

    def post(url, payload):
        sent["url"], sent["payload"] = url, payload
        return json.dumps([{"isDeleted": False, "teamId": 7}])

    assert lobby.join(post, 42, "{ABC}", 2026) == 7
    assert sent["payload"] == [{"teamId": -1}]      # a JSON ARRAY, not an object


def test_join_raises_when_espn_names_no_seat():
    with pytest.raises(ValueError, match="did not say"):
        lobby.join(lambda url, payload: json.dumps([{"isDeleted": False}]),
                   42, "{ABC}", 2026)
    with pytest.raises(ValueError, match="did not say"):
        lobby.join(lambda url, payload: json.dumps([{"teamId": -1}]),
                   42, "{ABC}", 2026)


# ---------------------------------------------------------------------------
# The run loop: counting, resilience, and the wait for a room to open.
# ---------------------------------------------------------------------------


class _FakeConn:
    """Stands in for the read-only board connection `farm` opens and closes.
    Nothing in these tests reaches the board -- `play_draft` is replaced."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _live_room(**overrides) -> dict:
    """A room the policy accepts against the REAL clock, since `farm` calls
    `pick_room` with no `now_ms` of its own."""
    import time as _time
    now_ms = _time.time() * 1000
    return _room(draftDate=now_ms + 300_000,
                 draftAvailableDate=now_ms + 210_000, **overrides)


def _stub_farm_deps(monkeypatch, rooms, tmp_path):
    # Claims are files on disk, so every loop test gets its own directory --
    # otherwise a test would see the claims of the farm process the owner may
    # have running against the real `data/farm-claims`.
    monkeypatch.setattr(mf.claims, "CLAIM_DIR", str(tmp_path / "claims"))
    # Same reasoning one line up, for the live files `farm`'s `finally` now
    # deletes alongside the claim. (`_isolated_live_dir` already covers this
    # file-wide; repeated here so a reader of this helper sees both halves of
    # the on-disk state a loop test touches.)
    monkeypatch.setattr(mf, "LIVE_DIR", str(tmp_path / "farm-live"))
    monkeypatch.setattr(mf, "load_cookies",
                        lambda: {"SWID": "{X}", "espn_s2": "s2"})
    monkeypatch.setattr(mf, "http_fetch", lambda cookies: (lambda url: "[]"))
    monkeypatch.setattr(mf, "open_board_db",
                        lambda path=None, out=print: _FakeConn())
    monkeypatch.setattr(lobby, "list_mock_leagues",
                        lambda fetch, season: list(rooms))
    monkeypatch.setattr(mf.time, "sleep", lambda seconds: None)


def test_a_recorded_draft_is_counted_once(monkeypatch, tmp_path):
    """`recorded` is both a status name and the counter the loop's `while`
    reads. Counting it in one place made a run asked for N stop at N/2."""
    _stub_farm_deps(monkeypatch, [_live_room(leagueId=11),
                                  _live_room(leagueId=12)], tmp_path)
    seen = []

    def fake_play(conn, corpus_path, cookies, room, rng, season=2026,
                  out=print):
        seen.append(room["leagueId"])
        return {"status": "recorded", "picks": 128, "league_id": room["leagueId"]}

    monkeypatch.setattr(mf, "play_draft", fake_play)
    counts = mf.farm(2, corpus_path=str(tmp_path / "c.duckdb"),
                     out=lambda *a: None)
    assert counts["recorded"] == 2
    assert counts["picks"] == 256
    assert counts["by_status"] == {"recorded": 2}
    assert seen == [11, 12]


def test_one_rooms_crash_does_not_end_the_night(monkeypatch, tmp_path):
    """This runs unattended for hours. A single room raising -- ESPN
    unreachable for a moment, a room that vanished -- must cost that room and
    nothing else."""
    _stub_farm_deps(monkeypatch, [_live_room(leagueId=21, teamsJoined=6),
                                  _live_room(leagueId=22, teamsJoined=1)],
                    tmp_path)

    def fake_play(conn, corpus_path, cookies, room, rng, season=2026,
                  out=print):
        if room["leagueId"] == 21:
            raise RuntimeError("ESPN said no")
        return {"status": "recorded", "picks": 128, "league_id": 22}

    monkeypatch.setattr(mf, "play_draft", fake_play)
    counts = mf.farm(1, corpus_path=str(tmp_path / "c.duckdb"),
                     out=lambda *a: None)
    assert counts["recorded"] == 1
    assert counts["by_status"] == {"crashed": 1, "recorded": 1}
    assert counts["attempted"] == 2


def test_a_room_is_never_attempted_twice(monkeypatch, tmp_path):
    """ESPN keeps a room in the directory after we take a seat, so without
    the exclusion the loop would re-join the room it just played."""
    _stub_farm_deps(monkeypatch, [_live_room(leagueId=31),
                                  _live_room(leagueId=32)], tmp_path)
    seen = []

    def fake_play(conn, corpus_path, cookies, room, rng, season=2026,
                  out=print):
        seen.append(room["leagueId"])
        return {"status": "incomplete", "league_id": room["leagueId"],
                "picks": 40}

    monkeypatch.setattr(mf, "play_draft", fake_play)
    # Both rooms fail, so the loop runs out of rooms; the lobby stub keeps
    # returning the same two and `pick_room` returns None once both are
    # excluded, which would spin forever -- so stop it after two attempts.
    calls = {"n": 0}

    def counted_sleep(seconds):
        calls["n"] += 1
        if calls["n"] > 6:
            raise KeyboardInterrupt

    monkeypatch.setattr(mf.time, "sleep", counted_sleep)
    counts = mf.farm(1, corpus_path=str(tmp_path / "c.duckdb"),
                     out=lambda *a: None)
    assert seen == [31, 32]
    assert counts["by_status"] == {"incomplete": 2}


def test_the_loop_waits_rather_than_filling_an_empty_room(monkeypatch,
                                                          tmp_path):
    """And says what it saw while waiting. An overnight run that only ever
    prints "nothing fits" cannot tell a lobby between batches from a lobby
    full of rooms nobody has joined, and the second is the one that means the
    floor is starving the run."""
    _stub_farm_deps(monkeypatch, [_live_room(leagueId=41, teamsJoined=0),
                                  _live_room(leagueId=42, teamsJoined=0)],
                    tmp_path)
    played = []
    monkeypatch.setattr(mf, "play_draft",
                        lambda *a, **k: played.append(1) or {
                            "status": "recorded", "picks": 128})
    lines = []

    def counted_sleep(seconds):
        if len(lines) > 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(mf.time, "sleep", counted_sleep)
    mf.farm(1, corpus_path=str(tmp_path / "c.duckdb"), out=lines.append)

    assert played == [], "an empty room is our bot drafting against ESPN's"
    waited = "\n".join(lines)
    assert "2 joinable" in waited
    assert "0/8" in waited and "floor is 1" in waited


def test_a_room_another_farm_process_holds_is_left_alone(monkeypatch,
                                                         tmp_path):
    """Two processes rank identically, so without the claim they would take
    two of the eight seats in the SAME room -- and since `draft_id` hashes
    (league_id, picks) the second record would overwrite the first, leaving
    one bot seat labelled `my_slot` and the other indistinguishable from a
    person."""
    _stub_farm_deps(monkeypatch, [_live_room(leagueId=61, teamsJoined=6),
                                  _live_room(leagueId=62, teamsJoined=2)],
                    tmp_path)
    # Stand in for the other process: it got there first and holds 61.
    assert mf.claims.claim(61) is True
    seen = []

    def fake_play(conn, corpus_path, cookies, room, rng, season=2026,
                  out=print):
        seen.append(room["leagueId"])
        # The room we DID take is claimed while we are in it.
        assert "62" in mf.claims.claimed()
        return {"status": "recorded", "picks": 128, "league_id": 62}

    monkeypatch.setattr(mf, "play_draft", fake_play)
    mf.farm(1, corpus_path=str(tmp_path / "c.duckdb"), out=lambda *a: None)
    assert seen == [62], "the fuller room was held, so the next one was taken"
    # Released on the way out, so the next poll (or the next process) may
    # have it; the other process's claim is untouched.
    assert mf.claims.claimed() == {"61"}


def test_a_claim_is_released_even_when_the_room_raises(monkeypatch, tmp_path):
    """A claim left behind by a process that is no longer in the room costs
    the next poll a room it could have played, and the TTL that would
    eventually retire it is ninety minutes long."""
    _stub_farm_deps(monkeypatch, [_live_room(leagueId=71, teamsJoined=4)],
                    tmp_path)

    def fake_play(conn, corpus_path, cookies, room, rng, season=2026,
                  out=print):
        raise RuntimeError("ESPN said no")

    monkeypatch.setattr(mf, "play_draft", fake_play)
    calls = {"n": 0}

    def counted_sleep(seconds):
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(mf.time, "sleep", counted_sleep)
    mf.farm(1, corpus_path=str(tmp_path / "c.duckdb"), out=lambda *a: None)
    assert mf.claims.claimed() == set()


def test_the_human_floor_is_passed_through_from_the_caller(monkeypatch,
                                                           tmp_path):
    """An evening run can afford to be pickier than a 5am one -- ESPN's own
    recommended draft times are 12:00, 17:00 and 20:00 -- so the floor is an
    argument rather than a constant."""
    _stub_farm_deps(monkeypatch, [_live_room(leagueId=81, teamsJoined=2)],
                    tmp_path)
    played = []
    monkeypatch.setattr(mf, "play_draft",
                        lambda *a, **k: played.append(1) or {
                            "status": "recorded", "picks": 128})
    lines = []

    def counted_sleep(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(mf.time, "sleep", counted_sleep)
    mf.farm(1, corpus_path=str(tmp_path / "c.duckdb"), min_humans=3,
            out=lines.append)
    assert played == []
    assert "floor is 3" in "\n".join(lines)


def test_the_loop_waits_for_the_room_to_open_before_connecting():
    """ESPN's draft socket answers HTTP 500 for a room that has not opened
    yet -- a failure that reads exactly like a bad token. `draftAvailableDate`
    is ESPN's own answer for when that stops being true."""
    import time as _time
    slept = []
    room = {"draftAvailableDate": _time.time() * 1000 + 30_000,
            "draftDate": _time.time() * 1000 + 120_000}
    real_sleep = mf.time.sleep
    try:
        mf.time.sleep = lambda seconds: slept.append(seconds)
        mf._wait_until_available(room, lambda *a: None)
    finally:
        mf.time.sleep = real_sleep
    assert len(slept) == 1
    assert 30 < slept[0] < 40      # 30s plus the margin, minus test overhead


def test_a_room_that_is_already_open_is_not_waited_for():
    import time as _time
    slept = []
    room = {"draftAvailableDate": _time.time() * 1000 - 10_000,
            "draftDate": _time.time() * 1000 + 60_000}
    real_sleep = mf.time.sleep
    try:
        mf.time.sleep = lambda seconds: slept.append(seconds)
        mf._wait_until_available(room, lambda *a: None)
    finally:
        mf.time.sleep = real_sleep
    assert slept == []


def test_the_open_time_falls_back_to_ninety_seconds_before_the_draft():
    """Every directory row observed carried `draftAvailableDate` ~90s before
    `draftDate`; a row without one is not a reason to connect too early."""
    import time as _time
    slept = []
    room = {"draftDate": _time.time() * 1000 + 200_000}
    real_sleep = mf.time.sleep
    try:
        mf.time.sleep = lambda seconds: slept.append(seconds)
        mf._wait_until_available(room, lambda *a: None)
    finally:
        mf.time.sleep = real_sleep
    assert len(slept) == 1
    assert 110 < slept[0] < 120    # 200s - 90s lead + 5s margin


def test_a_far_future_open_time_is_clamped_rather_than_slept_through():
    """`pick_room` bounds `draftDate`, but `draftAvailableDate` is a
    different field and ESPN's value for it is taken on trust. The sleep is
    uninterruptible, so one malformed date would hold a seat for hours and an
    overnight run asked for five drafts would return zero."""
    import time as _time
    slept = []
    room = {"draftAvailableDate": _time.time() * 1000 + 6 * 3600 * 1000,
            "draftDate": _time.time() * 1000 + 6 * 3600 * 1000}
    real_sleep = mf.time.sleep
    try:
        mf.time.sleep = lambda seconds: slept.append(seconds)
        mf._wait_until_available(room, lambda *a: None)
    finally:
        mf.time.sleep = real_sleep
    assert slept == [mf.MAX_AVAILABLE_WAIT_SECONDS]
    assert mf.MAX_AVAILABLE_WAIT_SECONDS <= lobby.MAX_LEAD_SECONDS + 60


# ---------------------------------------------------------------------------
# The overnight log: what it must never contain, and what it must not spin on.
# ---------------------------------------------------------------------------


# A realistically shaped SWID. The cookie's real value is a brace-wrapped
# GUID, and the test uses one because the scrubber's first line of defence is
# that shape -- a placeholder like "{X}" would pass a test the real cookie
# would fail.
FAKE_SWID = "{8491403C-1CE2-4C9E-9E45-6D3D9F8A11B2}"


class _FakeHTTPError(Exception):
    """Shaped like httpx's `HTTPStatusError` in the two ways that matter: it
    carries `request`/`response`, and its `str()` is prose wrapped around the
    whole request URL. That second property is the leak -- the URL is the
    invite POST, and the invite POST carries `memberId=<the SWID>`."""

    def __init__(self, url, status):
        super().__init__(f"Client error '{status}' for url '{url}'")
        self.request = type("R", (), {"url": url})()
        self.response = type("S", (), {"status_code": status})()


def test_a_join_failure_prints_the_status_code_and_not_the_login(monkeypatch):
    """A room filling between the directory read and the join is an ORDINARY
    race -- ESPN answers 400 -- so this line is one the overnight log prints
    on a normal night. It used to carry the percent-encoded SWID, which is
    the sole credential on the draft-socket handshake, into the one artifact
    of an unattended run that anybody reads (and pastes) in the morning."""
    url = lobby.invite_url(4242, FAKE_SWID, 2026)
    monkeypatch.setattr(lobby, "http_poster", lambda cookies: None)

    def refuse(post, league_id, swid, season):
        raise _FakeHTTPError(url, "400 Bad Request")

    monkeypatch.setattr(lobby, "join", refuse)
    lines = []
    result = mf.play_draft(None, None, {"SWID": FAKE_SWID, "espn_s2": "s2"},
                           _live_room(leagueId=4242), np.random.default_rng(0),
                           season=2026, out=lines.append)

    assert result["status"] == "join_failed"
    printed = "\n".join(lines)
    assert "400" in printed                       # what a reader needs
    assert FAKE_SWID not in printed
    assert "8491403C" not in printed              # nor any part of it
    assert "%7B" not in printed                   # nor the escaped form


def test_every_line_the_farm_prints_goes_through_the_scrubber(monkeypatch):
    """The guarantee is about the OUT CHANNEL, not about the call sites. A
    future line formatting an exception -- or the bare `traceback.format_exc`
    the run loop already prints -- must be covered without its author having
    to know the scrubber exists."""
    _stub_farm_deps(monkeypatch, [_live_room(leagueId=51)],
                    Path(mf.tempfile.mkdtemp()))
    lines = []

    def fake_play(conn, corpus_path, cookies, room, rng, season=2026,
                  out=print):
        # Nothing here redacts anything; `farm` wrapped `out` before we got it.
        out(f"pretending ESPN said no for {lobby.invite_url(51, FAKE_SWID, 2026)}")
        return {"status": "recorded", "picks": 128, "league_id": 51}

    monkeypatch.setattr(mf, "play_draft", fake_play)
    mf.farm(1, corpus_path=str(Path(mf.tempfile.mkdtemp()) / "c.duckdb"),
            out=lines.append)

    printed = "\n".join(lines)
    assert "memberId=" in printed                 # the URL is still readable
    assert "8491403C" not in printed
    assert "%7B" not in printed


def test_the_socket_url_form_of_the_login_is_scrubbed_too():
    """The invite POST is not the only URL carrying it: the draft socket's
    own JOIN puts the SWID in `4=` unescaped and again inside `5=`, and any
    handshake failure formats that URL into `_connect_session`'s log line."""
    from pipeline.draft_socket import socket_url
    scrubbed = redact.redact(
        f"rejected: {socket_url(1, 2, FAKE_SWID, 999)}")
    assert "8491403C" not in scrubbed
    assert scrubbed.count(redact.SWID_PLACEHOLDER) == 2   # `4=` and inside `5=`


def test_a_login_that_is_not_guid_shaped_is_scrubbed_by_value():
    """The pattern layer only knows the shapes ESPN has actually used. The
    literal-value layer is what makes this a fact about the credential rather
    than a bet on a regex -- `load_cookies` registers both cookies the moment
    the saved login is read."""
    odd = "not-a-guid-at-all-0000"
    redact.remember_secret(odd)
    assert odd not in redact.redact(f"failed for memberId={odd}&join=true")


def test_redacting_an_already_redacted_line_changes_nothing():
    """`farm` wraps `out` and then hands it to `play_draft`, which wraps it
    again. Double-wrapping has to be a no-op, not `memberId=<swid><swid>`."""
    once = redact.redact(f"POST {lobby.invite_url(7, FAKE_SWID, 2026)}")
    assert redact.redact(once) == once


class _FakeListener:
    """Enough of `DraftListener` for the play loop: it is our turn, the
    socket has named our team, and no pick ever lands."""

    def __init__(self, team_id):
        self.events = []
        self.my_team_id = team_id
        self.on_the_clock = team_id
        self.my_autodraft = None
        self.selected_espn_ids = set()


class _FakeSession:
    """A session whose socket is published (so the play loop's guard passes)
    but whose sends fail -- exactly the state `draft_socket` leaves behind
    while it is reconnecting, since `_Session.socket` is never set back to
    None once published."""

    def __init__(self, listener):
        self.listener = listener
        self.socket = object()
        self.error = None
        self.stopped = False

    def alive(self):
        return self.error is None

    def stop(self):
        self.stopped = True


class _DeadHandle:
    """A socket handle with nothing attached -- what draft_socket leaves
    behind between a drop and the next connect, and what a dead seat
    leaves behind for good."""

    def alive(self):
        return False


class _DeadSession(_FakeSession):
    def __init__(self, listener):
        super().__init__(listener)
        self.socket = _DeadHandle()


def test_a_socket_dead_for_a_minute_gets_a_fresh_token_then_the_room_is_abandoned(
        monkeypatch):
    """Observed live: a seat whose socket cycled for twenty minutes (JOIN
    accepted, greeting, close, no picks replayed, every SELECT refused) with
    the listener thread alive throughout -- so neither draft_socket's own
    give-up nor the idle timer fired, and ESPN autodrafted the rest of the
    seat under our name. The loop must notice a handle that has had nothing
    attached for SOCKET_DEAD_SECONDS, stop that session and connect a fresh
    one; and when the fresh one is just as dead, give the room up.
    """
    settings = _mock_settings()
    listener = _FakeListener(42)
    sessions = [_DeadSession(listener), _DeadSession(listener)]
    connects = []

    def connect(*args, **kwargs):
        connects.append(args)
        return sessions[len(connects) - 1]

    tool = mf.Tool(board=pd.DataFrame(), pool=_fake_pool(), pool_df=pd.DataFrame(),
                   crosswalk={}, espn_by_index=[], index_by_player={},
                   sendable=np.ones(6, dtype=bool))

    monkeypatch.setattr(lobby, "http_poster", lambda cookies: None)
    monkeypatch.setattr(lobby, "join",
                        lambda post, league_id, swid, season: 42)
    monkeypatch.setattr(mf, "http_fetch", lambda cookies: (lambda url: ""))
    monkeypatch.setattr(mf, "fetch_league_settings",
                        lambda fetch, league_id, season: {"raw": True})
    monkeypatch.setattr(mf.league, "from_espn", lambda raw: settings)
    monkeypatch.setattr(mf, "build_tool", lambda conn, s: tool)
    monkeypatch.setattr(mf, "DraftListener", lambda crosswalk: listener)
    monkeypatch.setattr(mf, "_wait_until_available", lambda room, out: None)
    monkeypatch.setattr(mf, "_connect_session", connect)
    monkeypatch.setattr(mf, "_make_pick", lambda *a, **k: False)
    monkeypatch.setattr(mf, "SOCKET_DEAD_SECONDS", 0.0)
    monkeypatch.setattr(mf, "MAX_SOCKET_RECONNECTS", 1)

    loops = {"n": 0}

    def bounded_sleep(seconds):
        loops["n"] += 1
        assert loops["n"] < 50, "the loop never gave the dead room up"

    monkeypatch.setattr(mf.time, "sleep", bounded_sleep)

    said = []
    result = mf.play_draft(None, None, {"SWID": FAKE_SWID, "espn_s2": "s2"},
                           _live_room(leagueId=61), np.random.default_rng(0),
                           season=2026, out=lambda line: said.append(line))

    assert result["status"] == "incomplete"
    # One fresh session after the first dead minute, then the room given up
    # when that one is dead too -- never a third.
    assert len(connects) == 2
    assert sessions[0].stopped and sessions[1].stopped
    assert any("reconnecting with a fresh token (1/1)" in line for line in said)
    assert any("abandoning this room" in line for line in said)


def test_a_failed_pick_still_sleeps_before_the_loop_polls_again(monkeypatch):
    """The pick branch used to `continue` past the poll sleep.

    `_make_pick` returns False IMMEDIATELY when the send raises, and a send
    raises synchronously for as long as `draft_socket` has the socket
    detached for a reconnect -- routine, and it takes seconds. With no sleep
    on that path the loop re-ran `draft_timeline` over every event,
    `_seed_rosters` over every pick so far and `_greedy_choice` over ~250
    candidates thousands of times, at 100% of a core, fighting the listener
    thread for the GIL at exactly the moment our pick was on the clock.
    """
    settings = _mock_settings()
    listener = _FakeListener(42)
    session = _FakeSession(listener)
    tool = mf.Tool(board=pd.DataFrame(), pool=_fake_pool(), pool_df=pd.DataFrame(),
                   crosswalk={}, espn_by_index=[], index_by_player={},
                   sendable=np.ones(6, dtype=bool))

    monkeypatch.setattr(lobby, "http_poster", lambda cookies: None)
    monkeypatch.setattr(lobby, "join",
                        lambda post, league_id, swid, season: 42)
    monkeypatch.setattr(mf, "http_fetch", lambda cookies: (lambda url: ""))
    monkeypatch.setattr(mf, "fetch_league_settings",
                        lambda fetch, league_id, season: {"raw": True})
    monkeypatch.setattr(mf.league, "from_espn", lambda raw: settings)
    monkeypatch.setattr(mf, "build_tool", lambda conn, s: tool)
    monkeypatch.setattr(mf, "DraftListener", lambda crosswalk: listener)
    monkeypatch.setattr(mf, "_wait_until_available", lambda room, out: None)
    monkeypatch.setattr(mf, "_connect_session",
                        lambda *a, **k: session)

    picks = {"n": 0}

    def failed_pick(*args, **kwargs):
        picks["n"] += 1
        return False                     # the send raised; nothing was picked

    monkeypatch.setattr(mf, "_make_pick", failed_pick)

    slept = []

    def counted_sleep(seconds):
        slept.append(seconds)
        if len(slept) >= 5:
            # Stand in for the socket thread noticing it is dead, so the
            # loop has an ordinary way out of this test.
            session.error = RuntimeError("socket gave up")

    monkeypatch.setattr(mf.time, "sleep", counted_sleep)

    result = mf.play_draft(None, None, {"SWID": FAKE_SWID, "espn_s2": "s2"},
                           _live_room(leagueId=61), np.random.default_rng(0),
                           season=2026, out=lambda *a: None)

    assert result["status"] == "incomplete"
    # The point: one poll sleep per failed pick, not zero.
    assert picks["n"] == len(slept) == 5
    assert all(s == mf.LOOP_POLL_SECONDS for s in slept)
    assert session.stopped


# ---------------------------------------------------------------------------
# Live progress: what the farm publishes while a draft is still running.
# ---------------------------------------------------------------------------
#
# The corpus only learns about a draft when it FINISHES, and the web API is a
# separate process, so for the forty minutes a room is being played these
# files are the only evidence outside this process that it exists at all.
# Everything below drives the real module against a real directory
# (`_isolated_live_dir`), with the clock passed in rather than mocked --
# `write_live`, `live_drafts` and `_live_is_stale` all take `now`, which is
# what makes a ninety-minute TTL testable in a millisecond.


def _live_timeline(n: int, teams: int = TRACE_TEAMS):
    """`n` picks of a snake, one per seat in order, as the socket reports it.

    ESPN team ids are 10 + slot rather than the slot itself, because `owners`
    is keyed by team id and the published seats by slot: an implementation
    that confused the two would pass against equal numbers.
    """
    slots = snake_slots(teams, TRACE_ROUNDS)
    return [mf.PickFrame(i, 10 + slots[i - 1], 900 + i, False)
            for i in range(1, n + 1)]


def _live_tool(n: int = 8) -> mf.Tool:
    """A Tool that can name the players `_live_timeline` picks. Only
    `crosswalk` is read by `live_payload`; the rest is scaffolding."""
    return mf.Tool(board=None, pool=None, pool_df=None,
                   crosswalk={900 + i: f"p{i}" for i in range(1, n + 1)},
                   espn_by_index=[], index_by_player={}, sendable=None)


def _live_owners(teams: int = TRACE_TEAMS) -> dict:
    return {10 + slot: human for slot, human in TRACE_OWNERS.items()}


def test_the_live_file_names_the_same_draft_the_corpus_will():
    """THE WHOLE POINT OF NOT USING THE CONTENT HASH. `mock_backfill` ids a
    draft by hashing its picks, which changes on every pick; this one is
    (league, season, started_at), which does not. So a page open on a live
    board keeps its draft when the last pick lands and the corpus row appears
    -- if these two ever disagreed, every finished draft would 404 the link
    that was watching it."""
    from datetime import datetime, timezone
    started = datetime(2026, 8, 23, 10, 4, tzinfo=timezone.utc)
    timeline = mf.draft_timeline(_trace_listener().events, TRACE_OWNERS)
    tool = _tool_for(timeline)

    live = mf.live_payload(timeline, tool, "1084102871", 2026, TRACE_TEAMS,
                           TRACE_ROUNDS, 5, started, owners=TRACE_OWNERS)
    record, _dropped = mf.build_record(
        timeline, tool, _mock_settings(), league_id="1084102871", season=2026,
        my_slot=5, started_at=started, teams=TRACE_TEAMS, rounds=TRACE_ROUNDS,
        owners=TRACE_OWNERS)

    assert live["draft_id"] == record.resolved_id()
    assert live["started_at"] == "2026-08-23T10:04:00Z"


def test_the_live_file_carries_every_pick_so_far_at_its_own_slot():
    timeline = _live_timeline(9)
    payload = mf.live_payload(timeline, _live_tool(9), "111", 2026,
                              TRACE_TEAMS, TRACE_ROUNDS, 3, None,
                              owners=_live_owners(), my_team_id=13)

    slots = snake_slots(TRACE_TEAMS, TRACE_ROUNDS)
    assert [p["pick_no"] for p in payload["picks"]] == list(range(1, 10))
    assert [p["slot"] for p in payload["picks"]] == slots[:9]
    assert [p["player_id"] for p in payload["picks"]] == [
        f"p{i}" for i in range(1, 10)]
    assert payload["teams"] == TRACE_TEAMS and payload["rounds"] == TRACE_ROUNDS
    assert payload["my_slot"] == 3
    # Our own seat is not counted, exactly as the corpus head counts it.
    assert payload["human_seats"] == mf.human_seats(_live_owners(), 13)


def test_a_seat_nobody_has_drafted_from_yet_is_unknown_not_empty():
    """`owners` is keyed by ESPN team id and the page draws slots, so the two
    are joined through the picks themselves -- which means in round one most
    of the room has no answer yet. Calling those seats "no owner" would paint
    a room full of people as ESPN's own padding."""
    payload = mf.live_payload(_live_timeline(3), _live_tool(3), "111", 2026,
                              TRACE_TEAMS, TRACE_ROUNDS, None, None,
                              owners=_live_owners())
    seats = {s["slot"]: s["had_owner"] for s in payload["seats"]}
    assert len(seats) == TRACE_TEAMS
    assert [seats[i] for i in (1, 2, 3)] == [TRACE_OWNERS[1], TRACE_OWNERS[2],
                                             TRACE_OWNERS[3]]
    assert all(seats[i] is None for i in range(4, TRACE_TEAMS + 1))


def test_a_draft_with_no_owner_census_publishes_nulls_rather_than_guesses():
    """The mTeam read failing is a real failure mode `play_draft` says out
    loud. It means unknown, and unknown is not False."""
    payload = mf.live_payload(_live_timeline(8), _live_tool(8), "111", 2026,
                              TRACE_TEAMS, TRACE_ROUNDS, 1, None, owners=None)
    assert all(s["had_owner"] is None for s in payload["seats"])
    assert payload["human_seats"] is None


def test_an_unresolvable_pick_keeps_its_place_in_the_numbering():
    """A pick the crosswalk cannot name still CONSUMED a turn. Dropping it
    would slide every later pick onto the wrong seat -- the same rule
    `taken_order_from` follows for the same reason."""
    timeline = [mf.PickFrame(1, 11, 901, False),
                mf.PickFrame(2, 12, None, False),
                mf.PickFrame(3, 13, 903, True)]
    payload = mf.live_payload(timeline, _live_tool(3), "111", 2026,
                              TRACE_TEAMS, TRACE_ROUNDS, 1, None, owners=None)
    assert [p["pick_no"] for p in payload["picks"]] == [1, 2, 3]
    assert [p["player_id"] for p in payload["picks"]] == ["p1", None, "p3"]
    assert [p["autodrafted"] for p in payload["picks"]] == [False, False, True]


def test_the_file_is_replaced_whole_and_leaves_no_debris(_isolated_live_dir):
    """The rename is the whole mechanism: this file is rewritten after every
    one of 128 picks, and a reader that caught one half-done would draw a
    truncated board."""
    first = mf.live_payload(_live_timeline(1), _live_tool(), "111", 2026,
                            TRACE_TEAMS, TRACE_ROUNDS, 1, None, owners=None)
    mf.write_live(first)
    second = mf.live_payload(_live_timeline(5), _live_tool(), "111", 2026,
                             TRACE_TEAMS, TRACE_ROUNDS, 1, None, owners=None)
    mf.write_live(second)

    assert [p.name for p in sorted(_isolated_live_dir.iterdir())] == \
        ["111.json"]
    drafts = mf.live_drafts()
    assert len(drafts) == 1 and len(drafts[0]["picks"]) == 5
    assert drafts[0]["pid"] == os.getpid()


def test_a_live_file_whose_farm_is_dead_is_swept(_isolated_live_dir):
    """A farm killed mid-draft leaves its file behind, and a page that kept
    serving it would show a draft as live that stopped moving days ago --
    worse than showing nothing, because nothing about it looks wrong."""
    dead = 4_000_000
    if _pid_exists(dead):
        pytest.skip("pid 4000000 is in use on this machine")
    _isolated_live_dir.mkdir(parents=True, exist_ok=True)
    path = _isolated_live_dir / "111.json"
    path.write_text(json.dumps({"league_id": "111", "draft_id": "mock:x",
                                "pid": dead, "written_at": time.time()}))
    assert mf.live_drafts() == []
    assert not path.exists()


def test_a_live_file_older_than_the_ttl_is_swept_even_if_the_pid_is_alive():
    """The backstop for a pid recycled onto some unrelated process, which is
    rare but not impossible on a machine that has been up for weeks -- the
    same TTL, and the same reason for it, as a room claim."""
    now = time.time()
    payload = mf.live_payload(_live_timeline(1), _live_tool(), "111", 2026,
                              TRACE_TEAMS, TRACE_ROUNDS, 1, None, owners=None)
    mf.write_live(payload, now=now - mf.LIVE_TTL_SECONDS - 1)
    assert mf.live_drafts(now=now) == []
    mf.write_live(payload, now=now - mf.LIVE_TTL_SECONDS + 60)
    assert len(mf.live_drafts(now=now)) == 1


def test_a_file_that_will_not_parse_retires_only_after_the_grace(
        _isolated_live_dir):
    """`os.replace` means a reader can never catch a half-written file, so an
    unreadable one is genuinely corrupt rather than mid-write -- but it is
    still aged off its mtime rather than swept on sight, because "unreadable
    therefore delete" is a rule that only ever gets more dangerous."""
    _isolated_live_dir.mkdir(parents=True, exist_ok=True)
    corrupt = _isolated_live_dir / "111.json"
    corrupt.write_text("not json at all")
    assert mf.live_drafts() == [] and corrupt.exists()

    old = time.time() - mf.LIVE_GRACE_SECONDS - 1
    os.utime(corrupt, (old, old))
    assert mf.live_drafts() == []
    assert not corrupt.exists()


def test_clearing_a_live_file_that_is_not_ours_does_nothing(
        _isolated_live_dir):
    """A process that stalled past the TTL can have had its room swept and
    re-entered by another farm. An unconditional unlink on the way out would
    then delete THAT process's live board."""
    _isolated_live_dir.mkdir(parents=True, exist_ok=True)
    other = os.getppid()
    assert other != os.getpid()
    path = _isolated_live_dir / "111.json"
    path.write_text(json.dumps({"league_id": "111", "pid": other,
                                "written_at": time.time()}))
    mf.clear_live("111")
    assert path.exists()

    mf.write_live(mf.live_payload(_live_timeline(1), _live_tool(), "111",
                                  2026, TRACE_TEAMS, TRACE_ROUNDS, 1, None))
    mf.clear_live("111")
    assert not path.exists()
    assert mf.live_drafts() == []


def test_clearing_a_room_we_never_published_is_harmless():
    mf.clear_live("999")                 # no file, no exception
    assert mf.live_drafts() == []


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_the_play_loop_publishes_once_per_pick_and_only_then(
        monkeypatch, _isolated_live_dir):
    """THE INTEGRATION THAT MATTERS: the loop wakes four times a second, and
    a room on a slow clock must not rewrite the same board a hundred times
    between picks. One write per pick, plus the one at zero picks that puts a
    room on the page before it has started.

    Also pins WHERE in the loop the write happens. `my_slot` is resolved from
    the socket a few lines above it, and a payload written before that
    resolution would carry the previous pick's answer -- None, on the pick
    that first identifies our seat.
    """
    import types

    from pipeline.espn_live import parse_frame

    settings = _mock_settings()
    listener = _FakeListener(42)
    listener.on_the_clock = None         # never our turn: no pick is attempted
    session = _FakeSession(listener)
    # `play_draft` prints the pool size on the way in, so unlike the payload
    # tests above this Tool needs a pool-shaped stand-in as well as a
    # crosswalk.
    tool = _live_tool(4)
    tool.pool = types.SimpleNamespace(player_id=[f"p{i}" for i in range(1, 5)])
    tool.sendable = np.ones(4, dtype=bool)

    monkeypatch.setattr(lobby, "http_poster", lambda cookies: None)
    monkeypatch.setattr(lobby, "join",
                        lambda post, league_id, swid, season: 42)
    monkeypatch.setattr(mf, "http_fetch", lambda cookies: (lambda url: ""))
    monkeypatch.setattr(mf, "fetch_league_settings",
                        lambda fetch, league_id, season: {"raw": True})
    monkeypatch.setattr(mf.league, "from_espn", lambda raw: settings)
    census = {**_live_owners(), 42: True}   # our own seat always has an owner
    monkeypatch.setattr(mf, "fetch_team_owners",
                        lambda fetch, league_id, season: census)
    monkeypatch.setattr(mf, "build_tool", lambda conn, s: tool)
    monkeypatch.setattr(mf, "DraftListener", lambda crosswalk: listener)
    monkeypatch.setattr(mf, "_wait_until_available", lambda room, out: None)
    monkeypatch.setattr(mf, "_connect_session", lambda *a, **k: session)

    published = []
    real_write = mf.write_live
    monkeypatch.setattr(mf, "write_live",
                        lambda payload, now=None: (published.append(payload),
                                                   real_write(payload, now))[1])

    # One SELECTED frame per poll. The third is OUR team, which is the first
    # moment the socket can name our slot (slot 3 of the snake).
    teams_by_pick = [11, 12, 42, 14]

    def counted_sleep(seconds):
        i = len(listener.events)
        if i < len(teams_by_pick):
            listener.events.append(
                parse_frame(f"SELECTED {teams_by_pick[i]} {901 + i}"))
        else:
            session.error = RuntimeError("socket gave up")

    monkeypatch.setattr(mf.time, "sleep", counted_sleep)

    result = mf.play_draft(None, None, {"SWID": FAKE_SWID, "espn_s2": "s2"},
                           _live_room(leagueId=71), np.random.default_rng(0),
                           season=2026, out=lambda *a: None)

    assert result["status"] == "incomplete" and result["picks"] == 4
    # Five writes: the empty board, then one per pick. Not one per poll.
    assert [len(p["picks"]) for p in published] == [0, 1, 2, 3, 4]
    assert [p["my_slot"] for p in published] == [None, None, None, 3, 3]
    assert published[-1]["league_id"] == "71"
    assert [p["slot"] for p in published[-1]["picks"]] == [1, 2, 3, 4]
    assert [p["player_id"] for p in published[-1]["picks"]] == [
        "p1", "p2", "p3", "p4"]
    # Still on disk: deleting it is `farm`'s job, in the same `finally` that
    # releases the room claim, not `play_draft`'s.
    assert (_isolated_live_dir / "71.json").exists()


def test_the_live_file_goes_when_the_room_claim_does(monkeypatch, tmp_path):
    """Both are removed in the same `finally`, on every path out -- a
    recorded draft, an abandoned room, a crash, a Ctrl-C. A crash between the
    last pick and here is exactly when the page must not keep showing a board
    that never moves again."""
    _stub_farm_deps(monkeypatch, [_live_room(leagueId=81, teamsJoined=6),
                                  _live_room(leagueId=82, teamsJoined=1)],
                    tmp_path)

    def fake_play(conn, corpus_path, cookies, room, rng, season=2026,
                  out=print):
        mf.write_live(mf.live_payload(_live_timeline(4), _live_tool(4),
                                      room["leagueId"], 2026, TRACE_TEAMS,
                                      TRACE_ROUNDS, 1, None))
        assert mf.live_drafts()          # published while the room is played
        if room["leagueId"] == 81:
            raise RuntimeError("ESPN said no")
        return {"status": "recorded", "picks": 128, "league_id": 82}

    monkeypatch.setattr(mf, "play_draft", fake_play)
    counts = mf.farm(1, corpus_path=str(tmp_path / "c.duckdb"),
                     out=lambda *a: None)

    # The crashed room and the recorded one, both cleaned up the same way.
    assert counts["by_status"] == {"crashed": 1, "recorded": 1}
    assert mf.live_drafts() == []
    assert mf.claims.claimed() == set()
