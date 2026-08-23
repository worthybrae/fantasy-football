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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import draft_log as dl
from pipeline import espn_mock_lobby as lobby
from pipeline import mock_farm as mf
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


def _trace_listener() -> DraftListener:
    """The real recorded draft, replayed through the real listener.

    Frames are filtered to the fantasydraft host: the capture also holds
    ESPN's unrelated bamgrid connection, whose frames are JSON blobs the
    draft protocol knows nothing about.
    """
    if not TRACE.exists():
        pytest.skip(f"{TRACE} is not present in this checkout")
    listener = DraftListener({})
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
    rooms = [_room(leagueId=1, teamsJoined=0),
             _room(leagueId=2, teamsJoined=6),
             _room(leagueId=3, teamsJoined=2)]
    assert [r["leagueId"] for r in lobby.rank_rooms(rooms, NOW_MS)] == [2, 3, 1]


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


def _stub_farm_deps(monkeypatch, rooms):
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
                                  _live_room(leagueId=12)])
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
                                  _live_room(leagueId=22, teamsJoined=1)])

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
                                  _live_room(leagueId=32)])
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
