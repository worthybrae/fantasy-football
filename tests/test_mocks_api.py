"""api.mocks: the page that lists every mock draft the farm is in or played.

WHAT THESE TESTS ARE ACTUALLY GUARDING. The listing and the board are display
code, and display code is usually cheap to get wrong and cheap to fix. Not
here: `made_by` is the entire reason the page exists. It is the difference
between "this corpus is 128 picks of ESPN's autodrafter reading its own
ranking back at us" and "this corpus is three people drafting", and every one
of its five answers is derived from two nullable columns whose nulls mean
DIFFERENT things (see `pipeline.draft_log.ensure_schema`). A branch that
quietly turns "nobody asked" into "a person picked" would not raise, would not
look wrong on screen, and would be believed.

Everything below runs against a real DuckDB corpus in `tmp_path` and a real
live directory in `tmp_path`, never the repo's own -- the owner may have three
farm processes writing both while this suite runs, and a test that opened the
real corpus read-write, or swept the real live directory, would interfere with
a draft that cannot be replayed. The board is the one thing stubbed: naming
players is `data/nfl.duckdb`'s job and a 250-player board build has nothing to
do with what is being tested here.
"""
import json
import os
import time

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.mocks import register_mock_routes
from pipeline import draft_log as dl
from pipeline import mock_farm as mf
from scoring.draft_sim import snake_slots

TEAMS = 4
ROUNDS = 2
SLOTS = snake_slots(TEAMS, ROUNDS)          # [1,2,3,4,4,3,2,1]
MY_SLOT = 2

# Which seats held a person when the room opened. Slot 3 is ESPN padding out
# an empty room; slot 2 is ours (we always have an owner, and we are a bot).
HAD_OWNER = {1: True, 2: True, 3: False, 4: True}

# Which picks ESPN's engine made. Slot 3 is on the engine throughout because
# nobody ever sat there; the seat-1 drafter wanders off after round one, which
# is the case `auto` exists to catch.
AUTODRAFTED = {1: False, 2: False, 3: True, 4: False,
               5: False, 6: True, 7: False, 8: True}

EXPECTED_MADE_BY = ["human", "us", "engine", "human",
                    "human", "engine", "us", "auto"]


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    """Never the real corpus and never the real live directory.

    Both are being written RIGHT NOW on the owner's machine by farm processes
    playing real drafts. The corpus is the one file in this project that
    cannot be rebuilt (`pipeline.draft_log`'s docstring), and sweeping a live
    file would make a draft in progress vanish off the page.
    """
    monkeypatch.setattr(dl, "CORPUS_PATH", str(tmp_path / "corpus.duckdb"))
    monkeypatch.setattr(mf, "LIVE_DIR", str(tmp_path / "farm-live"))
    return tmp_path


@pytest.fixture
def board(monkeypatch):
    """A four-player board, standing in for `data/nfl.duckdb`'s 250.

    Only the columns `api.live._board_cell` actually reads, with real values
    rather than zeros, so a test can assert that a NAME reached the cell
    rather than that a cell exists.
    """
    frame = pd.DataFrame([
        {"player_id": "p1", "name": "Ja'Marr Chase", "position": "WR",
         "team": "CIN", "bye": 10, "rank": 1, "tier": 1, "market_rank": 1.0,
         "espn_ppr_rank": 2, "vor": 90.5, "proj_scale": 1.0,
         "espn_proj": 340.0, "headshot": "http://x/1.png",
         "stats": {"ppg": 18.5, "points": 314.0}},
        {"player_id": "p2", "name": "Bijan Robinson", "position": "RB",
         "team": "ATL", "bye": 5, "rank": 2, "tier": 1, "market_rank": 3.0,
         "espn_ppr_rank": 1, "vor": 80.0, "proj_scale": 1.0,
         "espn_proj": 306.0, "headshot": None,
         "stats": {"ppg": 17.0, "points": 289.0}},
        {"player_id": "p3", "name": "CeeDee Lamb", "position": "WR",
         "team": "DAL", "bye": 7, "rank": 3, "tier": 1, "market_rank": 2.0,
         "espn_ppr_rank": 3, "vor": 75.0, "proj_scale": 1.0,
         "espn_proj": 289.0, "headshot": None,
         "stats": {"ppg": 16.0, "points": 272.0}},
        {"player_id": "p4", "name": "Malik Nabers", "position": "WR",
         "team": "NYG", "bye": 11, "rank": 4, "tier": 2, "market_rank": 6.0,
         "espn_ppr_rank": 5, "vor": 60.0, "proj_scale": 1.0,
         "espn_proj": 255.0, "headshot": None,
         "stats": {"ppg": 14.0, "points": 238.0}},
    ])
    monkeypatch.setattr("api.mocks._board_frame", lambda conn: frame)
    return frame


@pytest.fixture
def client(board):
    """The two endpoints on a bare app.

    `create_app` is deliberately not used: it opens `data/nfl.duckdb`
    read-write, which the owner's own dev server holds a lock on, and the
    board -- the only thing that connection is for here -- is stubbed anyway.
    """
    app = FastAPI()
    register_mock_routes(app, conn=None)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Fixture builders: a corpus draft and a live one.
# ---------------------------------------------------------------------------


def _picks_frame(player_ids, had_owner=HAD_OWNER, autodrafted=AUTODRAFTED,
                 draft_id="d"):
    rows = []
    for i, player_id in enumerate(player_ids, start=1):
        slot = SLOTS[i - 1]
        rows.append({
            "pick_no": i, "round": (i - 1) // TEAMS + 1, "slot": slot,
            "owner_key": dl.anonymous_key(draft_id, slot),
            "is_anonymous": True, "player_id": player_id,
            "position": "WR", "adp_rank": float(i), "proj_points": 200.0 - i,
            "autodrafted": None if autodrafted is None else autodrafted[i],
            "had_owner": None if had_owner is None else had_owner[slot],
        })
    return pd.DataFrame(rows)


def _record(recorded_at, league_id="111", my_slot=MY_SLOT,
            player_ids=("p1", "p2", "p3", "p4", "p1", "p2", "p3", "p4"),
            had_owner=HAD_OWNER, autodrafted=AUTODRAFTED, human_seats=3,
            started_at=None):
    """Write one completed mock into the temp corpus, return its draft_id."""
    corpus = dl.corpus_conn(dl.CORPUS_PATH)
    try:
        draft_id = dl.draft_id_for(dl.SOURCE_MOCK, league_id, 2026, started_at)
        record = dl.DraftRecord(
            source=dl.SOURCE_MOCK, league_id=league_id, season=2026,
            teams=TEAMS, rounds=ROUNDS, my_slot=my_slot,
            human_seats=human_seats, started_at=started_at,
            recorded_at=recorded_at, draft_id=draft_id,
            picks=_picks_frame(list(player_ids), had_owner, autodrafted,
                               draft_id))
        return dl.record(corpus, record)
    finally:
        corpus.close()


def _timeline(n, autodrafted=AUTODRAFTED):
    """`n` picks of the same draft as `_picks_frame`, as the farm sees them.

    ESPN team ids are deliberately NOT the slot numbers (10 + slot), because
    `owners` is keyed by team id and `seats` by slot: an implementation that
    confused the two would pass if they were equal.
    """
    frames = []
    for pick_no in range(1, n + 1):
        slot = SLOTS[pick_no - 1]
        frames.append(mf.PickFrame(pick_no, 10 + slot, 900 + pick_no,
                                   autodrafted[pick_no]))
    return frames


def _tool():
    """Only `crosswalk` is read by `live_payload`; the rest is scaffolding."""
    crosswalk = {900 + i: f"p{((i - 1) % 4) + 1}" for i in range(1, 9)}
    return mf.Tool(board=None, pool=None, pool_df=None, crosswalk=crosswalk,
                   espn_by_index=[], index_by_player={}, sendable=None)


def _owners() -> dict:
    """`HAD_OWNER` as ESPN reports it: keyed by team id, not by slot."""
    return {10 + slot: human for slot, human in HAD_OWNER.items()}


def _publish(n=3, league_id="222", my_slot=MY_SLOT, owners=...):
    """Publish `n` picks of a live draft, as `play_draft` would."""
    payload = mf.live_payload(
        _timeline(n), _tool(), league_id, 2026, TEAMS, ROUNDS, my_slot, None,
        owners=_owners() if owners is ... else owners,
        my_team_id=10 + MY_SLOT)
    mf.write_live(payload)
    return payload


# ---------------------------------------------------------------------------
# GET /api/mocks
# ---------------------------------------------------------------------------


def test_the_listing_is_empty_before_anything_has_been_farmed(client):
    """No corpus file at all is the ordinary state of a fresh install, not an
    error -- `_open_corpus` returns None rather than raising."""
    assert client.get("/api/mocks").json() == {"drafts": []}


def test_a_completed_draft_lists_its_real_shape(client):
    draft_id = _record(pd.Timestamp("2026-08-20 12:00:00"))
    row, = client.get("/api/mocks").json()["drafts"]
    assert row == {
        "id": draft_id, "league_id": "111", "status": "complete",
        "teams": 4, "rounds": 2, "picks_made": 8, "human_seats": 3,
        "my_slot": 2,
        # The corpus records when a draft was WRITTEN, never when it started
        # (the start time goes into the draft_id hash and nowhere else), so
        # the honest answer for a completed draft is null.
        "started_at": None, "recorded_at": "2026-08-20T12:00:00Z"}


def test_live_drafts_come_first_and_the_rest_newest_first(client):
    old = _record(pd.Timestamp("2026-08-18 09:00:00"), league_id="111")
    new = _record(pd.Timestamp("2026-08-21 09:00:00"), league_id="333")
    _publish(n=3, league_id="222")

    rows = client.get("/api/mocks").json()["drafts"]
    assert [r["status"] for r in rows] == ["live", "complete", "complete"]
    assert [r["id"] for r in rows][1:] == [new, old]
    assert rows[0]["league_id"] == "222"
    assert rows[0]["picks_made"] == 3


def test_a_live_draft_is_not_listed_twice_while_it_is_being_recorded(client):
    """The live file is deleted in `farm`'s `finally`, a beat AFTER
    `dl.record` lands, so for a second or two the draft is in both places.
    One row, not two -- and the live one, because it has the later state."""
    started = pd.Timestamp("2026-08-22 10:00:00").to_pydatetime()
    draft_id = _record(pd.Timestamp("2026-08-22 10:40:00"), league_id="444",
                       started_at=started)
    payload = mf.live_payload(_timeline(8), _tool(), "444", 2026, TEAMS,
                              ROUNDS, MY_SLOT, started,
                              owners=_owners(),
                              my_team_id=10 + MY_SLOT)
    mf.write_live(payload)

    # The two halves agree about identity -- that is what makes the dedupe
    # possible, and what keeps a page open on a live board from 404ing the
    # moment the draft finishes.
    assert payload["draft_id"] == draft_id
    rows = client.get("/api/mocks").json()["drafts"]
    assert [(r["id"], r["status"]) for r in rows] == [(draft_id, "live")]


# ---------------------------------------------------------------------------
# GET /api/mocks/{id}/board -- who made each pick
# ---------------------------------------------------------------------------


def test_a_completed_board_labels_every_pick_with_who_made_it(client):
    draft_id = _record(pd.Timestamp("2026-08-20 12:00:00"))
    board = client.get(f"/api/mocks/{draft_id}/board").json()

    assert board["active"] is True
    assert (board["teams"], board["rounds"]) == (TEAMS, ROUNDS)
    assert board["my_slot"] == MY_SLOT
    assert board["picks_made"] == 8
    # A finished draft has nobody on the clock.
    assert board["on_the_clock"] is None
    assert [c["made_by"] for c in board["cells"]] == EXPECTED_MADE_BY
    assert [c["overall"] for c in board["cells"]] == list(range(1, 9))
    assert [c["slot"] for c in board["cells"]] == SLOTS
    assert [c["round"] for c in board["cells"]] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_the_columns_carry_the_seat_labels(client):
    draft_id = _record(pd.Timestamp("2026-08-20 12:00:00"))
    columns = client.get(f"/api/mocks/{draft_id}/board").json()["columns"]
    assert [c["slot"] for c in columns] == [1, 2, 3, 4]
    assert [c["had_owner"] for c in columns] == [True, True, False, True]
    assert [c["is_me"] for c in columns] == [False, True, False, False]
    assert [c["team_name"] for c in columns] == [
        "Team 1", "Team 2", "Team 3", "Team 4"]


def test_a_draft_recorded_before_seat_labelling_is_honestly_unknown(client):
    """The 23 harvested drafts, and everything recorded before this feature.

    Their leagues 404 the moment the draft ends, so nobody can ever go back
    and ask ESPN who was in them: `had_owner` is null, `autodrafted` is null,
    `my_slot` is null, and every pick is `unknown`. Guessing "human" here
    would be the difference between a corpus of people and a corpus of ESPN's
    autodrafter, told with total confidence.
    """
    draft_id = _record(pd.Timestamp("2026-08-19 08:00:00"), league_id="555",
                       my_slot=None, had_owner=None, autodrafted=None,
                       human_seats=None)
    board = client.get(f"/api/mocks/{draft_id}/board").json()
    assert {c["made_by"] for c in board["cells"]} == {"unknown"}
    assert [c["had_owner"] for c in board["columns"]] == [None] * TEAMS
    assert [c["is_me"] for c in board["columns"]] == [False] * TEAMS
    assert board["my_slot"] is None


def test_a_seat_with_an_owner_but_no_autodraft_flag_stays_unknown(client):
    """`had_owner` true and `autodrafted` null is "the mTeam read worked and
    the socket never said" -- rare, and still not a licence to claim a person
    made the pick."""
    draft_id = _record(pd.Timestamp("2026-08-19 08:00:00"), league_id="666",
                       my_slot=None, autodrafted=None)
    board = client.get(f"/api/mocks/{draft_id}/board").json()
    by_slot = {c["slot"]: c["made_by"] for c in board["cells"]}
    assert by_slot[1] == "unknown"          # owner known, flag missing
    assert by_slot[3] == "engine"           # no owner: the flag adds nothing


def test_our_own_seat_is_never_called_human(client):
    """The bot plays epsilon-greedy and joined with a real member id, so its
    seat has an owner and its picks are not autodrafted -- every other branch
    would call it a person, and a fit that believed that would be learning our
    own exploration noise as somebody's tendencies."""
    draft_id = _record(pd.Timestamp("2026-08-20 12:00:00"))
    cells = client.get(f"/api/mocks/{draft_id}/board").json()["cells"]
    ours = [c["made_by"] for c in cells if c["slot"] == MY_SLOT]
    assert ours == ["us", "us"]


# ---------------------------------------------------------------------------
# The board's players
# ---------------------------------------------------------------------------


def test_the_cells_are_named_from_the_board(client):
    draft_id = _record(pd.Timestamp("2026-08-20 12:00:00"))
    board = client.get(f"/api/mocks/{draft_id}/board").json()
    first = board["cells"][0]["player"]
    assert first["name"] == "Ja'Marr Chase"
    assert first["position"] == "WR" and first["team"] == "CIN"
    assert first["bye"] == 10 and first["tier"] == 1
    assert first["market_rank"] == 1.0 and first["espn_ppr_rank"] == 2
    assert first["last_ppg"] == 18.5 and first["headshot"] == "http://x/1.png"
    # overall - market_rank: taken at 1 with a consensus rank of 1.
    assert first["value"] == 0.0
    # Taken 4th with a consensus rank of 6 -- three picks of value.
    assert board["cells"][3]["player"]["value"] == pytest.approx(-2.0)
    assert board["unresolved"] == 0


def test_a_player_the_board_no_longer_carries_is_named_by_his_id_and_counted(
        client):
    """A crosswalk gap, or a board rebuilt for a new season, must not blank
    the cell or drop the pick -- and the count is what says out loud that a
    page full of ids is a join failure rather than a strange draft."""
    draft_id = _record(pd.Timestamp("2026-08-20 12:00:00"),
                       player_ids=("p1", "ghost", "p3", "p4",
                                   "p1", "p2", "p3", "p4"))
    board = client.get(f"/api/mocks/{draft_id}/board").json()
    assert board["unresolved"] == 1
    ghost = board["cells"][1]["player"]
    assert ghost["player_id"] == "ghost" and ghost["name"] == "ghost"
    assert ghost["position"] is None and ghost["market_rank"] is None
    # Still a real cell at a real seat, with a real label.
    assert board["cells"][1]["made_by"] == "us"
    assert len(board["cells"]) == 8


# ---------------------------------------------------------------------------
# Live boards
# ---------------------------------------------------------------------------


def test_a_live_board_draws_from_the_file_alone(client):
    payload = _publish(n=3, league_id="222")
    board = client.get(f"/api/mocks/{payload['draft_id']}/board").json()

    assert board["active"] is True
    assert board["picks_made"] == 3
    # Three picks in a 4-team snake: slot 4 is next.
    assert board["on_the_clock"] == 4
    assert [c["made_by"] for c in board["cells"]] == ["human", "us", "engine"]
    assert [c["player"]["name"] for c in board["cells"]] == [
        "Ja'Marr Chase", "Bijan Robinson", "CeeDee Lamb"]
    # Seats nobody has drafted from yet are unknown, not "computer" -- the
    # room is a real room, it just has not reached slot 4.
    assert [c["had_owner"] for c in board["columns"]] == [True, True, False,
                                                          None]


def test_a_live_draft_with_no_owner_census_labels_nothing(client):
    """The mTeam read failing is a real failure mode (`play_draft` says so out
    loud), and it means unknown, not "empty room"."""
    payload = _publish(n=3, league_id="777", owners=None)
    board = client.get(f"/api/mocks/{payload['draft_id']}/board").json()
    assert [c["made_by"] for c in board["cells"]] == ["unknown", "us",
                                                      "unknown"]
    assert payload["human_seats"] is None


def test_an_unresolvable_live_pick_keeps_its_place_in_the_numbering(client):
    """A pick whose player the crosswalk could not name still HAPPENED. It
    draws no cell -- there is no id to draw -- but `picks_made` and the clock
    must count it, or every later pick lands on the wrong seat."""
    timeline = [mf.PickFrame(1, 11, 901, False),
                mf.PickFrame(2, 12, None, False),
                mf.PickFrame(3, 13, 903, True)]
    payload = mf.live_payload(timeline, _tool(), "888", 2026, TEAMS, ROUNDS,
                              MY_SLOT, None,
                              owners=_owners(),
                              my_team_id=10 + MY_SLOT)
    mf.write_live(payload)
    board = client.get(f"/api/mocks/{payload['draft_id']}/board").json()
    assert board["picks_made"] == 3
    assert board["on_the_clock"] == 4
    assert [c["overall"] for c in board["cells"]] == [1, 3]


# ---------------------------------------------------------------------------
# Staleness and unknown ids
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    """A pid that is not in use, so `os.kill(pid, 0)` really does fail."""
    dead = 4_000_000
    try:
        os.kill(dead, 0)
    except ProcessLookupError:
        return dead
    except PermissionError:
        pass
    pytest.skip("pid 4000000 is in use on this machine")


def _write_stale(payload: dict, tmp_path):
    """The file a farm killed mid-draft leaves behind."""
    path = tmp_path / "farm-live" / f"{payload['league_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload, pid=_dead_pid(),
                                    written_at=time.time())))
    return path


def test_a_live_file_whose_farm_is_dead_is_not_a_live_draft(client, tmp_path):
    payload = mf.live_payload(_timeline(3), _tool(), "999", 2026, TEAMS,
                              ROUNDS, MY_SLOT, None, owners=None)
    path = _write_stale(payload, tmp_path)

    assert client.get("/api/mocks").json() == {"drafts": []}
    board = client.get(f"/api/mocks/{payload['draft_id']}/board")
    assert board.status_code == 404
    # Swept on the read, so it cannot come back on the next one.
    assert not path.exists()


def test_a_dead_farm_leaves_the_corpus_row_it_did_manage_to_write(
        client, tmp_path):
    """A farm that recorded the draft and then died before deleting its live
    file. The draft is not live -- but it is not lost either, and the page has
    to serve the completed row rather than 404 on an id that exists."""
    started = pd.Timestamp("2026-08-22 10:00:00").to_pydatetime()
    draft_id = _record(pd.Timestamp("2026-08-22 10:40:00"), league_id="444",
                       started_at=started)
    payload = mf.live_payload(_timeline(8), _tool(), "444", 2026, TEAMS,
                              ROUNDS, MY_SLOT, started, owners=None)
    _write_stale(payload, tmp_path)

    rows = client.get("/api/mocks").json()["drafts"]
    assert [(r["id"], r["status"]) for r in rows] == [(draft_id, "complete")]
    board = client.get(f"/api/mocks/{draft_id}/board").json()
    assert [c["made_by"] for c in board["cells"]] == EXPECTED_MADE_BY


def test_an_unknown_id_is_a_404(client):
    _record(pd.Timestamp("2026-08-20 12:00:00"))
    response = client.get("/api/mocks/mock:nosuchdraft/board")
    assert response.status_code == 404
    assert "mock:nosuchdraft" in response.json()["detail"]


def test_a_draft_from_another_source_is_not_a_mock(client):
    """The corpus also holds imported league history. This page is about
    mocks, and an id from anywhere else is not one of its drafts."""
    corpus = dl.corpus_conn(dl.CORPUS_PATH)
    try:
        draft_id = dl.record(corpus, dl.DraftRecord(
            source=dl.SOURCE_HISTORY, league_id="111", season=2025,
            teams=TEAMS, rounds=ROUNDS,
            picks=_picks_frame(["p1", "p2"] * 4, None, None)))
    finally:
        corpus.close()
    assert client.get("/api/mocks").json() == {"drafts": []}
    assert client.get(f"/api/mocks/{draft_id}/board").status_code == 404


# ---------------------------------------------------------------------------
# picks the board does not carry
# ---------------------------------------------------------------------------
#
# THE BOARD IS NOT THE DRAFT POOL. `build_board` drops anyone ESPN's own
# ranking file leaves out (`scoring/board.py`'s `espn_unranked` filter, and
# that file is a top-500 pull), while ESPN's mock rooms happily offer those
# players -- Evan Engram, projected 106 points, went at pick 5 of a real
# farmed room. The cell used to draw his GSIS id in a grid of names.


def _off_board(monkeypatch, named):
    """A crosswalk that resolves pick 2 to a player the board frame has no row
    for, plus whatever `identify_players` can say about him."""
    monkeypatch.setattr("api.mocks.identify_players",
                        lambda conn, ids: {i: named for i in ids} if named else {})


def test_a_pick_the_board_does_not_carry_is_still_named(client, monkeypatch):
    _off_board(monkeypatch, {"name": "Evan Engram", "position": "TE",
                             "team": "DEN", "headshot": "http://x/e.png"})
    timeline = [mf.PickFrame(1, 11, 901, False),
                mf.PickFrame(2, 12, 950, False)]
    payload = mf.live_payload(timeline, _tool(), "555", 2026, TEAMS, ROUNDS,
                              MY_SLOT, None, owners=_owners(),
                              my_team_id=10 + MY_SLOT)
    # 950 is not in the crosswalk, so the farm records the raw id it saw.
    payload["picks"][1]["player_id"] = "00-0033881"
    mf.write_live(payload)

    board = client.get(f"/api/mocks/{payload['draft_id']}/board").json()
    off = board["cells"][1]["player"]

    assert off["name"] == "Evan Engram"
    assert (off["position"], off["team"]) == ("TE", "DEN")
    assert off["headshot"] == "http://x/e.png"
    # Identity, never ranks: he is genuinely not on this board, and a market
    # rank invented to fill the cell would be a number nobody computed.
    assert off["overall_rank"] is None
    assert off["market_rank"] is None
    assert off["value"] is None
    # A named pick is not an unresolved one -- the count means "still drawn as
    # an id", which is the thing worth knowing about.
    assert board["unresolved"] == 0


def test_a_pick_nothing_can_name_still_falls_back_to_its_id(client, monkeypatch):
    _off_board(monkeypatch, None)
    timeline = [mf.PickFrame(1, 11, 901, False),
                mf.PickFrame(2, 12, 950, False)]
    payload = mf.live_payload(timeline, _tool(), "556", 2026, TEAMS, ROUNDS,
                              MY_SLOT, None, owners=_owners(),
                              my_team_id=10 + MY_SLOT)
    payload["picks"][1]["player_id"] = "who-is-this"
    mf.write_live(payload)

    board = client.get(f"/api/mocks/{payload['draft_id']}/board").json()

    assert board["cells"][1]["player"]["name"] == "who-is-this"
    assert board["unresolved"] == 1
