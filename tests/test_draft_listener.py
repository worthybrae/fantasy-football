import json
from pathlib import Path

import pytest

from pipeline.draft_listener import DraftListener

FIXTURE = Path("tests/fixtures/espn_draft_socket.jsonl")


def _payloads():
    rows = [json.loads(l) for l in FIXTURE.read_text().splitlines() if l]
    return [str(r.get("payload") or "") for r in rows if r["kind"] == "ws-recv"]


def test_listener_tracks_who_is_on_the_clock():
    lis = DraftListener({})
    assert lis.on_the_clock is None
    lis.on_frame("SELECTING 3 30000")
    assert lis.on_the_clock == 3
    assert lis.ms_remaining == 30000
    lis.on_frame("SELECTING 4 25000")
    assert lis.on_the_clock == 4


def test_clock_frames_update_the_remaining_time_but_not_the_team():
    """CLOCK ticks during a pick. It must not be read as a turn change --
    doing so would make the UI claim a different manager is picking."""
    lis = DraftListener({})
    lis.on_frame("SELECTING 3 30000")
    lis.on_frame("CLOCK 0 21500")
    assert lis.on_the_clock == 3
    assert lis.ms_remaining == 21500


def test_listener_reports_the_draft_has_started():
    lis = DraftListener({})
    assert lis.started is False
    lis.on_frame("STATE 1")
    assert lis.started is True


def test_picks_fold_through_the_shared_definition():
    lis = DraftListener({111: "g1", 222: "g2"})
    for frame in ["STATE 1", "SELECTING 1 30000", "SELECTED 1 111 2",
                  "SELECTING 2 30000", "SELECTED 2 222 4"]:
        lis.on_frame(frame)
    out = lis.picks()
    assert list(out.rows["player_id"]) == ["g1", "g2"]
    assert list(out.rows["pick_no"]) == [1, 2]


def test_picks_is_recomputable_and_stable():
    """Folding twice must give the same answer. The listener accumulates
    frames; it must not consume them."""
    lis = DraftListener({111: "g1"})
    lis.on_frame("SELECTED 1 111 2")
    first = lis.picks()
    second = lis.picks()
    assert list(first.rows["pick_no"]) == list(second.rows["pick_no"]) == [1]


def test_a_garbage_frame_does_not_break_the_listener():
    """A live socket is not a contract. One unparseable frame must not stop
    the listener from tracking the rest of a draft."""
    lis = DraftListener({111: "g1"})
    lis.on_frame("")
    lis.on_frame("!!! not a protocol frame")
    lis.on_frame("SELECTED 1 111 2")
    assert list(lis.picks().rows["player_id"]) == ["g1"]


def test_listener_learns_its_own_team_from_the_token_frame():
    """TOKEN carries `1:<leagueId>:<teamId>:{SWID}:<nonce>` -- see
    _team_id_from_token's docstring for why TOKEN, not JOINED, is trusted."""
    lis = DraftListener({})
    assert lis.my_team_id is None
    lis.on_frame(
        "TOKEN 1:196877779:2:{8491403C-A53F-4257-8D52-F8AE32CED897}:-1781796296")
    assert lis.my_team_id == 2


def test_listener_ignores_a_malformed_token_frame():
    lis = DraftListener({})
    lis.on_frame("TOKEN garbage")
    assert lis.my_team_id is None


def test_a_malformed_token_does_not_erase_an_already_known_team():
    lis = DraftListener({})
    lis.on_frame("TOKEN 1:1:2:{SWID}:-1")
    lis.on_frame("TOKEN garbage")
    assert lis.my_team_id == 2


def test_listener_my_team_id_is_not_disturbed_by_other_teams_picking():
    """SELECTING/SELECTED carry a team id on every single frame of a real
    draft -- if my_team_id were read from those, it would be reassigned to
    whoever is currently picking, dozens of times over a draft."""
    lis = DraftListener({})
    lis.on_frame("TOKEN 1:196877779:2:{SWID}:-1781796296")
    lis.on_frame("SELECTING 5 30000")
    lis.on_frame("SELECTED 5 4430807 3")
    lis.on_frame("SELECTING 6 30000")
    assert lis.my_team_id == 2


def test_listener_my_team_id_is_not_taken_from_another_teams_joined():
    """JOINED fires for every team that connects, not just us -- team 3
    joining here must not overwrite the team id TOKEN already established
    (and JOINED alone, with no TOKEN, must not set my_team_id either)."""
    lis = DraftListener({})
    lis.on_frame("TOKEN 1:196877779:2:{8491403C-A53F-4257-8D52-F8AE32CED897}:-1781796296")
    lis.on_frame("JOINED 3 {2D4A53F4-2B63-4747-8A53-F42B63974765}")
    assert lis.my_team_id == 2


def test_joined_alone_never_sets_my_team_id():
    lis = DraftListener({})
    lis.on_frame("JOINED 2 {8491403C-A53F-4257-8D52-F8AE32CED897}")
    assert lis.my_team_id is None


def test_on_frame_reports_a_pick_landing():
    lis = DraftListener({111: "g1"})
    assert lis.on_frame("STATE 1") is False
    assert lis.on_frame("SELECTING 1 30000") is False
    assert lis.on_frame("SELECTED 1 111 2") is True


def test_on_frame_reports_my_team_becoming_known_even_with_no_pick():
    """TOKEN arrives long before the first pick -- the caller (api/live.py's
    on_change) must be told right away, not on the next unrelated event,
    since the very first pick of the draft may be ours."""
    lis = DraftListener({})
    assert lis.on_frame(
        "TOKEN 1:196877779:2:{8491403C-A53F-4257-8D52-F8AE32CED897}:-1781796296") is True


def test_on_frame_does_not_re_report_once_the_team_is_already_known():
    lis = DraftListener({})
    lis.on_frame("TOKEN 1:1:2:{SWID}:-1")
    assert lis.on_frame("SELECTING 3 30000") is False
    assert lis.on_frame("TOKEN 1:1:2:{SWID}:-1") is False


def test_on_frame_reports_false_for_garbage():
    lis = DraftListener({})
    assert lis.on_frame("") is False
    assert lis.on_frame("!!! not a protocol frame") is False


@pytest.mark.skipif(not FIXTURE.exists(), reason="no draft capture")
def test_the_real_capture_teaches_the_listener_its_own_team():
    """161 frames from a live ESPN mock. TOKEN and the matching JOINED both
    name team 2 as ours (see the module docstring/task brief); team 3's
    JOINED, ~150 frames later, must not overwrite it."""
    lis = DraftListener({})
    for p in _payloads():
        lis.on_frame(p)
    assert lis.my_team_id == 2


@pytest.mark.skipif(not FIXTURE.exists(), reason="no draft capture")
def test_the_real_capture_drives_the_listener_end_to_end():
    """161 frames from a live ESPN mock, replayed in order."""
    payloads = _payloads()
    ids = {}
    for p in payloads:
        if p.startswith("SELECTED "):
            ids[int(p.split()[2])] = f"p{p.split()[2]}"
    lis = DraftListener(ids)
    for p in payloads:
        lis.on_frame(p)
    out = lis.picks()
    assert lis.started is True
    assert lis.on_the_clock is not None
    assert len(out.rows) >= 9
    assert out.unmapped == []
    assert list(out.rows["pick_no"]) == list(range(1, len(out.rows) + 1))
