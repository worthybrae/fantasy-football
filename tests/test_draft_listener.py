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
