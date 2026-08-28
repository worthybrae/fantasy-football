import threading
import time

import pytest

from api.live_fake_socket import (
    FakeSocketHandle,
    frame_delays,
    load_frames,
    run_fake_socket_listener,
)
from pipeline.draft_listener import DraftListener

FAST = 0.001   # a whole 128-pick trace in a fifth of a second


class RecordingListener:
    """Counts what it was given, and reports a change on picks only."""

    def __init__(self):
        self.frames = []

    def on_frame(self, payload):
        self.frames.append(payload)
        return payload.startswith("SELECTED")


def test_replay_feeds_every_recorded_frame_in_order():
    lis = RecordingListener()
    run_fake_socket_listener(lis, "900001", 1, "{fake}", "tok",
                             pick_interval=FAST, loop=False)
    assert lis.frames == load_frames()
    # The capture is a real 128-pick room, not a stub -- if this ever reads
    # zero the trace file moved and every load number taken with it is noise.
    assert sum(f.startswith("SELECTED") for f in lis.frames) == 128


def test_on_change_fires_exactly_on_the_frames_that_changed_something():
    changes = []
    lis = RecordingListener()
    run_fake_socket_listener(lis, "900001", 1, "{fake}", "tok",
                             on_change=lambda: changes.append(1),
                             pick_interval=FAST, loop=False)
    assert len(changes) == sum(f.startswith("SELECTED") for f in lis.frames)


def test_on_change_tracks_a_real_listener_rather_than_the_stub():
    """The contract is `on_change` fires when `on_frame` reports a change, and
    the listener api/live.py actually uses is DraftListener -- which reports a
    change on a crosswalk-resolvable pick or on first learning our team, not
    on every SELECTED. So drive the real one and count its own answers."""
    lis = DraftListener({})
    expected = []
    changes = []

    real_on_frame = lis.on_frame

    def counting(payload):
        changed = real_on_frame(payload)
        expected.append(changed)
        return changed

    lis.on_frame = counting
    run_fake_socket_listener(lis, "900001", 1, "{fake}", "tok",
                             on_change=lambda: changes.append(1),
                             pick_interval=FAST, loop=False)
    assert sum(expected) > 0
    assert len(changes) == sum(expected)


def test_on_activity_fires_once_per_frame():
    beats = []
    lis = RecordingListener()
    run_fake_socket_listener(lis, "900001", 1, "{fake}", "tok",
                             on_activity=lambda: beats.append(1),
                             pick_interval=FAST, loop=False)
    assert len(beats) == len(lis.frames)


def test_stop_event_ends_the_replay_within_a_second():
    stop = threading.Event()
    lis = RecordingListener()
    thread = threading.Thread(
        target=run_fake_socket_listener,
        args=(lis, "900001", 1, "{fake}", "tok"),
        # Slow enough that a run which ignored the stop would still be
        # sleeping on its first frame when the join below gives up.
        kwargs={"stop_event": stop, "pick_interval": 60.0},
        daemon=True)
    thread.start()
    time.sleep(0.05)
    started = time.monotonic()
    stop.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert time.monotonic() - started < 1.0


def test_a_stop_already_set_never_delivers_a_frame():
    stop = threading.Event()
    stop.set()
    lis = RecordingListener()
    run_fake_socket_listener(lis, "900001", 1, "{fake}", "tok",
                             stop_event=stop, pick_interval=FAST)
    assert lis.frames == []


def test_the_handle_is_published_once_and_has_what_live_py_calls():
    handles = []
    lis = RecordingListener()
    run_fake_socket_listener(lis, "900001", 1, "{fake}", "tok",
                             on_socket=handles.append,
                             pick_interval=FAST, loop=False)
    assert len(handles) == 1
    handle = handles[0]
    # By name, because api/live.py reaches for exactly these: `alive()` gates
    # live_select, live_autodraft and the socket_alive flag /api/live/state
    # serves; `send()` is how a pick leaves the process. attach/detach match
    # pipeline.draft_socket.SocketHandle.
    for method in ("alive", "send", "attach", "detach"):
        assert callable(getattr(handle, method)), method
    assert handle.alive() is True
    handle.send("SELECT 4430807\n")
    handle.send("AUTODRAFT true\n")
    assert handle.sent == ["SELECT 4430807\n", "AUTODRAFT true\n"]


def test_the_handle_survives_attach_and_detach_calls():
    handle = FakeSocketHandle()
    handle.attach(object())
    handle.detach()
    assert handle.alive() is True


def test_loop_restarts_the_trace_instead_of_returning():
    stop = threading.Event()
    lis = RecordingListener()
    thread = threading.Thread(
        target=run_fake_socket_listener,
        args=(lis, "900001", 1, "{fake}", "tok"),
        kwargs={"stop_event": stop, "pick_interval": 0.0005, "loop": True},
        daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    n = len(load_frames())
    while time.monotonic() < deadline and len(lis.frames) <= n:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=2.0)
    assert len(lis.frames) > n


def test_picks_are_spaced_about_one_interval_apart():
    frames = load_frames()
    delays = frame_delays(frames, 2.0)
    picks = [i for i, f in enumerate(frames) if f.startswith("SELECTED")]
    # Cumulative wait up to and including each pick -- consecutive picks must
    # be one interval apart, which is the property the load test's "one pick
    # every two seconds" claim rests on.
    cumulative = 0.0
    stamps = []
    for i, delay in enumerate(delays):
        cumulative += delay
        if i in set(picks):
            stamps.append(cumulative)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert gaps
    assert all(abs(gap - 2.0) < 1e-6 for gap in gaps)
    assert stamps[0] == pytest.approx(2.0)


def test_a_trace_with_no_picks_still_paces_and_terminates(tmp_path):
    import json

    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(
        json.dumps({"kind": kind, "payload": payload})
        for kind, payload in [("ws-recv", "CLOCK 1 1000"),
                              ("http", "not a frame"),
                              ("ws-send", "SELECT 1"),
                              ("ws-recv", "CLOCK 1 900")]))
    lis = RecordingListener()
    run_fake_socket_listener(lis, "900001", 1, "{fake}", "tok",
                             pick_interval=FAST, loop=False,
                             trace_path=str(path))
    assert lis.frames == ["CLOCK 1 1000", "CLOCK 1 900"]
