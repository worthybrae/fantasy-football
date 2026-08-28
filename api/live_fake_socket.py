"""A stand-in for `pipeline.draft_socket.run_socket_listener` that needs no ESPN.

This exists for one reason: to drive api/live.py with many live rooms at once
so the scale-out work can be measured. Everything downstream of the socket --
the pick pump, `apply_picks` writing to the league's own DuckDB file, the
recompute, `/api/live/state` and `/api/live/board` -- is the code under test,
and all of it is reached by frames arriving on a listener. So the only thing
that has to be replaced is the frame source.

It replays `data/draft_room_trace.jsonl`, the real capture of a real ESPN
draft room this project already keeps for `tests/test_draft_listener.py`'s
sibling fixture. The websocket frames in that file are the `ws-recv` rows, and
they are fed to `listener.on_frame` verbatim: same verbs, same ordering, same
128 picks. Nothing here is synthesised, so a room driven by this module puts
the same shaped work through the listener as a room driven by ESPN.

The contract is `run_socket_listener`'s, deliberately, so api/live.py's
`_socket_run_fn` can call either one without knowing which it has:

  * `on_activity()` fires on every frame received;
  * `on_change()` fires exactly when `listener.on_frame` reports a change;
  * `stop_event` is polled cooperatively and returns promptly when set;
  * `on_socket` receives a handle exactly once, up front.

The one addition is `pick_interval`, which is the whole point of a replay: it
decides how fast the draft runs, and therefore how much recompute per second
one room costs. Bind it with `functools.partial` (see scripts/load_server.py).

WIRING IT IN. api/live.py resolves the runner off its own module global, so
either of these works and neither needs a change to it:

    import api.live
    from api.live_fake_socket import run_fake_socket_listener
    api.live.run_socket_listener = run_fake_socket_listener

or the `LIVE_FAKE_SOCKET=1` env switch in api/live.py, which imports the same
name from this module. `scripts/load_server.py` does the first, so the load
test runs whether or not the switch is present.
"""
import json
import threading
import os
import time
from pathlib import Path

# The capture, and the row kind that holds a websocket frame. Same file and
# same key `tests/test_draft_listener.py` reads its fixture with -- a jsonl of
# {"kind": ..., "payload": ...} rows, where "ws-recv" is a frame the room
# received and everything else (http requests, ws-send, ws-open/close) is not.
TRACE_PATH = "data/draft_room_trace.jsonl"
SOCKET_KIND = "ws-recv"
# The verb that means a pick landed. Pacing is expressed in picks rather than
# frames because a pick is what costs the server something: it is what makes
# `listener.on_frame` report a change, which is what runs apply_picks and the
# recompute.
PICK_VERB = "SELECTED"

_TRACE_CACHE: dict[str, list[str]] = {}
_TRACE_LOCK = threading.Lock()


class FakeSocketHandle:
    """What `on_socket` is handed -- the shape api/live.py touches, no socket.

    `live_select` and `live_autodraft` (api/live.py) do exactly two things
    with the handle they took off `state["socket"]`: they gate on `alive()`
    and then `send()` a line of text. `/api/live/state` calls `alive()` too,
    to tell the room whether the draft buttons should be enabled. So those
    are the methods that have to exist, and `attach`/`detach` are here to
    match `pipeline.draft_socket.SocketHandle` rather than because anything
    outside that module calls them.

    `alive()` is always True: there is no connection to drop, and a handle
    that reported False would make every select in a load run 503 before it
    reached any of the code being measured. Sent lines are kept so a test can
    assert on them.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.sent: list[str] = []

    def attach(self, ws) -> None:
        pass

    def detach(self) -> None:
        pass

    def alive(self) -> bool:
        return True

    def send(self, text: str) -> None:
        with self._lock:
            self.sent.append(text)


def load_frames(path: str = TRACE_PATH) -> list[str]:
    """Every websocket frame in the capture, in order, parsed once per path.

    Cached on the module: a hundred rooms replaying the same 600KB file would
    otherwise parse it a hundred times, and the point of the load test is to
    measure the server, not the generator.
    """
    with _TRACE_LOCK:
        cached = _TRACE_CACHE.get(path)
        if cached is not None:
            return cached
    frames = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == SOCKET_KIND:
            frames.append(str(row.get("payload") or ""))
    with _TRACE_LOCK:
        _TRACE_CACHE[path] = frames
    return frames


def frame_delays(frames: list[str], pick_interval: float) -> list[float]:
    """How long to wait before each frame, so picks land `pick_interval` apart.

    A real room is not a metronome: between two picks ESPN sends a CLOCK tick
    or two, an AUTOSUGGEST, sometimes a PONG. Those frames are cheap (the
    listener folds them and reports no change) and spacing them evenly across
    the gap is closer to the real arrival pattern than dumping them all at
    once and then sleeping. So the wait for one pick is shared out over the
    frames leading up to it.

    Frames after the last pick -- and every frame, in the degenerate case of a
    trace with no picks at all -- are spread at the average rate of the
    segments that did have one.
    """
    picks = [i for i, f in enumerate(frames) if f.startswith(PICK_VERB)]
    delays = [0.0] * len(frames)
    start = 0
    for end in picks:
        span = end - start + 1
        per = pick_interval / span
        for i in range(start, end + 1):
            delays[i] = per
        start = end + 1
    if start < len(frames):
        # The tail. Its rate is the one the body ran at, so a trace that ends
        # mid-round does not suddenly stall or sprint.
        span = (picks[-1] + 1) / len(picks) if picks else float(len(frames))
        per = pick_interval / max(span, 1.0)
        for i in range(start, len(frames)):
            delays[i] = per
    return delays


def _sleep_or_stopped(stop_event, seconds: float) -> bool:
    """Wait, returning True the moment `stop_event` is set.

    `Event.wait` wakes on the set rather than on the timeout, so a stop lands
    within a scheduler tick however long the remaining gap was -- no polling
    loop and no upper bound on the sleep slice needed.
    """
    if stop_event is None:
        if seconds > 0:
            time.sleep(seconds)
        return False
    return stop_event.wait(seconds)


# The pick cadence when the caller (api/live.py's pump, which passes no
# `pick_interval`) leaves it to the environment. Two seconds is a stress
# setting -- ten to twenty times what a real room produces, where a pick
# lands every thirty to sixty seconds -- so a load run that wants the
# realistic number sets this (scripts/load_server.py --pick-interval).
PICK_INTERVAL_ENV = "LIVE_FAKE_PICK_INTERVAL"
DEFAULT_PICK_INTERVAL = 2.0


def pick_interval_from_env() -> float:
    raw = (os.environ.get(PICK_INTERVAL_ENV) or "").strip()
    try:
        return float(raw) if raw else DEFAULT_PICK_INTERVAL
    except ValueError:
        return DEFAULT_PICK_INTERVAL


def run_fake_socket_listener(listener, league_id, team_id, swid, token,
                             on_change=None, stop_event=None, on_activity=None,
                             on_socket=None, pick_interval=None,
                             trace_path=TRACE_PATH, loop=True) -> None:
    """Replay a recorded draft room into `listener`. Blocking, like the real one.

    `league_id`, `team_id`, `swid` and `token` are accepted and ignored --
    there is nothing to authenticate against. They stay in the signature
    because this is a drop-in for `run_socket_listener` and the call site in
    api/live.py passes them positionally.

    `loop` restarts the trace when it runs out, which keeps `on_activity`
    firing so a long run never trips api/live.py's staleness check. The
    replayed picks are the same ones, so `picks_from_events` dedups them and
    the second pass mostly reports no change -- exactly what ESPN's own JOIN
    replay does after a reconnect.
    """
    if pick_interval is None:
        pick_interval = pick_interval_from_env()
    frames = load_frames(trace_path)
    if not frames:
        # Nothing to replay, and `loop` would otherwise spin on an empty
        # inner loop for the life of the room -- a busy core per room, for
        # a listener that can never report anything. Say so and stop; the
        # connect sees a listener that ended, which is the truth.
        print(f"live_fake_socket: no frames in {trace_path}; nothing to "
              "replay")
        return
    delays = frame_delays(frames, pick_interval)
    if on_socket is not None:
        # Immediately, unlike the real socket, which waits for its first
        # successful handshake because before that there is nothing to hand
        # out. Here the "connection" is a file that is already open, so the
        # honest moment to publish is now.
        on_socket(FakeSocketHandle())
    while stop_event is None or not stop_event.is_set():
        for frame, delay in zip(frames, delays):
            if _sleep_or_stopped(stop_event, delay):
                return
            if on_activity is not None:
                on_activity()
            if listener.on_frame(frame) and on_change is not None:
                on_change()
        if not loop:
            return
