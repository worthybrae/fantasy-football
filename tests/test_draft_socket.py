import threading

import pytest

from pipeline import draft_socket
from pipeline.draft_socket import SocketHandle


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, text):
        if self.closed:
            raise ConnectionError("closed")
        self.sent.append(text)

    def close(self):
        self.closed = True


def test_handle_sends_to_the_attached_socket():
    ws = FakeSocket()
    handle = SocketHandle()
    handle.attach(ws)
    handle.send("SELECT 4429795\n")
    assert ws.sent == ["SELECT 4429795\n"]
    assert handle.alive() is True


def test_handle_refuses_to_send_with_nothing_attached():
    handle = SocketHandle()
    assert handle.alive() is False
    with pytest.raises(ConnectionError):
        handle.send("SELECT 1\n")


def test_handle_sends_to_the_new_socket_after_a_reconnect():
    """ESPN drops a live draft socket every few minutes. A send that lands on
    the dropped one is a pick that silently never happened."""
    old, new = FakeSocket(), FakeSocket()
    handle = SocketHandle()
    handle.attach(old)
    handle.detach()
    handle.attach(new)
    handle.send("SELECT 7\n")
    assert old.sent == []
    assert new.sent == ["SELECT 7\n"]


def test_run_socket_listener_publishes_a_handle(monkeypatch):
    ws = FakeSocket()

    def fake_recv(timeout=None):
        raise TimeoutError

    ws.recv = fake_recv
    monkeypatch.setattr(draft_socket, "_connect", lambda url, cookie: ws)

    seen = []
    stop = threading.Event()

    class Listener:
        def on_frame(self, payload):
            return False

    def on_socket(handle):
        seen.append(handle)
        stop.set()

    draft_socket.run_socket_listener(
        Listener(), "1", "2", "{SWID}", 3, stop_event=stop,
        on_socket=on_socket)
    assert len(seen) == 1
    assert isinstance(seen[0], SocketHandle)


def test_reconnect_swap_detaches_before_the_second_connect_and_attaches_after(
        monkeypatch):
    """Drives the real run_socket_listener through one drop and reconnect --
    not the SocketHandle primitive directly -- because this is exactly the
    wiring a misplaced attach()/detach() would break silently: a SELECT sent
    in the gap between the drop and the reconnect must raise rather than
    queue up for a socket ESPN has already discarded, and a SELECT sent
    after the reconnect must land on the NEW socket, never the old one.

    The mid-window check runs from the first fake socket's own close() --
    called by run_socket_listener's finally, right after handle.detach() and
    before the loop goes back around to reconnect, which is exactly the gap
    under test. That callback's outcome is recorded into a dict rather than
    asserted with pytest.raises() directly: run_socket_listener wraps
    ws.close() in a bare except Exception: pass (a socket ESPN already
    dropped can't always be closed cleanly, and losing already-collected
    picks over a close error is not worth it), so an assertion raised from
    inside close() would be silently swallowed and the test would pass
    whether or not the bug it's checking for was actually there.
    """
    from websockets.exceptions import ConnectionClosed

    monkeypatch.setattr(draft_socket, "RECONNECT_BACKOFF_SECONDS", 0)

    first, second = FakeSocket(), FakeSocket()
    sockets = [first, second]
    handles = []
    mid_window = {"raised": None, "error": None}
    stop = threading.Event()

    def first_recv(timeout=None):
        raise ConnectionClosed(None, None)

    def first_close():
        # Between the drop and the second connect: nothing should be
        # attached, so this must raise ConnectionError, not queue onto
        # `first` and not silently succeed. Checked BEFORE marking `first`
        # closed on purpose: FakeSocket.send also raises ConnectionError
        # once `closed` is True, and that coincidence would mask a missing
        # handle.detach() -- a stale, still-attached handle would route the
        # send straight to `first.send()` and get the same exception type
        # for the wrong reason. Checking first isolates what is under test:
        # SocketHandle's own attached/detached state, not FakeSocket's.
        try:
            handles[0].send("SELECT 1\n")
        except Exception as exc:                        # noqa: BLE001 --
            # deliberately broad: any exception here is recorded and judged
            # by the assertion below, not by whether this callback itself
            # raises (see docstring -- it would be swallowed either way).
            mid_window["raised"] = isinstance(exc, ConnectionError)
            mid_window["error"] = exc
        else:
            mid_window["raised"] = False
        first.closed = True

    first.recv = first_recv
    first.close = first_close

    def second_recv(timeout=None):
        # The reconnect has already attached `second` by the time recv is
        # ever called on it -- send here to prove it lands on the new
        # socket, then stop the loop cleanly (no sleep, no third connect).
        handles[0].send("SELECT 7\n")
        stop.set()
        raise TimeoutError

    second.recv = second_recv

    def fake_connect(url, cookie_header):
        return sockets.pop(0)

    monkeypatch.setattr(draft_socket, "_connect", fake_connect)

    class Listener:
        def on_frame(self, payload):
            return False

    def on_socket(handle):
        handles.append(handle)

    draft_socket.run_socket_listener(
        Listener(), "1", "2", "{SWID}", 3, stop_event=stop,
        on_socket=on_socket)

    assert mid_window["raised"] is True, (
        f"send between the drop and the reconnect did not raise "
        f"ConnectionError (got {mid_window['error']!r})")
    assert first.sent == [], "the old socket must never receive a send"
    assert second.sent == ["SELECT 7\n"]
