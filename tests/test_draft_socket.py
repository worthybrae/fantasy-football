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
