"""Follow a live ESPN draft by watching the room's own websocket.

ESPN's REST API serves no real playerId while a draft runs -- measured, 15
minutes into a live mock, every one of 128 board slots still -1. The draft
room instead holds a socket at fantasydraft.espn.com speaking a plain-text
line protocol.

We observe that socket rather than opening our own. Its URL carries a token
whose trailing component has no known derivation, so reproducing it means
reverse-engineering ESPN's auth; a real browser authenticates itself. It also
means there is only ever one connection -- ours would be a second one as the
same team, which ESPN may not tolerate.

`DraftListener` is a pure state machine with no Playwright import, so the
whole protocol is testable from the recorded fixture. `run_listener` is the
only part that needs a browser.
"""
from pipeline.espn_live import LivePicks, parse_frame, picks_from_events


class DraftListener:
    """Accumulates draft frames and folds them into picks on demand."""

    def __init__(self, crosswalk: dict):
        self.crosswalk = dict(crosswalk or {})
        self.events = []
        self.on_the_clock = None
        self.ms_remaining = None
        self.started = False

    def on_frame(self, payload: str) -> None:
        event = parse_frame(payload)
        if event is None:
            return
        self.events.append(event)
        if event.verb == "STATE":
            self.started = True
        elif event.verb == "SELECTING" and event.args:
            # Only SELECTING changes whose turn it is. CLOCK ticks during a
            # pick, and reading it as a turn change would make the UI name
            # the wrong manager.
            self.on_the_clock = _as_int(event.args[0], self.on_the_clock)
            if len(event.args) > 1:
                self.ms_remaining = _as_int(event.args[1], self.ms_remaining)
        elif event.verb == "CLOCK" and len(event.args) > 1:
            self.ms_remaining = _as_int(event.args[1], self.ms_remaining)

    def picks(self) -> LivePicks:
        """Fold everything seen so far. Re-foldable: the event list is
        accumulated, never consumed, so this is safe to call per refresh."""
        return picks_from_events(self.events, self.crosswalk)


def _as_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def run_listener(listener, url: str, state_path: str, on_change=None) -> None:
    """Drive a browser to `url` and feed the draft socket into `listener`.

    Blocking -- the caller runs it on a thread. `on_change` fires after any
    frame that changed the pick count, which is the signal to recompute.

    Headless is deliberate: the page authenticates from the saved storage
    state and needs no interaction. The human drafts in their own browser;
    this one only listens.
    """
    from pathlib import Path

    from playwright.sync_api import sync_playwright

    seen = [0]

    def handle(payload):
        listener.on_frame(payload)
        count = len(listener.picks().rows)
        if count != seen[0]:
            seen[0] = count
            if on_change is not None:
                on_change()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        state = Path(state_path)
        context = browser.new_context(
            **({"storage_state": str(state)} if state.exists() else {}))
        page = context.new_page()
        page.on("websocket", lambda ws: (
            ws.on("framereceived", handle) if "fantasydraft" in ws.url else None))
        page.goto(url)
        try:
            while True:
                page.wait_for_timeout(1000)
        except BaseException:      # noqa: BLE001 -- KeyboardInterrupt is not Exception
            pass
        finally:
            context.close()
            browser.close()
