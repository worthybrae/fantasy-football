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
        # Set once, from TOKEN -- see the comment in on_frame for why TOKEN
        # and not JOINED. build_session runs before the socket connects (see
        # api/live.py's DraftSession.my_slot), so this is the only place the
        # session's own team, and therefore its own draft slot, is ever
        # learned.
        self.my_team_id = None

    def on_frame(self, payload: str) -> bool:
        """Fold one frame into accumulated state.

        Returns True when this frame changed something a caller watching for
        updates needs to react to: the pick count moved (a real,
        crosswalk-resolvable pick landed), or my_team_id became known for
        the first time. The second case matters even though no pick landed,
        because api/live.py's on_change must resolve my_slot the moment the
        socket names our team -- waiting for the next pick to notice would
        mean, at worst, missing a recommendation for the very first pick of
        the draft, which may be ours. (run_listener is the only caller that
        reads this return value; it needs a browser and so is not exercised
        by these tests, but everything it depends on -- this method's
        return value -- is.)
        """
        event = parse_frame(payload)
        if event is None:
            return False
        before_picks = len(self.picks().rows)
        had_team = self.my_team_id is not None
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
        elif event.verb == "TOKEN" and event.args:
            # TOKEN is the one frame ESPN addresses to *this* connection
            # alone: its teamId echoes the identity the socket authenticated
            # with (the same value socket_url sent as the URL's own `token`
            # parameter), sent once near the top of a session. JOINED, by
            # contrast, broadcasts to the whole room every time *any* team's
            # browser attaches -- in the captured fixture, team 3's JOINED
            # lands 150 frames after ours, well into the draft -- so it names
            # a team id with no way to tell "ours" from "theirs" short of
            # already knowing our own SWID from somewhere else. TOKEN needs
            # nothing else, so it is the only source used here. A malformed
            # TOKEN must not erase an already-known good id, so a failed
            # parse is ignored rather than assigned.
            parsed = _team_id_from_token(event.args[0])
            if parsed is not None:
                self.my_team_id = parsed
        after_picks = len(self.picks().rows)
        newly_learned_team = self.my_team_id is not None and not had_team
        return after_picks != before_picks or newly_learned_team

    def picks(self) -> LivePicks:
        """Fold everything seen so far. Re-foldable: the event list is
        accumulated, never consumed, so this is safe to call per refresh."""
        return picks_from_events(self.events, self.crosswalk)


def _as_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _team_id_from_token(arg: str) -> int | None:
    """teamId out of a TOKEN frame's `1:<leagueId>:<teamId>:{SWID}:<nonce>`
    argument (see espn_live.socket_url, which builds the URL this token
    echoes back). Malformed input returns None rather than raising -- a
    garbled frame must not take down the listener (see
    test_a_garbage_frame_does_not_break_the_listener)."""
    parts = (arg or "").split(":")
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def run_listener(listener, url: str, state_path: str, on_change=None,
                 headless: bool = False, stop_event=None) -> None:
    """Drive a browser to `url` and feed the draft socket into `listener`.

    Blocking -- the caller runs it on a thread. `on_change` fires after any
    frame `DraftListener.on_frame` reports as notable (see its docstring):
    the pick count changed, which is the signal to recompute, or my_team_id
    just became known, which is the signal to resolve my_slot.

    Visible by default, and that is load-bearing, not cosmetic. The draft
    socket's URL carries a token with no known derivation, so we cannot open
    a second connection of our own -- we can only observe the one connection
    a real, authenticated browser holds. That means this window has to be
    where the human actually drafts: if they picked in a separate browser of
    their own instead, this one would be a second session for the same
    team, untested and liable to get one of the two kicked. `headless=True`
    stays available as a parameter for a test, or a future use that only
    observes and never needs a human at the keyboard.

    `stop_event` (a `threading.Event`, optional) is the cooperative-stop
    signal: the poll loop below checks it once a second and returns once it
    is set, closing the browser on the way out. `None` (the default) means
    "no external stop signal" -- the old behaviour of looping until an
    exception (a `KeyboardInterrupt`, or the process dying) preserved for
    any caller that manages its own lifetime rather than a caller that needs
    to end one listener before starting the next.
    """
    from pathlib import Path

    from playwright.sync_api import sync_playwright

    def handle(payload):
        if listener.on_frame(payload) and on_change is not None:
            on_change()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        state = Path(state_path)
        context = browser.new_context(
            **({"storage_state": str(state)} if state.exists() else {}))
        page = context.new_page()
        page.on("websocket", lambda ws: (
            ws.on("framereceived", handle) if "fantasydraft" in ws.url else None))
        page.goto(url)
        try:
            while stop_event is None or not stop_event.is_set():
                page.wait_for_timeout(1000)
        except BaseException:      # noqa: BLE001 -- KeyboardInterrupt is not Exception
            pass
        finally:
            # Persist cookies before tearing down. Without this every connect
            # starts logged out and makes the user sign in again -- the state
            # file was only ever written by EspnClient's own login flow, which
            # this path never touches. Best effort: a browser the user closed
            # first cannot be read, and failing to save is not worth losing
            # the picks already collected over.
            try:
                Path(state_path).parent.mkdir(parents=True, exist_ok=True)
                context.storage_state(path=str(state_path))
            except Exception:                              # noqa: BLE001
                pass
            try:
                context.close()
                browser.close()
            except Exception:                              # noqa: BLE001
                pass
