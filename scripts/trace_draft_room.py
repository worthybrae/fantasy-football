"""Record every network call ESPN's draft room makes, to find the live feed.

`view=mDraftDetail` does not carry selections while a draft runs -- measured
in a live 8-team mock: `inProgress` true, 128 picks on the board, every
`playerId` still -1 fifteen minutes in. The room learns about picks from
somewhere else, and this finds where.

Opens a VISIBLE browser you drive yourself. Join a mock draft at
https://fantasy.espn.com/football/mockdraftlobby and let some picks happen,
then Ctrl-C (or kill the process -- see below).

    .venv/bin/python -u scripts/trace_draft_room.py

EVERY RECORD IS FLUSHED TO DISK AS IT ARRIVES. An earlier version collected
in memory and wrote on shutdown; Ctrl-C propagated through Playwright's
teardown, `context.close()` raised TargetClosedError, and the process died
before reaching the writes -- losing a whole capture. Nothing is buffered
now, so any exit at all keeps everything recorded up to that instant.

    data/draft_room_trace.jsonl   -- every request and frame, appended live

Run scripts/summarise_draft_trace.py afterwards to rank hosts and find
frames that mention a player.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.espn_league import STATE_PATH

TRACE = Path("data/draft_room_trace.jsonl")
LOBBY = "https://fantasy.espn.com/football/mockdraftlobby"


def main() -> int:
    from playwright.sync_api import sync_playwright

    TRACE.parent.mkdir(parents=True, exist_ok=True)
    fh = TRACE.open("w", buffering=1)          # line buffered
    counts = {"http": 0, "ws": 0}

    def record(kind: str, **fields):
        fh.write(json.dumps({"kind": kind, **fields}) + "\n")
        fh.flush()
        if kind == "http":
            counts["http"] += 1
        elif kind.startswith("ws"):
            counts["ws"] += 1
            if counts["ws"] % 25 == 0:
                print(f"  ws frames: {counts['ws']}", flush=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        state = Path(STATE_PATH)
        context = browser.new_context(
            **({"storage_state": str(state)} if state.exists() else {}))
        page = context.new_page()

        page.on("request", lambda r: record("http", method=r.method, url=r.url))

        def on_ws(ws):
            record("ws-open", url=ws.url)
            print(f"  WEBSOCKET OPENED: {ws.url}", flush=True)
            ws.on("framereceived", lambda f: record("ws-recv", url=ws.url, payload=f))
            ws.on("framesent", lambda f: record("ws-send", url=ws.url, payload=f))
            ws.on("close", lambda _: record("ws-close", url=ws.url))

        page.on("websocket", on_ws)

        page.goto(LOBBY)
        print("\nBrowser open. Join a mock draft and let a few picks happen.")
        print(f"Everything is being appended to {TRACE} as it arrives.")
        print("Ctrl-C when done -- the file is already complete either way.\n",
              flush=True)
        try:
            while True:
                page.wait_for_timeout(1000)
        except BaseException:                  # noqa: BLE001
            # Deliberately BaseException: KeyboardInterrupt is not an
            # Exception, and it is the expected way this ends. Nothing here
            # can lose data -- the file is already written -- so the only job
            # is to exit without a wall of Playwright teardown noise.
            print(f"\nstopping: {counts['http']} http, {counts['ws']} ws frames",
                  flush=True)
        finally:
            fh.close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException:                      # noqa: BLE001
        # Playwright's teardown raises TargetClosedError once the browser is
        # gone. The trace is on disk regardless; do not let that noise imply
        # the capture failed.
        print(f"trace saved to {TRACE}", flush=True)
        raise SystemExit(0)
