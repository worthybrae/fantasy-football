"""Find the live pick feed in a draft-room trace.

Reads `data/draft_room_trace.jsonl` (written by trace_draft_room.py) and
answers one question: did anything on the wire carry a draft pick?

Deliberately does NOT filter on guessed keywords. An earlier version looked
only for `playerId`/`overallPickNumber`, which are the names ESPN's REST
payload uses -- there is no reason a websocket protocol would reuse them,
and searching for them would have hidden the feed rather than found it.
Instead this ranks what is actually there and lets the shapes speak.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

TRACE = Path("data/draft_room_trace.jsonl")


def main() -> int:
    if not TRACE.exists():
        print(f"{TRACE} not found -- run trace_draft_room.py first")
        return 1

    rows = [json.loads(line) for line in TRACE.read_text().splitlines() if line]
    http = [r for r in rows if r["kind"] == "http"]
    frames = [r for r in rows if r["kind"] in ("ws-recv", "ws-send")]
    sockets = Counter(r["url"] for r in rows if r["kind"] == "ws-open")

    print(f"{len(rows)} records: {len(http)} http, {len(frames)} ws frames, "
          f"{len(sockets)} websocket(s)\n")

    if sockets:
        print("WEBSOCKETS")
        for url, n in sockets.most_common():
            recv = sum(1 for f in frames if f["url"] == url and f["kind"] == "ws-recv")
            print(f"  {recv:>5} frames in   {url}")
        print()

    if frames:
        # Group frames by their key set. A heartbeat and a pick have
        # different shapes, and the pick message is usually the rarer one
        # with the richer body.
        shapes = Counter()
        examples = {}
        for f in frames:
            payload = f.get("payload")
            try:
                obj = json.loads(payload) if isinstance(payload, str) else None
            except (ValueError, TypeError):
                obj = None
            key = (tuple(sorted(obj)) if isinstance(obj, dict)
                   else f"<non-json len={len(str(payload))}>")
            shapes[key] += 1
            examples.setdefault(key, payload)
        print("FRAME SHAPES (rarest last -- the pick message is usually rare)")
        for key, n in shapes.most_common():
            label = ", ".join(key) if isinstance(key, tuple) else key
            print(f"  {n:>5}  {label[:110]}")
            print(f"         e.g. {str(examples[key])[:200]}")
        print()

    print("TOP HTTP PATHS")
    paths = Counter()
    for r in http:
        u = urlsplit(r["url"])
        paths[f"{u.netloc}{u.path}"] += 1
    for p, n in paths.most_common(25):
        print(f"  {n:>5}  {p}")

    if not sockets:
        print("\nNo websocket opened. If picks were visibly landing in the "
              "browser, the room is polling over HTTP -- look for a path "
              "above that repeats on a short interval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
