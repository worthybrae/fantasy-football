"""Find where ESPN's draft socket token comes from.

The socket URL carries a token like

    1:196877779:2:{SWID}:-1781796296

and the last component has no derivation we have been able to reproduce --
not a Java hashCode of the SWID, league id, team id, or any combination of
them. If we cannot mint it ourselves we cannot open the socket ourselves,
which is the only reason this tool drives a browser at all.

So: capture every HTTP *response body* the draft room loads, plus the socket
URL, and search the bodies for the token. The earlier tracer logged request
URLs only, which is exactly the evidence needed and exactly what it discarded.

    .venv/bin/python -u scripts/find_draft_token.py "<espn draft url>"

You do NOT need to draft. Just let the room load. Ctrl-C once it has.

If the token turns up in a response body, we can fetch it with the saved
cookies and connect to the socket directly -- no browser, no second login.
If it does not, it is minted client-side and the browser stays.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.espn_league import STATE_PATH

OUT = Path("data/token_hunt.jsonl")
# The socket URL's fifth parameter: 1:<league>:<team>:{SWID}:<mystery>
TOKEN_RE = re.compile(r"\d+:\d+:\d+:\{[0-9A-Fa-f-]+\}:(-?\d+)")


def main(url: str) -> int:
    from playwright.sync_api import sync_playwright

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT.open("w", buffering=1)
    found = {"token": None, "socket": None}
    bodies = []

    def on_response(resp):
        # Only ESPN's own JSON/JS; skip images, fonts, ad networks.
        if "espn" not in resp.url:
            return
        try:
            body = resp.text()
        except Exception:                                   # noqa: BLE001
            return
        if not body or len(body) > 4_000_000:
            return
        bodies.append((resp.url, body))
        fh.write(json.dumps({"kind": "response", "url": resp.url,
                             "len": len(body)}) + "\n")
        m = TOKEN_RE.search(body)
        if m:
            print(f"\n  TOKEN-SHAPED STRING IN A RESPONSE BODY:\n"
                  f"    {m.group(0)}\n    from {resp.url[:120]}\n", flush=True)
            fh.write(json.dumps({"kind": "token-in-body", "url": resp.url,
                                 "match": m.group(0)}) + "\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        state = Path(STATE_PATH)
        context = browser.new_context(
            **({"storage_state": str(state)} if state.exists() else {}))
        page = context.new_page()
        page.on("response", on_response)

        def on_ws(ws):
            if "fantasydraft" not in ws.url:
                return
            found["socket"] = ws.url
            m = TOKEN_RE.search(ws.url)
            if m:
                found["token"] = m.group(1)
            print(f"\n  DRAFT SOCKET OPENED\n    {ws.url[:150]}\n", flush=True)
            fh.write(json.dumps({"kind": "socket", "url": ws.url}) + "\n")

        page.on("websocket", on_ws)
        page.goto(url)
        print("\nLet the draft room finish loading. Ctrl-C when it has.\n",
              flush=True)
        try:
            while True:
                page.wait_for_timeout(1000)
        except BaseException:                               # noqa: BLE001
            pass
        finally:
            try:
                context.close()
                browser.close()
            except Exception:                               # noqa: BLE001
                pass

    print("\n" + "=" * 60)
    if found["token"]:
        tok = found["token"]
        print(f"socket token component: {tok}")
        hits = [u for u, b in bodies if tok in b]
        if hits:
            print(f"\nFOUND IT in {len(hits)} response body/bodies:")
            for u in hits[:5]:
                print(f"  {u[:140]}")
            print("\n-> The token is served, not computed. We can fetch it "
                  "with the saved cookies and open the socket directly.")
        else:
            print(f"\nNOT in any of {len(bodies)} ESPN response bodies.")
            print("-> Minted client-side. Opening the socket ourselves would "
                  "mean reimplementing ESPN's JS; the browser stays.")
    else:
        print("No draft socket opened -- the page may not have entered the "
              "room. Try the URL you get once you are actually in the draft.")
    fh.close()
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: find_draft_token.py <espn draft url>")
        raise SystemExit(2)
    try:
        raise SystemExit(main(sys.argv[1]))
    except KeyboardInterrupt:
        raise SystemExit(0)
