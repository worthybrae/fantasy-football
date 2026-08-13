"""Prove the socket accepts a token we minted ourselves, during a live draft.

This is the one unproven link in the whole browserless path. Everything else
is verified: draftSecurity returns a token with cookies alone, CORS lets a
page read it, the listener follows a draft. The open question is whether the
`fantasydraft.espn.com` JOIN socket accepts a token we fetched and assembled
into the URL, rather than one ESPN's own page opened.

Run it WHILE A DRAFT IS ACTUALLY RUNNING (a mock is fine). Against a league
whose draft has not started, JOIN answers HTTP 500 -- that is "no draft to
join", not a failure of this approach, which is exactly why it needs a live
one to settle.

    ESPN_S2='...' SWID='{...}' \\
      .venv/bin/python -u scripts/test_direct_socket.py <leagueId> <teamId>

Get the cookies from your browser (DevTools > Application > Cookies on
espn.com) or from the saved login at data/espn_state.json. Read-only:
connects, prints the first frames, writes nothing.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring.config import CURRENT_SEASON

try:
    from pipeline.draft_socket import draft_security_token, http_fetch, socket_url
except Exception:  # noqa: BLE001
    draft_security_token = None


def _cookies() -> dict:
    s2 = os.environ.get("ESPN_S2")
    swid = os.environ.get("SWID")
    if s2 and swid:
        return {"espn_s2": s2, "SWID": swid}
    # Fall back to the saved login.
    from pipeline.draft_socket import load_cookies
    return load_cookies()


async def _listen(url: str, cookie_header: str) -> int:
    import websockets

    headers = {"Origin": "https://fantasy.espn.com", "Cookie": cookie_header}
    try:
        async with websockets.connect(
            url, additional_headers=headers, open_timeout=20
        ) as ws:
            print("\n  CONNECTED — no browser, self-minted token\n", flush=True)
            got = 0
            for _ in range(12):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=8)
                except asyncio.TimeoutError:
                    print("  (quiet — is a pick happening?)", flush=True)
                    break
                got += 1
                print(f"  <- {str(msg)[:110]}", flush=True)
            print(f"\n  RESULT: received {got} frames. "
                  f"{'The direct path works.' if got else 'Connected but silent.'}",
                  flush=True)
            return 0
    except Exception as e:  # noqa: BLE001
        print(f"\n  JOIN rejected: {type(e).__name__}: {str(e)[:140]}", flush=True)
        print("  If this is HTTP 500, the draft is probably not running yet.",
              flush=True)
        return 1


def main(league_id: str, team_id: int) -> int:
    if draft_security_token is None:
        print("pipeline.draft_socket not importable -- is `websockets` installed?")
        return 2
    cookies = _cookies()
    token = draft_security_token(http_fetch(cookies), league_id, team_id,
                                 CURRENT_SEASON)
    print(f"token from draftSecurity: {token}", flush=True)
    url = socket_url(league_id, team_id, cookies["SWID"], token)
    cookie_header = f"espn_s2={cookies['espn_s2']}; SWID={cookies['SWID']}"
    return asyncio.run(_listen(url, cookie_header))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: ESPN_S2=.. SWID=.. test_direct_socket.py <leagueId> <teamId>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], int(sys.argv[2])))
