"""Save an ESPN login so headless tools can authenticate.

Opens a real browser window at ESPN's login page and waits for you to sign
in. Disney SSO handles 2FA and the bot check there, which is the only place
they can be answered; everything afterwards runs headless off the cookies
this writes.

Run:  .venv/bin/python scripts/espn_login.py

Writes `espn_s2` and `SWID` to data/espn_state.json (gitignored). Anything
that can read that file can act as you on ESPN until the session expires.
"""
import sys

from pipeline.espn_league import STATE_PATH, EspnClient
from pipeline.draft_socket import load_cookies


def main() -> int:
    with EspnClient() as client:
        client.login()
    try:
        cookies = load_cookies()
    except RuntimeError as exc:
        print(f"Login did not complete: {exc}")
        return 1
    swid = cookies["SWID"]
    print(f"Saved to {STATE_PATH}. SWID {swid[:10]}..., espn_s2 "
          f"{len(cookies['espn_s2'])} chars.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
