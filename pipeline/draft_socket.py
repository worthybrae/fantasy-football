"""Connect straight to ESPN's draft websocket -- no browser.

`pipeline/draft_listener.py` (and the old `pipeline.espn_live.socket_url`)
predate a discovery: the socket URL's fifth query parameter is a composite
`1:<leagueId>:<teamId>:<swid>:<nonce>` string, and that trailing `<nonce>` is
served, as a bare integer body, by

    GET {BASE}/seasons/{season}/segments/0/leagues/{leagueId}
        /teams/{teamId}/draftSecurity

authenticated with nothing but the `espn_s2`/`SWID` cookies a normal login
already produces. Once that number is in hand, the whole URL can be minted
directly -- no Playwright, no second login, no piggybacking on a browser
window the human has to keep open just so this process can watch its
socket.

That only works when the team id is known up front, which a browser-relayed
JOIN never needed (it learned its own team from the socket's own TOKEN
frame -- see draft_listener.py). So this path is the default in api/live.py
only when the pasted URL carried a `teamId=`; `run_listener` stays as the
fallback for the waiting-room URL that doesn't.

`DraftListener` itself is untouched -- `run_socket_listener` feeds it frames
exactly like `run_listener` does, just from a socket this process opened
itself instead of one it observed.
"""
import json
import time
from pathlib import Path

from pipeline.espn_league import BASE, STATE_PATH

SOCKET_HOST = "wss://fantasydraft.espn.com"

# Headers the real request was verified against (see this module's
# docstring). `x-fantasy-source: kona` in particular: without it ESPN has,
# elsewhere in this codebase, been observed to silently degrade a response
# (the player-directory endpoint caps at 50 rows with no error) rather than
# reject it outright, so there is no test that would catch its absence --
# only the header being present at all protects against that failure mode.
DRAFT_SECURITY_HEADERS = {
    "accept": "application/json",
    "x-fantasy-source": "kona",
    "origin": "https://fantasy.espn.com",
    "referer": "https://fantasy.espn.com/",
}


def _draft_security_url(league_id, team_id, season) -> str:
    return (f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}"
            f"/teams/{team_id}/draftSecurity")


def draft_security_token(fetch, league_id, team_id, season: int) -> int:
    """The socket token's trailing component, minted fresh rather than
    observed from a browser.

    `fetch(url) -> body` is injected, the same seam `import_seasons` and
    `EspnClient.get_json` already use, so this is testable with no network.
    Unlike those callers, the body here is not a JSON object -- ESPN serves
    a bare integer, e.g. `2142383582`, as the whole response -- so `fetch`
    may hand back an already-parsed int, or the raw text/bytes a real HTTP
    client returns; both are accepted.

    A body that is not an integer under any reading (an HTML error page, an
    empty string, a JSON object ESPN started wrapping it in) is a real
    failure -- most likely expired or missing `espn_s2`/`SWID` cookies, or
    ESPN having changed this undocumented endpoint's shape -- and must be
    loud rather than quietly minting a garbage socket URL that would fail
    far away from this stack trace.
    """
    url = _draft_security_url(league_id, team_id, season)
    body = fetch(url)
    try:
        if isinstance(body, bool):
            raise ValueError("bool is not an integer token")
        if isinstance(body, int):
            return body
        if isinstance(body, float):
            if not body.is_integer():
                raise ValueError("non-integral float")
            return int(body)
        return int(str(body).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ESPN's draftSecurity endpoint did not return an integer "
            f"token for league {league_id} team {team_id} season {season} "
            f"(got {body!r}) -- espn_s2/SWID may have expired, or ESPN has "
            "changed this endpoint") from exc


def socket_url(league_id, team_id, swid: str, token) -> str:
    """The URL ESPN's own draft room opens, minted without a browser.

    Checked against the one real capture this codebase has
    (tests/fixtures/espn_draft_socket.jsonl's `ws-open` record):

      - `2` is the league id -- matches both the URL's own path segment
        (`league-<id>`) and the second field of the TOKEN frame's
        `1:<league>:<team>:<swid>:<nonce>` value (see
        draft_listener._team_id_from_token).
      - `4` is the SWID -- matches the fourth field of that same TOKEN
        value, which is also the literal `SWID` cookie.
      - `5` is exactly that `1:<league>:<team>:<swid>:<nonce>` string --
        byte for byte equal to the real TOKEN frame captured in the
        fixture. `draft_security_token` supplies only the trailing
        `<nonce>`; this function assembles the rest.
      - `1`, `6`, `7`, `8` are copied verbatim from the capture (`1=1`,
        `6=false`, `7=false`, `8=KONA`). Nothing here explains what they
        mean; they have simply never been observed to vary.

    `3` is passed `team_id` on the strength of one piece of internal
    evidence, not an independent confirmation: in the capture, `3=2` and
    the <team> field embedded inside `5` is also `2` -- i.e. the outer
    parameters 1-4 exactly reproduce, in order, the first four
    colon-separated fields of `5` (a literal `1`, then league, then team,
    then swid). That is suggestive of a real field rather than a
    coincidence of one number landing on the value 2 twice, but there is
    only ever one team's session in the capture -- nothing here has been
    checked against a draft where the URL's own team differed from `3`.
    Treat `3=team_id` as the best account of the evidence available, not a
    proven fact.

    The capture also carries a trailing `nocache=<n>`, dropped here: every
    other parameter repeats verbatim across the whole session's one
    connection, and `nocache` looks like an ordinary cache-buster a fetch
    layer adds, not a value the socket protocol itself reads -- but, same
    as `3`, that has never been checked against a second connection either.
    """
    return (f"{SOCKET_HOST}/game-1/league-{league_id}/JOIN"
            f"?1=1&2={league_id}&3={team_id}&4={swid}"
            f"&5=1:{league_id}:{team_id}:{swid}:{token}"
            f"&6=false&7=false&8=KONA")


def load_cookies(state_path: str = STATE_PATH) -> dict:
    """`espn_s2` and `SWID` out of the Playwright storage state a normal
    login already wrote -- the one piece of browser-derived state this
    module still depends on, since minting a token needs an authenticated
    session and there is nowhere else that lives.

    Raises RuntimeError, not KeyError, when the file is missing or lacks
    either cookie: that is the "you need to log in once" case a user has to
    be told about in words, not a stack trace pointing at a dict lookup.
    """
    path = Path(state_path)
    if not path.exists():
        raise RuntimeError(
            f"No saved ESPN login at {path} -- log in once (run an import, "
            "or anything that opens EspnClient) before connecting to a "
            "live draft.")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read the saved ESPN login at {path}: {exc}") from exc
    cookies = {c.get("name"): c.get("value")
               for c in (data.get("cookies") or []) if c.get("name")}
    missing = [name for name in ("espn_s2", "SWID") if not cookies.get(name)]
    if missing:
        raise RuntimeError(
            f"{path} is missing {' and '.join(missing)} -- the saved ESPN "
            "login has expired or never completed; log in again.")
    return {"espn_s2": cookies["espn_s2"], "SWID": cookies["SWID"]}


def http_fetch(cookies: dict):
    """A `fetch(url) -> body` callable hitting ESPN directly over httpx.

    This is the whole point of the task: `EspnClient.get_json` goes through
    a Playwright browser context, which is exactly the dependency being
    removed. httpx is already a project dependency (see requirements.txt);
    nothing new is needed to make one GET request with two cookies and four
    headers.
    """
    import httpx

    def fetch(url: str):
        response = httpx.get(url, cookies=cookies,
                             headers=DRAFT_SECURITY_HEADERS, timeout=10.0)
        response.raise_for_status()
        return response.text
    return fetch


# Observed in the fixture (tests/fixtures/espn_draft_socket.jsonl): the
# *client* sends `PING PING%20<ms>` and the server answers `PONG ...` a
# frame or two later, at roughly this cadence -- there is no PING from the
# server first. Whether ESPN actually drops a connection that never pings
# was never going to be testable without risking the live session this
# whole task exists to avoid disturbing, so this matches the real client's
# behaviour rather than finding out the hard way mid-draft.
PING_INTERVAL_SECONDS = 15.0
# How often the recv loop wakes up to check stop_event and the ping clock,
# even with no frame waiting -- same cadence run_listener's poll loop uses.
RECV_POLL_SECONDS = 1.0


def _ping_message() -> str:
    """Mirrors the client's own frame exactly, `%20` and all -- that is a
    literal percent-two-zero in the fixture's captured bytes, not a decoded
    space, so this reproduces it rather than "fixing" it into a real space.
    Whether the server parses the argument at all, or only cares that the
    verb is PING, was never established; there is no reason to send
    anything ESPN's own client didn't."""
    return f"PING PING%20{int(time.time() * 1000)}\n"


def _connect(url: str, cookie_header: str):
    """Open the real connection. The one part of this module that touches
    the network -- tests monkeypatch this name to inject a fake socket
    (`.recv(timeout=...)` / `.send(...)` / `.close()`), the same way
    api/live.py's tests monkeypatch `run_listener` wholesale rather than
    reaching into Playwright.

    `ping_interval=None` turns off the library's own protocol-level ping:
    ESPN speaks its own application-level PING/PONG as ordinary text
    frames (see `_ping_message`), a different mechanism entirely, and
    leaving the library's on would just be a second, redundant heartbeat.
    """
    from websockets.sync.client import connect

    return connect(url, additional_headers={"Cookie": cookie_header},
                   origin="https://fantasy.espn.com", ping_interval=None,
                   open_timeout=10)


def run_socket_listener(listener, league_id, team_id, swid, token,
                        on_change=None, stop_event=None) -> None:
    """Connect directly to ESPN's draft socket and feed it into `listener`.

    Blocking -- the caller runs it on a thread. Matches `run_listener`'s
    contract on purpose (`on_change` fires exactly when `listener.on_frame`
    reports a change; `stop_event` is polled cooperatively and the
    connection is always closed on the way out, success or exception) so
    api/live.py can call either one from the same `pump()` shape.

    Authenticates the handshake with the `SWID` cookie alone -- NOT the
    `espn_s2` session cookie. Verified against a live draft under three
    conditions (with espn_s2, with SWID only, with no cookie): the socket
    connects and streams real SELECTED frames whenever the URL carries a
    self-minted token and the handshake carries SWID, and rejects the bare
    no-cookie case. espn_s2 was accepted but never required. That is the
    whole reason this path can watch a stranger's draft: `swid` comes off
    the draft URL's memberId and `token` is minted by the bookmarklet from
    the user's own ESPN session, so nothing here needs a saved login on
    this machine -- the account session cookie stays in the user's browser
    and never reaches this process.

    `swid` and `token` are taken as already-resolved values rather than
    derived here, matching the split `draft_security_token` already draws:
    minting a token needs a network call, and this function's job is only
    the socket once one exists.
    """
    cookie_header = f"SWID={swid}"
    url = socket_url(league_id, team_id, swid, token)
    ws = _connect(url, cookie_header)
    try:
        last_ping = time.monotonic()
        while stop_event is None or not stop_event.is_set():
            now = time.monotonic()
            if now - last_ping >= PING_INTERVAL_SECONDS:
                ws.send(_ping_message())
                last_ping = now
            try:
                frame = ws.recv(timeout=RECV_POLL_SECONDS)
            except TimeoutError:
                continue
            if listener.on_frame(frame) and on_change is not None:
                on_change()
    finally:
        # Best effort, mirroring run_listener's own cleanup: a socket the
        # server already dropped cannot be closed twice without complaint,
        # and failing to close it is not worth losing whatever picks were
        # already collected over.
        try:
            ws.close()
        except Exception:                              # noqa: BLE001
            pass
