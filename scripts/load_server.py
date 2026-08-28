"""Serve the API with the draft socket faked, for load testing.

`scripts/load_live.py` drives N live rooms at once. Every one of them would
otherwise open a websocket to ESPN with a token ESPN never minted, and the
connect path would additionally fetch that league's settings, its team names
and (for a league ESPN thinks is real) its whole draft history. A hundred
rooms of that is a hundred rooms of somebody else's traffic, and none of it is
the thing being measured.

So this runner starts the ordinary app and replaces exactly the parts that
talk to ESPN:

  * `api.live.run_socket_listener` -> `run_fake_socket_listener`, which
    replays data/draft_room_trace.jsonl at one pick every `--pick-interval`
    seconds. api/live.py resolves the runner off its module global on every
    connect, so this reassignment is honoured without touching that file.
  * `api.live._team_view_fetch` -> a fetch that raises. Both callers
    (`fetch_league_settings`, `fetch_team_slots`) are documented best-effort
    and return None/{} on a fetch that raises, which is exactly the "ESPN did
    not answer" path a connect already has to survive.
  * `billing.is_free_draft` -> True. It reaches ESPN's mock-lobby directory on
    a cold cache, and a False answer additionally spawns a league-history
    import per room. Answering "mock" keeps both off the wire.

Everything else is the real thing: `provision_league` really copies the
universal tables into a new file per league, `build_session` really builds,
the listener thread really runs, and `/api/live/state` and `/api/live/board`
really recompute. That is the point.

The env switch `LIVE_FAKE_SOCKET=1` in api/live.py does the first of those
three on its own. This runner still exists because the other two are what
make a hundred-room run something you can do on a laptop without pointing it
at ESPN, and because it works whether or not the switch has landed.

    .venv/bin/python scripts/load_server.py --port 8010 \
        --db /tmp/load.duckdb --workers 3
"""
import argparse
import functools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _no_espn_fetch():
    """A `fetch(url)` that always fails, which both ESPN readers handle."""
    def fetch(url):
        raise RuntimeError("load_server: outbound ESPN fetches are disabled")
    return fetch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--db",
        help="DRAFT_DB_PATH for this process. Point it at a COPY of "
             "data/nfl.duckdb: DuckDB is single-writer and a load run must "
             "not take the real file's lock.")
    parser.add_argument(
        "--pick-interval", type=float, default=2.0,
        help="seconds between replayed picks in every room (default 2.0)")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="LIVE_BUILD_WORKERS for the connect pool, if that switch exists")
    parser.add_argument(
        "--real-espn", action="store_true",
        help="leave the ESPN settings/team-name/lobby calls alone. Only the "
             "socket is faked. Do not use this for a hundred-room run.")
    args = parser.parse_args(argv)

    # Before importing anything from the app: pipeline.db reads DRAFT_DB_PATH
    # at import time into DEFAULT_PATH, and api.main builds its app -- and its
    # connection -- at import time too.
    if args.db:
        os.environ["DRAFT_DB_PATH"] = args.db
    os.environ.setdefault("WARM_ON_BOOT", "0")
    # WITHOUT THIS THE LOAD TEST MEASURES NOTHING. The `espn_live` room
    # cookie is set `Secure` unless plaintext http has been explicitly
    # allowed (api/live._set_sid_cookie), and every http client -- httpx
    # included -- refuses to send a Secure cookie back over http://. So each
    # request would land in a brand new empty room: `/api/live/board` 404s,
    # `/api/live/state` reports inactive, and the numbers describe a server
    # doing nothing. This is the same local-development allowance the README
    # documents, and this runner binds 127.0.0.1 by default.
    os.environ.setdefault("ESPN_CUSTODY_ALLOW_PLAINTEXT_HTTP", "1")
    if args.workers is not None:
        os.environ["LIVE_BUILD_WORKERS"] = str(args.workers)

    import uvicorn

    import api.live
    from api.live_fake_socket import run_fake_socket_listener
    from api.main import app

    api.live.run_socket_listener = functools.partial(
        run_fake_socket_listener, pick_interval=args.pick_interval)
    faked = ["socket"]
    if not args.real_espn:
        api.live._team_view_fetch = _no_espn_fetch
        api.live.billing.is_free_draft = lambda league_id: True
        faked += ["settings/team-name fetch", "mock-lobby lookup"]

    print(f"load_server: db={os.environ.get('DRAFT_DB_PATH', 'data/nfl.duckdb')} "
          f"pick_interval={args.pick_interval}s "
          f"build_workers={os.environ.get('LIVE_BUILD_WORKERS', 'default')}",
          flush=True)
    print(f"load_server: faked -> {', '.join(faked)}", flush=True)
    print(f"load_server: pid={os.getpid()} listening on "
          f"http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port,
                log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
