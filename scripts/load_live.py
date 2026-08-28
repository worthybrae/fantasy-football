"""Drive N concurrent live draft rooms against a running server.

What it measures, and why those numbers and not others:

  CONNECT. One room's connect is the expensive request in this app -- it
  provisions a per-league DuckDB file by copying every universal table out of
  the 215MB shared database, then builds the whole board and pool for it.
  Bursty connects are the shape that actually happens (a league's twelve
  managers all click the bookmark in the same minute), so `--ramp` spreads the
  connects over a few seconds rather than starting them one at a time. Both
  the latency and the failures are reported: a connect that 503s under load is
  the thing this test is for.

  POLLING. Once connected, a room polls `/api/live/state` and
  `/api/live/board` every 2.5s, which is what the browser does. Those two are
  reported separately because they cost different things -- state recomputes,
  board reads.

  MEMORY. Peak RSS of the server process, sampled every second, because the
  per-league connections and per-room sessions are what a scale-out is
  supposed to bound.

It starts no server. Point it at one running with the draft socket faked --
`scripts/load_server.py`, or `LIVE_FAKE_SOCKET=1` if that switch has landed --
or every room will try to open a websocket to ESPN with a token ESPN never
minted.

    .venv/bin/python scripts/load_live.py --url http://127.0.0.1:8010 \
        --sessions 100 --seconds 120 --pid <server pid> --ramp 20

EACH SESSION GETS ITS OWN LEAGUE ID, which is the point: a shared id would
provision one file and reuse it, and the copy is most of what a first connect
costs. So the run leaves `data/leagues/9000NN.duckdb` behind, ~32MB each, and
deletes exactly the ids it created on the way out (`--keep-leagues` to keep
them). Nothing else under data/ is touched.
"""
import argparse
import math
import platform
import subprocess
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import httpx

# What the browser does: `web/src` polls both endpoints on this cadence.
POLL_SECONDS = 2.5
# The acceptance thresholds this run is judged against.
P95_STATE_BUDGET_MS = 300.0
PEAK_RSS_BUDGET_MB = 6 * 1024


def percentile(values, q: float) -> float:
    """Nearest-rank percentile. `q` is 0-100.

    Nearest-rank rather than an interpolating one on purpose: every number
    reported here is a latency that actually happened, so a p99 of "1.4s"
    names a real request somebody could go and look at.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = math.ceil(q / 100.0 * len(ordered))
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def read_rss_kb(pid: int):
    """Resident set size of one process in KB, or None if it has gone."""
    if platform.system() == "Linux":
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        except (OSError, ValueError):
            return None
        return None
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout.strip()
    return int(text) if text.isdigit() else None


def process_table():
    """(pid, ppid, rss_kb) for every process, or None if `ps` would not say.

    One `ps -A` per sample rather than one per process: the build pool has a
    child per worker and they come and go, so there is no fixed list to walk.
    """
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows or None


def rss_kb(pid: int):
    """(own, whole tree) resident set size in KB, or None.

    THE TREE IS THE NUMBER THAT MATTERS. With `LIVE_BUILD_WORKERS` above zero
    the connect's build runs in a child process (api/live_build.py), and its
    memory is exactly the memory the acceptance threshold is about -- reading
    only the parent would report a server that got dramatically lighter the
    moment the work moved next door.

    It is an upper bound, and deliberately so: a forked child shares most of
    its parent's pages copy-on-write and every process maps the same database
    file, so adding the RSS figures counts some pages more than once. Both
    numbers are reported, so a run can be read either way.
    """
    rows = process_table()
    if rows is None:
        own = read_rss_kb(pid)
        return None if own is None else (own, own)
    children = defaultdict(list)
    resident = {}
    for child, parent, kb in rows:
        children[parent].append(child)
        resident[child] = kb
    if pid not in resident:
        return None
    total, stack, seen = 0, [pid], set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        total += resident.get(current, 0)
        stack.extend(children.get(current, ()))
    return resident[pid], total


class RssSampler(threading.Thread):
    """Samples the server's RSS once a second for the life of the run."""

    def __init__(self, pid: int, interval: float = 1.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.samples = []       # whole tree
        self.own = []           # the server process by itself
        # `_halt`, not `_stop`: threading.Thread already has a `_stop`
        # method of its own and shadowing it breaks `join`.
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            reading = rss_kb(self.pid)
            if reading is not None:
                self.own.append(reading[0])
                self.samples.append(reading[1])
            self._halt.wait(self.interval)

    def halt(self):
        self._halt.set()


class Metrics:
    """Latencies and outcomes, per endpoint, across every session thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latencies = defaultdict(list)
        self.statuses = defaultdict(Counter)
        # A few bodies per endpoint, because "503×7" does not say WHICH 503
        # and the answers this API gives are specific ("the previous listener
        # did not stop in time", "no live session").
        self.details = defaultdict(list)

    def record(self, name: str, ms: float, outcome, detail=None) -> None:
        with self.lock:
            self.latencies[name].append(ms)
            self.statuses[name][outcome] += 1
            if detail and len(self.details[name]) < 3:
                self.details[name].append(detail)

    def ok_count(self, name: str) -> int:
        return self.statuses[name][200]

    def bad(self, name: str) -> Counter:
        return Counter({k: v for k, v in self.statuses[name].items() if k != 200})


class Deadline:
    """When the run stops polling.

    Not a fixed `start + seconds`, and the reason is the shape of this app: a
    connect takes tens of seconds under load, so with a hundred sessions the
    last room can still be building when a fixed deadline expires -- and the
    percentiles would then describe a handful of rooms, not a hundred. So
    every successful connect pushes the deadline out to `--seconds` after
    itself. The window that gets measured therefore always has every room in
    it, at the cost of the early rooms polling for longer.

    `hard` is the wall that keeps a pathological run finite.
    """

    def __init__(self, at: float, hard: float):
        self._at = at
        self._hard = hard
        self._lock = threading.Lock()

    def at(self) -> float:
        return min(self._at, self._hard)

    def not_before(self, when: float) -> None:
        with self._lock:
            self._at = max(self._at, when)


class Tally:
    """What happened to the ROOMS, as opposed to the requests.

    Worth counting separately because a run can look perfectly healthy at the
    request level and be measuring nothing. The failure that costs a whole
    run is a room cookie the client never sends back -- `espn_live` is set
    `Secure`, and no http client returns a Secure cookie over http:// -- and
    the symptom is every request landing in a fresh empty room: 200s from
    `/api/live/state` that say `active: false`, 404s from `/api/live/board`.
    So the first state poll of each session is read, and a run where the
    rooms never went active is called out rather than reported as a pass.
    (`scripts/load_server.py` sets the local-http allowance that avoids it.)
    """

    def __init__(self):
        self.polled = []
        self.active = []


def timed(metrics: Metrics, name: str, call):
    """Run one request, record how long it took and how it ended.

    A request that raises is recorded too, under its exception class name --
    a connection reset under load is a result, not a gap in the data, and
    dropping it would flatter every percentile below it.
    """
    started = time.monotonic()
    detail = None
    try:
        response = call()
        outcome = response.status_code
        if outcome != 200:
            detail = response.text[:200]
    except Exception as exc:      # noqa: BLE001 -- the failure IS the finding
        response, outcome = None, type(exc).__name__
        detail = str(exc)[:200]
    ms = (time.monotonic() - started) * 1000.0
    metrics.record(name, ms, outcome, detail)
    return response


def run_session(index: int, args, metrics: Metrics, deadline: Deadline,
                tally: Tally, ready: threading.Event) -> None:
    league_id = str(args.league_base + index)
    team_id = 1 + index % args.teams
    swid = "{%s}" % uuid.uuid4()
    ready.wait()
    # The ramp. Connects are deliberately bunched rather than paced one per
    # slot -- `--ramp 20` for 100 sessions is five connects a second, which is
    # a real league's worth of managers clicking the bookmark at once.
    stagger = args.ramp * index / max(args.sessions - 1, 1) if args.ramp else 0.0
    if stagger:
        time.sleep(stagger)

    poll_timeout = httpx.Timeout(args.poll_timeout, connect=10.0)
    with httpx.Client(base_url=args.url, timeout=poll_timeout) as client:
        # The cookie first, on its own request, exactly as the page does --
        # `/api/live/session` is what mints `espn_live`, and a connect that
        # carried no room cookie would land in no room.
        session = timed(metrics, "session",
                        lambda: client.post("/api/live/session"))
        if session is None or session.status_code != 200:
            return
        body = {"leagueId": league_id, "teamId": str(team_id), "swid": swid,
                "token": f"load-test-{index}", "season": str(args.season)}
        connect = timed(
            metrics, "connect",
            lambda: client.post("/api/live/connect-token", json=body,
                                timeout=httpx.Timeout(args.connect_timeout,
                                                      connect=10.0)))
        if connect is None or connect.status_code != 200:
            return
        # This room is up: nobody stops polling until it has had its share.
        deadline.not_before(time.monotonic() + args.seconds)
        if time.monotonic() >= deadline.at():
            # Connected, but the run is already over -- its connect latency
            # still counts, and the summary says how many rooms this was.
            return
        tally.polled.append(index)
        first = True
        while time.monotonic() < deadline.at():
            tick = time.monotonic()
            state = timed(metrics, "state", lambda: client.get("/api/live/state"))
            if first and state is not None and state.status_code == 200:
                first = False
                try:
                    tally.active.append(bool(state.json().get("active")))
                except ValueError:
                    pass
            timed(metrics, "board", lambda: client.get("/api/live/board"))
            slept = POLL_SECONDS - (time.monotonic() - tick)
            if slept > 0:
                time.sleep(min(slept, max(deadline.at() - time.monotonic(), 0.0)))
        if not args.no_stop:
            # Release the listener thread and let the server close this
            # league's connection, so the cleanup below is deleting files
            # nothing is still writing to.
            timed(metrics, "stop", lambda: client.post("/api/live/stop"))


def cleanup_leagues(args, verbose=True) -> int:
    """Delete only the league files this run created."""
    root = Path(args.leagues_root)
    removed = 0
    for index in range(args.sessions):
        league_id = args.league_base + index
        for path in (root / f"{league_id}.duckdb", root / f"{league_id}.duckdb.wal"):
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                pass
            except OSError as exc:
                if verbose:
                    print(f"  could not remove {path}: {exc}")
    return removed


def report(args, metrics: Metrics, sampler, tally: Tally, elapsed: float) -> bool:
    def line(name):
        values = metrics.latencies[name]
        if not values:
            return f"  {name:<8} no requests"
        return (f"  {name:<8} n={len(values):<6} "
                f"p50={percentile(values, 50):8.1f}ms "
                f"p95={percentile(values, 95):8.1f}ms "
                f"p99={percentile(values, 99):8.1f}ms "
                f"max={max(values):8.1f}ms")

    print()
    print("=" * 78)
    polled = len(tally.polled)
    active = sum(tally.active)
    print(f"{args.sessions} sessions · ramp {args.ramp}s · {elapsed:.0f}s wall "
          f"· {polled} rooms polled · {args.url}")
    print("=" * 78)
    print("latency")
    for name in ("session", "connect", "state", "board", "stop"):
        if metrics.latencies[name]:
            print(line(name))
    print()
    connect_ok = metrics.ok_count("connect")
    connect_bad = sum(metrics.bad("connect").values())
    print(f"connects  ok={connect_ok}  failed={connect_bad}  "
          f"p95={percentile(metrics.latencies['connect'], 95):.0f}ms")
    problems = False
    for name in ("session", "connect", "state", "board", "stop"):
        bad = metrics.bad(name)
        if bad:
            problems = True
            counts = "  ".join(f"{k}×{v}" for k, v in sorted(bad.items(), key=str))
            print(f"errors    {name:<8} {counts}")
            for text in metrics.details[name]:
                print(f"                   {text}")
    if not problems:
        print("errors    none -- every request answered 200")

    print(f"rooms     connected={metrics.ok_count('connect')}  polled={polled}  "
          f"live={active} (first /api/live/state said active)")
    if polled and active < polled:
        print(f"WARNING   {polled - active} room(s) never went active -- the "
              "numbers below do not describe a working draft. The usual cause "
              "is the Secure room cookie not coming back over http; run the "
              "server via scripts/load_server.py.")

    peak_mb = max(sampler.samples) / 1024.0 if sampler and sampler.samples else None
    if peak_mb is None:
        print("rss       not sampled (no --pid)")
    else:
        first_mb = sampler.samples[0] / 1024.0
        own_peak_mb = max(sampler.own) / 1024.0
        print(f"rss       start={first_mb:,.0f}MB  peak={peak_mb:,.0f}MB  "
              f"(server + build workers, {len(sampler.samples)} samples; "
              f"server process alone peaked at {own_peak_mb:,.0f}MB)")

    p95_state = percentile(metrics.latencies["state"], 95)
    state_ok = bool(metrics.latencies["state"]) and p95_state < P95_STATE_BUDGET_MS
    rss_ok = peak_mb is None or peak_mb < PEAK_RSS_BUDGET_MB
    poll_errors = sum(metrics.bad("state").values()) + sum(metrics.bad("board").values())
    # A run that measured nothing must not be allowed to report a pass on the
    # strength of how fast it measured nothing.
    valid = polled > 0 and active == polled and poll_errors == 0 and connect_bad == 0
    passed = state_ok and rss_ok and valid
    peak_text = f"{peak_mb:,.0f}MB" if peak_mb is not None else "unsampled"
    why = []
    if not state_ok:
        why.append("p95 state over budget")
    if not rss_ok:
        why.append("peak RSS over budget")
    if not valid:
        why.append("run not clean")
    print()
    print(f"VERDICT {'PASS' if passed else 'FAIL'}: p95 state {p95_state:.0f}ms "
          f"(budget {P95_STATE_BUDGET_MS:.0f}ms), peak RSS {peak_text} "
          f"(budget {PEAK_RSS_BUDGET_MB / 1024:.0f}GB), "
          f"{connect_bad} failed connects, {poll_errors} failed polls"
          + (f" -- {', '.join(why)}" if why else ""))
    return passed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--sessions", type=int, default=25)
    parser.add_argument(
        "--seconds", type=float, default=120,
        help="how long to poll AFTER the ramp finishes (default 120)")
    parser.add_argument(
        "--ramp", type=float, default=20,
        help="spread the connects over this many seconds (default 20)")
    parser.add_argument("--pid", type=int,
                        help="server pid, to sample its RSS once a second")
    parser.add_argument("--league-base", type=int, default=900000,
                        help="first league id; session i uses base+i")
    parser.add_argument("--teams", type=int, default=12,
                        help="team ids cycle 1..teams within each league")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--leagues-root", default="data/leagues")
    parser.add_argument("--connect-timeout", type=float, default=600.0,
                        help="read timeout for the connect POST (default 600)")
    parser.add_argument("--poll-timeout", type=float, default=30.0,
                        help="read timeout for the state/board polls")
    parser.add_argument(
        "--max-seconds", type=float, default=900,
        help="hard wall on the whole run, whatever the connects do "
             "(default 900)")
    parser.add_argument("--no-stop", action="store_true",
                        help="leave the rooms running at the end")
    parser.add_argument("--keep-leagues", action="store_true",
                        help="do not delete the data/leagues files this created")
    args = parser.parse_args(argv)

    if not args.keep_leagues:
        # Delete first as well as last: a previous run that was killed leaves
        # provisioned files behind, and reusing them would skip the copy this
        # test exists to measure.
        stale = cleanup_leagues(args, verbose=False)
        if stale:
            print(f"removed {stale} league file(s) left by an earlier run")

    metrics = Metrics()
    sampler = None
    if args.pid:
        if read_rss_kb(args.pid) is None:
            print(f"warning: pid {args.pid} is not readable -- no RSS numbers")
        else:
            sampler = RssSampler(args.pid)
            sampler.start()

    now = time.monotonic()
    deadline = Deadline(now + args.ramp + args.seconds,
                        hard=now + args.max_seconds)
    ready = threading.Event()
    tally = Tally()
    threads = [threading.Thread(target=run_session,
                                args=(i, args, metrics, deadline, tally, ready),
                                daemon=True)
               for i in range(args.sessions)]
    for thread in threads:
        thread.start()
    started = time.monotonic()
    print(f"driving {args.sessions} rooms at {args.url} for "
          f"{args.ramp + args.seconds:.0f}s ...", flush=True)
    ready.set()
    for thread in threads:
        # Generous: a session still inside a slow connect when the deadline
        # passes has to be waited out, or its result is lost.
        thread.join(timeout=args.connect_timeout + args.poll_timeout + 60)
    elapsed = time.monotonic() - started
    if sampler:
        sampler.halt()
        sampler.join(timeout=5)

    passed = report(args, metrics, sampler, tally, elapsed)

    if not args.keep_leagues:
        # After the stops, so nothing is mid-write. The server may still hold
        # a handle -- unlink is fine with that on both platforms; the file
        # goes when the last handle closes.
        time.sleep(2)
        removed = cleanup_leagues(args)
        print(f"cleanup   removed {removed} file(s) under {args.leagues_root}")
    else:
        print(f"cleanup   skipped -- league files left in {args.leagues_root}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
