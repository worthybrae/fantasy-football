"""Sleep-tolerant scheduler for `pipeline.auto_refit`.

Wakes every few hours, runs ONE `auto_refit` as a subprocess, and sleeps again.
Because `auto_refit` skips cheaply unless enough new labelled drafts have
accrued, most wakes cost a sub-second query and exit; a wake does the expensive
fit + leave-one-draft-out ladder only when there is genuinely new data. So a
tight-ish interval is fine and safe.

WHY A SLEEP LOOP AND NOT CRON. The laptop sleeps. cron fires on wall-clock time
and a missed slot is simply lost; a background process, by contrast, PAUSES when
the machine sleeps and RESUMES where it left off on wake -- `time.sleep`
included -- so this loop just picks up a little late after a lid-close and
carries on. There is no state to miss because the accrual gate lives in
`auto_refit`, not in the schedule: whenever this next runs, it does exactly the
work the current corpus justifies.

IT COEXISTS WITH THE FARM. This never writes the corpus (auto_refit opens it
read-only) and never spawns parallel fit workers (the ladder is single process),
so it runs happily alongside `make farm-mocks` writing new drafts. Each run is a
fresh subprocess, so a crash in one fit is logged and the loop survives to try
again next interval rather than taking the daemon down with it.

START IT (do this yourself; it is not started for you):

    cd /path/to/fantasy-football
    nohup .venv/bin/python -m pipeline.auto_refit_daemon \\
        --league-db data/leagues/1251381776.duckdb \\
        --interval-hours 3 >> auto_refit_daemon.out 2>&1 &

Watch it:   tail -f auto_refit_daemon.out
Stop it:    kill the PID that `nohup ... &` printed (or `pkill -f auto_refit_daemon`).

Every argument other than `--interval-hours` is forwarded verbatim to
`pipeline.auto_refit` (e.g. `--league-db`, `--min-new-drafts`), so this wrapper
adds only the schedule.
"""
from __future__ import annotations

import datetime as _dt
import subprocess
import sys
import time


def _stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _split_args(argv):
    """Pull `--interval-hours H` out; forward everything else to auto_refit."""
    interval_hours = 3.0
    forward = []
    i = 0
    while i < len(argv):
        if argv[i] == "--interval-hours" and i + 1 < len(argv):
            interval_hours = float(argv[i + 1])
            i += 2
        else:
            forward.append(argv[i])
            i += 1
    return interval_hours, forward


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    interval_hours, forward = _split_args(argv)
    interval = max(0.0, interval_hours) * 3600.0
    cmd = [sys.executable, "-m", "pipeline.auto_refit", *forward]

    print(f"[daemon {_stamp()}] starting; every {interval_hours}h run: "
          f"{' '.join(cmd)}", flush=True)
    while True:
        print(f"[daemon {_stamp()}] --- run start", flush=True)
        try:
            rc = subprocess.run(cmd).returncode
            print(f"[daemon {_stamp()}] --- run end rc={rc}", flush=True)
        except Exception as exc:                      # never let one run kill the loop
            print(f"[daemon {_stamp()}] run raised {exc!r}; continuing", flush=True)
        print(f"[daemon {_stamp()}] sleeping {interval_hours}h", flush=True)
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print(f"[daemon {_stamp()}] interrupted; exiting", flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
