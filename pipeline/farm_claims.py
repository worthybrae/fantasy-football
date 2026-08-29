"""Which rooms other farm processes are already sitting in.

WHY THIS EXISTS. `farm()` keeps a `played` set so one process does not
rejoin a room it has already finished. That set lives in memory, so two
processes started together share nothing -- and because `pick_room` ranks
deterministically (by `teamsJoined`, then experience, then soonest start),
two processes polling the same lobby pick the SAME room. That is worse than
wasteful:

  - we would hold two of the eight seats, so a quarter of the room we are
    trying to learn from is our own epsilon-greedy noise;
  - `draft_id` is a content hash of (league_id, picks), so both processes
    compute the SAME id and the second `record()` overwrites the first;
  - only one of the two seats survives as `my_slot`, so the OTHER bot seat
    is indistinguishable from a real drafter and gets fitted on as if it
    were one.

The last of those is the dangerous one: it silently teaches the model that
somebody drafts at random.

WHY A DIRECTORY OF FILES rather than a table. The corpus is DuckDB, which
allows one writer; making every process take that write lock to announce an
intention would serialize exactly the thing we are trying to parallelize. A
claim is a single `O_CREAT | O_EXCL` create, which is atomic on every
filesystem this runs on, needs no daemon, and leaves nothing to clean up if
the whole machine dies.

STALE CLAIMS. A process killed mid-draft leaves its file behind. Two things
retire one: the pid it names is gone, or the claim is older than
`CLAIM_TTL_SECONDS`. The TTL is the backstop for a pid that has been
recycled onto some unrelated process, which is rare but not impossible on a
machine that has been up for weeks.

A LIVE CLAIM IS RE-STAMPED AFTER EVERY PICK (`touch`), so the TTL never has
to cover a whole draft -- only the gap between two picks. That matters now
that the farm joins 12-team rooms: 192 picks on a 30-second clock is 96
minutes, and a TTL shorter than the draft would let a second process claim a
room the first is still sitting in.

A CLAIM IS EMPTY FOR AN INSTANT AFTER IT IS MADE, and that instant is the
one thing an exclusive create does not cover. `os.open` with `O_CREAT |
O_EXCL` publishes a zero-length file; the pid and timestamp land on the next
line of code. A second process reading in between sees a file it cannot
parse -- and "unparseable therefore stale therefore delete it" would sweep a
live claim away and put both processes in the same room, which is the entire
failure this module exists to prevent. So a file whose CONTENT cannot be
read falls back to its MTIME (see `_is_stale`): a claim written moments ago
is young whatever is in it, and a genuinely corrupt one still retires, just
`CLAIM_GRACE_SECONDS` later instead of instantly.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

CLAIM_DIR = os.environ.get("FARM_CLAIM_DIR", "data/farm-claims")

# Longer than any draft can run -- and "any draft" got longer when the farm
# started joining 12-team rooms. 192 picks on ESPN's 30-second clock is 96
# minutes of picking alone, before the up-to-15-minute lead the seat is held
# for and before a room that pauses. The old 90 minutes was under that, which
# would have let a LIVE draft's claim age out and its room be handed to
# another farm process mid-draft -- two of our seats in one room, which is
# the whole failure this module exists to prevent.
#
# Three hours covers a 12x16 room at the slowest clock with room to spare,
# and `touch` refreshes the stamp on every pick anyway, so the TTL only ever
# has to cover the gap between two picks rather than a whole draft. What it
# costs is how long a crashed process fences its room off, and that is
# bounded by the pid check long before the TTL matters: the TTL is the
# backstop for a recycled pid, not the ordinary path.
CLAIM_TTL_SECONDS = 3 * 60 * 60

# How long an unreadable claim file is given the benefit of the doubt before
# it is swept. Only ever compared against mtime, and only for a file whose
# contents will not parse: the write that fills a fresh claim follows its
# create by microseconds, so a minute is many thousands of times the window
# it has to cover while still retiring a hand-mangled file the same session.
CLAIM_GRACE_SECONDS = 60.0


def _dir() -> Path:
    path = Path(CLAIM_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _alive(pid: int) -> bool:
    """Does this pid still exist? Signal 0 checks without delivering."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Somebody else's process, so it exists. Not ours to claim against,
        # but "exists" is the only question here.
        return True
    return True


def _read(path: Path) -> tuple:
    """(pid, stamp) from a claim file, or (None, None) if it will not read."""
    try:
        pid, stamp = path.read_text().split(None, 1)
        return int(pid), float(stamp)
    except (OSError, ValueError):
        return None, None


def _is_stale(path: Path, now: float) -> bool:
    """Whether this claim may be swept and the room handed to somebody else.

    Deliberately conservative in both directions of doubt. A claim we cannot
    read is NOT assumed dead -- see the module docstring: every claim is
    unreadable for the microsecond between its exclusive create and its
    write, and sweeping one then hands the same room to two processes. It is
    aged off its mtime instead. A claim that has vanished under us is not
    stale either; there is nothing left to sweep, and saying "yes" would send
    the caller off to unlink a name that may by then belong to a different
    process's fresh claim.
    """
    pid, stamp = _read(path)
    if pid is None:
        try:
            age = now - path.stat().st_mtime
        except OSError:
            return False                # gone already; nothing to retire
        return age > CLAIM_GRACE_SECONDS
    if now - stamp > CLAIM_TTL_SECONDS:
        return True
    return not _alive(pid)


def claimed(now: float | None = None) -> set:
    """League ids currently held by a live farm process, stale ones swept."""
    now = time.time() if now is None else now
    held = set()
    for path in _dir().iterdir():
        if not path.is_file():
            continue
        if _is_stale(path, now):
            try:
                path.unlink()
            except OSError:
                pass                    # another process swept it first
            continue
        held.add(path.name)
    return held


def claim(league_id, now: float | None = None) -> bool:
    """Take the room, or return False if somebody else already has it.

    `O_CREAT | O_EXCL` is the whole mechanism: exactly one of two processes
    racing on the same room gets the file, and the loser picks another room
    rather than sitting in a seat beside its own twin.
    """
    now = time.time() if now is None else now
    path = _dir() / str(league_id)
    if path.exists() and _is_stale(path, now):
        try:
            path.unlink()
        except OSError:
            pass
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{os.getpid()} {now}")
    return True


def touch(league_id, now: float | None = None) -> bool:
    """Say the room is still ours. Called after every pick.

    A claim is stamped once, when it is taken, and a draft is long: without
    this the TTL alone has to cover the whole room, and a room that runs
    longer than the TTL is a room another farm process may claim while we are
    still sitting in it. Refreshing per pick turns the TTL into a bound on
    the gap between two picks -- a minute or two -- rather than on a draft.

    ONLY OUR OWN CLAIM. Same ownership check as `release`, and for the same
    reason from the other side: a claim naming another pid is that process's
    room, and one naming nobody at all is either a corrupt file or another
    process's claim in the microsecond between its exclusive create and its
    write (see the module docstring). Neither is ours to stamp. A claim that
    has VANISHED is re-created, because we are demonstrably still in that
    room.

    Never raises, and returns whether the stamp landed. A refresh that fails
    is not a reason to stop drafting: the worst case is the claim ageing out,
    which is where we were before this existed.
    """
    now = time.time() if now is None else now
    path = _dir() / str(league_id)
    if path.exists():
        pid, _stamp = _read(path)
        if pid != os.getpid():
            return False
    try:
        # Truncate-and-write, not a rename: a reader catching the empty
        # instant sees an unreadable claim, which `_is_stale` already ages
        # off its (freshly bumped) mtime rather than sweeping.
        with open(path, "w") as fh:
            fh.write(f"{os.getpid()} {now}")
    except OSError:
        return False
    return True


def release(league_id) -> None:
    """Give the room back, if the claim on it is still ours.

    The ownership check is not ceremony. A process that stalled long enough
    for its claim to age past the TTL can have had the room swept and
    re-claimed by somebody else while it was still playing; an unconditional
    unlink on the way out would then delete the OTHER process's live claim
    and free a room it is sitting in. Reading the pid first costs one open
    and removes that entirely.

    Never raises -- a release that fails must not end a night's farming, and
    the TTL retires the file either way.
    """
    path = _dir() / str(league_id)
    pid, _stamp = _read(path)
    if pid is not None and pid != os.getpid():
        return
    try:
        path.unlink()
    except OSError:
        pass
