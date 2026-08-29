"""pipeline.farm_claims: the file lock that keeps two farm processes out of
the same ESPN mock room.

WHY THIS IS WORTH TESTING AT ALL, given it is ninety lines of `os.open`.
Because the failure it prevents is silent. Two processes that pick the same
room take two of its eight seats, both compute the same `draft_id` (a content
hash of league and picks), and the second `record()` overwrites the first --
leaving ONE bot seat labelled `my_slot` and the other sitting in the corpus
indistinguishable from a person, teaching a fitted model that somebody drafts
at random. Nothing downstream can detect that, so the claim has to be right
here.

Every test drives the real module against a real directory (`tmp_path`), with
the clock passed in rather than mocked -- `claim`, `claimed` and `_is_stale`
all take `now`, which is what makes a ninety-minute TTL testable in a
millisecond.
"""
import os
import time

import pytest

from pipeline import farm_claims as fc


@pytest.fixture(autouse=True)
def claim_dir(tmp_path, monkeypatch):
    """Never the real `data/farm-claims`: the owner may have a farm running
    against it, and a test that swept its claims would put two processes in
    one room -- the exact bug this module exists to prevent."""
    monkeypatch.setattr(fc, "CLAIM_DIR", str(tmp_path / "claims"))
    return tmp_path / "claims"


def test_one_of_two_processes_racing_on_a_room_gets_it():
    assert fc.claim(1234) is True
    assert fc.claim(1234) is False
    assert fc.claimed() == {"1234"}


def test_a_released_room_can_be_taken_again():
    assert fc.claim(1234) is True
    fc.release(1234)
    assert fc.claimed() == set()
    assert fc.claim(1234) is True


def test_releasing_a_room_we_never_held_is_harmless():
    fc.release(9999)                     # no file, no exception
    assert fc.claimed() == set()


def test_a_claim_written_a_moment_ago_is_not_swept_as_corrupt(claim_dir):
    """THE RACE THE EXCLUSIVE CREATE DOES NOT COVER. `os.open` publishes a
    ZERO-LENGTH file; the pid and timestamp land on the next line of code. A
    second process reading in that window sees a file it cannot parse, and
    "unparseable therefore stale therefore delete it" would sweep a live
    claim and put both processes in the same room."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    (claim_dir / "1234").write_text("")          # mid-write, as seen by a peer
    assert fc.claimed() == {"1234"}
    assert fc.claim(1234) is False
    assert (claim_dir / "1234").exists()


def test_a_file_that_never_becomes_a_claim_does_retire(claim_dir):
    """The other half of the same rule: the grace is a delay, not an amnesty,
    so one hand-mangled file cannot fence a room off forever."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    corrupt = claim_dir / "1234"
    corrupt.write_text("not a claim at all")
    old = time.time() - fc.CLAIM_GRACE_SECONDS - 1
    os.utime(corrupt, (old, old))
    assert fc.claimed() == set()
    assert not corrupt.exists()


def test_a_claim_from_a_dead_process_is_swept(claim_dir):
    """A farm killed mid-draft leaves its file behind, and the room has to
    come back rather than being lost for the rest of the run."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    # A pid that cannot exist: Linux and macOS both reject 0 as a target and
    # `os.kill(0, 0)` would signal our own process GROUP, so a large unused
    # one is used instead and checked to be absent first.
    dead = 4_000_000
    if _exists(dead):
        pytest.skip("pid 4000000 is in use on this machine")
    (claim_dir / "1234").write_text(f"{dead} {time.time()}")
    assert fc.claimed() == set()
    assert fc.claim(1234) is True


def test_a_claim_older_than_the_ttl_is_swept_even_if_the_pid_is_alive():
    """The backstop for a pid recycled onto some unrelated process, which is
    rare but not impossible on a machine that has been up for weeks."""
    now = time.time()
    assert fc.claim(1234, now=now - fc.CLAIM_TTL_SECONDS - 1) is True
    assert fc.claimed(now=now) == set()
    assert fc.claim(1234, now=now) is True


def test_a_live_claim_survives_a_sweep():
    now = time.time()
    assert fc.claim(1234, now=now - fc.CLAIM_TTL_SECONDS + 60) is True
    assert fc.claimed(now=now) == {"1234"}


def test_releasing_somebody_elses_claim_does_nothing(claim_dir):
    """A process that stalled past the TTL can have had its room swept and
    re-claimed while it was still playing. An unconditional unlink on the way
    out would then free a room another process is sitting in."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    # A pid that is alive and is not ours: the pytest process's own parent.
    other = os.getppid()
    assert other != os.getpid()
    (claim_dir / "1234").write_text(f"{other} {time.time()}")
    fc.release(1234)
    assert (claim_dir / "1234").exists()
    assert fc.claimed() == {"1234"}


def test_claims_are_ids_as_strings_so_they_drop_into_exclude():
    """`pick_room`'s `exclude` normalises to `str`, and the farm loop unions
    this set straight into it -- so the two have to agree about the type."""
    fc.claim(1234)
    fc.claim("5678")
    assert fc.claimed() == {"1234", "5678"}


def _exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# Refreshing a claim: a draft can outlast any fixed TTL.
# ---------------------------------------------------------------------------


def test_the_ttl_covers_the_longest_room_the_farm_now_joins():
    """192 picks on ESPN's 30-second clock is 96 minutes of picking, plus up
    to 15 minutes holding the seat before the room starts. The old 90-minute
    TTL was under that, which would have let a live draft's claim age out."""
    twelve_by_sixteen = 12 * 16 * 30 + 15 * 60
    assert fc.CLAIM_TTL_SECONDS > twelve_by_sixteen


def test_a_touched_claim_does_not_age_out_under_a_long_draft():
    """The point of the refresh: the TTL then bounds the gap between two
    picks rather than the whole draft."""
    now = time.time()
    assert fc.claim(1234, now=now - fc.CLAIM_TTL_SECONDS + 60) is True
    assert fc.touch(1234, now=now) is True
    later = now + fc.CLAIM_TTL_SECONDS - 60
    assert fc.claimed(now=later) == {"1234"}
    # And without the refresh it would have gone.
    assert fc.claimed(now=now + fc.CLAIM_TTL_SECONDS + 60) == set()


def test_touching_somebody_elses_claim_does_nothing(claim_dir):
    """Same rule as `release`, from the other side: a claim naming another
    pid is that process's room, and stamping it would keep another farm's
    room alive (or worse, hand it to us)."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    other = claim_dir / "1234"
    other.write_text(f"{os.getpid() + 1} {time.time()}")

    assert fc.touch(1234) is False
    assert other.read_text().startswith(str(os.getpid() + 1))


def test_touching_an_unreadable_claim_does_nothing(claim_dir):
    """A claim that will not parse is either corrupt or another process's, in
    the microsecond between its exclusive create and its write. Neither is
    ours to stamp."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    (claim_dir / "1234").write_text("")

    assert fc.touch(1234) is False


def test_touching_a_room_whose_claim_vanished_puts_it_back():
    """We are demonstrably still in that room -- we just made a pick in it."""
    assert fc.touch(1234) is True
    assert fc.claimed() == {"1234"}
    assert fc.claim(1234) is False      # and it is ours, so nobody else's


def test_a_refresh_that_cannot_write_is_not_an_error(monkeypatch, tmp_path):
    """A failed refresh must not end a night's farming: the worst case is the
    claim ageing out, which is where we were before it existed."""
    monkeypatch.setattr(fc, "CLAIM_DIR", str(tmp_path / "nope" / "claims"))
    fc.claim(1234)
    monkeypatch.setattr("builtins.open", _refuse)

    assert fc.touch(1234) is False


def _refuse(*a, **k):
    raise OSError("read-only filesystem")
