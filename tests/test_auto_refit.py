"""Offline tests for the auto-refit safety gate and the swap/revert machinery.

No corpus, no fit: the gate is a pure function and the swap logic is exercised
against a throwaway git repo in tmp_path. These pin the safety story --
  * the gate does NOT swap on a tie or any regression, DOES swap on a strict win,
  * a swap whose post-swap parity check fails is REVERTED, live model untouched,
  * the min-new-drafts skip fires when too little new data has accrued --
without ever touching the real corpus or the real `scoring/nested_prior.py`.
"""
import subprocess

import pytest

from pipeline.auto_refit import Metrics, gate, should_skip, promote


# --------------------------------------------------------------------- the gate

CHAMP = Metrics(top1=0.2600, top5=0.6800, nll=2.6900)
FLOOR = 0.2300          # the flat baseline; a real candidate must clear it


def _swaps(cand, champ=CHAMP, floor=FLOOR):
    do, _why = gate(champ, cand, floor)
    return do


def test_gate_does_not_swap_on_a_top1_tie():
    # identical top-1, even with a better log-loss: a tie is not a win.
    assert not _swaps(Metrics(top1=0.2600, top5=0.70, nll=2.50))


def test_gate_does_not_swap_when_top1_regresses():
    assert not _swaps(Metrics(top1=0.2599, top5=0.70, nll=2.50))


def test_gate_does_not_swap_when_logloss_is_worse_even_if_top1_wins():
    # top-1 strictly better but log-loss worse (higher nll) => no swap.
    assert not _swaps(Metrics(top1=0.2650, top5=0.70, nll=2.7000))


def test_gate_does_not_swap_when_below_the_flat_floor():
    # beats the champion on both metrics but does not clear the flat baseline.
    champ = Metrics(top1=0.2200, top5=0.60, nll=2.90)
    assert not _swaps(Metrics(top1=0.2250, top5=0.61, nll=2.85),
                      champ=champ, floor=0.2300)


def test_gate_swaps_on_a_strict_top1_win_with_nonworse_logloss():
    assert _swaps(Metrics(top1=0.2650, top5=0.69, nll=2.6800))


def test_gate_swaps_when_logloss_is_exactly_tied_and_top1_wins():
    # "not worse" is <=, so an equal log-loss with a strict top-1 win passes.
    assert _swaps(Metrics(top1=0.2650, top5=0.69, nll=2.6900))


# ------------------------------------------------------------------- the skip

def test_skip_fires_when_too_few_new_labelled_drafts():
    state = {"last_success_labelled": 100}
    skip, _why, new = should_skip(110, state, min_new=25)
    assert skip and new == 10


def test_skip_does_not_fire_when_enough_new_drafts_accrued():
    state = {"last_success_labelled": 100}
    skip, _why, new = should_skip(130, state, min_new=25)
    assert not skip and new == 30


def test_first_run_with_no_state_does_work_when_corpus_is_large_enough():
    skip, _why, new = should_skip(80, None, min_new=25)
    assert not skip and new == 80


def test_first_run_with_no_state_still_skips_a_tiny_corpus():
    skip, _why, _new = should_skip(5, None, min_new=25)
    assert skip


# --------------------------------------------------- the swap / parity / revert

def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo with a committed 'live artifact' and a candidate
    file on the side, so `promote` runs its real git checkout/commit path."""
    (tmp_path / "scoring").mkdir()
    live = tmp_path / "scoring" / "nested_prior.py"
    live.write_text("CHAMPION = 1\n")
    cand = tmp_path / "candidate.py"
    cand.write_text("CANDIDATE = 2\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "add", "scoring/nested_prior.py")
    _git(tmp_path, "commit", "-qm", "seed champion")
    return tmp_path, str(live), str(cand)


def test_swap_that_breaks_parity_is_reverted(repo):
    root, live, cand = repo
    res = promote(str(root), cand, "scoring/nested_prior.py",
                  ["scoring/nested_prior.py"], "should not land",
                  run_parity=lambda: False, do_commit=True)

    assert not res.committed and not res.parity_ok
    # the live artifact is byte-for-byte the committed champion again.
    assert open(live).read() == "CHAMPION = 1\n"
    # and nothing was committed: still exactly one commit in history.
    log = _git(root, "log", "--oneline").stdout.strip().splitlines()
    assert len(log) == 1


def test_swap_that_passes_parity_is_committed(repo):
    root, live, cand = repo
    res = promote(str(root), cand, "scoring/nested_prior.py",
                  ["scoring/nested_prior.py"], "promote candidate",
                  run_parity=lambda: True, do_commit=True)

    assert res.committed and res.parity_ok and res.sha
    # the live artifact now holds the candidate bytes.
    assert open(live).read() == "CANDIDATE = 2\n"
    # a second commit landed, carrying the swap message.
    log = _git(root, "log", "--oneline").stdout.strip().splitlines()
    assert len(log) == 2
    assert "promote candidate" in log[0]


def test_promote_refuses_to_swap_a_dirty_live_artifact(repo):
    root, live, cand = repo
    # dirty the live artifact with an uncommitted edit; promote must abort
    # rather than risk clobbering it on a revert.
    open(live, "w").write("CHAMPION = 1  # local edit\n")
    res = promote(str(root), cand, "scoring/nested_prior.py",
                  ["scoring/nested_prior.py"], "should abort",
                  run_parity=lambda: True, do_commit=True)

    assert not res.committed and not res.parity_ok
    assert open(live).read() == "CHAMPION = 1  # local edit\n"
