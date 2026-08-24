"""Offline tests for the auto-refit safety gate and the swap/revert machinery.

No corpus, no fit: the gate is a pure function and the swap logic is exercised
against a throwaway git repo in tmp_path. These pin the safety story --
  * the gate does NOT swap on a tie or any regression, DOES swap on a strict win,
  * a swap whose post-swap parity check fails is REVERTED, live model untouched,
  * the min-new-drafts skip fires when too little new data has accrued --
without ever touching the real corpus or the real `scoring/nested_prior.py`.
"""
import subprocess

import numpy as np
import pytest

from pipeline import auto_refit as ar
from pipeline import measure_nested as mn
from pipeline.auto_refit import (
    Metrics,
    _aggregate,
    _route,
    candidate_heldout_columns,
    champion_heldout_columns,
    gate,
    promote,
    should_skip,
)


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


# ----------------------------------------- symmetric held-out scoring (the fix)
#
# The design flaw this replaced scored the CHAMPION full-width (in-sample) and
# the CANDIDATE held-out, so a genuinely better model could be blocked forever.
# The fix scores BOTH leave-one-draft-out, by the identical method. These pin
# that: neither column function runs a full-width path, and an identical
# candidate TIES (no swap) while a strictly-better held-out candidate SWAPS.


def test_champion_side_is_scored_held_out_not_full_width(monkeypatch):
    # The champion column is the leave-one-draft-out ladder
    # (`measure_nested._nested_perpick`, single process), NOT the frozen live
    # artifact scored full-width. Stub the ladder and assert the champion side
    # is exactly what it returns, called with workers=1.
    sentinel = (np.array([1, 0]), np.array([1, 1]),
                np.array([1, 1]), np.array([-0.1, -0.2]))
    seen = {}

    def fake_perpick(design, workers=1):
        seen["workers"] = workers
        return sentinel

    monkeypatch.setattr(mn, "_nested_perpick", fake_perpick)
    got = champion_heldout_columns(object())
    assert got is sentinel and seen["workers"] == 1


def test_candidate_and_champion_use_the_identical_held_out_columns(monkeypatch):
    # Same architecture on the same corpus => the candidate reuses the
    # champion's held-out columns rather than run a second, identical ladder.
    # Both sides therefore see the SAME held-out numbers -- the symmetry the
    # fix guarantees -- and (with no champion_cols) the candidate would run the
    # very same ladder itself.
    calls = []
    sentinel = (np.array([1, 0]), np.array([1, 1]),
                np.array([1, 1]), np.array([-0.1, -0.2]))

    def fake_perpick(design, workers=1):
        calls.append(workers)
        return sentinel

    monkeypatch.setattr(mn, "_nested_perpick", fake_perpick)

    champ_cols = champion_heldout_columns(object())
    # reuse path: champion's columns handed in, no second ladder run
    reused = candidate_heldout_columns(object(), champion_cols=champ_cols)
    assert reused is champ_cols
    assert len(calls) == 1                    # only the champion ran the ladder

    # standalone path (a future model would score itself here): same method
    standalone = candidate_heldout_columns(object())
    assert standalone is sentinel
    assert len(calls) == 2


def _routed_metrics(nested_cols):
    """Aggregate a hybrid's Metrics from mid/late nested columns, holding the
    flat-early routing constant (all picks mid/late), exactly as `main` does."""
    groups = np.array([0, 0, 1, 1])
    early = np.array([False, False, False, False])
    flat = np.zeros(4)
    h1, h5, lp = nested_cols
    return _aggregate(_route(early, flat, h1), _route(early, flat, h5),
                      _route(early, flat, lp), groups)


def test_identical_held_out_candidate_ties_and_does_not_swap():
    # A candidate scored by the SAME held-out method on the SAME picks produces
    # identical columns -> an exact top-1 tie -> the strict-greater gate holds.
    nested = (np.array([1, 0, 1, 0]), np.array([1, 1, 1, 1]),
              np.array([-0.5, -1.0, -0.5, -1.0]))
    champ = _routed_metrics(nested)
    cand = _routed_metrics(nested)
    assert champ.top1 == cand.top1            # symmetric held-out => exact tie
    do, _why = gate(champ, cand, flat_top1=0.0)
    assert not do


def test_strictly_better_held_out_candidate_swaps():
    # Held-out symmetry does NOT block a genuinely better model: one more top-1
    # hit and a no-worse log-loss clears the gate.
    champ = _routed_metrics((np.array([1, 0, 1, 0]), np.array([1, 1, 1, 1]),
                             np.array([-0.5, -1.0, -0.5, -1.0])))
    cand = _routed_metrics((np.array([1, 1, 1, 0]), np.array([1, 1, 1, 1]),
                            np.array([-0.5, -0.9, -0.5, -1.0])))
    assert cand.top1 > champ.top1 and cand.nll <= champ.nll
    do, _why = gate(champ, cand, flat_top1=0.0)
    assert do


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
