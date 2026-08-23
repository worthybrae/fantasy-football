"""pipeline.fit_archetypes: the K sweep, the decision, and the artifact.

What is under test here is a MEASUREMENT and a DECISION, not arithmetic. The EM
belongs to `scoring/mixture.py` and is tested in `tests/test_mixture.py`; the
folds, the clustered standard errors and the human-only replay belong to
`fit_prior`/`score_ladder` and are tested there. What is new is:

  * one mixture per fold yields BOTH the criterion K is chosen by and the top-1
    the ladder decides on, so the K chosen and the K scored cannot diverge,
  * ties and near-ties in the sweep go to the smaller K, which is the same rule
    the rest of the ladder runs on,
  * K=1 winning writes nothing, because K=1 IS the pooled fit and shipping it
    as a mixture would ship a rename,
  * the generated artifact is real Python whose own assertions hold.

The corpus itself is not touched: every fixture here is synthetic design
matrices, which is all `sweep` takes.
"""
import numpy as np
import pytest

from pipeline import fit_archetypes as fa
from scoring import mixture as mx
from scoring.draft_model import FEATURE_NAMES


def _corpus(n_drafts=4, seats_per_draft=4, picks_per_seat=6, split=True,
            seed=0):
    """A small labelled corpus in the shape `sweep` consumes.

    `split=True` plants two genuinely different drafters and alternates them
    within every draft, so a held-out draft contains seats of both classes --
    which is the only configuration where a mixture can possibly beat a pooled
    fit out of sample. `split=False` draws every seat from one drafter, which
    is what "there are no archetypes" looks like.
    """
    rng = np.random.default_rng(seed)
    beta_a = np.array([2.0, +3.0, 0.0])
    beta_b = np.array([2.0, -3.0, 0.0])
    X_list, chosen, seats, groups, buckets = [], [], [], [], []
    for d in range(n_drafts):
        for s in range(seats_per_draft):
            beta = beta_a if (s % 2 == 0 or not split) else beta_b
            for _ in range(picks_per_seat):
                X = rng.normal(size=(6, 3))
                scores = X @ beta
                probs = np.exp(scores - scores.max())
                X_list.append(X)
                chosen.append(int(rng.choice(6, p=probs / probs.sum())))
                seats.append(f"d{d}-s{s}")
                groups.append(f"draft-{d}")
                buckets.append("early")
    return X_list, chosen, seats, groups, buckets


# ------------------------------------------------------------ the feature set

def test_the_two_feature_sets_are_rung_3s_and_everything():
    """The mixture goes on top of whichever of the two won its OWN rung.

    Fitting it on a feature set that lost would confound the archetypes with
    the features in the opposite direction from the one the ladder is built to
    prevent -- a mixture rung that beat rung 4 because it quietly dropped six
    columns rung 4 had, or lost because it carried six rung 3 did not.
    """
    assert len(fa.feature_set("shipped23")) == 23
    assert fa.feature_set("all") == list(FEATURE_NAMES)
    assert set(fa.feature_set("shipped23")) < set(fa.feature_set("all"))
    with pytest.raises(ValueError, match="unknown feature set"):
        fa.feature_set("whatever")


# ----------------------------------------------------------------- the sweep

def test_every_k_is_scored_on_exactly_the_same_picks():
    """A sweep whose rows are measured on different samples is not a sweep.

    Every K here is scored by holding out each draft in turn and predicting its
    picks, so the denominators must be identical across K -- otherwise the
    criterion is partly "which K got the easier picks".
    """
    X_list, chosen, seats, groups, buckets = _corpus()
    reports = fa.sweep(X_list, chosen, seats, groups, buckets, ks=(1, 2),
                       lam=1.0, restarts=1, jobs=1)
    assert set(reports) == {1, 2}
    assert reports[1]["n"] == reports[2]["n"] == len(chosen)
    assert set(reports[1]["by_draft"]) == set(reports[2]["by_draft"])


def test_the_sweep_reports_both_criteria_from_one_fit_per_fold():
    """The held-out marginal likelihood chooses K and top-1 decides the rung,
    and both come off the same fold model. Fitting twice would make it possible
    for the K chosen and the K measured to be different objects."""
    X_list, chosen, seats, groups, buckets = _corpus()
    reports = fa.sweep(X_list, chosen, seats, groups, buckets, ks=(1,),
                       lam=1.0, restarts=1, jobs=1)
    report = reports[1]
    for key in ("heldout_ll", "heldout_per_pick", "top1", "top5", "logloss",
                "by_draft", "by_draft_n", "by_bucket", "pi_by_fold"):
        assert key in report, key
    assert report["heldout_per_pick"] == pytest.approx(
        report["heldout_ll"] / report["n"])
    # At K=1 the marginal likelihood IS the ordinary log-likelihood, because
    # there is nothing to integrate out. That identity is worth pinning: it is
    # the same K=1 control the EM's correctness check rests on.
    assert report["heldout_per_pick"] == pytest.approx(-report["logloss"])


def test_a_corpus_with_real_archetypes_prefers_more_than_one_class():
    """The sweep has to be ABLE to say K>1, or a K=1 result says nothing.

    Two planted drafters, alternating within every draft, so every held-out
    draft contains seats of both. If the criterion could not detect that, a K=1
    answer on the real corpus would be a fact about this module rather than
    about drafters.
    """
    X_list, chosen, seats, groups, buckets = _corpus(split=True, seed=3)
    reports = fa.sweep(X_list, chosen, seats, groups, buckets, ks=(1, 2),
                       lam=1.0, restarts=3, jobs=1)
    assert reports[2]["heldout_per_pick"] > reports[1]["heldout_per_pick"]
    assert fa.choose_k(reports) == 2


def test_a_corpus_with_one_kind_of_drafter_prefers_one_class():
    """The other direction, which is the outcome that ships nothing."""
    X_list, chosen, seats, groups, buckets = _corpus(n_drafts=6, split=False,
                                                     seed=4)
    reports = fa.sweep(X_list, chosen, seats, groups, buckets, ks=(1, 2, 3),
                       lam=10.0, restarts=2, jobs=1)
    # Not "K=2 scores worse" -- it scores within 1e-4 nats per pick either
    # way, which is the point. A second class on data from one drafter buys
    # nothing it can carry to a draft it has not seen, and the gate is that the
    # gain has to clear its own paired error.
    for k in (2, 3):
        gain = (reports[k]["heldout_per_pick"]
                - reports[1]["heldout_per_pick"])
        assert abs(gain) < 1e-3, (k, gain)
    assert fa.choose_k(reports) == 1


def test_a_gain_smaller_than_its_own_error_does_not_buy_a_class():
    """The tiebreak, at the resolution the criterion actually has.

    A K=3 that matches K=1 to four decimal places has bought three times the
    parameters and nothing else. Measured on synthetic data drawn from ONE
    drafter, K=2 still edges K=1 by about 4e-5 nats per pick -- a second class
    can always absorb a little of the folds' own noise -- so a bare `>` would
    report two kinds of drafter from a corpus that has one.
    """
    rng = np.random.default_rng(0)
    drafts = [f"d{i}" for i in range(20)]
    base = {d: -2.5 + rng.normal(scale=0.3) for d in drafts}

    def shifted(mean_gain, jitter):
        by_draft = {d: v + mean_gain + rng.normal(scale=jitter)
                    for d, v in base.items()}
        return {"heldout_per_pick": float(np.mean(list(by_draft.values()))),
                "heldout_by_draft": by_draft}

    one = {"heldout_per_pick": float(np.mean(list(base.values()))),
           "heldout_by_draft": base}
    # A mean gain of nothing, and per-draft differences of +/-0.05: noise.
    assert fa.choose_k({1: one, 2: shifted(0.0, 0.05)}) == 1
    # The same jitter with a mean gain ten times its own error: a separation.
    assert fa.choose_k({1: one, 2: shifted(0.5, 0.05)}) == 2


def test_running_the_folds_in_a_pool_gives_the_same_answer_as_serially():
    """`--jobs` is a wall-clock argument and must not be a modelling one.

    The folds are independent and each seeds its own EM from `seed`, so the
    pool can only change how long the sweep takes. A mismatch here would mean a
    fold was reading state the parent mutated, which on 43 folds and four K
    values would be a quietly different table every run.
    """
    X_list, chosen, seats, groups, buckets = _corpus(n_drafts=4, seed=5)
    serial = fa.sweep(X_list, chosen, seats, groups, buckets, ks=(1, 2),
                      lam=1.0, restarts=1, jobs=1)
    pooled = fa.sweep(X_list, chosen, seats, groups, buckets, ks=(1, 2),
                      lam=1.0, restarts=1, jobs=3)
    for k in (1, 2):
        assert serial[k]["top1"] == pytest.approx(pooled[k]["top1"])
        assert serial[k]["heldout_ll"] == pytest.approx(pooled[k]["heldout_ll"])


# -------------------------------------------------------------- the artifact

def test_the_generated_module_is_real_python_that_checks_itself():
    """The artifact carries its own assertions, and they have to pass.

    `BETAS` is K rows against one feature list and `WEIGHTS` is a distribution
    over classes; both are exactly the kind of thing that goes wrong silently
    in a generated file, which is why the generated file asserts them rather
    than the generator.
    """
    X_list, chosen, seats, _, _ = _corpus()
    model = mx.fit_mixture(X_list, chosen, seats, 2, lam=1.0, restarts=2)
    text = fa.render_module(model, ["reach", "fall", "need"], ["mock:a"],
                            {"k": 2}, "prose")
    namespace = {}
    exec(compile(text, "archetypes.py", "exec"), namespace)
    assert namespace["BETAS"].shape == (2, 3)
    assert namespace["WEIGHTS"].sum() == pytest.approx(1.0)
    assert namespace["POOLED"].shape == (3,)
    assert namespace["FEATURES"] == ["reach", "fall", "need"]
    assert namespace["DRAFT_IDS"] == ["mock:a"]


def test_the_artifact_names_every_class_by_its_share_of_seats():
    """A generated coefficient file is read, not executed, and a block of 29
    numbers with no heading is unreadable. Each class carries its weight in the
    comment above it for the same reason every other generated vector in this
    project carries its feature name beside each coefficient."""
    X_list, chosen, seats, _, _ = _corpus()
    model = mx.fit_mixture(X_list, chosen, seats, 2, lam=1.0, restarts=2)
    text = fa.render_module(model, ["reach", "fall", "need"], [], {}, "")
    assert "# ---- class 0," in text and "# ---- class 1," in text
    assert "of seats ----" in text
    assert "# reach" in text


def test_the_writer_is_dumb_and_the_decision_lives_in_main(tmp_path):
    """`write_archetypes` writes what it is given, with no rule in it.

    The rule is one expression in `main`, next to the numbers that make it, so
    that changing it is a diff somebody has to justify rather than a flag
    somebody can pass. A writer that could refuse would be a second place the
    decision lives.
    """
    X_list, chosen, seats, _, _ = _corpus()
    model = mx.fit_mixture(X_list, chosen, seats, 2, lam=1.0, restarts=1)
    target = tmp_path / "archetypes.py"
    written = fa.write_archetypes(model, ["reach", "fall", "need"], [], {}, "",
                                  path=target)
    assert written == target and target.exists()


def test_there_is_no_force_flag():
    """Same rule as `fit-prior` and `score-ladder`: a losing rung prints its
    numbers, writes nothing and exits 1. Shipping one anyway means editing the
    rule in `main`, which is a diff somebody has to justify."""
    import pathlib
    source = pathlib.Path(fa.__file__).read_text()
    assert "--force" not in source


def test_no_write_against_draft_log_anywhere_in_the_module():
    """A grep, as a test. A mock draft that is not written down when it happens
    is gone, and a farm loop is appending to this same file while the sweep
    runs -- for over an hour. This module reads it and must not be able to do
    anything else, even by accident."""
    import pathlib
    source = pathlib.Path(fa.__file__).read_text().upper()
    for statement in ("CREATE OR REPLACE", "CREATE TABLE", "DROP TABLE",
                      "DELETE FROM", "INSERT INTO", "UPDATE DRAFT_LOG",
                      "ALTER TABLE", "ENSURE_SCHEMA("):
        assert statement not in source, f"{statement} in fit_archetypes.py"
