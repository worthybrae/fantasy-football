"""pipeline.score_ladder: the human-only evaluation harness and the ladder.

What is under test here is a FILTER and a DECISION, not arithmetic. `fit`,
`feature_matrix` and `_softmax` belong to `draft_model` and are tested there;
the folds, the clustered standard errors and the replay belong to `fit_prior`
and are tested in `tests/test_fit_prior.py`. What is new is:

  * `autodrafted IS FALSE` is the evaluation's rule and `NOT autodrafted IS
    TRUE` is the fit's, they disagree on NULL, and that disagreement is the
    entire point. Most of this file exists to make swapping them fail.
  * a pick the evaluation drops still HAPPENED -- it still takes its player
    off the board for the next pick that is measured.
  * rung 3 is the 23 features that were already shipping, pinned by name, so
    that Tier 1 appending columns to `FEATURE_NAMES` cannot silently turn the
    human-only rung into a human-plus-new-features rung.
  * a rung that loses writes nothing, and no argument changes that.

The corpus fixtures are `tests/test_fit_prior.py`'s, imported rather than
copied: both modules replay the same corpus through the same loop, and two
hand-built 4x3 fixtures that drifted apart would make the two files' numbers
incomparable for no reason. Every corpus is a throwaway file under `tmp_path`.
Nothing here opens, let alone writes, the real `data/draft_corpus.duckdb`.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import fit_prior as fp
from pipeline import score_ladder as sl
from scoring.draft_model import FEATURE_NAMES

from tests.test_fit_prior import (POOL_SIZE, ROUNDS, TEAMS, _alternating_picks,
                                  _empty_league, _picks, _write_corpus)

N_PICKS = TEAMS * ROUNDS


# ------------------------------------------------------- the two predicates

def test_only_a_recorded_false_counts_as_human():
    """The evaluation's rule, spelled out over all three states of the column.

    NULL is the one that matters. It is 58% of the real corpus, it is an
    unmeasured mixture of people and ESPN's engine, and reading it as "not
    flagged, therefore a person" is how a human-only metric quietly stops
    being one.
    """
    assert sl.is_human_pick(False) is True
    assert sl.is_human_pick(True) is False
    for null in (None, pd.NA, np.nan, float("nan")):
        assert sl.is_human_pick(null) is False, f"{null!r} is not a person"


def test_the_fit_rule_and_the_evaluation_rule_disagree_on_null():
    """The pair, asserted as a pair. This is the test that fails if somebody
    "tightens" one of them to match the other.

    They are not two spellings of one idea. The fit asks "is there reason to
    believe a machine made this pick" and NULL is not such a reason; the
    evaluation asks "do I KNOW a person made this pick" and NULL is not
    knowledge. Both are right for their own question and each is wrong for the
    other's.
    """
    for null in (None, pd.NA, np.nan):
        assert fp.fit_keeps_pick(null) is True, "the fit keeps unknown picks"
        assert sl.is_human_pick(null) is False, "the evaluation does not"

    # And they agree everywhere else, which is what makes NULL the whole
    # difference between them rather than one of several.
    for known in (True, False):
        assert fp.fit_keeps_pick(known) is sl.is_human_pick(known)


# ---------------------------------------------- which picks are measured on

def _mixed_corpus(tmp_path):
    """One draft: pick 1 autodrafted, pick 6 a known person, the rest NULL.

    Every state of the column appears, and the single human pick sits far
    enough into the draft that five picks nobody measures have already come
    off the board before it.
    """
    flags = [None] * N_PICKS
    flags[0] = True
    flags[5] = False
    picks = _picks("mock:mixed", _alternating_picks(N_PICKS), autodrafted=flags)
    return _write_corpus(tmp_path / "corpus.duckdb", [("mock:mixed", picks)])


def _observations(corpus_path, league_path, keep_pick=None, drop_label=None):
    corpus, league_conn = fp.open_corpus(corpus_path), fp.open_league(league_path)
    try:
        ids = fp.snapshot_draft_ids(corpus)
        if keep_pick is None:
            return fp.build_corpus_observations(corpus, league_conn, ids)
        return fp.build_corpus_observations(
            corpus, league_conn, ids, keep_pick=keep_pick,
            drop_label=drop_label or sl.NOT_HUMAN)
    finally:
        corpus.close()
        league_conn.close()


def test_a_null_pick_is_excluded_from_evaluation_and_kept_by_the_fit(tmp_path):
    """The same twelve picks, two rules, two different answers -- both correct
    for their own question.

    The fit drops one pick (the recorded autodraft) and fits the other eleven,
    which is the standing ruling `fit_prior`'s docstring records. The
    evaluation keeps ONE, the only pick a person is known to have made. If
    these two numbers were ever the same, one of the rules would have stopped
    doing its job.
    """
    corpus_path = _mixed_corpus(tmp_path)
    league_path = _empty_league(tmp_path)

    fitted = _observations(corpus_path, league_path)
    human = _observations(corpus_path, league_path, keep_pick=sl.is_human_pick)

    assert len(fitted.observations) == N_PICKS - 1 == 11
    assert fitted.dropped["autodrafted"] == 1

    assert len(human.observations) == 1
    assert human.dropped[sl.NOT_HUMAN] == N_PICKS - 1 == 11
    assert human.picks_seen == N_PICKS


def test_swapping_the_two_predicates_changes_the_measurement(tmp_path):
    """The guard on the pair, at the call site rather than on the functions.

    `human_observations` passes `is_human_pick`. Handing the same replay the
    FIT's predicate instead -- the exact mistake this module is written
    against -- must produce a different set of observations. If it ever does
    not, the two rules have converged and the human-only metric is measuring
    an unlabelled population while still calling itself human-only.
    """
    corpus_path = _mixed_corpus(tmp_path)
    league_path = _empty_league(tmp_path)
    corpus, league_conn = fp.open_corpus(corpus_path), fp.open_league(league_path)
    try:
        ids = fp.snapshot_draft_ids(corpus)
        human = sl.human_observations(corpus, league_conn, ids)
        swapped = fp.build_corpus_observations(
            corpus, league_conn, ids, keep_pick=fp.fit_keeps_pick,
            drop_label=sl.NOT_HUMAN)
    finally:
        corpus.close()
        league_conn.close()

    assert len(human.observations) == 1
    assert len(swapped.observations) == 11
    assert len(swapped.observations) != len(human.observations)


def test_an_unmeasured_pick_still_takes_its_player_off_the_board(tmp_path):
    """Excluding a pick from the METRIC is not pretending it did not happen.

    Five picks precede the one human pick in this fixture and not one of them
    is measured, but all five removed a player. The human pick's choice set
    has to be the board those five left behind -- otherwise every feature that
    reads the pool is computed against a draft that never took place, and the
    number at the bottom of the ladder is a number about a fiction.
    """
    corpus_path = _mixed_corpus(tmp_path)
    human = _observations(corpus_path, _empty_league(tmp_path),
                          keep_pick=sl.is_human_pick)

    only = human.observations[0]
    assert only.overall_pick == 6
    assert len(only.pool) == POOL_SIZE - 5 == 19
    assert 0 <= only.chosen < len(only.pool)


def test_only_drafts_with_a_known_human_pick_are_named(tmp_path):
    """A draft with no `autodrafted IS FALSE` pick anywhere contributes
    nothing to a human-only measurement, so it does not belong in the snapshot
    the run reports. "37 of 60 drafts" is the honest description of the real
    corpus and "60" is not."""
    labelled = _picks("mock:labelled", _alternating_picks(N_PICKS),
                      autodrafted=[False] * N_PICKS)
    blank = _picks("mock:blank", _alternating_picks(N_PICKS))
    corpus_path = _write_corpus(tmp_path / "corpus.duckdb",
                                [("mock:labelled", labelled),
                                 ("mock:blank", blank)])
    corpus = fp.open_corpus(corpus_path)
    try:
        snapshot = fp.snapshot_draft_ids(corpus)
        assert sorted(snapshot) == ["mock:blank", "mock:labelled"]
        assert sl.labelled_draft_ids(corpus, snapshot) == ["mock:labelled"]
        assert sl.labelled_draft_ids(corpus, []) == []
    finally:
        corpus.close()


# ------------------------------------------------------------- the rungs

@pytest.fixture
def human_corpus(tmp_path):
    """Three drafts of entirely known-human picks, and an empty league."""
    drafts = []
    for d in range(3):
        draft_id = f"mock:human{d}"
        drafts.append((draft_id, _picks(draft_id, _alternating_picks(N_PICKS),
                                        autodrafted=[False] * N_PICKS)))
    return (_write_corpus(tmp_path / "corpus.duckdb", drafts),
            _empty_league(tmp_path))


def _designed(human_corpus):
    corpus_path, league_path = human_corpus
    corpus, league_conn = fp.open_corpus(corpus_path), fp.open_league(league_path)
    try:
        obs = sl.human_observations(corpus, league_conn,
                                    sl.labelled_draft_ids(
                                        corpus, fp.snapshot_draft_ids(corpus)))
    finally:
        corpus.close()
        league_conn.close()
    return fp.design(obs)


def test_rung_3_is_the_23_that_were_shipping_not_whatever_feature_names_holds(
        human_corpus, monkeypatch):
    """The decomposition the ladder exists for, defended at the fit itself.

    Tier 1 appends columns to `FEATURE_NAMES`. If rung 3 fitted "everything
    `feature_matrix` produces", the day those columns landed the human-only
    rung would silently become a human-plus-new-features rung, the two effects
    the ladder separates would be confounded again, and nothing would error.
    So the columns are counted where they are actually handed to `fit`.
    """
    X_list, chosen, groups, boards, buckets = _designed(human_corpus)
    assert len(FEATURE_NAMES) >= 23
    assert X_list[0].shape[1] == len(FEATURE_NAMES)

    widths = []
    real_fit = fp.fit
    monkeypatch.setattr(fp, "fit", lambda X, c, **kw: (
        widths.append(X[0].shape[1]), real_fit(X, c, **kw))[1])

    sl.ladder(X_list, chosen, groups, boards, buckets=buckets)

    assert widths, "rung 3 never fitted anything"
    assert set(widths) == {23}, f"rung 3 fitted {set(widths)} columns, not 23"


def test_every_rung_reports_all_three_metrics_on_the_same_picks(human_corpus):
    """One reported metric is a metric that was chosen after the fact. All
    three, for every rung, including the baseline -- a baseline row with a
    hole in it invites reading it as if it only had a top-1."""
    X_list, chosen, groups, boards, buckets = _designed(human_corpus)

    rungs = sl.ladder(X_list, chosen, groups, boards, buckets=buckets)

    assert [r.label for r in rungs] == ["ADP baseline", "shipped prior",
                                        "human-only refit"]
    for rung in rungs:
        for metric in ("top1", "top5", "logloss", "n", "by_draft", "by_draft_n"):
            assert metric in rung.report, f"{rung.label} has no {metric}"
        assert rung.report["n"] == len(chosen) == 3 * N_PICKS == 36
        assert sorted(rung.report["by_draft"]) == ["mock:human0", "mock:human1",
                                                   "mock:human2"]
        assert set(rung.report["by_draft_n"].values()) == {N_PICKS}
    # The market baseline's top-5 is the plain thing it says: the chosen
    # player was among the five cheapest available. This fixture always takes
    # the best or second-best on the board, so every pick is inside the top 5.
    assert rungs[0].report["top5"] == 1.0
    assert rungs[0].report["top1"] == pytest.approx(0.5)


def test_a_restricted_top1_weights_each_draft_by_its_own_pick_count():
    """The in-sample check re-aggregates over a SUBSET of drafts, and a plain
    mean of per-draft rates would let a draft that contributed ten picks count
    as much as one that contributed a hundred."""
    report = {"by_draft": {"a": 0.5, "b": 0.1, "c": 0.9},
              "by_draft_n": {"a": 100, "b": 10, "c": 50}}

    top1, n = sl.restricted_top1(report, ["a", "b"])

    assert n == 110
    assert top1 == pytest.approx((0.5 * 100 + 0.1 * 10) / 110)
    # A draft the report never scored contributes nothing rather than a NaN,
    # and asking for none of them is reported as none rather than a crash.
    assert sl.restricted_top1(report, ["a", "zz"])[1] == 100
    assert sl.restricted_top1(report, [])[1] == 0
    assert np.isnan(sl.restricted_top1(report, [])[0])


# ------------------------------------------------------------ the decision

def _anti_market_prior():
    """A shipped prior that predicts the exact opposite of the board.

    `reach` is `max(0, log1p(market_rank) - log1p(pick))`, so a large POSITIVE
    coefficient on it ranks the deepest available player first. Every pick in
    the fixture comes from the top two of the board, so this scores top-1
    zero -- which makes "does the human-only refit beat it" a question about
    the refit rather than about how the real shipped prior happens to do on a
    synthetic fixture.
    """
    beta = np.zeros(len(FEATURE_NAMES))
    beta[FEATURE_NAMES.index("reach")] = 50.0
    return beta


def _run(human_corpus, monkeypatch, tmp_path, shipped=None, extra=()):
    corpus_path, league_path = human_corpus
    target = tmp_path / "generated_human_prior.py"
    monkeypatch.setattr(sl, "HUMAN_PRIOR_MODULE", target)
    if shipped is not None:
        monkeypatch.setattr(sl, "COLD_START_PRIOR", shipped)
    code = sl.main(["--corpus", corpus_path, "--league-db", league_path,
                    *extra])
    return code, target


def test_a_winning_rung_writes_the_artifact(human_corpus, monkeypatch,
                                            tmp_path):
    code, target = _run(human_corpus, monkeypatch, tmp_path,
                        shipped=_anti_market_prior())

    assert code == 0
    assert target.exists()

    namespace = {}
    exec(compile(target.read_text(), str(target), "exec"), namespace)
    assert list(namespace["FEATURES"]) == FEATURE_NAMES
    assert len(namespace["PRIOR"]) == len(FEATURE_NAMES)
    assert np.isfinite(namespace["PRIOR"]).all()
    assert namespace["DRAFT_IDS"] == ["mock:human0", "mock:human1",
                                      "mock:human2"]
    assert namespace["PROVENANCE"]["folds"] == "leave-one-draft-out"
    assert namespace["PROVENANCE"]["picks_scored"] == 36
    assert namespace["PROVENANCE"]["population"].startswith(
        "autodrafted IS FALSE")
    assert namespace["PROVENANCE"]["delta_top1"] > 0


def test_a_column_outside_rung_3_is_zero_and_says_it_was_never_fitted(
        human_corpus, monkeypatch, tmp_path):
    """A 0.0 that was never in the fit and a 0.0 that came out of one are
    different facts, and only the list of fitted columns can tell them apart.
    The written vector is full length so that `len(PRIOR) == len(FEATURES)`
    holds and nothing downstream has to know which columns rung 3 had."""
    _, target = _run(human_corpus, monkeypatch, tmp_path,
                     shipped=_anti_market_prior())
    text = target.read_text()
    namespace = {}
    exec(compile(text, str(target), "exec"), namespace)

    outside = [f for f in FEATURE_NAMES if f not in set(sl.SHIPPED_23_FEATURES)]
    for name in outside:
        assert namespace["PRIOR"][FEATURE_NAMES.index(name)] == 0.0
        assert any(line.endswith("<- not in rung 3, not fitted")
                   and f"# {name}" in line
                   for line in text.splitlines()), f"{name} is unannotated"
    for name in sl.SHIPPED_23_FEATURES:
        assert "not fitted" not in [
            l for l in text.splitlines() if f"# {name} " in l or
            l.rstrip().endswith(f"# {name}")][0]


def test_a_losing_rung_writes_nothing_at_all(human_corpus, monkeypatch,
                                             tmp_path):
    """The whole rule in one assertion. The seam is the scorer rather than the
    data because the thing under test is the DECISION: constructing a corpus
    on which the refit genuinely loses would test the fixture, while forcing
    the measurement to report a loss tests what `main` does with one."""
    losing = {"top1": 0.0, "top5": 0.0, "logloss": 9.9, "n": 1,
              "by_draft": {}, "by_draft_n": {}, "by_bucket": {}}
    monkeypatch.setattr(fp, "leave_one_draft_out", lambda *a, **k: dict(losing))

    code, target = _run(human_corpus, monkeypatch, tmp_path)

    assert code == 1
    assert not target.exists(), "a losing rung wrote a prior"


def test_there_is_no_force_flag(human_corpus, monkeypatch, tmp_path):
    """Shipping a losing rung has to be a reviewable diff, not an argument.
    `main` reads exactly two options and neither of them is an override."""
    losing = {"top1": 0.0, "top5": 0.0, "logloss": 9.9, "n": 1,
              "by_draft": {}, "by_draft_n": {}, "by_bucket": {}}
    monkeypatch.setattr(fp, "leave_one_draft_out", lambda *a, **k: dict(losing))

    for flag in ("--force", "--yes", "--write", "--ship"):
        code, target = _run(human_corpus, monkeypatch, tmp_path, extra=[flag])
        assert code == 1
        assert not target.exists()


def test_a_corpus_with_no_labelled_pick_measures_nothing(tmp_path):
    """The pre-labelling state of this corpus, and the one answer that is not
    allowed: scoring the unlabelled picks as if they were human."""
    picks = _picks("mock:blank", _alternating_picks(N_PICKS))
    corpus_path = _write_corpus(tmp_path / "corpus.duckdb", [("mock:blank", picks)])

    assert sl.main(["--corpus", corpus_path,
                    "--league-db", _empty_league(tmp_path)]) == 1


# ----------------------------------------------------------- read-only-ness

def test_the_corpus_is_opened_read_only(human_corpus):
    corpus_path, _ = human_corpus
    conn = fp.open_corpus(corpus_path)
    try:
        with pytest.raises(Exception):
            conn.execute("DELETE FROM draft_log_pick")
    finally:
        conn.close()


def test_no_write_against_draft_log_anywhere_in_the_module():
    """A grep, as a test. A mock draft that is not written down when it
    happens is gone, and a farm loop is appending to this same file while the
    ladder runs. This module reads it and must not be able to do anything
    else, even by accident."""
    import pathlib
    source = pathlib.Path(sl.__file__).read_text().upper()
    # `ENSURE_SCHEMA(` with the paren, because the docstring names the
    # function to explain why it is never called and a bare substring would
    # make documenting the hazard indistinguishable from committing it.
    for statement in ("CREATE OR REPLACE", "CREATE TABLE", "DROP TABLE",
                      "DELETE FROM", "INSERT INTO", "UPDATE DRAFT_LOG",
                      "ALTER TABLE", "ENSURE_SCHEMA("):
        assert statement not in source, f"{statement} in pipeline/score_ladder.py"
