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
from scoring import draft_model as dm
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

    Tier 1 appended columns to `FEATURE_NAMES`. If rung 3 fitted "everything
    `feature_matrix` produces", the day those columns landed the human-only
    rung would silently become a human-plus-new-features rung, the two effects
    the ladder separates would be confounded again, and nothing would error.
    So the columns are counted where they are actually handed to `fit`, and
    rung 4's width is counted the same way in the same call: rung 3 runs
    first, so the first fold's width is rung 3's and the last fold's is
    rung 4's.
    """
    X_list, chosen, groups, boards, buckets = _designed(human_corpus)
    assert len(FEATURE_NAMES) >= 29
    assert X_list[0].shape[1] == len(FEATURE_NAMES)

    widths = []
    real_fit = fp.fit
    monkeypatch.setattr(fp, "fit", lambda X, c, **kw: (
        widths.append(X[0].shape[1]), real_fit(X, c, **kw))[1])

    sl.ladder(X_list, chosen, groups, boards, buckets=buckets)

    assert widths, "no rung fitted anything"
    assert set(widths) == {23, 29}, (
        f"the fitted rungs used {sorted(set(widths))} columns, not 23 then 29")
    assert widths[0] == 23, "rung 3 did not fit 23 columns"
    assert widths[-1] == 29, "rung 4 did not fit 29 columns"


def test_rung_4_is_rung_3_plus_the_six_and_names_them_from_one_place():
    """Rung 4 is a SUM, and both halves of it are pinned.

    The whole rung means "rung 3, and additionally these six". Two ways that
    stops being true without anything failing: `FEATURE_NAMES` grows a
    column and rung 4 was spelled as "all of it", or the six get re-listed
    here and drift from the definition in `draft_model`. Both are asserted
    against rather than commented about.
    """
    assert sl.TIER1_FEATURES == list(dm._POOL_SIGNAL_FEATURES)
    assert sl.RUNG_4_FEATURES == sl.SHIPPED_23_FEATURES + sl.TIER1_FEATURES
    assert len(sl.RUNG_4_FEATURES) == 29
    assert not (set(sl.TIER1_FEATURES) & set(sl.SHIPPED_23_FEATURES))
    # And the six are the ones the design document names, spelling included:
    # `dropoff_at_pos` is the spec's `pos_dropoff`, renamed off the `pos_`
    # prefix that `SUMMARY_FEATURES` reserves for the position dummies.
    assert set(sl.TIER1_FEATURES) == {
        "dropoff_at_pos", "vor", "durability", "proj_change", "last_of_tier",
        "slots_left_at_pos"}


def test_every_rung_reports_all_three_metrics_on_the_same_picks(human_corpus):
    """One reported metric is a metric that was chosen after the fact. All
    three, for every rung, including the baseline -- a baseline row with a
    hole in it invites reading it as if it only had a top-1."""
    X_list, chosen, groups, boards, buckets = _designed(human_corpus)

    rungs = sl.ladder(X_list, chosen, groups, boards, buckets=buckets)

    assert [r.label for r in rungs] == ["ADP baseline", "shipped prior",
                                        "human-only refit", "+ Tier 1"]
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


def test_the_in_sample_check_covers_every_fitted_rung(monkeypatch, capsys):
    """The shipped prior was fitted on some of the drafts it is scored on
    here, so part of its row is not held out for it. That flatters it and
    makes every delta above it conservative -- which is worth stating for
    rung 4 as well as rung 3, since the alternative is a reader assuming the
    caveat applies to whichever row the check happened to name."""
    def rung(label, top1):
        return sl.Rung(label, {"top1": top1,
                               "by_draft": {"a": top1, "b": top1, "c": top1},
                               "by_draft_n": {"a": 10, "b": 10, "c": 10}},
                       "leave-one-draft-out")

    monkeypatch.setattr(sl.mock_prior, "DRAFT_IDS", ["a"], raising=False)
    sl._print_contamination(
        [rung("ADP baseline", 0.10), rung("shipped prior", 0.20),
         rung("human-only refit", 0.24), rung("+ Tier 1", 0.25)],
        ["a", "b", "c"])

    out = capsys.readouterr().out
    assert "1 of these 3 drafts" in out
    # 20 human picks over the two drafts the shipped prior never saw, and
    # both refitted rungs re-stated over exactly those.
    assert "(20 human picks)" in out
    assert "human-only refit" in out and "+ Tier 1" in out
    assert "+0.0400" in out and "+0.0500" in out


# ------------------------------------------- rung 4's per-feature ablation

def test_the_ablation_cannot_be_called_without_saying_what_to_measure():
    """The near-miss this argument exists to make impossible.

    There are two ablations in this codebase and they default to two
    different sets: `fit_prior.ablate` to `UNMEASURED_FEATURES` and
    `draft_model.ablation` to `_NEW_FEATURES`, which is four columns from a
    2026-08-11 measurement and NONE of the six this rung is about. Reaching
    for the wrong one produces a full table of plausible deltas that tested
    nothing, exits 0, and reads exactly like a result. So this one has no
    default at all: leaving the argument off is a TypeError before any fit
    runs.
    """
    with pytest.raises(TypeError):
        sl.ablation([], [], [], sl.RUNG_4_FEATURES)      # no candidates


def test_the_candidates_the_run_uses_contain_every_tier_1_feature():
    """The other half of the same guard, at the value `main` actually passes.

    A required argument stops the six being forgotten; it does not stop them
    being left out of the list that gets passed. `ABLATION_CANDIDATES` is
    where that would happen, and this is the assertion that a column appended
    to `_POOL_SIGNAL_FEATURES` and not to `UNMEASURED_FEATURES` fails on.
    """
    assert set(sl.TIER1_FEATURES) <= set(sl.ABLATION_CANDIDATES)
    assert set(sl.ABLATION_CANDIDATES) <= set(sl.RUNG_4_FEATURES)


def test_ablating_a_column_the_model_never_had_is_refused():
    """A candidate outside the model would "drop" from a fit that never had
    it. The refit would be identical, the delta would come out at exactly
    0.0, and 0.0 is indistinguishable from a feature that was measured and
    found worthless -- so it is an error rather than a row."""
    with pytest.raises(ValueError, match="not in the model"):
        sl.ablation([], [], [], ["reach", "fall"], ["reach", "vor"])


def test_every_ablation_row_carries_the_error_of_its_own_delta(human_corpus):
    """The reason this is not `fit_prior.ablate`, asserted rather than argued.

    Rung 3 beat the shipped prior by +0.0167 against a paired error of
    0.0080. An individual feature is expected to be worth a fraction of that,
    so a per-feature delta printed without its error beside it invites being
    read at a precision it does not have. Every row therefore carries one,
    and it is the PAIRED error -- both sides scored the same picks in the
    same rooms.
    """
    X_list, chosen, groups, boards, buckets = _designed(human_corpus)

    table = sl.ablation(X_list, chosen, groups, sl.RUNG_4_FEATURES,
                        ["dropoff_at_pos", "slots_left_at_pos"])

    assert list(table["dropped"]) == ["none", "dropoff_at_pos",
                                      "slots_left_at_pos"]
    for column in ("top1", "top5", "logloss", "delta_top1", "delta_top1_se",
                   "delta_top5"):
        assert column in table.columns, f"the ablation reports no {column}"
    # The full-model row is the reference and is a delta of nothing from
    # itself, with no error, rather than a blank that reads as a missing
    # measurement.
    head = table.iloc[0]
    assert head["delta_top1"] == 0.0 and head["delta_top1_se"] == 0.0
    # And every delta is genuinely the full model minus that row's own refit,
    # not a relabelling: a dropped column has no coefficient at all.
    for row in table.itertuples():
        if row.dropped != "none":
            assert row.delta_top1 == pytest.approx(head["top1"] - row.top1)
            assert np.isfinite(row.delta_top1_se)


def test_only_the_six_can_change_what_ships(human_corpus):
    """Rung 4 selects over Tier 1 and reports over everything else.

    The ablation covers all fourteen `ABLATION_CANDIDATES` because the eight
    older ones have never been measured on humans and the rows are free. They
    are not candidates for CUTTING: rung 3 is pinned to all 23 by name, so
    dropping one of them inside this rung would fold "an older column stopped
    paying" into a rung whose headline is "the pool signals paid" -- two
    effects in one number, which is the confound the ladder exists to take
    apart.
    """
    table = pd.DataFrame([
        {"dropped": "none", "delta_top1": 0.0},
        {"dropped": "dropoff_at_pos", "delta_top1": +0.01},
        {"dropped": "vor", "delta_top1": -0.01},
        {"dropped": "last_of_tier", "delta_top1": 0.0},
        # An older column with a strong positive delta. It is rung 3's, it
        # is already shipping, and rung 4 must not "ship" it a second time
        # or read anything into it.
        {"dropped": "usage", "delta_top1": +0.05},
    ])

    assert sl.tier1_winners(table) == ["dropoff_at_pos"]
    # Exactly zero is CUT, not kept. A feature that bought nothing measurable
    # ships as a 0.0 meaning measured-and-rejected rather than as a
    # coefficient fitted on whatever the fold noise happened to be.
    assert "last_of_tier" not in sl.tier1_winners(table)


def test_coverage_counts_variation_inside_the_choice_set(human_corpus):
    """A conditional logit sees a column only through its variation inside one
    choice set: a column constant across the candidates divides straight back
    out of the softmax. So "how often does this vary" is the ceiling on how
    often the feature could have mattered at all, and it is reported next to
    the delta so a near-zero delta on a near-constant column reads as "it
    never had a chance" rather than as "it was measured and is worthless"."""
    X_list, chosen, groups, boards, buckets = _designed(human_corpus)

    table = sl.column_coverage(X_list, sl.TIER1_FEATURES)

    assert list(table["feature"]) == sl.TIER1_FEATURES
    assert (table["sets_varying"] >= 0).all() and (table["sets_varying"] <= 1).all()
    assert (table["nonzero_share"] >= 0).all()
    # `dropoff_at_pos` is pure pool state -- his projection minus the next
    # best available at his position -- so it varies in this fixture even
    # though the empty league database supplies no board columns at all.
    row = table.set_index("feature").loc["dropoff_at_pos"]
    assert row["sets_varying"] > 0.0
    # And `vor` is one of the four the corpus does not store. With no board
    # to join from, every entry is the neutral 0.0 -- which is exactly the
    # state this table exists to make visible rather than let a delta of
    # 0.0000 be read as a measurement.
    assert table.set_index("feature").loc["vor", "sets_varying"] == 0.0


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


def _by_feature_scorer(weights, base: float = 0.10):
    """A `leave_one_draft_out` whose top-1 is a stated function of the columns.

    The rung-4 decision is a chain of comparisons between fits that differ
    only in which columns they had, and on a 36-pick fixture whether six
    genuinely collinear signals help is a coin flip. Driving the SCORER makes
    the test about what `main` does with a set of numbers rather than about
    whether a synthetic corpus happens to produce them -- the same reasoning
    `test_a_losing_rung_writes_nothing_at_all` gives for stubbing the scorer
    rather than building a corpus that loses.

    `keep_idx` indexes `FEATURE_NAMES`, so the stub can name the columns it
    was handed and price them. Every fold of a given column set gets the same
    score, which makes the paired standard errors exactly 0.0 -- fine here,
    because the decision is on the delta and the errors are printed rather
    than decided on.
    """
    def stub(X_list, chosen, groups, boards=None, keep_idx=None, buckets=None):
        names = (FEATURE_NAMES if keep_idx is None
                 else [FEATURE_NAMES[i] for i in keep_idx])
        top1 = base + sum(weights.get(name, 0.0) for name in names)
        drafts = sorted(set(groups))
        n = len(chosen)
        return {"top1": top1, "top5": min(1.0, top1 + 0.4), "logloss": 3.0,
                "n": n, "by_draft": {g: top1 for g in drafts},
                "by_draft_n": {g: n // max(len(drafts), 1) for g in drafts},
                "by_bucket": {"early": [1, 2, 4], "mid": [1, 2, 4],
                              "late": [1, 2, 4]}}
    return stub


# One Tier 1 feature that pays, one that actively hurts, four that buy
# nothing. That is the shape the design document predicts ("expect most of
# these to lose") and it is the shape that exercises every branch of the
# selection: a keep, a negative cut, and a cut at exactly zero.
_ONE_WINNER = {"dropoff_at_pos": 0.010, "vor": -0.005}


def _run(human_corpus, monkeypatch, tmp_path, shipped=None, extra=(),
         weights=None):
    corpus_path, league_path = human_corpus
    target = tmp_path / "generated_human_prior.py"
    monkeypatch.setattr(sl, "HUMAN_PRIOR_MODULE", target)
    if shipped is not None:
        monkeypatch.setattr(sl, "COLD_START_PRIOR", shipped)
    if weights is not None:
        monkeypatch.setattr(fp, "leave_one_draft_out",
                            _by_feature_scorer(weights))
    code = sl.main(["--corpus", corpus_path, "--league-db", league_path,
                    *extra])
    return code, target


def test_a_winning_rung_writes_the_artifact(human_corpus, monkeypatch,
                                            tmp_path):
    code, target = _run(human_corpus, monkeypatch, tmp_path,
                        shipped=_anti_market_prior(), weights=_ONE_WINNER)

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
    # The selection, recorded in the artifact rather than only in a findings
    # document: a coefficient is read HERE, so which of the six earned one and
    # which was rejected has to travel with it.
    assert namespace["PROVENANCE"]["tier1_kept"] == ["dropoff_at_pos"]
    assert namespace["PROVENANCE"]["tier1_cut"] == [
        "vor", "durability", "proj_change", "last_of_tier",
        "slots_left_at_pos"]
    assert set(namespace["PROVENANCE"]["tier1_delta_top1"]) == set(
        sl.TIER1_FEATURES)
    # And the join those four board columns arrived by, for the same reason.
    assert "attributes_as_of" in namespace["PROVENANCE"]["board_joined_from"]


def test_a_cut_feature_and_an_unmeasured_one_do_not_read_the_same(
        human_corpus, monkeypatch, tmp_path):
    """Three things a 0.0 can mean, and only a list can tell them apart.

    A column the fit had and fitted to nearly nothing, a column offered to
    the ablation and rejected on `delta_top1`, and a column nothing asked
    about are three different claims that share one value. Rung 4 produces
    the first two, so the file has to distinguish them -- "measured and
    rejected" is a result and "never measured" is an absence of one, and
    reading the second as the first is how a feature quietly stops being
    re-measured.
    """
    _, target = _run(human_corpus, monkeypatch, tmp_path,
                     shipped=_anti_market_prior(), weights=_ONE_WINNER)
    text = target.read_text()
    namespace = {}
    exec(compile(text, str(target), "exec"), namespace)

    def line_for(name):
        return [l for l in text.splitlines()
                if f"# {name} " in l or l.rstrip().endswith(f"# {name}")][0]

    cut = [f for f in sl.TIER1_FEATURES if f != "dropoff_at_pos"]
    for name in cut:
        assert namespace["PRIOR"][FEATURE_NAMES.index(name)] == 0.0
        assert line_for(name).endswith("<- measured on humans, cut on "
                                       "delta_top1"), f"{name} is unannotated"
    for name in sl.SHIPPED_23_FEATURES + ["dropoff_at_pos"]:
        assert "<-" not in line_for(name), f"{name} is annotated as not fitted"


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


def test_the_per_feature_table_is_printed_even_when_the_rung_loses(
        human_corpus, monkeypatch, tmp_path, capsys):
    """A losing rung is where the ablation is worth the most.

    "The six lost" is barely a finding. "The six lost, five of them move
    nothing and the sixth is worth +0.004" says which one to keep working on
    and which five to stop reaching for. Running the diagnosis only on the
    winning branch would leave the outcome the design document tells us to
    expect reporting the least, so the table comes before the rule.
    """
    losing = {name: -0.01 for name in sl.TIER1_FEATURES}

    code, target = _run(human_corpus, monkeypatch, tmp_path,
                        shipped=_anti_market_prior(), weights=losing)
    out = capsys.readouterr().out

    assert code == 1 and not target.exists()
    for name in sl.TIER1_FEATURES:
        row = [l for l in out.splitlines() if l.strip().startswith(name)]
        assert row, f"{name} has no ablation row on the losing branch"
        # The delta AND the error of the delta, on the same line. A delta
        # printed alone is read at a precision it does not have.
        assert "delta_top1" in row[0] and "+/-" in row[0], row[0]
    assert "sets_varying" in out, "no coverage table on the losing branch"


def test_a_losing_rung_4_leaves_the_rung_3_that_already_won_alone(
        human_corpus, monkeypatch, tmp_path):
    """The chain, and the fallback it deliberately does NOT have.

    Rung 4 losing to rung 3 is a real expected outcome -- the design document
    says to expect most of the six to lose -- and the tempting response is to
    write rung 3 anyway, since rung 3 did win. That would silently replace a
    recorded measurement with a re-run of it on a corpus that has grown since,
    under the heading of a Tier 1 run that lost. So a losing rung 4 writes
    nothing at all and the file keeps whatever last won.
    """
    losing = {name: -0.01 for name in sl.TIER1_FEATURES}

    code, target = _run(human_corpus, monkeypatch, tmp_path,
                        shipped=_anti_market_prior(), weights=losing)

    assert code == 1
    assert not target.exists(), "a losing rung 4 rewrote the artifact"


def test_a_subset_that_does_not_beat_rung_3_is_not_written_either(
        human_corpus, monkeypatch, tmp_path):
    """The file carries the SUBSET, so the subset is what has to have won.

    The full six can clear rung 3 on the strength of a feature whose own
    ablation delta is at or below zero -- the deltas are not additive, and a
    column can help in company and not alone. Shipping on the full set's
    number would then put a figure in that file measured against a set of
    columns the file does not have.
    """
    # Every candidate's individual delta is zero (dropping any one changes
    # nothing), so nothing is kept -- while the six together are worth a
    # point, which clears the rung-4 gate.
    def stub(X_list, chosen, groups, boards=None, keep_idx=None, buckets=None):
        names = (FEATURE_NAMES if keep_idx is None
                 else [FEATURE_NAMES[i] for i in keep_idx])
        tier1 = set(names) & set(sl.TIER1_FEATURES)
        top1 = 0.10 + (0.01 if len(tier1) >= len(sl.TIER1_FEATURES) - 1 else 0.0)
        drafts = sorted(set(groups))
        return {"top1": top1, "top5": 0.5, "logloss": 3.0, "n": len(chosen),
                "by_draft": {g: top1 for g in drafts},
                "by_draft_n": {g: 12 for g in drafts},
                "by_bucket": {"early": [1, 2, 4]}}

    monkeypatch.setattr(fp, "leave_one_draft_out", stub)
    code, target = _run(human_corpus, monkeypatch, tmp_path,
                        shipped=_anti_market_prior())

    assert code == 1
    assert not target.exists(), "the artifact shipped a set nothing measured"


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


# -------------------------------------------------- the checked-in artifact

def test_the_generated_prior_on_disk_still_agrees_with_the_feature_list():
    """The one check nothing else performs, because nothing imports this file.

    `scoring/mock_prior.py` is asserted against `FEATURE_NAMES` by
    `draft_model` at import, which is what catches a rename that leaves a
    generated vector pointing at the wrong columns. `scoring/human_prior.py`
    has no such importer ON PURPOSE -- the serving swap waits for the end of
    the ladder -- so it would sit misaligned and silent until the day someone
    binds `COLD_START_PRIOR` to it. This is that assertion, run from the
    tests instead.
    """
    from scoring import human_prior

    assert list(human_prior.FEATURES) == FEATURE_NAMES, (
        "the checked-in human prior was generated against a different feature "
        "list -- regenerate with `make score-ladder`")
    assert len(human_prior.PRIOR) == len(FEATURE_NAMES)
    assert np.isfinite(human_prior.PRIOR).all()

    fitted = set(human_prior.PROVENANCE["features_fitted"])
    assert fitted <= set(FEATURE_NAMES)
    # The invariant that ties the annotation to the value: a column the fit
    # did not have carries exactly 0.0, never a small leftover. `fp.full_beta`
    # scatters the fitted coefficients into a zero vector precisely so this
    # holds, and it is what lets a reader trust the "cut" notes beside them.
    for name in FEATURE_NAMES:
        if name not in fitted:
            assert human_prior.PRIOR[FEATURE_NAMES.index(name)] == 0.0, (
                f"{name} was not fitted but carries a coefficient")


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
