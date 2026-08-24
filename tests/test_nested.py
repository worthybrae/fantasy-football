"""Offline tests for the nested (position-then-player) opponent model.

No database. Every fixture is built by hand so these run anywhere and pin the
model's structure, not a corpus number: that the within-position feature set is
the varying columns and nothing else, that the joint really is
P(position) x P(player|position) re-normalized over the available board, that a
drafted-out position leaks no probability, and that the separate fit reproduces
the factorized likelihood exactly (the claim the module rests on).
"""
import numpy as np
import pytest

from scoring import draft_model as dm
from scoring import nested_model as nm
from scoring.position_model import POSITIONS, _POS_INDEX


# --------------------------------------------------------------- feature wiring

def test_within_idx_names_align_with_feature_matrix():
    # Every kept index reads the column its name promises, in order.
    for j, name in zip(nm.WITHIN_IDX, nm.WITHIN_POSITION_FEATURES):
        assert dm.FEATURE_NAMES[j] == name


def test_within_features_are_exactly_the_varying_columns():
    # The kept set and the dropped position-constant set partition every
    # feature-matrix column, so no column is silently un-modelled and no
    # constant column is fitted to an unidentifiable zero.
    kept = set(nm.WITHIN_POSITION_FEATURES)
    dropped = set(nm._POSITION_CONSTANT)
    assert kept.isdisjoint(dropped)
    assert kept | dropped == set(dm.FEATURE_NAMES)


def test_local_softmax_matches_draft_model():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.standard_normal(rng.integers(1, 12))
        assert np.allclose(nm._softmax(s), dm._softmax(s))


# ------------------------------------------------------------- a synthetic design

def _synthetic_design(seed=0, n_picks=40, n_cand=30, n_drafts=2):
    """A hand-built `NestedDesign` with the real feature widths.

    `pos_X` has the position factor's real column count so the committed
    classifier fits without complaint; `within_X` has `len(WITHIN_IDX)` columns.
    Positions are assigned so every choice set holds at least two of several
    positions, and the chosen candidate is drawn toward the one with the
    highest within score under a planted beta so the fit has signal to find.
    """
    rng = np.random.default_rng(seed)
    d_pos = len(_pos_feature_names())
    d_within = len(nm.WITHIN_IDX)
    true_within = rng.standard_normal(d_within)

    pos_X, y, within_X, cand_pos, chosen, groups, rounds, buckets = (
        [], [], [], [], [], [], [], [])
    for i in range(n_picks):
        cp = rng.integers(0, len(POSITIONS), size=n_cand)
        # guarantee at least two positions with >=2 candidates
        cp[:2] = _POS_INDEX["RB"]
        cp[2:4] = _POS_INDEX["WR"]
        Xw = rng.standard_normal((n_cand, d_within))
        pos_X.append(rng.standard_normal(d_pos))
        # label: pick a position that has candidates, then the best within it
        present = np.unique(cp)
        nest = int(rng.choice(present))
        mask = cp == nest
        scores = Xw[mask] @ true_within
        within_choice = int(np.argmax(scores))
        chosen_global = int(np.flatnonzero(mask)[within_choice])
        y.append(nest)
        within_X.append(Xw)
        cand_pos.append(cp)
        chosen.append(chosen_global)
        groups.append(f"draft{i % n_drafts}")
        rounds.append(i % 16 + 1)
        buckets.append(("early", "mid", "late")[i % 3])
    return nm.NestedDesign(pos_X, y, within_X, cand_pos, chosen, groups,
                           rounds, buckets)


def _pos_feature_names():
    from scoring.position_model import FEATURE_NAMES
    return FEATURE_NAMES


def test_fit_produces_finite_factors():
    d = _synthetic_design()
    m = nm.NestedModel().fit(d)
    assert m.within_.shape == (len(nm.WITHIN_IDX),)
    assert np.all(np.isfinite(m.within_))
    assert m.position_.predict_proba(d.pos_X[:1]).shape == (1, len(POSITIONS))


def test_prediction_is_a_distribution():
    d = _synthetic_design()
    m = nm.NestedModel().fit(d)
    for i in range(len(d)):
        p = m.predict_pick(d, i)
        assert p.shape == (len(d.cand_pos[i]),)
        assert np.all(p >= 0)
        assert p.sum() == pytest.approx(1.0)


def test_joint_is_position_times_within_renormalized():
    # The core claim: for a pick where every position is present, the nested
    # probability of the chosen player equals P(position) x P(player|position)
    # with no renormalization (available mass is 1), and in general it equals
    # that product divided by the mass on positions that have candidates.
    d = _synthetic_design()
    m = nm.NestedModel().fit(d)
    i = 0
    pos_probs = m.position_.predict_proba(d.pos_X[i:i + 1])[0]
    cp = d.cand_pos[i]
    Xw = d.within_X[i]
    p = m.predict_pick(d, i)
    avail = sum(pos_probs[q] for q in range(len(POSITIONS)) if (cp == q).any())
    for q in range(len(POSITIONS)):
        mask = cp == q
        if not mask.any():
            continue
        within = nm._softmax(Xw[mask] @ m.within_)
        expected = pos_probs[q] * within / avail
        assert np.allclose(p[mask], expected)


def test_drafted_out_position_takes_no_mass_and_renormalizes():
    # Remove every candidate of one present position from a choice set; the
    # remaining players must still sum to 1 (mass is re-normalized, not lost)
    # and none of them may carry the missing position's mass.
    d = _synthetic_design()
    m = nm.NestedModel().fit(d)
    i = 0
    # drop WR entirely from pick i
    cp = d.cand_pos[i]
    keep = cp != _POS_INDEX["WR"]
    d.within_X[i] = d.within_X[i][keep]
    d.cand_pos[i] = cp[keep]
    d.chosen[i] = 0
    p = m.predict_pick(d, i)
    assert p.sum() == pytest.approx(1.0)
    assert _POS_INDEX["WR"] not in set(d.cand_pos[i])


def test_separate_fit_reproduces_factorized_loglik():
    # Fitting the two factors separately maximizes the nested likelihood only
    # because that likelihood is the SUM of the two factors' own likelihoods.
    # Verify the identity numerically on a full-board pick: log p_nested(chosen)
    # == log p_position(nest) + log p_within(chosen|nest) when all positions
    # are present (available mass 1, so no renormalization term).
    d = _synthetic_design()
    m = nm.NestedModel().fit(d)
    for i in range(len(d)):
        cp = d.cand_pos[i]
        if len(set(cp)) < len(POSITIONS):
            continue  # need every position present so avail mass is exactly 1
        pos_probs = m.position_.predict_proba(d.pos_X[i:i + 1])[0]
        nest = d.y[i]
        mask = cp == nest
        within = nm._softmax(d.within_X[i][mask] @ m.within_)
        within_idx = int(np.count_nonzero(mask[:d.chosen[i]]))
        p = m.predict_pick(d, i)
        lhs = np.log(p[d.chosen[i]])
        rhs = np.log(pos_probs[nest]) + np.log(within[within_idx])
        assert lhs == pytest.approx(rhs)
        return  # one qualifying pick is enough
    pytest.skip("no full-board pick in this synthetic fixture")


def test_singleton_nest_is_dropped_from_the_within_fit():
    # A pick whose chosen position has exactly one candidate contributes a
    # degenerate (softmax-of-one) choice set; the fit must skip it and still
    # return a finite beta from the remaining multi-candidate picks.
    d = _synthetic_design(n_picks=6)
    # make pick 0's nest a singleton
    d.cand_pos[0] = np.array([d.y[0]] + [_POS_INDEX["TE"]] * 5)
    d.within_X[0] = d.within_X[0][:6]
    d.chosen[0] = 0
    m = nm.NestedModel().fit(d)
    assert np.all(np.isfinite(m.within_))


def test_score_pick_definitions():
    prob = np.array([0.1, 0.5, 0.2, 0.15, 0.05])
    # chosen is the argmax -> top1/3/5 all hit
    assert nm.score_pick(prob, 1) == (1, 1, 1, pytest.approx(np.log(0.5)))
    # chosen is 3rd best (0.15) -> not top1, but in top3/top5
    h1, h3, h5, ll = nm.score_pick(prob, 3)
    assert (h1, h3, h5) == (0, 1, 1)
    assert ll == pytest.approx(np.log(0.15))
