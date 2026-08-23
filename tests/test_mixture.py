"""The latent-class conditional logit: the EM, and the live posterior.

The load-bearing test in here is `test_k_of_one_reproduces_the_pooled_fit`.
Everything else in this module measures a K>1 model, and a K>1 model is only
worth reading if the machinery underneath it is right -- a mis-wired E-step, a
weight that never reaches the objective, or a shrinkage target applied to the
wrong vector all still produce K coefficient vectors and a table.
"""
import numpy as np
import pytest

from scoring import draft_model as dm
from scoring import mixture as mx


# --------------------------------------------------------------- fixtures

def _choice_set(rng, n_candidates=6, n_features=3):
    return rng.normal(size=(n_candidates, n_features))


def _draw(X, beta, rng):
    """One pick sampled from the conditional logit this seat really is."""
    scores = X @ beta
    probs = np.exp(scores - scores.max())
    return int(rng.choice(len(scores), p=probs / probs.sum()))


def _two_class_corpus(n_seats=40, picks_per_seat=8, seed=0):
    """Seats drawn from two genuinely different drafters.

    Class A and class B differ in the sign of the second coefficient and agree
    on the first, which is the shape the real question has: everybody follows
    the board, and what separates them is a preference on top of it.
    """
    rng = np.random.default_rng(seed)
    beta_a = np.array([2.0, +3.0, 0.0])
    beta_b = np.array([2.0, -3.0, 0.0])
    X_list, chosen, seats, truth = [], [], [], {}
    for s in range(n_seats):
        seat = f"seat-{s:02d}"
        beta = beta_a if s % 2 == 0 else beta_b
        truth[seat] = 0 if s % 2 == 0 else 1
        for _ in range(picks_per_seat):
            X = _choice_set(rng)
            X_list.append(X)
            chosen.append(_draw(X, beta, rng))
            seats.append(seat)
    return X_list, chosen, seats, truth, (beta_a, beta_b)


@pytest.fixture(scope="module")
def two_class():
    return _two_class_corpus()


# ------------------------------------------------- the K=1 correctness check

def test_k_of_one_reproduces_the_pooled_fit(two_class):
    """K=1 must BE the pooled fit, not merely resemble it.

    This is the whole EM machinery's correctness check and it is exact by
    construction rather than by luck: at K=1 every responsibility is 1.0, so
    the M-step maximizes `LL(beta) - lam*||beta - pooled||^2`, and `pooled` is
    the argmax of both terms separately. Any deviation past the optimizer's own
    tolerance means the responsibilities, the weights or the shrinkage target
    are wired wrong -- and every K>1 number this module produces would then be
    junk that still looks like a table.
    """
    X_list, chosen, seats, _, _ = two_class
    pooled = dm.fit(X_list, chosen)
    model = mx.fit_mixture(X_list, chosen, seats, 1, pooled=pooled)

    assert model.k == 1
    assert model.pi == pytest.approx([1.0])
    # `abs=1e-4` is L-BFGS-B's own slack and nothing else. The two vectors are
    # the same optimum reached from two starting points (zeros for the pooled
    # fit, `pooled` itself for the M-step), so the residual is the optimizer's
    # stopping tolerance, three orders of magnitude below the smallest
    # coefficient this model carries.
    assert model.betas[0] == pytest.approx(pooled, abs=1e-4)


def test_k_of_one_is_the_pooled_fit_at_every_shrinkage_strength(two_class):
    """...and for ANY lambda, which is the part that makes it a real check.

    If the identity only held at lam=0 it would be saying nothing about the
    penalty. It holds at lam=1000 too, because the penalty's own minimum is at
    the same vector the likelihood's maximum is.
    """
    X_list, chosen, seats, _, _ = two_class
    pooled = dm.fit(X_list, chosen)
    for lam in (0.0, 1.0, 1000.0):
        model = mx.fit_mixture(X_list, chosen, seats, 1, pooled=pooled, lam=lam)
        assert model.betas[0] == pytest.approx(pooled, abs=1e-4), lam


def test_k_of_one_scores_exactly_what_the_pooled_vector_scores(two_class):
    """The prequential scorer at K=1 is the fixed-vector scorer.

    A one-class posterior can never move, so `score_mixture` reduces to
    `fit_prior.score` on the pooled beta. Asserted because the ladder compares
    a mixture rung against a fixed-vector rung, and if those two scorers
    disagreed at the one point where they must agree, the comparison would be
    measuring the scorer rather than the model.
    """
    from pipeline import fit_prior as fp
    X_list, chosen, seats, _, _ = two_class
    pooled = dm.fit(X_list, chosen)
    model = mx.fit_mixture(X_list, chosen, seats, 1, pooled=pooled)
    mine = mx.score_mixture(X_list, chosen, seats, model)
    theirs = fp.score(pooled, X_list, chosen)
    assert mine["top1"] == pytest.approx(theirs["top1"])
    assert mine["logloss"] == pytest.approx(theirs["logloss"], abs=1e-9)


# ------------------------------------------------------------ the E-step

def test_the_e_step_survives_a_seat_whose_pick_probabilities_underflow():
    """A full draft's worth of picks cannot be multiplied out in float64.

    The naive E-step is `pi_k * prod_t P_k(pick_t)`. A well-fitting class on a
    16-pick seat lands around exp(-90), which is 268 orders of magnitude of
    headroom spent and already outside float32. A BADLY fitting class -- which
    is what a random EM initialization produces at every restart, and what
    every class looks like to the seats that are not its own -- lands far past
    exp(-745) and is exactly 0.0. Both classes then normalize as 0/0, the
    responsibilities are NaN, and the fit returns a mixture of NaNs rather than
    raising.

    This asserts the failure is real for the product and absent for the log
    space that is actually used. The betas are deliberately extreme for the
    same reason the EM's own initialization is: that is when it happens.
    """
    rng = np.random.default_rng(7)
    X_list, chosen, seats = [], [], []
    for _ in range(16):
        X = rng.normal(size=(200, 3))
        X_list.append(X)
        chosen.append(int(rng.integers(200)))
        seats.append("one-seat")
    betas = np.array([[40.0, 0.0, 0.0], [0.0, 40.0, 0.0]])
    seat_of, labels = mx.seat_index(seats)
    seat_ll = mx.seat_log_likelihoods(X_list, chosen, seat_of, len(labels), betas)

    # The failure the log space is there to prevent, demonstrated rather than
    # asserted about in a comment.
    assert np.exp(seat_ll).max() == 0.0

    R, total = mx.responsibilities(seat_ll, np.array([0.5, 0.5]))
    assert np.isfinite(R).all()
    assert R.sum(axis=1) == pytest.approx([1.0])
    assert np.isfinite(total)


def test_responsibilities_are_the_normalized_posterior():
    """Small enough numbers that the closed form can be written out and checked."""
    seat_ll = np.array([[-1.0, -2.0], [-3.0, -3.0]])
    pi = np.array([0.25, 0.75])
    R, total = mx.responsibilities(seat_ll, pi)
    raw = pi * np.exp(seat_ll)
    assert R == pytest.approx(raw / raw.sum(axis=1, keepdims=True))
    assert total == pytest.approx(float(np.log(raw.sum(axis=1)).sum()))


# --------------------------------------------------------------- the fit

def test_em_recovers_two_planted_classes(two_class):
    """The identifiability sanity check: when archetypes exist, EM finds them.

    Without this, a K=1 result on the real corpus would be unreadable -- "no
    archetypes in this data" and "this EM cannot find archetypes" would look
    identical. The seats here really were generated by two drafters who differ
    by 6.0 on one coefficient, and the assertion is that the fitted classes
    separate on that coefficient and that the seats land in the right ones.
    """
    X_list, chosen, seats, truth, (beta_a, beta_b) = two_class
    model = mx.fit_mixture(X_list, chosen, seats, 2, lam=1.0, restarts=3, seed=1)

    assert model.k == 2
    # The planted difference is on column 1. Whichever class came out first,
    # the two must straddle the pooled fit on that column.
    spread = model.betas[:, 1].max() - model.betas[:, 1].min()
    assert spread > 2.0, model.betas

    seat_of, labels = mx.seat_index(seats)
    seat_ll = mx.seat_log_likelihoods(X_list, chosen, seat_of, len(labels), model.betas)
    R, _ = mx.responsibilities(seat_ll, model.pi)
    assigned = R.argmax(axis=1)
    # Class labels are arbitrary, so agreement is measured up to a flip.
    actual = np.array([truth[label] for label in labels])
    agree = max((assigned == actual).mean(), (assigned != actual).mean())
    assert agree > 0.85, agree


def test_em_never_decreases_the_observed_data_log_likelihood(two_class):
    """EM's one guarantee, checked on the real objective rather than assumed.

    The M-step maximizes the EXPECTED complete-data likelihood, which is a
    different number; if the two were confused anywhere in here this sequence
    would wobble.
    """
    X_list, chosen, seats, _, _ = two_class
    seat_of, labels = mx.seat_index(seats)
    pooled = dm.fit(X_list, chosen)
    rng = np.random.default_rng(3)
    R = rng.dirichlet(np.full(2, mx.INIT_ALPHA), size=len(labels))
    pi = R.mean(axis=0)
    betas = np.vstack([mx._weighted_fit(X_list, chosen, R[seat_of, j], pooled, 1.0)
                       for j in range(2)])
    history = []
    for _ in range(6):
        seat_ll = mx.seat_log_likelihoods(X_list, chosen, seat_of, len(labels), betas)
        R, loglik = mx.responsibilities(seat_ll, pi)
        history.append(loglik)
        pi = R.mean(axis=0)
        betas = np.vstack([mx._weighted_fit(X_list, chosen, R[seat_of, j], pooled, 1.0)
                           for j in range(2)])
    assert all(b >= a - 1e-6 for a, b in zip(history, history[1:])), history


def test_shrinkage_pulls_every_class_toward_the_pooled_fit(two_class):
    """A bigger lambda has to leave the classes closer to pooled, or the
    penalty is not reaching the objective.

    The whole defence against K=3 producing one sensible class and two
    memorized ones is this pull, and `draft_model.fit(weights=...)` is a new
    code path -- a `lam` that silently did nothing would still return a
    plausible-looking mixture.
    """
    X_list, chosen, seats, _, _ = two_class
    pooled = dm.fit(X_list, chosen)
    loose = mx.fit_mixture(X_list, chosen, seats, 2, pooled=pooled, lam=0.1,
                           restarts=2, seed=5)
    tight = mx.fit_mixture(X_list, chosen, seats, 2, pooled=pooled, lam=5000.0,
                           restarts=2, seed=5)
    assert (np.abs(tight.betas - pooled).max()
            < np.abs(loose.betas - pooled).max())
    # At a large enough lambda the classes collapse onto pooled entirely,
    # which is the mixture saying "no transferable difference" -- the same
    # thing `select_lambda`'s top-of-grid behavior expresses for one manager.
    assert np.abs(tight.betas - pooled).max() < 0.2


def test_a_weighted_fit_ignores_the_observations_it_is_given_no_weight_for():
    """Zero responsibility must mean zero influence, not a small one.

    Fitting class k on the whole corpus with a weight column is only the same
    thing as fitting it on its own members if the weights actually gate the
    gradient.
    """
    rng = np.random.default_rng(11)
    beta_a, beta_b = np.array([3.0, 0.0]), np.array([-3.0, 0.0])
    X_list, chosen, weights = [], [], []
    for i in range(60):
        X = rng.normal(size=(5, 2))
        X_list.append(X)
        chosen.append(_draw(X, beta_a if i < 30 else beta_b, rng))
        weights.append(1.0 if i < 30 else 0.0)
    pooled = np.zeros(2)
    weighted = mx._weighted_fit(X_list, chosen, np.array(weights), pooled, 0.0)
    only_first = dm.fit(X_list[:30], chosen[:30])
    assert weighted == pytest.approx(only_first, abs=1e-4)


def test_restarts_keep_the_best_training_likelihood(two_class):
    """More restarts can only help, because the best is what is kept.

    EM finds local optima, and a bad initialization collapses to one populated
    class -- which reads exactly like "there are no archetypes". This is the
    guard that lets a K=1 result be reported as a finding instead of as a
    possible optimizer failure.
    """
    X_list, chosen, seats, _, _ = two_class
    one = mx.fit_mixture(X_list, chosen, seats, 2, lam=1.0, restarts=1, seed=2)
    many = mx.fit_mixture(X_list, chosen, seats, 2, lam=1.0, restarts=4, seed=2)
    assert many.loglik >= one.loglik - 1e-9


def test_classes_come_back_sorted_by_population_weight(two_class):
    X_list, chosen, seats, _, _ = two_class
    model = mx.fit_mixture(X_list, chosen, seats, 3, lam=10.0, restarts=2, seed=4)
    assert list(model.pi) == sorted(model.pi, reverse=True)
    assert model.pi.sum() == pytest.approx(1.0)


# ------------------------------------------------------- live inference

def test_a_fresh_seat_starts_at_the_population_weights(two_class):
    X_list, chosen, seats, _, _ = two_class
    model = mx.fit_mixture(X_list, chosen, seats, 2, lam=1.0, restarts=2, seed=6)
    assert mx.seed_posterior(model.pi) == pytest.approx(model.pi)


def test_the_cold_start_prediction_is_the_pi_weighted_mixture(two_class):
    """At pick 1 the posterior IS pi, so the prediction is the population
    average of the classes and nothing about the seat has been assumed."""
    X_list, chosen, seats, _, _ = two_class
    model = mx.fit_mixture(X_list, chosen, seats, 2, lam=1.0, restarts=2, seed=6)
    X = X_list[0]
    manual = np.zeros(len(X))
    for weight, beta in zip(model.pi, model.betas):
        scores = X @ beta
        probs = np.exp(scores - scores.max())
        manual += weight * probs / probs.sum()
    assert mx.mixture_probs(X, model.betas, mx.seed_posterior(model.pi)) \
        == pytest.approx(manual)


def test_the_single_pick_answer_averages_probabilities_not_scores():
    """`softmax(mean(scores))` is a different distribution and is not the model.

    Averaging inside the exponent invents a (K+1)th drafter who is nobody: it
    is the geometric mean of the classes, which concentrates on candidates
    every class likes and suppresses one that half the population would take
    first. The two are asserted to DIFFER as well as the right one asserted
    correct, because a bug here would be invisible -- both are probability
    vectors over the same candidates.
    """
    X = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    betas = np.array([[3.0, 0.0], [0.0, 3.0]])
    posterior = np.array([0.5, 0.5])
    averaged = mx.mixture_probs(X, betas, posterior)

    scores = X @ betas.T @ posterior
    of_scores = np.exp(scores - scores.max())
    of_scores = of_scores / of_scores.sum()

    manual = np.zeros(len(X))
    for weight, beta in zip(posterior, betas):
        exp = np.exp(X @ beta - (X @ beta).max())
        manual += weight * exp / exp.sum()
    assert averaged == pytest.approx(manual)
    assert averaged.sum() == pytest.approx(1.0)
    assert not np.allclose(averaged, of_scores)
    # The candidate every class merely tolerates is where they disagree most.
    assert averaged[2] < of_scores[2]


def test_the_incremental_posterior_equals_the_batch_e_step():
    """`update_posterior` called pick by pick must equal one E-step over the
    seat's whole history. It is the same posterior; the only reason it is
    written incrementally is that a live caller cannot replay a seat's draft on
    every pick."""
    rng = np.random.default_rng(13)
    betas = np.array([[2.0, 1.0, 0.0], [2.0, -1.0, 0.0]])
    pi = np.array([0.4, 0.6])
    X_list, chosen = [], []
    for _ in range(9):
        X = _choice_set(rng, n_candidates=40)
        X_list.append(X)
        chosen.append(_draw(X, betas[0], rng))

    posterior = mx.seed_posterior(pi)
    for X, k in zip(X_list, chosen):
        posterior = mx.update_posterior(posterior, X, k, betas)

    seat_of, labels = mx.seat_index(["s"] * len(chosen))
    seat_ll = mx.seat_log_likelihoods(X_list, chosen, seat_of, len(labels), betas)
    batch, _ = mx.responsibilities(seat_ll, pi)
    assert posterior == pytest.approx(batch[0])


def test_a_seat_that_reveals_itself_moves_its_posterior():
    """The point of the whole exercise: after enough picks, a seat drawn from
    one class is recognised as that class."""
    rng = np.random.default_rng(17)
    betas = np.array([[1.0, 4.0, 0.0], [1.0, -4.0, 0.0]])
    posterior = mx.seed_posterior(np.array([0.5, 0.5]))
    for _ in range(10):
        X = _choice_set(rng, n_candidates=30)
        posterior = mx.update_posterior(posterior, X, _draw(X, betas[0], rng), betas)
    assert posterior[0] > 0.95, posterior


def test_sampling_a_class_follows_the_posterior():
    """A rollout draws one archetype per seat and holds it, so the draw has to
    be the posterior and nothing else."""
    rng = np.random.default_rng(19)
    draws = [mx.sample_class(np.array([0.2, 0.8]), rng) for _ in range(4000)]
    assert 0.75 < np.mean(draws) < 0.85
    certain = np.array([0.0, 1.0, 0.0])
    assert {mx.sample_class(certain, rng) for _ in range(20)} == {1}


# ------------------------------------------------------------- scoring

def test_the_score_report_has_the_same_shape_as_the_fixed_vector_scorer(two_class):
    """The ladder pairs a mixture rung against fixed-vector rungs with
    `paired_se`, which reads `by_draft`. A report missing a key would either
    crash the comparison or, worse, silently skip a rung."""
    from pipeline import fit_prior as fp
    X_list, chosen, seats, _, _ = two_class
    model = mx.fit_mixture(X_list, chosen, seats, 2, lam=1.0, restarts=2, seed=8)
    groups = ["d1"] * (len(chosen) // 2) + ["d2"] * (len(chosen) - len(chosen) // 2)
    buckets = ["early"] * len(chosen)
    mine = mx.score_mixture(X_list, chosen, seats, model, groups=groups,
                            buckets=buckets)
    theirs = fp.score(model.pooled, X_list, chosen, groups=groups, buckets=buckets)
    assert set(mine) == set(theirs)


def test_scoring_is_prequential_and_cannot_see_a_seats_later_picks(two_class):
    """Truncating a seat's history must not change how its earlier picks scored.

    The one way a prequential scorer gets quietly wrong is by updating a
    posterior before predicting with it, which would let every pick be scored
    partly by itself. Scoring the first half of the corpus alone has to give
    the identical per-pick numbers as scoring all of it and reading the first
    half off.
    """
    X_list, chosen, seats, _, _ = two_class
    model = mx.fit_mixture(X_list, chosen, seats, 2, lam=1.0, restarts=2, seed=9)
    half = len(chosen) // 2
    groups = ["all"] * len(chosen)
    whole = mx.score_mixture(X_list, chosen, seats, model, groups=groups)
    prefix = mx.score_mixture(X_list[:half], chosen[:half], seats[:half], model,
                              groups=groups[:half])
    # The prefix's own totals must be a prefix of the whole run's, which is
    # only true if no pick was scored using anything after it.
    running = mx.score_mixture(X_list[:half], chosen[:half], seats[:half], model)
    assert prefix["hits1"] == running["hits1"]
    assert whole["n"] == len(chosen)
    assert prefix["n"] == half


def test_the_held_out_marginal_is_the_number_k_is_chosen_by(two_class):
    """A mixture that really is two classes must beat a one-class model on
    seats neither was fitted on. If this did not hold on planted data, a K=1
    result on the real corpus would say nothing."""
    X_list, chosen, seats, _, _ = _two_class_corpus(n_seats=60, seed=21)
    hold = set(sorted(set(seats))[:20])
    train = [i for i, s in enumerate(seats) if s not in hold]
    test = [i for i, s in enumerate(seats) if s in hold]

    def sub(idx):
        return ([X_list[i] for i in idx], [chosen[i] for i in idx],
                [seats[i] for i in idx])

    Xtr, ctr, str_ = sub(train)
    Xte, cte, ste = sub(test)
    one = mx.fit_mixture(Xtr, ctr, str_, 1, lam=1.0)
    two = mx.fit_mixture(Xtr, ctr, str_, 2, lam=1.0, restarts=3, seed=1)
    assert (mx.seat_marginal_loglik(Xte, cte, ste, two)["per_pick"]
            > mx.seat_marginal_loglik(Xte, cte, ste, one)["per_pick"])


def test_describe_reads_a_class_against_pooled_not_against_zero():
    """`reach` is about -9 for everybody, so a description written against zero
    would say the same thing about every class."""
    pooled = np.array([-9.0, 2.0, 0.0])
    beta = np.array([-9.0, 2.0, 1.5])
    text = mx.describe_class(beta, pooled, ["reach", "fall", "qb_early"], top=1)
    assert text == "qb_early +1.50"


# --------------------------------------------------- the serving object

def test_seat_beliefs_seeds_an_unseen_seat_at_the_population_weights():
    """No entry means nothing observed, and nothing observed means `pi`.

    Written as the ABSENCE of an entry rather than as a pre-populated dict, so
    the cold start is not a case anyone has to remember to initialize: a slot
    that never picked and a slot that does not exist behave identically and
    correctly.
    """
    betas = np.array([[1.0, 0.0], [0.0, 1.0]])
    beliefs = mx.SeatBeliefs(betas=betas, pi=np.array([0.6, 0.4]))
    assert beliefs.k == 2
    assert beliefs.of("never-seen") == pytest.approx([0.6, 0.4])
    assert beliefs.posterior == {}


def test_seat_beliefs_observe_matches_the_free_function():
    betas = np.array([[2.0, 0.0], [0.0, 2.0]])
    beliefs = mx.SeatBeliefs(betas=betas, pi=np.array([0.5, 0.5]))
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    beliefs.observe("s", X, 0)
    assert beliefs.of("s") == pytest.approx(
        mx.update_posterior(np.array([0.5, 0.5]), X, 0, betas))


def test_a_drawn_archetype_is_one_of_the_classes_and_only_one():
    """`draw` hands the caller a real coefficient vector per seat, not a blend.

    The blend is `probs`, and it is for a single pick. If `draw` ever returned
    an average the rollouts would be simulating a drafter who is nobody, which
    is the exact failure the sampling design exists to prevent -- and it would
    be invisible, because an averaged vector is the same shape and produces
    plausible drafts.
    """
    betas = np.array([[3.0, 0.0], [0.0, 3.0]])
    beliefs = mx.SeatBeliefs(betas=betas, pi=np.array([0.5, 0.5]))
    rng = np.random.default_rng(0)
    drawn = beliefs.draw([1, 2, 3], rng)
    assert set(drawn) == {1, 2, 3}
    for beta in drawn.values():
        assert any(np.array_equal(beta, row) for row in betas), beta


def test_a_certain_seat_is_always_drawn_as_the_class_it_is():
    """Once a seat's posterior has collapsed, sampling stops adding noise:
    every rollout draws the class the evidence points at."""
    betas = np.array([[3.0, 0.0], [0.0, 3.0]])
    beliefs = mx.SeatBeliefs(betas=betas, pi=np.array([0.5, 0.5]),
                             posterior={7: np.array([0.0, 1.0])})
    rng = np.random.default_rng(1)
    for _ in range(20):
        assert np.array_equal(beliefs.draw([7], rng)[7], betas[1])
