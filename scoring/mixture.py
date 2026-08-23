"""Drafter archetypes: a latent-class conditional logit, fitted by EM.

WHAT THIS IS FOR. Every opponent in a mock draft is scored by ONE coefficient
vector -- `draft_model.COLD_START_PRIOR` -- so the model cannot express "seat 3
reaches for running backs, seat 6 waits on quarterback". Not poorly: at all.
The shipped prior is a single point in coefficient space and eight strangers
are all simulated from it. This module is the smallest thing that can say
otherwise: K coefficient vectors `betas[k]`, population weights `pi[k]`, and a
per-seat posterior over which of the K a given opponent is.

THE CLUSTERING UNIT IS A SEAT WITHIN ONE DRAFT, NOT A PERSON. Mock opponents
are strangers with no identity that survives the session -- `draft_log`
literally keys them `anon:{draft_id}:{slot}` (`ANONYMOUS_PREFIX`) -- so there
is no "manager" to pool across drafts and asking for one would be inventing an
entity the data does not contain. A seat is 2 to 16 picks by whoever was
sitting there for that hour, and that is the whole unit of behavior available.
`PickObservation.manager` already carries exactly that key, which is why
nothing upstream had to change to fit this.

THE EM, AND THE THREE THINGS THAT KEEP IT UPRIGHT.

  E-step  The responsibility of class k for seat s is `pi_k` times the product
          of that seat's pick likelihoods under `betas[k]`, normalized over k.
          Computed IN LOG SPACE, always. Sixteen conditional-logit
          probabilities over a ~200-player choice set multiply out to
          exp(-90)-ish even when the class FITS -- 268 orders of magnitude of
          headroom gone, and already outside float32. When it does not fit,
          which is every class's relationship to most seats and every class's
          state at a random initialization, the product runs past exp(-745) and
          is exactly 0.0; both classes then normalize as 0/0 and the fit
          returns a mixture of NaNs rather than raising. `_log_sum_exp` is the
          only normalization used anywhere in here.

  M-step  `pi_k` is the mean responsibility over seats. Each `betas[k]` is a
          responsibility-WEIGHTED conditional logit fit -- `draft_model.fit`
          with `weights=`, which is that function's own objective with each
          pick counted fractionally (see its docstring for why a fractional
          count cannot be done by duplicating rows).

  1  SHRINKAGE TOWARD THE POOLED FIT. Each `betas[k]` is fitted toward
     `pooled` under a ridge penalty, the same `prior=`/`lam=` mechanism a
     per-manager fit uses. 29 coefficients per class against ~2,700 human
     picks split three ways is ~31 picks per parameter, which is not enough to
     estimate a class from scratch; without the pull toward pooled, K=3 gives
     one sensible class and two that have memorized a handful of seats.

     This is also what makes the K=1 correctness check exact rather than
     approximate. At K=1 every responsibility is 1.0, so the M-step maximizes
     `LL(beta) - lam*||beta - pooled||^2`. `pooled` is by construction the
     argmax of the first term and obviously the argmax of the second, so the
     sum is maximized at `pooled` itself -- for ANY lam. K=1 therefore
     reproduces the pooled fit exactly, and `tests/test_mixture.py` asserts it.
     A K=1 that came back as anything else would mean the E-step, the weights
     or the shrinkage are wired wrong, and every K>1 number would be junk.

  2  K CHOSEN BY HELD-OUT LIKELIHOOD, leave-one-draft-out. Not AIC, not BIC:
     both charge a fixed price per parameter derived from asymptotics this
     sample is nowhere near, and both would be answering "how many parameters
     can 2,700 picks afford" when the question is "does splitting the seats
     predict a draft nobody in the fit has seen". `seat_marginal_loglik` is
     the number, and it is the exact quantity a live posterior-updating
     predictor achieves (see that function).

  3  MULTIPLE RANDOM RESTARTS, best training likelihood kept. EM finds local
     optima. A bad initialization collapses to one populated class and K-1
     empty ones, which is INDISTINGUISHABLE from "there are no archetypes" --
     and reporting K=1 without having ruled that out would be reporting the
     optimizer's failure as a finding about drafters.

K=1 WINNING IS A REAL OUTCOME AND IS A FINDING, NOT A FAILURE. This corpus is
~240 seats. Nothing here should be tuned until something splits.

LIVE INFERENCE: ROLLOUTS SAMPLE, THE SINGLE-PICK ANSWER AVERAGES.

`seed_posterior` starts every opponent seat at `pi`, so at pick 1 the mixture's
prediction is the `pi`-weighted average of the classes and the cold start
cannot be worse than a pooled vector fitted the same way. `update_posterior`
Bayes-updates a seat after each pick it makes.

  * For "who does this team take next", `mixture_probs` averages the K
    probability vectors under that seat's posterior. That is the posterior
    predictive distribution and it is the right answer for a single pick.
  * Inside a rollout, `sample_class` draws ONE archetype per seat and holds it
    for the whole simulated draft. Averaging inside a rollout would let a seat
    be zero-RB on one pick and QB-early on the next, which is not a drafter and
    would produce a draft no archetype would ever have produced. Sampling
    preserves within-draft consistency, and the spread ACROSS rollouts is then
    a genuine representation of not yet knowing who a seat is. Measured, that
    spread is in what a WHOLE ROLLOUT looks like -- the count of running backs
    gone in two rounds moves from a standard deviation of 1.9 to 3.5 -- and
    barely at all in any individual player's marginal survival, because eight
    seats drawn independently still average out to the population mixture
    inside one rollout. See `draft_sim.survival`'s docstring.

WHAT IS DELIBERATELY NOT HERE. Nothing in this module knows what a pool, a
board or a roster is. It takes design matrices and chosen indices, which is
exactly what `draft_model.feature_matrix` produces for the fit and what
`draft_sim._live_features` produces for serving. Those two are one definition
with two callers on purpose (`tests/test_draft_sim.py` asserts they agree end
to end) and a mixture that reimplemented either would reintroduce the fit/serve
split they exist to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scoring import draft_model as dm

# The ridge strength each class is shrunk toward `pooled` with, in
# pseudo-observations: `lam * ||beta_k - pooled||^2` is added to a summed (not
# averaged) negative log-likelihood, so a class holding 300 effective picks
# feels this ~9x more strongly than one holding all 2,700. That asymmetry is
# wanted -- the small class is exactly the one that cannot afford 29 free
# coefficients -- and it is why the default is a single number rather than
# something scaled per class.
#
# 10.0 sits in the middle of `draft_model.LAMBDA_GRID`, the grid the
# per-manager fits cross-validate their own shrinkage over. It is a DEFAULT,
# not a measurement: `pipeline/fit_archetypes.py` sweeps it against the same
# held-out likelihood that chooses K, and the value that ships is whatever that
# sweep picked.
DEFAULT_LAMBDA = 10.0

# EM restarts. Five rather than two because the failure this guards against is
# silent: a run that collapses to one populated class prints a clean K=1-shaped
# answer and nothing about it looks wrong.
DEFAULT_RESTARTS = 5

# EM iterations, and the tolerance that usually stops it far short of them.
# The tolerance is on the observed-data log-likelihood in nats over the whole
# training set -- an absolute scale, because that is the number being compared
# between restarts and between K.
DEFAULT_MAX_ITER = 60
DEFAULT_TOL = 1e-4

# Dirichlet concentration for the random initial responsibilities. BELOW 1.0 ON
# PURPOSE. At alpha=1.0 the draw is uniform over the simplex, whose mean is
# 1/K, so every class's first weighted fit sees essentially the whole corpus at
# equal weight, every `betas[k]` comes back at almost exactly `pooled`, the
# first E-step is then indifferent between identical classes, and EM sits at
# that saddle point. alpha=0.5 puts most of each seat's mass on one class --
# nearly a random hard assignment -- so the K first-round fits are genuinely
# different models and the E-step has something to separate.
INIT_ALPHA = 0.5

# Responsibilities below this are dropped from a class's M-step rather than
# fitted with a ~0 weight. Purely a speed measure and it changes no answer that
# float64 can see: a pick contributing 1e-9 of a gradient built from ~2,700
# picks is below the optimizer's own convergence tolerance. It matters because
# responsibilities go nearly degenerate for any seat with more than a handful
# of picks, so this makes each class's fit cost scale with that class's SHARE
# of the corpus and keeps one EM iteration at roughly the price of one pooled
# fit however large K is.
MIN_RESPONSIBILITY = 1e-6


def _log_sum_exp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(sum(exp(values))) without overflowing or underflowing.

    Written out rather than imported so that the one numerically delicate step
    in this module is visible in it. `scipy.special.logsumexp` would do the
    same thing; the point is that a reader auditing the E-step for the
    underflow this module's docstring warns about can see the max-subtraction
    happen here instead of taking it on trust.
    """
    peak = np.max(values, axis=axis, keepdims=True)
    # A class with zero weight contributes log(0) = -inf, and -inf - -inf is
    # NaN. Clamping the peak to a finite number leaves exp(-inf - c) = 0, which
    # is the right contribution, and the all--inf row (impossible here, since
    # some class always has positive weight) comes back as -inf rather than NaN.
    peak = np.where(np.isfinite(peak), peak, 0.0)
    total = peak + np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True))
    return np.squeeze(total, axis=axis)


# ------------------------------------------------------------------- seats

def seat_index(seats) -> tuple:
    """`(index per observation, sorted seat labels)`.

    Sorted rather than first-seen so that two runs over the same corpus number
    the seats identically. Nothing downstream depends on the ORDER, but a
    responsibility matrix whose rows mean different seats between two runs is
    the kind of thing that only shows up as an unreproducible number.
    """
    labels = sorted(set(seats))
    position = {label: i for i, label in enumerate(labels)}
    return np.array([position[s] for s in seats], dtype=int), labels


def pick_log_probs(X_list, chosen, beta) -> np.ndarray:
    """log P(the pick that was made | that pick's choice set), one per pick.

    The conditional logit's own likelihood, evaluated candidate-set by
    candidate-set, which is the atom every other number in this module is built
    from: the seat likelihoods the E-step normalizes, the marginal likelihood K
    is chosen by, and the live Bayes update all sum these.
    """
    beta = np.asarray(beta, dtype=float)
    out = np.empty(len(chosen), dtype=float)
    for i, (X, k) in enumerate(zip(X_list, chosen)):
        scores = X @ beta
        peak = scores.max()
        out[i] = scores[k] - (peak + np.log(np.exp(scores - peak).sum()))
    return out


def seat_log_likelihoods(X_list, chosen, seat_of, n_seats, betas) -> np.ndarray:
    """`(n_seats, K)`: each seat's total log-likelihood under each class.

    A sum of logs, never a product of probabilities. Sixteen picks over a
    200-candidate pool multiply to something around exp(-90) and a seat that
    played a full draft would underflow to exactly zero, taking the whole
    E-step with it.
    """
    betas = np.atleast_2d(betas)
    out = np.zeros((n_seats, len(betas)), dtype=float)
    for k, beta in enumerate(betas):
        out[:, k] = np.bincount(seat_of, weights=pick_log_probs(X_list, chosen, beta),
                                minlength=n_seats)
    return out


# --------------------------------------------------------------- the E step

def responsibilities(seat_ll: np.ndarray, pi: np.ndarray) -> tuple:
    """`(R, observed-data log-likelihood)`. R is `(n_seats, K)`, rows sum to 1.

    `R[s, k]` is P(seat s belongs to class k | that seat's picks), which is
    `pi_k * prod_t P_k(pick_t)` normalized over k -- done as
    `log pi_k + sum_t log P_k(pick_t)` normalized by `_log_sum_exp`.

    The second return value is `sum_s log sum_k pi_k prod_t P_k(pick_t)`: the
    marginal likelihood of the data with the class memberships integrated out.
    That is the quantity EM is guaranteed to increase every iteration, so it is
    both the convergence test and the number restarts are compared on. The
    complete-data likelihood that the M-step maximizes is NOT that number and
    is not comparable across restarts.
    """
    with np.errstate(divide="ignore"):
        joint = seat_ll + np.log(np.asarray(pi, dtype=float))
    norm = _log_sum_exp(joint, axis=1)
    return np.exp(joint - norm[:, None]), float(norm.sum())


# --------------------------------------------------------------- the M step

def _weighted_fit(X_list, chosen, weights, pooled, lam, start=None) -> np.ndarray:
    """One class's responsibility-weighted, pooled-shrunk conditional logit.

    Observations whose weight is below `MIN_RESPONSIBILITY` are dropped rather
    than passed in with a near-zero weight; see that constant. A class that
    ends up with NO observation above the floor is an empty class, and the fit
    it gets is `pooled` -- the honest answer, since the data has said nothing
    about it, and the same thing the ridge penalty alone would converge to.

    `start` is the previous EM iteration's coefficients for this class. The
    objective is convex so it cannot change the answer, only the number of
    L-BFGS steps taken to reach it, and after the first iteration that is the
    difference between a few steps and a full cold solve. See `draft_model.fit`.
    """
    keep = np.flatnonzero(np.asarray(weights) > MIN_RESPONSIBILITY)
    if len(keep) == 0:
        return np.asarray(pooled, dtype=float).copy()
    return dm.fit([X_list[i] for i in keep], [chosen[i] for i in keep],
                  prior=pooled, lam=lam, start=start,
                  weights=np.asarray(weights, dtype=float)[keep])


# ------------------------------------------------------------- the model

@dataclass
class Archetypes:
    """A fitted mixture: K coefficient vectors and their population weights.

    `pooled` rides along because it is not decoration -- it is the vector every
    class was shrunk toward, so it is the only reference against which a
    class's coefficients can be READ ("this class reaches 3 nats harder than
    the room average" is a statement about a difference). It is also what a
    K=1 model is, exactly, which makes the degenerate case inspectable.
    """
    betas: np.ndarray                 # (K, n_features)
    pi: np.ndarray                    # (K,)
    pooled: np.ndarray                # (n_features,)
    loglik: float = float("nan")      # observed-data log-likelihood, training
    lam: float = DEFAULT_LAMBDA
    n_iter: int = 0
    converged: bool = False
    restart: int = 0
    n_seats: int = 0
    n_picks: int = 0

    @property
    def k(self) -> int:
        return len(self.betas)

    def order_by_weight(self) -> "Archetypes":
        """The same model with its classes sorted by population weight.

        EM numbers its classes by whichever restart happened to win, so class 0
        means nothing across two runs. Sorting makes a printed table and a
        generated artifact stable enough to diff, and it is a relabelling: the
        mixture is invariant to it.
        """
        order = np.argsort(-self.pi)
        return Archetypes(betas=self.betas[order], pi=self.pi[order],
                          pooled=self.pooled, loglik=self.loglik, lam=self.lam,
                          n_iter=self.n_iter, converged=self.converged,
                          restart=self.restart, n_seats=self.n_seats,
                          n_picks=self.n_picks)


def fit_mixture(X_list, chosen, seats, k: int, pooled=None,
                lam: float = DEFAULT_LAMBDA, restarts: int = DEFAULT_RESTARTS,
                seed: int = 0, max_iter: int = DEFAULT_MAX_ITER,
                tol: float = DEFAULT_TOL) -> Archetypes:
    """Fit K archetypes by EM, keeping the best of `restarts` random starts.

    `seats` is one seat key per observation -- `PickObservation.manager`, which
    for corpus mocks is `anon:{draft_id}:{slot}`. Observations sharing a key
    are one drafter and get one responsibility vector between them.

    `pooled` is the plain unweighted, unpenalized pooled fit. Passed in when
    the caller already has it (a leave-one-draft-out loop refits it per fold
    anyway) and computed here otherwise. It is BOTH the shrinkage target and
    the K=1 answer, and it is deliberately the same object in both roles -- see
    the module docstring's correctness check.

    K=1 SKIPS THE RESTARTS, and not as an optimization. At K=1 there is one
    local optimum, it is `pooled`, and every restart would find it: running
    five of them would be five identical fits and would invite reading the
    restart count as evidence about a case where it says nothing.
    """
    seat_of, labels = seat_index(seats)
    n_seats = len(labels)
    pooled = (np.asarray(pooled, dtype=float) if pooled is not None
              else dm.fit(X_list, chosen))

    if k == 1:
        # The M-step at K=1, written out: all responsibilities are 1.0, so this
        # IS `_weighted_fit` with unit weights, which is `pooled` (both terms
        # of the penalized objective are maximized there). Computing it rather
        # than asserting it means the assertion in the tests is testing the
        # general path against this one instead of against itself.
        betas = _weighted_fit(X_list, chosen, np.ones(len(chosen)), pooled, lam)[None, :]
        pi = np.ones(1)
        _, loglik = responsibilities(
            seat_log_likelihoods(X_list, chosen, seat_of, n_seats, betas), pi)
        return Archetypes(betas=betas, pi=pi, pooled=pooled, loglik=loglik,
                          lam=lam, n_iter=1, converged=True, n_seats=n_seats,
                          n_picks=len(chosen))

    best = None
    for restart in range(max(restarts, 1)):
        rng = np.random.default_rng([seed, restart])
        # Random RESPONSIBILITIES, then straight into an M-step, rather than
        # random betas. A random beta is a random point in a 29-dimensional
        # coefficient space where almost everything is a model that predicts
        # nothing; a random partition of the seats is a model of the same
        # family as the answer, and every one of its K fits is a real
        # conditional logit on real picks.
        R = rng.dirichlet(np.full(k, INIT_ALPHA), size=n_seats)
        pi = R.mean(axis=0)
        betas = np.vstack([_weighted_fit(X_list, chosen, R[seat_of, j], pooled, lam)
                           for j in range(k)])

        previous, converged, iteration = -np.inf, False, 0
        for iteration in range(1, max_iter + 1):
            seat_ll = seat_log_likelihoods(X_list, chosen, seat_of, n_seats, betas)
            R, loglik = responsibilities(seat_ll, pi)
            if loglik - previous <= tol:
                # EM cannot decrease this, so the difference is >= 0 up to
                # floating point and `<=` is a convergence test rather than a
                # divergence guard. The check runs BEFORE the M-step so that
                # the returned betas are the ones this likelihood was computed
                # from, not a step past them.
                converged = True
                break
            previous = loglik
            pi = R.mean(axis=0)
            # Warm-started from this class's own previous coefficients. The
            # objective is convex, so this changes the cost of the M-step and
            # not its answer.
            betas = np.vstack([_weighted_fit(X_list, chosen, R[seat_of, j], pooled,
                                             lam, start=betas[j])
                               for j in range(k)])

        seat_ll = seat_log_likelihoods(X_list, chosen, seat_of, n_seats, betas)
        _, loglik = responsibilities(seat_ll, pi)
        candidate = Archetypes(betas=betas, pi=pi, pooled=pooled, loglik=loglik,
                               lam=lam, n_iter=iteration, converged=converged,
                               restart=restart, n_seats=n_seats,
                               n_picks=len(chosen))
        if best is None or candidate.loglik > best.loglik:
            best = candidate
    return best.order_by_weight()


# ------------------------------------------------------- held-out likelihood

def seat_marginal_loglik(X_list, chosen, seats, model: Archetypes) -> dict:
    """The mixture's total log-probability for seats it was not fitted on.

    `sum_s log sum_k pi_k prod_t P_k(pick_t)` -- the class memberships
    integrated out, because a held-out seat's class is exactly what is not
    known about it.

    THIS IS THE SAME NUMBER THE LIVE PREDICTOR EARNS, which is why it is the
    right criterion for choosing K rather than a convenience. By the chain rule
    the marginal probability of a seat's whole pick sequence factorizes into
    the product over picks of P(pick_t | picks before t), and P(pick_t | picks
    before t) under this model is precisely `mixture_probs` evaluated at the
    posterior `update_posterior` has reached after t-1 picks. Scoring the
    sequence marginal and serving a Bayes-updated posterior are two views of
    one quantity, so a K that wins here is a K that wins in the draft room.

    Returned per-pick as well as in total: the totals are not comparable across
    folds of different sizes, and per-pick nats is the unit the ladder's
    log-loss column is already in.
    """
    seat_of, labels = seat_index(seats)
    seat_ll = seat_log_likelihoods(X_list, chosen, seat_of, len(labels), model.betas)
    _, total = responsibilities(seat_ll, model.pi)
    n = max(len(chosen), 1)
    return {"loglik": total, "per_pick": total / n, "n": len(chosen),
            "n_seats": len(labels)}


# --------------------------------------------------------- live inference

def seed_posterior(pi) -> np.ndarray:
    """A seat we have seen nothing from: the population weights themselves.

    This is what makes the cold start safe. At pick 1 every opponent's
    posterior IS `pi`, so `mixture_probs` returns the `pi`-weighted average of
    the K classes, which is the mixture's own best guess about a stranger and
    is at least as good as the pooled vector it replaces. Most opponent picks
    in a mock happen before a seat has revealed much, so the cold start is the
    common case rather than the edge one.
    """
    return np.asarray(pi, dtype=float).copy()


def update_posterior(posterior, X, chosen: int, betas) -> np.ndarray:
    """That seat's posterior after watching it make one more pick.

    Bayes, in log space for the same reason the E-step is: a run of picks
    multiplies out to a number float64 cannot hold, and this function is called
    once per pick per seat for a whole draft.

    Exactly equivalent to re-running the E-step on that seat's picks so far,
    which is not an accident -- it is the same posterior, computed
    incrementally so a live caller does not have to replay the seat's history
    on every pick.
    """
    betas = np.atleast_2d(betas)
    scores = np.asarray(X, dtype=float) @ betas.T          # (candidates, K)
    peak = scores.max(axis=0)
    log_p = scores[chosen] - (peak + np.log(np.exp(scores - peak).sum(axis=0)))
    with np.errstate(divide="ignore"):
        joint = np.log(np.asarray(posterior, dtype=float)) + log_p
    return np.exp(joint - _log_sum_exp(joint, axis=0))


def mixture_probs(X, betas, posterior) -> np.ndarray:
    """P(each candidate) for ONE pick: the posterior-weighted average.

    THE AVERAGE OF THE PROBABILITIES, not a softmax of the average of the
    scores. Those are different distributions and only the first is the
    model: the posterior predictive is `sum_k P(class k | history) *
    P(candidate | class k)`, and averaging inside the exponent would invent a
    (K+1)th drafter who is nobody.

    This is the answer to "who does this team take next" and it is the ONLY
    place averaging is right. A rollout must sample instead -- see
    `sample_class`.
    """
    betas = np.atleast_2d(betas)
    scores = np.asarray(X, dtype=float) @ betas.T          # (candidates, K)
    peak = scores.max(axis=0)
    weights = np.exp(scores - peak)
    probs = weights / weights.sum(axis=0)
    return probs @ np.asarray(posterior, dtype=float)


@dataclass
class SeatBeliefs:
    """Who each opponent seat might be, carried through a live draft.

    THE OBJECT THAT THREADS THROUGH `draft_sim`. `betas` and `pi` are the
    fitted model; `posterior` is per-seat state and is the only thing that
    changes as a draft goes on. A seat with no entry is a seat nothing has been
    observed from, and `of` returns `pi` for it -- so the cold start is not a
    special case anywhere, it is the absence of an entry.

    The key is whatever the caller identifies a seat by. `draft_sim` uses the
    draft SLOT, which is the only stable handle a mock draft has: the person in
    it is anonymous and will not be there tomorrow.

    Mutable on purpose, and rebuilt rather than mutated by the live path: the
    posteriors are derived entirely from the picks already made, so a session
    that recomputes them from `taken_order` on each refresh can never drift out
    of step with the board, which one that accumulated them incrementally
    across reconnects and re-syncs eventually would.
    """
    betas: np.ndarray
    pi: np.ndarray
    posterior: dict = None

    def __post_init__(self):
        self.betas = np.atleast_2d(self.betas)
        self.pi = np.asarray(self.pi, dtype=float)
        self.posterior = dict(self.posterior or {})

    @property
    def k(self) -> int:
        return len(self.betas)

    def of(self, key) -> np.ndarray:
        """That seat's posterior, seeded at `pi` for a seat never seen."""
        found = self.posterior.get(key)
        return seed_posterior(self.pi) if found is None else found

    def observe(self, key, X, chosen: int) -> None:
        """Fold one pick that seat made into its posterior."""
        self.posterior[key] = update_posterior(self.of(key), X, chosen, self.betas)

    def probs(self, key, X) -> np.ndarray:
        """The single-pick answer for that seat: the posterior-weighted average."""
        return mixture_probs(X, self.betas, self.of(key))

    def draw(self, keys, rng) -> dict:
        """One archetype per seat, for ONE rollout, as `{key: beta}`.

        Drawn once here rather than per pick because a drafter does not change
        who he is halfway through a draft. See `sample_class`. The caller holds
        the returned mapping for the whole simulated draft, so within a rollout
        a seat is exactly one of the K classes and the roster it builds is one
        an archetype would really have built.
        """
        return {key: self.betas[sample_class(self.of(key), rng)] for key in keys}


def sample_class(posterior, rng) -> int:
    """One archetype for one seat for one whole rollout.

    HELD FOR THE WHOLE DRAFT BY THE CALLER, which is the entire point.
    Averaging the classes inside a rollout would let a seat draft zero-RB on
    one pick and QB-early on the next -- a sequence no archetype would ever
    produce, and a roster no drafter would ever build, fed straight into
    `roster_value`. Sampling keeps a simulated opponent internally consistent,
    and the spread ACROSS rollouts is then a real representation of not knowing
    who that seat is. What that spread shows up in is measured, not assumed:
    see `draft_sim.survival`.
    """
    posterior = np.asarray(posterior, dtype=float)
    return int(rng.choice(len(posterior), p=posterior / posterior.sum()))


# ---------------------------------------------------- prequential scoring

def score_mixture(X_list, chosen, seats, model: Archetypes, groups=None,
                  buckets=None) -> dict:
    """Top-1, top-5 and log-loss, predicting each pick the way serving does.

    PREQUENTIAL, which is the only honest way to score a model whose whole
    claim is that it learns who a seat is. Each pick is predicted from that
    seat's posterior as it stood BEFORE the pick -- `pi` for a seat's first
    pick, Bayes-updated after each one. No pick is ever predicted using itself
    or anything later, so this is directly comparable to a fixed-vector rung
    scored on the same picks: both see exactly the same history.

    The observations must arrive in draft order within a seat, which
    `fit_prior.build_corpus_observations` guarantees (it sorts each draft's
    picks by `pick_no` and appends). Out of order, the posteriors would be
    updated from the future and every number here would be optimistic.

    Same report shape as `fit_prior.score` -- the same keys, including the raw
    `hits1`/`hits5`/`ll` counts and `by_draft`/`by_draft_n` -- so that
    `cluster_se`, `paired_se` and `restricted_top1` work on a mixture rung
    without a special case, and a ladder row for this model is comparable to
    every row above it by construction rather than by inspection.
    """
    posteriors = {}
    hits1 = hits5 = 0
    ll = 0.0
    per_group, per_bucket = {}, {}
    for i, (X, k, seat) in enumerate(zip(X_list, chosen, seats)):
        posterior = posteriors.get(seat)
        if posterior is None:
            posterior = seed_posterior(model.pi)
        probs = mixture_probs(X, model.betas, posterior)
        order = np.argsort(-probs)
        hit1 = int(order[0] == k)
        hit5 = int(k in order[:5])
        hits1 += hit1
        hits5 += hit5
        ll += float(np.log(max(probs[k], 1e-12)))
        posteriors[seat] = update_posterior(posterior, X, k, model.betas)
        if groups is not None:
            tally = per_group.setdefault(groups[i], [0, 0])
            tally[0] += hit1
            tally[1] += 1
        if buckets is not None:
            tally = per_bucket.setdefault(buckets[i], [0, 0, 0])
            tally[0] += hit1
            tally[1] += hit5
            tally[2] += 1
    n = len(chosen)
    if not n:
        return {"top1": 0.0, "top5": 0.0, "logloss": float("inf"), "n": 0,
                "hits1": 0, "hits5": 0, "ll": 0.0}
    report = {"top1": hits1 / n, "top5": hits5 / n, "logloss": -ll / n, "n": n,
              "hits1": hits1, "hits5": hits5, "ll": ll}
    if groups is not None:
        report["by_draft"] = {g: h / t for g, (h, t) in per_group.items()}
        report["by_draft_n"] = {g: t for g, (h, t) in per_group.items()}
    if buckets is not None:
        report["by_bucket"] = per_bucket
    return report


def describe_class(beta, pooled, feature_names, top: int = 6) -> str:
    """The largest deviations of one class from the pooled fit, in words.

    THE CLASSES COME OUT UNNAMED. EM produces coefficient vectors, not
    personas, and the only honest description of a class is a reading of what
    its coefficients actually say against the room average. Written against
    `pooled` rather than against zero because the absolute value of `reach` is
    -9-ish for everybody and says nothing about which class this is.
    """
    beta = np.asarray(beta, dtype=float)
    pooled = np.asarray(pooled, dtype=float)
    diff = beta - pooled
    order = np.argsort(-np.abs(diff))[:top]
    return ", ".join(f"{feature_names[i]} {diff[i]:+.2f}" for i in order)
