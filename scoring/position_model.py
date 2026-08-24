"""A POSITION classifier, and the diagnostic it exists to feed. NOT a model.

WHAT THIS MEASURES AND WHY IT IS A SEPARATE MODULE. The opponent model in
`scoring/draft_model.py` predicts WHICH PLAYER a seat takes next, as a
conditional logit over the whole available board. A measured fact about that
problem is that the choice very nearly factorizes: against ESPN's board only
about 23% of human picks are the single best name overall, but ~53% are the
best name AT THEIR POSITION (72.9% top-2, 82.6% top-3 within position). So

    P(player) = P(position | draft state) x P(player | position, board),

and the second factor is already ~53-60% with rules a page long. That makes
the ceiling of the whole nested-logit idea a product in which one factor is
known and the other is not:

    nested top-1  ~=  P(position correct)  x  (~0.55, a conservative
                                               within-position hit rate).

This module estimates the UNKNOWN factor -- how well the POSITION alone can be
predicted from the drafting team's state -- so that product can be written
down before anyone builds the full nested model. If position prediction tops
out near 55%, the nested ceiling is ~0.55 x 0.55 ~= 0.30, barely past the
23% the board alone already gives on the player. If it reaches ~70%, the
nested ceiling is ~0.42, a real gain worth the build. This is the number that
decides.

IT IS A 6-WAY CLASSIFIER OVER POSITIONS, NOT A LOGIT OVER PLAYERS. The target
is one of {QB, RB, WR, TE, K, DST} and the features are TEAM- and
CONTEXT-level (the same for every candidate at a pick), because the thing
being predicted is the position, not a player. That is the whole difference
from `feature_matrix`, whose columns vary candidate-by-candidate inside one
choice set. Reusing that here would be answering the second factor again.

NOTHING HERE SHIPS. It touches neither `scoring/human_prior.py` (the champion)
nor `scoring/draft_model.py` / `scoring/draft_sim.py`. It reads the corpus
through `pipeline.fit_prior` / `pipeline.score_ladder` -- the same replay, the
same human-only filter, the same leave-one-draft-out folds -- and reports a
number. The write-up is the deliverable; see
`pipeline/measure_position.py`.

WHERE THE STATE COMES FROM. Every feature is read off the `PickObservation`
that the shared replay already builds for each human pick:

  * `overall_pick`             -> round, pick-in-round, how far through
  * `roster`  (pos -> count)   -> what this seat already holds (its history)
  * `recent`  (last 5, room)   -> the room-wide positional run
  * `last_pick_at_pos`         -> when this SEAT last took each position
  * `pool`    (available set)  -> board ranks still on the board, per position,
                                  which give scarcity and best-available

so this module never re-reads the database. The one seat-level signal the
shared observation does not carry verbatim is the seat's own last-5 SEQUENCE:
`last_pick_at_pos` stores only the most recent pick per position, not a count,
so "the seat's positional run" is encoded here as a recency flag per position
(did this seat take it within the last `RUN_WINDOW` rounds) rather than a
windowed count. That is a faithful reading of what the shared replay records,
and it keeps this module on the one tested corpus reader rather than a second
copy of the replay loop.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from scoring.draft_model import RUN_WINDOW, _round_bucket

# The six classes, in a fixed order. Every position vector, every coefficient
# row and every confusion axis in this module and its measurement is indexed
# by THIS list, so it is written once and imported rather than re-spelled.
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
_POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}

# Positions a flex slot accepts. Used only to price "flex room" -- how many of
# a team's flex-eligible picks past its starters are still bench-bound -- so it
# is the running-back/receiver/tight-end set every league in this corpus uses.
_FLEX_ELIGIBLE = ("RB", "WR", "TE")

# A rank on the board this far past the current pick still counts toward "how
# many startable players remain at this position". Two rounds of the room is
# the window in which a position "about to run dry" pulls a pick forward, which
# is the effect the scarcity columns exist to see; wider than that and every
# position looks deep, narrower and only the very next pick registers.
_SCARCITY_ROUNDS = 2

# What a board rank reads as when a position has NO player left on the board at
# all. Larger than any real dense rank in an 8-team pool (~250 players), so the
# "best available here" column reads such a position as maximally picked-over
# rather than as a steal, and the scarcity count for it is simply zero.
_ABSENT_RANK = 400.0


# ------------------------------------------------------------- the features

# The feature order, written out so a coefficient can be read back by name and
# so a vector fitted in one order is never applied in another. `features_for`
# builds its vector in exactly this order and a test pins the two agreeing.
FEATURE_NAMES = (
    ["round_frac", "pick_in_round", "overall_frac"]
    + [f"held_{p}" for p in POSITIONS]
    + [f"starters_left_{p}" for p in POSITIONS]
    + ["flex_room", "bench_room"]
    + [f"room_run_{p}" for p in POSITIONS]
    + [f"seat_recent_{p}" for p in POSITIONS]
    + [f"scarcity_{p}" for p in POSITIONS]
    + [f"push_{p}" for p in POSITIONS]
)

# The feature groups, for the leave-one-group-out ablation. Each name above
# belongs to exactly one group; the ablation drops a whole group so the report
# reads as "what does knowing the roster buy" rather than as forty near-zero
# single-column deltas. The assertion below is the guard that the two stay in
# step -- a feature renamed in `FEATURE_NAMES` and not here would silently fall
# out of every group and be un-ablated.
FEATURE_GROUPS = {
    "timing": ["round_frac", "pick_in_round", "overall_frac"],
    "roster_held": [f"held_{p}" for p in POSITIONS],
    "starters_remaining": [f"starters_left_{p}" for p in POSITIONS]
    + ["flex_room", "bench_room"],
    "room_run": [f"room_run_{p}" for p in POSITIONS],
    "seat_run": [f"seat_recent_{p}" for p in POSITIONS],
    "scarcity": [f"scarcity_{p}" for p in POSITIONS],
    "best_available": [f"push_{p}" for p in POSITIONS],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURE_NAMES), (
    "the feature groups and FEATURE_NAMES disagree -- a column is in one and "
    "not the other, so an ablation would drop the wrong set")


def _board_rank(pool) -> np.ndarray:
    """The rank the room is reading, ESPN where it exists and ADP where not.

    The measured fact this whole module rests on is stated against ESPN's
    board, so `espn_rank` is the rank of record. About one candidate in ten
    carries no ESPN rank (a late-added player the board never scored), and for
    those the corpus's own dense ADP re-ranking (`market_rank`) is the honest
    fallback -- it is the same board the conditional logit reads, so no pick is
    scored against a market that did not exist. A pool with neither column is a
    bare fixture and reads as all-`_ABSENT_RANK`, which the callers below treat
    as "nothing known here".
    """
    espn = pool["espn_rank"].to_numpy(dtype=float) if "espn_rank" in pool \
        else np.full(len(pool), np.nan)
    adp = pool["market_rank"].to_numpy(dtype=float) if "market_rank" in pool \
        else np.full(len(pool), np.nan)
    rank = np.where(np.isfinite(espn), espn, adp)
    return np.where(np.isfinite(rank), rank, _ABSENT_RANK)


def features_for(obs, settings) -> np.ndarray:
    """One context-level feature vector for one pick. Order is `FEATURE_NAMES`.

    Everything here is the same for every candidate at this pick -- it
    describes the DRAFTING TEAM and the board, not a player -- because the
    target is the position, not a name. The chosen player's position is the
    label and is read separately by `target_of`; it is deliberately not a
    feature of itself.
    """
    teams = max(int(settings.teams), 1)
    rounds = max(int(settings.rounds), 1)
    starters = settings.starters
    roster = obs.roster or {}

    # --- timing. `overall_pick` is 1-indexed, so pick `teams` is the last of
    # round 1. All three are fractions of the draft so a coefficient means the
    # same thing in a 15-round league and a 16-round one.
    round_no = (obs.overall_pick - 1) // teams + 1
    pick_in_round = ((obs.overall_pick - 1) % teams) / teams
    timing = [round_no / rounds, pick_in_round, obs.overall_pick / (teams * rounds)]

    # --- what the seat already holds, and how many starter slots that leaves
    # open at each position. `held` IS the seat's prior position history; the
    # count normalized by rounds keeps it on the same scale as the timing
    # block. `starters_left` is the pressure the roster puts on each position.
    held = [roster.get(p, 0) / rounds for p in POSITIONS]
    starters_left = [max(starters.get(p, 0) - roster.get(p, 0), 0)
                     for p in POSITIONS]

    # Flex and bench room: how many flex-eligible picks past the starters are
    # still unspent, and how much bench is left. Both fall out of the roster
    # and the settings; both are constant across a single-shape corpus and earn
    # their place only where league shapes vary, but they are cheap and the
    # standardizer divides a constant column straight back out.
    total_held = sum(roster.values())
    overflow = sum(max(roster.get(p, 0) - starters.get(p, 0), 0)
                   for p in _FLEX_ELIGIBLE)
    flex_used = min(settings.flex_slots, overflow)
    flex_room = settings.flex_slots - flex_used
    bench_used = max(0, total_held - sum(starters.values()) - flex_used)
    bench_room = max(0, settings.bench - bench_used)

    # --- the room-wide run: what the last `RUN_WINDOW` picks in the whole room
    # were, as a share. `recent` is stored most-recent-first and already capped
    # at `RUN_WINDOW`; a position on a tear here is one the board is thinning
    # for everyone.
    recent = list(obs.recent)[:RUN_WINDOW]
    room_run = [recent.count(p) / RUN_WINDOW for p in POSITIONS]

    # --- the seat's own run, as a recency flag per position. `last_pick_at_pos`
    # holds the overall pick of this seat's most recent selection at each
    # position (and nothing older), so this reads as "did this seat take the
    # position within the last `RUN_WINDOW` of its own picks" -- one round is
    # one seat-pick in a snake, so a `RUN_WINDOW`-round lookback is a
    # `RUN_WINDOW`-pick lookback. See the module docstring for why a count is
    # not available here.
    last_at = obs.last_pick_at_pos or {}
    seat_recent = []
    for p in POSITIONS:
        last = last_at.get(p)
        if last is None:
            seat_recent.append(0.0)
        else:
            last_round = (last - 1) // teams + 1
            seat_recent.append(1.0 if (round_no - last_round) < RUN_WINDOW
                               else 0.0)

    # --- scarcity and best-available, both read straight off the available
    # pool (the replay has already removed everyone taken). `scarcity` counts
    # players still on the board at each position whose rank is within the next
    # `_SCARCITY_ROUNDS` of the room -- a position with few startable names left
    # pulls its pick forward. `push` is how far the best remaining player at a
    # position sits ahead of the current pick: positive means a value is
    # falling there, and a position with nobody left reads maximally negative.
    positions = obs.pool["position"].to_numpy()
    rank = _board_rank(obs.pool)
    threshold = obs.overall_pick + _SCARCITY_ROUNDS * teams
    scarcity, push = [], []
    for p in POSITIONS:
        at_pos = rank[positions == p]
        scarcity.append(float(np.sum(at_pos <= threshold)) / teams)
        best = float(at_pos.min()) if at_pos.size else _ABSENT_RANK
        push.append((obs.overall_pick - best) / teams)

    vec = (timing + held + starters_left + [flex_room, bench_room]
           + room_run + seat_recent + scarcity + push)
    return np.asarray(vec, dtype=float)


def target_of(obs) -> int:
    """The class index of the position the seat actually took at this pick.

    The chosen player's position is the label. Read off the pool row the
    conditional logit calls `chosen`, so this classifier and that logit agree
    to the row on what was picked.
    """
    return _POS_INDEX[obs.pool.iloc[obs.chosen]["position"]]


# ------------------------------------------------------- the classifier

def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, max-subtracted so a large score cannot overflow."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class MultinomialLogistic:
    """L2-penalized softmax regression over the six positions, numpy + L-BFGS.

    The natural baseline for a K-way classifier and the one the headline rests
    on. Features are standardized on the TRAINING rows and the same shift and
    scale are applied at predict time, so a fold never learns the test set's
    spread; a column with no variance in a fold divides by one rather than by
    zero. The intercept is not penalized and not standardized.

    Convex, so the fit is unique and the optimizer's start point cannot change
    the answer. `scipy.optimize.minimize` with an analytic gradient is the same
    machinery `draft_model.fit` uses, so this takes on no dependency the project
    does not already carry.
    """

    def __init__(self, lam: float = 1.0):
        # Penalty strength on the standardized weights, normalized by the row
        # count inside the objective so the same `lam` means the same thing on
        # a fold of 3,000 picks and a fold of 5,000.
        self.lam = float(lam)
        self.mean_ = None
        self.std_ = None
        self.W_ = None   # (K, d)
        self.b_ = None   # (K,)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialLogistic":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        k = len(POSITIONS)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # A constant column has std 0; dividing by one leaves it constant and
        # its coefficient is then pinned by the penalty toward zero, which is
        # the right answer for a column that says nothing on this fold.
        self.std_ = np.where(std > 1e-12, std, 1.0)
        Xs = self._standardize(X)

        onehot = np.zeros((n, k))
        onehot[np.arange(n), y] = 1.0

        def objective(theta: np.ndarray):
            W = theta[: k * d].reshape(k, d)
            b = theta[k * d:]
            probs = _softmax_rows(Xs @ W.T + b)
            # Mean negative log-likelihood plus a mean-scaled L2 on W only.
            ll = np.sum(np.log(np.clip(probs[np.arange(n), y], 1e-12, None)))
            loss = -ll / n + 0.5 * self.lam * np.sum(W * W) / n
            resid = (probs - onehot) / n            # (n, k)
            gW = resid.T @ Xs + self.lam * W / n     # (k, d)
            gb = resid.sum(axis=0)                    # (k,)
            return loss, np.concatenate([gW.ravel(), gb])

        theta0 = np.zeros(k * d + k)
        result = minimize(objective, theta0, jac=True, method="L-BFGS-B")
        self.W_ = result.x[: k * d].reshape(k, d)
        self.b_ = result.x[k * d:]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self._standardize(np.asarray(X, dtype=float))
        return _softmax_rows(Xs @ self.W_.T + self.b_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def standardized_coefficients(self) -> np.ndarray:
        """The (K, d) weight matrix on standardized features.

        Standardized, so a coefficient's magnitude is comparable across columns
        of different natural scale -- which is the only thing that makes "what
        predicts the position" a readable table rather than a units contest.
        """
        return self.W_


class MLPClassifier:
    """A one-hidden-layer softmax net, numpy + L-BFGS, as a headroom check.

    Its whole job is to answer one question the linear model cannot: is the
    ceiling being set by the FEATURES or by the linearity? If a small net with
    the same inputs cannot beat the multinomial logit by more than its standard
    error, the position is about as predictable as it is going to get from this
    state and a fancier opponent model buys nothing. If it opens a real gap,
    the linear number understates the ceiling and that is worth knowing before
    the build.

    Non-convex, so the seed matters and is fixed; `tanh` hidden units and an
    L2 on both weight matrices. Bounded L-BFGS iterations keep a fold cheap.
    Deliberately small -- this is a probe, not a model to ship.
    """

    def __init__(self, hidden: int = 12, lam: float = 1.0,
                 seed: int = 0, maxiter: int = 300):
        self.hidden = int(hidden)
        self.lam = float(lam)
        self.seed = int(seed)
        self.maxiter = int(maxiter)
        self.mean_ = None
        self.std_ = None
        self.params_ = None
        self._shapes = None

    def _standardize(self, X):
        return (X - self.mean_) / self.std_

    def _unpack(self, theta, d, h, k):
        i = 0
        W1 = theta[i:i + d * h].reshape(d, h); i += d * h
        b1 = theta[i:i + h]; i += h
        W2 = theta[i:i + h * k].reshape(h, k); i += h * k
        b2 = theta[i:i + k]
        return W1, b1, W2, b2

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        h, k = self.hidden, len(POSITIONS)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)
        Xs = self._standardize(X)
        onehot = np.zeros((n, k))
        onehot[np.arange(n), y] = 1.0

        rng = np.random.default_rng(self.seed)
        # Small random init breaks the symmetry a zero init would leave in the
        # hidden layer; scaled by 1/sqrt(fan-in) so pre-activations start near
        # the linear part of tanh.
        theta0 = np.concatenate([
            (rng.standard_normal(d * h) / np.sqrt(d)),
            np.zeros(h),
            (rng.standard_normal(h * k) / np.sqrt(h)),
            np.zeros(k)])

        def objective(theta):
            W1, b1, W2, b2 = self._unpack(theta, d, h, k)
            H = np.tanh(Xs @ W1 + b1)                 # (n, h)
            probs = _softmax_rows(H @ W2 + b2)         # (n, k)
            ll = np.sum(np.log(np.clip(probs[np.arange(n), y], 1e-12, None)))
            reg = 0.5 * self.lam * (np.sum(W1 * W1) + np.sum(W2 * W2)) / n
            loss = -ll / n + reg
            dZ = (probs - onehot) / n                  # (n, k)
            gW2 = H.T @ dZ + self.lam * W2 / n
            gb2 = dZ.sum(axis=0)
            dH = (dZ @ W2.T) * (1.0 - H * H)           # (n, h)
            gW1 = Xs.T @ dH + self.lam * W1 / n
            gb1 = dH.sum(axis=0)
            return loss, np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])

        result = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                          options={"maxiter": self.maxiter})
        self.params_ = result.x
        self._shapes = (d, h, k)
        return self

    def predict_proba(self, X):
        d, h, k = self._shapes
        W1, b1, W2, b2 = self._unpack(self.params_, d, h, k)
        Xs = self._standardize(np.asarray(X, dtype=float))
        H = np.tanh(Xs @ W1 + b1)
        return _softmax_rows(H @ W2 + b2)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------- baselines
#
# Three predictors that need no features beyond round and roster, in ascending
# order of what they know. Each is the floor the multinomial logit has to
# clear, and each is fitted on the TRAINING drafts of a fold and scored on the
# held-out one exactly like the model, so the comparison is on equal footing.
# A model that cannot beat "look up the majority position for this round and
# roster" has learned nothing the corpus did not already say in a table.


def _majority(labels) -> int:
    """The most common class index among `labels`; WR's index on an empty set.

    Ties break toward the lower class index, which is deterministic; the empty
    fallback is WR because it is the corpus-wide plurality and so the least
    wrong constant guess.
    """
    if len(labels) == 0:
        return _POS_INDEX["WR"]
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=len(POSITIONS))
    return int(np.argmax(counts))


def roster_signature(roster: dict) -> tuple:
    """A coarse key for the round+roster lookup: capped counts at each position.

    Capped so the key has support -- an exact roster is nearly unique per pick
    and would make every lookup a miss, backing off to the round alone. The
    caps are where each position stops changing the answer: a fourth running
    back and a fifth say the same thing about need, one kicker or defense is
    the whole story for a streamed position.
    """
    caps = {"QB": 2, "RB": 4, "WR": 4, "TE": 2, "K": 1, "DST": 1}
    return tuple(min(roster.get(p, 0), caps[p]) for p in POSITIONS)


class AlwaysWR:
    """Predict the corpus plurality (WR) for every pick. The crudest floor."""

    def fit(self, rounds, rosters, y):
        return self

    def predict(self, rounds, rosters) -> np.ndarray:
        return np.full(len(rounds), _POS_INDEX["WR"], dtype=int)


class MajorityPerRound:
    """Predict the training set's most common position for this round number.

    Round is the single most informative scalar about the position -- kickers
    and defenses live in the last two rounds, quarterbacks bunch early and
    late -- so "the majority position for round N" is the floor that a model
    claiming to use the draft's shape has to beat.
    """

    def fit(self, rounds, rosters, y):
        rounds = np.asarray(rounds, dtype=int)
        y = np.asarray(y, dtype=int)
        self.table_ = {r: _majority(y[rounds == r]) for r in np.unique(rounds)}
        self.default_ = _majority(y)
        return self

    def predict(self, rounds, rosters) -> np.ndarray:
        return np.array([self.table_.get(int(r), self.default_)
                         for r in rounds], dtype=int)


class RoundRosterLookup:
    """Predict the majority position for this (round, coarse roster), backed off.

    The strongest thing that is still a lookup rather than a model: a table
    keyed on the round AND what the seat holds, so it can say "round 9 with two
    running backs and no tight end takes a tight end" where round alone cannot.
    Backs off to the round's majority when a (round, roster) pair is unseen in
    training, then to the global majority, so it never fails to answer.
    """

    def fit(self, rounds, rosters, y):
        rounds = np.asarray(rounds, dtype=int)
        y = np.asarray(y, dtype=int)
        pair, per_round = {}, {}
        keys = [(int(r), roster_signature(rs)) for r, rs in zip(rounds, rosters)]
        for key in set(keys):
            mask = np.array([k == key for k in keys])
            pair[key] = _majority(y[mask])
        for r in np.unique(rounds):
            per_round[int(r)] = _majority(y[rounds == r])
        self.pair_, self.per_round_ = pair, per_round
        self.default_ = _majority(y)
        return self

    def predict(self, rounds, rosters) -> np.ndarray:
        out = []
        for r, rs in zip(rounds, rosters):
            key = (int(r), roster_signature(rs))
            if key in self.pair_:
                out.append(self.pair_[key])
            else:
                out.append(self.per_round_.get(int(r), self.default_))
        return np.array(out, dtype=int)


# ---------------------------------------------------- the design, from a replay

class PositionDesign:
    """Everything a fold needs, extracted from one `CorpusObservations`.

    Built once from the shared replay so that the classifier, the baselines and
    the by-round table all read the same picks in the same order. Holds the
    feature matrix `X`, the labels `y`, the fold key `groups` (the draft id),
    the `rounds` and `rosters` the baselines key on, and the round `buckets` the
    by-round report sums into.
    """

    def __init__(self, X, y, groups, rounds, rosters, buckets):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)
        self.groups = list(groups)
        self.rounds = np.asarray(rounds, dtype=int)
        self.rosters = list(rosters)
        self.buckets = list(buckets)


def design_from_observations(corpus_obs) -> PositionDesign:
    """Turn the shared `CorpusObservations` into a `PositionDesign`.

    `corpus_obs` is what `score_ladder.human_observations` (or
    `fit_prior.build_corpus_observations`) returns: one `PickObservation` per
    KEPT pick, its draft id alongside, and that draft's `settings`. This walks
    them once, building the context feature vector and reading the label off
    each, and derives the round and round bucket from the same
    `overall_pick`/`teams` the conditional logit uses so the buckets line up
    with `draft_model`'s own.
    """
    X, y, groups, rounds, rosters, buckets = [], [], [], [], [], []
    for obs, draft_id in zip(corpus_obs.observations, corpus_obs.draft_ids):
        settings = corpus_obs.settings[draft_id]
        teams = max(int(settings.teams), 1)
        X.append(features_for(obs, settings))
        y.append(target_of(obs))
        groups.append(draft_id)
        rounds.append((obs.overall_pick - 1) // teams + 1)
        rosters.append(dict(obs.roster or {}))
        buckets.append(_round_bucket(obs.overall_pick, teams))
    return PositionDesign(X, y, groups, rounds, rosters, buckets)
