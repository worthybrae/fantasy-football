"""A NESTED (position-then-player) opponent model, and nothing that ships.

WHAT THIS IS. The champion in `scoring/human_prior.py` is a FLAT conditional
logit: one softmax over every available player at once, 0.2595 held-out top-1
on human picks. A measured fact about the same picks (see
`docs/superpowers/specs/2026-08-23-best-opponent-model-design.md` section 3b
and `scoring/position_model.py`) is that the choice very nearly FACTORIZES:

    P(player) = P(position | team state) x P(player | position, board).

The position half is ~0.53 held-out on its own (the committed
`position_model.MultinomialLogistic`, which beat an MLP and is NOT to be
swapped for something fancier), and the within-position half is ~0.53-0.60
with a page of rules. A ceiling measurement put the nested top-1 at ~0.29-0.32
-- past the flat champion's 0.2595 -- and this module realizes that
decomposition so `pipeline/measure_nested.py` can test whether it actually
beats the champion. It is a DIAGNOSTIC BUILD. It imports neither the champion
nor `draft_sim`; it touches no production serving path; wiring it into
`cold_start_fits`/`_run_draft`/`survival` is a separate integration task and is
deliberately out of scope here (five call sites, fit/serve parity risk).

HOW THE TWO FACTORS ARE FIT: SEPARATELY, AND WHY THAT IS EXACT, NOT A
SHORTCUT. The nested-logit log-likelihood of one pick is

    log P(pick) = log P(position*) + log P(player* | position*),

and we fix the inclusive-value (dissimilarity) parameter at lambda = 1, which
is precisely the "the choice factorizes" case the spec's ceiling arithmetic
assumes -- P(player) = P(position) x P(player|position) with no log-sum
coupling term. Under lambda = 1 the position factor's parameters and the
within-position factor's parameters appear in DISJOINT terms of that sum: the
position factor is scored on TEAM/CONTEXT features (the same for every
candidate, so they cancel out of any within-position softmax), and the
within-position factor is scored on PER-CANDIDATE features that are constant
across a position and therefore cancel out of the position softmax. Two sums
sharing no parameter are maximized by maximizing each independently. So
fitting the position classifier by its own MLE and the within-position
conditional logit by its own MLE jointly maximizes the nested likelihood.
There is nothing to gain from a joint optimizer, and a joint optimizer would
only reintroduce lambda, which we are deliberately holding at 1 to match the
ceiling the build is testing. If a future rung wants a free lambda it is a new
model, not a re-fit of this one.

WHICH FEATURES GO IN THE WITHIN-POSITION FACTOR. `draft_model.feature_matrix`
already builds all 34 candidate columns; the within-position logit reuses that
matrix, RESTRICTED to the same-position candidates, and keeps only the columns
that actually VARY between two players at the same position. A column that is
constant across a position (the position dummies, `need`, `run`,
`held_at_pos`, ...) contributes exactly zero to every within-position softmax
gradient -- it is not a weak within-position feature, it is no feature at all,
the same reason `draft_model` crosses its roster-shape columns with the
candidate's position -- so it is dropped from the within factor rather than
fitted to a coefficient the data can never move. See
`WITHIN_POSITION_FEATURES`. The position factor's own features are the
context-level ones in `position_model.features_for`; the two feature sets are
disjoint by construction, which is what makes the separate fit exact above.

NUMPY + SCIPY ONLY, reusing the two committed fitters unchanged
(`position_model.MultinomialLogistic` for the position factor,
`draft_model.fit` for the within-position conditional logit) so there is no
third copy of either softmax to drift out of step.
"""
from __future__ import annotations

import numpy as np

from scoring import draft_model as dm
from scoring import position_model as pm
from scoring.position_model import POSITIONS, _POS_INDEX

# ------------------------------------------------------- within-position features
#
# The columns of `draft_model.feature_matrix` that DIFFER between two players
# at the same position, and so carry information the within-position softmax
# can use. Everything omitted is constant across a position at a given pick
# (the five position dummies and the two `*_early` interactions; `need`,
# `run`, and the four roster-shape columns, all indexed by the candidate's own
# position; `slots_left_at_pos`), and a constant column drops out of the
# within-position softmax entirely -- its gradient is identically zero -- so it
# is excluded rather than fitted to a coefficient the choice sets cannot
# identify. The list is spelled by NAME and turned into indices against
# `draft_model.FEATURE_NAMES` at import, so a column reordered or renamed in
# the feature matrix cannot silently move which coefficient reads which column;
# the assertion below is the guard.
WITHIN_POSITION_FEATURES = [
    "reach", "fall",                 # where the candidate sits vs the market board
    "age", "no_track_record", "hype", "trend",   # the four newer per-player signals
    "usage", "efficiency", "played_share", "peak_gap",   # stat profile, centred in pos
    "dropoff_at_pos", "vor", "durability", "proj_change", "last_of_tier",  # pool signals
    "espn_reach", "espn_fall", "espn_list_pos",  # where the candidate sits vs ESPN's list
    "board_disagreement", "espn_proj_dropoff",
]

# Indices into a `feature_matrix` row, in `WITHIN_POSITION_FEATURES` order. A
# beta fitted on these columns is only meaningful applied to the same columns
# in the same order, exactly as `fit_prior.feature_indices` insists for the
# flat model.
WITHIN_IDX = [dm.FEATURE_NAMES.index(name) for name in WITHIN_POSITION_FEATURES]
assert all(dm.FEATURE_NAMES[j] == n for j, n in zip(WITHIN_IDX, WITHIN_POSITION_FEATURES)), (
    "WITHIN_POSITION_FEATURES and draft_model.FEATURE_NAMES disagree on which "
    "column is which -- a feature was reordered or renamed and the within-"
    "position logit would read the wrong column")

# Every name we keep must be a real feature-matrix column, and none of the
# kept columns may be one of the position-constant columns we mean to drop --
# a mistake either way would put a dead column into the fit or leave a live one
# out. The dropped set is spelled here so the intent is checkable, not implied.
_POSITION_CONSTANT = (
    [f"pos_{p}" for p in dm._POSITION_DUMMIES]
    + ["qb_early", "te_early", "need", "run"]
    + dm._ROSTER_SHAPE_FEATURES + ["slots_left_at_pos"])
assert set(WITHIN_POSITION_FEATURES).isdisjoint(_POSITION_CONSTANT), (
    "a column that is constant within a position leaked into the within-"
    "position feature list; it would be fitted to an unidentifiable zero")
assert set(WITHIN_POSITION_FEATURES) | set(_POSITION_CONSTANT) == set(dm.FEATURE_NAMES), (
    "the within-position kept set and the position-constant dropped set do not "
    "together account for every feature-matrix column -- a new column was "
    "added and assigned to neither, so it is silently un-modelled here")


# ------------------------------------------------------------- the per-pick design

class NestedDesign:
    """Everything both factors need for every kept pick, built once.

    Mirrors `position_model.PositionDesign` (which carries the POSITION
    factor's inputs) and adds the WITHIN-position factor's: for each pick, the
    same-order candidate feature matrix sliced to `WITHIN_IDX`, the class index
    of every candidate's position, and which candidate was chosen. Built from
    the one shared replay so this model reads exactly the picks, choice sets
    and folds the champion and the position ceiling were measured on.

    Fields, all indexed by pick i (0..n-1):
      pos_X[i]      context feature vector for the position factor (d_pos,)
      y[i]          class index of the chosen player's position (the label of
                    the position factor AND the nest the within factor scores)
      within_X[i]   (n_cand_i, len(WITHIN_IDX)) candidate features
      cand_pos[i]   (n_cand_i,) class index of each candidate's position
      chosen[i]     index of the picked candidate within the choice set
      groups[i]     draft id -- the leave-one-draft-out fold key
      rounds[i]     1-indexed round, for reporting
      buckets[i]    early/mid/late, `draft_model._round_bucket`, so this table
                    lines up with the champion's own by-round split
    """

    def __init__(self, pos_X, y, within_X, cand_pos, chosen, groups,
                 rounds, buckets):
        self.pos_X = np.asarray(pos_X, dtype=float)
        self.y = np.asarray(y, dtype=int)
        self.within_X = list(within_X)
        self.cand_pos = list(cand_pos)
        self.chosen = list(chosen)
        self.groups = list(groups)
        self.rounds = np.asarray(rounds, dtype=int)
        self.buckets = list(buckets)

    def __len__(self):
        return len(self.y)


def design_from_observations(corpus_obs) -> NestedDesign:
    """Turn the shared `CorpusObservations` into a `NestedDesign`.

    `corpus_obs` is what `score_ladder.human_observations` returns: one
    `PickObservation` per KEPT human pick, its draft id, and that draft's
    settings. This walks them once. For each pick it builds the POSITION
    factor's context vector with `position_model.features_for` (unchanged, so
    the position factor is byte-for-byte the one that won 0.53) and the WITHIN
    factor's candidate matrix with `draft_model.feature_matrix` sliced to the
    within-position columns (unchanged, so it is the exact matrix the champion
    and `draft_sim` build). The chosen index, the per-candidate positions, the
    label, the round and the bucket are all read off the same observation, so
    the two factors and the by-round report never disagree about what was
    picked or when.
    """
    pos_X, y, within_X, cand_pos, chosen = [], [], [], [], []
    groups, rounds, buckets = [], [], []
    for obs, draft_id in zip(corpus_obs.observations, corpus_obs.draft_ids):
        settings = corpus_obs.settings[draft_id]
        teams = max(int(settings.teams), 1)

        pos_X.append(pm.features_for(obs, settings))
        y.append(pm.target_of(obs))

        Xfull = dm.feature_matrix(obs, settings)
        within_X.append(Xfull[:, WITHIN_IDX])
        cand_pos.append(np.array(
            [_POS_INDEX[p] for p in obs.pool["position"].to_numpy()], dtype=int))
        chosen.append(int(obs.chosen))

        groups.append(draft_id)
        rounds.append((obs.overall_pick - 1) // teams + 1)
        buckets.append(dm._round_bucket(obs.overall_pick, teams))
    return NestedDesign(pos_X, y, within_X, cand_pos, chosen, groups,
                        rounds, buckets)


# ------------------------------------------------------------------ the model

def _softmax(scores: np.ndarray) -> np.ndarray:
    """Max-subtracted softmax over one choice set. Same shape as
    `draft_model._softmax`; duplicated here only to keep this diagnostic module
    from importing a private helper, and asserted against it in the tests."""
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class NestedModel:
    """P(player) = P(position | state) x P(player | position, board).

    Two fitted factors, held together at prediction time by re-normalizing
    over the players that are actually AVAILABLE at this pick:

      * `position_` is a `position_model.MultinomialLogistic` over the six
        positions, fit on the context features. It is the committed classifier
        that won 0.53 held-out; this model does not modify it.

      * `within_` is a single `draft_model.fit` conditional-logit vector,
        SHARED across positions, fit by pooling every pick's same-position
        choice set. One shared within-nest vector (rather than one per
        position) is the standard nested-logit form and the robust choice on
        ~4k human picks split six ways; a QB reach and a WR reach are priced by
        the same coefficient on `reach`, which is what "the best player at the
        position" means across positions.

    Both factors are fit by plain maximum likelihood -- the position factor
    with the L2 the classifier already carries, the within factor with no
    ridge and no prior, exactly as `fit_prior.leave_one_draft_out` fits the
    champion it is compared against, so neither side gets a regularization
    head start the other lacks.
    """

    def __init__(self, position_lam: float = 1.0):
        # Passed straight through to the committed classifier; its default is
        # 1.0 and the ceiling measurement used the default, so the position
        # factor here is the one that was measured at 0.53 unless a caller
        # deliberately changes it.
        self.position_lam = float(position_lam)
        self.position_ = None
        self.within_ = None

    def fit(self, design: NestedDesign, idx=None) -> "NestedModel":
        """Fit both factors on the picks named by `idx` (all picks if None).

        `idx` is how leave-one-draft-out passes the TRAINING picks: the caller
        selects the rows whose draft is not held out and fits on those, then
        scores the held-out draft with `predict_pick`. Fitting on an index set
        rather than a pre-sliced design keeps the heavy candidate matrices from
        being copied once per fold.
        """
        idx = np.arange(len(design)) if idx is None else np.asarray(idx, dtype=int)

        # --- position factor: the committed 6-way classifier, unchanged.
        self.position_ = pm.MultinomialLogistic(lam=self.position_lam).fit(
            design.pos_X[idx], design.y[idx])

        # --- within-position factor: one conditional logit over same-position
        # choice sets, pooled across positions. For each training pick, keep
        # the candidates at the CHOSEN player's position and record which of
        # them was taken; a singleton set (only one player left at that
        # position) is dropped because a softmax over one element is 1 for any
        # beta and so contributes nothing to the fit -- keeping it would only
        # slow the optimizer.
        Xw_list, chosen_within = [], []
        for i in idx:
            pos = design.y[i]                       # the chosen nest
            mask = design.cand_pos[i] == pos
            if mask.sum() < 2:
                continue
            Xw_list.append(design.within_X[i][mask])
            # index of the chosen candidate within the masked subset: the count
            # of same-position candidates ahead of it in board order.
            chosen_within.append(int(np.count_nonzero(mask[:design.chosen[i]])))
        # `draft_model.fit` with no prior and lam=0 is the plain pooled MLE,
        # the same estimator the champion's refit uses. On the rare empty fold
        # it returns a zero vector, which makes the within factor uniform --
        # harmless, since such a fold has no multi-candidate positions to score.
        self.within_ = dm.fit(Xw_list, chosen_within)
        return self

    def predict_from(self, pos_x, within_X, cand_pos) -> np.ndarray:
        """The nested probability of every candidate, from raw factor inputs.

        THE ONE PLACE the nested reconstruction is written, so the FIT
        (`predict_pick`, over a `NestedDesign`) and the SERVE
        (`draft_sim`, over a live `SimPool`) cannot compute it two different
        ways. Given the position factor's context vector `pos_x` (length
        `position_model.FEATURE_NAMES`), the within factor's candidate matrix
        `within_X` (`len(WITHIN_IDX)` columns, in board order) and each
        candidate's position class index `cand_pos`, it returns one vector over
        the choice set, in board order, summing to 1.

        For each position present on the board the within factor gives a
        distribution over that position's candidates and the position factor
        gives the mass for the position; their product is the joint,
        re-normalized by the position mass that actually has somewhere to land
        -- the standard nested-logit reconstruction restricted to the available
        set. A candidate whose position is not one of the six (class index -1,
        which a real pool never produces) matches no nest and takes zero mass.
        """
        pos_probs = self.position_.predict_proba(
            np.asarray(pos_x, dtype=float).reshape(1, -1))[0]     # (6,)
        cand_pos = np.asarray(cand_pos)
        within_X = np.asarray(within_X, dtype=float)
        joint = np.zeros(len(cand_pos), dtype=float)
        available_mass = 0.0
        for q in range(len(POSITIONS)):
            mask = cand_pos == q
            if not mask.any():
                continue
            within = _softmax(within_X[mask] @ self.within_)  # sums to 1 over pos
            joint[mask] = pos_probs[q] * within
            available_mass += pos_probs[q]
        # `available_mass` is >0 as long as any candidate remains, which is
        # always true for a real pick; guard the empty pool anyway rather than
        # divide by zero.
        if available_mass > 0:
            joint /= available_mass
        return joint

    def predict_pick(self, design: NestedDesign, i: int) -> np.ndarray:
        """The nested probability of every AVAILABLE candidate at pick i.

        A thin wrapper over `predict_from` that reads the three factor inputs
        off the fitted design. Sharing that one implementation with the serving
        path is what makes fit/serve parity a property of the code rather than
        of a test: the fit and the live room run the same arithmetic on the
        same feature definitions.
        """
        return self.predict_from(design.pos_X[i:i + 1], design.within_X[i],
                                 design.cand_pos[i])

    def predict_serve(self, X, pos_x, cand_pos, overall_pick=None,
                      teams=None) -> np.ndarray:
        """Serve from the FULL live feature matrix `X`.

        `X` is `draft_sim._live_features` over the available board -- every
        `draft_model.FEATURE_NAMES` column, in board order -- which is exactly
        the matrix `draft_model.feature_matrix` builds and which the tests pin
        equal. This slices it to `WITHIN_IDX` for the within factor and defers
        to `predict_from`, so the served vector equals the measured one.

        `overall_pick` and `teams` are accepted and IGNORED here: the nested
        model has no round-dependent behaviour. They are in the signature so
        the nested and the hybrid (`hybrid_model.HybridModel`) serve through
        one interface -- `draft_sim._nested_scores` calls `predict_serve` and
        does not know or care which of the two it holds.
        """
        X = np.asarray(X, dtype=float)
        return self.predict_from(pos_x, X[:, WITHIN_IDX], cand_pos)


def cold_start_nested() -> "NestedModel":
    """Reconstruct the pre-fit cold-start `NestedModel` from `nested_prior`.

    The nested analogue of binding `draft_model.COLD_START_PRIOR` to the flat
    prior: a `NestedModel` whose two factors carry the coefficients fitted on
    the whole human-pick corpus, built with NO corpus access. The position
    factor is rebuilt by setting the standardization and softmax weights the
    generator serialized straight onto a `MultinomialLogistic`; the within
    factor is the stored conditional-logit vector.

    Both feature-order lists are checked against the LIVE definitions, the same
    guard `draft_model` runs on `COLD_START_PRIOR`: a column reordered or
    renamed since the artifact was written trips an assert here rather than
    reading the wrong coefficient on every pick.
    """
    from scoring import nested_prior as npr

    assert list(npr.POSITION_FEATURES) == list(pm.FEATURE_NAMES), (
        "scoring/nested_prior.py was fitted against a different position "
        "feature order than position_model.FEATURE_NAMES holds now; "
        "regenerate it (scratchpad/fit_nested_prior.py)")
    assert list(npr.WITHIN_FEATURES) == list(WITHIN_POSITION_FEATURES), (
        "scoring/nested_prior.py was fitted against a different within-position "
        "feature order than nested_model.WITHIN_POSITION_FEATURES holds now; "
        "regenerate it (scratchpad/fit_nested_prior.py)")

    model = NestedModel(position_lam=npr.POSITION_LAM)
    clf = pm.MultinomialLogistic(lam=npr.POSITION_LAM)
    clf.mean_ = np.asarray(npr.POSITION_MEAN, dtype=float)
    clf.std_ = np.asarray(npr.POSITION_STD, dtype=float)
    clf.W_ = np.asarray(npr.POSITION_W, dtype=float)
    clf.b_ = np.asarray(npr.POSITION_B, dtype=float)
    model.position_ = clf
    model.within_ = np.asarray(npr.WITHIN_BETA, dtype=float)
    return model


# --------------------------------------------------- the position factor, SERVED
#
# The position factor is fitted on `position_model.features_for(obs, settings)`,
# a context vector read off a `PickObservation` whose `pool` is a pandas frame.
# At SERVE time -- inside `draft_sim._run_draft`/`survival` rollouts -- there is
# no such frame; there is a `SimPool` of numpy arrays and an `available` index.
# `position_features_live` builds the SAME vector from those arrays, the exact
# analogue of how `draft_sim._live_features` mirrors `draft_model.feature_matrix`
# for the within factor. It is written line-for-line against `features_for` and
# reads every scalar constant (`_SCARCITY_ROUNDS`, `_ABSENT_RANK`,
# `_FLEX_ELIGIBLE`, `RUN_WINDOW`) off `position_model` rather than re-spelling
# them, so the two cannot drift; `tests/test_draft_sim.py` asserts the two agree
# on a shared fixture, the same guard the within factor already carries.


def cand_pos_codes(positions) -> np.ndarray:
    """Each candidate's position as a class index into `POSITIONS`, -1 if none.

    The label the within factor groups on and the position factor's classes,
    read off the SimPool's `position` slice exactly as `design_from_observations`
    reads it off `obs.pool["position"]`.
    """
    return np.array([_POS_INDEX.get(p, -1) for p in positions], dtype=int)


def _board_rank_live(pool, available) -> np.ndarray:
    """`position_model._board_rank` over a `SimPool` slice: ESPN rank where it
    exists, dense market rank where not, `_ABSENT_RANK` where neither."""
    n = len(available)
    espn = getattr(pool, "espn_rank", None)
    espn = (np.full(n, np.nan) if espn is None
            else np.asarray(espn, dtype=float)[available])
    adp = np.asarray(pool.market_rank, dtype=float)[available]
    rank = np.where(np.isfinite(espn), espn, adp)
    return np.where(np.isfinite(rank), rank, pm._ABSENT_RANK)


def position_features_live(pool, available, overall_pick, roster, recent,
                           settings, last_pick_at_pos=None) -> np.ndarray:
    """`position_model.features_for`, SERVED from a `SimPool` + `available`.

    One context vector, in `position_model.FEATURE_NAMES` order, byte-for-byte
    the vector `features_for` builds for the same state. Every candidate at this
    pick shares it -- it describes the drafting team and the board, not a
    player. See the block comment above for why this mirror exists.
    """
    teams = max(int(settings.teams), 1)
    rounds = max(int(settings.rounds), 1)
    starters = settings.starters
    roster = roster or {}

    round_no = (overall_pick - 1) // teams + 1
    pick_in_round = ((overall_pick - 1) % teams) / teams
    timing = [round_no / rounds, pick_in_round, overall_pick / (teams * rounds)]

    held = [roster.get(p, 0) / rounds for p in POSITIONS]
    starters_left = [max(starters.get(p, 0) - roster.get(p, 0), 0)
                     for p in POSITIONS]

    total_held = sum(roster.values())
    overflow = sum(max(roster.get(p, 0) - starters.get(p, 0), 0)
                   for p in pm._FLEX_ELIGIBLE)
    flex_used = min(settings.flex_slots, overflow)
    flex_room = settings.flex_slots - flex_used
    bench_used = max(0, total_held - sum(starters.values()) - flex_used)
    bench_room = max(0, settings.bench - bench_used)

    window = list(recent)[:pm.RUN_WINDOW]
    room_run = [window.count(p) / pm.RUN_WINDOW for p in POSITIONS]

    last_at = last_pick_at_pos or {}
    seat_recent = []
    for p in POSITIONS:
        last = last_at.get(p)
        if last is None:
            seat_recent.append(0.0)
        else:
            last_round = (last - 1) // teams + 1
            seat_recent.append(1.0 if (round_no - last_round) < pm.RUN_WINDOW
                               else 0.0)

    positions = pool.position[available]
    rank = _board_rank_live(pool, available)
    threshold = overall_pick + pm._SCARCITY_ROUNDS * teams
    scarcity, push = [], []
    for p in POSITIONS:
        at_pos = rank[positions == p]
        scarcity.append(float(np.sum(at_pos <= threshold)) / teams)
        best = float(at_pos.min()) if at_pos.size else pm._ABSENT_RANK
        push.append((overall_pick - best) / teams)

    vec = (timing + held + starters_left + [flex_room, bench_room]
           + room_run + seat_recent + scarcity + push)
    return np.asarray(vec, dtype=float)


def score_pick(prob: np.ndarray, chosen: int):
    """Top-1, top-3, top-5 and log-loss of one prediction. Same definitions as
    `fit_prior.score` (argsort descending, chosen-in-top-k, -log p_chosen) so a
    nested number is directly comparable to a champion number.

    Returned per pick, not aggregated, because the measurement clusters and
    buckets these itself and a pre-averaged rate cannot be re-bucketed.
    """
    order = np.argsort(-prob)
    hit1 = int(order[0] == chosen)
    hit3 = int(chosen in order[:3])
    hit5 = int(chosen in order[:5])
    ll = float(np.log(max(prob[chosen], 1e-12)))
    return hit1, hit3, hit5, ll
