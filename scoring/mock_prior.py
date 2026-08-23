"""The cold-start prior's coefficients. GENERATED -- do not edit by hand.

`scoring.draft_model` imports `PRIOR` from here and binds it to
`COLD_START_PRIOR`, the coefficient vector every opponent in a league with no
draft history is simulated with. In a mock draft, where all eight opponents
are strangers, this vector IS the entire model.

Written by `pipeline/fit_prior.py` (`make fit-prior`), and ONLY when the fit
it describes beat the incumbent on held-out top-1 accuracy. There is no way
to write a losing vector here short of editing that rule, which is the point.
Hand-editing a number here would launder a guess as a measurement; refit
instead.

The previous vector, and the provenance of its numbers, is in this file's git
history.

PROVENANCE OF THESE NUMBERS

Fitted on 26 mock drafts from the cross-league draft corpus
(data/draft_corpus.duckdb), 3267 picks, on 2026-08-23.

Leave-one-DRAFT-out (every draft in this corpus is one season, so
leave-one-season-out is degenerate here):

    incumbent  top-1 0.2436  top-5 0.6385  log-loss 3.5463
    refit      top-1 0.2834  top-5 0.7086  log-loss 2.6221

top-1 is the decision metric and it is the one this had to win on. The other
two are recorded because they were measured, not because they decided.

Of the eight features Task 3 added, 3 earned a place on delta_top1 and
5 did not. The cut ones carry 0.0 here, which means MEASURED AND
REJECTED on this corpus -- a different fact from the 0.0 they carried before,
which meant not yet measured.

    kept: held_at_pos, first_at_pos_round, usage
    cut:  first_at_pos, rounds_since_pos, efficiency, played_share, peak_gap

The numbers and what they do not establish:
docs/superpowers/findings/2026-08-23-mock-corpus-features.md
"""
import numpy as np

# What was fitted, when, and against what. Machine-readable so a reader does
# not have to trust the prose above and a test can assert the two agree.
PROVENANCE = {
    'fitted_at': '2026-08-23',
    'corpus': 'data/draft_corpus.duckdb',
    'drafts': 26,
    'picks_fitted': 3267,
    'picks_excluded': {'my_slot': 48, 'autodrafted': 13, 'not_in_pool': 0, 'already_taken': 0},
    'folds': 'leave-one-draft-out',
    'incumbent_top1': 0.243649,
    'refit_top1': 0.28344,
    'refit_top5': 0.708601,
    'refit_logloss': 2.622129,
    'features_cut': ['first_at_pos', 'rounds_since_pos', 'efficiency', 'played_share', 'peak_gap'],
}

# The feature order these coefficients were fitted in, written out rather
# than assumed. `draft_model` asserts this equals `FEATURE_NAMES`: a
# coefficient vector applied to columns in a different order than it was
# fitted in is silently wrong, and the length check alone cannot see it.
FEATURES = [
    'reach',
    'fall',
    'pos_RB',
    'pos_WR',
    'pos_TE',
    'pos_K',
    'pos_DST',
    'qb_early',
    'te_early',
    'need',
    'run',
    'age',
    'no_track_record',
    'hype',
    'trend',
    'held_at_pos',
    'first_at_pos',
    'rounds_since_pos',
    'first_at_pos_round',
    'usage',
    'efficiency',
    'played_share',
    'peak_gap',
]

# The draft_ids this was fitted on, recorded because the corpus GROWS -- a
# farm loop adds roughly one draft every fifteen minutes -- so "the corpus"
# is not a reproducible input without the list.
DRAFT_IDS = [
    'mock:0028e1dc89c88eb5',
    'mock:00687860f7454916',
    'mock:0233ebe6ece98629',
    'mock:2dc424ddc7cc0f14',
    'mock:32a44ca29679ec24',
    'mock:3737b470da8def51',
    'mock:4f20a6596a6a69bd',
    'mock:564f6decbbc0047c',
    'mock:58054006f9406c11',
    'mock:58bc3b8ffb26c236',
    'mock:5ad16823220a9f72',
    'mock:5c68221be6e260f1',
    'mock:69022319b7b17b01',
    'mock:6ecbec89a8784261',
    'mock:7394533b30b8e0f1',
    'mock:8e90348faa08ec66',
    'mock:9609995b231d3a10',
    'mock:b4ad423e39dbd5c6',
    'mock:cc20e63568f7f952',
    'mock:d237677dfb1ed04c',
    'mock:e0263539b8ad194e',
    'mock:e12c7b7831d435d8',
    'mock:e23a7228a859bcbb',
    'mock:e7ada8d32b328964',
    'mock:e7f2cb895fec713a',
    'mock:ed5d3690611d75bd',
]

PRIOR = np.array([
    -11.238877,  # reach
    +2.599615,  # fall
    -0.141912,  # pos_RB
    -0.657625,  # pos_WR
    +0.139255,  # pos_TE
    +0.603227,  # pos_K
    +0.213625,  # pos_DST
    +3.526412,  # qb_early
    +1.764431,  # te_early
    +0.630082,  # need
    +1.783434,  # run
    -0.037368,  # age
    -0.529309,  # no_track_record
    -0.019710,  # hype
    -0.064520,  # trend
    -0.049080,  # held_at_pos
    +0.000000,  # first_at_pos         <- cut on delta_top1, not fitted
    +0.000000,  # rounds_since_pos     <- cut on delta_top1, not fitted
    +3.658854,  # first_at_pos_round
    -0.035498,  # usage
    +0.000000,  # efficiency           <- cut on delta_top1, not fitted
    +0.000000,  # played_share         <- cut on delta_top1, not fitted
    +0.000000,  # peak_gap             <- cut on delta_top1, not fitted
])

assert len(PRIOR) == len(FEATURES), (
    "the generated prior and its feature list disagree -- regenerate with "
    "`make fit-prior` rather than editing either by hand")
