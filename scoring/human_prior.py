"""The human-only cold-start prior. GENERATED -- do not edit by hand.

Fitted on picks a PERSON is known to have made -- `autodrafted IS FALSE`, not
merely "not flagged" -- and written only because it beat the shipped
`scoring/mock_prior.PRIOR` on held-out human top-1. `pipeline/score_ladder.py`
(`make score-ladder`) is the writer and the rule; there is no force flag, so a
losing refit leaves this file exactly as it was.

NOTHING IMPORTS THIS YET, AND THAT IS DELIBERATE. `scoring.draft_model` still
binds `COLD_START_PRIOR` to `mock_prior.PRIOR`. Swapping it over is one line,
and it belongs at the end of the ladder in
`docs/superpowers/specs/2026-08-23-richer-opponent-model-design.md` rather
than the middle: the mixture rung changes the SHAPE of what a seat is
simulated with (K vectors plus a per-seat posterior, threaded through
`_run_draft`, `survival`, `search_pick` and `predict_board`), and this file is
where those vectors will land. Serving a single vector from here first would
mean changing the serving path twice.

WHY THIS IS NOT `scoring/mock_prior.py`. That file has its own writer,
`make fit-prior`, fitting on a different and much larger population (every
pick not flagged as an autodraft, most of which is unlabelled). Both writers
would regenerate one file under two different rules, and whichever ran last
would win. See `pipeline/score_ladder.py`'s module docstring.

PROVENANCE OF THESE NUMBERS

Fitted on 2511 picks a person is KNOWN to have made
(`autodrafted IS FALSE`, at a seat that is not ours), across 38 drafts of
the cross-league draft corpus (data/draft_corpus.duckdb), on 2026-08-23.

Leave-one-draft-out: every draft held out in turn, the pooled fit rebuilt on
the other N-1 drafts' human picks, and the held-out draft's human picks scored
by a vector that never saw them. The fold is the draft because that is the
unit of independence -- one room, one board, eight strangers.

    ADP baseline       top-1 0.1768  top-5 0.5416  log-loss  8.3058
    shipped prior      top-1 0.2190  top-5 0.6161  log-loss  2.8818
    human-only refit   top-1 0.2358  top-5 0.6388  log-loss  2.7943

delta_top1 against the shipped prior: +0.0167 +/-0.0080, paired by draft.
top-1 is the decision metric and is the one this had to win on; the other two
are recorded because they were measured, not because they decided.

WHAT THIS DOES NOT ESTABLISH.

  * Not that the refit drafts better. It predicts opponents better on one
    population; nothing here measured a roster.
  * Not that it is better on the picks it was measured WITHOUT. 58% of this
    corpus carries a NULL autodraft label and is an unmeasured mixture of
    people and ESPN's engine. Those picks were excluded from the measurement,
    not shown to be anything.
  * Not that any FEATURE matters. Every column here was already shipping and
    none was added, dropped or reweighted by a selection rule. The only thing
    that changed between the shipped row and this one is which picks were
    fitted on. That is the point: it is one rung.
  * Not that it generalises past ESPN's public mock lobby, which is the only
    population that is both labelled and large enough to hold out by draft.
"""
import numpy as np

# What was fitted, when, against what, and what it beat. Machine-readable so a
# reader does not have to trust the prose above and a test can assert the two
# agree.
PROVENANCE = {
    'fitted_at': '2026-08-23',
    'corpus': 'data/draft_corpus.duckdb',
    'population': 'autodrafted IS FALSE (known human), not our seat',
    'drafts': 38,
    'picks_scored': 2511,
    'picks_excluded': {'my_slot': 608, 'not_human': 1738, 'not_in_pool': 0, 'already_taken': 0},
    'folds': 'leave-one-draft-out',
    'features_fitted': ['reach', 'fall', 'pos_RB', 'pos_WR', 'pos_TE', 'pos_K', 'pos_DST', 'qb_early', 'te_early', 'need', 'run', 'age', 'no_track_record', 'hype', 'trend', 'held_at_pos', 'first_at_pos', 'rounds_since_pos', 'first_at_pos_round', 'usage', 'efficiency', 'played_share', 'peak_gap'],
    'adp_top1': 0.176822,
    'shipped_top1': 0.219036,
    'shipped_prior_fitted_at': '2026-08-23',
    'shipped_prior_drafts_also_scored_here': 3,
    'refit_top1': 0.235763,
    'refit_top5': 0.638789,
    'refit_logloss': 2.794347,
    'delta_top1': 0.016726,
    'delta_top1_se': 0.00796,
}

# The feature order these coefficients were fitted in, written out rather than
# assumed. A coefficient vector applied to columns in a different order than
# it was fitted in is silently wrong on every pick, and a length check cannot
# see it.
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
    'dropoff_at_pos',
    'vor',
    'durability',
    'proj_change',
    'last_of_tier',
    'slots_left_at_pos',
]

# The drafts this was fitted and scored on: every draft in the corpus carrying
# at least one KNOWN human pick, snapshotted at the top of the run. Recorded
# because the corpus GROWS -- a farm loop adds roughly one draft every fifteen
# minutes -- so "the corpus" is not a reproducible input without the list.
DRAFT_IDS = [
    'mock:00687860f7454916',
    'mock:0233ebe6ece98629',
    'mock:038d9ceda1cb36f9',
    'mock:0447c2074aa7de7f',
    'mock:0b14dc4f2814961c',
    'mock:22789752b55cbdf7',
    'mock:227bc61114d0c592',
    'mock:25b2c23166704a18',
    'mock:3035da399b163b86',
    'mock:32a44ca29679ec24',
    'mock:39ade9db14aac2e5',
    'mock:3a0b44b74878a7dc',
    'mock:5c1d082a3fe4bf56',
    'mock:6590f653a56e198a',
    'mock:691a1ac37c589416',
    'mock:6c6280258bb260ce',
    'mock:83509c329853622e',
    'mock:8bd46dc6260f7337',
    'mock:8f93fbca2c672fd0',
    'mock:9446b84cf34c9e9b',
    'mock:9581926607965047',
    'mock:96b91127ec7cc3a4',
    'mock:972ba656c17ead93',
    'mock:977127860a1024e5',
    'mock:ae558040e59765be',
    'mock:b1c7c9e3493fcb3e',
    'mock:b935eb49b67de75e',
    'mock:c028397eef680323',
    'mock:c16770000a8323a1',
    'mock:c4608fbb93103d0f',
    'mock:c7bf90fde334e8f5',
    'mock:e249960b3c09bef6',
    'mock:e2bd8c0394a95fa2',
    'mock:e3bc8a5f8199fbb7',
    'mock:ebcfa63feee29442',
    'mock:ee73ecf605a47cc5',
    'mock:f28993f427c4231f',
    'mock:fd27665ffefbc984',
]

PRIOR = np.array([
    -9.054045,  # reach
    +2.017372,  # fall
    +0.417495,  # pos_RB
    -0.011367,  # pos_WR
    +0.187389,  # pos_TE
    +1.948025,  # pos_K
    +1.508460,  # pos_DST
    +3.015632,  # qb_early
    +1.314638,  # te_early
    +0.660876,  # need
    +0.595136,  # run
    +0.004690,  # age
    -0.469124,  # no_track_record
    -0.021900,  # hype
    -0.050123,  # trend
    -0.192803,  # held_at_pos
    -0.078173,  # first_at_pos
    +0.844171,  # rounds_since_pos
    +2.074899,  # first_at_pos_round
    -0.038064,  # usage
    -0.102498,  # efficiency
    +0.160314,  # played_share
    -0.005107,  # peak_gap
    +0.000000,  # dropoff_at_pos       <- not in rung 3, not fitted
    +0.000000,  # vor                  <- not in rung 3, not fitted
    +0.000000,  # durability           <- not in rung 3, not fitted
    +0.000000,  # proj_change          <- not in rung 3, not fitted
    +0.000000,  # last_of_tier         <- not in rung 3, not fitted
    +0.000000,  # slots_left_at_pos    <- not in rung 3, not fitted
])

assert len(PRIOR) == len(FEATURES), (
    "the generated human prior and its feature list disagree -- regenerate "
    "with `make score-ladder` rather than editing either by hand")
