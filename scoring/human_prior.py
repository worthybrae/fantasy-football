"""The human-only cold-start prior. GENERATED -- do not edit by hand.

Fitted on picks a PERSON is known to have made -- `autodrafted IS FALSE`, not
merely "not flagged" -- and written only because EVERY rung of the ladder beat
the one below it on held-out human top-1. `pipeline/score_ladder.py`
(`make score-ladder`) is the writer and the rule; there is no force flag, so a
losing rung leaves this file exactly as it was, still carrying whatever last
won.

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

Fitted on 2889 picks a person is KNOWN to have made
(`autodrafted IS FALSE`, at a seat that is not ours), across 43 drafts of
the cross-league draft corpus (data/draft_corpus.duckdb), on 2026-08-23.

Leave-one-draft-out: every draft held out in turn, the pooled fit rebuilt on
the other N-1 drafts' human picks, and the held-out draft's human picks scored
by a vector that never saw them. The fold is the draft because that is the
unit of independence -- one room, one board, eight strangers.

    ADP baseline       top-1 0.1776  top-5 0.5448  log-loss  8.3181
    shipped prior      top-1 0.2195  top-5 0.6206  log-loss  2.8712
    human-only refit   top-1 0.2326  top-5 0.6390  log-loss  2.7835
    + Tier 1, all six  top-1 0.2555  top-5 0.6556  log-loss  2.7026
    + Tier 1, shipped  top-1 0.2565  top-5 0.6549  log-loss  2.7019

delta_top1 of the shipped set against the human-only refit: +0.0239 +/-0.0078,
paired by draft. top-1 is the decision metric and is the one this had to win
on; the other two are recorded because they were measured, not because they
decided.

EACH TIER 1 FEATURE, DROPPED FROM THE FULL MODEL IN TURN. Positive earns a
place; at or below zero is 0.0 above, meaning MEASURED AND REJECTED. The +/-
is the standard error of the delta itself, paired by draft, and a delta
inside it has not been shown to be worth anything either way.

    dropoff_at_pos     delta_top1 +0.0017 +/-0.0016   kept
    vor                delta_top1 +0.0187 +/-0.0066   kept
    durability         delta_top1 -0.0003 +/-0.0006   cut
    proj_change        delta_top1 -0.0010 +/-0.0020   cut
    last_of_tier       delta_top1 +0.0007 +/-0.0013   kept
    slots_left_at_pos  delta_top1 +0.0010 +/-0.0062   kept

    kept: dropoff_at_pos, vor, last_of_tier, slots_left_at_pos
    cut:  durability, proj_change

WHAT THIS DOES NOT ESTABLISH.

  * Not that the refit drafts better. It predicts opponents better on one
    population; nothing here measured a roster.
  * Not that it is better on the picks it was measured WITHOUT. Most of this
    corpus carries a NULL autodraft label and is an unmeasured mixture of
    people and ESPN's engine. Those picks were excluded from the measurement,
    not shown to be anything.
  * Not that a KEPT Tier 1 feature was available to the drafter as measured.
    `vor`, `durability`, `proj_change` and the `tier` behind `last_of_tier`
    are not in the corpus -- `draft_log_pool` stores `adp_rank` and
    `proj_points` and nothing else -- so all four were joined from the board
    as it stands today rather than as it stood at each draft. They are
    preseason quantities and pick order is unaffected, but a coefficient on
    one of them is worth re-measuring once the corpus records them per draft.
    `dropoff_at_pos` and `slots_left_at_pos` compute from the stored pool and
    `settings` alone and carry no such caveat.
  * Not that a CUT feature is worthless. It was measured on this population,
    at this sample size, in the presence of these other 28 columns, and did
    not pay. `reach` carries roughly -9 here; a collinear addition to a model
    like that splits a coefficient rather than adding signal.
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
    'drafts': 43,
    'picks_scored': 2889,
    'picks_excluded': {'my_slot': 688, 'not_human': 1920, 'not_in_pool': 0, 'already_taken': 0},
    'folds': 'leave-one-draft-out',
    'rung': '4 (+ Tier 1 pool signals)',
    'features_fitted': ['reach', 'fall', 'pos_RB', 'pos_WR', 'pos_TE', 'pos_K', 'pos_DST', 'qb_early', 'te_early', 'need', 'run', 'age', 'no_track_record', 'hype', 'trend', 'held_at_pos', 'first_at_pos', 'rounds_since_pos', 'first_at_pos_round', 'usage', 'efficiency', 'played_share', 'peak_gap', 'dropoff_at_pos', 'vor', 'last_of_tier', 'slots_left_at_pos'],
    'tier1_kept': ['dropoff_at_pos', 'vor', 'last_of_tier', 'slots_left_at_pos'],
    'tier1_cut': ['durability', 'proj_change'],
    'tier1_delta_top1': {'dropoff_at_pos': 0.001731, 'vor': 0.018692, 'durability': -0.000346, 'proj_change': -0.001038, 'last_of_tier': 0.000692, 'slots_left_at_pos': 0.001038},
    'tier1_delta_top1_se': {'dropoff_at_pos': 0.001627, 'vor': 0.006635, 'durability': 0.000559, 'proj_change': 0.002004, 'last_of_tier': 0.001295, 'slots_left_at_pos': 0.006243},
    'board_joined_from': "today's build_board, not attributes_as_of (vor, durability, proj_change, tier)",
    'adp_top1': 0.17757,
    'shipped_top1': 0.219453,
    'shipped_prior_fitted_at': '2026-08-23',
    'shipped_prior_drafts_also_scored_here': 3,
    'rung3_top1': 0.232606,
    'rung4_all_six_top1': 0.255452,
    'refit_top1': 0.25649,
    'refit_top5': 0.654898,
    'refit_logloss': 2.701915,
    'delta_top1': 0.023884,
    'delta_top1_se': 0.007802,
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
    'mock:004116f169a351aa',
    'mock:00687860f7454916',
    'mock:0233ebe6ece98629',
    'mock:038d9ceda1cb36f9',
    'mock:0447c2074aa7de7f',
    'mock:063ee4abf4eecead',
    'mock:0b14dc4f2814961c',
    'mock:22789752b55cbdf7',
    'mock:227bc61114d0c592',
    'mock:25b2c23166704a18',
    'mock:262ad385da8185bb',
    'mock:3035da399b163b86',
    'mock:32a44ca29679ec24',
    'mock:39ade9db14aac2e5',
    'mock:3a0b44b74878a7dc',
    'mock:41e1313647ad5254',
    'mock:5c1d082a3fe4bf56',
    'mock:6590f653a56e198a',
    'mock:691a1ac37c589416',
    'mock:6c6280258bb260ce',
    'mock:7312c9b7e46d414f',
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
    -6.568392,  # reach
    +0.436514,  # fall
    +3.032322,  # pos_RB
    +2.813748,  # pos_WR
    +0.114657,  # pos_TE
    +0.380861,  # pos_K
    +0.340787,  # pos_DST
    +1.001721,  # qb_early
    +1.672620,  # te_early
    +1.654234,  # need
    +0.706883,  # run
    -0.023157,  # age
    -0.147991,  # no_track_record
    -0.001382,  # hype
    -0.035313,  # trend
    -0.313009,  # held_at_pos
    +0.950964,  # first_at_pos
    +0.820284,  # rounds_since_pos
    +0.800972,  # first_at_pos_round
    -0.003938,  # usage
    -0.018092,  # efficiency
    +0.104755,  # played_share
    -0.001722,  # peak_gap
    -0.055202,  # dropoff_at_pos
    +0.739315,  # vor
    +0.000000,  # durability           <- measured on humans, cut on delta_top1
    +0.000000,  # proj_change          <- measured on humans, cut on delta_top1
    +1.891957,  # last_of_tier
    -1.115066,  # slots_left_at_pos
])

assert len(PRIOR) == len(FEATURES), (
    "the generated human prior and its feature list disagree -- regenerate "
    "with `make score-ladder` rather than editing either by hand")
