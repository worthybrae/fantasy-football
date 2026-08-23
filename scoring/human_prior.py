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

Fitted on 3935 picks a person is KNOWN to have made
(`autodrafted IS FALSE`, at a seat that is not ours), across 57 drafts of
the cross-league draft corpus (data/draft_corpus.duckdb), on 2026-08-23.

THE ESPN BOARD RUNG. The champion prices candidates against `market_rank`;
this rung adds five columns computed against ESPN's own on-screen list
(`espn_rank`/`espn_proj`), the board an ESPN mock lobby actually reads. Every
rung is one leave-one-draft-out over the same human picks in the same rooms.

    champion (human_prior)   top-1 0.2574  top-5 0.6562  log-loss  2.7272
    + ESPN board, all five   top-1 0.2526  top-5 0.6717  log-loss  2.6632
    + ESPN board, shipped    top-1 0.2595  top-5 0.6686  log-loss  2.6959

delta_top1 of the shipped ESPN subset against the champion: +0.0020 +/-0.0044,
paired by draft. The rule for this rung is top-1 UP and log-loss NOT WORSE,
both recomputed on this snapshot; there is no force flag.

EACH ESPN FEATURE, DROPPED FROM THE FULL MODEL IN TURN. Positive earns a
place; at or below zero is 0.0, meaning MEASURED AND REJECTED. The +/- is the
standard error of the delta itself, paired by draft.

    espn_reach         delta_top1 +0.0003 +/-0.0030   kept
    espn_fall          delta_top1 +0.0003 +/-0.0037   kept
    espn_list_pos      delta_top1 -0.0053 +/-0.0064   cut
    board_disagreement delta_top1 +0.0023 +/-0.0031   kept
    espn_proj_dropoff  delta_top1 -0.0015 +/-0.0015   cut

    kept: espn_reach, espn_fall, board_disagreement
    cut:  espn_list_pos, espn_proj_dropoff

DOES ESPN'S BOARD SUBSUME THE MARKET'S? The pooled fit given both boards, its
coefficients side by side. If the room reads ESPN, `market_rank`'s `reach`
leans on `espn_reach` and gives up ground.

    reach              coef -8.0192   std 0.9545   standardized -7.6545
    fall               coef +3.6539   std 0.0277   standardized +0.1013
    espn_reach         coef +3.3472   std 1.1139   standardized +3.7286
    espn_fall          coef -5.2907   std 0.0169   standardized -0.0893
    vor                coef +0.3860   std 4.0100   standardized +1.5480
    espn_list_pos      coef -0.6796   std 1.7161   standardized -1.1663
    board_disagreement coef +4.1007   std 0.2564   standardized +1.0516
    espn_proj_dropoff  coef -0.1456   std 0.5715   standardized -0.0832
    dropoff_at_pos     coef +0.1202   std 0.5650   standardized +0.0679

WHAT THIS DOES NOT ESTABLISH.

  * Not that the refit drafts better. It predicts opponents better on one
    population; nothing here measured a roster.
  * Not that `espn_rank` was the board each draft was PLAYED against. The
    corpus backfilled it from today's preseason board (preseason-stable, the
    same provenance status as `vor`); a coefficient on it is worth
    re-measuring once every draft records its own ESPN board at record time.
  * Not that it generalises past ESPN's public mock lobby, the only
    population both labelled and large enough to hold out by draft.
"""
import numpy as np

# What was fitted, when, against what, and what it beat. Machine-readable so a
# reader does not have to trust the prose above and a test can assert the two
# agree.
PROVENANCE = {
    'fitted_at': '2026-08-23',
    'corpus': 'data/draft_corpus.duckdb',
    'population': 'autodrafted IS FALSE (known human), not our seat',
    'drafts': 57,
    'picks_scored': 3935,
    'picks_excluded': {'my_slot': 912, 'not_human': 2432, 'not_in_pool': 0, 'already_taken': 0},
    'folds': 'leave-one-draft-out',
    'rung': 'ESPN board features (+ on the human-only champion)',
    'features_fitted': ['reach', 'fall', 'pos_RB', 'pos_WR', 'pos_TE', 'pos_K', 'pos_DST', 'qb_early', 'te_early', 'need', 'run', 'age', 'no_track_record', 'hype', 'trend', 'held_at_pos', 'first_at_pos', 'rounds_since_pos', 'first_at_pos_round', 'usage', 'efficiency', 'played_share', 'peak_gap', 'dropoff_at_pos', 'vor', 'last_of_tier', 'slots_left_at_pos', 'espn_reach', 'espn_fall', 'board_disagreement'],
    'tier1_kept': ['dropoff_at_pos', 'vor', 'last_of_tier', 'slots_left_at_pos'],
    'tier1_cut': ['durability', 'proj_change'],
    'espn_kept': ['espn_reach', 'espn_fall', 'board_disagreement'],
    'espn_cut': ['espn_list_pos', 'espn_proj_dropoff'],
    'espn_delta_top1': {'espn_reach': 0.000254, 'espn_fall': 0.000254, 'espn_list_pos': -0.005337, 'board_disagreement': 0.002287, 'espn_proj_dropoff': -0.001525},
    'espn_delta_top1_se': {'espn_reach': 0.002985, 'espn_fall': 0.003678, 'espn_list_pos': 0.006414, 'board_disagreement': 0.003147, 'espn_proj_dropoff': 0.001462},
    'espn_rank_provenance': "backfilled from today's preseason ESPN board (preseason-stable, as vor)",
    'champion_top1': 0.257433,
    'champion_logloss': 2.727244,
    'espn_all_five_top1': 0.252605,
    'refit_top1': 0.259466,
    'refit_top5': 0.668615,
    'refit_logloss': 2.695896,
    'delta_top1': 0.002033,
    'delta_top1_se': 0.004353,
    'delta_logloss': -0.031348,
    'reach_coef': -8.019245,
    'espn_reach_coef': 3.347232,
    'vor_delta_top1_no_espn': 0.023634,
    'vor_delta_top1_with_espn': 0.005337,
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
    'espn_reach',
    'espn_fall',
    'espn_list_pos',
    'board_disagreement',
    'espn_proj_dropoff',
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
    'mock:181378b64a3fbd8a',
    'mock:1b0cc78dbc92bea9',
    'mock:22789752b55cbdf7',
    'mock:227bc61114d0c592',
    'mock:25b2c23166704a18',
    'mock:262ad385da8185bb',
    'mock:3035da399b163b86',
    'mock:32a44ca29679ec24',
    'mock:382a9f845d880b90',
    'mock:39ade9db14aac2e5',
    'mock:3a0b44b74878a7dc',
    'mock:41e1313647ad5254',
    'mock:4274dacd8d39aee6',
    'mock:45484ce2db59aa9c',
    'mock:554753b75e58023c',
    'mock:57b9343e83b32257',
    'mock:5c1d082a3fe4bf56',
    'mock:63f4238fdc0ab0ba',
    'mock:6590f653a56e198a',
    'mock:691a1ac37c589416',
    'mock:6c6280258bb260ce',
    'mock:7312c9b7e46d414f',
    'mock:83509c329853622e',
    'mock:8461abd00fdb7794',
    'mock:8bd46dc6260f7337',
    'mock:8f93fbca2c672fd0',
    'mock:9446b84cf34c9e9b',
    'mock:9581926607965047',
    'mock:96b91127ec7cc3a4',
    'mock:972ba656c17ead93',
    'mock:977127860a1024e5',
    'mock:abc1184aa88265b5',
    'mock:ae558040e59765be',
    'mock:b1c7c9e3493fcb3e',
    'mock:b935eb49b67de75e',
    'mock:c028397eef680323',
    'mock:c16770000a8323a1',
    'mock:c4608fbb93103d0f',
    'mock:c7bf90fde334e8f5',
    'mock:c9269583ed56d47c',
    'mock:e249960b3c09bef6',
    'mock:e2bd8c0394a95fa2',
    'mock:e3bc8a5f8199fbb7',
    'mock:e4eeb45099f5a041',
    'mock:e7bb4d9b636afd74',
    'mock:ebcfa63feee29442',
    'mock:ee73ecf605a47cc5',
    'mock:f28993f427c4231f',
    'mock:fd27665ffefbc984',
    'mock:fd342acab99d4c78',
]

PRIOR = np.array([
    -7.713723,  # reach
    +4.988896,  # fall
    +2.485491,  # pos_RB
    +2.121963,  # pos_WR
    +0.198773,  # pos_TE
    +2.488597,  # pos_K
    +0.288510,  # pos_DST
    +0.966162,  # qb_early
    +0.639503,  # te_early
    +1.670736,  # need
    +0.616810,  # run
    +0.006137,  # age
    -0.298222,  # no_track_record
    -0.046683,  # hype
    -0.009332,  # trend
    -0.317584,  # held_at_pos
    +0.964633,  # first_at_pos
    +0.702190,  # rounds_since_pos
    +0.940316,  # first_at_pos_round
    -0.020866,  # usage
    -0.058913,  # efficiency
    +0.162954,  # played_share
    +0.002012,  # peak_gap
    +0.027821,  # dropoff_at_pos
    +0.418933,  # vor
    +0.000000,  # durability           <- measured on humans, cut on delta_top1
    +0.000000,  # proj_change          <- measured on humans, cut on delta_top1
    +2.002389,  # last_of_tier
    -1.193458,  # slots_left_at_pos
    +0.690556,  # espn_reach
    -4.202055,  # espn_fall
    +0.000000,  # espn_list_pos        <- measured on humans, cut on delta_top1
    +4.669544,  # board_disagreement
    +0.000000,  # espn_proj_dropoff    <- measured on humans, cut on delta_top1
])

assert len(PRIOR) == len(FEATURES), (
    "the generated human prior and its feature list disagree -- regenerate "
    "with `make score-ladder` rather than editing either by hand")
