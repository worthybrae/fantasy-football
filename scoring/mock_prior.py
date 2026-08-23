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

This file's first version is not generated: it is the literal that lived in
`scoring/draft_model.py` until 2026-08-23, moved here unchanged, with its
own provenance moved with it. `make fit-prior` overwrites all of it the first
time a refit wins on top-1.

These are the pooled (league-average) coefficients fitted from six real
seasons of an actual PPR league -- 696 picks, eight managers, one league.
They already encode "draft roughly to the market": `reach` -8.1 penalises
taking a player the board ranks well below the pick, `fall` +3.8 rewards
taking a value that has slid, and `qb_early` +1.5 captures a structural fact
every league shares (quarterbacks go earlier than their raw value). An
unknown league's opponents drafting like the average of a real, observed one
is a defensible default and a measured one -- this same pooled fit scores
top-1 0.25 against real drafts.

`pos_DST` -11.14 IS NOT EVIDENCE ABOUT ANYBODY'S BEHAVIOR, AND IT PREDATES
THE BUG FIX THAT WOULD CHANGE IT. The comment this replaces once read the
number as "nobody drafts a defense in round two". It never could have meant
that: the fit behind it ran on a `draft_picks` table holding 712 rows across
six seasons with NOT ONE DST among them -- the positions present were
WR/RB/TE/QB/K only, and every season was missing exactly eight picks (2020
[60, 68, 72, 90, 102, 113, 126, 127]; 2025 [98, 99, 100, 104, 106, 107, 108,
112]; and so on) while all eight of that season's kickers were recorded.
Those eight gaps were the eight defenses. So the fit saw a defense in every
choice set, saw one chosen zero times, and drove this coefficient as negative
as the ridge allowed. It is a correct fit to data in which the event is
unobservable, which is a different thing from a measurement.

THE IMPORT BUG BEHIND THAT IS FIXED. `pipeline/espn_league._is_real_pick`
tested `playerId > 0` to drop ESPN's -1 padding, but ESPN's D/ST ids are
negative BY DESIGN -- -(16000 + proTeamId) -- so the predicate discarded every
defense ever drafted. 9b90610 replaced the threshold with a whitelist. A
re-import of the same six seasons should therefore land 760 picks rather than
712: five sixteen-round drafts plus a fifteen-round 2025, with the 48 missing
defenses restored. That figure is INFERRED from the gap structure, not
measured -- nobody has re-imported, which needs an ESPN login.

THE COEFFICIENT BELOW HAS NOT BEEN RE-FITTED AND IS LEFT EXACTLY AS FITTED.
Editing it by hand would launder a guess as a measurement. What a re-fit
would probably do is flip its sign: run on RECONSTRUCTED defense picks -- the
eight known gaps per season, identities assigned from that season's own
`historic_adp` -- the pooled `pos_DST` moved -9.77 -> +2.81. Read the SIGN
FLIP as robust and the MAGNITUDE as not: assigning by ADP makes every
reconstructed pick look like perfect value, and 2022's ADP table carried only
six defenses, so 46 rows went in rather than 48. The honest summary is that
this number is about to become wrong in a knowable direction, and the
correction is `make espn-import` then `make fit-managers`, or a winning
`make fit-prior`, not an edit here.

Until that is run, this coefficient still decides the end of a simulated
draft: with this prior driving every unresolved opponent, 7 of 8 simulated
teams finished a full 120-pick draft with an empty DST starter slot and
`gain_now` for every defense was exactly 0.0000 at every pick. That is
corrected in the simulator, ON TOP of the fit and clearly separated from it
-- see `scoring/draft_sim._must_fill_mask`, which imposes the roster floor the
coefficients cannot express, exactly as `_roster_cap` imposes the ceiling.
That mask is a roster rule, not a patch for this number, and it stays
whatever a re-fit turns this number into.

The eight trailing 0.0s are `draft_model.UNMEASURED_FEATURES`, padded so the
length assertion stays honest. A 0.0 here is not a fitted finding of "no
effect" -- it is the only value that says NOT YET MEASURED without pretending
otherwise, and it makes a cold-start league behave exactly as it did before
those columns existed, since a zero coefficient contributes nothing to any
score.
"""
import numpy as np

# What was fitted, when, and against what. Machine-readable so a reader does
# not have to trust the prose above and a test can assert the two agree.
PROVENANCE = {
    'fitted_at': '2026-05-01',
    'corpus': "one ESPN PPR league's imported history",
    'drafts': 6,
    'picks_fitted': 696,
    'picks_excluded': {},
    'folds': 'leave-one-season-out',
    'incumbent_top1': None,
    'refit_top1': 0.25,
    'refit_top5': None,
    'refit_logloss': None,
    'features_cut': [],
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
# is not a reproducible input without the list. Empty here: this vector
# predates the corpus and was fitted from a league database's own
# `draft_picks` table, not from `draft_log`.
DRAFT_IDS = [
]

PRIOR = np.array([
    -8.088053,  # reach
    +3.801103,  # fall
    +0.641107,  # pos_RB
    +0.341051,  # pos_WR
    +0.896758,  # pos_TE
    +3.417835,  # pos_K
    -11.141479,  # pos_DST
    +1.482160,  # qb_early
    +0.324423,  # te_early
    +2.246026,  # need
    +0.595798,  # run
    +0.050222,  # age
    -0.331173,  # no_track_record
    -0.063443,  # hype
    -0.003743,  # trend
    +0.000000,  # held_at_pos          <- not yet measured
    +0.000000,  # first_at_pos         <- not yet measured
    +0.000000,  # rounds_since_pos     <- not yet measured
    +0.000000,  # first_at_pos_round   <- not yet measured
    +0.000000,  # usage                <- not yet measured
    +0.000000,  # efficiency           <- not yet measured
    +0.000000,  # played_share         <- not yet measured
    +0.000000,  # peak_gap             <- not yet measured
])

assert len(PRIOR) == len(FEATURES), (
    "the generated prior and its feature list disagree -- regenerate with "
    "`make fit-prior` rather than editing either by hand")
