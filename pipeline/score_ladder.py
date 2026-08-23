"""The human-only evaluation harness, and the ladder. Run: make score-ladder

WHY THIS EXISTS AT ALL, WHICH IS THE ONLY THING ABOUT IT WORTH GETTING RIGHT.
`make fit-prior` measures a candidate prior against every pick in the corpus
that is not a recorded autodraft. That population is mostly not people. The
shipped prior scores 0.3010 top-1 on picks ESPN's engine made and 0.2308 on
picks a person made, so a mixed metric is roughly seven points of "how well do
we predict a bot" added to the number anyone reads as "how well do we predict
the room". Every change measured that way is measured wrong: a feature that
sharpens the bot half and does nothing for the human half looks like a win.

This module is the measurement that is not wrong. Everything here is scored on
HUMAN PICKS ONLY, held out by draft, with standard errors clustered by draft.
It is the yardstick the rest of `docs/superpowers/specs/2026-08-23-richer-
opponent-model-design.md` is judged by -- Tier 1's pool features and Tier 2's
latent-class mixture are both rungs added to the ladder below -- so it lands
first, and it establishes rung 3.

THE TWO FILTERS, WHICH ARE NOT THE SAME FILTER AND MUST NEVER BE MERGED.

`draft_log_pick.autodrafted` has three states and they are three facts:

    TRUE   ESPN's engine took the pick. Not a decision.
    FALSE  a person took the pick. Ground truth, from the room's own owner
           census plus the AUTODRAFT broadcasts (`pipeline.mock_farm`).
    NULL   nobody recorded either way.

    * THE FIT keeps TRUE out and keeps NULL in (`fit_prior.fit_keeps_pick`).
      Its question is "is there reason to believe a machine made this pick",
      and NULL is not such a reason. Every backfilled draft carries NULL on
      every pick -- a `drafted` table is player_id and pick_no and nothing
      else -- so dropping NULL would drop most of the corpus. That is a
      standing ruling and this module does not touch it.
    * THE EVALUATION keeps only FALSE (`is_human_pick`, below). Its question
      is "do I KNOW a person made this pick", and NULL is not knowledge. The
      23 harvested drafts and every farmed draft recorded before the owner
      census carry NULL and can never be labelled, because the rooms are gone.
      Scoring them as human would put an unknown mixture of engine picks back
      into the number this whole module exists to keep them out of.

The predicates therefore point in opposite directions on NULL, on purpose, and
`tests/test_score_ladder.py` asserts that swapping one for the other changes
the answer. Conflating them is the single easiest way to get this wrong, and
it would be invisible: the harness would still run, still print a table, and
still be measuring a population that is 58% unlabelled.

WHAT THE LADDER SEPARATES. Three effects would otherwise be confounded --
fitting on humans, the Tier 1 features, and the archetypes -- so each is one
rung and each must beat the rung below it on held-out human top-1:

    1  ADP baseline           the market alone, the floor
    2  shipped prior          scoring/mock_prior.PRIOR, engine-heavy corpus
    3  human-only refit       same 23 features, fitted on human picks only
    4  + Tier 1 features      the six pool signals, plus a per-feature cut
    5  + mixture, best K      another agent's rung

Rung 5 is not here. `human_observations` and `ladder` are written to be
extended with it rather than copied.

RUNG 4 DOES TWO THINGS AND THE SECOND IS THE ONE THAT MATTERS. The rung
itself is one more leave-one-draft-out, over rung 3's 23 columns plus the six
`_POOL_SIGNAL_FEATURES` -- that answers "do these six, together, predict a
person better". It does not answer which of them earned it, and a set of six
that wins by a point because one of them is worth a point is six features
where one belongs. So `ablation` below drops each candidate from the full
model in turn and reports its own `delta_top1` WITH a paired-by-draft
standard error, and only the candidates with a positive delta are fitted into
the artifact. A delta of +0.002 against a standard error of 0.005 is not a
small win, it is an unmeasured quantity, and printing it without the error
beside it is how one gets read as the other.

The ablation runs WHETHER OR NOT THE RUNG WINS, before the rule is applied.
It is diagnostic and not decisional, and a losing rung is the case where it
is worth the most: "the six lost" is barely a finding, while "the six lost,
five of them move nothing and the sixth is +0.004 +/-0.003" says what to try
next and what to stop trying. Gating the diagnosis on the win would leave the
outcome the design document says to expect reporting the least.

THE RULE, WHICH IS `fit_prior`'S RULE AND NOT THIS MODULE'S TO BEND.
`delta_top1` decides, against the rung below. Top-5 and log-loss are printed
for every rung, always, so neither can be reached for only when it flatters
the answer. There is deliberately NO force flag: the artifact is written if
and only if EVERY rung beat the one below it on held-out human top-1, and a
losing rung prints its numbers, writes nothing, and exits 1.

That "every rung" is a chain and not a preference. The artifact carries the
TOP rung's vector, so rung 4 losing to rung 3 does not fall back to writing
rung 3 -- it writes nothing at all and leaves the file holding the rung 3
that was already measured and already shipped. Regenerating rung 3 here on a
corpus that has grown since would silently replace a recorded measurement
with a different one under the heading of a Tier 1 run that lost.

THE PROVENANCE WEAKNESS RUNG 4 CARRIES, STATED WHERE IT CANNOT BE MISSED.
Four of the six Tier 1 columns -- `vor`, `durability`, `proj_change` and the
`tier` that `last_of_tier` is derived from -- are NOT in the corpus.
`draft_log_pool` stores `adp_rank` and `proj_points` per row and nothing
else, so `fit_prior.board_signals_by_player_id` joins those four from the
board as it stands TODAY rather than as it stood when each mock was drafted.
Pick order, `market_rank`, `reach` and `fall` are unaffected (they are stored
per draft), and all four joined columns are preseason quantities that move
with a projection refresh rather than with a result -- but that is a weaker
claim than `attributes_as_of` makes, and it is weaker in the direction that
matters: a feature that wins only because it was joined from a board the
drafter could not see is worse than no feature. So if `vor` or `last_of_tier`
is among the winners, the findings document has to say that its win rests on
this join and is worth re-measuring once the corpus records the columns per
draft. `dropoff_at_pos` and `slots_left_at_pos` are exempt -- they compute
from the stored pool and `settings` alone.

WHERE A WINNING RUNG 3 IS WRITTEN, AND WHY NOT `scoring/mock_prior.py`.
`scoring/human_prior.py`, a separate generated artifact. Three reasons:

  1. `mock_prior.py` already has a writer with a different rule.
     `make fit-prior` regenerates it from the MIXED corpus whenever that fit
     wins, so a human-only vector written there would be silently replaced by
     an engine-heavy one the next time anyone ran the other command. Two
     writers, two disciplines, one file is how a measurement gets lost.
  2. Its contract has no room for what comes next. `draft_model` imports one
     vector named `PRIOR` and asserts its length; Tier 2 needs K vectors plus
     K weights in a file. Widening `mock_prior.py` to hold both shapes would
     put the mixture into the file the live simulator imports before anything
     has decided that the mixture ships.
  3. Nothing serves it yet, ON PURPOSE. Swapping `COLD_START_PRIOR` over is a
     one-line change in `scoring/draft_model.py` and it belongs at the END of
     the ladder, not the middle of it -- the spec's own integration risk is
     the five call sites that thread a fit through `_run_draft`, `survival`,
     `search_pick` and `predict_board`, and Tier 2 owns that change.

READ-ONLY, ABSOLUTELY. The corpus cannot be rebuilt: a mock draft not written
down when it happens is gone, and a farm loop is writing new ones into this
same file while this runs. Both databases are opened read-only through
`fit_prior.open_corpus` / `open_league`, no statement of any kind is issued
against `draft_log*`, and `ensure_schema` is never called (DuckDB refuses even
a no-op CREATE against a read-only attachment -- see
`mock_backfill.corpus_report`). The draft ids are snapshotted once at the top
of the run and printed, because "the corpus" is not a reproducible input
without the list.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import draft_log as dl
from pipeline import fit_prior as fp
from scoring import mock_prior
from scoring.draft_model import (COLD_START_PRIOR, FEATURE_NAMES,
                                 UNMEASURED_FEATURES, _POOL_SIGNAL_FEATURES,
                                 _ROUND_BUCKET_ORDER)

# Where a winning rung 3 lands. See the module docstring for why this is not
# `scoring/mock_prior.py`.
HUMAN_PRIOR_MODULE = (Path(__file__).resolve().parent.parent
                      / "scoring" / "human_prior.py")

# RUNG 3 IS "THE CURRENT 23 FEATURES", AND THE 23 ARE NAMED HERE RATHER THAN
# SLICED OFF `FEATURE_NAMES`.
#
# The ladder's whole job is to separate three effects that would otherwise be
# confounded: fitting on humans, the Tier 1 pool signals, and the archetypes.
# Rung 3 is the first of those ALONE, so it must be fitted on exactly the
# columns that were already shipping -- and `FEATURE_NAMES` is a moving
# target, because Tier 1's rung appends to it. `FEATURE_NAMES[:23]` would have
# been silently correct today and silently WRONG the moment a new column was
# inserted anywhere but the end, at which point rung 3 would quietly become
# "humans plus one Tier 1 feature" and the decomposition the ladder exists for
# would be gone with no error anywhere.
#
# So the set is written out. The assertion below is the other half: a feature
# RENAMED out from under this list must fail loudly here, not drop a column
# from the fit and report a smaller number as if it were a measurement.
SHIPPED_23_FEATURES = [
    "reach", "fall", "pos_RB", "pos_WR", "pos_TE", "pos_K", "pos_DST",
    "qb_early", "te_early", "need", "run", "age", "no_track_record", "hype",
    "trend",                          # the 15 legacy columns
    "held_at_pos", "first_at_pos", "rounds_since_pos", "first_at_pos_round",
    "usage", "efficiency", "played_share", "peak_gap",
]
assert len(SHIPPED_23_FEATURES) == 23, "rung 3 is defined as 23 features"
assert set(SHIPPED_23_FEATURES) <= set(FEATURE_NAMES), (
    "a feature rung 3 is pinned to no longer exists in FEATURE_NAMES: "
    f"{sorted(set(SHIPPED_23_FEATURES) - set(FEATURE_NAMES))}. Rung 3 is the "
    "human-only refit of the columns that were already shipping, so a rename "
    "has to be reflected here deliberately rather than shrinking the fit")

# RUNG 4 IS RUNG 3 PLUS THE SIX, AND IT IS SPELLED AS THAT SUM ON PURPOSE.
#
# `TIER1_FEATURES` is `draft_model._POOL_SIGNAL_FEATURES`, imported rather
# than re-listed, because that constant is where the six are DEFINED -- the
# matrix columns, the `COLD_START_PRIOR` zeros and the note explaining why
# `dropoff_at_pos` is not the spec's `pos_dropoff` all hang off it. A second
# hand-written copy here would let the two drift, and a Tier 1 rung fitted
# on five of the six while claiming six is exactly the silent shrink the
# assertion under `SHIPPED_23_FEATURES` exists to prevent.
#
# Written as `SHIPPED_23_FEATURES + TIER1_FEATURES` rather than as
# `FEATURE_NAMES` even though the two are the same 29 columns today. They are
# the same by coincidence of what has been added so far, and the coincidence
# is load-bearing in one place: `fp.leave_one_draft_out(keep_idx=...)` is
# indexed against `FEATURE_NAMES`, so the moment rung 5 (or anything else)
# appends a 30th column, `FEATURE_NAMES` would quietly widen rung 4 into
# "the six plus whatever landed last" while still being labelled the Tier 1
# rung. The sum cannot: it names what it fits.
TIER1_FEATURES = list(_POOL_SIGNAL_FEATURES)
RUNG_4_FEATURES = SHIPPED_23_FEATURES + TIER1_FEATURES
assert len(TIER1_FEATURES) == 6, "Tier 1 is six features"
assert not (set(TIER1_FEATURES) & set(SHIPPED_23_FEATURES)), (
    "a Tier 1 feature is also pinned into rung 3, so rung 4 would be measuring"
    " it against a rung that already had it")
assert set(RUNG_4_FEATURES) <= set(FEATURE_NAMES), (
    "rung 4 names a column `feature_matrix` does not produce: "
    f"{sorted(set(RUNG_4_FEATURES) - set(FEATURE_NAMES))}")

# The ablation's candidates, and the reason this name is bound here at all.
#
# THE FAILURE THIS EXISTS TO PREVENT. There are two ablation functions in this
# codebase and they default to two different sets: `fit_prior.ablate` to
# `UNMEASURED_FEATURES` (fourteen columns) and `draft_model.ablation` to
# `_NEW_FEATURES` (four columns from 2026-08-11, and NONE of the six this rung
# is about). Reaching for the second produces a full table of plausible deltas
# that tested nothing of rung 4, exits 0, and reads exactly like a result.
# That has been a near-miss on this project once already. So `ablation` below
# takes `candidates` as a REQUIRED argument -- no default that could be the
# wrong one -- and `main` passes this list, by name, and nothing else.
#
# THE LIST IS THE SIX, AND THAT IS A DELIBERATE NARROWING OF
# `UNMEASURED_FEATURES`. Every one of the six is here; that is the assertion
# below and it is the only thing about this list that must never change. What
# is NOT here is the other eight of `UNMEASURED_FEATURES` -- the roster-shape
# and stat-profile columns -- for two reasons that point the same way:
#
#   1. They cannot change anything. Rung 3 is pinned to all 23 by name
#      (`SHIPPED_23_FEATURES`), and `tier1_winners` only ever selects out of
#      the six, because cutting an older column inside the Tier 1 rung would
#      fold "an already-shipping column stopped paying" into a rung whose
#      headline is "the six pool signals paid" -- two effects in one number,
#      the exact confound the ladder exists to take apart. Their rows would be
#      read and not acted on.
#   2. They are not free. Every row is a complete leave-one-draft-out
#      backtest: 43 pooled conditional-logit fits, about eight minutes each on
#      this corpus. Eight extra rows is an extra hour on a job that already
#      runs over one, and the first attempt at this rung was killed by the OS
#      at 2h18m with the table unfinished and nothing written. A measurement
#      that does not survive to print is worth less than a smaller one that
#      does.
#
# So the eight stay measured where they were measured -- `make fit-prior` on
# the mixed corpus, docs/superpowers/findings/2026-08-23-mock-corpus-features
# .md -- and re-measuring them on humans only is its own rung for its own run.
ABLATION_CANDIDATES = list(TIER1_FEATURES)
assert set(TIER1_FEATURES) <= set(ABLATION_CANDIDATES), (
    "a Tier 1 feature is not in the ablation's candidate list, so the "
    "ablation would never test it and rung 4 would ship a column that "
    "nothing measured -- which is the one way this rung can be silently wrong")
assert set(ABLATION_CANDIDATES) <= set(UNMEASURED_FEATURES), (
    "the ablation names a column that is not on `UNMEASURED_FEATURES`, so "
    "something is being measured here that no other fit knows is unmeasured")
assert set(ABLATION_CANDIDATES) <= set(RUNG_4_FEATURES), (
    "the ablation would try to drop a column rung 4 does not fit: "
    f"{sorted(set(ABLATION_CANDIDATES) - set(RUNG_4_FEATURES))}")

# The bucket a pick dropped by the EVALUATION filter is counted under.
# Deliberately not "autodrafted": most picks this filter drops are NULL, which
# is unknown, and a count printed under the word "autodrafted" would be a
# number that is not what its label says.
NOT_HUMAN = "not_human"


def is_human_pick(autodrafted) -> bool:
    """THE EVALUATION's rule: a pick counts only when `autodrafted IS FALSE`.

    The counterpart to `fit_prior.fit_keeps_pick`, and the opposite of it on
    NULL. Read the two together; the module docstring says why they differ and
    why neither may be used in the other's place.

    NULL is UNKNOWN and unknown is not human. It is tempting to read "not
    flagged as autodrafted" as "a person did it", which is exactly what the
    fit does and exactly what a measurement must not do: 58% of this corpus is
    NULL, those picks are an unmeasured mixture of people and ESPN's engine,
    and the whole point of scoring humans separately is that the engine half
    scores about seven points higher.

    `pd.isna` rather than `is None` because the value arrives from DuckDB via
    pandas, where a null boolean column comes back as `NaN`, `None` or
    `pd.NA` depending on the column's inferred dtype -- all three mean the
    same thing here and all three must be excluded.
    """
    return (not pd.isna(autodrafted)) and (not bool(autodrafted))


def labelled_draft_ids(corpus, draft_ids: list) -> list:
    """The snapshot restricted to drafts that carry at least one human pick.

    A draft with no `autodrafted IS FALSE` pick anywhere contributes nothing
    to a human-only measurement, and leaving it in the fold list would produce
    a leave-one-draft-out fold whose test set is empty. `leave_one_draft_out`
    already skips those, so this is not a correctness fix -- it is so that the
    printed snapshot names the drafts the numbers actually rest on. "37 of 60"
    is the honest description of this corpus and "60" is not.

    Restricted to `draft_ids` rather than querying the whole table, because
    the farm is writing new drafts into this database while this runs and the
    snapshot is the one list every query in the run must agree about.
    """
    if not draft_ids:
        return []
    rows = corpus.execute(
        """SELECT DISTINCT draft_id FROM draft_log_pick
           WHERE autodrafted IS FALSE AND draft_id IN ({})
           ORDER BY draft_id""".format(",".join(["?"] * len(draft_ids))),
        draft_ids).df()
    return rows["draft_id"].tolist()


def human_observations(corpus, league_conn, draft_ids: list):
    """The replay, over the same drafts, keeping only KNOWN human picks.

    `fit_prior.build_corpus_observations` with the evaluation predicate passed
    in -- not a second copy of that loop. That matters more than it looks:
    the loop's real content is the bookkeeping, and a pick this filter drops
    must STILL take its player off the board, still fill a roster slot, and
    still count toward the room's positional run. An autodrafted pick happened.
    A NULL-labelled pick happened. The next human pick has to be offered the
    board those picks left behind, or every feature that reads the pool is
    computed against a draft that never took place.

    Sharing the replay is also what makes rung 2 and rung 3 comparable at all:
    both are scored on choice sets built by one piece of code from one board.
    """
    return fp.build_corpus_observations(corpus, league_conn, draft_ids,
                                        keep_pick=is_human_pick,
                                        drop_label=NOT_HUMAN)


# ------------------------------------------------------------- the ladder

class Rung:
    """One row of the ladder: a label, a scored report, and how it was scored.

    `heldout` records whether the number is a genuine out-of-sample one. It is
    a field rather than a comment because the two rows that are NOT
    leave-one-draft-out -- the market baseline, which fits nothing, and the
    shipped prior, which was fitted elsewhere -- are exactly the rows where a
    reader might otherwise assume folds that were never run.
    """

    def __init__(self, label: str, report: dict, heldout: str):
        self.label = label
        self.report = report
        self.heldout = heldout

    @property
    def top1(self) -> float:
        return self.report["top1"]


def ladder(X_list, chosen, groups, boards, buckets=None) -> list:
    """Rungs 1 to 4, scored on the same human picks in the same rooms.

    Rung 1, the ADP baseline, is what the market alone predicts. A uniform
    distribution is not a baseline -- it would give the board's #1 and its
    #250 equal probability, so "beats the market" would be true by
    construction. `fit_prior.adp_baseline` builds the real one, off the same
    stored `market_rank` the model reads, so a better model is never confused
    with a worse yardstick.

    Rung 2, the shipped prior, is scored FULL WIDTH -- every column
    `feature_matrix` produces, against the whole of `COLD_START_PRIOR`. That
    is what "the shipped prior" means: the vector exactly as it ships, whose
    unmeasured columns carry 0.0 and therefore contribute nothing. Slicing it
    down to rung 3's 23 would score a vector nobody serves. It is scored with
    no folds and no refit, which is not a head start: it was fitted by
    `make fit-prior` under a DIFFERENT filter, so `main` prints exactly how
    many of these drafts it has already seen and re-runs the comparison
    without them.

    Rung 3 is leave-one-draft-out over `SHIPPED_23_FEATURES` -- pinned by
    name, see that constant -- trained on human picks only. The fold is the
    DRAFT because that is the unit of independence: one room, one board, eight
    strangers, and a random split over picks would put the first half of a
    draft in training and the second half in test, where the model has already
    been told who is gone.

    Rung 4 is the same fit over `RUNG_4_FEATURES`, which is rung 3's 23 plus
    the six Tier 1 pool signals. Same folds, same picks, same rooms, one
    difference: six more columns. That is the entire content of the rung, and
    it is why the two rows are directly subtractable -- `main` pairs their
    per-draft top-1 rather than comparing two standard errors, so everything
    that makes one room harder than another cancels.

    Rung 5 appends here. Nothing above is specific to a feature count --
    `keep_idx` is `leave_one_draft_out`'s own argument and a rung is one more
    call naming more columns.
    """
    return [
        Rung("ADP baseline", fp.adp_baseline(boards, chosen, groups=groups),
             "no fit"),
        Rung("shipped prior", fp.score(COLD_START_PRIOR, X_list, chosen,
                                       groups=groups, buckets=buckets),
             "fitted elsewhere"),
        Rung("human-only refit", fp.leave_one_draft_out(
            X_list, chosen, groups, buckets=buckets,
            keep_idx=fp.feature_indices(SHIPPED_23_FEATURES)),
            "leave-one-draft-out"),
        Rung("+ Tier 1", fp.leave_one_draft_out(
            X_list, chosen, groups, buckets=buckets,
            keep_idx=fp.feature_indices(RUNG_4_FEATURES)),
            "leave-one-draft-out"),
    ]


def ablation(X_list, chosen, groups, features, candidates, full=None,
             on_row=None):
    """Each candidate's own `delta_top1`, with the standard error of it.

    One row per candidate: refit the whole ladder rung WITHOUT that column and
    subtract. `delta_top1` reads as "held-out human top-1 the full model has
    that the model without this feature does not". Positive earns a place; at
    or below zero is cut, and a cut feature's coefficient is 0.0 rather than
    fitted-and-small (`fp.full_beta`).

    WHY THIS IS NOT `fit_prior.ablate`, WHICH ALREADY DOES THE FIRST HALF OF
    IT. That function returns the delta and not the uncertainty of the delta,
    and on this sample the uncertainty is the whole question. Rung 3 beat the
    shipped prior by +0.0167 against a paired standard error of 0.0080, so
    roughly two errors; an individual feature is expected to be worth a
    fraction of that, and `last_of_tier` fires on well under one per cent of
    candidates. A per-feature delta printed alone would be read as a
    measurement at a precision it does not have. `fp.paired_se` over the two
    per-draft series is the honest scale, and it is available for free because
    both sides scored the same picks in the same rooms.

    `candidates` IS REQUIRED AND HAS NO DEFAULT. `fit_prior.ablate` defaults
    to `UNMEASURED_FEATURES` and `draft_model.ablation` defaults to
    `_NEW_FEATURES`; those are two different sets, one of which contains none
    of the columns this rung exists to test, and the failure is silent -- a
    full table of plausible deltas that measured the wrong thing. There is no
    default here that could be wrong. See `ABLATION_CANDIDATES`.

    `features` is the FULL model the deltas are taken against, so it is passed
    rather than assumed to be `FEATURE_NAMES`. `fp.ablate` assumes that, and
    the assumption is true today only because rung 4 happens to be every
    column that exists; a rung 5 column would widen its "full" model out from
    under the row labelled rung 4.

    `full` is that model's already-computed report. Each row here is a
    complete leave-one-draft-out backtest -- forty-odd fits -- so recomputing
    the row that `ladder` just produced would be minutes of work to arrive at
    a number already in hand.

    `on_row` IS CALLED AS EACH ROW LANDS, and it exists because this loop is
    the long pole: about eight minutes per candidate on the real corpus, and
    the first attempt at this rung was killed by the OS partway through with
    every measured number still buffered inside an unreturned DataFrame. A
    row printed when it is computed is a row that survives the run being
    killed; a table printed at the end is not. `main` passes the printer.
    """
    features, candidates = list(features), list(candidates)
    unknown = [c for c in candidates if c not in set(features)]
    if unknown:
        raise ValueError(
            f"cannot ablate {unknown}: not in the model being ablated. A "
            "candidate outside `features` would be 'dropped' from a fit that "
            "never had it, and its delta would come out at exactly 0.0 -- "
            "indistinguishable from a feature measured and found worthless")
    if full is None:
        full = fp.leave_one_draft_out(X_list, chosen, groups,
                                      keep_idx=fp.feature_indices(features))
    rows = [{"dropped": "none", "top1": full["top1"], "top5": full["top5"],
             "logloss": full["logloss"], "delta_top1": 0.0,
             "delta_top1_se": 0.0, "delta_top5": 0.0}]
    if on_row is not None:
        on_row(rows[0])
    for feature in candidates:
        keep = [f for f in features if f != feature]
        cut = fp.leave_one_draft_out(X_list, chosen, groups,
                                     keep_idx=fp.feature_indices(keep))
        rows.append({
            "dropped": feature, "top1": cut["top1"], "top5": cut["top5"],
            "logloss": cut["logloss"],
            "delta_top1": full["top1"] - cut["top1"],
            "delta_top1_se": fp.paired_se(full["by_draft"], cut["by_draft"]),
            "delta_top5": full["top5"] - cut["top5"]})
        if on_row is not None:
            on_row(rows[-1])
    return pd.DataFrame(rows)


def _print_ablation_row(row) -> None:
    """One ablation row, printed the moment it is measured. See `on_row`."""
    print(f"  {row['dropped']:<18} top-1 {row['top1']:.4f}  "
          f"top-5 {row['top5']:.4f}  log-loss {row['logloss']:>7.4f}  "
          f"delta_top1 {row['delta_top1']:+.4f} +/-{row['delta_top1_se']:.4f}",
          flush=True)


def tier1_winners(table: pd.DataFrame, candidates=None) -> list:
    """The Tier 1 features that earned a place: `delta_top1` strictly above 0.

    `candidates` defaults to `TIER1_FEATURES` and that default is the point,
    because it is the one place the two halves of the table are told apart.
    The ablation reports all fourteen `ABLATION_CANDIDATES`; only the six are
    allowed to change what ships. The other eight are rung 3's, pinned by name
    in `SHIPPED_23_FEATURES`, and cutting one of them here would fold "an
    older column stopped paying" into a rung whose headline is "the six pool
    signals paid" -- two effects in one number, which is what the ladder is
    for separating.

    Strictly above zero, not "above zero minus a standard error" and not
    "above zero or close". A feature whose delta is inside its own error has
    not been shown to be worth anything, and the tie-break that matters is the
    one that leaves it out: `COLD_START_PRIOR` already carries five zeros that
    mean MEASURED AND REJECTED, and a sixth is a cheaper mistake than a
    coefficient fitted on noise and served on every pick of every rollout.
    """
    allowed = set(TIER1_FEATURES if candidates is None else candidates)
    kept = table[(table["dropped"] != "none")
                 & (table["delta_top1"] > 0)]["dropped"].tolist()
    return [f for f in TIER1_FEATURES if f in set(kept) & allowed]


def column_coverage(X_list, names) -> pd.DataFrame:
    """How often each column is non-zero, and how often it varies at all.

    Printed beside the ablation because the two answer different halves of
    "did this feature have a chance". A conditional logit sees a column ONLY
    through its variation inside a choice set: a column holding one value for
    every candidate divides straight back out of the softmax, so `sets_varying`
    below is the fraction of picks at which the feature could have moved the
    answer even in principle. `nonzero_share` is the fraction of candidate
    ROWS it fires on at all, which is the number that makes `last_of_tier`
    interpretable -- a feature true of a fraction of a per cent of candidates
    has a delta whose sampling error swamps it, and the honest report of such
    a delta says so rather than quoting three decimal places.

    Indexed off `FEATURE_NAMES` because the design matrices are full width:
    `fp.design` calls `feature_matrix`, which produces every column, and the
    rungs slice with `keep_idx` afterwards.
    """
    idx = [FEATURE_NAMES.index(name) for name in names]
    rows, n_sets = [], max(len(X_list), 1)
    stacked = (np.vstack([X[:, idx] for X in X_list]) if X_list
               else np.zeros((0, len(idx))))
    varying = np.zeros(len(idx))
    for X in X_list:
        block = X[:, idx]
        varying += (block.max(axis=0) != block.min(axis=0)).astype(float)
    for j, name in enumerate(names):
        column = stacked[:, j]
        rows.append({
            "feature": name,
            "nonzero_share": float((column != 0.0).mean()) if len(column) else 0.0,
            "sets_varying": float(varying[j] / n_sets),
            "std": float(column.std()) if len(column) else 0.0})
    return pd.DataFrame(rows)


def restricted_top1(report: dict, keep_drafts) -> tuple:
    """That report's top-1 over `keep_drafts` only, and how many picks that is.

    Used for one thing: the shipped prior was fitted on some of the drafts it
    is being scored on here, so its rung-2 number is partly in-sample. The
    honest check is to recompute both rows over the drafts it never saw.

    Weighted by each draft's own pick count, which is why `by_draft_n` exists.
    A plain mean of per-draft rates would let a draft that contributed ten
    human picks count as much as one that contributed a hundred and nine, and
    the restricted subset is exactly where those sizes differ most.
    """
    by_draft, sizes = report.get("by_draft", {}), report.get("by_draft_n", {})
    keep = [g for g in keep_drafts if g in by_draft and sizes.get(g)]
    n = sum(sizes[g] for g in keep)
    if not n:
        return float("nan"), 0
    return sum(by_draft[g] * sizes[g] for g in keep) / n, n


# ---------------------------------------------------------------- the module

_MODULE_TEMPLATE = '''"""The human-only cold-start prior. GENERATED -- do not edit by hand.

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

{provenance_prose}
"""
import numpy as np

# What was fitted, when, against what, and what it beat. Machine-readable so a
# reader does not have to trust the prose above and a test can assert the two
# agree.
PROVENANCE = {provenance_dict}

# The feature order these coefficients were fitted in, written out rather than
# assumed. A coefficient vector applied to columns in a different order than
# it was fitted in is silently wrong on every pick, and a length check cannot
# see it.
FEATURES = {features}

# The drafts this was fitted and scored on: every draft in the corpus carrying
# at least one KNOWN human pick, snapshotted at the top of the run. Recorded
# because the corpus GROWS -- a farm loop adds roughly one draft every fifteen
# minutes -- so "the corpus" is not a reproducible input without the list.
DRAFT_IDS = {draft_ids}

PRIOR = np.array([
{coefficients}])

assert len(PRIOR) == len(FEATURES), (
    "the generated human prior and its feature list disagree -- regenerate "
    "with `make score-ladder` rather than editing either by hand")
'''


def render_module(beta, features, draft_ids, provenance,
                  provenance_prose: str, fitted=None, cut=()) -> str:
    """The text of `scoring/human_prior.py`.

    One coefficient per line with its feature name beside it, in
    `FEATURE_NAMES` order, for the same reason `fit_prior.render_module` does
    it: the only thing anyone ever wants from a generated coefficient file is
    to read what a named coefficient came out as, and a diff between two
    generations should show which coefficients moved rather than one very long
    line.

    THREE THINGS A 0.0 CAN MEAN, AND ONLY A LIST CAN TELL THEM APART. The
    value is the same in all three cases and reading the number can never
    distinguish them, so the annotation is driven entirely by which list the
    name is on and never by the coefficient:

        in `fitted`      the fit had this column and this is what came out.
                         A 0.0 here means the column genuinely bought nothing.
        in `cut`         the column was offered to the ablation, measured on
                         human picks, and its own `delta_top1` was at or below
                         zero. MEASURED AND REJECTED.
        neither          the column exists in `FEATURE_NAMES` and this rung
                         never asked about it. NOT MEASURED, which is not the
                         same claim as rejected and must not read as one.

    Rung 3 had only the first and third: it is defined as "the current 23
    features refitted on humans", selects nothing, and cutting features there
    as well would have confounded the population effect with a feature effect.
    Rung 4 is where selection lives, so `cut` arrives with it.
    """
    fitted, cut = set(features if fitted is None else fitted), set(cut)
    overlap = fitted & cut
    assert not overlap, (
        f"{sorted(overlap)} is annotated both fitted and cut; one of the two "
        "lists is wrong and the file would claim a coefficient was rejected")
    width = max(len(name) for name in features)
    lines = []
    for name, value in zip(features, beta):
        if name in fitted:
            note = ""
        elif name in cut:
            note = "   <- measured on humans, cut on delta_top1"
        else:
            note = "   <- not measured by this rung, not fitted"
        lines.append(f"    {value:+.6f},  # {name:<{width}}{note}".rstrip())
    return _MODULE_TEMPLATE.format(
        provenance_prose=provenance_prose,
        provenance_dict=fp._pretty_dict(provenance),
        features=fp._pretty_list(features),
        draft_ids=fp._pretty_list(draft_ids),
        coefficients="\n".join(lines) + "\n")


def write_human_prior(beta, features, draft_ids, provenance, provenance_prose,
                      path: Path | None = None, fitted=None, cut=()) -> Path:
    """Write the generated module. Callers must have checked the rule first.

    Deliberately dumb: it writes what it is given. The decision lives in
    `main`, in one place, next to the numbers that make it -- not in a flag on
    a writer, where a future caller could pass a different one.
    """
    target = Path(path or HUMAN_PRIOR_MODULE)
    target.write_text(render_module(beta, features, draft_ids, provenance,
                                    provenance_prose, fitted=fitted, cut=cut))
    return target


# ------------------------------------------------------------------ the run

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _prose(rungs, shipped_rung, table, keep, delta, se, draft_ids,
           corpus_path) -> str:
    market, shipped, refit, tier1 = rungs
    cut = [f for f in TIER1_FEATURES if f not in set(keep)]
    per_feature = "\n".join(
        f"    {row.dropped:<18} delta_top1 {row.delta_top1:+.4f} "
        f"+/-{row.delta_top1_se:.4f}   {'kept' if row.dropped in set(keep) else 'cut'}"
        for row in table.itertuples()
        if row.dropped in set(TIER1_FEATURES))
    return f"""
Fitted on {tier1.report['n']} picks a person is KNOWN to have made
(`autodrafted IS FALSE`, at a seat that is not ours), across {len(draft_ids)} drafts of
the cross-league draft corpus ({corpus_path}), on {_now()}.

Leave-one-draft-out: every draft held out in turn, the pooled fit rebuilt on
the other N-1 drafts' human picks, and the held-out draft's human picks scored
by a vector that never saw them. The fold is the draft because that is the
unit of independence -- one room, one board, eight strangers.

    ADP baseline       top-1 {market.top1:.4f}  top-5 {market.report['top5']:.4f}  log-loss {market.report['logloss']:>7.4f}
    shipped prior      top-1 {shipped.top1:.4f}  top-5 {shipped.report['top5']:.4f}  log-loss {shipped.report['logloss']:>7.4f}
    human-only refit   top-1 {refit.top1:.4f}  top-5 {refit.report['top5']:.4f}  log-loss {refit.report['logloss']:>7.4f}
    + Tier 1, all six  top-1 {tier1.top1:.4f}  top-5 {tier1.report['top5']:.4f}  log-loss {tier1.report['logloss']:>7.4f}
    + Tier 1, shipped  top-1 {shipped_rung.top1:.4f}  top-5 {shipped_rung.report['top5']:.4f}  log-loss {shipped_rung.report['logloss']:>7.4f}

delta_top1 of the shipped set against the human-only refit: {delta:+.4f} +/-{se:.4f},
paired by draft. top-1 is the decision metric and is the one this had to win
on; the other two are recorded because they were measured, not because they
decided.

EACH TIER 1 FEATURE, DROPPED FROM THE FULL MODEL IN TURN. Positive earns a
place; at or below zero is 0.0 above, meaning MEASURED AND REJECTED. The +/-
is the standard error of the delta itself, paired by draft, and a delta
inside it has not been shown to be worth anything either way.

{per_feature}

    kept: {', '.join(keep) or '(none)'}
    cut:  {', '.join(cut) or '(none)'}

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
""".strip("\n")


def _print_rung(rung) -> None:
    report = rung.report
    se = (f"  +/-{fp.cluster_se(report['by_draft']):.4f}"
          if report.get("by_draft") else "")
    print(f"  {rung.label:<18} top-1 {report['top1']:.4f}  "
          f"top-5 {report['top5']:.4f}  log-loss {report['logloss']:>7.4f}"
          f"{se}  ({rung.heldout})")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    corpus_path = fp._option(argv, "--corpus") or dl.CORPUS_PATH
    league_path = fp._option(argv, "--league-db")

    corpus = fp.open_corpus(corpus_path)
    league_conn = fp.open_league(league_path)
    try:
        snapshot = fp.snapshot_draft_ids(corpus)
        draft_ids = labelled_draft_ids(corpus, snapshot)
        if not draft_ids:
            print("No draft in the corpus carries a pick labelled "
                  "`autodrafted IS FALSE`. There is nothing human to measure "
                  "on -- run `make farm-mocks`, which records the label.")
            return 1
        print(f"Corpus snapshot: {len(snapshot)} drafts with a pool, "
              f"{len(draft_ids)} of them carrying at least one KNOWN human "
              "pick.\nThe farm writes to this database while this runs, so "
              "the list is fixed here and\nprinted so the run can be "
              "reproduced exactly:\n  " + "\n  ".join(draft_ids))

        corpus_obs = human_observations(corpus, league_conn, draft_ids)
        X_list, chosen, groups, boards, buckets = fp.design(corpus_obs)
        print(f"\n{corpus_obs.picks_seen} picks in those drafts; "
              f"{len(chosen)} are known human picks at a seat that is not "
              "ours. Excluded:")
        for reason, n in corpus_obs.dropped.items():
            print(f"  {reason:<14} {n}")
        print(f"  ({NOT_HUMAN} is `autodrafted IS TRUE` plus `IS NULL`. NULL "
              "is unknown, and for a\n   MEASUREMENT unknown is not human -- "
              "see this module's docstring. The fit\n   rule in "
              "`make fit-prior` keeps those picks; this one cannot.)")
        if not chosen:
            print("No human picks. Nothing to measure.")
            return 1

        rungs = ladder(X_list, chosen, groups, boards, buckets=buckets)
        market, shipped, refit, tier1 = rungs
        print("\nHeld-out human top-1. Every rung is scored on the same "
              f"{len(chosen)} picks in the same\n{len(set(groups))} rooms. "
              "The +/- is clustered by draft: picks inside one room share a\n"
              "board, a set of opponents and everything already taken, so the "
              "binomial\nstandard error over 2,000 picks understates the real "
              "uncertainty by half.")
        for rung in rungs:
            _print_rung(rung)

        print("\nEach delta is paired by draft -- both sides scored the same "
              "picks in the same\nrooms, so what makes one room harder than "
              "another cancels in the difference.")
        for lower, upper in zip(rungs, rungs[1:]):
            delta = upper.top1 - lower.top1
            se = fp.paired_se(upper.report["by_draft"], lower.report["by_draft"])
            print(f"  {upper.label:<18} - {lower.label:<18} "
                  f"delta_top1 {delta:+.4f} +/-{se:.4f}")

        _print_contamination(rungs, draft_ids)
        _print_buckets(refit, tier1)

        # THE CHAIN. Every rung must beat the one below it, and rung 3 is
        # asserted here rather than assumed because the corpus GROWS: the
        # rung 3 that shipped was measured on 38 drafts and this run may hold
        # more, so "rung 3 already won once" is not a statement about the
        # numbers printed above.
        refit_delta = refit.top1 - shipped.top1
        refit_se = fp.paired_se(refit.report["by_draft"],
                                shipped.report["by_draft"])
        if refit_delta <= 0:
            print(f"\nRung 3 does NOT beat the shipped prior on held-out "
                  f"human top-1 ({refit_delta:+.4f}\n+/-{refit_se:.4f}) on "
                  "this snapshot. Nothing is written, and rung 4 is not "
                  "read:\na rung measured against a rung that lost is not a "
                  "measurement of anything.")
            return 1

        # WHICH of the six did anything, MEASURED BEFORE THE RUNG IS JUDGED
        # AND WHETHER OR NOT IT WINS. The ablation is diagnostic, not
        # decisional: it says which column was carrying the rung, and that is
        # the entire content of a negative result. "The six lost" is barely a
        # finding; "the six lost, five of them do nothing at all and the sixth
        # is worth +0.004 +/-0.003" says what to try next and what not to.
        # Running it only on the winning branch would mean the losing branch
        # -- the one the design document says to expect -- reports the least.
        #
        # `ABLATION_CANDIDATES`, passed explicitly, because the two ablation
        # functions in this codebase default to two different sets and one of
        # them contains none of the six -- see that constant. `full=` is
        # rung 4's own report so the row `ladder` already computed is not
        # computed a second time.
        print(f"\nPer-feature ablation over the {len(ABLATION_CANDIDATES)} "
              "Tier 1 pool signals -- the columns\nthis rung ADDS, and the "
              "only ones allowed to change what ships (see\n"
              "`ABLATION_CANDIDATES` for why rung 3's own 23 are not "
              "re-litigated here).\nEach row is the full rung-4 model refit "
              "without that one column, which is a\ncomplete "
              "leave-one-draft-out backtest and takes minutes; rows print as "
              "they\nland. delta_top1 > 0 earns a place; the +/- is the "
              "standard error of the delta\nitself, paired by draft, and a "
              "delta inside it is an unmeasured quantity\nrather than a small "
              "win.")
        table = ablation(X_list, chosen, groups, RUNG_4_FEATURES,
                         ABLATION_CANDIDATES, full=tier1.report,
                         on_row=_print_ablation_row)

        print("\nHow much of a chance each of the six had. `sets_varying` is "
              "the fraction of\nchoice sets where the column is not constant "
              "-- a constant column divides out\nof the softmax exactly, so "
              "that is the ceiling on how often it could have\nmattered. "
              "`nonzero_share` is the fraction of candidate ROWS it fires on "
              "at all.")
        print(column_coverage(X_list, TIER1_FEATURES).to_string(index=False))

        # ONLY THE SIX MAY CHANGE WHAT SHIPS. The other eight rows above are
        # reported, not acted on: rung 3 is pinned to all 23 by name, and
        # cutting one of them inside the Tier 1 rung would put "an older
        # column stopped paying" and "the pool signals paid" into one number.
        keep = tier1_winners(table)
        cut = [f for f in TIER1_FEATURES if f not in set(keep)]
        shipped_features = SHIPPED_23_FEATURES + keep
        print(f"\nOf the six, {len(keep)} earned a place on delta_top1 and "
              f"{len(cut)} did not.\n  kept: {', '.join(keep) or '(none)'}"
              f"\n  cut:  {', '.join(cut) or '(none)'}")

        # THE RULE FOR THIS RUNG. delta_top1 against rung 3, and nothing else.
        # Written as one expression, once, so that changing it is a one-line
        # diff somebody has to justify rather than a flag somebody can pass.
        full_delta = tier1.top1 - refit.top1
        full_se = fp.paired_se(tier1.report["by_draft"],
                               refit.report["by_draft"])
        if full_delta <= 0:
            print(f"\nRung 4 does NOT beat rung 3 on held-out human top-1 "
                  f"({full_delta:+.4f} +/-{full_se:.4f}).\nThe six Tier 1 "
                  "features, together, do not predict a person better than "
                  "the 23\nthat were already shipping. Nothing is written; "
                  f"{HUMAN_PRIOR_MODULE.name} still holds\nrung 3. A negative "
                  "result is a complete result -- write it up rather than\n"
                  "tuning until something wins. The per-feature table above "
                  "is that write-up's\ncontent: it says which column, if any, "
                  "was doing anything at all.")
            return 1

        # The configuration that would actually be WRITTEN, scored on the same
        # folds. Not the same row as rung 4 unless every candidate won, and
        # the difference is the whole reason it is re-run: the file carries
        # the selected subset, so the number that justifies the file has to be
        # the subset's number and not the full set's.
        shipped_rung = Rung(
            "+ Tier 1, shipped",
            fp.leave_one_draft_out(X_list, chosen, groups, buckets=buckets,
                                   keep_idx=fp.feature_indices(shipped_features)),
            "leave-one-draft-out")
        _print_rung(shipped_rung)

        delta = shipped_rung.top1 - refit.top1
        se = fp.paired_se(shipped_rung.report["by_draft"],
                          refit.report["by_draft"])
        print(f"\ndelta_top1 (shipped subset - rung 3): {delta:+.4f} "
              f"+/-{se:.4f}, paired by draft.")
        if delta <= 0:
            print("\nThe subset that survived the per-feature cut does NOT "
                  "beat rung 3, even though\nthe full six did. Nothing is "
                  f"written; {HUMAN_PRIOR_MODULE.name} still holds rung 3. "
                  "The\nartifact carries the SUBSET, so the subset is what "
                  "has to have won -- shipping\non the full set's number "
                  "would put a figure in that file that was measured\nagainst "
                  "a different set of columns.")
            return 1

        # The pooled vector over every labelled draft, fitted on the shipped
        # columns ONLY and scattered back to full length with 0.0 elsewhere.
        # `full_beta` and not `fit`, for the reason it documents: fitting all
        # of `FEATURE_NAMES` and zeroing the cut ones afterwards would ship
        # coefficients fitted in the presence of columns this model does not
        # have, each carrying mass the others were splitting with it.
        beta = fp.full_beta(X_list, chosen, shipped_features)
        provenance = {
            "fitted_at": _now(),
            "corpus": corpus_path,
            "population": "autodrafted IS FALSE (known human), not our seat",
            "drafts": len(draft_ids),
            "picks_scored": len(chosen),
            "picks_excluded": dict(corpus_obs.dropped),
            "folds": "leave-one-draft-out",
            "rung": "4 (+ Tier 1 pool signals)",
            "features_fitted": list(shipped_features),
            "tier1_kept": list(keep),
            "tier1_cut": list(cut),
            "tier1_delta_top1": {
                row.dropped: round(row.delta_top1, 6)
                for row in table.itertuples()
                if row.dropped in set(TIER1_FEATURES)},
            "tier1_delta_top1_se": {
                row.dropped: round(row.delta_top1_se, 6)
                for row in table.itertuples()
                if row.dropped in set(TIER1_FEATURES)},
            # The join those four columns arrived by, recorded in the artifact
            # rather than only in a findings document, because a coefficient
            # is read here and the caveat has to travel with it. See this
            # module's docstring.
            "board_joined_from": "today's build_board, not attributes_as_of "
                                 "(vor, durability, proj_change, tier)",
            "adp_top1": round(market.top1, 6),
            "shipped_top1": round(shipped.top1, 6),
            # WHICH shipped prior. `scoring/mock_prior.py` is regenerated by
            # `make fit-prior` whenever that fit wins, so "the shipped prior
            # scored 0.2184" is not a reproducible statement without naming
            # the generation it refers to.
            "shipped_prior_fitted_at": getattr(
                mock_prior, "PROVENANCE", {}).get("fitted_at"),
            "shipped_prior_drafts_also_scored_here": len(
                set(getattr(mock_prior, "DRAFT_IDS", [])) & set(draft_ids)),
            "rung3_top1": round(refit.top1, 6),
            "rung4_all_six_top1": round(tier1.top1, 6),
            "refit_top1": round(shipped_rung.top1, 6),
            "refit_top5": round(shipped_rung.report["top5"], 6),
            "refit_logloss": round(shipped_rung.report["logloss"], 6),
            "delta_top1": round(delta, 6),
            "delta_top1_se": round(se, 6),
        }
        target = write_human_prior(
            beta, FEATURE_NAMES, draft_ids, provenance,
            _prose(rungs, shipped_rung, table, keep, delta, se, draft_ids,
                   corpus_path),
            fitted=shipped_features, cut=cut)
        print(f"\nRung 4 wins on held-out human top-1 ({delta:+.4f} "
              f"+/-{se:.4f}). Wrote {target}.\nNothing imports it yet -- see "
              "that file's docstring for why the serving swap\nwaits for the "
              "end of the ladder.")
        return 0
    finally:
        corpus.close()
        league_conn.close()


def _print_contamination(rungs, draft_ids) -> None:
    """How much of rung 2's number is in-sample, and what it is without it.

    The shipped prior was written by `make fit-prior`, which fits on this same
    corpus under the fit filter. Any draft in both `mock_prior.DRAFT_IDS` and
    this run's snapshot is a draft rung 2 has already been trained on, and
    every pick of it flatters rung 2 -- which biases the comparison AGAINST
    rung 3, not for it. That direction is worth stating either way: a win here
    is conservative, and a loss would need this checked before it was
    believed.

    Reported rather than fixed by dropping the drafts. They are real held-out
    picks for rung 3 and throwing them away would shrink the sample to make a
    caveat go away.
    """
    seen = sorted(set(getattr(mock_prior, "DRAFT_IDS", [])) & set(draft_ids))
    shipped = rungs[1]
    if not seen:
        print("\nNone of these drafts appear in `mock_prior.DRAFT_IDS`, so "
              "the shipped prior\nhas never been fitted on one pick it is "
              "scored on above.")
        return
    clean = [g for g in draft_ids if g not in set(seen)]
    ship_top1, n = restricted_top1(shipped.report, clean)
    print(f"\nIn-sample check: {len(seen)} of these {len(draft_ids)} drafts "
          "were in the shipped prior's own\ntraining set "
          "(`mock_prior.DRAFT_IDS`), so part of its row above is not held out "
          "for\nit. That flatters the shipped prior and makes every delta "
          f"above it conservative.\nOver the {len(clean)} drafts it has never "
          f"seen ({n} human picks):")
    # Every fitted rung, not just rung 3. Rung 4 is compared against rung 3
    # and neither of them has a contamination problem -- both are refitted per
    # fold on this corpus -- but the row is printed for all of them so the
    # restricted column can be read straight down against the full one rather
    # than a reader having to hold two tables of different shape in mind.
    print(f"  {shipped.label:<18} top-1 {ship_top1:.4f}")
    for rung in rungs[2:]:
        top1, _ = restricted_top1(rung.report, clean)
        print(f"  {rung.label:<18} top-1 {top1:.4f}   "
              f"delta_top1 vs shipped {top1 - ship_top1:+.4f}")


def _print_buckets(lower, upper) -> None:
    """WHERE the accuracy moved, not just whether it did.

    The 2026-08-11 finding turned on exactly this table: a change that wins
    only in the rounds where a kicker and a defense have to be taken has won
    something much smaller than its headline number says. Same buckets
    `draft_model` draws (early 1-3, mid 4-8, late 9+) rather than a second set
    of boundaries, because this table gets read next to that backtest.

    Takes the two rungs being DECIDED between rather than a fixed pair, and
    labels its columns from them. Rung 4's question is what the six pool
    signals bought on top of rung 3, so rung 3 is the row it has to be laid
    against; against the shipped prior it would show the population effect and
    the feature effect added together, which is the confound the ladder is
    built to take apart.
    """
    a_all, b_all = lower.report.get("by_bucket"), upper.report.get("by_bucket")
    if not a_all or not b_all:
        return
    a_top1, b_top1 = f"{lower.label} top-1", f"{upper.label} top-1"
    a_top5, b_top5 = f"{lower.label} top-5", f"{upper.label} top-5"
    print(f"\nBy round bucket (early 1-3, mid 4-8, late 9+), {lower.label} "
          f"against\n{upper.label}, on the same held-out human picks:")
    print(f"  {'bucket':<7} {'n':>5}  {a_top1:>24} {b_top1:>18}"
          f"  {a_top5:>24} {b_top5:>18}")
    for bucket in _ROUND_BUCKET_ORDER:
        a, b = a_all.get(bucket), b_all.get(bucket)
        if not a or not b:
            continue
        print(f"  {bucket:<7} {a[2]:>5}  {a[0] / a[2]:>24.4f} "
              f"{b[0] / b[2]:>18.4f}  {a[1] / a[2]:>24.4f} "
              f"{b[1] / b[2]:>18.4f}")


if __name__ == "__main__":
    sys.exit(main())
