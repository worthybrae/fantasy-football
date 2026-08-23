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
    4  + Tier 1 features      another agent's rung
    5  + mixture, best K      another agent's rung

Rungs 4 and 5 are not here. `human_observations` and `ladder` are written to
be extended with them rather than copied.

THE RULE, WHICH IS `fit_prior`'S RULE AND NOT THIS MODULE'S TO BEND.
`delta_top1` decides, against the rung below. Top-5 and log-loss are printed
for every rung, always, so neither can be reached for only when it flatters
the answer. There is deliberately NO force flag: rung 3 writes its artifact if
and only if it beats the shipped prior on held-out human top-1, and a losing
rung prints its numbers, writes nothing, and exits 1.

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
    """Rungs 1 to 3, scored on the same human picks in the same rooms.

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

    Rungs 4 and 5 append here. Nothing above is specific to 23 features --
    `keep_idx` is `leave_one_draft_out`'s own argument and a Tier 1 rung is
    one more call naming more columns.
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
    ]


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
                  provenance_prose: str, fitted=None) -> str:
    """The text of `scoring/human_prior.py`.

    One coefficient per line with its feature name beside it, in
    `FEATURE_NAMES` order, for the same reason `fit_prior.render_module` does
    it: the only thing anyone ever wants from a generated coefficient file is
    to read what a named coefficient came out as, and a diff between two
    generations should show which coefficients moved rather than one very long
    line.

    No cut list, unlike the mock prior's writer, because rung 3 selects
    nothing. It is defined as "the current 23 features, refitted on humans" --
    it isolates the effect of the POPULATION and nothing else, and cutting
    features here as well would confound the two, which is the exact confound
    the ladder exists to prevent. Feature selection is Tier 1's rung.

    `fitted` is the set of columns this fit actually had. A column outside it
    carries 0.0, and that 0.0 is annotated NOT FITTED HERE -- a different fact
    from a coefficient that came out at zero, and the only place the two can
    be told apart is a list, never the value.
    """
    fitted = set(features if fitted is None else fitted)
    width = max(len(name) for name in features)
    lines = []
    for name, value in zip(features, beta):
        note = "" if name in fitted else "   <- not in rung 3, not fitted"
        lines.append(f"    {value:+.6f},  # {name:<{width}}{note}".rstrip())
    return _MODULE_TEMPLATE.format(
        provenance_prose=provenance_prose,
        provenance_dict=fp._pretty_dict(provenance),
        features=fp._pretty_list(features),
        draft_ids=fp._pretty_list(draft_ids),
        coefficients="\n".join(lines) + "\n")


def write_human_prior(beta, features, draft_ids, provenance, provenance_prose,
                      path: Path | None = None, fitted=None) -> Path:
    """Write the generated module. Callers must have checked the rule first.

    Deliberately dumb: it writes what it is given. The decision lives in
    `main`, in one place, next to the numbers that make it -- not in a flag on
    a writer, where a future caller could pass a different one.
    """
    target = Path(path or HUMAN_PRIOR_MODULE)
    target.write_text(render_module(beta, features, draft_ids, provenance,
                                    provenance_prose, fitted=fitted))
    return target


# ------------------------------------------------------------------ the run

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _prose(rungs, delta, se, draft_ids, corpus_path) -> str:
    market, shipped, refit = rungs
    return f"""
Fitted on {refit.report['n']} picks a person is KNOWN to have made
(`autodrafted IS FALSE`, at a seat that is not ours), across {len(draft_ids)} drafts of
the cross-league draft corpus ({corpus_path}), on {_now()}.

Leave-one-draft-out: every draft held out in turn, the pooled fit rebuilt on
the other N-1 drafts' human picks, and the held-out draft's human picks scored
by a vector that never saw them. The fold is the draft because that is the
unit of independence -- one room, one board, eight strangers.

    ADP baseline       top-1 {market.top1:.4f}  top-5 {market.report['top5']:.4f}  log-loss {market.report['logloss']:>7.4f}
    shipped prior      top-1 {shipped.top1:.4f}  top-5 {shipped.report['top5']:.4f}  log-loss {shipped.report['logloss']:>7.4f}
    human-only refit   top-1 {refit.top1:.4f}  top-5 {refit.report['top5']:.4f}  log-loss {refit.report['logloss']:>7.4f}

delta_top1 against the shipped prior: {delta:+.4f} +/-{se:.4f}, paired by draft.
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
        market, shipped, refit = rungs
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
        _print_buckets(shipped, refit)

        # THE RULE. delta_top1 against the rung below, and nothing else. One
        # expression, once, so that changing it is a one-line diff somebody
        # has to justify rather than a flag somebody can pass.
        delta = refit.top1 - shipped.top1
        se = fp.paired_se(refit.report["by_draft"], shipped.report["by_draft"])
        if delta <= 0:
            print(f"\nRung 3 does NOT beat the shipped prior on held-out "
                  f"human top-1 ({delta:+.4f}\n+/-{se:.4f}). Nothing is "
                  "written. A negative result is a complete result -- write "
                  "it\nup rather than tuning until something wins.")
            return 1

        # The pooled vector over every labelled draft, fitted on rung 3's
        # columns ONLY and scattered back to full length with 0.0 elsewhere.
        # `full_beta` and not `fit`, for the reason it documents: fitting all
        # of `FEATURE_NAMES` and zeroing the extras afterwards would ship
        # coefficients fitted in the presence of columns this model does not
        # have, each carrying mass the others were splitting with it.
        beta = fp.full_beta(X_list, chosen, SHIPPED_23_FEATURES)
        provenance = {
            "fitted_at": _now(),
            "corpus": corpus_path,
            "population": "autodrafted IS FALSE (known human), not our seat",
            "drafts": len(draft_ids),
            "picks_scored": len(chosen),
            "picks_excluded": dict(corpus_obs.dropped),
            "folds": "leave-one-draft-out",
            "features_fitted": list(SHIPPED_23_FEATURES),
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
            "refit_top1": round(refit.top1, 6),
            "refit_top5": round(refit.report["top5"], 6),
            "refit_logloss": round(refit.report["logloss"], 6),
            "delta_top1": round(delta, 6),
            "delta_top1_se": round(se, 6),
        }
        target = write_human_prior(beta, FEATURE_NAMES, draft_ids, provenance,
                                   _prose(rungs, delta, se, draft_ids,
                                          corpus_path),
                                   fitted=SHIPPED_23_FEATURES)
        print(f"\nRung 3 wins on held-out human top-1 ({delta:+.4f} "
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
    market, shipped, refit = rungs
    if not seen:
        print("\nNone of these drafts appear in `mock_prior.DRAFT_IDS`, so "
              "the shipped prior\nhas never been fitted on one pick it is "
              "scored on above.")
        return
    clean = [g for g in draft_ids if g not in set(seen)]
    ship_top1, n = restricted_top1(shipped.report, clean)
    refit_top1, _ = restricted_top1(refit.report, clean)
    print(f"\nIn-sample check: {len(seen)} of these {len(draft_ids)} drafts "
          "were in the shipped prior's own\ntraining set "
          "(`mock_prior.DRAFT_IDS`), so part of its row above is not held out "
          "for\nit. That flatters the shipped prior and makes rung 3's delta "
          f"conservative.\nOver the {len(clean)} drafts it has never seen "
          f"({n} human picks):")
    print(f"  {'shipped prior':<18} top-1 {ship_top1:.4f}")
    print(f"  {'human-only refit':<18} top-1 {refit_top1:.4f}   "
          f"delta_top1 {refit_top1 - ship_top1:+.4f}")


def _print_buckets(shipped, refit) -> None:
    """WHERE the accuracy moved, not just whether it did.

    The 2026-08-11 finding turned on exactly this table: a change that wins
    only in the rounds where a kicker and a defense have to be taken has won
    something much smaller than its headline number says. Same buckets
    `draft_model` draws (early 1-3, mid 4-8, late 9+) rather than a second set
    of boundaries, because this table gets read next to that backtest.
    """
    a_all, b_all = shipped.report.get("by_bucket"), refit.report.get("by_bucket")
    if not a_all or not b_all:
        return
    print("\nBy round bucket (early 1-3, mid 4-8, late 9+), on the same "
          "held-out human picks:")
    print(f"  {'bucket':<7} {'n':>5}  {'shipped top-1':>13} {'refit top-1':>11}"
          f"  {'shipped top-5':>13} {'refit top-5':>11}")
    for bucket in _ROUND_BUCKET_ORDER:
        a, b = a_all.get(bucket), b_all.get(bucket)
        if not a or not b:
            continue
        print(f"  {bucket:<7} {a[2]:>5}  {a[0] / a[2]:>13.4f} "
              f"{b[0] / b[2]:>11.4f}  {a[1] / a[2]:>13.4f} "
              f"{b[1] / b[2]:>11.4f}")


if __name__ == "__main__":
    sys.exit(main())
