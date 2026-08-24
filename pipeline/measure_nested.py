"""Measure the nested (position-then-player) opponent model against the flat prior.

THE DELIVERABLE. `scoring/position_model.py` established that the POSITION a
human takes is predictable at ~0.53 held-out, which puts the ceiling of a
nested top-1 at ~0.29-0.32 -- past the flat baseline's ~0.23 on human picks.
This script BUILDS OUT that ceiling: it fits the nested model
(`scoring/nested_model.py`) leave-one-draft-out on the same human-only corpus
the flat baseline was measured on, scores every held-out pick, and compares --
top-1, top-3, top-5 and log-loss, overall and by round bucket -- against the
SERVED flat cold-start prior (`draft_model.COLD_START_PRIOR`, i.e.
`scoring/mock_prior.PRIOR`) scored on the identical picks. That is the vector
this script actually scores; it is NOT `scoring/human_prior.py` (the human-only
refit, ~0.2595), which is a different artifact and is not the served opponent.

WHY THIS IS AN HONEST COMPARISON. Both models are scored on the SAME held-out
human picks in the SAME rooms, built by the one shared replay
(`score_ladder.human_observations`). The nested model is refit leave-one-
draft-out, so every pick it scores comes from a draft it never trained on. The
flat prior is scored full-width with no fold, exactly as the ladder scores it --
it was fitted elsewhere under a different filter, so it has already seen some
of these picks; that only makes the flat prior's number OPTIMISTIC, so a nested
win over it is if anything understated. Standard errors are clustered by draft;
the head-to-head delta is paired by draft, so everything that makes one room
harder than another cancels.

THE DECISION. If the nested model beats the flat prior on top-1, or ties top-1
and improves log-loss, the decomposition is a genuine advance and this script
says so with the numbers. If it does not, that is a real finding too -- the
factorization did not pay -- and the write-up says that instead. Either way the
result lands in `docs/superpowers/findings/2026-08-24-nested-model.md` AND on
stdout, so a background run whose console is lost still leaves the answer on
disk. This script measures and recommends; it wires nothing into serving.

Run:
    .venv/bin/python -m pipeline.measure_nested \
        [--league-db data/leagues/<id>.duckdb] [--limit N] [--workers W]

`--limit N` scores only the first N drafts (a smoke run that proves the script
end to end). `--workers W` (2..7) fans the leave-one-draft-out folds across a
bounded fork pool; the default is sequential (~8 minutes on the current
corpus), which is what a background run should use unless speed is wanted.
"""
import os
import sys

import numpy as np

import pipeline.fit_prior as fp
from pipeline import score_ladder as sl
from scoring import nested_model as nm
from scoring.draft_model import COLD_START_PRIOR, _softmax

FINDINGS = "docs/superpowers/findings/2026-08-24-nested-model.md"
# The served flat cold-start prior's human-pick top-1, for the headline
# callout. This is `COLD_START_PRIOR` (== `mock_prior.PRIOR`) -- the vector
# `_champion_perpick` below actually scores -- at the 0.2308 `score_ladder`
# documents for the shipped prior on human picks. (Not `human_prior`'s 0.2595;
# that is a different artifact and is not what this script scores.) The run
# also prints the live number for the current snapshot.
FLAT_PRIOR_TOP1 = 0.2308

# The fold pool reads the design off a module global so a forked worker
# inherits it without pickling ~4,600 candidate matrices per task. Set once,
# in the parent, before any worker is forked; never mutated after.
_DESIGN = None
_MAX_WORKERS = 7          # the bound the controller set on this process's forks


# ----------------------------------------------------------------- champion side

def _champion_perpick(X_list, chosen):
    """Per-pick top-1/3/5 hit flags and log-prob for the served flat prior.

    The same softmax over `COLD_START_PRIOR` that `fit_prior.score` runs, but
    returned per pick rather than aggregated, so these can be bucketed by round
    and clustered by draft exactly like the nested model's. Definitions match
    `nested_model.score_pick` (argsort descending, chosen-in-top-k,
    -log p_chosen) so the two columns are read on one ruler.
    """
    n = len(chosen)
    hit1 = np.zeros(n, int); hit3 = np.zeros(n, int)
    hit5 = np.zeros(n, int); ll = np.zeros(n, float)
    for i, (X, k) in enumerate(zip(X_list, chosen)):
        probs = _softmax(X @ COLD_START_PRIOR)
        order = np.argsort(-probs)
        hit1[i] = int(order[0] == k)
        hit3[i] = int(k in order[:3])
        hit5[i] = int(k in order[:5])
        ll[i] = float(np.log(max(probs[k], 1e-12)))
    return hit1, hit3, hit5, ll


# ------------------------------------------------------------------- nested side

def _score_fold(held):
    """Fit the nested model without draft `held`, score that draft's picks.

    Returns the held-out draft's pick indices and their per-pick hit flags and
    log-probs. Reads the design from the module global so it is cheap to run in
    a forked worker; the parent uses it directly too.
    """
    d = _DESIGN
    g = np.asarray(d.groups)
    tr = np.flatnonzero(g != held)
    te = np.flatnonzero(g == held)
    model = nm.NestedModel().fit(d, tr)
    hit1 = np.zeros(len(te), int); hit3 = np.zeros(len(te), int)
    hit5 = np.zeros(len(te), int); ll = np.zeros(len(te), float)
    for j, i in enumerate(te):
        prob = model.predict_pick(d, int(i))
        hit1[j], hit3[j], hit5[j], ll[j] = nm.score_pick(prob, d.chosen[int(i)])
    return te, hit1, hit3, hit5, ll


def _nested_perpick(design, workers=1):
    """Leave-one-draft-out per-pick nested scores over the whole design.

    Sequential by default. With `workers` in 2..7 the folds run in a bounded
    FORK pool -- forked, not spawned, so each worker inherits the already-built
    design from `_DESIGN` rather than re-reading the corpus or pickling the
    matrices. The database connections are closed before the pool is created
    (see `main`), so no worker shares a live DuckDB handle. Results are written
    back into full-length arrays by the returned test indices, so fold order
    does not matter and a parallel run is bit-identical to a sequential one.
    """
    n = len(design)
    hit1 = np.zeros(n, int); hit3 = np.zeros(n, int)
    hit5 = np.zeros(n, int); ll = np.zeros(n, float)
    folds = sorted(set(design.groups))

    if workers and workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(workers, _MAX_WORKERS)) as pool:
            for te, h1, h3, h5, l in pool.imap_unordered(_score_fold, folds):
                hit1[te] = h1; hit3[te] = h3; hit5[te] = h5; ll[te] = l
    else:
        for held in folds:
            te, h1, h3, h5, l = _score_fold(held)
            hit1[te] = h1; hit3[te] = h3; hit5[te] = h5; ll[te] = l
            print(f"  fold {held} scored ({len(te)} picks)", flush=True)
    return hit1, hit3, hit5, ll


# ------------------------------------------------------------------- aggregation

def _cluster_se(hit, groups):
    """Standard error of a rate, each draft one observation. Mirrors
    `fit_prior.cluster_se`: the spread of per-draft rates over sqrt(#drafts),
    because picks inside a room are not independent."""
    per = [hit[groups == g].mean() for g in sorted(set(groups))]
    per = np.asarray(per, dtype=float)
    if len(per) < 2:
        return float("nan")
    return float(per.std(ddof=1) / np.sqrt(len(per)))


def _paired_se(a, b, groups):
    """Standard error of the per-draft DIFFERENCE a-b, paired by draft.
    Mirrors `fit_prior.paired_se`: what makes one room hard hits both models
    and cancels in the difference."""
    gs = sorted(set(groups))
    diffs = np.array([a[groups == g].mean() - b[groups == g].mean() for g in gs])
    if len(diffs) < 2:
        return float("nan")
    return float(diffs.std(ddof=1) / np.sqrt(len(diffs)))


def _by_bucket(hit, buckets):
    out = {}
    b = np.asarray(buckets)
    for name in ("early", "mid", "late"):
        m = b == name
        out[name] = (float(hit[m].mean()) if m.any() else float("nan"),
                     int(m.sum()))
    return out


def _report(design, X_list, chosen, n_hit, workers):
    """Score both models, build the printed report, return it as text."""
    groups = np.asarray(design.groups)
    buckets = design.buckets

    c1, c3, c5, cll = _champion_perpick(X_list, chosen)
    n1, n3, n5, nll = _nested_perpick(design, workers=workers)
    n = len(chosen)

    def line(tag, h1, h3, h5, l):
        return (f"{tag:9s} top1 {h1.mean():6.4f} +/-{_cluster_se(h1, groups):.4f}   "
                f"top3 {h3.mean():6.4f}   top5 {h5.mean():6.4f}   "
                f"logloss {-l.mean():7.4f}")

    lines = [f"drafts {len(set(design.groups))}, human picks {n}\n"]
    lines.append(line("champion", c1, c3, c5, cll))
    lines.append(line("nested", n1, n3, n5, nll))

    d_top1 = n1.mean() - c1.mean()
    d_ll = (-nll.mean()) - (-cll.mean())
    lines.append(
        f"\ndelta top1 (nested - champion): {d_top1:+.4f} "
        f"+/-{_paired_se(n1, c1, groups):.4f}   "
        f"delta logloss: {d_ll:+.4f} (negative = nested better calibrated)")

    lines.append("\nby round bucket (top1 | top5):")
    cb1, cb5 = _by_bucket(c1, buckets), _by_bucket(c5, buckets)
    nb1, nb5 = _by_bucket(n1, buckets), _by_bucket(n5, buckets)
    lines.append(f"  {'bucket':7s} {'n':>4}   {'champ t1':>9} {'nest t1':>9}"
                 f"   {'champ t5':>9} {'nest t5':>9}   {'d_t1':>7}")
    for name in ("early", "mid", "late"):
        cv, cn = cb1[name]; nv, _ = nb1[name]
        cv5, _ = cb5[name]; nv5, _ = nb5[name]
        lines.append(f"  {name:7s} {cn:>4}   {cv:9.4f} {nv:9.4f}"
                     f"   {cv5:9.4f} {nv5:9.4f}   {nv - cv:+7.4f}")

    # The decision, stated in the artifact rather than left to the reader.
    beat_top1 = d_top1 > 0
    tie_top1 = abs(d_top1) < 1e-9
    if beat_top1:
        verdict = (f"VERDICT: the nested model BEATS the flat prior on top-1 "
                   f"({n1.mean():.4f} vs {c1.mean():.4f}, {d_top1:+.4f}). The "
                   f"position-then-player decomposition is a genuine advance.")
    elif tie_top1 and d_ll < 0:
        verdict = ("VERDICT: the nested model ties the flat prior on top-1 and "
                   "improves log-loss -- a calibration win under the spec's "
                   "shipping rule (log-loss down, top-1 not worse).")
    else:
        verdict = ("VERDICT: the nested model does NOT beat the flat prior. "
                   "The decomposition did not pay on held-out human picks; the "
                   "flat prior stays the baseline.")
    lines.append("\n" + verdict)
    lines.append(f"(flat baseline of record: COLD_START_PRIOR / mock_prior.PRIOR "
                 f"top-1 {FLAT_PRIOR_TOP1})")
    return "\n".join(lines), beat_top1, tie_top1, d_ll


def _write_findings(report, beat_top1, tie_top1, d_ll):
    headline = ("Nested BEATS the flat prior on top-1." if beat_top1
                else "Nested ties top-1 and improves log-loss."
                if (tie_top1 and d_ll < 0)
                else "Nested does NOT beat the flat prior.")
    body = (
        "# Nested (position-then-player) opponent model\n\n"
        f"**{headline}**\n\n"
        "Held out by draft, human picks only, standard errors clustered by "
        "draft, head-to-head delta paired by draft. The nested model is refit "
        "leave-one-draft-out; the served flat cold-start prior "
        "(`draft_model.COLD_START_PRIOR`, i.e. `scoring/mock_prior.PRIOR` -- "
        "NOT `scoring/human_prior.py`) is scored full-width on the identical "
        "picks, which if anything flatters it since it has already seen some of "
        "them.\n\n"
        "Model: `scoring/nested_model.py` -- "
        "P(player) = P(position | team state) x P(player | position, board), "
        "the position factor the committed `position_model.MultinomialLogistic` "
        "(the 0.53 classifier), the within-position factor a shared "
        "conditional logit over same-position candidates. Fit separately, which "
        "is exact under the factorized (lambda=1) nested likelihood.\n\n"
        "```\n" + report + "\n```\n\n"
        "This is a measurement and a recommendation. It ships nothing: wiring "
        "the nested model into `cold_start_fits`/`_run_draft`/`survival` is a "
        "separate integration with fit/serve parity across five call sites, "
        "deliberately out of scope here.\n")
    try:
        with open(FINDINGS, "w") as fh:
            fh.write(body)
    except OSError:
        pass


def main(argv=None):
    global _DESIGN
    argv = sys.argv[1:] if argv is None else argv
    league_db = None
    limit = None
    workers = 1
    for i, a in enumerate(argv):
        if a == "--league-db" and i + 1 < len(argv):
            league_db = argv[i + 1]
        elif a == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])
        elif a == "--workers" and i + 1 < len(argv):
            workers = int(argv[i + 1])

    corpus = fp.open_corpus()
    league = fp.open_league(league_db)
    ids = fp.snapshot_draft_ids(corpus)
    if limit is not None:
        ids = ids[:limit]
    print(f"corpus drafts {len(ids)}"
          + (f" (limited to {limit})" if limit is not None else ""), flush=True)

    obs = sl.human_observations(corpus, league, ids)
    _DESIGN = nm.design_from_observations(obs)

    # The champion needs the full 34-column feature matrix and chosen index;
    # `fit_prior.design` builds them from the SAME observations in the SAME
    # order as the nested design, so pick i lines up across both. Assert it, so
    # a future reorder cannot silently compare the two models on misaligned
    # picks.
    X_list, chosen, groups, boards, buckets = fp.design(obs)
    assert list(groups) == list(_DESIGN.groups), (
        "champion and nested designs are not aligned pick-for-pick")

    # Close the database handles before any fork: a forked worker must not
    # inherit a live DuckDB connection. The design is fully built above and
    # nothing below reads the corpus.
    corpus.close()
    league.close()

    report, beat_top1, tie_top1, d_ll = _report(
        _DESIGN, X_list, chosen, len(chosen), workers)
    print("\n" + report, flush=True)
    _write_findings(report, beat_top1, tie_top1, d_ll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
