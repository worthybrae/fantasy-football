"""Rung 5 of the ladder: drafter archetypes.

    .venv/bin/python -m pipeline.fit_archetypes --league-db data/leagues/X.duckdb \
        --jobs 8 --restarts 3 --ks 1,2,3,4

NOTHING HAS EVER FINISHED A FULL SWEEP WITH THIS, AND THE MODEL IT FITS IS
SUPERSEDED. The only run against the real corpus was halted at 23 of 43 folds
because it saturated the machine for two hours and stalled the live mock farm,
which is the higher priority; and while it ran, the opponent-model design moved
on from a flat latent-class mixture to a nested logit with positions as nests
(`docs/superpowers/specs/2026-08-23-best-opponent-model-design.md`, section 4).
What survives is the machinery and the measurement, kept so the question can be
re-asked cheaply rather than rebuilt: see
`docs/superpowers/findings/2026-08-23-drafter-archetypes.md` for what the
halted run did and did not establish, and READ THE COST NOTE UNDER `sweep`
BEFORE RUNNING THIS AGAIN.

WHAT THIS MEASURES, AND WHY IT IS A SEPARATE MODULE FROM `score_ladder`.
`pipeline/score_ladder.py` establishes rungs 1-3 -- the market, the shipped
prior, and the human-only refit -- and rung 4, the Tier 1 pool features, is
appended there by another change. This module is rung 5, the latent-class
mixture, and it lives on its own for the same reason `human_prior.py` is not
`mock_prior.py`: it has its own artifact, its own decision, and a fit that is
not a single coefficient vector. It BORROWS the harness rather than copying it
-- the same replay, the same human-only filter, the same folds, the same
clustered standard errors -- so every rung below it is measured by the code
that measured it there.

THE MODEL IS `scoring/mixture.py`. Read that first; nothing about the EM, the
shrinkage or the live posterior is decided here. What this module owns is the
measurement: which feature set, which K, and whether any of it beats the rung
below on held-out human top-1.

THREE NUMBERS COME OUT OF ONE PASS, WHICH IS WHY THE PASS IS SHAPED THIS WAY.
For every held-out draft and every K, one mixture is fitted on the other
drafts, and that one fit yields:

  * the held-out MARGINAL log-likelihood of the held-out seats -- the criterion
    K is chosen by (`mixture.seat_marginal_loglik`, which explains why that and
    not AIC/BIC), and
  * the held-out PREQUENTIAL top-1 -- each pick predicted from that seat's
    posterior as it stood before the pick, which is exactly how it would serve.

Fitting once and reading both off is not just cheaper; it guarantees the K
chosen and the K scored are the same object.

THE DECISION. `delta_top1` against the higher of rung 3 and rung 4, measured on
the same picks in the same folds, and nothing else. There is no force flag,
matching `fit-prior` and `score-ladder`: a losing rung prints its numbers,
writes nothing, and exits 1. K=1 winning the sweep is a real and expected
outcome -- 41 drafts and ~240 seats is thin -- and it is a finding, not a
failure. Nothing here is tuned until something splits.

COST, AND THE `--jobs` FLAG. MEASURED, AND IT IS WORSE THAN IT LOOKS. A
leave-one-draft-out sweep over K in {1,2,3,4} with three restarts is a few
hundred conditional-logit fits per fold, and on 43 drafts and 2,889 human picks
one fold took 25-40 MINUTES. Eight workers cleared 23 folds in 108 minutes and
the full run was heading for about four hours, during which it held the machine
at capacity and starved the live mock farm of CPU. Do not run this on a machine
that is doing anything else, and budget half a day rather than an hour.

Folds are independent, so they run in a process pool; `--jobs 1` keeps it
serial and is what the tests use. The pool is FORKED rather than spawned so the
children inherit the design matrices (~100MB) instead of pickling them once per
fold, which is safe here because the fork happens before any thread exists and
the children only read.

THE RUN PRINTS NOTHING UNTIL IT FINISHES, which is the first thing to fix if
anyone picks this up. The per-fold reports live in one list in `sweep` and are
aggregated at the end, so a run that is stopped part-way loses every fold it
completed. That is exactly what happened to the only real run there has been.

READ-ONLY, ABSOLUTELY. Same rule as `score_ladder`: the corpus cannot be
rebuilt, a farm loop is writing new drafts into it while this runs, both
databases are opened read-only, and no statement of any kind is issued against
`draft_log*`. The draft ids are snapshotted once and printed.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pipeline import draft_log as dl
from pipeline import fit_prior as fp
from pipeline import score_ladder as sl
from scoring import draft_model as dm
from scoring import mixture as mx
from scoring.draft_model import FEATURE_NAMES, _ROUND_BUCKET_ORDER

# Where a winning rung 5 lands. A separate generated module again, and for the
# third time for the same reason: it has a different writer, a different rule
# and a different SHAPE from every prior that already exists. `mock_prior.PRIOR`
# and `human_prior.PRIOR` are one vector each; this is K vectors plus K weights
# plus the pooled vector they were shrunk toward, and widening either of those
# files to hold that would put a mixture into a module `draft_model` imports
# before anything decided the mixture ships.
ARCHETYPES_MODULE = (Path(__file__).resolve().parent.parent
                     / "scoring" / "archetypes.py")

# The K values swept. 1 is not padding: it is the control, it is the pooled fit
# exactly (see `mixture.fit_mixture`), and a sweep that did not include it
# could not say that splitting the seats bought anything.
DEFAULT_KS = (1, 2, 3, 4)

# Fold state, set in the parent before the pool forks. Module-level because a
# forked child inherits it by address rather than by pickle -- the design
# matrices are ~100MB and there are 41 folds.
_SHARED: dict = {}


# ------------------------------------------------------------ the feature set

def feature_set(name: str) -> list:
    """The columns rung 5 is fitted on, by name.

    Two choices and no third. `shipped23` is rung 3's set, pinned by
    `score_ladder.SHIPPED_23_FEATURES`; `all` is everything `FEATURE_NAMES`
    holds, which is rung 3's set plus the six Tier 1 pool signals. The mixture
    goes on top of whichever of those two won its own rung, because the ladder
    exists to keep the three effects -- the population, the features, the
    archetypes -- from being confounded, and fitting the mixture on a feature
    set that lost would confound the last two in the other direction.
    """
    if name == "shipped23":
        return list(sl.SHIPPED_23_FEATURES)
    if name == "all":
        return list(FEATURE_NAMES)
    raise ValueError(f"unknown feature set {name!r}; use shipped23 or all")


# ------------------------------------------------------------------ the folds

def _fold(groups, holdout):
    train = [i for i, g in enumerate(groups) if g != holdout]
    test = [i for i, g in enumerate(groups) if g == holdout]
    return train, test


def _take(values, idx):
    return [values[i] for i in idx]


def _run_fold(holdout):
    """One held-out draft, every K. Runs in a pool worker; reads `_SHARED`.

    The pooled fit is computed ONCE for the fold and handed to every K, which
    is not only cheaper but necessary: `pooled` is the shrinkage target and the
    K=1 answer, so a K=2 model shrunk toward a different vector than the K=1
    model IS would not be comparable to it.

    Returns raw counts, never rates. The caller adds folds together, and a rate
    re-multiplied by a fold size is a rounding error looking for somewhere to
    land -- the same discipline `fit_prior.score` follows.
    """
    X_list, chosen, seats = _SHARED["X"], _SHARED["chosen"], _SHARED["seats"]
    groups, buckets = _SHARED["groups"], _SHARED["buckets"]
    train, test = _fold(groups, holdout)
    if not train or not test:
        return holdout, {}

    Xtr, ctr, str_ = _take(X_list, train), _take(chosen, train), _take(seats, train)
    Xte, cte, ste = _take(X_list, test), _take(chosen, test), _take(seats, test)
    bte = _take(buckets, test)
    pooled = dm.fit(Xtr, ctr)

    out = {}
    for k in _SHARED["ks"]:
        model = mx.fit_mixture(Xtr, ctr, str_, k, pooled=pooled,
                               lam=_SHARED["lam"], restarts=_SHARED["restarts"],
                               seed=_SHARED["seed"])
        marginal = mx.seat_marginal_loglik(Xte, cte, ste, model)
        report = mx.score_mixture(Xte, cte, ste, model, buckets=bte)
        out[k] = {
            "heldout_ll": marginal["loglik"],
            "hits1": report["hits1"], "hits5": report["hits5"],
            "ll": report["ll"], "n": report["n"],
            "by_bucket": report.get("by_bucket", {}),
            "train_pi": model.pi.tolist(),
            "train_loglik": model.loglik,
            "converged": model.converged,
        }
    return holdout, out


def _init_shared(shared):
    """Spawn-safe fallback: a child that did NOT inherit `_SHARED` gets it here.

    The pool is forked, so this normally does nothing the fork had not already
    done. It exists so that a platform where fork is unavailable still runs
    correctly rather than raising a KeyError deep inside a worker.
    """
    _SHARED.update(shared)


def sweep(X_list, chosen, seats, groups, buckets, ks=DEFAULT_KS,
          lam=mx.DEFAULT_LAMBDA, restarts=mx.DEFAULT_RESTARTS, seed: int = 0,
          jobs: int = 1, progress=None) -> dict:
    """Leave-one-draft-out, every K, both criteria. `{k: report}`.

    The fold is the DRAFT for the same reason it is everywhere else in this
    ladder: one room, one board, eight strangers, and every seat in it saw
    every other seat's picks. Holding out a SEAT rather than a draft would put
    the other seven seats of that room in training, and they collectively
    determine what was still on the board when the held-out seat picked.
    """
    _SHARED.update({"X": X_list, "chosen": chosen, "seats": seats,
                    "groups": groups, "buckets": buckets, "ks": list(ks),
                    "lam": lam, "restarts": restarts, "seed": seed})
    order = sorted(set(groups))

    results = []
    if jobs <= 1:
        for i, holdout in enumerate(order, start=1):
            results.append(_run_fold(holdout))
            if progress:
                progress(i, len(order))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(min(jobs, len(order)), initializer=_init_shared,
                      initargs=(dict(_SHARED),)) as pool:
            for i, item in enumerate(pool.imap_unordered(_run_fold, order),
                                     start=1):
                results.append(item)
                if progress:
                    progress(i, len(order))

    reports = {}
    for k in ks:
        hits1 = hits5 = scored = 0
        ll = heldout_ll = 0.0
        by_draft, by_draft_n, by_bucket, pis = {}, {}, {}, {}
        heldout_by_draft = {}
        for holdout, per_k in results:
            fold = per_k.get(k)
            if not fold or not fold["n"]:
                continue
            hits1 += fold["hits1"]
            hits5 += fold["hits5"]
            ll += fold["ll"]
            heldout_ll += fold["heldout_ll"]
            scored += fold["n"]
            by_draft[holdout] = fold["hits1"] / fold["n"]
            by_draft_n[holdout] = fold["n"]
            # Per pick, per draft, because `choose_k` pairs two K values by
            # draft the same way the ladder pairs two rungs: both scored the
            # same picks in the same rooms, so what makes one room harder than
            # another cancels in the difference.
            heldout_by_draft[holdout] = fold["heldout_ll"] / fold["n"]
            pis[holdout] = fold["train_pi"]
            for bucket, tally in fold["by_bucket"].items():
                running = by_bucket.setdefault(bucket, [0, 0, 0])
                for j in range(3):
                    running[j] += tally[j]
        if not scored:
            continue
        reports[k] = {
            "top1": hits1 / scored, "top5": hits5 / scored,
            "logloss": -ll / scored, "n": scored, "hits1": hits1,
            "hits5": hits5, "ll": ll, "by_draft": by_draft,
            "by_draft_n": by_draft_n, "by_bucket": by_bucket,
            # The criterion K is chosen by, per pick so folds of different
            # sizes are comparable and so the unit matches the log-loss column
            # every other rung already reports.
            "heldout_ll": heldout_ll, "heldout_per_pick": heldout_ll / scored,
            "heldout_by_draft": heldout_by_draft,
            "pi_by_fold": pis,
        }
    return reports


def heldout_se(a: dict, b: dict) -> float:
    """Standard error of the DIFFERENCE in held-out nats per pick, by draft.

    `fit_prior.paired_se`'s argument applied to the likelihood column: both K
    values are scored on the same picks in the same rooms, so most of what
    makes one draft harder than another -- how deep the board ran, how much the
    room reached -- hits both sides equally and cancels. Comparing two
    unpaired errors instead would charge the comparison for variation common to
    both, which on 43 drafts is most of it.
    """
    return fp.paired_se(a.get("heldout_by_draft", {}),
                        b.get("heldout_by_draft", {}))


def choose_k(reports: dict) -> int:
    """The largest K that BEATS every smaller one by more than one paired
    standard error of the held-out likelihood.

    A bare `>` is the wrong rule here and measurably so: on a synthetic corpus
    drawn from ONE drafter, K=2 still comes out ahead of K=1 by about 4e-5 nats
    per pick, because a second class can always absorb a little sampling noise
    from the folds it was fitted on. Selecting K on that would report "there
    are two kinds of drafter" from a corpus that has one -- the exact failure
    the rest of this ladder is built to avoid, arriving through the one
    decision the ladder's own `delta_top1` rule does not cover.

    So the gain has to clear its own noise, and the noise is the PAIRED error
    across drafts. One standard error, not two: this chooses K, and the rung
    still has to beat the rung below on `delta_top1` afterwards, so a K
    selected on a marginal likelihood gain ships nothing unless it also
    predicts better. Ties and near-ties therefore go to the SMALLER K, which is
    the same direction the rest of the ladder leans -- a rung that does not
    beat the one below does not ship.
    """
    best_k = min(reports)
    for k in sorted(reports):
        if k == best_k:
            continue
        gain = (reports[k]["heldout_per_pick"]
                - reports[best_k]["heldout_per_pick"])
        se = heldout_se(reports[k], reports[best_k])
        # A NaN error means fewer than two shared folds, which is not evidence
        # of anything; the smaller K keeps the tie.
        if np.isfinite(se) and gain > se:
            best_k = k
    return best_k


# ---------------------------------------------------------------- the module

_MODULE_TEMPLATE = '''"""Drafter archetypes. GENERATED -- do not edit by hand.

K coefficient vectors and their population weights, fitted as a latent-class
conditional logit by `scoring/mixture.py` and written by
`pipeline/fit_archetypes.py` ONLY because the mixture
beat the rung below it on held-out human top-1. There is no force flag, so a
losing sweep leaves this file exactly as it was.

WHAT A CLASS IS. A SEAT within one draft, not a person. Mock opponents are
strangers with no identity that survives the session, so there is nothing to
pool across drafts and the clustering unit is one anonymous seat's 2-16 picks.
The classes are UNNAMED: EM produces coefficient vectors, and the only honest
description of one is a reading of what its coefficients say against `POOLED`,
which is the vector every class was shrunk toward.

HOW THIS IS SERVED. Every opponent seat starts at `WEIGHTS` -- so at pick 1 the
prediction is the population average and the cold start cannot be worse than a
single pooled vector -- and is Bayes-updated after each pick that seat makes.
A single-pick answer averages the classes under that posterior; a ROLLOUT
samples one class per seat and holds it for the whole simulated draft. See
`scoring/mixture.py` for why those two are different.

PROVENANCE OF THESE NUMBERS

{provenance_prose}
"""
import numpy as np

# What was fitted, when, against what, and what it beat. Machine-readable so a
# reader does not have to trust the prose above and a test can assert the two
# agree.
PROVENANCE = {provenance_dict}

# The feature order these coefficients were fitted in, written out rather than
# assumed. A coefficient vector applied to columns in a different order than it
# was fitted in is silently wrong on every pick, and a length check cannot see
# it.
FEATURES = {features}

# The drafts this was fitted and scored on, snapshotted at the top of the run.
# Recorded because the corpus GROWS -- a farm loop adds roughly one draft every
# fifteen minutes -- so "the corpus" is not a reproducible input without the
# list.
DRAFT_IDS = {draft_ids}

# The population weight of each class: the share of SEATS the fit assigns to
# it, and the posterior every unseen opponent starts at.
WEIGHTS = np.array({weights})

# The pooled fit every class was shrunk toward. Not decoration: a class's
# coefficients are only readable as a difference from this, and this is exactly
# what a K=1 model would have been.
POOLED = np.array([
{pooled}])

# One row per class, in `FEATURES` order, sorted by population weight.
BETAS = np.array([
{betas}])

assert BETAS.shape == (len(WEIGHTS), len(FEATURES)), (
    "the generated archetypes and their feature list disagree -- regenerate "
    "by re-running pipeline/fit_archetypes.py rather than editing either "\n    "by hand")
assert abs(float(WEIGHTS.sum()) - 1.0) < 1e-6, (
    "the class weights are a distribution over classes and must sum to 1")
'''


def _rows(values, features, indent="    ") -> str:
    width = max(len(name) for name in features)
    return "\n".join(f"{indent}{v:+.6f},  # {name:<{width}}".rstrip()
                     for name, v in zip(features, values))


def render_module(model, features, draft_ids, provenance, provenance_prose) -> str:
    """The text of `scoring/archetypes.py`.

    One coefficient per line with its feature name beside it, the same way
    `fit_prior.render_module` and `score_ladder.render_module` do it and for
    the same reason: the only thing anyone wants from a generated coefficient
    file is to read what a named coefficient came out as, and a diff between
    two generations should show which coefficients moved.
    """
    blocks = []
    for k, (weight, beta) in enumerate(zip(model.pi, model.betas)):
        blocks.append(f"    # ---- class {k}, {weight:.1%} of seats ----\n"
                      f"    [\n{_rows(beta, features, indent='        ')}\n    ],")
    return _MODULE_TEMPLATE.format(
        provenance_prose=provenance_prose,
        provenance_dict=fp._pretty_dict(provenance),
        features=fp._pretty_list(features),
        draft_ids=fp._pretty_list(draft_ids),
        weights=np.round(model.pi, 6).tolist(),
        pooled=_rows(model.pooled, features) + "\n",
        betas="\n".join(blocks) + "\n")


def write_archetypes(model, features, draft_ids, provenance, provenance_prose,
                     path: Path | None = None) -> Path:
    """Write the generated module. Callers must have checked the rule first.

    Deliberately dumb: it writes what it is given. The decision lives in
    `main`, in one place, next to the numbers that make it.
    """
    target = Path(path or ARCHETYPES_MODULE)
    target.write_text(render_module(model, features, draft_ids, provenance,
                                    provenance_prose))
    return target


# ------------------------------------------------------------------- the run

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _prose(reports, best_k, bar_label, bar, rung5, delta, se, draft_ids,
           corpus_path, features, lam, restarts) -> str:
    lines = "\n".join(
        f"    K={k}  held-out nats/pick {reports[k]['heldout_per_pick']:+.4f}"
        f"   top-1 {reports[k]['top1']:.4f}   top-5 {reports[k]['top5']:.4f}"
        for k in sorted(reports))
    return f"""
Fitted on {rung5['n']} picks a person is KNOWN to have made
(`autodrafted IS FALSE`, at a seat that is not ours), across {len(draft_ids)} drafts of
the cross-league draft corpus ({corpus_path}), on {_now()}.
{len(features)} features, shrinkage lambda {lam}, {restarts} random restarts per fit.

THE K SWEEP. Leave-one-draft-out; every draft held out in turn, the mixture
refitted on the other N-1, and the held-out draft's seats scored by a model
that never saw them. K is chosen by held-out MARGINAL log-likelihood per pick
-- the class memberships integrated out, which is exactly what a live
posterior-updating predictor earns -- and not by AIC or BIC.

{lines}

Chosen K: {best_k}.

THE LADDER RUNG. {bar_label} is the rung below, measured on the same picks in
the same folds.

    {bar_label:<22} top-1 {bar['top1']:.4f}
    mixture, K={best_k:<12} top-1 {rung5['top1']:.4f}

delta_top1 {delta:+.4f} +/-{se:.4f}, paired by draft. top-1 is the decision
metric; top-5 and log-loss are recorded because they were measured, not because
they decided.

WHAT THIS DOES NOT ESTABLISH.

  * Not that the classes are PEOPLE. They are seats within single mock drafts,
    and a seat is 2 to 16 picks by whoever sat there for an hour. Nothing here
    followed a drafter across two rooms, because the corpus cannot.
  * Not that the classes are the RIGHT classes. EM finds a local optimum of a
    likelihood; a different restart, a different lambda or fifty more drafts
    could produce a different partition with a similar score.
  * Not that a class description is a personality. The words in the findings
    doc are a reading of coefficient differences from POOLED, written after
    the fit, on classes the fit did not name.
  * Not that it drafts better. It predicts opponents better on one population;
    nothing here measured a roster.
""".strip("\n")


def _print_report(label, report, extra="") -> None:
    se = (f"  +/-{fp.cluster_se(report['by_draft']):.4f}"
          if report.get("by_draft") else "")
    print(f"  {label:<22} top-1 {report['top1']:.4f}  "
          f"top-5 {report['top5']:.4f}  log-loss {report['logloss']:>7.4f}"
          f"{se}{extra}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    corpus_path = fp._option(argv, "--corpus") or dl.CORPUS_PATH
    league_path = fp._option(argv, "--league-db")
    jobs = int(fp._option(argv, "--jobs", 1))
    lam = float(fp._option(argv, "--lam", mx.DEFAULT_LAMBDA))
    restarts = int(fp._option(argv, "--restarts", mx.DEFAULT_RESTARTS))
    ks = tuple(int(k) for k in
               str(fp._option(argv, "--ks", "1,2,3,4")).split(","))
    forced_features = fp._option(argv, "--features")

    corpus = fp.open_corpus(corpus_path)
    league_conn = fp.open_league(league_path)
    try:
        snapshot = fp.snapshot_draft_ids(corpus)
        draft_ids = sl.labelled_draft_ids(corpus, snapshot)
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

        corpus_obs = sl.human_observations(corpus, league_conn, draft_ids)
        X_full, chosen, groups, boards, buckets = fp.design(corpus_obs)
        # The seat key. `build_corpus_observations` already carries it as
        # `manager`, which for a corpus mock is `anon:{draft_id}:{slot}` --
        # unique per seat per draft by construction (`draft_log.owner_key`),
        # which is exactly the clustering unit and is why nothing upstream had
        # to change to fit this.
        seats = [o.manager for o in corpus_obs.observations]
        print(f"\n{corpus_obs.picks_seen} picks in those drafts; "
              f"{len(chosen)} are known human picks at a seat that is not "
              f"ours, across {len(set(seats))} seats. Excluded:")
        for reason, n in corpus_obs.dropped.items():
            print(f"  {reason:<14} {n}")
        if not chosen:
            print("No human picks. Nothing to measure.")
            return 1

        # ---- the rung below: whichever of rung 3 and rung 4 is higher, on
        # these picks, in these folds. Measured here rather than read off
        # `human_prior.PROVENANCE` because the corpus grows: a number fitted on
        # 38 drafts is not the bar for a model measured on 41.
        print("\nThe rung below, re-measured on this snapshot (leave-one-draft-"
              "out, human picks only):")
        rungs = {}
        for name in ("shipped23", "all"):
            names = feature_set(name)
            started = time.time()
            rungs[name] = fp.leave_one_draft_out(
                X_full, chosen, groups, buckets=buckets,
                keep_idx=fp.feature_indices(names))
            _print_report(f"{name} ({len(names)} cols)", rungs[name],
                          f"  ({time.time() - started:.0f}s)")
        chosen_set = forced_features or max(rungs, key=lambda n: rungs[n]["top1"])
        bar = rungs[chosen_set]
        features = feature_set(chosen_set)
        bar_label = ("rung 3, the shipped 23" if chosen_set == "shipped23"
                     else "rung 4, all 29")
        print(f"\nThe mixture goes on `{chosen_set}` -- the higher of the two, "
              "so the archetypes are\nmeasured on top of the feature set that "
              "won its own rung rather than one that lost.")

        keep_idx = fp.feature_indices(features)
        X_list = [X[:, keep_idx] for X in X_full]

        # ---- the sweep
        print(f"\nK sweep, leave-one-draft-out over {len(set(groups))} drafts, "
              f"lambda {lam}, {restarts} restarts,\n{jobs} job(s). This is the "
              "slow part.")
        started = time.time()

        def progress(done, total):
            print(f"  fold {done}/{total}  ({time.time() - started:.0f}s)",
                  flush=True)

        reports = sweep(X_list, chosen, seats, groups, buckets, ks=ks, lam=lam,
                        restarts=restarts, jobs=jobs, progress=progress)
        if not reports:
            print("No fold produced a scoreable holdout.")
            return 1

        print("\nHeld-out marginal log-likelihood per pick is the criterion; "
              "top-1 is reported\nfor every K beside it because it is what the "
              "ladder decides on. `gain` is against\nK=1 and `+/-` is its "
              "PAIRED error across drafts -- a gain smaller than that is\n"
              "noise, and `choose_k` will not buy a class with it.")
        print(f"  {'K':<3} {'nats/pick':>10} {'gain':>9} {'+/-':>8} "
              f"{'top-1':>8} {'top-5':>8} {'log-loss':>9}  mean class weights")
        for k in sorted(reports):
            report = reports[k]
            pis = np.mean([sorted(v, reverse=True)
                           for v in report["pi_by_fold"].values()], axis=0)
            gain = (report["heldout_per_pick"]
                    - reports[min(reports)]["heldout_per_pick"])
            se = heldout_se(report, reports[min(reports)])
            print(f"  {k:<3} {report['heldout_per_pick']:>10.4f} "
                  f"{gain:>+9.4f} {se:>8.4f} "
                  f"{report['top1']:>8.4f} {report['top5']:>8.4f} "
                  f"{report['logloss']:>9.4f}  "
                  + " ".join(f"{p:.2f}" for p in np.atleast_1d(pis)))

        best_k = choose_k(reports)
        rung5 = reports[best_k]
        print(f"\nChosen K = {best_k}, by held-out marginal likelihood. Ties go "
              "to the smaller K.")
        _print_buckets(bar, rung5, bar_label, best_k)

        # THE RULE. delta_top1 against the rung below, and nothing else.
        delta = rung5["top1"] - bar["top1"]
        se = fp.paired_se(rung5["by_draft"], bar["by_draft"])
        print("\nPaired by draft -- both sides scored the same picks in the "
              "same rooms, so what\nmakes one room harder than another cancels "
              "in the difference.")
        _print_report(bar_label, bar)
        _print_report(f"mixture, K={best_k}", rung5)
        print(f"  {'delta_top1':<22} {delta:+.4f} +/-{se:.4f}")

        if best_k == 1 or delta <= 0:
            why = ("the sweep chose K=1, which IS the pooled fit -- the seats "
                   "in this corpus do\nnot separate into archetypes that "
                   "predict a draft nobody in the fit has seen"
                   if best_k == 1 else
                   f"the mixture does NOT beat {bar_label} on held-out human "
                   f"top-1 ({delta:+.4f}\n+/-{se:.4f})")
            print(f"\nNothing is written: {why}.\nA negative result is a "
                  "complete result -- write it up rather than tuning until\n"
                  "something wins.")
            return 1

        # The shipping model: refitted on EVERY labelled draft at the chosen K,
        # the same way `full_beta` produces the vector that ships. The fold
        # models were for measuring.
        pooled = dm.fit(X_list, chosen)
        model = mx.fit_mixture(X_list, chosen, seats, best_k, pooled=pooled,
                               lam=lam, restarts=restarts)
        provenance = {
            "fitted_at": _now(),
            "corpus": corpus_path,
            "population": "autodrafted IS FALSE (known human), not our seat",
            "drafts": len(draft_ids),
            "seats": len(set(seats)),
            "picks_scored": len(chosen),
            "picks_excluded": dict(corpus_obs.dropped),
            "folds": "leave-one-draft-out",
            "feature_set": chosen_set,
            "features_fitted": list(features),
            "k": best_k,
            "lam": lam,
            "restarts": restarts,
            "k_sweep_heldout_per_pick": {k: round(r["heldout_per_pick"], 6)
                                         for k, r in sorted(reports.items())},
            # The paired error of each K's gain over K=1, recorded beside the
            # gains themselves so a reader can see which of them cleared it.
            "k_sweep_heldout_se": {
                k: round(heldout_se(r, reports[min(reports)]), 6)
                for k, r in sorted(reports.items())},
            "k_sweep_top1": {k: round(r["top1"], 6)
                             for k, r in sorted(reports.items())},
            "bar": bar_label,
            "bar_top1": round(bar["top1"], 6),
            "mixture_top1": round(rung5["top1"], 6),
            "mixture_top5": round(rung5["top5"], 6),
            "mixture_logloss": round(rung5["logloss"], 6),
            "delta_top1": round(delta, 6),
            "delta_top1_se": round(se, 6),
        }
        target = write_archetypes(
            model, features, draft_ids, provenance,
            _prose(reports, best_k, bar_label, bar, rung5, delta, se,
                   draft_ids, corpus_path, features, lam, restarts))
        print(f"\nThe mixture wins on held-out human top-1 ({delta:+.4f} "
              f"+/-{se:.4f}). Wrote {target}.")
        for k, (weight, beta) in enumerate(zip(model.pi, model.betas)):
            print(f"  class {k} ({weight:.1%} of seats): "
                  + mx.describe_class(beta, model.pooled, features))
        return 0
    finally:
        corpus.close()
        league_conn.close()


def _print_buckets(bar, rung5, bar_label, best_k) -> None:
    """WHERE the accuracy moved, not just whether it did.

    Same buckets `draft_model` draws (early 1-3, mid 4-8, late 9+) and the same
    table `score_ladder` prints, because this row gets read next to that one. A
    mixture that wins only in the rounds where a kicker and a defense have to be
    taken has won something much smaller than its headline says.
    """
    a_all, b_all = bar.get("by_bucket"), rung5.get("by_bucket")
    if not a_all or not b_all:
        return
    print("\nBy round bucket (early 1-3, mid 4-8, late 9+), same held-out "
          "human picks:")
    print(f"  {'bucket':<7} {'n':>5}  {bar_label[:13]:>13} {'mixture':>11}"
          f"  {'top-5 below':>13} {'top-5 mix':>11}")
    for bucket in _ROUND_BUCKET_ORDER:
        a, b = a_all.get(bucket), b_all.get(bucket)
        if not a or not b:
            continue
        print(f"  {bucket:<7} {a[2]:>5}  {a[0] / a[2]:>13.4f} "
              f"{b[0] / b[2]:>11.4f}  {a[1] / a[2]:>13.4f} "
              f"{b[1] / b[2]:>11.4f}")


if __name__ == "__main__":
    sys.exit(main())
