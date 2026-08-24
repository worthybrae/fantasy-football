"""Measure the ceiling of a nested (position-then-player) opponent model.

Answers ONE question before we build the full nested logit: how well can the
POSITION a human takes next be predicted from the drafting team's state?  The
whole nested approach's top-1 ceiling is roughly P(position) x P(player|position),
and the second factor is ~0.53-0.60 with simple rules, so the first factor
decides whether nested can approach the owner's 50% target.

Diagnostic only: fits no production model, touches no champion.  Uses the
already-built `scoring.position_model` (its classifier, MLP, and three
baselines) and the shared human-pick replay from `score_ladder`, so the folds
and the choice sets line up with the rest of the ladder.  Held out by DRAFT.

Run:  python -m pipeline.measure_position [--league-db data/leagues/<id>.duckdb]
Writes its numbers to stdout AND to the findings path so a background run whose
console is lost still leaves the result on disk.
"""
import sys
import numpy as np

import pipeline.fit_prior as fp
from pipeline import score_ladder as sl
from scoring import position_model as pm

FINDINGS = "docs/superpowers/findings/2026-08-23-position-model-ceiling.md"
WITHIN_POS_HIT = 0.55   # conservative P(player | position); see spec


def _folds(groups):
    """Leave-one-draft-out over the fold key."""
    uniq = sorted(set(groups))
    g = np.asarray(groups)
    for held in uniq:
        yield g != held, g == held


def _acc_by_bucket(correct, buckets):
    out = {}
    for b in ("early", "mid", "late"):
        m = np.asarray([x == b for x in buckets])
        out[b] = (float(correct[m].mean()), int(m.sum())) if m.any() else (float("nan"), 0)
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    league_db = None
    for i, a in enumerate(argv):
        if a == "--league-db" and i + 1 < len(argv):
            league_db = argv[i + 1]

    corpus = fp.open_corpus()
    league = fp.open_league(league_db)
    ids = fp.snapshot_draft_ids(corpus)
    obs = sl.human_observations(corpus, league, ids)
    design = pm.design_from_observations(obs)
    X, y, groups = design.X, design.y, np.asarray(design.groups)
    n = len(y)

    # per-pick correctness for each model, accumulated across held-out folds
    models = {"multinomial": np.zeros(n, bool), "mlp": np.zeros(n, bool),
              "always_wr": np.zeros(n, bool), "majority_round": np.zeros(n, bool),
              "round_roster": np.zeros(n, bool)}
    rounds = design.rounds
    rosters = design.rosters

    for tr, te in _folds(groups):
        # the fitted classifiers
        ml = pm.MultinomialLogistic().fit(X[tr], y[tr])
        models["multinomial"][te] = ml.predict(X[te]) == y[te]
        try:
            mlp = pm.MLPClassifier().fit(X[tr], y[tr])
            models["mlp"][te] = mlp.predict(X[te]) == y[te]
        except Exception as exc:               # MLP is optional; never sink the run
            models["mlp"][te] = False
        # the three baselines key on rounds/rosters, not X
        rtr = [rosters[i] for i in np.where(tr)[0]]
        rte = [rosters[i] for i in np.where(te)[0]]
        for key, cls in (("always_wr", pm.AlwaysWR),
                         ("majority_round", pm.MajorityPerRound),
                         ("round_roster", pm.RoundRosterLookup)):
            b = cls().fit(rounds[tr], rtr, y[tr])
            models[key][te] = b.predict(rounds[te], rte) == y[te]

    # cluster SE by draft for the headline
    def clustered_se(correct):
        per = [correct[groups == g].mean() for g in sorted(set(groups))]
        per = np.asarray(per)
        return per.std(ddof=1) / np.sqrt(len(per))

    lines = []
    lines.append(f"drafts {len(set(groups))}, human picks {n}\n")
    order = ["always_wr", "majority_round", "round_roster", "multinomial", "mlp"]
    for k in order:
        acc = models[k].mean()
        se = clustered_se(models[k])
        bb = _acc_by_bucket(models[k], design.buckets)
        lines.append(f"{k:16s} acc {acc:6.4f} +/-{se:.4f}   "
                     f"early {bb['early'][0]:.3f} mid {bb['mid'][0]:.3f} late {bb['late'][0]:.3f}")
    best = max(models["multinomial"].mean(), models["mlp"].mean())
    ceil = best * WITHIN_POS_HIT
    lines.append(f"\nbest position acc {best:.4f}  x within-pos {WITHIN_POS_HIT} "
                 f"=> implied nested top-1 ceiling ~ {ceil:.4f}")
    lines.append(f"(champion flat model today: 0.2595)")
    report = "\n".join(lines)
    print(report)
    try:
        with open(FINDINGS, "w") as fh:
            fh.write("# Position-prediction ceiling\n\n**Diagnostic. "
                     "Held out by draft, human picks only.**\n\n```\n"
                     + report + "\n```\n")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
