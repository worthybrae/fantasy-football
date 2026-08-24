"""Measure a HYBRID opponent model: flat champion early, nested mid/late.

The nested model (`measure_nested`) beats the flat champion overall but LOSES
the early rounds -- there the best player overall and the best at his position
are the same consensus name, so splitting the call into two factors only
dilutes a near-deterministic pick.  The flat champion's single sharp board read
wins early; the nested model wins mid and late.  So route by round: use the
flat champion's prediction in the early bucket and the nested model's from mid
on.  Both per-pick prediction sets are exactly what `measure_nested` already
computes, held out by draft -- this reuses them and applies one routing rule,
so the hybrid is scored on the identical picks under the identical folds and is
directly comparable to both parents.

Diagnostic; ships nothing.  Run:
  python -m pipeline.measure_hybrid --league-db data/leagues/1251381776.duckdb
Writes to stdout and the findings file so a background run keeps its answer.
"""
import sys
import numpy as np

import pipeline.fit_prior as fp
from pipeline import score_ladder as sl
from pipeline import measure_nested as mn
from scoring import nested_model as nm

FINDINGS = "docs/superpowers/findings/2026-08-24-hybrid-model.md"


def _cluster_se(hit, groups):
    per = [hit[groups == g].mean() for g in sorted(set(groups))]
    per = np.asarray(per)
    return per.std(ddof=1) / np.sqrt(len(per))


def _bucket_acc(hit, buckets):
    b = np.asarray(buckets)
    return {k: (float(hit[b == k].mean()), int((b == k).sum()))
            for k in ("early", "mid", "late") if (b == k).any()}


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
    design = nm.design_from_observations(obs)
    X_list, chosen, groups, boards, buckets = fp.design(obs)
    assert list(groups) == list(design.groups), "designs misaligned"
    corpus.close(); league.close()

    groups = np.asarray(design.groups)
    buckets = np.asarray(design.buckets)

    # per-pick top-1 hit flags for both parents, held out by draft (nested) and
    # full-width (champion) -- exactly measure_nested's own two columns.
    c1, c3, c5, cll = mn._champion_perpick(X_list, chosen)
    n1, n3, n5, nll = mn._nested_perpick(design, workers=1)

    early = buckets == "early"
    # the routing rule: flat early, nested from mid on.
    def route(cflag, nflag):
        out = np.where(early, cflag, nflag)
        return out.astype(float)
    h1, h3, h5 = route(c1, n1), route(c3, n3), route(c5, n5)
    hll = np.where(early, cll, nll)

    def row(name, a1, a3, a5, all_):
        return (f"{name:10s} top1 {a1.mean():.4f} +/-{_cluster_se(a1, groups):.4f}  "
                f"top3 {a3.mean():.4f}  top5 {a5.mean():.4f}  logloss {all_.mean():7.4f}")

    L = [f"drafts {len(set(groups))}, human picks {len(chosen)}", ""]
    L.append(row("champion", c1, c3, c5, cll))
    L.append(row("nested", n1, n3, n5, nll))
    L.append(row("HYBRID", h1, h3, h5, hll))
    L.append("")
    L.append(f"hybrid - champion top1: {h1.mean()-c1.mean():+.4f}")
    L.append(f"hybrid - nested   top1: {h1.mean()-n1.mean():+.4f}")
    L.append("")
    L.append("by round bucket (hybrid uses flat early, nested mid/late):")
    ba_c = _bucket_acc(c1, buckets); ba_n = _bucket_acc(n1, buckets); ba_h = _bucket_acc(h1, buckets)
    for k in ("early", "mid", "late"):
        if k in ba_h:
            L.append(f"  {k:6s} n={ba_h[k][1]:5d}  champ {ba_c[k][0]:.4f}  "
                     f"nested {ba_n[k][0]:.4f}  hybrid {ba_h[k][0]:.4f}")
    best = max(c1.mean(), n1.mean(), h1.mean())
    winner = {c1.mean(): "champion", n1.mean(): "nested", h1.mean(): "HYBRID"}[best]
    L.append(f"\nbest overall top-1: {winner} ({best:.4f})")
    report = "\n".join(L)
    print(report, flush=True)
    try:
        with open(FINDINGS, "w") as fh:
            fh.write("# Hybrid opponent model (flat early, nested mid/late)\n\n"
                     "**Diagnostic. Held out by draft, human picks only. "
                     "Routes per pick: flat champion in the early bucket, nested "
                     "from mid on.**\n\n```\n" + report + "\n```\n")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
