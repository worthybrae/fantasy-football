"""Does reading ESPN's OWN board beat the champion at predicting a person?
Run: make espn-ladder

WHAT THIS DECIDES, AND WHY IT IS ITS OWN RUNG. The shipped human-only prior
(`scoring/human_prior.py`, "the champion") prices every candidate against
`market_rank` -- the cheat-sheet/FFC blend `reach`/`fall` are computed on. But
an ESPN mock lobby drafts off ESPN's own numbered list on screen, and the
measurement in docs/superpowers/specs/2026-08-23-best-opponent-model-design.md
is blunt about the cost: 23.1% of human picks are the top name on ESPN's list
against 15.9% on the market's, 47.4% top-3 against 36.8%. The model reads the
wrong board. `draft_model._ESPN_BOARD_FEATURES` adds five columns computed
against `espn_rank`/`espn_proj` instead -- `espn_reach`, `espn_fall`,
`espn_list_pos`, `board_disagreement`, `espn_proj_dropoff` -- and this module
measures whether they predict a HUMAN pick better than the champion does.

THE HARNESS IS `pipeline.score_ladder`, NOT A SECOND COPY OF IT. Everything
here reuses that module's machinery -- `human_observations` (the same replay
that builds rung 3 and rung 4), `ablation` (with `candidates=` passed by name,
never defaulted), `write_human_prior` (the same writer, the same annotation
rules), and `fit_prior`'s `leave_one_draft_out`, `full_beta`, `paired_se`,
`cluster_se`. It is a separate entry point rather than a rung appended to
`score_ladder.ladder()` for two reasons, both about not paying for what this
rung does not need:

  - The champion is a FIXED, already-decided feature set (its
    `features_fitted`), so its held-out score is recomputed on the current
    snapshot in ONE leave-one-draft-out. Re-deriving which columns the
    champion ships -- rung 3, rung 4, and rung 4's six-way ablation -- is nine
    more backtests that decide nothing here and would roughly double the run.
  - `score_ladder.ladder()` is pinned by name to four rungs and two tests
    assert exactly that shape. This rung is the design document's "+ new
    features, linear" step above rung 4, and it lands here so those stay
    green.

THE RULE, WHICH IS NOT THIS MODULE'S TO BEND. The ESPN rung must beat the
champion on held-out human top-1 AND not worsen held-out log-loss, both
recomputed on the same snapshot in the same rooms. There is deliberately NO
force flag. A losing rung prints its numbers, writes nothing, and exits 1 --
`scoring/human_prior.py` is left byte for byte as it was, still carrying the
champion. A losing result is a complete result; the per-feature ablation is
its content, because it says which column, if any, was doing anything.

Human picks only (`autodrafted IS FALSE`, at a seat that is not ours), held
out by draft, standard errors clustered by draft. The reasons are
`score_ladder`'s and are not restated here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np

from pipeline import draft_log as dl
from pipeline import fit_prior as fp
from pipeline import score_ladder as sl
from scoring import human_prior
from scoring.draft_model import FEATURE_NAMES, _ESPN_BOARD_FEATURES

# The five columns this rung is about, and the assertion that this is exactly
# what `draft_model` defines them as -- a hand-written copy here would let the
# two drift, and an ablation over five columns while claiming to test the ESPN
# board is the silent shrink `score_ladder.ABLATION_CANDIDATES` exists to
# prevent. `ablation` takes `candidates` as a required argument for the same
# reason; this list is what `main` passes it, by name.
ESPN_CANDIDATES = list(_ESPN_BOARD_FEATURES)
assert set(ESPN_CANDIDATES) <= set(FEATURE_NAMES), (
    "an ESPN board feature is not in FEATURE_NAMES: "
    f"{sorted(set(ESPN_CANDIDATES) - set(FEATURE_NAMES))}")


def champion_features() -> list:
    """The exact columns the shipped human-only prior fits.

    Read off the artifact's own provenance rather than reconstructed, because
    the champion IS `scoring/human_prior.py` and its feature set is a decided
    fact, not something to re-derive here. `market_rank`'s `reach`/`fall` and
    the four Tier 1 pool signals that earned their place are in it; the two
    Tier 1 columns cut on delta_top1 are not.
    """
    fitted = list(human_prior.PROVENANCE["features_fitted"])
    assert set(fitted) <= set(FEATURE_NAMES), (
        "the champion names a column FEATURE_NAMES does not have -- the "
        "artifact was generated against a different feature list")
    assert not (set(fitted) & set(ESPN_CANDIDATES)), (
        "an ESPN column is already in the champion, so this rung would be "
        "measuring it against a model that already had it")
    return fitted


def champion_cut() -> list:
    """The columns the champion measured on humans and REJECTED, carried
    forward so the regenerated artifact keeps annotating them "cut" rather
    than silently re-labelling them "not measured"."""
    return list(human_prior.PROVENANCE.get("tier1_cut", []))


def _rung(label, keep_names, X_list, chosen, groups, buckets):
    return sl.Rung(label, fp.leave_one_draft_out(
        X_list, chosen, groups, buckets=buckets,
        keep_idx=fp.feature_indices(keep_names)), "leave-one-draft-out")


def espn_winners(table) -> list:
    """The ESPN columns whose own `delta_top1` is strictly above zero.

    Strictly above, not "above zero minus a standard error": a feature whose
    delta is inside its own error has not been shown to be worth anything, and
    a zero coefficient it does not carry is a cheaper mistake than one fitted
    on noise and served on every pick. Returned in `_ESPN_BOARD_FEATURES`
    order so the shipped feature list is deterministic. This mirrors
    `score_ladder.tier1_winners`, narrowed to the ESPN candidates.
    """
    kept = set(table[(table["dropped"] != "none")
                     & (table["delta_top1"] > 0)]["dropped"].tolist())
    return [f for f in ESPN_CANDIDATES if f in kept]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _coefficient_comparison(X_list, chosen, features):
    """The pooled coefficients on the two boards' reach/fall, side by side.

    THE HYPOTHESIS, TESTED DIRECTLY. If the room reads ESPN's list, then a fit
    given BOTH boards should lean on `espn_reach` and let `market_rank`'s
    `reach` collapse toward zero. `full_beta` is the pooled maximum-likelihood
    vector -- the same one that would ship -- so these are the coefficients as
    fitted, not a per-fold average. Printed with each column's own standard
    deviation because the two reaches are on slightly different scales (dense
    market rank vs raw ESPN rank, though `log1p` compresses that to ~13% at
    the deep end), so the standardized column (coef * std) is the honest
    apples-to-apples magnitude.
    """
    beta = fp.full_beta(X_list, chosen, features)
    stacked = np.vstack(X_list)
    rows = []
    for name in ("reach", "fall", "espn_reach", "espn_fall", "vor",
                 "espn_list_pos", "board_disagreement", "espn_proj_dropoff",
                 "dropoff_at_pos"):
        if name not in features:
            continue
        i = FEATURE_NAMES.index(name)
        std = float(stacked[:, i].std())
        rows.append((name, beta[i], std, beta[i] * std))
    return rows


def vor_subsumption(X_list, chosen, groups, champ, full_features,
                    champion_report, espn_full_report):
    """vor's own delta_top1 with and without the ESPN board in the model.

    THE THEORY, TESTED DIRECTLY. vor earned +0.0187 in the champion, and one
    reading is that it won only as a RANK PROXY -- a restatement of "how far up
    the board is he". If that is all it was, then adding `espn_reach`/
    `espn_fall`/`espn_list_pos`, which read a board directly, should leave vor
    little left to explain and its delta_top1 should collapse toward zero. So
    vor is dropped from the champion set (no ESPN present) and from the full
    set (ESPN present), and the two deltas are compared on the SAME snapshot.

    Reuses the champion and full reports already computed as the two "full"
    baselines, so this costs one extra leave-one-draft-out per side rather than
    recomputing a rung already in hand. Returns None if the champion does not
    carry vor (nothing to subsume).
    """
    if "vor" not in champ:
        return None
    no_espn = sl.ablation(X_list, chosen, groups, champ, ["vor"],
                          full=champion_report)
    with_espn = sl.ablation(X_list, chosen, groups, full_features, ["vor"],
                            full=espn_full_report)

    def _row(tbl):
        r = tbl[tbl["dropped"] == "vor"].iloc[0]
        return float(r["delta_top1"]), float(r["delta_top1_se"])

    return _row(no_espn), _row(with_espn)


def _prose(champion, espn_full, shipped_rung, table, keep, cut_espn,
           draft_ids, corpus_path, coeffs, delta, se) -> str:
    per_feature = "\n".join(
        f"    {row.dropped:<18} delta_top1 {row.delta_top1:+.4f} "
        f"+/-{row.delta_top1_se:.4f}   "
        f"{'kept' if row.dropped in set(keep) else 'cut'}"
        for row in table.itertuples()
        if row.dropped in set(ESPN_CANDIDATES))
    coeff_lines = "\n".join(
        f"    {name:<18} coef {coef:+.4f}   std {std:.4f}   "
        f"standardized {coef * std:+.4f}"
        for name, coef, std, _ in coeffs)
    return f"""
Fitted on {shipped_rung.report['n']} picks a person is KNOWN to have made
(`autodrafted IS FALSE`, at a seat that is not ours), across {len(draft_ids)} drafts of
the cross-league draft corpus ({corpus_path}), on {_now()}.

THE ESPN BOARD RUNG. The champion prices candidates against `market_rank`;
this rung adds five columns computed against ESPN's own on-screen list
(`espn_rank`/`espn_proj`), the board an ESPN mock lobby actually reads. Every
rung is one leave-one-draft-out over the same human picks in the same rooms.

    champion (human_prior)   top-1 {champion.top1:.4f}  top-5 {champion.report['top5']:.4f}  log-loss {champion.report['logloss']:>7.4f}
    + ESPN board, all five   top-1 {espn_full.top1:.4f}  top-5 {espn_full.report['top5']:.4f}  log-loss {espn_full.report['logloss']:>7.4f}
    + ESPN board, shipped    top-1 {shipped_rung.top1:.4f}  top-5 {shipped_rung.report['top5']:.4f}  log-loss {shipped_rung.report['logloss']:>7.4f}

delta_top1 of the shipped ESPN subset against the champion: {delta:+.4f} +/-{se:.4f},
paired by draft. The rule for this rung is top-1 UP and log-loss NOT WORSE,
both recomputed on this snapshot; there is no force flag.

EACH ESPN FEATURE, DROPPED FROM THE FULL MODEL IN TURN. Positive earns a
place; at or below zero is 0.0, meaning MEASURED AND REJECTED. The +/- is the
standard error of the delta itself, paired by draft.

{per_feature}

    kept: {', '.join(keep) or '(none)'}
    cut:  {', '.join(cut_espn) or '(none)'}

DOES ESPN'S BOARD SUBSUME THE MARKET'S? The pooled fit given both boards, its
coefficients side by side. If the room reads ESPN, `market_rank`'s `reach`
leans on `espn_reach` and gives up ground.

{coeff_lines}

WHAT THIS DOES NOT ESTABLISH.

  * Not that the refit drafts better. It predicts opponents better on one
    population; nothing here measured a roster.
  * Not that `espn_rank` was the board each draft was PLAYED against. The
    corpus backfilled it from today's preseason board (preseason-stable, the
    same provenance status as `vor`); a coefficient on it is worth
    re-measuring once every draft records its own ESPN board at record time.
  * Not that it generalises past ESPN's public mock lobby, the only
    population both labelled and large enough to hold out by draft.
""".strip("\n")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    corpus_path = fp._option(argv, "--corpus") or dl.CORPUS_PATH
    league_path = fp._option(argv, "--league-db")

    corpus = fp.open_corpus(corpus_path)
    league_conn = fp.open_league(league_path)
    try:
        snapshot = fp.snapshot_draft_ids(corpus)
        draft_ids = sl.labelled_draft_ids(corpus, snapshot)
        if not draft_ids:
            print("No draft carries a `autodrafted IS FALSE` pick -- nothing "
                  "human to measure on.")
            return 1
        print(f"Corpus snapshot: {len(snapshot)} drafts with a pool, "
              f"{len(draft_ids)} carrying at least one KNOWN human pick.\n"
              "The farm writes while this runs, so the list is fixed here:\n  "
              + "\n  ".join(draft_ids))

        corpus_obs = sl.human_observations(corpus, league_conn, draft_ids)
        X_list, chosen, groups, boards, buckets = fp.design(corpus_obs)
        print(f"\n{corpus_obs.picks_seen} picks in those drafts; "
              f"{len(chosen)} are known human picks at a seat not ours.")
        if not chosen:
            print("No human picks. Nothing to measure.")
            return 1

        champ = champion_features()
        cut_carried = champion_cut()
        full_features = champ + ESPN_CANDIDATES

        # The champion, recomputed on THIS snapshot -- not the 0.2565 literal
        # from a smaller corpus. Then the same fit plus the five ESPN columns.
        champion = _rung("champion (human_prior)", champ,
                         X_list, chosen, groups, buckets)
        espn_full = _rung("+ ESPN board, all five", full_features,
                          X_list, chosen, groups, buckets)
        print("\nHeld-out human top-1, scored on the same "
              f"{len(chosen)} picks in {len(set(groups))} rooms "
              "(+/- clustered by draft):")
        for rung in (champion, espn_full):
            sl._print_rung(rung)
        full_delta = espn_full.top1 - champion.top1
        full_se = fp.paired_se(espn_full.report["by_draft"],
                               champion.report["by_draft"])
        print(f"\n  + ESPN board, all five - champion   delta_top1 "
              f"{full_delta:+.4f} +/-{full_se:.4f}  (paired by draft)")

        # Per-feature ablation over the five ESPN columns, `score_ladder`'s own
        # function with `candidates=` passed by name. Rows print as they land
        # so a run the OS kills still leaves every measured number on disk.
        print(f"\nPer-feature ablation over the {len(ESPN_CANDIDATES)} ESPN "
              "board columns -- the columns this rung ADDS. Each row is the "
              "full model refit\nwithout that one column (a complete "
              "leave-one-draft-out backtest); delta_top1 > 0\nearns a place.")
        table = sl.ablation(X_list, chosen, groups, full_features,
                            ESPN_CANDIDATES, full=espn_full.report,
                            on_row=sl._print_ablation_row)

        print("\nHow much of a chance each ESPN column had. `sets_varying` is "
              "the fraction of choice\nsets where it is not constant (a "
              "constant column cancels out of the softmax);\n`nonzero_share` "
              "is the fraction of candidate rows it fires on at all.")
        print(sl.column_coverage(X_list, ESPN_CANDIDATES).to_string(index=False))

        coeffs = _coefficient_comparison(X_list, chosen, full_features)
        print("\nDOES ESPN'S BOARD SUBSUME THE MARKET'S? Pooled coefficients, "
              "both boards in one fit.\nStandardized = coef * column std, the "
              "scale-free magnitude:")
        for name, coef, std, standardized in coeffs:
            print(f"  {name:<18} coef {coef:+.4f}   std {std:.4f}   "
                  f"standardized {standardized:+.4f}")

        # DOES THE ESPN BOARD SUBSUME vor'S WIN? vor's own delta_top1, dropped
        # from the model without the ESPN board and then with it, on this same
        # snapshot. If vor won only as a rank proxy, its delta collapses toward
        # zero once a board is read directly. Diagnostic only -- it changes
        # nothing about what ships.
        vor_sub = vor_subsumption(X_list, chosen, groups, champ,
                                  full_features, champion.report,
                                  espn_full.report)
        if vor_sub is not None:
            (vd0, vse0), (vd1, vse1) = vor_sub
            print("\nDOES THE ESPN BOARD SUBSUME vor'S WIN? vor's own "
                  "delta_top1, dropped from the model\nwithout the ESPN board "
                  "and then with it. If vor won only as a rank proxy, its "
                  "delta\ncollapses toward zero once a board is read directly.")
            print(f"  vor delta_top1, no ESPN in model   {vd0:+.4f} "
                  f"+/-{vse0:.4f}")
            print(f"  vor delta_top1, ESPN in model      {vd1:+.4f} "
                  f"+/-{vse1:.4f}")

        keep = espn_winners(table)
        cut_espn = [f for f in ESPN_CANDIDATES if f not in set(keep)]
        print(f"\nOf the five, {len(keep)} earned a place and {len(cut_espn)} "
              f"did not.\n  kept: {', '.join(keep) or '(none)'}"
              f"\n  cut:  {', '.join(cut_espn) or '(none)'}")

        if not keep:
            print("\nNo ESPN column earned a place. The subset is the champion "
                  "itself, so there is\nnothing to ship. "
                  f"{sl.HUMAN_PRIOR_MODULE.name} is left as it was. A negative "
                  "result is a\ncomplete result -- the per-feature table above "
                  "is its content.")
            return 1

        shipped_features = champ + keep
        shipped_rung = _rung("+ ESPN board, shipped", shipped_features,
                             X_list, chosen, groups, buckets)
        sl._print_rung(shipped_rung)
        delta = shipped_rung.top1 - champion.top1
        se = fp.paired_se(shipped_rung.report["by_draft"],
                          champion.report["by_draft"])
        dll = shipped_rung.report["logloss"] - champion.report["logloss"]
        print(f"\ndelta_top1 (shipped ESPN subset - champion): {delta:+.4f} "
              f"+/-{se:.4f}, paired by draft.\ndelta_logloss: {dll:+.4f} "
              "(negative is better; the gate is <= 0).")

        # THE RULE. top-1 strictly up AND log-loss not worse, both on this
        # snapshot. One expression, so changing it is a diff to justify.
        if not (delta > 0 and shipped_rung.report["logloss"]
                <= champion.report["logloss"]):
            reason = ("did not improve top-1" if delta <= 0
                      else "worsened log-loss")
            print(f"\nThe ESPN board rung {reason} against the champion "
                  f"({delta:+.4f} top-1, {dll:+.4f}\nlog-loss). Nothing is "
                  f"written; {sl.HUMAN_PRIOR_MODULE.name} still holds the "
                  "champion.\nA negative result is a complete result -- the "
                  "per-feature table and the coefficient\ncomparison above "
                  "are its content.")
            return 1

        # The pooled vector that ships: fitted on the shipped columns only,
        # scattered back to full length with 0.0 elsewhere. `full_beta` and
        # not `fit`-then-zero, for the reason it documents.
        beta = fp.full_beta(X_list, chosen, shipped_features)
        provenance = {
            "fitted_at": _now(),
            "corpus": corpus_path,
            "population": "autodrafted IS FALSE (known human), not our seat",
            "drafts": len(draft_ids),
            "picks_scored": len(chosen),
            "picks_excluded": dict(corpus_obs.dropped),
            "folds": "leave-one-draft-out",
            "rung": "ESPN board features (+ on the human-only champion)",
            "features_fitted": list(shipped_features),
            "tier1_kept": list(human_prior.PROVENANCE.get("tier1_kept", [])),
            "tier1_cut": list(cut_carried),
            "espn_kept": list(keep),
            "espn_cut": list(cut_espn),
            "espn_delta_top1": {
                row.dropped: round(row.delta_top1, 6)
                for row in table.itertuples()
                if row.dropped in set(ESPN_CANDIDATES)},
            "espn_delta_top1_se": {
                row.dropped: round(row.delta_top1_se, 6)
                for row in table.itertuples()
                if row.dropped in set(ESPN_CANDIDATES)},
            "espn_rank_provenance": "backfilled from today's preseason ESPN "
                                    "board (preseason-stable, as vor)",
            "champion_top1": round(champion.top1, 6),
            "champion_logloss": round(champion.report["logloss"], 6),
            "espn_all_five_top1": round(espn_full.top1, 6),
            "refit_top1": round(shipped_rung.top1, 6),
            "refit_top5": round(shipped_rung.report["top5"], 6),
            "refit_logloss": round(shipped_rung.report["logloss"], 6),
            "delta_top1": round(delta, 6),
            "delta_top1_se": round(se, 6),
            "delta_logloss": round(dll, 6),
            "reach_coef": round(dict((n, c) for n, c, _, _ in coeffs)
                                .get("reach", float("nan")), 6),
            "espn_reach_coef": round(dict((n, c) for n, c, _, _ in coeffs)
                                     .get("espn_reach", float("nan")), 6),
        }
        # The cut set the artifact annotates: the champion's own rejected
        # columns (still rejected) plus any ESPN column that lost here.
        cut = list(cut_carried) + list(cut_espn)
        target = sl.write_human_prior(
            beta, FEATURE_NAMES, draft_ids, provenance,
            _prose(champion, espn_full, shipped_rung, table, keep, cut_espn,
                   draft_ids, corpus_path, coeffs, delta, se),
            fitted=shipped_features, cut=cut)
        print(f"\nThe ESPN board rung wins ({delta:+.4f} top-1 +/-{se:.4f}, "
              f"{dll:+.4f} log-loss). Wrote {target}.\nNothing imports it yet "
              "-- see that file's docstring for why the serving swap waits.")
        return 0
    finally:
        corpus.close()
        league_conn.close()


if __name__ == "__main__":
    sys.exit(main())
