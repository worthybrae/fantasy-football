"""Fit manager models and write profiles. Run: python -m pipeline.fit_managers

Pass `--reduced` (or `make fit-managers REDUCED=1`) to additionally measure
whether a smaller per-manager model generalizes where the full one doesn't.
It is off by default because it nests a cross-validation inside a
cross-validation for every candidate and takes minutes, not seconds.
"""
import sys

from pipeline.db import get_conn, record_freshness
from scoring.draft_model import (ablation, positional_bias,
                                 reduced_model_report, write_backtest,
                                 write_profiles)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    conn = get_conn()
    profiles = write_profiles(conn)
    if profiles.empty:
        print("No draft history found -- run `make espn-import` first.")
        return 1
    record_freshness(conn, "manager_profiles", True, len(profiles))

    # Persisted, not just printed: the board reads it back through
    # /api/model and says so in the rail when the model loses to ADP.
    report = write_backtest(conn)
    personal = profiles[profiles["uses_personal"]]["manager"].nunique()
    total = profiles["manager"].nunique()
    print(f"Fitted {total} managers ({personal} with personal models, "
          f"{total - personal} pooled).")
    # The number that decides `uses_personal`, printed rather than buried in
    # the table: it is the whole argument for why a manager card says
    # "league average", and it is not readable from the summary line alone.
    print("\nPer-manager held-out gain (log-likelihood per pick against the "
          "pooled fit;\npositive means their own coefficients predict a "
          "held-out season better):")
    print(profiles.drop_duplicates("manager")[
        ["manager", "n_picks", "heldout_gain", "uses_personal"]
    ].to_string(index=False))
    print(f"\nBacktest, leave-one-season-out over {report['seasons']}:")
    print(f"  overall: top-1 {report['top1']:.0%}, top-5 {report['top5']:.0%}, "
          f"log-loss {report['logloss']:.3f} (ADP baseline {report['adp_logloss']:.3f})")
    for r in report["by_round"]:
        print(f"  {r['round_bucket']:>5}: top-1 {r['top1']:.0%}, "
              f"top-5 {r['top5']:.0%}  (n={r['n']})")
    print("\nFeature ablation (delta_top1 > 0 means the feature earns its place):")
    print(ablation(conn).to_string(index=False))
    print("\nLeague positional bias (positive = drafted ahead of the market):")
    print(positional_bias(conn).to_string(index=False))
    if "--reduced" in argv:
        print("\nReduced per-manager models (gain against pooled, same "
              "leave-one-season-out\nyardstick as heldout_gain above). Read "
              "the `nested` row: it is the only one\nwhose subset was chosen "
              "without seeing the season it is scored on.")
        print(reduced_model_report(conn).to_string(index=False))
    if not report["beats_adp"]:
        print("WARNING: the fitted model does not beat the ADP baseline "
              "out of sample. Treat simulator output as indicative only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
