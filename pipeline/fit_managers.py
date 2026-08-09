"""Fit manager models and write profiles. Run: python -m pipeline.fit_managers"""
import sys

from pipeline.db import get_conn, record_freshness
from scoring.draft_model import backtest, write_profiles


def main() -> int:
    conn = get_conn()
    profiles = write_profiles(conn)
    if profiles.empty:
        print("No draft history found -- run `make espn-import` first.")
        return 1
    record_freshness(conn, "manager_profiles", True, len(profiles))

    report = backtest(conn)
    personal = profiles[profiles["uses_personal"]]["manager"].nunique()
    total = profiles["manager"].nunique()
    print(f"Fitted {total} managers ({personal} with personal models, "
          f"{total - personal} pooled).")
    print(f"Backtest on {report['holdout_season']}: "
          f"top-1 {report['top1']:.0%}, top-5 {report['top5']:.0%}, "
          f"log-loss {report['logloss']:.3f} "
          f"(ADP baseline {report['adp_logloss']:.3f})")
    if not report["beats_adp"]:
        print("WARNING: the fitted model does not beat the ADP baseline "
              "out of sample. Treat simulator output as indicative only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
