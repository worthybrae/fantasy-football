"""Run one experiment, or all of them, and persist what they measured."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research import data, lab                                   # noqa: E402
from research.experiments import e001_strength_of_schedule       # noqa: E402

ALL = {e.EXPERIMENT.id: e.EXPERIMENT for e in (e001_strength_of_schedule,)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", nargs="*", help="experiment ids; default all")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    import duckdb
    from scoring import league
    from scoring.ppr import normalize_rules

    db = Path("data/nfl.duckdb")
    conn = duckdb.connect(str(db), read_only=True)
    rules = normalize_rules(league.load(conn).scoring)
    odds = data.odds()

    for eid in (args.ids or sorted(ALL)):
        exp = ALL[eid]
        print(f"running {exp.id}: {exp.title}")
        result = lab.run(exp, conn, odds, rules)
        print(f"  -> {len(result.findings)} findings at {result.git_rev or 'unknown rev'}")
        for note in result.notes:
            print(f"  note: {note[:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
