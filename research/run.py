"""Run one experiment, or all of them, and persist what they measured."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research import data, lab                                   # noqa: E402


def discover() -> dict:
    """Every experiment module that declares an EXPERIMENT, found by import.

    Discovered rather than listed because a hand-kept registry is a merge
    conflict waiting to happen -- three experiments were written in parallel
    and every one of them would have edited the same tuple. A module that
    forgets to declare EXPERIMENT is skipped silently on purpose: it is
    usually a helper, not a broken experiment.
    """
    import importlib
    import pkgutil

    import research.experiments as pkg

    found = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        m = importlib.import_module(f"research.experiments.{mod.name}")
        exp = getattr(m, "EXPERIMENT", None)
        if exp is not None:
            if exp.id in found:
                raise ValueError(
                    f"two experiments claim id {exp.id!r}: "
                    f"{found[exp.id].title!r} and {exp.title!r}")
            found[exp.id] = exp
    return found


ALL = discover()


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

    # The index is built from stored results, so it is only ever as current
    # as the last run -- rebuild it here rather than leaving it to be
    # forgotten.
    from research.index import build
    print("index:", build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
