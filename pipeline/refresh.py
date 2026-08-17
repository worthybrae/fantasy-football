"""Refresh all data sources into DuckDB. Run: python -m pipeline.refresh"""
import sys
import pandas as pd
from pipeline import sources
from pipeline.db import get_conn, write_table, record_freshness
from scoring.config import CURRENT_SEASON, HISTORY_SEASONS

# Which of the three scoring-format tokens ('ppr' | 'half' | 'std') each
# format-aware source's fetcher actually supports -- see the per-source
# comments in pipeline/sources.py for how each was verified against the live
# site. A source not listed here (espn_adp) is left PPR-only on purpose:
# ESPN's rank is explicitly PPR-scored, there's no other format to fetch.
FORMATS_BY_SOURCE = {
    "adp": sources.FORMAT_TOKENS,          # FFC: all three are real pages
    "fp_ecr": sources.FORMAT_TOKENS,       # FantasyPros: all three 200 (half/std share a page)
    "cbs_ranks": ("ppr", "std"),           # CBS: half-ppr/top200/ 404s
    "mfl_adp": ("ppr", "std"),             # MFL: IS_PPR is binary, no half option
}

def _fetch_multi_format(fetch_fn, formats, *args) -> pd.DataFrame:
    """Fetch every format a source supports and stack the results into one
    table, each row already tagged with its own `format` column by the
    fetcher itself. A single format failing (upstream 404, schema drift,
    timeout, ...) is logged and skipped rather than aborting the whole
    source -- the same isolation main() applies between sources, one level
    down. Only if every format fails does this raise, which then surfaces as
    this source's normal FAIL line below."""
    frames = []
    for fmt in formats:
        try:
            frames.append(fetch_fn(*args, fmt=fmt))
        except Exception as e:
            print(f"  WARN {fetch_fn.__name__} [{fmt}]: {e}")
    if not frames:
        raise ValueError(f"{fetch_fn.__name__}: no format returned data")
    return pd.concat(frames, ignore_index=True)

def main() -> int:
    conn = get_conn()
    jobs = {
        "weekly": lambda: sources.fetch_weekly(HISTORY_SEASONS),
        "snap_counts": lambda: sources.fetch_snap_counts(HISTORY_SEASONS),
        "depth_charts": lambda: sources.fetch_depth_charts(CURRENT_SEASON),
        "schedules": lambda: sources.fetch_schedules(CURRENT_SEASON),
        "adp": lambda: _fetch_multi_format(
            sources.fetch_adp, FORMATS_BY_SOURCE["adp"], CURRENT_SEASON),
        "players": lambda: sources.fetch_players(),
        "mfl_adp": lambda: _fetch_multi_format(
            sources.fetch_mfl_adp, FORMATS_BY_SOURCE["mfl_adp"], CURRENT_SEASON),
        "cbs_ranks": lambda: _fetch_multi_format(
            sources.fetch_cbs, FORMATS_BY_SOURCE["cbs_ranks"]),
        "espn_adp": lambda: sources.fetch_espn_adp(CURRENT_SEASON),  # PPR-only; not format-aware
        "fp_ecr": lambda: _fetch_multi_format(
            sources.fetch_fp_ecr, FORMATS_BY_SOURCE["fp_ecr"]),
        "sleeper_ids": sources.fetch_sleeper_ids,
    }
    summary = []
    any_failed = False
    for name, job in jobs.items():
        try:
            df = job()
            write_table(conn, name, df)
            record_freshness(conn, name, True, len(df))
            summary.append(f"  OK   {name}: {len(df)} rows")
        except Exception as e:  # one source failing must not abort the rest
            record_freshness(conn, name, False, 0)
            summary.append(f"  FAIL {name}: {e}")
            any_failed = True
    print("Refresh complete:\n" + "\n".join(summary))
    return 1 if any_failed else 0

if __name__ == "__main__":
    sys.exit(main())
