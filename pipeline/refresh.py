"""Refresh all data sources into DuckDB. Run: python -m pipeline.refresh"""
import sys
from pipeline import sources
from pipeline.db import get_conn, write_table, record_freshness
from scoring.config import CURRENT_SEASON, HISTORY_SEASONS

def main() -> int:
    conn = get_conn()
    jobs = {
        "weekly": lambda: sources.fetch_weekly(HISTORY_SEASONS),
        "snap_counts": lambda: sources.fetch_snap_counts(HISTORY_SEASONS),
        "depth_charts": lambda: sources.fetch_depth_charts(CURRENT_SEASON),
        "schedules": lambda: sources.fetch_schedules(CURRENT_SEASON),
        "adp": lambda: sources.fetch_adp(CURRENT_SEASON),
        "players": lambda: sources.fetch_players(),
        "mfl_adp": lambda: sources.fetch_mfl_adp(CURRENT_SEASON),
        "cbs_ranks": lambda: sources.fetch_cbs(),
        "espn_adp": lambda: sources.fetch_espn_adp(CURRENT_SEASON),
        "fp_ecr": sources.fetch_fp_ecr,
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
