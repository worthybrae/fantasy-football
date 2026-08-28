"""Refresh all data sources into DuckDB. Run: python -m pipeline.refresh"""
import sys
import pandas as pd
from pipeline import espn_projections, news, sources
from pipeline.db import get_conn, read_table, record_freshness, write_table
from scoring.config import CURRENT_SEASON, HISTORY_SEASONS, RECENCY_WEIGHTS

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

def main(warm: bool = False) -> int:
    """Refresh every source. `warm` rebuilds the caches this just retired.

    OFF BY DEFAULT, because the caller that wants it is the server and the
    caller that does not is the command line: `python -m pipeline.refresh`
    exits the moment this returns, so warming there builds three frames into
    a process that is about to free them. api/jobs `_run_refresh` runs this
    IN THE SERVER PROCESS, where those caches live and where the next
    request is about to want them, and passes True.
    """
    conn = get_conn()

    # Both news jobs want the same 249 board rows, and building the board
    # costs ~1.5s, so build it at most once per refresh -- and only if a news
    # job actually runs, which is why this is a closure and not a local
    # computed above the job table.
    _pool = {}

    def news_pool():
        if "pool" not in _pool:
            _pool["pool"] = news.news_pool(conn)
        return _pool["pool"]

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
        # ESPN's projected stat lines -- the carries, targets and yards
        # behind `espn_adp`'s one points number -- which the profile's usage
        # card prints as its projected column. Every season since 2018, not
        # just this one: write_table replaces the whole table, and the
        # history is what research/experiments benchmark our own model
        # against. Nine paced requests, ~12s. This was a hand-run script for
        # its first weeks, which is why a deployment refreshed by
        # api/jobs.py alone served the card with the column missing.
        "espn_projections": lambda: espn_projections.fetch_all(),
        # ESPN's own per-week D/ST stat lines -- the only source of team-defense
        # rows anywhere in this pipeline, since nflverse `weekly` carries
        # individual defenders and no defense. Scoped to the board's scoring
        # window (RECENCY_WEIGHTS), not HISTORY_SEASONS: that is exactly the
        # span `scoring.board` weights, and each season is a separate ~1MB
        # request, so fetching 2016 would cost seven more calls for rows
        # nothing reads.
        "dst_weekly": lambda: sources.fetch_espn_dst(sorted(RECENCY_WEIGHTS)),
        "fp_ecr": lambda: _fetch_multi_format(
            sources.fetch_fp_ecr, FORMATS_BY_SOURCE["fp_ecr"]),
        "sleeper_ids": sources.fetch_sleeper_ids,
        # Season-long player futures. Cheap (one request) and allowed to come
        # back empty: these are posted months before a season and pulled after
        # it, so "not up yet" is an ordinary state rather than a failure.
        "player_futures": lambda: sources.fetch_player_futures(CURRENT_SEASON),
        # LAST, and in this order, on purpose. Both read the board, which
        # reads almost every table above them, so they have to run after
        # those are current; and player_news is the slow one (~223 Google
        # News queries), so a refresh interrupted part-way still has every
        # cheap source and the injury table done.
        "player_status": lambda: news.fetch_player_status(news_pool()),
        # `existing` is what makes the TTL skip and the carry-on-failure
        # behaviour possible: write_table does CREATE OR REPLACE, so rows we
        # do not re-fetch have to be handed back in or they are gone.
        "player_news": lambda: news.fetch_player_news(
            news_pool(), existing=read_table(conn, "player_news")),
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

    # REBUILD THE CACHES THE LOOP ABOVE JUST RETIRED. Every key in
    # scoring/board_cache.py and its two siblings carries a fingerprint of
    # `meta`, and `record_freshness` moved a row of it for every source --
    # so the instant this loop ends, nothing in the process holds a usable
    # cached board and the next person to click pays the whole cold build.
    # Here nobody is waiting for it.
    #
    # Imported inside the branch rather than at module scope: this module is
    # imported by tooling that has no interest in the scoring stack, and the
    # command-line path never takes this at all. `warm` swallows and logs
    # whatever goes wrong.
    if warm:
        from scoring import board_cache
        board_cache.warm(conn)
    return 1 if any_failed else 0

if __name__ == "__main__":
    sys.exit(main())
