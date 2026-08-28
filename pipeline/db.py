import os
from pathlib import Path
import duckdb
import pandas as pd

# Env-overridable so a second instance (a test harness, a staging server) can
# run against a copy without colliding on the primary file's single-writer
# lock. Falls back to the repo's real database, unchanged, when unset.
DEFAULT_PATH = os.environ.get("DRAFT_DB_PATH", "data/nfl.duckdb")

# The two kinds of table, kept in one place because the whole multi-league
# design depends on the split being correct. A per-league database holds only
# LEAGUE_TABLES; UNIVERSAL_TABLES are read from the shared file. A table that
# is neither would be treated as universal by provisioning and leak one
# league's rows into every other -- so the classification is asserted complete
# in tests rather than left implicit.
LEAGUE_TABLES = frozenset({
    "league", "draft_picks", "draft_teams", "draft_order",
    "manager_profiles", "manager_tendencies", "drafted",
    "sim_board", "sim_results", "sim_survival",
    # How each season ended (pipeline/espn_league.parse_standings) and the
    # stored league reports built on it (api/reports.py). Both are one
    # league's business and nobody else's.
    "league_standings", "league_reports",
    # A league's own history beyond the draft (pipeline/league_activity.py):
    # who is in it, who played whom, every transaction, every week's
    # lineup, the raw ESPN answers those were parsed from. One league's
    # business and nobody else's.
    "league_members", "league_matchups", "league_transactions",
    "league_lineups", "league_draft_flags", "league_raw",
})
UNIVERSAL_TABLES = frozenset({
    "weekly", "snap_counts", "depth_charts", "players", "schedules",
    "adp", "espn_adp", "fp_ecr", "cbs_ranks", "mfl_adp",
    "historic_adp", "historic_espn_cs", "sleeper_ids", "meta", "model_backtest",
    # Season-long player betting markets (pipeline/sources.fetch_player_futures).
    # Universal for the same reason `adp` is: what a book prices Gibbs at to
    # lead the league in rushing does not depend on whose league is reading it.
    "player_futures",
    # ESPN's projected stat lines, every season since 2018
    # (pipeline/espn_projections.py). Universal for the same reason
    # `espn_adp` is: what ESPN projects Swift to carry is one fact, not one
    # per league. It is the projected column on the profile's usage card.
    "espn_projections",
    # ESPN's per-week D/ST stat lines (pipeline/sources.fetch_espn_dst).
    # Universal for the same reason `weekly` is: what the Broncos defense did
    # in week 4 is the same fact in every league. The LEAGUE-specific half is
    # the price list, which lives in `league.settings_json` as `dst_scoring`,
    # and the two are only combined at scoring time.
    "dst_weekly",
    # Player news + injury signals (pipeline/news.py). Universal, not
    # per-league: an article about Ja'Marr Chase says the same thing in
    # every league, so a new league's file should be seeded with whatever
    # has already been fetched rather than starting with an empty feed.
    "player_news", "player_status",
})

# Per-connection DuckDB limits. DuckDB's defaults size one database for the
# whole machine -- a thread per core, most of the RAM -- and the app holds
# one connection per live room. Two sizes: a WORKER connection builds a
# board and a pool (reads `weekly` whole, once) and is closed minutes later,
# so it gets real headroom; the app's own long-lived per-room connection
# only ever reads `drafted`, `league`, `draft_order` and the sim tables --
# a few hundred rows -- and two hundred of them must not each be allowed
# to grow to a quarter of a gigabyte.
WORKER_CONN_THREADS = 2
WORKER_CONN_MEMORY_LIMIT = "256MB"
PARENT_CONN_THREADS = 1
PARENT_CONN_MEMORY_LIMIT = "32MB"


def apply_conn_limits(conn, threads: int, memory_limit: str) -> None:
    conn.execute(f"SET threads={int(threads)}")
    conn.execute(f"SET memory_limit='{memory_limit}'")


def apply_worker_conn_limits(conn) -> None:
    apply_conn_limits(conn, WORKER_CONN_THREADS, WORKER_CONN_MEMORY_LIMIT)


def apply_parent_conn_limits(conn) -> None:
    apply_conn_limits(conn, PARENT_CONN_THREADS, PARENT_CONN_MEMORY_LIMIT)


def get_conn(path: str = DEFAULT_PATH) -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS meta (
        source VARCHAR PRIMARY KEY, ok BOOLEAN, rows BIGINT, refreshed_at TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS drafted (
        player_id VARCHAR PRIMARY KEY, pick_no INTEGER)""")
    # Databases created before pick_no existed are missing the column; adding
    # it here keeps an existing data/nfl.duckdb usable without a manual drop.
    columns = {r[1] for r in conn.execute("PRAGMA table_info('drafted')").fetchall()}
    if "pick_no" not in columns:
        conn.execute("ALTER TABLE drafted ADD COLUMN pick_no INTEGER")
    # PUT /api/draft-order persists via write_table, which does
    # CREATE OR REPLACE TABLE ... AS SELECT and so replaces this table's
    # inferred schema (no PRIMARY KEY) on every save. This initial DDL only
    # documents the intended shape -- the slot-uniqueness guarantee actually
    # lives in api/main.py's application-level duplicate-slot check, not here.
    conn.execute("""CREATE TABLE IF NOT EXISTS draft_order (
        slot INTEGER PRIMARY KEY, manager VARCHAR, is_me BOOLEAN)""")
    return conn

def write_table(conn, name: str, df: pd.DataFrame) -> None:
    conn.register("_incoming", df)
    conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _incoming")
    conn.unregister("_incoming")

def read_table(conn, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """The whole table as a frame, or only `columns` of it.

    `columns` is a projection pushed into SQL rather than a `df[cols]` after
    the fact, and the difference is the whole point: `weekly` is 174k rows
    across 150 columns and materializes as 244 MB of pandas, of which ~118 MB
    is object columns nothing scores (`headshot_url`, `game_id`,
    `season_type`, the `fg_*_list` strings...). Reading those out of DuckDB
    only to drop them costs the memory anyway, and on a box serving several
    drafts at once that memory is the constraint. See
    `scoring.board.WEEKLY_COLUMNS` for the caller this was written for and
    how its list is derived.

    A NAME NOT IN THE TABLE IS SKIPPED, NOT AN ERROR. Test fixtures seed a
    ten-column `weekly` (tests/test_board.py, tests/test_live_api.py) and an
    older database predates whatever column was added last; either way the
    caller's list is what it WANTS, and DuckDB would raise `Binder Error` on
    the first one that is missing. Every consumer of a projected frame
    already guards optional columns (`if c in wk.columns`), so absent is a
    state they handle and a hard failure is not.

    A list that intersects the table in nothing returns an empty frame -- the
    same answer a missing table gives, and the same one the callers' `.empty`
    checks already handle.

    THE FRAME'S COLUMNS COME BACK IN THE CALLER'S ORDER, not the table's, and
    duplicates in the list are read once: `columns` is a request, and the
    frame is the answer to it. Nothing downstream indexes a projected frame
    positionally today -- every consumer names its columns -- but a caller
    that reads its own list back off the frame should get its own list."""
    exists = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()[0]
    if not exists:
        return pd.DataFrame()
    if columns is None:
        return conn.execute(f"SELECT * FROM {name}").df()
    have = {r[1] for r in conn.execute(f"PRAGMA table_info('{name}')").fetchall()}
    want, seen = [], set()
    for col in columns:
        if col in have and col not in seen:
            seen.add(col)
            want.append(col)
    if not want:
        return pd.DataFrame()
    projection = ", ".join(f'"{c}"' for c in want)
    return conn.execute(f"SELECT {projection} FROM {name}").df()

def record_freshness(conn, source: str, ok: bool, rows: int) -> None:
    conn.execute(
        """INSERT INTO meta VALUES (?, ?, ?, now())
           ON CONFLICT (source) DO UPDATE SET
             ok = excluded.ok, rows = excluded.rows, refreshed_at = excluded.refreshed_at""",
        [source, ok, rows])
