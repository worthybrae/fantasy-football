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
})
UNIVERSAL_TABLES = frozenset({
    "weekly", "snap_counts", "depth_charts", "players", "schedules",
    "adp", "espn_adp", "fp_ecr", "cbs_ranks", "mfl_adp",
    "historic_adp", "historic_espn_cs", "sleeper_ids", "meta", "model_backtest",
    # Player news + injury signals (pipeline/news.py). Universal, not
    # per-league: an article about Ja'Marr Chase says the same thing in
    # every league, so a new league's file should be seeded with whatever
    # has already been fetched rather than starting with an empty feed.
    "player_news", "player_status",
})

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

def read_table(conn, name: str) -> pd.DataFrame:
    exists = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()[0]
    if not exists:
        return pd.DataFrame()
    return conn.execute(f"SELECT * FROM {name}").df()

def record_freshness(conn, source: str, ok: bool, rows: int) -> None:
    conn.execute(
        """INSERT INTO meta VALUES (?, ?, ?, now())
           ON CONFLICT (source) DO UPDATE SET
             ok = excluded.ok, rows = excluded.rows, refreshed_at = excluded.refreshed_at""",
        [source, ok, rows])
