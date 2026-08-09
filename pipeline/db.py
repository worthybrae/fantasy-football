from pathlib import Path
import duckdb
import pandas as pd

DEFAULT_PATH = "data/nfl.duckdb"

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
