from pathlib import Path
import duckdb
import pandas as pd

DEFAULT_PATH = "data/nfl.duckdb"

def get_conn(path: str = DEFAULT_PATH) -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS meta (
        source VARCHAR PRIMARY KEY, ok BOOLEAN, rows BIGINT, refreshed_at TIMESTAMP)""")
    conn.execute("CREATE TABLE IF NOT EXISTS drafted (player_id VARCHAR PRIMARY KEY)")
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
