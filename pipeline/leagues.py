"""Where each league's data lives, and how a new one is provisioned.

Per-league database files: the heavy universal data (nflverse stats, market
ranks) stays in one shared file, and each league gets a small file with only
its own draft history, manager fits and live-draft state. A connection points
at one league's file, so the rest of the app -- every read_table(conn, ...)
site -- never has to know which league it is looking at.

The default league is the existing single database, so the single-user path
this project started as does not move.
"""
import os
import re
from pathlib import Path

from pipeline.db import DEFAULT_PATH

DEFAULT_LEAGUE = "__default__"
LEAGUES_ROOT = "data/leagues"
# The one real league data/nfl.duckdb already belongs to -- this project's
# own league, with its draft history and fitted managers already in it. A
# connect naming this id is the existing user reconnecting to their own
# draft, not a stranger we have no history for, so it must resolve to that
# same file rather than a fresh, cold-started per-league one. Overridable via
# env var so a deployment for a different real league doesn't need a code
# change; read via os.environ.get so it is fixed once per process start,
# same as every other piece of deployment config.
DEFAULT_LEAGUE_ID = os.environ.get("DEFAULT_LEAGUE_ID", "53929318")


def _safe_id(league_id: str) -> str:
    """A filename that cannot escape the leagues directory.

    The id arrives from a URL the user pasted, so it is untrusted. Keep only
    the characters a real ESPN league id and season use; anything else --
    slashes, dots, traversal -- is stripped rather than escaped, because a
    stripped id that collides is a visible wrong-league error while a
    traversal that succeeds is a filesystem write outside the sandbox.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", str(league_id))
    return cleaned or "unknown"


def league_db_path(league_id: str, root: str = LEAGUES_ROOT) -> str:
    """The database file for a league. The default league -- the
    __default__ sentinel, or a real league id equal to this deployment's
    configured DEFAULT_LEAGUE_ID -- is the shared single-user database,
    unchanged; every other league is its own file."""
    if league_id == DEFAULT_LEAGUE or league_id == DEFAULT_LEAGUE_ID:
        return DEFAULT_PATH
    return str(Path(root) / f"{_safe_id(league_id)}.duckdb")


from pipeline.db import (UNIVERSAL_TABLES, apply_worker_conn_limits, get_conn,
                         read_table, write_table)


# The read-only copy of the universal database that provisioning reads
# from. The app holds the real file read-write for its whole life, and
# DuckDB's file lock is per process, so a build worker cannot open it --
# and copying a table through pandas in the app process leaves hundreds
# of megabytes resident that the allocator never returns (measured: the
# first pandas provision +860 MB, every later one +30-100 MB, for good).
# So the app writes this snapshot once at startup and again after every
# refresh, and every provisioning -- worker or app -- attaches it
# read-only and copies table to table inside DuckDB, which streams and
# frees. Several processes may hold it read-only at once.
SNAPSHOT_NAME = "universal-snapshot.duckdb"


def snapshot_path_for(universal_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(universal_path)),
                        SNAPSHOT_NAME)


def snapshot_universal(conn, universal_path: str,
                       snapshot_path: str | None = None) -> str | None:
    """Write the read-only snapshot: CHECKPOINT the live file so it holds
    everything committed, then copy it into place atomically. Returns the
    snapshot's path, or None when there is no file to copy (an in-memory
    database in a test). A reader mid-ATTACH on the old snapshot keeps the
    old inode; the replace never disturbs it."""
    import shutil
    if not universal_path or not os.path.exists(universal_path):
        return None
    target = snapshot_path or snapshot_path_for(universal_path)
    conn.execute("CHECKPOINT")
    tmp = f"{target}.tmp"
    shutil.copyfile(universal_path, tmp)
    os.replace(tmp, target)
    return target


def _provision_by_attach(path: str, snapshot: str) -> None:
    """Seed a league file from the snapshot inside DuckDB: ATTACH read-only,
    CREATE TABLE ... AS SELECT per universal table, DETACH. 1.9 s against
    the real 215 MB database (the pandas copy is 3.0 s) and, the reason it
    exists, no resident memory left behind in the caller."""
    dst = get_conn(path)
    try:
        apply_worker_conn_limits(dst)
        dst.execute(f"ATTACH '{snapshot}' AS src (READ_ONLY)")
        try:
            have = {r[0] for r in dst.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_catalog = 'src'").fetchall()}
            for table in sorted(UNIVERSAL_TABLES):
                if table in have:
                    dst.execute(
                        f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM src.{table}")
        finally:
            dst.execute("DETACH src")
        dst.execute("CHECKPOINT")
    finally:
        dst.close()


def provision_league(league_id: str, universal_path: str,
                     root: str = LEAGUES_ROOT,
                     snapshot: str | None = None) -> str:
    """Ensure a league's database exists, seeded with the universal tables.

    Idempotent: an existing file is left exactly as it is, so reconnecting
    mid-draft never wipes the `drafted` rows already recorded. Only the
    universal tables are copied; the league-specific tables start absent,
    which is precisely the cold-start state `cold_start_fits` handles.

    With `snapshot` naming a readable file the copy is DuckDB-native from
    that snapshot (see SNAPSHOT_NAME); otherwise, or if the attach fails,
    it is the table-by-table pandas copy from `universal_path`. A build
    worker must pass the snapshot: it cannot open `universal_path`, which
    the app process holds.
    """
    path = league_db_path(league_id, root=root)
    if league_id in (DEFAULT_LEAGUE, DEFAULT_LEAGUE_ID) or Path(path).exists():
        return path
    if snapshot and os.path.exists(snapshot):
        try:
            _provision_by_attach(path, snapshot)
            return path
        except Exception as exc:      # noqa: BLE001 -- the pandas copy is
            # the fallback for any refusal (a snapshot mid-replace, an
            # older DuckDB that cannot read it); a half-made file must not
            # be left to be mistaken for a provisioned league.
            print(f"leagues: attach provisioning of {league_id} failed "
                  f"({exc}); copying through pandas")
            for suffix in ("", ".wal"):
                try:
                    os.unlink(path + suffix)
                except FileNotFoundError:
                    pass
    src = get_conn(universal_path)
    try:
        dst = get_conn(path)
        try:
            for table in UNIVERSAL_TABLES:
                df = read_table(src, table)
                if not df.empty:
                    write_table(dst, table, df)
        finally:
            dst.close()
    finally:
        src.close()
    return path
    src = get_conn(universal_path)
    try:
        dst = get_conn(path)
        try:
            for table in UNIVERSAL_TABLES:
                df = read_table(src, table)
                if not df.empty:
                    write_table(dst, table, df)
        finally:
            dst.close()
    finally:
        src.close()
    return path
