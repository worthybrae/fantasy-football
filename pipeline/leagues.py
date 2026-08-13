"""Where each league's data lives, and how a new one is provisioned.

Per-league database files: the heavy universal data (nflverse stats, market
ranks) stays in one shared file, and each league gets a small file with only
its own draft history, manager fits and live-draft state. A connection points
at one league's file, so the rest of the app -- every read_table(conn, ...)
site -- never has to know which league it is looking at.

The default league is the existing single database, so the single-user path
this project started as does not move.
"""
import re
from pathlib import Path

from pipeline.db import DEFAULT_PATH

DEFAULT_LEAGUE = "__default__"
LEAGUES_ROOT = "data/leagues"


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
    """The database file for a league. The default league is the shared
    single-user database, unchanged; every other league is its own file."""
    if league_id == DEFAULT_LEAGUE:
        return DEFAULT_PATH
    return str(Path(root) / f"{_safe_id(league_id)}.duckdb")


from pipeline.db import UNIVERSAL_TABLES, get_conn, read_table, write_table


def provision_league(league_id: str, universal_path: str,
                     root: str = LEAGUES_ROOT) -> str:
    """Ensure a league's database exists, seeded with the universal tables.

    Idempotent: an existing file is left exactly as it is, so reconnecting
    mid-draft never wipes the `drafted` rows already recorded. Only the
    universal tables are copied; the league-specific tables start absent,
    which is precisely the cold-start state `cold_start_fits` handles.
    """
    path = league_db_path(league_id, root=root)
    if league_id == DEFAULT_LEAGUE or Path(path).exists():
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
