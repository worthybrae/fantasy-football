"""The snapshot the farm reads when the board database is locked.

`mock_farm.open_board_db` falls back to a file copy when another process
holds the write lock -- which is the normal case, because the API keeps a
read-write connection open for its whole life. That copy has to contain the
data, and for one very specific reason it did not.

MEASURED IN PRODUCTION. On the deployed instance the copy came out with no
tables in it at all:

    Catalog Error: Table with name players does not exist!

DuckDB does not write every committed change straight into the database
file. Recent writes live in a sidecar write-ahead log, `<name>.wal`, until a
checkpoint folds them in -- and a connection that stays open may not
checkpoint for a very long time. The deployed volume showed a 24 MB
`nfl.duckdb` beside a 5.7 MB `nfl.duckdb.wal`, and every table the refresh
had just written was in the second one.

Copying only the first file therefore produced an EMPTY database that opened
perfectly happily. The farm built a board from it -- `board built: 0 pool
players, 0 selectable` -- joined real ESPN rooms it could not pick in, and
recorded a draft with zero picks into the corpus.

This never showed up locally because a developer's `nfl.duckdb` was written
and checkpointed months ago; the data is in the main file and the WAL is
small or absent. It appears the moment a database is fresh, which is exactly
what a new deployment's volume is.
"""
import duckdb

from pipeline.mock_farm import open_board_db


def test_a_snapshot_carries_writes_that_are_still_in_the_wal(tmp_path):
    """The production failure, reproduced: a writer holds the file open and
    has not checkpointed, so the tables exist only in the sidecar log."""
    live = tmp_path / "nfl.duckdb"
    writer = duckdb.connect(str(live))
    writer.execute("CREATE TABLE players AS SELECT 1 AS id, 'Gibbs' AS name")

    # The write-ahead log is what this test is about. If DuckDB has already
    # checkpointed there is nothing here to lose and nothing to prove.
    assert (tmp_path / "nfl.duckdb.wal").exists(), "no WAL: nothing to test"

    lines = []
    conn = open_board_db(str(live), out=lines.append)

    assert conn.execute("SELECT count(*) FROM players").fetchone()[0] == 1
    assert any("snapshot" in line for line in lines), lines


def test_an_unlocked_database_is_read_directly(tmp_path):
    """No copy when nothing holds the lock -- the snapshot is a fallback, not
    the normal path, and copying a 65 MB file per farm process for no reason
    would be a real cost."""
    live = tmp_path / "nfl.duckdb"
    duckdb.connect(str(live)).close()

    lines = []
    open_board_db(str(live), out=lines.append)

    assert not any("snapshot" in line for line in lines), lines
