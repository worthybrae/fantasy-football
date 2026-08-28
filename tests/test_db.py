"""`read_table`'s column projection.

The round-trip and missing-table cases live in tests/test_pipeline.py, which
had `pipeline.db` covered long before this argument existed. What is here is
the projection alone: the thing that decides how much memory a board build
costs, and the one part of `read_table` that can silently return a frame
missing something its caller needed.
"""
import pandas as pd
import pytest

from pipeline.db import get_conn, read_table, write_table


@pytest.fixture
def conn(tmp_path):
    c = get_conn(str(tmp_path / "t.duckdb"))
    write_table(c, "weekly", pd.DataFrame([
        {"player_id": "p1", "season": 2025, "week": 1, "headshot_url": "x",
         "receptions": 8},
        {"player_id": "p2", "season": 2025, "week": 1, "headshot_url": "y",
         "receptions": 3}]))
    return c


def test_no_columns_reads_the_whole_table(conn):
    """The default is unchanged, because every other caller depends on it."""
    got = read_table(conn, "weekly")
    assert list(got.columns) == ["player_id", "season", "week",
                                 "headshot_url", "receptions"]


def test_columns_are_the_only_ones_read(conn):
    got = read_table(conn, "weekly", columns=["player_id", "receptions"])
    assert list(got.columns) == ["player_id", "receptions"]
    assert got["receptions"].tolist() == [8, 3]


def test_a_column_the_table_does_not_have_is_skipped_not_raised(conn):
    """The case this behaviour exists for: `scoring.board.WEEKLY_COLUMNS`
    names every column a league's scoring rules CAN reach, and a test fixture
    seeds ten of them (tests/test_board.py) while an older database is simply
    missing whatever was added last. DuckDB answers a `SELECT` naming an
    absent column with a Binder Error, which would take the board down over a
    column nothing in that database was ever going to score."""
    got = read_table(conn, "weekly",
                     columns=["player_id", "fg_made_60_", "receptions"])
    assert list(got.columns) == ["player_id", "receptions"]


def test_a_projection_the_table_shares_nothing_with_is_empty(conn):
    """Same answer a missing table gives, because the callers' `.empty`
    branch is the right one for both."""
    assert read_table(conn, "weekly", columns=["nope", "still_nope"]).empty


def test_a_repeated_column_is_read_once(conn):
    """`WEEKLY_COLUMNS` is assembled from three overlapping lists (the base
    columns, full PPR's rules, ESPN's stat map), so duplicates are ordinary
    rather than a mistake -- and `SELECT player_id, player_id` would hand
    pandas two columns of the same name."""
    got = read_table(conn, "weekly", columns=["player_id", "player_id"])
    assert list(got.columns) == ["player_id"]


def test_a_missing_table_ignores_the_projection(tmp_path):
    c = get_conn(str(tmp_path / "empty.duckdb"))
    assert read_table(c, "nope", columns=["player_id"]).empty
