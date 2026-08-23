"""Backfilling ESPN's board onto drafts that were recorded without it.

The corpus is the one thing in this project that cannot be rebuilt from a
source that still exists, and this is the first UPDATE ever run against it,
against a file three farm processes are writing to. So the tests here are
about the two things that could go wrong quietly: overwriting a value that was
already right, and reporting a fill that did not happen.
"""
import duckdb
import pandas as pd
import pytest

from pipeline import backfill_espn_board as bf
from pipeline import draft_log as dl


def _reference(rows=(("p1", 1.0, 310.5, 7), ("p2", 12.0, 240.0, 9))):
    return pd.DataFrame(rows, columns=["player_id", "espn_rank", "espn_proj",
                                       "bye"])


def _corpus_with_pool(path, rows, draft_id="mock:d1"):
    """A corpus holding one draft whose pool rows are `(player_id, rank)`."""
    conn = dl.corpus_conn(path)
    pool = pd.DataFrame({
        "player_id": [r[0] for r in rows],
        "position": ["RB"] * len(rows),
        "team": ["DET"] * len(rows),
        "adp_rank": [float(i + 1) for i in range(len(rows))],
        "proj_points": [200.0] * len(rows),
        "espn_rank": [r[1] for r in rows],
    })
    dl.record(conn, dl.DraftRecord(source=dl.SOURCE_MOCK, season=2026,
                                   draft_id=draft_id, pool=pool))
    conn.close()
    return path


def test_the_backfill_fills_every_row_it_can_match(tmp_path):
    path = _corpus_with_pool(str(tmp_path / "corpus.duckdb"),
                             [("p1", None), ("p2", None)])
    counts = bf.backfill(path, _reference(), out=lambda *_: None)

    assert counts == {"drafts": 1, "filled": 2, "unmatched": 0}
    conn = duckdb.connect(path)
    got = conn.execute("SELECT player_id, espn_rank, espn_proj, bye "
                       "FROM draft_log_pool ORDER BY player_id").fetchall()
    conn.close()
    assert got == [("p1", 1.0, 310.5, 7), ("p2", 12.0, 240.0, 9)]


def test_a_row_that_already_has_a_rank_is_never_rewritten(tmp_path):
    """The scope is `WHERE espn_rank IS NULL`, and this is why: a draft the
    farm recorded properly carries the board it was ACTUALLY played from, and
    today's board is only an acceptable substitute where there is nothing at
    all. Overwriting would trade a real value for an approximation."""
    path = _corpus_with_pool(str(tmp_path / "corpus.duckdb"),
                             [("p1", 99.0), ("p2", None)])
    counts = bf.backfill(path, _reference(), out=lambda *_: None)

    assert counts["filled"] == 1
    conn = duckdb.connect(path)
    got = dict(conn.execute("SELECT player_id, espn_rank "
                            "FROM draft_log_pool").fetchall())
    conn.close()
    assert got == {"p1": 99.0, "p2": 12.0}


def test_running_it_twice_changes_nothing_the_second_time(tmp_path):
    """Idempotent by construction: every row the reference can match comes
    out with a rank and no longer meets the scope. A backfill that has to be
    run exactly once is a backfill that will be run twice."""
    path = _corpus_with_pool(str(tmp_path / "corpus.duckdb"),
                             [("p1", None), ("p2", None)])
    first = bf.backfill(path, _reference(), out=lambda *_: None)
    second = bf.backfill(path, _reference(), out=lambda *_: None)

    assert first["filled"] == 2
    assert second == {"drafts": 0, "filled": 0, "unmatched": 0}


def test_a_player_espn_does_not_rank_is_counted_not_filled(tmp_path):
    """ESPN publishes 500 players; a draft pool is larger. Those rows stay
    NULL and are REPORTED, because a floor of unmatched rows is expected and
    a sudden rise in it is what a broken join key would look like."""
    path = _corpus_with_pool(str(tmp_path / "corpus.duckdb"),
                             [("p1", None), ("nobody", None)])
    counts = bf.backfill(path, _reference(), out=lambda *_: None)

    assert counts["filled"] == 1
    assert counts["unmatched"] == 1
    conn = duckdb.connect(path)
    assert conn.execute("SELECT espn_rank FROM draft_log_pool "
                        "WHERE player_id = 'nobody'").fetchone()[0] is None
    conn.close()


def test_each_draft_is_its_own_short_transaction(tmp_path):
    """One draft per connection, and the connection is closed between them:
    three farm sessions need the corpus write lock for the one INSERT that
    ends a forty-minute draft, and holding it across a whole backfill would
    make losing one of those possible."""
    path = str(tmp_path / "corpus.duckdb")
    conn = dl.corpus_conn(path)
    for i in (1, 2, 3):
        dl.record(conn, dl.DraftRecord(
            source=dl.SOURCE_MOCK, season=2026, draft_id=f"mock:d{i}",
            pool=pd.DataFrame({"player_id": ["p1"], "position": ["RB"],
                               "team": ["DET"], "adp_rank": [1.0],
                               "proj_points": [200.0]})))
    conn.close()

    opens = []
    real_connect = duckdb.connect

    def counting_connect(target, *args, **kwargs):
        opens.append(target)
        return real_connect(target, *args, **kwargs)

    duckdb.connect = counting_connect
    try:
        counts = bf.backfill(path, _reference(), out=lambda *_: None)
    finally:
        duckdb.connect = real_connect

    assert counts["filled"] == 3
    # One open to list the pending drafts, one per draft, one to count what
    # is left -- never one open held across all three.
    assert len(opens) == 5


def test_a_locked_corpus_is_waited_out_rather_than_given_up_on(tmp_path):
    """DuckDB refuses the CONNECTION when another process holds the write
    lock, so the retry belongs around the open. Losing a backfill because a
    farm happened to be recording at that second would be a job that has to
    be babysat."""
    path = _corpus_with_pool(str(tmp_path / "corpus.duckdb"), [("p1", None)])
    real_connect = duckdb.connect
    attempts = {"n": 0}

    def flaky_connect(target, *args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise duckdb.IOException(
                "Could not set lock on file: Conflicting lock is held")
        return real_connect(target, *args, **kwargs)

    duckdb.connect = flaky_connect
    bf.LOCK_RETRY_SECONDS, slow = 0.0, bf.LOCK_RETRY_SECONDS
    try:
        counts = bf.backfill(path, _reference(), out=lambda *_: None)
    finally:
        duckdb.connect = real_connect
        bf.LOCK_RETRY_SECONDS = slow

    assert attempts["n"] > 3          # it retried rather than failing
    assert counts["filled"] == 1


def test_anything_that_is_not_a_lock_failure_is_raised_at_once(tmp_path):
    """Waiting does not fix a corrupt file or a missing table, and a backfill
    that sits for three minutes on a typo helps nobody."""
    real_connect = duckdb.connect

    def broken_connect(*_a, **_k):
        raise duckdb.IOException("no such file or directory")

    duckdb.connect = broken_connect
    try:
        with pytest.raises(duckdb.IOException):
            bf.backfill(str(tmp_path / "nope.duckdb"), _reference(),
                        out=lambda *_: None)
    finally:
        duckdb.connect = real_connect


def test_the_reference_only_carries_players_espn_actually_ranks(monkeypatch):
    """What makes the UPDATE idempotent. A player with no ESPN rank
    contributes NO ROW rather than a row of nulls -- otherwise every run
    would rewrite him with the same nulls and he would be "filled" forever
    while the count said something had happened."""
    import scoring.board

    board = pd.DataFrame({"player_id": ["p1", "p9"],
                          "espn_ppr_rank": [1.0, None],
                          "bye": [7.0, 11.0],
                          "espn_id": [4001.0, None]})
    monkeypatch.setattr(scoring.board, "build_board", lambda *a, **k: board)

    conn = duckdb.connect()
    conn.execute("CREATE TABLE espn_adp (espn_id BIGINT, espn_name VARCHAR, "
                 "position VARCHAR, team VARCHAR, espn_adp DOUBLE, "
                 "espn_ppr_rank BIGINT, espn_proj DOUBLE)")
    conn.execute("INSERT INTO espn_adp VALUES "
                 "(4001, 'A', 'RB', 'DET', 1.2, 1, 310.5)")
    try:
        ref = bf.espn_reference(conn)
    finally:
        conn.close()

    assert list(ref["player_id"]) == ["p1"]
    assert ref["espn_proj"].iloc[0] == 310.5
    # VARCHAR in the corpus, whatever nflverse handed us on the board: the
    # cast happens here so the join is a string join on both sides.
    assert isinstance(ref["player_id"].iloc[0], str)
