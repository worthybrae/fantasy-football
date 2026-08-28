"""Lasts %: the empirical chance a player is still on the board at a later pick.

Every number here is a ratio of DRAFT COUNTS out of the corpus, so the tests
build a corpus whose counts are known by construction and assert the ratio
back. The interesting cases are the ones where the corpus cannot answer --
a player who is always gone by the pick we are conditioning on has an empty
denominator, and a player the corpus barely saw has a denominator too small
to trust -- because those are the ones that fall through to the fitted ADP
curve, and a fallback that silently reported 0% or 100% would be invisible
on the board.
"""
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

from pipeline import draft_log as dl
from scoring import availability as av

DRAFTS = 40


def _pool_row(player_id, position, adp, espn_rank, bye=5):
    return {"player_id": player_id, "position": position, "team": "FA",
            "adp_rank": adp, "proj_points": 200.0, "espn_rank": espn_rank,
            "espn_proj": 200.0, "bye": bye}


def _b_pick(i):
    """Player B's pick in draft `i`: 10, 11, ... 30 and round again."""
    return 10 + (i % 21)


def _write_corpus(path, drafts=DRAFTS):
    """Four players with hand-countable histories.

    A  taken at pick 1, 2 or 3 in every draft -- never available at pick 5.
    B  taken at picks 10..30, evenly spread -- the empirical case.
    C  pooled in every draft and never taken at all.
    D  pooled in five drafts only -- too few to answer with.
    """
    conn = dl.corpus_conn(str(path))
    try:
        for i in range(drafts):
            pool = [_pool_row("A", "RB", 2.0, 2.0),
                    _pool_row("B", "WR", 20.0, 20.0),
                    _pool_row("C", "TE", 100.0, 100.0)]
            picks = [{"pick_no": (i % 3) + 1, "round": 1, "slot": 1,
                      "owner_key": "o", "is_anonymous": False,
                      "player_id": "A", "position": "RB"},
                     {"pick_no": _b_pick(i), "round": 2, "slot": 2,
                      "owner_key": "o", "is_anonymous": False,
                      "player_id": "B", "position": "WR"}]
            if i < 5:
                pool.append(_pool_row("D", "QB", 40.0, 40.0))
                picks.append({"pick_no": 40, "round": 5, "slot": 3,
                              "owner_key": "o", "is_anonymous": False,
                              "player_id": "D", "position": "QB"})
            dl.record(conn, dl.DraftRecord(
                source=dl.SOURCE_MOCK, league_id="1", season=2026,
                started_at=f"draft-{i}", teams=8, rounds=16,
                picks=pd.DataFrame(picks), pool=pd.DataFrame(pool)))
    finally:
        conn.close()
    return str(path)


@pytest.fixture
def corpus(tmp_path):
    return _write_corpus(tmp_path / "corpus.duckdb")


def _at(table, ids, k, n, adp=None, rank=None):
    return av.availability_at(table, ids, k, n, adp, rank)


def test_table_counts_pooled_and_taken_drafts(corpus):
    t = av.load_table(corpus)
    idx = {str(p): i for i, p in enumerate(t.player_ids)}
    assert set(idx) == {"A", "B", "C", "D"}
    assert t.pooled[idx["A"]] == DRAFTS
    assert t.pooled[idx["D"]] == 5
    # Monotone in the pick, and every one of A's drafts has him gone by 3.
    assert t.taken_by[idx["A"], 0] == 0
    assert t.taken_by[idx["A"], 3] == DRAFTS
    assert t.taken_by[idx["C"], av.MAX_PICK] == 0
    assert (np.diff(t.taken_by[idx["B"]]) >= 0).all()


def test_empirical_ratio_is_the_share_of_drafts_he_lasted(corpus):
    """B at (k=8, n=16): of the drafts where he was still there at 8 -- all
    of them, he never goes before 10 -- the share where he went after 16."""
    t = av.load_table(corpus)
    expected = sum(1 for i in range(DRAFTS) if _b_pick(i) > 16) / DRAFTS
    got = _at(t, ["B"], 8, 16, np.array([np.nan]), np.array([np.nan]))
    assert got[0] == pytest.approx(expected)


def test_a_player_never_taken_reads_one(corpus):
    t = av.load_table(corpus)
    got = _at(t, ["C"], 5, 60, np.array([np.nan]), np.array([np.nan]))
    assert got[0] == pytest.approx(1.0)


def test_an_empty_denominator_falls_back_to_the_adp_curve(corpus):
    """A is gone by pick 3 in all forty drafts, so 'still there at 5' has no
    drafts in it at all. 0/0 is not an answer; the curve is."""
    t = av.load_table(corpus)
    got = _at(t, ["A"], 5, 11, np.array([2.0]), np.array([2.0]))
    assert got[0] == pytest.approx(av.fallback_probability(2.0, 5, 11, t))
    assert got[0] < 0.05, "a player who always goes in the first three picks"


def test_a_thin_denominator_falls_back_too(corpus):
    """D was pooled in five drafts -- under MIN_DRAFTS, so his own history
    is not the answer even though he has one."""
    t = av.load_table(corpus)
    assert 5 < av.MIN_DRAFTS
    got = _at(t, ["D"], 1, 20, np.array([40.0]), np.array([40.0]))
    assert got[0] == pytest.approx(av.fallback_probability(40.0, 1, 20, t))
    # ADP 40 with the default spread: still very likely there at pick 20.
    assert got[0] > 0.8


def test_a_fitted_bucket_beats_the_bare_adp(corpus):
    """A's bucket (ESPN ranks 0-7) was measured: forty drafts that all took
    him inside three picks. The curve has to know that, not just his rank."""
    t = av.load_table(corpus)
    bucket = int(2.0 // av.ADP_BUCKET)
    mu, sigma = t.adp_curve[bucket]
    assert mu == pytest.approx(np.mean([(i % 3) + 1 for i in range(DRAFTS)]),
                               abs=0.1)
    assert sigma >= av.MIN_SIGMA


def test_a_bucket_nobody_is_ever_drafted_from_is_not_fitted(corpus):
    """C's rank bucket has forty pool rows and no picks. A mean over the
    picks that happened would be a mean over nothing; the bucket is dropped
    and the player's own rank is the centre instead."""
    t = av.load_table(corpus)
    assert int(100.0 // av.ADP_BUCKET) not in t.adp_curve


def test_an_unknown_player_with_no_rank_at_all_reads_one(corpus):
    t = av.load_table(corpus)
    got = _at(t, ["nobody"], 5, 40, np.array([np.nan]), np.array([np.nan]))
    assert got[0] == pytest.approx(1.0)


def test_an_unknown_player_uses_consensus_when_adp_is_blank(corpus):
    t = av.load_table(corpus)
    got = _at(t, ["nobody"], 5, 40, np.array([np.nan]), np.array([90.0]))
    assert got[0] == pytest.approx(av.fallback_probability(90.0, 5, 40, t))
    assert got[0] > 0.9


def test_every_id_is_answered_in_one_vectorised_call(corpus):
    t = av.load_table(corpus)
    ids = ["A", "B", "C", "D", "nobody"]
    got = _at(t, ids, 8, 16,
              np.array([2.0, 20.0, 100.0, 40.0, np.nan]),
              np.array([2.0, 20.0, 100.0, 40.0, np.nan]))
    assert got.shape == (5,)
    assert ((got >= 0.0) & (got <= 1.0)).all()
    assert got[2] == pytest.approx(1.0)
    assert got[4] == pytest.approx(1.0)


def test_picks_past_the_end_of_the_table_do_not_run_off_it(corpus):
    t = av.load_table(corpus)
    got = _at(t, ["B"], 0, av.MAX_PICK + 500,
              np.array([np.nan]), np.array([np.nan]))
    assert got[0] == pytest.approx(0.0)


def test_an_empty_table_answers_one_for_everyone():
    t = av.AvailabilityTable.empty()
    got = _at(t, ["A", "B"], 3, 20, np.array([np.nan] * 2),
              np.array([np.nan] * 2))
    assert list(got) == [1.0, 1.0]


def test_a_missing_corpus_is_an_empty_table_not_an_error(tmp_path):
    t = av.load_table(str(tmp_path / "nothing-here.duckdb"))
    assert t.player_ids.size == 0
    assert t.corpus_mtime == 0.0


def test_cached_table_reloads_when_the_file_changes(corpus):
    first = av.cached_table(corpus)
    assert av.cached_table(corpus) is first, "unchanged file, same table"

    conn = dl.corpus_conn(corpus)
    try:
        dl.record(conn, dl.DraftRecord(
            source=dl.SOURCE_MOCK, league_id="1", season=2026,
            started_at="late-one", teams=8, rounds=16,
            picks=pd.DataFrame([{"pick_no": 1, "round": 1, "slot": 1,
                                 "owner_key": "o", "is_anonymous": False,
                                 "player_id": "E", "position": "RB"}]),
            pool=pd.DataFrame([_pool_row("E", "RB", 5.0, 5.0)])))
    finally:
        conn.close()
    os.utime(corpus, (first.corpus_mtime + 10, first.corpus_mtime + 10))

    second = av.cached_table(corpus)
    assert second is not first
    assert "E" in {str(p) for p in second.player_ids}


def test_a_locked_corpus_keeps_the_table_it_already_had(corpus, monkeypatch):
    """The farm holds the write lock for as long as it is recording a mock.
    A poll that lands in that window must serve the last good table, not
    an error and not an empty board."""
    first = av.cached_table(corpus)
    assert first.player_ids.size == 4

    def busy(*a, **k):
        raise duckdb.IOException("Could not set lock on file")
    monkeypatch.setattr(duckdb, "connect", busy)
    os.utime(corpus, (first.corpus_mtime + 20, first.corpus_mtime + 20))

    assert av.cached_table(corpus) is first
    monkeypatch.undo()
    # And the next poll after the lock clears does pick the change up.
    assert av.cached_table(corpus) is not first


def test_a_locked_corpus_with_nothing_cached_is_an_empty_table(
        tmp_path, monkeypatch):
    path = _write_corpus(tmp_path / "locked.duckdb")

    def busy(*a, **k):
        raise duckdb.IOException("Could not set lock on file")
    monkeypatch.setattr(duckdb, "connect", busy)
    t = av.cached_table(path)
    assert t.player_ids.size == 0
