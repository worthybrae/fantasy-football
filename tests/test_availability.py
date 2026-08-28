"""Lasts %: the empirical chance a player is still on the board at a later pick.

Every number here is a ratio of DRAFT COUNTS out of the corpus, so the tests
build a corpus whose counts are known by construction and assert the ratio
back. The interesting cases are the ones where the corpus cannot answer --
a player who is always gone by the pick we are conditioning on has an empty
denominator, a player the corpus barely saw has a denominator too small to
trust, and NOBODY has a recorded pick past the end of an 8-team draft -- because
those are the ones that fall through to the fitted ADP curve, and a fallback
that silently reported 0% or 100% would be invisible on the board.
"""
import os

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from pipeline import draft_log as dl
from scoring import availability as av

# Two full cycles of player B's pick, so his distribution is exactly uniform
# over picks 10-30 and the share that lasts past a pick can be worked out by
# hand rather than read off the generator.
DRAFTS = 42

# The eight players who fill ESPN rank bucket 0. A bucket needs
# MIN_BUCKET_PLAYERS distinct players before it is fitted at all, so one
# player cannot be his own curve.
EARLY = [f"a{i}" for i in range(8)]
# Eight more in bucket 12 that nobody ever drafts.
NEVER = [f"n{i}" for i in range(8)]


def _pool_row(player_id, position, adp, espn_rank, bye=5):
    return {"player_id": player_id, "position": position, "team": "FA",
            "adp_rank": adp, "proj_points": 200.0, "espn_rank": espn_rank,
            "espn_proj": 200.0, "bye": bye}


def _pick_row(pick_no, player_id, position="RB"):
    return {"pick_no": pick_no, "round": 1 + (pick_no - 1) // 8, "slot": 1,
            "owner_key": "o", "is_anonymous": False,
            "player_id": player_id, "position": position}


def _b_pick(i):
    """Player B's pick in draft `i`: 10, 11, ... 30 and round again."""
    return 10 + (i % 21)


def _write_corpus(path, drafts=DRAFTS):
    """A corpus with hand-countable histories.

    a0..a7  ESPN ranks 0-7, taken at picks 1-8 in every draft. Eight distinct
            players, all always drafted: bucket 0 is fitted from them.
    B       taken at picks 10-30, evenly spread -- the empirical case.
    n0..n7  ESPN ranks 96-103, pooled in every draft and never taken at all.
    D       pooled in five drafts only -- too few to answer with.
    """
    conn = dl.corpus_conn(str(path))
    try:
        for i in range(drafts):
            pool = [_pool_row(pid, "RB", float(n), float(n))
                    for n, pid in enumerate(EARLY)]
            pool.append(_pool_row("B", "WR", 20.0, 20.0))
            pool += [_pool_row(pid, "TE", 96.0 + n, 96.0 + n)
                     for n, pid in enumerate(NEVER)]
            picks = [_pick_row(n + 1, pid) for n, pid in enumerate(EARLY)]
            picks.append(_pick_row(_b_pick(i), "B", "WR"))
            if i < 5:
                pool.append(_pool_row("D", "QB", 40.0, 40.0))
                picks.append(_pick_row(40, "D", "QB"))
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


def _at(table, ids, k, n, adp=None, rank=None, positions=None):
    return av.availability_at(table, ids, k, n, adp, rank, positions)


def _hand_table(pooled, taken, curve=None, observed=av.MAX_PICK):
    """One player, `pooled` drafts, `taken` = {pick: how many took him}."""
    counts = np.zeros((1, av.MAX_PICK + 1), dtype=np.int64)
    for pick, n in (taken or {}).items():
        counts[0, pick] += n
    return av.AvailabilityTable(
        player_ids=np.array(["x"], dtype=object),
        pooled=np.array([pooled], dtype=np.int64),
        taken_by=np.cumsum(counts, axis=1),
        adp_curve=curve or {}, corpus_mtime=1.0, max_pick_observed=observed)


def test_table_counts_pooled_and_taken_drafts(corpus):
    t = av.load_table(corpus)
    idx = t.index
    assert set(idx) == set(EARLY) | set(NEVER) | {"B", "D"}
    assert t.pooled[idx["a0"]] == DRAFTS
    assert t.pooled[idx["D"]] == 5
    assert t.taken_by[idx["a0"], 0] == 0
    assert t.taken_by[idx["a0"], 1] == DRAFTS
    assert t.taken_by[idx["n0"], av.MAX_PICK] == 0
    assert (np.diff(t.taken_by[idx["B"]]) >= 0).all()
    # The deepest pick anyone was ever taken at: D, at 40.
    assert t.max_pick_observed == 40


def test_the_ratio_is_the_share_that_survived_the_picks_before_n(corpus):
    """"Still there when I make pick 16" means he survived picks 1 to 15.

    B goes uniformly at picks 10-30 and has never gone before 10, so at
    (k=8, n=16) the denominator is every draft and the numerator is the
    fifteen of his twenty-one landing spots that are 16 or later."""
    t = av.load_table(corpus)
    lasted = sum(1 for i in range(DRAFTS) if _b_pick(i) >= 16)
    got = _at(t, ["B"], 8, 16)
    assert got[0] == pytest.approx(lasted / DRAFTS)
    assert got[0] == pytest.approx(15 / 21, abs=1e-4)


def test_a_player_never_taken_reads_one(corpus):
    t = av.load_table(corpus)
    assert _at(t, ["n0"], 5, 40)[0] == pytest.approx(1.0)


def test_the_pick_on_the_clock_is_certain(corpus):
    """He is on the board right now, so the turn one pick from now cannot
    take him away -- `build_plan` passes the current pick as its first turn
    and must agree with `target_now` about it."""
    t = av.load_table(corpus)
    assert _at(t, ["a0"], 5, 6, np.array([2.0]))[0] == pytest.approx(1.0)
    assert _at(t, ["B"], 12, 13)[0] == pytest.approx(1.0)
    # And a turn already behind us is not a question either.
    assert _at(t, ["B"], 20, 15)[0] == pytest.approx(1.0)


def test_an_empty_denominator_falls_back_to_the_adp_curve(corpus):
    """a0 is gone by pick 1 in every draft, so "still there at 5" has no
    drafts in it at all. 0/0 is not an answer; the curve is."""
    t = av.load_table(corpus)
    got = _at(t, ["a0"], 5, 11, np.array([0.0]))
    assert got[0] == pytest.approx(av.fallback_probability(0.0, 5, 11, t))
    assert got[0] < 0.05, "a player who always goes in the first eight picks"


def test_a_thin_denominator_falls_back_too(corpus):
    """D was pooled in five drafts -- under MIN_DRAFTS, so his own history
    is not the answer even though he has one."""
    t = av.load_table(corpus)
    assert 5 < av.MIN_DRAFTS
    got = _at(t, ["D"], 1, 20, np.array([40.0]))
    assert got[0] == pytest.approx(av.fallback_probability(40.0, 1, 20, t))
    assert got[0] > 0.8


def test_the_min_drafts_boundary_is_where_the_spec_puts_it():
    """The bar for trusting a RATIO. Both of these players go at pick 10 in
    four fifths of their drafts and last in the rest, so the counts have a
    real number to offer -- and one of them has 24 drafts behind it, which
    is not enough to print."""
    curve = {av.SKILL: {}}
    twenty_five = _hand_table(25, {10: 20}, curve)
    twenty_four = _hand_table(24, {10: 19}, curve)
    adp = np.array([200.0])
    assert _at(twenty_five, ["x"], 0, 20, adp)[0] == pytest.approx(5 / 25)
    assert _at(twenty_four, ["x"], 0, 20, adp)[0] > 0.9


def test_the_parametric_ratio_is_the_normal_tail_conditioned_on_now():
    t = _hand_table(0, {}, {av.SKILL: {2: (20.0, 10.0)}})
    got = _at(t, ["nobody"], 10, 30, np.array([20.0]))
    expected = ndtr((20.0 - 29) / 10.0) / ndtr((20.0 - 10) / 10.0)
    assert got[0] == pytest.approx(expected)


def test_a_fitted_bucket_beats_the_bare_adp(corpus):
    """Bucket 0 was measured: eight players, every one of them gone inside
    eight picks. The curve has to know that, not just their rank."""
    t = av.load_table(corpus)
    mu, sigma = t.adp_curve[av.SKILL][0]
    assert mu == pytest.approx(np.mean(range(1, 9)), abs=0.1)
    assert sigma >= av.MIN_SIGMA


def test_a_bucket_nobody_is_ever_drafted_from_is_not_fitted(corpus):
    """n0..n7 are eight distinct players in one bucket, so the bucket is big
    enough -- and none of them has ever been taken, so a mean over the picks
    that happened would be a mean over nothing."""
    t = av.load_table(corpus)
    assert int(96 // av.ADP_BUCKET) not in t.adp_curve[av.SKILL]


def test_a_bucket_with_too_few_distinct_players_is_not_fitted(corpus):
    """B has forty-two pick rows in his bucket, all of them his own. A curve
    fitted from one player is that player, not the bucket."""
    t = av.load_table(corpus)
    assert int(20 // av.ADP_BUCKET) not in t.adp_curve[av.SKILL]


def test_an_unknown_player_with_no_rank_at_all_reads_one(corpus):
    t = av.load_table(corpus)
    assert _at(t, ["nobody"], 5, 40)[0] == pytest.approx(1.0)


def test_an_unknown_player_uses_consensus_when_adp_is_blank(corpus):
    t = av.load_table(corpus)
    got = _at(t, ["nobody"], 5, 40, np.array([np.nan]), np.array([90.0]))
    assert got[0] == pytest.approx(av.fallback_probability(90.0, 5, 40, t))
    assert got[0] > 0.9


def test_every_id_is_answered_in_one_vectorised_call(corpus):
    t = av.load_table(corpus)
    ids = ["a0", "B", "n0", "D", "nobody"]
    got = _at(t, ids, 8, 16,
              np.array([0.0, 20.0, 100.0, 40.0, np.nan]),
              np.array([0.0, 20.0, 100.0, 40.0, np.nan]))
    assert got.shape == (5,)
    assert ((got >= 0.0) & (got <= 1.0)).all()
    assert got[2] == pytest.approx(1.0)
    assert got[4] == pytest.approx(1.0)


# --- past the end of what the corpus has ever seen ---------------------------

def test_past_the_deepest_recorded_pick_only_the_censored_go_to_the_curve(
        corpus):
    """An 8-team corpus stops at pick 128 (here, 40), and what that means is
    different for two kinds of player.

    A player every one of his pooled drafts took inside that depth has
    COMPLETE evidence: "he is gone by 41" is a fact about him, not a gap in
    the record. A player who was still there at the end is right-censored --
    the corpus simply stopped watching -- and only he needs the curve.

    Throwing both to the curve made the number jump UP at the seam: kickers
    at rank 240 went 0.00 at pick 128 and 1.00 at pick 129, and six real
    defenses read 1% at pick 126 and 50% at 139."""
    t = av.load_table(corpus)
    assert t.max_pick_observed == 40

    # B goes at 10-30 in every draft he is pooled in. Nothing about him is
    # censored, so pick 60 is answered by the counts: he is not there.
    assert _at(t, ["B"], 10, 60, np.array([20.0]))[0] == pytest.approx(0.0)
    # n0 was still on the board when every one of those drafts ended.
    censored = _at(t, ["n0"], 10, 60, np.array([100.0]))[0]
    assert censored > 0.5, "the corpus never saw him taken"
    assert censored == pytest.approx(
        ndtr((100.0 - 59) / av.FALLBACK_SIGMA)
        / ndtr((100.0 - 40) / av.FALLBACK_SIGMA))


def test_the_curve_takes_over_from_where_the_counts_left_him(corpus):
    """The censored player's two halves are multiplied, not swapped: the
    counts say how often he reached the end of the corpus and the curve
    carries him from there. Anything else steps at the seam."""
    t = av.load_table(corpus)
    seam = _at(t, ["n0"], 5, t.max_pick_observed, np.array([100.0]))[0]
    across = _at(t, ["n0"], 5, t.max_pick_observed + 1, np.array([100.0]))[0]
    assert across <= seam + 1e-12


def test_nothing_ever_gets_more_likely_to_last_as_the_pick_gets_later(corpus):
    """The one property the whole number has to have. It is checked across
    the seam for every player in the corpus, from conditioning picks either
    side of the corpus's own depth, because the seam is exactly where two
    different estimators meet."""
    t = av.load_table(corpus)
    ids = list(t.index)
    ranks = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 20.0, 40.0,
                      96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0])
    order = np.array([{pid: i for i, pid in enumerate(
        EARLY + ["B", "D"] + NEVER)}[pid] for pid in ids])
    adp = ranks[order]
    for k in (0, 5, 39, 40, 45, 90):
        walk = np.array([_at(t, ids, k, n, adp) for n in range(k + 1, 201)])
        steps = np.diff(walk, axis=0)
        assert (steps <= 1e-12).all(), (k, ids[int(
            np.argmax(steps.max(axis=0)))])


def test_once_the_clock_is_past_the_corpus_the_curve_is_all_there_is(corpus):
    """A censored player asked about pick 150 when 140 picks have already
    been made. The counts have nothing to carry him with -- they stopped at
    40 -- so conditioning on 40 rather than on 140 is asking a different
    question, and it answers it two orders of magnitude too small."""
    t = av.load_table(corpus)
    depth = t.max_pick_observed
    got = _at(t, ["n0"], depth + 5, 60, np.array([100.0]))
    assert got[0] == pytest.approx(
        av.fallback_probability(100.0, depth + 5, 60, t))
    assert got[0] > _at(t, ["n0"], depth, 60, np.array([100.0]))[0]


def test_past_the_deepest_pick_a_player_with_no_rank_still_reads_one(corpus):
    t = av.load_table(corpus)
    assert _at(t, ["n0"], 5, 150)[0] == pytest.approx(1.0)


def test_picks_past_the_end_of_the_table_do_not_run_off_it(corpus):
    t = av.load_table(corpus)
    got = _at(t, ["B"], 0, av.MAX_PICK + 500, np.array([20.0]))
    assert 0.0 <= got[0] <= 1.0


# --- the curve is fitted per position group ---------------------------------

KICKERS = [f"k{i}" for i in range(8)]
LATE_WRS = [f"w{i}" for i in range(8)]
SLOW = [f"s{i}" for i in range(8)]
QUICK = [f"q{i}" for i in range(8)]


def _write_mixed_corpus(path, drafts=40):
    """A corpus shaped like the real one, where kickers and late wide
    receivers share an ESPN rank bucket and nothing else about them agrees.

    k0..k7   ESPN ranks 240-247, taken at picks 110-117 in every draft.
    w0..w7   ESPN ranks 240-247, taken in one draft out of forty.
    s0..s7   ranks 8-15, taken at picks 30-37.
    q0..q7   ranks 16-23, taken at picks 20-27 -- EARLIER than the bucket
             above them, which is what the running maximum has to fix.
    """
    conn = dl.corpus_conn(str(path))
    try:
        for i in range(drafts):
            pool, picks = [], []
            for n, pid in enumerate(KICKERS):
                pool.append(_pool_row(pid, "K", 240.0 + n, 240.0 + n))
                picks.append(_pick_row(110 + n, pid, "K"))
            for n, pid in enumerate(LATE_WRS):
                pool.append(_pool_row(pid, "WR", 240.0 + n, 240.0 + n))
                if i == 0:
                    picks.append(_pick_row(90 + n, pid, "WR"))
            for n, pid in enumerate(SLOW):
                pool.append(_pool_row(pid, "RB", 8.0 + n, 8.0 + n))
                picks.append(_pick_row(30 + n, pid, "RB"))
            for n, pid in enumerate(QUICK):
                pool.append(_pool_row(pid, "WR", 16.0 + n, 16.0 + n))
                picks.append(_pick_row(20 + n, pid, "WR"))
            dl.record(conn, dl.DraftRecord(
                source=dl.SOURCE_MOCK, league_id="1", season=2026,
                started_at=f"mixed-{i}", teams=8, rounds=16,
                picks=pd.DataFrame(picks), pool=pd.DataFrame(pool)))
    finally:
        conn.close()
    return str(path)


def test_kickers_do_not_bend_the_curve_the_skill_players_are_read_off(tmp_path):
    """The real corpus drafts a kicker in every mock at an ESPN rank around
    244, and fitting one curve over all positions let those rows tell the
    board that a wide receiver at ADP 244 goes at pick 113 -- he read 38%
    where his neighbours at 238 and 260 read 100%."""
    t = av.load_table(_write_mixed_corpus(tmp_path / "mixed.duckdb"))
    deep = int(244 // av.ADP_BUCKET)
    assert deep in t.adp_curve["K"]
    assert t.adp_curve["K"][deep][0] == pytest.approx(113.5, abs=1.0)
    assert deep not in t.adp_curve[av.SKILL], "the kickers stayed in their own"

    receiver = _at(t, ["stranger"], 0, 130, np.array([244.0]),
                   positions=["WR"])
    kicker = _at(t, ["stranger"], 0, 130, np.array([244.0]),
                 positions=["K"])
    assert receiver[0] > 0.95, "nobody has ever drafted a WR ranked 244"
    assert kicker[0] < 0.05, "every mock takes its kicker around pick 113"


def test_the_curve_never_goes_backwards_as_adp_gets_worse(tmp_path):
    """Bucket 2's players go earlier than bucket 1's in this corpus, which
    is noise: a worse ADP cannot mean an earlier pick, and a curve that says
    so prints a deeper player as less likely to last than a better one."""
    t = av.load_table(_write_mixed_corpus(tmp_path / "mixed.duckdb"))
    skill = t.adp_curve[av.SKILL]
    buckets = sorted(skill)
    mus = [skill[b][0] for b in buckets]
    sigmas = [skill[b][1] for b in buckets]
    assert mus == sorted(mus), mus
    assert sigmas == sorted(sigmas), sigmas
    assert skill[2][0] == pytest.approx(skill[1][0]), "held at the running max"


# --- the table itself -------------------------------------------------------

def test_an_empty_table_answers_one_for_everyone():
    t = av.AvailabilityTable.empty()
    got = _at(t, ["A", "B"], 3, 20, np.array([np.nan] * 2))
    assert list(got) == [1.0, 1.0]


def test_a_missing_corpus_is_an_empty_table_not_an_error(tmp_path):
    t = av.load_table(str(tmp_path / "nothing-here.duckdb"))
    assert t.player_ids.size == 0
    assert t.corpus_mtime == 0.0
    assert t.max_pick_observed == 0


def test_cached_table_reloads_when_the_file_changes(corpus):
    first = av.cached_table(corpus)
    assert av.cached_table(corpus) is first, "unchanged file, same table"

    conn = dl.corpus_conn(corpus)
    try:
        dl.record(conn, dl.DraftRecord(
            source=dl.SOURCE_MOCK, league_id="1", season=2026,
            started_at="late-one", teams=8, rounds=16,
            picks=pd.DataFrame([_pick_row(1, "E")]),
            pool=pd.DataFrame([_pool_row("E", "RB", 5.0, 5.0)])))
    finally:
        conn.close()
    os.utime(corpus, (first.corpus_mtime + 10, first.corpus_mtime + 10))

    second = av.cached_table(corpus)
    assert second is not first
    assert "E" in second.index


def test_a_locked_corpus_keeps_the_table_it_already_had(corpus, monkeypatch):
    """The farm holds the write lock for as long as it is recording a mock.
    A poll that lands in that window must serve the last good table, not
    an error and not an empty board."""
    first = av.cached_table(corpus)
    assert first.player_ids.size == 18

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
    # Nothing good was cached, so the empty table must not be remembered as
    # the answer for this file: the next poll has to try again.
    monkeypatch.undo()
    assert av.cached_table(path).player_ids.size == 18


def test_a_kicker_the_corpus_always_drafts_does_not_come_back_to_life(
        tmp_path):
    """Every mock takes its kicker around pick 113 and the corpus is 128
    picks deep. At a twelve-team round-11 turn -- pick 126, past the
    deepest pick ever recorded -- he is gone, and the fitted curve has
    nothing to say about him: K has no fitted bucket, so the curve would
    have read his ESPN rank of 240 as "goes at pick 240" and printed 100%.
    """
    t = av.load_table(_write_mixed_corpus(tmp_path / "mixed.duckdb"))
    assert t.max_pick_observed == 117
    for pick in (126, 139):
        got = _at(t, ["k0"], 100, pick, np.array([240.0]), positions=["K"])
        assert got[0] < 0.01, pick
    # And the receiver at the same rank, who the corpus has almost never
    # seen taken, is still the curve's problem and still on the board.
    receiver = _at(t, ["w0"], 100, 139, np.array([240.0]), positions=["WR"])
    assert receiver[0] > 0.9


def test_a_thin_history_that_is_complete_still_proves_he_is_gone(tmp_path):
    """The reviewer's kicker: pooled in ten drafts, taken by pick 20 in every
    one of them. Ten is under MIN_DRAFTS, so the ratio he is worth is not
    trusted -- but "every draft that ever had him took him by 20" is not a
    ratio, it is a fact, and the curve reading his ESPN rank of 240 as "goes
    at pick 240" printed him as a certainty at pick 30."""
    kicker = _hand_table(10, {20: 10}, {av.SKILL: {}}, observed=20)
    adp = np.array([240.0])
    assert _at(kicker, ["x"], 5, 30, adp)[0] == pytest.approx(0.0)
    assert _at(kicker, ["x"], 5, 21, adp)[0] == pytest.approx(0.0)
    # Before his last pick the thin ratio is still not the answer: MIN_DRAFTS
    # governs the in-range number exactly as it did.
    assert _at(kicker, ["x"], 5, 15, adp)[0] == pytest.approx(
        av.fallback_probability(240.0, 5, 15, kicker))


def test_a_handful_of_drafts_is_not_enough_to_call_him_gone(tmp_path):
    """Four drafts that all took him early is a coincidence, not a history."""
    thin = _hand_table(4, {20: 4}, {av.SKILL: {}}, observed=20)
    adp = np.array([240.0])
    assert _at(thin, ["x"], 5, 30, adp)[0] == pytest.approx(
        av.fallback_probability(240.0, 5, 30, thin))
    assert _at(thin, ["x"], 5, 30, adp)[0] > 0.5
