"""pipeline.mock_backfill: folding the ESPN mock drafts already sitting in
data/leagues/*.duckdb into the cross-league draft corpus.

Each synthetic league file below is seeded with just enough of the universal
reference tables (see pipeline.db's LEAGUE_TABLES/UNIVERSAL_TABLES split,
and tests/test_board.py's own `_seed` fixture, which this mirrors) for
`scoring.board.build_board` and `scoring.draft_sim.build_pool` to run --
which is the same pipeline the real backfill runs per draft, so a fixture
that exercised anything less would not be testing the real join.
"""
import os
import time

import duckdb
import pandas as pd
import pytest

from pipeline import draft_log as dl
from pipeline import mock_backfill
from pipeline.db import get_conn, write_table
from scoring.draft_sim import snake_slots

N_PICKS_COMPLETE = mock_backfill.MOCK_TEAMS * mock_backfill.MOCK_ROUNDS  # 128


def _seed_mock_league(path, n_picks: int) -> None:
    """A per-league database shaped like the real `data/leagues/*.duckdb`
    files: universal reference tables (weekly/adp/etc, populated with
    `n_picks` distinct WR players so every pick joins to a real pool row)
    plus a `drafted` table of `n_picks` picks -- complete when `n_picks ==
    128`, an abandoned connect otherwise. No `league` table, no
    `draft_order` rows: exactly what the real files carry.
    """
    conn = get_conn(str(path))
    weekly = pd.DataFrame([
        {"player_id": f"p{i}", "player_display_name": f"Player {i}",
         "position": "WR", "recent_team": "DET", "opponent_team": "GB",
         "season": 2025, "week": 1, "receptions": 5, "receiving_yards": 60,
         "targets": 7, "carries": 0}
        for i in range(n_picks)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 45.0, "spread_line": 1.0}]))
    write_table(conn, "adp", pd.DataFrame(
        columns=["adp_name", "position", "team", "adp"]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "espn_adp", pd.DataFrame(columns=[
        "espn_id", "espn_name", "position", "team", "espn_adp",
        "espn_ppr_rank", "espn_proj"]))
    write_table(conn, "fp_ecr", pd.DataFrame(columns=[
        "fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(columns=[
        "gsis_id", "espn_id", "sleeper_name", "position", "team"]))

    drafted = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n_picks)],
        "pick_no": range(1, n_picks + 1),
    })
    write_table(conn, "drafted", drafted)
    conn.close()


def test_a_complete_mock_draft_lands_with_all_128_picks(tmp_path):
    """The one shape a file is accepted at: 8 teams x 16 rounds. Every pick
    must land with the correct snake slot, an anonymous owner key, and NULL
    autodrafted -- nothing in a `drafted` table says whether ESPN's engine
    made the pick."""
    path = tmp_path / "12345.duckdb"
    _seed_mock_league(path, N_PICKS_COMPLETE)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))

    counts = mock_backfill.backfill([str(path)], corpus)

    assert counts["files_scanned"] == 1
    assert counts["drafts_recorded"] == 1
    assert counts["picks_recorded"] == 128
    assert counts["picks_dropped"] == 0

    got = dl.picks(corpus, source=dl.SOURCE_MOCK).sort_values("pick_no")
    assert len(got) == 128
    assert got["is_anonymous"].all()
    assert got["owner_key"].str.startswith(dl.ANONYMOUS_PREFIX).all()
    assert got["autodrafted"].isna().all()
    assert list(got["slot"]) == snake_slots(8, 16)
    assert list(got["round"]) == [(n - 1) // 8 + 1 for n in range(1, 129)]

    pool = corpus.execute(
        "SELECT * FROM draft_log_pool WHERE draft_id = ?",
        [got["draft_id"].iloc[0]]).df()
    assert len(pool) == 128


def test_an_incomplete_mock_draft_is_skipped_and_counted(tmp_path):
    """An abandoned connect (fewer than 128 picks) must not be guessed at --
    skip it, count it, record nothing."""
    path = tmp_path / "999.duckdb"
    _seed_mock_league(path, 50)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))

    counts = mock_backfill.backfill([str(path)], corpus)

    assert counts["incomplete"] == 1
    assert counts["drafts_recorded"] == 0
    assert len(dl.picks(corpus)) == 0


def test_a_duplicate_pick_no_with_a_compensating_gap_is_rejected(tmp_path):
    """`count(*) == max(pick_no) == 128` is satisfied by a corrupted table
    with one pick_no duplicated and another missing entirely -- both
    aggregates still read 128. The real assumption worth checking is that
    pick_no is exactly the contiguous set 1..128, once each, and this must
    be rejected rather than accepted on the aggregates alone."""
    path = tmp_path / "666.duckdb"
    _seed_mock_league(path, N_PICKS_COMPLETE)
    conn = get_conn(str(path))
    # pick_no 51 becomes a second 50 -- count(*) and max(pick_no) are both
    # still 128, but 51 is now missing and 50 is doubled.
    conn.execute("UPDATE drafted SET pick_no = 50 WHERE pick_no = 51")
    conn.close()
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))

    counts = mock_backfill.backfill([str(path)], corpus)

    assert counts["incomplete"] == 1
    assert counts["drafts_recorded"] == 0
    assert len(dl.picks(corpus)) == 0


def test_a_pick_whose_player_is_not_in_the_pool_is_dropped_and_counted(tmp_path):
    """A pick whose player_id the pool join can't find must not disappear
    silently. The module docstring's whole argument for `_picks_frame`
    counting a drop rather than swallowing it: a silent drop here would
    look exactly like a smaller corpus rather than like the join failure it
    actually is."""
    path = tmp_path / "222.duckdb"
    _seed_mock_league(path, N_PICKS_COMPLETE)
    conn = get_conn(str(path))
    # No `weekly`/`adp` row anywhere ever produces a pool entry for this id,
    # so the pool join for pick_no 64 must miss.
    conn.execute(
        "UPDATE drafted SET player_id = 'nobody_on_the_board' WHERE pick_no = 64")
    conn.close()
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))

    counts = mock_backfill.backfill([str(path)], corpus)

    assert counts["drafts_recorded"] == 1
    assert counts["picks_dropped"] == 1
    assert counts["picks_recorded"] == 127

    got = dl.picks(corpus, source=dl.SOURCE_MOCK)
    assert len(got) == 127
    assert "nobody_on_the_board" not in set(got["player_id"])


def test_rerunning_the_backfill_produces_no_duplicates(tmp_path):
    """Idempotent by draft_id, same as every other corpus writer -- running
    this twice (as a cron or a re-run after a crash would) must not double
    the picks."""
    path = tmp_path / "777.duckdb"
    _seed_mock_league(path, N_PICKS_COMPLETE)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))

    mock_backfill.backfill([str(path)], corpus)
    mock_backfill.backfill([str(path)], corpus)

    assert len(dl.picks(corpus)) == 128


def test_draft_id_is_stable_across_a_checkpoint_that_only_moves_mtime(tmp_path):
    """The CRITICAL bug this fix round exists for. Several real league files
    carry a `.wal` sidecar with a LATER mtime than the main file -- DuckDB
    replays the WAL on open, so a read-only connection already sees the
    complete picks, but the main file itself is not yet checkpointed. The
    first thing that opens such a file read-write and checkpoints it moves
    the main file's mtime forward for a draft that never changed. draft_id
    must not move with it, or the very next backfill inserts the same 128
    picks a second time."""
    path = tmp_path / "333.duckdb"
    _seed_mock_league(path, N_PICKS_COMPLETE)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))

    mock_backfill.backfill([str(path)], corpus)
    first_ids = set(dl.picks(corpus, source=dl.SOURCE_MOCK)["draft_id"])

    # Simulate a checkpoint (or any other reason the filesystem's mtime
    # moves) without a single pick changing.
    future = time.time() + 3600
    os.utime(path, (future, future))

    counts = mock_backfill.backfill([str(path)], corpus)
    second_ids = set(dl.picks(corpus, source=dl.SOURCE_MOCK)["draft_id"])

    assert first_ids == second_ids, "draft_id must not depend on the file's mtime"
    assert len(dl.picks(corpus, source=dl.SOURCE_MOCK)) == 128
    assert counts["stale_rows_removed"] == 0


def test_backfill_reconciles_a_stale_id_left_by_an_older_identity_scheme(tmp_path):
    """Migration path. The corpus already holds this file's 128 picks under
    a draft_id that does not match what the file's content hashes to now --
    standing in for a row written by the old, since-removed mtime-derived
    scheme. The next backfill must recognize the file's real content id,
    write it, and remove the stale row: 128 picks afterward, not 256."""
    path = tmp_path / "444.duckdb"
    _seed_mock_league(path, N_PICKS_COMPLETE)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    league_id = "444"
    stale_id = f"{dl.SOURCE_MOCK}:deadbeefdeadbeef"
    stale_picks = pd.DataFrame({
        "pick_no": range(1, N_PICKS_COMPLETE + 1),
        "round": [1] * N_PICKS_COMPLETE, "slot": [1] * N_PICKS_COMPLETE,
        "owner_key": [dl.anonymous_key(stale_id, 1)] * N_PICKS_COMPLETE,
        "is_anonymous": [True] * N_PICKS_COMPLETE,
        "player_id": [f"p{i}" for i in range(N_PICKS_COMPLETE)],
        "position": ["WR"] * N_PICKS_COMPLETE,
        "adp_rank": list(range(1, N_PICKS_COMPLETE + 1)),
        "proj_points": [100.0] * N_PICKS_COMPLETE,
    })
    dl.record(corpus, dl.DraftRecord(
        source=dl.SOURCE_MOCK, league_id=league_id, season=2026, teams=8,
        rounds=16, my_slot=None, draft_id=stale_id, picks=stale_picks))
    assert len(dl.picks(corpus, source=dl.SOURCE_MOCK)) == N_PICKS_COMPLETE

    counts = mock_backfill.backfill([str(path)], corpus)

    got = dl.picks(corpus, source=dl.SOURCE_MOCK)
    assert len(got) == N_PICKS_COMPLETE, "must land at 128, not 256 -- no duplication"
    assert stale_id not in set(got["draft_id"])
    assert counts["stale_rows_removed"] == 1


def test_backfill_never_touches_a_live_farmed_draft_in_the_same_league(tmp_path):
    """A concurrent live-mock-farming process can write a real, irreplaceable
    draft under the same league_id with `my_slot` set for a real seat. The
    reconciliation this module runs on every pass must never delete it,
    however different its draft_id is from the one just (re)recorded --
    `my_slot IS NULL` is what separates a backfilled row from a live one."""
    path = tmp_path / "555.duckdb"
    _seed_mock_league(path, N_PICKS_COMPLETE)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    league_id = "555"
    live_id = f"{dl.SOURCE_MOCK}:livefarmedlivefa"
    live_picks = pd.DataFrame({
        "pick_no": range(1, 5), "round": [1] * 4, "slot": [1, 2, 3, 4],
        "owner_key": ["espn:555:RealPerson"] * 4,
        "is_anonymous": [False] * 4,
        "player_id": ["x1", "x2", "x3", "x4"],
        "position": ["RB", "WR", "TE", "QB"],
        "adp_rank": [1.0, 2.0, 3.0, 4.0], "proj_points": [100.0] * 4,
    })
    dl.record(corpus, dl.DraftRecord(
        source=dl.SOURCE_MOCK, league_id=league_id, season=2026, teams=8,
        rounds=16, my_slot=3, draft_id=live_id, picks=live_picks))

    mock_backfill.backfill([str(path)], corpus)

    got = dl.picks(corpus, source=dl.SOURCE_MOCK)
    assert live_id in set(got["draft_id"]), "the live-farmed draft must survive"
    assert len(got[got["draft_id"] == live_id]) == 4


def test_a_league_with_no_picks_yet_is_skipped(tmp_path):
    """`get_conn` always creates an empty `drafted` table -- a league file
    that was provisioned but never drafted must be skipped, not treated as
    a zero-pick draft."""
    path = tmp_path / "111.duckdb"
    conn = get_conn(str(path))
    conn.close()
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))

    counts = mock_backfill.backfill([str(path)], corpus)

    assert counts["no_rows"] == 1
    assert counts["drafts_recorded"] == 0


def test_an_unopenable_file_is_skipped_not_fatal(tmp_path):
    """A file that is locked by another process, or corrupted, must not take
    the whole backfill down -- it is one bad file among many, not a reason
    to stop."""
    bad = tmp_path / "corrupt.duckdb"
    bad.write_bytes(b"not a real duckdb file")
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))

    counts = mock_backfill.backfill([str(bad)], corpus)

    assert counts["open_failed"] == 1
    assert counts["drafts_recorded"] == 0


def test_mock_settings_is_an_8_team_16_round_league():
    settings = mock_backfill._mock_settings()
    assert settings.teams == 8
    assert settings.rounds == 16


# ---------------------------------------------------------------------------
# corpus_report
#
# Two hand-built drafts, deliberately NOT 8-team, so a bucketing bug that
# only shows up off the backfill's own 8x16 shape (see corpus_report's "ROUND
# IS COMPUTED PER DRAFT FROM draft_log.teams" -- exactly the bug a hardcoded
# 8 would hide) has a chance to be caught here.
#
# D1 ("d1", teams=3, my_slot=None -> backfilled): 9 picks, rounds 1-3, so
# entirely in the "early" bucket. adp_rank is deliberately NOT pick_no, so
# the deviation numbers below are real arithmetic, not a trivially-zero
# fixture -- and pick 9's adp_rank is NULL, to exercise adp_null counting.
# `autodrafted` is left out of the frame entirely, matching a real backfill
# row: `record`'s `_shape` fills the missing column with NULL for every row.
#
# D2 ("d2", teams=2, my_slot=3 -> live-farmed): 14 picks, rounds 1-7 (rounds
# 1-3 early, 4-7 mid), adp_rank == pick_no throughout (deviation 0 -- a
# separate, deliberately trivial fixture from D1's, so the two populations'
# numbers are easy to tell apart in the assertions below), and a real
# `autodrafted` flag per pick, including two NULLs, to exercise "known vs
# unknown" on a population where the flag CAN be known.
#
# Expected numbers below were computed independently with the stdlib
# `statistics` module against this exact fixture (not by re-deriving
# `mock_backfill`'s own pandas code), so this is a real check against
# ground truth rather than the implementation checking itself.
# ---------------------------------------------------------------------------

def _seed_report_corpus(corpus) -> None:
    d1_picks = pd.DataFrame({
        "pick_no": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "position": ["RB", "WR", "RB", "WR", "RB", "TE", "WR", "RB", "DST"],
        "adp_rank": [1, 5, 2, 10, 4, 9, 1, 8, None],
    })
    dl.record(corpus, dl.DraftRecord(
        source=dl.SOURCE_MOCK, league_id="d1", season=2026, teams=3, rounds=3,
        my_slot=None, draft_id=f"{dl.SOURCE_MOCK}:reporttestd1aaaa",
        picks=d1_picks))

    d2_picks = pd.DataFrame({
        "pick_no": list(range(1, 15)),
        "position": ["RB", "WR", "RB", "WR", "RB", "WR",
                     "QB", "QB", "QB", "QB", "QB", "QB", "QB", "QB"],
        "adp_rank": list(range(1, 15)),
        "autodrafted": [True, False, True, False, True, False, None, None,
                        True, False, True, False, True, False],
    })
    dl.record(corpus, dl.DraftRecord(
        source=dl.SOURCE_MOCK, league_id="d2", season=2026, teams=2, rounds=7,
        my_slot=3, draft_id=f"{dl.SOURCE_MOCK}:reporttestd2bbbb",
        picks=d2_picks))


def test_corpus_report_computes_deviation_by_round_bucket_per_draft_teams(tmp_path):
    """Round must come from EACH draft's own `teams` column (3 for D1, 2 for
    D2), not a hardcoded 8 -- and the by-bucket deviation numbers must be
    real computed arithmetic, verified against an independent calculation."""
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    _seed_report_corpus(corpus)

    report = mock_backfill.corpus_report(corpus)

    combined = report["mock"]["combined"]
    assert combined["drafts"] == 2
    assert combined["picks"] == 23
    assert combined["adp_null"] == 1        # D1 pick 9 only
    assert combined["deviation"]["n"] == 22
    assert combined["deviation"]["mean"] == pytest.approx(0.9090909090909091)
    assert combined["deviation"]["median"] == pytest.approx(0.0)
    assert combined["by_bucket"]["early"]["n"] == 14
    assert combined["by_bucket"]["early"]["mean"] == pytest.approx(1.4285714285714286)
    assert combined["by_bucket"]["early"]["median"] == pytest.approx(0.0)
    assert combined["by_bucket"]["mid"] == {"n": 8, "mean": 0.0, "median": 0.0}
    assert combined["by_bucket"]["late"] == {"n": 0, "mean": None, "median": None}
    assert combined["first_round_positions"] == {"RB": 3, "WR": 2}
    assert combined["position_share_by_bucket"]["mid"] == {"QB": 100.0}
    assert combined["position_share_by_bucket"]["late"] == {}

    corpus.close()


def test_corpus_report_splits_backfilled_from_live_farmed_populations(tmp_path):
    """The whole point of the split (see corpus_report's own docstring): a
    backfilled draft's autodrafted flag is unknown on every pick, a
    live-farmed draft's is real ground truth, and neither population's
    numbers should be inferred from the other."""
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    _seed_report_corpus(corpus)

    report = mock_backfill.corpus_report(corpus)
    backfilled = report["mock"]["backfilled"]
    live = report["mock"]["live_farmed"]

    # D1 alone.
    assert backfilled["drafts"] == 1
    assert backfilled["picks"] == 9
    assert backfilled["adp_null"] == 1
    assert backfilled["autodrafted_known"] == 0
    assert backfilled["autodrafted_unknown"] == 9
    assert backfilled["autodrafted_share"] is None
    assert backfilled["deviation"] == {"n": 8, "mean": 2.5, "median": 2.0}
    assert backfilled["by_bucket"]["early"] == {"n": 8, "mean": 2.5, "median": 2.0}
    assert backfilled["by_bucket"]["mid"] == {"n": 0, "mean": None, "median": None}
    assert backfilled["first_round_positions"] == {"RB": 2, "WR": 1}
    assert backfilled["position_share_by_bucket"]["early"] == {
        "RB": 44.4, "WR": 33.3, "TE": 11.1, "DST": 11.1}

    # D2 alone.
    assert live["drafts"] == 1
    assert live["picks"] == 14
    assert live["adp_null"] == 0
    assert live["autodrafted_known"] == 12
    assert live["autodrafted_true"] == 6
    assert live["autodrafted_unknown"] == 2
    assert live["autodrafted_share"] == pytest.approx(0.5)
    assert live["deviation"] == {"n": 14, "mean": 0.0, "median": 0.0}
    assert live["by_bucket"]["early"] == {"n": 6, "mean": 0.0, "median": 0.0}
    assert live["by_bucket"]["mid"] == {"n": 8, "mean": 0.0, "median": 0.0}
    assert live["first_round_positions"] == {"RB": 1, "WR": 1}
    assert live["position_share_by_bucket"]["mid"] == {"QB": 100.0}

    corpus.close()


def test_corpus_report_on_an_empty_corpus_reports_zero_not_a_crash(tmp_path):
    """Nothing hardcodes a draft or pick count anywhere in this report --
    an empty corpus (before the first backfill or farm run) must describe
    itself as empty rather than raising on an empty frame."""
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))

    report = mock_backfill.corpus_report(corpus)

    for pop in report["mock"].values():
        assert pop["drafts"] == 0
        assert pop["picks"] == 0
        assert pop["deviation"] == {"n": 0, "mean": None, "median": None}
    corpus.close()


def test_open_corpus_read_only_cannot_write(tmp_path):
    """The connection `report_main` actually uses must be one DuckDB itself
    refuses to write through -- not merely a connection this module happens
    not to call INSERT/DELETE on. Deliberately NOT `dl.picks`/`dl.summary`
    here to read back with: both call `ensure_schema`, which issues `CREATE
    TABLE IF NOT EXISTS` -- and DuckDB rejects any CREATE, even a no-op one,
    against a read-only-attached database, so calling either against this
    connection would raise for a reason that has nothing to do with what
    this test is checking. `corpus_report` itself avoids that trap (see its
    own comment); a raw count is the equivalent read for a test."""
    path = tmp_path / "corpus.duckdb"
    seed = dl.corpus_conn(str(path))
    _seed_report_corpus(seed)
    seed.close()

    conn = mock_backfill._open_corpus_read_only(str(path))
    try:
        assert conn.execute(
            "SELECT count(*) FROM draft_log_pick").fetchone()[0] == 23
        with pytest.raises(duckdb.Error):
            conn.execute("DELETE FROM draft_log_pick")
    finally:
        conn.close()


def test_corpus_report_runs_end_to_end_against_a_true_read_only_connection(tmp_path):
    """The regression this fixture exists to catch: `dl.summary` (which an
    earlier version of `corpus_report` called directly) opens with
    `ensure_schema`, and DuckDB refuses that CREATE statement against a
    read-only-attached database even when the schema already matches --
    so `corpus_report` must never call it, only reproduce its query. This
    runs the whole function through the exact connection `report_main`
    uses in production, not a read-write stand-in."""
    path = tmp_path / "corpus.duckdb"
    seed = dl.corpus_conn(str(path))
    _seed_report_corpus(seed)
    seed.close()

    conn = mock_backfill._open_corpus_read_only(str(path))
    try:
        report = mock_backfill.corpus_report(conn)
    finally:
        conn.close()
    assert report["mock"]["combined"]["picks"] == 23


def test_report_main_on_a_missing_corpus_says_so_and_does_not_raise(tmp_path):
    """Before anything has ever been backfilled or farmed, there is no
    corpus file at all -- `report_main` must say that plainly rather than
    surface a raw duckdb IO error."""
    missing = str(tmp_path / "does-not-exist.duckdb")
    assert mock_backfill.report_main([missing]) == 0


# ---------------------------------------------------------------------------
# A file that says it is a different shape.
# ---------------------------------------------------------------------------


def _say_shape(path, teams=8, receptions=1.0, bench=None):
    """Give a seeded league file its own `league` row."""
    import dataclasses

    from scoring import league as league_mod

    base = mock_backfill._mock_settings()
    settings = dataclasses.replace(
        base, teams=teams,
        bench=base.bench if bench is None else bench,
        scoring={**base.scoring, "receptions": receptions})
    conn = get_conn(str(path))
    try:
        write_table(conn, "league", pd.DataFrame([
            {"season": 2026, "settings_json": league_mod.to_json(settings)}]))
    finally:
        conn.close()
    return settings


@pytest.mark.parametrize("shape,why", [
    ({"teams": 12}, "twelve seats, not eight"),
    ({"receptions": 0.0}, "standard scoring, not PPR"),
    ({"bench": 4}, "fifteen rounds, not sixteen"),
])
def test_a_file_that_says_another_shape_is_skipped(tmp_path, shape, why):
    """Everything here is priced and recorded as 8x16 PPR. That was safe
    while every mock was; now that the corpus is COUNTED per shape -- and the
    pooled depth is bounded by the shallowest shape in it -- a 12-team file
    recorded as 8x16 bends the answers for every other draft in the file."""
    path = tmp_path / "12345.duckdb"
    _seed_mock_league(path, N_PICKS_COMPLETE)
    _say_shape(path, **shape)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    try:
        counts = mock_backfill.backfill([str(path)], corpus)
        held = corpus.execute("SELECT count(*) FROM draft_log").fetchone()[0]
    finally:
        corpus.close()

    assert counts["wrong_shape"] == 1, why
    assert counts["drafts_recorded"] == 0
    assert held == 0


def test_a_file_that_says_the_shape_we_assume_is_recorded(tmp_path):
    """The guard is about disagreement, not about having an opinion."""
    path = tmp_path / "12345.duckdb"
    _seed_mock_league(path, N_PICKS_COMPLETE)
    _say_shape(path)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    try:
        counts = mock_backfill.backfill([str(path)], corpus)
    finally:
        corpus.close()

    assert counts["wrong_shape"] == 0 and counts["drafts_recorded"] == 1


def test_a_backfilled_draft_carries_its_scoring(tmp_path):
    """The shape a reader groups by is (teams, format), and the format is one
    number: written on its own rather than left inside the settings blob, the
    same as the live farm writes it."""
    path = tmp_path / "12345.duckdb"
    _seed_mock_league(path, N_PICKS_COMPLETE)
    corpus = dl.corpus_conn(str(tmp_path / "corpus.duckdb"))
    try:
        mock_backfill.backfill([str(path)], corpus)
        blob = corpus.execute(
            "SELECT scoring_json FROM draft_log").fetchone()[0]
    finally:
        corpus.close()

    assert dl.draft_format(blob) == "ppr"


def test_a_file_with_no_league_row_keeps_the_assumption(tmp_path):
    """Which is every file `data/leagues/` actually holds. `league.load`
    would answer `default_settings()` for one of these, which is exactly the
    shape being assumed -- so the guard reads the table itself."""
    path = tmp_path / "12345.duckdb"
    _seed_mock_league(path, N_PICKS_COMPLETE)
    conn = mock_backfill._open_read_only(str(path))
    try:
        assert mock_backfill.file_settings(conn) is None
    finally:
        conn.close()
