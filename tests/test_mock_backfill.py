"""pipeline.mock_backfill: folding the ESPN mock drafts already sitting in
data/leagues/*.duckdb into the cross-league draft corpus.

Each synthetic league file below is seeded with just enough of the universal
reference tables (see pipeline.db's LEAGUE_TABLES/UNIVERSAL_TABLES split,
and tests/test_board.py's own `_seed` fixture, which this mirrors) for
`scoring.board.build_board` and `scoring.draft_sim.build_pool` to run --
which is the same pipeline the real backfill runs per draft, so a fixture
that exercised anything less would not be testing the real join.
"""
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
