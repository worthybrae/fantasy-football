"""The board and pool caches key on what the data IS, not which file it
came from: league files provisioned from one snapshot share one build."""
import pandas as pd
import pytest

import scoring.board_cache as bc
from pipeline.db import get_conn, record_freshness, read_table
from pipeline.leagues import provision_league, snapshot_universal
from scoring import league as league_mod

try:
    from tests.test_live_api import _seed_minimal_live_db
except ImportError:
    from test_live_api import _seed_minimal_live_db


@pytest.fixture(autouse=True)
def _fresh_caches():
    bc.clear()
    yield
    bc.clear()


def _two_leagues(tmp_path, stamp_meta=True):
    """Two league files provisioned from one snapshot of a seeded database."""
    shared = str(tmp_path / "nfl.duckdb")
    _seed_minimal_live_db(shared)
    conn = get_conn(shared)
    if stamp_meta:
        record_freshness(conn, "weekly", True, 10)
        record_freshness(conn, "adp", True, 10)
    else:
        conn.execute("DELETE FROM meta")
    snapshot = snapshot_universal(conn, shared)
    conn.close()
    root = str(tmp_path / "lg")
    a = get_conn(provision_league("1", universal_path=shared, root=root, snapshot=snapshot))
    b = get_conn(provision_league("2", universal_path=shared, root=root, snapshot=snapshot))
    return a, b


def _counting(monkeypatch, name):
    calls = []
    real = getattr(bc, name)

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(bc, name, spy)
    return calls


def test_two_league_files_from_one_snapshot_share_one_board_build(tmp_path, monkeypatch):
    a, b = _two_leagues(tmp_path)
    builds = _counting(monkeypatch, "build_board")
    settings = league_mod.load(a)
    first = bc.cached_build_board(a, settings=settings)
    second = bc.cached_build_board(b, settings=settings)
    assert builds == [1]
    assert first.drop(columns=["drafted"]).equals(second.drop(columns=["drafted"]))
    # Different settings are a different board.
    other = league_mod.LeagueSettings(**{**settings.__dict__, "teams": settings.teams + 2})
    bc.cached_build_board(b, settings=other)
    assert builds == [1, 1]
    a.close(); b.close()


def test_a_database_with_no_meta_still_keys_on_its_file(tmp_path, monkeypatch):
    a, b = _two_leagues(tmp_path, stamp_meta=False)
    assert read_table(a, "meta").empty
    builds = _counting(monkeypatch, "build_board")
    settings = league_mod.load(a)
    bc.cached_build_board(a, settings=settings)
    bc.cached_build_board(b, settings=settings)
    assert builds == [1, 1], "same content, no meta: two files, two builds"
    a.close(); b.close()


def test_the_pool_is_shared_the_same_way(tmp_path, monkeypatch):
    a, b = _two_leagues(tmp_path)
    settings = league_mod.load(a)
    board_a = bc.cached_build_board(a, settings=settings)
    board_b = bc.cached_build_board(b, settings=settings)
    pools = _counting(monkeypatch, "build_pool")
    pa = bc.cached_build_pool(a, board_a, settings)
    pb = bc.cached_build_pool(b, board_b, settings)
    assert pools == [1]
    assert pa is pb, "the cached object itself, not a copy"
    # A different player set is a different pool.
    smaller = board_a.iloc[:-1]
    bc.cached_build_pool(a, smaller, settings)
    assert pools == [1, 1]
    a.close(); b.close()


def test_the_fingerprint_lives_here_and_is_still_importable_from_live():
    from api.live import board_fingerprint as via_live
    assert via_live is bc.board_fingerprint
    frame = pd.DataFrame({"player_id": ["b", "a"]})
    assert bc.board_fingerprint(frame) == bc.board_fingerprint(frame.iloc[::-1])
