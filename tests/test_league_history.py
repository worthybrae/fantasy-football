import threading

import pandas as pd

from pipeline.db import get_conn, read_table
from tests.test_espn_league import DRAFT_PAYLOAD, TEAM_PAYLOAD
from tests.test_league import ESPN_SETTINGS

SEASON_PAYLOAD = {**DRAFT_PAYLOAD, **TEAM_PAYLOAD, **ESPN_SETTINGS}


def _raw_fetch(seasons, calls=None):
    """A `fetch(url, cookies, headers) -> (status, text)` like espn_drafts.http_fetch."""
    import json

    def fetch(url, cookies, headers=None):
        if calls is not None:
            calls.append((url, cookies, headers))
        year = next((s for s in seasons if str(s) in url), None)
        if year is None:
            return 404, "not found"
        if "/players?" in url:
            return 200, json.dumps([])
        return 200, json.dumps(SEASON_PAYLOAD)
    return fetch


def test_json_fetch_adapts_status_to_exceptions():
    from pipeline.league_history import json_fetch
    calls = []
    fetch = json_fetch(_raw_fetch([2025], calls), {"SWID": "{X}", "espn_s2": "s"})
    body = fetch("https://x/seasons/2025/segments/0/leagues/1")
    assert body["draftDetail"]["drafted"] is True
    assert calls[0][1] == {"SWID": "{X}", "espn_s2": "s"}
    assert calls[0][2] == {"x-fantasy-source": "kona"}
    import pytest
    with pytest.raises(FileNotFoundError):
        fetch("https://x/seasons/2019/segments/0/leagues/1")

    def forbidden(url, cookies, headers=None):
        return 403, "no"
    with pytest.raises(PermissionError):
        json_fetch(forbidden, {})("https://x/whatever")


def test_json_fetch_sends_the_player_filter_header():
    from pipeline.league_history import json_fetch
    from pipeline.espn_league import PLAYERS_FILTER
    calls = []
    json_fetch(_raw_fetch([2025], calls), {})("https://x/seasons/2025/players?view=players_wl")
    assert calls[0][2]["X-Fantasy-Filter"] == PLAYERS_FILTER


def test_import_history_writes_into_the_leagues_own_file(tmp_path):
    from pipeline.league_history import import_history, is_fresh
    from pipeline import leagues
    universal = str(tmp_path / "nfl.duckdb")
    get_conn(universal).close()
    root = str(tmp_path / "leagues")
    summary = import_history(
        "424242", {"SWID": "{X}", "espn_s2": "s"}, fetch=_raw_fetch([2025, 2024]),
        current_season=2026, universal_path=universal, root=root,
        adp_fetch=lambda season: pd.DataFrame([
            {"adp_name": "Justin Jefferson", "position": "WR", "team": "MIN", "adp": 1.5}]))
    assert summary["seasons"] == [2025, 2024]
    conn = get_conn(leagues.league_db_path("424242", root=root))
    assert len(read_table(conn, "draft_picks")) == 6
    assert set(read_table(conn, "historic_adp")["season"]) == {2025, 2024}
    assert read_table(conn, "historic_adp").iloc[0]["adp_rank"] == 1
    assert is_fresh(conn)
    conn.close()
    # The universal file was not written to.
    assert read_table(get_conn(universal), "draft_picks").empty


def test_import_history_only_fetches_adp_for_missing_seasons(tmp_path):
    """A season already in `historic_adp` is not re-fetched on the next import."""
    from pipeline.league_history import import_history
    universal = str(tmp_path / "nfl.duckdb")
    get_conn(universal).close()
    root = str(tmp_path / "leagues")
    league_id = "77"

    def adp_row(name, team, adp):
        return pd.DataFrame([{"adp_name": name, "position": "WR", "team": team, "adp": adp}])

    import_history(
        league_id, {}, fetch=_raw_fetch([2025]), current_season=2026,
        universal_path=universal, root=root,
        adp_fetch=lambda season: adp_row("Justin Jefferson", "MIN", 1.5))

    calls = []

    def recording_adp_fetch(season):
        calls.append(season)
        return adp_row("CeeDee Lamb", "DAL", 2.0)

    summary = import_history(
        league_id, {}, fetch=_raw_fetch([2025, 2024]), current_season=2026,
        universal_path=universal, root=root, adp_fetch=recording_adp_fetch)
    assert summary["seasons"] == [2025, 2024]
    # 2025 was already on disk from the first import; only 2024 is new.
    assert calls == [2024]


def test_import_history_survives_a_missing_adp_year(tmp_path):
    from pipeline.league_history import import_history
    universal = str(tmp_path / "nfl.duckdb")
    get_conn(universal).close()

    def adp_fetch(season):
        raise RuntimeError("feed down")
    summary = import_history(
        "1", {}, fetch=_raw_fetch([2025]), current_season=2026,
        universal_path=universal, root=str(tmp_path / "lg"), adp_fetch=adp_fetch)
    assert summary["seasons"] == [2025]


def test_is_fresh_is_false_without_an_import(tmp_path):
    from pipeline.league_history import is_fresh
    assert not is_fresh(get_conn(str(tmp_path / "t.duckdb")))


def test_spawn_import_if_stale_skips_fresh_and_never_raises(tmp_path, monkeypatch):
    from pipeline import league_history
    universal = str(tmp_path / "nfl.duckdb")
    get_conn(universal).close()
    root = str(tmp_path / "lg")
    started = []

    def spawn(name, fn):
        started.append(name)
        fn()
    ok = league_history.spawn_import_if_stale(
        "5", {}, spawn=spawn, fetch=_raw_fetch([2025]), current_season=2026,
        universal_path=universal, root=root, adp_fetch=lambda s: pd.DataFrame(
            columns=["adp_name", "position", "team", "adp"]))
    assert ok and started == ["history-5"]
    # Fresh now: nothing spawned.
    assert not league_history.spawn_import_if_stale(
        "5", {}, spawn=spawn, universal_path=universal, root=root)
    assert started == ["history-5"]
    # A broken fetch inside the thread is logged, not raised.
    def boom(url, cookies, headers=None):
        raise RuntimeError("boom")
    assert league_history.spawn_import_if_stale(
        "6", {}, spawn=spawn, fetch=boom, current_season=2026,
        universal_path=universal, root=root)
