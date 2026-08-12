import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.db import get_conn, write_table
from pipeline.espn_live import LivePicks, build_crosswalk, translate

FIXTURE = Path("tests/fixtures/espn_live_draft.json")


def _payload(picks):
    return {"draftDetail": {"drafted": False, "picks": picks}}


def _pick(overall, player_id, team_id=1):
    return {"overallPickNumber": overall, "roundId": 1,
            "roundPickNumber": overall, "teamId": team_id,
            "playerId": player_id, "keeper": False}


def test_translate_maps_espn_ids_to_board_ids():
    out = translate(_payload([_pick(1, 111), _pick(2, 222)]),
                    {111: "g1", 222: "g2"})
    assert list(out.rows["player_id"]) == ["g1", "g2"]
    assert list(out.rows["pick_no"]) == [1, 2]
    assert out.unmapped == []


def test_translate_orders_by_overall_pick_not_payload_order():
    """pick_no drives roster attribution downstream -- draft_sim._drafted_state
    sorts on it and hands pick k to whoever was on the clock for pick k. ESPN
    has no obligation to serve picks in order, so relying on payload order
    would silently misattribute every roster."""
    out = translate(_payload([_pick(3, 333), _pick(1, 111), _pick(2, 222)]),
                    {111: "g1", 222: "g2", 333: "g3"})
    assert list(out.rows["pick_no"]) == [1, 2, 3]
    assert list(out.rows["player_id"]) == ["g1", "g2", "g3"]


def test_translate_reports_an_unmapped_pick_instead_of_dropping_it():
    """A dropped pick leaves a drafted player on the board and the tool
    recommends someone already gone. It must reach the UI."""
    out = translate(_payload([_pick(1, 111), _pick(2, 999)]), {111: "g1"})
    assert list(out.rows["player_id"]) == ["g1"]
    assert out.unmapped == [{"espn_player_id": 999, "overall_pick": 2}]


def test_translate_drops_espns_placeholder_picks():
    """ESPN pads an in-progress board with playerId -1 for slots nobody has
    filled yet. Those are not picks."""
    out = translate(_payload([_pick(1, 111), _pick(2, -1)]), {111: "g1"})
    assert list(out.rows["player_id"]) == ["g1"]
    assert out.unmapped == []


def test_translate_handles_an_empty_draft():
    out = translate(_payload([]), {111: "g1"})
    assert out.rows.empty
    assert list(out.rows.columns) == ["player_id", "pick_no"]
    assert out.unmapped == []


def test_build_crosswalk_reads_sleeper_ids(tmp_path):
    conn = get_conn(str(tmp_path / "x.duckdb"))
    write_table(conn, "sleeper_ids", pd.DataFrame([
        {"gsis_id": "g1", "espn_id": 111, "sleeper_name": "A", "position": "RB", "team": "DET"},
        {"gsis_id": "g2", "espn_id": 222, "sleeper_name": "B", "position": "WR", "team": "GB"},
        {"gsis_id": None, "espn_id": 333, "sleeper_name": "C", "position": "TE", "team": "SF"},
    ]))
    x = build_crosswalk(conn)
    assert x[111] == "g1" and x[222] == "g2"
    assert 333 not in x        # no gsis_id means no board player_id


@pytest.mark.skipif(not FIXTURE.exists(), reason="run Task 1's recorder first")
def test_translate_handles_the_real_recorded_payload():
    """Against a real ESPN payload, not an invented one. Everything the
    fixture carries must either map or be reported -- silence is the failure
    this guards."""
    payload = json.loads(FIXTURE.read_text())
    out = translate(payload, {})
    assert len(out.unmapped) > 0          # crosswalk empty, so all unmapped
    assert out.rows.empty
    # And every unmapped entry names both fields the UI needs.
    assert all({"espn_player_id", "overall_pick"} == set(u) for u in out.unmapped)
