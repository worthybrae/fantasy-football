import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.db import get_conn, write_table
from pipeline.espn_live import COLUMNS, LivePicks, apply_picks, build_crosswalk, translate

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


def test_build_crosswalk_maps_espn_ids_off_the_board():
    board = pd.DataFrame([
        {"player_id": "g1", "espn_id": 4429795.0},
        {"player_id": "g2", "espn_id": 4430807.0},
        {"player_id": "g3", "espn_id": None},      # no ESPN row for this player
    ])
    x = build_crosswalk(board)
    assert x == {4429795: "g1", 4430807: "g2"}


def test_build_crosswalk_survives_a_board_without_the_column():
    """A board built before espn_id existed, or by a fixture that omits it.
    An empty crosswalk makes every pick unmapped and loudly visible, which
    is the right failure -- a KeyError here would take the API down."""
    assert build_crosswalk(pd.DataFrame([{"player_id": "g1"}])) == {}
    assert build_crosswalk(None) == {}


SOCKET_FIXTURE = Path("tests/fixtures/espn_draft_socket.jsonl")


@pytest.mark.skipif(not SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_the_real_captured_picks_are_all_espn_ids_we_could_map():
    """Recorded from a live ESPN mock draft. The protocol is plain text:

        SELECTING <team> <msRemaining>
        SELECTED  <team> <espnPlayerId> <lineupSlotId> [<managerSWID>]

    This pins the shape the live consumer parses. It is the evidence that
    replaced a guess -- ESPN's REST API serves no real playerId at all while
    a draft runs, so this socket is the only live source.
    """
    frames = [json.loads(l) for l in SOCKET_FIXTURE.read_text().splitlines() if l]
    picks = [str(f.get("payload", "")).split() for f in frames
             if str(f.get("payload", "")).startswith("SELECTED ")]
    assert len(picks) >= 9, "capture should carry a round's worth of picks"
    for parts in picks:
        assert parts[0] == "SELECTED"
        assert parts[1].isdigit()                    # team id
        assert parts[2].isdigit()                    # espn player id
        assert parts[3].isdigit()                    # lineup slot id
    # Every id is a plausible ESPN player id, not a placeholder like the -1
    # the REST board is full of.
    assert all(int(p[2]) > 0 for p in picks)


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


def _conn_with_drafted(tmp_path, rows):
    conn = get_conn(str(tmp_path / "d.duckdb"))
    for player_id, pick_no in rows:
        conn.execute("INSERT INTO drafted VALUES (?, ?)", [player_id, pick_no])
    return conn


def test_apply_picks_replaces_rather_than_appends(tmp_path):
    """The constraint is the whole point: never patch incrementally.

    If ESPN's list and ours disagree, appending the difference produces a
    drafted table that is neither -- and every roster, need and cap
    downstream is computed against it. Replacing wholesale means the table
    always equals ESPN's list exactly.
    """
    conn = _conn_with_drafted(tmp_path, [("stale", 1), ("alsostale", 2)])
    live = LivePicks(pd.DataFrame([{"player_id": "g1", "pick_no": 1},
                                   {"player_id": "g2", "pick_no": 2}]),
                     [])
    assert apply_picks(conn, live) == 2
    got = conn.execute("SELECT player_id, pick_no FROM drafted ORDER BY pick_no").fetchall()
    assert got == [("g1", 1), ("g2", 2)]


def test_apply_picks_overwrites_a_manual_mark_espn_contradicts(tmp_path):
    """Reconciliation rule, stated once in the spec: ESPN wins."""
    conn = _conn_with_drafted(tmp_path, [("guessed_wrong", 1)])
    live = LivePicks(pd.DataFrame([{"player_id": "actual", "pick_no": 1}]), [])
    apply_picks(conn, live)
    got = conn.execute("SELECT player_id FROM drafted").fetchall()
    assert got == [("actual",)]


def test_apply_picks_clears_the_table_when_espn_reports_no_picks(tmp_path):
    """A draft that was reset, or a session pointed at the wrong league. The
    table must follow ESPN rather than keep whatever it had."""
    conn = _conn_with_drafted(tmp_path, [("g1", 1)])
    assert apply_picks(conn, LivePicks(pd.DataFrame(columns=COLUMNS), [])) == 0
    assert conn.execute("SELECT count(*) FROM drafted").fetchone()[0] == 0


def test_apply_picks_never_writes_a_null_pick_no(tmp_path):
    """_drafted_state raises on a null pick_no rather than misattributing.
    Guarding here means that path is unreachable from the poller.
    The table must survive a validation failure -- guard runs before DELETE."""
    conn = _conn_with_drafted(tmp_path, [("safe", 1)])
    live = LivePicks(pd.DataFrame([{"player_id": "g1", "pick_no": None}]), [])
    with pytest.raises(ValueError, match="null pick_no"):
        apply_picks(conn, live)
    # Verify the table was not touched by the failed attempt
    got = conn.execute("SELECT player_id FROM drafted").fetchall()
    assert got == [("safe",)]


def test_apply_picks_raises_on_duplicate_player_id_within_batch(tmp_path):
    """A duplicate player_id in a single batch indicates a crosswalk data-quality
    issue (two ESPN ids mapped to the same gsis_id). Raising surfaces the problem
    rather than silently deduplicating or overwriting. The table must survive."""
    conn = _conn_with_drafted(tmp_path, [("old", 1)])
    # Both ESPN ids (via different overall picks) map to the same board player_id
    live = LivePicks(pd.DataFrame([{"player_id": "same", "pick_no": 1},
                                   {"player_id": "same", "pick_no": 2}]), [])
    with pytest.raises(Exception):  # DuckDB raises on PRIMARY KEY violation
        apply_picks(conn, live)
    # Verify the table survived: DELETE succeeded, but the transaction rolled back
    got = conn.execute("SELECT player_id FROM drafted").fetchall()
    assert got == [("old",)]
