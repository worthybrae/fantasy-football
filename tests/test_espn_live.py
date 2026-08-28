import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.db import get_conn, write_table
from pipeline.espn_live import (COLUMNS, DraftEvent, LivePicks, apply_picks,
                                build_crosswalk, parse_frame,
                                picks_from_events, socket_url, translate)

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


def _frames():
    rows = [json.loads(l) for l in SOCKET_FIXTURE.read_text().splitlines() if l]
    return [str(r.get("payload") or "") for r in rows if r["kind"] == "ws-recv"]


def test_parse_frame_reads_a_pick():
    e = parse_frame("SELECTED 1 4430807 2")
    assert e == DraftEvent("SELECTED", ["1", "4430807", "2"])


def test_parse_frame_keeps_the_optional_manager_swid():
    """A human pick carries a trailing SWID, a bot pick does not. Parsing
    must not depend on the argument count."""
    e = parse_frame("SELECTED 2 4429795 2 {8491403C-A53F-4257-8D52-F8AE32CED897}")
    assert e.verb == "SELECTED"
    assert e.args[:3] == ["2", "4429795", "2"]
    assert e.args[3].startswith("{")


def test_parse_frame_ignores_blank_and_malformed_frames():
    assert parse_frame("") is None
    assert parse_frame("   ") is None


def test_parse_frame_does_not_choke_on_the_binary_init_blob():
    """INIT carries base64 of a binary structure. It must parse as an event
    like any other and simply not be interpreted -- raising here would kill
    the consumer on the first frame of every draft."""
    e = parse_frame("INIT AAAAAQAAAAELvB3TAAAAAgAAAAE=")
    assert e.verb == "INIT"


def test_picks_from_events_numbers_picks_by_stream_order():
    """`pick_no` is not in the protocol -- it is the position of the
    SELECTED event. `_drafted_state` attributes rosters by it, so an
    off-by-one here misassigns every roster silently."""
    events = [parse_frame(p) for p in [
        "STATE 1",
        "SELECTING 1 30000",
        "SELECTED 1 111 2",
        "AUTOSUGGEST 999",
        "SELECTING 2 30000",
        "SELECTED 2 222 4",
    ]]
    out = picks_from_events(events, {111: "g1", 222: "g2"})
    assert list(out.rows["player_id"]) == ["g1", "g2"]
    assert list(out.rows["pick_no"]) == [1, 2]
    assert out.unmapped == []


def test_picks_from_events_reports_an_unmapped_pick_without_shifting_pick_no():
    """An unmapped pick still consumed a pick slot. If it were skipped
    entirely the following picks would all shift up one and be attributed to
    the wrong teams."""
    events = [parse_frame(p) for p in
              ["SELECTED 1 111 2", "SELECTED 2 999 4", "SELECTED 3 333 2"]]
    out = picks_from_events(events, {111: "g1", 333: "g3"})
    assert list(out.rows["player_id"]) == ["g1", "g3"]
    assert list(out.rows["pick_no"]) == [1, 3]
    assert out.unmapped == [{"espn_player_id": 999, "overall_pick": 2}]


def test_picks_from_events_ignores_every_non_pick_verb():
    events = [parse_frame(p) for p in
              ["STATE 1", "CLOCK 0 63696", "PING x", "PONG x",
               "AUTOSUGGEST 4429795", "JOINED 2 {A}", "TOKEN 1:2:3",
               "AUTODRAFT 2 false", "SELECTING 1 30000"]]
    out = picks_from_events(events, {})
    assert out.rows.empty
    assert out.unmapped == []


def test_picks_from_events_reports_a_non_integer_player_id_instead_of_dropping_it():
    """A SELECTED frame whose id argument doesn't parse as int must still
    consume its pick number and land in `unmapped`. The bug this guards
    against incremented pick_no first but then `continue`d past both `rows`
    and `unmapped` on the failed int(), so the pick vanished entirely and
    that player stayed recommendable as if still on the board."""
    events = [parse_frame(p) for p in
              ["SELECTED 1 111 2", "SELECTED 2 abc 4", "SELECTED 3 333 2"]]
    out = picks_from_events(events, {111: "g1", 333: "g3"})
    assert list(out.rows["player_id"]) == ["g1", "g3"]
    assert list(out.rows["pick_no"]) == [1, 3]
    assert out.unmapped == [{"espn_player_id": None, "overall_pick": 2}]


def test_picks_from_events_a_short_selected_frame_still_consumes_a_pick_number():
    """A SELECTED frame missing its id argument entirely must not be skipped
    without incrementing pick_no. The bug this guards against shared one
    `continue` branch between the verb check and the arity check, ahead of
    the increment, so a short frame consumed no pick number at all and every
    later pick_no shifted down by one -- handing a real player to whoever
    was on the clock for the wrong pick."""
    events = [parse_frame(p) for p in
              ["SELECTED 1 111 2", "SELECTED 2", "SELECTED 3 333 2"]]
    out = picks_from_events(events, {111: "g1", 333: "g3"})
    assert list(out.rows["player_id"]) == ["g1", "g3"]
    assert list(out.rows["pick_no"]) == [1, 3]
    assert out.unmapped == [{"espn_player_id": None, "overall_pick": 2}]


def test_socket_url_is_the_shape_espn_actually_opened():
    url = socket_url("196877779", "{8491403C-A53F-4257-8D52-F8AE32CED897}",
                     "1:196877779:2:{8491403C-A53F-4257-8D52-F8AE32CED897}:-1781796296")
    assert url.startswith("wss://fantasydraft.espn.com/game-1/league-196877779/JOIN")
    assert "196877779" in url


@pytest.mark.skipif(not SOCKET_FIXTURE.exists(), reason="no draft capture")
def test_the_real_capture_folds_into_picks():
    """End to end against 161 frames recorded from a live ESPN mock draft.

    Every SELECTED in the capture must become either a row or an unmapped
    entry -- none may vanish -- and pick numbers must be dense and ordered.
    """
    frames = _frames()
    events = [e for e in (parse_frame(p) for p in frames) if e]
    selected = [e for e in events if e.verb == "SELECTED"]
    assert len(selected) >= 9

    ids = {int(e.args[1]): f"p{e.args[1]}" for e in selected}
    out = picks_from_events(events, ids)
    assert len(out.rows) == len(selected)
    assert out.unmapped == []
    assert list(out.rows["pick_no"]) == list(range(1, len(selected) + 1))

    half = dict(list(ids.items())[: len(ids) // 2])
    partial = picks_from_events(events, half)
    assert len(partial.rows) + len(partial.unmapped) == len(selected)


def test_picks_from_events_dedups_a_reconnect_replay():
    """ESPN re-sends every prior SELECTED when a socket rejoins a draft in
    progress. A player cannot be drafted twice, so a repeated id is that
    replay, not a new pick -- it must not consume a fresh pick_no, or one
    reconnect would double every pick and shift the whole board."""
    from pipeline.espn_live import parse_frame, picks_from_events

    xwalk = {1001: "a", 1002: "b", 1003: "c", 1004: "d"}
    frames = ["SELECTED 40 1001 2", "SELECTED 41 1002 2", "SELECTED 42 1003 2"]
    events = [parse_frame(f) for f in frames]

    once = picks_from_events(events, xwalk)
    assert list(once.rows["player_id"]) == ["a", "b", "c"]
    assert list(once.rows["pick_no"]) == [1, 2, 3]

    # Reconnect: the three replay verbatim, then a fourth, new pick lands.
    replayed = events + [parse_frame(f) for f in frames] \
        + [parse_frame("SELECTED 43 1004 2")]
    twice = picks_from_events(replayed, xwalk)
    assert list(twice.rows["player_id"]) == ["a", "b", "c", "d"]
    assert list(twice.rows["pick_no"]) == [1, 2, 3, 4]


def test_run_socket_listener_reconnects_after_a_drop(monkeypatch):
    """One dropped socket must not end the watch: ESPN ends even a pinged
    connection with 'no close frame received or sent' after a while, so
    run_socket_listener reopens and keeps folding picks. Both picks below land
    across two separate connections."""
    import threading

    from pipeline import draft_socket
    from pipeline.draft_listener import DraftListener
    from websockets.exceptions import ConnectionClosed

    monkeypatch.setattr(draft_socket, "RECONNECT_BACKOFF_SECONDS", 0.01)
    stop = threading.Event()
    scripts = [
        ["SELECTED 40 1001 2", "__DROP__"],   # conn 1: one frame, then dropped
        ["SELECTED 41 1002 2", "__STOP__"],   # conn 2: one frame, then we stop
    ]
    connects = {"n": 0}

    class FakeWS:
        def __init__(self, script):
            self.script = list(script)

        def send(self, _msg):
            pass

        def recv(self, timeout=None):
            if not self.script:
                raise TimeoutError
            item = self.script.pop(0)
            if item == "__DROP__":
                raise ConnectionClosed(None, None)
            if item == "__STOP__":
                stop.set()
                raise TimeoutError
            return item

        def close(self):
            pass

    def fake_connect(url, cookie_header):
        i = connects["n"]
        connects["n"] += 1
        return FakeWS(scripts[i] if i < len(scripts) else [])

    monkeypatch.setattr(draft_socket, "_connect", fake_connect)

    listener = DraftListener({1001: "a", 1002: "b"})
    activity = {"n": 0}
    draft_socket.run_socket_listener(
        listener, "100", "2", "{S}", "tok",
        on_change=lambda: None,
        on_activity=lambda: activity.__setitem__("n", activity["n"] + 1),
        stop_event=stop)

    assert connects["n"] >= 2, "did not reconnect after the drop"
    assert list(listener.picks().rows["player_id"]) == ["a", "b"]
    assert activity["n"] >= 2, "on_activity did not fire per frame"


def test_build_crosswalk_maps_dst_by_team():
    """DSTs never get an espn_id (ESPN omits team defenses from the rank sheet
    the board reads), so their picks -- SELECTed by ESPN's own negative D/ST
    id, -(16000+proTeamId) -- must be reconstructed from each DST row's NFL
    team, or every defense shows as a gap on the board."""
    import pandas as pd
    from pipeline.espn_live import build_crosswalk, _dst_espn_id
    board = pd.DataFrame([
        {"espn_id": 3117251.0, "player_id": "mccaffrey", "position": "RB", "team": "SF"},
        {"espn_id": float("nan"), "player_id": "adp_baltimore_defense",
         "position": "DST", "team": "BAL"},
        {"espn_id": float("nan"), "player_id": "adp_sf_defense",
         "position": "DST", "team": "SF"},
    ])
    x = build_crosswalk(board)
    assert x[3117251] == "mccaffrey"                       # ordinary espn_id path
    assert x[_dst_espn_id(33)] == "adp_baltimore_defense"  # -16033, Ravens
    assert x[-16025] == "adp_sf_defense"                   # 49ers, proTeamId 25


def _live(*pairs):
    return LivePicks(pd.DataFrame([{"player_id": p, "pick_no": n} for p, n in pairs],
                                  columns=COLUMNS), [])


def test_two_writers_on_one_file_serialise_instead_of_colliding(tmp_path):
    """Two leaguemates' listeners write the same league file's `drafted`
    (api/live.py keeps one room per browser). Unserialised, the second
    DELETE+INSERT lands inside the first's transaction and dies on the
    primary key; serialised per file, both land and the table ends up as
    the longer list."""
    import threading
    path = str(tmp_path / "league.duckdb")
    conn = get_conn(path)
    a, b = conn.cursor(), conn.cursor()
    start = threading.Barrier(2)
    errors = []

    def writer(cur, live):
        start.wait()
        for _ in range(25):
            try:
                apply_picks(cur, live)
            except Exception as exc:      # noqa: BLE001 -- what the test is for
                errors.append(exc)

    ta = threading.Thread(target=writer, args=(a, _live(("p1", 1), ("p2", 2))))
    tb = threading.Thread(target=writer, args=(b, _live(("p1", 1), ("p2", 2), ("p3", 3))))
    ta.start(); tb.start(); ta.join(); tb.join()
    assert errors == []
    rows = conn.execute("SELECT player_id, pick_no FROM drafted ORDER BY pick_no").fetchall()
    assert rows == [("p1", 1), ("p2", 2), ("p3", 3)]


def test_a_shorter_write_never_rolls_the_table_backwards(tmp_path):
    """A listener a frame behind the other must not shrink the shared table:
    the longer set stays, and the skipped write reports zero rows."""
    conn = get_conn(str(tmp_path / "league.duckdb"))
    assert apply_picks(conn, _live(("p1", 1), ("p2", 2), ("p3", 3))) == 3
    assert apply_picks(conn, _live(("p1", 1), ("p2", 2))) == 0
    rows = conn.execute("SELECT player_id FROM drafted ORDER BY pick_no").fetchall()
    assert [r[0] for r in rows] == ["p1", "p2", "p3"]
    # Same length is a replacement, not a shrink: ESPN still wins on content.
    assert apply_picks(conn, _live(("p1", 1), ("p2", 2), ("p9", 3))) == 3
    rows = conn.execute("SELECT player_id FROM drafted ORDER BY pick_no").fetchall()
    assert [r[0] for r in rows] == ["p1", "p2", "p9"]
