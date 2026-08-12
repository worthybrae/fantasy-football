"""Turn an in-progress ESPN draft payload into rows for the `drafted` table.

Pure translation: no network and no database beyond reading the crosswalk.
The poller in `api/live.py` supplies the payload and writes the result, which
keeps the part with all the edge cases testable against a recorded fixture.
"""
from typing import NamedTuple

import pandas as pd

from pipeline.espn_league import _is_real_pick

COLUMNS = ["player_id", "pick_no"]


class LivePicks(NamedTuple):
    rows: pd.DataFrame          # columns COLUMNS, sorted by pick_no
    unmapped: list              # [{"espn_player_id": int, "overall_pick": int}]


def build_crosswalk(board) -> dict:
    """ESPN player id -> board player_id, as an exact lookup.

    Reads `board["espn_id"]`, which `scoring.market._espn_ranks` resolves at
    board-build time along the same two paths it uses for ESPN's ranks. The
    name matching happens once, there, where there is no clock running.

    This replaced a `sleeper_ids.gsis_id` join, and the reason is measured
    rather than stylistic. Against a real ESPN mock draft, sleeper_ids
    resolved **13 of ESPN's top 100** and 2 of the first 9 actual picks --
    Bijan Robinson, Puka Nacua, Ja'Marr Chase, Jonathan Taylor, Jaxon
    Smith-Njigba, Christian McCaffrey and Amon-Ra St. Brown all failed. Its
    ESPN mappings are stale in exactly the place it matters, on the young
    players who go early. Through the board's own espn_id the same picks
    resolve 9 of 9, and 100 of ESPN's top 100.

    No unit test could have caught that: the fixture ids were invented, so
    they mapped by construction. Only a live draft exposed it.
    """
    if board is None or "espn_id" not in getattr(board, "columns", []):
        return {}
    pairs = board[["espn_id", "player_id"]].dropna()
    return {int(e): str(p) for e, p in zip(pairs["espn_id"], pairs["player_id"])}


def translate(payload: dict, crosswalk: dict) -> LivePicks:
    """Map a `mDraftDetail` payload onto board players.

    Sorted by ESPN's own `overallPickNumber`, never by payload order:
    `pick_no` is what `draft_sim._drafted_state` attributes rosters with, and
    ESPN makes no promise about the order it serves picks in.

    A pick whose player does not map is reported, not dropped. Dropping it
    would leave a drafted player on the board and let the tool recommend
    someone already gone -- silently, which is the worst way for this to fail.
    """
    picks = [p for p in (((payload.get("draftDetail") or {}).get("picks")) or [])
             if _is_real_pick(p)]
    picks.sort(key=lambda p: p.get("overallPickNumber") or 0)

    rows, unmapped = [], []
    for p in picks:
        espn_id = p.get("playerId")
        overall = p.get("overallPickNumber")
        player_id = crosswalk.get(int(espn_id)) if espn_id is not None else None
        if player_id is None:
            unmapped.append({"espn_player_id": int(espn_id),
                             "overall_pick": int(overall)})
            continue
        rows.append({"player_id": player_id, "pick_no": int(overall)})
    return LivePicks(pd.DataFrame(rows, columns=COLUMNS), unmapped)


def apply_picks(conn, live: LivePicks) -> int:
    """Make `drafted` equal ESPN's pick list exactly. Returns rows written.

    Wholesale replacement, never a patch. If our table and ESPN's list
    disagree -- a pick we missed, a manual mark that guessed wrong, a draft
    that was reset -- appending the difference produces a table that matches
    neither, and every roster, need and cap downstream is computed from it.
    Replacing means the table is always exactly what ESPN says.

    That is also the reconciliation rule for the manual `D` hotkey: ESPN wins.

    Validation (null pick_no) runs before any mutation. DELETE and INSERTs are
    wrapped in an explicit transaction so the table either fully becomes ESPN's
    list or is left entirely alone. A duplicate player_id within a batch raises
    rather than silently deduplicating, surfacing a crosswalk data-quality issue
    that should not be masked.
    """
    if not live.rows.empty and live.rows["pick_no"].isna().any():
        raise ValueError(
            "refusing to write a null pick_no: draft_sim._drafted_state "
            "cannot attribute such a pick to a team and would raise")
    try:
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM drafted")
        for row in live.rows.itertuples(index=False):
            conn.execute("INSERT INTO drafted VALUES (?, ?)",
                         [str(row.player_id), int(row.pick_no)])
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return len(live.rows)


SOCKET_HOST = "wss://fantasydraft.espn.com"
# ESPN's lineup slot ids, carried on every SELECTED frame. Kept for the
# consumer's benefit -- the board already knows each player's position, so
# nothing here needs to trust ESPN's slot to place a pick.
LINEUP_SLOTS = {0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "DST", 17: "K"}


class DraftEvent(NamedTuple):
    verb: str
    args: list


def parse_frame(payload: str) -> "DraftEvent | None":
    """One websocket frame -> one event, or None if it carries no event.

    Deliberately does not validate arity or interpret arguments. The protocol
    carries verbs this module has no interest in (CLOCK, PING, AUTOSUGGEST)
    and one, INIT, whose argument is a base64 binary blob -- raising on any
    of them would kill the consumer on the first frame of a real draft.
    Interpretation belongs to `picks_from_events`, which reads only what it
    understands.
    """
    text = (payload or "").strip()
    if not text:
        return None
    parts = text.split()
    return DraftEvent(parts[0], parts[1:])


def picks_from_events(events, crosswalk: dict) -> LivePicks:
    """Fold a draft event stream into the same shape `apply_picks` takes.

    `pick_no` is the 1-based position of each SELECTED event in the stream,
    because the protocol does not carry a pick number. `_drafted_state`
    attributes pick k to whoever was on the clock for pick k, so this
    ordering decides every roster downstream.

    An unmapped pick still consumes its number. Skipping it would shift every
    later pick up one and hand real players to the wrong teams -- silently,
    which is the failure mode this whole module is arranged to prevent.
    """
    rows, unmapped, pick_no = [], [], 0
    for event in events:
        if event is None or event.verb != "SELECTED" or len(event.args) < 2:
            continue
        pick_no += 1
        try:
            espn_id = int(event.args[1])
        except (TypeError, ValueError):
            continue
        player_id = crosswalk.get(espn_id)
        if player_id is None:
            unmapped.append({"espn_player_id": espn_id, "overall_pick": pick_no})
            continue
        rows.append({"player_id": player_id, "pick_no": pick_no})
    return LivePicks(pd.DataFrame(rows, columns=COLUMNS), unmapped)


def socket_url(league_id: str, swid: str, token: str) -> str:
    """The URL ESPN's own draft room opens.

    Query parameters are positional-ish and undocumented; these are the names
    and order observed in a real capture. `token` is the value the server
    sends back on its own TOKEN frame in a prior session.
    """
    return (f"{SOCKET_HOST}/game-1/league-{league_id}/JOIN"
            f"?1=1&2={league_id}&3=2&4={swid}&5={token}"
            f"&6=false&7=false&8=KONA")
