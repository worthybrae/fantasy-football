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
