"""Turn an in-progress ESPN draft payload into rows for the `drafted` table.

Pure translation: no network and no database beyond reading the crosswalk.
The poller in `api/live.py` supplies the payload and writes the result, which
keeps the part with all the edge cases testable against a recorded fixture.
"""
from typing import NamedTuple

import pandas as pd

from pipeline.db import read_table
from pipeline.espn_league import _is_real_pick

COLUMNS = ["player_id", "pick_no"]


class LivePicks(NamedTuple):
    rows: pd.DataFrame          # columns COLUMNS, sorted by pick_no
    unmapped: list              # [{"espn_player_id": int, "overall_pick": int}]


def build_crosswalk(conn) -> dict:
    """ESPN player id -> board player_id.

    `sleeper_ids.gsis_id` IS the board's player_id -- `scoring.market` joins
    ESPN to the board through exactly this column. Rows without a gsis_id
    cannot reach the board at all, so they are left out rather than mapped to
    something invented.
    """
    ids = read_table(conn, "sleeper_ids")
    if ids.empty or "espn_id" not in ids.columns:
        return {}
    usable = ids.dropna(subset=["espn_id", "gsis_id"])
    return {int(e): str(g) for e, g in zip(usable["espn_id"], usable["gsis_id"])}


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
    """
    if not live.rows.empty and live.rows["pick_no"].isna().any():
        raise ValueError(
            "refusing to write a null pick_no: draft_sim._drafted_state "
            "cannot attribute such a pick to a team and would raise")
    conn.execute("DELETE FROM drafted")
    for row in live.rows.itertuples(index=False):
        conn.execute("INSERT INTO drafted VALUES (?, ?)",
                     [str(row.player_id), int(row.pick_no)])
    return len(live.rows)
