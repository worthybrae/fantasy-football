"""Watch a live ESPN draft and print picks as they land, resolved to players.

End-to-end proof of the live pipeline without the API or any UI: opens the
draft room in a visible browser, observes its websocket, folds SELECTED frames
into picks, and resolves each ESPN player id against the board.

You draft in the window this opens. That is the design, not a limitation --
ESPN's socket URL carries a token with no known derivation, so we watch the
connection a real browser already holds rather than opening a second one as
the same team.

    .venv/bin/python -u scripts/try_listener.py "<espn draft room url>"

Answers two open questions at once:
  1. Can a draft room be entered by navigating straight to its URL, or does
     ESPN require going through the waiting room first?
  2. Does the listener actually receive frames and resolve them to players?

Read-only: nothing is written to the database.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.db import get_conn
from pipeline.draft_listener import DraftListener, run_listener
from pipeline.espn_league import STATE_PATH
from pipeline.espn_live import build_crosswalk
from scoring import league as league_mod
from scoring.board import build_board

LIVE_DB = Path("data/nfl.duckdb")


def main(url: str) -> int:
    # Work from a copy: DuckDB is single-writer and the API may hold the real
    # file. Nothing here writes, but the lock is taken on open either way.
    tmp = Path(tempfile.mkdtemp()) / "board.duckdb"
    shutil.copy(LIVE_DB, tmp)
    conn = get_conn(str(tmp))
    board = build_board(conn, settings=league_mod.load(conn))
    names = board.set_index("player_id")["name"].to_dict()
    crosswalk = build_crosswalk(board)
    print(f"board: {len(board)} players, {len(crosswalk)} with an ESPN id\n")

    listener = DraftListener(crosswalk)
    seen = [0]

    def on_change():
        picks = listener.picks()
        for row in picks.rows.itertuples(index=False):
            if row.pick_no <= seen[0]:
                continue
            who = names.get(row.player_id, row.player_id)
            print(f"  pick {row.pick_no:>3}  {who}", flush=True)
        seen[0] = len(picks.rows)
        for miss in picks.unmapped:
            print(f"  pick {miss['overall_pick']:>3}  UNMAPPED espn id "
                  f"{miss['espn_player_id']} -- still on your board",
                  flush=True)
        if listener.on_the_clock is not None:
            print(f"        on the clock: team {listener.on_the_clock}"
                  f"  ({listener.ms_remaining} ms)", flush=True)

    print("opening the draft room -- draft in the window that appears\n"
          "Ctrl-C when you have seen enough\n", flush=True)
    run_listener(listener, url, STATE_PATH, on_change=on_change)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: try_listener.py <espn draft room url>")
        raise SystemExit(2)
    try:
        raise SystemExit(main(sys.argv[1]))
    except KeyboardInterrupt:
        print("\nstopped")
        raise SystemExit(0)
