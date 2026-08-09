"""Run the draft simulator. Run: python -m pipeline.run_sim <my_slot> [rollouts]"""
import sys

from pipeline.db import get_conn, read_table
from scoring.draft_sim import DEFAULT_ROLLOUTS, run_sim


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m pipeline.run_sim <my_slot> [rollouts]")
        return 2
    my_slot = int(argv[1])
    rollouts = int(argv[2]) if len(argv) > 2 else DEFAULT_ROLLOUTS
    conn = get_conn()

    order = read_table(conn, "draft_order")
    if order.empty:
        teams = read_table(conn, "draft_teams")
        if teams.empty:
            print("No draft order and no imported teams -- "
                  "run `make espn-import` first.")
            return 1
        newest = teams[teams["season"] == teams["season"].max()]
        slot_managers = dict(zip(newest["slot"], newest["manager"]))
    else:
        slot_managers = dict(zip(order["slot"], order["manager"]))

    run_id = run_sim(conn, my_slot, slot_managers, n_rollouts=rollouts)
    results = read_table(conn, "sim_results")
    print(f"Run {run_id} — top candidates at slot {my_slot}:")
    for _, row in results.sort_values("rank").head(8).iterrows():
        print(f"  {row['rank']:>2}. {row['player_id']:<20} "
              f"{row['ev']:8.1f} ± {row['se']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
