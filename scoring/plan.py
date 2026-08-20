"""The draft plan: which positions to take in which rounds, and why.

`gain_now` answers one question -- "given the board right now, who?" -- and
answers it well, but it is a per-pick verdict with no memory and no
foresight past its own horizon. It can never say "you are going running
back twice and then waiting on quarterback", because it is never asked
about a round it is not standing in.

This module asks the other question. It runs the simulator's own policy
over the whole remaining draft many times and reads two things off the
result:

  * `rounds_plan` -- what position that policy actually took at each of my
    turns, as a distribution over the runs. A round where 90 of 120 drafts
    took a tight end is a plan; a round split 41/36 between running back
    and receiver is a genuine coin-flip and is reported as one.

  * `cliffs` -- how many projected points I lose at each position by waiting
    one more round, measured as the drop in the best available player at
    that position between one of my turns and the next. This is the reason
    the plan looks the way it does, and it is the number a human can act on
    without trusting anything else here: on today's board at slot 2, waiting
    from pick 2 to pick 15 costs 86 points of running back and 85 of
    receiver but exactly 0 of quarterback and 0 of tight end, because those
    two tiers are still intact at 15. That is the whole argument for not
    spending an early pick on them.

Both come out of ONE simulation pass, so the plan and its stated reason can
never describe different drafts.

The cliff table covers QB/RB/WR/TE only. Kickers and defenses do appear in
`rounds_plan` -- the roster requires one of each and the policy defers them
to the last rounds, which is worth showing -- but a cliff for them is not a
decision anyone makes: `need_kind` holds them out of contention until the
end of the draft regardless of what the tiers are doing.
"""
from collections import Counter, defaultdict

import numpy as np

from scoring.draft_sim import _run_draft, snake_slots

# Enough runs that a 90/10 split is not noise and a 45/40 split is honestly
# reported as close, without the wall-clock of a real search. At ~35ms per
# full 8-team draft this is a little over four seconds, which is why the
# caller computes it off the request path.
PLAN_DRAFTS = 120

# The positions a cliff is meaningful for. See the module docstring for why
# K and DST are excluded here but not from `rounds_plan`.
CLIFF_POSITIONS = ("QB", "RB", "WR", "TE")

# Below this, a position is simulation noise rather than a branch of the
# plan, and listing it invites reading three digits of meaning into one run
# out of 120.
MIN_PLAN_PCT = 5


def _best_by_position(pool, gone):
    """Best projected points still on the board at each cliff position.

    0.0 when a position is exhausted, which is the right floor: it means
    waiting costs you everything that was there.
    """
    avail = ~gone
    out = {}
    for pos in CLIFF_POSITIONS:
        mask = avail & (pool.position == pos)
        out[pos] = float(pool.points[mask].max()) if mask.any() else 0.0
    return out


def build_plan(pool, settings, slot_managers, my_slot, taken, betas, seed,
               taken_order=None, n_drafts=PLAN_DRAFTS, should_abort=None):
    """Simulate the rest of the draft `n_drafts` times and report the plan.

    `should_abort`, when given, is called once per simulated draft and
    returning True makes this give up and return None. It exists because
    this function is slow enough to matter to something else: measured on
    the real board, the live ranking takes 1.23s alone but 4.38s while a
    plan is building -- 3.5x slower, from CPython's GIL, since both are
    Python-level simulation loops. The ranking is what the user acts on
    while the clock runs; the plan describes rounds twenty minutes away. So
    the plan yields, checking between whole drafts (never mid-draft, which
    would leave a half-counted round in the aggregate) and returning None
    for the caller to retry in the next lull.

    Resumes from exactly the state `_run_draft` resumes from -- see its
    contract for what `taken`/`taken_order` must agree on. Rounds already
    played are reported as settled fact from `taken_order` rather than
    predicted: the simulation only covers picks from here on, so it has
    nothing to say about a pick that already happened, and inventing a
    distribution for one would be fiction.
    """
    slots = snake_slots(settings.teams, settings.rounds)
    my_offsets = [o for o, s in enumerate(slots) if s == my_slot]
    taken = np.asarray(taken, dtype=bool)
    already = len(taken_order) if taken_order is not None else int(taken.sum())

    # Which round each of my remaining turns is, keyed by overall pick
    # number so the walk below can recognise its own turns in pick order.
    round_of_pick = {o + 1: r for r, o in enumerate(my_offsets)}

    pos_counts = [Counter() for _ in my_offsets]
    best_sums = [defaultdict(float) for _ in my_offsets]
    counted = [0] * len(my_offsets)

    for k in range(n_drafts):
        if should_abort is not None and should_abort():
            return None
        rng = np.random.default_rng(int(seed) + k)
        record = []
        _run_draft(pool, settings, slot_managers, my_slot, taken, betas, rng,
                   taken_order=taken_order, record=record)
        by_pick = dict(record)
        gone = taken.copy()
        for pick_no in range(already + 1, len(slots) + 1):
            idx = by_pick.get(pick_no)
            if idx is None:
                break               # pool exhausted; nothing further to read
            r = round_of_pick.get(pick_no)
            if r is not None:
                # Snapshot BEFORE marking this pick gone: what was available
                # to me at my turn includes the player I am about to take.
                for pos, pts in _best_by_position(pool, gone).items():
                    best_sums[r][pos] += pts
                counted[r] += 1
                pos_counts[r][str(pool.position[idx])] += 1
            gone[idx] = True

    rounds_plan, best_available = [], []
    for r, o in enumerate(my_offsets):
        pick_no = o + 1
        entry = {"round": r + 1, "pick": pick_no}
        if pick_no <= already:
            idx = taken_order[o] if taken_order is not None else None
            actual = str(pool.position[idx]) if idx is not None else None
            entry.update(
                is_past=True, actual=actual,
                positions=([{"position": actual, "pct": 100}] if actual else []))
        else:
            total = sum(pos_counts[r].values())
            positions = []
            if total:
                for pos, n in pos_counts[r].most_common():
                    pct = int(round(100.0 * n / total))
                    if pct >= MIN_PLAN_PCT:
                        positions.append({"position": pos, "pct": pct})
            entry.update(is_past=False, actual=None, positions=positions)
        rounds_plan.append(entry)

        if counted[r]:
            best_available.append({
                "round": r + 1, "pick": pick_no,
                "by_position": {p: round(best_sums[r][p] / counted[r], 1)
                                for p in CLIFF_POSITIONS}})

    # The cliff at round r is the drop between my turn at r and my turn at
    # r+1, so the last round I simulated has no cliff -- there is no next
    # turn to wait for. Joined on `round` rather than by position in the
    # list, because `best_available` skips rounds already played.
    by_round = {e["round"]: e["by_position"] for e in best_available}
    cliffs = []
    for e in best_available:
        nxt = by_round.get(e["round"] + 1)
        if nxt is None:
            continue
        cliffs.append({
            "round": e["round"], "pick": e["pick"],
            "by_position": {p: round(max(0.0, e["by_position"][p] - nxt[p]), 1)
                            for p in CLIFF_POSITIONS}})

    return {
        "my_slot": my_slot,
        "teams": settings.teams,
        "rounds": settings.rounds,
        "n_drafts": n_drafts,
        "as_of_pick": already,
        "rounds_plan": rounds_plan,
        "cliffs": cliffs,
        "best_available": best_available,
    }
