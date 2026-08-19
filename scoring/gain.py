"""Rank the available pool by what a pick gains over waiting.

The board's `vor` says how much better a player is than a replacement-level
one at his position. On the clock that is the wrong question: what matters is
how much better he is than whoever will STILL BE THERE at your next pick. A
quarterback 91 points over replacement whose backup is 82 points over
replacement costs you 9 points to pass on, not 91 -- which is why ranking on
raw VOR reaches for quarterbacks and tight ends.

`gain_now` is that difference, weighted by whether the roster can actually
start the player. It is deterministic: the only stochastic input is
`survive`, the per-player probability of lasting to the next pick, which
draft_sim.survival() estimates. That is a far lower-variance quantity than
expected end-of-draft roster value -- it counts one event per rollout instead
of averaging a sum over fifteen simulated picks -- which is why the ranking
moved here and off search_pick.
"""
import numpy as np
import pandas as pd

from scoring.config import NEED_WEIGHTS
from scoring.draft_sim import FLEX_POSITIONS, _roster_cap

_NO_SLOT = "—"


def _flex_used(settings, counts: dict) -> int:
    """How many FLEX slots the roster's overflow already occupies."""
    return sum(max(0, counts.get(pos, 0) - settings.starters.get(pos, 0))
               for pos in FLEX_POSITIONS)


def need_kind(settings, counts: dict, position: str) -> str:
    """Which of NEED_WEIGHTS' four cases this position is in for this roster.

    `counts` is position -> how many the roster already holds, the shape
    draft_sim._seed_rosters produces.

    The roster-cap check runs first and is a hard ceiling, not a soft
    preference: `_roster_cap` is the same cap `draft_sim._legal_mask` uses to
    block a pick outright during simulation, so a position already at its cap
    (e.g. a 4th RB in this league's 2-starter/2-flex shape, or a 2nd K) is
    "capped" even if the roster overall still has bench room -- there is no
    slot, starter or bench, left for the position to occupy. Only a roster
    still under its own position cap can fall through to "bench".
    """
    caps = _roster_cap(settings)
    held = counts.get(position, 0)
    if held >= caps.get(position, held + 1):
        return "capped"
    if held < settings.starters.get(position, 0):
        return "starter"
    if position in FLEX_POSITIONS and _flex_used(settings, counts) < settings.flex_slots:
        return "flex"
    total = sum(counts.values())
    if total < settings.rounds:
        return "bench"
    return "capped"


def need_weight(settings, counts: dict, position: str) -> float:
    return float(NEED_WEIGHTS[need_kind(settings, counts, position)])


def fills_slot(settings, counts: dict, position: str) -> str:
    """The roster slot this player would occupy, as the UI labels it."""
    kind = need_kind(settings, counts, position)
    if kind == "starter":
        n = settings.starters.get(position, 0)
        held = counts.get(position, 0)
        return position if n <= 1 else f"{position}{held + 1}"
    if kind == "flex":
        return "FLEX"
    if kind == "bench":
        return "BENCH"
    return _NO_SLOT


def expected_best_next(values, survive) -> float:
    """Expected value of the best survivor among these players.

    Walks them best-first and charges each one the probability that he
    survives AND nobody better did. The tail where nobody survives
    contributes zero, which is the right floor: it means the position is
    gone and waiting costs you everything.
    """
    values = np.asarray(values, dtype=float)
    survive = np.asarray(survive, dtype=float)
    if values.size == 0:
        return 0.0
    total = 0.0
    none_better = 1.0
    for i in np.argsort(-values, kind="stable"):
        p = float(survive[i])
        total += float(values[i]) * p * none_better
        none_better *= 1.0 - p
        if none_better <= 1e-12:
            break
    return total


def rank_available(pool, settings, taken, counts: dict, survive) -> pd.DataFrame:
    """The available pool, ranked by gain_now descending.

    `taken` is the pool-aligned boolean mask of players already drafted,
    `counts` my own roster's position counts, `survive` the pool-aligned
    probability each player is still there at my next pick.
    """
    available = np.flatnonzero(~np.asarray(taken))
    if available.size == 0:
        return pd.DataFrame(columns=["player_id", "position", "proj_points",
                                     "vor_points", "gain_now", "survive_pct",
                                     "fills", "rank"])
    survive = np.asarray(survive, dtype=float)

    # One expected-best per position, not per player: it depends only on the
    # position's own survivors, so computing it inside the player loop would
    # redo the same sort once per player at that position.
    next_best = {}
    for pos in set(pool.position[available]):
        at_pos = available[pool.position[available] == pos]
        next_best[pos] = expected_best_next(pool.vor[at_pos], survive[at_pos])

    rows = []
    for idx in available:
        pos = str(pool.position[idx])
        weight = need_weight(settings, counts, pos)
        rows.append({
            "player_id": pool.player_id[idx],
            "position": pos,
            "proj_points": float(pool.points[idx]),
            "vor_points": float(pool.vor[idx]),
            "gain_now": weight * (float(pool.vor[idx]) - next_best[pos]),
            "survive_pct": float(survive[idx]) * 100.0,
            "fills": fills_slot(settings, counts, pos),
        })
    out = pd.DataFrame(rows).sort_values(
        "gain_now", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out
