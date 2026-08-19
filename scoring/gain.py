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


def available_by_vor(pool, taken) -> pd.DataFrame:
    """The available pool -- who is still on the board, full stop -- ranked
    by the board's own `vor_points` descending, for when there is no roster
    to rank FOR yet (my_slot not resolved: see api/live.py's DraftSession.my_slot
    and _recompute, which needs a slot to index `rosters`/`survival` into
    and skips entirely without one).

    A sibling function rather than a mode bolted onto `rank_available`,
    deliberately: that function's whole shape -- per-position `next_best`
    (needs `survive`), `need_weight`/`fills_slot` (needs `counts`, i.e. a
    roster), the capped-sorts-last tiebreak -- exists to answer "what does
    THIS PICK gain over waiting," a question with no meaning until there is
    a roster to gain FOR. Threading `settings=None, counts=None,
    survive=None` through all of that would turn every line of it into a
    None-check, for a function whose docstring and every existing caller
    promise a `gain_now`-ranked result. "Who is left" needs none of that
    machinery -- just the pool minus `taken` -- so it gets its own, much
    smaller function instead.

    Same eight-column shape `rank_available` returns (this is what keeps
    `LiveCandidate` one shape on the frontend, not two payload types to
    branch on): `gain_now`, `survive_pct` and `fills` are `None`, not `0.0`,
    `0.0` and `""` -- a fabricated zero here would print as a real
    recommendation ("take him now, he's worth nothing") for a player nobody
    has actually ranked yet. `rank` is still the list's own display order,
    1-based, over vor_points -- it is not a promise that this is optimal for
    anyone's roster.
    """
    available = np.flatnonzero(~np.asarray(taken))
    if available.size == 0:
        return pd.DataFrame(columns=["player_id", "position", "proj_points",
                                     "vor_points", "gain_now", "survive_pct",
                                     "fills", "rank"])
    rows = [{
        "player_id": pool.player_id[idx],
        "position": str(pool.position[idx]),
        "proj_points": float(pool.points[idx]),
        "vor_points": float(pool.vor[idx]),
        "gain_now": None,
        "survive_pct": None,
        "fills": None,
    } for idx in available]
    out = pd.DataFrame(rows).sort_values(
        "vor_points", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


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
            # Not part of the result -- dropped after the sort below.
            "capped": weight == 0.0,
        })
    # Capped players sort LAST, not by gain_now alone. NEED_WEIGHTS["capped"]
    # is 0.0, so `weight * (...)` is identically 0.0 for every one of them --
    # which puts them above every genuinely useful player whose gain is
    # negative. Late in a draft, when most positions are full, that is a
    # block of unpickable rows sitting at the top of the ranked list. `fills`
    # already labels them "—" so they are visible rather than misleading, but
    # last is where they belong.
    #
    # Sorted on a separate key rather than by pushing gain_now negative: the
    # spec pins gain_now at zero for a position at its roster cap ("gain_now
    # is zero for a position at its roster cap", §7), and that zero is the
    # honest number -- taking a player you cannot roster gains nothing, it
    # does not cost you points.
    out = pd.DataFrame(rows).sort_values(
        ["capped", "gain_now"], ascending=[True, False]).drop(
            columns="capped").reset_index(drop=True)
    out["rank"] = out.index + 1
    return out
