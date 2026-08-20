"""Rank the available pool by what a pick gains over waiting.

The board's `vor` says how much better a player is than a replacement-level
one at his position. On the clock that is the wrong question: what matters is
how much better he is than whoever will STILL BE THERE at your next pick. A
quarterback 91 points over replacement whose backup is 82 points over
replacement costs you 9 points to pass on, not 91 -- which is why ranking on
raw VOR reaches for quarterbacks and tight ends.

`gain_now` is that difference, weighted by whether the roster can actually
start the player and by whether this is the round to be filling that slot
at all (see `need_kind`: an open kicker slot in round 3 is not the same
need as an open WR1). It is deterministic: the only stochastic input is
`survive`, the per-player probability of lasting to the next pick, which
draft_sim.survival() estimates. That is a far lower-variance quantity than
expected end-of-draft roster value -- it counts one event per rollout instead
of averaging a sum over fifteen simulated picks -- which is why the ranking
moved here and off search_pick.
"""
import numpy as np
import pandas as pd

from scoring.config import NEED_WEIGHTS
from scoring.draft_sim import FLEX_POSITIONS, _roster_cap, must_fill_positions

# How many points of gain a position the plan is fully confident in is worth
# on top of its measured `gain_now`, when the caller passes a `plan_bias`.
#
# Chosen from measurement, and the measurement is worth stating plainly
# because it does NOT show what the steer was hoped to show. Simulating 50
# drafts per cell across slots 1/2/5/8 at steer strengths 0, 5, 10, 20 and
# 40, the change in end-of-draft roster value ranged from -17 to +15 points
# on rosters of ~2500 -- while the 95% confidence interval on a difference
# at that sample size is +/-19 to +/-26. Every result sat inside the noise.
# Steering the ranking toward the plan does not measurably improve the
# roster at any strength tested.
#
# It is applied anyway, at a strength the sweep showed no degradation for,
# for a reason that is about the room rather than the roster: the plan and
# the recommendation list are two views of the same model, and a user who
# reads "round 3 is for a tight end" directly above a list headed by a
# receiver is being shown a contradiction the model does not actually have.
# What the steer buys is coherence between the two panels, and it is priced
# low enough that a genuinely large edge -- the cliffs it is derived from run
# to 86 points -- still wins outright.
PLAN_STEER_POINTS = 10.0

_NO_SLOT = "—"


def _flex_used(settings, counts: dict) -> int:
    """How many FLEX slots the roster's overflow already occupies."""
    return sum(max(0, counts.get(pos, 0) - settings.starters.get(pos, 0))
               for pos in FLEX_POSITIONS)


def need_kind(settings, counts: dict, position: str,
              turns_left: int | None = None) -> str:
    """Which of NEED_WEIGHTS' cases this position is in for this roster.

    `counts` is position -> how many the roster already holds, the shape
    draft_sim._seed_rosters produces.

    The roster-cap check runs first and is a hard ceiling, not a soft
    preference: `_roster_cap` is the same cap `draft_sim._legal_mask` uses to
    block a pick outright during simulation, so a position already at its cap
    (e.g. a 4th RB in this league's 2-starter/2-flex shape, or a 2nd K) is
    "capped" even if the roster overall still has bench room -- there is no
    slot, starter or bench, left for the position to occupy. Only a roster
    still under its own position cap can fall through to "bench".

    "deferred" is an open starter slot that it is not yet time to fill, and
    it exists because "is the slot open" was the whole of this model and it
    is only half of roster need. The owner's complaint: a kicker and a
    defense in the top fifteen in round 3, both on an open starter slot
    weighted 1.0 -- the same weight as an empty WR1 -- in a draft where
    nobody fills either before the last couple of rounds.

    TWO CONDITIONS, and both matter:

    - The position has no depth value at all: its roster cap equals its
      starter count, so a second one can never be rostered even on the
      bench. `_roster_cap` puts K and DST at 1 and nothing else at its
      starter count, so those are the two today -- but this is read off the
      league's own roster shape, not a list of position names, and a league
      that let you carry two kickers would not qualify.
    - The roster can still fill it later: `turns_left` picks remain and
      fewer starter slots than that are open. This is
      `draft_sim.must_fill_positions`, the same line the simulator uses to
      decide when an opponent's remaining picks have run down to their
      unfilled slots, so the ranking starts recommending a defense in the
      same round the simulator starts expecting one.

    `turns_left` is my remaining picks INCLUDING the one on the clock. None
    means the caller cannot say (every offline caller, and the tests that
    predate this), and then nothing is deferred -- the historical behaviour,
    since a deferral rule with no idea how many picks are left would be
    guessing at the half of the question it exists to answer.
    """
    caps = _roster_cap(settings)
    held = counts.get(position, 0)
    starters = settings.starters.get(position, 0)
    if held >= caps.get(position, held + 1):
        return "capped"
    if held < starters:
        if (turns_left is not None
                and caps.get(position, starters + 1) <= starters
                and position not in must_fill_positions(settings, counts,
                                                        turns_left)):
            return "deferred"
        return "starter"
    if position in FLEX_POSITIONS and _flex_used(settings, counts) < settings.flex_slots:
        return "flex"
    total = sum(counts.values())
    if total < settings.rounds:
        return "bench"
    return "capped"


def need_weight(settings, counts: dict, position: str,
                turns_left: int | None = None) -> float:
    return float(NEED_WEIGHTS[need_kind(settings, counts, position,
                                        turns_left)])


def fills_slot(settings, counts: dict, position: str,
               turns_left: int | None = None) -> str:
    """The roster slot this player would occupy, as the UI labels it.

    A deferred slot is labelled like the starter slot it is -- "K", "DST" --
    because that IS the slot he would fill. The label answers where he goes,
    not whether now is the time; the ranking answers the second question.
    """
    kind = need_kind(settings, counts, position, turns_left)
    if kind in ("starter", "deferred"):
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

    Same nine-column shape `rank_available` returns (this is what keeps
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
                                     "vor_points", "gain_now", "plan_steer",
                                     "survive_pct",
                                     "fills", "rank"])
    rows = [{
        "player_id": pool.player_id[idx],
        "position": str(pool.position[idx]),
        "proj_points": float(pool.points[idx]),
        "vor_points": float(pool.vor[idx]),
        "gain_now": None,
        # 0.0 rather than None, unlike its neighbours: the others are None
        # because this function genuinely cannot know them without a roster,
        # while a steer of exactly nothing is the true and complete answer
        # here -- there is no plan to steer toward, so nothing about this
        # frame's ordering was influenced by one.
        "plan_steer": 0.0,
        "survive_pct": None,
        "fills": None,
    } for idx in available]
    out = pd.DataFrame(rows).sort_values(
        "vor_points", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def rank_available(pool, settings, taken, counts: dict, survive,
                   turns_left: int | None = None,
                   plan_bias: dict | None = None) -> pd.DataFrame:
    """The available pool, ranked by gain_now (plus any plan steer) descending.

    `taken` is the pool-aligned boolean mask of players already drafted,
    `counts` my own roster's position counts, `survive` the pool-aligned
    probability each player is still there at my next pick, `turns_left` my
    remaining picks including the one on the clock (None = unknown, which
    turns the deferral rule off -- see `need_kind`).

    `plan_bias` maps position -> the draft plan's confidence (0-1) that this
    round goes to that position, from `scoring.plan.build_plan`. Given it,
    each row also carries `plan_steer` and the SORT uses gain_now +
    plan_steer -- but `gain_now` itself is left exactly as measured, and it
    is `gain_now` the room displays. Folding the steer into it would have
    been simpler and is the wrong trade: the column is documented, is read
    as a points figure, and is the number a user checks the tool against.
    A separate column keeps the displayed quantity honest and lets the room
    mark which rows the plan moved. Without `plan_bias`, `plan_steer` is
    0.0 for every row and the ordering is bit-identical to before.
    """
    available = np.flatnonzero(~np.asarray(taken))
    if available.size == 0:
        return pd.DataFrame(columns=["player_id", "position", "proj_points",
                                     "vor_points", "gain_now", "plan_steer",
                                     "survive_pct",
                                     "fills", "rank"])
    survive = np.asarray(survive, dtype=float)

    # One expected-best per position, not per player: it depends only on the
    # position's own survivors, so computing it inside the player loop would
    # redo the same sort once per player at that position.
    next_best = {}
    for pos in set(pool.position[available]):
        at_pos = available[pool.position[available] == pos]
        next_best[pos] = expected_best_next(pool.vor[at_pos], survive[at_pos])

    # One need_kind per POSITION, not per player: it reads `counts` and the
    # league, neither of which varies down the loop.
    kinds = {str(pos): need_kind(settings, counts, str(pos), turns_left)
             for pos in next_best}

    rows = []
    for idx in available:
        pos = str(pool.position[idx])
        weight = float(NEED_WEIGHTS[kinds[pos]])
        rows.append({
            "player_id": pool.player_id[idx],
            "position": pos,
            "proj_points": float(pool.points[idx]),
            "vor_points": float(pool.vor[idx]),
            "gain_now": weight * (float(pool.vor[idx]) - next_best[pos]),
            # Weighted by `need` like the gain itself, which is what keeps a
            # steer off a position I cannot use: NEED_WEIGHTS["capped"] is
            # 0.0, so a plan bias can never resurrect a capped row from the
            # bottom of the board into contention.
            "plan_steer": (weight * PLAN_STEER_POINTS
                           * float((plan_bias or {}).get(pos, 0.0))),
            "survive_pct": float(survive[idx]) * 100.0,
            "fills": fills_slot(settings, counts, pos, turns_left),
            # Not part of the result -- dropped after the sort below.
            "capped": weight == 0.0,
            "deferred": kinds[pos] == "deferred",
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
    #
    # DEFERRED positions (see need_kind) sort after everything else for the
    # same structural reason, and it has to be a sort key rather than a
    # smaller weight because a weight CANNOT move the number that put them
    # there. Live at pick 19, round 3, an 8-team draft: no opponent takes a
    # kicker or a defense inside the horizon, so every one of them survives
    # at 100%, `expected_best_next` for the position equals its own best
    # available, and `gain_now` is EXACTLY 0.0 -- times any weight, still
    # exactly 0.0, and 0.0 outranks the -1 and -2 of real players who are
    # genuinely worth slightly less than what will survive. Houston Defense
    # 12th and Brandon Aubrey 13th, above Malik Nabers and Chris Olave.
    # Multiplying by NEED_WEIGHTS["deferred"] leaves that ordering to the
    # digit; only ordering the block last changes it. The weight is still
    # applied, and does the other half of the job -- see its comment in
    # scoring/config.py for the round-9 measurement where the gain is a real
    # +3.32 rather than an artifact.
    #
    # Not a filter: the rows are all still there, in gain order within the
    # block, and the moment `must_fill_positions` says the slots can no
    # longer be deferred they stop being deferred and rank on merit -- which
    # on the real board is a kicker at rank 1 in round 14 and a defense at
    # rank 1 in round 15.
    out = pd.DataFrame(rows)
    # The steer moves the ORDER, never the reported gain. Held in a scratch
    # column so the sort can read the combined figure while `gain_now` stays
    # the measured one; dropped with the other two scratch keys below.
    out["_ranked_by"] = out["gain_now"] + out["plan_steer"]
    out = out.sort_values(
        ["capped", "deferred", "_ranked_by"],
        ascending=[True, True, False]).drop(
            columns=["capped", "deferred", "_ranked_by"]).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out
