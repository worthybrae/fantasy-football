"""Monte Carlo draft simulator.

Roster value is the projected points of the best legal starting lineup, plus
an insurance term for the bench. Without that term every pick past the last
starting slot is worth exactly zero and the simulator's late rounds become
noise; with it, a backup behind a starter who is expected to miss games is
worth his own rate for the games he covers. That credit is earned once per
roster spot at the position -- not once per starter he backs up -- since one
bench player can't literally be in two places at once; see roster_value's
docstring for why.
"""
from typing import NamedTuple

import numpy as np
import pandas as pd

from pipeline.db import read_table
from scoring.board import _norm_name
from scoring.draft_model import EARLY_ROUNDS, FEATURE_NAMES, RUN_WINDOW

FLEX_POSITIONS = ("RB", "WR", "TE")
GAMES = 17
# Floor for players with no projection and no stat history, per position, so
# a K or a rookie DST never lands as NaN inside the lineup optimizer.
POSITION_FLOOR = {"QB": 180.0, "RB": 80.0, "WR": 80.0, "TE": 60.0,
                  "K": 110.0, "DST": 100.0}


def projections(conn, board: pd.DataFrame) -> pd.Series:
    """Projected season points per player_id.

    Ladder: ESPN's own season projection, then recency-weighted PPG scaled to
    a full season, then a per-position floor.
    """
    espn = read_table(conn, "espn_adp")
    lookup = {}
    if not espn.empty and "espn_proj" in espn.columns:
        valid = espn.dropna(subset=["espn_proj"])
        valid = valid[valid["espn_proj"] > 0]
        for _, row in valid.iterrows():
            lookup[(_norm_name(row["espn_name"]), row["position"])] = float(row["espn_proj"])

    values = []
    for _, row in board.iterrows():
        key = (_norm_name(row["name"]), row["position"])
        proj = lookup.get(key)
        if proj is None:
            stats = row.get("stats")
            ppg = stats.get("ppg") if isinstance(stats, dict) else None
            proj = float(ppg) * GAMES if ppg else None
        if proj is None or not np.isfinite(proj):
            proj = POSITION_FLOOR.get(row["position"], 80.0)
        values.append(proj)
    return pd.Series(values, index=board["player_id"].to_numpy(), dtype=float)


def best_lineup_points(roster, settings) -> float:
    """Points of the best legal starting lineup.

    Greedy is optimal here: FLEX accepts a superset of no dedicated slot's
    eligibility and every other slot is single-position, so filling dedicated
    slots best-first and handing FLEX the leftovers can never be beaten.
    """
    by_position = {}
    for pos, points in roster:
        by_position.setdefault(pos, []).append(points)
    for values in by_position.values():
        values.sort(reverse=True)

    total = 0.0
    leftovers = []
    for pos, count in settings.starters.items():
        values = by_position.get(pos, [])
        total += sum(values[:count])
        if pos in FLEX_POSITIONS:
            leftovers.extend(values[count:])
    leftovers.sort(reverse=True)
    return total + sum(leftovers[:settings.flex_slots])


def roster_value(roster, settings) -> float:
    """Starting-lineup points plus bench insurance.

    Each starter is expected to miss `(1 - durability/100) * 17` games. A
    single bench player is one body: he can cover for however many of his
    position's starters happen to be out, but he cannot be in two places at
    once, so the credit for him is earned once per roster, not once per
    starter he sits behind. Missed-game shares are summed across the
    starters at the position and clamped to 1.0 (a full season is the most
    coverage one backup can supply) before being multiplied by his own rate
    a single time. Crediting him separately behind every starter at a
    multi-slot position (e.g. twice at RB) would let one player's real
    season -- worth `backup_points` -- get counted as insurance more than
    once, which both overstates the position's value and, because Task 11's
    candidate search maximizes this exact quantity, would steer it toward
    overrating the third player at a thin, fragile position.
    """
    starters_only = [(pos, points) for pos, points, _ in roster]
    total = best_lineup_points(starters_only, settings)

    by_position = {}
    for pos, points, durability in roster:
        by_position.setdefault(pos, []).append((points, durability))
    for values in by_position.values():
        values.sort(reverse=True)

    for pos, count in settings.starters.items():
        values = by_position.get(pos, [])
        if len(values) <= count:
            continue
        # `values` is sorted descending and `backup_points` is the entry
        # right after the top `count` starters, so by construction it can
        # never exceed any of their points -- no per-starter min(...) cap is
        # needed to keep this backup from being credited for more than his
        # own rate.
        backup_points = values[count][0]
        missed_total = 0.0
        for _, durability in values[:count]:
            # `durability or 50.0` would be wrong here: 0.0 is a real,
            # legitimate value on this 0-100 scale (a player who played none
            # of his possible games) and Python's `or` treats it as falsy,
            # silently substituting the "unknown" default and understating
            # exactly the fragile-starter case this term exists to cover.
            safe_durability = 50.0 if durability is None or pd.isna(durability) else durability
            missed_total += max(0.0, 1.0 - safe_durability / 100.0)
        missed_total = min(1.0, missed_total)
        total += missed_total * backup_points
    return total


_POSITION_DUMMY_INDEX = {
    pos: FEATURE_NAMES.index(f"pos_{pos}")
    for pos in ("RB", "WR", "TE", "K", "DST")
}
_REACH = FEATURE_NAMES.index("reach")
_FALL = FEATURE_NAMES.index("fall")
_QB_EARLY = FEATURE_NAMES.index("qb_early")
_TE_EARLY = FEATURE_NAMES.index("te_early")
_NEED = FEATURE_NAMES.index("need")
_RUN = FEATURE_NAMES.index("run")


class SimPool(NamedTuple):
    player_id: np.ndarray
    norm: np.ndarray
    position: np.ndarray
    adp_rank: np.ndarray
    points: np.ndarray
    durability: np.ndarray


def snake_slots(teams: int, rounds: int) -> list:
    order = []
    for r in range(rounds):
        forward = list(range(1, teams + 1))
        order.extend(forward if r % 2 == 0 else forward[::-1])
    return order


def build_pool(conn, board: pd.DataFrame, settings) -> SimPool:
    points = projections(conn, board)
    ranked = board.copy()
    ranked["proj"] = ranked["player_id"].map(points)
    # Market rank is the model's notion of "where the board thinks he goes";
    # players the market never ranked sit at the back rather than dropping out,
    # since an opponent can still take them.
    ranked["market_rank"] = ranked["market_rank"].fillna(len(ranked) + 1)
    ranked = ranked.sort_values("market_rank").reset_index(drop=True)
    return SimPool(
        player_id=ranked["player_id"].to_numpy(),
        norm=ranked["name"].map(_norm_name).to_numpy(),
        position=ranked["position"].to_numpy(),
        adp_rank=np.arange(1, len(ranked) + 1, dtype=float),
        points=ranked["proj"].to_numpy(dtype=float),
        durability=ranked["durability"].fillna(50.0).to_numpy(dtype=float))


def _live_features(pool, available, overall_pick, roster, recent, settings):
    """Feature matrix for the currently available players, mirroring
    draft_model.feature_matrix exactly -- the fitted coefficients only mean
    anything against the same feature definitions they were fitted on."""
    positions = pool.position[available]
    ranks = pool.adp_rank[available]
    teams = max(settings.teams, 1)
    n = len(positions)
    X = np.zeros((n, len(FEATURE_NAMES)))

    delta = ranks - overall_pick
    X[:, _REACH] = np.maximum(0.0, delta) / teams
    X[:, _FALL] = np.maximum(0.0, -delta) / teams
    for pos, col in _POSITION_DUMMY_INDEX.items():
        X[:, col] = (positions == pos)

    round_no = (overall_pick - 1) // teams + 1
    early = 1.0 if round_no <= EARLY_ROUNDS else 0.0
    X[:, _QB_EARLY] = (positions == "QB") * early
    X[:, _TE_EARLY] = (positions == "TE") * early

    starters = settings.starters
    X[:, _NEED] = [1.0 if roster.get(p, 0) < starters.get(p, 0) else 0.0
                   for p in positions]
    window = recent[:RUN_WINDOW]
    X[:, _RUN] = [window.count(p) / RUN_WINDOW for p in positions]
    return X


def _roster_cap(settings) -> dict:
    """Most of each position anyone will carry. Learned coefficients cannot
    express a hard ceiling, so it is imposed as a mask instead."""
    caps = {pos: n + 2 for pos, n in settings.starters.items()}
    caps["QB"] = min(caps.get("QB", 3), 3)
    caps["K"] = 1
    caps["DST"] = 1
    return caps


# Greedy only weighs the best available players by projection. Scanning the
# whole pool would make every rollout O(pool) roster valuations, which at
# ~500 players x 15 picks x thousands of rollouts is the difference between
# seconds and hours -- and a player outside the top 40 by projection can
# never be the greedy pick anyway.
GREEDY_CANDIDATES = 40


def _greedy_choice(pool, available, roster, settings, caps):
    """My in-rollout policy: the available player who most increases roster
    value. One-ply greedy, which is what makes a rollout cheap enough to run
    thousands of times; the search in Task 11 is what looks further ahead."""
    current = [(pool.position[i], pool.points[i], pool.durability[i])
               for i in roster["indices"]]
    base = roster_value(current, settings)
    shortlist = available[np.argsort(-pool.points[available])][:GREEDY_CANDIDATES]
    best_idx, best_gain = None, -np.inf
    for i in shortlist:
        pos = pool.position[i]
        if roster["counts"].get(pos, 0) >= caps.get(pos, 99):
            continue
        gain = roster_value(current + [(pos, pool.points[i], pool.durability[i])],
                            settings) - base
        if gain > best_gain:
            best_idx, best_gain = i, gain
    return best_idx if best_idx is not None else (available[0] if len(available) else None)


def rollout(pool, settings, slot_managers, my_slot, taken, betas, rng,
            forced=None) -> float:
    n = len(pool.player_id)
    gone = taken.copy()
    caps = _roster_cap(settings)
    rounds = settings.rounds
    slots = snake_slots(settings.teams, rounds)
    rosters = {slot: {"counts": {}, "indices": []}
               for slot in range(1, settings.teams + 1)}
    recent = []
    already = int(gone.sum())

    for offset, slot in enumerate(slots[already:], start=already):
        available = np.flatnonzero(~gone)
        if len(available) == 0:
            break
        overall_pick = offset + 1
        roster = rosters[slot]
        if slot == my_slot:
            if forced is not None and not gone[forced]:
                choice = forced
                forced = None
            else:
                choice = _greedy_choice(pool, available, roster, settings, caps)
        else:
            beta = betas.get(slot_managers.get(slot))
            if beta is None:
                beta = np.zeros(len(FEATURE_NAMES))
            X = _live_features(pool, available, overall_pick, roster["counts"],
                               recent, settings)
            scores = X @ beta
            for j, i in enumerate(available):
                pos = pool.position[i]
                if roster["counts"].get(pos, 0) >= caps.get(pos, 99):
                    scores[j] = -np.inf
            if not np.isfinite(scores).any():
                choice = int(available[0])
            else:
                shifted = scores - np.nanmax(scores[np.isfinite(scores)])
                weights = np.where(np.isfinite(shifted), np.exp(shifted), 0.0)
                total = weights.sum()
                choice = int(available[0]) if total <= 0 else \
                    int(rng.choice(available, p=weights / total))
        if choice is None:
            break
        gone[choice] = True
        pos = pool.position[choice]
        rosters[slot]["counts"][pos] = rosters[slot]["counts"].get(pos, 0) + 1
        rosters[slot]["indices"].append(choice)
        recent.insert(0, pos)

    mine = rosters[my_slot]["indices"]
    return roster_value(
        [(pool.position[i], pool.points[i], pool.durability[i]) for i in mine],
        settings)
