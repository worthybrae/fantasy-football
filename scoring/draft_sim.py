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
import numpy as np
import pandas as pd

from pipeline.db import read_table
from scoring.board import _norm_name

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
