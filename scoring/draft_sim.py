"""Monte Carlo draft simulator.

Roster value is the projected points of the best legal starting lineup, plus
an insurance term for the bench. Without that term every pick past the last
starting slot is worth exactly zero and the simulator's late rounds become
noise; with it, a backup behind a starter who is expected to miss games is
worth his own rate for the games he covers -- capped at what the starter he's
covering for was worth, so a bench player never nets more credit than "what
he actually provides."
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

    Each starter is expected to miss `(1 - durability/100) * 17` games; the
    best bench player at that position covers those games at his own rate.
    Because the bench players considered here are always ranked below every
    starter at the position (best_lineup_points already claimed the top
    players as starters), the backup's own rate is never higher than the
    starter's -- the `min(...)` below is a belt-and-suspenders cap, not a
    difference/"drop-off" calculation, so a durable backup is never credited
    for more than he actually provides.
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
        backup_points = values[count][0]
        for starter_points, durability in values[:count]:
            # `durability or 50.0` would be wrong here: 0.0 is a real,
            # legitimate value on this 0-100 scale (a player who played none
            # of his possible games) and Python's `or` treats it as falsy,
            # silently substituting the "unknown" default and understating
            # exactly the fragile-starter case this term exists to cover.
            safe_durability = 50.0 if durability is None or pd.isna(durability) else durability
            missed = max(0.0, 1.0 - safe_durability / 100.0)
            total += missed * min(backup_points, starter_points)
    return total
