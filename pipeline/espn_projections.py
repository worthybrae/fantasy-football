"""ESPN's own PRESEASON projections, per player per season, with stat splits.

`pipeline/sources.parse_espn` already reads this payload and keeps exactly
one number from it -- `appliedTotal` -- and discards the 45-key `stats` dict
sitting beside it, which is where ESPN's projected carries, yards, targets
and receptions live. This module keeps the splits, and keeps them for past
seasons as well as the current one.

WHY THE HISTORY MATTERS. Without it there is no way to ask whether our own
projections are any good: ESPN's are the incumbent, they are free, and they
are what `scoring/board.projections` already leans on as the first rung of
its ladder. A model that cannot be compared to the thing it replaces is a
model nobody should trust.

These are genuinely PRESEASON numbers, not end-of-season ones dressed up.
Checked against realised outcomes: fantasy-point projections correlate 0.66
to 0.70 with what actually happened, at 70-117% mean absolute percentage
error. A projection contaminated with in-season results would sit above 0.95.

Seasons 2018 onwards answer; 2015-2017 return an HTTP error from the
endpoint. Coverage varies year to year for reasons ESPN does not explain --
2021, 2022, 2024 and 2026 are near-complete, while 2023 returns almost
nothing -- so callers must treat a thin season as thin rather than as
evidence about players.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import requests

from pipeline.sources import ESPN_URL, UA, _ESPN_POS
from scoring.config import CURRENT_SEASON

# ESPN's stat ids. Verified against a published ESPN player card (Bijan
# Robinson, 2026): 287 CAR / 1372 YDS / 8 TD / 76 REC / 708 YDS / 3 TD /
# 352.78 FPTS matched ids 23 / 24 / 25 / 53 / 42 / 43 and appliedTotal.
STAT_IDS = {
    "23": "proj_carries",     "24": "proj_rush_yards",  "25": "proj_rush_tds",
    "53": "proj_receptions",  "42": "proj_rec_yards",   "43": "proj_rec_tds",
    "58": "proj_targets",     "210": "proj_games",
    "0": "proj_pass_att",     "3": "proj_pass_yards",   "4": "proj_pass_tds",
    "20": "proj_interceptions",
}

FIRST_SEASON = 2018       # 2017 and earlier answer with an HTTP error


def fetch_projections(season: int, limit: int = 700) -> pd.DataFrame:
    """One season of ESPN preseason projections, one row per player."""
    headers = {**UA, "X-Fantasy-Filter": json.dumps(
        {"players": {"limit": limit,
                     "sortAdp": {"sortAsc": True, "sortPriority": 1}}})}
    resp = requests.get(ESPN_URL.format(year=season), headers=headers, timeout=45)
    resp.raise_for_status()

    rows = []
    for entry in resp.json().get("players", []):
        p = entry.get("player") or {}
        pos = _ESPN_POS.get(p.get("defaultPositionId"))
        if pos is None:
            continue
        block = next((st for st in p.get("stats") or []
                      if st.get("statSourceId") == 1
                      and st.get("statSplitTypeId") == 0
                      and st.get("seasonId") == season), None)
        if block is None:
            continue
        stats = block.get("stats") or {}
        row = {"season": season, "espn_id": p.get("id"),
               "espn_name": p.get("fullName"), "position": pos,
               "proj_points": block.get("appliedTotal")}
        # Absent rather than zero: ESPN omits a stat it does not project, and
        # a running back with no `53` has no receptions PROJECTED, which is a
        # different claim from a projection of zero receptions.
        for stat_id, name in STAT_IDS.items():
            row[name] = stats.get(stat_id)
        rows.append(row)

    return pd.DataFrame(rows, columns=["season", "espn_id", "espn_name",
                                       "position", "proj_points",
                                       *STAT_IDS.values()])


def fetch_all(seasons=None, pause: float = 1.2) -> pd.DataFrame:
    """Every season we can get. Paced, because this is somebody's API and a
    backfill asks for nine years at once."""
    seasons = seasons or range(FIRST_SEASON, CURRENT_SEASON + 1)
    frames = []
    for season in seasons:
        try:
            df = fetch_projections(season)
        except requests.HTTPError:
            continue          # 2017 and earlier; not an error worth raising
        if not df.empty:
            frames.append(df)
        time.sleep(pause)
    if not frames:
        raise ValueError("no seasons returned projections - upstream change?")
    return pd.concat(frames, ignore_index=True)
