"""Per-manager draft-pick model.

Each historical pick is treated as one choice from the set of players who were
available at that moment. That set is reconstructable: we know the full pick
order and that season's ADP pool, so subtracting everything taken before pick
N gives the pool the manager was actually choosing from.

Picks whose player has no ADP row that season are dropped from fitting. The
model's core feature is a player's position relative to market rank, which is
undefined without one -- and the import step reports how many picks this
removes per season so a bad join surfaces as a number, not a silent shrug.
"""
from typing import NamedTuple

import pandas as pd

from pipeline.db import read_table
from scoring.board import _norm_name

RUN_WINDOW = 5


class PickObservation(NamedTuple):
    season: int
    overall_pick: int
    manager: str
    chosen: int
    pool: pd.DataFrame
    roster: dict
    recent: list


def build_observations(conn) -> list:
    picks = read_table(conn, "draft_picks")
    teams = read_table(conn, "draft_teams")
    adp = read_table(conn, "historic_adp")
    if picks.empty or teams.empty or adp.empty:
        return []

    picks = picks.merge(teams[["season", "team_id", "manager"]],
                        on=["season", "team_id"], how="left")
    picks = picks.assign(norm=picks["player_name"].map(_norm_name))
    adp = adp.assign(norm=adp["adp_name"].map(_norm_name))

    out = []
    for season, season_picks in picks.groupby("season"):
        pool = adp[adp["season"] == season][["norm", "position", "adp_rank"]]
        pool = pool.sort_values("adp_rank").reset_index(drop=True)
        available = pool.copy()
        rosters, recent = {}, []
        for _, pick in season_picks.sort_values("overall_pick").iterrows():
            match = available.index[(available["norm"] == pick["norm"])
                                    & (available["position"] == pick["position"])]
            if len(match) == 0:
                # No ADP row for this player that season -- unusable as an
                # observation, but the pick still consumed a roster spot and
                # still counts toward positional runs.
                rosters.setdefault(pick["manager"], {})
                rosters[pick["manager"]][pick["position"]] = (
                    rosters[pick["manager"]].get(pick["position"], 0) + 1)
                recent.insert(0, pick["position"])
                continue
            reset = available.reset_index(drop=True)
            chosen = int(reset.index[(reset["norm"] == pick["norm"])
                                     & (reset["position"] == pick["position"])][0])
            out.append(PickObservation(
                season=int(season), overall_pick=int(pick["overall_pick"]),
                manager=pick["manager"], chosen=chosen, pool=reset,
                roster=dict(rosters.get(pick["manager"], {})),
                recent=list(recent[:RUN_WINDOW])))
            available = available.drop(index=match)
            rosters.setdefault(pick["manager"], {})
            rosters[pick["manager"]][pick["position"]] = (
                rosters[pick["manager"]].get(pick["position"], 0) + 1)
            recent.insert(0, pick["position"])
    return out
