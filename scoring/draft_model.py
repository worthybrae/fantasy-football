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

import numpy as np
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
        # historic_adp has no uniqueness guarantee on (norm, position) --
        # parse_adp builds it straight from the external payload, and
        # import_league writes it without deduping. Collapse to the
        # best-known (lowest) adp_rank per player before anything else, so
        # "one pick removes exactly one pool row" holds structurally rather
        # than by accident of the fixture. Same precedent as
        # scoring.board._dedupe_adp.
        pool = pool.sort_values("adp_rank").drop_duplicates(
            ["norm", "position"], keep="first").reset_index(drop=True)
        available = pool.copy()
        rosters, recent = {}, []
        for _, pick in season_picks.sort_values("overall_pick").iterrows():
            match = available.index[(available["norm"] == pick["norm"])
                                    & (available["position"] == pick["position"])]
            if len(match):
                reset = available.reset_index(drop=True)
                chosen = int(reset.index[(reset["norm"] == pick["norm"])
                                         & (reset["position"] == pick["position"])][0])
                out.append(PickObservation(
                    season=int(season), overall_pick=int(pick["overall_pick"]),
                    manager=pick["manager"], chosen=chosen, pool=reset,
                    roster=dict(rosters.get(pick["manager"], {})),
                    recent=list(recent[:RUN_WINDOW])))
                # Belt and braces: the pool is deduped above so `match`
                # should never carry more than one label, but drop only the
                # first to guarantee the pool never shrinks by more than one
                # row for a single pick even if that invariant is ever
                # violated upstream.
                available = available.drop(index=match[:1])
            # Bookkeeping runs unconditionally, matched or not: a pick with
            # no ADP row that season produces no observation, but it still
            # consumed a roster spot and still counts toward positional runs.
            rosters.setdefault(pick["manager"], {})
            rosters[pick["manager"]][pick["position"]] = (
                rosters[pick["manager"]].get(pick["position"], 0) + 1)
            recent.insert(0, pick["position"])
    return out


# QB is the dropped baseline: with a full set of position dummies plus an
# intercept-free softmax the columns would be collinear.
_POSITION_DUMMIES = ["RB", "WR", "TE", "K", "DST"]
FEATURE_NAMES = (["reach", "fall"]
                 + [f"pos_{p}" for p in _POSITION_DUMMIES]
                 + ["qb_early", "te_early", "need", "run"])
EARLY_ROUNDS = 3


def feature_matrix(obs: PickObservation, settings) -> np.ndarray:
    pool = obs.pool
    n = len(pool)
    ranks = pool["adp_rank"].to_numpy(dtype=float)
    positions = pool["position"].to_numpy()
    teams = max(settings.teams, 1)

    delta = ranks - obs.overall_pick
    reach = np.maximum(0.0, delta) / teams
    fall = np.maximum(0.0, -delta) / teams

    columns = [reach, fall]
    for pos in _POSITION_DUMMIES:
        columns.append((positions == pos).astype(float))

    # `overall_pick` is 1-indexed, so pick 8 in an 8-team league is round 1.
    round_no = (obs.overall_pick - 1) // teams + 1
    early = 1.0 if round_no <= EARLY_ROUNDS else 0.0
    columns.append((positions == "QB").astype(float) * early)
    columns.append((positions == "TE").astype(float) * early)

    starters = settings.starters
    need = np.array([1.0 if obs.roster.get(p, 0) < starters.get(p, 0) else 0.0
                     for p in positions])
    columns.append(need)

    recent = obs.recent[:RUN_WINDOW]
    run = np.array([recent.count(p) / RUN_WINDOW for p in positions])
    columns.append(run)

    return np.column_stack(columns) if n else np.zeros((0, len(FEATURE_NAMES)))
