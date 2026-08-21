"""E005 -- can we project the stat line better than ESPN does?

Every previous experiment here asked whether some input helps predict fantasy
POINTS, and every one of them found the input mostly does not reach the box
score. This asks a sharper question, and the right one for a projection
engine: ESPN publishes projected carries, rushing yards, targets, receptions
and receiving yards. How good are those, per stat, and does the dumbest
possible alternative -- what the player did last season -- already beat them
anywhere?

That second half is what makes this worth running. If a naive baseline
already matches ESPN on a stat, there is room to build. If it does not, then
ESPN's projection is the better free input and our effort belongs elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.lab import Experiment

# ESPN's projected split -> the actual weekly column it should be compared to.
PAIRS = {
    "proj_carries": "carries",
    "proj_rush_yards": "rushing_yards",
    "proj_targets": "targets",
    "proj_receptions": "receptions",
    "proj_rec_yards": "receiving_yards",
}
LABEL = {
    "proj_carries": "carries", "proj_rush_yards": "rushing yards",
    "proj_targets": "targets", "proj_receptions": "receptions",
    "proj_rec_yards": "receiving yards",
}
# A projection of ~zero carries for a receiver is not a forecast anyone is
# testing; it is a position fact. Compare only where the player actually did
# something, so the correlations measure forecasting rather than roster shape.
MIN_ACTUAL = 20
MIN_GAMES = 6
# 2023 answers with 89 usable projections against ~490 in a normal year, so
# it is excluded rather than allowed to speak for a whole season.
SPARSE_SEASONS = (2023,)


def run(conn, odds, rules) -> dict:
    from research.data import espn_crosswalk

    proj = conn.execute("select * from espn_projections").df()
    proj["espn_id"] = pd.to_numeric(proj["espn_id"], errors="coerce")
    proj = proj.merge(espn_crosswalk(), on="espn_id", how="inner")
    proj = proj[~proj.season.isin(SPARSE_SEASONS)]

    wk = conn.execute("select * from weekly").df()
    actual = wk.groupby(["season", "player_id", "position"]).agg(
        carries=("carries", "sum"), rushing_yards=("rushing_yards", "sum"),
        targets=("targets", "sum"), receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        g=("week", "nunique")).reset_index()
    actual = actual[actual.g >= MIN_GAMES]

    df = proj.merge(actual, left_on=["season", "gsis_id"],
                    right_on=["season", "player_id"], how="inner",
                    suffixes=("", "_a"))

    # The naive alternative: what the player did the season before. Not a
    # model -- the floor any model has to clear to be worth building.
    prior = actual.copy()
    prior["season"] += 1
    prior = prior.rename(columns={c: f"prev_{c}" for c in PAIRS.values()})
    df = df.merge(prior[["season", "player_id", *[f"prev_{c}" for c in PAIRS.values()]]],
                  on=["season", "player_id"], how="left")

    out: dict = {
        "seasons": f"{int(df.season.min())}-{int(df.season.max())}",
        "n_rows": int(len(df)),
        "min_actual": MIN_ACTUAL,
        "sparse_season": SPARSE_SEASONS[0],
    }

    wins = []
    for pcol, acol in PAIRS.items():
        sub = df.dropna(subset=[pcol, acol])
        sub = sub[sub[acol] >= MIN_ACTUAL]
        key = acol.replace("_yards", "_yds")
        out[f"n_{key}"] = int(len(sub))
        out[f"espn_{key}"] = round(float(sub[pcol].corr(sub[acol])), 3)

        both = sub.dropna(subset=[f"prev_{acol}"])
        out[f"last_{key}"] = round(float(both[f"prev_{acol}"].corr(both[acol])), 3)
        out[f"n_both_{key}"] = int(len(both))
        # Head to head on identical rows, so neither is flattered by an
        # easier sample: a player with no prior season cannot be scored by
        # the baseline at all, and ESPN finds rookies hardest.
        out[f"espn_h2h_{key}"] = round(float(both[pcol].corr(both[acol])), 3)
        gap = out[f"espn_h2h_{key}"] - out[f"last_{key}"]
        out[f"gap_{key}"] = round(gap, 3)
        wins.append((LABEL[pcol], gap))

    closest = min(wins, key=lambda t: t[1])
    widest = max(wins, key=lambda t: t[1])
    out["closest_stat"] = closest[0]
    out["closest_gap"] = round(closest[1], 3)
    out["widest_stat"] = widest[0]
    out["widest_gap"] = round(widest[1], 3)
    out["n_beaten"] = int(sum(1 for _, g in wins if g <= 0))

    # Rookies: the one place a baseline has nothing to say at all, and so the
    # one place ESPN's projection is not merely better but the only option.
    rookie = df[df[[f"prev_{c}" for c in PAIRS.values()]].isna().all(axis=1)]
    r_sub = rookie.dropna(subset=["proj_rec_yards", "receiving_yards"])
    r_sub = r_sub[r_sub.receiving_yards >= MIN_ACTUAL]
    out["n_rookie_rows"] = int(len(r_sub))
    out["espn_rookie_rec_yds"] = (round(float(r_sub.proj_rec_yards.corr(
        r_sub.receiving_yards)), 3) if len(r_sub) > 30 else None)

    out["_notes"] = [
        f"Season {SPARSE_SEASONS[0]} is excluded: ESPN returns 89 usable "
        f"projections for it against roughly 490 in a normal year, and a "
        f"season that thin would speak for players it never covered.",
        "These are ESPN's PRESEASON projections, confirmed rather than "
        "assumed: against realised outcomes they correlate 0.66-0.70 on "
        "fantasy points at 70-117% mean absolute percentage error. A "
        "projection contaminated by in-season results would sit above 0.95.",
        f"Rows are filtered to an actual of at least {MIN_ACTUAL}, because a "
        f"projected zero carries for a receiver is a position fact rather "
        f"than a forecast, and including those inflates every correlation.",
        "The baseline is last season's raw total with no adjustment for "
        "games missed, age or a change of team. It is a floor to clear, not "
        "a rival model, and a real projection should beat it comfortably.",
        "ESPN's id joins via nflverse's id table, not this database's "
        "`sleeper_ids`, which covers only 22-39% of recent seasons.",
    ]
    return out


EXPERIMENT = Experiment(
    id="e005",
    title="Where ESPN's projections are weakest",
    question="ESPN publishes projected carries, yards, targets and receptions. "
             "How accurate is each one, and does simply repeating last season "
             "already match it anywhere?",
    run=run,
    tags=("projections", "espn", "benchmark"),
    overturns="The project treats ESPN's projection as a single number to be "
              "consumed. It is five separate forecasts of very different "
              "quality, and the weakest are not the ones you would guess.",
)
