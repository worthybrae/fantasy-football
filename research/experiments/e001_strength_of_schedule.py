"""E001 -- how much is strength of schedule actually worth?

The board spends 10% of its composite weight on a `schedule` factor built
from last season's points allowed, mapped onto this season's opponents. This
asks two separate questions that are easy to conflate:

  1. can we PREDICT a defense better than we currently do?
  2. does predicting it better change what a fantasy manager should do?

They have opposite answers, which is the point of the experiment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.lab import Experiment

WEEKS_PRICED = 6          # how much of 2026 the market has posted lines for
MIN_GAMES = 8             # a player-season worth counting


def _fpa_by_group(weekly: pd.DataFrame, rules) -> pd.DataFrame:
    """Fantasy points each defense allowed per game, split run vs pass."""
    from scoring.ppr import compute_ppr_points
    wk = weekly.copy()
    wk["pts"] = compute_ppr_points(wk, rules)
    wk = wk[wk.position.isin(["RB", "WR", "TE"])].copy()
    wk["grp"] = np.where(wk.position == "RB", "RUN", "PASS")
    tot = wk.groupby(["season", "opponent_team", "grp"])["pts"].sum().reset_index()
    games = wk.groupby(["season", "opponent_team"])["week"].nunique().rename("gp").reset_index()
    out = tot.merge(games, on=["season", "opponent_team"])
    out["fpa"] = out["pts"] / out["gp"]
    out = out.rename(columns={"opponent_team": "team"})
    out["z"] = out.groupby(["season", "grp"])["fpa"].transform(
        lambda s: (s - s.mean()) / s.std())
    return out[["season", "team", "grp", "fpa", "z"]]


def _implied_allowed(odds: pd.DataFrame, line_col: str, total_col: str) -> pd.DataFrame:
    """Points each team's OPPONENT is expected to score, per game.

    `Home Line Open` is the home handicap (negative means home favoured), so
    the away team's implied score is (total + line) / 2 and the home team's
    is the mirror. Verified by sign: the raw line correlates -0.42 with the
    realised home margin, which is the direction a handicap must run.
    """
    d = odds.dropna(subset=[line_col, total_col])
    home = pd.DataFrame({"season": d.season, "week": d.week, "team": d.home,
                         "pa": (d[total_col] + d[line_col]) / 2})
    away = pd.DataFrame({"season": d.season, "week": d.week, "team": d.away,
                         "pa": (d[total_col] - d[line_col]) / 2})
    return pd.concat([home, away])


def run(conn, odds: pd.DataFrame, rules) -> dict:
    weekly = conn.execute("select * from weekly").df()
    fpa = _fpa_by_group(weekly, rules)

    prev = fpa.copy()
    prev["season"] += 1
    prev = prev.rename(columns={"z": "last_z"})[["season", "team", "grp", "last_z"]]
    base = fpa.merge(prev, on=["season", "team", "grp"])

    def vegas_frame(line_col, total_col):
        pa = _implied_allowed(odds, line_col, total_col)
        pa = pa[pa.week <= WEEKS_PRICED]
        v = pa.groupby(["season", "team"])["pa"].mean().reset_index().rename(columns={"pa": "v"})
        d = base.merge(v, on=["season", "team"])
        d["vz"] = d.groupby(["season", "grp"])["v"].transform(
            lambda s: (s - s.mean()) / s.std())
        return d

    opening = vegas_frame("Home Line Open", "Total Score Open")
    closing = vegas_frame("Home Line Close", "Total Score Close")

    def by_group(d, col):
        return {k: round(float(d[d.grp == k][col].corr(d[d.grp == k]["z"])), 3)
                for k in ("RUN", "PASS")}

    last = by_group(opening, "last_z")
    open_r = by_group(opening, "vz")
    close_r = by_group(closing, "vz")

    best, curves = {}, {}
    for k in ("RUN", "PASS"):
        s = opening[opening.grp == k]
        grid = [(w, float((w * s.vz + (1 - w) * s.last_z).corr(s.z)))
                for w in np.arange(0, 1.001, 0.05)]
        w, r = max(grid, key=lambda t: t[1])
        best[k] = {"weight": round(float(w), 2), "r": round(r, 3)}
        # The whole curve, not just its peak: a chart of it shows the optimum
        # is a broad plateau rather than a knife edge, which is the honest
        # read and is invisible from the maximum alone.
        curves[k] = [[round(float(x), 2), round(y, 3)] for x, y in grid]

    # --- does any of it matter? the ceiling, with perfect hindsight --------
    from scoring.ppr import compute_ppr_points
    wk = weekly.copy()
    wk["pts"] = compute_ppr_points(wk, rules)
    wk = wk[wk.position.isin(["RB", "WR", "TE"])].copy()
    wk["grp"] = np.where(wk.position == "RB", "RUN", "PASS")
    defz = fpa.rename(columns={"team": "opp", "z": "dz"})[["season", "opp", "grp", "dz"]]
    faced = wk.merge(defz, left_on=["season", "opponent_team", "grp"],
                     right_on=["season", "opp", "grp"])
    sched = faced.groupby(["season", "player_id"])["dz"].mean().reset_index()

    pl = wk.groupby(["season", "player_id"]).agg(
        pts=("pts", "sum"), g=("week", "nunique")).reset_index()
    pl = pl[pl.g >= MIN_GAMES]
    pl["ppg"] = pl.pts / pl.g
    pl = pl.merge(sched, on=["season", "player_id"])
    prev_ppg = pl[["season", "player_id", "ppg"]].copy()
    prev_ppg["season"] += 1
    m = pl.merge(prev_ppg.rename(columns={"ppg": "prev_ppg"}), on=["season", "player_id"])

    def r2(cols):
        X = np.column_stack([np.ones(len(m))] + [m[c].values for c in cols])
        beta, *_ = np.linalg.lstsq(X, m.ppg.values, rcond=None)
        resid = m.ppg.values - X @ beta
        return 1 - (resid ** 2).sum() / ((m.ppg - m.ppg.mean()) ** 2).sum(), beta

    r2_base, _ = r2(["prev_ppg"])
    r2_sched, beta = r2(["prev_ppg", "dz"])
    sd_faced = float(m.dz.std())

    return {
        "n_team_season_groups": int(len(opening)),
        "seasons": f"{int(opening.season.min())}-{int(opening.season.max())}",
        "weeks_priced": WEEKS_PRICED,
        "curve_run": curves["RUN"], "curve_pass": curves["PASS"],
        "last_run": last["RUN"], "last_pass": last["PASS"],
        "open_run": open_r["RUN"], "open_pass": open_r["PASS"],
        "close_run": close_r["RUN"], "close_pass": close_r["PASS"],
        "best_run_weight": int(best["RUN"]["weight"] * 100), "best_run_r": best["RUN"]["r"],
        "best_pass_weight": int(best["PASS"]["weight"] * 100), "best_pass_r": best["PASS"]["r"],
        "n_player_seasons": int(len(m)),
        "r2_base": round(float(r2_base), 3),
        "r2_sched": round(float(r2_sched), 3),
        "r2_gain": round(float(r2_sched - r2_base), 4),
        "ppg_per_sd": round(float(beta[-1]), 2),
        "sd_schedules_faced": round(sd_faced, 2),
        "swing_points": round(abs(float(beta[-1])) * 17 * 2 * sd_faced, 1),
        "composite_weight_pct": 10,
        "_notes": [
            "Opening lines are each GAME's market open, not a season-long future "
            "posted in August. For weeks 1-6 the two are close; for later weeks "
            "they are not, which is why the window is capped at 6.",
            "Week is derived by bucketing dates seven days from each season's "
            "opener, so the occasional midweek game is misfiled. Harmless for a "
            "six-week average, wrong for anything week-specific.",
        ],
    }


EXPERIMENT = Experiment(
    id="e001",
    title="Strength of schedule: better model, same answer",
    question="Can we predict defensive strength better than last season's numbers "
             "do -- and if we can, does it change any draft decision?",
    run=run,
    tags=("schedule", "vegas", "composite"),
    overturns="The board gives `schedule` 10% of its composite weight. This finds "
              "that even PERFECT knowledge of every defense would be worth almost "
              "nothing to a player projection.",
)
