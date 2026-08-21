"""E004 -- schedule strength in the fantasy playoffs, weeks 15 to 17.

E001 found season-long strength of schedule worth almost nothing, and said
exactly why: a player faces seventeen defenses, and the average of seventeen
draws has almost no spread left. That is an argument about arithmetic, not
about football, and it has an obvious corollary -- shrink the window and the
spread should come back.

Fantasy seasons are decided in weeks 15, 16 and 17. Three draws, not
seventeen. So this asks the same three questions E001 asked, restricted to
that window:

  1. does the spread survive three games?
  2. what is it worth in points?
  3. can any of it be seen in August, when the draft happens?

Only the third one is about drafting, and it is the one that decides whether
the answer is actionable or merely true.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.lab import RESULTS, Experiment
from research.experiments.e001_strength_of_schedule import (MIN_GAMES,
                                                            WEEKS_PRICED,
                                                            _fpa_by_group,
                                                            _implied_allowed)

# nflverse game ids carry the code a franchise used AT THE TIME, while every
# other column carries today's. Only two differ inside this database's range.
RELOCATED = {"OAK": "LV", "SD": "LAC"}

PLAYOFF_WEEKS = (15, 16, 17)   # the modern fantasy playoff window
LEGACY_WEEKS = (14, 15, 16)    # what most leagues used before the 17-game era
MIN_PLAYOFF_GAMES = 2          # 2 of the 3 -- a one-game sample is not a window
POSITIONS = ("RB", "WR", "TE")


def _true_weeks(conn, odds: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Re-key the odds workbook to real NFL weeks, from nflverse game ids.

    `data.odds()` derives week by bucketing dates seven days from the
    season opener, which E001 records as harmless for a six-week average and
    wrong for anything week-specific. This experiment IS week-specific --
    weeks 15 to 17 contain Saturday and Christmas games that bucket into the
    wrong slot -- so the week is taken from `weekly.game_id`
    (`SEASON_WW_AWAY_HOME`) instead, which carries the league's own numbering.
    Ids spell a franchise with its code of the day, so the Raiders' and
    Chargers' pre-move seasons are mapped forward before the join -- left
    alone they silently drop 64 regular-season games from 2016 to 2019.

    Playoff games are dropped here as well, and NOT because `data.odds()`
    forgot to: it tries, with `df["Playoff Game?"] != 1`, but the workbook
    writes that column as the string "Y", so the test passes every row and
    all 232 playoff games survive. E001 never notices because it keeps only
    weeks 1-6. This experiment would: a playoff REMATCH shares
    (season, home, away) with its regular-season meeting, so it would merge
    onto that game's week and double-count it. Filtering on the real flag
    leaves exactly nflverse's regular-season game count in every season.
    """
    ids = conn.execute(
        "select distinct game_id from weekly where season_type = 'REG'").df()
    parts = ids.game_id.str.split("_", expand=True)
    games = pd.DataFrame({"season": parts[0].astype(int),
                          "week": parts[1].astype(int),
                          "away": parts[2].replace(RELOCATED),
                          "home": parts[3].replace(RELOCATED)})
    reg = odds[odds["Playoff Game?"].isna()]
    keyed = reg.drop(columns=["week"]).merge(games, on=["season", "home", "away"])
    in_range = reg[reg.season.isin(set(games.season))]
    return keyed, 100.0 * len(keyed) / max(len(in_range), 1)


def _fit(df: pd.DataFrame, y: str, cols: list[str]):
    """Least squares with an intercept; returns (R2, coefficients)."""
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    beta, *_ = np.linalg.lstsq(X, df[y].values, rcond=None)
    resid = df[y].values - X @ beta
    return 1 - (resid ** 2).sum() / ((df[y] - df[y].mean()) ** 2).sum(), beta


def _scored(weekly: pd.DataFrame, rules) -> pd.DataFrame:
    """Skill-position weekly rows with fantasy points and E001's run/pass split."""
    from scoring.ppr import compute_ppr_points
    wk = weekly.copy()
    wk["pts"] = compute_ppr_points(wk, rules)
    wk = wk[wk.position.isin(POSITIONS)].copy()
    wk["grp"] = np.where(wk.position == "RB", "RUN", "PASS")
    return wk


def _window_value(wk: pd.DataFrame, defz: pd.DataFrame, weeks: tuple) -> dict:
    """What a soft window is worth: the regression, for one choice of weeks.

    Run twice -- once for the modern window and once for the pre-2021
    convention -- so the headline number cannot be an artefact of which
    three weeks were called the playoffs.
    """
    win = wk[wk.week.isin(weeks)]
    faced = win.merge(defz, left_on=["season", "opponent_team", "grp"],
                      right_on=["season", "opp", "grp"])
    sched = faced.groupby(["season", "player_id"])["dz"].mean().reset_index()
    sched = sched.rename(columns={"dz": "sched_z"})

    po = win.groupby(["season", "player_id"]).agg(
        pts=("pts", "sum"), g=("week", "nunique")).reset_index()
    po = po[po.g >= MIN_PLAYOFF_GAMES]
    po["ppg_po"] = po.pts / po.g

    full = wk.groupby(["season", "player_id"]).agg(
        pts=("pts", "sum"), g=("week", "nunique")).reset_index()
    full = full[full.g >= MIN_GAMES]
    full["ppg_full"] = full.pts / full.g

    pre = wk[wk.week < min(weeks)].groupby(["season", "player_id"]).agg(
        pts=("pts", "sum"), g=("week", "nunique")).reset_index()
    pre = pre[pre.g >= MIN_GAMES]
    pre["ppg_pre"] = pre.pts / pre.g

    m = (po[["season", "player_id", "ppg_po"]]
         .merge(full[["season", "player_id", "ppg_full"]], on=["season", "player_id"])
         .merge(sched, on=["season", "player_id"]))
    r2_base, _ = _fit(m, "ppg_po", ["ppg_full"])
    r2_sched, beta = _fit(m, "ppg_po", ["ppg_full", "sched_z"])

    mp = m.merge(pre[["season", "player_id", "ppg_pre"]], on=["season", "player_id"])
    r2_base_pre, _ = _fit(mp, "ppg_po", ["ppg_pre"])
    r2_sched_pre, beta_pre = _fit(mp, "ppg_po", ["ppg_pre", "sched_z"])

    # A player's season-long schedule, on the same players, so the spread
    # comparison is between two windows and not between two samples.
    all_faced = wk.merge(defz, left_on=["season", "opponent_team", "grp"],
                         right_on=["season", "opp", "grp"])
    season_sched = all_faced.groupby(["season", "player_id"])["dz"].mean().reset_index()
    season_sched = m[["season", "player_id"]].merge(season_sched, on=["season", "player_id"])

    return {
        "n": int(len(m)), "n_pre": int(len(mp)),
        "seasons": f"{int(m.season.min())}-{int(m.season.max())}",
        "sd_window": float(m.sched_z.std()),
        "sd_season": float(season_sched.dz.std()),
        "mean_ppg": float(m.ppg_po.mean()),
        "r2_base": float(r2_base), "r2_sched": float(r2_sched),
        "ppg_per_sd": float(beta[-1]),
        "r2_base_pre": float(r2_base_pre), "r2_sched_pre": float(r2_sched_pre),
        "ppg_per_sd_pre": float(beta_pre[-1]),
    }


def run(conn, odds: pd.DataFrame, rules) -> dict:
    weekly = conn.execute("select * from weekly where season_type = 'REG'").df()
    wk = _scored(weekly, rules)

    # A defense's quality over the whole season -- the honest measure of the
    # opponent, and the one E001 used. Its own weeks 15-17 points allowed
    # would contain the very points being predicted.
    fpa_season = _fpa_by_group(weekly, rules)
    defz = fpa_season.rename(columns={"team": "opp", "z": "dz"})[
        ["season", "opp", "grp", "dz"]]

    main = _window_value(wk, defz, PLAYOFF_WEEKS)
    legacy = _window_value(wk, defz, LEGACY_WEEKS)

    # -- 1. does the spread survive? ------------------------------------
    # Team level too: a player-level spread is diluted by injuries and
    # by-week starts, and the question is about schedules, not availability.
    def team_spread(weeks):
        f = wk[wk.week.isin(weeks)] if weeks else wk
        pairs = f[["season", "recent_team", "opponent_team", "grp", "week"]].drop_duplicates()
        j = pairs.merge(defz, left_on=["season", "opponent_team", "grp"],
                        right_on=["season", "opp", "grp"])
        g = j.groupby(["season", "recent_team", "grp"])["dz"].mean()
        return float(g.std())

    sd_teams_po = team_spread(PLAYOFF_WEEKS)
    sd_teams_season = team_spread(None)

    # What plain averaging predicts: the mean of n independent draws from a
    # distribution of width 1 has width 1/sqrt(n). Opponents are NOT drawn
    # independently -- divisional games repeat, and there are only 31 other
    # teams to draw from -- so this is a reference point to measure against,
    # not a claim.
    def theory(n_games):
        return float(1 / np.sqrt(n_games))

    team_games = wk[["season", "recent_team", "week"]].drop_duplicates().groupby(
        ["season", "recent_team"])["week"].nunique()
    mean_team_games = float(team_games.mean())

    # -- 3. can it be seen at draft time? -------------------------------
    # Target: how soft each defense actually was in weeks 15-17.
    fpa_po = _fpa_by_group(weekly[weekly.week.isin(PLAYOFF_WEEKS)], rules)
    target = fpa_po[["season", "team", "grp", "z"]].rename(columns={"z": "z_po"})

    prev = fpa_season.copy()
    prev["season"] += 1
    prev = prev.rename(columns={"z": "last_z"})[["season", "team", "grp", "last_z"]]
    same = fpa_season.rename(columns={"z": "same_z"})[["season", "team", "grp", "same_z"]]
    pred = target.merge(prev, on=["season", "team", "grp"]).merge(
        same, on=["season", "team", "grp"])

    keyed_odds, odds_matched = _true_weeks(conn, odds)

    def vegas(weeks, name):
        pa = _implied_allowed(keyed_odds, "Home Line Open", "Total Score Open")
        pa = pa[pa.week.isin(set(weeks))]
        v = pa.groupby(["season", "team"])["pa"].mean().reset_index()
        v = v.rename(columns={"pa": name})
        return v

    pred = pred.merge(vegas(range(1, WEEKS_PRICED + 1), "open6"), on=["season", "team"])
    pred = pred.merge(vegas(PLAYOFF_WEEKS, "openpo"), on=["season", "team"])
    for col in ("open6", "openpo"):
        pred[col + "_z"] = pred.groupby(["season", "grp"])[col].transform(
            lambda s: (s - s.mean()) / s.std())

    def by_group(col):
        return {k: round(float(pred[pred.grp == k][col].corr(
            pred[pred.grp == k]["z_po"])), 3) for k in ("RUN", "PASS")}

    last, same_r = by_group("last_z"), by_group("same_z")
    open6, openpo = by_group("open6_z"), by_group("openpo_z")

    e001 = json.loads((RESULTS / "e001.json").read_text())["findings"]

    swing = abs(main["ppg_per_sd"]) * len(PLAYOFF_WEEKS) * 2 * main["sd_window"]

    return {
        # sample
        "seasons": main["seasons"],
        "seasons_pred": f"{int(pred.season.min())}-{int(pred.season.max())}",
        "weeks_label": "15-17",
        "pre_weeks_label": f"1-{min(PLAYOFF_WEEKS) - 1}",
        "window_weeks": len(PLAYOFF_WEEKS),
        "min_playoff_games": MIN_PLAYOFF_GAMES,
        "min_games": MIN_GAMES,
        "n_player_seasons": main["n"],
        "n_player_seasons_pre": main["n_pre"],
        "n_team_season_groups": int(len(pred)),
        "odds_matched_pct": round(odds_matched, 1),
        "weeks_priced": WEEKS_PRICED,

        # 1. the spread
        "sd_one_defense": 1.0,
        "sd_season_faced": round(main["sd_season"], 2),
        "sd_playoff_faced": round(main["sd_window"], 2),
        "sd_season_teams": round(sd_teams_season, 2),
        "sd_playoff_teams": round(sd_teams_po, 2),
        "mean_games_season": round(mean_team_games, 1),
        "theory_season": round(theory(mean_team_games), 2),
        "theory_playoff": round(theory(len(PLAYOFF_WEEKS)), 2),
        "spread_ratio": round(main["sd_window"] / main["sd_season"], 1),

        # 2. what it is worth
        "r2_base": round(main["r2_base"], 3),
        "r2_sched": round(main["r2_sched"], 3),
        "r2_gain": round(main["r2_sched"] - main["r2_base"], 4),
        "ppg_per_sd": round(main["ppg_per_sd"], 2),
        "r2_base_pre": round(main["r2_base_pre"], 3),
        "r2_sched_pre": round(main["r2_sched_pre"], 3),
        "r2_gain_pre": round(main["r2_sched_pre"] - main["r2_base_pre"], 4),
        "ppg_per_sd_pre": round(main["ppg_per_sd_pre"], 2),
        "swing_points": round(swing, 1),
        "swing_pct_of_run": round(100 * swing / (main["mean_ppg"] * 3), 1),
        "mean_ppg_playoffs": round(main["mean_ppg"], 1),
        "legacy_weeks_label": "14-16",
        "legacy_sd": round(legacy["sd_window"], 2),
        "legacy_ppg_per_sd": round(legacy["ppg_per_sd"], 2),
        "legacy_r2_gain": round(legacy["r2_sched"] - legacy["r2_base"], 4),

        # 3. can it be seen in August
        "last_run": last["RUN"], "last_pass": last["PASS"],
        "open6_run": open6["RUN"], "open6_pass": open6["PASS"],
        "openpo_run": openpo["RUN"], "openpo_pass": openpo["PASS"],
        "same_run": same_r["RUN"], "same_pass": same_r["PASS"],

        # 4. E001, copied across so this post can compare against it
        "e001_last_run": e001["last_run"], "e001_last_pass": e001["last_pass"],
        "e001_open_run": e001["open_run"], "e001_open_pass": e001["open_pass"],
        "e001_sd_season": e001["sd_schedules_faced"],
        "e001_r2_gain": e001["r2_gain"],
        "e001_ppg_per_sd": e001["ppg_per_sd"],
        "e001_swing_points": e001["swing_points"],
        "composite_weight_pct": e001["composite_weight_pct"],

        "_notes": [
            "Weeks 15-17 is the modern fantasy playoff window. Before the "
            "17-game season most leagues played 14-16; that window is measured "
            "too and answers the same, so the finding is not an artefact of "
            "which three weeks are called the playoffs.",
            "The odds workbook's week is re-derived from nflverse game ids "
            "rather than by bucketing dates seven days from the opener, "
            "because Saturday and Christmas games in this window bucket into "
            "the wrong slot. E001's caveat does not apply here.",
            "data.odds() silently kept playoff games when this experiment "
            "was first run: its filter compared the Playoff Game? column "
            "against 1 while the workbook stores the string Y, so none of "
            "the 232 playoff rows were dropped. Found here and fixed in "
            "data.py; E001 was re-run afterwards and none of its findings "
            "moved, because its six-week window already excluded January.",
            "Opening lines for weeks 15-17 are each game's market open in "
            "December, NOT anything available in August. They are measured as "
            "an upper bound on what a market can know, and are labelled that "
            "way in the post. Only last season's defensive numbers, and the "
            "weeks 1-6 lines E001 uses as its August proxy, are draft-time "
            "inputs.",
            "Defensive quality is a defense's whole-season points allowed, not "
            "its weeks 15-17 points allowed: the latter contains the very "
            "points the regression predicts.",
        ],
    }


EXPERIMENT = Experiment(
    id="e004",
    title="Playoff-weeks schedule: the spread survives, the forecast does not",
    question="Season-long schedule averages out to nothing. Weeks 15 to 17 are "
             "only three games and they decide the league -- is playoff-weeks "
             "schedule strength worth acting on at draft time?",
    run=run,
    tags=("schedule", "playoffs", "vegas", "composite"),
    # `overturns` is deliberately None. The board's schedule factor is
    # season-long and worth 10% of the composite; this finding says a
    # playoff-weeks version of it would not be better, which is agreement
    # with what the project already does rather than a contradiction of it.
)
