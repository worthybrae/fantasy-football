"""E006 -- which owner tendencies are worth modelling, and on which axis?

An earlier pass measured that how far off ADP a manager picks carries from one
season to the next (r = +0.43 over 39 owner-season pairs) while which
POSITIONS he favours does not (-0.06 RB, -0.21 WR). That was taken as licence
to model reach personally and pool the rest.

It answered half the question. "Does last year's draft predict this year's"
and "does what he has done so far TODAY predict his next pick" are different
measurements, and a live draft assistant needs both: the first decides whether
a prior from history is worth carrying, the second decides whether the model
should update as picks land. A trait can pass one and fail the other. Position
is the obvious case -- dead as a prior, yet obviously live inside a draft,
because nobody drafts a third quarterback.

So this measures every trait on BOTH axes:

  * across-season -- manager's mean trait in season t against season t+1
  * within-draft  -- his early-round mean against his late-round mean, in the
    same draft, after removing the round-bucket average so that "everyone
    reaches more in round 12" is not read as a personal trait

Four traits, two of them new and untested:

  reach       how far from market rank he picks. The known positive.
  growth      does he buy the projection or the track record -- ESPN's
              projected ppg for the player that season minus what the player
              had actually been averaging, as of that draft.
  steadiness  does he buy weekly reliability -- the drafted player's
              coefficient of variation over the seasons before the draft.
  pos_share   share of picks spent on RB and on WR. The known NEGATIVE, kept
              in deliberately as a control: a method that reports position as
              persistent across seasons has disagreed with a measurement we
              already trust, and should be disbelieved before its new numbers
              are.

Everything is computed AS OF the draft. A 2021 pick is scored with the
projection published for 2021 and the actuals available before it, never with
what the player went on to do -- that is the difference between measuring a
manager's taste and measuring his luck.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.db import read_table
from research.data import espn_crosswalk
from research.lab import Experiment
from scoring.board import _ADP_POSITION_ALIASES, _norm_name
from scoring.config import RECENCY_WEIGHTS
from scoring.draft_model import _match_keys
from scoring.ppr import compute_ppr_points, normalize_rules

# A season needs enough games before it says anything about a player's rate or
# his steadiness. Same bar the board's `consistency` uses.
MIN_GAMES = 8
# How many prior seasons feed an as-of rate. RECENCY_WEIGHTS is a fixed map of
# absolute years, useless for a 2021 draft, so only its SHAPE is borrowed:
# most recent season weighted most.
LOOKBACK = 3
PRIOR_WEIGHTS = (0.5, 0.3, 0.2)
# Rounds 1-5 are where a manager is choosing between players rather than
# filling out a bench, and where the earlier pass measured its +0.43. The
# within-draft split runs over the WHOLE draft, because the question there is
# whether early behaviour predicts late behaviour.
EARLY_ROUNDS = 5
# The split point for the within-draft measure. Rounds 1-4 against 5+: an even
# split of a 15-round draft would put the whole split inside the bench.
SPLIT_ROUND = 5
MIN_PICKS_PER_HALF = 3


def _weighted(values: pd.Series, seasons: pd.Series, draft_season: int) -> float:
    """Recency-weighted mean of per-season values from before `draft_season`."""
    age = draft_season - seasons
    w = pd.Series(age).map(
        {i + 1: PRIOR_WEIGHTS[i] for i in range(LOOKBACK)}).astype(float)
    ok = w.notna() & values.notna()
    if not ok.any() or w[ok].sum() <= 0:
        return np.nan
    return float((values[ok] * w[ok]).sum() / w[ok].sum())


def _player_season_rates(weekly: pd.DataFrame, rules) -> pd.DataFrame:
    """Points per game and coefficient of variation, per player-season."""
    wk = weekly.copy()
    wk["_pts"] = compute_ppr_points(wk, normalize_rules(rules))
    per = wk.groupby(["player_id", "season"]).agg(
        pts=("_pts", "sum"), mean=("_pts", "mean"),
        sd=("_pts", "std"), games=("week", "nunique")).reset_index()
    per = per[per["games"] >= MIN_GAMES]
    per["ppg"] = per["pts"] / per["games"]
    per["cv"] = (per["sd"] / per["mean"].where(per["mean"] > 0))
    return per[["player_id", "season", "ppg", "cv"]]


def _pick_frame(conn, rules=None) -> pd.DataFrame:
    """One row per historical pick, with every trait scored as of that draft."""
    picks = read_table(conn, "draft_picks")
    teams = read_table(conn, "draft_teams")
    adp = read_table(conn, "historic_adp")
    weekly = read_table(conn, "weekly")
    proj = read_table(conn, "espn_projections")
    if picks.empty or teams.empty or adp.empty:
        return pd.DataFrame()

    picks = picks.merge(teams[["season", "team_id", "manager"]],
                        on=["season", "team_id"], how="left")

    # -- reach: market rank as of that season, same join the model uses ------
    adp = adp.assign(norm=adp["adp_name"].map(_norm_name),
                     position=adp["position"].replace(_ADP_POSITION_ALIASES))
    adp = adp.assign(key=_match_keys(adp, "adp_name"))
    adp = adp[adp["key"].notna()].sort_values("adp_rank").drop_duplicates(
        ["season", "key"], keep="first")
    picks = picks.assign(key=_match_keys(picks, "player_name", "nfl_team"))
    picks = picks.merge(adp[["season", "key", "adp_rank"]],
                        on=["season", "key"], how="left")
    # POSITIVE means he took the player EARLIER than the market: a reach.
    picks["reach"] = picks["adp_rank"] - picks["overall_pick"]

    # -- as-of player rates ---------------------------------------------------
    rates = _player_season_rates(weekly, rules)
    xwalk = espn_crosswalk().rename(columns={"gsis_id": "player_id"})
    picks = picks.merge(xwalk, left_on="espn_player_id", right_on="espn_id",
                        how="left")

    proj = proj.merge(xwalk, on="espn_id", how="inner")
    proj = proj[proj["proj_games"] > 0].copy()
    proj["proj_ppg"] = proj["proj_points"] / proj["proj_games"]
    proj = proj.drop_duplicates(["player_id", "season"])

    growth, steady = [], []
    for _, p in picks.iterrows():
        pid, season = p.get("player_id"), int(p["season"])
        if not isinstance(pid, str):
            growth.append(np.nan); steady.append(np.nan); continue
        hist = rates[(rates["player_id"] == pid) & (rates["season"] < season)]
        prior_ppg = _weighted(hist["ppg"], hist["season"], season)
        steady.append(_weighted(hist["cv"], hist["season"], season))
        row = proj[(proj["player_id"] == pid) & (proj["season"] == season)]
        if row.empty or np.isnan(prior_ppg):
            growth.append(np.nan)
        else:
            growth.append(float(row["proj_ppg"].iloc[0]) - prior_ppg)
    picks["growth"] = growth
    # NEGATED so that, like every other trait here, a bigger number is more of
    # the thing the trait is named for. A low coefficient is a steady player.
    picks["steadiness"] = [-s if s == s else np.nan for s in steady]
    return picks


def _round_of(picks: pd.DataFrame) -> pd.Series:
    return picks["round"].astype(float)


def _p_value(r: float, n: int) -> float:
    """Two-sided p for a Pearson r, so a small sample cannot masquerade.

    At n = 40 anything under about |0.31| fails to clear p = .05. Without this
    the difference between "we measured no effect" and "we could not have
    detected one" is invisible, and a -0.22 gets written up as an
    anti-correlation when it is a shrug.
    """
    from scipy import stats
    if not np.isfinite(r) or n < 3 or abs(r) >= 1:
        return np.nan
    t = r * np.sqrt((n - 2) / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), n - 2))


def _across_season(per_season: pd.DataFrame, trait: str) -> tuple:
    """Correlate a manager's trait in season t with the same manager in t+1."""
    a = per_season.dropna(subset=[trait])[["manager", "season", trait]]
    nxt = a.assign(season=a["season"] - 1).rename(columns={trait: "next"})
    pairs = a.merge(nxt[["manager", "season", "next"]], on=["manager", "season"])
    if len(pairs) < 3:
        return np.nan, len(pairs)
    return float(pairs[trait].corr(pairs["next"])), len(pairs)


def _within_draft(picks: pd.DataFrame, trait: str) -> tuple:
    """Early-round mean against late-round mean, same manager, same draft.

    The round-bucket mean is removed first. Reaches grow in the late rounds
    for everyone -- the board thins and the remaining players have no ADP
    worth honouring -- so a raw early/late correlation would report that
    shared drift as a personal trait for all eight managers at once.
    """
    df = picks.dropna(subset=[trait]).copy()
    if df.empty:
        return np.nan, 0
    df["_r"] = _round_of(df)
    df["_dev"] = df[trait] - df.groupby(["season", "_r"])[trait].transform("mean")
    df["_half"] = np.where(df["_r"] < SPLIT_ROUND, "early", "late")
    halves = df.groupby(["manager", "season", "_half"])["_dev"].agg(["mean", "size"])
    halves = halves[halves["size"] >= MIN_PICKS_PER_HALF]["mean"].unstack("_half")
    halves = halves.dropna()
    if len(halves) < 3:
        return np.nan, len(halves)
    return float(halves["early"].corr(halves["late"])), len(halves)


def run(conn=None, odds=None, rules=None) -> dict:
    """`odds` is unused -- run.py hands every experiment the same three."""
    if conn is None:
        from pipeline.db import get_conn
        conn = get_conn()
    picks = _pick_frame(conn, rules)
    if picks.empty:
        raise RuntimeError("no draft history imported -- nothing to measure")

    early = picks[_round_of(picks) <= EARLY_ROUNDS]
    per_season = early.groupby(["manager", "season"]).agg(
        reach=("reach", "mean"), growth=("growth", "mean"),
        steadiness=("steadiness", "mean")).reset_index()
    # Position is a SHARE of picks, not a mean of a per-pick number, so it is
    # built separately rather than bent into the same aggregation.
    for pos in ("RB", "WR"):
        share = (early.assign(_hit=(early["position"] == pos).astype(float))
                 .groupby(["manager", "season"])["_hit"].mean()
                 .rename(f"pos_{pos}").reset_index())
        per_season = per_season.merge(share, on=["manager", "season"], how="left")
    picks["pos_RB"] = (picks["position"] == "RB").astype(float)
    picks["pos_WR"] = (picks["position"] == "WR").astype(float)

    findings = {}
    for trait in ("reach", "growth", "steadiness", "pos_RB", "pos_WR"):
        r_across, n_across = _across_season(per_season, trait)
        r_within, n_within = _within_draft(picks, trait)
        for axis, r, n in (("across", r_across, n_across),
                           ("within", r_within, n_within)):
            pv = _p_value(r, n)
            findings[f"{axis}_{trait}"] = None if np.isnan(r) else round(r, 3)
            findings[f"{axis}_n_{trait}"] = n
            findings[f"{axis}_p_{trait}"] = None if np.isnan(pv) else round(pv, 3)
            findings[f"{axis}_sig_{trait}"] = bool(pv == pv and pv < 0.05)

    findings["n_picks"] = int(len(picks))
    findings["n_managers"] = int(picks["manager"].nunique())
    findings["n_seasons"] = int(picks["season"].nunique())
    findings["coverage_reach"] = round(float(picks["reach"].notna().mean()), 3)
    findings["coverage_growth"] = round(float(picks["growth"].notna().mean()), 3)
    findings["coverage_steadiness"] = round(float(picks["steadiness"].notna().mean()), 3)
    findings["_notes"] = [
        "Every trait is scored AS OF the draft: the projection published for "
        "that season, and only actuals from before it.",
        "The within-draft correlation removes the (season, round) mean first, "
        "so shared late-round drift is not read as a personal trait.",
        f"Rounds 1-{EARLY_ROUNDS} for the across-season measure; the whole "
        f"draft, split at round {SPLIT_ROUND}, for the within-draft one.",
    ]
    return findings


EXPERIMENT = Experiment(
    id="e006",
    title="Owner tendencies: which ones persist, and on which axis",
    question="A manager's habits should help predict his next pick -- but "
             "which habits, and does history predict them or does the draft "
             "in front of us?",
    run=run,
    tags=("managers", "persistence", "draft-model"),
    overturns="The pick model fits six per-manager position dummies. Which "
              "positions a manager favours is measured here at -0.07 and "
              "-0.14 season to season -- indistinguishable from zero -- so "
              "those are six coefficients per owner chasing nothing, on ~85 "
              "picks each. Inside a draft the same tendency is significant "
              "and NEGATIVE: a manager who takes backs early takes fewer "
              "late. The effect is roster balance, which `need` already "
              "prices, and its sign is the opposite of a preference.",
)
