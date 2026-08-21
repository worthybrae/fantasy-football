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
# filling out a bench, and where the earlier pass measured its +0.43.
EARLY_ROUNDS = 5
# ...but truncating there throws away two thirds of the draft. Weighting every
# pick by ROUND_DECAY**(round-1) keeps all of them while letting the early
# ones dominate, and measures reach at +0.52 against the truncated +0.41 --
# a lower-variance estimator of the same habit. A sweep of windows and decay
# rates is in the post; this is the one that survived it.
ROUND_DECAY = 0.8
# The per-pick traits are personal habits and get the decay. A POSITION SHARE
# does not: it is a fact about the finished roster, and how many tight ends a
# manager ends up carrying is the whole question -- weighting rounds 12-16
# down to nothing would erase exactly the picks that answer it.
DECAYED_TRAITS = ("reach", "growth", "steadiness")
POSITION_SHARES = ("QB", "RB", "WR", "TE")
# The split point for the within-draft measure. Rounds 1-4 against 5+: an even
# split of a 16-round draft would put the whole split inside the bench.
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


# Six ways of asking "does he draft youth". Tested together, corrected
# together -- the point of trying six is that the first one failing is not
# evidence, and the point of correcting is that the best of six is not either.
YOUTH_ADP_WINDOW = 12
YOUNG_MAX_EXP = 2


def _with_ages(conn, picks: pd.DataFrame) -> pd.DataFrame:
    """Age and experience per pick, and the same relative to ADP neighbours.

    The peer comparison is the one that matters. A manager's AVERAGE draftee
    age is mostly a fact about who fell to him -- between-manager spread in it
    is half the size of one manager's own year-to-year swing -- so it measures
    the board rather than the drafter. Comparing each pick to the players
    ranked within YOUTH_ADP_WINDOW of him asks the question the average
    cannot: offered a tier, did he take the younger man?
    """
    pl = read_table(conn, "players")
    if pl.empty:
        return picks
    pl = pl.copy()
    pl["birth_date"] = pd.to_datetime(pl["birth_date"], errors="coerce")
    out = picks.merge(pl[["gsis_id", "display_name", "birth_date", "rookie_season"]],
                      left_on="player_id", right_on="gsis_id", how="left")
    kick = pd.to_datetime(out["season"].astype(str) + "-09-01")
    out["age"] = (kick - out["birth_date"]).dt.days / 365.25
    out["exp"] = out["season"] - out["rookie_season"].astype(float)

    adp = read_table(conn, "historic_adp")
    if adp.empty:
        out["age_peer"] = np.nan
        out["exp_peer"] = np.nan
        return out
    adp = adp.assign(norm=adp["adp_name"].map(_norm_name),
                     position=adp["position"].replace(_ADP_POSITION_ALIASES))
    names = pl.assign(_n=pl["display_name"].map(_norm_name)).drop_duplicates("_n")
    adp = adp.merge(names[["_n", "birth_date", "rookie_season"]],
                    left_on="norm", right_on="_n", how="left")
    akick = pd.to_datetime(adp["season"].astype(str) + "-09-01")
    adp["age"] = (akick - adp["birth_date"]).dt.days / 365.25
    adp["exp"] = adp["season"] - adp["rookie_season"].astype(float)
    for col in ("age", "exp"):
        peers = []
        for _, row in out.iterrows():
            rank = row.get("adp_rank")
            if rank != rank:
                peers.append(np.nan)
                continue
            pool = adp.loc[(adp["season"] == row["season"])
                           & (adp["adp_rank"].sub(rank).abs() <= YOUTH_ADP_WINDOW),
                           col].dropna()
            peers.append(row[col] - pool.mean() if len(pool) >= 5 else np.nan)
        out[f"{col}_peer"] = peers
    return out


def _youth_shapes(picks: pd.DataFrame) -> dict:
    """Per manager-season value for each candidate shape of the youth trait."""
    p = picks.copy()
    p["_young"] = (p["exp"] <= YOUNG_MAX_EXP).astype(float)
    p["_rookie"] = (p["exp"] <= 0).astype(float)
    p["_capital"] = (17 - _round_of(p)).clip(lower=1)
    g = p.groupby(["manager", "season"])
    early = p[_round_of(p) <= 8].groupby(["manager", "season"])
    return {
        "young_count": g["_young"].sum(),
        "rookie_count": g["_rookie"].sum(),
        "young_capital": p.assign(_x=p["_young"] * p["_capital"]).groupby(
            ["manager", "season"])["_x"].sum(),
        "age_peer": g["age_peer"].mean(),
        "exp_peer": g["exp_peer"].mean(),
        "young_early": early["_young"].mean(),
    }


def _pairs_needed(r: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """Owner-season pairs to detect a correlation of `r`, via Fisher's z.

    Published beside every null here, because "we found nothing" and "we could
    not have found it" are different claims and only one of them is an answer.
    """
    from scipy import stats
    if not np.isfinite(r) or abs(r) >= 1 or r == 0:
        return -1
    z = 0.5 * np.log((1 + r) / (1 - r))
    return int(np.ceil((stats.norm.ppf(1 - alpha / 2)
                        + stats.norm.ppf(power)) ** 2 / z ** 2 + 3))


def _round_of(picks: pd.DataFrame) -> pd.Series:
    return picks["round"].astype(float)


def _decayed_mean(picks: pd.DataFrame, trait: str) -> pd.DataFrame:
    """A manager-season's trait, with early picks weighted most.

    Late picks are constrained -- by what is left on the board and by the
    roster slots still open -- so they say less about taste than a first-round
    choice does. Truncating to rounds 1-5 is the crude version of that idea
    and measures reach at +0.41; decaying over the whole draft measures the
    same habit at +0.52, because it discounts the late rounds without
    discarding them.
    """
    d = picks.dropna(subset=[trait]).copy()
    d["_w"] = ROUND_DECAY ** (_round_of(d) - 1)
    d["_x"] = d[trait] * d["_w"]
    g = d.groupby(["manager", "season"]).agg(
        _x=("_x", "sum"), _w=("_w", "sum")).reset_index()
    g = g[g["_w"] > 0]
    g[trait] = g["_x"] / g["_w"]
    return g[["manager", "season", trait]]


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


# Seven traits on two axes is fourteen correlations, and at that many a
# |r| of 0.35 turns up somewhere by luck alone about half the time -- which
# is exactly how a -0.40 for steadiness first read as a discovery. Every
# reported p is therefore accompanied by a FAMILY-WISE one, from permuting
# manager labels and asking how often the whole family's best cell beats this
# one. Reporting only the per-cell p would publish the artefact.
FAMILY_PERMUTATIONS = 2000
FAMILY_SEED = 20260821


def _pairs_r(mat: np.ndarray) -> float:
    """Correlate column t against column t+1 down a manager x season matrix."""
    if mat.shape[1] < 2:
        return np.nan
    a = np.concatenate([mat[:, i] for i in range(mat.shape[1] - 1)])
    b = np.concatenate([mat[:, i] for i in range(1, mat.shape[1])])
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _cols_r(mat: np.ndarray) -> float:
    """Correlate two columns (early against late) of one matrix."""
    ok = ~(np.isnan(mat[:, 0]) | np.isnan(mat[:, 1]))
    if ok.sum() < 3 or np.std(mat[ok, 0]) == 0 or np.std(mat[ok, 1]) == 0:
        return np.nan
    return float(np.corrcoef(mat[ok, 0], mat[ok, 1])[0, 1])


def _family_pvalues(cells: dict) -> dict:
    """Family-wise p per cell, by permuting who each season's numbers belong to.

    `cells` maps a name to (observed_r, matrix, statistic). The null shuffles
    values within each column -- destroying the manager linkage while keeping
    every season's distribution intact -- so it asks precisely "could this
    much year-to-year agreement arise with no persistent owner at all".
    """
    rng = np.random.default_rng(FAMILY_SEED)
    names = [k for k, (r, _, _) in cells.items() if r == r]
    best = np.zeros(FAMILY_PERMUTATIONS)
    draws = {n: np.zeros(FAMILY_PERMUTATIONS) for n in names}
    for i in range(FAMILY_PERMUTATIONS):
        top = 0.0
        for n in names:
            _, mat, stat = cells[n]
            sh = np.column_stack([rng.permutation(mat[:, j])
                                  for j in range(mat.shape[1])])
            r = stat(sh)
            draws[n][i] = r
            if r == r and abs(r) > top:
                top = abs(r)
        best[i] = top
    return {n: float(np.mean(best >= abs(cells[n][0]))) for n in names}


def _across_season(per_season: pd.DataFrame, trait: str) -> tuple:
    """Correlate a manager's trait in season t with the same manager in t+1."""
    a = per_season.dropna(subset=[trait])[["manager", "season", trait]]
    mat = a.pivot(index="manager", columns="season", values=trait)
    mat = mat[sorted(mat.columns)].to_numpy(dtype=float)
    n = int(np.sum(~(np.isnan(mat[:, :-1]) | np.isnan(mat[:, 1:]))))
    return _pairs_r(mat), n, mat


def _within_draft(picks: pd.DataFrame, trait: str) -> tuple:
    """Early-round mean against late-round mean, same manager, same draft.

    The round-bucket mean is removed first. Reaches grow in the late rounds
    for everyone -- the board thins and the remaining players have no ADP
    worth honouring -- so a raw early/late correlation would report that
    shared drift as a personal trait for all eight managers at once.
    """
    df = picks.dropna(subset=[trait]).copy()
    if df.empty:
        return np.nan, 0, np.empty((0, 2))
    df["_r"] = _round_of(df)
    df["_dev"] = df[trait] - df.groupby(["season", "_r"])[trait].transform("mean")
    df["_half"] = np.where(df["_r"] < SPLIT_ROUND, "early", "late")
    halves = df.groupby(["manager", "season", "_half"])["_dev"].agg(["mean", "size"])
    halves = halves[halves["size"] >= MIN_PICKS_PER_HALF]["mean"].unstack("_half")
    halves = halves.dropna()
    if len(halves) < 3 or not {"early", "late"} <= set(halves.columns):
        return np.nan, len(halves), np.empty((0, 2))
    mat = halves[["early", "late"]].to_numpy(dtype=float)
    return _cols_r(mat), len(halves), mat


def run(conn=None, odds=None, rules=None) -> dict:
    """`odds` is unused -- run.py hands every experiment the same three."""
    if conn is None:
        from pipeline.db import get_conn
        conn = get_conn()
    picks = _pick_frame(conn, rules)
    if picks.empty:
        raise RuntimeError("no draft history imported -- nothing to measure")
    picks = _with_ages(conn, picks)

    per_season = None
    for trait in DECAYED_TRAITS:
        col = _decayed_mean(picks, trait)
        per_season = col if per_season is None else per_season.merge(
            col, on=["manager", "season"], how="outer")
    # Position is a SHARE of the FINISHED roster, not a decayed mean of a
    # per-pick number, and it is measured over the whole draft: the first
    # version of this experiment took its share over rounds 1-5 and reported
    # that positional preference does not persist. That was the wrong window
    # for the question. Which position a manager takes EARLY is dictated by
    # the board; how many tight ends he walks away with is a choice, and it
    # is the most persistent thing measured here.
    for pos in POSITION_SHARES:
        share = (picks.assign(_hit=(picks["position"] == pos).astype(float))
                 .groupby(["manager", "season"])["_hit"].mean()
                 .rename(f"pos_{pos}").reset_index())
        per_season = per_season.merge(share, on=["manager", "season"], how="left")
        picks[f"pos_{pos}"] = (picks["position"] == pos).astype(float)

    traits = DECAYED_TRAITS + tuple(f"pos_{p}" for p in POSITION_SHARES)
    cells, findings = {}, {}
    for trait in traits:
        r_across, n_across, m_across = _across_season(per_season, trait)
        r_within, n_within, m_within = _within_draft(picks, trait)
        cells[f"across_{trait}"] = (r_across, m_across, _pairs_r)
        cells[f"within_{trait}"] = (r_within, m_within, _cols_r)
        for axis, r, n in (("across", r_across, n_across),
                           ("within", r_within, n_within)):
            pv = _p_value(r, n)
            findings[f"{axis}_{trait}"] = None if np.isnan(r) else round(r, 3)
            findings[f"{axis}_n_{trait}"] = n
            findings[f"{axis}_p_{trait}"] = None if np.isnan(pv) else round(pv, 3)
    # The six youth shapes join the SAME family. Asking the question six ways
    # and correcting only within the six would still let the best of them be
    # compared against a bar the other traits never had to clear.
    youth_mats = {}
    for name, series in _youth_shapes(picks).items():
        m = series.rename("v").reset_index().pivot(
            index="manager", columns="season", values="v")
        m = m[sorted(m.columns)].to_numpy(dtype=float)
        youth_mats[f"youth_{name}"] = m
        cells[f"youth_{name}"] = (_pairs_r(m), m, _pairs_r)

    family = _family_pvalues(cells)
    for key, pf in family.items():
        findings[f"{key}_pfam"] = round(pf, 3)
        # SIGNIFICANCE IS THE FAMILY-WISE CALL, not the per-cell one. The
        # per-cell p stays published beside it so the gap between the two is
        # visible rather than quietly resolved.
        findings[f"{key}_sig"] = bool(pf < 0.05)
    for key in cells:
        findings.setdefault(f"{key}_pfam", None)
        findings.setdefault(f"{key}_sig", False)

    # The comparison that justifies the decay, measured rather than
    # remembered: the same trait under the old rounds-1-EARLY_ROUNDS flat
    # mean. The post argues from the gap between these two, so the post may
    # not be the only place the old number exists.
    flat = (picks[_round_of(picks) <= EARLY_ROUNDS]
            .groupby(["manager", "season"])["reach"].mean().reset_index())
    r_flat, _, _ = _across_season(flat, "reach")
    findings["across_reach_truncated"] = None if np.isnan(r_flat) else round(r_flat, 3)

    # Leave-one-manager-out on the headline result. With eight managers a
    # single one can carry a correlation, and "it survives dropping any of
    # them" is the claim -- so it has to be a measurement, not an assurance.
    jack = []
    for m in sorted(picks["manager"].dropna().unique()):
        sub = picks[picks["manager"] != m]
        share = (sub.assign(_hit=(sub["position"] == "TE").astype(float))
                 .groupby(["manager", "season"])["_hit"].mean()
                 .rename("pos_TE").reset_index())
        r_j, _, _ = _across_season(share, "pos_TE")
        if r_j == r_j:
            jack.append(r_j)
    findings["across_pos_TE_jack_lo"] = round(min(jack), 2) if jack else None
    findings["across_pos_TE_jack_hi"] = round(max(jack), 2) if jack else None

    # The best of the six, and what it would take to confirm it. Recorded so
    # that "we looked for this and could not see it" is a result someone can
    # read rather than a search someone repeats.
    youth = {k: cells[k][0] for k in cells if k.startswith("youth_")}
    youth = {k: v for k, v in youth.items() if v == v}
    if youth:
        best = max(youth, key=lambda k: abs(youth[k]))
        findings["youth_best_shape"] = best.replace("youth_", "")
        findings["youth_best_r"] = round(youth[best], 3)
        findings["youth_best_pfam"] = round(family.get(best, float("nan")), 3)
        findings["youth_pairs_needed"] = _pairs_needed(youth[best])
        findings["youth_shapes_tried"] = len(youth)
    findings["pairs_have"] = int(max(
        findings.get(f"across_n_{t}", 0) for t in traits))
    # The same question for the traits that DID survive, so the contrast is a
    # measurement rather than a claim: TE needs a fraction of the history we
    # already have, which is the whole reason it is visible and youth is not.
    for t in ("reach", "pos_TE", "pos_WR"):
        r = findings.get(f"across_{t}")
        if r is not None:
            findings[f"pairs_needed_{t}"] = _pairs_needed(r)

    findings["decay"] = ROUND_DECAY
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
        f"Per-pick traits are weighted {ROUND_DECAY}^(round-1) across the "
        f"whole draft. Position is an unweighted share of the finished "
        f"roster: an earlier version took it over rounds 1-{EARLY_ROUNDS} "
        f"and concluded positional preference does not persist, which is the "
        f"wrong window for a question about what a manager ends up with.",
        f"Family-wise p is from {FAMILY_PERMUTATIONS} permutations of which "
        f"manager each season's numbers belong to, across all fourteen cells "
        f"at once. Significance is that number, not the per-cell p.",
        "The within-draft split is at round "
        f"{SPLIT_ROUND}, over drafts of 16 rounds.",
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
    overturns="Positional preference was measured over rounds 1-5 and "
              "reported as not persisting. That is the wrong window: which "
              "position a manager takes early is dictated by the board, "
              "while how many tight ends he ends up with is a choice, and "
              "over the whole draft it is the most persistent thing here "
              "(+0.69). The pick model's six per-manager position dummies "
              "were the right instinct at the wrong resolution -- what "
              "carries is a target roster shape, and inside a draft a "
              "manager moves AGAINST it as he fills up, which is why a "
              "preference-shaped feature had the sign backwards."
)
