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
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipeline.db import read_table, write_table
from scoring import league as league_mod
from scoring.board import _ADP_POSITION_ALIASES, _norm_name, adp_match_key
from scoring.player_history import attributes_as_of

RUN_WINDOW = 5

_ATTRIBUTE_DEFAULTS = {"age": np.nan, "ppg_std": 0.0, "missed_rate": 0.0,
                       "no_track_record": True, "prod_rank": np.nan,
                       "trend": 0.0}


def _enrich_pool(conn, pool: pd.DataFrame, season: int,
                 espn: pd.DataFrame) -> pd.DataFrame:
    """Attach the market reference and the player-at-pick-time attributes.

    `market_rank` prefers ESPN's dense rank -- the league drafts on ESPN,
    off ESPN's board, so that is the ordering the room actually saw -- and
    falls back to the FFC `adp_rank` for a player ESPN's top-N did not
    reach. Both are dense 1..N per season, so the two are on one scale.
    """
    pool = pool.copy()
    if espn.empty:
        pool["market_rank"] = pool["adp_rank"]
    else:
        season_espn = espn[espn["season"] == season].copy()
        season_espn["key"] = _match_keys(season_espn, "espn_name")
        season_espn = season_espn.dropna(subset=["key"]).drop_duplicates("key")
        ranks = season_espn.set_index("key")["espn_rank"]
        pool["market_rank"] = pool["key"].map(ranks).fillna(pool["adp_rank"])

    attrs = attributes_as_of(conn, season)
    if not attrs.empty:
        # `adp_match_key` carries no team for a non-DST position, so two
        # different past players can share (position, normalized name) --
        # a real risk with common names, not a contrived one. Guessing
        # which one's numbers belong to the player actually drafted would
        # put a wrong value on a "no_track_record: False" row; left-joining
        # against a key that appears twice would also duplicate the pool
        # row, breaking "one pick removes exactly one pool row". Drop both
        # sides of the collision so it falls through to "unknown" instead.
        attrs = attrs.drop_duplicates("key", keep=False)
    if attrs.empty:
        for col, default in _ATTRIBUTE_DEFAULTS.items():
            pool[col] = default
    else:
        pool = pool.merge(attrs, on="key", how="left")
        pool["no_track_record"] = pool["no_track_record"].fillna(True).astype(bool)
        for col, default in _ATTRIBUTE_DEFAULTS.items():
            if col not in ("no_track_record", "prod_rank"):
                pool[col] = pool[col].fillna(default)

    # Positive means the market is ahead of what the player has actually
    # done -- taking him is a leap of faith. NaN when he has no production
    # to rank, which `feature_matrix` reads as neutral rather than as zero.
    pool["hype"] = pool["prod_rank"] - pool["market_rank"]
    return pool


def _match_keys(frame: pd.DataFrame, name_col: str, team_col: str = "team") -> list:
    """`board.adp_match_key` for every row.

    `team_col` differs by table: `historic_adp` carries `team`, `draft_picks`
    carries `nfl_team`. A missing column (a historic_adp written before it
    carried one) yields no team, which makes DSTs unmatchable rather than
    wrongly matched.
    """
    teams = (frame[team_col] if team_col in frame.columns
             else pd.Series([None] * len(frame), index=frame.index))
    return [adp_match_key(name, position, team) for name, position, team
            in zip(frame[name_col], frame["position"], teams)]


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
    espn = read_table(conn, "historic_espn")
    if picks.empty or teams.empty or adp.empty:
        return []

    picks = picks.merge(teams[["season", "team_id", "manager"]],
                        on=["season", "team_id"], how="left")
    picks = picks.assign(norm=picks["player_name"].map(_norm_name))
    # Defensive: a historic_adp written before import_league normalized it
    # still says "PK" for kickers, which would leave pos_K an all-zero,
    # unidentified column in the feature matrix even once the join is keyed
    # correctly. Same alias table the board applies to the live ADP feed.
    adp = adp.assign(norm=adp["adp_name"].map(_norm_name),
                     position=adp["position"].replace(_ADP_POSITION_ALIASES))
    adp = adp.assign(key=_match_keys(adp, "adp_name"))
    picks = picks.assign(key=_match_keys(picks, "player_name", "nfl_team"))

    out = []
    for season, season_picks in picks.groupby("season"):
        pool = adp[adp["season"] == season][["norm", "position", "adp_rank", "key"]]
        # A DST with no team has no join key at all (see adp_match_key); it
        # is unmatchable, and leaving it in would let two of them match each
        # other.
        pool = pool[pool["key"].map(lambda k: k is not None)]
        # historic_adp has no uniqueness guarantee on the join key --
        # parse_adp builds it straight from the external payload, and
        # import_league writes it without deduping. Collapse to the
        # best-known (lowest) adp_rank per player before anything else, so
        # "one pick removes exactly one pool row" holds structurally rather
        # than by accident of the fixture. Same precedent as
        # scoring.board._dedupe_adp.
        pool = pool.sort_values("adp_rank").drop_duplicates(
            "key", keep="first").reset_index(drop=True)
        pool = _enrich_pool(conn, pool, season, espn)
        available = pool.copy()
        rosters, recent = {}, []
        for _, pick in season_picks.sort_values("overall_pick").iterrows():
            key = pick["key"]
            match = (available.index[available["key"] == key]
                     if key is not None else available.index[[]])
            if len(match):
                reset = available.reset_index(drop=True)
                chosen = int(reset.index[reset["key"] == key][0])
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
_NEW_FEATURES = ["age", "volatility", "no_track_record", "hype", "trend"]
FEATURE_NAMES = (["reach", "fall"]
                 + [f"pos_{p}" for p in _POSITION_DUMMIES]
                 + ["qb_early", "te_early", "need", "run"]
                 + _NEW_FEATURES)
EARLY_ROUNDS = 3

# Divisors that put each new feature on roughly the same scale as the
# others, so no coefficient has to be tiny or huge to matter. They are not
# fitted -- changing one just rescales its coefficient -- but keeping the
# columns comparable makes the ridge penalty treat them even-handedly.
VOLATILITY_SCALE = 10.0
HYPE_SCALE = 50.0


def _log_rank_features(market_rank, pick_no):
    """Rounds-of-reach on a log axis.

    Linear rank made the gap from rank 1 to 5 (0.5 units) ten times smaller
    than the gap from 100 to 140 (5.0), so the fit spent its range on
    deep-bench noise and left the entire top of the board within ~2x of
    itself in probability -- a coin flip for the first pick of the draft.
    Log rank inverts that, matching how drafts actually behave. log1p so
    rank 1 and pick 1 are finite; no division by `teams`, which was a
    rounds-based scaling that means nothing on a log axis.
    """
    delta = np.log1p(market_rank) - np.log1p(pick_no)
    return np.maximum(0.0, delta), np.maximum(0.0, -delta)


def _centre_within_position(values, positions):
    """Age relative to typical for the position, so the coefficient reads as
    'younger than his peers' rather than tracking that tight ends last
    longer than running backs. Unknown ages are neutral, not young."""
    out = np.zeros(len(values), dtype=float)
    values = np.asarray(values, dtype=float)
    for pos in set(positions):
        mask = positions == pos
        known = mask & np.isfinite(values)
        if known.any():
            out[known] = values[known] - values[known].mean()
    return out


def feature_matrix(obs: PickObservation, settings) -> np.ndarray:
    pool = obs.pool
    n = len(pool)
    ranks = pool["market_rank"].to_numpy(dtype=float)
    positions = pool["position"].to_numpy()
    teams = max(settings.teams, 1)

    reach, fall = _log_rank_features(ranks, obs.overall_pick)

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

    columns.append(_centre_within_position(pool["age"].to_numpy(), positions))
    columns.append(pool["ppg_std"].to_numpy(dtype=float) / VOLATILITY_SCALE
                   + pool["missed_rate"].to_numpy(dtype=float))
    columns.append(pool["no_track_record"].to_numpy().astype(float))
    hype = pool["hype"].to_numpy(dtype=float)
    columns.append(np.nan_to_num(hype, nan=0.0) / HYPE_SCALE)
    columns.append(pool["trend"].to_numpy(dtype=float))

    return np.column_stack(columns) if n else np.zeros((0, len(FEATURE_NAMES)))


def _shifted_exp(scores: np.ndarray) -> np.ndarray:
    """exp(scores - scores.max()), the max-subtracted exponentiation shared by
    softmax normalization and the log-sum-exp term. Both need it; computing it
    once and reusing it (rather than each recomputing it from scores) halves
    the exp() calls per observation on the hot path fit() iterates over."""
    return np.exp(scores - scores.max())


def _softmax(scores: np.ndarray) -> np.ndarray:
    exp = _shifted_exp(scores)
    return exp / exp.sum()


def log_likelihood(beta, X_list, chosen_list) -> float:
    total = 0.0
    for X, k in zip(X_list, chosen_list):
        scores = X @ beta
        exp = _shifted_exp(scores)
        log_sum_exp = scores.max() + np.log(exp.sum())
        total += scores[k] - log_sum_exp
    return float(total)


def neg_log_likelihood(beta, X_list, chosen_list, prior=None, lam=0.0):
    """Value and gradient of the ridge-penalized negative log-likelihood.

    Convex in beta, which is why L-BFGS-B finds the global optimum rather than
    a local one -- the reason this uses scipy instead of a hand-rolled loop.
    """
    value = 0.0
    grad = np.zeros_like(beta, dtype=float)
    for X, k in zip(X_list, chosen_list):
        scores = X @ beta
        exp = _shifted_exp(scores)
        exp_sum = exp.sum()
        probs = exp / exp_sum
        log_sum_exp = scores.max() + np.log(exp_sum)
        value -= scores[k] - log_sum_exp
        grad += probs @ X - X[k]
    if prior is not None and lam:
        diff = beta - prior
        value += lam * float(diff @ diff)
        grad += 2.0 * lam * diff
    return value, grad


def fit(X_list, chosen_list, prior=None, lam: float = 0.0) -> np.ndarray:
    n_features = X_list[0].shape[1] if X_list else len(FEATURE_NAMES)
    start = np.zeros(n_features) if prior is None else np.asarray(prior, dtype=float).copy()
    if not X_list:
        return start
    result = minimize(neg_log_likelihood, start,
                      args=(X_list, chosen_list, prior, lam),
                      jac=True, method="L-BFGS-B")
    if not result.success:
        warnings.warn(
            f"draft_model.fit: L-BFGS-B did not converge ({result.message}); "
            f"returning its result anyway from {len(X_list)} choice sets",
            RuntimeWarning,
        )
    return result.x


def prepare(observations, settings):
    X_list = [feature_matrix(o, settings) for o in observations]
    chosen = [o.chosen for o in observations]
    managers = [o.manager for o in observations]
    seasons = [o.season for o in observations]
    return X_list, chosen, managers, seasons


LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MIN_PICKS_FOR_PERSONAL = 20

# Decay rate for the ADP baseline in backtest(): the pool passed to
# feature_matrix is always sorted ascending by adp_rank (build_observations
# never reorders it), so pool position IS market rank order and a softmax
# over -TEMPERATURE * position gives a real probability distribution over
# "who does the market think goes next" -- unlike a uniform distribution,
# which assigns the market's #1 player and its #200th the same probability
# and so would be beaten by nearly anything. TEMPERATURE=1.0 means each step
# down the ADP board is ~e times less likely than the one before it: sharp
# enough to be a meaningful baseline (most snake-draft picks land within a
# few spots of the top of the board), not so sharp that it degenerates into
# "always predict index 0" and stops being a distribution worth comparing
# log-loss against.
ADP_BASELINE_TEMPERATURE = 1.0

# Plain-language templates for the coefficients worth surfacing. Keyed by
# feature name; each maps (deviation from pooled) -> a phrase.
_PHRASES = {
    "reach": ("reaches for players the market ranks later",
              "avoids reaches, drafts in market order"),
    "fall": ("chases players who slide", "ignores players who slide"),
    "pos_RB": ("leans RB", "fades RB"),
    "pos_WR": ("leans WR", "fades WR"),
    "pos_TE": ("leans TE", "fades TE"),
    "pos_K": ("takes kickers early", "leaves kickers late"),
    "pos_DST": ("takes defenses early", "leaves defenses late"),
    "qb_early": ("early-QB guy", "waits on QB"),
    "te_early": ("early-TE guy", "waits on TE"),
    "need": ("fills starting slots first", "ignores roster needs"),
    "run": ("chases positional runs", "fades positional runs"),
    # Task 4 columns. `describe` takes the top-3 |diff| features and drops
    # any name missing here without looking further down the list -- with
    # five of sixteen features unnamed, the top three could easily be new
    # ones, and a manager with a real, strong deviation reported back as
    # "drafts close to league average". Sign matters: `age` is centred
    # (positive = older than the position average), `hype` is
    # prod_rank - market_rank (positive = market ranks him ahead of his own
    # production -- a leap of faith), so a positive coefficient on either
    # means "leans toward more of that", not "leans toward the word that
    # comes first alphabetically".
    "age": ("leans veteran", "leans youth"),
    "volatility": ("chases volatile, injury-prone players",
                   "prefers steady, durable players"),
    "no_track_record": ("bets on unproven players", "avoids unproven players"),
    "hype": ("chases hype over production", "fades hype, trusts production"),
    "trend": ("targets players trending up", "sticks with steady production"),
}


def select_lambda(X_list, chosen_list, seasons, prior, grid=None) -> float:
    """Leave-one-season-out cross-validation over the ridge strength.

    Seasons, not random folds: picks inside one draft are not independent of
    each other, so a random split would leak the same draft across train and
    test and pick a lambda that is too loose.
    """
    grid = grid or LAMBDA_GRID
    unique = sorted(set(seasons))
    if len(unique) < 2:
        return grid[-1]                      # one season: shrink hard
    best, best_ll = grid[-1], -np.inf
    for lam in grid:
        total = 0.0
        for holdout in unique:
            train = [i for i, s in enumerate(seasons) if s != holdout]
            test = [i for i, s in enumerate(seasons) if s == holdout]
            if not train or not test:
                continue
            beta = fit([X_list[i] for i in train], [chosen_list[i] for i in train],
                       prior=prior, lam=lam)
            total += log_likelihood(beta, [X_list[i] for i in test],
                                    [chosen_list[i] for i in test])
        if total > best_ll:
            best, best_ll = lam, total
    return best


def fit_all(conn, settings=None) -> dict:
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    if not observations:
        return {}
    X_list, chosen, managers, seasons = prepare(observations, settings)
    pooled = fit(X_list, chosen)
    fits = {"__pooled__": pooled}
    for manager in sorted(set(managers)):
        idx = [i for i, m in enumerate(managers) if m == manager]
        Xm = [X_list[i] for i in idx]
        cm = [chosen[i] for i in idx]
        sm = [seasons[i] for i in idx]
        lam = select_lambda(Xm, cm, sm, prior=pooled)
        fits[manager] = fit(Xm, cm, prior=pooled, lam=lam)
    return fits


def _heldout_gain(X_list, chosen, seasons, pooled) -> float:
    """Per-pick log-likelihood advantage of a personal fit over pooled.

    Positive means the manager's own coefficients predict held-out picks
    better than the league-wide ones. Negative means they do not, and the
    simulator should use pooled for that manager.
    """
    unique = sorted(set(seasons))
    if len(unique) < 2:
        return -np.inf
    personal_ll = pooled_ll = 0.0
    n = 0
    for holdout in unique:
        train = [i for i, s in enumerate(seasons) if s != holdout]
        test = [i for i, s in enumerate(seasons) if s == holdout]
        if not train or not test:
            continue
        lam = select_lambda([X_list[i] for i in train], [chosen[i] for i in train],
                            [seasons[i] for i in train], prior=pooled)
        beta = fit([X_list[i] for i in train], [chosen[i] for i in train],
                   prior=pooled, lam=lam)
        Xt = [X_list[i] for i in test]
        ct = [chosen[i] for i in test]
        personal_ll += log_likelihood(beta, Xt, ct)
        pooled_ll += log_likelihood(pooled, Xt, ct)
        n += len(test)
    return (personal_ll - pooled_ll) / n if n else -np.inf


def describe(beta, pooled, top: int = 3) -> str:
    diff = np.asarray(beta) - np.asarray(pooled)
    order = np.argsort(-np.abs(diff))
    phrases = []
    for i in order[:top]:
        name = FEATURE_NAMES[i]
        if name not in _PHRASES or abs(diff[i]) < 0.05:
            continue
        high, low = _PHRASES[name]
        phrases.append(high if diff[i] > 0 else low)
    return ", ".join(phrases) if phrases else "drafts close to league average"


def write_profiles(conn, settings=None) -> pd.DataFrame:
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    if not observations:
        empty = pd.DataFrame(columns=[
            "manager", "feature", "value", "pooled_value", "n_picks",
            "heldout_gain", "uses_personal", "summary"])
        write_table(conn, "manager_profiles", empty)
        return empty

    X_list, chosen, managers, seasons = prepare(observations, settings)
    pooled = fit(X_list, chosen)
    rows = []
    for manager in sorted(set(managers)):
        idx = [i for i, m in enumerate(managers) if m == manager]
        Xm, cm = [X_list[i] for i in idx], [chosen[i] for i in idx]
        sm = [seasons[i] for i in idx]
        lam = select_lambda(Xm, cm, sm, prior=pooled)
        beta = fit(Xm, cm, prior=pooled, lam=lam)
        gain = _heldout_gain(Xm, cm, sm, pooled)
        uses_personal = bool(len(idx) >= MIN_PICKS_FOR_PERSONAL and gain > 0)
        effective = beta if uses_personal else pooled
        summary = describe(effective, pooled) if uses_personal else \
            "league average, not enough signal"
        for i, name in enumerate(FEATURE_NAMES):
            rows.append({"manager": manager, "feature": name,
                         "value": float(beta[i]), "pooled_value": float(pooled[i]),
                         "n_picks": len(idx),
                         "heldout_gain": float(gain) if np.isfinite(gain) else None,
                         "uses_personal": uses_personal, "summary": summary})
    profiles = pd.DataFrame(rows)
    write_table(conn, "manager_profiles", profiles)
    return profiles


def backtest(conn, settings=None) -> dict:
    """Hold out the newest season and score against an ADP-only baseline.

    If the fitted model does not beat "the market's next-best player is next
    off the board", that is the finding, and the board must not present
    simulator output as authoritative.
    """
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    if not observations:
        return {"holdout_season": None, "top1": 0.0, "top5": 0.0,
                "logloss": float("inf"), "adp_top1": 0.0,
                "adp_logloss": float("inf"), "beats_adp": False}
    X_list, chosen, managers, seasons = prepare(observations, settings)
    holdout = max(seasons)
    train = [i for i, s in enumerate(seasons) if s != holdout]
    test = [i for i, s in enumerate(seasons) if s == holdout]
    pooled = fit([X_list[i] for i in train], [chosen[i] for i in train]) \
        if train else np.zeros(len(FEATURE_NAMES))

    fits = {}
    for manager in set(managers):
        idx = [i for i in train if managers[i] == manager]
        if len(idx) < MIN_PICKS_FOR_PERSONAL:
            fits[manager] = pooled
            continue
        lam = select_lambda([X_list[i] for i in idx], [chosen[i] for i in idx],
                            [seasons[i] for i in idx], prior=pooled)
        fits[manager] = fit([X_list[i] for i in idx], [chosen[i] for i in idx],
                            prior=pooled, lam=lam)

    hits1 = hits5 = 0
    ll = adp_ll = 0.0
    adp_hits1 = 0
    for i in test:
        X, k = X_list[i], chosen[i]
        probs = _softmax(X @ fits.get(managers[i], pooled))
        order = np.argsort(-probs)
        hits1 += int(order[0] == k)
        hits5 += int(k in order[:5])
        ll += np.log(max(probs[k], 1e-12))
        # ADP baseline: the pool is sorted ascending by adp_rank (see
        # build_observations), so pool position IS market rank order and
        # index 0 is the market's next player. A uniform distribution over
        # the pool is not a real baseline -- it would assign the market's #1
        # player and its #200th the same probability, so "beats the market"
        # would be true almost by construction. Score the market's own
        # ranking with a softmax over pool position instead: a real
        # probability distribution to compare the fitted model against.
        adp_hits1 += int(k == 0)
        adp_probs = _softmax(-ADP_BASELINE_TEMPERATURE * np.arange(len(X)))
        adp_ll += np.log(max(adp_probs[k], 1e-12))
    n = max(len(test), 1)
    report = {"holdout_season": int(holdout), "top1": hits1 / n, "top5": hits5 / n,
              "logloss": -ll / n, "adp_top1": adp_hits1 / n,
              "adp_logloss": -adp_ll / n}
    report["beats_adp"] = bool(report["logloss"] < report["adp_logloss"])
    return report


def _round_bucket(overall_pick: int, teams: int) -> str:
    round_no = (overall_pick - 1) // max(teams, 1) + 1
    if round_no <= EARLY_ROUNDS:
        return "early"
    return "mid" if round_no <= 8 else "late"


def positional_bias(conn, settings=None) -> pd.DataFrame:
    """How far ahead of the market this league takes each position.

    `mean_gap` is `market_rank - overall_pick`, so positive means the league
    drafts that position earlier than the market ranks it (a player ranked
    2 taken at pick 1 scores +1); negative means the league lets it slide
    past where the market has it. Pooled across managers on purpose: a
    league-wide habit forced through eight per-manager coefficients would
    spend scarce data re-learning the same thing eight times.
    """
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    rows = []
    for obs in observations:
        chosen = obs.pool.iloc[obs.chosen]
        rows.append({"position": chosen["position"],
                     "round_bucket": _round_bucket(obs.overall_pick, settings.teams),
                     "gap": float(chosen["market_rank"]) - obs.overall_pick})
    if not rows:
        return pd.DataFrame(columns=["position", "round_bucket", "mean_gap", "n"])
    df = pd.DataFrame(rows)
    out = df.groupby(["position", "round_bucket"], as_index=False).agg(
        mean_gap=("gap", "mean"), n=("gap", "size"))
    return out


def write_backtest(conn, settings=None) -> dict:
    """Run the backtest and persist it as a one-row `model_backtest` table.

    Spec Part 3: "If the model does not beat ADP-only, that is the finding,
    and the board should not present simulator output as authoritative."
    Printing it to stdout from `make fit-managers` cannot reach the board, so
    nothing stopped the board presenting the simulator as authoritative
    regardless. `/api/model` serves this row and the rail warns on it.
    """
    report = backtest(conn, settings)
    write_table(conn, "model_backtest", pd.DataFrame([report]))
    return report
