import numpy as np
import pandas as pd
import pytest
from pipeline.db import get_conn, write_table
from pipeline.import_league import _HISTORIC_ADP_COLUMNS, normalize_historic_adp
from scoring.draft_model import (FEATURE_NAMES, PickObservation,
                                  build_observations, feature_matrix)
from scoring import league

def _seed(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    # Two managers, four picks, one season. ADP order: A, B, C, D.
    write_table(conn, "draft_picks", pd.DataFrame([
        {"season": 2025, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 11, "player_name": "Player A",
         "position": "RB", "nfl_team": "DET", "keeper": False},
        {"season": 2025, "overall_pick": 2, "round": 1, "round_pick": 2,
         "team_id": 2, "espn_player_id": 12, "player_name": "Player C",
         "position": "WR", "nfl_team": "GB", "keeper": False},
        {"season": 2025, "overall_pick": 3, "round": 2, "round_pick": 1,
         "team_id": 2, "espn_player_id": 13, "player_name": "Player B",
         "position": "RB", "nfl_team": "CHI", "keeper": False},
        {"season": 2025, "overall_pick": 4, "round": 2, "round_pick": 2,
         "team_id": 1, "espn_player_id": 14, "player_name": "Ghost",
         "position": "TE", "nfl_team": "MIN", "keeper": False},
    ]))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2025, "team_id": 2, "manager": "dan", "slot": 2},
    ]))
    write_table(conn, "historic_adp", pd.DataFrame([
        {"season": 2025, "adp_name": "Player A", "position": "RB", "adp_rank": 1},
        {"season": 2025, "adp_name": "Player B", "position": "RB", "adp_rank": 2},
        {"season": 2025, "adp_name": "Player C", "position": "WR", "adp_rank": 3},
        {"season": 2025, "adp_name": "Player D", "position": "TE", "adp_rank": 4},
    ]))
    return conn

def _seed_kdst(tmp_path):
    """One season with a WR, a K and a DST pick, and a historic_adp written
    the way `pipeline.import_league` writes it after normalization.

    draft_picks come from ESPN: positions K/DST, `player_name` from ESPN's
    `fullName` ("Ravens D/ST"), `nfl_team` from ESPN_PRO_TEAMS. The ADP feed
    calls the same two players a "PK" and "Ravens".
    """
    conn = get_conn(str(tmp_path / "kdst.duckdb"))
    write_table(conn, "draft_picks", pd.DataFrame([
        {"season": 2025, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 11, "player_name": "Real Receiver",
         "position": "WR", "nfl_team": "DET", "keeper": False},
        {"season": 2025, "overall_pick": 2, "round": 1, "round_pick": 2,
         "team_id": 2, "espn_player_id": 12, "player_name": "Boot Leg",
         "position": "K", "nfl_team": "GB", "keeper": False},
        {"season": 2025, "overall_pick": 3, "round": 2, "round_pick": 1,
         "team_id": 2, "espn_player_id": 13, "player_name": "Ravens D/ST",
         "position": "DST", "nfl_team": "BAL", "keeper": False},
    ]))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2025, "team_id": 2, "manager": "dan", "slot": 2},
    ]))
    raw = pd.DataFrame([
        {"adp_name": "Real Receiver", "position": "WR", "team": "DET", "adp": 1.0},
        {"adp_name": "Boot Leg", "position": "PK", "team": "GB", "adp": 2.0},
        {"adp_name": "Ravens", "position": "DEF", "team": "BAL", "adp": 3.0},
    ])
    # parse_adp already maps DEF -> DST; PK and the team spellings are what
    # normalize_historic_adp is for.
    raw["position"] = raw["position"].replace({"DEF": "DST"})
    adp = normalize_historic_adp(raw).sort_values("adp").reset_index(drop=True)
    adp["adp_rank"] = adp.index + 1
    adp["season"] = 2025
    write_table(conn, "historic_adp", adp[_HISTORIC_ADP_COLUMNS])
    return conn


def test_kicker_and_defense_picks_produce_observations(tmp_path):
    """K and DST picks have to reach the fitting set.

    The ADP feed labels kickers "PK" and gives a defense its nickname
    ("Ravens") while ESPN says "K" and "Ravens D/ST", and historic_adp was
    written straight through with neither normalized and no team column. Of
    these three picks only the WR survived, and in a 15-round draft K plus
    DST is 2/15 of every season -- about 13% of all picks, guaranteed
    unmatched, enough on its own to drag a healthy import under the import
    report's 80% threshold. It also left pos_K and pos_DST all-zero,
    unidentified columns in the feature matrix.
    """
    obs = build_observations(_seed_kdst(tmp_path))

    assert len(obs) == 3
    chosen = {o.overall_pick: o.pool.iloc[o.chosen]["position"] for o in obs}
    assert chosen == {1: "WR", 2: "K", 3: "DST"}


def test_kicker_and_defense_reach_the_feature_matrix_position_dummies(tmp_path):
    """The point of matching them: pos_K and pos_DST stop being all-zero."""
    obs = build_observations(_seed_kdst(tmp_path))
    settings = league.default_settings()
    columns = {name: i for i, name in enumerate(FEATURE_NAMES)}

    first = feature_matrix(obs[0], settings)          # full three-player pool
    assert first[:, columns["pos_K"]].sum() == 1.0
    assert first[:, columns["pos_DST"]].sum() == 1.0


def test_pool_shrinks_as_players_come_off_the_board(tmp_path):
    obs = build_observations(_seed(tmp_path))
    assert [len(o.pool) for o in obs] == [4, 3, 2]

def test_chosen_index_points_at_the_player_actually_taken(tmp_path):
    obs = build_observations(_seed(tmp_path))
    first = obs[0]
    assert first.pool.iloc[first.chosen]["norm"] == "player a"
    second = obs[1]
    assert second.pool.iloc[second.chosen]["norm"] == "player c"

def test_picks_with_no_adp_row_are_dropped(tmp_path):
    # "Ghost" is not in historic_adp, so pick 4 produces no observation.
    obs = build_observations(_seed(tmp_path))
    assert len(obs) == 3
    assert all(o.overall_pick != 4 for o in obs)

def test_roster_counts_reflect_that_managers_prior_picks_only(tmp_path):
    obs = build_observations(_seed(tmp_path))
    third = obs[2]                       # dan's second pick, overall 3
    assert third.manager == "dan"
    assert third.roster == {"WR": 1}

def test_recent_positions_are_newest_first(tmp_path):
    obs = build_observations(_seed(tmp_path))
    assert obs[2].recent == ["WR", "RB"]

def _seed_duplicate_adp(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    # "Player A" and "Player A." both normalize to "player a"/RB -- a
    # duplicate (norm, position) pair within one season's historic_adp,
    # which is reachable from real data since parse_adp/import_league
    # build historic_adp with no uniqueness guarantee.
    write_table(conn, "draft_picks", pd.DataFrame([
        {"season": 2025, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 11, "player_name": "Player A",
         "position": "RB", "nfl_team": "DET", "keeper": False},
        {"season": 2025, "overall_pick": 2, "round": 1, "round_pick": 2,
         "team_id": 2, "espn_player_id": 13, "player_name": "Player B",
         "position": "RB", "nfl_team": "CHI", "keeper": False},
        {"season": 2025, "overall_pick": 3, "round": 2, "round_pick": 1,
         "team_id": 1, "espn_player_id": 12, "player_name": "Player C",
         "position": "WR", "nfl_team": "GB", "keeper": False},
    ]))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2025, "team_id": 2, "manager": "dan", "slot": 2},
    ]))
    write_table(conn, "historic_adp", pd.DataFrame([
        {"season": 2025, "adp_name": "Player A", "position": "RB", "adp_rank": 1},
        {"season": 2025, "adp_name": "Player A.", "position": "RB", "adp_rank": 7},
        {"season": 2025, "adp_name": "Player B", "position": "RB", "adp_rank": 2},
        {"season": 2025, "adp_name": "Player C", "position": "WR", "adp_rank": 3},
    ]))
    return conn

def test_duplicate_adp_rows_collapse_to_one_pool_entry(tmp_path):
    obs = build_observations(_seed_duplicate_adp(tmp_path))
    # 3 picks against a 3-player deduped pool (A, B, C) -- not 4, which is
    # what a naive read of historic_adp (with the "Player A." duplicate
    # still present) would give for the first pick.
    assert [len(o.pool) for o in obs] == [3, 2, 1]
    for o in obs:
        assert not o.pool.duplicated(subset=["norm", "position"]).any()

def _seed_two_seasons(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "draft_picks", pd.DataFrame([
        # 2024: worthy takes an RB, dan takes a WR.
        {"season": 2024, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 21, "player_name": "Player X",
         "position": "RB", "nfl_team": "DAL", "keeper": False},
        {"season": 2024, "overall_pick": 2, "round": 1, "round_pick": 2,
         "team_id": 2, "espn_player_id": 22, "player_name": "Player Y",
         "position": "WR", "nfl_team": "SEA", "keeper": False},
        # 2025: same two managers, fresh draft. overall_pick restarts at 1.
        {"season": 2025, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 11, "player_name": "Player A",
         "position": "RB", "nfl_team": "DET", "keeper": False},
        {"season": 2025, "overall_pick": 2, "round": 1, "round_pick": 2,
         "team_id": 2, "espn_player_id": 12, "player_name": "Player C",
         "position": "WR", "nfl_team": "GB", "keeper": False},
    ]))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2024, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2024, "team_id": 2, "manager": "dan", "slot": 2},
        {"season": 2025, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2025, "team_id": 2, "manager": "dan", "slot": 2},
    ]))
    write_table(conn, "historic_adp", pd.DataFrame([
        {"season": 2024, "adp_name": "Player X", "position": "RB", "adp_rank": 1},
        {"season": 2024, "adp_name": "Player Y", "position": "WR", "adp_rank": 2},
        {"season": 2025, "adp_name": "Player A", "position": "RB", "adp_rank": 1},
        {"season": 2025, "adp_name": "Player C", "position": "WR", "adp_rank": 2},
    ]))
    return conn

def test_season_boundary_resets_pool_roster_and_recent(tmp_path):
    obs = build_observations(_seed_two_seasons(tmp_path))
    # groupby("season") visits seasons in ascending order: 2024, then 2025.
    assert [o.season for o in obs] == [2024, 2024, 2025, 2025]
    first_2025 = obs[2]
    assert first_2025.overall_pick == 1
    assert len(first_2025.pool) == 2   # fresh 2025 pool, no 2024 leftovers
    assert first_2025.roster == {}     # worthy's 2024 RB pick doesn't carry over
    assert first_2025.recent == []     # nothing carries across the season boundary

def _settings():
    return league.default_settings()

def test_feature_matrix_shape_and_column_order(tmp_path):
    obs = build_observations(_seed(tmp_path))[0]
    X = feature_matrix(obs, _settings())
    assert X.shape == (len(obs.pool), len(FEATURE_NAMES))
    assert FEATURE_NAMES[0] == "reach" and FEATURE_NAMES[1] == "fall"

def test_reach_and_fall_are_nonnegative_and_mutually_exclusive(tmp_path):
    obs = build_observations(_seed(tmp_path))[2]     # overall pick 3
    X = feature_matrix(obs, _settings())
    reach, fall = X[:, 0], X[:, 1]
    assert (reach >= 0).all() and (fall >= 0).all()
    assert ((reach == 0) | (fall == 0)).all()

def test_reach_measures_rounds_of_reach_required(tmp_path):
    obs = build_observations(_seed(tmp_path))[0]      # overall pick 1, 8 teams
    X = feature_matrix(obs, _settings())
    ranks = obs.pool["adp_rank"].to_numpy()
    expected = np.maximum(0, ranks - obs.overall_pick) / 8
    assert np.allclose(X[:, 0], expected)

def test_need_is_one_while_short_of_a_starter(tmp_path):
    obs = build_observations(_seed(tmp_path))[2]      # dan has 1 WR, 0 RB, 0 TE
    X = feature_matrix(obs, _settings())
    need = X[:, FEATURE_NAMES.index("need")]
    positions = obs.pool["position"].tolist()
    # obs[2]'s pool is Player B (RB, rank 2) and Player D (TE, rank 4) -- the
    # two players still on the board for dan's second pick. Dan has 0 RB and
    # 0 TE so far, and the default league starts 2 RB / 1 TE, so both are
    # short of a starter and should read as needs.
    assert positions == ["RB", "TE"]
    assert need[positions.index("RB")] == 1.0
    assert need[positions.index("TE")] == 1.0

def test_run_counts_same_position_picks_in_the_recent_window(tmp_path):
    obs = build_observations(_seed(tmp_path))[2]      # recent == ["WR", "RB"]
    X = feature_matrix(obs, _settings())
    run = X[:, FEATURE_NAMES.index("run")]
    positions = obs.pool["position"].tolist()
    assert run[positions.index("RB")] == 1.0 / 5

def _pool(rows):
    return pd.DataFrame(rows, columns=["norm", "position", "adp_rank"])

@pytest.mark.parametrize("pick,early", [(8, True), (24, True), (25, False)])
def test_qb_and_te_early_flags_the_round_boundary(pick, early):
    # 8-team league, EARLY_ROUNDS=3: pick 8 is round 1, pick 24 is round 3
    # (both early); pick 25 is round 4, the first NOT-early pick.
    pool = _pool([("qb1", "QB", 1), ("rb1", "RB", 2), ("te1", "TE", 3)])
    obs = PickObservation(season=2025, overall_pick=pick, manager="x",
                           chosen=0, pool=pool, roster={}, recent=[])
    X = feature_matrix(obs, _settings())
    qb_early = X[:, FEATURE_NAMES.index("qb_early")]
    te_early = X[:, FEATURE_NAMES.index("te_early")]
    positions = pool["position"].tolist()
    expected = 1.0 if early else 0.0
    assert qb_early[positions.index("QB")] == expected
    assert te_early[positions.index("TE")] == expected
    # The interaction only ever fires on QB/RB rows respectively -- a non-QB
    # row must never carry qb_early, and vice versa for te_early.
    assert qb_early[positions.index("RB")] == 0.0
    assert te_early[positions.index("RB")] == 0.0
    assert qb_early[positions.index("TE")] == 0.0
    assert te_early[positions.index("QB")] == 0.0

def test_need_is_zero_once_a_starter_slot_is_already_full(tmp_path):
    pool = _pool([("rb1", "RB", 1)])
    settings = _settings()  # default league starts 2 RB
    full = PickObservation(season=2025, overall_pick=1, manager="x", chosen=0,
                            pool=pool, roster={"RB": 2}, recent=[])
    short = PickObservation(season=2025, overall_pick=1, manager="x", chosen=0,
                             pool=pool, roster={"RB": 1}, recent=[])
    need_full = feature_matrix(full, settings)[:, FEATURE_NAMES.index("need")]
    need_short = feature_matrix(short, settings)[:, FEATURE_NAMES.index("need")]
    assert need_full[0] == 0.0
    assert need_short[0] == 1.0

def test_need_is_zero_for_a_position_absent_from_starters():
    # A league that doesn't start K at all -- e.g. a superflex-style config
    # from `from_espn` with no kicker slot mapped.
    settings = league.LeagueSettings(
        season=0, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "DST": 1},
        flex_slots=2, bench=5, scoring={}, draft_type="SNAKE",
    )
    pool = _pool([("k1", "K", 1)])
    obs = PickObservation(season=2025, overall_pick=1, manager="x", chosen=0,
                           pool=pool, roster={}, recent=[])
    need = feature_matrix(obs, settings)[:, FEATURE_NAMES.index("need")]
    assert need[0] == 0.0

def test_feature_matrix_of_an_empty_pool_has_the_right_shape():
    empty_pool = _pool([])
    obs = PickObservation(season=2025, overall_pick=1, manager="x", chosen=0,
                           pool=empty_pool, roster={}, recent=[])
    X = feature_matrix(obs, _settings())
    assert X.shape == (0, len(FEATURE_NAMES))

from scoring.draft_model import fit, log_likelihood, neg_log_likelihood

def _synthetic(beta_true, n_choices=40, pool=20, seed=0):
    """Generate choices from a known beta so the fit can be checked for recovery."""
    rng = np.random.default_rng(seed)
    X_list, chosen = [], []
    for _ in range(n_choices):
        X = rng.normal(size=(pool, len(beta_true)))
        p = np.exp(X @ beta_true)
        p = p / p.sum()
        X_list.append(X)
        chosen.append(int(rng.choice(pool, p=p)))
    return X_list, chosen

def test_gradient_matches_finite_differences():
    beta = np.array([0.3, -0.7, 0.1])
    X_list, chosen = _synthetic(beta, n_choices=5, pool=6, seed=1)
    point = np.array([0.1, 0.2, -0.3])
    _, grad = neg_log_likelihood(point, X_list, chosen)
    eps = 1e-6
    for i in range(len(point)):
        bumped = point.copy()
        bumped[i] += eps
        numeric = (neg_log_likelihood(bumped, X_list, chosen)[0]
                   - neg_log_likelihood(point, X_list, chosen)[0]) / eps
        assert abs(numeric - grad[i]) < 1e-4

def test_fit_recovers_known_beta():
    beta_true = np.array([1.5, -1.0, 0.5])
    X_list, chosen = _synthetic(beta_true, n_choices=3000, pool=15, seed=2)
    beta_hat = fit(X_list, chosen)
    assert np.allclose(beta_hat, beta_true, atol=0.2)

def test_shrinkage_pulls_a_thin_fit_toward_the_prior():
    beta_true = np.array([1.5, -1.0, 0.5])
    prior = np.array([0.0, 0.0, 0.0])
    X_list, chosen = _synthetic(beta_true, n_choices=6, pool=15, seed=3)
    loose = fit(X_list, chosen, prior=prior, lam=0.01)
    tight = fit(X_list, chosen, prior=prior, lam=100.0)
    assert np.linalg.norm(tight - prior) < np.linalg.norm(loose - prior)

def test_log_likelihood_of_uniform_beta_is_pool_entropy():
    X_list, chosen = _synthetic(np.array([1.0, 0.0, 0.0]), n_choices=4, pool=10, seed=4)
    zero = np.zeros(3)
    assert abs(log_likelihood(zero, X_list, chosen) - 4 * np.log(1 / 10)) < 1e-9

def test_gradient_matches_finite_differences_with_a_prior_penalty():
    # test_gradient_matches_finite_differences above only ever calls
    # neg_log_likelihood with prior=None, lam=0.0, so it never exercises the
    # penalty term's contribution to the value or the gradient. Repeat the
    # same check with both supplied so a value/gradient mismatch in the
    # penalty (the exact bug class the module warns about) would be caught.
    beta = np.array([0.3, -0.7, 0.1])
    X_list, chosen = _synthetic(beta, n_choices=5, pool=6, seed=1)
    prior = np.array([0.2, 0.1, -0.4])
    lam = 2.5
    point = np.array([0.1, 0.2, -0.3])
    _, grad = neg_log_likelihood(point, X_list, chosen, prior=prior, lam=lam)
    eps = 1e-6
    for i in range(len(point)):
        bumped = point.copy()
        bumped[i] += eps
        numeric = (neg_log_likelihood(bumped, X_list, chosen, prior=prior, lam=lam)[0]
                   - neg_log_likelihood(point, X_list, chosen, prior=prior, lam=lam)[0]) / eps
        assert abs(numeric - grad[i]) < 1e-4

def test_fit_on_an_empty_choice_set_returns_without_calling_the_optimizer(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("minimize should not be called for an empty X_list")
    monkeypatch.setattr("scoring.draft_model.minimize", _boom)
    assert np.array_equal(fit([], []), np.zeros(len(FEATURE_NAMES)))
    prior = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(fit([], [], prior=prior, lam=1.0), prior)

def test_fit_warns_and_still_returns_x_when_the_optimizer_fails_to_converge(monkeypatch):
    # Thin per-manager samples (Task 8's regime -- ~105 picks against 12
    # parameters inside a lambda sweep) are exactly where L-BFGS-B can hit
    # ABNORMAL_TERMINATION_IN_LNSRCH. A silent non-convergence would return a
    # plausible-looking but wrong beta with no signal, which is the failure
    # mode this test guards against. Faked via monkeypatch since a real
    # reliably-unconvergeable L-BFGS-B call isn't practical to construct.
    class _FakeResult:
        success = False
        message = "ABNORMAL_TERMINATION_IN_LNSRCH"
        x = np.array([9.0, 9.0, 9.0])

    monkeypatch.setattr("scoring.draft_model.minimize", lambda *a, **k: _FakeResult())
    X_list, chosen = _synthetic(np.array([1.0, 0.0, 0.0]), n_choices=3, pool=5, seed=5)
    with pytest.warns(RuntimeWarning, match="did not converge") as record:
        beta_hat = fit(X_list, chosen)
    assert str(len(X_list)) in str(record[0].message)
    assert np.array_equal(beta_hat, _FakeResult.x)

from scoring.draft_model import (backtest, describe, fit_all, select_lambda,
                                 write_profiles)

def _seed_many(tmp_path, seasons=(2023, 2024, 2025)):
    """Two managers with opposite tastes, repeated across seasons.

    `early` always takes the best RB available; `late` always takes the best
    WR. A fitted model should separate them on the position dummies.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    names = [f"Player {i}" for i in range(1, 21)]
    positions = ["RB" if i % 2 else "WR" for i in range(1, 21)]
    picks, adp = [], []
    for season in seasons:
        for i, (name, pos) in enumerate(zip(names, positions), start=1):
            adp.append({"season": season, "adp_name": name,
                        "position": pos, "adp_rank": i})
        taken = set()
        for pick_no in range(1, 9):
            manager_pos = "RB" if pick_no % 2 else "WR"
            team_id = 1 if pick_no % 2 else 2
            choice = next(n for n, p in zip(names, positions)
                          if p == manager_pos and n not in taken)
            taken.add(choice)
            picks.append({"season": season, "overall_pick": pick_no,
                          "round": (pick_no - 1) // 2 + 1,
                          "round_pick": (pick_no - 1) % 2 + 1,
                          "team_id": team_id, "espn_player_id": pick_no,
                          "player_name": choice,
                          "position": manager_pos, "nfl_team": "DET",
                          "keeper": False})
    write_table(conn, "draft_picks", pd.DataFrame(picks))
    write_table(conn, "draft_teams", pd.DataFrame(
        [{"season": s, "team_id": t, "manager": m, "slot": t}
         for s in seasons for t, m in ((1, "rbguy"), (2, "wrguy"))]))
    write_table(conn, "historic_adp", pd.DataFrame(adp))
    return conn

def test_select_lambda_returns_a_value_from_the_grid():
    beta_true = np.array([1.0, -0.5, 0.2])
    X_list, chosen = _synthetic(beta_true, n_choices=60, pool=10, seed=5)
    seasons = [2023] * 20 + [2024] * 20 + [2025] * 20
    lam = select_lambda(X_list, chosen, seasons, prior=np.zeros(3),
                        grid=[0.01, 1.0, 100.0])
    assert lam in (0.01, 1.0, 100.0)

def test_fit_all_separates_managers_with_opposite_tastes(tmp_path):
    fits = fit_all(_seed_many(tmp_path))
    assert set(fits) >= {"rbguy", "wrguy", "__pooled__"}
    rb_idx = FEATURE_NAMES.index("pos_RB")
    wr_idx = FEATURE_NAMES.index("pos_WR")
    assert fits["rbguy"][rb_idx] - fits["rbguy"][wr_idx] > \
           fits["wrguy"][rb_idx] - fits["wrguy"][wr_idx]

def test_backtest_reports_accuracy_against_an_adp_baseline(tmp_path):
    report = backtest(_seed_many(tmp_path))
    assert report["holdout_season"] == 2025
    assert 0.0 <= report["top1"] <= 1.0
    assert 0.0 <= report["top5"] <= 1.0
    assert isinstance(report["beats_adp"], bool)

def test_beats_adp_is_false_for_a_model_no_better_than_the_market(tmp_path, monkeypatch):
    # _seed_many's draft is chalk: every pick is the market's next player by
    # ADP rank (see the fixture docstring -- pick order and ADP rank order
    # are constructed to coincide). A real ADP baseline should therefore
    # score that holdout season well. A model forced to predict *uniformly*
    # over the pool (beta == 0 makes every softmax uniform, regardless of X)
    # carries no market information and must lose to that baseline -- this
    # is the check the brief's original uniform-distribution "baseline"
    # could never fail, since a uniform baseline is beaten by almost
    # anything. It guards the fix in backtest() that replaced the uniform
    # ADP distribution with a real one.
    monkeypatch.setattr("scoring.draft_model.fit",
                        lambda *a, **k: np.zeros(len(FEATURE_NAMES)))
    report = backtest(_seed_many(tmp_path))
    assert report["beats_adp"] is False

def test_write_profiles_marks_thin_managers_as_pooled(tmp_path):
    conn = _seed_many(tmp_path, seasons=(2025,))     # 4 picks each: very thin
    profiles = write_profiles(conn)
    assert set(profiles.columns) == {
        "manager", "feature", "value", "pooled_value", "n_picks",
        "heldout_gain", "uses_personal", "summary"}
    assert (profiles["n_picks"] == 4).all()
    from pipeline.db import read_table
    assert not read_table(conn, "manager_profiles").empty

def test_describe_names_the_strongest_deviations():
    pooled = np.zeros(len(FEATURE_NAMES))
    beta = pooled.copy()
    beta[FEATURE_NAMES.index("reach")] = -2.0
    text = describe(beta, pooled)
    assert "reach" in text.lower()


def _seed_with_espn(tmp_path):
    """One season, two players, ESPN and FFC disagreeing on the order."""
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "draft_picks", pd.DataFrame([
        {"season": 2025, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 11, "player_name": "Player A",
         "position": "RB", "nfl_team": "DET", "keeper": False},
        {"season": 2025, "overall_pick": 2, "round": 1, "round_pick": 2,
         "team_id": 2, "espn_player_id": 12, "player_name": "Player B",
         "position": "WR", "nfl_team": "GB", "keeper": False},
    ]))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2025, "team_id": 2, "manager": "dan", "slot": 2},
    ]))
    write_table(conn, "historic_adp", pd.DataFrame([
        {"season": 2025, "adp_name": "Player A", "position": "RB",
         "team": "DET", "adp_rank": 1},
        {"season": 2025, "adp_name": "Player B", "position": "WR",
         "team": "GB", "adp_rank": 2},
    ]))
    # ESPN reverses them.
    write_table(conn, "historic_espn", pd.DataFrame([
        {"season": 2025, "espn_name": "Player B", "position": "WR",
         "espn_rank": 1, "adp_usable": True},
        {"season": 2025, "espn_name": "Player A", "position": "RB",
         "espn_rank": 2, "adp_usable": True},
    ]))
    return conn


def test_pool_market_rank_prefers_espn_over_ffc(tmp_path):
    obs = build_observations(_seed_with_espn(tmp_path))
    pool = obs[0].pool.set_index("norm")
    assert pool.loc["player a"]["adp_rank"] == 1        # FFC, unchanged
    assert pool.loc["player a"]["market_rank"] == 2     # ESPN's view
    assert pool.loc["player b"]["market_rank"] == 1


def test_pool_falls_back_to_ffc_when_espn_lacks_the_player(tmp_path):
    from pipeline.db import write_table as wt
    conn = _seed_with_espn(tmp_path)
    wt(conn, "historic_espn", pd.DataFrame([
        {"season": 2025, "espn_name": "Player B", "position": "WR",
         "espn_rank": 1, "adp_usable": True}]))
    pool = build_observations(conn)[0].pool.set_index("norm")
    assert pool.loc["player a"]["market_rank"] == 1     # fell back to adp_rank


def test_pool_carries_player_attributes_with_neutral_defaults(tmp_path):
    obs = build_observations(_seed_with_espn(tmp_path))
    pool = obs[0].pool
    for col in ("market_rank", "hype", "age", "ppg_std", "missed_rate",
                "no_track_record", "trend"):
        assert col in pool.columns
    # No `weekly` table at all, so nobody has a track record.
    assert pool["no_track_record"].all()
    assert (pool["ppg_std"] == 0.0).all()
    assert (pool["trend"] == 0.0).all()


def test_pool_hype_is_market_ahead_of_production(tmp_path):
    conn = _seed_with_espn(tmp_path)
    write_table(conn, "weekly", pd.DataFrame([
        # Player B produced far less than the market's view of him.
        {"player_id": "b", "player_display_name": "Player B", "position": "WR",
         "recent_team": "GB", "opponent_team": "DET", "season": 2024, "week": w,
         "receptions": 1, "receiving_yards": 5, "targets": 2, "carries": 0}
        for w in range(1, 18)] + [
        {"player_id": "a", "player_display_name": "Player A", "position": "RB",
         "recent_team": "DET", "opponent_team": "GB", "season": 2024, "week": w,
         "receptions": 9, "receiving_yards": 90, "targets": 11, "carries": 5}
        for w in range(1, 18)]))
    pool = build_observations(conn)[0].pool.set_index("norm")
    # B: market_rank 1, prod_rank 2 -> hype +1. A: market 2, prod 1 -> -1.
    assert pool.loc["player b"]["hype"] > pool.loc["player a"]["hype"]


def test_pool_treats_a_name_collision_in_attributes_as_no_track_record(tmp_path):
    """`adp_match_key` carries no team for a non-DST position, so two
    different past players who happen to share (position, normalized name)
    collide on the same `attributes_as_of` key -- confirmed reachable with
    real NFL data, common names repeat across eras/rosters. Guessing which
    one's numbers belong to the player actually drafted would put a wrong
    value on a `no_track_record: False` row; left-joining the pool against a
    key that appears twice would also duplicate that pool row, breaking
    "one pick removes exactly one pool row"."""
    conn = _seed_with_espn(tmp_path)
    write_table(conn, "weekly", pd.DataFrame(
        [{"player_id": "p1", "player_display_name": "Player A", "position": "RB",
          "recent_team": "DET", "opponent_team": "GB", "season": 2024, "week": w,
          "receptions": 1, "receiving_yards": 5, "targets": 2, "carries": 10}
         for w in range(1, 18)] +
        [{"player_id": "p2", "player_display_name": "Player A", "position": "RB",
          "recent_team": "CHI", "opponent_team": "GB", "season": 2024, "week": w,
          "receptions": 1, "receiving_yards": 5, "targets": 2, "carries": 10}
         for w in range(1, 18)]))
    obs = build_observations(conn)
    pool = obs[0].pool
    assert len(pool) == 2                        # not duplicated by the merge
    row = pool.set_index("norm").loc["player a"]
    assert row["no_track_record"] == True
