import numpy as np
import pandas as pd
import pytest
from pipeline.db import get_conn, write_table
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
