import pandas as pd
from pipeline.db import get_conn, write_table
from scoring.draft_model import build_observations

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
