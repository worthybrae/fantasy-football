import pandas as pd
from scoring.similarity import player_season_features, find_twins, value_neighbors

def _wk(pid, name, season, games, rec, yds, tgt, team="AAA", pos="WR"):
    """games identical weekly lines for one player-season."""
    return [{"player_id": pid, "player_display_name": name, "position": pos,
             "recent_team": team, "opponent_team": "ZZZ", "season": season,
             "week": w, "receptions": rec, "receiving_yards": yds,
             "targets": tgt, "carries": 0}
            for w in range(1, games + 1)]

def test_features_shares_and_ppg():
    # one player is the whole team: target_share == 1.0
    wk = pd.DataFrame(_wk("p1", "A", 2025, 4, rec=5, yds=50, tgt=8))
    f = player_season_features(wk).iloc[0]
    assert f["games"] == 4 and f["target_share"] == 1.0
    assert abs(f["ppg"] - 10.0) < 1e-9          # 5 + 50*0.1
    assert abs(f["yards_per_opp"] - 50 / 8) < 1e-9

def test_clone_is_top_twin_with_100():
    rows = (_wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9, team="AAA")
            + _wk("clone", "Clone", 2023, 10, rec=6, yds=80, tgt=9, team="BBB")
            + _wk("other", "Other", 2024, 10, rec=2, yds=20, tgt=3, team="CCC"))
    out = find_twins(pd.DataFrame(rows), "me")
    assert out["mode"] == "stat_twins" and out["target_season"] == 2025
    top = out["players"][0]
    assert top["player_id"] == "clone" and top["similarity"] == 100.0

def test_own_seasons_excluded():
    rows = _wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9) + _wk("me", "Me", 2024, 10, rec=6, yds=80, tgt=9) \
           + _wk("x", "X", 2024, 10, rec=5, yds=70, tgt=8, team="BBB")
    out = find_twins(pd.DataFrame(rows), "me")
    assert all(p["player_id"] != "me" for p in out["players"])

def test_next_ppg():
    rows = (_wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9)
            + _wk("x", "X", 2024, 10, rec=6, yds=80, tgt=9, team="BBB")
            + _wk("x", "X", 2025, 10, rec=10, yds=100, tgt=12, team="BBB"))
    out = find_twins(pd.DataFrame(rows), "me")
    x2024 = next(p for p in out["players"] if p["season"] == 2024)
    assert abs(x2024["next_ppg"] - 20.0) < 1e-9  # X's 2025: 10 rec + 100*0.1

def test_min_games_filter_returns_none():
    wk = pd.DataFrame(_wk("me", "Me", 2025, 2, rec=6, yds=80, tgt=9))
    assert find_twins(wk, "me") is None

def test_value_neighbors():
    board = pd.DataFrame({
        "player_id": ["a", "b", "c", "d"], "name": ["A", "B", "C", "D"],
        "position": ["WR", "WR", "WR", "RB"],
        "vor": [10.0, 9.0, 1.0, 9.5], "rank": [1, 2, 3, 4],
        "adp": [5.0, 8.0, 90.0, 6.0]})
    out = value_neighbors(board, "a", top_n=2)
    ids = [p["player_id"] for p in out["players"]]
    assert out["mode"] == "value_neighbors"
    assert ids == ["b", "c"]  # same position only, nearest vor first, self excluded
