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
    # clone needs a 2024 season on file: comps without a next season are
    # excluded outright (see test_twins_without_next_season_excluded).
    rows = (_wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9, team="AAA")
            + _wk("clone", "Clone", 2023, 10, rec=6, yds=80, tgt=9, team="BBB")
            + _wk("clone", "Clone", 2024, 10, rec=4, yds=50, tgt=6, team="BBB")
            + _wk("other", "Other", 2024, 10, rec=2, yds=20, tgt=3, team="CCC"))
    out = find_twins(pd.DataFrame(rows), "me")
    assert out["mode"] == "stat_twins" and out["target_season"] == 2025
    top = out["players"][0]
    assert top["player_id"] == "clone" and top["similarity"] == 100.0

def test_own_seasons_excluded():
    # x gets a 2025 season so its 2024 comp survives the next-season filter
    # and the players list is non-empty (an empty list would pass trivially).
    rows = _wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9) + _wk("me", "Me", 2024, 10, rec=6, yds=80, tgt=9) \
           + _wk("x", "X", 2024, 10, rec=5, yds=70, tgt=8, team="BBB") \
           + _wk("x", "X", 2025, 10, rec=5, yds=70, tgt=8, team="BBB")
    out = find_twins(pd.DataFrame(rows), "me")
    assert len(out["players"]) > 0
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
        "market_rank": [5.0, 8.0, 90.0, 6.0]})
    out = value_neighbors(board, "a", top_n=2)
    ids = [p["player_id"] for p in out["players"]]
    assert out["mode"] == "value_neighbors"
    assert ids == ["b", "c"]  # same position only, nearest vor first, self excluded
    assert out["players"][0]["market_rank"] == 8.0
    assert "adp" not in out["players"][0]

def test_next_ppg_low_games_included():
    # Target player in 2024 matches "comp" in 2024
    # "comp" also has 2025 with 2 games (below MIN_GAMES threshold)
    # next_ppg should show 2025's ppg even though games < MIN_GAMES
    rows = (_wk("me", "Me", 2024, 10, rec=6, yds=80, tgt=9)
            + _wk("comp", "Comp", 2024, 10, rec=6, yds=80, tgt=9, team="BBB")
            + _wk("comp", "Comp", 2025, 2, rec=10, yds=100, tgt=12, team="BBB"))
    out = find_twins(pd.DataFrame(rows), "me")
    comp_2024 = next(p for p in out["players"] if p["player_id"] == "comp" and p["season"] == 2024)
    # comp's next season (2025) has ppg = 10 rec + 100*0.1 = 20.0 despite 2 < MIN_GAMES
    assert abs(comp_2024["next_ppg"] - 20.0) < 1e-9

def test_target_share_with_trade():
    # Player plays weeks 1-4 for AAA and weeks 5-8 for BBB
    # Team totals should sum across both teams
    rows = (
        [{"player_id": "me", "player_display_name": "Me", "position": "WR",
          "recent_team": "AAA", "opponent_team": "ZZZ", "season": 2025,
          "week": w, "receptions": 5, "receiving_yards": 50, "targets": 8, "carries": 0}
         for w in range(1, 5)]
        + [{"player_id": "me", "player_display_name": "Me", "position": "WR",
            "recent_team": "BBB", "opponent_team": "ZZZ", "season": 2025,
            "week": w, "receptions": 5, "receiving_yards": 50, "targets": 8, "carries": 0}
           for w in range(5, 9)]
        + [{"player_id": "other_aaa", "player_display_name": "OtherAAA", "position": "WR",
            "recent_team": "AAA", "opponent_team": "ZZZ", "season": 2025,
            "week": w, "receptions": 2, "receiving_yards": 20, "targets": 4, "carries": 0}
           for w in range(1, 5)]
        + [{"player_id": "other_bbb", "player_display_name": "OtherBBB", "position": "WR",
            "recent_team": "BBB", "opponent_team": "ZZZ", "season": 2025,
            "week": w, "receptions": 2, "receiving_yards": 20, "targets": 4, "carries": 0}
           for w in range(5, 9)]
    )
    f = player_season_features(pd.DataFrame(rows))
    me = f[f["player_id"] == "me"].iloc[0]
    # "me": 8 weeks * 8 targets/week = 64 targets
    # AAA season: (me 4 weeks * 8) + (other_aaa 4 weeks * 4) = 32 + 16 = 48
    # BBB season: (me 4 weeks * 8) + (other_bbb 4 weeks * 4) = 32 + 16 = 48
    # "me" team_targets should count both: 48 + 48 = 96
    assert me["games"] == 8
    assert me["targets"] == 64
    assert me["team_targets"] == 96
    target_share = me["targets"] / me["team_targets"]
    assert abs(target_share - 64/96) < 1e-9
    assert target_share <= 1.0

def test_twins_without_next_season_excluded():
    # The whole point of a comp is the "what happened next" trend signal --
    # a comp with no following season in the data (retired, injured out of
    # the league, or the season simply hasn't been played yet) shows a
    # dangling arrow instead of a signal, so it is dropped entirely.
    rows = (_wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9)
            + _wk("nonext", "NoNext", 2024, 10, rec=6, yds=80, tgt=9, team="BBB")
            + _wk("hasnext", "HasNext", 2023, 10, rec=5, yds=70, tgt=8, team="CCC")
            + _wk("hasnext", "HasNext", 2024, 10, rec=7, yds=90, tgt=10, team="CCC"))
    out = find_twins(pd.DataFrame(rows), "me")
    ids = [(p["player_id"], p["season"]) for p in out["players"]]
    assert ("nonext", 2024) not in ids       # perfect match, but no next season
    assert ("hasnext", 2023) in ids
    assert all(p["next_ppg"] is not None for p in out["players"])

def _players(*rows):
    return pd.DataFrame(list(rows), columns=["gsis_id", "display_name", "birth_date", "rookie_season"])

def test_twins_age_filter_exact_match():
    # "me" is 25 during 2025 (born Jan 2000). "sameage" was also 25 during
    # its 2023 comp season; "offage" was 33. Only the same-age comp remains.
    rows = (_wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9)
            + _wk("sameage", "Same Age", 2023, 10, rec=6, yds=80, tgt=9, team="BBB")
            + _wk("sameage", "Same Age", 2024, 10, rec=5, yds=60, tgt=7, team="BBB")
            + _wk("offage", "Off Age", 2023, 10, rec=6, yds=80, tgt=9, team="CCC")
            + _wk("offage", "Off Age", 2024, 10, rec=5, yds=60, tgt=7, team="CCC"))
    players = _players(("me", "Me", "2000-01-01", 2022),
                       ("sameage", "Same Age", "1998-01-01", 2020),
                       ("offage", "Off Age", "1990-01-01", 2012))
    out = find_twins(pd.DataFrame(rows), "me", players=players)
    ids = {p["player_id"] for p in out["players"]}
    assert ids == {"sameage"}
    assert out["target_age"] == 25
    assert out["players"][0]["age"] == 25

def test_twins_age_filter_skipped_without_target_birthdate():
    # Target absent from the players table: age matching degrades to the
    # plain stat-similarity pool instead of returning nothing.
    rows = (_wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9)
            + _wk("comp", "Comp", 2023, 10, rec=6, yds=80, tgt=9, team="BBB")
            + _wk("comp", "Comp", 2024, 10, rec=5, yds=60, tgt=7, team="BBB"))
    players = _players(("comp", "Comp", "1998-01-01", 2020))
    out = find_twins(pd.DataFrame(rows), "me", players=players)
    assert {p["player_id"] for p in out["players"]} == {"comp"}
    assert out["target_age"] is None

def test_age_counts_full_years_at_september_first():
    from scoring.similarity import _age_in_season
    assert _age_in_season("2000-08-15", 2025) == 25   # birthday before opening week
    assert _age_in_season("2000-09-15", 2025) == 24   # birthday after Sept 1
    assert _age_in_season(None, 2025) is None


def test_features_are_scored_under_the_leagues_rules():
    """`ppg` here is not a display number: it is one of the seven FEATURES the
    twin distance is computed over AND what `find_twins` reports as
    `next_ppg`. Priced in fixed full PPR it picked PPR comparables for a
    half-PPR league and attached a PPR forecast to them, with nothing on the
    page saying so."""
    wk = pd.DataFrame(_wk("p1", "A", 2025, 4, rec=5, yds=50, tgt=8))
    assert player_season_features(wk).iloc[0]["ppg"] == 10.0          # 5 + 5.0
    half = player_season_features(wk, {"receptions": 0.5, "receiving_yards": 0.1})
    assert half.iloc[0]["ppg"] == 7.5                                 # 2.5 + 5.0
    std = player_season_features(wk, {"receiving_yards": 0.1})
    assert std.iloc[0]["ppg"] == 5.0
    # Counting columns are counts, not points -- untouched by any rule set.
    for frame in (half, std):
        assert frame.iloc[0]["receptions"] == 20
        assert frame.iloc[0]["target_share"] == 1.0


def test_the_leagues_rules_can_change_which_twin_is_closest():
    """The match itself moves, not just the number printed beside it.

    The target catches 6 for 80 on 12 targets. `Catcher` (7 for 40) is the
    closer comp in full PPR -- 11.0 ppg against the target's 14.0, and almost
    the same catches a game. `Runner` (1 for 90) is the closer comp with
    receptions worth nothing -- 9.0 against 8.0, and a similar yards per
    opportunity -- even though he catches five fewer balls a game. Both
    comps carry an identical following season, so `next_ppg` follows.

    Verified against real data too: Ja'Marr Chase's five comps on
    data/nfl.duckdb are Keenan Allen / Michael Thomas / Diontae Johnson /
    Davante Adams / DeAndre Hopkins in PPR and swap in Justin Jefferson at
    half and Justin Jefferson + Michael Pittman at standard.
    """
    rows = (_wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=12, team="AAA")
            + _wk("catcher", "Catcher", 2023, 10, rec=7, yds=40, tgt=12, team="BBB")
            + _wk("catcher", "Catcher", 2024, 10, rec=7, yds=40, tgt=12, team="BBB")
            + _wk("runner", "Runner", 2023, 10, rec=1, yds=90, tgt=12, team="CCC")
            + _wk("runner", "Runner", 2024, 10, rec=1, yds=90, tgt=12, team="CCC"))
    wk = pd.DataFrame(rows)
    ppr = find_twins(wk, "me")["players"][0]
    assert ppr["name"] == "Catcher"
    assert ppr["ppg"] == 11.0 and ppr["next_ppg"] == 11.0            # 7 + 4.0
    std = find_twins(wk, "me", rules={"receiving_yards": 0.1})["players"][0]
    assert std["name"] == "Runner"
    assert std["ppg"] == 9.0 and std["next_ppg"] == 9.0              # 90 x 0.1


def test_find_twins_ignores_rules_when_the_features_frame_is_supplied():
    """Documented contract, pinned: `rules` only ever prices the frame this
    function would otherwise build. A caller handing over `season_features`
    (which build_profile does, off the rules-keyed profile cache) has already
    priced it, and a second, disagreeing rule set here would silently rank the
    comps on one scale and report them on another."""
    rows = (_wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9, team="AAA")
            + _wk("clone", "Clone", 2023, 10, rec=6, yds=80, tgt=9, team="BBB")
            + _wk("clone", "Clone", 2024, 10, rec=4, yds=50, tgt=6, team="BBB"))
    wk = pd.DataFrame(rows)
    feats = player_season_features(wk)                       # full PPR
    supplied = find_twins(wk, "me", season_features=feats,
                          rules={"receiving_yards": 0.1})["players"][0]
    assert supplied["ppg"] == 14.0                           # the frame's, not the rules'
