import pandas as pd
import pytest
from pipeline.db import get_conn, write_table
from scoring.player_history import attributes_as_of


def _seed(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    rows = []
    # Steady: 10 ppg in 2021, 10 in 2022, 10 in 2023.
    for season, ppg in ((2021, 10), (2022, 10), (2023, 10)):
        rows += [{"player_id": "p_steady", "player_display_name": "Steady Sam",
                  "position": "RB", "recent_team": "DET", "opponent_team": "GB",
                  "season": season, "week": w, "receptions": ppg, "receiving_yards": 0,
                  "targets": ppg, "carries": 0} for w in range(1, 18)]
    # Rising: 5, 10, 20.
    for season, ppg in ((2021, 5), (2022, 10), (2023, 20)):
        rows += [{"player_id": "p_rising", "player_display_name": "Rising Rick",
                  "position": "WR", "recent_team": "GB", "opponent_team": "DET",
                  "season": season, "week": w, "receptions": ppg, "receiving_yards": 0,
                  "targets": ppg, "carries": 0} for w in range(1, 18)]
    # Fragile: one 8-game season.
    rows += [{"player_id": "p_hurt", "player_display_name": "Hurt Harry",
              "position": "RB", "recent_team": "CHI", "opponent_team": "GB",
              "season": 2023, "week": w, "receptions": 12, "receiving_yards": 0,
              "targets": 12, "carries": 0} for w in range(1, 9)]
    write_table(conn, "weekly", pd.DataFrame(rows))
    write_table(conn, "players", pd.DataFrame([
        {"gsis_id": "p_steady", "display_name": "Steady Sam",
         "birth_date": "1996-09-01", "rookie_season": 2019},
        {"gsis_id": "p_rising", "display_name": "Rising Rick",
         "birth_date": "2002-09-01", "rookie_season": 2021},
    ]))
    return conn


def test_uses_only_seasons_before_the_draft(tmp_path):
    """The no-lookahead rule: attributes for the 2023 draft may not see 2023."""
    a = attributes_as_of(_seed(tmp_path), 2023).set_index("key")
    steady = a.loc["RB|steady sam"]
    # 2021 + 2022 only, both 10 ppg -> flat.
    assert steady["trend"] == pytest.approx(0.0)
    # Hurt Harry only played in 2023, so as of the 2023 draft he is unknown.
    assert "RB|hurt harry" not in a.index


def test_trend_is_positive_for_a_rising_player(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    assert a.loc["WR|rising rick"]["trend"] > 0
    assert a.loc["RB|steady sam"]["trend"] == pytest.approx(0.0)


# `ppg_std` and `missed_rate` had a test each here. They fed the proposed
# `volatility` feature, which `ablation()` measured at -0.0029 and cut; the
# columns outlived it by a branch, filled by both pools and read by nothing.
# Removed with them rather than left as coverage of a dead column.


def test_age_is_computed_at_the_drafts_september(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    assert a.loc["RB|steady sam"]["age"] == pytest.approx(28.0, abs=0.05)
    assert a.loc["WR|rising rick"]["age"] == pytest.approx(22.0, abs=0.05)


def test_age_is_nan_when_birth_date_is_unknown(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    assert pd.isna(a.loc["RB|hurt harry"]["age"])


def test_prod_rank_orders_by_most_recent_prior_ppg(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    # 2023 ppg: rising 20 > hurt 12 > steady 10.
    assert a.loc["WR|rising rick"]["prod_rank"] < a.loc["RB|hurt harry"]["prod_rank"]
    assert a.loc["RB|hurt harry"]["prod_rank"] < a.loc["RB|steady sam"]["prod_rank"]


def test_no_track_record_is_false_for_everyone_with_history(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024)
    assert not a["no_track_record"].any()


def test_empty_before_the_first_season(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2020)
    assert a.empty
    assert list(a.columns) == ["key", "age", "no_track_record",
                               "prod_rank", "trend"]


def _seed_tie(tmp_path):
    conn = get_conn(str(tmp_path / "tie.duckdb"))
    rows = []
    # Two RBs post the exact same 2023 ppg (15) -- a tie in most-recent
    # production. A third player sits clearly behind them at 10 ppg.
    for pid, name, team in (("p_tie_a", "Tie A", "DET"), ("p_tie_b", "Tie B", "GB")):
        rows += [{"player_id": pid, "player_display_name": name,
                  "position": "RB", "recent_team": team, "opponent_team": "CHI",
                  "season": 2023, "week": w, "receptions": 15, "receiving_yards": 0,
                  "targets": 15, "carries": 0} for w in range(1, 18)]
    rows += [{"player_id": "p_third", "player_display_name": "Third Guy",
              "position": "RB", "recent_team": "CHI", "opponent_team": "GB",
              "season": 2023, "week": w, "receptions": 10, "receiving_yards": 0,
              "targets": 10, "carries": 0} for w in range(1, 18)]
    write_table(conn, "weekly", pd.DataFrame(rows))
    write_table(conn, "players", pd.DataFrame(columns=[
        "gsis_id", "display_name", "birth_date", "rookie_season"]))
    return conn


def test_prod_rank_is_dense_for_tied_players(tmp_path):
    """Two players with identical most-recent-season ppg must share a rank,
    and the next distinct player down must be exactly one rank behind --
    not two, which would invent a distinction the data doesn't support."""
    a = attributes_as_of(_seed_tie(tmp_path), 2024).set_index("key")
    tie_a = a.loc["RB|tie a"]["prod_rank"]
    tie_b = a.loc["RB|tie b"]["prod_rank"]
    third = a.loc["RB|third guy"]["prod_rank"]
    assert tie_a == tie_b
    assert third == tie_a + 1
