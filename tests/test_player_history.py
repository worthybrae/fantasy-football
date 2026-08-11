import pandas as pd
import pytest
from pipeline.db import get_conn, write_table
from scoring.player_history import COLUMNS, attributes_as_of


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
    # Fragile: eight games of a season his team played all seventeen of.
    rows += [{"player_id": "p_hurt", "player_display_name": "Hurt Harry",
              "position": "RB", "recent_team": "CHI", "opponent_team": "GB",
              "season": 2023, "week": w, "receptions": 12, "receiving_yards": 0,
              "targets": 12, "carries": 0} for w in range(1, 9)]
    # Harry's teammate, present for all seventeen. `played_share` counts team
    # games from the weekly rows, so without someone who played the full
    # season CHI would look like an 8-game team and Harry like an ever-present.
    rows += [{"player_id": "p_teammate", "player_display_name": "Every Week Eddie",
              "position": "WR", "recent_team": "CHI", "opponent_team": "GB",
              "season": 2023, "week": w, "receptions": 4, "receiving_yards": 0,
              "targets": 4, "carries": 0} for w in range(1, 18)]
    # Faded: a career year in 2022, then half of it in 2023. Trend and
    # peak_gap disagree about him on purpose.
    for season, ppg in ((2022, 20), (2023, 8)):
        rows += [{"player_id": "p_faded", "player_display_name": "Faded Fred",
                  "position": "WR", "recent_team": "MIN", "opponent_team": "GB",
                  "season": season, "week": w, "receptions": ppg,
                  "receiving_yards": 0, "targets": ppg, "carries": 0}
                 for w in range(1, 18)]
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


def test_usage_is_opportunities_per_game_not_per_season(tmp_path):
    """Volume has to be a rate, or every 17-game player outranks every
    injured one on `usage` for a reason `played_share` already carries."""
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    # Sam: 10 targets a week, no carries, 17 games.
    assert a.loc["RB|steady sam"]["usage"] == pytest.approx(10.0)
    # Harry: 12 a week over 8 games -- higher per game despite a third of
    # Sam's season total.
    assert a.loc["RB|hurt harry"]["usage"] == pytest.approx(12.0)


def test_efficiency_is_points_per_opportunity(tmp_path):
    """The column that separates a grinder from a big-play threat. Sam
    catches ten balls for nothing (1.0 PPR point each, per opportunity);
    Eddie catches four. Same rate, wildly different volume -- which is the
    distinction `usage` and `efficiency` exist to keep apart."""
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    assert a.loc["RB|steady sam"]["efficiency"] == pytest.approx(1.0)
    assert a.loc["WR|every week eddie"]["efficiency"] == pytest.approx(1.0)
    assert a.loc["WR|every week eddie"]["usage"] == pytest.approx(4.0)


def test_played_share_counts_games_his_team_actually_played(tmp_path):
    """8 of 17, not 8 of 8. The denominator comes from the team's weekly
    rows, so a player who missed half a season reads as having missed it."""
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    assert a.loc["RB|hurt harry"]["played_share"] == pytest.approx(8 / 17)
    assert a.loc["WR|every week eddie"]["played_share"] == pytest.approx(1.0)


def test_peak_gap_and_trend_disagree_about_a_faded_player(tmp_path):
    """Both read Fred as declining, but they are not the same column: the
    gap says how far below his best he finished, in points per game, while
    the trend is a slope. A player flat for three years well under a career
    year has a trend near zero and a large gap."""
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    fred = a.loc["WR|faded fred"]
    assert fred["peak_gap"] == pytest.approx(12.0)   # 20 ppg peak, 8 last
    assert fred["trend"] < 0
    # A player at his own best right now has no gap, whatever his trend.
    assert a.loc["WR|rising rick"]["peak_gap"] == pytest.approx(0.0)
    assert a.loc["WR|rising rick"]["trend"] > 0


def test_profile_columns_read_no_season_at_or_after_the_draft(tmp_path):
    """The no-lookahead rule applies to the new columns too, not just trend.
    Fred's 2023 collapse must be invisible to the 2023 draft."""
    a = attributes_as_of(_seed(tmp_path), 2023).set_index("key")
    fred = a.loc["WR|faded fred"]
    assert fred["usage"] == pytest.approx(20.0)      # 2022 only
    assert fred["peak_gap"] == pytest.approx(0.0)    # 2022 was his best so far
    # Harry played only in 2023, so the 2023 draft has no profile for him.
    assert "RB|hurt harry" not in a.index


def test_empty_before_the_first_season(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2020)
    assert a.empty
    # Against COLUMNS rather than a copy of it: the point of this assertion
    # is that the empty frame carries the same columns as a populated one,
    # so callers can index it without a special case -- not that the list has
    # any particular length.
    assert list(a.columns) == COLUMNS


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
