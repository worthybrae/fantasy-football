"""Every fact on a manager's profile, on a league small enough to check by
hand: four teams, two seasons, known answers."""
import json

import duckdb
import pandas as pd
import pytest

from pipeline.db import write_table
from scoring import manager_profile as mp

A, B, C, D = ("{A}", "{B}", "{C}", "{D}")


def _members(season, counters=None):
    rows = []
    for i, swid in enumerate((A, B, C, D), start=1):
        row = {"season": season, "member_id": swid, "display_name": swid.strip("{}").lower(),
               "first_name": None, "last_name": None, "team_id": i,
               "draft_day_rank": i, "acquisitions": 10 * i, "drops": 10 * i,
               "trades": 0, "lineup_moves": 20, "ir_moves": 0}
        row.update((counters or {}).get(swid, {}))
        rows.append(row)
    return rows


def _standings(season, finals, points):
    """finals: team_id -> final_rank; points: team_id -> points_for."""
    rows = []
    for tid in (1, 2, 3, 4):
        rows.append({"season": season, "team_id": tid, "manager": f"m{tid}",
                     "team_name": f"Team {tid}", "wins": 0, "losses": 0, "ties": 0,
                     "points_for": points[tid], "points_against": 0.0,
                     "playoff_seed": finals[tid] if finals[tid] <= 2 else None,
                     "final_rank": finals[tid]})
    return rows


def _matchup(season, period, home, away, hp, ap, tier="NONE"):
    winner = "HOME" if hp > ap else "AWAY" if ap > hp else "TIE"
    return {"season": season, "matchup_period": period, "tier": tier,
            "home_team_id": home, "away_team_id": away, "home_points": hp,
            "away_points": ap, "winner": winner,
            "home_points_by_week_json": json.dumps({str(period): hp}),
            "away_points_by_week_json": json.dumps({str(period): ap})}


@pytest.fixture
def league(tmp_path):
    """2024: A wins the title, D last. 2025: B wins, A last.

    Regular season is three weeks of round-robin; week 4 is the final
    between the top two. A's 2024 regular season: beat B, beat C, beat D
    (3-0) while scoring the most every week -- all-play 3-0-... so luck 0.
    D's 2024: lost every game but out-scored one team each week (all-play
    1 of 3 each week => expected 1 win over 3 games) -- luck -1.0."""
    conn = duckdb.connect(str(tmp_path / "league.duckdb"))
    write_table(conn, "league_members", pd.DataFrame(_members(2024) + _members(2025)))
    write_table(conn, "league_standings", pd.DataFrame(
        _standings(2024, {1: 1, 2: 2, 3: 3, 4: 4}, {1: 400.0, 2: 350.0, 3: 300.0, 4: 250.0})
        + _standings(2025, {1: 4, 2: 1, 3: 2, 4: 3}, {1: 250.0, 2: 400.0, 3: 350.0, 4: 300.0})))
    m = [
        # 2024 regular season. Scores: A 130, B 120, C 110, D 100 each week,
        # except week 3 where D (105) beats C (95).
        _matchup(2024, 1, 1, 2, 130, 120), _matchup(2024, 1, 3, 4, 110, 100),
        _matchup(2024, 2, 1, 3, 130, 110), _matchup(2024, 2, 2, 4, 120, 100),
        _matchup(2024, 3, 1, 4, 130, 100), _matchup(2024, 3, 2, 3, 120, 95),
        _matchup(2024, 4, 1, 2, 140, 120, tier="WINNERS_BRACKET"),
        _matchup(2024, 4, 3, 4, 100, 90, tier="LOSERS_CONSOLATION_LADDER"),
        # 2025: B dominates, A collapses.
        _matchup(2025, 1, 2, 1, 130, 100), _matchup(2025, 1, 3, 4, 120, 110),
        _matchup(2025, 2, 2, 3, 130, 120), _matchup(2025, 2, 1, 4, 100, 110),
        _matchup(2025, 3, 2, 4, 130, 110), _matchup(2025, 3, 1, 3, 100, 120),
        _matchup(2025, 4, 2, 3, 140, 120, tier="WINNERS_BRACKET"),
    ]
    write_table(conn, "league_matchups", pd.DataFrame(m))
    return conn


def test_members_resolves_every_member_once_with_seasons(league):
    df = mp.members(league)
    assert sorted(df.member_id) == sorted([A, B, C, D])
    assert df[df.member_id == A].iloc[0].seasons == [2024, 2025]


def test_finishes_read_the_standings_through_the_members_table(league):
    f = mp.finishes(league, A)
    assert f["n"] == 2
    assert [s["season"] for s in f["seasons"]] == [2024, 2025]
    assert f["seasons"][0]["final_rank"] == 1 and f["seasons"][1]["final_rank"] == 4
    # draft_day_rank 1 both years: projected 1st, finished 1st then 4th.
    assert f["seasons"][0]["outperformance"] == 0 and f["seasons"][1]["outperformance"] == -3
    assert f["titles"] == [2024] and f["avg_finish"] == 2.5
    assert f["points_per_season"] == 325.0


def test_playoffs_come_from_the_bracket_not_the_seed(league):
    p = mp.playoffs(league, A)
    assert p["appearances"] == [2024] and p["bracket_wins"] == 1 and p["bracket_losses"] == 0
    assert p["titles"] == [2024]
    d = mp.playoffs(league, D)
    assert d["appearances"] == [] and d["consolation"] == {"wins": 0, "losses": 1}
    assert d["last_place"] == [2024]
    assert mp.playoffs(league, B)["bracket_losses"] == 1


def test_head_to_head_counts_every_meeting_both_ways(league):
    h2h = mp.head_to_head(league)
    ab = h2h[(A, B)]
    # 2024 wk1 A beat B, wk4 final A beat B; 2025 wk1 B beat A.
    assert (ab["wins"], ab["losses"], ab["games"]) == (2, 1, 3)
    assert ab["playoff_games"] == 1
    assert ab["margin"] == pytest.approx((130 - 120) + (140 - 120) + (100 - 130))
    ba = h2h[(B, A)]
    assert (ba["wins"], ba["losses"]) == (1, 2) and ba["margin"] == -ab["margin"]


def test_luck_is_actual_wins_minus_all_play_expectation(league):
    a = mp.luck(league, A)
    s24 = next(s for s in a["seasons"] if s["season"] == 2024)
    assert s24["actual_wins"] == 3 and s24["expected_wins"] == pytest.approx(3.0)
    assert s24["luck"] == pytest.approx(0.0)
    d = mp.luck(league, D)
    s24 = next(s for s in d["seasons"] if s["season"] == 2024)
    # D lost all three but out-scored C in week 3: all-play 1 of 9.
    assert s24["actual_wins"] == 0 and s24["expected_wins"] == pytest.approx(1 / 3)
    assert s24["luck"] == pytest.approx(-1 / 3)
    assert d["n"] == 6      # regular-season games over two seasons


def test_seasons_strip_names_the_champion_and_the_last_place(league):
    strip = mp.seasons_strip(league)
    assert [s["season"] for s in strip] == [2024, 2025]
    assert strip[0]["champion"]["member_id"] == A and strip[0]["last"]["member_id"] == D
    assert strip[0]["top_scorer"]["member_id"] == A
    assert strip[1]["champion"]["member_id"] == B and strip[1]["last"]["member_id"] == A
