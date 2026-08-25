import json

import pandas as pd
import pytest

from pipeline.db import get_conn, write_table
from scoring import league as league_mod

TEAMS = 8


def _settings(season, teams=TEAMS):
    return league_mod.LeagueSettings(
        season=season, teams=teams,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
        flex_slots=0, bench=1, scoring={}, draft_type="SNAKE",
        pick_order=tuple(range(1, teams + 1)))


def seed_league(path, seasons=(2024, 2025), teams=TEAMS, rounds=None):
    """An 8-team league: team i drafts picks with a fixed ADP gap.

    Team 1 always gets steals (value +10), team 8 always reaches (-10), the
    rest are on the market. Standings: team i finished i-th in 2024 with
    (10 - i) wins; 2025 is in progress (null finish).
    """
    conn = get_conn(path)
    rows, adp, standings, leagues = [], [], [], []
    for season in seasons:
        s = _settings(season, teams)
        rounds_n = rounds or s.rounds
        from scoring.draft_sim import snake_slots
        slots = snake_slots(teams, rounds_n)
        leagues.append({"season": season, "settings_json": league_mod.to_json(s),
                        "name": "Test League"})
        for i, slot in enumerate(slots):
            overall = i + 1
            name = f"P{season}_{overall}"
            gap = {1: 10, teams: -10}.get(slot, 0)
            rank = max(1, overall - gap)
            rows.append({"season": season, "overall_pick": overall,
                         "round": i // teams + 1, "round_pick": i % teams + 1,
                         "team_id": slot, "espn_player_id": overall,
                         "player_name": name,
                         "position": "RB" if i % 2 else "WR", "nfl_team": "DET",
                         "keeper": False})
            adp.append({"season": season, "adp_name": name,
                        "position": "RB" if i % 2 else "WR", "team": "DET",
                        "adp_rank": rank})
        for t in range(1, teams + 1):
            done = season == 2024
            standings.append({
                "season": season, "team_id": t, "manager": f"m{t}",
                "team_name": f"Team {t}", "wins": 10 - t if done else 0,
                "losses": t + 4 if done else 0, "ties": 0,
                "points_for": 2000.0 - 50 * t if done else 0.0,
                "points_against": 1800.0, "playoff_seed": t if done and t <= 4 else None,
                "final_rank": t if done else None})
    write_table(conn, "draft_picks", pd.DataFrame(rows))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": season, "team_id": t, "manager": f"m{t}", "slot": t}
        for season in seasons for t in range(1, teams + 1)]))
    write_table(conn, "historic_adp", pd.DataFrame(adp))
    write_table(conn, "league_standings", pd.DataFrame(standings))
    write_table(conn, "league", pd.DataFrame(leagues))
    return conn


def test_verdict_bands_match_the_ticker():
    from scoring.league_report import verdict
    assert verdict(None, 8) is None
    assert verdict(2, 8) == "market" and verdict(-2, 8) == "market"
    assert verdict(3, 8) == "value" and verdict(7, 8) == "value"
    assert verdict(8, 8) == "steal"
    assert verdict(-3, 8) == "early" and verdict(-7, 8) == "early"
    assert verdict(-8, 8) == "reach"
    # `round` floors at 2, so a 1-team league still has bands.
    assert verdict(2.4, 1) == "market" and verdict(3, 1) == "steal"
    # JS `Math.round` rounds half AWAY FROM ZERO on the negative side but
    # toward +infinity overall (-2.5 -> -2, 2.5 -> 3), not Python's banker's
    # rounding (round(2.5) == 2, round(-2.5) == -2 too, but round(7.5) == 8
    # while round(-7.5) == -8) -- these four pin the exact half-integer
    # inputs where the two disagree.
    assert verdict(2.5, 8) == "value"
    assert verdict(-2.5, 8) == "market"
    assert verdict(7.5, 8) == "steal"
    assert verdict(-7.5, 8) == "early"


def test_pick_value_is_null_past_the_last_pick():
    from scoring.league_report import pick_value
    assert pick_value(10, 4.0, 120) == 6.0
    assert pick_value(10, None, 120) is None
    assert pick_value(98, 212.0, 120) is None


def test_letter_cut_points_scale_with_team_count():
    from scoring.league_report import letter
    assert [letter(r, 8) for r in range(8)] == ["A", "B", "B", "C", "C", "D", "D", "F"]
    assert [letter(r, 10) for r in range(10)] == ["A", "B", "B", "C", "C", "C", "D", "D", "F", "F"]


def test_names_for_prefers_the_newest_season(tmp_path):
    from scoring.league_report import names_for
    conn = get_conn(str(tmp_path / "t.duckdb"))
    # Team 1 is only in draft_teams (no league_standings row ever covers it):
    # the fallback loop must still pick the newest season, not the oldest.
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2023, "team_id": 1, "manager": "old", "slot": 1},
        {"season": 2025, "team_id": 1, "manager": "new", "slot": 1},
    ]))
    # Team 2 is only in league_standings, covering the other path in the
    # same call.
    write_table(conn, "league_standings", pd.DataFrame([
        {"season": 2024, "team_id": 2, "manager": "s-old", "team_name": "Old Team 2",
         "wins": 0, "losses": 0, "ties": 0, "points_for": 0.0, "points_against": 0.0,
         "playoff_seed": None, "final_rank": None},
        {"season": 2025, "team_id": 2, "manager": "s-new", "team_name": "New Team 2",
         "wins": 0, "losses": 0, "ties": 0, "points_for": 0.0, "points_against": 0.0,
         "playoff_seed": None, "final_rank": None},
    ]))
    names = names_for(conn)
    assert names[1] == ("new", None)
    assert names[2] == ("s-new", "New Team 2")


def test_historical_picks_join_adp_and_names(tmp_path):
    from scoring.league_report import historical_picks, PICK_COLUMNS
    conn = seed_league(str(tmp_path / "t.duckdb"))
    picks = historical_picks(conn, 2024)
    assert list(picks.columns) == PICK_COLUMNS
    assert len(picks) == TEAMS * _settings(2024).rounds
    first = picks.sort_values("overall_pick").iloc[0]
    assert first["manager"] == "m1" and first["team_name"] == "Team 1"
    assert first["value"] == pytest.approx(0.0)      # rank floors at 1 for pick 1
    t1 = picks[picks["team_id"] == 1]
    assert (t1["value"].iloc[1:] == 10).all()
    # Team 8's final pick (overall 56) has seeded ADP rank 66 -- past this
    # draft's own last pick (56) -- so it exercises the past-the-last-pick
    # null rule end to end: null value, null verdict, not a giant reach.
    t8_last = picks[(picks["team_id"] == 8) & (picks["overall_pick"] == 56)].iloc[0]
    assert pd.isna(t8_last["value"]) and pd.isna(t8_last["verdict"])
    assert set(picks[picks["team_id"] == 8]["verdict"].dropna()) == {"reach"}


def test_historical_picks_is_empty_without_adp_for_that_season(tmp_path):
    from scoring.league_report import historical_picks
    conn = seed_league(str(tmp_path / "t.duckdb"), seasons=(2024,))
    assert historical_picks(conn, 2023).empty


def test_live_picks_reads_the_room(tmp_path):
    from scoring.league_report import live_picks, PICK_COLUMNS
    settings = _settings(2026, teams=2)
    board_by_id = {
        "a": {"player_id": "a", "name": "A Star", "position": "WR", "market_rank": 4.0},
        "b": {"player_id": "b", "name": "B Steady", "position": "RB", "market_rank": 1.0},
        # Market rank past the last pick: value must be null.
        "c": {"player_id": "c", "name": "C Kicker", "position": "K", "market_rank": 250.0},
    }
    names = {1: ("m1", "Team 1"), 2: ("m2", "Team 2")}
    drafted = [("a", 1), ("b", 2), ("c", 3), ("zzz", 4)]
    picks = live_picks(drafted, board_by_id, settings, names, {2: "ESPN Two"}, 2026)
    assert list(picks.columns) == PICK_COLUMNS
    assert len(picks) == 4
    a = picks[picks["overall_pick"] == 1].iloc[0]
    # teams=2 makes round = max(2, teams) = 2, same as ON_MARKET_SLOTS -- the
    # "value"/"early" band (`ON_MARKET_SLOTS < size < round`) is an empty
    # interval for any round <= 3, so a 3-slot move can only ever be "market"
    # or "reach" here, never "early". Ported PickTicker.tsx's own `verdict`,
    # called with these exact numbers, agrees: reach.
    assert a["manager"] == "m1" and a["value"] == pytest.approx(-3.0) and a["verdict"] == "reach"
    b = picks[picks["overall_pick"] == 2].iloc[0]
    assert b["manager"] == "m2" and b["team_name"] == "Team 2"
    assert pd.isna(picks[picks["overall_pick"] == 3].iloc[0]["value"])
    unknown = picks[picks["overall_pick"] == 4].iloc[0]
    assert unknown["player_name"] == "zzz" and pd.isna(unknown["market_rank"])


def test_live_picks_falls_back_to_espn_team_name(tmp_path):
    from scoring.league_report import live_picks
    settings = _settings(2026, teams=2)
    picks = live_picks([("a", 1), ("b", 2)], {}, settings, {}, {1: "Piss Floor"}, 2026)
    assert picks.iloc[0]["manager"] == "Piss Floor" and picks.iloc[0]["team_name"] == "Piss Floor"
    assert picks.iloc[1]["manager"] == "Team 2"


def test_draft_grades_rank_teams(tmp_path):
    from scoring.league_report import historical_picks, draft_grades
    conn = seed_league(str(tmp_path / "t.duckdb"))
    grades = draft_grades(historical_picks(conn, 2024), TEAMS)
    assert [g["manager"] for g in grades][0] == "m1"
    assert [g["manager"] for g in grades][-1] == "m8"
    assert [g["grade"] for g in grades] == ["A", "B", "B", "C", "C", "D", "D", "F"]
    top = grades[0]
    assert top["steals"] == _settings(2024).rounds - 1 and top["reaches"] == 0
    assert top["best_pick"]["player_name"].startswith("P2024_")
    assert top["best_pick"]["verdict"] == "steal"
    assert top["worst_pick"]["value"] == pytest.approx(0.0)
    assert top["shape"]["positions"] == {"WR": 4, "RB": 3}
    assert top["shape"]["first_round"]["QB"] is None
    # Team 8's final pick is past the draft's own last pick (see
    # test_historical_picks_join_adp_and_names) and so is ungraded, not a
    # reach: one fewer graded pick and one fewer reach than a full sweep.
    assert grades[-1]["reaches"] == _settings(2024).rounds - 1
    assert grades[-1]["graded_picks"] == _settings(2024).rounds - 1


def test_draft_grades_with_no_gradeable_picks():
    from scoring.league_report import draft_grades, PICK_COLUMNS
    assert draft_grades(pd.DataFrame(columns=PICK_COLUMNS), 8) == []


def test_team_profiles_summarise_history(tmp_path):
    from scoring.league_report import team_profiles
    conn = seed_league(str(tmp_path / "t.duckdb"))
    profiles = {p["manager"]: p for p in team_profiles(conn)}
    m1 = profiles["m1"]
    assert m1["team_name"] == "Team 1"
    assert [s["season"] for s in m1["seasons"]] == [2024, 2025]
    assert m1["titles"] == [2024] and m1["playoffs"] == [2024]
    assert m1["completed"] == 1
    assert m1["win_pct"] == pytest.approx(9 / 14)
    assert m1["avg_finish"] == 1.0
    assert m1["drafts"] == 2
    # Both seasons repeat the same snake draft, so team 1's own first pick of
    # each season is the one exception (value 0, "market") among its 14
    # career picks -- 12 of 14 are the seed's +10 steal, giving a mean of
    # 60/7 (~8.57) and a steal rate of 6/7 (~0.857), not the "> 9"/"> 0.9"
    # the brief's comment implied by (wrongly) treating "every pick but the
    # first" as true career-wide rather than per-season.
    assert m1["mean_value"] > 8        # every pick but the first is +10
    assert m1["steal_rate"] > 0.8 and m1["reach_rate"] == 0.0
    assert m1["first_pick_positions"] == {"WR": 2}
    assert m1["career_best"]["value"] == 10.0
    m5 = profiles["m5"]
    assert m5["playoffs"] == [] and m5["titles"] == []
    assert m5["mean_value"] == pytest.approx(0.0)


def test_team_profiles_without_standings_still_have_draft_habits(tmp_path):
    from scoring.league_report import team_profiles
    conn = seed_league(str(tmp_path / "t.duckdb"))
    conn.execute("DROP TABLE league_standings")
    profiles = {p["manager"]: p for p in team_profiles(conn)}
    assert profiles["m8"]["seasons"] == [] and profiles["m8"]["win_pct"] is None
    assert profiles["m8"]["reach_rate"] == 1.0


def test_power_rankings_blend_draft_and_history():
    from scoring.league_report import power_rankings
    grades = [{"manager": "a", "team_name": "A", "value_per_pick": 5.0, "value_total": 50},
              {"manager": "b", "team_name": "B", "value_per_pick": 0.0, "value_total": 0},
              {"manager": "c", "team_name": "C", "value_per_pick": -5.0, "value_total": -50}]
    profiles = [{"manager": "a", "win_pct": 0.2, "completed": 3},
                {"manager": "b", "win_pct": 0.9, "completed": 3},
                {"manager": "c", "win_pct": None, "completed": 0}]
    ranks = power_rankings(grades, profiles)
    assert [r["manager"] for r in ranks] == ["b", "a", "c"]
    assert ranks[0]["rank"] == 1
    c = next(r for r in ranks if r["manager"] == "c")
    assert c["first_year"] is True
    # With equal drafts, history decides.
    tie = [{"manager": m, "team_name": m, "value_per_pick": 0.0, "value_total": 0} for m in "ab"]
    assert power_rankings(tie, profiles)[0]["manager"] == "b"


def test_build_facts_assembles_the_document(tmp_path):
    from scoring.league_report import build_facts
    conn = seed_league(str(tmp_path / "t.duckdb"))
    facts = build_facts(conn, "424242", 2024)
    assert facts["status"] == "ready"
    assert facts["league_id"] == "424242" and facts["season"] == 2024
    assert facts["league_name"] == "Test League" and facts["teams"] == TEAMS
    assert len(facts["report_cards"]) == TEAMS and len(facts["power_rankings"]) == TEAMS
    assert len(facts["profiles"]) == TEAMS
    assert facts["report_cards"][0]["grade"] == "A"
    assert facts["intro"] is None and facts["report_cards"][0]["blurb"] is None
    json.dumps(facts)      # must be plain JSON


def test_build_facts_fails_with_a_reason_without_picks(tmp_path):
    from scoring.league_report import build_facts
    conn = seed_league(str(tmp_path / "t.duckdb"), seasons=(2024,))
    facts = build_facts(conn, "1", 2023)
    assert facts["status"] == "failed" and "no graded picks" in facts["reason"]


def test_build_facts_takes_live_picks(tmp_path):
    from scoring.league_report import build_facts, live_picks
    conn = seed_league(str(tmp_path / "t.duckdb"))
    settings = _settings(2026)
    board = {"x": {"player_id": "x", "name": "X", "position": "RB", "market_rank": 20.0}}
    # Overall pick 32 is team 1's (m1's) own slot in this 8-team/7-round snake
    # draft (see test_historical_picks_join_adp_and_names / snake_slots) --
    # pick 1 can never grade as a steal (there is no rank below 1 for the
    # market to have missed), so this is the smallest change that actually
    # exercises a steal end to end: 32 - 20 = 12, past the 8-slot threshold.
    picks = live_picks([("x", 32)], board, settings, {1: ("m1", "Team 1")}, {}, 2026)
    facts = build_facts(conn, "424242", 2026, picks=picks)
    assert facts["status"] == "ready"
    assert facts["report_cards"][0]["manager"] == "m1"
    assert facts["report_cards"][0]["best_pick"]["verdict"] == "steal"


def test_merge_prose_fills_or_marks_numbers_only():
    from scoring.league_report import merge_prose
    facts = {"status": "ready", "intro": None,
             "report_cards": [{"manager": "m1", "nickname": None, "blurb": None}],
             "power_rankings": [{"manager": "m1", "line": None}]}
    done = merge_prose(dict(facts), {"intro": "hi", "cards": [
        {"manager": "m1", "nickname": "The Thief", "blurb": "Stole."}],
        "rankings": [{"manager": "m1", "line": "Top."}]}, "claude-haiku-4-5")
    assert done["status"] == "ready" and done["model"] == "claude-haiku-4-5"
    assert done["intro"] == "hi"
    assert done["report_cards"][0]["nickname"] == "The Thief"
    assert done["power_rankings"][0]["line"] == "Top."
    quiet = merge_prose(dict(facts), None, None)
    assert quiet["status"] == "numbers_only" and quiet["model"] is None
    failed = merge_prose({"status": "failed", "reason": "x"}, None, None)
    assert failed["status"] == "failed"
