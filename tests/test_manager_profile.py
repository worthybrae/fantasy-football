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


def _txn(season, week, tid, team, member, type_, status, execution, items, bid=0.0,
         related=None, when="2025-10-01T12:00:00Z"):
    return {"season": season, "week": week, "txn_id": tid, "related_txn_id": related,
            "team_id": team, "member_id": member, "type": type_, "status": status,
            "execution_type": execution, "bid_amount": bid,
            "proposed_at": pd.Timestamp(when), "items_json": json.dumps(items)}


def _add(pid, to_team):
    return {"type": "ADD", "playerId": pid, "fromTeamId": 0, "toTeamId": to_team,
            "fromLineupSlotId": -1, "toLineupSlotId": 20}


def _drop(pid, from_team):
    return {"type": "DROP", "playerId": pid, "fromTeamId": from_team, "toTeamId": 0,
            "fromLineupSlotId": 20, "toLineupSlotId": -1}


def _trade_item(pid, from_team, to_team):
    return {"type": "TRADE", "playerId": pid, "fromTeamId": from_team, "toTeamId": to_team,
            "fromLineupSlotId": 20, "toLineupSlotId": 20}


def _lineup(season, week, team, pid, name, pos, slot, actual, projected=None):
    return {"season": season, "week": week, "team_id": team, "player_id": pid,
            "player_name": name, "position": pos, "lineup_slot": slot,
            "actual_points": actual, "projected_points": projected,
            "acquisition_type": "DRAFT", "injury_status": "ACTIVE"}


@pytest.fixture
def activity(league):
    """On top of `league`: 2025 transactions and lineups.

    A (team 1) claims player 100 in week 2 and wins it; claims 101 in week
    3 and loses it to B (same process run, B's executed, A's failed); adds
    102 as a free agent on a Tuesday; cancels a claim on 103.
    A proposes a trade to B in week 3 (A sends 200, gets 300); B accepts.
    After the trade, 300 scores 20 and 25 for A in weeks 4-5; 200 scores
    10 and 10 for B. Balance for A: (20+25) - (10+10) = +25.
    B proposes a trade to A in week 4; A declines."""
    conn = league
    t = [
        _txn(2025, 2, "w1", 1, A, "WAIVER", "PENDING", "EXECUTE", [_add(100, 1)],
             when="2025-09-15T18:00:00Z"),
        _txn(2025, 2, "w1p", 1, A, "WAIVER", "EXECUTED", "PROCESS", [_add(100, 1)],
             related="w1", when="2025-09-17T07:00:00Z"),
        _txn(2025, 3, "w2", 1, A, "WAIVER", "PENDING", "EXECUTE", [_add(101, 1)]),
        _txn(2025, 3, "w2p", 1, A, "WAIVER", "FAILED_INVALIDPLAYERSOURCE", "PROCESS",
             [_add(101, 1)], related="w2", when="2025-09-24T07:00:00Z"),
        _txn(2025, 3, "w2b", 2, B, "WAIVER", "PENDING", "EXECUTE", [_add(101, 2)]),
        _txn(2025, 3, "w2bp", 2, B, "WAIVER", "EXECUTED", "PROCESS", [_add(101, 2)],
             related="w2b", when="2025-09-24T07:00:00Z"),
        _txn(2025, 3, "fa1", 1, A, "FREEAGENT", "EXECUTED", "EXECUTE",
             [_add(102, 1), _drop(50, 1)], when="2025-09-23T15:00:00Z"),   # a Tuesday
        _txn(2025, 3, "wc", 1, A, "WAIVER", "PENDING", "EXECUTE", [_add(103, 1)]),
        _txn(2025, 3, "wcc", 1, A, "WAIVER", "CANCELED", "CANCEL", [_add(103, 1)], related="wc"),
        _txn(2025, 3, "tp1", 1, A, "TRADE_PROPOSAL", "PENDING", "EXECUTE",
             [_trade_item(200, 1, 2), _trade_item(300, 2, 1)]),
        _txn(2025, 3, "ta1", 2, B, "TRADE_ACCEPT", "EXECUTED", "PROCESS",
             [_trade_item(200, 1, 2), _trade_item(300, 2, 1)], related="tp1",
             when="2025-09-25T10:00:00Z"),
        _txn(2025, 4, "tp2", 2, B, "TRADE_PROPOSAL", "PENDING", "EXECUTE",
             [_trade_item(201, 2, 1), _trade_item(301, 1, 2)]),
        _txn(2025, 4, "td2", 1, A, "TRADE_DECLINE", "EXECUTED", "EXECUTE", [], related="tp2"),
        _txn(2025, 3, "lm", 1, A, "FUTURE_ROSTER", "EXECUTED", "EXECUTE",
             [{"type": "LINEUP", "playerId": 300, "fromLineupSlotId": 20, "toLineupSlotId": 2}]),
    ]
    write_table(conn, "league_transactions", pd.DataFrame(t))
    lu = [
        _lineup(2025, 3, 1, 200, "Sent Back", "RB", 2, 12.0),
        _lineup(2025, 3, 2, 300, "Got Back", "RB", 2, 15.0),
        _lineup(2025, 4, 1, 300, "Got Back", "RB", 2, 20.0),
        _lineup(2025, 5, 1, 300, "Got Back", "RB", 2, 25.0),
        _lineup(2025, 4, 2, 200, "Sent Back", "RB", 2, 10.0),
        _lineup(2025, 5, 2, 200, "Sent Back", "RB", 2, 10.0),
    ]
    write_table(conn, "league_lineups", pd.DataFrame(lu))
    return conn


def test_waivers_count_claims_wins_losses_and_free_agents(activity):
    w = mp.waivers(activity, A)
    assert w["n"] == 3                      # claims made: 100, 101, 103
    assert w["claims"] == 3 and w["won"] == 1 and w["lost"] == 1 and w["canceled"] == 1
    assert w["free_agent_adds"] == 1 and w["drops"] == 1
    assert w["adds_by_weekday"]["Tue"] == 1
    assert w["busiest_week"] == {"season": 2025, "week": 3, "moves": 2}   # won/lost/fa: week 3 has fa + lost
    assert "bids" not in w                  # no FAAB this league
    b = mp.waivers(activity, B)
    assert b["won"] == 1 and b["lost"] == 0


def test_trades_count_proposals_outcomes_partners_and_balance(activity):
    t = mp.trades(activity, A)
    assert t["proposed"] == 1 and t["received"] == 1
    assert t["accepted"] == 1 and t["declined_by_me"] == 1 and t["declined_by_them"] == 0
    assert t["vetoed"] == 0
    assert t["partners"] == [{"member_id": B, "display_name": "b", "trades": 1}]
    assert t["n"] == 1                      # executed trades the balance rests on
    assert t["balance"] == pytest.approx(25.0)
    assert t["ledger"][0]["sent"] == [{"player_id": 200, "name": "Sent Back"}]
    assert t["ledger"][0]["received"] == [{"player_id": 300, "name": "Got Back"}]
    assert t["ledger"][0]["balance"] == pytest.approx(25.0)
    b = mp.trades(activity, B)
    assert b["balance"] == pytest.approx(-25.0) and b["declined_by_them"] == 1


def test_player_names_come_from_any_week_a_player_was_rostered(activity):
    names = mp.player_names(activity)
    assert names[300] == "Got Back" and 999 not in names


def test_trades_balance_and_partners_span_every_team_in_a_multi_team_trade(league):
    """A sends 200 to B and 201 to C, and receives 300 from B. The balance
    must subtract each sent player's rest-of-season points from the team
    that actually received them, not just the first sent item's team, and
    `partners` must list both B and C, not just the first counterpart
    `counterpart()` happens to find."""
    conn = league
    t = [
        _txn(2025, 3, "tp1", 1, A, "TRADE_PROPOSAL", "PENDING", "EXECUTE",
             [_trade_item(200, 1, 2), _trade_item(201, 1, 3), _trade_item(300, 2, 1)]),
        _txn(2025, 3, "ta1", 2, B, "TRADE_ACCEPT", "EXECUTED", "PROCESS",
             [_trade_item(200, 1, 2), _trade_item(201, 1, 3), _trade_item(300, 2, 1)],
             related="tp1", when="2025-09-25T10:00:00Z"),
    ]
    write_table(conn, "league_transactions", pd.DataFrame(t))
    lu = [
        _lineup(2025, 4, 1, 300, "Got Back", "RB", 2, 20.0),
        _lineup(2025, 5, 1, 300, "Got Back", "RB", 2, 25.0),
        _lineup(2025, 4, 2, 200, "Sent To B", "RB", 2, 10.0),
        _lineup(2025, 5, 2, 200, "Sent To B", "RB", 2, 10.0),
        _lineup(2025, 4, 3, 201, "Sent To C", "RB", 2, 5.0),
        _lineup(2025, 5, 3, 201, "Sent To C", "RB", 2, 5.0),
    ]
    write_table(conn, "league_lineups", pd.DataFrame(lu))
    t_a = mp.trades(conn, A)
    assert t_a["n"] == 1
    # +45 (300, rest of season for A) - 20 (200, rest of season for B)
    # - 10 (201, rest of season for C)
    assert t_a["balance"] == pytest.approx(15.0)
    assert {p["member_id"] for p in t_a["partners"]} == {B, C}
    assert sum(p["trades"] for p in t_a["partners"]) == 2


SLOTS = {"0": 1, "2": 1, "4": 1, "23": 1, "20": 3}     # QB, RB, WR, FLEX, 3 bench


def test_optimal_points_fills_flex_with_the_best_leftover():
    entries = [("QB", 20.0), ("RB", 15.0), ("RB", 12.0), ("WR", 10.0), ("WR", 18.0), ("TE", 9.0)]
    # QB 20 + RB 15 + WR 18 + FLEX best of {RB 12, WR 10, TE 9} = 12 -> 65
    assert mp.optimal_points(entries, SLOTS) == pytest.approx(65.0)


@pytest.fixture
def lineups_league(league):
    conn = league
    raw = pd.DataFrame([{"season": 2025, "view": "mSettings", "week": 0,
                         "fetched_at": pd.Timestamp.now(tz="UTC"),
                         "payload_json": json.dumps({"settings": {
                             "rosterSettings": {"lineupSlotCounts": SLOTS},
                             "acquisitionSettings": {"isUsingAcquisitionBudget": False}}})}])
    write_table(conn, "league_raw", raw)
    # A, 2025 week 1: started QB 20, RB 8, WR 18, FLEX RB 5; benched RB 15 (OUT starter RB 8).
    # Optimal: 20 + 15 + 18 + 8 = 61; started 51; left 10.
    rows = [
        _lineup(2025, 1, 1, 1, "Q", "QB", 0, 20.0), _lineup(2025, 1, 1, 2, "R1", "RB", 2, 8.0),
        _lineup(2025, 1, 1, 3, "W", "WR", 4, 18.0), _lineup(2025, 1, 1, 4, "R2", "RB", 23, 5.0),
        _lineup(2025, 1, 1, 5, "R3", "RB", 20, 15.0),
        # week 2: optimal lineup started (hit). QB 10, RB 10, WR 10, FLEX 10, bench 1.
        _lineup(2025, 2, 1, 1, "Q", "QB", 0, 10.0), _lineup(2025, 2, 1, 2, "R1", "RB", 2, 10.0),
        _lineup(2025, 2, 1, 3, "W", "WR", 4, 10.0), _lineup(2025, 2, 1, 4, "R2", "RB", 23, 10.0),
        _lineup(2025, 2, 1, 5, "R3", "RB", 20, 1.0),
    ]
    df = pd.DataFrame(rows)
    df.loc[(df.week == 1) & (df.player_id == 2), "injury_status"] = "OUT"
    write_table(conn, "league_lineups", df)
    flags = pd.DataFrame([
        {"season": 2025, "overall_pick": 1, "round": 1, "team_id": 1, "member_id": A,
         "player_id": 1, "autodraft": False, "keeper": False},
        {"season": 2025, "overall_pick": 5, "round": 2, "team_id": 1, "member_id": A,
         "player_id": 2, "autodraft": True, "keeper": False},
    ])
    write_table(conn, "league_draft_flags", flags)
    return conn


def test_lineups_measure_points_left_hit_rate_and_bad_starts(lineups_league):
    l = mp.lineups(lineups_league, A)
    assert l["n"] == 2
    weeks = {w["week"]: w for w in l["weeks"]}
    assert weeks[1]["started"] == 51.0 and weeks[1]["optimal"] == 61.0 and weeks[1]["left"] == 10.0
    assert weeks[2]["left"] == 0.0
    assert l["bench_points_left"] == 5.0 and l["hit_rate"] == 0.5
    assert l["started_out"] == 1
    assert l["worst_week"] == {"season": 2025, "week": 1, "left": 10.0}
    assert l["moves_per_week"] is None or l["moves_per_week"] > 0


def test_draft_flags_give_an_autodraft_rate(lineups_league):
    d = mp.draft_flags(lineups_league, A)
    assert d["n"] == 2 and d["autodraft_rate"] == 0.5 and d["autodrafts"] == 1


def test_profile_assembles_every_section_with_names(lineups_league):
    p = mp.profile(lineups_league, A)
    assert p["member_id"] == A and p["display_name"] == "a"
    for key in ("finishes", "playoffs", "head_to_head", "luck", "waivers", "trades",
                "lineups", "draft_flags"):
        assert key in p
    assert p["head_to_head"][0]["opponent"]["display_name"] in {"b", "c", "d"}
    assert mp.profile(lineups_league, "{NOBODY}") is None


def test_overview_has_the_strip_and_a_grid_with_a_defining_line(lineups_league):
    o = mp.league_overview(lineups_league)
    assert [s["season"] for s in o["seasons"]] == [2024, 2025]
    grid = {m["member_id"]: m for m in o["members"]}
    assert grid[A]["titles"] == 1 and grid[A]["seasons"] == 2
    assert isinstance(grid[A]["defining_line"], str) and grid[A]["defining_line"]
    # D holds the league's extreme on luck (-0.33 over two seasons, n=6 >= 5):
    # rule 8 fires and D gets that line, not the fallback.
    assert grid[D]["defining_line"] == "the unluckiest record in the league"


def test_defining_line_falls_back_to_the_record_with_no_extreme():
    no_extreme = {"n": 0}
    rookie = {
        "finishes": {"n": 0, "seasons": []}, "playoffs": no_extreme, "luck": no_extreme,
        "waivers": no_extreme, "trades": no_extreme, "lineups": no_extreme,
        "draft_flags": no_extreme,
    }
    veteran = {
        "finishes": {"n": 1, "avg_finish": 3.0,
                     "seasons": [{"wins": 7, "losses": 6}]},
        "playoffs": no_extreme, "luck": no_extreme, "waivers": no_extreme,
        "trades": no_extreme, "lineups": no_extreme, "draft_flags": no_extreme,
    }
    facts_by_member = {"rookie": rookie, "veteran": veteran}
    assert mp.defining_line("rookie", facts_by_member) == "first season in the league"
    assert mp.defining_line("veteran", facts_by_member) == "7-6 last season, avg finish 3.0"
