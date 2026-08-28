"""The plan: who to target at each remaining turn, and why.

The availability table these tests run against is built by hand rather than
loaded from a corpus -- `tests/test_availability.py` is where the counting is
checked, and here the counts are only an input we want to be able to set to
0.02 exactly.
"""
import numpy as np
import pytest

from scoring import plan as pl
from scoring.availability import MAX_PICK, AvailabilityTable
from scoring.league import LeagueSettings

IDS = ["rb1", "rb2", "rb3", "wr1", "wr2", "wr3", "te1", "te2", "qb1", "qb2"]
POSITIONS = ["RB", "RB", "RB", "WR", "WR", "WR", "TE", "TE", "QB", "QB"]
PROJ = [300.0, 270.0, 240.0, 280.0, 265.0, 200.0, 150.0, 145.0, 260.0, 250.0]
NAMES = {"rb1": "Rb One", "rb2": "Rb Two", "rb3": "Rb Three",
         "wr1": "Wr One", "wr2": "Wr Two", "wr3": "Wr Three",
         "te1": "Te One", "te2": "Te Two", "qb1": "Qb One", "qb2": "Qb Two"}
ALWAYS_THERE = {pid: {} for pid in IDS}


def settings():
    return LeagueSettings(
        season=2026, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=2, bench=5, scoring={}, draft_type="SNAKE")


def make_table(taken, pooled=100):
    """`taken` is player_id -> {pick: how many of his drafts took him there}."""
    ids = list(taken)
    counts = np.zeros((len(ids), MAX_PICK + 1), dtype=np.int64)
    for i, pid in enumerate(ids):
        for pick, n in taken[pid].items():
            counts[i, pick] += n
    return AvailabilityTable(
        player_ids=np.array(ids, dtype=object),
        pooled=np.full(len(ids), pooled, dtype=np.int64),
        taken_by=np.cumsum(counts, axis=1),
        adp_curve={}, corpus_mtime=1.0)


def plan_args(table, **over):
    args = dict(
        proj=list(PROJ), positions=list(POSITIONS), player_ids=list(IDS),
        espn_rank=[float(i + 1) for i in range(len(IDS))],
        espn_adp=[float(i + 1) for i in range(len(IDS))],
        market_rank=[float(i + 1) for i in range(len(IDS))],
        byes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        health=[5] * len(IDS), names=dict(NAMES),
        roster_counts={}, settings=settings(), turns=[6, 11],
        picks_made=0, favourites=set(), table=table)
    args.update(over)
    return args


# --- edge -------------------------------------------------------------------

def test_edge_excludes_the_candidate_himself():
    """rb1's edge is what he is worth over the best OTHER running back who
    survives -- counting himself would make every certain survivor read 0."""
    edge = pl.edge_at([300.0, 290.0, 280.0], ["RB", "RB", "WR"],
                      [1.0, 1.0, 1.0])
    assert edge[0] == pytest.approx(10.0)
    assert edge[1] == pytest.approx(-10.0)
    # The only WR has nobody to be measured against: waiting gets you none.
    assert edge[2] == pytest.approx(280.0)


def test_expected_best_excluding_matches_the_vectorised_form():
    rng = np.random.default_rng(7)
    proj = rng.uniform(50, 300, 40)
    avail = rng.uniform(0, 1, 40)
    positions = ["RB" if i % 2 else "WR" for i in range(40)]
    edge = pl.edge_at(proj, positions, avail)
    for i in (0, 1, 17, 39):
        one = pl.expected_best_excluding(proj, avail, positions,
                                         positions[i], i)
        assert edge[i] == pytest.approx(proj[i] - one)


def test_a_certain_survivor_is_the_whole_expectation():
    got = pl.expected_best_excluding([300.0, 290.0, 100.0], [0.0, 1.0, 1.0],
                                     ["RB", "RB", "RB"], "RB", 0)
    assert got == pytest.approx(290.0)


# --- the plan ---------------------------------------------------------------

def test_a_two_percent_survivor_is_neither_target_nor_alternate():
    """The Gibbs case. Pick 6 of the draft, and the consensus number one is
    taken before it in 98 of 100 recorded drafts. The old room recommended
    him anyway; the plan must not name him at all."""
    table = make_table({**ALWAYS_THERE, "rb1": {5: 98}})
    turns = pl.build_plan(**plan_args(table, turns=[6, 11], picks_made=0))
    first = turns[0]
    assert first["target"]["player_id"] != "rb1"
    assert "rb1" not in [a["player_id"] for a in first["alternates"]]
    assert first["target"]["lasts_pct"] >= 50.0


def test_a_favourite_clears_the_lower_bar_a_stranger_does_not():
    """Both are there 40% of the time -- above the favourites threshold and
    below the general one."""
    table = make_table({**ALWAYS_THERE, "rb1": {5: 60}, "rb2": {5: 60}})
    args = plan_args(table, turns=[6], picks_made=0)
    assert pl.build_plan(**args)[0]["target"]["player_id"] != "rb1"

    args = plan_args(table, turns=[6], picks_made=0, favourites={"rb1"})
    assert pl.build_plan(**args)[0]["target"]["player_id"] == "rb1"


def test_the_favourite_bonus_breaks_a_tie():
    table = make_table(ALWAYS_THERE)
    proj = list(PROJ)
    proj[1] = proj[0]           # rb2 is now exactly as good as rb1
    args = plan_args(table, turns=[6], picks_made=0, proj=proj,
                     favourites={"rb2"})
    assert pl.build_plan(**args)[0]["target"]["player_id"] == "rb2"


def test_a_planned_name_is_not_planned_again():
    table = make_table(ALWAYS_THERE)
    turns = pl.build_plan(**plan_args(table, turns=[6, 11, 22]))
    named = [t["target"]["player_id"] for t in turns if t["target"]]
    assert len(named) == len(set(named))
    for i, turn in enumerate(turns):
        later = [t["target"]["player_id"] for t in turns[i + 1:] if t["target"]]
        for alt in turn["alternates"]:
            assert alt["player_id"] not in later


def test_a_position_at_its_roster_cap_is_never_targeted():
    """A second kicker cannot be rostered at all, so his projection is not
    the question -- NEED_WEIGHTS zeroes him however good he looks."""
    table = make_table({**ALWAYS_THERE, "k1": {}})
    args = plan_args(
        table, turns=[6, 11, 22], roster_counts={"K": 1},
        player_ids=IDS + ["k1"], positions=POSITIONS + ["K"],
        proj=PROJ + [400.0], espn_rank=[float(i + 1) for i in range(11)],
        espn_adp=[float(i + 1) for i in range(11)],
        market_rank=[float(i + 1) for i in range(11)],
        byes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        health=[5] * 11, names={**NAMES, "k1": "K One"})
    named = [t["target"]["player_id"] for t in pl.build_plan(**args)
             if t["target"]]
    assert "k1" not in named


def test_the_payload_is_exactly_the_agreed_shape():
    table = make_table(ALWAYS_THERE)
    turns = pl.build_plan(**plan_args(table, turns=[6, 11]))
    assert [set(t) for t in turns] == [
        {"pick_no", "round", "target", "alternates"}] * 2
    assert turns[0]["pick_no"] == 6 and turns[0]["round"] == 1
    assert turns[1]["pick_no"] == 11 and turns[1]["round"] == 2
    target = turns[0]["target"]
    assert set(target) == {"player_id", "lasts_pct", "edge_pts", "pros", "cons"}
    assert len(turns[0]["alternates"]) == 2
    for alt in turns[0]["alternates"]:
        assert set(alt) == {"player_id", "lasts_pct", "edge_pts",
                            "pros", "cons"}
        assert alt["pros"] == [] and alt["cons"] == []


def test_the_last_turn_has_no_next_pick_to_be_measured_against():
    table = make_table(ALWAYS_THERE)
    turns = pl.build_plan(**plan_args(table, turns=[6, 11]))
    assert turns[0]["target"]["edge_pts"] is not None
    assert turns[1]["target"]["edge_pts"] is None


def test_a_turn_with_nobody_eligible_says_so():
    table = make_table({pid: {6: 100} for pid in IDS})
    turns = pl.build_plan(**plan_args(table, turns=[6], picks_made=0))
    assert turns[0]["target"] is None
    assert turns[0]["alternates"] == []


# --- target_now -------------------------------------------------------------

def test_target_now_treats_everyone_available_as_certain():
    """On the clock there is no chance about it: the man is there. His
    survival is still reported, because it is the cost of waiting."""
    table = make_table({**ALWAYS_THERE, "rb1": {10: 98}})
    cards = pl.target_now(**plan_args(table, turns=[6, 20], picks_made=5))
    assert cards[0]["player_id"] == "rb1"
    assert cards[0]["lasts_pct"] == pytest.approx(2.0)
    assert len(cards) == 3
    assert cards[0]["pros"], "the cards carry their own reasons"


def test_target_now_prices_against_the_turn_after_this_one():
    table = make_table(ALWAYS_THERE)
    cards = pl.target_now(**plan_args(table, turns=[6, 20], picks_made=5))
    assert cards[0]["edge_pts"] == pytest.approx(30.0)


# --- reasons ----------------------------------------------------------------

def test_reasons_come_in_the_order_the_spec_fixes():
    pros, cons = pl.reasons_for(
        favourite=True, lasts=0.82, at_pick=11, edge=12.0, next_pick=22,
        position="RB", slot="RB1", espn_rank=10.0, espn_adp=12.0,
        bye=7, health=5, bye_mates=[])
    assert pros == ["★ favourite",
                    "82% still there at pick 11",
                    "+12.0 pts over the next RB you'd get at pick 22",
                    "fills RB1"]
    assert cons == []


def test_a_negative_edge_is_a_con_in_the_waiting_language():
    pros, cons = pl.reasons_for(
        favourite=False, lasts=0.6, at_pick=11, edge=-4.2, next_pick=22,
        position="WR", slot=None, espn_rank=None, espn_adp=None,
        bye=None, health=None, bye_mates=[])
    assert cons == ["−4.2 pts vs waiting for WR"]


def test_an_adp_ahead_of_the_rank_says_he_may_go_earlier():
    def call(rank, adp):
        return pl.reasons_for(
            favourite=False, lasts=0.6, at_pick=11, edge=None, next_pick=None,
            position="RB", slot=None, espn_rank=rank, espn_adp=adp,
            bye=None, health=None, bye_mates=[])

    assert "ADP 20 vs ESPN 30 — may go earlier" in call(30.0, 20.0)[1]
    assert "ADP 30 vs ESPN 20 — may last" in call(20.0, 30.0)[0]
    pros, cons = call(20.0, 25.0)       # inside the gap: neither
    assert not any("ADP" in r for r in pros + cons)


def test_a_bye_that_stacks_and_a_health_meter_that_does_not_are_cons():
    pros, cons = pl.reasons_for(
        favourite=False, lasts=0.6, at_pick=11, edge=None, next_pick=None,
        position="RB", slot=None, espn_rank=None, espn_adp=None,
        bye=7, health=2, bye_mates=["Rb One", "Wr One"])
    assert cons == ["bye week 7 stacks with Rb One", "health 2/5"]


def test_one_team_mate_on_the_bye_is_not_a_stack():
    pros, cons = pl.reasons_for(
        favourite=False, lasts=0.6, at_pick=11, edge=None, next_pick=None,
        position="RB", slot=None, espn_rank=None, espn_adp=None,
        bye=7, health=4, bye_mates=["Rb One"])
    assert cons == []


def test_no_list_runs_past_four_reasons():
    pros, cons = pl.reasons_for(
        favourite=True, lasts=0.82, at_pick=11, edge=12.0, next_pick=22,
        position="RB", slot="RB1", espn_rank=20.0, espn_adp=40.0,
        bye=7, health=1, bye_mates=["A", "B"])
    assert len(pros) == pl.MAX_REASONS
    assert 0 < len(cons) <= pl.MAX_REASONS


def test_the_plan_names_the_roster_mate_a_bye_would_stack_with():
    """Everyone on this board is on bye 7 and one is already rostered, so by
    the third turn the plan is stacking a third player onto that week."""
    table = make_table(ALWAYS_THERE)
    turns = pl.build_plan(**plan_args(
        table, turns=[6, 11, 22], byes=[7] * len(IDS),
        names={**NAMES, "old1": "Old One"},
        roster_byes={"old1": 7}))
    third = turns[2]["target"]
    assert any("bye week 7" in c for c in third["cons"]), third["cons"]
