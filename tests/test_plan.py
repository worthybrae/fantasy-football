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
        adp_curve={}, corpus_mtime=1.0, max_pick_observed=MAX_PICK)


def settings12():
    """A twelve-team league, the shape the reviewer's cases are set in."""
    return LeagueSettings(
        season=2026, teams=12,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=1, bench=7, scoring={}, draft_type="SNAKE")


def board_args(table, ids, positions, proj, **over):
    """Plan arguments for an arbitrary little board."""
    n = len(ids)
    args = dict(
        proj=list(proj), positions=list(positions), player_ids=list(ids),
        espn_rank=[float(i + 1) for i in range(n)],
        espn_adp=[float(i + 1) for i in range(n)],
        market_rank=[float(i + 1) for i in range(n)],
        byes=[5 + (i % 9) for i in range(n)], health=[5] * n,
        names={p: p for p in ids}, roster_counts={}, settings=settings12(),
        turns=[6, 19], picks_made=0, favourites=set(), table=table)
    args.update(over)
    return args


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
    # The only WR has no field to be measured against, so he is not priced
    # at all rather than priced at his whole projection.
    assert edge[2] == pytest.approx(0.0)


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
    # Taken at pick 5 in every draft, so nobody survives to make pick 6.
    table = make_table({pid: {5: 100} for pid in IDS})
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


# --- the score is the roster's weight on what the pick GAINS -----------------

def test_a_bench_quarterback_nine_points_clear_beats_a_receiver_two_clear():
    """Round 12 of a twelve-team draft, roster QB1/RB4/WR3/TE2. Both are
    bench needs at 0.35, so the pick is decided by what it gains: 9.1 points
    against 2.2. Weighting the PROJECTION instead scored the quarterback
    -185.9 and the receiver -110.2, and the plan took the receiver."""
    ids = ["qb_hi", "qb_lo", "wr_hi", "wr_lo"]
    table = make_table(dict.fromkeys(ids, {}))
    turns = pl.build_plan(**board_args(
        table, ids, ["QB", "QB", "WR", "WR"], [300.0, 290.9, 173.0, 170.8],
        roster_counts={"QB": 1, "RB": 4, "WR": 3, "TE": 2},
        turns=[133, 145]))
    assert turns[0]["target"]["player_id"] == "qb_hi"
    assert turns[0]["target"]["edge_pts"] == pytest.approx(9.1)
    assert [a["player_id"] for a in turns[0]["alternates"]] == ["wr_hi",
                                                                "wr_lo"]


def test_a_position_at_its_cap_is_not_a_candidate_at_any_turn():
    """A second kicker cannot be rostered in any slot, starter or bench, so
    he is not on the list however the arithmetic reads."""
    ids = ["wr_hi", "wr_lo", "rb_hi", "rb_lo", "k1"]
    table = make_table(dict.fromkeys(ids, {}))
    args = board_args(
        table, ids, ["WR", "WR", "RB", "RB", "K"],
        [150.0, 145.0, 140.0, 138.0, 130.0],
        roster_counts={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1},
        turns=[100, 112])
    named = []
    for turn in pl.build_plan(**args):
        if turn["target"]:
            named.append(turn["target"]["player_id"])
        named += [a["player_id"] for a in turn["alternates"]]
    assert "k1" not in named
    cards = pl.target_now(**args)
    assert "k1" not in [c["player_id"] for c in cards]


def test_a_lone_tight_end_does_not_outrank_a_running_back_worth_more():
    """The 120-point tight end is the last one on the board, so waiting
    "costs" his whole projection -- a level, against the 5-point difference
    the running backs are separated by. Comparing those two decides the pick
    on which position happens to be thin in our own list."""
    ids = ["rb_hi", "rb_lo", "te1"]
    table = make_table(dict.fromkeys(ids, {}))
    turns = pl.build_plan(**board_args(
        table, ids, ["RB", "RB", "TE"], [220.0, 215.0, 120.0],
        roster_counts={"QB": 1, "WR": 2, "TE": 1}))
    assert turns[0]["target"]["player_id"] == "rb_hi"


def test_a_later_turn_is_not_priced_against_a_name_the_plan_already_spent():
    """rb1 and rb2 are the target and an alternate at turn one. At turn two
    the only running backs the plan can still expect are rb3 and rb4, so
    rb3's edge is 280 - 100 = 180. Counting rb1 would price him at -20 and
    the plan would take somebody else."""
    ids = ["rb1", "rb2", "rb3", "rb4", "wr1"]
    table = make_table(dict.fromkeys(ids, {}))
    turns = pl.build_plan(**board_args(
        table, ids, ["RB", "RB", "RB", "RB", "WR"],
        [300.0, 290.0, 280.0, 100.0, 50.0], turns=[6, 19, 30]))
    assert turns[0]["target"]["player_id"] == "rb1"
    assert "rb2" in [a["player_id"] for a in turns[0]["alternates"]]
    assert turns[1]["target"]["player_id"] == "rb3"
    assert turns[1]["target"]["edge_pts"] == pytest.approx(180.0)


def test_the_pick_on_the_clock_is_certain_inside_the_plan_too():
    """api/live.py passes the current pick as the plan's first turn. He is
    on the board now, so 70% of drafts having taken him by pick 6 is not a
    reason to plan around him -- it is the reason to take him."""
    table = make_table({**ALWAYS_THERE, "rb1": {6: 70}})
    turns = pl.build_plan(**plan_args(table, turns=[6, 19], picks_made=5))
    assert turns[0]["target"]["player_id"] == "rb1"
    assert turns[0]["target"]["lasts_pct"] == pytest.approx(100.0)
    # And the turn after it is priced on his real chances of lasting.
    assert pl.target_now(**plan_args(
        table, turns=[6, 19], picks_made=5))[0]["player_id"] == "rb1"


def test_the_roster_the_plan_builds_is_what_the_next_turn_needs():
    ids = ["rb1", "rb2", "rb3", "wr1"]
    table = make_table(dict.fromkeys(ids, {}))
    turns = pl.build_plan(**board_args(
        table, ids, ["RB", "RB", "RB", "WR"],
        [300.0, 290.0, 280.0, 50.0], turns=[6, 19, 30]))
    assert "fills RB1" in turns[0]["target"]["pros"]
    assert "fills RB2" in turns[1]["target"]["pros"]


def test_health_level_bands_the_board_the_way_the_room_draws_it():
    got = pl.health_level([5.0, 9.9, 10.0, 12.9, 13.0, 14.9, 15.0, 16.2,
                           16.3, 20.0])
    assert list(got) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert np.isnan(pl.health_level([np.nan])[0])
