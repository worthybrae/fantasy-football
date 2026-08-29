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
    assert set(target) == {"player_id", "lasts_pct", "edge_pts",
                           "edge_at_pick", "pros", "cons"}
    assert len(turns[0]["alternates"]) == 2
    for alt in turns[0]["alternates"]:
        assert set(alt) == {"player_id", "lasts_pct", "edge_pts",
                            "edge_at_pick", "pros", "cons"}
        assert alt["pros"] == [] and alt["cons"] == []
    # THE EDGE IS PRICED AT THE TURN AFTER THIS ONE, and says so: turn 6 is
    # measured against pick 11, and the last turn against nothing.
    assert target["edge_at_pick"] == 11
    assert all(a["edge_at_pick"] == 11 for a in turns[0]["alternates"])
    assert turns[1]["target"]["edge_at_pick"] is None


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
    # The rank comes first after the star because it is the first thing the
    # order is built from, and five pros is room for the slot as well.
    assert pros == ["★ favourite",
                    "ESPN's #10 overall",
                    "82% still there at pick 11",
                    "+12.0 pts over the next RB you'd get at pick 22",
                    "fills RB1"]
    assert cons == []


def test_a_negative_edge_is_a_con_in_the_waiting_language():
    pros, cons = pl.reasons_for(
        favourite=False, lasts=0.6, at_pick=11, edge=-4.2, next_pick=22,
        position="WR", slot=None, espn_rank=None, espn_adp=None,
        bye=None, health=None, bye_mates=[])
    assert cons == ["−4.2 pts vs waiting for WR at pick 22"]


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


def test_no_list_runs_past_its_cap():
    pros, cons = pl.reasons_for(
        favourite=True, lasts=0.82, at_pick=11, edge=12.0, next_pick=22,
        position="RB", slot="RB1", espn_rank=20.0, espn_adp=40.0,
        bye=7, health=1, bye_mates=["A", "B"])
    assert len(pros) == pl.MAX_PROS == 5
    assert 0 < len(cons) <= pl.MAX_CONS == 4


def test_a_starred_target_keeps_the_two_reasons_that_explain_the_pick():
    """Five pros, because four was one short. A favourite spent his four on
    the star, the rank, the chance and the edge -- and dropped the slot he
    fills and the drop-off that says why he was taken over a higher-ranked
    name, which is the pair that explains the pick."""
    pros, _cons = pl.reasons_for(
        favourite=True, lasts=0.65, at_pick=6, edge=53.7, next_pick=11,
        position="RB", slot="RB2", espn_rank=7.0, espn_adp=7.0,
        bye=None, health=5, drop_off=True)
    assert pros == ["★ favourite",
                    "ESPN's #7 overall",
                    "65% still there at pick 6",
                    "+53.7 pts over the next RB you'd get at pick 11",
                    "biggest drop-off at RB before pick 11"]


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


# --- the order is ESPN's, moved by the roster and the stars -----------------

def test_the_edge_decides_between_neighbours_and_never_across_the_board():
    """Round 12 of a twelve-team draft, roster QB1/RB4/WR3/TE2. The
    quarterback gains 9.1 points over the next one and the receiver 2.2.

    Two places apart on ESPN's list, that drop-off is the whole difference
    between them and it takes the pick. Twenty places apart it takes
    nothing: points are not comparable across positions, which is how a
    quarterback ESPN ranks 26th came to be recommended over the 7th player
    on the board."""
    ids = ["wr_hi", "wr_lo", "qb_hi", "qb_lo"]
    positions = ["WR", "WR", "QB", "QB"]
    proj = [173.0, 170.8, 300.0, 290.9]
    roster = {"QB": 1, "RB": 4, "WR": 3, "TE": 2}
    table = make_table(dict.fromkeys(ids, {}))

    near = pl.build_plan(**board_args(
        table, ids, positions, proj, roster_counts=roster,
        turns=[133, 145]))[0]
    assert near["target"]["player_id"] == "qb_hi"
    assert near["target"]["edge_pts"] == pytest.approx(9.1)
    assert ("biggest drop-off at QB before pick 145"
            in near["target"]["pros"])

    far = pl.build_plan(**board_args(
        table, ids, positions, proj, roster_counts=roster,
        espn_rank=[1.0, 2.0, 40.0, 41.0], espn_adp=[1.0, 2.0, 40.0, 41.0],
        market_rank=[1.0, 2.0, 40.0, 41.0], turns=[133, 145]))[0]
    assert far["target"]["player_id"] == "wr_hi"
    assert far["target"]["edge_pts"] == pytest.approx(2.2)
    # He led on ESPN's order too, so there is nothing to explain away.
    assert not any("drop-off" in pro for pro in far["target"]["pros"])
    assert [a["player_id"] for a in far["alternates"]] == ["wr_lo", "qb_hi"]


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
    """rb1, rb2 and rb3 are the target and the alternates at turn one. At
    turn two the only running backs the plan can still expect are rb4 and
    rb5, so rb4's edge is 280 - 100 = 180. Counting rb1 would price him at
    -20, and the card would argue against its own pick."""
    ids = ["rb1", "rb2", "rb3", "rb4", "rb5", "rb6"]
    table = make_table(dict.fromkeys(ids, {}))
    turns = pl.build_plan(**board_args(
        table, ids, ["RB"] * 6,
        [300.0, 290.0, 285.0, 280.0, 100.0, 90.0], turns=[6, 19, 30]))
    assert turns[0]["target"]["player_id"] == "rb1"
    assert [a["player_id"] for a in turns[0]["alternates"]] == ["rb2", "rb3"]
    assert turns[1]["target"]["player_id"] == "rb4"
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
    ids = ["rb1", "rb2", "rb3", "rb4", "wr1"]
    table = make_table(dict.fromkeys(ids, {}))
    turns = pl.build_plan(**board_args(
        table, ids, ["RB", "RB", "RB", "RB", "WR"],
        [300.0, 290.0, 280.0, 270.0, 50.0], turns=[6, 19, 30]))
    assert "fills RB1" in turns[0]["target"]["pros"]
    assert "fills RB2" in turns[1]["target"]["pros"]


def test_health_level_bands_the_board_the_way_the_room_draws_it():
    got = pl.health_level([5.0, 9.9, 10.0, 12.9, 13.0, 14.9, 15.0, 16.2,
                           16.3, 20.0])
    assert list(got) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert np.isnan(pl.health_level([np.nan])[0])


def test_the_favourite_bonus_is_exactly_eight_ranks():
    """A tier, not a round. The starred receiver at ESPN 15 goes ahead of a
    stranger at 8 and still behind a stranger at 6 -- all three the same
    position, so the roster moves none of them.

    THE TURN IS PICK 15, at or past all three ADPs, so nobody is reaching
    and the bonus is the only thing moving anybody. It is a single turn, so
    nobody is waitable either: the two other things in the priority are
    switched off to measure this one."""
    ids = ["wr6", "wr8", "wr15"]
    table = make_table(dict.fromkeys(ids, {}))
    args = board_args(table, ids, ["WR", "WR", "WR"],
                      [200.0, 190.0, 180.0], espn_rank=[6.0, 8.0, 15.0],
                      espn_adp=[6.0, 8.0, 15.0], market_rank=[6.0, 8.0, 15.0],
                      favourites={"wr15"}, turns=[15])
    turn = pl.build_plan(**args)[0]
    assert turn["target"]["player_id"] == "wr6"
    assert [a["player_id"] for a in turn["alternates"]] == ["wr15", "wr8"]
    # Seven ranks would not have been enough, which is what makes it a
    # measured bonus rather than a thumb on the scale.
    plain = pl.build_plan(**board_args(
        table, ids, ["WR", "WR", "WR"], [200.0, 190.0, 180.0],
        espn_rank=[6.0, 8.0, 15.0], espn_adp=[6.0, 8.0, 15.0],
        market_rank=[6.0, 8.0, 15.0], turns=[15]))[0]
    assert [a["player_id"] for a in plain["alternates"]] == ["wr8", "wr15"]


def test_with_nothing_to_choose_on_the_drop_off_the_order_is_espns():
    """The control for the window. Every man here is the last of his
    position on the board, so nobody is priced (see `_best_other`) and every
    edge is zero: with nothing for the drop-off to say, the target and the
    alternates are ESPN's first three, in ESPN's order."""
    ids = ["a", "b", "c", "d", "e"]
    table = make_table(dict.fromkeys(ids, {}))
    turn = pl.build_plan(**board_args(
        table, ids, ["RB", "WR", "TE", "QB", "DST"],
        [300.0, 280.0, 150.0, 260.0, 90.0], turns=[6, 19]))[0]
    assert turn["target"]["edge_pts"] == pytest.approx(0.0)
    assert turn["target"]["player_id"] == "a"
    assert [x["player_id"] for x in turn["alternates"]] == ["b", "c"]


# --- the owner's case ---------------------------------------------------------

# The screenshot's four names, plus the next man at each of their three
# positions -- the players the edge is measured against. A position with one
# man left on it is not priced at all (see `_best_other`), so a board of
# four names would have no drop-off to compare.
OWNER_IDS = ["gibbs", "cmc", "arsb", "allen", "rb3", "wr2", "qb2"]
OWNER_POS = ["RB", "RB", "WR", "QB", "RB", "WR", "QB"]
OWNER_RANK = [1.0, 7.0, 8.0, 26.0, 40.0, 41.0, 42.0]
# Josh Allen outprojects every one of them and his position falls off a
# cliff behind him (+47), which is exactly why the old score put him first.
# McCaffrey's own drop-off is bigger (+53.7) and St. Brown's smaller (+34).
OWNER_PROJ = [310.0, 300.0, 290.0, 380.0, 245.0, 256.0, 333.0]
# The screenshot's percentages, as counts out of a hundred recorded drafts:
# taken before pick 6 in 98, 35, 9 and 1 of them.
OWNER_TABLE = {"gibbs": {5: 98}, "cmc": {5: 35}, "arsb": {5: 9},
               "allen": {5: 1}, "rb3": {}, "wr2": {}, "qb2": {}}


def owner_args(**over):
    """The live room before pick 1 of an 8-team draft, from seat 6."""
    args = dict(
        proj=list(OWNER_PROJ), positions=list(OWNER_POS),
        player_ids=list(OWNER_IDS), espn_rank=list(OWNER_RANK),
        espn_adp=list(OWNER_RANK), market_rank=list(OWNER_RANK),
        byes=[5, 6, 7, 8, 9, 10, 11], health=[5] * 7,
        names={p: p for p in OWNER_IDS}, roster_counts={},
        settings=settings(), turns=[6, 11], picks_made=0,
        favourites={"allen", "arsb"}, table=make_table(OWNER_TABLE))
    args.update(over)
    return args


def test_the_room_does_not_target_a_quarterback_espn_ranks_twenty_sixth():
    """THE OWNER'S SCREENSHOT. Seat 6, before a pick has been made. The room
    recommended Josh Allen -- ESPN's 26th player, a starred favourite at 99%
    to last, "+47 pts over the next QB" -- over Christian McCaffrey and
    Amon-Ra St. Brown. Quarterback is deep in a one-quarterback league and
    the edge said otherwise, which is what a raw point difference compared
    across positions does.

    Pick 11 is five picks away, so every one of these four is a man this
    seat could probably still have at it: they are all waitable and the
    fifteen ranks cancel. What is left is ESPN's order moved by the star and
    by the reach -- St. Brown 8 - 8 = 0, McCaffrey 7, Allen 26 - 8 + 4.5 for
    being fifteen picks ahead of his own ADP = 22.5 -- and the starred
    receiver takes the card. Allen is two tiers away from it."""
    turn = pl.build_plan(**owner_args())[0]
    assert turn["pick_no"] == 6
    assert turn["target"]["player_id"] == "arsb"
    assert turn["target"]["edge_pts"] == pytest.approx(34.0)
    assert [a["player_id"] for a in turn["alternates"]] == ["cmc", "allen"]
    # The 2%-to-last consensus number one is not on the card at all.
    assert turn["target"]["player_id"] != "gibbs"
    assert "gibbs" not in [a["player_id"] for a in turn["alternates"]]
    # The edge is priced at the NEXT turn, and every string that names a
    # pick names the pick it was measured at.
    assert turn["target"]["edge_at_pick"] == 11
    assert turn["target"]["pros"][:2] == ["★ favourite", "ESPN's #8 overall"]
    assert "+34.0 pts over the next WR you'd get at pick 11" in \
        turn["target"]["pros"]
    assert "91% still there at pick 6" in turn["target"]["pros"]
    # McCaffrey's position falls off harder (+53.7), so the card does not
    # claim the drop-off it did not win on -- and it says out loud that the
    # man it named could probably have been had at pick 11 anyway.
    assert not any("drop-off" in pro for pro in turn["target"]["pros"])
    assert ("you could probably wait — 91 % still there at pick 11"
            in turn["target"]["cons"])


def test_no_edge_carries_a_quarterback_up_the_board():
    """Allen's +47 is the second biggest on this board and he is still not
    the target, an alternate on the edge, or anywhere near the card: the
    edge breaks ties inside two ranks and he is twenty-two behind."""
    turn = pl.build_plan(**owner_args(
        proj=[310.0, 300.0, 290.0, 900.0, 245.0, 256.0, 333.0]))[0]
    assert turn["target"]["player_id"] == "arsb"
    assert [a["player_id"] for a in turn["alternates"]] == ["cmc", "allen"]


def test_a_quarterback_ranked_twenty_sixth_is_not_the_pick_over_the_field():
    """The same board with nobody starred: the star was not what was wrong
    with the recommendation. While a running back or a receiver ESPN ranks
    inside the top fifteen is eligible, the quarterback is not the target at
    any turn."""
    turns = pl.build_plan(**owner_args(favourites=set()))
    named = [t["target"]["player_id"] for t in turns if t["target"]]
    assert named == ["cmc", "rb3"]
    assert "allen" not in named
    # He is an alternate at the first turn, which is the honest place for
    # him: the third name, not the recommendation.
    assert [a["player_id"] for a in turns[0]["alternates"]] == ["arsb",
                                                                "allen"]


def test_the_target_now_cards_follow_the_same_order():
    """On the clock, with the same board. Nobody is filtered for being
    unlikely to last -- they are all there this second -- and the same
    priorities pick the same three names."""
    cards = pl.target_now(**owner_args(turns=[6, 11], picks_made=5))
    # Five picks in, Gibbs is 61% to last rather than 2%, so he is waitable
    # like everybody else here and the fifteen ranks cancel again: St. Brown
    # (8 - 8), Gibbs (1) and McCaffrey (7), in that order, with the first
    # two inside two ranks of each other and the bigger drop-off (34 against
    # 10) settling which of the pair leads.
    assert [c["player_id"] for c in cards] == ["arsb", "gibbs", "cmc"]
    assert cards[0]["edge_pts"] == pytest.approx(34.0)
    assert cards[0]["edge_at_pick"] == 11
    # Allen is twenty-two ranks off the best priority once his reach is
    # counted, and +47 buys him nothing.
    assert "allen" not in [c["player_id"] for c in cards]


# --- can you wait? ----------------------------------------------------------

# The owner's second complaint, with the seat he actually drafts from: seat 6
# of an eight-team draft, turns 6, 11, 22 and 27. Josh Allen is ESPN's 26th
# player with an ADP of 21, and he is starred. The four names ahead of him
# are the running backs and receivers ESPN ranks in the teens; the three
# behind him are the field each position is priced against.
WAIT_IDS = ["rb_hi", "wr_hi", "wr_2", "rb_2", "allen",
            "rb_lo", "wr_lo", "qb_lo"]
WAIT_POS = ["RB", "WR", "WR", "RB", "QB", "RB", "WR", "QB"]
WAIT_RANK = [12.0, 14.0, 15.0, 16.0, 26.0, 44.0, 46.0, 48.0]
WAIT_ADP = [12.0, 14.0, 15.0, 16.0, 21.0, 44.0, 46.0, 48.0]
# Allen outprojects everybody and the next quarterback is 47 points behind
# him, which is the whole of the old argument for taking him in round 2.
WAIT_PROJ = [310.0, 290.0, 285.0, 300.0, 380.0, 245.0, 256.0, 333.0]
# The pick each was taken at, out of a hundred recorded drafts. Allen lasts
# to pick 11 in ninety of them and to 22 in sixty; conditioned on having
# reached 21, he lasts to 27 in forty. The four ahead of him are 60-70% to
# be gone by 22, which is what makes them picks and him a wait.
WAIT_TAKEN = {"rb_hi": {9: 30, 15: 40}, "wr_hi": {9: 40, 16: 35},
              "wr_2": {9: 40, 16: 35}, "rb_2": {9: 30, 15: 40},
              "allen": {8: 10, 15: 30, 24: 36},
              "rb_lo": {}, "wr_lo": {}, "qb_lo": {}}


def wait_args(**over):
    args = dict(
        proj=list(WAIT_PROJ), positions=list(WAIT_POS),
        player_ids=list(WAIT_IDS), espn_rank=list(WAIT_RANK),
        espn_adp=list(WAIT_ADP), market_rank=list(WAIT_RANK),
        byes=[5, 6, 7, 8, 9, 10, 11, 12], health=[5] * 8,
        names={p: p for p in WAIT_IDS}, roster_counts={},
        settings=settings(), turns=[6, 11, 22, 27], picks_made=0,
        favourites={"allen"}, table=make_table(WAIT_TAKEN))
    args.update(over)
    return args


def test_a_starred_quarterback_who_will_last_is_not_the_round_two_pick():
    """THE OWNER'S COMPLAINT. Josh Allen recommended at pick 11 of an
    eight-team draft: ESPN's 26th player, ten picks ahead of his own ADP,
    and 60% to still be there at pick 22. The running back ESPN ranks 16th
    is 30% to be there -- so the wait is the whole argument, and it names
    the man who will be gone.

    Allen is not filtered out. He is eligible, he is starred, and he is the
    second name on the card: 26 - 8 for the star + 2 for a ten-pick reach
    against the sixteenth-ranked back's 16."""
    turns = pl.build_plan(**wait_args())
    at_11 = turns[1]
    assert at_11["pick_no"] == 11
    assert at_11["target"]["player_id"] == "rb_2"
    assert at_11["target"]["player_id"] != "allen"
    # 70% of recorded drafts have taken him by 22: that is the pro the pick
    # is made on, and it is a new one.
    assert ("likely gone before your next pick (70 %)"
            in at_11["target"]["pros"])
    # He is the honest second name, not the recommendation.
    assert at_11["alternates"][0]["player_id"] == "allen"
    # And the turn before it belongs to the twelfth-ranked back, who is
    # 70% to be gone by 11 himself.
    assert turns[0]["target"]["player_id"] == "rb_hi"


def test_the_same_quarterback_eleven_picks_later_is_the_pick():
    """Pick 22, with the top of the board gone and 21 picks in. Allen is
    40% to last to pick 27 now, so the fifteen ranks that held him off at 11
    are not charged; his ADP is behind us, so the reach is not charged
    either. ESPN's 26th player, starred, is the best thing on the board
    that will not still be there -- and this time the room says so."""
    turn = pl.build_plan(**wait_args(
        turns=[22, 27], picks_made=21,
        roster_counts={"RB": 1, "WR": 1}))[0]
    assert turn["pick_no"] == 22
    assert turn["target"]["player_id"] == "allen"
    assert ("likely gone before your next pick (60 %)"
            in turn["target"]["pros"])
    # Nothing on his card calls it a reach any more.
    assert not any("reach" in con for con in turn["target"]["cons"])


def test_a_favourite_who_will_be_gone_beats_a_stranger_who_will_last():
    """Spec section 1's fourth case. The starred fifteenth is 30% to still
    be there at my next turn and the strangers ESPN ranks 6th and 10th are
    certain to be: the starred one is the pick, because the other two are
    picks I can also make at pick 11.

    THE CONTROL is the same board with nobody lasting, where ESPN's 6th
    goes back ahead of the starred 15th -- which is the favourites bonus
    being a tier and not a round, exactly as it was before the wait
    existed."""
    ids = ["wr6", "wr10", "wr15"]
    ranks = [6.0, 10.0, 15.0]
    proj = [200.0, 195.0, 190.0]
    common = dict(espn_rank=ranks, espn_adp=ranks, market_rank=ranks,
                  settings=settings(), turns=[6, 11], picks_made=0,
                  favourites={"wr15"})
    lasting = make_table({"wr6": {}, "wr10": {}, "wr15": {9: 70}})
    turn = pl.build_plan(**board_args(
        lasting, ids, ["WR", "WR", "WR"], proj, **common))[0]
    assert turn["target"]["player_id"] == "wr15"

    nobody_lasts = make_table({pid: {9: 70} for pid in ids})
    turn = pl.build_plan(**board_args(
        nobody_lasts, ids, ["WR", "WR", "WR"], proj, **common))[0]
    assert turn["target"]["player_id"] == "wr6"


def test_a_favourite_is_waitable_only_when_he_is_really_quite_safe():
    """The two lines, from one board. A stranger is waited on at a coin
    flip; a starred player has to be 65% safe before the plan will risk the
    same wait, which is the same allowance his eligibility gets."""
    board = pl._Board(
        proj=[100.0, 100.0], positions=["WR", "WR"],
        player_ids=["star", "stranger"], espn_rank=[10.0, 10.0],
        espn_adp=[10.0, 10.0], market_rank=[10.0, 10.0], byes=None,
        health=None, names={}, favourites={"star"})
    assert list(board.waitable([0.30, 0.30])) == [pl.WAITABLE, pl.WAITABLE]
    # 40% gone: the stranger is still waited on, the favourite is taken.
    assert list(board.waitable([0.40, 0.40])) == [0.0, pl.WAITABLE]
    assert list(board.waitable([0.60, 0.60])) == [0.0, 0.0]


def test_the_reach_is_half_a_rank_for_every_pick_past_six():
    """A six-pick reach is free, a ten-pick reach costs two ranks and a
    thirty-pick reach twelve. A player with no ADP is not reaching, and
    neither is one whose ADP is already behind us."""
    board = pl._Board(
        proj=[100.0] * 5, positions=["WR"] * 5,
        player_ids=["a", "b", "c", "d", "e"], espn_rank=[20.0] * 5,
        espn_adp=[17.0, 21.0, 41.0, np.nan, 5.0],
        market_rank=[20.0] * 5, byes=None, health=None, names={},
        favourites=set())
    reach = board.reach(11)
    assert list(reach) == [6.0, 10.0, 30.0, 0.0, 0.0]
    priority = board.priority(np.zeros(5), np.zeros(5), reach)
    assert list(priority) == [20.0, 22.0, 32.0, 20.0, 20.0]


def test_the_wait_and_the_reach_say_themselves_on_the_card():
    """The three strings section 1 fixes, in the words it fixes them in."""
    pros, cons = pl.reasons_for(
        favourite=False, lasts=0.28, at_pick=22, edge=None, next_pick=27,
        position="RB", slot=None, espn_rank=30.0, espn_adp=30.0, bye=None,
        health=None, gone_next=0.72, waitable=False, reach=8.0, reach_at=22)
    assert "likely gone before your next pick (72 %)" in pros
    assert "reach: ADP 30 at pick 22" in cons

    pros, cons = pl.reasons_for(
        favourite=True, lasts=0.9, at_pick=22, edge=None, next_pick=27,
        position="QB", slot=None, espn_rank=26.0, espn_adp=21.0, bye=None,
        health=None, gone_next=0.19, waitable=True, reach=0.0, reach_at=22)
    assert cons[0] == "you could probably wait — 81 % still there at pick 27"
    assert not any("likely gone" in pro for pro in pros)


def test_the_survival_line_is_not_printed_twice_on_one_card():
    """On the clock `target_now` measures "still there" at the very turn the
    waiting is about, so the bare percentage and the line that judges it
    would be the same number as a pro and as a con."""
    pros, cons = pl.reasons_for(
        favourite=False, lasts=0.6, at_pick=13, edge=None, next_pick=13,
        position="RB", slot=None, espn_rank=5.0, espn_adp=5.0, bye=None,
        health=None, gone_next=0.4, waitable=True, reach=0.0, reach_at=4)
    assert not any("still there at pick 13" in pro for pro in pros)
    assert cons == ["you could probably wait — 60 % still there at pick 13"]


def test_a_kicker_is_not_targeted_before_the_round_kickers_go_in():
    """A deferred starter slot is 200 places back, which is "not on this
    board" without pretending the slot is filled. When the roster's
    remaining picks run down to its open slots, `need_kind` calls it a
    starter and the same rule targets him."""
    ids = ["k1", "wr1"]
    table = make_table(dict.fromkeys(ids, {}))
    args = dict(table=table, ids=ids, positions=["K", "WR"],
                proj=[130.0, 120.0])
    early = pl.build_plan(**board_args(
        args["table"], ids, ["K", "WR"], [130.0, 120.0],
        espn_rank=[5.0, 20.0], espn_adp=[5.0, 20.0], market_rank=[5.0, 20.0],
        roster_counts={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "DST": 1},
        turns=[100, 112, 124]))
    assert early[0]["target"]["player_id"] == "wr1"
    late = pl.build_plan(**board_args(
        table, ids, ["K", "WR"], [130.0, 120.0],
        espn_rank=[5.0, 20.0], espn_adp=[5.0, 20.0], market_rank=[5.0, 20.0],
        roster_counts={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "DST": 1},
        turns=[124]))
    assert late[0]["target"]["player_id"] == "k1"
