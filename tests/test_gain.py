import numpy as np
import pandas as pd
import pytest

from scoring.gain import (available_by_vor, expected_best_next, fills_slot,
                          need_kind, need_weight, rank_available)
from scoring.league import LeagueSettings


def settings():
    return LeagueSettings(
        season=2026, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=2, bench=5, scoring={}, draft_type="SNAKE")


def test_need_kind_open_starter():
    assert need_kind(settings(), {"RB": 1}, "RB") == "starter"


def test_need_kind_flex_when_starters_full():
    assert need_kind(settings(), {"RB": 2}, "RB") == "flex"


def test_need_kind_bench_when_flex_full():
    # _roster_cap gives RB a cap of starters(2)+2 = 4, so a fourth RB would
    # itself be at the position's roster cap -- not a useful probe of the
    # "flex full, bench still open" branch. Use one extra RB and one extra
    # WR to fill both FLEX slots (RB starters 2 + 1 flex, WR starters 2 + 1
    # flex) while RB stays under its own cap.
    counts = {"RB": 3, "WR": 3}
    assert need_kind(settings(), counts, "RB") == "bench"


def test_need_kind_capped_at_the_roster_cap():
    # _roster_cap caps K at 1
    assert need_kind(settings(), {"K": 1}, "K") == "capped"


def test_need_kind_capped_when_position_hits_its_own_roster_cap():
    """A position at its own ceiling is capped even when the overall roster
    is nowhere near `rounds`.

    Read from `_roster_cap` rather than hardcoded: the caps are written
    relative to the league's starters and flex slots so they track a league
    that changes shape, and a literal here silently pins one league's
    arithmetic (it pinned RB at 4, and broke the day RB's honest depth
    became starters + flex + 2)."""
    from scoring.draft_sim import _roster_cap
    cap = _roster_cap(settings())["RB"]
    assert need_kind(settings(), {"RB": cap}, "RB") == "capped"
    assert need_kind(settings(), {"RB": cap - 1}, "RB") != "capped"


def test_need_weight_is_zero_when_capped():
    assert need_weight(settings(), {"K": 1}, "K") == 0.0


# --- Roster need is "is the slot open" AND "is now the time to fill it".
# Reported live: a kicker and a defense in the top fifteen at pick 19 (round
# 3), both on an open starter slot weighted 1.0 -- the same weight as an
# empty WR1 -- in a 16-round draft where neither slot gets filled before the
# last couple of rounds.


def test_need_kind_defers_a_slot_you_can_only_ever_hold_one_of():
    """K and DST in round 3: the slot is open, and it is not the round.

    `_roster_cap` puts both at 1, equal to their starter count, so a second
    one can never be rostered even on the bench -- the pick buys the slot
    and nothing else, and any later pick buys it just as well. With 13 of my
    15 picks still to come and two open starter slots, it can wait.
    """
    assert need_kind(settings(), {}, "K", turns_left=13) == "deferred"
    assert need_kind(settings(), {}, "DST", turns_left=13) == "deferred"
    assert need_weight(settings(), {}, "K", turns_left=13) < \
        need_weight(settings(), {}, "K", turns_left=2)


def test_need_kind_never_defers_a_position_with_depth_value():
    """Read off the roster shape, not a list of position names.

    Every other position's cap is above its starter count (RB 2+2, WR 2+2,
    TE 1+2, QB capped at 3), so a second one is a real bench asset and the
    slot is a full starter need from the first pick of the draft.
    """
    for pos in ("QB", "RB", "WR", "TE"):
        assert need_kind(settings(), {}, pos, turns_left=15) == "starter"


def test_need_kind_stops_deferring_when_the_slots_must_be_filled():
    """The lift, on the simulator's own line.

    `draft_sim.must_fill_positions` says a roster whose open starter slots
    have caught up with its remaining picks has to spend every one of them
    on those slots. Here that is two picks and two empty slots (K and DST),
    so both are starter needs again -- which is what puts a kicker back at
    the top of the list in the last rounds instead of the bottom.
    """
    counts = {"QB": 1, "RB": 4, "WR": 4, "TE": 2, }
    assert need_kind(settings(), counts, "K", turns_left=3) == "deferred"
    assert need_kind(settings(), counts, "K", turns_left=2) == "starter"
    assert need_kind(settings(), counts, "DST", turns_left=2) == "starter"


def test_need_kind_without_turns_left_is_exactly_what_it_always_was():
    """None is "the caller cannot say how many picks are left", which is
    every offline caller. A deferral rule with no idea how many picks remain
    would be guessing at the half of the question it exists to answer, so it
    does not fire at all."""
    assert need_kind(settings(), {}, "K") == "starter"
    assert need_kind(settings(), {}, "K", turns_left=None) == "starter"
    assert need_weight(settings(), {}, "K") == 1.0


def test_fills_slot_still_names_the_slot_a_deferred_pick_would_fill():
    """The label answers WHERE he goes, not whether now is the time -- that
    is the ranking's job. A deferred kicker still fills K, and only a
    capped one gets the "no slot" dash."""
    assert fills_slot(settings(), {}, "K", turns_left=13) == "K"
    assert fills_slot(settings(), {}, "DST", turns_left=13) == "DST"
    assert fills_slot(settings(), {"K": 1}, "K", turns_left=13) == "—"


def test_fills_slot_names_the_open_starter():
    assert fills_slot(settings(), {"RB": 1}, "RB") == "RB2"
    assert fills_slot(settings(), {}, "QB") == "QB"
    assert fills_slot(settings(), {"RB": 2}, "RB") == "FLEX"
    assert fills_slot(settings(), {"RB": 3, "WR": 3}, "RB") == "BENCH"
    assert fills_slot(settings(), {"K": 1}, "K") == "—"


def test_expected_best_next_is_the_best_when_survival_is_certain():
    assert expected_best_next([40.0, 90.0, 10.0], [1.0, 1.0, 1.0]) == 90.0


def test_expected_best_next_is_zero_when_nobody_survives():
    assert expected_best_next([40.0, 90.0], [0.0, 0.0]) == 0.0


def test_expected_best_next_weights_by_survival():
    """The best player survives half the time; otherwise the runner-up does.

    0.5*100 + 0.5*(1.0*60) = 80.
    """
    assert expected_best_next([100.0, 60.0], [0.5, 1.0]) == pytest.approx(80.0)


def test_gain_now_prefers_the_position_with_a_cliff_behind_it():
    """The Josh Allen case.

    The QB has the larger raw value over replacement, but the next QB is
    nearly as good, so waiting costs almost nothing. The WR is worth less in
    absolute terms but nothing comparable survives to the next pick.
    """
    pool = _pool(
        player_id=["qb1", "qb2", "wr1", "wr2"],
        position=["QB", "QB", "WR", "WR"],
        points=[391.0, 382.0, 241.0, 190.0],
        vor=[91.0, 82.0, 76.0, 25.0],
    )
    taken = np.zeros(4, dtype=bool)
    survive = np.array([0.31, 0.49, 0.22, 0.74])
    out = rank_available(pool, settings(), taken, {}, survive)
    assert out.iloc[0]["player_id"] == "wr1"
    assert out.set_index("player_id").loc["qb1", "gain_now"] < \
        out.set_index("player_id").loc["wr1", "gain_now"]


def test_rank_available_drops_taken_players():
    pool = _pool(player_id=["a", "b"], position=["RB", "RB"],
                 points=[200.0, 150.0], vor=[50.0, 0.0])
    taken = np.array([True, False])
    out = rank_available(pool, settings(), taken, {}, np.array([0.5, 0.5]))
    assert list(out["player_id"]) == ["b"]


def test_rank_available_zeroes_a_capped_position():
    pool = _pool(player_id=["k1"], position=["K"], points=[130.0], vor=[20.0])
    out = rank_available(pool, settings(), np.zeros(1, dtype=bool),
                         {"K": 1}, np.array([0.9]))
    assert out.iloc[0]["gain_now"] == 0.0
    assert out.iloc[0]["fills"] == "—"


def test_rank_available_sorts_capped_players_last():
    """A capped player must not outrank a useful one.

    NEED_WEIGHTS["capped"] is 0.0, so gain_now is identically 0.0 for every
    capped player -- and 0.0 sorts above every useful player whose gain is
    negative, which is most of the list once the good options at a position
    are gone. Late in a draft that put a block of rows the roster cannot
    even hold at the top of the ranked list.

    Here K is capped (one K already rostered, _roster_cap caps K at 1) and
    the two RBs both fill a real starting slot. The better RB's gain is
    positive and the worse RB's is negative -- and it is the negative one
    that the old sort put below the kicker.
    """
    pool = _pool(player_id=["k1", "rb1", "rb2"], position=["K", "RB", "RB"],
                 points=[130.0, 220.0, 180.0], vor=[20.0, 80.0, 40.0])
    out = rank_available(pool, settings(), np.zeros(3, dtype=bool),
                         {"K": 1}, np.array([0.9, 0.5, 0.5]))
    assert list(out["player_id"]) == ["rb1", "rb2", "k1"]
    # The premise: the kicker's gain really is the larger number.
    by_id = dict(zip(out["player_id"], out["gain_now"]))
    assert by_id["k1"] == 0.0
    assert by_id["rb2"] < 0.0
    # And `capped` is a sort key, not a served column.
    assert "capped" not in out.columns


def test_rank_available_sorts_a_deferred_kicker_below_a_negative_gain():
    """The owner's second report, reproduced: pick 19, round 3.

    No opponent takes a kicker or a defense inside the horizon, so both
    survive at 100%, `expected_best_next` for the position equals its own
    best available, and `gain_now` is EXACTLY 0.0 -- which outranks the -1
    and -2 of real players who are genuinely worth slightly less than what
    will survive. Houston Defense 12th and Brandon Aubrey 13th, above Malik
    Nabers and Chris Olave.

    A smaller need weight cannot fix that: any weight times exactly zero is
    exactly zero. Ordering the deferred block last is what does it, and the
    number it carries stays honest rather than being forced negative.
    """
    pool = _pool(player_id=["k1", "dst1", "wr0", "wr1", "wr2"],
                 position=["K", "DST", "WR", "WR", "WR"],
                 points=[130.0, 120.0, 250.0, 240.0, 230.0],
                 vor=[12.0, 5.0, 25.0, 19.0, 17.0])
    taken = np.zeros(5, dtype=bool)
    survive = np.array([1.0, 1.0, 0.90, 0.78, 0.30])
    out = rank_available(pool, settings(), taken, {}, survive, turns_left=13)
    assert list(out["player_id"]) == ["wr0", "wr1", "wr2", "k1", "dst1"]
    by_id = dict(zip(out["player_id"], out["gain_now"]))
    # The premise, unchanged by the fix: the two zeros really are the larger
    # numbers, and the two receivers really are worth less than what will
    # survive at their own position.
    assert by_id["k1"] == 0.0 and by_id["dst1"] == 0.0
    assert by_id["wr1"] < 0.0 and by_id["wr2"] < 0.0
    assert "deferred" not in out.columns
    # Without the deferral rule -- an offline caller that cannot say how
    # many picks are left -- this is exactly the reported ordering.
    old = rank_available(pool, settings(), taken, {}, survive)
    assert list(old["player_id"]) == ["wr0", "k1", "dst1", "wr1", "wr2"]


def test_rank_available_puts_the_kicker_back_on_top_when_it_must_be_filled():
    """The other end of the same rule: nothing is being hidden.

    Two picks left and two empty starter slots (K and DST), so
    `must_fill_positions` says both are now needs -- the kicker ranks on
    merit, above a receiver the roster can no longer start.
    """
    pool = _pool(player_id=["k1", "wr1"], position=["K", "WR"],
                 points=[130.0, 240.0], vor=[12.0, 19.0])
    counts = {"QB": 1, "RB": 4, "WR": 4, "TE": 2}
    taken = np.zeros(2, dtype=bool)
    survive = np.array([0.4, 0.9])
    out = rank_available(pool, settings(), taken, counts, survive,
                         turns_left=2)
    assert list(out["player_id"]) == ["k1", "wr1"]
    assert out.iloc[0]["fills"] == "K"
    assert out.iloc[0]["gain_now"] > 0


def test_available_by_vor_ranks_by_vor_points_descending():
    """Defect 2 (post-merge fix): before my_slot is known there is no
    roster to rank a pick FOR, but "who is still on the board" needs
    neither -- ranked here purely by the board's own vor_points."""
    pool = _pool(player_id=["a", "b", "c"], position=["RB", "WR", "QB"],
                 points=[200.0, 150.0, 300.0], vor=[10.0, 40.0, 25.0])
    taken = np.zeros(3, dtype=bool)
    out = available_by_vor(pool, taken)
    assert list(out["player_id"]) == ["b", "c", "a"]
    assert list(out["rank"]) == [1, 2, 3]


def test_available_by_vor_drops_taken_players():
    pool = _pool(player_id=["a", "b"], position=["RB", "RB"],
                 points=[200.0, 150.0], vor=[50.0, 10.0])
    taken = np.array([True, False])
    out = available_by_vor(pool, taken)
    assert list(out["player_id"]) == ["b"]


def test_available_by_vor_leaves_gain_now_survive_pct_fills_null():
    """Never a fabricated 0.0/0.0/"" -- a real recommendation has not been
    computed for any of these players, and the caller (api/live.py) must be
    able to tell that apart from a genuine zero."""
    pool = _pool(player_id=["a"], position=["RB"], points=[200.0], vor=[10.0])
    out = available_by_vor(pool, np.zeros(1, dtype=bool))
    row = out.iloc[0]
    assert row["gain_now"] is None
    assert row["survive_pct"] is None
    assert row["fills"] is None
    # And this must actually survive to JSON as `null`, not NaN -- the same
    # pitfall api/live.py's _float_or_none/_int_or_none helpers exist for
    # elsewhere in this codebase.
    import json
    encoded = json.dumps(out.to_dict(orient="records"))
    assert '"gain_now": null' in encoded


def test_available_by_vor_returns_the_same_columns_as_rank_available():
    """One LiveCandidate shape either way -- the frontend must not branch on
    two payload types (see the task brief)."""
    pool = _pool(player_id=["a"], position=["RB"], points=[200.0], vor=[10.0])
    empty = available_by_vor(pool, np.zeros(1, dtype=bool))
    ranked = rank_available(pool, settings(), np.zeros(1, dtype=bool), {},
                            np.array([0.5]))
    assert list(empty.columns) == list(ranked.columns)


def test_available_by_vor_empty_when_nobody_is_left():
    pool = _pool(player_id=["a"], position=["RB"], points=[200.0], vor=[10.0])
    out = available_by_vor(pool, np.array([True]))
    assert out.empty
    assert list(out.columns) == ["player_id", "position", "proj_points",
                                 "vor_points", "gain_now", "plan_steer",
                                 "survive_pct", "fills", "rank"]


class _pool:
    """Minimal stand-in for SimPool: gain.py reads four fields."""

    def __init__(self, player_id, position, points, vor):
        self.player_id = np.array(player_id, dtype=object)
        self.position = np.array(position, dtype=object)
        self.points = np.array(points, dtype=float)
        self.vor = np.array(vor, dtype=float)


def test_plan_steer_is_absent_and_inert_without_a_plan():
    """Every existing caller passes no plan_bias, so the default path must be
    bit-identical to what it was before steering existed -- not merely
    similar. A steer that leaks in at 0.0-ish rather than exactly 0.0 would
    reorder ties silently."""
    pool = _pool(player_id=["a", "b", "c"], position=["RB", "WR", "QB"],
                 points=[200.0, 190.0, 260.0], vor=[50.0, 48.0, 55.0])
    taken = np.zeros(3, dtype=bool)
    surv = np.array([0.5, 0.5, 0.5])
    out = rank_available(pool, settings(), taken, {}, surv)
    assert list(out["plan_steer"]) == [0.0, 0.0, 0.0]
    same = rank_available(pool, settings(), taken, {}, surv, plan_bias={})
    pd.testing.assert_frame_equal(out, same)
    pd.testing.assert_frame_equal(
        out, rank_available(pool, settings(), taken, {}, surv, plan_bias=None))


def test_plan_steer_moves_the_order_without_touching_the_reported_gain():
    """`gain_now` is displayed in the room and read as a points figure. The
    steer is allowed to change WHERE a row sorts and never what its gain
    says it is -- otherwise the number the user checks the tool against
    quietly stops being the measured one."""
    pool = _pool(player_id=["rb", "wr"], position=["RB", "WR"],
                 points=[200.0, 190.0], vor=[50.0, 46.0])
    taken = np.zeros(2, dtype=bool)
    surv = np.array([0.5, 0.5])
    plain = rank_available(pool, settings(), taken, {}, surv)
    assert list(plain["player_id"]) == ["rb", "wr"], "RB leads unaided"

    steered = rank_available(pool, settings(), taken, {}, surv,
                             plan_bias={"WR": 1.0})
    assert list(steered["player_id"]) == ["wr", "rb"], "the plan moved it"
    # ...but every gain_now is the same number it was, per player.
    before = dict(zip(plain["player_id"], plain["gain_now"]))
    after = dict(zip(steered["player_id"], steered["gain_now"]))
    assert before == after
    assert dict(zip(steered["player_id"], steered["plan_steer"]))["rb"] == 0.0


def test_plan_steer_is_bounded_by_a_real_edge():
    """The steer is priced at PLAN_STEER_POINTS so that a genuine gap still
    wins outright -- the cliffs it is derived from run to 86 points, and a
    plan must not talk you off a player who is that much better."""
    from scoring.gain import PLAN_STEER_POINTS
    pool = _pool(player_id=["rb", "wr"], position=["RB", "WR"],
                 points=[260.0, 190.0], vor=[50.0 + 2 * PLAN_STEER_POINTS, 46.0])
    out = rank_available(pool, settings(), np.zeros(2, dtype=bool), {},
                         np.array([0.5, 0.5]), plan_bias={"WR": 1.0})
    assert list(out["player_id"]) == ["rb", "wr"]


def test_plan_steer_cannot_resurrect_a_capped_position():
    """A capped row carries NEED_WEIGHTS["capped"] == 0.0, and the steer is
    weighted by the same factor, so full plan confidence in a position I have
    no room for still adds exactly nothing. Without that weighting the steer
    would lift unpickable rows off the bottom of the board."""
    pool = _pool(player_id=["k1", "rb"], position=["K", "RB"],
                 points=[130.0, 200.0], vor=[20.0, 5.0])
    out = rank_available(pool, settings(), np.zeros(2, dtype=bool),
                         {"K": 1}, np.array([0.9, 0.9]),
                         plan_bias={"K": 1.0})
    assert list(out["player_id"]) == ["rb", "k1"], "capped K still sorts last"
    steer = dict(zip(out["player_id"], out["plan_steer"]))
    assert steer["k1"] == 0.0
