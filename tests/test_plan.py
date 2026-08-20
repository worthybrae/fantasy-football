"""The draft plan: rounds_plan, cliffs, and what each is allowed to claim."""
import numpy as np
import pytest
from scoring import league
from scoring.draft_model import FEATURE_NAMES
from scoring.draft_sim import SimPool, snake_slots
from scoring.plan import CLIFF_POSITIONS, MIN_PLAN_PCT, build_plan

S = league.default_settings()   # QB/2RB/2WR/TE/2FLEX/K/DST, 5 bench = 15 rounds
POSITIONS = ("RB", "WR", "QB", "TE", "K", "DST")


def _pool(n=180):
    """Big enough that an 8-team, 15-round draft (120 picks) never exhausts
    it -- a plan built off a pool that runs dry reports cliffs of 0.0 for a
    reason that has nothing to do with positional scarcity."""
    return SimPool(
        player_id=np.array([f"p{i}" for i in range(n)]),
        norm=np.array([f"player {i}" for i in range(n)]),
        position=np.array(list(POSITIONS) * (n // len(POSITIONS))),
        adp_rank=np.arange(1, n + 1, dtype=float),
        points=np.linspace(300.0, 60.0, n),
        availability=np.full(n, 90.0),
        vor=np.linspace(300.0, 60.0, n),
        market_rank=np.arange(1, n + 1, dtype=float),
        age=np.full(n, np.nan),
        no_track_record=np.full(n, True), hype=np.full(n, np.nan),
        trend=np.zeros(n))


def _betas(managers):
    return {m: np.zeros(len(FEATURE_NAMES)) for m in managers}


def _args(my_slot=2, n=180):
    pool = _pool(n)
    managers = {i: f"m{i}" for i in range(1, S.teams + 1)}
    return dict(pool=pool, settings=S, slot_managers=managers,
                my_slot=my_slot, taken=np.zeros(len(pool.player_id), bool),
                betas=_betas(managers.values()), seed=7)


def test_plan_covers_every_one_of_my_turns_at_the_right_pick_numbers():
    """An off-by-one in the snake walk would shift the whole plan a round
    without failing anything -- the shape stays valid, it just describes
    somebody else's draft."""
    a = _args(my_slot=2)
    plan = build_plan(taken_order=[], n_drafts=6, **a)
    slots = snake_slots(S.teams, S.rounds)
    expected = [o + 1 for o, s in enumerate(slots) if s == 2]
    assert [e["pick"] for e in plan["rounds_plan"]] == expected
    assert [e["round"] for e in plan["rounds_plan"]] == list(range(1, S.rounds + 1))
    assert plan["my_slot"] == 2 and plan["as_of_pick"] == 0


def test_plan_percentages_are_a_ranked_distribution_with_noise_dropped():
    plan = build_plan(taken_order=[], n_drafts=20, **_args())
    for e in plan["rounds_plan"]:
        pcts = [d["pct"] for d in e["positions"]]
        assert pcts == sorted(pcts, reverse=True), e
        assert all(p >= MIN_PLAN_PCT for p in pcts), e
        # Dropping sub-threshold branches and integer rounding both lose a
        # little, but the listed branches must still be most of the mass.
        assert 80 <= sum(pcts) <= 100, e
        assert all(d["position"] in POSITIONS for d in e["positions"]), e


def test_plan_is_deterministic_under_a_fixed_seed():
    """The plan is shown next to the recommendations and is meant to be
    stable between picks; a plan that reshuffles on every poll reads as
    noise no matter how good the underlying number is."""
    a, b = _args(), _args()
    assert (build_plan(taken_order=[], n_drafts=8, **a)
            == build_plan(taken_order=[], n_drafts=8, **b))


def test_past_rounds_report_what_happened_rather_than_predicting_it():
    """The simulation starts from the current state, so it has nothing to
    say about a pick already made. Reporting a distribution for one would be
    fiction dressed as a forecast."""
    a = _args(my_slot=2)
    pool = a["pool"]
    slots = snake_slots(S.teams, S.rounds)
    # Play the first 20 picks straight down the board.
    taken_order = list(range(20))
    a["taken"][:20] = True
    plan = build_plan(taken_order=taken_order, n_drafts=8, **a)
    assert plan["as_of_pick"] == 20

    past = [e for e in plan["rounds_plan"] if e["is_past"]]
    assert past, "slot 2 picks inside the first 20"
    for e in past:
        idx = taken_order[e["pick"] - 1]
        assert e["actual"] == str(pool.position[idx])
        assert e["positions"] == [{"position": e["actual"], "pct": 100}]
    # And the rounds still to come are predictions, not history.
    future = [e for e in plan["rounds_plan"] if not e["is_past"]]
    assert future and all(e["actual"] is None for e in future)
    assert [e["pick"] for e in past] == sorted(e["pick"] for e in past)
    assert max(e["pick"] for e in past) <= 20 < min(e["pick"] for e in future)


def test_cliffs_are_non_negative_and_stop_one_turn_short():
    """A cliff is a DROP in the best player still available, and the pool
    only shrinks, so a negative cliff means the walk marked players gone in
    the wrong order. The final turn has no next turn to wait for."""
    plan = build_plan(taken_order=[], n_drafts=10, **_args())
    assert len(plan["cliffs"]) == len(plan["best_available"]) - 1
    for c in plan["cliffs"]:
        assert set(c["by_position"]) == set(CLIFF_POSITIONS)
        assert all(v >= 0.0 for v in c["by_position"].values()), c


def test_best_available_is_the_board_i_faced_including_the_player_i_took():
    """Pins the round-1 numbers exactly, which nothing else here can.

    From slot 1 the first pick of the draft is mine and nothing has happened
    yet, so `best_available` at round 1 is simply the pool's best at each
    position -- no simulation involved, identical in every run. That makes
    it the one place an exact assertion is possible, and it is worth having:
    every other check in this file compares the plan against itself, so a
    walk that marks a pick gone one step too early stays perfectly
    self-consistent while describing a board that never existed. The
    snapshot must be taken BEFORE my own pick is removed -- the player I am
    about to draft was available to me.
    """
    a = _args(my_slot=1)
    pool = a["pool"]
    plan = build_plan(taken_order=[], n_drafts=4, **a)
    first = plan["best_available"][0]
    assert first["round"] == 1 and first["pick"] == 1
    for pos in CLIFF_POSITIONS:
        want = float(pool.points[pool.position == pos].max())
        assert first["by_position"][pos] == pytest.approx(want, abs=0.05), pos


def test_cliffs_line_up_with_the_rounds_they_describe():
    """`best_available` skips rounds already played, so its first entry is
    not round 1 mid-draft. Every entry carries its own round number and the
    cliff at r is exactly the drop from r to r+1."""
    a = _args(my_slot=2)
    a["taken"][:20] = True
    plan = build_plan(taken_order=list(range(20)), n_drafts=8, **a)
    future_rounds = [e["round"] for e in plan["rounds_plan"] if not e["is_past"]]
    assert future_rounds[0] > 1, "fixture must actually resume mid-draft"
    assert [e["round"] for e in plan["best_available"]] == future_rounds
    assert [c["round"] for c in plan["cliffs"]] == future_rounds[:-1]
    ba = {e["round"]: e["by_position"] for e in plan["best_available"]}
    for c in plan["cliffs"]:
        for pos in CLIFF_POSITIONS:
            drop = ba[c["round"]][pos] - ba[c["round"] + 1][pos]
            assert c["by_position"][pos] == pytest.approx(max(0.0, drop), abs=0.2)


def test_cliffs_cover_only_the_positions_a_cliff_means_something_for():
    """K and DST are held out of contention by `need_kind` until the end of
    the draft regardless of their tiers, so a cliff for them is not a
    decision anyone makes -- but they may still appear in the plan itself,
    because the roster requires one of each."""
    plan = build_plan(taken_order=[], n_drafts=10, **_args())
    assert set(CLIFF_POSITIONS) == {"QB", "RB", "WR", "TE"}
    for c in plan["cliffs"]:
        assert "K" not in c["by_position"] and "DST" not in c["by_position"]
    for e in plan["best_available"]:
        assert set(e["by_position"]) == set(CLIFF_POSITIONS)
    # Nothing in rounds_plan is restricted that way.
    listed = {d["position"] for e in plan["rounds_plan"] for d in e["positions"]}
    assert listed <= set(POSITIONS)
