import numpy as np
import pandas as pd
import pytest

from scoring.gain import (expected_best_next, fills_slot, need_kind,
                          need_weight, rank_available)
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
    # _roster_cap gives RB a cap of starters(2)+2 = 4: a fifth RB is capped
    # even though the overall roster (4 players) is nowhere near `rounds`.
    assert need_kind(settings(), {"RB": 4}, "RB") == "capped"


def test_need_weight_is_zero_when_capped():
    assert need_weight(settings(), {"K": 1}, "K") == 0.0


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


class _pool:
    """Minimal stand-in for SimPool: gain.py reads four fields."""

    def __init__(self, player_id, position, points, vor):
        self.player_id = np.array(player_id, dtype=object)
        self.position = np.array(position, dtype=object)
        self.points = np.array(points, dtype=float)
        self.vor = np.array(vor, dtype=float)
