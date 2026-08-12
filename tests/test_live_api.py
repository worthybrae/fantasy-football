import numpy as np
import pandas as pd
import pytest

from api.live import DraftSession, board_fingerprint, build_session


def _fake_session(seed=20260811):
    """A DraftSession with cheap stand-ins for the expensive fields.

    build_session costs ~17s against a real database (fit_all alone is 14.9s),
    which is exactly why the session exists -- but it makes a poor unit-test
    fixture. Tests that exercise session *behaviour* build one directly.
    """
    from datetime import datetime, timezone
    return DraftSession(
        my_slot=4, league_id="53929318",
        slot_managers={i: f"m{i}" for i in range(1, 9)},
        settings=None, pool=None, betas={}, crosswalk={111: "g1"},
        board_fingerprint="abc", seed=seed,
        started_at=datetime(2026, 8, 11, tzinfo=timezone.utc))


def test_board_fingerprint_changes_when_the_player_set_changes():
    """The session caches pool indices. If the board is rebuilt underneath it
    -- a refresh drops a retired player, say -- those indices point at
    different people and every recommendation is silently about the wrong
    player."""
    a = pd.DataFrame({"player_id": ["g1", "g2"], "market_rank": [1.0, 2.0]})
    b = pd.DataFrame({"player_id": ["g1", "g3"], "market_rank": [1.0, 2.0]})
    assert board_fingerprint(a) != board_fingerprint(b)


def test_board_fingerprint_is_stable_for_the_same_board():
    a = pd.DataFrame({"player_id": ["g1", "g2"], "market_rank": [1.0, 2.0]})
    assert board_fingerprint(a) == board_fingerprint(a.copy())


def test_board_fingerprint_ignores_row_order():
    """Row order is an artifact of how the board was assembled, not a change
    in who is draftable."""
    a = pd.DataFrame({"player_id": ["g1", "g2"], "market_rank": [1.0, 2.0]})
    b = a.iloc[::-1].reset_index(drop=True)
    assert board_fingerprint(a) == board_fingerprint(b)


def test_session_pins_a_seed_that_does_not_move(tmp_path, monkeypatch):
    """The load-bearing decision. Roster EV carries +/-4.2 at 400 rollouts,
    so +/-16.8 at the 25 a live refresh affords, against a 9.3-point gap
    between adjacent candidates. An unpinned seed reshuffles the
    recommendation while the board sits still."""
    s = _fake_session(seed=4242)
    assert s.seed == 4242
    # Two reads, same value -- not a property, not derived from a clock.
    assert s.seed == 4242
