"""Live draft mode: one long-lived session, refreshed as picks land.

`make sim` pays 0.9s building the board, 1.1s building the pool and 14.9s in
`fit_all` on every invocation. During a draft none of that changes -- the
coefficients come from history, the board and pool are static -- so the
session builds them once and every refresh costs only `search_pick`.
"""
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline.db import read_table
from pipeline.espn_live import build_crosswalk
from scoring import league as league_mod
from scoring.board import build_board
from scoring.draft_model import FEATURE_NAMES, fit_all
from scoring.draft_sim import build_pool

# Pinned, not generated. See DraftSession.seed.
DEFAULT_SEED = 20260811


@dataclass(frozen=True)
class DraftSession:
    my_slot: int
    # The ESPN league being polled. Task 6's poller builds its URL from this;
    # it lives on the session because a session is tied to one draft.
    league_id: str
    slot_managers: dict
    settings: object
    pool: object
    betas: dict
    crosswalk: dict
    board_fingerprint: str
    # Pinned for the session's lifetime, never derived from a clock or a
    # counter. `search_pick` already uses common random numbers within a
    # call; holding the seed fixed ACROSS calls is what makes a changed
    # recommendation mean a changed board rather than a different sample.
    seed: int
    started_at: datetime


def board_fingerprint(board: pd.DataFrame) -> str:
    """Identity of the draftable set, order-independent.

    The session caches pool indices. If the board is rebuilt underneath it,
    those indices point at different players and every recommendation is
    silently about the wrong person. Row order is an artifact of assembly,
    not a change in who is draftable, so it is sorted out.
    """
    ids = sorted(str(p) for p in board["player_id"])
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def build_session(conn, my_slot: int, seed: int = DEFAULT_SEED) -> DraftSession:
    """Everything expensive, once. Roughly 17s against a real database."""
    settings = league_mod.load(conn)
    board = build_board(conn, settings=settings)
    pool = build_pool(conn, board, settings)
    fits = fit_all(conn, settings)
    pooled = fits.get("__pooled__", np.zeros(len(FEATURE_NAMES)))
    betas = {m: fits.get(m, pooled) for m in fits if m != "__pooled__"}

    draft_order = read_table(conn, "draft_order")
    slot_managers = dict(zip(draft_order["slot"].astype(int),
                             draft_order["manager"]))
    league_row = read_table(conn, "league")
    league_id = (str(league_row["league_id"].iloc[0])
                 if not league_row.empty else "")
    return DraftSession(
        my_slot=my_slot, league_id=league_id, slot_managers=slot_managers,
        settings=settings, pool=pool, betas=betas,
        crosswalk=build_crosswalk(conn),
        board_fingerprint=board_fingerprint(board), seed=seed,
        started_at=datetime.now(timezone.utc))


import threading

from scoring.draft_sim import _drafted_state, search_pick, snake_slots

STALE_AFTER_SECONDS = 15
# Measured on the live board: 25 -> 4.7s, 100 -> 19.5s, 200 -> 35.2s.
# Picks arrive every ~20-30s; the clock is ~90s.
ROLLOUTS_FAR, ROLLOUTS_NEAR, ROLLOUTS_NOW = 25, 100, 200


def rollouts_for(picks_until: int) -> int:
    """Budget by the time actually available.

    Because the seed is pinned, raising N between refreshes refines the same
    scenario set rather than resampling a different one -- the estimate
    converges instead of jumping. A negative distance means a desync ran the
    pick count past my turn; treat that as "now" rather than searching
    nothing.
    """
    if picks_until <= 0:
        return ROLLOUTS_NOW
    if picks_until <= 2:
        return ROLLOUTS_NEAR
    return ROLLOUTS_FAR


def picks_until_turn(settings, my_slot: int, picks_made: int) -> int:
    """How many other teams pick before I do.

    Zero means I am on the clock -- including both halves of a snake turn,
    where I pick twice with nobody in between.
    """
    slots = snake_slots(settings.teams, settings.rounds)
    for offset in range(picks_made, len(slots)):
        if slots[offset] == my_slot:
            return offset - picks_made
    return 0


def _is_stale(last_poll_at, now) -> bool:
    """Never having polled counts as stale: the UI must not present an empty
    board as a current one."""
    if last_poll_at is None:
        return True
    return (now - last_poll_at).total_seconds() > STALE_AFTER_SECONDS


def register_live_routes(app, conn):
    """Mount live-draft endpoints. In-process state only, same lifetime as
    `create_app`'s connection -- a restart mid-draft means starting again,
    which is correct: the cached pool would be stale anyway."""
    state = {"session": None, "last_poll_at": None, "unmapped": [],
             "candidates": [], "as_of_pick": None, "computing_for": None}
    lock = threading.Lock()

    def _recompute(session, picks_made):
        """Run one search and store it, unless a newer pick landed first.

        A result computed against a board that has since changed is worse
        than no result -- it recommends a player who may already be gone. So
        the pick count is captured before the search and re-checked after;
        if it moved, this result is discarded rather than served.
        """
        cur = conn.cursor()
        try:
            taken, taken_order = _drafted_state(cur, session.pool)
            until = picks_until_turn(session.settings, session.my_slot, picks_made)
            frame = search_pick(
                session.pool, session.settings, session.slot_managers,
                session.my_slot, taken, session.betas,
                n_rollouts=rollouts_for(until), seed=session.seed,
                taken_order=taken_order)
        finally:
            cur.close()
        with lock:
            if state["as_of_pick"] is not None and state["as_of_pick"] > picks_made:
                return          # superseded while we were computing
            state["candidates"] = frame.to_dict(orient="records")
            state["as_of_pick"] = picks_made

    @app.post("/api/live/start")
    def live_start(my_slot: int):
        with lock:
            if state["session"] is not None:
                return {"active": True, "reused": True}
        cur = conn.cursor()
        try:
            session = build_session(cur, my_slot)
        finally:
            cur.close()
        with lock:
            state.update({"session": session, "candidates": [],
                          "as_of_pick": None, "unmapped": [],
                          "last_poll_at": None})
        return {"active": True, "reused": False,
                "board_fingerprint": session.board_fingerprint,
                "seed": session.seed}

    @app.get("/api/live/state")
    def live_state():
        now = datetime.now(timezone.utc)
        with lock:
            session = state["session"]
            if session is None:
                return {"active": False, "picks_made": 0, "on_the_clock": None,
                        "candidates": [], "candidates_as_of_pick": None,
                        "last_poll_at": None, "stale": True,
                        "unmapped_picks": []}
            snapshot = dict(state)
        cur = conn.cursor()
        try:
            picks_made = cur.execute("SELECT count(*) FROM drafted").fetchone()[0]
        finally:
            cur.close()
        slots = snake_slots(session.settings.teams, session.settings.rounds)
        on_clock = slots[picks_made] if picks_made < len(slots) else None
        return {
            "active": True,
            "picks_made": int(picks_made),
            "on_the_clock": on_clock,
            "my_slot": session.my_slot,
            "candidates": snapshot["candidates"],
            "candidates_as_of_pick": snapshot["as_of_pick"],
            "last_poll_at": (snapshot["last_poll_at"].isoformat()
                             if snapshot["last_poll_at"] else None),
            "stale": _is_stale(snapshot["last_poll_at"], now),
            "unmapped_picks": snapshot["unmapped"],
        }

    @app.post("/api/live/stop")
    def live_stop():
        with lock:
            state.update({"session": None, "candidates": [],
                          "as_of_pick": None, "unmapped": [],
                          "last_poll_at": None})
        return {"active": False}

    return state, _recompute
