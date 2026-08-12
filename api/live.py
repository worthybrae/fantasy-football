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
