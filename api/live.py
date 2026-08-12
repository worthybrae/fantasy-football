"""Live draft mode: one long-lived session, refreshed as picks land.

`make sim` pays 0.9s building the board, 1.1s building the pool and 14.9s in
`fit_all` on every invocation. During a draft none of that changes -- the
coefficients come from history, the board and pool are static -- so the
session builds them once and every refresh costs only `search_pick`.
"""
import dataclasses
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
    # None until the socket names our team (see live_connect's on_change) --
    # build_session runs before the socket ever connects, so at construction
    # time this is genuinely unknown whenever the pasted URL carried no
    # teamId=. Frozen, so learning it later means dataclasses.replace-ing the
    # whole session, never mutating this field in place.
    my_slot: int | None
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


def build_session(conn, my_slot: int | None, seed: int = DEFAULT_SEED,
                  league_id: str = "") -> DraftSession:
    """Everything expensive, once. Roughly 17s against a real database.

    `league_id` is passed in rather than looked up, because the database has
    no record of it: the `league` table is `(season, settings_json)` and no
    table anywhere carries a league id. An earlier version read
    `league["league_id"]` and raised KeyError the first time a real connect
    ran. The caller already has it -- parsed from the URL the user pasted,
    which is the only place it exists.
    """
    settings = league_mod.load(conn)
    board = build_board(conn, settings=settings)
    pool = build_pool(conn, board, settings)
    fits = fit_all(conn, settings)
    pooled = fits.get("__pooled__", np.zeros(len(FEATURE_NAMES)))
    betas = {m: fits.get(m, pooled) for m in fits if m != "__pooled__"}

    draft_order = read_table(conn, "draft_order")
    slot_managers = dict(zip(draft_order["slot"].astype(int),
                             draft_order["manager"]))
    return DraftSession(
        my_slot=my_slot, league_id=league_id, slot_managers=slot_managers,
        settings=settings, pool=pool, betas=betas,
        crosswalk=build_crosswalk(board),
        board_fingerprint=board_fingerprint(board), seed=seed,
        started_at=datetime.now(timezone.utc))


import re
import threading

from fastapi import HTTPException
from pydantic import BaseModel

from pipeline.draft_listener import DraftListener, run_listener
from pipeline.espn_league import STATE_PATH, parse_league_id
from pipeline.espn_live import apply_picks
from scoring.draft_sim import _drafted_state, search_pick, snake_slots


class ConnectBody(BaseModel):
    # No my_slot field. A URL carrying teamId= resolves it immediately (see
    # _team_id_from_url / _slot_for_team); one that doesn't (the natural
    # waiting-room URL to paste) leaves it None until the socket's TOKEN
    # frame names our team (see live_connect's on_change) -- there is no
    # third case left for a human to fill in by hand.
    url: str


def _resolve_league_id(url: str) -> str:
    """League id from anything ESPN shows you.

    A real league's draft URL, a mock's, or a bare id all carry the same
    thing. Treating a mock as an ordinary league is deliberate: it is what
    lets a mock draft rehearse the whole system without a special path
    through it that would then be the untested one on draft night.
    """
    try:
        return parse_league_id(url)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="No league id in that URL. Open your draft room and copy "
                   "the address bar -- it should contain leagueId=.")


def _team_id_from_url(url: str) -> int | None:
    """ESPN team id from a pasted URL, if present.

    The draft socket speaks team ids -- `SELECTING 2 30000` and `SELECTED 2
    4429795 2` are both keyed by teamId, never by draft slot -- so reading
    it off the URL when ESPN put it there beats asking the drafter to state
    their own slot number by hand.
    """
    m = re.search(r"teamId=(\d+)", url or "")
    return int(m.group(1)) if m else None


def _slot_for_team(cur, team_id: int):
    """Translate an ESPN team id to a draft slot, or None if it cannot be.

    A team id is not a draft slot -- team 4 is not necessarily drafting 4th.
    The translation goes through the manager in two hops, and it has to,
    because the obvious one-hop route does not exist: `draft_teams` has a
    `slot` column and **every row of it is null**, all 48 across six seasons.
    It was never populated. An earlier version of this function read it
    directly and crashed on `int(NAType)` the first time a real league hit it.

    So: `draft_teams` maps team id to manager, and `draft_order` -- which the
    user sets for the upcoming draft -- maps manager to slot.

    Returns None rather than raising when the chain breaks, which is the
    normal case for a mock draft: its managers are strangers who appear in
    neither table. Turn detection does not actually need a slot, because the
    socket says `SELECTING <teamId>` outright; the slot is only wanted for
    the simulator's snake ordering. A caller that needs one should say so and
    handle its absence, not receive a fabricated number -- a wrong slot
    attributes every pick to the wrong manager and nothing downstream can
    detect that it happened.
    """
    teams = read_table(cur, "draft_teams")
    if teams.empty or "manager" not in teams.columns:
        return None
    rows = teams[teams["team_id"] == team_id]
    if rows.empty:
        return None
    manager = rows.sort_values("season").iloc[-1]["manager"]

    order = read_table(cur, "draft_order")
    if order.empty or "manager" not in order.columns:
        return None
    mine = order[order["manager"] == manager]
    if mine.empty or pd.isna(mine.iloc[0]["slot"]):
        return None
    return int(mine.iloc[0]["slot"])


STALE_AFTER_SECONDS = 15
# Measured on the live board: 25 -> 4.7s, 100 -> 19.5s, 200 -> 35.2s.
# Picks arrive every ~20-30s; the clock is ~90s.
ROLLOUTS_FAR, ROLLOUTS_NEAR, ROLLOUTS_NOW = 25, 100, 200

# How long _stop_listener waits for the previous listener thread to notice
# stop_event and exit (browser close included) before refusing a reconnect
# rather than risking two sockets for the same team. A module constant, not
# a literal default, so a test can shrink it and exercise the refusal path
# without a real ten-second wait.
LISTENER_STOP_TIMEOUT = 10.0


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
             "candidates": [], "as_of_pick": None, "computing_for": None,
             # Bumped by live_start and live_stop. A stop/start cycle resets
             # as_of_pick to None, which blinds the pick-count guard below --
             # a stale _recompute launched under the old session would see
             # `state["as_of_pick"] is not None` as False and sail through.
             # The generation is the guard that catches session identity
             # rather than pick count; the two check different things and
             # dropping either leaves a hole.
             "generation": 0,
             # The socket listener's own lifecycle, set only by live_connect
             # and cleared only by _stop_listener. `listener` is the live
             # DraftListener instance -- checked by identity (`is`), not by
             # generation number, so a superseded listener's own in-flight
             # websocket callback (which is not gated by `listener_stop`;
             # see run_listener) can still recognise it is no longer the
             # active one and skip writing. `listener_thread` is what
             # _stop_listener joins on. `listener_error` carries the text of
             # any exception that killed the thread, so a dead listener is
             # visible on /api/live/state instead of failing silently.
             "listener": None, "listener_thread": None,
             "listener_stop": None, "listener_error": None}
    lock = threading.Lock()

    def _stop_listener(timeout: float = LISTENER_STOP_TIMEOUT) -> bool:
        """Signal the active listener thread to stop and wait for it to exit.

        Not called with `lock` held: joining a thread while holding it would
        block anyone else who needs `state` -- including, briefly, the very
        thread being joined, whose `on_change` callback takes the lock for
        its identity check (see live_connect). There is nothing under the
        lock this function needs atomically; it reads the two handles it
        needs, then does its blocking work outside it.

        Returns True once the thread is confirmed stopped (or none was
        running) and clears its state. Returns False if it did not exit
        within `timeout` -- callers must treat that as "a second listener
        may still be alive" and refuse to start a new one rather than risk
        two sockets for the same team.
        """
        with lock:
            stop_event = state["listener_stop"]
            thread = state["listener_thread"]
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                return False
        with lock:
            state["listener"] = None
            state["listener_thread"] = None
            state["listener_stop"] = None
        return True

    def _recompute(session, picks_made):
        """Run one search and store it, unless superseded meanwhile.

        A result computed against a board that has since changed is worse
        than no result -- it recommends a player who may already be gone. So
        the pick count and the session generation are both captured before
        the search and re-checked after: if either moved, this result is
        discarded rather than served.

        `session.my_slot` can still be None here -- the socket hasn't named
        our team yet -- and `search_pick` needs a real slot to index into
        (rosters, snake order, ...), not something to guess at. Skip the
        search rather than pass it a fabricated one; candidates stay empty
        until my_slot resolves, which /api/live/state already reports
        honestly via session.my_slot being null.
        """
        if session.my_slot is None:
            return
        with lock:
            generation = state["generation"]
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
            if state["generation"] != generation:
                return          # session stopped/restarted while computing
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
            state["generation"] += 1
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
                        "unmapped_picks": [], "listener_error": None,
                        "listener_alive": False}
            snapshot = dict(state)
        cur = conn.cursor()
        try:
            picks_made = cur.execute("SELECT count(*) FROM drafted").fetchone()[0]
        finally:
            cur.close()
        slots = snake_slots(session.settings.teams, session.settings.rounds)
        on_clock = slots[picks_made] if picks_made < len(slots) else None
        thread = snapshot["listener_thread"]
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
            # A dead listener is the worst failure mode this system has --
            # the board looks current and simply stops updating. Silence is
            # not an option: listener_error carries the exception that
            # killed the thread (None if it never had one, e.g. a session
            # built via /api/live/start with no socket at all), and
            # listener_alive is the thread's live status, so a hang with no
            # exception is still visible even though it sets no error.
            "listener_error": snapshot["listener_error"],
            "listener_alive": thread.is_alive() if thread is not None else False,
        }

    @app.post("/api/live/stop")
    def live_stop():
        stopped = _stop_listener()
        with lock:
            state["generation"] += 1
            state.update({"session": None, "candidates": [],
                          "as_of_pick": None, "unmapped": [],
                          "last_poll_at": None, "listener": None,
                          "listener_thread": None, "listener_stop": None,
                          "listener_error": None})
        return {"active": False, "listener_stopped": stopped}

    @app.post("/api/live/connect")
    def live_connect(body: ConnectBody):
        league_id = _resolve_league_id(body.url)

        cur = conn.cursor()
        try:
            # The socket speaks team ids, not slots (see _slot_for_team's
            # docstring). A teamId on the URL -- present on a draft-room URL,
            # absent on the waiting-room one ESPN redirects there from --
            # resolves my_slot immediately, an instant answer the connect
            # screen can show right away. Otherwise my_slot stays None: the
            # socket's own TOKEN frame will name our team once it connects
            # (see on_change below), and that -- not a guess -- is what fills
            # it in. Resolved before anything is stopped below: a request
            # that turns out to be invalid must never tear down a listener
            # that was working.
            team_id = _team_id_from_url(body.url)
            my_slot = _slot_for_team(cur, team_id) if team_id is not None else None

            # Exactly one listener may run at a time: the draft socket's URL
            # carries a token with no known derivation, so we can only ever
            # observe the one connection a real browser holds, never open a
            # second. Rather than refuse a reconnect outright -- which would
            # trap a caller recovering from a dead listener (listener_error
            # set) behind a separate, easy-to-forget /api/live/stop call --
            # the old listener is always stopped and joined FIRST, so the
            # two can never overlap. If it will not stop in time, refuse
            # instead of racing it.
            if not _stop_listener():
                raise HTTPException(
                    status_code=503,
                    detail="the previous listener did not stop in time -- try again")

            session = build_session(cur, my_slot, league_id=league_id)
        finally:
            cur.close()
        listener = DraftListener(session.crosswalk)
        stop_event = threading.Event()

        def pump():
            """Write every pick the socket reports, then recompute.

            `apply_picks` replaces the table wholesale, so re-folding the
            whole event stream on each change is correct rather than
            wasteful -- and it keeps `picks_from_events` the single
            definition of pick numbering.
            """
            def on_change():
                nonlocal session
                with lock:
                    # `stop_event` only asks run_listener's poll loop to
                    # exit -- it does not gate an in-flight websocket
                    # callback Playwright may already be running when a
                    # newer connect supersedes this listener. Checking
                    # identity here, not just trusting the stop signal, is
                    # what keeps a superseded listener's write a no-op
                    # instead of a race against the session that replaced
                    # it (see the module-level `state["listener"]` comment).
                    if state["listener"] is not listener:
                        return
                c2 = conn.cursor()
                try:
                    # my_slot is still unknown exactly when the URL carried
                    # no teamId -- the case DraftListener.on_frame's
                    # "newly learned team" signal exists for, so this check
                    # runs on the very first on_change, potentially before
                    # any pick has landed. DraftSession is frozen, so the
                    # resolved session replaces `session` (closed over by
                    # this function, one-to-one with `listener`) rather than
                    # mutating it; the replacement is mirrored into
                    # state["session"] under the same identity guard as
                    # every other write here, so /api/live/state picks up
                    # the real my_slot on its very next read. If
                    # _slot_for_team still can't resolve it (a mock draft's
                    # opponents, say), my_slot just stays None -- this retries
                    # every future on_change rather than giving up once.
                    if session.my_slot is None and listener.my_team_id is not None:
                        resolved = _slot_for_team(c2, listener.my_team_id)
                        if resolved is not None:
                            session = dataclasses.replace(session, my_slot=resolved)
                            with lock:
                                if state["listener"] is listener:
                                    state["session"] = session
                    apply_picks(c2, listener.picks())
                    made = c2.execute("SELECT count(*) FROM drafted").fetchone()[0]
                finally:
                    c2.close()
                _recompute(session, made)

            try:
                run_listener(listener, body.url, STATE_PATH,
                            on_change=on_change, stop_event=stop_event)
            except Exception as exc:      # noqa: BLE001 -- any failure (bad
                # url, Playwright missing, ESPN unreachable) must reach
                # /api/live/state instead of dying silently on a daemon
                # thread with a stderr trace nobody watches during a live
                # draft. Only recorded if this is still the active listener
                # -- a superseded one that's mid-shutdown raising on its way
                # out must not clobber the error of whatever replaced it.
                with lock:
                    if state["listener"] is listener:
                        state["listener_error"] = str(exc)

        thread = threading.Thread(target=pump, daemon=True)
        with lock:
            state.update({"session": session, "listener": listener,
                          "listener_thread": thread, "listener_stop": stop_event,
                          "listener_error": None,
                          "candidates": [], "as_of_pick": None,
                          "unmapped": [], "last_poll_at": None})
            state["generation"] = state.get("generation", 0) + 1
        thread.start()
        # `my_slot` is echoed back deliberately -- None here means genuinely
        # undetected yet, not a default of some kind. When it IS resolved
        # (from the URL's teamId now, or the socket's TOKEN shortly after),
        # it went through draft_teams -> draft_order by manager, and a
        # league that re-randomised its draft order since draft_order was
        # last set would get a silently wrong answer that no data source
        # here can detect. A human glancing at "you're drafting from slot 4"
        # catches that in a second; nothing else catches it at all. So the
        # real value (or its absence) is returned for the connect screen to
        # show, never a guess standing in for either.
        return {"connected": True, "league_id": league_id,
                "board_fingerprint": session.board_fingerprint,
                "my_slot": session.my_slot}

    return state, _recompute
