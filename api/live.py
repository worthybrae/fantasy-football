"""Live draft mode: one long-lived session, refreshed as picks land.

`make sim` pays 0.9s building the board, 1.1s building the pool and 14.9s in
`fit_all` on every invocation. During a draft none of that changes -- the
coefficients come from history, the board and pool are static -- so the
session builds them once and every refresh costs only `survival` plus
`rank_available` (see SURVIVAL_ROLLOUTS below for why that is cheap).
"""
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline.db import get_conn, read_table
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
    # counter. `survival` seeds rollout i from (seed, i); holding the seed
    # fixed ACROSS calls is what makes a changed recommendation mean a
    # changed board rather than a different sample.
    seed: int
    started_at: datetime
    # The full board DataFrame build_session already pays to build. It used to
    # be discarded once `pool` was derived from it; /api/live/board keeps it so
    # it can serve rich per-player stats for every pick without rebuilding.
    # Defaults to None so callers that construct a DraftSession directly (test
    # fixtures, any future caller) need not supply it -- the board endpoint
    # then simply falls back to the raw player id for every cell.
    board: object = None
    # slot (1-based) -> ESPN team display name, fetched once on connect (see
    # pipeline.espn_teams.fetch_team_slots) and attached via
    # dataclasses.replace, since this dataclass is frozen. Empty until then,
    # and empty forever for a session with no league context (live_start) or
    # one whose team-name fetch failed -- /api/live/board falls back to
    # "Team {slot}" for any missing column. default_factory so the shared empty
    # default is not one dict aliased across every session.
    team_slots: dict = dataclasses.field(default_factory=dict)
    # player_id -> board row (as a dict), for POST /api/live/select's SELECT
    # payload lookup (see _espn_id_for). Built once in build_session from the
    # same `board.to_dict(orient="records")` _board_index already reads for
    # the board grid, and it inherits that call's exact limitation: a board
    # with a duplicated player_id keeps only the LAST matching row (dict
    # construction, last key wins), silently. That is not a new risk this
    # field introduces -- _board_index has always had it -- and build_board
    # is expected to never emit two rows for one player_id; it is noted here
    # because a SELECT built off a silently-wrong row is a real pick sent to
    # ESPN, not a display glitch. default_factory so the shared empty dict is
    # not aliased across every session, same reasoning as team_slots.
    board_by_id: dict = dataclasses.field(default_factory=dict)
    # PROVENANCE of `settings`: True only when it came from THIS connect's
    # live ESPN fetch (_league_settings_from_espn), False when build_session
    # fell back to the database's own (league_mod.load) or to the cold-start
    # default. It exists for exactly one consumer -- `settings.pick_order`,
    # which is only ever safe to trust from the live fetch.
    #
    # The database's `league` table holds one row PER SEASON, each with that
    # season's real pick order, and `load()` takes the newest. On the default
    # league's data/nfl.duckdb that is 2025's [7, 5, 2, 8, 4, 1, 6, 3] --
    # last year's shuffle. ESPN team ids are stable across seasons, so
    # `pick_order.index(team_id) + 1` answers confidently and WRONGLY: on
    # this database team 1 resolves to slot 6 (draft_order says 1), team 2 to
    # slot 3 (says 2), team 3 to slot 8 (says 3) -- every one of the eight is
    # wrong. And a wrong slot attributes every pick to the wrong manager with
    # nothing downstream able to detect it (see _slot_for_team's docstring).
    #
    # A flag rather than `sess.settings.season == CURRENT_SEASON`: the season
    # check ROTS. CURRENT_SEASON is a hand-bumped constant, so the year
    # somebody forgets to bump it is the year a stale row passes the check
    # silently -- and a season number is in any case only a proxy for the
    # question actually being asked, which is "did this list come off ESPN
    # just now". This states that fact outright and cannot drift from it.
    settings_from_espn: bool = False


def _attach_espn_proj(conn, board):
    """Left-join ESPN's projected season points onto the board by espn_id.

    espn_adp already carries `espn_proj` (the projection pull that feeds the
    player page's proj_ppg), but build_board keeps only rank and id from that
    table. /api/live/board's trending icon compares this year's projected ppg
    to last year's actual, so the projection rides on here. Best-effort: a
    table written before the projection column existed, or a board with no
    espn_id, simply leaves the column absent and every proj_ppg null. DST rows
    have a null espn_id and so never match -- correct, since ESPN projects no
    per-game line for a defense.
    """
    if "espn_id" not in getattr(board, "columns", []):
        return board
    espn = read_table(conn, "espn_adp")
    if espn.empty or "espn_proj" not in espn.columns:
        return board
    proj = espn[["espn_id", "espn_proj"]].dropna().drop_duplicates("espn_id")
    return board.merge(proj, on="espn_id", how="left")


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
                  league_id: str = "", settings=None,
                  progress=None) -> DraftSession:
    """Everything expensive, once. 4.1-34.8s against a real database.

    `league_id` is passed in rather than looked up, because the database has
    no record of it: the `league` table is `(season, settings_json)` and no
    table anywhere carries a league id. An earlier version read
    `league["league_id"]` and raised KeyError the first time a real connect
    ran. The caller already has it -- parsed from the URL the user pasted,
    which is the only place it exists.

    `progress` (a ConnectProgress, or None for the no-op) is how the connect
    screen learns what this function is doing WHILE it does it. Three of its
    stages live in here because all three of the expensive steps do:
    measured against data/nfl.duckdb, build_board 1.5-1.9s, build_pool
    2.2-3.6s, and fit_all either 6-13ms (a league with no history: cold
    start) or 27.5-30.7s (the owner's own league: 696 picks over six
    seasons, eight per-manager fits). That last number is why fit_all is
    reported per manager rather than as one opaque wait -- it is 79-88% of
    the whole connect, and it is the one stage with a real fraction to show.

    The docstring's old figure ("roughly 17s") was the module docstring's
    make-sim measurement and predated the board and pool getting slower; the
    range above is measured, per stage, and is in the report.
    """
    progress = progress or _NO_PROGRESS
    # `settings` is passed in when the connect flow fetched the league's real
    # roster/scoring from ESPN (see _league_settings_from_espn); None falls
    # back to whatever the database holds (the owner's imported league, or the
    # cold-start default). Everything downstream -- the board, the pool, the
    # round count, the roster the simulator drafts FOR -- keys off it.
    # Recorded BEFORE the fallback overwrites `settings`, because after it
    # the two sources are indistinguishable -- and one field of the result,
    # `pick_order`, is only safe to read when it came from ESPN (see
    # DraftSession.settings_from_espn, and _slot_from_pick_order's own
    # docstring for what reading a stale one does).
    settings_from_espn = settings is not None
    if settings is None:
        # WHICH fallback, because they are not the same thing and the connect
        # screen has to be able to say. `league_mod.load` returns the
        # database's own settings if the `league` table has a row and the
        # built-in cold-start default if it does not -- and for any league
        # provisioned by this app the table is ALWAYS absent, since `league`
        # is a LEAGUE_TABLE and provision_league copies only the universal
        # ones. So "we fell back to what you imported" and "we fell back to a
        # generic 8-team PPR league you have never seen" both looked
        # identical, and the second is the one that is actually reached for
        # every non-default league. One extra read of a table that holds at
        # most one row per season, only on this path.
        progress.fact(settings_source="saved" if not read_table(conn, "league").empty
                      else "default")
        settings = league_mod.load(conn)

    progress.begin("board")
    board = build_board(conn, settings=settings)
    board = _attach_espn_proj(conn, board)
    progress.ok("board", f"{len(board)} players")
    progress.fact(players=int(len(board)))

    progress.begin("pool")
    pool = build_pool(conn, board, settings)
    # The value this step actually discovers: where replacement level sits.
    # That is the whole point of pricing a pool -- every vor number the room
    # shows is measured from these two ranks -- and it is read off the
    # league's own settings, so a 12-team league genuinely reads differently
    # from an 8-team one. `.get` with a dash: a league with no RB or WR
    # starter slot at all is absurd but not impossible, and inventing a
    # baseline for a position nobody starts would be inventing a value.
    ranks = settings.replacement_ranks
    progress.ok("pool", f"RB{ranks.get('RB', '-')} · WR{ranks.get('WR', '-')} "
                        "baseline")

    progress.begin("history")
    # Ticked per manager, because this is the stage the owner actually waits
    # on and it is the only one with a real fraction to report. `seen` is
    # written by fit_all's callback on this same thread (fit_all is
    # synchronous), so no synchronisation is needed for it.
    seen = {"done": 0, "total": 0, "seasons": ()}

    def _on_manager(done, total, seasons):
        seen.update(done=done, total=total, seasons=tuple(seasons))
        progress.value("history", f"{done} of {total} managers"
                       if done else f"{total} managers to fit")

    fits = fit_all(conn, settings, on_manager=_on_manager)
    if seen["total"]:
        progress.ok("history", f"{seen['total']} managers · "
                               f"{len(seen['seasons'])} seasons")
        progress.fact(managers=int(seen["total"]),
                      seasons=len(seen["seasons"]))
    else:
        # cold_start_fits: no imported draft history for this league at all,
        # which is the normal case for a mock and for anyone's first connect.
        # Said outright rather than left as a silent "0 managers": the market
        # prior IS the model in that case, and it is a different tool than
        # the one that has read six of your drafts.
        progress.ok("history", "none · market prior")
        progress.fact(managers=0, seasons=0)
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
        started_at=datetime.now(timezone.utc), board=board,
        board_by_id={str(r["player_id"]): r
                     for r in board.to_dict(orient="records")},
        settings_from_espn=settings_from_espn)


import re
import threading
import time

from fastapi import HTTPException
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed

from pipeline.draft_listener import DraftListener, run_listener
from pipeline.draft_socket import run_socket_listener
from pipeline.espn_league import STATE_PATH, parse_league_id
from pipeline.espn_live import ESPN_PRO_TEAM_BY_ABBREV, _dst_espn_id, apply_picks
from pipeline.espn_teams import fetch_league_settings, fetch_team_slots
from pipeline.espn_teams import http_fetch as _team_view_fetch
from pipeline import leagues as leagues_mod
from pipeline.leagues import DEFAULT_LEAGUE, provision_league
from scoring.config import CURRENT_SEASON
from scoring.draft_sim import (_drafted_state, _seed_rosters, horizon_picks,
                               horizon_target, snake_slots, survival)
from scoring.gain import available_by_vor, rank_available


def _ordinal(n: int) -> str:
    """1 -> '1st'. For the one sentence the connect screen exists to say --
    "you pick 2nd of 8" -- which reads as a seat, not as a field value."""
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# The connect's stage list, in the order the connect ACTUALLY runs them --
# measured, not assumed (see .superpowers/sdd/connect-experience-report.md for
# the per-stage timings against the real database). Two rows are conditional
# (see _connect_plan): `reset` only exists when a listener was actually
# running, and `socket` only on the bookmarklet path, which is the only one
# that owns a socket handle of its own.
#
# The two rows the task brief's own stage list did not have are here because
# they are measured to be the second and third most expensive things a
# connect does: `pool` (build_pool, 2.2-3.6s) and `league` (provisioning a new
# league's file is a 33MB table-by-table copy, 2.9s). Hiding three seconds
# inside another row's spinner is the dead-screen problem this task exists to
# fix, one row further down.
_STAGE_LABELS = (
    ("token", "Reading your draft token"),
    ("reset", "Clearing the previous session"),
    ("settings", "Reading league settings"),
    ("league", "Opening this league's database"),
    ("slot", "Finding your slot"),
    ("board", "Building the player board"),
    ("pool", "Pricing the pool against replacement"),
    ("history", "Loading draft history"),
    ("teams", "Naming the teams"),
    ("socket", "Opening the draft socket"),
    ("ranking", "Ranking the board"),
)


def _connect_plan(token_path: bool, had_listener: bool):
    """The stages THIS connect will actually run, published up front so the
    screen can draw the ones still to come as pending rather than growing a
    list a row at a time.

    Conditional rows are dropped, never rendered as a stage that then never
    runs: `reset` when nothing was listening (the overwhelmingly common case
    -- a first connect has nothing to stop, and _stop_listener returns in
    microseconds), and `socket` on the browser-observer path, which watches a
    socket a real ESPN tab holds and so has no handle of its own to open (see
    state["socket"]).
    """
    skip = set()
    if not had_listener:
        skip.add("reset")
    if not token_path:
        skip.add("socket")
    return [(k, "Reading the draft URL"
             if (k == "token" and not token_path) else lbl)
            for k, lbl in _STAGE_LABELS if k not in skip]


class ConnectProgress:
    """One connect's own progress record: what it has done, what it is doing,
    and the real value each step discovered.

    Exists because the work is genuinely slow and genuinely interesting, and
    until now all of it happened behind a spinner. Measured against the real
    database (data/nfl.duckdb, 249 players, 8 teams, six seasons of history):
    a connect to the owner's own league blocks for 32-35s, of which fit_all is
    27.5-30.7s; a connect to a fresh mock league blocks for 8.6s, of which
    provisioning the league file is 2.9s and build_pool 2.2-3.6s. "Several
    seconds on a dead screen" was an understatement by an order of magnitude.

    EVERY value on it is a real discovered value. There is no timer, no
    minimum display time and no synthetic step anywhere in this class: a stage
    that finishes in 20ms flashes past, and the elapsed figure it publishes is
    a real monotonic delta.

    THREAD SAFETY. Three threads write to one of these -- the connect request
    thread (stages 1-9), the listener thread (`socket`, from run_socket_
    listener's on_socket / the pump's error handler) and the recompute worker
    (`ranking`) -- and a fourth, whichever request thread is serving
    /api/live/connect-progress, reads it. So NOTHING mutates the stage list
    directly: every mutation is handed to `publish` as a callable, and
    register_live_routes runs it under the same `lock` every other write to
    `state` takes, with the same identity guard (a superseded connect's late
    callback must be a no-op, exactly as a superseded listener's is). The
    snapshot published under that lock is rebuilt from scratch each time and
    never mutated afterwards, so the endpoint can serve it after releasing.

    CALLER'S PRECONDITION: never call a method on this while holding `lock` --
    `lock` is a plain threading.Lock, not an RLock, so a progress call from
    inside a `with lock:` block would deadlock the whole app. The call sites
    that live next to a locked write (on_socket, the pump's error handler,
    recompute_worker) all make the call after the block, and say so.
    """

    def __init__(self, plan=(), publish=None):
        self._stages = [{"key": k, "label": lbl, "status": "pending",
                         "value": None, "ms": None} for k, lbl in plan]
        self._by_key = {s["key"]: s for s in self._stages}
        self._publish = publish
        self._started = time.monotonic()
        # Frozen the moment nothing is left running, so the handoff screen's
        # "5.6s" is how long the connect actually took rather than how long
        # the user has been reading the result.
        self._ended = None
        self._stage_started = {}
        self._facts = {}
        self._error = None

    # -- mutation. Each one is a no-op for a key this plan does not carry, so
    # a caller need not know which conditional rows are in play. --

    def _apply(self, mutate):
        if self._publish is None:
            mutate()
        else:
            self._publish(self, mutate)

    def _open(self, key):
        """The stage under `key`, if it is still open to being changed. A
        terminal stage is never reopened: the socket stage can be failed by
        the pump's error handler and completed by on_socket, and whichever
        genuinely happened first is the one that is true."""
        stage = self._by_key.get(key)
        if stage is None or stage["status"] in ("ok", "warn", "failed"):
            return None
        return stage

    def begin(self, key):
        def run():
            stage = self._open(key)
            if stage is None or stage["status"] == "running":
                return
            stage["status"] = "running"
            self._stage_started[key] = time.monotonic()
        self._apply(run)

    def value(self, key, value):
        """A live value on a stage still running -- the manager-fit counter,
        which is the only place a real fraction exists to show (fit_all
        genuinely fits N of M managers). Never a percentage of elapsed time."""
        def run():
            stage = self._open(key)
            if stage is not None:
                stage["value"] = value
        self._apply(run)

    def _finish(self, key, status, value):
        def run():
            stage = self._open(key)
            if stage is None:
                return
            stage["status"] = status
            stage["value"] = value
            started = self._stage_started.get(key)
            stage["ms"] = (None if started is None
                           else round((time.monotonic() - started) * 1000))
            self._settle()
        self._apply(run)

    def _settle(self):
        """Stop the clock the first time nothing is left to do. A terminal
        stage is never reopened (see `_open`), so this cannot un-settle."""
        if self._ended is None and self.phase() != "connecting":
            self._ended = time.monotonic()

    def ok(self, key, value=None):
        self._finish(key, "ok", value)

    def warn(self, key, value):
        """Done, but not the way it was meant to be -- the connect carries on.
        The one that matters is `settings`: _league_settings_from_espn returns
        None on ANY failure and build_session then silently uses the
        database's own roster, which decides the round count and every
        replacement level. Silence there was the bug."""
        self._finish(key, "warn", value)

    def fail(self, key, value, hint=None):
        """Stopped here. `key=None` (or a key already terminal) fails the
        first stage still open, so a caller that only knows "the connect died"
        -- the pump's error handler, which cannot know which stage was live --
        still lands the failure on a real row rather than nowhere."""
        def run():
            if self.phase() == "failed":
                return          # the FIRST failure is the one that matters
            stage = self._open(key) if key else None
            if stage is None:
                stage = next((s for s in self._stages
                              if s["status"] in ("pending", "running")), None)
            if stage is None:
                return
            stage["status"] = "failed"
            stage["value"] = value
            started = self._stage_started.get(stage["key"])
            stage["ms"] = (None if started is None
                           else round((time.monotonic() - started) * 1000))
            self._error = {"stage": stage["key"], "label": stage["label"],
                           "detail": value, "hint": hint}
            self._settle()
        self._apply(run)

    def fact(self, **kw):
        """What the connect learned, for the handoff screen's fact grid. Only
        ever set from a value actually read -- a fact nobody discovered is
        absent, and the screen drops the cell rather than inventing one."""
        def run():
            self._facts.update(kw)
        self._apply(run)

    # -- reading --

    def phase(self) -> str:
        if any(s["status"] == "failed" for s in self._stages):
            return "failed"
        if any(s["status"] in ("pending", "running") for s in self._stages):
            return "connecting"
        return "ready"

    def snapshot(self) -> dict:
        """A fresh, self-contained copy -- built under `lock` by the endpoint
        and never mutated afterwards, so it can be serialised after the lock
        releases without a torn read.

        `elapsed_ms` is computed HERE, per request, rather than stored at the
        last mutation: build_board alone runs for ~1.9s without a single
        stage transition, and an elapsed figure that only moved when a stage
        landed would read as a stopped clock exactly during the waits it
        exists to measure.
        """
        end = self._ended if self._ended is not None else time.monotonic()
        return {
            "phase": self.phase(),
            "stages": [dict(s) for s in self._stages],
            "facts": dict(self._facts),
            "error": dict(self._error) if self._error else None,
            "elapsed_ms": round((end - self._started) * 1000),
        }


# The no-op progress every caller that isn't a connect gets: live_start and
# every test that calls build_session directly. An empty plan means every
# method above returns at its first lookup, so the instrumentation costs those
# callers nothing and needs no `if progress is not None` at any call site.
_NO_PROGRESS = ConnectProgress()


class ConnectBody(BaseModel):
    # No my_slot field. A URL carrying teamId= resolves it immediately (see
    # _team_id_from_url / _slot_from_pick_order / _slot_for_team); one that
    # doesn't (the natural waiting-room URL to paste) leaves it None until
    # the socket's TOKEN frame names our team (see live_connect's on_change)
    # -- there is no third case left for a human to fill in by hand.
    url: str


class TokenBody(BaseModel):
    """What the bookmarklet delivers.

    Deliberately NOT the account session. `token` is ESPN's per-draft
    draftSecurity value -- scoped to this one draft, worthless once it ends --
    and `swid` identifies the account but is not a login credential on its
    own. The espn_s2 session cookie never reaches here: the bookmarklet uses
    it only to fetch this token from ESPN (on ESPN's own origin, where the
    cookie stays) and forwards just the result. So the most this endpoint ever
    holds is a two-hour nonce, in memory, which is the whole point of doing it
    this way rather than taking a password.
    """
    leagueId: str
    teamId: str
    swid: str
    token: str
    season: str



# --- Surviving a restart -----------------------------------------------------
#
# WHAT A RESTART ACTUALLY LOSES, measured against this file rather than
# assumed. `drafted` is already durable: every pick is written to the league's
# own DuckDB file by the listener path (apply_picks in _launch_listener's
# on_change), so the board itself survives. `build_session` rebuilds everything
# else from that same database -- board, pool, betas, crosswalk, slot_managers,
# settings, board_by_id -- and the connect path re-fetches team names and
# league settings from ESPN, and `seed` is the pinned DEFAULT_SEED constant, so
# it comes back identical. What is NOT derivable is the four fields the
# bookmarklet delivered (leagueId, teamId, swid, token) plus the season the
# ESPN fetches need: they exist nowhere in the database, nowhere in any URL
# this process kept, and cannot be re-minted here -- minting needs espn_s2,
# which by design never reaches this process. `my_slot` is recomputable in the
# ordinary case (ESPN's pickOrder, or imported draft history) but NOT in the
# case this tool is most often pointed at: a mock draft, whose managers appear
# in no history and whose slot was learned from the socket's own round-1
# ordering (_slot_from_socket). With a dead token there are no frames to
# relearn it from, so it is saved too.
#
# WHERE, and why not in the league's own DuckDB file, which already holds
# `drafted` for that league:
#
#   1. A RESTART MUST FIND IT WITHOUT ALREADY KNOWING THE LEAGUE. Which league
#      was live is exactly what a restarted process does not know.
#      pipeline/leagues.py provisions a file per league and data/leagues
#      currently holds twenty of them at ~28MB each; opening every one to look
#      for a session row would be slow and would take a single-writer lock on
#      each. This record sits at a path derived from the one database
#      create_app already opened (see session_record_path), so finding it is
#      one stat() and the league id is inside it.
#   2. SECRETS MUST NOT BE COPIED. provision_league seeds each new league's
#      file by copying UNIVERSAL_TABLES out of this same database
#      (pipeline/leagues.py). A `live_session` table there would be one
#      classification mistake away from having the token duplicated into every
#      league file ever provisioned, permanently. A separate file cannot be
#      picked up by that loop at all.
#   3. IT MUST BE DELETABLE AND MODE-RESTRICTED. 0600 and unlink are
#      properties a file has and a row does not: a DELETE leaves the value in
#      the database file's freed pages, and DuckDB has no per-row permissions.
#
# WHAT THIS PUTS ON DISK, AND WHY THAT IS A DELIBERATE CHANGE OF POSTURE.
# TokenBody's docstring just above is the whole design of the bookmarklet: the
# espn_s2 account session never leaves the user's browser, and only a per-draft
# nonce plus public ids reach this process. live_connect_token's own comment on
# state["token"] used to finish that sentence -- "In memory only: a draft token
# is a short-lived nonce, and writing it to disk is the one thing that would
# turn a breach into a leak." This code writes it to disk, so that comment was
# rewritten rather than left to contradict what the code now does. The trade,
# stated here so nobody widens it by accident:
#
#   ON DISK: leagueId, teamId, season, swid, the draftSecurity token, the
#   resolved slot, and a timestamp. NOT espn_s2 -- it has never reached this
#   process and still does not.
#
#   WHAT SOMEBODY WHO READS THE FILE CAN DO: open ESPN's draft socket as this
#   team and send SELECT -- make this user's picks -- for as long as the draft
#   is running. That is a WRITE capability on one draft, not account access:
#   `swid` is the account's public GUID (ESPN puts it in its own URLs) and is
#   not a credential on its own, and the token is scoped to this one draft and
#   refused once it ends. They cannot log in, read the account, or reach any
#   other league.
#
#   WHY IT IS STILL WORTH IT: there is no reconnect without the token and no
#   way to re-mint it here, so "restore the session with the owner touching
#   nothing" and "never write the token" are mutually exclusive.
#
#   HOW THE EXPOSURE IS BOUNDED: mode 0600, owner-only, written whole via
#   os.replace, next to a database that already holds this league's entire
#   draft. Deleted when the draft's last pick lands, when /api/live/stop runs,
#   when a connect replaces it with a session that has no token (the
#   browser-observer path), and when a restore finds it older than
#   SESSION_RECORD_MAX_AGE_SECONDS.
SESSION_RECORD_SUFFIX = ".live-session.json"
SESSION_RECORD_VERSION = 1

# A draft runs a couple of hours and ESPN's draftSecurity token is minted per
# draft (TokenBody's docstring calls it a two-hour nonce). A record older than
# this describes a draft that is over, so restoring it would spend 4.1-34.8s
# rebuilding a board for a token ESPN is going to refuse. Twelve hours rather
# than two: the job of this number is to bound how long a dead token can sit on
# disk after a draft that was never stopped cleanly (browser closed, laptop
# slept), and erring generous costs one visible, actionable failure while
# erring tight loses a live draft that ran long.
SESSION_RECORD_MAX_AGE_SECONDS = 12 * 3600


def session_record_path(db_path: str) -> str:
    """Where this deployment's live-session record lives.

    Derived from the app's own database path rather than being a constant,
    because DRAFT_DB_PATH is how a second instance runs against a copy (see
    pipeline/db.DEFAULT_PATH) -- the whole test suite and every scratch server
    does exactly that. A fixed path would have one of those restore the
    other's draft, or worse, delete its token.
    """
    return str(db_path) + SESSION_RECORD_SUFFIX


def save_session_record(db_path, league_id, team_id, season, swid, token,
                        my_slot=None) -> str:
    """Write (or replace) the live-session record. Returns its path.

    Created with mode 0600 at open() time, not chmod-ed afterwards: a chmod
    leaves a window in which the token is world-readable. Written to a
    temporary file and os.replace-d into place so a concurrent reader can only
    ever see a complete record -- a half-written one parses as garbage, which
    load_session_record correctly treats as "no session", and losing a live
    draft to a torn read would be a real bug rather than a theoretical one.
    """
    path = session_record_path(db_path)
    tmp = f"{path}.tmp"
    body = {
        "version": SESSION_RECORD_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "league_id": str(league_id),
        "team_id": str(team_id),
        "season": str(season or ""),
        "swid": str(swid),
        "token": str(token),
        "my_slot": None if my_slot is None else int(my_slot),
    }
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(body, fh)
    os.replace(tmp, path)
    return path


def load_session_record(db_path, now=None):
    """The saved session, or None if there is not a usable one.

    EVERY failure is None, never an exception: no file at all (the normal case
    -- every start that is not a mid-draft restart), a truncated or hand-edited
    file, a version this build does not know, a record missing a field the
    socket cannot open without, a non-numeric team id, or one too old to still
    carry a live token. A restore is a convenience and the API has to come up
    either way, so nothing in here is allowed to stop it.

    A record found too old is DELETED, not merely ignored. That is the only
    automatic bound on how long the token stays on disk when a draft was never
    stopped cleanly, and an expired token has no value worth keeping.

    `now` is injectable so the age rule is testable without touching a clock;
    production passes nothing.
    """
    path = session_record_path(db_path)
    try:
        with open(path) as fh:
            record = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        # Unreadable, or not JSON. Left in place rather than deleted: this is
        # somebody's token, and a transient read failure must not be the
        # reason it is thrown away.
        return None
    if (not isinstance(record, dict)
            or record.get("version") != SESSION_RECORD_VERSION):
        return None
    if not all(record.get(k) for k in ("league_id", "team_id", "swid", "token")):
        return None
    try:
        int(record["team_id"])
    except (TypeError, ValueError):
        # The socket path has no browser JOIN to learn the team from and
        # _slot_for_team compares integers -- the same up-front rejection
        # live_connect_token makes on the way in.
        return None
    try:
        saved_at = datetime.fromisoformat(record.get("saved_at") or "")
    except ValueError:
        saved_at = None
    now = now or datetime.now(timezone.utc)
    if saved_at is None or saved_at.tzinfo is None:
        # No usable timestamp means the age rule can never fire for this
        # record, i.e. the token would sit on disk forever. Treated as
        # expired, which is also what a hand-mangled record deserves.
        clear_session_record(db_path)
        return None
    if (now - saved_at).total_seconds() > SESSION_RECORD_MAX_AGE_SECONDS:
        clear_session_record(db_path)
        return None
    return record


def clear_session_record(db_path) -> None:
    """Forget the saved session.

    Missing is success: every caller is a path that cannot know whether a
    record exists (a stop with no session, a browser connect that never had a
    token, the last pick of a draft nobody connected with a token). Any OTHER
    OSError propagates deliberately -- a token this failed to delete is still
    readable on disk, and that is worth a 500 on /api/live/stop or a recorded
    listener_error rather than a silent success.
    """
    try:
        os.unlink(session_record_path(db_path))
    except FileNotFoundError:
        pass


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


def _season_from_url(url: str) -> int:
    """Season id from a pasted draft URL, or the current season as a default.

    A live draft URL carries `seasonId=2026`; a bare id or a waiting-room URL
    may not, so CURRENT_SEASON stands in. The season is only ever used to
    fetch ESPN's team-name view for a draft happening now, and that fetch is
    best-effort -- a wrong season merely falls the board's columns back to
    "Team {slot}" placeholders (fetch_team_slots returns {}), never breaks the
    connect. TokenBody carries `season` outright, so only the URL path needs
    this.
    """
    m = re.search(r"seasonId=(\d+)", url or "")
    return int(m.group(1)) if m else CURRENT_SEASON


def _attach_team_slots(session, league_id, season):
    """Fetch ESPN's real team names for this league and pin them to the
    session, so /api/live/board can name every column without re-fetching.

    Lives on the connect path, not in build_session: build_session is also
    called by live_start with no league context, and it should never do a
    network fetch. fetch_team_slots is best-effort (returns {} on any failure,
    including no network), so this never raises into a connect. DraftSession is
    frozen, so a non-empty result is attached with dataclasses.replace rather
    than mutation; an empty one leaves the session untouched (its team_slots
    default stays {}, and the board endpoint falls back to placeholders).
    """
    slots = fetch_team_slots(_team_view_fetch(), league_id, season)
    return dataclasses.replace(session, team_slots=slots) if slots else session


def _league_settings_from_espn(league_id, season):
    """The league's real roster/scoring as a LeagueSettings, or None.

    Fetches ESPN's mSettings (best-effort) and runs it through
    league.from_espn, so the session drafts for the league's ACTUAL roster --
    how many of each starter, flex and bench, and therefore how many rounds --
    rather than the cold-start default. None on any failure, so the caller
    falls back to the database's own settings and a connect never fails over a
    settings fetch.

    One backfill: if ESPN's scoring items don't map to anything the model
    scores (from_espn leaves `scoring` empty), the roster is still ESPN's but
    the scoring is filled from the default rules -- an empty scoring dict would
    otherwise zero out every projection downstream. The roster is the part this
    task is about; it is taken from ESPN verbatim.
    """
    raw = fetch_league_settings(_team_view_fetch(), league_id, season)
    if not raw:
        return None
    try:
        settings = league_mod.from_espn(raw)
    except Exception:      # noqa: BLE001 -- best-effort; fall back to the db's
        return None
    if not settings.scoring:
        settings = dataclasses.replace(
            settings, scoring=dict(league_mod.default_settings().scoring))
    return settings


def _slot_from_pick_order(settings, team_id: int | None) -> int | None:
    """My draft slot straight from ESPN's own pick order -- tried before
    both `_slot_for_team` and `_slot_from_socket` (see their own docstrings
    for why each of those can come back empty on a mock).

    `settings.draftSettings.pickOrder` (scoring.league.LeagueSettings.
    pick_order, carried in from `_league_settings_from_espn`) is a list of
    team ids in draft-slot order -- pickOrder[0] is slot 1's team id, and so
    on (see pipeline/espn_teams.py's module docstring, which reads the same
    field for the board's column names). Unlike `_slot_for_team` this needs
    no imported draft history -- a mock draft's strangers are simply team
    ids in this list, not managers in a table that has never heard of them
    -- and unlike `_slot_from_socket` it needs no round of the draft to have
    actually happened: it is exact and available the moment ESPN's league
    settings are fetched, at connect, before the socket has sent a single
    frame.

    CALLER'S PRECONDITION, and it is not optional: `settings` must be the
    LeagueSettings from THIS connect's live ESPN fetch. The database's own
    settings carry a pick order too -- last completed season's, since the
    `league` table is one row per season and `load()` takes the newest -- and
    because ESPN team ids are stable across seasons this function will
    happily index into it and return a confident, wrong slot rather than
    None. Both call sites hold to this: _provision_and_build passes the
    fetched `settings` local (None when the fetch failed, which falls through
    here), and _resolve_slot gates on `sess.settings_from_espn`, the flag
    build_session sets for exactly this reason. Any third caller must do the
    same. The check cannot live in here -- a bare LeagueSettings does not
    know where it came from -- which is why the flag is on the session.

    None whenever it cannot answer, never a guess: no settings (the ESPN
    fetch failed, or this session was built with none), no team_id yet, or
    a team_id ESPN's own pickOrder does not carry -- `().index(...)` would
    raise ValueError, caught here so the caller falls through to the next
    resolver instead of the whole connect blowing up. (Verified live against
    a real mock draft that this league's pickOrder is in fact populated --
    see the post-merge-fixes report -- but a league that does not publish
    one at all is exactly this fallthrough case, and both the older
    resolvers below still cover it.)
    """
    if settings is None or team_id is None:
        return None
    pick_order = getattr(settings, "pick_order", None) or ()
    try:
        return pick_order.index(team_id) + 1
    except ValueError:
        return None


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


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# NaN -> None serialization, the same rule api/main.py applies to board rows:
# a missing/NA cell must reach JSON as null, not as NaN (which FastAPI's
# encoder rejects) and not as a pandas/numpy scalar. Kept local to the board
# endpoint rather than imported from main to avoid a cross-module dependency
# between the two route files.
def _int_or_none(value):
    return None if value is None or pd.isna(value) else int(value)


def _float_or_none(value):
    return None if value is None or pd.isna(value) else float(value)


def _str_or_none(value):
    return None if value is None or pd.isna(value) else str(value)


def _board_index(board) -> dict:
    """player_id -> board row (a Series), for /api/live/board's cell join.

    A None board (a session built without one, e.g. a test fixture) yields an
    empty index, so every pick falls back to its raw id rather than the grid
    silently dropping picks. Built once per request, not once per cell.
    """
    if board is None or getattr(board, "empty", True):
        return {}
    return {str(row["player_id"]): row for _, row in board.iterrows()}


def _board_cell(player_id, pick_no, teams: int, slots: list, by_id: dict) -> dict:
    """One drafted pick as a board-grid cell: its snake round/slot plus the
    rich player payload the front end draws in the column.

    round/slot come from the snake order, not from any teamId ESPN stores --
    `drafted` carries none (it is (player_id, pick_no)); the pick's overall
    number alone fixes both. `value` = overall - market_rank is the steal/reach
    signal: a player who fell past his ADP (drafted later than his consensus
    rank) scores positive, a reach negative, and null when there is no market
    rank to compare against.

    A player id the board does not carry -- a crosswalk miss, or a manually
    marked id -- still produces a cell, named by its raw id with everything
    else null, so the grid never silently drops a pick.
    """
    overall = int(pick_no)
    idx = overall - 1
    slot = slots[idx] if 0 <= idx < len(slots) else None
    rnd = (idx // teams) + 1 if teams else None
    row = by_id.get(str(player_id))
    if row is None:
        player = {"player_id": str(player_id), "name": str(player_id),
                  "position": None, "team": None, "bye": None,
                  "overall_rank": None, "tier": None, "market_rank": None,
                  "espn_ppr_rank": None, "vor": None, "last_ppg": None,
                  "last_points": None, "proj_ppg": None, "value": None}
    else:
        market_rank = _float_or_none(row.get("market_rank"))
        stats = row.get("stats")
        stats = stats if isinstance(stats, dict) else {}
        espn_proj = _float_or_none(row.get("espn_proj"))
        # Converted into the league's scoring the same way the board's own
        # proj_points is, using the board's `proj_scale` column
        # (scoring/board.projection_scale). ESPN publishes this number in full
        # PPR only. The trending icon compares it against `last_ppg`, which is
        # `stats.ppg` -- now the league's points -- so leaving the projection
        # in PPR would have made every high-reception player in a half-PPR
        # room look like a breakout: two different currencies, one arrow.
        proj_scale = _float_or_none(row.get("proj_scale"))
        if espn_proj is not None and proj_scale is not None:
            espn_proj = espn_proj * proj_scale
        player = {
            "player_id": str(row["player_id"]),
            "name": _str_or_none(row.get("name")),
            "position": _str_or_none(row.get("position")),
            "team": _str_or_none(row.get("team")),
            "bye": _int_or_none(row.get("bye")),
            "overall_rank": _int_or_none(row.get("rank")),
            "tier": _int_or_none(row.get("tier")),
            "market_rank": market_rank,
            # ESPN's own PPR rank, for the hype/lame icon: ESPN vs the market
            # consensus (market_rank). A big gap either way is the signal.
            "espn_ppr_rank": _int_or_none(row.get("espn_ppr_rank")),
            "vor": _float_or_none(row.get("vor")),
            "last_ppg": _float_or_none(stats.get("ppg")),
            "last_points": _float_or_none(stats.get("points")),
            # This year's ESPN projected points per game (17-game season, the
            # same denominator last_ppg uses), for the trending icon. Null for
            # anyone ESPN doesn't project (DST, deep rookies).
            "proj_ppg": round(espn_proj / 17, 1) if espn_proj else None,
            # >0 = fell past ADP (a steal), <0 = reach; null with no ADP.
            "value": None if market_rank is None else float(overall) - market_rank,
        }
    return {"overall": overall, "round": rnd, "slot": slot, "player": player}


def _league_settings_payload(settings) -> dict:
    """The session's real league shape, as /api/live/state's `settings` key
    serves it -- the rail's RosterPanel/ClockPanel read teams/rounds/starter
    counts off this instead of the two hardcoded 8-team/15-round constants
    they used to carry (LEAGUE_TEAMS, duplicated once in ClockPanel.tsx and
    once in DraftRoom.tsx, each commented as cross-referencing the other).

    `settings=None` (the inactive response, no session at all) returns the
    same five keys with null/empty values rather than omitting them, so the
    client never has to branch on whether the key exists -- only on whether
    its values are null, the same convention `listener_error`/
    `listener_alive` already use below.
    """
    if settings is None:
        return {"teams": None, "rounds": None, "starters": {},
                "flex_slots": None, "bench": None, "scoring_format": None}
    return {
        "teams": settings.teams,
        "rounds": settings.rounds,
        "starters": dict(settings.starters),
        "flex_slots": settings.flex_slots,
        "bench": settings.bench,
        # scoring.league.scoring_format reads settings.scoring's own
        # receptions value into one of ppr/half/std -- the same three-way
        # call scoring.market's consensus already makes (see its own
        # docstring). Not a new rule, just the first place this session's
        # format reaches the UI, for the topbar's league-identity line.
        "scoring_format": league_mod.scoring_format(settings),
    }


def _my_roster(session, taken_order) -> list:
    """This session's own roster so far, in pick order -- the payload
    /api/live/state's `my_roster` key serves the rail's RosterPanel.

    Replays `taken_order` with `_seed_rosters`, the exact same replay
    `_recompute` already pays for on every poll (see its own docstring) to
    seed `need_weight`'s roster counts -- nothing new is computed here, only
    read back. `rosters[my_slot]["indices"]` holds pool indices in the order
    they were drafted (see `_seed_rosters`'s own docstring: a `None` entry in
    `taken_order` -- a pick whose player fell out of the pool -- still
    consumes a turn but is never added to any roster's `indices`, so it
    needs no handling here).

    Each index is mapped back to a board row the same way `_board_cell`
    does: `session.pool.player_id[idx]` to a player id, then
    `session.board_by_id` to the row. A miss (`row is None`) should not
    happen -- `pool` and `board_by_id` are both built from the same `board`
    DataFrame in `build_session`, so every pool index's player_id is
    expected to resolve -- but it is skipped rather than raised, so a
    genuine discrepancy costs one roster row rather than the whole
    /api/live/state response.

    Empty, not an error, when `session.my_slot` is still None: the socket
    has not yet named our team (see `DraftSession.my_slot`'s own docstring),
    so there is genuinely no "my roster" to report -- an empty list is the
    honest answer, not a guess at which slot is ours.
    """
    if session.my_slot is None:
        return []
    rosters, _ = _seed_rosters(session.pool, session.settings, taken_order)
    roster = []
    for idx in rosters[session.my_slot]["indices"]:
        pid = str(session.pool.player_id[idx])
        row = session.board_by_id.get(pid)
        if row is None:
            continue
        roster.append({
            "player_id": pid,
            "name": _str_or_none(row.get("name")),
            "position": _str_or_none(row.get("position")),
            "proj_points": _float_or_none(row.get("proj_points")),
        })
    return roster


def _slot_from_socket(listener, teams: int):
    """Derive my draft slot from the socket alone, when history cannot.

    `_slot_for_team` translates a team id through imported draft history; a
    mock draft (or any league not yet imported) has none, so it returns None
    and every recommendation stays blank because the simulator has no slot to
    reason from. But the socket carries enough to name the slot without any
    history: in a snake draft the first round's overall pick order IS the slot
    order, so the team picking k-th of the first `teams` picks drafts from
    slot k. My team's own first-round pick position -- or, before it has
    picked, its position on the clock during round 1 -- names its slot exactly.

    Returns None until my team is seen in round 1 (picked or on the clock),
    which is honest: with no history and no round-1 appearance yet the slot
    genuinely is not known, and a guess would attribute picks to the wrong
    manager. Replay-safe: it dedups repeated SELECTED frames by player the
    same way picks_from_events does (a repeat is ESPN replaying the draft on a
    reconnect JOIN, not a second pick), since the raw event list it reads
    still contains those replays even though the drafted table does not.
    """
    mine = listener.my_team_id
    if mine is None or not teams:
        return None
    seen_players, pick_teams = set(), []   # pick_teams[i] = team id of pick i+1
    for event in listener.events:
        if event is None or event.verb != "SELECTED":
            continue
        pid = _safe_int(event.args[1] if len(event.args) >= 2 else None)
        if pid is not None and pid in seen_players:
            continue                       # replayed pick -- already counted
        if pid is not None:
            seen_players.add(pid)
        pick_teams.append(_safe_int(event.args[0]) if event.args else None)
    # Round 1: overall pick k (1-based) is slot k.
    for i, team in enumerate(pick_teams[:teams], start=1):
        if team == mine:
            return i
    # Not yet picked. On the clock during round 1 -> the next pick's slot.
    if len(pick_teams) < teams and listener.on_the_clock == mine:
        return len(pick_teams) + 1
    return None


# Picks arrive pushed, not polled, so any real gap means the socket is wedged
# rather than merely quiet. The heartbeat (on_activity) now fires on every
# frame -- CLOCK ticks about once a second while any clock runs -- so a live
# socket stamps well inside this window; only a genuine wedge, or the couple
# of seconds a reconnect's backoff+handshake takes, approaches it. Ten
# seconds spans a reconnect without crying wolf, yet still surfaces a truly
# dead listener while a 30-second pick clock leaves time to react.
STALE_AFTER_SECONDS = 10
# Measured on the live board, roughly 0.19s/rollout: 25 -> 4.7s, 100 -> 19.5s,
# 200 -> 35.2s. Those were far too slow to stay current: the on-the-clock
# budget (200 -> ~35s) meant a fast mock, where auto-picks land every few
# seconds, blew several picks past our own turn before the recommendation for
# it finished -- the sidebar showed "no recommendation" exactly when it
# mattered. Budgets are cut so a recompute lands in ~2-8s: FAR (just watching
# opponents pick) is smallest because it runs on every single pick and a
# slightly coarse estimate there is harmless; NOW is largest because it is our
# actual decision, but still inside a real 30-90s clock with room to spare.
# The seed is pinned, so these coarser passes still converge on the same
# scenario set -- fewer rollouts is a noisier estimate, not a different one,
# and a current top-3 beats a precise answer for a pick already gone.
ROLLOUTS_FAR, ROLLOUTS_NEAR, ROLLOUTS_NOW = 12, 25, 40

# How long _stop_listener waits for the previous listener thread to notice
# stop_event and exit (browser close included) before refusing a reconnect
# rather than risking two sockets for the same team. A module constant, not
# a literal default, so a test can shrink it and exercise the refusal path
# without a real ten-second wait.
LISTENER_STOP_TIMEOUT = 10.0

# survival() runs ONE set of rollouts that stop at the turn being measured,
# not one full draft per candidate, so the old clock-rationed budget
# (12/25/40, see rollouts_for below) is no longer the constraint it was
# priced against. This is the whole recompute cost now, and it buys a
# materially tighter survival estimate for a fraction of what search_pick
# cost: measured against the real production pool (data/nfl.duckdb, 249
# players, 8 teams), survival(n_rollouts=400) plus rank_available together
# ran in 0.1-0.8s across picks_made 0/8/50/100 -- an order of magnitude
# under even the cheapest old ROLLOUTS_FAR budget's ~2.3s (12 rollouts *
# ~0.19s), let alone the 30-90s pick clock this has to fit inside.
#
# Re-measured on the same pool once the horizon landed (see
# draft_sim.horizon_picks): 0.97-1.49s across the same four pick counts,
# worst case at pick 1. It costs more because it simulates more -- each
# rollout now runs to a turn a full round of opponent picks away instead of
# stopping at whatever turn came next, which at pick 1 of an 8-team draft is
# 13 simulated picks instead of 1. Still an order of magnitude under the
# pick clock, and it is the only reason the ranking has any signal in it at
# a short gap, so the trade is not close.
SURVIVAL_ROLLOUTS = 400


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


# "How many picks has this draft made", as SQL. Deliberately max(pick_no) and
# NOT count(*), which is what every one of these call sites used to run.
# `drafted` holds no row for a pick whose player the crosswalk could not
# resolve -- those are reported in `unmapped_picks` and deliberately never
# written (see pipeline/espn_live.picks_from_events) -- nor for one that
# DELETE /api/drafted took back out. Counting rows therefore under-reports the
# draft by exactly the number of those holes, and everything derived from the
# number then names a turn that has already gone by: the slot on the clock,
# how many picks until mine (picks_until_turn just above), the pick a ranking
# was measured against. On a mid-draft JOIN, where the whole draft replays at
# once, a single unmapped pick in round 1 was enough to leave the room a full
# turn behind for the rest of the draft.
#
# Sound because pick numbers are 1-based and dense by contract: every writer
# numbers from 1 (picks_from_events counts SELECTED frames, POST /api/drafted
# takes max+1, espn_live.translate copies ESPN's own overallPickNumber), and
# scoring.draft_sim._drafted_state refuses a row below 1 outright. So the
# highest number written IS the count of picks made, holes and all. It agrees
# with `len(taken_order)` from that same function, which is what matters:
# the two are compared against each other (candidates_as_of_pick vs
# picks_made) and a mismatch would leave the room's "recomputing" banner on
# forever.
PICKS_MADE_SQL = "SELECT coalesce(max(pick_no), 0) FROM drafted"


def _is_stale(last_poll_at, now) -> bool:
    """Never having polled counts as stale: the UI must not present an empty
    board as a current one."""
    if last_poll_at is None:
        return True
    return (now - last_poll_at).total_seconds() > STALE_AFTER_SECONDS


class SelectBody(BaseModel):
    player_id: str


# How long to wait for ESPN to echo the pick back on SELECTED. Long enough to
# cover a round trip on a busy draft server, short enough that a wedged
# request does not eat the pick clock it exists to protect.
SELECT_TIMEOUT_SECONDS = 8.0
SELECT_POLL_SECONDS = 0.1


class AutodraftBody(BaseModel):
    """The state to put ESPN's autodraft into -- not "flip it".

    A target state rather than a toggle on purpose: the client is polling
    the current value every 2.5s, so a "flip" sent against a value that
    changed in between (ESPN flipping us ON at the same moment the user
    clicked to turn it off) would act on a state that no longer exists. A
    target is idempotent under exactly that race.
    """
    on: bool


# Same 8s bound and 0.1s cadence as the SELECT round trip above, for the
# same two reasons: long enough for a round trip to a busy draft server,
# short enough that a wedged request cannot eat the pick clock. Named
# separately rather than reusing SELECT's so the two can diverge -- the
# only measurement there is of ESPN's AUTODRAFT echo is fast (the capture's
# outbound `AUTODRAFT false` at line 1186 is answered by `AUTODRAFT 2
# false` at line 1188, two frames later), but one capture is not a bound.
AUTODRAFT_TIMEOUT_SECONDS = 8.0
AUTODRAFT_POLL_SECONDS = 0.1


def _espn_id_for(board_row) -> int | None:
    """The ESPN player id to send in a SELECT.

    Board rows carry `espn_id` for everyone ESPN ranks as a player. D/ST rows
    carry None, because ESPN models a defense as a negative synthetic id
    derived from the pro team -- the same rule build_crosswalk already reads
    picks back through (see its docstring), applied here in the other
    direction.
    """
    espn_id = board_row.get("espn_id")
    if espn_id is not None and not pd.isna(espn_id):
        return int(espn_id)
    if board_row.get("position") == "DST":
        pro = ESPN_PRO_TEAM_BY_ABBREV.get(str(board_row.get("team", "")).upper())
        if pro is not None:
            return _dst_espn_id(pro)
    return None


def register_live_routes(app, conn, db_path):
    """Mount live-draft endpoints.

    In-process state, with ONE exception. Everything in `state` below lives
    and dies with `create_app`'s connection, and that used to be the whole
    story -- "a restart mid-draft means starting again, which is correct:
    the cached pool would be stale anyway", as this docstring said until the
    restart resilience work. Half of that was right and half was not. The
    cached pool genuinely is disposable: it is rebuilt from the database in
    4.1-34.8s and comes back identical. But the session's IDENTITY is not
    rebuildable at all -- the league, team, swid and draftSecurity token
    came from the bookmarklet and exist nowhere else -- so losing it meant
    going back to the ESPN tab and clicking the bookmark again, mid-draft,
    on a thirty-second clock, for every crash and every restart.
    So those few fields are written to disk (see save_session_record, and
    the security note above it for exactly what that costs), and
    _restore_saved_session at the bottom of this function rebuilds the rest
    on a background thread at startup.

    `db_path` is the app's own database file -- the same one `conn` was
    opened against in `create_app`. It is passed (rather than derived from
    `conn`) so `live_connect` can hand it to `provision_league` as the
    `universal_path` a new league's file is seeded from; `conn` itself stays
    the connection the default league's session builds against, exactly as
    before Task 5.
    """
    state = {"session": None, "last_poll_at": None, "unmapped": [],
             "candidates": [], "as_of_pick": None, "computing_for": None,
             # The pick `candidates` was ranked against (see _recompute).
             # Written and cleared with `candidates` everywhere, never on
             # its own: a horizon left over from a previous session would
             # caption the new one's list with the old one's pick number.
             "horizon_pick": None,
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
             "listener_stop": None, "listener_error": None,
             # The recompute worker's own last failure, same job
             # `listener_error` does for the listener thread and separate
             # from it because they fail independently: the listener can be
             # perfectly healthy (frames arriving, picks landing, the board
             # updating) while ranking has stopped dead. `listener_alive`
             # tracks the listener thread and says nothing about this one,
             # so without this key a dead worker is invisible -- candidates
             # and as_of_pick simply freeze at the pick they last reached
             # and the room keeps presenting them. Set and cleared only in
             # recompute_worker, under `lock` and identity-guarded like
             # every other write, so a superseded listener's worker cannot
             # clobber its replacement's status.
             "recompute_error": None,
             # The live socket's send path, published by run_socket_listener
             # via its on_socket callback (see pipeline.draft_socket.
             # SocketHandle) exactly once, right after the first successful
             # connect. None whenever no socket session is running -- either
             # never started, still connecting, or torn down by
             # _stop_listener -- so /api/live/select can refuse a SELECT
             # rather than pretend one has somewhere to go. Only ever
             # non-None for the connect-token (bookmarklet) path;
             # live_connect's browser observer has no socket of its own to
             # publish, so a session started that way always finds this None
             # and /api/live/select correctly refuses with 503.
             "socket": None,
             # The per-session recompute worker (see _launch_listener):
             # survival/rank_available still take real time (a fraction of a
             # second, see SURVIVAL_ROLLOUTS), so it runs here, off the
             # frame-reading thread, or a fast draft's frames would pile up
             # unread behind it. Tracked so _stop_listener joins it before
             # closing league_conn -- the worker holds a cursor on that
             # connection mid-search, so closing it out from under the worker
             # would be a use-after-close, the same hazard listener_thread
             # guards against.
             "recompute_thread": None,
             # The session's own connection when it was built for a
             # non-default league (None for the default league, which uses
             # `conn` and never touches this). Set only by live_connect,
             # closed and cleared only by _stop_listener, and only once it
             # has confirmed the listener thread actually exited -- never on
             # a join timeout, when the thread might still be using it (see
             # _stop_listener's and live_stop's docstrings). DuckDB is
             # single-writer per file, so a league's connection left open
             # after its session truly ends would make every future
             # reconnect to that same league fail; closing one a thread is
             # still using would be worse.
             "league_conn": None,
             # The live ConnectProgress for the most recent connect (see the
             # class, and GET /api/live/connect-progress). Deliberately the
             # object, not a rendered snapshot: it is mutated only under
             # `lock` and snapshotted only under `lock`, so the endpoint
             # cannot serve a half-written stage, and the elapsed figure is
             # computed at request time instead of freezing between stages.
             # Survives the connect that built it -- the screen is still
             # reading it while the socket opens and the first ranking lands,
             # both of which happen after the connect handler has returned.
             "connect": None,
             # Bumped for every connect attempt, valid or not. The identity
             # guard for progress writes, exactly as `listener` is for state
             # writes: a superseded connect's late callback (its listener
             # thread dying, its recompute worker finishing) must not write
             # over the record of the connect that replaced it. A counter and
             # not the object itself because the object is what it guards.
             "connect_seq": 0,
             # The startup restore's own thread (see _restore_saved_session),
             # or None when there was no saved session to restore. Kept for
             # exactly two reasons: /api/live/state reports `restoring` off
             # its is_alive(), which is the only thing that tells the room
             # "your draft is coming back" apart from "there is no draft",
             # and a test can join it instead of sleeping. Never joined by
             # the app itself -- it is a daemon and the API must come up
             # without waiting for it, which is the whole point of it being
             # a thread.
             "restore_thread": None,
             # Why the restore could not rebuild the session, if it failed
             # before it ever reached a listener (a board build that raised,
             # a league file that would not open). Separate from
             # listener_error, which needs a listener to exist to be set --
             # a restore that dies in build_session has none, and without
             # this key that failure is indistinguishable from "no draft was
             # running", which is the exact silence this whole change is
             # about. The socket refusing an expired token is NOT this: that
             # happens after the listener is registered, and is reported
             # through listener_error, exactly as it always was.
             "restore_error": None}
    lock = threading.Lock()

    def _new_progress(token_path: bool, **facts):
        """Open a progress record for a connect that is about to run.

        Publishes the whole stage plan up front -- every row pending -- so
        the screen draws the shape of the work on its first poll rather than
        growing a list one row at a time. `facts` are what the caller
        already knows before any work happens (the league and team ids off
        the token); everything else is added as it is discovered.

        Returns `(progress, seq)`. The sequence number is returned rather
        than only captured in the closure because the startup restore runs
        the same work on a thread nobody is waiting on, and has to be able
        to check -- at the instant it registers its listener -- whether a
        real connect has superseded it in the meantime (see
        _launch_listener's `guard_seq`). The two HTTP callers ignore it:
        they hold the request thread, so their own supersession is already
        handled by _connect_work's _stop_listener.
        """
        with lock:
            state["connect_seq"] += 1
            seq = state["connect_seq"]
            # Read under the same lock as the sequence bump: whether there is
            # a listener to stop decides whether the plan carries a `reset`
            # row, and a plan that disagrees with what the connect then does
            # would leave a row spinning forever or land a value nowhere.
            had_listener = state["listener"] is not None

        def publish(prog, mutate):
            with lock:
                if state["connect_seq"] != seq:
                    return          # superseded -- see state["connect_seq"]
                mutate()
                state["connect"] = prog
        progress = ConnectProgress(_connect_plan(token_path, had_listener),
                                   publish)
        progress.fact(**facts)      # the first publish, which registers it
        return progress, seq

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

        Also closes the previous session's per-league connection, if it had
        one, once the thread is confirmed stopped -- never on the timeout
        path, since a thread that has not joined might still be using it.
        This is what makes a reconnect to the same league safe: DuckDB is
        single-writer per file, so opening that league's connection again
        (see live_connect) would deadlock or error against one this function
        left open.
        """
        with lock:
            stop_event = state["listener_stop"]
            thread = state["listener_thread"]
            recompute_thread = state["recompute_thread"]
        if stop_event is not None:
            stop_event.set()
        # Both the frame-reading thread AND the recompute worker share this
        # stop_event and must be confirmed exited before league_conn closes:
        # each may hold a cursor on it (the worker for the whole of one
        # survival/rank_available ranking), so closing it under either is a
        # use-after-close. A timeout on either means "a thread may still be
        # using the connection" -- refuse the reconnect and leave the
        # connection open rather than corrupt it, exactly as for the listener
        # alone before the worker existed.
        for t in (thread, recompute_thread):
            if t is not None and t.is_alive():
                t.join(timeout=timeout)
                if t.is_alive():
                    return False
        with lock:
            state["listener"] = None
            state["listener_thread"] = None
            state["listener_stop"] = None
            state["recompute_thread"] = None
            state["socket"] = None
            old_league_conn = state["league_conn"]
            state["league_conn"] = None
        if old_league_conn is not None:
            old_league_conn.close()
        return True

    def _recompute(session, picks_made):
        """Run one ranking and store it, unless superseded meanwhile.

        A result computed against a board that has since changed is worse
        than no result -- it recommends a player who may already be gone. So
        the pick count and the session generation are both captured before
        the ranking and re-checked after: if either moved, this result is
        discarded rather than served.

        `session.my_slot` can still be None here -- the socket hasn't named
        our team yet -- and `survival` needs a real slot to index into
        (rosters, snake order, ...), not something to guess at. Skip the
        ranking rather than pass it a fabricated one; candidates stay empty
        until my_slot resolves, which /api/live/state already reports
        honestly via session.my_slot being null.
        """
        if session.my_slot is None:
            return
        with lock:
            generation = state["generation"]
            # The connection this session's own data lives on: the shared
            # `conn` for the default league, or the league's own connection
            # live_connect opened and recorded in state. Read together with
            # `generation` under the same lock so the two describe the same
            # session -- a torn read (this session's generation, some other
            # session's connection) would search against the wrong league's
            # `drafted` table.
            active_conn = state["league_conn"] or conn
        cur = active_conn.cursor()
        try:
            taken, taken_order = _drafted_state(cur, session.pool)
            # My own roster so far, so need_weight can see which slots are
            # still open. _seed_rosters replays every pick to the slot that
            # was on the clock for it, which is the same attribution the
            # simulator resumes from.
            rosters, _ = _seed_rosters(session.pool, session.settings, taken_order)
            counts = rosters[session.my_slot]["counts"]
            # Is the pick on the clock RIGHT NOW our own? survival() cannot
            # work this out from the pick count alone -- `_next_pick_for`
            # scans inclusively, so "my pick is now" and "my pick is next"
            # look identical to it and it answers "now", which pins every
            # available player's survival at 1.0 and makes gain_now
            # identically zero for the leader at every position (see
            # survival's own docstring). This is the only caller that is
            # ever asked WHILE the user is on the clock, and it is the one
            # whose answer is read at exactly that moment, so it is the one
            # that has to say which turn it means.
            #
            # Derived from `len(taken_order)`, not the `picks_made`
            # argument: `taken_order` is what survival() itself counts
            # `already` from, and picks_made was read on the listener
            # thread before this ranking was queued, so a pick landing in
            # between would leave the two disagreeing by one -- exactly the
            # off-by-one this is here to close.
            snake = snake_slots(session.settings.teams, session.settings.rounds)
            on_the_clock = (len(taken_order) < len(snake)
                            and snake[len(taken_order)] == session.my_slot)
            # WHICH pick this ranking is measured against. Not my
            # immediately-next turn: at a 1-3 opponent-pick gap that step is
            # ~zero for everybody and the ranking has no signal left (a
            # defense 5th and a kicker 6th at pick 1 of the owner's mock --
            # see horizon_picks for the measurements). Not an arbitrarily
            # distant one either: past about a round and a half the survival
            # column the room prints reads 0% for every row it shows, which
            # is the same loss of signal from the other side (see
            # horizon_ceiling for that sweep). `horizon_target` is the whole
            # rule -- floor, ceiling and the anchor they are measured from
            # -- and it is the SAME call survival() makes below, so the
            # number served and the number measured cannot differ.
            #
            # The result is a pick that exists, and usually one of mine; at
            # the wheel, where my own turns offer only 2 opponent picks or
            # 14, it is the pick a round and a half out instead, which is
            # somebody else's turn and is still exactly the pick survival
            # was counted to. At my last pick of the draft it is the
            # off-the-end sentinel and `horizon_is_end_of_draft` below is
            # what the room renders instead.
            #
            # Computed from the same `len(taken_order)`/`on_the_clock` pair
            # survival() is called with, in the same critical section, so
            # the number served can never describe a different pick than
            # the one the ranking actually used.
            h = horizon_picks(session.settings)
            horizon = horizon_target(session.settings, session.my_slot,
                                     len(taken_order), on_the_clock, h)
            # How many picks I have left, INCLUDING the one on the clock if
            # it is mine -- `snake[len(taken_order):]` starts at the pick
            # about to be made, which is exactly that reading. It is what
            # lets need_kind tell an open kicker slot in round 3 (thirteen
            # picks left, fill it whenever) from the same slot in round 14
            # (two picks left, two empty slots, fill it now).
            my_turns_left = sum(1 for s in snake[len(taken_order):]
                                if s == session.my_slot)
            # survival()'s avail_pct is already a 0-1 probability (see its
            # docstring and the "counts / max(n_rollouts, 1)" line it
            # returns) -- rank_available wants exactly that, no rescaling.
            avail = survival(
                session.pool, session.settings, session.slot_managers,
                session.my_slot, taken, session.betas,
                n_rollouts=SURVIVAL_ROLLOUTS, seed=session.seed,
                taken_order=taken_order, on_the_clock=on_the_clock,
                horizon=h)["avail_pct"].to_numpy()
            frame = rank_available(session.pool, session.settings, taken,
                                   counts, avail, my_turns_left)
        finally:
            cur.close()
        with lock:
            if state["generation"] != generation:
                return          # session stopped/restarted while computing
            if state["as_of_pick"] is not None and state["as_of_pick"] > picks_made:
                return          # superseded while we were computing
            state["candidates"] = frame.to_dict(orient="records")
            state["as_of_pick"] = picks_made
            # Stored WITH the candidates it belongs to, under the same lock
            # and behind the same two staleness guards, rather than
            # recomputed in live_state from that request's own pick count:
            # a list ranked against pick 18 must never be captioned "vs.
            # waiting until pick 31" because a pick landed in between. The
            # pair is written together or not at all.
            state["horizon_pick"] = int(horizon)

    def _provision_and_build(league_id, team_id, settings=None, progress=None):
        """Open (provisioning if needed) the connection this league's session
        lives on, and build the session against it.

        `settings` (a LeagueSettings, optional) is the league's real roster and
        scoring fetched from ESPN by the caller; None lets build_session read
        the database's own settings (the owner's imported league, or the
        cold-start default).

        The default league -- the __default__ sentinel or this deployment's
        DEFAULT_LEAGUE_ID -- keeps building against the shared `conn`, which
        already holds its draft history and fitted managers. Every other
        league gets its own connection to its own file, provisioned
        idempotently (a reconnect mid-draft must not wipe the `drafted` rows
        already recorded there) from this app's own database, so a league's
        rows never mix with another's or with the shared one's. Until
        per-league ESPN history import lands this is a genuine cold start for
        any league we've never seen.

        `root` is read off the module, not provision_league's import-time
        default, so a test that reassigns pipeline.leagues.LEAGUES_ROOT is
        honoured. Returns (work_conn, league_conn, session); league_conn is
        None for the default league and otherwise the caller's to register in
        state and eventually close. On any failure building the session the
        freshly opened league_conn is closed here -- nothing else holds it
        yet -- so a provisioned file is never left locked.
        """
        progress = progress or _NO_PROGRESS
        progress.begin("league")
        league_conn = None
        if (league_id and league_id != DEFAULT_LEAGUE
                and league_id != leagues_mod.DEFAULT_LEAGUE_ID):
            league_path = leagues_mod.league_db_path(
                league_id, root=leagues_mod.LEAGUES_ROOT)
            # Read BEFORE provisioning, because provision_league is
            # idempotent and afterwards the two cases are indistinguishable.
            # Worth telling apart on screen: seeding a new league's file is a
            # table-by-table copy of every universal table (2.9s measured
            # against the real 33MB database, the third most expensive thing
            # a connect does), and it happens exactly once per league -- so a
            # first connect that pauses here is doing something real, and
            # every later one flashes past.
            existed = os.path.exists(league_path)
            league_path = provision_league(
                league_id, universal_path=db_path, root=leagues_mod.LEAGUES_ROOT)
            league_conn = get_conn(league_path)
            progress.ok("league", "already provisioned" if existed
                        else "new file · seeded from the shared database")
        else:
            progress.ok("league", "the shared database")
        work_conn = league_conn if league_conn is not None else conn
        try:
            cur = work_conn.cursor()
            try:
                # The socket speaks team ids, not slots (see _slot_for_team's
                # docstring). ESPN's own pick order (_slot_from_pick_order)
                # is tried first -- exact, and already known the instant
                # `settings` was fetched, well before the socket has opened
                # -- with _slot_for_team (imported draft history) as the
                # fallback for a league whose settings fetch failed or
                # published no pickOrder. If neither resolves, my_slot stays
                # None here and the socket's TOKEN frame names our team once
                # it connects (see the my_slot back-fill in
                # _launch_listener's on_change / _resolve_slot, which tries
                # the same two plus _slot_from_socket). Read off `work_conn`,
                # the same connection build_session uses just below, so a
                # resolved slot cannot come from a different league's draft
                # order.
                #
                # `settings` here is _league_settings_from_espn's own return
                # value, which is None when the fetch failed -- so this call
                # site satisfies _slot_from_pick_order's precondition (see
                # its docstring) for free: a failed fetch falls through to
                # _slot_for_team rather than indexing last season's stale
                # order. It is the LATER resolution, off sess.settings, that
                # has to gate on the flag (see _resolve_slot).
                progress.begin("slot")
                my_slot = None
                if team_id is not None:
                    my_slot = (_slot_from_pick_order(settings, team_id)
                               or _slot_for_team(cur, team_id))
                # "of 8" comes from the settings this session is actually
                # being built with, and only when they are known at this
                # point -- ESPN's fetch has happened, the database's fallback
                # has not (build_session does that a few lines below). No
                # count rather than a count that could disagree with the
                # roster the board is about to be built for.
                teams = getattr(settings, "teams", None)
                if my_slot is not None:
                    progress.ok("slot", f"you pick {_ordinal(my_slot)}"
                                + (f" of {teams}" if teams else ""))
                    progress.fact(my_slot=int(my_slot))
                else:
                    # Genuinely unknown, not a default: no teamId in the
                    # pasted url, or a league whose pickOrder ESPN did not
                    # publish and whose managers are in no imported history
                    # (a mock's strangers). The socket's own TOKEN/SELECTING
                    # frames name the team later (see _resolve_slot), which
                    # is what this row says rather than guessing a seat --
                    # a wrong slot attributes every pick to the wrong manager
                    # and nothing downstream can detect it.
                    progress.warn("slot", "waiting on the draft socket")
                session = build_session(cur, my_slot, league_id=league_id,
                                        settings=settings, progress=progress)
            finally:
                cur.close()
        except Exception:
            if league_conn is not None:
                league_conn.close()
            raise
        return work_conn, league_conn, session

    _SCORING_LABELS = {"ppr": "PPR", "half": "half-PPR", "std": "standard"}

    def _connect_work(progress, league_id, team_id, season):
        """Everything both connect endpoints do between validating their own
        input and launching the listener, in one place so the two paths
        cannot drift -- and so the stages are recorded identically for both.

        Unchanged in order and in effect from the two copies it replaces:
        stop the previous listener, fetch ESPN's settings (best-effort),
        provision + build, fetch ESPN's team names (best-effort). The only
        additions are the progress marks around each and the ones inside
        _provision_and_build/build_session.
        """
        progress.begin("reset")
        # Exactly one listener at a time. Rather than refuse a reconnect --
        # which would trap a caller recovering from a dead listener behind a
        # separate, easy-to-forget /api/live/stop -- the old one is always
        # stopped and joined FIRST. That is also where its per-league
        # connection is closed, so the single-writer DuckDB file is free
        # before _provision_and_build reopens it. If it will not stop in
        # time, refuse rather than race it.
        if not _stop_listener():
            progress.fail(
                "reset", "the previous listener is still running",
                hint="Wait a few seconds and click the bookmark again. Two "
                     "sockets for one team is the one thing this refuses to "
                     "risk.")
            raise HTTPException(
                status_code=503,
                detail="the previous listener did not stop in time -- try again")
        progress.ok("reset", "stopped")

        progress.begin("settings")
        # The league's real roster/scoring from ESPN, so the session drafts for
        # the actual roster (rounds, starters) rather than the cold-start
        # default. None on any failure -> build_session reads the db's own.
        espn_settings = _league_settings_from_espn(league_id, season)
        if espn_settings is not None:
            progress.ok("settings", _settings_line(espn_settings))
            progress.fact(settings_source="espn", **_settings_facts(espn_settings))
        else:
            # THE SILENT FAILURE THIS SCREEN EXISTS TO SURFACE. This returns
            # None on any failure at all -- no network, a league whose
            # settings are not published, a mock that has already been torn
            # down (verified: the owner's own mock league id 1132152457 now
            # 404s) -- and build_session then quietly uses whatever the
            # database holds. That fallback decides the round count and every
            # replacement level, so a league that is not shaped like the
            # saved one is wrong everywhere downstream and nothing said so.
            # The row says only what is known HERE -- ESPN did not answer.
            # WHICH fallback is used is not known until build_session has
            # opened the league's database (the design mock's "unavailable ·
            # using saved" asserts an answer this line cannot have yet, and
            # for a league provisioned by this app it is the wrong one). The
            # shape actually used is published in `facts` a few lines below,
            # and the screen's callout is what carries it.
            progress.warn("settings", "unavailable")

        try:
            work_conn, league_conn, session = _provision_and_build(
                league_id, team_id, settings=espn_settings, progress=progress)
        except Exception as exc:      # noqa: BLE001 -- re-raised immediately;
            # this only records WHERE it died before FastAPI turns it into a
            # 500. Without it a build that raises (a duplicated player_id in
            # the ADP feed took build_session down once already, see
            # build_pool's own comment) leaves the row it died on spinning
            # and the screen says nothing about which step failed. `fail`
            # with no key lands on whichever stage was still open, which is
            # exactly the one that raised.
            progress.fail(None, f"{type(exc).__name__}: {exc}",
                          hint="This is a fault in the board build, not in "
                               "your league. The helper's log has the "
                               "traceback.")
            raise
        # Authoritative, whatever the source: these come off the settings the
        # session was ACTUALLY built with, so the handoff screen's fact grid
        # can never describe a league the board was not built for.
        # `settings_source` is set here only for the ESPN case -- build_session
        # owns the other two, because only it can tell "saved" from "default".
        progress.fact(**_settings_facts(session.settings))
        if session.settings_from_espn:
            progress.fact(settings_source="espn")

        progress.begin("teams")
        # Real ESPN team names for the board's columns, fetched once here (the
        # URL carries the season). Best-effort -- a failure leaves team_slots
        # empty and the board falls back to "Team {slot}".
        session = _attach_team_slots(session, league_id, season)
        teams = session.settings.teams
        named = len(session.team_slots)
        if named:
            progress.ok("teams", f"{named} of {teams}")
            progress.fact(
                team_names=[session.team_slots.get(s)
                            for s in range(1, teams + 1)],
                my_team=session.team_slots.get(session.my_slot))
        else:
            progress.warn("teams", f"unavailable · Team 1-{teams}")
        return work_conn, league_conn, session

    def _settings_line(settings) -> str:
        """The one line that makes 'reading league settings' worth showing:
        what it actually read."""
        fmt = _SCORING_LABELS.get(league_mod.scoring_format(settings), "?")
        return f"{settings.teams} teams · {fmt} · {settings.rounds} rounds"

    def _settings_facts(settings) -> dict:
        return {"teams": settings.teams, "rounds": settings.rounds,
                "scoring_format": league_mod.scoring_format(settings),
                "starters": dict(settings.starters),
                "flex_slots": settings.flex_slots, "bench": settings.bench}

    def _remember_my_slot(league_id, slot) -> None:
        """Update the saved session's slot, if this session is the saved one.

        UPDATES ONLY -- it never creates a record. Two guards, and both are
        load-bearing rather than defensive padding:

          * the record file must already exist, so a slot resolved by a
            session started some other way (the browser observer, or
            /api/live/start) can never write a token-bearing record, and a
            record /api/live/stop just deleted is never resurrected by a
            late frame from the listener it was stopping;
          * state["token"] must still name THIS league, because that dict is
            what the record is rebuilt from and it is replaced by every
            token connect -- writing another league's slot into this
            league's record would silently mis-seat the restored board.

        Failures are swallowed. This runs on the frame-reading thread, and an
        unwritable record is worth losing one restart's saved slot -- which
        the socket relearns from ESPN's JOIN replay anyway, whenever the
        token is still good -- rather than killing the listener mid-draft.
        """
        if not os.path.exists(session_record_path(db_path)):
            return
        with lock:
            tok = state.get("token")
        if not tok or str(tok.get("league_id")) != str(league_id):
            return
        try:
            save_session_record(db_path, league_id=tok["league_id"],
                                team_id=tok["team_id"], season=tok["season"],
                                swid=tok["swid"], token=tok["token"],
                                my_slot=slot)
        except OSError:      # noqa: BLE001 -- see the docstring
            pass

    def _socket_run_fn(league_id, team_id, swid, token, progress):
        """The bookmarklet path's `run_fn`, for _launch_listener.

        One factory rather than the same closure written out at each call
        site: the live connect and the startup restore open the same socket
        the same way, and the reason _connect_work exists -- two copies of a
        connect's steps will drift -- applies here unchanged.
        """
        def run_fn(listener, on_change, on_activity, stop_event):
            def _on_socket(handle):
                # Published once, right after run_socket_listener's first
                # successful connect (see SocketHandle's own docstring in
                # pipeline/draft_socket.py). Identity-guarded exactly like
                # every other write in this closure family: if a newer
                # /api/live/connect-token has already superseded this
                # listener by the time the connect finishes, this callback
                # must not resurrect a socket for a session that is no
                # longer the active one.
                with lock:
                    if state["listener"] is listener:
                        state["socket"] = handle
                # After the block, never inside it: `lock` is a plain Lock
                # and ConnectProgress takes it (see its docstring). This is
                # the first and only moment ESPN itself has accepted the
                # token -- the handshake behind run_socket_listener's first
                # successful connect -- so it is the honest place to say so.
                progress.ok("socket", "connected")

            run_socket_listener(listener, league_id, team_id, swid, token,
                                on_change=on_change, stop_event=stop_event,
                                on_activity=on_activity, on_socket=_on_socket)
        return run_fn

    def _launch_listener(work_conn, league_conn, league_id, session, run_fn,
                         progress=None, guard_seq=None, guard_gen=None):
        """Register a built session's listener thread and start it.

        `run_fn(listener, on_change, stop_event)` is what actually opens and
        pumps frames -- `run_listener` (browser observer) or
        `run_socket_listener` (direct socket). Everything around it is
        identical for both, so it lives here once: the pick pump, the my_slot
        back-fill, the identity guard that keeps a superseded listener's
        in-flight callback a no-op, the listener_error capture, and the state
        registration + generation bump.

        `guard_seq` (a connect sequence number from _new_progress) makes the
        registration conditional, and returns None instead of a body when it
        no longer holds. Only the startup restore passes one, and it is what
        makes the restore safe against a bookmarklet click landing while the
        restore is still building:

          * a connect that starts BEFORE this registration has already
            bumped connect_seq in _new_progress, which is the first thing
            either connect endpoint does -- so the check below, made inside
            the same critical section as the registration itself, sees it
            and this restore stands down;
          * a connect that starts AFTER this registration runs
            _stop_listener (in _connect_work) with this listener already in
            `state`, so it is stopped and joined the ordinary way.

        There is no third interleaving, which is why the check has to be
        inside the lock with the state.update rather than before it. The two
        HTTP callers pass nothing and register unconditionally, exactly as
        before.

        `guard_gen` is the session generation the same way, and it catches
        the one supersession a connect sequence cannot see: POST
        /api/live/stop bumps `generation` and nothing else, so a user who
        stops the draft during the seconds a restore is still building would
        otherwise have the restore bring a session back up underneath them.
        """
        progress = progress or _NO_PROGRESS
        listener = DraftListener(session.crosswalk)
        stop_event = threading.Event()
        # DraftSession is frozen, so the my_slot back-fill replaces the
        # session object rather than mutating it. `current` is that one
        # mutable cell, closed over by the callbacks one-to-one with `listener`.
        current = {"session": session}
        # How many picks this draft has in total, for the "the draft is
        # finished, so delete the saved token" check in on_change. Read once
        # here rather than per pick: `settings` is fixed for a session's
        # lifetime (the my_slot back-fill replaces the session but never its
        # settings). getattr with a 0 default because live_start builds
        # sessions whose settings a test may leave as None -- 0 disables the
        # check rather than raising on the frame-reading thread.
        total_picks = ((getattr(session.settings, "teams", 0) or 0)
                       * (getattr(session.settings, "rounds", 0) or 0))

        # The recompute request queue -- coalescing, depth one. search_pick
        # (this engine's predecessor) was seconds-slow; running it inline in
        # the frame callback (as this used to) blocked the socket read loop
        # for its whole duration, so in a fast draft frames -- picks, CLOCK
        # heartbeats -- piled up unread and the board fell behind. The
        # replacement (survival + rank_available) is far cheaper, but the
        # callback still only records "recompute wanted at pick N" and
        # returns instantly; the worker below does the ranking, off-thread
        # regardless of how fast it is, since nothing here depends on it
        # staying slow to justify the split.
        # Coalescing (a single latest-wins slot, not a queue) is deliberate: a
        # burst of quick picks collapses to one recompute of the final state
        # instead of a backlog of stale ones, and _recompute's own as_of_pick
        # guard drops any result a newer pick has already outrun.
        recompute_cv = threading.Condition()
        pending = {"session": None, "made": None}

        def request_recompute(sess, made):
            with recompute_cv:
                pending["session"] = sess
                pending["made"] = made
                recompute_cv.notify()

        def recompute_worker():
            while not stop_event.is_set():
                with recompute_cv:
                    while pending["made"] is None and not stop_event.is_set():
                        # Timed wait, not an infinite one: stop_event is set
                        # from _stop_listener without touching this condition,
                        # so the worker must poll it to notice a shutdown.
                        recompute_cv.wait(timeout=0.5)
                    if stop_event.is_set():
                        return
                    sess, made = pending["session"], pending["made"]
                    pending["session"] = pending["made"] = None
                # The expensive part, outside the condition lock. Skip it
                # entirely if a newer connect has already superseded this
                # listener -- no point searching against a session about to be
                # torn down, and _recompute's generation guard would discard
                # it anyway.
                with lock:
                    if state["listener"] is not listener:
                        continue
                # Guarded, because this loop IS the thread's whole body: an
                # exception propagating out of _recompute returns from
                # recompute_worker and nothing ever ranks again for the rest
                # of the draft. Candidates and as_of_pick freeze at whatever
                # pick they last reached, the listener stays perfectly
                # healthy (frames arriving, picks landing, the board
                # updating), and listener_alive -- which tracks the LISTENER
                # thread -- keeps reading true. Nothing reported it.
                #
                # Both known raisers are real, not hypothetical.
                # _drafted_state raises ValueError on a drafted row with a
                # null pick_no; live_state already wraps that same call in
                # try/except ValueError for exactly this reason.
                # `rosters[session.my_slot]` in _recompute is a bare dict
                # index over slots 1..teams and KeyErrors on anything
                # outside that range.
                #
                # Caught broadly on purpose. This is a daemon thread with no
                # other reporting path, and the failure being closed here is
                # "ranking stops silently" -- so an exception nobody
                # anticipated has to be reported too, not lost. It is
                # recorded rather than re-raised: one bad pick row must cost
                # the ranking that pick, not the rest of the draft, so the
                # loop stays alive and the next request self-heals (the
                # success branch clears the error).
                try:
                    _recompute(sess, made)
                except Exception as exc:      # noqa: BLE001 -- see above
                    with lock:
                        if state["listener"] is listener:
                            state["recompute_error"] = \
                                f"{type(exc).__name__}: {exc}"
                    # OUTSIDE the block above: `lock` is a plain Lock and
                    # every ConnectProgress method takes it (see the class
                    # docstring), so calling one from inside would deadlock.
                    # Only the FIRST ranking is a connect stage -- `fail` is
                    # a no-op once the stage is terminal, so a recompute that
                    # dies at pick 40 does not reach back and mark a connect
                    # that finished half an hour ago.
                    progress.fail("ranking", f"{type(exc).__name__}: {exc}",
                                  hint="The board is still live; the ranked "
                                       "list will retry on the next pick.")
                else:
                    with lock:
                        if state["listener"] is listener:
                            state["recompute_error"] = None
                        ranked = len(state["candidates"])
                    progress.ok("ranking", f"{ranked} ranked")

        def _resolve_slot(c2) -> bool:
            """Resolve my_slot from ESPN's pick order, history, or the
            socket, if not yet known; return True if it just resolved.

            Only reached at all when the connect path could not resolve
            my_slot up front (see _provision_and_build) -- normally because
            team_id was not yet known there (the browser-observer path with
            no teamId= in the pasted URL). Runs from on_activity (every
            frame) as well as on_change (every pick), because the moment
            that matters most -- our own team coming ON the clock -- arrives
            as a SELECTING frame, which changes no pick count and so never
            reaches on_change.

            Same order as the connect-time resolution: ESPN's pick order
            (_slot_from_pick_order) first -- exact, no history or socket
            frame needed; history (_slot_for_team) next; the socket's
            round-1 ordering (_slot_from_socket) last, for a mock or an
            un-imported league whose settings also carried no pickOrder,
            where no history can translate the team id and the slot would
            otherwise stay unknown -- leaving every recommendation blank,
            including for our first pick.

            The pick-order step is gated on `settings_from_espn`, which is
            the difference between this and the connect-time resolution:
            there `settings` is the fetch's own return value and is None when
            the fetch failed, so a failed fetch falls through on its own. Here
            it is `sess.settings`, which build_session has ALREADY replaced
            with the database's on that same failure -- and the database's
            pick order is last season's (see the flag's own comment on
            DraftSession, and the eight wrong slots it produces on this
            deployment's own data/nfl.duckdb). The reachable trigger is the
            default league + a failed settings fetch + no teamId= in the
            pasted url, i.e. the natural waiting-room url this endpoint's own
            ConnectBody docstring names. Without the gate the stale order
            wins outright over _slot_for_team, which reads the CURRENT,
            manually-configured draft_order and would have been right.
            """
            sess = current["session"]
            if sess.my_slot is not None or listener.my_team_id is None:
                return False
            resolved = None
            if sess.settings_from_espn:
                resolved = _slot_from_pick_order(sess.settings,
                                                 listener.my_team_id)
            if resolved is None:
                resolved = _slot_for_team(c2, listener.my_team_id)
            if resolved is None:
                resolved = _slot_from_socket(
                    listener, getattr(sess.settings, "teams", 0))
            if resolved is None:
                return False
            new = dataclasses.replace(sess, my_slot=resolved)
            current["session"] = new
            with lock:
                if state["listener"] is listener:
                    state["session"] = new
            # Write the slot into the saved session, so a restart does not
            # have to relearn it. This branch runs at most once per session
            # (it returns False immediately for a session that already has a
            # slot), and it is the ONLY place the slot is ever discovered in
            # the case that needs it most: a mock draft, whose managers are
            # in no imported history and whose pickOrder ESPN may not
            # publish, where _slot_from_socket read it off the socket's own
            # round-1 ordering. With an expired token there are no frames to
            # read it from a second time, so without this the restored board
            # would not know which column is ours. Outside the lock above
            # because it is file I/O -- the same rule ConnectProgress's
            # callers follow for `lock`.
            _remember_my_slot(league_id, resolved)
            return True

        def pump():
            def on_activity():
                # Liveness heartbeat: stamp last_poll_at on every frame so a
                # healthy socket never reads as stale (CLOCK ticks ~1/s while
                # any clock runs). Identity-guarded like every other write, so
                # a superseded listener cannot keep the board looking fresh
                # after it has been replaced.
                now = datetime.now(timezone.utc)
                with lock:
                    if state["listener"] is not listener:
                        return
                    state["last_poll_at"] = now
                # Resolve my_slot as soon as the socket reveals it -- crucially
                # on the SELECTING frame that puts our team on the clock, which
                # on_change never sees. Only touches the database until the
                # slot is known; once resolved this is just the stamp above.
                if current["session"].my_slot is None:
                    c2 = work_conn.cursor()
                    try:
                        if _resolve_slot(c2):
                            made = c2.execute(
                                PICKS_MADE_SQL).fetchone()[0]
                            request_recompute(current["session"], made)
                    finally:
                        c2.close()

            def on_change():
                # on_activity first: stamps freshness and (on the browser
                # path, which has no per-frame hook of its own) is the only
                # place my_slot gets resolved.
                on_activity()
                # stop_event only asks the poll loop to exit; it does not gate
                # an in-flight callback a newer connect may have superseded.
                # Checking identity here is what keeps a superseded listener's
                # write a no-op instead of a race (see state["listener"]).
                with lock:
                    if state["listener"] is not listener:
                        return
                c2 = work_conn.cursor()
                try:
                    live = listener.picks()
                    apply_picks(c2, live)
                    made = c2.execute(PICKS_MADE_SQL).fetchone()[0]
                finally:
                    c2.close()
                # Surface picks the crosswalk could not resolve. They are
                # computed on every fold (LivePicks.unmapped) but were being
                # discarded, so /api/live/state's unmapped_picks read empty
                # even when a real pick -- most often a D/ST whose id didn't
                # map -- silently vanished from the board. Storing them makes
                # that visible (and is how a wrong D/ST id scheme gets caught:
                # the raw espn id shows up here). Same identity guard as every
                # other write.
                with lock:
                    mine = state["listener"] is listener
                    if mine:
                        state["unmapped"] = live.unmapped
                # The draft is over, so the saved token is worthless -- ESPN
                # refuses it from here on -- and there is nothing left for a
                # restart to reconnect to. Deleted at the moment that becomes
                # true rather than left to the age rule, so the file with the
                # token in it is gone the second it stops being useful. This
                # is the listener path, the one writer of `drafted`; it adds
                # no writer and no table. `>=`, not `==`: the pick count is
                # max(pick_no) (see PICKS_MADE_SQL), which a manually
                # inserted row can push past the last snake slot. on_change
                # only fires when the pick count MOVED, so this runs once at
                # the end of a draft and not on every later frame. Outside
                # the lock above because it is a syscall, and nothing reads
                # the record under `lock`.
                if mine and total_picks and made >= total_picks:
                    clear_session_record(db_path)
                # Hand off to the worker instead of searching here -- this
                # callback runs on the socket read thread and must return fast.
                request_recompute(current["session"], made)

            try:
                run_fn(listener, on_change, on_activity, stop_event)
            except Exception as exc:      # noqa: BLE001 -- any failure (bad
                # url/token, ESPN unreachable, Playwright missing) must reach
                # /api/live/state instead of dying silently on a daemon
                # thread. Only recorded if this is still the active listener,
                # so a superseded one mid-shutdown cannot clobber the error of
                # whatever replaced it.
                with lock:
                    if state["listener"] is listener:
                        state["listener_error"] = str(exc)
                # Outside the lock (see the recompute worker's own note, and
                # ConnectProgress's docstring). This is the failure the
                # connect screen has to be able to name: run_socket_listener
                # raises here after MAX_EMPTY_RECONNECTS frameless attempts,
                # which is what an expired draft token looks like, and until
                # now it surfaced only as a red pill inside a draft room the
                # user had already been handed. `fail` lands on the socket
                # stage on the bookmarklet path and, on the browser path
                # (which has no socket row at all), on whichever stage was
                # still open -- never nowhere.
                progress.fail("socket", str(exc),
                              hint="Go back to your ESPN draft tab and click "
                                   "the Draft Helper bookmark again -- it "
                                   "mints a fresh token.")

        thread = threading.Thread(target=pump, daemon=True)
        recompute_thread = threading.Thread(target=recompute_worker, daemon=True)
        # Pick count as of launch, for the one recompute this function
        # requests below. Read HERE -- on the connect handler's own thread,
        # on the connection it just built this session with, and BEFORE that
        # connection is registered in `state` -- so a concurrent
        # /api/live/stop can only ever be closing the PREVIOUS session's
        # league_conn, never this one. It is the same single COUNT(*) that
        # on_change already runs once per pick, on the same connection, and
        # `drafted` always exists (pipeline/db.py's get_conn creates it).
        made_at_launch = 0
        c0 = work_conn.cursor()
        try:
            made_at_launch = c0.execute(PICKS_MADE_SQL).fetchone()[0]
        finally:
            c0.close()
        with lock:
            # Superseded before we ever registered (see `guard_seq`). Nothing
            # has been published yet -- neither thread is started, so there
            # is nothing to stop -- and the connection this build opened is
            # ours alone to close, which happens just below.
            superseded = (guard_seq is not None
                          and (state["connect_seq"] != guard_seq
                               or state["generation"] != guard_gen))
            if not superseded:
                state.update({"session": session, "listener": listener,
                              "listener_thread": thread,
                              "listener_stop": stop_event,
                              "recompute_thread": recompute_thread,
                              "listener_error": None, "recompute_error": None,
                              "league_conn": league_conn,
                              "candidates": [], "as_of_pick": None,
                              "horizon_pick": None,
                              "unmapped": [], "last_poll_at": None})
                state["generation"] = state.get("generation", 0) + 1
        if superseded:
            # DuckDB tolerates a second connection to a file already open in
            # this process (verified against the duckdb 1.5.5 this project
            # pins: two connections, one writing, both fine -- it shares the
            # database instance), so the connect that beat us has already
            # reopened this league's file and closing ours cannot pull it
            # out from under anyone.
            if league_conn is not None:
                league_conn.close()
            return None
        recompute_thread.start()
        # ONE recompute at launch, when the slot is already known. Without it
        # nothing ever asked for a ranking until a pick landed:
        # request_recompute was called only from on_change (a pick) and from
        # on_activity gated on _resolve_slot having JUST returned True --
        # which can never happen once connect has already resolved my_slot,
        # since _resolve_slot returns False immediately for a session that
        # has one. So a connect at pick 0 with the slot known sat on the []
        # this function initialises `candidates` to, through the owner's
        # entire first pick. (The vor fallback in live_state now guarantees
        # the room is never EMPTY; this is what makes a real, slot-aware
        # ranking -- gain_now, survive_pct, fills -- actually arrive. Both
        # are needed: a full recompute measures 0.97-1.49s against the real
        # 249-player pool at 400 rollouts, which is over a second of board
        # the fallback has to cover, and reconnecting mid-draft while on the
        # clock would otherwise get no gain_now at all until the next pick.)
        #
        # Requested BEFORE thread.start(), i.e. before any frame can arrive:
        # `pending` is a coalescing depth-one slot (latest write wins, not
        # max), so a request made after the listener was already running
        # could overwrite a fresher on_change request with this stale
        # `made_at_launch`. Ordering it ahead of the listener removes that
        # race rather than guarding against it.
        #
        # Skipped when my_slot is None: _recompute returns immediately in
        # that case anyway (survival needs a real slot), and on_activity's
        # _resolve_slot path still fires the recompute the moment the socket
        # names our team -- that path is unchanged and still needed.
        if session.my_slot is not None:
            request_recompute(session, made_at_launch)
            progress.begin("ranking")
        else:
            # Terminal, and honestly so: with no slot there is nothing to rank
            # FOR, and this connect will never request one (see the note
            # above). The room still fills -- live_state's vor fallback serves
            # the pool ranked by value over replacement -- and the slot-aware
            # ranking arrives on its own the moment the socket names our team.
            # Left `pending` instead, the screen would wait for a stage that
            # is never coming.
            progress.warn("ranking", "waiting on your slot")
        progress.begin("socket")
        thread.start()
        # my_slot is echoed back deliberately: None means genuinely undetected
        # yet, not a default. A resolved value went through draft_teams ->
        # draft_order by manager, which a re-randomised draft order would make
        # silently wrong -- a human glancing at "you're drafting from slot 4"
        # catches that; nothing else does. So the real value (or its absence)
        # is returned, never a guess.
        return {"connected": True, "league_id": league_id,
                "board_fingerprint": session.board_fingerprint,
                "my_slot": session.my_slot}

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
                          "as_of_pick": None, "horizon_pick": None,
                          "unmapped": [], "last_poll_at": None,
                          "recompute_error": None})
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
                        # Same "present with a null/false value, never
                        # omitted" convention the rest of this branch
                        # follows: no session means no ranking and so no
                        # pick it was measured against.
                        "horizon_pick": None,
                        "horizon_is_end_of_draft": False,
                        "last_poll_at": None, "stale": True,
                        "unmapped_picks": [], "listener_error": None,
                        "listener_alive": False, "recompute_error": None,
                        # Same "present with a null/false value, never
                        # omitted" convention listener_alive already
                        # follows on this branch: the room reads it to
                        # decide whether the draft buttons are live, and an
                        # absent key would read as undefined -- falsy by
                        # luck rather than by contract.
                        "socket_alive": False,
                        # Null, never False. There is no listener on this
                        # branch and so nothing has heard ESPN say either
                        # way -- and "we have not been told" is a different
                        # fact from "ESPN says autodraft is off" (see
                        # DraftListener.my_autodraft). Present rather than
                        # omitted, same convention as every other key here.
                        "autodraft": None,
                        "token_received": state.get("token") is not None,
                        "ms_remaining": None,
                        # No listener at all on this branch, so genuinely
                        # unstarted rather than unknown -- same reasoning as
                        # every other false/null default here (see
                        # listener_alive's own comment just above).
                        "draft_started": False,
                        # THE ONE THING THAT MAKES A BACKGROUND RESTORE
                        # HONEST. After a restart with a saved session there
                        # genuinely is no session yet -- build_session takes
                        # 4.1-34.8s (see its docstring) and the API answers
                        # requests throughout -- so this branch is what a
                        # poll sees for those seconds. Without this key it
                        # is indistinguishable from "no draft is running",
                        # which is the exact silence the restore exists to
                        # remove: the owner would be told nothing is
                        # happening at the moment something is. `restore_
                        # error` is the other half, for a restore that got
                        # as far as trying and could not rebuild at all.
                        "restoring": (state["restore_thread"] is not None
                                      and state["restore_thread"].is_alive()),
                        "restore_error": state["restore_error"],
                        "settings": _league_settings_payload(None),
                        "my_roster": []}
            snapshot = dict(state)
            listener = snapshot["listener"]
            # Straight off the listener, read here rather than after `lock`
            # releases: not because a single int attribute read is unsafe
            # (pipeline/draft_listener.py sets it with a plain assignment,
            # effectively atomic under the GIL) but so this value is drawn
            # from the same consistent snapshot as everything else below --
            # the same reasoning the picks_made query already follows. None
            # until the first CLOCK or SELECTING frame has landed (see
            # DraftListener.ms_remaining's own field comment) -- never a
            # decayed or interpolated guess, so a genuinely stale value
            # never gets rendered as a live one.
            ms_remaining = listener.ms_remaining if listener is not None else None
            # Whether the draft has actually started, straight off
            # DraftListener.started -- set once, True, by the socket's own
            # STATE frame (pipeline/draft_listener.py). Read here, same
            # snapshot and same reasoning as ms_remaining just above: Defect
            # 4 needs the rail to tell "the draft has not started yet" apart
            # from "started, waiting on someone else's pick" -- both used to
            # render as the same "Waiting on the room" heading, with no way
            # for the drafter to know which one they were looking at.
            draft_started = listener.started if listener is not None else False
            # ESPN's own autodraft flag for OUR team -- true means ESPN
            # is making this session's picks itself, which is the alarm the
            # room's clock panel exists to raise. True/False/None, where
            # None means ESPN has not said yet (no AUTODRAFT frame for our
            # team, or TOKEN has not named our team) -- see
            # DraftListener.my_autodraft for why that must not read as
            # False.
            #
            # Read here, inside `lock`, for the same reason ms_remaining and
            # draft_started just above are: so this response is one
            # consistent snapshot. `lock` does not exclude the listener
            # thread, which folds frames without holding it -- but this read
            # needs the consistency more than those two do, not less. It is
            # a COMPOUND read (my_team_id, then a dict lookup keyed by it)
            # rather than one int attribute, and a torn one is a real bug:
            # taking my_team_id from before a reconnect resolved it and the
            # dict from after would report a flag looked up under the wrong
            # team's key. `my_autodraft` performs both halves inside the one
            # property so the pair is drawn at a single instant.
            autodraft = listener.my_autodraft if listener is not None else None
            # Whether a SELECT actually has somewhere to go, read here for
            # the same reason ms_remaining is: it belongs to this response's
            # one consistent snapshot. Exactly the condition
            # /api/live/select's 503 already gates on (`socket is None or
            # not socket.alive()`), served so the room can gate the draft
            # buttons on the same fact instead of on `on_the_clock ===
            # my_slot` alone. Without it, a run_socket_listener reconnect --
            # where the handle is detached but the listener thread is alive
            # and `stale` has not tripped, since CLOCK frames stamped
            # last_poll_at a moment ago -- leaves the buttons enabled and
            # every click 503s (spec section 6).
            #
            # Calling alive() under `lock` is safe: SocketHandle's own lock
            # is only ever taken in attach/detach/alive/send, and none of
            # those calls back into anything that takes `lock` (on_socket is
            # invoked AFTER attach has released it), so there is no lock
            # ordering to invert. The call itself is a `is not None` under an
            # uncontended lock.
            socket_handle = snapshot["socket"]
            socket_alive = socket_handle is not None and socket_handle.alive()
            # Same connection choice as _recompute: the league this session
            # belongs to, not always the shared `conn`, or the picks-made
            # count (and the on-the-clock slot derived from it) would be
            # read off the wrong league's `drafted` table.
            #
            # Read while STILL holding `lock`, not after releasing it: this
            # is a plain HTTP GET handler, running on whatever thread
            # FastAPI happens to hand the request -- unlike _recompute (see
            # its own comment), nothing here ties this call to the thread
            # that owns the connection. A concurrent /api/live/stop, on a
            # different request thread, closes exactly this connection
            # under the same lock; snapshotting it and querying after
            # release would leave a window where that close lands in
            # between, and this count then runs against a closed
            # connection. The query is one fast COUNT(*), so holding the
            # lock across it costs negligible contention, and it cannot
            # deadlock: the query itself takes no lock of its own, and
            # nothing else holds `lock` while blocking on the database.
            cur = (snapshot["league_conn"] or conn).cursor()
            try:
                picks_made = cur.execute(PICKS_MADE_SQL).fetchone()[0]
                # My own roster so far. Only queried when my_slot is known --
                # _my_roster returns [] unconditionally otherwise, so the
                # extra read would be wasted -- and wrapped against the one
                # documented failure of _drafted_state: a drafted row with no
                # pick_no cannot be attributed to a slot at all (see its own
                # docstring), which must cost this response its roster, not
                # the clock and listener health the rest of it still owes.
                # Measured against the real production pool (data/nfl.duckdb,
                # 249 players, 8 teams): _drafted_state + _seed_rosters
                # together run in ~0.5-0.6ms even at picks_made=120 -- three
                # orders of magnitude under the 2.5s poll cadence, so this
                # runs on every poll rather than being cached against the
                # pick count.
                my_roster = []
                # Defect 2: "who is available" (the pool minus `drafted`) is
                # always knowable and must render even before "how they rank
                # for YOUR roster" is -- that part is what genuinely needs a
                # slot. `_drafted_state` is called ONCE here and both halves
                # read off it: `taken_order` attributes the picks (my_roster),
                # `taken` is the mask the fallback ranking needs. One call,
                # not one per branch, because both are wanted on the same
                # request now and they must describe the same instant.
                #
                # THE FALLBACK IS GATED ON THE RANKED LIST BEING EMPTY, NOT
                # ON my_slot BEING UNKNOWN, and that is the whole fix for
                # this defect. Gated on `my_slot is None` the two halves of
                # the original repair cancelled out and the owner's original
                # complaint came straight back: connect resolves my_slot from
                # ESPN's pickOrder before a single frame arrives
                # (_slot_from_pick_order), so the `else` branch could no
                # longer run -- while nothing requested a recompute at launch,
                # leaving `state["candidates"]` at the [] _launch_listener
                # initialises it to until the FIRST PICK LANDED. Reproduced
                # end to end against a real TestClient (pickOrder
                # [3,7,1,2,4,5,6,8], url carrying teamId=2, run_listener
                # stubbed to a no-op): connect returned my_slot=4 and
                # /api/live/state then served picks_made=0,
                # len(candidates)=0, as_of_pick=None -- and stayed empty at
                # 3s. The same seed with the teamId stripped out of the url
                # served my_slot=None and len(candidates)=1. An empty list
                # renders as the literal string "No candidates yet.", so the
                # worst case was the owner in slot 1, on the clock for pick 1,
                # 30-second timer, empty board.
                #
                # `not candidates` covers BOTH branches with one condition
                # and keeps the single-payload-shape property: whatever the
                # reason the ranked list is empty -- my_slot unknown, the
                # launch recompute still running (0.97-1.49s measured on the
                # real 249-player pool, see _launch_listener), a dead
                # recompute worker -- the room gets the pool instead of
                # nothing. It cannot mask a real result: `rank_available` and
                # `available_by_vor` build off the same `~taken` mask, so the
                # ranked list is empty only when the fallback would be too.
                #
                # available_by_vor needs only `taken` (cheap: see the timing
                # note just above) -- ranked by the board's own vor_points,
                # with gain_now/survive_pct/fills honestly None rather than a
                # fabricated 0.0/"" (scoring/gain.py's own docstring).
                # candidates_as_of_pick is simply `picks_made` here: unlike
                # the async-computed slot-ranked list, this is never stale --
                # it is recomputed against the current `taken` mask on every
                # single poll -- so DraftRoom's "recomputing for pick N"
                # banner (candidates_as_of_pick < picks_made) correctly never
                # fires for it.
                candidates = snapshot["candidates"]
                candidates_as_of_pick = snapshot["as_of_pick"]
                horizon_pick = snapshot["horizon_pick"]
                try:
                    taken, taken_order = _drafted_state(cur, session.pool)
                except ValueError:
                    taken = taken_order = None
                if taken_order is not None and session.my_slot is not None:
                    my_roster = _my_roster(session, taken_order)
                if not candidates and taken is not None:
                    candidates = available_by_vor(
                        session.pool, taken).to_dict(orient="records")
                    candidates_as_of_pick = int(picks_made)
                    # This fallback list is ranked by vor_points alone,
                    # against nothing -- gain_now/survive_pct are None on
                    # every row of it. Naming a horizon pick here would
                    # caption a list that was never measured against one.
                    horizon_pick = None
            finally:
                cur.close()
        slots = snake_slots(session.settings.teams, session.settings.rounds)
        on_clock = slots[picks_made] if picks_made < len(slots) else None
        thread = snapshot["listener_thread"]
        # The horizon _recompute actually measured this list against, split
        # into the two things the room has to be able to say. A horizon past
        # the last pick of the draft is `_horizon_pick_for`'s off-the-end
        # sentinel -- I hold no turn after this one, so survival ran to the
        # end of the draft. That is a real state (my final pick, every
        # round-15 wheel) and it must read as "the end of the draft", never
        # as pick 121 of a 120-pick draft. `horizon_pick` is None both then
        # and before any ranking exists, which the room already handles by
        # dropping the clause; the flag is what tells the two apart.
        horizon_is_end_of_draft = (horizon_pick is not None
                                   and horizon_pick > len(slots))
        return {
            "active": True,
            "picks_made": int(picks_made),
            "on_the_clock": on_clock,
            "horizon_pick": (None if horizon_pick is None
                             or horizon_is_end_of_draft else int(horizon_pick)),
            "horizon_is_end_of_draft": horizon_is_end_of_draft,
            "my_slot": session.my_slot,
            "draft_started": draft_started,
            "candidates": candidates,
            "candidates_as_of_pick": candidates_as_of_pick,
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
            # The recompute worker fails independently of the listener, and
            # its failure is quieter: the clock keeps ticking, the board
            # keeps filling, and only the ranking stops. Reported separately
            # for that reason -- see the state key's own comment.
            "recompute_error": snapshot["recompute_error"],
            "socket_alive": socket_alive,
            "autodraft": autodraft,
            "token_received": snapshot.get("token") is not None,
            "ms_remaining": ms_remaining,
            # Present on both branches with the same meaning, same
            # convention as listener_alive and autodraft: a session exists,
            # so whatever the restore was doing is over. It reads off the
            # same thread handle rather than a hardcoded False because the
            # restore losing a race to a real connect (see _launch_listener's
            # guard_seq) leaves the thread finishing up for a moment after
            # the connect's session is already live, and reporting that
            # honestly costs nothing.
            "restoring": (snapshot["restore_thread"] is not None
                          and snapshot["restore_thread"].is_alive()),
            "restore_error": snapshot["restore_error"],
            "settings": _league_settings_payload(session.settings),
            "my_roster": my_roster,
        }

    @app.get("/api/live/connect-progress")
    def live_connect_progress():
        """What the connect is doing right now, stage by stage.

        A separate endpoint from /api/live/state, and a deliberately tiny
        one: it touches no database at all (state's own handler runs a
        COUNT(*), a _drafted_state replay and a vor fallback ranking on every
        call), because the connect screen polls this several times a second
        while the connect thread is busy building a board. It also has to
        answer BEFORE there is a session, which is precisely the window
        /api/live/state reports as `active: false` and nothing else.

        Serving it while a connect is blocked in POST /api/live/connect-token
        works because that handler is a plain `def`, so Starlette runs it in
        the threadpool and the event loop stays free -- verified against a
        real uvicorn (a 5s blocking sync POST, GETs answering in 2-18ms
        throughout). That is the whole reason the connect endpoints keep
        their existing synchronous contract, response shape and error
        semantics: the progress is carried by a second request, not by
        turning the first one into a job queue.

        `phase: "idle"` (never a 404) for a helper that has not been
        connected since it started -- the same "present with an empty value"
        convention live_state's inactive branch follows.
        """
        with lock:
            progress = state["connect"]
            body = progress.snapshot() if progress is not None else None
        if body is None:
            return {"phase": "idle", "stages": [], "facts": {},
                    "error": None, "elapsed_ms": 0}
        return body

    @app.get("/api/live/board")
    def live_board():
        """The full draft-board grid: every column named, every pick placed.

        Read-only and purely additive to the live session -- it touches none
        of the recompute/listener machinery, only the same `drafted` rows
        live_state reads. The board itself (player stats) and the slot->name
        map were both computed once, on connect, and are read off the session.
        """
        with lock:
            session = state["session"]
            if session is None:
                return {"active": False}
            snapshot = dict(state)
            # Same connection choice and the same under-the-lock discipline as
            # live_state (see its long comment): read the drafted rows off the
            # league this session belongs to, and while STILL holding `lock`,
            # so a concurrent /api/live/stop closing that connection cannot
            # land between the snapshot and the query. Unlike live_state this
            # pulls the rows themselves, not just a count -- the grid needs
            # each pick's player id and overall number.
            cur = (snapshot["league_conn"] or conn).cursor()
            try:
                picks = cur.execute(
                    "SELECT player_id, pick_no FROM drafted").fetchall()
            finally:
                cur.close()

        settings = session.settings
        teams, rounds = settings.teams, settings.rounds
        slots = snake_slots(teams, rounds)
        # The highest pick number, not the row count -- see PICKS_MADE_SQL,
        # which live_state runs for exactly this number. Computed from the
        # rows already in hand rather than a second query, but it has to be
        # the same quantity: the two endpoints are polled together and the
        # room draws one clock from them.
        picks_made = max((int(pick_no) for _, pick_no in picks), default=0)
        on_clock = slots[picks_made] if picks_made < len(slots) else None

        columns = [
            {"slot": slot,
             "team_name": session.team_slots.get(slot, f"Team {slot}"),
             "is_me": slot == session.my_slot}
            for slot in range(1, teams + 1)]

        by_id = _board_index(session.board)
        cells = [_board_cell(pid, pick_no, teams, slots, by_id)
                 for pid, pick_no in picks]

        return {
            "active": True,
            "teams": teams,
            "rounds": rounds,
            "my_slot": session.my_slot,
            "on_the_clock": on_clock,
            "picks_made": picks_made,
            "columns": columns,
            "cells": cells,
        }

    @app.post("/api/live/select")
    def live_select(body: SelectBody):
        """Make the pick: send SELECT on the live draft socket and wait for
        ESPN's own SELECTED to confirm it landed.

        Writes nothing to `drafted`. `on_change` already records every
        SELECTED frame this socket sees, including this one (see
        DraftListener.on_frame / selected_espn_ids) -- a second writer here
        could let the board claim a pick ESPN did not actually take, and the
        one table write that must win is ESPN's own confirmation, not a
        guess made before it arrives.

        `socket.send` is wrapped in `except (ConnectionError, OSError,
        ConnectionClosed)`, not just the first two. SocketHandle.send (see
        pipeline/draft_socket.py) raises the builtin ConnectionError only on
        its "nothing attached" path. In the race where this request reads
        the socket reference just before the listener thread detaches and
        closes it, the real websockets.sync.client connection underneath
        raises websockets.exceptions.ConnectionClosed instead -- and that is
        NOT a subclass of ConnectionError (its MRO is ConnectionClosed ->
        WebSocketException -> Exception -> BaseException -> object).
        Catching only the builtin would let exactly that race leak an
        unhandled exception into a 500, on the one failure -- a dropped
        socket -- this endpoint exists to turn into a clean 503 instead.
        Nothing broader than these three is caught: a real bug here (a
        TypeError from a malformed payload, say) must still surface as a
        500, not be laundered into "the socket is down."
        """
        with lock:
            session = state["session"]
            socket = state["socket"]
            listener = state["listener"]
            active_conn = state["league_conn"] or conn
            if session is None:
                raise HTTPException(status_code=409, detail="no live draft session")
            if socket is None or not socket.alive():
                raise HTTPException(
                    status_code=503, detail="the draft socket is not connected")
            cur = active_conn.cursor()
            try:
                picks_made = cur.execute(PICKS_MADE_SQL).fetchone()[0]
                already = cur.execute(
                    "SELECT count(*) FROM drafted WHERE player_id = ?",
                    [body.player_id]).fetchone()[0]
            finally:
                cur.close()

        if already:
            raise HTTPException(
                status_code=409, detail="that player is already drafted")
        if session.my_slot is None:
            raise HTTPException(
                status_code=409, detail="this session has no draft slot yet")
        # picks_until_turn falls through to 0 ("my turn") once picks_made
        # reaches the end of the slot list -- its loop simply has nothing
        # left to range over, not "it is my turn again." Nothing else in this
        # endpoint rules that state out on its own: session stays non-None,
        # the socket can still be alive, and any board player nobody drafted
        # (there are always more of those than teams*rounds) sails past the
        # `already` check. live_state and live_board already guard this exact
        # shape (`picks_made < len(slots)`, api/live.py's on_clock
        # computation); mirror it here rather than changing
        # picks_until_turn's own contract, which Task 3's review established
        # has no other production caller and whose existing tests pin its
        # current return value.
        slots = snake_slots(session.settings.teams, session.settings.rounds)
        if picks_made >= len(slots):
            raise HTTPException(status_code=409, detail="the draft is over")
        if picks_until_turn(session.settings, session.my_slot, picks_made) != 0:
            raise HTTPException(status_code=409, detail="it is not your turn")

        row = session.board_by_id.get(body.player_id)
        if row is None:
            raise HTTPException(
                status_code=400, detail=f"unknown player {body.player_id}")
        espn_id = _espn_id_for(row)
        if espn_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"no ESPN id for {row.get('name', body.player_id)} -- "
                "this is a crosswalk gap, pick him in ESPN directly")

        # `selected_espn_ids` accumulates for the whole session and is never
        # cleared -- ESPN replays the draft so far on every JOIN, so it holds
        # every SELECTED this socket has ever seen, reconnect replays
        # included. The wait loop below only tests membership, so an id that
        # was ALREADY in the set returns on its very first iteration: a 200
        # carrying `picks_made + 1`, which is a pick number that belongs to
        # somebody else, for a pick this request never made. The dialog
        # closes saying it landed and the user walks away without a player.
        #
        # It takes `drafted` and `selected_espn_ids` disagreeing to get here,
        # since the `already` check above would otherwise have caught it --
        # which is exactly the unmapped-pick case the rest of this file
        # acknowledges is real (see state["unmapped"]): ESPN confirmed a
        # player the crosswalk could not resolve, so he is in the set and not
        # in `drafted`. Checked BEFORE the send, so the guard cannot be
        # confused by this request's own confirmation arriving.
        if listener is not None and espn_id in listener.selected_espn_ids:
            raise HTTPException(
                status_code=409,
                detail=f"ESPN has already confirmed a pick of "
                f"{row.get('name', body.player_id)} -- the board has not "
                "recorded it (most likely a crosswalk gap), so check the "
                "ESPN draft room rather than picking him again")

        try:
            socket.send(f"SELECT {espn_id}\n")
        except (ConnectionError, OSError, ConnectionClosed) as exc:
            raise HTTPException(
                status_code=503, detail=f"could not reach ESPN: {exc}") from exc

        # Poll rather than block on a signal: the listener thread fills
        # selected_espn_ids from its own frame-reading loop with nothing to
        # notify this request when it happens. `listener` was captured once,
        # above, under `lock` -- not re-read on every iteration. If the
        # session restarts mid-wait (a user reconnecting while this request
        # is still in flight), that stale reference is the SAFE side of the
        # tradeoff: the old listener's thread has already stopped reading
        # frames (see _stop_listener, which joins it before a new one
        # starts), so its selected_espn_ids simply stops changing and this
        # request correctly times out at 504 -- it never reports a
        # confirmation on behalf of a listener that no longer speaks for the
        # active session, even if ESPN's replay-on-reconnect later confirms
        # the same pick to the NEW listener this request never sees.
        deadline = time.monotonic() + SELECT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if listener is not None and espn_id in listener.selected_espn_ids:
                return {"player_id": body.player_id, "espn_id": espn_id,
                        "pick_no": picks_made + 1}
            time.sleep(SELECT_POLL_SECONDS)
        raise HTTPException(
            status_code=504,
            detail="ESPN did not confirm the pick -- check the ESPN draft room "
            "before picking again")

    @app.post("/api/live/autodraft")
    def live_autodraft(body: AutodraftBody):
        """Turn ESPN's autodraft on or off, and report only what ESPN
        confirmed.

        Miss a pick and ESPN puts your team on autodraft: it starts making
        the picks for you, and nothing in its UI comes to this tool to say
        so. `DraftListener` now hears that (see its AUTODRAFT branch); this
        is the way back out.

        The same round trip as /api/live/select, deliberately -- send the
        command, wait for the server's own frame to echo it back, return on
        confirmation only. Nothing here writes the flag into the listener:
        `on_frame` records it when, and only when, ESPN's AUTODRAFT frame
        arrives, so a request that times out leaves the state exactly as
        ESPN last stated it rather than claiming a change it could not
        verify. That is the same no-optimistic-state rule the pick dialog
        follows, and it matters more here, not less: a user who wrongly
        believes they have taken autodraft back off will stop watching the
        clock.

        Both frame formats verified against data/draft_room_trace.jsonl:
          - OUTBOUND `AUTODRAFT <true|false>`, with NO team id -- line 1186,
            the real client turning its own autodraft back off. The socket
            already identifies the team it speaks for (it is the `3=` and
            `5=` team of the JOIN url, see draft_socket.socket_url).
          - INBOUND `AUTODRAFT <teamId> <true|false>` -- line 1188, ESPN
            answering that send two frames later. Broadcast to the whole
            room for every team, which is exactly why the wait below has to
            match on OUR team id rather than merely on the verb: teams 2, 3
            and 7 all flip across this one capture, and confirming on a
            neighbour's frame would report a change that never happened to
            us.

        There is no off-turn guard, unlike /api/live/select, and that is a
        reading of the capture rather than an omission: the client's own
        `AUTODRAFT false` at line 1186 sits between two OTHER teams' picks
        (line 1185 `SELECTED 4 4432708 12`, line 1189 `SELECTING 3 30000`).
        Turning autodraft off while somebody else is on the clock is what
        ESPN's own room does -- and it is the case that matters most, since
        a user who has just been flipped onto autodraft has by definition
        already lost their turn.

        `socket.send`'s except clause is the same triple as live_select's,
        for the same documented reason (see its docstring): ConnectionClosed
        is not a subclass of ConnectionError, and the socket dropping mid-
        request is the one failure this endpoint exists to turn into a clean
        503 rather than a 500.
        """
        with lock:
            session = state["session"]
            socket = state["socket"]
            listener = state["listener"]
            if session is None:
                raise HTTPException(status_code=409, detail="no live draft session")
            if socket is None or not socket.alive():
                raise HTTPException(
                    status_code=503, detail="the draft socket is not connected")
            # Without our own team id there is no way to tell our echo from
            # the seven other teams ESPN broadcasts the same verb for, so
            # this request could only ever guess -- refuse instead. In
            # practice a live socket resolves this within a frame or two of
            # connecting (TOKEN lands eight frames in, see the capture), so
            # this is the first second of a session, not a dead end.
            if listener is None or listener.my_team_id is None:
                raise HTTPException(
                    status_code=409,
                    detail="this session does not know its ESPN team yet -- "
                    "give the draft socket a moment and try again")
            # Already there, as far as ESPN's own last word goes. Returning
            # here rather than sending is not a shortcut: ESPN has never
            # been observed re-broadcasting AUTODRAFT for a value that did
            # not change, so a redundant send would most likely sit out the
            # full timeout and then 504 -- reporting failure for a state
            # that is already exactly what was asked for. This is also what
            # makes the endpoint idempotent under the one real race: ESPN
            # flipping us on (or the user double-clicking) between the poll
            # that drew the switch and the click that acted on it.
            current = listener.my_autodraft
            if current == body.on:
                return {"autodraft": body.on, "changed": False}

        try:
            socket.send(f"AUTODRAFT {'true' if body.on else 'false'}\n")
        except (ConnectionError, OSError, ConnectionClosed) as exc:
            raise HTTPException(
                status_code=503, detail=f"could not reach ESPN: {exc}") from exc

        # Polled, not signalled, and against a `listener` captured once
        # under `lock` -- both exactly as live_select does it, and the
        # stale-reference tradeoff there applies here unchanged: a session
        # restarted mid-wait leaves this reading a listener whose thread has
        # already stopped, so the flag simply stops changing and this
        # request times out honestly instead of confirming on behalf of a
        # listener that no longer speaks for the live session.
        deadline = time.monotonic() + AUTODRAFT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if listener.my_autodraft == body.on:
                return {"autodraft": body.on, "changed": True}
            time.sleep(AUTODRAFT_POLL_SECONDS)
        raise HTTPException(
            status_code=504,
            detail="ESPN did not confirm the autodraft change -- as far as "
            f"this tool knows autodraft is still "
            f"{'on' if current else 'off' if current is False else 'unknown'}; "
            "check the ESPN draft room")

    @app.post("/api/live/stop")
    def live_stop():
        stopped = _stop_listener()
        with lock:
            state["generation"] += 1
            # league_conn is deliberately left out of this update.
            # _stop_listener already closed and cleared it on the path where
            # it confirmed the listener thread had actually exited -- this
            # is None on that path, and there is nothing left to do. On a
            # timeout (stopped is False) the thread is confirmed still
            # alive, and closing the connection out from under a thread that
            # might be using it right now would turn "stop this listener"
            # into "corrupt a DuckDB connection in use" -- a leaked
            # connection is recoverable on restart, a use-after-close is
            # not. `session`/`listener`/`listener_thread` are still cleared
            # unconditionally below (unchanged from before this connection
            # existed): that only detaches new work from the old thread via
            # the identity guard, it does not touch anything the thread
            # itself might still hold open.
            state.update({"session": None, "candidates": [],
                          "as_of_pick": None, "horizon_pick": None,
                          "unmapped": [], "last_poll_at": None,
                          "listener": None,
                          "listener_thread": None, "listener_stop": None,
                          "recompute_thread": None, "listener_error": None,
                          "recompute_error": None, "restore_error": None})
        # An explicit stop is the user saying this draft is over for them, so
        # the saved token goes with it -- there is nothing left that a restart
        # should silently reconnect to. Done AFTER the listener is stopped, so
        # its own last on_change (which can also delete the record, at the end
        # of a draft) cannot race a recreate; and outside `lock`, because it
        # is a syscall and nothing reads the record under the lock.
        # Deliberately not swallowed: a token this failed to delete is still
        # on disk, and the user who pressed stop deserves to hear that rather
        # than a quiet 200.
        clear_session_record(db_path)
        return {"active": False, "listener_stopped": stopped}

    @app.post("/api/live/connect")
    def live_connect(body: ConnectBody):
        progress, _seq = _new_progress(token_path=False)
        progress.begin("token")
        # Validate the one thing that can be invalid (the league id) BEFORE
        # tearing down a working listener -- an invalid request must never
        # stop one that was running. Team id / slot resolution never raises
        # (see _slot_for_team's docstring).
        try:
            league_id = _resolve_league_id(body.url)
        except HTTPException as exc:
            progress.fail("token", exc.detail)
            raise
        progress.ok("token", f"league {league_id}")
        progress.fact(league_id=league_id)

        team_id = _team_id_from_url(body.url)
        season = _season_from_url(body.url)
        work_conn, league_conn, session = _connect_work(
            progress, league_id, team_id, season)

        def run_fn(listener, on_change, on_activity, stop_event):
            # The browser observer: watches the socket a real ESPN tab holds.
            # Still the fallback for the waiting-room URL that carries no
            # teamId, where the direct socket has no team to open with. It has
            # no per-frame hook, so on_activity is unused here; on_change still
            # stamps last_poll_at each pick.
            run_listener(listener, body.url, STATE_PATH,
                         on_change=on_change, stop_event=stop_event)

        # The saved session, if there was one, described a session this one
        # has just replaced -- and this path holds no token of its own to
        # save in its place, since it watches a socket the user's own browser
        # opened. So the record goes, rather than being left to restore a
        # session that is no longer the live one on the next restart: the
        # record must always describe the CURRENT session or nothing at all.
        clear_session_record(db_path)
        return _launch_listener(work_conn, league_conn, league_id, session,
                                run_fn, progress=progress)

    @app.post("/api/live/connect-token")
    def live_connect_token(body: TokenBody):
        """Open the draft socket directly, from a token the bookmarklet minted
        in the user's own browser -- no browser window on this machine.

        The bookmarklet runs on the ESPN draft page, makes the one call that
        needs the account session (minting the per-draft draftSecurity token,
        which never leaves that page), and delivers only the token plus the
        public ids here. `run_socket_listener` opens the socket with the SWID
        and token alone -- espn_s2 is not needed (see its docstring) -- so a
        stranger's league with no saved login on this machine still produces
        a live board. This is the whole point of the bookmarklet path, and
        the reason the browser-window fallback (`/api/live/connect`) can be
        avoided whenever the drafter can click a bookmark.
        """
        progress, _seq = _new_progress(token_path=True, league_id=body.leagueId)
        progress.begin("token")
        # NOTHING about this step reaches ESPN: the token is a per-draft nonce
        # the bookmarklet already minted on ESPN's own page, and the first
        # thing that actually puts it in front of ESPN is the socket handshake
        # at the bottom of this file. So this row says what it really is --
        # the four fields arrived and the team id is a number -- and it is the
        # SOCKET row that reports whether ESPN accepted the token.
        if not (body.leagueId and body.teamId and body.swid and body.token):
            progress.fail(
                "token", "the bookmarklet sent an incomplete token",
                hint="Open your ESPN draft room and click the Draft Helper "
                     "bookmark from inside it, not from another tab.")
            raise HTTPException(
                status_code=422,
                detail="missing leagueId, teamId, swid, or token")
        # The socket path needs a team id up front (it has no browser JOIN to
        # learn it from), and _slot_for_team compares against integer team
        # ids -- so a non-numeric teamId is a real, up-front error here rather
        # than a silent None slot later.
        try:
            team_id = int(body.teamId)
        except (TypeError, ValueError):
            progress.fail(
                "token", f"team id {body.teamId!r} is not a number",
                hint="Open your ESPN draft room and click the Draft Helper "
                     "bookmark from inside it, not from another tab.")
            raise HTTPException(status_code=422, detail="teamId must be numeric")
        progress.ok("token", f"team {team_id} · season {body.season or '?'}")

        work_conn, league_conn, session = _connect_work(
            progress, body.leagueId, team_id, body.season)

        run_fn = _socket_run_fn(body.leagueId, body.teamId, body.swid,
                                body.token, progress)

        # Record the token so /api/live/state's token_received stays truthful
        # for the connect screen. Set before launch; _launch_listener's own
        # state.update never touches "token".
        with lock:
            state["token"] = {
                "league_id": body.leagueId, "team_id": body.teamId,
                "swid": body.swid, "token": body.token, "season": body.season,
                "received_at": datetime.now(timezone.utc).isoformat(),
            }
        # And to disk, which is the whole of this session that a restart
        # cannot rebuild for itself (see save_session_record, and the long
        # note above it for exactly what that costs -- including why the
        # older comment here, "in memory only ... writing it to disk is the
        # one thing that would turn a breach into a leak", is no longer what
        # this code does). Written AFTER the build succeeded and before the
        # listener starts, so a connect that died in build_session leaves
        # whatever was already saved alone rather than replacing a working
        # session with one that has just proved unbuildable.
        #
        # Best-effort: a record that cannot be written costs the next restart
        # its automatic reconnect, and the owner can still click the
        # bookmarklet -- strictly better than failing a connect that
        # otherwise worked, on a thirty-second pick clock.
        try:
            save_session_record(
                db_path, league_id=body.leagueId, team_id=body.teamId,
                season=body.season, swid=body.swid, token=body.token,
                my_slot=session.my_slot)
        except OSError:      # noqa: BLE001 -- see above
            pass
        return _launch_listener(work_conn, league_conn, body.leagueId, session,
                                run_fn, progress=progress)

    def _restore_saved_session(record):
        """Rebuild the session the previous process was running, and reopen
        its socket, from the record on disk.

        WHEN: on a thread started at app construction, not inline and not on
        the first request that needs a session. All three were weighed and
        the reasons are worth keeping.

          * Inline at startup. build_session is 4.1-34.8s against the real
            database (see its own docstring), 27.5-30.7s of that fit_all.
            The API would answer nothing for that whole window -- not
            /api/live/state, not the connect screen, and not
            /api/live/connect-token, which is the manual way out of a
            restore that is going to fail. Blocking the one escape hatch on
            the operation most likely to need it is the wrong order.
          * Lazily, on the first request that needs a session. The room
            polls /api/live/state every 2.5s (web/src/pages/DraftRoom.tsx),
            so the rebuild would land inside one poll and pile ~14 more
            behind it, with nothing anywhere able to say what is being
            waited for.
          * This. The API is up immediately; /api/live/state answers
            `active: false, restoring: true` while the rebuild runs, and the
            room's existing poll loop picks the session up the moment it
            lands -- it already treats active:false as a transient state and
            recovers on its own, so no client change is needed for this to
            work (only to say something nicer than "Not connected" during
            it).

        The cost, stated rather than hidden: for those seconds the room DOES
        look disconnected. `restoring` is the whole answer to that, and it
        is why the key exists.

        WHAT IT TRUSTS FROM THE RECORD: the four connect fields and the
        season, which cannot be rebuilt from anything (see the note above
        save_session_record). Not the board, not the pool, not the settings,
        not the manager fits -- _connect_work rebuilds all of those from the
        league's own database and re-fetches ESPN's settings and team names
        exactly as a click would, so a restored session is not a stale
        snapshot of the old one, it is the same session built again. The
        saved slot is used ONLY as a fallback (see below).

        FAILURE IS THE EXPECTED CASE, not the edge case. ESPN's
        draftSecurity token is minted per draft and dies with it, so a
        restore against an expired token is the normal outcome of restarting
        after the draft finished, or of a record whose draft has since been
        cancelled. It is bounded and loud already:
        run_socket_listener gives up after MAX_EMPTY_RECONNECTS (5)
        frameless attempts at RECONNECT_BACKOFF_SECONDS (2s) apart and
        raises "the draft token may have expired; click the Draft Helper
        bookmark again to mint a fresh one", which pump() records as
        listener_error and fails the `socket` progress stage with the same
        hint a live connect would give. So the failure surfaces on
        /api/live/state within ~10s, with the exact sentence that tells the
        owner what to do, and never as a silent dead session or a loop.
        """
        league_id = record["league_id"]
        with lock:
            # Captured before any work, and re-checked at the instant this
            # registers (see _launch_listener's guard_gen): /api/live/stop
            # bumps this and nothing else, so it is the only way to notice
            # that the user stopped the draft while the rebuild was running.
            generation = state["generation"]
        progress, seq = _new_progress(token_path=True, league_id=league_id)
        progress.begin("token")
        # Read off disk, not off ESPN -- the same thing live_connect_token's
        # own token row means, and equally not a statement that the token is
        # still good. It is the `socket` row that reports whether ESPN
        # accepted it, here exactly as there.
        progress.ok("token", f"restored · team {record['team_id']} · "
                             f"season {record['season'] or '?'}")
        progress.fact(restored=True)
        try:
            work_conn, league_conn, session = _connect_work(
                progress, league_id, int(record["team_id"]), record["season"])
        except Exception as exc:      # noqa: BLE001 -- this is a bare daemon
            # thread with no request to raise into, and a restore that dies
            # in the board build has no listener for listener_error to hang
            # off. Recorded so the failure is visible on /api/live/state
            # instead of being a process that came up saying "no draft".
            # _connect_work has already marked the stage it died on.
            with lock:
                if state["connect_seq"] == seq:
                    state["restore_error"] = f"{type(exc).__name__}: {exc}"
            return
        # The saved slot, and ONLY when this connect could not work one out
        # for itself. Precedence deliberately that way round: ESPN's own
        # pickOrder and the league's draft_order are current, and a draft
        # order re-randomised since the record was written must win over it.
        # The saved value is what covers the case nothing else can -- a mock
        # draft, where the slot was originally read off the socket's round-1
        # ordering (_slot_from_socket) and there are no frames to read it
        # from again when the token is dead.
        if session.my_slot is None and record.get("my_slot") is not None:
            session = dataclasses.replace(session,
                                          my_slot=int(record["my_slot"]))
            progress.fact(my_slot=int(session.my_slot))
        with lock:
            # Same in-memory token record a live connect keeps, so
            # /api/live/state's token_received is true for a restored
            # session too -- the landing page reads it to offer the board
            # rather than the install guide (web/src/pages/Landing.tsx).
            state["token"] = {
                "league_id": league_id, "team_id": record["team_id"],
                "swid": record["swid"], "token": record["token"],
                "season": record["season"],
                "received_at": record.get("saved_at"),
            }
        _launch_listener(work_conn, league_conn, league_id, session,
                         _socket_run_fn(league_id, record["team_id"],
                                        record["swid"], record["token"],
                                        progress),
                         progress=progress, guard_seq=seq,
                         guard_gen=generation)

    # The restart-resilience entry point, and the only thing in this file
    # that runs without a request behind it. Nothing happens at all unless a
    # record is actually on disk, which is true only between a bookmarklet
    # connect and the end of that draft -- so every ordinary start, and every
    # test that builds an app against a scratch database, skips this in one
    # stat(). The thread is a daemon and is never joined by the app: create_app
    # must return, and uvicorn must bind its port, without waiting on a
    # 4.1-34.8s board build.
    _saved = load_session_record(db_path)
    if _saved is not None:
        state["restore_thread"] = threading.Thread(
            target=_restore_saved_session, args=(_saved,), daemon=True)
        state["restore_thread"].start()

    return state, _recompute
