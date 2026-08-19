"""Live draft mode: one long-lived session, refreshed as picks land.

`make sim` pays 0.9s building the board, 1.1s building the pool and 14.9s in
`fit_all` on every invocation. During a draft none of that changes -- the
coefficients come from history, the board and pool are static -- so the
session builds them once and every refresh costs only `survival` plus
`rank_available` (see SURVIVAL_ROLLOUTS below for why that is cheap).
"""
import dataclasses
import hashlib
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
                  league_id: str = "", settings=None) -> DraftSession:
    """Everything expensive, once. Roughly 17s against a real database.

    `league_id` is passed in rather than looked up, because the database has
    no record of it: the `league` table is `(season, settings_json)` and no
    table anywhere carries a league id. An earlier version read
    `league["league_id"]` and raised KeyError the first time a real connect
    ran. The caller already has it -- parsed from the URL the user pasted,
    which is the only place it exists.
    """
    # `settings` is passed in when the connect flow fetched the league's real
    # roster/scoring from ESPN (see _league_settings_from_espn); None falls
    # back to whatever the database holds (the owner's imported league, or the
    # cold-start default). Everything downstream -- the board, the pool, the
    # round count, the roster the simulator drafts FOR -- keys off it.
    if settings is None:
        settings = league_mod.load(conn)
    board = build_board(conn, settings=settings)
    board = _attach_espn_proj(conn, board)
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
        started_at=datetime.now(timezone.utc), board=board,
        board_by_id={str(r["player_id"]): r
                     for r in board.to_dict(orient="records")})


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
from scoring.draft_sim import (_drafted_state, _seed_rosters, snake_slots,
                               survival)
from scoring.gain import rank_available


class ConnectBody(BaseModel):
    # No my_slot field. A URL carrying teamId= resolves it immediately (see
    # _team_id_from_url / _slot_for_team); one that doesn't (the natural
    # waiting-room URL to paste) leaves it None until the socket's TOKEN
    # frame names our team (see live_connect's on_change) -- there is no
    # third case left for a human to fill in by hand.
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

# survival() runs ONE set of rollouts that stop at my next turn, not one full
# draft per candidate, so the old clock-rationed budget (12/25/40, see
# rollouts_for below) is no longer the constraint it was priced against.
# This is the whole recompute cost now, and it buys a materially tighter
# survival estimate for a fraction of what search_pick cost: measured
# against the real production pool (data/nfl.duckdb, 249 players, 8 teams),
# survival(n_rollouts=400) plus rank_available together ran in 0.1-0.8s
# across picks_made 0/8/50/100 -- an order of magnitude under even the
# cheapest old ROLLOUTS_FAR budget's ~2.3s (12 rollouts * ~0.19s), let alone
# the 30-90s pick clock this has to fit inside.
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
    """Mount live-draft endpoints. In-process state only, same lifetime as
    `create_app`'s connection -- a restart mid-draft means starting again,
    which is correct: the cached pool would be stale anyway.

    `db_path` is the app's own database file -- the same one `conn` was
    opened against in `create_app`. It is passed (rather than derived from
    `conn`) so `live_connect` can hand it to `provision_league` as the
    `universal_path` a new league's file is seeded from; `conn` itself stays
    the connection the default league's session builds against, exactly as
    before Task 5.
    """
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
             "league_conn": None}
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
            # survival()'s avail_pct is already a 0-1 probability (see its
            # docstring and the "counts / max(n_rollouts, 1)" line it
            # returns) -- rank_available wants exactly that, no rescaling.
            avail = survival(
                session.pool, session.settings, session.slot_managers,
                session.my_slot, taken, session.betas,
                n_rollouts=SURVIVAL_ROLLOUTS, seed=session.seed,
                taken_order=taken_order,
                on_the_clock=on_the_clock)["avail_pct"].to_numpy()
            frame = rank_available(session.pool, session.settings, taken,
                                   counts, avail)
        finally:
            cur.close()
        with lock:
            if state["generation"] != generation:
                return          # session stopped/restarted while computing
            if state["as_of_pick"] is not None and state["as_of_pick"] > picks_made:
                return          # superseded while we were computing
            state["candidates"] = frame.to_dict(orient="records")
            state["as_of_pick"] = picks_made

    def _provision_and_build(league_id, team_id, settings=None):
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
        league_conn = None
        if (league_id and league_id != DEFAULT_LEAGUE
                and league_id != leagues_mod.DEFAULT_LEAGUE_ID):
            league_path = provision_league(
                league_id, universal_path=db_path, root=leagues_mod.LEAGUES_ROOT)
            league_conn = get_conn(league_path)
        work_conn = league_conn if league_conn is not None else conn
        try:
            cur = work_conn.cursor()
            try:
                # The socket speaks team ids, not slots (see _slot_for_team's
                # docstring). A resolvable teamId fixes my_slot immediately;
                # otherwise it stays None and the socket's TOKEN frame names
                # our team once it connects (see the my_slot back-fill in
                # _launch_listener's on_change). Read off `work_conn`, the
                # same connection build_session uses just below, so a resolved
                # slot cannot come from a different league's draft order.
                my_slot = (_slot_for_team(cur, team_id)
                           if team_id is not None else None)
                session = build_session(cur, my_slot, league_id=league_id,
                                        settings=settings)
            finally:
                cur.close()
        except Exception:
            if league_conn is not None:
                league_conn.close()
            raise
        return work_conn, league_conn, session

    def _launch_listener(work_conn, league_conn, league_id, session, run_fn):
        """Register a built session's listener thread and start it.

        `run_fn(listener, on_change, stop_event)` is what actually opens and
        pumps frames -- `run_listener` (browser observer) or
        `run_socket_listener` (direct socket). Everything around it is
        identical for both, so it lives here once: the pick pump, the my_slot
        back-fill, the identity guard that keeps a superseded listener's
        in-flight callback a no-op, the listener_error capture, and the state
        registration + generation bump.
        """
        listener = DraftListener(session.crosswalk)
        stop_event = threading.Event()
        # DraftSession is frozen, so the my_slot back-fill replaces the
        # session object rather than mutating it. `current` is that one
        # mutable cell, closed over by the callbacks one-to-one with `listener`.
        current = {"session": session}

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
                else:
                    with lock:
                        if state["listener"] is listener:
                            state["recompute_error"] = None

        def _resolve_slot(c2) -> bool:
            """Resolve my_slot from history or the socket if not yet known;
            return True if it just resolved.

            Runs from on_activity (every frame) as well as on_change (every
            pick), because the moment that matters most -- our own team coming
            ON the clock -- arrives as a SELECTING frame, which changes no pick
            count and so never reaches on_change. History (_slot_for_team)
            first; the socket's round-1 ordering (_slot_from_socket) is the
            fallback for a mock or an un-imported league, where no history can
            translate the team id and the slot would otherwise stay unknown --
            leaving every recommendation blank, including for our first pick.
            """
            sess = current["session"]
            if sess.my_slot is not None or listener.my_team_id is None:
                return False
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
                                "SELECT count(*) FROM drafted").fetchone()[0]
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
                    made = c2.execute(
                        "SELECT count(*) FROM drafted").fetchone()[0]
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
                    if state["listener"] is listener:
                        state["unmapped"] = live.unmapped
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

        thread = threading.Thread(target=pump, daemon=True)
        recompute_thread = threading.Thread(target=recompute_worker, daemon=True)
        with lock:
            state.update({"session": session, "listener": listener,
                          "listener_thread": thread, "listener_stop": stop_event,
                          "recompute_thread": recompute_thread,
                          "listener_error": None, "recompute_error": None,
                          "league_conn": league_conn,
                          "candidates": [], "as_of_pick": None,
                          "unmapped": [], "last_poll_at": None})
            state["generation"] = state.get("generation", 0) + 1
        recompute_thread.start()
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
                          "as_of_pick": None, "unmapped": [],
                          "last_poll_at": None, "recompute_error": None})
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
                        "listener_alive": False, "recompute_error": None,
                        # Same "present with a null/false value, never
                        # omitted" convention listener_alive already
                        # follows on this branch: the room reads it to
                        # decide whether the draft buttons are live, and an
                        # absent key would read as undefined -- falsy by
                        # luck rather than by contract.
                        "socket_alive": False,
                        "token_received": state.get("token") is not None,
                        "ms_remaining": None,
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
                picks_made = cur.execute("SELECT count(*) FROM drafted").fetchone()[0]
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
                if session.my_slot is not None:
                    try:
                        _, taken_order = _drafted_state(cur, session.pool)
                    except ValueError:
                        taken_order = None
                    if taken_order is not None:
                        my_roster = _my_roster(session, taken_order)
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
            # The recompute worker fails independently of the listener, and
            # its failure is quieter: the clock keeps ticking, the board
            # keeps filling, and only the ranking stops. Reported separately
            # for that reason -- see the state key's own comment.
            "recompute_error": snapshot["recompute_error"],
            "socket_alive": socket_alive,
            "token_received": snapshot.get("token") is not None,
            "ms_remaining": ms_remaining,
            "settings": _league_settings_payload(session.settings),
            "my_roster": my_roster,
        }

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
        picks_made = len(picks)
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
                picks_made = cur.execute(
                    "SELECT count(*) FROM drafted").fetchone()[0]
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
                          "as_of_pick": None, "unmapped": [],
                          "last_poll_at": None, "listener": None,
                          "listener_thread": None, "listener_stop": None,
                          "recompute_thread": None, "listener_error": None,
                          "recompute_error": None})
        return {"active": False, "listener_stopped": stopped}

    @app.post("/api/live/connect")
    def live_connect(body: ConnectBody):
        # Validate the one thing that can be invalid (the league id) BEFORE
        # tearing down a working listener -- an invalid request must never
        # stop one that was running. Team id / slot resolution never raises
        # (see _slot_for_team's docstring).
        league_id = _resolve_league_id(body.url)

        # Exactly one listener at a time. Rather than refuse a reconnect --
        # which would trap a caller recovering from a dead listener behind a
        # separate, easy-to-forget /api/live/stop -- the old one is always
        # stopped and joined FIRST. That is also where its per-league
        # connection is closed, so the single-writer DuckDB file is free
        # before _provision_and_build reopens it. If it will not stop in
        # time, refuse rather than race it.
        if not _stop_listener():
            raise HTTPException(
                status_code=503,
                detail="the previous listener did not stop in time -- try again")

        team_id = _team_id_from_url(body.url)
        season = _season_from_url(body.url)
        # The league's real roster/scoring from ESPN, so the session drafts for
        # the actual roster (rounds, starters) rather than the cold-start
        # default. None on any failure -> build_session reads the db's own.
        espn_settings = _league_settings_from_espn(league_id, season)
        work_conn, league_conn, session = _provision_and_build(
            league_id, team_id, settings=espn_settings)
        # Real ESPN team names for the board's columns, fetched once here (the
        # URL carries the season). Best-effort -- a failure leaves team_slots
        # empty and the board falls back to "Team {slot}".
        session = _attach_team_slots(session, league_id, season)

        def run_fn(listener, on_change, on_activity, stop_event):
            # The browser observer: watches the socket a real ESPN tab holds.
            # Still the fallback for the waiting-room URL that carries no
            # teamId, where the direct socket has no team to open with. It has
            # no per-frame hook, so on_activity is unused here; on_change still
            # stamps last_poll_at each pick.
            run_listener(listener, body.url, STATE_PATH,
                         on_change=on_change, stop_event=stop_event)

        return _launch_listener(work_conn, league_conn, league_id, session, run_fn)

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
        if not (body.leagueId and body.teamId and body.swid and body.token):
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
            raise HTTPException(status_code=422, detail="teamId must be numeric")

        # Same one-listener-at-a-time teardown as live_connect, and for the
        # same reason: stop and join the old one (closing its connection)
        # before _provision_and_build reopens this league's single-writer file.
        if not _stop_listener():
            raise HTTPException(
                status_code=503,
                detail="the previous listener did not stop in time -- try again")

        # The league's real roster/scoring from ESPN (TokenBody carries the
        # season). None on failure -> the db's own settings.
        espn_settings = _league_settings_from_espn(body.leagueId, body.season)
        work_conn, league_conn, session = _provision_and_build(
            body.leagueId, team_id, settings=espn_settings)
        # Real ESPN team names for the board's columns. TokenBody carries the
        # season outright. Best-effort, same as the browser path.
        session = _attach_team_slots(session, body.leagueId, body.season)

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

            run_socket_listener(listener, body.leagueId, body.teamId, body.swid,
                                body.token, on_change=on_change,
                                stop_event=stop_event, on_activity=on_activity,
                                on_socket=_on_socket)

        # Record the token so /api/live/state's token_received stays truthful
        # for the connect screen. In memory only: a draft token is a
        # short-lived nonce, and writing it to disk is the one thing that
        # would turn a breach into a leak. Set before launch;
        # _launch_listener's own state.update never touches "token".
        with lock:
            state["token"] = {
                "league_id": body.leagueId, "team_id": body.teamId,
                "swid": body.swid, "token": body.token, "season": body.season,
                "received_at": datetime.now(timezone.utc).isoformat(),
            }
        return _launch_listener(work_conn, league_conn, body.leagueId, session,
                                run_fn)

    return state, _recompute
