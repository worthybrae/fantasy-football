"""Join real ESPN mock drafts, play them, and write them into the corpus.

WHAT THIS IS FOR. `pipeline.mock_backfill` harvested the 23 completed mocks
that happened to be sitting in `data/leagues/`; that pile does not grow by
itself. This module is the thing that makes it grow: it watches ESPN's mock
lobby, takes a seat in an 8-team PPR snake room, plays all 16 rounds, and
records the finished draft through `pipeline.draft_log.record` with
`source=SOURCE_MOCK` -- the same corpus, the same shape, so a later refit
reads one table and cannot tell which writer produced a row except by the
two fields this writer knows and the backfill does not (`my_slot` and
per-pick `autodrafted`).

WHAT IT DOES NOT REINVENT. The transport is finished and lives elsewhere:
`pipeline.draft_socket` mints the token, builds the socket URL, reconnects
when ESPN drops the connection and publishes a `SocketHandle` to send on;
`pipeline.draft_listener.DraftListener` parses SELECTED / SELECTING / TOKEN /
AUTODRAFT frames. `pipeline.espn_mock_lobby` finds a room and takes the seat.
What is written here is the loop AROUND those: whose turn is it, what do we
pick, did ESPN accept it, and is the draft over.

THE PICK DISCIPLINE IS COPIED FROM `/api/live/select`, DELIBERATELY. A pick
is not made when we send `SELECT <espnId>`; it is made when ESPN echoes
`SELECTED`. Nothing here writes an optimistic pick into any state of its own.
The reasons are the same as that endpoint's and they bite harder unattended:
a SELECT can lose a race to another team, or name a player ESPN thinks is
already gone, and a bot that believed its own send would draft on from a
roster that never existed, misattribute every later pick, and record the lot.
So the send is followed by a wait on `listener.selected_espn_ids`, and a send
that is not confirmed inside `SELECT_CONFIRM_SECONDS` is retried with the
NEXT candidate rather than assumed.

TEAM COUNT AND ROUND COUNT ARE READ, NEVER ASSUMED. The room-selection policy
filters to `leagueSize == 8`, so in practice every room played is 8x16 -- but
the shape is still taken from the room itself: `leagueSize` off the directory
row, and the roster (and therefore `LeagueSettings.rounds`) out of the
league's own `?view=mSettings`. A wrong team count does not fail, it shifts
the snake: every pick lands on the wrong slot, every owner_key names the
wrong seat, and the corpus is poisoned in a way nothing downstream can
detect. `slot_team_map` below turns that from an unverifiable assumption into
a checked fact -- ESPN's own SELECTED frames carry the team id, so if the
assumed snake were wrong, one slot would show two different teams -- and a
draft that fails the check is NOT recorded.

THE POLICY IS EPSILON-GREEDY, epsilon = 0.2. Four fifths of our picks are
`draft_sim._greedy_choice`, the same one-ply opportunity-cost policy the
simulator gives itself in a rollout; one fifth is a uniformly random
cap-legal player. The exploration is not decoration: a bot that always played
its own top candidate would write a corpus in which our seat's picks are a
deterministic function of the board, which teaches a model fitted on that
corpus nothing except our own ranking read back to us. Note that our own
seat's picks are a BOT's, not a person's -- `my_slot` is recorded on the
draft head precisely so a later fit can exclude them.

WHY NOT THE FULL SEARCH. `search_pick` is the tool's real recommendation and
it costs seconds per pick (measured ~0.19s/rollout in api/live.py's own
comment). A mock's pick clock is short and unforgiving -- miss it and ESPN
puts the seat on autodraft -- and this process is unattended, so the cheap
deterministic policy is the one that reliably gets a pick in on time.
`_greedy_choice` is the tool's own choice at one ply, which is what "the
tool's own top candidate" means here.

BEING A DECENT LOBBY CITIZEN. Rooms are ranked by `teamsJoined` descending
(see `espn_mock_lobby`), so the seat taken is one that helps a room of real
people fill rather than one squatted in an empty room. The socket is always
closed and the listener thread always stopped on the way out -- normal
finish, exception, or Ctrl-C alike -- so a crash does not leave a process
holding a connection to a room it is no longer playing.
"""
import json
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from pipeline import draft_log as dl
from pipeline import espn_mock_lobby as lobby
from pipeline.draft_listener import DraftListener
from pipeline.draft_socket import (draft_security_token, http_fetch,
                                   load_cookies, run_socket_listener)
from pipeline.espn_live import build_crosswalk
from pipeline.espn_teams import fetch_league_settings
from scoring import league
from scoring.board import build_board
from scoring.config import CURRENT_SEASON
from scoring.draft_sim import (_greedy_choice, _legal_mask, _opportunity_gap,
                               _roster_cap, _seed_rosters, _turns_left,
                               build_pool, snake_slots)

# The exploration rate, fixed by the task rather than tuned. See the module
# docstring: a purely greedy bot writes a corpus that only reflects our own
# ranking function.
EPSILON = 0.2

# How often the play loop wakes to re-read the listener's accumulated state.
# Frames arrive pushed, so this is not polling ESPN -- it is polling an
# in-process list -- and a quarter second is well inside any pick clock while
# costing nothing.
LOOP_POLL_SECONDS = 0.25
# How long one SELECT gets to be echoed back as SELECTED before we give up on
# that candidate and try the next one. Generous next to a mock's clock (30s
# observed) because the cost of retrying too early is sending a second pick
# into a room that was about to confirm the first.
SELECT_CONFIRM_SECONDS = 8.0
# The whole turn's budget across all candidate retries. Past this the clock
# has almost certainly run out and ESPN has autodrafted for us, which is a
# recorded fact (`autodrafted`) rather than an error.
TURN_BUDGET_SECONDS = 25.0
# Give up on a room that stops producing picks. A live draft moves at worst
# one pick per clock expiry; several minutes of nothing means the room died,
# the socket is wedged, or the draft was abandoned by ESPN. Long enough to
# ride out a reconnect (draft_socket backs off ~2s and ESPN replays on JOIN),
# short enough that a dead room does not eat the night.
IDLE_TIMEOUT_SECONDS = 300.0
# ESPN's draft socket does NOT accept a JOIN for a room whose draft has not
# opened yet: connecting ~6 minutes ahead of `draftDate` was answered with
# "server rejected WebSocket connection: HTTP 500", twice, against two
# different rooms we had already successfully joined and read settings for.
# The directory's own `draftAvailableDate` (~90s before `draftDate`) is ESPN
# saying when the room opens, so the loop waits for it rather than treating
# the 500 as a bad token. This margin is added on top, because the two clocks
# are ESPN's and ours and there is no reason to test how well they agree.
AVAILABLE_MARGIN_SECONDS = 5.0
# Fallback lead when a directory row carries no `draftAvailableDate`: the
# measured gap between the two dates in every row observed.
AVAILABLE_LEAD_SECONDS = 90.0
# How many times to try opening the socket, and how long to wait between
# tries. More than one because a room that has just opened may still be
# spinning up on ESPN's side, and abandoning a seat we already hold over a
# single 500 wastes both the seat and the wait for it.
SOCKET_ATTEMPTS = 4
SOCKET_RETRY_SECONDS = 20.0
# How long one attempt gets to produce a connected socket. draft_socket's own
# reconnect loop gives up after MAX_EMPTY_RECONNECTS frameless attempts,
# which is a handful of seconds, so this only has to outlast that.
SOCKET_OPEN_SECONDS = 30.0
# How long to wait for the socket to name our team (the TOKEN frame) before
# concluding the connection is not really working. TOKEN lands about eight
# frames into a session (see draft_listener's own note), so this is dozens of
# times the observed latency.
TOKEN_TIMEOUT_SECONDS = 90.0
# How long a room may sit between "we joined" and "the first pick" before we
# walk away. A room is picked with at most MAX_LEAD_SECONDS of lead, and ESPN
# holds the lobby countdown itself, so this only fires when a room never
# starts at all.
START_TIMEOUT_SECONDS = 1800.0
# Between drafts: long enough that the lobby directory has refreshed and we
# are not hammering it, short enough not to miss the next batch of rooms
# (measured: a new 8-team PPR room every ~5 minutes).
BETWEEN_DRAFTS_SECONDS = 20.0
# How long to wait between lobby polls when nothing in the lobby fits.
LOBBY_RETRY_SECONDS = 45.0
# The corpus is a single-writer DuckDB file that other tools in this repo
# (mock_backfill, draft_log's own CLI) also write. This farm therefore opens
# it only around the one INSERT at the end of a draft rather than holding the
# write lock for the whole night -- but that INSERT still has to survive
# arriving while somebody else holds it, because losing a draft that took
# forty minutes to play over a two-second lock is not an acceptable failure.
CORPUS_LOCK_WAIT_SECONDS = 180.0
CORPUS_LOCK_RETRY_SECONDS = 5.0


class PickFrame(NamedTuple):
    """One confirmed pick, as ESPN's own frames describe it.

    `autodrafted` is None, not False, when ESPN has never sent an AUTODRAFT
    frame for that team -- the same three-valued honesty
    `DraftListener.my_autodraft` keeps, and the same reason: "ESPN has not
    said" and "ESPN said no" are different facts and the corpus stores which
    one it has.
    """
    pick_no: int
    team_id: int | None
    espn_id: int | None
    autodrafted: bool | None


def draft_timeline(events) -> list:
    """Fold a socket event stream into one `PickFrame` per pick, in order.

    THE PICK NUMBERING HERE MUST MATCH `espn_live.picks_from_events` EXACTLY,
    because the recorded `pick_no` is what attributes a pick to a slot and
    everything else in this file is keyed off it. So the same three rules are
    reproduced deliberately rather than approximated:

      - Every SELECTED frame consumes a pick number, including one whose id
        argument is missing or unparseable. Skipping it would shift every
        later pick down by one.
      - A repeated ESPN player id is a REPLAY, not a second pick: ESPN
        re-sends the whole draft on every (re)JOIN, and this loop reconnects
        whenever ESPN drops the socket. Deduped on first occurrence, before
        a pick number is consumed.
      - A frame with no parseable id cannot be deduped (there is nothing to
        key on) and still counts as its own pick.

    `tests/test_mock_farm.py` pins that agreement against the real recorded
    draft rather than trusting this comment.

    The autodraft flag is carried forward from the AUTODRAFT frames seen so
    far, which is what makes it a PER-PICK fact rather than a snapshot of the
    room's current state. ESPN sends `AUTODRAFT <team> true` the moment a
    team's clock expires and the auto-pick lands immediately after (verified
    in data/draft_room_trace.jsonl: line 1112 is the flag, 1113 the pick), so
    reading the flag at the moment the SELECTED arrives attributes it to the
    right pick.

    One honest limit: picks that were already made before this process's
    FIRST connect arrive as a replay, and ESPN replays the autodraft state as
    it stands now rather than as it stood at each of those picks. The farm
    always joins before the draft starts, so that case does not arise in
    practice -- but it is why this is described as ESPN's own frames rather
    than as a guarantee.
    """
    autodraft: dict = {}
    seen: set = set()
    out: list = []
    pick_no = 0
    for event in events:
        if event is None:
            continue
        if event.verb == "AUTODRAFT" and len(event.args) > 1:
            team = _as_int(event.args[0])
            flag = _autodraft_flag(event.args[1])
            if team is not None and flag is not None:
                autodraft[team] = flag
            continue
        if event.verb != "SELECTED":
            continue
        espn_id = _as_int(event.args[1]) if len(event.args) >= 2 else None
        if espn_id is not None and espn_id in seen:
            continue                      # replayed pick -- already counted
        pick_no += 1
        if espn_id is not None:
            seen.add(espn_id)
        team = _as_int(event.args[0]) if event.args else None
        out.append(PickFrame(pick_no, team, espn_id, autodraft.get(team)))
    return out


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _autodraft_flag(arg):
    """`true`/`false`, or None for anything else -- the same rule (and the
    same reason for it) as `draft_listener._autodraft_flag`: an unrecognised
    value must not read as "autodraft is off"."""
    text = (arg or "").strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def slot_team_map(timeline, slots) -> dict:
    """slot -> ESPN team id, built from the picks themselves.

    THIS IS THE TEAM-COUNT CHECK, and it is the reason a wrong `leagueSize`
    cannot silently poison the corpus. `slots` is the snake this run assumed;
    each pick carries the team ESPN says made it. In a real snake every pick
    at a given slot belongs to the same team, so if the assumed team count
    were wrong the mapping would collide almost immediately.

    Raises on a collision rather than returning a best guess. The caller's
    answer is to refuse to record the draft: a draft attributed to the wrong
    seats is worse than no draft, because it is indistinguishable from a good
    one once it is in the table.
    """
    mapping: dict = {}
    for frame in timeline:
        if frame.team_id is None or frame.pick_no > len(slots):
            continue
        slot = slots[frame.pick_no - 1]
        known = mapping.get(slot)
        if known is None:
            mapping[slot] = frame.team_id
        elif known != frame.team_id:
            raise ValueError(
                f"slot {slot} was drafted by both team {known} and team "
                f"{frame.team_id} -- the assumed draft shape "
                f"({len(set(slots))} teams x {len(slots) // max(len(set(slots)), 1)} "
                "rounds) does not match the room ESPN actually ran, so every "
                "slot attribution in this draft is suspect")
    return mapping


def slot_for_team(timeline, slots, team_id, on_the_clock=None):
    """Our draft slot, from the socket alone.

    Same argument as `api/live._slot_from_socket`: in a snake the overall
    pick order IS the slot order, so the slot of any pick our team has
    already made names our slot, and before we have picked, being on the
    clock names the slot of the pick about to happen. Generalised past round
    1 only because `slots` here is the full snake rather than the first round
    -- the first occurrence is still what answers.

    None until one of those two is true, which is honest rather than
    unhelpful: a guessed slot attributes every pick of the draft to the wrong
    seat.
    """
    if team_id is None:
        return None
    for frame in timeline:
        if frame.team_id == team_id and frame.pick_no <= len(slots):
            return slots[frame.pick_no - 1]
    if on_the_clock == team_id and len(timeline) < len(slots):
        return slots[len(timeline)]
    return None


def slot_from_pick_order(settings, team_id):
    """Our slot out of ESPN's own `draftSettings.pickOrder`, or None.

    CALLER'S PRECONDITION, identical to `api/live._slot_from_pick_order`'s:
    `settings` must be the LeagueSettings from THIS room's live ESPN fetch. A
    pick order from anywhere else (a saved league row, a previous season)
    indexes happily and answers wrongly, because ESPN team ids are stable
    across seasons. Every call in this module passes the value
    `fetch_league_settings` returned moments earlier for this exact room, so
    the precondition holds by construction.

    Used only as the EARLY answer -- before any pick has happened -- and
    superseded by `slot_for_team` the moment the socket can answer for
    itself, because the socket's answer is derived from picks that actually
    landed rather than from a field ESPN populates before the room fills.
    """
    order = list(getattr(settings, "pick_order", ()) or ())
    if team_id is None or team_id not in order:
        return None
    return order.index(team_id) + 1


def choose_index(pool, available, roster, settings, caps, gap, turns_left,
                 rng, epsilon: float = EPSILON):
    """Epsilon-greedy: our own top candidate, or a uniformly random legal one.

    Returns a pool index, or None when there is nothing left to take.

    THE COIN IS FLIPPED UNCONDITIONALLY, before either branch runs, and that
    is not incidental. `rng` is a seeded `numpy.random.Generator`; drawing a
    variable number of values per pick would make the whole draft's random
    stream depend on which branch each earlier pick took, so a run could not
    be reproduced from its seed and the test below could not pin both
    branches from one generator.

    The random branch is uniform over CAP-LEGAL players only (`_legal_mask`,
    the same mask the simulator's own policy filters with), not over the
    whole board: drafting a fourth quarterback is not exploration, it is a
    roster the league rules do not allow, and it would teach a model fitted
    on this corpus that such rosters happen. Only if the caps leave nothing
    at all does it fall back to the unmasked set -- the same
    genuinely-unavoidable case `_greedy_choice` documents for itself.

    `available` must already be restricted to players we can actually SELECT
    (see `Tool.sendable`). A candidate with no ESPN id could be chosen and
    then not sent, which would burn the clock and hand the pick to ESPN's
    autodrafter.
    """
    roll = float(rng.random())
    if len(available) == 0:
        return None
    if roll < epsilon:
        legal = available[_legal_mask(pool, available, roster["counts"], caps)]
        if len(legal) == 0:
            legal = available
        return int(legal[int(rng.integers(len(legal)))])
    idx = _greedy_choice(pool, available, roster, settings, caps, gap=gap,
                         turns_left=turns_left)
    return None if idx is None else int(idx)


@dataclass
class Tool:
    """Everything the policy needs about the board, built once per draft.

    `sendable` is the part that is easy to forget and expensive to get wrong:
    a pool row whose player has no ESPN id (a crosswalk gap) can be ranked
    but cannot be picked, because a SELECT frame is addressed by ESPN's id.
    Candidates are filtered by it before the policy ever sees them.
    """
    board: pd.DataFrame
    pool: object
    pool_df: pd.DataFrame
    crosswalk: dict                 # espn id -> board player_id
    espn_by_index: list             # pool index -> espn id (or None)
    index_by_player: dict           # board player_id -> pool index
    sendable: np.ndarray            # pool-aligned bool


def build_tool(conn, settings) -> Tool:
    """Board, pool and crosswalk for one room's settings.

    Rebuilt per draft rather than once per process, because it is a function
    of the room's own settings -- roster shape decides replacement level,
    which decides every VOR the policy compares on. Measured elsewhere in
    this codebase at 1.5-1.9s (board) plus 2.2-3.6s (pool); against a draft
    that runs 20-40 minutes that is not worth caching around.
    """
    board = build_board(conn, settings=settings)
    pool = build_pool(conn, board, settings)
    crosswalk = build_crosswalk(board)

    # espn id -> player_id inverted to player_id -> espn id. First wins: two
    # ESPN ids mapping to one board player would be a crosswalk defect, and
    # picking either of them selects the same person, so there is nothing to
    # decide between them.
    espn_by_player: dict = {}
    for espn_id, player_id in crosswalk.items():
        espn_by_player.setdefault(str(player_id), int(espn_id))

    index_by_player = {str(pid): i for i, pid in enumerate(pool.player_id)}
    espn_by_index = [espn_by_player.get(str(pid)) for pid in pool.player_id]
    sendable = np.array([e is not None for e in espn_by_index], dtype=bool)

    # Same columns, same source, same order as `mock_backfill._pool_frame` --
    # the two writers' pool snapshots have to be comparable or a reader
    # joining picks to pools would get different answers depending on which
    # module recorded the draft. `team` is not a SimPool field, so it comes
    # off the board by player_id exactly as that function does it.
    pool_df = pd.DataFrame({
        "player_id": pool.player_id,
        "position": pool.position,
        "adp_rank": pool.adp_rank,
        "proj_points": pool.points,
    })
    team_by_player = (board.drop_duplicates("player_id", keep="first")
                      .set_index("player_id")["team"])
    pool_df["team"] = pool_df["player_id"].map(team_by_player)

    return Tool(board=board, pool=pool, pool_df=pool_df, crosswalk=crosswalk,
                espn_by_index=espn_by_index, index_by_player=index_by_player,
                sendable=sendable)


def taken_order_from(timeline, tool) -> list:
    """One entry per pick made so far: the pool index taken, or None.

    None for a pick whose player the crosswalk could not resolve. That is the
    shape `_seed_rosters` documents and it matters: a pick nobody can name
    still CONSUMED a turn, so it has to occupy its position in the list or
    every later pick is handed to the wrong slot.
    """
    out = []
    for frame in timeline:
        player_id = (tool.crosswalk.get(frame.espn_id)
                     if frame.espn_id is not None else None)
        out.append(tool.index_by_player.get(str(player_id))
                   if player_id is not None else None)
    return out


def build_record(timeline, tool, settings, league_id, season, my_slot,
                 started_at, teams: int, rounds: int) -> tuple:
    """The finished draft as a `DraftRecord`, plus how many picks were
    dropped for want of a pool row.

    Shaped to match `mock_backfill`'s output field for field, so the corpus
    holds one kind of mock-draft row rather than two: anonymous owner keys,
    the draft's own pool snapshot, `settings_json` carrying the roster shape
    the draft actually ran under. Two fields the backfill has to leave NULL
    are real here and are set for real -- `my_slot` (we know which seat we
    took) and per-pick `autodrafted` (the socket told us).

    OUR OWN SEAT IS STILL ANONYMOUS. It is tempting to give it a stable owner
    key, and it would be wrong: the picks at that seat are this bot's
    epsilon-greedy policy, not a person's tendencies, and an owner key is
    what `owner_profile` builds a personal profile from. `my_slot` on the
    draft head is how a reader identifies (and, for fitting human behaviour,
    excludes) those picks.

    A pick whose player has no pool row is DROPPED and COUNTED, never written
    half-filled -- the same discipline, and the same reasoning, as
    `mock_backfill._picks_frame`: a silent drop looks like a smaller corpus
    rather than like the join failure it is. `pick_no` is preserved on the
    rows that survive, so a gap stays visible as a gap.
    """
    slots = snake_slots(teams, rounds)
    draft_id = dl.draft_id_for(dl.SOURCE_MOCK, league_id, season, started_at)
    rows = []
    for frame in timeline:
        if frame.pick_no > len(slots):
            break
        slot = slots[frame.pick_no - 1]
        player_id = (tool.crosswalk.get(frame.espn_id)
                     if frame.espn_id is not None else None)
        rows.append({
            "pick_no": frame.pick_no,
            "round": (frame.pick_no - 1) // teams + 1,
            "slot": slot,
            "owner_key": dl.anonymous_key(draft_id, slot),
            "is_anonymous": True,
            "player_id": None if player_id is None else str(player_id),
            "autodrafted": frame.autodrafted,
        })
    picks = pd.DataFrame(rows, columns=[
        "pick_no", "round", "slot", "owner_key", "is_anonymous", "player_id",
        "autodrafted"])
    picks = picks.merge(
        tool.pool_df[["player_id", "position", "adp_rank", "proj_points"]],
        on="player_id", how="left")
    missing = picks["position"].isna()
    dropped = int(missing.sum())

    record = dl.DraftRecord(
        source=dl.SOURCE_MOCK, league_id=str(league_id), season=int(season),
        teams=int(teams), rounds=int(rounds), my_slot=my_slot,
        settings_json=league.to_json(settings), started_at=started_at,
        picks=picks[~missing].reset_index(drop=True), pool=tool.pool_df,
        draft_id=draft_id)
    return record, dropped


# ---------------------------------------------------------------------------
# The live half: joining a room and playing it.
# ---------------------------------------------------------------------------


class _Session:
    """The socket, the listener and the thread that feeds it, as one thing
    that can be stopped exactly once.

    One object rather than a pile of locals because the stop path has to run
    on EVERY exit -- a finished draft, a raised exception, a
    KeyboardInterrupt from the terminal -- and a bot that leaves a live
    socket attached to a room it is no longer playing is the wedged-process
    failure this loop must not have. `play_draft` calls `stop()` from a
    `finally`, `_connect_session` calls it on an attempt that never
    connected, and `stop()` is idempotent so both happening is harmless. The
    listener thread is a daemon as well, so even a stop that times out cannot
    keep the interpreter alive.
    """

    def __init__(self, listener, league_id, team_id, swid, token, out):
        self.listener = listener
        self.socket = None
        self.error = None
        self._out = out
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="mock-farm-socket", daemon=True,
            kwargs={"league_id": league_id, "team_id": team_id, "swid": swid,
                    "token": token})

    def _run(self, league_id, team_id, swid, token):
        def on_socket(handle):
            self.socket = handle
            self._ready.set()
        try:
            run_socket_listener(self.listener, league_id, team_id, swid,
                                token, stop_event=self._stop,
                                on_socket=on_socket)
        except BaseException as exc:            # noqa: BLE001 -- reported, not
            # swallowed: this thread's death is how a bad token or a room
            # that rejected us shows up, and the play loop reads `error` to
            # tell that apart from a merely quiet draft.
            self.error = exc
        finally:
            self._ready.set()

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            self._out("  warning: the socket thread did not stop in 10s "
                      "(daemon, so it dies with the process)")

    def wait_for_socket(self, seconds: float) -> bool:
        self._ready.wait(seconds)
        return self.socket is not None and self.error is None

    def alive(self) -> bool:
        return self._thread.is_alive() and self.error is None


def _wait_until_available(room, out) -> None:
    """Sleep until ESPN says the room is enterable.

    See AVAILABLE_MARGIN_SECONDS: the draft socket rejects a JOIN for a room
    that has not opened yet, with a bare HTTP 500 that reads exactly like a
    bad token. `draftAvailableDate` is the directory's own answer for when
    that stops being true; a row without one falls back to
    AVAILABLE_LEAD_SECONDS before `draftDate`, the gap measured in every row
    observed.

    Interruptible only by killing the process, which is fine: this is at most
    MAX_LEAD_SECONDS of waiting by construction, since that is the newest
    room the selection policy will pick.
    """
    available_ms = room.get("draftAvailableDate")
    if not available_ms and room.get("draftDate"):
        available_ms = float(room["draftDate"]) - AVAILABLE_LEAD_SECONDS * 1000
    if not available_ms:
        return
    seconds = (float(available_ms) / 1000.0 - time.time()
               + AVAILABLE_MARGIN_SECONDS)
    if seconds <= 0:
        return
    out(f"  room opens in {seconds:.0f}s -- waiting before connecting")
    time.sleep(seconds)


def _connect_session(listener, league_id, team_id, swid, fetch, season, out):
    """A started `_Session` whose socket is connected, or None.

    The token is minted fresh on every attempt rather than once outside the
    loop: it is a nonce for a connection that has not happened yet, and a
    retry twenty seconds later is a different connection. An attempt that
    never connects is stopped here, so a caller only ever receives a session
    it has to stop.
    """
    for attempt in range(1, SOCKET_ATTEMPTS + 1):
        token = draft_security_token(fetch, league_id, team_id, season)
        session = _Session(listener, league_id, team_id, swid, token, out)
        session.start()
        if session.wait_for_socket(SOCKET_OPEN_SECONDS):
            return session
        out(f"  socket attempt {attempt}/{SOCKET_ATTEMPTS} failed: "
            f"{session.error}")
        session.stop()
        if attempt < SOCKET_ATTEMPTS:
            time.sleep(SOCKET_RETRY_SECONDS)
    return None


def _wait_for(predicate, seconds: float, poll: float = LOOP_POLL_SECONDS):
    """Poll `predicate` until it is true or `seconds` elapse. Returns the
    last value. Polled rather than signalled for the same reason
    /api/live/select polls: the listener thread fills its state from its own
    frame loop and has nothing to notify a waiter with."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(poll)
    return predicate()


def _make_pick(session, tool, settings, slots, timeline, my_slot, caps,
               turns, rng, out) -> bool:
    """Choose and land one pick of ours. True when ESPN confirmed it.

    Candidates are tried in order until one is confirmed or the turn budget
    runs out. A SELECT that goes unanswered is nearly always a player who
    went off the board in the seconds between our snapshot and our send, so
    the answer is the next candidate, not a retry of the same one.
    """
    offset = len(timeline)
    taken_order = taken_order_from(timeline, tool)
    taken = np.zeros(len(tool.pool.player_id), dtype=bool)
    for idx in taken_order:
        if idx is not None:
            taken[idx] = True
    rosters, _recent = _seed_rosters(tool.pool, settings, taken_order)
    roster = rosters.get(my_slot, {"counts": {}, "indices": []})
    gap = _opportunity_gap(slots, offset, my_slot)
    turns_left = turns[offset] if offset < len(turns) else 1

    rejected: set = set()
    deadline = time.monotonic() + TURN_BUDGET_SECONDS
    while time.monotonic() < deadline:
        usable = np.flatnonzero(~taken)
        usable = usable[tool.sendable[usable]]
        if len(rejected):
            usable = usable[~np.isin(usable, list(rejected))]
        idx = choose_index(tool.pool, usable, roster, settings, caps, gap,
                           turns_left, rng)
        if idx is None:
            out("  no candidate left to send")
            return False
        espn_id = tool.espn_by_index[idx]
        name = str(tool.pool.player_id[idx])
        try:
            session.socket.send(f"SELECT {espn_id}\n")
        except Exception as exc:                # noqa: BLE001 -- the socket
            # dropping mid-turn is ordinary (draft_socket reconnects on its
            # own); the pick is simply not made, and the loop will see it is
            # still our turn and try again.
            out(f"  SELECT {name} could not be sent: {exc}")
            return False
        confirmed = _wait_for(
            lambda: espn_id in session.listener.selected_espn_ids,
            SELECT_CONFIRM_SECONDS)
        if confirmed:
            out(f"  pick {offset + 1}: {name} (espn {espn_id})")
            return True
        out(f"  ESPN did not confirm {name} (espn {espn_id}) -- next candidate")
        rejected.add(int(idx))
    return False


def record_draft(record, corpus_path, out) -> str:
    """Write one finished draft, waiting out anybody else holding the corpus.

    Opened here and closed immediately, rather than held open for the whole
    run: the corpus is a single-writer file that `mock_backfill` and
    `draft_log`'s own CLI also write, and a farm that squatted the write lock
    all night would make every other tool in the repo fail with a lock error.
    The cost is that this call can arrive while one of those IS holding it,
    which is what the retry is for -- forty minutes of drafting must not be
    thrown away over a lock somebody is about to release.
    """
    deadline = time.monotonic() + CORPUS_LOCK_WAIT_SECONDS
    while True:
        try:
            corpus = dl.corpus_conn(corpus_path)
        except Exception as exc:                # noqa: BLE001
            if "lock" not in str(exc).lower() or time.monotonic() > deadline:
                raise
            out(f"  the corpus is locked by another process -- retrying "
                f"in {CORPUS_LOCK_RETRY_SECONDS:.0f}s")
            time.sleep(CORPUS_LOCK_RETRY_SECONDS)
            continue
        try:
            return dl.record(corpus, record)
        finally:
            corpus.close()


def play_draft(conn, corpus_path, cookies, room, rng,
               season: int = CURRENT_SEASON, out=print) -> dict:
    """Join one room, play it to the end, and record it. Returns a summary.

    The status vocabulary is deliberately explicit -- `recorded`,
    `incomplete`, `no_settings`, `no_slot`, `bad_shape`, `join_failed`,
    `socket_failed` -- because this runs unattended and "it did not work" is
    not a usable morning report. Nothing here raises for an ordinary failure;
    the caller counts statuses and moves on to the next room.
    """
    league_id = room["leagueId"]
    swid = cookies["SWID"]
    out(f"room {league_id}: {room.get('teamsJoined')}/{room.get('leagueSize')} "
        f"joined, {room.get('experienceType')}, starts in "
        f"{max(0, int((room.get('draftDate', 0) - time.time() * 1000) / 1000))}s")

    fetch = http_fetch(cookies)
    post = lobby.http_poster(cookies)
    try:
        team_id = lobby.join(post, league_id, swid, season)
    except Exception as exc:                    # noqa: BLE001 -- a room can
        # fill between the directory read and the join; that is an ordinary
        # race, not a bug, and the answer is the next room.
        out(f"  join failed: {exc}")
        return {"status": "join_failed", "league_id": league_id}
    started_at = datetime.now(timezone.utc)
    out(f"  joined as team {team_id}")

    # The room's REAL shape, from the room. See the module docstring: this is
    # the one thing that must never be assumed, and a room that will not tell
    # us is a room we do not play rather than one we guess about.
    raw = fetch_league_settings(fetch, league_id, season)
    settings = None
    if raw:
        try:
            settings = league.from_espn(raw)
        except Exception as exc:                # noqa: BLE001
            out(f"  could not read league settings: {exc}")
    if settings is None:
        out("  ESPN published no usable settings for this room -- skipping "
            "rather than guessing its roster shape")
        return {"status": "no_settings", "league_id": league_id}
    if not settings.scoring:
        out("  ESPN's scoring items mapped to nothing -- skipping")
        return {"status": "no_settings", "league_id": league_id}
    teams, rounds = int(settings.teams), int(settings.rounds)
    listed = room.get("leagueSize")
    if listed is not None and int(listed) != teams:
        # The directory and the league's own settings disagreeing about seat
        # count is exactly the situation that silently misattributes every
        # pick. Refuse rather than pick a side.
        out(f"  the lobby says {listed} teams and the league says {teams} -- "
            "skipping rather than attributing picks to a guessed shape")
        return {"status": "bad_shape", "league_id": league_id}
    out(f"  shape: {teams} teams x {rounds} rounds "
        f"({league.scoring_format(settings)})")

    # Built BEFORE the wait below, not after: it is the only slow step left
    # (board + pool, measured 4-6s together) and the wait is dead time we
    # already have to spend, so spending it here means the socket opens the
    # moment the room does.
    tool = build_tool(conn, settings)
    out(f"  board built: {len(tool.pool.player_id)} pool players, "
        f"{int(tool.sendable.sum())} selectable")

    slots = snake_slots(teams, rounds)
    caps = _roster_cap(settings)
    turns = _turns_left(slots, rounds)
    total_picks = len(slots)
    listener = DraftListener(tool.crosswalk)

    _wait_until_available(room, out)
    session = _connect_session(listener, league_id, team_id, swid, fetch,
                               season, out)
    if session is None:
        out("  gave up opening the draft socket for this room")
        return {"status": "socket_failed", "league_id": league_id}

    try:
        if not _wait_for(lambda: listener.my_team_id is not None,
                         TOKEN_TIMEOUT_SECONDS):
            out("  ESPN never sent a TOKEN frame naming our team")
            return {"status": "socket_failed", "league_id": league_id}
        if listener.my_team_id != team_id:
            # The socket authenticated as a different seat than the invite
            # assigned. Every slot attribution downstream keys off this, so
            # trust the socket (it is what ESPN is actually running) and say
            # so rather than carrying two numbers.
            out(f"  ESPN's socket says we are team {listener.my_team_id}, "
                f"not {team_id} -- using the socket's answer")
            team_id = listener.my_team_id

        my_slot = slot_from_pick_order(settings, team_id)
        timeline = []
        picks_seen = -1
        last_progress = time.monotonic()
        start_deadline = time.monotonic() + START_TIMEOUT_SECONDS
        last_autodraft_nudge = 0.0

        while True:
            if not session.alive():
                out(f"  socket thread stopped: {session.error}")
                break
            timeline = draft_timeline(listener.events)
            n = len(timeline)
            if n != picks_seen:
                picks_seen, last_progress = n, time.monotonic()
            if n >= total_picks:
                break
            if n == 0:
                if time.monotonic() > start_deadline:
                    out("  the draft never started")
                    break
            elif time.monotonic() - last_progress > IDLE_TIMEOUT_SECONDS:
                out(f"  no pick in {IDLE_TIMEOUT_SECONDS:.0f}s at {n}/"
                    f"{total_picks} -- abandoning this room")
                break

            # Take the socket's own answer for our slot as soon as it can
            # give one: it is derived from picks that actually landed, where
            # pickOrder is a field ESPN fills in before a room is full.
            from_socket = slot_for_team(timeline, slots, team_id,
                                        listener.on_the_clock)
            if from_socket is not None:
                if my_slot is not None and my_slot != from_socket:
                    out(f"  pickOrder said slot {my_slot}, the socket says "
                        f"{from_socket} -- trusting the socket")
                my_slot = from_socket

            # ESPN flips a team onto autodraft the moment it misses a pick,
            # and never flips it back on its own. Left alone, one missed
            # clock would turn the rest of our seat into ESPN's ADP engine.
            # Same frame format and same "wait for ESPN's echo" discipline as
            # /api/live/autodraft; nudged at most once every few seconds so a
            # room that ignores us is not spammed.
            if (listener.my_autodraft is True
                    and time.monotonic() - last_autodraft_nudge > 5.0):
                last_autodraft_nudge = time.monotonic()
                try:
                    session.socket.send("AUTODRAFT false\n")
                    out("  ESPN put us on autodraft -- asked it to stop")
                except Exception as exc:        # noqa: BLE001
                    out(f"  could not turn autodraft off: {exc}")

            if (my_slot is not None and listener.on_the_clock == team_id
                    and session.socket is not None):
                _make_pick(session, tool, settings, slots, timeline, my_slot,
                           caps, turns, rng, out)
                continue

            time.sleep(LOOP_POLL_SECONDS)

        timeline = draft_timeline(listener.events)
    finally:
        # Always, on every path out of the loop above -- a finished draft, an
        # abandoned room, an exception, a KeyboardInterrupt. A bot that keeps
        # a socket attached to a room it has stopped playing is the wedged
        # process this whole structure exists to rule out.
        session.stop()

    n = len(timeline)
    if n < total_picks:
        # The same completeness rule `mock_backfill` applies to a file on
        # disk: a draft that did not finish is not recorded. A partial draft
        # written under a `teams`/`rounds` header describing a full one is
        # indistinguishable, to every later reader, from a complete draft
        # with holes in it.
        out(f"  incomplete: {n}/{total_picks} picks -- not recorded")
        return {"status": "incomplete", "league_id": league_id, "picks": n}

    try:
        mapping = slot_team_map(timeline, slots)
    except ValueError as exc:
        out(f"  refusing to record: {exc}")
        return {"status": "bad_shape", "league_id": league_id}
    if len(mapping) != teams:
        out(f"  refusing to record: only {len(mapping)} of {teams} slots were "
            "ever drafted from")
        return {"status": "bad_shape", "league_id": league_id}

    my_slot = slot_for_team(timeline, slots, team_id) or my_slot
    if my_slot is None:
        out("  refusing to record: never learned which slot was ours")
        return {"status": "no_slot", "league_id": league_id}

    record, dropped = build_record(timeline, tool, settings, league_id, season,
                                   my_slot, started_at, teams, rounds)
    draft_id = record_draft(record, corpus_path, out)
    autodrafted = sum(1 for f in timeline if f.autodrafted)
    out(f"  recorded {draft_id}: {len(record.picks)} picks, slot {my_slot}, "
        f"{autodrafted} autodrafted, {dropped} dropped")
    return {"status": "recorded", "league_id": league_id, "draft_id": draft_id,
            "picks": int(len(record.picks)), "dropped": dropped,
            "autodrafted": autodrafted, "my_slot": my_slot}


def open_board_db(path: str | None = None, out=print):
    """A read-only connection to the universal database the board is built
    from, snapshotting it first if something else holds the write lock.

    DuckDB is single-writer PER FILE and it does not let a reader in while a
    writer holds the lock -- so with `make up` running (the dev API keeps a
    read-write connection to data/nfl.duckdb open for the whole session) even
    `read_only=True` fails outright. This farm is meant to run overnight
    alongside that, so refusing to start would be refusing to do the job.

    The snapshot is a plain file copy, read-only afterwards, and it is
    explicitly a snapshot: the farm only READS reference tables (weekly, adp,
    projections) that a nightly refresh rewrites wholesale, so a copy taken
    now is exactly as good as the live file for the length of one night's
    drafting. It is announced rather than done quietly, because the one thing
    it could hide is a refresh that has not landed yet.
    """
    import duckdb

    from pipeline.db import DEFAULT_PATH

    target = path or DEFAULT_PATH
    try:
        return duckdb.connect(target, read_only=True)
    except Exception as exc:                    # noqa: BLE001
        if "lock" not in str(exc).lower():
            raise
        snapshot = str(Path(tempfile.mkdtemp(prefix="mock-farm-")) /
                       Path(target).name)
        out(f"{target} is locked by another process ({exc.__class__.__name__})"
            f" -- reading a snapshot copy at {snapshot}")
        shutil.copy2(target, snapshot)
        return duckdb.connect(snapshot, read_only=True)


def farm(n: int, season: int = CURRENT_SEASON, seed: int = 0,
         db_path: str | None = None, corpus_path: str | None = None,
         out=print) -> dict:
    """Play up to `n` mock drafts, one at a time, and record each one.

    ONE SESSION AT A TIME, on purpose. ESPN authenticates every socket with
    the same SWID; two concurrent drafts as the same member is untested and
    is the sort of thing that gets an account's sockets dropped. The intel
    notes concurrency as the lever to pull if throughput turns out to be the
    problem, and explicitly only after one session has been observed working
    end to end.
    """
    cookies = load_cookies()
    fetch = http_fetch(cookies)
    rng = np.random.default_rng(seed)
    conn = open_board_db(db_path, out=out)
    counts = {"attempted": 0, "recorded": 0, "picks": 0}
    played: set = set()
    try:
        while counts["recorded"] < n:
            rooms = lobby.list_mock_leagues(fetch, season)
            room = lobby.pick_room(rooms, exclude=played)
            if room is None:
                out(f"no 8-team PPR snake room in the lobby's "
                    f"{len(rooms)} rows right now -- waiting "
                    f"{LOBBY_RETRY_SECONDS:.0f}s")
                time.sleep(LOBBY_RETRY_SECONDS)
                continue
            played.add(str(room["leagueId"]))
            counts["attempted"] += 1
            result = play_draft(conn, corpus_path, cookies, room, rng,
                                season=season, out=out)
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            if result["status"] == "recorded":
                counts["recorded"] += 1
                counts["picks"] += result["picks"]
                out(f"{counts['recorded']}/{n} drafts recorded")
            if counts["recorded"] < n:
                time.sleep(BETWEEN_DRAFTS_SECONDS)
    except KeyboardInterrupt:
        # Not swallowed into a clean exit: the `finally` below still closes
        # both connections and `_Session.__exit__` has already stopped any
        # live socket, so what remains is to say what was done and let the
        # non-zero-ness of an interrupt be visible in the summary.
        out("interrupted -- stopping after "
            f"{counts['recorded']} recorded draft(s)")
    finally:
        conn.close()
        # What the corpus now holds, not just what this run did -- and read
        # after the board connection is closed, in its own short-lived
        # connection, for the same single-writer reason `record_draft`
        # explains. Best effort: a summary that cannot be read (somebody else
        # is mid-write) is not a reason to lose this run's own counts.
        try:
            corpus = dl.corpus_conn(corpus_path)
            try:
                out(dl.summary(corpus).to_string(index=False))
            finally:
                corpus.close()
        except Exception as exc:                # noqa: BLE001
            out(f"could not summarise the corpus: {exc}")
    return counts


def main(argv: list) -> int:
    """Farm N mock drafts into the cross-league corpus.

    Run: python -m pipeline.mock_farm [N] [--seed S]
    """
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a.split("=")[0]: a.partition("=")[2]
             for a in argv[1:] if a.startswith("--")}
    n = int(args[0]) if args and args[0] else 1
    seed = int(flags.get("--seed") or 0)
    counts = farm(n, seed=seed)
    print(json.dumps(counts, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
