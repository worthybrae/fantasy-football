"""Join real ESPN mock drafts, play them, and write them into the corpus.

WHAT THIS IS FOR. `pipeline.mock_backfill` harvested the 23 completed mocks
that happened to be sitting in `data/leagues/`; that pile does not grow by
itself. This module is the thing that makes it grow: it watches ESPN's mock
lobby, takes a seat in a snake room of one of the shapes it is farming,
plays every round, and records the finished draft through
`pipeline.draft_log.record` with `source=SOURCE_MOCK` -- the same corpus and
the same columns, so a later refit reads one table and cannot tell which
writer produced a row except by the two fields this writer knows and the
backfill does not (`my_slot` and per-pick `autodrafted`).

WHICH SHAPE IT JOINS is a rotation, not a constant: each pass counts what
the corpus already holds per `(teams, format)` and prefers the shape it is
shortest of that has a room open right now (`espn_mock_lobby.FARM_SHAPES`,
`draft_log.shape_counts`). Every draft is recorded under its own real shape,
read from the room's own settings, so a corpus of mixed shapes is not a
mixed-up corpus -- `scoring.availability` counts each shape separately.

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
import os
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
from pipeline import farm_claims as claims
from pipeline import redact
from pipeline.draft_listener import DraftListener
from pipeline.draft_socket import (draft_security_token, http_fetch,
                                   load_cookies, run_socket_listener)
from pipeline.espn_live import build_crosswalk
from pipeline.espn_teams import fetch_league_settings, fetch_team_owners
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
# A seat whose socket is DEAD, as opposed to reconnecting. draft_socket's
# own reconnect takes seconds and ESPN replays the draft on JOIN, so a
# handle with nothing attached is ordinary for that long. Observed live
# (room 457868220, 2026-08-25): a seat whose socket cycled for twenty
# minutes -- every JOIN accepted, a greeting, a close, no picks ever
# replayed, every SELECT on our turn refused with "not connected" -- while
# the listener thread stayed alive, so neither the give-up in draft_socket
# nor the idle timer here ever fired. A handle that has had nothing
# attached for this long is that seat, and the answer is a fresh token and
# a new session; failing that, the room is abandoned rather than played to
# the end by ESPN's autodraft under our name.
SOCKET_DEAD_SECONDS = 60.0
# How many times one draft gets that fresh session before it is abandoned.
MAX_SOCKET_RECONNECTS = 2
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
# The most we will ever sleep waiting for a room to open. Room SELECTION
# bounds `draftDate` (at most `lobby.MAX_LEAD_SECONDS` ahead) but nothing
# bounds `draftAvailableDate`, which is a separate field and arrives from
# ESPN unvalidated -- so one malformed or far-future value would put an
# uninterruptible `time.sleep` of hours in front of a seat we are holding,
# and an overnight `N=5` would come back in the morning with zero drafts.
# The two dates are ~90s apart in every row observed, so anything past the
# selection bound is not a room starting late, it is a value we should not
# be trusting; we wait the bound out and let the socket attempt fail fast
# rather than spend the night on it.
MAX_AVAILABLE_WAIT_SECONDS = lobby.MAX_LEAD_SECONDS + AVAILABLE_MARGIN_SECONDS
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
# (measured: a new 8-team PPR room every ~5 minutes; the other farmed shapes
# are rarer, which is the other reason not to wait long here).
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
    """One confirmed pick, as ESPN's frames and the room's roster describe it.

    `autodrafted` is None, not False, only when NOTHING has said -- neither an
    AUTODRAFT frame for that team nor the room's own owner census (see
    `draft_timeline`'s `owners` argument). "ESPN has not said" and "ESPN said
    no" are different facts and the corpus stores which one it has; what
    changed is that the first case is now rare rather than usual, because a
    seat with no owner is known to be autodrafting before a single frame
    arrives.

    `seconds_to_pick` and `clock_seconds` are the turn's timing, and they are
    None together whenever this pick's turn was never seen opening -- a
    connect that lands mid-turn is the real case. None, never 0.0: an
    instantaneous pick is a real and interesting thing (somebody clicking the
    top of the list the moment it is their turn) and it must stay
    distinguishable from a pick we simply could not time. Defaulted so the
    hand-built `PickFrame`s in the tests, and any caller folding an event
    list that carries no timestamps, keep working unchanged.
    """
    pick_no: int
    team_id: int | None
    espn_id: int | None
    autodrafted: bool | None
    seconds_to_pick: float | None = None
    clock_seconds: float | None = None


def draft_timeline(events, owners: dict | None = None) -> list:
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

    THE AUTODRAFT FLAG IS A PER-SLOT STATE MACHINE, and both halves of it are
    needed. Neither alone answers.

      - THE INITIAL STATE comes from `owners` (ESPN team id -> does a real
        person sit there, from `espn_teams.fetch_team_owners`). A seat ESPN
        padded the room out with has no owner and is autodrafting from pick
        one; a seat with an owner starts human. This is the half that was
        missing, and its absence is why the corpus came back 0 True / 32
        False / 96 NULL per draft: ESPN sends NO AUTODRAFT frame for a seat
        that was never human, so those picks -- the majority of most rooms --
        were simply unknown.
      - THE TRANSITIONS come from the frames ESPN broadcasts to the whole
        room. `AUTODRAFT <teamId> true` flips that seat to the engine,
        `AUTODRAFT <teamId> false` flips it back, and both directions really
        happen: in data/draft_room_trace.jsonl team 3 goes true after pick
        13, false after 22 and true again after 29, while team 7 goes true
        after pick 38 and stays. That is the human who drafts three rounds
        and wanders off, which is the case worth catching.

    Each pick is stamped with ITS OWN slot's state at the moment the SELECTED
    landed, not the room's state at the end -- which is why this folds the
    retained event list rather than reading `DraftListener.autodraft_by_team`,
    a latest-value dict that answers only for "now". ESPN sends
    `AUTODRAFT <team> true` the moment a team's clock expires and the
    auto-pick lands immediately after (verified in the capture: line 1112 is
    the flag, 1113 the pick), so reading the flag as the SELECTED arrives
    attributes it to the right pick.

    THE TIMING IS A SECOND STATE MACHINE OVER THE SAME FRAMES, and it is
    folded here rather than measured live for exactly the reason the
    autodraft flag is: each pick wants ITS OWN turn's numbers, and a
    latest-value field on the listener only ever answers "now".

      - `SELECTING <teamId> <clock_ms>` OPENS a turn. The third argument is
        how long that turn is allowed to last (30000 in every one of the 128
        turns in data/draft_room_trace.jsonl, but read rather than assumed --
        it is a room setting, and "took 25 seconds" means opposite things on
        a 30-second clock and a 90-second one).
      - That team's next `SELECTED` CLOSES it. `seconds_to_pick` is the
        difference between the two frames' `received_at`, which
        `DraftListener.on_frame` stamps from a monotonic clock.
      - EXACTLY ONE TURN IS OPEN AT A TIME, because the room has exactly one
        clock. A `SELECTING` for a different team therefore replaces the open
        turn rather than joining it -- otherwise a turn that somehow never
        produced a pick would sit there and attach its start time to that
        team's NEXT pick, several minutes later, and report a deliberation
        that never happened.
      - A REPEATED `SELECTING` FOR THE TEAM ALREADY ON THE CLOCK KEEPS THE
        FIRST TIMESTAMP. ESPN re-sends the room's current state on every
        (re)JOIN and this process reconnects whenever the socket drops, so
        the repeat is a replay of a turn already in progress; taking the
        later stamp would silently shorten a real deliberation to the age of
        the reconnect. The same reconnect replays the whole draft's SELECTED
        frames, and those are deduped above -- BEFORE this fold sees them --
        so a replay can neither close a turn twice nor emit a second timing
        for a pick already counted.
      - A pick with NO open turn for its team is None/None, not zero: the
        farm's own connect can land mid-turn, and "we did not see this turn
        start" is not "he picked instantly". A negative difference (which
        monotonic makes unreachable in one session, but a hand-assembled or
        merged event list could still produce) is discarded the same way,
        because a duration that cannot be true is worse than a missing one.
      - AUTODRAFTED PICKS ARE NOT SPECIAL-CASED and should not be. ESPN
        makes them the instant the clock expires, so they fall out of this
        arithmetic at `seconds_to_pick ~= clock_seconds` -- which is the
        signal, not noise: it is how a model tells "the engine picked" from
        "a person picked quickly" without trusting the flag.

    `owners` omitted, or missing a team, leaves that seat NULL until a frame
    says otherwise -- the old behaviour, kept for the case where the mTeam
    read failed. NULL now means genuinely unknown rather than "usual".

    One honest limit remains: picks already made before this process's FIRST
    connect arrive as a replay, and ESPN replays the autodraft state as it
    stands now rather than as it stood at each of those picks. The farm
    always joins before the draft starts, so that case does not arise in
    practice -- and with `owners` in hand a never-human seat is right even
    then, because it never changed.
    """
    # A seat with no owner is on the engine from pick one; a seat with one
    # starts human. Copied, never mutated in place: the caller's map is the
    # room's census and is read again when the record is built.
    autodraft: dict = {team: (not human)
                       for team, human in (owners or {}).items()}
    seen: set = set()
    out: list = []
    pick_no = 0
    # The one open turn, or Nones when nobody is on a clock we watched start.
    # `open_at` is a `received_at` stamp off the frame itself, so an event
    # list assembled without a listener (no timestamps) simply produces
    # None durations rather than failing.
    open_team = None
    open_at = None
    open_clock_ms = None
    for event in events:
        if event is None:
            continue
        if event.verb == "SELECTING" and event.args:
            team = _as_int(event.args[0])
            clock_ms = _as_int(event.args[1]) if len(event.args) > 1 else None
            if team != open_team:
                open_team, open_at = team, event.received_at
                open_clock_ms = clock_ms
            elif open_clock_ms is None:
                # Same team, still on the clock: keep the earlier start (see
                # the docstring on reconnect replays) but take a clock length
                # the first frame did not carry.
                open_clock_ms = clock_ms
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
        seconds = None
        clock_seconds = None
        if team is not None and team == open_team:
            landed = event.received_at
            if open_at is not None and landed is not None and landed >= open_at:
                seconds = float(landed - open_at)
            if open_clock_ms is not None:
                clock_seconds = open_clock_ms / 1000.0
            # Closed whether or not it could be timed: this pick consumed the
            # turn, and leaving it open would hand its start to the next one.
            open_team = open_at = open_clock_ms = None
        out.append(PickFrame(pick_no, team, espn_id, autodraft.get(team),
                             seconds, clock_seconds))
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
    # Display only, and defaulted so a test can build a Tool without a board.
    # The log this bot leaves behind is the only account of an unattended
    # night that anyone reads in the morning, and "pick 14: 00-0039139" is not
    # an account of anything.
    names: list | None = None


def espn_pool_columns(board: pd.DataFrame, conn) -> pd.DataFrame:
    """`player_id` -> the board as ESPN publishes it: rank, projection, bye.

    THE LIST THE ROOM IS ACTUALLY READING. Everything else the pool snapshot
    stores prices a player against the consensus market; the people in an
    ESPN mock are looking at ESPN's own ranking on screen, and the
    measurement behind this (docs/superpowers/specs/
    2026-08-23-best-opponent-model-design.md) puts 23.1% of human picks on
    the top name of that list against 15.9% on the market's. Recorded per
    draft, in the draft's own pool snapshot, because ESPN's rank is a
    preseason quantity that a later reader cannot recover for a draft played
    weeks ago except by assuming it never moved.

    THREE COLUMNS, TWO SOURCES, and the split is not arbitrary. `board`
    carries `espn_ppr_rank` (ESPN's PPR ranking, market.py's own name for
    it) and `bye`, both already joined to our `player_id`. It does NOT carry
    `espn_proj`: `_BOARD_COLUMNS` drops it, because the board's own
    `proj_points` is that number re-priced into the league's scoring rules
    (see `proj_scale`), and re-adding it to the board would change a frame
    half this project reads. So the raw projection is taken from `espn_adp`
    itself, joined on the `espn_id` the board already carries -- ESPN's own
    key, not a name match, so this join cannot put one player's projection on
    another's row.

    Shared with `pipeline.backfill_espn_board` deliberately: the backfill
    exists to give already-recorded drafts the same three columns this farm
    now writes at record time, and two implementations of "ESPN's rank for
    this player" would be two different answers in one column.
    """
    from pipeline.db import read_table

    if board is None or board.empty:
        return pd.DataFrame(columns=["player_id", "espn_rank", "espn_proj",
                                     "bye"])
    # One row per player: the board can hold a duplicate player_id (two ADP
    # rows folded onto one nflverse id), and `first` is the same tie-break
    # `build_tool` already uses for `team` and `name` just below.
    rows = board.drop_duplicates("player_id", keep="first")
    out = pd.DataFrame({
        # The board's own key, uncast, so this joins to a pool frame exactly
        # the way `build_tool`'s `team` map already does. A caller joining to
        # the corpus (where `player_id` is VARCHAR) casts on its own side.
        "player_id": rows["player_id"],
        "espn_rank": (pd.to_numeric(rows["espn_ppr_rank"], errors="coerce")
                      if "espn_ppr_rank" in rows.columns else np.nan),
        "bye": (pd.to_numeric(rows["bye"], errors="coerce")
                if "bye" in rows.columns else np.nan),
    })

    espn = read_table(conn, "espn_adp")
    if (not espn.empty and {"espn_id", "espn_proj"} <= set(espn.columns)
            and "espn_id" in rows.columns):
        keyed = espn.assign(_key=pd.to_numeric(espn["espn_id"],
                                               errors="coerce"))
        keyed = keyed.dropna(subset=["_key"]).drop_duplicates("_key",
                                                              keep="first")
        proj = keyed.set_index("_key")["espn_proj"]
        out["espn_proj"] = (pd.to_numeric(rows["espn_id"], errors="coerce")
                            .map(proj).to_numpy(dtype=float))
    else:
        # No `espn_adp` in this database (a fixture, or a snapshot taken
        # before the ESPN job ran). NULL rather than an absent column, so the
        # frame's shape does not depend on what the database happened to
        # hold -- `draft_log._shape` would fill it anyway, and a caller
        # counting coverage should see a column full of nulls rather than a
        # KeyError.
        out["espn_proj"] = np.nan
    return out[["player_id", "espn_rank", "espn_proj", "bye"]].reset_index(
        drop=True)


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

    # ESPN's own rank, projection and bye, joined on the same key. This is
    # the one place the two mock writers now DIFFER: `mock_backfill` reads a
    # finished league file and has no board contemporaneous with the draft it
    # is importing, so it leaves these NULL and
    # `pipeline.backfill_espn_board` fills them from today's board afterwards
    # -- which is honest for a preseason quantity and would not be for
    # anything that moves in-season. Written here at record time because this
    # writer HAS the right board in hand: it is the board this draft was
    # played from, three lines up.
    pool_df = pool_df.merge(espn_pool_columns(board, conn),
                            on="player_id", how="left")

    name_by_player = (board.drop_duplicates("player_id", keep="first")
                      .set_index("player_id")["name"]
                      if "name" in board.columns else {})
    names = [str(name_by_player.get(str(pid), pid)) for pid in pool.player_id]

    return Tool(board=board, pool=pool, pool_df=pool_df, crosswalk=crosswalk,
                espn_by_index=espn_by_index, index_by_player=index_by_player,
                sendable=sendable, names=names)


def human_seats(owners: dict | None, my_team_id: int | None) -> int | None:
    """How many seats held a real person, ours not counted. None if unasked.

    None rather than 0 when `owners` is empty or absent: "the mTeam read
    failed" and "the room was entirely computers" are different facts about a
    draft, and only one of them is a reason to distrust its picks. Our own
    team is subtracted whether or not it appears in the map, so a census taken
    before our own seat was written still answers the same number.
    """
    if not owners:
        return None
    return sum(1 for team, human in owners.items()
               if human and team != my_team_id)


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
                 started_at, teams: int, rounds: int,
                 owners: dict | None = None,
                 my_team_id: int | None = None) -> tuple:
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

    `owners` (ESPN team id -> is a person sitting there) is stamped onto each
    pick as `had_owner` and counted onto the draft head as `human_seats`. The
    per-pick copy is what a fit reads while walking picks; the per-draft count
    is what a later query filters whole drafts on ("only rooms with at least
    three people in them"), and it EXCLUDES OUR OWN SEAT, because we always
    have an owner and we are a bot -- counting ourselves would make every room
    look one person more human than it was. `my_team_id` is therefore needed
    as well as `my_slot`: `owners` is keyed by ESPN team id and the head count
    is taken over that map, not over the picks.

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
            "had_owner": (None if owners is None
                          else owners.get(frame.team_id)),
            # Straight off the frame, both of them, including the Nones --
            # `draft_timeline` has already decided which turns it could time
            # and a default invented here would be a fact nobody observed.
            "seconds_to_pick": frame.seconds_to_pick,
            "clock_seconds": frame.clock_seconds,
        })
    picks = pd.DataFrame(rows, columns=[
        "pick_no", "round", "slot", "owner_key", "is_anonymous", "player_id",
        "autodrafted", "had_owner", "seconds_to_pick", "clock_seconds"])
    picks = picks.merge(
        tool.pool_df[["player_id", "position", "adp_rank", "proj_points"]],
        on="player_id", how="left")
    missing = picks["position"].isna()
    dropped = int(missing.sum())

    record = dl.DraftRecord(
        source=dl.SOURCE_MOCK, league_id=str(league_id), season=int(season),
        teams=int(teams), rounds=int(rounds), my_slot=my_slot,
        human_seats=human_seats(owners, my_team_id),
        # THE SCORING TABLE ON ITS OWN, as well as inside `settings_json`.
        # A draft's shape is (teams, format) -- see `draft_log.draft_format`
        # -- and the format is one number in this table, so a counter that
        # groups the corpus by shape should not have to parse a whole
        # LeagueSettings to find it. The two cannot disagree: they are the
        # same object, written twice, in the same call.
        scoring_json=json.dumps(dict(settings.scoring or {})),
        settings_json=league.to_json(settings), started_at=started_at,
        picks=picks[~missing].reset_index(drop=True), pool=tool.pool_df,
        draft_id=draft_id)
    return record, dropped


# ---------------------------------------------------------------------------
# Live progress: what a draft publishes while it is still being played.
# ---------------------------------------------------------------------------
#
# WHY A FILE PER ROOM AND NOT THE CORPUS. `record_draft` writes when a draft
# FINISHES, so for the thirty to forty minutes one is being played it exists
# nowhere but this process's own memory -- and the web API is a separate
# process (uvicorn), which cannot see that. The obvious fix, recording to the
# corpus after every pick, is the one thing that must not happen here: DuckDB
# is single-writer, three concurrent farm sessions x 128 picks is ~380
# acquisitions of that one write lock per cycle, each of them rewriting all
# 128 pick rows (`dl.record` is delete-then-insert by design), contending with
# each other AND with the page's own reads. The corpus is also the one file in
# this repo that cannot be rebuilt if it is damaged -- see
# `pipeline.draft_log`'s module docstring -- which makes it the last place in
# the project to put a hot write path.
#
# So live progress is published exactly the way a room claim is: one small
# file per league in a directory, no daemon, nothing to clean up if the whole
# machine dies. Two things differ from `pipeline.farm_claims`, both
# deliberately:
#
#   - THE WRITE IS A TMP FILE PLUS `os.replace`, not an exclusive create. A
#     claim is written once and never touched again; this is rewritten after
#     every pick, and a reader that caught a rewrite half-done would draw a
#     truncated board. `os.replace` is atomic on every filesystem this runs
#     on, so a reader sees either the previous pick's file or this one's and
#     never a mixture -- which also means the "a claim is empty for an instant
#     after it is made" race that farm_claims documents at length simply does
#     not arise here, and an unreadable file really is a corrupt one.
#   - THE PID AND TIMESTAMP LIVE INSIDE THE JSON rather than being the whole
#     of the file, because there is a payload to carry as well.
#
# STALENESS IS THE SAME PROBLEM AND TAKES THE SAME ANSWER. A farm killed
# mid-draft leaves its file behind, and a page that kept serving it would show
# a draft as "live" that stopped moving days ago -- worse than showing nothing,
# because nothing about it looks wrong. A file whose pid is gone, or whose
# stamp is older than the TTL, is retired on the next read, which is
# `farm_claims._is_stale`'s rule against `farm_claims`'s own clock.

# Env-overridable for the same two reasons FARM_CLAIM_DIR is: a test must
# never write the directory a real farm is publishing into, and a second
# instance may want its own.
LIVE_DIR = os.environ.get("FARM_LIVE_DIR", "data/farm-live")

# The claim's own constants rather than two numbers that could drift apart. A
# live file and a claim have exactly the same lifetime -- both are created
# before the first pick of a room and both are removed in the same `finally`
# in `farm` -- so a live file that has aged out while its claim has not (or
# the reverse) would only ever be a bug.
LIVE_TTL_SECONDS = claims.CLAIM_TTL_SECONDS
LIVE_GRACE_SECONDS = claims.CLAIM_GRACE_SECONDS


def _live_dir() -> Path:
    path = Path(LIVE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def live_path(league_id) -> Path:
    """Where one room's live file lives. Named by league id, exactly as a
    claim is, so the two can be read against each other by eye."""
    return _live_dir() / f"{league_id}.json"


def iso_utc(when) -> str | None:
    """A datetime as the API serves it: UTC, no microseconds, `Z`.

    Normalised here rather than left to `isoformat()` so every timestamp the
    page ever receives has one shape. NOT what `draft_id` is computed from --
    that takes the datetime itself (see `live_payload`), because
    `dl.draft_id_for` hashes `started_at.isoformat()` and a draft whose id
    changed when it was recorded would appear on the page twice.
    """
    if when is None:
        return None
    if getattr(when, "tzinfo", None) is not None:
        when = when.astimezone(timezone.utc)
    return when.replace(microsecond=0, tzinfo=None).isoformat() + "Z"


def seat_owners(timeline, slots, teams: int, owners: dict | None) -> list:
    """Per-seat `had_owner`, as far as the picks so far can say.

    `owners` is keyed by ESPN team id and the page draws slots, so the two
    have to be joined through the picks themselves -- the same derivation
    `slot_team_map` makes at record time, with two differences that matter
    while a draft is still running:

      - A seat that has not picked yet has no known team, so its answer is
        None (unknown), not False. In round one that is most of the room for
        the first minute, and calling those seats "computer" would label a
        board of real people as an empty room.
      - A collision (two teams drafting at one slot) is taken first-wins
        rather than raised on. `slot_team_map` refuses to record such a draft
        and that is the right call for the corpus; a live view that threw
        would take the whole page down over one seat's label.
    """
    team_by_slot: dict = {}
    for frame in timeline:
        if frame.team_id is None or frame.pick_no > len(slots):
            continue
        team_by_slot.setdefault(slots[frame.pick_no - 1], frame.team_id)
    seats = []
    for slot in range(1, int(teams) + 1):
        team = team_by_slot.get(slot)
        had_owner = (None if owners is None or team is None
                     else owners.get(team))
        seats.append({
            "slot": slot,
            "had_owner": None if had_owner is None else bool(had_owner)})
    return seats


def live_payload(timeline, tool, league_id, season, teams, rounds, my_slot,
                 started_at, owners: dict | None = None,
                 my_team_id: int | None = None) -> dict:
    """This draft as it stands right now, in enough detail to draw the board
    without the corpus.

    EVERYTHING THE PAGE NEEDS AND NOTHING IT DOES NOT. The player pool is
    deliberately absent: it is 250 rows that do not change during a draft, and
    the API resolves every name, rank and headshot out of the board it already
    holds (`data/nfl.duckdb`) by player_id anyway. What only this process
    knows is the room -- its shape, which seat is ours, which seats hold
    people, and the picks in order -- so that is what is written.

    `draft_id` IS THE ID THE CORPUS WILL USE, computed from the same
    (league_id, season, started_at) `build_record` passes to
    `dl.draft_id_for`, and passed the `started_at` DATETIME rather than its
    printed form so the two hashes cannot differ. That is what lets a draft
    keep its identity when it stops being live and becomes a corpus row: the
    page's link survives the transition instead of 404ing at the finish line.
    Note this is NOT the content hash `pipeline.mock_backfill` computes -- a
    hash of the picks would change on every pick, which is exactly what an id
    must not do while a draft is still filling in.

    A pick whose player the crosswalk cannot resolve carries `player_id:
    null` and keeps its place, the same way `taken_order_from` keeps one: a
    pick nobody can name still happened, and dropping it would slide every
    later pick onto the wrong seat.
    """
    slots = snake_slots(int(teams), int(rounds))
    picks = []
    for frame in timeline:
        if frame.pick_no > len(slots):
            break
        player_id = (tool.crosswalk.get(frame.espn_id)
                     if frame.espn_id is not None else None)
        picks.append({
            "pick_no": int(frame.pick_no),
            "slot": int(slots[frame.pick_no - 1]),
            "player_id": None if player_id is None else str(player_id),
            "autodrafted": (None if frame.autodrafted is None
                            else bool(frame.autodrafted)),
            # The turn's timing, carried through unchanged from the frame.
            # `clock_seconds` is the only thing that can drive a real
            # countdown on the page: ESPN's SELECTING frame is the one place
            # the clock's FULL length is ever stated (its CLOCK frames carry
            # only what is left), so a reader that never saw a turn open
            # cannot recover it later at any price. None when this pick's
            # turn was never seen opening -- never guessed, because a room on
            # a 90-second clock shown a 30-second one is a countdown that
            # lies twice a minute.
            "clock_seconds": (None if frame.clock_seconds is None
                              else float(frame.clock_seconds)),
            "seconds_to_pick": (None if frame.seconds_to_pick is None
                                else float(frame.seconds_to_pick)),
        })
    return {
        "league_id": str(league_id),
        "draft_id": dl.draft_id_for(dl.SOURCE_MOCK, league_id, season,
                                    started_at),
        "season": int(season),
        "teams": int(teams),
        "rounds": int(rounds),
        "my_slot": None if my_slot is None else int(my_slot),
        "started_at": iso_utc(started_at),
        "human_seats": human_seats(owners, my_team_id),
        "seats": seat_owners(timeline, slots, int(teams), owners),
        "picks": picks,
    }


def publish_pick(league_id, build, warned: bool, out) -> bool:
    """A pick has landed: hold the room, then publish the board.

    THE CLAIM COMES FIRST AND IS NOT BEST-EFFORT. A claim is stamped once,
    when the room is taken, and a 12x16 room on a 30-second clock runs for 96
    minutes of picking -- longer than any TTL that also lets a crashed
    process's room be reused the same morning. Refreshing the stamp on every
    pick turns `CLAIM_TTL_SECONDS` into a bound on the gap between two picks
    instead, so a live draft can never be swept out from under itself and
    handed to a second farm process (see `pipeline.farm_claims`).

    THE LIVE FILE IS best-effort, and stays that way. A board the page cannot
    draw is not a reason to stop drafting: the draft is recorded from this
    process's own memory at the end regardless. `build` is a callable rather
    than a payload so that building one is inside the guard too. `warned` is
    said ONCE per room and returned so the caller can carry it -- an
    unwritable directory would otherwise fill an overnight log with 128
    copies of one line.
    """
    claims.touch(league_id)
    try:
        write_live(build())
    except Exception as exc:                    # noqa: BLE001 -- see above
        if not warned:
            out(f"  could not publish live progress: {exc}")
            return True
    return warned


def write_live(payload: dict, now: float | None = None) -> Path:
    """Publish one room's progress, atomically.

    Written beside the target and renamed over it. The rename is the whole
    mechanism: a reader that opens the file mid-write would otherwise get a
    JSON document cut off at whatever byte the writer had reached, and this
    file is rewritten after every one of 128 picks. The tmp name carries the
    pid so two writers could not share it -- one claim per room means two
    cannot happen, but surviving it costs nothing.

    `pid` and `written_at` are stamped here rather than by the caller: they
    describe the WRITE, not the draft, and a payload built in one process and
    written by another would otherwise claim the wrong owner.
    """
    now = time.time() if now is None else now
    directory = _live_dir()
    name = str(payload["league_id"])
    tmp = directory / f".{name}.{os.getpid()}.tmp"
    body = dict(payload, pid=os.getpid(), written_at=float(now))
    with open(tmp, "w") as fh:
        json.dump(body, fh, default=str)
    path = directory / f"{name}.json"
    os.replace(tmp, path)
    return path


def read_live(path) -> dict | None:
    """One live file's payload, or None if it will not read as one."""
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _live_is_stale(path: Path, payload: dict | None, now: float) -> bool:
    """Whether this file is a phantom rather than a draft in progress.

    Same two retirements as `farm_claims._is_stale` and for the same reasons:
    the pid it names is gone (a farm killed mid-draft), or its stamp is older
    than the TTL (the backstop for a pid recycled onto some unrelated process
    on a machine that has been up for weeks). A file that will not parse is
    aged off its mtime instead of being trusted or swept immediately -- the
    atomic rename means that should never happen, so the one case left is a
    genuinely corrupt or hand-edited file, and giving it the grace period
    costs a minute and cannot delete a draft that is merely being written.
    """
    pid = payload.get("pid") if payload else None
    stamp = payload.get("written_at") if payload else None
    if not isinstance(pid, int) or not isinstance(stamp, (int, float)):
        try:
            age = now - path.stat().st_mtime
        except OSError:
            return False                # gone already; nothing to retire
        return age > LIVE_GRACE_SECONDS
    if now - float(stamp) > LIVE_TTL_SECONDS:
        return True
    # The same signal-0 check `farm_claims` makes, called rather than copied
    # so the two modules cannot come to disagree about what "alive" means.
    return not claims._alive(pid)


def live_drafts(now: float | None = None) -> list:
    """Every draft a live farm process is currently playing, stale ones swept.

    The sweep happens on READ, exactly as `farm_claims.claimed` does it: the
    process that would have cleaned up is by definition the one that died, so
    the only code guaranteed to run afterwards is somebody else's read.
    """
    now = time.time() if now is None else now
    out = []
    for path in sorted(_live_dir().glob("*.json")):
        if not path.is_file():
            continue
        payload = read_live(path)
        if _live_is_stale(path, payload, now):
            try:
                path.unlink()
            except OSError:
                pass                    # another reader swept it first
            continue
        if payload is not None:
            out.append(payload)
    return out


def clear_live(league_id) -> None:
    """Stop advertising this room as live, if the file is still ours.

    The ownership check is `farm_claims.release`'s, for its reason: a process
    that stalled long enough for its file to age past the TTL can have had the
    room swept and re-entered by another farm, and an unconditional unlink
    would then delete THAT process's live board. Never raises -- a page that
    keeps showing one phantom draft is not worth ending a night's farming
    over, and the staleness sweep retires it either way.
    """
    path = live_path(league_id)
    payload = read_live(path)
    pid = payload.get("pid") if payload else None
    if isinstance(pid, int) and pid != os.getpid():
        return
    try:
        path.unlink()
    except OSError:
        pass


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

    Interruptible only by killing the process, so the wait is CLAMPED rather
    than taken on trust. `pick_room` bounds `draftDate` to
    `lobby.MAX_LEAD_SECONDS` ahead, but `draftAvailableDate` is a different
    field and nothing bounds it -- see MAX_AVAILABLE_WAIT_SECONDS for why one
    bad value would otherwise cost the whole night.
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
    if seconds > MAX_AVAILABLE_WAIT_SECONDS:
        # Said out loud rather than clamped quietly: a room whose two dates
        # disagree by more than the selection window is either ESPN sending
        # something new or a room we should not have picked, and both are
        # worth seeing in the morning's log.
        out(f"  room says it opens in {seconds:.0f}s, past the "
            f"{MAX_AVAILABLE_WAIT_SECONDS:.0f}s a picked room can be away "
            "-- waiting that long and no longer")
        seconds = MAX_AVAILABLE_WAIT_SECONDS
    out(f"  room opens in {seconds:.0f}s -- waiting before connecting")
    time.sleep(seconds)


def _attached(handle) -> bool:
    """Whether the session's socket handle currently holds a connection.

    `None` (not published yet) and a handle with no `alive` (the fakes the
    tests hand in) both read as attached: the dead-socket rule above is
    about a handle that can say it is empty and has said so for a minute,
    not about anything that cannot answer."""
    alive = getattr(handle, "alive", None)
    return True if alive is None else bool(alive())


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
        name = (tool.names[idx] if tool.names else str(tool.pool.player_id[idx]))
        name = f"{name} ({tool.pool.position[idx]})"
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
    not a usable morning report.

    `bad_shape` covers both ways a room can turn out not to be the room the
    lobby advertised: a seat count the directory and the league disagree
    about, and a scoring format that is not one of `FARM_SHAPES` once the
    league's own settings are read. Neither is recoverable by drafting
    anyway, and both are decided before the board is built. Nothing here raises for an ordinary failure;
    the caller counts statuses and moves on to the next room.

    `out` is wrapped in the credential scrubber here as well as in `farm`,
    because this is a public entry point somebody will one day call on its
    own. Wrapping twice costs one extra pass over each line and cannot
    corrupt it -- `redact` is idempotent by construction.
    """
    out = redact.redacting(out)
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
        #
        # The status code, not httpx's sentence. `HTTPStatusError` stringifies
        # to prose wrapped around the whole request URL, and that URL carries
        # `memberId=<the SWID cookie>` -- the sole credential on the draft
        # socket handshake -- so the plainest failure this loop has (a room
        # that filled while we were reading the directory: ESPN answers 400)
        # used to print the login into the overnight log. `redacted_error`
        # keeps what a morning reader needs from it, which is the code: 400
        # means the room went, 401 means the cookies expired, and those are
        # different mornings.
        out(f"  join failed: {redact.redacted_error(exc)}")
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
    fmt = league.scoring_format(settings)
    out(f"  shape: {teams} teams x {rounds} rounds ({fmt})")
    # THE REAL SHAPE, GATED. The directory row said what ESPN's lobby
    # advertises; this is what the league itself is scored by, and they are
    # not always the same thing. The directory publishes stat IDS and not
    # point values, so a half-PPR room and a full-PPR one both read "PPR with
    # receptions scored" from the lobby and only the settings can tell them
    # apart -- which means the rotation can be handed a shape it never asked
    # for, right after taking a seat.
    #
    # A draft is recorded under its REAL shape (`build_record` writes this
    # same scoring table), so playing it anyway would file 128 picks under a
    # shape nobody is farming, count towards nothing, and drag the pooled
    # depth of the whole corpus down to whatever that room reached. Better to
    # lose the seat.
    #
    # `league.scoring_format` and `draft_log.draft_format` are the same rule
    # (both call `scoring.league.format_for_receptions`), so the shape gated
    # here is exactly the shape the corpus will count it under.
    if (teams, fmt) not in lobby.FARM_SHAPES:
        out(f"  the room is really {teams}:{fmt}, which is not one of the "
            f"shapes being farmed -- leaving without drafting")
        return {"status": "bad_shape", "league_id": league_id}

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

        # THE OWNER CENSUS, TAKEN HERE AND NOWHERE ELSE. A mock league 404s
        # the moment its draft ends -- verified -- so this is the last chance
        # to learn which of the eight seats holds a person, and every use of
        # it downstream reads this one map rather than asking ESPN again.
        #
        # Taken after the socket is up rather than straight after the invite
        # POST, and that is the whole accuracy of the number. `pick_room`
        # takes a seat up to MAX_LEAD_SECONDS (15 minutes) before the draft
        # starts and people keep arriving in that window; a census taken at
        # the invite would record the room as we found it rather than as it
        # drafted, and would call a seat "computer" that a person took two
        # minutes later. By this line the room has passed
        # `draftAvailableDate`, ESPN has accepted our JOIN and named our team,
        # and the lobby countdown is nearly out -- the roster is as final as
        # it gets while still being readable.
        owners = fetch_team_owners(fetch, league_id, season)
        if owners:
            out(f"  seats: {human_seats(owners, team_id)} of "
                f"{len(owners)} held by a person (ours excluded)")
        else:
            # Said out loud: without it every seat starts NULL and the draft
            # lands in the corpus with the same unusable autodraft column
            # this whole read exists to fix.
            out("  could not read the room's owners -- autodraft state will "
                "be unknown for any seat ESPN never sends a frame about")

        my_slot = slot_from_pick_order(settings, team_id)
        timeline = []
        picks_seen = -1
        # The pick count the live file was last written at, kept apart from
        # `picks_seen` because the two answer different questions: that one
        # is the idle timer's "has this room moved", read before the start
        # and idle checks below, and this one is "has the page been told",
        # which is only decidable after `my_slot` has been resolved.
        published = -1
        live_published_warning = False
        last_progress = time.monotonic()
        start_deadline = time.monotonic() + START_TIMEOUT_SECONDS
        last_autodraft_nudge = 0.0
        # When the socket handle last went empty, or None while it holds a
        # connection. See SOCKET_DEAD_SECONDS.
        detached_since = None
        reconnects = 0

        while True:
            if not session.alive():
                out(f"  socket thread stopped: {session.error}")
                break

            # THE SEAT'S SOCKET IS DEAD, not merely reconnecting: nothing has
            # been attached to the handle for SOCKET_DEAD_SECONDS. The
            # listener thread is alive (checked above) and cycling, which is
            # the one shape neither its own give-up nor the idle timer below
            # catches. Stop it, mint a fresh token, join again -- the shared
            # listener keeps every event it has, and ESPN's replay on JOIN
            # is deduped on the way in -- and if that seat is dead too,
            # abandon the room.
            if not _attached(session.socket):
                if detached_since is None:
                    detached_since = time.monotonic()
                elif time.monotonic() - detached_since >= SOCKET_DEAD_SECONDS:
                    if reconnects >= MAX_SOCKET_RECONNECTS:
                        out(f"  the draft socket has been down for "
                            f"{SOCKET_DEAD_SECONDS:.0f}s again -- abandoning "
                            "this room")
                        break
                    reconnects += 1
                    out(f"  the draft socket has been down for "
                        f"{SOCKET_DEAD_SECONDS:.0f}s -- reconnecting with a "
                        f"fresh token ({reconnects}/{MAX_SOCKET_RECONNECTS})")
                    session.stop()
                    fresh = _connect_session(listener, league_id, team_id,
                                             swid, fetch, season, out)
                    if fresh is None:
                        out("  could not reconnect -- abandoning this room")
                        break
                    session = fresh
                    detached_since = None
            else:
                detached_since = None
            timeline = draft_timeline(listener.events, owners)
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

            # PUBLISH WHAT THE PAGE READS. Only when the pick count actually
            # moved: this loop wakes four times a second and a room on a slow
            # clock would otherwise rewrite the same board a hundred times
            # between picks. Written HERE rather than beside the `picks_seen`
            # bookkeeping above because `my_slot` is the field that decides
            # which column of the published board is ours, and a payload
            # written a few lines earlier would carry the previous pick's
            # answer -- which, on the first pick of the draft, is None.
            #
            # The zeroth write matters as much as the rest: `published`
            # starts at -1, so a room that has joined but not started yet
            # publishes an empty board immediately and appears on the page as
            # a draft about to happen rather than as nothing at all.
            if n != published:
                published = n
                live_published_warning = publish_pick(
                    league_id,
                    lambda: live_payload(timeline, tool, league_id, season,
                                         teams, rounds, my_slot, started_at,
                                         owners=owners, my_team_id=team_id),
                    live_published_warning, out)

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

            # BOTH authorities have to agree that it is our turn, and the
            # second clause is not belt-and-braces. ESPN's SELECTING frame
            # is what sets `on_the_clock`, and it arrives a beat AFTER the
            # SELECTED that ended the previous pick -- including the SELECTED
            # confirming our own. So for a fraction of a second after we
            # pick, ESPN's last word still names us while the snake has
            # already moved on, and a loop trusting `on_the_clock` alone
            # would re-enter `_make_pick` for a turn that is not ours and
            # spend the whole turn budget sending SELECTs ESPN ignores --
            # long enough, at a snake turn, to miss the pick that IS ours.
            # `slots[n]` is the same "is it my turn" test /api/live/select
            # makes with `picks_until_turn`, against the same snake this
            # draft is recorded under.
            if (my_slot is not None and session.socket is not None
                    and n < len(slots) and slots[n] == my_slot
                    and listener.on_the_clock == team_id):
                _make_pick(session, tool, settings, slots, timeline, my_slot,
                           caps, turns, rng, out)

            # UNCONDITIONAL, including immediately after a pick attempt --
            # this used to be a `continue` past it, and that was a busy-spin
            # waiting for the one moment it would fire. `_make_pick` returns
            # FALSE IMMEDIATELY when the send raises, and a send raises
            # synchronously whenever `draft_socket` has detached the socket
            # to reconnect (`SocketHandle.send` on a detached handle is an
            # instant ConnectionError, and `session.socket` stays non-None
            # once published, so the guard above still passes). ESPN dropping
            # the socket mid-turn is routine and the reconnect takes seconds,
            # during which the loop re-ran `draft_timeline` over every event,
            # `_seed_rosters` over every pick so far and `_greedy_choice`
            # over ~250 candidates, thousands of times, with no sleep --
            # burning a core and fighting the listener thread for the GIL at
            # exactly the moment our pick was on the clock. A quarter second
            # after a pick lands costs nothing: the next thing this loop can
            # usefully see is a frame that has not arrived yet.
            time.sleep(LOOP_POLL_SECONDS)

        timeline = draft_timeline(listener.events, owners)
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
                                   my_slot, started_at, teams, rounds,
                                   owners=owners, my_team_id=team_id)
    draft_id = record_draft(record, corpus_path, out)
    autodrafted = sum(1 for f in timeline if f.autodrafted)
    unknown = sum(1 for f in timeline if f.autodrafted is None)
    out(f"  recorded {draft_id}: {len(record.picks)} picks, slot {my_slot}, "
        f"{autodrafted} autodrafted, {unknown} unknown, "
        f"{record.human_seats} human seats, {dropped} dropped")
    return {"status": "recorded", "league_id": league_id, "draft_id": draft_id,
            "picks": int(len(record.picks)), "dropped": dropped,
            "autodrafted": autodrafted, "my_slot": my_slot,
            "human_seats": record.human_seats}


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
        # TWO WAYS TO BE REFUSED, and both mean the same thing here: somebody
        # else has this file open on terms we cannot join, so copy it.
        #
        #   "lock"          another PROCESS holds the write lock. The dev API,
        #                   or the deployed one, keeping a connection for its
        #                   whole life.
        #   "configuration" the same PROCESS already has it open read-write.
        #                   DuckDB caches one instance per file per process
        #                   and every connection to it must agree on the
        #                   configuration, so `read_only=True` is refused
        #                   outright rather than queued.
        #
        # Only the first was matched here for a long time, so an in-process
        # caller got the raw ConnectionException instead of the snapshot this
        # function exists to provide.
        why = str(exc).lower()
        if "lock" not in why and "configuration" not in why:
            raise
        snapshot = str(Path(tempfile.mkdtemp(prefix="mock-farm-")) /
                       Path(target).name)
        out(f"{target} is held by another connection ({exc.__class__.__name__})"
            f" -- reading a snapshot copy at {snapshot}")
        shutil.copy2(target, snapshot)
        # THE WRITE-AHEAD LOG MUST COME TOO, and leaving it behind was a real
        # outage. DuckDB does not fold a commit into the database file
        # immediately; it appends to `<name>.wal` and folds it in at a
        # checkpoint, which a long-lived connection may not reach for hours.
        # A fresh deployment is the worst case: the volume showed a 24 MB
        # `nfl.duckdb` beside a 5.7 MB `nfl.duckdb.wal`, and EVERY table the
        # refresh had just written was in the second file. Copying only the
        # first produced a database that opened perfectly and contained
        # nothing --
        #
        #     Catalog Error: Table with name players does not exist!
        #
        # -- so the farm built a board of `0 pool players, 0 selectable`,
        # joined real ESPN rooms it could not pick in, and recorded a draft
        # with zero picks into the corpus. DuckDB replays the log when it
        # opens the copy, so bringing it along is the whole fix.
        wal = Path(str(target) + ".wal")
        if wal.exists():
            shutil.copy2(wal, snapshot + ".wal")
        return duckdb.connect(snapshot, read_only=True)


def farm(n: int, season: int = CURRENT_SEASON, seed: int = 0,
         db_path: str | None = None, corpus_path: str | None = None,
         min_humans: int = lobby.MIN_TEAMS_JOINED, out=print) -> dict:
    """Play up to `n` mock drafts, one at a time, and record each one.

    ONE SESSION PER PROCESS, BUT SEVERAL PROCESSES ARE NOW SAFE. One ESPN
    account can hold seats in several drafts at once -- confirmed by the
    owner -- so the only thing that ever made a second farm process dangerous
    was that it would pick the SAME room as the first. `pick_room` ranks
    deterministically, so two loops polling the same lobby agree on the best
    row; both would join it, we would hold two of the eight seats, and
    because `draft_id` is a content hash of (league_id, picks) both would
    compute the same id and the second `record()` would overwrite the first
    -- leaving one bot seat labelled `my_slot` and the OTHER bot seat
    indistinguishable, to every later reader, from a person. That is a corpus
    that quietly teaches the model somebody drafts at random.

    `pipeline.farm_claims` is the fix and this is the only place it is used:
    the rooms other live processes hold are excluded from the ranking, and
    the room this one is about to join is claimed BEFORE the invite POST and
    released in a `finally` afterwards. A claim that loses a race just means
    the next room down is tried, in the same poll, rather than a wasted wait.

    `min_humans` is the floor on `teamsJoined` -- see `espn_mock_lobby` for
    why the default is 1 rather than the 4 that would read better. Below the
    floor the loop WAITS and re-polls rather than filling an empty room with
    its own bot, and says what it saw while waiting so an unattended log
    shows whether the floor is starving the run.

    THE SHAPE IS CHOSEN BY WHAT THE CORPUS LACKS. Every pass reads the
    per-shape draft counts out of the corpus and hands them to `rank_rooms`,
    which puts the least-recorded shape with a joinable room first and then
    ranks within it exactly as before (fullest room, then experience, then
    soonest start). One line per pass names every shape, what is recorded,
    and what the lobby is offering.

    EVERYTHING PRINTED BELOW THIS LINE IS SCRUBBED. The wrap happens once,
    here, rather than at the call sites that format an exception -- there are
    six of them today, one is a bare `traceback.format_exc()`, and the next
    one somebody adds would have to remember. See `pipeline.redact`: the
    thing being kept out of this log is `SWID`, which rides in the query
    string of both the invite POST and the socket JOIN, and this log is the
    only account of an unattended night that anyone reads in the morning --
    which makes it the most likely thing in the repo to be pasted into a
    chat.
    """
    out = redact.redacting(out)
    cookies = load_cookies()
    fetch = http_fetch(cookies)
    rng = np.random.default_rng(seed)
    conn = open_board_db(db_path, out=out)
    # Statuses live in their own nested dict rather than alongside the
    # totals. Flattening them collided: `recorded` is both a status name and
    # the counter the loop's own `while` reads, so a recorded draft
    # incremented it twice and a run asked for N stopped at N/2.
    counts = {"attempted": 0, "recorded": 0, "picks": 0, "by_status": {}}
    played: set = set()
    try:
        while counts["recorded"] < n:
            try:
                rooms = lobby.list_mock_leagues(fetch, season)
            except Exception as exc:            # noqa: BLE001 -- a directory
                # read that fails is nearly always a transient HTTP blip, and
                # ending an overnight run over one is worse than waiting.
                out(f"could not read the lobby ({exc}) -- retrying in "
                    f"{LOBBY_RETRY_SECONDS:.0f}s")
                time.sleep(LOBBY_RETRY_SECONDS)
                continue
            # Rooms another live farm process is sitting in are excluded the
            # same way rooms this one has already played are: by league id,
            # before ranking. Swept of dead claims on every read, so a
            # process killed mid-draft does not fence its room off forever.
            skip = played | claims.claimed()
            # WHAT THE CORPUS IS SHORT OF, re-read every pass rather than
            # once at the top: a pass takes as long as a draft (30-40
            # minutes) and this process has just added one to a shape's
            # count, as may three other farms against the same file. Cheap
            # (one grouped count over ~900 head rows) and lock-tolerant, so a
            # poll that lands while somebody is recording gets {} and ranks
            # on fullness alone rather than failing the pass.
            recorded_by_shape = dl.shape_counts(corpus_path)
            # Every pass says what the rotation is looking at, whether or not
            # it finds a room: the shape counts are the only account of why
            # the farm chose what it chose, and the open counts are the only
            # way to tell "the corpus is short of 12-team standard" from
            # "ESPN is not serving 12-team standard tonight".
            report = lobby.lobby_report(rooms, exclude=skip)
            out(lobby.shape_line(recorded_by_shape, report))
            ranked = lobby.rank_rooms(rooms, exclude=skip,
                                      min_teams_joined=min_humans,
                                      counts=recorded_by_shape)
            # Claim, then join -- never the other way round. Two processes can
            # rank identically and both reach this line; exactly one of them
            # wins the exclusive create, and the loser drops to the next room
            # in its own ranking instead of taking a second seat in the first.
            room = next((r for r in ranked
                         if claims.claim(r["leagueId"])), None)
            if room is None:
                # Three different reasons to be here and they call for three
                # different mornings, so they are said apart rather than
                # collapsed into "nothing fits".
                if ranked:
                    out(f"another farm process holds all {len(ranked)} "
                        f"qualifying room(s) -- waiting "
                        f"{LOBBY_RETRY_SECONDS:.0f}s")
                elif report["open"] == 0:
                    out(f"no joinable snake room of any farmed shape in the "
                        f"lobby's {report['rows']} rows right now -- waiting "
                        f"{LOBBY_RETRY_SECONDS:.0f}s")
                else:
                    out(f"{report['open']} joinable room(s) of a farmed "
                        f"shape in the lobby's {report['rows']} rows, but "
                        f"the fullest holds {report['best']}/"
                        f"{report['size']} and the floor is {min_humans} "
                        f"-- waiting {LOBBY_RETRY_SECONDS:.0f}s")
                time.sleep(LOBBY_RETRY_SECONDS)
                continue
            league_id = str(room["leagueId"])
            played.add(league_id)
            counts["attempted"] += 1
            try:
                result = play_draft(conn, corpus_path, cookies, room, rng,
                                    season=season, out=out)
            except KeyboardInterrupt:
                raise
            except Exception as exc:            # noqa: BLE001 -- one room's
                # unexpected failure must not end the night's farming. The
                # traceback is printed rather than summarised: this runs
                # unattended, and the log is the only account of it there
                # will be in the morning.
                import traceback
                out(f"room {league_id} raised: {exc}")
                out(traceback.format_exc())
                result = {"status": "crashed", "league_id": league_id}
            finally:
                # On every path out, including the KeyboardInterrupt that
                # re-raises past it. A claim left behind by a process that is
                # no longer in the room costs the next poll a room it could
                # have played, and the TTL that would eventually retire it is
                # ninety minutes long.
                claims.release(league_id)
                # The live file goes with the claim, in the same breath and
                # for the same reason: by this line the draft has either been
                # recorded to the corpus or abandoned, and either way this
                # process is no longer in the room, so anything still
                # advertising it as live is a phantom. Here rather than after
                # `record_draft` inside `play_draft` deliberately -- a draft
                # that crashed, timed out, or was refused for a bad shape
                # never reaches that line, and those are exactly the runs that
                # would otherwise leave a board on the page that never moves
                # again. The staleness sweep is the backstop for a process
                # that dies before it gets here at all, not the primary path.
                clear_live(league_id)
            status = result["status"]
            counts["by_status"][status] = counts["by_status"].get(status, 0) + 1
            if status == "recorded":
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

    Run: python -m pipeline.mock_farm [N] [--seed=S] [--min-humans=H]

    `--min-humans` is the floor on how many seats a room must already hold
    before we will take one of the rest (default `lobby.MIN_TEAMS_JOINED`).
    Raising it in the evening, when ESPN's own recommended draft times are,
    buys a more human corpus; raising it overnight starves the run. See
    `espn_mock_lobby`'s docstring for the measurement.

    Several of these can now run at once against the same account: rooms are
    claimed through `pipeline.farm_claims`, so two processes never sit in the
    same draft.
    """
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a.split("=")[0]: a.partition("=")[2]
             for a in argv[1:] if a.startswith("--")}
    n = int(args[0]) if args and args[0] else 1
    seed = int(flags.get("--seed") or 0)
    min_humans = int(flags.get("--min-humans") or lobby.MIN_TEAMS_JOINED)
    counts = farm(n, seed=seed, min_humans=min_humans)
    print(json.dumps(counts, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
