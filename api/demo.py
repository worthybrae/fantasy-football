"""A real ESPN mock draft, live, as the thing the landing page opens with.

WHY THIS EXISTS. Every draft tool's landing page shows a screenshot of itself.
This one has something better available: `pipeline/mock_farm.py` sits in real
ESPN mock drafts around the clock to build the corpus the model is fitted on,
and those drafts are happening WHILE somebody is reading the page. So the hero
is not a picture of the product, it is the product, running, against picks
that landed seconds ago against real people.

WHAT IT SERVES, AND WHAT IT REFUSES TO INVENT:

  * The picks are the farm's own record (`data/farm-live/*.json`), which is
    written as the socket receives them.
  * The names, positions and prices are this project's own board -- the same
    `cached_build_board` every other endpoint reads, so the numbers on the
    landing page are the numbers in the product and cannot drift from them.
  * The recommendation is `api.live.rank_and_plan` over what is genuinely still on
    the board in that room, for the team genuinely on the clock.

There is no "demo mode" anywhere in here. If the farm is not running, this
answers `live: false` and the page says so rather than replaying something
old dressed up as live.

CACHED HARD, because this is the one endpoint a stranger hits. The farm writes
a file per pick; a landing page polling it does not need to be closer to the
truth than a few seconds, and the board work behind a recommendation is real
work. One build per cache window, shared by every visitor.

READ-ONLY AND ANONYMOUS. It touches no session, sets no cookie, and names no
person: the seats are "slot 4", the drafters are strangers in a public mock,
and the only identity involved is the farm's own seat.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import threading
import time

import numpy as np
import pandas as pd

from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from api import http_cache
from pipeline.db import get_conn
from scoring.board_cache import cached_build_board
from scoring.draft_sim import snake_slots
from scoring.headshot import thumb

# Games a team plays. The same divisor the room's own Proj/G column uses, so
# the landing page and the product cannot mean two things by "per game".
SEASON_GAMES = 17

# Where `pipeline/mock_farm.py` writes what it is watching, one file per
# league. Read with `FARM_LIVE_DIR` -- the farm's own variable, not a second
# one -- so pointing the farm somewhere else cannot leave this reading an
# empty directory and reporting "nothing is drafting" while it drafts.
#
# THIS DIRECTION IS ONE-WAY AND MUST STAY THAT WAY. The farm is the process
# doing the work that matters; this endpoint is a page decoration. It only
# ever reads, it holds no lock, and the farm writes each file as a tmp plus
# `os.replace` (atomic), so a read landing mid-write sees the old file whole
# rather than half of the new one. A file this endpoint cannot parse is
# skipped, never repaired.
FARM_DIR = os.environ.get("FARM_LIVE_DIR", "data/farm-live")

# How stale a farm file may be before the draft behind it is treated as over.
# The farm rewrites on every pick, and a mock draft with nobody picking still
# autodrafts within the clock (30s in every lobby room observed), so a file
# untouched for this long is a room that has finished -- or a seat whose
# socket has died and stopped hearing picks (see mock_farm.SOCKET_DEAD_SECONDS),
# which must not be shown as "running now" with a board frozen at pick 60.
STALE_SECONDS = 120.0

# The CEILING on how long a built answer is reused, not the resolution the
# page runs at. What actually invalidates this cache is the room moving:
# entries are keyed on `_identity` -- (which room, which pick) -- so a pick
# landing retires the answer that predates it immediately, and every reader
# between two picks shares one build. This number only bounds how long a
# QUIET room's answer stands, since the board underneath it can be refreshed
# by other work.
#
# That keying is what lets the page be pushed rather than polled: the event
# stream below wakes readers the moment the identity changes, and the fetch
# it wakes them for cannot be served a stale answer.
#
# Thirty rather than the twelve this started at, for the one case that costs:
# during the owner's own draft the `drafted` table moves every pick, which
# invalidates the board cache this reads, so a short window here would mean a
# board build per pick competing with the ranking the drafter is waiting on.
# Thirty seconds is still well inside one seat's clock.
CACHE_SECONDS = 30.0

# The floor under rebuilds. A build is real work -- an availability lookup and
# a ranking, measured at 2.8-4.0s warm -- and the identity key above would
# otherwise put one behind every pick. Most of the time that is what you want
# (a pick every 30-90 seconds), but a room whose empty seats are all
# autodrafting fires picks seconds apart, and chasing those would keep a core
# busy for as long as the room lasts. Eight seconds is under any pace a person
# picks at, so a room with people in it is never held back by this; a burst of
# autodrafts arrives on screen as a group instead of one at a time.
MIN_REBUILD_SECONDS = 8.0

# How long "nothing is drafting" is reused, which is deliberately NOT the
# same. The farm rotates rooms: a draft finishes, its file goes, and a new
# room's file appears seconds later -- and a visitor who lands in that gap
# would otherwise be shown an empty hero for a full cache window. Measured
# exactly that way: a server started mid-rotation cached the empty answer and
# the landing page had a hole in it for thirty seconds.
#
# A negative answer is also cheap to recompute (it is a directory listing --
# no board, no pool, no rollout), so there is nothing to protect.
EMPTY_CACHE_SECONDS = 5.0

# WHERE THE PAGE KEEPS A ROOM TO FALL BACK ON. The farm deletes a room's file
# the instant its draft ends (`mock_farm.clear_live`), which is correct -- it
# publishes rooms it is PLAYING -- and leaves this page with nothing to show
# between one room finishing and the next reaching its fourth pick. Measured
# on a running farm, that gap is minutes long and lands on a visitor as a
# black page under a card claiming a draft is happening behind it.
#
# So the endpoint keeps its own copy. Every live room it serves is archived
# here in full, and when no room is live the page replays the freshest of
# them, pick by pick, at the pace the picks were actually made. Nothing is
# generated: a replayed draft is a real ESPN mock with real people in it,
# shown a few minutes after the fact instead of during -- and the page says
# which of the two it is showing rather than letting a reader assume.
REPLAY_DIR = os.environ.get("DEMO_REPLAY_DIR", "data/demo-replay")

# How many rooms the archive holds. More than one so a reader who stays
# through a gap, or comes back an hour later, is not shown the same draft
# again; few enough that the directory stays a handful of files.
REPLAY_KEEP = 5

# The floor and ceiling on a replayed pick's dwell. Real pick times are used
# -- the corpus records them and so does the archive -- but the extremes are
# unwatchable at both ends: a first-round pick somebody made in 2 seconds
# would flash past (and would have this endpoint rebuilding a board every 2
# seconds), and a seat that sat on its clock for the full 90 leaves the page
# apparently frozen. Clamped, not replaced.
REPLAY_MIN_SECONDS = 8.0
REPLAY_MAX_SECONDS = 45.0

# How much of the room the hero shows: the last few picks, and the few players
# the board would take next. Both are "enough to see the shape of it" rather
# than a full board, which is what the product itself is for.
RECENT_PICKS = 8

# How much of the ranked board to serve. This is not a teaser any more: the
# landing page renders the room's OWN `AvailableList` against it, so the list
# has to be long enough to scroll, filter and sort like the real thing. Sixty
# is what the room shows above the fold plus room to move.
SHORTLIST = 60


_CACHE: dict = {}
_LOCK = threading.Lock()

# The simulation pool, kept because building one is 2.2-3.6s (see
# `mock_farm.build_tool`) and it changes only when the board or the league's
# settings do. Keyed on the board's own identity so a refreshed board builds a
# fresh pool rather than ranking against last week's players.
_POOL: dict = {}


def _cached_pool(conn, board, settings):
    key = (id(conn), len(board), league_key(settings))
    with _LOCK:
        hit = _POOL.get(key)
    if hit is not None:
        return hit
    from scoring.draft_sim import build_pool
    pool = build_pool(conn, board, settings)
    with _LOCK:
        _POOL.clear()          # one league, one pool: never a growing map
        _POOL[key] = pool
    return pool


def league_key(settings) -> str:
    from scoring import league
    return league.to_json(settings)


def _live_files(now: float) -> list:
    """The farm's records, freshest first, ignoring rooms that have gone quiet."""
    out = []
    for path in glob.glob(os.path.join(FARM_DIR, "*.json")):
        try:
            stat = os.stat(path)
        except OSError:
            continue
        if now - stat.st_mtime > STALE_SECONDS:
            continue
        out.append((stat.st_mtime, path))
    return [path for _mtime, path in sorted(out, reverse=True)]


def _record(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _pick_of(record: dict) -> int:
    return len(record.get("picks") or [])


# The rounds worth showing a stranger. Not an aesthetic preference -- it is
# where this tool has anything to say. Through the first eight rounds the board
# is choosing between real starters and the gap between them is tens of points;
# by round fourteen the only players left are kickers and defences, every price
# is zero, and a hero built on that would be advertising the product at its
# least useful moment. Measured on a live farm draft: round 14 offered "Cameron
# Dicker, +2" as its best available.
#
# This is where a room is PICKED UP from. How long it is then kept is
# `STICKY_PICKS` below.
INTERESTING_ROUNDS = 8

# And not the first pick either: an empty board is eight empty columns and a
# recommendation nobody has had to make a decision against yet.
WARMED_UP_PICKS = 4

# The round past which a room is shown REWOUND rather than where it stands
# (see `_moment`). Later than `INTERESTING_ROUNDS` on purpose: a room picked
# up while it was interesting is worth following a couple of rounds past that
# as it stands, because a reader watching a draft is watching a story and the
# story survives the board thinning out. Past this round it does not, and the
# room is shown at the moment it had something to say instead.
#
# This used to decide when the page LEFT a room as well, which is now
# `STICKY_PICKS`.
REWIND_ROUND = 10

# HOW LONG A ROOM GETS THE HERO, in that room's own picks. This is the whole
# of the rule now: the page shows a draft, keeps it for ten of its picks, and
# then hands the hero to the next live room. It does not weigh the rooms up on
# every call and take whichever is winning.
#
# WHY IT IS A ROTATION AND NOT A WINNER. The farm sits in several rooms at
# once and they run neck and neck, so "the furthest along" is a different room
# after nearly every pick in the building -- and a page that re-chose on every
# poll was a slideshow of strangers rather than a draft. The room it is on
# never got long enough on screen for a reader to see a decision get made and
# answered.
#
# Ten because it is about a round of a lobby room (eight to twelve seats):
# long enough to watch a turn come round, short enough that no single draft
# owns the front page and every room the farm is in gets shown.
#
# The turn ends early for the two things that end a draft rather than pause
# it -- the room finishing, and its file going quiet -- and for nothing else.
STICKY_PICKS = 10

# The room the page is currently showing: which room, the pick it was on when
# it was chosen, when that was, and which rooms have already had a turn in the
# rotation now under way. Without this memory the hero changed drafts on
# nearly every poll, which is what a reader reported and what the rotation
# tests reproduce against the old rule.
#
# Process-local, seeded from the last persisted answer at startup (see
# `register_demo_routes`) so a restart is not itself a jump.
_SHOWING: dict = {}
_SHOWING_LOCK = threading.Lock()


def _round_of(record: dict) -> int:
    """The round the room is in now, one-based; 0 for a record with no teams."""
    teams = int(record.get("teams") or 0)
    return _pick_of(record) // teams + 1 if teams else 0


def _unfinished(record: dict) -> bool:
    total = int(record.get("teams") or 0) * int(record.get("rounds") or 0)
    return bool(total) and _pick_of(record) < total


def _held(record: dict, since, chosen_at, now: float) -> bool:
    """Whether the room on screen is still inside its turn.

    Spent by the room making `STICKY_PICKS` picks, and ended early by the two
    things that end a draft rather than pause it: the room finishing, and its
    file going quiet (which `_live_files` has already dealt with by the time
    this is asked -- a quiet room is not among the records at all).

    The clock is the backstop for the third case, a room whose file keeps
    being rewritten while its picks have stopped. A turn that can only be
    spent in picks would never be spent there, and the page would sit on a
    frozen room for as long as something kept touching the file. So a turn
    that has not seen a single pick land is given `STALE_SECONDS` -- this
    module's own answer to "a room that has not moved is over" -- and no more.
    """
    if since is None:
        return False
    made = _pick_of(record) - int(since)
    if made >= STICKY_PICKS or not _unfinished(record):
        return False
    return made > 0 or now - float(chosen_at or 0.0) <= STALE_SECONDS


def _in_the_interesting_part(record: dict) -> bool:
    """Whether this room is in the rounds worth putting in front of a
    stranger: warmed up, and not yet past `INTERESTING_ROUNDS`."""
    if not int(record.get("teams") or 0) or _pick_of(record) < WARMED_UP_PICKS:
        return False
    return _round_of(record) <= INTERESTING_ROUNDS


def _next_up(live: list, seen: set):
    """Whose turn it is now, given the rooms that have already had one.

    A TIER, AND THEN A ROUND ROBIN INSIDE IT, in that order.

    The tier is the rooms in their interesting rounds, and it comes first
    because a turn spent outside it is a wasted one either way: a room in
    round twelve is drawn REWOUND to round three for the whole of its turn
    (see `_moment`), so it sits frozen on screen for ten picks the reader
    cannot see, and a room two picks in is an empty board with a
    recommendation nobody has had to make a decision against yet. Showing the
    interesting room a second time is a better page than either. A room
    outside the tier is not shut out, it is waiting -- it joins the moment it
    warms up, and it is all there is to show when nothing is in the tier.

    Inside the tier it is a round robin in league id order, because the order
    has to be the same answer every time it is asked and a league id is the
    only thing about a room that does not move while it drafts. Ordering on
    picks or on mtime is what the page was doing before, and those change
    under it constantly. A room that has not had a turn in the rotation now
    under way goes first, so the page works its way round the farm rather
    than settling on a pair.

    When everybody in the tier has had a turn the rotation starts again,
    which is why the caller clears `seen` on the answer this gives from a
    full house.
    """
    order = sorted(live, key=lambda r: str(r.get("league_id") or ""))
    tier = [r for r in order if _in_the_interesting_part(r)] or order
    pool = [r for r in tier
            if str(r.get("league_id") or "") not in seen] or tier
    return pool[0] if pool else None


def _liveliest(records: list, now: float | None = None):
    """The room a stranger should be shown.

    THE ROOM ALREADY ON SCREEN, for as long as its turn runs -- ten of its own
    picks, see `STICKY_PICKS`. Then the next room in the rotation, see
    `_next_up`. With one room live there is nothing to rotate to and it simply
    keeps its place.

    A room that finishes, or whose farm file goes quiet, is left the moment it
    does: it is not among the records, or not unfinished, and either way the
    turn is over rather than waiting out its ten picks.
    """
    now = time.time() if now is None else now
    with _SHOWING_LOCK:
        showing = _SHOWING.get("league_id")
        since = _SHOWING.get("since_pick")
        chosen_at = _SHOWING.get("chosen_at")
        seen = set(_SHOWING.get("seen") or ())
        live = [r for r in records if _unfinished(r)]
        if showing is not None:
            for record in live:
                if (str(record.get("league_id")) == showing
                        and _held(record, since, chosen_at, now)):
                    return record

        # Only rooms still drafting are in the rotation, and only they are
        # remembered as having had a turn: a set that kept the ids of rooms
        # that have finished would grow all day and would eventually claim
        # every live room had already been shown.
        seen &= {str(r.get("league_id") or "") for r in live}
        chosen = _next_up(live, seen)
        if chosen is None:
            _SHOWING.clear()
            return None
        league_id = str(chosen.get("league_id") or "")
        if league_id in seen:
            seen = set()        # round complete; everybody goes again
        _SHOWING["league_id"] = league_id
        _SHOWING["since_pick"] = _pick_of(chosen)
        _SHOWING["chosen_at"] = now
        _SHOWING["seen"] = seen | {league_id}
        return chosen


def _board_frame(conn):
    """The project's own board, from the cache every other endpoint uses.

    WHAT THIS COSTS, AND WHO PAYS. A cache miss is a real build (1.6-1.9s of
    CPU, measured in `scoring/board_cache.py`), and this endpoint is the one a
    stranger can hit. Two things keep that from landing on work that matters:

      * The cache key includes the `drafted` state, and nothing about a mock
        draft in the farm touches this database's `drafted` table -- the farm
        runs in its own processes against its own snapshot (see
        `mock_farm.open_board_db`, which copies the file precisely because the
        API holds the write lock). So while somebody browses, the key is
        stable: one build, then hits.
      * The endpoint's own cache means a burst of visitors is one build
        between them, not one each.

    The case that DOES cost something is the owner's own live draft, where
    `drafted` moves every pick and each pick therefore invalidates this key.
    That is why `CACHE_SECONDS` is what it is: a board rebuilt at most once a
    window, on a machine already doing the ranking, rather than once per
    visitor per pick.
    """
    return cached_build_board(conn)


def _on_the_clock(record: dict) -> int | None:
    """Whose turn it is, from the draft's own order rather than the socket.

    `snake_slots` is the same function the room uses, so the seat named here
    is the seat the product would name. One-based, and None once the board is
    full.
    """
    teams = int(record.get("teams") or 0)
    rounds = int(record.get("rounds") or 0)
    made = _pick_of(record)
    order = snake_slots(teams, rounds)
    return int(order[made]) if made < len(order) else None


def _text(value):
    """A string, or None -- never a NaN or a float on the wire."""
    if value is None or (not isinstance(value, str) and value != value):
        return None
    return str(value) or None


def _num(value, digits=0):
    """A float, rounded, or None -- never a NaN on the wire."""
    try:
        if value is None or value != value:      # NaN is the only x != x
            return None
        return round(float(value), digits) if digits else round(float(value))
    except (TypeError, ValueError):
        return None


def _board_payload(record: dict, picks: list, board, slot: int | None) -> dict:
    """The room's grid, in `/api/live/board`'s own shape.

    Served so the landing page can render `DraftBoardGrid` and `PickTicker` --
    the room's actual components -- rather than a second drawing of the same
    thing. Every field here is one those components already read; nothing is
    invented for the page.

    `my_slot` is null and every column's `is_me` is false, which is the honest
    answer: this is somebody else's draft and no seat in it belongs to the
    reader. The components already handle that (it is the browser-observer
    case), so no demo-only branch is needed anywhere in them.
    """
    names = {}
    for row in board.itertuples():
        names[str(getattr(row, "player_id", ""))] = row

    teams = int(record.get("teams") or 0)
    cells = []
    for pick in picks:
        pid = str(pick.get("player_id"))
        source = names.get(pid)
        if source is None:
            continue
        overall = int(pick.get("pick_no") or 0)
        cells.append({
            "overall": overall,
            "round": (overall - 1) // teams + 1 if teams else 1,
            "slot": int(pick.get("slot") or 0),
            "player": {
                "player_id": pid,
                "name": getattr(source, "name", None),
                # Already sized -- this is a board row and the board sizes
                # its own column (scoring/board.py). `thumb` is idempotent, so
                # a board that arrived from anywhere else is sized too.
                "headshot": thumb(_text(getattr(source, "headshot", None))),
                "position": getattr(source, "position", None),
                "team": getattr(source, "team", None),
                "bye": _num(getattr(source, "bye", None)),
                "overall_rank": _num(getattr(source, "rank", None)),
                "tier": _num(getattr(source, "tier", None)),
                "market_rank": _num(getattr(source, "market_rank", None)),
                "espn_ppr_rank": _num(getattr(source, "espn_ppr_rank", None)),
                # The rest of `BoardPlayer`, present even where this demo has
                # nothing to put in them. The room's components read these as
                # `number | null` and guard on `!== null` -- an ABSENT key is
                # `undefined`, which walks straight past that guard and into
                # `.toFixed()`, and the throw lands inside a click handler, so
                # the pick simply refused to open its profile. Null is the
                # answer that a guard can read.
                "vor": _num(getattr(source, "vor", None), 1),
                # This board projects a season; the room prints per game, on
                # the same 17-game denominator `_shortlist` already divides by
                # just below. Not ESPN's own projection, which is what the
                # live room serves here -- the farm's records carry no ESPN
                # figures, and the model's own number is the one this page is
                # showing off anyway.
                "proj_ppg": _num((getattr(source, "proj_points", None) or 0)
                                 / SEASON_GAMES, 1) or None,
                # Last season, which nothing in the farm's records or this
                # board carries: the grid's tooltip prints these two only when
                # they are there, and inventing them from a projection would
                # be printing this year's guess as last year's result.
                "last_ppg": None,
                "last_points": None,
                # The room draws this as the pick's move against the market.
                # The SAME arithmetic `api/live.py` does, in the same order:
                # where he actually went minus where the market had him, so
                # a positive number is a player who FELL (a steal) and a
                # negative one a player taken early (a reach). This was
                # written the other way round for a while, and every arrow
                # on the demo rail pointed the wrong way until the ticker's
                # tooltip printed both numbers beside the verdict.
                "value": (None if _num(getattr(source, "market_rank", None)) is None
                          else round(overall - _num(getattr(source, "market_rank", None)))),
            },
            # Nobody is named. A public mock's seats are strangers, and the
            # farm's own seat is not worth pointing at.
            # `PickMaker` is a bare string in the room's own contract --
            # 'human' | 'auto' | ... -- not an object. The farm records which
            # of the two this was, and the grid already labels both.
            "made_by": "auto" if pick.get("autodrafted") else "human",
        })

    return {
        "active": True,
        "teams": teams,
        "rounds": int(record.get("rounds") or 0),
        "my_slot": None,
        "on_the_clock": slot,
        "picks_made": len(picks),
        "columns": [{"slot": i + 1, "team_name": f"Seat {i + 1}", "is_me": False}
                    for i in range(teams)],
        "cells": cells,
    }


def _roster_payload(picks: list, slot: int | None, board, settings: dict) -> list:
    """What the seat on the clock has drafted, in the roster panel's own shape.

    Starters first in the room's own order, then the bench -- every slot the
    seat will fill by the end of the draft, empty ones included, so the panel
    reads as a roster filling up rather than as a list of picks. `urgent` is
    left false throughout: it means "you need this and the draft is running
    out", which is a claim about a reader's own team and this is not one.
    """
    order = ["QB", "RB", "WR", "TE", "K", "DST"]
    starters = settings.get("starters") or {}
    names = {str(getattr(r, "player_id", "")): r for r in board.itertuples()}

    mine = []
    for pick in picks:
        if slot is not None and pick.get("slot") != slot:
            continue
        source = names.get(str(pick.get("player_id")))
        if source is None:
            continue
        mine.append({
            "player_id": str(pick.get("player_id")),
            "name": getattr(source, "name", None),
            "position": getattr(source, "position", None),
            "team": getattr(source, "team", None),
            "bye": _num(getattr(source, "bye", None)),
            "proj_points": _num(getattr(source, "proj_points", None)),
            # Week 1, the number the rail actually prints (see the live room's
            # own `_my_roster`). Served here too so the landing page's rail and
            # the room's rail cannot show different currencies for the same
            # player.
            "wk1_points": _num(getattr(source, "proj_wk1", None), 1),
        })

    starter_slots, used = [], set()
    for position in order:
        for n in range(int(starters.get(position, 0) or 0)):
            label = position if starters.get(position, 0) == 1 else f"{position}{n + 1}"
            taken = next((p for i, p in enumerate(mine)
                          if i not in used and p["position"] == position), None)
            if taken is not None:
                used.add(mine.index(taken))
            starter_slots.append({"slot": label, "player": taken, "urgent": False})
    for n in range(int(settings.get("flex_slots") or 0)):
        taken = next((p for i, p in enumerate(mine)
                      if i not in used and p["position"] in ("RB", "WR", "TE")), None)
        if taken is not None:
            used.add(mine.index(taken))
        starter_slots.append({"slot": f"FLEX{n + 1}" if settings.get("flex_slots", 0) > 1 else "FLEX",
                              "player": taken, "urgent": False})
    # FLEX sits with the RB/WR group it is actually filled from, not stranded
    # after DST where the loops above happen to leave it -- the same order the
    # live room's rail reads in (`SLOT_DISPLAY_ORDER` in web/src/pages/
    # DraftRoom.tsx, which sorts its own slots exactly this way and for the
    # owner's exact same reason). Two rooms drawing the same panel had two
    # different roster orders, and this one was the odd one.
    #
    # Sorted AFTER assignment, never by reordering the loops above: `order`
    # and the flex loop decide which slot a player lands in (dedicated slots
    # first, FLEX only from what is left over), so reordering them to change a
    # display would be changing an assignment rule. By here every player is
    # already placed. `sorted` is stable, so slots sharing a rank keep the
    # order they were built in -- RB1 above RB2, FLEX1 above FLEX2.
    display = ["QB", "RB", "WR", "FLEX", "TE", "K", "DST"]

    def rank(entry) -> int:
        base = entry["slot"].rstrip("0123456789")
        return display.index(base) if base in display else len(display)

    slots = sorted(starter_slots, key=rank)
    # THE BENCH IS PART OF THE ROSTER WHETHER OR NOT IT IS FULL. Empty bench
    # rows were missing here, so the panel showed ten slots in a room where
    # every seat drafts sixteen players -- it read as a complete roster at
    # 5/10 when the seat was five picks into a sixteen-round draft, and the
    # room's own rail never does that.
    #
    # The count comes from the ROOM (rounds, minus the starters above),
    # because that is how many players these seats are actually going to
    # draft. The starters' shape is this deployment's, for the reason
    # `_settings_payload` gives; the bench is whatever is left of the room's
    # rounds after them.
    overflow = [p for i, p in enumerate(mine) if i not in used]
    bench_slots = max(int(settings.get("rounds") or 0) - len(slots),
                      len(overflow))
    for n in range(bench_slots):
        slots.append({"slot": f"BN{n + 1}",
                      "player": overflow[n] if n < len(overflow) else None,
                      "urgent": False})
    return slots


def _settings_payload(conn, teams: int, rounds: int,
                      fmt: str | None = None) -> dict:
    """The league shape, in `/api/live/state`'s own `settings` shape.

    Served so the landing page can hand it to the same `AvailableList` the
    draft room uses. The starters come from this deployment's league (that is
    what the board was priced against, so a different roster here would label
    tiers the prices do not match); the teams, rounds and scoring format come
    from the ROOM, because that is what the picks below are actually being
    made in. A room that did not say its format -- an older live file -- gets
    this deployment's, which is what every room got before the farm recorded
    anything but PPR.
    """
    from scoring import league
    settings = league.load(conn)
    return {
        "teams": teams or settings.teams,
        "rounds": rounds or settings.rounds,
        "starters": dict(settings.starters),
        "flex_slots": settings.flex_slots,
        "bench": settings.bench,
        "scoring_format": str(fmt) if fmt else league.scoring_format(settings),
    }


def _seat_counts(picks: list, slot: int | None, positions: dict) -> dict:
    """What the seat on the clock has already drafted, by position.

    Read off the room's own picks rather than assumed, because it is what
    decides the order below: a seat holding three receivers wants a back, and
    a board that ignored that would rank players for nobody in particular.
    """
    counts: dict = {}
    for pick in picks:
        if slot is not None and pick.get("slot") != slot:
            continue
        position = positions.get(str(pick.get("player_id")))
        if position:
            counts[position] = counts.get(position, 0) + 1
    return counts


def _ranked(conn, board, picks: list, slot: int | None, limit: int,
            teams: int | None = None, rounds: int | None = None,
            fmt: str | None = None) -> list:
    """The room's own ranked list, for the seat on the clock.

    THE RAW BOARD IS THE WRONG LIST, and this is the difference between
    showing the product and showing a spreadsheet. Ordered by value alone,
    the best available at pick 60 of an 8-team draft is a KICKER -- worth
    twelve points over replacement and worth nothing to anybody, because no
    roster needs one for another six rounds. So the hero runs the same
    functions the draft room runs (`api.live.rank_and_plan`: ESPN's order,
    who is still there at this seat's next turn counted from recorded
    drafts, what waiting would cost, and `scoring/plan.target_now` for the
    three the cards show), against the seat genuinely on the clock and the
    roster it genuinely holds. Rows carry the room's own candidate keys, so
    the landing page hands them to the same component, plus the few things
    the room reads from its join table (name, team, bye, adp, board_rank).

    The three `target_now` picks lead the list, in their order; the rest
    follow in ESPN order. `limit` caps the whole. `teams`/`rounds` are the
    ROOM's shape, so a seat's turns are the room's and not the stored
    league's; the stored league still supplies the roster shape.

    `fmt` is the room's scoring, off the live file (`mock_farm.live_payload`),
    and it goes with `teams` into the counted table: "will he last" is
    answered from drafts of this size and this scoring once the corpus holds
    enough of them. None -- an older live file, or a room that never said --
    falls back to this deployment's own league, which is the same assumption
    `_settings_payload` already makes about the roster.
    """
    from api import live as live_mod
    from scoring import league
    from scoring.availability import cached_table
    from scoring.plan import health_level, target_now

    settings = league.load(conn)
    # The ROOM's scoring if it said, this deployment's league otherwise.
    shape_format = str(fmt) if fmt else league.scoring_format(settings)
    if teams or rounds:
        # `rounds` is not a field but the roster's size (starters + flex +
        # bench), so the room's round count is set through its bench. The
        # reshaped settings are what `_cached_pool` keys on too, and that
        # cache is a SINGLE slot (it clears before every insert): a second
        # shape evicts the first, and two rooms of different shapes shown
        # in turn would rebuild the pool on every alternation. Today every
        # farm room is the same shape, so the slot holds; a second shape is
        # a reason to give `_cached_pool` more than one.
        import dataclasses
        lineup = sum(int(v) for v in (settings.starters or {}).values()) \
            + int(settings.flex_slots or 0)
        settings = dataclasses.replace(
            settings, teams=int(teams or settings.teams),
            bench=max(0, int(rounds or settings.rounds) - lineup))
    pool = _cached_pool(conn, board, settings)
    if isinstance(board, pd.DataFrame) and "espn_rank" not in board.columns and conn is not None:
        board = live_mod._attach_espn_rank(conn, board)
    positions = {str(pid): str(pos) for pid, pos in zip(pool.player_id, pool.position)}
    index_by_player = {str(pid): i for i, pid in enumerate(pool.player_id)}
    taken_ids = {str(p.get("player_id")) for p in picks}
    taken = np.array([str(pid) in taken_ids for pid in pool.player_id], dtype=bool)
    taken_order = [index_by_player[str(p.get("player_id"))]
                   for p in picks if str(p.get("player_id")) in index_by_player]
    counts = _seat_counts(picks, slot, positions)
    mine = [index_by_player[str(p.get("player_id"))] for p in picks
            if slot is not None and p.get("slot") == slot
            and str(p.get("player_id")) in index_by_player]

    table = cached_table()
    # No plan for the landing room -- nobody is drafting from it -- so the
    # rows and the turns are all it asks for.
    rows, _plan, turns = live_mod.rank_and_plan(
        board, pool, taken, taken_order, counts, mine, int(slot or 1),
        # `settings` carries the room's teams and rounds (reshaped above) but
        # this deployment's SCORING, since the board was priced against it --
        # so `rank_and_plan` would read the wrong half of the shape off it.
        # The room's own format goes through `fmt`.
        settings, frozenset(), table, picks_made=len(picks), with_plan=False,
        fmt=shape_format)

    by_id = {}
    if isinstance(board, pd.DataFrame):
        for row in board.itertuples():
            by_id[str(getattr(row, "player_id", ""))] = row
    by_row = {r["player_id"]: r for r in rows}

    # The cards: the same score the plan uses, with everyone available now.
    leaders = []
    if rows and turns:
        ids = [r["player_id"] for r in rows]
        col = lambda name: np.array([  # noqa: E731 -- one-liners over the rows
            (np.nan if r.get(name) is None else float(r[name])) for r in rows], dtype=float)
        games = np.array([
            (np.nan if by_id.get(pid) is None else
             float(getattr(by_id[pid], "career_games_pg", np.nan) or np.nan))
            for pid in ids], dtype=float)
        try:
            leaders = [t["player_id"] for t in target_now(
                proj=col("proj_points"), positions=np.array([r["position"] for r in rows], dtype=object),
                player_ids=ids, espn_rank=col("espn_rank"), espn_adp=col("espn_adp"),
                market_rank=col("market_rank"),
                byes=np.array([(np.nan if by_id.get(pid) is None else
                                float(getattr(by_id[pid], "bye", np.nan) or np.nan))
                               for pid in ids], dtype=float),
                health=health_level(games), roster_counts=dict(counts),
                settings=settings, turns=turns, picks_made=len(picks),
                favourites=set(), table=table,
                teams=int(settings.teams), fmt=shape_format,
                names={pid: str(getattr(by_id[pid], "name", pid)) if by_id.get(pid) is not None else pid
                       for pid in ids})]
        except Exception:      # noqa: BLE001 -- the cards are a garnish on
            # a list that stands on its own; a scoring failure costs them,
            # not the page
            leaders = []

    ordered = [by_row[pid] for pid in leaders if pid in by_row]
    ordered += [r for r in rows if r["player_id"] not in set(leaders)]

    out = []
    for entry in ordered[:limit]:
        pid = entry["player_id"]
        source = by_id.get(pid)
        adp = _num(getattr(source, "market_rank", None)) if source is not None else None
        rank = _num(getattr(source, "rank", None)) if source is not None else None
        out.append({
            **entry,
            # And the few things the room reads from its own join table, which
            # this page fetches separately (`/api/players`): served here too so
            # a reader gets names even before that lands.
            "name": getattr(source, "name", None) if source is not None else None,
            "team": getattr(source, "team", None) if source is not None else None,
            "bye": _num(getattr(source, "bye", None)) if source is not None else None,
            "adp": adp,
            "vs_adp": None if (rank is None or adp is None) else round(adp - rank),
            "board_rank": rank,
        })
    return out


def _shortlist(board, taken: set, limit: int = SHORTLIST) -> list:
    """The board itself: what is still available, in this tool's own order.

    NOT A TOP FOUR AND NOT A TEASER. This is the ranked available list the
    draft room draws, cut to the top of it, with the columns that make it an
    argument rather than a leaderboard:

      `rank`      where this tool puts him, which is the whole product;
      `adp`       where the market puts him, so the two can disagree in
                  public -- a row where they differ by forty picks is the
                  clearest thing on the page;
      `proj_g`    points a game, because a season total is a number nobody
                  has intuitions about;
      `over_replacement` what he is worth against what the position would
                  still offer later, which is the number no other board
                  publishes at all.

    Ranked by the board's own order rather than re-sorted here: a landing page
    that ordered players differently from the room would be advertising a
    product that does not exist.
    """
    rows = []
    for row in board.itertuples():
        pid = getattr(row, "player_id", None)
        if pid is None or str(pid) in taken:
            continue
        vor = _num(getattr(row, "vor", None))
        if vor is None:
            continue
        adp = _num(getattr(row, "market_rank", None))
        rank = _num(getattr(row, "rank", None))
        rows.append({
            "player_id": str(pid),
            "name": getattr(row, "name", None),
            "position": getattr(row, "position", None),
            "team": getattr(row, "team", None),
            "bye": _num(getattr(row, "bye", None)),
            "rank": rank,
            "adp": adp,
            # Positive where the market would let him fall past this board's
            # own price -- the direction a drafter is shopping in. None where
            # either number is missing rather than a zero that would read as
            # "the market agrees".
            "vs_adp": None if (rank is None or adp is None) else round(adp - rank),
            "proj_g": _num(getattr(row, "proj_points", None) / SEASON_GAMES
                           if _num(getattr(row, "proj_points", None)) else None, 1),
            "over_replacement": vor,
        })
        if len(rows) >= limit:
            break
    return rows


def _moment(record: dict) -> tuple:
    """Which pick of this draft to show, and whether that is where it is now.

    A live room is shown live. A room already past its useful rounds is shown
    REWOUND to the end of its third round -- the same draft, the same real
    people, the same board, at the moment the tool had something to say.

    The alternative was to show whatever instant the room happens to be in
    when a stranger loads the page, which for a draft in round fifteen is four
    defences being autodrafted and a best-available kicker worth two points.
    That is a true picture of the wrong thing.

    The rewind is stated in the payload (`at_pick`, `live_moment`) so the page
    can label it, because "a real draft happening now" and "round 3 of a real
    draft happening now" are different claims and only one of them is true
    here.
    """
    teams = int(record.get("teams") or 0)
    picks = record.get("picks") or []
    made = len(picks)
    # Live up to the rewind round, not the interesting one: a room the page
    # picked up in round eight (see `_liveliest`) is shown where it is while
    # its turn runs on into round nine.
    if not teams or made // teams + 1 < REWIND_ROUND:
        return made, True
    return min(made, teams * 3), False


# HOW OFTEN THE EVENT STREAM LOOKS. Not how often it speaks: the loop below
# wakes on this interval, stats the farm's files, and says nothing unless the
# room actually moved. A quarter of a second would be a nicer number and buys
# nothing -- these picks are made by people with a minute and a half to think.
EVENT_TICK = 1.0

# A comment down an idle stream, because something between here and the reader
# will eventually close a connection that has been silent too long. Every
# proxy's timeout is different; twenty seconds is under all the common ones.
EVENT_KEEPALIVE = 20.0

# The longest a stream is held open before the client is asked to reconnect.
# A browser tab left open for a week should not pin a server-side generator
# for a week: EventSource reconnects on its own, so ending the response is
# free, and the reconnect re-reads the room from scratch.
EVENT_MAX_SECONDS = 900.0


def _archive(record: dict) -> None:
    """Keep this room, so the page has something real to show when the farm
    has nothing.

    Written on the way past: the endpoint has already read and chosen this
    record, so archiving it is one file write per pick and no extra reading.
    Only rooms that are worth replaying are kept -- far enough in to have a
    board, not so far that the replay would open in the rounds where every
    price is zero.

    Atomic, for `write_live`'s reason: this file is rewritten constantly and a
    reader that caught it half-written would see a draft with its last pick
    cut in half.
    """
    picks = record.get("picks") or []
    teams = int(record.get("teams") or 0)
    if not teams or len(picks) < teams * 2:
        return
    name = str(record.get("league_id") or "")
    if not name:
        return
    try:
        os.makedirs(REPLAY_DIR, exist_ok=True)
        tmp = os.path.join(REPLAY_DIR, f".{name}.{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        os.replace(tmp, os.path.join(REPLAY_DIR, f"{name}.json"))
    except OSError:
        return                  # an unwritable archive is not worth a 500 on
        # the front page; the live half of this endpoint is unaffected.
    # Oldest out, so the directory is a handful of rooms rather than a season.
    try:
        kept = sorted(glob.glob(os.path.join(REPLAY_DIR, "*.json")),
                      key=os.path.getmtime, reverse=True)
        for stale in kept[REPLAY_KEEP:]:
            os.remove(stale)
    except OSError:
        pass


def _dwell(pick: dict) -> float:
    """How long a replayed pick sits on screen: the time it really took."""
    seconds = pick.get("seconds_to_pick")
    if not seconds:
        # Never seen opening -- the farm connected mid-turn, or this is an
        # archive old enough to predate the timings. The middle of the clamp
        # is the honest placeholder: it makes no claim the file does not
        # support, and the replay keeps moving.
        seconds = (REPLAY_MIN_SECONDS + REPLAY_MAX_SECONDS) / 2
    return min(REPLAY_MAX_SECONDS, max(REPLAY_MIN_SECONDS, float(seconds)))


_REPLAY: dict = {}


def _replay_plan() -> list:
    """The archived rooms, each with its playable window and its timings.

    Cached on the archive's own mtimes, because this is consulted once a
    second by the event stream and re-reading five drafts a second to learn
    that none of them changed would be the most expensive thing on the page.

    A room's playable window is rounds one through `INTERESTING_ROUNDS`, from
    `WARMED_UP_PICKS` in -- the same two bounds `_liveliest` and `_moment`
    apply to a live room, for the same reasons. The replay is the same
    product showing the same part of a draft; it just is not happening now.
    """
    try:
        paths = sorted(glob.glob(os.path.join(REPLAY_DIR, "*.json")),
                       key=os.path.getmtime, reverse=True)[:REPLAY_KEEP]
        stamp = tuple((p, os.path.getmtime(p)) for p in paths)
    except OSError:
        return []
    if _REPLAY.get("stamp") == stamp:
        return _REPLAY["plan"]
    plan = []
    for path in paths:
        record = _record(path)
        if record is None:
            continue
        teams = int(record.get("teams") or 0)
        picks = record.get("picks") or []
        if not teams:
            continue
        window = picks[:teams * INTERESTING_ROUNDS]
        if len(window) <= WARMED_UP_PICKS:
            continue
        # Cumulative screen time, one entry per pick in the window. Index i
        # holds the moment pick i+1 lands, so a lookup is a walk up this list.
        marks = []
        running = 0.0
        for pick in window:
            running += _dwell(pick)
            marks.append(running)
        opening = marks[WARMED_UP_PICKS - 1]
        plan.append({"record": record, "marks": marks, "opening": opening,
                     "length": running - opening})
    _REPLAY["stamp"] = stamp
    _REPLAY["plan"] = plan
    return plan


def _replayed(now: float):
    """The archived room to show, and how far into it, from the clock alone.

    NO STATE AND NO SCHEDULER. Where the replay has got to is a function of
    the current time, so every worker process, every reader, and the event
    stream all compute the same answer without coordinating -- and a server
    restarted mid-replay picks it up where it was rather than starting the
    draft over. The archived rooms are played end to end, then again, which
    is also what stops a visitor who stays through a long gap from watching
    the same eight rounds twice in a row.
    """
    plan = _replay_plan()
    if not plan:
        return None, None
    total = sum(entry["length"] for entry in plan)
    if total <= 0:
        return None, None
    offset = now % total
    for entry in plan:
        if offset >= entry["length"]:
            offset -= entry["length"]
            continue
        elapsed = entry["opening"] + offset
        marks = entry["marks"]
        at_pick = WARMED_UP_PICKS
        while at_pick < len(marks) and marks[at_pick - 1] <= elapsed:
            at_pick += 1
        # The turn now open started when the last pick landed, which is what
        # the live half means by `written_at` too -- so the countdown, the
        # staleness rules and everything else downstream read a replayed room
        # exactly as they read a live one.
        landed = now - (elapsed - marks[at_pick - 1])
        record = dict(entry["record"],
                      picks=(entry["record"].get("picks") or [])[:at_pick],
                      written_at=landed)
        return record, at_pick
    return None, None


def _dir_revision(now: float) -> str:
    """A cheap fingerprint of the farm's output: names and modification times.

    Stats, no parsing. This is what the event loop runs every second, and its
    only job is to answer "has anything at all changed" fast enough that
    asking once a second costs nothing. A change here is a HINT -- it is the
    identity below, not this, that decides whether a reader is woken.
    """
    parts = []
    for path in _live_files(now):
        try:
            stat = os.stat(path)
        except OSError:
            continue
        parts.append(f"{os.path.basename(path)}:{stat.st_mtime_ns}")
    return "|".join(parts)


def _identity(now: float) -> tuple:
    """Which room the page is showing, and which pick of it.

    THE CACHE KEY AND THE WAKE SIGNAL, both. Two readers whose identity
    matches must be shown the same board, and a reader whose identity has
    moved must not be shown the old one -- so the same function decides when
    an entry dies and when a watcher is told.

    Deliberately narrower than `_dir_revision`: the farm sits in several rooms
    at once and the page shows one of them. A pick landing in a room nobody is
    looking at changes every file's mtime and changes nothing here, which is
    the difference between waking every reader on every pick in the building
    and waking them on the picks in their own room.

    Parsing rather than stating: this reads the same files `_build` does, at
    about a millisecond for the four a farm keeps, and only when the cheap
    fingerprint above has already said something moved.
    """
    records = [r for r in (_record(p) for p in _live_files(now)) if r]
    record = _liveliest(records, now)
    if record is None:
        # Nothing is drafting, so the page is on the archive -- whose current
        # pick is a function of the clock and therefore moves without any
        # file changing. This is why the event stream cannot rely on mtimes
        # alone to decide when to wake a reader.
        replayed, at_pick = _replayed(now)
        if replayed is None:
            return ("empty", len(records))
        return ("replay", str(replayed.get("league_id")), int(at_pick))
    at_pick, _live_moment = _moment(record)
    return (str(record.get("league_id")), int(at_pick))


def _observed_clock() -> float | None:
    """The pick clock every room on file agrees on, or None if they do not.

    THE FALLBACK FOR A ROOM WHOSE OWN CLOCK WAS NOT PUBLISHED. ESPN states a
    clock's length only in the frame that opens a turn, so a farm process
    that predates the field -- or one that connected mid-turn -- leaves a
    live room with picks and no timing. That is most of them until the fleet
    turns over, and it is the difference between the page having a countdown
    and not.

    Not a guess and not a constant typed in here: it is read back out of the
    archive, and used only when every timed pick on file carries the SAME
    length. Measured across the last twenty mocks in the corpus that is 2,559
    picks and exactly one value, 30 seconds -- an ESPN mock room's clock. If
    a room with a different one ever lands in the archive the set stops
    agreeing, this returns None, and the page shows no timer rather than a
    disputed one.

    A room's OWN published clock always wins; this is consulted only when
    there is none. The moment the farm's processes restart it stops being
    consulted at all.
    """
    values = set()
    for entry in _replay_plan():
        for pick in entry["record"].get("picks") or []:
            value = pick.get("clock_seconds")
            if value:
                values.add(float(value))
    if len(values) == 1:
        return values.pop()
    if values:
        # The archive holds rooms that ran on different clocks. There is no
        # answer to borrow, and picking one of them would be picking.
        return None
    # Nothing on file is timed at all. That is where this ends up once the
    # archive has cycled through enough rooms published by processes that
    # never recorded a clock -- so the question goes to the corpus, which
    # holds hundreds of drafts and is not a rolling window of five.
    return _corpus_clock()


# How long the corpus's answer stands before it is asked again. It is a
# property of ESPN's mock rooms rather than of anything here, and it has not
# moved across the whole corpus, so an hour is frequent enough to notice if
# it ever does.
CLOCK_CACHE_SECONDS = 3600.0

# And how long a FAILURE stands, which must not be the same. The farm holds a
# write lock on the corpus while it records a finished draft, so a read
# landing in that second answers nothing -- and caching that for an hour
# would take the page's countdown away for an hour because of a lock held for
# a moment.
CLOCK_RETRY_SECONDS = 60.0

_CLOCK: dict = {}


def _corpus_clock(now: float | None = None) -> float | None:
    """The one pick clock the recorded drafts agree on, if they agree.

    Same rule as the archive's, over a much larger sample, and cached for an
    hour because it is a database query behind a page a stranger can hit.
    Read-only and best-effort: the farm writes to this database at the end of
    every draft, and a lock held at that moment means no countdown for the
    next few seconds, never an error on the front page.
    """
    now = time.time() if now is None else now
    if _CLOCK:
        stands = (CLOCK_CACHE_SECONDS if _CLOCK["value"] is not None
                  else CLOCK_RETRY_SECONDS)
        if _CLOCK["at"] + stands > now:
            return _CLOCK["value"]
    value = None
    try:
        import duckdb
        from pipeline import draft_log as dl
        conn = duckdb.connect(dl.CORPUS_PATH, read_only=True)
        try:
            rows = conn.execute(
                "SELECT DISTINCT clock_seconds FROM draft_log_pick"
                " WHERE clock_seconds IS NOT NULL LIMIT 2").fetchall()
        finally:
            conn.close()
        if len(rows) == 1 and rows[0][0]:
            value = float(rows[0][0])
    except Exception:          # noqa: BLE001 -- see above.
        value = None
    _CLOCK["at"] = now
    _CLOCK["value"] = value
    return value


def _clock(record: dict, picks: list, live_moment: bool) -> tuple:
    """The pick clock for the turn now open: its length, and when it opened.

    Both come from the farm rather than from arithmetic here. The length is
    the last one ESPN stated when it opened a turn (`clock_seconds`, carried
    on each published pick); the moment is `written_at`, which is exactly
    when the last pick landed because the farm publishes on the pick count
    changing and on nothing else -- and ESPN opens the next seat's turn in
    the same instant it closes the last one.

    None, both of them, whenever any part of that is missing or does not
    apply:

      * a farm process old enough to predate `clock_seconds`, AND an archive
        that cannot supply one either -- see `_observed_clock`, consulted
        only when the room itself is silent and only when every room on file
        agrees;
      * a rewound room (`live_moment` false), where the picks on screen are a
        real draft's third round and the seconds since them are hours;
      * a room whose turn was never seen opening.

    A page given None shows no timer at all, which is the honest rendering of
    "this room is live and its clock is not knowable from here".
    """
    if not live_moment:
        return None, None
    length = None
    for pick in reversed(picks):
        value = pick.get("clock_seconds")
        if value:
            length = float(value)
            break
    if length is None:
        # This room never published one. See `_observed_clock` for what is
        # used instead and why it is not a guess.
        length = _observed_clock()
    started = record.get("written_at")
    if length is None or not started:
        return None, None
    return length, float(started)


def _build(conn, now: float | None = None) -> dict:
    # The caller's clock, when it has one. `demo_live` works out the cache key
    # before it builds, and a replay advances between those two instants often
    # enough to matter: built under its own timestamp, a payload could be
    # filed under the pick BEFORE the one it actually contains, and every
    # reader after it would miss a cache they should have hit.
    now = time.time() if now is None else now
    records = [r for r in (_record(p) for p in _live_files(now)) if r]
    record = _liveliest(records, now)
    mode = "live"
    if record is None:
        # Nothing is drafting: replay a room that was. Said plainly in the
        # payload rather than passed off as live -- "happening now" is the
        # one claim on this page a reader could catch out, and a draft that
        # finished eight minutes ago is still a real draft between real
        # people, which is the whole of what the page is showing off.
        record, _at = _replayed(now)
        mode = "replay"
    if record is None:
        return {"live": False, "mode": "none", "rooms": len(records)}
    if mode == "live":
        # Kept for the next gap, on the way past. See `_archive`.
        _archive(record)

    board = _board_frame(conn)
    names = {}
    for row in board.itertuples():
        names[str(getattr(row, "player_id", ""))] = {
            "name": getattr(row, "name", None),
            "position": getattr(row, "position", None),
            "team": getattr(row, "team", None),
        }

    at_pick, live_moment = _moment(record)
    picks = (record.get("picks") or [])[:at_pick]
    taken = {str(p.get("player_id")) for p in picks}
    recent = []
    for pick in picks[-RECENT_PICKS:]:
        pid = str(pick.get("player_id"))
        known = names.get(pid, {})
        recent.append({
            "pick_no": pick.get("pick_no"),
            "slot": pick.get("slot"),
            "name": known.get("name"),
            "position": known.get("position"),
            "team": known.get("team"),
            # The farm's own flag. Worth serving: a room where every pick is
            # autodrafted is not the same evidence as one with seven people in
            # it, and the page says how many are real.
            "autodrafted": bool(pick.get("autodrafted")),
        })

    teams = int(record.get("teams") or 0)
    rounds = int(record.get("rounds") or 0)
    made = len(picks)
    clock_seconds, turn_started_at = _clock(record, picks, live_moment)
    return {
        # "There is a room to draw", which is what the page branches on --
        # not "this is happening this second". That is `mode`.
        "live": True,
        # Which of the two this is: a draft going on right now, or one from
        # the archive being replayed at the pace its picks were really made.
        # The room's status pill says so in as many words.
        "mode": mode,
        # Which room this is. Read back at startup so a restarted server
        # carries on showing the room it was showing -- see `_SHOWING`.
        "league_id": str(record.get("league_id") or ""),
        # Whether the picks above are where that room is RIGHT NOW, or an
        # earlier round of it. The page says which; see `_moment`.
        "live_moment": live_moment,
        "rooms": len(records),
        "teams": teams,
        "rounds": rounds,
        "humans": record.get("human_seats"),
        "started_at": record.get("started_at"),
        "picks_made": made,
        "picks_total": teams * rounds,
        "round": made // teams + 1 if teams else None,
        "on_the_clock": _on_the_clock({**record, "picks": picks}),
        # THE COUNTDOWN, as two facts and no ticking. The seat on the clock
        # has `clock_seconds` from `turn_started_at`, both epoch-honest; the
        # page subtracts and renders. A server counting down would be a
        # number that is wrong the moment it is serialised and gets wronger
        # over the network, and would need a request per second to stay
        # nearly right. Null together when the clock is not knowable -- see
        # `_clock`.
        "clock_seconds": clock_seconds,
        "turn_started_at": turn_started_at,
        # This server's own clock at build time, so the page can measure the
        # turn's age against the SERVER's timeline rather than the browser's.
        # A laptop three minutes off (they routinely are) would otherwise show
        # a countdown three minutes wrong, or a permanently expired one.
        "server_now": now,
        # The room's own settings, because `AvailableList` draws finish tiers
        # from them -- the last startable back is a different rank in an
        # 8-team league than a 12-team one, and the farm now plays both. The
        # scoring format comes off the live file for the same reason.
        "settings": _settings_payload(conn, teams, rounds,
                                      record.get("scoring_format")),
        # The grid and the roster, in the room's own shapes, so the landing
        # page can run `DraftBoardGrid`, `PickTicker` and `RosterPanel`
        # unmodified. See `_board_payload`.
        "board": _board_payload(record, picks, board,
                                _on_the_clock({**record, "picks": picks})),
        "roster": _roster_payload(picks, _on_the_clock({**record, "picks": picks}),
                                  board, _settings_payload(
                                      conn, teams, rounds,
                                      record.get("scoring_format"))),
        "recent": [row for row in recent if row["name"]],
        "shortlist": _ranked(conn, board, picks, _on_the_clock({**record, "picks": picks}),
                             SHORTLIST, teams=teams, rounds=rounds,
                             fmt=record.get("scoring_format")),
    }


# WHERE THE LAST ANSWER SURVIVES A RESTART. A build is 2.8-4.0s warm and 7.6s
# cold (measured), and until this file existed the first visitor after a
# server start paid for one while looking at an empty room. The payload is
# written here after every build and read back at startup, so a restarted
# server begins with a board already in hand and refreshes it behind the
# reader rather than in front of them.
DEMO_LAST = os.environ.get("DEMO_LAST_PATH", "data/demo-last.json")

# How old a restored answer may be before it is dropped rather than shown.
# Past this it is not a picture of a draft, it is a picture of last night.
LAST_MAX_AGE = 6 * 3600.0

# Whether registering the routes also warms them. On in a server, and worth
# an off switch for the two callers that do not want a background build the
# moment an app object exists: a test that has just primed the cache by hand,
# and any process that imports the app to read its routes rather than serve
# them.
WARM_ON_REGISTER = os.environ.get("DEMO_WARM", "1") != "0"

_REFRESH: dict = {"running": False, "warming": False}


def refresh_in_background(build) -> bool:
    """Rebuild off the request path, one at a time. Returns whether it started.

    THE READER NEVER WAITS FOR A BUILD. That is the whole rule: a build is
    seconds of real work, and a page that puts one in front of a visitor
    spends those seconds showing an empty room with a spinner in it -- which
    is what this endpoint used to do to the first person through the door
    after every restart.

    The single-flight guard is not an optimisation. Without it a burst of
    readers arriving on a stale entry would each start a build of the same
    answer, and the machine would spend its cores computing one board several
    times over.

    A MODULE-LEVEL FUNCTION, not the closure it used to be, and that is the
    point of it being here rather than inside `register_demo_routes`. A test
    that needs the page to hold still has to be able to stop the rebuild, and
    a closure cannot be patched: `mock.patch.object(demo, "refresh_behind",
    create=True)` invented an attribute nothing read, the real thread started
    anyway, and the test then failed or passed depending on which of the two
    got there first. The flag is set AND cleared in here for the same reason
    -- a replacement that does nothing leaves no state behind for the next
    test in the process to trip over.
    """
    with _LOCK:
        if _REFRESH["running"]:
            return False
        _REFRESH["running"] = True

    def run():
        try:
            build()
        finally:
            with _LOCK:
                _REFRESH["running"] = False

    threading.Thread(target=run, name="demo-refresh", daemon=True).start()
    return True


def _persist(payload: dict) -> None:
    """Keep the built answer, so the next process start is not a cold one."""
    try:
        os.makedirs(os.path.dirname(DEMO_LAST) or ".", exist_ok=True)
        tmp = f"{DEMO_LAST}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, DEMO_LAST)
    except OSError:
        pass


def _restore(now: float) -> dict | None:
    """The answer this server built before it was restarted, if it is recent.

    Downgraded to `replay` on the way in, whatever it said when it was
    written: the room it describes was live an hour ago and is not live now,
    and the pill above it must not claim otherwise for the few seconds before
    the first real build lands. The board, the picks and the prices in it are
    a real draft's either way -- which is exactly what `replay` means here.
    """
    try:
        with open(DEMO_LAST, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("live"):
        return None
    built_at = payload.get("server_now")
    if not built_at or now - float(built_at) > LAST_MAX_AGE:
        return None
    return dict(payload, mode="replay")


def _seed_archive(now: float) -> None:
    """Give a server with no archive one, out of the corpus.

    The archive normally fills itself: every live room this endpoint serves is
    kept. That leaves one hole, and it is the worst one -- a machine that has
    never served a live room. A fresh clone, a new deployment, or a restart
    that lands inside one of the farm's gaps all start with nothing to show
    and no way to get anything until a draft happens to be running.

    The corpus has hundreds of real drafts and every field the archive needs,
    so it is read once, at startup, only when the archive is empty. Read-only
    and best-effort: the farm writes to this database at the end of every
    draft, and a lock held at that moment is a reason to skip seeding, never a
    reason to fail a start.
    """
    try:
        if glob.glob(os.path.join(REPLAY_DIR, "*.json")):
            return
        import duckdb
        from pipeline import draft_log as dl
        conn = duckdb.connect(dl.CORPUS_PATH, read_only=True)
    except Exception:          # noqa: BLE001 -- no corpus, or it is busy.
        return
    try:
        drafts = conn.execute(
            "SELECT draft_id, league_id, season, teams, rounds, human_seats,"
            " recorded_at FROM draft_log WHERE source = ? AND teams > 0"
            " ORDER BY recorded_at DESC LIMIT ?",
            [dl.SOURCE_MOCK, REPLAY_KEEP]).fetchall()
        for draft_id, league_id, season, teams, rounds, humans, ended in drafts:
            picks = conn.execute(
                "SELECT pick_no, slot, player_id, autodrafted, seconds_to_pick,"
                " clock_seconds FROM draft_log_pick WHERE draft_id = ?"
                " ORDER BY pick_no", [draft_id]).fetchall()
            if len(picks) < int(teams) * 2:
                continue
            _archive({
                "league_id": str(league_id),
                "season": int(season or 0),
                "teams": int(teams),
                "rounds": int(rounds or 0),
                "human_seats": humans,
                "started_at": str(ended),
                "written_at": now,
                "picks": [{"pick_no": int(n), "slot": int(slot or 0),
                           "player_id": None if pid is None else str(pid),
                           "autodrafted": auto,
                           "seconds_to_pick": took,
                           "clock_seconds": clock}
                          for n, slot, pid, auto, took, clock in picks],
            })
    except Exception:          # noqa: BLE001 -- see above.
        pass
    finally:
        try:
            conn.close()
        except Exception:      # noqa: BLE001
            pass


def register_demo_routes(app, conn=None):
    """The hero's data (`GET /api/demo/live`) and its wake signal
    (`GET /api/demo/events`), both cached and anonymous."""

    def build_now():
        """One build, cached and kept. Never called from a request thread."""
        stamp = time.time()
        try:
            identity = _identity(stamp)
        except Exception:      # noqa: BLE001 -- see the build's own guard.
            identity = None
        try:
            built = _build(conn if conn is not None else get_conn(), stamp)
        except Exception:      # noqa: BLE001 -- the landing page must render
            # for a visitor whether or not the farm, the board cache or the
            # database are having a good day. "Nothing is drafting" is a
            # truthful degradation; a 500 on the front page is not.
            built = {"live": False, "mode": "none", "rooms": 0}
        now = time.monotonic()
        ttl = CACHE_SECONDS if built.get("live") else EMPTY_CACHE_SECONDS
        with _LOCK:
            _CACHE["live"] = (identity, now + ttl, built, now)
        if built.get("live"):
            _persist(built)
        return built

    @app.get("/api/demo/live")
    def demo_live(response: Response):
        # The landing page's hero, anonymous and the same for everybody. Five
        # seconds is roughly how often a mock room picks, so a shared cache
        # holding it that long never shows a pick late by more than one tick
        # -- and the event stream beside it is what actually tells a reader
        # to come back for the next one.
        http_cache.public(response, 5)
        now = time.monotonic()
        with _LOCK:
            hit = _CACHE.get("live")
        if hit is None:
            # Nothing has ever been built and nothing was restored -- the
            # only case left where a reader waits, and it lasts one request.
            built = build_now()
            if not built.get("live") and _REFRESH["warming"]:
                return dict(built, warming=True)
            return built

        try:
            identity = _identity(time.time())
        except Exception:      # noqa: BLE001
            identity = None
        # Three clocks, and they mean different things. The identity is the
        # room: matching it says this answer is ABOUT the same pick of the
        # same draft. The deadline is the ceiling: the board underneath an
        # unchanged room can still be rebuilt by other work, and an answer
        # that stands for an hour because nobody picked would be showing
        # prices that have since moved. The third is the floor -- an answer
        # this young is served even to a reader whose room HAS moved,
        # because the alternative is one rebuild per autodraft. See
        # `MIN_REBUILD_SECONDS`.
        fresh = hit[1] > now and (hit[0] == identity
                                  or now - hit[3] < MIN_REBUILD_SECONDS)
        payload = hit[2]
        if not payload.get("live") and (_REFRESH["warming"]
                                        or _REFRESH["running"]):
            # Not "between drafts" -- this process simply has not finished
            # its first build yet. The page says the two differently, because
            # one of them is over in a couple of seconds and the other is
            # not, and a reader who is told the wrong one leaves.
            payload = dict(payload, warming=True)
            return payload
        if not fresh:
            # Stale, so somebody should rebuild -- but not this reader, and
            # not while they wait. They get the answer that exists; the next
            # one along gets the new one, and the event stream will have
            # already told them to come back for it. Through the module-level
            # `refresh_in_background`, which is the seam a test can hold
            # still; see it for why that is not a detail.
            refresh_in_background(build_now)
        return payload

    @app.get("/api/demo/events")
    async def demo_events(request: Request):
        """Tell the page when the room moved. It fetches; this never sends a
        board.

        WHY A STREAM AND NOT A POLL. The page was polling on a timer, which
        makes a live draft look like a slideshow: a pick lands and sits there
        unshown for whatever is left of the window, and every reader pays a
        request per window whether or not anything happened. A draft is an
        event stream and this is the shape of one -- a pick lands, everybody
        watching hears about it inside a second, and a quiet room costs one
        comment every twenty.

        WHY IT CARRIES NO PAYLOAD. What changed is one line; what a reader
        needs is a board, a ranked shortlist and a plan drawn from recorded
        rollouts. Sending only the signal keeps this loop free of the work
        (it stats four files a second and holds no database handle) and keeps
        exactly one path to the board -- the cached endpoint above, which a
        hundred woken readers hit as one build and ninety-nine cache hits.
        Pushing the payload down every open stream would build it per reader
        and duplicate the endpoint in the bargain.

        SSE rather than a WebSocket, for the same reason the draft room uses
        it: this is one-directional -- the page has nothing to say back -- and
        EventSource reconnects on its own, through proxies that treat a socket
        upgrade as something to think about.
        """
        async def stream():
            told = None
            fingerprint = None
            idle = 0.0
            spent = 0.0
            while spent < EVENT_MAX_SECONDS:
                if await request.is_disconnected():
                    return
                try:
                    now = time.time()
                    fresh = _dir_revision(now)
                    # A replay advances on the clock, with no file changing at
                    # all, so the mtime shortcut cannot be the only trigger --
                    # it would leave a replayed room frozen on screen. Cheap
                    # anyway in that state: when nothing is drafting there are
                    # no farm files to parse, and the archive's timings are
                    # already in memory.
                    replaying = told is not None and told[0] == "replay"
                    if fresh != fingerprint or replaying or told is None:
                        fingerprint = fresh
                        # Otherwise it is not worth parsing: something moved,
                        # but most somethings are another room's pick.
                        identity = _identity(now)
                        if identity != told:
                            told = identity
                            idle = 0.0
                            yield ("data: " + json.dumps(
                                {"league": identity[0], "pick": identity[1]})
                                + "\n\n")
                except Exception:  # noqa: BLE001 -- a directory that cannot
                    # be read is a farm that is down, which the fetch this
                    # would have triggered reports for itself. Killing the
                    # stream over it would take the page's live-ness with it
                    # and leave a reader on a board that never updates again.
                    pass
                if idle >= EVENT_KEEPALIVE:
                    idle = 0.0
                    yield ": still here\n\n"
                await asyncio.sleep(EVENT_TICK)
                idle += EVENT_TICK
                spent += EVENT_TICK

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                # A stream has nothing for a cache to hold, and a proxy that
                # tried would buffer it instead of passing it through.
                "Cache-Control": http_cache.NO_STORE,
                "Connection": "keep-alive",
                # nginx buffers a response body by default, which for a
                # stream means the reader gets an hour of events at once.
                "X-Accel-Buffering": "no",
            },
        )

    # ARRIVE WITH AN ANSWER. Three steps, cheapest first: put the last
    # process's answer back in the cache so the very first request is served
    # from memory; make sure there is an archive to fall back on if the farm
    # is between rooms; then build for real, off the main thread, so what is
    # in hand is current within seconds of the port opening.
    #
    # Done here rather than on a startup event because this function IS the
    # startup for this endpoint -- it is called while the app is being built,
    # and a warm cache is not worth a second lifecycle hook in an app that
    # has none.
    # LAST, so a failure to warm can never cost the routes above it.
    if not WARM_ON_REGISTER:
        return
    stamp = time.time()
    restored = _restore(stamp)
    if restored is not None:
        if restored.get("league_id"):
            league_id = str(restored["league_id"])
            # Counted from the pick the ROOM is on, which is not the pick the
            # last answer drew. A room past `REWIND_ROUND` is served rewound
            # (see `_moment`), so `picks_made` in that answer can be sixty
            # picks behind the room -- and a turn counted from there is spent
            # the instant it is seeded, which turned a restart into exactly
            # the jump this memory exists to prevent.
            made = None
            for record in (_record(path) for path in _live_files(stamp)):
                if record and str(record.get("league_id") or "") == league_id:
                    made = _pick_of(record)
                    break
            # A full turn rather than whatever was left of the old one: the
            # persisted answer does not say when the turn began, and half a
            # turn made up here would be a guess on the page's behalf. A room
            # that is no longer live is not seeded at all -- there is nothing
            # to hold on to and the rotation should start clean.
            with _SHOWING_LOCK:
                if made is not None and not _SHOWING.get("league_id"):
                    _SHOWING["league_id"] = league_id
                    _SHOWING["since_pick"] = made
                    _SHOWING["chosen_at"] = stamp
                    _SHOWING["seen"] = {league_id}
        mark = time.monotonic()
        with _LOCK:
            # Identity `None` so it matches nothing: this is served
            # immediately and replaced by the first real build, rather than
            # standing until a ceiling expires.
            _CACHE["live"] = (None, mark + CACHE_SECONDS, restored, mark)
    def warm():
        try:
            _seed_archive(stamp)
            build_now()
        finally:
            _REFRESH["warming"] = False

    _REFRESH["warming"] = True
    threading.Thread(target=warm, name="demo-warm", daemon=True).start()
