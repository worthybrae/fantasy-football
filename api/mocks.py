"""Every mock draft the farm is in or has played, and the board for each.

TWO SOURCES, ONE LIST. A mock draft is visible to this process in exactly one
of two ways, and which one depends only on whether it has finished:

  - WHILE IT IS BEING PLAYED it lives in `data/farm-live/<league_id>.json`,
    written by `pipeline.mock_farm` after every pick (see that module's "Live
    progress" section for why a file and not a table). The farm is a separate
    process, so a file is the only thing this one can see of it at all.
  - ONCE IT IS RECORDED it lives in the draft corpus, `draft_log` +
    `draft_log_pick`, and the live file is deleted in the same `finally` that
    releases the room claim.

The id is the SAME across that transition -- `draft_id_for(mock, league_id,
season, started_at)`, computed identically on both sides -- so a page open on
a live board does not lose its draft when the last pick lands.

READ-ONLY, ALWAYS, AND WITH NO SCHEMA CALL. The farm holds a write connection
to the corpus for the second or two it takes to record a draft, and DuckDB is
single-writer per file: a read-write connection opened here would either fail
or, worse, succeed and lock the farm out of writing the draft it just spent
forty minutes playing. So every corpus read below opens `read_only=True`, per
request, and closes immediately. `ensure_schema` is NOT called on it --
DuckDB refuses any CREATE against a read-only database, including a no-op
`CREATE TABLE IF NOT EXISTS`, which is the exact bug `pipeline.mock_backfill`'s
`corpus_report` hit and solved by copying `summary`'s query rather than
calling it. The same rule applies here: nothing in this module may write to,
replace, or drop anything in that file.

WHY THE PLAYER DETAIL COMES FROM `data/nfl.duckdb` AND NOT THE CORPUS. The
corpus stores what a pick WAS -- player_id, position, adp_rank, proj_points --
and nothing a person would recognise: no names, no headshots, no tiers. The
board this process already holds has all of it, keyed by the same player_id,
and uvicorn keeps that database open read-write for the whole session, so it
is one dictionary lookup per cell rather than a second file handle. The cell
payload itself is built by `api.live._board_cell`, called rather than copied,
because the front end draws a mock board with the same component it draws the
live one with: two implementations of `BoardPlayer` would drift, and the drift
would show up as a column silently going blank on one page and not the other.
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import HTTPException

from api.live import (_board_cell, _board_index, _attach_espn_proj,
                      identify_players)
from pipeline import draft_log as dl
from pipeline import mock_farm as farm
from scoring.board_cache import cached_build_board
from scoring.draft_sim import snake_slots

# How hard to try when the corpus is locked. The farm takes the write lock
# only around `dl.record` -- a delete and an insert of 128 rows, then close --
# so a request that collides with one is unlucky rather than doomed, and a
# short retry covers nearly all of that window. Past it the honest answer is
# 503 (try again), not an empty list: a page that quietly dropped every
# completed draft for two seconds would look exactly like data loss.
CORPUS_LOCK_TRIES = 4
CORPUS_LOCK_RETRY_SECONDS = 0.15

# What `made_by` can say. Kept as a tuple so a typo in one of the branches
# below is a test failure rather than a label the front end has no icon for.
MADE_BY = ("us", "engine", "auto", "human", "unknown")


def _open_corpus():
    """A read-only connection to the corpus, or None if there is no file yet.

    None rather than an exception for a missing file: an installation that has
    never run `make farm-mocks` or `make mock-backfill` has no corpus, and
    "no drafts yet" is the ordinary state of things there, not an error.

    A LOCKED file IS an exception, and deliberately a 503 rather than a
    silently short answer -- see CORPUS_LOCK_TRIES. Unlike
    `mock_backfill._open_corpus_read_only`, which copies the whole file and
    reads the snapshot, nothing here falls back to a copy: that costs a
    multi-megabyte file copy per request and returns data that is stale in a
    way the caller cannot see, which is a bad trade for an endpoint that is
    polled.
    """
    import duckdb

    target = dl.CORPUS_PATH
    if not Path(target).exists():
        return None
    last = None
    for attempt in range(CORPUS_LOCK_TRIES):
        try:
            return duckdb.connect(target, read_only=True)
        except Exception as exc:                # noqa: BLE001
            if "lock" not in str(exc).lower():
                raise
            last = exc
            if attempt + 1 < CORPUS_LOCK_TRIES:
                time.sleep(CORPUS_LOCK_RETRY_SECONDS)
    raise HTTPException(
        status_code=503,
        detail="the draft corpus is being written by the farm right now -- "
               f"try again in a moment ({last})")


def _query(conn, sql: str, params: list) -> list:
    """Run one corpus query, treating a missing table as no rows.

    A corpus file whose tables have not been created yet is reachable: DuckDB
    writes the file on connect, and a farm that crashed before its first
    record leaves exactly that. `ensure_schema` is not an option here (see the
    module docstring), so the absence is caught rather than fixed.
    """
    import duckdb

    try:
        return conn.execute(sql, params).fetchall()
    except duckdb.CatalogException:
        return []


def _board_frame(conn):
    """The board every mock cell is named from.

    `cached_build_board` rather than `build_board`: this is the same cache the
    player pages and the live room already share, so the first mock board
    costs whatever a board costs (~1.6-1.9s) and every one after it is free
    until a refresh or a pick invalidates the key.

    Built against THIS installation's own league settings, not the mock's.
    That is the right call even though a mock room has its own roster shape:
    everything read out of the board here is identity (name, team, bye,
    headshot) or market (ADP, ESPN's rank, last year's points), none of which
    is a function of the room -- and rebuilding per draft would cost a board
    build per page load and share nothing with the rest of the process.

    A separate function, and the one place `conn` is touched, so a test can
    replace it with a small frame instead of seeding a whole database.
    """
    if conn is None:
        return None
    return _attach_espn_proj(conn, cached_build_board(conn))


def _made_by(slot, my_slot, had_owner, autodrafted) -> str:
    """Who actually made this pick -- the whole point of the page.

    ORDER IS THE MEANING HERE, so each branch is stated with its reason:

      - OUR OWN SEAT FIRST, before anything else is looked at. The farm's bot
        plays epsilon-greedy (`mock_farm.choose_index`), so it is neither a
        person nor ESPN's engine, and it always has an owner attached (we
        joined with a member id) -- which means every other test below would
        call it "human". A corpus fitted on that belief would be learning our
        own exploration noise as somebody's draft tendencies, which is the
        failure `my_slot` exists on the draft head to prevent.
      - `had_owner` FALSE is ESPN's computer filling an empty room: a seat no
        person ever sat in drafts from ESPN's own ranking from pick one, and
        `mock_farm.draft_timeline` seeds it as autodrafting before a single
        frame arrives, so its `autodrafted` flag says nothing extra.
      - `had_owner` NULL is UNKNOWN, never a guess. Two populations land here
        and neither can be recovered: every draft recorded before seat
        labelling existed, and the 23 harvested drafts from
        `pipeline.mock_backfill`, whose leagues 404 the moment their draft
        ends so nobody can ever ask ESPN who was in them.
      - Only with a KNOWN OWNER does `autodrafted` mean what the page wants it
        to mean: a person who joined and then wandered off, ESPN taking the
        seat over mid-draft. That is "auto"; a pick they actually made is
        "human"; and a seat with an owner but no autodraft flag at all (the
        mTeam read succeeded and the socket said nothing, which the state
        machine in `draft_timeline` makes vanishingly rare) is honestly
        "unknown" rather than assumed to be one or the other.
    """
    if my_slot is not None and slot is not None and int(slot) == int(my_slot):
        return "us"
    if had_owner is None:
        return "unknown"
    if not had_owner:
        return "engine"
    if autodrafted is None:
        return "unknown"
    return "auto" if autodrafted else "human"


def _board_response(conn, teams, rounds, my_slot, had_owner: dict,
                    picks: list) -> dict:
    """One draft as the grid draws it: `LiveBoard` plus the two mock columns.

    `picks` is (pick_no, slot, player_id, autodrafted) per landed pick, in
    order, from whichever of the two sources this draft came from -- the whole
    reason both paths funnel through here rather than each building a payload
    of its own is that a live board and a completed one must be the same
    document, so the page keeps drawing across the moment a draft finishes.

    `active` IS TRUE FOR A COMPLETED DRAFT TOO. On /api/live/board that flag
    means "there is a session", and the front end reads it as "this response
    carries a board" -- every read site checks it before touching any other
    field (see LiveBoard in web/src/api.ts). A finished mock has a board, so
    the flag is true; whether it is still being played is `status` on the
    listing, not this.

    `picks_made` IS THE HIGHEST PICK NUMBER, not the number of cells, exactly
    as /api/live/board computes it. A pick whose player could not be resolved
    to a board row is dropped at record time (`mock_farm.build_record`) and
    skipped below, and counting rows would quietly renumber the whole draft
    around that hole rather than leaving it visible as one.
    """
    # `teams`/`rounds` are nullable columns in `draft_log`. Every mock ever
    # written has both, but a null would otherwise reach `snake_slots` and
    # raise a 500 on a page whose job is to display what is there: an empty
    # grid says "this row is malformed" far more usefully than a stack trace.
    teams = int(teams or 0)
    rounds = int(rounds or 0)
    slots = snake_slots(teams, rounds)
    by_id = _board_index(_board_frame(conn))

    columns = [{"slot": slot,
                # ESPN's real team names are not recorded for a mock (the
                # league 404s once the draft ends, and the farm never asks),
                # so the column is named by its seat. `had_owner` is what
                # actually distinguishes the columns on this page.
                "team_name": f"Team {slot}",
                "is_me": my_slot is not None and slot == int(my_slot),
                "had_owner": had_owner.get(slot)}
               for slot in range(1, teams + 1)]

    cells = []
    unresolved = 0
    # Named before the loop, in one query: the board is a 250-player pool and
    # a 16-round room draws from a much longer list than that (see
    # `identify_players`). Only the ids the board misses are asked about.
    named = identify_players(conn, [p[2] for p in picks
                                    if p[2] is not None
                                    and str(p[2]) not in by_id])
    for pick_no, slot, player_id, autodrafted in picks:
        slot = None if slot is None else int(slot)
        if player_id is None:
            # A pick the crosswalk never resolved. It keeps its pick NUMBER
            # (see `picks_made` above) but there is nothing to draw in the
            # cell -- not even a fallback name, since the id itself is what is
            # missing.
            continue
        cell = _board_cell(player_id, pick_no, teams, slots, by_id, named)
        if cell["player"]["name"] == str(player_id):
            # Still named by its raw id: not on the board AND not in the
            # identity tables either. Counted rather than passed over
            # silently -- a board rebuilt for a new season, or a crosswalk
            # gap, would turn a page full of names into a page full of ids,
            # and this is the number that says so out loud.
            #
            # An off-board player who IS named (see `identify_players`) is
            # not counted here any more: the cell reads "Evan Engram · TE ·
            # DEN" with no ranks beside it, which is a complete answer to
            # what that pick was, not a hole in the page.
            unresolved += 1
        cell["made_by"] = _made_by(slot, my_slot, had_owner.get(slot),
                                   autodrafted)
        cells.append(cell)

    picks_made = max((int(p[0]) for p in picks), default=0)
    return {
        "active": True,
        "teams": teams,
        "rounds": rounds,
        "my_slot": None if my_slot is None else int(my_slot),
        "on_the_clock": slots[picks_made] if picks_made < len(slots) else None,
        "picks_made": picks_made,
        "columns": columns,
        "cells": cells,
        # How many cells fell back to naming a player by his id. Not part of
        # the LiveBoard shape -- an addition, like `had_owner` and `made_by`
        # -- and additive for a TypeScript client, which ignores keys its
        # interface does not declare.
        "unresolved": unresolved,
    }


def _live_summary(payload: dict) -> dict:
    """One live file as a row of the listing."""
    picks = payload.get("picks") or []
    return {
        "id": payload.get("draft_id"),
        "league_id": payload.get("league_id"),
        "status": "live",
        "teams": payload.get("teams"),
        "rounds": payload.get("rounds"),
        "picks_made": max((int(p.get("pick_no") or 0) for p in picks),
                          default=0),
        "human_seats": payload.get("human_seats"),
        "my_slot": payload.get("my_slot"),
        "started_at": payload.get("started_at"),
        # The live file has no recorded_at -- it has not been recorded. Kept
        # in the shape anyway so both kinds of row carry the same keys and the
        # page never has to branch on which source a row came from.
        "recorded_at": None,
    }


def _live_board(conn, payload: dict) -> dict:
    """A live file as a board response."""
    had_owner = {int(seat["slot"]): seat.get("had_owner")
                 for seat in (payload.get("seats") or [])}
    picks = [(p.get("pick_no"), p.get("slot"), p.get("player_id"),
              p.get("autodrafted"))
             for p in (payload.get("picks") or [])]
    return _board_response(conn, payload.get("teams"), payload.get("rounds"),
                           payload.get("my_slot"), had_owner, picks)


# Deliberately `max(p.pick_no)` and not `count(*)`, for the reason
# `_board_response` gives: a dropped pick must leave a hole in the numbering
# rather than shorten the draft. LEFT JOIN so a head row with no picks at all
# still appears -- it is a real row in a corpus that cannot be rebuilt, and a
# listing that hid it would hide the only evidence something went wrong.
_LIST_SQL = """
    SELECT d.draft_id, d.league_id, d.teams, d.rounds, d.my_slot,
           d.human_seats, d.recorded_at, max(p.pick_no) AS picks_made
    FROM draft_log d LEFT JOIN draft_log_pick p USING (draft_id)
    WHERE d.source = ?
    GROUP BY d.draft_id, d.league_id, d.teams, d.rounds, d.my_slot,
             d.human_seats, d.recorded_at
    ORDER BY d.recorded_at DESC
"""


def register_mock_routes(app, conn):
    """Mount the mock-draft endpoints.

    `conn` is the app's own `data/nfl.duckdb` connection, passed for exactly
    one purpose: naming players (see `_board_frame`). The corpus is opened per
    request instead, read-only, and closed before the response is built --
    holding it open for the process lifetime would keep a reader in the file
    the farm needs to write.
    """

    @app.get("/api/mocks")
    def list_mocks():
        """Every mock draft, live ones first, then completed newest first.

        Live drafts are read first and their ids remembered, so the two-second
        window between `dl.record` landing and the live file being deleted
        shows one row (live, 128 picks) rather than two of the same draft. The
        row flips to "complete" on the next poll.
        """
        live = [_live_summary(p) for p in farm.live_drafts()]
        live.sort(key=lambda row: (row["started_at"] or ""), reverse=True)
        seen = {row["id"] for row in live}

        done = []
        corpus = _open_corpus()
        if corpus is not None:
            try:
                rows = _query(corpus, _LIST_SQL, [dl.SOURCE_MOCK])
            finally:
                corpus.close()
            for (draft_id, league_id, teams, rounds, my_slot, human_seats,
                 recorded_at, picks_made) in rows:
                if draft_id in seen:
                    continue
                done.append({
                    "id": draft_id,
                    "league_id": None if league_id is None else str(league_id),
                    "status": "complete",
                    "teams": None if teams is None else int(teams),
                    "rounds": None if rounds is None else int(rounds),
                    "picks_made": 0 if picks_made is None else int(picks_made),
                    "human_seats": (None if human_seats is None
                                    else int(human_seats)),
                    "my_slot": None if my_slot is None else int(my_slot),
                    # NULL, AND NOT recorded_at DRESSED UP AS A START TIME.
                    # `draft_log` stores when a draft was WRITTEN, never when
                    # it began -- the start time goes into the draft_id hash
                    # and nowhere else, and a sha1 does not come back. Half an
                    # hour separates the two for a mock, so serving one as the
                    # other would put a wrong time on every completed row with
                    # nothing to reveal it. `recorded_at` is served under its
                    # own name beside it instead.
                    "started_at": None,
                    "recorded_at": farm.iso_utc(recorded_at),
                })

        return {"drafts": live + done}

    @app.get("/api/mocks/{draft_id}/board")
    def mock_board(draft_id: str):
        """The snake board for one draft, live or completed.

        Live is checked first for the same reason the listing prefers it: for
        the moment a draft exists in both places, the live file is the one
        with the pick that just landed. A live file that has gone stale (its
        farm process is dead) never reaches this -- `live_drafts` sweeps it on
        read -- so a crashed draft falls through to its corpus row if it was
        recorded before the crash, and 404s if it was not.
        """
        for payload in farm.live_drafts():
            if payload.get("draft_id") == draft_id:
                return _live_board(conn, payload)

        corpus = _open_corpus()
        head = None
        picks = []
        if corpus is not None:
            try:
                # `source` as well as the id: the corpus is shared with
                # imported league history (`draft_log.backfill_history`), and
                # this page is about mocks. An id from anywhere else is not a
                # draft this endpoint has anything to say about.
                rows = _query(corpus,
                              "SELECT teams, rounds, my_slot FROM draft_log "
                              "WHERE draft_id = ? AND source = ?",
                              [draft_id, dl.SOURCE_MOCK])
                head = rows[0] if rows else None
                if head is not None:
                    picks = _query(
                        corpus,
                        "SELECT pick_no, slot, player_id, autodrafted, "
                        "had_owner FROM draft_log_pick WHERE draft_id = ? "
                        "ORDER BY pick_no", [draft_id])
            finally:
                corpus.close()
        if head is None:
            raise HTTPException(status_code=404,
                                detail=f"no mock draft {draft_id}")

        teams, rounds, my_slot = head
        # `had_owner` is stored per PICK (see `draft_log.ensure_schema` for
        # why) but describes the SEAT: it is whether a real ESPN member held
        # that slot when the draft opened, which cannot change mid-draft. So
        # the first non-null value at a slot answers for the whole column, and
        # a slot whose every pick left it null stays unknown.
        had_owner: dict = {}
        for _pick_no, slot, _player_id, _auto, seat_had_owner in picks:
            if slot is None or seat_had_owner is None:
                continue
            had_owner.setdefault(int(slot), bool(seat_had_owner))
        return _board_response(
            conn, teams, rounds, my_slot, had_owner,
            [(p[0], p[1], p[2], p[3]) for p in picks])
