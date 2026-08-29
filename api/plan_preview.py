"""The plan a seat would get, before anybody has drafted anything.

WHAT THIS ANSWERS. Every other plan in this product is conditioned on a room:
the socket names your seat, the picks so far are known, and `build_plan` walks
your remaining turns from where the draft actually is. That is the right answer
during a draft and no answer at all to the question a reader asks on the way in
-- *what would this tell me on draft night?* -- because they have not connected
anything yet and there is no room to condition on.

So this is the same machinery run against an empty board: `picks_made = 0`, no
roster, the seat described by two query parameters rather than discovered from
a socket. It is what the landing page shows a stranger and what the dashboard
shows somebody whose draft is on Thursday.

FOUR THINGS COME BACK, and they answer different questions:

  * `picks`   -- the overall pick numbers this seat owns, first four. Pure
                 arithmetic on the snake; no data behind it at all.
  * `opening` -- what people actually DO from this seat, counted: the three
                 most-walked position paths over the first five rounds. Read
                 off the recorded corpus, human picks only, exactly the query
                 `api/market.py` runs for its seat pages -- see `_openings`.
  * `position_runs` -- when each of the four positions a drafter waits on
                 first comes off the board, as a median over recorded drafts.
                 "QBs go in round 6" is this number.
  * `targets` -- `scoring/plan.py`'s own answer for the first three turns,
                 with the caller's favourites folded in if they have any.

THE CORPUS IS OPTIONAL AND THE BOARD IS NOT. A fresh install has no recorded
drafts and this endpoint still has something true to say: the plan is built
from the board, so `opening` and `position_runs` come back empty rather than
taking the page down with them. A board that cannot be built is a different
kind of failure and is allowed to be one.

WHOSE ANSWER IT IS. Signed out, the favourites list is empty and this is a
public reading of a public corpus. Signed in, the plan reaches for that
account's starred players a round early, which makes the body personal -- so
the response is marked private in BOTH cases. The alternative is a
`Cache-Control` that depends on whether a cookie was present, which is one
deploy away from a shared cache holding somebody's list under a URL every
reader sends. It is a cheap answer to compute and a cheap one to not cache.

NO EDITS ANYWHERE ELSE. This is a new module with one registration line in
`api/main.py`, deliberately: `api/account.py`, `api/billing.py` and
`scoring/plan.py` are all being changed on other branches at the same time,
and a route that only reads them cannot collide with any of that.
"""
from __future__ import annotations

import threading
import time

from fastapi import HTTPException, Request, Response

from api import billing, http_cache

# The seat, and how big the league is. The same bounds `api/account.py`'s
# outlook uses and stated here rather than imported: below four teams a snake
# is not a snake, above sixteen ESPN will not host it. A copy of two integers
# is cheaper than a dependency on a module three people are editing.
MIN_TEAMS = 4
MAX_TEAMS = 16
DEFAULT_TEAMS = 10
# The middle of a ten-team draft, clamped to the league actually asked about
# so `?teams=4` with no slot is answered rather than refused.
DEFAULT_SLOT = 5

# How many turns the preview shows. Three is the shape of a draft's opening --
# who you take, what you take next, and what the two of those leave you needing
# -- and it is as many cards as fit on a phone without a scroll.
PREVIEW_TURNS = 3

# How many turns are actually PLANNED, which is one more than are shown. Every
# turn's `edge_pts` is priced against the turn after it ("what taking him now
# beats waiting for"), so the last turn of a plan has no edge at all. Planning
# a fourth turn and throwing it away is what stops the third card printing a
# dash where every other card prints a number.
PLAN_TURNS = PREVIEW_TURNS + 1

# How many opening paths to name. Three is a shape ("this seat opens RB-heavy,
# or WR-heavy, or balanced"); eight is a table, which is what the archive's own
# seat page is for.
TOP_OPENINGS = 3

# The positions a drafter is deciding when to stop waiting on. RB and WR are
# not here on purpose: they go from pick one in every draft ever recorded, so
# "the median first RB is pick 1" is arithmetic dressed as a finding.
RUN_POSITIONS = ("QB", "TE", "K", "DST")

# How long an answer stands. The inputs are a seat, a favourites list and the
# board, and all three are in the key -- so this is not a staleness window, it
# is the thing that stops a reader dragging the two selects on the dashboard
# from rebuilding a plan per keystroke. A plan is ~40 ms of availability
# arithmetic over the whole board plus the board's own copy.
PREVIEW_TTL_SECONDS = 60.0

# Bounded because the key holds a favourites list, so it grows with readers
# rather than with pages -- the same reason `api/account.py` bounds its own.
_PREVIEW_MAX_ENTRIES = 64

# How often the board's identity is re-read. `board_key` is three DuckDB
# queries, which on a cache HIT would be the entire cost of the request; five
# seconds retires every entry well inside the minute they are kept for anyway.
_BOARD_IDENTITY_TTL_SECONDS = 5.0


def _float_or_none(value):
    """A JSON number, or null for a NaN. `float('nan')` is not valid JSON, and
    a column that is NaN by design -- ESPN publishes no ADP before drafts open
    -- deserves a null rather than a zero."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _text_or_none(value):
    """A string, or null. `None` and `NaN` both mean "nothing here", and a
    bare `value or None` would keep the NaN: it is truthy."""
    if value is None or value != value:
        return None
    text = str(value).strip()
    return text or None


def _snake_picks(teams: int, slot: int, rounds: int) -> list:
    """The overall pick numbers this seat owns, in order.

    The snake stated once: odd rounds run out from slot 1, even rounds run
    back. The same derivation `api/market.py` uses for its turn pages and
    `api/account.py` uses for the outlook's columns.
    """
    picks = []
    for rnd in range(1, rounds + 1):
        base = (rnd - 1) * teams
        picks.append(base + (slot if rnd % 2 == 1 else teams - slot + 1))
    return picks


def _openings(slot: int) -> dict:
    """What this seat's first five rounds look like, counted.

    THE SHAPE IS THE CORPUS'S, NOT THE CALLER'S. Every number here is read off
    recorded drafts, and the drafts that were recorded are one league size --
    whichever the farm has played most (`api/market._shape`). A caller asking
    about a twelve-team seat gets eight-team paths back, which is not a bug and
    is not hidden either: `corpus.teams` is in the answer so the page can say
    which drafts it counted. Answering with nothing at all until the farm has
    played twelve-team rooms would be worse -- the opening shape of a seat is
    the most transferable thing in this payload.

    A seat the corpus does not have (slot 11 of an eight-team archive) gets
    empty lists rather than a 404: the plan below is still perfectly good, and
    a page that 500s because one card has no data is a page that breaks for
    everybody the moment the farm changes shape.

    The path query is `api/market._slot_payload`'s, run against this module's
    own connection: one string per draft, this seat's first five positions in
    order, then grouped. A draft where the seat missed a pick to the clock is
    dropped by the HAVING rather than counted as a shorter path -- a
    four-position path is not a strategy, it is a gap.
    """
    from api import market

    empty = {"opening": [], "position_runs": {}, "corpus": None,
             "opening_rounds": market.SEQUENCE_ROUNDS, "opening_observed": 0}
    try:
        conn = market._corpus()
    except HTTPException:
        # No corpus at all, or one the farm is mid-write on. Neither is a
        # reason to refuse a plan.
        return empty
    try:
        try:
            teams, rounds = market._shape(conn)
        except HTTPException:
            return empty
        drafts = conn.execute(
            "SELECT count(*) FROM draft_log WHERE teams = ? AND rounds = ?",
            [teams, rounds]).fetchone()[0]
        corpus = {"teams": int(teams), "rounds": int(rounds),
                  "drafts": int(drafts)}
        if not 1 <= slot <= teams:
            return {**empty, "corpus": corpus}

        human = market._human_picks_sql()
        paths = conn.execute(f"""
            SELECT path, count(*) AS n FROM (
                SELECT pk.draft_id,
                       string_agg(pk.position, '-' ORDER BY pk.round) AS path
                FROM draft_log_pick pk JOIN draft_log d USING (draft_id)
                WHERE d.teams = ? AND d.rounds = ? AND pk.slot = ?
                  AND pk.round <= ? AND {human}
                GROUP BY 1 HAVING count(*) = ?
            ) GROUP BY 1 ORDER BY 2 DESC LIMIT ?
        """, [teams, rounds, slot, market.SEQUENCE_ROUNDS,
              market.SEQUENCE_ROUNDS, TOP_OPENINGS]).fetchall()

        # The denominator is every path walked from this seat, not the three
        # shown: a share out of the top three would read 100% across them and
        # say nothing about how decided the seat is. Counted with the same
        # HAVING, so it is the same population the numerators came from.
        walked = conn.execute(f"""
            SELECT count(*) FROM (
                SELECT pk.draft_id
                FROM draft_log_pick pk JOIN draft_log d USING (draft_id)
                WHERE d.teams = ? AND d.rounds = ? AND pk.slot = ?
                  AND pk.round <= ? AND {human}
                GROUP BY 1 HAVING count(*) = ?
            )
        """, [teams, rounds, slot, market.SEQUENCE_ROUNDS,
              market.SEQUENCE_ROUNDS]).fetchone()[0]

        # EVERY PICK COUNTS HERE, autodrafts included, and that is the
        # difference between this query and the one above it. A path is a
        # STRATEGY and only a person has one; a run is the board emptying, and
        # "when does the first quarterback go" does not care who took him.
        # Same call `api/market._players_payload` makes for the same reason.
        runs = conn.execute("""
            SELECT position, median(first_pick) FROM (
                SELECT pk.draft_id, pk.position, min(pk.pick_no) AS first_pick
                FROM draft_log_pick pk JOIN draft_log d USING (draft_id)
                WHERE d.teams = ? AND d.rounds = ?
                  AND pk.position IN (SELECT unnest(?))
                GROUP BY 1, 2
            ) GROUP BY 1
        """, [teams, rounds, list(RUN_POSITIONS)]).fetchall()
    finally:
        conn.close()

    return {
        "opening": [{"path": str(path).split("-"), "count": int(n),
                     "share": round(float(n) / walked, 4) if walked else 0.0}
                    for path, n in paths],
        "opening_rounds": market.SEQUENCE_ROUNDS,
        "opening_observed": int(walked),
        # Every position that was asked about, so a page can print a dash for
        # one the corpus has never seen rather than crash on a missing key.
        "position_runs": {pos: None for pos in RUN_POSITIONS}
        | {str(pos): _float_or_none(pick) for pos, pick in runs},
        "corpus": corpus,
    }


def _ranked_board(cur):
    """The cached board with ESPN's rank and ADP attached.

    `api/live.py._attach_espn_rank` is the one place that knows how those two
    columns are assembled, so this borrows it rather than growing a second,
    quietly divergent copy -- the same borrow `api/account.py` makes for the
    outlook. A board with no ESPN columns is not a failure: `availability_at`
    falls back to the consensus rank, and the plan is built from what is
    there.
    """
    from scoring.board_cache import cached_build_board

    board = cached_build_board(cur)
    try:
        from api.live import _attach_espn_rank
    except Exception:          # noqa: BLE001 -- no ESPN columns, not a failure
        return board
    return _attach_espn_rank(cur, board)


def _column(frame, name):
    """One column of `frame` as a float array, or a column of NaN.

    The missing column is a real path, not a defensive one: `_ranked_board`
    degrades to a board with no `espn_rank` and no `espn_adp` when `api/live`
    cannot be imported, and `pd.to_numeric(None)` is a scalar NaN rather than
    a Series -- fine right up to the `.to_numpy()` that follows it.
    """
    import numpy as np
    import pandas as pd

    if name not in frame.columns:
        return np.full(len(frame.index), np.nan, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _plan_rows(board, settings, turns: list, favourites: set) -> list:
    """`scoring/plan.py`'s answer for `turns`, with the players named.

    NOTHING IS RE-RANKED HERE. The target and its two alternates are whoever
    `build_plan` named, in the order it named them, with the reasons it wrote.
    This function's whole job is to turn player ids into rows a card can draw:
    a second implementation of "what should I take" that disagreed with the
    draft room by one name would be worse than no preview at all.

    `picks_made = 0` and an empty roster, which is what makes this a pre-draft
    answer: every probability is "from the start of the draft", nobody has
    been taken, and the plan drafts into an empty roster as it walks.
    """
    import numpy as np
    import pandas as pd

    from scoring.availability import cached_table
    from scoring.plan import build_plan, health_level

    ids = [str(pid) for pid in board["player_id"]]
    positions = np.asarray([str(p) for p in board["position"]], dtype=object)
    index = pd.Index(ids)
    frame = board.set_index(index)
    frame = frame[~frame.index.duplicated(keep="first")]

    names_col = frame.reindex(ids).get("name")
    names = {pid: (_text_or_none(n) or pid)
             for pid, n in zip(ids, [] if names_col is None else names_col)}

    proj = np.nan_to_num(_column(board, "proj_points"), nan=0.0)
    plan = build_plan(
        proj=proj, positions=positions, player_ids=ids,
        espn_rank=_column(board, "espn_rank"),
        espn_adp=_column(board, "espn_adp"),
        market_rank=_column(board, "market_rank"),
        byes=_column(board, "bye"),
        health=health_level(_column(board, "career_games_pg")),
        roster_counts={}, settings=settings, turns=turns, picks_made=0,
        favourites=set(favourites), table=cached_table(), names=names,
    )

    rows = {} if frame.empty else frame.to_dict("index")

    def _card(row):
        """One plan row as a card: who he is, plus the plan's own numbers."""
        if row is None:
            return None
        pid = str(row["player_id"])
        who = rows.get(pid) or {}
        return {
            "player_id": pid,
            "name": _text_or_none(who.get("name")) or pid,
            "position": _text_or_none(who.get("position")),
            "team": _text_or_none(who.get("team")),
            "headshot": _text_or_none(who.get("headshot")),
            "lasts_pct": _float_or_none(row.get("lasts_pct")),
            "edge_pts": _float_or_none(row.get("edge_pts")),
            "edge_at_pick": row.get("edge_at_pick"),
            "favourite": pid in favourites,
            "pros": list(row.get("pros") or ()),
            "cons": list(row.get("cons") or ()),
        }

    return [{
        "pick_no": int(turn["pick_no"]),
        "round": int(turn["round"]),
        "target": _card(turn.get("target")),
        "alternates": [_card(alt) for alt in (turn.get("alternates") or [])],
    } for turn in plan[:PREVIEW_TURNS]]


def register_plan_preview_routes(app, conn, store=None):
    """`GET /api/plan/preview`.

    `conn` is the board's connection, required for the same reason
    `api/account.py`'s registration requires one: the plan IS the board, and a
    registration that could be made without one would mount a route that can
    only fail. `store` is the credential store, passed through to
    `billing._account_ids` so a test can hand in its own.
    """
    # PER APP, not per module. The key carries the board's identity, so two
    # apps in one process could safely share the answers -- but not the
    # identity memo, which would let an app pointed at a second league file
    # read the first one's answer to "has the board moved".
    preview_cache: dict = {}
    preview_lock = threading.Lock()
    board_identity: list = [0.0, None]      # [read at, value]

    def _board_identity(cur):
        from scoring.board_cache import board_key

        with preview_lock:
            at, value = board_identity
            if value is not None and time.monotonic() - at < _BOARD_IDENTITY_TTL_SECONDS:
                return value
        # Outside the lock: three queries, and a second request arriving
        # during them should read the board rather than queue behind us.
        value = board_key(cur)
        with preview_lock:
            board_identity[:] = [time.monotonic(), value]
        return value

    @app.get("/api/plan/preview")
    def plan_preview(request: Request, response: Response,
                     teams: int = DEFAULT_TEAMS, slot: int | None = None):
        """What this seat would be told, before a draft exists.

        422 with a plain sentence for a seat that does not exist, matching the
        favourites outlook: the controls that send these are a pair of
        selects, so a value outside the bounds is a client bug and is worth
        naming rather than clamping silently.

        No 401. A reader who has connected nothing is the ordinary caller here
        -- this is the landing page -- and they get the same plan with no
        favourites folded into it.
        """
        if not MIN_TEAMS <= teams <= MAX_TEAMS:
            raise HTTPException(
                status_code=422,
                detail=f"A league has between {MIN_TEAMS} and {MAX_TEAMS} "
                       f"teams. That one has {teams}.")
        # After the team bound, so the clamp is never against a nonsense size.
        if slot is None:
            slot = min(DEFAULT_SLOT, teams)
        if not 1 <= slot <= teams:
            raise HTTPException(
                status_code=422,
                detail=f"Slot {slot} does not exist in a {teams}-team league.")

        # Signed out is not an error (see the docstring): no ids, no stars.
        account = billing._account_ids(request, store)
        try:
            saved = [str(pid) for pid in billing.favorites(account)] if account else []
        except billing.StoreError:
            # The favourites store being down costs this reader their stars,
            # not their plan. Every other number in the body is public.
            saved = []

        # PRIVATE ALWAYS, said out loud rather than left to a default. See the
        # module docstring: the body is personal whenever the caller has a
        # list, and a header that changed with the cookie is one deploy away
        # from a shared cache serving one account's stars to the next.
        http_cache.private(response)

        cur = conn.cursor()
        try:
            # The board's identity completes the key and is read BEFORE the
            # board is built: on a hit it is the only work the request does.
            key = (int(teams), int(slot), tuple(saved), _board_identity(cur))
            with preview_lock:
                hit = preview_cache.get(key)
                if hit is not None and time.monotonic() - hit[0] < PREVIEW_TTL_SECONDS:
                    return hit[1]

            from dataclasses import replace

            from scoring import league as league_mod

            # THE STORED LEAGUE'S SCORING, THE CALLER'S SIZE. Roster slots and
            # scoring rules come from whatever league this deployment has
            # imported (`league.load` falls back to the default league on a
            # fresh install); the team count is the seat being asked about,
            # which is what decides the snake and therefore every pick number
            # in the answer. Replacing one field of a frozen dataclass rather
            # than building a new one keeps the two in step: a league that
            # adds a flex slot changes the plan here as well as in the room.
            settings = replace(league_mod.load(cur), teams=int(teams))
            picks = _snake_picks(int(teams), int(slot), int(settings.rounds))
            answer = {
                "teams": int(teams),
                "slot": int(slot),
                "picks": picks[:4],
                **_openings(int(slot)),
                "targets": _plan_rows(_ranked_board(cur), settings,
                                      picks[:PLAN_TURNS], set(saved)),
            }
        finally:
            cur.close()

        with preview_lock:
            preview_cache[key] = (time.monotonic(), answer)
            while len(preview_cache) > _PREVIEW_MAX_ENTRIES:
                preview_cache.pop(next(iter(preview_cache)))
        return answer

    return app
