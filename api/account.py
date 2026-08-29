"""What a signed-in account keeps between drafts.

Today that is one thing: the five to twenty-five players it has starred. The
draft room's plan reaches for them first -- a favourite clears a lower
availability bar and carries a bonus into the target score (spec §4) -- so
this is a preference with teeth, not a decoration.

WHY THERE IS A MODULE FOR IT. This product has no accounts in the usual sense.
It has never asked for an email or a password; the only identity it has is the
one `api/billing.py` had to invent to attach a purchase to, which is
`store.account_id(swid)` -- a keyed hash of the connected ESPN account. These
routes borrow that identity and nothing else, which gives them their whole
shape:

  * a browser with no custody session has no id to write under, so it gets a
    401 rather than an invented one. Inventing one would put a stranger's rows
    under a key the entitlement table also uses;
  * a read looks under every id version `account_ids` yields, because rotating
    the custody key changes the id and does not change who the person is;
  * a write goes to the newest.

THE ONE EXCEPTION TO THE 401 is `GET /api/account/me`, which answers a
question an anonymous browser is allowed to ask: how many founder seats are
left, and does this browser hold one. It writes nothing under an invented id
-- with no session there is nothing to write -- and the page that reads it is
the landing page, where every reader is signed out by definition.

WHY THE IDS ARE CHECKED AGAINST THE BOARD. A favourite id travels into the
draft plan and comes back out as a star beside a row. An id that names nobody
would be a star beside a blank, or an entry the plan silently drops -- a bug
visible only during a live draft, which is the worst place to find one. The
board is already built and cached (`scoring/board_cache`), so the check costs
a set membership test at the one moment the value enters the system.

NO TRANSPORT GUARD OF ITS OWN, deliberately, and it is worth saying why given
that `api/custody.py`'s disconnect has one. Two reasons. The first is that the
protection is already there and is stronger: `CredentialTransportGuard` runs as
ASGI middleware, before routing and before body validation, and refuses ANY
plaintext request carrying the custody cookie whatever path it is for. The
second is that a `require_secure` here would refuse the owner's own machine:
`billing._account_ids` falls back to the saved local ESPN login on a loopback
request with no forwarding headers, and a loopback request is `http` by
definition. Nothing here reads or writes a credential; the body is a list of
public player ids.

That first reason is INHERITED, not owned, and the registration below says so
again: the guard is installed by `custody.install_credential_guards`, which
runs from `register_custody_routes`. `api/main.py` mounts both on the same
app, which is why these routes are safe there. An app that mounted these
routes WITHOUT the custody ones would have no transport guard at all.
"""
import threading
import time

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel

from api import billing, http_cache
from api.billing import StoreError
from pipeline import redact

# Both bounds are the product's, not the store's (spec §5). The floor exists
# because a plan built from two names is not a plan; the ceiling because a
# list of everybody ranks nobody, and because it caps what one account can
# write into a shared table.
MIN_FAVORITES = 5
MAX_FAVORITES = 25

# -- your guys, by pick -------------------------------------------------------
#
# WHAT THE OUTLOOK IS. The favourites list says who somebody wants. It does
# not say whether he can have them, and on a 12-team board from the eighth
# seat the honest answer for half the list is "no". `GET
# /api/account/favorites/outlook` draws that: for each favourite, the chance
# he is still on the board at each of the first eight turns of a chosen seat,
# read off the SAME counted table the live room reads (`scoring/availability`,
# a ratio of recorded-draft counts, not a simulation).
#
# CONDITIONED ON NOTHING, which is what makes it a pre-draft answer rather
# than a room one. The room asks `availability_at(..., k=picks_made, ...)`:
# given the draft is here and he is still on the board, will he last. This
# asks with `k = 0` -- from the start, before anybody has picked -- because
# nobody is drafting yet and there is no board state to condition on. That is
# also why pick 1 reads 100% for everybody and why it should: the first pick
# of a draft cannot have taken anyone away.

# Eight turns, i.e. the first eight rounds. Far enough that a favourite the
# corpus never lets past round two has visibly run out, short enough that the
# grid still fits a phone without becoming a spreadsheet.
OUTLOOK_ROUNDS = 8

# The seat, and how big the league is. Bounds are the product's: below four
# teams a snake is not a snake, above sixteen ESPN will not host it.
MIN_TEAMS = 4
MAX_TEAMS = 16
DEFAULT_TEAMS = 10
# The middle of a ten-team draft, CLAMPED to the league actually asked about:
# `?teams=4` with no slot is a caller saying "the smallest league, wherever
# you like", and answering it with a 422 about seat five would be refusing a
# request nobody made.
DEFAULT_SLOT = 5

# "Still likely there." A coin flip is the line, so `best_pick` is the last
# turn at which waiting is better than even -- the pick a reader can plan to
# take him at rather than the pick he has to reach at.
BEST_PICK_FLOOR = 50.0

# HOW LONG AN ANSWER IS KEPT, and what it is an answer to. The inputs are a
# saved list, a seat, and the board -- none of which move during the minute
# somebody spends sliding the two controls, and all three of which are IN the
# key, so a stale answer here is one served inside sixty seconds of a board
# rebuild rather than one served under the wrong list. The cost it removes is
# not the availability arithmetic (half a millisecond for twenty-five players
# across eight picks) but the board: `cached_build_board` copies its frame on
# every call and ESPN's ranks are attached on top, which is ~13 ms per request
# for an answer that has not changed.
OUTLOOK_TTL_SECONDS = 60.0

# Bounded because the key holds a favourites list, so it grows with readers
# rather than with pages. Sixty-four is a few dozen people sliding controls at
# once; past that the oldest entry goes.
_OUTLOOK_MAX_ENTRIES = 64

# HOW OFTEN THE SERVER'S OWN HALF OF THE KEY IS RE-READ, and why it is not
# read per request. `board_cache.board_key` is three DuckDB queries against
# the league file -- 3 ms on an idle connection and three to five times that
# on a server thread competing for one -- which on a cache HIT is the entire
# cost of the request. The league's scoring format (the other half of the
# shape the answer is computed for) is one more such read, on the same timer
# and for the same reason. Both can change without anybody asking, so they
# get a timer instead of being dropped from the key: a rebuilt board retires
# every entry within five seconds, comfortably inside the minute an answer is
# kept for anyway.
_BOARD_IDENTITY_TTL_SECONDS = 5.0


class FavoritesBody(BaseModel):
    players: list[str]


# How many of the offending ids a refusal names, and how much of each one.
# The message goes into a response body, an access log and whatever error
# reporter the page installs, and every character of it came from the caller.
_SHOWN = 5
_SHOWN_CHARS = 40


def _quote(ids) -> str:
    """The ids a refusal names: a few of them, short, and scrubbed.

    THREE BOUNDS, because this is client input on its way back out. The count,
    so a list of twenty-five wrong ids is not twenty-five wrong ids in a log
    line. The length of each, so one id cannot be a page of text. And
    `redact`, the same scrubber `custody.safe_validation_error_handler` puts
    FastAPI's own 422 through -- an id is not a credential, but nothing stops
    somebody pasting one into this field, and the answer to "what did I just
    send" should never be "here it is again, in your network tab".
    """
    shown = sorted(ids)[:_SHOWN]
    text = ", ".join(redact.redact(str(i))[:_SHOWN_CHARS] for i in shown)
    extra = len(ids) - len(shown)
    return f"{text} and {extra} more" if extra > 0 else text


def _picks_for(teams: int, slot: int) -> list:
    """The overall pick numbers this seat owns in the first `OUTLOOK_ROUNDS`
    rounds of a snake.

    The same derivation the live room uses (`api/live.py`'s `rank_and_plan`):
    lay the snake out slot by slot and take the offsets that are mine. Overall
    pick numbers, never round numbers, because that is the axis
    `availability_at` measures on.
    """
    from scoring.draft_sim import snake_slots

    snake = snake_slots(int(teams), OUTLOOK_ROUNDS)
    return [i + 1 for i, seat in enumerate(snake) if seat == int(slot)]


def _float_or_none(value):
    """A JSON number, or null for a NaN. `None` is the honest answer for a
    column that is NaN by design -- ESPN publishes no ADP before drafts open
    -- and `float('nan')` is not valid JSON."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _int_or_none(value):
    number = _float_or_none(value)
    return None if number is None else int(number)


def _numeric_column(frame, name):
    """One column of `frame` as a float Series, or a column of NaN.

    `frame.get(name)` answers `None` for a column that is not there, and
    `pd.to_numeric(None, errors="coerce")` is a scalar NaN rather than a
    Series -- fine right up to the `.to_numpy()` that follows it, which is an
    AttributeError and a 500. The missing column is a REAL path, not a
    defensive one: `_ranked_board` degrades to a board with no `espn_rank` and
    no `espn_adp` when `api/live.py` cannot be imported. Mirrors
    `api/live.py._board_column`, which exists for the same reason.
    """
    import numpy as np
    import pandas as pd

    values = None if name not in frame.columns else frame[name]
    if values is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(values, errors="coerce").astype(float)


def _text_or_none(value):
    """A string, or null for a missing one. `None` and `NaN` both mean "this
    column has nothing for him", and a bare `value or None` would keep the
    NaN: `float('nan')` is truthy, and it is not valid JSON."""
    if value is None or value != value:
        return None
    text = str(value).strip()
    return text or None


def _ranked_board(cur):
    """The cached board with ESPN's rank and ADP attached.

    `api/live.py._attach_espn_rank` is the one place that knows how those two
    columns are assembled -- the lobby's PPR rank with the printable cheat
    sheet filling in behind it, and the ADP joined out of `espn_adp` by
    `espn_id` only when the season's column is usable at all -- so this
    borrows it rather than growing a second, quietly divergent copy. Imported
    here rather than at the top of the file: `api/live.py` is a large module
    and this route is the only thing in this one that needs it.

    A board with no ESPN columns is not a failure. `availability_at` reads the
    ADP only for the players the corpus cannot answer for, and falls back to
    the consensus rank for those; the response simply carries nulls in the two
    ESPN fields.
    """
    from scoring.board_cache import cached_build_board

    board = cached_build_board(cur)
    try:
        from api.live import _attach_espn_rank
    except Exception:          # noqa: BLE001 -- no ESPN columns, not a failure
        return board
    return _attach_espn_rank(cur, board)


def _outlook_players(board, saved: list, picks: list,
                     teams=None, fmt=None) -> list:
    """One row per saved favourite, in the saved order.

    THE ORDER IS THE PREFERENCE (see `billing.favorites`), so the answer is
    built by walking `saved` rather than by walking the board.

    `teams` and `fmt` are the shape the answer is for -- the size the reader
    picked, and the scoring of the league this deployment holds -- and go
    straight to `availability_at`, which reads that shape's own counts once
    the corpus holds enough drafts of it and the pooled ones until then.

    A saved id the board no longer names keeps its row and carries nulls. It
    is a rare case -- the write checked every id against the board -- but the
    board is rebuilt from new data and a player can leave it, and the two
    alternatives are both worse: dropping the row silently shortens somebody's
    list, and asking `availability_at` about an id it has never seen returns
    the "nobody has ranked him, so nobody is about to draft him" answer, which
    would print an unknown player at 100% for every pick.
    """
    import numpy as np
    import pandas as pd

    from scoring.availability import availability_at, cached_table

    # KEYED BY STRING, AND DEDUPLICATED. `reindex` refuses an index with any
    # repeated label at all -- not just a repeated one that was asked for --
    # so a board carrying a player twice would be a 500 rather than a
    # duplicated row. It should not happen and it is one line to survive.
    board = board.set_index(pd.Index([str(pid) for pid in board["player_id"]]))
    board = board[~board.index.duplicated(keep="first")]
    known = [pid for pid in saved if pid in board.index]
    rows = board.reindex(known) if known else None

    curves = {}
    if known:
        espn_adp = _numeric_column(rows, "espn_adp")
        market_rank = _numeric_column(rows, "market_rank")
        positions = np.asarray([str(p) for p in rows["position"]], dtype=object)
        table = cached_table()
        # One vectorised call per pick rather than one per player: twenty-five
        # players is one gather out of the counts, and eight of those is the
        # whole computation.
        at_pick = {pick: availability_at(
            table, known, 0, pick,
            espn_adp.to_numpy(dtype=float), market_rank.to_numpy(dtype=float),
            positions=positions, teams=teams, fmt=fmt) for pick in picks}
        for i, pid in enumerate(known):
            curves[pid] = [round(float(at_pick[pick][i]) * 100, 1)
                           for pick in picks]

    def _best(curve) -> int | None:
        """The LAST pick still better than even, not the first. The question
        the card answers is "how long can I wait", so a favourite who reads 90%
        at pick 5 and 60% at pick 16 is a pick-16 player."""
        found = None
        for pick, chance in zip(picks, curve):
            if chance is not None and chance >= BEST_PICK_FLOOR:
                found = pick
        return found

    known_rows = {} if rows is None else {
        pid: row for pid, row in zip(known, rows.to_dict("records"))}
    players = []
    for pid in saved:
        row = known_rows.get(pid)
        curve = curves.get(pid, [None] * len(picks))
        players.append({
            "player_id": pid,
            "name": None if row is None else _text_or_none(row.get("name")),
            "position": None if row is None else _text_or_none(row.get("position")),
            "team": None if row is None else _text_or_none(row.get("team")),
            "headshot": None if row is None else _text_or_none(row.get("headshot")),
            "espn_rank": None if row is None else _int_or_none(row.get("espn_rank")),
            "espn_adp": None if row is None else _float_or_none(row.get("espn_adp")),
            "market_rank": None if row is None else _int_or_none(row.get("market_rank")),
            "avail": curve,
            "best_pick": _best(curve),
        })
    return players


def register_account_routes(app, conn, store=None):
    """`GET /api/account/me`, and `GET`/`PUT /api/account/favorites`.

    `conn` is the board's connection, and it is REQUIRED rather than optional
    like the connection every other router here takes: the write's only real
    validation is "does this id name a player", and a registration that could
    be made without a board would be a registration that silently accepts
    anything. `store` is the credential store, passed through to
    `billing._account_ids` so a test can hand in its own.

    MOUNT `register_custody_routes` ON THE SAME APP. These routes have no
    transport guard of their own (see the module docstring); the one that
    covers them is the ASGI middleware `custody.install_credential_guards`
    installs, and `register_custody_routes` is what installs it. `api/main.py`
    is the only place that mounts both, and it is the only configuration these
    routes are safe in.
    """

    def _account(request: Request) -> tuple:
        """`(ids, source)` for a request that must have an account, else 401.

        The source matters to exactly one thing here -- whether a founder seat
        may be claimed -- and it comes back with the ids because working it
        out separately would mean resolving the cookie and reading the
        credential store twice for one request.
        """
        ids, source = billing.account_context(request, store)
        if not ids:
            raise HTTPException(
                status_code=401,
                detail="Connect your ESPN account to keep a favourites list.")
        return ids, source

    def _account_ids(request: Request) -> list:
        """The ids alone, for the two routes that give nothing away."""
        return _account(request)[0]

    # PER APP, not per module. The key below carries the board's identity, so
    # two apps in one process could safely share a dictionary -- but the
    # board-identity memo could not, and an app pointed at a second league
    # file would read the first one's answer to "has the board moved". Both
    # live here instead, which also means a test's app starts cold.
    outlook_cache: dict = {}
    outlook_lock = threading.Lock()
    # [read at, (board key, the league's scoring format)]
    server_identity: list = [0.0, None]

    def _identity(cur):
        """What the SERVER brings to an answer: `(board key, the league's
        scoring format)`.

        Both are re-read at most every `_BOARD_IDENTITY_TTL_SECONDS` -- see
        that constant for why the board's key is on a timer rather than on
        every request, and note that the format is the same kind of quantity:
        a DuckDB read whose answer changes only when a league is imported,
        which is also when the board is rebuilt. One timer over the pair
        rather than two, so a cache HIT still does no queries at all.

        THE SHAPE IS HALF THE READER'S AND HALF THE LEAGUE'S. `teams` is a
        control on the page -- somebody planning for a ten-team draft can ask
        about one -- but the scoring is not: this page is read by an account
        connected to one ESPN league, and asking it to name its own format
        would be asking a question the server can already answer. A
        deployment with no league imported reads "ppr", which is what
        `scoring.league.scoring_format` says about a league it cannot see.
        """
        from scoring import league as league_mod
        from scoring.board_cache import board_key

        with outlook_lock:
            at, value = server_identity
            if value is not None and time.monotonic() - at < _BOARD_IDENTITY_TTL_SECONDS:
                return value
        # Outside the lock: it is a handful of queries, and a second request
        # arriving during them should read the board rather than queue behind
        # us.
        value = (board_key(cur), league_mod.scoring_format(league_mod.load(cur)))
        with outlook_lock:
            server_identity[:] = [time.monotonic(), value]
        return value

    def _board_ids() -> set:
        """Every player id the board knows, as strings.

        Through the cache, so this is a dictionary lookup during a draft
        rather than the ~1.6 s `build_board` costs cold -- the same cache
        `/api/players` and the live room read, so an id this route accepts is
        an id the room can name.
        """
        from scoring.board_cache import cached_build_board

        cur = conn.cursor()
        try:
            board = cached_build_board(cur)
        finally:
            cur.close()
        return {str(player_id) for player_id in board["player_id"]}

    @app.get("/api/account/me")
    def account_me(request: Request, response: Response):
        """Who this browser is to us, which today is one question: founder?

        200 WITHOUT A SESSION, and that is the whole reason this route reads
        the account the quiet way rather than through the `_account_ids` above.
        The landing page is the main caller and its reader is by definition not
        connected yet -- "N founder spots left, connect to claim one" is an
        offer, and an offer that 401s is a page that cannot make it. So an
        anonymous caller gets `connected: false`, `founder: false`, and the
        same honest count of seats as everybody else.

        PRIVATE, said out loud rather than left to `DefaultPrivate`. Two
        readers get different bodies from the same URL with no query string
        between them, so this is the exact shape a shared cache serves to the
        wrong person. The seat count alone would be cacheable; the sentence
        above it is not.

        AND THE SEAT IS CLAIMED HERE. This is the request the dashboard makes
        on every load, so it is the one that covers somebody who never opens a
        real league's draft at all -- see `billing.claim_founder` for why the
        claim is a side effect of routes that already hold the account rather
        than an endpoint of its own.
        """
        http_cache.private(response)
        ids, source = billing.account_context(request, store)
        try:
            # Ordered: the claim first, so a seat taken by THIS request is
            # already out of the count the same response reports.
            #
            # AND ONLY FOR A CONNECTED ACCOUNT. The other source is the saved
            # ESPN login on the machine this process runs on, which is an
            # identity for reading and not one to give a seat to: on a
            # checkout with the deployment's `.env` loaded, that id is
            # computed from a LOCAL custody key and written to the SHARED
            # store, so the seat would belong to nobody the deployment can
            # ever recognise. See `billing.CONNECTED`.
            ordinal = (billing.claim_founder(ids)
                       if ids and source == billing.CONNECTED else None)
            left = billing.founders_left()
        except StoreError as exc:
            raise billing._unavailable(exc) from None
        return {"connected": bool(ids),
                "founder": ordinal is not None,
                "ordinal": ordinal,
                "founders_left": left}

    @app.get("/api/account/favorites")
    def account_favorites(request: Request):
        """This account's list, in its own order. Empty until it saves one."""
        ids, source = _account(request)
        # A founder's seat, taken on the way past, and only for a connected
        # account -- see `/api/account/me` above for what the other source is
        # and why it may not claim. Quietly: this route exists to answer a
        # different question, and a billing store that is briefly away is not
        # a reason to refuse somebody their own list.
        if source == billing.CONNECTED:
            billing.claim_founder_quietly(ids)
        try:
            return {"players": billing.favorites(ids)}
        except StoreError as exc:
            raise billing._unavailable(exc) from None

    @app.put("/api/account/favorites")
    def save_account_favorites(body: FavoritesBody, request: Request):
        """Replace the list. All of it, or none of it.

        422 for every refusal, with the reason in `detail` as a plain sentence:
        the picker shows it verbatim, and a client that sent 26 names needs to
        be told which rule it broke rather than that "the body was invalid".
        """
        ids = _account_ids(request)
        players = [str(player_id).strip() for player_id in body.players]

        if not MIN_FAVORITES <= len(players) <= MAX_FAVORITES:
            raise HTTPException(
                status_code=422,
                detail=f"Pick between {MIN_FAVORITES} and {MAX_FAVORITES} "
                       f"players. That list has {len(players)}.")

        # Before the board check, because a duplicate is a client bug with a
        # specific fix, and because the table's key is (account, player): a
        # list stored with the duplicate dropped would come back shorter than
        # the one that was sent, which is worse than a refusal.
        seen: set = set()
        repeated = set()
        for player_id in players:
            if player_id in seen:
                repeated.add(player_id)
            seen.add(player_id)
        if repeated:
            raise HTTPException(
                status_code=422,
                detail=f"That list names the same player twice: "
                       f"{_quote(repeated)}.")

        unknown = set(players) - _board_ids()
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Not players on this board: {_quote(unknown)}.")

        try:
            stored = billing.set_favorites(ids, players)
        except StoreError as exc:
            raise billing._unavailable(exc) from None
        return {"players": stored}

    @app.get("/api/account/favorites/outlook")
    def account_favorites_outlook(request: Request, response: Response,
                                  teams: int = DEFAULT_TEAMS,
                                  slot: int | None = None):
        """Each favourite's chance of still being there at each of my picks.

        `teams` and `slot` describe a seat rather than a league: this is the
        page somebody reads BEFORE they have connected anything, so there is
        no league settings row to take them from and they arrive as query
        parameters with the most common draft (ten teams, the middle seat) as
        the default.

        422 with a plain sentence for a seat that does not exist, matching the
        PUT above -- the controls that send these are a pair of selects, so a
        value outside the bounds is a client bug and deserves to be named.
        """
        account = _account_ids(request)
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

        try:
            saved = [str(pid) for pid in billing.favorites(account)]
        except StoreError as exc:
            raise billing._unavailable(exc) from None

        # PRIVATE, and said out loud rather than left to `DefaultPrivate`.
        # The body is one person's favourites list; a shared cache holding it
        # under a URL every reader sends would serve one account's players to
        # the next.
        http_cache.private(response)

        cur = conn.cursor()
        try:
            # The board's identity is read BEFORE the board is built: it
            # completes this answer's key, and on a hit it is the only work
            # the request does.
            identity = _identity(cur)
            fmt = identity[1]
            key = (tuple(saved), int(teams), int(slot), identity)
            with outlook_lock:
                hit = outlook_cache.get(key)
                if hit is not None and time.monotonic() - hit[0] < OUTLOOK_TTL_SECONDS:
                    return hit[1]

            picks = _picks_for(teams, slot)
            answer = {
                "teams": int(teams),
                "slot": int(slot),
                "picks": picks,
                "players": _outlook_players(_ranked_board(cur), saved, picks,
                                            teams=int(teams), fmt=fmt),
            }
        finally:
            cur.close()

        with outlook_lock:
            outlook_cache[key] = (time.monotonic(), answer)
            while len(outlook_cache) > _OUTLOOK_MAX_ENTRIES:
                outlook_cache.pop(next(iter(outlook_cache)))
        return answer

    return app
