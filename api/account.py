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
"""
from fastapi import HTTPException, Request
from pydantic import BaseModel

from api import billing
from api.billing import StoreError

# Both bounds are the product's, not the store's (spec §5). The floor exists
# because a plan built from two names is not a plan; the ceiling because a
# list of everybody ranks nobody, and because it caps what one account can
# write into a shared table.
MIN_FAVORITES = 5
MAX_FAVORITES = 25


class FavoritesBody(BaseModel):
    players: list[str]


def register_account_routes(app, conn, store=None):
    """`GET`/`PUT /api/account/favorites`.

    `conn` is the board's connection, and it is REQUIRED rather than optional
    like the connection every other router here takes: the write's only real
    validation is "does this id name a player", and a registration that could
    be made without a board would be a registration that silently accepts
    anything. `store` is the credential store, passed through to
    `billing._account_ids` so a test can hand in its own.
    """

    def _account_ids(request: Request) -> list:
        ids = billing._account_ids(request, store)
        if not ids:
            raise HTTPException(
                status_code=401,
                detail="Connect your ESPN account to keep a favourites list.")
        return ids

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

    @app.get("/api/account/favorites")
    def account_favorites(request: Request):
        """This account's list, in its own order. Empty until it saves one."""
        ids = _account_ids(request)
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
                       f"{', '.join(sorted(repeated))}.")

        unknown = sorted(set(players) - _board_ids())
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Not players on this board: {', '.join(unknown)}.")

        try:
            stored = billing.set_favorites(ids[0], players)
        except StoreError as exc:
            raise billing._unavailable(exc) from None
        return {"players": stored}

    return app
