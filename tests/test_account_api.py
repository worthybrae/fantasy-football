"""The favourites a signed-in account keeps, over a real TestClient.

WHAT THIS IS GUARDING. Favourites are the first per-account preference this
product has ever stored, and they are stored beside the entitlements, keyed by
the same HMAC of the ESPN account. Three things therefore have to be true and
each has a test below:

  * a browser with no custody session gets 401 and writes nothing. The list is
    small and harmless, but the key it would be written under is the same one a
    $9.99 purchase is written under, and an endpoint that invents an account id
    for an anonymous caller is an endpoint that can be made to collide with a
    real one;
  * rows written under a retired custody key are still the account's
    favourites. Key rotation changes the id; it does not change who the person
    is, and `entitled()` already reads every version for exactly this reason;
  * a save REPLACES. A partial write -- the delete landing and the insert not
    -- would leave somebody who edited their list with no list at all, so the
    two statements are one transaction.

The board check is here rather than in the frontend because the id ends up in
the draft room's plan: an id that names nobody would be a star beside a blank
row, and the cheapest place to refuse it is the write.

Entirely offline: no ESPN call, no Stripe, no socket. The custody session is
faked the way `tests/test_custody_api.py` fakes one -- at `custody_for` and the
store behind it -- so the route runs its real `_account_ids` lookup.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import billing
from api.main import create_app
from pipeline.db import get_conn, write_table

POSITIONS = ["QB", "RB", "WR", "TE"]
TEAMS = ["DET", "GB", "KC", "SF", "BUF", "DAL"]

# The account this browser resolves to under the current custody key, and the
# id the same person's rows carry from before a rotation.
SWID = "{7A1F9C34-BEEF-4D01-9A55-C0FFEE001122}"
NEW_ID = "acct-key2"
OLD_ID = "acct-key1"


def _seed(path, n=30):
    """A board with enough players to fill a 25-long favourites list.

    Thirty rather than the one `tests/test_api.py` seeds, because the bounds
    this endpoint enforces are 5 and 25: a fixture that could not supply 26
    REAL ids would leave the upper bound tested only against ids the board
    check would have refused anyway, which proves nothing about the bound.
    """
    conn = get_conn(path)
    rows = []
    for i in range(1, n + 1):
        pos = POSITIONS[i % len(POSITIONS)]
        team = TEAMS[i % len(TEAMS)]
        opp = TEAMS[(i + 1) % len(TEAMS)]
        rows.extend(
            {"player_id": f"p{i}", "player_display_name": f"Player {i:02d}",
             "position": pos, "recent_team": team, "opponent_team": opp,
             "season": 2025, "week": w,
             "receptions": 8 - (i % 5), "receiving_yards": 100 - i,
             "targets": 10 - (i % 4), "carries": i % 7}
            for w in range(1, 18))
    write_table(conn, "weekly", pd.DataFrame(rows))
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": t, "away_team": TEAMS[(j + 1) % len(TEAMS)], "week": 1,
         "total_line": 45.0, "spread_line": 1.0}
        for j, t in enumerate(TEAMS)]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": f"Player {i:02d}", "position": POSITIONS[i % 4],
         "team": TEAMS[i % len(TEAMS)], "adp": float(i)}
        for i in range(1, n + 1)]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "snap_counts", pd.DataFrame(
        columns=["player", "team", "season", "offense_pct"]))
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp",
                 "espn_ppr_rank"]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave",
                 "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))
    conn.close()


@pytest.fixture(autouse=True)
def _isolated(tmp_path):
    """Never the real favourites database."""
    billing.reset_for_tests(str(tmp_path / "billing.duckdb"))
    yield
    billing.reset_for_tests(str(tmp_path / "billing.duckdb"))


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    """One seeded board for the file. Building it costs a second or two and
    nothing below writes to it, so every test can share the same one."""
    path = str(tmp_path_factory.mktemp("account") / "t.duckdb")
    _seed(path)
    return path


@pytest.fixture(scope="module")
def client(board):
    """One app for the file, for the same reason the board is.

    `create_app` is not free -- it mounts every router this server has, and
    two of them start background work in a deployment -- and nothing under
    test here is per-app: the favourites live in the billing store, which the
    autouse fixture above points at a fresh file for every test. Building
    fifteen apps to answer fifteen requests would only add fifteen sets of
    module-level state to a suite that already has tests counting board
    builds.
    """
    return TestClient(create_app(board))


def _sign_in(monkeypatch, ids=(NEW_ID,)):
    """The custody session a connected browser holds, faked at the two seams
    `billing._account_ids` really reads: the cookie resolver, and the
    credential store that turns a SWID into every id version it could hold
    rows under.

    Patched at `api.billing.custody_for`, NOT at `api.custody.custody_for`:
    billing imports the name, so the module that resolves the cookie for these
    routes is the one holding the reference."""
    monkeypatch.setattr(
        billing, "custody_for",
        lambda request, store=None: type("R", (), {"swid": SWID})())
    monkeypatch.setattr(
        billing, "_custody_store",
        lambda store=None: type("S", (), {
            "account_ids": staticmethod(lambda swid: list(ids))})())


def _signed_out(monkeypatch):
    """No cookie, and no saved local login to fall back on either."""
    monkeypatch.setattr(billing, "custody_for",
                        lambda request, store=None: None)
    monkeypatch.setattr("pipeline.espn_drafts.saved_session", lambda: None)


SIX = ["p7", "p3", "p19", "p11", "p2", "p25"]


# -- the round trip -----------------------------------------------------------


def test_an_account_starts_with_no_favourites(client, monkeypatch):
    _sign_in(monkeypatch)
    res = client.get("/api/account/favorites")

    assert res.status_code == 200
    assert res.json() == {"players": []}


def test_a_saved_list_comes_back_in_the_order_it_was_sent(client, monkeypatch):
    """ORDER IS THE POINT, not the set. The picker is an ordered list -- the
    first name is the one the plan reaches for first -- so a store that
    answered with the ids sorted, or in whatever order the rows came back,
    would silently rewrite everybody's preference."""
    _sign_in(monkeypatch)
    saved = client.put("/api/account/favorites", json={"players": SIX})

    assert saved.status_code == 200
    assert saved.json() == {"players": SIX}
    assert client.get("/api/account/favorites").json() == {"players": SIX}


def test_saving_again_replaces_rather_than_adds(client, monkeypatch):
    _sign_in(monkeypatch)
    client.put("/api/account/favorites", json={"players": SIX})
    second = ["p1", "p4", "p5", "p6", "p8"]

    client.put("/api/account/favorites", json={"players": second})

    # Not the union, and not eleven names: the second list is the whole answer.
    assert client.get("/api/account/favorites").json() == {"players": second}


def test_twenty_five_is_allowed(client, monkeypatch):
    _sign_in(monkeypatch)
    full = [f"p{i}" for i in range(1, 26)]

    assert client.put("/api/account/favorites",
                      json={"players": full}).status_code == 200
    assert client.get("/api/account/favorites").json()["players"] == full


# -- what is refused ----------------------------------------------------------


@pytest.mark.parametrize("count", [0, 4, 26])
def test_a_list_outside_five_to_twenty_five_is_refused(client, monkeypatch,
                                                       count):
    """Both bounds with REAL board ids, so the only thing that can have
    refused them is the count."""
    _sign_in(monkeypatch)
    players = [f"p{i}" for i in range(1, count + 1)]

    res = client.put("/api/account/favorites", json={"players": players})

    assert res.status_code == 422
    assert "5" in res.json()["detail"] and "25" in res.json()["detail"]
    # And nothing was written on the way to refusing.
    assert client.get("/api/account/favorites").json() == {"players": []}


def test_a_player_who_is_not_on_the_board_is_refused(client, monkeypatch):
    _sign_in(monkeypatch)
    res = client.put("/api/account/favorites",
                     json={"players": ["p1", "p2", "p3", "p4", "nobody"]})

    assert res.status_code == 422
    assert "nobody" in res.json()["detail"]
    assert client.get("/api/account/favorites").json() == {"players": []}


def test_the_same_player_twice_is_refused(client, monkeypatch):
    """The table's key is (account, player), so a duplicate cannot be stored
    twice however it is sent. Refusing says so; accepting would answer the
    next GET with a shorter list than the one that was saved."""
    _sign_in(monkeypatch)
    res = client.put("/api/account/favorites",
                     json={"players": ["p1", "p2", "p3", "p4", "p5", "p2"]})

    assert res.status_code == 422
    assert "p2" in res.json()["detail"]


def test_a_signed_out_browser_can_neither_read_nor_write(client, monkeypatch):
    """THE ONE THAT MATTERS. There is no account id to write under, and
    inventing one would put a stranger's rows under a key the entitlement
    table also uses."""
    _signed_out(monkeypatch)

    assert client.get("/api/account/favorites").status_code == 401
    assert client.put("/api/account/favorites",
                      json={"players": SIX}).status_code == 401


# -- key rotation -------------------------------------------------------------


def test_favourites_written_under_a_retired_key_are_still_read(client,
                                                               monkeypatch):
    """A rotated custody key computes a different id for the same person.
    `account_ids` yields both, newest first, and the rows keep the id they
    were written under -- so the read has to look under every version, exactly
    as `entitled()` does."""
    billing.set_favorites(OLD_ID, SIX)
    _sign_in(monkeypatch, ids=(NEW_ID, OLD_ID))

    assert client.get("/api/account/favorites").json() == {"players": SIX}


def test_a_save_lands_under_the_newest_key(client, monkeypatch):
    """Writes go to the current id, so a list saved today does not have to be
    found through a version that may be retired tomorrow."""
    _sign_in(monkeypatch, ids=(NEW_ID, OLD_ID))
    client.put("/api/account/favorites", json={"players": SIX})

    assert billing.favorites([NEW_ID]) == SIX
    assert billing.favorites([OLD_ID]) == []


def test_the_newest_key_with_rows_wins(client, monkeypatch):
    """A list saved after a rotation is the answer, not a merge with whatever
    the same person chose under the old key -- a merge could hold more than
    twenty-five names and none of them in a coherent order."""
    billing.set_favorites(OLD_ID, ["p1", "p2", "p3", "p4", "p5"])
    billing.set_favorites(NEW_ID, SIX)

    assert billing.favorites([NEW_ID, OLD_ID]) == SIX


# -- all or none --------------------------------------------------------------


def test_a_save_that_cannot_finish_leaves_the_old_list_alone():
    """THE REASON THE TWO STATEMENTS ARE ONE TRANSACTION. A replace is a
    DELETE and an INSERT; if the delete committed and the insert did not, an
    edit would leave the account with nothing, and the Dashboard would offer
    the onboarding panel to somebody who had already chosen.

    Forced through `set_favorites` directly rather than the route, which
    refuses a repeated id before the store ever sees it -- the point here is
    the store's own promise, not the endpoint's validation.
    """
    billing.set_favorites(NEW_ID, SIX)

    with pytest.raises(billing.StoreError):
        # A repeated id violates the table's (account, player) key halfway
        # through the insert.
        billing.set_favorites(NEW_ID, ["p1", "p2", "p1", "p4", "p5"])

    assert billing.favorites([NEW_ID]) == SIX


# -- the store being away -----------------------------------------------------


def test_a_store_that_cannot_answer_is_a_503(client, monkeypatch):
    """503 rather than the 500 an unhandled StoreError becomes, the same
    distinction every other route through this store draws: the request was
    fine and the database is not."""
    _sign_in(monkeypatch)

    def _broken(*_a, **_k):
        raise billing.StoreError("gone")

    monkeypatch.setattr(billing, "_db", _broken)

    assert client.get("/api/account/favorites").status_code == 503
    assert client.put("/api/account/favorites",
                      json={"players": SIX}).status_code == 503
