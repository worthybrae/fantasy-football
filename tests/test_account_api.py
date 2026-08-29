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
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

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

    Closed at the end of the module rather than left to the garbage
    collector, so the app's shutdown handlers run inside this file instead of
    whenever the interpreter next feels like it.
    """
    with TestClient(create_app(board)) as ready:
        yield ready


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


def test_an_id_that_is_not_a_string_is_refused_without_being_echoed(
        client, monkeypatch):
    """Pydantic refuses the body before the handler sees it, and the 422 that
    comes back carries no `input` key.

    That last half is the point and it is not FastAPI's default:
    `custody.safe_validation_error_handler` replaces the default handler
    precisely because Pydantic attaches the WHOLE rejected body to every error
    as `input`. Pinned here because these routes inherit that handler from
    `register_custody_routes` -- an app that mounted them alone would echo the
    body instead."""
    _sign_in(monkeypatch)
    res = client.put("/api/account/favorites",
                     json={"players": [1, 2, 3, 4, 5]})

    assert res.status_code == 422
    problems = res.json()["detail"]
    assert isinstance(problems, list) and problems
    assert all(set(problem) <= {"type", "loc", "msg"} for problem in problems)


def test_a_refusal_names_only_a_few_ids_and_scrubs_them(client, monkeypatch):
    """The refusal quotes the caller back at itself, so it is bounded on every
    axis: how many ids, how long each one is, and whether it can smuggle
    something credential-shaped through `redact`."""
    _sign_in(monkeypatch)
    huge = "x" * 500
    res = client.put("/api/account/favorites",
                     json={"players": ["a1", "b2", "c3", "d4", "e5", "f6",
                                       "g7", huge]})

    assert res.status_code == 422
    detail = res.json()["detail"]
    assert "and 3 more" in detail            # eight wrong ids, five named
    assert huge not in detail                # and none of them at full length
    assert len(detail) < 400


def test_a_signed_out_browser_can_neither_read_nor_write(client, monkeypatch):
    """THE ONE THAT MATTERS. There is no account id to write under, and
    inventing one would put a stranger's rows under a key the entitlement
    table also uses."""
    _signed_out(monkeypatch)

    assert client.get("/api/account/favorites").status_code == 401
    assert client.put("/api/account/favorites",
                      json={"players": SIX}).status_code == 401


def test_reading_the_list_never_builds_the_board(client, monkeypatch):
    """The board is the WRITE's validation and has no business on the read.

    It matters during a draft: the room polls, and a GET that touched
    `cached_build_board` would be a cache lookup on a good day and a
    seconds-long rebuild on the day the cache was just invalidated by a pick.
    """
    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], SIX)

    def _explode(*_a, **_k):
        raise AssertionError("the read must not build a board")

    monkeypatch.setattr("scoring.board_cache.cached_build_board", _explode)

    assert client.get("/api/account/favorites").json() == {"players": SIX}


# -- key rotation -------------------------------------------------------------


def test_favourites_written_under_a_retired_key_are_still_read(client,
                                                               monkeypatch):
    """A rotated custody key computes a different id for the same person.
    `account_ids` yields both, newest first, and the rows keep the id they
    were written under -- so the read has to look under every version, exactly
    as `entitled()` does."""
    billing.set_favorites([OLD_ID], SIX)
    _sign_in(monkeypatch, ids=(NEW_ID, OLD_ID))

    assert client.get("/api/account/favorites").json() == {"players": SIX}


def test_a_save_lands_under_the_newest_key(client, monkeypatch):
    """Writes go to the current id, so a list saved today does not have to be
    found through a version that may be retired tomorrow."""
    _sign_in(monkeypatch, ids=(NEW_ID, OLD_ID))
    client.put("/api/account/favorites", json={"players": SIX})

    assert billing.favorites([NEW_ID]) == SIX
    assert billing.favorites([OLD_ID]) == []


def test_clearing_a_list_clears_it_under_every_key(client, monkeypatch):
    """A save DELETES under every id version and writes the newest.

    Delete only under the newest and a rotated account that empties its list
    finds the old key's rows again on the next read, because `favorites` falls
    through to the version that still has rows -- the clear silently reverts
    to whatever they chose before the rotation. Latent while the endpoint's
    floor is five names, and a bug the moment anything can clear one.
    """
    billing.set_favorites([OLD_ID], SIX)

    billing.set_favorites([NEW_ID, OLD_ID], [])

    assert billing.favorites([NEW_ID, OLD_ID]) == []
    assert billing.favorites([OLD_ID]) == []


def test_a_save_needs_somewhere_to_put_it():
    """No account ids is a caller bug, not an empty write: the delete would
    match nothing and the insert would have no id to use."""
    with pytest.raises(ValueError):
        billing.set_favorites([], SIX)


def test_the_newest_key_with_rows_wins(client, monkeypatch):
    """A list saved after a rotation is the answer, not a merge with whatever
    the same person chose under the old key -- a merge could hold more than
    twenty-five names and none of them in a coherent order."""
    billing.set_favorites([OLD_ID], ["p1", "p2", "p3", "p4", "p5"])
    billing.set_favorites([NEW_ID], SIX)

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
    billing.set_favorites([NEW_ID], SIX)

    with pytest.raises(billing.StoreError):
        # A repeated id violates the table's (account, player) key halfway
        # through the insert.
        billing.set_favorites([NEW_ID], ["p1", "p2", "p1", "p4", "p5"])

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


# -- boot work is not this request's work -------------------------------------
#
# WHAT HAPPENED. On 2026-08-27, minutes after a deploy, the Railway edge log
# recorded one `/api/account/favorites` at 24,026 ms with the two either side
# of it at 85 ms. It was the first thing in that container to ask the billing
# store a question, and `billing._db()` answered by making it wait for the
# corpus seed: every mock room the farm had recorded since the table was last
# topped up, written one row per statement, one commit and one fsync each, on
# a network volume. Boot work, sized by how far the corpus had run ahead --
# and no request should ever be able to discover how big that is.

FIRST_REQUEST_BUDGET_SECONDS = 1.0


def _corpus_of_mocks(path, count):
    """A draft corpus with `count` mock rooms in it, as `draft_log` writes."""
    import duckdb
    from pipeline import draft_log as dl

    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE draft_log (draft_id VARCHAR, source VARCHAR, "
                 "league_id VARCHAR)")
    conn.executemany("INSERT INTO draft_log VALUES (?, ?, ?)",
                     [[f"d{i}", dl.SOURCE_MOCK, str(i)] for i in range(count)])
    conn.close()
    return str(path)


@contextmanager
def _held_write_lock(path):
    """Another process holding the corpus read-write, which is the farm.

    A real subprocess rather than a second connection in this one: DuckDB's
    single-writer lock is what the farm actually takes, and an in-process
    second open fails for a different reason (a configuration clash) down a
    different line of the driver.
    """
    code = ("import sys, time, duckdb\n"
            "c = duckdb.connect(sys.argv[1])\n"
            "c.execute('SELECT 1').fetchall()\n"
            "print('held', flush=True)\n"
            "time.sleep(120)\n")
    holder = subprocess.Popen([sys.executable, "-c", code, str(path)],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held", "no lock holder"
        yield
    finally:
        holder.terminate()
        holder.wait(timeout=30)


def test_the_first_request_is_quick_with_the_corpus_locked(
        client, monkeypatch, tmp_path):
    """The farm holds the corpus most of the time, and the seed's answer to
    that is to give up. Giving up must be immediate: a request that waited on
    somebody else's write lock would be a stall nobody could reproduce,
    because whether it happens at all depends on what the farm is doing."""
    from pipeline import draft_log as dl

    corpus = _corpus_of_mocks(tmp_path / "corpus.duckdb", 900)
    monkeypatch.setattr(dl, "CORPUS_PATH", corpus)
    billing.reset_for_tests(str(tmp_path / "locked-billing.duckdb"))
    _sign_in(monkeypatch)

    with _held_write_lock(corpus):
        started = time.monotonic()
        res = client.get("/api/account/favorites")
        took = time.monotonic() - started

    assert res.status_code == 200
    assert took < FIRST_REQUEST_BUDGET_SECONDS, f"{took:.2f}s on the first read"


def test_the_first_request_does_not_pay_for_the_corpus_backlog(
        client, monkeypatch, tmp_path):
    """The unlocked case, which is the one that actually cost 24 seconds.

    A corpus the table has never seen is the ordinary state of a container
    that has just booted onto a volume the farm has been writing to, and the
    seed still has to happen. The claim being tested is only that it happens
    SOMEWHERE ELSE, so the write is held open rather than merely made large:
    a stopwatch against a real backlog measures how quick this laptop is, and
    a gate measures the thing that broke.
    """
    from pipeline import draft_log as dl

    corpus = _corpus_of_mocks(tmp_path / "corpus.duckdb", 20)
    monkeypatch.setattr(dl, "CORPUS_PATH", corpus)
    billing.reset_for_tests(str(tmp_path / "backlog-billing.duckdb"))
    _sign_in(monkeypatch)

    # The seed's write, stopped mid-flight. On a volume whose fsync costs
    # milliseconds this is what 854 rooms one row at a time felt like.
    released = threading.Event()
    real_insert = billing._insert_mock_rooms

    def held(store, league_ids):
        assert released.wait(10), "the seed was never released"
        return real_insert(store, league_ids)

    monkeypatch.setattr(billing, "_insert_mock_rooms", held)

    started = time.monotonic()
    res = client.get("/api/account/favorites")
    took = time.monotonic() - started

    assert res.status_code == 200
    assert took < FIRST_REQUEST_BUDGET_SECONDS, f"{took:.2f}s on the first read"

    # And the work was not skipped, only moved: every room is in the table
    # once the seed's own thread gets to finish.
    released.set()
    assert billing._seed_done.wait(30)
    assert billing.is_free_draft("7") is True


# -- your guys, by pick -------------------------------------------------------
#
# WHAT THE OUTLOOK IS FOR. The favourites list is a wish; the outlook is
# whether the wish survives contact with a draft. Every number in it comes out
# of the same counted table the live room reads, so the things worth pinning
# here are the ones that are this ROUTE's rather than
# `scoring/availability.py`'s:
#
#   * it needs a session, like every other route in this file -- the answer is
#     one person's list and would otherwise be served to anybody who asked;
#   * a seat that does not exist is refused rather than clamped. Two selects
#     send these, so a value outside the bounds is a client bug;
#   * the rows follow the SAVED order. The order is the preference, and an
#     answer sorted by anything else -- rank, availability, the board -- is a
#     different person's list;
#   * a favourite the corpus always takes in the first three picks reads zero
#     from pick five on. That is the whole product claim, and it is asserted
#     against a corpus whose counts are known by construction.
#
# The corpus is synthetic and pinned at `cached_table` for exactly that last
# reason: the real one is a data file, and a test whose expected value comes
# out of it is a test that changes every time the farm plays a mock.

from pipeline import draft_log as dl                            # noqa: E402
from scoring import availability as av                          # noqa: E402

# Enough drafts to clear `availability.MIN_DRAFTS`, or every ratio below falls
# through to the fitted curve and stops being a count.
OUTLOOK_DRAFTS = 30

# Who the synthetic corpus takes, and when. `p11` at pick 55 is what gives the
# corpus a depth worth conditioning on: without a deep pick nothing past the
# third would be a counted answer at all.
ALWAYS_EARLY = {"p7": 1, "p3": 2, "p19": 3}
LATE = {"p11": 55}

# THE ONE WHOSE ANSWER CROSSES THE LINE. Taken at pick 20 in two drafts out of
# every three and at pick 30 in the third, so from the default seat he reads
# 100% at picks 5 and 16 and 33% at pick 25 -- his last turn above a coin flip
# is the SECOND of them, not the first. `best_pick` is the only field with a
# direction to get wrong, and this player is the one that says which way.
SPLIT = "p6"
SPLIT_EARLY = 20
SPLIT_LATE = 30

NEVER = "p25"
OUTLOOK_LIST = ["p7", "p3", "p19", SPLIT, "p11", NEVER, "p2"]


def _corpus(path, drafts=OUTLOOK_DRAFTS):
    """A corpus with hand-countable histories, over the seeded board's ids.

    Modelled on `tests/test_availability.py::_write_corpus` -- same recorder,
    same row shapes -- but drafted from `p1..p30` so the ids line up with the
    board this file seeds and the route can be asked about real favourites.
    """
    conn = dl.corpus_conn(str(path))
    try:
        for draft in range(drafts):
            pool, picks = [], []
            for i, player_id in enumerate(OUTLOOK_LIST):
                pool.append({"player_id": player_id, "position": "RB",
                             "team": "FA", "adp_rank": float(i + 1),
                             "proj_points": 200.0, "espn_rank": float(i + 1),
                             "espn_proj": 200.0, "bye": 5})
            taken = {**ALWAYS_EARLY, **LATE,
                     SPLIT: SPLIT_LATE if draft % 3 == 0 else SPLIT_EARLY}
            for player_id, pick_no in taken.items():
                picks.append({"pick_no": pick_no, "round": 1 + (pick_no - 1) // 8,
                              "slot": 1, "owner_key": "o", "is_anonymous": False,
                              "player_id": player_id, "position": "RB"})
            dl.record(conn, dl.DraftRecord(
                source=dl.SOURCE_MOCK, league_id="1", season=2026,
                started_at=f"outlook-{draft}", teams=8, rounds=16,
                picks=pd.DataFrame(picks), pool=pd.DataFrame(pool)))
    finally:
        conn.close()
    return str(path)


@pytest.fixture(scope="module")
def outlook_table(tmp_path_factory):
    """One counted table for the file. Nothing below writes to it."""
    return av.load_table(
        _corpus(tmp_path_factory.mktemp("outlook") / "corpus.duckdb"))


@pytest.fixture
def counted(monkeypatch, outlook_table):
    """Pin the route's availability table to the synthetic corpus.

    At `scoring.availability.cached_table`, which is where the route looks it
    up -- `api/account.py` imports the name inside the function that uses it,
    so the module attribute is what it reads.
    """
    monkeypatch.setattr(av, "cached_table", lambda *_a, **_k: outlook_table)
    return outlook_table


def _outlook(client, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return client.get("/api/account/favorites/outlook"
                      + (f"?{query}" if query else ""))


def test_the_outlook_needs_a_session(client, monkeypatch):
    """THE SAME GATE THE LIST ITSELF HAS. There is no account id to read
    favourites under, and an outlook is those favourites with a number beside
    each one."""
    _signed_out(monkeypatch)

    assert _outlook(client).status_code == 401


@pytest.mark.parametrize("params", [
    {"teams": 3}, {"teams": 17}, {"teams": 10, "slot": 0},
    {"teams": 10, "slot": 11}, {"teams": 12, "slot": 13},
])
def test_a_seat_that_does_not_exist_is_refused(client, monkeypatch, counted,
                                               params):
    """422 with a sentence, not a clamp. The controls that send these are two
    selects; a value outside the bounds means the client is confused, and
    answering for some other seat would hide it."""
    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], OUTLOOK_LIST)

    res = _outlook(client, **params)

    assert res.status_code == 422
    assert isinstance(res.json()["detail"], str)


def test_the_shape(client, monkeypatch, counted):
    """Every key the card reads, and the picks belonging to the seat asked for.

    Ten teams and the fifth seat is the default draft, and its first eight
    turns are the snake's own: 5, then 16 (round two counts back), then 25.
    """
    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], OUTLOOK_LIST)

    res = _outlook(client)

    assert res.status_code == 200
    assert res.headers["cache-control"] == "private, no-store"
    body = res.json()
    assert body["teams"] == 10 and body["slot"] == 5
    assert body["picks"] == [5, 16, 25, 36, 45, 56, 65, 76]
    assert len(body["players"]) == len(OUTLOOK_LIST)
    for player in body["players"]:
        assert set(player) == {
            "player_id", "name", "position", "team", "headshot", "espn_rank",
            "espn_adp", "market_rank", "avail", "best_pick"}
        assert len(player["avail"]) == len(body["picks"])
        assert all(chance is None or 0.0 <= chance <= 100.0
                   for chance in player["avail"])
        assert player["best_pick"] is None or player["best_pick"] in body["picks"]


def test_the_seat_decides_the_picks(client, monkeypatch, counted):
    """The first seat of an eight-team league turns at 1 and 16, not at 5."""
    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], OUTLOOK_LIST)

    body = _outlook(client, teams=8, slot=1).json()

    assert body["picks"] == [1, 16, 17, 32, 33, 48, 49, 64]


def test_a_player_the_corpus_always_takes_early_is_gone_by_pick_five(
        client, monkeypatch, counted):
    """THE CLAIM THE CARD MAKES. Thirty drafts, and every one of them took
    `p7` with the first pick -- so the share of them in which he was still
    there when pick 5 was made is zero, and it is zero at every turn after it.

    `p25` is the control: pooled in all thirty and taken in none, so the same
    arithmetic reads 100 and the zero cannot be the route answering zero for
    everybody. Only as far as the corpus goes, though -- its deepest recorded
    pick is 55, and a player it never saw taken is right-censored past that,
    so picks 65 and 76 are the fitted curve's answer rather than a count (see
    `scoring/availability.py` on censoring). The five turns inside the depth
    are the counted ones.
    """
    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], OUTLOOK_LIST)

    body = _outlook(client).json()
    rows = {player["player_id"]: player for player in body["players"]}

    assert body["picks"][:5] == [5, 16, 25, 36, 45]
    assert rows["p7"]["avail"] == [0.0] * len(body["picks"])
    assert rows["p7"]["best_pick"] is None
    assert rows[NEVER]["avail"][:5] == [100.0] * 5
    assert rows[NEVER]["best_pick"] is not None


def test_the_best_pick_is_the_last_turn_above_a_coin_flip(client, monkeypatch,
                                                          counted):
    """NOT THE FIRST ONE. The card answers "how long can I wait", so a
    favourite who reads 100% at pick 5, 100% at pick 16 and 33% at pick 25 is
    a pick-16 player -- and a `best_pick` that took the first turn over the
    line would call him a pick-5 player, which is the same as saying reach for
    him now.

    `p6` is taken at pick 20 in two drafts out of every three and at pick 30
    in the third, so the crossing is between two turns rather than at one.
    """
    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], OUTLOOK_LIST)

    body = _outlook(client).json()
    row = next(p for p in body["players"] if p["player_id"] == SPLIT)

    assert body["picks"][:3] == [5, 16, 25]
    assert row["avail"][:3] == [100.0, 100.0, 33.3]
    assert row["best_pick"] == 16


def test_the_rows_follow_the_saved_order(client, monkeypatch, counted):
    """ORDER IS THE PREFERENCE (the same rule the GET above keeps). Sorting by
    rank, or by how likely each one is to last, would be a different person's
    list -- and the saved order here is deliberately neither."""
    _sign_in(monkeypatch)
    reversed_six = list(reversed(OUTLOOK_LIST))
    billing.set_favorites([NEW_ID], reversed_six)

    body = _outlook(client).json()

    assert [player["player_id"] for player in body["players"]] == reversed_six


def test_a_favourite_the_board_no_longer_names_carries_nulls(
        client, monkeypatch, counted):
    """A saved id can outlive the board that validated it -- the board is
    rebuilt from new data and players leave it.

    The row stays, with nulls. Dropping it would silently shorten somebody's
    list; asking the counts about an id nothing has ranked returns "nobody is
    about to draft him", which would print a player who no longer exists at
    100% for every pick.
    """
    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], ["p7", "gone-from-the-board", "p3", "p19",
                                     "p11"])

    body = _outlook(client).json()
    missing = body["players"][1]

    assert missing["player_id"] == "gone-from-the-board"
    assert missing["name"] is None and missing["market_rank"] is None
    assert missing["avail"] == [None] * len(body["picks"])
    assert missing["best_pick"] is None


def test_saving_a_second_list_changes_the_outlook(client, monkeypatch, counted):
    """THE FAVOURITES ARE IN THE CACHE KEY, and this is the test that says so.

    Both requests are inside the sixty seconds an answer is kept for, and
    everything else about them -- the seat, the board, the session -- is
    identical. A key made of only those would answer the second one with the
    list that was replaced, which is the state somebody lands in the instant
    they close the picker.
    """
    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], OUTLOOK_LIST)
    first = _outlook(client, teams=12, slot=6).json()

    shorter = OUTLOOK_LIST[:5]
    billing.set_favorites([NEW_ID], shorter)
    second = _outlook(client, teams=12, slot=6).json()

    assert [p["player_id"] for p in first["players"]] == OUTLOOK_LIST
    assert [p["player_id"] for p in second["players"]] == shorter


def test_a_league_size_on_its_own_gets_a_seat_that_exists_in_it(
        client, monkeypatch, counted):
    """`?teams=4` is a caller saying "the smallest league, wherever you like".
    The default seat is the middle of a TEN-team draft, so answering it
    literally would refuse the request with a 422 about a fifth seat nobody
    asked for."""
    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], OUTLOOK_LIST)

    body = _outlook(client, teams=4).json()

    assert body["teams"] == 4 and body["slot"] == 4
    assert body["picks"] == [4, 5, 12, 13, 20, 21, 28, 29]


def test_the_board_can_arrive_without_espns_own_columns(client, monkeypatch,
                                                        counted):
    """`_ranked_board` hands back the plain board when `api/live.py` cannot be
    imported, and the plain board has neither `espn_rank` nor `espn_adp`.

    The two fields go null and every percentage is still counted -- ESPN's ADP
    is only read for the players the corpus cannot answer for, and the corpus
    answers for all of these. Before `_numeric_column`, this path was an
    AttributeError inside `to_numpy` and a 500.
    """
    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], OUTLOOK_LIST)
    monkeypatch.delattr("api.live._attach_espn_rank")

    res = _outlook(client, teams=8, slot=2)

    assert res.status_code == 200
    rows = {p["player_id"]: p for p in res.json()["players"]}
    assert rows["p7"]["espn_rank"] is None and rows["p7"]["espn_adp"] is None
    assert rows["p7"]["market_rank"] is not None      # the board's own column
    assert rows["p7"]["avail"] == [0.0] * 8
    assert rows[NEVER]["avail"][0] == 100.0


def test_a_board_that_names_a_player_twice_still_answers(client, monkeypatch,
                                                         counted):
    """A repeated `player_id` used to be a 500 for everybody, not a duplicated
    row for one: `reindex` refuses an index with ANY repeated label, whether or
    not the repeat is one of the ids being asked about."""
    from scoring import board_cache

    real = board_cache.cached_build_board

    def _doubled(cur, *args, **kwargs):
        board = real(cur, *args, **kwargs)
        return pd.concat([board, board.head(2)], ignore_index=True)

    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], OUTLOOK_LIST)
    monkeypatch.setattr(board_cache, "cached_build_board", _doubled)

    res = _outlook(client, teams=8, slot=3)

    assert res.status_code == 200
    assert len(res.json()["players"]) == len(OUTLOOK_LIST)


def test_the_same_question_is_answered_once(client, monkeypatch, counted):
    """The board is the expensive half of this answer and none of its inputs
    move while somebody slides two selects, so a repeat is served from the
    cache. Counted at `_outlook_players`, which is everything downstream of
    the board build."""
    from api import account

    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], OUTLOOK_LIST)
    built = []
    real = account._outlook_players
    monkeypatch.setattr(account, "_outlook_players",
                        lambda *a, **k: (built.append(1), real(*a, **k))[1])

    first = _outlook(client, teams=14, slot=9).json()
    second = _outlook(client, teams=14, slot=9).json()

    assert first == second
    assert len(built) == 1
    # A different seat is a different question, so it is asked.
    _outlook(client, teams=14, slot=10)
    assert len(built) == 2


def test_a_store_that_cannot_answer_is_a_503_here_too(client, monkeypatch,
                                                      counted):
    _sign_in(monkeypatch)

    def _broken(*_a, **_k):
        raise billing.StoreError("gone")

    monkeypatch.setattr(billing, "_db", _broken)

    assert _outlook(client).status_code == 503


# -- founders, over the same routes -------------------------------------------
#
# `GET /api/account/me` is the one route in this file that answers an
# anonymous browser, and the reason is the landing page: "N founder spots
# left -- connect ESPN to claim one" is an offer, and an offer that 401s is a
# page that cannot make it. Everything else here is the claim itself, which is
# a side effect of routes that exist to answer other questions (see
# `billing.claim_founder`) rather than an endpoint of its own -- so the tests
# below make ordinary requests and then ask what happened.

SEATS = 5


@pytest.fixture
def seats(monkeypatch):
    """A small, known number of founder seats.

    Set rather than left to the default: the real one is a hundred, and a test
    that filled it would be a hundred claims to prove one rule.
    """
    monkeypatch.setenv(billing.FOUNDERS_LIMIT_ENV, str(SEATS))
    return SEATS


def test_a_signed_out_browser_is_told_how_many_seats_are_left(
        client, monkeypatch, seats):
    """200, not 401. The reader of this answer has not connected anything yet
    -- that is the entire point of showing it to them."""
    _signed_out(monkeypatch)

    res = client.get("/api/account/me")

    assert res.status_code == 200
    assert res.json() == {"connected": False, "founder": False,
                          "ordinal": None, "founders_left": SEATS}
    # One person's answer, and never in a shared cache: two browsers get
    # different bodies from this URL with no query string between them.
    assert res.headers["cache-control"] == "private, no-store"


def test_asking_who_i_am_claims_a_seat(client, monkeypatch, seats):
    """The dashboard makes this request on every load, so it is the one that
    covers somebody who never opens a real league's draft at all."""
    _sign_in(monkeypatch)

    body = client.get("/api/account/me").json()

    assert body == {"connected": True, "founder": True, "ordinal": 1,
                    "founders_left": SEATS - 1}


def test_asking_twice_is_the_same_seat(client, monkeypatch, seats):
    """A hundred pageviews must not be a hundred founders."""
    _sign_in(monkeypatch)
    first = client.get("/api/account/me").json()

    assert client.get("/api/account/me").json() == first
    assert billing.founders_taken() == 1


def test_a_rotated_key_keeps_the_seat_it_already_has(client, monkeypatch,
                                                     seats):
    """Rotation changes the id and does not change who the person is. A read
    that only looked at the newest would lose them their seat and spend
    another one on them."""
    _sign_in(monkeypatch, ids=(OLD_ID,))
    assert client.get("/api/account/me").json()["ordinal"] == 1

    _sign_in(monkeypatch, ids=(NEW_ID, OLD_ID))
    after = client.get("/api/account/me").json()

    assert after["ordinal"] == 1
    assert after["founders_left"] == SEATS - 1


def test_reading_the_favourites_claims_a_seat_too(client, monkeypatch, seats):
    """The other authenticated read this app makes on a dashboard load. It
    claims for the same reason: the account is already in hand."""
    _sign_in(monkeypatch)

    assert client.get("/api/account/favorites").status_code == 200

    body = client.get("/api/account/me").json()
    assert body["ordinal"] == 1
    assert body["founders_left"] == SEATS - 1


def test_the_favourites_read_survives_a_billing_store_that_cannot_claim(
        client, monkeypatch, seats):
    """The claim is a side effect. A store that is briefly away is a reason to
    serve the list without a founder badge, not a reason to refuse the list."""
    _sign_in(monkeypatch)
    monkeypatch.setattr(billing, "claim_founder",
                        lambda ids: (_ for _ in ()).throw(
                            billing.StoreError("gone")))

    assert client.get("/api/account/favorites").json() == {"players": []}


def test_a_latecomer_is_told_the_seats_are_gone(client, monkeypatch, seats):
    """Zero left, and no invented ordinal for somebody who has none."""
    for i in range(SEATS):
        billing.claim_founder([f"earlier-{i}"])
    _sign_in(monkeypatch)

    body = client.get("/api/account/me").json()

    assert body == {"connected": True, "founder": False, "ordinal": None,
                    "founders_left": 0}


def test_me_is_a_503_when_the_store_cannot_answer(client, monkeypatch, seats):
    """The count is the answer here, so an unreadable store has nothing
    honest to say -- 503, like every other route in this file."""
    _sign_in(monkeypatch)

    def _broken(*_a, **_k):
        raise billing.StoreError("gone")

    monkeypatch.setattr(billing, "_db", _broken)

    assert client.get("/api/account/me").status_code == 503


# -- the machine the server runs on -------------------------------------------


def _local_login(monkeypatch):
    """No cookie, and the saved ESPN login answering instead.

    Patched at `_local_swid` rather than by faking a loopback address: a
    TestClient request comes from "testclient", so `is_local_request` would
    refuse it and the branch under test would never run.
    """
    monkeypatch.setattr(billing, "custody_for",
                        lambda request, store=None: None)
    monkeypatch.setattr(billing, "_local_swid", lambda request: SWID)
    monkeypatch.setattr(
        billing, "_custody_store",
        lambda store=None: type("S", (), {
            "account_ids": staticmethod(lambda swid: [NEW_ID])})())


def test_the_owners_own_machine_is_never_made_a_founder(client, monkeypatch,
                                                        seats):
    """It reads rows under a real id and is given nothing.

    That id is computed from whatever custody key the machine happens to have
    -- a local one on a checkout, even with the deployment's `.env` loaded and
    the shared store underneath -- so a seat claimed here is one the
    deployment can never match to a person, and one fewer for a real reader.
    The first `make up` would have taken #1.
    """
    _local_login(monkeypatch)

    body = client.get("/api/account/me").json()

    # Connected, because it IS an account for everything else it does.
    assert body == {"connected": True, "founder": False, "ordinal": None,
                    "founders_left": SEATS}
    assert billing.founders_taken() == 0


def test_the_favourites_read_claims_nothing_for_a_local_login(
        client, monkeypatch, seats):
    """The other claiming route, under the same rule."""
    _local_login(monkeypatch)

    assert client.get("/api/account/favorites").status_code == 200
    assert billing.founders_taken() == 0
def test_the_outlook_asks_about_the_shape_the_reader_chose(client, monkeypatch,
                                                           counted):
    """THE SEAT IS A SHAPE, not just a set of pick numbers. `availability_at`
    reads a shape's own counts once the corpus holds enough drafts of it, so
    a twelve-team question has to arrive as a twelve-team question -- the
    picks alone would leave it answered from eight-team drafts forever.

    The format is the LEAGUE's, not the reader's: this page is read by an
    account connected to one ESPN league, and the server already knows what
    that league scores.
    """
    _sign_in(monkeypatch)
    billing.set_favorites([NEW_ID], OUTLOOK_LIST)
    asked = []
    real = av.availability_at

    def spy(table, ids, k, n, *args, **kwargs):
        asked.append((kwargs.get("teams"), kwargs.get("fmt")))
        return real(table, ids, k, n, *args, **kwargs)

    monkeypatch.setattr(av, "availability_at", spy)

    assert _outlook(client, teams=12, slot=3).status_code == 200

    assert asked, "the route never asked the counted table anything"
    assert {teams for teams, _ in asked} == {12}
    # This fixture's board has no ESPN league imported, and "ppr" is what
    # `scoring.league.scoring_format` says about a league it cannot see -- the
    # format every source has always been read as.
    assert {fmt for _, fmt in asked} == {"ppr"}


def test_the_outlook_players_pass_the_shape_straight_through(monkeypatch):
    """The other half of the same wire, tested where the value can be forced:
    a standard-scoring league asks about standard-scoring drafts."""
    import api.account as account

    board = pd.DataFrame([{"player_id": "p7", "name": "Player 07",
                           "position": "RB", "team": "DET", "espn_rank": 1.0,
                           "espn_adp": 1.0, "market_rank": 1}])
    asked = []

    def spy(table, ids, k, n, *args, **kwargs):
        asked.append((kwargs.get("teams"), kwargs.get("fmt")))
        import numpy as np
        return np.ones(len(list(ids)))

    monkeypatch.setattr(av, "availability_at", spy)
    monkeypatch.setattr(av, "cached_table", lambda *_a, **_k: av.AvailabilityTable.empty())

    account._outlook_players(board, ["p7"], [1, 16], teams=12, fmt="std")

    assert asked == [(12, "std"), (12, "std")]
