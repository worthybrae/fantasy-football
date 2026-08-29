"""`GET /api/plan/preview` -- the plan a seat gets before a draft exists.

WHAT THIS IS GUARDING. The front door shows a stranger what draft night would
look like, and it does it by running the room's own planner against an empty
board. Four properties make that honest rather than decorative, and each has a
test below:

  * the shape is the shape a card can draw -- three turns, a target and two
    alternates on each, every player named and positioned. A card that has to
    guess at a missing key is a card that renders a dash on the page that is
    supposed to sell the tool;
  * a seat that does not exist is refused in a sentence, not clamped. The
    controls that send these are two selects, so an out-of-range value is a
    client bug and deserves to be named;
  * a signed-in caller's favourites reach the plan. That is the whole
    difference between this and a public ranking, and it is the reason the
    answer is marked private;
  * a warm hit does not rebuild the board. The board is the expensive part of
    every request here (~13 ms of frame copy on a hit, seconds cold), and the
    endpoint is called from a page with two selects on it.

Entirely offline: no ESPN call, no Stripe, no socket, and a corpus written by
hand into a temporary file. The custody session is faked the way
`tests/test_account_api.py` fakes one -- at `billing.custody_for` and the store
behind it -- so the route runs its real `_account_ids` lookup.
"""
import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.market as market
from api import billing
from api.main import create_app
from pipeline.db import get_conn, write_table

POSITIONS = ["QB", "RB", "WR", "TE"]
TEAMS = ["DET", "GB", "KC", "SF", "BUF", "DAL"]

SWID = "{7A1F9C34-BEEF-4D01-9A55-C0FFEE001122}"
ACCOUNT = "acct-preview"


def _seed(path, n=40):
    """A board with enough players to plan four turns out of.

    Forty rather than a handful: `build_plan` strikes a turn's three names off
    every later turn, so a board of ten would run out of eligible players
    before the fourth turn and the test would be measuring the fixture.
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
    path = str(tmp_path_factory.mktemp("preview") / "t.duckdb")
    _seed(path)
    return path


@pytest.fixture(scope="module")
def client(board):
    """One app for the file. `create_app` mounts every router this server has
    and is not free; nothing under test here is per-app except the preview
    cache, which the tests that care about it key around explicitly."""
    with TestClient(create_app(board)) as ready:
        yield ready


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A tiny archive: four 4-team, 5-round drafts, all walking one path from
    seat 2 except the last, which walks a different one. Five rounds because
    that is `SEQUENCE_ROUNDS`, and a path one round short is dropped by the
    query's own HAVING rather than counted."""
    path = tmp_path / "corpus.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("""CREATE TABLE draft_log (
        draft_id VARCHAR, source VARCHAR, league_id VARCHAR, season INTEGER,
        recorded_at TIMESTAMP, teams INTEGER, rounds INTEGER, my_slot INTEGER,
        scoring_json VARCHAR, settings_json VARCHAR, human_seats INTEGER)""")
    conn.execute("""CREATE TABLE draft_log_pick (
        draft_id VARCHAR, pick_no INTEGER, round INTEGER, slot INTEGER,
        owner_key VARCHAR, is_anonymous BOOLEAN, player_id VARCHAR,
        position VARCHAR, adp_rank DOUBLE, proj_points DOUBLE,
        autodrafted BOOLEAN, had_owner BOOLEAN, seconds_to_pick DOUBLE,
        clock_seconds DOUBLE)""")
    conn.execute("""CREATE TABLE draft_log_pool (
        draft_id VARCHAR, player_id VARCHAR, position VARCHAR, team VARCHAR,
        adp_rank DOUBLE, proj_points DOUBLE, espn_rank DOUBLE,
        espn_proj DOUBLE, bye INTEGER)""")

    def _pick(draft, pick_no, rnd, slot, pos, auto=False):
        conn.execute(
            "INSERT INTO draft_log_pick VALUES (?, ?, ?, ?, 'o', FALSE,"
            " ?, ?, 1, 1, ?, TRUE, 5.0, 30.0)",
            [draft, pick_no, rnd, slot, f"{pos}{pick_no}", pos, auto])

    for i, draft in enumerate(("d1", "d2", "d3", "d4")):
        conn.execute(
            "INSERT INTO draft_log VALUES (?, 'mock', '1', 2026,"
            " TIMESTAMP '2026-08-24 12:00:00', 4, 5, 4, NULL, NULL, 3)",
            [draft])
        # Seat 2 of a 4-team snake owns picks 2, 7, 10, 15, 18.
        path_walked = (["RB", "WR", "WR", "RB", "TE"] if i < 3
                       else ["WR", "WR", "RB", "RB", "TE"])
        for rnd, (pick_no, pos) in enumerate(
                zip((2, 7, 10, 15, 18), path_walked), start=1):
            _pick(draft, pick_no, rnd, 2, pos)
        # A quarterback in round 3 of every draft, taken by seat 1, so the
        # median first QB is a number these tests can state.
        _pick(draft, 9, 3, 1, "QB")
    conn.close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    yield path
    market._CACHE.clear()


def _sign_in(monkeypatch, ids=(ACCOUNT,)):
    """The custody session a connected browser holds, faked at the two seams
    `billing._account_ids` really reads."""
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


# -- the shape ----------------------------------------------------------------


def test_a_stranger_gets_a_plan(client, monkeypatch):
    """No account, no 401. This is the landing page: the ordinary caller here
    has connected nothing, and a front door that refuses anonymous readers is
    a front door nobody opens."""
    _signed_out(monkeypatch)

    res = client.get("/api/plan/preview?teams=10&slot=6")

    assert res.status_code == 200
    body = res.json()
    assert body["teams"] == 10 and body["slot"] == 6


def test_the_picks_are_this_seat_s_own_snake(client, monkeypatch):
    """Seat 6 of ten owns 6, then 15 (the second round running back), then 26,
    then 35. Pure arithmetic, and the one number on the page a reader can
    check against their own draft board."""
    _signed_out(monkeypatch)

    body = client.get("/api/plan/preview?teams=10&slot=6").json()

    assert body["picks"] == [6, 15, 26, 35]


def test_three_turns_each_with_a_target_and_two_alternates(client, monkeypatch):
    """The shape the cards draw. Three turns because that is the opening of a
    draft; a target and two alternates because that is what the room shows."""
    _signed_out(monkeypatch)

    body = client.get("/api/plan/preview?teams=10&slot=6").json()

    assert len(body["targets"]) == 3
    assert [t["pick_no"] for t in body["targets"]] == [6, 15, 26]
    assert [t["round"] for t in body["targets"]] == [1, 2, 3]
    for turn in body["targets"]:
        assert turn["target"] is not None
        assert len(turn["alternates"]) == 2


def test_every_named_player_carries_what_a_card_prints(client, monkeypatch):
    """A card needs a name, a position and the plan's two numbers. An id is
    storage; printing one on the front door would be showing a reader our
    database instead of their draft."""
    _signed_out(monkeypatch)

    body = client.get("/api/plan/preview?teams=10&slot=6").json()

    target = body["targets"][0]["target"]
    for key in ("player_id", "name", "position", "team", "headshot",
                "lasts_pct", "edge_pts", "pros", "cons", "favourite"):
        assert key in target, key
    assert target["name"] != target["player_id"]
    assert target["position"] in POSITIONS
    assert isinstance(target["pros"], list)


def test_the_third_turn_still_has_an_edge(client, monkeypatch):
    """A turn's edge is priced against the turn AFTER it, so the last turn of
    a plan has none. The endpoint plans a fourth turn and throws it away for
    exactly this: the third card must not be the only one printing a dash."""
    _signed_out(monkeypatch)

    body = client.get("/api/plan/preview?teams=10&slot=6").json()

    third = body["targets"][2]["target"]
    assert third["edge_pts"] is not None
    assert third["edge_at_pick"] == 35


def test_the_answer_is_never_written_down_by_a_shared_cache(client, monkeypatch):
    """Private even signed out. The body becomes personal the moment the
    caller has a favourites list, and a header that changed with the cookie is
    one deploy away from serving one account's stars to the next."""
    _signed_out(monkeypatch)

    res = client.get("/api/plan/preview?teams=10&slot=6")

    assert "private" in res.headers["cache-control"].lower()


# -- what is refused ----------------------------------------------------------


@pytest.mark.parametrize("teams", [3, 17, 0])
def test_a_league_size_nobody_plays_is_refused(client, monkeypatch, teams):
    _signed_out(monkeypatch)

    res = client.get(f"/api/plan/preview?teams={teams}")

    assert res.status_code == 422
    assert "4" in res.json()["detail"] and "16" in res.json()["detail"]


def test_a_seat_the_league_does_not_have_is_refused(client, monkeypatch):
    _signed_out(monkeypatch)

    res = client.get("/api/plan/preview?teams=8&slot=11")

    assert res.status_code == 422
    assert "11" in res.json()["detail"] and "8" in res.json()["detail"]


def test_no_slot_is_the_middle_of_the_league_asked_about(client, monkeypatch):
    """`?teams=4` with no slot is "the smallest league, wherever you like",
    and answering it with a 422 about seat five would refuse a request nobody
    made."""
    _signed_out(monkeypatch)

    assert client.get("/api/plan/preview?teams=12").json()["slot"] == 5
    assert client.get("/api/plan/preview?teams=4").json()["slot"] == 4


# -- the corpus ---------------------------------------------------------------


def test_the_opening_is_what_this_seat_actually_walked(client, corpus,
                                                       monkeypatch):
    """Three of four recorded drafts open RB-WR-WR-RB-TE from seat 2, so that
    path leads with a 0.75 share of the paths walked -- not of the three
    shown, which would read 100% across them and say nothing."""
    _signed_out(monkeypatch)

    body = client.get("/api/plan/preview?teams=4&slot=2").json()

    assert body["opening_observed"] == 4
    assert body["opening"][0]["path"] == ["RB", "WR", "WR", "RB", "TE"]
    assert body["opening"][0]["count"] == 3
    assert body["opening"][0]["share"] == 0.75
    assert body["opening"][1]["path"] == ["WR", "WR", "RB", "RB", "TE"]


def test_the_shape_the_numbers_came_from_is_stated(client, corpus, monkeypatch):
    """The corpus is one league size and the caller may be asking about
    another. Saying which drafts were counted is what keeps that honest."""
    _signed_out(monkeypatch)

    body = client.get("/api/plan/preview?teams=12&slot=2").json()

    assert body["corpus"] == {"teams": 4, "rounds": 5, "drafts": 4}
    # Still answered from the four-team archive: the opening shape of a seat
    # is the most transferable thing here, and an empty card would be worse.
    assert body["opening"][0]["path"] == ["RB", "WR", "WR", "RB", "TE"]


def test_position_runs_say_when_the_wait_ends(client, corpus, monkeypatch):
    """Every recorded draft takes its first quarterback at pick 9, so that is
    the median. The positions nobody took come back as nulls rather than
    missing keys -- a page prints a dash for a null and crashes on a hole."""
    _signed_out(monkeypatch)

    body = client.get("/api/plan/preview?teams=4&slot=2").json()

    assert body["position_runs"]["QB"] == 9.0
    assert body["position_runs"]["TE"] == 18.0
    assert body["position_runs"]["K"] is None
    assert body["position_runs"]["DST"] is None


def test_a_seat_the_archive_has_never_held_still_gets_a_plan(client, corpus,
                                                             monkeypatch):
    """Slot 11 of a four-team archive. The corpus cards go empty; the plan,
    which is the board's answer and not the corpus's, does not."""
    _signed_out(monkeypatch)

    body = client.get("/api/plan/preview?teams=12&slot=11").json()

    assert body["opening"] == []
    assert body["corpus"]["teams"] == 4
    assert len(body["targets"]) == 3


def test_no_archive_at_all_is_not_a_failure(client, tmp_path, monkeypatch):
    """A fresh install has recorded nothing. The plan is built from the board,
    so it still has something true to say."""
    _signed_out(monkeypatch)
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(tmp_path / "gone.duckdb"))
    market._CACHE.clear()

    res = client.get("/api/plan/preview?teams=10&slot=1")

    assert res.status_code == 200
    body = res.json()
    assert body["opening"] == [] and body["corpus"] is None
    assert len(body["targets"]) == 3


def test_a_corpus_that_breaks_mid_read_costs_the_cards_and_not_the_plan(
        client, corpus, monkeypatch):
    """The four counted reads run against a file the farm rewrites under us.

    A DuckDB error there -- a table this build has not got, a column renamed,
    the file swapped between the open and the query -- used to escape as a 500
    on the whole endpoint, which is the dashboard's plan card and the landing
    page's hero at once. The opening cards are allowed to be missing; the plan
    is built from the board and is not.
    """
    _signed_out(monkeypatch)
    monkeypatch.setattr(market, "_human_picks_sql", lambda: "nonesuch = 1")

    res = client.get("/api/plan/preview?teams=6&slot=3")

    assert res.status_code == 200
    body = res.json()
    assert body["opening"] == [] and body["corpus"] is None
    assert body["position_runs"] == {}
    assert len(body["targets"]) == 3


# -- favourites ---------------------------------------------------------------


def _named(body) -> list:
    """Every player this plan put on a card, in order: each turn's target
    followed by its two alternates."""
    rows = []
    for turn in body["targets"]:
        rows.extend([turn["target"], *turn["alternates"]])
    return [row["player_id"] for row in rows if row]


def _stars(body) -> list:
    """Five players to star: the ones this plan already named SECOND at each
    turn, padded from the bottom of the board.

    A favourite is worth eight ranks (`plan.FAVOURITE_BONUS`) -- a thumb on
    the scale, not a veto -- so the players worth starring in a test are the
    ones the plan already found arguable. Starring the fortieth-ranked man and
    expecting him in round one would be testing a product nobody built.
    """
    seconds = [turn["alternates"][0]["player_id"] for turn in body["targets"]
               if turn["alternates"]]
    return (seconds + [f"p{i}" for i in range(40, 0, -1)
                       if f"p{i}" not in seconds])[:5]


def test_a_saved_list_reaches_the_plan(client, monkeypatch):
    """THE WHOLE DIFFERENCE between this and a public ranking. Star the
    players a plan called close, and the cards it deals change."""
    _sign_in(monkeypatch)
    plain = client.get("/api/plan/preview?teams=10&slot=6").json()

    stars = _stars(plain)
    assert client.put("/api/account/favorites",
                      json={"players": stars}).status_code == 200
    starred = client.get("/api/plan/preview?teams=10&slot=6").json()

    assert _named(plain) != _named(starred)
    assert set(_named(starred)) & set(stars)


def test_a_starred_player_is_marked_as_one(client, monkeypatch):
    """The card says whose idea he was. Without the flag the reader cannot
    tell a pick the tool made from one it made because they asked."""
    _sign_in(monkeypatch)
    stars = _stars(client.get("/api/plan/preview?teams=10&slot=6").json())
    client.put("/api/account/favorites", json={"players": stars})

    body = client.get("/api/plan/preview?teams=10&slot=6").json()

    rows = [row for turn in body["targets"]
            for row in [turn["target"], *turn["alternates"]] if row]
    for row in rows:
        assert row["favourite"] == (row["player_id"] in stars), row["player_id"]
    assert any(row["favourite"] for row in rows)


def test_two_readers_with_different_lists_get_different_plans(client,
                                                              monkeypatch):
    """The favourites are IN the cache key, not merely an input to it. Without
    that, the first reader's plan is served to the second for a minute --
    which is both wrong and a small leak of what the first reader starred."""
    _sign_in(monkeypatch, ids=("reader-b",))
    plain = _named(client.get("/api/plan/preview?teams=10&slot=7").json())

    _sign_in(monkeypatch, ids=("reader-a",))
    stars = _stars(client.get("/api/plan/preview?teams=10&slot=7").json())
    client.put("/api/account/favorites", json={"players": stars})
    starred = _named(client.get("/api/plan/preview?teams=10&slot=7").json())

    # Reader B asks again, inside the minute reader A's answer is cached under
    # the same seat. They must not be handed it.
    _sign_in(monkeypatch, ids=("reader-b",))
    again = _named(client.get("/api/plan/preview?teams=10&slot=7").json())

    assert starred != plain
    assert again == plain


# -- the cache ----------------------------------------------------------------


def test_a_warm_hit_does_not_touch_the_board(client, monkeypatch):
    """The board is the expensive part of this request -- a frame copy on a
    hit, seconds cold -- and this endpoint is called from a page with two
    selects on it. The second identical request must do no work at all beyond
    reading the memoized board identity.

    Patched at `scoring.board_cache.cached_build_board`, which is where the
    route imports it from at call time.
    """
    _signed_out(monkeypatch)
    import scoring.board_cache as bc

    calls = []
    real = bc.cached_build_board
    monkeypatch.setattr(bc, "cached_build_board",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    # A seat nothing else in this file asks for, so the cache starts cold.
    client.get("/api/plan/preview?teams=14&slot=9")
    after_first = len(calls)
    client.get("/api/plan/preview?teams=14&slot=9")

    assert after_first >= 1
    assert len(calls) == after_first


def test_a_different_seat_is_a_different_answer(client, monkeypatch):
    """The seat is in the key. It has to be: two seats of the same league own
    different picks, and serving one seat's plan to the other is the one
    mistake nobody would notice until draft night."""
    _signed_out(monkeypatch)

    first = client.get("/api/plan/preview?teams=10&slot=1").json()
    second = client.get("/api/plan/preview?teams=10&slot=10").json()

    assert first["picks"] != second["picks"]
    assert first["targets"][0]["pick_no"] == 1
    assert second["targets"][0]["pick_no"] == 10
