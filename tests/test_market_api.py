"""The draft archive: whose picks count, and what the numbers mean.

The corpus is the one thing this deployment has that nobody else does, so
these cover the two claims the page makes about it -- that behaviour is
counted from people rather than from ESPN's autodrafter, and that "still
there" is counted over the drafts a player was actually on the board for.
"""
import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.market as market


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A tiny archive: two 4-team, 2-round drafts."""
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
    for draft in ("d1", "d2"):
        conn.execute(
            "INSERT INTO draft_log VALUES (?, 'mock', '1', 2026,"
            " TIMESTAMP '2026-08-24 12:00:00', 4, 2, 4, NULL, NULL, 3)",
            [draft])
        for pid in ("star", "mid", "late", "never"):
            conn.execute("INSERT INTO draft_log_pool VALUES"
                         " (?, ?, 'RB', 'DET', 1.0, 100.0, 2.0, 90.0, 9)",
                         [draft, pid])
    # Seat 1 takes the star in both drafts, by hand.
    for draft in ("d1", "d2"):
        conn.execute("INSERT INTO draft_log_pick VALUES"
                     " (?, 1, 1, 1, 'o', FALSE, 'star', 'RB', 1, 1, FALSE,"
                     " TRUE, 5.0, 30.0)", [draft])
    # Seat 2's pick is the autodrafter in one draft and a person in the other.
    conn.execute("INSERT INTO draft_log_pick VALUES"
                 " ('d1', 2, 1, 2, 'o', FALSE, 'mid', 'RB', 2, 1, TRUE, TRUE,"
                 " 30.0, 30.0)")
    conn.execute("INSERT INTO draft_log_pick VALUES"
                 " ('d2', 2, 1, 2, 'o', FALSE, 'late', 'RB', 3, 1, FALSE,"
                 " TRUE, 6.0, 30.0)")
    # Seat 4 is the farm's own seat (my_slot = 4) -- excluded from behaviour.
    for draft in ("d1", "d2"):
        conn.execute("INSERT INTO draft_log_pick VALUES"
                     " (?, 4, 1, 4, 'o', FALSE, 'mid', 'RB', 2, 1, FALSE,"
                     " TRUE, 4.0, 30.0)", [draft])
    conn.close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    yield path
    market._CACHE.clear()


def _app():
    app = FastAPI()
    market.register_market_routes(app, conn=None)
    return app


@pytest.fixture
def client():
    return TestClient(_app())


def test_the_archive_is_open(corpus, client):
    """It is the data store: public mock drafts played by strangers, with no
    credential in it and no account attached to a seat. It was behind a login
    for one build, which locked out the reader it was built for."""
    assert client.get("/api/market/overview").status_code == 200


def test_overview_counts_people_separately(corpus, client):
    body = client.get("/api/market/overview").json()
    assert body["drafts"] == 2
    assert body["picks"] == 6
    # Six picks, of which one was ESPN's autodrafter and two were the farm's
    # own seat: three opinions.
    assert body["human_picks"] == 3
    assert body["teams"] == 4 and body["rounds"] == 2


def test_a_turn_counts_only_what_people_did(corpus, client):
    body = client.get("/api/market/slot/2").json()
    turn = next(t for t in body["turns"] if t["pick_no"] == 2)
    # The autodrafted pick is not an opinion, so seat 2's turn was observed
    # once, not twice -- and `late` owns it outright rather than tying `mid`.
    assert turn["observed"] == 1
    assert [p["player_id"] for p in turn["players"]] == ["late"]
    assert turn["top_share"] == 1.0


def test_the_farms_own_seat_is_not_the_market(corpus, client):
    """Seat 4 drafts with this tool. Counting it would be the project
    marking its own homework."""
    body = client.get("/api/market/slot/4").json()
    assert all(t["observed"] == 0 or t["players"] == [] for t in body["turns"])


def test_availability_counts_every_pick(corpus, client):
    """A player taken by a bot is gone just as thoroughly."""
    body = client.get("/api/market/players?slot=3").json()
    rows = {p["player_id"]: p for p in body["players"]}
    # `mid` was taken by the autodrafter in d1 and by the farm seat in both,
    # so he is off the board in both drafts by pick 4 -- despite no human ever
    # choosing him, which is why he appears in no turn's names.
    assert rows["mid"]["taken"] == 2
    # Seat 3 picks at 3 and 6. At pick 3 he is gone in the draft where the
    # autodrafter took him at pick 2 and still there in the other -- half.
    # By pick 6 the farm seat has taken him in both.
    assert body["turns"][:2] == [3, 6]
    assert rows["mid"]["survive"][0] == 0.5
    assert rows["mid"]["survive"][1] == 0.0


def test_a_player_nobody_took_survives_every_draft(corpus, client):
    body = client.get("/api/market/players?slot=1").json()
    rows = {p["player_id"]: p for p in body["players"]}
    assert rows["never"]["taken"] == 0
    assert rows["never"]["median"] is None
    assert all(share == 1.0 for share in rows["never"]["survive"])


def test_the_denominator_is_the_drafts_he_was_pooled_in(corpus, client):
    body = client.get("/api/market/players?slot=1").json()
    rows = {p["player_id"]: p for p in body["players"]}
    assert rows["star"]["drafts"] == 2
    assert rows["star"]["taken_share"] == 1.0
    assert rows["star"]["earliest"] == 1 and rows["star"]["latest"] == 1


def test_seats_outside_the_room_are_refused(corpus, client):
    assert client.get("/api/market/slot/9").status_code == 404


def test_the_snake_turns_are_this_seats_own_picks(corpus, client):
    body = client.get("/api/market/players?slot=2").json()
    # 4 teams: seat 2 picks 2nd, then 7th (the round comes back the other way).
    assert body["turns"][:2] == [2, 7]


# -- who goes at a given overall pick ----------------------------------------
#
# The waiting room asks this one: a seat is a list of pick numbers, and what
# makes one seat different from another is who is usually on the board at them.


def test_a_pick_names_who_usually_goes_there(corpus, client):
    """Pick 1 is the star in both recorded drafts; pick 2 is one person's
    choice and one autodraft, so it is counted once."""
    body = client.get("/api/market/at-picks?picks=1,2").json()
    by_pick = {p["pick_no"]: p for p in body["picks"]}

    assert by_pick[1]["observed"] == 2
    assert by_pick[1]["top"][0]["player_id"] == "star"
    assert by_pick[1]["top"][0]["count"] == 2
    assert by_pick[1]["top"][0]["share"] == 1.0

    # The autodrafted pick 2 is not somebody's opinion; the human one is.
    assert by_pick[2]["observed"] == 1
    assert by_pick[2]["top"][0]["player_id"] == "late"


def test_a_pick_nobody_ever_reached_says_so(corpus, client):
    """A 14-team room's later picks run past anything this archive has seen.
    The row still comes back -- with nothing in it, and an honest zero -- so
    the page can say "no record" rather than draw a name it invented."""
    body = client.get("/api/market/at-picks?picks=999").json()
    row = body["picks"][0]

    assert row["pick_no"] == 999
    assert row["observed"] == 0
    assert row["top"] == []


def test_the_answer_says_what_it_was_counted_from(corpus, client):
    """Every number here comes from 8-team mocks in the real corpus and from
    4-team ones in this fixture. A page that prints them next to a 14-team
    room has to be able to say so."""
    body = client.get("/api/market/at-picks?picks=1").json()

    assert body["drafts"] == 2
    assert body["teams"] == 4 and body["rounds"] == 2


def test_a_flood_of_picks_is_refused_rather_than_run(corpus, client):
    assert client.get("/api/market/at-picks?picks=" + ",".join(
        str(n) for n in range(1, 60))).status_code == 400
    assert client.get("/api/market/at-picks?picks=nonsense").status_code == 400


def test_an_adp_only_player_gets_his_real_name(corpus, monkeypatch):
    """`players` is the nflverse roster and does not carry a rookie with no
    NFL season yet, or any defense -- scoring/board synthesizes those under
    ids of its own (`adp_jadarian_price`). Looking one up in the roster alone
    printed the id at the reader, which is what this stops."""
    class _Row:
        player_id = "star"
        name = "Jadarian Price"
        headshot = None

    class _Board:
        def itertuples(self):
            return iter((_Row(),))

    class _RosterOnly:
        """A board database whose `players` table knows nobody in this draft."""
        def execute(self, *args, **kwargs):
            class _R:
                def fetchall(self_inner):
                    return []
            return _R()

    monkeypatch.setattr(market, "cached_build_board", None, raising=False)
    monkeypatch.setattr("scoring.board_cache.cached_build_board",
                        lambda conn, *a, **k: _Board())

    app = FastAPI()
    market.register_market_routes(app, conn=_RosterOnly())
    body = TestClient(app).get("/api/market/at-picks?picks=1").json()

    assert body["picks"][0]["top"][0]["name"] == "Jadarian Price"


def test_a_pick_carries_what_each_option_is_projected_for(corpus, monkeypatch):
    """A name alone does not price a seat. The waiting room shows the two or
    three players who actually go at a pick, and the only way to weigh one
    against another there is what each is projected to score -- so the row
    carries the board's `proj_points` beside the share."""
    class _Row:
        player_id = "star"
        name = "Star Back"
        headshot = None
        proj_points = 238.5

    class _Board:
        def itertuples(self):
            return iter((_Row(),))

    class _RosterOnly:
        def execute(self, *args, **kwargs):
            class _R:
                def fetchall(self_inner):
                    return []
            return _R()

    monkeypatch.setattr("scoring.board_cache.cached_build_board",
                        lambda conn, *a, **k: _Board())

    app = FastAPI()
    market.register_market_routes(app, conn=_RosterOnly())
    body = TestClient(app).get("/api/market/at-picks?picks=1").json()

    assert body["picks"][0]["top"][0]["proj_points"] == 238.5


def test_a_turn_carries_what_its_names_are_projected_for(corpus, monkeypatch):
    """Same reason as the pick above, one page over: the three names printed
    under a turn's bar are the only place the seat page says WHO, and a reader
    choosing between two of them is choosing on points. Seat 1 takes `star`
    in both drafts, so his turn is the one that must quote him."""
    class _Row:
        player_id = "star"
        name = "Star Back"
        headshot = "https://example.test/star.png"
        proj_points = 238.5

    class _Board:
        def itertuples(self):
            return iter((_Row(),))

    class _RosterOnly:
        def execute(self, *args, **kwargs):
            class _R:
                def fetchall(self_inner):
                    return []
            return _R()

    monkeypatch.setattr("scoring.board_cache.cached_build_board",
                        lambda conn, *a, **k: _Board())

    app = FastAPI()
    market.register_market_routes(app, conn=_RosterOnly())
    body = TestClient(app).get("/api/market/slot/1").json()
    turn = next(t for t in body["turns"] if t["pick_no"] == 1)

    assert turn["players"][0]["proj_points"] == 238.5
    assert turn["players"][0]["headshot"] == "https://example.test/star.png"


def test_a_page_the_board_missed_is_not_kept_for_ten_minutes(corpus, client):
    """The failure this exists for: the board is built on demand, and a
    request that lands while it is cold or unreadable gets a payload with no
    projections and with the ADP-only ids -- the rookies and the defenses --
    printed as `adp_carnell_tate` instead of a name. That answer is served,
    because the shares around it are the corpus's own facts, but it must not
    be the answer for the next ten minutes."""
    import time

    client.get("/api/market/slot/1")
    expiry = market._CACHE[("slot", 1)][0]

    assert expiry - time.monotonic() <= market.DEGRADED_SECONDS


def test_a_whole_answer_keeps_its_full_life(corpus, monkeypatch):
    """The other half: a payload the board DID answer for is cached for the
    full `CACHE_SECONDS`, so the short life above is a recovery path and not
    a tenfold increase in how often this page is built."""
    import time

    class _Row:
        player_id = "star"
        name = "Star Back"
        headshot = None
        proj_points = 238.5

    class _Board:
        def itertuples(self):
            return iter((_Row(),))

    class _RosterOnly:
        def execute(self, *args, **kwargs):
            class _R:
                def fetchall(self_inner):
                    return []
            return _R()

    monkeypatch.setattr("scoring.board_cache.cached_build_board",
                        lambda conn, *a, **k: _Board())
    app = FastAPI()
    market.register_market_routes(app, conn=_RosterOnly())
    TestClient(app).get("/api/market/slot/1")
    expiry = market._CACHE[("slot", 1)][0]

    assert expiry - time.monotonic() > market.DEGRADED_SECONDS


def test_a_seat_nobody_drafted_from_is_not_a_thin_answer():
    """The farm's own seat has no counted picks at all, so there is nothing
    for the board to have priced. Rebuilding that every twenty seconds
    forever would be this rule misreading an honest empty page."""
    assert market._board_answered([]) is True


def test_a_turns_projection_is_null_without_a_board(corpus, client):
    """A missing projection is a null beside a real share, never a zero: who
    went where is the corpus's own fact and does not need the board."""
    turn = next(t for t in client.get("/api/market/slot/1").json()["turns"]
                if t["pick_no"] == 1)

    assert turn["players"][0]["player_id"] == "star"
    assert turn["players"][0]["proj_points"] is None


def test_a_board_that_will_not_build_costs_the_projection_not_the_pick(
        corpus, client):
    """The archive answers without a board at all -- who went where is the
    corpus's own fact. A missing projection is a null, never a zero and never
    a 500: `client` is wired with no board database whatsoever."""
    row = client.get("/api/market/at-picks?picks=1").json()["picks"][0]

    assert row["top"][0]["player_id"] == "star"
    assert row["top"][0]["proj_points"] is None


def test_the_export_is_a_zip_of_the_whole_corpus(corpus, client):
    """One download carries the archive: every draft row, every pick, and the
    overview the page shows, as JSON files anybody can load without us."""
    import io
    import json
    import zipfile

    r = client.get("/api/market/export.zip")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/zip")
    assert "attachment" in r.headers.get("content-disposition", "")

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert {"overview.json", "drafts.json", "picks.json", "README.txt"} <= names

    drafts = json.loads(zf.read("drafts.json"))
    assert len(drafts) == 2
    assert drafts[0]["teams"] == 4 and drafts[0]["rounds"] == 2
    assert drafts[0]["picks_made"] == 3

    picks = json.loads(zf.read("picks.json"))
    assert len(picks) == 6
    star = [p for p in picks if p["player_id"] == "star"]
    assert len(star) == 2 and star[0]["pick_no"] == 1
    assert all("autodrafted" in p and "draft_id" in p for p in picks)

    overview = json.loads(zf.read("overview.json"))
    assert overview["drafts"] == 2
