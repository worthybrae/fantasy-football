# tests/test_seo.py
"""The pages a search engine reads: ADP from the corpus, rendered as HTML."""
import html
import threading
import time
import xml.etree.ElementTree as ET

import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import market, seo
from pipeline import draft_log as dl


@pytest.fixture(autouse=True)
def _no_warm_thread(monkeypatch):
    """Every `_client(board)` call below registers the routes fresh, and
    with `WARM_ON_REGISTER` on that means a daemon thread opening the real
    corpus (or racing `market._CACHE.clear()` in the next test's fixture
    setup) once per test, for a whole suite that runs in a couple of
    seconds. Off for the duration of this file; production keeps the
    default.

    The remembered board goes with it. `seo._board_facts` holds the last
    board it managed to build, so that a page never disappears when one
    build fails (see its docstring), and that memory is module state:
    without this, the fake board one test below hands it would still be
    there for the next test whose board is supposed to fail.
    """
    monkeypatch.setattr(seo, "WARM_ON_REGISTER", False)
    monkeypatch.setattr(seo, "_last_board_facts", {})


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Ten drafts, 4 teams x 2 rounds (8 picks each), so shares are round
    numbers. `star` goes first in every draft; `mid` goes 2nd, 3rd or 4th
    (picks 2,3,4,2,3,4,2,3,4,2 -- mean 2.9); `late` is taken in two drafts
    (20%); `once` is taken in the last draft only (10%, still above
    MIN_SHARE's 2%). `twin_a` and `twin_b` share a display name to exercise
    slug collisions. `ghost` is taken in every draft (TE) but never gets a
    name in the `board` fixture -- an unresolved name, to prove such a
    player gets no page."""
    path = tmp_path / "corpus.duckdb"
    conn = dl.corpus_conn(str(path))
    for i in range(10):
        conn.execute(
            "INSERT INTO draft_log (draft_id, source, league_id, season,"
            " recorded_at, teams, rounds, my_slot, scoring_json, settings_json,"
            " human_seats) VALUES (?, 'mock', '1', 2026, ?::TIMESTAMP, 4, 2,"
            " NULL, NULL, NULL, 3)", [f"d{i}", f"2026-08-24 {i:02d}:00:00"])
        picks = [("star", 1), ("mid", 2 + (i % 3)), ("twin_a", 5), ("twin_b", 6),
                 ("ghost", 8)]
        if i < 2:
            picks.append(("late", 7))
        if i == 9:
            picks.append(("once", 7))
        # A SYNTHETIC ID, which is what every defense and every rookie is on
        # this board: no `players` row anywhere, so the only thing that can
        # name him is the board itself. Forty of the two hundred published
        # players are these.
        if 2 <= i <= 8:
            picks.append(("adp_seattle_defense", 7))
        for pid, pick_no in picks:
            if pid == "ghost":
                position = "TE"
            elif pid in ("star", "mid"):
                position = "RB"
            elif pid.startswith("adp_"):
                position = "DST"
            else:
                position = "WR"
            conn.execute(
                "INSERT INTO draft_log_pick (draft_id, pick_no, round, slot,"
                " owner_key, is_anonymous, player_id, position, adp_rank,"
                " proj_points, autodrafted, had_owner, seconds_to_pick,"
                " clock_seconds) VALUES (?, ?, ?, ?, 'o', FALSE, ?, ?, 1.0,"
                " 100.0, FALSE, TRUE, 5.0, 30.0)",
                [f"d{i}", pick_no, (pick_no - 1) // 4 + 1, (pick_no - 1) % 4 + 1,
                 pid, position])
        # THE POOL, which is what `of` and the availability strip count
        # against. Everybody is on the board in every draft here, so the
        # honest denominator and the draft count agree and every share above
        # stays what it was before the pool existed.
        for pid in ("star", "mid", "twin_a", "twin_b", "ghost", "late", "once",
                    "adp_seattle_defense"):
            conn.execute(
                "INSERT INTO draft_log_pool (draft_id, player_id, position,"
                " team, adp_rank, proj_points, espn_rank, espn_proj, bye)"
                " VALUES (?, ?, 'RB', 'CHI', 1.0, 100.0, 1.0, 100.0, 9)",
                [f"d{i}", pid])
    conn.close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    # AND THE RENDERED PAGES. `seo.rendered` keys on `_stamp`, whose
    # universal half is `board_cache._identity_key` -- and the `board`
    # fixture below is an in-memory database with no `meta` rows, so every
    # test in this file that seeds the same ten drafts computes the SAME
    # stamp. Without this, one test's HTML answers the next test's request.
    # A deployment does not have this problem: its `meta` is stamped.
    seo.clear_pages()
    yield path
    market._CACHE.clear()
    seo.clear_pages()


@pytest.fixture
def board():
    """The universal tables the pages read names, teams and ESPN's ADP from.

    `ghost` (a player in the `corpus` fixture) has a row with no name: found,
    but unresolved, which is the case under test. A row is deliberate here
    rather than no row at all -- an id `_names` has never heard of falls
    through to `_board_names`, which builds a real board from this
    connection's (deliberately incomplete) tables, and that is a much
    heavier, unrelated code path with warnings of its own to earn on every
    test that shares this fixture."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE players (gsis_id VARCHAR, display_name VARCHAR,"
                 " headshot VARCHAR, birth_date VARCHAR, rookie_season INTEGER,"
                 " height DOUBLE, weight DOUBLE)")
    conn.executemany("INSERT INTO players VALUES (?, ?, ?, ?, ?, ?, ?)", [
        ("star", "D'Andre Swift", "https://img.test/swift.png",
         "1999-01-14", 2020, 70.0, 215.0),
        ("mid", "Amon-Ra St. Brown", None, "1999-10-24", 2021, 72.0, 197.0),
        # `late` deliberately has no bio, no history, no projection and no
        # ESPN id anywhere below: he is the player every crosswalk misses.
        ("late", "Late Guy", None, None, None, None, None),
        ("once", "Once Guy", None, None, None, None, None),
        ("twin_a", "Josh Allen", None, None, None, None, None),
        ("twin_b", "Josh Allen", None, None, None, None, None),
        ("ghost", None, None, None, None, None, None),
    ])
    conn.execute("CREATE TABLE historic_adp (season BIGINT, adp_name VARCHAR,"
                 " position VARCHAR, team VARCHAR, adp_rank BIGINT)")
    conn.executemany("INSERT INTO historic_adp VALUES (?, ?, ?, ?, ?)", [
        (2024, "D'Andre Swift", "RB", "CHI", 41), (2025, "D'Andre Swift", "RB", "CHI", 33),
        # A trap: same name as the corpus's `late`, different position. He is
        # a WR there, so this row is somebody else and must not reach him.
        (2025, "Late Guy", "TE", "NYJ", 120),
    ])
    conn.execute("CREATE TABLE historic_espn_cs (season BIGINT, cs_rank BIGINT,"
                 " position VARCHAR, cs_name VARCHAR, team VARCHAR, auction_value DOUBLE)")
    conn.executemany("INSERT INTO historic_espn_cs VALUES (?, ?, ?, ?, ?, ?)", [
        (2024, 44, "RB", "D'Andre Swift", "CHI", 12.0),
        (2026, 29, "RB", "D'Andre Swift", "CHI", 17.0),
    ])
    conn.execute("CREATE TABLE espn_projections (season BIGINT, espn_id BIGINT,"
                 " espn_name VARCHAR, position VARCHAR, proj_points DOUBLE,"
                 " proj_games DOUBLE, proj_carries DOUBLE, proj_rush_yards DOUBLE,"
                 " proj_receptions DOUBLE, proj_rec_yards DOUBLE)")
    conn.executemany("INSERT INTO espn_projections VALUES (?, ?, ?, ?, ?, ?,"
                     " ?, ?, ?, ?)", [
        (2026, 4259545, "D'Andre Swift", "RB", 212.5, 17, 233, 998, 41, 310),
        # The same trap on the other key: `once` is a WR in the corpus.
        (2026, 4242424, "Once Guy", "RB", 88.8, 17, 100, 400, 10, 60),
    ])
    conn.execute("CREATE TABLE player_futures (season BIGINT, market VARCHAR,"
                 " espn_id BIGINT, american VARCHAR, implied_pct DOUBLE,"
                 " provider VARCHAR)")
    conn.execute("INSERT INTO player_futures VALUES (2026, 'rush_yards',"
                 " 4259545, '+1400', 6.7, 'DraftKings')")
    conn.execute("CREATE TABLE weekly (player_id VARCHAR, season INTEGER,"
                 " week INTEGER, season_type VARCHAR, opponent_team VARCHAR,"
                 " fantasy_points_ppr DOUBLE)")
    conn.executemany("INSERT INTO weekly VALUES (?, ?, ?, 'REG', ?, ?)", [
        ("star", 2025, 1, "GB", 11.4), ("star", 2025, 2, "DET", 28.9),
        ("star", 2025, 3, "MIN", 6.2),
    ])
    conn.execute("CREATE TABLE player_news (player_id VARCHAR, name VARCHAR,"
                 " headline VARCHAR, url VARCHAR, published_at TIMESTAMP,"
                 " source VARCHAR, attribution VARCHAR)")
    conn.execute("INSERT INTO player_news VALUES ('star', 'D''Andre Swift',"
                 " 'Swift takes every first-team rep', 'https://news.test/swift',"
                 " '2026-08-23 09:00:00'::TIMESTAMP, 'Wire Service', NULL)")
    conn.execute("CREATE TABLE player_status (player_id VARCHAR, name VARCHAR,"
                 " position VARCHAR, team VARCHAR, sleeper_id VARCHAR,"
                 " injury_status VARCHAR, injury_body_part VARCHAR,"
                 " injury_notes VARCHAR, practice_participation INTEGER)")
    conn.executemany("INSERT INTO player_status VALUES (?, ?, 'RB', 'CHI', NULL,"
                     " ?, ?, NULL, ?)", [
        ("star", "D'Andre Swift", "Questionable", "hamstring", 1),
        ("mid", "Amon-Ra St. Brown", "None", None, None),
    ])
    conn.execute("CREATE TABLE depth_charts (dt DATE, gsis_id VARCHAR, team VARCHAR)")
    conn.executemany("INSERT INTO depth_charts VALUES (?, ?, ?)", [
        ("2026-08-01", "star", "PHI"), ("2026-08-20", "star", "CHI"),
        ("2026-08-20", "mid", "DET"),
    ])
    conn.execute("CREATE TABLE sleeper_ids (gsis_id VARCHAR, espn_id BIGINT)")
    conn.execute("INSERT INTO sleeper_ids VALUES ('star', 4259545)")
    conn.execute("CREATE TABLE espn_adp (espn_id BIGINT, espn_name VARCHAR,"
                 " position VARCHAR, team VARCHAR, espn_adp DOUBLE)")
    conn.executemany("INSERT INTO espn_adp VALUES (?, ?, ?, ?, ?)", [
        (4259545, "D'Andre Swift", "RB", "CHI", 61.0),
        (999, "Amon-Ra St. Brown", "RB", "DET", 5.5),   # RB in the fixture corpus
    ])
    yield conn
    conn.close()


# -- slugs -------------------------------------------------------------------

def test_slugs_are_ascii_lowercase_and_stable():
    assert seo.slug("D'Andre Swift") == "dandre-swift"
    assert seo.slug("Amon-Ra St. Brown") == "amon-ra-st-brown"
    assert seo.slug("Seattle Seahawks D/ST") == "seattle-seahawks-dst"
    assert seo.slug("Ka'imi Fairbairn") == "kaimi-fairbairn"
    assert seo.slug("  ") == "player"


# -- the aggregation ---------------------------------------------------------

def test_build_adp_places_every_player_by_average_pick(corpus, board):
    data = seo.build_adp(board)
    assert data["drafts"] == 10 and data["teams"] == 4 and data["rounds"] == 2
    assert str(data["updated"]) == "2026-08-24"
    ranked = [p["player_id"] for p in data["players"]]
    assert ranked[:2] == ["star", "mid"]
    assert "ghost" not in ranked   # a row in `players`, but no resolved name
    star = data["by_slug"]["dandre-swift"]
    assert star["adp"] == 1.0 and star["taken"] == 10 and star["share"] == 1.0
    assert star["usual_round"] == 1 and star["rank"] == 1 and star["pos_rank"] == 1
    assert star["hist"][0] == 10 and len(star["hist"]) == 8
    mid = data["by_slug"]["amon-ra-st-brown"]
    assert mid["adp"] == 2.9 and mid["p10"] == 2 and mid["p90"] == 4
    assert mid["low"] == 2 and mid["high"] == 4


def test_build_adp_resolves_name_team_and_espn_adp(corpus, board):
    star = seo.build_adp(board)["by_slug"]["dandre-swift"]
    assert star["name"] == "D'Andre Swift"
    assert star["headshot"] == "https://img.test/swift.png"
    assert star["team"] == "CHI"            # latest depth chart, not the older one
    assert star["espn_adp"] == 61.0         # through the sleeper crosswalk
    mid = seo.build_adp(board)["by_slug"]["amon-ra-st-brown"]
    assert mid["espn_adp"] == 5.5           # by name, no crosswalk row


def test_build_adp_drops_a_headshot_that_is_not_a_plain_https_url(corpus, board):
    board.execute("UPDATE players SET headshot = 'javascript:alert(1)' WHERE gsis_id = 'late'")
    data = seo.build_adp(board)
    late = data["by_slug"]["late-guy"]
    assert late["headshot"] is None
    body = _client(board).get(f"/adp/{late['slug']}").text
    assert "javascript:" not in body
    assert "<img" not in body


def test_build_adp_counts_autodrafted_and_the_farms_own_seat(tmp_path, monkeypatch, board):
    """ADP is where a player GOES, not what a human chose -- unlike
    `market._human_picks_sql()`, which excludes autodrafted picks and picks
    made by the farm's own seat (`slot == my_slot`) for its behavioural
    questions. A separate tmp corpus, so the shared fixture's numbers above
    are untouched."""
    path = tmp_path / "every_pick_counts.duckdb"
    conn = dl.corpus_conn(str(path))
    for i in range(5):
        conn.execute(
            "INSERT INTO draft_log (draft_id, source, league_id, season,"
            " recorded_at, teams, rounds, my_slot, scoring_json, settings_json,"
            " human_seats) VALUES (?, 'mock', '1', 2026, ?::TIMESTAMP, 4, 2,"
            " 2, NULL, NULL, 3)", [f"a{i}", f"2026-08-24 {i:02d}:00:00"])
        # `star`: autodrafted. `mid`: taken by slot 2, which is `my_slot` --
        # the farm's own seat -- but not itself autodrafted.
        conn.execute(
            "INSERT INTO draft_log_pick (draft_id, pick_no, round, slot,"
            " owner_key, is_anonymous, player_id, position, adp_rank,"
            " proj_points, autodrafted, had_owner, seconds_to_pick,"
            " clock_seconds) VALUES (?, 1, 1, 1, 'o', FALSE, 'star', 'RB', 1.0,"
            " 100.0, TRUE, TRUE, 5.0, 30.0)", [f"a{i}"])
        conn.execute(
            "INSERT INTO draft_log_pick (draft_id, pick_no, round, slot,"
            " owner_key, is_anonymous, player_id, position, adp_rank,"
            " proj_points, autodrafted, had_owner, seconds_to_pick,"
            " clock_seconds) VALUES (?, 2, 1, 2, 'o', FALSE, 'mid', 'RB', 1.0,"
            " 100.0, FALSE, TRUE, 5.0, 30.0)", [f"a{i}"])
    conn.close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    data = seo.build_adp(board)
    taken = {p["player_id"]: p["taken"] for p in data["players"]}
    assert taken["star"] == 5      # autodrafted, still counted
    assert taken["mid"] == 5       # the farm's own seat, still counted


def test_build_adp_gives_colliding_names_distinct_slugs(corpus, board):
    data = seo.build_adp(board)
    assert data["by_slug"]["josh-allen"]["player_id"] == "twin_a"
    assert data["by_slug"]["josh-allen-2"]["player_id"] == "twin_b"


def test_build_adp_drops_a_player_below_the_share_floor(corpus, board, monkeypatch):
    monkeypatch.setattr(seo, "MIN_SHARE", 0.15)
    data = seo.build_adp(board)
    ids = {p["player_id"] for p in data["players"]}
    assert "once" not in ids and "late" in ids     # 10% out, 20% in


def test_build_adp_with_no_drafts_is_empty_not_an_error(tmp_path, monkeypatch, board):
    path = tmp_path / "empty.duckdb"
    dl.corpus_conn(str(path)).close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    data = seo.build_adp(board)
    assert data["players"] == [] and data["drafts"] == 0 and data["updated"] is None


def test_build_adp_skips_a_player_whose_name_does_not_resolve(corpus):
    """With no board connection nothing resolves a name, so nobody is
    publishable -- not even under a raw id like `star`. The corpus is still
    read: `drafts` reflects it, there just isn't a page for anyone."""
    data = seo.build_adp(None)
    assert data["players"] == []
    assert data["by_slug"] == {}
    assert data["drafts"] == 10


# -- the pages ---------------------------------------------------------------

def _client(board):
    app = FastAPI()
    seo.register_seo_routes(app, conn=board)
    return TestClient(app)


def test_the_index_lists_every_player_with_provenance(corpus, board):
    r = _client(board).get("/adp")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "<title>ESPN Mock Draft ADP 2026 (4-team PPR)" in body
    assert "10 real ESPN mock drafts" in body and "Aug 24, 2026" in body
    assert 'href="/adp/dandre-swift"' in body
    assert "Amon-Ra St. Brown" in html.unescape(body)
    assert '<link rel="canonical" href="https://espnfantasydraft.com/adp"' in body
    # Nothing on these pages is hydrated and nothing the content says waits
    # on a script. Three are allowed here and no fourth: two blocks of
    # structured data, which is not code; the analytics tag, which appends
    # its own node and blocks nothing; and the table sorter, which is
    # progressive enhancement over a table that is already in draft order.
    assert body.count("<script") == 4, "an unexpected script reached the page"
    assert body.count('<script type="application/ld+json">') == 2
    assert body.count("table.sortable") == 1, "the sort script is in twice"


def test_a_player_page_states_his_figures(corpus, board):
    body = _client(board).get("/adp/amon-ra-st-brown").text
    assert "<title>Amon-Ra St. Brown ADP" in body
    assert "2.9" in body                       # the ADP
    assert "picks 2" in body and "4" in body   # the range
    assert "round 1" in body.lower()
    assert "5.5" in body                       # ESPN's own ADP beside it
    assert '"@type": "BreadcrumbList"' in body
    assert "<svg" in body                       # the distribution, drawn
    assert 'href="/adp/dandre-swift"' in body   # a neighbour by ADP


def test_a_player_page_is_the_same_for_a_slug_collision(corpus, board):
    c = _client(board)
    assert "twin_a" not in c.get("/adp/josh-allen").text
    assert c.get("/adp/josh-allen-2").status_code == 200


def test_round_and_position_pages_filter(corpus, board):
    c = _client(board)
    r1 = html.unescape(c.get("/adp/round/1").text)
    assert "D'Andre Swift" in r1 and "Once Guy" not in r1
    assert 'href="/adp/round/2"' in r1
    rb = html.unescape(c.get("/adp/rb").text)
    assert "D'Andre Swift" in rb and "Josh Allen" not in rb
    assert "<title>RB ADP" in rb


def test_an_empty_position_page_says_so_and_is_not_linked(corpus, board):
    """K goes undrafted in the fixture. Its page exists (the dispatch
    whitelist is POSITIONS) but says so in its own words rather than
    borrowing the empty-corpus line, and the index does not link it."""
    c = _client(board)
    k = c.get("/adp/k")
    assert k.status_code == 200
    assert "No K has gone in these drafts often enough" in k.text
    assert "No drafts recorded yet" not in k.text
    index = c.get("/adp").text
    assert 'href="/adp/rb"' in index and 'href="/adp/k"' not in index


def test_unknown_pages_are_404_and_noindex(corpus, board):
    c = _client(board)
    for path in ("/adp/nobody-here", "/adp/round/17", "/adp/round/0", "/adp/ol"):
        r = c.get(path)
        assert r.status_code == 404, path
        assert '<meta name="robots" content="noindex">' in r.text


def test_an_empty_corpus_still_serves_the_index(tmp_path, monkeypatch, board):
    path = tmp_path / "empty.duckdb"
    dl.corpus_conn(str(path)).close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    c = _client(board)
    r = c.get("/adp")
    assert r.status_code == 200 and "No drafts recorded yet" in r.text
    assert c.get("/adp/dandre-swift").status_code == 404


# -- the sitemap and the registration order ----------------------------------

def test_the_sitemap_lists_every_page_once(corpus, board):
    r = _client(board).get("/sitemap.xml")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/xml")
    root = ET.fromstring(r.text)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [u.find("s:loc", ns).text for u in root.findall("s:url", ns)]
    assert len(locs) == len(set(locs))
    assert "https://espnfantasydraft.com/" in locs
    assert "https://espnfantasydraft.com/adp" in locs
    assert "https://espnfantasydraft.com/adp/dandre-swift" in locs
    assert "https://espnfantasydraft.com/adp/josh-allen-2" in locs
    assert "https://espnfantasydraft.com/adp/round/2" in locs
    assert "https://espnfantasydraft.com/adp/round/3" not in locs   # two rounds
    assert "https://espnfantasydraft.com/adp/rb" in locs
    assert "https://espnfantasydraft.com/adp/k" not in locs   # no K in the fixture at all
    assert "https://espnfantasydraft.com/adp/te" not in locs  # `ghost` is TE, but never published
    mods = {u.find("s:lastmod", ns).text for u in root.findall("s:url", ns)}
    assert mods == {"2026-08-24"}


def test_an_empty_corpus_sitemap_has_only_the_static_pages(tmp_path, monkeypatch, board):
    path = tmp_path / "empty.duckdb"
    dl.corpus_conn(str(path)).close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    root = ET.fromstring(_client(board).get("/sitemap.xml").text)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [u.find("s:loc", ns).text for u in root.findall("s:url", ns)]
    assert locs == ["https://espnfantasydraft.com/", "https://espnfantasydraft.com/mocks",
                    "https://espnfantasydraft.com/adp"]


def test_an_unrecognized_position_does_not_500_adp_or_the_sitemap(tmp_path, monkeypatch, board):
    """A position outside the six in `POSITIONS` (or a NULL, read back as
    `""`) used to blow up `sorted(..., key=POSITIONS.index)` with a
    ValueError. `/adp` and `/sitemap.xml` must still answer, and the
    unrecognized position gets no position page -- but the player himself,
    resolved by name, still gets his own."""
    board.execute("INSERT INTO players (gsis_id, display_name)"
                  " VALUES ('fb', 'Fulton Reese')")
    path = tmp_path / "fb.duckdb"
    conn = dl.corpus_conn(str(path))
    for i in range(10):
        conn.execute(
            "INSERT INTO draft_log (draft_id, source, league_id, season,"
            " recorded_at, teams, rounds, my_slot, scoring_json, settings_json,"
            " human_seats) VALUES (?, 'mock', '1', 2026, ?::TIMESTAMP, 4, 2,"
            " NULL, NULL, NULL, 3)", [f"fb{i}", f"2026-08-24 {i:02d}:00:00"])
        conn.execute(
            "INSERT INTO draft_log_pick (draft_id, pick_no, round, slot,"
            " owner_key, is_anonymous, player_id, position, adp_rank,"
            " proj_points, autodrafted, had_owner, seconds_to_pick,"
            " clock_seconds) VALUES (?, 1, 1, 1, 'o', FALSE, 'fb', 'FB', 1.0,"
            " 100.0, FALSE, TRUE, 5.0, 30.0)", [f"fb{i}"])
    conn.close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    c = _client(board)
    assert c.get("/adp").status_code == 200
    r = c.get("/sitemap.xml")
    assert r.status_code == 200
    root = ET.fromstring(r.text)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [u.find("s:loc", ns).text for u in root.findall("s:url", ns)]
    assert "https://espnfantasydraft.com/adp/fb" not in locs
    fb_slug = seo.slug("Fulton Reese")
    assert f"https://espnfantasydraft.com/adp/{fb_slug}" in locs
    assert c.get(f"/adp/{fb_slug}").status_code == 200


def test_the_pages_are_reachable_with_the_spa_mounted(corpus, board, tmp_path):
    """`register_spa`'s fallback matches every path. These routes must be
    registered first, and the SPA must still answer everything else."""
    from api.static import register_spa
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    app = FastAPI()
    seo.register_seo_routes(app, conn=board)
    assert register_spa(app, dist)
    c = TestClient(app)
    assert "D'Andre Swift" in html.unescape(c.get("/adp").text)
    assert c.get("/sitemap.xml").headers["content-type"].startswith("application/xml")
    assert c.get("/mocks").text == "<html>spa</html>"


def test_every_page_answers_a_head_request(corpus, board):
    """FastAPI's `@app.get` registers GET alone, so a HEAD arrived as a 405
    with an empty body while the same URL served a page to GET. Crawlers
    mostly use GET, but link previewers and uptime checks use HEAD, and a
    405 there reads as a broken URL."""
    c = _client(board)
    for path in ("/adp", "/adp/dandre-swift", "/adp/round/1", "/adp/rb",
                 "/sitemap.xml"):
        r = c.head(path)
        assert r.status_code == 200, f"HEAD {path} -> {r.status_code}"


def test_a_head_request_to_an_unknown_page_is_still_404(corpus, board):
    assert _client(board).head("/adp/nobody").status_code == 404


def test_the_index_explains_itself_in_prose_and_an_faq(corpus, board):
    """The index is the page meant to rank for the ADP query, and a bare
    table gives a search engine nothing to read. The prose states the shape
    the figures come from; the FAQ carries `FAQPage` structured data."""
    body = _client(board).get("/adp").text
    assert "average draft position" in body.lower()
    assert '"@type": "FAQPage"' in body or '"@type":"FAQPage"' in body
    assert "<details" in body


def test_a_position_page_does_not_repeat_the_index_faq(corpus, board):
    """Six position pages carrying the same FAQ markup as the index is the
    duplicate-content pattern the whole exercise is trying to avoid."""
    body = _client(board).get("/adp/rb").text
    assert "FAQPage" not in body


def test_a_team_falls_back_to_espns_by_name_when_the_crosswalk_misses(corpus, board):
    """`late` has no depth-chart row and no `sleeper_ids` row -- a rookie in
    August, whose team the crosswalk cannot reach. ESPN's board carries him
    by name, and the ADP fallback beside this one already matches that way,
    so the team should follow the same two steps rather than print an
    em dash."""
    board.execute("INSERT INTO espn_adp VALUES (777, 'Late Guy', 'WR', 'NYJ', 88.0)")
    d = seo.build_adp(board)
    late = next(p for p in d["players"] if p["player_id"] == "late")
    assert late["team"] == "NYJ"


def test_a_depth_chart_team_still_beats_espns(corpus, board):
    """The depth charts are the fresher source; ESPN's board is the fallback,
    not an override. `star` is CHI on the newest depth-chart row and CHI on
    ESPN's board too, so give ESPN a different team to tell them apart."""
    board.execute("UPDATE espn_adp SET team = 'SEA' WHERE espn_id = 4259545")
    d = seo.build_adp(board)
    star = next(p for p in d["players"] if p["player_id"] == "star")
    assert star["team"] == "CHI"


def test_the_analytics_tag_is_on_the_server_rendered_pages(corpus, board):
    """The ADP pages are most of what a search engine sees of this site, and
    they are not the SPA -- a tag in `web/index.html` alone would leave every
    one of them uncounted."""
    c = _client(board)
    for path in ("/adp", "/adp/dandre-swift", "/adp/round/1", "/adp/rb"):
        body = c.get(path).text
        assert "G-YP3EKCMQJJ" in body, f"no analytics tag on {path}"


def test_analytics_does_not_fire_off_the_production_hostname(corpus, board):
    """A local dev server, a vite preview and this suite's own screenshot runs
    would otherwise report themselves as traffic."""
    body = _client(board).get("/adp").text
    assert "location.hostname === 'espnfantasydraft.com'" in body


# -- staying warm ------------------------------------------------------------

def test_the_second_render_of_a_page_is_the_first_ones_bytes(corpus, board):
    """`/adp` is a pure function of the ADP aggregate, so rendering it twice
    for one snapshot is work done twice for one answer. A crawler walking the
    232-URL sitemap is the caller that makes this worth holding."""
    c = _client(board)
    calls = []
    first = c.get("/adp").text

    real = seo.render
    seo.render = lambda *a, **k: calls.append(a[0]) or real(*a, **k)
    try:
        second = c.get("/adp").text
        # The other three page shapes too, since each keys itself.
        c.get("/adp/round/1")
        c.get("/adp/rb")
        c.get("/adp/dandre-swift")
        c.get("/sitemap.xml")
        # ...all of which were built by the first pass through them.
        c.get("/adp/round/1")
        c.get("/adp/rb")
        c.get("/adp/dandre-swift")
        c.get("/sitemap.xml")
    finally:
        seo.render = real

    assert second == first
    # Four templates, rendered once each: the repeats came from the cache,
    # and `/adp` itself was never rendered again at all.
    assert sorted(calls) == ["adp_index.html", "adp_player.html", "adp_round.html"]


def test_a_new_draft_in_the_corpus_retires_every_rendered_page(corpus, board):
    """The cache may never outlive what it describes. The stamp
    (`seo._stamp`) moves the moment the corpus does, and a stamp that does
    not match empties the whole set rather than ageing pages out one by one
    -- half a crawl on yesterday's snapshot beside half on today's would be
    a worse answer than either."""
    c = _client(board)
    before = c.get("/adp").text
    assert "10 real ESPN mock drafts" in before

    conn = dl.corpus_conn(str(corpus))
    conn.execute(
        "INSERT INTO draft_log (draft_id, source, league_id, season,"
        " recorded_at, teams, rounds, my_slot, scoring_json, settings_json,"
        " human_seats) VALUES ('d10', 'mock', '1', 2026,"
        " '2026-08-25 00:00:00'::TIMESTAMP, 4, 2, NULL, NULL, NULL, 3)")
    conn.execute(
        "INSERT INTO draft_log_pick (draft_id, pick_no, round, slot,"
        " owner_key, is_anonymous, player_id, position, adp_rank,"
        " proj_points, autodrafted, had_owner, seconds_to_pick,"
        " clock_seconds) VALUES ('d10', 1, 1, 1, 'o', FALSE, 'star', 'RB',"
        " 1.0, 100.0, FALSE, TRUE, 5.0, 30.0)")
    conn.close()

    # The aggregate is what the pages hang off, so retiring it is what a
    # deployment's warm thread does every few minutes.
    seo.rebuild_adp(board)
    after = c.get("/adp").text
    assert "11 real ESPN mock drafts" in after


def test_rebuild_adp_does_not_hand_back_the_answer_it_is_replacing(corpus, board):
    """THE BUG THE WARM LOOP WOULD OTHERWISE HAVE. `adp_data` serves the
    cached aggregate for `market.CACHE_SECONDS`, so a warm thread calling it
    inside that window would get the entry it is trying to renew, renew
    nothing, and leave the next reader after the lapse paying the 0.9-2.8s
    rebuild -- which was the measured 1.9s TTFB on `/adp`."""
    built = []
    real = seo.build_adp
    seo.build_adp = lambda conn: built.append(1) or real(conn)
    try:
        seo.adp_data(board)
        seo.adp_data(board)          # cached: no second build
        assert len(built) == 1
        seo.rebuild_adp(board)       # forced: a real rebuild
        assert len(built) == 2
        seo.adp_data(board)          # and the fresh entry is what is served
        assert len(built) == 2
    finally:
        seo.build_adp = real


def test_the_warm_interval_is_inside_the_life_of_what_it_warms():
    """A warm loop slower than the cache it is renewing renews nothing: the
    entry lapses between passes and a reader pays for the rebuild, which is
    exactly the state this replaced."""
    assert seo.WARM_SECONDS < market.CACHE_SECONDS


def test_a_reader_is_not_made_to_wait_for_the_warm_rebuild(corpus, board):
    """THE REGRESSION THE FIRST VERSION OF THE WARM LOOP SHIPPED WITH.

    Every /adp route asks `adp_data` for the aggregate before it can reach
    its own page cache. Hold the aggregate's lock across `build_adp` -- as
    `rebuild_adp` first did -- and the four-minutely background refresh
    becomes a stall: a reader arriving during it queues behind the whole
    1.7s build. That is a worse version of the 1.9s the warm thread exists
    to remove, and it happens two and a half times as often.

    So the build runs with nothing held and only the finished answer is
    swapped in (`market.store`). A reader racing a rebuild is served the
    previous answer immediately.
    """
    c = _client(board)
    c.get("/adp")                       # prime the aggregate and the page
    primed = seo.adp_data(board)

    started = threading.Event()
    took = []
    real = seo.build_adp

    def slow(conn):
        """A second of work, and no database: the point is the wait, and
        this thread must not be reading `board` beside the request that is
        about to."""
        started.set()
        time.sleep(1.0)
        return primed

    seo.build_adp = slow
    try:
        def warm():
            began = time.monotonic()
            seo.rebuild_adp(board)
            took.append(time.monotonic() - began)

        thread = threading.Thread(target=warm, daemon=True)
        thread.start()
        assert started.wait(5), "the rebuild never started"

        began = time.monotonic()
        res = c.get("/adp")
        waited = time.monotonic() - began
        thread.join(10)
    finally:
        seo.build_adp = real

    # The rebuild really was slow -- otherwise the reader's speed below
    # would be proving nothing at all.
    assert took and took[0] >= 1.0
    # ...and the reader went straight through it, with the answer that was
    # already in hand rather than a wait for the one being built.
    assert res.status_code == 200
    assert "10 real ESPN mock drafts" in res.text
    assert waited < 0.1, f"a reader waited {waited:.2f}s behind the rebuild"


# -- the board, and what happens when it will not build ----------------------

def _board_frame(name: str = "Seattle Defense"):
    """What `cached_build_board` hands back, cut down to the columns these
    pages read. One row, for the id `players` cannot name."""
    import pandas as pd
    return pd.DataFrame([{
        "player_id": "adp_seattle_defense", "name": name, "headshot": None,
        "team": "SEA", "bye": 9, "tier": 3, "market_rank": 88.0,
        "espn_ppr_rank": 140, "proj_points": 96.0, "rookie": False,
        "espn_id": None, "market_sources": {"fp_tier": 6, "espn": 140.0},
        "stats": None}])


def test_a_board_that_fails_twice_is_asked_a_third_time(corpus, board, monkeypatch):
    """THE COLD-START RACE. `cached_build_board` raises when another thread
    is part-way through building the same board on the same connection,
    which is what a keep-warm pass landing on a request looks like. Forty of
    the two hundred published players -- every D/ST, every rookie -- have no
    `players` row and no name without it, and a player who cannot be named
    gets no page: the sitemap went 228 to 187 and stayed there until the
    next warm pass."""
    from scoring import board_cache
    calls = []

    def flaky(conn, *args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("another thread is building this board")
        return _board_frame()

    monkeypatch.setattr(board_cache, "cached_build_board", flaky)
    d = seo.build_adp(board)
    assert len(calls) == 3, "the board was not retried"
    seattle = d["by_slug"]["seattle-defense"]
    assert seattle["player_id"] == "adp_seattle_defense"
    assert seattle["bye"] == 9 and seattle["tier"] == 3


def test_a_board_that_stops_answering_does_not_cost_a_page(corpus, board, monkeypatch):
    """And when the retries run out, the last board this process saw stands
    in. A stale tier beside a fresh ADP is a smaller lie than a page that is
    not there."""
    from scoring import board_cache
    calls = []

    def once_then_never(conn, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return _board_frame()
        raise RuntimeError("gone")

    monkeypatch.setattr(board_cache, "cached_build_board", once_then_never)
    first = seo.build_adp(board)
    assert "seattle-defense" in first["by_slug"]
    second = seo.build_adp(board)
    assert len(calls) == 4, "the second build did not retry three times"
    assert set(second["by_slug"]) == set(first["by_slug"])
    assert second["by_slug"]["seattle-defense"]["bye"] == 9


def test_a_board_that_never_builds_costs_the_board_fields_and_no_more(corpus, board):
    """The fixture board is a partial database that cannot build a board at
    all, and nothing has ever been remembered. The corpus's own figures --
    the headline of every one of these pages -- do not depend on it."""
    d = seo.build_adp(board)
    assert "seattle-defense" not in d["by_slug"]     # nothing can name him
    star = d["by_slug"]["dandre-swift"]
    assert star["adp"] == 1.0 and star["taken"] == 10
    assert star["bye"] is None and star["tier"] is None


def test_the_board_is_fetched_once_per_build(corpus, board, monkeypatch):
    """Names and facts come off ONE frame. `market._board_names` used to
    build a second one, which on a cold process is a second chance to lose
    the defenses."""
    from scoring import board_cache
    calls = []
    monkeypatch.setattr(board_cache, "cached_build_board",
                        lambda conn, *a, **k: calls.append(1) or _board_frame())
    seo.build_adp(board)
    assert len(calls) == 1


def test_the_facts_land_on_the_right_player(corpus, board):
    """Bio, season history, projection, futures, last season, news and the
    injury report, all onto `star` and none of them onto anybody else."""
    d = seo.build_adp(board)
    star = d["by_slug"]["dandre-swift"]
    assert star["age"] and star["rookie_season"] == 2020
    assert [s["season"] for s in star["seasons"]] == [2024, 2025, 2026]
    assert star["seasons"][0] == {"season": 2024, "adp": 41, "cheat": 44, "auction": 12.0}
    assert star["auction"] == 17.0 and star["cheat_rank"] == 29
    assert star["projection"]["points"] == 212.5
    assert {line["label"] for line in star["projection"]["lines"]} == {
        "Carries", "Rush yards", "Catches", "Rec yards"}
    assert star["futures"][0]["market"] == "Most rushing yards"
    assert star["last_season"] == {
        "season": 2025, "games": 3, "points": 46.5, "ppg": 15.5, "best": 28.9,
        "worst": 6.2, "best_week": 2, "best_opponent": "DET"}
    assert star["news"][0]["headline"] == "Swift takes every first-team rep"
    assert star["news"][0]["url"] == "https://news.test/swift"
    assert star["status"] == {"injury": "Questionable", "body_part": "hamstring",
                              "practice": 1.0}


def test_a_player_the_crosswalks_miss_gets_blanks_not_a_neighbours_numbers(corpus, board):
    """THE FAILURE MODE THIS WHOLE ENRICHMENT COULD HAVE. `late` has no
    crosswalk row, no history, no projection and no news. Every one of those
    fields must come back empty rather than picking up the player beside him,
    and his page must not print another man's figures."""
    d = seo.build_adp(board)
    late = d["by_slug"]["late-guy"]
    for field in ("age", "rookie_season", "projection", "status", "cheat_rank",
                  "auction", "espn_adp", "consensus", "bye"):
        assert late[field] is None, f"{field} came from somewhere"
    for field in ("seasons", "futures", "news", "sources"):
        assert late[field] == [], f"{field} came from somewhere"
    # ...including from the tight end of the same name in `historic_adp`, and
    # from the running back called Once Guy in `espn_projections`.
    assert seo.build_adp(board)["by_slug"]["once-guy"]["projection"] is None
    body = _client(board).get("/adp/late-guy").text
    assert "212.5" not in body            # star's projection
    assert "first-team rep" not in body   # star's news
    assert "Questionable" not in body     # star's injury designation


def test_the_player_page_says_whether_he_will_still_be_there(corpus, board):
    """`star` goes first in all ten drafts, so he is gone by pick 8 in every
    one of them -- a counted 0%, not a modelled one. The curve beside the
    chips is drawn from the same numbers."""
    d = seo.build_adp(board)
    star = d["by_slug"]["dandre-swift"]
    assert [c["pick"] for c in star["still_there"]] == list(seo.STILL_THERE_PICKS)
    assert star["still_there"][0] == {"pick": 8, "pct": 0, "round": 2}
    assert star["curve"].startswith("M0,")
    body = _client(board).get("/adp/dandre-swift").text
    assert "Still on the board at" in body
    assert 'aria-label="Share of drafts in which' in body


def test_the_player_page_carries_the_new_sections(corpus, board):
    body = html.unescape(_client(board).get("/adp/dandre-swift").text)
    for heading in ("Still on the board at", "Where the room takes him",
                    "Season by season", "What he did, and what he is projected for",
                    "Latest on D'Andre Swift", "Goes around the same pick"):
        assert heading in body, heading
    assert "212.5" in body                       # the projection
    assert "Swift takes every first-team rep" in body
    assert "Questionable" in body
    # The season history, as a row: 2024, market rank 41, cheat sheet 44.
    assert '<td class="mono">2024</td>' in body
    assert '<td class="n">41</td>' in body and '<td class="n">44</td>' in body


def test_the_player_pages_structured_data_parses_and_names_a_person(corpus, board):
    blocks = _ld_json(_client(board).get("/adp/dandre-swift").text)
    kinds = {block["@type"]: block for block in blocks}
    assert "BreadcrumbList" in kinds
    person = kinds["Person"]
    assert person["name"] == "D'Andre Swift"
    assert person["url"] == "https://espnfantasydraft.com/adp/dandre-swift"
    assert person["affiliation"]["name"] == "CHI"
    assert person["image"].startswith("https://")


def _ld_json(body: str) -> list:
    import json
    import re
    return [json.loads(m) for m in re.findall(
        r'<script type="application/ld\+json">(.*?)</script>', body, re.S)]


def test_the_index_leads_with_risers_fallers_and_the_runs(corpus, board, monkeypatch):
    """Eight picks apart is the production threshold and the fixture's two
    ESPN-ranked players are 1.9 apart, so the threshold is moved rather than
    the fixture: what is under test is that the sections are built and drawn,
    not the constant."""
    monkeypatch.setattr(seo, "MOVE_PICKS", 1)
    seo.clear_pages()
    market._CACHE.clear()
    body = html.unescape(_client(board).get("/adp").text)
    assert "Risers" in body and "Fallers" in body and "When the runs start" in body
    # `star` is taken at 1.0 and ESPN's board would reach him at 2.9, so he
    # rises two picks and `mid` falls two. Matched on the mover markup, not
    # on the bare digits -- "-2" is also the tail of the `josh-allen-2` slug.
    assert '<span class="d big">+2</span>' in body
    assert '<span class="d big">-2</span>' in body
    # The runs: the first RB goes at pick 1 and the first tight end at pick 8.
    assert "<b>pick 1</b>" in body or '<span class="nm">pick 1</span>' in body
    assert '<span class="nm">pick 8</span>' in body


def test_the_index_table_sorts_and_says_so_once(corpus, board):
    body = _client(board).get("/adp").text
    assert body.count("table.sortable") == 1
    assert body.count('class="sortable"') == 1
    assert 'data-k="n"' in body and 'aria-sort' in body
    # ...and nowhere else. A player page has no table to sort.
    assert "table.sortable" not in _client(board).get("/adp/dandre-swift").text


def test_a_position_page_ranks_within_the_position(corpus, board):
    body = html.unescape(_client(board).get("/adp/rb").text)
    assert "RB rank" in body
    assert "D'Andre Swift" in body and "Josh Allen" not in body


def test_the_round_page_shows_the_mix_and_what_each_seat_sees(corpus, board):
    body = html.unescape(_client(board).get("/adp/round/1").text)
    assert "What this round is made of" in body
    assert 'class="mix"' in body
    assert "What a seat sees" in body
    # Seat 1's first-round pick is pick 1, and `star` went there every time.
    assert "Seat 1 — pick 1" in body
    assert '<span class="d big">100%</span>' in body
    assert "Who goes in round 1" in body


def test_the_sitemap_holds_one_url_per_page_that_exists(corpus, board):
    d = seo.build_adp(board)
    root = ET.fromstring(_client(board).get("/sitemap.xml").text)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [u.find("s:loc", ns).text for u in root.findall("s:url", ns)]
    positions = {p["position"] for p in d["players"]}
    assert len(locs) == 3 + len(positions) + d["rounds"] + len(d["players"])
    assert len(locs) == len(set(locs))


def test_the_stylesheet_keeps_a_wide_table_inside_a_narrow_screen(corpus, board):
    """/adp measured 492 CSS pixels of content in a 390 pixel viewport before
    this, which simply cut off the last three columns."""
    body = _client(board).get("/adp").text
    assert "@media (max-width:720px)" in body
    assert ".scroll{overflow-x:auto" in body
    assert "@media (prefers-reduced-motion:reduce)" in body
    assert ":focus-visible{outline" in body
    assert "th{position:sticky" in body


def test_an_empty_corpus_still_has_every_key_the_pages_read(tmp_path, monkeypatch, board):
    """A fresh deployment renders the same templates against no drafts at
    all, and a missing key is a 500 rather than an empty section."""
    path = tmp_path / "empty.duckdb"
    dl.corpus_conn(str(path)).close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    d = seo.build_adp(board)
    for key in ("players", "by_slug", "drafts", "teams", "rounds", "updated",
                "stamp", "risers", "fallers", "runs", "round_mix", "at_pick",
                "seconds"):
        assert key in d, key
    c = _client(board)
    assert c.get("/adp").status_code == 200
    assert c.get("/adp/rb").status_code == 200
    assert c.get("/sitemap.xml").status_code == 200


def test_every_published_player_carries_every_optional_field(corpus, board):
    """`seo.BLANKS` is what a template may ask for without guarding. A field
    that reaches a template as Jinja's Undefined is a 500 the moment anything
    compares it."""
    for p in seo.build_adp(board)["players"]:
        for field in seo.BLANKS:
            assert field in p, f"{p['name']} has no {field}"


def test_both_season_sources_hand_the_page_the_same_keys():
    """THE BUG THIS CAUGHT. Two things fill in `last_season`: the week-by-week
    table, which can see the best and worst week, and the board's own summary,
    which cannot. The board path used to emit four keys of the eight, and the
    player page -- which asks for `worst` without guarding, as it is entitled
    to -- was a 500 for every player `weekly` had no row for."""
    from scoring import adp_facts

    class _Row:
        player_id, bye, tier, market_rank, espn_ppr_rank = "x", 9, 2, 12.0, 11
        proj_points, rookie, team, market_sources, espn_id = 210.0, False, "CHI", None, None
        stats = {"season": 2025, "games": 17, "ppg": 12.4, "points": 210.8}

    player = {"player_id": "x", "name": "Board Only", "position": "RB", "team": None}
    adp_facts._from_board(player, _Row())
    assert set(player["last_season"]) == set(adp_facts._EMPTY_SEASON)
    assert player["last_season"]["points"] == 210.8
    assert player["last_season"]["worst"] is None
