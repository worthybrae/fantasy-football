# tests/test_seo.py
"""The pages a search engine reads: ADP from the corpus, rendered as HTML."""
import html
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
    default."""
    monkeypatch.setattr(seo, "WARM_ON_REGISTER", False)


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
        for pid, pick_no in picks:
            if pid == "ghost":
                position = "TE"
            elif pid in ("star", "mid"):
                position = "RB"
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
                 " headshot VARCHAR)")
    conn.executemany("INSERT INTO players VALUES (?, ?, ?)", [
        ("star", "D'Andre Swift", "https://img.test/swift.png"),
        ("mid", "Amon-Ra St. Brown", None),
        ("late", "Late Guy", None),
        ("once", "Once Guy", None),
        ("twin_a", "Josh Allen", None),
        ("twin_b", "Josh Allen", None),
        ("ghost", None, None),
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
    assert star["round_mode"] == 1 and star["rank"] == 1 and star["pos_rank"] == 1
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
    # These pages carry no application JavaScript: nothing to hydrate, nothing
    # the content waits on. What is allowed is structured data, which is not
    # code, and the one analytics tag, which appends its own script node and
    # blocks nothing. Anything else appearing here is a regression.
    assert body.count("<script") == 3, "an unexpected script reached the page"
    assert body.count('<script type="application/ld+json">') == 2


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
    board.execute("INSERT INTO players VALUES ('fb', 'Fulton Reese', NULL)")
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
