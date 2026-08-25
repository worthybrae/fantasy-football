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


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Ten drafts, 4 teams x 2 rounds (8 picks each), so shares are round
    numbers. `star` goes first in every draft; `mid` goes 2nd, 3rd or 4th
    (picks 2,3,4,2,3,4,2,3,4,2 -- mean 2.9); `late` is taken in two drafts
    (20%); `once` in one (10%, still above MIN_SHARE's 2%). `twin_a` and
    `twin_b` share a display name to exercise slug collisions."""
    path = tmp_path / "corpus.duckdb"
    conn = dl.corpus_conn(str(path))
    for i in range(10):
        conn.execute(
            "INSERT INTO draft_log (draft_id, source, league_id, season,"
            " recorded_at, teams, rounds, my_slot, scoring_json, settings_json,"
            " human_seats) VALUES (?, 'mock', '1', 2026, ?::TIMESTAMP, 4, 2,"
            " NULL, NULL, NULL, 3)", [f"d{i}", f"2026-08-24 {i:02d}:00:00"])
        picks = [("star", 1), ("mid", 2 + (i % 3)), ("twin_a", 5), ("twin_b", 6)]
        if i < 2:
            picks.append(("late", 7))
        if i == 0:
            picks.append(("once", 8))
        for pid, pick_no in picks:
            conn.execute(
                "INSERT INTO draft_log_pick (draft_id, pick_no, round, slot,"
                " owner_key, is_anonymous, player_id, position, adp_rank,"
                " proj_points, autodrafted, had_owner, seconds_to_pick,"
                " clock_seconds) VALUES (?, ?, ?, ?, 'o', FALSE, ?, ?, 1.0,"
                " 100.0, FALSE, TRUE, 5.0, 30.0)",
                [f"d{i}", pick_no, (pick_no - 1) // 4 + 1, (pick_no - 1) % 4 + 1,
                 pid, "RB" if pid in ("star", "mid") else "WR"])
    conn.close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    yield path
    market._CACHE.clear()


@pytest.fixture
def board():
    """The universal tables the pages read names, teams and ESPN's ADP from."""
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


def test_build_adp_survives_a_missing_board(corpus):
    data = seo.build_adp(None)
    star = data["by_slug"]["star"]          # the id itself, as the archive does
    assert star["name"] == "star" and star["team"] is None and star["espn_adp"] is None


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
    assert "<script" not in body.replace('<script type="application/ld+json">', "")


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
