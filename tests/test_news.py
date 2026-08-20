"""pipeline/news.py -- player news + injury signals.

Nothing here touches the network. Both article feeds are recorded fixtures
captured from the live endpoints on 2026-08-19
(tests/fixtures/google_news_josh_allen.xml,
tests/fixtures/espn_news.json, tests/fixtures/sleeper_players.json), and
every fetch goes through the injectable `fetch(url) -> body` seam.
"""
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from pipeline import news
from pipeline.news import (ATTR_EXACT, ATTR_NAME, NEWS_COLUMNS, STATUS_COLUMNS,
                           fetch_player_news, fetch_player_status,
                           google_news_query, parse_espn_news,
                           parse_google_news, parse_sleeper_status)

FIXTURES = Path(__file__).parent / "fixtures"
GOOGLE_XML = (FIXTURES / "google_news_josh_allen.xml").read_text()
ESPN_JSON = (FIXTURES / "espn_news.json").read_text()
SLEEPER_JSON = (FIXTURES / "sleeper_players.json").read_text()


def _pool(rows=None):
    """A board-shaped pool: exactly the columns `news_pool` returns."""
    rows = rows or [
        ("00-0034857", "Josh Allen", "QB", "BUF", 3918298),
        ("00-0039139", "Jahmyr Gibbs", "RB", "DET", 4429795),
        ("00-0038542", "Bijan Robinson", "RB", "ATL", 4430807),
        ("adp_buffalo", "Buffalo Defense", "DST", "BUF", None),
    ]
    return pd.DataFrame(rows, columns=["player_id", "name", "position",
                                       "team", "espn_id"])


def _fetch_from(bodies, calls=None, fail=()):
    """A `fetch` seam over a {url-substring: body} table."""
    def fetch(url):
        if calls is not None:
            calls.append(url)
        for needle in fail:
            if needle in url:
                raise RuntimeError(f"boom: {needle}")
        for needle, body in bodies.items():
            if needle in url:
                return body
        raise KeyError(url)
    return fetch


# --------------------------------------------------------------- the query

def test_query_is_quoted_name_plus_team_nickname_plus_football():
    """The measured shape, term by term. See google_news_query's docstring
    for the numbers behind each one."""
    url = google_news_query("Josh Allen", "BUF")
    assert "%22Josh+Allen%22" in url          # the name is a quoted phrase
    assert "Bills" in url                     # team: ~74% -> ~90%
    assert url.endswith("+football&hl=en-US&gl=US&ceid=US:en") or "+football&" in url
    assert "QB" not in url                    # position: measured worse


def test_the_query_says_football_and_never_fantasy_football():
    """THE TRAP. 'football' and 'fantasy football' look like the same idea
    and one of them is wrong.

    Bare 'football' filters OUT namesakes who play another sport: across
    four players it took surname-relevance from 90/92/95/94% to 97/92/97/97%
    at no cost in volume. 'fantasy football' pulls IN mock drafts and
    start/sit columns that merely mention the player -- the same four
    players measured 80/77/88/78%, worse than no suffix at all.

    So this asserts the presence of one and the absence of the other in the
    same test, on purpose: a later edit that reaches for the fantasy variant
    fails here rather than quietly costing 15 points of precision."""
    url = google_news_query("Brock Bowers", "LV")
    assert "football" in url
    assert "fantasy" not in url.lower()


def test_query_speaks_nflverse_team_abbreviations():
    """The board's `team` column is nflverse's spelling -- LA for the Rams,
    WAS, LV -- while other feeds say LAR/WSH/OAK. Both must resolve."""
    assert "Rams" in google_news_query("Puka Nacua", "LA")
    assert "Rams" in google_news_query("Puka Nacua", "LAR")
    assert "Commanders" in google_news_query("Jayden Daniels", "WSH")
    assert "Raiders" in google_news_query("Ashton Jeanty", "LV")


def test_query_for_an_unknown_team_is_still_a_usable_name_search():
    """No nickname (an unmapped or missing team) drops that term only -- the
    name and the sport still narrow the search."""
    url = google_news_query("Some Rookie", None)
    assert "%22Some+Rookie%22" in url
    assert "football" in url


# ------------------------------------------------------- Google News feed

def test_parse_google_news_reads_a_recorded_feed():
    rows = parse_google_news(GOOGLE_XML, "00-0034857", "Josh Allen")
    assert rows, "the recorded feed has items"
    first = rows[0]
    assert set(first) == set(NEWS_COLUMNS[:-1])
    assert first["attribution"] == ATTR_NAME
    assert first["url"].startswith("https://news.google.com/rss/articles/")
    assert first["published_at"] is None or first["published_at"].tzinfo is None


def test_parse_google_news_strips_the_publication_off_the_headline():
    """Google appends ' - Publication' to every title AND sends the
    publication as its own element. Storing both would print the source
    twice in the feed panel."""
    rows = parse_google_news(GOOGLE_XML, "p1", "Josh Allen")
    snow = [r for r in rows if "snow game" in r["headline"]][0]
    assert snow["source"] == "Democrat and Chronicle"
    assert snow["headline"].endswith("commercial")
    assert " - Democrat and Chronicle" not in snow["headline"]


def test_parse_google_news_skips_a_malformed_item_without_losing_the_feed():
    """The fixture carries seven <item>s: one has no <link> (nothing to open,
    so it is dropped) and one has an unparseable pubDate (still a real
    headline, so it is kept with published_at null)."""
    rows = parse_google_news(GOOGLE_XML, "p1", "Josh Allen", limit=50)
    headlines = [r["headline"] for r in rows]
    assert "Item with no link at all" not in headlines
    undated = [r for r in rows if r["headline"] == "Item with an unparseable pubDate"]
    assert len(undated) == 1 and undated[0]["published_at"] is None


def test_parse_google_news_raises_on_a_body_that_is_not_xml():
    """A captive-portal HTML page or a truncated response must raise, so the
    caller can carry the player's stored rows forward rather than write an
    empty feed over them."""
    with pytest.raises(Exception):
        parse_google_news("<html>nope", "p1", "Josh Allen")


def test_a_name_matched_feed_can_carry_another_players_news():
    """THE COLLISION CASE, with a real item.

    The fixture's 'DEFENSE ON FIRE: Jacksonville Jaguars' Defense Set to
    DOMINATE With Travon Walker & Josh Allen' came back from a live
    '"Josh Allen" Jaguars' query on 2026-08-19: it is about the Jaguars'
    Josh Allen, not the Bills quarterback, and no query shape and no cheap
    filter can tell them apart -- they have the same name.

    So the guarantee is NOT that such an item never appears. It is that
    every item found this way is stamped ATTR_NAME and is therefore
    distinguishable downstream from an id-tagged one. If this assertion ever
    has to change, the UI's honesty about attribution changes with it.
    """
    rows = parse_google_news(GOOGLE_XML, "00-0034857", "Josh Allen", limit=50)
    wrong = [r for r in rows if "DEFENSE ON FIRE" in r["headline"]]
    assert len(wrong) == 1
    assert wrong[0]["attribution"] == ATTR_NAME
    assert all(r["attribution"] == ATTR_NAME for r in rows)


def test_google_items_are_capped_per_player():
    rows = parse_google_news(GOOGLE_XML, "p1", "Josh Allen", limit=2)
    assert len(rows) == 2


# ------------------------------------------------------------- ESPN feed

def test_espn_items_are_attributed_by_athlete_id():
    payload = json.loads(ESPN_JSON)
    by_id = {4430807: ("00-0038542", "Bijan Robinson"),
             4429795: ("00-0039139", "Jahmyr Gibbs")}
    rows = parse_espn_news(payload, by_id)
    assert rows and all(r["attribution"] == ATTR_EXACT for r in rows)
    assert {r["player_id"] for r in rows} == {"00-0038542", "00-0039139"}
    assert all(r["source"] == "ESPN" for r in rows)
    assert all(r["url"].startswith("https://www.espn.com/") for r in rows)


def test_espn_articles_about_players_off_the_board_produce_nothing():
    """The fixture's Vita Vea and Tuli Tuipulotu tags are real; neither is a
    draftable fantasy player, so neither may be stored unattributed."""
    payload = json.loads(ESPN_JSON)
    assert parse_espn_news(payload, {}) == []


def test_espn_article_with_no_athlete_tag_is_dropped():
    """The fixture's 'Colts TE Tyler Warren injures groin in practice' has no
    athlete category at all -- exactly the kind of item a name search would
    happily mis-attribute and an id join correctly refuses to."""
    payload = json.loads(ESPN_JSON)
    rows = parse_espn_news(payload, {4430807: ("p1", "Bijan Robinson")})
    assert not any("Tyler Warren" in r["headline"] for r in rows)


def test_espn_malformed_article_is_skipped():
    """'Article with no web link' has a headline but nothing to open."""
    payload = json.loads(ESPN_JSON)
    rows = parse_espn_news(payload, {4430807: ("p1", "Bijan Robinson")})
    assert not any(r["headline"] == "Article with no web link" for r in rows)


# ------------------------------------------------- the whole news table

def _both_feeds(**kw):
    return _fetch_from({"news.google.com": GOOGLE_XML,
                        "site.web.api.espn.com": ESPN_JSON}, **kw)


def test_fetch_player_news_combines_both_feeds():
    df = fetch_player_news(_pool(), fetch=_both_feeds(), max_workers=2)
    assert list(df.columns) == NEWS_COLUMNS
    assert set(df["attribution"]) == {ATTR_EXACT, ATTR_NAME}
    assert df["fetched_at"].notna().all()


def test_exact_items_sort_ahead_of_name_matched_ones_for_a_player():
    """A consumer that takes the head of a player's rows must get the items
    we are certain about first."""
    df = fetch_player_news(_pool(), fetch=_both_feeds(), max_workers=2)
    gibbs = df[df["player_id"] == "00-0039139"]
    assert gibbs.iloc[0]["attribution"] == ATTR_EXACT
    exact = (gibbs["attribution"] == ATTR_EXACT).to_numpy()
    assert list(exact) == sorted(exact, reverse=True)


def test_a_defense_never_gets_a_name_matched_query():
    """'Buffalo Defense' is not a person; searching that phrase returns
    whatever was written about the Bills, which is not news about the unit
    being drafted."""
    calls = []
    fetch_player_news(_pool(), fetch=_both_feeds(calls=calls), max_workers=1)
    assert not any("Defense" in c for c in calls)


def test_a_player_with_no_news_simply_has_no_rows():
    """An empty (but valid) feed is not an error -- a quiet news day must not
    show up as a failed source on the readiness strip."""
    empty = ('<?xml version="1.0"?><rss version="2.0"><channel>'
             "<title>none</title></channel></rss>")
    fetch = _fetch_from({"news.google.com": empty,
                         "site.web.api.espn.com": json.dumps({"articles": []})})
    df = fetch_player_news(_pool(), fetch=fetch, max_workers=1)
    assert df.empty
    assert list(df.columns) == NEWS_COLUMNS


def test_one_players_feed_failing_does_not_lose_the_others():
    fetch = _fetch_from({"news.google.com": GOOGLE_XML,
                         "site.web.api.espn.com": ESPN_JSON},
                        fail=("Gibbs",))
    df = fetch_player_news(_pool(), fetch=fetch, max_workers=1)
    named = df[df["attribution"] == ATTR_NAME]["player_id"].unique()
    assert "00-0034857" in named                     # Josh Allen still fetched
    assert "00-0039139" not in named                 # Gibbs' query failed
    assert "00-0039139" in set(df["player_id"])      # but his ESPN items survive


def test_a_failing_feed_carries_the_players_stored_rows_forward():
    """The failure behaviour that matters on draft night: a blip must never
    blank a profile that had news yesterday."""
    stored = pd.DataFrame([{
        "player_id": "00-0039139", "name": "Jahmyr Gibbs",
        "headline": "Yesterday's headline", "url": "https://example.com/1",
        "published_at": pd.Timestamp("2026-08-18"), "source": "Old Wire",
        "attribution": ATTR_NAME,
        "fetched_at": pd.Timestamp.utcnow().tz_localize(None) - timedelta(days=3),
    }], columns=NEWS_COLUMNS)
    fetch = _fetch_from({"news.google.com": GOOGLE_XML,
                         "site.web.api.espn.com": ESPN_JSON}, fail=("Gibbs",))
    df = fetch_player_news(_pool(), fetch=fetch, existing=stored, max_workers=1)
    assert "Yesterday's headline" in set(df["headline"])


def test_the_espn_feed_failing_still_yields_the_name_matched_rows():
    fetch = _fetch_from({"news.google.com": GOOGLE_XML},
                        fail=("site.web.api.espn.com",))
    df = fetch_player_news(_pool(), fetch=fetch, max_workers=1)
    assert not df.empty
    assert set(df["attribution"]) == {ATTR_NAME}


def test_every_feed_failing_raises_rather_than_blanking_the_table():
    """write_table does CREATE OR REPLACE, so returning an empty frame when
    the whole internet is unreachable would delete the cached news. Raising
    makes refresh record FAIL and leave the previous table alone."""
    fetch = _fetch_from({}, fail=("http",))
    with pytest.raises(ValueError):
        fetch_player_news(_pool(), fetch=fetch, max_workers=1)


def test_recently_fetched_players_are_not_requeried():
    """Google News RSS sends no ETag and no Last-Modified, so our own
    fetched_at is the only 'has this changed' signal available."""
    recent = pd.Timestamp.utcnow().tz_localize(None) - timedelta(hours=1)
    stored = pd.DataFrame([{
        "player_id": "00-0034857", "name": "Josh Allen",
        "headline": "Cached headline", "url": "https://example.com/cached",
        "published_at": pd.Timestamp("2026-08-19"), "source": "Wire",
        "attribution": ATTR_NAME, "fetched_at": recent,
    }], columns=NEWS_COLUMNS)
    calls = []
    df = fetch_player_news(_pool(), fetch=_both_feeds(calls=calls),
                           existing=stored, ttl_hours=12, max_workers=1)
    assert not any("Josh+Allen" in c for c in calls)
    assert "Cached headline" in set(df["headline"])
    assert any("Gibbs" in c for c in calls)          # the others still refresh


def test_ttl_zero_refetches_everyone():
    recent = pd.Timestamp.utcnow().tz_localize(None)
    stored = pd.DataFrame([{
        "player_id": "00-0034857", "name": "Josh Allen",
        "headline": "Cached headline", "url": "https://example.com/cached",
        "published_at": pd.Timestamp("2026-08-19"), "source": "Wire",
        "attribution": ATTR_NAME, "fetched_at": recent,
    }], columns=NEWS_COLUMNS)
    calls = []
    fetch_player_news(_pool(), fetch=_both_feeds(calls=calls),
                      existing=stored, ttl_hours=0, max_workers=1)
    assert any("Josh+Allen" in c for c in calls)


def test_stored_exact_rows_are_never_carried_forward():
    """ESPN is one cheap request for the whole board, so its half of the
    table is always rebuilt -- carrying it forward would accumulate articles
    that have long fallen off the feed."""
    stale = pd.DataFrame([{
        "player_id": "00-0034857", "name": "Josh Allen",
        "headline": "Stale ESPN item", "url": "https://www.espn.com/old",
        "published_at": pd.Timestamp("2026-01-01"), "source": "ESPN",
        "attribution": ATTR_EXACT,
        "fetched_at": pd.Timestamp.utcnow().tz_localize(None),
    }], columns=NEWS_COLUMNS)
    df = fetch_player_news(_pool(), fetch=_both_feeds(), existing=stale,
                           max_workers=1)
    assert "Stale ESPN item" not in set(df["headline"])


def test_the_same_article_is_stored_once_per_player():
    df = fetch_player_news(_pool(), fetch=_both_feeds(), max_workers=2)
    assert not df.duplicated(["player_id", "url"]).any()


# ---------------------------------------------------- injury signals

def test_sleeper_status_matches_the_pool_by_name_and_position():
    """Note Puka Nacua: the board says team LA (nflverse), Sleeper says LAR.
    The match is on (name, position) precisely so an abbreviation
    disagreement cannot drop an injury."""
    pool = pd.DataFrame(
        [("00-0039139", "Jahmyr Gibbs", "RB", "DET", None),
         ("00-0039075", "Puka Nacua", "WR", "LA", None),
         ("00-0034857", "George Kittle", "TE", "SF", None)],
        columns=["player_id", "name", "position", "team", "espn_id"])
    df = parse_sleeper_status(json.loads(SLEEPER_JSON), pool)
    assert list(df.columns) == STATUS_COLUMNS
    byname = df.set_index("name")
    assert byname.loc["Puka Nacua", "injury_status"] == "Questionable"
    assert byname.loc["George Kittle", "injury_body_part"] == "Achilles"
    assert pd.isna(byname.loc["Jahmyr Gibbs", "injury_status"])


def test_sleeper_status_does_not_depend_on_sleeper_ids():
    """The fixture's Bijan Robinson, Jahmyr Gibbs and Puka Nacua all have
    gsis_id AND espn_id null -- that is the live payload, not a trimmed
    fixture. Joining on Sleeper's ids would drop the top of the board, which
    is why this matches on name instead."""
    payload = json.loads(SLEEPER_JSON)
    top = [p for p in payload.values()
           if p["full_name"] in {"Bijan Robinson", "Jahmyr Gibbs", "Puka Nacua"}]
    assert top and all(p["gsis_id"] is None and p["espn_id"] is None for p in top)
    pool = pd.DataFrame([("00-0038542", "Bijan Robinson", "RB", "ATL", None)],
                        columns=["player_id", "name", "position", "team", "espn_id"])
    df = parse_sleeper_status(payload, pool)
    assert df.iloc[0]["sleeper_id"] == "9509"


def test_sleeper_status_ignores_inactive_and_unrostered_players():
    """The fixture holds a real second Josh Allen -- a retired guard -- and a
    rostered-nowhere Le'Veon Bell. Neither may supply an injury note."""
    pool = pd.DataFrame([("x", "Josh Allen", "G", None, None),
                         ("y", "Le'Veon Bell", "RB", "PIT", None)],
                        columns=["player_id", "name", "position", "team", "espn_id"])
    df = parse_sleeper_status(json.loads(SLEEPER_JSON), pool)
    assert df["sleeper_id"].isna().all()


def test_sleeper_status_drops_an_ambiguous_name_and_position_pair():
    """Constructed, because the live payload has ZERO duplicate
    (name, position) pairs among active rostered fantasy players -- which is
    exactly why the name match is safe here and not in the news query. If
    that ever changes, an ambiguous injury note is worse than none, so both
    sides are dropped rather than one picked arbitrarily."""
    payload = {
        "1": {"full_name": "Josh Allen", "position": "QB", "team": "BUF",
              "active": True, "injury_status": "Questionable"},
        "2": {"full_name": "Josh Allen", "position": "QB", "team": "JAX",
              "active": True, "injury_status": "Out"},
        "3": {"full_name": "Jahmyr Gibbs", "position": "RB", "team": "DET",
              "active": True, "injury_status": "Questionable"},
    }
    pool = pd.DataFrame([("a", "Josh Allen", "QB", "BUF", None),
                         ("b", "Jahmyr Gibbs", "RB", "DET", None)],
                        columns=["player_id", "name", "position", "team", "espn_id"])
    df = parse_sleeper_status(payload, pool).set_index("name")
    assert pd.isna(df.loc["Josh Allen", "injury_status"])
    assert df.loc["Jahmyr Gibbs", "injury_status"] == "Questionable"


def test_every_pool_player_gets_a_status_row_even_unmatched():
    """Jake Moody is the one real board player Sleeper has no rostered entry
    for (Sleeper carries him with team null). A missing row would make the
    downstream join drop him from the board; an empty row does not."""
    pool = pd.DataFrame([("k1", "Jake Moody", "K", "WAS", None)],
                        columns=["player_id", "name", "position", "team", "espn_id"])
    df = parse_sleeper_status(json.loads(SLEEPER_JSON), pool)
    assert len(df) == 1 and pd.isna(df.iloc[0]["sleeper_id"])
    assert df.iloc[0]["player_id"] == "k1"


def test_news_updated_epoch_millis_becomes_a_timestamp():
    pool = pd.DataFrame([("00-0038542", "Bijan Robinson", "RB", "ATL", None)],
                        columns=["player_id", "name", "position", "team", "espn_id"])
    df = parse_sleeper_status(json.loads(SLEEPER_JSON), pool)
    assert isinstance(df.iloc[0]["news_updated"], pd.Timestamp)
    assert df.iloc[0]["news_updated"].year >= 2020


def test_fetch_player_status_goes_through_the_injectable_seam():
    pool = pd.DataFrame([("00-0039075", "Puka Nacua", "WR", "LA", None)],
                        columns=["player_id", "name", "position", "team", "espn_id"])
    calls = []
    df = fetch_player_status(pool, fetch=_fetch_from(
        {"api.sleeper.app": SLEEPER_JSON}, calls=calls))
    assert calls == [news.SLEEPER_PLAYERS_URL]
    assert df.iloc[0]["injury_status"] == "Questionable"


# ------------------------------------------------------------- plumbing

def test_both_tables_are_registered_as_universal():
    """A table in neither UNIVERSAL_TABLES nor LEAGUE_TABLES is treated as
    universal by provisioning anyway, so the classification has to be
    explicit -- and news is genuinely universal, not per-league."""
    from pipeline.db import LEAGUE_TABLES, UNIVERSAL_TABLES
    assert {"player_news", "player_status"} <= UNIVERSAL_TABLES
    assert {"player_news", "player_status"}.isdisjoint(LEAGUE_TABLES)


def test_the_tables_round_trip_and_show_up_on_the_readiness_strip(tmp_path):
    """/api/landing/status lists whatever `meta` holds, so a source that goes
    through write_table + record_freshness appears there with no endpoint
    change -- which is the whole reason this follows the existing pattern."""
    from pipeline.db import get_conn, read_table, record_freshness, write_table
    conn = get_conn(str(tmp_path / "t.duckdb"))
    df = fetch_player_news(_pool(), fetch=_both_feeds(), max_workers=1)
    write_table(conn, "player_news", df)
    record_freshness(conn, "player_news", True, len(df))
    back = read_table(conn, "player_news")
    assert len(back) == len(df)
    assert list(back.columns) == NEWS_COLUMNS
    meta = read_table(conn, "meta")
    assert "player_news" in set(meta["source"])


def test_news_pool_is_the_board(monkeypatch):
    """The board is the authority on player_id and espn_id; re-deriving
    either here would be a second copy of a matching rule that has to agree
    to the character."""
    board = pd.DataFrame([{"player_id": "p1", "name": "A", "position": "RB",
                           "team": "DET", "espn_id": 1, "rank": 1,
                           "composite": 0.9}])
    import scoring.board
    monkeypatch.setattr(scoring.board, "build_board", lambda conn, *a, **k: board)
    out = news.news_pool(object())
    assert list(out.columns) == ["player_id", "name", "position", "team", "espn_id"]


def test_refresh_runs_both_news_jobs_and_stamps_them_into_meta(tmp_path, monkeypatch):
    """The wiring, end to end, with every network call stubbed.

    `get_conn` is monkeypatched rather than steered by DRAFT_DB_PATH because
    refresh.main() defaults to the owner's real data/nfl.duckdb, and a test
    that writes there would overwrite his board.
    """
    from pipeline import db, refresh
    from pipeline.db import get_conn, read_table

    conn = get_conn(str(tmp_path / "refresh.duckdb"))
    monkeypatch.setattr(refresh, "get_conn", lambda *a, **k: conn)
    # Every non-news source stubbed to one cheap row, so the run exercises
    # the job table without hitting ten upstreams.
    for attr in dir(refresh.sources):
        if attr.startswith("fetch_"):
            monkeypatch.setattr(refresh.sources, attr,
                                lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(news, "news_pool", lambda c: _pool())
    monkeypatch.setattr(news, "http_fetch",
                        lambda *a, **k: _fetch_from({
                            "news.google.com": GOOGLE_XML,
                            "site.web.api.espn.com": ESPN_JSON,
                            "api.sleeper.app": SLEEPER_JSON}))

    assert refresh.main() == 0
    assert not read_table(conn, "player_news").empty
    assert len(read_table(conn, "player_status")) == len(_pool())
    meta = set(read_table(conn, "meta")["source"])
    assert {"player_news", "player_status"} <= meta
    # Both are classified, which is what stops provisioning treating an
    # unclassified table as universal by accident.
    assert {"player_news", "player_status"} <= db.UNIVERSAL_TABLES
