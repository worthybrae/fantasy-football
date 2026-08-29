# tests/test_seo.py
"""The pages a search engine reads: ADP from the corpus, rendered as HTML."""
import html
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import scoring.profile  # noqa: F401 -- see below
from api import market, seo
from pipeline import draft_log as dl

# `scoring/profile.py` BINDS `cached_build_board` BY NAME AT IMPORT
# (`from scoring.board_cache import cached_build_board`), so whichever test
# first causes that module to be imported decides what the name means for the
# rest of the run. Several tests below hand `board_cache` a fake board for
# `scoring/adp_facts.board_rows`, and the player pages now build a profile
# too -- so without the import above, the first of those tests imported
# `scoring.profile` while its fake was installed and every later test's
# profile was built from a one-row defense frame. Imported here, at
# collection, the binding is made before any monkeypatch exists.


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
    """Ten drafts, 4 teams x 6 rounds (24 picks each), so shares are round
    numbers -- and so that every behaviour these pages have can be told
    apart from the plausible wrong version of it.

    WHO IS IN IT, AND WHAT EACH ONE IS FOR.

    `star` (RB) goes first in every draft: the concentrated player, and the
    ten human picks the deliberation clock needs to say anything at all.

    `mid` (RB) goes 2nd, 3rd or 4th (2,3,4,2,3,4,2,3,4,2): mean 2.9 against
    a median of 3.0, so a page that prints the mean where it says median is
    a page that fails.

    `swing` (QB) is the awkward one, and deliberately: picks 9, 10, 11, 14,
    15, 17, 18, 22, 23, 24. Mean 16.3 against a median of 16.0. p10 is 10 and
    p20 is 11. Rounds 3 and 6 hold three of his picks each and round 4 holds
    two -- so the modal round is a tie that is NOT the median's round, and
    the median's round is the answer.

    `ghost` goes at pick 8 in eight drafts and pick 24 in two, which puts the
    median of the first tight end taken at 8 where the average would say 11.

    `late` (WR) is taken in two drafts and POOLED IN FOUR: the honest
    denominator, 50% of the drafts he was on the board for against 20% of
    the drafts. `once` (WR) is taken in the last draft only (10%, still
    above MIN_SHARE's 2%) and carries a sentinel ESPN ADP on the board
    fixture.

    `twin_a` and `twin_b` share a display name, to exercise slug collisions
    -- and they are the corpus's two robots: `twin_a` is autodrafted in
    every draft and `twin_b` is taken by slot 2, which is `my_slot`, the
    farm's own seat. Both are 88 and 99 seconds where every human pick is
    30, so a clock that counts them says something different from a clock
    that does not.

    `ghost` (TE) is taken in every draft but never gets a name in the
    `board` fixture -- an unresolved name, to prove such a player gets no
    page. He is also where the tight end run starts, above.
 `adp_seattle_defense` is the same problem one step worse: a
    synthetic id with no `players` row at all, the way every D/ST and every
    rookie reaches these pages.
    """
    path = tmp_path / "corpus.duckdb"
    conn = dl.corpus_conn(str(path))
    # The awkward quarterback, one pick per draft. See the docstring.
    swing = [9, 10, 11, 14, 15, 17, 18, 22, 23, 24]
    # The tight end, whose first-pick median (8) and average (11.2) differ.
    ghost = [8, 8, 8, 8, 24, 24, 8, 8, 8, 8]
    for i in range(10):
        conn.execute(
            "INSERT INTO draft_log (draft_id, source, league_id, season,"
            " recorded_at, teams, rounds, my_slot, scoring_json, settings_json,"
            " human_seats) VALUES (?, 'mock', '1', 2026, ?::TIMESTAMP, 4, 6,"
            " 2, NULL, NULL, 3)", [f"d{i}", f"2026-08-24 {i:02d}:00:00"])
        picks = [("star", 1), ("mid", 2 + (i % 3)), ("twin_a", 5), ("twin_b", 6),
                 ("ghost", ghost[i]), ("swing", swing[i])]
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
            elif pid == "swing":
                position = "QB"
            elif pid.startswith("adp_"):
                position = "DST"
            else:
                position = "WR"
            # THE TWO KINDS OF PICK THE CLOCK MUST NOT COUNT. `twin_a` is
            # the autodrafter and `twin_b` is the farm's own seat; every
            # other pick here is a person, and every person takes 30
            # seconds.
            autodrafted = pid == "twin_a"
            seconds = 88.0 if pid == "twin_a" else 99.0 if pid == "twin_b" else 30.0
            conn.execute(
                "INSERT INTO draft_log_pick (draft_id, pick_no, round, slot,"
                " owner_key, is_anonymous, player_id, position, adp_rank,"
                " proj_points, autodrafted, had_owner, seconds_to_pick,"
                " clock_seconds) VALUES (?, ?, ?, ?, 'o', FALSE, ?, ?, 1.0,"
                " 100.0, ?, TRUE, ?, 120.0)",
                [f"d{i}", pick_no, (pick_no - 1) // 4 + 1, (pick_no - 1) % 4 + 1,
                 pid, position, autodrafted, seconds])
        # THE POOL, which is what `of` and the availability strip count
        # against. Everybody is on the board in every draft except `late`,
        # who is on it in four -- ESPN added him to its board in the middle
        # of the summer, and the drafts before that could not have taken
        # him. He is taken in two, which is 20% of the drafts and 50% of the
        # drafts he could have gone in.
        for pid in ("star", "mid", "twin_a", "twin_b", "ghost", "once", "swing",
                    "adp_seattle_defense"):
            conn.execute(
                "INSERT INTO draft_log_pool (draft_id, player_id, position,"
                " team, adp_rank, proj_points, espn_rank, espn_proj, bye)"
                " VALUES (?, ?, 'RB', 'CHI', 1.0, 100.0, 1.0, 100.0, 9)",
                [f"d{i}", pid])
        if i < 4:
            conn.execute(
                "INSERT INTO draft_log_pool (draft_id, player_id, position,"
                " team, adp_rank, proj_points, espn_rank, espn_proj, bye)"
                " VALUES (?, 'late', 'WR', 'NYJ', 1.0, 100.0, 1.0, 100.0, 9)",
                [f"d{i}"])
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
        ("swing", "Swing Guy", None, None, None, None, None),
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
        # THE OTHER SCALE. This feed has sent participation as a share and as
        # a percentage; 60 here means 60%, not 6000%.
        ("swing", "Swing Guy", "Questionable", "knee", 60),
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
        # THE SENTINEL. ESPN publishes five hundred players and parks
        # everybody its lobby never drafts just under 170 -- 329 rows of the
        # real five hundred sit above 165. It is not a draft position and
        # must not be read as one.
        (555, "Once Guy", "WR", "NYJ", 169.5),
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
    assert data["drafts"] == 10 and data["teams"] == 4 and data["rounds"] == 6
    assert str(data["updated"]) == "2026-08-24"
    ranked = [p["player_id"] for p in data["players"]]
    assert ranked[:2] == ["star", "mid"]
    assert "ghost" not in ranked   # a row in `players`, but no resolved name
    star = data["by_slug"]["dandre-swift"]
    assert star["adp"] == 1.0 and star["taken"] == 10 and star["share"] == 1.0
    assert star["usual_round"] == 1 and star["rank"] == 1 and star["pos_rank"] == 1
    assert star["hist"][0] == 10 and len(star["hist"]) == 24
    mid = data["by_slug"]["amon-ra-st-brown"]
    assert mid["adp"] == 2.9 and mid["p10"] == 2 and mid["p90"] == 4
    assert mid["low"] == 2 and mid["high"] == 4


# -- the six behaviours the fixture exists to tell apart ---------------------
#
# Each of these fails if the obvious wrong version of the code is written
# instead, and the docstring names which wrong version. A fixture where every
# player is pooled in every draft, every pick is human and every pick is in
# round 1 passes all six either way, which is what this corpus was rebuilt to
# stop being.

def test_the_denominator_is_the_drafts_he_was_on_the_board_for(corpus, board):
    """MUTATION: `of = total`. `late` was taken in two drafts and was on the
    board in four -- ESPN added him in the middle of the summer. He goes in
    half the drafts that could have taken him and a fifth of all of them."""
    late = seo.build_adp(board)["by_slug"]["late-guy"]
    assert late["taken"] == 2 and late["of"] == 4
    assert late["share"] == 0.2 and late["of_share"] == 0.5
    body = _client(board).get("/adp/late-guy").text
    assert "50% of the 4 drafts" in body


def test_the_middle_of_his_picks_is_a_median_not_an_average(corpus, board):
    """MUTATION: median -> mean. `mid` averages 2.9 with a median of 3;
    `swing` averages 14.7 with a median of 14."""
    d = seo.build_adp(board)
    mid = d["by_slug"]["amon-ra-st-brown"]
    assert mid["adp"] == 2.9 and mid["median"] == 3.0
    swing = d["by_slug"]["swing-guy"]
    assert swing["adp"] == 16.3 and swing["median"] == 16.0
    assert "median 3" in _client(board).get("/adp/amon-ra-st-brown").text


def test_the_range_is_the_tenth_percentile_not_the_twentieth(corpus, board):
    """MUTATION: p10 -> p20. `swing` goes at 9, 10, 11, 14, 15, 17, 18, 22,
    23, 24: the tenth percentile of that is 10 and the twentieth is 11."""
    swing = seo.build_adp(board)["by_slug"]["swing-guy"]
    assert swing["p10"] == 10 and swing["p90"] == 23
    assert swing["low"] == 9 and swing["high"] == 24


def test_the_usual_round_is_the_median_round_not_the_modal_one(corpus, board):
    """MUTATION: the median rule -> `Counter(...).most_common(1)`. Three of
    `swing`'s ten picks are in round 3 and three are in round 6, so the mode
    is a tie neither side of which is where the middle of his picks falls:
    round 4, on two picks. A player the rooms really do concentrate keeps his
    round either way, which is the other half of the rule."""
    d = seo.build_adp(board)
    swing = d["by_slug"]["swing-guy"]
    assert swing["usual_round"] == 4 and swing["usual_share"] == 0.2
    assert d["by_slug"]["dandre-swift"]["usual_round"] == 1


def test_a_position_run_starts_at_the_median_first_pick(corpus, board):
    """MUTATION: `_runs` median -> avg. `ghost` is the only tight end here,
    so the first TE off the board is his own pick: 8 in eight drafts and 24
    in two, which is a median of 8 and an average of 11.2. One room reaching
    for a position in the last round does not move where the run starts."""
    runs = {r["position"]: r for r in seo.build_adp(board)["runs"]}
    assert runs["TE"]["pick"] == 8 and runs["TE"]["round"] == 2
    assert runs["TE"]["drafts"] == 10


def test_the_deliberation_clock_counts_people_only(corpus, board):
    """MUTATION: drop `autodrafted` and `my_slot` from `_clock`. Every human
    pick in this corpus takes 30 seconds. `twin_a` is autodrafted in every
    draft (88 seconds) and `twin_b` is taken by the farm's own seat (99), and
    neither number is a fact about anybody deliberating."""
    d = seo.build_adp(board)
    assert d["seconds"] == 30.0
    assert d["by_slug"]["dandre-swift"]["seconds"] == 30.0
    assert d["by_slug"]["josh-allen"]["seconds"] is None      # autodrafted
    assert d["by_slug"]["josh-allen-2"]["seconds"] is None    # the farm's seat


def test_a_seat_snakes_back_on_an_even_round(corpus, board):
    """MUTATION: drop the snake from `_seats`. Round 2 of a 4-team draft runs
    backwards: seat 1 picks last at pick 8 and seat 4 picks first at pick 5.
    Round 3 runs out again."""
    d = seo.build_adp(board)
    assert [(s["slot"], s["pick"]) for s in seo._seats(d, 2)] == [(1, 8), (2, 7), (4, 5)]
    assert [(s["slot"], s["pick"]) for s in seo._seats(d, 3)] == [(1, 9), (2, 10), (4, 12)]
    body = html.unescape(_client(board).get("/adp/round/2").text)
    assert "Seat 1 — pick 8" in body and "Seat 4 — pick 5" in body


def test_a_gap_measured_over_a_handful_of_rooms_says_so(corpus, board, monkeypatch):
    """`once` is taken in one draft of ten. Whatever his gap against ESPN
    works out to, it is the average of one room, and the figure says which
    rooms it came from instead of pointing green or amber at a drafter."""
    monkeypatch.setattr(seo, "MOVE_PICKS", 1)
    seo.clear_pages()
    market._CACHE.clear()
    # ...with ESPN ranking him, so there is a gap to caveat at all.
    board.execute("UPDATE espn_adp SET espn_adp = 120.0 WHERE espn_id = 555")
    d = seo.build_adp(board)
    once = d["by_slug"]["once-guy"]
    assert once["of_share"] < seo.MOVER_MIN_SHARE and once["vs_espn"] is not None
    assert once not in d["risers"] and once not in d["fallers"]
    body = _client(board).get("/adp/once-guy").text
    assert "from the 10% of rooms that drafted him" in body
    assert "rooms that drafted him at all" in body
    assert 'class="v early"' not in body and 'class="v late"' not in body
    # ...muted, and muted in a way that beats `.fig .v`'s own colour.
    assert 'class="v dim"' in body
    assert ".fig .v.dim{color:var(--text-2)}" in body
    # ...while a player the rooms do take keeps the colour.
    swift = _client(board).get("/adp/dandre-swift").text
    assert 'class="v early"' in swift or 'class="v late"' in swift


def test_espns_undrafted_sentinel_is_not_a_draft_position(corpus, board):
    """MUTATION: read a sentinel ADP as a rank. `once` carries ESPN's
    169.5, which is where ESPN parks a player its lobby never drafts -- 329
    of its five hundred rows are up there. He has no place on the vs-ESPN
    scale, and his page says so instead of comparing him with a placeholder.
    """
    d = seo.build_adp(board)
    once = d["by_slug"]["once-guy"]
    assert once["espn_adp"] == 169.5 and once["espn_undrafted"] is True
    assert once["vs_espn"] is None and once["espn_pick"] is None
    body = _client(board).get("/adp/once-guy").text
    assert "undrafted on ESPN's board" in body
    assert "comes off it around pick" not in body
    # ...and the players ESPN really does rank still get the comparison.
    assert d["by_slug"]["dandre-swift"]["vs_espn"] is not None


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
    assert "https://espnfantasydraft.com/adp/round/6" in locs
    assert "https://espnfantasydraft.com/adp/round/7" not in locs   # six rounds
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
    228-URL sitemap is the caller that makes this worth holding."""
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
        " '2026-08-25 00:00:00'::TIMESTAMP, 4, 6, NULL, NULL, NULL, 3)")
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


def test_a_board_that_fails_once_is_asked_again(corpus, board, monkeypatch):
    """THE COLD-START RACE. `cached_build_board` raises when another thread
    is part-way through building the same board on the same connection,
    which is what a keep-warm pass landing on a request looks like. Forty of
    the two hundred published players -- every D/ST, every rookie -- have no
    `players` row and no name without it, and a player who cannot be named
    gets no page: the sitemap went 228 to 187 and stayed there until the
    next warm pass. The race clears in about a second, which is what the one
    retry is for."""
    from scoring import board_cache
    calls = []

    def flaky(conn, *args, **kwargs):
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("another thread is building this board")
        return _board_frame()

    monkeypatch.setattr(board_cache, "cached_build_board", flaky)
    d = seo.build_adp(board)
    assert len(calls) == 2, "the board was not retried"
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
    assert len(calls) == 3, "the second build did not retry"
    assert set(second["by_slug"]) == set(first["by_slug"])
    assert second["by_slug"]["seattle-defense"]["bye"] == 9


def test_a_remembered_board_a_day_old_is_not_served(corpus, board, monkeypatch):
    """The trade stops being worth making. A bye week from yesterday's board
    is a wrong answer rather than an old one, and these pages already print a
    blank for every player the crosswalks miss."""
    from scoring import board_cache
    monkeypatch.setattr(board_cache, "cached_build_board",
                        lambda conn, *a, **k: _board_frame())
    assert "seattle-defense" in seo.build_adp(board)["by_slug"]

    def gone(conn, *args, **kwargs):
        raise RuntimeError("gone")

    monkeypatch.setattr(board_cache, "cached_build_board", gone)
    monkeypatch.setattr(seo, "_last_board_at",
                        time.time() - seo.BOARD_MEMORY_SECONDS - 1)
    d = seo.build_adp(board)
    assert "seattle-defense" not in d["by_slug"]
    # ...and the memory is dropped rather than kept for the next build.
    assert seo._last_board_facts == {}
    # The corpus's own figures are untouched by any of it.
    assert d["by_slug"]["dandre-swift"]["adp"] == 1.0


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
    # The name a person says out loud, not the abbreviation: this node is
    # read by something that has never seen a depth chart.
    assert person["affiliation"]["name"] == "Chicago Bears"
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
    # Green for the rooms getting there first, amber for them waiting.
    assert '<span class="d big early">+2</span>' in body
    assert '<span class="d big late">-2</span>' in body
    # ...and every mover says how often the rooms take him at all.
    assert "taken in 100%" in body
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


# -- what a page says, and how it reads --------------------------------------

def test_every_title_and_description_fits_a_search_result(corpus, board):
    """A title past sixty characters is cut off, and on these pages the half
    that would be cut is the half that says which player it is about."""
    c = _client(board)
    for path in ("/adp", "/adp/rb", "/adp/qb", "/adp/round/1", "/adp/round/6",
                 "/adp/dandre-swift", "/adp/amon-ra-st-brown", "/adp/swing-guy",
                 "/adp/late-guy"):
        body = c.get(path).text
        title = html.unescape(re.search(r"<title>(.*?)</title>", body).group(1))
        desc = html.unescape(
            re.search(r'<meta name="description" content="(.*?)">', body).group(1))
        assert len(title) <= seo.TITLE_MAX, (path, len(title), title)
        assert len(desc) <= seo.DESC_MAX, (path, len(desc), desc)
        assert title and desc


def test_the_brand_is_the_part_of_a_title_that_gives_way():
    short = "Round 11 of an ESPN mock draft"
    assert seo._title(short).endswith("ESPN Draft Assist")
    long = "Jacory Croskey-Merritt ADP – ESPN mock drafts 2026"
    assert seo._title(long) == long
    assert len(seo._title(long)) <= seo.TITLE_MAX


def test_a_gap_that_rounds_to_nothing_does_not_print_minus_zero(corpus, board):
    """`'%+.0f' % -0.4` is "-0", which reads as a negative quantity of
    nothing."""
    assert seo._signed(-0.4) == "0" and seo._signed(-0.04, 1) == "0.0"
    assert seo._signed(2.0) == "+2" and seo._signed(-8.0) == "-8"
    assert seo._signed(-8.25, 1) == "-8.2"
    body = _client(board).get("/adp").text
    assert ">-0<" not in body and ">+0<" not in body


def test_the_prose_on_these_pages_clears_the_contrast_floor(corpus, board):
    """--text-3 is #565b69 on #0a0b0e: 2.9 to 1, against the 4.5 small text
    needs. Prose and labels take --text-2, which is 7.4 to 1."""
    body = _client(board).get("/adp").text
    assert "color:var(--text-3)" not in body
    assert "--text-2:#98a0b0" in body


def test_the_vs_espn_column_is_green_for_early_and_amber_for_late(corpus, board,
                                                                 monkeypatch):
    """Colour is never the only carrier -- the sign is in the cell and the
    legend is under the table -- but the two directions must not be the wrong
    way round: getting to him first is the good news."""
    monkeypatch.setattr(seo, "MOVE_PICKS", 1)
    seo.clear_pages()
    market._CACHE.clear()
    body = _client(board).get("/adp").text
    assert ".early,.fig .v.early{color:var(--ok)}" in body
    assert ".late,.fig .v.late{color:var(--accent)}" in body
    assert "before ESPN&#39;s board would" in body or "before ESPN's board would" in body
    # ...and the ESPN column is a different number from the vs-ESPN one.
    assert "published average draft position" in body


def test_a_defense_is_not_a_he(corpus, board, monkeypatch):
    """Twenty-three of the two hundred published players are defenses, and
    the copy on this page is written in sentences."""
    from scoring import board_cache
    monkeypatch.setattr(board_cache, "cached_build_board",
                        lambda conn, *a, **k: _board_frame())
    raw = _client(board).get("/adp/seattle-defense").text
    # The prose only: the shared stylesheet and the analytics tag are not
    # sentences anybody reads.
    body = html.unescape(re.sub(r"<(style|script)\b.*?</\1>", "", raw, flags=re.S)).lower()
    for pronoun in (" he ", " him ", " his "):
        assert pronoun not in body, pronoun
    assert "rooms take it at pick" in body
    kinds = {block["@type"] for block in _ld_json(raw)}
    assert "SportsTeam" in kinds and "Person" not in kinds


def test_practice_participation_reads_on_either_scale(corpus, board):
    """The feed has sent this as a share and as a percentage. 1 is a full
    week; 60 is 60% of one, not 6000%."""
    d = seo.build_adp(board)
    assert d["by_slug"]["dandre-swift"]["status"]["practice"] == 1.0
    assert d["by_slug"]["swing-guy"]["status"]["practice"] == 0.6
    assert "practice participation 60%" in _client(board).get("/adp/swing-guy").text


def test_the_season_sparkline_is_spaced_by_year(corpus, board):
    """MUTATION: x = i * step. Two of these seasons are four years apart and
    two are two, and a line that draws both gaps the same width is a line
    about a career this player did not have."""
    from scoring.adp_facts import sparkline
    rows = [{"season": 2020, "adp": 10, "cheat": None},
            {"season": 2024, "adp": 20, "cheat": None},
            {"season": 2026, "adp": 30, "cheat": None}]
    xs = [float(point.split(",")[0]) for point in sparkline(rows, width=120).split()]
    assert xs == [0.0, 80.0, 120.0]


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


# -- the profile the app draws, on a page a crawler can read -----------------
#
# These sections are the popup's fifteen cards, rendered from the popup's own
# payload (`scoring.profile_cache.cached_profile`). What the tests below are
# for is the seam: that the numbers really are that payload's rather than a
# second calculation, that the page survives the payload not being there at
# all, and that a defense -- which has almost none of it -- is a page rather
# than an exception.

@pytest.fixture
def profiled(tmp_path):
    """A universal database a REAL profile can be built from, keyed the way
    the `corpus` fixture is.

    The `board` fixture above is deliberately partial -- `cached_build_board`
    cannot build from it, which is what the retry and fallback tests need --
    so nothing there can answer a profile. This one is the other case: the
    same two ids (`star`, `mid`) with weekly rows behind them, so the board
    builds, the profile builds, and the page renders every section.

    Modelled on `tests/test_profile.py`'s own `_seed`, which is the minimum
    `build_board` and `build_profile` are known to work against.
    """
    import pandas as pd
    from pipeline.db import get_conn, write_table
    from scoring import board_cache, profile_cache

    conn = get_conn(str(tmp_path / "universal.duckdb"))
    weekly = []
    for pid, name, team in (("star", "D'Andre Swift", "CHI"),
                            ("mid", "Amon-Ra St. Brown", "DET")):
        for season, per_week in ((2024, 5), (2025, 7)):
            for week in range(1, 13):
                weekly.append({
                    "player_id": pid, "player_display_name": name,
                    "position": "RB", "recent_team": team, "opponent_team": "GB",
                    "season": season, "week": week,
                    "carries": per_week + 8, "rushing_yards": per_week * 9,
                    "rushing_tds": 1 if week % 4 == 0 else 0,
                    "targets": per_week, "receptions": per_week - 2,
                    "receiving_yards": per_week * 7,
                    "receiving_tds": 1 if week % 6 == 0 else 0})
    write_table(conn, "weekly", pd.DataFrame(weekly))
    # `market._names` reads this one, which is what gives the pages their
    # slugs; the board and the profile read `weekly` above.
    write_table(conn, "players", pd.DataFrame([
        {"gsis_id": "star", "display_name": "D'Andre Swift", "headshot": None,
         "birth_date": "1999-01-14", "rookie_season": 2020, "height": 70.0,
         "weight": 215.0},
        {"gsis_id": "mid", "display_name": "Amon-Ra St. Brown", "headshot": None,
         "birth_date": "1999-10-24", "rookie_season": 2021, "height": 72.0,
         "weight": 197.0}]))
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "CHI", "away_team": "GB", "week": 1,
         "total_line": 44.0, "spread_line": 2.0},
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "D'Andre Swift", "position": "RB", "team": "CHI", "adp": 12.0},
        {"adp_name": "Amon-Ra St. Brown", "position": "RB", "team": "DET", "adp": 5.1}]))
    for name, columns in (
            ("depth_charts", ["gsis_id", "depth_team", "formation", "week", "position"]),
            ("snap_counts", ["player", "team", "season", "offense_pct"]),
            ("espn_adp", ["espn_id", "espn_name", "position", "espn_adp", "espn_ppr_rank"]),
            ("fp_ecr", ["fp_name", "team", "position", "rank_ecr", "rank_ave",
                        "rank_std", "fp_tier"]),
            ("sleeper_ids", ["gsis_id", "espn_id", "sleeper_name", "position", "team"])):
        write_table(conn, name, pd.DataFrame(columns=columns))
    # `write_table` does not stamp `meta`, which is what both caches key on --
    # so without this a second test in the same run is served the first one's
    # board and the first one's profiles.
    board_cache.clear()
    profile_cache.clear()
    yield conn
    conn.close()
    board_cache.clear()
    profile_cache.clear()


def _profile(conn, player_id: str):
    """The payload `/api/players/{id}/profile` would serve for this player,
    fetched the way `api/main.py` fetches it."""
    from scoring import league
    from scoring.profile_cache import cached_profile
    return cached_profile(conn, player_id, None, league.load(conn))


def test_the_profile_sections_print_the_app_s_own_numbers(corpus, profiled):
    """THE WHOLE POINT OF READING ONE PAYLOAD. Every figure in these
    sections has to be the one `/api/players/{id}/profile` serves the draft
    room -- not a second calculation over the same tables that agrees today
    and drifts the first time either is fixed. So the assertions below are
    against the payload itself, not against literals."""
    payload = _profile(profiled, "star")
    assert payload is not None, "the fixture cannot build a profile at all"
    body = html.unescape(_client(profiled).get("/adp/dandre-swift").text)

    for heading in ("How he grades", "Every season he has played",
                    "by week", "What his points are made of"):
        assert heading in body, heading

    # The meters, off the header the board hands the popup.
    header = payload["header"]
    assert f"{header['career_games_pg']:.1f}" in body
    assert f"{header['consistency_cv']:.2f}" in body
    # The season table: the average, the finish, and the pool the steadiness
    # rank is out of -- three different shapes of number from three keys.
    season = payload["seasons"][0]
    assert f"{season['ppg']:.1f}" in body
    assert f"RB{season['pos_finish']}" in body
    assert f"{season['cv_rank']} of {season['cv_rank_n']}" in body
    # The week chart's own axis is the game log's opponents.
    assert payload["game_log"][0]["opponent"] in body
    # And the market row's positional place, which only the payload has.
    assert f"RB{payload['header']['market_pos']['board']}" in body


def test_the_season_and_week_figures_are_the_league_s_points_not_ppr(corpus, profiled):
    """`cached_profile` is handed `league.load(conn)`, the same settings the
    room's board is priced under, so a half-PPR league's page cannot print
    PPR points beside a board row priced correctly. The check is that the
    page's average IS the payload's -- and the payload's is the league's,
    which tests/test_profile.py owns."""
    payload = _profile(profiled, "star")
    body = html.unescape(_client(profiled).get("/adp/dandre-swift").text)
    assert f"{payload['seasons'][0]['ppg']:.1f} a game" in body


def test_a_profile_that_will_not_build_costs_the_sections_not_the_page(
        corpus, profiled, monkeypatch):
    """THE PAGE IS NOT THE PROFILE. Everything above these sections comes out
    of the corpus and needs nothing from the universal database; the profile
    is an enrichment on top of it, exactly like `scoring/adp_facts.attach`.
    A universal database that cannot answer must cost the enrichment, never
    the page -- and never a 500 on the 203 pages a crawler is walking."""
    from scoring import profile_cache

    def boom(*args, **kwargs):
        raise RuntimeError("no profile today")

    monkeypatch.setattr(profile_cache, "cached_profile", boom)
    monkeypatch.setattr(seo, "_profile_failed", False)
    seo.clear_pages()
    res = _client(profiled).get("/adp/dandre-swift")
    assert res.status_code == 200
    body = html.unescape(res.text)
    # The corpus's own page, whole.
    assert "D'Andre Swift ADP" in body and "Where the room takes him" in body
    # And not one section that would have needed the payload.
    for heading in ("How he grades", "Every season he has played",
                    "What his points are made of", "The team around him"):
        assert heading not in body, heading


def test_the_profile_failure_is_reported_once_not_per_page(corpus, profiled,
                                                           monkeypatch, capsys):
    """A profile failing on every page of a crawl is one fact about the
    database, not two hundred."""
    from scoring import profile_cache

    def boom(*args, **kwargs):
        raise RuntimeError("no profile today")

    monkeypatch.setattr(profile_cache, "cached_profile", boom)
    monkeypatch.setattr(seo, "_profile_failed", False)
    seo.clear_pages()
    capsys.readouterr()
    client = _client(profiled)
    client.get("/adp/dandre-swift")
    client.get("/adp/amon-ra-st-brown")
    said = [line for line in capsys.readouterr().out.splitlines()
            if "profile enrichment unavailable" in line]
    assert len(said) == 1, said


def test_a_defense_gets_a_page_and_whatever_its_profile_has(corpus, board,
                                                            monkeypatch):
    """A defense has no weekly rows, no o-line, no news and no comparable
    seasons, and `cached_profile` answers for it with almost nothing or with
    nothing at all. Either way the page is a page: the corpus knows exactly
    where a defense goes, which is what somebody searching for one wants."""
    from scoring import board_cache
    monkeypatch.setattr(board_cache, "cached_build_board",
                        lambda conn, *a, **k: _board_frame())
    seo.clear_pages()
    res = _client(board).get("/adp/seattle-defense")
    assert res.status_code == 200
    body = html.unescape(res.text)
    assert "Seattle Defense ADP" in body
    # A DEFENSE IS NOT A HE, in the new sections as much as the old ones.
    assert "Where the room takes it" in body
    assert "Every season he has played" not in body


def test_a_player_page_stays_inside_its_size_budget(corpus, profiled):
    """WHAT A PAGE COSTS TO SEND. These pages are read by a crawler working
    through a 228-URL sitemap and by a person on a phone; the profile
    sections roughly doubled one, and the ceiling is what stops the next
    section being added without anybody measuring.

    Measured against the real corpus and the real universal database, the
    heaviest player page is 56 KB (a tight end: five cards of history, a
    full game log and eight news items). The fixture's pages are far
    smaller, so this is a guard rail rather than a measurement -- see the
    task report for the real figures."""
    client = _client(profiled)
    for player in seo.adp_data(profiled)["players"]:
        body = client.get(f"/adp/{player['slug']}").text
        assert len(body.encode()) <= seo.PLAYER_PAGE_MAX_BYTES, (
            f"{player['slug']} is {len(body.encode()) / 1024:.0f} KB")


def test_the_warm_pass_renders_the_first_player_pages_itself(corpus, board,
                                                             monkeypatch):
    """A crawler must not be the one who pays for a cold profile.

    `cached_profile` is ~250 ms on a first look at a player, so warming the
    aggregate and leaving the pages cold would only move the stall from
    /adp onto whichever player page was opened first. The loop renders the
    first `PROFILE_WARM` of them on its own thread instead.
    """
    monkeypatch.setattr(seo, "WARM_ON_REGISTER", True)
    monkeypatch.setattr(seo, "PROFILE_WARM", 2)
    # Far enough away that the loop's second pass cannot run inside this
    # test and reach a connection the fixture has since closed.
    monkeypatch.setattr(seo, "WARM_SECONDS", 3600.0)
    seo.clear_pages()
    app = FastAPI()
    seo.register_seo_routes(app, conn=board)

    deadline = time.time() + 10
    while time.time() < deadline and ("player", "dandre-swift") not in seo._pages:
        time.sleep(0.05)
    warmed = {key[1] for key in seo._pages if key[0] == "player"}
    assert warmed == {"dandre-swift", "amon-ra-st-brown"}, warmed


def test_nothing_on_a_player_page_can_push_it_sideways():
    """MOBILE OVERFLOW IS A WHOLE-PAGE FAILURE, not a wide element: a table
    that will not fit drags the document's scroll width with it and every
    heading on the page moves under the reader's thumb.

    Two scrollports, because there are two places something wide sits.
    `.scroll` is for a section at the page's own width and breaks out
    through the gutters to use them; `.xscroll` is for something wide inside
    a card or a figure, where those negative margins would put a table
    through the side of the box holding it.
    """
    css = (Path(seo.TEMPLATES) / "base.html").read_text(encoding="utf-8")
    assert ".scroll{overflow-x:auto" in css
    assert ".xscroll{overflow-x:auto" in css
    page = (Path(seo.TEMPLATES) / "adp_player.html").read_text(encoding="utf-8")
    # Every wide thing the profile sections add is inside one of the two.
    assert '<div class="scroll">\n<table class="sched">' in page
    assert '<div class="xscroll">\n<svg viewBox="0 0 720 128"' in page


# ---------------------------------------------------------------------------
# The pages name the archive's real scoring, not the word PPR by habit.
# ---------------------------------------------------------------------------


@pytest.fixture
def standard_corpus(corpus, monkeypatch):
    """The same ten drafts, scored without receptions.

    Written onto the corpus the other tests use rather than built again: the
    only thing under test here is what the pages CALL the shape, and every
    figure on them should be identical."""
    conn = dl.corpus_conn(str(corpus))
    try:
        conn.execute("UPDATE draft_log SET scoring_json = ?",
                     ['{"receptions": 0.0}'])
    finally:
        conn.close()
    market._CACHE.clear()
    return corpus


def test_the_pages_say_the_archives_real_format(standard_corpus, board):
    """The farm records standard-scoring rooms now, so the most-recorded
    shape can be a standard one -- and a page that printed PPR over it would
    be attributing its numbers to a game nobody in it played."""
    body = _client(board).get("/adp").text

    assert "<title>ESPN Mock Draft ADP 2026 (4-team standard)" in body
    assert "(4-team standard, 6 rounds)" in body
    assert "Why 4-team standard?" in html.unescape(body)
    assert "4-team PPR" not in body


def test_the_ppr_wording_is_still_there_when_the_archive_is_ppr(corpus, board):
    """The other half of the same claim: nothing about the default changed."""
    body = _client(board).get("/adp").text

    assert "<title>ESPN Mock Draft ADP 2026 (4-team PPR)" in body
    assert "4-team standard" not in body
