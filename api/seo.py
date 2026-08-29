"""The pages a search engine reads.

The SPA is one document with a title, and every route in it is drawn by
JavaScript from live data. That is fine for a reader and nearly invisible
to a crawler: nothing on the site says, in text it can index, what a
player's draft position is or what this product does.

What the site has that nothing else does is the corpus -- hundreds of
real ESPN mock drafts, recorded pick by pick and growing while the farm
runs. These routes turn it into plain HTML: one page per player, per round
and per position, plus an index and a sitemap. No JavaScript on any of
them. A crawler gets the finished page, and so does a reader on a slow
connection.

WHAT IS PUBLIC AND WHAT IS NOT. These pages carry corpus AGGREGATES: where
a player goes, how often, in which round. The per-seat and per-turn
analysis on /archive stays behind an account, and nothing here reads a
session or a cookie.

REGISTERED BEFORE THE SPA. `api/static.py`'s fallback matches every path,
so `register_seo_routes` has to run before `register_spa` or these pages
would be served the index document instead.
"""
from __future__ import annotations

import math
import os
import re
import threading
import time
import unicodedata
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median, median_low

from fastapi import HTTPException

from api import http_cache
from api import market
from pipeline.db import read_table
from scoring.headshot import thumb

SITE = "https://espnfantasydraft.com"
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

# A player taken in fewer drafts than this share has no page. "Taken in 1
# of 663 drafts" is not a draft position, it is one room's mistake, and a
# site of such pages is thin content that costs the pages worth having.
MIN_SHARE = 0.02

TEMPLATES = Path(__file__).resolve().parent / "templates"

# What a search result actually shows. Google renders about 600 pixels of
# title and 920 of description, which is roughly this many characters at the
# sizes it uses; past them the tail is replaced by an ellipsis, and a title
# whose distinguishing half is in the tail is a title that reads the same as
# the other two hundred. The brand suffix is what gives way -- see `_title`.
TITLE_MAX = 60
DESC_MAX = 160
BRAND = " – ESPN Draft Assist"

# Abbreviation to the name a person says out loud, for the one place the
# abbreviation is not good enough: `Person.affiliation.name` in the
# structured data, which is read by machines that have never seen a depth
# chart. Relocations map to the current name; an abbreviation not in here
# (an expansion team, a feed that says WSH) falls back to itself, which is
# what the visible pages print anyway.
TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "JAC": "Jacksonville Jaguars", "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders",
    "OAK": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "SD": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams", "STL": "Los Angeles Rams",
    "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants", "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders", "WSH": "Washington Commanders",
}


def _title(core: str) -> str:
    """`core`, with the brand on the end when the two of them still fit.

    THE BRAND IS THE PART WORTH LOSING. Every page here wants the same three
    things in its title -- who or what it is about, that it is ESPN mock
    draft ADP, and the year -- and on a long name that is already most of
    sixty characters. A title cut off mid-suffix looks broken; a title
    without the suffix reads fine.
    """
    return core + BRAND if len(core) + len(BRAND) <= TITLE_MAX else core


def _clip(text: str, extra: str) -> str:
    """`text`, plus `extra` if the two of them fit a search result."""
    return text + extra if len(text) + len(extra) <= DESC_MAX else text


def _signed(value, digits: int = 0) -> str:
    """A signed number that never says "-0".

    `'%+.0f' % -0.4` is "-0", which reads as a negative quantity of nothing
    and is the sort of thing a reader notices instead of the number beside
    it. Anything that rounds to zero IS zero, and prints without a sign.
    """
    rounded = round(float(value), digits)
    if rounded == 0:
        return f"{0:.{digits}f}"
    return f"{rounded:+.{digits}f}"

# Whether registering the routes also warms the ADP cache. On in a server,
# and worth an off switch for the one caller that does not want a
# background build the moment an app object exists: the test suite, which
# calls `register_seo_routes` dozens of times a run and has no interest in
# a daemon thread opening a corpus behind every one of them. See
# `api/demo.py`'s `DEMO_WARM` for the same switch on the same principle.
WARM_ON_REGISTER = os.environ.get("SEO_WARM", "1") != "0"

# How often the warm thread rebuilds the ADP aggregate.
#
# THE NUMBER THAT MATTERS IS `market.CACHE_SECONDS`, WHICH THIS IS UNDER.
# `build_adp` costs 0.9-2.8s against the real corpus and `adp_data` holds
# its answer for ten minutes; warming it once at boot therefore fixed the
# first ten minutes and nothing after that. The measured symptom was a 1.9s
# TTFB on `/adp` roughly every other five-minute edge window -- whichever
# crawler or reader happened to arrive after the entry lapsed paid to
# rebuild it. Rebuilding on a timer that is comfortably inside the entry's
# life means the entry never lapses while anyone is looking, and the couple
# of seconds is spent on a thread nobody is waiting on.
WARM_SECONDS = 240.0

# The `market._cached` key the aggregate lives under. Named because the warm
# thread has to be able to retire it (`market.evict`) rather than be handed
# the copy it is trying to replace.
ADP_KEY = "seo-adp"

# How hard `build_adp` tries to get the board, how long it waits between
# tries, and how long it will serve the last board it saw. See
# `_board_facts`: the failure this defends against is another thread
# part-way through building the same board on the same connection, which
# clears in about a second -- so one retry a second later is the whole of
# what waiting can buy, and a third attempt is two seconds of a six-second
# budget spent on a board that is not coming.
#
# THE MEMORY EXPIRES. Bye weeks, tiers and the consensus come off the board,
# and a day is the point past which serving them is worse than serving
# nothing: the pages fall back to blanks, which is what they print for every
# player the crosswalks miss anyway.
BOARD_ATTEMPTS = 2
BOARD_RETRY_SECONDS = 1.0
BOARD_MEMORY_SECONDS = 24 * 60 * 60.0

# How many rendered pages to keep. The sitemap is 228 URLs and a crawler
# works through it in a burst, so this holds a whole crawl and then some.
PAGE_CACHE_MAX = 400


def slug(name: str) -> str:
    """`D'Andre Swift` -> `dandre-swift`; `Seattle Seahawks D/ST` ->
    `seattle-seahawks-dst`. ASCII, lowercase, hyphens; apostrophes, dots and
    slashes vanish rather than becoming separators, so the abbreviation and
    the contraction read as one word the way they are said."""
    folded = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    folded = re.sub(r"['./]", "", folded.lower())
    folded = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    return folded or "player"


def _percentile(sorted_picks: list, q: float) -> int:
    return sorted_picks[int(round(q * (len(sorted_picks) - 1)))]


def _teams_by_player(conn) -> dict:
    """gsis_id -> team, from the newest depth chart row per player."""
    if conn is None:
        return {}
    try:
        dc = read_table(conn, "depth_charts")
        if dc.empty or not {"gsis_id", "team", "dt"}.issubset(dc.columns):
            return {}
        latest = dc.dropna(subset=["gsis_id"]).sort_values("dt").drop_duplicates("gsis_id", keep="last")
        return {str(r.gsis_id): str(r.team) for r in latest.itertuples() if r.team == r.team and r.team}
    except Exception:      # noqa: BLE001 -- a board that cannot be read costs teams,
        # not the page: everything after `read_table` is pandas work on data
        # that came from the board, and a board with a surprising shape
        # should cost teams the same way a board that will not open does.
        return {}


def _espn_adp_by_player(conn, ids: dict) -> dict:
    """gsis_id -> ESPN's published ADP. Crosswalk first, then name and
    position, the two-step every ESPN join in this codebase makes.
    `ids` maps player_id -> (name, position)."""
    if conn is None:
        return {}
    try:
        espn = read_table(conn, "espn_adp")
        sleeper = read_table(conn, "sleeper_ids")
        if espn.empty or "espn_adp" not in espn.columns:
            return {}
        from scoring.profile import _norm_name
        out: dict = {}
        if not sleeper.empty and {"gsis_id", "espn_id"}.issubset(sleeper.columns):
            xwalk = sleeper.dropna(subset=["gsis_id", "espn_id"]).drop_duplicates("espn_id")
            joined = espn.merge(xwalk[["gsis_id", "espn_id"]], on="espn_id")
            for r in joined.itertuples():
                if r.espn_adp == r.espn_adp:
                    out[str(r.gsis_id)] = float(r.espn_adp)
        by_name = {}
        if {"espn_name", "position"}.issubset(espn.columns):
            for r in espn.itertuples():
                if r.espn_adp == r.espn_adp:
                    by_name.setdefault((_norm_name(str(r.espn_name)), str(r.position)), float(r.espn_adp))
        for pid, (name, pos) in ids.items():
            if pid not in out:
                hit = by_name.get((_norm_name(name), pos))
                if hit is not None:
                    out[pid] = hit
        return out
    except Exception:      # noqa: BLE001 -- a board that cannot be read costs
        # ESPN's own ADP, not the page; the merge and the two itertuples
        # loops below are as much "reading the board" as `read_table` is.
        return {}


def _team_fallback_from_espn(conn, ids: dict | None = None) -> dict:
    """gsis_id -> team from `espn_adp`, for players the depth charts do not
    carry (a rookie in August). Crosswalk first, then name and position --
    the same two steps `_espn_adp_by_player` takes, because the crosswalk
    misses exactly the players the depth charts miss: a rookie has no
    `sleeper_ids` row either, and printing an em dash for the team of a
    player ESPN's own board names is a gap for no reason.

    `ids` maps player_id -> (name, position), and is what the second step
    needs; without it only the crosswalk runs."""
    if conn is None:
        return {}
    try:
        espn = read_table(conn, "espn_adp")
        if espn.empty or "team" not in espn.columns:
            return {}
        sleeper = read_table(conn, "sleeper_ids")
        out: dict = {}
        if not sleeper.empty and {"gsis_id", "espn_id"}.issubset(sleeper.columns):
            joined = espn.merge(sleeper.dropna(subset=["gsis_id", "espn_id"])[["gsis_id", "espn_id"]],
                                on="espn_id")
            out = {str(r.gsis_id): str(r.team) for r in joined.itertuples()
                   if r.team == r.team and r.team}
        if ids and {"espn_name", "position"}.issubset(espn.columns):
            from scoring.profile import _norm_name
            by_name = {}
            for r in espn.itertuples():
                if r.team == r.team and r.team:
                    by_name.setdefault((_norm_name(str(r.espn_name)), str(r.position)),
                                       str(r.team))
            for pid, (name, pos) in ids.items():
                if pid not in out:
                    hit = by_name.get((_norm_name(name), pos))
                    if hit is not None:
                        out[pid] = hit
        return out
    except Exception:      # noqa: BLE001 -- a board that cannot be read costs
        # a fallback team, not the page. The name step below `read_table` is
        # as much "reading the board" as the read itself.
        return {}


# The picks the "still there at" strip reports, and the depth its curve is
# drawn to. Eight readings rather than a table of a hundred: the first is one
# turn away in an 8-team room, the last is round 12, and between them they
# cover every question a drafter actually asks about a name ("can I wait a
# round?"). The curve behind them is the continuous version.
STILL_THERE_PICKS = (8, 16, 24, 36, 48, 60, 72, 96)
CURVE_PICKS = 100

# How far apart the mock rooms and ESPN's own ADP have to be before a player
# is called a riser or a faller. Eight picks is a full turn in an 8-team room:
# below it the two lists are saying the same thing with rounding, above it a
# drafter reading ESPN's list would genuinely miss him.
MOVE_PICKS = 8

# How many of them either list holds.
MOVERS = 10

# How often the rooms have to take a player before "he is a riser" is about
# him at all.
#
# WITHOUT THIS THE LIST IS 100% ARTEFACT. Measured on the real corpus, every
# one of the thirty-two names that cleared MOVE_PICKS was taken in under a
# tenth of the drafts he was on the board for -- David Njoku "+61" is twenty
# drafts out of eight hundred and fifty-four. Twenty rooms reaching for a
# tight end is twenty rooms, not a market: the other 834 left him there, and
# the ADP those twenty picks average to is not a draft position anybody would
# plan around. A quarter is the floor at which the mean is describing the
# room rather than the tail of it -- and at that floor the riser list on the
# real board is short, which is the honest answer.
MOVER_MIN_SHARE = 0.25

# How much of a player's own draft history a round has to hold before that
# round's page names him. Five per cent of his picks -- one draft in twenty
# that took him took him here. Below it the row was saying nothing: the round
# pages listed anybody whose 10th-to-90th percentile crossed the round, which
# for a player with a wide range is fourteen of the sixteen pages, printing
# "0%" beside his name on most of them.
ROUND_MIN_SHARE = 0.05

# Where ESPN's published ADP stops being a draft position and becomes a
# sentinel. ESPN publishes five hundred players and 329 of them sit between
# 165 and 170 -- 106 at exactly 169.98 -- because a player its lobby never
# drafts still needs a number, and that pile IS the number. Ordering on it
# spreads ten players across eleven implied picks out of 0.73 ADP points,
# which is not a ranking, and six of the ten risers this page used to print
# came out of that band. Read as "unranked", and ESPN's own expert rank
# stands in as far as it is worth reading -- three hundred is two and a half
# times deeper than an 8-team room reaches, and past it the rank is as
# uninformative as the ADP.
ESPN_UNDRAFTED = 165.0
ESPN_RANK_DEPTH = 300

# Every field a player MAY have, and what "we could not answer that" looks
# like. Set on every player before a page is rendered, so a template can ask
# for any of them without guarding, and so the shape of a player is written
# down in one place rather than inferred from nine sources that each fill in
# a few of it. A blank here is the answer for anybody the crosswalks miss --
# see scoring/adp_facts.py on why it is never somebody else's number.
BLANKS = {
    "bye": None, "tier": None, "fp_tier": None, "consensus": None,
    "espn_rank": None, "espn_pick": None, "espn_order": None, "vs_espn": None,
    "espn_undrafted": False,
    "proj_points": None, "rookie": False, "sources": [],
    "age": None, "rookie_season": None, "height": None, "weight": None,
    "seasons": [], "spark": None, "auction": None, "cheat_rank": None,
    "last_season": None, "projection": None, "futures": [],
    "news": [], "status": None, "still_there": [], "curve": None,
}


def _empty() -> dict:
    return {"players": [], "by_slug": {}, "drafts": 0, "teams": 0, "rounds": 0,
            "updated": None, "stamp": ("empty",), "risers": [], "fallers": [],
            "runs": [], "round_mix": {}, "at_pick": {}, "seconds": None}


def _stamp(conn, drafts: int, teams: int, rounds: int, updated) -> tuple:
    """What this snapshot of the ADP aggregate IS, for the page cache.

    Two halves, because these pages are built out of two databases. The
    CORPUS half is the draft count, the shape, and the newest recorded
    draft: nothing in `build_adp` can move without one of those moving,
    since every figure on every page is a statistic over exactly those
    drafts. The UNIVERSAL half is `board_cache`'s own identity key -- the
    `meta` fingerprint -- because names, teams and ESPN's published ADP come
    off `conn` and change when the instance refreshes itself.

    The point of a stamp rather than a timer is that the warm thread rebuilds
    the aggregate every few minutes and usually finds the same drafts in it.
    An identical stamp means the 227 rendered pages hanging off the last one
    are still correct, and none of them has to be built again.

    Unreadable identity is its own answer, not a raised exception: the pages
    are already written to lose the board rather than the page, and a stamp
    that cannot be computed simply means nothing is cached under it.
    """
    try:
        from scoring.board_cache import _identity_key
        universal = _identity_key(conn) if conn is not None else ()
    except Exception:      # noqa: BLE001 -- see the docstring
        universal = ()
    return (int(drafts), int(teams), int(rounds), str(updated), universal)


_last_board_facts: dict = {}
_last_board_at = 0.0
_board_lock = threading.Lock()


def _board_facts(conn) -> dict:
    """The board's row per player: fetched once per build, retried, remembered.

    A PAGE MUST NOT DISAPPEAR BECAUSE TWO THREADS WANTED THE BOARD AT ONCE.
    Forty of the two hundred published players -- every D/ST and every rookie
    -- exist only on the board: `scoring/board._add_adp_only_players` invents
    their ids and their names, and `players` has no row to fall back on. When
    `cached_build_board` raised (a keep-warm build and a request landing on
    one DuckDB connection at the same moment, which is what a cold start
    looks like), those forty lost their names, `build_adp` drops a player it
    cannot name, and the sitemap went from 228 URLs to 187 until the next
    keep-warm pass. A crawler that walks a sitemap missing a fifth of its
    pages does not come back for them.

    So: a second attempt a second later, which is longer than the race it is
    waiting out, and if that does not come either, the last board this
    process saw. Its bye weeks and tiers are minutes old at worst, and a
    stale tier beside a fresh ADP is a smaller lie than a page that is not
    there.

    FOR A DAY, AND NOT LONGER. Past that the trade stops being worth making:
    a bye week from yesterday's board is a wrong answer rather than an old
    one, and these pages already know how to print a blank -- every player
    the crosswalks miss gets one. After a day the memory is dropped and the
    fields go with it. The names are the exception this whole mechanism
    exists for, and they go too: a page that says nothing about a player is
    better than a page that says something untrue about him.
    """
    global _last_board_facts, _last_board_at
    if conn is None:
        return {}
    from scoring import adp_facts
    for attempt in range(1, BOARD_ATTEMPTS + 1):
        try:
            rows = adp_facts.board_rows(conn)
        except Exception:      # noqa: BLE001 -- see the docstring; the next
            # attempt is the answer, and the remembered board after that.
            if attempt < BOARD_ATTEMPTS:
                time.sleep(BOARD_RETRY_SECONDS)
            continue
        if rows:
            with _board_lock:
                _last_board_facts = rows
                _last_board_at = time.time()
        return rows
    with _board_lock:
        if time.time() - _last_board_at > BOARD_MEMORY_SECONDS:
            _last_board_facts = {}
            return {}
        return dict(_last_board_facts)


def _names_from_board(board: dict, missing: set) -> dict:
    """The names `players` does not carry, off the board already in hand.

    `market._board_names` does the same job by building its own board, which
    is a second call to `cached_build_board` and a second chance to lose the
    defenses on a cold process. This one reads the frame `_board_facts`
    already fought for.
    """
    out = {}
    for pid in missing:
        row = board.get(pid)
        if row is None:
            continue
        name = getattr(row, "name", None)
        shot = getattr(row, "headshot", None)
        # Already sized: `headshot` is a board column and the board sizes it
        # at build (scoring/board.py).
        out[pid] = {"name": None if name is None or name != name else str(name),
                    "headshot": None if shot is None or shot != shot else str(shot)}
    return out


def build_adp(conn) -> dict:
    """Every player's draft position out of the corpus, in one pass.

    Every pick counts -- the autodrafter's and the farm's own seat's
    included -- because the question is where a player GOES, not what
    people think. The archive filters those out for its behavioural
    questions; ADP is the other kind of question.

    A player whose name never resolves gets no page: not an h1 reading
    "adp_kansas_city_defense", just nobody at that slot. He is still read
    out of the corpus and counted toward `total`, so a cold board costs
    pages, never the drafts count.
    """
    corpus = market._corpus()
    try:
        try:
            teams, rounds = market._shape(corpus)
        except HTTPException:
            return _empty()
        total, latest = corpus.execute(
            "SELECT count(*), max(recorded_at) FROM draft_log"
            " WHERE teams = ? AND rounds = ?", [teams, rounds]).fetchone()
        rows = corpus.execute(
            "SELECT pk.player_id, pk.position, pk.pick_no"
            " FROM draft_log_pick pk JOIN draft_log d USING (draft_id)"
            " WHERE d.teams = ? AND d.rounds = ? AND pk.player_id IS NOT NULL"
            " AND pk.pick_no BETWEEN 1 AND ?",
            [teams, rounds, teams * rounds]).fetchall()
        pooled = _pooled(corpus, teams, rounds)
        clock, corpus_clock = _clock(corpus, teams, rounds)
        runs = _runs(corpus, teams, rounds)
    finally:
        corpus.close()
    total = int(total or 0)
    if total == 0:
        return _empty()

    picks: dict = {}
    positions: dict = {}
    # The position taken at every pick of every draft, so a round page can say
    # what its own eight picks are made of without a second query.
    mix: Counter = Counter()
    for pid, pos, pick_no in rows:
        picks.setdefault(str(pid), []).append(int(pick_no))
        position = str(pos or "").upper()
        positions.setdefault(str(pid), position)
        mix[((int(pick_no) - 1) // teams + 1, position)] += 1

    # ONE FETCH OF THE BOARD FOR THE WHOLE BUILD. The names below and every
    # board-derived field in `adp_facts.attach` come off this one frame.
    board = _board_facts(conn)
    names = market._names(conn)
    missing = {pid for pid in picks if pid not in names}
    if missing:
        names.update(_names_from_board(board, missing))
    teams_by = _teams_by_player(conn)

    players = []
    for pid, taken in picks.items():
        share = len(taken) / total
        if share < MIN_SHARE:
            continue
        entry = names.get(pid)
        name = entry.get("name") if entry else None
        if not name:
            continue
        headshot = entry.get("headshot")
        if not (isinstance(headshot, str) and headshot.startswith("https://")):
            headshot = None
        ordered = sorted(taken)
        hist = [0] * (teams * rounds)
        for p in ordered:
            hist[p - 1] += 1
        # THE HONEST DENOMINATOR, where the corpus has one. `draft_log_pool`
        # records who was on the board in each draft, and a player ESPN added
        # to its board in the middle of the summer was not available to be
        # taken in the drafts before that. Counting him against every draft
        # ever recorded understates how reliably rooms take him. Falls back
        # to the draft count for a corpus with no pool rows at all.
        of = pooled.get(pid) or total
        usual_round, usual_share = _usual_round(ordered, teams, rounds, of)
        players.append({
            "player_id": pid,
            "name": name,
            "headshot": headshot,
            "position": positions.get(pid) or "",
            "team": teams_by.get(pid),
            "adp": round(mean(ordered), 1),
            "median": float(median(ordered)),
            "p10": _percentile(ordered, 0.10),
            "p90": _percentile(ordered, 0.90),
            "low": ordered[0],
            "high": ordered[-1],
            "taken": len(ordered),
            "share": round(share, 3),
            "of": of,
            "of_share": round(min(len(ordered) / of, 1.0), 3),
            "round_low": (ordered[0] - 1) // teams + 1,
            "round_p10": (_percentile(ordered, 0.10) - 1) // teams + 1,
            "round_p90": (_percentile(ordered, 0.90) - 1) // teams + 1,
            "usual_round": usual_round,
            "usual_share": usual_share,
            "hist": hist,
            "seconds": clock.get(pid),
            "hist_peak": max(hist) or 1,
        })

    players.sort(key=lambda p: (p["adp"], -p["taken"], p["player_id"]))
    ids = {p["player_id"]: (p["name"], p["position"]) for p in players}
    espn = _espn_adp_by_player(conn, ids)
    espn_team = _team_fallback_from_espn(conn, ids)
    pos_seen: Counter = Counter()
    for i, p in enumerate(players, start=1):
        p["rank"] = i
        pos_seen[p["position"]] += 1
        p["pos_rank"] = pos_seen[p["position"]]
        p["espn_adp"] = espn.get(p["player_id"])
        # The sentinel, named on the page rather than printed as a figure:
        # "ESPN ADP 169.6" reads like a draft position and is not one.
        p["espn_undrafted"] = (p["espn_adp"] is not None
                               and p["espn_adp"] >= ESPN_UNDRAFTED)
        if not p["team"]:
            p["team"] = espn_team.get(p["player_id"])

    # Everything the universal database knows about these two hundred names:
    # bye, tier, the five consensus sources, the season history, last season,
    # this season's projection, the sportsbook, the injury report, the news.
    # One pass over the whole list -- see scoring/adp_facts.py.
    try:
        from scoring import adp_facts
        adp_facts.attach(conn, players, board=board)
    except Exception:      # noqa: BLE001 -- the corpus's own figures are the
        # headline of every one of these pages and none of them needs this.
        # A universal database that cannot be read costs the enrichments.
        pass
    for p in players:
        if p.get("seasons"):
            p["spark"] = _spark(p["seasons"])
    _espn_gap(players)
    _availability(players, teams)
    for p in players:
        for field, blank in BLANKS.items():
            p.setdefault(field, blank)
    risers, fallers = _movers(players)

    # Slugs: a collision takes -2, -3 in player_id order, so two players
    # with one name keep the same addresses from one snapshot to the next.
    by_base: dict = {}
    for p in sorted(players, key=lambda p: p["player_id"]):
        by_base.setdefault(slug(p["name"]), []).append(p)
    by_slug = {}
    for base, group in by_base.items():
        for n, p in enumerate(group, start=1):
            p["slug"] = base if n == 1 else f"{base}-{n}"
            by_slug[p["slug"]] = p

    updated = latest.date() if isinstance(latest, datetime) else None
    return {"players": players, "by_slug": by_slug, "drafts": total,
            "teams": teams, "rounds": rounds, "updated": updated,
            "stamp": _stamp(conn, total, teams, rounds, updated),
            "risers": risers, "fallers": fallers, "runs": runs,
            "round_mix": _round_mix(mix, rounds),
            "at_pick": _at_pick(players, teams, rounds),
            "seconds": corpus_clock}


def _usual_round(ordered: list, teams: int, rounds: int, of: int) -> tuple:
    """The round to call his usual one, and the share of his picks in it.

    THE MODE OVER A FLAT DISTRIBUTION IS NOISE. `Counter(...).most_common(1)`
    was what this used to be, and on the real board it was wrong on 61 pages
    of 203: a player taken twenty times over a hundred and twenty picks has
    no modal round worth the name, and whichever round happened to hold three
    of the twenty won -- David Njoku's page said "Usual round 11" off three
    picks of eight hundred and fifty, and round 11's page then listed him as
    a player who "usually" goes there in 0% of drafts.

    So the answer is the round holding the MEDIAN pick, which is where half
    of them are on either side of, and a mode only overrules it when it is
    actually a peak: at least twice the share a uniform spread would leave in
    any one round. AGAINST THE DRAFTS HE WAS ON THE BOARD FOR, not against
    his own picks -- three picks of twenty is 2.4 times uniform and still
    three picks, which is how Njoku got round 11 in the first place. Where a
    player's picks really are concentrated, the modal round is the median's
    round anyway and this test never has to fire. Ties among modal rounds go
    to the median's round, or to whichever tied round is nearest it --
    fourteen of them resolved toward the earliest pick before, which is a
    bias, not a tie-break.
    """
    per_round = Counter((pick - 1) // teams + 1 for pick in ordered)
    middle = (median_low(ordered) - 1) // teams + 1
    top = max(per_round.values())
    if rounds and of and top / of >= 2.0 / rounds:
        tied = [rnd for rnd, count in per_round.items() if count == top]
        chosen = (middle if middle in tied
                  else min(tied, key=lambda rnd: (abs(rnd - middle), rnd)))
    else:
        chosen = middle
    return chosen, round(per_round[chosen] / len(ordered), 3)


# ---------------------------------------------------------------------------
# The corpus, past the picks themselves
# ---------------------------------------------------------------------------

def _pooled(corpus, teams: int, rounds: int) -> dict:
    """player_id -> how many of these drafts he was on the board for.

    An older corpus, or a fixture that only ever wrote picks, has no pool
    rows; an empty answer means every share falls back to the draft count,
    which is what these pages said before this existed.
    """
    try:
        rows = corpus.execute(
            "SELECT pl.player_id, count(DISTINCT pl.draft_id)"
            " FROM draft_log_pool pl JOIN draft_log d USING (draft_id)"
            " WHERE d.teams = ? AND d.rounds = ? AND pl.player_id IS NOT NULL"
            " GROUP BY 1", [teams, rounds]).fetchall()
    except Exception:      # noqa: BLE001 -- no pool table on this corpus
        return {}
    return {str(pid): int(n) for pid, n in rows if n}


def _clock(corpus, teams: int, rounds: int) -> tuple:
    """How long a room takes over him, and how long it takes over anybody.

    HUMAN PICKS ONLY, unlike everything else on these pages. Where a player
    GOES counts every pick, autodrafts included, because the autodrafter is
    part of the market. How LONG a pick took is a question about a person
    thinking, and an autodrafted seat answers it in a tenth of a second --
    the median over every pick in the corpus is 1.4 seconds, which is not a
    fact about anybody's deliberation.
    """
    where = ("d.teams = ? AND d.rounds = ? AND pk.seconds_to_pick IS NOT NULL"
             " AND COALESCE(pk.autodrafted, FALSE) = FALSE"
             " AND (d.my_slot IS NULL OR pk.slot <> d.my_slot)")
    try:
        rows = corpus.execute(
            "SELECT pk.player_id, median(pk.seconds_to_pick), count(*)"
            " FROM draft_log_pick pk JOIN draft_log d USING (draft_id)"
            f" WHERE {where} GROUP BY 1", [teams, rounds]).fetchall()
        overall = corpus.execute(
            "SELECT median(pk.seconds_to_pick)"
            " FROM draft_log_pick pk JOIN draft_log d USING (draft_id)"
            f" WHERE {where}", [teams, rounds]).fetchone()
    except Exception:      # noqa: BLE001 -- an older corpus without a clock
        return {}, None
    # Ten is enough for a median to be about him rather than about one
    # drafter who walked away from the keyboard.
    per_player = {str(pid): round(float(secs), 1)
                  for pid, secs, n in rows if secs is not None and n >= 10}
    median_all = None if not overall or overall[0] is None else round(float(overall[0]), 1)
    return per_player, median_all


def _runs(corpus, teams: int, rounds: int) -> list:
    """When the first player at each position comes off the board.

    The median over every draft, not the average: the question a reader has
    is "when does the run start", and one room reaching for a kicker in round
    four should not move the answer for the other 853.
    """
    try:
        rows = corpus.execute(
            "WITH firsts AS ("
            "  SELECT pk.draft_id, pk.position, min(pk.pick_no) AS first_pick"
            "  FROM draft_log_pick pk JOIN draft_log d USING (draft_id)"
            "  WHERE d.teams = ? AND d.rounds = ? AND pk.position IS NOT NULL"
            "  GROUP BY 1, 2)"
            " SELECT position, median(first_pick), count(*) FROM firsts"
            " GROUP BY 1 ORDER BY 2", [teams, rounds]).fetchall()
    except Exception:      # noqa: BLE001
        return []
    out = []
    for position, pick, drafts in rows:
        name = str(position or "").upper()
        if name not in POSITIONS or pick is None:
            continue
        at = int(round(float(pick)))
        out.append({"position": name, "pick": at, "drafts": int(drafts),
                    "round": (at - 1) // teams + 1})
    return out


def _round_mix(mix: Counter, rounds: int) -> dict:
    """round -> the positions its picks were spent on, commonest first."""
    out: dict = {}
    for n in range(1, rounds + 1):
        counts = [(pos, count) for (rnd, pos), count in mix.items()
                  if rnd == n and pos]
        total = sum(count for _pos, count in counts)
        if not total:
            continue
        counts.sort(key=lambda item: (-item[1], item[0]))
        out[n] = [{"position": pos, "share": round(count / total, 3),
                   "pct": round(count * 100 / total, 1)}
                  for pos, count in counts]
    return out


def _at_pick(players: list, teams: int, rounds: int) -> dict:
    """overall pick -> the three names most often taken there.

    Out of the histograms already computed, not a second query: `hist[p - 1]`
    is exactly "how many drafts took him at pick p", and the top of that
    column across the published players is who a seat at that pick sees.
    """
    out: dict = {}
    for pick in range(1, teams * rounds + 1):
        column = [(p["hist"][pick - 1], p) for p in players if p["hist"][pick - 1]]
        if not column:
            continue
        total = sum(count for count, _p in column)
        column.sort(key=lambda item: (-item[0], item[1]["adp"]))
        out[pick] = [{"name": p["name"], "slug": p["slug"], "position": p["position"],
                      "count": count, "pct": round(count * 100 / total)}
                     for count, p in column[:3]]
    return out


def _espn_gap(players: list) -> None:
    """Where ESPN's board would have taken him, against where the rooms did.

    SUBTRACTING ESPN'S ADP FROM OURS IS THE WRONG SUM, and it was the first
    thing tried. ESPN publishes a draft position measured in ESPN's own
    leagues, over a universe of five hundred players; this corpus is 8-team
    16-round mocks, 128 picks deep. Every kicker in the corpus therefore
    reads as a hundred picks "early" -- an 8-team room has to fill 128 slots
    and ESPN's leagues do not -- and the arithmetic produces 203 risers and
    no fallers, which is a fact about league size and not about a player.

    So the two lists are put on ONE SCALE first. Take the players ESPN ranks,
    order them ESPN's way, and read off this corpus's own ADP at each
    position: if you drafted straight down ESPN's board, its nth name would
    go at the pick this corpus's nth name goes at. That implied pick is
    comparable with his real one, the difference is in picks, and the
    differences sum to zero -- so a riser is a player the rooms genuinely
    move up, not a player two lists happened to count differently.

    ESPN'S PUBLISHED ADP IS THE ORDERING KEY where there is one, because it
    is drafting behaviour and so is the thing being compared; the expert PPR
    rank stands in for the handful it does not reach. NEITHER OF THEM IS
    READ PAST THE POINT WHERE IT STOPS SAYING ANYTHING -- see
    `ESPN_UNDRAFTED`. A player ESPN's board does not rank at all has no
    place on this scale, so he gets no `vs_espn` and his page drops the
    sentence rather than comparing him with a placeholder.
    """
    covered = [p for p in players if _espn_key(p) is not None]
    if len(covered) < 2:
        return
    # `players` is already in this corpus's own draft order, so the picks
    # come out ascending without another sort.
    picks = [p["adp"] for p in covered]
    order = sorted(covered, key=lambda p: (_espn_key(p), p["adp"]))
    for i, p in enumerate(order):
        p["espn_pick"] = picks[i]
        p["espn_order"] = i + 1
        p["vs_espn"] = round(picks[i] - p["adp"], 1)


def _espn_key(p: dict) -> float | None:
    """Where ESPN's board puts him, on whichever of its two numbers still
    means something -- or nothing at all, which is its own answer."""
    adp = p.get("espn_adp")
    if adp is not None and adp < ESPN_UNDRAFTED:
        return float(adp)
    rank = p.get("espn_rank")
    if rank is not None and rank <= ESPN_RANK_DEPTH:
        return float(rank)
    return None


def _movers(players: list) -> tuple:
    """The risers and the fallers, out of the players rooms actually take.

    TWO FILTERS, AND THE SECOND ONE IS THE IMPORTANT ONE. The gap against
    ESPN has to clear MOVE_PICKS -- a full turn in an 8-team room -- and the
    player has to be one this corpus can speak for at all: taken in at least
    MOVER_MIN_SHARE of the drafts he was on the board for. An ADP over twenty
    picks out of eight hundred drafts is the average of the rooms that
    reached, and the rooms that reached are the only rooms in it.
    """
    moved = [p for p in players
             if p.get("vs_espn") is not None
             and (p.get("of_share") or 0) >= MOVER_MIN_SHARE]
    risers = sorted((p for p in moved if p["vs_espn"] >= MOVE_PICKS),
                    key=lambda p: -p["vs_espn"])[:MOVERS]
    fallers = sorted((p for p in moved if p["vs_espn"] <= -MOVE_PICKS),
                     key=lambda p: p["vs_espn"])[:MOVERS]
    return risers, fallers


# ---------------------------------------------------------------------------
# Still there at pick N
# ---------------------------------------------------------------------------

def _availability(players: list, teams: int) -> None:
    """"Will he still be there at pick N", measured, for every player at once.

    `scoring.availability` counts the same corpus these pages are built from
    -- how many of the drafts he was pooled in had taken him by each pick --
    and it is already cached in the process for the live draft room, keyed on
    the corpus file's mtime. So this costs one gather per pick, not a second
    pass over 109k rows.

    `k = 0` throughout: these pages are read before a draft, by somebody who
    has not made a pick yet, so the question is the unconditional one.
    """
    if not players:
        return
    try:
        import numpy as np
        from pipeline import draft_log as dl
        from scoring import availability as av
        # The path at CALL time. `cached_table`'s default was bound when that
        # module was imported, and the corpus these pages read is chosen by
        # `market.dl.CORPUS_PATH`, which a test moves.
        table = av.cached_table(dl.CORPUS_PATH)
        if not table.pooled.size:
            return
        ids = [p["player_id"] for p in players]
        positions = [p["position"] for p in players]
        espn = np.array([p["espn_adp"] if p.get("espn_adp") is not None else np.nan
                         for p in players], dtype=float)
        readings = {n: av.availability_at(table, ids, k=0, n=n, espn_adp=espn,
                                          positions=positions)
                    for n in STILL_THERE_PICKS}
        curve = np.stack([av.availability_at(table, ids, k=0, n=n, espn_adp=espn,
                                             positions=positions)
                          for n in range(1, CURVE_PICKS + 1)])
    except Exception:      # noqa: BLE001 -- a corpus mid-write, or numpy
        # missing on some future slimmer image. The strip disappears; every
        # other figure on the page is unaffected.
        return
    for i, p in enumerate(players):
        p["still_there"] = [{"pick": n, "pct": int(round(float(readings[n][i]) * 100)),
                             "round": (n - 1) // teams + 1}
                            for n in STILL_THERE_PICKS]
        p["curve"] = _curve_path([float(v) for v in curve[:, i]])


def _curve_path(values: list, width: int = 960, height: int = 100) -> str:
    """A step curve as one SVG `path` d attribute.

    A STEP, NOT A LINE, because the underlying thing is a step: he is there
    until a pick takes him, and drawing a diagonal between two picks would
    claim a player is 40% available halfway through pick 23.

    Written out at render time for two hundred players, so it is emitted
    once here and only when the value actually moves -- a curve that is flat
    at 100 for sixty picks is six characters, not sixty.
    """
    if not values:
        return ""
    step = width / max(len(values) - 1, 1)

    def y(v: float) -> float:
        return round(height - v * height, 1)

    parts = [f"M0,{y(values[0])}"]
    last = values[0]
    for i, value in enumerate(values[1:], start=1):
        if abs(value - last) < 0.005 and i != len(values) - 1:
            continue
        parts.append(f"H{round(i * step, 1)}V{y(value)}")
        last = value
    return "".join(parts)


def _spark(seasons: list) -> str | None:
    from scoring.adp_facts import sparkline
    try:
        return sparkline(seasons)
    except Exception:      # noqa: BLE001
        return None


_adp_lock = threading.Lock()


def adp_data(conn) -> dict:
    """`build_adp`, once per CACHE_SECONDS, shared by every page.

    Serialized here, not just inside `market._cached`: that lock only
    guards its own dict, not the `build()` call, so two callers racing a
    cold cache -- the warm thread below and the first real request, or two
    requests that both land before either finishes -- can both end up
    inside `build_adp` at once, running raw queries on connections that are
    not safe to share across threads mid-query. This lock means the second
    caller waits and gets the first caller's answer from cache instead.
    """
    with _adp_lock:
        return market._cached(ADP_KEY, lambda: build_adp(conn))


def rebuild_adp(conn) -> dict:
    """`build_adp` whether or not the cached answer has expired yet, without
    making anybody wait for it.

    What the warm thread calls. `adp_data` would hand back the entry this is
    trying to replace for as long as that entry is alive, so the rebuild has
    to be forced -- but forcing it must not cost a reader anything, and the
    first version of this cost them everything.

    NO LOCK AROUND THE BUILD. Held across `build_adp`, `_adp_lock` turned a
    background refresh into a stall: every /adp route asks `adp_data` for the
    aggregate before it can reach its own page cache, so a reader arriving
    during the four-minutely rebuild queued behind it and waited the full
    1.7s. That is a worse version of the 1.9s this warm thread was written to
    remove, and it happened two and a half times as often. So the work is
    done on this thread with nothing held, and only the finished answer is
    swapped in -- `market.store`, one assignment under that module's own
    lock. A reader racing a rebuild is served the previous answer at once and
    never blocks.

    Safe to build outside `_adp_lock` because of what that lock is for: two
    callers inside `build_adp` at the same moment on ONE connection (see
    `adp_data`). The warm thread was given `conn.cursor()` of its own for
    exactly this reason, so it is not sharing statement state with the
    readers it now runs alongside.
    """
    built = build_adp(conn)
    market.store(ADP_KEY, built)
    return built


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_env = None
_env_lock = threading.Lock()


def _templates():
    global _env
    with _env_lock:
        if _env is None:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
            _env = Environment(loader=FileSystemLoader(str(TEMPLATES)),
                               autoescape=select_autoescape(["html", "xml"]))
            _env.filters["signed"] = _signed
        return _env


def render(name: str, **ctx) -> str:
    ctx.setdefault("site", SITE)
    ctx.setdefault("positions", POSITIONS)
    return _templates().get_template(name).render(**ctx)


_pages: dict = {}
_pages_stamp = None
_pages_lock = threading.Lock()


def rendered(stamp, key, build) -> str:
    """`build()`, at most once per page per snapshot of the corpus.

    These pages are pure functions of the ADP aggregate: same aggregate,
    same bytes, every time. The aggregate is rebuilt on a timer by the warm
    thread and mostly comes back identical (`_stamp` says when it does), so
    holding the finished HTML costs one dict and saves rendering the same
    227 pages for every crawl that walks the sitemap.

    A stamp that does not match the one in hand empties the whole cache
    rather than ageing entries out one at a time: the pages are a set, they
    all describe the same snapshot, and half a crawl reading the old
    snapshot beside half reading the new one is a worse answer than either.
    """
    global _pages_stamp
    with _pages_lock:
        if _pages_stamp != stamp:
            _pages.clear()
            _pages_stamp = stamp
        hit = _pages.get(key)
    if hit is not None:
        return hit
    html = build()
    with _pages_lock:
        # Only if the snapshot is still the one this was built from -- a warm
        # thread may have swapped it while this render was running, and the
        # entry would then be one page describing yesterday among 231
        # describing today.
        if _pages_stamp == stamp and len(_pages) < PAGE_CACHE_MAX:
            _pages[key] = html
    return html


def clear_pages() -> None:
    """Drop every rendered page. For tests, and for anything that changes
    what these pages say without changing the corpus under them."""
    global _pages_stamp
    with _pages_lock:
        _pages.clear()
        _pages_stamp = None


# ---------------------------------------------------------------------------
# The profile the app draws, as static HTML
# ---------------------------------------------------------------------------
#
# ONE PAYLOAD, TWO RENDERERS. Everything in this section is read out of
# `scoring.profile_cache.cached_profile` -- the same object the app serves at
# `/api/players/{id}/profile` and the same one the draft room's popup builds
# its fifteen cards from. A page that ran its own queries would be a second
# answer to every question the popup already answers, and the two would drift
# the first time either was fixed. What this section adds is the DRAWING: the
# popup is React, and a crawler and a reader on a slow line get none of it.
#
# THE CUT POINTS ARE PORTS, NOT NEW RULES. `_HEALTH_CUTS`, `_CHANGE_CUTS`,
# `_BAR_*`, `_FALLBACK_STARTERS` and the four tone functions below are the
# app's own, from web/src/components/draft/{panels,finish,weeks}.ts and
# draft/AvailableList.tsx. Ported rather than re-derived: a season graded
# green here and amber in the room would be two answers about one year.

# How many pages the keep-warm pass renders after each `rebuild_adp`, in ESPN
# rank order. The first render of a player page is the only one that pays for
# `cached_profile`, and forty is the depth a crawler and a reader both reach
# first -- the whole first five rounds of an 8-team room.
PROFILE_WARM = 40

# What a player page is allowed to weigh, finished.
#
# These pages are read by a crawler working through a 228-URL sitemap in a
# burst and by a person on a phone on a train. The profile sections roughly
# doubled one -- the heaviest three measured against the real corpus (854
# drafts, 203 pages) are 61.2 KB, against a 56.5 KB median and a 29.6 KB
# page for a player the board carries no profile for -- and this ceiling is
# what stops the next section being added
# without anybody measuring. Enforced by the suite, not at render time: a
# page that has grown past it is a thing to go and look at, not a thing to
# truncate under a reader.
PLAYER_PAGE_MAX_BYTES = 90 * 1024

# The app's `SEASON_WEEKS` (18: the season, not the games a team plays) and
# `BAR_CEILING` (one scale for every player, so a bad player's best week
# cannot draw as tall as a stud's). See web/src/components/draft/weeks.ts.
WEEK_SLOTS = 18
BAR_CEILING = 30.0
# GAMES a team plays, which is `WEEK_SLOTS` minus the bye -- deliberately a
# separate number, and the one the app divides a season projection by
# (`SEASON_GAMES` in draft/weeks.ts, `GAMES` in scoring/board.py). A peer's
# projected points a game must be the same arithmetic the popup does or the
# two would not subtract.
SEASON_GAMES = 17

# `SHOWN` in UsageLine.tsx: three seasons of columns, because a percentage
# needs its width to stay legible. The line behind them is not drawn here.
USAGE_SEASONS = 3

# How many comparable seasons and how many board peers the page lists. The
# popup pages through twelve at a time and shows five peers; a static page
# has one page, and these are what fit before the section stops being read.
PROFILE_COMPS = 8
PROFILE_PEERS = 5

# Games per season a career averages, cut so the five steps spread the real
# board -- AvailableList.tsx's `HEALTH_CUTS`.
_HEALTH_CUTS = (10.0, 13.0, 15.0, 16.3)
# Projected points per game against the recency-weighted actual, same file's
# `CHANGE_CUTS`. Outliers are pinned rather than allowed to set the scale.
_CHANGE_CUTS = (-2.0, -0.5, 0.5, 2.0)

# How many of a position start, which is what makes a finish mean anything --
# `FALLBACK_STARTERS` in draft/finish.ts. These pages have no league to ask
# (nothing here reads a session), so the fallback IS the yardstick, and it is
# the twelve-team shape the app falls back to before a room is connected.
_FALLBACK_STARTERS = {"QB": 12, "TE": 12, "RB": 24, "WR": 24, "K": 12, "DST": 12}

# What a week was worth and what colour that makes it -- `BAR_THRESHOLDS` in
# draft/weeks.ts.
_BAR_THRESHOLDS = {"QB": (10.0, 20.0), "K": (5.0, 10.0), "DST": (5.0, 10.0)}
_BAR_DEFAULT = (10.0, 15.0)

# The two five-step ramps the app draws with, as class names this page's own
# stylesheet answers to (see base.html).
#
# TWO, BECAUSE THE APP HAS TWO. `e1..e5` is the panel ramp -- the one a
# season's finish, a week's points and a schedule's softness are graded on --
# and `m1..m5` is the meter ramp the board's Health, Reliable and Growth
# columns use. They agree at both ends and at the fourth step and differ in
# the middle, and collapsing them into one would repaint a column of the app.
_PANEL_RAMP = ("e1", "e2", "e3", "e4", "e5")
_METER_RAMP = ("m1", "m2", "m3", "m4", "m5")

# The game log's own columns per position, from WeekByWeek.tsx's `LOG`. A
# position not in here falls through to the payload's `stat_line`, which is
# what that component does.
_LOG_COLUMNS = {
    "QB": (("C/A", "ca"), ("Yds", "pass_yards"), ("TD", "pass_tds"),
           ("Int", "interceptions"), ("Ru", "rush_yards")),
    "RB": (("Car", "carries"), ("Ru", "rush_yards"), ("Rec", "receptions"),
           ("Re", "rec_yards"), ("TD", "tds"), ("Snap", "snap_pct"),
           ("Tgt%", "target_pct")),
    "WR": (("Tgt", "targets"), ("Rec", "receptions"), ("Yds", "yards"),
           ("TD", "tds"), ("Snap", "snap_pct"), ("Tgt%", "target_pct")),
}
_LOG_COLUMNS["TE"] = _LOG_COLUMNS["WR"]

# The usage rows per position, from UsageLine.tsx's `ROWS` and `SHARES`. Each
# is (label, how to read a season, how to read the projection, how to print
# it). `None` for the projection is the card declining rather than guessing:
# ESPN projects no completions, no team total and no snaps.
# Each row is (label, how to read a season, how to read the projection, how
# to print it, whether the season figure is a per-game rate, whether up is
# worse). A projection of `None` is the card declining rather than guessing:
# ESPN projects no completions, no team total and no snaps at all, so
# neither share and neither completion rate has a projected column.
#
# A `(numerator, denominator)` pair is a ratio -- catch rate is receptions
# over targets, not a column anybody stores -- and is per-game on neither
# side, since the games cancel.
_SHARE_ROWS = (
    ("Snap %", ("snap_share",), None, "pct", False, False),
    ("Target %", ("target_share",), None, "pct", False, False),
)
# Availability, last because it qualifies every row above it: a rate is per
# game, so a career of 11-game seasons reads identically to a career of
# 17-game ones until this row says otherwise.
_GAMES_ROW = ("Games", ("games",), ("games",), "count", False, False)
_USAGE_ROWS = {
    "QB": (("Att / g", ("attempts",), ("attempts",), "rate", True, False),
           ("Comp %", (("completions",), ("attempts",)), None, "pct", False, False),
           ("Pass yds / g", ("pass_yards",), ("pass_yards",), "rate", True, False),
           ("Pass TD / g", ("pass_tds",), ("pass_tds",), "rate", True, False),
           # Up is worse for exactly one row on this card: interceptions
           # rising is a quarterback getting worse.
           ("INT / g", ("interceptions",), ("interceptions",), "rate", True, True),
           ("Rush yds / g", ("rush_yards",), ("rush_yards",), "rate", True, False),
           # `tds` is rushing plus receiving and never a throw, so for a
           # quarterback it is his legs and nothing else.
           ("Rush TD / g", ("tds",), ("tds",), "rate", True, False),
           _GAMES_ROW),
    "RB": (("Car / g", ("carries",), ("carries",), "rate", True, False),
           ("Rec / g", ("receptions",), ("receptions",), "rate", True, False),
           ("Yds / g", ("rush_yards", "rec_yards"), ("yards",), "rate", True, False),
           # Yards per opportunity: what one touch or one look was worth.
           ("Yds / opp", (("rush_yards", "rec_yards"), ("carries", "targets")),
            (("yards",), ("carries", "targets")), "rate", False, False),
           ("TD / g", ("tds",), ("tds",), "rate", True, False),
           _GAMES_ROW),
    "WR": (("Tgt / g", ("targets",), ("targets",), "rate", True, False),
           ("Rec / g", ("receptions",), ("receptions",), "rate", True, False),
           ("Catch %", (("receptions",), ("targets",)),
            (("receptions",), ("targets",)), "pct", False, False),
           ("Yds / g", ("rec_yards", "rush_yards"), ("yards",), "rate", True, False),
           ("Yds / opp", (("rec_yards", "rush_yards"), ("targets", "carries")),
            (("yards",), ("targets", "carries")), "rate", False, False),
           ("TD / g", ("tds",), ("tds",), "rate", True, False),
           _GAMES_ROW),
}
_USAGE_ROWS["TE"] = _USAGE_ROWS["WR"]

# The step a season-to-season move has to clear to earn a colour, as a
# fraction of what it moved FROM -- UsageLine.tsx's `MOVE`. Relative, because
# three points of snap share is a real change at 20% and rounding at 90%.
_USAGE_MOVE = 0.05

# The five sources, labels and order of `scoring/market.py`'s consensus, the
# same list MarketRow.tsx keeps.
_MARKET_SOURCES = (("ffc", "FFC"), ("espn", "ESPN"), ("fp", "FantasyPros"),
                   ("mfl", "MFL"), ("cbs", "CBS"))


def _num_or_none(value):
    """A float, or None for a null, a NaN or anything that is not a number."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _level(value, cuts) -> int | None:
    """Which of five steps `value` lands on. One is the worst outcome and
    five the best in every meter on the page, which is what makes a glance
    across them mean one thing."""
    value = _num_or_none(value)
    if value is None:
        return None
    return 1 + sum(1 for cut in cuts if value >= cut)


def _starters_at(position: str) -> int:
    return _FALLBACK_STARTERS.get(position, 24)


def _finish_level(finish, starters: int) -> int | None:
    """`finishTone` in draft/finish.ts, as a step rather than a class name."""
    finish = _num_or_none(finish)
    if finish is None:
        return None
    if finish <= starters / 4:
        return 5
    if finish <= starters / 2:
        return 4
    if finish <= starters:
        return 3
    if finish <= starters * 2:
        return 2
    return 1


def _bar_level(points, position: str) -> int:
    """`barTone` in draft/weeks.ts: good, mid, bad -- the ramp's ends and its
    middle, which is the three steps a week is graded on."""
    amber, green = _BAR_THRESHOLDS.get(position, _BAR_DEFAULT)
    points = _num_or_none(points) or 0.0
    return 5 if points >= green else 3 if points >= amber else 1


def _band_level(value, cuts) -> int | None:
    """The step a value on a descending ladder of cuts takes: `cuts` is five
    steps' worth of floors, best first. `shareTone`, `placeTone` and the
    schedule strip's own ramp are all this shape."""
    value = _num_or_none(value)
    if value is None:
        return None
    for i, floor in enumerate(cuts):
        if value >= floor:
            return 5 - i
    return 1


def _place_level(rank, pool, ceils) -> int | None:
    """The step a PLACE takes: `rank / pool` against ceilings, best first.

    NOT `_band_level(1 - rank / pool, floors)`, WHICH IS THE SAME STATEMENT
    ONLY IN EXACT ARITHMETIC. The app asks "is this place inside the top
    fifth / quarter / half of its field" and asks it as `rank / pool <=
    ceiling` -- `steadyTone` in web/src/components/draft/panels.ts and
    `placeTone` in web/src/components/profile/LineQuality.tsx. Subtracted
    from one and compared against the complement it stops agreeing at every
    exact boundary: 1 - 24/30 is 0.19999999999999996, which is under a 0.2
    floor that 24/30 sits exactly on. 24/30, 9/10, 18/20 and every other
    place that lands on a cut were painted one step darker here than in the
    room. Same rank, same pool, two colours.
    """
    rank, pool = _num_or_none(rank), _num_or_none(pool)
    if rank is None or pool is None or pool <= 0:
        return None
    place = rank / pool
    for i, ceiling in enumerate(ceils):
        if place <= ceiling:
            return 5 - i
    return 1


_ORDINALS = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(n: int) -> str:
    """21 -> "21st". `ordinal` in profile/payload.ts, for the same rows."""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{_ORDINALS.get(n % 10, 'th')}"


def _fmt(value, digits: int = 1) -> str:
    value = _num_or_none(value)
    return "—" if value is None else f"{value:.{digits}f}"


def _pct_str(share) -> str:
    share = _num_or_none(share)
    return "—" if share is None else f"{round(share * 100):.0f}%"


def _count_str(value) -> str:
    """Games are counted, not measured: seventeen is "17", and only the
    projection's fractional 16.4 spends a decimal place. `count` in
    UsageLine.tsx."""
    value = _num_or_none(value)
    if value is None:
        return "—"
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _year(season) -> str:
    return f"’{str(int(season))[2:]}"


def _sum_or_none(row: dict, keys) -> float | None:
    """The sum of `keys`, or None when every one of them is missing. A row
    where nothing is known must not print a zero -- `projTotal` in
    UsageLine.tsx makes the same distinction for the same reason."""
    values = [_num_or_none(row.get(k)) for k in keys]
    if all(v is None for v in values):
        return None
    return sum(v or 0.0 for v in values)


def _usage_value(row: dict, spec, per_game: bool):
    """One usage cell: a per-game rate, or one column over another.

    `spec` is either a tuple of column names (summed, and divided by games
    where the source is a season rather than the already-per-game
    projection), or a pair of them (a ratio -- catch rate is receptions over
    targets, not a column anybody stores). A zero denominator is None, not
    zero: a back with no targets has no catch rate, and 0% would say he
    dropped everything.
    """
    if spec and isinstance(spec[0], tuple):
        top = _sum_or_none(row, spec[0])
        bottom = _sum_or_none(row, spec[1])
        return None if top is None or not bottom else top / bottom
    total = _sum_or_none(row, spec)
    if total is None:
        return None
    if not per_game:
        return total
    games = _num_or_none(row.get("games"))
    return total / games if games else None


def _move_tone(now, prev, invert: bool = False) -> str:
    """Green if the number is up on the one before it, red if it is down --
    `tone` in UsageLine.tsx, including the one row where up is worse."""
    now, prev = _num_or_none(now), _num_or_none(prev)
    if now is None or prev is None or prev == 0:
        return ""
    move = ((now - prev) / abs(prev)) * (-1 if invert else 1)
    if move >= _USAGE_MOVE:
        return "pf-up"
    if move <= -_USAGE_MOVE:
        return "down"
    return ""


_profile_failed = False


def _profile_unavailable(exc: BaseException) -> None:
    """Say once per process that the profile could not be had.

    ONCE, because a profile failing on all 203 pages of a crawl is one fact
    about the database, not two hundred -- and a log that repeats it two
    hundred times is a log nobody reads the second line of.
    """
    global _profile_failed
    if not _profile_failed:
        _profile_failed = True
        print(f"seo: profile enrichment unavailable, pages render "
              f"without it: {exc!r}", flush=True)


def cached_profile_or_none(conn, player_id: str, settings=None):
    """`scoring.profile_cache.cached_profile`, or None however it fails.

    THE PAGE IS NOT THE PROFILE. Every figure above these sections comes out
    of the corpus and needs nothing from the universal database; the profile
    is an enrichment on top, exactly like `scoring/adp_facts.attach`, and a
    universal database that cannot answer must cost the enrichment rather
    than the page. So this swallows -- and says so once per process.

    NOTHING IS HELD WHILE THIS RUNS. `cached_profile` single-flights on its
    own lock and takes one of `scoring.board_cache.BUILD_SLOTS`' two build
    permits, and it can sit behind another thread's build for a second or
    more. It is called from inside `rendered`'s `build()`, which runs with
    neither `_pages_lock` nor `_adp_lock` held -- see `rendered` for why the
    page cache is written after the build rather than around it.
    """
    if conn is None:
        return None
    try:
        from scoring import league
        from scoring.profile_cache import cached_profile
        return cached_profile(conn, player_id, None, settings or league.load(conn))
    except Exception as exc:      # noqa: BLE001 -- see the docstring
        _profile_unavailable(exc)
        return None


def _meters(payload: dict) -> list:
    """The four the board's own columns carry: how much of the season he is
    there for, where he finishes, how steady he is week to week, and which
    way the projection moves. Each is a step of five with the number it was
    cut from beside it -- the bars are the glance, the number is what makes
    them falsifiable."""
    header = payload["header"]
    position = str(header.get("position") or "")
    starters = _starters_at(position)
    out = []

    games_pg = _num_or_none(header.get("career_games_pg"))
    level = _level(games_pg, _HEALTH_CUTS)
    if level is not None:
        out.append({"label": "Health", "level": level, "cls": _METER_RAMP[level - 1],
                    "value": _fmt(games_pg), "note": "games a season, career"})

    # The projection's own place first, and a played season only where there
    # is no projection: the app's Finish panel draws the forecast as the last
    # column on the ladder the played seasons are on, and it is the column a
    # reader carries away.
    finish = _num_or_none(header.get("proj_pos_finish"))
    note = f"projected finish, of {starters} starters"
    if finish is None:
        played = [s for s in payload.get("seasons") or [] if (s.get("games") or 0) > 0]
        if played:
            finish = _num_or_none(played[0].get("pos_finish"))
            note = f"{played[0]['season']} finish, of {starters} starters"
    level = _finish_level(finish, starters)
    if level is not None:
        # `of` so the page can say what the place is a place among. A
        # DEFENSE HAS NOTHING ELSE THAT SAYS IT: it gets no season table, so
        # the twelve-team caveat under that table -- the only place the
        # yardstick was ever stated -- never renders on its page.
        out.append({"label": "Finish", "level": level, "cls": _PANEL_RAMP[level - 1],
                    "value": f"{position}{int(finish)}", "note": note,
                    "of": starters})

    pct = _num_or_none(header.get("consistency_pct"))
    if pct is not None:
        # `steadyLevel` in AvailableList.tsx: the percentile IS the meter,
        # so an even fifth of the position lands on each bar by construction.
        level = min(5, max(1, math.ceil(pct * 5)))
        cv = _num_or_none(header.get("consistency_cv"))
        out.append({"label": "Reliable", "level": level, "cls": _METER_RAMP[level - 1],
                    "value": "—" if cv is None else f"{cv:.2f}",
                    "note": f"steadier than {(level - 1) * 20}–{level * 20}% "
                            "of his position"})

    change = _num_or_none(header.get("proj_change"))
    level = _level(change, _CHANGE_CUTS)
    if level is not None:
        out.append({"label": "Growth", "level": level, "cls": _METER_RAMP[level - 1],
                    "value": _signed(change, 1), "note": "points a game vs last season"})
    return out


def _season_rows(payload: dict) -> dict | None:
    """Every season he has played, as the four panels state them: games, the
    average, where it finished him, and how steady it was. Newest first, the
    order the payload has them in, with the projection as its own last row --
    the one line here that has not happened."""
    seasons = payload.get("seasons") or []
    if not seasons:
        return None
    position = str(payload["header"].get("position") or "")
    starters = _starters_at(position)
    rows = []
    for s in seasons:
        finish = _num_or_none(s.get("pos_finish"))
        level = _finish_level(finish, starters)
        ppg = _num_or_none(s.get("ppg"))
        cv_rank, cv_of = _num_or_none(s.get("cv_rank")), _num_or_none(s.get("cv_rank_n"))
        rows.append({
            "season": int(s["season"]),
            "age": s.get("age"), "nfl": s.get("nfl_season"),
            "games": s.get("games"),
            "ppg": _fmt(ppg),
            "ppg_cls": _PANEL_RAMP[_bar_level(ppg, position) - 1],
            "finish": "—" if finish is None else f"{position}{int(finish)}",
            "finish_cls": "" if level is None else _PANEL_RAMP[level - 1],
            "ppg_rank": ("—" if s.get("pos_rank_ppg") is None
                         else f"{int(s['pos_rank_ppg'])} of {int(s['pos_rank_ppg_n'])}"),
            "steady": ("—" if cv_rank is None or not cv_of
                       else f"{int(cv_rank)} of {int(cv_of)}"),
            # `steadyTone`'s own ceilings, in its own order.
            "steady_cls": ("" if cv_rank is None or not cv_of else
                           _PANEL_RAMP[_place_level(cv_rank, cv_of,
                                                    (0.25, 0.5, 0.75, 0.9)) - 1]),
        })
    proj_ppg = _num_or_none((payload.get("summary") or {}).get("proj_ppg"))
    proj_finish = _num_or_none(payload["header"].get("proj_pos_finish"))
    level = _finish_level(proj_finish, starters)
    proj = None
    if proj_ppg is not None or proj_finish is not None:
        proj = {"season": payload["bio"].get("season"),
                "ppg": _fmt(proj_ppg),
                "ppg_cls": ("" if proj_ppg is None
                            else _PANEL_RAMP[_bar_level(proj_ppg, position) - 1]),
                "finish": ("—" if proj_finish is None
                           else f"{position}{int(proj_finish)}"),
                "finish_cls": "" if level is None else _PANEL_RAMP[level - 1]}
    return {"position": position, "starters": starters, "rows": rows, "proj": proj}


def _weeks(payload: dict) -> dict | None:
    """Last season by week: eighteen slots, one per week of the season rather
    than one per game he appeared in -- a row of twelve bars beside a row of
    eighteen would make six missed games invisible. `WeekByWeek` and
    `weekCols` draw exactly this.

    A week with no game is a mark on the baseline, not a bar of no height:
    "did not play" and "played and scored nothing" are different claims.
    """
    log = payload.get("game_log") or []
    played = [g for g in log if not g.get("dnp")]
    if not played:
        return None
    season = max(int(g["season"]) for g in played)
    position = str(payload["header"].get("position") or "")
    rows = {int(g["week"]): g for g in played if int(g["season"]) == season}
    if not rows:
        return None
    cols = []
    for week in range(1, WEEK_SLOTS + 1):
        game = rows.get(week)
        points = None if game is None else _num_or_none(game.get("ppr_points"))
        cols.append({
            "week": week,
            "label": (game or {}).get("opponent") or "—",
            "value": "·" if points is None else f"{round(points):.0f}",
            # The bar's own height in the drawing, so the template does no
            # arithmetic: 80 units of chart above a baseline at 104, with a
            # floor of five so a game he played and scored nothing in is
            # still a bar. A week with no game is not one -- see `empty`.
            "h": (0 if points is None else
                  max(5, round(min(1.0, max(0.0, points / BAR_CEILING)) * 80))),
            "cls": "" if points is None else _PANEL_RAMP[_bar_level(points, position) - 1],
            "empty": points is None,
        })
    summary = next((s for s in payload.get("seasons") or []
                    if int(s["season"]) == season), None)
    heads = _LOG_COLUMNS.get(position)
    lines = []
    for game in sorted(rows.values(), key=lambda g: -int(g["week"])):
        stats = game.get("stats") or {}
        cells = []
        if heads is None:
            cells.append(game.get("stat_line") or "—")
        else:
            for _, key in heads:
                if key == "ca":
                    cells.append(f"{stats.get('completions', 0)}/{stats.get('attempts', 0)}")
                elif key == "tds":
                    cells.append(str(int((stats.get("rush_tds") or 0)
                                         + (stats.get("rec_tds") or 0))))
                elif key == "yards":
                    cells.append(str(int((stats.get("rec_yards") or 0)
                                         + (stats.get("rush_yards") or 0))))
                elif key in ("snap_pct", "target_pct"):
                    cells.append(_pct_str(game.get(key)))
                else:
                    cells.append(str(int(stats.get(key) or 0)))
        points = _num_or_none(game.get("ppr_points")) or 0.0
        lines.append({"week": int(game["week"]),
                      "opponent": game.get("opponent") or "—",
                      "cells": cells, "points": _fmt(points),
                      "cls": _PANEL_RAMP[_bar_level(points, position) - 1]})
    return {"season": season, "cols": cols, "lines": lines,
            "heads": [h for h, _ in heads] if heads else ["Line"],
            "avg": _fmt(summary["ppg"]) if summary else None,
            "finish": (f"{position}{int(summary['pos_finish'])}"
                       if summary and summary.get("pos_finish") is not None else None)}


def _usage(payload: dict) -> dict | None:
    """What the points were made of, season by season, and how big a share of
    the offence they came off.

    Share first, rates under it: a rate is what he did, a share is how much
    of his offence he was, which is the half that survives a coaching change.
    Three seasons of columns and the projection beside them -- what
    `UsageLine` draws, at the same width and for the same reason.

    A ROW NOTHING CAN ANSWER IS DROPPED, not dashed across, and a row of
    genuine zeros goes the same way: a quarterback's target share is not
    missing, it is 0.000 every season, and a flat row of noughts is a career
    of facts that never happened. There is no `K` or `DST` entry in
    `_USAGE_ROWS` and that is deliberate -- a kicker's season row carries no
    skill columns, so the lookup falls through and the section is not drawn.
    """
    seasons = payload.get("seasons") or []
    position = str(payload["header"].get("position") or "")
    shown = list(reversed(seasons[:USAGE_SEASONS]))
    if not shown:
        return None
    projected = (payload.get("summary") or {}).get("proj_usage")
    printers = {"pct": _pct_str, "count": _count_str, "rate": lambda v: _fmt(v)}
    rows = []
    for label, of, proj_of, fmt, per_game, invert in (
            tuple(_SHARE_ROWS) + tuple(_USAGE_ROWS.get(position, ()))):
        values = [_usage_value(s, of, per_game) for s in shown]
        proj = (_usage_value(projected, proj_of, False)
                if proj_of and projected else None)
        if not any(v not in (None, 0) for v in values) and proj in (None, 0):
            continue
        printer = printers[fmt]
        cells, previous = [], None
        for value in values:
            cells.append({"text": "—" if value is None else printer(value),
                          "tone": _move_tone(value, previous, invert),
                          "blank": value is None})
            if value is not None:
                previous = value
        last = next((v for v in reversed(values) if v is not None), None)
        rows.append({
            "label": label, "cells": cells, "share": of in (("snap_share",),
                                                            ("target_share",)),
            "proj": ({"text": "—" if proj is None else printer(proj),
                      "tone": _move_tone(proj, last, invert),
                      "blank": proj is None} if projected else None)})
    if not rows:
        return None
    # The seam between the two shares and the rates under them, drawn only
    # when there is a first half above it to be separated from.
    first_rate = next((i for i, r in enumerate(rows) if not r["share"]), -1)
    return {"head": [_year(s["season"]) for s in shown], "rows": rows,
            "projected": projected is not None,
            "seam": first_rate if first_rate > 0 else -1}


def _schedule(payload: dict) -> dict | None:
    """The season as eighteen weeks, ranked on how much the defence in front
    of him gave up to his position last year.

    THE DIRECTION IS INVERTED FROM INTUITION AND EVERY LABEL SAYS SO: rank 1
    is the SOFTEST defence, and reading the strip as a difficulty ranking
    turns the best week of the season into the worst. `ScheduleRanks` draws
    the same weeks on the same ramp.
    """
    weeks = payload.get("schedule") or []
    if not any(w.get("pct") is not None for w in weeks):
        return None
    teams = next((int(w["rank_n"]) for w in weeks if w.get("rank_n")), None)
    sos = _num_or_none((payload.get("outlook") or {}).get("sos_pct"))
    note = None
    if sos is not None and teams:
        # `softestRank`: `sos_pct` is a percentile rank over the 32 teams, so
        # the place it came from is exact rather than reconstructed.
        # `math.floor(x + 0.5)`, not `round`: Python rounds a half to even
        # and JavaScript rounds it up, and this number has to be the one
        # `softestRank` produces for the same percentile.
        ascending = math.floor(sos / 100 * teams + 0.5)
        note = f"{_ordinal(min(teams, max(1, teams - ascending + 1)))} easiest"
    out = []
    for w in weeks:
        pct = _num_or_none(w.get("pct"))
        level = _band_level(pct, (75, 60, 40, 25))
        out.append({"week": w.get("week"),
                    "opponent": w.get("opponent") or "bye",
                    "away": w.get("home") is False,
                    "rank": ("—" if w.get("rank") is None
                             else f"{int(w['rank'])}/{int(w['rank_n'])}"),
                    "fpa": _fmt(w.get("fpa_pg")),
                    "cls": "" if level is None else _PANEL_RAMP[level - 1]})
    return {"note": note, "weeks": out, "teams": teams}


def _oline(payload: dict) -> dict | None:
    """The line in front of him: where it places, what the place is made of,
    and who is on it. Null for every defense by construction -- the o-line is
    a fact about the eleven who leave the field when that unit comes on."""
    oline = payload.get("oline")
    if not oline:
        return None
    parts = []
    for key, label in (("continuity", "Same five"), ("availability", "Available"),
                       ("returning", "Returning")):
        share = _num_or_none(oline.get(key))
        if share is None:
            continue
        level = _band_level(share, (0.9, 0.8, 0.7, 0.6))
        parts.append({"label": label, "fill": round(min(1.0, share) * 100),
                      "value": _pct_str(share), "cls": _PANEL_RAMP[level - 1]})
    years, years_pct = _num_or_none(oline.get("experience")), _num_or_none(
        oline.get("experience_pct"))
    if years is not None and years_pct is not None:
        level = _band_level(years_pct, (80, 60, 40, 20))
        parts.append({"label": "Experience", "fill": round(min(100.0, years_pct)),
                      "value": f"{round(years)}yr", "cls": _PANEL_RAMP[level - 1]})
    starters = []
    for s in oline.get("starters") or []:
        if not s.get("name"):
            continue
        rank, of = _num_or_none(s.get("avail_rank")), _num_or_none(s.get("avail_rank_of"))
        # `placeTone`'s own ceilings: top fifth, top two fifths, and down.
        level = _place_level(rank, of, (0.2, 0.4, 0.6, 0.8))
        starters.append({"position": s.get("position") or "—", "name": s["name"],
                         "place": "—" if rank is None or not of else f"{int(rank)}/{int(of)}",
                         "cls": "" if level is None else _PANEL_RAMP[level - 1]})
    return {"team": oline.get("team"), "rank": _ordinal(int(oline["rank"])),
            "teams": int(oline["teams"]), "parts": parts, "starters": starters}


def _comparables(payload: dict, slugs: dict) -> dict | None:
    """The seasons his looks like, and what they became.

    `stat_twins` only. `value_neighbors` -- the fallback for a player with no
    stat line to match, which is every defense and every rookie -- is a list
    of board positions, not of seasons, and reading it under this heading
    would be the page claiming a resemblance nobody measured.
    """
    similar = payload.get("similar") or {}
    if similar.get("mode") != "stat_twins":
        return None
    rows = []
    for p in similar.get("players") or []:
        season, ppg = p.get("season"), _num_or_none(p.get("ppg"))
        if season is None or ppg is None:
            continue
        move = (None if p.get("next_ppg") is None
                else _num_or_none(p["next_ppg"]) - ppg)
        rows.append({
            "name": p.get("name"),
            # Only a twin the board still knows has a page of his own: Todd
            # Gurley's 2017 is not a player anybody can draft this year, and
            # the payload marks those with a null rank.
            "slug": slugs.get(p.get("player_id")) if p.get("rank") is not None else None,
            "season": season, "ppg": _fmt(ppg),
            "change": "—" if move is None else _signed(move, 1),
            # Green up, red down, and nothing at all for a move that rounds
            # to nothing: the same two tokens the rest of the page spends on
            # "better" and "worse".
            "tone": ("" if move is None else "pf-up" if round(move, 1) > 0
                     else "down" if round(move, 1) < 0 else ""),
            "match": "—" if p.get("similarity") is None else f"{round(p['similarity'])}%",
        })
        if len(rows) >= PROFILE_COMPS:
            break
    if not rows:
        return None
    cohort = payload.get("cohort")
    note = None
    if cohort and cohort.get("n"):
        changes = [_num_or_none(c.get("change")) for c in cohort.get("players") or []]
        changes = [c for c in changes if c is not None]
        avg = (sum(changes) / len(changes) if changes
               else _num_or_none(cohort.get("median_change")))
        if avg is not None:
            note = {"value": _signed(avg, 1),
                    "tone": ("pf-up" if round(avg, 1) > 0
                             else "down" if round(avg, 1) < 0 else ""),
                    "n": int(cohort["n"]), "declined": int(cohort.get("declined") or 0)}
    return {"rows": rows, "note": note,
            "age": (payload.get("similar") or {}).get("target_age")}


def _peers(payload: dict, slugs: dict) -> list:
    """Who else is on the shelf: the players on THIS year's board scored
    against him, with what taking one instead would cost or buy per game."""
    data = payload.get("similar_players") or {}
    proj = _num_or_none((payload.get("summary") or {}).get("proj_ppg"))
    out = []
    for p in (data.get("players") or [])[:PROFILE_PEERS]:
        points = _num_or_none(p.get("proj_points"))
        per_game = None if points is None else points / SEASON_GAMES
        gap = None if per_game is None or proj is None else per_game - proj
        out.append({"name": p.get("name"), "slug": slugs.get(p.get("player_id")),
                    "ppg": _fmt(per_game),
                    "gap": "" if gap is None else _signed(gap, 1),
                    # `SAME` in SimilarPlayers.tsx: half a point a game is
                    # two players the same size, and colouring it would be
                    # the card claiming a choice nobody has to make.
                    "tone": ("" if gap is None or abs(gap) < 0.5
                             else "pf-up" if gap > 0 else "down")})
    return out


def _room(payload: dict) -> dict | None:
    """His own place in his own position room, off the depth chart the
    payload carries. `depthGroup` in profile/payload.ts: whichever group
    actually holds him comes before the one his board position names -- a
    receiver taking snaps at running back belongs in the room he is really
    competing in."""
    groups = payload.get("depth_chart") or []
    position = str(payload["header"].get("position") or "")
    mine = next((g for g in groups if any(p.get("is_me") for p in g.get("players") or [])),
                None)
    if mine is None:
        mine = next((g for g in groups if g.get("position") == position), None)
    if not mine or not mine.get("players"):
        return None
    return {"position": mine.get("position"), "team": payload["header"].get("team"),
            "players": [{"name": p.get("name"), "rank": p.get("rank"),
                         "me": bool(p.get("is_me"))} for p in mine["players"]]}


def _market(payload: dict) -> dict:
    """Where the market has him, as places among his own position as well as
    overall: a roster is filled by position, so "WR7" is the number the row
    is really about and the overall is the scale it sits on. `MarketRow`
    prints the same rows off the same header."""
    header = payload["header"]
    position = str(header.get("position") or "")
    places = header.get("market_pos") or {}

    def place(key):
        value = places.get(key)
        return None if value is None else f"{position}{int(value)}"

    edge = _num_or_none(header.get("edge"))
    return {"rank": header.get("rank"), "board_pos": place("board"),
            "consensus_pos": place("consensus"),
            "spread": _num_or_none(header.get("market_spread")),
            "pos": {key: place(key) for key, _ in _MARKET_SOURCES},
            "edge": None if edge is None else round(edge),
            "edge_text": None if edge is None else _signed(edge, 0),
            # `edgeTone` in profile/payload.ts: under ten slots the board and
            # the market take him in the same round of any league this tool
            # supports, so there is no decision in the gap.
            "edge_tone": ("" if edge is None else "pf-up" if round(edge) >= 10
                          else "pf-accent" if round(edge) > 0
                          else "" if round(edge) == 0 else "down")}


def _profile_news(payload: dict) -> list:
    """The feed's rows, shaped the way `scoring/adp_facts._news` shapes its
    own, so the page's one news list can hold both without knowing which
    filled it."""
    out = []
    for item in payload.get("news") or []:
        published = item.get("published_at")
        if isinstance(published, str):
            try:
                published = datetime.fromisoformat(published)
            except ValueError:
                published = None
        out.append({"headline": item.get("headline"), "url": item.get("url"),
                    "source": item.get("source"), "date": published})
    return out


def _link_or_none(url) -> str | None:
    """A news url this page will hang an `<a href>` on, or None.

    `player_news.url` is whatever the feed that filled the row put there,
    and the page has no say in it. Anything that is not plain http(s) is
    printed as text instead: `javascript:` and `data:` are a script the
    page would be handing a reader on a click, and a bare path or an empty
    string is a link into this site that goes nowhere. The headline is the
    thing worth reading either way, so a row never disappears over its url.
    """
    url = (url or "").strip()
    return url if url.lower().startswith(("http://", "https://")) else None


def _merge_news(existing: list, extra: list) -> list:
    """One list, newest first, with nothing said twice.

    The two sources ARE the same table (`player_news`): `adp_facts` reads it
    by id for the whole board in one pass and `scoring.profile.player_news`
    reads it for one player. A page that printed both would print every
    headline twice, and one that dropped either would lose the rows the other
    matched -- so they are merged on the url, and on the headline for a row
    that has none.
    """
    out, seen = [], set()
    for item in list(existing or []) + list(extra or []):
        key = (item.get("url") or "").strip() or (item.get("headline") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        # Deduplicated on the url the feed gave, printed with the one the
        # page is willing to link: a `javascript:` row is still that row.
        out.append({**item, "url": _link_or_none(item.get("url"))})
    # ON A COMMON TYPE. `adp_facts` hands over a `date` and the profile a
    # `datetime`, which Python will not order against each other -- and the
    # first merged page raised rather than sorted.
    def when(item):
        stamp = item.get("date")
        if isinstance(stamp, datetime):
            return stamp
        if isinstance(stamp, date):
            return datetime(stamp.year, stamp.month, stamp.day)
        return datetime.min

    out.sort(key=when, reverse=True)
    return out


def profile_view(conn, player: dict, slugs: dict, settings=None) -> dict | None:
    """Everything the popup draws about one player, ready for the template.

    `slugs` maps player_id -> the slug of his own page, so a comparable or a
    board peer who is on this board becomes a link and one who is not stays
    text. Returns None when there is no profile to draw -- an unknown id, a
    universal database that cannot answer, or a payload the shapers below
    cannot make a section out of.

    THE GUARD COVERS THE SHAPING, NOT ONLY THE FETCH. `cached_profile_or_none`
    swallows everything the database can do wrong and then hands the payload
    to fifteen shapers that swallow nothing: `_meters` and `_season_rows`
    read `payload["header"]` and `payload["bio"]` by subscript, `_oline`
    reads `int(oline["rank"])`, `_weeks` reads `int(g["week"])`. Every one of
    those is a key the profile builder is entitled to leave null, and a null
    in any of them used to be a 500 on a page that needs none of it. Same
    bargain as the fetch, then, and the same one line per process.
    """
    payload = cached_profile_or_none(conn, player["player_id"], settings)
    if payload is None:
        return None
    try:
        return _profile_view(payload, slugs)
    except Exception as exc:      # noqa: BLE001 -- see the docstring
        _profile_unavailable(exc)
        return None


def _profile_view(payload: dict, slugs: dict) -> dict:
    """`profile_view`'s shaping, split out so its guard can be one `try`
    around the lot rather than fifteen."""
    header = payload.get("header") or {}
    status = payload.get("status") or {}
    slot = status.get("depth_chart_order")
    vegas = payload.get("vegas") or {}
    summary = payload.get("summary") or {}
    return {
        "position": str(header.get("position") or ""),
        "season": (payload.get("bio") or {}).get("season"),
        "proj_ppg": _fmt(summary.get("proj_ppg")),
        "proj_delta": (None if summary.get("proj_delta") is None
                       else _signed(summary["proj_delta"], 1)),
        "meters": _meters(payload),
        "seasons": _season_rows(payload),
        "weeks": _weeks(payload),
        "usage": _usage(payload),
        "schedule": _schedule(payload),
        "room": _room(payload),
        "oline": _oline(payload),
        "vegas": ({"implied": _fmt(vegas.get("implied")),
                   # A PLACE, spelled the way the O-line card's own lead
                   # spells one, because the two cards are making the same
                   # kind of statement out of the same thirty-two teams.
                   "rank": (None if vegas.get("rank") is None
                            else _ordinal(int(vegas["rank"]))),
                   "teams": vegas.get("teams"), "priced": vegas.get("priced")}
                  if _num_or_none(vegas.get("implied")) is not None else None),
        "market": _market(payload),
        "comps": _comparables(payload, slugs),
        "peers": _peers(payload, slugs),
        "news": _profile_news(payload),
        "status": ({"injury": status.get("injury_status"),
                    "body_part": status.get("injury_body_part"),
                    "notes": status.get("injury_notes"),
                    "slot": (None if not slot else
                             f"{status.get('depth_chart_position') or header.get('position')}"
                             f"{int(slot)}")}
                   if status else None),
    }


def _provenance(data: dict) -> str:
    if not data["drafts"]:
        return "No drafts recorded yet."
    when = data["updated"].strftime("%b %-d, %Y") if data["updated"] else "today"
    return (f"From {data['drafts']} real ESPN mock drafts "
            f"({data['teams']}-team PPR, {data['rounds']} rounds), updated {when}.")


def _person(p: dict, image: str | None) -> dict:
    """The player, as structured data.

    `Person` for a player and `SportsTeam` for a D/ST, because a defense is
    not a person and saying so would be the sort of wrong that a rich result
    is built on. Only fields this page can actually stand behind: the name,
    the page's own URL, the face when there is one, and the team when the
    depth charts or ESPN's board named it.
    """
    url = f"{SITE}/adp/{p['slug']}"
    if p["position"] == "DST":
        node = {"@context": "https://schema.org", "@type": "SportsTeam",
                "name": p["name"], "url": url, "sport": "American football"}
    else:
        node = {"@context": "https://schema.org", "@type": "Person",
                "name": p["name"], "url": url,
                "jobTitle": f"{p['position']}, American football"}
        if p.get("team"):
            # THE NAME, NOT THE ABBREVIATION. `affiliation.name` is read by
            # something that has never seen a depth chart, and "CHI" is not
            # the name of a team to anybody but us.
            node["affiliation"] = {
                "@type": "SportsTeam",
                "name": TEAM_NAMES.get(p["team"].upper(), p["team"]),
                "sport": "American football"}
    if image:
        node["image"] = image
    return node


def _crumbs(*items) -> dict:
    """`BreadcrumbList` structured data from (name, path) pairs."""
    return {"@context": "https://schema.org", "@type": "BreadcrumbList",
            "itemListElement": [
                {"@type": "ListItem", "position": i, "name": name, "item": SITE + path}
                for i, (name, path) in enumerate(items, start=1)]}


def _index_faq(data: dict) -> list:
    """The questions somebody arriving from a search for this page's own
    query still has, answered in the page's own figures. Rendered as
    `<details>` and, alongside, as `FAQPage` structured data.

    The index alone. Repeating one FAQ across the six position pages would
    make them near-duplicates of each other and of the index, which is the
    thin-content pattern the share floor in `build_adp` already guards
    against on the player pages."""
    drafts = data["drafts"]
    teams, rounds = data["teams"], data["rounds"]
    corpus = (f"{drafts} ESPN mock drafts this site recorded"
              if drafts else "the ESPN mock drafts this site records")
    return [
        ("What does ADP mean?",
         "Average draft position: the average pick number a player was taken "
         f"at, across every draft he appeared in. In {corpus}, a player with "
         "an ADP of 24.0 went around the 24th pick on average. Sometimes "
         "earlier, sometimes later, which is what the range column shows."),
        ("Where do these numbers come from?",
         f"From {corpus}. These are real drafts, played out pick by pick, "
         "not projections and not an average of other sites' rankings. Every "
         "draft counted here is one this site watched from the inside."),
        ("What does the range column mean?",
         "The 10th to the 90th percentile of his picks: eight drafts in ten "
         "took him somewhere between those two numbers. A narrow range means "
         "the room agrees on him; a wide one means he is a reach for some "
         "drafters and a steal for others."),
        (f"Why {teams}-team PPR?" if drafts else "Which league shape is this?",
         f"That is the shape ESPN's public mock draft lobby runs: {teams} "
         f"teams, {rounds} rounds, PPR scoring. Draft position moves with "
         "league size, so mixing shapes into one number would blur all of "
         "them. Every figure on these pages comes from that one shape."),
        ("Is this ESPN's own ADP?",
         "No. ESPN publishes its own average draft position, drawn from real "
         "leagues; each player page shows it beside the mock figure so you "
         "can see where the two disagree. Mock drafters and league drafters "
         "do not behave the same way, and the gap is often the interesting "
         "part."),
        ("Will he still be there when my turn comes?",
         "Every player page answers that at picks 8, 16, 24, 36, 48, 60, 72 "
         "and 96, and draws the whole curve to pick 100. The figure is "
         "counted, not modelled: it is the share of the drafts he was on the "
         "board for in which nobody had taken him yet."),
        ("What makes a riser or a faller?",
         "Both lists compare where these rooms take a player with where "
         "ESPN's own board would. The two lists are put on one scale first "
         "-- order the players ESPN's way and read this corpus's own ADP off "
         "at each position -- because ESPN measures draft position over a "
         "much larger player pool than an "
         f"{teams}-team room ever reaches. A riser goes at least {MOVE_PICKS} "
         "picks before ESPN's board would take him."),
        ("How often does this update?",
         "Daily. New mock drafts are recorded continuously and every page "
         "here is rebuilt from the full corpus, so the count in the line "
         "under the heading goes up over the course of a season."),
    ]


def _tiers(players: list) -> list:
    """FantasyPros' tiers over one position's players, in draft order.

    A tier is the market saying "these are interchangeable, and the next
    group is not" -- which is the one piece of grouping a position page can
    show that a sorted list cannot. Only tiers with a player in them appear,
    and a position nobody has tiered gets nothing.
    """
    groups: dict = {}
    for p in players:
        if p.get("fp_tier"):
            groups.setdefault(int(p["fp_tier"]), []).append(p)
    out = []
    for tier in sorted(groups):
        members = groups[tier]
        out.append({"tier": tier, "players": members, "count": len(members),
                    "first": min(p["adp"] for p in members),
                    "last": max(p["adp"] for p in members)})
    return out


def _faq_schema(faq: list) -> dict:
    """`FAQPage` structured data from (question, answer) pairs."""
    return {"@context": "https://schema.org", "@type": "FAQPage",
            "mainEntity": [
                {"@type": "Question", "name": q,
                 "acceptedAnswer": {"@type": "Answer", "text": a}}
                for q, a in faq]}


def _round_rows(data: dict, n: int) -> list:
    """Who goes in round `n`, with the two shares that mean different things.

    A ROUND HAS TO HOLD SOME OF HIM TO LIST HIM. Membership used to be
    "his 10th-to-90th percentile crosses this round", which for a player
    taken anywhere from pick 1 to pick 123 is most of the site -- and the
    row it printed said 0%. Now a round lists him when at least
    ROUND_MIN_SHARE of the picks that took him landed in it.

    `picks_share` is that number: of the times he came off the board, how
    often it was here. `share` is the other question -- of the drafts he was
    on the board for, how often he went here -- and it is the smaller of the
    two for anybody the rooms often leave alone. Both are printed, labelled
    as what they are.

    Straight out of the histogram either way: the picks that make up round
    `n` are a contiguous slice of it. No second query.
    """
    teams = data["teams"]
    first, last = (n - 1) * teams + 1, n * teams
    rows = []
    for p in data["players"]:
        count = sum(p["hist"][first - 1:last])
        if not count or count / p["taken"] < ROUND_MIN_SHARE:
            continue
        rows.append({"p": p, "count": count,
                     "picks_share": round(count / p["taken"], 3),
                     "share": round(count / p["of"], 3) if p["of"] else 0.0,
                     "usual": p["usual_round"] == n})
    rows.sort(key=lambda row: (not row["usual"], row["p"]["adp"]))
    return rows


def round_players(data: dict, n: int) -> list:
    """The players `_round_rows` lists, in the order it lists them."""
    return [row["p"] for row in _round_rows(data, n)]


def _present(data: dict) -> list:
    """The positions somebody actually went at in this corpus, in board
    order. A page must not link to a position page with nobody on it."""
    seen = {p["position"] for p in data["players"]}
    return [pos for pos in POSITIONS if pos in seen]


def _seats(data: dict, n: int) -> list:
    """What the first, middle and last seat of the draft see in round `n`.

    CHOOSING A SEAT IS CHOOSING A LIST OF PICK NUMBERS -- that is the whole
    of what a snake draft does to you -- and the only way to compare two
    seats is to look at who is on the board when each comes round. Three
    seats rather than all of them: the two ends and the middle are the shape
    of the answer, and eight columns of three names would be a table nobody
    reads.

    The snake, stated once: odd rounds run out from slot 1, even rounds run
    back from the last slot.
    """
    teams = data["teams"]
    out = []
    for slot in sorted({1, (teams + 1) // 2, teams}):
        pick = (n - 1) * teams + (slot if n % 2 else teams - slot + 1)
        out.append({"slot": slot, "pick": pick,
                    "top": data["at_pick"].get(pick, [])})
    return out


def _missing(path: str):
    from fastapi.responses import HTMLResponse
    return HTMLResponse(render("adp_missing.html", title="Not found – ESPN Draft Assist",
                               description="No such page.", path=path, noindex=True),
                        status_code=404)


def register_seo_routes(app, conn=None):
    """The crawlable site: `/adp`, its player, round and position pages, and
    `/sitemap.xml`. Register before `api/static.register_spa`.

    Every route answers HEAD as well as GET. FastAPI's `@app.get` registers
    the one method -- unlike Starlette's own `Route`, which adds HEAD next to
    GET -- so a HEAD to a page that GET serves fine came back 405. Crawlers
    mostly use GET, but link previewers and uptime checks use HEAD."""
    from fastapi.responses import HTMLResponse, Response

    # Five minutes. These pages are rendered from the same ADP aggregate the
    # archive is, they carry nobody's name, and a crawler working through a
    # 228-URL sitemap does it in a burst -- which should be reading one build
    # of that aggregate rather than asking this process for it 228 times.
    SHARED_CACHE_SECONDS = 300

    def page(html: str) -> HTMLResponse:
        """A crawlable page, cacheable by anybody for five minutes.

        `_missing` deliberately does not go through here: a 404 falls back to
        whatever the CDN does with an unmarked one, which is a shorter window
        than this, and a page that appears in the corpus an hour from now
        should not be denied for five minutes because it did not exist when
        the first crawler asked.
        """
        res = HTMLResponse(html)
        http_cache.public(res, SHARED_CACHE_SECONDS)
        return res

    def data():
        return adp_data(conn)

    def index_page(position: str | None):
        d = data()
        players = d["players"] if position is None else [
            p for p in d["players"] if p["position"] == position]
        present = _present(d)
        season = d["updated"].year if d["updated"] else datetime.now().year
        shape = f"{d['teams']}-team PPR" if d["drafts"] else "PPR"
        if position is None:
            heading = f"ESPN Mock Draft ADP {season} ({shape})"
            path = "/adp"
            desc = (f"Average draft position of every player in {d['drafts']} real ESPN "
                    f"mock drafts ({shape}): ADP, typical range, and how often each is taken.")
            crumbs = _crumbs(("ADP", "/adp"))
            # Prose, on the one page here meant to answer the bare ADP query.
            # A table by itself gives a reader arriving cold nothing to read
            # and a search engine nothing to match beyond player names.
            intro = (
                f"Average draft position for every player taken in {d['drafts']} "
                "ESPN mock drafts, recorded pick by pick as they were played. "
                "These are not projections and not a blend of other sites' "
                "rankings: each number below is where ESPN drafters actually "
                "took him, and how far apart they disagreed."
                if d["drafts"] else
                "Average draft position for every player taken in the ESPN mock "
                "drafts this site records, pick by pick, as they are played.")
            faq = _index_faq(d)
        else:
            heading = f"{position} ADP – ESPN mock drafts {season}"
            path = f"/adp/{position.lower()}"
            desc = (f"Where every {position} goes in {d['drafts']} real ESPN mock drafts "
                    f"({shape}): ADP, range, and position rank.")
            crumbs = _crumbs(("ADP", "/adp"), (position, path))
            intro = (
                f"Where every {position} went in {d['drafts']} real ESPN mock "
                f"drafts ({shape}), ordered by average draft position. The range "
                "column is the 10th to the 90th percentile of his picks."
                if d["drafts"] else None)
            faq = None
        # Risers and fallers are recomputed for a position page rather than
        # filtered from the index's ten, or a page of tight ends would show
        # whichever two of them happened to make the whole board's list.
        if position is None:
            risers, fallers = d["risers"], d["fallers"]
        else:
            risers, fallers = _movers(players)
        return page(rendered(d["stamp"], ("index", position), lambda: render(
            "adp_index.html", title=_title(heading), description=desc,
            path=path, heading=heading, provenance=_provenance(d), players=players,
            rounds=d["rounds"], teams=d["teams"], position=position, positions=present,
            breadcrumbs=crumbs, intro=intro, faq=faq,
            faq_schema=_faq_schema(faq) if faq else None,
            risers=risers, fallers=fallers, runs=d["runs"], move=MOVE_PICKS,
            mover_floor=MOVER_MIN_SHARE,
            tiers=_tiers(players) if position else [])))

    @app.api_route("/adp", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def adp_index():
        return index_page(None)

    @app.api_route("/adp/round/{n}", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def adp_round(n: int):
        d = data()
        if not d["drafts"] or n < 1 or n > d["rounds"]:
            return _missing(f"/adp/round/{n}")
        teams = d["teams"]
        first, last = (n - 1) * teams + 1, n * teams
        rows = _round_rows(d, n)
        desc = (f"Who goes in round {n} (picks {first}–{last}) of an ESPN mock draft, "
                f"from {d['drafts']} recorded drafts")
        # The names, one at a time, for as many as the result will show.
        named: list = []
        for row in rows[:4]:
            if not row["usual"]:
                continue
            if len(_clip(desc, f": {', '.join(named + [row['p']['name']])}.")) > len(desc):
                named.append(row["p"]["name"])
        desc += f": {', '.join(named)}." if named else "."
        return page(rendered(d["stamp"], ("round", n), lambda: render(
            "adp_round.html", title=_title(f"Round {n} of an ESPN mock draft"),
            description=desc, path=f"/adp/round/{n}", n=n, first=first, last=last,
            rows=rows, mix=d["round_mix"].get(n, []), round_min=ROUND_MIN_SHARE,
            seats=_seats(d, n), drafts=d["drafts"], teams=teams,
            rounds=d["rounds"], provenance=_provenance(d), positions=_present(d),
            breadcrumbs=_crumbs(("ADP", "/adp"), (f"Round {n}", f"/adp/round/{n}")))))

    def render_player(target, d: dict, p: dict) -> str:
        """One player's page, built at most once per snapshot of the corpus.

        `target` is the connection the PROFILE is built on, which is not
        always the one the routes read: the keep-warm loop below renders the
        first forty of these on its own cursor, for the reason its own
        comment gives.

        THE PROFILE IS BUILT INSIDE `build`, WHICH `rendered` CALLS WITH
        NOTHING HELD. `cached_profile` costs ~250 ms on a first look at a
        player and takes one of `scoring.board_cache`'s two build permits;
        holding `_pages_lock` or `_adp_lock` across it would queue every
        other reader of every other page behind one player's dossier.
        """
        i = p["rank"] - 1
        near = [q for q in d["players"][max(0, i - 4): i + 5] if q is not p]
        pct = int(round(p["of_share"] * 100))
        desc = (f"{p['name']} ADP {p['adp']:.1f} in {d['drafts']} real ESPN mock drafts: "
                f"usually picks {p['p10']}–{p['p90']}, round {p['usual_round']}, "
                f"taken in {pct}% of drafts, {p['position']}{p['pos_rank']}.")
        if p.get("vs_espn") and abs(p["vs_espn"]) >= 1:
            way = "earlier" if p["vs_espn"] > 0 else "later"
            desc = _clip(desc, f" {abs(p['vs_espn']):.0f} picks {way} than ESPN's board.")
        # A DEFENSE IS NOT A HE. Every sentence on this page that would have
        # said "he" asks for one of these instead.
        pron = ({"s": "it", "o": "it", "p": "its"} if p["position"] == "DST"
                else {"s": "he", "o": "him", "p": "his"})
        # base.html hangs `og:image` on this, and a card image has to be at
        # least 200px square to be shown at all -- so the social tag asks for
        # a bigger one than the 128-pixel `img` on the page, off the same
        # url. `thumb` replaces a width it already set (scoring/headshot.py),
        # which is what makes that possible here without carrying the
        # original around.
        social = thumb(p["headshot"], 320)

        def build() -> str:
            # Every player on this board, so a comparable season or a board
            # peer who has a page of his own becomes a link and one who does
            # not stays text.
            slugs = {q["player_id"]: q["slug"] for q in d["players"]}
            prof = profile_view(target, p, slugs)
            return render(
                "adp_player.html",
                title=_title(f"{p['name']} ADP – ESPN mock drafts "
                             f"{d['updated'].year if d['updated'] else ''}".strip()),
                description=desc, path=f"/adp/{p['slug']}", p=p, drafts=d["drafts"],
                teams=d["teams"], rounds=d["rounds"], picks_total=d["teams"] * d["rounds"],
                near=near, headshot=social, face=thumb(p["headshot"], 256),
                round_min=ROUND_MIN_SHARE, pron=pron, mover_floor=MOVER_MIN_SHARE,
                corpus_seconds=d["seconds"], curve_picks=CURVE_PICKS,
                prof=prof, news=_merge_news(p.get("news"),
                                            prof["news"] if prof else []),
                provenance=_provenance(d), person_schema=_person(p, social),
                breadcrumbs=_crumbs(("ADP", "/adp"),
                                    (p["position"], f"/adp/{p['position'].lower()}"),
                                    (p["name"], f"/adp/{p['slug']}")))

        return rendered(d["stamp"], ("player", p["slug"]), build)

    @app.api_route("/adp/{key}", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def adp_player(key: str):
        if key.upper() in POSITIONS:
            return index_page(key.upper())
        d = data()
        p = d["by_slug"].get(key)
        if p is None:
            return _missing(f"/adp/{key}")
        # A CURSOR PER REQUEST, NEVER THE SHARED `conn`. `render_player`
        # builds the profile off whatever connection it is handed, and a
        # DuckDBPyConnection carries the statement and result state of the
        # query running on it -- two request threads issuing queries on one
        # connection do not queue, they overwrite each other. Six of these
        # at once (which is how a crawler walks a 228-URL sitemap, and how
        # `test_six_player_pages_at_once_each_get_their_whole_page` walks
        # it) left five pages with every profile section missing, wrote
        # those stripped pages into the page cache under the corpus's own
        # stamp, and reported it once per process and never again.
        #
        # `conn.cursor()` shares the database and gives this request its own
        # state -- the convention `api/main.py`'s handlers, its sim worker
        # and the keep-warm loop below all already follow.
        cur = conn.cursor() if conn is not None else None
        try:
            body = render_player(cur, d, p)
        finally:
            if cur is not None:
                cur.close()
        return page(body)

    @app.api_route("/sitemap.xml", methods=["GET", "HEAD"])
    def sitemap():
        d = data()

        def build() -> str:
            lastmod = d["updated"].isoformat() if d["updated"] else None
            urls = ["/", "/mocks", "/adp"]
            if d["drafts"]:
                urls += [f"/adp/{pos.lower()}" for pos in _present(d)]
                urls += [f"/adp/round/{n}" for n in range(1, d["rounds"] + 1)]
                urls += [f"/adp/{p['slug']}" for p in d["players"]]
            body = ['<?xml version="1.0" encoding="UTF-8"?>',
                    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
            for path in urls:
                body.append("<url><loc>%s%s</loc>%s</url>" % (
                    SITE, path, f"<lastmod>{lastmod}</lastmod>" if lastmod else ""))
            body.append("</urlset>")
            return "\n".join(body)

        res = Response(content=rendered(d["stamp"], ("sitemap",), build),
                       media_type="application/xml")
        http_cache.public(res, SHARED_CACHE_SECONDS)
        return res

    # A sitemap of 228 URLs does not get crawled one at a time -- it gets
    # crawled in a burst, and a cold `build_adp` (0.9-2.8s, and it can
    # trigger a board build besides) must not be the price whichever request
    # in that burst happens to land first. Warm it off the request thread;
    # `adp_data`'s own lock means a real request racing this thread waits for
    # the same answer rather than building its own.
    #
    # AND THEN KEEP WARMING IT. Warming once at registration fixed the first
    # ten minutes and nothing after: `adp_data`'s entry lives for
    # `market.CACHE_SECONDS`, and the reader who arrived after it lapsed paid
    # the rebuild. Measured, that was a 1.9s TTFB on `/adp` about every other
    # five-minute edge window. The loop below renews the entry from inside
    # its own lifetime, so nobody is ever the one who finds it cold.
    #
    # LAST, so a failure to warm can never cost the routes above it.
    if not WARM_ON_REGISTER:
        return

    # A dedicated cursor, not the shared `conn`, for this background thread:
    # `api/demo.py` registers its own warm thread against this same `conn`
    # (see `api/main.py`), and the two can run at the same moment. A
    # DuckDBPyConnection is not safe for concurrent queries from two threads
    # -- see `adp_data`'s docstring for what that corrupts -- and `_adp_lock`
    # only serializes callers that go through `adp_data`, not a different
    # module reading `conn` directly. `conn.cursor()` shares the underlying
    # database but gives this thread its own statement/result state, the
    # same convention `api/main.py`'s ~20 handlers and its own sim worker
    # already use for a background thread next to request handlers.
    cur = conn.cursor() if conn is not None else None

    def _warm_pages(d: dict) -> None:
        """The first `PROFILE_WARM` player pages, rendered before anybody
        asks for them.

        THE PROFILE IS WHAT MAKES THIS WORTH DOING. Every other page here is
        a template over an aggregate already in memory and renders in single
        milliseconds; a player page now also calls `cached_profile`, which is
        ~250 ms the first time a player is asked for and 20 ms after. Warming
        the aggregate and leaving the profiles cold would move the stall from
        `/adp` onto whichever player page a crawler happened to open first.

        BY ESPN'S OWN RANK, not by this corpus's ADP: the pages a reader
        arrives at from a search are the players a search engine has heard
        of, and ESPN's cheat sheet is the closest thing here to that order.
        Anybody it does not rank goes last, in board order.

        SEQUENTIAL, ON THIS ONE THREAD. `cached_profile` single-flights and
        `board_cache` hands out two build permits; forty of these in parallel
        would be forty threads contending for two permits and one DuckDB
        cursor, on a background job nobody is waiting for.

        Renders that hit the page cache cost nothing -- `rendered` keys on
        the same stamp `rebuild_adp` just computed, so a pass that finds the
        corpus unchanged does forty dictionary lookups.
        """
        players = sorted(d.get("players") or [],
                         key=lambda q: (q.get("espn_rank") is None,
                                        q.get("espn_rank") or 0, q["rank"]))
        started, done = time.time(), 0
        for player in players[:PROFILE_WARM]:
            try:
                render_player(cur, d, player)
                done += 1
            except Exception:      # noqa: BLE001 -- one page that will not
                # build is one page a reader pays for later, which is what
                # every page cost before this loop existed. The others in
                # the list are still worth warming.
                pass
        # One line a pass, not one a page: forty lines every four minutes
        # would bury everything else in the log.
        print(f"seo: warmed {done} of {min(len(players), PROFILE_WARM)} "
              f"player pages in {time.time() - started:.1f}s", flush=True)

    def _warm():
        while True:
            try:
                # `rebuild_adp` on every pass including the first, so the
                # boot warm cannot be a no-op against an entry some other
                # app object in this process left behind.
                _warm_pages(rebuild_adp(cur))
            except Exception:      # noqa: BLE001 -- a failed warm just means
                # the next request pays for `build_adp` itself, same as
                # before this loop existed. Never fatal to the thread: a
                # corpus mid-write is a 503 from `market._corpus`, and it
                # will be writable again long before the next pass.
                pass
            time.sleep(WARM_SECONDS)

    threading.Thread(target=_warm, name="seo-adp-warm", daemon=True).start()
