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

import os
import re
import threading
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median

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

# How many rendered pages to keep. The sitemap is 232 URLs and a crawler
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

# Every field a player MAY have, and what "we could not answer that" looks
# like. Set on every player before a page is rendered, so a template can ask
# for any of them without guarding, and so the shape of a player is written
# down in one place rather than inferred from nine sources that each fill in
# a few of it. A blank here is the answer for anybody the crosswalks miss --
# see scoring/adp_facts.py on why it is never somebody else's number.
BLANKS = {
    "bye": None, "tier": None, "fp_tier": None, "consensus": None,
    "espn_rank": None, "espn_pick": None, "espn_order": None, "vs_espn": None,
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
    An identical stamp means the 232 rendered pages hanging off the last one
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

    names = market._names(conn)
    missing = {pid for pid in picks if pid not in names}
    if missing:
        names.update(market._board_names(conn, missing))
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
            "round_mode": Counter((p - 1) // teams + 1 for p in ordered).most_common(1)[0][0],
            "round_low": (ordered[0] - 1) // teams + 1,
            "round_p10": (_percentile(ordered, 0.10) - 1) // teams + 1,
            "round_p90": (_percentile(ordered, 0.90) - 1) // teams + 1,
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
        if not p["team"]:
            p["team"] = espn_team.get(p["player_id"])

    # Everything the universal database knows about these two hundred names:
    # bye, tier, the five consensus sources, the season history, last season,
    # this season's projection, the sportsbook, the injury report, the news.
    # One pass over the whole list -- see scoring/adp_facts.py.
    try:
        from scoring import adp_facts
        adp_facts.attach(conn, players)
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
    rank stands in for the handful it does not reach.
    """
    covered = [p for p in players
               if p.get("espn_adp") is not None or p.get("espn_rank") is not None]
    if len(covered) < 2:
        return
    # `players` is already in this corpus's own draft order, so the picks
    # come out ascending without another sort.
    picks = [p["adp"] for p in covered]
    order = sorted(covered, key=lambda p: (p["espn_adp"] if p.get("espn_adp") is not None
                                           else float(p["espn_rank"]), p["adp"]))
    for i, p in enumerate(order):
        p["espn_pick"] = picks[i]
        p["espn_order"] = i + 1
        p["vs_espn"] = round(picks[i] - p["adp"], 1)


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
    232 pages for every crawl that walks the sitemap.

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
            node["affiliation"] = {"@type": "SportsTeam", "name": p["team"],
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


def round_players(data: dict, n: int) -> list:
    """Who goes in round `n`: everyone whose usual range crosses it, most
    often first, then by ADP."""
    teams = data["teams"]
    first, last = (n - 1) * teams + 1, n * teams
    hits = [p for p in data["players"]
            if p["round_mode"] == n or (p["p10"] <= last and p["p90"] >= first)]
    return sorted(hits, key=lambda p: (p["round_mode"] != n, p["adp"]))


def _round_rows(data: dict, n: int) -> list:
    """The round's players, each with how often this round is where he went.

    Straight out of the histogram: the picks that make up round `n` are a
    contiguous slice of it, and their sum over the drafts he was on the board
    for is the share the page wants. No second query, and the same
    denominator the rest of the site uses.
    """
    teams = data["teams"]
    first, last = (n - 1) * teams + 1, n * teams
    rows = []
    for p in round_players(data, n):
        count = sum(p["hist"][first - 1:last])
        rows.append({"p": p, "count": count,
                     "share": round(count / p["of"], 3) if p["of"] else 0.0,
                     "usual": p["round_mode"] == n})
    return rows


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
    # 232-URL sitemap does it in a burst -- which should be reading one build
    # of that aggregate rather than asking this process for it 232 times.
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
            "adp_index.html", title=f"{heading} – ESPN Draft Assist", description=desc,
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
        players = round_players(d, n)
        usual = [p["name"] for p in players if p["round_mode"] == n][:4]
        desc = (f"Who goes in round {n} (picks {first}–{last}) of an ESPN mock draft, "
                f"from {d['drafts']} recorded drafts"
                + (f": {', '.join(usual)}." if usual else "."))
        return page(rendered(d["stamp"], ("round", n), lambda: render(
            "adp_round.html", title=f"Round {n} of an ESPN mock draft – who goes there – ESPN Draft Assist",
            description=desc, path=f"/adp/round/{n}", n=n, first=first, last=last,
            rows=_round_rows(d, n), mix=d["round_mix"].get(n, []),
            seats=_seats(d, n), drafts=d["drafts"], teams=teams,
            rounds=d["rounds"], provenance=_provenance(d), positions=_present(d),
            breadcrumbs=_crumbs(("ADP", "/adp"), (f"Round {n}", f"/adp/round/{n}")))))

    @app.api_route("/adp/{key}", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def adp_player(key: str):
        if key.upper() in POSITIONS:
            return index_page(key.upper())
        d = data()
        p = d["by_slug"].get(key)
        if p is None:
            return _missing(f"/adp/{key}")
        i = p["rank"] - 1
        near = [q for q in d["players"][max(0, i - 4): i + 5] if q is not p]
        pct = int(round(p["of_share"] * 100))
        desc = (f"{p['name']} ADP {p['adp']:.1f} in {d['drafts']} real ESPN mock drafts: "
                f"usually picks {p['p10']}–{p['p90']}, round {p['round_mode']}, "
                f"taken in {pct}% of drafts, {p['position']}{p['pos_rank']}.")
        if p.get("vs_espn") and abs(p["vs_espn"]) >= 1:
            way = "earlier" if p["vs_espn"] > 0 else "later"
            desc += f" {abs(p['vs_espn']):.0f} picks {way} than ESPN's board."
        # base.html hangs `og:image` on this, and a card image has to be at
        # least 200px square to be shown at all -- so the social tag asks for
        # a bigger one than the 128-pixel `img` on the page, off the same
        # url. `thumb` replaces a width it already set (scoring/headshot.py),
        # which is what makes that possible here without carrying the
        # original around.
        social = thumb(p["headshot"], 320)
        return page(rendered(d["stamp"], ("player", p["slug"]), lambda: render(
            "adp_player.html", title=f"{p['name']} ADP – ESPN mock drafts {d['updated'].year if d['updated'] else ''} – ESPN Draft Assist",
            description=desc, path=f"/adp/{p['slug']}", p=p, drafts=d["drafts"],
            teams=d["teams"], rounds=d["rounds"], picks_total=d["teams"] * d["rounds"],
            near=near, headshot=social, face=thumb(p["headshot"], 256),
            corpus_seconds=d["seconds"], curve_picks=CURVE_PICKS,
            provenance=_provenance(d), person_schema=_person(p, social),
            breadcrumbs=_crumbs(("ADP", "/adp"), (p["position"], f"/adp/{p['position'].lower()}"),
                                (p["name"], f"/adp/{p['slug']}")))))

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

    # A sitemap of 232 URLs does not get crawled one at a time -- it gets
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

    def _warm():
        while True:
            try:
                # `rebuild_adp` on every pass including the first, so the
                # boot warm cannot be a no-op against an entry some other
                # app object in this process left behind.
                rebuild_adp(cur)
            except Exception:      # noqa: BLE001 -- a failed warm just means
                # the next request pays for `build_adp` itself, same as
                # before this loop existed. Never fatal to the thread: a
                # corpus mid-write is a 503 from `market._corpus`, and it
                # will be writable again long before the next pass.
                pass
            time.sleep(WARM_SECONDS)

    threading.Thread(target=_warm, name="seo-adp-warm", daemon=True).start()
