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
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from fastapi import HTTPException

from api import market
from pipeline.db import read_table

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


def _team_fallback_from_espn(conn) -> dict:
    """gsis_id -> team from `espn_adp` through the crosswalk, for players the
    depth charts do not carry (a rookie in August)."""
    if conn is None:
        return {}
    try:
        espn = read_table(conn, "espn_adp")
        sleeper = read_table(conn, "sleeper_ids")
        if espn.empty or sleeper.empty or "team" not in espn.columns:
            return {}
        joined = espn.merge(sleeper.dropna(subset=["gsis_id", "espn_id"])[["gsis_id", "espn_id"]],
                            on="espn_id")
        return {str(r.gsis_id): str(r.team) for r in joined.itertuples() if r.team == r.team and r.team}
    except Exception:      # noqa: BLE001 -- a board that cannot be read costs
        # a fallback team, not the page.
        return {}


def _empty() -> dict:
    return {"players": [], "by_slug": {}, "drafts": 0, "teams": 0, "rounds": 0, "updated": None}


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
    finally:
        corpus.close()
    total = int(total or 0)
    if total == 0:
        return _empty()

    picks: dict = {}
    positions: dict = {}
    for pid, pos, pick_no in rows:
        picks.setdefault(str(pid), []).append(int(pick_no))
        positions.setdefault(str(pid), str(pos or "").upper())

    names = market._names(conn)
    missing = {pid for pid in picks if pid not in names}
    if missing:
        names.update(market._board_names(conn, missing))
    teams_by = _teams_by_player(conn)
    espn_team = _team_fallback_from_espn(conn)

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
        players.append({
            "player_id": pid,
            "name": name,
            "headshot": headshot,
            "position": positions.get(pid) or "",
            "team": teams_by.get(pid) or espn_team.get(pid),
            "adp": round(mean(ordered), 1),
            "median": float(median(ordered)),
            "p10": _percentile(ordered, 0.10),
            "p90": _percentile(ordered, 0.90),
            "low": ordered[0],
            "high": ordered[-1],
            "taken": len(ordered),
            "share": round(share, 3),
            "round_mode": Counter((p - 1) // teams + 1 for p in ordered).most_common(1)[0][0],
            "hist": hist,
        })

    players.sort(key=lambda p: (p["adp"], -p["taken"], p["player_id"]))
    espn = _espn_adp_by_player(conn, {p["player_id"]: (p["name"], p["position"]) for p in players})
    pos_seen: Counter = Counter()
    for i, p in enumerate(players, start=1):
        p["rank"] = i
        pos_seen[p["position"]] += 1
        p["pos_rank"] = pos_seen[p["position"]]
        p["espn_adp"] = espn.get(p["player_id"])

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
            "teams": teams, "rounds": rounds, "updated": updated}


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
        return market._cached("seo-adp", lambda: build_adp(conn))


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


def _provenance(data: dict) -> str:
    if not data["drafts"]:
        return "No drafts recorded yet."
    when = data["updated"].strftime("%b %-d, %Y") if data["updated"] else "today"
    return (f"From {data['drafts']} real ESPN mock drafts "
            f"({data['teams']}-team PPR, {data['rounds']} rounds), updated {when}.")


def _crumbs(*items) -> dict:
    """`BreadcrumbList` structured data from (name, path) pairs."""
    return {"@context": "https://schema.org", "@type": "BreadcrumbList",
            "itemListElement": [
                {"@type": "ListItem", "position": i, "name": name, "item": SITE + path}
                for i, (name, path) in enumerate(items, start=1)]}


def round_players(data: dict, n: int) -> list:
    """Who goes in round `n`: everyone whose usual range crosses it, most
    often first, then by ADP."""
    teams = data["teams"]
    first, last = (n - 1) * teams + 1, n * teams
    hits = [p for p in data["players"]
            if p["round_mode"] == n or (p["p10"] <= last and p["p90"] >= first)]
    return sorted(hits, key=lambda p: (p["round_mode"] != n, p["adp"]))


def _missing(path: str):
    from fastapi.responses import HTMLResponse
    return HTMLResponse(render("adp_missing.html", title="Not found – ESPN Draft Assist",
                               description="No such page.", path=path, noindex=True),
                        status_code=404)


def register_seo_routes(app, conn=None):
    """The crawlable site: `/adp`, its player, round and position pages, and
    `/sitemap.xml`. Register before `api/static.register_spa`."""
    from fastapi.responses import HTMLResponse, Response

    def data():
        return adp_data(conn)

    def index_page(position: str | None):
        d = data()
        players = d["players"] if position is None else [
            p for p in d["players"] if p["position"] == position]
        present = [pos for pos in POSITIONS if pos in {p["position"] for p in d["players"]}]
        season = d["updated"].year if d["updated"] else datetime.now().year
        shape = f"{d['teams']}-team PPR" if d["drafts"] else "PPR"
        if position is None:
            heading = f"ESPN Mock Draft ADP {season} ({shape})"
            path = "/adp"
            desc = (f"Average draft position of every player in {d['drafts']} real ESPN "
                    f"mock drafts ({shape}): ADP, typical range, and how often each is taken.")
            crumbs = _crumbs(("ADP", "/adp"))
        else:
            heading = f"{position} ADP – ESPN mock drafts {season}"
            path = f"/adp/{position.lower()}"
            desc = (f"Where every {position} goes in {d['drafts']} real ESPN mock drafts "
                    f"({shape}): ADP, range, and position rank.")
            crumbs = _crumbs(("ADP", "/adp"), (position, path))
        return HTMLResponse(render(
            "adp_index.html", title=f"{heading} – ESPN Draft Assist", description=desc,
            path=path, heading=heading, provenance=_provenance(d), players=players,
            rounds=d["rounds"], position=position, positions=present, breadcrumbs=crumbs))

    @app.get("/adp", response_class=HTMLResponse)
    def adp_index():
        return index_page(None)

    @app.get("/adp/round/{n}", response_class=HTMLResponse)
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
        return HTMLResponse(render(
            "adp_round.html", title=f"Round {n} of an ESPN mock draft – who goes there – ESPN Draft Assist",
            description=desc, path=f"/adp/round/{n}", n=n, first=first, last=last,
            players=players, rounds=d["rounds"], provenance=_provenance(d),
            breadcrumbs=_crumbs(("ADP", "/adp"), (f"Round {n}", f"/adp/round/{n}"))))

    @app.get("/adp/{key}", response_class=HTMLResponse)
    def adp_player(key: str):
        if key.upper() in POSITIONS:
            return index_page(key.upper())
        d = data()
        p = d["by_slug"].get(key)
        if p is None:
            return _missing(f"/adp/{key}")
        i = p["rank"] - 1
        near = [q for q in d["players"][max(0, i - 4): i + 5] if q is not p]
        pct = int(round(p["share"] * 100))
        desc = (f"{p['name']} ADP {p['adp']:.1f} in {d['drafts']} real ESPN mock drafts: "
                f"usually picks {p['p10']}–{p['p90']}, round {p['round_mode']}, "
                f"taken in {pct}% of drafts, {p['position']}{p['pos_rank']}.")
        return HTMLResponse(render(
            "adp_player.html", title=f"{p['name']} ADP – ESPN mock drafts {d['updated'].year if d['updated'] else ''} – ESPN Draft Assist",
            description=desc, path=f"/adp/{p['slug']}", p=p, drafts=d["drafts"],
            teams=d["teams"], rounds=d["rounds"], picks_total=d["teams"] * d["rounds"],
            peak=max(p["hist"]) or 1, near=near, headshot=p["headshot"],
            provenance=_provenance(d),
            breadcrumbs=_crumbs(("ADP", "/adp"), (p["position"], f"/adp/{p['position'].lower()}"),
                                (p["name"], f"/adp/{p['slug']}"))))

    @app.get("/sitemap.xml")
    def sitemap():
        d = data()
        stamp = d["updated"].isoformat() if d["updated"] else None
        urls = ["/", "/mocks", "/adp"]
        if d["drafts"]:
            present = [pos for pos in POSITIONS if pos in {p["position"] for p in d["players"]}]
            urls += [f"/adp/{pos.lower()}" for pos in present]
            urls += [f"/adp/round/{n}" for n in range(1, d["rounds"] + 1)]
            urls += [f"/adp/{p['slug']}" for p in d["players"]]
        body = ['<?xml version="1.0" encoding="UTF-8"?>',
                '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
        for path in urls:
            body.append("<url><loc>%s%s</loc>%s</url>" % (
                SITE, path, f"<lastmod>{stamp}</lastmod>" if stamp else ""))
        body.append("</urlset>")
        return Response(content="\n".join(body), media_type="application/xml")

    # A sitemap of 232 URLs does not get crawled one at a time -- it gets
    # crawled in a burst, and a cold `build_adp` (~2s, and it can trigger a
    # board build besides) must not be the price whichever request in that
    # burst happens to land first. Warm it once, off the request thread, at
    # registration time; `adp_data`'s own lock means a real request racing
    # this thread waits for the same answer rather than building its own.
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
        try:
            adp_data(cur)
        except Exception:      # noqa: BLE001 -- a failed warm just means the
            # first request pays for `build_adp` itself, same as before.
            pass

    threading.Thread(target=_warm, name="seo-adp-warm", daemon=True).start()
