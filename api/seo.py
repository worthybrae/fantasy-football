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
    except Exception:      # noqa: BLE001 -- a board that cannot be read costs teams
        return {}
    if dc.empty or not {"gsis_id", "team", "dt"}.issubset(dc.columns):
        return {}
    latest = dc.dropna(subset=["gsis_id"]).sort_values("dt").drop_duplicates("gsis_id", keep="last")
    return {str(r.gsis_id): str(r.team) for r in latest.itertuples() if r.team}


def _espn_adp_by_player(conn, ids: dict) -> dict:
    """gsis_id -> ESPN's published ADP. Crosswalk first, then name and
    position, the two-step every ESPN join in this codebase makes.
    `ids` maps player_id -> (name, position)."""
    if conn is None:
        return {}
    try:
        espn = read_table(conn, "espn_adp")
        sleeper = read_table(conn, "sleeper_ids")
    except Exception:      # noqa: BLE001
        return {}
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


def _team_fallback_from_espn(conn) -> dict:
    """gsis_id -> team from `espn_adp` through the crosswalk, for players the
    depth charts do not carry (a rookie in August)."""
    if conn is None:
        return {}
    try:
        espn = read_table(conn, "espn_adp")
        sleeper = read_table(conn, "sleeper_ids")
    except Exception:      # noqa: BLE001
        return {}
    if espn.empty or sleeper.empty or "team" not in espn.columns:
        return {}
    joined = espn.merge(sleeper.dropna(subset=["gsis_id", "espn_id"])[["gsis_id", "espn_id"]],
                        on="espn_id")
    return {str(r.gsis_id): str(r.team) for r in joined.itertuples() if r.team == r.team and r.team}


def _empty() -> dict:
    return {"players": [], "by_slug": {}, "drafts": 0, "teams": 0, "rounds": 0, "updated": None}


def build_adp(conn) -> dict:
    """Every player's draft position out of the corpus, in one pass.

    Every pick counts -- the autodrafter's and the farm's own seat's
    included -- because the question is where a player GOES, not what
    people think. The archive filters those out for its behavioural
    questions; ADP is the other kind of question.
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
        ordered = sorted(taken)
        entry = names.get(pid) or {}
        name = entry.get("name") or pid
        hist = [0] * (teams * rounds)
        for p in ordered:
            hist[p - 1] += 1
        players.append({
            "player_id": pid,
            "name": name,
            "headshot": entry.get("headshot"),
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


def adp_data(conn) -> dict:
    """`build_adp`, once per CACHE_SECONDS, shared by every page."""
    return market._cached("seo-adp", lambda: build_adp(conn))
