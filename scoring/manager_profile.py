"""A manager's profile, from the league's own history.

Pure functions over a league database. No network, no model. Every
section returns plain dicts and lists so the API can serialise them as
they are, and every section carries `n` -- the number of observations it
rests on -- so the page can decline to put an adjective on three data
points.

The manager key is the ESPN member id (SWID). Teams are season-scoped and
resolve through `league_members`, which is also how `league_standings`
(keyed on team) is read here.
"""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from pipeline.db import read_table

PLAYOFF_TIERS = {"WINNERS_BRACKET"}
CONSOLATION_TIERS = {"WINNERS_CONSOLATION_LADDER", "LOSERS_CONSOLATION_LADDER"}


def _num(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _int(value):
    if value is None or pd.isna(value):
        return None
    return int(value)


def members(conn) -> pd.DataFrame:
    """One row per member across seasons: `member_id`, `display_name` (the
    newest season's), `seasons` (ascending), `teams` ({season: team_id})."""
    m = read_table(conn, "league_members")
    if m.empty:
        return pd.DataFrame(columns=["member_id", "display_name", "seasons", "teams"])
    rows = []
    for swid, g in m.sort_values("season").groupby("member_id"):
        played = g[g.team_id.notna()]
        rows.append({
            "member_id": swid,
            "display_name": g.iloc[-1].display_name,
            "first_name": g.iloc[-1].first_name,
            "seasons": [int(s) for s in played.season],
            "teams": {int(r.season): int(r.team_id) for r in played.itertuples()},
        })
    return pd.DataFrame(rows)


def _team_of(conn) -> dict:
    """(season, team_id) -> member_id, for joining team-keyed tables."""
    m = read_table(conn, "league_members")
    if m.empty:
        return {}
    return {(int(r.season), int(r.team_id)): r.member_id
            for r in m[m.team_id.notna()].itertuples()}


def _teams_for(conn, member_id: str) -> dict:
    """season -> team_id for one member."""
    return {season: team for (season, team), swid in _team_of(conn).items()
            if swid == member_id}


def finishes(conn, member_id: str) -> dict:
    mine = _teams_for(conn, member_id)
    st = read_table(conn, "league_standings")
    mem = read_table(conn, "league_members")
    seasons = []
    for season in sorted(mine):
        row = st[(st.season == season) & (st.team_id == mine[season])]
        me = mem[(mem.season == season) & (mem.member_id == member_id)]
        if row.empty:
            continue
        r = row.iloc[0]
        projected = _int(me.iloc[0].draft_day_rank) if not me.empty else None
        final = _int(r.final_rank)
        seasons.append({
            "season": int(season), "team_name": r.team_name,
            "wins": _int(r.wins), "losses": _int(r.losses), "ties": _int(r.ties),
            "points_for": _num(r.points_for), "points_against": _num(r.points_against),
            "playoff_seed": _int(r.playoff_seed), "final_rank": final,
            "draft_day_rank": projected,
            "outperformance": (projected - final) if projected is not None and final is not None else None,
        })
    done = [s for s in seasons if s["final_rank"] is not None]
    games = sum((s["wins"] or 0) + (s["losses"] or 0) + (s["ties"] or 0) for s in done)
    wins = sum(s["wins"] or 0 for s in done)
    points = [s["points_for"] for s in done if s["points_for"] is not None]
    return {
        "n": len(done),
        "seasons": seasons,
        "titles": [s["season"] for s in done if s["final_rank"] == 1],
        "avg_finish": round(sum(s["final_rank"] for s in done) / len(done), 2) if done else None,
        "win_pct": round(wins / games, 3) if games else None,
        "points_per_season": round(sum(points) / len(points), 1) if points else None,
        "outperformance": round(sum(s["outperformance"] for s in done
                                    if s["outperformance"] is not None), 1) if done else None,
    }


def _matchups(conn) -> pd.DataFrame:
    m = read_table(conn, "league_matchups")
    return m if not m.empty else pd.DataFrame(columns=[
        "season", "matchup_period", "tier", "home_team_id", "away_team_id",
        "home_points", "away_points", "winner"])


def _sides(conn):
    """Every matchup as two rows, one per side: member, opponent, points,
    opponent points, won/lost/tied, tier. Byes are dropped."""
    owner = _team_of(conn)
    rows = []
    for r in _matchups(conn).itertuples():
        if pd.isna(r.away_team_id) or pd.isna(r.home_team_id):
            continue
        season = int(r.season)
        home = owner.get((season, int(r.home_team_id)))
        away = owner.get((season, int(r.away_team_id)))
        if home is None or away is None:
            continue
        hp, ap = _num(r.home_points) or 0.0, _num(r.away_points) or 0.0
        for me, them, pf, pa in ((home, away, hp, ap), (away, home, ap, hp)):
            result = "T" if r.winner == "TIE" else ("W" if pf > pa else "L")
            if r.winner == "UNDECIDED":
                continue
            rows.append({"season": season, "period": int(r.matchup_period), "tier": r.tier,
                         "member_id": me, "opponent_id": them, "points": pf,
                         "against": pa, "result": result})
    return pd.DataFrame(rows, columns=["season", "period", "tier", "member_id",
                                       "opponent_id", "points", "against", "result"])


def playoffs(conn, member_id: str) -> dict:
    sides = _sides(conn)
    mine = sides[sides.member_id == member_id]
    bracket = mine[mine.tier.isin(PLAYOFF_TIERS)]
    consolation = mine[mine.tier.isin(CONSOLATION_TIERS)]
    st = read_table(conn, "league_standings")
    teams = _teams_for(conn, member_id)
    last, titles = [], []
    for season, team in teams.items():
        row = st[(st.season == season) & (st.team_id == team)]
        if row.empty or pd.isna(row.iloc[0].final_rank):
            continue
        size = int(st[st.season == season].team_id.nunique())
        if int(row.iloc[0].final_rank) == size:
            last.append(int(season))
        if int(row.iloc[0].final_rank) == 1:
            titles.append(int(season))
    return {
        "n": int(len(bracket) + len(consolation)),
        "appearances": sorted(int(s) for s in bracket.season.unique()),
        "bracket_wins": int((bracket.result == "W").sum()),
        "bracket_losses": int((bracket.result == "L").sum()),
        "consolation": {"wins": int((consolation.result == "W").sum()),
                        "losses": int((consolation.result == "L").sum())},
        "titles": sorted(titles),
        "last_place": sorted(last),
    }


def head_to_head(conn) -> dict:
    """(member, opponent) -> games, wins, losses, ties, margin, playoff_games."""
    out = defaultdict(lambda: {"games": 0, "wins": 0, "losses": 0, "ties": 0,
                               "margin": 0.0, "playoff_games": 0})
    for r in _sides(conn).itertuples():
        rec = out[(r.member_id, r.opponent_id)]
        rec["games"] += 1
        rec["wins"] += r.result == "W"
        rec["losses"] += r.result == "L"
        rec["ties"] += r.result == "T"
        rec["margin"] += r.points - r.against
        rec["playoff_games"] += r.tier in PLAYOFF_TIERS
    for rec in out.values():
        rec["margin"] = round(rec["margin"], 2)
        rec["n"] = rec["games"]
    return dict(out)


def luck(conn, member_id: str) -> dict:
    """All-play: each regular-season week, my points against every other
    team's that week. Expected wins is the all-play win fraction times
    games played; luck is actual minus expected."""
    sides = _sides(conn)
    regular = sides[sides.tier == "NONE"]
    seasons = []
    total_n = 0
    for season, g in regular.groupby("season"):
        mine = g[g.member_id == member_id]
        if mine.empty:
            continue
        allplay_w = allplay_n = 0
        for r in mine.itertuples():
            others = g[(g.period == r.period) & (g.member_id != member_id)]
            allplay_w += int((others.points < r.points).sum()) + 0.5 * int((others.points == r.points).sum())
            allplay_n += len(others)
        games = len(mine)
        actual = int((mine.result == "W").sum()) + 0.5 * int((mine.result == "T").sum())
        expected = games * (allplay_w / allplay_n) if allplay_n else 0.0
        # Not rounded here: rounding an all-play fraction to 2dp (e.g. 1/3 ->
        # 0.33) throws away exactly the precision a caller comparing against
        # the exact fraction needs. Round for display, not for storage.
        seasons.append({"season": int(season), "games": games, "actual_wins": actual,
                        "expected_wins": expected, "luck": actual - expected})
        total_n += games
    return {"n": total_n, "seasons": seasons,
            "luck": round(sum(s["luck"] for s in seasons), 2) if seasons else None}


def _name(conn, member_id: str) -> str:
    m = members(conn)
    hit = m[m.member_id == member_id]
    return str(hit.iloc[0].display_name) if not hit.empty else member_id


def seasons_strip(conn) -> list:
    """Per season: champion, runner-up, last place, top scorer."""
    st = read_table(conn, "league_standings")
    owner = _team_of(conn)
    out = []
    if st.empty:
        return out
    for season, g in st.sort_values("season").groupby("season"):
        season = int(season)

        def who(team_id):
            swid = owner.get((season, int(team_id)))
            return {"member_id": swid, "display_name": _name(conn, swid) if swid else None}

        ranked = g[g.final_rank.notna()].sort_values("final_rank")
        scored = g[g.points_for.notna()].sort_values("points_for", ascending=False)
        out.append({
            "season": season,
            "teams": int(g.team_id.nunique()),
            "complete": not ranked.empty,
            "champion": who(ranked.iloc[0].team_id) if not ranked.empty else None,
            "runner_up": who(ranked.iloc[1].team_id) if len(ranked) > 1 else None,
            "last": who(ranked.iloc[-1].team_id) if not ranked.empty else None,
            "top_scorer": who(scored.iloc[0].team_id) if not scored.empty else None,
        })
    return out
