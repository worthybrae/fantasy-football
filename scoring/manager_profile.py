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
from pipeline.espn_league import (ESPN_BENCH_SLOT, ESPN_FLEX_SLOT, ESPN_IR_SLOT,
                                  ESPN_SLOT_POSITIONS)

PLAYOFF_TIERS = {"WINNERS_BRACKET"}
CONSOLATION_TIERS = {"WINNERS_CONSOLATION_LADDER", "LOSERS_CONSOLATION_LADDER"}
FLEX_POSITIONS = {"RB", "WR", "TE"}


def _num(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _int(value):
    if value is None or pd.isna(value):
        return None
    return int(value)


def _str(value):
    if value is None or pd.isna(value):
        return None
    return str(value)


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
            "display_name": _str(g.iloc[-1].display_name),
            "first_name": _str(g.iloc[-1].first_name),
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
    # `st` is a columnless frame when `league_standings` does not exist yet
    # (a league whose activity has been walked but never drafted) --
    # `st.season` below would raise AttributeError on that shape, so there
    # is nothing to loop over rather than nothing found per season.
    for season in sorted(mine) if not st.empty else []:
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
    # Same guard as `finishes`: a columnless `st` (no `league_standings`
    # table yet) has no `.season`/`.team_id` to filter on.
    for season, team in (teams.items() if not st.empty else []):
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


WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _txns(conn) -> pd.DataFrame:
    t = read_table(conn, "league_transactions")
    if t.empty:
        return pd.DataFrame(columns=["season", "week", "txn_id", "related_txn_id", "team_id",
                                     "member_id", "type", "status", "execution_type",
                                     "bid_amount", "proposed_at", "items_json", "items"])
    t = t.copy()
    t["items"] = t.items_json.map(lambda s: json.loads(s) if isinstance(s, str) else [])
    return t


def player_names(conn) -> dict:
    """player id -> the name the roster gave it, from any week it was rostered."""
    lu = read_table(conn, "league_lineups")
    if lu.empty:
        return {}
    named = lu[lu.player_name.notna()].drop_duplicates("player_id", keep="last")
    return {int(r.player_id): str(r.player_name) for r in named.itertuples()}


def _faab(conn) -> set:
    """Seasons whose settings used an acquisition budget. Read from the raw
    mSettings answer when it is stored; absent otherwise (or when the
    league database has no `league_raw` table at all, as in tests)."""
    raw = read_table(conn, "league_raw")
    out = set()
    if raw.empty:
        return out
    for r in raw[raw.view == "mSettings"].itertuples():
        settings = (json.loads(r.payload_json).get("settings") or {}).get("acquisitionSettings") or {}
        if settings.get("isUsingAcquisitionBudget"):
            out.add(int(r.season))
    return out


def waivers(conn, member_id: str) -> dict:
    t = _txns(conn)
    mine = t[t.member_id == member_id]
    claims = mine[(mine.type == "WAIVER") & (mine.execution_type == "EXECUTE")]
    outcomes = mine[(mine.type == "WAIVER") & (mine.execution_type != "EXECUTE")]
    won = outcomes[outcomes.status == "EXECUTED"]
    lost = outcomes[outcomes.status.str.startswith("FAILED", na=False)]
    canceled = outcomes[outcomes.status == "CANCELED"]
    fa = mine[(mine.type == "FREEAGENT") & (mine.status == "EXECUTED")]
    # Count DROP items across executed free-agent adds and won waiver claims
    # (a claim or an FA add can carry a drop alongside it in the same items list).
    drops = 0
    for items in pd.concat([fa["items"], won["items"]]):
        for item in items:
            if item.get("type") == "DROP":
                drops += 1
    by_day = {d: 0 for d in WEEKDAYS}
    for ts in pd.concat([fa.proposed_at, claims.proposed_at]):
        if pd.notna(ts):
            by_day[WEEKDAYS[pd.Timestamp(ts).dayofweek]] += 1
    moves = pd.concat([fa, won, lost]).groupby(["season", "week"]).size()
    busiest = None
    if len(moves):
        (season, week), count = moves.idxmax(), int(moves.max())
        busiest = {"season": int(season), "week": int(week), "moves": count}
    out = {
        "n": int(len(claims)),
        "claims": int(len(claims)), "won": int(len(won)), "lost": int(len(lost)),
        "canceled": int(len(canceled)),
        "win_rate": round(len(won) / (len(won) + len(lost)), 3) if (len(won) + len(lost)) else None,
        "free_agent_adds": int(len(fa)), "drops": int(drops),
        "adds_by_weekday": by_day, "busiest_week": busiest,
        "seasons": sorted(int(s) for s in mine.season.unique()),
    }
    faab = _faab(conn)
    bids = claims[claims.season.isin(faab)] if faab else claims.iloc[0:0]
    if len(bids):
        out["bids"] = {"n": int(len(bids)), "mean": round(float(bids.bid_amount.mean()), 1),
                       "max": float(bids.bid_amount.max()),
                       "spent": float(won[won.season.isin(faab)].bid_amount.sum())}
    return out


def _rest_of_season_points(lu: pd.DataFrame, season: int, after_week: int,
                           player_id: int, team_id: int) -> float:
    rows = lu[(lu.season == season) & (lu.week > after_week) & (lu.player_id == player_id)
              & (lu.team_id == team_id) & (lu.actual_points.notna())]
    return float(rows.actual_points.sum())


def trades(conn, member_id: str) -> dict:
    t = _txns(conn)
    teams = _teams_for(conn, member_id)
    lu = read_table(conn, "league_lineups")
    names = player_names(conn)
    display = {r.member_id: r.display_name for r in members(conn).itertuples()}
    owner = _team_of(conn)
    proposals = t[t.type == "TRADE_PROPOSAL"]

    def counterpart(items, season):
        for i in items:
            for key in ("toTeamId", "fromTeamId"):
                tid = i.get(key)
                if tid and tid != teams.get(season):
                    return owner.get((season, int(tid)))
        return None

    def involves(row):
        return any(teams.get(int(row.season)) in (i.get("fromTeamId"), i.get("toTeamId"))
                   for i in row["items"])

    mine_p = proposals[proposals.member_id == member_id]
    received = proposals[(proposals.member_id != member_id) & proposals.apply(involves, axis=1)] \
        if len(proposals) else proposals
    accepts = t[(t.type == "TRADE_ACCEPT")]
    declines = t[t.type == "TRADE_DECLINE"]
    vetoes = t[t.type == "TRADE_VETO"]
    my_ids = set(mine_p.txn_id)
    their_ids = set(received.txn_id)
    # ESPN's TRADE_ACCEPT rows carry status None in some seasons and
    # EXECUTED in others; both mean the trade went through.
    executed = accepts[accepts.related_txn_id.isin(my_ids | their_ids)
                       & (accepts.status.isna() | (accepts.status == "EXECUTED"))]
    partners = defaultdict(int)
    ledger = []
    for r in executed.itertuples():
        season = int(r.season)
        my_team = teams.get(season)
        items = r.items
        # `with`/`with_name` below keep only the first counterpart -- ESPN
        # trades are two-team in practice -- but `partners` must count every
        # distinct counterpart team the items actually touch, once per trade.
        other = counterpart(items, season)
        other_teams = {int(tid) for i in items for tid in (i.get("fromTeamId"), i.get("toTeamId"))
                       if tid and tid != my_team}
        for tid in other_teams:
            partners[owner.get((season, tid))] += 1
        sent = [i for i in items if i.get("fromTeamId") == my_team]
        got = [i for i in items if i.get("toTeamId") == my_team]
        week = int(r.week)
        # Each sent item's own toTeamId is the team that player now scores
        # for -- not just the first sent item's, in case a trade sends
        # players to more than one team.
        balance = (sum(_rest_of_season_points(lu, season, week, int(i["playerId"]), my_team) for i in got)
                   - sum(_rest_of_season_points(lu, season, week, int(i["playerId"]), int(i["toTeamId"])) for i in sent))
        ledger.append({
            "season": season, "week": week, "with": other,
            "with_name": display.get(other, other),
            "sent": [{"player_id": int(i["playerId"]), "name": names.get(int(i["playerId"]), str(i["playerId"]))} for i in sent],
            "received": [{"player_id": int(i["playerId"]), "name": names.get(int(i["playerId"]), str(i["playerId"]))} for i in got],
            "balance": round(balance, 1),
        })
    ledger.sort(key=lambda x: (x["season"], x["week"]), reverse=True)
    return {
        "n": len(ledger),
        "proposed": int(len(mine_p)), "received": int(len(received)),
        "accepted": len(ledger),
        "declined_by_me": int(declines[declines.member_id == member_id].related_txn_id.isin(their_ids).sum()),
        "declined_by_them": int(declines[declines.member_id != member_id].related_txn_id.isin(my_ids).sum()),
        "vetoed": int(vetoes.related_txn_id.isin(my_ids | their_ids).sum()),
        "partners": [{"member_id": m, "display_name": display.get(m, m), "trades": c}
                     for m, c in sorted(partners.items(), key=lambda kv: -kv[1])],
        "balance": round(sum(l["balance"] for l in ledger), 1) if ledger else None,
        "ledger": ledger,
    }


def _slot_counts(conn, season: int) -> dict:
    raw = read_table(conn, "league_raw")
    if raw.empty:
        return {}
    hit = raw[(raw.view == "mSettings") & (raw.season == season)]
    if hit.empty:
        return {}
    settings = json.loads(hit.iloc[0].payload_json).get("settings") or {}
    return (settings.get("rosterSettings") or {}).get("lineupSlotCounts") or {}


def optimal_points(entries: list, slot_counts: dict) -> float:
    """The most a roster could have scored that week under the league's
    starting slots. `entries` is [(position, points)]. Fixed slots take
    their position's best scorers first; FLEX takes the best of what is
    left among RB/WR/TE. Greedy is exact here because every fixed slot
    admits one position and FLEX is filled last from the remainder."""
    pool = sorted(((pts, pos) for pos, pts in entries if pts is not None), reverse=True)
    total = 0.0
    for slot_id, count in slot_counts.items():
        pos = ESPN_SLOT_POSITIONS.get(int(slot_id))
        if pos is None:
            continue
        for _ in range(int(count)):
            pick = next((e for e in pool if e[1] == pos), None)
            if pick is None:
                break
            pool.remove(pick)
            total += pick[0]
    for _ in range(int(slot_counts.get(str(ESPN_FLEX_SLOT), 0))):
        pick = next((e for e in pool if e[1] in FLEX_POSITIONS), None)
        if pick is None:
            break
        pool.remove(pick)
        total += pick[0]
    return round(total, 2)


def lineups(conn, member_id: str) -> dict:
    lu = read_table(conn, "league_lineups")
    teams = _teams_for(conn, member_id)
    mem = read_table(conn, "league_members")
    weeks = []
    started_out = 0
    for season, team in teams.items():
        slots = _slot_counts(conn, season)
        rows = lu[(lu.season == season) & (lu.team_id == team)] if not lu.empty else lu
        if rows.empty or not slots:
            continue
        for week, g in rows.groupby("week"):
            scored = g[g.actual_points.notna()]
            if scored.empty:
                continue
            starters = scored[~scored.lineup_slot.isin([ESPN_BENCH_SLOT, ESPN_IR_SLOT])]
            started = float(starters.actual_points.sum())
            optimal = optimal_points([(r.position, float(r.actual_points))
                                      for r in scored.itertuples()], slots)
            started_out += int(starters.injury_status.isin(["OUT", "IR", "SUSPENSION"]).sum())
            weeks.append({"season": int(season), "week": int(week), "started": round(started, 2),
                          "optimal": optimal, "left": round(max(0.0, optimal - started), 2)})
    n = len(weeks)
    moves = mem[(mem.member_id == member_id) & mem.lineup_moves.notna()]
    moves_per_week = None
    if not moves.empty and n:
        seasons = {w["season"] for w in weeks}
        per_season = {int(r.season): int(r.lineup_moves) for r in moves.itertuples()}
        counted = [per_season[s] for s in seasons if s in per_season]
        weeks_in = sum(1 for w in weeks if w["season"] in per_season)
        moves_per_week = round(sum(counted) / weeks_in, 1) if weeks_in else None
    worst = max(weeks, key=lambda w: w["left"]) if weeks else None
    return {
        "n": n,
        "weeks": weeks,
        "bench_points_left": round(sum(w["left"] for w in weeks) / n, 1) if n else None,
        "hit_rate": round(sum(1 for w in weeks if w["left"] <= 0.5) / n, 3) if n else None,
        "started_out": started_out,
        "worst_week": {"season": worst["season"], "week": worst["week"], "left": worst["left"]} if worst else None,
        "moves_per_week": moves_per_week,
    }


def draft_flags(conn, member_id: str) -> dict:
    f = read_table(conn, "league_draft_flags")
    mine = f[f.member_id == member_id] if not f.empty else f
    n = int(len(mine))
    auto = int(mine.autodraft.sum()) if n else 0
    return {"n": n, "autodrafts": auto, "autodraft_rate": round(auto / n, 3) if n else None,
            "keepers": int(mine.keeper.sum()) if n else 0,
            "seasons": sorted(int(s) for s in mine.season.unique()) if n else []}


# The rules a defining line is chosen by, most extreme rank first. Each is
# (label, section, key, higher_is_it, minimum n). The member holding the
# league's top (or bottom) value on a fact with enough behind it gets that
# line; a member with no extreme gets their record.
DEFINING_RULES = [
    ("wins more than anyone", "finishes", "win_pct", True, 2),
    ("outperforms the draft board", "finishes", "outperformance", True, 2),
    ("trades more than anyone", "trades", "accepted", True, 2),
    ("works the wire harder than anyone", "waivers", "claims", True, 5),
    ("leaves the most points on the bench", "lineups", "bench_points_left", True, 5),
    ("sets the best lineups", "lineups", "hit_rate", True, 5),
    ("the luckiest record in the league", "luck", "luck", True, 5),
    ("the unluckiest record in the league", "luck", "luck", False, 5),
    ("lets the computer draft", "draft_flags", "autodraft_rate", True, 5),
]


def defining_line(member_id: str, facts_by_member: dict) -> str:
    mine = facts_by_member[member_id]
    for label, section, key, higher, minimum in DEFINING_RULES:
        values = {m: f[section].get(key) for m, f in facts_by_member.items()
                  if f[section].get("n", 0) >= minimum and f[section].get(key) is not None}
        if len(values) < 2 or member_id not in values:
            continue
        best = max(values, key=values.get) if higher else min(values, key=values.get)
        if best != member_id:
            continue
        if len(set(values.values())) < 2:
            continue
        return label
    fin = mine["finishes"]
    if fin["n"]:
        last = fin["seasons"][-1]
        return f"{last['wins']}-{last['losses']} last season, avg finish {fin['avg_finish']}"
    return "first season in the league"


def _facts(conn, member_id: str) -> dict:
    return {"finishes": finishes(conn, member_id), "playoffs": playoffs(conn, member_id),
            "luck": luck(conn, member_id), "waivers": waivers(conn, member_id),
            "trades": trades(conn, member_id), "lineups": lineups(conn, member_id),
            "draft_flags": draft_flags(conn, member_id)}


def league_overview(conn) -> dict:
    m = members(conn)
    facts = {r.member_id: _facts(conn, r.member_id) for r in m.itertuples()}
    grid = []
    for r in m.itertuples():
        f = facts[r.member_id]["finishes"]
        grid.append({
            "member_id": r.member_id, "display_name": r.display_name,
            "seasons": len(r.seasons), "titles": len(f["titles"]),
            "avg_finish": f["avg_finish"], "win_pct": f["win_pct"],
            "defining_line": defining_line(r.member_id, facts),
        })
    grid.sort(key=lambda g: (g["avg_finish"] is None, g["avg_finish"] or 0))
    return {"seasons": seasons_strip(conn), "members": grid}


def _draft_chapter(conn, display_name: str):
    """The league-report branch's draft habits for this manager, matched
    on the display name both tables carry. None when the branch's tables
    are not there yet or the manager has no graded picks."""
    try:
        from scoring.league_report import team_profiles
    except ImportError:
        return None
    try:
        return next((p for p in team_profiles(conn) if p.get("manager") == display_name), None)
    except Exception:      # noqa: BLE001 -- no picks, no ADP: no chapter
        return None


def profile(conn, member_id: str) -> dict | None:
    m = members(conn)
    hit = m[m.member_id == member_id]
    if hit.empty:
        return None
    me = hit.iloc[0]
    display = {r.member_id: r.display_name for r in m.itertuples()}
    h2h = [{"opponent": {"member_id": b, "display_name": display.get(b, b)}, **rec}
           for (a, b), rec in head_to_head(conn).items() if a == member_id]
    h2h.sort(key=lambda x: -x["games"])
    return {
        "member_id": member_id, "display_name": me.display_name,
        "first_name": me.first_name, "seasons": me.seasons,
        "head_to_head": h2h,
        "draft": _draft_chapter(conn, me.display_name),
        **_facts(conn, member_id),
    }
