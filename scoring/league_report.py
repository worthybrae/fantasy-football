"""Everything a league report says in numbers.

Pure functions over a league's database (and, for the season being drafted,
the live room's own tables): which picks were steals and which were
reaches, a draft grade per team, a profile per manager from how they have
drafted and finished over the years, and a power ranking that blends this
draft with that history. No network, no model. `scoring/blurbs.py` writes
prose against the dict `build_facts` returns; `api/reports.py` stores it.
"""
from __future__ import annotations

import math

import pandas as pd

from pipeline.db import read_table
from scoring import league as league_mod
from scoring.board import _ADP_POSITION_ALIASES
from scoring.draft_sim import snake_slots

PICK_COLUMNS = ["season", "overall_pick", "round", "team_id", "manager", "team_name",
                "player_name", "position", "market_rank", "value", "verdict"]

# The ticker's five bands (web/src/components/draft/PickTicker.tsx `verdict`),
# so the report card never disagrees with what the room said as the pick
# landed. `abs(value) <= 2` is on the market; past that, one round's worth of
# picks separates "value" from "steal" and "early" from "reach".
ON_MARKET_SLOTS = 2

# Power ranking weights. This draft counts more than the past because the
# page is a draft report card first; the past still counts because a manager
# who has finished top-three four years running has earned some doubt about
# a C draft.
DRAFT_WEIGHT = 0.6
HISTORY_WEIGHT = 0.4

# Grade cut points as fractions of the league: top eighth A, next quarter B,
# middle quarter C, next quarter D, bottom eighth F. Eight teams grade
# 1-2-2-2-1.
GRADE_CUTS = ((1 / 8, "A"), (3 / 8, "B"), (5 / 8, "C"), (7 / 8, "D"))


def pick_value(overall_pick: int, market_rank, last_pick: int):
    """Picks past ADP (positive is a steal), or None when there is no ADP or
    the market ranked the player past the draft's last pick -- the kicker
    and defense rule from `api/live._board_cell`.

    `pd.isna` rather than a `None`/`math.isnan` check: a nullable `Float64`
    column (both `historical_picks` and `live_picks` cast `market_rank` to
    one before this runs, so a missing lookup can arrive here) holds its
    missing values as `pd.NA`, which `math.isnan` cannot even be asked
    about -- `float(pd.NA)` raises before that check would run."""
    if pd.isna(market_rank):
        return None
    if float(market_rank) > last_pick:
        return None
    return float(overall_pick) - float(market_rank)


def verdict(value, teams: int):
    if pd.isna(value):
        return None
    # `math.floor(value + 0.5)`, not Python's `round` -- `round` uses
    # banker's rounding (round(2.5) == 2, round(-7.5) == -8), which disagrees
    # with PickTicker.tsx's `Math.round` (round half toward +infinity: 2.5 ->
    # 3, -7.5 -> -7) at real, plausible half-integer ADP deltas.
    slots = math.floor(value + 0.5)
    size = abs(slots)
    rnd = max(2, int(teams))
    if size <= ON_MARKET_SLOTS:
        return "market"
    if slots > 0:
        return "steal" if size >= rnd else "value"
    return "reach" if size >= rnd else "early"


def letter(rank0: int, n: int) -> str:
    frac = (rank0 + 1) / n
    for cut, grade in GRADE_CUTS:
        if frac <= cut + 1e-9:
            return grade
    return "F"


def settings_for(conn, season: int):
    table = read_table(conn, "league")
    if not table.empty:
        row = table[table["season"] == season]
        if not row.empty:
            return league_mod.from_json(row.iloc[0]["settings_json"])
    return league_mod.load(conn)


def league_name(conn, league_id: str) -> str:
    table = read_table(conn, "league")
    if not table.empty and "name" in table.columns:
        names = table.sort_values("season")["name"].dropna()
        if not names.empty:
            return str(names.iloc[-1])
    return f"League {league_id}"


def names_for(conn) -> dict:
    """team_id -> (manager, team_name) from the newest season each appears in.

    A row with no manager is SKIPPED, not stringified. `str(None)` is
    "None" and `str(float('nan'))` is "nan", and either one goes straight
    onto a public report page and into the model's prompt as if it were
    somebody's name. Absent here, the caller falls back to the ESPN team
    name or "Team N", which is what a nameless team is actually called.
    (The import should no longer produce one -- `parse_standings` and
    `parse_draft_teams` both fall back to ESPN's own team label now -- but
    every league file imported before that keeps its nulls.)
    """
    out = {}
    standings = read_table(conn, "league_standings")
    if not standings.empty:
        for _, row in standings.sort_values("season").iterrows():
            if pd.isna(row["manager"]):
                continue
            out[int(row["team_id"])] = (str(row["manager"]), row.get("team_name"))
    teams = read_table(conn, "draft_teams")
    if not teams.empty:
        # Descending, so the "first wins" guard below keeps the NEWEST
        # season for a team_id standings never covered -- sorted ascending,
        # that same guard would instead lock in the oldest one, the reverse
        # of this function's contract.
        for _, row in teams.sort_values("season", ascending=False).iterrows():
            tid = int(row["team_id"])
            if tid not in out and not pd.isna(row["manager"]):
                out[tid] = (str(row["manager"]), None)
    return out


def _with_verdicts(frame: pd.DataFrame, teams: int, last_pick: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["value"] = [pick_value(o, m, last_pick)
                      for o, m in zip(frame["overall_pick"], frame["market_rank"])]
    frame["verdict"] = [verdict(v, teams) for v in frame["value"]]
    frame["value"] = frame["value"].astype("Float64")
    return frame[PICK_COLUMNS].sort_values("overall_pick").reset_index(drop=True)


def historical_picks(conn, season: int) -> pd.DataFrame:
    """One season's picks from `draft_picks`, priced by that year's ADP."""
    from scoring.draft_model import _match_keys
    picks = read_table(conn, "draft_picks")
    adp = read_table(conn, "historic_adp")
    if picks.empty or adp.empty:
        return pd.DataFrame(columns=PICK_COLUMNS)
    picks = picks[picks["season"] == season].copy()
    adp = adp[adp["season"] == season].copy()
    if picks.empty or adp.empty:
        return pd.DataFrame(columns=PICK_COLUMNS)
    adp = adp.assign(position=adp["position"].replace(_ADP_POSITION_ALIASES))
    adp = adp.assign(key=_match_keys(adp, "adp_name"))
    adp = adp[adp["key"].notna()].sort_values("adp_rank").drop_duplicates("key")
    rank_by_key = dict(zip(adp["key"], adp["adp_rank"].astype(float)))
    picks = picks.assign(key=_match_keys(picks, "player_name", "nfl_team"))
    picks["market_rank"] = [rank_by_key.get(k) for k in picks["key"]]
    picks["market_rank"] = picks["market_rank"].astype("Float64")
    names = names_for(conn)
    picks["manager"] = [names.get(int(t), (f"Team {t}", None))[0] for t in picks["team_id"]]
    picks["team_name"] = [names.get(int(t), (None, None))[1] or f"Team {t}"
                          for t in picks["team_id"]]
    settings = settings_for(conn, season)
    teams = int(settings.teams) or int(picks["team_id"].nunique())
    picks["round"] = ((picks["overall_pick"].astype(int) - 1) // teams) + 1
    # The draft's OWN last pick, not the widest ADP rank in play -- the
    # kicker/defense rule (`api/live.py:1177-1178`, ported in `pick_value`)
    # exists precisely to null a market rank that sits past what this draft
    # ever reached. Bounding at the ADP column's own max instead would make
    # the rule unreachable here: every `market_rank` value comes from that
    # same column, so nothing could ever exceed it. A late, shallow league
    # can genuinely hand its final pick to a player the market never expected
    # to go that early -- that pick is supposed to grade as ungraded, not as
    # an enormous reach.
    last_pick = int(picks["overall_pick"].max())
    return _with_verdicts(picks, teams, last_pick)


def live_picks(drafted: list, board_by_id: dict, settings, names: dict,
               team_slots: dict, season: int,
               trust_pick_order: bool = True) -> pd.DataFrame:
    """The season being drafted, from the room: `drafted` rows and the board.

    `drafted` is `[(player_id, pick_no), ...]`; `board_by_id` is the
    session's `player_id -> board row` dict; `names` is `names_for(conn)`;
    `team_slots` is the session's `slot -> ESPN team name`. The team behind
    a slot is `settings.pick_order[slot - 1]`; a manager the history does
    not know is named by the ESPN team name, then "Team N".

    `trust_pick_order` IS THE PRECONDITION `api/live._slot_from_pick_order`
    spells out, carried in by the caller because a bare LeagueSettings does
    not know where it came from. `pick_order` names a team per slot only
    when the settings are from THIS connect's live ESPN fetch; the
    database's newest `league` row carries LAST season's order, and because
    ESPN team ids are stable that lookup returns a confident, wrong manager
    instead of nothing. False (or an empty order) means the slot is all we
    know: name the team from `team_slots`, then "Team N", and leave
    `team_id` null rather than looking a SLOT up in `names`, which is keyed
    by team id. `api/reports.on_draft_complete` passes
    `session.settings_from_espn` here, the same flag every other consumer
    gates on.
    """
    teams, rounds = int(settings.teams), int(settings.rounds)
    slots = snake_slots(teams, rounds)
    order = list(settings.pick_order or ()) if trust_pick_order else []
    rows = []
    for player_id, pick_no in drafted:
        overall = int(pick_no)
        slot = slots[overall - 1] if 0 < overall <= len(slots) else None
        team_id = order[slot - 1] if slot is not None and slot - 1 < len(order) else None
        espn_name = team_slots.get(slot) if slot is not None else None
        manager, team_name = names.get(int(team_id), (None, None)) if team_id is not None else (None, None)
        manager = manager or espn_name or f"Team {slot}"
        team_name = team_name or espn_name or f"Team {slot}"
        row = board_by_id.get(str(player_id)) or {}
        rank = row.get("market_rank")
        rows.append({
            "season": season, "overall_pick": overall,
            "round": ((overall - 1) // teams) + 1 if teams else None,
            "team_id": team_id, "manager": manager, "team_name": team_name,
            "player_name": row.get("name") or str(player_id),
            "position": row.get("position"),
            "market_rank": None if rank is None or pd.isna(rank) else float(rank),
        })
    if not rows:
        return pd.DataFrame(columns=PICK_COLUMNS)
    frame = pd.DataFrame(rows)
    frame["market_rank"] = frame["market_rank"].astype("Float64")
    return _with_verdicts(frame, teams, len(slots))


def _pick_record(row) -> dict:
    return {"player_name": row["player_name"], "position": row["position"],
            "round": int(row["round"]) if pd.notna(row["round"]) else None,
            "overall_pick": int(row["overall_pick"]),
            "market_rank": None if pd.isna(row["market_rank"]) else float(row["market_rank"]),
            "value": None if pd.isna(row["value"]) else round(float(row["value"]), 1),
            "verdict": row["verdict"]}


def _shape(team: pd.DataFrame) -> dict:
    positions = team["position"].dropna().value_counts().to_dict()
    first = {}
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        at = team[team["position"] == pos]
        first[pos] = int(at["round"].min()) if not at.empty else None
    return {"positions": {str(k): int(v) for k, v in positions.items()},
            "first_round": first}


def draft_grades(picks: pd.DataFrame, teams: int) -> list:
    """One record per team, best grade first."""
    if picks.empty:
        return []
    out = []
    for manager, team in picks.groupby("manager", sort=False):
        graded = team[team["value"].notna()]
        n = int(len(graded))
        total = float(graded["value"].sum()) if n else 0.0
        best = graded.loc[graded["value"].astype(float).idxmax()] if n else None
        worst = graded.loc[graded["value"].astype(float).idxmin()] if n else None
        out.append({
            "manager": str(manager),
            "team_name": str(team.iloc[0]["team_name"]),
            "team_id": None if pd.isna(team.iloc[0]["team_id"]) else int(team.iloc[0]["team_id"]),
            "value_total": round(total, 1),
            "value_per_pick": round(total / n, 2) if n else None,
            "graded_picks": n,
            "steals": int((team["verdict"] == "steal").sum()),
            "reaches": int((team["verdict"] == "reach").sum()),
            "best_pick": _pick_record(best) if best is not None else None,
            "worst_pick": _pick_record(worst) if worst is not None else None,
            "shape": _shape(team),
        })
    # Rank on value per pick; a team with nothing gradeable sorts last.
    out.sort(key=lambda g: (g["value_per_pick"] is None,
                            -(g["value_per_pick"] or 0.0), -g["value_total"]))
    n = len(out)
    for i, g in enumerate(out):
        g["grade"] = letter(i, n)
    return out


def _all_historical_picks(conn) -> pd.DataFrame:
    picks = read_table(conn, "draft_picks")
    if picks.empty:
        return pd.DataFrame(columns=PICK_COLUMNS)
    frames = [historical_picks(conn, int(s)) for s in sorted(picks["season"].unique())]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PICK_COLUMNS)


def _model_notes(conn, manager: str) -> list:
    profiles = read_table(conn, "manager_profiles")
    if profiles.empty or "summary" not in profiles.columns:
        return []
    mine = profiles[(profiles["manager"] == manager) & profiles["summary"].notna()]
    return [str(s) for s in mine["summary"].tolist() if str(s).strip()]


def _num(value):
    return None if value is None or pd.isna(value) else float(value)


def team_profiles(conn) -> list:
    """One record per manager: seasons finished, titles, and draft habits."""
    standings = read_table(conn, "league_standings")
    picks = _all_historical_picks(conn)
    managers = []
    for m in (standings["manager"].tolist() if not standings.empty else []) + (
            picks["manager"].tolist() if not picks.empty else []):
        if m not in managers:
            managers.append(m)
    out = []
    for manager in managers:
        mine = (standings[standings["manager"] == manager].sort_values("season")
                if not standings.empty else pd.DataFrame())
        seasons = []
        for _, row in mine.iterrows():
            seasons.append({
                "season": int(row["season"]),
                "wins": None if pd.isna(row["wins"]) else int(row["wins"]),
                "losses": None if pd.isna(row["losses"]) else int(row["losses"]),
                "ties": None if pd.isna(row["ties"]) else int(row["ties"]),
                "points_for": _num(row["points_for"]),
                "final_rank": None if pd.isna(row["final_rank"]) else int(row["final_rank"]),
                "playoff_seed": None if pd.isna(row["playoff_seed"]) else int(row["playoff_seed"]),
            })
        done = [s for s in seasons if s["final_rank"] is not None]
        games = sum((s["wins"] or 0) + (s["losses"] or 0) + (s["ties"] or 0) for s in done)
        wins = sum((s["wins"] or 0) + 0.5 * (s["ties"] or 0) for s in done)
        ppg_games = [(s["points_for"], (s["wins"] or 0) + (s["losses"] or 0) + (s["ties"] or 0))
                     for s in done if s["points_for"] is not None]
        ppg_total_games = sum(g for _, g in ppg_games)
        my_picks = picks[picks["manager"] == manager] if not picks.empty else pd.DataFrame()
        graded = my_picks[my_picks["value"].notna()] if not my_picks.empty else pd.DataFrame()
        firsts = {}
        if not my_picks.empty:
            for _, season_picks in my_picks.groupby("season"):
                pos = season_picks.sort_values("overall_pick").iloc[0]["position"]
                if pos is not None and not pd.isna(pos):
                    firsts[str(pos)] = firsts.get(str(pos), 0) + 1
        team_name = None
        if not mine.empty and "team_name" in mine.columns:
            names = mine["team_name"].dropna()
            team_name = str(names.iloc[-1]) if not names.empty else None
        if team_name is None and not my_picks.empty:
            team_name = str(my_picks.sort_values("season").iloc[-1]["team_name"])
        out.append({
            "manager": str(manager),
            "team_name": team_name or str(manager),
            "seasons": seasons,
            "titles": [s["season"] for s in done if s["final_rank"] == 1],
            "playoffs": [s["season"] for s in seasons if s["playoff_seed"] is not None],
            "completed": len(done),
            "win_pct": (wins / games) if games else None,
            "avg_finish": round(sum(s["final_rank"] for s in done) / len(done), 2) if done else None,
            "ppg": round(sum(p for p, _ in ppg_games) / ppg_total_games, 1) if ppg_total_games else None,
            "drafts": int(my_picks["season"].nunique()) if not my_picks.empty else 0,
            "mean_value": round(float(graded["value"].astype(float).mean()), 2) if len(graded) else None,
            "steal_rate": round(float((graded["verdict"] == "steal").mean()), 3) if len(graded) else None,
            "reach_rate": round(float((graded["verdict"] == "reach").mean()), 3) if len(graded) else None,
            "first_pick_positions": firsts,
            "career_best": _pick_record(graded.loc[graded["value"].astype(float).idxmax()]) if len(graded) else None,
            "career_worst": _pick_record(graded.loc[graded["value"].astype(float).idxmin()]) if len(graded) else None,
            "model_notes": _model_notes(conn, manager),
        })
    return out


def _z(values: list) -> list:
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return [0.0 for _ in values]
    mean = sum(present) / len(present)
    var = sum((v - mean) ** 2 for v in present) / len(present)
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0 for _ in values]
    return [0.0 if v is None else (v - mean) / sd for v in values]


def power_rankings(grades: list, profiles: list) -> list:
    """Blend this draft (`DRAFT_WEIGHT`) with career finish (`HISTORY_WEIGHT`),
    each z-scored across the field; a manager with no completed seasons has
    no history term to blend in, so their score is the draft term alone."""
    by_manager = {p["manager"]: p for p in profiles}
    draft_z = _z([g["value_per_pick"] for g in grades])
    history = [by_manager.get(g["manager"], {}).get("win_pct") for g in grades]
    completed = [by_manager.get(g["manager"], {}).get("completed", 0) for g in grades]
    history_z = _z(history)
    rows = []
    for g, dz, hz, done in zip(grades, draft_z, history_z, completed):
        first_year = not done
        score = DRAFT_WEIGHT * dz + HISTORY_WEIGHT * (0.0 if first_year else hz)
        rows.append({"manager": g["manager"], "team_name": g["team_name"],
                     "score": round(score, 3), "first_year": first_year,
                     "_tiebreak": g["value_total"]})
    rows.sort(key=lambda r: (-r["score"], -r["_tiebreak"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        del r["_tiebreak"]
    return rows


def build_facts(conn, league_id: str, season: int, picks: pd.DataFrame | None = None) -> dict:
    """The report, numbers only. `picks` overrides the historical read for a
    season the room just drafted (see `live_picks`)."""
    settings = settings_for(conn, season)
    frame = picks if picks is not None else historical_picks(conn, season)
    base = {"league_id": str(league_id), "season": int(season),
            "league_name": league_name(conn, league_id),
            "teams": int(settings.teams) or int(frame["manager"].nunique() if not frame.empty else 0),
            "generated_at": None, "model": None, "intro": None}
    if frame.empty or frame["value"].notna().sum() == 0:
        return {**base, "status": "failed",
                "reason": f"no graded picks for {season}: the draft is not imported, "
                          "or no ADP is on file for that year",
                "power_rankings": [], "report_cards": [], "profiles": []}
    grades = draft_grades(frame, base["teams"] or int(frame["manager"].nunique()))
    profiles = team_profiles(conn)
    known = {p["manager"] for p in profiles}
    for g in grades:
        if g["manager"] not in known:
            profiles.append({"manager": g["manager"], "team_name": g["team_name"],
                             "seasons": [], "titles": [], "playoffs": [], "completed": 0,
                             "win_pct": None, "avg_finish": None, "ppg": None, "drafts": 0,
                             "mean_value": None, "steal_rate": None, "reach_rate": None,
                             "first_pick_positions": {}, "career_best": None,
                             "career_worst": None, "model_notes": []})
    graded_managers = {g["manager"] for g in grades}
    profiles = [p for p in profiles if p["manager"] in graded_managers]
    for g in grades:
        g["nickname"] = None
        g["blurb"] = None
    ranks = power_rankings(grades, profiles)
    for r in ranks:
        r["line"] = None
    return {**base, "status": "ready", "power_rankings": ranks,
            "report_cards": grades, "profiles": profiles}


def merge_prose(facts: dict, prose: dict | None, model: str | None) -> dict:
    """Lay the writer's words onto the numbers. No prose is a complete
    report with a different status, not an error."""
    if facts.get("status") == "failed":
        return facts
    facts["model"] = model if prose is not None else None
    if prose is None:
        facts["status"] = "numbers_only"
        return facts
    cards = {c["manager"]: c for c in prose.get("cards", [])}
    lines = {r["manager"]: r for r in prose.get("rankings", [])}
    facts["intro"] = prose.get("intro")
    for card in facts["report_cards"]:
        got = cards.get(card["manager"], {})
        card["nickname"] = got.get("nickname")
        card["blurb"] = got.get("blurb")
    for row in facts["power_rankings"]:
        row["line"] = lines.get(row["manager"], {}).get("line")
    facts["status"] = "ready"
    return facts
