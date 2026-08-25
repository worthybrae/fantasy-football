"""Everything a league report says in numbers.

Pure functions over a league's database (and, for the season being drafted,
the live room's own tables): which picks were steals and which were
reaches, a draft grade per team, a profile per manager from how they have
drafted and finished over the years, and a power ranking that blends this
draft with that history. No network, no model. `scoring/blurbs.py` writes
prose against the dict `build_facts` returns; `api/reports.py` stores it.
"""
from __future__ import annotations

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
    slots = round(value)
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
    """team_id -> (manager, team_name) from the newest season each appears in."""
    out = {}
    standings = read_table(conn, "league_standings")
    if not standings.empty:
        for _, row in standings.sort_values("season").iterrows():
            out[int(row["team_id"])] = (str(row["manager"]), row.get("team_name"))
    teams = read_table(conn, "draft_teams")
    if not teams.empty:
        for _, row in teams.sort_values("season").iterrows():
            tid = int(row["team_id"])
            if tid not in out:
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
               team_slots: dict, season: int) -> pd.DataFrame:
    """The season being drafted, from the room: `drafted` rows and the board.

    `drafted` is `[(player_id, pick_no), ...]`; `board_by_id` is the
    session's `player_id -> board row` dict; `names` is `names_for(conn)`;
    `team_slots` is the session's `slot -> ESPN team name`. The team behind
    a slot is `settings.pick_order[slot - 1]`; a manager the history does
    not know is named by the ESPN team name, then "Team N".
    """
    teams, rounds = int(settings.teams), int(settings.rounds)
    slots = snake_slots(teams, rounds)
    order = list(settings.pick_order or ())
    rows = []
    for player_id, pick_no in drafted:
        overall = int(pick_no)
        slot = slots[overall - 1] if 0 < overall <= len(slots) else None
        team_id = order[slot - 1] if slot is not None and slot - 1 < len(order) else slot
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
