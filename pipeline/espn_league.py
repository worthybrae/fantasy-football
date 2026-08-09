"""ESPN fantasy league client: draft history, rosters, and league settings.

Parsers are pure functions over the JSON ESPN's read API returns, so they are
testable against literal fixtures with no network or browser. The Playwright
client and the season walk live further down.
"""
import pandas as pd

# ESPN lineup slot ids -> our position vocabulary. Slots we do not model
# (individual defensive positions, punter, head coach) are absent by design.
ESPN_SLOT_POSITIONS = {
    0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "DST", 17: "K",
}
ESPN_FLEX_SLOT = 23      # RB/WR/TE
ESPN_BENCH_SLOT = 20
ESPN_IR_SLOT = 21

# `defaultPositionId` on a player, which uses a different id space than slots.
ESPN_POSITION_BY_ID = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}

# ESPN `proTeamId` -> nflverse team abbreviation. 0 is free agent.
ESPN_PRO_TEAMS = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LA",
    15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA",
    27: "TB", 28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}

_PICK_COLUMNS = ["season", "overall_pick", "round", "round_pick",
                 "team_id", "espn_player_id", "keeper"]


def parse_draft_picks(payload: dict, season: int) -> pd.DataFrame:
    picks = ((payload.get("draftDetail") or {}).get("picks")) or []
    rows = [{"season": season,
             "overall_pick": p.get("overallPickNumber"),
             "round": p.get("roundId"),
             "round_pick": p.get("roundPickNumber"),
             "team_id": p.get("teamId"),
             "espn_player_id": p.get("playerId"),
             "keeper": bool(p.get("keeper"))}
            for p in picks]
    df = pd.DataFrame(rows, columns=_PICK_COLUMNS)
    return df.sort_values("overall_pick").reset_index(drop=True)


def parse_draft_teams(payload: dict, season: int) -> pd.DataFrame:
    # `owners` holds member GUIDs; the human-readable name lives in `members`.
    # A team with no owner (orphan/abandoned) falls back to its team name so
    # the manager column is never null -- the model keys on it.
    members = {m.get("id"): m.get("displayName")
               for m in (payload.get("members") or [])}
    rows = []
    for t in payload.get("teams") or []:
        owners = t.get("owners") or []
        manager = next((members[o] for o in owners if members.get(o)), None)
        rows.append({"season": season, "team_id": t.get("id"),
                     "manager": manager or t.get("name"),
                     "slot": t.get("draftDayPickOrder")})
    return pd.DataFrame(rows, columns=["season", "team_id", "manager", "slot"])


def parse_player_directory(payload) -> pd.DataFrame:
    # The season player endpoint returns a bare list; the league views nest it
    # under "players". Accept either so one parser serves both.
    entries = payload if isinstance(payload, list) else (payload.get("players") or [])
    rows = []
    for entry in entries:
        p = entry.get("player", entry)
        pos = ESPN_POSITION_BY_ID.get(p.get("defaultPositionId"))
        if pos is None:
            continue
        rows.append({"espn_player_id": p.get("id"),
                     "player_name": p.get("fullName"),
                     "position": pos,
                     "nfl_team": ESPN_PRO_TEAMS.get(p.get("proTeamId"))})
    return pd.DataFrame(rows, columns=["espn_player_id", "player_name",
                                       "position", "nfl_team"])


def parse_settings(payload: dict, season: int) -> dict:
    s = payload.get("settings") or {}
    return {
        "season": season,
        "teams": (s.get("size") or 0),
        "lineup_slots": (s.get("rosterSettings") or {}).get("lineupSlotCounts") or {},
        "scoring_items": (s.get("scoringSettings") or {}).get("scoringItems") or [],
        "draft_type": (s.get("draftSettings") or {}).get("type"),
        "pick_order": (s.get("draftSettings") or {}).get("pickOrder") or [],
    }
