"""ESPN fantasy league client: draft history, rosters, and league settings.

Parsers are pure functions over the JSON ESPN's read API returns, so they are
testable against literal fixtures with no network or browser. The Playwright
client and the season walk live further down.
"""
import re
from pathlib import Path

import pandas as pd

from pipeline.db import write_table, read_table
from scoring import league as league_mod

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


BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
VIEWS = "view=mDraftDetail&view=mTeam&view=mSettings"
LOGIN_URL = "https://www.espn.com/login"
STATE_PATH = "data/espn_state.json"


def parse_league_id(url_or_id: str) -> str:
    m = re.search(r"leagueId=(\d+)", url_or_id)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{3,})\b", url_or_id)
    if not m:
        raise ValueError(f"no league id in {url_or_id!r}")
    return m.group(1)


def season_url(league_id: str, season: int, current_season: int) -> str:
    # ESPN serves the live season under /seasons/{year}/... and every earlier
    # season under /leagueHistory/{id}?seasonId=. Hitting the wrong one for a
    # given year returns 404, not a redirect.
    if season >= current_season:
        return f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}?{VIEWS}"
    return f"{BASE}/leagueHistory/{league_id}?seasonId={season}&{VIEWS}"


def players_url(season: int) -> str:
    return f"{BASE}/seasons/{season}/players?view=players_wl"


def _unwrap(payload):
    # leagueHistory returns a single-element list; the seasons endpoint returns
    # the object directly.
    if isinstance(payload, list) and payload:
        return payload[0]
    return payload


def import_seasons(conn, league_id: str, current_season: int, fetch,
                   max_back: int = 15) -> dict:
    """Walk seasons backward from current_season, newest first.

    `fetch(url) -> parsed JSON` is injected so tests can serve fixtures. It
    must raise on a missing season; two consecutive misses end the walk, which
    tolerates one gap year without running to `max_back` on every import.
    """
    picks, teams, leagues, directories = [], [], [], []
    seasons, misses = [], 0
    for season in range(current_season, current_season - max_back, -1):
        # Only a missing season (404 -> FileNotFoundError) counts as a
        # "miss" the walk can tolerate. Anything else -- a persistent
        # auth failure, a 5xx, a JSON decode error -- is a real problem
        # with a real cause and must propagate, not be silently absorbed
        # into 15 wasted iterations and a misleading "no drafted seasons
        # found" at the end.
        try:
            payload = _unwrap(fetch(season_url(league_id, season, current_season)))
            has_picks = bool((payload.get("draftDetail") or {}).get("picks"))
            # Fetched on the same iteration, under the same handling, as
            # the season payload: a season that exists should have both.
            players_payload = fetch(players_url(season)) if has_picks else None
        except FileNotFoundError:
            misses += 1
            if misses >= 2 and seasons:
                break
            continue
        if not has_picks:
            misses += 1
            if misses >= 2 and seasons:
                break
            continue
        misses = 0
        settings = league_mod.from_espn(parse_settings(payload, season))
        if settings.draft_type != "SNAKE":
            raise ValueError(
                f"season {season} draft type is {settings.draft_type}; "
                "only SNAKE is supported")
        seasons.append(season)
        picks.append(parse_draft_picks(payload, season))
        teams.append(parse_draft_teams(payload, season))
        leagues.append({"season": season,
                        "settings_json": league_mod.to_json(settings)})
        directory = parse_player_directory(players_payload)
        directory["season"] = season
        directories.append(directory)

    if not seasons:
        raise ValueError("no drafted seasons found -- check the league id and login")

    all_picks = pd.concat(picks, ignore_index=True)
    directory = pd.concat(directories, ignore_index=True)
    all_picks = all_picks.merge(
        directory, on=["season", "espn_player_id"], how="left")

    write_table(conn, "draft_picks", all_picks)
    write_table(conn, "draft_teams", pd.concat(teams, ignore_index=True))
    write_table(conn, "league", pd.DataFrame(leagues))
    return {"seasons": seasons, "picks": len(all_picks)}


def validate_import(conn) -> list[str]:
    from scoring.draft_model import _match_keys
    picks = read_table(conn, "draft_picks")
    teams = read_table(conn, "draft_teams")
    adp = read_table(conn, "historic_adp")
    # Each season keeps its own settings row (team count and roster shape can
    # change year over year) -- `league_mod.load(conn)` only returns the
    # newest one, so it is wrong to apply league-wide here. Build the
    # per-season map once, and fall back to the newest/default settings only
    # for a season with no row of its own (shouldn't happen in practice).
    league_table = read_table(conn, "league")
    settings_by_season = {int(row["season"]): league_mod.from_json(row["settings_json"])
                          for _, row in league_table.iterrows()}
    lines = []
    for season, grp in picks.groupby("season"):
        settings = settings_by_season.get(int(season), league_mod.load(conn))
        expected = settings.teams * settings.rounds
        managers = teams[teams["season"] == season]["manager"].nunique()
        if adp.empty:
            rate = 0.0
        else:
            # Same join key build_observations fits on, so this number
            # reports the real match rate rather than a second, subtly
            # different one -- the report is the spec's honesty mechanism and
            # has to agree with what actually gets fitted.
            season_adp = adp[adp["season"] == season]
            known = {key for key in _match_keys(season_adp, "adp_name")
                     if key is not None}
            hit = [key is not None and key in known
                   for key in _match_keys(grp, "player_name", "nfl_team")]
            rate = float(sum(hit)) / len(grp) if len(grp) else 0.0
        flag = "" if len(grp) == expected else f"  <-- expected {expected}"
        lines.append(f"  {season}: {len(grp)} picks, {managers} managers, "
                     f"ADP match {rate:.0%}{flag}")
        if rate and rate < 0.8:
            lines.append(f"         low ADP match for {season} -- "
                         "picks below the threshold are excluded from fitting")
    return lines


class EspnClient:
    """Playwright-backed fetcher that reuses a saved ESPN login.

    First run with no saved state (or a state ESPN has expired) opens a real
    browser window at the ESPN login page and waits for the human. Disney SSO
    handles 2FA and bot checks there, which is the only place they can be
    answered. Everything after that is headless.
    """

    def __init__(self, state_path: str = STATE_PATH):
        self.state_path = Path(state_path)
        self._pw = None
        self._browser = None
        self._context = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._open_context()
        return self

    def __exit__(self, *exc):
        for closer in (self._context, self._browser):
            if closer is not None:
                closer.close()
        if self._pw is not None:
            self._pw.stop()

    def _open_context(self):
        self._browser = self._pw.chromium.launch(headless=True)
        kwargs = {"storage_state": str(self.state_path)} if self.state_path.exists() else {}
        self._context = self._browser.new_context(**kwargs)

    def login(self):
        """Open a visible window and block until the user has signed in."""
        print("ESPN login required -- a browser window is opening.")
        print("Sign in, then return here; the import continues automatically.")
        browser = self._pw.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(LOGIN_URL)
        # espn_s2 is only set once the SSO flow completes.
        page.wait_for_function(
            "() => document.cookie.includes('espn_s2') || "
            "document.cookie.includes('SWID')", timeout=300_000)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        context.storage_state(path=str(self.state_path))
        context.close()
        browser.close()
        # Rebuild the headless context so it picks up the new cookies.
        self._context.close()
        self._browser.close()
        self._open_context()

    def get_json(self, url: str):
        response = self._context.request.get(url)
        if response.status in (401, 403):
            self.login()
            response = self._context.request.get(url)
        if response.status == 404:
            raise FileNotFoundError(url)
        if not response.ok:
            raise RuntimeError(f"{response.status} for {url}")
        return response.json()
