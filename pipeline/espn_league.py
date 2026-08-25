"""ESPN fantasy league client: draft history, rosters, and league settings.

Parsers are pure functions over the JSON ESPN's read API returns, so they are
testable against literal fixtures with no network or browser. The Playwright
client and the season walk live further down.
"""
import json
import re
import time
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

# ESPN's own D/ST player ids: one per team, at -(16000 + proTeamId). Built
# from ESPN_PRO_TEAMS above rather than hardcoded, so it tracks the same team
# table everything else in this module keys on. Duplicated in intent (not in
# code) by `pipeline/espn_live._dst_espn_id`, which cannot be imported here --
# espn_live imports THIS module, so the dependency only runs one way.
DST_ESPN_IDS = frozenset(-(16000 + pro_team_id) for pro_team_id in ESPN_PRO_TEAMS)


def _is_real_pick(pick: dict) -> bool:
    """ESPN pads a draft board with placeholder picks whose playerId is -1.

    Every scheduled-but-undrafted slot looks like a real pick otherwise --
    it carries a team, a round, and an overall number. Even completed
    drafts carry a few (a round nobody filled). They are not picks anyone
    made, so they must never reach the model.

    DO NOT TIGHTEN THIS BACK TO `playerId > 0`. That is what it used to be,
    and it silently threw away EVERY D/ST pick this league has ever made,
    because ESPN's negative id space is not all sentinels -- it holds real
    team entities, one per proTeamId:

        -16001..-16034   D/ST          (defaultPositionId 16)
        -15001..-15034   team QB       (15)
        -14001..-14034   head coach    (14)

    Measured against ESPN's own 2025 player universe (2876 entries): 96 of
    them carry a negative id, exactly 32 in each of those three bands, every
    one exactly -(N000 + proTeamId), and there is no other negative id
    anywhere in the universe. So a `> 0` threshold does not select "real
    player, not padding" -- it selects "not a team". The damage was measured
    too: `draft_picks` held 712 rows over six seasons with ZERO DST, every
    season short by exactly eight picks (2020 [60, 68, 72, 90, 102, 113, 126,
    127], 2025 [98, 99, 100, 104, 106, 107, 108, 112], ...) while all eight of
    that season's kickers were recorded. Those gaps are the defenses. Downstream
    that taught the manager model that a defense is never chosen and drove
    `scoring.draft_model.COLD_START_PRIOR`'s `pos_DST` to -11.14, an event the
    fit could not have seen. That prior has since been refitted on real mock
    drafts (`make fit-prior`) and reads +0.21; the per-manager fits on this
    deployment's own league still carry the damage until it is re-imported.

    So this is a whitelist, not a threshold: a real pick names a real player
    id, or one of the 32 D/ST ids. Everything else -- -1, 0, null, or any
    wider sentinel ESPN adds later -- is padding. Coaches and team QBs are
    deliberately NOT admitted: this tool models no such roster slot
    (ESPN_SLOT_POSITIONS has no 14 or 15), so a league that drafted one would
    get a pick with no position, which is the same poison as a padding row.
    """
    player_id = pick.get("playerId")
    if player_id is None:
        return False
    return player_id > 0 or player_id in DST_ESPN_IDS


def parse_draft_picks(payload: dict, season: int) -> pd.DataFrame:
    picks = [p for p in (((payload.get("draftDetail") or {}).get("picks")) or [])
             if _is_real_pick(p)]
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

# How long the visible login window waits for a human, and how often it
# checks. Five minutes is the old wait_for_function timeout, kept: Disney SSO
# with 2FA on a phone genuinely takes minutes.
LOGIN_TIMEOUT_SECONDS = 300.0
LOGIN_POLL_MS = 500


def parse_league_id(url_or_id: str) -> str:
    m = re.search(r"leagueId=(\d+)", url_or_id)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{3,})\b", url_or_id)
    if not m:
        raise ValueError(f"no league id in {url_or_id!r}")
    return m.group(1)


def season_url(league_id: str, season: int, current_season: int | None = None) -> str:
    """Primary URL for a season, live or historical.

    ESPN documents /leagueHistory/{id}?seasonId= as the way to read a prior
    season, but it 404s for leagues whose prior seasons are still served
    under the dated path -- which, in practice, is how a league that has
    kept the same id reads back. So the dated path is tried first for every
    year and `history_url` is the fallback; `current_season` is accepted
    only so existing callers keep working.
    """
    return f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}?{VIEWS}"


def history_url(league_id: str, season: int) -> str:
    """Fallback for leagues whose prior seasons only answer here."""
    return f"{BASE}/leagueHistory/{league_id}?seasonId={season}&{VIEWS}"


def fetch_season(fetch, league_id: str, season: int, current_season: int):
    """Fetch a season payload, trying the dated path then leagueHistory.

    Raises FileNotFoundError only when BOTH forms 404, so a genuinely
    absent season still reads as a miss to the walk.
    """
    try:
        return _unwrap(fetch(season_url(league_id, season, current_season)))
    except FileNotFoundError:
        return _unwrap(fetch(history_url(league_id, season)))


def players_url(season: int) -> str:
    return f"{BASE}/seasons/{season}/players?view=players_wl"


# The player directory is capped at 50 rows unless the request carries a
# filter raising the limit -- and it answers 200 either way, so an
# unfiltered pull looks like a healthy fetch that silently resolves almost
# no names. `sources.fetch_espn_adp` already needs the same header for the
# same reason. 3000 comfortably covers a season's full directory (2020,
# the deepest here, returns ~7.9k raw rows of which ~3k are fantasy
# positions).
PLAYERS_FILTER = json.dumps({"players": {"limit": 3000}})


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
            payload = fetch_season(fetch, league_id, season, current_season)
            # A season whose draft has not happened yet still returns a full
            # board -- every slot present, every playerId -1. Importing that
            # as history would teach the model 120 picks nobody made, so a
            # season counts only once ESPN says the draft is done AND at
            # least one pick names a real player.
            detail = payload.get("draftDetail") or {}
            has_picks = bool(detail.get("drafted")) and any(
                _is_real_pick(p) for p in (detail.get("picks") or []))
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
        # A league that ends its draft early leaves whole rounds unfilled --
        # ESPN pads the board to roster size regardless, and those padding
        # slots are dropped as placeholder picks. That is a league habit, not
        # a data problem, so name it as short rounds rather than flagging the
        # count as suspicious. Anything that is not a whole number of rounds
        # short really is unexpected.
        short = expected - len(grp)
        if short == 0:
            flag = ""
        elif short > 0 and short % settings.teams == 0:
            rounds_short = short // settings.teams
            flag = (f"  ({rounds_short} of {settings.rounds} rounds not drafted"
                    " -- draft ended early)")
        else:
            flag = f"  <-- expected {expected}"
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
        # Poll the context's own cookie jar, and require BOTH cookies.
        #
        # The obvious `page.wait_for_function` on `document.cookie` is wrong,
        # though not for the reason this comment used to give. It claimed
        # `espn_s2` is HttpOnly and therefore invisible to `document.cookie`;
        # measured in a real browser, it is NOT HttpOnly and does appear
        # there. What is still true is the half that actually matters here:
        # `SWID` is set on page load, before anyone has signed in, so an `||`
        # between the two returns instantly and saves an anonymous session --
        # which then fails later, at the socket, as "the saved ESPN login has
        # expired", pointing at the wrong thing entirely.
        #
        # `context.cookies()` is kept regardless: it reads the real jar rather
        # than a string the page happens to expose, so it stays correct if
        # ESPN ever does set the flag this comment used to assume.
        deadline = time.monotonic() + LOGIN_TIMEOUT_SECONDS
        while True:
            names = {c["name"] for c in context.cookies()}
            if {"espn_s2", "SWID"} <= names:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Timed out waiting for an ESPN sign-in "
                    f"({LOGIN_TIMEOUT_SECONDS:.0f}s). The browser window "
                    "never reached a signed-in session -- cookies seen: "
                    f"{sorted(names) or 'none'}.")
            page.wait_for_timeout(LOGIN_POLL_MS)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        context.storage_state(path=str(self.state_path))
        context.close()
        browser.close()
        # Rebuild the headless context so it picks up the new cookies.
        self._context.close()
        self._browser.close()
        self._open_context()

    def get_json(self, url: str):
        # The player-directory endpoint needs a filter header to return more
        # than 50 rows. Applying it by URL rather than by parameter keeps
        # `fetch` a plain one-argument callable, which is what lets
        # `import_seasons` be driven by a fixture in the tests.
        headers = {"X-Fantasy-Filter": PLAYERS_FILTER} if "/players?" in url else None
        response = self._context.request.get(url, headers=headers)
        if response.status in (401, 403):
            self.login()
            response = self._context.request.get(url, headers=headers)
        if response.status == 404:
            raise FileNotFoundError(url)
        if not response.ok:
            raise RuntimeError(f"{response.status} for {url}")
        return response.json()
