import json
import re

import nfl_data_py as nfl
import pandas as pd
import requests

ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
WEEKLY_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet"

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
ESPN_URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
            "seasons/{year}/segments/0/leaguedefaults/3?view=kona_player_info")
FP_URL = "https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php"
SLEEPER_URL = "https://api.sleeper.app/v1/players/nfl"
_ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
_FANTASY_POS = {"QB", "RB", "WR", "TE", "K", "DST"}

def _normalize_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the new nflverse `stats_player_week` schema to what
    downstream code expects: rename `team` -> `recent_team` (the historical
    column name) when needed, and drop non-regular-season rows."""
    if "recent_team" not in df.columns and "team" in df.columns:
        df = df.rename(columns={"team": "recent_team"})
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    return df

def fetch_weekly(years):
    # nfl.import_weekly_data reads from the old `player_stats` release, which
    # nflverse has stopped publishing for the current season. The weekly
    # player stats now live under the `stats_player` release tag instead.
    df = pd.concat([pd.read_parquet(WEEKLY_URL.format(year=year)) for year in years])
    return _normalize_weekly(df)

def fetch_snap_counts(years):
    return nfl.import_snap_counts(years)

def fetch_players(_=None):
    # Static bio data; birth_date drives age-matched stat twins.
    df = nfl.import_players()
    return df[["gsis_id", "display_name", "birth_date", "rookie_season"]].dropna(
        subset=["gsis_id"])

def fetch_depth_charts(season):
    return nfl.import_depth_charts([season])

def fetch_schedules(season):
    return nfl.import_schedules([season])

def parse_adp(payload: dict) -> pd.DataFrame:
    rows = [{"adp_name": p["name"],
             "position": "DST" if p["position"] == "DEF" else p["position"],
             "team": p.get("team"), "adp": p["adp"]}
            for p in payload.get("players", [])]
    return pd.DataFrame(rows, columns=["adp_name", "position", "team", "adp"])

def fetch_adp(year: int, teams: int = 12) -> pd.DataFrame:
    resp = requests.get(ADP_URL, params={"teams": teams, "year": year}, timeout=30)
    resp.raise_for_status()
    return parse_adp(resp.json())

def _espn_season_projection(p: dict, year: int):
    # statSourceId 1 = projection (0 = actuals), statSplitTypeId 0 = full
    # season (1 = weekly); appliedTotal is the projected season points.
    for st in p.get("stats") or []:
        if (st.get("statSourceId") == 1 and st.get("statSplitTypeId") == 0
                and st.get("seasonId") == year):
            return st.get("appliedTotal")
    return None

def parse_espn(payload: dict, year: int | None = None) -> pd.DataFrame:
    # `team` is carried for the DST rows' sake: ESPN names a defense
    # "Ravens D/ST" while the board names it from the ADP feed's nickname, so
    # the two never match by name and every DST silently fell through to a
    # position floor. A team has exactly one defense, so team is the join.
    from pipeline.espn_league import ESPN_PRO_TEAMS
    rows = []
    for entry in payload.get("players", []):
        p = entry.get("player") or {}
        pos = _ESPN_POS.get(p.get("defaultPositionId"))
        if pos is None:
            continue
        rows.append({
            "espn_id": p.get("id"), "espn_name": p.get("fullName"), "position": pos,
            "team": ESPN_PRO_TEAMS.get(p.get("proTeamId")),
            "espn_adp": (p.get("ownership") or {}).get("averageDraftPosition"),
            "espn_ppr_rank": ((p.get("draftRanksByRankType") or {}).get("PPR") or {}).get("rank"),
            "espn_proj": _espn_season_projection(p, year) if year else None,
        })
    return pd.DataFrame(rows, columns=["espn_id", "espn_name", "position", "team",
                                       "espn_adp", "espn_ppr_rank", "espn_proj"])

def espn_adp_is_usable(df: pd.DataFrame) -> bool:
    """Whether a season's `espn_adp` column carries real draft positions.

    ESPN serves a reset value -- 170.0 for every player, verified on the
    2025 season -- for some completed seasons, while that season's
    `espn_ppr_rank` stays correct. A constant column is not a ranking, and
    trusting it silently would order a whole season arbitrarily.
    """
    if "espn_adp" not in df.columns:
        return False
    values = pd.to_numeric(df["espn_adp"], errors="coerce").dropna()
    return len(values) > 1 and values.nunique() > 1

def fetch_espn_adp(year: int, limit: int = 500) -> pd.DataFrame:
    headers = {**UA, "X-Fantasy-Filter": json.dumps(
        {"players": {"limit": limit,
                     "sortAdp": {"sortAsc": True, "sortPriority": 1}}})}
    resp = requests.get(ESPN_URL.format(year=year), headers=headers, timeout=30)
    resp.raise_for_status()
    df = parse_espn(resp.json(), year=year)
    if df.empty:
        raise ValueError("no rows parsed - upstream schema drift?")
    return df

# IS_PPR=1 restricts to PPR-scored leagues; PERIOD=DRAFT widens the sample
# to the whole draft season (~2.8k drafts vs ~280 for the recent window).
MFL_ADP_URL = "https://api.myfantasyleague.com/{year}/export?TYPE=adp&JSON=1&IS_PPR=1&PERIOD=DRAFT"
MFL_PLAYERS_URL = "https://api.myfantasyleague.com/{year}/export?TYPE=players&JSON=1"
CBS_URL = "https://www.cbssports.com/fantasy/football/rankings/ppr/top200/"
_MFL_POS = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "PK": "K"}

def parse_mfl(adp_payload: dict, players_payload: dict) -> pd.DataFrame:
    """Join MFL's adp export to its player directory.

    Names arrive "Last, First"; positions use PK for kickers; team-defense
    rows (position "Def") and ids missing from the directory are dropped --
    the board joins by (name, position), which team defenses can't do.
    """
    directory = {p.get("id"): p for p in
                 (players_payload.get("players") or {}).get("player", [])}
    rows = []
    for entry in (adp_payload.get("adp") or {}).get("player", []):
        p = directory.get(entry.get("id"))
        pos = _MFL_POS.get((p or {}).get("position"))
        if p is None or pos is None:
            continue
        last, _, first = (p.get("name") or "").partition(", ")
        rows.append({"mfl_name": f"{first} {last}".strip(), "position": pos,
                     "avg_pick": pd.to_numeric(entry.get("averagePick"), errors="coerce")})
    df = pd.DataFrame(rows, columns=["mfl_name", "position", "avg_pick"])
    if df.empty:
        return pd.DataFrame(columns=["mfl_name", "position", "mfl_rank"])
    df = df.dropna(subset=["avg_pick"]).sort_values("avg_pick").reset_index(drop=True)
    df["mfl_rank"] = df.index + 1
    return df[["mfl_name", "position", "mfl_rank"]]

def fetch_mfl_adp(year: int) -> pd.DataFrame:
    adp = requests.get(MFL_ADP_URL.format(year=year), headers=UA, timeout=30)
    adp.raise_for_status()
    players = requests.get(MFL_PLAYERS_URL.format(year=year), headers=UA, timeout=60)
    players.raise_for_status()
    df = parse_mfl(adp.json(), players.json())
    if df.empty:
        raise ValueError("no rows parsed - upstream schema drift?")
    return df

_CBS_RANK = re.compile(r'<div class="rank">(\d+)</div>')
_CBS_PLAYER = re.compile(r'href="/nfl/players/[^/]+/([a-z0-9-]+)/')
_CBS_POS = re.compile(r'<span class="team position">\s*([A-Z]+)')

def parse_cbs(html: str) -> pd.DataFrame:
    """Rank + full name (from the href slug -- the visible name is
    abbreviated to "J. Gibbs") + position, per player-row block.

    Matched within each block, not across the whole page, so a row without
    a /nfl/players/ href (e.g. a team-defense row) is skipped instead of
    bleeding its rank onto the next player.
    """
    rows = []
    for block in html.split('class="player-row')[1:]:
        rank = _CBS_RANK.search(block)
        player = _CBS_PLAYER.search(block)
        pos = _CBS_POS.search(block)
        if rank and player and pos:
            rows.append({"cbs_name": player.group(1).replace("-", " "),
                         "position": pos.group(1),
                         "cbs_rank": int(rank.group(1))})
    df = pd.DataFrame(rows, columns=["cbs_name", "position", "cbs_rank"])
    # The page embeds each player several times (responsive layout copies).
    return df.sort_values("cbs_rank").drop_duplicates(
        ["cbs_name", "position"], keep="first").reset_index(drop=True)

def fetch_cbs() -> pd.DataFrame:
    resp = requests.get(CBS_URL, headers=UA, timeout=30)
    resp.raise_for_status()
    df = parse_cbs(resp.text)
    if df.empty:
        raise ValueError("no rows parsed - upstream markup drift?")
    return df

def parse_fp_ecr(html: str) -> pd.DataFrame:
    cols = ["fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std", "fp_tier"]
    m = re.search(r"var ecrData = (\{.*?\});", html, re.DOTALL)
    if not m:
        return pd.DataFrame(columns=cols)
    data = json.loads(m.group(1))
    rows = [{"fp_name": p.get("player_name"), "team": p.get("player_team_id"),
             "position": p.get("player_position_id"),
             "rank_ecr": p.get("rank_ecr"),
             "rank_ave": pd.to_numeric(p.get("rank_ave"), errors="coerce"),
             "rank_std": pd.to_numeric(p.get("rank_std"), errors="coerce"),
             "fp_tier": p.get("tier")}
            for p in data.get("players", [])]
    return pd.DataFrame(rows, columns=cols)

def fetch_fp_ecr() -> pd.DataFrame:
    resp = requests.get(FP_URL, headers=UA, timeout=30)
    resp.raise_for_status()
    df = parse_fp_ecr(resp.text)
    if df.empty:
        raise ValueError("no rows parsed - upstream schema drift?")
    return df

def parse_sleeper(payload: dict) -> pd.DataFrame:
    rows = []
    for p in payload.values():
        gsis, espn = p.get("gsis_id"), p.get("espn_id")
        if not gsis or not espn or p.get("position") not in _FANTASY_POS:
            continue
        rows.append({"gsis_id": str(gsis).strip(), "espn_id": espn,
                     "sleeper_name": p.get("full_name"),
                     "position": p.get("position"), "team": p.get("team")})
    return pd.DataFrame(rows, columns=["gsis_id", "espn_id", "sleeper_name",
                                       "position", "team"])

def fetch_sleeper_ids() -> pd.DataFrame:
    resp = requests.get(SLEEPER_URL, headers=UA, timeout=60)
    resp.raise_for_status()
    return parse_sleeper(resp.json())
