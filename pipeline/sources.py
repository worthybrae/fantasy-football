import json
import re

import nfl_data_py as nfl
import pandas as pd
import requests

ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{fmt}"
WEEKLY_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet"

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
ESPN_URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
            "seasons/{year}/segments/0/leaguedefaults/3?view=kona_player_info")
FP_URL = "https://www.fantasypros.com/nfl/rankings/{fmt}-cheatsheets.php"
SLEEPER_URL = "https://api.sleeper.app/v1/players/nfl"
_ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
_FANTASY_POS = {"QB", "RB", "WR", "TE", "K", "DST"}

# Scoring-format tokens shared with the board/market consumer side -- every
# format-aware fetcher in this module accepts one of exactly these three and
# tags its output rows with it in a `format` column. 'ppr' is the default
# everywhere so a caller that doesn't know about formats yet (existing
# refresh jobs, import_league's historic-ADP backfill) keeps getting exactly
# today's rows, just carrying an extra format='ppr' column.
FORMAT_TOKENS = ("ppr", "half", "std")

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
    # Static bio data; birth_date drives age-matched stat twins. `headshot` is
    # nflverse's own NFL.com CDN url -- the profile popup's one image, taken
    # from the source this project already reads rather than hotlinked off a
    # site nothing else here depends on. Null for anyone it has no photo of,
    # which the card renders as no image rather than a broken one.
    #
    # `height` (inches) and `weight` (pounds) are two of the features the
    # profile's Similar players card scores on. Taken from here rather than
    # from a second source because this is already the table every other
    # static fact about a player comes from, and a body measured in one
    # place cannot disagree with itself. A database refreshed before these
    # were added simply has neither column, and `similar_players` drops the
    # features it cannot see rather than failing.
    df = nfl.import_players()
    cols = ["gsis_id", "display_name", "birth_date", "rookie_season",
            "headshot", "height", "weight"]
    return df[[c for c in cols if c in df.columns]].dropna(subset=["gsis_id"])

def fetch_depth_charts(season):
    return nfl.import_depth_charts([season])

def fetch_schedules(season):
    return nfl.import_schedules([season])

# FFC serves one endpoint per scoring format outright -- /adp/ppr,
# /adp/half-ppr, /adp/standard -- and all three return distinct player pools
# (verified: standard's #1 overall is Saquon Barkley, ppr's is Ja'Marr
# Chase). This is the only one of the five sources where every format token
# maps to a genuinely separate live page.
_FFC_FMT_SLUG = {"ppr": "ppr", "half": "half-ppr", "std": "standard"}

def parse_adp(payload: dict, fmt: str = "ppr") -> pd.DataFrame:
    rows = [{"adp_name": p["name"],
             "position": "DST" if p["position"] == "DEF" else p["position"],
             "team": p.get("team"), "adp": p["adp"], "format": fmt}
            for p in payload.get("players", [])]
    return pd.DataFrame(rows, columns=["adp_name", "position", "team", "adp", "format"])

def fetch_adp(year: int, teams: int = 12, fmt: str = "ppr") -> pd.DataFrame:
    if fmt not in _FFC_FMT_SLUG:
        raise ValueError(f"FFC has no {fmt!r} ADP (only {sorted(_FFC_FMT_SLUG)})")
    url = ADP_URL.format(fmt=_FFC_FMT_SLUG[fmt])
    resp = requests.get(url, params={"teams": teams, "year": year}, timeout=30)
    resp.raise_for_status()
    return parse_adp(resp.json(), fmt=fmt)

def _espn_season_projection(p: dict, year: int):
    # statSourceId 1 = projection (0 = actuals), statSplitTypeId 0 = full
    # season (1 = weekly); appliedTotal is the projected season points.
    for st in p.get("stats") or []:
        if (st.get("statSourceId") == 1 and st.get("statSplitTypeId") == 0
                and st.get("seasonId") == year):
            return st.get("appliedTotal")
    return None


# The week the roster rail prices a pick in. A season total is a number nobody
# has a feel for -- 323 is good and 297 is fine and only a reader who already
# knows the scale can tell -- where "18.4 in week 1" reads against a Sunday
# anybody in the league has watched. Week 1 rather than a rolling "next week"
# because this is a DRAFT tool: every roster it shows is being built before a
# ball is thrown, and week 1 is the first game the roster will ever play.
ESPN_PROJECTION_WEEK = 1


def _espn_week_projection(p: dict, year: int, week: int = ESPN_PROJECTION_WEEK):
    """ESPN's projected points for one week, or None.

    The same `stats` list the season projection is read from, with the WEEKLY
    split (`statSplitTypeId: 1`) and the week in `scoringPeriodId`. Measured
    live against the 2026 feed: 293 of the top 300 by ADP carry a week-1
    projection, and the misses are the same tail that has no season projection
    either -- deep bench, unprojected rookies.

    `appliedTotal` is scored under ESPN's own default PPR rules, exactly like
    the season number, so it is converted for a non-PPR league by the same
    `proj_scale` (see scoring/board.py) rather than re-derived here.
    """
    for st in p.get("stats") or []:
        if (st.get("statSourceId") == 1 and st.get("statSplitTypeId") == 1
                and st.get("seasonId") == year
                and st.get("scoringPeriodId") == week):
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
            "espn_wk1": _espn_week_projection(p, year) if year else None,
        })
    return pd.DataFrame(rows, columns=["espn_id", "espn_name", "position", "team",
                                       "espn_adp", "espn_ppr_rank", "espn_proj",
                                       "espn_wk1"])

# ESPN's printable preseason draft cheat sheet: the PPR top 300 as it stood
# BEFORE each season, which is exactly what `draft_model` needs and what the
# `kona_player_info` API above does not reliably give (see
# `draft_model._enrich_pool`). The filename convention changed in 2023.
CHEATSHEET_URL = "https://g.espncdn.com/s/ffldraftkit/{yy}/{name}"
_CHEATSHEET_RENAMED_FROM = 2023

# "12. (RB5) Bijan Robinson, ATL $56 11" -> rank, position, position rank,
# name, team, auction value. The trailing integer is the bye week, which
# nothing reads. Text extracts in a four-column layout, so one physical line
# holds four entries and `findall` (not a per-line parse) is what recovers
# them.
#
# The space before `$` is optional: a long enough name pushes the column
# flush against the team ("Donovan Peoples-Jones, CLE$0 13"), which silently
# cost one player in each of 2021 and 2022 when this required whitespace.
_CHEATSHEET_ROW = re.compile(
    r"(\d+)\.\s+\((QB|RB|WR|TE|K|DST|D/ST)(\d+)\)\s+([^,]+),\s+([A-Z]{2,3})\s*\$(\d+)")

_CHEATSHEET_COLUMNS = ["season", "cs_rank", "position", "cs_name", "team",
                       "auction_value"]


def cheatsheet_url(year: int) -> str:
    yy = f"{year % 100:02d}"
    name = (f"NFL{yy}_CS_PPR300.pdf" if year >= _CHEATSHEET_RENAMED_FROM
            else f"NFLDK{year}_CS_PPR300.pdf")
    return CHEATSHEET_URL.format(yy=yy, name=name)


def parse_espn_cheatsheet_text(text: str, year: int) -> pd.DataFrame:
    """Rows from one cheat sheet's extracted text, deduped to one per rank.

    The extract yields 301 matches for 300 players: rank 1 appears a second
    time as a header artifact. Deduping on rank (keeping the first) is what
    makes "one row per rank, 1..300" hold, rather than trusting the count.

    Split from `parse_espn_cheatsheet` so the parse is testable from a
    literal string rather than requiring a binary PDF fixture.
    """
    rows = [{"season": year, "cs_rank": int(rank),
             # "D/ST" is ESPN's spelling in the position tag; the rest of the
             # codebase says "DST".
             "position": "DST" if pos == "D/ST" else pos,
             "cs_name": name.strip(), "team": team,
             "auction_value": float(value)}
            for rank, pos, _posrank, name, team, value
            in _CHEATSHEET_ROW.findall(text)]
    df = pd.DataFrame(rows, columns=_CHEATSHEET_COLUMNS)
    if df.empty:
        return df
    return (df.sort_values("cs_rank").drop_duplicates("cs_rank", keep="first")
              .reset_index(drop=True))


def parse_espn_cheatsheet(data: bytes, year: int) -> pd.DataFrame:
    from pypdf import PdfReader          # deferred: only this path needs it
    from io import BytesIO

    text = "".join(page.extract_text() for page in PdfReader(BytesIO(data)).pages)
    return parse_espn_cheatsheet_text(text, year)


def fetch_espn_cheatsheet(year: int) -> pd.DataFrame:
    resp = requests.get(cheatsheet_url(year), headers=UA, timeout=60)
    resp.raise_for_status()
    df = parse_espn_cheatsheet(resp.content, year)
    if df.empty:
        raise ValueError(f"no rows parsed from {cheatsheet_url(year)} "
                         "- upstream layout drift?")
    return df


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


# ESPN's D/ST lineup slot. The player endpoint above answers with the 50
# most-owned players in the whole league unless the filter narrows it, and
# defenses are nowhere near that list -- so without `filterSlotIds` this
# returns a healthy 200 carrying zero defenses, the same silent-empty failure
# mode `pipeline/espn_league.PLAYERS_FILTER` exists to prevent. 40 comfortably
# covers 32 teams; `sortPercOwned` only makes the truncation predictable if it
# ever matters.
ESPN_DST_SLOT = 16
ESPN_DST_FILTER = json.dumps({"players": {
    "filterSlotIds": {"value": [ESPN_DST_SLOT]}, "limit": 40,
    "sortPercOwned": {"sortAsc": False, "sortPriority": 1}}})

# The stat blocks worth keeping out of a D/ST payload.
#   statSourceId 0  = what actually happened. 1 is ESPN's projection, which
#                     is a different question and is already answered for
#                     defenses by `espn_proj` on the `espn_adp` table.
#   scoringPeriodId = the NFL week; 0 is the season aggregate, which is a sum
#                     of the weekly rows and would double every total if kept
#                     beside them.
_ESPN_ACTUAL_STATS = 0
_ESPN_SEASON_AGGREGATE = 0

_DST_KEY_COLUMNS = ["season", "week", "espn_id", "team", "applied_total"]


def parse_espn_dst(payload: dict) -> pd.DataFrame:
    """One row per defense per week, carrying ESPN's own raw D/ST stat line.

    Wide, one column per ESPN stat id (`scoring.ppr.dst_stat_column` names
    them, so the writer and `compute_dst_points` cannot drift). Wide rather
    than long because every consumer wants `sum(stat x points)` over a
    defense's weeks, which is one vectorised pass over columns and a
    self-join in long form.

    `season` comes from the STAT BLOCK's own `seasonId`, never from the URL
    year. They disagree, routinely and on purpose: asking ESPN for season
    2026 in August 2026 returns 2025's completed weeks, because 2026 has not
    been played. Keying on the URL year would file last season's games under
    this one.

    `applied_total` is ESPN's own scored total for that week, under
    leaguedefaults/3's rules. It is kept beside the raw stats deliberately --
    it is the check that our scoring is ESPN's scoring (summing
    `scoring.ppr.DEFAULT_DST_RULES` over these columns reproduces it exactly,
    0 mismatches in 3744 defense-weeks) and the honest fallback for a league
    whose own D/ST rules we do not have.

    ESPN omits a stat id from a week's line entirely when it is zero for that
    game, so the union of ids across a season decides the columns and any
    gap is filled with 0.0 -- absent means zero here, not missing.
    """
    from pipeline.espn_league import ESPN_PRO_TEAMS
    from scoring.ppr import dst_stat_column

    rows = []
    for entry in (payload.get("players") or []):
        player = entry.get("player", entry)
        if player.get("id") is None:
            continue
        team = ESPN_PRO_TEAMS.get(player.get("proTeamId"))
        for block in (player.get("stats") or []):
            if block.get("statSourceId") != _ESPN_ACTUAL_STATS:
                continue
            if (block.get("scoringPeriodId") or 0) == _ESPN_SEASON_AGGREGATE:
                continue
            row = {"season": int(block.get("seasonId")),
                   "week": int(block.get("scoringPeriodId")),
                   "espn_id": int(player["id"]),
                   "team": team,
                   "applied_total": float(block.get("appliedTotal") or 0.0)}
            for stat_id, value in (block.get("stats") or {}).items():
                row[dst_stat_column(stat_id)] = float(value)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_DST_KEY_COLUMNS)
    df = pd.DataFrame(rows)
    stat_cols = sorted((c for c in df.columns if c not in _DST_KEY_COLUMNS),
                       key=lambda c: int(c.rsplit("_", 1)[1]))
    df[stat_cols] = df[stat_cols].fillna(0.0)
    # A defense cannot play twice in a week. ESPN has served more than one
    # block for the same (season, week) before -- the season-aggregate id
    # appears twice in the 2026 payload -- so collapse rather than trust it,
    # keeping the first, and sort so the table is stable across refreshes.
    df = df.drop_duplicates(["season", "week", "espn_id"], keep="first")
    return df.sort_values(["season", "week", "espn_id"]).reset_index(
        drop=True)[_DST_KEY_COLUMNS + stat_cols]


def fetch_espn_dst(seasons) -> pd.DataFrame:
    """Per-week D/ST stat lines for `seasons`, stacked into one frame.

    Same host, same endpoint and same default league (3) as `fetch_espn_adp`
    -- which matters beyond tidiness: `espn_proj` for a defense is that
    league's projection, so a defense's history and its projection are quoted
    in the same currency and `scoring.board.projection_scale` can convert
    between that and the owner's rules honestly.

    A season that returns nothing raises rather than being skipped: the whole
    failure mode this guards against is a filtered player endpoint answering
    200 with an empty list, which looks like a healthy fetch and silently
    empties the table (see ESPN_DST_FILTER).
    """
    frames = []
    for season in seasons:
        resp = requests.get(ESPN_URL.format(year=season),
                            headers={**UA, "X-Fantasy-Filter": ESPN_DST_FILTER},
                            timeout=60)
        resp.raise_for_status()
        df = parse_espn_dst(resp.json())
        if df.empty:
            raise ValueError(f"no D/ST rows parsed for {season} "
                             "- upstream schema drift?")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["season", "week", "espn_id"], keep="first")
    return out.sort_values(["season", "week", "espn_id"]).reset_index(drop=True)


# IS_PPR is a binary flag (1 = PPR-scored leagues, 0 = standard), not a
# scoring-format enum, so MFL has no half-PPR ADP export at all -- there is
# no third value to ask for (verified: IS_PPR=1 and IS_PPR=0 both return
# distinct player pools; nothing in between exists). 'half' is deliberately
# absent from _MFL_FMT_IS_PPR; callers that iterate formats must skip it for
# this source. PERIOD=DRAFT widens the sample to the whole draft season
# (~2.8k drafts vs ~280 for the recent window).
MFL_ADP_URL = "https://api.myfantasyleague.com/{year}/export?TYPE=adp&JSON=1&IS_PPR={is_ppr}&PERIOD=DRAFT"
MFL_PLAYERS_URL = "https://api.myfantasyleague.com/{year}/export?TYPE=players&JSON=1"
CBS_URL = "https://www.cbssports.com/fantasy/football/rankings/{fmt}/top200/"
_MFL_POS = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "PK": "K"}
_MFL_FMT_IS_PPR = {"ppr": "1", "std": "0"}

def parse_mfl(adp_payload: dict, players_payload: dict, fmt: str = "ppr") -> pd.DataFrame:
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
        return pd.DataFrame(columns=["mfl_name", "position", "mfl_rank", "format"])
    df = df.dropna(subset=["avg_pick"]).sort_values("avg_pick").reset_index(drop=True)
    df["mfl_rank"] = df.index + 1
    df["format"] = fmt
    return df[["mfl_name", "position", "mfl_rank", "format"]]

def fetch_mfl_adp(year: int, fmt: str = "ppr") -> pd.DataFrame:
    if fmt not in _MFL_FMT_IS_PPR:
        raise ValueError(f"MFL has no {fmt!r} ADP (only {sorted(_MFL_FMT_IS_PPR)})")
    adp_url = MFL_ADP_URL.format(year=year, is_ppr=_MFL_FMT_IS_PPR[fmt])
    adp = requests.get(adp_url, headers=UA, timeout=30)
    adp.raise_for_status()
    players = requests.get(MFL_PLAYERS_URL.format(year=year), headers=UA, timeout=60)
    players.raise_for_status()
    df = parse_mfl(adp.json(), players.json(), fmt=fmt)
    if df.empty:
        raise ValueError("no rows parsed - upstream schema drift?")
    return df

_CBS_RANK = re.compile(r'<div class="rank">(\d+)</div>')
_CBS_PLAYER = re.compile(r'href="/nfl/players/[^/]+/([a-z0-9-]+)/')
_CBS_POS = re.compile(r'<span class="team position">\s*([A-Z]+)')

# CBS has no half-PPR top-200 page -- /rankings/half-ppr/top200/ 404s
# (verified); ppr and standard both 200 with distinct rankings (227 vs 225
# rows, verified). 'half' is deliberately absent; callers that iterate
# formats must skip it for this source.
_CBS_FMT_SLUG = {"ppr": "ppr", "std": "standard"}

def parse_cbs(html: str, fmt: str = "ppr") -> pd.DataFrame:
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
                         "cbs_rank": int(rank.group(1)), "format": fmt})
    df = pd.DataFrame(rows, columns=["cbs_name", "position", "cbs_rank", "format"])
    # The page embeds each player several times (responsive layout copies).
    return df.sort_values("cbs_rank").drop_duplicates(
        ["cbs_name", "position"], keep="first").reset_index(drop=True)

def fetch_cbs(fmt: str = "ppr") -> pd.DataFrame:
    if fmt not in _CBS_FMT_SLUG:
        raise ValueError(f"CBS has no {fmt!r} rankings (only {sorted(_CBS_FMT_SLUG)})")
    url = CBS_URL.format(fmt=_CBS_FMT_SLUG[fmt])
    resp = requests.get(url, headers=UA, timeout=30)
    resp.raise_for_status()
    df = parse_cbs(resp.text, fmt=fmt)
    if df.empty:
        raise ValueError("no rows parsed - upstream markup drift?")
    return df

# FantasyPros only publishes a distinct preseason cheat sheet for PPR --
# half-ppr-cheatsheets.php and standard-cheatsheets.php (and the bare
# cheatsheets.php) all 302-redirect to the same consensus-cheatsheets.php
# page (verified: identical ecrData blob for half and standard). `requests`
# follows the redirect transparently, so both still fetch real, non-empty
# rankings -- just the same ones for 'half' and 'std' alike, distinct from
# 'ppr'. That's an upstream limitation, not a 404, so both formats are kept.
_FP_FMT_SLUG = {"ppr": "ppr", "half": "half-ppr", "std": "standard"}

def parse_fp_ecr(html: str, fmt: str = "ppr") -> pd.DataFrame:
    cols = ["fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std",
            "fp_tier", "format"]
    m = re.search(r"var ecrData = (\{.*?\});", html, re.DOTALL)
    if not m:
        return pd.DataFrame(columns=cols)
    data = json.loads(m.group(1))
    rows = [{"fp_name": p.get("player_name"), "team": p.get("player_team_id"),
             "position": p.get("player_position_id"),
             "rank_ecr": p.get("rank_ecr"),
             "rank_ave": pd.to_numeric(p.get("rank_ave"), errors="coerce"),
             "rank_std": pd.to_numeric(p.get("rank_std"), errors="coerce"),
             "fp_tier": p.get("tier"), "format": fmt}
            for p in data.get("players", [])]
    return pd.DataFrame(rows, columns=cols)

def fetch_fp_ecr(fmt: str = "ppr") -> pd.DataFrame:
    if fmt not in _FP_FMT_SLUG:
        raise ValueError(f"FantasyPros has no {fmt!r} cheatsheet (only {sorted(_FP_FMT_SLUG)})")
    url = FP_URL.format(fmt=_FP_FMT_SLUG[fmt])
    resp = requests.get(url, headers=UA, timeout=30)
    resp.raise_for_status()
    df = parse_fp_ecr(resp.text, fmt=fmt)
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

# ESPN's own season-long futures board, which is the only player-level betting
# market anywhere in this pipeline. The game lines that ride along with
# nflverse `schedules` price a TEAM; these price a player against the field.
#
# Leader markets, not milestones: "most regular season rushing yards" is a
# price on finishing first, not an over/under on a yardage total. A short
# price is a market read on CEILING, which is a different question from ADP
# (where the room takes him) and from the projection (what he is expected to
# do), and that is the whole reason to carry it.
#
# Keyed by ESPN's own athlete id rather than by name. Every other market
# source here joins on a folded name and pays for it -- see
# `oline.reconcile_pfr_to_gsis`, where a son collides with his father -- and
# this one comes with the id the board already carries on `espn_adp`.
FUTURES_URL = ("https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
               "/seasons/{year}/futures?limit=50")

# The markets worth keeping. ESPN posts a couple of dozen, most of them team
# or coach awards that say nothing about a draft pick.
_FUTURES_MARKETS = {
    "Most Regular Season Passing Yards": "pass_yards",
    "Most Regular Season Rushing Yards": "rush_yards",
    "Most Regular Season Receiving Yards": "rec_yards",
    "Regular Season MVP": "mvp",
    "Offensive Player of the Year": "opoy",
    "Offensive Rookie of the Year": "oroy",
}

_FUTURES_COLUMNS = ["season", "market", "espn_id", "american", "implied_pct",
                    "provider"]


def _american_to_implied(price: str | None) -> float | None:
    """A moneyline price as the probability it states.

    +600 is 100 / 700 = 14.3%; -150 is 150 / 250 = 60%. The book's margin is
    left in, because taking it out means assuming how it is distributed across
    a sixty-runner field, and every player on the board is quoted on the same
    inflated scale anyway.
    """
    if not price:
        return None
    try:
        n = float(str(price).replace("+", ""))
    except ValueError:
        return None
    if n == 0:
        return None
    p = 100 / (n + 100) if n > 0 else (-n) / ((-n) + 100)
    return round(p * 100, 2)


def fetch_player_futures(year: int) -> pd.DataFrame:
    """Season-long player futures, one row per (market, player).

    Returns an EMPTY frame rather than raising when ESPN serves no futures for
    a season -- these are posted months before a season and pulled after it, so
    "not up yet" is an ordinary state for this source and not a failed refresh.
    """
    resp = requests.get(FUTURES_URL.format(year=year), headers=UA, timeout=30)
    resp.raise_for_status()
    rows = []
    for item in resp.json().get("items", []):
        market = _FUTURES_MARKETS.get(item.get("name") or item.get("displayName"))
        if market is None:
            continue
        for book in item.get("futures", []):
            provider = (book.get("provider") or {}).get("name")
            for entry in book.get("books", []):
                ref = (entry.get("athlete") or {}).get("$ref", "")
                if "/athletes/" not in ref:
                    continue
                espn_id = ref.split("/athletes/")[-1].split("?")[0]
                if not espn_id.isdigit():
                    continue
                rows.append({
                    "season": year,
                    "market": market,
                    "espn_id": int(espn_id),
                    "american": entry.get("value"),
                    "implied_pct": _american_to_implied(entry.get("value")),
                    "provider": provider,
                })
    return pd.DataFrame(rows, columns=_FUTURES_COLUMNS)
