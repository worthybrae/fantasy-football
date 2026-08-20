"""Player news + structured injury signals, pulled by `make refresh` and cached.

WHY THIS IS A REFRESH JOB AND NOT A LIVE CALL: the owner asked for a recent
article feed on the player profile. A Google News query costs ~0.5s and the
profile is opened while a draft clock is running, so fetching at draft time
would put a network round-trip in front of the one screen that has to be
instant. `make refresh` already downloads far larger things (the weekly
parquet files dwarf all of this), so the cost lands where nobody is waiting.

THREE FEEDS, DELIBERATELY NOT MERGED INTO ONE UNDIFFERENTIATED THING:

  * ESPN's league news feed -- one request, articles tagged with ESPN
    *athlete ids*. The board already carries `espn_id` for 223 of its 249
    rows, so this join is EXACT: the article really is about that player.
    Narrow, though: measured 2026-08-19, one call returned 50 articles
    carrying 138 distinct athlete tags, of which 87 were players on this
    board (116 attributable items).

  * Google News RSS, one query per player. Broad -- every draftable player
    gets a feed -- but matched on NAME, so attribution is INFERRED, not
    certain. See `google_news_query` for the measurements.

  * Sleeper's player dump, for the structured injury fields. One request,
    covers everybody.

The two article feeds land in the same `player_news` table but every row
carries `attribution`, which is the whole point: ATTR_EXACT rows are
athlete-id-tagged and ATTR_NAME rows are name-matched, so the UI can say so
rather than pretending a ~90%-precise guess is the same kind of fact as an
id join. Do not collapse that column.
"""
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import pandas as pd

# A real user agent, not a bare python-httpx/0.x. Google News RSS answered
# both, but ESPN's edge does not: `site.api.espn.com` (the host the task
# brief named) returns 403 Access Denied to this machine no matter what UA
# is sent, while `site.web.api.espn.com` -- same path, same payload shape --
# answers 200. Verified 2026-08-19 against both hosts.
UA = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")}

GOOGLE_NEWS_URL = ("https://news.google.com/rss/search"
                   "?q={query}&hl=en-US&gl=US&ceid=US:en")
# ESPN caps this feed at 50 articles server-side: limit=100/200/500/1000 all
# return exactly 50 (identical byte counts, verified 2026-08-19). Passing a
# bigger number buys nothing, so the constant says 50 honestly instead of
# implying a knob that does not exist.
ESPN_NEWS_URL = ("https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/news"
                 "?limit=50")
SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"

# The two values of `player_news.attribution`. ATTR_EXACT means ESPN tagged
# the article with this player's athlete id. ATTR_NAME means we searched his
# name and team and believe the result is about him.
ATTR_EXACT = "espn_athlete_id"
ATTR_NAME = "name_team_query"

# How many name-matched items to keep per player. The raw feed is ~100 items
# / 135 KB per player; at 223 players that would be 30 MB of redirect URLs
# for a panel that shows a handful of recent headlines. Exact-attribution
# items are NOT capped -- there are only ever a few and they are the good
# ones.
GOOGLE_ITEMS_PER_PLAYER = 10

# Skip a player whose name-matched rows were fetched less than this long
# ago and carry the stored rows forward instead. `make refresh` on the
# intended cadence (daily or slower) always re-fetches; this only stops a
# second run in the same session from re-querying Google 223 times for
# headlines that cannot have moved much. Google News RSS sends no ETag and
# no Last-Modified, so a conditional GET is not available -- our own
# fetched_at is the only "has this changed" signal there is.
DEFAULT_TTL_HOURS = 12

# Modest on purpose. 223 players * ~0.5s serial is ~2 minutes; four workers
# brings that to well under a minute while keeping us at roughly 8 requests
# per second against a free public feed we do not pay for.
MAX_WORKERS = 4

NEWS_COLUMNS = ["player_id", "name", "headline", "url", "published_at",
                "source", "attribution", "fetched_at"]
STATUS_COLUMNS = ["player_id", "name", "position", "team", "sleeper_id",
                  "injury_status", "injury_body_part", "injury_notes",
                  "practice_participation", "depth_chart_position",
                  "depth_chart_order", "news_updated", "fetched_at"]

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _norm_name(name) -> str:
    """Normalized join key for a player name.

    A deliberate copy of `scoring.board._norm_name` rather than an import:
    this module is in the ingestion half and must stay importable (and
    testable) without dragging in the scoring package. The two must agree --
    if board's normalization changes, change this with it.
    """
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[.'\-]", "", s.lower())
    return " ".join(p for p in s.split() if p not in _SUFFIXES)


# Nicknames, because that is how headlines are written: "Josh Allen Bills"
# is a query, "Josh Allen BUF" is not. Keyed by the nflverse abbreviations
# the board's `team` column actually uses (LA for the Rams, WAS, LV), with
# the same alias set scoring/board.py keeps for feeds that spell them
# differently.
TEAM_NICKNAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
    "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders",
}
_TEAM_ALIASES = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX",
                 "SD": "LAC", "OAK": "LV", "STL": "LA"}


def google_news_query(name: str, team) -> str:
    """The search string for one player: quoted full name plus team nickname.

    NAME + TEAM, AND NOTHING ELSE. This shape was measured, not guessed --
    NFL names collide (the Bills quarterback Josh Allen and the Jaguars
    Josh Allen are the canonical case), and the obvious fixes are not all
    improvements. Counting how many of the returned headlines are about the
    Buffalo quarterback, two independent runs on 2026-08-19:

        Josh Allen                             102 items,  74 relevant
        Josh Allen Bills                       102 items,  89 relevant
        Josh Allen Bills QB fantasy football   100 items,  75 relevant
        "Josh Allen"            (quoted)       102 items,  77 relevant
        "Josh Allen" Bills      (quoted)       102 items,  93 relevant

    So adding the TEAM is a real gain -- ~74% to ~90% precision at no cost
    in volume, quoted or not. Adding the POSITION or "fantasy football"
    makes it WORSE, not better: those terms pull in generic fantasy-football
    roundups that merely mention the player. Do not "improve" this by adding
    them back, and do not simplify it back to a bare name. Quoting the name
    costs nothing (same 102 items) and stops a two-word name from matching
    the two words separately, so it stays.

    ~10% of these items are still about somebody else, which is why every
    row they produce is stamped ATTR_NAME. That marker, not a filter, is how
    the uncertainty is carried downstream -- a filter cannot help, because
    two players with the same name produce headlines with the same words in
    them. (The specific Josh Allen collision has softened since 2024, when
    the Jaguars player started going by Josh Hines-Allen: on 2026-08-19 the
    bare query returned no Jaguars items at all. It has not vanished --
    tests/fixtures/google_news_josh_allen.xml holds a real one that came
    back that day -- and the next same-name pair will not have renamed
    himself.)
    """
    nickname = TEAM_NICKNAMES.get(_TEAM_ALIASES.get(str(team), str(team)))
    q = f'"{name}"' + (f" {nickname}" if nickname else "")
    return GOOGLE_NEWS_URL.format(query=quote_plus(q))


def _utc_naive(dt: datetime | None):
    """Timestamps go into DuckDB tz-naive and in UTC, because the two feeds
    disagree about how they say it (Google sends RFC-822 '... GMT', ESPN
    sends ISO-8601 '...Z') and a column that mixes aware and naive values
    cannot be sorted."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _parse_rfc822(value):
    try:
        return _utc_naive(parsedate_to_datetime(value))
    except (TypeError, ValueError):
        return None


def _parse_iso8601(value):
    try:
        return _utc_naive(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


def parse_google_news(xml_text: str, player_id: str, name: str,
                      limit: int = GOOGLE_ITEMS_PER_PLAYER) -> list[dict]:
    """Rows from one player's Google News RSS body.

    Tolerant on purpose: a single malformed <item> (no title, no link, an
    unparseable pubDate) is skipped and the rest of the feed is kept, and a
    body that is not XML at all raises so the caller can carry that player's
    stored rows forward instead of writing an empty feed over them.
    """
    root = ET.fromstring(xml_text)          # raises on a truncated/HTML body
    rows = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            continue                        # nothing to show, nothing to open
        src_el = item.find("source")
        publication = (src_el.text or "").strip() if src_el is not None else None
        # Google appends " - Publication" to every title and also sends the
        # publication as its own element. Strip the duplicate so the stored
        # headline is the headline.
        if publication and title.endswith(f" - {publication}"):
            title = title[: -(len(publication) + 3)].strip()
        rows.append({
            "player_id": player_id, "name": name, "headline": title,
            # This is the news.google.com/rss/articles/... redirect, not the
            # publisher's own URL. Google stopped putting a decodable target
            # in that token, so the redirect IS the link; it opens fine in a
            # browser and costs ~300 bytes a row.
            "url": link,
            "published_at": _parse_rfc822(item.findtext("pubDate")),
            "source": publication or None,
            "attribution": ATTR_NAME,
        })
    rows.sort(key=lambda r: (r["published_at"] is not None, r["published_at"]),
              reverse=True)
    return rows[:limit]


def parse_espn_news(payload: dict, by_espn_id: dict) -> list[dict]:
    """Rows from ESPN's league feed, one per (article, tagged pool player).

    `by_espn_id` maps ESPN athlete id -> (player_id, name). An article
    tagged with four athletes produces four rows if all four are on the
    board and none if none are; articles about players we do not draft are
    dropped rather than stored unattributed.
    """
    rows = []
    for article in payload.get("articles") or []:
        headline = (article.get("headline") or "").strip()
        url = (((article.get("links") or {}).get("web") or {}).get("href") or "").strip()
        if not headline or not url:
            continue
        published = _parse_iso8601(article.get("published"))
        for cat in article.get("categories") or []:
            if cat.get("type") != "athlete":
                continue
            # Both spellings appear in the payload; `athlete.id` is the one
            # that equals the board's espn_id (verified against Bijan
            # Robinson: athlete.id 4430807 == board espn_id 4430807, while
            # the category's own `id` is 444378, a category key).
            athlete_id = (cat.get("athlete") or {}).get("id") or cat.get("athleteId")
            try:
                athlete_id = int(athlete_id)
            except (TypeError, ValueError):
                continue
            hit = by_espn_id.get(athlete_id)
            if hit is None:
                continue
            player_id, name = hit
            rows.append({"player_id": player_id, "name": name,
                         "headline": headline, "url": url,
                         "published_at": published, "source": "ESPN",
                         "attribution": ATTR_EXACT})
    return rows


def parse_sleeper_status(payload: dict, pool: pd.DataFrame) -> pd.DataFrame:
    """Structured injury signals for the pool, matched by (name, position).

    WHY NAME MATCHING HERE, when Sleeper publishes a `gsis_id` and the task
    brief said that made the join exact: it does not, today. Measured on the
    live payload (12,221 players, 2026-08-19), only 180 of the 1,061 active
    fantasy-position players with a team carry a gsis_id, and 61 of the top
    300 by search_rank -- Bijan Robinson, Jahmyr Gibbs and Ja'Marr Chase all
    have gsis_id AND espn_id null. Joining on gsis_id would silently drop
    four fifths of the board, including the first three players off it. That
    is also why this does not extend `sources.parse_sleeper`/`sleeper_ids`,
    which requires both ids to be non-null and therefore describes mostly
    retired players (1,316 rows, only 184 of them on a roster).

    The name match is safe HERE in a way the news query is not: it is
    restricted to Sleeper's active, rostered fantasy players, and in that
    set (norm name, position) has zero duplicates -- so there is no Josh
    Allen problem, because the Jaguars linebacker is not a fantasy position.
    Measured match rate against the 223 non-DST board rows: 222.
    """
    rows = []
    for sleeper_id, p in (payload or {}).items():
        if not p.get("active") or not p.get("team") or not p.get("full_name"):
            continue
        rows.append({
            "sleeper_id": str(sleeper_id),
            "_norm": _norm_name(p["full_name"]),
            "position": p.get("position"),
            "injury_status": p.get("injury_status"),
            "injury_body_part": p.get("injury_body_part"),
            "injury_notes": p.get("injury_notes"),
            "practice_participation": p.get("practice_participation"),
            "depth_chart_position": p.get("depth_chart_position"),
            "depth_chart_order": p.get("depth_chart_order"),
            # Sleeper's own "when did this player last make news" stamp, in
            # epoch MILLIseconds. Kept because it is the cheapest freshness
            # signal there is. NOT kept: injury_start_date, which the brief
            # listed as useful but which is null for all 12,221 players.
            "news_updated": p.get("news_updated"),
        })
    sleeper = pd.DataFrame(rows, columns=["sleeper_id", "_norm", "position",
                                          "injury_status", "injury_body_part",
                                          "injury_notes", "practice_participation",
                                          "depth_chart_position",
                                          "depth_chart_order", "news_updated"])
    if not sleeper.empty:
        # A duplicate (name, position) among active players is ambiguous, and
        # an ambiguous injury note is worse than none: drop both sides.
        counts = sleeper.groupby(["_norm", "position"])["sleeper_id"].transform("size")
        sleeper = sleeper[counts == 1]

    left = pool.copy()
    left["_norm"] = left["name"].map(_norm_name)
    merged = left.merge(sleeper, on=["_norm", "position"], how="left")
    merged["news_updated"] = pd.to_datetime(
        pd.to_numeric(merged["news_updated"], errors="coerce"),
        unit="ms", errors="coerce")
    merged["depth_chart_order"] = pd.to_numeric(
        merged["depth_chart_order"], errors="coerce").astype("Int64")
    merged["fetched_at"] = pd.Timestamp.utcnow().tz_localize(None)
    return merged.reindex(columns=STATUS_COLUMNS)


def http_fetch(timeout: float = 20.0):
    """A `fetch(url) -> body_text` callable, the injectable seam every test
    replaces. Mirrors `pipeline.espn_teams.http_fetch`; httpx is already a
    project dependency and its Client is thread-safe, which is what lets the
    Google News fan-out share one connection pool.
    """
    import httpx

    client = httpx.Client(headers=UA, timeout=timeout, follow_redirects=True)

    def fetch(url: str) -> str:
        response = client.get(url)
        response.raise_for_status()
        return response.text
    return fetch


def _empty_news() -> pd.DataFrame:
    return pd.DataFrame(columns=NEWS_COLUMNS)


def fetch_espn_news(pool: pd.DataFrame, fetch) -> pd.DataFrame:
    """Exact, athlete-id-tagged items for the pool. One request for everyone."""
    by_espn_id = {}
    for row in pool.itertuples(index=False):
        espn_id = getattr(row, "espn_id", None)
        if espn_id is None or pd.isna(espn_id):
            continue
        by_espn_id[int(espn_id)] = (row.player_id, row.name)
    payload = json.loads(fetch(ESPN_NEWS_URL))
    rows = parse_espn_news(payload, by_espn_id)
    df = pd.DataFrame(rows, columns=NEWS_COLUMNS[:-1])
    df["fetched_at"] = pd.Timestamp.utcnow().tz_localize(None)
    return df


def fetch_google_news(pool: pd.DataFrame, fetch, skip_ids=frozenset(),
                      max_workers: int = MAX_WORKERS,
                      limit: int = GOOGLE_ITEMS_PER_PLAYER):
    """Name-matched items, one query per player.

    Returns (rows_df, failed_ids, attempted_count) -- the count matters
    because "every player we asked about failed" and "we asked about nobody"
    are different situations to the caller, and only the first is a reason to
    fail the whole source.

    DSTs are excluded: a team defense is not a person, "Denver Defense" as a
    search phrase returns whatever the newspaper wrote about the Broncos, and
    an item attributed to a defense on that basis would be noise wearing a
    player's name. They simply have no news feed.

    A player whose fetch or parse raises is collected into `failed_ids`
    rather than aborting -- 223 independent requests over a free feed will
    occasionally lose one, and the caller carries that player's stored rows
    forward so a blip never blanks a profile.
    """
    targets = [r for r in pool.itertuples(index=False)
               if r.position != "DST" and r.player_id not in skip_ids]
    failed = set()
    collected = []

    def one(row):
        try:
            body = fetch(google_news_query(row.name, row.team))
            return parse_google_news(body, row.player_id, row.name, limit=limit)
        except Exception:                     # noqa: BLE001 -- per-player isolation
            failed.add(row.player_id)
            return []

    if targets:
        with ThreadPoolExecutor(max_workers=max_workers) as pool_exec:
            for rows in pool_exec.map(one, targets):
                collected.extend(rows)
    df = pd.DataFrame(collected, columns=NEWS_COLUMNS[:-1])
    df["fetched_at"] = pd.Timestamp.utcnow().tz_localize(None)
    return df, failed, len(targets)


def fetch_player_news(pool: pd.DataFrame, fetch=None, existing=None,
                      ttl_hours: float = DEFAULT_TTL_HOURS,
                      max_workers: int = MAX_WORKERS,
                      limit: int = GOOGLE_ITEMS_PER_PLAYER) -> pd.DataFrame:
    """The whole `player_news` table: ESPN's exact items plus a name-matched
    feed per player, with unchanged players carried forward.

    Fails as a whole ONLY when every feed it actually asked errored -- ESPN
    down AND every attempted Google query failing. Anything less degrades to
    fewer rows, the same one-source-failing-does-not-abort-the-rest rule
    pipeline/refresh.py applies one level up. The distinction that matters:
    "we asked and nobody has news" is a legitimate empty result and must NOT
    raise, or a quiet news day would show up as a red dot on the readiness
    strip.
    """
    fetch = fetch or http_fetch()
    existing = _empty_news() if existing is None else existing
    if not existing.empty:
        existing = existing.reindex(columns=NEWS_COLUMNS)

    # Only the name-matched half is carried forward: ESPN is a single cheap
    # request, so it is always re-read and always fully replaced.
    stored = existing[existing["attribution"] == ATTR_NAME] if not existing.empty else _empty_news()
    fresh_ids = set()
    if not stored.empty and ttl_hours:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(hours=ttl_hours)
        last = stored.groupby("player_id")["fetched_at"].max()
        fresh_ids = set(last[last >= cutoff].index)

    espn, espn_ok = _empty_news(), True
    try:
        espn = fetch_espn_news(pool, fetch)
    except Exception as e:                    # noqa: BLE001 -- one feed of three
        espn_ok = False
        print(f"  WARN espn news feed: {e}")

    google, failed, attempted = fetch_google_news(
        pool, fetch, skip_ids=fresh_ids, max_workers=max_workers, limit=limit)
    if failed:
        print(f"  WARN google news: {len(failed)} player(s) failed; "
              "stored rows carried forward")
    # Carry forward both the skipped-because-fresh players and the failed
    # ones. `carry` is the reason a failure is invisible on the profile: the
    # last good feed stays until a later refresh replaces it.
    carry_ids = fresh_ids | failed
    carried = stored[stored["player_id"].isin(carry_ids)] if not stored.empty else _empty_news()

    if not espn_ok and attempted and len(failed) == attempted:
        raise ValueError("every news feed failed")

    out = pd.concat([espn, google, carried], ignore_index=True)
    if out.empty:
        return _empty_news()
    out = out.drop_duplicates(["player_id", "url"], keep="first")
    # Exact-attribution rows first, then newest-first inside each player, so
    # a consumer that takes the head of a player's rows gets the items we are
    # sure about before the ones we merely believe.
    out["_exact"] = (out["attribution"] == ATTR_EXACT).astype(int)
    out = out.sort_values(["player_id", "_exact", "published_at"],
                          ascending=[True, False, False], na_position="last")
    return out.drop(columns="_exact").reindex(columns=NEWS_COLUMNS).reset_index(drop=True)


def fetch_player_status(pool: pd.DataFrame, fetch=None) -> pd.DataFrame:
    """The whole `player_status` table: one row per pool player, injury
    fields filled in where Sleeper knows about him."""
    fetch = fetch or http_fetch(timeout=60.0)
    payload = json.loads(fetch(SLEEPER_PLAYERS_URL))
    return parse_sleeper_status(payload, pool)


def news_pool(conn) -> pd.DataFrame:
    """Who gets news: exactly the players on the draft board.

    The board is the authority on this codebase's `player_id` (a gsis id for
    a player with stats history, a synthesized `adp_<name>` for one without)
    and on `espn_id`, and reproducing either here would be two copies of a
    matching rule that has to agree to the character or the news silently
    attaches to nothing. pipeline/ already imports from scoring/ for exactly
    this reason (see pipeline/import_league.py, pipeline/run_sim.py).

    Deferred import so `import pipeline.news` -- and every test in this file
    -- stays free of the scoring package and of the ~1.5s board build.
    """
    from scoring.board import build_board
    board = build_board(conn)
    if board.empty:
        return pd.DataFrame(columns=["player_id", "name", "position", "team", "espn_id"])
    return board[["player_id", "name", "position", "team", "espn_id"]].copy()
