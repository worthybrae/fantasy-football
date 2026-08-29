"""What the ADP pages know about a player that the corpus does not.

The corpus records where a player went, and nothing else. It does not know
his name, his bye week, what ESPN's cheat sheet made of him in 2022, or that
he was limited in practice on Tuesday. All of that lives in the universal
database this deployment refreshes daily, and this module is the one place
`api/seo.py` reads it.

ONE PASS, THE WHOLE LIST AT ONCE. `build_adp` runs on a warm thread every
four minutes and the pages are a template fill over its result, so nothing
here may be per-page and nothing may be per-player: every source is read
once, turned into a dictionary, and hung onto the two hundred player dicts
in a single loop.

NEVER THE WRONG PLAYER. Three keys, in descending order of confidence:

  1. the board's own `player_id`, which is the corpus's `player_id` --
     `gsis_id` for anybody who has taken an NFL snap and a synthetic
     `adp_<name>` / `adp_<team>_defense` for the rookies and defenses that
     `scoring/board._add_adp_only_players` invents;
  2. the ESPN id the BOARD resolved for him, which is already the two-step
     every ESPN join in this codebase makes (crosswalk, then name and
     position) and covers 179 of 203 where `sleeper_ids` alone covers 61;
  3. normalised name plus position, for the season history, which is keyed
     by name and has no id to offer.

Where none of them matches, the field is absent and the page prints nothing.
A blank says what our coverage is; another player's numbers under this
player's name is a lie, and with a crosswalk this thin the difference comes
up on about one player in eight.

EVERY SOURCE FAILS ALONE. Each helper is wrapped, so a table that is
missing, empty, or an older shape than this code expects costs its own
fields and nothing around them. The pages are already written to lose the
board rather than the page; this is that rule one level further down.
"""
from __future__ import annotations

from datetime import date, datetime

# How many headlines a player page carries. Three is a paragraph's worth of
# "what happened to him lately" and stops a busy name (a starter ruled out in
# August collects a dozen in a day) from crowding the numbers off the page.
NEWS_LIMIT = 3

# The consensus sources, in the order the player page's table lists them and
# with the names a reader would recognise. `espn` is ESPN's PPR expert rank,
# not their ADP -- the two sit in different rows on purpose.
SOURCES = (("ffc", "Fantasy Football Calculator"),
           ("fp", "FantasyPros ECR"),
           ("cbs", "CBS"),
           ("mfl", "MyFantasyLeague"),
           ("espn", "ESPN cheat sheet"))

# The stat lines worth printing beside a projection, per position group. A
# quarterback's projection is a passing line and a running back's is not, and
# printing all fourteen columns for everybody would bury the two that matter.
_PROJ_LINES = {
    "QB": (("proj_pass_yards", "Pass yards"), ("proj_pass_tds", "Pass TD"),
           ("proj_interceptions", "INT"), ("proj_rush_yards", "Rush yards")),
    "RB": (("proj_carries", "Carries"), ("proj_rush_yards", "Rush yards"),
           ("proj_receptions", "Catches"), ("proj_rec_yards", "Rec yards")),
    "WR": (("proj_targets", "Targets"), ("proj_receptions", "Catches"),
           ("proj_rec_yards", "Rec yards"), ("proj_rec_tds", "Rec TD")),
    "TE": (("proj_targets", "Targets"), ("proj_receptions", "Catches"),
           ("proj_rec_yards", "Rec yards"), ("proj_rec_tds", "Rec TD")),
}

# How a DraftKings futures market reads in a sentence.
_FUTURES = {"mvp": "MVP", "opoy": "Offensive Player of the Year",
            "oroy": "Offensive Rookie of the Year",
            "pass_yards": "Most passing yards", "rush_yards": "Most rushing yards",
            "rec_yards": "Most receiving yards"}


def _num(value):
    """`value` as a float, or None for NULL, NaN and anything unparseable.

    Every source here arrives through DuckDB or pandas, and both of them say
    "no value" in more than one way: None from a SQL NULL, NaN from a left
    merge, and pandas' own NA from a nullable integer column. All three mean
    the page prints an em dash, and `value != value` is the only test that
    catches the middle one.
    """
    if value is None or value != value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value):
    got = _num(value)
    return None if got is None else int(round(got))


def _text(value):
    if value is None or value != value:
        return None
    got = str(value).strip()
    return got or None


def _norm(name):
    """The name key every source in this codebase joins on. Imported late:
    `scoring.board` pulls in the whole scoring stack, and this module is
    imported by a web route that may never call `attach`."""
    from scoring.board import _norm_name
    return _norm_name(str(name or ""))


def _latest_season(conn, table: str, default=None):
    try:
        row = conn.execute(f"SELECT max(season) FROM {table}").fetchone()
    except Exception:      # noqa: BLE001 -- no such table, or no season column
        return default
    return None if not row or row[0] is None else int(row[0])


def _in_clause(ids):
    return ",".join("?" * len(ids))


# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------

def board_rows(conn) -> dict:
    """player_id -> the board's row for him, or nothing.

    The board is the single richest source these pages have and it is keyed
    the way the corpus is, so no matching is involved at all. It is also
    already built: `api/market._board_names` asks for it on every one of
    these builds to name the defenses, and `scoring/board_cache` hands the
    second caller the same frame.
    """
    if conn is None:
        return {}
    try:
        from scoring.board_cache import cached_build_board
        board = cached_build_board(conn)
    except Exception:      # noqa: BLE001 -- a board that will not build costs
        # the bye, the tier and the consensus. The corpus's own figures --
        # which are the headline of every page here -- do not depend on it.
        return {}
    out = {}
    try:
        for row in board.itertuples():
            pid = _text(getattr(row, "player_id", None))
            if pid:
                out[pid] = row
    except Exception:      # noqa: BLE001 -- a frame with a surprising shape
        return {}
    return out


def _from_board(player: dict, row) -> None:
    """The board's facts, onto one player dict."""
    player["bye"] = _int(getattr(row, "bye", None))
    player["tier"] = _int(getattr(row, "tier", None))
    player["consensus"] = _num(getattr(row, "market_rank", None))
    player["espn_rank"] = _int(getattr(row, "espn_ppr_rank", None))
    player["proj_points"] = _num(getattr(row, "proj_points", None))
    player["rookie"] = bool(getattr(row, "rookie", False))
    if not player.get("team"):
        player["team"] = _text(getattr(row, "team", None))

    sources = getattr(row, "market_sources", None)
    if isinstance(sources, dict):
        player["fp_tier"] = _int(sources.get("fp_tier"))
        player["sources"] = [
            {"key": key, "label": label, "rank": _num(sources.get(key))}
            for key, label in SOURCES if _num(sources.get(key)) is not None]

    stats = getattr(row, "stats", None)
    if isinstance(stats, dict) and _int(stats.get("games")) and _num(stats.get("points")):
        player["last_season"] = {
            "season": _int(stats.get("season")),
            "games": _int(stats.get("games")),
            "ppg": _num(stats.get("ppg")),
            "points": _num(stats.get("points")),
        }


# ---------------------------------------------------------------------------
# The universal tables
# ---------------------------------------------------------------------------

def _espn_ids(conn, players: list, board: dict) -> dict:
    """player_id -> his ESPN id, from the board first and the crosswalk after.

    The board is the better source: it resolved the id through the gsis
    crosswalk AND the name-and-position fallback when it was built, which is
    179 of 203 where `sleeper_ids` alone reaches 61. The crosswalk is read
    anyway for whoever the board could not answer for, and for the case where
    there is no board to read at all.
    """
    out = {}
    for player in players:
        row = board.get(player["player_id"])
        espn_id = _int(getattr(row, "espn_id", None)) if row is not None else None
        if espn_id is not None:
            out[player["player_id"]] = espn_id
    missing = [p["player_id"] for p in players if p["player_id"] not in out]
    if not missing:
        return out
    try:
        rows = conn.execute(
            "SELECT gsis_id, espn_id FROM sleeper_ids"
            f" WHERE gsis_id IN ({_in_clause(missing)}) AND espn_id IS NOT NULL",
            missing).fetchall()
    except Exception:      # noqa: BLE001 -- no crosswalk table on this database
        return out
    for gsis_id, espn_id in rows:
        got = _int(espn_id)
        if got is not None:
            out.setdefault(str(gsis_id), got)
    return out


def _bio(conn, ids: list) -> dict:
    """gsis_id -> birth date, rookie season, height and weight."""
    if not ids:
        return {}
    try:
        rows = conn.execute(
            "SELECT gsis_id, birth_date, rookie_season, height, weight"
            f" FROM players WHERE gsis_id IN ({_in_clause(ids)})", ids).fetchall()
    except Exception:      # noqa: BLE001 -- an older `players` without the
        # bio columns costs an age and a height, nothing else.
        return {}
    out = {}
    for pid, born, rookie_season, height, weight in rows:
        out[str(pid)] = {"age": _age(born), "rookie_season": _int(rookie_season),
                         "height": _int(height), "weight": _int(weight)}
    return out


def _age(born) -> int | None:
    """Whole years today. `birth_date` is stored as text, so it is parsed
    rather than subtracted, and an unparseable one is simply no age."""
    if isinstance(born, datetime):
        born = born.date()
    if not isinstance(born, date):
        text = _text(born)
        if not text:
            return None
        try:
            born = datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    today = date.today()
    years = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    return years if 15 < years < 60 else None


def _history(conn) -> tuple:
    """Two season histories, keyed by normalised name and position.

    `historic_adp` is where the market had him each August back to 2020;
    `historic_espn_cs` is ESPN's own cheat sheet, which reaches this season
    and carries an auction value beside the rank. Name plus position because
    neither table has an id to offer -- and position, not name alone, because
    the same name has belonged to a quarterback and a linebacker before now.
    """
    adp: dict = {}
    try:
        for season, name, position, rank in conn.execute(
                "SELECT season, adp_name, position, adp_rank FROM historic_adp"
        ).fetchall():
            key = (_norm(name), _text(position) or "")
            if _int(rank) is not None:
                adp.setdefault(key, {})[int(season)] = _int(rank)
    except Exception:      # noqa: BLE001 -- no history is a shorter table
        adp = {}
    cheat: dict = {}
    try:
        for season, rank, position, name, value in conn.execute(
                "SELECT season, cs_rank, position, cs_name, auction_value"
                " FROM historic_espn_cs").fetchall():
            key = (_norm(name), _text(position) or "")
            if _int(rank) is not None:
                cheat.setdefault(key, {})[int(season)] = (_int(rank), _num(value))
    except Exception:      # noqa: BLE001
        cheat = {}
    return adp, cheat


def _projections(conn) -> tuple:
    """This season's ESPN projection, by ESPN id and by name-and-position."""
    season = _latest_season(conn, "espn_projections")
    if season is None:
        return {}, {}, None
    try:
        rows = conn.execute(
            "SELECT * FROM espn_projections WHERE season = ?", [season]).fetchall()
        columns = [d[0] for d in conn.description]
    except Exception:      # noqa: BLE001
        return {}, {}, None
    by_id, by_name = {}, {}
    for row in rows:
        record = dict(zip(columns, row))
        espn_id = _int(record.get("espn_id"))
        if espn_id is not None:
            by_id[espn_id] = record
        key = (_norm(record.get("espn_name")), _text(record.get("position")) or "")
        by_name.setdefault(key, record)
    return by_id, by_name, season


def _projection_view(record: dict, position: str) -> dict | None:
    points = _num(record.get("proj_points"))
    if points is None:
        return None
    lines = []
    for column, label in _PROJ_LINES.get(position.upper(), ()):
        value = _num(record.get(column))
        if value is not None and abs(value) >= 0.5:
            lines.append({"label": label, "value": int(round(value))})
    return {"points": round(points, 1), "games": _int(record.get("proj_games")),
            "lines": lines}


def _futures(conn) -> dict:
    """ESPN id -> the sportsbook markets he is quoted in this season."""
    season = _latest_season(conn, "player_futures")
    if season is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT espn_id, market, american, implied_pct, provider"
            " FROM player_futures WHERE season = ?", [season]).fetchall()
    except Exception:      # noqa: BLE001
        return {}
    out: dict = {}
    for espn_id, market_name, american, pct, provider in rows:
        key = _int(espn_id)
        share = _num(pct)
        if key is None or share is None:
            continue
        out.setdefault(key, []).append({
            "market": _FUTURES.get(str(market_name), str(market_name)),
            "odds": _text(american), "pct": share,
            "provider": _text(provider) or "DraftKings"})
    for entries in out.values():
        entries.sort(key=lambda e: -e["pct"])
    return out


def _last_season(conn, ids: list) -> dict:
    """gsis_id -> what he actually did last season, week by week.

    The board carries games, points per game and a season total already, and
    they agree with this; what it does not carry is the shape of the season,
    which is the interesting half. A 17-game 14.0 with a 38-point week is not
    the same player as a 17-game 14.0 that never cleared 22.
    """
    season = _latest_season(conn, "weekly")
    if season is None or not ids:
        return {}
    try:
        rows = conn.execute(
            "SELECT player_id, count(*), sum(fantasy_points_ppr),"
            " max(fantasy_points_ppr), min(fantasy_points_ppr),"
            " arg_max(week, fantasy_points_ppr), arg_max(opponent_team, fantasy_points_ppr)"
            " FROM weekly WHERE season = ? AND season_type = 'REG'"
            f" AND player_id IN ({_in_clause(ids)})"
            " AND fantasy_points_ppr IS NOT NULL GROUP BY 1",
            [season, *ids]).fetchall()
    except Exception:      # noqa: BLE001 -- a fixture `weekly` without these
        # columns costs the season line, not the page.
        return {}
    out = {}
    for pid, games, total, best, worst, week, opponent in rows:
        played = _int(games) or 0
        points = _num(total)
        # A zero season is not a season this table can speak for. nflverse
        # computes `fantasy_points_ppr` from the passing, rushing and
        # receiving lines and nothing else, so every kicker in it reads 0.0
        # over seventeen games -- and "PPR points 0" beside a kicker who
        # scored 140 is worse than no line at all.
        if not played or not points:
            continue
        out[str(pid)] = {
            "season": season, "games": played,
            "points": round(points, 1), "ppg": round(points / played, 1),
            "best": None if _num(best) is None else round(_num(best), 1),
            "worst": None if _num(worst) is None else round(_num(worst), 1),
            "best_week": _int(week), "best_opponent": _text(opponent)}
    return out


def _dst_season(conn) -> dict:
    """team -> last season's defensive fantasy line, for the D/STs.

    A defense has no `weekly` row and no `players` row: it is a synthetic
    board id whose only real key is the team abbreviation.
    """
    season = _latest_season(conn, "dst_weekly")
    if season is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT team, count(*), sum(applied_total), max(applied_total)"
            " FROM dst_weekly WHERE season = ? AND applied_total IS NOT NULL"
            " GROUP BY 1", [season]).fetchall()
    except Exception:      # noqa: BLE001
        return {}
    out = {}
    for team, games, total, best in rows:
        played = _int(games) or 0
        points = _num(total)
        if not played or points is None:
            continue
        out[str(team).upper()] = {
            "season": season, "games": played, "points": round(points, 1),
            "ppg": round(points / played, 1),
            "best": None if _num(best) is None else round(_num(best), 1),
            "worst": None, "best_week": None, "best_opponent": None}
    return out


def _news(conn, ids: list) -> dict:
    """gsis_id -> his newest headlines, newest first."""
    if not ids:
        return {}
    try:
        rows = conn.execute(
            "SELECT player_id, headline, url, source, published_at"
            f" FROM player_news WHERE player_id IN ({_in_clause(ids)})"
            " ORDER BY published_at DESC", ids).fetchall()
    except Exception:      # noqa: BLE001
        return {}
    out: dict = {}
    for pid, headline, url, source, published in rows:
        entries = out.setdefault(str(pid), [])
        if len(entries) >= NEWS_LIMIT:
            continue
        text = _text(headline)
        if not text:
            continue
        link = _text(url)
        entries.append({
            "headline": text,
            # Only a plain web link is printed. These rows come from a feed
            # this process does not control, and an href is the one field on
            # these pages that would execute something if it carried a
            # scheme -- the same rule `build_adp` applies to a headshot.
            "url": link if link and link.startswith(("https://", "http://")) else None,
            "source": _text(source),
            "date": published.date() if isinstance(published, datetime) else None})
    return out


def _status(conn, ids: list) -> dict:
    """gsis_id -> injury designation and practice participation.

    "None" is a real value in this table and means healthy, so it is dropped
    rather than printed: a page that says "Injury status: None" reads like a
    gap in the data rather than a fit player.
    """
    if not ids:
        return {}
    try:
        rows = conn.execute(
            "SELECT player_id, injury_status, injury_body_part, practice_participation"
            f" FROM player_status WHERE player_id IN ({_in_clause(ids)})",
            ids).fetchall()
    except Exception:      # noqa: BLE001
        return {}
    out = {}
    for pid, status, body_part, practice in rows:
        designation = _text(status)
        if designation and designation.lower() in ("none", "active", "healthy"):
            designation = None
        share = _num(practice)
        if designation is None and share is None:
            continue
        out[str(pid)] = {"injury": designation, "body_part": _text(body_part),
                         "practice": None if share is None else round(share, 2)}
    return out


# ---------------------------------------------------------------------------
# The one pass
# ---------------------------------------------------------------------------

def attach(conn, players: list) -> None:
    """Hang every universal fact onto every player dict, in place.

    `players` is `build_adp`'s list: each entry already carries
    `player_id`, `name`, `position` and whatever team the depth charts knew.
    Nothing is returned -- the aggregate is the dict the pages render, and a
    parallel structure keyed the same way would only be a second thing to
    keep in step with it.
    """
    if not players:
        return
    board = board_rows(conn)
    for player in players:
        row = board.get(player["player_id"])
        if row is not None:
            _from_board(player, row)
    if conn is None:
        return

    espn_ids = _espn_ids(conn, players, board)
    # Everybody who is a real player rather than one of the board's synthetic
    # `adp_<team>_defense` rows: those have no `players` row, no `weekly` row
    # and no injury report, and asking for them is four queries with a
    # guaranteed empty answer.
    gsis = [p["player_id"] for p in players
            if not str(p["player_id"]).startswith("adp_")]
    teams = {(p.get("team") or "").upper() for p in players if p["position"] == "DST"}

    bio = _bio(conn, gsis)
    adp_history, cheat_history = _history(conn)
    proj_by_id, proj_by_name, proj_season = _projections(conn)
    futures = _futures(conn)
    last = _last_season(conn, gsis)
    dst = _dst_season(conn) if teams else {}
    news = _news(conn, gsis)
    status = _status(conn, gsis)

    for player in players:
        pid = player["player_id"]
        position = player["position"]
        key = (_norm(player["name"]), position)

        got = bio.get(pid)
        if got:
            player.update({k: v for k, v in got.items() if v is not None})

        seasons = _season_table(adp_history.get(key), cheat_history.get(key))
        if seasons:
            player["seasons"] = seasons
            current = seasons[-1]
            if current.get("auction") is not None:
                player["auction"] = current["auction"]
            if current.get("cheat") is not None:
                player["cheat_rank"] = current["cheat"]

        espn_id = espn_ids.get(pid)
        record = proj_by_id.get(espn_id) if espn_id is not None else None
        if record is None:
            record = proj_by_name.get(key)
        if record is not None:
            view = _projection_view(record, position)
            if view is not None:
                view["season"] = proj_season
                player["projection"] = view

        if espn_id is not None and espn_id in futures:
            player["futures"] = futures[espn_id]

        line = last.get(pid)
        if line is None and position == "DST":
            line = dst.get((player.get("team") or "").upper())
        if line is not None:
            player["last_season"] = line

        if pid in news:
            player["news"] = news[pid]
        if pid in status:
            player["status"] = status[pid]


def _season_table(adp: dict | None, cheat: dict | None) -> list:
    """One row per season either source has an opinion in, oldest first."""
    if not adp and not cheat:
        return []
    rows = []
    for season in sorted(set(adp or {}) | set(cheat or {})):
        rank = (adp or {}).get(season)
        cheat_rank, value = (cheat or {}).get(season, (None, None))
        rows.append({"season": season, "adp": rank, "cheat": cheat_rank,
                     "auction": None if value is None else round(value, 1)})
    return rows


def sparkline(seasons: list, width: int = 140, height: int = 34) -> str | None:
    """The season history as one `polyline` points attribute.

    Drawn from whichever rank each season has -- the market's if it exists,
    ESPN's cheat sheet otherwise -- because the shape being asked about is
    "has he been drifting up or down", and a line that breaks wherever one
    feed missed a year answers it worse than a line that does not.

    Y IS INVERTED: rank 1 is the top of the box. Ranks are a scale where
    smaller is better and a chart that draws them the other way up says the
    opposite of what it means.
    """
    points = [(row["season"], row["adp"] if row["adp"] is not None else row["cheat"])
              for row in seasons or []]
    points = [(season, rank) for season, rank in points if rank is not None]
    if len(points) < 2:
        return None
    ranks = [rank for _season, rank in points]
    low, high = min(ranks), max(ranks)
    span = (high - low) or 1
    step = width / (len(points) - 1)
    return " ".join(
        f"{round(i * step, 1)},{round((rank - low) / span * (height - 6) + 3, 1)}"
        for i, (_season, rank) in enumerate(points))
