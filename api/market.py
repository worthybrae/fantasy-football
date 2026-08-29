"""What real drafts actually do, from the seat you are actually in.

WHY THIS EXISTS. Every draft tool publishes an average draft position, and an
average is the one number a draft never produces: nobody drafts at the mean.
`pipeline/mock_farm.py` has been sitting in real ESPN mock drafts around the
clock, so this deployment holds hundreds of complete drafts made by people --
who they took, in what order, from which seat, and how much they disagreed
with each other. That is a market, observed rather than published, and these
endpoints are the questions worth asking of it:

  * From MY seat, what does the room do at each of my turns -- and which of
    those turns are effectively decided before I sit down (pick 1 is the same
    player three drafts in four) versus genuinely open (round 8 is sixty
    different names)?
  * What does a whole draft from my seat usually look like -- the position
    paths people actually walk, in order, not a strategy article's version.
  * For one player: not his ADP, but the SHAPE of where he goes. Earliest,
    latest, the spread between, and how often he is still there at each of my
    own turns -- which is the empirical twin of the model's `LASTS` column,
    counted rather than simulated.

WHOSE PICKS COUNT. Behaviour questions read HUMAN picks only: ESPN's
autodrafter filling an empty seat is not a person's opinion, and neither is
the farm's own seat, which drafts with this tool and would be this project
marking its own homework. Availability questions ("was he still there") read
EVERY pick, because a player taken by a bot is gone exactly as thoroughly.

READ-ONLY, AND POLITE ABOUT THE LOCK. The farm writes to this database at the
end of every draft it finishes. Every read here is `read_only=True`, wrapped,
and cached for minutes at a time -- a lock held for a moment costs a stale
answer, never an error on a page.

OPEN. This is the data store, and there is nothing in it to protect: it holds
public mock drafts played by strangers, aggregated, with no credential, no
account and no name attached to a seat. It was gated behind an account at
first on the reasoning that the corpus is the thing this deployment has that
nobody else does -- which is an argument for showing it to people, not for
hiding it from them.
"""
from __future__ import annotations

import io
import json
import threading
import time
import zipfile

import numpy as np
from fastapi import HTTPException, Response

from api import http_cache
from pipeline import draft_log as dl
from scoring.headshot import thumb

# How long an answer stands. The corpus grows by one draft every few minutes
# and no question here moves visibly in that time -- a tenth of a percent on
# one share -- so this is about not opening the database on every keystroke of
# a page that is mostly reading and filtering.
CACHE_SECONDS = 600.0

# How long a shared cache in front of this process may hold one. Shorter than
# the process's own window on purpose: this one is not invalidated by a
# restart, so a deploy that changes what these answers say cannot clear it,
# and five minutes is how long the wrong answer would stand.
SHARED_CACHE_SECONDS = 300

# How long a THIN answer stands. The board is built on demand and can be cold,
# mid-rebuild or momentarily unreadable when a request lands, and every helper
# here that needs it is written to lose the board rather than the page: no
# projections, and no names for the ids only the board carries (the ADP-only
# rookies and defenses -- see `_names`). That is the right call for one
# response and the wrong one for ten minutes, which is how long a payload
# built during those few seconds used to sit in the cache: a seat whose page
# read "adp_carnell_tate 6%" beside "David Montgomery 10%", with no ppg on
# either, until the entry expired. Twenty seconds is long enough that a board
# genuinely down is not rebuilt on every request, and short enough that a
# reader who reloads gets the real page.
DEGRADED_SECONDS = 20.0

# The turns whose position path counts as "a strategy". Five rounds is where
# the shape of a draft is decided and is short enough that the paths group:
# over the whole sixteen, nearly every draft is its own unique string and the
# list says nothing.
SEQUENCE_ROUNDS = 5

# How many names to name per turn. Six covers the head of every turn's
# distribution measured (at the most open turn in the draft the sixth name is
# already under 4%) without turning one row into a page.
TOP_PER_TURN = 6

# The pick axis, in bins. Sixteen bins over a 128-pick draft is one bin per
# round, which is the unit a drafter thinks in -- "he goes in the third" --
# and is fine enough to show a two-humped distribution, which is the shape
# worth spotting (a player the room disagrees about, not one it is unsure
# about).
HIST_BINS = 16

_CACHE: dict = {}
_LOCK = threading.Lock()


def _cached(key, build, complete=None):
    """One answer per key per `CACHE_SECONDS`, computed at most once.

    `complete(value)` says whether what was built is the whole answer. One
    that is not -- the board did not answer, so the payload is missing the
    part only the board can supply -- is still served, and still cached so a
    burst of requests does not each retry a board that is down, but only for
    `DEGRADED_SECONDS`. An endpoint that does not depend on the board passes
    nothing and keeps the full life.
    """
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    value = build()
    ttl = CACHE_SECONDS if complete is None or complete(value) else DEGRADED_SECONDS
    with _LOCK:
        _CACHE[key] = (now + ttl, value)
    return value


def store(key, value, ttl: float = CACHE_SECONDS) -> None:
    """Put an answer built elsewhere into the cache, with a full life.

    For a caller that wants to pay for a REBUILD off the request path and
    then hand it over -- the background thread in `api/seo.py`, which exists
    so that no reader is ever the one who finds `build_adp` cold.

    A swap, deliberately, and not a retire-then-rebuild: the entry is
    replaced in one step under this lock, so there is never a moment when
    the key is missing and a reader arriving in it starts a build of their
    own. That was the whole point of doing the work outside the lock, and
    an eviction would have handed the cost straight back.
    """
    with _LOCK:
        _CACHE[key] = (time.monotonic() + ttl, value)


def _board_answered(rows) -> bool:
    """Whether the board contributed to these rows at all.

    Read off `proj_points`, which only the board can fill: with a board there
    the head of any turn's distribution is priced, and without one every row
    in the payload is a null. It is deliberately not "is every name filled
    in" -- an id in the corpus that has since fallen off the board entirely
    (an ADP list is a moving thing) would then mark every payload thin
    forever and rebuild each of them every twenty seconds.

    No rows at all is not a thin answer: it is what the farm's own seat
    honestly looks like, and it has nothing for the board to say.
    """
    rows = list(rows)
    return not rows or any(row.get("proj_points") is not None for row in rows)


def _corpus():
    import duckdb
    try:
        return duckdb.connect(dl.CORPUS_PATH, read_only=True)
    except Exception as exc:      # noqa: BLE001 -- a farm mid-write, or no
        # corpus at all on a fresh install. Both are "come back in a minute",
        # not a server error.
        raise HTTPException(
            status_code=503,
            detail="The draft archive is busy being written to. Try again in "
                   "a moment.") from exc


def shape_filter(alias: str = "d") -> str:
    """SQL for "this draft is the one the archive is about".

    `_shape` leaves the chosen shape's draft ids in a temp table on the same
    connection, and every query here filters through it rather than
    restating columns. That is not tidiness: the shape is `(teams, rounds,
    format)` and the format lives in a JSON blob, so a `WHERE teams = ? AND
    rounds = ?` cannot express it -- and a 10-team PPR archive quietly
    including 10-team standard drafts is a page of numbers about two
    different games.
    """
    return (f"{alias + '.' if alias else ''}draft_id IN "
            "(SELECT draft_id FROM shape_draft)")


def _shape(conn) -> tuple:
    """The league shape the archive is about: `(teams, rounds, format)`.

    Mixing shapes is the one thing that makes every number here meaningless.
    Pick 14 is round 2 in an 8-team draft and round 1 in a 16-team one; and
    in PPR a receiver goes a round before the same receiver in standard. So
    ONE shape is chosen -- the most recorded -- and it is stated in every
    answer rather than assumed by the page.

    THE TIE-BREAK IS TOTAL AND DETERMINISTIC (`draft_log.dominant_shape`,
    shared with the fitted prior so the two cannot disagree about what the
    corpus is): most drafts first, then the smaller shape, then PPR before
    half before standard. Two shapes with the same count is not a
    hypothetical while the farm is rotating over five of them, and a page
    whose shape flipped between requests -- because DuckDB is free to return
    equal groups in any order -- would serve two different archives under one
    URL and cache whichever it saw first.

    Leaves that shape's draft ids in a temp table (`shape_draft`) on this
    connection, which is what `shape_filter` reads. Temp, so it belongs to
    this request's connection and cannot outlive it; a read-only DuckDB
    connection allows it because nothing is written to the file.
    """
    rows = conn.execute(
        "SELECT draft_id, teams, rounds,"
        " coalesce(scoring_json, settings_json) AS scoring"
        " FROM draft_log WHERE teams > 0 AND rounds > 0"
        " ORDER BY draft_id").fetchall()
    counts: dict = {}
    for draft_id, teams, rounds, scoring in rows:
        shape = (int(teams), int(rounds), dl.draft_format(scoring))
        counts.setdefault(shape, []).append(str(draft_id))
    if not counts:
        raise HTTPException(status_code=404,
                            detail="No drafts have been recorded yet.")
    teams, rounds, fmt = dl.dominant_shape(
        {shape: len(ids) for shape, ids in counts.items()})
    conn.execute("CREATE OR REPLACE TEMP TABLE shape_draft AS "
                 "SELECT unnest(?::VARCHAR[]) AS draft_id",
                 [counts[(teams, rounds, fmt)]])
    return teams, rounds, fmt


def _human_picks_sql() -> str:
    """The picks that are somebody's opinion.

    Excludes ESPN's autodrafter (an empty seat's picks are its ADP list read
    aloud) and the farm's own seat (this tool's picks, which would make every
    "the room agrees with you" answer circular).
    """
    return ("COALESCE(pk.autodrafted, FALSE) = FALSE"
            " AND (d.my_slot IS NULL OR pk.slot <> d.my_slot)")


def _overview_payload() -> dict:
    conn = _corpus()
    try:
        teams, rounds, fmt = _shape(conn)
        row = conn.execute(
            "SELECT count(*), min(recorded_at), max(recorded_at),"
            f" median(human_seats) FROM draft_log WHERE {shape_filter('')}"
        ).fetchone()
        picks = conn.execute(
            "SELECT count(*) FROM draft_log_pick pk JOIN draft_log d"
            f" USING (draft_id) WHERE {shape_filter()}").fetchone()[0]
        human = conn.execute(
            f"SELECT count(*) FROM draft_log_pick pk JOIN draft_log d"
            f" USING (draft_id) WHERE {shape_filter()}"
            f" AND {_human_picks_sql()}").fetchone()[0]
    finally:
        conn.close()
    return {
        "drafts": int(row[0]),
        "picks": int(picks),
        "human_picks": int(human),
        "teams": teams,
        "rounds": rounds,
        # The archive's own shape, said rather than implied. Every count
        # above is over drafts of exactly this size AND this scoring, so a
        # page that printed "PPR" over a standard-scoring archive would be
        # attributing the numbers to a game nobody played.
        "format": fmt,
        "from": None if row[1] is None else str(row[1]),
        "to": None if row[2] is None else str(row[2]),
        "median_humans": None if row[3] is None else float(row[3]),
    }


def _names(conn) -> dict:
    """player_id -> display name and headshot, from this deployment's board.

    The corpus records who was taken, not who he is: it is a record of drafts
    and has no business holding a name that changes when a player is traded.

    TWO SOURCES, IN ORDER. `players` is the nflverse roster, keyed by
    `gsis_id`, and it covers every player who has taken an NFL snap. It does
    NOT cover everyone a draft can contain: `scoring/board._add_adp_only_
    players` synthesizes rows for players who appear in the ADP consensus and
    nowhere else -- rookies with no NFL season yet, and every defense -- under
    ids of its own making (`adp_jadarian_price`, `adp_seattle_defense`). Those
    ids are real picks in the corpus, and looking them up in `players` returns
    nothing, so the archive printed the id itself: rows reading
    "adp_jonathon_brooks" next to rows reading "Derrick Henry".

    The board is where those names live, so the board is the fallback -- read
    ONLY when an id actually failed to resolve, since building one costs
    seconds on a cold cache (`scoring/board_cache`) and the roster answers for
    almost everything.
    """
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT gsis_id, display_name, headshot FROM players").fetchall()
    except Exception:      # noqa: BLE001 -- a board that cannot be read costs
        # names, not the page: every row still has a position and a market.
        return {}
    # `thumb` (scoring/headshot.py): these are the faces on the archive's
    # pick cards and on every /adp page, drawn no larger than 72 pixels,
    # and the stored url is nflverse's 3400x2450 original.
    return {str(pid): {"name": name, "headshot": thumb(shot)}
            for pid, name, shot in rows if pid}


def _board_names(conn, missing: set) -> dict:
    """The names `players` does not carry, from the board itself.

    Called only with the ids that failed to resolve (see `_names`), and it is
    allowed to fail: a board that will not build costs a few names, never the
    answer around them.
    """
    if conn is None or not missing:
        return {}
    try:
        from scoring.board_cache import cached_build_board
        board = cached_build_board(conn)
    except Exception:      # noqa: BLE001
        return {}
    out = {}
    for row in board.itertuples():
        pid = str(getattr(row, "player_id", ""))
        if pid in missing:
            name = getattr(row, "name", None)
            shot = getattr(row, "headshot", None)
            # Already sized: `headshot` is a board column and the board
            # sizes it at build (scoring/board.py). `thumb` is idempotent, so
            # this stays correct either way.
            out[pid] = {"name": None if name is None else str(name),
                        "headshot": None if shot is None or shot != shot else str(shot)}
    return out


def _proj(board_conn, ids) -> dict:
    """player_id -> what this deployment's board projects him for, or nothing.

    The corpus records who was taken; it has no opinion on what he is worth,
    and it should not -- a draft from three weeks ago carries that day's
    projection, and reading it back would price a seat off a stale number.
    The board is where the current one lives.

    OFF THE BOARD, NOT OUT OF `players`: `proj_points` is computed
    (`scoring/board.py` scales ESPN's projection into this league's scoring),
    so there is no column to select. `cached_build_board` is the same cached
    frame `_board_names` already builds on this endpoint for every defense and
    every rookie in the corpus, so asking for it here usually costs nothing
    on top -- and this payload is itself cached for ten minutes.

    A board that will not build costs the projections and nothing else: the
    names, the shares and the picks around them are the corpus's own facts and
    do not depend on it.
    """
    wanted = {str(pid) for pid in ids}
    if board_conn is None or not wanted:
        return {}
    try:
        from scoring.board_cache import cached_build_board
        board = cached_build_board(board_conn)
    except Exception:      # noqa: BLE001 -- see the docstring: a projection
        # this payload cannot reach is a null in one field, not a failed page.
        return {}
    out = {}
    for row in board.itertuples():
        pid = str(getattr(row, "player_id", ""))
        if pid not in wanted:
            continue
        points = getattr(row, "proj_points", None)
        # The board's own per-game delta on his recent seasons
        # (scoring/board.py: proj_ppg - w_ppg), served beside the projection
        # so a name at a pick says whether it is priced above or below what
        # he just did.
        change = getattr(row, "proj_change", None)
        # NaN is not a projection. It compares unequal to itself, which is the
        # check here, and a JSON `NaN` would reach the page as a number it
        # cannot format.
        if points is not None and points == points:
            out[pid] = {
                "points": float(points),
                "change": (float(change)
                           if change is not None and change == change else None),
            }
    return out


def _resolve(board_conn, ids) -> dict:
    """Names for these ids: the roster first, the board for whatever it missed.

    One place, so every answer this module serves names the same player the
    same way -- and so the board is built at most once per payload, only when
    something actually needs it.
    """
    names = _names(board_conn)
    missing = {str(pid) for pid in ids if str(pid) not in names}
    if missing:
        names.update(_board_names(board_conn, missing))
    return names


def _turn_picks(teams: int, slot: int) -> list:
    """The overall pick numbers this seat owns, in order. The snake, stated
    once -- odd rounds run out from slot 1, even rounds run back."""
    picks = []
    for rnd in range(1, 1 + 16):
        base = (rnd - 1) * teams
        picks.append(base + (slot if rnd % 2 == 1 else teams - slot + 1))
    return picks


def _slot_payload(slot: int, board_conn) -> dict:
    conn = _corpus()
    try:
        teams, rounds, fmt = _shape(conn)
        if not 1 <= slot <= teams:
            raise HTTPException(status_code=404,
                                detail=f"This archive has seats 1 to {teams}.")
        human = _human_picks_sql()

        # WHO GOES HERE, per turn. `QUALIFY` keeps the head of each turn's
        # distribution without a second pass over the picks.
        rows = conn.execute(f"""
            WITH mine AS (
                SELECT pk.pick_no, pk.round, pk.player_id, pk.position
                FROM draft_log_pick pk JOIN draft_log d USING (draft_id)
                WHERE {shape_filter()} AND pk.slot = ? AND {human}
            ),
            counted AS (
                SELECT pick_no, round, player_id, position, count(*) AS n
                FROM mine GROUP BY 1, 2, 3, 4
            ),
            totals AS (
                SELECT pick_no, sum(n) AS total, count(DISTINCT player_id) AS names
                FROM counted GROUP BY 1
            )
            SELECT c.pick_no, c.round, c.player_id, c.position, c.n,
                   t.total, t.names
            FROM counted c JOIN totals t USING (pick_no)
            QUALIFY row_number() OVER (
                PARTITION BY c.pick_no ORDER BY c.n DESC, c.player_id) <= ?
            ORDER BY c.pick_no, c.n DESC
        """, [slot, TOP_PER_TURN]).fetchall()

        mix = conn.execute(f"""
            SELECT pk.pick_no, pk.position, count(*) AS n
            FROM draft_log_pick pk JOIN draft_log d USING (draft_id)
            WHERE {shape_filter()} AND pk.slot = ? AND {human}
            GROUP BY 1, 2 ORDER BY 1, 3 DESC
        """, [slot]).fetchall()

        # THE PATHS PEOPLE WALK. One string per draft -- this seat's first
        # five positions in order -- then grouped. A draft where this seat
        # missed a pick to the clock is dropped by the HAVING rather than
        # counted as a shorter path: a four-position path is not a strategy,
        # it is a gap.
        paths = conn.execute(f"""
            SELECT path, count(*) AS n FROM (
                SELECT pk.draft_id,
                       string_agg(pk.position, '-' ORDER BY pk.round) AS path
                FROM draft_log_pick pk JOIN draft_log d USING (draft_id)
                WHERE {shape_filter()} AND pk.slot = ?
                  AND pk.round <= ? AND {human}
                GROUP BY 1 HAVING count(*) = ?
            ) GROUP BY 1 ORDER BY 2 DESC LIMIT 8
        """, [slot, SEQUENCE_ROUNDS, SEQUENCE_ROUNDS]).fetchall()
        walked = sum(n for _path, n in paths)
    finally:
        conn.close()

    names = _resolve(board_conn, (row[2] for row in rows))
    # What the board projects for the names at these turns. Same source as
    # `/at`, so a player quoted at pick 13 here and pick 13 there is quoted the
    # same number; missing for anybody the board cannot price -- see `_proj`.
    proj = _proj(board_conn, (row[2] for row in rows))
    positions = {}
    for pick_no, position, n in mix:
        positions.setdefault(int(pick_no), []).append((str(position or "?"), int(n)))

    turns = {}
    for pick_no, rnd, player_id, position, n, total, distinct in rows:
        turn = turns.setdefault(int(pick_no), {
            "pick_no": int(pick_no),
            "round": int(rnd),
            # How many times this seat's turn was taken by a person at all.
            # Named rather than assumed constant: late rounds are thick with
            # autodrafts, so the sample thins as the draft goes on and a
            # reader is entitled to know it.
            "observed": int(total),
            "distinct": int(distinct),
            "players": [],
            "positions": [],
        })
        turn["players"].append({
            "player_id": str(player_id),
            "name": names.get(str(player_id), {}).get("name"),
            "headshot": names.get(str(player_id), {}).get("headshot"),
            "position": str(position or "?"),
            "count": int(n),
            "share": round(float(n) / float(total), 4) if total else 0.0,
            # The season total, not per game: the page divides it by the same
            # SEASON_GAMES every other projection in this app is printed over.
            "proj_points": (proj.get(str(player_id)) or {}).get("points"),
            "proj_change": (proj.get(str(player_id)) or {}).get("change"),
        })
    for pick_no, turn in turns.items():
        total = float(turn["observed"]) or 1.0
        turn["positions"] = [{"position": pos, "share": round(n / total, 4)}
                             for pos, n in positions.get(pick_no, [])]
        # The one number that says how decided a turn is: the share of the
        # single most common pick. 0.75 is "this turn is spoken for"; 0.06 is
        # "nobody agrees and you are free".
        turn["top_share"] = turn["players"][0]["share"] if turn["players"] else 0.0

    return {
        "slot": slot,
        "teams": teams,
        "rounds": rounds,
        "format": fmt,
        "turns": [turns[p] for p in sorted(turns)],
        "sequences": [{
            "path": str(path).split("-"),
            "count": int(n),
            "share": round(float(n) / walked, 4) if walked else 0.0,
        } for path, n in paths],
        "sequence_rounds": SEQUENCE_ROUNDS,
        "sequences_observed": int(walked),
    }


def _players_payload(slot: int, board_conn) -> dict:
    conn = _corpus()
    try:
        teams, rounds, fmt = _shape(conn)
        if not 1 <= slot <= teams:
            raise HTTPException(status_code=404,
                                detail=f"This archive has seats 1 to {teams}.")
        # EVERY pick counts here, autodrafts included: "was he still on the
        # board" does not care who took him. `draft_log_pool` is what makes
        # the denominator honest -- a player who was not in a draft's pool at
        # all must not count as having lasted through it.
        rows = conn.execute(f"""
            WITH d AS (
                SELECT draft_id, my_slot FROM draft_log
                WHERE {shape_filter('')}
            ),
            pool AS (
                SELECT pl.draft_id, pl.player_id, pl.position, pl.team,
                       pl.espn_rank, pl.adp_rank
                FROM draft_log_pool pl JOIN d USING (draft_id)
            ),
            taken AS (
                SELECT pk.draft_id, pk.player_id, min(pk.pick_no) AS pick_no
                FROM draft_log_pick pk JOIN d USING (draft_id)
                GROUP BY 1, 2
            )
            SELECT p.player_id, any_value(p.position), any_value(p.team),
                   count(*) AS pooled,
                   list(t.pick_no) FILTER (WHERE t.pick_no IS NOT NULL) AS picks,
                   avg(p.espn_rank), avg(p.adp_rank)
            FROM pool p LEFT JOIN taken t USING (draft_id, player_id)
            GROUP BY 1
        """).fetchall()
    finally:
        conn.close()

    names = _resolve(board_conn, (row[0] for row in rows))
    turns = [p for p in _turn_picks(teams, slot) if p <= teams * rounds]
    total_picks = teams * rounds
    edges = np.linspace(0, total_picks, HIST_BINS + 1)

    players = []
    for player_id, position, team, pooled, picks, espn_rank, adp in rows:
        taken = np.asarray(picks or [], dtype=float)
        pooled = int(pooled)
        if pooled == 0:
            continue
        hist = (np.histogram(taken, bins=edges)[0] / pooled).round(4).tolist() \
            if taken.size else [0.0] * HIST_BINS
        # Still there at pick P: every draft where he was never taken, plus
        # every draft where he went later than P. Counted over the drafts he
        # was POOLED in, so a player added to the board halfway through the
        # week is not credited with surviving the drafts he was absent from.
        survive = [round(float(pooled - int((taken <= p).sum())) / pooled, 4)
                   for p in turns]
        players.append({
            "player_id": str(player_id),
            "name": names.get(str(player_id), {}).get("name"),
            "headshot": names.get(str(player_id), {}).get("headshot"),
            "position": str(position or "?"),
            "team": None if team is None else str(team),
            "drafts": pooled,
            "taken": int(taken.size),
            "taken_share": round(float(taken.size) / pooled, 4),
            # The shape, in five numbers. p10/p90 rather than the extremes
            # alone, because one drafter reaching four rounds early is a
            # story about him and not about the market.
            "earliest": int(taken.min()) if taken.size else None,
            "p10": int(round(float(np.quantile(taken, 0.10)))) if taken.size else None,
            "median": int(round(float(np.median(taken)))) if taken.size else None,
            "p90": int(round(float(np.quantile(taken, 0.90)))) if taken.size else None,
            "latest": int(taken.max()) if taken.size else None,
            # The published numbers, for the gap. This is the whole argument
            # of the page: where the market says he goes versus where it
            # actually takes him.
            "espn_rank": None if espn_rank is None else round(float(espn_rank), 1),
            "adp": None if adp is None else round(float(adp), 1),
            "hist": hist,
            "survive": survive,
        })
    players.sort(key=lambda row: (row["median"] is None, row["median"] or 0))
    return {"slot": slot, "teams": teams, "rounds": rounds, "format": fmt,
            "turns": turns, "bins": HIST_BINS, "picks_total": total_picks,
            "players": players}


# How many picks one `at-picks` call may ask about. A seat owns sixteen turns;
# thirty-two is two seats' worth, which is the most a page could want to
# compare at once, and it keeps one request from becoming a scan of the corpus.
MAX_AT_PICKS = 32

# Names per pick. Three, because a pick is rarely one player's: the waiting
# room draws all of them side by side, and the second and third names are the
# whole of what "this turn is open" looks like -- one name at 33% and two at
# 21% and 12% is a different seat from one name at 78%.
TOP_AT_PICK = 3


def _at_picks_payload(picks: tuple, board_conn) -> dict:
    """Who usually goes at each of these overall pick numbers.

    THE QUESTION A SEAT ASKS. In a snake, choosing a seat is choosing a list of
    pick numbers, and what makes one list better than another is who is on the
    board when they come round. This answers that from the corpus rather than
    from an ADP table: at pick 23, across every recorded mock, these are the
    players people actually took.

    COUNTED BY PICK NUMBER, ACROSS SHAPES ON PURPOSE. `_shape` exists because
    ROUND is not comparable between league sizes -- pick 14 is round 2 in an
    8-team draft and round 1 in a 16-team one. A pick NUMBER is comparable in
    the one way this question needs: at pick 23, twenty-two players are off the
    board either way. The room being drawn may be 12 or 14 teams; the archive
    is 8-team mocks, and the payload says so (`teams`/`rounds`/`drafts`) so the
    page can attribute it rather than implying it is that room's own history.

    A pick nobody ever reached (a 14-team room's pick 200, an archive that
    stops at 128) comes back present and empty -- `observed: 0`, no names. The
    page says "no record"; it does not get to invent one.

    HUMAN PICKS ONLY, like every other behaviour question here: an autodrafted
    seat is ESPN's ADP list read aloud, and the farm's own seat is this tool
    marking its own homework.
    """
    conn = _corpus()
    try:
        human = _human_picks_sql()
        rows = conn.execute(f"""
            WITH mine AS (
                SELECT pk.pick_no, pk.player_id, pk.position
                FROM draft_log_pick pk JOIN draft_log d USING (draft_id)
                WHERE pk.pick_no IN ({','.join('?' * len(picks))}) AND {human}
            ),
            counted AS (
                SELECT pick_no, player_id, position, count(*) AS n
                FROM mine GROUP BY 1, 2, 3
            ),
            totals AS (
                SELECT pick_no, sum(n) AS total FROM counted GROUP BY 1
            )
            SELECT c.pick_no, c.player_id, c.position, c.n, t.total
            FROM counted c JOIN totals t USING (pick_no)
            QUALIFY row_number() OVER (
                PARTITION BY c.pick_no ORDER BY c.n DESC, c.player_id) <= ?
            ORDER BY c.pick_no, c.n DESC
        """, [*picks, TOP_AT_PICK]).fetchall()
        teams, rounds, fmt = _shape(conn)
        drafts = int(conn.execute("SELECT count(*) FROM draft_log").fetchone()[0])
    finally:
        conn.close()

    names = _resolve(board_conn, (row[1] for row in rows))
    proj = _proj(board_conn, (row[1] for row in rows))
    out = {pick: {"pick_no": pick, "observed": 0, "top": []} for pick in picks}
    for pick_no, player_id, position, n, total in rows:
        entry = out[int(pick_no)]
        entry["observed"] = int(total)
        entry["top"].append({
            "player_id": str(player_id),
            "name": names.get(str(player_id), {}).get("name"),
            "headshot": names.get(str(player_id), {}).get("headshot"),
            "position": str(position or "?"),
            "count": int(n),
            "share": round(float(n) / float(total), 4) if total else 0.0,
            # What he is projected for, so the two or three names at one pick
            # can be weighed against each other rather than only counted. Null
            # where the board cannot answer -- see `_proj`.
            "proj_points": (proj.get(str(player_id)) or {}).get("points"),
            "proj_change": (proj.get(str(player_id)) or {}).get("change"),
        })
    return {"picks": [out[p] for p in picks],
            "drafts": drafts, "teams": teams, "rounds": rounds,
            "format": fmt}


def _parse_picks(raw: str) -> tuple:
    """`6,23,34` -> (6, 23, 34). Refuses anything else rather than guessing.

    Deduped and ordered, because the cache key is this tuple and `6,6,23` and
    `23,6` are the same question asked twice.
    """
    try:
        wanted = sorted({int(part) for part in raw.split(",") if part.strip()})
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="`picks` is a comma-separated list of pick numbers.") from None
    if not wanted or len(wanted) > MAX_AT_PICKS:
        raise HTTPException(
            status_code=400,
            detail=f"Ask about 1 to {MAX_AT_PICKS} picks at a time.")
    if wanted[0] < 1:
        raise HTTPException(status_code=400, detail="Pick numbers start at 1.")
    return tuple(wanted)


def _export_zip(names: dict) -> bytes:
    """The whole corpus as a zip of JSON files -- see the export route.

    Built in memory: the corpus this serves is ~55k pick rows, a few MB of
    JSON, and a temp file would just be a second copy with a lifetime to
    manage. `names` is the board's player_id -> name map, included so the
    download is readable without this deployment's database.
    """
    conn = _corpus()
    try:
        teams, rounds, fmt = _shape(conn)
        draft_rows = conn.execute(
            "SELECT d.draft_id, d.source, d.league_id, d.season,"
            " d.recorded_at, d.teams, d.rounds, d.my_slot, d.human_seats,"
            " count(pk.pick_no) AS picks_made"
            " FROM draft_log d LEFT JOIN draft_log_pick pk USING (draft_id)"
            f" WHERE {shape_filter()}"
            " GROUP BY ALL ORDER BY d.recorded_at DESC").fetchall()
        pick_rows = conn.execute(
            "SELECT pk.draft_id, pk.pick_no, pk.round, pk.slot, pk.player_id,"
            " pk.position, pk.adp_rank, pk.proj_points, pk.autodrafted,"
            " pk.had_owner, pk.seconds_to_pick"
            " FROM draft_log_pick pk JOIN draft_log d USING (draft_id)"
            f" WHERE {shape_filter()}"
            " ORDER BY pk.draft_id, pk.pick_no").fetchall()
    finally:
        conn.close()

    drafts = [{
        "draft_id": r[0], "source": r[1], "league_id": r[2],
        "season": r[3], "recorded_at": None if r[4] is None else str(r[4]),
        "teams": r[5], "rounds": r[6], "my_slot": r[7],
        "human_seats": r[8], "picks_made": int(r[9]),
    } for r in draft_rows]
    picks = [{
        "draft_id": r[0], "pick_no": r[1], "round": r[2], "slot": r[3],
        "player_id": r[4], "position": r[5], "adp_rank": r[6],
        "proj_points": r[7], "autodrafted": r[8], "had_owner": r[9],
        "seconds_to_pick": r[10],
    } for r in pick_rows]
    picked_ids = {p["player_id"] for p in picks}
    players = {pid: {"name": row.get("name")}
               for pid, row in names.items() if pid in picked_ids}

    readme = (
        "The draft archive, exported.\n\n"
        "overview.json  what the archive page reports: totals and the shape.\n"
        "drafts.json    one row per recorded draft, newest first.\n"
        "picks.json     every pick in those drafts, in draft order.\n"
        "players.json   player_id -> display name, for the ids picks.json uses.\n\n"
        "autodrafted marks picks made by ESPN\'s engine rather than a person.\n"
        "my_slot is the seat this project\'s own recorder held; its picks are\n"
        "neither a person nor ESPN\'s engine and are excluded from the page\'s\n"
        "behaviour numbers.\n")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme)
        zf.writestr("overview.json", json.dumps(_overview_payload(), indent=1))
        zf.writestr("drafts.json", json.dumps(drafts, indent=1))
        zf.writestr("picks.json", json.dumps(picks, indent=1))
        zf.writestr("players.json", json.dumps(players, indent=1))
    return buf.getvalue()


def register_market_routes(app, conn=None):
    """`GET /api/market/...` -- the draft archive, open to anybody.

    No session is read and no cookie is touched: every answer is the same
    aggregate for every reader, which is also why one cache serves all of
    them.
    """

    # Five minutes on all of them (SHARED_CACHE_SECONDS). These are
    # aggregates over hundreds of recorded drafts and they move only when the
    # farm finishes another one, which is a handful of times an hour at most
    # -- so a reader is never shown anything meaningfully old, and the
    # process-local cache above stops being asked at all for the pages people
    # actually open.

    @app.get("/api/market/overview")
    def market_overview(response: Response):
        http_cache.public(response, SHARED_CACHE_SECONDS)
        return _cached("overview", _overview_payload)

    @app.get("/api/market/export.zip")
    def market_export():
        """The corpus as one download. Cached like every other answer here:
        the zip is the same bytes for every reader, and rebuilding a few MB
        of JSON per click would be the cheapest denial of service on the
        page."""
        body = _cached("export", lambda: _export_zip(_names(conn)))
        res = Response(
            content=body,
            media_type="application/zip",
            headers={"Content-Disposition":
                     'attachment; filename="draft-archive.zip"'})
        http_cache.public(res, SHARED_CACHE_SECONDS)
        return res

    @app.get("/api/market/slot/{slot}")
    def market_slot(slot: int, response: Response):
        http_cache.public(response, SHARED_CACHE_SECONDS)
        return _cached(
            ("slot", slot), lambda: _slot_payload(slot, conn),
            lambda v: _board_answered(p for t in v["turns"] for p in t["players"]))

    @app.get("/api/market/at-picks")
    def market_at_picks(response: Response, picks: str = ""):
        """Who usually goes at each of these overall picks -- the waiting
        room's question, asked of every recorded mock at once."""
        http_cache.public(response, SHARED_CACHE_SECONDS)
        wanted = _parse_picks(picks)
        return _cached(
            ("at-picks", wanted), lambda: _at_picks_payload(wanted, conn),
            lambda v: _board_answered(p for r in v["picks"] for p in r["top"]))

    @app.get("/api/market/players")
    def market_players(response: Response, slot: int = 1):
        http_cache.public(response, SHARED_CACHE_SECONDS)
        return _cached(("players", slot), lambda: _players_payload(slot, conn))
