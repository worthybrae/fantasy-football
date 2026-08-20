"""Last completed season's points, game by game, for every player.

WHAT THIS IS FOR: the draft room's available table draws an inline bar chart
per row -- one bar per week of the season just finished, coloured on fixed
thresholds (web/src/components/draft/AvailableList.tsx). That is a per-player
array, and the only place it can come from is a pass over `weekly`.

WHY IT IS CACHED HERE RATHER THAN COMPUTED IN `build_board`, which is the
obvious home for anything /api/players serves: the board cache
(scoring/board_cache.py) keys on the `drafted` table, so its entry is thrown
away on EVERY PICK -- twelve to fifteen times an hour on draft night, more in
a fast mock. This frame depends on nothing a pick can change: the same weekly
rows, priced by the same league rules, produce the same arrays until a
pipeline refresh rewrites `weekly`. Keyed the way scoring/profile_cache.py is
keyed -- database + `meta` + the league's scoring rules -- it is built once
per refresh and survives every pick, which is what the brief asked for.

MEASURED, on a copy of data/nfl.duckdb (174,376 weekly rows, 18,540 of them
in 2025): 0.069s for the filtered read and 0.032s for the grouped pass, so
~0.10s once per refresh against a 1.7-1.8s board build that already happens
far more often. The whole payload is 2,019 players x 18 weeks and adds ~24 KB
to /api/players' 193 KB body (see the endpoint for the before/after timing).

WHY NOT REUSE `profile_cache.prior_weekly`, which is the identical slice of
`weekly`, already cached, already keyed on the same three things: reaching it
means calling `cached_profile_frames`, and a miss there builds the WHOLE 46 MB
profile frame set -- the snap-share aggregate, the comparable pool, a 0.76s
line rating -- for a board request that needs none of it. A warm profile cache
would have made it free and a cold one would have put 1.89s on the first
/api/players of every process. Reading 18,540 rows again for 0.069s is the
cheaper of the two, and it keeps this module free of the profile's dependencies
(importing it here would also close the cycle board -> profile_cache -> board).

WHICH SEASON: `max(season)` in `weekly`, which IS the last COMPLETE season
rather than an in-progress one -- not by luck and not by a date check.
`pipeline.refresh` fetches `weekly` for `scoring.config.HISTORY_SEASONS`, a
hand-maintained range that ends at the last finished season (2016-2025 with
CURRENT_SEASON at 2026), so a partial season cannot be in the table to be
picked. It is also the same rule `board._latest_season_stats` uses for the
`stats` column the table already shows beside this chart; the two MUST name
the same season, because the header says which year the bars are and the row
next to it would otherwise be a different one.
"""
import threading
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd

# Imported, not re-derived -- the same rule scoring/profile_cache.py follows:
# two "has the data changed" signals that disagree by one table would serve a
# chart built before a refresh next to a board built after it.
from scoring.board_cache import _db_key, _meta_key
from scoring.ppr import compute_ppr_points, normalize_rules, prices_kicking

# One entry per (database, scoring rules). Small -- ~1.2 MB for 2,019 players
# x 18 weeks of floats -- so unlike the profile cache's 46 MB entries the
# bound here is not about memory; it is about not keeping a pre-refresh frame
# alive forever. 4 matches scoring/profile_cache.py's for the same reasons: a
# handful of test databases in one process, or one database across a refresh.
_MAX_ENTRIES = 4

_lock = threading.Lock()
_cache: "OrderedDict[tuple, SeasonGamePoints]" = OrderedDict()


@dataclass(frozen=True)
class SeasonGamePoints:
    """Every player's week-by-week points for one season.

    `by_player` maps player_id -> a TUPLE of length `weeks`, indexed by week
    (index 0 is week 1), holding the player's points that week or None for a
    week he has no row in -- a bye, an injury, a season that started late.
    The dense shape is deliberate: the chart draws every player against the
    same 18 slots, so a back who played eight games reads as half a season
    rather than as a full one with short bars, and a missed week reads as a
    gap rather than as a zero.

    Tuples, not lists, and handed back WITHOUT copying: this object is served
    straight into a JSON response, and a mutable value shared out of a
    process-wide cache is exactly the bug `profile_cache.ProfileFrames.copy`
    exists for. Making the values immutable removes the hazard instead of
    paying to copy 2,019 sequences per request.

    `season` is None (and `by_player` empty) for a database with no `weekly`
    table at all, which is what a bare test fixture and a machine that has
    never refreshed both look like.
    """
    season: int | None
    weeks: int
    by_player: dict


_EMPTY = SeasonGamePoints(season=None, weeks=0, by_player={})


def _build(conn, rules: dict | None) -> SeasonGamePoints:
    # Filter pushed into SQL, the same way scoring/profile.py reads its
    # per-player slice: the whole `weekly` table is 174k rows and 0.247s to
    # read, one season of it is 18.5k rows and 0.069s.
    exists = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'weekly'"
    ).fetchone()[0]
    if not exists:
        return _EMPTY
    w = conn.execute(
        "SELECT * FROM weekly WHERE season = (SELECT max(season) FROM weekly)").df()
    if w.empty or "week" not in w.columns or "player_id" not in w.columns:
        return _EMPTY

    if not prices_kicking(rules) and "position" in w.columns:
        # A league that prices no kicking scores every kicking stat at zero,
        # so a kicker's real season comes out as a row of 0.0s -- seventeen
        # red bars claiming he never scored, when what actually happened is
        # that this league does not count what he did. scoring/profile.py
        # already refuses to show that column of zeros, on exactly this test
        # (`prices_kicking(rules)`, not `position == "K"`, so a league that
        # DOES price kicking gets real bars). Dropping the rows here gives
        # the same players the same empty state a rookie gets -- "no games",
        # which is honest -- rather than a false one.
        w = w[w["position"] != "K"]
        if w.empty:
            return _EMPTY

    pts = compute_ppr_points(w, rules)
    g = pd.DataFrame({"player_id": w["player_id"], "week": w["week"], "pts": pts})
    # groupby drops rows whose key is null, which is the right answer for the
    # 18 rows of 2025 that carry no player_id: points that cannot be
    # attributed to anybody belong on nobody's chart. It also collapses a
    # duplicated (player, week) row by summing, which this database does not
    # currently contain (verified: 0 such groups in 2025) but which would
    # otherwise make `pivot` raise rather than draw a slightly wrong bar.
    g = g.groupby(["player_id", "week"], as_index=False)["pts"].sum()
    if g.empty:
        return _EMPTY

    weeks = int(g["week"].max())
    wide = (g.pivot(index="player_id", columns="week", values="pts")
            .reindex(columns=range(1, weeks + 1))
            .round(1))
    by_player = {
        str(pid): tuple(None if pd.isna(v) else float(v) for v in row)
        for pid, row in zip(wide.index, wide.to_numpy())
    }
    return SeasonGamePoints(season=int(w["season"].max()), weeks=weeks,
                            by_player=by_player)


def cached_game_points(conn, rules: dict | None = None) -> SeasonGamePoints:
    """Last completed season's per-game points, built once per refresh.

    `rules` is the league's `settings.scoring`; None means full PPR.
    `normalize_rules` collapses a rules dict that IS full PPR onto None, so a
    PPR league shares one entry with every rules-less caller -- the same
    normalization `profile_cache.cached_profile_frames` does, and for the same
    reason: two spellings of one league must not be two cache entries.
    """
    rules = normalize_rules(rules)
    rules_key = None if rules is None else tuple(sorted(rules.items()))
    key = (_db_key(conn), _meta_key(conn), rules_key)

    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit

    # Built outside the lock, exactly as board_cache and profile_cache build
    # theirs: two racing requests on a cold cache both do the ~0.1s of work
    # and the later one wins the slot, which is cheaper than serializing every
    # /api/players behind one mutex.
    frame = _build(conn, rules)

    with _lock:
        _cache[key] = frame
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)

    return frame


def clear() -> None:
    """Drop every cached frame. Test hook, and the escape valve for the gap
    this shares with scoring/board_cache.py and scoring/profile_cache.py: a
    `write_table` onto `weekly` that does not also stamp `meta` leaves this
    cache serving the old season."""
    with _lock:
        _cache.clear()
