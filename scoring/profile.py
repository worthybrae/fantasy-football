"""Assemble the player-profile payload: history, outlook, similar seasons."""
import math
import numpy as np
import pandas as pd
from pipeline.db import read_table
# IMPORTED from the writer, never re-spelled here. The attribution marker is
# the whole contract between pipeline/news.py and this payload: an id-tagged
# item and a name-matched one are different kinds of fact and the card
# presents them differently. A literal "espn_athlete_id" typed a second time
# in this file would silently stop matching the day somebody renames the
# constant, and the failure mode is not an exception -- it is every exact
# item quietly losing its tiebreak and sorting as if it were inferred.
# (ATTR_NAME is deliberately not imported: nothing here needs to NAME the
# inferred case. Every row's marker is passed through verbatim, so a third
# value pipeline/news.py might add one day reaches the client untouched
# rather than being collapsed into one of the two this file knows.)
from pipeline.news import ATTR_EXACT
from scoring import factors, league
from scoring.board import _norm_name, _adapt_depth_charts
from scoring.board_cache import cached_build_board
from scoring.config import RECENCY_WEIGHTS
from scoring.profile_cache import (RANK_MIN_GAMES, _table_columns,
                                   cached_profile_frames, season_rank_frame,
                                   snap_share_by_season)
from scoring.ppr import compute_ppr_points, normalize_rules, prices_kicking
from scoring.similarity import (_age_in_season, player_season_features,
                                find_twins, value_neighbors)

_KDST_POSITIONS = {"K", "DST"}

# The comparable cohort's band, and the one number on this page that is a
# CALIBRATION rather than a derivation -- named as such, the way
# scoring/config.py names STREAMED_REPLACEMENT_RANK and NEED_WEIGHTS, so
# nobody later mistakes it for something that fell out of the data.
#
# There is no ground truth for "how close is close enough to be a
# comparable". What there is, measured on data/nfl.duckdb for Jahmyr Gibbs'
# 21.6-ppg third season, is a straight trade of sample size against
# tightness:
#
#     +-1 ppg / same year     n = 2     median -- (too few to have a shape)
#     +-3 ppg / +-1 year      n = 19    median -2.8 ppg, 13 of 19 declined
#     +-4 ppg / +-2 years     n = 40    median -2.6 ppg
#
# +-3/+-1 is the tightest band that still produces a distribution rather
# than an anecdote. n=2 is not a cohort; +-4/+-2 doubles the sample for a
# median that barely moves, by admitting backs two years further along the
# age curve, which is the one axis the cohort exists to hold still.
#
# The band is SERVED in the payload (`cohort.ppg_band` / `cohort.exp_band`)
# rather than only applied, because a card that says "19 comparable seasons"
# without saying what made them comparable is asking to be believed rather
# than read.
COMP_PPG_BAND = 3.0
COMP_EXP_BAND = 1

# How many headlines travel with one profile. The second calibration on this
# page, and like COMP_PPG_BAND above it is a judgement stated rather than a
# number that fell out of the data -- but the trade it balances was measured,
# on the 2,346 rows pipeline/news.py wrote for the 249-player board:
#
#   * ONE ITEM IS EXPENSIVE FOR WHAT IT SAYS. Median 494 bytes, of which the
#     Google News redirect URL alone is 256 -- Google stopped putting a
#     decodable target in that token, so the redirect IS the link and it
#     cannot be shortened. A headline is ~60 characters of actual content
#     inside a 494-byte row.
#   * THE PAYLOAD IT JOINS IS NOT BIG. A skill player's profile is 33 KB
#     today (23 KB of it game_log); a kicker's is 3.3 KB and a defense's
#     3.0 KB. Eight items add 4.1 KB median / 5.0 KB worst case -- +12% on a
#     running back, but +150% on a kicker, which is the case that decided
#     this. The whole thing is fetched on a click while a draft clock runs.
#   * THE STORED FEED HOLDS AT MOST 14 (pipeline's GOOGLE_ITEMS_PER_PLAYER
#     is 10 name-matched, plus 1-4 id-tagged; median 10). So 8 is a real
#     trim rather than a decorative cap -- which is the point of having one
#     here at all. A serving cap set AT the storage cap does nothing today
#     and silently grows the payload the day somebody raises the pipeline's.
#   * IT ALMOST NEVER COSTS AN ID-TAGGED ITEM. Ordering by recency and
#     cutting at 8 drops an exact item for 4 of the 87 players who have one;
#     at 6 it is 5 players, at 10 it is 3. Below ~6 the losses start and the
#     bytes saved (3.0 KB vs 4.1 KB) do not pay for them.
#
# What this is NOT sized for is completeness. The feed is a "what has been
# said about him lately" panel next to a pick clock, not an archive: the
# median stored item is already 5 days old and the oldest is 9 months.
NEWS_ITEMS = 8

# Distinguishable from an explicit `snap_share=None`, which means "there is
# no snap data" -- a real, different answer from "work it out yourself".
_UNSET = object()


def _scrub(v):
    if isinstance(v, dict):
        return {k: _scrub(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_scrub(x) for x in v]
    if v is None or (pd.api.types.is_scalar(v) and pd.isna(v)):
        # Catches pd.NA/NaT (e.g. an empty adp table leaves the whole `adp`
        # column as pd.NA) in addition to float NaN, which the isinstance
        # checks below would otherwise miss.
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _num(row, col):
    v = row.get(col, 0)
    return 0 if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _round_or_none(v, ndigits):
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    return round(float(v), ndigits)


def _int_or_none(v):
    """`int(v)` unless there is nothing to convert.

    Ranks arrive as floats out of pandas' `rank`/`transform`, and a season
    that did not qualify has no rank at all. `int(nan)` is a ValueError and
    `_scrub` never sees it, so the None has to happen here.
    """
    if v is None or pd.isna(v):
        return None
    return int(v)


def _stat_line(row, position):
    if position == "QB":
        comp = int(_num(row, "completions"))
        att = int(_num(row, "attempts"))
        pyds = int(_num(row, "passing_yards"))
        ptd = int(_num(row, "passing_tds"))
        ints = int(_num(row, "passing_interceptions"))
        line = f"{comp}/{att}, {pyds} yds, {ptd} TD, {ints} INT"
        car = _num(row, "carries")
        if car > 0:
            line += f" · {int(car)} car, {int(_num(row, 'rushing_yards'))} yds"
        return line
    if position == "RB":
        car = int(_num(row, "carries"))
        ryds = int(_num(row, "rushing_yards"))
        rtd = int(_num(row, "rushing_tds"))
        line = f"{car} car, {ryds} yds, {rtd} TD"
        tgt = _num(row, "targets")
        if tgt > 0:
            line += (f" · {int(_num(row, 'receptions'))} rec, "
                      f"{int(_num(row, 'receiving_yards'))} yds")
        return line
    if position == "K":
        # Before kicking was scorable a kicker fell through to the WR/TE
        # branch below and rendered "0 tgt, 0 rec, 0 yds, 0 TD" -- true of
        # every kicker who ever played, and information about none of them.
        # The bands are the ones the league actually prices
        # (scoring.league.ESPN_STAT_COLUMNS), so the line reads as the points
        # beside it were earned. `fg_long` is the one number a kicker is
        # actually discussed in terms of, and it is a per-game maximum rather
        # than a count, so it is shown only when there was a kick to have a
        # long.
        fgm = int(_num(row, "fg_made"))
        fga = int(_num(row, "fg_att"))
        line = f"{fgm}/{fga} FG"
        long = int(_num(row, "fg_long"))
        if long > 0:
            line += f", long {long}"
        line += f" · {int(_num(row, 'pat_made'))}/{int(_num(row, 'pat_att'))} XP"
        return line
    # WR/TE (and fallback for other positions)
    tgt = int(_num(row, "targets"))
    rec = int(_num(row, "receptions"))
    yds = int(_num(row, "receiving_yards"))
    rtd = int(_num(row, "receiving_tds"))
    line = f"{tgt} tgt, {rec} rec, {yds} yds, {rtd} TD"
    car = _num(row, "carries")
    if car > 0:
        line += f" · {int(car)} car, {int(_num(row, 'rushing_yards'))} yds"
    return line


# Payload key -> weekly-table column. Every game_log row carries all of these
# keys (zero-filled when the column is absent) so the frontend's
# position-aware column configs can index into a uniform shape.
_GAME_STAT_COLS = {
    "completions": "completions", "attempts": "attempts",
    "pass_yards": "passing_yards", "pass_tds": "passing_tds",
    "interceptions": "passing_interceptions",
    "carries": "carries", "rush_yards": "rushing_yards",
    "rush_tds": "rushing_tds", "targets": "targets",
    "receptions": "receptions", "rec_yards": "receiving_yards",
    "rec_tds": "receiving_tds",
}

# The kicking half of the same contract, kept separate so the twelve keys
# above stay exactly the twelve keys they were. Additive on purpose: an
# existing client indexes the keys it knows by name and ignores the rest, so
# a QB's row growing five zero-valued kicking keys changes nothing it renders,
# while a kicker's row finally carries the numbers behind its points.
#
# NOTHING ON THE FRONTEND READS THESE FIVE TODAY, and that is not an oversight
# waiting on a config file. The per-position stat-column tables that would have
# consumed them (`web/src/statColumns.ts`, feeding `GameLog.tsx` and
# `SeasonTable.tsx`) were never wired into the rebuilt player card and have
# been deleted. What the card actually renders per game is `stat_line`, which
# `_stat_line` already builds with a real K branch, so a kicker's week reads
# "2/2 FG, long 56 - 2/2 XP" rather than a row of receiving zeros. These keys
# stay because they are the structured form of that same line and cost one
# groupby -- a client that wants columns can have them without a second pass
# over the backend. They are not declared in `web/src/api.ts`'s `GameStats`.
_KICK_STAT_COLS = {
    "fg_made": "fg_made", "fg_att": "fg_att", "fg_long": "fg_long",
    "pat_made": "pat_made", "pat_att": "pat_att",
}
_GAME_STAT_COLS = {**_GAME_STAT_COLS, **_KICK_STAT_COLS}


def _game_stats(row):
    return {k: int(_num(row, col)) for k, col in _GAME_STAT_COLS.items()}


def season_summaries(weekly: pd.DataFrame, snaps: pd.DataFrame | None, player_id: str,
                     *, season_features: pd.DataFrame | None = None,
                     snap_share=_UNSET, rules: dict | None = None,
                     season_ranks=_UNSET) -> list[dict]:
    """Per-season rows for one player.

    Only TWO things here are league-wide: `pos_finish`, which ranks this
    player against every other player-season in `season_features`, and the
    per-position rank pair (`pos_rank_ppg` / `cv_rank`) that comes out of
    `season_ranks`. Everything else reads `weekly[weekly.player_id ==
    player_id]` and `snaps` for this player's name/team/season. So a caller
    that already holds the three league-wide aggregates -- the features
    frame, the snap-share aggregate and the rank frame -- may pass them in
    and hand `weekly` nothing but this player's rows, and `snaps` nothing at
    all. That is what scoring/profile.py's `build_profile` does now:
    recomputing them per click cost 0.211s and 0.343s respectively on
    data/nfl.duckdb (see scoring/profile_cache.py). Pass none of them and
    the behaviour is what it always was, which is what every test here does.

    `season_ranks` follows `season_features`' convention, not
    `snap_share`'s: `_UNSET` means "derive it from what I gave you"
    (`profile_cache.season_rank_frame`), and it carries the same hazard
    `pos_finish` already carries -- derived from one player's rows it says
    "1st of 1". A caller narrowing `weekly` to one player MUST pass the
    league-wide frame, and `build_profile` does. `None` means "this database
    cannot produce ranks", and every rank key comes back null.

    `rules` is the league's `settings.scoring`; None is full PPR. It prices
    `ppg_std` here directly, and -- on the path that computes them -- `ppg`,
    `points` and `pos_finish` through `player_season_features`. A caller
    supplying `season_features` MUST have priced that frame under the same
    rules, or a row would carry a half-PPR volatility beside a full-PPR
    average: `build_profile` gets both from `cached_profile_frames(conn,
    rules)`, which keys on them (scoring/profile_cache.py). The same applies
    to `season_ranks`, whose coefficient of variation divides one
    rules-priced number by another.
    """
    if weekly.empty:
        return []
    rules = normalize_rules(rules)
    feats = (player_season_features(weekly, rules) if season_features is None
             else season_features)
    # Positional finish by total season points (the standard "finished RB12"
    # framing), ranked across every player in the weekly table.
    # `.assign` rather than `feats["pos_finish"] = ...`: `feats` may now be
    # a frame owned by the caller (profile_cache's cached aggregate), and
    # writing a column into it would leave `pos_finish` stuck on the cached
    # object for every later click. profile_cache hands out copies too --
    # this is the belt to that braces, and it costs 1.2ms.
    feats = feats.assign(pos_finish=feats.groupby(["season", "position"])["points"]
                         .rank(ascending=False, method="min"))
    mine = feats[feats["player_id"] == player_id].copy()
    if mine.empty:
        return []

    # team per player-season, needed for the snap-share join (not present on
    # player_season_features output)
    team_by_season = (weekly[weekly["player_id"] == player_id]
                       .groupby("season")["recent_team"].last())
    mine["team"] = mine["season"].map(team_by_season)
    mine["_norm_name"] = mine["name"].map(_norm_name)

    share = snap_share_by_season(snaps) if snap_share is _UNSET else snap_share
    if share is not None:
        mine = mine.merge(share, on=["_norm_name", "team", "season"], how="left")
        mine = mine.rename(columns={"offense_pct": "snap_share"})
    else:
        mine["snap_share"] = np.nan

    # Positional rank by points per game, and the scale-adjusted volatility
    # rank beside it -- both league-wide, both built once per database in
    # profile_cache.season_rank_frame (read that function for why the
    # volatility rank is a coefficient of variation and not a sigma).
    # A season below RANK_MIN_GAMES has no row in the frame and comes back
    # null on every one of these keys, which is the honest answer: it was
    # not ranked, so there is no rank to show.
    ranks = (season_rank_frame(weekly, feats, rules)
             if season_ranks is _UNSET else season_ranks)
    rank_cols = ["pos_rank_ppg", "pos_rank_ppg_n", "cv", "cv_rank",
                 "cv_rank_n", "cv_pos_median",
                 # The usage card's colour: where each share and rate places
                 # among the same position that season. See
                 # profile_cache.season_rank_frame for why snap share is not
                 # among them.
                 "snap_share_pctl", "target_share_pctl", "carries_pg_pctl",
                 "targets_pg_pctl",
                 "receptions_pg_pctl", "yards_pg_pctl"]
    if ranks is not None and not ranks.empty:
        mine = mine.merge(ranks[["player_id", "season"] + rank_cols],
                          on=["player_id", "season"], how="left")
    else:
        for c in rank_cols:
            mine[c] = np.nan

    # Passing aggregates aren't part of player_season_features (that frame
    # feeds twin matching in similarity.py and must not change) -- aggregate
    # them here from the raw weekly rows instead.
    pass_cols = {k: _GAME_STAT_COLS[k] for k in
                 ("completions", "attempts", "pass_yards", "pass_tds", "interceptions")}
    wk_mine = weekly[weekly["player_id"] == player_id].copy()
    for out, col in pass_cols.items():
        wk_mine[out] = (pd.to_numeric(wk_mine[col], errors="coerce").fillna(0)
                        if col in wk_mine.columns else 0.0)
    passing = wk_mine.groupby("season", as_index=False)[list(pass_cols)].sum()
    mine = mine.merge(passing, on="season", how="left")
    for out in pass_cols:
        mine[out] = mine[out].fillna(0)

    # Kicking, aggregated here for the same reason as passing: it is not part
    # of player_season_features, which feeds twin matching and must not
    # change shape. Every season row carries these keys whatever the
    # position, matching how the passing keys already behave -- a WR's row
    # has carried `pass_yards: 0` since this function was written. `fg_long`
    # is a season MAXIMUM, not a sum; summing per-game longs would produce a
    # number with no meaning at all.
    for out, col in _KICK_STAT_COLS.items():
        wk_mine[out] = (pd.to_numeric(wk_mine[col], errors="coerce").fillna(0)
                        if col in wk_mine.columns else 0.0)
    kick_sums = [k for k in _KICK_STAT_COLS if k != "fg_long"]
    kicking = wk_mine.groupby("season", as_index=False)[kick_sums].sum()
    kicking["fg_long"] = (wk_mine.groupby("season")["fg_long"].max()
                          .reindex(kicking["season"]).to_numpy())
    mine = mine.merge(kicking, on="season", how="left")
    for out in _KICK_STAT_COLS:
        mine[out] = mine[out].fillna(0)

    # Week-to-week volatility for the consistency chart. Weekly rows are
    # games played by definition (dnp zero-fill exists only in game_log), so
    # no exclusion is needed; sample std is NaN -> None for 1-game seasons.
    wk_mine["_ppr"] = compute_ppr_points(wk_mine, rules)
    mine["ppg_std"] = mine["season"].map(wk_mine.groupby("season")["_ppr"].std())

    mine = mine.sort_values("season", ascending=False)
    rows = []
    for _, r in mine.iterrows():
        rows.append({
            "season": int(r["season"]),
            "games": int(r["games"]),
            "ppg": _round_or_none(r["ppg"], 1),
            "ppg_std": _round_or_none(r["ppg_std"], 2),
            "pos_finish": int(r["pos_finish"]),
            "completions": int(r["completions"]),
            "attempts": int(r["attempts"]),
            "pass_yards": int(r["pass_yards"]),
            "pass_tds": int(r["pass_tds"]),
            "interceptions": int(r["interceptions"]),
            "targets": int(r["targets"]),
            "target_share": _round_or_none(r["target_share"], 3),
            "carries": int(r["carries"]),
            "rec_yards": int(r["rec_yards"]),
            "rush_yards": int(r["rush_yards"]),
            "tds": int(r["tds"]),
            "receptions": int(r["receptions"]),
            "yards_per_opp": _round_or_none(r["yards_per_opp"], 1),
            "snap_share": _round_or_none(r["snap_share"], 3),
            "fg_made": int(r["fg_made"]),
            "fg_att": int(r["fg_att"]),
            "fg_long": int(r["fg_long"]),
            "pat_made": int(r["pat_made"]),
            "pat_att": int(r["pat_att"]),
            # Additive, like the kicking block above it and the passing block
            # before that: an existing client indexes the keys it knows by
            # name and ignores the rest, so nothing it already renders moves.
            # Every one of these is null for a season under RANK_MIN_GAMES --
            # a rank nobody computed is not a rank of zero.
            "pos_rank_ppg": _int_or_none(r["pos_rank_ppg"]),
            "pos_rank_ppg_n": _int_or_none(r["pos_rank_ppg_n"]),
            "cv": _round_or_none(r["cv"], 3),
            "cv_rank": _int_or_none(r["cv_rank"]),
            "cv_rank_n": _int_or_none(r["cv_rank_n"]),
            "cv_pos_median": _round_or_none(r["cv_pos_median"], 3),
            # One object rather than five loose keys: they are read together,
            # by one card, and a season row is already wide.
            "pcts": {k: _round_or_none(r[f"{k}_pctl"], 3) for k in
                     ("snap_share", "target_share", "carries_pg",
                      "targets_pg", "receptions_pg", "yards_pg")},
        })
    return rows


_SUMMARY_STATS = ["completions", "attempts", "pass_yards", "pass_tds",
                  "interceptions", "carries", "rush_yards", "targets",
                  "receptions", "rec_yards", "tds"]


def career_summary(seasons: list[dict]) -> dict:
    """Recency-weighted per-game averages over the scoring window.

    Same weights as the production factor (RECENCY_WEIGHTS), renormalized
    over the seasons the player actually has; older seasons carry weight 0
    and drop out entirely.
    """
    weighted = [(RECENCY_WEIGHTS.get(s["season"], 0.0), s) for s in seasons]
    weighted = [(w, s) for w, s in weighted if w > 0 and s["games"] > 0]
    total = sum(w for w, _ in weighted)
    if total == 0:
        return {"w_ppg": None, "w_stats": {}}
    w_ppg = sum(w * s["ppg"] for w, s in weighted) / total
    stats = {k: round(sum(w * s[k] / s["games"] for w, s in weighted) / total, 1)
             for k in _SUMMARY_STATS}
    return {"w_ppg": round(w_ppg, 1), "w_stats": stats}


# What each market is called on the card. Short, because the card has room
# for a label and a price and nothing else.
_FUTURES_LABELS = {
    "rush_yards": "Rush leader",
    "rec_yards": "Rec leader",
    "pass_yards": "Pass leader",
    "mvp": "MVP",
    "opoy": "OPOY",
    "oroy": "Off rookie",
}


# How many of a player's own position a market has to price before the card
# will rank him inside it. Three: two is a coin toss dressed as a ranking.
_POS_FIELD_MIN = 3


def _player_futures(conn, player_id: str, name: str, position: str,
                    season: int) -> list[dict]:
    """This player's season-long betting markets, with his place in each field.

    Joined on ESPN's athlete id through the sleeper crosswalk -- the same two
    steps `_espn_projection` takes, and for the same reason: every other market
    source in this project joins on a folded name and pays for it.

    A price alone says little to anyone who does not read odds daily, so each
    market carries the size of the field and where he sits in it. "+600" is
    "second shortest of sixty-one", and that is the sentence a reader needs.
    """
    futures = read_table(conn, "player_futures")
    if futures.empty or "market" not in futures.columns:
        return []
    if "season" in futures.columns:
        futures = futures[futures["season"] == season]
    if futures.empty:
        return []

    espn_id = None
    sleeper = read_table(conn, "sleeper_ids")
    if not sleeper.empty and {"gsis_id", "espn_id"}.issubset(sleeper.columns):
        hit = sleeper[sleeper["gsis_id"] == player_id]
        ids = pd.to_numeric(hit["espn_id"], errors="coerce").dropna()
        if not ids.empty:
            espn_id = int(ids.iloc[0])
    if espn_id is None:
        espn = read_table(conn, "espn_adp")
        if not espn.empty and {"espn_id", "espn_name"}.issubset(espn.columns):
            hit = espn[(espn.get("position") == position)
                       & (espn["espn_name"].map(_norm_name) == _norm_name(name))]
            ids = pd.to_numeric(hit["espn_id"], errors="coerce").dropna()
            if not ids.empty:
                espn_id = int(ids.iloc[0])
    if espn_id is None:
        return []

    futures = futures.copy()
    futures["espn_id"] = pd.to_numeric(futures["espn_id"], errors="coerce")

    # Every field's players, priced a second way: by where a draft room takes
    # them. ESPN's own ADP, joined on the same athlete id the futures carry,
    # so no name folding is involved anywhere in this comparison.
    #
    # This is the whole point of the pair of ranks the card prints. A field
    # is one set of players; ranking it by the book's price and ranking it by
    # ADP are two opinions about the SAME set, so the two numbers subtract.
    # A back the books make 4th likeliest to lead the league in rushing while
    # rooms draft him 11th of that same field is a disagreement worth seeing,
    # and neither number alone shows it.
    adp = read_table(conn, "espn_adp")
    adp_by_id = pd.Series(dtype=float)
    pos_by_id = pd.Series(dtype=object)
    if not adp.empty and {"espn_id", "espn_adp"}.issubset(adp.columns):
        marks = adp[["espn_id", "espn_adp"]].copy()
        marks["espn_id"] = pd.to_numeric(marks["espn_id"], errors="coerce")
        marks["espn_adp"] = pd.to_numeric(marks["espn_adp"], errors="coerce")
        marks = marks.dropna().drop_duplicates("espn_id")
        adp_by_id = marks.set_index("espn_id")["espn_adp"]
    if not adp.empty and {"espn_id", "position"}.issubset(adp.columns):
        who = adp[["espn_id", "position"]].copy()
        who["espn_id"] = pd.to_numeric(who["espn_id"], errors="coerce")
        who = who.dropna().drop_duplicates("espn_id")
        pos_by_id = who.set_index("espn_id")["position"]

    out = []
    for market, field in futures.groupby("market"):
        mine = field[field["espn_id"] == espn_id]
        if mine.empty:
            continue
        # Shortest price first: the favourite is 1st. `implied_pct` rather than
        # the American number, which does not sort (+600 is longer than -150
        # and reads as larger).
        order = field.sort_values("implied_pct", ascending=False).reset_index(drop=True)
        place = int(order.index[order["espn_id"] == espn_id][0]) + 1

        my_row = field.index[field["espn_id"] == espn_id][0]

        # The same field by ADP, over only the players a room actually
        # drafts. Anyone unranked is dropped rather than sent to the back:
        # sitting an undrafted player last would make everybody above him
        # look better than the draft board really says they are.
        drafted = field["espn_id"].map(adp_by_id).dropna().sort_values()
        adp_place = (int(list(drafted.index).index(my_row)) + 1
                     if my_row in drafted.index else None)

        # AND THE PAIR THE CARD ACTUALLY PRINTS: the same two rankings, but
        # over his own position only.
        #
        # Whole-field ranks compare things that are not comparable. Every one
        # of these markets is open to the league, so a back sits behind
        # thirty quarterbacks in MVP and ahead of every one of them in ADP,
        # and the card would paint that structural fact as a disagreement
        # worth acting on. Among BACKS, both numbers are answering the same
        # question and the gap between them means something.
        same = field[field["espn_id"].map(pos_by_id) == position]
        pos_place = pos_adp_place = None
        pos_field = int(len(same))
        if pos_field >= _POS_FIELD_MIN and my_row in same.index:
            by_price = same.sort_values("implied_pct", ascending=False)
            pos_place = int(list(by_price.index).index(my_row)) + 1
            pos_drafted = same["espn_id"].map(adp_by_id).dropna().sort_values()
            pos_adp_place = (int(list(pos_drafted.index).index(my_row)) + 1
                             if my_row in pos_drafted.index else None)
        out.append({
            "market": str(market),
            "label": _FUTURES_LABELS.get(str(market), str(market)),
            "american": _str_or_none_price(mine.iloc[0].get("american")),
            "implied_pct": _round_or_none(mine.iloc[0].get("implied_pct"), 1),
            # What the FAVOURITE in this market is priced at, so a card can
            # draw one player's chance against the best chance anybody has.
            # 14% is a different fact in a market whose leader sits at 18%
            # than in one whose leader sits at 40%, and the place alone --
            # "4th of 61" -- cannot separate those two.
            "top_pct": _round_or_none(order.iloc[0].get("implied_pct"), 1),
            "place": place,
            "field": int(len(field)),
            # Where the same field puts him by ADP, and how many of it a room
            # drafts at all. Null when nobody drafts HIM -- a longshot on a
            # book's board who is not on anybody else's.
            "adp_place": adp_place,
            "adp_field": int(len(drafted)),
            # The same two, among his own position. Null where the field
            # holds too few of his position to rank him against.
            "pos_place": pos_place,
            "pos_adp_place": pos_adp_place,
            "pos_field": pos_field,
        })
    # Shortest price first, so the market that likes him most leads.
    out.sort(key=lambda f: -(f["implied_pct"] or 0))
    return out


def _str_or_none_price(v) -> str | None:
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _with_futures(vegas: dict | None, futures: list[dict]) -> dict | None:
    """The team's implied totals and the player's own markets in one object.

    They are one card and one idea -- what the market thinks of the offence he
    plays in, and of him -- so a player with no lines on his team but a price
    to lead the league still gets a card.
    """
    if vegas is None and not futures:
        return None
    if vegas is None:
        return {"implied": None, "rank": None, "teams": None, "priced": 0,
                "weeks_total": None, "weeks": [], "futures": futures}
    return {**vegas, "futures": futures}


def _market_pos_ranks(board: pd.DataFrame, player_id: str) -> dict:
    """Each market number as a place among his own position.

    "Consensus 19" is a fact about 250 players; "WR7" is the one a drafter
    actually uses, because a roster is filled by position and the gap between
    WR7 and WR8 is a decision where the gap between 19 and 20 is not.

    Ranked over THIS board, which is the draftable pool -- the same population
    every other rank on the card counts in, so board and consensus places are
    read against the same denominator.

    Sources live two ways on a board row: `rank`, `market_rank`, `ffc_rank` and
    `espn_ppr_rank` are columns, and the rest sit inside `market_sources` as a
    dict per row. Both are expanded here rather than only the easy half: a card
    that showed a position for two sources and a bare number for three would
    read as though the other three were a different kind of thing.
    """
    if board is None or board.empty or "position" not in board.columns:
        return {}
    cols = {"board": "rank", "consensus": "market_rank",
            "ffc": "ffc_rank", "espn": "espn_ppr_rank"}
    frame = pd.DataFrame({"player_id": board["player_id"],
                          "position": board["position"]})
    for key, col in cols.items():
        if col in board.columns:
            frame[key] = pd.to_numeric(board[col], errors="coerce")
    if "market_sources" in board.columns:
        for key in ("fp", "mfl", "cbs"):
            frame[key] = [
                (src or {}).get(key) if isinstance(src, dict) else None
                for src in board["market_sources"]]
            frame[key] = pd.to_numeric(frame[key], errors="coerce")

    mine = frame[frame["player_id"] == player_id]
    if mine.empty:
        return {}
    out = {}
    for key in [k for k in frame.columns if k not in ("player_id", "position")]:
        value = mine.iloc[0][key]
        if pd.isna(value):
            continue
        # `method="min"`, so two players a source ranks identically share the
        # better place -- the rule every other rank in this project uses.
        place = (frame.groupby("position")[key].rank(ascending=True, method="min")
                 [mine.index[0]])
        if not pd.isna(place):
            out[key] = int(place)
    return out


def _vegas(schedules: pd.DataFrame, team: str | None,
           season: int) -> dict | None:
    """What the market prices this player's offence at, week by week.

    An implied team total is the half of a betting line that is about scoring:
    (total + spread) / 2 for the home side, (total - spread) / 2 for the away
    one -- the same arithmetic `factors.environment_factor` already runs for
    the board's `environment` percentile, so this card and that factor cannot
    disagree about whose offence the market likes.

    A TEAM total, not a player prop. Nothing here says how many of Detroit's
    27.8 points go to one back, and the card's own label has to say so.

    Only the weeks a line is actually posted for. Books price the front of a
    season and a scattering beyond it -- about seven of seventeen at the
    moment -- and the rest are absent rather than zero, because "no line yet"
    and "a low-scoring game" are opposite claims.
    """
    if schedules is None or schedules.empty or not team:
        return None
    sc = schedules
    if "season" in sc.columns:
        sc = sc[sc["season"] == season]
    if sc.empty or not {"total_line", "spread_line"}.issubset(sc.columns):
        return None
    priced = sc.dropna(subset=["total_line", "spread_line"])
    weeks = []
    for _, g in priced.iterrows():
        for side, other, sign in (("home_team", "away_team", 1),
                                  ("away_team", "home_team", -1)):
            if g.get(side) != team:
                continue
            weeks.append({
                "week": _int_or_none(g.get("week")),
                "opponent": None if pd.isna(g.get(other)) else str(g.get(other)),
                "home": side == "home_team",
                "implied": _round_or_none(
                    (g["total_line"] + sign * g["spread_line"]) / 2, 1),
            })
    if not weeks:
        return None
    weeks.sort(key=lambda w: (w["week"] is None, w["week"]))

    # Ranked over every team the market has priced at all, so the denominator
    # is the one the number came from rather than a hopeful 32.
    env = factors.environment_factor(sc)
    rank = None
    if not env.empty and (env["team"] == team).any():
        order = env.sort_values("env_raw", ascending=False).reset_index(drop=True)
        rank = int(order.index[order["team"] == team][0]) + 1
    mine = env[env["team"] == team]
    return {
        "implied": _round_or_none(mine.iloc[0]["env_raw"], 1) if not mine.empty else None,
        "rank": rank,
        "teams": int(len(env)) if not env.empty else None,
        "priced": len(weeks),
        "weeks_total": int(sc["week"].nunique()) if "week" in sc.columns else None,
        "weeks": weeks,
    }


def _espn_projected_usage(conn, player_id: str, name: str, position: str,
                          season: int) -> dict | None:
    """ESPN's projected stat line for the season being drafted, per game.

    The same source and the same two-step match as `_espn_projection` -- the
    sleeper crosswalk first, then name and position -- because a projection's
    points and the carries behind them have to describe the same player.

    Per game rather than per season, so the row reads against the seasons
    beside it: 325 carries and 19.1 a game are the same fact, but only one of
    them can be compared with what he did last year.

    NO SHARES. A share needs a projected team total and this table has no
    trustworthy one: its 439 rows cover 5 to 14 players a team, and summing
    them gives Arizona 533 targets against 88 pass attempts and Atlanta 411
    targets against none at all. Snap share it does not project at all. The
    card leaves both blank in the projected column rather than dividing by a
    denominator that is wrong by a quarter and saying nothing about it.
    """
    proj = read_table(conn, "espn_projections")
    if proj.empty or "season" not in proj.columns:
        return None
    rows = proj[proj["season"] == season]
    if rows.empty:
        return None
    hit = pd.DataFrame()
    sleeper = read_table(conn, "sleeper_ids")
    if not sleeper.empty and "espn_id" in rows.columns:
        xwalk = sleeper[["gsis_id", "espn_id"]].dropna().drop_duplicates("espn_id")
        joined = rows.merge(xwalk, on="espn_id")
        hit = joined[joined["gsis_id"] == player_id]
    if hit.empty and {"position", "espn_name"}.issubset(rows.columns):
        hit = rows[(rows["position"] == position)
                   & (rows["espn_name"].map(_norm_name) == _norm_name(name))]
    if hit.empty:
        return None
    r = hit.iloc[0]
    games = r.get("proj_games")
    if games is None or pd.isna(games) or float(games) <= 0:
        return None
    games = float(games)

    def per_game(*cols):
        total = 0.0
        seen = False
        for c in cols:
            v = r.get(c)
            if v is not None and not pd.isna(v):
                total += float(v)
                seen = True
        return round(total / games, 1) if seen else None

    # Touchdowns are two ids added together for a skill player and a third
    # for a quarterback, kept apart here because they are not the same fact:
    # `tds` on a season row is rushing plus receiving (see
    # similarity.py, where the column is built) and never counts a throw, so
    # a projected `tds` that folded passing in would be compared against a
    # column that excludes it.
    #
    # Rushing yards on their own as well as the combined `yards`, because a
    # quarterback's rushing row has nothing to do with his receiving zero and
    # the combined figure would silently answer for both.
    return {"games": round(games, 1),
            "carries": per_game("proj_carries"),
            "targets": per_game("proj_targets"),
            "receptions": per_game("proj_receptions"),
            "yards": per_game("proj_rush_yards", "proj_rec_yards"),
            "rush_yards": per_game("proj_rush_yards"),
            "tds": per_game("proj_rush_tds", "proj_rec_tds"),
            "attempts": per_game("proj_pass_att"),
            "pass_yards": per_game("proj_pass_yards"),
            "pass_tds": per_game("proj_pass_tds"),
            "interceptions": per_game("proj_interceptions")}


def _espn_projection(conn, player_id: str, name: str, position: str) -> float | None:
    """Projected season points from the ESPN table, or None.

    gsis<->espn crosswalk first (same as scoring/market.py), then a
    name+position fallback. Missing espn_proj column (a table written
    before the projection pull existed) degrades to None.
    """
    espn = read_table(conn, "espn_adp")
    if espn.empty or "espn_proj" not in espn.columns:
        return None
    e = espn.dropna(subset=["espn_proj"])
    e = e[e["espn_proj"] > 0]
    if e.empty:
        return None
    sleeper = read_table(conn, "sleeper_ids")
    if not sleeper.empty:
        xwalk = sleeper[["gsis_id", "espn_id"]].dropna().drop_duplicates("espn_id")
        hit = e.merge(xwalk, on="espn_id")
        hit = hit[hit["gsis_id"] == player_id]
        if not hit.empty:
            return float(hit.iloc[0]["espn_proj"])
    hit = e[(e["position"] == position)
            & (e["espn_name"].map(_norm_name) == _norm_name(name))]
    if not hit.empty:
        return float(hit.iloc[0]["espn_proj"])
    return None


def game_log(weekly: pd.DataFrame, player_id: str, *,
             season_len: pd.Series | None = None,
             rules: dict | None = None) -> list[dict]:
    """Week-by-week rows, each week's points scored under `rules`.

    The payload key stays `ppr_points` -- it is what the client reads and
    what the consistency chart is keyed on -- but the number in it is this
    league's points, not full PPR, whenever `rules` says so. None is full
    PPR, so every existing caller is unchanged.
    """
    if weekly.empty:
        return []
    wk = weekly[weekly["player_id"] == player_id].copy()
    if wk.empty:
        return []
    wk["ppr_points"] = compute_ppr_points(wk, normalize_rules(rules))
    # League-wide last week per season tells how long the season ran (17 vs
    # 18-week eras included); weeks the player has no row for -- injury,
    # bye, healthy scratch -- become zeroed `dnp` rows so the log shows the
    # games missed, not just the games played. Consumers computing per-game
    # averages must exclude dnp rows.
    # This is the one thing here that is NOT per-player, so a caller passing
    # only this player's rows in `weekly` MUST supply it -- derived from
    # their own rows it would be the last week they played, and every week
    # they missed at the end of a season would vanish from the log instead
    # of showing up as a dnp.
    if season_len is None:
        season_len = weekly.groupby("season")["week"].max()
    by_week = {(int(r["season"]), int(r["week"])): r for _, r in wk.iterrows()}
    position = wk["position"].iloc[-1] if "position" in wk.columns else None
    zero = pd.Series(0.0, index=wk.columns)
    rows = []
    for season in sorted(wk["season"].unique(), reverse=True):
        for week in range(int(season_len[season]), 0, -1):
            r = by_week.get((int(season), week))
            dnp = r is None
            if dnp:
                r = zero
            rows.append({
                "season": int(season),
                "week": week,
                "opponent": None if dnp else r.get("opponent_team"),
                "stat_line": "Did not play" if dnp else _stat_line(r, r.get("position")),
                "stats": _game_stats(r),
                "ppr_points": 0.0 if dnp else round(float(r["ppr_points"]), 1),
                "dnp": dnp,
            })
    return rows


_DEPTH_POSITIONS = ["QB", "RB", "WR", "TE"]


def team_depth_chart(depth: pd.DataFrame, team: str, player_id: str) -> list[dict]:
    """Latest-snapshot offensive depth chart for the player's team.

    Works on the RAW depth_charts schema (dt/team/pos_abb/pos_rank); older
    or unknown schemas degrade to an empty list and the UI hides the card.
    """
    need = {"gsis_id", "team", "pos_abb", "pos_rank", "player_name"}
    if depth.empty or not need.issubset(depth.columns):
        return []
    d = depth[depth["team"] == team].copy()
    if d.empty:
        return []
    if "dt" in d.columns:
        d = d[d["dt"] == d["dt"].max()]
    d["pos_rank"] = pd.to_numeric(d["pos_rank"], errors="coerce")
    out = []
    for pos in _DEPTH_POSITIONS:
        rows = (d[d["pos_abb"] == pos].dropna(subset=["pos_rank"])
                .sort_values("pos_rank").drop_duplicates("gsis_id").head(4))
        players = [{"name": r["player_name"], "rank": int(r["pos_rank"]),
                    "is_me": r["gsis_id"] == player_id}
                   for _, r in rows.iterrows()]
        if players:
            out.append({"position": pos, "players": players})
    return out


def weekly_difficulty(schedules: pd.DataFrame, prior_weekly: pd.DataFrame,
                      team: str, position: str,
                      rules: dict | None = None) -> list[dict]:
    """Week-by-week matchup difficulty for the player's team and position.

    fpa_pg = opponent's prior-season points allowed per game to this
    position, scored under `rules` (None = full PPR). Points allowed is as
    format-dependent as points scored -- a defense that concedes catches
    gives up a third less in a standard league -- and this number sits on the
    same page as the board's `schedule` factor, which `build_board` already
    computes under the league's rules via `factors.schedule_factor`. Leaving
    it full PPR put two differently-priced versions of one quantity in front
    of the same reader.

    pct = its percentile among all teams (high = allows a lot = soft
    matchup). Weeks without a game (bye) carry a null opponent.

    rank / rank_n = the same ordering as a plain league rank, 1 = SOFTEST
    (allows the most), out of however many defences the prior season has
    rows for (32 on data/nfl.duckdb, for every position). Added because the
    owner asked for it in those words: a percentile is a number you have to
    convert before you can use it, and "41st percentile" and "20th of 32"
    are the same fact with one of them readable at a glance. `pct` is left
    exactly as it was -- it is what the existing schedule strip renders, and
    changing it would move a number nobody asked to move.

    Note the DIRECTION, because it is the opposite of a difficulty rank: a
    HIGH `fpa_pg` is a GOOD matchup, so rank 1 goes to the defence that gave
    up the most, not the least (2025, RB: Cincinnati 1st at 28.3 allowed,
    Denver 32nd at 16.9).

    K is included exactly when the league prices kicking, and for the same
    reason scoring/board.py stops neutralising a kicker's `schedule` factor
    then: both numbers are the same quantity -- prior-season points allowed
    to this position, under these rules -- and the board already shows its
    percentile on the header of this very page. Leaving this list empty for a
    league that DOES score kicking would put a schedule percentile on the
    card with nothing behind it; filling it for a league that does NOT would
    be seventeen weeks of "0.0 points allowed".

    DST has no positional FPA at all (no weekly rows to allow points to), so
    it is always an empty list and the card stays hidden.
    """
    positions = set(_DEPTH_POSITIONS)
    if prices_kicking(rules):
        positions.add("K")
    if schedules.empty or prior_weekly.empty or position not in positions:
        return []
    wk = prior_weekly.copy()
    wk["ppr_points"] = compute_ppr_points(wk, normalize_rules(rules))
    def_games = wk.groupby("opponent_team")["week"].nunique()
    allowed = (wk[wk["position"] == position]
               .groupby("opponent_team")["ppr_points"].sum() / def_games).dropna()
    if allowed.empty:
        return []
    pct = allowed.rank(pct=True) * 100
    # `method="min"` so a tie reads as a shared position ("two teams 5th")
    # rather than a fractional rank, matching every other rank in this file.
    rank = allowed.rank(ascending=False, method="min")
    rank_n = int(len(allowed))
    games = {}
    mine = schedules[(schedules["home_team"] == team) | (schedules["away_team"] == team)]
    for _, g in mine.iterrows():
        week = int(g["week"])
        if week > 18:
            continue
        home = g["home_team"] == team
        games[week] = (g["away_team"] if home else g["home_team"], home)
    rows = []
    for week in range(1, 19):
        opp, home = games.get(week, (None, None))
        rows.append({
            "week": week, "opponent": opp, "home": home,
            "fpa_pg": round(float(allowed[opp]), 1) if opp in allowed.index else None,
            "pct": round(float(pct[opp])) if opp in pct.index else None,
            "rank": int(rank[opp]) if opp in rank.index else None,
            "rank_n": rank_n if opp in rank.index else None,
        })
    return rows


def comparable_cohort(pool: pd.DataFrame, feats: pd.DataFrame, player_id: str,
                      rookie_season: int | None = None) -> dict | None:
    """Seasons that looked like this player's latest one, and what happened next.

    `similar` (stat twins) answers "who does he most resemble" and returns
    five names. This answers a different question the card needs -- "what
    does a season like this usually do next" -- and that needs a
    distribution, not a top five. So the cohort is not a nearest-neighbour
    list at all: it is every qualifying season inside a band, unranked, with
    the median and the decline count that only mean anything over a whole
    group.

    THE BAND IS A CALIBRATION, NOT A DERIVATION. See COMP_PPG_BAND above for
    the three bands that were measured and why +-3 ppg / +-1 NFL season was
    picked, and note that the payload carries the band it used so the card
    can say what it did.

    The target season is the latest one this player has, chosen the same way
    `find_twins` chooses its own (`sort_values("season").iloc[-1]`) so the
    two blocks on the page are talking about the same season. It is also
    excluded from its own cohort -- a season is not a comparable for itself
    -- but the player's OTHER seasons are NOT excluded. That is deliberate:
    Gibbs' own 2024 is one of the nineteen seasons that most resembles his
    2025, and dropping it would be dropping evidence for a tidiness that
    helps nobody. (Where it happens to be moot, as here: 2025 has no 2026 to
    change into, so it could never have entered its own cohort anyway.)

    `change` is served per comparable, not just `next_ppg`. The raw next
    year is the number the twin block already shows and it makes the reader
    do the subtraction against a different starting point for every row;
    the card wants the distribution of MOVEMENT, which is one column.

    `rookie_season` is passed in rather than looked up here -- `player_bio`
    has already resolved it out of `players` for the header, and doing it
    twice cost 2.3 ms of `drop_duplicates` over 25,044 rows per click. None
    (a player nflverse has no rookie season for) drops the experience band
    and keeps the points-per-game one, which is a wider cohort honestly
    labelled: the payload's `exp_band` comes back null to say so.

    None when there is no target season at all (no weekly history: rookies,
    defenses) -- the same "hide the card" signal the empty lists elsewhere
    give.
    """
    if feats.empty or pool.empty:
        return None
    mine = feats[feats["player_id"] == player_id]
    if mine.empty:
        return None
    target = mine.sort_values("season").iloc[-1]
    exp = (int(target["season"]) - int(rookie_season) + 1
           if rookie_season is not None else None)

    band = pool[(pool["position"] == target["position"])
                & ((pool["ppg"] - target["ppg"]).abs() <= COMP_PPG_BAND)]
    if exp is not None:
        band = band[(band["nfl_season"] - exp).abs() <= COMP_EXP_BAND]
    band = band[~((band["player_id"] == player_id)
                  & (band["season"] == target["season"]))]

    change = band["change"]
    comps = [{"player_id": r["player_id"], "name": r["name"],
              "season": int(r["season"]),
              "nfl_season": _int_or_none(r["nfl_season"]),
              "ppg": round(float(r["ppg"]), 1),
              "next_ppg": round(float(r["next_ppg"]), 1),
              "change": round(float(r["change"]), 1)}
             for _, r in band.sort_values("change").iterrows()]
    return {
        "season": int(target["season"]),
        "position": target["position"],
        "ppg": round(float(target["ppg"]), 1),
        "nfl_season": exp,
        # The calibration, served so the card can state it.
        "ppg_band": COMP_PPG_BAND,
        "exp_band": COMP_EXP_BAND if exp is not None else None,
        "min_games": RANK_MIN_GAMES,
        "n": len(comps),
        "median_change": _round_or_none(change.median(), 1),
        "declined": int((change < 0).sum()),
        "improved": int((change > 0).sum()),
        "players": comps,
    }


def _target_share_by_game(mine: pd.DataFrame,
                          team_week: pd.DataFrame) -> dict:
    """(season, week) -> share of his team's targets that game.

    None where the team total is zero or missing rather than 0.0: a game the
    denominator cannot be built for is a gap, and a zero would read as a
    receiver nobody threw to.
    """
    need = {"season", "week", "recent_team", "targets"}
    if mine.empty or not need.issubset(mine.columns) or team_week.empty:
        return {}
    m = mine[["season", "week", "recent_team", "targets"]].copy()
    m["targets"] = pd.to_numeric(m["targets"], errors="coerce")
    m = m.merge(team_week, on=["season", "week", "recent_team"], how="left")
    share = m["targets"] / m["team_targets"].where(m["team_targets"] > 0)
    return {(int(r.season), int(r.week)): v
            for r, v in zip(m.itertuples(), share) if pd.notna(v)}


def snap_share_by_game(conn, crosswalk: pd.DataFrame, snap_columns,
                       player_id: str) -> dict:
    """(season, week) -> offensive snap share, for every regular-season game.

    The season means the profile already carries hides the story the card
    wants to tell. Gibbs' 2025 reads 0.66 as one number; week by week it is
    66, 56, 69, 62, 52, 69, 56, 66, 50, 73, 74, 70, 69, 81, 86, 69, 71 -- a
    role that grew through the year, which is a different thing to know
    about a running back than "two thirds of the snaps".

    Keyed through `oline.reconcile_pfr_to_gsis` (see
    profile_cache._pfr_crosswalk) rather than by name: the season aggregate
    can afford a name+team+season join because it is averaging, and this
    cannot -- attaching the wrong Michael Jordan's snaps to seventeen
    individual games is a per-game lie, not a smoothed one.

    REG only, unlike `snap_share`, which averages every row `snap_counts`
    has including playoffs. This has to be REG because it is joined onto
    `game_log`, whose rows come from `weekly` -- a REG-only table on this
    database (verified: `SELECT DISTINCT season_type FROM weekly` is
    ['REG']). A playoff snap row would otherwise land on the week-19-and-up
    rows the log does not have, or worse, collide with a regular-season week
    number from an 18-week era.

    An empty dict is the answer for a player the crosswalk cannot resolve
    (6.6% of skill-position snap rows), for a database with no snap table,
    and for a defense. Every game_log row then carries a null, not a zero.
    """
    if crosswalk is None or crosswalk.empty:
        return {}
    hit = crosswalk[crosswalk["gsis_id"] == player_id]
    if hit.empty:
        return {}
    # `snap_columns` comes off the frame profile_cache already read, rather
    # than an information_schema query per request (1.1 ms, measured).
    cols = set(snap_columns)
    if not {"season", "week", "offense_pct", "pfr_player_id"}.issubset(cols):
        return {}
    # The filter is pushed into SQL for the same measured reason
    # `_player_weekly` pushes its own: snap_counts is 253,106 rows and this
    # wants at most ~120 of them (1.6 ms against 0.165s for the whole table).
    where = "pfr_player_id = ?"
    params = [hit.iloc[0]["pfr_player_id"]]
    if "game_type" in cols:
        where += " AND game_type = 'REG'"
    rows = conn.execute(
        f"SELECT season, week, offense_pct FROM snap_counts WHERE {where}",
        params).df()
    return {(int(r["season"]), int(r["week"])): r["offense_pct"]
            for _, r in rows.iterrows() if not pd.isna(r["offense_pct"])}


def team_line_quality(lq: pd.DataFrame, season: int, team, position) -> dict | None:
    """The team's offensive line, ranked against the other 31.

    `scoring/oline.py` has been built, tested and unused since it was
    written -- imported by nothing but its own test file. This is the wiring.
    The composite AND its four components are all served, because the
    composite alone is an opaque 0-100 and the parts are what make it
    arguable: Detroit's 2026 line is 21st at 44.4 not because it is falling
    apart (continuity 0.815) but because its five starters have missed a
    quarter of their careers between them (availability 0.748) -- a
    different thing for a manager to know.

    `rank` is out of `len(lq)` (32 on a full database), 1 = best line, which
    is the direction `line_quality` already sorts in.

    DST GETS NULL, DELIBERATELY. A defense's own team's offensive line tells
    you nothing about the defense -- it is a fact about the eleven players
    who leave the field when it comes on. Serving it anyway would put a
    plausible-looking number on the card that means nothing, which is worse
    than a blank. Kickers keep it: a kicker's attempts come from his own
    offence moving the ball, so the line is the same input for him as for a
    running back, just weaker.
    """
    if lq is None or lq.empty or position == "DST" or not isinstance(team, str):
        return None
    ranked = lq.reset_index(drop=True)
    hit = ranked.index[ranked["team"] == team]
    if len(hit) == 0:
        return None
    r = ranked.loc[hit[0]]
    return {
        "season": int(season),
        "team": team,
        "rank": int(hit[0]) + 1,
        "teams": int(len(ranked)),
        "line_quality": _round_or_none(r["line_quality"], 1),
        "continuity": _round_or_none(r["continuity_raw"], 3),
        "availability": _round_or_none(r["availability_raw"], 3),
        "returning": _round_or_none(r["returning_raw"], 3),
        "experience": _round_or_none(r["experience_raw"], 1),
    }


def player_bio(players: pd.DataFrame, player_id: str, season: int) -> dict:
    """Birth date, rookie season, and the two numbers derived from them.

    AGE IS AS OF SEPTEMBER 1 OF THE SEASON YEAR, which is not a new
    convention invented here: it is `similarity._age_in_season`, which the
    stat-twin block has always matched comparables on, imported rather than
    re-implemented so the "age 23" on one half of the card cannot disagree
    with the "age 23" on the other. September 1 is opening week, so the
    number is "how old was he while he was playing that season" rather than
    "how old was he on some arbitrary January boundary" -- and for a player
    born in the middle of a season, the first is the honest one. (Jahmyr
    Gibbs, born 2002-03-20: 23 through the whole of 2025, 24 through 2026.)

    `nfl_season` is 1-BASED -- a rookie year is his 1st NFL season, not his
    0th -- from `players.rookie_season`, the same column `oline._experience`
    counts service time off. Gibbs' rookie season is 2023, so 2025 is his
    third and the 2026 he is being drafted for is his fourth.

    `season` is the season being drafted, so the header reads as the player
    will be this year. Each row of `seasons` carries its own `age` and
    `nfl_season` for the same two facts as of THAT year.

    Everything is None for a player the `players` table has no row for --
    every defense, and any player nflverse has no biography for. Nulls, not
    zeros: "age 0" is a claim, "age unknown" is the truth.
    """
    out = {"season": int(season), "birth_date": None, "rookie_season": None,
           "age": None, "nfl_season": None}
    if players.empty or "gsis_id" not in players.columns:
        return out
    hit = players[players["gsis_id"] == player_id]
    if hit.empty:
        return out
    row = hit.iloc[0]
    birth = row.get("birth_date")
    if birth is not None and not pd.isna(birth):
        out["birth_date"] = str(pd.Timestamp(birth).date())
        out["age"] = _age_in_season(birth, int(season))
    rookie = row.get("rookie_season")
    if rookie is not None and not pd.isna(rookie):
        out["rookie_season"] = int(rookie)
        out["nfl_season"] = int(season) - int(rookie) + 1
    return out


def _outlook(weekly: pd.DataFrame, depth: pd.DataFrame, schedules: pd.DataFrame,
             player_row: dict, rules: dict | None = None) -> dict:
    """Depth slot, implied points, strength of schedule and bye week.

    `weekly` is only ever read as `weekly[weekly.season == max(season)]`
    below, so handing this the latest-season slice instead of the whole
    table produces the identical frame (the re-slice becomes a no-op that
    keeps every row, index and order intact) -- which is what
    `build_profile` does, off profile_cache's `prior_weekly`.

    `depth` likewise is only read as `adapted[adapted.gsis_id == player_id]`,
    so a slice already narrowed to this player is enough.
    """
    team = player_row.get("team")
    position = player_row.get("position")
    player_id = player_row.get("player_id")

    depth_slot = None
    adapted = _adapt_depth_charts(depth)
    if not adapted.empty and "gsis_id" in adapted.columns:
        mine = adapted[adapted["gsis_id"] == player_id]
        if not mine.empty:
            ranks = pd.to_numeric(mine["depth_team"], errors="coerce").dropna()
            if not ranks.empty:
                depth_slot = int(ranks.min())

    implied_points = None
    sos_raw = None
    sos_pct = None
    bye = None

    if not schedules.empty:
        env = factors.environment_factor(schedules)
        erow = env[env["team"] == team]
        if not erow.empty:
            implied_points = _round_or_none(erow.iloc[0]["env_raw"], 1)

        if not weekly.empty:
            prior = weekly[weekly["season"] == weekly["season"].max()]
            # The same `rules` build_board hands schedule_factor, so the
            # profile's `sos_raw` is the raw number behind the board's
            # `schedule` percentile rather than a second, PPR-priced version
            # of it. `environment_factor` and `bye_weeks` above and below read
            # only betting lines and the schedule grid, so no scoring rule
            # touches them.
            sos = factors.schedule_factor(prior, schedules, normalize_rules(rules))
            pos_sos = sos[sos["position"] == position]
            srow = pos_sos[pos_sos["team"] == team]
            if not srow.empty:
                sos_raw = _round_or_none(srow.iloc[0]["sos_raw"], 1)
                pct = pos_sos["sos_raw"].rank(pct=True) * 100
                sos_pct = _round_or_none(pct.loc[srow.index[0]], 1)

        byes = factors.bye_weeks(schedules)
        brow = byes[byes["team"] == team]
        if not brow.empty:
            b = brow.iloc[0]["bye"]
            bye = None if pd.isna(b) else int(b)

    return {"depth_slot": depth_slot, "implied_points": implied_points,
            "sos_raw": sos_raw, "sos_pct": sos_pct, "bye": bye}


def _proj_pos_finish(board: pd.DataFrame, player_id: str) -> int | None:
    """Where a projection places a player among his own position.

    The same shape `pos_finish` gives a season that has been played -- a rank
    within (position), best first -- so the popup's Finish panel can draw the
    projection as one more column on the ladder the played seasons are already
    on, rather than as a number in a different unit beside them.

    Ranked over everyone the board carries a projection for, NOT over the
    startable slice: the pool a finish is graded against is the whole position
    (see `finishTone`, which does the grading against the league's starter
    count afterwards). `method="min"` so two identical projections share the
    better place instead of both landing on the average of two, which is what
    a finish means everywhere else in this project.

    None when this player has no projection -- a rookie the board could not
    price, or a defense in a league that does not score one. A rank invented
    for a player with nothing to rank would be the one number on the card that
    came from nowhere.
    """
    if "proj_points" not in board.columns or "position" not in board.columns:
        return None
    priced = board[board["proj_points"].notna()]
    if priced.empty:
        return None
    ranks = priced.groupby("position")["proj_points"].rank(ascending=False,
                                                           method="min")
    hit = ranks[priced["player_id"] == player_id]
    if hit.empty or pd.isna(hit.iloc[0]):
        return None
    return int(hit.iloc[0])


def _enrich_twins(twins: dict, board: pd.DataFrame) -> dict:
    board_idx = board.set_index("player_id")
    for p in twins["players"]:
        if p["player_id"] in board_idx.index:
            r = board_idx.loc[p["player_id"]]
            p["rank"] = int(r["rank"])
            market_rank = r["market_rank"]
            p["market_rank"] = None if pd.isna(market_rank) else float(market_rank)
        else:
            p["rank"] = None
            p["market_rank"] = None
    return twins


def _player_weekly(conn, player_id: str) -> pd.DataFrame:
    """This player's weekly rows only.

    `read_table(conn, "weekly")` pulled all 174,373 rows x 145 columns
    (0.247s) so that `season_summaries` and `game_log` could immediately
    throw away everything but the ~100 rows belonging to one player. The
    same filter in SQL costs 36ms.

    VERIFIED, not assumed, because both callers are order-sensitive
    (`groupby("season")["recent_team"].last()` and `wk["position"].iloc[-1]`
    both depend on row order, and a differently-ordered scan would quietly
    change a team or a position): for all 249 board players this returns a
    frame `assert_frame_equal`-identical -- values, order AND dtypes -- to
    `read_table(conn, "weekly")[weekly.player_id == player_id]`. The
    mechanism is DuckDB's `preserve_insertion_order`, which defaults to
    true and makes a filtered scan yield storage order just as a full scan
    does. If that setting is ever turned off for this connection, this is
    the line to revisit.
    """
    cols = _table_columns(conn, "weekly")
    if not cols:
        return pd.DataFrame()
    if "player_id" not in cols:
        return read_table(conn, "weekly")
    return conn.execute("SELECT * FROM weekly WHERE player_id = ?", [player_id]).df()


def _depth_slice(conn, team, player_id: str) -> pd.DataFrame:
    """The depth-chart rows for this player's team, plus this player's own.

    depth_charts is the biggest table the profile touched -- 416,885 rows
    over 142 daily snapshots, 0.233s to read, 275 MB in memory -- and the
    two consumers between them want one team's rows (`team_depth_chart`)
    and one player's rows (`_outlook`). The union of the two filters is a
    single query, 18ms, and both consumers then narrow it exactly as they
    did before: `team_depth_chart` takes `depth.team == team` and the
    latest `dt` WITHIN that team, which the OR-clause cannot disturb
    because it never removes a row of that team.

    Verified identical (`assert_frame_equal`, dtypes included) to
    `read_table(conn, "depth_charts")[(team match) | (gsis_id match)]` for
    all 249 board players. A schema without both columns -- the fixture in
    tests/test_profile.py writes depth_charts with no `team` column at all
    -- falls back to the full read, which is what those rows are sized for.
    """
    cols = _table_columns(conn, "depth_charts")
    if not cols:
        return pd.DataFrame()
    if not {"team", "gsis_id"}.issubset(cols):
        return read_table(conn, "depth_charts")
    # A NaN/None team must match nothing, which is what `= NULL` does and
    # what pandas' `depth["team"] == nan` did.
    team = team if isinstance(team, str) else None
    return conn.execute(
        "SELECT * FROM depth_charts WHERE team = ? OR gsis_id = ?",
        [team, player_id]).df()


def _ts(v):
    """A timestamp as an ISO-8601 string, or None.

    Every other date on this page is stringified before it leaves
    (`player_bio` does `str(pd.Timestamp(birth).date())`), and it has to be:
    `_scrub` passes a `datetime`/`pd.Timestamp` straight through -- it is a
    scalar and it is not NaN -- and `tests/test_profile.py::test_no_nan_
    anywhere` calls `json.dumps(payload, allow_nan=False)`, which raises
    TypeError on one. A date rendered by whatever the web framework happens
    to do with a datetime is also a payload key whose format nobody chose.
    """
    if v is None or pd.isna(v):
        return None
    return pd.Timestamp(v).isoformat()


# Written by pipeline/news.py's NEWS_COLUMNS / STATUS_COLUMNS. Named here as
# the minimum this reader needs rather than as the full column list, so a
# later column ADDED by the pipeline does not make these reads fall back to
# empty.
_NEWS_NEEDS = {"player_id", "headline", "url", "published_at", "source",
               "attribution", "fetched_at"}
_STATUS_NEEDS = {"player_id", "sleeper_id", "injury_status",
                 "injury_body_part", "injury_notes", "depth_chart_position",
                 "depth_chart_order", "news_updated", "fetched_at"}


def player_news(conn, player_id: str, columns=None,
                limit: int = NEWS_ITEMS) -> list[dict]:
    """This player's recent headlines, newest first. `[]` when there are none.

    ATTRIBUTION SURVIVES INTACT, which is the reason this returns a list of
    dicts rather than a list of (headline, url) pairs. `pipeline/news.py`
    mixes two feeds into one table and stamps every row with which one it
    came from: ATTR_EXACT rows were tagged with the player's ESPN athlete id
    by ESPN itself, ATTR_NAME rows came back from a name+team search that
    measured ~93-96% relevant. Those are not the same claim, the card shows
    them differently, and flattening them into one undifferentiated feed
    would present a good guess as a fact. The marker is passed through
    verbatim -- not translated to a boolean, not dropped.

    ORDER: most recent first; an exact item wins a tie on `published_at`;
    `url` breaks what is left, only so the order is deterministic. The
    middle term is a rule about what SHOULD happen rather than one that
    changes today's output -- measured on the 2,346 stored rows, 114 of
    2,192 (player, published_at) groups hold more than one item, and not one
    of those ties mixes the two attributions (Google sends batches of items
    sharing a pubDate to the second). Without the last term those 114 groups
    would come back in whatever order the scan produced, which is stable in
    practice and guaranteed by nothing.

    Recency is the PRIMARY key and exactness only the tiebreak, deliberately:
    attribution measures how sure we are that an item is about this player,
    not how much it matters. A three-week-old ESPN article is not more use
    to a manager on the clock than eight things written this week, so it is
    not pinned above them. See NEWS_ITEMS for what the cap costs.

    NOT BEHIND `cached_profile_frames`, unlike every other cross-player
    frame the card needed: this is a per-player read of ~8 rows out of a
    2,346-row table, the same shape as `_player_weekly` and `_depth_slice`
    above it. Caching the table instead would put ~1.2 MB into a 46 MB
    cache entry and `.copy()` it on every click to hand back eight rows.

    WHAT THE WHOLE ADDITION COSTS, this function plus `player_status`: 1.58
    ms on a warm click that was 82.2 ms without it, +1.9% -- measured as a
    single-process A/B (40 alternating pairs, the two calls stubbed out and
    restored in the same interpreter) so that process-to-process variance
    could not be read as a cost. Both are per-player SQL, so nothing here
    touches the 1.89s cold build.

    `.fetchall()` rather than `.df()`: eight rows do not need a DataFrame,
    and going through pandas would cost a conversion and hand back NaT for
    a null timestamp where DuckDB hands back None.

    `columns` is the table's column names, off `ProfileFrames.news_columns`,
    on the same argument `snap_share_by_game` takes `snap_columns`: the
    information_schema round trip this replaces measured 0.53 ms against
    0.62 ms for the query itself -- the schema lookup cost almost as much as
    the read. None (the default) means "look it up", so a caller holding no
    frames -- every test of this function -- still works; an empty frozenset
    means the table is not there.
    """
    if columns is None:
        columns = _table_columns(conn, "player_news")
    if not _NEWS_NEEDS.issubset(columns):
        # Missing table (no `make refresh` has written one yet -- true of
        # data/nfl.duckdb as this was wired) or a schema this reader cannot
        # honour. An empty feed is the honest answer, and it is the same one
        # a player with no news gets; there is no half-served version of an
        # item whose attribution is unknown.
        return []
    rows = conn.execute(
        "SELECT headline, url, published_at, source, attribution, fetched_at "
        "FROM player_news WHERE player_id = ? "
        "ORDER BY published_at DESC NULLS LAST, (attribution = ?) DESC, url "
        "LIMIT ?", [player_id, ATTR_EXACT, int(limit)]).fetchall()
    # `_scrub`ed here rather than only by `build_profile`: an all-null
    # `source` column (a feed that sent no publication names) comes out of
    # pandas as float nan, and these two are reachable without going through
    # the payload assembly at all. It is idempotent, so the outer scrub still
    # sees exactly what it saw before.
    return [
        _scrub({
            "headline": h,
            "url": u,
            "published_at": _ts(p),
            "source": s,
            "attribution": a,
            # Per item, not per feed, because it genuinely varies within one
            # player's rows: fetch_player_news re-reads ESPN every run but
            # carries a player's name-matched rows forward when his query
            # fails or is inside the TTL, so the two halves can be hours or
            # days apart. It is the only thing that lets the card say how old
            # the FEED is, as opposed to how old the news in it is.
            "fetched_at": _ts(f),
        })
        for h, u, p, s, a, f in rows
    ]


def player_status(conn, player_id: str, columns=None) -> dict | None:
    """Sleeper's injury and depth-chart signals for one player, or None.

    SERVED AS A SIBLING OF `header`, NOT INSIDE `news`. This is the one field
    on the card that changes a pick: a manager wants to see "Questionable"
    before he spends the pick, not after scrolling a feed. Nested under the
    news it would be `payload.news.status.injury_status` and would read as
    another headline; at the top level the header component reaches it with
    one lookup and can render it beside the name.

    NULL IS NOT "HEALTHY", and this never invents the word. Measured on the
    live Sleeper payload (12,221 players, 2026-08-19), `injury_status` takes
    the values Questionable / IR / PUP / Sus / Out / Doubtful / NA / DNR /
    COV, and null -- there is no "Healthy" and no "Active" in it. A player
    with nothing wrong with him is published with the field empty, so the
    honest payload is `injury_status: None` and the card decides whether a
    blank badge is worth drawing. `source` names Sleeper for the same
    reason `attribution` exists on a headline: a status is a claim by
    somebody, and the client should be able to say who.

    RETURNS None -- not a dict of nulls -- WHEN SLEEPER HAS NO ROW FOR HIM.
    `pipeline.news.parse_sleeper_status` reindexes onto the whole board, so
    every board player HAS a `player_status` row; a `sleeper_id` of null is
    what "we looked and Sleeper does not carry this player" looks like. All
    26 defenses are in that state (a team defense is not a Sleeper player)
    and 1 of 24 kickers. A dict whose every value is null says the same
    thing in a shape the client has to inspect field by field.

    DELIBERATELY ABSENT: `practice_participation`. It is a real column on
    the table and reads like exactly what an injury panel wants, but it is
    null for all 3,186 active rostered players in the live payload (1 of
    12,221 overall, and he is not on any roster) -- so serving it would put
    an always-blank line on the card. pipeline/news.py's own comment about
    `injury_start_date` records the same shape of finding. If Sleeper ever
    starts populating it, re-measure and add it back.

    `columns` behaves exactly as it does in `player_news` above: the cached
    column list, or None to look it up.
    """
    if columns is None:
        columns = _table_columns(conn, "player_status")
    if not _STATUS_NEEDS.issubset(columns):
        return None
    row = conn.execute(
        "SELECT sleeper_id, injury_status, injury_body_part, injury_notes, "
        "depth_chart_position, depth_chart_order, news_updated, fetched_at "
        "FROM player_status WHERE player_id = ? LIMIT 1", [player_id]).fetchone()
    # `pd.isna`, not `is None`. A `player_status` whose `sleeper_id` is null
    # for EVERY row -- a DST-only fixture, or a database whose Sleeper fetch
    # failed outright -- gives pandas an all-null column, which DuckDB stores
    # as DOUBLE and hands back as float nan rather than None. `is None` would
    # miss that and return a dict of nans; the same reasoning applies to
    # `depth_chart_order` below, where `int(nan)` is a ValueError.
    if row is None or pd.isna(row[0]):
        return None
    _, status, part, notes, slot, order, updated, fetched = row
    # Scrubbed for the same reason `player_news` scrubs: `injury_status` and
    # friends are all-null often enough that pandas hands DuckDB a float
    # column, and a nan in the payload is a TypeError out of
    # `json.dumps(allow_nan=False)`, not a visible bug.
    return _scrub({
        "injury_status": status,
        "injury_body_part": part,
        "injury_notes": notes,
        # Sleeper's own depth chart, which is a different source from the
        # nflverse `depth_charts` table behind `outlook.depth_slot` -- kept
        # separate rather than merged, because two sources disagreeing is
        # information and a silently-preferred one is not.
        "depth_chart_position": slot,
        "depth_chart_order": _int_or_none(order),
        # Sleeper's "when did this player last make news" stamp. The cheapest
        # freshness signal on the panel: it says whether the injury line is
        # current without the client having to read the headlines.
        "news_updated": _ts(updated),
        "fetched_at": _ts(fetched),
        "source": "sleeper",
    })


def build_profile(conn, player_id: str, weights: dict | None = None,
                  settings: "league.LeagueSettings | None" = None) -> dict | None:
    # `settings` (the league's roster shape AND its scoring rules) is loaded
    # here the same way `build_board` loads it, and for the same reason: every
    # derived number below is priced in the league's points, not in full PPR.
    # Before this it was never loaded at all, so a half-PPR league's profile
    # showed PPR ppg, PPR volatility, a PPR game log, PPR stat twins matched
    # on PPR features with a PPR "next year" forecast, and a PPR points-
    # allowed schedule -- every one of them beside a header row the board had
    # already priced correctly.
    #
    # Passed to `cached_build_board` too, not left to default: without it the
    # two would load `league.load(conn)` separately, which is the same answer
    # today but would silently diverge the first time a caller (api/live.py's
    # connect flow already does this for the board) hands in settings fetched
    # live from ESPN instead of the database's.
    settings = settings or league.load(conn)
    rules = settings.scoring
    # Was `build_board(conn, weights)` -- every profile click rebuilt the
    # whole 249-row board (all factors, composite, VOR, tiers, a five-source
    # market consensus) just to read one row back out, measured at ~3.1s
    # end to end. cached_build_board (scoring/board_cache.py) reuses the same
    # board `GET /api/players` just built, or built for a prior profile
    # click with the same weights/settings/drafted state, and only
    # recomputes when one of those actually changed -- see that module's
    # docstring for the exact key and the staleness failure it guards
    # against (a profile showing a just-picked player as still available).
    board = cached_build_board(conn, weights, settings)
    match = board[board["player_id"] == player_id]
    if match.empty:
        return None
    row = match.iloc[0]
    header = row.to_dict()
    # Each market number as a place among his own position -- see
    # `_market_pos_ranks` for why a drafter needs WR7 and not 19.
    header["market_pos"] = _market_pos_ranks(board, player_id)
    factors_out = {k: header[k] for k in
                   ("production", "durability", "role", "environment", "schedule")}
    # The projection as a PLACE, next to the places his played seasons took.
    # `proj_points` alone is a number nobody has intuitions about; "RB2" is
    # the same fact in the unit the rest of the card is already written in.
    header["proj_pos_finish"] = _proj_pos_finish(board, player_id)

    # Was four full `read_table` calls -- weekly (174,373 rows), snap_counts
    # (253,106), depth_charts (416,885) and schedules -- 0.614s of reading
    # 844k rows per click, before a single one of them was filtered down to
    # one player. cached_profile_frames (scoring/profile_cache.py) holds the
    # league-wide aggregates that survive the filtering (the season-features
    # frame, the snap-share aggregate, the latest season's weekly rows,
    # season lengths, schedules, players) and re-derives them only when
    # `meta` says a pipeline refresh has happened -- the same signal
    # scoring/board_cache.py keys on, imported from it rather than
    # re-invented. The two genuinely per-player reads have their filters
    # pushed into SQL instead.
    frames = cached_profile_frames(conn, rules)
    schedules = frames.schedules
    wk_mine = _player_weekly(conn, player_id)
    depth = _depth_slice(conn, header["team"], player_id)

    if header["position"] == "K" and not prices_kicking(rules):
        # Kickers have weekly rows, and this branch STILL blanks them -- but
        # on the league's scoring rather than on the position, which is what
        # the original reasoning was actually about.
        #
        # It read: "the PPR formula doesn't score kicking stats, so every one
        # of those rows nets 0 points -- a history of zeros is misleading,
        # not informative." Every word of that is still true of a league that
        # prices no kicking, and every word of it stops being true the moment
        # one does. Deleting the branch outright would have been wrong in the
        # other direction: `scoring.league.ESPN_STAT_COLUMNS` only teaches
        # this app to READ a league's kicking rules, it does not give a
        # league that has none. A database whose `league` table predates that
        # map, and any league that genuinely scores kicking at nothing, still
        # produces exactly the column of zeros this was written to hide -- so
        # the test is `prices_kicking(rules)`, not `position == "K"`.
        #
        # DST is not mentioned because it does not reach here: `weekly` holds
        # no defense rows at all, so `season_summaries` and `game_log` return
        # empty on their own and a defense's profile has no season table and
        # no game log to blank.
        #
        # THAT IS NOW TRUE FOR A DIFFERENT REASON THAN IT USED TO BE. This
        # comment used to say defensive scoring "is not derivable from this
        # database", and bc7b714 made it derivable: `from_espn` reads a
        # league's D/ST rules out of `pointsOverrides["16"]` and
        # `ppr.compute_dst_points` scores ESPN's own per-week D/ST stat lines
        # from the `dst_weekly` table. What did NOT change is this function:
        # `dst_weekly` is keyed by team and ESPN stat id and is read by
        # `board.dst_team_history` only. Nothing joins it to `weekly`, so a
        # defense still has nothing per-game to show HERE. Giving one a game
        # log means teaching `_player_weekly` about `dst_weekly`, which is a
        # real piece of work and not a config toggle -- and note `dst_weekly`
        # is absent entirely until `make refresh` has run its job.
        seasons = []
        logs = []
    else:
        seasons = season_summaries(wk_mine, None, player_id,
                                   season_features=frames.season_features,
                                   snap_share=frames.snap_share, rules=rules,
                                   season_ranks=frames.season_ranks)
        logs = game_log(wk_mine, player_id, season_len=frames.season_len,
                        rules=rules)
    outlook_out = _outlook(frames.prior_weekly, depth, schedules, header, rules)

    if header["position"] in _KDST_POSITIONS:
        similar = value_neighbors(board, player_id)
    else:
        players = frames.players
        # `wk_mine` is ignored: find_twins' only use of its `weekly`
        # argument is `player_season_features`, which is what
        # `season_features` supplies. The guard stays on the FULL table
        # being empty, which is what it always tested -- `frames`
        # carries that flag for exactly this line.
        # No `rules` argument: `season_features` is supplied, and
        # find_twins ignores `rules` entirely on that path (it is only used
        # to price the frame it is not being asked to build). The frame IS
        # priced under this league's rules -- cached_profile_frames was given
        # them above and keys its cache on them -- so both the distance the
        # twins are matched on and the `next_ppg` they are reported with are
        # this league's points.
        twins = (find_twins(wk_mine, player_id, players=players,
                            season_features=frames.season_features)
                 if not frames.weekly_empty else None)
        similar = (_enrich_twins(twins, board) if twins is not None
                   else value_neighbors(board, player_id))

    summary = career_summary(seasons)
    proj_total = _espn_projection(conn, player_id, header["name"], header["position"])
    # Re-priced by the board's own `proj_scale` (scoring/board.projection_scale)
    # rather than shown raw. `proj_total` is ESPN's PPR-shaped season number;
    # `w_ppg` two lines down is this league's points; `proj_delta` subtracts
    # one from the other. Differencing two currencies produced a number that
    # was not wrong so much as meaningless -- in a standard league it read a
    # high-reception WR as a huge positive "projected improvement" that was
    # nothing but the missing reception points. The board row carries the
    # factor so this page and the draft room cannot disagree about it.
    proj_scale = header.get("proj_scale")
    if proj_total and proj_scale is not None and not pd.isna(proj_scale):
        proj_total = proj_total * float(proj_scale)
    summary["proj_ppg"] = round(proj_total / 17, 1) if proj_total else None
    # The stat line behind that number, per game, for the usage card's own
    # projected column. Same table, same match, so the points and the carries
    # cannot describe two different players.
    summary["proj_usage"] = _espn_projected_usage(
        conn, player_id, header["name"], header["position"], frames.draft_season)
    summary["proj_delta"] = (round(summary["proj_ppg"] - summary["w_ppg"], 1)
                             if summary["proj_ppg"] is not None and summary["w_ppg"] is not None
                             else None)

    # Identical frame to the old `weekly[weekly.season == weekly.season.max()]`
    # -- profile_cache slices it off the full table with that exact
    # expression, so rows, index and order all match; an empty `weekly`
    # still yields the empty frame this used to fall back to.
    prior = frames.prior_weekly

    # -- everything the redesigned player card added, all of it additive ----
    #
    # `bio` is the season being drafted; each season row also gets the same
    # two facts as of ITS year, which is what makes a history table readable
    # (a 16-ppg season at 21 and a 16-ppg season at 30 are not the same
    # season). Annotated after the fact rather than inside `season_summaries`
    # so that function keeps its "one player's weekly rows plus three
    # league-wide aggregates" shape and does not grow a `players` argument.
    bio = player_bio(frames.players, player_id, frames.draft_season)
    # nflverse's own photo url, null until a `make refresh` has run against a
    # players table that carries the column -- the card renders no image
    # rather than a broken one, so an unrefreshed database is not a regression.
    #
    # Both column names are checked, not just `headshot`: a `players` table
    # can be missing `gsis_id` too (every fixture that seeds one without it,
    # and any database older than that column), and indexing on a column that
    # is not there raises where the whole point of this line is to degrade to
    # None.
    bio["headshot"] = None
    people = frames.players
    if {"gsis_id", "headshot"}.issubset(people.columns):
        row = people[people["gsis_id"] == player_id]
        if not row.empty and pd.notna(row["headshot"].iloc[0]):
            bio["headshot"] = row["headshot"].iloc[0]
    for s in seasons:
        s["age"] = (_age_in_season(bio["birth_date"], s["season"])
                    if bio["birth_date"] else None)
        s["nfl_season"] = (s["season"] - bio["rookie_season"] + 1
                           if bio["rookie_season"] else None)

    # Per-game snap share, hung on the log rows it belongs to rather than
    # served as a parallel list -- the log already carries (season, week) and
    # the opponent for each of them. A dnp row keeps its null: he took no
    # snaps because he did not play, which is not a snap share of zero.
    snaps_by_game = snap_share_by_game(conn, frames.pfr_to_gsis,
                                       frames.snap_columns, player_id)
    for g in logs:
        g["snap_pct"] = (None if g["dnp"] else
                         _round_or_none(snaps_by_game.get((g["season"], g["week"])), 3))

    # Per-game target share, on the same rows and for the same reason. The
    # season figure averages a role away: a receiver who saw a fifth of the
    # targets in September and a tenth in December reads the same as one who
    # was steady all year, and which of those he is decides the pick.
    #
    # Off THIS player's own weekly rows and the league-wide team totals -- so
    # the week he was traded divides his targets by the team he actually
    # played for, which is the team his own row names.
    targets_by_game = _target_share_by_game(wk_mine, frames.team_week_targets)
    for g in logs:
        g["target_pct"] = (None if g["dnp"] else
                           _round_or_none(targets_by_game.get((g["season"], g["week"])), 3))

    payload = {
        "header": header,
        # Second, next to the header rather than at the bottom with the feed,
        # because that is where it is used: "Questionable" changes a pick and
        # has to be reachable in one lookup from the component that draws the
        # name. See `player_status` for why it is None rather than a dict of
        # nulls for a defense, and why nothing here ever says "Healthy".
        "status": player_status(conn, player_id, frames.status_columns),
        "factors": factors_out,
        "summary": summary,
        "seasons": seasons,
        "game_log": logs,
        "outlook": outlook_out,
        # What the market prices his offence at, week by week. Team totals,
        # not player props -- see `_vegas`.
        "vegas": _with_futures(
            _vegas(schedules, header.get("team"), frames.draft_season),
            _player_futures(conn, player_id, header["name"], header["position"],
                            frames.draft_season)),
        "depth_chart": team_depth_chart(depth, header["team"], player_id),
        "schedule": weekly_difficulty(schedules, prior, header["team"],
                                      header["position"], rules),
        "similar": similar,
        "bio": bio,
        # Same three-aggregate contract as `season_summaries` above: the pool
        # and the features frame both come out of the cache priced under this
        # league's rules, so the band, the median and the decline count are
        # this league's points and not PPR's.
        "cohort": (None if not seasons else
                   comparable_cohort(frames.comp_pool, frames.season_features,
                                     player_id, bio["rookie_season"])),
        "oline": team_line_quality(frames.line_quality, frames.draft_season,
                                   header["team"], header["position"]),
        # Last, and a list -- empty for a player nobody wrote about, and for
        # every defense (pipeline/news.py runs no query for a DST: "Denver
        # Defense" as a search phrase returns whatever the newspaper wrote
        # about the Broncos). Never a placeholder item.
        "news": player_news(conn, player_id, frames.news_columns),
    }
    return _scrub(payload)
