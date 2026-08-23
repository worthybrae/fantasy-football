"""Monte Carlo draft simulator.

Roster value is the projected points of the best legal starting lineup, plus
an insurance term for the bench. Without that term every pick past the last
starting slot is worth exactly zero and the simulator's late rounds become
noise; with it, a backup behind a starter who is expected to miss games is
worth his own rate for the games he covers. That credit is earned once per
roster spot at the position -- not once per starter he backs up -- since one
bench player can't literally be in two places at once; see roster_value's
docstring for why. It is also earned only by a player the starting lineup did
NOT assign: at a multi-slot position the player just past the dedicated count
is usually filling a FLEX slot, and best_lineup_points has already paid him.

The rate that term multiplies is an *availability* rate (games played over
games possible, 0-100), not the board's `durability` column. The board's
column is a within-position percentile (see
scoring.factors.normalize_within_position), which averages 50 by
construction; reading it as a rate would model the median player at every
position as missing half the season.
"""
import warnings
from datetime import datetime, timezone
from itertools import zip_longest
from typing import NamedTuple

import numpy as np
import pandas as pd

from pipeline.db import read_table, write_table
from scoring import factors
from scoring.board import (FANTASY_POSITIONS, GAMES, POSITION_FLOOR,
                           _norm_name, adp_match_key, projections)
from scoring.config import CURRENT_SEASON, RECENCY_WEIGHTS
from scoring.draft_model import (COLD_START_PRIOR, EARLY_ROUNDS, FEATURE_NAMES,
                                 FFC_BLEND_WEIGHT, FLEX_POSITIONS, HYPE_SCALE,
                                 RUN_WINDOW, _ATTRIBUTE_DEFAULTS,
                                 _centre_within_position, _log_rank_features,
                                 _STAT_PROFILE_FEATURES, position_groups,
                                 pool_signal_features, roster_shape_features)
from scoring.player_history import assert_no_column_collision, attributes_as_of

# Re-exported, not redefined. `slots_left_at_pos` counts the FLEX slots a
# team has left, and that column is built on the FITTING side too, so the
# tuple moved to `draft_model` where both callers can read one copy of it.
# The name stays bound here because `scoring.gain` and the tests import it
# from this module.

# Availability (0-100) for a player with no weekly history to compute one
# from: rookies, kickers, and every DST. A realistic full-season availability
# rate, not the neutral 50 the board uses for a missing *percentile* -- 50
# here would mean "expected to miss half the season", which is a claim about
# the player, not an admission of ignorance.
DEFAULT_AVAILABILITY = 90.0


def _lineup_assignment(by_position: dict, settings):
    """Fill the starting lineup greedily and report what it consumed.

    `by_position` maps position -> that position's points, sorted descending.
    Returns `(points, assigned)`, where `assigned[pos]` is how many of that
    position's players the starting lineup actually used: its dedicated slots
    plus any FLEX slot its leftovers claimed.

    Greedy is optimal here: FLEX accepts a superset of no dedicated slot's
    eligibility and every other slot is single-position, so filling dedicated
    slots best-first and handing FLEX the leftovers can never be beaten.

    `assigned` is the half `best_lineup_points` throws away and `roster_value`
    cannot do without: the first genuinely benched player at a position is
    `values[assigned[pos]]`, not `values[starters[pos]]`. At RB and WR those
    two differ exactly when a FLEX slot is filled from that position, which
    is the normal case.
    """
    total = 0.0
    assigned = {}
    leftovers = []
    for pos, count in settings.starters.items():
        values = by_position.get(pos, [])
        total += sum(values[:count])
        assigned[pos] = min(count, len(values))
        if pos in FLEX_POSITIONS:
            leftovers.extend((points, pos) for points in values[count:])
    leftovers.sort(key=lambda item: item[0], reverse=True)
    for points, pos in leftovers[:settings.flex_slots]:
        total += points
        assigned[pos] += 1
    return total, assigned


def best_lineup_points(roster, settings) -> float:
    """Points of the best legal starting lineup.

    `roster` is a list of `(position, points)`. Unchanged public behaviour:
    the assignment detail its callers don't need stays inside
    `_lineup_assignment`.
    """
    by_position = {}
    for pos, points in roster:
        by_position.setdefault(pos, []).append(points)
    for values in by_position.values():
        values.sort(reverse=True)
    return _lineup_assignment(by_position, settings)[0]


def roster_value(roster, settings) -> float:
    """Starting-lineup points plus bench insurance.

    `roster` is a list of `(position, points, availability)`, where
    availability is a games-played rate on a 0-100 scale (see
    `DEFAULT_AVAILABILITY` and `_availability`) -- NOT the board's
    `durability` percentile.

    Each starter is expected to miss `(1 - availability/100) * 17` games. A
    single bench player is one body: he can cover for however many of his
    position's starters happen to be out, but he cannot be in two places at
    once, so the credit for him is earned once per roster, not once per
    starter he sits behind. Missed-game shares are summed across the
    starters at the position and clamped to 1.0 (a full season is the most
    coverage one backup can supply) before being multiplied by his own rate
    a single time. Crediting him separately behind every starter at a
    multi-slot position (e.g. twice at RB) would let one player's real
    season -- worth `backup_points` -- get counted as insurance more than
    once, which both overstates the position's value and, because Task 11's
    candidate search maximizes this exact quantity, would steer it toward
    overrating the third player at a thin, fragile position.

    The backup is the best player the starting lineup did not assign, which
    is `values[assigned[pos]]` and not `values[count]`: at RB and WR the
    player just past the dedicated count is normally filling a FLEX slot, and
    `_lineup_assignment` has already paid him. Reading him as the backup
    scored a roster with no bench at all above its own starting lineup.
    """
    by_position = {}
    for pos, points, availability in roster:
        by_position.setdefault(pos, []).append((points, availability))
    for values in by_position.values():
        values.sort(reverse=True)

    total, assigned = _lineup_assignment(
        {pos: [points for points, _ in values]
         for pos, values in by_position.items()}, settings)

    for pos, count in settings.starters.items():
        values = by_position.get(pos, [])
        used = assigned.get(pos, 0)
        if len(values) <= used:
            continue
        # `values` is sorted descending and `backup_points` is the first
        # entry the starting lineup left unassigned, so by construction it
        # can never exceed any starter's points -- no per-starter min(...)
        # cap is needed to keep this backup from being credited for more
        # than his own rate.
        backup_points = values[used][0]
        missed_total = 0.0
        for _, availability in values[:count]:
            # `availability or DEFAULT_AVAILABILITY` would be wrong here:
            # 0.0 is a real, legitimate value on this 0-100 scale (a player
            # who played none of his possible games) and Python's `or`
            # treats it as falsy, silently substituting the "unknown"
            # default and understating exactly the fragile-starter case this
            # term exists to cover.
            safe = (DEFAULT_AVAILABILITY if availability is None or pd.isna(availability)
                    else availability)
            missed_total += max(0.0, 1.0 - safe / 100.0)
        missed_total = min(1.0, missed_total)
        total += missed_total * backup_points
    return total


_POSITION_DUMMY_INDEX = {
    pos: FEATURE_NAMES.index(f"pos_{pos}")
    for pos in ("RB", "WR", "TE", "K", "DST")
}
_REACH = FEATURE_NAMES.index("reach")
_FALL = FEATURE_NAMES.index("fall")
_QB_EARLY = FEATURE_NAMES.index("qb_early")
_TE_EARLY = FEATURE_NAMES.index("te_early")
_NEED = FEATURE_NAMES.index("need")
_RUN = FEATURE_NAMES.index("run")
_AGE = FEATURE_NAMES.index("age")
_NO_TRACK_RECORD = FEATURE_NAMES.index("no_track_record")
_HYPE = FEATURE_NAMES.index("hype")
_TREND = FEATURE_NAMES.index("trend")
_HELD_AT_POS = FEATURE_NAMES.index("held_at_pos")
_FIRST_AT_POS = FEATURE_NAMES.index("first_at_pos")
_ROUNDS_SINCE_POS = FEATURE_NAMES.index("rounds_since_pos")
_FIRST_AT_POS_ROUND = FEATURE_NAMES.index("first_at_pos_round")
# (SimPool field name, column index) for the stat profile, derived from
# draft_model's own list rather than hand-written. A column added there with
# no matching `SimPool` field raises AttributeError the first time
# `_live_features` runs, which is the loud failure -- the quiet one is a
# hand-kept list that stops matching and serves a zero where the fit saw a
# number.
_STAT_PROFILE_INDEX = tuple((name, FEATURE_NAMES.index(name))
                            for name in _STAT_PROFILE_FEATURES)
_DROPOFF_AT_POS = FEATURE_NAMES.index("dropoff_at_pos")
_VOR = FEATURE_NAMES.index("vor")
_DURABILITY = FEATURE_NAMES.index("durability")
_PROJ_CHANGE = FEATURE_NAMES.index("proj_change")
_LAST_OF_TIER = FEATURE_NAMES.index("last_of_tier")
_SLOTS_LEFT_AT_POS = FEATURE_NAMES.index("slots_left_at_pos")


class SimPool(NamedTuple):
    player_id: np.ndarray
    norm: np.ndarray
    position: np.ndarray
    adp_rank: np.ndarray
    points: np.ndarray
    # Games-played rate on a 0-100 scale, NOT board["durability"] (a
    # within-position percentile). Named for what it is, so the two can never
    # be swapped by accident again.
    availability: np.ndarray
    # The board's own value over replacement, the second candidate source the
    # spec asks for ("the union of the top available by VOR and the top by
    # market rank").
    vor: np.ndarray
    # Task 4: the same player attributes the historical fit pool carries
    # (draft_model._enrich_pool), so `_live_features` can mirror
    # `feature_matrix` exactly. `market_rank` here is build_pool's own dense
    # 1..k re-ranking of the board (identical to `adp_rank` above -- both
    # fields are kept, named for what each is used for, so a future split
    # doesn't require touching every call site). It is NOT
    # `board["market_rank"]`, the raw multi-source consensus average
    # (scoring.market.add_market): that value is never re-ranked within
    # `board`, can exceed len(board), and reads on a different scale than
    # the dense rank draft_model's coefficients were fitted on -- see
    # build_pool's own comment for the inversion that mixing the two causes.
    market_rank: np.ndarray
    age: np.ndarray
    no_track_record: np.ndarray
    hype: np.ndarray
    trend: np.ndarray
    # Task 3: the stat profile, the other half of what `feature_matrix` reads
    # off a fitting pool. Named exactly as in `_STAT_PROFILE_FEATURES` --
    # `_live_features` looks them up by that name (see `_STAT_PROFILE_INDEX`),
    # so a rename on one side is an AttributeError rather than a wrong column.
    #
    # Defaulted to None, meaning "no value known", which `_live_features`
    # expands to an all-NaN column and `_centre_within_position` then reads as
    # neutral 0.0 -- exactly what a pool whose `attributes_as_of` join found
    # nothing already produces (see `_ATTRIBUTE_DEFAULTS`). The default exists
    # for hand-built fixtures only: `build_pool`, the one production
    # constructor, always populates all four, and a test asserts it does.
    usage: np.ndarray = None
    efficiency: np.ndarray = None
    played_share: np.ndarray = None
    peak_gap: np.ndarray = None
    # The remaining `_POOL_SIGNAL_FEATURES` inputs, straight off the board.
    # `vor` and `points` above are the other two; they were already carried
    # for candidate selection and the lineup value, and reading them here
    # rather than adding second copies is the point of naming SimPool's
    # fields after what they are.
    #
    # `durability` is the board's WITHIN-POSITION PERCENTILE, which is the
    # column `availability` above exists to not be -- see that field and
    # `_availability`. The model reads the percentile (a preference between
    # comparable players); `roster_value` reads the rate (an expectation
    # about games). Two different quantities, two fields, named apart so the
    # two can never be swapped by accident.
    #
    # Defaulted to None for hand-built fixtures, exactly like the stat
    # profile above: `_live_features` expands None to an all-NaN column,
    # which `pool_signal_features` reads as the neutral 0.0 -- the same
    # value a pool whose board join found nothing produces. `build_pool`,
    # the one production constructor, always populates all three.
    durability: np.ndarray = None
    proj_change: np.ndarray = None
    tier: np.ndarray = None


def snake_slots(teams: int, rounds: int) -> list:
    order = []
    for r in range(rounds):
        forward = list(range(1, teams + 1))
        order.extend(forward if r % 2 == 0 else forward[::-1])
    return order


def _availability(conn) -> dict:
    """Games-played rate per player_id, on a 0-100 scale.

    `board["durability"]` cannot be used for this. It is produced by
    `factors.normalize_within_position`, i.e. `rank(pct=True) * 100`, so it
    is a within-position percentile: it averages 50 by construction, which
    read as a rate means "the median player at every position misses half the
    season", and makes any two starters at a multi-slot position sum past
    `roster_value`'s `min(1.0, ...)` clamp so the backup always collects his
    full projected season. The actual rate is `durability_factor`'s
    `durability_raw` (games / possible), which the board drops before
    `_BOARD_COLUMNS`.

    Same `RECENCY_WEIGHTS` window `build_board` scores on, and for the same
    reason: `durability_raw` counts possible games from a player's first
    season in the data, so deep history would punish veterans for decade-old
    injuries.
    """
    weekly = read_table(conn, "weekly")
    if weekly.empty or "season" not in weekly.columns:
        return {}
    weekly = weekly[weekly["season"].isin(RECENCY_WEIGHTS)]
    if weekly.empty:
        return {}
    raw = factors.durability_factor(weekly)
    rate = raw["durability_raw"].clip(lower=0.0, upper=1.0) * 100.0
    return dict(zip(raw["player_id"], rate))


def _cheatsheet_ranks(conn, ranked: pd.DataFrame, season: int) -> pd.Series:
    """This season's ESPN preseason cheat-sheet rank per board row.

    All-NaN when the table is absent or does not cover `season`, which makes
    `build_pool` fall through to `ffc_rank` -- the same fallback ladder
    `draft_model._enrich_pool` uses, so the fit and the simulator degrade
    together rather than diverging.
    """
    cs = read_table(conn, "historic_espn_cs")
    if cs.empty or "season" not in cs.columns:
        return pd.Series(np.nan, index=ranked.index)
    cs = cs[cs["season"] == season].copy()
    if cs.empty:
        return pd.Series(np.nan, index=ranked.index)
    cs["key"] = [adp_match_key(name, position, team) for name, position, team
                 in zip(cs["cs_name"], cs["position"], cs["team"])]
    cs = cs.dropna(subset=["key"]).sort_values("cs_rank").drop_duplicates(
        "key", keep="first")
    return ranked["key"].map(cs.set_index("key")["cs_rank"])


def _board_signal(ranked: pd.DataFrame, name: str) -> np.ndarray:
    """One `_POOL_SIGNAL_FEATURES` input off the board, or all-NaN without it.

    `_BOARD_COLUMNS` carries all three, so this only ever falls back for a
    bare fixture board -- and it falls back to NaN, which
    `pool_signal_features` reads as the neutral 0.0 on both sides of the
    fit/serve line. The alternative, substituting some plausible number, would
    serve the model a value the fit never saw for a player nothing is known
    about.
    """
    if name in ranked.columns:
        return pd.to_numeric(ranked[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(ranked), np.nan)


def _sliced_or_unknown(values, available, n) -> np.ndarray:
    """A SimPool field sliced to the available candidates, or all-NaN.

    Only the fixture path hands None (see `SimPool`); `build_pool` fills every
    one of these. NaN is the neutral both `feature_matrix` and
    `pool_signal_features` already read as "nothing known", so a fixture that
    omits a column produces the same column a real pool produces for a player
    the board join missed.
    """
    return np.full(n, np.nan) if values is None else values[available]


def build_pool(conn, board: pd.DataFrame, settings) -> SimPool:
    # The board already carries this (build_board computes it to rank on);
    # recomputing it here would be a second, silently divergent copy. A bare
    # fixture board without the column still falls back to computing it.
    #
    # Assigned POSITIONALLY, never `.map()`ed on `player_id` -- the same
    # hazard, and the same fix, as build_board's own `uni["proj_points"] =
    # proj.to_numpy(...)` (see its comment): `_add_adp_only_players`
    # synthesizes `player_id` from the normalized name with no position in
    # the key, so one name at two positions in the ADP feed produces two
    # board rows sharing an id, and `.map()` against a duplicate-valued
    # index raises InvalidIndexError. build_board was fixed and this was
    # not, three lines down the same call chain -- so an ADP-feed name
    # collision still took `build_session` down and no live draft could
    # start. Both sides of the branch are one value per board row in board
    # row order (`projections` builds its Series by iterating
    # `board.iterrows()`), so positional assignment gives every row -- both
    # colliding ones included -- its own correct value.
    if "proj_points" in board.columns:
        points = board["proj_points"].to_numpy(dtype=float)
    else:
        points = projections(conn, board).to_numpy(dtype=float)
    ranked = board.copy()
    ranked["proj"] = points

    # `adp_rank`/`market_rank` (the SimPool fields, not the board column)
    # must land on the scale draft_model's reach/fall coefficients were
    # fitted on: a dense 1..k rank, matching historic_adp.adp_rank /
    # draft_model._enrich_pool's market_rank -- both dense per-season ranks,
    # not `board`'s raw, multi-source consensus average (scoring/market.py),
    # which is never re-ranked within `board` and can exceed len(board).
    #
    # SOURCE, not just scale: the ordering comes from this season's ESPN
    # preseason cheat sheet, falling back to `ffc_rank` (Fantasy Football
    # Calculator) and then to the consensus -- the same ladder, in the same
    # order, that `draft_model._enrich_pool` fits on. See its docstring for
    # why the cheat sheet and not ESPN's API. Ranking on
    # `board["market_rank"]`, the five-source consensus, would put the fit
    # and the simulator on two different orderings of the same players, so a
    # `reach` coefficient learned against one would be applied to a board
    # that disagrees with it. A board built without either column (a bare
    # fixture) falls back to the consensus, which keeps such fixtures
    # working without silently changing what production reads.
    # Folding a fillna sentinel into the same sort_values as the real
    # market_rank values is not safe: a real rank of 500 on a 5-row board
    # (the pinned regression fixture below) can exceed even a
    # len(board) + 1 sentinel, scrambling a genuinely-ranked player behind
    # the unranked ones. Nor is returning the raw value for players who do
    # carry one, and only filling the gaps: it leaks the board's wider,
    # unbounded scale into `market_rank`, which distorts reach/fall for
    # every ranked player, not just the unranked ones -- and can still let
    # an unranked player's *filled* rank land ahead of a real one, since a
    # dense pool position can be smaller than a real rank far past
    # len(board). Ranking densely among only the players who carry a real
    # market_rank, then USING that dense rank (not the raw value) as
    # `market_rank` itself, avoids both: "rank k" means the same thing on
    # the board, in `historic_adp`, and in the fit, an unranked player can
    # never end up ranked ahead of a real one, and no fillna is needed at
    # all -- there is no longer a NaN to fill. Unranked players are
    # appended after the last real rank, not interleaved, so they stay
    # takeable but sit far down the board (an opponent can still draft
    # them).
    # The season being DRAFTED, which is always the current one -- not
    # `settings.season`. `league.load` returns the newest *imported* season's
    # rules (2025 for a league whose last completed draft was 2025), which is
    # the right source for teams/starters/scoring and the wrong answer to
    # "which season's board is this". Reading `settings.season or
    # CURRENT_SEASON` meant the fallback only ever fired for a league with no
    # import history at all; with any history it silently ranked this year's
    # board on last year's cheat sheet and read player attributes one season
    # stale (`attributes_as_of` looks strictly before `season`, so a 2026
    # draft never saw 2025's production).
    season = CURRENT_SEASON
    team_col = (ranked["team"] if "team" in ranked.columns
                else pd.Series([None] * len(ranked), index=ranked.index))
    ranked = ranked.assign(key=[
        adp_match_key(name, position, team) for name, position, team
        in zip(ranked["name"], ranked["position"], team_col)])
    ranked["_cs_rank"] = _cheatsheet_ranks(conn, ranked, season)

    rank_col = "ffc_rank" if "ffc_rank" in ranked.columns else "market_rank"
    # The cheat sheet leads; `rank_col` both fills the players it never
    # ranked (it stops at 300) and breaks ties within it.
    has_rank = ranked["_cs_rank"].notna() | ranked[rank_col].notna()
    known = ranked[has_rank].sort_values(["_cs_rank", rank_col], kind="stable",
                                         na_position="last")
    unknown = ranked[~has_rank]
    ranked = pd.concat([known, unknown], ignore_index=True)

    # The same cheat-sheet/FFC blend `draft_model._enrich_pool` fits against,
    # at the same weight, because a reach coefficient learned on one board
    # means something else applied to another. That file carries the
    # measurement; this one must not drift from it, which is why the weight
    # is imported rather than repeated.
    #
    # Only the players `rank_col` actually ranks take part: the tail the cheat
    # sheet never reached and FFC never priced has no second opinion to blend,
    # and giving it one by position in this list would be inventing a market.
    # Those keep the cheat-sheet-led order they already have.
    cs_dense = pd.Series(np.arange(1, len(ranked) + 1, dtype=float),
                         index=ranked.index)
    ffc = pd.to_numeric(ranked.get(rank_col), errors="coerce")
    both = ffc.notna()
    if both.any():
        ffc_dense = ffc[both].rank(method="first")
        blended = cs_dense.copy()
        blended[both] = ((1.0 - FFC_BLEND_WEIGHT) * cs_dense[both]
                         + FFC_BLEND_WEIGHT * ffc_dense)
        ranked = ranked.loc[blended.sort_values(kind="stable").index].reset_index(drop=True)
    dense_rank = np.arange(1, len(ranked) + 1, dtype=float)
    availability = ranked["player_id"].map(_availability(conn))
    # `vor` is the board's headline ranking and the spec's second candidate
    # source. A board built without it (a bare fixture) falls back to the
    # projection, which keeps the second source meaningful rather than
    # collapsing it onto market order.
    vor_col = ranked["vor"] if "vor" in ranked.columns else ranked["proj"]

    # Task 4: the same player attributes the historical fit pool carries
    # (draft_model._enrich_pool), joined the same way -- on adp_match_key,
    # read strictly from seasons before the draft season, so nothing here is
    # hindsight. A player the join misses (no season of prior stats under
    # this key) gets the same neutral defaults and no_track_record=True.
    # `key` is already on `ranked`: the cheat-sheet join above needs it too,
    # so it is built once, before the sort.
    attrs = attributes_as_of(conn, season)
    if not attrs.empty:
        # Same collision guard as _enrich_pool: a non-DST key carries no
        # team, so two different past players can share (position,
        # normalized name). Dropping both sides of a collision falls
        # through to "unknown" rather than guessing whose numbers apply.
        attrs = attrs.drop_duplicates("key", keep=False)
    if attrs.empty:
        for col, default in _ATTRIBUTE_DEFAULTS.items():
            ranked[col] = default
    else:
        assert_no_column_collision(ranked)
        ranked = ranked.merge(attrs, on="key", how="left")
        ranked["no_track_record"] = ranked["no_track_record"].fillna(True).astype(bool)
        for col, default in _ATTRIBUTE_DEFAULTS.items():
            if col not in ("no_track_record", "prod_rank"):
                ranked[col] = ranked[col].fillna(default)
    # Positive means the market is ahead of what the player has actually
    # done. NaN when he has no production to rank, which feature_matrix
    # (and _live_features) read as neutral rather than as zero.
    #
    # `dense_rank`, not the raw board `market_rank`: `_enrich_pool` computes
    # this as `prod_rank - market_rank` against its own dense per-season
    # rank, so subtracting anything else here would fit and apply `hype` on
    # two different scales. The left merge above cannot reorder or duplicate
    # rows (`attrs` is deduped on `key`), so `dense_rank` still lines up
    # positionally with `ranked`.
    ranked["hype"] = ranked["prod_rank"] - dense_rank

    return SimPool(
        player_id=ranked["player_id"].to_numpy(),
        norm=ranked["name"].map(_norm_name).to_numpy(),
        position=ranked["position"].to_numpy(),
        adp_rank=dense_rank,
        points=ranked["proj"].to_numpy(dtype=float),
        availability=availability.fillna(DEFAULT_AVAILABILITY).to_numpy(dtype=float),
        vor=pd.to_numeric(vor_col, errors="coerce").fillna(-np.inf).to_numpy(dtype=float),
        # Same dense rank as `adp_rank` above -- see the comment at the top
        # of this function for why `board["market_rank"]`'s raw value must
        # not reach here.
        market_rank=dense_rank,
        age=ranked["age"].to_numpy(dtype=float),
        no_track_record=ranked["no_track_record"].to_numpy(dtype=bool),
        hype=ranked["hype"].to_numpy(dtype=float),
        trend=ranked["trend"].to_numpy(dtype=float),
        # The stat profile, filled from the same `attributes_as_of` merge
        # above. `_ATTRIBUTE_DEFAULTS` already put NaN on every row the join
        # missed, so an unknown player reaches `_live_features` in the same
        # state he reaches `feature_matrix` in.
        **{name: ranked[name].to_numpy(dtype=float)
           for name in _STAT_PROFILE_FEATURES},
        # The three board columns the model could not see until Tier 1.
        # Assigned positionally like everything else here (see the top of
        # this function for why `.map()` on `player_id` is unsafe), and NOT
        # re-derived: `durability`, `proj_change` and `tier` are computed by
        # `build_board` and a second computation of any of them here would be
        # a copy that drifts.
        durability=_board_signal(ranked, "durability"),
        proj_change=_board_signal(ranked, "proj_change"),
        tier=_board_signal(ranked, "tier"))


def _live_features(pool, available, overall_pick, roster, recent, settings,
                   last_pick_at_pos=None):
    """Feature matrix for the currently available players, mirroring
    draft_model.feature_matrix exactly -- the fitted coefficients only mean
    anything against the same feature definitions they were fitted on.

    This function is what the model is SERVED from; `feature_matrix` is what
    it is FITTED on. Where the two could drift they now call one shared
    implementation (`draft_model.roster_shape_features`,
    `pool_signal_features`, `_centre_within_position`) rather than each
    writing the arithmetic out, and `tests/test_draft_sim.py` asserts the two
    agree end to end anyway.
    A disagreement here is silent: the shapes still line up, nothing raises,
    and every live prediction is simply computed from a vector the fit never
    saw.

    `last_pick_at_pos` is `{position: overall_pick}` for THIS team's most
    recent pick at each position, the same thing
    `draft_model.PickObservation.last_pick_at_pos` carries. None means the
    team has taken nothing yet, matching that field's own default. Both
    production callers (`_run_draft`, `survival`) thread the real thing;
    `_seed_rosters` reconstructs it for a resumed draft.
    """
    positions = pool.position[available]
    ranks = pool.market_rank[available]
    teams = max(settings.teams, 1)
    n = len(positions)
    X = np.zeros((n, len(FEATURE_NAMES)))

    X[:, _REACH], X[:, _FALL] = _log_rank_features(ranks, overall_pick)
    for pos, col in _POSITION_DUMMY_INDEX.items():
        X[:, col] = (positions == pos)

    round_no = (overall_pick - 1) // teams + 1
    early = 1.0 if round_no <= EARLY_ROUNDS else 0.0
    X[:, _QB_EARLY] = (positions == "QB") * early
    X[:, _TE_EARLY] = (positions == "TE") * early

    # `need` and `run` are grouped by position (the fixed six-value
    # vocabulary, not `np.unique(positions)` -- see _legal_mask's docstring:
    # np.unique's sort showed up as its own measurable cost once the
    # per-element comprehension it was replacing was gone) rather than
    # computed with a Python-level comprehension over every element. `X`
    # starts zeroed, so a position whose need/run value is 0.0 needs no
    # write.
    starters = settings.starters
    window = recent[:RUN_WINDOW]
    for pos in FANTASY_POSITIONS:
        mask = positions == pos
        if roster.get(pos, 0) < starters.get(pos, 0):
            X[mask, _NEED] = 1.0
        run_share = window.count(pos) / RUN_WINDOW
        if run_share:
            X[mask, _RUN] = run_share

    # One grouping pass shared by `age`, the four stat-profile columns and
    # the roster-shape block below. Regrouping per column instead costs 87us
    # of a 158us call on a 250-candidate pool, on a function that runs once
    # per pick per rollout. See `draft_model.position_groups`.
    groups = position_groups(positions)

    X[:, _AGE] = _centre_within_position(pool.age[available], positions, groups)
    X[:, _NO_TRACK_RECORD] = pool.no_track_record[available].astype(float)
    X[:, _HYPE] = np.nan_to_num(pool.hype[available], nan=0.0) / HYPE_SCALE
    X[:, _TREND] = pool.trend[available]

    # The roster-shape block, from the SAME function `feature_matrix` calls.
    (X[:, _HELD_AT_POS], X[:, _FIRST_AT_POS], X[:, _ROUNDS_SINCE_POS],
     X[:, _FIRST_AT_POS_ROUND]) = roster_shape_features(
        positions, roster, last_pick_at_pos or {}, round_no, teams,
        max(settings.rounds, 1), groups)

    # The stat profile. `values is None` is a fixture-only path (see
    # `SimPool`); an all-NaN column centres to 0.0, which is the same neutral
    # a missed attribute join already produces on both sides.
    for name, col in _STAT_PROFILE_INDEX:
        X[:, col] = _centre_within_position(
            _sliced_or_unknown(getattr(pool, name), available, n),
            positions, groups)

    # The pool-signal block, from the SAME function `feature_matrix` calls.
    # `dropoff_at_pos` is the only column in the model that summarizes the
    # rest of the choice set, so it is the one that HAS to be sliced by
    # `available` rather than read off the whole pool: the drop from a player
    # to the next man at his position is a different number once the next man
    # is gone, and that is the whole signal.
    (X[:, _DROPOFF_AT_POS], X[:, _VOR], X[:, _DURABILITY], X[:, _PROJ_CHANGE],
     X[:, _LAST_OF_TIER], X[:, _SLOTS_LEFT_AT_POS]) = pool_signal_features(
        positions, pool.points[available], pool.vor[available],
        _sliced_or_unknown(pool.durability, available, n),
        _sliced_or_unknown(pool.proj_change, available, n),
        _sliced_or_unknown(pool.tier, available, n),
        roster, settings.starters, settings.flex_slots, groups)
    return X


# Positions a FLEX slot never meaningfully absorbs, so a second backup can
# never enter the starting lineup and is pure dead weight. QB is literally
# flex-ineligible; TE is eligible on paper and essentially never started
# there in practice once the dedicated slot is filled.
_SHALLOW_POSITIONS = ("QB", "TE")


def _roster_cap(settings) -> dict:
    """Most of each position anyone will carry. Learned coefficients cannot
    express a hard ceiling, so it is imposed as a mask instead.

    Two depths. QB and TE carry `starters + 1` -- the starter and one backup
    -- because nothing past that can ever reach the lineup (see
    _SHALLOW_POSITIONS). RB and WR carry `starters + flex_slots + 2`: they
    are what the FLEX slots actually get filled with AND what a bench is
    made of, and they churn hardest on byes and injuries.

    Written relative to `starters`/`flex_slots` rather than as flat numbers
    so it still holds where the league changes shape: a superflex or 2-QB
    league starts two quarterbacks and correctly allows three.

    TOTAL CAPACITY MUST EXCEED THE NUMBER OF ROUNDS, and that is not a
    stylistic preference -- it is the invariant that keeps the caps meaning
    anything at all. `_run_draft` and `_greedy_choice` both fall back to
    `available[0]` when no cap-legal player remains, and that fallback
    ignores caps completely. So a cap table summing to less than `rounds`
    does not produce a short roster, it produces a silent cap violation on
    the picks past the total, with nothing reporting it. That is exactly
    what happened when QB and TE were first tightened to `starters + 1`
    while RB and WR were left at `starters + 2`: the six caps summed to 14
    against 15 rounds, and 480 violations appeared across 60 simulated
    drafts -- WR reaching 5 against a cap of 4, QB reaching 3 against 2.
    `_assert_capacity` below makes the invariant explicit rather than
    trusting the arithmetic to stay true as the rule is edited.
    """
    caps = {}
    for pos, n in settings.starters.items():
        if pos in ("K", "DST"):
            caps[pos] = 1
        elif pos in _SHALLOW_POSITIONS:
            caps[pos] = n + 1
        else:
            caps[pos] = n + settings.flex_slots + 2
    _assert_capacity(caps, settings)
    return caps


def _assert_capacity(caps, settings) -> None:
    """Guard the one invariant `_roster_cap` cannot express in its own
    arithmetic: there must be more cap-legal roster room than there are
    picks to make. Raising here is right because every alternative is worse
    -- the caller's fallback would quietly draft past the ceiling, and a
    warning would be swallowed by the same daemon threads that swallow
    everything else in a live draft."""
    total = sum(caps.values())
    if total < settings.rounds:
        raise ValueError(
            f"roster caps allow only {total} players across "
            f"{len(caps)} positions but the draft runs {settings.rounds} "
            f"rounds: the last {settings.rounds - total} picks would bypass "
            f"the caps entirely via the no-legal-player fallback. Caps: "
            f"{caps}")


def _turns_left(slots, rounds: int) -> list:
    """For every offset in the snake, how many picks the slot on the clock at
    that offset still has, COUNTING the pick at that offset itself.

    Precomputed once per simulation rather than counted per pick: every slot
    appears exactly `rounds` times in `slots`, so this is one pass, and the
    obvious `sum(1 for s in slots[offset:] if s == slot)` inside the rollout
    loop would be O(picks) per pick, i.e. O(120 x 120 x rollouts).
    """
    seen, out = {}, []
    for slot in slots:
        seen[slot] = seen.get(slot, 0) + 1
        out.append(rounds - seen[slot] + 1)
    return out


def must_fill_positions(settings, counts, turns_left: int) -> tuple:
    """The starter slots a roster with `turns_left` picks left can no longer
    defer -- empty whenever it still has slack.

    One definition of "now is the time to fill it", used from two places
    that must not disagree about it: `_must_fill_mask`, which forces a
    simulated opponent's pick onto those slots, and `gain.need_kind`, which
    stops calling a deferrable slot a full starter need. If they drifted,
    the ranking would recommend a kicker on a different round from the one
    the simulator expects kickers to go in.

    The line is `open starter slots >= remaining picks`: at that point every
    pick left has to become one of them. `turns_left` counts the pick on the
    clock (see `_turns_left`). Measured over 8 seeds of the real board, the
    earliest pick at which it binds for anybody is 106 -- round 14 of 15 --
    so it is inert for the whole draft up to there.
    """
    needed = tuple(pos for pos, n in settings.starters.items()
                   if counts.get(pos, 0) < n)
    if not needed:
        return ()
    open_slots = sum(n - counts.get(pos, 0)
                     for pos, n in settings.starters.items()
                     if counts.get(pos, 0) < n)
    return needed if open_slots >= turns_left else ()


def _must_fill_mask(pool, indices, counts, settings, turns_left: int):
    """Which of `indices` fill a starter slot this roster can no longer defer,
    or None when the constraint does not bind (which is nearly always).

    A ROSTER RULE, NOT A CORRECTION TO ANY PARTICULAR FIT. This is the floor
    that `_roster_cap` is the ceiling of, and it is here for the same stated
    reason: "learned coefficients cannot express a hard ceiling, so it is
    imposed as a mask instead." A manager with two picks left and an empty
    kicker and defense slot takes a kicker and a defense. That is not a
    tendency to be fitted, it is what the roster rules leave them. That
    argument is the whole justification and it holds for every fit, good or
    bad -- KEEP THIS EVEN AFTER THE DEFENSE COEFFICIENT IS RE-FITTED. It is
    also one half of a shared definition: `must_fill_positions` is called from
    here and from `gain.need_kind`, which are required not to disagree about
    when a slot stops being deferrable, so deleting this side would
    desynchronise the ranking from the simulator.

    WHAT WENT WRONG WITHOUT IT, measured on the fit that stood until
    2026-08-23. `COLD_START_PRIOR`'s `pos_DST` was -11.14, and every
    per-manager fit on this deployment's own database shrinks to that same
    value. (The cold-start half of that has since been refitted on the mock
    corpus and now reads +0.21 -- see the end of this docstring for what that
    changes and what it does not.) Measured on the
    real 249-player pool (8 teams, 15 rounds), `_run_draft` run out to all 120
    picks, 8 seeds, with the real fitted betas and again with cold-start:
    EXACTLY ONE defense was drafted per simulated draft, every time at pick
    113 -- and pick 113 is my own slot's round-15 pick, taken by
    `_greedy_choice`. No modelled opponent drafted a defense at any pick of
    any of the sixteen simulations, so 7 of the 8 simulated teams finished the
    draft with an empty DST starter slot.

    Downstream that made every defense unpickable by construction: `survival`
    runs these same opponents, so DST availability came back at 1.000 at pick
    0, 21, 50, 99 and 110 alike; `gain.expected_best_next` then equals the
    position's own leader, and `gain_now` was exactly 0.0000 for every
    defense at every point of the draft. The tool could never surface one,
    and an owner following it literally reaches the last round without a
    defense.

    WHY THAT FAILURE IS NOT A FACT ABOUT DRAFTERS, and why it is expected to
    go away. The fit that produced `pos_DST` ran on a `draft_picks` table
    holding 712 rows over six seasons with NOT ONE DST among them -- the
    positions present were WR/RB/TE/QB/K only, every season missing exactly
    eight late picks (2020 [60, 68, 72, 90, 102, 113, 126, 127], 2025
    [98, 99, 100, 104, 106, 107, 108, 112], and so on) while all 8 kickers
    were recorded every year. Those gaps were the defenses: the importer's
    `_is_real_pick` tested `playerId > 0`, and ESPN's D/ST ids are negative by
    design, so it threw every one away. The fit saw defenses in every choice
    set, saw one chosen zero times, and drove `pos_DST` as negative as the
    ridge allowed -- a correct fit to data in which the event is
    unobservable, not a measurement of how anybody drafts.

    THAT IMPORT BUG IS FIXED (9b90610), AND HALF THE COEFFICIENT NOW IS TOO.
    `make fit-prior` refitted the cold-start prior on 26 real ESPN mock
    drafts, where 207 defenses actually get drafted, and `pos_DST` came out
    at +0.21 -- the predicted sign flip, at a magnitude an order below the
    +2.81 a reconstruction had guessed
    (docs/superpowers/findings/2026-08-23-mock-corpus-features.md). So a
    COLD-START league's opponents now take defenses on their own and the
    failure above is no longer live for them.

    IT IS STILL LIVE FOR THIS DEPLOYMENT'S OWN LEAGUE, whose per-manager fits
    come from `make fit-managers` against `draft_picks` and have not been
    re-imported or re-fitted. That needs `make espn-import && make
    fit-managers`.

    WHERE THE COEFFICIENT IS FIXED THIS MASK BECOMES INERT RATHER THAN WRONG
    -- a constraint that binds no earlier than pick 106 simply stops being
    reached. Do not read its inertness as a reason to delete it: the roster
    floor at the top of this docstring is why it exists, and that never
    depended on the coefficient.

    WHAT IT FIXES, measured the same way. All 8 defenses are drafted in all
    16 simulations, none of the 8 teams finishes short at any starter
    position, and every first kicker still goes at pick 68-87 exactly as
    before. On the live side, at the owner's own final pick with an empty DST
    slot and the same board state either way: the best defense went from rank
    3, gain_now +0.000, survival 1.000 to rank 1, gain_now +4.977, survival
    0.235. `survival` costs the same (1.45 / 1.01 / 1.23 / 1.07 / 0.21s at
    picks_made 0 / 21 / 50 / 99 / 110, against 1.45 / 0.99 / 1.22 / 1.11 /
    0.19s before).

    WHAT THIS DOES NOT DO. It does not touch the rounds where the fit is
    good. The constraint binds only when a roster's remaining picks have run
    down to its unfilled starter slots, which cannot happen early: at pick 1
    every roster has 8 starter slots and 15 picks. Measured over the same 8
    seeds, the earliest pick at which it binds for anybody is 106 -- round 14
    of 15. At picks_made=99 the live ranking is identical to the digit before
    and after. It adds no preference between positions, changes no
    probability, and is not consulted at all while a roster still has slack
    -- the softmax over `X @ beta` decides every one of those picks exactly
    as before.

    AND IT IS DELIBERATELY CONSERVATIVE, which is the honest limit of what
    can be claimed here. It puts every simulated defense in round 15. The
    real drafts it should be reproducing took theirs between picks 60 and
    127, mean 105.6 -- that is INFERRED, not read: the pick numbers come from
    the eight gaps each season leaves in `draft_picks`, which are the eight
    defenses by elimination (every other position is fully accounted for, and
    2025's 112 rows are 120 minus 8). So a defense's urgency in rounds 9-14
    is still understated and `gain_now` for one is still 0.00 there. The way
    to close that is a re-import and a re-fit, which will let the fit read the
    real timing off real picks instead of a mask asserting it. Widening this
    mask to cover rounds 9-14 by hand would be inventing exactly the
    measurement that is now one `make` target away.

    `turns_left` is this slot's remaining picks including the current one
    (see `_turns_left`). Returns None, never an empty mask, when the
    constraint does not bind or when no player at a still-needed position is
    in `indices` at all -- so a caller can apply the result unconditionally
    and can never be left with nothing to pick.

    WHO IT APPLIES TO. In `_run_draft`, opponents only: my own slot picks
    through `_greedy_choice`, which values a roster rather than sampling a
    fit, and is measurably not the problem -- it took the defense at pick 113
    in all 16 simulations above, and its policy is not something this
    correction should be quietly changing. In `survival` it applies to every
    pick between now and my horizon, mine included, because that loop has no
    my-slot branch and does not need one: it is counting who comes off the
    board, and an intervening pick of my own removes a player exactly like
    anybody else's.
    """
    needed = must_fill_positions(settings, counts, turns_left)
    if not needed:
        return None
    # Same fixed-vocabulary OR as `_legal_mask`, and for the same reason:
    # np.unique's sort and a per-player Python comprehension both showed up
    # as measurable costs inside the rollout loop (see its docstring).
    positions = pool.position[indices]
    mask = np.zeros(len(indices), dtype=bool)
    for pos in needed:
        mask |= positions == pos
    return mask if mask.any() else None


# Heuristic cutoff chosen for speed, not a proven bound -- marginal
# roster-value gain and raw points don't have to rank identically. Scanning
# every legal remaining player would make every rollout O(pool) roster
# valuations, which at ~500 players x 15 picks x thousands of rollouts is
# the difference between seconds and hours, so only the top GREEDY_CANDIDATES
# by points are evaluated.
GREEDY_CANDIDATES = 40


def _legal_mask(pool, indices, counts, caps) -> np.ndarray:
    """Which of `indices` sit at a position `counts` (a roster's position ->
    count-so-far dict) has not yet hit its cap.

    Vectorized over the fixed, six-value position vocabulary (QB, RB, WR,
    TE, K, DST -- `scoring.board.FANTASY_POSITIONS`) rather than a per-player
    Python loop, and iterating that fixed set directly rather than computing
    `np.unique(indices' positions)` to discover it. Task 10's review flagged
    the original
    `[counts.get(pool.position[i], 0) < caps.get(pool.position[i], 99) for i in indices]`
    as a per-pick O(available) Python comprehension, multiplied by candidates
    x rollouts x picks-per-rollout in Task 11's search_pick. Profiling a
    realistic call (12 candidates x 300 rollouts, a 400-player pool, a full
    15-round 8-team draft -- this task's own numbers) confirmed it: roughly
    40% of total wall time, from re-indexing `pool.position` and calling two
    dict.get()s per element, up to ~500 times, on every single pick of every
    rollout. Grouping by position turns that into a handful of vectorized
    numpy comparisons instead -- same inputs, same output, just not one
    Python-level iteration per player. Swapping `np.unique` for the fixed
    vocabulary avoids `unique`'s own sort, which showed up as a measurable
    cost in its own right once the per-element comprehension it replaced was
    gone. (`_live_features`'s `need`/`run` columns had the identical shape of
    cost and got the same two fixes.)
    """
    positions = pool.position[indices]
    mask = np.ones(len(indices), dtype=bool)
    for pos in FANTASY_POSITIONS:
        if counts.get(pos, 0) >= caps.get(pos, 99):
            mask[positions == pos] = False
    return mask


def _replacement_points(pool, settings) -> dict:
    """Projected points of the last starter-caliber player at each position.

    `LeagueSettings.replacement_ranks` already derives those ranks from the
    league's roster shape, and the board's VOR column already uses them --
    the simulator simply was not. Without this a one-ply greedy compares a
    380-point quarterback against a 300-point running back and takes the
    quarterback, ignoring that the next quarterback available is worth 300
    while the next running back is worth 150.
    """
    ranks = settings.replacement_ranks
    out = {}
    for pos, rank in ranks.items():
        points = np.sort(pool.points[pool.position == pos])[::-1]
        if len(points) == 0:
            out[pos] = 0.0
        else:
            out[pos] = float(points[min(rank, len(points)) - 1])
    return out


def _picks_until_my_next_turn(slots, offset, my_slot):
    """How many other teams pick between this pick of mine and my next one.

    None when this is my last pick of the draft, which is a different thing
    from zero: at my last pick nothing I pass over can be taken from me, so
    there is no opportunity cost to price, and `_greedy_choice` falls back to
    comparing raw roster value.
    """
    for j in range(offset + 1, len(slots)):
        if slots[j] == my_slot:
            return j - offset - 1
    return None


def _opportunity_gap(slots, offset, my_slot):
    """The opportunity-cost horizon for this pick: how many other teams pick
    before the board can next be taken from me.

    At the snake turn a team picks twice in a row, so
    `_picks_until_my_next_turn` is a truthful 0 -- nothing is taken from me
    between those two picks. Feeding that 0 to `_greedy_choice` is the
    problem: `_next_turn_survivors(available, 0)` returns the ENTIRE pool, so
    every position's best available player is his own forgone alternative and
    scores `delta(i) - delta(i) == 0.0` exactly. Every position leader ties at
    zero, the strict `>` keeps whichever came first, and the shortlist is
    sorted by RAW POINTS -- so the turn pick silently went to the highest
    projected scorer on the board regardless of position. That is a
    quarterback essentially always (top QB 369.7 vs top RB 364.9 on today's
    board), and it measured: slot 8 took a QB at pick 8, again at 24, again
    at 40, drafted no running back until round 6, and finished 240 points of
    roster value behind slot 5 -- last of the eight slots.

    Back-to-back picks are one combined selection of two players, and what I
    decline across the PAIR is what the following gap exposes. So skip over
    consecutive picks of mine and report the first real gap; `None` still
    means no next turn at all, where there is genuinely nothing to forgo.
    """
    for j in range(offset + 1, len(slots)):
        if slots[j] == my_slot:
            gap = j - offset - 1
            if gap > 0:
                return gap
            offset = j          # consecutive pick: look through to the next
    return None


def _next_turn_survivors(pool, available, gap):
    """Who is plausibly still on the board when I pick again.

    The next `gap` picks are approximated as the top `gap` available players
    by market rank. That is not what will happen -- managers reach and let
    players slide, which is the whole subject of `draft_model` -- but it is
    what the room believes will happen, it costs one argsort, and it is the
    same reasoning a human drafter does out loud ("those six will be gone").
    Running the real pick model here instead would mean simulating the
    intervening picks inside every candidate evaluation of every rollout.
    """
    order = available[np.argsort(pool.market_rank[available], kind="stable")]
    return order[gap:]


def _greedy_choice(pool, available, roster, settings, caps, gap=None,
                   turns_left=None):
    """My in-rollout policy: the available, cap-legal player whose roster
    value most exceeds what I could get at his position when I pick again.

    `gap` is how many other teams pick before the board can next be taken
    from me (`_opportunity_gap`). Given it, each candidate is scored against
    the best player at his own position expected to survive that long, which
    is the opportunity cost of taking him now. Without it -- my last pick of
    the draft -- there is no next turn to forgo, and candidates are compared
    on raw roster value.

    This replaced a static replacement level (`_replacement_points`, still
    used by the board's VOR column) and the difference is the whole point.
    Replacement level is a season-long constant: it prices a quarterback
    against the 9th-best QB *of the original board* however deep the draft
    has gone. That is right on average and wrong exactly when it matters. At
    pick 13 of this league it valued Josh Allen at +76 over replacement and
    the best available running back at +73, and took Allen -- correctly, on
    its own terms, and absurdly on the board's. What it could not see is that
    seven picks later the quarterback position would be barely worse while
    the running back position would have fallen off a cliff. Opportunity cost
    is a function of when I pick next, not of the league's roster shape.

    One-ply greedy is still what makes a rollout cheap enough to run
    thousands of times; the search in Task 11 is what looks further ahead.
    The subtraction happens on the final scalar, never on the points that go
    into `roster_value`: that function is always called with real points, for
    the candidate and for every player already rostered alike, so its sort,
    FLEX assignment and bench-insurance logic never compare an adjusted
    number against a real one. Only the per-candidate delta is shifted, and
    only to rank candidates against each other -- the roster this builds, and
    the value `roster_value` later reports for it, stay priced in real
    points throughout.

    Legality is filtered before the shortlist is built, not inside the loop
    over it. The shortlist is only the top GREEDY_CANDIDATES by raw points,
    and late in a draft those top point-scorers can all sit at a position
    I'm already capped on (e.g. QB once I have 3). Filtering inside the loop
    (skip-and-continue) left `best_idx` unset in exactly that case and fell
    through to a fallback that picked the single best remaining player
    ignoring caps entirely -- silently drafting a 4th QB. Filtering first
    means the shortlist is never spent on players I cannot legally take.
    """
    current = [(pool.position[i], pool.points[i], pool.availability[i])
               for i in roster["indices"]]
    base = roster_value(current, settings)
    legal = available[_legal_mask(pool, available, roster["counts"], caps)]
    if len(legal) == 0:
        # No cap-legal player remains anywhere in the pool -- genuinely
        # unavoidable, not a shortlist artifact -- so caps no longer apply.
        return available[0] if len(available) else None
    # The same roster floor every simulated OPPONENT gets (see
    # `_must_fill_mask`, whose docstring calls itself "the floor that
    # _roster_cap is the ceiling of"). It was applied only to opponents, and
    # my own policy got away without it purely because the caps used to be
    # tight enough to force diversification as a side effect: at RB/WR caps
    # of 4 the late rounds had nowhere else to go. Loosening those caps to
    # their honest depth exposed the gap -- my roster finished with an empty
    # defense slot in 60 of 60 drafts at slots 2, 5 and 8, costing 90-130
    # points of roster value. A ceiling was doing a floor's job.
    if turns_left is not None:
        must_fill = _must_fill_mask(pool, legal, roster["counts"], settings,
                                    turns_left)
        if must_fill is not None and len(legal[must_fill]):
            legal = legal[must_fill]
    shortlist = legal[np.argsort(-pool.points[legal])][:GREEDY_CANDIDATES]

    def delta(i):
        return roster_value(
            current + [(pool.position[i], pool.points[i], pool.availability[i])],
            settings) - base

    # What this position is worth to me if I wait. Computed per position that
    # actually appears on the shortlist, in the same units as `delta` -- a
    # roster-value increment, not raw points -- so the subtraction compares
    # like with like and a position I have no room for nets out near zero on
    # both sides instead of being penalized twice.
    forgone = {}
    if gap is not None:
        survivors = _next_turn_survivors(pool, available, gap)
        for pos in {pool.position[i] for i in shortlist}:
            at_pos = survivors[pool.position[survivors] == pos]
            forgone[pos] = (delta(at_pos[np.argmax(pool.points[at_pos])])
                            if len(at_pos) else 0.0)

    best_idx, best_gain = None, -np.inf
    for i in shortlist:
        gain = delta(i) - forgone.get(pool.position[i], 0.0)
        if gain > best_gain:
            best_idx, best_gain = i, gain
    return best_idx if best_idx is not None else legal[0]


def _seed_rosters(pool, settings, taken_order):
    """Replay the picks already made and hand each one to the slot that was
    on the clock for it.

    `taken_order` is one entry per pick already made, in pick order: a pool
    index, or None for a pick whose player is not in the pool (drafted, then
    dropped off the board by a later refresh) -- a None still consumed its
    turn, so it still advances the snake.

    Returns `(rosters, recent)` in exactly the shape `_run_draft` resumes
    from, so a mid-draft run starts with every team holding what it actually
    took: `need` and the roster caps see real counts, and my own earlier
    picks are part of the roster the search is valuing.

    `last_pick` is the third piece of that state: `{position: overall_pick}`
    for each team's most recent pick at each position, which is what
    `rounds_since_pos` reads. `counts` says how many and carries no timing,
    so it cannot answer it. A resumed draft that rebuilt counts but not this
    would tell every opponent they have never taken a running back, one pick
    after they took one.

    A None entry (a pick whose player has since left the pool) advances the
    snake but updates nothing, because its position is unknowable -- the same
    treatment `counts` and `recent` already give it.
    """
    slots = snake_slots(settings.teams, settings.rounds)
    rosters = {slot: {"counts": {}, "indices": [], "last_pick": {}}
               for slot in range(1, settings.teams + 1)}
    recent = []
    for offset, idx in enumerate(taken_order or []):
        if offset >= len(slots):
            break                       # more picks than the draft has turns
        if idx is None:
            recent.insert(0, None)
            continue
        slot = slots[offset]
        pos = pool.position[idx]
        rosters[slot]["counts"][pos] = rosters[slot]["counts"].get(pos, 0) + 1
        rosters[slot]["indices"].append(int(idx))
        rosters[slot]["last_pick"][pos] = offset + 1        # 1-based pick no
        recent.insert(0, pos)
    return rosters, recent[:RUN_WINDOW]


def _run_draft(pool, settings, slot_managers, my_slot, taken, betas, rng,
               forced=None, taken_order=None, *, record=None) -> dict:
    """Simulate the remainder of one snake draft and return every slot's
    roster, as `{slot: {"counts": {position: n, ...}, "indices": [pool
    index, ...], "last_pick": {position: overall_pick, ...}}}`.

    Resume contract: `taken` marks exactly the players drafted so far in
    THIS draft -- the number of picks already made, used as an index into the
    snake pick order to resume from. If `taken` is ever built any other way
    (a player unavailable for a reason other than "already picked", or a
    count that does not match the real number of picks made), every
    remaining slot assignment misaligns silently: nothing here can detect
    that from `taken` alone.

    `taken_order` is the same information WITH attribution: the pool index of
    each already-made pick, in pick order (see `_seed_rosters`). It is what
    lets a mid-draft run know who holds what. Without it every roster resumes
    empty, so every opponent's `need` reads 1.0 at every position and the
    roster caps re-arm from zero -- a manager already holding three QBs is
    free to take three more. Callers that know the pick order must pass it;
    `run_sim` always does. Omitting it while players are already off the
    board warns rather than failing silently.

    `record`, when given a list, receives `(overall_pick, pool index)` for
    every pick this call simulates, in pick order. The per-slot return value
    already says WHO holds a player but not WHEN he went, and the predicted
    draft board is exactly that mapping. Opt-in because the common caller
    (`rollout`, run thousands of times inside a search) has no use for it.

    `rollout` wraps this and returns only my `roster_value` as a scalar --
    Task 11's search depends on that scalar return type -- so this returns
    the full per-slot state instead, for anything (including tests) that
    needs to inspect roster composition beyond my own final number.
    """
    gone = taken.copy()
    caps = _roster_cap(settings)
    rounds = settings.rounds
    slots = snake_slots(settings.teams, rounds)
    turns_left = _turns_left(slots, rounds)
    if taken_order is None and gone.any():
        warnings.warn(
            "draft_sim: players are marked taken but no taken_order was given; "
            "their picks cannot be attributed to a team, so every roster "
            "resumes empty (needs, roster caps and my own roster are wrong)",
            RuntimeWarning, stacklevel=2)
    rosters, recent = _seed_rosters(pool, settings, taken_order)
    already = len(taken_order) if taken_order is not None else int(gone.sum())
    if already >= len(slots):
        # The draft is already over by the contract above -- there is
        # nothing left to simulate, so report the reconstructed rosters as
        # they stand: an explicit, documented answer rather than an accident
        # of `slots[already:]` happening to be an empty slice.
        return rosters

    for offset, slot in enumerate(slots[already:], start=already):
        available = np.flatnonzero(~gone)
        if len(available) == 0:
            break
        overall_pick = offset + 1
        roster = rosters[slot]
        if slot == my_slot:
            if forced is not None and not gone[forced]:
                choice = forced
                forced = None
            else:
                choice = _greedy_choice(
                    pool, available, roster, settings, caps,
                    gap=_opportunity_gap(slots, offset, my_slot),
                    turns_left=turns_left[offset])
        else:
            beta = betas.get(slot_managers.get(slot))
            if beta is None:
                # A zeros beta makes every score 0, so the softmax below is
                # uniform: this opponent drafts the ENTIRE remaining pool at
                # equal odds, ignoring ADP, need and position entirely. That
                # is not "unmodeled" -- it is an actively wrong, confident
                # model of a fantasy manager, and it is silent: nothing here
                # would ever surface a fallback that fires. It is also the
                # common case, not a rare one -- `slot_managers.get(slot)` is
                # `None` for any slot ESPN hasn't published a draft order
                # for, which is every slot until ESPN sets one, and `betas`
                # is `{}` for any league with no draft history to fit
                # (api/live.py's `build_session` builds it as `{m:
                # fits.get(m, pooled) for m in fits if m != "__pooled__"}`,
                # which is empty at cold start since fit_all then returns
                # only `__pooled__`). COLD_START_PRIOR is the fix: a real,
                # measured "drafts like the market" prior (see its docstring
                # at scoring/draft_model.py), so an unresolvable opponent
                # follows ADP instead of drafting at random.
                beta = COLD_START_PRIOR
            legal = available[_legal_mask(pool, available, roster["counts"], caps)]
            # The roster floor, applied on top of the fit rather than inside
            # it: once an opponent's remaining picks have run down to their
            # unfilled starter slots, those slots are what the picks are for.
            # Without it no modelled opponent ever drafted a defense at all
            # -- 7 of 8 simulated teams finished with an empty DST slot, and
            # `pos_DST` is fitted on a `draft_picks` table containing zero
            # DST rows. The whole argument, with the measurements, is in
            # `_must_fill_mask`'s docstring; it returns None (leaving `legal`
            # exactly as the fit left it) for every pick where the constraint
            # does not bind, which measures as everything before round 14.
            must_fill = _must_fill_mask(pool, legal, roster["counts"],
                                        settings, turns_left[offset])
            if must_fill is not None:
                legal = legal[must_fill]
            if len(legal) == 0:
                choice = int(available[0])      # no legal player exists at all
            else:
                X = _live_features(pool, legal, overall_pick, roster["counts"],
                                   recent, settings, roster["last_pick"])
                scores = X @ beta
                # `scores.max()` is one of `scores` itself, so it always
                # contributes exp(0) == 1 -- weights.sum() is always >= 1,
                # never zero, so no zero-division guard is needed here.
                weights = np.exp(scores - scores.max())
                choice = int(rng.choice(legal, p=weights / weights.sum()))
        gone[choice] = True
        pos = pool.position[choice]
        rosters[slot]["counts"][pos] = rosters[slot]["counts"].get(pos, 0) + 1
        rosters[slot]["indices"].append(choice)
        rosters[slot]["last_pick"][pos] = overall_pick
        if record is not None:
            record.append((overall_pick, int(choice)))
        recent.insert(0, pos)

    return rosters


def _my_value(pool, rosters, my_slot, settings) -> float:
    mine = rosters[my_slot]["indices"]
    return roster_value(
        [(pool.position[i], pool.points[i], pool.availability[i]) for i in mine],
        settings)


def rollout(pool, settings, slot_managers, my_slot, taken, betas, rng,
            forced=None, taken_order=None) -> float:
    """Simulate the remainder of one snake draft and return my end-of-draft
    roster_value. See `_run_draft` for the full resume contract, including
    what `taken_order` is for."""
    rosters = _run_draft(pool, settings, slot_managers, my_slot, taken, betas,
                         rng, forced, taken_order=taken_order)
    return _my_value(pool, rosters, my_slot, settings)


DEFAULT_ROLLOUTS = 300
DEFAULT_CANDIDATES = 12


def horizon_picks(settings) -> int:
    """How many opponent picks a turn has to be away before measuring
    `gain_now` against it says anything.

    One full round of the draft, counted in opponent picks: a round is
    `teams` picks and one of them is mine, so `teams - 1` are somebody
    else's. Derived from `settings.teams`, never a hardcoded 7.

    THE SNAKE MAKES THIS THE NATURAL LINE. My two turns in a round pair are
    `2 * (slot - 1)` and `2 * (teams - slot)` opponent picks apart, and those
    two always sum to `2 * teams - 2` -- so one of them is always below
    `teams - 1` and the other always above it, for every slot. In this
    league (verified against snake_slots, not assumed): slot 1 gets 0 and
    14, slot 2 gets 2 and 12, slot 4 gets 6 and 8, slot 8 gets 14 and 0.
    `teams - 1` is therefore exactly the line between a turnaround -- my two
    picks at the wheel, or near it, where the board barely moves in between
    -- and a genuine wait of a full round. Skipping the first and measuring
    to the second is the whole rule.

    Why measuring to the turnaround is worthless, in the owner's own words:
    "just because I don't have a TE in the 2nd round doesn't mean I should
    go for a TE in round 3 since a lot of TEs are in round 9." What tells a
    round-3 tight end from a round-9 one is the SHAPE of the position's
    supply curve -- TE1 -> TE3 is a slide, RB1 -> RB9 is a cliff -- and one
    step over one or two opponent picks cannot see a shape at all. Measured
    on the real 249-player pool, 8 teams, pick 1 on the clock, `gain_now`
    positive for how many players at all:

        gap (opponent picks)   1    2    3    4    5    6    7   14
        players with gain > 0  1    3    4    5    6    7    8   13

    (Slots 2..8 and then slot 1, all at pick 1, horizon off.) Under ten and
    the bottom of a top-ten list is the zero tail -- every position's leader
    tied at 0.0, ordered by nothing, which is how a defense and a kicker
    placed 5th and 6th in round 1. `teams - 1` is the smallest threshold
    that clears that at every slot: with it, the first kicker at pick 1
    ranks 12th-17th and the first defense 11th-16th, where before they
    ranked 6th-12th and 5th-11th at every slot but the one whose next turn
    was already a full round away.

    Both extremes cost something and both were measured, so this is a
    trade-off, not a free parameter:

    - Shorter (H=0, the old behavior) is the defect: at a 1-3 pick gap
      everything survives at ~100%, `gain.expected_best_next` collapses to
      the position's own leader, and `gain_now` is 0.0 for the leader at
      every position including K and DST.
    - Longer does NOT keep getting better. A horizon past my next real turn
      prices a wait I never actually take, and as it lengthens `gain_now`
      slides back toward raw `need_weight * vor_points` -- the ranking the
      spec's "Gain now" section exists to replace, the one that reaches for
      quarterbacks and tight ends. Measured at H = 2*teams-2 = 14 (one full
      round-TRIP, the other natural anchor): at pick 9 the horizon jumps
      past BOTH of slot 2's real turns (15 and 18) out to 31, and two tight
      ends climb into the top ten (McBride 10th -> 4th); in round 3 Josh
      Allen climbs 9th -> 4th. At H=30 the top of the board is raw VOR
      order. So the rule takes the FIRST turn that is a real wait, not the
      furthest one available.

    This is only the FLOOR. "First turn that is a real wait" does not bound
    the far end at all, and the snake guarantees a case where the first such
    turn is a full round TRIP away -- see `horizon_ceiling`, which is the
    other half of the same rule and carries the measurements for that end.
    """
    return settings.teams - 1


def horizon_ceiling(settings) -> int:
    """How far away a turn may be and still be worth measuring against.

    A round and a half of opponent picks, `3 * (teams - 1) // 2` -- 10 in
    this league, 16 in a 12-team one. The COMPANION of `horizon_picks`, and
    the two exist to steer between the two ways this measurement dies. Both
    were measured; neither is a preference:

    - TOO NEAR (what `horizon_picks` is the floor against): at a 0-2
      opponent-pick step every available player survives at ~100%,
      `gain.expected_best_next` collapses to the position's own leader and
      `gain_now` is 0.0 for the best player at every position -- six rows
      tied at zero, ordered by pandas' sort, which is how a defense placed
      5th and a kicker 6th in round 1.
    - TOO FAR (what this is the ceiling against): the survival column the
      room prints goes to zero for every row it shows, so the number the
      owner reads carries nothing, and the ranking slides toward raw
      `vor_points`. Measured on the real 249-player board (8 teams, 15
      rounds, slot 2 on the clock at pick 15, 400 rollouts), holding the
      state fixed and moving ONLY the horizon -- `med` is the median
      survival of the fifteen rows the room actually shows, `rho` is the
      rank correlation of `gain_now` with `vor_points` over the top 50 of
      the board:

          opponent picks   0     4     7    10    12    14    18    21
          shown-row med  1.00  0.90  0.73  0.31  0.10  0.07  0.01  0.00
          rho(gain,vor)  .685  .646  .677  .735  .762  .758  .851  .892

      Two other states (slot 5 at pick 12, slot 4 at pick 13) put the same
      cliff in the same place: the median shown row is still 0.87-0.91 at 8
      opponent picks, 0.19-0.48 at 10, and 0.09-0.12 by 11.

    So the usable window is roughly 7-10 opponent picks for this league, and
    10 is where it is cut: past that the survival distribution is ~0.0 for
    everything the room displays, which is the mirror image of the ~1.0-for-
    everything failure the floor exists to prevent. Do NOT collapse this
    back toward either end without re-running that sweep.

    WHY A CEILING IS NEEDED AT ALL, given the floor. A snake's two gaps for
    a slot are `2 * (slot - 1)` and `2 * (teams - slot)` opponent picks and
    they sum to `2 * teams - 2`, so the walk in `_horizon_pick_for` produces
    exactly two distances: my next turn's own gap when it clears the floor,
    or -- when it does not, which happens at every slot on every second turn
    and at slots 1 and 8 on EVERY turn -- the turn after it, which is always
    exactly `2 * (teams - 1)` away. That second case is 14 opponent picks
    here, four past the cliff above, and no floor can bound it: a rule that
    only asks "is this far enough" will always take it.
    """
    return 3 * (settings.teams - 1) // 2


def _pick_after_opponent_picks(slots, my_slot, anchor, n):
    """The nearest pick after `anchor` with exactly `n` opponent picks
    between it and `anchor`, or None if the draft ends first.

    Used only by the ceiling: when no turn of mine sits inside the usable
    window, this is the pick the ranking is measured to instead. It is a
    real pick of this draft and it is what `horizon_pick` then names -- just
    not one of mine, which is the honest answer, since the alternative is a
    number that names my turn and measures a distribution with nothing in
    it. `len(slots)` is the last pick of the draft; a ceiling that would
    land past it returns None so the caller keeps the turn it found rather
    than colliding with `_horizon_pick_for`'s off-the-end sentinel.
    """
    seen = 0
    for pick in range(anchor + 1, len(slots) + 1):
        if slots[pick - 1] != my_slot:
            seen += 1
            if seen == n:
                return pick + 1 if pick + 1 <= len(slots) else None
    return None


def _horizon_pick_for(settings, my_slot, already, horizon: int = 0, *,
                      advised: int | None = None,
                      ceiling: int | None = None) -> int:
    """The overall pick number (1-based) of my next turn at least `horizon`
    opponent picks after the pick this advice is FOR, capped at `ceiling`.

    `already` is the pick count the walk starts from; the scan is INCLUSIVE
    of the pick at that offset, which is what lets `on_the_clock` (see
    `survival`) choose between "my pick is now" and "my pick is next" by
    shifting `already` rather than by a second code path.

    `advised` is the pick the ranking is FOR, and it is the anchor every
    distance below is measured from -- NOT "now". They are the same thing
    while I am on the clock and they are not while I am not: at pick 11 of
    an 8-team draft with my turns at 2, 15, 18, 31, the list being built is
    a preview of pick 15, so the wait it prices starts at 15. Measuring from
    11 instead was the live bug this parameter closes: 15 came back 4
    opponent picks away and 18 six, both under a `teams - 1` threshold, so
    the walk landed on 31 -- thirteen picks out and two of my own turns
    later, a horizon the owner could not have waited for even in principle,
    and one at which nothing the room showed survived at all. Anchored on
    15, the same state answers exactly what pick 15 answers when it arrives,
    so the preview and the live list agree. Defaults to `already`, which is
    the on-the-clock reading and the one `_next_pick_for` wants.

    Walks my remaining turns AFTER the anchor and counts, for each, how many
    picks between the anchor and there are somebody else's: a turn `k`
    places later in my own sequence has `k` of my own picks in front of it,
    so the opponent count to my `k`-th candidate at pick P is
    `(P - 1 - anchor) - k`. The first turn that clears `horizon` wins.

    `ceiling` (opponent picks; `horizon_ceiling` for the rule and the
    measurements) is the other end. When the turn the walk finds is further
    away than that, the answer is the pick exactly `ceiling` opponent picks
    after the anchor instead -- a real pick of this draft, but not
    necessarily one of mine. That is deliberate and it is the point: at the
    wheel my own turns offer a choice between 2 opponent picks away and 14,
    and neither measures anything (see `horizon_ceiling` for both
    distributions).

    `horizon <= 0` returns my very next turn and nothing else applies -- no
    anchor, no ceiling. That is the historical meaning `_next_pick_for` and
    every offline caller (run_sim, search_pick) depend on, and it is why
    they are still this one function rather than a second copy of the same
    arithmetic.

    Three fallbacks, and they are different things:

    - No turn is far enough away (late in the draft, where my remaining
      turns simply run out before `horizon` picks do): the LAST turn I have.
      It is the most informative horizon that actually exists for me, and it
      is a real pick number the room can name. The ceiling cannot bind here
      -- that turn is nearer than `horizon`, which is nearer than `ceiling`.
    - No turn remains AFTER the anchor (the anchor is my final pick, whether
      I am on the clock for it or previewing it): `len(slots) + 1`, the same
      off-the-end answer `_next_pick_for` has always given, which `survival`
      reads as "simulate to the end of the draft". That number is not a pick
      that exists, so the caller must not print it -- see api/live.py's
      `horizon_is_end_of_draft`.
    - No turn remains at all: the same sentinel, for the same reason.
    """
    slots = snake_slots(settings.teams, settings.rounds)
    mine = [offset + 1 for offset in range(already, len(slots))
            if slots[offset] == my_slot]
    if not mine:
        return len(slots) + 1
    if horizon <= 0:
        # No horizon means no anchor and no ceiling either: the answer is my
        # very next turn, which is what `_next_pick_for` and every offline
        # caller (run_sim, search_pick) have always meant by this call. It
        # has to be said outright rather than falling out of the walk,
        # because the walk now measures FROM the advised pick and never
        # returns the anchor itself -- so a zero horizon would otherwise
        # answer with the turn AFTER my next one, which is not "my next
        # pick" by any reading. The arithmetic below is still the only copy
        # of it; this is the one case that needs no arithmetic at all.
        return mine[0]
    anchor = already if advised is None else advised
    if ceiling is None:
        ceiling = horizon_ceiling(settings)
    # Strictly after the anchor: when the anchor IS my next turn (the
    # preview reading) that turn is the pick being advised, not a horizon to
    # measure it against -- measuring a pick against itself is the
    # everything-survives degeneracy in its purest form.
    candidates = [pick for pick in mine if pick > anchor]
    if not candidates:
        return len(slots) + 1
    for k, pick in enumerate(candidates):
        gap = (pick - 1 - anchor) - k
        if gap >= horizon:
            if gap > ceiling:
                capped = _pick_after_opponent_picks(slots, my_slot, anchor,
                                                    ceiling)
                if capped is not None:
                    return capped
            return pick
    return candidates[-1]


def horizon_target(settings, my_slot, already, on_the_clock: bool,
                   horizon: int) -> int:
    """The pick a live ranking is measured against: ONE expression of the
    rule, called by `survival` (which measures it) and by api/live.py (which
    serves the number as `horizon_pick`).

    Both used to derive `start` and call `_horizon_pick_for` themselves, in
    two places, from the same two inputs. That is exactly the shape a
    drifting caption comes from -- a list ranked against one pick and
    captioned with another -- and now neither of them can say a different
    number than the other, whatever the rule becomes.

    `already` is the pick count (len(taken_order)), NOT the shifted start;
    `on_the_clock` is whether the pick about to be made is mine. Those are
    the two facts a caller has, and every derived quantity is worked out
    here: the scan start (inclusive, so it has to skip my own pick when that
    pick is the one on the clock) and the anchor the horizon is measured
    from (that same pick when it is mine, my next turn when it is not --
    see `_horizon_pick_for`'s `advised`).
    """
    start = already + 1 if on_the_clock else already
    advised = (start if on_the_clock
               else _next_pick_for(settings, my_slot, already))
    return _horizon_pick_for(settings, my_slot, start, horizon,
                             advised=advised)


def _next_pick_for(settings, my_slot, already) -> int:
    """My very next turn: `_horizon_pick_for` at horizon zero, not a second
    implementation of it. run_sim and search_pick reach the pick arithmetic
    only through this name and mean exactly this."""
    return _horizon_pick_for(settings, my_slot, already, 0)


SEARCH_COLUMNS = ["player_id", "ev", "se", "applied_pct", "rank"]

# A candidate has to have a real chance of still being there at my next pick
# for the number attached to him to mean anything. Below this, forcing him
# mostly does not happen and the "expected value of taking him" is really the
# expected value of the greedy baseline.
MIN_CANDIDATE_AVAIL = 0.25


def _candidate_indices(pool, available, avail_pct, n_candidates) -> list:
    """Which players to actually evaluate at my next pick.

    Two sources, interleaved: best by market rank, and best by VOR -- the
    board's own headline ranking, and what the spec asks for. Slicing
    `n_candidates` off each source, concatenating, and truncating back to
    `n_candidates` returned the market list verbatim: its entries are already
    unique and already fill the slice, so the second source never operated at
    all. (With points and market rank deliberately anti-correlated, the
    candidates came back p0..p11 and the top twelve by points, p59..p48,
    contributed nothing.) Interleaving takes roughly half from each and tops
    up from whichever list still has entries when the two overlap.

    Both sources are drawn only from players who might actually still be
    there when my turn comes. Forcing a candidate an opponent already took
    falls through to the greedy policy, and under common random numbers every
    such candidate then produces the identical draft: at slot 8 against
    ADP-disciplined opponents, four of twelve candidates came back
    byte-identical (ev=1332.449153, se=7.406298) and ranked 3rd through 6th,
    for players with a 7-22% chance of being available. If nobody clears the
    bar -- only possible in a pool small enough that everything turns over --
    fall back to the whole available set rather than returning nothing.
    """
    live = available[avail_pct[available] >= MIN_CANDIDATE_AVAIL]
    if len(live) == 0:
        live = available
    by_market = live[np.argsort(pool.adp_rank[live], kind="stable")]
    by_vor = live[np.argsort(-pool.vor[live], kind="stable")]
    picked, seen = [], set()
    for from_market, from_vor in zip_longest(by_market, by_vor):
        for idx in (from_market, from_vor):
            if idx is None or int(idx) in seen:
                continue
            seen.add(int(idx))
            picked.append(int(idx))
            if len(picked) == n_candidates:
                return picked
    return picked


def search_pick(pool, settings, slot_managers, my_slot, taken, betas,
                n_rollouts: int = DEFAULT_ROLLOUTS,
                n_candidates: int = DEFAULT_CANDIDATES, seed: int = 0,
                taken_order=None, avail_pct=None):
    """Expected end-of-draft roster value for each candidate at my next pick.

    Candidates come from `_candidate_indices`: market rank and VOR, among the
    players likely to survive to that pick.

    Rollout i uses seed (seed, i) for every candidate -- common random numbers,
    so all candidates face identical opponent behavior and the comparison
    between them is far less noisy than independent sampling at the same cost.

    `applied_pct` reports the share of rollouts in which the candidate was
    actually still on the board at my turn, i.e. in which forcing him meant
    anything at all; the rest fell through to the greedy policy. Candidates
    below `MIN_CANDIDATE_AVAIL` are dropped rather than ranked, so the board
    never shows a confident ΔEV for a player who will not be there. (`forced`
    ending up on my roster is exactly equivalent to it having applied: if he
    was gone at my turn he can never come back, and if it applied he is
    rostered.)

    `avail_pct`, pool-aligned, is that survival estimate. `run_sim` computes
    it once and shares it with the `sim_survival` table; left out, it is
    computed here.

    `taken_order` is passed straight through to every rollout; see
    `_run_draft` for what it is and why a mid-draft run needs it.
    """
    available = np.flatnonzero(~taken)
    if len(available) == 0:
        return pd.DataFrame(columns=SEARCH_COLUMNS)
    if avail_pct is None:
        avail_pct = survival(pool, settings, slot_managers, my_slot, taken,
                             betas, n_rollouts=n_rollouts, seed=seed,
                             taken_order=taken_order)["avail_pct"].to_numpy()
    candidates = _candidate_indices(pool, available, np.asarray(avail_pct),
                                    n_candidates)

    rows = []
    for idx in candidates:
        values, applied = [], 0
        for i in range(n_rollouts):
            rosters = _run_draft(pool, settings, slot_managers, my_slot, taken,
                                 betas, rng=np.random.default_rng([seed, i]),
                                 forced=idx, taken_order=taken_order)
            values.append(_my_value(pool, rosters, my_slot, settings))
            applied += int(idx in rosters[my_slot]["indices"])
        values = np.array(values)
        rows.append({"player_id": pool.player_id[idx],
                     "ev": float(values.mean()),
                     "se": float(values.std(ddof=1) / np.sqrt(len(values)))
                     if len(values) > 1 else 0.0,
                     "applied_pct": applied / max(n_rollouts, 1)})
    frame = pd.DataFrame(rows)
    kept = frame[frame["applied_pct"] >= MIN_CANDIDATE_AVAIL]
    if not kept.empty:
        frame = kept
    out = frame.sort_values("ev", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out[SEARCH_COLUMNS]


def survival(pool, settings, slot_managers, my_slot, taken, betas,
             n_rollouts: int = DEFAULT_ROLLOUTS, seed: int = 0,
             taken_order=None, on_the_clock: bool = False,
             horizon: int = 0):
    """Probability each player is still available when my measured turn
    arrives -- my next turn by default, or the first one `horizon` opponent
    picks away.

    Counted from the same rollout machinery, but stopping at that pick
    rather than running the draft out -- this is the "who can I wait on"
    number, and it only depends on what happens before my turn.

    `taken_order` carries the same pick attribution `_run_draft` needs, and
    for the same reason: without it every opponent between now and my turn
    resumes with an empty roster, so `need` reads 1.0 everywhere and the
    caps re-arm from zero.

    `on_the_clock` names WHICH turn is being measured, because the pick
    count alone cannot say. `_next_pick_for` scans from `already`
    INCLUSIVELY, so when the pick about to be made is my own it answers
    "my next turn is this pick", the rollout loop below has nothing to
    range over, and every available player comes back at exactly 1.0.

    That is the right answer for `search_pick`/`run_sim`, which value the
    pick they are about to make: a player who is on the board right now is
    there with certainty, and `_run_draft(forced=idx)` forces him at that
    same pick. It is the WRONG answer for the live ranking, whose whole
    question is what will still be there AFTER this pick -- with survival
    pinned at 1.0, `gain.expected_best_next` collapses to the position's
    own leader and `gain_now` is identically 0.0 for the best player at
    every position, so which of six position leaders (a kicker as readily
    as a running back) reaches the top three is decided by pandas' unstable
    sort rather than by any number. So the live caller passes
    `on_the_clock=True` and gets the turn after this one; the default keeps
    the offline callers unchanged.

    The measured turn is then found from `already + 1`, which leaves the
    player about to be taken with this very pick in the pool: survival is
    over-estimated by exactly one player, since one of the survivors
    counted here is the one I am about to remove myself. That is the
    deliberate choice -- the alternative is guessing which player that is,
    and a guess would be wrong far more often than one player in a
    position's tail matters.

    `horizon` (opponent picks; `horizon_picks` for the floor,
    `horizon_ceiling` for the far end, `horizon_target` for the walk that
    puts them together) is what makes the measurement mean anything when my
    next turn is close. It is measured from the pick the ranking is FOR --
    the one on the clock when that is mine, my next turn when it is not --
    and it is bounded at BOTH ends, because both ends destroy the
    measurement in mirror-image ways. `gain_now` is one step of a
    position's supply curve -- now versus the turn measured here -- and a
    step taken over one or two opponent picks is near zero for everybody,
    at which point the ranking has no signal left in it and the order of
    the top of the board is whatever the sort happened to produce. Observed
    live at pick 1 of an 8-team mock with the owner in slot 2, one opponent
    pick before their turn: every available player back at ~100% survival, a
    defense 5th and a kicker 6th in round 1. Stepping past that turn to one
    a full round of opponent picks away is what lets the curve's SHAPE show
    -- the owner's own framing, that a tight end being available in round 9
    is not a reason to take one in round 3.
    `horizon=0` is my very next turn, the historical behavior, and stays
    the default so `run_sim`/`search_pick` are untouched.

    This also retires a known artifact of measuring to the immediate next
    turn: at the wheel (my two picks back to back, no opponent in between)
    the rollout range was empty and survival was a uniform 1.0, so `gain_now`
    was degenerate for one turn of every round pair -- truthful about the
    one-step question, useless as a ranking. A horizon of at least one
    opponent pick can never land on the second half of a wheel, because the
    walk keeps going until it finds a turn far enough away.
    """
    if taken_order is None and taken.any():
        warnings.warn(
            "draft_sim.survival: players are marked taken but no taken_order "
            "was given; opponents resume with empty rosters, so their needs "
            "and roster caps are wrong", RuntimeWarning, stacklevel=2)
    already = len(taken_order) if taken_order is not None else int(taken.sum())
    start = already + 1 if on_the_clock else already
    # Both the pick measured to and the number api/live.py serves for it come
    # from this one function, so the caption can never name a different pick
    # than the one these rollouts stopped at.
    target = horizon_target(settings, my_slot, already, on_the_clock, horizon)
    slots = snake_slots(settings.teams, settings.rounds)
    turns_left = _turns_left(slots, settings.rounds)
    caps = _roster_cap(settings)
    counts = np.zeros(len(pool.player_id))

    for i in range(n_rollouts):
        rng = np.random.default_rng([seed, i])
        gone = taken.copy()
        seeded, seeded_recent = _seed_rosters(pool, settings, taken_order)
        rosters = {slot: state["counts"] for slot, state in seeded.items()}
        # Carried beside `rosters` rather than folded into it because this
        # loop indexes `rosters[slot]` as the bare counts dict throughout
        # (caps, `_must_fill_mask`). Same information `_run_draft` keeps
        # under `roster["last_pick"]`, and it has to be here too or the two
        # disagree about the same opponents -- `survival` IS what "will he
        # still be there" is counted from.
        last_picks = {slot: state["last_pick"] for slot, state in seeded.items()}
        recent = list(seeded_recent)
        for offset in range(start, min(target - 1, len(slots))):
            slot = slots[offset]
            available = np.flatnonzero(~gone)
            if len(available) == 0:
                break
            beta = betas.get(slot_managers.get(slot))
            if beta is None:
                # Same fallback and the same reason as `_run_draft` above:
                # zeros made the softmax uniform, so an unresolved opponent
                # drafted the whole remaining pool at equal odds. This is
                # what turned into the live bug -- pick 22 of an 8-team mock,
                # 9 opponent picks before the owner's next turn, and EVERY
                # available player (Derrick Henry and Josh Jacobs included)
                # came back at 94-98% survival, because uniform draws give
                # each of ~249 available players survival ~= 1 - 9/249 =
                # 96.4% regardless of how good he is. `expected_best_next`
                # then equalled each position's own best player,
                # `gain.gain_now` collapsed to 0 for every position leader,
                # and the top-3 board was a QB, a kicker and a TE at 0 --
                # pandas' unstable sort deciding the order among six tied
                # zeros, not the model. COLD_START_PRIOR makes the fallback
                # follow the market instead of drafting at random.
                beta = COLD_START_PRIOR
            X = _live_features(pool, available, offset + 1, rosters[slot],
                               recent, settings, last_picks[slot])
            scores = X @ beta
            for j, idx in enumerate(available):
                pos = pool.position[idx]
                if rosters[slot].get(pos, 0) >= caps.get(pos, 99):
                    scores[j] = -np.inf
            # The same roster floor `_run_draft` applies, and it has to be
            # here too or the two disagree about the same opponents: this
            # loop IS what "will he still be there" is counted from, so an
            # opponent who never drafts a defense here hands every defense a
            # survival of 1.000 and a `gain_now` of exactly 0.0000 at every
            # pick of the draft (measured -- see `_must_fill_mask`).
            # Re-checked against `finite` rather than applied blind: a
            # position at its cap is already -inf, and forcing the pick onto
            # a set with no finite score left would leave the softmax below
            # with nothing to sample.
            must_fill = _must_fill_mask(pool, available, rosters[slot],
                                        settings, turns_left[offset])
            if must_fill is not None and np.isfinite(scores[must_fill]).any():
                scores[~must_fill] = -np.inf
            finite = np.isfinite(scores)
            if not finite.any():
                choice = int(available[0])
            else:
                shifted = scores - scores[finite].max()
                weights = np.where(finite, np.exp(shifted), 0.0)
                total = weights.sum()
                choice = int(available[0]) if total <= 0 else \
                    int(rng.choice(available, p=weights / total))
            gone[choice] = True
            pos = pool.position[choice]
            rosters[slot][pos] = rosters[slot].get(pos, 0) + 1
            last_picks[slot][pos] = offset + 1
            recent.insert(0, pos)
        counts += ~gone

    return pd.DataFrame({"player_id": pool.player_id,
                         "avail_pct": counts / max(n_rollouts, 1)})


def _drafted_state(conn, pool):
    """`(taken, taken_order)` for the players already off the board.

    `drafted.pick_no` is written by `POST /api/drafted` and is the only
    record of what order the board was marked up in, which is what makes
    attribution possible at all: pick k in the snake order belongs to the
    slot that was on the clock for pick k.

    Rows whose `pick_no` is null predate the column (`pipeline/db.py` adds it
    to existing databases with a plain ALTER, so anything drafted before that
    migration has no value). Those picks are genuinely unattributable -- any
    order we invented for them, including "nulls first", would hand real
    players to the wrong teams and silently corrupt every roster, need and
    cap downstream. So this refuses to guess and asks the user to re-mark
    them, which is cheap: un-toggle and re-toggle in draft order.

    `taken_order` is indexed BY `pick_no`, not by row: entry i is overall pick
    i+1, and a pick number this table holds no row for is a None. That is the
    contract `_seed_rosters` reads it under (it walks the snake by position,
    so entry i is handed to whoever was on the clock for pick i+1) and it is
    NOT the same as "one entry per row" -- `drafted` is routinely missing pick
    numbers it never had a row for:

      - a pick whose player the crosswalk could not resolve is reported in
        `LivePicks.unmapped` and deliberately never written here (see
        `pipeline/espn_live.picks_from_events`), and
      - `DELETE /api/drafted/{id}` takes a manually-marked pick back out and
        leaves its number behind.

    Building the list row by row made every pick AFTER such a hole shift one
    slot to the left, silently: on a mid-draft JOIN, where ESPN replays the
    whole draft at once, one unmapped pick in round 1 re-attributed every
    round after it and the owner's own picks showed up in a neighbour's
    column, which is the "it doesnt show my previous picks" report this
    indexing fixes. `len(taken_order)` is therefore the number of picks
    actually MADE (the highest pick number seen), which is also what
    `_run_draft` and `survival` resume from as `already` -- counting rows
    there had them resume from the wrong turn by exactly the number of holes.
    """
    drafted = read_table(conn, "drafted")
    if drafted.empty:
        return np.zeros(len(pool.player_id), dtype=bool), []
    if "pick_no" not in drafted.columns or drafted["pick_no"].isna().any():
        missing = int(drafted["pick_no"].isna().sum()) \
            if "pick_no" in drafted.columns else len(drafted)
        raise ValueError(
            f"{missing} drafted player(s) have no pick_no, so they cannot be "
            "attributed to a team and the simulation would run against wrong "
            "rosters. Un-mark and re-mark them in draft order (or clear the "
            "drafted table) and run again.")
    ordered = drafted.sort_values("pick_no")
    numbers = [int(n) for n in ordered["pick_no"]]
    # Every writer of this table numbers from 1 (`picks_from_events` counts
    # SELECTED frames, `POST /api/drafted` uses max+1, `translate` copies
    # ESPN's own 1-based overallPickNumber), so a number below 1 cannot be
    # placed in the snake at all -- exactly as unattributable as the null
    # above, and refused the same way rather than silently wrapping onto the
    # end of the list via a negative index.
    if numbers[0] < 1:
        raise ValueError(
            f"drafted pick_no {numbers[0]} is not a pick number -- pick "
            "numbers start at 1, so this row cannot be attributed to a team. "
            "Un-mark and re-mark it in draft order (or clear the drafted "
            "table) and run again.")
    position_of = {pid: i for i, pid in enumerate(pool.player_id)}
    # Sized by the LAST pick number, not the row count, so the holes above
    # stay in the order as Nones. A drafted player who is no longer on the
    # board (a refresh dropped him) lands as a None here too, for the same
    # reason and with the same effect: he still consumed his turn, so he must
    # not shift every later pick onto the wrong slot.
    taken_order = [None] * numbers[-1]
    taken = np.zeros(len(pool.player_id), dtype=bool)
    for pid, pick_no in zip(ordered["player_id"], numbers):
        idx = position_of.get(pid)
        # Last row wins for a duplicated pick_no, which no writer produces
        # (apply_picks rebuilds the table wholesale from one numbering, and
        # POST /api/drafted takes max+1) -- noted because the alternative,
        # dropping one of the two, would lose a pick rather than a position.
        taken_order[pick_no - 1] = idx
        if idx is not None:
            taken[idx] = True
    return taken, taken_order


def run_sim(conn, my_slot: int, slot_managers: dict,
            n_rollouts: int = DEFAULT_ROLLOUTS, seed: int = 0) -> str:
    from scoring import league as league_mod
    from scoring.board import build_board
    from scoring.draft_model import fit_all

    settings = league_mod.load(conn)
    board = build_board(conn, settings=settings)
    pool = build_pool(conn, board, settings)

    fits = fit_all(conn, settings)
    # This used to reject a fit with no per-manager entries, on the reasoning
    # that "no fitted opponent" meant every beta defaulted to zeros -- a
    # uniform-random draft, a confident wrong answer. That reasoning is stale.
    # `fit_all` now returns a market-following prior as `__pooled__` when a
    # league has no history (draft_model.cold_start_fits), and every opponent
    # without a personal fit inherits it via the `setdefault(manager, pooled)`
    # below. So "only __pooled__" now means "everyone drafts to the board",
    # which is a measured default, not noise. The only genuinely broken case
    # is a degenerate pooled vector -- missing or all zeros -- which no code
    # path produces today but is cheap to refuse rather than silently simulate
    # a uniform draft.
    pooled_check = fits.get("__pooled__")
    if pooled_check is None or not np.any(pooled_check):
        raise ValueError(
            "no usable opponent model: the pooled coefficients are missing or "
            "all zero, which would make every opponent draft uniformly at "
            "random. This should not happen -- fit_all returns a market prior "
            "even with no history -- so it points at a build error upstream.")
    profiles = read_table(conn, "manager_profiles")
    pooled = fits.get("__pooled__", np.zeros(len(FEATURE_NAMES)))
    betas = {}
    for manager, beta in fits.items():
        if manager == "__pooled__":
            continue
        # `profiles.empty` alone is not a safe guard here: read_table
        # returns a columnless pd.DataFrame() when `manager_profiles`
        # doesn't exist yet (e.g. `make sim` run before `make
        # fit-managers`), and indexing a columnless frame by "manager" below
        # raises KeyError rather than yielding an empty (safely-filterable)
        # result -- a real crash the brief's original
        # `rows = profiles[profiles["manager"] == manager]` did not guard
        # against, only reachable once fit_all has real per-manager fits to
        # iterate (an empty conn's fit_all returns just "__pooled__", which
        # never enters this loop body -- exactly why the brief's own tests
        # never hit it). No personal profile on record defaults to pooled,
        # same as write_profiles' own default when there isn't enough
        # signal.
        if profiles.empty or "manager" not in profiles.columns:
            personal = False
        else:
            rows = profiles[profiles["manager"] == manager]
            personal = bool(rows["uses_personal"].iloc[0]) if not rows.empty else False
        betas[manager] = beta if personal else pooled

    # `reach` is max(0, log1p(market_rank) - log1p(pick)) (Task 4), so a
    # positive coefficient says "the further past his market rank a player
    # is, the more I want him" -- exp(reach * beta) on a deep-pool player
    # then dominates every other term and the manager drafts the deepest
    # available player in the pool, making that manager's simulated
    # behaviour meaningless rather than merely noisy. Cheap to check, and
    # there is nowhere else it surfaces.
    # A slot with no manager, or a manager with no fit, silently fell through
    # to a zeros beta -- the same uniform-random opponent the guard above
    # refuses to run with, arriving one layer later. It is reachable without
    # any user error: ESPN leaves `draftDayPickOrder` null until it publishes
    # a draft order, so every team in the upcoming season carries a null
    # slot, and a caller building {slot: manager} from those rows collapses
    # all eight teams into one entry keyed None.
    expected = set(range(1, settings.teams + 1))
    missing = sorted(expected - set(slot_managers))
    if missing:
        raise ValueError(
            f"draft order covers slots {sorted(slot_managers)} but this "
            f"league has {settings.teams} -- slots {missing} have no "
            "manager. ESPN leaves the draft order unset until it publishes "
            "one, so assign it in the rail (or PUT /api/draft-order) before "
            "simulating.")
    # An unrecognized name is a legitimate case -- a manager who joined this
    # year has no history to fit -- so they draft like the league average
    # rather than at random.
    for slot, manager in slot_managers.items():
        betas.setdefault(manager, pooled)

    reaching = sorted(m for m, beta in betas.items() if beta[_REACH] > 0)
    if reaching:
        warnings.warn(
            f"draft_sim: fitted reach coefficient is positive for {reaching}; "
            "those managers will be simulated drafting the deepest available "
            "players. Check the market_rank scale against historic_adp.",
            RuntimeWarning)

    taken, taken_order = _drafted_state(conn, pool)

    # Survival first: search_pick needs it to keep candidates it has no real
    # chance of getting out of the ranking, and computing it once means that
    # costs nothing beyond the sim_survival table we were writing anyway.
    avail = survival(pool, settings, slot_managers, my_slot, taken, betas,
                     n_rollouts=n_rollouts, seed=seed, taken_order=taken_order)
    results = search_pick(pool, settings, slot_managers, my_slot, taken, betas,
                          n_rollouts=n_rollouts, seed=seed,
                          taken_order=taken_order,
                          avail_pct=avail["avail_pct"].to_numpy())
    cells = predict_board(pool, settings, slot_managers, my_slot, taken, betas,
                          n_rollouts=n_rollouts, seed=seed,
                          taken_order=taken_order)

    run_id = f"{my_slot}-{n_rollouts}-{seed}-{len(taken_order)}"
    results.insert(0, "run_id", run_id)
    avail.insert(0, "run_id", run_id)
    cells.insert(0, "run_id", run_id)
    # Provenance (spec Part 4). scoring/board.py merges these tables into
    # every board unconditionally, and write_table replaces them wholesale,
    # so without a timestamp and the slot/pick they were computed for, a
    # reloaded page shows Avail% and ΔEV from an arbitrarily old run -- maybe
    # for a different slot -- with nothing on screen saying so. /api/sim/latest
    # serves these three and the rail renders them on load.
    results["my_slot"] = my_slot
    results["pick_no"] = _next_pick_for(settings, my_slot, len(taken_order))
    results["created_at"] = datetime.now(timezone.utc).isoformat()
    write_table(conn, "sim_results", results)
    write_table(conn, "sim_survival", avail)
    write_table(conn, "sim_board", cells)
    return run_id


def _assign_primaries(counts: dict) -> dict:
    """Each pick's most likely player, walking the board in draft order.

    Was a global assignment minimizing total -log(prob), which maximizes the
    joint likelihood of all 120 cells at once. That is a coherent
    statistical object and the wrong one for a draft board: it would trade
    pick 1 away to improve pick 13, and did -- a real run showed a 12%
    player at pick 1 while the 16% player sat in the hover, because he
    scored 40% at pick 13.

    A board is read top to bottom and the earliest picks are the ones that
    must be right, so earliest pick wins. Not the maximum-likelihood board,
    deliberately. A pick whose every candidate is already claimed keeps its
    own best anyway: a visible repeat beats inventing a pick nobody made.
    """
    primary, claimed = {}, set()
    for overall_pick in sorted(counts):
        cell = counts[overall_pick]
        ranked = sorted(cell, key=lambda idx: (-cell[idx], idx))
        pick = next((idx for idx in ranked if idx not in claimed), ranked[0])
        primary[overall_pick] = pick
        claimed.add(pick)
    return primary


def predict_board(pool, settings, slot_managers, my_slot, taken, betas,
                  n_rollouts: int = DEFAULT_ROLLOUTS, seed: int = 0,
                  alternates: int = 2, taken_order=None) -> pd.DataFrame:
    """Who the model thinks goes at every pick of the draft.

    Counts, across `n_rollouts` full drafts, which player each overall pick
    number produced. Two things come out of those counts:

    The raw per-cell frequencies, which are the honest distribution and
    become the alternates hover shows.

    A deduped primary (`_assign_primaries`). Taking each cell's most
    frequent player independently puts a consensus first-rounder in four
    adjacent cells, which reads as a bug rather than as "he could go at any
    of these". So the picks are walked in draft order and each claims its
    own most likely player not already claimed by an earlier pick -- the
    earliest picks are the ones a board is actually read for, so they win
    any contest over a shared player. A pick only repeats a player already
    claimed when every one of its own recorded candidates is already gone,
    which is a property of that input (the contesting picks collectively
    produced too few distinct players), not something the algorithm could
    avoid. The name in the cell is therefore not the maximum-likelihood
    board, deliberately, which is why the alternates matter and are kept.

    Picks already made are reported from `taken_order` as certain, with no
    alternates: they are facts, not predictions.
    """
    slots = snake_slots(settings.teams, settings.rounds)
    teams = max(settings.teams, 1)

    counts = {}
    for i in range(n_rollouts):
        record = []
        _run_draft(pool, settings, slot_managers, my_slot, taken, betas,
                   rng=np.random.default_rng([seed, i]),
                   taken_order=taken_order, record=record)
        for overall_pick, idx in record:
            counts.setdefault(overall_pick, {})
            counts[overall_pick][idx] = counts[overall_pick].get(idx, 0) + 1

    predicted = sorted(counts)
    rows = []

    # -- picks already made: facts, in pick order --
    for offset, idx in enumerate(taken_order or []):
        if offset >= len(slots):
            # More picks on record than the draft has turns -- the same
            # condition `_seed_rosters` breaks on, and for the same reason:
            # there is no slot that was on the clock for them. Emitting them
            # anyway meant a slot of 0, which no column on the grid carries,
            # so they vanished off the page with no trace.
            break
        if idx is None:
            continue
        rows.append(_board_row(pool, slots, teams, offset + 1, 0, idx, 1.0, True))

    if predicted:
        primary = _assign_primaries(counts)
        for overall_pick in predicted:
            cell = counts[overall_pick]
            chosen = primary[overall_pick]
            rows.append(_board_row(pool, slots, teams, overall_pick, 0, chosen,
                                   cell[chosen] / n_rollouts, False))
            ranked = sorted(((n, -idx) for idx, n in cell.items()
                             if idx != chosen), reverse=True)
            for alt_rank, (n, neg_idx) in enumerate(ranked[:alternates], start=1):
                rows.append(_board_row(pool, slots, teams, overall_pick,
                                       alt_rank, -neg_idx, n / n_rollouts, False))

    board = pd.DataFrame(rows, columns=["overall_pick", "round", "round_pick",
                                        "slot", "alt_rank", "player_id",
                                        "prob", "certain"])
    return board.sort_values(["overall_pick", "alt_rank"]).reset_index(drop=True)


def _board_row(pool, slots, teams, overall_pick, alt_rank, idx, prob, certain):
    """One grid cell. `overall_pick` must be a real turn (1..len(slots)) --
    both callers in `predict_board` guarantee it, the already-made loop by
    breaking past the end of the draft and the predicted loop because
    `_run_draft`'s `record` only ever walks `slots`. Indexed directly rather
    than falling back to a slot of 0, which no column on the grid carries and
    which therefore silently dropped the row."""
    offset = overall_pick - 1
    return {"overall_pick": overall_pick,
            "round": offset // teams + 1,
            "round_pick": offset % teams + 1,
            "slot": slots[offset],
            "alt_rank": alt_rank,
            "player_id": pool.player_id[idx],
            "prob": float(prob),
            "certain": bool(certain)}
