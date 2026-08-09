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
from scoring.board import FANTASY_POSITIONS, _norm_name, adp_match_key
from scoring.config import RECENCY_WEIGHTS
from scoring.draft_model import EARLY_ROUNDS, FEATURE_NAMES, RUN_WINDOW

FLEX_POSITIONS = ("RB", "WR", "TE")
GAMES = 17
# Floor for players with no projection and no stat history, per position, so
# a K or a rookie DST never lands as NaN inside the lineup optimizer.
POSITION_FLOOR = {"QB": 180.0, "RB": 80.0, "WR": 80.0, "TE": 60.0,
                  "K": 110.0, "DST": 100.0}
# Availability (0-100) for a player with no weekly history to compute one
# from: rookies, kickers, and every DST. A realistic full-season availability
# rate, not the neutral 50 the board uses for a missing *percentile* -- 50
# here would mean "expected to miss half the season", which is a claim about
# the player, not an admission of ignorance.
DEFAULT_AVAILABILITY = 90.0


def projections(conn, board: pd.DataFrame) -> pd.Series:
    """Projected season points per player_id.

    Ladder: ESPN's own season projection, then recency-weighted PPG scaled to
    a full season, then a per-position floor.
    """
    espn = read_table(conn, "espn_adp")
    lookup = {}
    if not espn.empty and "espn_proj" in espn.columns:
        valid = espn.dropna(subset=["espn_proj"])
        valid = valid[valid["espn_proj"] > 0]
        teams = (valid["team"] if "team" in valid.columns
                 else pd.Series([None] * len(valid), index=valid.index))
        # DSTs key on team, not name: ESPN says "Ravens D/ST" and the board
        # says whatever the ADP feed's nickname is, so a name join never hit
        # and every defense fell through to POSITION_FLOOR.
        for (_, row), team in zip(valid.iterrows(), teams):
            key = adp_match_key(row["espn_name"], row["position"], team)
            if key is not None:
                lookup[key] = float(row["espn_proj"])

    values = []
    for _, row in board.iterrows():
        key = adp_match_key(row["name"], row["position"], row.get("team"))
        proj = lookup.get(key) if key is not None else None
        if proj is None:
            stats = row.get("stats")
            ppg = stats.get("ppg") if isinstance(stats, dict) else None
            proj = float(ppg) * GAMES if ppg else None
        if proj is None or not np.isfinite(proj):
            proj = POSITION_FLOOR.get(row["position"], 80.0)
        values.append(proj)
    return pd.Series(values, index=board["player_id"].to_numpy(), dtype=float)


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


def build_pool(conn, board: pd.DataFrame, settings) -> SimPool:
    points = projections(conn, board)
    ranked = board.copy()
    ranked["proj"] = ranked["player_id"].map(points)

    # `adp_rank` must land on the scale draft_model's reach/fall coefficients
    # were fitted on: historic_adp.adp_rank is a dense rank over the players
    # ONE season's ADP source actually ranked, not over `board`'s broader
    # union of every player with a stat line plus ADP-only rookies and K/DST.
    # Folding a fillna sentinel into the same sort_values as the real
    # market_rank values is not safe here: fp_rank/mfl_rank/cbs_rank (which
    # feed into market_rank, see scoring/market.py) are raw external ranks,
    # not re-ranked within `board`, so a real market_rank can exceed
    # len(board) once board is a filtered subset of what those sources rank
    # -- and a sentinel of len(board) + 1 would then be SMALLER than that
    # real value, scrambling a genuinely-ranked player behind the unranked
    # ones. Ranking densely among only the players who carry a real
    # market_rank keeps "rank k" meaning roughly the same thing whether it
    # came from board or a historic ADP feed. Unranked players are appended
    # after the last real rank, not interleaved, so they stay takeable but
    # sit far down the board (an opponent can still draft them).
    has_rank = ranked["market_rank"].notna()
    known = ranked[has_rank].sort_values("market_rank", kind="stable")
    unknown = ranked[~has_rank]
    ranked = pd.concat([known, unknown], ignore_index=True)
    availability = ranked["player_id"].map(_availability(conn))
    # `vor` is the board's headline ranking and the spec's second candidate
    # source. A board built without it (a bare fixture) falls back to the
    # projection, which keeps the second source meaningful rather than
    # collapsing it onto market order.
    vor_col = ranked["vor"] if "vor" in ranked.columns else ranked["proj"]
    return SimPool(
        player_id=ranked["player_id"].to_numpy(),
        norm=ranked["name"].map(_norm_name).to_numpy(),
        position=ranked["position"].to_numpy(),
        adp_rank=np.arange(1, len(ranked) + 1, dtype=float),
        points=ranked["proj"].to_numpy(dtype=float),
        availability=availability.fillna(DEFAULT_AVAILABILITY).to_numpy(dtype=float),
        vor=pd.to_numeric(vor_col, errors="coerce").fillna(-np.inf).to_numpy(dtype=float))


def _live_features(pool, available, overall_pick, roster, recent, settings):
    """Feature matrix for the currently available players, mirroring
    draft_model.feature_matrix exactly -- the fitted coefficients only mean
    anything against the same feature definitions they were fitted on."""
    positions = pool.position[available]
    ranks = pool.adp_rank[available]
    teams = max(settings.teams, 1)
    n = len(positions)
    X = np.zeros((n, len(FEATURE_NAMES)))

    delta = ranks - overall_pick
    X[:, _REACH] = np.maximum(0.0, delta) / teams
    X[:, _FALL] = np.maximum(0.0, -delta) / teams
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
    return X


def _roster_cap(settings) -> dict:
    """Most of each position anyone will carry. Learned coefficients cannot
    express a hard ceiling, so it is imposed as a mask instead."""
    caps = {pos: n + 2 for pos, n in settings.starters.items()}
    caps["QB"] = min(caps.get("QB", 3), 3)
    caps["K"] = 1
    caps["DST"] = 1
    return caps


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


def _greedy_choice(pool, available, roster, settings, caps):
    """My in-rollout policy: the available, cap-legal player who most
    increases roster value. One-ply greedy, which is what makes a rollout
    cheap enough to run thousands of times; the search in Task 11 is what
    looks further ahead.

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
    shortlist = legal[np.argsort(-pool.points[legal])][:GREEDY_CANDIDATES]
    best_idx, best_gain = None, -np.inf
    for i in shortlist:
        pos = pool.position[i]
        gain = roster_value(current + [(pos, pool.points[i], pool.availability[i])],
                            settings) - base
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
    """
    slots = snake_slots(settings.teams, settings.rounds)
    rosters = {slot: {"counts": {}, "indices": []}
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
        recent.insert(0, pos)
    return rosters, recent[:RUN_WINDOW]


def _run_draft(pool, settings, slot_managers, my_slot, taken, betas, rng,
               forced=None, taken_order=None) -> dict:
    """Simulate the remainder of one snake draft and return every slot's
    roster, as `{slot: {"counts": {position: n, ...}, "indices": [pool
    index, ...]}}`.

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

    `rollout` wraps this and returns only my `roster_value` as a scalar --
    Task 11's search depends on that scalar return type -- so this returns
    the full per-slot state instead, for anything (including tests) that
    needs to inspect roster composition beyond my own final number.
    """
    gone = taken.copy()
    caps = _roster_cap(settings)
    rounds = settings.rounds
    slots = snake_slots(settings.teams, rounds)
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
                choice = _greedy_choice(pool, available, roster, settings, caps)
        else:
            beta = betas.get(slot_managers.get(slot))
            if beta is None:
                beta = np.zeros(len(FEATURE_NAMES))
            legal = available[_legal_mask(pool, available, roster["counts"], caps)]
            if len(legal) == 0:
                choice = int(available[0])      # no legal player exists at all
            else:
                X = _live_features(pool, legal, overall_pick, roster["counts"],
                                   recent, settings)
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


def _next_pick_for(settings, my_slot, already) -> int:
    slots = snake_slots(settings.teams, settings.rounds)
    for offset in range(already, len(slots)):
        if slots[offset] == my_slot:
            return offset + 1
    return len(slots) + 1


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
             taken_order=None):
    """Probability each player is still available when my next turn arrives.

    Counted from the same rollout machinery, but stopping at my next pick
    rather than running the draft out -- this is the "who can I wait on"
    number, and it only depends on what happens before my turn.

    `taken_order` carries the same pick attribution `_run_draft` needs, and
    for the same reason: without it every opponent between now and my turn
    resumes with an empty roster, so `need` reads 1.0 everywhere and the
    caps re-arm from zero.
    """
    if taken_order is None and taken.any():
        warnings.warn(
            "draft_sim.survival: players are marked taken but no taken_order "
            "was given; opponents resume with empty rosters, so their needs "
            "and roster caps are wrong", RuntimeWarning, stacklevel=2)
    already = len(taken_order) if taken_order is not None else int(taken.sum())
    target = _next_pick_for(settings, my_slot, already)
    slots = snake_slots(settings.teams, settings.rounds)
    caps = _roster_cap(settings)
    counts = np.zeros(len(pool.player_id))

    for i in range(n_rollouts):
        rng = np.random.default_rng([seed, i])
        gone = taken.copy()
        seeded, seeded_recent = _seed_rosters(pool, settings, taken_order)
        rosters = {slot: state["counts"] for slot, state in seeded.items()}
        recent = list(seeded_recent)
        for offset in range(already, min(target - 1, len(slots))):
            slot = slots[offset]
            available = np.flatnonzero(~gone)
            if len(available) == 0:
                break
            beta = betas.get(slot_managers.get(slot))
            if beta is None:
                beta = np.zeros(len(FEATURE_NAMES))
            X = _live_features(pool, available, offset + 1, rosters[slot],
                               recent, settings)
            scores = X @ beta
            for j, idx in enumerate(available):
                pos = pool.position[idx]
                if rosters[slot].get(pos, 0) >= caps.get(pos, 99):
                    scores[j] = -np.inf
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
    position_of = {pid: i for i, pid in enumerate(pool.player_id)}
    # A drafted player who is no longer on the board (a refresh dropped him)
    # still consumed his turn, so he stays in the order as a None rather than
    # shifting every later pick onto the wrong slot.
    taken_order = [position_of.get(pid) for pid in ordered["player_id"]]
    taken = np.zeros(len(pool.player_id), dtype=bool)
    for idx in taken_order:
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
    if not [m for m in fits if m != "__pooled__"]:
        # With no fitted opponent, every beta is zeros and every opponent
        # draws uniformly over roughly 500 available players. Measured on a
        # 500-player pool: the consensus number one comes back 100% likely to
        # still be there at slot 8, and the top ten average 98.75%. That is
        # not a degraded answer, it is a confident wrong one, and the board
        # merges it with no way to tell.
        raise ValueError(
            "no manager models could be fitted from the imported draft "
            "history -- run `make espn-import` and then `make fit-managers` "
            "first. Without them every opponent would pick uniformly at "
            "random and the simulation's numbers would be meaningless.")
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

    # `reach` is max(0, adp_rank - pick) / teams, so a positive coefficient
    # says "the further past his market rank a player is, the more I want
    # him" -- exp(reach * beta) on a rank-500 player then dominates every
    # other term and the manager drafts the deepest player in the pool. It is
    # a symptom, most likely of build_pool's dense 1..k rank not lining up
    # with the population historic_adp's per-season rank was fitted over, and
    # it makes that manager's simulated behaviour meaningless rather than
    # merely noisy. Cheap to check, and there is nowhere else it surfaces.
    reaching = sorted(m for m, beta in betas.items() if beta[_REACH] > 0)
    if reaching:
        warnings.warn(
            f"draft_sim: fitted reach coefficient is positive for {reaching}; "
            "those managers will be simulated drafting the deepest available "
            "players. Check the adp_rank scale against historic_adp.",
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

    run_id = f"{my_slot}-{n_rollouts}-{seed}-{len(taken_order)}"
    results.insert(0, "run_id", run_id)
    avail.insert(0, "run_id", run_id)
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
    return run_id
