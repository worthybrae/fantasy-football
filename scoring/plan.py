"""What to take now, and what to take at every turn after this one.

Two numbers and one walk.

EDGE is what a player is worth over the man you would get instead. Not over
replacement level (that is the board's `vor`, and it is a season-long
question), and not over the best player left at his position (that counts the
candidate himself, so a certain survivor reads zero however far ahead of his
position he is). It is his projection minus the expected best OTHER player at
his position who is still there at your next turn -- the same
`expected_best_next` walk `scoring/gain.py` has always used, over the
empirical survival probabilities from `scoring/availability.py` instead of a
simulated opponent model.

THE PLAN is a greedy walk over the turns you have left. At each one it scores
everyone who is likely enough to still be there (`THRESHOLD`, lowered to
`FAVOURITE_THRESHOLD` for the account's favourites), takes the best, adds him
to the roster it is carrying, and moves on. Greedy rather than a search: the
input probabilities are measured to two digits, the roster need weights are a
five-value table, and no amount of search makes a plan for pick 118 mean
anything -- what the panel is for is the shape of the next few rounds, which
one pass gets right.

WHY THE SCORE IS NOT THE EDGE. The edge is a claim about the board that can
be checked against the two players it compares, so it is what the room prints.
The score is a ranking quantity and carries the roster in it -- `weight *
(proj - best_next)`, the roster's need applied to what the pick GAINS rather
than to the projection it gains it from, and a position already at its roster
cap is not a candidate at all. Same split, and the same two functions
(`need_kind`, `NEED_WEIGHTS`), as `gain.rank_available` before it.
"""
from __future__ import annotations

import numpy as np

from scoring.availability import AvailabilityTable, availability_at
from scoring.config import NEED_WEIGHTS
from scoring.gain import expected_best_next, fills_slot, need_kind

# The favourites bonus. A favourite is a claim about a player the account
# made in the quiet before the draft, which is a better moment for that
# judgement than the eight seconds on the clock -- but it is a preference,
# not a projection, so it is worth about one tier and no more.
NEED_BONUS = 1.15

# How likely a player has to be to still be there for the plan to plan on
# him. Half is the honest line for "expect him": below it the plan would be
# writing down names that are usually gone by the time you get there.
THRESHOLD = 0.50

# Favourites are planned for on a coin-flip's worth less certainty. The
# account said it wants this player; naming him at the turn he might last to
# is the whole point of having asked.
FAVOURITE_THRESHOLD = 0.35

# At most four reasons per list. More than four bullets on a card is not read
# -- and the ORDER below is the priority, so the four that survive are the
# four that matter most. Pros and cons are capped separately, deliberately:
# capping the combined list would let four pros bury the one con (a bye
# stack, a health meter at 1) that the card exists to warn about, since every
# pro sorts ahead of every con in the spec's order.
MAX_REASONS = 4

# How far ESPN's ADP has to sit from ESPN's own rank before it is worth
# saying. Inside six picks the two lists are telling the same story.
ADP_GAP = 6.0

# A health meter at or below this is worth saying out loud. See `health_level`
# for what the meter is.
HEALTH_CON = 2

# How many already-rostered players on the same bye make a stack. Two: he
# would be the third, which is the week the lineup cannot be filled.
BYE_STACK_MIN = 2

# The board's `career_games_pg` cut points for the five-bar health meter,
# copied from web/src/components/draft/AvailableList.tsx (`HEALTH_CUTS`) so
# the room's picture and the plan's reason cannot disagree. THE BOARD HAS NO
# `health` COLUMN: the meter is derived, and `build_plan`'s `health` argument
# is the derived 1-5 level, not the raw games-per-season.
HEALTH_CUTS = (10.0, 13.0, 15.0, 16.3)


def health_level(games_pg) -> np.ndarray:
    """The board's `career_games_pg` as the room's 1-5 health meter.

    NaN in, NaN out -- a defense and a rookie with no NFL season have no
    games to average, and a fabricated 3 would read as a real measurement.
    """
    games = np.asarray(games_pg, dtype=float)
    level = np.ones(games.shape)
    for cut in HEALTH_CUTS:
        level += (games >= cut)
    return np.where(np.isfinite(games), level, np.nan)


def _best_other(proj: np.ndarray, positions: np.ndarray,
                avail: np.ndarray, present: np.ndarray = None) -> np.ndarray:
    """Expected best survivor at each player's position, EXCLUDING himself.

    `present` is who counts as the field. The plan passes the players it has
    not already struck off: pricing a candidate against a man the plan itself
    has claimed for an earlier turn is pricing him against somebody who will
    not be there.

    A CANDIDATE WITH NO FIELD AT ALL IS PRICED AGAINST HIMSELF, so his edge
    is zero rather than his whole projection. The walk's own answer for an
    empty field is 0 ("the position is gone and waiting costs you
    everything"), which is right when the position's players are all likely
    to be TAKEN and wrong when there simply are none of them on our board --
    and the two are not distinguishable from the arithmetic. A level and a
    differential cannot be compared inside one score: a lone tight end at 120
    would outscore a 220-point running back with two behind him, and the
    moment the plan strikes off the last other player at a position the
    remaining one would leap to the top of the next turn. So a field of
    nobody is not priced.

    `gain.expected_best_next` answers this for a whole position in one walk:
    best first, each player charged the probability that he survives and
    nobody better did. Removing one player changes the field, so the answer
    is genuinely per-player -- but not per-player work. Walking the position
    once forwards and once backwards gives every exclusion at once:

        head[i]  = the walk's value from the players better than i
        prefix[i]= the chance none of them survived
        tail[i]  = the walk's value from the players worse than i, given
                   that it got that far

    and the answer for i is `head[i] + prefix[i] * tail[i]`. The obvious
    alternative -- divide the full walk by `1 - p_i` -- is the same number
    where it is defined and a division by zero for any player certain to
    survive, which on the clock is most of the board.
    """
    out = np.zeros(proj.shape, dtype=float)
    if present is None:
        present = np.ones(proj.shape, dtype=bool)
    for pos in np.unique(positions):
        at = np.flatnonzero((positions == pos) & present)
        alone = np.flatnonzero(positions == pos)
        # Everyone at this position whose field is empty: the players not in
        # `present` at all, and the one man who IS the field.
        out[alone] = proj[alone]
        if at.size == 0:
            continue
        order = at[np.argsort(-proj[at], kind="stable")]
        value = proj[order]
        p = np.clip(avail[order], 0.0, 1.0)
        contribution = value * p
        survives_none = 1.0 - p
        prefix = np.concatenate(([1.0], np.cumprod(survives_none)[:-1]))
        head = np.concatenate(([0.0], np.cumsum(contribution * prefix)[:-1]))
        tail = np.zeros(order.size)
        for i in range(order.size - 2, -1, -1):
            tail[i] = contribution[i + 1] + survives_none[i + 1] * tail[i + 1]
        out[order] = head + prefix * tail
        if order.size == 1:
            # One man in the field is one man with no field of his own.
            out[order] = proj[order]
    return out


def expected_best_excluding(proj, avail, positions, pos, exclude_idx) -> float:
    """The expected best player at `pos` other than `exclude_idx`.

    The single-player form of `_best_other`, straight off
    `gain.expected_best_next` -- kept because it is the definition the edge
    is written against, and the vectorised walk is checked against it.
    """
    proj = np.asarray(proj, dtype=float)
    avail = np.asarray(avail, dtype=float)
    positions = np.asarray([str(p) for p in positions], dtype=object)
    at = np.flatnonzero(positions == str(pos))
    if exclude_idx is not None:
        at = at[at != int(exclude_idx)]
    return float(expected_best_next(proj[at], avail[at]))


def edge_at(proj, positions, avail_next) -> np.ndarray:
    """Points over the next man at the same position, per candidate.

    Positive means taking him now is worth more than waiting for what
    survives to the turn `avail_next` was measured at; negative means the
    position will still be there and this pick is spending an early turn on
    it. A player who is the last of his position on the board reads 0, not
    his whole projection: see `_best_other` on why a field of nobody is not
    priced.
    """
    proj = np.nan_to_num(np.asarray(proj, dtype=float), nan=0.0)
    positions = np.asarray([str(p) for p in positions], dtype=object)
    avail = np.asarray(avail_next, dtype=float)
    return proj - _best_other(proj, positions, avail)


def _number(value):
    """None for anything that is not a real number -- NaN included."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def reasons_for(*, favourite: bool, lasts, at_pick, edge, next_pick,
                position: str, slot, espn_rank, espn_adp, bye, health,
                bye_mates=()) -> tuple[list, list]:
    """The rule-derived pros and cons for one target, in the fixed order.

    `lasts` is a probability, not a percentage. `bye_mates` are the display
    names of the players already on the roster who share his bye week --
    `build_plan` keeps that list as it walks, since it is drafting into the
    same roster.
    """
    pros: list[str] = []
    cons: list[str] = []

    if favourite:
        pros.append("★ favourite")

    lasts = _number(lasts)
    if lasts is not None and at_pick is not None:
        pros.append(f"{lasts * 100:.0f}% still there at pick {int(at_pick)}")

    edge = _number(edge)
    if edge is not None and abs(edge) >= 0.05:
        if edge > 0 and next_pick is not None:
            pros.append(f"+{edge:.1f} pts over the next {position} you'd "
                        f"get at pick {int(next_pick)}")
        elif edge < 0:
            cons.append(f"−{abs(edge):.1f} pts vs waiting for {position}")

    if slot:
        # "fills RB2" names the slot; FLEX and BENCH read better as
        # themselves, and the capped dash is not a reason for anything.
        label = {"FLEX": "flex", "BENCH": "bench",
                 "—": None}.get(str(slot), f"fills {slot}")
        if label:
            pros.append(label)

    rank, adp = _number(espn_rank), _number(espn_adp)
    if rank is not None and adp is not None:
        if adp < rank - ADP_GAP:
            cons.append(f"ADP {adp:.0f} vs ESPN {rank:.0f} — may go earlier")
        elif adp > rank + ADP_GAP:
            pros.append(f"ADP {adp:.0f} vs ESPN {rank:.0f} — may last")

    bye = _number(bye)
    if bye and len(bye_mates) >= BYE_STACK_MIN:
        cons.append(f"bye week {int(bye)} stacks with {bye_mates[0]}")

    health = _number(health)
    if health is not None and health <= HEALTH_CON:
        cons.append(f"health {health:.0f}/5")

    return pros[:MAX_REASONS], cons[:MAX_REASONS]


class _Board:
    """The candidate arrays, converted once instead of once per turn."""

    def __init__(self, *, proj, positions, player_ids, espn_rank, espn_adp,
                 market_rank, byes, health, names, favourites):
        self.ids = [str(p) for p in player_ids]
        self.size = len(self.ids)
        self.proj = np.nan_to_num(np.asarray(proj, dtype=float), nan=0.0)
        self.positions = np.asarray([str(p) for p in positions], dtype=object)
        self.espn_rank = _optional(espn_rank, self.size)
        self.espn_adp = _optional(espn_adp, self.size)
        self.market_rank = _optional(market_rank, self.size)
        self.byes = _optional(byes, self.size)
        self.health = _optional(health, self.size)
        self.names = dict(names or {})
        self.favourites = set(favourites or ())
        self.is_favourite = np.array([pid in self.favourites
                                      for pid in self.ids], dtype=bool)
        self.bonus = np.where(self.is_favourite, NEED_BONUS, 1.0)

    def need(self, settings, roster, turns_left):
        """(weight per player, "cannot be rostered" mask).

        One `need_kind` call per position, not per player: it reads the
        league and the roster, neither of which varies down the board.
        """
        kinds = {pos: need_kind(settings, roster, pos, turns_left)
                 for pos in np.unique(self.positions)}
        weights = np.array([NEED_WEIGHTS[kinds[pos]]
                            for pos in self.positions])
        capped = np.array([kinds[pos] == "capped" for pos in self.positions])
        return weights, capped

    def name(self, i) -> str:
        return self.names.get(self.ids[i], self.ids[i])


def _optional(values, size: int) -> np.ndarray:
    if values is None:
        return np.full(size, np.nan)
    return np.asarray(values, dtype=float)


def _row(board, i, lasts, edge, pros=(), cons=()) -> dict:
    return {"player_id": board.ids[i],
            "lasts_pct": (None if lasts is None
                          else round(float(lasts) * 100, 1)),
            "edge_pts": None if edge is None else round(float(edge), 1),
            "pros": list(pros), "cons": list(cons)}


def build_plan(*, proj, positions, player_ids, espn_rank, espn_adp,
               market_rank, byes, health, roster_counts, settings,
               turns: list, picks_made: int, favourites: set,
               table: AvailabilityTable, names: dict,
               roster_byes: dict | None = None) -> list:
    """A target and two alternates for each of the user's remaining turns.

    `turns` are overall pick numbers (`draft_sim.snake_slots`), `picks_made`
    the pick everything is conditioned on -- every probability in the plan is
    "given he is still on the board right now", which is what makes the last
    turn's numbers mean anything at all.

    `health` is the 1-5 meter (see `health_level`), `byes` the board's `bye`
    column, and `roster_byes` maps the ids ALREADY on the roster to their bye
    weeks, so the bye-stack reason can see the players the plan did not pick
    itself. `names` supplies display names for both.

    A target and its two alternates are struck off for every later turn: a
    plan that names the same three faces at every remaining round is a list
    of the three best players, which the room already shows. They are struck
    out of the PRICING as well as the eligibility -- what a later turn can
    expect at a position does not include the man this plan has already
    spent an earlier turn on.

    `roster_byes` is for the STARTERS already drafted, not the whole roster:
    a bye week is a lineup problem, and two benched players sharing one is
    not a lineup problem. Picking which of the drafted players are starters
    is the caller's job (api/live.py), since it is the caller that knows the
    roster slots.
    """
    board = _Board(proj=proj, positions=positions, player_ids=player_ids,
                   espn_rank=espn_rank, espn_adp=espn_adp,
                   market_rank=market_rank, byes=byes, health=health,
                   names=names, favourites=favourites)
    turns = sorted(int(t) for t in turns)
    if board.size == 0 or not turns:
        return []

    # Every turn's availability up front: turn t needs its own (who is likely
    # to be there) and turn t+1's (what waiting would get you), so computing
    # them in the loop would do each one twice.
    avail = [availability_at(table, board.ids, picks_made, pick,
                             board.espn_adp, board.market_rank,
                             board.positions)
             for pick in turns]

    roster = dict(roster_counts or {})
    stack = _bye_stack(board, roster_byes)

    planned = np.zeros(board.size, dtype=bool)
    plan = []
    for t, pick_no in enumerate(turns):
        here = avail[t]
        next_pick = turns[t + 1] if t + 1 < len(turns) else None
        # Nothing to wait for at the last turn, so the whole projection is
        # what the pick is worth and the edge is not a number anyone can
        # check.
        best_next = (_best_other(board.proj, board.positions, avail[t + 1],
                                 ~planned)
                     if next_pick is not None
                     else np.zeros(board.size))
        turns_left = len(turns) - t
        weights, capped = board.need(settings, roster, turns_left)
        # THE WEIGHT MULTIPLIES THE DIFFERENCE, not the projection. Weighting
        # the level and subtracting an unweighted expectation compares two
        # quantities on different scales: a 300-point quarterback nine points
        # clear of the next one scored 0.35 * 300 - 290.9 = -185.9 while a
        # 173-point receiver two points clear scored -110.2, so the room
        # planned the receiver. What the roster's need scales is what the
        # pick GAINS.
        score = weights * (board.proj - best_next) * board.bonus

        eligible = ~planned & ~capped & ((here >= THRESHOLD)
                                         | (board.is_favourite
                                            & (here >= FAVOURITE_THRESHOLD)))
        row = {"pick_no": pick_no,
               "round": (pick_no - 1) // settings.teams + 1,
               "target": None, "alternates": []}
        plan.append(row)
        if not eligible.any():
            continue

        ranked = np.flatnonzero(eligible)
        ranked = ranked[np.argsort(-score[ranked], kind="stable")][:3]
        target = int(ranked[0])
        edge = (None if next_pick is None
                else board.proj[target] - best_next[target])
        pros, cons = reasons_for(
            favourite=bool(board.is_favourite[target]),
            lasts=here[target], at_pick=pick_no, edge=edge,
            next_pick=next_pick, position=str(board.positions[target]),
            slot=fills_slot(settings, roster, str(board.positions[target]),
                            turns_left),
            espn_rank=board.espn_rank[target], espn_adp=board.espn_adp[target],
            bye=board.byes[target], health=board.health[target],
            bye_mates=stack.get(_bye(board, target), []))
        row["target"] = _row(board, target, here[target], edge, pros, cons)
        row["alternates"] = [
            _row(board, int(i), here[int(i)],
                 None if next_pick is None
                 else board.proj[int(i)] - best_next[int(i)])
            for i in ranked[1:3]]

        # The plan drafts him, so every later turn sees the roster he leaves
        # behind -- which is what makes the walk a plan rather than three
        # copies of the same ranking.
        position = str(board.positions[target])
        roster[position] = roster.get(position, 0) + 1
        bye = _bye(board, target)
        if bye is not None:
            stack.setdefault(bye, []).append(board.name(target))
        planned[ranked] = True
    return plan


def _bye(board, i):
    week = _number(board.byes[i])
    return None if week is None else int(week)


def _bye_stack(board, roster_byes) -> dict:
    """Bye week -> the names already on the roster who are on it."""
    stack: dict = {}
    for pid, bye in (roster_byes or {}).items():
        week = _number(bye)
        if week is not None:
            stack.setdefault(int(week), []).append(
                board.names.get(str(pid), str(pid)))
    return stack


def target_now(*, proj, positions, player_ids, espn_rank, espn_adp,
               market_rank, byes, health, roster_counts, settings,
               turns: list, picks_made: int, favourites: set,
               table: AvailabilityTable, names: dict,
               roster_byes: dict | None = None) -> list:
    """The three cards for the pick on the clock.

    The same score as `build_plan`, with one difference that is not a
    detail: on the clock every available player is there with probability 1,
    so nobody is filtered out for being unlikely to last. The percentage is
    still reported -- against the NEXT turn, which is the pick a reader is
    weighing this one against -- because "2% still there at pick 20" is
    exactly the argument for taking him now.
    """
    board = _Board(proj=proj, positions=positions, player_ids=player_ids,
                   espn_rank=espn_rank, espn_adp=espn_adp,
                   market_rank=market_rank, byes=byes, health=health,
                   names=names, favourites=favourites)
    if board.size == 0:
        return []
    turns = sorted(int(t) for t in turns)
    on_the_clock = picks_made + 1
    later = [t for t in turns if t > on_the_clock]
    next_pick = later[0] if later else None

    if next_pick is None:
        here = None
        best_next = np.zeros(board.size)
    else:
        here = availability_at(table, board.ids, picks_made, next_pick,
                               board.espn_adp, board.market_rank,
                               board.positions)
        best_next = _best_other(board.proj, board.positions, here)

    # My remaining picks INCLUDING this one, the reading `need_kind` wants.
    # None when there are none left to count, which is its "cannot say".
    turns_left = sum(1 for t in turns if t >= on_the_clock) or None
    roster = dict(roster_counts or {})
    weights, capped = board.need(settings, roster, turns_left)
    score = weights * (board.proj - best_next) * board.bonus
    # A player at his position's roster cap has no slot to go in, starter or
    # bench, so he is not a card however the numbers read.
    score = np.where(capped, -np.inf, score)
    stack = _bye_stack(board, roster_byes)

    cards = []
    for i in np.argsort(-score, kind="stable")[:3]:
        if not np.isfinite(score[int(i)]):
            break
        i = int(i)
        edge = None if next_pick is None else board.proj[i] - best_next[i]
        pros, cons = reasons_for(
            favourite=bool(board.is_favourite[i]),
            lasts=None if here is None else here[i], at_pick=next_pick,
            edge=edge, next_pick=next_pick,
            position=str(board.positions[i]),
            slot=fills_slot(settings, roster, str(board.positions[i]),
                            turns_left),
            espn_rank=board.espn_rank[i], espn_adp=board.espn_adp[i],
            bye=board.byes[i], health=board.health[i],
            bye_mates=stack.get(_bye(board, i), []))
        cards.append(_row(board, i, None if here is None else here[i],
                          edge, pros, cons))
    return cards
