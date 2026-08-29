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

THE PLAN is a greedy walk over the turns you have left. At each one it takes
the first man in ESPN's order who is likely enough to still be there
(`THRESHOLD`, lowered to `FAVOURITE_THRESHOLD` for the account's
favourites), whom the roster can still use, and whom the turn after this one
would probably NOT hand you anyway, adds him to the roster it is carrying,
and moves on. Greedy rather than a search: the input probabilities are measured
to two digits, the roster need penalties are a four-value table, and no
amount of search makes a plan for pick 118 mean anything -- what the panel is
for is the shape of the next few rounds, which one pass gets right.

WHY THE EDGE DOES NOT RANK ANYTHING. The edge is a claim about the board
that can be checked against the two players it compares, so it is what the
room prints -- and it is a raw point difference, which is exactly why it
cannot also be the order. Points are not comparable across positions: a
quarterback scores half again what a receiver does and his position falls off
a cliff, so ranking by "points over the next man at my position" puts a
quarterback at the top of a one-quarterback league's first round. The room
recommended Josh Allen, ESPN's 26th player, over Christian McCaffrey at pick
6, with the edge as its whole argument.

THE ORDER IS ESPN'S ORDER (spec decision 1), moved by four things and
nothing else:

    priority = espn_rank + need_penalty - favourite_bonus
                        + waitable + reach_penalty         (lowest first)

`need_penalty` is `need_kind`'s word for the roster, in RANKS rather than a
multiplier -- a flex body is worth about four places, a bench body twenty,
and a kicker before the round anybody takes kickers is worth two hundred,
which is to say never. A position at its roster cap is not a candidate at
all. `favourite_bonus` is eight ranks, about a tier: the account's own
judgement, made in the quiet before the draft, is worth a tier and not a
round. A player ESPN does not rank uses the consensus (`market_rank`); with
neither he sorts last, by projection.

CAN YOU WAIT? That is the question the first three miss and the one that
separates a good pick from a good player: a pick is a good pick when the man
is ranked well AND is unlikely to be there when you come back. So a
candidate the table expects to still be sitting there at your NEXT turn is
charged `WAITABLE` ranks and the pick goes to somebody who will not be.
Fifteen is a round and a half of an eight-team draft -- more than a tier,
because a player you can have at either of two turns is not a decision at
the first of them. The line is the same one the plan plans on
(`THRESHOLD`, `FAVOURITE_THRESHOLD` for a starred player), and it is a step
rather than a slope on purpose: "will he be there" is a yes or no asked on a
clock, the counts behind it are two digits wide, and a smooth function of
them would order two players by the third digit of a probability.

REACHING IS THE OTHER HALF. ESPN's rank says how good a player is; ESPN's
ADP says when the room actually takes him. Taking a man twenty picks before
the room does is paying for something the next turn would have handed you,
so `reach_penalty` charges half a rank per pick of the gap past
`REACH_FREE`. Six picks are free because a rank and an ADP inside a round of
each other are telling the same story (`ADP_GAP` draws that same line for
the reasons); a ten-pick reach costs two ranks and a thirty-pick reach
twelve. Josh Allen, ESPN 26 with an ADP of 21, at pick 11 of an eight-team
draft: two ranks of reach on top of a rank that was already eighteen behind.

WHERE THE DROP-OFF GETS ITS SAY. Two players two ranks apart are not a
decision; the position behind them is. So a run of candidates whose
priorities sit within `TIE_RANKS` of each other is settled by the biggest
EDGE, and outside that tie nobody wins on the edge however large it is.
The window used to be twelve ranks, which is a tier rather than a tie, and a
quarterback's raw-point cliff ate a whole round of the draft with it. This
is the whole of the edge's authority: it breaks a tie, it does not cross a
tier.
"""
from __future__ import annotations

import numpy as np

from scoring.availability import AvailabilityTable, availability_at
from scoring.gain import expected_best_next, fills_slot, need_kind

# The favourites bonus, IN RANKS. A favourite is a claim about a player the
# account made in the quiet before the draft, which is a better moment for
# that judgement than the eight seconds on the clock -- but it is a
# preference, not a projection, so it is worth about one tier and no more.
# Eight places: a starred player at ESPN 15 beats a stranger at 8 and still
# loses to a stranger at 6.
#
# IT ONLY EVER HELPS. A subtraction from a rank cannot turn round on a
# player the way the old multiplier on a signed score could, so a favourite
# is never worse off for being one.
FAVOURITE_BONUS = 8.0

# What the roster does to ESPN's order, in ranks. `need_kind`'s five words
# (`scoring/gain.py`), the same five `NEED_WEIGHTS` scores for the
# simulation -- here as places rather than a multiplier, because the thing
# being moved is a rank.
#
# A starter slot is ESPN's order untouched: the list is already sorted by
# who is worth taking, and an open starting job is the ordinary case. FLEX
# is most of a tier back, BENCH a round and a half -- a bench body is a real
# pick, just never before a man who would start. DEFERRED is a kicker or a
# defense before the round anybody takes one, and 200 is "not on this board":
# it is a starter slot that WILL be filled, so it is not capped, but naming
# a kicker in round 3 is the complaint that put `need_kind`'s deferred case
# there in the first place.
NEED_PENALTY = {"starter": 0.0, "flex": 4.0, "bench": 20.0, "deferred": 200.0}

# What a player you can probably still have at your next turn costs, IN
# RANKS. See the module docstring: this is the plan's answer to "can you
# wait?", and it is the difference between a good player and a good pick.
# Fifteen is a round and a half of an eight-team draft -- deliberately more
# than the favourites bonus and more than a bench slot, because a man
# available at either of two turns is not a decision at the first of them.
WAITABLE = 15.0

# How far apart two priorities can be and still be a tie the drop-off
# settles. Two ranks is ESPN saying "these are the same player"; the twelve
# this replaced was a tier, and a tier is not the edge's to reorder.
TIE_RANKS = 2.0

# The reach: how many picks earlier than his ESPN ADP a turn is. The first
# `REACH_FREE` are free -- a rank and an ADP inside a round of each other
# are telling the same story -- and every pick past that costs `REACH_RATE`
# ranks. Half a rank a pick: a ten-pick reach is worth two ranks, which
# nudges, and a thirty-pick reach is worth twelve, which decides.
REACH_FREE = 6.0
REACH_RATE = 0.5

# How likely a player has to be to still be there for the plan to plan on
# him. Half is the honest line for "expect him": below it the plan would be
# writing down names that are usually gone by the time you get there.
#
# IT IS ALSO THE WAITABLE LINE, asked one turn later: "likely enough to be
# there for me to plan on him at this turn" and "likely enough to be there
# for me to wait for him past it" are the same question about the same
# table, and two different numbers for it would only be two numbers to
# explain.
THRESHOLD = 0.50

# Favourites are planned for on a coin-flip's worth less certainty. The
# account said it wants this player; naming him at the turn he might last to
# is the whole point of having asked. The same allowance runs the other way:
# a favourite is only "waitable" when he is really quite safe, so a starred
# player on the edge of going is taken rather than gambled on.
FAVOURITE_THRESHOLD = 0.35

# How many reasons a card prints, per list. The ORDER below is the priority,
# so what survives the cap is what matters most.
#
# PROS ARE FIVE AND CONS ARE FOUR. Pros and cons are capped separately,
# deliberately: capping the combined list would let the pros bury the one con
# (a bye stack, a health meter at 1) that the card exists to warn about,
# since every pro sorts ahead of every con in the spec's order. Four pros was
# one short once the rank became a reason -- a starred target spent its four
# on the star, the rank, the chance and the edge, and dropped both the slot
# it fills and the drop-off line that says why it was taken over a
# higher-ranked name, which is to say it dropped the two that explain the
# pick. Five is the room's own limit on a card that is still read at a
# glance; the cons stay at four, where nothing has ever been cut.
MAX_PROS = 5
MAX_CONS = 4

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
                bye_mates=(), drop_off: bool = False, gone_next=None,
                waitable: bool = False, reach=None,
                reach_at=None) -> tuple[list, list]:
    """The rule-derived pros and cons for one target, in the fixed order.

    `lasts` is a probability, not a percentage. `bye_mates` are the display
    names of the players already on the roster who share his bye week --
    `build_plan` keeps that list as it walks, since it is drafting into the
    same roster. `drop_off` is True when this player's edge is the biggest
    on the board, which is the card's answer to "why him and not the man
    ESPN ranks above him".

    `gone_next` is 1 - P(still there at the turn AFTER this one), `waitable`
    whether that cleared the line the priority charges `WAITABLE` for, and
    `reach`/`reach_at` how many picks early `reach_at` is against his ADP.
    The three of them are the priority's own arithmetic said out loud: the
    card has to be able to explain a pick it made for a new reason.

    THE WAIT AND THE REACH LEAD THE CONS because they are the two that argue
    against the pick itself rather than against a detail of it. A card that
    runs out of room has said the thing worth saying.
    """
    pros: list[str] = []
    cons: list[str] = []

    rank, adp = _number(espn_rank), _number(espn_adp)

    if favourite:
        pros.append("★ favourite")

    # WHY THE RANK IS A REASON. It is the first thing the order is built
    # from (`_Board.priority`), so a card that did not say it was arguing
    # for its pick with everything except the argument.
    if rank is not None:
        pros.append(f"ESPN's #{rank:.0f} overall")

    lasts = _number(lasts)
    gone = _number(gone_next)
    likely_gone = gone is not None and gone >= THRESHOLD
    could_wait = bool(waitable) and gone is not None and next_pick is not None
    # THE SURVIVAL LINE IS SAID ONCE. `target_now` measures "still there" at
    # the very turn the waiting is about, so the bare percentage and the
    # line that judges it would be the same number twice -- once as a pro
    # and once as a con, which reads as a contradiction rather than a
    # reason. The judged line wins where they collide: it is the one that
    # says what to do about the number. `build_plan`'s two picks are
    # genuinely different picks, so both are printed there.
    same_pick = (at_pick is not None and next_pick is not None
                 and int(at_pick) == int(next_pick))
    if (lasts is not None and at_pick is not None
            and not (same_pick and (likely_gone or could_wait))):
        pros.append(f"{lasts * 100:.0f}% still there at pick {int(at_pick)}")

    if likely_gone:
        pros.append(f"likely gone before your next pick ({gone * 100:.0f} %)")

    if could_wait:
        cons.append(f"you could probably wait — {(1 - gone) * 100:.0f} % "
                    f"still there at pick {int(next_pick)}")

    reach = _number(reach)
    if (reach is not None and reach > REACH_FREE
            and adp is not None and reach_at is not None):
        cons.append(f"reach: ADP {adp:.0f} at pick {int(reach_at)}")

    # THE EDGE IS PRICED AT `next_pick`, NOT AT THIS TURN -- it is what
    # taking him now is worth over the man you would get if you waited, so
    # the pick it names is the turn after the one the card is for. Naming
    # this turn's pick was the caption bug the room shipped with.
    edge = _number(edge)
    if edge is not None and abs(edge) >= 0.05:
        at = None if next_pick is None else f" at pick {int(next_pick)}"
        if edge > 0 and next_pick is not None:
            pros.append(f"+{edge:.1f} pts over the next {position} you'd "
                        f"get{at}")
        elif edge < 0:
            cons.append(f"−{abs(edge):.1f} pts vs waiting for "
                        f"{position}{at or ''}")

    # WHY HE IS AHEAD OF A PLAYER ESPN RANKS HIGHER: no position on the
    # board falls off harder than his before the next turn. Said only when
    # that is true of him, since it is the one thing the edge is allowed to
    # claim (see `_ranked`).
    if drop_off and next_pick is not None:
        pros.append(f"biggest drop-off at {position} before "
                    f"pick {int(next_pick)}")

    if slot:
        # "fills RB2" names the slot; FLEX and BENCH read better as
        # themselves, and the capped dash is not a reason for anything.
        label = {"FLEX": "flex", "BENCH": "bench",
                 "—": None}.get(str(slot), f"fills {slot}")
        if label:
            pros.append(label)

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

    return pros[:MAX_PROS], cons[:MAX_CONS]


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

    def need_penalty(self, settings, roster, turns_left):
        """(rank penalty per player, "cannot be rostered" mask).

        One `need_kind` call per position, not per player: it reads the
        league and the roster, neither of which varies down the board.
        """
        kinds = {pos: need_kind(settings, roster, pos, turns_left)
                 for pos in np.unique(self.positions)}
        penalty = np.array([NEED_PENALTY.get(kinds[pos], 0.0)
                            for pos in self.positions])
        capped = np.array([kinds[pos] == "capped" for pos in self.positions])
        return penalty, capped

    def waitable(self, gone_next) -> np.ndarray:
        """`WAITABLE` ranks for everybody the next turn would hand you anyway.

        `gone_next` is 1 - P(still there at the turn AFTER this one | now),
        so a small number means "you can come back for him". The line is
        `THRESHOLD`, and `FAVOURITE_THRESHOLD` for a starred player -- the
        same two the plan's own eligibility uses, one turn later.
        """
        gone = np.asarray(gone_next, dtype=float)
        line = np.where(self.is_favourite, FAVOURITE_THRESHOLD, THRESHOLD)
        return WAITABLE * (gone < line)

    def reach(self, pick_no) -> np.ndarray:
        """How many picks earlier than ESPN's ADP this turn is.

        Zero for a player ESPN has no usable ADP for: a reach is a claim
        about when the room takes him, and without an ADP there is nothing
        to be early of. Zero, too, for the ordinary case of a turn at or
        past his ADP -- taking a man the room would already have taken is
        not reaching, it is being late.
        """
        gap = np.where(np.isfinite(self.espn_adp),
                       self.espn_adp - float(pick_no), 0.0)
        return np.maximum(0.0, gap)

    def priority(self, penalty: np.ndarray, waitable: np.ndarray,
                 reach: np.ndarray) -> np.ndarray:
        """ESPN's order as the plan reads it. LOWEST FIRST.

        `espn_rank + need_penalty - favourite_bonus + waitable +
        reach_penalty`, with the consensus standing in for a player ESPN
        does not rank -- the same fallback the room's own list order uses
        (spec section 1), so the plan and the table cannot disagree about
        who is ahead of whom. A player with neither rank is infinite here
        and sorts last; `_by_priority` orders those few by projection, which
        is the only thing left to say about them.

        `waitable` comes in already in ranks (`waitable`); `reach` comes in
        as picks and is charged here, so the free `REACH_FREE` and the rate
        live in one place and the reasons can read the raw number.
        """
        base = np.where(np.isfinite(self.espn_rank),
                        self.espn_rank, self.market_rank)
        rank = np.where(np.isfinite(base), base, np.inf)
        return (rank + penalty - FAVOURITE_BONUS * self.is_favourite
                + waitable
                + REACH_RATE * np.maximum(0.0, reach - REACH_FREE))

    def name(self, i) -> str:
        return self.names.get(self.ids[i], self.ids[i])


def _optional(values, size: int) -> np.ndarray:
    if values is None:
        return np.full(size, np.nan)
    return np.asarray(values, dtype=float)


def _by_priority(priority, proj, edge, keep) -> np.ndarray:
    """The kept indices, best first.

    `priority` decides it (see `_Board.priority`); the rest is tie-breaking,
    in order: the players with no rank at all fall to the back and sort by
    projection among themselves, then the bigger EDGE takes it -- which is
    where the old score ended up, as the tie-break it was always fit to be
    -- and the board's own order settles the rest so a poll is stable.
    """
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return idx
    unranked = ~np.isfinite(priority[idx])
    order = np.lexsort((idx, -np.asarray(edge, dtype=float)[idx],
                        np.where(unranked, -np.asarray(proj)[idx], 0.0),
                        priority[idx]))
    return idx[order]


def _ranked(priority, proj, edge, keep, drop_off: bool = True) -> np.ndarray:
    """The kept indices in the order the room reads them.

    `_by_priority` first, then the drop-off breaks the ties in it: a RUN of
    players whose priorities are all within `TIE_RANKS` of the first of them
    is reordered by EDGE, biggest first, and the next run starts where that
    stops. A tie on the edge falls back to the priority order, so a board
    where the drop-off says nothing is ESPN's list exactly.

    THE RUN IS MEASURED FROM ITS OWN LEADER, not from the man before it, so
    a long ladder of players a rank apart cannot chain into one group and
    let the edge reorder the whole board two ranks at a time.

    `drop_off` is False at a turn with nothing after it: the edge there is
    not a number (there is no next pick to price against, so the walk hands
    back the whole projection), and sorting even a tie by raw projection
    across positions is the mistake this module exists to stop making.
    """
    idx = _by_priority(priority, proj, edge, keep)
    if not drop_off or idx.size == 0:
        return idx
    edge = np.asarray(edge, dtype=float)
    ordered = np.asarray(priority, dtype=float)[idx]
    runs, start = [], 0
    while start < idx.size:
        # The unranked sort last and by projection (`_by_priority`); an
        # infinite priority is not within two of anything, itself included.
        if not np.isfinite(ordered[start]):
            runs.append(idx[start:])
            break
        end = start + 1
        while (end < idx.size and np.isfinite(ordered[end])
               and ordered[end] <= ordered[start] + TIE_RANKS):
            end += 1
        run = idx[start:end]
        runs.append(run[np.argsort(-edge[run], kind="stable")])
        start = end
    return np.concatenate(runs)


def _row(board, i, lasts, edge, edge_at=None, pros=(), cons=()) -> dict:
    return {"player_id": board.ids[i],
            "lasts_pct": (None if lasts is None
                          else round(float(lasts) * 100, 1)),
            "edge_pts": None if edge is None else round(float(edge), 1),
            # The pick the edge is priced at -- the turn AFTER the one this
            # row belongs to, since the edge is what taking him now beats.
            # Null when there is no later turn to wait for.
            "edge_at_pick": None if edge_at is None else int(edge_at),
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
        gain = board.proj - best_next
        penalty, capped = board.need_penalty(settings, roster, turns_left)
        # CAN YOU WAIT FOR HIM? P(still there at the turn after this one),
        # from the same table, is already computed as `avail[t + 1]`. At the
        # last turn there is nothing to wait for, so nobody is waitable:
        # every man on the board is gone by a pick that never comes.
        gone_next = (1.0 - avail[t + 1] if next_pick is not None
                     else np.ones(board.size))
        wait = board.waitable(gone_next)
        reach = board.reach(pick_no)
        # ESPN'S ORDER, MOVED BY THE ROSTER, THE STARS, THE WAIT AND THE
        # REACH, then its ties broken by the drop-off. `gain` is a raw point
        # difference and comparing one across positions ranks quarterbacks
        # (see the module docstring), so it settles a tie and never the
        # board.
        priority = board.priority(penalty, wait, reach)

        eligible = ~planned & ~capped & ((here >= THRESHOLD)
                                         | (board.is_favourite
                                            & (here >= FAVOURITE_THRESHOLD)))
        row = {"pick_no": pick_no,
               "round": (pick_no - 1) // settings.teams + 1,
               "target": None, "alternates": []}
        plan.append(row)
        if not eligible.any():
            continue

        ranked = _ranked(priority, board.proj, gain, eligible,
                         drop_off=next_pick is not None)[:3]
        target = int(ranked[0])
        edge = (None if next_pick is None
                else board.proj[target] - best_next[target])
        # NOBODY'S POSITION FALLS OFF HARDER THAN HIS before the next turn,
        # which is the card's answer for a man ESPN ranks below somebody
        # else on the list. Claimed only when it is true of the whole
        # eligible board, and only when there is a drop-off to have.
        drop_off = bool(edge is not None and edge >= 0.05
                        and gain[target] >= np.max(gain[eligible]))
        pros, cons = reasons_for(
            favourite=bool(board.is_favourite[target]),
            lasts=here[target], at_pick=pick_no, edge=edge,
            next_pick=next_pick, position=str(board.positions[target]),
            drop_off=drop_off, gone_next=gone_next[target],
            waitable=bool(wait[target] > 0), reach=float(reach[target]),
            reach_at=pick_no,
            slot=fills_slot(settings, roster, str(board.positions[target]),
                            turns_left),
            espn_rank=board.espn_rank[target], espn_adp=board.espn_adp[target],
            bye=board.byes[target], health=board.health[target],
            bye_mates=stack.get(_bye(board, target), []))
        row["target"] = _row(board, target, here[target], edge, next_pick,
                             pros, cons)
        row["alternates"] = [
            _row(board, int(i), here[int(i)],
                 None if next_pick is None
                 else board.proj[int(i)] - best_next[int(i)], next_pick)
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

    The same priority as `build_plan`, with one difference that is not a
    detail: on the clock every available player is there with probability 1,
    so nobody is filtered out for being unlikely to last. The percentage is
    still reported -- against the NEXT turn, which is the pick a reader is
    weighing this one against -- because "2% still there at pick 20" is
    exactly the argument for taking him now. That is also the number the
    wait is charged on: on the clock "will he last" and "can I wait" are one
    question, and the card says it once (see `reasons_for`).
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
    gain = board.proj - best_next
    penalty, capped = board.need_penalty(settings, roster, turns_left)
    # THE WAIT IS THE SAME QUESTION ON THE CLOCK: `here` is already measured
    # at my next turn, so 1 - it is exactly "gone before I am back". The
    # reach is measured at the pick I am actually making, which is this one
    # and not the turn the percentages name.
    gone_next = (np.ones(board.size) if here is None else 1.0 - here)
    wait = board.waitable(gone_next)
    reach = board.reach(on_the_clock)
    priority = board.priority(penalty, wait, reach)
    # A player at his position's roster cap has no slot to go in, starter or
    # bench, so he is not a card however the numbers read.
    stack = _bye_stack(board, roster_byes)

    cards = []
    order = _ranked(priority, board.proj, gain, ~capped,
                    drop_off=next_pick is not None)
    biggest = np.max(gain[~capped]) if order.size else np.inf
    for i in order[:3]:
        i = int(i)
        edge = None if next_pick is None else board.proj[i] - best_next[i]
        pros, cons = reasons_for(
            favourite=bool(board.is_favourite[i]),
            lasts=None if here is None else here[i], at_pick=next_pick,
            edge=edge, next_pick=next_pick,
            position=str(board.positions[i]),
            drop_off=bool(i == int(order[0]) and edge is not None
                          and edge >= 0.05 and gain[i] >= biggest),
            gone_next=gone_next[i], waitable=bool(wait[i] > 0),
            reach=float(reach[i]), reach_at=on_the_clock,
            slot=fills_slot(settings, roster, str(board.positions[i]),
                            turns_left),
            espn_rank=board.espn_rank[i], espn_adp=board.espn_adp[i],
            bye=board.byes[i], health=board.health[i],
            bye_mates=stack.get(_bye(board, i), []))
        cards.append(_row(board, i, None if here is None else here[i],
                          edge, next_pick, pros, cons))
    return cards
