"""Will he still be there? Measured, not simulated.

The room used to answer this with `draft_sim.survival`: four hundred rollouts
of a fitted opponent model, a second of numpy per pick per room. This module
answers it by counting instead. `data/draft_corpus.duckdb` holds every mock
the farm has played -- 854 drafts and 109k picks today -- and the question
"was he still on the board at pick N" is a ratio of two draft counts, which
is a table lookup once the counts are precomputed.

EVERY PICK COUNTS, autodrafts included. "Was he still there" does not care
who took him or why, which is the same rule api/market.py's archive page
states in its own docstring. And `draft_log_pool` is the denominator, never
the draft count: a player who was not in a draft's pool at all must not be
credited with having lasted through it, or every player added to ESPN's board
late in the summer reads as unkillable.

CONDITIONED ON NOW, which is the whole reason this is not just a survival
curve. By the time the room asks, `k` picks have already been made and the
player is still on the board -- so the drafts where he went before `k` are
not evidence about him any more, they are evidence about a world that did not
happen. The answer is the share of the drafts where he lasted past `k` in
which he also survived every pick before `n`.

THREE PLACES THE COUNTS CANNOT ANSWER, and all three fall through to a
parametric curve fitted from the same rows (see `_fit_curve`):

  * A player the corpus has never seen. ESPN's board is longer than the pool
    the farm's mocks are drafted from.
  * A conditioned denominator under `MIN_DRAFTS`.
  * A pick deeper than the counts can speak to, FOR A PLAYER THE CORPUS
    STOPPED WATCHING. The 8-team drafts are 16 rounds, so no pick past 128
    exists in them and a player still on the board at the end of every
    recorded draft would otherwise count as having lasted through pick 150 --
    which on a 12-team board at a round-13 turn printed 100% for well over
    half the list.

    "How deep the counts speak" is the SHALLOWEST SHAPE's depth, not the
    deepest pick in the file. Once the farm records a second shape the two
    are different numbers, and taking the second would let one 12-team draft
    turn 854 8-team drafts that merely stopped at 128 into evidence about
    pick 150 (see `_shape_counts`).

    That is right-censoring, and it is not the same as having no evidence. A
    player every one of his pooled drafts TOOK inside that depth has complete
    evidence: "he is gone by 129" is a fact about him. Sending him to the
    curve as well made the number jump upwards at the seam -- kickers at ESPN
    rank 240 read 0.00 at pick 128 and 1.00 at pick 129, and six defenses
    read 1% at pick 126 and 50% at 139 -- so only the censored go there, and
    the counts carry them as far as the corpus goes before the curve takes
    over (see `availability_at`).

A fallback is not a nicety here: without it those cases return 0/0 and 1/1,
i.e. "certainly gone" and "certainly there", which are the two most confident
things this module could possibly say.

CONDITIONED ON THE SHAPE, once there is enough of one. A draft's shape is
`(teams, format)` -- see `pipeline.draft_log.draft_format` -- and it changes
the answer twice over: twelve teams means twelve picks a round rather than
eight, so pick 30 is early rather than late, and PPR means a receiver goes
where standard scoring leaves him. The counts are therefore kept per shape
as well as pooled, and a room reading its own shape's counts gets them as
soon as that shape holds `MIN_SHAPE_DRAFTS` drafts. Below that it reads the
pooled counts, which is what every room read before the farm went looking
for other shapes -- a slightly wrong answer from 854 drafts beats a right
one from nine. The parametric curve stays pooled at every size: it is fitted
across ADP buckets rather than per player, and splitting it five ways would
empty the buckets it needs.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import ndtr

from pipeline import draft_log as dl

# The pick axis. Overall pick number, never scaled by team count: ADP is
# already in overall picks, and a 12-team league's round 3 is picks 25-36 on
# the same axis the 8-team corpus was recorded on. 300 covers a 12-team
# 16-round draft (192) with room, at 301 int64 columns per player.
MAX_PICK = 300

# Below this many drafts a ratio is not a probability. Twelve drafts that all
# went one way would print 0% or 100% with the same confidence as eight
# hundred, and the fallback -- which at least knows what his ADP is -- is the
# better answer.
MIN_DRAFTS = 25

# How many drafts of ONE SHAPE it takes before that shape answers for itself
# rather than borrowing the pooled corpus. Above MIN_DRAFTS, deliberately:
# MIN_DRAFTS is the floor on one player's conditioned denominator inside a
# table that has already been chosen, while this is the floor on choosing the
# table at all, and choosing it wrongly costs every player on the board. Sixty
# drafts is roughly where a mid-round player's own denominator clears
# MIN_DRAFTS in the shape rather than only in the pool -- so the shape's table
# starts being used at about the point it can answer with counts instead of
# handing most of the board to the curve.
MIN_SHAPE_DRAFTS = 60

# How many drafts of a shape it takes before that shape gets a vote on how
# deep the POOLED counts can speak (see `_shape_counts`). The pooled depth is
# the shallowest shape's, which is right when a shape's depth is a fact about
# the shape and wrong when it is a fact about one room: a single 12-team mock
# that fell over at pick 30 -- the socket dropped, everybody left -- would
# otherwise pull every player's depth down to 30 and hand most of the board
# to the parametric curve.
#
# Five, the same size of evidence `MIN_COMPLETE_DRAFTS` asks for and for the
# same kind of reason: this is not a ratio anybody prints, it is the bar for
# "this is where drafts of this shape end" being a statement about the shape
# rather than about a bad night. Deliberately a different constant, since
# what they bound has nothing else in common.
MIN_DEPTH_DRAFTS = 5

# Width of the ADP buckets the fallback curve is fitted in. Narrow enough
# that the top of the board (where one pick of ADP is a real difference in
# when a player goes) is not averaged flat, wide enough that every bucket in
# the measured range holds thousands of pick rows.
ADP_BUCKET = 8

# How many DISTINCT players a bucket needs before it is a bucket rather than
# a player. Rows are not the test: one player pooled in 854 drafts brings 854
# rows of his own habits, and a "curve" fitted from him says nothing about
# the next player who happens to share his ADP.
MIN_BUCKET_PLAYERS = 8

# The spread to use when a player's bucket was never fitted -- his own ADP is
# then the centre and this is the width. 12 picks is the pooled within-bucket
# standard deviation measured on the real corpus (11.6, rounded up), so an
# unmeasured player is given the same uncertainty as a measured one rather
# than a made-up one.
FALLBACK_SIGMA = 12.0

# No bucket is allowed to be a step function. The first bucket on the real
# corpus (ESPN ranks 0-7) has a standard deviation of 2.5 picks, and a
# hand-built test corpus can produce less than one; at sigma near zero the
# normal tail flips from 1 to 0 across a single pick, which is a claim no
# sample of drafts supports.
MIN_SIGMA = 1.0

# How many drafts it takes before "every one of them took him by pick P" is
# a fact rather than a coincidence. Far below MIN_DRAFTS, and deliberately:
# MIN_DRAFTS is the bar for trusting a RATIO -- 12 drafts out of 40 is a
# number nobody should print -- while this is the bar for a statement with no
# ratio in it at all. Five drafts that unanimously took a player before pick
# 21 say he is not there at pick 30; the alternative is what the room did
# before, which was to read a kicker's ESPN rank of 240 off the curve and
# print him as a certainty.
MIN_COMPLETE_DRAFTS = 5

# A bucket is only fitted from the picks that actually happened, so it can
# only be trusted where most of the players in it were actually taken. The
# corpus is 8-team 16-round -- 128 picks over a pool of 250 -- so past about
# ESPN rank 130 the players in a bucket are mostly undrafted, and the mean of
# the few picks that did happen describes the drafters who reached, not the
# bucket. Measured taken-share by bucket on the real corpus: 100% through
# rank 88, 90-99% to rank 128, then 47%, 55%, 26%, 26%, 17% and below. Half
# is the line: above it the mean is about where the bucket goes, below it the
# bucket is dropped and a player's own ADP is used as his centre instead --
# which for a rank-200 player is the difference between "gone by pick 80" and
# "still there", and he IS still there.
MIN_BUCKET_TAKEN = 0.5

# THE CURVE IS FITTED PER POSITION GROUP. Kickers and defenses are drafted in
# every single mock, at ESPN ranks in the 240s, which is where the skill
# players nobody drafts also live. Fitted blind to position those rows own
# buckets 17, 30 and 31 (mu 106-123 on the real corpus), and a wide receiver
# at ADP 244 read 38% still-there while his neighbours at 238 and 260 read
# 100%. Three groups, because those are the three populations: a kicker's ADP
# means "round 15", a defense's means "round 14", and everyone else's means
# roughly the pick he goes at.
SKILL = "skill"
_GROUPS = {"K": "K", "PK": "K", "DST": "DST", "D/ST": "DST", "DEF": "DST"}


def position_group(position) -> str:
    """Which curve a position is read off: "K", "DST" or `SKILL`."""
    if position is None:
        return SKILL
    return _GROUPS.get(str(position).strip().upper(), SKILL)


class ShapeCounts(NamedTuple):
    """One shape's own counts: `pooled` and `taken_by` exactly as the table's
    pooled arrays, over the SAME player index, plus how many drafts of that
    shape the corpus holds.

    `pooled` and `taken_by` are None for a shape under `MIN_SHAPE_DRAFTS`:
    the count is still worth carrying (it is what says how close the shape is
    to answering for itself, and what `api/seo.py` will read to decide
    whether a shape gets a page), the arrays are not worth the megabyte.

    `max_pick_observed` IS PER SHAPE and not a copy of the table's. An 8-team
    room stops at pick 128 and a 12-team one runs to 192, so reading an
    8-team shape's counts at pick 150 against a 12-team depth would report
    everybody still on the board at 128 as still there at 150. That is the
    censoring the module docstring is about, arrived at from the shape side --
    and it is why the POOLED depth is the shallowest shape's rather than the
    deepest pick in the file (see `_shape_counts`).

    It is carried for every shape, including the ones under
    `MIN_SHAPE_DRAFTS` that hold no arrays: a handful of 12-team drafts is
    enough to make the pooled counts censored past 128, and the pooled depth
    has to know that long before the shape is big enough to answer for
    itself.

    IT IS THE MEDIAN of the shape's drafts' last picks, not the deepest pick
    anybody in it ever made. A shape's depth is meant to say where drafts of
    this shape END, and one room that ran to 192 while the rest stopped at
    128 is not evidence that they all did -- nor is one room that fell over
    at pick 30 evidence that none of them got past it. The median moves only
    when most of the shape moves.
    """
    pooled: np.ndarray | None
    taken_by: np.ndarray | None
    drafts: int
    max_pick_observed: int = 0


@dataclass
class AvailabilityTable:
    """The corpus, counted once, in the shape the room asks questions in.

    `taken_by[i, p]` is how many of player i's pooled drafts had taken him by
    overall pick p -- monotone in p by construction, and never larger than
    `pooled[i]`, because a pick only counts when the same draft's pool row
    exists (see `load_table`'s join).

    `adp_curve` is `group -> {bucket: (mu, sigma)}` with the group from
    `position_group`. IT IS FITTED ON THE POOL SNAPSHOT'S OWN NUMBERS --
    `draft_log_pool.espn_rank`, falling back to `adp_rank` -- and QUERIED
    with the board's ESPN ADP (consensus rank when ADP is blank). Those are
    not the same column: one is where ESPN ranked him on the day the mock was
    played, the other is where ESPN's lobby says he is going now. They are
    the same KIND of number, both roughly "the pick he is expected to go at",
    which is what makes the bucket a bucket -- but a change to either side
    (ESPN publishing a rank rather than an ADP, the pool storing something
    else) breaks the correspondence quietly, so it is written down here.

    `max_pick_observed` is how deep the POOLED counts can be trusted: the
    shallowest shape's own deepest pick, not the deepest pick in the file.
    See `_shape_counts` for why the two are different and what goes wrong
    when the second is used. Past it the counts hold no evidence at all --
    see the module docstring.
    """
    player_ids: np.ndarray
    pooled: np.ndarray
    taken_by: np.ndarray
    adp_curve: dict          # group -> {bucket: (mu, sigma)}
    corpus_mtime: float = 0.0
    max_pick_observed: int = 0
    # (teams, format) -> ShapeCounts. Empty when the corpus holds no draft
    # whose shape can be read, which is what every hand-built table in a test
    # and every reader that does not ask about a shape sees.
    shapes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.index = {str(pid): i for i, pid in enumerate(self.player_ids)}
        # Each group's curve as two arrays indexed by bucket, so a whole
        # board's worth of fallbacks is one gather rather than 250 dict
        # lookups. NaN is "this bucket was not fitted" and is what
        # `availability_at` replaces with the player's own ADP.
        self.curves = {}
        for group, curve in self.adp_curve.items():
            size = (max(curve) + 1) if curve else 0
            mu = np.full(size, np.nan)
            sigma = np.full(size, np.nan)
            for bucket, (bucket_mu, bucket_sigma) in curve.items():
                mu[bucket] = bucket_mu
                sigma[bucket] = bucket_sigma
            self.curves[group] = (mu, sigma)

    def curve_for(self, group: str):
        """(mu, sigma) arrays for a group, empty when it was never fitted."""
        return self.curves.get(group, (np.empty(0), np.empty(0)))

    def counts_for(self, teams=None, fmt=None):
        """`(pooled, taken_by, max_pick_observed)` for a shape.

        The shape's own arrays once it holds `MIN_SHAPE_DRAFTS` drafts, and
        the pooled corpus otherwise -- including when the caller names no
        shape at all, which is every reader that has no league to speak for
        (the ADP pages, a test, a room whose settings never arrived).

        Falling back rather than answering thinly is the whole design: a
        12-team room on a corpus with nine 12-team drafts wants the 854-draft
        answer, which is wrong about the pick axis, not the nine-draft one,
        which is wrong about everything.
        """
        pooled = (self.pooled, self.taken_by, self.max_pick_observed)
        if teams is None or fmt is None:
            return pooled
        try:
            key = (int(teams), str(fmt))
        except (TypeError, ValueError):
            return pooled
        shape = self.shapes.get(key)
        if (shape is None or shape.pooled is None
                or shape.drafts < MIN_SHAPE_DRAFTS):
            return pooled
        return shape.pooled, shape.taken_by, shape.max_pick_observed

    def shape_drafts(self, teams=None, fmt=None) -> int:
        """How many drafts of one shape the corpus holds. 0 for a shape it
        has never seen."""
        try:
            key = (int(teams), str(fmt))
        except (TypeError, ValueError):
            return 0
        shape = self.shapes.get(key)
        return 0 if shape is None else int(shape.drafts)

    @classmethod
    def empty(cls) -> "AvailabilityTable":
        """What every reader gets when there is no corpus to read.

        Not an exception and not a zero: an empty table answers 1.0 for a
        player with no ADP and falls straight through to his own ADP for one
        who has it, so a fresh install (or a corpus mid-write) shows a room
        with no measured percentages rather than a room that claims everyone
        is gone.
        """
        return cls(player_ids=np.empty(0, dtype=object),
                   pooled=np.zeros(0, dtype=np.int64),
                   taken_by=np.zeros((0, MAX_PICK + 1), dtype=np.int64),
                   adp_curve={}, corpus_mtime=0.0, max_pick_observed=0)


def _fit_curve(frame: pd.DataFrame) -> dict:
    """ADP bucket -> (mu, sigma) of the pick the players in it went at.

    `frame` carries one row per (draft, pooled player): `group`, `bucket`,
    `pick` (NaN when he was never taken), `taken` and `player`.

    Fitted on the pool rows the corpus can actually see the answer for: the
    mean and standard deviation of the picks that HAPPENED, in buckets held
    by enough distinct players (`MIN_BUCKET_PLAYERS`) where most of them were
    drafted (`MIN_BUCKET_TAKEN`).

    MONOTONE IN THE BUCKET, forced. A worse ADP cannot mean an earlier pick,
    and a sample that says so is telling us about its own noise: the corpus
    has buckets where the mean pick falls as the rank rises, and left alone
    the room prints a deeper player as LESS likely to last than a better one
    sitting right above him in the same list. The running maximum keeps the
    curve reading in one direction; the same is done to sigma, since
    uncertainty about where a player goes does not shrink as the board gets
    thinner.
    """
    curves: dict = {}
    if frame.empty:
        return curves
    for group, rows in frame.groupby("group", sort=False):
        fitted = {}
        for bucket, in_bucket in rows.groupby("bucket", sort=True):
            if in_bucket["player"].nunique() < MIN_BUCKET_PLAYERS:
                continue
            if in_bucket["taken"].mean() < MIN_BUCKET_TAKEN:
                continue
            picks = in_bucket.loc[in_bucket["taken"], "pick"].to_numpy(float)
            if picks.size < MIN_DRAFTS:
                continue
            sigma = float(picks.std(ddof=1)) if picks.size > 1 else 0.0
            fitted[int(bucket)] = (float(picks.mean()), max(sigma, MIN_SIGMA))
        running_mu, running_sigma = -np.inf, MIN_SIGMA
        for bucket in sorted(fitted):
            mu, sigma = fitted[bucket]
            running_mu = max(running_mu, mu)
            running_sigma = max(running_sigma, sigma)
            fitted[bucket] = (running_mu, running_sigma)
        if fitted:
            curves[str(group)] = fitted
    return curves


def load_table(corpus_path: str = dl.CORPUS_PATH) -> AvailabilityTable:
    """Read the corpus and count it, or hand back an empty table.

    READ-ONLY, and a failure is not an error. The farm holds the corpus's
    write lock while it records a mock, so "could not open" is a normal thing
    for a poll to meet -- the same judgement api/market.py's `_corpus` makes,
    except that this one is on the draft room's own recompute path and cannot
    answer a poll with a 503.
    """
    import duckdb
    try:
        mtime = os.path.getmtime(corpus_path)
    except OSError:
        return AvailabilityTable.empty()
    try:
        conn = duckdb.connect(corpus_path, read_only=True)
    except Exception:      # noqa: BLE001 -- locked, missing, or not a corpus
        return AvailabilityTable.empty()
    try:
        # One query, one row per (draft, pooled player): the pick he went at
        # in that draft or NULL, the published rank the draft was reading,
        # and his position (the curve is fitted per position group).
        # LEFT JOIN from the pool rather than a scan of the picks, which is
        # what keeps `taken_by <= pooled` true -- a pick with no pool row
        # would otherwise count in the numerator and nowhere in the
        # denominator. `min(pick_no)` because a corpus that ever recorded a
        # player twice in one draft must still count him once.
        rows = conn.execute("""
            WITH taken AS (
                SELECT draft_id, player_id, min(pick_no) AS pick_no
                FROM draft_log_pick GROUP BY 1, 2
            )
            SELECT l.draft_id AS draft_id,
                   l.player_id AS player_id,
                   l.position AS position,
                   coalesce(l.espn_rank, l.adp_rank) AS value,
                   t.pick_no AS pick_no
            FROM draft_log_pool l
            LEFT JOIN taken t USING (draft_id, player_id)
        """).df()
        # The heads, separately and small (one row per draft): the shape is
        # two fields of the draft, and joining them onto 200k pool rows in
        # SQL would carry a kilobyte of settings JSON per row to learn one
        # word. `draft_format` reads either stored blob -- see it for why
        # both are asked for.
        heads = conn.execute(
            "SELECT draft_id, teams, "
            "coalesce(scoring_json, settings_json) AS scoring "
            "FROM draft_log WHERE teams IS NOT NULL "
            "ORDER BY draft_id").fetchall()
    except Exception:      # noqa: BLE001 -- a file that is not a corpus yet
        return AvailabilityTable.empty()
    finally:
        conn.close()
    if rows.empty:
        return AvailabilityTable.empty()

    codes, ids = pd.factorize(rows["player_id"].astype(str))
    n_players = len(ids)
    pooled = np.bincount(codes, minlength=n_players).astype(np.int64)

    pick = _floats(rows["pick_no"])
    was_taken = np.isfinite(pick)
    max_pick_observed = int(pick[was_taken].max()) if was_taken.any() else 0
    # Clipped rather than dropped: a corpus that one day records a deeper
    # draft than MAX_PICK should count those picks at the end of the axis,
    # not lose them out of the numerator.
    at = np.clip(pick[was_taken], 1, MAX_PICK).astype(np.int64)
    width = MAX_PICK + 1
    flat = codes[was_taken] * width + at
    counts = np.bincount(flat, minlength=n_players * width)
    taken_by = np.cumsum(counts.reshape(n_players, width), axis=1)

    value = _floats(rows["value"])
    known = np.isfinite(value)
    curve = _fit_curve(pd.DataFrame({
        "group": [position_group(p) for p in rows["position"][known]],
        "bucket": np.floor_divide(value[known], ADP_BUCKET).astype(np.int64),
        "pick": pick[known], "taken": was_taken[known],
        "player": codes[known]}))
    shapes, pooled_depth = _shape_counts(rows["draft_id"], heads, codes,
                                         n_players, pick, was_taken)
    return AvailabilityTable(
        player_ids=np.asarray(ids, dtype=object), pooled=pooled,
        taken_by=taken_by, adp_curve=curve, corpus_mtime=mtime,
        max_pick_observed=min(pooled_depth or max_pick_observed, MAX_PICK),
        shapes=shapes)


def _shape_counts(draft_column, heads, codes, n_players: int, pick,
                  was_taken) -> tuple:
    """`({shape: ShapeCounts}, the pooled depth)`.

    `codes` are the player row indices of the pooled rows and `pick` /
    `was_taken` their picks, so every shape's arrays land on the SAME player
    index as the pooled ones and `table.index` answers for all of them.

    THE POOLED DEPTH IS THE SHALLOWEST SHAPE'S, NOT THE DEEPEST PICK -- and a
    shape's own depth is the MEDIAN of its drafts' last picks. This is the
    whole reason the depth is computed here rather than off the pick column,
    and getting it wrong is not a rounding error. The pooled counts
    mix every shape together, so a player is "still on the board at pick 150"
    in them only if every group of drafts they were pooled in actually
    reached pick 150. An 8-team draft ends at 128 and records nothing after
    it; a single 12-team draft reaching 192 would raise a corpus-wide maximum
    to 192, and `availability_at` would then read `n = 150` as inside the
    counts -- turning 854 8-team drafts that simply STOPPED at 128 into
    evidence that the player lasted to 150. Measured on the real corpus when
    this was wrong: 173 censored players flipped off the fitted curve and
    read ~90% still there at pick 150, against ~10% from the curve.

    So: one shape's drafts can only speak as far as that shape went, and the
    pooled table can only speak as far as the shallowest of them.

    TWO THINGS KEEP ONE BAD ROOM OUT OF THAT MINIMUM, because the rule above
    is a floor and a floor is exactly what a single broken draft can drag
    down. A shape votes only once it holds `MIN_DEPTH_DRAFTS` drafts -- one
    12-team mock that died at pick 30 says nothing yet about where 12-team
    mocks end -- and a shape's depth is the median of its drafts' last picks
    rather than the deepest or the shallowest, so one truncated room among
    hundreds does not move it either. Shapes whose drafts hold no picks at
    all are skipped (they bound nothing), and a corpus where no shape can
    vote leaves the caller its own maximum, which is where this was before
    shapes existed.

    A SHAPE'S DRAFTS ARE COUNTED FROM THE POOL, not from the head rows: a
    draft with no pool snapshot (every draft `draft_log.backfill_history`
    imports) contributes nothing to any count, and letting it push a shape
    over MIN_SHAPE_DRAFTS would switch a room onto a table built from
    nothing. `draft_log.shape_counts`, which the farm's rotation reads,
    counts the heads instead and says so -- the two answer different
    questions.

    Arrays are built only for the shapes that will be used; the rest keep
    their draft count and their depth. On the real corpus this is one shape,
    so the extra work is one bincount over 200k rows.
    """
    shape_by_draft = {}
    for draft_id, teams, scoring in heads:
        try:
            shape_by_draft[str(draft_id)] = (int(teams),
                                             dl.draft_format(scoring))
        except (TypeError, ValueError):
            continue
    draft_codes, draft_ids = pd.factorize(draft_column.astype(str))
    # A LIST, not an object array: numpy reads a list of 2-tuples as a 2-D
    # array of numbers, and every shape would then be a row rather than a key.
    per_draft = [shape_by_draft.get(str(d)) for d in draft_ids]
    # Drafts whose shape cannot be read are a group of their own for the
    # depth: they are in the pooled counts like everybody else, so they bound
    # the pooled depth like everybody else. They are not a SHAPE -- nothing
    # can ask for them by name -- so they never reach `out`.
    groups = {x for x in per_draft}
    # Each draft's OWN last pick, which is what a shape's depth is a median
    # of. Grouped once here rather than per shape: it is one pass over the
    # picks either way, and every group below is a slice of the answer.
    last_pick = np.zeros(len(draft_ids), dtype=float)
    if was_taken.any():
        by_draft = pd.Series(pick[was_taken]).groupby(
            draft_codes[was_taken]).max()
        last_pick[by_draft.index.to_numpy()] = by_draft.to_numpy()
    out: dict = {}
    depths = []
    width = MAX_PICK + 1
    for shape in groups:
        mine = np.array([x == shape for x in per_draft], dtype=bool)
        rows_here = mine[draft_codes]
        drafted = rows_here & was_taken
        drafts = int(mine.sum())
        # The median of this shape's drafts' last picks, over the drafts that
        # made a pick at all -- a draft with a pool and no picks says nothing
        # about where the shape ends. Floored (`int` of a .5 median on an
        # even count), which errs shallow, which is the safe direction: the
        # cost of a depth one pick short is one pick answered by the curve.
        ended = last_pick[mine]
        ended = ended[ended > 0]
        depth = min(int(np.median(ended)), MAX_PICK) if ended.size else 0
        if depth and ended.size >= MIN_DEPTH_DRAFTS:
            depths.append(depth)
        if shape is None:
            continue
        if drafts < MIN_SHAPE_DRAFTS:
            out[shape] = ShapeCounts(None, None, drafts, depth)
            continue
        pooled = np.bincount(codes[rows_here],
                             minlength=n_players).astype(np.int64)
        counts = np.bincount(
            codes[drafted] * width + np.clip(pick[drafted], 1,
                                             MAX_PICK).astype(np.int64),
            minlength=n_players * width)
        out[shape] = ShapeCounts(
            pooled, np.cumsum(counts.reshape(n_players, width), axis=1),
            drafts, depth)
    return out, (min(depths) if depths else 0)


def _floats(series) -> np.ndarray:
    """A pandas column as plain float64 with NULL as NaN.

    DuckDB's `.df()` hands back pandas' masked integer dtype for a nullable
    INTEGER column, and asking a masked array for a float dtype raises rather
    than filling; `pick_no` is NULL for every player who was never taken,
    which is most of the corpus.
    """
    return series.to_numpy(dtype="float64", na_value=np.nan)


_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def cached_table(corpus_path: str = dl.CORPUS_PATH) -> AvailabilityTable:
    """The table for this corpus, rebuilt only when the file has moved.

    Counting the real corpus is ~100 ms of DuckDB and numpy (854 drafts,
    109k picks), which is fine once per process and not fine once per poll.
    The mtime is the whole invalidation rule: the farm only ever appends
    drafts, so a file that has not changed cannot have a different answer.

    A LOCKED CORPUS KEEPS THE LAST GOOD TABLE, and an unreadable corpus is
    never cached at all. `load_table` returns an empty table when it cannot
    read the file, and serving that would blank every percentage in every
    live room for as long as the farm's write takes; storing it would keep
    them blank until the file changed again.

    The load runs under the lock, and the cache is checked a second time
    inside it: fifty rooms recomputing on the same poll are fifty
    simultaneous misses, and every one of them would otherwise open its own
    read-only connection and count the whole corpus for a table the others
    are already building.
    """
    try:
        mtime = os.path.getmtime(corpus_path)
    except OSError:
        mtime = 0.0
    with _CACHE_LOCK:
        current = _CACHE.get(corpus_path)
        if current is not None and current.corpus_mtime == mtime:
            return current
    with _CACHE_LOCK:
        current = _CACHE.get(corpus_path)
        if current is not None and current.corpus_mtime == mtime:
            return current      # somebody loaded it while we waited
        table = load_table(corpus_path)
        if table.player_ids.size == 0:
            return current if current is not None else table
        _CACHE[corpus_path] = table
        return table


def _survivor(value, k, n: int, mu, sigma):
    """P(still there when pick n is made | still there after k picks).

    "Still there at pick n" means he survived every pick before it, so the
    numerator is the tail past `n - 1` -- the same off-by-one the counted
    path takes, so the two answers meet rather than differ by one pick.

    `k` may be an array: a player the counts carry to the corpus's depth is
    conditioned on THAT pick rather than on the current one, which is what
    makes the two halves of his answer multiply into one curve instead of
    stepping where they meet.
    """
    mu = np.where(np.isnan(mu), value, mu)
    sigma = np.where(np.isnan(sigma), FALLBACK_SIGMA, sigma)
    sigma = np.maximum(sigma, MIN_SIGMA)
    at_k = ndtr((mu - k) / sigma)
    at_n = ndtr((mu - (n - 1)) / sigma)
    # A denominator of zero means the curve says he was gone long before now
    # -- while the board says he is still here. The board is the fact and the
    # curve is the model, but the model's answer to "and will he last another
    # round" is still no, so 0.0 is what it gets to say.
    return np.divide(at_n, at_k, out=np.zeros_like(at_n), where=at_k > 1e-12)


def fallback_probability(value: float, k: int, n: int,
                         table: AvailabilityTable, position=None) -> float:
    """The parametric answer for one player, by his ADP (or consensus rank).

    Exposed because it is the answer for every player the corpus cannot
    speak for, and a test (or a reader) should be able to ask for it by
    itself rather than infer it from a board.
    """
    bucket = int(value // ADP_BUCKET)
    curve_mu, curve_sigma = table.curve_for(position_group(position))
    mu = sigma = np.nan
    if 0 <= bucket < curve_mu.size:
        mu, sigma = curve_mu[bucket], curve_sigma[bucket]
    got = _survivor(np.array([float(value)]), k, n,
                    np.array([mu]), np.array([sigma]))
    return float(np.clip(got, 0.0, 1.0)[0])


def availability_at(table: AvailabilityTable, player_ids, k: int, n: int,
                    espn_adp=None, market_rank=None, positions=None,
                    teams=None, fmt=None) -> np.ndarray:
    """P(still there when pick n is made | still there now), per player id.

    `k` is the number of picks already made and `n` an overall pick number,
    so "still there at n" means he survived picks 1..n-1 and the counted
    answer is `(pooled - taken_by[n-1]) / (pooled - taken_by[k])`. Past the
    corpus's own depth that is still the answer for a player it always saw
    drafted, and the counts times the curve for one it never saw the end of
    -- see the module docstring on censoring.

    `espn_adp`, `market_rank` and `positions` are aligned with `player_ids`;
    the first two may hold NaN. They are only read for the players the
    corpus cannot answer for. ESPN's ADP first because it is the list the
    room's drafters are reading (see pipeline/draft_log.py on why the pool
    snapshot stores ESPN's own numbers), the consensus rank when ESPN has no
    ADP for him, and 1.0 when there is neither -- a player nobody has ranked
    is a player nobody is about to draft. `positions` chooses which of the
    fitted curves he is read off; without it everyone is read off the skill
    curve, which is wrong for a kicker and harmless for anyone else.

    `teams` and `fmt` are the asking league's shape -- 10 and "ppr" for a
    ten-team PPR room. Given both, the counts come from that shape alone once
    the corpus holds `MIN_SHAPE_DRAFTS` drafts of it, and from the whole
    corpus until then (`table.counts_for`); given neither, from the whole
    corpus, which is what every reader with no league to speak for wants. The
    parametric fallback is pooled either way.

    Vectorised over the whole board: one gather out of `taken_by` and one
    normal tail, because this runs for every remaining turn of every room on
    every poll.
    """
    ids = [str(p) for p in player_ids]
    size = len(ids)
    out = np.ones(size, dtype=float)
    if size == 0:
        return out
    k = int(np.clip(k, 0, MAX_PICK))
    n = int(np.clip(n, 0, MAX_PICK))
    # The pick on the clock cannot take anybody away: he is on the board
    # right now, and `n = k + 1` is the very next pick. `build_plan` passes
    # the current turn as its first turn and has to agree with `target_now`
    # about it. A turn already behind us is not a question either.
    if n <= k + 1:
        return out

    idx = np.fromiter((table.index.get(pid, -1) for pid in ids),
                      dtype=np.int64, count=size)
    have = idx >= 0
    counted = np.zeros(size, dtype=bool)
    # What the counts contribute to a censored player's answer: the share of
    # his drafts he was still there at the end of. 1.0 for everyone the
    # counts do not speak for at all, so the curve stands alone for them.
    carried = np.ones(size, dtype=float)
    # Which pick the curve conditions on: now, or the corpus's own depth for
    # a player the counts have already carried that far.
    given = np.full(size, float(k))
    # The shape's own counts, or the pooled corpus. Read once, before the
    # gather, because every line below is about one table or the other and
    # nothing here should have to remember which.
    all_pooled, all_taken_by, depth = table.counts_for(teams, fmt)
    if all_pooled.size and have.any():
        where = np.flatnonzero(have)
        rows = idx[where]
        pooled = all_pooled[rows].astype(float)
        denominator = pooled - all_taken_by[rows, k]
        numerator = pooled - all_taken_by[rows, n - 1]
        enough = denominator >= MIN_DRAFTS
        # Every pooled draft took him inside the recorded depth. Nothing
        # about him is censored, so the counts answer any pick, however deep.
        complete = all_taken_by[rows, depth] >= pooled
        # And every pooled draft took him before THIS pick, which is not a
        # ratio and does not need MIN_DRAFTS behind it -- it needs enough
        # drafts to not be a coincidence (MIN_COMPLETE_DRAFTS) and one draft
        # in which he actually reached the pick we are conditioning on, or
        # the answer is the 0/0 the curve exists for.
        gone = ((pooled >= MIN_COMPLETE_DRAFTS) & (numerator <= 0)
                & (denominator > 0))
        direct = (enough & ((n <= depth) | complete)) | gone
        out[where[direct]] = numerator[direct] / denominator[direct]
        counted[where[direct]] = True
        # The rest of the counted players are asked about a pick past the
        # corpus's reach: the counts take them to the end of it and the curve
        # carries them from there, conditioned on that same pick so the two
        # halves meet instead of stepping.
        carry = enough & ~direct
        carried[where[carry]] = ((pooled - all_taken_by[rows, depth])[carry]
                                 / denominator[carry])
        # The depth, or the clock if the clock is already past it. Once `k`
        # is deeper than anything the corpus recorded, the counts have
        # nothing left to carry him with -- `taken_by` has been flat since
        # the depth, so `carried` is 1.0 -- and conditioning the curve on the
        # depth rather than on `k` asks how likely he was to reach pick 128
        # when he has demonstrably reached 140. Understated by two orders of
        # magnitude across the back half of a 12-team draft: 0.0009 against
        # 0.0339 at k=140, n=150.
        given[where[carry]] = float(max(depth, k))

    rest = ~counted
    if rest.any():
        out[rest] = carried[rest]
        adp = _as_array(espn_adp, size)
        market = _as_array(market_rank, size)
        value = np.where(np.isfinite(adp), adp, market)
        ranked = rest & np.isfinite(value)
        if ranked.any():
            groups = np.array([position_group(p)
                               for p in (positions if positions is not None
                                         else [None] * size)], dtype=object)
            mu = np.full(size, np.nan)
            sigma = np.full(size, np.nan)
            bucket = np.zeros(size, dtype=np.int64)
            bucket[ranked] = np.floor_divide(value[ranked],
                                             ADP_BUCKET).astype(np.int64)
            for group in np.unique(groups[ranked]):
                curve_mu, curve_sigma = table.curve_for(str(group))
                if not curve_mu.size:
                    continue
                mine = ranked & (groups == group)
                inside = mine & (bucket >= 0) & (bucket < curve_mu.size)
                mu[inside] = curve_mu[bucket[inside]]
                sigma[inside] = curve_sigma[bucket[inside]]
            out[ranked] = carried[ranked] * _survivor(
                value[ranked], given[ranked], n, mu[ranked], sigma[ranked])
    return np.clip(out, 0.0, 1.0)


def _as_array(values, size: int) -> np.ndarray:
    """A float array of `size`, all-NaN when the caller passed nothing."""
    if values is None:
        return np.full(size, np.nan)
    out = np.asarray(values, dtype=float)
    if out.size != size:
        raise ValueError(f"expected {size} values, got {out.size}")
    return out
