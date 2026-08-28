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
which he also lasted past `n`.

WHEN THE COUNTS RUN OUT there is a parametric fallback fitted from the same
rows -- see `_fit_curve` for what it can and cannot say. It is used for a
player the corpus has never seen (the board is longer than the corpus's pool)
and for one whose conditioned denominator is too thin to be a probability
(`MIN_DRAFTS`). A fallback is not a nicety here: without it the two cases
above return 0/0 and 1/1, i.e. "certainly gone" and "certainly there", which
are the two most confident things this module could possibly say.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

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

# Width of the ADP buckets the fallback curve is fitted in. Narrow enough
# that the top of the board (where one pick of ADP is a real difference in
# when a player goes) is not averaged flat, wide enough that every bucket in
# the measured range holds thousands of pick rows.
ADP_BUCKET = 8

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


@dataclass
class AvailabilityTable:
    """The corpus, counted once, in the shape the room asks questions in.

    `taken_by[i, p]` is how many of player i's pooled drafts had taken him by
    overall pick p -- monotone in p by construction, and never larger than
    `pooled[i]`, because a pick only counts when the same draft's pool row
    exists (see `load_table`'s join).
    """
    player_ids: np.ndarray
    pooled: np.ndarray
    taken_by: np.ndarray
    adp_curve: dict          # bucket -> (mu, sigma) of the pick he went at
    corpus_mtime: float = 0.0

    def __post_init__(self):
        self.index = {str(pid): i for i, pid in enumerate(self.player_ids)}
        # The curve as two arrays indexed by bucket, so a whole board's worth
        # of fallbacks is one gather rather than 250 dict lookups. NaN is
        # "this bucket was not fitted" and is what `availability_at` replaces
        # with the player's own ADP.
        size = (max(self.adp_curve) + 1) if self.adp_curve else 0
        self.curve_mu = np.full(size, np.nan)
        self.curve_sigma = np.full(size, np.nan)
        for bucket, (mu, sigma) in self.adp_curve.items():
            self.curve_mu[bucket] = mu
            self.curve_sigma[bucket] = sigma

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
                   adp_curve={}, corpus_mtime=0.0)


def _fit_curve(value: np.ndarray, pick: np.ndarray,
               taken: np.ndarray) -> dict:
    """ADP bucket -> (mu, sigma) of the pick the players in it went at.

    Fitted on the pool rows the corpus can actually see the answer for: the
    mean and standard deviation of the picks that HAPPENED, in buckets where
    most of them did (`MIN_BUCKET_TAKEN`). A bucket with too few rows
    (`MIN_DRAFTS`) is not fitted either.
    """
    if value.size == 0:
        return {}
    bucket = np.floor_divide(value, ADP_BUCKET).astype(np.int64)
    frame = pd.DataFrame({"bucket": bucket, "pick": pick, "taken": taken})
    curve = {}
    for b, rows in frame.groupby("bucket", sort=True):
        if len(rows) < MIN_DRAFTS or rows["taken"].mean() < MIN_BUCKET_TAKEN:
            continue
        picks = rows.loc[rows["taken"], "pick"].to_numpy(dtype=float)
        if picks.size < MIN_DRAFTS:
            continue
        sigma = float(picks.std(ddof=1)) if picks.size > 1 else 0.0
        curve[int(b)] = (float(picks.mean()), max(sigma, MIN_SIGMA))
    return curve


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
        # in that draft or NULL, and the published rank the draft was reading.
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
            SELECT l.player_id AS player_id,
                   coalesce(l.espn_rank, l.adp_rank) AS value,
                   t.pick_no AS pick_no
            FROM draft_log_pool l
            LEFT JOIN taken t USING (draft_id, player_id)
        """).df()
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
    curve = _fit_curve(value[known],
                       np.where(was_taken[known], pick[known], np.nan),
                       was_taken[known])
    return AvailabilityTable(
        player_ids=np.asarray(ids, dtype=object), pooled=pooled,
        taken_by=taken_by, adp_curve=curve, corpus_mtime=mtime)


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

    Counting the real corpus is ~95 ms of DuckDB and numpy (854 drafts,
    109k picks), which is fine once per process and not fine once per poll. The mtime is the whole
    invalidation rule: the farm only ever appends drafts, so a file that has
    not changed cannot have a different answer in it.

    A LOCKED CORPUS KEEPS THE LAST GOOD TABLE. `load_table` returns an empty
    table when it cannot read the file, and serving that would blank every
    percentage in every live room for as long as the farm's write takes. The
    empty table is not cached either, so the next poll tries again.
    """
    try:
        mtime = os.path.getmtime(corpus_path)
    except OSError:
        mtime = 0.0
    with _CACHE_LOCK:
        current = _CACHE.get(corpus_path)
    if current is not None and current.corpus_mtime == mtime:
        return current
    table = load_table(corpus_path)
    if table.player_ids.size == 0 and current is not None:
        return current
    with _CACHE_LOCK:
        _CACHE[corpus_path] = table
    return table


def _survivor(value, k: int, n: int, mu, sigma):
    """P(pick > n | pick > k) under the normal fitted at (mu, sigma)."""
    mu = np.where(np.isnan(mu), value, mu)
    sigma = np.where(np.isnan(sigma), FALLBACK_SIGMA, sigma)
    sigma = np.maximum(sigma, MIN_SIGMA)
    at_k = ndtr((mu - k) / sigma)
    at_n = ndtr((mu - n) / sigma)
    # A denominator of zero means the curve says he was gone long before now
    # -- while the board says he is still here. The board is the fact and the
    # curve is the model, but the model's answer to "and will he last another
    # round" is still no, so 0.0 is what it gets to say.
    return np.divide(at_n, at_k, out=np.zeros_like(at_n), where=at_k > 1e-12)


def fallback_probability(value: float, k: int, n: int,
                         table: AvailabilityTable) -> float:
    """The parametric answer for one player, by his ADP (or consensus rank).

    Exposed because it is the answer for every player the corpus cannot
    speak for, and a test (or a reader) should be able to ask for it by
    itself rather than infer it from a board.
    """
    bucket = int(value // ADP_BUCKET)
    mu = sigma = np.nan
    if 0 <= bucket < table.curve_mu.size:
        mu, sigma = table.curve_mu[bucket], table.curve_sigma[bucket]
    got = _survivor(np.array([float(value)]), k, n,
                    np.array([mu]), np.array([sigma]))
    return float(np.clip(got, 0.0, 1.0)[0])


def availability_at(table: AvailabilityTable, player_ids, k: int, n: int,
                    espn_adp=None, market_rank=None) -> np.ndarray:
    """P(still there at pick n | still there at pick k), one per player id.

    `espn_adp` and `market_rank` are aligned with `player_ids` and may hold
    NaN; they are only read for the players the corpus cannot answer for.
    ESPN's ADP first because it is the list the room's drafters are reading
    (see pipeline/draft_log.py on why the pool snapshot stores ESPN's own
    numbers), the consensus rank when ESPN has no ADP for him, and 1.0 when
    there is neither -- a player nobody has ranked is a player nobody is
    about to draft.

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

    idx = np.fromiter((table.index.get(pid, -1) for pid in ids),
                      dtype=np.int64, count=size)
    have = idx >= 0
    counted = np.zeros(size, dtype=bool)
    if table.pooled.size and have.any():
        rows = idx[have]
        pooled = table.pooled[rows].astype(float)
        denominator = pooled - table.taken_by[rows, k]
        numerator = pooled - table.taken_by[rows, n]
        enough = denominator >= MIN_DRAFTS
        answered = np.flatnonzero(have)[enough]
        out[answered] = numerator[enough] / denominator[enough]
        counted[answered] = True

    rest = ~counted
    if rest.any():
        adp = _as_array(espn_adp, size)
        market = _as_array(market_rank, size)
        value = np.where(np.isfinite(adp), adp, market)
        ranked = rest & np.isfinite(value)
        if ranked.any():
            v = value[ranked]
            bucket = np.floor_divide(v, ADP_BUCKET).astype(np.int64)
            inside = (bucket >= 0) & (bucket < table.curve_mu.size)
            mu = np.full(v.shape, np.nan)
            sigma = np.full(v.shape, np.nan)
            if table.curve_mu.size:
                mu[inside] = table.curve_mu[bucket[inside]]
                sigma[inside] = table.curve_sigma[bucket[inside]]
            out[ranked] = _survivor(v, k, n, mu, sigma)
    return np.clip(out, 0.0, 1.0)


def _as_array(values, size: int) -> np.ndarray:
    """A float array of `size`, all-NaN when the caller passed nothing."""
    if values is None:
        return np.full(size, np.nan)
    out = np.asarray(values, dtype=float)
    if out.size != size:
        raise ValueError(f"expected {size} values, got {out.size}")
    return out
