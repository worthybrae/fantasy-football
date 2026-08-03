# Deep history (2016–2025) for profiles, unchanged board scoring

**Date:** 2026-08-02 · **Status:** approved

## Problem

The pipeline only pulls 2023–2025 weekly stats. Player profiles, game logs,
and especially stat twins would benefit from a deeper pool, but the draft
board must not change: `durability_factor` divides career games by seasons
since the player's *first season in the data*, so loading 2016+ would let a
2017 injury drag down a veteran's 2026 durability score.

## Decision (Approach A of three considered)

More data where it helps, none where it hurts:

1. `scoring/config.py`: `HISTORY_SEASONS = list(range(2016, 2026))`.
   `RECENCY_WEIGHTS` unchanged — its keys double as the definition of the
   board's scoring window (2023–2025).
2. `scoring/board.py::build_board`: filter `weekly` to seasons in
   `RECENCY_WEIGHTS` immediately after reading. Board output stays
   byte-for-byte identical to today.
3. Profile/similarity/game-log paths read the full `weekly` table directly
   and inherit the depth for free. The frontend hardcodes no seasons.

Verified: nflverse publishes `stats_player_week_{2016..2022}.parquet` (HTTP
200) and snap counts cover 2012+.

## Rejected

- **B — extend recency weights over 10 seasons:** era noise, every ranking
  shifts days before the draft, durability metric needs a redesign.
- **C — separate `weekly_history` table:** duplicate schema/refresh/joins for
  what a one-line season filter achieves.

## Tests

Board built with an extra monster 2016 season for a player must produce
identical factor values to a board without those rows. Existing suite guards
the rest.
