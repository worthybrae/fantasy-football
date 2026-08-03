# Weighted stat summary + ESPN projection with growth signal

**Date:** 2026-08-02 · **Status:** approved (user-specified)

## Feature

A summary strip at the top of the player drawer, above the charts:

1. **Recency-weighted per-game stat line.** Weighted average of each
   season-summary stat divided by games played, using the same
   `RECENCY_WEIGHTS` as the production factor (2025 50% / 2024 30% / 2023
   20%), renormalized over the seasons the player actually has. Stats:
   ppg plus completions/attempts/pass_yards/pass_tds/interceptions/
   carries/rush_yards/targets/receptions/rec_yards/tds per game. The
   frontend picks position-relevant columns (same config as the tables).
2. **ESPN projected PPG.** `parse_espn` additionally extracts the season
   projection (stats entry with statSourceId=1, statSplitTypeId=0,
   seasonId=year → appliedTotal, verified live) into a new `espn_proj`
   column on the `espn_adp` table. Projected PPG = season total / 17.
   Zero/missing projections → None.
3. **Growth signal.** If projected PPG > weighted-avg PPG → "growth
   expected" badge (↑, green); below → decline (↓, red). Missing either
   side → no badge.

## Plumbing

`build_profile` maps the player to their ESPN row via the sleeper_ids
gsis↔espn crosswalk with a name+position fallback (same strategy as
scoring/market.py), then emits:

```
"summary": { "w_ppg", "w_stats": {<stat>: per-game float}, "proj_ppg", "proj_delta" }
```

Graceful degradation: missing espn_proj column (pre-refresh table), no
crosswalk hit, or no seasons → the affected fields are None and the UI
hides what it can't show.

## Testing

Backend TDD: parse_espn projection extraction; weighted math over uneven
seasons; crosswalk + fallback join; degradation without espn_proj.
Frontend: tsc + screenshots via the temp verification stack.
