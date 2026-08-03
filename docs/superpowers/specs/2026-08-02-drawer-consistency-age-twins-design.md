# Drawer rework: consistency chart, collapsed game log, age-matched twins

**Date:** 2026-08-02 · **Status:** approved

## Changes

### 1. New `players` source table

`pipeline/sources.py::fetch_players()` wraps `nfl_data_py.import_players()`,
keeping `gsis_id`, `display_name`, `birth_date`, `rookie_season`. Registered
as a `players` job in `pipeline/refresh.py`. Populated on next `make refresh`.

### 2. Age-matched stat twins

`find_twins(weekly, player_id, players=None)`:

- Age for a season = full years old on **Sept 1 of the season year** (opening
  week; late-September birthdays don't bump early).
- Candidate seasons must be from a player at **exactly the target's age**
  (user chose exact over ±1), on top of existing filters (position,
  ≥ MIN_GAMES, has a next season).
- Missing birth date on a candidate → excluded. Missing birth date on the
  target (or absent/empty `players` table) → age filter skipped entirely;
  behaves exactly as today.
- Z-scores stay computed over the full position pool (stable scale); the age
  filter applies to candidates only.
- Payload gains `target_age` (top level) and `age` per comp so the UI can
  show the matching.

`build_profile` reads the `players` table and passes it through.

### 3. Consistency chart replaces Factors

- Drawer's Factors section and `FactorBars` component are removed (factor
  values still drive the composite/board; unused component file deleted).
- `season_summaries` gains `ppg_std`: sample std dev (ddof=1) of per-game
  PPR points across games played; `null` for single-game seasons. Weekly
  rows are played games by definition — DNP zero-fill lives only in
  `game_log`, so no exclusion logic is needed here.
- New `SeasonRangeChart` component in the old Factors slot: x = season
  ('19…'25, chronological), point = avg PPG, vertical whiskers = ±1σ,
  single categorical hue (slot 1 `#3987e5`), per-season hover tooltip
  (avg · σ · games).

### 4. Collapsible game log

`GameLog.tsx`: each season block is an accordion, **all collapsed by
default**. Collapsed header shows `{season} · {games played} gp · avg
{avg} pts` (avg over played games only). Clicking toggles that season's
per-game rows. Pure frontend state; no API change.

## Testing

- Backend TDD: age filter (same-age in, off-age out, no-birth-date target
  degrades to unfiltered), Sept-1 age boundary, `ppg_std` math and
  single-game null, players passthrough in `build_profile`.
- Frontend: `tsc --noEmit` + live browser screenshots (no JS test harness in
  repo).
