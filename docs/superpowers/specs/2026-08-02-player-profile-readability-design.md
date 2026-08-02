# Player Profile Readability — Design

**Date:** 2026-08-02
**Status:** Approved (verbal, in-session)

## Goal

Make the player profile drawer easier to read and easier to use for comparing
fantasy-point performance against underlying real stats. Four priorities, all
in scope: position-aware season columns, an aligned game-log table, chart
tooltips that carry the full stat line, and visual cues on fantasy-point
values. Layout: keep the side-drawer pattern but widen it.

## Why the API must change

The profile payload sends each game's stats as a pre-formatted string
(`stat_line`, built in `scoring/profile.py:_stat_line`) and the season
summaries omit passing stats entirely. The frontend therefore cannot build
aligned columns or show a QB's passing seasons. Structured stats must come
from the API (Approach A). Frontend-only string parsing (Approach B) was
rejected as fragile and incapable of position-aware season columns; a full
tabbed redesign (Approach C) was rejected as scope creep.

## 1. API payload changes

`scoring/profile.py`:

- **`game_log` rows** gain a `stats` object with raw per-game numbers, always
  the same keys (zero-filled where absent):
  `completions, attempts, pass_yards, pass_tds, interceptions, carries,
  rush_yards, rush_tds, targets, receptions, rec_yards, rec_tds`.
  `stat_line` is kept (used by the chart tooltip until the frontend formats
  its own, and by any existing consumers/tests).
- **`season_summaries` rows** gain `completions, attempts, pass_yards,
  pass_tds, interceptions`, aggregated from the `weekly` table the same way
  the existing counting stats are.

`web/src/api.ts`: extend `GameLogRow` with `stats: GameStats` and
`SeasonSummary` with the five passing fields.

## 2. Shared position→columns config

New `web/src/statColumns.ts`: a single definition mapping position to an
ordered list of column descriptors `{ key, label, format }`, with two
variants per position (season table vs game log) only where the source keys
differ (season summaries use `rec_yards`/`rush_yards` naming; game stats use
the same names, so one config should serve both).

- **QB:** Cmp/Att (combined column), Pass Yds, Pass TD, INT, Car, Rush Yds
- **RB:** Car, Rush Yds, Rush TD, Tgt, Rec, Rec Yds
- **WR/TE:** Tgt, Rec, Rec Yds, Rec TD, Car, Rush Yds
- **Fallback (unknown):** current generic set

Season table additionally keeps its season-level columns (Season, G, PPG,
Tgt%, Yds/Opp, Snap%) around the position stat columns; TD-related and share
columns stay where sensible. K/DST continue to render no history (unchanged
suppression in `build_profile`).

## 3. Game log as an aligned table

Replace the `<details>` text list in `GameLog.tsx` with a real table:
`Wk | Opp | [position stat columns] | Pts`, grouped by season with a season
header row, newest season first (API order preserved). Expanded by default.
Numbers right-aligned, `mono` class. The component receives the player's
position to select columns.

## 4. Chart linkage and point cues

- `WeeklyChart` bar `<title>` tooltips include the full stat line (formatted
  from `stats` via the shared config, falling back to `stat_line`).
- Game-log `Pts` cells get background shading relative to that season's own
  average PPG: above-average tinted green, below tinted red, near-average
  neutral. Shading must be subtle (readable text, both themes) and derived
  from data already in the payload.
- Season table `PPG` cells get the same treatment across the player's
  seasons.

## 5. Wider drawer

`.player-drawer` widens to `clamp(560px, 60vw, 840px)` on desktop; existing
small-screen behavior (full width) unchanged. Tables must fit each
position's column set without horizontal scrolling at the widened width;
the scroll-wrapper stays as a safety net only.

## Error handling

- Missing/None stats render as 0 (counting stats) exactly as the API
  zero-fills them; percent/derived fields keep the existing `—` convention.
- The drawer always fetches the profile live (no client-side caching of the
  payload), so the frontend may require the new `stats` shape outright — no
  dual-shape fallback rendering. API and frontend ship together.

## Testing

- Python: extend existing profile tests to assert the new `stats` object and
  season passing aggregates (QB fixture must show non-zero passing fields;
  RB/WR fields zero-filled as appropriate).
- Frontend: `npm run build` and lint clean. Existing component behavior
  (K history suppression, empty states) unchanged.
