# Drawer charts v2: big consistency chart, position finish, snap share

**Date:** 2026-08-02 · **Status:** approved

## Changes

1. **Weekly points chart removed.** `WeeklyChart` component deleted; the
   game-log accordion (with DNP rows) remains the per-game view.
2. **Avg & volatility chart goes full width.** Taller plot (~200px), larger
   marks, svg scales to the drawer width (viewBox padded to a minimum of 6
   season slots so short careers don't blow up the type scale).
3. **Position finish graphic.** `season_summaries` gains `pos_finish`: rank
   within (season, position) by **total season PPR points** (user choice;
   rank 1 = most points, computed across all players in the weekly table).
   New `PositionFinishChart`: line + dots by season, y inverted (rank 1 at
   top), each point directly labeled ("RB12"), ticks at 1/12/24/…
4. **Snap share chart.** Per-season bars of `snap_share` (already in the
   payload from snap_counts) as a percentage, 0–100 axis. Rendered only for
   RB/WR/TE. Seasons with null snap data are skipped.

All three charts use the single categorical hue (slot 1 `#3987e5`) —
single-series magnitude charts, per the dataviz color rules.

## Testing

Backend TDD for `pos_finish` (rank math, cross-position isolation, ties).
Frontend: tsc + live screenshots via the temporary verification stack.
