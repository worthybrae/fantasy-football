# Player page redesign: dossier layout, depth chart, schedule calendar

**Date:** 2026-08-03 · **Status:** approved (user-specified)

## Brief (user's words, distilled)

The full-page player view still looks like the old drawer. Wanted: loading
animations between pages; a compact/modern chart treatment; a larger,
readable rankings component showing every source; attention-grabbing stat
tiles instead of small plain chips; a team depth-chart component; a
schedule-easiness calendar component.

## Backend additions (profile payload)

1. `depth_chart`: latest-snapshot offensive depth for the player's team —
   `[{position: QB|RB|WR|TE, players: [{name, rank, is_me}]}]`, top 4 per
   group, profiled player flagged. Empty for old/absent depth schemas.
2. `schedule`: 18 week rows for the player's team —
   `[{week, opponent, home, fpa_pg, pct}]`; `fpa_pg` = opponent's prior-
   season PPR points allowed per game to the player's position, `pct` its
   percentile among teams (high = soft matchup). Bye weeks carry null
   opponent. Empty for K/DST (no meaningful positional FPA).

## Visual design

Existing identity kept (dark terminal/mono draft-ops tokens); this pass is
layout + hierarchy, with the **schedule calendar as the signature element**
(18 heat-tinted week cells using the existing ok/fail boom-bust tints).

Two-column dossier grid (12-col, stacks on narrow):
- Row 1: stat tiles (hero W-avg PPG, ESPN proj + growth delta, implied
  points, per-game stats as small tiles) · rankings panel (big aggregate
  number, five source rows with rank bars, FP tier).
- Row 2: avg&volatility chart · snap share chart (both compacted: shorter
  plots, tighter slots, smaller marks).
- Row 3: depth chart card (position columns, player highlighted) · schedule
  calendar.
- Full-width: season history, game log, similar players.

Loading: skeleton screens (shimmer, `prefers-reduced-motion` respected) for
the player page while the roster/profile load; content fades in. Old
Market/Outlook chip sections retire — their data lives in the rankings
panel and tiles.

## Testing

Backend TDD (depth grouping/ordering/is_me; difficulty math, bye rows, pct
direction, K empty). Frontend: tsc + screenshot iteration.
