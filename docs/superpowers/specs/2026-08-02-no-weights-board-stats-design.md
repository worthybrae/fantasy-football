# Design: Remove Weights UI, Full-Width Board with Landing-Page Stats

Date: 2026-08-02
Status: approved

## Goal

The draft board becomes one big full-width player list. The Weights sidebar
goes away, along with every column the composite model powers (Composite,
VOR, Edge, Rank, Tier). The list leans on ESPN PPR rank (default sort),
market rank, and real last-season stats shown directly on the landing page.

## Decisions made

- **Score columns:** dropped entirely from the UI (not recomputed with fixed
  weights). Backend scoring machinery stays — profiles still use it.
- **Board stats:** always show last-season fantasy summary (G, PPG, Pts);
  when a single-position tab (QB/RB/WR/TE) is active, append that
  position's counting-stat columns.
- **Order:** default sort is ESPN PPR rank; model Rank/Tier columns are
  removed.

## Backend (one change: `scoring/board.py`)

`build_board` merges latest-season per-player stats onto each board row as a
nested `stats` object (same serialization pattern as `market_sources`):

```
stats: {
  season, games, ppg, points,
  carries, rush_yards, targets, receptions, rec_yards, tds,
  completions, attempts, pass_yards, pass_tds, interceptions,
} | null
```

- Computed from the `weekly` table via `player_season_features()`
  (`scoring/similarity.py`) restricted to the latest season, plus a
  passing-stats groupby (same approach as `scoring/profile.py`).
- Rookies, K, and DST have no weekly rows → `stats: null` → dash cells.
- No snap-share / target-share on the board (extra joins; profile drawer
  keeps them).
- `composite`/`vor`/`tier`/`edge`/`rank` remain in the payload — the API
  contract is unchanged except for the added `stats` field; the UI just
  stops rendering the score fields.

Rejected alternative: a separate `/api/stats` endpoint joined client-side —
extra round-trip for no benefit.

## Frontend

- **Rail deleted entirely.** `WeightSliders.tsx` removed; no collapse
  toggle, no `railCollapsed` state or localStorage key. Table is
  full-width. "Hide drafted" checkbox moves up beside the position tabs.
- **Weights gone from the client.** `api.ts` drops the `Weights` type,
  `DEFAULT_WEIGHTS`, and `w_*` params from `fetchPlayers()` and
  `fetchProfile()` — the server's own defaults apply. App's 300 ms
  slider-debounce refetch goes away (plain fetch on mount).
  `PlayerProfile` loses its `weights` prop.
- **Columns:** ✓ toggle · ESPN PPR (default sort) · Name (+R badge) · Pos ·
  Team · Bye · Mkt · **G · PPG · Pts**, then position-specific columns only
  on single-position tabs:
  - QB: Cmp/Att, Pass Yds, Pass TD, INT, Car, Rush Yds
  - RB: Car, Rush Yds, Tgt, Rec, Rec Yds, TD
  - WR/TE: Tgt, Rec, Rec Yds, Car, Rush Yds, TD
  - All stat columns numerically sortable, nulls last (`sortUndefined`).
    Cmp/Att sorts on attempts.
- **Removed with the score columns:** VOR micro-bars (`maxVorByPosition`),
  tier-band shading (`showTierBreaks` prop and tier logic), and the
  screen-reader row announcement switches from model rank to ESPN PPR rank.
- `Player` TS interface drops `composite`, `vor`, `edge`, `rank`, `tier`,
  and the factor fields; gains `stats`.

## Error handling

Unchanged: existing load/error/empty states in App stay as-is. Null stats
render as em-dashes.

## Testing

- Backend: board rows carry a populated `stats` dict for a player with
  weekly data; `stats` is `None` for an ADP-only (rookie/K/DST) row.
- Existing board tests keep passing — no payload fields removed.
- Frontend verified by running the app (no JS test harness in repo).
