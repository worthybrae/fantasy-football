# Player Profile Drawer — Design

**Date:** 2026-08-02
**Status:** Approved by user (drawer UX, stat-profile twins, cross-year comps amendment)

## Purpose

The draft board answers "who do I take"; it doesn't answer "who IS this player."
This feature adds a player profile drawer — the research surface: full historical
stats, everything the database knows about the player, and similar player-seasons
across years with next-season outcomes (a lightweight trend predictor).

## Interaction

- Clicking a board row opens a slide-over drawer (~70% viewport width) over the
  board; the board stays mounted behind it. Close via Esc, click-outside, or ✕.
- The drafted toggle MOVES to the drawer header and to a dedicated per-row button
  on the board (row-click no longer toggles drafted — fixes mis-click drafting).
- Similar players inside the drawer are clickable: clicking swaps the drawer to
  that player's profile in place (profile-hopping). Board state (sort, filters,
  scroll) is untouched throughout.

## Profile content (top to bottom)

1. **Header** — name, position, team, bye, rookie badge; chips for rank, tier,
   VOR, composite, ADP, edge; "Mark drafted" button.
2. **Factor bars** — the five factors (production, durability, role, environment,
   schedule) as labeled horizontal bars, 0–100.
3. **Weekly scoring chart** — PPR points per game for 2023–2025, colored by
   season, per-season average line. Shows trajectory, volatility, injury gaps.
4. **Season summaries table** — one row per season: games, PPR pts/gm, targets,
   target share, carries, receiving yards, rushing yards, total TDs, receptions,
   yards per opportunity, snap share (snap counts joined by cleaned name+team,
   blank if unmatched).
5. **Game log** — collapsible, newest first: season, week, opponent, key stat
   line (pos-appropriate), PPR points.
6. **2026 outlook** — depth chart slot, team implied points, position schedule
   strength (both raw + within-position percentile), bye week.
7. **Similar player-seasons** — top 5 (see below).

## Similarity (cross-year stat twins)

- **Target vector**: the player's latest season with ≥4 games.
- **Candidates**: every player-season 2023–2025 at the same position with ≥4
  games, excluding the target player's own seasons.
- **Features** (computed from weekly data per player-season): PPR pts/gm, games,
  target share, carry share, yards per opportunity, TD per opportunity,
  receptions/gm. Z-scored within position across all candidate seasons.
  Usage features (target share, carry share, ppg) weighted 2x vs efficiency
  features.
- **Distance**: weighted Euclidean; similarity = 100 × exp(−distance/k) with k
  chosen so a clone scores 100 and typical position peers land ~40–70.
- **Output per comp**: player name, season year, similarity score, that season's
  PPR pts/gm — and **next-season PPR pts/gm** (the trend signal; null for 2025
  seasons, no observable next year). Plus, when the comp player is on the 2026
  board: their current rank/ADP (spot cheaper current versions).
- **Fallback**: rookies/K/DST (no qualifying stat season) get value-neighbors
  instead — 5 same-position players nearest in VOR on the current board,
  labeled as such in the UI.

## API

One new endpoint, existing endpoints untouched:

`GET /api/players/{player_id}/profile` → single JSON payload:

```
{
  header: {…board row fields…},
  factors: {production, durability, role, environment, schedule},
  seasons: [{season, games, ppg, targets, target_share, carries, rec_yards,
             rush_yards, tds, receptions, yards_per_opp, snap_share|null}],
  game_log: [{season, week, opponent, stat_line, ppr_points}],
  outlook: {depth_slot|null, implied_points|null, sos_raw|null, sos_pct|null, bye|null},
  similar: {mode: "stat_twins"|"value_neighbors",
            players: [{player_id|null, name, season|null, similarity|null,
                       ppg, next_ppg|null, rank|null, adp|null}]}
}
```

- Computed on demand from DuckDB (~700 players, milliseconds); no precomputation.
- 404 for unknown player_id. ADP-only players (synthetic `adp_*` ids) return
  header/outlook/similar-fallback with empty seasons/game_log.
- Backend module: `scoring/profile.py` (pure functions) + a thin route in
  `api/main.py`. Similarity in `scoring/similarity.py`.

## Frontend

- `web/src/components/PlayerProfile.tsx` (drawer + sections),
  `FactorBars.tsx`, `WeeklyChart.tsx` (SVG, no chart library),
  `SimilarPlayers.tsx`. Types extend `web/src/api.ts`.
- Dark styling consistent with the board. Chart follows dataviz guidance
  (loaded at implementation time).

## Error handling

- Profile fetch failure → drawer shows readable error, board unaffected.
- Missing pieces (no snap match, no depth slot, no lines) render as "—";
  sections with zero data (rookie game log) collapse to a short note.

## Testing

- pytest: season aggregation math against hand-computed weekly rows; similarity
  (a cloned stat line ranks #1 with ~100 score; own seasons excluded; next_ppg
  correctness; rookie fallback returns value neighbors); profile endpoint shape,
  404, ADP-only player.
- Frontend: typecheck + build; manual browser verification.

## Out of scope (v2+)

- Cohort/grouping explorer (arbitrary stat filters across years)
- Multi-player side-by-side comparison view
- Weekly usage trend charts (target share by week)
