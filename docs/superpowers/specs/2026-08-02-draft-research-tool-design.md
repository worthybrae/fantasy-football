# Fantasy Draft Research Tool — Design

**Date:** 2026-08-02
**Status:** Approved by user

## Purpose

Draft-prep research tool for an 8-team PPR league (QB/2RB/2WR/TE/2Flex/K/DST, 5 bench).
Produces the user's own player rankings from historical stats, depth charts, schedule
strength, and Vegas odds, displayed alongside market ADP so the gap between the two
surfaces draft-day edges. Primary use: live during the draft (August/September 2026).

The legacy Selenium/pro-football-reference scraper and its SQLite database are deleted —
the database was empty and nflverse now provides all of that data pre-cleaned.

## Architecture

```
nfl_data_py + FFC ADP API → pipeline (Python) → DuckDB file → FastAPI → React (Vite + TS)
```

Three units, each independently understandable and testable:

1. **Pipeline** (`pipeline/`) — fetches and stores raw data. Re-run to refresh.
2. **Scoring engine** (`scoring/`) — pure functions from stored data to ranked players.
3. **App** — FastAPI backend (`api/`) + React draft board (`web/`).

## Data pipeline

Sources (all free, no keys required):

- **Weekly player stats, 2018–2025** — `nfl_data_py.import_weekly_data`. PPR points
  computed under the league's exact scoring.
- **Snap counts, 2018–2025** — `nfl_data_py.import_snap_counts`.
- **Current depth charts** — `nfl_data_py.import_depth_charts` (updates through preseason).
- **2026 schedule with Vegas lines** — `nfl_data_py.import_schedules`; includes
  DraftKings `spread_line` and `total_line` → team implied points and schedule strength.
- **PPR ADP** — Fantasy Football Calculator free JSON API. Only 12-team PPR is offered;
  used as market consensus (the user's-rank-vs-ADP gap is what matters, not the absolute
  pick number).

Behavior:

- Single `refresh` CLI command (e.g. `python -m pipeline.refresh`), idempotent —
  drop-and-reload per source table into a single DuckDB file (`data/nfl.duckdb`).
- Graceful degradation: if preseason data is partial (missing Week 1 lines, sparse depth
  charts, ADP not yet live), the pipeline loads what exists and records per-source
  freshness in a `meta` table; downstream scoring treats missing factors as neutral and
  the UI shows data freshness.

## Scoring engine

Each fantasy-relevant player gets a composite score built from factors, each normalized
to 0–100 within position:

| Factor | Definition |
| --- | --- |
| Production | Recency-weighted PPR points/game over last 3 seasons (e.g. 50/30/20) |
| Durability | Games-played rate over last 3 seasons |
| Role | Depth chart rank + prior-year snap share and target share |
| Environment | Team implied points derived from Vegas spread/total lines |
| Schedule | Opponent-strength from lines + prior-year fantasy points allowed by position |

- Composite = weighted sum; weights live in one config object and are adjustable at
  runtime from the UI.
- **VOR**: composite is converted to value-over-replacement using this league's exact
  starting lineup (8 teams, QB/2RB/2WR/TE/2Flex/K/DST) — replacement level per position
  derived from starters + flex allocation. VOR ranks players across positions.
- Tiers: gap-based clustering on VOR within position.
- Rookies/no-history players: production/durability factors are neutral (position median);
  role and environment carry them. Flagged as rookies in output.
- Pure functions over dataframes; no I/O in scoring code.

## API

FastAPI, serving the React app's needs:

- `GET /players?weights=…` — ranked players with composite, per-factor breakdown, VOR,
  tier, ADP, edge (ADP rank − user rank), bye, position, team, drafted flag.
  Weights passed as query params (or defaults) — scoring recomputes on request; data is
  small enough (~400 players) that recompute-per-request is fine.
- `POST /drafted/{player_id}` / `DELETE /drafted/{player_id}` — draft-night state,
  persisted to a small local table so a refresh doesn't lose the board.
- `GET /meta` — data freshness per source.

## React draft board

Vite + React + TypeScript, TanStack Table:

- Sortable/filterable board: rank, tier, player, pos, team, bye, VOR, composite,
  per-factor breakdown (expandable), ADP, edge column.
- Position filter tabs (ALL/QB/RB/WR/TE/K/DST/FLEX).
- Weight sliders panel — moving a slider refetches `/players` with new weights.
- Click row → mark drafted (greyed out / hidden toggle).
- Data freshness indicator from `/meta`.

## Error handling

- Pipeline: per-source try/fail-loud with a summary at the end; a source failing doesn't
  abort the others; freshness table records last success.
- API: missing factor data → neutral score contribution, flagged in the breakdown.
- UI: empty/stale states driven by `/meta`.

## Testing

- **pytest on scoring** (the highest-risk unit): PPR point calculation against
  hand-computed examples, normalization, recency weighting, VOR replacement levels for
  this exact roster, tier clustering, rookie neutral-fill.
- Thin FastAPI tests (endpoints return coherent shapes).
- Frontend: kept simple, no test harness beyond TypeScript.

## Out of scope (v2+)

- ML projection model
- In-season tools (start/sit, waivers, weekly matchups)
- Player props / The Odds API integration
- Multi-league support
