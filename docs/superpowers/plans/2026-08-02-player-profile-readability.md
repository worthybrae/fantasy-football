# Player Profile Readability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the player-profile drawer's history readable and comparable: structured per-game stats from the API, position-aware columns in both tables, an aligned game-log table, chart tooltips carrying the stat line, and boom/bust shading on fantasy-point cells.

**Architecture:** The API (`scoring/profile.py`) starts sending structured per-game stats and season passing aggregates alongside the existing pre-formatted `stat_line`. A single frontend config (`web/src/statColumns.ts`) maps position → column definitions and drives both `GameLog` and `SeasonTable`, plus the point-shading helper. The drawer is already `min(70vw, 900px)` wide (App.css `.player-drawer`) — no layout widening needed.

**Tech Stack:** Python (pandas, DuckDB, FastAPI), pytest; React 19 + TypeScript, Vite, oxlint. No frontend test runner — verification is `tsc -b && vite build` + `oxlint`.

**Spec:** `docs/superpowers/specs/2026-08-02-player-profile-readability-design.md`

## Global Constraints

- Dark-only app; shading colors must derive from existing `--ok` (#34d399) / `--fail` (#f87171) hues at low alpha so `--text-h` text stays readable.
- `stat_line` stays in the payload (chart tooltip + existing tests use it).
- K/DST history suppression in `build_profile` unchanged.
- `similarity.py` `FEATURES`/aggregation must NOT change (twin matching would shift).
- Frontend requires the new `stats` shape outright — no dual-shape fallback.
- Python: run `.venv/bin/pytest -q` from repo root. Web: run `npm run build` and `npm run lint` from `web/`.
- Numbers in tables: right-aligned, `mono` class (`font-variant-numeric: tabular-nums` already set).

---

### Task 1: API — structured per-game stats in `game_log`

**Files:**
- Modify: `scoring/profile.py` (add `_GAME_STAT_COLS`, `_game_stats`; extend `game_log` rows)
- Test: `tests/test_profile.py`

**Interfaces:**
- Produces: each `game_log` row gains `"stats"`: a dict with int values for exactly these 12 keys: `completions, attempts, pass_yards, pass_tds, interceptions, carries, rush_yards, rush_tds, targets, receptions, rec_yards, rec_tds`. Missing weekly columns zero-fill. Task 3's `GameStats` type mirrors these keys.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_profile.py` (after `test_game_log_line_and_order`):

```python
def test_game_log_includes_structured_stats():
    rows = game_log(_weekly_rows(), "p1")
    s = rows[0]["stats"]
    assert s["targets"] == 9 and s["receptions"] == 6
    assert s["rec_yards"] == 80 and s["rec_tds"] == 1
    # columns absent from the weekly frame zero-fill
    assert s["pass_yards"] == 0 and s["completions"] == 0 and s["carries"] == 0
    assert set(s) == {"completions", "attempts", "pass_yards", "pass_tds",
                      "interceptions", "carries", "rush_yards", "rush_tds",
                      "targets", "receptions", "rec_yards", "rec_tds"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_profile.py::test_game_log_includes_structured_stats -q`
Expected: FAIL with `KeyError: 'stats'`

- [ ] **Step 3: Implement**

In `scoring/profile.py`, below `_stat_line` add:

```python
# Payload key -> weekly-table column. Every game_log row carries all 12 keys
# (zero-filled when the column is absent) so the frontend's position-aware
# column configs can index into a uniform shape.
_GAME_STAT_COLS = {
    "completions": "completions", "attempts": "attempts",
    "pass_yards": "passing_yards", "pass_tds": "passing_tds",
    "interceptions": "passing_interceptions",
    "carries": "carries", "rush_yards": "rushing_yards",
    "rush_tds": "rushing_tds", "targets": "targets",
    "receptions": "receptions", "rec_yards": "receiving_yards",
    "rec_tds": "receiving_tds",
}


def _game_stats(row):
    return {k: int(_num(row, col)) for k, col in _GAME_STAT_COLS.items()}
```

In `game_log`, add to the appended row dict (before `"ppr_points"`):

```python
            "stats": _game_stats(r),
```

- [ ] **Step 4: Run the full python suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (the api-shape test `test_api.py:159` checks superset membership, unaffected).

- [ ] **Step 5: Commit**

```bash
git add scoring/profile.py tests/test_profile.py
git commit -m "feat: structured per-game stats in profile game_log"
```

---

### Task 2: API — passing aggregates in `season_summaries`

**Files:**
- Modify: `scoring/profile.py:season_summaries`
- Test: `tests/test_profile.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: each season row gains int fields `completions, attempts, pass_yards, pass_tds, interceptions` (zero when columns absent). Task 3's `SeasonSummary` type adds the same five fields.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_profile.py`:

```python
def _qb_weekly_rows():
    return pd.DataFrame(
        [{"player_id": "q1", "player_display_name": "QB Guy",
          "position": "QB", "recent_team": "DET", "opponent_team": "GB",
          "season": 2025, "week": w, "completions": 20, "attempts": 30,
          "passing_yards": 250, "passing_tds": 2, "passing_interceptions": 1,
          "carries": 4, "rushing_yards": 20, "rushing_tds": 0,
          "targets": 0, "receptions": 0, "receiving_yards": 0,
          "receiving_tds": 0}
         for w in range(1, 6)])

def test_season_summaries_passing_aggregates():
    s = season_summaries(_qb_weekly_rows(), pd.DataFrame(), "q1")[0]
    assert s["completions"] == 100 and s["attempts"] == 150
    assert s["pass_yards"] == 1250 and s["pass_tds"] == 10
    assert s["interceptions"] == 5

def test_season_summaries_passing_zero_filled_without_columns():
    # WR fixture has no passing columns at all -- fields must still exist
    s = season_summaries(_weekly_rows(), pd.DataFrame(), "p1")[0]
    assert s["pass_yards"] == 0 and s["attempts"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_profile.py -q -k passing`
Expected: both FAIL with `KeyError`

- [ ] **Step 3: Implement**

In `season_summaries`, after the snap-share merge/else block and before `mine = mine.sort_values(...)`:

```python
    # Passing aggregates aren't part of player_season_features (that frame
    # feeds twin matching in similarity.py and must not change) -- aggregate
    # them here from the raw weekly rows instead.
    pass_cols = {"completions": "completions", "attempts": "attempts",
                 "pass_yards": "passing_yards", "pass_tds": "passing_tds",
                 "interceptions": "passing_interceptions"}
    wk_mine = weekly[weekly["player_id"] == player_id].copy()
    for out, col in pass_cols.items():
        wk_mine[out] = (pd.to_numeric(wk_mine[col], errors="coerce").fillna(0)
                        if col in wk_mine.columns else 0.0)
    passing = wk_mine.groupby("season", as_index=False)[list(pass_cols)].sum()
    mine = mine.merge(passing, on="season", how="left")
    for out in pass_cols:
        mine[out] = mine[out].fillna(0)
```

In the row-building loop, add (after `"ppg"`):

```python
            "completions": int(r["completions"]),
            "attempts": int(r["attempts"]),
            "pass_yards": int(r["pass_yards"]),
            "pass_tds": int(r["pass_tds"]),
            "interceptions": int(r["interceptions"]),
```

- [ ] **Step 4: Run the full python suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scoring/profile.py tests/test_profile.py
git commit -m "feat: season passing aggregates in profile summaries"
```

---

### Task 3: Frontend — types and shared position→columns config

**Files:**
- Modify: `web/src/api.ts` (add `GameStats`, extend `GameLogRow` + `SeasonSummary`)
- Create: `web/src/statColumns.ts`

**Interfaces:**
- Consumes: Task 1/2 payload shapes.
- Produces (used by Tasks 4–6):
  - `interface GameStats` — 12 numeric fields matching Task 1's keys.
  - `gameColumnsFor(position: string): StatColumn<GameStats>[]`
  - `seasonColumnsFor(position: string): StatColumn<SeasonSummary>[]`
  - `ptsShadeClass(points: number, avg: number): string` — returns `''` or one of `pts-hot-1|pts-hot-2|pts-cold-1|pts-cold-2`.
  - `interface StatColumn<T> { label: string; value: (row: T) => string }`

- [ ] **Step 1: Extend `web/src/api.ts`**

Replace the `SeasonSummary` and `GameLogRow` interfaces with:

```ts
export interface SeasonSummary {
  season: number; games: number; ppg: number; targets: number;
  target_share: number | null; carries: number; rec_yards: number;
  rush_yards: number; tds: number; receptions: number;
  yards_per_opp: number | null; snap_share: number | null;
  completions: number; attempts: number; pass_yards: number;
  pass_tds: number; interceptions: number;
}
export interface GameStats {
  completions: number; attempts: number; pass_yards: number;
  pass_tds: number; interceptions: number; carries: number;
  rush_yards: number; rush_tds: number; targets: number;
  receptions: number; rec_yards: number; rec_tds: number;
}
export interface GameLogRow {
  season: number; week: number; opponent: string | null; stat_line: string;
  stats: GameStats; ppr_points: number;
}
```

- [ ] **Step 2: Create `web/src/statColumns.ts`**

```ts
import type { GameStats, SeasonSummary } from './api'

// One position -> columns definition drives both the season table and the
// game log, so the two read consistently. Add a position here and both
// tables pick it up.
export interface StatColumn<T> {
  label: string
  value: (row: T) => string
}

const n = (v: number) => String(v)

const GAME_QB: StatColumn<GameStats>[] = [
  { label: 'Cmp/Att', value: (s) => `${s.completions}/${s.attempts}` },
  { label: 'Pass Yds', value: (s) => n(s.pass_yards) },
  { label: 'Pass TD', value: (s) => n(s.pass_tds) },
  { label: 'INT', value: (s) => n(s.interceptions) },
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
]
const GAME_RB: StatColumn<GameStats>[] = [
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
  { label: 'Rush TD', value: (s) => n(s.rush_tds) },
  { label: 'Tgt', value: (s) => n(s.targets) },
  { label: 'Rec', value: (s) => n(s.receptions) },
  { label: 'Rec Yds', value: (s) => n(s.rec_yards) },
]
const GAME_WRTE: StatColumn<GameStats>[] = [
  { label: 'Tgt', value: (s) => n(s.targets) },
  { label: 'Rec', value: (s) => n(s.receptions) },
  { label: 'Rec Yds', value: (s) => n(s.rec_yards) },
  { label: 'Rec TD', value: (s) => n(s.rec_tds) },
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
]

export function gameColumnsFor(position: string): StatColumn<GameStats>[] {
  if (position === 'QB') return GAME_QB
  if (position === 'RB') return GAME_RB
  return GAME_WRTE // WR/TE and any fallback
}

const fmtPct = (v: number | null) => (v === null ? '—' : `${(v * 100).toFixed(1)}%`)
const fmt1 = (v: number | null) => (v === null ? '—' : v.toFixed(1))

// Shared season-table tail (skill/usage rates); QB gets only Snap% -- target
// share and yards-per-opportunity are receiving/rushing constructs.
const SEASON_QB: StatColumn<SeasonSummary>[] = [
  { label: 'Cmp/Att', value: (s) => `${s.completions}/${s.attempts}` },
  { label: 'Pass Yds', value: (s) => n(s.pass_yards) },
  { label: 'Pass TD', value: (s) => n(s.pass_tds) },
  { label: 'INT', value: (s) => n(s.interceptions) },
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
  { label: 'Snap%', value: (s) => fmtPct(s.snap_share) },
]
const SEASON_RB: StatColumn<SeasonSummary>[] = [
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
  { label: 'Tgt', value: (s) => n(s.targets) },
  { label: 'Rec', value: (s) => n(s.receptions) },
  { label: 'Rec Yds', value: (s) => n(s.rec_yards) },
  { label: 'TD', value: (s) => n(s.tds) },
  { label: 'Tgt%', value: (s) => fmtPct(s.target_share) },
  { label: 'Yds/Opp', value: (s) => fmt1(s.yards_per_opp) },
  { label: 'Snap%', value: (s) => fmtPct(s.snap_share) },
]
const SEASON_WRTE: StatColumn<SeasonSummary>[] = [
  { label: 'Tgt', value: (s) => n(s.targets) },
  { label: 'Rec', value: (s) => n(s.receptions) },
  { label: 'Rec Yds', value: (s) => n(s.rec_yards) },
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
  { label: 'TD', value: (s) => n(s.tds) },
  { label: 'Tgt%', value: (s) => fmtPct(s.target_share) },
  { label: 'Yds/Opp', value: (s) => fmt1(s.yards_per_opp) },
  { label: 'Snap%', value: (s) => fmtPct(s.snap_share) },
]

export function seasonColumnsFor(position: string): StatColumn<SeasonSummary>[] {
  if (position === 'QB') return SEASON_QB
  if (position === 'RB') return SEASON_RB
  return SEASON_WRTE
}

// Boom/bust shading relative to the player's own average: a game at 150%+
// of average reads clearly "boom", under 50% clearly "bust", with a lighter
// step either side and a neutral band around average so ordinary games stay
// unshaded.
export function ptsShadeClass(points: number, avg: number): string {
  if (avg <= 0) return ''
  const ratio = points / avg
  if (ratio >= 1.5) return 'pts-hot-2'
  if (ratio >= 1.15) return 'pts-hot-1'
  if (ratio <= 0.5) return 'pts-cold-2'
  if (ratio <= 0.85) return 'pts-cold-1'
  return ''
}
```

- [ ] **Step 3: Verify it compiles**

Run (from `web/`): `npm run build`
Expected: FAILS — `GameLog.tsx`/`SeasonTable.tsx` don't use new fields yet but `stats` is now required on `GameLogRow`; if the build passes (no consumer constructs a `GameLogRow` literal), that's fine too. Either way `npm run lint` must not report new issues in the two files just touched. Do NOT fix other components in this task — Tasks 4–5 own them. If the build fails only in files owned by later tasks, note it and proceed.

- [ ] **Step 4: Commit**

```bash
git add web/src/api.ts web/src/statColumns.ts
git commit -m "feat: frontend types + shared position stat-column config"
```

---

### Task 4: Game log becomes an aligned, shaded table

**Files:**
- Modify: `web/src/components/GameLog.tsx` (full rewrite)
- Modify: `web/src/components/PlayerProfile.tsx:166` (pass `position`)
- Modify: `web/src/App.css` (replace `.game-log-*` list styles with table styles; add `.pts-*` shading classes)

**Interfaces:**
- Consumes: `gameColumnsFor`, `ptsShadeClass` from `web/src/statColumns.ts`; `GameLogRow.stats`.
- Produces: `GameLog` props become `{ gameLog: GameLogRow[]; position: string }`.

- [ ] **Step 1: Rewrite `GameLog.tsx`**

```tsx
import type { GameLogRow } from '../api'
import { gameColumnsFor, ptsShadeClass } from '../statColumns'

interface GameLogProps {
  gameLog: GameLogRow[]
  position: string
}

export default function GameLog({ gameLog, position }: GameLogProps) {
  if (gameLog.length === 0) {
    return <p className="drawer-placeholder">No game log available.</p>
  }

  const columns = gameColumnsFor(position)

  // Group into season blocks without re-sorting -- the API already returns
  // newest-first (season desc, week desc), and grouping by first-encounter
  // order preserves that even if a future caller passes a differently
  // ordered list.
  const groups: { season: number; games: GameLogRow[] }[] = []
  for (const game of gameLog) {
    const current = groups[groups.length - 1]
    if (current && current.season === game.season) {
      current.games.push(game)
    } else {
      groups.push({ season: game.season, games: [game] })
    }
  }

  // One table across all seasons keeps every column aligned vertically --
  // that's the whole point of the log -- with a full-width season row
  // separating the blocks. Shading compares each game to its own season's
  // average, not the career's, so a breakout year doesn't paint every
  // earlier season cold.
  return (
    <div className="game-log-wrap">
      <table className="game-log-table mono">
        <thead>
          <tr>
            <th>Wk</th>
            <th className="game-log-opp">Opp</th>
            {columns.map((c) => (
              <th key={c.label}>{c.label}</th>
            ))}
            <th>Pts</th>
          </tr>
        </thead>
        {groups.map(({ season, games }) => {
          const avg = games.reduce((sum, g) => sum + g.ppr_points, 0) / games.length
          return (
            <tbody key={season}>
              <tr className="game-log-season-row">
                <td colSpan={columns.length + 3}>
                  {season} · avg {avg.toFixed(1)} pts
                </td>
              </tr>
              {games.map((g) => (
                <tr key={`${g.season}-${g.week}`}>
                  <td>{g.week}</td>
                  <td className="game-log-opp">{g.opponent ?? '—'}</td>
                  {columns.map((c) => (
                    <td key={c.label}>{c.value(g.stats)}</td>
                  ))}
                  <td className={ptsShadeClass(g.ppr_points, avg)}>
                    {g.ppr_points.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          )
        })}
      </table>
    </div>
  )
}
```

- [ ] **Step 2: Pass position from `PlayerProfile.tsx`**

Change the Game log section (`PlayerProfile.tsx:164-167`) to:

```tsx
            <section className="drawer-section">
              <h3>Game log</h3>
              <GameLog gameLog={profile.game_log} position={header.position} />
            </section>
```

- [ ] **Step 3: Replace the game-log CSS in `App.css`**

Delete the `.game-log-summary`, `.game-log-summary:hover`, `.game-log-list`, and `.game-log-row` rules (keep `.game-log-season` / `.game-log-season-header` only if still referenced — after this rewrite they are not, so delete them too). Add:

```css
.game-log-wrap {
  overflow-x: auto;
}

.game-log-table {
  font-size: 12px;
  white-space: nowrap;
  border-collapse: collapse;
  width: 100%;
}

.game-log-table th,
.game-log-table td {
  text-align: right;
  padding: 3px 8px;
  border-bottom: 1px solid var(--border);
}

.game-log-table th {
  color: var(--text-h);
  font-weight: 600;
}

.game-log-table .game-log-opp {
  text-align: left;
}

.game-log-season-row td {
  text-align: left;
  color: var(--text-h);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  padding-top: 12px;
}

/* boom/bust shading -- --ok/--fail hues at low alpha so text stays legible */
.pts-hot-1 { background: rgba(52, 211, 153, 0.12); }
.pts-hot-2 { background: rgba(52, 211, 153, 0.28); }
.pts-cold-1 { background: rgba(248, 113, 113, 0.10); }
.pts-cold-2 { background: rgba(248, 113, 113, 0.24); }
```

- [ ] **Step 4: Verify build and lint**

Run (from `web/`): `npm run build && npm run lint`
Expected: build may still fail in `SeasonTable.tsx` ONLY if Task 3 introduced a compile error there (it should not have — SeasonTable's fields are additive). Everything else clean.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/GameLog.tsx web/src/components/PlayerProfile.tsx web/src/App.css
git commit -m "feat: aligned position-aware game-log table with boom/bust shading"
```

---

### Task 5: Position-aware season table with PPG shading

**Files:**
- Modify: `web/src/components/SeasonTable.tsx` (full rewrite)
- Modify: `web/src/components/PlayerProfile.tsx:161` (pass `position`)

**Interfaces:**
- Consumes: `seasonColumnsFor`, `ptsShadeClass` from `web/src/statColumns.ts`.
- Produces: `SeasonTable` props become `{ seasons: SeasonSummary[]; position: string }`.

- [ ] **Step 1: Rewrite `SeasonTable.tsx`**

```tsx
import type { SeasonSummary } from '../api'
import { ptsShadeClass, seasonColumnsFor } from '../statColumns'

interface SeasonTableProps {
  seasons: SeasonSummary[]
  position: string
}

export default function SeasonTable({ seasons, position }: SeasonTableProps) {
  if (seasons.length === 0) {
    return <p className="drawer-placeholder">No season history available.</p>
  }

  const columns = seasonColumnsFor(position)
  const rows = [...seasons].sort((a, b) => b.season - a.season)

  // Games-weighted career PPG, so a 3-game injury season doesn't drag the
  // reference point the way a plain mean of season PPGs would.
  const totalGames = rows.reduce((sum, s) => sum + s.games, 0)
  const careerPpg = totalGames > 0
    ? rows.reduce((sum, s) => sum + s.ppg * s.games, 0) / totalGames
    : 0

  return (
    <div className="season-table-wrap">
      <table className="season-table mono">
        <thead>
          <tr>
            <th>Season</th>
            <th>G</th>
            <th>PPG</th>
            {columns.map((c) => (
              <th key={c.label}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => (
            <tr key={s.season}>
              <td>{s.season}</td>
              <td>{s.games}</td>
              <td className={ptsShadeClass(s.ppg, careerPpg)}>{s.ppg.toFixed(1)}</td>
              {columns.map((c) => (
                <td key={c.label}>{c.value(s)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 2: Pass position from `PlayerProfile.tsx`**

Change the Season history section (`PlayerProfile.tsx:159-162`) to:

```tsx
            <section className="drawer-section">
              <h3>Season history</h3>
              <SeasonTable seasons={profile.seasons} position={header.position} />
            </section>
```

- [ ] **Step 3: Align season-table cells with the game log**

In `App.css`, replace the `.season-table` rule with:

```css
.season-table {
  font-size: 12px;
  white-space: nowrap;
  border-collapse: collapse;
  width: 100%;
}

.season-table th,
.season-table td {
  text-align: right;
  padding: 3px 8px;
  border-bottom: 1px solid var(--border);
}

.season-table th {
  color: var(--text-h);
  font-weight: 600;
}

.season-table th:first-child,
.season-table td:first-child {
  text-align: left;
}
```

- [ ] **Step 4: Verify build and lint**

Run (from `web/`): `npm run build && npm run lint`
Expected: clean — this is the last consumer of the new types.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/SeasonTable.tsx web/src/components/PlayerProfile.tsx web/src/App.css
git commit -m "feat: position-aware season table with PPG shading"
```

---

### Task 6: Chart tooltip stat lines + final verification

**Files:**
- Modify: `web/src/components/WeeklyChart.tsx:192` (tooltip title)

**Interfaces:**
- Consumes: `GameLogRow.stat_line` (already in payload — the server's `_stat_line` formatting is nicer prose than joining column values, and it's already covered by python tests; the spec's "format from stats via config" is satisfied more simply by reusing it).

- [ ] **Step 1: Extend the bar tooltip**

In `WeeklyChart.tsx`, change the `title` construction (line ~192) to:

```tsx
            const title = `${seasonTag(game.season)} wk ${game.week} — ${game.ppr_points.toFixed(1)} pts vs ${game.opponent} — ${game.stat_line}`
```

- [ ] **Step 2: Full verification pass**

Run: `.venv/bin/pytest -q` (repo root) and `npm run build && npm run lint` (from `web/`).
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add web/src/components/WeeklyChart.tsx
git commit -m "feat: weekly-chart tooltips carry the full stat line"
```
