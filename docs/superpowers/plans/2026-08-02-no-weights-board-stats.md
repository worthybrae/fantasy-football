# No-Weights Board + Landing-Page Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Weights sidebar and every composite-model column from the draft board; make it one full-width list sorted by ESPN PPR rank showing real last-season stats.

**Architecture:** One backend change — `build_board` merges a nested `stats` dict (latest-season aggregates from the `weekly` table) onto each row, reusing `player_season_features()`. The frontend drops all weight plumbing (server-side query-param defaults take over), deletes the rail, and renders G/PPG/Pts always plus position-specific counting-stat columns on single-position tabs.

**Tech Stack:** FastAPI + pandas + DuckDB backend, React + TypeScript + @tanstack/react-table frontend, pytest.

**Spec:** `docs/superpowers/specs/2026-08-02-no-weights-board-stats-design.md`

## Global Constraints

- The API payload keeps `composite`/`vor`/`tier`/`edge`/`rank` fields (backend scoring machinery untouched); only the frontend stops rendering them.
- Board stats come only from the `weekly` table — no snap-share/target-share joins on the board.
- Rookies/K/DST (no weekly rows) get `stats = None` → em-dash cells.
- Default board sort stays `espn_ppr_rank` ascending.
- **The repo has unrelated uncommitted changes.** Every commit must `git add` only the exact files listed in its task — never `git add -A` or `git add .`.
- Python tests: `.venv/bin/python -m pytest tests/ -q` from repo root (or `make test`).
- Frontend typecheck/build: `cd web && npm run build` (or `make build` from root).

---

### Task 1: Backend — `stats` object on every board row

**Files:**
- Modify: `scoring/board.py` (imports ~line 48-52, `_BOARD_COLUMNS` ~line 60, new helper before `build_board`, merge inside `build_board` ~line 257)
- Test: `tests/test_board.py` (new test + update `test_board_shape_and_join`, `test_board_column_contract`, `test_empty_database_does_not_crash`)

**Interfaces:**
- Consumes: `player_season_features(weekly)` from `scoring/similarity.py` (returns per player-season: `games`, `points`, `ppg`, `targets`, `carries`, `rec_yards`, `rush_yards`, `tds`, `receptions`, …).
- Produces: each `build_board` row gains a `stats` column — a dict with keys `season, games, ppg, points, carries, rush_yards, targets, receptions, rec_yards, tds, completions, attempts, pass_yards, pass_tds, interceptions` (ints except `ppg`/`points`, rounded to 1 decimal) — or NaN (→ JSON `null`) for players with no weekly rows. Task 3's `BoardStats` TS interface mirrors this exactly.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_board.py` (the `_seed` fixture at the top of the file gives p1 17 games of 8 rec / 90 rec-yds / 10 targets in 2025, and an ADP-only "Rookie Guy"):

```python
def test_board_stats_summary(tmp_path):
    board = build_board(_seed(tmp_path))
    s = board[board["player_id"] == "p1"].iloc[0]["stats"]
    assert s["season"] == 2025 and s["games"] == 17
    assert s["receptions"] == 8 * 17
    assert s["rec_yards"] == 90 * 17
    assert s["targets"] == 10 * 17
    # PPR: 8 rec + 9.0 rec-yd pts = 17.0 per game
    assert s["ppg"] == 17.0
    assert s["points"] == 17.0 * 17
    # no passing columns in this fixture -> zero-filled, not missing
    assert s["pass_yards"] == 0 and s["attempts"] == 0
    # ADP-only player has no weekly rows -> no stats dict at all
    rook = board[board["name"] == "Rookie Guy"].iloc[0]
    assert not isinstance(rook["stats"], dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_board.py::test_board_stats_summary -q`
Expected: FAIL with `KeyError: 'stats'`

- [ ] **Step 3: Implement `_latest_season_stats` and merge it**

In `scoring/board.py`, add to the imports block:

```python
from scoring.similarity import player_season_features
```

Append `"stats"` to `_BOARD_COLUMNS`:

```python
_BOARD_COLUMNS = [
    "player_id", "name", "position", "team", "bye", "production", "durability",
    "role", "environment", "schedule", "composite", "vor", "tier", "market_rank",
    "market_spread", "market_sources", "espn_ppr_rank", "edge", "rookie",
    "drafted", "rank", "stats",
]
```

Add this helper above `build_board` (mirrors `profile.py`'s passing-aggregate pattern — passing stats are deliberately not part of `player_season_features`, which feeds twin matching and must not change):

```python
# Payload key -> weekly-table column for passing aggregates, which aren't
# part of player_season_features (that frame feeds twin matching in
# similarity.py and must not change).
_PASS_COLS = {
    "completions": "completions", "attempts": "attempts",
    "pass_yards": "passing_yards", "pass_tds": "passing_tds",
    "interceptions": "passing_interceptions",
}


def _latest_season_stats(weekly: pd.DataFrame) -> pd.DataFrame:
    """Per-player latest-season stat summary, one nested dict per row.

    Serialized like market_sources: the API's NaN->None pass turns a
    missing merge (rookie/K/DST with no weekly rows) into JSON null.
    """
    if weekly.empty:
        return pd.DataFrame(columns=["player_id", "stats"])
    feats = player_season_features(weekly)
    latest = feats[feats["season"] == feats["season"].max()].copy()

    wk = weekly[weekly["season"] == weekly["season"].max()].copy()
    for out, col in _PASS_COLS.items():
        wk[out] = (pd.to_numeric(wk[col], errors="coerce").fillna(0)
                   if col in wk.columns else 0.0)
    passing = wk.groupby("player_id", as_index=False)[list(_PASS_COLS)].sum()
    latest = latest.merge(passing, on="player_id", how="left")
    for out in _PASS_COLS:
        latest[out] = latest[out].fillna(0)

    latest["stats"] = latest.apply(lambda r: {
        "season": int(r["season"]), "games": int(r["games"]),
        "ppg": round(float(r["ppg"]), 1), "points": round(float(r["points"]), 1),
        "carries": int(r["carries"]), "rush_yards": int(r["rush_yards"]),
        "targets": int(r["targets"]), "receptions": int(r["receptions"]),
        "rec_yards": int(r["rec_yards"]), "tds": int(r["tds"]),
        "completions": int(r["completions"]), "attempts": int(r["attempts"]),
        "pass_yards": int(r["pass_yards"]), "pass_tds": int(r["pass_tds"]),
        "interceptions": int(r["interceptions"]),
    }, axis=1)
    return latest[["player_id", "stats"]]
```

In `build_board`, right after the `add_market(...)` line:

```python
    uni = add_market(uni, espn, fp, sleeper)
    uni = uni.merge(_latest_season_stats(weekly), on="player_id", how="left")
```

- [ ] **Step 4: Update the three column-contract tests**

In `tests/test_board.py`: append `"stats"` to the `expected` list in `test_board_column_contract`, to the `for col in [...]` list in `test_empty_database_does_not_crash`, and to the `for col in [...]` list in `test_board_shape_and_join`.

- [ ] **Step 5: Run the full python suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all PASS (if `test_board_stats_summary`'s ppg assertion fails on exact float equality, compare with `round(s["ppg"], 1)` — the dict already stores rounded values, so exact `== 17.0` should hold).

- [ ] **Step 6: Commit**

```bash
git add scoring/board.py tests/test_board.py
git commit -m "feat: latest-season stats object on every board row"
```

---

### Task 2: Frontend — remove weights plumbing and the rail

**Files:**
- Modify: `web/src/api.ts`, `web/src/App.tsx`, `web/src/components/PlayerProfile.tsx`, `web/src/App.css`
- Delete: `web/src/components/WeightSliders.tsx`

**Interfaces:**
- Consumes: server-side defaults on `/api/players` and `/api/players/{id}/profile` (all `w_*` query params optional).
- Produces: `fetchPlayers(): Promise<Player[]>` and `fetchProfile(playerId: string): Promise<PlayerProfileData>` — no weights anywhere. Task 3 builds on this state of `App.tsx`/`PlayerTable` props.

- [ ] **Step 1: Strip weights from `web/src/api.ts`**

Delete the `Weights` interface and `DEFAULT_WEIGHTS` const. Change the two fetchers to:

```ts
export async function fetchPlayers(): Promise<Player[]> {
  const res = await fetch('/api/players')
  if (!res.ok) {
    throw new Error(`Failed to load players (${res.status}): ${await detailText(res)}`)
  }
  return (await res.json()).players
}
```

```ts
export async function fetchProfile(playerId: string): Promise<PlayerProfileData> {
  const res = await fetch(`/api/players/${playerId}/profile`)
  if (!res.ok) throw new Error(`profile ${res.status}: ${await detailText(res)}`)
  return res.json()
}
```

- [ ] **Step 2: Remove weights + rail from `web/src/App.tsx`**

- Imports: drop `DEFAULT_WEIGHTS`, `type Weights`, and `WeightSliders`.
- Delete: `RAIL_COLLAPSED_KEY`, the `weights` state, the `railCollapsed` state + its localStorage effect.
- `loadPlayers` loses its parameter (`fetchPlayers()`), and the debounce effect becomes a plain mount effect:

```tsx
  useEffect(() => {
    loadPlayers()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
```

- `handleToggleDrafted`: `loadPlayers(weights)` → `loadPlayers()`, dep array `[weights]` → `[]` (update its comment — it's memoized for PlayerTable's `columns` memo, no longer weight-keyed).
- JSX: delete the entire `<aside className=...rail...>` block. Move the `hide-drafted` label into `<main>` on the same row as the tabs:

```tsx
        <main className="main">
          <div className="board-controls">
            <PositionTabs value={positionFilter} onChange={setPositionFilter} />
            <label className="hide-drafted">
              <input
                type="checkbox"
                checked={hideDrafted}
                onChange={(e) => setHideDrafted(e.target.checked)}
              />
              Hide drafted
            </label>
          </div>
```

- `<PlayerProfile ...>`: remove the `weights={weights}` prop.
- The board-keydown effect's comment mentions weight sliders — trim those references.

- [ ] **Step 3: Remove weights from `web/src/components/PlayerProfile.tsx`**

- Import: drop `type Weights`; props interface loses `weights: Weights`; destructure without it.
- `loadProfile`: `fetchProfile(forPlayerId, weights)` → `fetchProfile(forPlayerId)`.
- The refetch effect deps `[playerId, weights]` → `[playerId]`; since the drawer now only refetches on player swaps, `prevPlayerIdRef` and the `isNewPlayer` dance are dead — simplify to:

```tsx
  useEffect(() => {
    setProfile(null)
    loadProfile(playerId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playerId])
```

Update the comments above it (delete the weights-only-change explanation and `prevPlayerIdRef`).

- [ ] **Step 4: Delete the slider component and its CSS**

```bash
rm web/src/components/WeightSliders.tsx
```

In `web/src/App.css`: delete every `.rail`-prefixed rule block (`.rail`, `.rail.rail-collapsed`, `.rail-toggle`, `.rail.rail-collapsed .rail-toggle`, `.rail-toggle:hover`, `.rail-content`) and any `.weights`/slider-specific rules WeightSliders used (search `weight`). Adjust `.app-body` so `<main>` spans full width (it was a flex/grid row of rail + main; make it a single column — keep `.main`'s own overflow/height rules intact since it's the scroll container). Add:

```css
.board-controls {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
```

Keep the existing `.hide-drafted` rule (it still applies; adjust selector only if it was scoped under `.rail-content`).

- [ ] **Step 5: Typecheck/build**

Run: `cd web && npm run build`
Expected: PASS. (`PlayerTable` untouched this task — it never imported weights.)

- [ ] **Step 6: Commit**

```bash
git add web/src/api.ts web/src/App.tsx web/src/components/PlayerProfile.tsx web/src/App.css
git rm web/src/components/WeightSliders.tsx
git commit -m "feat: remove weights UI and rail; full-width board"
```

---

### Task 3: Frontend — stats columns, drop score columns

**Files:**
- Modify: `web/src/api.ts`, `web/src/statColumns.ts`, `web/src/components/PlayerTable.tsx`, `web/src/App.tsx`, `web/src/components/PlayerProfile.tsx`

**Interfaces:**
- Consumes: the `stats` dict from Task 1; the weights-free `App.tsx`/`api.ts` from Task 2.
- Produces: `BoardStats` interface in `api.ts`; `boardColumnsFor(positionFilter: string): BoardStatColumn[]` in `statColumns.ts`; `PlayerTable` props gain `positionFilter: string` and lose `showTierBreaks` + `maxVorByPosition`.

- [ ] **Step 1: Update the `Player` type in `web/src/api.ts`**

Add above `Player`:

```ts
export interface BoardStats {
  season: number; games: number; ppg: number; points: number;
  carries: number; rush_yards: number; targets: number; receptions: number;
  rec_yards: number; tds: number; completions: number; attempts: number;
  pass_yards: number; pass_tds: number; interceptions: number;
}
```

Replace the `Player` interface (drops `rank`, `tier`, the five factor fields, `composite`, `vor`, `edge`; adds `stats`):

```ts
export interface Player {
  player_id: string; name: string; position: string;
  team: string; bye: number | null;
  market_rank: number | null; market_spread: number | null; market_sources: MarketSources;
  espn_ppr_rank: number | null;
  stats: BoardStats | null;
  rookie: boolean; drafted: boolean;
}
```

(The API still sends the score fields; TypeScript just stops declaring them.)

- [ ] **Step 2: Add board stat column defs to `web/src/statColumns.ts`**

Import `BoardStats` from `./api` and append:

```ts
// Board (landing page) stat columns. `cell` renders the display string,
// `sortValue` the number TanStack sorts on -- split because Cmp/Att renders
// two numbers but sorts on attempts.
export interface BoardStatColumn {
  id: string
  label: string
  cell: (s: BoardStats) => string
  sortValue: (s: BoardStats) => number
}

export const BOARD_SUMMARY: BoardStatColumn[] = [
  { id: 'g', label: 'G', cell: (s) => n(s.games), sortValue: (s) => s.games },
  { id: 'ppg', label: 'PPG', cell: (s) => s.ppg.toFixed(1), sortValue: (s) => s.ppg },
  { id: 'pts', label: 'Pts', cell: (s) => s.points.toFixed(1), sortValue: (s) => s.points },
]

const BOARD_QB: BoardStatColumn[] = [
  { id: 'cmp_att', label: 'Cmp/Att', cell: (s) => `${s.completions}/${s.attempts}`, sortValue: (s) => s.attempts },
  { id: 'pass_yards', label: 'Pass Yds', cell: (s) => n(s.pass_yards), sortValue: (s) => s.pass_yards },
  { id: 'pass_tds', label: 'Pass TD', cell: (s) => n(s.pass_tds), sortValue: (s) => s.pass_tds },
  { id: 'interceptions', label: 'INT', cell: (s) => n(s.interceptions), sortValue: (s) => s.interceptions },
  { id: 'carries', label: 'Car', cell: (s) => n(s.carries), sortValue: (s) => s.carries },
  { id: 'rush_yards', label: 'Rush Yds', cell: (s) => n(s.rush_yards), sortValue: (s) => s.rush_yards },
]
const BOARD_RB: BoardStatColumn[] = [
  { id: 'carries', label: 'Car', cell: (s) => n(s.carries), sortValue: (s) => s.carries },
  { id: 'rush_yards', label: 'Rush Yds', cell: (s) => n(s.rush_yards), sortValue: (s) => s.rush_yards },
  { id: 'targets', label: 'Tgt', cell: (s) => n(s.targets), sortValue: (s) => s.targets },
  { id: 'receptions', label: 'Rec', cell: (s) => n(s.receptions), sortValue: (s) => s.receptions },
  { id: 'rec_yards', label: 'Rec Yds', cell: (s) => n(s.rec_yards), sortValue: (s) => s.rec_yards },
  { id: 'tds', label: 'TD', cell: (s) => n(s.tds), sortValue: (s) => s.tds },
]
const BOARD_WRTE: BoardStatColumn[] = [
  { id: 'targets', label: 'Tgt', cell: (s) => n(s.targets), sortValue: (s) => s.targets },
  { id: 'receptions', label: 'Rec', cell: (s) => n(s.receptions), sortValue: (s) => s.receptions },
  { id: 'rec_yards', label: 'Rec Yds', cell: (s) => n(s.rec_yards), sortValue: (s) => s.rec_yards },
  { id: 'carries', label: 'Car', cell: (s) => n(s.carries), sortValue: (s) => s.carries },
  { id: 'rush_yards', label: 'Rush Yds', cell: (s) => n(s.rush_yards), sortValue: (s) => s.rush_yards },
  { id: 'tds', label: 'TD', cell: (s) => n(s.tds), sortValue: (s) => s.tds },
]

// Summary always; a single-position tab appends that position's counting
// stats. ALL/FLEX/K/DST get summary only (mixed positions / no weekly stats).
export function boardColumnsFor(positionFilter: string): BoardStatColumn[] {
  if (positionFilter === 'QB') return [...BOARD_SUMMARY, ...BOARD_QB]
  if (positionFilter === 'RB') return [...BOARD_SUMMARY, ...BOARD_RB]
  if (positionFilter === 'WR' || positionFilter === 'TE') return [...BOARD_SUMMARY, ...BOARD_WRTE]
  return BOARD_SUMMARY
}
```

(`n` is the existing `const n = (v: number) => String(v)` helper — it stays.)

- [ ] **Step 3: Rework `web/src/components/PlayerTable.tsx` columns**

- Props: remove `showTierBreaks` and `maxVorByPosition` (and their doc comments); add `positionFilter: string`.
- Import `boardColumnsFor` from `../statColumns`.
- Delete the `rank`, `tier`, `vor`, `composite`, and `edge` column defs, plus the `edgeClass` helper and the VOR bar cell.
- After the `bye` and `market_rank` ("Mkt") columns, spread the stat columns; the `columns` memo becomes keyed on `[onToggleDrafted, positionFilter]`:

```tsx
      ...boardColumnsFor(positionFilter).map((c): ColumnDef<Player> => ({
        id: c.id,
        // null stats (rookies/K/DST) -> undefined so sortUndefined applies
        accessorFn: (row) => (row.stats ? c.sortValue(row.stats) : undefined),
        header: c.label,
        sortUndefined: 'last',
        sortDescFirst: true,
        cell: ({ row }) => (row.original.stats ? c.cell(row.original.stats) : '—'),
      })),
```

- Column order ends up: drafted-toggle · espn_ppr_rank · name · position · team · bye · market_rank · stat columns. Move the existing `market_rank` def before the spread so Mkt sits left of the stats.
- `NUMERIC_COLUMNS`: replace the old set with one that includes every stat column id, built next to it at module scope (WR covers the WR/TE set; `g`/`ppg`/`pts` are in every result):

```tsx
const NUMERIC_COLUMNS = new Set([
  'bye', 'market_rank', 'espn_ppr_rank',
  ...['QB', 'RB', 'WR'].flatMap((p) => boardColumnsFor(p).map((c) => c.id)),
])
```

The `<td>` className test (`NUMERIC_COLUMNS.has(cell.column.id) ? 'mono' : undefined`) stays as-is.
- Delete the tier-band machinery: `tierBandGate`, `tierBandFlags` memo, and the `tier-band` entry in the row `classNames` array.

- [ ] **Step 4: Update `web/src/App.tsx` wiring**

- Delete the `maxVorByPosition` memo (and its comment).
- `<PlayerTable ...>`: remove `showTierBreaks` and `maxVorByPosition` props; add `positionFilter={positionFilter}`.
- `selectedAnnouncement`: `rank ${p.rank}` no longer exists on the type — use ESPN PPR rank:

```tsx
    return p ? `${p.name}, ${p.position} ${p.team}, ESPN rank ${p.espn_ppr_rank ?? 'unranked'}` : ''
```

- [ ] **Step 5: Update the drawer header chips in `web/src/components/PlayerProfile.tsx`**

`header` is typed `Player`, which no longer has `rank`/`tier`/`vor`/`composite`/`edge`. Replace the chips block (currently Rank/Tier/VOR/Composite/Mkt/Edge) with market-only chips:

```tsx
              <div className="drawer-chips">
                <span className="chip">ESPN PPR {header.espn_ppr_rank !== null ? `#${Math.round(header.espn_ppr_rank)}` : '—'}</span>
                <span className="chip">Mkt {fmt1(header.market_rank)}</span>
              </div>
```

If nothing else uses `fmt1` after this, it still does (Mkt chip) — keep it.

- [ ] **Step 6: Typecheck/build**

Run: `cd web && npm run build`
Expected: PASS. If `tsc` flags other components reading dropped `Player` fields (e.g. a stray `header.production`), replace or remove those usages in the same spirit — the UI renders no model scores anywhere.

- [ ] **Step 7: Commit**

```bash
git add web/src/api.ts web/src/statColumns.ts web/src/components/PlayerTable.tsx web/src/App.tsx web/src/components/PlayerProfile.tsx
git commit -m "feat: board stat columns; drop composite/VOR/edge/rank/tier from UI"
```

---

### Task 4: End-to-end verification

**Files:** none modified (fixes go back through the relevant task's files if anything surfaces).

**Interfaces:** consumes the running app: `make api` (FastAPI :8000), `make web` (Vite :5173), or `make up` for both.

- [ ] **Step 1: Full test suite + build**

Run: `make test && make build`
Expected: both PASS.

- [ ] **Step 2: Smoke-check the API payload**

With the API running (`make api` in the background):

```bash
curl -s 'http://localhost:8000/api/players' | python3 -c "
import json,sys; ps=json.load(sys.stdin)['players']
withstats=[p for p in ps if p['stats']]; print(len(ps),'players,',len(withstats),'with stats')
print(withstats[0]['name'], withstats[0]['stats'])
rook=[p for p in ps if p['stats'] is None]; print('null-stats rows:', len(rook))"
```

Expected: most players have a stats dict with `season`/`games`/`ppg`/`points` etc.; rookies/K/DST rows show `stats: null` (non-zero count).

- [ ] **Step 3: Visual check in the browser**

Open `http://localhost:5173` and verify:
- No sidebar; table spans full width; "Hide drafted" sits on the tabs row.
- ALL tab columns: ✓ · ESPN PPR · Name · Pos · Team · Bye · Mkt · G · PPG · Pts (no Rank/Tier/VOR/Composite/Edge).
- RB tab adds Car/Rush Yds/Tgt/Rec/Rec Yds/TD; QB tab adds Cmp/Att/Pass Yds/Pass TD/INT/Car/Rush Yds.
- Rookie/K/DST rows show em-dashes in stat cells; clicking a stat header sorts descending first with dash rows last.
- Row click opens the drawer; chips show ESPN PPR + Mkt only; Esc closes; ↑/↓/Enter/D keyboard nav still works.

- [ ] **Step 4: Done — no commit needed unless fixes were made**
