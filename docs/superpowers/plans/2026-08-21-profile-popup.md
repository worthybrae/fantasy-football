# Player Profile Popup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the in-draft player profile as a single compact popup drawn in the board's own chart language, with every card on screen at once.

**Architecture:** `Chart` and its `Col` type come out of `CellTip.tsx` into a shared module so the hover panels and the popup draw from one implementation. The four season panels are built by pure functions that both callers share, so a season cannot be green in one place and olive in the other. Layout is then rebuilt inside `PlayerProfile.tsx`, which already renders only inside `PlayerOverlay`.

**Tech Stack:** React 19, TypeScript 6, Vite 8, plain CSS in `web/src/App.css`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-21-profile-popup-design.md`

**Design reference:** `.design/profile-popup/Main.dc.html` and `Sparse.dc.html` are the markup spec — exact sizes, gaps, colours and copy. Read the artboard for the section you are building. Published canvas: https://claude.ai/code/artifact/988e9406-682b-4dbc-9b41-a9cd94d7818f

## Global Constraints

- **No new dependencies.** `web/package.json` gains nothing.
- **Verification is build + lint + render.** This repo has no front-end test framework and is not getting one in this plan. Every task ends with `npm run build` and `npm run lint`, both run **bare** — never piped into `tail`, `head`, or anything else, because a pipe returns the pipe's exit code and hides a failing `tsc -b`. Check the exit code.
- **The Python suite is not touched.** If you need it anyway, copy the DB first: a dev server holds a lock on `data/nfl.duckdb`. Run `cp data/nfl.duckdb /tmp/t.duckdb && DRAFT_DB_PATH=/tmp/t.duckdb .venv/bin/pytest -q`.
- **Colour means one thing.** Every panel, the schedule strip and the game log's points column take the same five-step ramp with the same cut points: `#0ca30c` / `#7cb342` / `#b3c132` / `#ffb454` / `#d03b3b`, already defined as the `.ctip-col-*` tone classes at `web/src/App.css:4854`.
- **Comment the why, not the what.** Match the register of the file you are in — this codebase explains decisions, not syntax. No emoji.
- **Two real payloads to check against**, both from the running API on :8000:
  - full: `GET /api/players/00-0039139/profile` (Jahmyr Gibbs, 3 seasons)
  - sparse: `GET /api/players/adp_jeremiyah_love/profile` (rookie, 0 seasons, Questionable)
- **Commit per task**, on `main`, with the trailers this repo uses.

---

### Task 1: Extract the shared chart

**Files:**
- Create: `web/src/components/draft/Chart.tsx`
- Modify: `web/src/components/draft/CellTip.tsx` (delete the local `Col` type at :61 and `Chart` at :72, import them instead)

**Interfaces:**
- Produces: `export type Col = { key: string | number; label: string; value: string; tone: string; fill: number; units?: { filled: number; of: number }; empty?: boolean; projected?: boolean }` and `export function Chart({ cols }: { cols: Col[] }): ReactNode`

- [ ] **Step 1: Create the module by moving the code verbatim**

Move the `Col` type and the `Chart` function out of `CellTip.tsx` into `web/src/components/draft/Chart.tsx` with no behavioural change — same JSX, same class names, same `Math.max(6, c.fill * 100)` floor, same `is-proj` / `is-forecast` handling. Export both. Carry the explanatory comments with them, and add one at the top saying why the module exists: the profile popup draws the same picture, and a second implementation is how the two would drift.

- [ ] **Step 2: Import them back into CellTip**

```tsx
import { Chart, type Col } from './Chart'
```

- [ ] **Step 3: Verify the hover panels are unchanged**

Run bare, checking exit codes:

```bash
cd web && npm run build
npm run lint
```

Expected: exit 0 from both. `tsc -b` proves every `Col` construction in `CellTip.tsx` still type-checks against the moved type.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/draft/Chart.tsx web/src/components/draft/CellTip.tsx
git commit -m "refactor: the hover panels' chart moves somewhere the profile can reach it"
```

---

### Task 2: Share the panel builders

**Files:**
- Create: `web/src/components/draft/panels.ts`
- Modify: `web/src/components/draft/CellTip.tsx` (`HealthBody`, `FinishBody`, `SteadyBody`, `ChangeBody` call the shared builders instead of building `Col[]` inline)

**Interfaces:**
- Consumes: `Col` from Task 1.
- Produces, all taking the profile payload's `seasons` array (and what each needs beside it), all returning `Col[]`:
  - `healthCols(seasons: SeasonSummary[]): Col[]`
  - `finishCols(seasons: SeasonSummary[], starters: number): Col[]`
  - `steadyCols(seasons: SeasonSummary[]): Col[]`
  - `perGameCols(seasons: SeasonSummary[], projPpg: number | null, position: string): Col[]`
  - `weekCols(gameLog: GameLogRow[], season: number, position: string): Col[]`
  - and `steadyTone(rank: number, pool: number): string`, moved out of `CellTip.tsx`.

- [ ] **Step 1: Move the four builders**

Lift the `cols` construction out of each of `CellTip.tsx`'s four bodies into these functions, unchanged: `availabilityRows` + `HEALTH_TONES` for health; `finishTone` / `finishPosition` for finish; `steadyTone` and the `1 - rank / pool` fill for steady; `barTone` plus the ceiling and the hollow `projected: true` column for per game. Keep `seasonLength()` with health — it is the reason a 16-game 2019 is not drawn as one short.

- [ ] **Step 2: Rewrite the four bodies to call them**

Each body keeps its own empty-state guard, its own head and note, and hands `Chart` the builder's output. `ChangeBody`'s note (`+/-N vs last season`) and `SteadyBody`'s (`1 = steadiest of N RBs`) stay where they are — they are copy, not chart maths.

- [ ] **Step 3: Verify**

```bash
cd web && npm run build
npm run lint
```

Expected: exit 0 from both.

- [ ] **Step 4: Check the hover panels by eye**

Start the app (`make up` from the repo root), open `/draft`, hover the Health, Games, Finish, Steady and Change cells on a row with three seasons. Each panel must look exactly as it did before this task. Screenshot them.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/draft/panels.ts web/src/components/draft/CellTip.tsx
git commit -m "refactor: one place decides what a season's column looks like"
```

---

### Task 3: The popup's header, status line and four panels

**Files:**
- Modify: `web/src/components/PlayerProfile.tsx` (replace the header block and the first grid cells)
- Modify: `web/src/App.css` (new `.pp-pop-*` rules)
- Reference: `.design/profile-popup/Main.dc.html`, top three sections

**Interfaces:**
- Consumes: `Chart` (Task 1), `healthCols` / `finishCols` / `steadyCols` / `perGameCols` (Task 2).

- [ ] **Step 1: Build the header row**

Name, position chip, then `team · age N · NFL yr N · bye N` from `bio` and `header`. Right-aligned: Your board, ADP, Tier, VOR — `header.rank`, `header.market_rank`, `header.tier`, `header.vor`. This replaces `VerdictStrip`; delete its usage here, not the file (Task 6 decides its fate).

- [ ] **Step 2: Build the status line**

One row: injury designation from `status`, the depth-chart slot, and the edge over consensus on the right. Green check and neutral border when `status.injury_status` is null; amber triangle and `--accent-border` when it is not. `InjuryStatus.tsx` already owns this decision — reuse it rather than re-deriving. This also absorbs `RoomGap`'s edge number.

- [ ] **Step 3: Build the four panels**

One flex row, four cards, each a title, a mono note, and a `<Chart>`: Health (`of 17`), Finish (position), Steady (`of N`), Per game (the delta). Per game takes `grow: 1.35` — it has four columns to the others' three.

- [ ] **Step 4: Verify both payloads**

```bash
cd web && npm run build
npm run lint
```

Then render against the live API and screenshot: Gibbs (`00-0039139`) shows three season columns per panel; the rookie (`adp_jeremiyah_love`) shows baseline marks and the projection alone. Neither may overflow its card.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/PlayerProfile.tsx web/src/App.css
git commit -m "feat: the profile opens with the four panels the board already taught"
```

---

### Task 4: The week chart and its game log

**Files:**
- Modify: `web/src/components/profile/WeekByWeek.tsx`
- Modify: `web/src/components/PlayerProfile.tsx` (the card that holds it)
- Modify: `web/src/App.css`
- Reference: `.design/profile-popup/Main.dc.html`, the "2025 by week" card

- [ ] **Step 1: Redraw the chart in the shared language**

`WeekByWeek` builds its columns with `weekCols` from Task 2 and renders `<Chart>`. A bye or a missed week is the baseline mark, not a zero-height bar — "did not play" and "played and scored nothing" are different claims and the existing code already knows it.

- [ ] **Step 2: Add the log underneath, in the same card**

A hairline (`1px`, `--border`), then rows: week, opponent, `stat_line` (ellipsised, one line), and `ppr_points` coloured by the same `barTone` as the bar above it. Header row in 8px caps: Wk / Opp / Line / Pts.

- [ ] **Step 3: Cap the visible rows**

Six most recent played games, the rest scrolling inside the card (`max-height`, `overflow-y: auto`). Do not render DNPs as log rows — they are already in the chart above as baseline marks, and a log row saying nothing happened is a row that costs a reader a line.

- [ ] **Step 4: Verify**

```bash
cd web && npm run build
npm run lint
```

Gibbs must show 18 chart columns and a scrollable log whose first row is week 18 at CHI, `19 car, 80 yds, 0 TD · 3 rec, 33 yds`, 20.3. The rookie has no `game_log` at all: the whole card must be absent, not an empty frame.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/profile/WeekByWeek.tsx web/src/components/PlayerProfile.tsx web/src/App.css
git commit -m "feat: the week chart carries the games it is made of"
```

---

### Task 5: The two dense card rows

**Files:**
- Modify: `web/src/components/profile/ScheduleRanks.tsx`, `DepthChartCard`, `MarketRow.tsx`, `UsageLine.tsx`, `LineQuality.tsx`, `ValueNeighbors.tsx`
- Modify: `web/src/components/PlayerProfile.tsx`, `web/src/App.css`
- Reference: `.design/profile-popup/Main.dc.html`, rows "Schedule · Room · Market" and "Usage · Blocking · Near you"

- [ ] **Step 1: Row one — Schedule, Room, Market**

Schedule becomes the 18-bar softness strip with `schedule[].pct` on the ramp and a note of the season figure (`6th softest of 32`). Room is the position group from `depth_chart` with the player's own row in `--text-1`. Market is the ranking sources as label/value rows.

- [ ] **Step 2: Row two — Usage, Blocking, Near you**

Usage is per-game carries, targets and yards from `summary.w_stats`. Blocking leads with `oline.rank` as a large mono figure over `of 32`, then continuity and returning. Near you is the board's rank neighbours with the player's own row accented.

- [ ] **Step 3: Keep each card's own empty state**

A card with nothing to say renders nothing. The rookie has no `summary.w_stats` — Usage must be absent rather than a frame of dashes.

- [ ] **Step 4: Verify**

```bash
cd web && npm run build
npm run lint
```

Screenshot both payloads. Six cards for Gibbs; the rookie drops Usage and keeps the rest.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/profile web/src/components/PlayerProfile.tsx web/src/App.css
git commit -m "feat: six cards where the profile had six sections"
```

---

### Task 6: Comparable seasons, news, and the sparse notice

**Files:**
- Modify: `web/src/components/profile/SimilarPlayers` (wherever it lives — it is imported by `PlayerProfile.tsx`), `CohortNext.tsx`, `NewsPanel.tsx`, `MissingData.tsx`
- Modify: `web/src/components/PlayerProfile.tsx`, `web/src/App.css`
- Reference: `.design/profile-popup/Main.dc.html` bottom row, and all of `Sparse.dc.html`

- [ ] **Step 1: Comparable seasons**

Player, season, their PPG, and the move their next year made — green up, red down. The card's note carries the cohort verdict from `cohort`: `13 of 19 declined`. `similar.mode` decides the title: `stat_twins` means "Comparable seasons", `value_neighbors` means "Near you" with the note `no stat twins yet`.

- [ ] **Step 2: News beside it**

Two headlines with source. `NewsPanel` already has the attribution distinction the README describes — keep it.

- [ ] **Step 3: The sparse notice**

When a payload has no seasons, `MissingData` renders the dashed notice under the status line naming what is empty and what is real. Do not render it when there is nothing missing.

- [ ] **Step 4: Verify**

```bash
cd web && npm run build
npm run lint
```

The rookie must look like `Sparse.dc.html`: notice present, per-game showing the projection alone, "Near you" instead of stat twins.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/profile web/src/components/PlayerProfile.tsx web/src/App.css
git commit -m "feat: a thin profile says which parts are empty and which are real"
```

---

### Task 7: Draft from the popup

**Files:**
- Modify: `web/src/pages/DraftRoom.tsx` (resolve a player id to its `LiveCandidate` and hand the overlay a draft callback)
- Modify: `web/src/components/draft/PlayerOverlay.tsx` (pass it through; rewrite the banner copy)
- Modify: `web/src/components/PlayerProfile.tsx` (the footer button)
- Modify: `web/src/App.css`

**Interfaces:**
- Consumes: `handleDraftClick(c: LiveCandidate)` at `web/src/pages/DraftRoom.tsx:596`, which sets `confirming` and opens `ConfirmPick`.
- Produces: `onDraftPlayer?: (playerId: string) => void` on both `PlayerOverlayProps` and `PlayerProfileProps`.

- [ ] **Step 1: Resolve the id in the room**

`DraftRoom` already resolves ids against its join table for `handleSelectPlayer`. Add a callback that finds the `LiveCandidate` for a player id and calls the existing `handleDraftClick`, so the popup enters the same confirm flow as the board — one pick path, not two. If no candidate matches (already drafted, or not on the ranked list), pass nothing and the footer renders as "Back to the board".

- [ ] **Step 2: Rewrite the banner**

`.player-overlay-clock`'s second sentence goes. It becomes `You are on the clock.` alone. The "Back to the board" button stays: closing is still the right move when the player you are reading is not the one you want.

- [ ] **Step 3: The footer button**

Full-width, accent, `DRAFT <SURNAME>`. It opens `ConfirmPick` — it does not pick on its own. `ConfirmPick` mounts above the overlay on the same z-index tier (see the comment at `DraftRoom.tsx:934`), so check it is not painted behind the popup.

- [ ] **Step 4: Update the spec's own record**

`docs/superpowers/specs/2026-08-21-profile-popup-design.md` says this reverses `fee5551`. Leave the spec alone — it is the record of the decision — but make sure the banner's code comment no longer claims picks cannot be made here, since they now can.

- [ ] **Step 5: Verify**

```bash
cd web && npm run build
npm run lint
```

Then, in a mock draft: open a profile off the board, confirm the footer drafts through `ConfirmPick`, and confirm the banner appears on your turn with the shortened copy and does not sit under the confirm dialog.

- [ ] **Step 6: Commit**

```bash
git add web/src/pages/DraftRoom.tsx web/src/components/draft/PlayerOverlay.tsx web/src/components/PlayerProfile.tsx web/src/App.css
git commit -m "feat: the profile can take the player it is describing"
```

---

## Self-review notes

- Spec coverage: header/status/panels (Task 3), week chart + log (Task 4), the six dense cards (Task 5), comparables/news/sparse (Task 6), draft action and banner (Task 7), the `Chart` extraction and shared builders the spec names as the enabling move (Tasks 1-2). Every spec section maps to a task.
- The spec's "out of scope" items — the README's stale route description and `CohortNext` rendering nothing for Gibbs — are deliberately absent from this plan. The cohort payload feeding the new comparables card in Task 6 may resolve the second one incidentally; if it does not, it stays a separate fix.
