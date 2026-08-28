# Draft Room Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The live draft room orders by ESPN's rank, shows an empirical "still there at your next pick" chance and a points-over-next-pick edge, carries a per-round plan with pros/cons, and lets signed-in users pick favourites the plan targets — with the 400-rollout model gone from the live path.

**Architecture:** Two new pure modules (`scoring/availability.py`, `scoring/plan.py`) replace `survival` + `rank_available` inside `api/live.py`'s recompute and `api/demo.py`'s landing room. A `favorite_player` table beside `entitlement` and `api/account.py` routes hold favourites. The frontend swaps the rail's columns/cards and adds a plan panel and a favourites picker.

**Tech Stack:** Python 3.10, numpy/pandas, DuckDB (corpus read-only), FastAPI, React 19 + Vite + vitest.

**Spec:** `docs/superpowers/specs/2026-08-29-draft-room-redesign-design.md`

## Global Constraints

- `scoring/draft_sim.survival`, `scoring/gain.rank_available`, `search_pick`, `run_sim` are NOT deleted; the sim/report pages keep using them.
- The corpus is opened `read_only=True`; a lock held by the farm means "use the last table", never an error on a poll.
- Overall pick number is the availability axis; no team-count scaling.
- `MIN_DRAFTS = 25`, favourites bonus 1.15, thresholds 0.50 / 0.35 (favourites), `MAX_PICK = 300`, ADP bucket width 8, favourites 5 ≤ n ≤ 25 — copy these exactly.
- The candidate payload keys are exactly: `player_id, position, proj_points, espn_rank, espn_pos_rank, espn_adp, market_rank, lasts_pct, lasts_at_pick, edge_pts, need, favourite, rank`. `plan` is a list of turn objects `{pick_no, round, target, alternates}`; `target`/alternates are `{player_id, lasts_pct, edge_pts, pros, cons}` (alternates carry empty pros/cons).
- Existing tests stay green except the ten pre-existing `tests/test_live_api.py` failures, which Task 4 replaces.
- Commit per task with explicit `git add` paths on branch `draft-room`; normal prose in comments and commit messages, trailer lines as the repo uses.

## Workstreams

- **S — scoring** (Task 1): `scoring/availability.py`, `scoring/plan.py`, tests. No other files.
- **A — account** (Task 2): `api/account.py`, `api/billing.py` (table + helpers only), `api/main.py` (one `register_account_routes(app)` line), tests.
- **F — frontend** (Task 3): `web/src/**` only, against the payload contract above (mock data in vitest until Task 4 lands).
- **L — live wiring** (Task 4, after Task 1): `api/live.py`, `api/demo.py`, `tests/test_live_api.py`, `tests/test_demo_live.py`.

---

### Task 1: `scoring/availability.py` + `scoring/plan.py`

**Files:** create `scoring/availability.py`, `scoring/plan.py`, `tests/test_availability.py`, `tests/test_plan.py`.

**Interfaces (produced):**

```python
# scoring/availability.py
MAX_PICK = 300; MIN_DRAFTS = 25; ADP_BUCKET = 8
class AvailabilityTable:
    player_ids: np.ndarray            # object, len P
    pooled: np.ndarray                # int, len P
    taken_by: np.ndarray              # int, shape (P, MAX_PICK + 1); taken_by[i, p] = drafts taken at pick <= p
    adp_curve: dict[int, tuple[float, float]]   # bucket -> (mu, sigma) of actual pick vs ESPN ADP
    corpus_mtime: float
def load_table(corpus_path: str = dl.CORPUS_PATH) -> AvailabilityTable       # read-only; empty table when locked/missing
def cached_table(corpus_path: str = dl.CORPUS_PATH) -> AvailabilityTable     # refreshes when the file's mtime moves
def availability_at(table, player_ids, k: int, n: int, espn_adp, market_rank) -> np.ndarray
    # P(still there at pick n | still there at pick k), float in [0, 1], per player id; fallbacks per spec §2

# scoring/plan.py
NEED_BONUS = 1.15; THRESHOLD = 0.50; FAVOURITE_THRESHOLD = 0.35
def expected_best_excluding(proj, avail, positions, pos, exclude_idx) -> float
def edge_at(proj, positions, avail_next) -> np.ndarray         # edge per candidate at the next turn
def build_plan(*, proj, positions, player_ids, espn_rank, espn_adp, market_rank, byes, health,
               roster_counts, settings, turns: list[int], picks_made: int, favourites: set[str],
               table: AvailabilityTable, names: dict[str, str]) -> list[dict]
def target_now(...same inputs...) -> list[dict]                 # top three for the current pick, P = 1
def reasons_for(...) -> tuple[list[str], list[str]]             # (pros, cons) per spec §4 order
```

- [ ] Tests first (`tests/test_availability.py`): build a corpus DuckDB in `tmp_path` with `draft_log`, `draft_log_pool`, `draft_log_pick` (read `pipeline/draft_log.py` for the schema) holding 40 drafts where player A is taken at picks 1-3 always, player B at picks 10-30 uniformly, player C never taken, player D pooled in only 5 drafts. Assert: A at (k=5, n=11) falls back (denominator 0) to the ADP curve; B at (k=8, n=16) ≈ share of his picks > 16 among > 8; C = 1.0; D uses the fallback; `cached_table` reloads when mtime changes; a locked corpus yields the previous table.
- [ ] Implement `availability.py`. `taken_by` built with one SQL (`SELECT player_id, pick_no FROM draft_log_pick`) and numpy cumulative counts; `pooled` from `draft_log_pool`.
- [ ] Tests (`tests/test_plan.py`): the Gibbs case (P(next)=0.02 → never target/alternate at pick 6); edge excludes self; favourite bonus and lower threshold; a planned target is not re-targeted; reasons order and the ADP-vs-ESPN con; `target_now` uses P = 1.
- [ ] Implement `plan.py` reusing `scoring.gain.need_kind` / `NEED_WEIGHTS`.
- [ ] `.venv/bin/pytest -q tests/test_availability.py tests/test_plan.py`; commit.

### Task 2: favourites store + API

**Files:** modify `api/billing.py` (table `favorite_player`, `favorites(account_ids) -> list[str]`, `set_favorites(account_id, ids)`), create `api/account.py` (`register_account_routes(app, store=None)`), modify `api/main.py` (register), create `tests/test_account_api.py`.

- [ ] Table DDL in both backends (`_db()` creates it beside `entitlement`; Postgres TIMESTAMPTZ).
- [ ] Routes per spec §5: 401 without a custody session (use `billing._account_ids(request, store)`); `PUT` validates 5-25 and that every id is on the board (`cached_build_board(cur)` player ids); replaces atomically (DELETE + INSERT in one transaction).
- [ ] Tests: round trip on DuckDB; bounds 4 and 26 → 422; unknown id → 422; no session → 401; reads across two account ids.
- [ ] Commit.

### Task 3: frontend

**Files:** `web/src/api.ts` (types), `web/src/components/draft/AvailableList.tsx`, new `TargetCards.tsx` (replaces `TopThree.tsx` usage), new `PlanPanel.tsx`, new `components/FavoritesPicker.tsx`, `components/Dashboard.tsx`, `pages/DraftRoom.tsx`, `App.css`, vitest files.

- [ ] Types: `LiveCandidate` per the contract; `LivePlan`; `fetchFavorites()/saveFavorites()`.
- [ ] AvailableList: columns per spec §1 (ESPN, Pos, ADP, Consensus, Lasts %, Edge, Proj/G + meters), default sort `espn_rank`, ★ for favourites, tooltips rewritten. Remove every reference to `gain_now/vor_points/fills/survive_pct`.
- [ ] TargetCards: three cards from `plan[0]` (on the clock) else the next turn; pros as green bullets, cons as amber.
- [ ] PlanPanel under the roster: rows per turn.
- [ ] FavoritesPicker on the Dashboard when `favorites.length === 0` (search over `/api/players`, position filter, headshots, ordered selection, counter, Save) and the "Your guys" card with Edit afterwards.
- [ ] vitest: picker bounds; PlanPanel renders reasons; AvailableList default order is ESPN rank. `npm test && npm run build`. Commit.

### Task 4: live wiring (after Task 1)

**Files:** `api/live.py`, `api/demo.py`, `tests/test_live_api.py`, `tests/test_demo_live.py`.

- [ ] `_recompute` per spec §6; `DraftSession` gains `favourites: frozenset[str]` resolved at connect from `billing.favorites(account_ids)` (empty when no session); `/api/live/state` returns `plan`; `horizon_pick`, `horizon_is_end_of_draft`, `candidates_as_of_pick` semantics per spec.
- [ ] Remove `RECOMPUTE_SLOTS`, `rollouts_for_load`, `rollouts_for`, `SURVIVAL_ROLLOUTS`, `ROLLOUTS_REDUCED`, `LIVE_RECOMPUTE_SLOTS` handling (README row too); keep the recompute worker and its coalescing.
- [ ] `api/demo.py` landing room uses `availability_at` + `target_now`.
- [ ] Rewrite the recompute/connect tests that monkeypatch `live.survival`/`rank_available` against `live.availability_at`/`live.build_plan`; the ten pre-existing failures must be gone, not skipped.
- [ ] `.venv/bin/pytest -q tests/test_live_api.py tests/test_live_sessions.py tests/test_demo_live.py tests/test_api.py`; a 25-room `make load-test` to confirm recompute cost dropped (report p95 state). Commit.

### Task 5: integration + docs

- [ ] Full suite green; `npm test && npm run build`.
- [ ] README: room columns, favourites, the plan; remove the rollouts/`LIVE_RECOMPUTE_SLOTS` rows.
- [ ] Commit; merge `draft-room` to main.
