# ESPN Default Sort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Board opens sorted by ESPN's PPR expert rank (nulls last); our ranking stays available as a sortable column.

**Architecture:** Extend the existing ESPN join in `scoring/market.py` to carry `espn_ppr_rank` onto the board; add the column to the contract/types; flip the table's initial sorting state with a nulls-last sortingFn.

**Tech Stack:** existing (pandas 1.5.3, FastAPI, React + TanStack).

**Spec:** `docs/superpowers/specs/2026-08-02-espn-default-sort-design.md`

## Global Constraints

- pandas 1.5.3 — no `.replace(0, pd.NA)`.
- `espn_ppr_rank` is nullable float end-to-end; JSON-safe via existing scrub paths.
- Default sort: `espn_ppr_rank` ascending, nulls last in BOTH directions (explicit TanStack `sortingFn`, don't rely on `basic`).
- Tier-band gating unchanged (our-rank sort + single-position tab).
- Use `.venv/bin/pytest` (suite currently 91 passing); commit conventionally.

---

### Task 1: espn_ppr_rank end-to-end + default sort

**Files:**
- Modify: `scoring/market.py`, `tests/test_market.py`, `tests/test_board.py`, `tests/test_api.py`, `web/src/api.ts`, `web/src/components/PlayerTable.tsx`, `README.md` (keyboard/columns blurb if it lists columns)

**Interfaces:**
- Produces: board/API/profile-header field `espn_ppr_rank: float | None`; TS `Player.espn_ppr_rank: number | null`; board default sorting `[{ id: 'espn_ppr_rank', desc: false }]`.

- [ ] **Step 1: Failing tests.** In `tests/test_market.py` extend the existing fixtures: assert `add_market` output has `espn_ppr_rank` == the fixture's value for the crosswalk-joined player AND the name-fallback player, and NaN/None when espn table empty. In `tests/test_board.py` add `espn_ppr_rank` to the column-contract assertion (seed the espn_adp table with one row incl. espn_ppr_rank in the existing `_seed`-style helper if needed). In `tests/test_api.py` assert the players payload row carries the key.
- [ ] **Step 2: Run to verify failures.**
- [ ] **Step 3: Implement backend.** In `scoring/market.py::_espn_ranks`, also map `espn_ppr_rank` through BOTH paths (crosswalk `by_id`-style mapping and the name-fallback reindex) — return it alongside the adp-rank series (adjust the helper's return signature and `add_market` to attach `out["espn_ppr_rank"]`; do NOT include it in `_RANK_COLS` — it is display data, not a consensus input). Ensure it survives the final `.drop(columns=...)`, and add it to `scoring/board.py::_BOARD_COLUMNS`.
- [ ] **Step 4: Full suite green.** Real-data smoke: build the board, confirm ~300–500 players have espn_ppr_rank, top ESPN ranks are plausible stars.
- [ ] **Step 5: Frontend.** `api.ts`: add the field to `Player`. `PlayerTable.tsx`: new "ESPN" column right before "Rank" (0 decimals, "—" null); initial `useState<SortingState>([{ id: 'espn_ppr_rank', desc: false }])`; custom `sortingFn` on this column: nulls last regardless of direction (`sortUndefined` alone is insufficient for null — use a custom fn comparing with null checks). Verify tier bands still appear only under our-rank sort on a position tab.
- [ ] **Step 6: Verify.** `npm run build` clean; browser check: default order matches the ESPN column ascending with "—" rows at the bottom; clicking "Rank" restores our order; keyboard nav follows the visible order.
- [ ] **Step 7: Commit.** `git add -A && git commit -m "feat: espn ppr rank column as default board order"`
