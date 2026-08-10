# Predicted Draft Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the upcoming draft as a grid — rounds down the side, managers across the top, the model's predicted pick in every cell, with the second and third choices on hover and a condensed player profile on click.

**Architecture:** A new `predict_board` runs full rollouts through the existing engine and counts what player each overall pick number produced, then deduplicates the per-cell winners with a one-to-one assignment so no player occupies two cells. `run_sim` persists the result to a `sim_board` table, a new endpoint serves it, and a new `/draft-board` route renders it.

**Tech Stack:** Python 3, numpy, pandas, scipy (`linear_sum_assignment`), duckdb, FastAPI, React + TypeScript + Vite, react-router-dom.

**Spec:** `docs/superpowers/specs/2026-08-09-predicted-draft-board-design.md`

## Global Constraints

- Every Python test runs offline. No network, no browser, no reading `data/nfl.duckdb`. Tests seed DuckDB under pytest's `tmp_path` and use literal fixtures, matching `tests/test_draft_sim.py` and `tests/test_api.py`.
- Existing behavior must not change. The full existing suite (260 tests) must stay green after every task, and existing tests must pass unmodified unless a task explicitly sanctions a change.
- Determinism: every sampling path takes an explicit `numpy.random.Generator`. Never the global numpy RNG.
- Position vocabulary is exactly `QB, RB, WR, TE, K, DST`.
- No new CSS tokens. `web/src/index.css` defines the full palette: `--bg-0`, `--bg-1`, `--bg-2`, `--border`, `--text-1`, `--text-2`, `--text-3`, `--accent`, `--accent-bg`, `--accent-border`, `--ok`, `--fail`, `--pos-qb`, `--pos-rb`, `--pos-wr`, `--pos-te`, `--pos-k`, `--pos-dst`, and a `--space-*` scale.
- The app is dark-theme only, on purpose. Do not add a light palette.
- Run tests with `.venv/bin/pytest`. Frontend gate is `cd web && npm run build`, which must pass with no TypeScript errors.
- Commit after every task.

## File Structure

| File | Responsibility |
|---|---|
| `scoring/draft_sim.py` (modify) | `_run_draft` gains pick recording; new `predict_board`; `run_sim` persists `sim_board` |
| `api/main.py` (modify) | `GET /api/sim/board` |
| `web/src/api.ts` (modify) | `fetchSimBoard()` and its types |
| `web/src/components/DraftGrid.tsx` (new) | The grid: rounds, snake arrows, manager columns, cells, hover popover |
| `web/src/components/DraftBoardPage.tsx` (new) | Route shell: fetches, header, renders `DraftGrid` and `PlayerCard` |
| `web/src/components/PlayerCard.tsx` (new) | Condensed profile modal |
| `web/src/App.tsx` (modify) | `/draft-board` route |
| `web/src/components/TopBar.tsx` (modify) | Nav link between board and grid |
| `web/src/App.css` (modify) | Grid and card styles |
| `README.md` (modify) | Document the view |

---

### Task 1: Record the pick sequence

`_run_draft` returns per-slot rosters but never says which player went at which overall pick. The grid is exactly that mapping.

**Files:**
- Modify: `scoring/draft_sim.py` (`_run_draft`)
- Test: `tests/test_draft_sim.py`

**Interfaces:**
- Consumes: `_run_draft(pool, settings, slot_managers, my_slot, taken, betas, rng, forced=None, taken_order=None) -> dict`
- Produces: the same function with one added keyword-only parameter, `record=None`. When a list is passed, `_run_draft` appends `(overall_pick, pool_index)` for every pick it simulates, in pick order. The return value is unchanged, so `rollout` and every existing caller are unaffected.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft_sim.py`:

```python
def test_run_draft_records_every_pick_in_order_when_asked():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    record = []
    rosters = _run_draft(pool, S, slots, 4, taken, _flat_betas(slots.values()),
                         rng=np.random.default_rng(3), record=record)
    picks = [p for p, _ in record]
    assert picks == list(range(1, len(picks) + 1))
    # Every recorded index is a real pool index, and no player goes twice.
    indices = [i for _, i in record]
    assert len(set(indices)) == len(indices)
    assert all(0 <= i < len(pool.player_id) for i in indices)
    # The record and the rosters describe the same draft.
    from_rosters = sorted(i for st in rosters.values() for i in st["indices"])
    assert sorted(indices) == from_rosters


def test_run_draft_without_record_is_unchanged():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    betas = _flat_betas(slots.values())
    a = _run_draft(pool, S, slots, 4, taken, betas, rng=np.random.default_rng(9))
    b = _run_draft(pool, S, slots, 4, taken, betas, rng=np.random.default_rng(9),
                   record=[])
    assert {s: st["indices"] for s, st in a.items()} == \
           {s: st["indices"] for s, st in b.items()}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_sim.py::test_run_draft_records_every_pick_in_order_when_asked -v`
Expected: FAIL with `TypeError: _run_draft() got an unexpected keyword argument 'record'`

- [ ] **Step 3: Write minimal implementation**

Change `_run_draft`'s signature to add the parameter:

```python
def _run_draft(pool, settings, slot_managers, my_slot, taken, betas, rng,
               forced=None, taken_order=None, *, record=None) -> dict:
```

Add to its docstring, after the `taken_order` paragraph:

```
    `record`, when given a list, receives `(overall_pick, pool index)` for
    every pick this call simulates, in pick order. The per-slot return value
    already says WHO holds a player but not WHEN he went, and the predicted
    draft board is exactly that mapping. Opt-in because the common caller
    (`rollout`, run thousands of times inside a search) has no use for it.
```

Then, immediately after the existing `rosters[slot]["indices"].append(choice)` line inside the pick loop, add:

```python
        if record is not None:
            record.append((overall_pick, int(choice)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_draft_sim.py -v`
Expected: all passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: 262 passed (260 + 2 new)

- [ ] **Step 6: Commit**

```bash
git add scoring/draft_sim.py tests/test_draft_sim.py
git commit -m "feat: _run_draft can record its pick sequence"
```

---

### Task 2: predict_board

**Files:**
- Modify: `scoring/draft_sim.py`
- Test: `tests/test_draft_sim.py`

**Interfaces:**
- Consumes: `_run_draft(..., record=[])` from Task 1; `snake_slots(teams, rounds)`; `_seed_rosters(pool, settings, taken_order)`; `SimPool` with fields `player_id, norm, position, adp_rank, points, availability, vor`
- Produces:
  - `scoring.draft_sim.ASSIGNMENT_MISS_COST = 50.0`
  - `predict_board(pool, settings, slot_managers, my_slot, taken, betas, n_rollouts=DEFAULT_ROLLOUTS, seed=0, alternates=2, taken_order=None) -> pd.DataFrame` with columns, in this order: `overall_pick, round, round_pick, slot, alt_rank, player_id, prob, certain`
    - one row per (overall_pick, alt_rank); `alt_rank` 0 is the primary, 1 and 2 the alternates
    - `certain` is True only for picks already made, whose row comes from `taken_order` with `prob` 1.0 and no alternates
    - picks with no recorded outcome (the pool ran dry) produce no rows

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft_sim.py`:

```python
def test_predict_board_covers_every_pick_and_sums_sensibly():
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=20, seed=1)
    assert list(board.columns) == ["overall_pick", "round", "round_pick",
                                   "slot", "alt_rank", "player_id", "prob",
                                   "certain"]
    primaries = board[board["alt_rank"] == 0]
    # 60-player pool, 8 teams x 15 rounds -- the pool runs dry at pick 60.
    assert primaries["overall_pick"].tolist() == list(range(1, 61))
    assert (primaries["prob"] > 0).all()
    assert (primaries["prob"] <= 1).all()
    assert not primaries["certain"].any()


def test_predict_board_round_and_slot_follow_the_snake():
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=10, seed=2)
    p = board[board["alt_rank"] == 0].set_index("overall_pick")
    assert (p.loc[1, "round"], p.loc[1, "slot"]) == (1, 1)
    assert (p.loc[8, "round"], p.loc[8, "slot"]) == (1, 8)
    assert (p.loc[9, "round"], p.loc[9, "slot"]) == (2, 8)
    assert (p.loc[16, "round"], p.loc[16, "slot"]) == (2, 1)
    assert p.loc[9, "round_pick"] == 1


def test_predict_board_never_puts_one_player_in_two_cells():
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=25, seed=4)
    primaries = board[board["alt_rank"] == 0]["player_id"]
    assert len(set(primaries)) == len(primaries)


def test_predict_board_keeps_a_deduped_player_as_an_alternate():
    """The dedupe must not hide information.

    With ADP-disciplined opponents the same early player wins several
    adjacent cells on raw frequency. Assignment gives him exactly one, and
    the cells he lost must still list him among their alternates -- that is
    the whole reason hover exists.
    """
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=40, seed=5, alternates=2)
    top = board[(board["overall_pick"] == 1) & (board["alt_rank"] == 0)]
    top_id = top["player_id"].iloc[0]
    nearby = board[(board["overall_pick"].between(2, 5))
                   & (board["alt_rank"] > 0)]["player_id"].tolist()
    assert top_id in nearby


def test_predict_board_alternates_are_ranked_and_never_repeat_the_primary():
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=30, seed=6, alternates=2)
    for pick, grp in board.groupby("overall_pick"):
        grp = grp.sort_values("alt_rank")
        assert grp["alt_rank"].tolist() == list(range(len(grp)))
        assert len(set(grp["player_id"])) == len(grp)
        alts = grp[grp["alt_rank"] > 0]["prob"].tolist()
        assert alts == sorted(alts, reverse=True)


def test_predict_board_is_deterministic_under_a_fixed_seed():
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    args = (pool, S, slots, 4, taken, _adp_betas(slots.values()))
    a = predict_board(*args, n_rollouts=12, seed=7)
    b = predict_board(*args, n_rollouts=12, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_predict_board_reports_already_made_picks_as_certain():
    from scoring.draft_sim import predict_board
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    taken[:3] = True
    board = predict_board(pool, S, slots, 4, taken, _adp_betas(slots.values()),
                          n_rollouts=10, seed=8, taken_order=[0, 1, 2])
    made = board[board["certain"]]
    assert made["overall_pick"].tolist() == [1, 2, 3]
    assert made["player_id"].tolist() == [pool.player_id[i] for i in (0, 1, 2)]
    assert (made["prob"] == 1.0).all()
    assert (made["alt_rank"] == 0).all()
    # A certain pick has no alternates, and the predictions start after it.
    assert board[board["overall_pick"] <= 3]["alt_rank"].max() == 0
    assert not board[board["overall_pick"] == 4]["certain"].any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_sim.py -k predict_board -v`
Expected: FAIL with `ImportError: cannot import name 'predict_board'`

- [ ] **Step 3: Write minimal implementation**

Add the scipy import near the top of `scoring/draft_sim.py`, beside the existing imports:

```python
from scipy.optimize import linear_sum_assignment
```

Append to `scoring/draft_sim.py`:

```python
# Cost charged for assigning a player to a cell he never once won. Any
# finite value works as long as it dominates every real -log(prob): with
# probabilities floored at 1/n_rollouts, a real cost never exceeds
# log(n_rollouts), which stays under 50 for any rollout count anyone will
# run. It must be finite -- linear_sum_assignment rejects an infeasible
# matrix outright, and a board where one cell has no candidate should still
# render the other 119.
ASSIGNMENT_MISS_COST = 50.0


def predict_board(pool, settings, slot_managers, my_slot, taken, betas,
                  n_rollouts: int = DEFAULT_ROLLOUTS, seed: int = 0,
                  alternates: int = 2, taken_order=None) -> pd.DataFrame:
    """Who the model thinks goes at every pick of the draft.

    Counts, across `n_rollouts` full drafts, which player each overall pick
    number produced. Two things come out of those counts:

    The raw per-cell frequencies, which are the honest distribution and
    become the alternates hover shows.

    A deduped primary. Taking each cell's most frequent player independently
    puts a consensus first-rounder in four adjacent cells, which reads as a
    bug rather than as "he could go at any of these". So the frequencies are
    treated as an assignment problem -- picks on one side, players on the
    other, cost -log(prob) -- and solved for the highest-likelihood board in
    which nobody appears twice. The name in the cell is therefore the most
    coherent single draft the model can tell, not 120 independent answers,
    which is why the alternates matter and are kept.

    Picks already made are reported from `taken_order` as certain, with no
    alternates: they are facts, not predictions.
    """
    slots = snake_slots(settings.teams, settings.rounds)
    teams = max(settings.teams, 1)
    already = len(taken_order) if taken_order is not None else int(taken.sum())

    counts = {}
    for i in range(n_rollouts):
        record = []
        _run_draft(pool, settings, slot_managers, my_slot, taken, betas,
                   rng=np.random.default_rng([seed, i]),
                   taken_order=taken_order, record=record)
        for overall_pick, idx in record:
            counts.setdefault(overall_pick, {})
            counts[overall_pick][idx] = counts[overall_pick].get(idx, 0) + 1

    predicted = sorted(counts)
    rows = []

    # -- picks already made: facts, in pick order --
    for offset in range(already):
        overall_pick = offset + 1
        if taken_order is None or offset >= len(taken_order):
            continue
        idx = taken_order[offset]
        if idx is None:
            continue
        rows.append(_board_row(pool, slots, teams, overall_pick, 0, idx, 1.0, True))

    if predicted:
        # -- assignment over the predicted picks --
        players = sorted({idx for c in counts.values() for idx in c})
        col_of = {idx: j for j, idx in enumerate(players)}
        cost = np.full((len(predicted), len(players)), ASSIGNMENT_MISS_COST)
        for r, overall_pick in enumerate(predicted):
            for idx, n in counts[overall_pick].items():
                cost[r, col_of[idx]] = -np.log(n / n_rollouts)
        rows_idx, cols_idx = linear_sum_assignment(cost)
        primary = {predicted[r]: players[c] for r, c in zip(rows_idx, cols_idx)}

        for overall_pick in predicted:
            cell = counts[overall_pick]
            chosen = primary.get(overall_pick)
            # An assignment can hand a cell a player it never produced, when
            # that frees a better fit elsewhere. Such a cell has no real
            # primary, so fall back to its own most frequent player and let
            # the duplicate stand -- a visible repeat beats inventing a pick
            # the model never made.
            if chosen is None or chosen not in cell:
                chosen = max(cell, key=lambda k: (cell[k], -k))
            rows.append(_board_row(pool, slots, teams, overall_pick, 0, chosen,
                                   cell[chosen] / n_rollouts, False))
            ranked = sorted(((n, -idx, idx) for idx, n in cell.items()
                             if idx != chosen), reverse=True)
            for alt_rank, (n, _, idx) in enumerate(ranked[:alternates], start=1):
                rows.append(_board_row(pool, slots, teams, overall_pick,
                                       alt_rank, idx, n / n_rollouts, False))

    board = pd.DataFrame(rows, columns=["overall_pick", "round", "round_pick",
                                        "slot", "alt_rank", "player_id",
                                        "prob", "certain"])
    return board.sort_values(["overall_pick", "alt_rank"]).reset_index(drop=True)


def _board_row(pool, slots, teams, overall_pick, alt_rank, idx, prob, certain):
    offset = overall_pick - 1
    return {"overall_pick": overall_pick,
            "round": offset // teams + 1,
            "round_pick": offset % teams + 1,
            "slot": slots[offset] if offset < len(slots) else 0,
            "alt_rank": alt_rank,
            "player_id": pool.player_id[idx],
            "prob": float(prob),
            "certain": bool(certain)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_draft_sim.py -k predict_board -v`
Expected: 7 passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: 269 passed

- [ ] **Step 6: Commit**

```bash
git add scoring/draft_sim.py tests/test_draft_sim.py
git commit -m "feat: predict who goes at every pick, deduped by assignment"
```

---

### Task 3: Persist and serve the board

**Files:**
- Modify: `scoring/draft_sim.py` (`run_sim`), `api/main.py`
- Test: `tests/test_draft_sim.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `predict_board` from Task 2; `run_sim(conn, my_slot, slot_managers, n_rollouts=DEFAULT_ROLLOUTS, seed=0) -> str`
- Produces:
  - `run_sim` additionally writes the `sim_board` table: `run_id, overall_pick, round, round_pick, slot, alt_rank, player_id, prob, certain`
  - `GET /api/sim/board` returning
    `{"run": {"run_id", "my_slot", "created_at"} | null, "teams": int, "rounds": int, "order": [{"slot", "manager"}], "cells": [{"overall_pick", "round", "round_pick", "slot", "alt_rank", "player_id", "name", "position", "team", "prob", "certain"}]}`
    with `cells: []` and `run: null` when no sim has run

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft_sim.py`:

```python
def test_run_sim_writes_sim_board(tmp_path, monkeypatch):
    import pandas as pd
    from pipeline.db import get_conn, write_table, read_table
    from scoring import draft_model, league
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "league", pd.DataFrame(
        [{"season": 2026, "settings_json": league.to_json(league.default_settings())}]))
    _seed_board_tables(conn)
    monkeypatch.setattr(draft_model, "fit_all", lambda *a, **k: {
        "__pooled__": np.zeros(len(FEATURE_NAMES)),
        **{f"m{i}": np.zeros(len(FEATURE_NAMES)) for i in range(1, 9)}})
    run_id = run_sim(conn, my_slot=1,
                     slot_managers={i: f"m{i}" for i in range(1, 9)},
                     n_rollouts=3, seed=0)
    board = read_table(conn, "sim_board")
    assert list(board.columns) == ["run_id", "overall_pick", "round",
                                   "round_pick", "slot", "alt_rank",
                                   "player_id", "prob", "certain"]
    assert not board.empty
    assert set(board["run_id"]) == {run_id}
    primaries = board[board["alt_rank"] == 0]["player_id"]
    assert len(set(primaries)) == len(primaries)
```

Add the helper above it, next to the other `tests/test_draft_sim.py` fixtures:

```python
def _seed_board_tables(conn):
    """Minimal weekly/schedules/adp/espn tables so build_board returns rows."""
    import pandas as pd
    from pipeline.db import write_table
    write_table(conn, "weekly", pd.DataFrame([
        {"player_id": f"p{n}", "player_display_name": f"Player {n}",
         "position": ["QB", "RB", "WR", "TE"][n % 4], "recent_team": "DET",
         "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 5, "receiving_yards": 60, "targets": 7, "carries": 3}
        for n in range(1, 30) for w in range(1, 18)]))
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 48.0, "spread_line": 2.0}]))
    write_table(conn, "adp", pd.DataFrame(
        columns=["adp_name", "position", "team", "adp"]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    write_table(conn, "espn_adp", pd.DataFrame(
        columns=["espn_id", "espn_name", "position", "espn_adp",
                 "espn_ppr_rank", "espn_proj"]))
    write_table(conn, "fp_ecr", pd.DataFrame(
        columns=["fp_name", "team", "position", "rank_ecr", "rank_ave",
                 "rank_std", "fp_tier"]))
    write_table(conn, "sleeper_ids", pd.DataFrame(
        columns=["gsis_id", "espn_id", "sleeper_name", "position", "team"]))
```

Append to `tests/test_api.py`:

```python
def test_sim_board_is_empty_without_a_run(tmp_path):
    body = _client(tmp_path).get("/api/sim/board").json()
    assert body["run"] is None
    assert body["cells"] == []
    assert body["teams"] == 8
    assert body["rounds"] == 15


def test_sim_board_serves_cells_with_player_details(tmp_path):
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "sim_results", pd.DataFrame(
        [{"run_id": "r1", "player_id": "p1", "ev": 1.0, "se": 0.1, "rank": 1,
          "applied_pct": 1.0, "my_slot": 4, "pick_no": 0,
          "created_at": "2026-08-09 12:00:00"}]))
    write_table(conn, "sim_board", pd.DataFrame([
        {"run_id": "r1", "overall_pick": 1, "round": 1, "round_pick": 1,
         "slot": 1, "alt_rank": 0, "player_id": "p1", "prob": 0.42,
         "certain": False},
        {"run_id": "r1", "overall_pick": 1, "round": 1, "round_pick": 1,
         "slot": 1, "alt_rank": 1, "player_id": "nope", "prob": 0.2,
         "certain": False},
    ]))
    conn.close()
    body = TestClient(create_app(path)).get("/api/sim/board").json()
    assert body["run"]["my_slot"] == 4
    primary = [c for c in body["cells"] if c["alt_rank"] == 0][0]
    assert primary["name"] == "A Star"
    assert primary["position"] == "WR"
    assert primary["prob"] == 0.42
    # A cell naming a player the board no longer carries still renders,
    # with the id standing in for the name rather than vanishing.
    alt = [c for c in body["cells"] if c["alt_rank"] == 1][0]
    assert alt["name"] == "nope"
    assert alt["position"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api.py -k sim_board tests/test_draft_sim.py -k sim_board -v`
Expected: FAIL — 404 on `/api/sim/board`, and `sim_board` table missing.

- [ ] **Step 3: Write minimal implementation**

In `scoring/draft_sim.py`'s `run_sim`, after the existing `avail = survival(...)` call and before the `run_id` is built, add:

```python
    cells = predict_board(pool, settings, slot_managers, my_slot, taken, betas,
                          n_rollouts=n_rollouts, seed=seed,
                          taken_order=taken_order)
```

Then, alongside the existing `write_table(conn, "sim_survival", avail)`, add:

```python
    cells.insert(0, "run_id", run_id)
    write_table(conn, "sim_board", cells)
```

(Insert `run_id` after it is computed, next to the existing `results.insert(0, "run_id", run_id)` line, so all three frames are stamped together.)

In `api/main.py`, add this handler. Register it beside the existing `/api/sim/latest` handler, which is already declared before `/api/sim/{run_id}` so the path parameter cannot swallow it:

```python
    @app.get("/api/sim/board")
    def sim_board():
        cur = conn.cursor()
        try:
            settings = league.load(cur)
            order = draft_order()["order"]
            cells = read_table(cur, "sim_board")
            latest = read_table(cur, "sim_results")
            run = None
            if not latest.empty:
                head = latest.iloc[0]
                run = {"run_id": head["run_id"],
                       "my_slot": int(head["my_slot"]),
                       "created_at": str(head["created_at"])}
            if cells.empty:
                return {"run": run, "teams": settings.teams,
                        "rounds": settings.rounds, "order": order, "cells": []}
            board = build_board(cur, settings=settings)
            names = board.set_index("player_id")[["name", "position", "team"]]
            out = []
            for _, c in cells.iterrows():
                pid = c["player_id"]
                # A cell can name a player the board no longer carries (a
                # refresh between runs). Render the id rather than dropping
                # the cell, so the grid never silently loses a pick.
                if pid in names.index:
                    row = names.loc[pid]
                    name, position, team = row["name"], row["position"], row["team"]
                else:
                    name, position, team = pid, None, None
                out.append({"overall_pick": int(c["overall_pick"]),
                            "round": int(c["round"]),
                            "round_pick": int(c["round_pick"]),
                            "slot": int(c["slot"]),
                            "alt_rank": int(c["alt_rank"]),
                            "player_id": pid, "name": name,
                            "position": position, "team": team,
                            "prob": float(c["prob"]),
                            "certain": bool(c["certain"])})
            return {"run": run, "teams": settings.teams,
                    "rounds": settings.rounds, "order": order, "cells": out}
        finally:
            cur.close()
```

Add `from scoring.board import build_board` to `api/main.py`'s imports if it is not already there.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_api.py tests/test_draft_sim.py -q`
Expected: all passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: 272 passed

- [ ] **Step 6: Commit**

```bash
git add scoring/draft_sim.py api/main.py tests/
git commit -m "feat: persist the predicted board and serve it"
```

---

### Task 4: The condensed profile card

**Files:**
- Create: `web/src/components/PlayerCard.tsx`
- Modify: `web/src/App.css`

**Interfaces:**
- Consumes: `fetchProfile(playerId)` from `web/src/api.ts` (the same call `PlayerProfile` uses), `SeasonRangeChart`
- Produces: `PlayerCard` props `{ playerId: string; onClose: () => void }`

- [ ] **Step 1: Read what the profile payload carries**

Read `web/src/components/PlayerProfile.tsx` and the `PlayerProfileData` type in `web/src/api.ts` before writing anything. Use the field names they already use — do not invent names for the header chips, the factor values, or the outlook fields.

- [ ] **Step 2: Write the card**

Create `web/src/components/PlayerCard.tsx`. It renders, in this order:

1. A header: name, position, team, bye week, and a close button.
2. A decision row: VOR rank, tier, market rank, edge, projected points.
3. Sim context when the profile's player has them: `Avail%` and `ΔEV`. These come from the board row, so accept them as optional props `avail_pct?: number | null` and `ev?: number | null` supplied by the caller, rather than refetching the whole player list.
4. The five factor bars — production, role, environment, schedule, durability — each a labelled 0-100 bar.
5. `<SeasonRangeChart />` with the same props `PlayerProfile` passes it.
6. An outlook line: depth slot, implied team points, strength of schedule.
7. A `Full profile →` link to `/players/<slug>` using the existing `playerSlug` helper.

It must:
- Render inside a fixed-position backdrop that closes on click, with the card itself stopping propagation.
- Close on `Escape` via a `useEffect` keydown listener that is removed on unmount.
- Show a loading state while the profile fetch is in flight and an error state if it rejects.
- Guard against a stale response overwriting a newer one, the same way `PlayerProfile` does with its `playerIdRef` — read that comment before writing this, and follow the same pattern.

- [ ] **Step 3: Add the styles**

Append to `web/src/App.css`, again using only existing tokens:

```css
.card-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 20;
}
.player-card {
  width: min(440px, 92vw);
  max-height: 86vh;
  overflow-y: auto;
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: var(--space-4);
}
.player-card header { display: flex; align-items: baseline; gap: 8px; }
.player-card header h2 { margin: 0; font-size: 1.05rem; }
.card-sub { color: var(--text-3); font-size: 0.8rem; }
.card-close { margin-left: auto; background: none; border: 0; color: var(--text-3); cursor: pointer; font: inherit; }
.card-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; font-size: 0.8rem; }
.card-row div { color: var(--text-2); }
.card-row strong { color: var(--text-1); }
.card-sim { background: var(--accent-bg); padding: 6px 8px; border-radius: 4px; }
.card-factors { list-style: none; margin: 10px 0; padding: 0; }
.card-factors li { display: flex; align-items: center; gap: 8px; font-size: 0.75rem; margin-bottom: 3px; }
.card-factor-name { width: 82px; color: var(--text-3); }
.card-factor-track { flex: 1; height: 6px; background: var(--bg-2); border-radius: 3px; }
.card-factor-fill { display: block; height: 100%; border-radius: 3px; background: var(--accent); }
.card-outlook { color: var(--text-2); font-size: 0.78rem; }
.card-full { display: inline-block; margin-top: 10px; color: var(--accent); font-size: 0.8rem; }
```

- [ ] **Step 4: Typecheck and build**

Run: `cd web && npm run build`
Expected: succeeds with no TypeScript errors.

- [ ] **Step 5: Visual pass**

Run `make up`, open `/draft-board`, and confirm: the grid renders with position colors, hovering a cell shows the alternates, clicking a name opens the card, Escape and backdrop clicks both close it, and your own column is visibly marked. Report what you saw.

- [ ] **Step 6: Commit**

```bash
git add web/src
git commit -m "feat: condensed player card on the draft board"
```

---

### Task 5: The grid

**Files:**
- Create: `web/src/components/DraftGrid.tsx`, `web/src/components/DraftBoardPage.tsx`
- Modify: `web/src/api.ts`, `web/src/App.tsx`, `web/src/components/TopBar.tsx`, `web/src/App.css`

**Interfaces:**
- Consumes: `GET /api/sim/board` from Task 3; `PlayerCard` from Task 4
- Produces:
  - `web/src/api.ts`: `SimBoardCell`, `SimBoard` types and `fetchSimBoard(): Promise<SimBoard>`
  - `DraftGrid` props `{ board: SimBoard; onSelectPlayer: (playerId: string) => void }`
  - `DraftBoardPage` as the `/draft-board` route element

- [ ] **Step 1: Add the API client**

Append to `web/src/api.ts`:

```ts
export interface SimBoardCell {
  overall_pick: number
  round: number
  round_pick: number
  slot: number
  alt_rank: number
  player_id: string
  name: string
  position: string | null
  team: string | null
  prob: number
  certain: boolean
}

export interface SimBoard {
  run: { run_id: string; my_slot: number; created_at: string } | null
  teams: number
  rounds: number
  order: DraftOrderEntry[]
  cells: SimBoardCell[]
}

export async function fetchSimBoard(): Promise<SimBoard> {
  const res = await fetch('/api/sim/board')
  if (!res.ok) throw new Error('Failed to load the predicted board')
  return res.json()
}
```

- [ ] **Step 2: Write the grid component**

Create `web/src/components/DraftGrid.tsx`:

```tsx
import { useMemo, useState } from 'react'
import type { SimBoard, SimBoardCell } from '../api'

interface DraftGridProps {
  board: SimBoard
  onSelectPlayer: (playerId: string) => void
}

/** Cells keyed by overall pick, primary first. */
function byPick(cells: SimBoardCell[]): Map<number, SimBoardCell[]> {
  const map = new Map<number, SimBoardCell[]>()
  for (const c of cells) {
    const list = map.get(c.overall_pick) ?? []
    list.push(c)
    map.set(c.overall_pick, list)
  }
  for (const list of map.values()) list.sort((a, b) => a.alt_rank - b.alt_rank)
  return map
}

/** Overall pick number for a (round, slot) in a snake draft. Round 1 runs
 *  slot 1..teams, round 2 runs teams..1, and so on. */
function pickNumber(round: number, slot: number, teams: number): number {
  const offsetInRound = round % 2 === 1 ? slot - 1 : teams - slot
  return (round - 1) * teams + offsetInRound + 1
}

export default function DraftGrid({ board, onSelectPlayer }: DraftGridProps) {
  const cells = useMemo(() => byPick(board.cells), [board.cells])
  const [hovered, setHovered] = useState<number | null>(null)
  const rounds = Array.from({ length: board.rounds }, (_, i) => i + 1)
  const mySlot = board.run?.my_slot ?? null

  return (
    <div className="grid-wrap">
      <table className="draft-grid">
        <thead>
          <tr>
            <th className="grid-round">Round</th>
            <th className="grid-dir" aria-label="Pick direction" />
            {board.order.map((e) => (
              <th key={e.slot} className={e.slot === mySlot ? 'grid-mine' : undefined}>
                {e.manager}
                {e.slot === mySlot && <span className="grid-you"> (you)</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rounds.map((round) => (
            <tr key={round}>
              <td className="grid-round">{round}</td>
              <td className="grid-dir">{round % 2 === 1 ? '›' : '‹'}</td>
              {board.order.map((e) => {
                const pick = pickNumber(round, e.slot, board.teams)
                const list = cells.get(pick) ?? []
                const primary = list[0]
                const alts = list.slice(1)
                const mine = e.slot === mySlot
                if (!primary) {
                  return <td key={e.slot} className="grid-cell grid-empty" />
                }
                const state = primary.certain ? 'is-certain' : mine ? 'is-projected' : 'is-predicted'
                const pos = (primary.position ?? '').toLowerCase()
                return (
                  <td
                    key={e.slot}
                    className={`grid-cell ${state}`}
                    style={pos ? { borderLeftColor: `var(--pos-${pos})` } : undefined}
                    onMouseEnter={() => setHovered(pick)}
                    onMouseLeave={() => setHovered((h) => (h === pick ? null : h))}
                  >
                    <button
                      type="button"
                      className="grid-pick"
                      onClick={() => onSelectPlayer(primary.player_id)}
                    >
                      <span className="grid-name">{primary.name}</span>
                      <span className="grid-meta">
                        {primary.team ?? '—'}
                        {primary.position ? `, ${primary.position}` : ''}
                        {/* A projected cell is what a greedy plan does, not a
                            forecast -- showing its near-1.0 frequency as a
                            probability would read as confidence it hasn't got. */}
                        {!primary.certain && !mine && (
                          <span className="grid-prob">{Math.round(primary.prob * 100)}%</span>
                        )}
                      </span>
                    </button>
                    {hovered === pick && alts.length > 0 && (
                      <div className="grid-alts" role="tooltip">
                        <div className="grid-alts-title">Also in play</div>
                        {alts.map((a) => (
                          <button
                            key={a.player_id}
                            type="button"
                            className="grid-alt"
                            onClick={() => onSelectPlayer(a.player_id)}
                          >
                            <span>{a.name}</span>
                            <span className="grid-prob">{Math.round(a.prob * 100)}%</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 3: Write the page shell**

Create `web/src/components/DraftBoardPage.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { fetchSimBoard, type SimBoard } from '../api'
import DraftGrid from './DraftGrid'
import PlayerCard from './PlayerCard'
import TopBar from './TopBar'
import FreshnessBadge from './FreshnessBadge'

export default function DraftBoardPage() {
  const [board, setBoard] = useState<SimBoard | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)

  useEffect(() => {
    fetchSimBoard()
      .then(setBoard)
      .catch((e) => setError(e instanceof Error ? e.message : 'Load failed'))
  }, [])

  return (
    <div className="app">
      <TopBar search="" onSearch={() => {}} meta={<FreshnessBadge />} searchShortcutDisabled />
      <div className="app-body">
        <main className="main">
          {error && <p className="error">{error}</p>}
          {!board && !error && <p className="refreshing-hint">Loading…</p>}
          {board && board.order.length === 0 && (
            <p className="rail-empty">
              Run `make espn-import` to load your league, then set the draft
              order on the board page.
            </p>
          )}
          {board && board.order.length > 0 && (
            <>
              {!board.run && (
                <p className="rail-empty">
                  No simulation has run yet. Set your slot on the board page and
                  press Run — this grid fills in with the prediction.
                </p>
              )}
              <DraftGrid board={board} onSelectPlayer={setSelected} />
            </>
          )}
        </main>
      </div>
      {selected && <PlayerCard playerId={selected} onClose={() => setSelected(null)} />}
    </div>
  )
}
```

- [ ] **Step 4: Register the route and the nav link**

In `web/src/App.tsx`, add the import and the route:

```tsx
import DraftBoardPage from './components/DraftBoardPage'
```

```tsx
      <Route path="/draft-board" element={<DraftBoardPage />} />
```

In `web/src/components/TopBar.tsx`, add `import { NavLink } from 'react-router-dom'` and render the two links immediately before the `{meta}` slot in its JSX:

```tsx
      <nav className="top-nav">
        <NavLink to="/" end>Board</NavLink>
        <NavLink to="/draft-board">Grid</NavLink>
      </nav>
```

- [ ] **Step 5: Add the styles**

Append to `web/src/App.css`, using only tokens already defined in `index.css`:

```css
.top-nav { display: flex; gap: var(--space-3); margin-right: var(--space-3); }
.top-nav a { color: var(--text-3); text-decoration: none; font-size: 0.85rem; }
.top-nav a.active { color: var(--text-1); }

.grid-wrap { overflow-x: auto; }
.draft-grid { border-collapse: collapse; font-size: 0.78rem; }
.draft-grid th,
.draft-grid td { border: 1px solid var(--border); vertical-align: top; }
.draft-grid th {
  background: var(--bg-1);
  color: var(--text-2);
  padding: 6px 8px;
  text-align: left;
  font-weight: 500;
  white-space: nowrap;
}
.draft-grid th.grid-mine { color: var(--accent); }
.grid-you { color: var(--text-3); }
.grid-round,
.grid-dir {
  background: var(--bg-1);
  color: var(--text-3);
  text-align: center;
  padding: 6px;
  font-variant-numeric: tabular-nums;
}
.grid-dir { width: 20px; }
.grid-cell { position: relative; width: 150px; border-left-width: 3px; }
.grid-cell.is-predicted { background: var(--bg-0); }
.grid-cell.is-projected { background: var(--bg-2); }
.grid-cell.is-certain { background: var(--bg-1); }
.grid-empty { background: var(--bg-0); height: 42px; }
.grid-pick {
  display: block;
  width: 100%;
  background: none;
  border: 0;
  padding: 5px 7px;
  text-align: left;
  color: var(--text-1);
  cursor: pointer;
  font: inherit;
}
.grid-pick:hover { background: var(--accent-bg); }
.grid-name { display: block; }
.grid-meta { display: block; color: var(--text-3); font-size: 0.72rem; }
.grid-prob { margin-left: 6px; }
.grid-alts {
  position: absolute;
  z-index: 5;
  top: 100%;
  left: 0;
  min-width: 170px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  padding: 5px;
}
.grid-alts-title {
  color: var(--text-3);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 2px 4px;
}
.grid-alt {
  display: flex;
  justify-content: space-between;
  width: 100%;
  background: none;
  border: 0;
  padding: 3px 4px;
  color: var(--text-2);
  cursor: pointer;
  font: inherit;
  font-size: 0.74rem;
}
.grid-alt:hover { background: var(--accent-bg); color: var(--text-1); }
```

- [ ] **Step 6: Typecheck and build**

Run: `cd web && npm run build`
Expected: succeeds with no TypeScript errors. `PlayerCard` already exists from Task 4.

- [ ] **Step 7: Commit**

```bash
git add web/src
git commit -m "feat: predicted draft board grid"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the grid**

Add a subsection to the existing "Draft simulation" section of `README.md` covering:

- The `/draft-board` route and the header link to it
- That every cell is the model's most likely pick, and hovering shows the second and third choices with their probabilities
- That the cell name is deduplicated across the whole board by a one-to-one assignment, so it is the most coherent single draft rather than 120 independent answers — and that this is why the alternates matter
- That your own column is filled by the rollouts' greedy policy, is labelled as projected, and deliberately shows no probability, because a near-deterministic policy's frequency is not a forecast
- That a pick already marked drafted renders as fact
- That the grid reflects the last run plus the current drafted state, and re-running is a deliberate action

- [ ] **Step 2: Verify**

Run: `.venv/bin/pytest -q && cd web && npm run build`
Expected: all tests pass, build succeeds.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: predicted draft board"
```
