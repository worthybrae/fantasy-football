# Draft Room Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank the draft board in projected points, recommend by what a pick gains over waiting, and let the user make ESPN picks from inside Draft Helper.

**Architecture:** VOR is rebuilt from `proj_points` instead of the within-position percentile `composite`, which makes cross-position ranking meaningful. A new pure module ranks the available pool by `gain_now` — need-weighted value over the best player at that position expected to survive to the user's next pick — using the simulator's existing `survival()` for the probabilities. The websocket that `api/live.py` already holds gains a send path, so `POST /api/live/select` makes the pick ESPN would otherwise take from the browser. The `/draft` route is rebuilt around a pinned clock/roster rail and a tabbed main area.

**Tech Stack:** Python 3.10, pandas, numpy, DuckDB, FastAPI, `websockets` (sync client), pytest. React 19, TypeScript, Vite, react-router-dom 7. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-18-draft-room-design.md`

## Global Constraints

- No new Python or npm dependencies. Everything needed is already in `requirements.txt` and `web/package.json`.
- Python tests: `.venv/bin/pytest`. Frontend typecheck/build: `cd web && npm run build`. Lint: `cd web && npm run lint`.
- `composite` and the five factors (`production`, `durability`, `role`, `environment`, `schedule`) stay within-position percentiles. Do not change `scoring/factors.py`.
- The board's `drafted` table has exactly one writer: the listener path in `api/live.py`. No new code writes to it.
- The UI must only ever show a pick ESPN confirmed. No optimistic pick state.
- Existing routes `/` and `/players/:slug` are untouched.
- `need_weight` values live in one dict in `scoring/config.py`: starter `1.0`, flex `0.75`, bench `0.35`, capped `0.0`.
- Commit after each task with a `feat:` / `fix:` / `test:` prefix, matching the repo's existing commit style.

---

### Task 1: Projected points and points-based VOR on the board

Today `scoring/composite.apply_vor` differences two within-position percentiles and `scoring/board.py:340` sorts the whole board on the result. This task changes the unit to projected fantasy points.

`scoring/draft_sim.projections()` already computes projected season points but lives in the simulator, which imports from `scoring/board.py` — so the function moves to `board.py` (where `adp_match_key` and `read_table` already are) and `draft_sim` imports it back. That direction has no cycle.

`projections()` reads `row.get("stats")` as its second-rung fallback, and `stats` is merged into the board frame *after* the scoring block today. So the `_latest_season_stats` merge moves up, above the scoring block. `add_market` computes `edge = market_rank - rank`, so ranking must still happen before `add_market` — the order below preserves that.

**Files:**
- Modify: `scoring/board.py` — move `projections`, `POSITION_FLOOR`, `GAMES` in; reorder `build_board`; add `proj_points` to `_BOARD_COLUMNS`
- Modify: `scoring/draft_sim.py:52-85` — delete `projections`, import it from `board`; `build_pool` prefers `board["proj_points"]`
- Modify: `scoring/composite.py:18-32` — `apply_vor` gains a `column` parameter
- Test: `tests/test_composite.py`, `tests/test_board.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `scoring.board.projections(conn, board: pd.DataFrame) -> pd.Series` — projected season points, indexed by `player_id`.
  - `scoring.board.POSITION_FLOOR: dict[str, float]`, `scoring.board.GAMES: int`.
  - `scoring.composite.apply_vor(df, replacement_ranks=None, column="composite") -> pd.DataFrame` — writes a `vor` column as `df[column] - df[column]` at that position's replacement rank.
  - Board frames gain a `proj_points: float` column, and `vor` is now in projected fantasy points.

- [ ] **Step 1: Write the failing test for `apply_vor`'s column parameter**

Add to `tests/test_composite.py`:

```python
def test_apply_vor_differences_the_named_column():
    """VOR must be expressible in projected points, not just composite.

    The board ranks across positions on this number, and a difference of
    two within-position percentiles has no cross-position meaning -- that
    is what put a TE at ADP 149 thirteenth overall.
    """
    df = pd.DataFrame({
        "player_id": ["a", "b", "c", "d"],
        "position": ["RB", "RB", "TE", "TE"],
        "composite": [90.0, 50.0, 90.0, 50.0],
        "proj_points": [280.0, 150.0, 140.0, 100.0],
    })
    out = apply_vor(df, {"RB": 2, "TE": 2}, column="proj_points")
    assert list(out["vor"]) == [130.0, 0.0, 40.0, 0.0]


def test_apply_vor_still_defaults_to_composite():
    df = pd.DataFrame({
        "player_id": ["a", "b"],
        "position": ["RB", "RB"],
        "composite": [90.0, 50.0],
        "proj_points": [280.0, 150.0],
    })
    out = apply_vor(df, {"RB": 2})
    assert list(out["vor"]) == [40.0, 0.0]
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/pytest tests/test_composite.py::test_apply_vor_differences_the_named_column -v`
Expected: FAIL — `apply_vor() got an unexpected keyword argument 'column'`

- [ ] **Step 3: Add the parameter**

In `scoring/composite.py`, replace `apply_vor`'s body's use of `"composite"` with the parameter:

```python
def apply_vor(df: pd.DataFrame, replacement_ranks: dict | None = None,
              column: str = "composite") -> pd.DataFrame:
    """Value over replacement, as a difference within `column`.

    `column` exists because the two callers want different units. The board
    ranks across positions and must difference `proj_points` -- a difference
    of two within-position percentiles (which is what `composite` is, see
    factors.normalize_within_position) has no cross-position meaning, and
    ranking on one put a TE at ADP 149 thirteenth overall. Callers that only
    compare within a position can keep the composite default.
    """
    ranks = REPLACEMENT_RANK if replacement_ranks is None else replacement_ranks
    out = df.copy()
    out["vor"] = 0.0
    for pos, grp in out.groupby("position"):
        ranked = grp.sort_values(column, ascending=False)
        idx = min(ranks.get(pos, 9), len(ranked)) - 1
        replacement = ranked.iloc[idx][column]
        out.loc[grp.index, "vor"] = grp[column] - replacement
    return out
```

- [ ] **Step 4: Run both composite tests**

Run: `.venv/bin/pytest tests/test_composite.py -v`
Expected: PASS

- [ ] **Step 5: Move `projections` into `scoring/board.py`**

Cut `GAMES`, `POSITION_FLOOR`, and the whole `projections` function (currently `scoring/draft_sim.py:38-85`) and paste them into `scoring/board.py`, directly above `_latest_season_stats`. `board.py` already imports `read_table` and defines `adp_match_key`, so the only new import it needs is `numpy as np`.

Then in `scoring/draft_sim.py`, replace the deleted definitions with an import, extending the existing board import line:

```python
from scoring.board import (FANTASY_POSITIONS, GAMES, POSITION_FLOOR,
                           _norm_name, adp_match_key, projections)
```

Leave `DEFAULT_AVAILABILITY` and `FLEX_POSITIONS` in `draft_sim.py` — nothing on the board uses them.

- [ ] **Step 6: Run the suite to confirm the move is behavior-neutral**

Run: `.venv/bin/pytest -q`
Expected: PASS. Any failure here is an import error from the move, not new behavior.

- [ ] **Step 7: Commit the move**

```bash
git add scoring/composite.py scoring/draft_sim.py scoring/board.py tests/test_composite.py
git commit -m "refactor: projections move to board.py; apply_vor takes a column"
```

- [ ] **Step 8: Write the failing test for a points-ranked board**

Add to `tests/test_board.py`. Follow the file's existing fixture style for building a `conn`; if it has a helper that builds a small board, reuse it rather than writing a new one.

```python
def test_board_vor_is_in_projected_points(tmp_path):
    """The board's headline ranking must be cross-position comparable.

    Regression for the Mark Andrews case: with VOR built from composite (a
    within-position percentile), a TE at ADP 149 ranked 13th overall because
    being far above TE10 in percentile scored the same as being far above
    RB22 in percentile.
    """
    conn = _fixture_conn(tmp_path)          # existing helper in this file
    board = build_board(conn)
    assert "proj_points" in board.columns
    # vor is a points difference, so it moves on the same scale as proj_points
    rbs = board[board["position"] == "RB"]
    assert (rbs["vor"] - (rbs["proj_points"] - rbs["proj_points"].nlargest(
        22).iloc[-1])).abs().max() < 1e-6
    # and the board is sorted by it
    assert board["vor"].is_monotonic_decreasing
```

- [ ] **Step 9: Run it and confirm it fails**

Run: `.venv/bin/pytest tests/test_board.py::test_board_vor_is_in_projected_points -v`
Expected: FAIL — `proj_points` is not a board column.

- [ ] **Step 10: Reorder `build_board` and switch the VOR unit**

In `scoring/board.py`'s `build_board`, move the stats merge above the scoring block and rewrite the scoring block. The current block is:

```python
    uni["composite"] = compute_composite(uni, weights or DEFAULT_WEIGHTS)
    uni = apply_vor(uni, settings.replacement_ranks)
    uni = assign_tiers(uni)
    uni = uni.sort_values("vor", ascending=False).reset_index(drop=True)
    uni["rank"] = uni.index + 1
    uni = add_market(uni, espn, fp, sleeper, mfl=mfl, cbs=cbs, fmt=fmt)
```

Replace it with:

```python
    # `stats` is merged before scoring, not after: projections() falls back
    # to recency-weighted PPG out of this column when ESPN has no season
    # projection for a player, and it has to run before the ranking that
    # depends on it. add_market still runs after, because its `edge` column
    # is market_rank - rank and needs the rank this block assigns.
    uni = uni.merge(_latest_season_stats(weekly), on="player_id", how="left")

    uni["composite"] = compute_composite(uni, weights or DEFAULT_WEIGHTS)
    # Projected season points, and VOR as a difference of them. NOT a
    # difference of composites: composite is a within-position percentile
    # (factors.normalize_within_position), so differencing it produces a
    # number with no cross-position meaning -- and this frame is then sorted
    # across positions. See the spec's section 2.
    proj = projections(conn, uni)
    uni["proj_points"] = uni["player_id"].map(proj).astype(float)
    uni = apply_vor(uni, settings.replacement_ranks, column="proj_points")
    uni = assign_tiers(uni)
    uni = uni.sort_values("vor", ascending=False).reset_index(drop=True)
    uni["rank"] = uni.index + 1
    uni = add_market(uni, espn, fp, sleeper, mfl=mfl, cbs=cbs, fmt=fmt)
```

Then delete the now-duplicated merge further down:

```python
    uni = uni.merge(_latest_season_stats(weekly), on="player_id", how="left")
```

And add `"proj_points"` to `_BOARD_COLUMNS`, after `"composite"`:

```python
_BOARD_COLUMNS = [
    "player_id", "name", "position", "team", "bye", "production", "durability",
    "role", "environment", "schedule", "composite", "proj_points", "vor",
    "tier", "market_rank",
    "market_spread", "market_sources", "espn_ppr_rank", "espn_id", "ffc_rank", "edge",
    "rookie", "drafted", "rank",
    "stats", "avail_pct", "ev", "ev_se",
]
```

- [ ] **Step 11: Run the board tests**

Run: `.venv/bin/pytest tests/test_board.py -v`
Expected: PASS. Tests that assert a specific board order will fail — that is the point of this change. Update their expected order to the points-based one and note in the test why it moved. Do not weaken an assertion to make it pass.

- [ ] **Step 12: Make `build_pool` reuse the board's column**

In `scoring/draft_sim.build_pool`, replace:

```python
    points = projections(conn, board)
```

with:

```python
    # The board already carries this (build_board computes it to rank on);
    # recomputing it here would be a second, silently divergent copy. A bare
    # fixture board without the column still falls back to computing it.
    if "proj_points" in board.columns:
        points = board.set_index("player_id")["proj_points"].astype(float)
    else:
        points = projections(conn, board)
```

- [ ] **Step 13: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS. Fix any test whose expected ranking moved, adjusting the expectation to the points-based ordering.

- [ ] **Step 14: Sanity-check against real data**

Run:

```bash
.venv/bin/python -c "
from pipeline.db import connect
from scoring.board import build_board
b = build_board(connect())
print(b[['rank','name','position','proj_points','vor','market_rank']].head(20).to_string())
"
```

Expected: Mark Andrews, Travis Kelce and the defenses are no longer inside the top 30. Their ADPs are 149, 104 and 176+. If they still are, the reorder did not take effect — do not proceed.

- [ ] **Step 15: Commit**

```bash
git add scoring/board.py scoring/draft_sim.py tests/test_board.py
git commit -m "fix: rank the board on projected points, not a percentile difference"
```

---

### Task 2: `gain_now` — value over the next man up

A pure module, no database and no network, so it is fully unit-testable. It answers "what do I lose by waiting" rather than "who is best", which is the number that stops the tool recommending a quarterback whose backup is nearly as good.

**Files:**
- Create: `scoring/gain.py`
- Modify: `scoring/config.py` — add `NEED_WEIGHTS`
- Test: `tests/test_gain.py`

**Interfaces:**
- Consumes: `scoring.draft_sim.SimPool` (fields `player_id`, `position`, `points`, `vor`), `scoring.draft_sim._roster_cap`, `scoring.league.LeagueSettings`.
- Produces:
  - `scoring.config.NEED_WEIGHTS: dict[str, float]` — keys `"starter"`, `"flex"`, `"bench"`, `"capped"`.
  - `scoring.gain.need_kind(settings, counts: dict, position: str) -> str` — one of those four keys.
  - `scoring.gain.need_weight(settings, counts: dict, position: str) -> float`
  - `scoring.gain.fills_slot(settings, counts: dict, position: str) -> str` — a display label: `"QB"`, `"RB2"`, `"FLEX"`, `"BENCH"`, `"—"`.
  - `scoring.gain.expected_best_next(values, survive) -> float`
  - `scoring.gain.rank_available(pool, settings, taken, counts, survive) -> pd.DataFrame` with columns `player_id, position, proj_points, vor_points, gain_now, survive_pct, fills, rank`, sorted by `gain_now` descending, `rank` starting at 1.

- [ ] **Step 1: Add the weights to config**

In `scoring/config.py`, below `FLEX_SHARES`:

```python
# How much a position is worth to a roster that already holds `counts`.
# Multiplies the value a pick gains over waiting (see scoring/gain.py), so a
# position at its roster cap contributes nothing however good the player is.
# Starting values, to calibrate against replayed drafts -- not derived.
NEED_WEIGHTS = {"starter": 1.0, "flex": 0.75, "bench": 0.35, "capped": 0.0}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_gain.py`:

```python
import numpy as np
import pandas as pd
import pytest

from scoring.gain import (expected_best_next, fills_slot, need_kind,
                          need_weight, rank_available)
from scoring.league import LeagueSettings


def settings():
    return LeagueSettings(
        season=2026, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=2, bench=5, scoring={}, draft_type="SNAKE")


def test_need_kind_open_starter():
    assert need_kind(settings(), {"RB": 1}, "RB") == "starter"


def test_need_kind_flex_when_starters_full():
    assert need_kind(settings(), {"RB": 2}, "RB") == "flex"


def test_need_kind_bench_when_flex_full():
    # 2 RB starters + 2 WR starters + 2 more of either fill both FLEX slots
    counts = {"RB": 4, "WR": 2}
    assert need_kind(settings(), counts, "RB") == "bench"


def test_need_kind_capped_at_the_roster_cap():
    # _roster_cap caps K at 1
    assert need_kind(settings(), {"K": 1}, "K") == "capped"


def test_need_weight_is_zero_when_capped():
    assert need_weight(settings(), {"K": 1}, "K") == 0.0


def test_fills_slot_names_the_open_starter():
    assert fills_slot(settings(), {"RB": 1}, "RB") == "RB2"
    assert fills_slot(settings(), {}, "QB") == "QB"
    assert fills_slot(settings(), {"RB": 2}, "RB") == "FLEX"
    assert fills_slot(settings(), {"RB": 4, "WR": 2}, "RB") == "BENCH"
    assert fills_slot(settings(), {"K": 1}, "K") == "—"


def test_expected_best_next_is_the_best_when_survival_is_certain():
    assert expected_best_next([40.0, 90.0, 10.0], [1.0, 1.0, 1.0]) == 90.0


def test_expected_best_next_is_zero_when_nobody_survives():
    assert expected_best_next([40.0, 90.0], [0.0, 0.0]) == 0.0


def test_expected_best_next_weights_by_survival():
    """The best player survives half the time; otherwise the runner-up does.

    0.5*100 + 0.5*(1.0*60) = 80.
    """
    assert expected_best_next([100.0, 60.0], [0.5, 1.0]) == pytest.approx(80.0)


def test_gain_now_prefers_the_position_with_a_cliff_behind_it():
    """The Josh Allen case.

    The QB has the larger raw value over replacement, but the next QB is
    nearly as good, so waiting costs almost nothing. The WR is worth less in
    absolute terms but nothing comparable survives to the next pick.
    """
    pool = _pool(
        player_id=["qb1", "qb2", "wr1", "wr2"],
        position=["QB", "QB", "WR", "WR"],
        points=[391.0, 382.0, 241.0, 190.0],
        vor=[91.0, 82.0, 76.0, 25.0],
    )
    taken = np.zeros(4, dtype=bool)
    survive = np.array([0.31, 0.49, 0.22, 0.74])
    out = rank_available(pool, settings(), taken, {}, survive)
    assert out.iloc[0]["player_id"] == "wr1"
    assert out.set_index("player_id").loc["qb1", "gain_now"] < \
        out.set_index("player_id").loc["wr1", "gain_now"]


def test_rank_available_drops_taken_players():
    pool = _pool(player_id=["a", "b"], position=["RB", "RB"],
                 points=[200.0, 150.0], vor=[50.0, 0.0])
    taken = np.array([True, False])
    out = rank_available(pool, settings(), taken, {}, np.array([0.5, 0.5]))
    assert list(out["player_id"]) == ["b"]


def test_rank_available_zeroes_a_capped_position():
    pool = _pool(player_id=["k1"], position=["K"], points=[130.0], vor=[20.0])
    out = rank_available(pool, settings(), np.zeros(1, dtype=bool),
                         {"K": 1}, np.array([0.9]))
    assert out.iloc[0]["gain_now"] == 0.0
    assert out.iloc[0]["fills"] == "—"


class _pool:
    """Minimal stand-in for SimPool: gain.py reads four fields."""

    def __init__(self, player_id, position, points, vor):
        self.player_id = np.array(player_id, dtype=object)
        self.position = np.array(position, dtype=object)
        self.points = np.array(points, dtype=float)
        self.vor = np.array(vor, dtype=float)
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `.venv/bin/pytest tests/test_gain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scoring.gain'`

- [ ] **Step 4: Write the module**

Create `scoring/gain.py`:

```python
"""Rank the available pool by what a pick gains over waiting.

The board's `vor` says how much better a player is than a replacement-level
one at his position. On the clock that is the wrong question: what matters is
how much better he is than whoever will STILL BE THERE at your next pick. A
quarterback 91 points over replacement whose backup is 82 points over
replacement costs you 9 points to pass on, not 91 -- which is why ranking on
raw VOR reaches for quarterbacks and tight ends.

`gain_now` is that difference, weighted by whether the roster can actually
start the player. It is deterministic: the only stochastic input is
`survive`, the per-player probability of lasting to the next pick, which
draft_sim.survival() estimates. That is a far lower-variance quantity than
expected end-of-draft roster value -- it counts one event per rollout instead
of averaging a sum over fifteen simulated picks -- which is why the ranking
moved here and off search_pick.
"""
import numpy as np
import pandas as pd

from scoring.config import NEED_WEIGHTS
from scoring.draft_sim import FLEX_POSITIONS, _roster_cap

_NO_SLOT = "—"


def _flex_used(settings, counts: dict) -> int:
    """How many FLEX slots the roster's overflow already occupies."""
    return sum(max(0, counts.get(pos, 0) - settings.starters.get(pos, 0))
               for pos in FLEX_POSITIONS)


def need_kind(settings, counts: dict, position: str) -> str:
    """Which of NEED_WEIGHTS' four cases this position is in for this roster.

    `counts` is position -> how many the roster already holds, the shape
    draft_sim._seed_rosters produces.
    """
    caps = _roster_cap(settings)
    held = counts.get(position, 0)
    if held >= caps.get(position, held + 1):
        return "capped"
    if held < settings.starters.get(position, 0):
        return "starter"
    if position in FLEX_POSITIONS and _flex_used(settings, counts) < settings.flex_slots:
        return "flex"
    total = sum(counts.values())
    if total < settings.rounds:
        return "bench"
    return "capped"


def need_weight(settings, counts: dict, position: str) -> float:
    return float(NEED_WEIGHTS[need_kind(settings, counts, position)])


def fills_slot(settings, counts: dict, position: str) -> str:
    """The roster slot this player would occupy, as the UI labels it."""
    kind = need_kind(settings, counts, position)
    if kind == "starter":
        n = settings.starters.get(position, 0)
        held = counts.get(position, 0)
        return position if n <= 1 else f"{position}{held + 1}"
    if kind == "flex":
        return "FLEX"
    if kind == "bench":
        return "BENCH"
    return _NO_SLOT


def expected_best_next(values, survive) -> float:
    """Expected value of the best survivor among these players.

    Walks them best-first and charges each one the probability that he
    survives AND nobody better did. The tail where nobody survives
    contributes zero, which is the right floor: it means the position is
    gone and waiting costs you everything.
    """
    values = np.asarray(values, dtype=float)
    survive = np.asarray(survive, dtype=float)
    if values.size == 0:
        return 0.0
    total = 0.0
    none_better = 1.0
    for i in np.argsort(-values, kind="stable"):
        p = float(survive[i])
        total += float(values[i]) * p * none_better
        none_better *= 1.0 - p
        if none_better <= 1e-12:
            break
    return total


def rank_available(pool, settings, taken, counts: dict, survive) -> pd.DataFrame:
    """The available pool, ranked by gain_now descending.

    `taken` is the pool-aligned boolean mask of players already drafted,
    `counts` my own roster's position counts, `survive` the pool-aligned
    probability each player is still there at my next pick.
    """
    available = np.flatnonzero(~np.asarray(taken))
    if available.size == 0:
        return pd.DataFrame(columns=["player_id", "position", "proj_points",
                                     "vor_points", "gain_now", "survive_pct",
                                     "fills", "rank"])
    survive = np.asarray(survive, dtype=float)

    # One expected-best per position, not per player: it depends only on the
    # position's own survivors, so computing it inside the player loop would
    # redo the same sort once per player at that position.
    next_best = {}
    for pos in set(pool.position[available]):
        at_pos = available[pool.position[available] == pos]
        next_best[pos] = expected_best_next(pool.vor[at_pos], survive[at_pos])

    rows = []
    for idx in available:
        pos = str(pool.position[idx])
        weight = need_weight(settings, counts, pos)
        rows.append({
            "player_id": pool.player_id[idx],
            "position": pos,
            "proj_points": float(pool.points[idx]),
            "vor_points": float(pool.vor[idx]),
            "gain_now": weight * (float(pool.vor[idx]) - next_best[pos]),
            "survive_pct": float(survive[idx]) * 100.0,
            "fills": fills_slot(settings, counts, pos),
        })
    out = pd.DataFrame(rows).sort_values(
        "gain_now", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_gain.py -v`
Expected: PASS

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scoring/gain.py scoring/config.py tests/test_gain.py
git commit -m "feat: rank by what a pick gains over waiting, not raw VOR"
```

---

### Task 3: Serve `gain_now` from `/api/live/state`

`api/live.py:_recompute` currently calls `search_pick`, which ranks twelve candidates by expected end-of-draft roster value at 12–40 rollouts. The module's own docstring records `se=7.406298` on that estimate, which is larger than the spread it is resolving, so the top slot is decided by noise. Replace it with `survival()` plus `rank_available`.

`survival()` is one cheap pass, so the rollout budget can be flat and generous instead of shrinking as the clock runs down.

**Files:**
- Modify: `api/live.py` — `_recompute` (around `api/live.py:625-668`), `rollouts_for` (`api/live.py:465-488`), the `/api/live/state` response
- Test: `tests/test_live_api.py`

**Interfaces:**
- Consumes: `scoring.gain.rank_available` from Task 2; `scoring.draft_sim.survival`, `scoring.draft_sim._seed_rosters`, `scoring.draft_sim._drafted_state`.
- Produces: `/api/live/state`'s `candidates` array now carries objects shaped `{player_id, position, proj_points, vor_points, gain_now, survive_pct, fills, rank}` instead of `{player_id, ev, se, applied_pct, rank}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_live_api.py`, following the file's existing pattern for standing up an app with a fake session (reuse whatever helper it already has; do not invent a new fixture style).

```python
def test_state_candidates_carry_gain_now(live_app):
    """The recommendation is gain_now, not simulated end-of-draft EV.

    EV's standard error was larger than the spread between good candidates,
    so the top slot moved with the sampling seed. gain_now is deterministic
    given the survival estimate.
    """
    client, state = live_app          # existing helper in this file
    _run_one_recompute(client, state) # existing helper in this file
    body = client.get("/api/live/state").json()
    assert body["candidates"], "no recommendation produced"
    row = body["candidates"][0]
    assert set(row) >= {"player_id", "position", "proj_points", "vor_points",
                        "gain_now", "survive_pct", "fills", "rank"}
    assert "ev" not in row
    gains = [c["gain_now"] for c in body["candidates"]]
    assert gains == sorted(gains, reverse=True)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/pytest tests/test_live_api.py::test_state_candidates_carry_gain_now -v`
Expected: FAIL — the candidate rows still carry `ev`.

- [ ] **Step 3: Replace the search in `_recompute`**

In `api/live.py`, change the import on line 151:

```python
from scoring.draft_sim import (_drafted_state, _seed_rosters, snake_slots,
                               survival)
from scoring.gain import rank_available
```

(`search_pick` is no longer imported here. It stays in `scoring/draft_sim.py` for `run_sim`.)

Then replace the body between `cur = active_conn.cursor()` and `finally:` in `_recompute`:

```python
        cur = active_conn.cursor()
        try:
            taken, taken_order = _drafted_state(cur, session.pool)
            # My own roster so far, so need_weight can see which slots are
            # still open. _seed_rosters replays every pick to the slot that
            # was on the clock for it, which is the same attribution the
            # simulator resumes from.
            rosters, _ = _seed_rosters(session.pool, session.settings, taken_order)
            counts = rosters[session.my_slot]["counts"]
            avail = survival(
                session.pool, session.settings, session.slot_managers,
                session.my_slot, taken, session.betas,
                n_rollouts=SURVIVAL_ROLLOUTS, seed=session.seed,
                taken_order=taken_order)["avail_pct"].to_numpy()
            frame = rank_available(session.pool, session.settings, taken,
                                   counts, avail / 100.0)
        finally:
            cur.close()
```

`survival()` returns `avail_pct` on a 0–100 scale (see its docstring and `scoring/draft_sim.py:948`), and `rank_available` wants a probability, hence the `/ 100.0`. If a check of `survival`'s return shows it is already 0–1, drop the division and say so in the commit message.

- [ ] **Step 4: Replace the rollout budget**

`rollouts_for` existed because `search_pick` ran a full draft per candidate per rollout and had to be rationed against the pick clock. `survival` stops at the next turn and runs once, so the budget is flat. Above `rollouts_for` in `api/live.py`, add:

```python
# survival() runs ONE set of rollouts that stop at my next turn, not one full
# draft per candidate, so the old clock-rationed budget (12/25/40, see
# rollouts_for) is no longer the constraint it was priced against. This is
# the whole recompute cost now, and it buys a materially tighter survival
# estimate for a fraction of what search_pick cost.
SURVIVAL_ROLLOUTS = 400
```

Leave `rollouts_for` and its constants in place — `picks_until_turn` is still used by the UI's "picks until your turn", and removing a tested helper is out of scope for this task.

- [ ] **Step 5: Run the live-api tests**

Run: `.venv/bin/pytest tests/test_live_api.py -v`
Expected: PASS. Tests that assert on `ev`/`applied_pct` in candidates will fail — update them to the new columns. Do not delete a test to make it pass; if a test's subject no longer exists, rewrite it against the replacement and say so.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add api/live.py tests/test_live_api.py
git commit -m "feat: live recommendation ranks on gain_now, not simulated EV"
```

---

### Task 4: A send path on the live socket

`run_socket_listener` opens its websocket into a local `ws` and replaces it on every reconnect, so nothing outside can send on it. This task publishes a handle.

**Files:**
- Modify: `pipeline/draft_socket.py` — add `SocketHandle`, add `on_socket` to `run_socket_listener`
- Test: `tests/test_draft_socket.py` (create if absent; otherwise extend)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `pipeline.draft_socket.SocketHandle` with `send(text: str) -> None`, `alive() -> bool`, `attach(ws) -> None`, `detach() -> None`.
  - `run_socket_listener(..., on_socket=None)` — called with the handle once, on the first successful connect.
  - `SocketHandle.send` raises `ConnectionError` when no socket is attached.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_draft_socket.py`:

```python
import threading

import pytest

from pipeline import draft_socket
from pipeline.draft_socket import SocketHandle


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, text):
        if self.closed:
            raise ConnectionError("closed")
        self.sent.append(text)

    def close(self):
        self.closed = True


def test_handle_sends_to_the_attached_socket():
    ws = FakeSocket()
    handle = SocketHandle()
    handle.attach(ws)
    handle.send("SELECT 4429795\n")
    assert ws.sent == ["SELECT 4429795\n"]
    assert handle.alive() is True


def test_handle_refuses_to_send_with_nothing_attached():
    handle = SocketHandle()
    assert handle.alive() is False
    with pytest.raises(ConnectionError):
        handle.send("SELECT 1\n")


def test_handle_sends_to_the_new_socket_after_a_reconnect():
    """ESPN drops a live draft socket every few minutes. A send that lands on
    the dropped one is a pick that silently never happened."""
    old, new = FakeSocket(), FakeSocket()
    handle = SocketHandle()
    handle.attach(old)
    handle.detach()
    handle.attach(new)
    handle.send("SELECT 7\n")
    assert old.sent == []
    assert new.sent == ["SELECT 7\n"]


def test_run_socket_listener_publishes_a_handle(monkeypatch):
    ws = FakeSocket()

    def fake_recv(timeout=None):
        raise TimeoutError

    ws.recv = fake_recv
    monkeypatch.setattr(draft_socket, "_connect", lambda url, cookie: ws)

    seen = []
    stop = threading.Event()

    class Listener:
        def on_frame(self, payload):
            return False

    def on_socket(handle):
        seen.append(handle)
        stop.set()

    draft_socket.run_socket_listener(
        Listener(), "1", "2", "{SWID}", 3, stop_event=stop,
        on_socket=on_socket)
    assert len(seen) == 1
    assert isinstance(seen[0], SocketHandle)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/pytest tests/test_draft_socket.py -v`
Expected: FAIL — `cannot import name 'SocketHandle'`

- [ ] **Step 3: Add `SocketHandle`**

In `pipeline/draft_socket.py`, above `run_socket_listener`:

```python
class SocketHandle:
    """The live draft socket, as much of it as a sender needs.

    `run_socket_listener` owns the connection and swaps it on every
    reconnect -- ESPN ends even a pinged draft socket after a few minutes.
    A sender holding the raw socket would keep writing into a dead one, and
    a SELECT that lands there is a pick that silently never happened, on a
    clock. So the socket lives behind this handle: the listener attaches and
    detaches as it reconnects, senders only ever see the current one, and a
    send with nothing attached is a loud ConnectionError rather than a
    no-op.

    The lock is not about the socket's own thread-safety. It is about the
    swap: attach/detach run on the listener thread while send runs on
    whatever thread FastAPI hands the request.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ws = None

    def attach(self, ws) -> None:
        with self._lock:
            self._ws = ws

    def detach(self) -> None:
        with self._lock:
            self._ws = None

    def alive(self) -> bool:
        with self._lock:
            return self._ws is not None

    def send(self, text: str) -> None:
        with self._lock:
            ws = self._ws
            if ws is None:
                raise ConnectionError(
                    "the draft socket is not connected -- click the Draft "
                    "Helper bookmark again to reconnect")
            ws.send(text)
```

Add `import threading` to the module's imports.

- [ ] **Step 4: Publish it from the listener**

Change `run_socket_listener`'s signature to accept `on_socket=None`, and in its body:

```python
    cookie_header = f"SWID={swid}"
    url = socket_url(league_id, team_id, swid, token)
    handle = SocketHandle()
    if on_socket is not None:
        on_socket(handle)
    empty_reconnects = 0
```

Then inside the connect loop, immediately after `ws = _connect(url, cookie_header)` succeeds:

```python
        handle.attach(ws)
```

and inside the existing `finally:` block that closes `ws`, before the close attempt:

```python
            handle.detach()
```

The handle is published once, not per connect: it is stable across reconnects by design, which is the whole reason it exists.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_draft_socket.py -v`
Expected: PASS

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pipeline/draft_socket.py tests/test_draft_socket.py
git commit -m "feat: SocketHandle exposes the live draft socket for sending"
```

---

### Task 5: `POST /api/live/select`

Makes the pick. Sends `SELECT <espnPlayerId>` and waits for ESPN's `SELECTED` frame before answering. Writes nothing to `drafted` — the listener path already does that when the frame arrives, and two writers would let the board claim a pick ESPN did not take.

**Files:**
- Modify: `api/live.py` — record the handle in `state`, add the endpoint
- Modify: `pipeline/draft_listener.py` — record confirmed ESPN ids so the endpoint can wait on one
- Test: `tests/test_live_api.py`

**Interfaces:**
- Consumes: `pipeline.draft_socket.SocketHandle` from Task 4; `pipeline.espn_live._dst_espn_id`.
- Produces:
  - `DraftListener.selected_espn_ids: set[int]` — every ESPN player id seen in a `SELECTED` frame.
  - `POST /api/live/select` with body `{"player_id": str}`. `200 {"player_id", "espn_id", "pick_no"}` on confirmation; `409` off-turn or already drafted; `400` unresolvable ESPN id; `503` socket down; `504` sent but unconfirmed.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_live_api.py`:

```python
def test_select_off_turn_is_rejected_and_sends_nothing(live_app_on_socket):
    """A SELECT off-turn is at best ignored and at worst a misdraft."""
    client, state, ws = live_app_on_socket   # my_slot is NOT on the clock
    body = client.post("/api/live/select", json={"player_id": "p1"})
    assert body.status_code == 409
    assert ws.sent == []


def test_select_sends_the_espn_id_and_waits_for_confirmation(live_app_on_clock):
    client, state, ws, listener = live_app_on_clock
    ws.on_send(lambda text: listener.on_frame("SELECTED 30 4429795 3 {SWID}\n"))
    res = client.post("/api/live/select", json={"player_id": "00-0039139"})
    assert res.status_code == 200
    assert ws.sent == ["SELECT 4429795\n"]
    assert res.json()["espn_id"] == 4429795


def test_select_resolves_a_defense_to_its_negative_espn_id(live_app_on_clock):
    """Board rows for D/ST carry espn_id None; the real id is -(16000+team)."""
    client, state, ws, listener = live_app_on_clock
    ws.on_send(lambda text: listener.on_frame("SELECTED 30 -16033 3 {SWID}\n"))
    res = client.post("/api/live/select", json={"player_id": "dst_bal"})
    assert ws.sent == ["SELECT -16033\n"]
    assert res.json()["espn_id"] == -16033


def test_select_times_out_without_confirmation_and_writes_nothing(live_app_on_clock):
    """The one case the tool cannot resolve. The pick may or may not have
    landed, so the board must not claim either way."""
    client, state, ws, listener = live_app_on_clock
    before = _drafted_count(state)
    res = client.post("/api/live/select", json={"player_id": "00-0039139"})
    assert res.status_code == 504
    assert _drafted_count(state) == before


def test_select_with_no_socket_is_503(live_app_on_clock_no_socket):
    client, state = live_app_on_clock_no_socket
    assert client.post("/api/live/select",
                       json={"player_id": "00-0039139"}).status_code == 503


def test_select_an_already_drafted_player_is_409(live_app_on_clock):
    client, state, ws, listener = live_app_on_clock
    _mark_drafted(state, "00-0039139")
    assert client.post("/api/live/select",
                       json={"player_id": "00-0039139"}).status_code == 409
    assert ws.sent == []
```

Build `live_app_on_socket`, `live_app_on_clock` and `live_app_on_clock_no_socket` on top of whatever session helper `tests/test_live_api.py` already has, adding a fake socket with a `sent` list and an `on_send` hook. `_drafted_count` and `_mark_drafted` are small helpers over the test's DuckDB connection.

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/pytest tests/test_live_api.py -k select -v`
Expected: FAIL — 404, the route does not exist.

- [ ] **Step 3: Record confirmed ESPN ids on the listener**

In `pipeline/draft_listener.py`, in `DraftListener.__init__`:

```python
        # Every ESPN player id this socket has confirmed as picked. The
        # select endpoint waits on this rather than on the crosswalk-resolved
        # pick rows, because a player ESPN confirmed but the crosswalk cannot
        # map still went off the board -- and the person who just clicked
        # DRAFT needs to know it landed either way.
        self.selected_espn_ids = set()
```

and in `on_frame`, where `SELECTED` is handled (add the branch alongside the existing verb handling):

```python
        elif event.verb == "SELECTED" and len(event.args) > 1:
            espn_id = _as_int(event.args[1], None)
            if espn_id is not None:
                self.selected_espn_ids.add(espn_id)
```

If `SELECTED` is already handled in that method, add the two lines that populate the set to the existing branch rather than adding a second one.

- [ ] **Step 4: Record the handle in `api/live.py` state**

In the `state` dict initializer (`api/live.py:524`), alongside `"listener": None`:

```python
             # The live socket's send path, published by run_socket_listener
             # (see pipeline/draft_socket.SocketHandle). None whenever no
             # socket session is running -- /api/live/select refuses rather
             # than pretending.
             "socket": None,
```

In `_stop_listener`, alongside `state["listener"] = None`:

```python
            state["socket"] = None
```

In `live_connect_token` (`api/live.py:1161`), pass the callback into `run_socket_listener`:

```python
            def _on_socket(handle):
                with lock:
                    if state["listener"] is listener:
                        state["socket"] = handle

            run_socket_listener(listener, body.leagueId, body.teamId, body.swid,
                                body.token, on_change=on_change,
                                stop_event=stop_event, on_activity=on_activity,
                                on_socket=_on_socket)
```

Match the existing call's keyword style at that site; the identity guard mirrors what the other callbacks there already do.

- [ ] **Step 5: Add the endpoint**

In `api/live.py`, near the other route definitions:

```python
class SelectBody(BaseModel):
    player_id: str


# How long to wait for ESPN to echo the pick back. Long enough to cover a
# round trip on a busy draft server, short enough that a wedged request does
# not eat the pick clock it is supposed to protect.
SELECT_TIMEOUT_SECONDS = 8.0
SELECT_POLL_SECONDS = 0.1


def _espn_id_for(board_row) -> int | None:
    """The ESPN player id to send in a SELECT.

    Board rows carry `espn_id` for everyone ESPN ranks as a player. D/ST rows
    carry None, because ESPN models a defense as a negative synthetic id
    derived from the pro team -- the same rule build_crosswalk already reads
    picks back through, applied in the other direction.
    """
    espn_id = board_row.get("espn_id")
    if espn_id is not None and not pd.isna(espn_id):
        return int(espn_id)
    if board_row.get("position") == "DST":
        pro = ESPN_PRO_TEAM_BY_ABBREV.get(str(board_row.get("team", "")).upper())
        if pro is not None:
            return _dst_espn_id(pro)
    return None


    @app.post("/api/live/select")
    def live_select(body: SelectBody):
        """Make the pick.

        Writes nothing to `drafted`. The listener already records every
        SELECTED frame, including this one, so letting this endpoint write
        too would give the board two writers that can disagree -- and the
        one that must win is ESPN's own confirmation.
        """
        with lock:
            session = state["session"]
            socket = state["socket"]
            listener = state["listener"]
            active_conn = state["league_conn"] or conn
            if session is None:
                raise HTTPException(409, "no live draft session")
            if socket is None or not socket.alive():
                raise HTTPException(503, "the draft socket is not connected")
            cur = active_conn.cursor()
            try:
                picks_made = cur.execute(
                    "SELECT count(*) FROM drafted").fetchone()[0]
                already = cur.execute(
                    "SELECT count(*) FROM drafted WHERE player_id = ?",
                    [body.player_id]).fetchone()[0]
            finally:
                cur.close()

        if already:
            raise HTTPException(409, "that player is already drafted")
        if session.my_slot is None:
            raise HTTPException(409, "this session has no draft slot yet")
        if picks_until_turn(session.settings, session.my_slot, picks_made) != 0:
            raise HTTPException(409, "it is not your turn")

        row = session.board_by_id.get(body.player_id)
        if row is None:
            raise HTTPException(400, f"unknown player {body.player_id}")
        espn_id = _espn_id_for(row)
        if espn_id is None:
            raise HTTPException(
                400, f"no ESPN id for {row.get('name', body.player_id)} -- "
                "this is a crosswalk gap, pick him in ESPN directly")

        try:
            socket.send(f"SELECT {espn_id}\n")
        except (ConnectionError, OSError) as exc:
            raise HTTPException(503, f"could not reach ESPN: {exc}") from exc

        deadline = time.monotonic() + SELECT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if listener is not None and espn_id in listener.selected_espn_ids:
                return {"player_id": body.player_id, "espn_id": espn_id,
                        "pick_no": picks_made + 1}
            time.sleep(SELECT_POLL_SECONDS)
        raise HTTPException(
            504, "ESPN did not confirm the pick -- check the ESPN draft room "
            "before picking again")
```

Add the imports this needs at the top of `api/live.py`: `time`, `pandas as pd`, `HTTPException` from `fastapi`, and `from pipeline.espn_live import ESPN_PRO_TEAM_BY_ABBREV, _dst_espn_id`. Several are likely already imported — check before adding a duplicate.

`session.board_by_id` may not exist. If `DraftSession` has no board lookup, add one in `build_session` (`api/live.py:128`):

```python
        board_by_id={r["player_id"]: r for r in board.to_dict(orient="records")},
```

and the matching field on the `DraftSession` dataclass/NamedTuple.

- [ ] **Step 6: Run the select tests**

Run: `.venv/bin/pytest tests/test_live_api.py -k select -v`
Expected: PASS

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add api/live.py pipeline/draft_listener.py tests/test_live_api.py
git commit -m "feat: POST /api/live/select makes the pick on ESPN's socket"
```

---

### Task 6: Frontend API client

**Files:**
- Modify: `web/src/api.ts`

**Interfaces:**
- Consumes: the `/api/live/state` shape from Task 3, `/api/live/select` from Task 5.
- Produces:
  - `type LiveCandidate = { player_id: string; position: string; proj_points: number; vor_points: number; gain_now: number; survive_pct: number; fills: string; rank: number }` — replaces the existing `ev`/`se`/`applied_pct` shape.
  - `selectPlayer(playerId: string): Promise<{ player_id: string; espn_id: number; pick_no: number }>` — throws an `Error` whose message is the API's `detail`.

- [ ] **Step 1: Replace the candidate type**

In `web/src/api.ts`, replace the `LiveCandidate` type (around `web/src/api.ts:154`) with:

```typescript
// One row of the ranked available list. Sorted by `gain_now` descending on
// the server (scoring/gain.py) -- the value this pick gains over the best
// player at the same position expected to survive to your next pick,
// weighted by whether your roster can start him. `vor_points` is the raw
// value over replacement it is derived from; the two differ most exactly
// where the old EV ranking used to reach.
export type LiveCandidate = {
  player_id: string
  position: string
  proj_points: number
  vor_points: number
  gain_now: number
  survive_pct: number
  fills: string
  rank: number
}
```

- [ ] **Step 2: Add the select call**

Append to `web/src/api.ts`, matching the file's existing fetch/error style:

```typescript
export type SelectResult = { player_id: string; espn_id: number; pick_no: number }

// Makes the pick on ESPN. Resolves only once ESPN echoed it back, so a
// resolved promise means the pick is real -- there is no optimistic state
// anywhere above this.
export async function selectPlayer(playerId: string): Promise<SelectResult> {
  const res = await fetch(`${API}/api/live/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ player_id: playerId }),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    throw new Error(detail?.detail ?? `Pick failed (${res.status})`)
  }
  return res.json()
}
```

Use whatever base-URL constant the file already defines in place of `API`.

- [ ] **Step 3: Typecheck**

Run: `cd web && npm run build`
Expected: FAIL, in `LiveDraft.tsx` only — it reads `ev` and `applied_pct`, which no longer exist. That is expected; Task 7 replaces that file.

- [ ] **Step 4: Commit**

```bash
git add web/src/api.ts
git commit -m "feat: gain_now candidate type and selectPlayer in the api client"
```

---

### Task 7: Draft room shell — pinned clock and roster rail

`/draft` already routes to `web/src/pages/LiveDraft.tsx`. This task replaces that page with the new room. The design is published at https://claude.ai/code/artifact/ae269a33-666c-429c-b7ce-40018ba6d691 — match its layout, and take every color, size and radius from the existing tokens in `web/src/index.css` rather than from the mock's inlined copies.

This task builds the shell and the right rail. The main area renders a placeholder that Tasks 8 and 9 fill.

**Files:**
- Create: `web/src/pages/DraftRoom.tsx`
- Create: `web/src/components/draft/ClockPanel.tsx`
- Create: `web/src/components/draft/RosterPanel.tsx`
- Create: `web/src/components/draft/tone.ts`
- Delete: `web/src/pages/LiveDraft.tsx`
- Modify: `web/src/App.tsx` — `/draft` renders `DraftRoom`
- Modify: `web/src/App.css` — room styles

**Interfaces:**
- Consumes: `fetchLiveState`, `fetchPlayers`, `LiveState`, `LiveCandidate`, `Player` from `web/src/api.ts`.
- Produces:
  - `DraftRoom` default export.
  - `ClockPanel({ state, secondsLeft }: { state: LiveState; secondsLeft: number | null })`.
  - `RosterPanel({ slots }: { slots: RosterSlot[] })` where `type RosterSlot = { slot: string; player: Player | null; urgent: boolean }`, exported from `RosterPanel.tsx`.
  - `DraftRoom` owns the 2.5s poll of `/api/live/state` and the one-time `/api/players` join table, both lifted from the deleted `LiveDraft.tsx`.

- [ ] **Step 1: Read what is being replaced**

Read `web/src/pages/LiveDraft.tsx` in full. The polling effect, the `pollAgeLabel` helper, the stale handling and the players join table are all correct and move across unchanged. What does not move is the single-recommendation banner and `buildReason` — the reason text argued "take the one who won't last" while the candidate filter had already dropped exactly those players, and the ranked list replaces it.

- [ ] **Step 2: Build `ClockPanel`**

Create `web/src/components/draft/ClockPanel.tsx`. It renders, from `LiveState`:
- Whether you are on the clock (`on_the_clock === my_slot`), as the panel's heading and its accent treatment.
- The countdown, `mm:ss`, from a `secondsLeft` prop the parent ticks. When `secondsLeft` is null, render `--:--` rather than a zero.
- A progress bar of the remaining fraction.
- Three figures: this pick number, your next pick number, and the gap in picks.

Compute the next pick and gap client-side from `picks_made`, `my_slot`, and the league's team count using the snake rule: rounds alternate direction, so in round `r` (0-based) slot `s` picks at `r*teams + (r % 2 === 0 ? s : teams - s + 1)`.

When `state.listener_error` is set or `state.listener_alive` is false, the panel renders that message in place of the clock. Draft night's worst failure is a board that looks current and has stopped updating.

- [ ] **Step 3: Build `RosterPanel`**

Create `web/src/components/draft/RosterPanel.tsx`. One row per roster slot in order: QB, RB1, RB2, WR1, WR2, TE, FLEX1, FLEX2, K, DST, then bench. Filled rows show the position badge, name and projected points; open rows show `empty`, and rows the recommendation is treating as a need show `needs a starter` in accent with an accent-tinted row background.

Export `RosterSlot` from this file so Task 8 can reuse the type.

- [ ] **Step 4: Build `DraftRoom`**

Create `web/src/pages/DraftRoom.tsx` with the shell: a top bar, a body that is a flex row of the main column and a 340px rail, and a tab strip above the main area with `Available` and `Snake Board`. The rail holds `ClockPanel` then `RosterPanel` and never scrolls out of view; the page itself does not scroll.

Carry the poll effect and the players join over from `LiveDraft.tsx`. Add a one-second ticker for the clock display, the same pattern `LiveDraft.tsx` already uses for `nowMs`.

For this task the main area renders `<div className="draft-main-placeholder">Available list — Task 8</div>` under both tabs.

- [ ] **Step 5: Rescue the two helpers worth keeping, then swap the route**

`LiveDraft.tsx` carries two helpers the new room still needs. Move them before deleting the file:

- `pollAgeLabel` (`web/src/pages/LiveDraft.tsx:13`) — seconds-resolution staleness, deliberately distinct from `api.ts`'s minute-rounding `ageLabel`. Move it into `ClockPanel.tsx`.
- `riskTone` (`web/src/pages/LiveDraft.tsx:37`) — the red/amber/green survival ramp. Move it into a new `web/src/components/draft/tone.ts` and export it, so Task 8's `AvailableList` and `TopThree` both import the one copy.

Then in `web/src/App.tsx`, replace the `LiveDraft` import and its `/draft` element with `DraftRoom`, and update the file's leading comment to describe the room rather than the board. Delete `web/src/pages/LiveDraft.tsx`.

- [ ] **Step 6: Add the styles**

Add the room's classes to `web/src/App.css`, below the existing live-board rules. Use the existing custom properties for every value. Do not add a second copy of the token block; `web/src/index.css` already defines it.

- [ ] **Step 7: Typecheck and lint**

Run: `cd web && npm run build && npm run lint`
Expected: PASS

- [ ] **Step 8: Look at it**

Run: `make up`, open http://localhost:5173/draft
Expected: the shell renders, the rail is pinned, and with no live session the clock panel says so rather than showing a fabricated countdown.

- [ ] **Step 9: Commit**

```bash
git add web/src/pages/DraftRoom.tsx web/src/components/draft web/src/App.tsx web/src/App.css
git rm web/src/pages/LiveDraft.tsx
git commit -m "feat: draft room shell with a pinned clock and roster rail"
```

---

### Task 8: Available list, top three, and the confirm dialog

**Files:**
- Create: `web/src/components/draft/AvailableList.tsx`
- Create: `web/src/components/draft/TopThree.tsx`
- Create: `web/src/components/draft/ConfirmPick.tsx`
- Modify: `web/src/pages/DraftRoom.tsx` — render them
- Modify: `web/src/App.css`

**Interfaces:**
- Consumes: `LiveCandidate`, `Player`, `selectPlayer` from Task 6; `RosterSlot` from Task 7.
- Produces:
  - `AvailableList({ candidates, players, onDraft }: { candidates: LiveCandidate[]; players: Record<string, Player>; onDraft: (c: LiveCandidate) => void })`.
  - `TopThree({ candidates, players, onDraft })` — same props, renders `candidates.slice(0, 3)`.
  - `ConfirmPick({ candidate, player, status, error, onConfirm, onCancel })` where `status: 'idle' | 'sending' | 'done' | 'failed'`.

- [ ] **Step 1: Build `AvailableList`**

A table with columns: rank, position badge, player (name plus team and bye), `proj_points`, `vor_points`, `gain_now`, `survive_pct`, ADP (`player.market_rank`), `fills`, and a draft button.

Rules that carry meaning and must not be dropped:
- `survive_pct` is colored on a red → amber → green ramp. Reuse `riskTone` from the deleted `LiveDraft.tsx:37` — it is the same three-stop `color-mix` idea and the same semantics. Move it into this file rather than reimplementing it.
- `gain_now` is the sort column and reads at full emphasis; `vor_points` reads at secondary emphasis. The visual hierarchy is the argument: the big raw number is not the one to act on.
- `fills` renders in accent when it names an open starter slot, and muted for `BENCH` or `—`.
- The draft button is disabled unless it is your turn.

Add a search input and position filter pills above the table. Both filter client-side; the server sends the whole ranked list.

- [ ] **Step 2: Build `TopThree`**

Three cards above the table. Each shows rank, position, name, team, a draft button, then `gain_now`, `survive_pct` and `fills` as three figures, then one sentence of reasoning built from those numbers — the fills slot, the gain, and the survival percentage. The first card carries an accent border.

Write the sentence from the data, not from a template with a fixed claim: it must say what the numbers say for that row, including when the honest reading is that waiting is fine.

- [ ] **Step 3: Build `ConfirmPick`**

A modal over the room. Shows the player, what he fills, the gain, and the roster count after. Two buttons: cancel, and confirm. On confirm the parent calls `selectPlayer`.

The status prop drives the body: `sending` disables both buttons and says the pick is with ESPN; `done` closes the modal; `failed` shows `error` verbatim and leaves cancel enabled. A 504 must read as "we could not confirm it" and tell the user to check ESPN, not as "it failed" — the API's own message already says this, so render it rather than substituting one.

- [ ] **Step 4: Wire them into `DraftRoom`**

Replace the Available placeholder with `TopThree` then `AvailableList`, both fed from `state.candidates`. Hold `confirming: LiveCandidate | null` and the status in `DraftRoom`. On a successful select, clear the modal and let the next poll bring the pick back from the server. Do not mutate the local candidate list.

- [ ] **Step 5: Typecheck and lint**

Run: `cd web && npm run build && npm run lint`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add web/src/components/draft web/src/pages/DraftRoom.tsx web/src/App.css
git commit -m "feat: ranked available list, top three, and pick confirmation"
```

---

### Task 9: Snake board tab and the turn switch

**Files:**
- Modify: `web/src/pages/DraftRoom.tsx`
- Modify: `web/src/components/DraftBoardGrid.tsx` if the room needs a prop it does not take
- Modify: `web/src/App.css`

**Interfaces:**
- Consumes: `fetchBoard`, `LiveBoard` from `web/src/api.ts`; `DraftBoardGrid` as it exists.
- Produces: no new exports. `DraftRoom` gains `tab: 'available' | 'board'` state and the auto-switch.

- [ ] **Step 1: Render the board under its tab**

Carry the `/api/live/board` poll over from the deleted `LiveDraft.tsx` — it already has its own error slot so a hiccup in one endpoint does not blank the other, and that stays. Render `DraftBoardGrid` under the Snake Board tab. Give the grid its own scroll container so the page still does not scroll.

- [ ] **Step 2: Auto-switch on the clock**

When `state.on_the_clock` becomes `state.my_slot` and the current tab is `board`, switch to `available`.

Implement this as an edge, not a condition: track the previous on-the-clock slot and switch only on the transition into your turn. A plain `if (onClock) setTab('available')` would fight the user every render if they deliberately switched to the board while on the clock.

- [ ] **Step 3: Typecheck and lint**

Run: `cd web && npm run build && npm run lint`
Expected: PASS

- [ ] **Step 4: Run the whole suite once more**

Run: `.venv/bin/pytest -q && cd web && npm run build`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/src/pages/DraftRoom.tsx web/src/components/DraftBoardGrid.tsx web/src/App.css
git commit -m "feat: snake board tab, and switch to available when you're on the clock"
```

---

## Verification before calling this done

- [ ] `.venv/bin/pytest -q` passes.
- [ ] `cd web && npm run build && npm run lint` passes.
- [ ] The real board no longer ranks Mark Andrews, Travis Kelce or a defense inside the top 30 (Task 1 Step 14).
- [ ] `/api/live/state` candidates are sorted by `gain_now` and carry no `ev` field.
- [ ] A `SELECT` is sent only when it is your turn, and `drafted` is written only by the listener.
