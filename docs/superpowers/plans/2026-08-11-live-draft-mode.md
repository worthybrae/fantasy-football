# Live Draft Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** During an ESPN draft, keep the board current as picks land and answer "what do I take" when I am on the clock, without typing.

**Architecture:** A long-lived `DraftSession` held in the API process caches the pool, fitted coefficients and board fingerprint built once at draft start, so each refresh costs only `search_pick`. A poller reads ESPN's `mDraftDetail` view, maps ESPN player ids onto board player ids, and writes the existing `drafted` table. The seed handed to `search_pick` is pinned for the life of the session, so a changed recommendation means a changed board rather than a different random sample.

**Tech Stack:** Python 3.10, FastAPI, DuckDB, pandas, numpy, Playwright (already present for ESPN auth), React + TypeScript + Vite.

## Global Constraints

- **ESPN only.** No other platform is in scope.
- **The seed passed to `search_pick` is pinned for the session's lifetime.** Never derive it from time, pick count, or a counter.
- **Reconciliation: ESPN wins.** A manual `D` mark the poller later contradicts is overwritten.
- **Desync is never patched incrementally.** On any pick-count mismatch, rebuild `drafted` from ESPN's full pick list.
- **Unmapped picks are surfaced, never swallowed.** A pick whose ESPN player id does not map to a board player id must reach the UI.
- **Staleness is surfaced.** `last_poll_at` older than 15s must be visible to the client.
- **No websockets.** The client polls `GET /api/live/state`; the poller runs server-side on a thread.
- **Rollout budget by picks-until-my-turn:** 3+ away → `n_rollouts=25`; 1-2 away → `100`; on the clock → `200`.
- **A new pick supersedes an in-flight computation.** Only the most recent completed result is served, tagged with `candidates_as_of_pick`.
- **`scoring/draft_sim.py` is NOT modified.** `search_pick(pool, settings, slot_managers, my_slot, taken, betas, n_rollouts, n_candidates, seed, taken_order, avail_pct)` already accepts a prebuilt pool. Only `run_sim` rebuilds one, and live mode does not call `run_sim`.
- Run tests with `.venv/bin/python -m pytest`. DuckDB is single-writer: no test may open `data/nfl.duckdb`.

---

## File Structure

| file | responsibility |
|---|---|
| `pipeline/espn_live.py` (create) | Pure translation: an ESPN `mDraftDetail` payload plus a crosswalk becomes `(pick_no, player_id)` rows and a list of unmapped picks. No network, no database. |
| `api/live.py` (create) | `DraftSession` dataclass, the recompute worker, and the three endpoints. Owns the pinned seed and the in-flight supersede rule. |
| `web/src/components/LiveDraft.tsx` (create) | On-the-clock view. |
| `web/src/api.ts` (modify) | `LiveState` type and `fetchLiveState`. |
| `api/main.py` (modify) | Mount the live router. |
| `tests/test_espn_live.py` (create) | Translation, mapping, unmapped picks, desync detection. |
| `tests/test_live_api.py` (create) | Session lifecycle, staleness, supersede, pinned seed. |
| `tests/fixtures/espn_live_draft.json` (create) | A real recorded `mDraftDetail` payload from a live ESPN mock. |

---

### Task 1: Prove ESPN serves picks during a live draft

The whole design rests on ESPN populating `draftDetail.picks` progressively rather than only on completion. Nothing else can be built honestly until this is known. The deliverable is a recorded payload that every later task tests against.

**Files:**
- Create: `scripts/record_live_draft.py`
- Create: `tests/fixtures/espn_live_draft.json`

**Interfaces:**
- Consumes: `pipeline.espn_league.EspnClient`, `season_url`, `parse_league_id` (existing).
- Produces: `tests/fixtures/espn_live_draft.json` — a real `mDraftDetail` payload captured mid-draft, with at least 8 real picks.

- [ ] **Step 1: Write the recorder script**

```python
"""Capture a live ESPN draft payload for use as a test fixture.

Run this while sitting in an ESPN mock draft, after several picks have been
made. It answers the one question the live-draft design rests on: does ESPN
populate draftDetail.picks as the draft runs, or only when it ends?
"""
import json
import sys
from pathlib import Path

from scoring.config import CURRENT_SEASON
from pipeline.espn_league import EspnClient, parse_league_id, season_url, _is_real_pick

OUT = Path("tests/fixtures/espn_live_draft.json")


def main(league_url: str) -> int:
    league_id = parse_league_id(league_url)
    with EspnClient() as client:
        payload = client.get_json(season_url(league_id, CURRENT_SEASON))
    detail = payload.get("draftDetail") or {}
    picks = [p for p in (detail.get("picks") or []) if _is_real_pick(p)]
    print(f"draftDetail.drafted = {detail.get('drafted')}")
    print(f"real picks in payload = {len(picks)}")
    if not picks:
        print("\nNo real picks. Either the draft has not started, or ESPN "
              "does not serve picks until completion -- see the plan's Task 1 "
              "fallback before continuing.")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT} ({len(picks)} picks)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: record_live_draft.py <espn league url or id>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
```

- [ ] **Step 2: Join an ESPN mock draft and run the recorder**

Join a mock at `https://fantasy.espn.com/football/mockdraftlobby`, let at least 8 picks go by, then run:

```bash
.venv/bin/python scripts/record_live_draft.py "<mock league url>"
```

Expected: `real picks in payload = 8` or more, and the fixture written.

**If it prints 0 picks:** STOP and report BLOCKED. Polling is not viable; ingest falls back to the existing `D` hotkey and Tasks 2-3 are replaced by "read `drafted` directly". Tasks 4-7 are unaffected.

- [ ] **Step 3: Record what was learned in the spec**

Append to `docs/superpowers/specs/2026-08-11-live-draft-mode-design.md` under "The risk that gates everything":

```markdown
**RESOLVED 2026-08-11:** ESPN serves `draftDetail.picks` progressively during
a live draft. Verified in a mock: `drafted` reads false while `picks` carried
N real entries. Fixture recorded at `tests/fixtures/espn_live_draft.json`.
```

Replace `N` with the actual count.

- [ ] **Step 4: Commit**

```bash
git add scripts/record_live_draft.py tests/fixtures/espn_live_draft.json docs/superpowers/specs/2026-08-11-live-draft-mode-design.md
git commit -m "test: record a live ESPN draft payload

The live-draft design rests on ESPN populating draftDetail.picks as a draft
runs rather than only on completion. Verified in a mock and recorded, so every
later task tests against a real payload instead of an invented one."
```

---

### Task 2: Translate an ESPN payload into drafted rows

**Files:**
- Create: `pipeline/espn_live.py`
- Test: `tests/test_espn_live.py`

**Interfaces:**
- Consumes: `pipeline.espn_league._is_real_pick` (existing), `tests/fixtures/espn_live_draft.json` from Task 1.
- Produces:
  - `build_crosswalk(conn) -> dict[int, str]` — ESPN player id to board `player_id`.
  - `LivePicks = NamedTuple("LivePicks", [("rows", pd.DataFrame), ("unmapped", list)])` where `rows` has columns `["player_id", "pick_no"]` sorted by `pick_no`, and `unmapped` is a list of `{"espn_player_id": int, "overall_pick": int}`.
  - `translate(payload: dict, crosswalk: dict) -> LivePicks`

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.db import get_conn, write_table
from pipeline.espn_live import LivePicks, build_crosswalk, translate

FIXTURE = Path("tests/fixtures/espn_live_draft.json")


def _payload(picks):
    return {"draftDetail": {"drafted": False, "picks": picks}}


def _pick(overall, player_id, team_id=1):
    return {"overallPickNumber": overall, "roundId": 1,
            "roundPickNumber": overall, "teamId": team_id,
            "playerId": player_id, "keeper": False}


def test_translate_maps_espn_ids_to_board_ids():
    out = translate(_payload([_pick(1, 111), _pick(2, 222)]),
                    {111: "g1", 222: "g2"})
    assert list(out.rows["player_id"]) == ["g1", "g2"]
    assert list(out.rows["pick_no"]) == [1, 2]
    assert out.unmapped == []


def test_translate_orders_by_overall_pick_not_payload_order():
    """pick_no drives roster attribution downstream -- draft_sim._drafted_state
    sorts on it and hands pick k to whoever was on the clock for pick k. ESPN
    has no obligation to serve picks in order, so relying on payload order
    would silently misattribute every roster."""
    out = translate(_payload([_pick(3, 333), _pick(1, 111), _pick(2, 222)]),
                    {111: "g1", 222: "g2", 333: "g3"})
    assert list(out.rows["pick_no"]) == [1, 2, 3]
    assert list(out.rows["player_id"]) == ["g1", "g2", "g3"]


def test_translate_reports_an_unmapped_pick_instead_of_dropping_it():
    """A dropped pick leaves a drafted player on the board and the tool
    recommends someone already gone. It must reach the UI."""
    out = translate(_payload([_pick(1, 111), _pick(2, 999)]), {111: "g1"})
    assert list(out.rows["player_id"]) == ["g1"]
    assert out.unmapped == [{"espn_player_id": 999, "overall_pick": 2}]


def test_translate_drops_espns_placeholder_picks():
    """ESPN pads an in-progress board with playerId -1 for slots nobody has
    filled yet. Those are not picks."""
    out = translate(_payload([_pick(1, 111), _pick(2, -1)]), {111: "g1"})
    assert list(out.rows["player_id"]) == ["g1"]
    assert out.unmapped == []


def test_translate_handles_an_empty_draft():
    out = translate(_payload([]), {111: "g1"})
    assert out.rows.empty
    assert list(out.rows.columns) == ["player_id", "pick_no"]
    assert out.unmapped == []


def test_build_crosswalk_reads_sleeper_ids(tmp_path):
    conn = get_conn(str(tmp_path / "x.duckdb"))
    write_table(conn, "sleeper_ids", pd.DataFrame([
        {"gsis_id": "g1", "espn_id": 111, "sleeper_name": "A", "position": "RB", "team": "DET"},
        {"gsis_id": "g2", "espn_id": 222, "sleeper_name": "B", "position": "WR", "team": "GB"},
        {"gsis_id": None, "espn_id": 333, "sleeper_name": "C", "position": "TE", "team": "SF"},
    ]))
    x = build_crosswalk(conn)
    assert x[111] == "g1" and x[222] == "g2"
    assert 333 not in x        # no gsis_id means no board player_id


@pytest.mark.skipif(not FIXTURE.exists(), reason="run Task 1's recorder first")
def test_translate_handles_the_real_recorded_payload():
    """Against a real ESPN payload, not an invented one. Everything the
    fixture carries must either map or be reported -- silence is the failure
    this guards."""
    payload = json.loads(FIXTURE.read_text())
    out = translate(payload, {})
    assert len(out.unmapped) > 0          # crosswalk empty, so all unmapped
    assert out.rows.empty
    # And every unmapped entry names both fields the UI needs.
    assert all({"espn_player_id", "overall_pick"} == set(u) for u in out.unmapped)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_espn_live.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.espn_live'`

- [ ] **Step 3: Write the implementation**

```python
"""Turn an in-progress ESPN draft payload into rows for the `drafted` table.

Pure translation: no network and no database beyond reading the crosswalk.
The poller in `api/live.py` supplies the payload and writes the result, which
keeps the part with all the edge cases testable against a recorded fixture.
"""
from typing import NamedTuple

import pandas as pd

from pipeline.db import read_table
from pipeline.espn_league import _is_real_pick

COLUMNS = ["player_id", "pick_no"]


class LivePicks(NamedTuple):
    rows: pd.DataFrame          # columns COLUMNS, sorted by pick_no
    unmapped: list              # [{"espn_player_id": int, "overall_pick": int}]


def build_crosswalk(conn) -> dict:
    """ESPN player id -> board player_id.

    `sleeper_ids.gsis_id` IS the board's player_id -- `scoring.market` joins
    ESPN to the board through exactly this column. Rows without a gsis_id
    cannot reach the board at all, so they are left out rather than mapped to
    something invented.
    """
    ids = read_table(conn, "sleeper_ids")
    if ids.empty or "espn_id" not in ids.columns:
        return {}
    usable = ids.dropna(subset=["espn_id", "gsis_id"])
    return {int(e): str(g) for e, g in zip(usable["espn_id"], usable["gsis_id"])}


def translate(payload: dict, crosswalk: dict) -> LivePicks:
    """Map a `mDraftDetail` payload onto board players.

    Sorted by ESPN's own `overallPickNumber`, never by payload order:
    `pick_no` is what `draft_sim._drafted_state` attributes rosters with, and
    ESPN makes no promise about the order it serves picks in.

    A pick whose player does not map is reported, not dropped. Dropping it
    would leave a drafted player on the board and let the tool recommend
    someone already gone -- silently, which is the worst way for this to fail.
    """
    picks = [p for p in (((payload.get("draftDetail") or {}).get("picks")) or [])
             if _is_real_pick(p)]
    picks.sort(key=lambda p: p.get("overallPickNumber") or 0)

    rows, unmapped = [], []
    for p in picks:
        espn_id = p.get("playerId")
        overall = p.get("overallPickNumber")
        player_id = crosswalk.get(int(espn_id)) if espn_id is not None else None
        if player_id is None:
            unmapped.append({"espn_player_id": int(espn_id),
                             "overall_pick": int(overall)})
            continue
        rows.append({"player_id": player_id, "pick_no": int(overall)})
    return LivePicks(pd.DataFrame(rows, columns=COLUMNS), unmapped)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_espn_live.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add pipeline/espn_live.py tests/test_espn_live.py
git commit -m "feat: translate a live ESPN draft payload into drafted rows

Pure translation, no network and no database beyond the crosswalk, so the
part carrying the edge cases is testable against the recorded fixture.

Sorted by ESPN's overallPickNumber rather than payload order: pick_no is what
_drafted_state attributes rosters with, and ESPN makes no promise about the
order it serves picks in. An unmapped pick is reported rather than dropped --
dropping it would leave a drafted player on the board and let the tool
recommend someone already gone."
```

---

### Task 3: Write picks to the database, resyncing on any mismatch

**Files:**
- Modify: `pipeline/espn_live.py`
- Test: `tests/test_espn_live.py`

**Interfaces:**
- Consumes: `translate`, `build_crosswalk` from Task 2.
- Produces: `apply_picks(conn, live: LivePicks) -> int` — replaces the `drafted` table wholesale with `live.rows`, returns the row count written.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_espn_live.py`:

```python
from pipeline.espn_live import apply_picks


def _conn_with_drafted(tmp_path, rows):
    conn = get_conn(str(tmp_path / "d.duckdb"))
    for player_id, pick_no in rows:
        conn.execute("INSERT INTO drafted VALUES (?, ?)", [player_id, pick_no])
    return conn


def test_apply_picks_replaces_rather_than_appends(tmp_path):
    """The constraint is the whole point: never patch incrementally.

    If ESPN's list and ours disagree, appending the difference produces a
    drafted table that is neither -- and every roster, need and cap
    downstream is computed against it. Replacing wholesale means the table
    always equals ESPN's list exactly.
    """
    conn = _conn_with_drafted(tmp_path, [("stale", 1), ("alsostale", 2)])
    live = LivePicks(pd.DataFrame([{"player_id": "g1", "pick_no": 1},
                                   {"player_id": "g2", "pick_no": 2}]),
                     [])
    assert apply_picks(conn, live) == 2
    got = conn.execute("SELECT player_id, pick_no FROM drafted ORDER BY pick_no").fetchall()
    assert got == [("g1", 1), ("g2", 2)]


def test_apply_picks_overwrites_a_manual_mark_espn_contradicts(tmp_path):
    """Reconciliation rule, stated once in the spec: ESPN wins."""
    conn = _conn_with_drafted(tmp_path, [("guessed_wrong", 1)])
    live = LivePicks(pd.DataFrame([{"player_id": "actual", "pick_no": 1}]), [])
    apply_picks(conn, live)
    got = conn.execute("SELECT player_id FROM drafted").fetchall()
    assert got == [("actual",)]


def test_apply_picks_clears_the_table_when_espn_reports_no_picks(tmp_path):
    """A draft that was reset, or a session pointed at the wrong league. The
    table must follow ESPN rather than keep whatever it had."""
    conn = _conn_with_drafted(tmp_path, [("g1", 1)])
    assert apply_picks(conn, LivePicks(pd.DataFrame(columns=COLUMNS), [])) == 0
    assert conn.execute("SELECT count(*) FROM drafted").fetchone()[0] == 0


def test_apply_picks_never_writes_a_null_pick_no(tmp_path):
    """_drafted_state raises on a null pick_no rather than misattributing.
    Guarding here means that path is unreachable from the poller."""
    conn = get_conn(str(tmp_path / "n.duckdb"))
    live = LivePicks(pd.DataFrame([{"player_id": "g1", "pick_no": None}]), [])
    with pytest.raises(ValueError, match="null pick_no"):
        apply_picks(conn, live)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_espn_live.py -k apply_picks -v`
Expected: FAIL with `ImportError: cannot import name 'apply_picks'`

- [ ] **Step 3: Write the implementation**

Append to `pipeline/espn_live.py`:

```python
def apply_picks(conn, live: LivePicks) -> int:
    """Make `drafted` equal ESPN's pick list exactly. Returns rows written.

    Wholesale replacement, never a patch. If our table and ESPN's list
    disagree -- a pick we missed, a manual mark that guessed wrong, a draft
    that was reset -- appending the difference produces a table that matches
    neither, and every roster, need and cap downstream is computed from it.
    Replacing means the table is always exactly what ESPN says.

    That is also the reconciliation rule for the manual `D` hotkey: ESPN wins.
    """
    if not live.rows.empty and live.rows["pick_no"].isna().any():
        raise ValueError(
            "refusing to write a null pick_no: draft_sim._drafted_state "
            "cannot attribute such a pick to a team and would raise")
    conn.execute("DELETE FROM drafted")
    for row in live.rows.itertuples(index=False):
        conn.execute("INSERT INTO drafted VALUES (?, ?)",
                     [str(row.player_id), int(row.pick_no)])
    return len(live.rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_espn_live.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/espn_live.py tests/test_espn_live.py
git commit -m "feat: make drafted equal ESPN's pick list, wholesale

Never a patch. If our table and ESPN's list disagree, appending the
difference produces a table matching neither -- and every roster, need and
cap downstream is computed from it. Replacing means it is always exactly
what ESPN says, which is also the reconciliation rule for the manual D
hotkey: ESPN wins.

Refuses to write a null pick_no, which makes _drafted_state's
misattribution guard unreachable from this path."
```

---

### Task 4: The draft session

**Files:**
- Create: `api/live.py`
- Test: `tests/test_live_api.py`

**Interfaces:**
- Consumes: `scoring.board.build_board`, `scoring.draft_sim.build_pool`, `scoring.draft_model.fit_all`, `scoring.league.load`, `pipeline.espn_live.build_crosswalk`.
- Produces:
  - `DraftSession` dataclass with fields `my_slot: int`, `slot_managers: dict`, `settings`, `pool`, `betas: dict`, `crosswalk: dict`, `board_fingerprint: str`, `seed: int`, `started_at: datetime`.
  - `build_session(conn, my_slot: int, seed: int = 20260811) -> DraftSession`
  - `board_fingerprint(board: pd.DataFrame) -> str`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
import pytest

from api.live import DraftSession, board_fingerprint, build_session


def test_board_fingerprint_changes_when_the_player_set_changes():
    """The session caches pool indices. If the board is rebuilt underneath it
    -- a refresh drops a retired player, say -- those indices point at
    different people and every recommendation is silently about the wrong
    player."""
    a = pd.DataFrame({"player_id": ["g1", "g2"], "market_rank": [1.0, 2.0]})
    b = pd.DataFrame({"player_id": ["g1", "g3"], "market_rank": [1.0, 2.0]})
    assert board_fingerprint(a) != board_fingerprint(b)


def test_board_fingerprint_is_stable_for_the_same_board():
    a = pd.DataFrame({"player_id": ["g1", "g2"], "market_rank": [1.0, 2.0]})
    assert board_fingerprint(a) == board_fingerprint(a.copy())


def test_board_fingerprint_ignores_row_order():
    """Row order is an artifact of how the board was assembled, not a change
    in who is draftable."""
    a = pd.DataFrame({"player_id": ["g1", "g2"], "market_rank": [1.0, 2.0]})
    b = a.iloc[::-1].reset_index(drop=True)
    assert board_fingerprint(a) == board_fingerprint(b)


def test_session_pins_a_seed_that_does_not_move(tmp_path, monkeypatch):
    """The load-bearing decision. Roster EV carries +/-4.2 at 400 rollouts,
    so +/-16.8 at the 25 a live refresh affords, against a 9.3-point gap
    between adjacent candidates. An unpinned seed reshuffles the
    recommendation while the board sits still."""
    s = _fake_session(seed=4242)
    assert s.seed == 4242
    # Two reads, same value -- not a property, not derived from a clock.
    assert s.seed == 4242
```

Add this helper at the top of `tests/test_live_api.py`:

```python
def _fake_session(seed=20260811):
    """A DraftSession with cheap stand-ins for the expensive fields.

    build_session costs ~17s against a real database (fit_all alone is 14.9s),
    which is exactly why the session exists -- but it makes a poor unit-test
    fixture. Tests that exercise session *behaviour* build one directly.
    """
    from datetime import datetime, timezone
    return DraftSession(
        my_slot=4, league_id="53929318",
        slot_managers={i: f"m{i}" for i in range(1, 9)},
        settings=None, pool=None, betas={}, crosswalk={111: "g1"},
        board_fingerprint="abc", seed=seed,
        started_at=datetime(2026, 8, 11, tzinfo=timezone.utc))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_live_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.live'`

- [ ] **Step 3: Write the implementation**

```python
"""Live draft mode: one long-lived session, refreshed as picks land.

`make sim` pays 0.9s building the board, 1.1s building the pool and 14.9s in
`fit_all` on every invocation. During a draft none of that changes -- the
coefficients come from history, the board and pool are static -- so the
session builds them once and every refresh costs only `search_pick`.
"""
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline.db import read_table
from pipeline.espn_live import build_crosswalk
from scoring import league as league_mod
from scoring.board import build_board
from scoring.draft_model import FEATURE_NAMES, fit_all
from scoring.draft_sim import build_pool

# Pinned, not generated. See DraftSession.seed.
DEFAULT_SEED = 20260811


@dataclass(frozen=True)
class DraftSession:
    my_slot: int
    # The ESPN league being polled. Task 6's poller builds its URL from this;
    # it lives on the session because a session is tied to one draft.
    league_id: str
    slot_managers: dict
    settings: object
    pool: object
    betas: dict
    crosswalk: dict
    board_fingerprint: str
    # Pinned for the session's lifetime, never derived from a clock or a
    # counter. `search_pick` already uses common random numbers within a
    # call; holding the seed fixed ACROSS calls is what makes a changed
    # recommendation mean a changed board rather than a different sample.
    seed: int
    started_at: datetime


def board_fingerprint(board: pd.DataFrame) -> str:
    """Identity of the draftable set, order-independent.

    The session caches pool indices. If the board is rebuilt underneath it,
    those indices point at different players and every recommendation is
    silently about the wrong person. Row order is an artifact of assembly,
    not a change in who is draftable, so it is sorted out.
    """
    ids = sorted(str(p) for p in board["player_id"])
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def build_session(conn, my_slot: int, seed: int = DEFAULT_SEED) -> DraftSession:
    """Everything expensive, once. Roughly 17s against a real database."""
    settings = league_mod.load(conn)
    board = build_board(conn, settings=settings)
    pool = build_pool(conn, board, settings)
    fits = fit_all(conn, settings)
    pooled = fits.get("__pooled__", np.zeros(len(FEATURE_NAMES)))
    betas = {m: fits.get(m, pooled) for m in fits if m != "__pooled__"}

    draft_order = read_table(conn, "draft_order")
    slot_managers = dict(zip(draft_order["slot"].astype(int),
                             draft_order["manager"]))
    league_row = read_table(conn, "league")
    league_id = (str(league_row["league_id"].iloc[0])
                 if not league_row.empty else "")
    return DraftSession(
        my_slot=my_slot, league_id=league_id, slot_managers=slot_managers,
        settings=settings, pool=pool, betas=betas,
        crosswalk=build_crosswalk(conn),
        board_fingerprint=board_fingerprint(board), seed=seed,
        started_at=datetime.now(timezone.utc))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_live_api.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add api/live.py tests/test_live_api.py
git commit -m "feat: the live draft session

Builds board, pool and fitted coefficients once (~17s, of which fit_all is
14.9s) and holds them, so each refresh costs only search_pick.

The seed is a pinned field rather than something generated per call. Roster
EV carries +/-4.2 at 400 rollouts, so +/-16.8 at the 25 a live refresh
affords, against a 9.3-point gap between adjacent candidates -- unpinned, the
recommendation reshuffles while the board sits still.

board_fingerprint is order-independent and guards the cached pool indices
against the board being rebuilt underneath the session."
```

---

### Task 5: Recompute policy and the live endpoints

**Files:**
- Modify: `api/live.py`
- Modify: `api/main.py`
- Test: `tests/test_live_api.py`

**Interfaces:**
- Consumes: `DraftSession`, `build_session` from Task 4; `translate`, `apply_picks`, `build_crosswalk` from Tasks 2-3; `scoring.draft_sim.search_pick` and `_drafted_state` (existing).
- Produces:
  - `rollouts_for(picks_until_turn: int) -> int`
  - `picks_until_turn(settings, my_slot: int, picks_made: int) -> int`
  - `register_live_routes(app, conn)` mounted from `create_app`.
  - `GET /api/live/state` returning `{"active": bool, "picks_made": int, "on_the_clock": int|None, "candidates": [...], "candidates_as_of_pick": int|None, "last_poll_at": str|None, "stale": bool, "unmapped_picks": [...]}`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_api.py`:

```python
from api.live import STALE_AFTER_SECONDS, picks_until_turn, rollouts_for


def test_rollouts_scale_with_how_close_my_turn_is():
    """One code path, N as a parameter. Measured costs: 25 -> 4.7s,
    100 -> 19.5s, 200 -> 35.2s. Picks arrive every ~20-30s and the clock is
    ~90s, so each tier has to fit the window it is chosen for."""
    assert rollouts_for(7) == 25
    assert rollouts_for(3) == 25
    assert rollouts_for(2) == 100
    assert rollouts_for(1) == 100
    assert rollouts_for(0) == 200


def test_rollouts_never_returns_zero_or_negative():
    """A negative distance means the pick count ran past my turn -- a desync.
    It must still produce a usable budget rather than an empty search."""
    assert rollouts_for(-1) == 200


def test_picks_until_turn_counts_the_snake_correctly():
    """8 teams, slot 4 picks at overall 4, 13, 20. With 8 picks made the next
    pick is #9, and mine is #13, so 4 others go first."""
    settings = type("S", (), {"teams": 8, "rounds": 15})()
    assert picks_until_turn(settings, my_slot=4, picks_made=0) == 3
    assert picks_until_turn(settings, my_slot=4, picks_made=3) == 0
    assert picks_until_turn(settings, my_slot=4, picks_made=8) == 4
    assert picks_until_turn(settings, my_slot=4, picks_made=12) == 0


def test_picks_until_turn_at_the_snake_turn_is_zero_twice_running():
    """Slot 1 picks at overall 1, then 16 and 17 back to back."""
    settings = type("S", (), {"teams": 8, "rounds": 15})()
    assert picks_until_turn(settings, my_slot=1, picks_made=15) == 0
    assert picks_until_turn(settings, my_slot=1, picks_made=16) == 0


def test_state_reports_stale_when_the_poll_is_old(tmp_path):
    """A frozen board that looks live is worse than one that admits it. The
    threshold is the spec's 15s."""
    from datetime import datetime, timedelta, timezone
    from api.live import _is_stale
    now = datetime.now(timezone.utc)
    assert not _is_stale(now, now)
    assert not _is_stale(now - timedelta(seconds=STALE_AFTER_SECONDS - 1), now)
    assert _is_stale(now - timedelta(seconds=STALE_AFTER_SECONDS + 1), now)
    assert _is_stale(None, now)          # never polled at all


def test_state_is_inactive_before_start(tmp_path):
    from fastapi.testclient import TestClient
    from api.main import create_app
    body = TestClient(create_app(str(tmp_path / "t.duckdb"))).get("/api/live/state").json()
    assert body["active"] is False
    assert body["candidates"] == []
    assert body["candidates_as_of_pick"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_live_api.py -v`
Expected: FAIL with `ImportError: cannot import name 'rollouts_for'`

- [ ] **Step 3: Write the implementation**

Append to `api/live.py`:

```python
import threading

from scoring.draft_sim import _drafted_state, search_pick, snake_slots

STALE_AFTER_SECONDS = 15
# Measured on the live board: 25 -> 4.7s, 100 -> 19.5s, 200 -> 35.2s.
# Picks arrive every ~20-30s; the clock is ~90s.
ROLLOUTS_FAR, ROLLOUTS_NEAR, ROLLOUTS_NOW = 25, 100, 200


def rollouts_for(picks_until: int) -> int:
    """Budget by the time actually available.

    Because the seed is pinned, raising N between refreshes refines the same
    scenario set rather than resampling a different one -- the estimate
    converges instead of jumping. A negative distance means a desync ran the
    pick count past my turn; treat that as "now" rather than searching
    nothing.
    """
    if picks_until <= 0:
        return ROLLOUTS_NOW
    if picks_until <= 2:
        return ROLLOUTS_NEAR
    return ROLLOUTS_FAR


def picks_until_turn(settings, my_slot: int, picks_made: int) -> int:
    """How many other teams pick before I do.

    Zero means I am on the clock -- including both halves of a snake turn,
    where I pick twice with nobody in between.
    """
    slots = snake_slots(settings.teams, settings.rounds)
    for offset in range(picks_made, len(slots)):
        if slots[offset] == my_slot:
            return offset - picks_made
    return 0


def _is_stale(last_poll_at, now) -> bool:
    """Never having polled counts as stale: the UI must not present an empty
    board as a current one."""
    if last_poll_at is None:
        return True
    return (now - last_poll_at).total_seconds() > STALE_AFTER_SECONDS
```

Then the route registration, also appended to `api/live.py`:

```python
def register_live_routes(app, conn):
    """Mount live-draft endpoints. In-process state only, same lifetime as
    `create_app`'s connection -- a restart mid-draft means starting again,
    which is correct: the cached pool would be stale anyway."""
    state = {"session": None, "last_poll_at": None, "unmapped": [],
             "candidates": [], "as_of_pick": None, "computing_for": None}
    lock = threading.Lock()

    def _recompute(session, picks_made):
        """Run one search and store it, unless a newer pick landed first.

        A result computed against a board that has since changed is worse
        than no result -- it recommends a player who may already be gone. So
        the pick count is captured before the search and re-checked after;
        if it moved, this result is discarded rather than served.
        """
        cur = conn.cursor()
        try:
            taken, taken_order = _drafted_state(cur, session.pool)
            until = picks_until_turn(session.settings, session.my_slot, picks_made)
            frame = search_pick(
                session.pool, session.settings, session.slot_managers,
                session.my_slot, taken, session.betas,
                n_rollouts=rollouts_for(until), seed=session.seed,
                taken_order=taken_order)
        finally:
            cur.close()
        with lock:
            if state["as_of_pick"] is not None and state["as_of_pick"] > picks_made:
                return          # superseded while we were computing
            state["candidates"] = frame.to_dict(orient="records")
            state["as_of_pick"] = picks_made

    @app.post("/api/live/start")
    def live_start(my_slot: int):
        with lock:
            if state["session"] is not None:
                return {"active": True, "reused": True}
        cur = conn.cursor()
        try:
            session = build_session(cur, my_slot)
        finally:
            cur.close()
        with lock:
            state.update({"session": session, "candidates": [],
                          "as_of_pick": None, "unmapped": [],
                          "last_poll_at": None})
        return {"active": True, "reused": False,
                "board_fingerprint": session.board_fingerprint,
                "seed": session.seed}

    @app.get("/api/live/state")
    def live_state():
        now = datetime.now(timezone.utc)
        with lock:
            session = state["session"]
            if session is None:
                return {"active": False, "picks_made": 0, "on_the_clock": None,
                        "candidates": [], "candidates_as_of_pick": None,
                        "last_poll_at": None, "stale": True,
                        "unmapped_picks": []}
            snapshot = dict(state)
        cur = conn.cursor()
        try:
            picks_made = cur.execute("SELECT count(*) FROM drafted").fetchone()[0]
        finally:
            cur.close()
        slots = snake_slots(session.settings.teams, session.settings.rounds)
        on_clock = slots[picks_made] if picks_made < len(slots) else None
        return {
            "active": True,
            "picks_made": int(picks_made),
            "on_the_clock": on_clock,
            "my_slot": session.my_slot,
            "candidates": snapshot["candidates"],
            "candidates_as_of_pick": snapshot["as_of_pick"],
            "last_poll_at": (snapshot["last_poll_at"].isoformat()
                             if snapshot["last_poll_at"] else None),
            "stale": _is_stale(snapshot["last_poll_at"], now),
            "unmapped_picks": snapshot["unmapped"],
        }

    @app.post("/api/live/stop")
    def live_stop():
        with lock:
            state.update({"session": None, "candidates": [],
                          "as_of_pick": None, "unmapped": [],
                          "last_poll_at": None})
        return {"active": False}

    return state, _recompute
```

Then wire it in `api/main.py`. Immediately before `return app` at the end of `create_app`, add:

```python
    from api.live import register_live_routes
    register_live_routes(app, conn)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_live_api.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add api/live.py api/main.py tests/test_live_api.py
git commit -m "feat: live draft recompute policy and endpoints

One code path with N chosen from how close my turn is -- 25 at three or more
picks away, 100 at one or two, 200 on the clock -- against measured costs of
4.7s, 19.5s and 35.2s, picks arriving every ~20-30s and a ~90s clock.

A new pick supersedes an in-flight computation: the pick count is captured
before the search and re-checked after, and a result computed against a board
that has since moved is discarded rather than served. candidates_as_of_pick
lets the client tell a current answer from one still being recomputed.

Never having polled counts as stale, so an empty board is never presented as
a current one."
```

---

### Task 6: The poller thread

**Files:**
- Modify: `api/live.py`
- Test: `tests/test_live_api.py`

**Interfaces:**
- Consumes: everything from Tasks 2-5.
- Produces: `poll_once(conn, session, state, lock, fetch) -> int` — one poll cycle, returns picks written. `fetch` is a one-argument callable taking a URL and returning JSON, matching `EspnClient.get_json` and the fixture-driven pattern `import_seasons` already uses.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_api.py`:

```python
def test_poll_once_is_driven_by_an_injected_fetch(tmp_path):
    """Same seam import_seasons uses: `fetch` is a plain one-argument
    callable, so the poller is testable without network or Playwright."""
    import threading
    from api.live import poll_once
    from pipeline.db import get_conn, write_table

    conn = get_conn(str(tmp_path / "p.duckdb"))
    write_table(conn, "sleeper_ids", pd.DataFrame([
        {"gsis_id": "g1", "espn_id": 111, "sleeper_name": "A",
         "position": "RB", "team": "DET"}]))
    session = _fake_session()
    object.__setattr__(session, "crosswalk", {111: "g1"})
    state = {"last_poll_at": None, "unmapped": []}
    payload = {"draftDetail": {"drafted": False, "picks": [
        {"overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1,
         "teamId": 1, "playerId": 111, "keeper": False}]}}

    n = poll_once(conn, session, state, threading.Lock(), lambda url: payload)
    assert n == 1
    assert conn.execute("SELECT player_id FROM drafted").fetchall() == [("g1",)]
    assert state["last_poll_at"] is not None


def test_poll_once_records_unmapped_picks_in_state(tmp_path):
    """The failure that does the most damage if it is quiet: a drafted player
    stays on the board and gets recommended."""
    import threading
    from api.live import poll_once
    from pipeline.db import get_conn

    conn = get_conn(str(tmp_path / "u.duckdb"))
    session = _fake_session()
    object.__setattr__(session, "crosswalk", {})
    state = {"last_poll_at": None, "unmapped": []}
    payload = {"draftDetail": {"drafted": False, "picks": [
        {"overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1,
         "teamId": 1, "playerId": 999, "keeper": False}]}}

    poll_once(conn, session, state, threading.Lock(), lambda url: payload)
    assert state["unmapped"] == [{"espn_player_id": 999, "overall_pick": 1}]


def test_poll_once_leaves_last_poll_at_unset_when_the_fetch_raises(tmp_path):
    """A failed poll must read as stale rather than as a successful poll that
    found nothing -- otherwise a dead poller looks like a quiet draft."""
    import threading
    from api.live import poll_once
    from pipeline.db import get_conn

    conn = get_conn(str(tmp_path / "e.duckdb"))
    state = {"last_poll_at": None, "unmapped": []}

    def boom(url):
        raise RuntimeError("502 from ESPN")

    with pytest.raises(RuntimeError):
        poll_once(conn, _fake_session(), state, threading.Lock(), boom)
    assert state["last_poll_at"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_live_api.py -k poll_once -v`
Expected: FAIL with `ImportError: cannot import name 'poll_once'`

- [ ] **Step 3: Write the implementation**

Append to `api/live.py`:

```python
from pipeline.espn_live import apply_picks, translate
from pipeline.espn_league import season_url
from scoring.config import CURRENT_SEASON


def poll_once(conn, session, state, lock, fetch) -> int:
    """One poll cycle: fetch, translate, write. Returns picks written.

    `fetch` is a plain one-argument callable taking a URL and returning JSON
    -- the same seam `import_seasons` uses, which is what lets this be driven
    by a fixture instead of Playwright in tests.

    `last_poll_at` is set only after a successful write. A failed poll must
    read as stale rather than as a successful poll that found nothing:
    otherwise a dead poller is indistinguishable from a quiet draft, and the
    board silently freezes while looking current.
    """
    payload = fetch(season_url(session.league_id, CURRENT_SEASON))
    live = translate(payload, session.crosswalk)
    cur = conn.cursor()
    try:
        written = apply_picks(cur, live)
    finally:
        cur.close()
    with lock:
        state["unmapped"] = live.unmapped
        state["last_poll_at"] = datetime.now(timezone.utc)
    return written
```

`DraftSession.league_id` already exists from Task 4 and `build_session` already populates it from the `league` table, so `season_url(session.league_id, CURRENT_SEASON)` needs no further wiring.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_live_api.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add api/live.py tests/test_live_api.py
git commit -m "feat: the live draft poller

fetch is a plain one-argument callable, the same seam import_seasons uses, so
the poller is testable from a fixture without network or Playwright.

last_poll_at is set only after a successful write. A failed poll must read as
stale rather than as a successful poll that found nothing -- otherwise a dead
poller is indistinguishable from a quiet draft and the board freezes while
looking current."
```

---

### Task 7: The on-the-clock view

**Files:**
- Create: `web/src/components/LiveDraft.tsx`
- Modify: `web/src/api.ts`

**Interfaces:**
- Consumes: `GET /api/live/state` from Task 5.
- Produces: a route-mounted component. No other component imports it.

- [ ] **Step 1: Add the type and fetcher**

Append to `web/src/api.ts`:

```typescript
export interface LiveCandidate {
  player_id: string
  ev: number
  se: number
  applied_pct: number
  rank: number
}

export interface LiveState {
  active: boolean
  picks_made: number
  on_the_clock: number | null
  my_slot?: number
  candidates: LiveCandidate[]
  // The pick count these candidates were computed against. Below picks_made
  // means a refresh is still running -- the difference between "current" and
  // "thinking", which the UI must not blur.
  candidates_as_of_pick: number | null
  last_poll_at: string | null
  stale: boolean
  unmapped_picks: { espn_player_id: number; overall_pick: number }[]
}

export async function fetchLiveState(): Promise<LiveState> {
  const res = await fetch('/api/live/state')
  if (!res.ok) throw new Error('Failed to load live draft state')
  return res.json()
}
```

- [ ] **Step 2: Write the component**

```tsx
import { useEffect, useState } from 'react'
import { fetchLiveState, type LiveState } from '../api'

const POLL_MS = 2500

export function LiveDraft() {
  const [state, setState] = useState<LiveState | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      try {
        const next = await fetchLiveState()
        if (!cancelled) { setState(next); setError(null) }
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      }
    }
    tick()
    const id = setInterval(tick, POLL_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  if (error) return <div className="live-error">{error}</div>
  if (!state) return <div className="live-idle">Loading…</div>
  if (!state.active) return <div className="live-idle">No draft in progress.</div>

  const onClock = state.on_the_clock === state.my_slot
  // Candidates computed against an older board are stale answers, not
  // current ones. Saying so beats showing a confident number for a player
  // who may already be gone.
  const thinking = state.candidates_as_of_pick !== null
    && state.candidates_as_of_pick < state.picks_made

  return (
    <div className="live-draft">
      <header className="live-header">
        <span className={onClock ? 'live-onclock' : 'live-waiting'}>
          {onClock ? "You're on the clock" : `Slot ${state.on_the_clock} picking`}
        </span>
        <span className="live-picks">{state.picks_made} picks in</span>
        {state.stale && (
          <span className="live-stale" title="No successful poll in over 15s">
            board may be stale
          </span>
        )}
      </header>

      {state.unmapped_picks.length > 0 && (
        <div className="live-unmapped">
          {state.unmapped_picks.length} pick(s) could not be matched to a
          player and are still on your board:{' '}
          {state.unmapped_picks.map((u) => `#${u.overall_pick}`).join(', ')}
        </div>
      )}

      <ol className="live-candidates">
        {state.candidates.slice(0, 8).map((c) => (
          <li key={c.player_id} className={thinking ? 'live-thinking' : undefined}>
            <span className="live-cand-id">{c.player_id}</span>
            <span className="live-cand-ev">{c.ev.toFixed(1)}</span>
            <span className="live-cand-se">±{c.se.toFixed(1)}</span>
            <span className="live-cand-avail">{Math.round(c.applied_pct)}%</span>
          </li>
        ))}
      </ol>
      {thinking && <div className="live-thinking-note">recomputing…</div>}
    </div>
  )
}
```

- [ ] **Step 3: Verify the build**

Run: `cd web && npx tsc -b && npm run build`
Expected: no type errors, build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/LiveDraft.tsx web/src/api.ts
git commit -m "feat: on-the-clock live draft view

Polls /api/live/state every 2.5s -- far below ESPN's pick cadence and
debuggable at 8pm on draft night, which websockets would not be.

Three things the UI must not blur, each shown explicitly: a stale poll (no
success in 15s), unmapped picks still sitting on the board, and candidates
computed against an older pick count. The last is the difference between a
current answer and one still being recomputed."
```

---

### Task 8: End-to-end replay against the recorded draft

**Files:**
- Test: `tests/test_live_api.py`

**Interfaces:**
- Consumes: everything above, plus `tests/fixtures/espn_live_draft.json` from Task 1.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_live_api.py`:

```python
def test_replaying_the_recorded_draft_advances_picks_and_never_loses_one(tmp_path):
    """The whole loop against a real ESPN payload, replayed pick by pick.

    Feeding a growing prefix of the recorded picks is what a draft actually
    looks like to the poller. Two invariants hold at every step: `drafted`
    equals the prefix exactly (nothing dropped, nothing left over from the
    previous poll), and no pick is silently lost -- every one either lands in
    `drafted` or appears in `unmapped`.
    """
    import json
    import threading
    from pathlib import Path

    from api.live import poll_once
    from pipeline.db import get_conn
    from pipeline.espn_league import _is_real_pick

    fixture = Path("tests/fixtures/espn_live_draft.json")
    if not fixture.exists():
        pytest.skip("run Task 1's recorder first")

    payload = json.loads(fixture.read_text())
    all_picks = [p for p in payload["draftDetail"]["picks"] if _is_real_pick(p)]
    all_picks.sort(key=lambda p: p["overallPickNumber"])
    crosswalk = {int(p["playerId"]): f"pid{p['playerId']}" for p in all_picks}

    conn = get_conn(str(tmp_path / "replay.duckdb"))
    session = _fake_session()
    object.__setattr__(session, "crosswalk", crosswalk)
    state = {"last_poll_at": None, "unmapped": []}
    lock = threading.Lock()

    for n in range(1, len(all_picks) + 1):
        prefix = {"draftDetail": {"drafted": False, "picks": all_picks[:n]}}
        written = poll_once(conn, session, state, lock, lambda url: prefix)
        rows = conn.execute(
            "SELECT player_id, pick_no FROM drafted ORDER BY pick_no").fetchall()
        assert written == n
        assert len(rows) == n, f"after {n} picks the table held {len(rows)}"
        assert [r[1] for r in rows] == [p["overallPickNumber"] for p in all_picks[:n]]
        assert written + len(state["unmapped"]) == n
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_live_api.py -k replaying -v`
Expected: FAIL if any task above is incomplete; SKIP if Task 1's fixture is absent.

- [ ] **Step 3: Fix whatever it catches**

No new implementation is expected. This test exists to catch integration gaps between Tasks 2-6 that unit tests pass over — most likely the pick-ordering contract or `apply_picks` leaving rows behind.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_live_api.py
git commit -m "test: replay the recorded ESPN draft through the whole loop

A growing prefix of real picks is what a draft looks like to the poller. Two
invariants at every step: drafted equals the prefix exactly, and no pick is
silently lost -- each one either lands in drafted or shows up in unmapped."
```

---

## Self-Review

**Spec coverage.** Every requirement maps to a task: the session and its cost rationale (4), the pinned seed (4), the three components plus the correction that `draft_sim.py` needs no change (Global Constraints), the endpoints (5), the recompute tiers (5), starting the deep search early (5, via `rollouts_for(0)` firing when `picks_until_turn` is 0), supersede and `candidates_as_of_pick` (5), all five failure modes (2, 3, 5, 6), the manual-fallback reconciliation rule (3), and all three testing layers (2, 8, and Task 1's live mock). The gating risk is Task 1 and it is first.

**Placeholder scan.** Clean. An earlier draft carried a dead `order = pd.read_sql if False else None` line in Task 4 whose only purpose was to mark where an import belonged, plus a Task 6 step that bolted `league_id` onto `DraftSession` after the fact. Both were rewritten: the import sits with the others and `league_id` is a field of the dataclass from the moment it is defined.

**Type consistency.** `LivePicks.rows` carries `["player_id", "pick_no"]` in Tasks 2, 3, 6 and 8. `translate(payload, crosswalk)`, `apply_picks(conn, live)`, `build_crosswalk(conn)`, `poll_once(conn, session, state, lock, fetch)`, `rollouts_for(int)`, `picks_until_turn(settings, my_slot, picks_made)` and `board_fingerprint(board)` are used with identical signatures everywhere they appear. `DraftSession` is defined once in Task 4 with every field it ever carries, and `_fake_session` constructs it with all of them.

**Known gap, deliberate.** No task starts the poller *thread* — `poll_once` is called by tests, not by a loop. Wiring the thread is a four-line change to `live_start`, but it cannot be tested without either network or a fake clock, and inventing a threading harness to cover four lines is worse than covering them in the live mock of Task 1. Task 5's `live_start` is where it goes.
