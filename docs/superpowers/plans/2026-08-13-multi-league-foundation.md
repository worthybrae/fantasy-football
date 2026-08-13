# Multi-League Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let each ESPN user's league run in isolation, so many people can use the tool at once without colliding — the data foundation for a hosted product.

**Architecture:** Per-league database files. The heavy universal data (nflverse stats, market ranks — ~880k rows) lives in one shared read-only file; each league gets its own small file with just its league-specific tables (`draft_picks`, `draft_teams`, `draft_order`, `manager_profiles`, `manager_tendencies`, `league`, `drafted`). A connection routes to a league's file, so the dozen existing `read_table(conn, ...)` sites never change — the `conn` simply points at the right database. `write_table`'s `CREATE OR REPLACE` semantics stay correct because each file is one league. This is the least invasive isolation that actually scales, and it keeps the working single-user path untouched: the default league IS today's `data/nfl.duckdb`.

**Tech Stack:** Python 3.10, DuckDB, FastAPI, pandas.

## Global Constraints

- **The single-user path must not change behaviour.** The default league resolves to the existing `data/nfl.duckdb`; every current test and the live app keep working unchanged.
- **Universal data is never duplicated per league.** A per-league file holds only the league-specific tables; universal tables are read from the shared file.
- **Cold-start already works** (`draft_model.cold_start_fits`) — a league with no history runs on the market prior. Per-league import is therefore optional for a usable draft, not a blocker.
- **`drafted` is per-league.** Two live drafts must never write to the same `drafted` rows.
- Run tests with `.venv/bin/python -m pytest`. No test may open `data/nfl.duckdb`; use `tmp_path`.
- DuckDB is single-writer per file. Two connections to one league file conflict; the session owns its league's connection.

---

## File Structure

| file | responsibility |
|---|---|
| `pipeline/leagues.py` (create) | Resolve a league id to its database path; provision a new league's file by attaching the shared universal data; the one place that knows the per-league layout. |
| `pipeline/db.py` (modify) | `UNIVERSAL_TABLES` / `LEAGUE_TABLES` constants — the single source of truth for which tables are which. |
| `api/live.py` (modify) | Build a session against a league's own connection, not the one global connection. |
| `tests/test_leagues.py` (create) | Path resolution, provisioning, isolation between two leagues. |

The universal-vs-league split, verified against the live schema:

- **Universal (shared):** `weekly`, `snap_counts`, `depth_charts`, `players`, `schedules`, `adp`, `espn_adp`, `fp_ecr`, `cbs_ranks`, `mfl_adp`, `historic_adp`, `historic_espn_cs`, `sleeper_ids`, `meta`, `model_backtest`.
- **League-specific:** `league`, `draft_picks`, `draft_teams`, `draft_order`, `manager_profiles`, `manager_tendencies`, `drafted`, `sim_board`, `sim_results`, `sim_survival`.

---

### Task 1: Name which tables are universal and which are per-league

The whole plan hinges on this split being explicit and correct. Encode it once.

**Files:**
- Modify: `pipeline/db.py`
- Test: `tests/test_leagues.py`

**Interfaces:**
- Produces: `UNIVERSAL_TABLES: frozenset[str]`, `LEAGUE_TABLES: frozenset[str]` in `pipeline/db.py`.

- [ ] **Step 1: Write the failing test**

```python
def test_table_split_is_complete_and_disjoint():
    """Every table the app uses is classified exactly once. A table in
    neither set would silently be treated as universal by provisioning and
    leak one league's data into every other; a table in both is a
    contradiction the code cannot honour."""
    from pipeline.db import UNIVERSAL_TABLES, LEAGUE_TABLES
    assert UNIVERSAL_TABLES.isdisjoint(LEAGUE_TABLES)
    # The league-specific set, exact -- adding a per-league table without
    # listing it here is the bug this pins.
    assert LEAGUE_TABLES == frozenset({
        "league", "draft_picks", "draft_teams", "draft_order",
        "manager_profiles", "manager_tendencies", "drafted",
        "sim_board", "sim_results", "sim_survival"})
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_leagues.py -v`
Expected: FAIL — `ImportError: cannot import name 'UNIVERSAL_TABLES'`

- [ ] **Step 3: Implement**

Add to `pipeline/db.py`:

```python
# The two kinds of table, kept in one place because the whole multi-league
# design depends on the split being correct. A per-league database holds only
# LEAGUE_TABLES; UNIVERSAL_TABLES are read from the shared file. A table that
# is neither would be treated as universal by provisioning and leak one
# league's rows into every other -- so the classification is asserted complete
# in tests rather than left implicit.
LEAGUE_TABLES = frozenset({
    "league", "draft_picks", "draft_teams", "draft_order",
    "manager_profiles", "manager_tendencies", "drafted",
    "sim_board", "sim_results", "sim_survival",
})
UNIVERSAL_TABLES = frozenset({
    "weekly", "snap_counts", "depth_charts", "players", "schedules",
    "adp", "espn_adp", "fp_ecr", "cbs_ranks", "mfl_adp",
    "historic_adp", "historic_espn_cs", "sleeper_ids", "meta", "model_backtest",
})
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_leagues.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/db.py tests/test_leagues.py
git commit -m "feat: name which tables are universal vs per-league"
```

---

### Task 2: Resolve a league id to a database path

**Files:**
- Create: `pipeline/leagues.py`
- Test: `tests/test_leagues.py`

**Interfaces:**
- Consumes: `pipeline.db.DEFAULT_PATH`.
- Produces:
  - `DEFAULT_LEAGUE = "__default__"` — the sentinel for the single baked-in league.
  - `league_db_path(league_id: str, root: str = "data/leagues") -> str`
  - `LEAGUES_ROOT = "data/leagues"`

- [ ] **Step 1: Write the failing test**

```python
def test_default_league_resolves_to_the_existing_database():
    """The single-user path must not move. The default league is today's
    data/nfl.duckdb, so nothing about the current app changes."""
    from pipeline.leagues import DEFAULT_LEAGUE, league_db_path
    from pipeline.db import DEFAULT_PATH
    assert league_db_path(DEFAULT_LEAGUE) == DEFAULT_PATH


def test_a_real_league_gets_its_own_file_under_the_leagues_root():
    from pipeline.leagues import league_db_path
    p = league_db_path("53929318", root="/tmp/lg")
    assert p == "/tmp/lg/53929318.duckdb"


def test_league_id_is_sanitised_into_the_filename():
    """A league id reaches this from a pasted URL. It must not be able to
    escape the leagues directory via path characters."""
    from pipeline.leagues import league_db_path
    p = league_db_path("../../etc/passwd", root="/tmp/lg")
    assert p.startswith("/tmp/lg/")
    assert ".." not in p.split("/tmp/lg/")[1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_leagues.py -k path -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.leagues'`

- [ ] **Step 3: Implement**

```python
"""Where each league's data lives, and how a new one is provisioned.

Per-league database files: the heavy universal data (nflverse stats, market
ranks) stays in one shared file, and each league gets a small file with only
its own draft history, manager fits and live-draft state. A connection points
at one league's file, so the rest of the app -- every read_table(conn, ...)
site -- never has to know which league it is looking at.

The default league is the existing single database, so the single-user path
this project started as does not move.
"""
import re
from pathlib import Path

from pipeline.db import DEFAULT_PATH

DEFAULT_LEAGUE = "__default__"
LEAGUES_ROOT = "data/leagues"


def _safe_id(league_id: str) -> str:
    """A filename that cannot escape the leagues directory.

    The id arrives from a URL the user pasted, so it is untrusted. Keep only
    the characters a real ESPN league id and season use; anything else --
    slashes, dots, traversal -- is stripped rather than escaped, because a
    stripped id that collides is a visible wrong-league error while a
    traversal that succeeds is a filesystem write outside the sandbox.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", str(league_id))
    return cleaned or "unknown"


def league_db_path(league_id: str, root: str = LEAGUES_ROOT) -> str:
    """The database file for a league. The default league is the shared
    single-user database, unchanged; every other league is its own file."""
    if league_id == DEFAULT_LEAGUE:
        return DEFAULT_PATH
    return str(Path(root) / f"{_safe_id(league_id)}.duckdb")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_leagues.py -k path -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/leagues.py tests/test_leagues.py
git commit -m "feat: resolve a league id to its own database file"
```

---

### Task 3: Provision a new league's database from the shared universal data

A new league's file needs the universal tables (to build a board) without copying them into every file. DuckDB's `ATTACH` reads another database's tables in place; the new file `CREATE TABLE ... AS SELECT` copies only what it must and leaves the rest attached.

**Decision, stated in code:** the universal tables ARE copied into the per-league file on provision, not attached at query time. Attaching at every query would mean threading the attach through all ~12 read sites; copying keeps those sites untouched at the cost of disk. The universal data is ~50MB; at a few hundred leagues that is real but affordable, and the attach optimisation can replace this later behind the same `provision_league` interface. The copy is the honest, working first version.

**Files:**
- Modify: `pipeline/leagues.py`
- Test: `tests/test_leagues.py`

**Interfaces:**
- Consumes: `pipeline.db.get_conn`, `UNIVERSAL_TABLES`, `read_table`, `write_table`; `league_db_path`.
- Produces: `provision_league(league_id, universal_path, root=LEAGUES_ROOT) -> str` — creates the league's file if absent, seeded with the universal tables; returns its path. Idempotent.

- [ ] **Step 1: Write the failing test**

```python
def test_provision_copies_universal_tables_but_not_league_ones(tmp_path):
    """A provisioned league can build a board (universal data present) but
    starts with no draft history of its own (league tables empty), which is
    the cold-start case the model already handles."""
    from pipeline.db import get_conn, write_table, read_table
    from pipeline.leagues import provision_league
    import pandas as pd

    shared = str(tmp_path / "universal.duckdb")
    conn = get_conn(shared)
    write_table(conn, "weekly", pd.DataFrame([{"player_id": "p1", "season": 2025}]))
    write_table(conn, "draft_picks", pd.DataFrame([{"season": 2025, "overall_pick": 1}]))
    conn.close()

    path = provision_league("999", universal_path=shared, root=str(tmp_path / "lg"))
    lg = get_conn(path)
    # universal came across
    assert not read_table(lg, "weekly").empty
    # league-specific did NOT -- this league has its own (empty) history
    assert read_table(lg, "draft_picks").empty
    lg.close()


def test_provision_is_idempotent(tmp_path):
    """Connecting twice must not reprovision and wipe a draft in progress."""
    from pipeline.db import get_conn, write_table, read_table
    from pipeline.leagues import provision_league
    import pandas as pd

    shared = str(tmp_path / "universal.duckdb")
    get_conn(shared).close()
    path = provision_league("999", universal_path=shared, root=str(tmp_path / "lg"))
    lg = get_conn(path)
    lg.execute("INSERT INTO drafted VALUES ('p1', 1)")
    lg.close()

    again = provision_league("999", universal_path=shared, root=str(tmp_path / "lg"))
    assert again == path
    lg = get_conn(path)
    assert not read_table(lg, "drafted").empty      # the in-progress draft survived
    lg.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_leagues.py -k provision -v`
Expected: FAIL — `ImportError: cannot import name 'provision_league'`

- [ ] **Step 3: Implement**

Append to `pipeline/leagues.py`:

```python
from pipeline.db import UNIVERSAL_TABLES, get_conn, read_table, write_table


def provision_league(league_id: str, universal_path: str,
                     root: str = LEAGUES_ROOT) -> str:
    """Ensure a league's database exists, seeded with the universal tables.

    Idempotent: an existing file is left exactly as it is, so reconnecting
    mid-draft never wipes the `drafted` rows already recorded. Only the
    universal tables are copied; the league-specific tables start absent,
    which is precisely the cold-start state `cold_start_fits` handles.
    """
    path = league_db_path(league_id, root=root)
    if league_id == DEFAULT_LEAGUE or Path(path).exists():
        return path
    src = get_conn(universal_path)
    try:
        dst = get_conn(path)
        try:
            for table in UNIVERSAL_TABLES:
                df = read_table(src, table)
                if not df.empty:
                    write_table(dst, table, df)
        finally:
            dst.close()
    finally:
        src.close()
    return path
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_leagues.py -k provision -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full suite (nothing else should move)**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass — no existing code calls `provision_league` yet.

- [ ] **Step 6: Commit**

```bash
git add pipeline/leagues.py tests/test_leagues.py
git commit -m "feat: provision a league's database from the shared universal data

The universal tables are copied into the per-league file rather than attached
at query time, so the dozen read_table sites never change. Idempotent, so a
reconnect mid-draft keeps the drafted rows already recorded."
```

---

### Task 4: Two leagues' live-draft state cannot collide

The payoff: prove that a pick in one league does not appear in another. This is the isolation the whole plan exists for.

**Files:**
- Test: `tests/test_leagues.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3.

- [ ] **Step 1: Write the failing test**

```python
def test_two_leagues_have_independent_drafted_tables(tmp_path):
    """The whole point. A pick marked in league A must be invisible to
    league B -- they are different files, so a global `drafted` table can no
    longer merge two live drafts into one."""
    from pipeline.db import get_conn, read_table
    from pipeline.leagues import provision_league

    shared = str(tmp_path / "universal.duckdb")
    get_conn(shared).close()
    root = str(tmp_path / "lg")

    a = get_conn(provision_league("aaa", universal_path=shared, root=root))
    b = get_conn(provision_league("bbb", universal_path=shared, root=root))
    a.execute("INSERT INTO drafted VALUES ('gibbs', 1)")

    assert read_table(a, "drafted")["player_id"].tolist() == ["gibbs"]
    assert read_table(b, "drafted").empty       # league B never saw it
    a.close()
    b.close()
```

- [ ] **Step 2: Run to verify it fails or passes**

Run: `.venv/bin/python -m pytest tests/test_leagues.py -k independent -v`
Expected: PASS (the isolation is structural — this test documents and locks it).

- [ ] **Step 3: Commit**

```bash
git add tests/test_leagues.py
git commit -m "test: two leagues' live-draft state is isolated by construction"
```

---

### Task 5: The live session builds against its league's own connection

Wire it in. Today `register_live_routes` closes over one connection to the single database. A connect carrying a league id must build the session against that league's file instead.

**Files:**
- Modify: `api/live.py`
- Test: `tests/test_live_api.py`

**Interfaces:**
- Consumes: `provision_league`, `league_db_path` (Tasks 2-3); the existing `build_session`, `TokenBody`, `ConnectBody`.
- Produces: a session built on `get_conn(provision_league(league_id, universal_path=DEFAULT_PATH))` when a league id is supplied, falling back to the shared connection for the default league.

- [ ] **Step 1: Write the failing test**

```python
def test_connect_with_a_league_id_isolates_its_drafted_table(tmp_path, monkeypatch):
    """A connect naming a real league builds against that league's own file,
    so marking a pick there does not touch the default database."""
    from fastapi.testclient import TestClient
    from api.main import create_app
    from pipeline.db import get_conn, read_table

    default_path = str(tmp_path / "default.duckdb")
    _seed_minimal_live_db(default_path)
    monkeypatch.setattr("api.live.run_listener", lambda *a, **k: None)
    monkeypatch.setattr("pipeline.leagues.LEAGUES_ROOT", str(tmp_path / "lg"))

    client = TestClient(create_app(default_path))
    r = client.post("/api/live/connect", json={
        "url": "https://fantasy.espn.com/football/draft?leagueId=777&teamId=2"})
    assert r.status_code == 200
    client.post("/api/live/stop")

    # The default database's drafted table is untouched by league 777's session.
    assert read_table(get_conn(default_path), "drafted").empty
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_live_api.py -k with_a_league_id -v`
Expected: FAIL — the session builds against the default connection, so no per-league file is created.

- [ ] **Step 3: Implement**

In `api/live.py`'s `live_connect`, after resolving `league_id`, build the session against that league's connection. Replace the `build_session(cur, ...)` call so it uses a connection from `provision_league(league_id, universal_path=<the app's own db path>)` for a non-default league. Store that connection on the session state so `_recompute` and the listener use it, and close it in `live_stop`. Thread the app's own database path into `register_live_routes` (it already closes over `conn`; add the path alongside so `provision_league`'s `universal_path` is the real shared file, not a hardcoded default).

Exact wiring:
- `register_live_routes(app, conn, db_path)` — add `db_path`, passed from `create_app` where `conn = get_conn(db_path)` is built.
- In `live_connect`: `from pipeline.leagues import provision_league, DEFAULT_LEAGUE`; if `league_id and league_id != DEFAULT_LEAGUE`, `league_path = provision_league(league_id, universal_path=db_path)` and `league_conn = get_conn(league_path)`; build the session on `league_conn.cursor()`. For the default league, keep using `conn`. Track `league_conn` in `state` and close it in `_stop_listener`/`live_stop` so a per-league file is not left locked.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_live_api.py -q`
Expected: all pass, including the new isolation test.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all pass — the default-league path is unchanged, so existing tests do not move.

- [ ] **Step 6: Commit**

```bash
git add api/live.py tests/test_live_api.py
git commit -m "feat: build the live session against the connecting league's own file

A connect carrying a league id provisions and opens that league's database, so
concurrent drafts in different leagues have independent boards, fits and
drafted state. The default league still resolves to the shared database, so the
single-user path is unchanged."
```

---

## Out of scope

- Per-league ESPN history import (cold-start covers a usable draft without it; import is a later enrichment).
- A connection pool / eviction policy for many concurrent leagues (each session opens and closes its own; a registry with limits is a later concern).
- Replacing the universal-table copy with `ATTACH` (an optimisation behind the same `provision_league` interface).
- Accounts, payments, deployment.

## Self-Review

**Spec coverage.** The spec's five slices: cold-start (slice 1) already shipped; league-scoped data (slice 2) is Tasks 1-4; per-user session (part of slice 4) is Task 5. Per-user import (slice 3) and the board cache (slice 5) are explicitly deferred to Out of Scope with reasons — cold-start removes import from the critical path, and the cache is an optimisation. The concurrency registry is named as deferred.

**Placeholder scan.** Task 5's Step 3 describes wiring in prose plus exact signatures rather than a single code block, because it threads through an existing function whose surrounding code the implementer must read; the exact names (`register_live_routes(app, conn, db_path)`, `provision_league(league_id, universal_path=db_path)`, `league_conn`) are given so nothing is left to guess. Every other code step is complete and runnable.

**Type consistency.** `league_db_path(league_id, root)`, `provision_league(league_id, universal_path, root)`, `DEFAULT_LEAGUE`, `UNIVERSAL_TABLES`/`LEAGUE_TABLES` are used with identical signatures across Tasks 1-5. `provision_league` returns a path (str) consumed by `get_conn` everywhere.
