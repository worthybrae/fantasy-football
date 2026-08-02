# Fantasy Draft Research Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draft-prep tool for an 8-team PPR league: nflverse data pipeline → DuckDB → composite scoring engine with VOR → FastAPI → React draft board.

**Architecture:** Three isolated units. `pipeline/` fetches free data (nfl_data_py, Fantasy Football Calculator ADP) into a single DuckDB file. `scoring/` is pure functions from dataframes to a ranked board. `api/` (FastAPI) serves the board and draft-night state to `web/` (Vite + React + TS).

**Tech Stack:** Python 3.10, nfl_data_py, pandas, DuckDB, FastAPI, uvicorn, pytest; Vite, React 18, TypeScript, @tanstack/react-table.

**Spec:** `docs/superpowers/specs/2026-08-02-draft-research-tool-design.md`

## Global Constraints

- Python 3.10 (system is 3.10.13); create venv at `.venv/`, deps in `requirements.txt`.
- Free data sources only — no API keys anywhere.
- League config (never hardcode elsewhere, always import from `scoring/config.py`): 8 teams, lineup QB/2RB/2WR/TE/2Flex(W-R-T)/K/DST, PPR scoring, current season 2026, history seasons 2023–2025.
- `data/` directory (DuckDB file) is gitignored.
- All API routes prefixed `/api`.
- Scoring functions are pure: dataframes in, dataframes out, no I/O.
- K and DST have no weekly stats in nfl_data_py: they enter the board from ADP and are scored by the environment factor only (documented deviation, approved in design).
- Snap counts are ingested (cheap, useful later) but the role factor uses depth-chart rank + target/carry share from weekly data — snap-count name-joins are unreliable (documented simplification).
- Schedule factor uses prior-season fantasy-points-allowed-by-position only; Vegas lines feed the environment factor (simplification of spec's "lines + FPA" — lines measure offense, FPA measures defense).
- Missing factor data → neutral value 50 (factors are 0–100 within-position percentiles).
- Commit after each task with a conventional-commit message.

---

### Task 1: Legacy teardown + Python project scaffolding

**Files:**
- Delete: `chromedriver_test.py`, `ingest.py`, `ingest_v2.py`, `test.py`, `t.py`, `initialize_chromedriver.py`, `create_tables.py`, `chromedriver`, `queries/`, `models/`, `schemas/`, `helpers/`, `database/`
- Create: `requirements.txt`, `.gitignore` (modify), `pipeline/__init__.py`, `scoring/__init__.py`, `api/__init__.py`, `tests/__init__.py`, `README.md` (overwrite)

**Interfaces:**
- Produces: a clean repo where `pytest` runs (0 tests, exit 0 with `--co` empty ok) and `.venv` has all Python deps installed.

- [ ] **Step 1: Delete legacy files**

```bash
git rm -r chromedriver_test.py ingest.py ingest_v2.py test.py t.py initialize_chromedriver.py create_tables.py queries models schemas helpers database requirements.txt
rm -f chromedriver
```

- [ ] **Step 2: Write requirements.txt**

```
nfl_data_py>=0.3.2
pandas>=1.5
duckdb>=0.10
fastapi>=0.110
uvicorn>=0.29
requests>=2.31
pytest>=8.0
httpx>=0.27
```

- [ ] **Step 3: Update .gitignore**

Append lines: `data/`, `.venv/`, `__pycache__/`, `web/node_modules/`, `web/dist/`

- [ ] **Step 4: Create venv and install**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

- [ ] **Step 5: Create package skeletons**

Empty `__init__.py` in `pipeline/`, `scoring/`, `api/`, `tests/`. Overwrite `README.md` with a short project description (what it is, `refresh` → `api` → `web` usage; expand later in Task 9).

- [ ] **Step 6: Verify**

Run: `.venv/bin/python -c "import nfl_data_py, duckdb, fastapi"` → no error. `.venv/bin/pytest` → "no tests ran".

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: remove legacy scraper, scaffold python project"
```

---

### Task 2: PPR point calculation

**Files:**
- Create: `scoring/ppr.py`
- Test: `tests/test_ppr.py`

**Interfaces:**
- Produces: `compute_ppr_points(df: pd.DataFrame) -> pd.Series` — takes a dataframe with nfl_data_py weekly-stats columns, returns PPR fantasy points per row. Missing columns are treated as 0 (use `df.get(col, 0)` pattern via a helper).

Scoring rules: pass yds ×0.04, pass TD ×4, INT −2, rush yds ×0.1, rush TD ×6, reception ×1, rec yds ×0.1, rec TD ×6, fumbles lost −2 (sum of `sack_fumbles_lost`, `rushing_fumbles_lost`, `receiving_fumbles_lost`), 2-pt conversions ×2 (sum of `passing_2pt_conversions`, `rushing_2pt_conversions`, `receiving_2pt_conversions`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ppr.py
import pandas as pd
from scoring.ppr import compute_ppr_points

def test_receiver_line():
    # 8 rec, 100 yds, 1 TD = 8 + 10 + 6 = 24.0
    df = pd.DataFrame([{"receptions": 8, "receiving_yards": 100, "receiving_tds": 1}])
    assert compute_ppr_points(df).iloc[0] == 24.0

def test_qb_line():
    # 300 pass yds, 3 TD, 1 INT, 20 rush yds = 12 + 12 - 2 + 2 = 24.0
    df = pd.DataFrame([{"passing_yards": 300, "passing_tds": 3,
                        "interceptions": 1, "rushing_yards": 20}])
    assert compute_ppr_points(df).iloc[0] == 24.0

def test_fumbles_and_2pt():
    df = pd.DataFrame([{"rushing_fumbles_lost": 1, "sack_fumbles_lost": 1,
                        "rushing_2pt_conversions": 1}])
    assert compute_ppr_points(df).iloc[0] == -2.0

def test_missing_columns_are_zero():
    df = pd.DataFrame([{"receptions": 5}])
    assert compute_ppr_points(df).iloc[0] == 5.0
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_ppr.py -v` → ImportError.

- [ ] **Step 3: Implement**

```python
# scoring/ppr.py
import pandas as pd

_RULES = {
    "passing_yards": 0.04, "passing_tds": 4.0, "interceptions": -2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receptions": 1.0, "receiving_yards": 0.1, "receiving_tds": 6.0,
    "sack_fumbles_lost": -2.0, "rushing_fumbles_lost": -2.0, "receiving_fumbles_lost": -2.0,
    "passing_2pt_conversions": 2.0, "rushing_2pt_conversions": 2.0, "receiving_2pt_conversions": 2.0,
}

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0)
    return pd.Series(0.0, index=df.index)

def compute_ppr_points(df: pd.DataFrame) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for col, pts in _RULES.items():
        total = total + _col(df, col) * pts
    return total
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_ppr.py -v` → 4 passed.

- [ ] **Step 5: Commit** — `git add scoring tests && git commit -m "feat: PPR point calculation"`

---

### Task 3: Data pipeline (DuckDB helpers, sources, refresh CLI)

**Files:**
- Create: `pipeline/db.py`, `pipeline/sources.py`, `pipeline/refresh.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `pipeline.db.get_conn(path: str = "data/nfl.duckdb") -> duckdb.DuckDBPyConnection` (creates parent dir; ensures `meta` and `drafted` tables exist)
  - `pipeline.db.write_table(conn, name: str, df: pd.DataFrame) -> None` (drop-and-recreate from dataframe)
  - `pipeline.db.read_table(conn, name: str) -> pd.DataFrame` (empty DataFrame if table missing)
  - `pipeline.db.record_freshness(conn, source: str, ok: bool, rows: int) -> None` (upsert into `meta(source, ok, rows, refreshed_at)`)
  - `pipeline.sources.fetch_weekly(years: list[int]) -> pd.DataFrame`
  - `pipeline.sources.fetch_snap_counts(years: list[int]) -> pd.DataFrame`
  - `pipeline.sources.fetch_depth_charts(season: int) -> pd.DataFrame`
  - `pipeline.sources.fetch_schedules(season: int) -> pd.DataFrame`
  - `pipeline.sources.parse_adp(payload: dict) -> pd.DataFrame` (pure, testable) and `fetch_adp(year: int, teams: int = 12) -> pd.DataFrame`
  - `python -m pipeline.refresh` CLI. Table names written: `weekly`, `snap_counts`, `depth_charts`, `schedules`, `adp`.

- [ ] **Step 1: Write failing tests** (db helpers + ADP parsing only; network fetches are verified manually in Step 5)

```python
# tests/test_pipeline.py
import pandas as pd
from pipeline.db import get_conn, write_table, read_table, record_freshness
from pipeline.sources import parse_adp

def test_write_and_read_roundtrip(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "weekly", pd.DataFrame({"a": [1, 2]}))
    assert read_table(conn, "weekly")["a"].tolist() == [1, 2]

def test_read_missing_table_returns_empty(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    assert read_table(conn, "nope").empty

def test_freshness_upsert(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    record_freshness(conn, "weekly", True, 100)
    record_freshness(conn, "weekly", True, 200)
    meta = read_table(conn, "meta")
    assert len(meta) == 1 and meta.iloc[0]["rows"] == 200

def test_parse_adp():
    payload = {"players": [
        {"player_id": 1, "name": "Justin Jefferson", "position": "WR", "team": "MIN", "adp": 3.2},
        {"player_id": 2, "name": "49ers Defense", "position": "DEF", "team": "SF", "adp": 140.1},
    ]}
    df = parse_adp(payload)
    assert list(df.columns) == ["adp_name", "position", "team", "adp"]
    assert df.iloc[1]["position"] == "DST"  # FFC "DEF" mapped to nflverse "DST"
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_pipeline.py -v` → ImportError.

- [ ] **Step 3: Implement db.py**

```python
# pipeline/db.py
from pathlib import Path
import duckdb
import pandas as pd

DEFAULT_PATH = "data/nfl.duckdb"

def get_conn(path: str = DEFAULT_PATH) -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS meta (
        source VARCHAR PRIMARY KEY, ok BOOLEAN, rows BIGINT, refreshed_at TIMESTAMP)""")
    conn.execute("CREATE TABLE IF NOT EXISTS drafted (player_id VARCHAR PRIMARY KEY)")
    return conn

def write_table(conn, name: str, df: pd.DataFrame) -> None:
    conn.register("_incoming", df)
    conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _incoming")
    conn.unregister("_incoming")

def read_table(conn, name: str) -> pd.DataFrame:
    exists = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()[0]
    if not exists:
        return pd.DataFrame()
    return conn.execute(f"SELECT * FROM {name}").df()

def record_freshness(conn, source: str, ok: bool, rows: int) -> None:
    conn.execute(
        """INSERT INTO meta VALUES (?, ?, ?, now())
           ON CONFLICT (source) DO UPDATE SET
             ok = excluded.ok, rows = excluded.rows, refreshed_at = excluded.refreshed_at""",
        [source, ok, rows])
```

- [ ] **Step 4: Implement sources.py and refresh.py**

```python
# pipeline/sources.py
import nfl_data_py as nfl
import pandas as pd
import requests

ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"

def fetch_weekly(years):
    return nfl.import_weekly_data(years)

def fetch_snap_counts(years):
    return nfl.import_snap_counts(years)

def fetch_depth_charts(season):
    return nfl.import_depth_charts([season])

def fetch_schedules(season):
    return nfl.import_schedules([season])

def parse_adp(payload: dict) -> pd.DataFrame:
    rows = [{"adp_name": p["name"],
             "position": "DST" if p["position"] == "DEF" else p["position"],
             "team": p.get("team"), "adp": p["adp"]}
            for p in payload.get("players", [])]
    return pd.DataFrame(rows, columns=["adp_name", "position", "team", "adp"])

def fetch_adp(year: int, teams: int = 12) -> pd.DataFrame:
    resp = requests.get(ADP_URL, params={"teams": teams, "year": year}, timeout=30)
    resp.raise_for_status()
    return parse_adp(resp.json())
```

```python
# pipeline/refresh.py
"""Refresh all data sources into DuckDB. Run: python -m pipeline.refresh"""
from pipeline import sources
from pipeline.db import get_conn, write_table, record_freshness
from scoring.config import CURRENT_SEASON, HISTORY_SEASONS

def main() -> None:
    conn = get_conn()
    jobs = {
        "weekly": lambda: sources.fetch_weekly(HISTORY_SEASONS),
        "snap_counts": lambda: sources.fetch_snap_counts(HISTORY_SEASONS),
        "depth_charts": lambda: sources.fetch_depth_charts(CURRENT_SEASON),
        "schedules": lambda: sources.fetch_schedules(CURRENT_SEASON),
        "adp": lambda: sources.fetch_adp(CURRENT_SEASON),
    }
    summary = []
    for name, job in jobs.items():
        try:
            df = job()
            write_table(conn, name, df)
            record_freshness(conn, name, True, len(df))
            summary.append(f"  OK   {name}: {len(df)} rows")
        except Exception as e:  # one source failing must not abort the rest
            record_freshness(conn, name, False, 0)
            summary.append(f"  FAIL {name}: {e}")
    print("Refresh complete:\n" + "\n".join(summary))

if __name__ == "__main__":
    main()
```

Note: `scoring/config.py` does not exist yet — create a minimal version in this task (Task 5 extends it):

```python
# scoring/config.py
CURRENT_SEASON = 2026
HISTORY_SEASONS = [2023, 2024, 2025]
```

- [ ] **Step 5: Run tests, then run the real refresh** — `.venv/bin/pytest tests/test_pipeline.py -v` → 4 passed. Then `.venv/bin/python -m pipeline.refresh` and verify output lists row counts (weekly should be ~15k rows; `adp` and `depth_charts` may legitimately FAIL/be small this early in preseason — that is acceptable, note which succeeded in the task report). Then verify: `.venv/bin/python -c "from pipeline.db import get_conn, read_table; c = get_conn(); print(read_table(c, 'meta'))"`.

- [ ] **Step 6: Commit** — `git add pipeline scoring/config.py tests && git commit -m "feat: data pipeline with DuckDB storage and refresh CLI"`

---

### Task 4: Scoring factors

**Files:**
- Create: `scoring/factors.py`
- Test: `tests/test_factors.py`

**Interfaces:**
- Consumes: `scoring.ppr.compute_ppr_points`, `scoring.config` (Task 5 finalizes values; import `RECENCY_WEIGHTS`, `CURRENT_SEASON` — add `RECENCY_WEIGHTS = {2025: 0.5, 2024: 0.3, 2023: 0.2}` to config in this task).
- Produces (all pure; all take nfl_data_py-shaped dataframes):
  - `player_seasons(weekly) -> pd.DataFrame` — per player_id×season: `ppg` (PPR pts/game), `games`, plus carried `player_name`, `position`, `team` (last observed). Adds `ppr_points` via `compute_ppr_points`.
  - `production_factor(weekly) -> pd.DataFrame[player_id, production_raw]` — recency-weighted ppg using `RECENCY_WEIGHTS`, weights renormalized over seasons the player actually played.
  - `durability_factor(weekly) -> pd.DataFrame[player_id, durability_raw]` — total games / (17 × seasons-in-league-window since first appearance).
  - `role_factor(depth_charts, weekly) -> pd.DataFrame[player_id, role_raw]` — from latest depth chart week: `role_raw = 1 / depth_team_rank` (1.0 starter, 0.5 second string…), blended 50/50 with prior-season (2025) share of team targets+carries where available.
  - `environment_factor(schedules) -> pd.DataFrame[team, env_raw]` — mean implied points per team over games that have `total_line` and `spread_line`: home implied = (total+spread)/2, away = (total−spread)/2 (nflverse `spread_line` is positive when home favored).
  - `schedule_factor(weekly_2025, schedules_2026) -> pd.DataFrame[team, position, sos_raw]` — 2025 PPR points allowed per defense per position (`opponent_team` column in weekly), averaged over each team's 2026 opponents; higher = easier schedule.
  - `bye_weeks(schedules) -> pd.DataFrame[team, bye]` — the week in 1..18 where the team doesn't appear.
  - `normalize_within_position(df, raw_col, out_col) -> pd.DataFrame` — adds `out_col` = percentile rank ×100 of `raw_col` grouped by `position`; NaN raw → 50.0.

- [ ] **Step 1: Write failing tests with small synthetic dataframes**

```python
# tests/test_factors.py
import pandas as pd
from scoring import factors

def _weekly(rows):
    base = {"player_id": "p1", "player_display_name": "A Player", "position": "WR",
            "recent_team": "MIN", "opponent_team": "GB", "season": 2025, "week": 1,
            "receptions": 5, "receiving_yards": 50, "targets": 8, "carries": 0}
    return pd.DataFrame([{**base, **r} for r in rows])

def test_production_recency_weighting():
    # 10 ppg in 2025 (weight .5), 20 ppg in 2024 (weight .3) -> (10*.5+20*.3)/.8 = 13.75
    wk = _weekly([
        {"season": 2025, "receptions": 10, "receiving_yards": 0},   # 10 pts, 1 game
        {"season": 2024, "receptions": 20, "receiving_yards": 0},   # 20 pts, 1 game
    ])
    prod = factors.production_factor(wk)
    assert abs(prod.iloc[0]["production_raw"] - 13.75) < 1e-6

def test_environment_implied_points():
    sched = pd.DataFrame([{"home_team": "MIN", "away_team": "GB",
                           "total_line": 50.0, "spread_line": 4.0, "week": 1}])
    env = factors.environment_factor(sched)
    env = env.set_index("team")["env_raw"]
    assert env["MIN"] == 27.0 and env["GB"] == 23.0

def test_bye_weeks():
    rows = [{"home_team": "MIN", "away_team": "GB", "week": w} for w in range(1, 19) if w != 7]
    byes = factors.bye_weeks(pd.DataFrame(rows)).set_index("team")["bye"]
    assert byes["MIN"] == 7

def test_normalize_within_position_neutral_fill():
    df = pd.DataFrame({"position": ["WR", "WR", "WR"], "x": [1.0, 3.0, None]})
    out = factors.normalize_within_position(df, "x", "x_n")
    assert out["x_n"].iloc[2] == 50.0
    assert out["x_n"].iloc[1] > out["x_n"].iloc[0]

def test_schedule_factor_points_allowed():
    # GB allowed 30 WR ppg in 2025; MIN plays GB twice in 2026
    wk25 = _weekly([{"receptions": 30, "receiving_yards": 0, "opponent_team": "GB"}])
    sched26 = pd.DataFrame([
        {"home_team": "MIN", "away_team": "GB", "week": 1},
        {"home_team": "GB", "away_team": "MIN", "week": 8},
    ])
    sos = factors.schedule_factor(wk25, sched26)
    row = sos[(sos["team"] == "MIN") & (sos["position"] == "WR")]
    assert row.iloc[0]["sos_raw"] == 30.0
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_factors.py -v` → ImportError.

- [ ] **Step 3: Implement scoring/factors.py**

```python
# scoring/factors.py
import pandas as pd
from scoring.ppr import compute_ppr_points
from scoring.config import RECENCY_WEIGHTS

def player_seasons(weekly: pd.DataFrame) -> pd.DataFrame:
    wk = weekly.copy()
    wk["ppr_points"] = compute_ppr_points(wk)
    grp = wk.groupby(["player_id", "season"]).agg(
        points=("ppr_points", "sum"), games=("week", "nunique"),
        player_name=("player_display_name", "last"),
        position=("position", "last"), team=("recent_team", "last")).reset_index()
    grp["ppg"] = grp["points"] / grp["games"]
    return grp

def production_factor(weekly: pd.DataFrame) -> pd.DataFrame:
    ps = player_seasons(weekly)
    ps["w"] = ps["season"].map(RECENCY_WEIGHTS).fillna(0.0)
    ps["wppg"] = ps["ppg"] * ps["w"]
    agg = ps.groupby("player_id").agg(wsum=("wppg", "sum"), wtot=("w", "sum")).reset_index()
    agg["production_raw"] = agg["wsum"] / agg["wtot"].replace(0, pd.NA)
    return agg[["player_id", "production_raw"]]

def durability_factor(weekly: pd.DataFrame) -> pd.DataFrame:
    ps = player_seasons(weekly)
    first = ps.groupby("player_id")["season"].min()
    games = ps.groupby("player_id")["games"].sum()
    latest = ps["season"].max()
    possible = (latest - first + 1) * 17
    out = (games / possible).rename("durability_raw").reset_index()
    return out[["player_id", "durability_raw"]]

def role_factor(depth_charts: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    dc = depth_charts.copy()
    if "formation" in dc.columns:
        dc = dc[dc["formation"] == "Offense"]
    if dc.empty:
        return pd.DataFrame(columns=["player_id", "role_raw"])
    if "week" in dc.columns and dc["week"].notna().any():
        dc = dc[dc["week"] == dc["week"].max()]
    dc["depth_rank_score"] = 1.0 / pd.to_numeric(dc["depth_team"], errors="coerce").clip(lower=1)
    dc = dc.rename(columns={"gsis_id": "player_id"})[["player_id", "depth_rank_score"]]
    dc = dc.groupby("player_id", as_index=False)["depth_rank_score"].max()

    wk = weekly[weekly["season"] == weekly["season"].max()].copy()
    for c in ("targets", "carries"):
        wk[c] = pd.to_numeric(wk[c], errors="coerce").fillna(0) if c in wk.columns else 0.0
    wk["opps"] = wk["targets"] + wk["carries"]
    team_opps = wk.groupby("recent_team")["opps"].transform("sum")
    wk["opp_share"] = wk["opps"] / team_opps.replace(0, pd.NA)
    share = wk.groupby("player_id", as_index=False)["opp_share"].sum()

    out = dc.merge(share, on="player_id", how="outer")
    # 50/50 blend; if one side missing, use the other alone
    out["role_raw"] = out[["depth_rank_score", "opp_share"]].mean(axis=1, skipna=True)
    return out[["player_id", "role_raw"]]

def environment_factor(schedules: pd.DataFrame) -> pd.DataFrame:
    sc = schedules.dropna(subset=["total_line", "spread_line"]).copy()
    home = pd.DataFrame({"team": sc["home_team"],
                         "implied": (sc["total_line"] + sc["spread_line"]) / 2})
    away = pd.DataFrame({"team": sc["away_team"],
                         "implied": (sc["total_line"] - sc["spread_line"]) / 2})
    both = pd.concat([home, away])
    if both.empty:
        return pd.DataFrame(columns=["team", "env_raw"])
    return both.groupby("team", as_index=False)["implied"].mean().rename(
        columns={"implied": "env_raw"})

def schedule_factor(weekly_prior: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    wk = weekly_prior.copy()
    wk["ppr_points"] = compute_ppr_points(wk)
    weeks = wk.groupby("opponent_team")["week"].nunique().rename("def_games")
    allowed = wk.groupby(["opponent_team", "position"])["ppr_points"].sum().reset_index()
    allowed = allowed.merge(weeks, on="opponent_team")
    allowed["fpa_pg"] = allowed["ppr_points"] / allowed["def_games"]

    games = pd.concat([
        schedules.rename(columns={"home_team": "team", "away_team": "opp"})[["team", "opp"]],
        schedules.rename(columns={"away_team": "team", "home_team": "opp"})[["team", "opp"]]])
    merged = games.merge(allowed, left_on="opp", right_on="opponent_team")
    return merged.groupby(["team", "position"], as_index=False)["fpa_pg"].mean().rename(
        columns={"fpa_pg": "sos_raw"})

def bye_weeks(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules[schedules["week"] <= 18]
    playing = pd.concat([
        reg[["week", "home_team"]].rename(columns={"home_team": "team"}),
        reg[["week", "away_team"]].rename(columns={"away_team": "team"})])
    rows = []
    for team, grp in playing.groupby("team"):
        missing = sorted(set(range(1, 19)) - set(grp["week"]))
        rows.append({"team": team, "bye": missing[0] if missing else None})
    return pd.DataFrame(rows)

def normalize_within_position(df: pd.DataFrame, raw_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = (out.groupby("position")[raw_col].rank(pct=True) * 100)
    out[out_col] = out[out_col].fillna(50.0)
    return out
```

Also append to `scoring/config.py`: `RECENCY_WEIGHTS = {2025: 0.5, 2024: 0.3, 2023: 0.2}`

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_factors.py -v` → all pass. Also run full suite: `.venv/bin/pytest -v`.

- [ ] **Step 5: Commit** — `git add scoring tests && git commit -m "feat: scoring factors (production, durability, role, environment, schedule, byes)"`

---

### Task 5: Composite score, VOR, tiers, league config

**Files:**
- Modify: `scoring/config.py`
- Create: `scoring/composite.py`
- Test: `tests/test_composite.py`

**Interfaces:**
- Consumes: factor columns produced in Task 6's board assembly: `production`, `durability`, `role`, `environment`, `schedule` (each 0–100), plus `position`.
- Produces:
  - `scoring.config.DEFAULT_WEIGHTS: dict[str, float]` = `{"production": 0.35, "role": 0.25, "environment": 0.20, "schedule": 0.10, "durability": 0.10}`
  - `scoring.config.REPLACEMENT_RANK: dict[str, int]` = `{"QB": 9, "RB": 22, "WR": 24, "TE": 10, "K": 9, "DST": 9}` (8 teams; 16 flex slots allocated ≈ RB 6 / WR 8 / TE 2; +1 buffer for QB/K/DST)
  - `compute_composite(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series` — weighted mean of the five factor columns, weights renormalized to sum 1; unknown weight keys raise `ValueError`.
  - `apply_vor(df: pd.DataFrame) -> pd.DataFrame` — adds `vor` = composite − composite of the `REPLACEMENT_RANK[pos]`-th player (by composite desc) at that position (last player if roster smaller).
  - `assign_tiers(df: pd.DataFrame) -> pd.DataFrame` — adds integer `tier` (1 = best) within position: sort by vor desc; new tier where the drop to the next player > mean+std of all consecutive drops at that position (single-player positions → tier 1).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_composite.py
import pandas as pd
import pytest
from scoring.composite import compute_composite, apply_vor, assign_tiers

def _df(n, pos="WR"):
    return pd.DataFrame({
        "position": [pos] * n,
        "production": [100.0 - i * 10 for i in range(n)],
        "durability": [50.0] * n, "role": [50.0] * n,
        "environment": [50.0] * n, "schedule": [50.0] * n})

def test_composite_weighted_mean():
    df = _df(1)
    # equal weights on two factors: (100+50)/2 = 75
    s = compute_composite(df, {"production": 1.0, "role": 1.0})
    assert s.iloc[0] == 75.0

def test_composite_rejects_unknown_key():
    with pytest.raises(ValueError):
        compute_composite(_df(1), {"vibes": 1.0})

def test_vor_replacement_is_zero():
    df = _df(30)
    df["composite"] = compute_composite(df, {"production": 1.0})
    out = apply_vor(df)
    # WR replacement rank is 24 -> the 24th WR has vor == 0
    r = out.sort_values("composite", ascending=False).iloc[23]
    assert r["vor"] == 0.0
    assert out.sort_values("composite", ascending=False).iloc[0]["vor"] > 0

def test_tiers_break_on_gap():
    df = _df(4)
    df["composite"] = [100.0, 99.0, 98.0, 60.0]  # big gap before the last player
    df["vor"] = df["composite"]
    out = assign_tiers(df)
    tiers = out.sort_values("vor", ascending=False)["tier"].tolist()
    assert tiers[0] == tiers[1] == tiers[2] == 1 and tiers[3] == 2
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_composite.py -v` → ImportError.

- [ ] **Step 3: Implement**

Replace `scoring/config.py` with:

```python
# scoring/config.py
CURRENT_SEASON = 2026
HISTORY_SEASONS = [2023, 2024, 2025]
RECENCY_WEIGHTS = {2025: 0.5, 2024: 0.3, 2023: 0.2}

# League: 8 teams, QB/2RB/2WR/TE/2Flex(W-R-T)/K/DST, PPR, 5 bench
LEAGUE_TEAMS = 8

DEFAULT_WEIGHTS = {
    "production": 0.35, "role": 0.25, "environment": 0.20,
    "schedule": 0.10, "durability": 0.10,
}

# starters*8 + flex allocation (16 flex slots ~ RB 6 / WR 8 / TE 2) + 1 buffer for single-slot
REPLACEMENT_RANK = {"QB": 9, "RB": 22, "WR": 24, "TE": 10, "K": 9, "DST": 9}
```

```python
# scoring/composite.py
import pandas as pd
from scoring.config import DEFAULT_WEIGHTS, REPLACEMENT_RANK

FACTORS = list(DEFAULT_WEIGHTS)

def compute_composite(df: pd.DataFrame, weights: dict) -> pd.Series:
    unknown = set(weights) - set(FACTORS)
    if unknown:
        raise ValueError(f"unknown weight keys: {unknown}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    score = pd.Series(0.0, index=df.index)
    for name, w in weights.items():
        score = score + df[name] * (w / total)
    return score

def apply_vor(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vor"] = 0.0
    for pos, grp in out.groupby("position"):
        ranked = grp.sort_values("composite", ascending=False)
        idx = min(REPLACEMENT_RANK.get(pos, 9), len(ranked)) - 1
        replacement = ranked.iloc[idx]["composite"]
        out.loc[grp.index, "vor"] = grp["composite"] - replacement
    return out

def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tier"] = 1
    for pos, grp in out.groupby("position"):
        ranked = grp.sort_values("vor", ascending=False)
        drops = -ranked["vor"].diff().fillna(0)  # positive gaps between consecutive players
        if len(drops) > 2 and drops.std() > 0:
            threshold = drops.mean() + drops.std()
        else:
            threshold = float("inf")
        tier = (drops > threshold).cumsum() + 1
        out.loc[ranked.index, "tier"] = tier
    return out
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_composite.py -v` then full suite.

- [ ] **Step 5: Commit** — `git add scoring tests && git commit -m "feat: composite score, VOR, tiers, league config"`

---

### Task 6: Board assembly

**Files:**
- Create: `scoring/board.py`
- Test: `tests/test_board.py`

**Interfaces:**
- Consumes: `pipeline.db.read_table`, everything from `scoring.factors`, `scoring.composite`, `scoring.config`.
- Produces: `build_board(conn, weights: dict | None = None) -> pd.DataFrame` with columns:
  `player_id, name, position, team, bye, production, durability, role, environment, schedule, composite, vor, tier, adp, edge, rookie, drafted, rank`.
  Sorted by `vor` desc, `rank` = 1..n. This exact shape is the API contract for Task 7.

Assembly logic (document in module docstring):
1. Read tables `weekly`, `depth_charts`, `schedules`, `adp`, `drafted` via `read_table`.
2. Player universe = players in latest-season weekly data ∪ ADP list (any ADP row whose normalized name+position isn't already in the universe becomes a new row — this is how rookies and K/DST enter; depth charts only contribute the role factor). K/DST get all factors neutral (50) except `environment`; DST ADP joins by team.
3. Name join for ADP: `_norm_name(s)` = lowercase, strip periods/apostrophes/hyphens, drop suffixes (jr, sr, ii, iii, iv, v), collapse whitespace. Join on (`_norm_name`, `position`); DST joins on `team`.
4. Factor columns: compute raw factors, merge onto universe, then `normalize_within_position` each into 0–100; missing → 50 (the normalize helper already does this).
5. `rookie` = True where player has no rows in `weekly` (production/durability were neutral-filled).
6. `edge` = ADP rank within universe (ascending adp) − board rank. Positive = market undervalues. NaN adp → edge NaN.
7. `drafted` = player_id in `drafted` table.
8. `composite` via `compute_composite(df, weights or DEFAULT_WEIGHTS)`, then `apply_vor`, `assign_tiers`.

- [ ] **Step 1: Write failing test with an in-memory DuckDB seeded with synthetic tables**

```python
# tests/test_board.py
import pandas as pd
from pipeline.db import get_conn, write_table
from scoring.board import build_board, _norm_name

def _seed(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    weekly = pd.DataFrame([
        {"player_id": "p1", "player_display_name": "Amon-Ra St. Brown", "position": "WR",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 8, "receiving_yards": 90, "targets": 10, "carries": 0}
        for w in range(1, 18)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET", "adp": 5.1},
        {"adp_name": "Rookie Guy", "position": "WR", "team": "GB", "adp": 90.0}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    return conn

def test_norm_name():
    assert _norm_name("Amon-Ra St. Brown") == _norm_name("Amon-Ra St Brown")
    assert _norm_name("Odell Beckham Jr.") == _norm_name("odell beckham")

def test_board_shape_and_join(tmp_path):
    board = build_board(_seed(tmp_path))
    star = board[board["player_id"] == "p1"].iloc[0]
    assert star["adp"] == 5.1               # ADP joined despite punctuation differences
    assert not star["rookie"]
    rook = board[board["name"] == "Rookie Guy"].iloc[0]
    assert rook["rookie"] and rook["production"] == 50.0
    assert list(board["rank"]) == sorted(board["rank"].tolist())
    for col in ["vor", "tier", "composite", "edge", "drafted", "bye"]:
        assert col in board.columns
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement scoring/board.py**

```python
# scoring/board.py
"""Assemble the draft board: universe -> factors -> normalize -> composite -> VOR -> tiers."""
import re
import pandas as pd
from pipeline.db import read_table
from scoring import factors
from scoring.composite import compute_composite, apply_vor, assign_tiers
from scoring.config import DEFAULT_WEIGHTS

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def _norm_name(name: str) -> str:
    s = re.sub(r"[.'\-]", "", str(name).lower())
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return " ".join(parts)

def build_board(conn, weights: dict | None = None) -> pd.DataFrame:
    weekly = read_table(conn, "weekly")
    depth = read_table(conn, "depth_charts")
    sched = read_table(conn, "schedules")
    adp = read_table(conn, "adp")
    drafted = read_table(conn, "drafted")

    # -- universe from weekly history --
    ps = factors.player_seasons(weekly)
    latest = ps[ps["season"] == ps["season"].max()]
    uni = latest[["player_id", "player_name", "position", "team"]].rename(
        columns={"player_name": "name"}).drop_duplicates("player_id")

    # -- add unmatched ADP players (rookies, K, DST) to the universe --
    uni["norm"] = uni["name"].map(_norm_name)
    adp = adp.copy()
    if not adp.empty:
        adp["norm"] = adp["adp_name"].map(_norm_name)
        known = set(zip(uni["norm"], uni["position"]))
        unmatched = adp[~adp.apply(lambda r: (r["norm"], r["position"]) in known, axis=1)]
        extra = unmatched.assign(
            player_id="adp_" + unmatched["norm"].str.replace(" ", "_"),
            name=unmatched["adp_name"])[["player_id", "name", "position", "team", "norm"]]
        uni = pd.concat([uni, extra], ignore_index=True)

    # -- raw factors --
    for raw in [factors.production_factor(weekly), factors.durability_factor(weekly),
                factors.role_factor(depth, weekly) if not depth.empty
                else pd.DataFrame(columns=["player_id", "role_raw"])]:
        uni = uni.merge(raw, on="player_id", how="left")
    env = factors.environment_factor(sched) if not sched.empty else pd.DataFrame(columns=["team", "env_raw"])
    uni = uni.merge(env, on="team", how="left")
    prior = weekly[weekly["season"] == weekly["season"].max()] if not weekly.empty else weekly
    sos = factors.schedule_factor(prior, sched) if not (sched.empty or weekly.empty) \
        else pd.DataFrame(columns=["team", "position", "sos_raw"])
    uni = uni.merge(sos, on=["team", "position"], how="left")
    byes = factors.bye_weeks(sched) if not sched.empty else pd.DataFrame(columns=["team", "bye"])
    uni = uni.merge(byes, on="team", how="left")

    # -- normalize each factor to 0-100 within position (NaN -> 50) --
    for raw_col, col in [("production_raw", "production"), ("durability_raw", "durability"),
                         ("role_raw", "role"), ("env_raw", "environment"), ("sos_raw", "schedule")]:
        if raw_col not in uni.columns:
            uni[raw_col] = pd.NA
        uni = factors.normalize_within_position(uni, raw_col, col)

    uni["rookie"] = ~uni["player_id"].isin(set(weekly.get("player_id", pd.Series(dtype=str))))

    # -- ADP join: name+position for players, team for DST --
    if not adp.empty:
        players_adp = adp[~adp["position"].isin(["DST"])][["norm", "position", "adp"]]
        uni = uni.merge(players_adp, on=["norm", "position"], how="left")
        dst_adp = adp[adp["position"] == "DST"][["team", "adp"]].rename(columns={"adp": "adp_dst"})
        uni = uni.merge(dst_adp, on="team", how="left")
        uni.loc[uni["position"] == "DST", "adp"] = uni["adp_dst"]
        uni = uni.drop(columns=["adp_dst"])
    else:
        uni["adp"] = pd.NA

    # -- score --
    uni["composite"] = compute_composite(uni, weights or DEFAULT_WEIGHTS)
    uni = apply_vor(uni)
    uni = assign_tiers(uni)
    uni = uni.sort_values("vor", ascending=False).reset_index(drop=True)
    uni["rank"] = uni.index + 1
    uni["adp_rank"] = uni["adp"].rank(method="first")
    uni["edge"] = uni["adp_rank"] - uni["rank"]
    drafted_ids = set(drafted["player_id"]) if not drafted.empty else set()
    uni["drafted"] = uni["player_id"].isin(drafted_ids)

    cols = ["rank", "player_id", "name", "position", "team", "bye", "tier",
            "production", "durability", "role", "environment", "schedule",
            "composite", "vor", "adp", "edge", "rookie", "drafted"]
    return uni[cols]
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_board.py -v`, then full suite. Then smoke-test against the real DB from Task 3: `.venv/bin/python -c "from pipeline.db import get_conn; from scoring.board import build_board; b = build_board(get_conn()); print(b.head(25).to_string())"` — verify the top 25 looks like plausible fantasy players (stars near the top).

- [ ] **Step 5: Commit** — `git add scoring tests && git commit -m "feat: draft board assembly with ADP join and rookie handling"`

---

### Task 7: FastAPI backend

**Files:**
- Create: `api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `pipeline.db.get_conn`, `scoring.board.build_board`, `scoring.config.DEFAULT_WEIGHTS`.
- Produces (all under `/api`):
  - `GET /api/players?w_production=&w_role=&w_environment=&w_schedule=&w_durability=` — each weight optional float, defaults from `DEFAULT_WEIGHTS`; returns `{"players": [<board row as dict>...]}` (NaN → null via `df.where(df.notna(), None)` before `to_dict`).
  - `POST /api/drafted/{player_id}` → `{"drafted": true}`; `DELETE /api/drafted/{player_id}` → `{"drafted": false}` (idempotent both ways).
  - `GET /api/meta` → `{"sources": [{source, ok, rows, refreshed_at}...]}`.
  - App factory pattern: `create_app(db_path: str = DEFAULT_PATH) -> FastAPI` so tests can point at a temp DB; module-level `app = create_app()` for uvicorn. Run: `.venv/bin/uvicorn api.main:app --reload`.

- [ ] **Step 1: Write failing tests** (reuse the `_seed` helper pattern from `tests/test_board.py` — copy it, seeding the same four tables)

```python
# tests/test_api.py
import pandas as pd
from fastapi.testclient import TestClient
from pipeline.db import get_conn, write_table
from api.main import create_app

def _seed(path):
    conn = get_conn(path)
    weekly = pd.DataFrame([
        {"player_id": "p1", "player_display_name": "A Star", "position": "WR",
         "recent_team": "DET", "opponent_team": "GB", "season": 2025, "week": w,
         "receptions": 8, "receiving_yards": 90, "targets": 10, "carries": 0}
        for w in range(1, 18)])
    write_table(conn, "weekly", weekly)
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "A Star", "position": "WR", "team": "DET", "adp": 5.1}]))
    write_table(conn, "depth_charts", pd.DataFrame(
        columns=["gsis_id", "depth_team", "formation", "week", "position"]))
    conn.close()

def _client(tmp_path):
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    return TestClient(create_app(path))

def test_players_endpoint(tmp_path):
    c = _client(tmp_path)
    body = c.get("/api/players").json()
    assert body["players"][0]["name"] == "A Star"
    assert body["players"][0]["adp"] == 5.1

def test_players_custom_weights(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/players", params={"w_production": 1.0, "w_role": 0.0,
                                      "w_environment": 0.0, "w_schedule": 0.0,
                                      "w_durability": 0.0})
    assert r.status_code == 200

def test_drafted_roundtrip(tmp_path):
    c = _client(tmp_path)
    pid = c.get("/api/players").json()["players"][0]["player_id"]
    assert c.post(f"/api/drafted/{pid}").json() == {"drafted": True}
    assert any(p["drafted"] for p in c.get("/api/players").json()["players"])
    assert c.delete(f"/api/drafted/{pid}").json() == {"drafted": False}

def test_meta(tmp_path):
    c = _client(tmp_path)
    assert "sources" in c.get("/api/meta").json()
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement api/main.py**

```python
# api/main.py
import pandas as pd
from fastapi import FastAPI
from pipeline.db import get_conn, read_table, DEFAULT_PATH
from scoring.board import build_board
from scoring.config import DEFAULT_WEIGHTS

def create_app(db_path: str = DEFAULT_PATH) -> FastAPI:
    app = FastAPI(title="Draft Board API")
    conn = get_conn(db_path)

    @app.get("/api/players")
    def players(w_production: float = DEFAULT_WEIGHTS["production"],
                w_role: float = DEFAULT_WEIGHTS["role"],
                w_environment: float = DEFAULT_WEIGHTS["environment"],
                w_schedule: float = DEFAULT_WEIGHTS["schedule"],
                w_durability: float = DEFAULT_WEIGHTS["durability"]):
        weights = {"production": w_production, "role": w_role,
                   "environment": w_environment, "schedule": w_schedule,
                   "durability": w_durability}
        board = build_board(conn, weights)
        # astype(object) first, else float columns silently revert None -> NaN
        # and FastAPI's JSON encoder rejects NaN
        board = board.astype(object).where(board.notna(), None)
        return {"players": board.to_dict(orient="records")}

    @app.post("/api/drafted/{player_id}")
    def draft(player_id: str):
        conn.execute("INSERT OR IGNORE INTO drafted VALUES (?)", [player_id])
        return {"drafted": True}

    @app.delete("/api/drafted/{player_id}")
    def undraft(player_id: str):
        conn.execute("DELETE FROM drafted WHERE player_id = ?", [player_id])
        return {"drafted": False}

    @app.get("/api/meta")
    def meta():
        m = read_table(conn, "meta")
        m["refreshed_at"] = m["refreshed_at"].astype(str)
        m = m.astype(object).where(m.notna(), None)
        return {"sources": m.to_dict(orient="records")}

    return app

app = create_app()
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_api.py -v`, then full suite. Smoke: `.venv/bin/uvicorn api.main:app --port 8000 &`, `curl -s localhost:8000/api/players | head -c 400`, kill it.

- [ ] **Step 5: Commit** — `git add api tests && git commit -m "feat: FastAPI board endpoints with drafted state"`

---

### Task 8: React app scaffold + board table

**Files:**
- Create: `web/` (Vite react-ts scaffold), `web/src/api.ts`, `web/src/App.tsx`, `web/src/components/PlayerTable.tsx`, modify `web/vite.config.ts`

**Interfaces:**
- Consumes: `GET /api/players` shape from Task 7.
- Produces: `Player` TS type and `fetchPlayers(weights)` in `web/src/api.ts`; `<PlayerTable players onToggleDrafted>` component. Task 9 adds controls around them.

- [ ] **Step 1: Scaffold**

```bash
npm create vite@latest web -- --template react-ts
cd web && npm install && npm install @tanstack/react-table
```

- [ ] **Step 2: Proxy `/api` to FastAPI** — in `web/vite.config.ts`:

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
```

- [ ] **Step 3: API client** — `web/src/api.ts`:

```ts
export interface Player {
  rank: number; player_id: string; name: string; position: string;
  team: string; bye: number | null; tier: number;
  production: number; durability: number; role: number;
  environment: number; schedule: number;
  composite: number; vor: number; adp: number | null; edge: number | null;
  rookie: boolean; drafted: boolean;
}

export interface Weights {
  production: number; role: number; environment: number;
  schedule: number; durability: number;
}

export const DEFAULT_WEIGHTS: Weights = {
  production: 0.35, role: 0.25, environment: 0.2, schedule: 0.1, durability: 0.1,
}

export async function fetchPlayers(w: Weights): Promise<Player[]> {
  const params = new URLSearchParams(
    Object.entries(w).map(([k, v]) => [`w_${k}`, String(v)]))
  const res = await fetch(`/api/players?${params}`)
  return (await res.json()).players
}

export async function setDrafted(playerId: string, drafted: boolean): Promise<void> {
  await fetch(`/api/drafted/${playerId}`, { method: drafted ? 'POST' : 'DELETE' })
}

export async function fetchMeta(): Promise<{ sources: { source: string; ok: boolean; rows: number; refreshed_at: string }[] }> {
  return (await fetch('/api/meta')).json()
}
```

- [ ] **Step 4: PlayerTable component** — `web/src/components/PlayerTable.tsx` using @tanstack/react-table: columns Rank, Tier, Name (+ "R" badge if rookie), Pos, Team, Bye, VOR (1 decimal), Composite (1 decimal), ADP, Edge (green if > 0, red if < 0). Row `onClick` calls `onToggleDrafted(player)`; drafted rows get `opacity: 0.35; text-decoration: line-through`. Header click sorts (use `getSortedRowModel`). Props: `{ players: Player[]; onToggleDrafted: (p: Player) => void }`.

- [ ] **Step 5: Minimal App.tsx** — load players with `DEFAULT_WEIGHTS` on mount, render `<PlayerTable>`, toggle drafted optimistically then call `setDrafted` and refetch.

- [ ] **Step 6: Verify** — `cd web && npm run build` (typechecks), then with the API running (`.venv/bin/uvicorn api.main:app`) run `npm run dev` and confirm the board renders real players in the browser.

- [ ] **Step 7: Commit** — `git add web && git commit -m "feat: react draft board with sortable table"`

---

### Task 9: Draft-night controls + README

**Files:**
- Create: `web/src/components/WeightSliders.tsx`, `web/src/components/PositionTabs.tsx`, `web/src/components/FreshnessBadge.tsx`
- Modify: `web/src/App.tsx`, `README.md`

**Interfaces:**
- Consumes: `fetchPlayers`, `fetchMeta`, `Weights`, `Player` from `web/src/api.ts`.
- Produces: finished app.

- [ ] **Step 1: WeightSliders** — one `<input type="range" min="0" max="1" step="0.05">` per weight key with the current value shown; `onChange` updates local weights state; App debounces 300ms then refetches players. Props: `{ weights: Weights; onChange: (w: Weights) => void }`.

- [ ] **Step 2: PositionTabs** — buttons `ALL | QB | RB | WR | TE | FLEX | K | DST`; FLEX filters to RB/WR/TE. Props: `{ value: string; onChange: (v: string) => void }`. Filtering happens in App (client-side).

- [ ] **Step 3: FreshnessBadge** — on mount fetch `/api/meta`; render one chip per source: green if `ok`, red if not, tooltip = `rows` + `refreshed_at`. Warn visibly if `adp` or `schedules` failed (draft-critical sources).

- [ ] **Step 4: App composition** — layout: header (title + FreshnessBadge), left sidebar (WeightSliders + "hide drafted" checkbox), main (PositionTabs + PlayerTable). Keep styling minimal-but-clean in `App.css`: dark background, monospace numerals, tier boundaries shown as a subtle horizontal rule between tier groups when sorted by rank.

- [ ] **Step 5: README** — overwrite with: what this is, setup (`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`, `cd web && npm install`), usage (1. `python -m pipeline.refresh`, 2. `uvicorn api.main:app`, 3. `npm run dev` in `web/`), how scoring works (factors → weights → VOR → tiers, league config in `scoring/config.py`), data sources and their quirks (ADP is 12-team consensus; K/DST scored on environment only).

- [ ] **Step 6: Verify** — `cd web && npm run build`; full pytest suite; manual browser check: sliders reorder the board, tabs filter, clicking a row greys it out and survives page reload.

- [ ] **Step 7: Commit** — `git add web README.md && git commit -m "feat: weight sliders, position tabs, freshness badge, README"`
