# Player Profile Drawer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A slide-over player profile drawer: full history (season summaries, weekly chart, game log), 2026 outlook, and cross-year similar player-seasons with next-season outcomes.

**Architecture:** Two new pure modules (`scoring/similarity.py`, `scoring/profile.py`) compute everything from the existing DuckDB tables; one new FastAPI endpooint composes them; the React app gets a drawer that replaces the row-expand chevron and takes over row-click (drafted toggling moves to a dedicated button).

**Tech Stack:** pandas 1.5.3 / numpy / DuckDB / FastAPI (existing venv), React 18 + TS + hand-rolled SVG chart (no chart lib).

**Spec:** `docs/superpowers/specs/2026-08-02-player-profile-design.md`

## Global Constraints

- pandas is 1.5.3 — NEVER use `.replace(0, pd.NA)` (RecursionError); use `.replace(0, np.nan)`.
- Scoring modules are pure dataframe functions; `build_profile` may call `read_table` (same I/O boundary rule as `build_board`).
- Similarity constants (exact): `MIN_GAMES = 4`, `DECAY = 2.0`, features `["ppg", "games", "target_share", "carry_share", "yards_per_opp", "td_per_opp", "rec_pg"]`, weights 2.0 for `ppg`/`target_share`/`carry_share`, 1.0 for the rest; similarity = `100 * exp(-distance / DECAY)`; candidates are same-position player-seasons 2023–2025 with ≥ MIN_GAMES games, excluding ALL of the target player's own seasons; target vector = the player's latest season with ≥ MIN_GAMES games.
- JSON responses must never contain NaN — scrub to `None` before returning.
- All API routes under `/api`; per-request `conn.cursor()` pattern (see `api/main.py` existing routes).
- The board's expandable factor-breakdown chevron is REMOVED (superseded by the drawer — approved design decision). Row-click opens the profile; drafted toggling moves to a dedicated button cell and the drawer header.
- Use `.venv/bin/pytest`. Commit after each task with a conventional-commit message.
- Real data lives in `data/nfl.duckdb` (~700-player board); use it for smoke tests, never for unit tests (unit tests use synthetic frames / seeded temp DBs).

---

### Task 1: Cross-year similarity engine

**Files:**
- Create: `scoring/similarity.py`
- Test: `tests/test_similarity.py`

**Interfaces:**
- Consumes: `scoring.ppr.compute_ppr_points`.
- Produces:
  - `player_season_features(weekly: pd.DataFrame) -> pd.DataFrame` — one row per (player_id, season) with columns: `player_id, season, name, position, games, points, ppg, targets, carries, rec_yards, rush_yards, tds, receptions, team_targets, team_carries, target_share, carry_share, yards_per_opp, td_per_opp, rec_pg`. Share/efficiency columns are NaN when the denominator is 0.
  - `find_twins(weekly: pd.DataFrame, player_id: str, top_n: int = 5) -> dict | None` — None when the player has no season with ≥ MIN_GAMES games; else `{"mode": "stat_twins", "target_season": int, "players": [{"player_id", "name", "season", "similarity", "ppg", "next_ppg"} ...]}` sorted most-similar first. `similarity` rounded to 1 decimal; `next_ppg` is that player's following-season ppg or None.
  - `value_neighbors(board: pd.DataFrame, player_id: str, top_n: int = 5) -> dict` — `{"mode": "value_neighbors", "players": [{"player_id", "name", "season": None, "similarity": None, "ppg": None, "next_ppg": None, "rank", "adp"} ...]}` — same-position board players nearest in `vor`, excluding self, sorted by |vor diff|.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_similarity.py
import pandas as pd
from scoring.similarity import player_season_features, find_twins, value_neighbors

def _wk(pid, name, season, games, rec, yds, tgt, team="AAA", pos="WR"):
    """games identical weekly lines for one player-season."""
    return [{"player_id": pid, "player_display_name": name, "position": pos,
             "recent_team": team, "opponent_team": "ZZZ", "season": season,
             "week": w, "receptions": rec, "receiving_yards": yds,
             "targets": tgt, "carries": 0}
            for w in range(1, games + 1)]

def test_features_shares_and_ppg():
    # one player is the whole team: target_share == 1.0
    wk = pd.DataFrame(_wk("p1", "A", 2025, 4, rec=5, yds=50, tgt=8))
    f = player_season_features(wk).iloc[0]
    assert f["games"] == 4 and f["target_share"] == 1.0
    assert abs(f["ppg"] - 10.0) < 1e-9          # 5 + 50*0.1
    assert abs(f["yards_per_opp"] - 50 / 8) < 1e-9

def test_clone_is_top_twin_with_100():
    rows = (_wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9, team="AAA")
            + _wk("clone", "Clone", 2023, 10, rec=6, yds=80, tgt=9, team="BBB")
            + _wk("other", "Other", 2024, 10, rec=2, yds=20, tgt=3, team="CCC"))
    out = find_twins(pd.DataFrame(rows), "me")
    assert out["mode"] == "stat_twins" and out["target_season"] == 2025
    top = out["players"][0]
    assert top["player_id"] == "clone" and top["similarity"] == 100.0

def test_own_seasons_excluded():
    rows = _wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9) + _wk("me", "Me", 2024, 10, rec=6, yds=80, tgt=9) \
           + _wk("x", "X", 2024, 10, rec=5, yds=70, tgt=8, team="BBB")
    out = find_twins(pd.DataFrame(rows), "me")
    assert all(p["player_id"] != "me" for p in out["players"])

def test_next_ppg():
    rows = (_wk("me", "Me", 2025, 10, rec=6, yds=80, tgt=9)
            + _wk("x", "X", 2024, 10, rec=6, yds=80, tgt=9, team="BBB")
            + _wk("x", "X", 2025, 10, rec=10, yds=100, tgt=12, team="BBB"))
    out = find_twins(pd.DataFrame(rows), "me")
    x2024 = next(p for p in out["players"] if p["season"] == 2024)
    assert abs(x2024["next_ppg"] - 20.0) < 1e-9  # X's 2025: 10 rec + 100*0.1

def test_min_games_filter_returns_none():
    wk = pd.DataFrame(_wk("me", "Me", 2025, 2, rec=6, yds=80, tgt=9))
    assert find_twins(wk, "me") is None

def test_value_neighbors():
    board = pd.DataFrame({
        "player_id": ["a", "b", "c", "d"], "name": ["A", "B", "C", "D"],
        "position": ["WR", "WR", "WR", "RB"],
        "vor": [10.0, 9.0, 1.0, 9.5], "rank": [1, 2, 3, 4],
        "adp": [5.0, 8.0, 90.0, 6.0]})
    out = value_neighbors(board, "a", top_n=2)
    ids = [p["player_id"] for p in out["players"]]
    assert out["mode"] == "value_neighbors"
    assert ids == ["b", "c"]  # same position only, nearest vor first, self excluded
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_similarity.py -v` → ImportError.

- [ ] **Step 3: Implement scoring/similarity.py**

```python
# scoring/similarity.py
"""Cross-year player-season similarity (stat twins) and board value-neighbors."""
import numpy as np
import pandas as pd
from scoring.ppr import compute_ppr_points

FEATURES = ["ppg", "games", "target_share", "carry_share",
            "yards_per_opp", "td_per_opp", "rec_pg"]
FEATURE_WEIGHTS = {"ppg": 2.0, "target_share": 2.0, "carry_share": 2.0,
                   "games": 1.0, "yards_per_opp": 1.0, "td_per_opp": 1.0,
                   "rec_pg": 1.0}
MIN_GAMES = 4
DECAY = 2.0

_STAT_COLS = ["targets", "carries", "receiving_yards", "rushing_yards",
              "receiving_tds", "rushing_tds", "receptions"]

def player_season_features(weekly: pd.DataFrame) -> pd.DataFrame:
    wk = weekly.copy()
    wk["ppr_points"] = compute_ppr_points(wk)
    for c in _STAT_COLS:
        wk[c] = (pd.to_numeric(wk[c], errors="coerce").fillna(0)
                 if c in wk.columns else 0.0)
    for c in ("targets", "carries"):
        wk[f"_team_{c}"] = wk.groupby(["season", "recent_team"])[c].transform("sum")
    g = wk.groupby(["player_id", "season"]).agg(
        name=("player_display_name", "last"), position=("position", "last"),
        games=("week", "nunique"), points=("ppr_points", "sum"),
        targets=("targets", "sum"), carries=("carries", "sum"),
        rec_yards=("receiving_yards", "sum"), rush_yards=("rushing_yards", "sum"),
        rec_tds=("receiving_tds", "sum"), rush_tds=("rushing_tds", "sum"),
        receptions=("receptions", "sum"),
        team_targets=("_team_targets", "last"),
        team_carries=("_team_carries", "last")).reset_index()
    opps = (g["targets"] + g["carries"]).replace(0, np.nan)
    g["ppg"] = g["points"] / g["games"]
    g["tds"] = g["rec_tds"] + g["rush_tds"]
    g["target_share"] = g["targets"] / g["team_targets"].replace(0, np.nan)
    g["carry_share"] = g["carries"] / g["team_carries"].replace(0, np.nan)
    g["yards_per_opp"] = (g["rec_yards"] + g["rush_yards"]) / opps
    g["td_per_opp"] = g["tds"] / opps
    g["rec_pg"] = g["receptions"] / g["games"]
    return g

def find_twins(weekly: pd.DataFrame, player_id: str, top_n: int = 5) -> dict | None:
    feats = player_season_features(weekly)
    feats = feats[feats["games"] >= MIN_GAMES]
    mine = feats[feats["player_id"] == player_id]
    if mine.empty:
        return None
    target = mine.sort_values("season").iloc[-1]
    pool = feats[feats["position"] == target["position"]].copy()
    zcols = []
    for f in FEATURES:
        col = pool[f].astype(float)
        std = col.std()
        z = f + "_z"
        if pd.isna(std) or std == 0:
            pool[z] = 0.0
        else:
            pool[z] = ((col - col.mean()) / std).fillna(0.0)
        zcols.append(z)
    w = np.array([FEATURE_WEIGHTS[f] for f in FEATURES])
    tvec = pool.loc[(pool["player_id"] == player_id)
                    & (pool["season"] == target["season"]), zcols
                    ].iloc[0].to_numpy(dtype=float)
    cand = pool[pool["player_id"] != player_id].copy()
    if cand.empty:
        return {"mode": "stat_twins", "target_season": int(target["season"]),
                "players": []}
    diffs = cand[zcols].to_numpy(dtype=float) - tvec
    cand["distance"] = np.sqrt(((diffs ** 2) * w).sum(axis=1) / w.sum())
    cand["similarity"] = (100 * np.exp(-cand["distance"] / DECAY)).round(1)
    nxt = feats[["player_id", "season", "ppg"]].copy()
    nxt["season"] = nxt["season"] - 1
    nxt = nxt.rename(columns={"ppg": "next_ppg"})
    cand = cand.merge(nxt, on=["player_id", "season"], how="left")
    cand = cand.sort_values("distance").head(top_n)
    players = [{"player_id": r["player_id"], "name": r["name"],
                "season": int(r["season"]),
                "similarity": float(r["similarity"]),
                "ppg": round(float(r["ppg"]), 1),
                "next_ppg": (None if pd.isna(r["next_ppg"])
                             else round(float(r["next_ppg"]), 1))}
               for _, r in cand.iterrows()]
    return {"mode": "stat_twins", "target_season": int(target["season"]),
            "players": players}

def value_neighbors(board: pd.DataFrame, player_id: str, top_n: int = 5) -> dict:
    me = board[board["player_id"] == player_id].iloc[0]
    pool = board[(board["position"] == me["position"])
                 & (board["player_id"] != player_id)].copy()
    pool["_d"] = (pool["vor"] - me["vor"]).abs()
    pool = pool.sort_values("_d").head(top_n)
    players = [{"player_id": r["player_id"], "name": r["name"], "season": None,
                "similarity": None, "ppg": None, "next_ppg": None,
                "rank": int(r["rank"]),
                "adp": (None if pd.isna(r["adp"]) else float(r["adp"]))}
               for _, r in pool.iterrows()]
    return {"mode": "value_neighbors", "players": players}
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_similarity.py -v`, then full suite (`.venv/bin/pytest -q`, expect 49 + 6 new).

- [ ] **Step 5: Commit** — `git add scoring/similarity.py tests/test_similarity.py && git commit -m "feat: cross-year player-season similarity engine"`

---

### Task 2: Profile assembly (seasons, game log, outlook, payload)

**Files:**
- Create: `scoring/profile.py`
- Test: `tests/test_profile.py`

**Interfaces:**
- Consumes: `scoring.similarity.player_season_features/find_twins/value_neighbors`, `scoring.board.build_board` and `scoring.board._norm_name`, `scoring.factors.environment_factor/schedule_factor/bye_weeks`, `pipeline.db.read_table`.
- Produces: `build_profile(conn, player_id: str, weights: dict | None = None) -> dict | None` — None for unknown ids; else the exact spec payload (`header`, `factors`, `seasons`, `game_log`, `outlook`, `similar`), fully JSON-safe (no NaN/numpy types). Helpers (also exported for tests): `season_summaries(weekly, snaps, player_id) -> list[dict]`, `game_log(weekly, player_id) -> list[dict]`.

Behavior details:
- `season_summaries`: rows from `player_season_features` for this player (all seasons, no MIN_GAMES filter), each dict: `season, games, ppg, targets, target_share, carries, rec_yards, rush_yards, tds, receptions, yards_per_opp, snap_share`. `snap_share` = mean `offense_pct` from `snap_counts` joined on (`_norm_name(player) == _norm_name(name)`, team, season); None when unmatched or snaps empty. Round floats to 3 decimals (shares) / 1 decimal (ppg, yards_per_opp).
- `game_log`: one dict per weekly row, sorted season desc then week desc: `season, week, opponent, stat_line, ppr_points` (1 decimal). `stat_line` by position — QB: `"C/A, PYDS yds, PTD TD, INT INT"` + `" · N car, YDS yds"` when carries > 0; RB: `"N car, YDS yds, RTD TD"` + `" · R rec, RYDS yds"` when targets > 0; WR/TE: `"T tgt, R rec, YDS yds, RTD TD"` + `" · N car, YDS yds"` when carries > 0. Missing stat columns count as 0; integers formatted without decimals.
- `outlook`: `depth_slot` = min adapted depth rank for this player (reuse `scoring.board._adapt_depth_charts`, match on `gsis_id == player_id`; None if absent/empty), `implied_points` = team's `env_raw` (1 decimal), `sos_raw` = (team, position) `sos_raw` (1 decimal), `sos_pct` = percentile of that team's sos among all teams for that position (0–100, 1 decimal), `bye`. All None-safe when schedules/depth empty.
- `build_profile`: board row lookup (None → caller 404s); `factors` = the five factor columns off the board row; K/DST or no qualifying stat season → `value_neighbors(board, ...)`; else `find_twins` enriched: for each twin whose `player_id` is on the current board, add `rank` and `adp` (else None). Scrub every value: numpy scalars → python (`.item()`), NaN → None (write a `_scrub(value)` recursive helper for dicts/lists).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_profile.py
import pandas as pd
from pipeline.db import get_conn, write_table
from scoring.profile import season_summaries, game_log, build_profile

def _weekly_rows():
    return pd.DataFrame(
        [{"player_id": "p1", "player_display_name": "Amon-Ra St. Brown",
          "position": "WR", "recent_team": "DET", "opponent_team": "GB",
          "season": 2025, "week": w, "receptions": 6, "receiving_yards": 80,
          "receiving_tds": 1, "targets": 9, "carries": 0}
         for w in range(1, 11)])

def test_season_summaries_math_and_snap_join():
    snaps = pd.DataFrame([{"player": "Amon-Ra St Brown", "team": "DET",
                           "season": 2025, "offense_pct": 0.9}])
    s = season_summaries(_weekly_rows(), snaps, "p1")[0]
    assert s["season"] == 2025 and s["games"] == 10
    assert s["ppg"] == 20.0            # 6 + 8 + 6 = 20 per game
    assert s["snap_share"] == 0.9      # matched despite punctuation
    assert s["target_share"] == 1.0

def test_game_log_line_and_order():
    rows = game_log(_weekly_rows(), "p1")
    assert rows[0]["week"] == 10       # newest first
    assert rows[0]["stat_line"] == "9 tgt, 6 rec, 80 yds, 1 TD"
    assert rows[0]["ppr_points"] == 20.0

def _seed(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "weekly", _weekly_rows())
    write_table(conn, "schedules", pd.DataFrame([
        {"home_team": "DET", "away_team": "GB", "week": 1,
         "total_line": 51.0, "spread_line": 3.0}]))
    write_table(conn, "adp", pd.DataFrame([
        {"adp_name": "Amon-Ra St Brown", "position": "WR", "team": "DET",
         "adp": 5.1},
        {"adp_name": "Rookie Guy", "position": "WR", "team": "GB", "adp": 90.0}]))
    write_table(conn, "depth_charts",
                pd.DataFrame(columns=["gsis_id", "depth_team", "formation",
                                      "week", "position"]))
    write_table(conn, "snap_counts",
                pd.DataFrame(columns=["player", "team", "season", "offense_pct"]))
    return conn

def test_build_profile_shape(tmp_path):
    p = build_profile(_seed(tmp_path), "p1")
    assert p["header"]["name"] == "Amon-Ra St. Brown"
    assert set(p["factors"]) == {"production", "durability", "role",
                                 "environment", "schedule"}
    assert p["seasons"][0]["games"] == 10
    assert len(p["game_log"]) == 10
    assert p["outlook"]["implied_points"] == 27.0
    assert p["outlook"]["bye"] is None or isinstance(p["outlook"]["bye"], int)

def test_build_profile_unknown_and_rookie(tmp_path):
    from scoring.board import build_board
    conn = _seed(tmp_path)
    assert build_profile(conn, "nope") is None
    # rookie (ADP-only): empty history, value-neighbors fallback
    board = build_board(conn)
    rk = board[board["name"] == "Rookie Guy"].iloc[0]["player_id"]
    prof = build_profile(conn, rk)
    assert prof["seasons"] == [] and prof["game_log"] == []
    assert prof["similar"]["mode"] == "value_neighbors"

def test_no_nan_anywhere(tmp_path):
    import math, json
    p = build_profile(_seed(tmp_path), "p1")
    json.dumps(p, allow_nan=False)  # raises if any NaN survived scrubbing
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement scoring/profile.py** — follow the Behavior details above. Skeleton:

```python
# scoring/profile.py
"""Assemble the player-profile payload: history, outlook, similar seasons."""
import math
import numpy as np
import pandas as pd
from pipeline.db import read_table
from scoring import factors
from scoring.board import build_board, _norm_name, _adapt_depth_charts
from scoring.similarity import (player_season_features, find_twins,
                                value_neighbors)

def _scrub(v):
    if isinstance(v, dict):
        return {k: _scrub(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_scrub(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, np.bool_):
        return bool(v)
    return v

def _num(row, col):
    v = row.get(col, 0)
    return 0 if v is None or (isinstance(v, float) and math.isnan(v)) else v
```

`_stat_line(row, position)` per the Behavior details (use `_num`, format ints with `int(...)`). `season_summaries` filters `player_season_features(weekly)` to the player and left-joins snap share (`snaps` grouped by (norm name, team, season) mean `offense_pct`). `game_log` sorts `[weekly.player_id == player_id]` by season/week desc and builds dicts. `outlook(weekly, depth, schedules, player_row)` computes the four values (guard every empty frame). `build_profile(conn, player_id, weights=None)` composes and `_scrub`s the final dict; `header` is the board row `to_dict()`.

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_profile.py -v`, then full suite. Real-data smoke (write a scratch script if easier):

```python
from pipeline.db import get_conn
from scoring.board import build_board
from scoring.profile import build_profile
import json
conn = get_conn()
pid = build_board(conn).iloc[0]["player_id"]
print(json.dumps(build_profile(conn, pid), indent=1, allow_nan=False)[:1500])
```

Inspect: seasons populated for the top player, twins list has 5 plausible names/seasons, no crash, no NaN error.

- [ ] **Step 5: Commit** — `git add scoring/profile.py tests/test_profile.py && git commit -m "feat: player profile assembly with outlook and twins"`

---

### Task 3: Profile API endpoint

**Files:**
- Modify: `api/main.py`
- Test: `tests/test_api.py` (append)

**Interfaces:**
- Consumes: `scoring.profile.build_profile`.
- Produces: `GET /api/players/{player_id}/profile` → the payload verbatim; 404 `{"detail": "unknown player_id"}` for unknown ids.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_api.py`, reuse its `_seed`/`_client` helpers — add `snap_counts` empty-table seeding to `_seed` if absent)

```python
def test_profile_endpoint(tmp_path):
    c = _client(tmp_path)
    pid = c.get("/api/players").json()["players"][0]["player_id"]
    p = c.get(f"/api/players/{pid}/profile")
    assert p.status_code == 200
    body = p.json()
    assert body["header"]["player_id"] == pid
    assert {"factors", "seasons", "game_log", "outlook", "similar"} <= set(body)

def test_profile_404(tmp_path):
    assert _client(tmp_path).get("/api/players/nope/profile").status_code == 404
```

- [ ] **Step 2: Run to verify failure** — 404 test may pass by accident; the shape test fails (405/404 route missing).

- [ ] **Step 3: Implement** (inside `create_app`, following the existing cursor pattern)

```python
    @app.get("/api/players/{player_id}/profile")
    def player_profile(player_id: str):
        cur = conn.cursor()
        try:
            payload = build_profile(cur, player_id)
            if payload is None:
                raise HTTPException(status_code=404, detail="unknown player_id")
            return payload
        finally:
            cur.close()
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_api.py -v`, full suite. Smoke: run uvicorn, `curl -s localhost:8000/api/players | python3 -c "import sys,json; print(json.load(sys.stdin)['players'][0]['player_id'])"`, curl that id's `/profile`, confirm JSON, kill server.

- [ ] **Step 5: Commit** — `git add api/main.py tests/test_api.py && git commit -m "feat: player profile endpoint"`

---

### Task 4: Drawer shell + interaction rework

**Files:**
- Create: `web/src/components/PlayerProfile.tsx`, `web/src/components/FactorBars.tsx`, `web/src/components/SimilarPlayers.tsx`
- Modify: `web/src/api.ts`, `web/src/App.tsx`, `web/src/components/PlayerTable.tsx`, `web/src/App.css`

**Interfaces:**
- Consumes: `GET /api/players/{id}/profile` (Task 3 payload).
- Produces for Task 5: `PlayerProfile` renders sections in order with placeholders `<section id="weekly-chart" />`, seasons/game-log slots that Task 5 fills; exported types in `api.ts`: `SeasonSummary`, `GameLogRow`, `SimilarPlayer`, `Outlook`, `PlayerProfileData`, and `fetchProfile(playerId: string): Promise<PlayerProfileData>`.

- [ ] **Step 1: api.ts additions**

```ts
export interface SeasonSummary {
  season: number; games: number; ppg: number; targets: number;
  target_share: number | null; carries: number; rec_yards: number;
  rush_yards: number; tds: number; receptions: number;
  yards_per_opp: number | null; snap_share: number | null;
}
export interface GameLogRow {
  season: number; week: number; opponent: string; stat_line: string;
  ppr_points: number;
}
export interface SimilarPlayer {
  player_id: string | null; name: string; season: number | null;
  similarity: number | null; ppg: number | null; next_ppg: number | null;
  rank: number | null; adp: number | null;
}
export interface Outlook {
  depth_slot: number | null; implied_points: number | null;
  sos_raw: number | null; sos_pct: number | null; bye: number | null;
}
export interface PlayerProfileData {
  header: Player;
  factors: { production: number; durability: number; role: number;
             environment: number; schedule: number };
  seasons: SeasonSummary[];
  game_log: GameLogRow[];
  outlook: Outlook;
  similar: { mode: 'stat_twins' | 'value_neighbors'; players: SimilarPlayer[] };
}
export async function fetchProfile(playerId: string): Promise<PlayerProfileData> {
  const res = await fetch(`/api/players/${playerId}/profile`)
  if (!res.ok) throw new Error(`profile ${res.status}: ${await detailText(res)}`)
  return res.json()
}
```

- [ ] **Step 2: PlayerTable rework** — REMOVE the expand chevron column and `factor-breakdown-row` rendering (and the `expanded` state); row `onClick` now calls a new `onSelectPlayer(p: Player)` prop; ADD a narrow "✓" button column (title "toggle drafted", `stopPropagation`) that calls `onToggleDrafted(p)`. Keep `showTierBreaks` behavior unchanged. Update `App.tsx` props accordingly.

- [ ] **Step 3: PlayerProfile drawer** — props `{ playerId: string; onClose: () => void; onToggleDrafted: (p: Player) => void; onSelectPlayer: (id: string) => void }`. Fetches on `playerId` change (loading + error states inside the drawer). Fixed-position right drawer, `width: min(70vw, 900px)`, full height, scrollable, above the board; semi-transparent backdrop div (click → onClose); `Esc` keydown → onClose (add/remove listener in effect). Header: name, pos/team/bye, rookie badge, chips (Rank, Tier, VOR, Composite, ADP, Edge), "Mark drafted"/"Undo draft" button reflecting `header.drafted`. Renders `<FactorBars factors={…} />`, `<section id="weekly-chart" />` placeholder, seasons/game-log placeholder sections ("coming in next task" text acceptable this task), outlook chips (depth slot as `WR2`-style using header.position + depth_slot, implied points, SoS with percentile, bye), `<SimilarPlayers …>`.
- [ ] **Step 4: FactorBars** — five labeled horizontal bars, 0–100 scaling, value shown right-aligned; reuse the accent palette from App.css.
- [ ] **Step 5: SimilarPlayers** — for `stat_twins`: line per comp — `Name '24 · 97.2 sim · 15.1 ppg → 18.3 next` plus `#12 / ADP 20.4` chips when on the board; whole line clickable when `player_id` present → `onSelectPlayer(player_id)`. For `value_neighbors`: heading "Similar draft value (no stat history)" and name + rank/ADP lines. `next_ppg` null renders `→ —`.
- [ ] **Step 6: App wiring** — `selectedPlayerId: string | null` state; row click sets it; drawer's comp click swaps it; after a drafted toggle from the drawer, refetch players AND re-fetch the open profile (drafted flag in header stays true). Board sort/filter/scroll must be untouched by opening/closing (drawer is an overlay, no route change).
- [ ] **Step 7: Verify** — `cd web && npm run build` clean; browser check (API + dev server, Playwright): click row → drawer opens with header/factors/similar; Esc closes; comp click swaps profile; ✓ button (not row click) toggles drafted; drawer "Mark drafted" works and board reflects it.
- [ ] **Step 8: Commit** — `git add web && git commit -m "feat: player profile drawer with factors and similar players"`

---

### Task 5: History visuals (weekly chart, season table, game log) + polish

**Files:**
- Create: `web/src/components/WeeklyChart.tsx`, `web/src/components/SeasonTable.tsx`, `web/src/components/GameLog.tsx`
- Modify: `web/src/components/PlayerProfile.tsx`, `web/src/App.css`, `README.md`

**Interfaces:**
- Consumes: `PlayerProfileData` (`seasons`, `game_log`) and the placeholder sections from Task 4.
- Produces: finished feature.

- [ ] **Step 0: Load the dataviz skill** (Skill tool, name `dataviz`) BEFORE writing any chart code, and follow its guidance for colors/axes/accessibility within the app's dark theme.
- [ ] **Step 1: WeeklyChart** — hand-rolled SVG (no chart lib), one bar per game in chronological order, grouped by season with a small gap and season labels; distinct color per season (3 max, consistent with dataviz palette guidance); horizontal dashed per-season average line; y-axis ticks at sensible intervals; `<title>` per bar (`'25 wk 8 — 21.4 pts vs GB`). Empty `game_log` → render nothing (Task 4's section collapses).
- [ ] **Step 2: SeasonTable** — one row per season (desc): season, games, PPG, targets, tgt share (%), carries, rec yds, rush yds, TD, rec, yds/opp, snap % — nulls as "—", shares as percentages (1 decimal).
- [ ] **Step 3: GameLog** — collapsible (`<details>` or button-toggled), default collapsed; rows newest first: `'25 wk 10 · vs GB · 9 tgt, 6 rec, 80 yds, 1 TD · 20.0`; season separators.
- [ ] **Step 4: Wire into PlayerProfile** — replace Task 4 placeholders; section order per spec (chart → season table → game log → outlook → similar).
- [ ] **Step 5: README** — add a short "Player profiles" paragraph under the features section (what the drawer shows; cross-year twins with next-season outcome; how similarity works in one sentence).
- [ ] **Step 6: Verify** — `npm run build` clean; full pytest suite still green; browser check: chart renders 3 seasons with averages for a veteran (e.g., top RB), rookie profile shows fallback text and no chart, game log expands, everything readable in dark theme.
- [ ] **Step 7: Commit** — `git add web README.md && git commit -m "feat: profile history chart, season table, game log"`
