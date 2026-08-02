# Board Redesign + Market Consensus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Multi-source market consensus (FFC + ESPN + FantasyPros, joined via the Sleeper ID crosswalk) replacing single-source ADP, then a full pro-terminal visual redesign of the board.

**Architecture:** Three new pipeline sources land in DuckDB; a pure `scoring/market.py` converts each to overall ranks and blends a consensus + spread that `build_board` attaches (replacing `adp`); API/profile carry the new fields; the frontend gets a token-based visual system, top bar with search, and keyboard navigation.

**Tech Stack:** existing (pandas 1.5.3, DuckDB, FastAPI, React 18 + TS + TanStack). New: `@fontsource-variable/inter` + `@fontsource/jetbrains-mono` npm packages (self-hosted fonts).

**Spec:** `docs/superpowers/specs/2026-08-02-board-redesign-market-consensus-design.md`

## Global Constraints

- pandas 1.5.3 — NEVER `.replace(0, pd.NA)`; use `np.nan`.
- Free/keyless sources only; ESPN + FantasyPros fetches send a browser-ish `User-Agent`; a failing source fails ONLY itself (existing per-source isolation + freshness rows).
- ESPN position map (exact): `{1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}`.
- Consensus math (exact): per-source overall ranks — FFC `adp` ranked ascending, ESPN `averageDraftPosition` ranked ascending, FantasyPros `rank_ecr` used directly; `market_rank` = mean of available sources rounded to 1 decimal (None if zero sources); `market_spread` = max − min (None with < 2 sources); spread badge shown in UI only when spread ≥ 12.
- `edge` = `market_rank` − board `rank` (positive = market undervalues); board column `adp` is REMOVED (raw FFC value lives only in `market_sources.ffc`).
- Board/profile payloads add: `market_rank: float|None`, `market_spread: float|None`, `market_sources: {ffc, espn, fp, fp_tier}` (each `float|int|None`). `value_neighbors` and `SimilarPlayer` swap `adp` → `market_rank`.
- JSON-safety: everything through the existing `_scrub` (handles nested dicts).
- UI: no CDN assets — fonts via npm packages; no new pages/routing/state libraries; keyboard map exactly `/` search, `↑/↓` selection, `Enter` profile, `D` drafted, `Esc` close/clear.
- Frontend tasks MUST load the `frontend-design` skill (and `dataviz` where charts are touched) BEFORE writing UI code.
- Use `.venv/bin/pytest`; suite currently 73 passing. Commit per task with conventional messages. Real data smoke against `data/nfl.duckdb`; unit tests use synthetic frames/temp DBs only.

---

### Task 1: New pipeline sources (ESPN, FantasyPros, Sleeper crosswalk)

**Files:**
- Modify: `pipeline/sources.py`, `pipeline/refresh.py`
- Test: `tests/test_pipeline.py` (append)

**Interfaces:**
- Produces (parsers pure; fetchers do network):
  - `parse_espn(payload: dict) -> pd.DataFrame` cols `espn_id, espn_name, position, espn_adp, espn_ppr_rank` (rows with unmapped positions dropped)
  - `fetch_espn_adp(year: int, limit: int = 500) -> pd.DataFrame`
  - `parse_fp_ecr(html: str) -> pd.DataFrame` cols `fp_name, team, position, rank_ecr, rank_ave, rank_std, fp_tier` (empty df when the `ecrData` blob is absent)
  - `fetch_fp_ecr() -> pd.DataFrame`
  - `parse_sleeper(payload: dict) -> pd.DataFrame` cols `gsis_id, espn_id, sleeper_name, position, team` (rows kept only when BOTH ids present and position in QB/RB/WR/TE/K/DST; `gsis_id` stripped of whitespace)
  - `fetch_sleeper_ids() -> pd.DataFrame`
  - Refresh writes tables `espn_adp`, `fp_ecr`, `sleeper_ids` (jobs added to the dict in `pipeline/refresh.py::main`).

- [ ] **Step 1: Write failing tests** (append to `tests/test_pipeline.py`)

```python
from pipeline.sources import parse_espn, parse_fp_ecr, parse_sleeper

def test_parse_espn():
    payload = {"players": [
        {"player": {"id": 4429795, "fullName": "Jahmyr Gibbs", "defaultPositionId": 2,
                    "ownership": {"averageDraftPosition": 1.77},
                    "draftRanksByRankType": {"PPR": {"rank": 1}}}},
        {"player": {"id": 1, "fullName": "Some Lineman", "defaultPositionId": 9}},
    ]}
    df = parse_espn(payload)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["espn_id"] == 4429795 and r["position"] == "RB"
    assert r["espn_adp"] == 1.77 and r["espn_ppr_rank"] == 1

def test_parse_fp_ecr():
    html = ('<script>var x = 1; var ecrData = {"players": [{"player_name": "JaMarr Chase",'
            '"player_team_id": "CIN", "player_position_id": "WR", "rank_ecr": 1,'
            '"rank_ave": "1.77", "rank_std": "1.2", "tier": 1}]};</script>')
    df = parse_fp_ecr(html)
    assert df.iloc[0]["rank_ecr"] == 1 and df.iloc[0]["position"] == "WR"
    assert parse_fp_ecr("<html>no data</html>").empty

def test_parse_sleeper():
    payload = {
        "a": {"gsis_id": " 00-0038543", "espn_id": 4429795, "full_name": "Jahmyr Gibbs",
              "position": "RB", "team": "DET", "active": True},
        "b": {"gsis_id": None, "espn_id": 99, "full_name": "No Gsis", "position": "WR"},
        "c": {"gsis_id": "00-1", "espn_id": 5, "full_name": "A Lineman", "position": "OT"},
    }
    df = parse_sleeper(payload)
    assert len(df) == 1 and df.iloc[0]["gsis_id"] == "00-0038543"
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_pipeline.py -v` → ImportError.

- [ ] **Step 3: Implement in `pipeline/sources.py`**

```python
import json
import re

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
ESPN_URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
            "seasons/{year}/segments/0/leaguedefaults/3?view=kona_player_info")
FP_URL = "https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php"
SLEEPER_URL = "https://api.sleeper.app/v1/players/nfl"
_ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
_FANTASY_POS = {"QB", "RB", "WR", "TE", "K", "DST"}

def parse_espn(payload: dict) -> pd.DataFrame:
    rows = []
    for entry in payload.get("players", []):
        p = entry.get("player") or {}
        pos = _ESPN_POS.get(p.get("defaultPositionId"))
        if pos is None:
            continue
        rows.append({
            "espn_id": p.get("id"), "espn_name": p.get("fullName"), "position": pos,
            "espn_adp": (p.get("ownership") or {}).get("averageDraftPosition"),
            "espn_ppr_rank": ((p.get("draftRanksByRankType") or {}).get("PPR") or {}).get("rank"),
        })
    return pd.DataFrame(rows, columns=["espn_id", "espn_name", "position",
                                       "espn_adp", "espn_ppr_rank"])

def fetch_espn_adp(year: int, limit: int = 500) -> pd.DataFrame:
    headers = {**UA, "X-Fantasy-Filter": json.dumps(
        {"players": {"limit": limit,
                     "sortAdp": {"sortAsc": True, "sortPriority": 1}}})}
    resp = requests.get(ESPN_URL.format(year=year), headers=headers, timeout=30)
    resp.raise_for_status()
    return parse_espn(resp.json())

def parse_fp_ecr(html: str) -> pd.DataFrame:
    cols = ["fp_name", "team", "position", "rank_ecr", "rank_ave", "rank_std", "fp_tier"]
    m = re.search(r"var ecrData = (\{.*?\});", html, re.DOTALL)
    if not m:
        return pd.DataFrame(columns=cols)
    data = json.loads(m.group(1))
    rows = [{"fp_name": p.get("player_name"), "team": p.get("player_team_id"),
             "position": p.get("player_position_id"),
             "rank_ecr": p.get("rank_ecr"),
             "rank_ave": pd.to_numeric(p.get("rank_ave"), errors="coerce"),
             "rank_std": pd.to_numeric(p.get("rank_std"), errors="coerce"),
             "fp_tier": p.get("tier")}
            for p in data.get("players", [])]
    return pd.DataFrame(rows, columns=cols)

def fetch_fp_ecr() -> pd.DataFrame:
    resp = requests.get(FP_URL, headers=UA, timeout=30)
    resp.raise_for_status()
    return parse_fp_ecr(resp.text)

def parse_sleeper(payload: dict) -> pd.DataFrame:
    rows = []
    for p in payload.values():
        gsis, espn = p.get("gsis_id"), p.get("espn_id")
        if not gsis or not espn or p.get("position") not in _FANTASY_POS:
            continue
        rows.append({"gsis_id": str(gsis).strip(), "espn_id": espn,
                     "sleeper_name": p.get("full_name"),
                     "position": p.get("position"), "team": p.get("team")})
    return pd.DataFrame(rows, columns=["gsis_id", "espn_id", "sleeper_name",
                                       "position", "team"])

def fetch_sleeper_ids() -> pd.DataFrame:
    resp = requests.get(SLEEPER_URL, headers=UA, timeout=60)
    resp.raise_for_status()
    return parse_sleeper(resp.json())
```

In `pipeline/refresh.py`, add to `jobs`:

```python
        "espn_adp": lambda: sources.fetch_espn_adp(CURRENT_SEASON),
        "fp_ecr": sources.fetch_fp_ecr,
        "sleeper_ids": sources.fetch_sleeper_ids,
```

- [ ] **Step 4: Run tests, then the real refresh** — pytest green (73 + 3). `.venv/bin/python -m pipeline.refresh`: expect all 8 sources OK (espn_adp ~300–500 rows, fp_ecr ~500, sleeper_ids ~1500–3000). Record actual counts in the report; a FAIL on espn/fp is a stop-and-investigate (they were verified working today), not an accept.

- [ ] **Step 5: Commit** — `git add pipeline tests && git commit -m "feat: espn, fantasypros, sleeper crosswalk pipeline sources"`

---

### Task 2: Market consensus module

**Files:**
- Create: `scoring/market.py`
- Test: `tests/test_market.py`

**Interfaces:**
- Consumes: `scoring.board._norm_name`.
- Produces: `add_market(board: pd.DataFrame, espn: pd.DataFrame, fp: pd.DataFrame, sleeper: pd.DataFrame) -> pd.DataFrame` — takes a board frame that still carries the FFC `adp` column plus the three source tables (any may be empty), returns the board with `adp` DROPPED and `market_rank`, `market_spread`, `market_sources` added, `edge` recomputed as `market_rank − rank`.

Behavior (exact):
1. `ffc_rank` = `board["adp"].rank(method="first")` (NaN adp → NaN rank).
2. ESPN: rows with non-null `espn_adp`; `espn_rank` = rank ascending of `espn_adp`. Join to board via sleeper crosswalk (`espn_id → gsis_id` = board `player_id`); rows unmatched by crosswalk fall back to `_norm_name(espn_name)` + `position` against the board (DST: ESPN DST names don't match team names — DST joins by crosswalk only, else stays unmatched).
3. FantasyPros: `fp_rank` = `rank_ecr`; join `_norm_name(fp_name)` + `position`; rows with `position == "DST"` join by `team` instead.
4. `market_rank` = row-wise mean of the available ranks, rounded 1 decimal; None (NaN) when no source matched. `market_spread` = max − min when ≥ 2 sources, else NaN.
5. `market_sources` = per-row dict `{"ffc": float|None, "espn": float|None, "fp": float|None, "fp_tier": int|None}` (fp_tier from the FP join).
6. `edge` = `market_rank − rank`; drop columns `adp` and any intermediates.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_market.py
import numpy as np
import pandas as pd
from scoring.market import add_market

def _board():
    return pd.DataFrame({
        "player_id": ["g1", "g2", "g3"],
        "name": ["Jahmyr Gibbs", "Amon-Ra St. Brown", "Ravens DST"],
        "position": ["RB", "WR", "DST"], "team": ["DET", "DET", "BAL"],
        "rank": [1, 2, 3], "adp": [1.6, 5.1, np.nan]})

def _espn():
    return pd.DataFrame({"espn_id": [101, 102], "espn_name": ["Jahmyr Gibbs", "Amon-Ra St Brown"],
                         "position": ["RB", "WR"], "espn_adp": [1.77, 6.0],
                         "espn_ppr_rank": [1, 4]})

def _sleeper():
    return pd.DataFrame({"gsis_id": ["g1"], "espn_id": [101],
                         "sleeper_name": ["Jahmyr Gibbs"], "position": ["RB"], "team": ["DET"]})

def _fp():
    return pd.DataFrame({"fp_name": ["Amon-Ra St. Brown", "Baltimore Ravens"],
                         "team": ["DET", "BAL"], "position": ["WR", "DST"],
                         "rank_ecr": [3, 40], "rank_ave": [3.0, 41.0],
                         "rank_std": [1.0, 5.0], "fp_tier": [1, 5]})

def test_consensus_all_paths():
    out = add_market(_board(), _espn(), _fp(), _sleeper())
    g1 = out[out["player_id"] == "g1"].iloc[0]   # FFC rank 1, ESPN rank 1 (via crosswalk)
    assert g1["market_rank"] == 1.0 and g1["market_sources"]["espn"] == 1.0
    g2 = out[out["player_id"] == "g2"].iloc[0]   # FFC 2, ESPN 2 (name fallback), FP 3
    assert g2["market_rank"] == round((2 + 2 + 3) / 3, 1)
    assert g2["market_spread"] == 1.0
    assert g2["market_sources"]["fp_tier"] == 1
    dst = out[out["player_id"] == "g3"].iloc[0]  # FP only, joined by team
    assert dst["market_sources"]["fp"] == 40.0 and dst["market_rank"] == 40.0
    assert pd.isna(dst["market_spread"])         # single source
    assert "adp" not in out.columns

def test_edge_uses_market_rank():
    out = add_market(_board(), _espn(), _fp(), _sleeper())
    g2 = out[out["player_id"] == "g2"].iloc[0]
    assert g2["edge"] == g2["market_rank"] - g2["rank"]

def test_all_sources_empty():
    empty = pd.DataFrame()
    out = add_market(_board(), empty, empty, empty)
    r = out.iloc[0]
    assert r["market_rank"] == 1.0              # FFC alone still ranks
    assert out[out["player_id"] == "g3"].iloc[0]["market_sources"]["ffc"] is None or \
           pd.isna(out[out["player_id"] == "g3"].iloc[0]["market_sources"]["ffc"])

def test_no_sources_at_all():
    board = _board().assign(adp=np.nan)
    empty = pd.DataFrame()
    out = add_market(board, empty, empty, empty)
    assert out["market_rank"].isna().all()
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement `scoring/market.py`**

```python
# scoring/market.py
"""Blend FFC / ESPN / FantasyPros into a market consensus rank per board player.

NOTE: scoring.board imports add_market (Task 3), so importing board at module
level here would be circular — _norm_name is imported inside the helpers.
"""
import numpy as np
import pandas as pd

_RANK_COLS = ["ffc_rank", "espn_rank", "fp_rank"]

def _norm(name):
    from scoring.board import _norm_name  # deferred: avoids circular import
    return _norm_name(name)

def _espn_ranks(board, espn, sleeper):
    out = pd.Series(np.nan, index=board.index)
    if espn is None or espn.empty:
        return out
    e = espn.dropna(subset=["espn_adp"]).copy()
    e["espn_rank"] = e["espn_adp"].rank(method="first")
    if sleeper is not None and not sleeper.empty:
        xwalk = sleeper[["gsis_id", "espn_id"]].drop_duplicates("espn_id")
        e = e.merge(xwalk, on="espn_id", how="left")
    else:
        e["gsis_id"] = None
    by_id = e.dropna(subset=["gsis_id"]).set_index("gsis_id")["espn_rank"]
    mapped = board["player_id"].map(by_id)
    # name+position fallback for espn rows without a crosswalk hit (never DST)
    rest = e[e["gsis_id"].isna() & (e["position"] != "DST")].copy()
    if not rest.empty:
        rest["norm"] = rest["espn_name"].map(_norm)
        by_name = rest.set_index(["norm", "position"])["espn_rank"]
        key = pd.MultiIndex.from_arrays([board["name"].map(_norm), board["position"]])
        fallback = pd.Series(by_name.reindex(key).to_numpy(), index=board.index)
        mapped = mapped.fillna(fallback)
    return mapped

def _fp_ranks(board, fp):
    ranks = pd.Series(np.nan, index=board.index)
    tiers = pd.Series(np.nan, index=board.index)
    if fp is None or fp.empty:
        return ranks, tiers
    f = fp.dropna(subset=["rank_ecr"]).copy()
    players = f[f["position"] != "DST"].copy()
    players["norm"] = players["fp_name"].map(_norm)
    by_name = players.set_index(["norm", "position"])
    key = pd.MultiIndex.from_arrays([board["name"].map(_norm), board["position"]])
    ranks = pd.Series(by_name["rank_ecr"].reindex(key).to_numpy(), index=board.index, dtype=float)
    tiers = pd.Series(by_name["fp_tier"].reindex(key).to_numpy(), index=board.index, dtype=float)
    dst = f[f["position"] == "DST"].drop_duplicates("team").set_index("team")
    is_dst = board["position"] == "DST"
    ranks.loc[is_dst] = board.loc[is_dst, "team"].map(dst["rank_ecr"]).astype(float)
    tiers.loc[is_dst] = board.loc[is_dst, "team"].map(dst["fp_tier"]).astype(float)
    return ranks, tiers

def add_market(board, espn, fp, sleeper):
    out = board.copy()
    out["ffc_rank"] = out["adp"].rank(method="first")
    out["espn_rank"] = _espn_ranks(out, espn, sleeper)
    out["fp_rank"], out["fp_tier"] = _fp_ranks(out, fp)
    ranks = out[_RANK_COLS].astype(float)
    out["market_rank"] = ranks.mean(axis=1, skipna=True).round(1)
    n = ranks.notna().sum(axis=1)
    spread = ranks.max(axis=1) - ranks.min(axis=1)
    out["market_spread"] = spread.where(n >= 2)
    def _val(v):
        return None if pd.isna(v) else float(v)
    out["market_sources"] = [
        {"ffc": _val(r.ffc_rank), "espn": _val(r.espn_rank), "fp": _val(r.fp_rank),
         "fp_tier": None if pd.isna(r.fp_tier) else int(r.fp_tier)}
        for r in out.itertuples()]
    out["edge"] = out["market_rank"] - out["rank"]
    return out.drop(columns=["adp"] + _RANK_COLS + ["fp_tier"])
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_market.py -v`, then full suite (existing board/api tests still pass because `build_board` is unchanged so far).

- [ ] **Step 5: Commit** — `git add scoring/market.py tests/test_market.py && git commit -m "feat: multi-source market consensus module"`

---

### Task 3: Board / profile / API integration

**Files:**
- Modify: `scoring/board.py`, `scoring/similarity.py` (`value_neighbors`), `scoring/profile.py`
- Test: update `tests/test_board.py`, `tests/test_api.py`, `tests/test_profile.py`, `tests/test_similarity.py`

**Interfaces:**
- Consumes: `scoring.market.add_market`.
- Produces:
  - `build_board` output: `adp` and old `adp_rank` gone; new `market_rank`, `market_spread`, `market_sources`; `edge = market_rank − rank`. Column contract line in `_BOARD_COLUMNS` updated accordingly. `build_board` reads `espn_adp`, `fp_ecr`, `sleeper_ids` via `read_table` and calls `add_market` after VOR/tiers/rank assignment (edge no longer computed in board.py itself).
  - `value_neighbors` rows: `adp` key → `market_rank`.
  - Profile payload: `header` naturally carries the new fields; `similar.players[*].adp` → `market_rank`; twins enrichment joins `market_rank` instead of `adp`.
- Ordering note: `add_market` needs the FFC `adp` column still present — board keeps the existing `_merge_adp` FFC join, then hands off to `add_market` which consumes and drops it.

Test updates (write them BEFORE the code changes — this is the task's failing-test step):
- `tests/test_board.py`: seed now also writes empty `espn_adp`/`fp_ecr`/`sleeper_ids` tables (with correct columns); assertions change from `adp` to `market_rank` (`star["market_rank"] == 1.0` given the single-source seed where the star's FFC adp is lowest); `market_sources` dict present with `ffc` non-null; column-contract test updated.
- `tests/test_api.py`: `_seed` gains the three empty tables; `test_players_endpoint` asserts `market_rank` and absence of `adp` in the payload row.
- `tests/test_profile.py`: `_seed` gains the tables; enrichment test asserts twins carry `market_rank` key.
- `tests/test_similarity.py`: `value_neighbors` test frame renames its `adp` column to `market_rank` and asserts the output key.

- [ ] **Step 1: Update the tests listed above** (failing first).
- [ ] **Step 2: Run to verify failures** — `.venv/bin/pytest -q` shows the updated tests failing.
- [ ] **Step 3: Implement** — board reads the three tables, calls `add_market(uni, espn, fp, sleeper)` after `rank` assignment, removes its own `adp_rank`/`edge` computation; `_BOARD_COLUMNS` updated to `[..., "market_rank", "market_spread", "market_sources", "edge", ...]`; `value_neighbors` and `_enrich_twins` switch to `market_rank`.
- [ ] **Step 4: Full suite green; real-data smoke** — build the board, print top 15 with `market_rank`/`market_spread`; verify consensus values are plausible (top players' market_rank ≈ 1–15, spreads mostly < 15) and count how many players matched 3, 2, 1, 0 sources — report the histogram.
- [ ] **Step 5: Commit** — `git add scoring tests && git commit -m "feat: board and profile carry market consensus fields"`

---

### Task 4: Frontend market display

**Files:**
- Modify: `web/src/api.ts`, `web/src/components/PlayerTable.tsx`, `web/src/components/PlayerProfile.tsx`, `web/src/components/SimilarPlayers.tsx`

**Interfaces:**
- Consumes: new payload fields from Task 3.
- Produces: `Player` type — remove `adp`, add `market_rank: number | null`, `market_spread: number | null`, `market_sources: { ffc: number | null; espn: number | null; fp: number | null; fp_tier: number | null }`; `SimilarPlayer.adp` → `market_rank`.

- [ ] **Step 1: Types** — update `web/src/api.ts` per above.
- [ ] **Step 2: Board column** — replace the ADP column with "Mkt": renders `market_rank` (1 decimal); when `market_spread !== null && market_spread >= 12` append a muted `±N` (spread/2 rounded — shows as e.g. `24.3 ±9`); cell `title` tooltip lists `FFC · ESPN · FP` per-source values ("—" for null) and FP tier. Edge column unchanged (values now derive from consensus server-side).
- [ ] **Step 3: Drawer** — header chip "ADP" → "Mkt" (market_rank); add a compact "Market" block under Outlook: three labeled values (FFC / ESPN / FP) + "FP tier N" when present. `SimilarPlayers` board chips show `#rank / Mkt N`.
- [ ] **Step 4: Verify** — `npm run build` clean; with API running, browser check: Mkt column populated, tooltip shows three sources, spread suffix appears for at least one disagreement player (find one via sort), drawer market block renders.
- [ ] **Step 5: Commit** — `git add web && git commit -m "feat: market consensus display in board and drawer"`

---

### Task 5: Design system + top bar + search (pro terminal, part 1)

**Files:**
- Modify: `web/src/App.css`, `web/src/index.css`, `web/src/App.tsx`, `web/package.json`
- Create: `web/src/components/TopBar.tsx`

**Interfaces:**
- Produces: CSS custom-property token system (`--bg-0/1/2`, `--border`, `--text-1/2/3`, `--accent`, `--pos-qb/rb/wr/te/k/dst`, spacing/type scale) that Task 6 consumes; `TopBar` with props `{ search: string; onSearch: (s: string) => void; meta: ... }`; App-level `search` state filtering the board (case-insensitive substring on name OR team, composed with position tab + hide-drafted filters).

- [ ] **Step 0: Load the `frontend-design` skill BEFORE writing any code** and apply its guidance to the token/typography choices within the pro-terminal direction (dense, restrained color, signal-only accents).
- [ ] **Step 1: Fonts** — `cd web && npm install @fontsource-variable/inter @fontsource/jetbrains-mono`; import both in `web/src/main.tsx`; set `--font-ui` / `--font-mono` tokens; tabular numerals (`font-variant-numeric: tabular-nums`) on all numeric cells.
- [ ] **Step 2: Token system** — rewrite `index.css`/`App.css` base layers: three background elevations, one border color, three text levels, position hues (distinct, dark-theme-legible, CVD-aware — check with the dataviz palette guidance if in doubt), accent reserved for interactive states. Remove ad-hoc colors from existing rules by migrating them onto tokens.
- [ ] **Step 3: TopBar** — replaces the current header row: title (left, compact), search input (center, placeholder "Search players…  /", `/` keydown at document level focuses it unless typing in an input, `Esc` clears+blurs), FreshnessBadge (right, restyle to small dots + tooltip; keep its failure-warning behavior).
- [ ] **Step 4: Left rail** — restyle WeightSliders + hide-drafted into the token system; add collapse toggle (chevron button; collapsed state shows a thin rail with an expand affordance; persisted in `localStorage`).
- [ ] **Step 5: Search wiring** — App `search` state → filter before position/drafted filters; zero matches renders a single muted empty-state row ("No players match — Esc to clear").
- [ ] **Step 6: Verify** — build clean; browser check: fonts load offline (network tab: no external requests), `/` focuses, `Esc` clears, search narrows with tabs still applying, rail collapses and persists across reload.
- [ ] **Step 7: Commit** — `git add web && git commit -m "feat: design tokens, top bar with search, collapsible rail"`

---

### Task 6: Board visual system + keyboard nav (pro terminal, part 2)

**Files:**
- Modify: `web/src/components/PlayerTable.tsx`, `web/src/components/PlayerProfile.tsx` (token inheritance only), `web/src/App.css`, `web/src/App.tsx`, `README.md`

**Interfaces:**
- Consumes: tokens from Task 5.
- Produces: finished redesign.

- [ ] **Step 0: Load the `frontend-design` skill (and `dataviz` if you touch chart colors) BEFORE writing code.**
- [ ] **Step 1: Table restyle** — sticky header (`position: sticky; top: 0` inside the scroll container, elevated background); ~32px rows; position badges (small rounded chip, position hue at low-alpha background + full-strength text); tier group bands (alternating faint background per tier when sorted by rank, replacing the border rule — keep the `showTierBreaks` gating by single-position tabs); edge chips (signed, green/red/neutral-gray for |edge| < 3); VOR micro-bar (absolute-positioned low-alpha bar behind the number, width scaled to position-max VOR); drafted rows: dim + strikethrough (keep), ✓ becomes ↩ undo icon on drafted rows.
- [ ] **Step 2: Keyboard nav** — App-level selection state (`selectedIndex` over the VISIBLE row order): `↑/↓` move (with scroll-into-view), `Enter` opens profile for selection, `D` toggles drafted, all inert while an input or the drawer has focus (except `Esc` which closes the drawer first, then clears search). Visible selection highlight row style.
- [ ] **Step 3: Drawer token inheritance** — swap drawer's hardcoded colors to tokens; position badge in drawer header; NO structural changes.
- [ ] **Step 4: README** — update the features section: market consensus (three sources + spread), search, keyboard shortcuts table.
- [ ] **Step 5: Verify** — build clean; full pytest still green; browser check: sticky header while scrolling 200 rows, badges/tier bands/edge chips/VOR bars render, keyboard flow works end-to-end (navigate → Enter → Esc → D), drafted undo works, drawer matches the new look.
- [ ] **Step 6: Commit** — `git add web README.md && git commit -m "feat: pro-terminal board visuals and keyboard navigation"`
