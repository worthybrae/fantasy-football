# Draft History Import, Manager Modeling, and Draft Simulation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import this league's ESPN draft history, fit a per-manager pick-prediction model to it, and simulate the upcoming draft so the board can show who survives to my next pick and which available player leads to the best final roster.

**Architecture:** A Playwright-authenticated ESPN client writes raw draft history and league settings to DuckDB. League settings replace three hardcoded constants (team count, replacement ranks, scoring rules). A conditional-logit model fits per-manager pick behavior with ridge shrinkage toward a pooled fit. A Monte Carlo rollout simulator uses those models to rank candidates at each of my picks, persisting results to DuckDB that the existing FastAPI board reads.

**Tech Stack:** Python 3, pandas, numpy, scipy (new), duckdb, Playwright (new), FastAPI, React + TypeScript + Vite.

**Spec:** `docs/superpowers/specs/2026-08-09-draft-history-simulator-design.md`

## Global Constraints

- Every Python test runs offline. No test makes a network call, launches a browser, or reads `data/nfl.duckdb`. Tests seed a DuckDB file under pytest's `tmp_path` and use literal dict payloads as fixtures, matching `tests/test_pipeline.py` and `tests/test_board.py`.
- New dependencies: `scipy>=1.11` and `playwright>=1.40` appended to `requirements.txt`.
- Existing behavior must not change when no ESPN import has run. `build_board` on a database without a `league` table must produce byte-identical output to today. The full existing suite must stay green after every task.
- Name matching between ESPN and nflverse/ADP data always goes through `scoring.board._norm_name`. Never write a second normalizer.
- Position vocabulary is exactly `QB, RB, WR, TE, K, DST` (`scoring.board.FANTASY_POSITIONS`). ESPN's `DEF`/`D/ST` and `PK` spellings are normalized on the way in.
- Sim and model code is deterministic under a fixed seed. Every function that samples takes an explicit `numpy.random.Generator`; none call the global numpy RNG.
- Run tests with `.venv/bin/pytest`. Run a single test with `.venv/bin/pytest tests/test_x.py::test_y -v`.
- Commit after every task. Branch is `draft-history-simulator`.

## File Structure

| File | Responsibility |
|---|---|
| `pipeline/espn_league.py` (new) | ESPN payload parsers + Playwright client + season walk + import driver |
| `scoring/league.py` (new) | `LeagueSettings` dataclass, ESPN settings → league structure, replacement-rank derivation, config fallback |
| `scoring/draft_model.py` (new) | Choice-set reconstruction, feature matrix, conditional-logit fit, shrinkage, backtest, profile text |
| `scoring/draft_sim.py` (new) | Projections, lineup optimizer, roster value, rollout engine, candidate search |
| `scoring/config.py` (modify) | Add `FLEX_SHARES`; existing constants become the fallback path |
| `scoring/ppr.py` (modify) | `compute_ppr_points` takes an optional rules dict |
| `scoring/composite.py` (modify) | `apply_vor` takes optional replacement ranks |
| `scoring/board.py` (modify) | Load effective league settings, thread them through |
| `pipeline/db.py` (modify) | `drafted` table gains `pick_no` |
| `api/main.py` (modify) | League, managers, draft-order, sim endpoints; sim columns on `/api/players` |
| `web/src/components/DraftRail.tsx` (new) | Draft-order editor + manager cards |
| `web/src/api.ts`, `web/src/components/PlayerTable.tsx`, `web/src/App.tsx` (modify) | Sim columns, rail mount, header strip |

---

## Phase 1 — Import and league structure

### Task 1: ESPN payload parsers

Pure functions over literal dicts. No network, no browser.

**Files:**
- Create: `pipeline/espn_league.py`
- Test: `tests/test_espn_league.py`

**Interfaces:**
- Consumes: `scoring.board._norm_name`, `scoring.board.FANTASY_POSITIONS`
- Produces:
  - `ESPN_SLOT_POSITIONS: dict[int, str]`
  - `ESPN_POSITION_BY_ID: dict[int, str]`
  - `parse_draft_picks(payload: dict, season: int) -> pd.DataFrame` with columns `season, overall_pick, round, round_pick, team_id, espn_player_id, keeper`
  - `parse_draft_teams(payload: dict, season: int) -> pd.DataFrame` with columns `season, team_id, manager, slot`
  - `parse_player_directory(payload: list | dict) -> pd.DataFrame` with columns `espn_player_id, player_name, position, nfl_team`

- [ ] **Step 1: Write the failing test**

Create `tests/test_espn_league.py`:

```python
import pandas as pd
from pipeline.espn_league import (
    parse_draft_picks, parse_draft_teams, parse_player_directory,
)

DRAFT_PAYLOAD = {
    "draftDetail": {
        "drafted": True,
        "picks": [
            {"overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1,
             "teamId": 3, "playerId": 4046537, "keeper": False},
            {"overallPickNumber": 2, "roundId": 1, "roundPickNumber": 2,
             "teamId": 7, "playerId": 3117251, "keeper": False},
            {"overallPickNumber": 9, "roundId": 2, "roundPickNumber": 1,
             "teamId": 7, "playerId": 2977187, "keeper": True},
        ],
    }
}

TEAM_PAYLOAD = {
    "teams": [
        {"id": 3, "name": "Team Alpha", "owners": ["{AAA}"], "draftDayPickOrder": 1},
        {"id": 7, "name": "Team Bravo", "owners": ["{BBB}"], "draftDayPickOrder": 2},
    ],
    "members": [
        {"id": "{AAA}", "displayName": "worthy"},
        {"id": "{BBB}", "displayName": "dan"},
    ],
}

def test_parse_draft_picks():
    df = parse_draft_picks(DRAFT_PAYLOAD, 2025)
    assert list(df.columns) == ["season", "overall_pick", "round", "round_pick",
                                "team_id", "espn_player_id", "keeper"]
    assert len(df) == 3
    assert df.iloc[0]["season"] == 2025
    assert df.iloc[0]["espn_player_id"] == 4046537
    assert df.iloc[2]["keeper"]

def test_parse_draft_teams_uses_member_display_name():
    df = parse_draft_teams(TEAM_PAYLOAD, 2025)
    assert df.set_index("team_id")["manager"].to_dict() == {3: "worthy", 7: "dan"}
    assert df.set_index("team_id")["slot"].to_dict() == {3: 1, 7: 2}

def test_parse_draft_teams_falls_back_to_team_name():
    payload = {"teams": [{"id": 3, "name": "Orphan", "owners": [], "draftDayPickOrder": 1}],
               "members": []}
    assert parse_draft_teams(payload, 2025).iloc[0]["manager"] == "Orphan"

def test_parse_player_directory_maps_positions_and_skips_unknown():
    payload = [
        {"id": 4046537, "fullName": "Justin Jefferson",
         "defaultPositionId": 3, "proTeamId": 16},
        {"id": 999, "fullName": "Some Punter", "defaultPositionId": 7, "proTeamId": 16},
        {"id": 16018, "fullName": "Vikings D/ST",
         "defaultPositionId": 16, "proTeamId": 16},
    ]
    df = parse_player_directory(payload)
    assert list(df.columns) == ["espn_player_id", "player_name", "position", "nfl_team"]
    assert df.set_index("espn_player_id")["position"].to_dict() == {4046537: "WR", 16018: "DST"}
    assert df.set_index("espn_player_id")["nfl_team"].to_dict() == {4046537: "MIN", 16018: "MIN"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_espn_league.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.espn_league'`

- [ ] **Step 3: Write minimal implementation**

Create `pipeline/espn_league.py`:

```python
"""ESPN fantasy league client: draft history, rosters, and league settings.

Parsers are pure functions over the JSON ESPN's read API returns, so they are
testable against literal fixtures with no network or browser. The Playwright
client and the season walk live further down.
"""
import pandas as pd

# ESPN lineup slot ids -> our position vocabulary. Slots we do not model
# (individual defensive positions, punter, head coach) are absent by design.
ESPN_SLOT_POSITIONS = {
    0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "DST", 17: "K",
}
ESPN_FLEX_SLOT = 23      # RB/WR/TE
ESPN_BENCH_SLOT = 20
ESPN_IR_SLOT = 21

# `defaultPositionId` on a player, which uses a different id space than slots.
ESPN_POSITION_BY_ID = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}

# ESPN `proTeamId` -> nflverse team abbreviation. 0 is free agent.
ESPN_PRO_TEAMS = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LA",
    15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA",
    27: "TB", 28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}

_PICK_COLUMNS = ["season", "overall_pick", "round", "round_pick",
                 "team_id", "espn_player_id", "keeper"]


def parse_draft_picks(payload: dict, season: int) -> pd.DataFrame:
    picks = ((payload.get("draftDetail") or {}).get("picks")) or []
    rows = [{"season": season,
             "overall_pick": p.get("overallPickNumber"),
             "round": p.get("roundId"),
             "round_pick": p.get("roundPickNumber"),
             "team_id": p.get("teamId"),
             "espn_player_id": p.get("playerId"),
             "keeper": bool(p.get("keeper"))}
            for p in picks]
    df = pd.DataFrame(rows, columns=_PICK_COLUMNS)
    return df.sort_values("overall_pick").reset_index(drop=True)


def parse_draft_teams(payload: dict, season: int) -> pd.DataFrame:
    # `owners` holds member GUIDs; the human-readable name lives in `members`.
    # A team with no owner (orphan/abandoned) falls back to its team name so
    # the manager column is never null -- the model keys on it.
    members = {m.get("id"): m.get("displayName")
               for m in (payload.get("members") or [])}
    rows = []
    for t in payload.get("teams") or []:
        owners = t.get("owners") or []
        manager = next((members[o] for o in owners if members.get(o)), None)
        rows.append({"season": season, "team_id": t.get("id"),
                     "manager": manager or t.get("name"),
                     "slot": t.get("draftDayPickOrder")})
    return pd.DataFrame(rows, columns=["season", "team_id", "manager", "slot"])


def parse_player_directory(payload) -> pd.DataFrame:
    # The season player endpoint returns a bare list; the league views nest it
    # under "players". Accept either so one parser serves both.
    entries = payload if isinstance(payload, list) else (payload.get("players") or [])
    rows = []
    for entry in entries:
        p = entry.get("player", entry)
        pos = ESPN_POSITION_BY_ID.get(p.get("defaultPositionId"))
        if pos is None:
            continue
        rows.append({"espn_player_id": p.get("id"),
                     "player_name": p.get("fullName"),
                     "position": pos,
                     "nfl_team": ESPN_PRO_TEAMS.get(p.get("proTeamId"))})
    return pd.DataFrame(rows, columns=["espn_player_id", "player_name",
                                       "position", "nfl_team"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_espn_league.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/espn_league.py tests/test_espn_league.py
git commit -m "feat: ESPN draft history payload parsers"
```

---

### Task 2: League settings derivation

**Files:**
- Create: `scoring/league.py`
- Modify: `scoring/config.py`
- Modify: `pipeline/espn_league.py` (add `parse_settings`)
- Test: `tests/test_league.py`, `tests/test_espn_league.py`

**Interfaces:**
- Consumes: `parse_draft_teams` from Task 1
- Produces:
  - `pipeline.espn_league.parse_settings(payload: dict, season: int) -> dict` with keys `season, teams, lineup_slots, scoring_items, draft_type, pick_order`
  - `scoring.league.LeagueSettings` frozen dataclass with fields `season: int, teams: int, starters: dict[str, int], flex_slots: int, bench: int, scoring: dict[str, float], unmapped_scoring: list[str], draft_type: str, pick_order: list[int]` and properties `rounds: int`, `replacement_ranks: dict[str, int]`
  - `scoring.league.from_espn(settings: dict) -> LeagueSettings`
  - `scoring.league.default_settings() -> LeagueSettings` — built from `scoring/config.py` constants
  - `scoring.league.load(conn) -> LeagueSettings` — newest row of the `league` table, or `default_settings()` when absent
  - `scoring.config.FLEX_SHARES: dict[str, float]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_league.py`:

```python
import pandas as pd
from pipeline.db import get_conn, write_table
from scoring import league
from scoring.config import LEAGUE_TEAMS, REPLACEMENT_RANK

# 8 teams, QB/2RB/2WR/TE/2FLEX/K/DST + 5 bench, full PPR.
ESPN_SETTINGS = {
    "settings": {
        "size": 8,
        "rosterSettings": {"lineupSlotCounts": {
            "0": 1, "2": 2, "4": 2, "6": 1, "16": 1, "17": 1,
            "20": 5, "21": 1, "23": 2,
        }},
        "scoringSettings": {"scoringItems": [
            {"statId": 3, "points": 0.04}, {"statId": 4, "points": 4.0},
            {"statId": 20, "points": -2.0}, {"statId": 24, "points": 0.1},
            {"statId": 25, "points": 6.0}, {"statId": 53, "points": 1.0},
            {"statId": 42, "points": 0.1}, {"statId": 43, "points": 6.0},
            {"statId": 72, "points": -2.0}, {"statId": 101, "points": 6.0},
        ]},
        "draftSettings": {"type": "SNAKE", "pickOrder": [3, 7, 1, 2, 4, 5, 6, 8]},
    }
}

def _settings():
    from pipeline.espn_league import parse_settings
    return league.from_espn(parse_settings(ESPN_SETTINGS, 2026))

def test_starters_and_rounds():
    s = _settings()
    assert s.teams == 8
    assert s.starters == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}
    assert s.flex_slots == 2
    assert s.bench == 5
    # 8 starting slots + 2 flex + 5 bench; IR is not drafted.
    assert s.rounds == 15

def test_replacement_ranks_match_todays_hardcoded_values():
    assert _settings().replacement_ranks == REPLACEMENT_RANK

def test_scoring_maps_espn_stat_ids_to_nflverse_columns():
    s = _settings()
    assert s.scoring["receptions"] == 1.0
    assert s.scoring["passing_yards"] == 0.04
    # ESPN has one "fumble lost" stat; nflverse splits it three ways.
    assert s.scoring["sack_fumbles_lost"] == -2.0
    assert s.scoring["rushing_fumbles_lost"] == -2.0
    assert s.scoring["receiving_fumbles_lost"] == -2.0

def test_unmapped_scoring_items_are_reported_not_dropped_silently():
    # 101 is a kick-return TD, which nflverse weekly player stats do not carry.
    assert "101" in _settings().unmapped_scoring

def test_default_settings_reproduce_config_constants():
    s = league.default_settings()
    assert s.teams == LEAGUE_TEAMS
    assert s.replacement_ranks == REPLACEMENT_RANK
    assert s.rounds == 15

def test_load_returns_defaults_when_no_league_table(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    assert league.load(conn) == league.default_settings()

def test_load_reads_newest_season_from_league_table(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    rows = pd.DataFrame([
        {"season": 2025, "settings_json": league.to_json(_settings())},
        {"season": 2026, "settings_json": league.to_json(_settings())},
    ])
    write_table(conn, "league", rows)
    assert league.load(conn).season == 2026
```

Append to `tests/test_espn_league.py`:

```python
def test_parse_settings_extracts_pick_order_and_draft_type():
    from pipeline.espn_league import parse_settings
    from tests.test_league import ESPN_SETTINGS
    out = parse_settings(ESPN_SETTINGS, 2026)
    assert out["season"] == 2026
    assert out["teams"] == 8
    assert out["draft_type"] == "SNAKE"
    assert out["pick_order"] == [3, 7, 1, 2, 4, 5, 6, 8]
    assert out["lineup_slots"]["23"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_league.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scoring.league'`

- [ ] **Step 3: Write minimal implementation**

Add to `scoring/config.py`:

```python
# How the league's FLEX slots historically get filled, by position. Used to
# derive REPLACEMENT_RANK from roster shape instead of hardcoding it.
FLEX_SHARES = {"RB": 0.375, "WR": 0.5, "TE": 0.125}
```

Add to `pipeline/espn_league.py`:

```python
def parse_settings(payload: dict, season: int) -> dict:
    s = payload.get("settings") or {}
    return {
        "season": season,
        "teams": (s.get("size") or 0),
        "lineup_slots": (s.get("rosterSettings") or {}).get("lineupSlotCounts") or {},
        "scoring_items": (s.get("scoringSettings") or {}).get("scoringItems") or [],
        "draft_type": (s.get("draftSettings") or {}).get("type"),
        "pick_order": (s.get("draftSettings") or {}).get("pickOrder") or [],
    }
```

Create `scoring/league.py`:

```python
"""League structure, derived from ESPN settings when available.

Everything the board and simulator need to know about league shape lives in
one immutable `LeagueSettings`. Without an ESPN import, `load` returns
`default_settings()`, which reproduces the constants in `scoring/config.py`
exactly -- so a database with no `league` table produces the board it always
produced.
"""
import json
from dataclasses import dataclass, field, asdict

from pipeline.db import read_table
from scoring.config import FLEX_SHARES, LEAGUE_TEAMS, REPLACEMENT_RANK
from scoring.ppr import DEFAULT_RULES

# ESPN scoring statId -> the nflverse weekly column(s) it scores. ESPN carries
# one "fumble lost" stat where nflverse splits it by how the fumble happened,
# so 72 fans out to all three and each gets the same points.
ESPN_STAT_COLUMNS = {
    3: ["passing_yards"], 4: ["passing_tds"], 20: ["passing_interceptions"],
    24: ["rushing_yards"], 25: ["rushing_tds"],
    53: ["receptions"], 42: ["receiving_yards"], 43: ["receiving_tds"],
    72: ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"],
    19: ["passing_2pt_conversions"], 26: ["rushing_2pt_conversions"],
    44: ["receiving_2pt_conversions"],
}

_FLEX_POSITIONS = ("RB", "WR", "TE")


@dataclass(frozen=True)
class LeagueSettings:
    season: int
    teams: int
    starters: dict
    flex_slots: int
    bench: int
    scoring: dict
    draft_type: str
    pick_order: tuple = ()
    unmapped_scoring: tuple = ()

    @property
    def rounds(self) -> int:
        return sum(self.starters.values()) + self.flex_slots + self.bench

    @property
    def replacement_ranks(self) -> dict:
        """Last starter-caliber player at each position.

        `teams * starters + share of the league's flex slots`, plus a
        one-player buffer for positions no flex slot accepts -- without it a
        single-slot position like QB would put replacement level at the very
        last startable player, which is a cliff rather than a baseline.
        """
        total_flex = self.teams * self.flex_slots
        out = {}
        for pos, n in self.starters.items():
            base = self.teams * n
            if pos in _FLEX_POSITIONS:
                out[pos] = base + round(total_flex * FLEX_SHARES.get(pos, 0.0))
            else:
                out[pos] = base + 1
        return out


def from_espn(settings: dict) -> LeagueSettings:
    from pipeline.espn_league import (ESPN_SLOT_POSITIONS, ESPN_FLEX_SLOT,
                                      ESPN_BENCH_SLOT)
    slots = {int(k): int(v) for k, v in (settings.get("lineup_slots") or {}).items()}
    starters = {pos: slots.get(slot_id, 0)
                for slot_id, pos in ESPN_SLOT_POSITIONS.items()}
    starters = {pos: n for pos, n in starters.items() if n > 0}

    scoring, unmapped = {}, []
    for item in settings.get("scoring_items") or []:
        cols = ESPN_STAT_COLUMNS.get(item.get("statId"))
        if not cols:
            unmapped.append(str(item.get("statId")))
            continue
        for col in cols:
            scoring[col] = float(item.get("points") or 0.0)

    return LeagueSettings(
        season=settings["season"],
        teams=settings["teams"],
        starters=starters,
        flex_slots=slots.get(ESPN_FLEX_SLOT, 0),
        bench=slots.get(ESPN_BENCH_SLOT, 0),
        scoring=scoring,
        draft_type=settings.get("draft_type") or "SNAKE",
        pick_order=tuple(settings.get("pick_order") or ()),
        unmapped_scoring=tuple(unmapped),
    )


def default_settings() -> LeagueSettings:
    """Today's hardcoded league, expressed as a LeagueSettings.

    Kept in sync with REPLACEMENT_RANK by a test, not by discipline.
    """
    return LeagueSettings(
        season=0, teams=LEAGUE_TEAMS,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=2, bench=5, scoring=dict(DEFAULT_RULES),
        draft_type="SNAKE",
    )


def to_json(settings: LeagueSettings) -> str:
    return json.dumps(asdict(settings))


def from_json(blob: str) -> LeagueSettings:
    d = json.loads(blob)
    d["pick_order"] = tuple(d.get("pick_order") or ())
    d["unmapped_scoring"] = tuple(d.get("unmapped_scoring") or ())
    return LeagueSettings(**d)


def load(conn) -> LeagueSettings:
    table = read_table(conn, "league")
    if table.empty:
        return default_settings()
    newest = table.sort_values("season").iloc[-1]
    return from_json(newest["settings_json"])
```

Modify `scoring/ppr.py` — rename the module constant so `league.py` can import it, keeping behavior identical:

```python
DEFAULT_RULES = {
    "passing_yards": 0.04, "passing_tds": 4.0, "passing_interceptions": -2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receptions": 1.0, "receiving_yards": 0.1, "receiving_tds": 6.0,
    "sack_fumbles_lost": -2.0, "rushing_fumbles_lost": -2.0, "receiving_fumbles_lost": -2.0,
    "passing_2pt_conversions": 2.0, "rushing_2pt_conversions": 2.0, "receiving_2pt_conversions": 2.0,
}
_RULES = DEFAULT_RULES  # back-compat for existing imports
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_league.py tests/test_espn_league.py -v`
Expected: all passed. `test_replacement_ranks_match_todays_hardcoded_values` passing is the load-bearing one — it proves the derivation reproduces `{"QB": 9, "RB": 22, "WR": 24, "TE": 10, "K": 9, "DST": 9}`.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all passed, no regressions.

- [ ] **Step 6: Commit**

```bash
git add scoring/league.py scoring/config.py scoring/ppr.py pipeline/espn_league.py tests/test_league.py tests/test_espn_league.py
git commit -m "feat: derive league structure from ESPN settings"
```

---

### Task 3: Thread league settings through scoring

Make scoring rules and replacement ranks parameters, defaulting to today's values.

**Files:**
- Modify: `scoring/ppr.py`, `scoring/factors.py`, `scoring/composite.py`, `scoring/board.py`
- Test: `tests/test_ppr.py`, `tests/test_composite.py`, `tests/test_board.py`

**Interfaces:**
- Consumes: `scoring.league.LeagueSettings`, `scoring.league.load`
- Produces:
  - `compute_ppr_points(df, rules: dict | None = None) -> pd.Series`
  - `apply_vor(df, replacement_ranks: dict | None = None) -> pd.DataFrame`
  - `factors.player_seasons(weekly, rules=None)`, `factors.production_factor(weekly, rules=None)`, `factors.durability_factor(weekly, rules=None)`, `factors.role_factor(depth_charts, weekly)` unchanged, `factors.schedule_factor(weekly_prior, schedules, rules=None)`
  - `build_board(conn, weights=None, settings: LeagueSettings | None = None)` — when `settings` is None it calls `league.load(conn)`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ppr.py`:

```python
def test_custom_rules_override_defaults():
    from scoring.ppr import compute_ppr_points
    import pandas as pd
    df = pd.DataFrame([{"receptions": 5, "receiving_yards": 100}])
    assert compute_ppr_points(df).iloc[0] == 15.0            # 5*1.0 + 100*0.1
    half = compute_ppr_points(df, {"receptions": 0.5, "receiving_yards": 0.1})
    assert half.iloc[0] == 12.5                              # 5*0.5 + 100*0.1
```

Append to `tests/test_composite.py`:

```python
def test_apply_vor_accepts_custom_replacement_ranks():
    import pandas as pd
    from scoring.composite import apply_vor
    df = pd.DataFrame({"position": ["RB"] * 5,
                       "composite": [90.0, 80.0, 70.0, 60.0, 50.0]})
    # Replacement at RB2 (index 1) -> the 80.0 player is the baseline.
    out = apply_vor(df, {"RB": 2})
    assert out.sort_values("composite", ascending=False)["vor"].tolist() == [
        10.0, 0.0, -10.0, -20.0, -30.0]
```

Append to `tests/test_board.py`:

```python
def test_board_uses_league_settings_when_present(tmp_path):
    import json
    from pipeline.db import write_table
    from scoring import league
    conn = _seed(tmp_path)
    # Half-PPR, and a much shallower WR replacement level than the default 24.
    settings = league.LeagueSettings(
        season=2026, teams=8,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots=0, bench=5,
        scoring={"receptions": 0.5, "receiving_yards": 0.1},
        draft_type="SNAKE")
    write_table(conn, "league", pd.DataFrame(
        [{"season": 2026, "settings_json": league.to_json(settings)}]))
    board = build_board(conn)
    # WR replacement rank is now teams*2 = 16, not 24.
    assert not board.empty

def test_board_without_league_table_is_unchanged(tmp_path):
    # Same fixture, no `league` table -> the pre-existing expectations hold.
    board = build_board(_seed(tmp_path))
    star = board[board["player_id"] == "p1"].iloc[0]
    assert star["market_rank"] == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ppr.py tests/test_composite.py tests/test_board.py -v`
Expected: FAIL — `compute_ppr_points() takes 1 positional argument but 2 were given`, and `apply_vor()` likewise.

- [ ] **Step 3: Write minimal implementation**

`scoring/ppr.py`:

```python
def compute_ppr_points(df: pd.DataFrame, rules: dict | None = None) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for col, pts in (rules or DEFAULT_RULES).items():
        total = total + _col(df, col) * pts
    return total
```

`scoring/composite.py`:

```python
def apply_vor(df: pd.DataFrame, replacement_ranks: dict | None = None) -> pd.DataFrame:
    ranks = replacement_ranks or REPLACEMENT_RANK
    out = df.copy()
    out["vor"] = 0.0
    for pos, grp in out.groupby("position"):
        ranked = grp.sort_values("composite", ascending=False)
        idx = min(ranks.get(pos, 9), len(ranked)) - 1
        replacement = ranked.iloc[idx]["composite"]
        out.loc[grp.index, "vor"] = grp["composite"] - replacement
    return out
```

`scoring/factors.py` — add an optional `rules` argument to the three functions that score points, threading it into `compute_ppr_points`:

```python
def player_seasons(weekly: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    wk = weekly.copy()
    wk["ppr_points"] = compute_ppr_points(wk, rules)
    ...  # rest unchanged

def production_factor(weekly: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    ps = player_seasons(weekly, rules)
    ...  # rest unchanged

def durability_factor(weekly: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    ps = player_seasons(weekly, rules)
    ...  # rest unchanged

def schedule_factor(weekly_prior: pd.DataFrame, schedules: pd.DataFrame,
                    rules: dict | None = None) -> pd.DataFrame:
    wk = weekly_prior.copy()
    wk["ppr_points"] = compute_ppr_points(wk, rules)
    ...  # rest unchanged
```

`scoring/board.py` — accept settings, load them when absent, pass them down:

```python
def build_board(conn, weights: dict | None = None,
                settings: "LeagueSettings | None" = None) -> pd.DataFrame:
    settings = settings or league.load(conn)
    rules = settings.scoring
    ...
    prod = factors.production_factor(weekly, rules)
    dura = factors.durability_factor(weekly, rules)
    ...
    sos = factors.schedule_factor(prior, sched, rules)
    ...
    uni = apply_vor(uni, settings.replacement_ranks)
```

Add `from scoring import league` to the imports in `scoring/board.py`.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all passed. Existing board tests passing unchanged is the proof that the default path is untouched.

- [ ] **Step 5: Commit**

```bash
git add scoring/ppr.py scoring/factors.py scoring/composite.py scoring/board.py tests/
git commit -m "feat: thread league settings through scoring, keeping config defaults"
```

---

### Task 4: Playwright ESPN client and import driver

**Files:**
- Modify: `pipeline/espn_league.py`
- Modify: `requirements.txt`, `Makefile`, `.gitignore`
- Create: `pipeline/import_league.py`
- Test: `tests/test_espn_league.py`

**Interfaces:**
- Consumes: Task 1 parsers, `scoring.league.from_espn`, `sources.fetch_adp`
- Produces:
  - `pipeline.espn_league.parse_league_id(url_or_id: str) -> str`
  - `pipeline.espn_league.season_url(league_id, season, current_season) -> str`
  - `pipeline.espn_league.EspnClient(state_path: str)` with `get_json(url: str) -> dict` and context-manager support
  - `pipeline.espn_league.import_seasons(conn, league_id, current_season, fetch) -> dict` — `fetch` is a callable `(url) -> dict`, injected so tests pass a fake and never touch the network
  - `pipeline.espn_league.validate_import(conn) -> list[str]` — human-readable report lines

- [ ] **Step 1: Write the failing test**

Append to `tests/test_espn_league.py`:

```python
import pytest
from pipeline.db import get_conn, read_table
from pipeline.espn_league import (
    parse_league_id, season_url, import_seasons, validate_import,
)
from tests.test_league import ESPN_SETTINGS

def test_parse_league_id_from_url_and_bare_id():
    assert parse_league_id("https://fantasy.espn.com/football/league?leagueId=123456") == "123456"
    assert parse_league_id("123456") == "123456"

def test_season_url_uses_league_history_for_prior_seasons():
    assert "leagueHistory/99" in season_url("99", 2024, current_season=2026)
    assert "seasonId=2024" in season_url("99", 2024, current_season=2026)
    assert "seasons/2026/segments/0/leagues/99" in season_url("99", 2026, current_season=2026)

def _fake_fetch(seasons):
    """Serve league + player-directory payloads for `seasons`, 404 otherwise."""
    def fetch(url):
        year = next((s for s in seasons if str(s) in url), None)
        if year is None:
            raise FileNotFoundError(url)
        if "/players?" in url:
            return [{"id": 4046537, "fullName": "Justin Jefferson",
                     "defaultPositionId": 3, "proTeamId": 16},
                    {"id": 3117251, "fullName": "Saquon Barkley",
                     "defaultPositionId": 2, "proTeamId": 21}]
        return {**DRAFT_PAYLOAD, **TEAM_PAYLOAD, **ESPN_SETTINGS}
    return fetch

def test_import_seasons_walks_back_and_writes_tables(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    summary = import_seasons(conn, "99", current_season=2026,
                             fetch=_fake_fetch([2025, 2024]))
    assert summary["seasons"] == [2025, 2024]
    picks = read_table(conn, "draft_picks")
    assert len(picks) == 6                     # 3 picks x 2 seasons
    assert set(picks["player_name"]) == {"Justin Jefferson", "Saquon Barkley"}
    assert not read_table(conn, "league").empty
    assert not read_table(conn, "draft_teams").empty

def test_import_seasons_stops_after_two_consecutive_missing_seasons(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    # 2025 present, 2024 and 2023 missing -> walk halts, 2022 never requested.
    summary = import_seasons(conn, "99", current_season=2026,
                             fetch=_fake_fetch([2025, 2022]))
    assert summary["seasons"] == [2025]

def test_import_rejects_non_snake_draft(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    auction = {**ESPN_SETTINGS}
    auction["settings"] = {**ESPN_SETTINGS["settings"],
                           "draftSettings": {"type": "AUCTION", "pickOrder": []}}
    def fetch(url):
        if "2025" not in url:
            raise FileNotFoundError(url)
        if "/players?" in url:
            return []
        return {**DRAFT_PAYLOAD, **TEAM_PAYLOAD, **auction}
    with pytest.raises(ValueError, match="AUCTION"):
        import_seasons(conn, "99", current_season=2026, fetch=fetch)

def test_validate_import_reports_pick_count_and_adp_match_rate(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    import_seasons(conn, "99", current_season=2026, fetch=_fake_fetch([2025]))
    lines = "\n".join(validate_import(conn))
    assert "2025" in lines
    assert "ADP match" in lines
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_espn_league.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_league_id'`

- [ ] **Step 3: Write minimal implementation**

Append to `pipeline/espn_league.py`:

```python
import re
from pathlib import Path

from pipeline.db import write_table, read_table
from scoring import league as league_mod

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
VIEWS = "view=mDraftDetail&view=mTeam&view=mSettings"
LOGIN_URL = "https://www.espn.com/login"
STATE_PATH = "data/espn_state.json"


def parse_league_id(url_or_id: str) -> str:
    m = re.search(r"leagueId=(\d+)", url_or_id)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{3,})\b", url_or_id)
    if not m:
        raise ValueError(f"no league id in {url_or_id!r}")
    return m.group(1)


def season_url(league_id: str, season: int, current_season: int) -> str:
    # ESPN serves the live season under /seasons/{year}/... and every earlier
    # season under /leagueHistory/{id}?seasonId=. Hitting the wrong one for a
    # given year returns 404, not a redirect.
    if season >= current_season:
        return f"{BASE}/seasons/{season}/segments/0/leagues/{league_id}?{VIEWS}"
    return f"{BASE}/leagueHistory/{league_id}?seasonId={season}&{VIEWS}"


def players_url(season: int) -> str:
    return f"{BASE}/seasons/{season}/players?view=players_wl"


def _unwrap(payload):
    # leagueHistory returns a single-element list; the seasons endpoint returns
    # the object directly.
    if isinstance(payload, list) and payload:
        return payload[0]
    return payload


def import_seasons(conn, league_id: str, current_season: int, fetch,
                   max_back: int = 15) -> dict:
    """Walk seasons backward from current_season, newest first.

    `fetch(url) -> parsed JSON` is injected so tests can serve fixtures. It
    must raise on a missing season; two consecutive misses end the walk, which
    tolerates one gap year without running to `max_back` on every import.
    """
    picks, teams, leagues, directories = [], [], [], []
    seasons, misses = [], 0
    for season in range(current_season, current_season - max_back, -1):
        try:
            payload = _unwrap(fetch(season_url(league_id, season, current_season)))
        except Exception:
            misses += 1
            if misses >= 2 and seasons:
                break
            continue
        if not ((payload.get("draftDetail") or {}).get("picks")):
            misses += 1
            if misses >= 2 and seasons:
                break
            continue
        misses = 0
        settings = league_mod.from_espn(parse_settings(payload, season))
        if settings.draft_type != "SNAKE":
            raise ValueError(
                f"season {season} draft type is {settings.draft_type}; "
                "only SNAKE is supported")
        seasons.append(season)
        picks.append(parse_draft_picks(payload, season))
        teams.append(parse_draft_teams(payload, season))
        leagues.append({"season": season,
                        "settings_json": league_mod.to_json(settings)})
        directory = parse_player_directory(fetch(players_url(season)))
        directory["season"] = season
        directories.append(directory)

    if not seasons:
        raise ValueError("no drafted seasons found -- check the league id and login")

    all_picks = pd.concat(picks, ignore_index=True)
    directory = pd.concat(directories, ignore_index=True)
    all_picks = all_picks.merge(
        directory, on=["season", "espn_player_id"], how="left")

    write_table(conn, "draft_picks", all_picks)
    write_table(conn, "draft_teams", pd.concat(teams, ignore_index=True))
    write_table(conn, "league", pd.DataFrame(leagues))
    return {"seasons": seasons, "picks": len(all_picks)}


def validate_import(conn) -> list[str]:
    from scoring.board import _norm_name
    picks = read_table(conn, "draft_picks")
    teams = read_table(conn, "draft_teams")
    adp = read_table(conn, "historic_adp")
    settings = league_mod.load(conn)
    lines = []
    for season, grp in picks.groupby("season"):
        expected = settings.teams * settings.rounds
        managers = teams[teams["season"] == season]["manager"].nunique()
        if adp.empty:
            rate = 0.0
        else:
            season_adp = adp[adp["season"] == season]
            known = set(zip(season_adp["adp_name"].map(_norm_name),
                            season_adp["position"]))
            hit = grp.apply(
                lambda r: (_norm_name(r["player_name"]), r["position"]) in known,
                axis=1)
            rate = float(hit.mean()) if len(grp) else 0.0
        flag = "" if len(grp) == expected else f"  <-- expected {expected}"
        lines.append(f"  {season}: {len(grp)} picks, {managers} managers, "
                     f"ADP match {rate:.0%}{flag}")
        if rate and rate < 0.8:
            lines.append(f"         low ADP match for {season} -- "
                         "picks below the threshold are excluded from fitting")
    return lines


class EspnClient:
    """Playwright-backed fetcher that reuses a saved ESPN login.

    First run with no saved state (or a state ESPN has expired) opens a real
    browser window at the ESPN login page and waits for the human. Disney SSO
    handles 2FA and bot checks there, which is the only place they can be
    answered. Everything after that is headless.
    """

    def __init__(self, state_path: str = STATE_PATH):
        self.state_path = Path(state_path)
        self._pw = None
        self._browser = None
        self._context = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._open_context()
        return self

    def __exit__(self, *exc):
        for closer in (self._context, self._browser):
            if closer is not None:
                closer.close()
        if self._pw is not None:
            self._pw.stop()

    def _open_context(self):
        self._browser = self._pw.chromium.launch(headless=True)
        kwargs = {"storage_state": str(self.state_path)} if self.state_path.exists() else {}
        self._context = self._browser.new_context(**kwargs)

    def login(self):
        """Open a visible window and block until the user has signed in."""
        print("ESPN login required -- a browser window is opening.")
        print("Sign in, then return here; the import continues automatically.")
        browser = self._pw.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(LOGIN_URL)
        # espn_s2 is only set once the SSO flow completes.
        page.wait_for_function(
            "() => document.cookie.includes('espn_s2') || "
            "document.cookie.includes('SWID')", timeout=300_000)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        context.storage_state(path=str(self.state_path))
        context.close()
        browser.close()
        # Rebuild the headless context so it picks up the new cookies.
        self._context.close()
        self._browser.close()
        self._open_context()

    def get_json(self, url: str):
        response = self._context.request.get(url)
        if response.status in (401, 403):
            self.login()
            response = self._context.request.get(url)
        if response.status == 404:
            raise FileNotFoundError(url)
        if not response.ok:
            raise RuntimeError(f"{response.status} for {url}")
        return response.json()
```

Create `pipeline/import_league.py`:

```python
"""Import ESPN league history. Run: python -m pipeline.import_league <league>"""
import sys

from pipeline import sources
from pipeline.db import get_conn, write_table, record_freshness
from pipeline.espn_league import (EspnClient, import_seasons, parse_league_id,
                                  validate_import)
from scoring.config import CURRENT_SEASON


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m pipeline.import_league <league-url-or-id>")
        return 2
    league_id = parse_league_id(argv[1])
    conn = get_conn()
    with EspnClient() as client:
        summary = import_seasons(conn, league_id, CURRENT_SEASON, client.get_json)

    # Historical ADP is what makes "reach" measurable: a pick only means
    # something against where the market had that player THAT year.
    frames = []
    for season in summary["seasons"]:
        try:
            df = sources.fetch_adp(season)
        except Exception as e:
            print(f"  WARN historic ADP {season}: {e}")
            continue
        df = df.sort_values("adp").reset_index(drop=True)
        df["adp_rank"] = df.index + 1
        df["season"] = season
        frames.append(df[["season", "adp_name", "position", "adp_rank"]])
    if frames:
        rows = pd.concat(frames, ignore_index=True)
        write_table(conn, "historic_adp", rows)
        record_freshness(conn, "historic_adp", True, len(rows))

    print(f"Imported {summary['picks']} picks across "
          f"{len(summary['seasons'])} seasons:")
    print("\n".join(validate_import(conn)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

Add `import pandas as pd` to the top of `pipeline/import_league.py`.

Append to `requirements.txt`:

```
scipy>=1.11
playwright>=1.40
```

Add to `Makefile` (and add `espn-import` to `.PHONY`):

```make
espn-import: ## import ESPN draft history: make espn-import LEAGUE=<url-or-id>
	.venv/bin/python -m pipeline.import_league "$(LEAGUE)"
```

Add to `.gitignore`:

```
data/espn_state.json
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_espn_league.py -v`
Expected: all passed. No test constructs an `EspnClient`; the client is exercised only through the injected `fetch`.

- [ ] **Step 5: Install Playwright's browser**

Run: `.venv/bin/pip install -r requirements.txt && .venv/bin/playwright install chromium`
Expected: chromium downloaded.

- [ ] **Step 6: Commit**

```bash
git add pipeline/espn_league.py pipeline/import_league.py requirements.txt Makefile .gitignore tests/test_espn_league.py
git commit -m "feat: ESPN league import with Playwright auth and validation report"
```

---

## Phase 2 — Opponent model

### Task 5: Choice-set reconstruction

Rebuild, for each historical pick, the pool of players that were available at that moment.

**Files:**
- Create: `scoring/draft_model.py`
- Test: `tests/test_draft_model.py`

**Interfaces:**
- Consumes: `draft_picks`, `draft_teams`, `historic_adp` tables; `scoring.board._norm_name`
- Produces:
  - `scoring.draft_model.PickObservation` — a `NamedTuple` with fields `season: int, overall_pick: int, manager: str, chosen: int, pool: pd.DataFrame, roster: dict, recent: list`
    - `chosen` is the row index within `pool` of the player actually taken
    - `pool` has columns `norm, position, adp_rank`
    - `roster` maps position to count already on that manager's roster
    - `recent` is the positions of the previous five picks, newest first
  - `scoring.draft_model.build_observations(conn) -> list[PickObservation]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_draft_model.py`:

```python
import pandas as pd
from pipeline.db import get_conn, write_table
from scoring.draft_model import build_observations

def _seed(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    # Two managers, four picks, one season. ADP order: A, B, C, D.
    write_table(conn, "draft_picks", pd.DataFrame([
        {"season": 2025, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 11, "player_name": "Player A",
         "position": "RB", "nfl_team": "DET", "keeper": False},
        {"season": 2025, "overall_pick": 2, "round": 1, "round_pick": 2,
         "team_id": 2, "espn_player_id": 12, "player_name": "Player C",
         "position": "WR", "nfl_team": "GB", "keeper": False},
        {"season": 2025, "overall_pick": 3, "round": 2, "round_pick": 1,
         "team_id": 2, "espn_player_id": 13, "player_name": "Player B",
         "position": "RB", "nfl_team": "CHI", "keeper": False},
        {"season": 2025, "overall_pick": 4, "round": 2, "round_pick": 2,
         "team_id": 1, "espn_player_id": 14, "player_name": "Ghost",
         "position": "TE", "nfl_team": "MIN", "keeper": False},
    ]))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2025, "team_id": 2, "manager": "dan", "slot": 2},
    ]))
    write_table(conn, "historic_adp", pd.DataFrame([
        {"season": 2025, "adp_name": "Player A", "position": "RB", "adp_rank": 1},
        {"season": 2025, "adp_name": "Player B", "position": "RB", "adp_rank": 2},
        {"season": 2025, "adp_name": "Player C", "position": "WR", "adp_rank": 3},
        {"season": 2025, "adp_name": "Player D", "position": "TE", "adp_rank": 4},
    ]))
    return conn

def test_pool_shrinks_as_players_come_off_the_board(tmp_path):
    obs = build_observations(_seed(tmp_path))
    assert [len(o.pool) for o in obs] == [4, 3, 2]

def test_chosen_index_points_at_the_player_actually_taken(tmp_path):
    obs = build_observations(_seed(tmp_path))
    first = obs[0]
    assert first.pool.iloc[first.chosen]["norm"] == "player a"
    second = obs[1]
    assert second.pool.iloc[second.chosen]["norm"] == "player c"

def test_picks_with_no_adp_row_are_dropped(tmp_path):
    # "Ghost" is not in historic_adp, so pick 4 produces no observation.
    obs = build_observations(_seed(tmp_path))
    assert len(obs) == 3
    assert all(o.overall_pick != 4 for o in obs)

def test_roster_counts_reflect_that_managers_prior_picks_only(tmp_path):
    obs = build_observations(_seed(tmp_path))
    third = obs[2]                       # dan's second pick, overall 3
    assert third.manager == "dan"
    assert third.roster == {"WR": 1}

def test_recent_positions_are_newest_first(tmp_path):
    obs = build_observations(_seed(tmp_path))
    assert obs[2].recent == ["WR", "RB"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scoring.draft_model'`

- [ ] **Step 3: Write minimal implementation**

Create `scoring/draft_model.py`:

```python
"""Per-manager draft-pick model.

Each historical pick is treated as one choice from the set of players who were
available at that moment. That set is reconstructable: we know the full pick
order and that season's ADP pool, so subtracting everything taken before pick
N gives the pool the manager was actually choosing from.

Picks whose player has no ADP row that season are dropped from fitting. The
model's core feature is a player's position relative to market rank, which is
undefined without one -- and the import step reports how many picks this
removes per season so a bad join surfaces as a number, not a silent shrug.
"""
from typing import NamedTuple

import numpy as np
import pandas as pd

from pipeline.db import read_table
from scoring.board import _norm_name

RUN_WINDOW = 5


class PickObservation(NamedTuple):
    season: int
    overall_pick: int
    manager: str
    chosen: int
    pool: pd.DataFrame
    roster: dict
    recent: list


def build_observations(conn) -> list:
    picks = read_table(conn, "draft_picks")
    teams = read_table(conn, "draft_teams")
    adp = read_table(conn, "historic_adp")
    if picks.empty or teams.empty or adp.empty:
        return []

    picks = picks.merge(teams[["season", "team_id", "manager"]],
                        on=["season", "team_id"], how="left")
    picks = picks.assign(norm=picks["player_name"].map(_norm_name))
    adp = adp.assign(norm=adp["adp_name"].map(_norm_name))

    out = []
    for season, season_picks in picks.groupby("season"):
        pool = adp[adp["season"] == season][["norm", "position", "adp_rank"]]
        pool = pool.sort_values("adp_rank").reset_index(drop=True)
        available = pool.copy()
        rosters, recent = {}, []
        for _, pick in season_picks.sort_values("overall_pick").iterrows():
            key = (pick["norm"], pick["position"])
            match = available.index[(available["norm"] == pick["norm"])
                                    & (available["position"] == pick["position"])]
            if len(match) == 0:
                # No ADP row for this player that season -- unusable as an
                # observation, but the pick still consumed a roster spot and
                # still counts toward positional runs.
                rosters.setdefault(pick["manager"], {})
                rosters[pick["manager"]][pick["position"]] = (
                    rosters[pick["manager"]].get(pick["position"], 0) + 1)
                recent.insert(0, pick["position"])
                continue
            reset = available.reset_index(drop=True)
            chosen = int(reset.index[(reset["norm"] == pick["norm"])
                                     & (reset["position"] == pick["position"])][0])
            out.append(PickObservation(
                season=int(season), overall_pick=int(pick["overall_pick"]),
                manager=pick["manager"], chosen=chosen, pool=reset,
                roster=dict(rosters.get(pick["manager"], {})),
                recent=list(recent[:RUN_WINDOW])))
            available = available.drop(index=match)
            rosters.setdefault(pick["manager"], {})
            rosters[pick["manager"]][pick["position"]] = (
                rosters[pick["manager"]].get(pick["position"], 0) + 1)
            recent.insert(0, pick["position"])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_draft_model.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add scoring/draft_model.py tests/test_draft_model.py
git commit -m "feat: reconstruct historical draft choice sets"
```

---

### Task 6: Feature matrix

**Files:**
- Modify: `scoring/draft_model.py`
- Test: `tests/test_draft_model.py`

**Interfaces:**
- Consumes: `PickObservation` from Task 5, `scoring.league.LeagueSettings`
- Produces:
  - `scoring.draft_model.FEATURE_NAMES: list[str]` — exactly `["reach", "fall", "pos_RB", "pos_WR", "pos_TE", "pos_K", "pos_DST", "qb_early", "te_early", "need", "run"]` (QB is the dropped position baseline)
  - `scoring.draft_model.feature_matrix(obs: PickObservation, settings) -> np.ndarray` of shape `(len(obs.pool), len(FEATURE_NAMES))`
  - `scoring.draft_model.EARLY_ROUNDS = 3`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft_model.py`:

```python
import numpy as np
from scoring.draft_model import FEATURE_NAMES, feature_matrix
from scoring import league

def _settings():
    return league.default_settings()

def test_feature_matrix_shape_and_column_order(tmp_path):
    obs = build_observations(_seed(tmp_path))[0]
    X = feature_matrix(obs, _settings())
    assert X.shape == (len(obs.pool), len(FEATURE_NAMES))
    assert FEATURE_NAMES[0] == "reach" and FEATURE_NAMES[1] == "fall"

def test_reach_and_fall_are_nonnegative_and_mutually_exclusive(tmp_path):
    obs = build_observations(_seed(tmp_path))[2]     # overall pick 3
    X = feature_matrix(obs, _settings())
    reach, fall = X[:, 0], X[:, 1]
    assert (reach >= 0).all() and (fall >= 0).all()
    assert ((reach == 0) | (fall == 0)).all()

def test_reach_measures_rounds_of_reach_required(tmp_path):
    obs = build_observations(_seed(tmp_path))[0]      # overall pick 1, 8 teams
    X = feature_matrix(obs, _settings())
    ranks = obs.pool["adp_rank"].to_numpy()
    expected = np.maximum(0, ranks - obs.overall_pick) / 8
    assert np.allclose(X[:, 0], expected)

def test_need_is_one_while_short_of_a_starter(tmp_path):
    obs = build_observations(_seed(tmp_path))[2]      # dan has 1 WR
    X = feature_matrix(obs, _settings())
    need = X[:, FEATURE_NAMES.index("need")]
    positions = obs.pool["position"].tolist()
    # Default league starts 2 WR, so a second WR still counts as a need.
    assert need[positions.index("WR")] == 1.0 if "WR" in positions else True
    assert need[positions.index("RB")] == 1.0

def test_run_counts_same_position_picks_in_the_recent_window(tmp_path):
    obs = build_observations(_seed(tmp_path))[2]      # recent == ["WR", "RB"]
    X = feature_matrix(obs, _settings())
    run = X[:, FEATURE_NAMES.index("run")]
    positions = obs.pool["position"].tolist()
    assert run[positions.index("RB")] == 1.0 / 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'FEATURE_NAMES'`

- [ ] **Step 3: Write minimal implementation**

Append to `scoring/draft_model.py`:

```python
# QB is the dropped baseline: with a full set of position dummies plus an
# intercept-free softmax the columns would be collinear.
_POSITION_DUMMIES = ["RB", "WR", "TE", "K", "DST"]
FEATURE_NAMES = (["reach", "fall"]
                 + [f"pos_{p}" for p in _POSITION_DUMMIES]
                 + ["qb_early", "te_early", "need", "run"])
EARLY_ROUNDS = 3


def feature_matrix(obs: PickObservation, settings) -> np.ndarray:
    pool = obs.pool
    n = len(pool)
    ranks = pool["adp_rank"].to_numpy(dtype=float)
    positions = pool["position"].to_numpy()
    teams = max(settings.teams, 1)

    delta = ranks - obs.overall_pick
    reach = np.maximum(0.0, delta) / teams
    fall = np.maximum(0.0, -delta) / teams

    columns = [reach, fall]
    for pos in _POSITION_DUMMIES:
        columns.append((positions == pos).astype(float))

    # `overall_pick` is 1-indexed, so pick 8 in an 8-team league is round 1.
    round_no = (obs.overall_pick - 1) // teams + 1
    early = 1.0 if round_no <= EARLY_ROUNDS else 0.0
    columns.append((positions == "QB").astype(float) * early)
    columns.append((positions == "TE").astype(float) * early)

    starters = settings.starters
    need = np.array([1.0 if obs.roster.get(p, 0) < starters.get(p, 0) else 0.0
                     for p in positions])
    columns.append(need)

    recent = obs.recent[:RUN_WINDOW]
    run = np.array([recent.count(p) / RUN_WINDOW for p in positions])
    columns.append(run)

    return np.column_stack(columns) if n else np.zeros((0, len(FEATURE_NAMES)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_draft_model.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add scoring/draft_model.py tests/test_draft_model.py
git commit -m "feat: draft-pick feature matrix"
```

---

### Task 7: Conditional-logit fit with shrinkage

**Files:**
- Modify: `scoring/draft_model.py`
- Test: `tests/test_draft_model.py`

**Interfaces:**
- Consumes: `feature_matrix`, `PickObservation`
- Produces:
  - `scoring.draft_model.neg_log_likelihood(beta, X_list, chosen_list) -> tuple[float, np.ndarray]` — value and gradient
  - `scoring.draft_model.fit(X_list, chosen_list, prior=None, lam=0.0) -> np.ndarray`
  - `scoring.draft_model.log_likelihood(beta, X_list, chosen_list) -> float`
  - `scoring.draft_model.prepare(observations, settings) -> tuple[list[np.ndarray], list[int], list[str], list[int]]` returning `(X_list, chosen_list, managers, seasons)`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft_model.py`:

```python
from scoring.draft_model import fit, log_likelihood, neg_log_likelihood

def _synthetic(beta_true, n_choices=40, pool=20, seed=0):
    """Generate choices from a known beta so the fit can be checked for recovery."""
    rng = np.random.default_rng(seed)
    X_list, chosen = [], []
    for _ in range(n_choices):
        X = rng.normal(size=(pool, len(beta_true)))
        p = np.exp(X @ beta_true)
        p = p / p.sum()
        X_list.append(X)
        chosen.append(int(rng.choice(pool, p=p)))
    return X_list, chosen

def test_gradient_matches_finite_differences():
    beta = np.array([0.3, -0.7, 0.1])
    X_list, chosen = _synthetic(beta, n_choices=5, pool=6, seed=1)
    point = np.array([0.1, 0.2, -0.3])
    _, grad = neg_log_likelihood(point, X_list, chosen)
    eps = 1e-6
    for i in range(len(point)):
        bumped = point.copy()
        bumped[i] += eps
        numeric = (neg_log_likelihood(bumped, X_list, chosen)[0]
                   - neg_log_likelihood(point, X_list, chosen)[0]) / eps
        assert abs(numeric - grad[i]) < 1e-4

def test_fit_recovers_known_beta():
    beta_true = np.array([1.5, -1.0, 0.5])
    X_list, chosen = _synthetic(beta_true, n_choices=3000, pool=15, seed=2)
    beta_hat = fit(X_list, chosen)
    assert np.allclose(beta_hat, beta_true, atol=0.2)

def test_shrinkage_pulls_a_thin_fit_toward_the_prior():
    beta_true = np.array([1.5, -1.0, 0.5])
    prior = np.array([0.0, 0.0, 0.0])
    X_list, chosen = _synthetic(beta_true, n_choices=6, pool=15, seed=3)
    loose = fit(X_list, chosen, prior=prior, lam=0.01)
    tight = fit(X_list, chosen, prior=prior, lam=100.0)
    assert np.linalg.norm(tight - prior) < np.linalg.norm(loose - prior)

def test_log_likelihood_of_uniform_beta_is_pool_entropy():
    X_list, chosen = _synthetic(np.array([1.0, 0.0, 0.0]), n_choices=4, pool=10, seed=4)
    zero = np.zeros(3)
    assert abs(log_likelihood(zero, X_list, chosen) - 4 * np.log(1 / 10)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'fit'`

- [ ] **Step 3: Write minimal implementation**

Append to `scoring/draft_model.py`:

```python
from scipy.optimize import minimize


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def log_likelihood(beta, X_list, chosen_list) -> float:
    total = 0.0
    for X, k in zip(X_list, chosen_list):
        scores = X @ beta
        total += scores[k] - (scores.max() + np.log(np.exp(scores - scores.max()).sum()))
    return float(total)


def neg_log_likelihood(beta, X_list, chosen_list, prior=None, lam=0.0):
    """Value and gradient of the ridge-penalized negative log-likelihood.

    Convex in beta, which is why L-BFGS-B finds the global optimum rather than
    a local one -- the reason this uses scipy instead of a hand-rolled loop.
    """
    value = 0.0
    grad = np.zeros_like(beta, dtype=float)
    for X, k in zip(X_list, chosen_list):
        scores = X @ beta
        probs = _softmax(scores)
        value -= scores[k] - (scores.max()
                              + np.log(np.exp(scores - scores.max()).sum()))
        grad += probs @ X - X[k]
    if prior is not None and lam:
        diff = beta - prior
        value += lam * float(diff @ diff)
        grad += 2.0 * lam * diff
    return value, grad


def fit(X_list, chosen_list, prior=None, lam: float = 0.0) -> np.ndarray:
    n_features = X_list[0].shape[1] if X_list else len(FEATURE_NAMES)
    start = np.zeros(n_features) if prior is None else np.asarray(prior, dtype=float).copy()
    if not X_list:
        return start
    result = minimize(neg_log_likelihood, start,
                      args=(X_list, chosen_list, prior, lam),
                      jac=True, method="L-BFGS-B")
    return result.x


def prepare(observations, settings):
    X_list = [feature_matrix(o, settings) for o in observations]
    chosen = [o.chosen for o in observations]
    managers = [o.manager for o in observations]
    seasons = [o.season for o in observations]
    return X_list, chosen, managers, seasons
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_draft_model.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add scoring/draft_model.py tests/test_draft_model.py
git commit -m "feat: conditional-logit fit with ridge shrinkage"
```

---

### Task 8: Manager fitting, lambda selection, backtest, and profiles

**Files:**
- Modify: `scoring/draft_model.py`
- Create: `pipeline/fit_managers.py`
- Modify: `Makefile`
- Test: `tests/test_draft_model.py`

**Interfaces:**
- Consumes: everything from Tasks 5–7
- Produces:
  - `scoring.draft_model.select_lambda(X_list, chosen, seasons, prior, grid=None) -> float` — leave-one-season-out
  - `scoring.draft_model.fit_all(conn, settings=None) -> dict[str, np.ndarray]` — manager to coefficients, plus key `"__pooled__"`
  - `scoring.draft_model.describe(beta, pooled) -> str`
  - `scoring.draft_model.backtest(conn, settings=None) -> dict` with keys `holdout_season, top1, top5, logloss, adp_top1, adp_logloss, beats_adp`
  - `scoring.draft_model.write_profiles(conn, settings=None) -> pd.DataFrame` — writes the `manager_profiles` table with columns `manager, feature, value, pooled_value, n_picks, heldout_gain, uses_personal, summary`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft_model.py`:

```python
from scoring.draft_model import (backtest, describe, fit_all, select_lambda,
                                 write_profiles)

def _seed_many(tmp_path, seasons=(2023, 2024, 2025)):
    """Two managers with opposite tastes, repeated across seasons.

    `early` always takes the best RB available; `late` always takes the best
    WR. A fitted model should separate them on the position dummies.
    """
    conn = get_conn(str(tmp_path / "t.duckdb"))
    names = [f"Player {i}" for i in range(1, 21)]
    positions = ["RB" if i % 2 else "WR" for i in range(1, 21)]
    picks, adp = [], []
    for season in seasons:
        for i, (name, pos) in enumerate(zip(names, positions), start=1):
            adp.append({"season": season, "adp_name": name,
                        "position": pos, "adp_rank": i})
        taken = set()
        for pick_no in range(1, 9):
            manager_pos = "RB" if pick_no % 2 else "WR"
            team_id = 1 if pick_no % 2 else 2
            choice = next(n for n, p in zip(names, positions)
                          if p == manager_pos and n not in taken)
            taken.add(choice)
            picks.append({"season": season, "overall_pick": pick_no,
                          "round": (pick_no - 1) // 2 + 1,
                          "round_pick": (pick_no - 1) % 2 + 1,
                          "team_id": team_id, "espn_player_id": pick_no,
                          "player_name": choice,
                          "position": manager_pos, "nfl_team": "DET",
                          "keeper": False})
    write_table(conn, "draft_picks", pd.DataFrame(picks))
    write_table(conn, "draft_teams", pd.DataFrame(
        [{"season": s, "team_id": t, "manager": m, "slot": t}
         for s in seasons for t, m in ((1, "rbguy"), (2, "wrguy"))]))
    write_table(conn, "historic_adp", pd.DataFrame(adp))
    return conn

def test_select_lambda_returns_a_value_from_the_grid():
    beta_true = np.array([1.0, -0.5, 0.2])
    X_list, chosen = _synthetic(beta_true, n_choices=60, pool=10, seed=5)
    seasons = [2023] * 20 + [2024] * 20 + [2025] * 20
    lam = select_lambda(X_list, chosen, seasons, prior=np.zeros(3),
                        grid=[0.01, 1.0, 100.0])
    assert lam in (0.01, 1.0, 100.0)

def test_fit_all_separates_managers_with_opposite_tastes(tmp_path):
    fits = fit_all(_seed_many(tmp_path))
    assert set(fits) >= {"rbguy", "wrguy", "__pooled__"}
    rb_idx = FEATURE_NAMES.index("pos_RB")
    wr_idx = FEATURE_NAMES.index("pos_WR")
    assert fits["rbguy"][rb_idx] - fits["rbguy"][wr_idx] > \
           fits["wrguy"][rb_idx] - fits["wrguy"][wr_idx]

def test_backtest_reports_accuracy_against_an_adp_baseline(tmp_path):
    report = backtest(_seed_many(tmp_path))
    assert report["holdout_season"] == 2025
    assert 0.0 <= report["top1"] <= 1.0
    assert 0.0 <= report["top5"] <= 1.0
    assert isinstance(report["beats_adp"], bool)

def test_write_profiles_marks_thin_managers_as_pooled(tmp_path):
    conn = _seed_many(tmp_path, seasons=(2025,))     # 4 picks each: very thin
    profiles = write_profiles(conn)
    assert set(profiles.columns) == {
        "manager", "feature", "value", "pooled_value", "n_picks",
        "heldout_gain", "uses_personal", "summary"}
    assert (profiles["n_picks"] == 4).all()
    from pipeline.db import read_table
    assert not read_table(conn, "manager_profiles").empty

def test_describe_names_the_strongest_deviations():
    pooled = np.zeros(len(FEATURE_NAMES))
    beta = pooled.copy()
    beta[FEATURE_NAMES.index("reach")] = -2.0
    text = describe(beta, pooled)
    assert "reach" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_lambda'`

- [ ] **Step 3: Write minimal implementation**

Append to `scoring/draft_model.py`:

```python
from pipeline.db import write_table
from scoring import league as league_mod

LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MIN_PICKS_FOR_PERSONAL = 20

# Plain-language templates for the coefficients worth surfacing. Keyed by
# feature name; each maps (deviation from pooled) -> a phrase.
_PHRASES = {
    "reach": ("reaches for players the market ranks later",
              "avoids reaches, drafts in market order"),
    "fall": ("chases players who slide", "ignores players who slide"),
    "pos_RB": ("leans RB", "fades RB"),
    "pos_WR": ("leans WR", "fades WR"),
    "pos_TE": ("leans TE", "fades TE"),
    "pos_K": ("takes kickers early", "leaves kickers late"),
    "pos_DST": ("takes defenses early", "leaves defenses late"),
    "qb_early": ("early-QB guy", "waits on QB"),
    "te_early": ("early-TE guy", "waits on TE"),
    "need": ("fills starting slots first", "ignores roster needs"),
    "run": ("chases positional runs", "fades positional runs"),
}


def select_lambda(X_list, chosen_list, seasons, prior, grid=None) -> float:
    """Leave-one-season-out cross-validation over the ridge strength.

    Seasons, not random folds: picks inside one draft are not independent of
    each other, so a random split would leak the same draft across train and
    test and pick a lambda that is too loose.
    """
    grid = grid or LAMBDA_GRID
    unique = sorted(set(seasons))
    if len(unique) < 2:
        return grid[-1]                      # one season: shrink hard
    best, best_ll = grid[-1], -np.inf
    for lam in grid:
        total = 0.0
        for holdout in unique:
            train = [i for i, s in enumerate(seasons) if s != holdout]
            test = [i for i, s in enumerate(seasons) if s == holdout]
            if not train or not test:
                continue
            beta = fit([X_list[i] for i in train], [chosen_list[i] for i in train],
                       prior=prior, lam=lam)
            total += log_likelihood(beta, [X_list[i] for i in test],
                                    [chosen_list[i] for i in test])
        if total > best_ll:
            best, best_ll = lam, total
    return best


def fit_all(conn, settings=None) -> dict:
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    if not observations:
        return {}
    X_list, chosen, managers, seasons = prepare(observations, settings)
    pooled = fit(X_list, chosen)
    fits = {"__pooled__": pooled}
    for manager in sorted(set(managers)):
        idx = [i for i, m in enumerate(managers) if m == manager]
        Xm = [X_list[i] for i in idx]
        cm = [chosen[i] for i in idx]
        sm = [seasons[i] for i in idx]
        lam = select_lambda(Xm, cm, sm, prior=pooled)
        fits[manager] = fit(Xm, cm, prior=pooled, lam=lam)
    return fits


def _heldout_gain(X_list, chosen, seasons, pooled) -> float:
    """Per-pick log-likelihood advantage of a personal fit over pooled.

    Positive means the manager's own coefficients predict held-out picks
    better than the league-wide ones. Negative means they do not, and the
    simulator should use pooled for that manager.
    """
    unique = sorted(set(seasons))
    if len(unique) < 2:
        return -np.inf
    personal_ll = pooled_ll = 0.0
    n = 0
    for holdout in unique:
        train = [i for i, s in enumerate(seasons) if s != holdout]
        test = [i for i, s in enumerate(seasons) if s == holdout]
        if not train or not test:
            continue
        lam = select_lambda([X_list[i] for i in train], [chosen[i] for i in train],
                            [seasons[i] for i in train], prior=pooled)
        beta = fit([X_list[i] for i in train], [chosen[i] for i in train],
                   prior=pooled, lam=lam)
        Xt = [X_list[i] for i in test]
        ct = [chosen[i] for i in test]
        personal_ll += log_likelihood(beta, Xt, ct)
        pooled_ll += log_likelihood(pooled, Xt, ct)
        n += len(test)
    return (personal_ll - pooled_ll) / n if n else -np.inf


def describe(beta, pooled, top: int = 3) -> str:
    diff = np.asarray(beta) - np.asarray(pooled)
    order = np.argsort(-np.abs(diff))
    phrases = []
    for i in order[:top]:
        name = FEATURE_NAMES[i]
        if name not in _PHRASES or abs(diff[i]) < 0.05:
            continue
        high, low = _PHRASES[name]
        phrases.append(high if diff[i] > 0 else low)
    return ", ".join(phrases) if phrases else "drafts close to league average"


def write_profiles(conn, settings=None) -> pd.DataFrame:
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    if not observations:
        empty = pd.DataFrame(columns=[
            "manager", "feature", "value", "pooled_value", "n_picks",
            "heldout_gain", "uses_personal", "summary"])
        write_table(conn, "manager_profiles", empty)
        return empty

    X_list, chosen, managers, seasons = prepare(observations, settings)
    pooled = fit(X_list, chosen)
    rows = []
    for manager in sorted(set(managers)):
        idx = [i for i, m in enumerate(managers) if m == manager]
        Xm, cm = [X_list[i] for i in idx], [chosen[i] for i in idx]
        sm = [seasons[i] for i in idx]
        lam = select_lambda(Xm, cm, sm, prior=pooled)
        beta = fit(Xm, cm, prior=pooled, lam=lam)
        gain = _heldout_gain(Xm, cm, sm, pooled)
        uses_personal = bool(len(idx) >= MIN_PICKS_FOR_PERSONAL and gain > 0)
        effective = beta if uses_personal else pooled
        summary = describe(effective, pooled) if uses_personal else \
            "league average, not enough signal"
        for i, name in enumerate(FEATURE_NAMES):
            rows.append({"manager": manager, "feature": name,
                         "value": float(beta[i]), "pooled_value": float(pooled[i]),
                         "n_picks": len(idx),
                         "heldout_gain": float(gain) if np.isfinite(gain) else None,
                         "uses_personal": uses_personal, "summary": summary})
    profiles = pd.DataFrame(rows)
    write_table(conn, "manager_profiles", profiles)
    return profiles


def backtest(conn, settings=None) -> dict:
    """Hold out the newest season and score against an ADP-only baseline.

    If the fitted model does not beat "the market's next-best player is next
    off the board", that is the finding, and the board must not present
    simulator output as authoritative.
    """
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    if not observations:
        return {"holdout_season": None, "top1": 0.0, "top5": 0.0,
                "logloss": float("inf"), "adp_top1": 0.0,
                "adp_logloss": float("inf"), "beats_adp": False}
    X_list, chosen, managers, seasons = prepare(observations, settings)
    holdout = max(seasons)
    train = [i for i, s in enumerate(seasons) if s != holdout]
    test = [i for i, s in enumerate(seasons) if s == holdout]
    pooled = fit([X_list[i] for i in train], [chosen[i] for i in train]) \
        if train else np.zeros(len(FEATURE_NAMES))

    fits = {}
    for manager in set(managers):
        idx = [i for i in train if managers[i] == manager]
        if len(idx) < MIN_PICKS_FOR_PERSONAL:
            fits[manager] = pooled
            continue
        lam = select_lambda([X_list[i] for i in idx], [chosen[i] for i in idx],
                            [seasons[i] for i in idx], prior=pooled)
        fits[manager] = fit([X_list[i] for i in idx], [chosen[i] for i in idx],
                            prior=pooled, lam=lam)

    hits1 = hits5 = 0
    ll = adp_ll = 0.0
    adp_hits1 = 0
    for i in test:
        X, k = X_list[i], chosen[i]
        probs = _softmax(X @ fits.get(managers[i], pooled))
        order = np.argsort(-probs)
        hits1 += int(order[0] == k)
        hits5 += int(k in order[:5])
        ll += np.log(max(probs[k], 1e-12))
        # ADP baseline: the pool is sorted by adp_rank, so index 0 is the
        # market's next player, and the baseline is uniform-free -- it always
        # predicts index 0 and assigns a geometric-ish distribution otherwise.
        adp_hits1 += int(k == 0)
        adp_probs = np.full(len(X), 1.0 / len(X))
        adp_ll += np.log(adp_probs[k])
    n = max(len(test), 1)
    report = {"holdout_season": int(holdout), "top1": hits1 / n, "top5": hits5 / n,
              "logloss": -ll / n, "adp_top1": adp_hits1 / n,
              "adp_logloss": -adp_ll / n}
    report["beats_adp"] = bool(report["logloss"] < report["adp_logloss"])
    return report
```

Create `pipeline/fit_managers.py`:

```python
"""Fit manager models and write profiles. Run: python -m pipeline.fit_managers"""
import sys

from pipeline.db import get_conn, record_freshness
from scoring.draft_model import backtest, write_profiles


def main() -> int:
    conn = get_conn()
    profiles = write_profiles(conn)
    if profiles.empty:
        print("No draft history found -- run `make espn-import` first.")
        return 1
    record_freshness(conn, "manager_profiles", True, len(profiles))

    report = backtest(conn)
    personal = profiles[profiles["uses_personal"]]["manager"].nunique()
    total = profiles["manager"].nunique()
    print(f"Fitted {total} managers ({personal} with personal models, "
          f"{total - personal} pooled).")
    print(f"Backtest on {report['holdout_season']}: "
          f"top-1 {report['top1']:.0%}, top-5 {report['top5']:.0%}, "
          f"log-loss {report['logloss']:.3f} "
          f"(ADP baseline {report['adp_logloss']:.3f})")
    if not report["beats_adp"]:
        print("WARNING: the fitted model does not beat the ADP baseline "
              "out of sample. Treat simulator output as indicative only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Add to `Makefile` (and to `.PHONY`):

```make
fit-managers: ## fit per-manager pick models from imported draft history
	.venv/bin/python -m pipeline.fit_managers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_draft_model.py -v`
Expected: 19 passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add scoring/draft_model.py pipeline/fit_managers.py Makefile tests/test_draft_model.py
git commit -m "feat: manager model fitting, backtest, and profiles"
```

---

## Phase 3 — Simulator

### Task 9: Projections, lineup optimizer, and roster value

**Files:**
- Create: `scoring/draft_sim.py`
- Test: `tests/test_draft_sim.py`

**Interfaces:**
- Consumes: `scoring.league.LeagueSettings`, the board frame from `build_board`, `espn_adp` table
- Produces:
  - `scoring.draft_sim.FLEX_POSITIONS = ("RB", "WR", "TE")`
  - `scoring.draft_sim.projections(conn, board: pd.DataFrame) -> pd.Series` indexed by `player_id`
  - `scoring.draft_sim.best_lineup_points(roster: list[tuple[str, float]], settings) -> float` — roster is `(position, projected_points)` pairs
  - `scoring.draft_sim.roster_value(roster: list[tuple[str, float, float]], settings) -> float` — triples of `(position, projected_points, durability)` where durability is the 0–100 factor

- [ ] **Step 1: Write the failing test**

Create `tests/test_draft_sim.py`:

```python
import numpy as np
import pandas as pd
from pipeline.db import get_conn, write_table
from scoring import league
from scoring.draft_sim import best_lineup_points, projections, roster_value

S = league.default_settings()   # QB/2RB/2WR/TE/2FLEX/K/DST, 5 bench

def test_best_lineup_fills_dedicated_slots_then_flex():
    roster = [("QB", 300.0), ("RB", 250.0), ("RB", 200.0), ("RB", 180.0),
              ("WR", 240.0), ("WR", 220.0), ("WR", 190.0),
              ("TE", 150.0), ("K", 120.0), ("DST", 110.0)]
    # Starters: QB 300, RB 250+200, WR 240+220, TE 150, K 120, DST 110 = 1590
    # FLEX x2 take the best leftovers: RB 180 and WR 190 = 370
    assert best_lineup_points(roster, S) == 1960.0

def test_best_lineup_ignores_surplus_beyond_flex():
    roster = [("WR", 100.0)] * 10
    # 2 WR starters + 2 FLEX = 4 slots filled, the other six are bench.
    assert best_lineup_points(roster, S) == 400.0

def test_best_lineup_handles_an_unfilled_slot():
    assert best_lineup_points([("QB", 300.0)], S) == 300.0

def test_roster_value_adds_backup_insurance():
    healthy = [("QB", 300.0, 100.0)]
    starter_only = roster_value(healthy, S)
    with_backup = roster_value(healthy + [("QB", 200.0, 100.0)], S)
    # A backup behind a perfectly durable starter adds nothing.
    assert with_backup == starter_only

def test_roster_value_backup_matters_more_behind_a_fragile_starter():
    fragile = [("RB", 300.0, 0.0)]
    solo = roster_value(fragile, S)
    covered = roster_value(fragile + [("RB", 200.0, 100.0)], S)
    assert covered > solo

def test_projections_prefer_espn_then_fall_back_to_weighted_ppg(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "espn_adp", pd.DataFrame([
        {"espn_id": 1, "espn_name": "Has Projection", "position": "WR",
         "espn_adp": 5.0, "espn_ppr_rank": 3, "espn_proj": 289.0}]))
    board = pd.DataFrame([
        {"player_id": "p1", "name": "Has Projection", "position": "WR",
         "team": "DET", "stats": {"ppg": 15.0}},
        {"player_id": "p2", "name": "No Projection", "position": "WR",
         "team": "GB", "stats": {"ppg": 10.0}},
        {"player_id": "p3", "name": "Nothing At All", "position": "K",
         "team": "GB", "stats": None},
    ])
    proj = projections(conn, board)
    assert proj["p1"] == 289.0
    assert proj["p2"] == 170.0                 # 10.0 ppg x 17
    assert proj["p3"] > 0                       # position floor, never NaN
    assert not proj.isna().any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_sim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scoring.draft_sim'`

- [ ] **Step 3: Write minimal implementation**

Create `scoring/draft_sim.py`:

```python
"""Monte Carlo draft simulator.

Roster value is the projected points of the best legal starting lineup, plus
an insurance term for the bench. Without that term every pick past the last
starting slot is worth exactly zero and the simulator's late rounds become
noise; with it, a backup is worth the games his starter is expected to miss
times the drop-off to him.
"""
import numpy as np
import pandas as pd

from pipeline.db import read_table
from scoring.board import _norm_name

FLEX_POSITIONS = ("RB", "WR", "TE")
GAMES = 17
# Floor for players with no projection and no stat history, per position, so
# a K or a rookie DST never lands as NaN inside the lineup optimizer.
POSITION_FLOOR = {"QB": 180.0, "RB": 80.0, "WR": 80.0, "TE": 60.0,
                  "K": 110.0, "DST": 100.0}


def projections(conn, board: pd.DataFrame) -> pd.Series:
    """Projected season points per player_id.

    Ladder: ESPN's own season projection, then recency-weighted PPG scaled to
    a full season, then a per-position floor.
    """
    espn = read_table(conn, "espn_adp")
    lookup = {}
    if not espn.empty and "espn_proj" in espn.columns:
        valid = espn.dropna(subset=["espn_proj"])
        valid = valid[valid["espn_proj"] > 0]
        for _, row in valid.iterrows():
            lookup[(_norm_name(row["espn_name"]), row["position"])] = float(row["espn_proj"])

    values = []
    for _, row in board.iterrows():
        key = (_norm_name(row["name"]), row["position"])
        proj = lookup.get(key)
        if proj is None:
            stats = row.get("stats")
            ppg = stats.get("ppg") if isinstance(stats, dict) else None
            proj = float(ppg) * GAMES if ppg else None
        if proj is None or not np.isfinite(proj):
            proj = POSITION_FLOOR.get(row["position"], 80.0)
        values.append(proj)
    return pd.Series(values, index=board["player_id"].to_numpy(), dtype=float)


def best_lineup_points(roster, settings) -> float:
    """Points of the best legal starting lineup.

    Greedy is optimal here: FLEX accepts a superset of no dedicated slot's
    eligibility and every other slot is single-position, so filling dedicated
    slots best-first and handing FLEX the leftovers can never be beaten.
    """
    by_position = {}
    for pos, points in roster:
        by_position.setdefault(pos, []).append(points)
    for values in by_position.values():
        values.sort(reverse=True)

    total = 0.0
    leftovers = []
    for pos, count in settings.starters.items():
        values = by_position.get(pos, [])
        total += sum(values[:count])
        if pos in FLEX_POSITIONS:
            leftovers.extend(values[count:])
    leftovers.sort(reverse=True)
    return total + sum(leftovers[:settings.flex_slots])


def roster_value(roster, settings) -> float:
    """Starting-lineup points plus bench insurance.

    Each starter is expected to miss `(1 - durability/100) * 17` games; the
    best bench player at that position covers those games at his own rate, so
    the insurance credit is missed-game share times the drop-off to him.
    """
    starters_only = [(pos, points) for pos, points, _ in roster]
    total = best_lineup_points(starters_only, settings)

    by_position = {}
    for pos, points, durability in roster:
        by_position.setdefault(pos, []).append((points, durability))
    for values in by_position.values():
        values.sort(reverse=True)

    for pos, count in settings.starters.items():
        values = by_position.get(pos, [])
        if len(values) <= count:
            continue
        backup_points = values[count][0]
        for starter_points, durability in values[:count]:
            missed = max(0.0, 1.0 - (durability or 50.0) / 100.0)
            total += missed * min(backup_points, starter_points)
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_draft_sim.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add scoring/draft_sim.py tests/test_draft_sim.py
git commit -m "feat: projections, lineup optimizer, and roster valuation"
```

---

### Task 10: Rollout engine

**Files:**
- Modify: `scoring/draft_sim.py`
- Test: `tests/test_draft_sim.py`

**Interfaces:**
- Consumes: `scoring.draft_model.FEATURE_NAMES`, `feature_matrix`, `PickObservation`, Task 9 valuation
- Produces:
  - `scoring.draft_sim.snake_slots(teams: int, rounds: int) -> list[int]` — 1-indexed slot on the clock for each overall pick
  - `scoring.draft_sim.SimPool` — a `NamedTuple` with `player_id, norm, position, adp_rank, points, durability` as aligned numpy arrays
  - `scoring.draft_sim.build_pool(conn, board, settings) -> SimPool`
  - `scoring.draft_sim.rollout(pool, settings, slot_managers, my_slot, taken, betas, rng, forced=None) -> float`
    - `taken` is a boolean array over the pool marking players already off the board
    - `forced` is an optional pool index my slot must take at its first turn
    - returns my end-of-draft `roster_value`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft_sim.py`:

```python
from scoring.draft_model import FEATURE_NAMES
from scoring.draft_sim import SimPool, rollout, snake_slots

def test_snake_slots_reverses_every_other_round():
    assert snake_slots(4, 3) == [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4]

def _pool(n=60):
    positions = np.array(["RB", "WR", "QB", "TE", "K", "DST"] * (n // 6))
    return SimPool(
        player_id=np.array([f"p{i}" for i in range(n)]),
        norm=np.array([f"player {i}" for i in range(n)]),
        position=positions,
        adp_rank=np.arange(1, n + 1, dtype=float),
        points=np.linspace(300.0, 60.0, n),
        durability=np.full(n, 90.0))

def _flat_betas(managers):
    return {m: np.zeros(len(FEATURE_NAMES)) for m in managers}

def test_rollout_is_deterministic_under_a_fixed_seed():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    args = (pool, S, slots, 4, taken, _flat_betas(slots.values()))
    a = rollout(*args, rng=np.random.default_rng(7))
    b = rollout(*args, rng=np.random.default_rng(7))
    assert a == b

def test_rollout_returns_a_positive_roster_value():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    value = rollout(pool, S, slots, 4, taken, _flat_betas(slots.values()),
                    rng=np.random.default_rng(1))
    assert value > 0

def test_forcing_the_top_player_beats_forcing_the_worst():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    betas = _flat_betas(slots.values())
    best = np.mean([rollout(pool, S, slots, 1, taken, betas,
                            rng=np.random.default_rng(i), forced=0)
                    for i in range(20)])
    worst = np.mean([rollout(pool, S, slots, 1, taken, betas,
                             rng=np.random.default_rng(i), forced=len(pool.points) - 1)
                     for i in range(20)])
    assert best > worst

def test_rollout_respects_already_taken_players():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    taken[:40] = True                    # only 20 players left, 8 teams x 15 rounds
    value = rollout(pool, S, slots, 1, taken, _flat_betas(slots.values()),
                    rng=np.random.default_rng(3))
    assert value > 0                     # runs out of players without crashing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_sim.py -v`
Expected: FAIL with `ImportError: cannot import name 'snake_slots'`

- [ ] **Step 3: Write minimal implementation**

Append to `scoring/draft_sim.py`:

```python
from typing import NamedTuple

from scoring.draft_model import EARLY_ROUNDS, FEATURE_NAMES, RUN_WINDOW

_POSITION_DUMMY_INDEX = {
    pos: FEATURE_NAMES.index(f"pos_{pos}")
    for pos in ("RB", "WR", "TE", "K", "DST")
}
_REACH = FEATURE_NAMES.index("reach")
_FALL = FEATURE_NAMES.index("fall")
_QB_EARLY = FEATURE_NAMES.index("qb_early")
_TE_EARLY = FEATURE_NAMES.index("te_early")
_NEED = FEATURE_NAMES.index("need")
_RUN = FEATURE_NAMES.index("run")


class SimPool(NamedTuple):
    player_id: np.ndarray
    norm: np.ndarray
    position: np.ndarray
    adp_rank: np.ndarray
    points: np.ndarray
    durability: np.ndarray


def snake_slots(teams: int, rounds: int) -> list:
    order = []
    for r in range(rounds):
        forward = list(range(1, teams + 1))
        order.extend(forward if r % 2 == 0 else forward[::-1])
    return order


def build_pool(conn, board: pd.DataFrame, settings) -> SimPool:
    points = projections(conn, board)
    ranked = board.copy()
    ranked["proj"] = ranked["player_id"].map(points)
    # Market rank is the model's notion of "where the board thinks he goes";
    # players the market never ranked sit at the back rather than dropping out,
    # since an opponent can still take them.
    ranked["market_rank"] = ranked["market_rank"].fillna(len(ranked) + 1)
    ranked = ranked.sort_values("market_rank").reset_index(drop=True)
    return SimPool(
        player_id=ranked["player_id"].to_numpy(),
        norm=ranked["name"].map(_norm_name).to_numpy(),
        position=ranked["position"].to_numpy(),
        adp_rank=np.arange(1, len(ranked) + 1, dtype=float),
        points=ranked["proj"].to_numpy(dtype=float),
        durability=ranked["durability"].fillna(50.0).to_numpy(dtype=float))


def _live_features(pool, available, overall_pick, roster, recent, settings):
    """Feature matrix for the currently available players, mirroring
    draft_model.feature_matrix exactly -- the fitted coefficients only mean
    anything against the same feature definitions they were fitted on."""
    positions = pool.position[available]
    ranks = pool.adp_rank[available]
    teams = max(settings.teams, 1)
    n = len(positions)
    X = np.zeros((n, len(FEATURE_NAMES)))

    delta = ranks - overall_pick
    X[:, _REACH] = np.maximum(0.0, delta) / teams
    X[:, _FALL] = np.maximum(0.0, -delta) / teams
    for pos, col in _POSITION_DUMMY_INDEX.items():
        X[:, col] = (positions == pos)

    round_no = (overall_pick - 1) // teams + 1
    early = 1.0 if round_no <= EARLY_ROUNDS else 0.0
    X[:, _QB_EARLY] = (positions == "QB") * early
    X[:, _TE_EARLY] = (positions == "TE") * early

    starters = settings.starters
    X[:, _NEED] = [1.0 if roster.get(p, 0) < starters.get(p, 0) else 0.0
                   for p in positions]
    window = recent[:RUN_WINDOW]
    X[:, _RUN] = [window.count(p) / RUN_WINDOW for p in positions]
    return X


def _roster_cap(settings) -> dict:
    """Most of each position anyone will carry. Learned coefficients cannot
    express a hard ceiling, so it is imposed as a mask instead."""
    caps = {pos: n + 2 for pos, n in settings.starters.items()}
    caps["QB"] = min(caps.get("QB", 3), 3)
    caps["K"] = 1
    caps["DST"] = 1
    return caps


# Greedy only weighs the best available players by projection. Scanning the
# whole pool would make every rollout O(pool) roster valuations, which at
# ~500 players x 15 picks x thousands of rollouts is the difference between
# seconds and hours -- and a player outside the top 40 by projection can
# never be the greedy pick anyway.
GREEDY_CANDIDATES = 40


def _greedy_choice(pool, available, roster, settings, caps):
    """My in-rollout policy: the available player who most increases roster
    value. One-ply greedy, which is what makes a rollout cheap enough to run
    thousands of times; the search in Task 11 is what looks further ahead."""
    current = [(pool.position[i], pool.points[i], pool.durability[i])
               for i in roster["indices"]]
    base = roster_value(current, settings)
    shortlist = available[np.argsort(-pool.points[available])][:GREEDY_CANDIDATES]
    best_idx, best_gain = None, -np.inf
    for i in shortlist:
        pos = pool.position[i]
        if roster["counts"].get(pos, 0) >= caps.get(pos, 99):
            continue
        gain = roster_value(current + [(pos, pool.points[i], pool.durability[i])],
                            settings) - base
        if gain > best_gain:
            best_idx, best_gain = i, gain
    return best_idx if best_idx is not None else (available[0] if len(available) else None)


def rollout(pool, settings, slot_managers, my_slot, taken, betas, rng,
            forced=None) -> float:
    n = len(pool.player_id)
    gone = taken.copy()
    caps = _roster_cap(settings)
    rounds = settings.rounds
    slots = snake_slots(settings.teams, rounds)
    rosters = {slot: {"counts": {}, "indices": []}
               for slot in range(1, settings.teams + 1)}
    recent = []
    already = int(gone.sum())

    for offset, slot in enumerate(slots[already:], start=already):
        available = np.flatnonzero(~gone)
        if len(available) == 0:
            break
        overall_pick = offset + 1
        roster = rosters[slot]
        if slot == my_slot:
            if forced is not None and not gone[forced]:
                choice = forced
                forced = None
            else:
                choice = _greedy_choice(pool, available, roster, settings, caps)
        else:
            beta = betas.get(slot_managers.get(slot))
            if beta is None:
                beta = np.zeros(len(FEATURE_NAMES))
            X = _live_features(pool, available, overall_pick, roster["counts"],
                               recent, settings)
            scores = X @ beta
            for j, i in enumerate(available):
                pos = pool.position[i]
                if roster["counts"].get(pos, 0) >= caps.get(pos, 99):
                    scores[j] = -np.inf
            if not np.isfinite(scores).any():
                choice = int(available[0])
            else:
                shifted = scores - np.nanmax(scores[np.isfinite(scores)])
                weights = np.where(np.isfinite(shifted), np.exp(shifted), 0.0)
                total = weights.sum()
                choice = int(available[0]) if total <= 0 else \
                    int(rng.choice(available, p=weights / total))
        if choice is None:
            break
        gone[choice] = True
        pos = pool.position[choice]
        rosters[slot]["counts"][pos] = rosters[slot]["counts"].get(pos, 0) + 1
        rosters[slot]["indices"].append(choice)
        recent.insert(0, pos)

    mine = rosters[my_slot]["indices"]
    return roster_value(
        [(pool.position[i], pool.points[i], pool.durability[i]) for i in mine],
        settings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_draft_sim.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add scoring/draft_sim.py tests/test_draft_sim.py
git commit -m "feat: draft rollout engine"
```

---

### Task 11: Candidate search and result persistence

**Files:**
- Modify: `scoring/draft_sim.py`
- Create: `pipeline/run_sim.py`
- Modify: `Makefile`
- Test: `tests/test_draft_sim.py`

**Interfaces:**
- Consumes: Task 10 rollout, `scoring.draft_model.fit_all`
- Produces:
  - `scoring.draft_sim.search_pick(pool, settings, slot_managers, my_slot, taken, betas, n_rollouts, n_candidates=12, seed=0) -> pd.DataFrame` with columns `player_id, ev, se, rank`
  - `scoring.draft_sim.survival(pool, settings, slot_managers, my_slot, taken, betas, n_rollouts, seed=0) -> pd.DataFrame` with columns `player_id, avail_pct`
  - `scoring.draft_sim.run_sim(conn, my_slot, slot_managers, n_rollouts=300, seed=0) -> str` — returns the run id and writes `sim_results` and `sim_survival`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft_sim.py`:

```python
from scoring.draft_sim import search_pick, survival

def test_search_pick_ranks_the_best_candidate_first():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    out = search_pick(pool, S, slots, 1, taken, _flat_betas(slots.values()),
                      n_rollouts=25, n_candidates=6, seed=11)
    assert list(out.columns) == ["player_id", "ev", "se", "rank"]
    assert out["rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert out["ev"].is_monotonic_decreasing
    assert (out["se"] >= 0).all()

def test_search_pick_uses_common_random_numbers():
    # Same seed, same candidates -> byte-identical EVs across calls.
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    betas = _flat_betas(slots.values())
    a = search_pick(pool, S, slots, 1, taken, betas, n_rollouts=15,
                    n_candidates=4, seed=5)
    b = search_pick(pool, S, slots, 1, taken, betas, n_rollouts=15,
                    n_candidates=4, seed=5)
    assert a["ev"].tolist() == b["ev"].tolist()

def _adp_betas(managers):
    """Opponents who follow market order: a negative `reach` coefficient
    penalizes players whose ADP rank sits later than the current pick."""
    beta = np.zeros(len(FEATURE_NAMES))
    beta[FEATURE_NAMES.index("reach")] = -3.0
    return {m: beta for m in managers}

def test_survival_probabilities_are_between_zero_and_one_and_favor_late_adp():
    pool = _pool()
    slots = {i: f"m{i}" for i in range(1, 9)}
    taken = np.zeros(len(pool.player_id), dtype=bool)
    out = survival(pool, S, slots, 8, taken, _adp_betas(slots.values()),
                   n_rollouts=40, seed=2)
    assert ((out["avail_pct"] >= 0) & (out["avail_pct"] <= 1)).all()
    first = out[out["player_id"] == "p0"]["avail_pct"].iloc[0]
    last = out[out["player_id"] == "p59"]["avail_pct"].iloc[0]
    assert last > first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_sim.py -v`
Expected: FAIL with `ImportError: cannot import name 'search_pick'`

- [ ] **Step 3: Write minimal implementation**

Append to `scoring/draft_sim.py`:

```python
from pipeline.db import write_table

DEFAULT_ROLLOUTS = 300
DEFAULT_CANDIDATES = 12


def _next_pick_for(settings, my_slot, already) -> int:
    slots = snake_slots(settings.teams, settings.rounds)
    for offset in range(already, len(slots)):
        if slots[offset] == my_slot:
            return offset + 1
    return len(slots) + 1


def search_pick(pool, settings, slot_managers, my_slot, taken, betas,
                n_rollouts: int = DEFAULT_ROLLOUTS,
                n_candidates: int = DEFAULT_CANDIDATES, seed: int = 0):
    """Expected end-of-draft roster value for each candidate at my next pick.

    Candidates are the best available by market rank and by projection, since
    those two disagree exactly where the interesting decisions are.

    Rollout i uses seed (seed, i) for every candidate -- common random numbers,
    so all candidates face identical opponent behavior and the comparison
    between them is far less noisy than independent sampling at the same cost.
    """
    available = np.flatnonzero(~taken)
    if len(available) == 0:
        return pd.DataFrame(columns=["player_id", "ev", "se", "rank"])
    by_market = available[np.argsort(pool.adp_rank[available])][:n_candidates]
    by_points = available[np.argsort(-pool.points[available])][:n_candidates]
    candidates = list(dict.fromkeys(list(by_market) + list(by_points)))[:n_candidates]

    rows = []
    for idx in candidates:
        values = np.array([
            rollout(pool, settings, slot_managers, my_slot, taken, betas,
                    rng=np.random.default_rng([seed, i]), forced=int(idx))
            for i in range(n_rollouts)])
        rows.append({"player_id": pool.player_id[idx],
                     "ev": float(values.mean()),
                     "se": float(values.std(ddof=1) / np.sqrt(len(values)))
                     if len(values) > 1 else 0.0})
    out = pd.DataFrame(rows).sort_values("ev", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out[["player_id", "ev", "se", "rank"]]


def survival(pool, settings, slot_managers, my_slot, taken, betas,
             n_rollouts: int = DEFAULT_ROLLOUTS, seed: int = 0):
    """Probability each player is still available when my next turn arrives.

    Counted from the same rollout machinery, but stopping at my next pick
    rather than running the draft out -- this is the "who can I wait on"
    number, and it only depends on what happens before my turn.
    """
    already = int(taken.sum())
    target = _next_pick_for(settings, my_slot, already)
    slots = snake_slots(settings.teams, settings.rounds)
    caps = _roster_cap(settings)
    counts = np.zeros(len(pool.player_id))

    for i in range(n_rollouts):
        rng = np.random.default_rng([seed, i])
        gone = taken.copy()
        rosters = {slot: {} for slot in range(1, settings.teams + 1)}
        recent = []
        for offset in range(already, min(target - 1, len(slots))):
            slot = slots[offset]
            available = np.flatnonzero(~gone)
            if len(available) == 0:
                break
            beta = betas.get(slot_managers.get(slot))
            if beta is None:
                beta = np.zeros(len(FEATURE_NAMES))
            X = _live_features(pool, available, offset + 1, rosters[slot],
                               recent, settings)
            scores = X @ beta
            for j, idx in enumerate(available):
                pos = pool.position[idx]
                if rosters[slot].get(pos, 0) >= caps.get(pos, 99):
                    scores[j] = -np.inf
            finite = np.isfinite(scores)
            if not finite.any():
                choice = int(available[0])
            else:
                shifted = scores - scores[finite].max()
                weights = np.where(finite, np.exp(shifted), 0.0)
                total = weights.sum()
                choice = int(available[0]) if total <= 0 else \
                    int(rng.choice(available, p=weights / total))
            gone[choice] = True
            pos = pool.position[choice]
            rosters[slot][pos] = rosters[slot].get(pos, 0) + 1
            recent.insert(0, pos)
        counts += ~gone

    return pd.DataFrame({"player_id": pool.player_id,
                         "avail_pct": counts / max(n_rollouts, 1)})


def run_sim(conn, my_slot: int, slot_managers: dict,
            n_rollouts: int = DEFAULT_ROLLOUTS, seed: int = 0) -> str:
    from scoring import league as league_mod
    from scoring.board import build_board
    from scoring.draft_model import fit_all

    settings = league_mod.load(conn)
    board = build_board(conn, settings=settings)
    pool = build_pool(conn, board, settings)

    fits = fit_all(conn, settings)
    profiles = read_table(conn, "manager_profiles")
    pooled = fits.get("__pooled__", np.zeros(len(FEATURE_NAMES)))
    betas = {}
    for manager, beta in fits.items():
        if manager == "__pooled__":
            continue
        rows = profiles[profiles["manager"] == manager]
        personal = bool(rows["uses_personal"].iloc[0]) if not rows.empty else False
        betas[manager] = beta if personal else pooled

    drafted = read_table(conn, "drafted")
    drafted_ids = set(drafted["player_id"]) if not drafted.empty else set()
    taken = np.isin(pool.player_id, list(drafted_ids))

    results = search_pick(pool, settings, slot_managers, my_slot, taken, betas,
                          n_rollouts=n_rollouts, seed=seed)
    avail = survival(pool, settings, slot_managers, my_slot, taken, betas,
                     n_rollouts=n_rollouts, seed=seed)

    run_id = f"{my_slot}-{n_rollouts}-{seed}-{int(taken.sum())}"
    results.insert(0, "run_id", run_id)
    avail.insert(0, "run_id", run_id)
    results["my_slot"] = my_slot
    write_table(conn, "sim_results", results)
    write_table(conn, "sim_survival", avail)
    return run_id
```

Create `pipeline/run_sim.py`:

```python
"""Run the draft simulator. Run: python -m pipeline.run_sim <my_slot> [rollouts]"""
import sys

from pipeline.db import get_conn, read_table
from scoring.draft_sim import DEFAULT_ROLLOUTS, run_sim


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m pipeline.run_sim <my_slot> [rollouts]")
        return 2
    my_slot = int(argv[1])
    rollouts = int(argv[2]) if len(argv) > 2 else DEFAULT_ROLLOUTS
    conn = get_conn()

    order = read_table(conn, "draft_order")
    if order.empty:
        teams = read_table(conn, "draft_teams")
        if teams.empty:
            print("No draft order and no imported teams -- "
                  "run `make espn-import` first.")
            return 1
        newest = teams[teams["season"] == teams["season"].max()]
        slot_managers = dict(zip(newest["slot"], newest["manager"]))
    else:
        slot_managers = dict(zip(order["slot"], order["manager"]))

    run_id = run_sim(conn, my_slot, slot_managers, n_rollouts=rollouts)
    results = read_table(conn, "sim_results")
    print(f"Run {run_id} — top candidates at slot {my_slot}:")
    for _, row in results.sort_values("rank").head(8).iterrows():
        print(f"  {row['rank']:>2}. {row['player_id']:<20} "
              f"{row['ev']:8.1f} ± {row['se']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

Add to `Makefile` (and to `.PHONY`):

```make
sim: ## run the draft simulator: make sim SLOT=4 [ROLLOUTS=300]
	.venv/bin/python -m pipeline.run_sim "$(SLOT)" $(ROLLOUTS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_draft_sim.py -v`
Expected: 14 passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add scoring/draft_sim.py pipeline/run_sim.py Makefile tests/test_draft_sim.py
git commit -m "feat: candidate search, survival probabilities, and sim persistence"
```

---

## Phase 4 — API and board

### Task 12: Draft-order table and API endpoints

**Files:**
- Modify: `api/main.py`, `pipeline/db.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `scoring.league.load`, `manager_profiles`, `draft_teams`
- Produces:
  - `GET /api/league` → `{"season", "teams", "starters", "flex_slots", "bench", "rounds", "derived": bool, "unmapped_scoring": [...]}`
  - `GET /api/managers` → `{"managers": [{"manager", "summary", "n_picks", "uses_personal", "heldout_gain", "coefficients": [{"feature", "value", "pooled_value"}]}]}`
  - `GET /api/draft-order` → `{"order": [{"slot", "manager"}], "my_slot": int | null, "source": "espn" | "manual" | "none"}`
  - `PUT /api/draft-order` body `{"order": [{"slot", "manager"}], "my_slot": int}` → persisted to a `draft_order` table
  - `pipeline.db.get_conn` also creates `draft_order (slot INTEGER PRIMARY KEY, manager VARCHAR, is_me BOOLEAN)`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api.py`:

```python
def test_league_endpoint_reports_fallback_when_not_imported(tmp_path):
    client = _client(tmp_path)
    body = client.get("/api/league").json()
    assert body["derived"] is False
    assert body["teams"] == 8
    assert body["rounds"] == 15

def test_league_endpoint_reports_derived_settings(tmp_path):
    from scoring import league
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    settings = league.LeagueSettings(
        season=2026, teams=10,
        starters={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1},
        flex_slots=1, bench=6, scoring={"receptions": 0.5},
        draft_type="SNAKE", unmapped_scoring=("101",))
    write_table(conn, "league", pd.DataFrame(
        [{"season": 2026, "settings_json": league.to_json(settings)}]))
    conn.close()
    body = TestClient(create_app(path)).get("/api/league").json()
    assert body["derived"] is True
    assert body["teams"] == 10
    assert body["unmapped_scoring"] == ["101"]

def test_managers_endpoint_groups_coefficients(tmp_path):
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "manager_profiles", pd.DataFrame([
        {"manager": "worthy", "feature": "reach", "value": -1.2,
         "pooled_value": -0.4, "n_picks": 105, "heldout_gain": 0.08,
         "uses_personal": True, "summary": "sticks to market order"},
        {"manager": "worthy", "feature": "run", "value": 0.6,
         "pooled_value": 0.1, "n_picks": 105, "heldout_gain": 0.08,
         "uses_personal": True, "summary": "sticks to market order"},
    ]))
    conn.close()
    body = TestClient(create_app(path)).get("/api/managers").json()
    assert len(body["managers"]) == 1
    entry = body["managers"][0]
    assert entry["manager"] == "worthy"
    assert entry["n_picks"] == 105
    assert entry["uses_personal"] is True
    assert {c["feature"] for c in entry["coefficients"]} == {"reach", "run"}

def test_draft_order_round_trip(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/draft-order").json()["source"] == "none"
    payload = {"order": [{"slot": 1, "manager": "worthy"},
                         {"slot": 2, "manager": "dan"}], "my_slot": 2}
    assert client.put("/api/draft-order", json=payload).status_code == 200
    body = client.get("/api/draft-order").json()
    assert body["source"] == "manual"
    assert body["my_slot"] == 2
    assert body["order"] == [{"slot": 1, "manager": "worthy"},
                             {"slot": 2, "manager": "dan"}]

def test_draft_order_seeds_from_imported_teams(tmp_path):
    path = str(tmp_path / "t.duckdb")
    _seed(path)
    conn = get_conn(path)
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "old", "slot": 1},
        {"season": 2026, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2026, "team_id": 2, "manager": "dan", "slot": 2},
    ]))
    conn.close()
    body = TestClient(create_app(path)).get("/api/draft-order").json()
    assert body["source"] == "espn"
    assert [e["manager"] for e in body["order"]] == ["worthy", "dan"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api.py -v`
Expected: FAIL with 404s on the new routes.

- [ ] **Step 3: Write minimal implementation**

Add to `pipeline/db.py` inside `get_conn`:

```python
    conn.execute("""CREATE TABLE IF NOT EXISTS draft_order (
        slot INTEGER PRIMARY KEY, manager VARCHAR, is_me BOOLEAN)""")
```

Add to `api/main.py` inside `create_app`:

```python
    @app.get("/api/league")
    def league_info():
        cur = conn.cursor()
        try:
            derived = not read_table(cur, "league").empty
            s = league.load(cur)
            return {"season": s.season, "teams": s.teams,
                    "starters": s.starters, "flex_slots": s.flex_slots,
                    "bench": s.bench, "rounds": s.rounds, "derived": derived,
                    "unmapped_scoring": list(s.unmapped_scoring)}
        finally:
            cur.close()

    @app.get("/api/managers")
    def managers():
        cur = conn.cursor()
        try:
            profiles = read_table(cur, "manager_profiles")
            if profiles.empty:
                return {"managers": []}
            out = []
            for manager, grp in profiles.groupby("manager"):
                head = grp.iloc[0]
                gain = head["heldout_gain"]
                out.append({
                    "manager": manager,
                    "summary": head["summary"],
                    "n_picks": int(head["n_picks"]),
                    "uses_personal": bool(head["uses_personal"]),
                    "heldout_gain": None if pd.isna(gain) else float(gain),
                    "coefficients": [
                        {"feature": r["feature"], "value": float(r["value"]),
                         "pooled_value": float(r["pooled_value"])}
                        for _, r in grp.iterrows()],
                })
            return {"managers": out}
        finally:
            cur.close()

    @app.get("/api/draft-order")
    def draft_order():
        cur = conn.cursor()
        try:
            saved = read_table(cur, "draft_order")
            if not saved.empty:
                saved = saved.sort_values("slot")
                me = saved[saved["is_me"]]
                return {"order": [{"slot": int(r["slot"]), "manager": r["manager"]}
                                  for _, r in saved.iterrows()],
                        "my_slot": int(me.iloc[0]["slot"]) if not me.empty else None,
                        "source": "manual"}
            teams = read_table(cur, "draft_teams")
            if teams.empty:
                return {"order": [], "my_slot": None, "source": "none"}
            newest = teams[teams["season"] == teams["season"].max()]
            newest = newest.dropna(subset=["slot"]).sort_values("slot")
            return {"order": [{"slot": int(r["slot"]), "manager": r["manager"]}
                              for _, r in newest.iterrows()],
                    "my_slot": None, "source": "espn"}
        finally:
            cur.close()

    @app.put("/api/draft-order")
    def set_draft_order(payload: dict = Body(...)):
        entries = payload.get("order") or []
        if not entries:
            raise HTTPException(status_code=422, detail="order must not be empty")
        my_slot = payload.get("my_slot")
        rows = pd.DataFrame([{"slot": int(e["slot"]), "manager": e["manager"],
                              "is_me": int(e["slot"]) == my_slot}
                             for e in entries])
        cur = conn.cursor()
        try:
            write_table(cur, "draft_order", rows)
            return {"saved": len(rows)}
        finally:
            cur.close()
```

Add imports at the top of `api/main.py`:

```python
import pandas as pd
from fastapi import Body
from pipeline.db import write_table
from scoring import league
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_api.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add api/main.py pipeline/db.py tests/test_api.py
git commit -m "feat: league, manager, and draft-order endpoints"
```

---

### Task 13: Sim endpoints and board columns

**Files:**
- Modify: `api/main.py`, `pipeline/db.py`, `scoring/board.py`
- Test: `tests/test_api.py`, `tests/test_board.py`

**Interfaces:**
- Consumes: `scoring.draft_sim.run_sim`
- Produces:
  - `POST /api/sim` body `{"my_slot": int, "rollouts": int}` → `{"run_id": str, "status": "running"}`; runs in a `threading.Thread`
  - `GET /api/sim/{run_id}` → `{"status": "running" | "done" | "error", "detail": str | null}`
  - `/api/players` rows gain `avail_pct`, `ev`, `ev_se` (all `None` when no sim has run)
  - `pipeline.db.get_conn` recreates `drafted` as `(player_id VARCHAR PRIMARY KEY, pick_no INTEGER)`
  - `scoring.board.build_board` merges `sim_survival` and `sim_results` when present

- [ ] **Step 1: Write the failing test**

Append to `tests/test_board.py`:

```python
def test_board_sim_columns_are_null_without_a_sim(tmp_path):
    board = build_board(_seed(tmp_path))
    for col in ("avail_pct", "ev", "ev_se"):
        assert col in board.columns
        assert board[col].isna().all()

def test_board_merges_sim_results_when_present(tmp_path):
    from pipeline.db import write_table
    conn = _seed(tmp_path)
    write_table(conn, "sim_survival", pd.DataFrame(
        [{"run_id": "r1", "player_id": "p1", "avail_pct": 0.42}]))
    write_table(conn, "sim_results", pd.DataFrame(
        [{"run_id": "r1", "player_id": "p1", "ev": 1580.5, "se": 4.2,
          "rank": 1, "my_slot": 4}]))
    board = build_board(conn)
    row = board[board["player_id"] == "p1"].iloc[0]
    assert row["avail_pct"] == 0.42
    assert row["ev"] == 1580.5
    assert row["ev_se"] == 4.2
```

Append to `tests/test_api.py`:

```python
def test_sim_endpoint_starts_and_completes(tmp_path):
    import time
    client = _client(tmp_path)
    client.put("/api/draft-order", json={
        "order": [{"slot": s, "manager": f"m{s}"} for s in range(1, 9)],
        "my_slot": 1})
    started = client.post("/api/sim", json={"my_slot": 1, "rollouts": 3})
    assert started.status_code == 200
    run_id = started.json()["run_id"]
    for _ in range(200):
        status = client.get(f"/api/sim/{run_id}").json()
        if status["status"] != "running":
            break
        time.sleep(0.05)
    assert status["status"] == "done", status.get("detail")

def test_sim_status_for_unknown_run_is_404(tmp_path):
    assert _client(tmp_path).get("/api/sim/nope").status_code == 404

def test_players_expose_sim_columns_as_null_without_a_sim(tmp_path):
    row = _client(tmp_path).get("/api/players").json()["players"][0]
    assert row["avail_pct"] is None
    assert row["ev"] is None
    assert row["ev_se"] is None

def test_drafted_records_pick_order(tmp_path):
    client = _client(tmp_path)
    assert client.post("/api/drafted/p1").json()["pick_no"] == 1
    assert client.post("/api/drafted/p2").json()["pick_no"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_api.py tests/test_board.py -v`
Expected: FAIL — `KeyError: 'avail_pct'` and 404 on `/api/sim`.

- [ ] **Step 3: Write minimal implementation**

`pipeline/db.py` — add `pick_no` to `drafted`:

```python
    conn.execute("""CREATE TABLE IF NOT EXISTS drafted (
        player_id VARCHAR PRIMARY KEY, pick_no INTEGER)""")
    # Databases created before pick_no existed are missing the column; adding
    # it here keeps an existing data/nfl.duckdb usable without a manual drop.
    columns = {r[1] for r in conn.execute("PRAGMA table_info('drafted')").fetchall()}
    if "pick_no" not in columns:
        conn.execute("ALTER TABLE drafted ADD COLUMN pick_no INTEGER")
```

`scoring/board.py` — extend `_BOARD_COLUMNS` with `"avail_pct", "ev", "ev_se"` and merge before returning:

```python
    sim_avail = read_table(conn, "sim_survival")
    sim_ev = read_table(conn, "sim_results")
    if sim_avail.empty:
        uni["avail_pct"] = pd.NA
    else:
        uni = uni.merge(sim_avail[["player_id", "avail_pct"]],
                        on="player_id", how="left")
    if sim_ev.empty:
        uni["ev"] = pd.NA
        uni["ev_se"] = pd.NA
    else:
        ev = sim_ev[["player_id", "ev", "se"]].rename(columns={"se": "ev_se"})
        uni = uni.merge(ev, on="player_id", how="left")
```

`api/main.py` — sim endpoints and drafted pick numbering:

```python
import threading
import uuid

    _sim_runs: dict[str, dict] = {}

    @app.post("/api/sim")
    def start_sim(payload: dict = Body(...)):
        my_slot = int(payload.get("my_slot") or 0)
        if my_slot < 1:
            raise HTTPException(status_code=422, detail="my_slot is required")
        rollouts = int(payload.get("rollouts") or DEFAULT_ROLLOUTS)
        order = draft_order()
        slot_managers = {e["slot"]: e["manager"] for e in order["order"]}
        run_id = uuid.uuid4().hex[:12]
        _sim_runs[run_id] = {"status": "running", "detail": None}

        def worker():
            cur = conn.cursor()
            try:
                run_sim(cur, my_slot, slot_managers, n_rollouts=rollouts)
                _sim_runs[run_id] = {"status": "done", "detail": None}
            except Exception as e:
                _sim_runs[run_id] = {"status": "error", "detail": str(e)}
            finally:
                cur.close()

        threading.Thread(target=worker, daemon=True).start()
        return {"run_id": run_id, "status": "running"}

    @app.get("/api/sim/{run_id}")
    def sim_status(run_id: str):
        state = _sim_runs.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown run_id")
        return state
```

Change the drafted endpoint to record pick order:

```python
    @app.post("/api/drafted/{player_id}")
    def draft(player_id: str):
        cur = conn.cursor()
        try:
            next_pick = cur.execute(
                "SELECT coalesce(max(pick_no), 0) + 1 FROM drafted").fetchone()[0]
            cur.execute("INSERT OR IGNORE INTO drafted VALUES (?, ?)",
                        [player_id, next_pick])
            return {"drafted": True, "pick_no": next_pick}
        finally:
            cur.close()
```

Add imports to `api/main.py`:

```python
from scoring.draft_sim import DEFAULT_ROLLOUTS, run_sim
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add api/main.py pipeline/db.py scoring/board.py tests/
git commit -m "feat: sim endpoints, board sim columns, drafted pick order"
```

---

### Task 14: Frontend rail and board columns

**Files:**
- Create: `web/src/components/DraftRail.tsx`
- Modify: `web/src/api.ts`, `web/src/components/PlayerTable.tsx`, `web/src/App.tsx`, `web/src/App.css`

**Interfaces:**
- Consumes: Task 12 and 13 endpoints
- Produces:
  - `web/src/api.ts`: `Player` gains `avail_pct: number | null`, `ev: number | null`, `ev_se: number | null`; new `fetchManagers()`, `fetchDraftOrder()`, `saveDraftOrder(order, mySlot)`, `startSim(mySlot, rollouts)`, `pollSim(runId)`, and types `Manager`, `DraftOrderEntry`
  - `DraftRail` component props `{ onSimComplete: () => void }`

- [ ] **Step 1: Add the API client functions**

Append to `web/src/api.ts`:

```ts
export interface ManagerCoefficient {
  feature: string
  value: number
  pooled_value: number
}

export interface Manager {
  manager: string
  summary: string
  n_picks: number
  uses_personal: boolean
  heldout_gain: number | null
  coefficients: ManagerCoefficient[]
}

export interface DraftOrderEntry {
  slot: number
  manager: string
}

export interface DraftOrder {
  order: DraftOrderEntry[]
  my_slot: number | null
  source: 'espn' | 'manual' | 'none'
}

export async function fetchManagers(): Promise<Manager[]> {
  const res = await fetch('/api/managers')
  if (!res.ok) throw new Error('Failed to load managers')
  return (await res.json()).managers
}

export async function fetchDraftOrder(): Promise<DraftOrder> {
  const res = await fetch('/api/draft-order')
  if (!res.ok) throw new Error('Failed to load draft order')
  return res.json()
}

export async function saveDraftOrder(order: DraftOrderEntry[], mySlot: number) {
  const res = await fetch('/api/draft-order', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ order, my_slot: mySlot }),
  })
  if (!res.ok) throw new Error('Failed to save draft order')
}

export async function startSim(mySlot: number, rollouts: number): Promise<string> {
  const res = await fetch('/api/sim', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ my_slot: mySlot, rollouts }),
  })
  if (!res.ok) throw new Error('Failed to start simulation')
  return (await res.json()).run_id
}

export async function pollSim(runId: string): Promise<{ status: string; detail: string | null }> {
  const res = await fetch(`/api/sim/${runId}`)
  if (!res.ok) throw new Error('Failed to read simulation status')
  return res.json()
}
```

Add to the `Player` interface in `web/src/api.ts`:

```ts
  avail_pct: number | null
  ev: number | null
  ev_se: number | null
```

- [ ] **Step 2: Write the rail component**

Create `web/src/components/DraftRail.tsx`:

```tsx
import { useEffect, useState } from 'react'
import {
  fetchDraftOrder, fetchManagers, pollSim, saveDraftOrder, startSim,
  type DraftOrderEntry, type Manager,
} from '../api'

// Coefficients worth showing on a card. The rest are position dummies that
// only make sense relative to each other, which a three-bar summary cannot
// convey honestly.
const SHOWN_FEATURES = ['reach', 'fall', 'need', 'run', 'qb_early', 'te_early']

interface DraftRailProps {
  onSimComplete: () => void
}

export default function DraftRail({ onSimComplete }: DraftRailProps) {
  const [managers, setManagers] = useState<Manager[]>([])
  const [order, setOrder] = useState<DraftOrderEntry[]>([])
  const [mySlot, setMySlot] = useState<number | null>(null)
  const [rollouts, setRollouts] = useState(300)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([fetchManagers(), fetchDraftOrder()])
      .then(([m, o]) => {
        setManagers(m)
        setOrder(o.order)
        setMySlot(o.my_slot)
      })
      .catch((e) => setError(e instanceof Error ? e.message : 'Load failed'))
  }, [])

  async function handleRun() {
    if (mySlot === null) return
    setRunning(true)
    setError(null)
    try {
      await saveDraftOrder(order, mySlot)
      const runId = await startSim(mySlot, rollouts)
      // The sim runs in a background thread server-side; poll rather than
      // hold a request open for the length of the run.
      for (;;) {
        const status = await pollSim(runId)
        if (status.status === 'done') break
        if (status.status === 'error') throw new Error(status.detail ?? 'Simulation failed')
        await new Promise((r) => setTimeout(r, 500))
      }
      onSimComplete()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Simulation failed')
    } finally {
      setRunning(false)
    }
  }

  const names = managers.map((m) => m.manager)
  const byName = new Map(managers.map((m) => [m.manager, m]))

  return (
    <aside className="draft-rail">
      <section className="rail-section">
        <h2>Draft order</h2>
        {order.length === 0 && <p className="rail-empty">Run `make espn-import` to load your league.</p>}
        <ol className="order-list">
          {order.map((entry, i) => (
            <li key={entry.slot}>
              <span className="slot-no">{entry.slot}</span>
              <select
                value={entry.manager}
                onChange={(e) => {
                  const next = [...order]
                  next[i] = { ...entry, manager: e.target.value }
                  setOrder(next)
                }}
              >
                {names.map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
              <label className="is-me">
                <input
                  type="radio"
                  name="my-slot"
                  checked={mySlot === entry.slot}
                  onChange={() => setMySlot(entry.slot)}
                />
                me
              </label>
            </li>
          ))}
        </ol>
        <div className="rail-run">
          <label>
            Rollouts{' '}
            <input
              type="number"
              min={10}
              step={50}
              value={rollouts}
              onChange={(e) => setRollouts(Number(e.target.value))}
            />
          </label>
          <button onClick={handleRun} disabled={running || mySlot === null}>
            {running ? 'Simulating…' : 'Run simulation'}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="rail-section">
        <h2>Managers</h2>
        {order.map((entry) => {
          const m = byName.get(entry.manager)
          if (!m) return null
          const shown = m.coefficients
            .filter((c) => SHOWN_FEATURES.includes(c.feature))
            .sort((a, b) => Math.abs(b.value - b.pooled_value) - Math.abs(a.value - a.pooled_value))
            .slice(0, 3)
          return (
            <article key={entry.slot} className="manager-card">
              <header>
                <span className="slot-no">{entry.slot}</span>
                <strong>{m.manager}</strong>
                <span className={m.uses_personal ? 'chip chip-personal' : 'chip chip-pooled'}>
                  {m.uses_personal ? 'personal model' : 'league average'}
                </span>
              </header>
              <p className="manager-summary">{m.summary}</p>
              <ul className="coef-bars">
                {shown.map((c) => {
                  const delta = c.value - c.pooled_value
                  const width = Math.min(100, Math.abs(delta) * 40)
                  return (
                    <li key={c.feature}>
                      <span className="coef-name">{c.feature}</span>
                      <span className="coef-track">
                        <span
                          className={delta >= 0 ? 'coef-fill pos' : 'coef-fill neg'}
                          style={{ width: `${width}%` }}
                        />
                      </span>
                    </li>
                  )
                })}
              </ul>
              <footer>{m.n_picks} picks</footer>
            </article>
          )
        })}
      </section>
    </aside>
  )
}
```

- [ ] **Step 3: Add the board columns**

In `web/src/components/PlayerTable.tsx`, add two columns to the column definitions, placed immediately after the existing `edge` column:

```tsx
  {
    id: 'avail_pct',
    label: 'Avail%',
    title: 'Probability this player is still available at your next pick',
    value: (p: Player) => p.avail_pct,
    render: (p: Player) =>
      p.avail_pct === null ? '' : `${Math.round(p.avail_pct * 100)}%`,
  },
  {
    id: 'ev',
    label: 'ΔEV',
    title: 'Expected starting-lineup points versus the best available option',
    value: (p: Player) => p.ev,
    // Rendered relative to the best EV on the board, so the top candidate
    // reads as a dash and everything else reads as what it costs you.
    render: (p: Player, ctx: { bestEv: number | null }) => {
      if (p.ev === null || ctx.bestEv === null) return ''
      const delta = p.ev - ctx.bestEv
      return delta === 0 ? '—' : delta.toFixed(1)
    },
  },
```

Compute `bestEv` once per render inside `PlayerTable` and pass it into `render`:

```tsx
  const bestEv = useMemo(() => {
    const values = players.map((p) => p.ev).filter((v): v is number => v !== null)
    return values.length ? Math.max(...values) : null
  }, [players])
```

- [ ] **Step 4: Mount the rail**

In `web/src/App.tsx`, render the rail inside `app-body` alongside `main`, and reload the board when a sim finishes:

```tsx
        <main className="main">
          {/* ...unchanged... */}
        </main>
        <DraftRail onSimComplete={loadPlayers} />
```

Add `import DraftRail from './components/DraftRail'` to the imports.

- [ ] **Step 5: Add rail styles**

First change `.app-body` in `web/src/App.css` from a column to a row — today it is `flex-direction: column` with `.main` as its only child, so a sibling rail would stack below the board instead of beside it:

```css
.app-body {
  display: flex;
  flex-direction: row;
  flex: 1;
  min-height: 0;
}
```

`.main` already carries `flex: 1; min-width: 0`, which is what it needs in a row container. Leave it alone.

Then append the rail styles. Use only tokens that already exist in `web/src/index.css`: `--bg-0`, `--bg-1`, `--bg-2`, `--border`, `--text-1`, `--text-2`, `--text-3`, `--accent`, `--accent-bg`, `--accent-border`, `--ok`, `--fail`, and the `--space-*` scale. Do not invent new tokens.

```css
.draft-rail {
  width: 280px;
  flex: 0 0 280px;
  overflow-y: auto;
  background: var(--bg-1);
  border-left: 1px solid var(--border);
  padding: 12px;
}

.rail-section + .rail-section { margin-top: 20px; }
.rail-section h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }
.order-list { list-style: none; margin: 0; padding: 0; }
.order-list li { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.slot-no { width: 18px; text-align: right; opacity: 0.6; font-variant-numeric: tabular-nums; }
.rail-run { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
.rail-run input { width: 70px; }
.manager-card { border: 1px solid var(--border); border-radius: 6px; padding: 8px; margin-bottom: 8px; }
.manager-card header { display: flex; align-items: center; gap: 6px; }
.manager-summary { font-size: 0.85rem; opacity: 0.85; margin: 6px 0; }
.chip { font-size: 0.7rem; padding: 1px 6px; border-radius: 10px; }
.chip-personal { background: var(--accent-bg); color: var(--accent); }
.chip-pooled { background: var(--bg-2); color: var(--text-3); }
.coef-bars { list-style: none; margin: 0; padding: 0; }
.coef-bars li { display: flex; align-items: center; gap: 6px; font-size: 0.75rem; }
.coef-name { width: 70px; color: var(--text-3); }
.coef-track { flex: 1; height: 6px; background: var(--bg-2); border-radius: 3px; }
.coef-fill { display: block; height: 100%; border-radius: 3px; }
.coef-fill.pos { background: var(--ok); }
.coef-fill.neg { background: var(--fail); }
.manager-card footer { font-size: 0.7rem; color: var(--text-3); margin-top: 4px; }
.rail-empty { font-size: 0.8rem; color: var(--text-3); }
.sim-status { font-size: 0.8rem; color: var(--text-2); }
```

- [ ] **Step 6: Add the header status strip**

`TopBar` already accepts a `meta` slot (today it holds `<FreshnessBadge />`). Add the sim status beside it in `web/src/App.tsx` so the header reads `Slot 4 · next pick 2.13 · sim 2h old`:

```tsx
  const [simStatus, setSimStatus] = useState<string | null>(null)

  // Round-and-pick notation for the next turn, from the snake order: pick
  // `n` in an 8-team league is round floor((n-1)/8)+1, pick ((n-1)%8)+1.
  function nextPickLabel(slot: number, drafted: number, teams: number, rounds: number) {
    for (let offset = drafted; offset < teams * rounds; offset++) {
      const round = Math.floor(offset / teams)
      const onClock = round % 2 === 0 ? (offset % teams) + 1 : teams - (offset % teams)
      if (onClock === slot) return `${round + 1}.${(offset % teams) + 1}`
    }
    return '—'
  }
```

Render it inside the `meta` prop:

```tsx
        meta={
          <>
            <FreshnessBadge />
            {simStatus && <span className="sim-status">{simStatus}</span>}
          </>
        }
```

`DraftRail` sets it: extend its props to `{ onSimComplete: () => void; onStatus: (text: string) => void }` and call `onStatus(...)` with `Slot ${mySlot} · next pick ${label} · sim just now` once a run finishes, and `Slot ${mySlot} · simulating…` while one is in flight. When no sim has ever run, leave `simStatus` null so the header is unchanged.

- [ ] **Step 7: Typecheck and build**

Run: `cd web && npm run build`
Expected: build succeeds with no TypeScript errors.

- [ ] **Step 8: Commit**

```bash
git add web/src
git commit -m "feat: draft rail with manager profiles and board sim columns"
```

---

### Task 15: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the new workflow**

Add a "Draft simulation" section to `README.md` after "How scoring works", covering:

- One-time setup: `.venv/bin/playwright install chromium`
- `make espn-import LEAGUE=<url-or-id>` — what it pulls, that the first run opens a browser window for login, and that the saved state lives at `data/espn_state.json` and is gitignored
- What the validation summary reports and what a low ADP match rate means
- `make fit-managers` — what the backtest line means, and that a "does not beat the ADP baseline" warning means the simulator output is indicative only
- `make sim SLOT=4 ROLLOUTS=300` — or the Run button in the rail
- The two new board columns, Avail% and ΔEV, and what blank means for each
- That league size, roster shape, scoring rules, and replacement ranks now come from ESPN when a league has been imported, and fall back to `scoring/config.py` otherwise

Update the "Project structure" list with `pipeline/espn_league.py`, `pipeline/import_league.py`, `pipeline/fit_managers.py`, `pipeline/run_sim.py`, `scoring/league.py`, `scoring/draft_model.py`, `scoring/draft_sim.py`.

- [ ] **Step 2: Verify the whole suite and build**

Run: `.venv/bin/pytest -q && cd web && npm run build`
Expected: all tests pass, build succeeds.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: draft history import and simulation workflow"
```

---

## Verification

After Task 15, confirm end to end against the real league:

1. `make espn-import LEAGUE=<your league url>` — browser opens, log in, import completes. Check the summary: seasons found, picks per season matching `teams x rounds`, ADP match rate above 80% per season.
2. `make fit-managers` — read the backtest line. Note whether it beats the ADP baseline.
3. `make api` and `make web` — open the board, confirm the rail lists your managers with sensible one-line reads, set your slot, click Run.
4. After the run, confirm Avail% and ΔEV populate and that Avail% is near 0 for the top few players and near 1 for deep bench players.
