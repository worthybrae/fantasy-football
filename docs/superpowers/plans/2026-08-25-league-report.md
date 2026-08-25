# League Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A public, phone-friendly page per league-season with a draft report card (grade + Haiku-written blurb per team), power rankings and team histories, generated automatically when a live draft ends and on demand by the owner.

**Architecture:** ESPN standings are parsed from the payload the importer already fetches into a new `league_standings` table, and history is imported into each league's own DuckDB file with the connecting account's cookies. `scoring/league_report.py` computes every number (pick verdicts, grades, profiles, power rankings) as pure functions; `scoring/blurbs.py` makes exactly one `claude-haiku-4-5` structured-output call to write prose against those numbers; `api/reports.py` runs the build on a daemon thread, stores the payload in `league_reports`, and serves it. The React page reads the stored copy and is routed outside `MobileGate`.

**Tech Stack:** Python 3 / FastAPI / DuckDB / pandas; `anthropic` SDK 1.x; React 19 + TypeScript + react-router 7 (Vite); pytest; Playwright (in `.venv`) for the browser check.

**Spec:** `docs/superpowers/specs/2026-08-25-league-report-design.md`

## Global Constraints

- Model is `claude-haiku-4-5`, one call per report, `max_tokens` 4096, structured output via `output_config={"format": {"type": "json_schema", ...}}`. No `thinking` parameter.
- `ANTHROPIC_API_KEY` unset means no client and a `numbers_only` report; nothing raises and nothing on the page looks broken.
- New per-league tables `league_standings` and `league_reports` must be added to `LEAGUE_TABLES` in `pipeline/db.py:17` and to the exact-set assertion in `tests/test_leagues.py:19`.
- Report routes register in `api/main.py` after `register_seo_routes` and before `register_spa`.
- The report page is routed as a sibling of `<MobileGate>` in `web/src/App.tsx`, mobile-first, `lr-` class prefix, own sheet `web/src/league.css`.
- `pick value = overall_pick - market_rank`; NULL when `market_rank > last pick of that draft`. Verdict bands exactly as `web/src/components/draft/PickTicker.tsx:43-60`.
- Grade is rank-based: fractions 1/8, 3/8, 5/8, 7/8 of team count. Power score `0.6 * z(value_per_pick) + 0.4 * z(win_pct)`; a first-year manager's history z is 0.
- Generation is gated: POST requires custody session + league in the account's list + `require_paid`; auto-trigger requires `entitled()` unless billing is off. Mock leagues never generate.
- `requirements.txt` dependency additions need a justification paragraph (see `requirements.txt:18-30` for the two precedents).
- Run tests with `.venv/bin/pytest -q`. A running `make up` locks the test DB and causes ~47 unrelated `test_live_api` failures; stop it before a full run.
- Commit after every task. Persisted text (code, comments, commits) is normal prose.

---

## File map

| Path | Responsibility |
|---|---|
| `pipeline/espn_league.py` | + `parse_standings(payload, season)`; `import_seasons` writes `league_standings` and a `name` column on `league` |
| `pipeline/db.py` | + two table names in `LEAGUE_TABLES` |
| `pipeline/league_history.py` (new) | cookie-fetch adapter; `import_history`; freshness check; background spawn |
| `scoring/league_report.py` (new) | verdicts, pick frames (historical + live), grades, profiles, power rankings, `build_facts`, `merge_prose` |
| `scoring/blurbs.py` (new) | Haiku writer: schema, prompt, one call + one retry, validation, `default_client()` |
| `api/reports.py` (new) | `league_reports` storage, `build_report`, single-flight spawn, three routes, live hooks (`on_draft_complete`, `cookies_for_connect`) |
| `api/main.py` | register report routes |
| `api/live.py` | two hook calls: import-on-connect, build-on-complete |
| `requirements.txt` | + `anthropic` |
| `web/src/api.ts` | + `LeagueReport*` types and three fetchers |
| `web/src/pages/LeagueReport.tsx` (new) | the page |
| `web/src/league.css` (new) | its sheet |
| `web/src/App.tsx` | route outside the gate |
| `web/src/components/Dashboard.tsx` | report link / build button per league card |
| `tests/test_espn_league.py`, `tests/test_leagues.py`, `tests/test_league_history.py` (new), `tests/test_league_report.py` (new), `tests/test_blurbs.py` (new), `tests/test_reports_api.py` (new) | tests |

---

### Task 1: Standings out of the ESPN payload

**Files:**
- Modify: `pipeline/espn_league.py:109-122` (beside `parse_draft_teams`), `pipeline/espn_league.py:229-295` (`import_seasons`)
- Modify: `pipeline/db.py:17-21`
- Modify: `tests/test_leagues.py:19-22`
- Test: `tests/test_espn_league.py`

**Interfaces:**
- Produces: `parse_standings(payload: dict, season: int) -> pd.DataFrame` with columns `season, team_id, manager, team_name, wins, losses, ties, points_for, points_against, playoff_seed, final_rank`. `playoff_seed` and `final_rank` are `pd.NA` when ESPN reports 0 or omits them. `import_seasons` writes table `league_standings` with those columns and adds a `name` column (league name, `settings.name`, may be None) to the `league` table.

Verified against a real fetch of league 53929318 on 2026-08-25: `teams[].record.overall.{wins,losses,ties,pointsFor,pointsAgainst}`, `teams[].playoffSeed` (0 in progress), `teams[].rankCalculatedFinal` (1 = champion, 0 in progress), `teams[].name` (full team name), `settings.name`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_espn_league.py` (uses the module's existing `DRAFT_PAYLOAD`, `TEAM_PAYLOAD`, `ESPN_SETTINGS`, `_fake_fetch`, `get_conn`, `read_table` imports):

```python
STANDINGS_PAYLOAD = {
    "settings": {"name": "The Big League"},
    "teams": [
        {"id": 3, "name": "Team Alpha", "owners": ["{AAA}"], "draftDayPickOrder": 1,
         "playoffSeed": 1, "rankCalculatedFinal": 2,
         "record": {"overall": {"wins": 10, "losses": 4, "ties": 0,
                                "pointsFor": 1800.5, "pointsAgainst": 1600.25}}},
        # No owner, still in progress: manager falls back to the team name,
        # seed and final rank are 0 and must store as NULL.
        {"id": 7, "name": "Team Bravo", "owners": [], "draftDayPickOrder": 2,
         "playoffSeed": 0, "rankCalculatedFinal": 0,
         "record": {"overall": {"wins": 0, "losses": 0, "ties": 0,
                                "pointsFor": 0.0, "pointsAgainst": 0.0}}},
        # No record at all: every number null, row still present.
        {"id": 9, "name": "Team Charlie", "owners": ["{BBB}"]},
    ],
    "members": [
        {"id": "{AAA}", "displayName": "worthy"},
        {"id": "{BBB}", "displayName": "dan"},
    ],
}


def test_parse_standings_reads_record_seed_and_final_rank():
    from pipeline.espn_league import parse_standings
    df = parse_standings(STANDINGS_PAYLOAD, 2024)
    assert list(df.columns) == [
        "season", "team_id", "manager", "team_name", "wins", "losses", "ties",
        "points_for", "points_against", "playoff_seed", "final_rank"]
    alpha = df[df["team_id"] == 3].iloc[0]
    assert alpha["manager"] == "worthy" and alpha["team_name"] == "Team Alpha"
    assert (alpha["wins"], alpha["losses"], alpha["ties"]) == (10, 4, 0)
    assert alpha["points_for"] == 1800.5 and alpha["points_against"] == 1600.25
    assert alpha["playoff_seed"] == 1 and alpha["final_rank"] == 2
    bravo = df[df["team_id"] == 7].iloc[0]
    assert bravo["manager"] == "Team Bravo"
    assert pd.isna(bravo["playoff_seed"]) and pd.isna(bravo["final_rank"])
    charlie = df[df["team_id"] == 9].iloc[0]
    assert pd.isna(charlie["wins"]) and pd.isna(charlie["points_for"])


def test_import_seasons_writes_standings_and_league_name(tmp_path):
    def fetch(url):
        if "2025" not in url:
            raise FileNotFoundError(url)
        if "/players?" in url:
            return []
        return {**DRAFT_PAYLOAD, **STANDINGS_PAYLOAD, **ESPN_SETTINGS,
                "settings": {**ESPN_SETTINGS["settings"], "name": "The Big League"}}
    conn = get_conn(str(tmp_path / "t.duckdb"))
    import_seasons(conn, "99", current_season=2026, fetch=fetch)
    standings = read_table(conn, "league_standings")
    assert set(standings["team_id"]) == {3, 7, 9}
    assert standings["season"].unique().tolist() == [2025]
    league = read_table(conn, "league")
    assert league.iloc[0]["name"] == "The Big League"
```

Add `import pandas as pd` at the top of the test file if it is not already imported.

Update the exact-set assertion in `tests/test_leagues.py:19-22` to:

```python
    assert LEAGUE_TABLES == frozenset({
        "league", "draft_picks", "draft_teams", "draft_order",
        "manager_profiles", "manager_tendencies", "drafted",
        "sim_board", "sim_results", "sim_survival",
        "league_standings", "league_reports"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_espn_league.py -k standings tests/test_leagues.py::test_table_split_is_complete_and_disjoint -v`
Expected: 3 FAIL (`ImportError: cannot import name 'parse_standings'`, and the frozenset mismatch).

- [ ] **Step 3: Implement**

In `pipeline/db.py:17-21`:

```python
LEAGUE_TABLES = frozenset({
    "league", "draft_picks", "draft_teams", "draft_order",
    "manager_profiles", "manager_tendencies", "drafted",
    "sim_board", "sim_results", "sim_survival",
    # How each season ended (pipeline/espn_league.parse_standings) and the
    # stored league reports built on it (api/reports.py). Both are one
    # league's business and nobody else's.
    "league_standings", "league_reports",
})
```

In `pipeline/espn_league.py`, after `parse_draft_teams`:

```python
STANDINGS_COLUMNS = ["season", "team_id", "manager", "team_name", "wins", "losses",
                     "ties", "points_for", "points_against", "playoff_seed",
                     "final_rank"]


def _rank_or_none(value):
    # ESPN reports 0 for a seed or a final rank it has not decided yet (a
    # season in progress). Zero is not a rank; store the absence.
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def parse_standings(payload: dict, season: int) -> pd.DataFrame:
    """One row per team: how the season ended, from the `mTeam` view.

    `record.overall` is the regular season; `rankCalculatedFinal` is the
    playoff-resolved finish (1 is the champion). The manager resolves the
    same way `parse_draft_teams` resolves it so the two tables agree on the
    name a person is known by across seasons.
    """
    members = {m.get("id"): m.get("displayName")
               for m in (payload.get("members") or [])}
    rows = []
    for t in payload.get("teams") or []:
        owners = t.get("owners") or []
        manager = next((members[o] for o in owners if members.get(o)), None)
        overall = ((t.get("record") or {}).get("overall")) or {}
        rows.append({
            "season": season, "team_id": t.get("id"),
            "manager": manager or t.get("name"),
            "team_name": t.get("name"),
            "wins": overall.get("wins"), "losses": overall.get("losses"),
            "ties": overall.get("ties"),
            "points_for": overall.get("pointsFor"),
            "points_against": overall.get("pointsAgainst"),
            "playoff_seed": _rank_or_none(t.get("playoffSeed")),
            "final_rank": _rank_or_none(t.get("rankCalculatedFinal")),
        })
    df = pd.DataFrame(rows, columns=STANDINGS_COLUMNS)
    for col in ("wins", "losses", "ties", "playoff_seed", "final_rank"):
        df[col] = df[col].astype("Int64")
    for col in ("points_for", "points_against"):
        df[col] = df[col].astype("Float64")
    return df
```

In `import_seasons` (`pipeline/espn_league.py:229-295`): add `standings = []` to the first line's tuple, append `standings.append(parse_standings(payload, season))` right after `teams.append(parse_draft_teams(payload, season))`, change the `leagues.append` to carry the name:

```python
        leagues.append({"season": season,
                        "settings_json": league_mod.to_json(settings),
                        "name": (payload.get("settings") or {}).get("name")})
```

and after `write_table(conn, "draft_teams", ...)` add:

```python
    write_table(conn, "league_standings", pd.concat(standings, ignore_index=True))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_espn_league.py tests/test_leagues.py tests/test_league.py -v`
Expected: all PASS (the existing import test still passes; `TEAM_PAYLOAD` teams have no `record`, which yields null rows).

- [ ] **Step 5: Commit**

```bash
git add pipeline/db.py pipeline/espn_league.py tests/test_espn_league.py tests/test_leagues.py
git commit -m "The importer keeps how each season ended, and the league's name"
```

---

### Task 2: History import into a league's own file, with an account's cookies

**Files:**
- Create: `pipeline/league_history.py`
- Test: `tests/test_league_history.py`

**Interfaces:**
- Consumes: `import_seasons(conn, league_id, current_season, fetch)` (`pipeline/espn_league.py:229`); `provision_league(league_id, universal_path, root)` and `league_db_path` (`pipeline/leagues.py`); `normalize_historic_adp` (`pipeline/import_league.py:16`); `sources.fetch_adp(season)`; `KONA`, `http_fetch()` (`pipeline/espn_drafts.py`).
- Produces:
  - `json_fetch(fetch, cookies) -> Callable[[str], Any]` — adapter to the one-argument JSON fetch `import_seasons` wants; raises `FileNotFoundError` on 404, `PermissionError` on 401/403, `RuntimeError` on other non-2xx.
  - `is_fresh(conn, max_age_days=7) -> bool` — True when `meta.source == 'league'` was refreshed within the window.
  - `import_history(league_id, cookies, fetch=None, current_season=CURRENT_SEASON, universal_path=DEFAULT_PATH, root=LEAGUES_ROOT, adp_fetch=None) -> dict` — provisions the league file, imports seasons + historic ADP, records freshness, returns the `import_seasons` summary.
  - `spawn_import_if_stale(league_id, cookies, spawn=None, **kwargs) -> bool` — returns True if a thread was started. Never raises.

- [ ] **Step 1: Write the failing tests**

`tests/test_league_history.py`:

```python
import threading

import pandas as pd

from pipeline.db import get_conn, read_table
from tests.test_espn_league import DRAFT_PAYLOAD, TEAM_PAYLOAD
from tests.test_league import ESPN_SETTINGS

SEASON_PAYLOAD = {**DRAFT_PAYLOAD, **TEAM_PAYLOAD, **ESPN_SETTINGS}


def _raw_fetch(seasons, calls=None):
    """A `fetch(url, cookies, headers) -> (status, text)` like espn_drafts.http_fetch."""
    import json

    def fetch(url, cookies, headers=None):
        if calls is not None:
            calls.append((url, cookies, headers))
        year = next((s for s in seasons if str(s) in url), None)
        if year is None:
            return 404, "not found"
        if "/players?" in url:
            return 200, json.dumps([])
        return 200, json.dumps(SEASON_PAYLOAD)
    return fetch


def test_json_fetch_adapts_status_to_exceptions():
    from pipeline.league_history import json_fetch
    calls = []
    fetch = json_fetch(_raw_fetch([2025], calls), {"SWID": "{X}", "espn_s2": "s"})
    body = fetch("https://x/seasons/2025/segments/0/leagues/1")
    assert body["draftDetail"]["drafted"] is True
    assert calls[0][1] == {"SWID": "{X}", "espn_s2": "s"}
    assert calls[0][2] == {"x-fantasy-source": "kona"}
    import pytest
    with pytest.raises(FileNotFoundError):
        fetch("https://x/seasons/2019/segments/0/leagues/1")

    def forbidden(url, cookies, headers=None):
        return 403, "no"
    with pytest.raises(PermissionError):
        json_fetch(forbidden, {})("https://x/whatever")


def test_json_fetch_sends_the_player_filter_header():
    from pipeline.league_history import json_fetch
    from pipeline.espn_league import PLAYERS_FILTER
    calls = []
    json_fetch(_raw_fetch([2025], calls), {})("https://x/seasons/2025/players?view=players_wl")
    assert calls[0][2]["X-Fantasy-Filter"] == PLAYERS_FILTER


def test_import_history_writes_into_the_leagues_own_file(tmp_path):
    from pipeline.league_history import import_history, is_fresh
    from pipeline import leagues
    universal = str(tmp_path / "nfl.duckdb")
    get_conn(universal).close()
    root = str(tmp_path / "leagues")
    summary = import_history(
        "424242", {"SWID": "{X}", "espn_s2": "s"}, fetch=_raw_fetch([2025, 2024]),
        current_season=2026, universal_path=universal, root=root,
        adp_fetch=lambda season: pd.DataFrame([
            {"adp_name": "Justin Jefferson", "position": "WR", "team": "MIN", "adp": 1.5}]))
    assert summary["seasons"] == [2025, 2024]
    conn = get_conn(leagues.league_db_path("424242", root=root))
    assert len(read_table(conn, "draft_picks")) == 6
    assert set(read_table(conn, "historic_adp")["season"]) == {2025, 2024}
    assert read_table(conn, "historic_adp").iloc[0]["adp_rank"] == 1
    assert is_fresh(conn)
    conn.close()
    # The universal file was not written to.
    assert read_table(get_conn(universal), "draft_picks").empty


def test_import_history_survives_a_missing_adp_year(tmp_path):
    from pipeline.league_history import import_history
    universal = str(tmp_path / "nfl.duckdb")
    get_conn(universal).close()

    def adp_fetch(season):
        raise RuntimeError("feed down")
    summary = import_history(
        "1", {}, fetch=_raw_fetch([2025]), current_season=2026,
        universal_path=universal, root=str(tmp_path / "lg"), adp_fetch=adp_fetch)
    assert summary["seasons"] == [2025]


def test_is_fresh_is_false_without_an_import(tmp_path):
    from pipeline.league_history import is_fresh
    assert not is_fresh(get_conn(str(tmp_path / "t.duckdb")))


def test_spawn_import_if_stale_skips_fresh_and_never_raises(tmp_path, monkeypatch):
    from pipeline import league_history
    universal = str(tmp_path / "nfl.duckdb")
    get_conn(universal).close()
    root = str(tmp_path / "lg")
    started = []

    def spawn(name, fn):
        started.append(name)
        fn()
    ok = league_history.spawn_import_if_stale(
        "5", {}, spawn=spawn, fetch=_raw_fetch([2025]), current_season=2026,
        universal_path=universal, root=root, adp_fetch=lambda s: pd.DataFrame(
            columns=["adp_name", "position", "team", "adp"]))
    assert ok and started == ["history-5"]
    # Fresh now: nothing spawned.
    assert not league_history.spawn_import_if_stale(
        "5", {}, spawn=spawn, universal_path=universal, root=root)
    assert started == ["history-5"]
    # A broken fetch inside the thread is logged, not raised.
    def boom(url, cookies, headers=None):
        raise RuntimeError("boom")
    assert league_history.spawn_import_if_stale(
        "6", {}, spawn=spawn, fetch=boom, current_season=2026,
        universal_path=universal, root=root)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_league_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.league_history'`.

- [ ] **Step 3: Implement `pipeline/league_history.py`**

```python
"""Import one league's ESPN history into that league's own database file.

`pipeline/import_league.py` is the command-line importer for the machine
owner's league: it logs in with Playwright and writes to the default
database. This module is the same import for any league an account has
connected through the site -- the account's own ESPN cookies do the
authentication, and the rows land in `data/leagues/<id>.duckdb` (the
default league still resolves to the shared file, see
`pipeline.leagues.league_db_path`).

It runs in the background when an account connects to a real league's
room, so that by the time the draft ends the history a report card needs
is already on disk and the draft-end job needs no network call and no
credential.
"""
from __future__ import annotations

import json
import threading
import traceback

import pandas as pd

from pipeline import sources
from pipeline.db import DEFAULT_PATH, get_conn, read_table, record_freshness, write_table
from pipeline.espn_drafts import KONA, http_fetch
from pipeline.espn_league import PLAYERS_FILTER, import_seasons
from pipeline.import_league import _HISTORIC_ADP_COLUMNS, normalize_historic_adp
from pipeline.leagues import LEAGUES_ROOT, league_db_path, provision_league
from scoring.config import CURRENT_SEASON

FRESH_DAYS = 7


def json_fetch(fetch, cookies: dict):
    """The one-argument JSON fetch `import_seasons` wants, over a cookie fetch.

    `fetch(url, cookies, headers) -> (status, text)` is the shape of
    `espn_drafts.http_fetch`. 404 becomes FileNotFoundError because that is
    what the season walk reads as "no such season"; 401/403 becomes
    PermissionError so an expired session is a named failure rather than
    fifteen silent misses.
    """
    def get(url: str):
        headers = dict(KONA)
        if "/players?" in url:
            headers["X-Fantasy-Filter"] = PLAYERS_FILTER
        status, text = fetch(url, cookies, headers)
        if status == 404:
            raise FileNotFoundError(url)
        if status in (401, 403):
            raise PermissionError(f"{status} for {url}")
        if status >= 400:
            raise RuntimeError(f"{status} for {url}")
        return json.loads(text)
    return get


def is_fresh(conn, max_age_days: int = FRESH_DAYS) -> bool:
    meta = read_table(conn, "meta")
    if meta.empty:
        return False
    row = meta[meta["source"] == "league"]
    if row.empty or not bool(row.iloc[0]["ok"]):
        return False
    age = pd.Timestamp.now() - pd.Timestamp(row.iloc[0]["refreshed_at"])
    return age.days < max_age_days


def import_history(league_id: str, cookies: dict, fetch=None,
                   current_season: int = CURRENT_SEASON,
                   universal_path: str = DEFAULT_PATH, root: str = LEAGUES_ROOT,
                   adp_fetch=None) -> dict:
    """Import drafted seasons, standings and historic ADP for one league.

    Returns `import_seasons`'s summary. Raises what `import_seasons` raises
    (no drafted seasons, an auction league, a dead session); the caller
    decides whether that is a log line or an error.
    """
    raw = fetch if fetch is not None else http_fetch()
    fetch_adp = adp_fetch if adp_fetch is not None else sources.fetch_adp
    path = provision_league(league_id, universal_path, root=root)
    conn = get_conn(path)
    try:
        summary = import_seasons(conn, league_id, current_season, json_fetch(raw, cookies))
        existing = read_table(conn, "historic_adp")
        have = set(existing["season"]) if not existing.empty else set()
        frames = [existing] if not existing.empty else []
        for season in summary["seasons"]:
            if season in have:
                continue
            try:
                df = fetch_adp(season)
            except Exception as exc:      # noqa: BLE001 -- one missing year
                print(f"  WARN historic ADP {season}: {exc}")
                continue
            df = normalize_historic_adp(df)
            df = df.sort_values("adp").reset_index(drop=True)
            df["adp_rank"] = df.index + 1
            df["season"] = season
            frames.append(df[_HISTORIC_ADP_COLUMNS])
        if frames:
            rows = pd.concat(frames, ignore_index=True)
            write_table(conn, "historic_adp", rows)
            record_freshness(conn, "historic_adp", True, len(rows))
        record_freshness(conn, "league", True, len(summary["seasons"]))
        return summary
    finally:
        conn.close()


def _spawn(name: str, fn) -> None:
    threading.Thread(target=fn, name=f"job-{name}", daemon=True).start()


def spawn_import_if_stale(league_id: str, cookies: dict, spawn=None, **kwargs) -> bool:
    """Start a background import unless this league was imported recently.

    Never raises: this is called from a request path that must not fail
    because a side job could not start. Returns whether a job started.
    """
    root = kwargs.get("root", LEAGUES_ROOT)
    try:
        path = league_db_path(league_id, root=root)
        from pathlib import Path
        if Path(path).exists():
            conn = get_conn(path)
            try:
                if is_fresh(conn):
                    return False
            finally:
                conn.close()
    except Exception:      # noqa: BLE001
        traceback.print_exc()
        return False

    def run():
        try:
            summary = import_history(league_id, cookies, **kwargs)
            print(f"league {league_id}: imported {summary['picks']} picks "
                  f"across {len(summary['seasons'])} seasons")
        except Exception as exc:      # noqa: BLE001 -- a background job
            print(f"league {league_id}: history import failed: {exc}")
    (spawn or _spawn)(f"history-{league_id}", run)
    return True
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_league_history.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/league_history.py tests/test_league_history.py
git commit -m "A league's history imports into its own file with the account's cookies"
```

---

### Task 3: Pick verdicts, pick frames and draft grades

**Files:**
- Create: `scoring/league_report.py`
- Test: `tests/test_league_report.py`

**Interfaces:**
- Consumes: `_match_keys(frame, name_col, team_col)` and `_norm_name` (`scoring/draft_model.py:176`, `scoring/board.py:126`); `_ADP_POSITION_ALIASES` (`scoring/board.py`); `snake_slots(teams, rounds)` (`scoring/draft_sim.py:316`); `league.from_json`, `LeagueSettings.rounds`, `.teams`, `.pick_order`.
- Produces:
  - `PICK_COLUMNS = ["season", "overall_pick", "round", "team_id", "manager", "team_name", "player_name", "position", "market_rank", "value", "verdict"]`
  - `pick_value(overall_pick: int, market_rank: float | None, last_pick: int) -> float | None`
  - `verdict(value: float | None, teams: int) -> str | None` — one of `"steal" | "value" | "market" | "early" | "reach"`, None for no ADP.
  - `settings_for(conn, season) -> LeagueSettings` — that season's row of `league`, else `league.load(conn)`.
  - `names_for(conn) -> dict[int, tuple[str, str | None]]` — `team_id -> (manager, team_name)` from the newest season each team appears in (`league_standings` first, `draft_teams` fallback).
  - `historical_picks(conn, season: int) -> pd.DataFrame` with `PICK_COLUMNS`, from `draft_picks` + `historic_adp`.
  - `live_picks(drafted: list[tuple[str, int]], board_by_id: dict, settings, names: dict, team_slots: dict, season: int) -> pd.DataFrame` with `PICK_COLUMNS`, from the room.
  - `draft_grades(picks: pd.DataFrame, teams: int) -> list[dict]` — one dict per team, keys `manager, team_name, team_id, grade, value_total, value_per_pick, graded_picks, steals, reaches, best_pick, worst_pick, shape`; sorted best grade first.
  - `letter(rank0: int, n: int) -> str`.

Note: the spec names `draft_model.build_observations` as the matching path. This task uses the same keys (`_match_keys`) and the same per-season dedupe on `historic_adp` but skips `_enrich_pool`, which loads player attributes the report does not need; the historical market rank is `adp_rank`. The live season uses the room board's blended `market_rank`, the number the ticker showed during the draft.

- [ ] **Step 1: Write the failing tests**

`tests/test_league_report.py`:

```python
import json

import pandas as pd
import pytest

from pipeline.db import get_conn, write_table
from scoring import league as league_mod

TEAMS = 8


def _settings(season, teams=TEAMS):
    return league_mod.LeagueSettings(
        season=season, teams=teams,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
        flex_slots=0, bench=1, scoring={}, draft_type="SNAKE",
        pick_order=tuple(range(1, teams + 1)))


def seed_league(path, seasons=(2024, 2025), teams=TEAMS, rounds=None):
    """An 8-team league: team i drafts picks with a fixed ADP gap.

    Team 1 always gets steals (value +10), team 8 always reaches (-10), the
    rest are on the market. Standings: team i finished i-th in 2024 with
    (10 - i) wins; 2025 is in progress (null finish).
    """
    conn = get_conn(path)
    rows, adp, standings, leagues = [], [], [], []
    for season in seasons:
        s = _settings(season, teams)
        rounds_n = rounds or s.rounds
        from scoring.draft_sim import snake_slots
        slots = snake_slots(teams, rounds_n)
        leagues.append({"season": season, "settings_json": league_mod.to_json(s),
                        "name": "Test League"})
        for i, slot in enumerate(slots):
            overall = i + 1
            name = f"P{season}_{overall}"
            gap = {1: 10, teams: -10}.get(slot, 0)
            rank = max(1, overall - gap)
            rows.append({"season": season, "overall_pick": overall,
                         "round": i // teams + 1, "round_pick": i % teams + 1,
                         "team_id": slot, "espn_player_id": overall,
                         "player_name": name,
                         "position": "RB" if i % 2 else "WR", "nfl_team": "DET",
                         "keeper": False})
            adp.append({"season": season, "adp_name": name,
                        "position": "RB" if i % 2 else "WR", "team": "DET",
                        "adp_rank": rank})
        for t in range(1, teams + 1):
            done = season == 2024
            standings.append({
                "season": season, "team_id": t, "manager": f"m{t}",
                "team_name": f"Team {t}", "wins": 10 - t if done else 0,
                "losses": t + 4 if done else 0, "ties": 0,
                "points_for": 2000.0 - 50 * t if done else 0.0,
                "points_against": 1800.0, "playoff_seed": t if done and t <= 4 else None,
                "final_rank": t if done else None})
    write_table(conn, "draft_picks", pd.DataFrame(rows))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": season, "team_id": t, "manager": f"m{t}", "slot": t}
        for season in seasons for t in range(1, teams + 1)]))
    write_table(conn, "historic_adp", pd.DataFrame(adp))
    write_table(conn, "league_standings", pd.DataFrame(standings))
    write_table(conn, "league", pd.DataFrame(leagues))
    return conn


def test_verdict_bands_match_the_ticker():
    from scoring.league_report import verdict
    assert verdict(None, 8) is None
    assert verdict(2, 8) == "market" and verdict(-2, 8) == "market"
    assert verdict(3, 8) == "value" and verdict(7, 8) == "value"
    assert verdict(8, 8) == "steal"
    assert verdict(-3, 8) == "early" and verdict(-7, 8) == "early"
    assert verdict(-8, 8) == "reach"
    # `round` floors at 2, so a 1-team league still has bands.
    assert verdict(2.4, 1) == "market" and verdict(3, 1) == "steal"


def test_pick_value_is_null_past_the_last_pick():
    from scoring.league_report import pick_value
    assert pick_value(10, 4.0, 120) == 6.0
    assert pick_value(10, None, 120) is None
    assert pick_value(98, 212.0, 120) is None


def test_letter_cut_points_scale_with_team_count():
    from scoring.league_report import letter
    assert [letter(r, 8) for r in range(8)] == ["A", "B", "B", "C", "C", "D", "D", "F"]
    assert [letter(r, 10) for r in range(10)] == ["A", "B", "B", "C", "C", "C", "D", "D", "F", "F"]


def test_historical_picks_join_adp_and_names(tmp_path):
    from scoring.league_report import historical_picks, PICK_COLUMNS
    conn = seed_league(str(tmp_path / "t.duckdb"))
    picks = historical_picks(conn, 2024)
    assert list(picks.columns) == PICK_COLUMNS
    assert len(picks) == TEAMS * _settings(2024).rounds
    first = picks.sort_values("overall_pick").iloc[0]
    assert first["manager"] == "m1" and first["team_name"] == "Team 1"
    assert first["value"] == pytest.approx(0.0)      # rank floors at 1 for pick 1
    t1 = picks[picks["team_id"] == 1]
    assert (t1["value"].iloc[1:] == 10).all()
    assert set(picks[picks["team_id"] == 8]["verdict"]) == {"reach"}


def test_historical_picks_is_empty_without_adp_for_that_season(tmp_path):
    from scoring.league_report import historical_picks
    conn = seed_league(str(tmp_path / "t.duckdb"), seasons=(2024,))
    assert historical_picks(conn, 2023).empty


def test_live_picks_reads_the_room(tmp_path):
    from scoring.league_report import live_picks, PICK_COLUMNS
    settings = _settings(2026, teams=2)
    board_by_id = {
        "a": {"player_id": "a", "name": "A Star", "position": "WR", "market_rank": 4.0},
        "b": {"player_id": "b", "name": "B Steady", "position": "RB", "market_rank": 1.0},
        # Market rank past the last pick: value must be null.
        "c": {"player_id": "c", "name": "C Kicker", "position": "K", "market_rank": 250.0},
    }
    names = {1: ("m1", "Team 1"), 2: ("m2", "Team 2")}
    drafted = [("a", 1), ("b", 2), ("c", 3), ("zzz", 4)]
    picks = live_picks(drafted, board_by_id, settings, names, {2: "ESPN Two"}, 2026)
    assert list(picks.columns) == PICK_COLUMNS
    assert len(picks) == 4
    a = picks[picks["overall_pick"] == 1].iloc[0]
    assert a["manager"] == "m1" and a["value"] == pytest.approx(-3.0) and a["verdict"] == "early"
    b = picks[picks["overall_pick"] == 2].iloc[0]
    assert b["manager"] == "m2" and b["team_name"] == "Team 2"
    assert pd.isna(picks[picks["overall_pick"] == 3].iloc[0]["value"])
    unknown = picks[picks["overall_pick"] == 4].iloc[0]
    assert unknown["player_name"] == "zzz" and pd.isna(unknown["market_rank"])


def test_live_picks_falls_back_to_espn_team_name(tmp_path):
    from scoring.league_report import live_picks
    settings = _settings(2026, teams=2)
    picks = live_picks([("a", 1), ("b", 2)], {}, settings, {}, {1: "Piss Floor"}, 2026)
    assert picks.iloc[0]["manager"] == "Piss Floor" and picks.iloc[0]["team_name"] == "Piss Floor"
    assert picks.iloc[1]["manager"] == "Team 2"


def test_draft_grades_rank_teams(tmp_path):
    from scoring.league_report import historical_picks, draft_grades
    conn = seed_league(str(tmp_path / "t.duckdb"))
    grades = draft_grades(historical_picks(conn, 2024), TEAMS)
    assert [g["manager"] for g in grades][0] == "m1"
    assert [g["manager"] for g in grades][-1] == "m8"
    assert [g["grade"] for g in grades] == ["A", "B", "B", "C", "C", "D", "D", "F"]
    top = grades[0]
    assert top["steals"] == _settings(2024).rounds - 1 and top["reaches"] == 0
    assert top["best_pick"]["player_name"].startswith("P2024_")
    assert top["best_pick"]["verdict"] == "steal"
    assert top["worst_pick"]["value"] == pytest.approx(0.0)
    assert top["shape"]["positions"] == {"WR": 4, "RB": 3} or sum(top["shape"]["positions"].values()) == 7
    assert top["shape"]["first_round"]["QB"] is None
    assert grades[-1]["reaches"] == _settings(2024).rounds


def test_draft_grades_with_no_gradeable_picks():
    from scoring.league_report import draft_grades, PICK_COLUMNS
    assert draft_grades(pd.DataFrame(columns=PICK_COLUMNS), 8) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_league_report.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'scoring.league_report'`.

- [ ] **Step 3: Implement the first half of `scoring/league_report.py`**

```python
"""Everything a league report says in numbers.

Pure functions over a league's database (and, for the season being drafted,
the live room's own tables): which picks were steals and which were
reaches, a draft grade per team, a profile per manager from how they have
drafted and finished over the years, and a power ranking that blends this
draft with that history. No network, no model. `scoring/blurbs.py` writes
prose against the dict `build_facts` returns; `api/reports.py` stores it.
"""
from __future__ import annotations

import math

import pandas as pd

from pipeline.db import read_table
from scoring import league as league_mod
from scoring.board import _ADP_POSITION_ALIASES, _norm_name
from scoring.draft_sim import snake_slots

PICK_COLUMNS = ["season", "overall_pick", "round", "team_id", "manager", "team_name",
                "player_name", "position", "market_rank", "value", "verdict"]

# The ticker's five bands (web/src/components/draft/PickTicker.tsx `verdict`),
# so the report card never disagrees with what the room said as the pick
# landed. `abs(value) <= 2` is on the market; past that, one round's worth of
# picks separates "value" from "steal" and "early" from "reach".
ON_MARKET_SLOTS = 2

# Power ranking weights. This draft counts more than the past because the
# page is a draft report card first; the past still counts because a manager
# who has finished top-three four years running has earned some doubt about
# a C draft.
DRAFT_WEIGHT = 0.6
HISTORY_WEIGHT = 0.4

# Grade cut points as fractions of the league: top eighth A, next quarter B,
# middle quarter C, next quarter D, bottom eighth F. Eight teams grade
# 1-2-2-2-1.
GRADE_CUTS = ((1 / 8, "A"), (3 / 8, "B"), (5 / 8, "C"), (7 / 8, "D"))


def pick_value(overall_pick: int, market_rank, last_pick: int):
    """Picks past ADP (positive is a steal), or None when there is no ADP or
    the market ranked the player past the draft's last pick -- the kicker
    and defense rule from `api/live._board_cell`."""
    if market_rank is None or (isinstance(market_rank, float) and math.isnan(market_rank)):
        return None
    if float(market_rank) > last_pick:
        return None
    return float(overall_pick) - float(market_rank)


def verdict(value, teams: int):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    slots = round(value)
    size = abs(slots)
    rnd = max(2, int(teams))
    if size <= ON_MARKET_SLOTS:
        return "market"
    if slots > 0:
        return "steal" if size >= rnd else "value"
    return "reach" if size >= rnd else "early"


def letter(rank0: int, n: int) -> str:
    frac = (rank0 + 1) / n
    for cut, grade in GRADE_CUTS:
        if frac <= cut + 1e-9:
            return grade
    return "F"


def settings_for(conn, season: int):
    table = read_table(conn, "league")
    if not table.empty:
        row = table[table["season"] == season]
        if not row.empty:
            return league_mod.from_json(row.iloc[0]["settings_json"])
    return league_mod.load(conn)


def league_name(conn, league_id: str) -> str:
    table = read_table(conn, "league")
    if not table.empty and "name" in table.columns:
        names = table.sort_values("season")["name"].dropna()
        if not names.empty:
            return str(names.iloc[-1])
    return f"League {league_id}"


def names_for(conn) -> dict:
    """team_id -> (manager, team_name) from the newest season each appears in."""
    out = {}
    standings = read_table(conn, "league_standings")
    if not standings.empty:
        for _, row in standings.sort_values("season").iterrows():
            out[int(row["team_id"])] = (str(row["manager"]), row.get("team_name"))
    teams = read_table(conn, "draft_teams")
    if not teams.empty:
        for _, row in teams.sort_values("season").iterrows():
            tid = int(row["team_id"])
            if tid not in out:
                out[tid] = (str(row["manager"]), None)
    return out


def _with_verdicts(frame: pd.DataFrame, teams: int, last_pick: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["value"] = [pick_value(o, m, last_pick)
                      for o, m in zip(frame["overall_pick"], frame["market_rank"])]
    frame["verdict"] = [verdict(v, teams) for v in frame["value"]]
    frame["value"] = frame["value"].astype("Float64")
    return frame[PICK_COLUMNS].sort_values("overall_pick").reset_index(drop=True)


def historical_picks(conn, season: int) -> pd.DataFrame:
    """One season's picks from `draft_picks`, priced by that year's ADP."""
    from scoring.draft_model import _match_keys
    picks = read_table(conn, "draft_picks")
    adp = read_table(conn, "historic_adp")
    if picks.empty or adp.empty:
        return pd.DataFrame(columns=PICK_COLUMNS)
    picks = picks[picks["season"] == season].copy()
    adp = adp[adp["season"] == season].copy()
    if picks.empty or adp.empty:
        return pd.DataFrame(columns=PICK_COLUMNS)
    adp = adp.assign(position=adp["position"].replace(_ADP_POSITION_ALIASES))
    adp = adp.assign(key=_match_keys(adp, "adp_name"))
    adp = adp[adp["key"].notna()].sort_values("adp_rank").drop_duplicates("key")
    rank_by_key = dict(zip(adp["key"], adp["adp_rank"].astype(float)))
    picks = picks.assign(key=_match_keys(picks, "player_name", "nfl_team"))
    picks["market_rank"] = [rank_by_key.get(k) for k in picks["key"]]
    picks["market_rank"] = picks["market_rank"].astype("Float64")
    names = names_for(conn)
    picks["manager"] = [names.get(int(t), (f"Team {t}", None))[0] for t in picks["team_id"]]
    picks["team_name"] = [names.get(int(t), (None, None))[1] or f"Team {t}"
                          for t in picks["team_id"]]
    settings = settings_for(conn, season)
    teams = int(settings.teams) or int(picks["team_id"].nunique())
    picks["round"] = ((picks["overall_pick"].astype(int) - 1) // teams) + 1
    last_pick = int(picks["overall_pick"].max())
    return _with_verdicts(picks, teams, last_pick)


def live_picks(drafted: list, board_by_id: dict, settings, names: dict,
               team_slots: dict, season: int) -> pd.DataFrame:
    """The season being drafted, from the room: `drafted` rows and the board.

    `drafted` is `[(player_id, pick_no), ...]`; `board_by_id` is the
    session's `player_id -> board row` dict; `names` is `names_for(conn)`;
    `team_slots` is the session's `slot -> ESPN team name`. The team behind
    a slot is `settings.pick_order[slot - 1]`; a manager the history does
    not know is named by the ESPN team name, then "Team N".
    """
    teams, rounds = int(settings.teams), int(settings.rounds)
    slots = snake_slots(teams, rounds)
    order = list(settings.pick_order or ())
    rows = []
    for player_id, pick_no in drafted:
        overall = int(pick_no)
        slot = slots[overall - 1] if 0 < overall <= len(slots) else None
        team_id = order[slot - 1] if slot is not None and slot - 1 < len(order) else slot
        espn_name = team_slots.get(slot) if slot is not None else None
        manager, team_name = names.get(int(team_id), (None, None)) if team_id is not None else (None, None)
        manager = manager or espn_name or f"Team {slot}"
        team_name = team_name or espn_name or f"Team {slot}"
        row = board_by_id.get(str(player_id)) or {}
        rank = row.get("market_rank")
        rows.append({
            "season": season, "overall_pick": overall,
            "round": ((overall - 1) // teams) + 1 if teams else None,
            "team_id": team_id, "manager": manager, "team_name": team_name,
            "player_name": row.get("name") or str(player_id),
            "position": row.get("position"),
            "market_rank": None if rank is None or pd.isna(rank) else float(rank),
        })
    if not rows:
        return pd.DataFrame(columns=PICK_COLUMNS)
    frame = pd.DataFrame(rows)
    frame["market_rank"] = frame["market_rank"].astype("Float64")
    return _with_verdicts(frame, teams, len(slots))


def _pick_record(row) -> dict:
    return {"player_name": row["player_name"], "position": row["position"],
            "round": int(row["round"]) if pd.notna(row["round"]) else None,
            "overall_pick": int(row["overall_pick"]),
            "market_rank": None if pd.isna(row["market_rank"]) else float(row["market_rank"]),
            "value": None if pd.isna(row["value"]) else round(float(row["value"]), 1),
            "verdict": row["verdict"]}


def _shape(team: pd.DataFrame) -> dict:
    positions = team["position"].dropna().value_counts().to_dict()
    first = {}
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        at = team[team["position"] == pos]
        first[pos] = int(at["round"].min()) if not at.empty else None
    return {"positions": {str(k): int(v) for k, v in positions.items()},
            "first_round": first}


def draft_grades(picks: pd.DataFrame, teams: int) -> list:
    """One record per team, best grade first."""
    if picks.empty:
        return []
    out = []
    for manager, team in picks.groupby("manager", sort=False):
        graded = team[team["value"].notna()]
        n = int(len(graded))
        total = float(graded["value"].sum()) if n else 0.0
        best = graded.loc[graded["value"].astype(float).idxmax()] if n else None
        worst = graded.loc[graded["value"].astype(float).idxmin()] if n else None
        out.append({
            "manager": str(manager),
            "team_name": str(team.iloc[0]["team_name"]),
            "team_id": None if pd.isna(team.iloc[0]["team_id"]) else int(team.iloc[0]["team_id"]),
            "value_total": round(total, 1),
            "value_per_pick": round(total / n, 2) if n else None,
            "graded_picks": n,
            "steals": int((team["verdict"] == "steal").sum()),
            "reaches": int((team["verdict"] == "reach").sum()),
            "best_pick": _pick_record(best) if best is not None else None,
            "worst_pick": _pick_record(worst) if worst is not None else None,
            "shape": _shape(team),
        })
    # Rank on value per pick; a team with nothing gradeable sorts last.
    out.sort(key=lambda g: (g["value_per_pick"] is None,
                            -(g["value_per_pick"] or 0.0), -g["value_total"]))
    n = len(out)
    for i, g in enumerate(out):
        g["grade"] = letter(i, n)
    return out
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_league_report.py -v`
Expected: all PASS. If `test_draft_grades_rank_teams` fails on `top["steals"]`: pick 1 for team 1 has value 0 (rank floors at 1), so team 1 has `rounds - 1` steals. If the shape assertion trips, the position alternation in the seed decides WR/RB counts; the second half of that `or` accepts any split summing to `rounds`.

- [ ] **Step 5: Commit**

```bash
git add scoring/league_report.py tests/test_league_report.py
git commit -m "A pick's verdict in Python, and a draft grade per team"
```

---

### Task 4: Team profiles, power rankings, the facts document

**Files:**
- Modify: `scoring/league_report.py`
- Test: `tests/test_league_report.py`

**Interfaces:**
- Consumes: Task 3.
- Produces:
  - `team_profiles(conn) -> list[dict]` keyed on manager: `manager, team_name, seasons: [{season, wins, losses, ties, points_for, final_rank, playoff_seed}], titles: [season], playoffs: [season], completed: int, win_pct: float | None, avg_finish: float | None, ppg: float | None, drafts: int, mean_value: float | None, steal_rate: float | None, reach_rate: float | None, first_pick_positions: {pos: count}, career_best: pick | None, career_worst: pick | None, model_notes: [str]`
  - `power_rankings(grades: list, profiles: list) -> list[dict]`: `rank, manager, team_name, score, first_year: bool`
  - `build_facts(conn, league_id: str, season: int, picks: pd.DataFrame | None = None) -> dict` — the payload without prose. `status` is `"ready"` (numbers computed; prose pending) or `"failed"` with `reason`.
  - `merge_prose(facts: dict, prose: dict | None, model: str | None) -> dict` — fills `intro`, `nickname`, `blurb`, `line`; sets `status` to `"ready"` when prose is present, `"numbers_only"` when `prose is None`; leaves `"failed"` alone.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_league_report.py`:

```python
def test_team_profiles_summarise_history(tmp_path):
    from scoring.league_report import team_profiles
    conn = seed_league(str(tmp_path / "t.duckdb"))
    profiles = {p["manager"]: p for p in team_profiles(conn)}
    m1 = profiles["m1"]
    assert m1["team_name"] == "Team 1"
    assert [s["season"] for s in m1["seasons"]] == [2024, 2025]
    assert m1["titles"] == [2024] and m1["playoffs"] == [2024]
    assert m1["completed"] == 1
    assert m1["win_pct"] == pytest.approx(9 / 14)
    assert m1["avg_finish"] == 1.0
    assert m1["drafts"] == 2
    assert m1["mean_value"] > 9        # every pick but the first is +10
    assert m1["steal_rate"] > 0.9 and m1["reach_rate"] == 0.0
    assert m1["first_pick_positions"] == {"WR": 2}
    assert m1["career_best"]["value"] == 10.0
    m5 = profiles["m5"]
    assert m5["playoffs"] == [] and m5["titles"] == []
    assert m5["mean_value"] == pytest.approx(0.0)


def test_team_profiles_without_standings_still_have_draft_habits(tmp_path):
    from scoring.league_report import team_profiles
    conn = seed_league(str(tmp_path / "t.duckdb"))
    conn.execute("DROP TABLE league_standings")
    profiles = {p["manager"]: p for p in team_profiles(conn)}
    assert profiles["m8"]["seasons"] == [] and profiles["m8"]["win_pct"] is None
    assert profiles["m8"]["reach_rate"] == 1.0


def test_power_rankings_blend_draft_and_history():
    from scoring.league_report import power_rankings
    grades = [{"manager": "a", "team_name": "A", "value_per_pick": 5.0, "value_total": 50},
              {"manager": "b", "team_name": "B", "value_per_pick": 0.0, "value_total": 0},
              {"manager": "c", "team_name": "C", "value_per_pick": -5.0, "value_total": -50}]
    profiles = [{"manager": "a", "win_pct": 0.2, "completed": 3},
                {"manager": "b", "win_pct": 0.9, "completed": 3},
                {"manager": "c", "win_pct": None, "completed": 0}]
    ranks = power_rankings(grades, profiles)
    assert [r["manager"] for r in ranks] == ["b", "a", "c"] or [r["manager"] for r in ranks] == ["a", "b", "c"]
    assert ranks[0]["rank"] == 1
    c = next(r for r in ranks if r["manager"] == "c")
    assert c["first_year"] is True
    # With equal drafts, history decides.
    tie = [{"manager": m, "team_name": m, "value_per_pick": 0.0, "value_total": 0} for m in "ab"]
    assert power_rankings(tie, profiles)[0]["manager"] == "b"


def test_build_facts_assembles_the_document(tmp_path):
    from scoring.league_report import build_facts
    conn = seed_league(str(tmp_path / "t.duckdb"))
    facts = build_facts(conn, "424242", 2024)
    assert facts["status"] == "ready"
    assert facts["league_id"] == "424242" and facts["season"] == 2024
    assert facts["league_name"] == "Test League" and facts["teams"] == TEAMS
    assert len(facts["report_cards"]) == TEAMS and len(facts["power_rankings"]) == TEAMS
    assert len(facts["profiles"]) == TEAMS
    assert facts["report_cards"][0]["grade"] == "A"
    assert facts["intro"] is None and facts["report_cards"][0]["blurb"] is None
    json.dumps(facts)      # must be plain JSON


def test_build_facts_fails_with_a_reason_without_picks(tmp_path):
    from scoring.league_report import build_facts
    conn = seed_league(str(tmp_path / "t.duckdb"), seasons=(2024,))
    facts = build_facts(conn, "1", 2023)
    assert facts["status"] == "failed" and "no graded picks" in facts["reason"]


def test_build_facts_takes_live_picks(tmp_path):
    from scoring.league_report import build_facts, live_picks
    conn = seed_league(str(tmp_path / "t.duckdb"))
    settings = _settings(2026)
    board = {"x": {"player_id": "x", "name": "X", "position": "RB", "market_rank": 20.0}}
    picks = live_picks([("x", 1)], board, settings, {1: ("m1", "Team 1")}, {}, 2026)
    facts = build_facts(conn, "424242", 2026, picks=picks)
    assert facts["status"] == "ready"
    assert facts["report_cards"][0]["manager"] == "m1"
    assert facts["report_cards"][0]["best_pick"]["verdict"] == "steal"


def test_merge_prose_fills_or_marks_numbers_only():
    from scoring.league_report import merge_prose
    facts = {"status": "ready", "intro": None,
             "report_cards": [{"manager": "m1", "nickname": None, "blurb": None}],
             "power_rankings": [{"manager": "m1", "line": None}]}
    done = merge_prose(dict(facts), {"intro": "hi", "cards": [
        {"manager": "m1", "nickname": "The Thief", "blurb": "Stole."}],
        "rankings": [{"manager": "m1", "line": "Top."}]}, "claude-haiku-4-5")
    assert done["status"] == "ready" and done["model"] == "claude-haiku-4-5"
    assert done["intro"] == "hi"
    assert done["report_cards"][0]["nickname"] == "The Thief"
    assert done["power_rankings"][0]["line"] == "Top."
    quiet = merge_prose(dict(facts), None, None)
    assert quiet["status"] == "numbers_only" and quiet["model"] is None
    failed = merge_prose({"status": "failed", "reason": "x"}, None, None)
    assert failed["status"] == "failed"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_league_report.py -v -k "profiles or power or facts or prose"`
Expected: FAIL with `ImportError` on each new name.

- [ ] **Step 3: Implement the second half of `scoring/league_report.py`**

Append:

```python
def _all_historical_picks(conn) -> pd.DataFrame:
    picks = read_table(conn, "draft_picks")
    if picks.empty:
        return pd.DataFrame(columns=PICK_COLUMNS)
    frames = [historical_picks(conn, int(s)) for s in sorted(picks["season"].unique())]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PICK_COLUMNS)


def _model_notes(conn, manager: str) -> list:
    profiles = read_table(conn, "manager_profiles")
    if profiles.empty or "summary" not in profiles.columns:
        return []
    mine = profiles[(profiles["manager"] == manager) & profiles["summary"].notna()]
    return [str(s) for s in mine["summary"].tolist() if str(s).strip()]


def _num(value):
    return None if value is None or pd.isna(value) else float(value)


def team_profiles(conn) -> list:
    """One record per manager: seasons finished, titles, and draft habits."""
    standings = read_table(conn, "league_standings")
    picks = _all_historical_picks(conn)
    managers = []
    for m in (standings["manager"].tolist() if not standings.empty else []) + (
            picks["manager"].tolist() if not picks.empty else []):
        if m not in managers:
            managers.append(m)
    out = []
    for manager in managers:
        mine = (standings[standings["manager"] == manager].sort_values("season")
                if not standings.empty else pd.DataFrame())
        seasons = []
        for _, row in mine.iterrows():
            seasons.append({
                "season": int(row["season"]),
                "wins": None if pd.isna(row["wins"]) else int(row["wins"]),
                "losses": None if pd.isna(row["losses"]) else int(row["losses"]),
                "ties": None if pd.isna(row["ties"]) else int(row["ties"]),
                "points_for": _num(row["points_for"]),
                "final_rank": None if pd.isna(row["final_rank"]) else int(row["final_rank"]),
                "playoff_seed": None if pd.isna(row["playoff_seed"]) else int(row["playoff_seed"]),
            })
        done = [s for s in seasons if s["final_rank"] is not None]
        games = sum((s["wins"] or 0) + (s["losses"] or 0) + (s["ties"] or 0) for s in done)
        wins = sum((s["wins"] or 0) + 0.5 * (s["ties"] or 0) for s in done)
        ppg_games = [(s["points_for"], (s["wins"] or 0) + (s["losses"] or 0) + (s["ties"] or 0))
                     for s in done if s["points_for"] is not None]
        ppg_total_games = sum(g for _, g in ppg_games)
        my_picks = picks[picks["manager"] == manager] if not picks.empty else pd.DataFrame()
        graded = my_picks[my_picks["value"].notna()] if not my_picks.empty else pd.DataFrame()
        firsts = {}
        if not my_picks.empty:
            for _, season_picks in my_picks.groupby("season"):
                pos = season_picks.sort_values("overall_pick").iloc[0]["position"]
                if pos is not None and not pd.isna(pos):
                    firsts[str(pos)] = firsts.get(str(pos), 0) + 1
        team_name = None
        if not mine.empty and "team_name" in mine.columns:
            names = mine["team_name"].dropna()
            team_name = str(names.iloc[-1]) if not names.empty else None
        if team_name is None and not my_picks.empty:
            team_name = str(my_picks.sort_values("season").iloc[-1]["team_name"])
        out.append({
            "manager": str(manager),
            "team_name": team_name or str(manager),
            "seasons": seasons,
            "titles": [s["season"] for s in done if s["final_rank"] == 1],
            "playoffs": [s["season"] for s in seasons if s["playoff_seed"] is not None],
            "completed": len(done),
            "win_pct": round(wins / games, 3) if games else None,
            "avg_finish": round(sum(s["final_rank"] for s in done) / len(done), 2) if done else None,
            "ppg": round(sum(p for p, _ in ppg_games) / ppg_total_games, 1) if ppg_total_games else None,
            "drafts": int(my_picks["season"].nunique()) if not my_picks.empty else 0,
            "mean_value": round(float(graded["value"].astype(float).mean()), 2) if len(graded) else None,
            "steal_rate": round(float((graded["verdict"] == "steal").mean()), 3) if len(graded) else None,
            "reach_rate": round(float((graded["verdict"] == "reach").mean()), 3) if len(graded) else None,
            "first_pick_positions": firsts,
            "career_best": _pick_record(graded.loc[graded["value"].astype(float).idxmax()]) if len(graded) else None,
            "career_worst": _pick_record(graded.loc[graded["value"].astype(float).idxmin()]) if len(graded) else None,
            "model_notes": _model_notes(conn, manager),
        })
    return out


def _z(values: list) -> list:
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return [0.0 for _ in values]
    mean = sum(present) / len(present)
    var = sum((v - mean) ** 2 for v in present) / len(present)
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0 for _ in values]
    return [0.0 if v is None else (v - mean) / sd for v in values]


def power_rankings(grades: list, profiles: list) -> list:
    by_manager = {p["manager"]: p for p in profiles}
    draft_z = _z([g["value_per_pick"] for g in grades])
    history = [by_manager.get(g["manager"], {}).get("win_pct") for g in grades]
    completed = [by_manager.get(g["manager"], {}).get("completed", 0) for g in grades]
    history_z = _z(history)
    rows = []
    for g, dz, hz, done in zip(grades, draft_z, history_z, completed):
        first_year = not done
        score = DRAFT_WEIGHT * dz + HISTORY_WEIGHT * (0.0 if first_year else hz)
        rows.append({"manager": g["manager"], "team_name": g["team_name"],
                     "score": round(score, 3), "first_year": first_year,
                     "_tiebreak": g["value_total"]})
    rows.sort(key=lambda r: (-r["score"], -r["_tiebreak"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        del r["_tiebreak"]
    return rows


def build_facts(conn, league_id: str, season: int, picks: pd.DataFrame | None = None) -> dict:
    """The report, numbers only. `picks` overrides the historical read for a
    season the room just drafted (see `live_picks`)."""
    settings = settings_for(conn, season)
    frame = picks if picks is not None else historical_picks(conn, season)
    base = {"league_id": str(league_id), "season": int(season),
            "league_name": league_name(conn, league_id),
            "teams": int(settings.teams) or int(frame["manager"].nunique() if not frame.empty else 0),
            "generated_at": None, "model": None, "intro": None}
    if frame.empty or frame["value"].notna().sum() == 0:
        return {**base, "status": "failed",
                "reason": f"no graded picks for {season}: the draft is not imported, "
                          "or no ADP is on file for that year",
                "power_rankings": [], "report_cards": [], "profiles": []}
    grades = draft_grades(frame, base["teams"] or int(frame["manager"].nunique()))
    profiles = team_profiles(conn)
    known = {p["manager"] for p in profiles}
    for g in grades:
        if g["manager"] not in known:
            profiles.append({"manager": g["manager"], "team_name": g["team_name"],
                             "seasons": [], "titles": [], "playoffs": [], "completed": 0,
                             "win_pct": None, "avg_finish": None, "ppg": None, "drafts": 0,
                             "mean_value": None, "steal_rate": None, "reach_rate": None,
                             "first_pick_positions": {}, "career_best": None,
                             "career_worst": None, "model_notes": []})
    graded_managers = {g["manager"] for g in grades}
    profiles = [p for p in profiles if p["manager"] in graded_managers]
    for g in grades:
        g["nickname"] = None
        g["blurb"] = None
    ranks = power_rankings(grades, profiles)
    for r in ranks:
        r["line"] = None
    return {**base, "status": "ready", "power_rankings": ranks,
            "report_cards": grades, "profiles": profiles}


def merge_prose(facts: dict, prose: dict | None, model: str | None) -> dict:
    """Lay the writer's words onto the numbers. No prose is a complete
    report with a different status, not an error."""
    if facts.get("status") == "failed":
        return facts
    facts["model"] = model if prose is not None else None
    if prose is None:
        facts["status"] = "numbers_only"
        return facts
    cards = {c["manager"]: c for c in prose.get("cards", [])}
    lines = {r["manager"]: r for r in prose.get("rankings", [])}
    facts["intro"] = prose.get("intro")
    for card in facts["report_cards"]:
        got = cards.get(card["manager"], {})
        card["nickname"] = got.get("nickname")
        card["blurb"] = got.get("blurb")
    for row in facts["power_rankings"]:
        row["line"] = lines.get(row["manager"], {}).get("line")
    facts["status"] = "ready"
    return facts
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_league_report.py -v`
Expected: all PASS. The power-ranking test allows either order for `a`/`b` because the z-blend with three teams is close; the tie case is the decisive assertion.

- [ ] **Step 5: Commit**

```bash
git add scoring/league_report.py tests/test_league_report.py
git commit -m "Team profiles, power rankings, and the report's numbers as one document"
```

---

### Task 5: The writer, and the `anthropic` dependency

**Files:**
- Create: `scoring/blurbs.py`
- Modify: `requirements.txt`
- Test: `tests/test_blurbs.py`

**Interfaces:**
- Consumes: `facts` dict from `build_facts` (Task 4).
- Produces:
  - `MODEL = "claude-haiku-4-5"`, `MAX_TOKENS = 4096`, `SCHEMA` (JSON schema dict), `SYSTEM` (frozen prompt string).
  - `compact_facts(facts) -> dict` — what the model sees.
  - `write_blurbs(client, facts) -> dict | None` — `{"intro": str, "cards": [{manager, nickname, blurb}], "rankings": [{manager, line}]}` or None.
  - `default_client()` — `anthropic.Anthropic()` when `ANTHROPIC_API_KEY` is set, else None.
  - `enabled() -> bool`.

The `client` duck type used: `client.messages.create(**kwargs)` returning an object with `.content` (list of blocks with `.type == "text"` and `.text`), `.stop_reason`, and `._request_id`. Tests pass a fake with exactly that surface.

- [ ] **Step 1: Install the SDK and pin it**

Run: `.venv/bin/pip install "anthropic>=1.0,<2"` and confirm `.venv/bin/python -c "import anthropic, inspect; print(anthropic.__version__); print('output_config' in inspect.signature(anthropic.Anthropic().messages.create).parameters or 'output_config' in str(inspect.signature(anthropic.Anthropic(api_key='x').messages.create)))"` prints the version and `True`. If it prints `False`, the installed SDK predates the parameter: pass it as `extra_body={"output_config": {...}}` instead and note that in the module docstring.

Append to `requirements.txt`:

```
# The Anthropic SDK, for scoring/blurbs.py -- the one call that writes a
# league's draft report card. The request could be made with httpx, which is
# already here; the SDK is what carries the retry policy for 429s and 5xxs,
# the typed errors the writer branches on, and the structured-output
# parameter that pins the response to a schema. Each of those is a thing this
# code would otherwise re-implement and get subtly wrong. Capped at the major
# version for the same reason stripe is.
anthropic>=1.0,<2
```

- [ ] **Step 2: Write the failing tests**

`tests/test_blurbs.py`:

```python
import json

import pytest


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason
        self._request_id = "req_test"


class FakeClient:
    """Answers each `messages.create` from a queue of responses or exceptions."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                answer = outer.answers.pop(0)
                if isinstance(answer, Exception):
                    raise answer
                return answer
        self.messages = _Messages()


def _facts():
    return {
        "league_id": "1", "season": 2026, "league_name": "The League", "teams": 2,
        "status": "ready", "intro": None,
        "power_rankings": [
            {"rank": 1, "manager": "ann", "team_name": "Ann's", "score": 1.0, "first_year": False, "line": None},
            {"rank": 2, "manager": "bob", "team_name": "Bob's", "score": -1.0, "first_year": True, "line": None}],
        "report_cards": [
            {"manager": "ann", "team_name": "Ann's", "team_id": 1, "grade": "A",
             "value_total": 30.0, "value_per_pick": 3.0, "graded_picks": 10, "steals": 2, "reaches": 0,
             "best_pick": {"player_name": "X", "position": "RB", "round": 3, "overall_pick": 5,
                           "market_rank": 20.0, "value": 15.0, "verdict": "steal"},
             "worst_pick": None, "shape": {"positions": {"RB": 5}, "first_round": {"QB": 4}},
             "nickname": None, "blurb": None},
            {"manager": "bob", "team_name": "Bob's", "team_id": 2, "grade": "F",
             "value_total": -30.0, "value_per_pick": -3.0, "graded_picks": 10, "steals": 0, "reaches": 3,
             "best_pick": None, "worst_pick": None, "shape": {"positions": {}, "first_round": {}},
             "nickname": None, "blurb": None}],
        "profiles": [
            {"manager": "ann", "team_name": "Ann's", "seasons": [], "titles": [2024], "playoffs": [2024],
             "completed": 1, "win_pct": 0.7, "avg_finish": 1.0, "ppg": 120.0, "drafts": 2,
             "mean_value": 2.0, "steal_rate": 0.2, "reach_rate": 0.0,
             "first_pick_positions": {"RB": 2}, "career_best": None, "career_worst": None,
             "model_notes": ["reaches for tight ends"]},
            {"manager": "bob", "team_name": "Bob's", "seasons": [], "titles": [], "playoffs": [],
             "completed": 0, "win_pct": None, "avg_finish": None, "ppg": None, "drafts": 0,
             "mean_value": None, "steal_rate": None, "reach_rate": None,
             "first_pick_positions": {}, "career_best": None, "career_worst": None,
             "model_notes": []}],
    }


def _good():
    return json.dumps({"intro": "What a night.",
                       "cards": [{"manager": "ann", "nickname": "The Bandit", "blurb": "Stole X."},
                                 {"manager": "bob", "nickname": "Rookie", "blurb": "First draft."}],
                       "rankings": [{"manager": "ann", "line": "Top."},
                                    {"manager": "bob", "line": "Bottom."}]})


def test_write_blurbs_makes_one_structured_call():
    from scoring import blurbs
    client = FakeClient([_Response(_good())])
    out = blurbs.write_blurbs(client, _facts())
    assert out["intro"] == "What a night."
    assert [c["manager"] for c in out["cards"]] == ["ann", "bob"]
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["max_tokens"] == 4096
    assert "thinking" not in call
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert call["system"] == blurbs.SYSTEM
    body = call["messages"][0]["content"]
    assert "ann" in body and "bob" in body and "The Bandit" not in body


def test_compact_facts_drops_what_the_model_does_not_need():
    from scoring.blurbs import compact_facts
    compact = compact_facts(_facts())
    assert set(compact) == {"league_name", "season", "teams", "power_rankings", "report_cards", "profiles"}
    card = compact["report_cards"][0]
    assert "team_id" not in card and "nickname" not in card and "blurb" not in card
    assert card["grade"] == "A" and card["best_pick"]["player_name"] == "X"
    prof = compact["profiles"][0]
    assert prof["model_notes"] == ["reaches for tight ends"] and "seasons" in prof


def test_missing_manager_is_retried_once_with_the_failure_named():
    from scoring import blurbs
    short = json.dumps({"intro": "x", "cards": [{"manager": "ann", "nickname": "n", "blurb": "b"}],
                        "rankings": [{"manager": "ann", "line": "l"}]})
    client = FakeClient([_Response(short), _Response(_good())])
    out = blurbs.write_blurbs(client, _facts())
    assert out is not None and len(out["cards"]) == 2
    assert len(client.calls) == 2
    retry = client.calls[1]["messages"]
    assert retry[0]["role"] == "user" and retry[1]["role"] == "assistant"
    assert retry[2]["role"] == "user" and "bob" in retry[2]["content"]


def test_two_bad_answers_yield_none():
    from scoring import blurbs
    client = FakeClient([_Response("not json"), _Response("{}")])
    assert blurbs.write_blurbs(client, _facts()) is None
    assert len(client.calls) == 2


def test_unknown_manager_fails_validation():
    from scoring import blurbs
    extra = json.loads(_good())
    extra["cards"].append({"manager": "zed", "nickname": "n", "blurb": "b"})
    client = FakeClient([_Response(json.dumps(extra)), _Response(json.dumps(extra))])
    assert blurbs.write_blurbs(client, _facts()) is None


def test_api_errors_yield_none_and_do_not_raise():
    import anthropic
    from scoring import blurbs
    err = anthropic.APIConnectionError(request=None)
    client = FakeClient([err])
    assert blurbs.write_blurbs(client, _facts()) is None


def test_refusal_yields_none():
    from scoring import blurbs
    client = FakeClient([_Response("", stop_reason="refusal")])
    assert blurbs.write_blurbs(client, _facts()) is None


def test_none_client_is_skipped():
    from scoring import blurbs
    assert blurbs.write_blurbs(None, _facts()) is None


def test_default_client_off_without_a_key(monkeypatch):
    from scoring import blurbs
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert blurbs.default_client() is None and not blurbs.enabled()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert blurbs.enabled() and blurbs.default_client() is not None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_blurbs.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'scoring.blurbs'`.

- [ ] **Step 4: Implement `scoring/blurbs.py`**

```python
"""The one Claude call that writes a league's report card.

Every number on the page is computed in `scoring/league_report.py`; this
module turns those numbers into an intro, a nickname and a blurb per team,
and a one-liner per power ranking. One call for the whole league, so the
jokes stay consistent across teams and the cost stays at a couple of
cents. The response is pinned to a JSON schema and then checked: every
manager present exactly once, nobody invented. A bad answer is retried once
with the failure named; a second bad answer is None, and the report is
stored as numbers only.

`client` is injected so tests pass a fake; `default_client()` builds the
real one from `ANTHROPIC_API_KEY`, and returns None without it, which is
the local default and the same off-switch shape `api/billing.enabled` has.
"""
from __future__ import annotations

import json
import os

KEY_ENV = "ANTHROPIC_API_KEY"
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 4096

SYSTEM = """You write the morning-after draft report card for a fantasy football league. You are one of the league-mates: you roast your friends, you are specific, and you are funny because you are accurate.

Rules:
- Every joke cites a fact from the numbers you are given: a pick, a round, a grade, a record, a title, a habit. Never invent a player, a pick, a season, a record or a trade.
- Do not mention real people outside this league.
- Keep it clean enough for a group chat that has somebody's parent in it: no slurs, nothing sexual, no jokes about a person's body.
- Team names are the managers' own jokes; use them as written. Managers are named by their display names; refer to each manager by the exact `manager` string you were given.
- A nickname is 2-4 words, a title the league would actually use.
- A blurb is two or three sentences. Lead with the grade's reason (the best or worst pick, the pattern), then the history if there is one.
- A first-year manager has no history; say so rather than inventing one.
- A ranking line is one sentence, at most 20 words.
- The intro is two or three sentences about the draft as a whole.

Answer with JSON matching the schema and nothing else."""

SCHEMA = {
    "type": "object",
    "properties": {
        "intro": {"type": "string"},
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "manager": {"type": "string"},
                    "nickname": {"type": "string"},
                    "blurb": {"type": "string"},
                },
                "required": ["manager", "nickname", "blurb"],
                "additionalProperties": False,
            },
        },
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "manager": {"type": "string"},
                    "line": {"type": "string"},
                },
                "required": ["manager", "line"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["intro", "cards", "rankings"],
    "additionalProperties": False,
}

_CARD_KEYS = ("manager", "team_name", "grade", "value_total", "value_per_pick",
              "graded_picks", "steals", "reaches", "best_pick", "worst_pick", "shape")
_PROFILE_KEYS = ("manager", "team_name", "seasons", "titles", "playoffs", "completed",
                 "win_pct", "avg_finish", "ppg", "drafts", "mean_value", "steal_rate",
                 "reach_rate", "first_pick_positions", "career_best", "career_worst",
                 "model_notes")
_RANK_KEYS = ("rank", "manager", "team_name", "score", "first_year")


def enabled() -> bool:
    return bool(os.environ.get(KEY_ENV, "").strip())


def default_client():
    if not enabled():
        return None
    import anthropic
    return anthropic.Anthropic()


def compact_facts(facts: dict) -> dict:
    """The numbers the model needs and nothing it does not."""
    return {
        "league_name": facts.get("league_name"),
        "season": facts.get("season"),
        "teams": facts.get("teams"),
        "power_rankings": [{k: r.get(k) for k in _RANK_KEYS} for r in facts.get("power_rankings", [])],
        "report_cards": [{k: c.get(k) for k in _CARD_KEYS} for c in facts.get("report_cards", [])],
        "profiles": [{k: p.get(k) for k in _PROFILE_KEYS} for p in facts.get("profiles", [])],
    }


def _validate(text: str, managers: set) -> tuple:
    """(parsed, problem). `problem` names what is wrong, for the retry."""
    try:
        data = json.loads(text)
    except ValueError:
        return None, "the answer was not valid JSON"
    if not isinstance(data, dict):
        return None, "the answer was not a JSON object"
    for field in ("cards", "rankings"):
        rows = data.get(field)
        if not isinstance(rows, list):
            return None, f"`{field}` is missing"
        seen = [r.get("manager") for r in rows if isinstance(r, dict)]
        missing = sorted(managers - set(seen))
        extra = sorted(set(seen) - managers)
        dupes = sorted({m for m in seen if seen.count(m) > 1})
        if missing:
            return None, f"`{field}` is missing managers: {', '.join(missing)}"
        if extra:
            return None, f"`{field}` names managers not in this league: {', '.join(extra)}"
        if dupes:
            return None, f"`{field}` repeats managers: {', '.join(dupes)}"
    if not isinstance(data.get("intro"), str):
        return None, "`intro` is missing"
    return data, None


def _text_of(response) -> str:
    return "".join(getattr(b, "text", "") for b in getattr(response, "content", [])
                   if getattr(b, "type", None) == "text")


def write_blurbs(client, facts: dict) -> dict | None:
    if client is None:
        return None
    import anthropic

    managers = {c["manager"] for c in facts.get("report_cards", [])}
    if not managers:
        return None
    messages = [{"role": "user",
                 "content": "Here is the league:\n" + json.dumps(compact_facts(facts), sort_keys=True)}]
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM, messages=messages,
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            request_id = getattr(getattr(exc, "response", None), "headers", {}).get("request-id") \
                if getattr(exc, "response", None) is not None else None
            print(f"blurbs: {type(exc).__name__}: {exc} (request {request_id})")
            return None
        except Exception as exc:      # noqa: BLE001 -- the job never raises out of here
            print(f"blurbs: {type(exc).__name__}: {exc}")
            return None
        if getattr(response, "stop_reason", None) == "refusal":
            print(f"blurbs: refused (request {getattr(response, '_request_id', None)})")
            return None
        text = _text_of(response)
        data, problem = _validate(text, managers)
        if data is not None:
            return data
        print(f"blurbs: attempt {attempt + 1} rejected: {problem} "
              f"(request {getattr(response, '_request_id', None)})")
        messages = messages + [
            {"role": "assistant", "content": text or "{}"},
            {"role": "user", "content": f"That answer was rejected: {problem}. "
                                        "Answer again with every manager exactly once, "
                                        "as JSON matching the schema."}]
    return None
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_blurbs.py -v`
Expected: all PASS. If `anthropic.APIConnectionError(request=None)` rejects `None` in the installed version, construct it with `request=httpx2.Request("GET", "https://x")` (`import httpx2`) in the test instead.

- [ ] **Step 6: Commit**

```bash
git add scoring/blurbs.py tests/test_blurbs.py requirements.txt
git commit -m "One Haiku call writes the report card, and stays out of the numbers"
```

---

### Task 6: Storage, the build job, and the report routes

**Files:**
- Create: `api/reports.py`
- Modify: `api/main.py:843-856` (between `register_seo_routes` and `register_spa`)
- Test: `tests/test_reports_api.py`

**Interfaces:**
- Consumes: `league_report.build_facts`, `merge_prose`; `blurbs.write_blurbs`, `default_client`, `MODEL`; `league_history.import_history`, `is_fresh`; `leagues.league_db_path`, `DEFAULT_LEAGUE_ID`; `billing.require_paid`, `is_free_draft`, `enabled`, `entitled`; `drafts.session_for`; `espn_drafts.league_entries`; `get_conn`.
- Produces:
  - `REPORTS_DDL`, `store_report(conn, payload)`, `load_report(conn, season) -> dict | None`, `list_reports(conn) -> list[dict]`.
  - `open_read(league_id, root=None)` — read-only DuckDB connection or None when no file (retries a lock 4×0.15s, then raises 503).
  - `build_report(league_id, season, picks=None, client=None, root=None) -> dict` — the whole pipeline; stores and returns the payload.
  - `spawn_build(league_id, season, picks=None, client=None, spawn=None, before=None, root=None) -> bool` — single flight per `(league_id, season)`; `before` is an optional callable run first on the thread (the POST path's history import). Returns False if one is already running.
  - `building(league_id, season) -> bool`.
  - `register_report_routes(app, store=None, fetch=None, spawn=None, root=None)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_reports_api.py`:

```python
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pipeline.db import get_conn, read_table, write_table
from tests.test_api import _seed
from tests.test_league_report import seed_league


@pytest.fixture
def league_root(tmp_path, monkeypatch):
    """A leagues root with one imported league, 424242, and billing off."""
    root = tmp_path / "leagues"
    root.mkdir()
    seed_league(str(root / "424242.duckdb")).close()
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return str(root)


def _app(tmp_path, league_root, session=None, entries=None, spawn=None):
    from api.main import create_app
    from api import reports
    path = str(tmp_path / "nfl.duckdb")
    _seed(path)
    app = create_app(path)
    # The routes are already registered by create_app against the real root;
    # register a second set against the fixture root would clash, so tests
    # build a bare app and register only the report routes.
    from fastapi import FastAPI
    bare = FastAPI()
    reports.register_report_routes(
        bare, root=league_root, spawn=spawn or (lambda name, fn: fn()),
        session_for=(lambda request: session),
        entries_for=(lambda sess: entries or []))
    return TestClient(bare)


def test_store_and_load_round_trip(tmp_path):
    from api import reports
    conn = get_conn(str(tmp_path / "t.duckdb"))
    assert reports.list_reports(conn) == []
    assert reports.load_report(conn, 2024) is None
    reports.store_report(conn, {"season": 2024, "status": "ready", "model": "m", "x": 1})
    reports.store_report(conn, {"season": 2024, "status": "ready", "model": "m", "x": 2})
    reports.store_report(conn, {"season": 2023, "status": "numbers_only", "model": None})
    got = reports.load_report(conn, 2024)
    assert got["x"] == 2 and got["generated_at"]
    rows = reports.list_reports(conn)
    assert [r["season"] for r in rows] == [2024, 2023]
    assert set(rows[0]) == {"season", "generated_at", "status"}


def test_build_report_stores_numbers_only_without_a_client(league_root):
    from api import reports
    payload = reports.build_report("424242", 2024, client=None, root=league_root)
    assert payload["status"] == "numbers_only" and payload["model"] is None
    conn = get_conn(f"{league_root}/424242.duckdb")
    assert reports.load_report(conn, 2024)["report_cards"][0]["grade"] == "A"


def test_build_report_uses_the_writer(league_root):
    from api import reports
    from tests.test_blurbs import FakeClient, _Response
    managers = [f"m{i}" for i in range(1, 9)]
    answer = json.dumps({"intro": "Hi", "cards": [
        {"manager": m, "nickname": "N", "blurb": "B"} for m in managers],
        "rankings": [{"manager": m, "line": "L"} for m in managers]})
    payload = reports.build_report("424242", 2024, client=FakeClient([_Response(answer)]),
                                   root=league_root)
    assert payload["status"] == "ready" and payload["model"] == "claude-haiku-4-5"
    assert payload["intro"] == "Hi" and payload["report_cards"][0]["nickname"] == "N"


def test_build_report_stores_a_failure(league_root):
    from api import reports
    payload = reports.build_report("424242", 2019, root=league_root)
    assert payload["status"] == "failed" and "no graded picks" in payload["reason"]


def test_spawn_build_is_single_flight(league_root):
    from api import reports
    held = []
    def spawn(name, fn):
        held.append(fn)          # do not run yet
    assert reports.spawn_build("424242", 2024, spawn=spawn, root=league_root)
    assert reports.building("424242", 2024)
    assert not reports.spawn_build("424242", 2024, spawn=spawn, root=league_root)
    held[0]()
    assert not reports.building("424242", 2024)


def test_get_reports_lists_and_serves(tmp_path, league_root):
    from api import reports
    reports.build_report("424242", 2024, root=league_root)
    client = _app(tmp_path, league_root)
    assert client.get("/api/leagues/999/reports").json() == []
    assert client.get("/api/leagues/999/report/2024").status_code == 404
    rows = client.get("/api/leagues/424242/reports").json()
    assert rows[0]["season"] == 2024 and rows[0]["status"] == "numbers_only"
    resp = client.get("/api/leagues/424242/report/2024")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "public, max-age=300"
    assert resp.json()["league_name"] == "Test League"
    assert client.get("/api/leagues/424242/report/2019").status_code == 404


def test_post_requires_a_session_that_owns_the_league(tmp_path, league_root):
    client = _app(tmp_path, league_root, session=None)
    assert client.post("/api/leagues/424242/report/2024").status_code == 403

    class Sess:
        swid = "{X}"
        cookies = {"SWID": "{X}", "espn_s2": "s"}
    client = _app(tmp_path, league_root, session=Sess(), entries=[{"league_id": "1"}])
    assert client.post("/api/leagues/424242/report/2024").status_code == 403


def test_post_refuses_a_mock_league(tmp_path, league_root, monkeypatch):
    from api import billing
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: True)

    class Sess:
        swid = "{X}"
        cookies = {}
    client = _app(tmp_path, league_root, session=Sess(), entries=[{"league_id": "424242"}])
    assert client.post("/api/leagues/424242/report/2024").status_code == 400


def test_post_builds_and_answers_202(tmp_path, league_root, monkeypatch):
    from api import billing, reports
    from pipeline import league_history
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: False)
    imported = []
    monkeypatch.setattr(reports, "import_history",
                        lambda league_id, cookies, **kw: imported.append(league_id))

    class Sess:
        swid = "{X}"
        cookies = {"SWID": "{X}", "espn_s2": "s"}
    client = _app(tmp_path, league_root, session=Sess(), entries=[{"league_id": "424242"}])
    resp = client.post("/api/leagues/424242/report/2024")
    assert resp.status_code == 202 and resp.json() == {"status": "building"}
    # The synchronous spawn ran the job: history was refreshed, report stored.
    assert imported == ["424242"]
    assert client.get("/api/leagues/424242/report/2024").status_code == 200


def test_post_while_building_does_not_start_another(tmp_path, league_root, monkeypatch):
    from api import billing, reports
    monkeypatch.setattr(billing, "is_free_draft", lambda league_id: False)
    held = []

    class Sess:
        swid = "{X}"
        cookies = {}
    client = _app(tmp_path, league_root, session=Sess(), entries=[{"league_id": "424242"}],
                  spawn=lambda name, fn: held.append(fn))
    assert client.post("/api/leagues/424242/report/2024").status_code == 202
    assert client.post("/api/leagues/424242/report/2024").status_code == 202
    assert len(held) == 1
    held[0]()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_reports_api.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'api.reports'`.

- [ ] **Step 3: Implement `api/reports.py`**

```python
"""League reports: build, store, serve.

A report is one JSON document per league-season -- power rankings, a
report card per team, a profile per manager -- computed by
`scoring/league_report.py`, written up by `scoring/blurbs.py`, and stored
in the league's own database in `league_reports`. Stored, never rebuilt on
read: a shared link costs nothing after the first build and every reader
sees the same blurbs.

Two ways in. The live listener calls `on_draft_complete` when the last pick
of a real draft lands (`api/live.py`). The owner calls
`POST /api/leagues/{id}/report/{season}` to build a past season or rebuild
this one; that path refreshes the league's history with the session's
cookies first. Both run the build on a daemon thread, one flight per
league-season, because a build takes seconds and the request must not.

Reading is public. Building is the owner's, and gated by billing exactly
as the draft itself is.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
import time
import traceback
from pathlib import Path

import duckdb
from fastapi import HTTPException, Request, Response

from api import billing
from pipeline import espn_drafts
from pipeline.db import get_conn
from pipeline.league_history import import_history, is_fresh
from pipeline.leagues import LEAGUES_ROOT, league_db_path
from scoring import blurbs, league_report
from scoring.config import CURRENT_SEASON

REPORTS_DDL = """CREATE TABLE IF NOT EXISTS league_reports (
    season BIGINT, generated_at TIMESTAMP, model VARCHAR, status VARCHAR,
    payload_json VARCHAR)"""

LOCK_TRIES = 4
LOCK_RETRY_SECONDS = 0.15
CACHE_HEADER = "public, max-age=300"

_lock = threading.Lock()
_inflight: set = set()


# -- storage -----------------------------------------------------------------


def store_report(conn, payload: dict) -> None:
    conn.execute(REPORTS_DDL)
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    payload = dict(payload)
    payload["generated_at"] = now.isoformat(timespec="seconds") + "Z"
    conn.execute("DELETE FROM league_reports WHERE season = ?", [int(payload["season"])])
    conn.execute("INSERT INTO league_reports VALUES (?, ?, ?, ?, ?)",
                 [int(payload["season"]), now, payload.get("model"),
                  payload.get("status"), json.dumps(payload)])


def _has_table(conn) -> bool:
    return bool(conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'league_reports'"
    ).fetchone()[0])


def load_report(conn, season: int) -> dict | None:
    if not _has_table(conn):
        return None
    row = conn.execute("SELECT payload_json FROM league_reports WHERE season = ?",
                       [int(season)]).fetchone()
    return json.loads(row[0]) if row else None


def list_reports(conn) -> list:
    if not _has_table(conn):
        return []
    rows = conn.execute(
        "SELECT season, generated_at, status FROM league_reports ORDER BY season DESC"
    ).fetchall()
    return [{"season": int(s), "generated_at": g.isoformat(timespec="seconds") + "Z",
             "status": st} for s, g, st in rows]


def _open(path: str, read_only: bool):
    last = None
    for attempt in range(LOCK_TRIES):
        try:
            return duckdb.connect(path, read_only=True) if read_only else get_conn(path)
        except Exception as exc:                # noqa: BLE001
            if "lock" not in str(exc).lower():
                raise
            last = exc
            if attempt + 1 < LOCK_TRIES:
                time.sleep(LOCK_RETRY_SECONDS)
    raise HTTPException(status_code=503,
                        detail=f"this league's database is busy -- try again in a moment ({last})")


def open_read(league_id: str, root: str | None = None):
    """A read-only connection to a league's file, or None when there is none."""
    path = league_db_path(league_id, root=root or LEAGUES_ROOT)
    if not Path(path).exists():
        return None
    return _open(path, read_only=True)


# -- the job -------------------------------------------------------------------


def build_report(league_id: str, season: int, picks=None, client=None,
                 root: str | None = None) -> dict:
    """Compute, write, store. Returns the stored payload.

    The league's connection is held for the reads, released for the model
    call, and reopened for the one write, so the live listener writing
    `drafted` to the same file is never blocked for the seconds the writer
    takes.
    """
    path = league_db_path(league_id, root=root or LEAGUES_ROOT)
    conn = _open(path, read_only=False)
    try:
        facts = league_report.build_facts(conn, league_id, int(season), picks=picks)
    finally:
        conn.close()
    prose = None
    if facts["status"] != "failed":
        prose = blurbs.write_blurbs(client if client is not None else blurbs.default_client(),
                                    facts)
    payload = league_report.merge_prose(facts, prose, blurbs.MODEL)
    conn = _open(path, read_only=False)
    try:
        store_report(conn, payload)
        return load_report(conn, int(season))
    finally:
        conn.close()


def building(league_id: str, season: int) -> bool:
    with _lock:
        return (str(league_id), int(season)) in _inflight


def _spawn(name: str, fn) -> None:
    threading.Thread(target=fn, name=f"job-{name}", daemon=True).start()


def spawn_build(league_id: str, season: int, picks=None, client=None, spawn=None,
                before=None, root: str | None = None) -> bool:
    """Start one build for this league-season unless one is running."""
    key = (str(league_id), int(season))
    with _lock:
        if key in _inflight:
            return False
        _inflight.add(key)

    def run():
        try:
            if before is not None:
                before()
            build_report(league_id, season, picks=picks, client=client, root=root)
            print(f"league {league_id}: report for {season} stored")
        except Exception:      # noqa: BLE001 -- a background job
            print(f"league {league_id}: report for {season} failed")
            traceback.print_exc()
        finally:
            with _lock:
                _inflight.discard(key)
    (spawn or _spawn)(f"report-{league_id}-{season}", run)
    return True


# -- the live room's hooks -----------------------------------------------------


def cookies_for_connect(request: Request, swid: str, espn_s2: str | None, store=None) -> dict | None:
    """The ESPN cookies a connect request can act with, or None.

    The bookmarklet may send `espn_s2` in the body; a returning browser
    carries the custody cookie instead; the machine owner has a saved
    login. Same order `api/drafts.session_for` uses.
    """
    if espn_s2:
        return espn_drafts.cookies_for(swid, espn_s2)
    from api.drafts import session_for
    session = session_for(request, store)
    return session.cookies if session is not None else None


def on_draft_complete(league_id: str, season, swid: str | None, session, drafted: list,
                      store=None, spawn=None, root: str | None = None) -> bool:
    """The last pick landed: build this season's report if the room may.

    Free instances build every real draft; a paid instance builds only a
    draft somebody paid for, checked on the connecting account's SWID
    because there is no request on the listener thread. Mocks never build.
    Never raises.
    """
    try:
        if billing.is_free_draft(league_id):
            return False
        year = int(season or CURRENT_SEASON)
        if billing.enabled():
            if not swid:
                return False
            from api.custody import _store
            if not billing.entitled(_store(store).account_ids(swid), league_id, year):
                return False
        conn = open_read(league_id, root=root)
        try:
            names = league_report.names_for(conn) if conn is not None else {}
        finally:
            if conn is not None:
                conn.close()
        picks = league_report.live_picks(
            drafted, getattr(session, "board_by_id", {}) or {}, session.settings, names,
            getattr(session, "team_slots", {}) or {}, year)
        return spawn_build(league_id, year, picks=picks, spawn=spawn, root=root)
    except Exception:      # noqa: BLE001
        traceback.print_exc()
        return False


# -- routes --------------------------------------------------------------------


def _reads_conn(league_id: str, root):
    conn = open_read(league_id, root=root)
    if conn is None:
        return None
    return conn


def register_report_routes(app, store=None, fetch=None, spawn=None, root: str | None = None,
                           session_for=None, entries_for=None):
    """`GET /api/leagues/{id}/reports`, `GET .../report/{season}`,
    `POST .../report/{season}`. `session_for` and `entries_for` are injectable
    for tests; the defaults read custody and the account's ESPN league list."""
    from api.drafts import session_for as _default_session_for

    def _session(request):
        if session_for is not None:
            return session_for(request)
        return _default_session_for(request, store)

    def _entries(session):
        if entries_for is not None:
            return entries_for(session)
        return espn_drafts.league_entries(session.swid, session.cookies, None, fetch=fetch)

    @app.get("/api/leagues/{league_id}/reports")
    def reports_index(league_id: str):
        conn = _reads_conn(league_id, root)
        if conn is None:
            return []
        try:
            return list_reports(conn)
        finally:
            conn.close()

    @app.get("/api/leagues/{league_id}/report/{season}")
    def report(league_id: str, season: int, response: Response):
        conn = _reads_conn(league_id, root)
        if conn is None:
            raise HTTPException(status_code=404, detail="No report for this league.")
        try:
            payload = load_report(conn, season)
        finally:
            conn.close()
        if payload is None:
            raise HTTPException(status_code=404, detail="No report for this season.")
        response.headers["Cache-Control"] = CACHE_HEADER
        return payload

    @app.post("/api/leagues/{league_id}/report/{season}", status_code=202)
    def build(league_id: str, season: int, request: Request):
        session = _session(request)
        if session is None:
            raise HTTPException(status_code=403, detail="Connect your ESPN account first.")
        if not any(str(e.get("league_id")) == str(league_id) for e in _entries(session)):
            raise HTTPException(status_code=403, detail="That league is not one of yours.")
        if billing.is_free_draft(league_id):
            raise HTTPException(status_code=400, detail="Mock drafts have no report card.")
        billing.require_paid(request, league_id, season, store)
        cookies = session.cookies

        def refresh_history():
            try:
                import_history(league_id, cookies, current_season=CURRENT_SEASON, root=root or LEAGUES_ROOT)
            except Exception as exc:      # noqa: BLE001 -- the build runs on what is there
                print(f"league {league_id}: history refresh failed: {exc}")
        spawn_build(league_id, season, spawn=spawn, before=refresh_history, root=root)
        return {"status": "building"}
```

Note `import_history` is imported at module top so the test can monkeypatch `reports.import_history`.

In `api/main.py`, between `register_seo_routes(app, conn)` and the SPA block:

```python
    # League reports: public reads of a stored report, owner-only builds.
    # Before the SPA for the same reason the SEO pages are.
    from api.reports import register_report_routes
    register_report_routes(app)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_reports_api.py tests/test_api.py -q`
Expected: all PASS. If `test_post_builds_and_answers_202` fails because `import_history` was called with `root=` only, the monkeypatched lambda accepts `**kw`; check the failure text.

- [ ] **Step 5: Commit**

```bash
git add api/reports.py api/main.py tests/test_reports_api.py
git commit -m "League reports build on a thread, store in the league's file, and serve publicly"
```

---

### Task 7: The live room imports history on connect and builds the report on the last pick

**Files:**
- Modify: `api/live.py:2445-2499` (`on_change`), `api/live.py:3452-3597` (`live_connect_token`), `api/live.py:3412-3441` (`live_connect`)
- Test: `tests/test_live_api.py`

**Interfaces:**
- Consumes: `reports.on_draft_complete(league_id, season, swid, session, drafted)`, `reports.cookies_for_connect(request, swid, espn_s2)`, `league_history.spawn_import_if_stale(league_id, cookies)`, `espn_drafts.saved_session()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_api.py`, next to the existing connect-token tests (line ~2202; reuse that test's fixture/monkeypatch setup for a connect-token call — copy the request body and monkeypatches from `test_connect_token_resolves_slot_and_wires_socket_picks` verbatim into a helper if the file does not already have one):

```python
def test_connect_token_spawns_a_history_import(tmp_path, monkeypatch):
    """The connect request is the one moment the account's cookies are in
    hand, so that is when the league's history is fetched -- not at the end
    of the draft, which has no request and no credential."""
    from api import live, reports
    from pipeline import league_history
    spawned = []
    monkeypatch.setattr(league_history, "spawn_import_if_stale",
                        lambda league_id, cookies, **kw: spawned.append((league_id, cookies)) or True)
    monkeypatch.setattr(reports, "cookies_for_connect",
                        lambda request, swid, espn_s2, store=None: {"SWID": swid, "espn_s2": "s2"})
    client, body = _connect_token_setup(tmp_path, monkeypatch)   # the existing test's setup
    resp = client.post("/api/live/connect-token", json=body)
    assert resp.status_code == 200
    assert spawned == [(body["leagueId"], {"SWID": body["swid"], "espn_s2": "s2"})]


def test_last_pick_calls_on_draft_complete(tmp_path, monkeypatch):
    """The `made >= total_picks` branch hands the room to the report job with
    what it needs: league, season, the connecting SWID, the session and the
    drafted rows -- once."""
    from api import live, reports
    calls = []
    monkeypatch.setattr(reports, "on_draft_complete",
                        lambda league_id, season, swid, session, drafted, **kw: calls.append(
                            (league_id, season, swid, len(drafted))) or True)
    # Drive the existing socket-picks fixture through every pick of a
    # 2-team, 1-round session (see the existing test for how the fake
    # socket feeds picks), then assert.
    client, body, feed = _connect_token_with_picks(tmp_path, monkeypatch, teams=2, rounds=1)
    feed([("p1", 1), ("p2", 2)])
    assert calls == [(body["leagueId"], body["season"], body["swid"], 2)]
```

The two helpers `_connect_token_setup` and `_connect_token_with_picks` do not exist yet: extract them from `test_connect_token_resolves_slot_and_wires_socket_picks` (`tests/test_live_api.py:2213`) so that test and the new ones share one setup. Read that test first; it monkeypatches the socket runner and the team-slot fetch, and posts a `TokenBody`. If extracting is more than a mechanical move, keep the original test intact and write the new tests as copies of its body with the assertions above.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_live_api.py -k "history_import or on_draft_complete" -v`
Expected: FAIL (`spawned == []`, `calls == []`).

- [ ] **Step 3: Wire the hooks in `api/live.py`**

At the top of `api/live.py` with the other `from api ...` / `from pipeline ...` imports:

```python
from api import reports
from pipeline import league_history
```

In `on_change` (`api/live.py:2489-2490`), replace

```python
                if mine and total_picks and made >= total_picks:
                    clear_session_record(db_path)
```

with

```python
                if mine and total_picks and made >= total_picks:
                    clear_session_record(db_path)
                    # The draft is over: hand the room to the report card.
                    # Read the picks here, where the connection is, and let
                    # the job compute off this thread. The token record is
                    # the connecting account (None on the browser path).
                    c3 = work_conn.cursor()
                    try:
                        drafted = c3.execute(
                            "SELECT player_id, pick_no FROM drafted "
                            "WHERE pick_no IS NOT NULL ORDER BY pick_no").fetchall()
                    finally:
                        c3.close()
                    with lock:
                        token = state.get("token") or {}
                    reports.on_draft_complete(
                        league_id, token.get("season"), token.get("swid"),
                        current["session"], [(str(p), int(n)) for p, n in drafted])
```

`league_id` is the `_launch_listener` parameter (`api/live.py:2199`), in scope inside `on_change`.

In `live_connect_token`, after the `work_conn, league_conn, session = _connect_work(...)` line (`api/live.py:3526`):

```python
        # The account's cookies are in hand right now and at no later point
        # in this draft, so this is when the league's history is fetched for
        # the report card. Background, idempotent, never blocks the connect.
        if not billing.is_free_draft(body.leagueId):
            cookies = reports.cookies_for_connect(request, body.swid, body.espn_s2)
            if cookies:
                league_history.spawn_import_if_stale(body.leagueId, cookies)
```

`billing` is already imported in `api/live.py` (it is used for `require_paid` at `:3161`); if the module is imported under another name, use that name.

In `live_connect` (`api/live.py:3412-3441`), after `_connect_work`:

```python
        # The browser-observer path is the machine owner's: the saved login
        # is the credential, when there is one.
        local = espn_drafts.saved_session()
        if local is not None and not billing.is_free_draft(league_id):
            league_history.spawn_import_if_stale(
                league_id, espn_drafts.cookies_for(local[0], local[1]))
```

Add `from pipeline import espn_drafts` to the imports if `api/live.py` does not already import it.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_live_api.py tests/test_demo_live.py -q`
Expected: all PASS. If a pre-existing connect-token test now fails because `spawn_import_if_stale` tried to open `data/leagues/1.duckdb`: it checks `Path(path).exists()` first and the test root has no such file, so it spawns a thread whose `import_history` fails and prints; that is harmless but noisy. If it is noisy in the suite, monkeypatch `league_history.spawn_import_if_stale` to a no-op in `tests/conftest.py` with an autouse fixture, mirroring the SEO warm-up fixture there, and have the two new tests override it.

- [ ] **Step 5: Commit**

```bash
git add api/live.py tests/test_live_api.py tests/conftest.py
git commit -m "The room fetches the league's history on connect and orders the report card on the last pick"
```

---

### Task 8: The report page

**Files:**
- Modify: `web/src/api.ts` (append)
- Create: `web/src/pages/LeagueReport.tsx`, `web/src/league.css`
- Modify: `web/src/App.tsx`

**Interfaces:**
- Consumes: the three endpoints from Task 6.
- Produces: `fetchLeagueReports(leagueId)`, `fetchLeagueReport(leagueId, season)`, `buildLeagueReport(leagueId, season)`, and the types `LeagueReport`, `ReportCard`, `ReportPick`, `PowerRank`, `TeamProfile`, `ReportSummary`; route `/leagues/:leagueId/report/:season`.

- [ ] **Step 1: Types and fetchers in `web/src/api.ts`**

Append at the end of the file:

```ts
/** One drafted pick as the report card cites it. */
export interface ReportPick {
  player_name: string
  position: string | null
  round: number | null
  overall_pick: number
  market_rank: number | null
  value: number | null
  verdict: 'steal' | 'value' | 'market' | 'early' | 'reach' | null
}

export interface ReportCard {
  manager: string
  team_name: string
  team_id: number | null
  grade: string
  value_total: number
  value_per_pick: number | null
  graded_picks: number
  steals: number
  reaches: number
  best_pick: ReportPick | null
  worst_pick: ReportPick | null
  shape: { positions: Record<string, number>; first_round: Record<string, number | null> }
  nickname: string | null
  blurb: string | null
}

export interface PowerRank {
  rank: number
  manager: string
  team_name: string
  score: number
  first_year: boolean
  line: string | null
}

export interface TeamSeason {
  season: number
  wins: number | null
  losses: number | null
  ties: number | null
  points_for: number | null
  final_rank: number | null
  playoff_seed: number | null
}

export interface TeamProfile {
  manager: string
  team_name: string
  seasons: TeamSeason[]
  titles: number[]
  playoffs: number[]
  completed: number
  win_pct: number | null
  avg_finish: number | null
  ppg: number | null
  drafts: number
  mean_value: number | null
  steal_rate: number | null
  reach_rate: number | null
  first_pick_positions: Record<string, number>
  career_best: ReportPick | null
  career_worst: ReportPick | null
  model_notes: string[]
}

/** `numbers_only` is a complete report with no prose -- the writer was off
 *  or declined. `failed` carries a `reason`. */
export interface LeagueReport {
  league_id: string
  season: number
  league_name: string
  teams: number
  generated_at: string | null
  model: string | null
  status: 'ready' | 'numbers_only' | 'failed'
  reason?: string
  intro: string | null
  power_rankings: PowerRank[]
  report_cards: ReportCard[]
  profiles: TeamProfile[]
}

export interface ReportSummary {
  season: number
  generated_at: string
  status: LeagueReport['status']
}

export async function fetchLeagueReports(leagueId: string): Promise<ReportSummary[]> {
  const res = await fetch(`/api/leagues/${encodeURIComponent(leagueId)}/reports`)
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}

/** Null on 404: no report for that season yet. */
export async function fetchLeagueReport(leagueId: string, season: number | string): Promise<LeagueReport | null> {
  const res = await fetch(`/api/leagues/${encodeURIComponent(leagueId)}/report/${season}`)
  if (res.status === 404) return null
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}

export async function buildLeagueReport(leagueId: string, season: number | string): Promise<{ status: string }> {
  const res = await fetch(`/api/leagues/${encodeURIComponent(leagueId)}/report/${season}`, { method: 'POST' })
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}
```

- [ ] **Step 2: The page, `web/src/pages/LeagueReport.tsx`**

```tsx
import { useEffect, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import {
  fetchLeagueReport, type LeagueReport, type ReportCard, type ReportPick, type TeamProfile,
} from '../api'
import { useDocumentMeta } from '../lib/documentMeta'
import { Logo } from '../components/Logo'
import '../league.css'

// A report is built once and stored; this page only reads it. While the
// owner's build is in flight the GET answers 404, so a page opened with
// `#building` (the dashboard sends it there right after the POST) polls
// for two minutes before giving up.
const POLL_MS = 3000
const POLL_LIMIT = 40

const VERDICT_WORD: Record<NonNullable<ReportPick['verdict']>, string> = {
  steal: 'Steal', value: 'Value', market: 'On the market', early: 'Early', reach: 'Reach',
}

function tone(verdict: ReportPick['verdict']): string {
  if (verdict === 'steal' || verdict === 'value') return 'is-steal'
  if (verdict === 'reach' || verdict === 'early') return 'is-reach'
  return 'is-flat'
}

function gap(pick: ReportPick): string {
  if (pick.value === null) return 'no ADP'
  const n = Math.round(Math.abs(pick.value))
  return pick.value > 0 ? `${n} past ADP` : pick.value < 0 ? `${n} before ADP` : 'at ADP'
}

function PickLine({ label, pick }: { label: string; pick: ReportPick | null }) {
  if (!pick) return null
  return (
    <p className="lr-pick">
      <span className="lr-pick-label">{label}</span>
      <span className="lr-pick-name">{pick.player_name}</span>
      {pick.position && <span className={`lr-pos lr-pos-${pick.position.toLowerCase()}`}>{pick.position}</span>}
      <span className="mono lr-pick-where">R{pick.round ?? '?'} · #{pick.overall_pick}</span>
      <span className={`lr-pick-verdict ${tone(pick.verdict)}`}>
        {pick.verdict ? VERDICT_WORD[pick.verdict] : 'No ADP'} · {gap(pick)}
      </span>
    </p>
  )
}

function Shape({ card }: { card: ReportCard }) {
  const order = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
  const entries = order.filter((p) => card.shape.positions[p]).map((p) => [p, card.shape.positions[p]] as const)
  if (entries.length === 0) return null
  return (
    <p className="lr-shape">
      {entries.map(([pos, n]) => (
        <span key={pos} className={`lr-shape-chip lr-pos-${pos.toLowerCase()}`}>{n} {pos}</span>
      ))}
    </p>
  )
}

function Card({ card }: { card: ReportCard }) {
  return (
    <li className={`lr-card lr-grade-${card.grade.toLowerCase()}`}>
      <div className="lr-card-head">
        <span className="lr-grade" aria-label={`Grade ${card.grade}`}>{card.grade}</span>
        <div className="lr-card-who">
          <p className="lr-card-team">{card.team_name}</p>
          <p className="lr-card-manager">
            {card.manager}
            {card.nickname && <span className="lr-nickname"> · “{card.nickname}”</span>}
          </p>
        </div>
        <p className="mono lr-card-stat">
          <span className="is-steal">{card.steals} steal{card.steals === 1 ? '' : 's'}</span>
          <span className="lr-sep">·</span>
          <span className="is-reach">{card.reaches} reach{card.reaches === 1 ? '' : 'es'}</span>
        </p>
      </div>
      {card.blurb && <p className="lr-blurb">{card.blurb}</p>}
      <PickLine label="Best" pick={card.best_pick} />
      <PickLine label="Worst" pick={card.worst_pick} />
      <Shape card={card} />
    </li>
  )
}

function record(p: TeamProfile['seasons'][number]): string {
  if (p.wins === null || p.losses === null) return '—'
  const base = `${p.wins}-${p.losses}${p.ties ? `-${p.ties}` : ''}`
  return p.final_rank ? `${base} · ${ordinal(p.final_rank)}` : base
}

function ordinal(n: number): string {
  const s = ['th', 'st', 'nd', 'rd']
  const v = n % 100
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`
}

function habits(p: TeamProfile): string {
  const bits: string[] = []
  const opens = Object.entries(p.first_pick_positions).sort((a, b) => b[1] - a[1])[0]
  if (opens) bits.push(`opens ${opens[0]}`)
  if (p.steal_rate !== null && p.steal_rate >= 0.2) bits.push('finds steals')
  if (p.reach_rate !== null && p.reach_rate >= 0.2) bits.push('reaches')
  if (p.career_best) bits.push(`best pick ever: ${p.career_best.player_name} (R${p.career_best.round ?? '?'})`)
  return bits.join(' · ')
}

function History({ profile }: { profile: TeamProfile }) {
  return (
    <li className="lr-history">
      <div className="lr-history-head">
        <p className="lr-history-team">{profile.team_name}</p>
        <p className="lr-history-manager">{profile.manager}</p>
        <p className="mono lr-history-pct">
          {profile.win_pct === null ? 'first season' : `${Math.round(profile.win_pct * 100)}% career`}
        </p>
      </div>
      {profile.seasons.length > 0 && (
        <ul className="lr-seasons">
          {profile.seasons.map((s) => (
            <li key={s.season} className={`lr-season${s.final_rank === 1 ? ' is-title' : ''}`}>
              <span className="mono lr-season-year">{s.season}</span>
              <span className="mono lr-season-rec">{record(s)}</span>
              {s.final_rank === 1 && <span className="lr-crown" aria-label="champion">♛</span>}
            </li>
          ))}
        </ul>
      )}
      {habits(profile) && <p className="lr-habits">{habits(profile)}</p>}
    </li>
  )
}

export default function LeagueReport() {
  const { leagueId = '', season = '' } = useParams()
  const location = useLocation()
  const [report, setReport] = useState<LeagueReport | null | undefined>(undefined)
  const [error, setError] = useState<string | null>(null)
  const [polls, setPolls] = useState(0)
  const building = report === null && location.hash === '#building' && polls < POLL_LIMIT

  useDocumentMeta({
    title: report ? `${report.league_name} ${report.season} draft report card` : 'Draft report card',
    description: 'Power rankings, a grade for every team and how each manager has drafted and finished over the years.',
  })

  useEffect(() => {
    let cancelled = false
    fetchLeagueReport(leagueId, season)
      .then((body) => { if (!cancelled) setReport(body) })
      .catch((err) => { if (!cancelled) setError(String(err.message || err)) })
    return () => { cancelled = true }
  }, [leagueId, season, polls])

  useEffect(() => {
    if (!building) return
    const id = window.setTimeout(() => setPolls((n) => n + 1), POLL_MS)
    return () => window.clearTimeout(id)
  }, [building, polls])

  return (
    <div className="lr-page">
      <header className="lr-bar">
        <Link to="/" className="lr-logo" aria-label="Home"><Logo /></Link>
      </header>
      {error && <p className="lr-note is-fail">{error}</p>}
      {report === undefined && !error && <p className="lr-note">Loading…</p>}
      {report === null && building && (
        <p className="lr-note">Writing the report card… this takes a few seconds.</p>
      )}
      {report === null && !building && (
        <p className="lr-note">No report for this season.</p>
      )}
      {report && report.status === 'failed' && (
        <p className="lr-note is-fail">This report could not be built: {report.reason}</p>
      )}
      {report && report.status !== 'failed' && (
        <main className="lr-main">
          <section className="lr-hero">
            <p className="lr-kicker">{report.league_name} · {report.season}</p>
            <h1 className="lr-title">Draft report card</h1>
            {report.intro && <p className="lr-intro">{report.intro}</p>}
          </section>

          <section className="lr-section">
            <h2 className="lr-h2">Power rankings</h2>
            <ol className="lr-ranks">
              {report.power_rankings.map((r) => (
                <li key={r.manager} className="lr-rank">
                  <span className="mono lr-rank-n">{r.rank}</span>
                  <div className="lr-rank-who">
                    <p className="lr-rank-team">{r.team_name}</p>
                    <p className="lr-rank-manager">{r.manager}{r.first_year && <span className="lr-first"> · first draft</span>}</p>
                    {r.line && <p className="lr-rank-line">{r.line}</p>}
                  </div>
                </li>
              ))}
            </ol>
          </section>

          <section className="lr-section">
            <h2 className="lr-h2">Report cards</h2>
            <ul className="lr-cards">
              {report.report_cards.map((c) => <Card key={c.manager} card={c} />)}
            </ul>
          </section>

          <section className="lr-section">
            <h2 className="lr-h2">The teams over the years</h2>
            <ul className="lr-histories">
              {report.profiles.map((p) => <History key={p.manager} profile={p} />)}
            </ul>
          </section>

          <footer className="lr-foot mono">
            {report.generated_at && <span>Generated {report.generated_at.slice(0, 10)}</span>}
            {report.status === 'numbers_only' && <span> · numbers only, no write-up</span>}
          </footer>
        </main>
      )}
    </div>
  )
}
```

- [ ] **Step 3: The sheet, `web/src/league.css`**

```css
/* The league report page. Mobile first: the link is opened from a group
   chat on a phone. Widths above 720px get two columns of cards. Tokens are
   the app's own (index.css); `is-steal` / `is-reach` / `is-flat` carry the
   same meaning they do on the draft board's ticker. */

.lr-page {
  min-height: 100vh;
  background: var(--bg-0, #0f1115);
  color: var(--text-1, #e6e6e6);
  font-family: var(--font-ui);
  font-size: var(--text-md);
  line-height: 1.5;
}
.lr-bar { display: flex; align-items: center; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.lr-logo { display: inline-flex; color: inherit; text-decoration: none; }

.lr-main { max-width: 960px; margin: 0 auto; padding: 16px; }
.lr-note { max-width: 960px; margin: 24px auto; padding: 0 16px; color: var(--text-2, #a7abb4); }
.lr-note.is-fail { color: var(--fail); }

.lr-hero { padding: 8px 0 20px; }
.lr-kicker { margin: 0; color: var(--text-3, #7c8190); font-size: var(--text-sm); text-transform: uppercase; letter-spacing: .08em; }
.lr-title { margin: 4px 0 8px; font-size: 28px; line-height: 1.1; }
.lr-intro { margin: 0; font-size: var(--text-lg); color: var(--text-1); }

.lr-section { padding: 12px 0 20px; border-top: 1px solid var(--border); }
.lr-h2 { margin: 0 0 12px; font-size: var(--text-xl); }

.lr-ranks { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
.lr-rank { display: grid; grid-template-columns: 36px 1fr; gap: 10px; align-items: start; }
.lr-rank-n { font-size: 22px; color: var(--accent); line-height: 1.1; }
.lr-rank-team { margin: 0; font-weight: 600; }
.lr-rank-manager { margin: 0; color: var(--text-3); font-size: var(--text-sm); }
.lr-rank-line { margin: 2px 0 0; color: var(--text-2); }
.lr-first { color: var(--accent); }

.lr-cards, .lr-histories { list-style: none; margin: 0; padding: 0; display: grid; gap: 12px; }
@media (min-width: 720px) { .lr-cards, .lr-histories { grid-template-columns: 1fr 1fr; } }

.lr-card { border: 1px solid var(--border); border-radius: var(--radius); padding: 12px; background: var(--bg-1, #15181e); }
.lr-card-head { display: grid; grid-template-columns: 48px 1fr auto; gap: 10px; align-items: center; }
.lr-grade { font-family: var(--font-mono); font-size: 34px; font-weight: 700; line-height: 1; text-align: center; }
.lr-grade-a .lr-grade { color: var(--ok); }
.lr-grade-b .lr-grade { color: var(--ramp-strong); }
.lr-grade-c .lr-grade { color: var(--ramp-mid); }
.lr-grade-d .lr-grade { color: var(--accent); }
.lr-grade-f .lr-grade { color: var(--fail); }
.lr-card-team { margin: 0; font-weight: 600; }
.lr-card-manager { margin: 0; color: var(--text-3); font-size: var(--text-sm); }
.lr-nickname { color: var(--text-2); font-style: italic; }
.lr-card-stat { margin: 0; font-size: var(--text-sm); text-align: right; }
.lr-sep { margin: 0 6px; color: var(--text-3); }
.is-steal { color: var(--ok); }
.is-reach { color: var(--fail); }
.is-flat { color: var(--text-3); }

.lr-blurb { margin: 10px 0 8px; }
.lr-pick { margin: 4px 0; display: flex; flex-wrap: wrap; gap: 6px; align-items: baseline; font-size: var(--text-sm); }
.lr-pick-label { color: var(--text-3); width: 40px; }
.lr-pick-name { font-weight: 600; }
.lr-pick-where { color: var(--text-3); }
.lr-pos { font-family: var(--font-mono); font-size: var(--text-xs); padding: 0 4px; border-radius: var(--radius); color: #fff; }
.lr-pos-qb { background: var(--pos-qb); }
.lr-pos-rb { background: var(--pos-rb); }
.lr-pos-wr { background: var(--pos-wr); }
.lr-pos-te { background: var(--pos-te); }
.lr-pos-k { background: var(--pos-k); }
.lr-pos-dst { background: var(--pos-dst); }
.lr-shape { margin: 8px 0 0; display: flex; flex-wrap: wrap; gap: 6px; }
.lr-shape-chip { font-family: var(--font-mono); font-size: var(--text-xs); padding: 1px 6px; border-radius: 10px; color: #fff; }

.lr-history { border: 1px solid var(--border); border-radius: var(--radius); padding: 12px; }
.lr-history-head { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; }
.lr-history-team { margin: 0; font-weight: 600; }
.lr-history-manager { margin: 0; color: var(--text-3); font-size: var(--text-sm); grid-column: 1; }
.lr-history-pct { grid-column: 2; grid-row: 1 / span 2; color: var(--text-2); font-size: var(--text-sm); align-self: center; }
.lr-seasons { list-style: none; margin: 8px 0 0; padding: 0; display: flex; flex-wrap: wrap; gap: 6px 14px; }
.lr-season { display: inline-flex; gap: 6px; align-items: baseline; font-size: var(--text-sm); }
.lr-season-year { color: var(--text-3); }
.lr-season.is-title .lr-season-rec { color: var(--accent); }
.lr-crown { color: var(--accent); }
.lr-habits { margin: 8px 0 0; color: var(--text-2); font-size: var(--text-sm); }

.lr-foot { padding: 16px 0 32px; color: var(--text-3); font-size: var(--text-xs); }
```

Check the actual token names for page background and text (`grep -n '^\s*--bg\|^\s*--text-[0-9]' web/src/index.css`) and replace `--bg-0`, `--bg-1`, `--text-1/2/3` with the real names; the fallbacks after the comma keep the page legible if a name is wrong, but the point is to use the real ones.

- [ ] **Step 4: Route it outside the gate in `web/src/App.tsx`**

```tsx
import LeagueReport from './pages/LeagueReport'
```

and replace the `return (...)` with:

```tsx
  return (
    <Routes>
      {/* The league report is a link dropped in a group chat and opened on
          a phone. It is the one page that lives outside the desktop gate. */}
      <Route path="/leagues/:leagueId/report/:season" element={<LeagueReport />} />
      <Route
        path="*"
        element={
          <MobileGate>
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/draft" element={<DraftRoom />} />
              <Route path="/mocks" element={<MockDrafts />} />
              <Route path="/archive" element={<Market />} />
              <Route path="/archive/data" element={<ArchiveData />} />
              <Route path="/live" element={<Live />} />
              <Route path="/room/:leagueId" element={<WaitingRoomPage />} />
            </Routes>
          </MobileGate>
        }
      />
    </Routes>
  )
```

Keep the existing route comments on their routes.

- [ ] **Step 5: Type-check and build**

Run: `cd web && npm run build`
Expected: `tsc -b` clean, Vite build succeeds. Fix any type error in the page (the most likely: `useParams` returning `string | undefined`, handled by the defaults above).

- [ ] **Step 6: Commit**

```bash
git add web/src/api.ts web/src/pages/LeagueReport.tsx web/src/league.css web/src/App.tsx
git commit -m "A league's draft report card as a page a phone can read"
```

---

### Task 9: The dashboard links to it

**Files:**
- Modify: `web/src/components/Dashboard.tsx:1-6` (imports), `:79-110` (state), `:278-310` (`.db-league-foot`)

**Interfaces:**
- Consumes: `fetchLeagueReports`, `buildLeagueReport` (Task 8).

- [ ] **Step 1: State and fetch**

Add to the imports:

```tsx
import { useNavigate } from 'react-router-dom'
import {
  buildLeagueReport, fetchLeagueReports, mintDraftToken,
  type ReportSummary, type TokenConnectParams, type UpcomingDraft,
} from '../api'
```

(merge with the existing `../api` import line). Inside the component, after `const [roomCount, ...]`:

```tsx
  const navigate = useNavigate()
  // league_id -> its stored reports, newest first. One read per card when
  // the dashboard mounts; undefined until it lands, so the card shows no
  // report link rather than a wrong one.
  const [reports, setReports] = useState<Record<string, ReportSummary[]>>({})
  const [buildingFor, setBuildingFor] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    for (const league of leagues) {
      if ((league.name ?? '').toLowerCase().includes('mock')) continue
      fetchLeagueReports(league.league_id)
        .then((rows) => { if (!cancelled) setReports((r) => ({ ...r, [league.league_id]: rows })) })
        .catch(() => { /* no link, then */ })
    }
    return () => { cancelled = true }
  }, [leagues])

  const build = useCallback(async (league: UpcomingDraft) => {
    const season = league.season ?? new Date().getFullYear()
    setBuildingFor(league.league_id)
    try {
      await buildLeagueReport(league.league_id, season)
      navigate(`/leagues/${league.league_id}/report/${season}#building`)
    } catch (err) {
      setError(String((err as Error).message || err))
      setBuildingFor(null)
    }
  }, [navigate])
```

- [ ] **Step 2: The link in the card foot**

Inside `<div className="db-league-foot">`, after the existing button (both branches of the ternary), add:

```tsx
                      {!(league.name ?? '').toLowerCase().includes('mock') && (
                        reports[league.league_id]?.length ? (
                          <Link
                            className="db-go db-league-go db-go-view"
                            to={`/leagues/${league.league_id}/report/${reports[league.league_id][0].season}`}
                          >
                            Report card
                          </Link>
                        ) : (
                          <button
                            type="button"
                            className="db-go db-league-go db-go-view"
                            disabled={buildingFor === league.league_id}
                            onClick={() => build(league)}
                          >
                            {buildingFor === league.league_id ? 'Building…' : 'Build report card'}
                          </button>
                        )
                      )}
```

The foot is a flex row already (`.db-league-foot` in `App.css`); if two controls overflow on the card, add `flex-wrap: wrap; gap: 8px;` to `.db-league-foot` in `web/src/App.css`.

- [ ] **Step 3: Build**

Run: `cd web && npm run build`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/Dashboard.tsx web/src/App.css
git commit -m "The dashboard's league cards link to the report card, or offer to build it"
```

---

### Task 10: Verify end to end, then merge

**Files:** none new.

- [ ] **Step 1: Full Python suite**

Stop `make up` if it is running (it locks the test DB). Run: `.venv/bin/pytest -q`
Expected: all pass. Report any failure verbatim; do not claim green without the output.

- [ ] **Step 2: Build a real report for the default league, numbers only**

With no `ANTHROPIC_API_KEY` set:

```bash
.venv/bin/python -c "
from api import reports
p = reports.build_report('53929318', 2025)
print(p['status'], p['league_name'], p['teams'])
for c in p['report_cards']: print(c['grade'], c['manager'], c['value_per_pick'], c['best_pick'] and c['best_pick']['player_name'])
for r in p['power_rankings']: print(r['rank'], r['manager'], r['score'], r['first_year'])
"
```

Expected: `numbers_only`, eight cards with grades A..F, eight ranks. `league_standings` does not exist in `data/nfl.duckdb` until the owner's league is re-imported (`make espn-import LEAGUE=53929318`, which needs the Playwright login), so every profile shows `first_year: True` until then; run the import if the saved login works and rerun the build to see records. Note in the final report whether the import ran.

- [ ] **Step 3: One real Haiku call**

If an `ANTHROPIC_API_KEY` is available in the shell, rerun Step 2's command with it exported and confirm `status == 'ready'`, then print `p['intro']` and each card's `nickname` and `blurb`. Read them for the rules in `SYSTEM` (nothing invented, nothing outside the league). If no key is available, say so in the report; the code path is covered by the fake-client tests.

- [ ] **Step 4: The page on a phone viewport**

Start the API (`make api`, or `make up` for both) and run, from `.venv`:

```bash
.venv/bin/python - <<'EOF'
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True)
        pg = await ctx.new_page()
        await pg.goto("http://localhost:5173/leagues/53929318/report/2025", wait_until="networkidle")
        assert await pg.locator(".lr-card").count() == 8, "expected eight cards"
        assert await pg.locator(".lr-rank").count() == 8
        assert await pg.locator("text=numbers only").count() == 1
        assert await pg.locator(".mobile-gate, [class*=mobile-gate]").count() == 0, "the gate rendered"
        width = await pg.evaluate("document.documentElement.scrollWidth")
        assert width <= 390, f"page scrolls sideways: {width}"
        await pg.screenshot(path="/private/tmp/claude-501/-Users-worthy-Test-fantasy-football/5bc97c4b-d3db-403b-8f30-f9977bc3cc4a/scratchpad/report-phone.png", full_page=True)
        await b.close()
asyncio.run(main())
EOF
```

Look at the screenshot. Also load `http://localhost:5173/?signedout=1` at 1280×800 to confirm the landing page is unchanged, and the signed-in dashboard to confirm the league card shows "Report card".

- [ ] **Step 5: Merge**

Per the owner's standing instruction, finished work goes straight to `main` with no PR. If the work was done on a branch, `git checkout main && git merge --no-ff <branch>`; otherwise the commits are already on `main`. Run `.venv/bin/pytest -q` once more on `main`.

- [ ] **Step 6: Report**

State plainly: tests (count, pass/fail), whether the default league import ran, whether a real Haiku call was made and what it wrote, the screenshot path, and that `ANTHROPIC_API_KEY` must be added to Railway's variable store (alongside the variables listed in `README.md:256-267`) for production reports to carry prose.

---

## Self-review

**Spec coverage.** Standings table + league name (Task 1). Per-league import with cookies, idempotent, on connect (Tasks 2, 7). This season's picks from the room (Task 3 `live_picks`, Task 7). Verdict port, grades, profiles, power rankings, payload, `numbers_only`/`failed` statuses (Tasks 3, 4). Writer: one Haiku call, schema, validation, one retry, off-switch, dependency paragraph (Task 5). Storage table, single-flight job, connection held for the write only, three routes with the stated gates and cache header, auto-trigger with `entitled` (Tasks 6, 7). Page outside the gate, five sections, building/404 states, dashboard link and build button, `api.ts` functions (Tasks 8, 9). Tests as listed in the spec, plus the phone-viewport check (Task 10). Out of scope items untouched.

**Deviation from the spec, stated:** Task 3 matches picks to ADP with the same keys `build_observations` uses but without `_enrich_pool`, so a historical pick is priced by `historic_adp.adp_rank` rather than the cheat-sheet blend; the live season is priced by the board's `market_rank`. The spec's "through `build_observations`" is satisfied in matching, not in enrichment.

**Type consistency.** `PICK_COLUMNS` order is used by `_with_verdicts` and asserted by tests. `names_for` returns `dict[int, tuple[str, str|None]]` and both `historical_picks` and `live_picks` consume it that way. `draft_grades` output keys match `_CARD_KEYS` in `blurbs` plus `nickname`/`blurb` added by `build_facts`. `power_rankings` keys match `_RANK_KEYS` plus `line`. `merge_prose` reads `cards`/`rankings`/`intro`, the schema's three properties. `on_draft_complete(league_id, season, swid, session, drafted)` is called with exactly those five positionals from `api/live.py`. `spawn(name, fn)` is the shape shared by `league_history`, `reports`, and `api/jobs._spawn`.
