# Draft Model Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the draft simulator's predictions match how this league actually drafts, by fitting against the rankings its managers actually see, on a scale that can tell elite players apart, with a self-policy that does not draft a quarterback first.

**Architecture:** Historical ESPN rankings join the existing FFC ones as a second market reference. `reach`/`fall` move to a log scale and four player-at-pick-time features are added, all computed by one module shared between historical fitting and current-season simulation. A leave-one-season-out harness with a per-round breakdown and a feature ablation table decides which of them survive. Separately, the board's dedupe becomes a draft-order greedy and the in-rollout self-policy gains a replacement baseline.

**Tech Stack:** Python 3, numpy, pandas, scipy, duckdb, FastAPI.

**Spec:** `docs/superpowers/specs/2026-08-10-draft-model-calibration-design.md`

## Global Constraints

- Every Python test runs offline. No network call, no browser, no reading `data/nfl.duckdb`. Tests seed a DuckDB file under pytest's `tmp_path` with literal fixtures, matching `tests/test_draft_model.py` and `tests/test_draft_sim.py`.
- **No lookahead.** A feature describing a pick in season S may use data from seasons strictly before S, and the player's age at S. Never season S's own results or anything later. Every feature test asserts this explicitly.
- **`feature_matrix` and `_live_features` must produce identical vectors.** `scoring/draft_model.py::feature_matrix` (pandas, used for fitting) and `scoring/draft_sim.py::_live_features` (numpy, used in rollouts) are two implementations of one definition. A mismatch silently invalidates every simulation while everything still runs. `tests/test_draft_sim.py` already has parity tests — they must be extended to cover every new feature, not just kept passing.
- Existing behavior must not change except where a task says so. The full suite (279 tests) must stay green after every task.
- Determinism: every sampling path takes an explicit `numpy.random.Generator`. Never the global numpy RNG.
- Position vocabulary is exactly `QB, RB, WR, TE, K, DST`. Name matching always goes through `scoring.board.adp_match_key`. Never write a second normalizer.
- Run tests with `.venv/bin/pytest`. **The API server holds a lock on `data/nfl.duckdb`**; if `pytest` fails at collection with `IOException: Could not set lock`, run it from another directory with `PYTHONPATH` set rather than stopping the server or moving the file.
- Commit after every task.

## File Structure

| File | Responsibility |
|---|---|
| `pipeline/sources.py` (modify) | Per-season ESPN validation helper |
| `pipeline/import_league.py` (modify) | Write `historic_espn` alongside `historic_adp` |
| `scoring/player_history.py` (new) | Player attributes as of a season: age, volatility, track record, production rank, trend |
| `scoring/draft_model.py` (modify) | Pool enrichment, log-rank + new features, league bias, LOSO validation harness |
| `scoring/draft_sim.py` (modify) | `SimPool` fields, `_live_features` parity, dedupe objective, self-policy |
| `pipeline/fit_managers.py` (modify) | Report per-round accuracy and the ablation table |
| `README.md` (modify) | Document what changed and how to read the report |

---

### Task 1: Historical ESPN rankings

**Files:**
- Modify: `pipeline/sources.py`, `pipeline/import_league.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `sources.fetch_espn_adp(year, limit=500) -> pd.DataFrame` with columns `espn_id, espn_name, position, espn_adp, espn_ppr_rank, espn_proj`
- Produces:
  - `sources.espn_adp_is_usable(df) -> bool` — False when `espn_adp` is constant across all rows or entirely null
  - `import_league` writes table `historic_espn`: `season, espn_name, position, espn_rank, adp_usable`
    - `espn_rank` is a dense 1..N rank per season, ordered by `espn_ppr_rank` ascending, tie-broken by `espn_adp` when usable
    - `adp_usable` is that season's `espn_adp_is_usable` verdict, stored so downstream code and the report can see it

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_espn_adp_is_usable_rejects_a_constant_column():
    # ESPN returns 170.0 for every player in some completed seasons -- a
    # reset value, not a ranking. Verified against the live 2025 season.
    from pipeline.sources import espn_adp_is_usable
    constant = pd.DataFrame({"espn_adp": [170.0, 170.0, 170.0]})
    assert not espn_adp_is_usable(constant)
    assert not espn_adp_is_usable(pd.DataFrame({"espn_adp": [None, None]}))
    assert espn_adp_is_usable(pd.DataFrame({"espn_adp": [1.7, 2.6, 3.8]}))


def test_espn_adp_is_usable_on_a_missing_column():
    from pipeline.sources import espn_adp_is_usable
    assert not espn_adp_is_usable(pd.DataFrame({"other": [1, 2]}))
```

Append to `tests/test_espn_league.py`:

```python
def test_historic_espn_ranks_by_ppr_rank_and_records_adp_usability(tmp_path):
    """Ranks come from espn_ppr_rank, which survives seasons where ESPN's
    espn_adp column resets to a constant."""
    import pandas as pd
    from pipeline.db import get_conn, read_table, write_table
    from pipeline.import_league import build_historic_espn

    frames = {
        2025: pd.DataFrame([  # corrupt ADP, sane ranks
            {"espn_name": "Saquon Barkley", "position": "RB",
             "espn_adp": 170.0, "espn_ppr_rank": 4},
            {"espn_name": "Jahmyr Gibbs", "position": "RB",
             "espn_adp": 170.0, "espn_ppr_rank": 5},
        ]),
        2024: pd.DataFrame([
            {"espn_name": "CeeDee Lamb", "position": "WR",
             "espn_adp": 6.3, "espn_ppr_rank": 1},
            {"espn_name": "Alvin Kamara", "position": "RB",
             "espn_adp": 8.3, "espn_ppr_rank": 2},
        ]),
    }
    out = build_historic_espn(frames)
    assert list(out.columns) == ["season", "espn_name", "position",
                                 "espn_rank", "adp_usable"]
    y25 = out[out["season"] == 2025].sort_values("espn_rank")
    assert y25["espn_name"].tolist() == ["Saquon Barkley", "Jahmyr Gibbs"]
    assert y25["espn_rank"].tolist() == [1, 2]      # dense, not raw ppr_rank
    assert not y25["adp_usable"].any()
    assert out[out["season"] == 2024]["adp_usable"].all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pipeline.py -k espn_adp_is_usable tests/test_espn_league.py -k historic_espn -v`
Expected: FAIL with `ImportError: cannot import name 'espn_adp_is_usable'`

- [ ] **Step 3: Write minimal implementation**

Add to `pipeline/sources.py`:

```python
def espn_adp_is_usable(df: pd.DataFrame) -> bool:
    """Whether a season's `espn_adp` column carries real draft positions.

    ESPN serves a reset value -- 170.0 for every player, verified on the
    2025 season -- for some completed seasons, while that season's
    `espn_ppr_rank` stays correct. A constant column is not a ranking, and
    trusting it silently would order a whole season arbitrarily.
    """
    if "espn_adp" not in df.columns:
        return False
    values = pd.to_numeric(df["espn_adp"], errors="coerce").dropna()
    return len(values) > 1 and values.nunique() > 1
```

Add to `pipeline/import_league.py`:

```python
def build_historic_espn(frames: dict) -> pd.DataFrame:
    """One dense 1..N ranking per season from ESPN's own rankings.

    `espn_ppr_rank` is the ordering, not `espn_adp`: the rank column is
    reliable in every season checked, the ADP column is not (see
    `sources.espn_adp_is_usable`). ADP breaks ties only where it is usable.
    Re-ranking densely rather than passing ESPN's raw rank through means
    the scale matches `historic_adp`'s, so the two references are
    comparable.
    """
    rows = []
    for season, df in sorted(frames.items()):
        if df.empty or "espn_ppr_rank" not in df.columns:
            continue
        usable = sources.espn_adp_is_usable(df)
        ranked = df.dropna(subset=["espn_ppr_rank"]).copy()
        ranked["_adp"] = (pd.to_numeric(ranked["espn_adp"], errors="coerce")
                          if usable else 0.0)
        ranked = ranked.sort_values(["espn_ppr_rank", "_adp"]).reset_index(drop=True)
        ranked["espn_rank"] = ranked.index + 1
        ranked["season"] = season
        ranked["adp_usable"] = usable
        rows.append(ranked[["season", "espn_name", "position",
                            "espn_rank", "adp_usable"]])
    cols = ["season", "espn_name", "position", "espn_rank", "adp_usable"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
```

In `import_league.main`, alongside the existing per-season `fetch_adp` loop, collect ESPN frames and write the table:

```python
    espn_frames = {}
    for season in summary["seasons"]:
        try:
            espn_frames[season] = sources.fetch_espn_adp(season, limit=500)
        except Exception as e:
            print(f"  WARN historic ESPN {season}: {e}")
    historic_espn = build_historic_espn(espn_frames)
    if not historic_espn.empty:
        write_table(conn, "historic_espn", historic_espn)
        record_freshness(conn, "historic_espn", True, len(historic_espn))
        unusable = sorted(historic_espn[~historic_espn["adp_usable"]]["season"].unique())
        if unusable:
            print(f"  note: ESPN ADP unusable for {unusable} "
                  "(reset to a constant) -- ranked on espn_ppr_rank instead")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py tests/test_espn_league.py -q`
Expected: all passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: 282 passed (279 + 3 new)

- [ ] **Step 6: Commit**

```bash
git add pipeline/sources.py pipeline/import_league.py tests/
git commit -m "feat: import ESPN's own historical rankings"
```

---

### Task 2: Player attributes as of a season

**Files:**
- Create: `scoring/player_history.py`
- Test: `tests/test_player_history.py`

**Interfaces:**
- Consumes: the `weekly` and `players` tables; `scoring.board.adp_match_key`; `scoring.ppr.compute_ppr_points`
- Produces:
  - `scoring.player_history.attributes_as_of(conn, season, rules=None) -> pd.DataFrame` with columns `key, age, ppg_std, missed_rate, no_track_record, prod_rank, trend`
    - one row per player with weekly data strictly before `season`, keyed by `adp_match_key(name, position, team)`
    - `age` is years at 1 September of `season`, from `players.birth_date`; NaN when unknown
    - `ppg_std` is the standard deviation of per-season points per game across prior seasons; 0.0 for a single season
    - `missed_rate` is `1 - games_played / (17 * seasons_present)`
    - `no_track_record` is True when the player has no prior weekly rows at all
    - `prod_rank` is a dense 1..N rank by the player's most recent prior season's points per game, best first
    - `trend` is the slope of points per game across prior seasons, per season; 0.0 with fewer than two

- [ ] **Step 1: Write the failing test**

Create `tests/test_player_history.py`:

```python
import pandas as pd
import pytest
from pipeline.db import get_conn, write_table
from scoring.player_history import attributes_as_of


def _seed(tmp_path):
    conn = get_conn(str(tmp_path / "t.duckdb"))
    rows = []
    # Steady: 10 ppg in 2021, 10 in 2022, 10 in 2023.
    for season, ppg in ((2021, 10), (2022, 10), (2023, 10)):
        rows += [{"player_id": "p_steady", "player_display_name": "Steady Sam",
                  "position": "RB", "recent_team": "DET", "opponent_team": "GB",
                  "season": season, "week": w, "receptions": ppg, "receiving_yards": 0,
                  "targets": ppg, "carries": 0} for w in range(1, 18)]
    # Rising: 5, 10, 20.
    for season, ppg in ((2021, 5), (2022, 10), (2023, 20)):
        rows += [{"player_id": "p_rising", "player_display_name": "Rising Rick",
                  "position": "WR", "recent_team": "GB", "opponent_team": "DET",
                  "season": season, "week": w, "receptions": ppg, "receiving_yards": 0,
                  "targets": ppg, "carries": 0} for w in range(1, 18)]
    # Fragile: one 8-game season.
    rows += [{"player_id": "p_hurt", "player_display_name": "Hurt Harry",
              "position": "RB", "recent_team": "CHI", "opponent_team": "GB",
              "season": 2023, "week": w, "receptions": 12, "receiving_yards": 0,
              "targets": 12, "carries": 0} for w in range(1, 9)]
    write_table(conn, "weekly", pd.DataFrame(rows))
    write_table(conn, "players", pd.DataFrame([
        {"gsis_id": "p_steady", "display_name": "Steady Sam",
         "birth_date": "1996-09-01", "rookie_season": 2019},
        {"gsis_id": "p_rising", "display_name": "Rising Rick",
         "birth_date": "2002-09-01", "rookie_season": 2021},
    ]))
    return conn


def test_uses_only_seasons_before_the_draft(tmp_path):
    """The no-lookahead rule: attributes for the 2023 draft may not see 2023."""
    a = attributes_as_of(_seed(tmp_path), 2023).set_index("key")
    steady = a.loc["RB|steady sam"]
    # 2021 + 2022 only, both 10 ppg -> flat, zero spread.
    assert steady["ppg_std"] == pytest.approx(0.0)
    assert steady["trend"] == pytest.approx(0.0)
    # Hurt Harry only played in 2023, so as of the 2023 draft he is unknown.
    assert "RB|hurt harry" not in a.index


def test_trend_is_positive_for_a_rising_player(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    assert a.loc["WR|rising rick"]["trend"] > 0
    assert a.loc["RB|steady sam"]["trend"] == pytest.approx(0.0)


def test_volatility_separates_steady_from_rising(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    assert a.loc["RB|steady sam"]["ppg_std"] == pytest.approx(0.0)
    assert a.loc["WR|rising rick"]["ppg_std"] > 0


def test_missed_rate_flags_a_short_season(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    # Hurt Harry: 8 of 17 possible games in his one season.
    assert a.loc["RB|hurt harry"]["missed_rate"] == pytest.approx(9 / 17)
    assert a.loc["RB|steady sam"]["missed_rate"] == pytest.approx(0.0)


def test_age_is_computed_at_the_drafts_september(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    assert a.loc["RB|steady sam"]["age"] == pytest.approx(28.0, abs=0.05)
    assert a.loc["WR|rising rick"]["age"] == pytest.approx(22.0, abs=0.05)


def test_age_is_nan_when_birth_date_is_unknown(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    assert pd.isna(a.loc["RB|hurt harry"]["age"])


def test_prod_rank_orders_by_most_recent_prior_ppg(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024).set_index("key")
    # 2023 ppg: rising 20 > hurt 12 > steady 10.
    assert a.loc["WR|rising rick"]["prod_rank"] < a.loc["RB|hurt harry"]["prod_rank"]
    assert a.loc["RB|hurt harry"]["prod_rank"] < a.loc["RB|steady sam"]["prod_rank"]


def test_no_track_record_is_false_for_everyone_with_history(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2024)
    assert not a["no_track_record"].any()


def test_empty_before_the_first_season(tmp_path):
    a = attributes_as_of(_seed(tmp_path), 2020)
    assert a.empty
    assert list(a.columns) == ["key", "age", "ppg_std", "missed_rate",
                               "no_track_record", "prod_rank", "trend"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_player_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scoring.player_history'`

- [ ] **Step 3: Write minimal implementation**

Create `scoring/player_history.py`:

```python
"""What was knowable about a player on a given draft day.

Every column here answers "what did the room know when this pick was made",
which is the only thing a model of that pick may use. A feature for a 2022
pick may read 2016-2021 and the player's age in 2022, and nothing later --
the functions below take the draft season and filter on it rather than
leaving that discipline to their callers.

Keyed by `adp_match_key`, the same join key the rest of the pipeline uses,
so these attach to a choice pool without a second name normalizer.
"""
import numpy as np
import pandas as pd

from pipeline.db import read_table
from scoring.board import adp_match_key
from scoring.ppr import compute_ppr_points

COLUMNS = ["key", "age", "ppg_std", "missed_rate", "no_track_record",
           "prod_rank", "trend"]
GAMES = 17
# Drafts happen at the end of August, so age at 1 September of the draft
# year is the age the room would have said out loud.
DRAFT_MONTH_DAY = "-09-01"


def _slope(seasons: np.ndarray, values: np.ndarray) -> float:
    """Points-per-game change per season. Zero with fewer than two seasons,
    which is an absence of evidence rather than a flat trajectory -- callers
    that need to tell those apart read `no_track_record`."""
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(seasons, values, 1)[0])


def attributes_as_of(conn, season: int, rules: dict | None = None) -> pd.DataFrame:
    weekly = read_table(conn, "weekly")
    if weekly.empty:
        return pd.DataFrame(columns=COLUMNS)
    prior = weekly[weekly["season"] < season]
    if prior.empty:
        return pd.DataFrame(columns=COLUMNS)

    wk = prior.copy()
    wk["ppr_points"] = compute_ppr_points(wk, rules)
    per_season = wk.groupby(["player_id", "season"], as_index=False).agg(
        points=("ppr_points", "sum"), games=("week", "nunique"),
        name=("player_display_name", "last"), position=("position", "last"),
        team=("recent_team", "last"))
    per_season["ppg"] = per_season["points"] / per_season["games"]

    rows = []
    for player_id, grp in per_season.groupby("player_id"):
        grp = grp.sort_values("season")
        latest = grp.iloc[-1]
        key = adp_match_key(latest["name"], latest["position"], latest["team"])
        if key is None:
            continue
        ppg = grp["ppg"].to_numpy(dtype=float)
        rows.append({
            "player_id": player_id, "key": key,
            "ppg_std": float(ppg.std(ddof=0)) if len(ppg) > 1 else 0.0,
            "missed_rate": float(max(0.0, 1.0 - grp["games"].sum()
                                     / (GAMES * len(grp)))),
            "no_track_record": False,
            "last_ppg": float(latest["ppg"]),
            "trend": _slope(grp["season"].to_numpy(dtype=float), ppg),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=COLUMNS)

    # Age: `players.birth_date` is the same source the stat-twin matcher
    # already uses. Unknown stays NaN rather than becoming a guess -- the
    # feature builder centres within position and treats NaN as neutral.
    people = read_table(conn, "players")
    if people.empty or "birth_date" not in people.columns:
        out["age"] = np.nan
    else:
        born = people[["gsis_id", "birth_date"]].rename(columns={"gsis_id": "player_id"})
        out = out.merge(born, on="player_id", how="left")
        draft_day = pd.Timestamp(f"{season}{DRAFT_MONTH_DAY}")
        birth = pd.to_datetime(out["birth_date"], errors="coerce")
        out["age"] = (draft_day - birth).dt.days / 365.25

    out = out.sort_values("last_ppg", ascending=False).reset_index(drop=True)
    out["prod_rank"] = out.index + 1
    return out[COLUMNS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_player_history.py -v`
Expected: 9 passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: 291 passed

- [ ] **Step 6: Commit**

```bash
git add scoring/player_history.py tests/test_player_history.py
git commit -m "feat: player attributes as of a draft day"
```

---

### Task 3: The choice pool carries the new columns

**Files:**
- Modify: `scoring/draft_model.py` (`build_observations`)
- Test: `tests/test_draft_model.py`

**Interfaces:**
- Consumes: `historic_espn` from Task 1, `attributes_as_of` from Task 2
- Produces: `PickObservation.pool` gains columns beyond `norm, position, adp_rank, key`:
  - `market_rank` — ESPN's dense rank for that season when `historic_espn` covers the player, else the FFC `adp_rank`. This is what the log-rank features read.
  - `hype` — `prod_rank - market_rank`, positive when the market is ahead of the player's production. NaN when either is unknown.
  - `age`, `ppg_std`, `missed_rate`, `no_track_record`, `trend` — from `attributes_as_of`, with `no_track_record` True and the rest neutral for a player with no history
  - `build_observations` still drops picks with no ADP row, unchanged

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft_model.py`:

```python
def _seed_with_espn(tmp_path):
    """One season, two players, ESPN and FFC disagreeing on the order."""
    conn = get_conn(str(tmp_path / "t.duckdb"))
    write_table(conn, "draft_picks", pd.DataFrame([
        {"season": 2025, "overall_pick": 1, "round": 1, "round_pick": 1,
         "team_id": 1, "espn_player_id": 11, "player_name": "Player A",
         "position": "RB", "nfl_team": "DET", "keeper": False},
        {"season": 2025, "overall_pick": 2, "round": 1, "round_pick": 2,
         "team_id": 2, "espn_player_id": 12, "player_name": "Player B",
         "position": "WR", "nfl_team": "GB", "keeper": False},
    ]))
    write_table(conn, "draft_teams", pd.DataFrame([
        {"season": 2025, "team_id": 1, "manager": "worthy", "slot": 1},
        {"season": 2025, "team_id": 2, "manager": "dan", "slot": 2},
    ]))
    write_table(conn, "historic_adp", pd.DataFrame([
        {"season": 2025, "adp_name": "Player A", "position": "RB",
         "team": "DET", "adp_rank": 1},
        {"season": 2025, "adp_name": "Player B", "position": "WR",
         "team": "GB", "adp_rank": 2},
    ]))
    # ESPN reverses them.
    write_table(conn, "historic_espn", pd.DataFrame([
        {"season": 2025, "espn_name": "Player B", "position": "WR",
         "espn_rank": 1, "adp_usable": True},
        {"season": 2025, "espn_name": "Player A", "position": "RB",
         "espn_rank": 2, "adp_usable": True},
    ]))
    return conn


def test_pool_market_rank_prefers_espn_over_ffc(tmp_path):
    obs = build_observations(_seed_with_espn(tmp_path))
    pool = obs[0].pool.set_index("norm")
    assert pool.loc["player a"]["adp_rank"] == 1        # FFC, unchanged
    assert pool.loc["player a"]["market_rank"] == 2     # ESPN's view
    assert pool.loc["player b"]["market_rank"] == 1


def test_pool_falls_back_to_ffc_when_espn_lacks_the_player(tmp_path):
    from pipeline.db import write_table as wt
    conn = _seed_with_espn(tmp_path)
    wt(conn, "historic_espn", pd.DataFrame([
        {"season": 2025, "espn_name": "Player B", "position": "WR",
         "espn_rank": 1, "adp_usable": True}]))
    pool = build_observations(conn)[0].pool.set_index("norm")
    assert pool.loc["player a"]["market_rank"] == 1     # fell back to adp_rank


def test_pool_carries_player_attributes_with_neutral_defaults(tmp_path):
    obs = build_observations(_seed_with_espn(tmp_path))
    pool = obs[0].pool
    for col in ("market_rank", "hype", "age", "ppg_std", "missed_rate",
                "no_track_record", "trend"):
        assert col in pool.columns
    # No `weekly` table at all, so nobody has a track record.
    assert pool["no_track_record"].all()
    assert (pool["ppg_std"] == 0.0).all()
    assert (pool["trend"] == 0.0).all()


def test_pool_hype_is_market_ahead_of_production(tmp_path):
    conn = _seed_with_espn(tmp_path)
    write_table(conn, "weekly", pd.DataFrame([
        # Player B produced far less than the market's view of him.
        {"player_id": "b", "player_display_name": "Player B", "position": "WR",
         "recent_team": "GB", "opponent_team": "DET", "season": 2024, "week": w,
         "receptions": 1, "receiving_yards": 5, "targets": 2, "carries": 0}
        for w in range(1, 18)] + [
        {"player_id": "a", "player_display_name": "Player A", "position": "RB",
         "recent_team": "DET", "opponent_team": "GB", "season": 2024, "week": w,
         "receptions": 9, "receiving_yards": 90, "targets": 11, "carries": 5}
        for w in range(1, 18)]))
    pool = build_observations(conn)[0].pool.set_index("norm")
    # B: market_rank 1, prod_rank 2 -> hype +1. A: market 2, prod 1 -> -1.
    assert pool.loc["player b"]["hype"] > pool.loc["player a"]["hype"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_model.py -k pool_market_rank -v`
Expected: FAIL with `KeyError: 'market_rank'`

- [ ] **Step 3: Write minimal implementation**

In `scoring/draft_model.py`, add the import and a helper, then extend the per-season pool build inside `build_observations`:

```python
from scoring.player_history import attributes_as_of

_ATTRIBUTE_DEFAULTS = {"age": np.nan, "ppg_std": 0.0, "missed_rate": 0.0,
                       "no_track_record": True, "prod_rank": np.nan,
                       "trend": 0.0}


def _enrich_pool(conn, pool: pd.DataFrame, season: int,
                 espn: pd.DataFrame) -> pd.DataFrame:
    """Attach the market reference and the player-at-pick-time attributes.

    `market_rank` prefers ESPN's dense rank -- the league drafts on ESPN,
    off ESPN's board, so that is the ordering the room actually saw -- and
    falls back to the FFC `adp_rank` for a player ESPN's top-N did not
    reach. Both are dense 1..N per season, so the two are on one scale.
    """
    pool = pool.copy()
    if espn.empty:
        pool["market_rank"] = pool["adp_rank"]
    else:
        season_espn = espn[espn["season"] == season].copy()
        season_espn["key"] = _match_keys(season_espn, "espn_name")
        season_espn = season_espn.dropna(subset=["key"]).drop_duplicates("key")
        ranks = season_espn.set_index("key")["espn_rank"]
        pool["market_rank"] = pool["key"].map(ranks).fillna(pool["adp_rank"])

    attrs = attributes_as_of(conn, season)
    if attrs.empty:
        for col, default in _ATTRIBUTE_DEFAULTS.items():
            pool[col] = default
    else:
        pool = pool.merge(attrs, on="key", how="left")
        pool["no_track_record"] = pool["no_track_record"].fillna(True).astype(bool)
        for col, default in _ATTRIBUTE_DEFAULTS.items():
            if col not in ("no_track_record", "prod_rank"):
                pool[col] = pool[col].fillna(default)

    # Positive means the market is ahead of what the player has actually
    # done -- taking him is a leap of faith. NaN when he has no production
    # to rank, which `feature_matrix` reads as neutral rather than as zero.
    pool["hype"] = pool["prod_rank"] - pool["market_rank"]
    return pool
```

Inside `build_observations`, read `historic_espn` once and call the helper where the pool is built for each season:

```python
    espn = read_table(conn, "historic_espn")
    ...
    for season, season_picks in picks.groupby("season"):
        pool = adp[adp["season"] == season][["norm", "position", "adp_rank", "key"]]
        ...  # existing key filter and dedupe, unchanged
        pool = _enrich_pool(conn, pool, season, espn)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_draft_model.py -q`
Expected: all passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: 295 passed

- [ ] **Step 6: Commit**

```bash
git add scoring/draft_model.py tests/test_draft_model.py
git commit -m "feat: choice pool carries ESPN ranks and player attributes"
```

---

### Task 4: Log-rank features and the four new ones

This is the parity-critical task. `feature_matrix` and `_live_features` are two implementations of one definition.

**Files:**
- Modify: `scoring/draft_model.py` (`FEATURE_NAMES`, `feature_matrix`), `scoring/draft_sim.py` (`SimPool`, `build_pool`, `_live_features`)
- Test: `tests/test_draft_model.py`, `tests/test_draft_sim.py`

**Interfaces:**
- Consumes: the enriched pool from Task 3
- Produces:
  - `FEATURE_NAMES` becomes exactly:
    `["reach", "fall", "pos_RB", "pos_WR", "pos_TE", "pos_K", "pos_DST", "qb_early", "te_early", "need", "run", "age", "volatility", "no_track_record", "hype", "trend"]`
  - `reach = max(0, log1p(market_rank) - log1p(pick_no))`, `fall = max(0, log1p(pick_no) - log1p(market_rank))` — no division by `teams`
  - `age` is centred within position across the pool; NaN becomes 0.0 (neutral)
  - `volatility` is `ppg_std` scaled by dividing by 10.0, plus `missed_rate`
  - `hype` is divided by 50.0 to keep it on the same order as the others; NaN becomes 0.0
  - `SimPool` gains fields `market_rank, age, ppg_std, missed_rate, no_track_record, hype, trend`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_draft_model.py`:

```python
def test_log_rank_makes_the_top_of_the_board_matter_more():
    """The defect this fixes: with linear rank the gap from rank 1 to 5 was
    ten times SMALLER than the gap from 100 to 140, so the model treated
    deep-bench noise as more meaningful than the first pick of the draft."""
    import numpy as np
    from scoring.draft_model import feature_matrix, FEATURE_NAMES, PickObservation
    from scoring import league
    ri = FEATURE_NAMES.index("reach")

    def reach_at(ranks, pick):
        pool = pd.DataFrame({
            "norm": [f"p{r}" for r in ranks], "position": ["RB"] * len(ranks),
            "adp_rank": ranks, "market_rank": ranks, "hype": [0.0] * len(ranks),
            "age": [np.nan] * len(ranks), "ppg_std": [0.0] * len(ranks),
            "missed_rate": [0.0] * len(ranks),
            "no_track_record": [True] * len(ranks), "trend": [0.0] * len(ranks)})
        obs = PickObservation(season=2025, overall_pick=pick, manager="m",
                              chosen=0, pool=pool, roster={}, recent=[])
        return feature_matrix(obs, league.default_settings())[:, ri]

    top = reach_at([1, 5], 1)
    deep = reach_at([100, 140], 1)
    assert (top[1] - top[0]) > (deep[1] - deep[0])


def test_new_features_are_present_and_neutral_without_history():
    import numpy as np
    from scoring.draft_model import feature_matrix, FEATURE_NAMES, PickObservation
    from scoring import league
    assert FEATURE_NAMES[-5:] == ["age", "volatility", "no_track_record",
                                  "hype", "trend"]
    pool = pd.DataFrame({
        "norm": ["a", "b"], "position": ["RB", "WR"], "adp_rank": [1.0, 2.0],
        "market_rank": [1.0, 2.0], "hype": [np.nan, np.nan],
        "age": [np.nan, np.nan], "ppg_std": [0.0, 0.0], "missed_rate": [0.0, 0.0],
        "no_track_record": [True, True], "trend": [0.0, 0.0]})
    obs = PickObservation(season=2025, overall_pick=1, manager="m", chosen=0,
                          pool=pool, roster={}, recent=[])
    X = feature_matrix(obs, league.default_settings())
    assert X.shape == (2, len(FEATURE_NAMES))
    for name in ("age", "volatility", "hype", "trend"):
        assert (X[:, FEATURE_NAMES.index(name)] == 0.0).all()
    assert (X[:, FEATURE_NAMES.index("no_track_record")] == 1.0).all()
    assert np.isfinite(X).all()


def test_age_is_centred_within_position():
    import numpy as np
    from scoring.draft_model import feature_matrix, FEATURE_NAMES, PickObservation
    from scoring import league
    pool = pd.DataFrame({
        "norm": ["a", "b", "c"], "position": ["RB", "RB", "WR"],
        "adp_rank": [1.0, 2.0, 3.0], "market_rank": [1.0, 2.0, 3.0],
        "hype": [0.0, 0.0, 0.0], "age": [24.0, 28.0, 30.0],
        "ppg_std": [0.0, 0.0, 0.0], "missed_rate": [0.0, 0.0, 0.0],
        "no_track_record": [False, False, False], "trend": [0.0, 0.0, 0.0]})
    obs = PickObservation(season=2025, overall_pick=1, manager="m", chosen=0,
                          pool=pool, roster={}, recent=[])
    age = feature_matrix(obs, league.default_settings())[:, FEATURE_NAMES.index("age")]
    assert age[0] == pytest.approx(-2.0)   # RB mean 26
    assert age[1] == pytest.approx(2.0)
    assert age[2] == pytest.approx(0.0)    # lone WR is its own mean
```

Append to `tests/test_draft_sim.py` — extend the existing parity fixture so every new column is non-trivially different across players, then assert full-matrix equality:

```python
def test_live_features_matches_feature_matrix_on_the_new_columns():
    """The two implementations of one feature definition must agree. A
    mismatch silently invalidates every simulation while everything runs."""
    import numpy as np
    from scoring.draft_model import FEATURE_NAMES, PickObservation, feature_matrix
    from scoring.draft_sim import SimPool, _live_features
    import pandas as pd

    pool_df = pd.DataFrame({
        "norm": ["a", "b", "c", "d"],
        "position": ["RB", "WR", "QB", "TE"],
        "adp_rank": [1.0, 12.0, 40.0, 90.0],
        "market_rank": [2.0, 10.0, 55.0, 80.0],
        "hype": [-4.0, 7.0, np.nan, 25.0],
        "age": [23.0, 29.0, 34.0, np.nan],
        "ppg_std": [1.5, 6.0, 0.0, 3.25],
        "missed_rate": [0.0, 0.35, 0.1, 0.0],
        "no_track_record": [False, False, True, False],
        "trend": [2.5, -1.75, 0.0, 0.4]})
    obs = PickObservation(season=2026, overall_pick=9, manager="m", chosen=0,
                          pool=pool_df, roster={"RB": 1}, recent=["WR", "RB"])
    sim = SimPool(
        player_id=np.array(["a", "b", "c", "d"]),
        norm=pool_df["norm"].to_numpy(), position=pool_df["position"].to_numpy(),
        adp_rank=pool_df["adp_rank"].to_numpy(),
        points=np.array([300.0, 250.0, 380.0, 190.0]),
        availability=np.full(4, 90.0), vor=np.array([150.0, 120.0, 80.0, 60.0]),
        market_rank=pool_df["market_rank"].to_numpy(),
        age=pool_df["age"].to_numpy(), ppg_std=pool_df["ppg_std"].to_numpy(),
        missed_rate=pool_df["missed_rate"].to_numpy(),
        no_track_record=pool_df["no_track_record"].to_numpy(),
        hype=pool_df["hype"].to_numpy(), trend=pool_df["trend"].to_numpy())
    available = np.arange(4)
    np.testing.assert_allclose(
        _live_features(sim, available, 9, {"RB": 1}, ["WR", "RB"], S),
        feature_matrix(obs, S))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_model.py -k log_rank tests/test_draft_sim.py -k new_columns -v`
Expected: FAIL — `reach` still linear, and `SimPool` has no `market_rank`.

- [ ] **Step 3: Write minimal implementation**

In `scoring/draft_model.py`:

```python
_NEW_FEATURES = ["age", "volatility", "no_track_record", "hype", "trend"]
FEATURE_NAMES = (["reach", "fall"]
                 + [f"pos_{p}" for p in _POSITION_DUMMIES]
                 + ["qb_early", "te_early", "need", "run"]
                 + _NEW_FEATURES)

# Divisors that put each new feature on roughly the same scale as the
# others, so no coefficient has to be tiny or huge to matter. They are not
# fitted -- changing one just rescales its coefficient -- but keeping the
# columns comparable makes the ridge penalty treat them even-handedly.
VOLATILITY_SCALE = 10.0
HYPE_SCALE = 50.0


def _log_rank_features(market_rank, pick_no):
    """Rounds-of-reach on a log axis.

    Linear rank made the gap from rank 1 to 5 (0.5 units) ten times smaller
    than the gap from 100 to 140 (5.0), so the fit spent its range on
    deep-bench noise and left the entire top of the board within ~2x of
    itself in probability -- a coin flip for the first pick of the draft.
    Log rank inverts that, matching how drafts actually behave. log1p so
    rank 1 and pick 1 are finite; no division by `teams`, which was a
    rounds-based scaling that means nothing on a log axis.
    """
    delta = np.log1p(market_rank) - np.log1p(pick_no)
    return np.maximum(0.0, delta), np.maximum(0.0, -delta)


def _centre_within_position(values, positions):
    """Age relative to typical for the position, so the coefficient reads as
    'younger than his peers' rather than tracking that tight ends last
    longer than running backs. Unknown ages are neutral, not young."""
    out = np.zeros(len(values), dtype=float)
    values = np.asarray(values, dtype=float)
    for pos in set(positions):
        mask = positions == pos
        known = mask & np.isfinite(values)
        if known.any():
            out[known] = values[known] - values[known].mean()
    return out
```

Then in `feature_matrix`, replace the reach/fall block and append the five new columns:

```python
    ranks = pool["market_rank"].to_numpy(dtype=float)
    reach, fall = _log_rank_features(ranks, obs.overall_pick)
    ...  # position dummies, qb_early, te_early, need, run unchanged
    columns.append(_centre_within_position(pool["age"].to_numpy(), positions))
    columns.append(pool["ppg_std"].to_numpy(dtype=float) / VOLATILITY_SCALE
                   + pool["missed_rate"].to_numpy(dtype=float))
    columns.append(pool["no_track_record"].to_numpy().astype(float))
    hype = pool["hype"].to_numpy(dtype=float)
    columns.append(np.nan_to_num(hype, nan=0.0) / HYPE_SCALE)
    columns.append(pool["trend"].to_numpy(dtype=float))
```

In `scoring/draft_sim.py`, add the seven fields to `SimPool`, populate them in `build_pool` from `attributes_as_of(conn, settings.season or CURRENT_SEASON)` joined on `adp_match_key`, and mirror the same six lines in `_live_features` using the shared helpers imported from `draft_model`:

```python
from scoring.draft_model import (EARLY_ROUNDS, FEATURE_NAMES, HYPE_SCALE,
                                 RUN_WINDOW, VOLATILITY_SCALE,
                                 _centre_within_position, _log_rank_features)
```

Importing the helpers rather than retyping them is what makes the parity test a check on the wiring rather than on two hand-copied formulas.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_draft_model.py tests/test_draft_sim.py -q`
Expected: all passed, including the pre-existing parity tests

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add scoring/draft_model.py scoring/draft_sim.py tests/
git commit -m "feat: log-rank features plus age, volatility, hype and trend"
```

---

### Task 5: League positional bias

**Files:**
- Modify: `scoring/draft_model.py`
- Test: `tests/test_draft_model.py`

**Interfaces:**
- Consumes: enriched observations from Task 3
- Produces: `scoring.draft_model.positional_bias(conn) -> pd.DataFrame` with columns `position, round_bucket, mean_gap, n` — the average of `market_rank - overall_pick` for picks of that position in that bucket, over all managers. Buckets are `early` (rounds 1-3), `mid` (4-8), `late` (9+), matching `EARLY_ROUNDS`.

This is a reported measurement and a pooled adjustment, not eight per-manager coefficients: a shared effect fitted once rather than eight times against ~105 picks each.

- [ ] **Step 1: Write the failing test**

```python
def test_positional_bias_measures_how_early_a_league_takes_a_position(tmp_path):
    """If the league takes QBs 20 picks ahead of where the market ranks
    them, that is a fact about the league, not about any one manager."""
    from scoring.draft_model import positional_bias
    conn = _seed_with_espn(tmp_path)
    bias = positional_bias(conn)
    assert list(bias.columns) == ["position", "round_bucket", "mean_gap", "n"]
    assert (bias["n"] > 0).all()
    # Player A: market_rank 2, taken at pick 1 -> gap +1 (taken early).
    rb = bias[(bias.position == "RB") & (bias.round_bucket == "early")]
    assert rb["mean_gap"].iloc[0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_model.py -k positional_bias -v`
Expected: FAIL with `ImportError: cannot import name 'positional_bias'`

- [ ] **Step 3: Write minimal implementation**

```python
def _round_bucket(overall_pick: int, teams: int) -> str:
    round_no = (overall_pick - 1) // max(teams, 1) + 1
    if round_no <= EARLY_ROUNDS:
        return "early"
    return "mid" if round_no <= 8 else "late"


def positional_bias(conn, settings=None) -> pd.DataFrame:
    """How far ahead of the market this league takes each position.

    Positive `mean_gap` means the league drafts that position earlier than
    the market ranks it. Pooled across managers on purpose: a league-wide
    habit forced through eight per-manager coefficients would spend scarce
    data re-learning the same thing eight times.
    """
    settings = settings or league_mod.load(conn)
    observations = build_observations(conn)
    rows = []
    for obs in observations:
        chosen = obs.pool.iloc[obs.chosen]
        rows.append({"position": chosen["position"],
                     "round_bucket": _round_bucket(obs.overall_pick, settings.teams),
                     "gap": float(chosen["market_rank"]) - obs.overall_pick})
    if not rows:
        return pd.DataFrame(columns=["position", "round_bucket", "mean_gap", "n"])
    df = pd.DataFrame(rows)
    out = df.groupby(["position", "round_bucket"], as_index=False).agg(
        mean_gap=("gap", "mean"), n=("gap", "size"))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_draft_model.py -k positional_bias -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

```bash
.venv/bin/pytest -q
git add scoring/draft_model.py tests/test_draft_model.py
git commit -m "feat: measure the league's positional bias against the market"
```

---

### Task 6: The validation harness

**Files:**
- Modify: `scoring/draft_model.py` (`backtest`), `pipeline/fit_managers.py`
- Test: `tests/test_draft_model.py`

**Interfaces:**
- Consumes: `prepare`, `fit`, `_softmax`, `FEATURE_NAMES`
- Produces:
  - `scoring.draft_model.backtest(conn, settings=None, features=None) -> dict` — now leave-one-season-out over every season, with keys `seasons, top1, top5, logloss, adp_top1, adp_logloss, beats_adp, by_round`
    - `by_round` is a list of `{round_bucket, top1, top5, n}`
    - `features`, when given, is a list of `FEATURE_NAMES` entries to restrict the fit to — this is what makes ablation possible
  - `scoring.draft_model.ablation(conn, settings=None) -> pd.DataFrame` with columns `dropped, top1, top5, delta_top1` — the full model plus one row per new feature removed

- [ ] **Step 1: Write the failing test**

```python
def test_backtest_rotates_through_every_season(tmp_path):
    from scoring.draft_model import backtest
    report = backtest(_seed_many(tmp_path, seasons=(2023, 2024, 2025)))
    assert sorted(report["seasons"]) == [2023, 2024, 2025]
    assert 0.0 <= report["top1"] <= 1.0
    assert 0.0 <= report["top5"] <= 1.0


def test_backtest_reports_accuracy_by_round(tmp_path):
    from scoring.draft_model import backtest
    report = backtest(_seed_many(tmp_path, seasons=(2023, 2024, 2025)))
    buckets = {r["round_bucket"] for r in report["by_round"]}
    assert buckets <= {"early", "mid", "late"} and buckets
    assert sum(r["n"] for r in report["by_round"]) > 0


def test_backtest_can_restrict_to_a_feature_subset(tmp_path):
    from scoring.draft_model import backtest, FEATURE_NAMES
    conn = _seed_many(tmp_path, seasons=(2023, 2024, 2025))
    subset = [f for f in FEATURE_NAMES if f != "trend"]
    full = backtest(conn)
    cut = backtest(conn, features=subset)
    assert isinstance(cut["top1"], float)
    assert full["top1"] >= 0.0 and cut["top1"] >= 0.0


def test_ablation_has_a_row_per_new_feature(tmp_path):
    from scoring.draft_model import ablation
    table = ablation(_seed_many(tmp_path, seasons=(2023, 2024, 2025)))
    assert list(table.columns) == ["dropped", "top1", "top5", "delta_top1"]
    assert "none" in table["dropped"].tolist()
    for f in ("age", "volatility", "hype", "trend"):
        assert f in table["dropped"].tolist()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_model.py -k "backtest_rotates or by_round or ablation" -v`
Expected: FAIL — `KeyError: 'seasons'`, then `ImportError` for `ablation`.

- [ ] **Step 3: Write minimal implementation**

Rewrite `backtest` to loop over every season as holdout, accumulating hits and log-loss across all of them, and to accept a `features` list that masks columns before fitting. Add:

```python
def ablation(conn, settings=None) -> pd.DataFrame:
    """Each new feature's contribution, measured rather than argued.

    At roughly 105 picks per manager a feature that does not pay for itself
    is worse than absent: it fits noise and drags every other coefficient
    with it. This table is what decides which of them ship.
    """
    full = backtest(conn, settings)
    rows = [{"dropped": "none", "top1": full["top1"], "top5": full["top5"],
             "delta_top1": 0.0}]
    for feature in _NEW_FEATURES:
        keep = [f for f in FEATURE_NAMES if f != feature]
        cut = backtest(conn, settings, features=keep)
        rows.append({"dropped": feature, "top1": cut["top1"],
                     "top5": cut["top5"],
                     "delta_top1": full["top1"] - cut["top1"]})
    return pd.DataFrame(rows)
```

In `pipeline/fit_managers.py`, print the per-round table, the ablation table, and the positional bias:

```python
    print(f"Backtest, leave-one-season-out over {report['seasons']}:")
    print(f"  overall: top-1 {report['top1']:.0%}, top-5 {report['top5']:.0%}, "
          f"log-loss {report['logloss']:.3f} (ADP baseline {report['adp_logloss']:.3f})")
    for r in report["by_round"]:
        print(f"  {r['round_bucket']:>5}: top-1 {r['top1']:.0%}, "
              f"top-5 {r['top5']:.0%}  (n={r['n']})")
    print("\nFeature ablation (delta_top1 > 0 means the feature earns its place):")
    print(ablation(conn).to_string(index=False))
    print("\nLeague positional bias (positive = drafted ahead of the market):")
    print(positional_bias(conn).to_string(index=False))
```

- [ ] **Step 4: Run tests, full suite, commit**

```bash
.venv/bin/pytest tests/test_draft_model.py -q
.venv/bin/pytest -q
git add scoring/draft_model.py pipeline/fit_managers.py tests/test_draft_model.py
git commit -m "feat: leave-one-season-out backtest with per-round and ablation tables"
```

---

### Task 7: The dedupe becomes a draft-order greedy

**Files:**
- Modify: `scoring/draft_sim.py` (`_assign_primaries`, remove `ASSIGNMENT_MISS_COST`)
- Test: `tests/test_draft_sim.py`

**Interfaces:**
- Produces: `_assign_primaries(counts, n_rollouts) -> dict[int, int]` — same signature, new rule. Picks are processed in ascending `overall_pick`; each takes its highest-count player not already claimed. A pick whose every candidate is claimed keeps its own most likely player, duplicate and all, because inventing a pick the model never made would be worse.
- `scipy.optimize.linear_sum_assignment` is no longer imported by this module.

- [ ] **Step 1: Write the failing test**

```python
def test_assign_primaries_gives_pick_one_its_most_likely_player():
    """The defect this fixes, from a real run: pick 1 showed a 12% player
    while a 16% player sat in the hover, because the global assignment
    'saved' him for pick 13 where he scored 40%."""
    from scoring.draft_sim import _assign_primaries
    counts = {1: {10: 6, 11: 8}, 13: {11: 20, 12: 5}}
    primary = _assign_primaries(counts, n_rollouts=50)
    assert primary[1] == 11          # pick 1 gets its own most likely
    assert primary[13] == 12         # 11 is claimed, 13 takes the next


def test_assign_primaries_processes_picks_in_draft_order():
    from scoring.draft_sim import _assign_primaries
    counts = {5: {1: 10}, 2: {1: 9, 2: 3}, 9: {1: 30, 3: 2}}
    primary = _assign_primaries(counts, n_rollouts=50)
    assert primary[2] == 1           # earliest pick wins the contested player
    assert primary[5] != 1 and primary[9] != 1


def test_assign_primaries_allows_a_duplicate_only_when_forced():
    from scoring.draft_sim import _assign_primaries
    counts = {1: {7: 5}, 2: {7: 5}}   # one candidate, two picks
    primary = _assign_primaries(counts, n_rollouts=10)
    assert primary == {1: 7, 2: 7}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_sim.py -k assign_primaries -v`
Expected: FAIL — the global assignment gives pick 1 to player 10.

- [ ] **Step 3: Write minimal implementation**

```python
def _assign_primaries(counts: dict, n_rollouts: int) -> dict:
    """Each pick's most likely player, walking the board in draft order.

    Was a global assignment minimizing total -log(prob), which maximizes the
    joint likelihood of all 120 cells at once. That is a coherent
    statistical object and the wrong one for a draft board: it would trade
    pick 1 away to improve pick 13, and did -- a real run showed a 12%
    player at pick 1 while the 16% player sat in the hover, because he
    scored 40% at pick 13.

    A board is read top to bottom and the earliest picks are the ones that
    must be right, so earliest pick wins. Not the maximum-likelihood board,
    deliberately. A pick whose every candidate is already claimed keeps its
    own best anyway: a visible repeat beats inventing a pick nobody made.
    """
    primary, claimed = {}, set()
    for overall_pick in sorted(counts):
        cell = counts[overall_pick]
        ranked = sorted(cell, key=lambda idx: (-cell[idx], idx))
        pick = next((idx for idx in ranked if idx not in claimed), ranked[0])
        primary[overall_pick] = pick
        claimed.add(pick)
    return primary
```

Delete `ASSIGNMENT_MISS_COST` and the `linear_sum_assignment` import.

- [ ] **Step 4: Run tests, full suite, commit**

```bash
.venv/bin/pytest tests/test_draft_sim.py -q
.venv/bin/pytest -q
git add scoring/draft_sim.py tests/test_draft_sim.py
git commit -m "fix: the earliest pick gets its most likely player"
```

---

### Task 8: The self-policy values replacement, not raw points

**Files:**
- Modify: `scoring/draft_sim.py` (`_greedy_choice`)
- Test: `tests/test_draft_sim.py`

**Interfaces:**
- Produces: `_greedy_choice` ranks candidates by the increase in `roster_value` computed against replacement-adjusted points, using `settings.replacement_ranks`. `roster_value`'s own definition is unchanged — it remains what the search reports.

- [ ] **Step 1: Write the failing test**

```python
def test_greedy_takes_the_running_back_over_the_higher_scoring_quarterback():
    """The defect this fixes, from a real run: the simulated user opened
    with Josh Allen, market rank 24, at pick 4. A QB outscores every RB in
    raw points, so a one-ply greedy on raw points always takes one early --
    the error value over replacement exists to prevent."""
    import numpy as np
    from scoring.draft_sim import SimPool, _greedy_choice, _roster_cap
    pool = SimPool(
        player_id=np.array(["qb", "rb"]), norm=np.array(["qb", "rb"]),
        position=np.array(["QB", "RB"]), adp_rank=np.array([1.0, 2.0]),
        points=np.array([380.0, 300.0]), availability=np.full(2, 95.0),
        vor=np.array([0.0, 0.0]), market_rank=np.array([1.0, 2.0]),
        age=np.array([28.0, 24.0]), ppg_std=np.zeros(2),
        missed_rate=np.zeros(2), no_track_record=np.array([False, False]),
        hype=np.zeros(2), trend=np.zeros(2))
    roster = {"counts": {}, "indices": []}
    choice = _greedy_choice(pool, np.arange(2), roster, S, _roster_cap(S))
    assert pool.position[choice] == "RB"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_draft_sim.py -k higher_scoring_quarterback -v`
Expected: FAIL — the greedy picks the QB.

- [ ] **Step 3: Write minimal implementation**

Add a replacement-level helper and use it inside `_greedy_choice`:

```python
def _replacement_points(pool, settings) -> dict:
    """Projected points of the last starter-caliber player at each position.

    `LeagueSettings.replacement_ranks` already derives those ranks from the
    league's roster shape, and the board's VOR column already uses them --
    the simulator simply was not. Without this a one-ply greedy compares a
    380-point quarterback against a 300-point running back and takes the
    quarterback, ignoring that the next quarterback available is worth 300
    while the next running back is worth 150.
    """
    ranks = settings.replacement_ranks
    out = {}
    for pos, rank in ranks.items():
        points = np.sort(pool.points[pool.position == pos])[::-1]
        if len(points) == 0:
            out[pos] = 0.0
        else:
            out[pos] = float(points[min(rank, len(points)) - 1])
    return out
```

In `_greedy_choice`, compute `replacement = _replacement_points(pool, settings)` once and subtract it when valuing a candidate, so the marginal gain reflects points above the position's replacement rather than raw points.

- [ ] **Step 4: Run tests, full suite, commit**

```bash
.venv/bin/pytest tests/test_draft_sim.py -q
.venv/bin/pytest -q
git add scoring/draft_sim.py tests/test_draft_sim.py
git commit -m "fix: the in-rollout policy drafts on value over replacement"
```

---

### Task 9: Run it for real and decide what ships

This task produces a decision and a report, not new behavior. It is a task because the spec's central promise — that features are kept only if they earn their place — does not happen unless someone runs the numbers and acts on them.

**Files:**
- Modify: `scoring/draft_model.py` (only if the ablation says to cut a feature)
- Report: `.superpowers/sdd/<workspace>/task-9-report.md`

- [ ] **Step 1: Re-import so the ESPN history exists**

The API server may hold the database lock. If `make espn-import` fails with `IOException: Could not set lock`, report it and stop rather than killing the server.

Run: `make espn-import LEAGUE="https://fantasy.espn.com/football/league?leagueId=53929318"`
Expected: six seasons, and a note naming any season whose ESPN ADP was unusable.

- [ ] **Step 2: Fit and read the report**

Run: `make fit-managers`
Record verbatim: the overall leave-one-season-out top-1 and top-5, the per-round table, the ablation table, and the positional bias table.

- [ ] **Step 3: Compare against the gate**

The model before this work scored **top-1 16%, top-5 53%** on a single 2025 holdout. The gate in the spec is that the reworked model beats those numbers on the six-season rotation.

If it does not, stop and report that. Do not tune features to chase the number — the honest finding is that the league drafts less predictably than it looks, and that is worth saying.

- [ ] **Step 4: Cut every feature that did not earn its place**

For each new feature whose `delta_top1` is at or below zero, remove it from `_NEW_FEATURES` and `FEATURE_NAMES`, delete its column from `feature_matrix` and `_live_features`, and drop the `SimPool` field if nothing else reads it. Re-run the full suite and the parity tests after each removal.

Report which features were cut and their numbers. Cutting two of four is a fine outcome; cutting all four means the log-rank change carried the whole improvement, which is also a finding.

- [ ] **Step 5: Verify the original symptom is gone**

Run a sim and check the predicted board directly:

```bash
make sim SLOT=8 ROLLOUTS=300
```

Then confirm, and report the actual numbers:
- The pick-1 primary is the player ESPN ranks first (Jahmyr Gibbs, ADP 1.7 in 2026), and his probability is meaningfully above the second-place player's rather than within noise of it.
- No cell shows a primary that its own hover reports as more likely elsewhere.
- Your own column does not open with a quarterback.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: cut the features that did not earn their place"
```

---

### Task 10: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document what changed**

Update the "Draft simulation" section to cover:

- The market reference is ESPN's own rankings, because the league drafts on ESPN off ESPN's board. FFC is still imported and still used as a fallback for players ESPN's top-N does not reach.
- `espn_ppr_rank` is the ranking, not `espn_adp`, because ESPN resets the ADP column to a constant in some completed seasons. The import names any season where that happened.
- `reach` and `fall` are on a log scale, with one sentence on why: linear rank made the difference between the first and fifth ranked players smaller than the difference between rank 100 and rank 140.
- Whichever of age, volatility, hype and trend survived Task 9's ablation, with the numbers, and a note that the others were measured and cut.
- The backtest is now leave-one-season-out with a per-round breakdown, and how to read the ablation and positional-bias tables `make fit-managers` prints.
- The grid's cell names now come from a draft-order greedy: the earliest pick gets its most likely player. Replace whatever the README currently says about board-wide deduplication.
- The simulated version of you drafts on value over replacement.

- [ ] **Step 2: Verify and commit**

```bash
.venv/bin/pytest -q && cd web && npm run build
git add README.md
git commit -m "docs: what the recalibrated draft model does differently"
```
