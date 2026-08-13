# Multi-League Foundation — Design

**Goal:** Let the draft helper serve any ESPN user's league, not only the one
whose history is baked into the database today — the prerequisite for a hosted,
paid product.

**Status:** design. Grounded in a schema audit run 2026-08-13, not assumptions —
several earlier tasks this project shipped bugs from specifying columns that did
not exist (`draft_teams.slot` is null in all 48 rows; there is no `league_id`
column anywhere). Every claim below was checked against `data/nfl.duckdb`.

## What the audit found

The data splits cleanly into two kinds, and the split is the whole reason this
is tractable rather than a rewrite.

**Universal — shared across every league, one copy, never per-user:**

| table | rows | what |
|---|---|---|
| `weekly` | 174,373 | nflverse player-week stats |
| `snap_counts` | 253,106 | snap data (feeds O-line, usage) |
| `depth_charts` | 416,885 | depth / role |
| `players` | 25,044 | bio, birth dates |
| `schedules` | 272 | NFL schedule |
| `adp`, `espn_adp`, `fp_ecr`, `cbs_ranks`, `mfl_adp` | ~1,800 | market ranks |
| `historic_adp`, `historic_espn_cs` | ~3,300 | historical market |
| `sleeper_ids` | 1,316 | id crosswalk |

That is ~880,000 rows of shared data. **None of it is duplicated per user.**

**League-specific — small, and the only thing that needs isolating:**

| table | rows (one league) | what |
|---|---|---|
| `league` | 6 | settings per season (`settings_json`) |
| `draft_picks` | 712 | that league's draft history |
| `draft_teams` | 48 | `team_id → manager` per season |
| `draft_order` | 8 | current draft's `slot → manager` |
| `manager_profiles` | 120 | fitted per-manager coefficients |
| `manager_tendencies` | 106 | descriptive per-manager stats |
| `drafted` | live | picks made in the running draft |
| `sim_board`/`sim_results`/`sim_survival` | live | sim outputs |

A whole league is under ~1,000 rows. Isolating this is cheap.

## The two real problems

### 1. Cold start — the hard blocker (verified broken)

A brand-new user's league has no `draft_picks`, so `fit_all` returns an **empty
dict** — no `__pooled__`, no coefficients, nothing. The simulator has no
opponent model at all and cannot run. Confirmed by wiping the history tables and
calling `fit_all`: `[]`.

This is the piece flagged at the very start of the project as the novel, hard
part, and it is real. Without a fix, the product works for exactly one person.

**The fix is a market prior.** Opponents with no fitted history draft to the
blended market board — and we already *measured* that the board predicts real
drafts (top-1 0.2514 on this league's six seasons). A league with zero history
gets a `__pooled__` coefficient vector whose weight sits on the market-following
features (`reach`/`fall`), so simulated opponents take the best available player
by market rank with realistic noise. No history required.

As history accrues (this or future seasons), the fit shrinks from the market
prior toward the league's own tendencies — the same ridge-shrinkage machinery
`select_lambda` already uses, just anchored on the prior instead of on zero.

`league.load` already falls back to `default_settings()` for an unknown league,
so scoring/roster settings are NOT a cold-start blocker — only the opponent
model is.

### 2. Per-user isolation and concurrency

- **Isolation:** the league-specific tables must be scoped per league. Options:
  a `league_id` column on each (smallest change, keeps one database), or a
  per-league attached database. The column approach is preferred — the tables
  are tiny and a single shared connection to the universal data stays simple.
- **The board is universal and cacheable.** `build_board` reads only universal
  tables plus league *settings* (for scoring). Two leagues with the same scoring
  rules share a board entirely; differing rules differ only in the fantasy-point
  valuation, not the underlying stats. `build_board` can be cached by scoring
  ruleset, computed once rather than per user.
- **`fit_all` is the expensive per-league step** (~15s). It runs once per league
  per import, not per connect — cache the fitted coefficients in
  `manager_profiles` (already persisted there) and load them at session build.
- **Concurrency:** drafts cluster into a few August weekends. Each live draft is
  one session + one socket + one cached board. The session already exists; what
  multi-user adds is a session *registry* keyed by user, replacing today's single
  module-level session, and a `drafted` table scoped per user.

## Architecture

```
                shared, read-only during drafts
   ┌─────────────────────────────────────────────┐
   │ universal DB: weekly, players, market, ...   │
   └─────────────────────────────────────────────┘
                        │ build_board (cached by ruleset)
                        ▼
   per user:  league import → fit_all → manager coeffs  (once, at import)
                        │                     │ or, no history:
                        │                     └→ cold-start market prior
                        ▼
   live draft:  DraftSession(user) ── socket ── drafted(user) ── search_pick
                        ▲
             session registry, keyed by user
```

## Build order (each slice ships working software)

1. **Cold-start prior.** A `cold_start_fits(settings)` returning a market-
   following `__pooled__`, used by `fit_all` when there are no observations.
   Tested in isolation: an opponent under the prior picks the better market rank
   more often than chance. **This is the first slice — it unblocks strangers and
   risks nothing already working.**
2. **League-scoped data.** `league_id` on the league-specific tables, defaulting
   to the existing single league so current code and all 453 tests keep passing.
3. **Per-user league import.** Reuse `EspnClient` / `import_seasons` to pull a
   new user's league history on first connect, keyed by their `league_id`.
4. **Session registry.** Replace the single module-level live session with a
   per-user map; scope `drafted` per user.
5. **Board cache by ruleset.** Compute `build_board` once per distinct scoring
   configuration rather than per user.

Slices 1-2 are safe to build now behind the working single-user path. 3-5 are the
larger lift and want their own plan.

## Out of scope

Payments, accounts, the extension store listing, and deployment. This is the
data and model foundation; those sit on top of it.

## Testing

Every slice ships with tests driven by synthetic fixtures via `pipeline.db`.
No test opens `data/nfl.duckdb`. The cold-start slice specifically verifies the
prior produces market-following behaviour by mutation — break the prior, watch
the opponent stop preferring the better rank.
