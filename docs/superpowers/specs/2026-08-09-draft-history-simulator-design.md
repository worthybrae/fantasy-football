# Draft History Import, Manager Modeling, and Draft Simulation

Date: 2026-08-09

## Goal

Import this league's own ESPN draft history, fit a per-manager pick-prediction
model to it, and use that model to simulate the upcoming draft so the board can
answer two questions at every pick:

1. Who is likely to still be available when it is my turn again?
2. Which player available now leads to the best final roster?

This is a pre-draft study, run before draft night. It is not a live draft-room
client, though the design keeps the same views usable mid-draft.

## Context

The repo already has a working draft board: `pipeline/` pulls nflverse stats and
five market-consensus sources into DuckDB, `scoring/` computes factors, a
composite score, VOR, and tiers, `api/` serves it, and `web/` renders a sortable
React board with drafted-player tracking.

Three things in that stack are hardcoded to one league shape and will become
derived: `LEAGUE_TEAMS` and `REPLACEMENT_RANK` in `scoring/config.py`, and the
PPR scoring rules in `scoring/ppr.py`.

ESPN season projections (`espn_adp.espn_proj`) are already ingested and used in
player profiles. They become the basis for the simulator's objective function.

## Decisions

| Question | Decision |
|---|---|
| Primary use | Pre-draft study: enter the draft order, simulate my slot |
| Opponent model | Per-manager tendencies learned from league draft history |
| History depth | One ESPN league ID, 5+ seasons, walked back through `leagueHistory` |
| Objective | Projected points of my best legal starting lineup |
| Search | Optimize each of my picks by rollout, not a fixed strategy grid |
| ESPN auth | Playwright: one manual login, saved storage state, headless after |
| Draft format | Plain snake, no keepers, no auction |
| Surface | Integrated into the existing board, not a separate page |
| Model family | Conditional logit per manager, ridge-shrunk toward a pooled fit |

## Architecture

Four new Python modules and one new frontend rail:

```
pipeline/espn_league.py    ESPN client (Playwright auth) + parsers
scoring/league.py          derived league structure, with config.py fallback
scoring/draft_model.py     conditional-logit fit + manager profiles
scoring/draft_sim.py       rollouts, roster valuation, per-pick search
web/src/components/DraftRail.tsx   order editor + manager cards
```

Data flows in one direction: import writes raw tables, model fitting reads them
and writes coefficients, the simulator reads coefficients plus the board and
writes results, and the API serves results the board already knows how to
render.

---

## Part 1: ESPN import

### Authentication

`pipeline/espn_league.py` uses Playwright against ESPN's read API.

1. Look for `data/espn_state.json`. The file is gitignored.
2. If it is missing, or a request returns 401 or redirects to login, launch a
   visible Chromium window at the ESPN login page and print a waiting message.
   Poll until the league URL loads authenticated, save the storage state, close
   the window, and continue the import.
3. Every later run is headless and reuses the saved state.

No ESPN password is ever stored or scripted. Disney SSO handles 2FA and bot
checks in the visible window, which is exactly where a human can answer them.

`playwright` is added to `requirements.txt`. Setup requires
`playwright install chromium`, documented in the README.

### Endpoints

Current season:

```
/apis/v3/games/ffl/seasons/{year}/segments/0/leagues/{id}
  ?view=mDraftDetail&view=mTeam&view=mSettings
```

Prior seasons:

```
/apis/v3/games/ffl/leagueHistory/{id}
  ?seasonId={year}&view=mDraftDetail&view=mTeam&view=mSettings
```

Player names for a given season come from that season's own directory,
`/apis/v3/games/ffl/seasons/{year}/players?view=players_wl`. The current-season
feed cannot resolve players who have since retired, which is most of a
five-year-old draft.

The season walk starts at `CURRENT_SEASON - 1` and steps backward until two
consecutive seasons 404.

### Tables

`draft_picks`

| column | notes |
|---|---|
| season | |
| overall_pick, round, round_pick | from `overallPickNumber`, `roundId`, `roundPickNumber` |
| team_id | ESPN team id |
| espn_player_id, player_name, position, nfl_team | |
| keeper | recorded even though the league does not use keepers, so a format change surfaces as data rather than as silently wrong model input |

`draft_teams`: season, team_id, manager, slot.

`historic_adp`: season, adp_name, position, adp_rank. Fetched via the existing
`sources.fetch_adp(year)`, which is already parameterized by year. This is what
makes "reach" measurable — a pick only means something relative to where the
market had that player *that* year.

### Validation

The importer prints a summary and does not silently proceed past a problem:

- seasons found, and draft type per season (hard failure on anything but snake)
- picks per season against expected `teams x rounds`
- manager continuity across seasons
- ADP match rate per season, using `board._norm_name` for name normalization

Picks that do not match an ADP row are stored but excluded from model fitting,
and the excluded count is reported per season. A season matching at 40% is a
data bug, and the import should say so rather than produce a confident model.

Entry point: `make espn-import LEAGUE=<url-or-id>`.

---

## Part 2: Derived league structure

The same `mSettings` view populates a `league` table, one row per season.

| ESPN field | derives |
|---|---|
| `settings.size` | league team count |
| `rosterSettings.lineupSlotCounts` | starters by position, FLEX count, bench and IR depth, total draft rounds |
| `scoringSettings.scoringItems` | the scoring formula, as statId to points |
| `draftSettings.type` | snake versus auction |
| `draftSettings.pickOrder` | this year's draft order, once ESPN publishes it |

### Scoring becomes league-driven

`scoring/ppr.py` gains a rules argument. ESPN statIds map to nflverse weekly
columns (3 to `passing_yards`, 53 to `receptions`, 42 to `receiving_yards`, and
so on). Any scoring item that cannot be mapped is printed at import time, listed
by name, rather than dropped. A league running TE premium or half-PPR gets a
correct board where it previously got a silently wrong one.

### Replacement rank becomes computed

`REPLACEMENT_RANK` is derived as

```
teams * starters_at_pos + round(teams * flex_slots * flex_share_pos)
```

plus a one-player buffer for positions that no flex slot accepts (QB, K, DST).

Flex shares stay a config constant, defaulting to RB 0.375 / WR 0.5 / TE 0.125.
For this league — 8 teams, 2 FLEX — that reproduces today's values exactly:
QB 8 + 1 = 9, RB 16 + 6 = 22, WR 16 + 8 = 24, TE 8 + 2 = 10, K and DST 9.
League size or roster shape changes now flow through to VOR without editing
code.

### Fallback

Derived settings apply only when the `league` table exists. With no import the
board uses today's `scoring/config.py` constants and produces a byte-identical
result, so existing tests are unaffected. The board header reports which mode is
active.

---

## Part 3: Opponent model

`scoring/draft_model.py`.

### Formulation

Each historical pick is one choice from the set of players available at that
moment, which is reconstructable from pick order plus that season's ADP pool.
Probability that manager *m* takes player *p* is proportional to
`exp(beta_m . x(p, state))`.

Features:

| feature | captures |
|---|---|
| `reach` = max(0, adp_rank - pick_no) / teams | rounds of reach required. Strongly negative for ADP-disciplined managers, near zero for gunslingers |
| `fall` = max(0, pick_no - adp_rank) / teams | whether the manager chases players who slid |
| position dummies, one dropped | baseline positional taste |
| QB x early, TE x early | the two positions where managers actually differ in early rounds |
| `need` | roster is still short of a starter at that position |
| `run` | same-position picks in the last five, scaled |

That is roughly 12 parameters per manager against roughly 105 observed picks
(7 seasons x 15 rounds, where 15 is 10 starters plus 5 bench and is itself
derived from `lineupSlotCounts`). Roster caps, such as no fourth
quarterback, are applied as a hard mask rather than learned. The data cannot
support learning them, and getting them wrong would distort the coefficients
that matter.

### Fitting

Two stages:

1. Pool all managers and fit `beta_pool` by maximum likelihood over roughly 840
   picks.
2. Per manager, maximize `log-likelihood - lambda * ||beta_m - beta_pool||^2`,
   with lambda chosen by leave-one-season-out cross-validation on held-out
   log-likelihood.

The negative log-likelihood of a conditional logit with a ridge penalty is
convex, so L-BFGS-B reaches the global optimum. This is the reason for adding
`scipy` rather than hand-rolling an optimizer.

### Overfitting guard

Every manager profile reports held-out log-likelihood for three models: the
manager's own fit, the pooled fit, and a pure-ADP baseline. If a manager's
personal model does not beat pooled out of sample, the simulator uses pooled for
that manager and the profile says so in plain words. With roughly 105 picks per
manager this will genuinely happen for some of them.

### Backtest

Hold out the most recent season entirely. Report top-1 and top-5 pick accuracy
and log-loss against the ADP-only baseline. This number goes in the import
report. If the model does not beat ADP-only, that is the finding, and the board
should not present simulator output as authoritative.

---

## Part 4: Simulator

`scoring/draft_sim.py`.

### Roster value

The objective is projected points of the best legal starting lineup.

Projection ladder per player: `espn_proj`, then recency-weighted PPG times 17,
then position replacement level.

Lineup fill is greedy: the best player at each dedicated slot, then the best
remaining RB/WR/TE into each FLEX. Slot eligibility here is nested, so greedy is
optimal and no linear program is needed.

### Bench valuation

Pure starting-lineup value makes every pick after roughly round 9 worth zero,
which would make the simulator's late rounds random noise. Roster value
therefore adds an insurance term: each starter's expected missed games, taken
from the existing durability factor, multiplied by the drop-off to the best
eligible bench player at that position. One extra term, no new data, and late
rounds stay meaningful.

### Rollouts and search

A rollout walks the snake from the current state to the end of the draft.
Opponents sample from their fitted model over available players. My picks use a
greedy marginal-value policy.

At each of my real picks, the search takes roughly 12 candidates — the union of
the top available by VOR and the top by market rank — forces each one, runs N
rollouts, and scores by end-of-draft roster value. Results report mean and
standard error, so two candidates that are actually tied read as tied.

Common random numbers are used across candidates: rollout *i* uses the same seed
for every candidate, so all candidates face identical opponent behavior. This
cuts the rollouts needed for a given resolution by roughly an order of
magnitude.

Cost at a single slot is roughly 15 picks x 12 candidates x 300 rollouts, about
40 seconds single-core and under 10 seconds multiprocessed. A `--rollouts` flag
trades runtime for resolution.

### Result tables

- `sim_results`: run_id, created_at, my_slot, pick_no, player_id, ev, se, rank
- `sim_survival`: run_id, player_id, availability probability by upcoming pick
- `manager_profiles`: manager, coefficient name, value, pooled value, n_picks,
  held-out log-likelihood gain

Results are persisted rather than recomputed per request.

---

## Part 5: API

| endpoint | purpose |
|---|---|
| `GET /api/league` | derived structure, seasons imported, unmapped scoring rules |
| `GET /api/managers` | profiles: coefficients, league baseline, n_picks, held-out gain |
| `GET /api/draft-order` | slot to manager, seeded from ESPN `pickOrder` |
| `PUT /api/draft-order` | override the order |
| `POST /api/sim` | start a run in a background thread, return `run_id` |
| `GET /api/sim/{run_id}` | status and progress, then results |

`/api/players` gains three nullable columns merged from the latest completed
sim: `avail_pct`, `ev`, `ev_se`. They are null until a sim has run, so the board
degrades to exactly what it is today.

A sim takes 10 to 40 seconds, so `POST /api/sim` returns immediately and the
frontend polls. It does not block a request for the duration of the run.

The `drafted` table gains a `pick_no` column. Pick attribution is then implied
by insertion order combined with the known draft order, which is what lets the
same rail and columns keep working if players are toggled off as a real draft
runs.

---

## Part 6: Board integration

### New columns

Both are blank when no sim has run.

- **Avail%** — probability the player is still available at my next pick. This
  is the column that answers "who can I wait on".
- **ΔEV** — expected end-of-draft starting-lineup points relative to the best
  available option. The top candidate renders as a dash and others render
  negative. Only the evaluated candidates get a value; blank is itself the
  signal that a player is not in contention.

### Right rail, collapsible

**Draft order editor** at the top: one row per slot, manager selectable per
slot, "You" marked. Seeded from ESPN's published order and editable, so "what if
I am at slot 3" is answerable. Rollout count input and a Run button.

**Manager cards** below, in slot order. Each card shows:

- a one-line read generated from the fitted coefficients, for example
  "reaches ~1.4 rounds early, early-TE guy, fades positional runs"
- the three strongest coefficients as bars against the league average
- pick count
- a confidence chip: "personal model" or "league average, not enough signal"

No card implies more certainty than the held-out likelihood supports.

**Header strip**: current slot, next pick in round-and-pick notation, and the
age of the latest sim.

---

## Testing

All Python tests run offline against fixture JSON. No test hits ESPN.

- ESPN payload parsers: draft detail, teams, settings, player directory
- Settings derivation: statId to scoring rules, slot counts to replacement ranks
- Name matching and ADP join rate against a fixture season
- Model recovers a known `beta` from synthetic managers generated by that same
  `beta`
- Shrinkage behaves: a manager with two picks lands near pooled, a manager with
  many distinctive picks does not
- Lineup optimizer returns the true optimum on small hand-checked rosters
- Simulator is deterministic under a fixed seed
- Backtest harness runs end to end on fixture data
- API: new endpoints, and `/api/players` unchanged when no sim exists

Frontend tests follow the existing component patterns.

## Out of scope

- Auction drafts
- Keeper leagues
- In-draft trades
- Live polling of an active ESPN draft room
- Per-manager player-level preferences, such as favoring a specific NFL team.
  Roughly 105 picks per manager cannot support it.

## Risks

**Sample size.** Roughly 105 picks per manager is thin for 12 parameters. The
shrinkage, the held-out comparison, and the backtest exist specifically to keep
this honest. The realistic outcome is that some managers get a personal model
and some fall back to pooled.

**ESPN API drift.** The read API is undocumented and has changed shape before,
as the existing `sources.py` comments record. Parsers raise on empty results
rather than returning empty frames, matching how the other sources behave.

**Auth expiry.** Saved storage state expires. The importer detects it and
reopens the login window rather than failing.

**Greedy rollout policy.** Opponent and self policies inside a rollout are
one-ply greedy, not optimal. This is the standard approach and is adequate for
ranking candidates at the current pick, but the reported EV is a lower bound on
what perfect play would achieve.
