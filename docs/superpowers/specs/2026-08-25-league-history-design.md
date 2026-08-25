# League history and manager profiles

**Date:** 2026-08-25
**Status:** approved in conversation, awaiting spec review
**Builds on:** `2026-08-25-league-report-design.md`, in progress on the
`worktree-league-report` branch (per-league database files, cookie-backed
history import, `league_standings`, draft grades, team profiles). This spec
adds the rest of a league's history to that import and turns it into a
profile of every manager in the league. It is implemented after that branch
merges, against the code it lands.

## The problem

The league page (`/league/:leagueId`, `web/src/pages/LeaguePage.tsx`) is a
shell: the draft's date, your team, a promise that more will live there.
What a league actually is, to the people in it, is eight managers and six
years of arguments -- who always wins, who never sets a lineup, who trades
with whom, who got lucky in 2023. ESPN holds every fact behind those
arguments and shows none of them beyond this season's standings.

The product is the profile of each manager: what they do, measured over
every season the league has played, with the draft (already modelled) as
one chapter among finishes, playoffs, waivers, trades and lineups.

Explicitly not in this version: waiver-wire suggestions, trade
suggestions, anything forward-looking. Historical data only.

## What ESPN gives us

Probed 2026-08-25 against league 53929318 with the owner's saved cookies,
every view, seasons 2026 back to 2017. The full note is in the working
memory (`espn-league-history-api-probe`); what the design depends on:

- **Seasons.** The dated path
  `/apis/v3/games/ffl/seasons/{yr}/segments/0/leagues/{id}` answers for
  every season the league has (2020 to 2026 here). `status.previousSeasons`
  on the current season lists the floor. Earlier seasons 404 on both the
  dated and the `leagueHistory` paths.
- **`mTeam`.** `members[]` with SWID, display name, first and last name;
  `teams[]` with `record.overall`, `playoffSeed`, `rankCalculatedFinal`,
  `draftDayProjectedRank`, `waiverRank`, `primaryOwner`, and
  `transactionCounter` (`acquisitions`, `drops`, `trades`, `moveToActive`,
  `moveToIR`, `matchupAcquisitionTotals`). The SWID is stable across
  seasons: the same eight members appear 2021 through 2025 while team
  names change every year.
- **`mSettings`.** Playoff format, matchup periods, waiver type and
  budget, trade deadline and veto votes, roster and scoring settings.
- **`mDraftDetail`.** Every pick with `memberId`, `autoDraftTypeId`,
  `keeper`, `roundId`, `overallPickNumber`. Already imported by
  `import_seasons`.
- **`mMatchupScore`.** `schedule[]` with `matchupPeriodId`,
  `playoffTierType` (`NONE`, `WINNERS_BRACKET`,
  `WINNERS_CONSOLATION_LADDER`, `LOSERS_CONSOLATION_LADDER`), `winner`,
  and per side `teamId`, `totalPoints`, `pointsByScoringPeriod`.
- **`mTransactions2&scoringPeriodId=N`**, N = 1..17. The season's
  transaction log a week at a time (N = 0 and 18 are empty). About 570
  events a season: `DRAFT`, `FREEAGENT`, `WAIVER` (status `EXECUTED`,
  `PENDING`, `CANCELED`, `FAILED_INVALIDPLAYERSOURCE`,
  `FAILED_ROSTERLIMIT`, ...), `ROSTER` and `FUTURE_ROSTER` (lineup
  changes), `TRADE_PROPOSAL`, `TRADE_DECLINE`, `TRADE_ACCEPT`,
  `TRADE_UPHOLD`, `TRADE_VETO`. Each carries `proposedDate`, `teamId`,
  `bidAmount`, `status`, `executionType` and `items[]` with `playerId`,
  `fromTeamId`, `toTeamId`, `fromLineupSlotId`, `toLineupSlotId`. Verified
  for 2025 and 2021.
- **`mRoster&scoringPeriodId=N`.** That week's roster for every team:
  `lineupSlotId`, `acquisitionType`, `acquisitionDate`, and the player's
  `stats[]` rows for the period with `statSourceId` 0 (actual) and 1
  (projected). 600 to 800 KB a week.
- **Dead ends.** `kona_league_communication` (the activity feed) answers
  400 on the league path and 404 at `/communication/` for this league.
  Not needed: the transaction log covers it. `mStandings`, `mSchedule`,
  `mPendingTransactions` and `mPositionalRatings` add nothing the views
  above do not carry.

## Alignment with the league-report branch

Read from the branch at `be62a26`, so the seams below are the ones that
exist rather than the ones the other spec described:

- `pipeline/league_history.py: import_history(league_id, cookies, fetch=None,
  current_season, universal_path, root, adp_fetch)` opens the league's file
  through `provision_league`, runs `import_seasons` with `json_fetch(raw,
  cookies)`, writes `historic_adp`, and records `meta.source = "league"`.
  `is_fresh(conn)` reads that row (seven days). `spawn_import_if_stale`
  runs it on a thread from the live connect path. This spec's week walk is
  a second step inside `import_history`, after `import_seasons`, on the
  same connection and the same `json_fetch`; the summary it returns gains
  the activity counts.
- `pipeline/espn_league.py: parse_standings(payload, season)` writes
  `league_standings` with `manager` as the member's display name (falling
  back to the team name), the same resolution `parse_draft_teams` uses.
  This spec does not change that table. The SWID lives in
  `league_members`, which joins to standings on `(season, team_id)`, and
  the per-season activity counters go there too rather than widening a
  table another branch owns.
- `scoring/league_report.py: team_profiles(conn)` returns one record per
  `manager` (display name) with `seasons`, `titles`, `playoffs`, `win_pct`,
  `avg_finish`, `ppg`, `mean_value`, `steal_rate`, `reach_rate`,
  `first_pick_positions`, `career_best`, `career_worst`. The draft chapter
  of a manager profile is that record, looked up by the display name
  `league_members` carries for the member; nothing about the draft is
  recomputed here.
- `api/reports.py` is not written yet (task 6 of that plan). The
  ownership check both APIs need -- the request's custody session holds a
  league list containing this id -- is written once, in
  `api/league_access.py: owned_league(request, store, league_id) ->
  Session`, and the reports routes are expected to call it when they land.
- The player profile (`GET /api/players/{id}/profile`, the popup in
  `web/src/components/profile/`) is where a player name goes when clicked.
  A manager's picks, trades and lineups name players; every one opens that
  popup, so the league pages reuse the profile rather than describing a
  player themselves.

## Decisions made with the owner

- **Scope of v1:** everything above, including weekly lineups. The owner
  chose the full slice over a lighter one.
- **Import trigger:** on the first visit to a league's page, with the
  visiting account's ESPN session, showing progress as it runs. Not an
  owner-only command.
- **Audience:** signed-in accounts that hold the league. Nothing public
  in this version; the league-report spec's public link is a separate
  feature.
- **Approach:** extend the league-report import rather than build a second
  one. One walk over ESPN per league, one file per league, one credential
  flow. The alternative -- a separate history service with its own fetch
  and storage -- was rejected as two importers with the same cookies and
  two freshness rules to explain.

## Data

All new tables are per-league (`LEAGUE_TABLES` in `pipeline/db.py`; the
exactness assertion in `tests/test_leagues.py` gains every one). The
manager key everywhere is the ESPN member id (SWID). Team ids and names
are season-scoped and resolve through `league_members`.

### `league_members`

One row per member per season.

```
season, member_id, display_name, first_name, last_name, team_id,
draft_day_rank, acquisitions, drops, trades, lineup_moves, ir_moves
```

`team_id` is the team whose `primaryOwner` is this member; a member who
owns no team that season (a co-owner, or a member who left) has NULL, and
so do the counters. A team with several `owners` still has one primary,
and that is the manager the profile is about. `display_name` is the same
value `parse_standings` stores as `manager`, so the two tables agree on
the name and join on `(season, team_id)`.

The counters are the team's own from the same `mTeam` payload:
`draftDayProjectedRank`, and `transactionCounter`'s `acquisitions`,
`drops`, `trades`, `moveToActive` (as `lineup_moves`) and `moveToIR`.
They are ESPN's per-season totals and stay useful even if a week's
transaction fetch fails.

### `league_standings`

Defined and written by the league-report branch. Read here, not changed.

### `league_matchups`

One row per matchup.

```
season, matchup_period, tier, home_team_id, away_team_id,
home_points, away_points, winner, home_points_by_week_json,
away_points_by_week_json
```

`tier` is `playoffTierType` verbatim. `winner` is `HOME`, `AWAY`, `TIE`
or `UNDECIDED`. A bye (no `away`) is stored with NULL away columns.

### `league_transactions`

One row per transaction. `txn_id` is ESPN's; the same event can appear in
more than one week's answer, so the import de-duplicates on it.

```
season, week, txn_id, team_id, member_id, type, status, execution_type,
bid_amount, proposed_at, items_json
```

`member_id` is the team's primary owner that season, resolved at import
time so profile queries never join on team ids. `items_json` is the
`items[]` array as ESPN sent it.

### `league_lineups`

One row per roster entry per team per week.

```
season, week, team_id, player_id, player_name, position, lineup_slot,
actual_points, projected_points, acquisition_type, injury_status
```

`lineup_slot` is ESPN's `lineupSlotId` (20 is bench, 21 is IR; the
starting slots are the ones in `rosterSettings.lineupSlotCounts`).
`actual_points` is the period's `statSourceId = 0` row's `appliedTotal`,
`projected_points` the `statSourceId = 1` row's; either is NULL when ESPN
sent no such row (a bye week, a player added after the games).

### `league_raw`

The answer ESPN gave, kept so a parser fix is a re-parse and never a
refetch.

```
season, view, week, fetched_at, payload_json
```

`week` is 0 for season-level views. One row per `(season, view, week)`,
replaced on refetch. Six seasons of one league is about 15 MB of JSON;
acceptable, and it is the thing that makes every parser testable against
reality.

## Import: `pipeline/league_history.py`

The league-report branch's `import_history` walks seasons through
`import_seasons`. This spec adds `import_activity(conn, league_id,
fetch_json, seasons, current_season, progress)` and calls it from
`import_history` right after `import_seasons`, with the seasons that walk
found. One function, one connection, one credential; the walk stays one
walk from the caller's point of view.

### The walk

For each season, newest first, ending after two consecutive 404s exactly
as `import_seasons` does:

1. Season-level views, one request each: `mTeam`, `mSettings`,
   `mDraftDetail` (what `import_seasons` already fetches, read from the
   same response), `mMatchupScore`.
2. Week-level views, for `scoringPeriodId` 1 through
   `status.finalScoringPeriod` (17 in every season probed): `mTransactions2`
   and `mRoster`. Two requests a week.

About 40 requests a season; about 240 for a six-season league. Requests
are sequential with a short pause; ESPN has not rate-limited the probe at
that pace and the owner's importer already runs the same way.

### Freshness

- A finished season (`status.currentMatchupPeriod` past the final period,
  or any season before the current one) is fetched once. Its `league_raw`
  rows are the record; the walk skips a season whose season-level rows
  exist.
- The current season is refetched when its newest `league_raw` row is
  older than 24 hours. Only weeks up to `latestScoringPeriod` are
  fetched, and a week whose transactions and roster are already stored
  and whose games are over is not fetched again.
- The whole import is idempotent: tables are rebuilt from `league_raw`
  after every walk, so a partial earlier run leaves nothing inconsistent.

### Parsing

Pure functions in the same module, each `(payload) -> DataFrame`, each
tested against a trimmed fixture taken from the probe:
`parse_members`, `parse_standings_extras`, `parse_matchups`,
`parse_transactions`, `parse_lineups`. Player names and positions for
`league_lineups` come from the roster payload's own `playerPoolEntry`
(`fullName`, `defaultPositionId` through the existing
`ESPN_SLOT_POSITIONS` map in `pipeline/espn_league.py`), so the table does
not depend on the players directory.

### Progress

One import in flight per league. A module-level record, guarded by a
lock, holds the stage list the way `api/live.py` holds the connect's:

```
{"league_id", "started_at", "phase": "running" | "done" | "failed",
 "stages": [{"label": "2023 · week 9 of 17", "done": bool}, ...],
 "seasons_done": [2025, 2024], "error": str | null}
```

The stage list is published up front (one row per season the league
reports in `previousSeasons`, plus the current one; weeks tick inside a
season's row) so the page can draw the whole shape before the first
fetch returns.

### The one-writer rule

DuckDB has one writer per file. The import takes its own connection to
the league's file, writes each parsed table with `write_table` as the
season completes, and retries a locked write once after a short wait, the
same rule the league-report job follows. The live listener for the same
league writes `drafted` to the same file; an import running during a
draft is unlikely (the page triggers it, and it is fresh within a day) but
the retry makes it safe.

## Facts: `scoring/manager_profile.py`

Pure functions over a connection. No network, no model. Every function
returns plain dicts and lists so the API serialises them as they are.

`profile(conn, member_id) -> dict` and `league_overview(conn) -> dict`
are the two entry points. Every fact carries its sample size (`n`) so the
page can decline to make a claim off three data points.

### Per member, per season and all-time

- **Finishes.** From `league_standings` and `league_members`: wins,
  losses, ties, points for and against, playoff seed, final rank,
  `draft_day_rank`, and `outperformance = draft_day_rank - final_rank`
  (positive means the season went better than ESPN's draft-day
  projection). All-time: seasons played, titles (`final_rank = 1`),
  average finish, win percentage, points per game.
- **Playoffs.** From `league_matchups` where `tier` is a bracket or
  ladder: appearances (a season with a `WINNERS_BRACKET` matchup), bracket
  wins and losses, titles confirmed by the bracket, consolation record,
  last-place seasons (`final_rank = teams`).
- **Head-to-head.** Against every other member, over every season both
  were in the league: games, wins, losses, total margin, playoff
  meetings. Built once for the league as a symmetric grid and read from
  there.
- **Luck.** All-play record: for each regular-season week the member's
  points against every other team's points that week; expected wins are
  the all-play win fraction times games played. `luck = actual_wins -
  expected_wins`. Reported per season and summed.
- **Draft.** The league-report spec's draft-habit facts (`mean_value`,
  `steal_rate`, `reach_rate`, `first_pick_positions`, earliest QB and TE
  rounds, career best and worst picks), reused from
  `scoring/league_report.py` rather than recomputed, plus from
  `mDraftDetail`: autodraft rate (`autoDraftTypeId != 0`) and picks per
  position by round.
- **Waivers and free agents.** From `league_transactions`: waiver claims
  a season, claims won (`EXECUTED`), claims lost to a higher priority or
  bid (`FAILED_INVALIDPLAYERSOURCE` where the player went elsewhere the
  same process run), claims canceled, free-agent adds, drops, the busiest
  week, and adds by day of the week. Bid sizing (mean, max, spent) only
  when `acquisitionSettings.isUsingAcquisitionBudget` was true that
  season; the fact is absent otherwise, not zero.
- **Trades.** Proposals made and received (the proposing team is the
  transaction's `teamId`; the counterpart is the other `toTeamId` in
  `items`), accepted, declined, vetoed, upheld; partners by count; and
  trade balance: for each executed trade, the points the players received
  scored for the receiving team over the rest of that season minus the
  points the players sent scored for the other side, from
  `league_lineups`. Reported with the number of trades it rests on.
- **Lineups.** From `league_lineups`, regular-season weeks with actual
  points present: points scored by starters, the optimal lineup's points
  (the best legal assignment of that week's roster to the league's
  starting slots, computed against `rosterSettings.lineupSlotCounts`),
  `bench_points_left = optimal - started` per week and its mean, optimal
  hit rate (weeks where started equals optimal within 0.5), weeks with a
  starter on bye or with `injury_status` `OUT` or `IR`, and lineup moves
  a week from `league_members.lineup_moves`.

### League overview

`league_overview(conn)`: the seasons strip (per season: champion, runner
up, last place, highest scorer, the settings that changed from the year
before) and the member grid (per member: seasons, titles, average finish,
win percentage, and a `defining_line` chosen by rule from the facts --
the single most extreme rank the member holds across the league:
"trades more than anyone", "leaves the most points on the bench", "best
draft-day outperformer"; ties and small samples fall back to the plain
record).

## API: `api/league_history.py`

Registered in `create_app` after `register_seo_routes` and before
`register_spa`, beside the reports routes when they land. Every route
goes through `api/league_access.py: owned_league`, which requires a
custody session whose league list (`espn_drafts.league_entries`, the
source `GET /api/espn/drafts` uses, cached the way that route caches it)
contains the league; otherwise 403. Mock league ids are refused with 400.

- `POST /api/leagues/{id}/history` -- starts the import with the session's
  cookies on a daemon thread. 202 `{"status": "running"}` when started or
  already running; 200 `{"status": "fresh"}` when nothing needs fetching.
- `GET /api/leagues/{id}/history/progress` -- the progress record above,
  or `{"phase": "idle"}` when no import has run in this process.
- `GET /api/leagues/{id}/history` -- `league_overview` plus `seasons: [..]`
  and `imported_at`. 404 when the league file has no `league_members`
  table yet (the page then POSTs). Cached in memory for five minutes per
  league, invalidated when an import finishes.
- `GET /api/leagues/{id}/managers/{member_id}` -- `profile`. 404 for an
  unknown member. Same cache rule.

Progress is polled, not streamed: an import is a minute or two of work
with a stage changing every few seconds, and the page already polls the
lobby on a 12-second clock. A 2-second poll on one small endpoint costs
nothing and needs no new stream plumbing.

## Web

### `web/src/pages/LeaguePage.tsx`

On mount, `GET /api/leagues/{id}/history`. On 404, `POST`, then poll
`progress` every 2 seconds and render the stage list as it fills; when
`phase` is `done`, fetch the overview and render. The masthead (name,
team, draft clock, way in) stays as it is; the placeholder section is
replaced by:

1. **Seasons strip.** One column per season, newest last: the champion,
   the last place, and the visiting member's own finish, each as a name
   and a small figure. A season with a stored report card (the
   league-report spec's `GET /api/leagues/{id}/reports`) links to it.
2. **Managers.** A card per member: name, seasons in the league, titles,
   average finish, win percentage, and the `defining_line`. Sorted by
   average finish. Each card links to the profile.

Importing state: the stage list in the room's own monospaced face, the
finished seasons ticking, with the sections above drawn as skeletons
beneath it. Failed state: the error, and a "Try again" that POSTs.

### `web/src/pages/ManagerPage.tsx` at `/league/:leagueId/manager/:memberId`

The profile, in sections that mirror the facts: a header (name, seasons,
titles, record), then Finishes (a row per season: record, points, seed,
finish, and the draft-day projection beside the finish), Playoffs,
Head-to-head (a grid against every other member, coloured by margin),
Luck (expected against actual wins, per season), Draft (reusing the
report spec's habit figures), Waivers, Trades (a ledger, newest first,
with the balance), Lineups (bench points left by season, hit rate, the
worst week). Every figure that rests on fewer than five observations is
shown with its `n` and no adjective.

Both pages take `web/src/league.css` (the league-report spec creates it
with the `lr-` prefix; the pages here use `lg-`, already started in
`web/src/pages/league.css`, which is merged into `league.css` so the
league has one sheet).

### `web/src/api.ts`

`startLeagueHistory`, `fetchLeagueHistoryProgress`, `fetchLeagueHistory`,
`fetchManagerProfile`, with an exported interface per payload.

## Testing

- `tests/test_league_history.py` (the branch's file, extended): each
  parser against a trimmed fixture from the probe (one `mTeam`, one `mMatchupScore`, one week of
  `mTransactions2`, one week of `mRoster`, cut to two teams); the walk
  against a fake fetch that serves two seasons and 404s a third, asserting
  request count, de-duplicated transactions, and that a second walk
  fetches nothing for the finished season and only the stale week of the
  current one; a locked file retried once.
- `tests/test_manager_profile.py`: a hand-built league of four teams over
  two seasons with known answers for every fact -- a title, a last place,
  an all-play luck of exactly +1.0, a trade with a computable balance, a
  week where the optimal lineup beats the started one by a known margin,
  a waiver claim that lost to a higher priority. Small-sample facts carry
  `n` and the defining line falls back to the record.
- `tests/test_league_history_api.py`: `TestClient(create_app(tmp_path))`
  with a fake fetch in the branch's `_raw_fetch` shape and the import run
  synchronously. `owned_league` covered here once, for both APIs. 403 without a
  session and for a league not on the account; 400 for a mock id; 404 then
  202 then 200 across a first visit; progress reflects the fake walk; the
  overview cache invalidates when the import finishes.
- Web: Playwright as the owner against the dev server: the league page
  in importing state (stage list present), in imported state (seasons
  strip and manager grid render), and a manager page with every section
  present and no console errors.

## Out of scope

- Public sharing of profiles or the league page.
- Prose or LLM-written lines; the `defining_line` is chosen by rule.
- Waiver-wire or trade suggestions, projections, anything forward-looking.
- Seasons before the league's `previousSeasons` floor (ESPN serves none).
- Co-owner attribution beyond the primary owner.
- The activity/chat feed.
- Refitting the draft model per league from the new tables.
