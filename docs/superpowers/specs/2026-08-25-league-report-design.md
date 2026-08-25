# League report: team profiles, draft report card, power rankings

**Date:** 2026-08-25
**Status:** approved in conversation, awaiting spec review

## The problem

The tool knows more about a league than anyone in it. Six seasons of draft
picks with the market's price on every one, a fitted pick model per manager,
and the live room's own steal/reach reading on every pick of this year's
draft. None of that reaches the other seven people in the league. The
morning after the draft is when a league argues about who won it, and the
only artifact this product leaves behind is a database on one person's
laptop.

What is wanted is a page the owner can drop in the league group chat: a
draft report card with a grade and a funny, specific blurb for every team,
power rankings, and a profile of each team built from how it has drafted
and finished over the years. Blurbs are written by a cheap Claude call.
Everything else is arithmetic.

Two facts about the repository shape the design:

- **No standings are stored.** The importer keeps draft picks, teams and
  settings, and nothing about how a season ended. The ESPN payload it
  already fetches (`view=mTeam`) carries each team's `record.overall`
  (wins, losses, ties, pointsFor, pointsAgainst), `playoffSeed` and
  `rankCalculatedFinal`, so standings are a parser change, not a new fetch.
- **Only the default league has history.** `pipeline/import_league.py`
  writes to `data/nfl.duckdb` through a bare `get_conn()`. The per-league
  files under `data/leagues/` hold universal tables and live-room state and
  nothing else. A report for any connected league means importing that
  league's history into its own file, with that account's ESPN cookies.

## Decisions made with the owner

- **Audience:** a public, read-only link for the whole league. Viewing is
  free. Reports are meant to be read on phones.
- **Scope:** any league the account has connected, not only the default.
- **Trigger:** generated automatically when a live draft in the room
  completes. An owner-only endpoint also regenerates or backfills a season,
  because without one nothing can produce a report for a past season and
  nothing on production can be tested before draft night.
- **Billing:** generation is gated by the existing `require_paid` /
  `entitled` check for that league and season. Mock leagues never generate.
- **Approach:** Python computes every number; one Haiku call writes the
  prose against those numbers; the whole result is stored in the league's
  database and the page serves the stored copy. The alternatives were
  generating on page read (every cold hit pays and waits, blurbs differ per
  viewer, a shared link is unbounded spend) and letting the model do the
  analysis (unreliable arithmetic, nothing unit-testable). Both rejected.

## Data

### `league_standings` (new per-league table)

One row per team per season, parsed from the same payload `import_seasons`
already reads:

```
season, team_id, manager, team_name, wins, losses, ties,
points_for, points_against, playoff_seed, final_rank
```

`manager` uses the same resolution as `draft_teams` (member display name,
falling back to team name) so the two tables join on `(season, team_id)` and
also agree on the name a person is known by across seasons. `final_rank` is
ESPN's `rankCalculatedFinal`, which is the playoff-resolved finish; when it
is 0 or absent (a season in progress), it is stored as NULL. The current,
unfinished season is stored too, with whatever partial record ESPN reports,
so "this year so far" can appear on a profile without a second code path.

Added to `LEAGUE_TABLES` in `pipeline/db.py`; the exactness assertion in
`tests/test_leagues.py` is updated with it. The field names above are
ESPN's documented ones; the first implementation step verifies them against
a real fetch of league 53929318 and adjusts the parser, not the schema.

### Per-league history import

`import_seasons(conn, league_id, current_season, fetch)` already takes its
connection and its fetch. Two changes around it:

1. A fetch adapter that turns the cookie-authenticated
   `fetch(url, cookies, headers) -> (status, text)` of `pipeline/espn_drafts.py`
   into the `fetch(url) -> parsed JSON` that `import_seasons` expects,
   raising `FileNotFoundError` on 404 so the season walk still reads a
   missing year as a miss.
2. `pipeline/league_history.py: import_history(league_id, cookies, fetch)`
   opens `league_db_path(league_id)` (the default id still resolves to
   `data/nfl.duckdb`, so the owner's own league keeps working unchanged),
   runs `import_seasons` with that adapter, then writes `historic_adp` for
   every imported season exactly as `import_league.main` does today.
   `historic_adp` is a universal table, so a season already present in the
   file is not fetched again.

The import runs in a daemon thread when an account connects to a real
league's room (the connect path in `api/live.py` is the one place the
request's ESPN cookies are in hand). It is idempotent and skips entirely
when the league's `league` table was written within the last seven days.
By the time a draft ends, the history is already on disk and the draft-end
job needs no ESPN call and no credential.

### This season's picks

For the season being drafted, `draft_picks` will not exist until the next
import; the room's `drafted` table (player_id, pick_no) and the board the
room built are the source. `scoring/league_report.py` accepts either shape:
a frame of `(season, overall_pick, team_slot_or_id, player_key, market_rank)`
built from `draft_picks` + `historic_adp` for past seasons (through
`draft_model.build_observations`, which already does the ADP matching and
drops picks with no ADP row that year) or from `drafted` + the room's board
for the live season. The mapping from pick number to team is the snake
order from `league.settings_json.pick_order`, the same source the room uses.

## Computation: `scoring/league_report.py`

Pure functions over a connection and a season. No network, no model.

### Pick verdicts

A Python port of `verdict()` in `web/src/components/draft/PickTicker.tsx`
so the report card and the ticker never disagree about what a steal is.
`value = overall_pick - market_rank`, NULL when the market ranked the
player past the last pick of that draft (the kicker/defense rule from
`api/live.py`). Bands, with `round = max(2, teams)`:

| `abs(value)` | sign | verdict |
|---|---|---|
| ≤ 2 | any | on the market |
| < round | > 0 | value |
| ≥ round | > 0 | steal |
| < round | < 0 | early |
| ≥ round | < 0 | reach |

### Draft grades (per team, one season)

- `value_total`: sum of non-null `value` over the team's picks.
- `value_per_pick`: `value_total / count(non-null)`.
- `steals`, `reaches`: counts by verdict.
- `best_pick`, `worst_pick`: the picks with the highest and lowest `value`,
  each carrying player name, position, round, pick and the ADP it was
  measured against.
- `shape`: count of picks by position, and the round of the first QB, TE,
  K and DST.
- `grade`: letter from the team's `value_per_pick` rank within the league
  (top 1 A, next 2 B, middle C, next 2 D, bottom 1 F for eight teams; the
  cut points scale with team count as fractions 1/8, 3/8, 5/8, 7/8). A
  grade is a rank, not an absolute, because the room's ADP feed shifts
  year to year and a fixed threshold would grade whole drafts up or down
  together.

### Team profiles (per manager, all seasons)

Keyed on `manager`, because team names change and team ids can be reused.

- `seasons`: list of `(season, wins, losses, ties, points_for, final_rank,
  playoff_seed)` from `league_standings`.
- `titles`: seasons with `final_rank == 1`. `playoffs`: seasons with a
  playoff seed. `avg_finish`, `win_pct`, `points_for_per_game` over
  completed seasons.
- Draft habits over every season with picks: `mean_value` (positive means
  this manager tends to get players past their ADP), `steal_rate`,
  `reach_rate`, `first_pick_positions` (what they open with, as counts),
  `earliest_qb_round`, `earliest_te_round`, and the single best and worst
  picks of their career by `value`.
- `this_season`: the draft grade record above, when a report exists for the
  season being built.

`manager_tendencies` and `manager_profiles` are fitted-model artifacts that
exist only after `make fit-managers`. The profile does not depend on them;
if `manager_profiles.summary` rows exist for the manager they are attached
as `model_notes` and passed to the writer as extra color.

### Power rankings (one season)

A blended z-score, highest first:

```
score = 0.6 * z(value_per_pick, this draft)
      + 0.4 * z(win_pct, completed seasons; 0 for a first-year manager)
```

A first-year manager has no history and is ranked on the draft alone, and
the payload says so. Ties broken by `value_total`. The weights are
constants at the top of the module, named, with a one-line reason each.

### The report payload

One JSON document, the thing stored and served:

```
{
  "league_id", "season", "league_name", "teams": N, "generated_at", "model",
  "status": "ready" | "numbers_only" | "failed",
  "intro": str | null,
  "power_rankings": [{"rank", "manager", "team_name", "score", "line": str | null}],
  "report_cards": [{"manager", "team_name", "grade", "value_total",
                    "value_per_pick", "steals", "reaches", "best_pick",
                    "worst_pick", "shape", "nickname": str | null,
                    "blurb": str | null}],
  "profiles": [{"manager", "seasons", "titles", "playoffs", "avg_finish",
                "win_pct", "mean_value", "steal_rate", "reach_rate",
                "first_pick_positions", "career_best", "career_worst"}]
}
```

`status: numbers_only` is a report built without a writer (no API key, or
the call failed after its retry); every `str | null` above is null and the
page renders without prose. It is a complete report, not an error, and the
page must not look broken. `failed` is reserved for the numbers themselves
failing (no picks, no settings), with a `reason`.

## The writer: `scoring/blurbs.py`

`write_blurbs(client, facts) -> dict` makes exactly one call per report.
`facts` is the payload above minus the prose fields, compacted: every
number rounded, player names as strings, nothing the model does not need.
For an eight-team league that is roughly 4-6K input tokens and 2-3K output
tokens, about two cents on Haiku. The client is injected so tests pass a
fake; production builds `anthropic.Anthropic()` once per job.

- Model: `claude-haiku-4-5`, chosen by the owner for cost. No thinking
  parameter (Haiku 4.5 does not take adaptive thinking).
- Structured output: `output_config.format` with a JSON schema requiring
  `intro`, `cards: [{manager, nickname, blurb}]` and
  `rankings: [{manager, line}]`, `additionalProperties: false`. The
  response is parsed with `json.loads`, then checked: every manager in the
  facts appears exactly once in `cards` and `rankings`, and no manager
  appears that is not in the facts. A response that fails the check is
  retried once with the failure named in a follow-up user message; a second
  failure returns `None` and the report stores as `numbers_only`.
- `max_tokens`: 4096. The schema bounds the shape; a blurb is 2-3
  sentences and the prompt says so.
- System prompt, frozen text in the module: the voice (a league-mate
  roasting friends, not a broadcaster; specific over generic; every joke
  must cite a fact from the numbers it was given; never invent a player,
  pick or season; never mention real people outside the league; keep it
  clean enough for a group chat that has somebody's parent in it). The
  facts go in the user message as JSON. Nothing in the request varies by
  time or request id, so if a second call ever happens in the same five
  minutes the system prompt is a cache hit.
- Errors: `anthropic.RateLimitError` and `APIStatusError >= 500` are
  retried by the SDK's own two retries; anything else is caught, logged
  with `response._request_id` where there is one, and yields `None`. The
  job never raises out of the writer.
- Off switch: `ANTHROPIC_API_KEY` unset means `client` is `None` and the
  writer is skipped. Same shape as `billing.enabled()`: a local checkout
  works without any key and produces a `numbers_only` report.

The `anthropic` package is added to `requirements.txt`, pinned to its
major version, with the paragraph the file asks for: the alternative is
hand-rolling the request against `httpx`, and the SDK is what carries the
retry policy, the typed errors and the structured-output parameter that
this code would otherwise re-implement and get subtly wrong.

## Storage: `league_reports` (new per-league table)

```
season BIGINT, generated_at TIMESTAMP, model VARCHAR, status VARCHAR,
payload_json VARCHAR
```

One row per season, replaced on regeneration. Added to `LEAGUE_TABLES`.
Reports are stored, never rebuilt on read: a shared link costs nothing
after the first build and every reader sees the same blurbs.

## The job

`api/reports.py: build_report(league_id, season) -> dict` is the whole
pipeline: open the league's connection, compute, write prose, store. It
runs on a daemon thread the same way `api/jobs.py` runs the refresh loop,
with one flight per `(league_id, season)` at a time. A second request
while one is running is answered 202 with `{"status": "building"}` rather
than starting another; the eight-concurrent-builds incident that
`scoring/board_cache.py` documents is the reason.

DuckDB has one writer per file. The report job writes to the league file;
the live listener for the same league also writes to it (`drafted`).
The build takes a fresh connection, holds it for the write only, and
retries the write once after a short wait if the file is locked. Reads
during a build are unaffected.

## API: `api/reports.py`

Registered in `create_app` after `register_seo_routes` and before
`register_spa`, so the SPA catch-all does not swallow it.

- `GET /api/leagues/{league_id}/reports` — public. `[{season, generated_at,
  status}]` for stored reports, newest first. `[]` for a league with none
  or a league file that does not exist. Never 404s.
- `GET /api/leagues/{league_id}/report/{season}` — public. The stored
  payload. 404 when there is none. `Cache-Control: public, max-age=300`.
- `POST /api/leagues/{league_id}/report/{season}` — owner only. The
  request's custody session must exist and that league must be in the
  account's league list (`espn_drafts.league_entries`, the same source
  `GET /api/espn/drafts` uses); otherwise 403. Then `require_paid`. Then,
  if history is missing or stale, import it with the session's cookies
  before building. Answers 202 `{"status": "building"}` and the page polls
  the GET. Mock league ids are refused with 400: there is nothing to
  report on.

Auto-trigger: in the live listener's `on_change`, at the existing
`made >= total_picks` branch, after `clear_session_record`, the same
`build_report` is spawned for `(league_id, season)` if the league is not a
mock and `entitled(store.account_id(session.swid), league_id, season)`
holds. The check is `entitled` rather than `require_paid` because there is
no request on the listener thread. It fires once, because that branch
fires once.

Regeneration is unlimited for the owner in this version. The cost is
about two cents a press and the gate is already a paid account; a limit
can come later if anyone finds a reason to press it a thousand times.

## Web

### `/leagues/:leagueId/report/:season`

A new page, `web/src/pages/LeagueReport.tsx`, with its own sheet
`web/src/league.css` (imported by the page, `lr-` class prefix, the
established per-page pattern). **Routed outside `MobileGate`:** the whole
point is a link opened from a group chat on a phone. The route element is
placed as a sibling of the gate in `App.tsx`, and the page is designed
mobile-first, single column, with the desktop layout as the enhancement.

Sections, in order:

1. Header: league name, season, "Draft report card", and the intro blurb
   when there is one.
2. Power rankings: an ordered list, rank, team name, manager, the one-line
   take. A first-year manager's row says "first draft" where history would
   go.
3. Report cards: one card per team in grade order. Grade as the big
   figure; nickname and blurb; best pick and worst pick each with the
   verdict word and the ADP gap; the roster shape as a small position
   count row. When `blurb` is null the card shows the numbers and no empty
   frame.
4. Team histories: one row per manager, seasons across as a compact record
   line (`10-4 · 2nd`, a crown on a title year), career win%, and the draft
   habit line ("opens RB, reaches for TEs, best pick ever: ...").
5. Footer: when it was generated, and "numbers only" when there is no
   prose so nobody reads a missing blurb as a broken page.

While a build is in flight (GET returns 404 and the owner just pressed
POST), the page polls every 3 seconds up to two minutes and shows a
building state. A visitor hitting a 404 with no build sees a plain
"no report for this season" page.

### Dashboard

Each `.db-league` card in `web/src/components/Dashboard.tsx` gets a
"Report card" link to the newest stored season when
`GET /api/leagues/{id}/reports` is non-empty, and a "Build report card"
button (POST, then navigate to the page in its building state) when it is
empty and the league is not a mock. One fetch per card, made when the
dashboard mounts.

### `web/src/api.ts`

Three functions in the file's existing style: `fetchLeagueReports`,
`fetchLeagueReport`, `buildLeagueReport`, with an exported interface per
payload.

## Testing

- `tests/test_espn_league.py`: standings parsed from a literal payload,
  including a team with `rankCalculatedFinal: 0` (stored NULL) and a
  member with no display name (falls back to team name, same as
  `draft_teams`).
- `tests/test_leagues.py`: the `LEAGUE_TABLES` assertion gains both tables.
- `tests/test_league_report.py`: the verdict port against the bands above
  including the ±2 and `round` boundaries; grades on a seeded eight-team
  season (extending `_seed_draft_history` from `tests/test_api.py` with
  standings and enough picks per team); a first-year manager's power
  ranking; a league with no ADP rows for a season yields `failed` with a
  reason rather than an exception.
- `tests/test_blurbs.py`: with a fake client, the request carries the
  model, the schema and every manager; a response missing a manager is
  retried once with the failure named; two failures yield `None`; an
  `APIConnectionError` yields `None` and does not raise; a `None` client
  is skipped without a call.
- `tests/test_reports_api.py`: `TestClient(create_app(tmp_path))` with the
  job's writer replaced by a fake. GET on an unknown league is `[]` / 404;
  POST without custody is 403; POST for a league not in the account's list
  is 403; POST for a mock id is 400; a stored report round-trips through
  GET with the cache header; a second POST during a build answers 202
  without a second build.
- Web: the existing Playwright signed-out check (`?signedout=1`) extended
  with the report page at a phone viewport, asserting the page is not
  gated and every card renders in `numbers_only`.

## Out of scope

- Weekly matchup scores and head-to-head records (`view=mMatchupScore`).
  Final standings are enough for a profile; week-by-week is a second
  fetch per season and nothing in the report card needs it.
- A model bigger than Haiku, or a per-team call. One call keeps the jokes
  consistent across teams and is the cheap version the owner asked for.
- Rate-limiting regeneration.
- Refitting the manager model per league from the imported history. The
  report reads fitted artifacts if they exist and never triggers a fit.
