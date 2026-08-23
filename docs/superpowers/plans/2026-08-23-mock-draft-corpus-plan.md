# Implementation plan: mock-draft corpus and next-pick prediction

Spec: `docs/superpowers/specs/2026-08-23-mock-draft-corpus-design.md`

## Context discovered after the spec was written

Three findings change the order of work. They are facts, measured on this
machine, and they override the spec's phase ordering where they conflict.

**1. 45 completed mock drafts are already on disk.** `data/leagues/` holds 47
per-league DuckDB files; 45 have `drafted` rows and 23 of those hold exactly
128 picks, which is a complete draft. That is roughly 5,760 picks of stranger
behavior against the league history's 696. No ESPN access is needed to use
them, so harvesting them comes FIRST and the farm loop comes last.

**2. Those drafts are 8 teams x 16 rounds.** Confirmed from
`data/draft_room_trace.jsonl`, a recording of a real ESPN mock: 128 `SELECTED`
frames, 128 `SELECTING` frames, and team ids exactly 1..8. The same capture
shows 8 `AUTODRAFT` frames in one draft, so autodrafted seats are common and
the flag is load-bearing.

**3. The lobby join is mapped but unusable tonight.** From the trace plus
`mockdraftlobby.js`: the lobby lists rooms by `subType: "MOCKDRAFT_LOBBY"`
(each carrying `leagueId`, `full`, `teamsJoined`, `leagueSize`,
`draftAvailableDate`), a seat is claimed with
`POST .../leagues/{id}/invites?memberId={SWID}`, and from there
`draftSecurity` and the websocket are already implemented. All of it needs
`espn_s2` + `SWID`, which are not on this machine until someone runs
`scripts/espn_login.py` at a real browser.

## Global Constraints

- **Python 3, existing venv.** Run tests with `.venv/bin/pytest -q`. Never
  add a dependency.
- **Never destroy corpus rows.** `pipeline/draft_log.ensure_schema` uses
  `CREATE TABLE IF NOT EXISTS` deliberately: a draft that is not written down
  when it happens is gone. No `CREATE OR REPLACE`, no `DROP`, anywhere near
  these tables.
- **Never write to `data/leagues/*.duckdb`.** Open them `read_only=True`.
  They are the user's live league state.
- **`delta_top1` decides `FEATURE_NAMES`.** A feature ships only if it beats
  the incumbent set on held-out top-1 accuracy. This rule exists so a metric
  cannot be reached for only when it flatters the conclusion. Report log-loss
  and top-5 too, but never let them decide.
- **Unknown is NULL, never a default.** A backfilled draft does not know
  which seats were autodrafted. That is `NULL`, not `False`.
- **Follow the file's existing comment density.** These modules explain WHY
  at length. Match that; do not write bare code into a file full of reasoning.
- **Tests assert behavior, not implementation.** No test that only checks a
  function was called.

## Task 1: corpus schema + backfill the 45 drafts on disk

**Files:** `pipeline/draft_log.py` (edit), `pipeline/mock_backfill.py` (new),
`Makefile` (edit), `tests/test_mock_backfill.py` (new),
`tests/test_draft_log.py` (edit).

**1a. Schema.** Add to `ensure_schema`, after the existing DDL:

```
ALTER TABLE draft_log_pick ADD COLUMN IF NOT EXISTS autodrafted BOOLEAN
```

Add `"autodrafted"` to the end of `_PICK_COLUMNS`. `_shape` already fills
columns a caller omits, so existing writers keep working and get NULL.

**1b. Backfill.** New module `pipeline/mock_backfill.py`, entry point
`main(argv)`, wired as `make mock-backfill`.

For each `data/leagues/*.duckdb`, opened READ ONLY:

- Read `drafted` (`player_id`, `pick_no`). Skip the file if the table is
  absent or empty.
- **Accept only complete drafts.** `count(*) == max(pick_no)` and
  `max(pick_no) == 128`. Anything else is an abandoned connect — skip it and
  count it in the summary. Do not guess at partial drafts.
- Settings are `teams=8, rounds=16`. Assert `teams * rounds == max(pick_no)`
  so the assumption fails loudly if a differently-shaped draft ever appears.
  Record it in `settings_json` so the assumption is legible in the data.
- Derive each pick's slot from `draft_sim.snake_slots(8, 16)` indexed by
  `pick_no - 1`.
- `owner_key` is `draft_log.anonymous_key(draft_id, slot)` and
  `is_anonymous` is True: these seats are strangers with no identity that
  survives the session.
- `autodrafted` is NULL for every pick. We cannot know from `drafted` alone.
- `my_slot` is NULL. Nothing in these files records which seat was the user.
- Build the pool snapshot with `scoring.board.build_board(conn)` then
  `scoring.draft_sim.build_pool(conn, board, settings)`, and write the
  `_POOL_COLUMNS` fields from it. Per-pick `position`, `adp_rank` and
  `proj_points` come from that same pool, joined on `player_id`.
- `draft_id` via `draft_log.draft_id_for(SOURCE_MOCK, league_id, season,
  started_at)`, where `league_id` is the filename stem and `started_at` is
  the file's mtime — mocks in the same league on different nights must not
  collide, and mtime is the only timestamp these files carry. Say so in a
  comment.
- Write with `draft_log.record`, which is idempotent by `draft_id`, so
  re-running the backfill is safe.

Print a summary: files scanned, drafts recorded, picks recorded, and files
skipped with the reason. A pick whose `player_id` is not in the pool is
dropped and counted — report the number, because a silent join failure here
would look like a smaller corpus rather than a bug.

**Tests:** two synthetic league DBs in `tmp_path` (one complete, one
partial); assert the complete one lands with 128 picks, correct snake slots,
NULL `autodrafted`, anonymous owner keys; assert the partial one is skipped
and counted; assert re-running produces no duplicates. Add a
`test_draft_log.py` case that the new column survives `ensure_schema` on a
corpus that already has rows.

## Task 2: report what the corpus contains

**Files:** `pipeline/mock_backfill.py` (extend) or a small new reporting
function, `tests/test_mock_backfill.py` (extend).

Before anything is fitted, the corpus has to be described. Add
`make corpus-report`, printing:

- drafts and picks by `source`
- for mock drafts: distribution of first-round positions, share of picks at
  each position by round bucket (early 1-3, mid, late), and mean absolute
  deviation of pick number from `adp_rank`

That last number is the one that matters and it must be stated plainly in the
output: if mock drafters sit ~0 picks from ADP, the corpus is measuring
ESPN's autodraft, and a prior fitted on it will just relearn the market. The
spec's phase-0 "count the humans" question is answered by this number, from
data already on disk, rather than by sitting in a lobby.

## Task 3: new features

**Files:** `scoring/draft_model.py` (edit), `tests/test_draft_model.py`
(edit).

Add to `feature_matrix`, and to `FEATURE_NAMES` in the same order:

- `pos_count` — count the picking team already holds at the candidate's
  position, from `obs.roster`. Generalizes `need`, which is binary.
- `first_at_pos` — 1.0 when `obs.roster.get(position, 0) == 0`.
- `pos_gap` — rounds since that team last took the candidate's position,
  0.0 when they never have. Needs the picking team's own pick history;
  `PickObservation` must carry it if it does not already.
- `first_at_pos_round` — `first_at_pos` multiplied by the round number,
  divided by `settings.rounds` to keep it on the other columns' scale.
- the stat profile: `usage`, `efficiency`, `played_share`, `peak_gap`, each
  `_centre_within_position`-ed, read off the pool where they are already
  carried and currently unread.

**This is a conditional logit.** A column with the same value for every
candidate in a choice set cancels in the softmax and cannot matter. Every
feature above varies by candidate; a test must assert that property directly
— build a choice set, add a constant column, and assert `log_likelihood` is
unchanged.

Extend `COLD_START_PRIOR` with a 0.0 for each new feature so the existing
length assertion holds. Task 4 replaces those values with fitted ones.

## Task 4: refit the prior on the corpus

**Files:** `scoring/draft_model.py` (edit), `scoring/mock_prior.py`
(new, generated), `pipeline/fit_prior.py` (new), `Makefile` (edit),
`tests/test_fit_prior.py` (new).

Move the coefficient literal out of `draft_model.py` into
`scoring/mock_prior.py`, which `draft_model` imports and binds to
`COLD_START_PRIOR`. Every existing reader keeps its name. The existing
`assert len(COLD_START_PRIOR) == len(FEATURE_NAMES)` stays.

`make fit-prior` runs `pipeline/fit_prior.py`, which:

- builds observations from the corpus's mock drafts, excluding any pick whose
  `autodrafted` is True and any pick at `my_slot`
- fits one pooled beta
- backtests it leave-one-draft-out against the incumbent prior, reporting
  top-1, top-5 and log-loss for both
- **writes `scoring/mock_prior.py` only if the new beta wins on top-1.** No
  `--force`. A losing prior is reported and discarded; shipping one means
  changing the decision rule in a reviewable diff.

Also run the feature comparison from Task 3 here: incumbent 15 against each
new feature added, reporting `delta_top1` per feature, so the decision about
which features ship is made from measurement. Write the numbers to
`docs/superpowers/findings/2026-08-23-mock-corpus-features.md`.

**Tests:** a losing beta is not written; a winning one is; the generated
module imports and has one coefficient per `FEATURE_NAMES` entry.

## Task 5: pulse the available list

**Files:** `web/src/components/draft/AvailableList.tsx` (edit),
`web/src/App.css` (edit).

Every row pulses red, intensity scaled by `1 - survive_pct / 100`.

- Set a CSS custom property on the `<tr>` from the candidate's
  `survive_pct`; drive the animation from it.
- **Animate `opacity` and `box-shadow` only.** Never `background-color` —
  this table renders hundreds of rows and only compositor-friendly properties
  are affordable.
- **`survive_pct === null` gets no animation at all.** Null means there is no
  roster to survive for, not a zero chance of surviving. The existing
  `riskTone` call already refuses to color a dash for this reason; follow it.
- **Honor `prefers-reduced-motion: reduce`** with a static tint carrying the
  same information.
- Rows already marked `avail-row-taken` do not pulse.

Verify with `cd web && npm run build` (typecheck + production build).

## Task 6: the farm loop

**Files:** `pipeline/espn_mock_lobby.py` (new), `pipeline/mock_farm.py`
(new), `Makefile` (edit), `tests/test_mock_farm.py` (new).

Depends on ESPN cookies existing at `data/espn_state.json`. Build and test it
regardless — the tests are all offline.

**6a. Lobby client.** `pipeline/espn_mock_lobby.py`:

- `list_mock_leagues(fetch)` — open rooms, filtered to not `full` and
  `draftAvailableDate` in the future. The exact list endpoint is NOT yet
  known; the lobby bundle fetches by `subType: "MOCKDRAFT_LOBBY"`. Discover
  it by probing the documented ESPN v3 league endpoints with the saved
  cookies, and if it cannot be found, raise a clear error naming what was
  tried. Do not fabricate an endpoint that returns nothing.
- `join(fetch, league_id, swid)` — `POST .../leagues/{id}/invites?memberId={swid}`,
  returning the assigned `team_id`.

**6b. Farm loop.** `pipeline/mock_farm.py`, `make farm-mocks N=50`:

join a room, connect via `draft_socket.run_socket_listener`, play
epsilon-greedy on our turns (`epsilon=0.2`: the tool's own top candidate
otherwise a uniformly random legal pick), send picks as `SELECT` frames the
way `/api/live/select` does, record the completed draft with
`source=SOURCE_MOCK`, `my_slot` set, and per-pick `autodrafted` from the
listener's `AUTODRAFT` frames. Repeat N times with a delay between drafts.

**Tests are offline.** Replay `data/draft_room_trace.jsonl` through the
listener and assert the resulting `DraftRecord` has 128 picks, the right
slots, and the autodraft flags the capture shows. Test the epsilon-greedy
policy against a fake pool with a fixed seed, both branches.
