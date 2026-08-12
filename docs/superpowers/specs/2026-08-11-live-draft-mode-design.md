# Live Draft Mode

**Goal:** During an ESPN draft, keep the board current as picks land and tell
me what to take when I am on the clock, without typing.

Today the loop is manual and batch: mark players drafted with the `D` hotkey,
run `make sim`, read a static grid. Every piece of the underlying engine
already supports mid-draft resume. What is missing is the loop, the ingest,
and a view built for the ninety seconds you actually have.

## Scope

This is one sub-project of a larger direction (a multi-league live draft
helper). The others were deliberately deferred, and naming them here is what
keeps them out of this implementation:

| deferred | why not now |
|---|---|
| Multi-league support | Every table is single-league keyed. Large, touches every module, and ships nothing usable in this season's draft. |
| Cold-start model for leagues with no history | The hard, novel piece. Needs multi-league underneath to matter. |
| Mock drafts as an ADP source | Measured worse for this league than ESPN's board (top-1 0.2011 against 0.2356). Mocks appear here only as a test harness. |
| Player profiles | Already built — PlayerProfile, GameLog, SnapShareChart, ScheduleCalendar, SimilarPlayers, StatTiles. |

**ESPN only.** No other platform is in scope.

## What already exists

Worth stating precisely, because it is most of the work and the design
depends on not rebuilding it:

- `drafted` table with `pick_no`, written by `POST /api/drafted/{player_id}`,
  cleared by `DELETE`. The `D` hotkey and the hide-drafted filter are live.
- `draft_sim._drafted_state(conn, pool)` turns that table into
  `(taken, taken_order)`, and refuses to guess when `pick_no` is null rather
  than silently misattributing picks.
- `draft_sim._run_draft(...)` resumes from any number of completed picks:
  `already = len(taken_order)` indexes into the snake order.
- `draft_sim.search_pick(...)` returns expected end-of-draft roster value per
  candidate — the "what should I take now" answer.
- `pipeline/espn_league.py` fetches `view=mDraftDetail`, and
  `parse_draft_picks` already parses `draftDetail.picks` into
  `(overall_pick, round, team_id, espn_player_id, ...)`.
- Playwright auth with a saved `storageState`.

## The measurements that drove this design

Taken on the live board, not estimated:

| operation | cost |
|---|---|
| `build_board` | 0.9s |
| `build_pool` | 1.1s |
| `fit_all` | **14.9s** |
| `search_pick` @ 25 rollouts | 4.7s |
| @ 50 | 9.1s |
| @ 100 | 19.5s |
| @ 200 | 35.2s |

Roster EV carries **±4.2** at 400 rollouts. Standard error scales as
1/sqrt(n), so at 25 rollouts it is **±16.8** — and adjacent candidates in the
same run differed by **9.3 points**. Two consequences, and they are the two
load-bearing decisions:

1. `fit_all` at 14.9s must not run per refresh. Nothing it computes changes
   during a draft.
2. At the rollout budget continuous refresh allows, the top candidates are
   statistically indistinguishable. A naive implementation would reshuffle
   the recommendation every twenty seconds while the board sat still.

## Architecture

### The session

One long-lived object in the API process, built once at draft start (~17s)
and never rebuilt:

```python
class DraftSession(NamedTuple):
    league_id: str
    my_slot: int
    slot_managers: dict          # slot -> manager
    settings: LeagueSettings
    pool: SimPool                # build_pool output, index-stable for the draft
    betas: dict                  # manager -> fitted coefficients
    board_fingerprint: str       # detects the board changing underneath us
    seed: int                    # pinned; see below
    started_at: datetime
```

Every refresh then costs only `search_pick`.

### The pinned seed

This is what makes continuous refresh trustworthy. `search_pick` already uses
common random numbers *within* a call. Live mode pins the seed *across* calls,
so every refresh rolls out the same set of hypothetical drafts. A change in
the recommendation then means the board changed — a player came off, a roster
need shifted — rather than the dice landing differently.

Because the seed is fixed, raising `n_rollouts` between refreshes refines the
same scenario set rather than resampling a different one: the estimate
converges instead of jumping.

Without this, ±16.8 of noise across a 9.3-point gap makes the top
recommendation flip at random, and the tool is least trustworthy exactly when
it is being relied on.

### Components

| file | responsibility |
|---|---|
| `pipeline/espn_live.py` (new) | Poll `view=mDraftDetail`, filter placeholders with the existing `_is_real_pick`, map `espn_player_id` to `player_id`, write `drafted`. Reuses the saved Playwright `storageState`. |
| `api/live.py` (new) | Session lifecycle and state. Owns the cached pool/betas and the pinned seed. |
| `web/src/components/LiveDraft.tsx` (new) | On-the-clock view: who is up, your ranked candidates, board updating as picks land, staleness banner. |
| `scoring/draft_sim.py` (modified) | Accept a prebuilt pool rather than rebuilding one. No structural change — `search_pick` and `_drafted_state` already do the work. |

### Endpoints

- `POST /api/live/start` — body `{my_slot}`. Builds the session, returns its
  id and the initial state. Idempotent: starting twice returns the existing
  session rather than building a second.
- `GET /api/live/state` — current picks, whose turn it is, ranked candidates,
  `last_poll_at`, `unmapped_picks`, and `candidates_as_of_pick` (the pick
  count the candidates were computed against, so the client can distinguish
  a current answer from one still being recomputed).
- `POST /api/live/stop` — tears the session down.

### Data flow

```
espn_live poller  --writes-->  drafted table
                                   |
                          session detects pick count changed
                                   |
                          search_pick(pool, ..., seed=pinned, n_rollouts=N)
                                   |
   LiveDraft.tsx  <--polls--  GET /api/live/state
```

One direction. No websockets, no push. The client polls every 2-3s, far below
ESPN's pick cadence, and every stage is inspectable when something goes wrong
at 8pm on draft night.

## Recompute policy

One code path. `N` is chosen from how close my turn is:

| picks until my turn | N | cost | why |
|---|---|---|---|
| 3 or more | 25 | 4.7s | Situational awareness between picks, which arrive every ~20-30s |
| 1-2 | 100 | 19.5s | Sharpening as it approaches |
| on the clock | 200 | 35.2s | ~90s clock, 55s of margin |

Because the seed is pinned, these are refinements of one scenario set rather
than three different answers.

**Start the deep search before the clock starts, not when it starts.** The
moment the pick before mine lands I am next up, and nothing about the board
will change again until I act. So the N=200 search begins then, and its result
is waiting the instant the clock does start. Waiting until I am formally on
the clock would burn 35 of my 90 seconds computing something that could have
been computed during someone else's turn.

**A new pick supersedes an in-flight computation.** Picks arrive every ~20-30s
and an N=100 refresh costs 19.5s, so the two will overlap. A result computed
against a board that has since changed is worse than no result: it recommends
a player who may already be gone. Any refresh still running when a new pick
lands is abandoned and restarted against the new board. Only the most recent
completed computation is ever served, and `/api/live/state` reports the pick
count its candidates were computed against so the client can tell the
difference between "thinking" and "current".

## Failure modes

Ordered by how much damage each does.

| failure | consequence | handling |
|---|---|---|
| **ESPN player id does not map to ours** | A drafted player silently stays on the board and gets recommended | Loud. `unmapped_picks` is surfaced in the UI immediately and never swallowed. Crosswalk coverage must be *proven* against a real payload before draft night, not assumed. |
| **Poller dies mid-draft** | Board freezes while looking live; stale numbers read as confident ones | `last_poll_at` on every state response. Older than 15s and the UI says so. |
| **Desync** (ESPN has 14 picks, we have 12) | Every roster, need and cap downstream is wrong | Never patch incrementally. On any count mismatch, resync from ESPN's full pick list and rebuild `drafted` from scratch. |
| **Null `pick_no`** | Picks unattributable to teams | Already handled: `_drafted_state` raises. Keep it. |
| **Board rebuilt mid-draft** | Cached pool indices no longer match | `board_fingerprint` pinned at start; the session refuses to serve if it changes. |

### Manual fallback

The `D` hotkey stays live — not as a parallel ingest path but as the escape
hatch for the first two rows above. Reconciliation rule, stated once:
**ESPN wins.** A manual mark that the poller later contradicts is overwritten.

## Testing

- **Unit** — poller translation against recorded ESPN payloads as fixtures.
  No network in tests. Covers placeholder filtering, id mapping, and the
  unmapped-pick path.
- **Integration** — the full loop replayed against a recorded mock draft:
  picks land, session refreshes, recommendations update, unmapped picks
  surface, staleness trips.
- **Live** — an actual ESPN mock draft. The only thing that proves the poller
  works against the real endpoint.

## The risk that gates everything

**Does ESPN populate `draftDetail.picks` during a live draft, or only on
completion?**

The design assumes progressively. Evidence for: `import_seasons` already
guards with `bool(detail.get("drafted")) and any(_is_real_pick(p) ...)` —
which exists precisely because ESPN serves a full placeholder board for
*upcoming* drafts, implying the structure is live before completion. That is
suggestive, not proof.

This must be verified in an ESPN mock draft as the first implementation task.
If picks do not appear until the draft ends, polling is dead and ingest falls
back to the manual `D` hotkey with the rest of the design unchanged — the
session, the pinned seed, the recompute policy and the on-the-clock view all
still stand.
