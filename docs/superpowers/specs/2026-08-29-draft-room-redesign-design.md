# Draft room redesign: ESPN order, empirical availability, plan, favorites

Date: 2026-08-29. Source: the owner's brain dump of 2026-08-28
(`2026-08-28-draft-strategy-brain-dump.md`) and the ranking-pipeline map.

## The problem

- The room ranks by `gain_now`, which inside a position is a straight
  function of VOR; a player's own survival never enters his rank. With pick 6
  the room recommends the consensus #1 at 2 % "lasts" (`scoring/gain.py:278`,
  `:336-340`).
- `gain_now` is priced against a "horizon" pick that is not one of the
  user's turns; the cards' captions say "next pick". Two different picks
  behind one number.
- Every recompute runs 400 rollouts of an opponent model (`draft_sim.survival`),
  0.97-1.49 s of Python per pick per room. At 100 rooms this is the CPU wall.
- No per-round plan exists; no per-account preferences exist.

## Decisions

1. The room's order is ESPN's rank. No re-ranking by our model in the live
   room. The model (`survival`, `rank_available`, `search_pick`) stays for
   the simulation and report pages.
2. The one computed dynamic number is the chance a player is still there at
   the user's next pick, measured from recorded ESPN mock drafts.
3. Points are a differential against who the user can expect to get at their
   next pick, at the same position.
4. A per-round plan names a target (and two alternates) for each of the
   user's remaining turns, with rule-derived pros and cons.
5. Signed-in users pick 5-25 favourite players once; the plan targets them.

## 1. Order and columns

Source of ESPN's rank: `espn_adp.espn_ppr_rank` (ESPN's draft-lobby rank,
refreshed by `pipeline.refresh`). Fallback for a player missing there:
`historic_espn_cs.cs_rank` for the current season. Players with neither keep
the consensus rank and sort after ranked players.

Per row the room shows: ESPN rank, ESPN positional rank (derived: order of
ESPN rank within the position), ESPN ADP (`espn_adp.espn_adp`, blank when
`espn_adp_is_usable()` is false), Consensus (`market_rank`, the existing
five-source median, unchanged), Lasts %, Edge pts, projected points per game,
and the existing health/finish/reliable/growth meters. `gain_now`,
`vor_points` and `fills` are gone from the payload.

Default sort: ESPN rank ascending. All columns stay sortable.

## 2. Lasts % — `scoring/availability.py`

Data: `data/draft_corpus.duckdb` (`draft_log_pool`, `draft_log_pick`; 854
drafts, 109k picks, 252 players today). `draft_log_pool` is the denominator
(drafts where the player was in the pool).

Precomputed once per process and refreshed when the corpus file's mtime
changes (`AvailabilityTable`):

- `players`: array of player ids present in the pool table.
- `pooled[i]`: drafts the player was pooled in.
- `taken_by[i, p]` for p = 0..MAX_PICK (MAX_PICK = 300): number of those
  drafts in which he was taken at overall pick ≤ p. Monotone in p.
- A parametric fallback fitted from the same rows: for ADP buckets of width
  8, the mean and standard deviation of the pick a player actually went at,
  as a function of ESPN ADP. Stored as a small table (`adp_bucket → (mu, sigma)`).

Query, vectorised over the available players of one room:

```
P(still there at pick N | still there at pick K)
  = (pooled - taken_by[N]) / (pooled - taken_by[K])
```

where K = the current pick (picks made) and N = the user's next turn. When
the denominator is below `MIN_DRAFTS = 25`, or the player is not in the
corpus, the fallback is used: with mu/sigma from the player's ESPN ADP
bucket (ADP itself when the bucket is empty; consensus rank when ADP is
blank), P = 1 − Φ((N − mu)/sigma) conditioned the same way on K. Players
with no ADP and no consensus rank get P = 1.

Overall pick number is the axis. Team count is not scaled: ADP is already in
overall picks and the corpus is 8-team; a 12-team league's round 3 is picks
25-36 on the same axis. The number is exposed as `lasts_pct` (0-100) and
the pick it refers to as `lasts_at_pick`. When the user has no further turn,
`lasts_pct` is null.

`availability_at(table, player_ids, k, n)` also serves the plan for every
future turn, and `api/demo.py` uses the same function for the landing room
so the two agree.

## 3. Edge pts — `scoring/plan.py`

For candidate i at the user's current pick K with next turn N:

```
edge_i = proj_i − E[max proj over players j ≠ i, position(j) = position(i),
                     available now, weighted by P_j(N | K)]
```

computed exactly as `gain.expected_best_next` does today (sorted by
projection, product of "nobody better survived") but over the empirical
probabilities and excluding the candidate. Projections are the board's
`proj_points` under the league's scoring. Positive edge means "more than
you would get by waiting". Exposed as `edge_pts`.

Need weight: `gain.need_kind`/`NEED_WEIGHTS` are reused unchanged to tag
each candidate with `need` ∈ {starter, flex, deferred, bench, capped}; the
room shows it as the existing "fills" hint text.

## 4. Plan — `scoring/plan.py`

Input: the user's remaining turns T₁ < T₂ < … < Tᵣ (overall picks), the
available pool with projections, positions, ESPN ranks, the roster counts
so far, the favourites set, and the availability table.

Greedy, one pass:

```
roster = counts so far
for each turn Tₜ:
    P_t = availability at Tₜ conditioned on now
    eligible = players with P_t ≥ 0.50, or favourites with P_t ≥ 0.35
    for each eligible i:
        score_i = weight(need_kind(position_i, roster)) * proj_i
                  * (1.15 if favourite else 1.0)
                  − E[best other at position_i at T_{t+1}]   (0 for the last turn)
    target = argmax score; alternates = next two by score at other or same position
    roster[position(target)] += 1
    mark target and alternates as planned (a planned target is not eligible
    later)
```

Output per turn: `{pick_no, round, target: {player_id, lasts_pct, edge_pts,
reasons: [...]}, alternates: [{player_id, lasts_pct, edge_pts}, ...]}`.

Reasons are rule-derived strings, at most four, in this order:

- `"★ favourite"` when in the favourites set.
- `"{lasts_pct}% still there at pick {T}"`.
- `"+{edge} pts over the next {POS} you'd get at pick {T'}"` when edge > 0,
  `"−{|edge|} pts vs waiting for {POS}"` when edge < 0 (a con).
- `"fills {slot}"` from `need_kind` (e.g. "fills RB2", "flex").
- `"ADP {adp} vs ESPN {rank} — may go earlier"` when ESPN ADP is more than
  6 picks ahead of ESPN rank (a con); the reverse as a pro.
- `"bye week {n} stacks with {name}"` when two already-rostered starters share
  his bye (a con).
- `"health {h}/5"` when the board's health meter is ≤ 2 (a con).

Pros and cons are separate lists in the payload (`pros`, `cons`).

On the clock, the "Target now" cards are the top three by the same score
for the current pick (P = 1 for everyone available), so the recommendation
and the plan use one rule.

## 5. Favourites — `api/account.py`

Table `favorite_player(account_id VARCHAR, player_id VARCHAR, ord INTEGER,
created_at TIMESTAMP, PRIMARY KEY (account_id, player_id))` in the billing
store (DuckDB file or Postgres, same backend selection as `entitlement`).
`account_id` is `store.account_id(swid)` under the current key; reads use
every `account_ids(swid)`, writes the newest, exactly as `entitled()` does.

Routes (all require a custody session; 401 otherwise):

- `GET /api/account/favorites` → `{"players": [player_id, ...]}` in `ord`.
- `PUT /api/account/favorites` body `{"players": [...]}`: 5 ≤ n ≤ 25,
  ids must exist on the board, replaces the set. 422 otherwise.

Frontend: on the signed-in Dashboard, when the account has no favourites,
an onboarding panel (`FavoritesPicker`) with search, position filter,
headshots, ordered selection, a counter "5-25", and Save. Afterwards a
compact "Your guys" card with Edit. The draft room marks favourites with ★
in the list and the plan.

Favourites live in the room's session so the plan can use them: the connect
resolves the account's favourites once and stores the id set on the
`DraftSession`; `PUT` while a room is live updates the room on the next poll.

## 6. Live room wiring

`api/live.py` `_recompute(s, session, picks_made)`:

1. `taken`, `taken_order`, roster counts, `on_the_clock`, my turns — as today.
2. `lasts = availability_at(table, ids, k=picks_made, n=next_turn)`.
3. `edge`, `need` per candidate from `scoring/plan.py`.
4. `plan = build_plan(...)` for the remaining turns.
5. `state["candidates"]` rows: `player_id, position, proj_points, espn_rank,
   espn_pos_rank, espn_adp, market_rank, lasts_pct, lasts_at_pick, edge_pts,
   need, favourite, rank` where `rank` is the ESPN-order position in the
   list (unranked players last, by consensus).
6. `state["plan"]` = the plan payload; `/api/live/state` returns it under
   `plan`; `horizon_pick` is removed from the payload.

`RECOMPUTE_SLOTS`, `rollouts_for_load`, `SURVIVAL_ROLLOUTS` are removed from
`api/live.py`; the recompute worker stays (coalescing) but a recompute is
milliseconds.

`api/demo.py` computes the landing room's candidates with the same functions.

## 7. Frontend

- `AvailableList.tsx`: columns per §1; default sort ESPN rank; ★ for
  favourites; header tooltips rewritten to describe the numbers exactly.
- `TopThree.tsx` → `TargetCards.tsx`: three cards from the plan's current
  turn (on the clock) or the next turn (waiting), each with the pros/cons
  lists.
- `PlanPanel.tsx`: the remaining turns as rows (pick, round, target with
  headshot, lasts %, edge, alternates), in the rail under the roster.
- `FavoritesPicker.tsx` and the Dashboard "Your guys" card.
- `api.ts` types updated; `LiveCandidate` loses `gain_now/gain_next/
  edge_next/survive_pct/fills/vor_points`.

## 8. Tests

- `tests/test_availability.py`: conditional probability from a hand-built
  corpus; the empty-denominator fallback; the parametric fallback; the
  refresh-on-mtime behaviour.
- `tests/test_plan.py`: edge excludes the candidate; the greedy plan never
  targets the same player twice; favourites lower the threshold and add the
  bonus; reasons in the specified order; Gibbs case — at pick 6 with a
  player whose P(next) is 0.02, he is neither the target nor an alternate.
- `tests/test_live_api.py`: the recompute tests that monkeypatch `survival`
  and `rank_available` are rewritten against the new functions; the ten
  pre-existing failures are replaced by these.
- `tests/test_account_api.py`: favourites round trip, bounds, 401 without a
  session, ids validated against the board.
- Frontend vitest: the picker's 5-25 bounds; the plan panel renders reasons.

## Out of scope

- Any change to the simulation/report pages or their model.
- Cross-platform consensus beyond the existing five-source median.
- An LLM-written pros/cons text.
