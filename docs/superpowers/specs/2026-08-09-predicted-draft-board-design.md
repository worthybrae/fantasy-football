# Predicted Draft Board

Date: 2026-08-09

## Goal

Show the upcoming draft as a grid — rounds down the side, managers across the
top, one player per cell — filled with the model's prediction of who goes
where. Hovering a cell reveals the second and third most likely players for
that pick. Clicking one opens a condensed player profile.

This is the original request from the draft-simulator work ("predict who will
pick what") rendered as the artifact people actually think in: a draft board.

## Context

The draft simulator already ships. `scoring/draft_sim.py` runs Monte Carlo
rollouts where opponents sample from per-manager conditional-logit models
fitted to the league's own ESPN draft history, and produces two board columns:
`Avail%` (probability a player lasts to my next pick) and `ΔEV` (expected
end-of-draft starting-lineup points relative to the best alternative).

Neither of those fills a grid. `search_pick` evaluates only my next pick.
`survival` simulates only up to my next turn. Producing a prediction for all
120 cells is new work, though it reuses the entire rollout engine.

The position color palette this needs already exists. `web/src/index.css`
defines `--pos-qb` through `--pos-dst` as "distinct, dark-legible, CVD-checked"
with a stated minimum perceptual distance between adjacent pairs. They were
added for badges and have never been used at grid scale, which is what they
were designed for.

## Decisions

| Question | Decision |
|---|---|
| Cell contents | Predicted 2026 draft, not historical replay or live tracking |
| Resolving the distribution | Most likely player per cell, deduped by one-to-one assignment |
| Placement | Its own route, `/draft-board` |
| My column | Same visual treatment, marked as projected rather than predicted |
| Alternates | Second and third choices on hover |
| Click behavior | Condensed profile popup, with a link to the full page |
| Theme | Dark, matching the rest of the app |

---

## Part 1: Producing the board

### `predict_board` in `scoring/draft_sim.py`

Runs N full rollouts and counts, for every overall pick number, which player
was taken there. It reuses the existing rollout engine, the fitted opponent
models, and the current `taken` state, so its predictions agree with the
`Avail%` and `ΔEV` columns rather than forming a second opinion.

Signature:

```
predict_board(pool, settings, slot_managers, my_slot, taken, betas,
              n_rollouts=DEFAULT_ROLLOUTS, seed=0, alternates=2,
              taken_order=None) -> pd.DataFrame
```

Returns one row per (overall_pick, alt_rank) with columns
`overall_pick, round, round_pick, slot, alt_rank, player_id, prob`.
`alt_rank` 0 is the primary; 1 and 2 are the alternates.

### Two outputs per cell

**Raw frequencies.** The `alternates` most frequent players at that pick,
excluding whoever the assignment gave the cell as its primary, each with its
own probability. These are the "second and third options" hover reveals, and
they are the honest per-cell distribution alongside the primary.

**A deduped primary.** The frequencies form a picks-by-players matrix. Running
`scipy.optimize.linear_sum_assignment` on `-log(prob)` selects the
highest-likelihood assignment in which no player occupies two cells. That is
the name rendered in the cell, stored as `alt_rank = 0`.

The dedupe is not cosmetic. Without it a consensus first-rounder appears in
four adjacent cells and the grid reads as broken. With it, the primary name is
the most consistent single story the model can tell, while the hover still
reports that he was genuinely in play at all four picks.

`linear_sum_assignment` runs on a 120-by-N matrix, which is fast. Players who
never appear in any cell are excluded from the matrix rather than padded into
it.

### Cost and storage

One extra set of rollouts on top of the existing candidate search: roughly 4
seconds against the current 54, since the search runs twelve candidates over
the same rollout count.

`run_sim` calls it and writes a new `sim_board` table:
`run_id, overall_pick, round, round_pick, slot, alt_rank, player_id, prob`.
Three rows per cell (one primary plus two alternates), 360 rows for a
15-round 8-team league.

### My column is a plan, not a prediction

The rollouts fill my slot with the greedy marginal-value policy, not with a
fitted model of my own behavior. Greedy is near-deterministic given the state,
so my cells carry probabilities close to 1 that mean "this is what the plan
does", not "this is what will happen".

The grid therefore marks my column and labels those cells *projected*, and
does not render a probability on them. Borrowing the opponent columns'
confidence language for a deterministic policy would be a lie by presentation.

---

## Part 2: The grid

New route `/draft-board`, new component `web/src/components/DraftGrid.tsx`,
new API client function `fetchSimBoard()`.

### Layout

A **Round** column, a narrow snake-direction column rendering `›` on
left-to-right rounds and `‹` on right-to-left ones, then one column per
manager in draft-slot order. Rows are rounds, from `settings.rounds`.

Width is about 1250px for eight managers plus the round gutter, which is why
this took its own route rather than sharing the board page with its 280px
rail. Narrower screens scroll horizontally inside the grid container; the page
body never scrolls sideways.

### Cell states

| State | When | Rendering |
|---|---|---|
| Fact | The pick is already marked drafted | Full-strength position color, no probability |
| Predicted | An opponent cell from the model | Position color at reduced strength, probability as small muted text |
| Projected | My column | Reduced strength, left edge marker on the column, no probability |

Cells show player name, NFL team, and position, colored from the existing
`--pos-*` tokens. No new color tokens.

### Hover

Reveals the second and third most likely players for that cell with their
probabilities, in a small anchored popover. The data is already in the fetched
rows, so this costs no request.

### Empty state

With no sim run, the grid renders its frame — rounds, manager columns, snake
arrows — with blank cells, and a line pointing at the rail on the board page.
The shape of the draft is visible before there is a prediction to put in it.

### Theme

Dark, matching the app. `index.css` commits to a single dark theme
deliberately ("a draft-night tool run in a dim room, not a document that should
flip with OS preference"), and a light grid would read as pasted in.

---

## Part 3: The condensed profile

New component `web/src/components/PlayerCard.tsx`, opened by clicking a cell.
It calls the existing `/api/players/{id}/profile` endpoint, so there is no
backend change.

The full profile page is built for research. This card has to answer "do I take
him?" in about three seconds.

**Keeps:**

- Header: name, position, team, bye week
- Decision row: VOR rank, tier, market rank, edge, projected points
- Sim context when a run exists: `Avail%` at my next pick and `ΔEV`. The full
  profile page does not show these, and on this screen they are the most
  relevant numbers on the card.
- The five factor bars, compact: production, role, environment, schedule,
  durability
- One sparkline of weekly PPR points across the recency window, reusing
  `SeasonRangeChart`
- Outlook line: depth slot, implied team points, strength of schedule
- The drafted toggle

**Drops:** game log, full season table, similar players, snap-share chart,
per-source market breakdown, schedule calendar. A "Full profile" link goes to
the existing `/players/:slug` page for any of it.

Dismisses on Escape, backdrop click, or the close button. No conflict with the
board page's Escape handling, since this is a separate route.

---

## Testing

Python side gets real coverage:

- `predict_board` frequency counting against a fixture with known rollout
  outcomes
- The assignment dedupe: a fixture where the naive top-per-cell repeats a
  player, asserting the deduped board does not, and that the repeated player
  still appears in the alternates
- Determinism under a fixed seed
- The `sim_board` schema and row count
- The new endpoint, including the no-sim case returning an empty board rather
  than an error

Frontend verification is `npm run build` plus a visual pass. The repo has no
frontend test runner, and standing one up is a separate project.

## Out of scope

- Editing the predicted board by hand
- Exporting it
- More than two alternates per cell (a primary plus a second and third choice)
- Historical replay of past drafts in this grid
- Live auto-refresh as picks come in; the grid reflects the last sim run,
  including the drafted state as of that run (`certain` is computed inside
  `predict_board` and frozen into `sim_board`, so marking a player drafted
  afterwards does not change his cell), and re-running is a deliberate action

## Risks

**The primary name is a consistency choice, not the per-cell mode.** Assignment
can place a cell's second-most-likely player as its primary when that produces
a better global fit. The hover shows the raw frequencies, so the information is
never hidden, but a user reading only primaries is reading the most coherent
draft rather than 120 independent most-likely answers. This is the correct
tradeoff for something presented as a board, and it is why hover exists.

**Prediction quality is bounded by the model.** The backtest reports whether
the fitted model beats an ADP baseline out of sample. Rendering predictions in
a confident-looking grid raises the stakes on that warning reaching the user,
which the rail already surfaces.

**Greedy self-projection.** My column shows what a one-ply greedy policy would
do, which is adequate for sketching a plan but is not the same search that
produces `ΔEV` for my next pick. The two can disagree at my next turn. The
cell labeling is what keeps that honest.
