# Do the six Tier 1 pool signals predict a person better than the 23 already shipping?

**Measured 2026-08-23 on 43 labelled ESPN mock drafts (2,889 known-human
picks). Answer: yes, and four of the six shipped. Rung 4 beats rung 3 by
+0.0239 +/-0.0078 held-out human top-1, about three standard errors. Read
two caveats before quoting that: essentially all of it is `vor`, and `vor`
is one of four columns joined from TODAY's board rather than the board each
mock was drafted against.**

This is rung 4 of the ladder in
`docs/superpowers/specs/2026-08-23-richer-opponent-model-design.md`. Rung 3 --
the same 23 features refitted on human picks only -- is the bar, and it is
recomputed inside this same run rather than quoted from the run that shipped
it, because the corpus has grown since.

## What was measured, and on what

`pipeline/score_ladder.py`, run as
`make score-ladder LEAGUE_DB=data/leagues/1251381776.duckdb`. About 43
minutes; almost all of it is nine leave-one-draft-out backtests at 43 pooled
conditional-logit fits each.

**Everything is scored on HUMAN picks only** -- `autodrafted IS FALSE`, at a
seat that is not ours. NULL is unknown, and for a measurement unknown is not
human: the fit rule in `make fit-prior` keeps NULL picks and this evaluation
cannot, because the shipped prior scores about seven points higher on picks
ESPN's engine made than on picks a person made, and a mixed number is partly
"how well do we predict a bot".

**The draft_id snapshot.** The corpus grows while a run is going -- the farm
adds roughly one draft every fifteen minutes -- so the set of drafts is fixed
at the first query and printed. This run: **66 drafts carry a pool, 43 of them
carry at least one pick labelled `autodrafted IS FALSE`.** All 43 were farmed
live by `pipeline.mock_farm` (they carry `my_slot`); the 23 harvested drafts
have NULL on every pick and can never be labelled, because the rooms are gone.
All 43 are 8-team, 16-round, PPR, season 2026. The list is in
`scoring/human_prior.py`'s `DRAFT_IDS`.

**Picks.** 5,497 in those drafts, **2,889 scored**. Every exclusion, counted:

| excluded | n | why |
|---|---|---|
| `my_slot` | 688 | our own seat. Part policy, part deliberate epsilon-greedy randomness, so not anyone's choice |
| `not_human` | 1,920 | `autodrafted IS TRUE` (1,442) plus `IS NULL` (478) |
| player not in the pool | 0 | would be a join failure in the recording |
| player already taken | 0 | would be a duplicate pick |

Across the 43 the label reads 3,577 False / 1,442 True / 478 NULL. Position
mix of the 3,577 known-human picks: WR 1,287, RB 1,153, QB 399, TE 368,
DST 188, K 182.

**The fold is the DRAFT.** One room, one board, eight strangers, 128 picks
that all see each other. Every draft is held out in turn and the pooled fit is
rebuilt on the other 42. The `+/-` on a row is that row's own top-1 clustered
by draft; the `+/-` on a delta is the standard error of the DIFFERENCE, paired
by draft -- both sides score the same picks in the same rooms, so what makes
one room harder than another cancels.

## The ladder

| | held-out human top-1 | top-5 | log-loss |
|---|---|---|---|
| ADP baseline (softmax over the board) | 0.1776 +/-0.0086 | 0.5448 | 8.3181 |
| shipped prior `mock_prior.PRIOR` (engine-heavy) | 0.2195 +/-0.0106 | 0.6206 | 2.8712 |
| rung 3: human-only refit, 23 features | 0.2326 +/-0.0083 | 0.6390 | 2.7835 |
| rung 4: + all six Tier 1 | 0.2555 +/-0.0110 | 0.6556 | 2.7026 |
| **rung 4 shipped: + the four that earned it** | **0.2565** +/-0.0108 | 0.6549 | 2.7019 |

Paired deltas, each against the rung below:

| | delta_top1 |
|---|---|
| shipped prior - ADP baseline | +0.0419 +/-0.0098 |
| rung 3 - shipped prior | +0.0132 +/-0.0076 |
| rung 4 (all six) - rung 3 | +0.0228 +/-0.0078 |
| **rung 4 (shipped subset) - rung 3** | **+0.0239 +/-0.0078** |

**+0.0239 +/-0.0078 is the number the rule read**, and it is the subset's
number rather than the full six's on purpose: the artifact carries the subset,
so the subset is what had to win. It is about 3.1 standard errors.

**Rung 3 here is 0.2326, not the 0.2358 recorded when it shipped.** Same
features, same rule, different corpus: rung 3 was measured on 38 drafts /
2,511 human picks and this run holds 43 / 2,889. That is why the ladder
recomputes every rung inside one run instead of quoting the last one. The
comparison that decides is the one where both rows scored the same 2,889
picks.

**In-sample check.** 3 of the 43 drafts are in `mock_prior.DRAFT_IDS`, so part
of the shipped prior's row is not held out for it. That flatters rung 2 and
makes every delta above it conservative. Over the 40 drafts it has never seen
(2,822 human picks): shipped prior 0.2162, rung 3 0.2314, rung 4 0.2534. The
gaps widen slightly, as expected.

## Per-feature: which of the six earn a place

Each row drops one column from the full 29-feature rung-4 model and re-runs
the whole leave-one-draft-out backtest. `delta_top1` is what the full model
has that the model without that column does not. Positive earns a place; at or
below zero is cut, and a cut column's coefficient is 0.0 rather than
fitted-and-small.

| dropped | top-1 | log-loss | delta_top1 | in SEs | verdict |
|---|---|---|---|---|---|
| none (all 29) | 0.2555 | 2.7026 | — | — | — |
| `vor` | 0.2368 | 2.7638 | **+0.0187** +/-0.0066 | 2.8 | **kept** |
| `dropoff_at_pos` | 0.2537 | 2.7024 | **+0.0017** +/-0.0016 | 1.1 | **kept** |
| `slots_left_at_pos` | 0.2544 | 2.7101 | **+0.0010** +/-0.0062 | 0.2 | **kept** |
| `last_of_tier` | 0.2548 | 2.7062 | **+0.0007** +/-0.0013 | 0.5 | **kept** |
| `durability` | 0.2558 | 2.7023 | -0.0003 +/-0.0006 | -0.6 | cut |
| `proj_change` | 0.2565 | 2.7023 | -0.0010 +/-0.0020 | -0.5 | cut |

**Ships:** `vor`, `dropoff_at_pos`, `last_of_tier`, `slots_left_at_pos`.
**Cut:** `durability`, `proj_change`.

**Only `vor` is distinguishable from zero.** Its delta is 2.8 standard errors
and it is worth 54 held-out picks; the other three "winners" are 1.1, 0.5 and
0.2 standard errors, which is to say they are unmeasured rather than small.
They ship because the rule is "positive earns a place" and bending it upward
for features I happen to doubt is the same failure as bending it downward for
ones I like. **They are the first three to re-measure when the corpus grows**,
and nobody should quote +0.0010 for `slots_left_at_pos` as if it meant
anything: its own error is six times the estimate.

Read another way: `vor` alone accounts for essentially the whole rung. The
full six beat rung 3 by +0.0228 and `vor` on its own is worth +0.0187 of it.

`last_of_tier` behaves exactly as the design document predicted. It fires on
**0.77% of candidate rows** -- it is true of one player in 130 -- so its delta
is dominated by sampling noise, and the +0.0007 above should be read as "no
measurable effect" and not as a small positive one. Its coefficient came out
at +1.89, which is large, but a large coefficient on a column that is almost
always zero moves almost nothing.

### How much of a chance each column had

A conditional logit sees a column only through its variation INSIDE one choice
set: a column holding the same value for every candidate divides straight back
out of the softmax and contributes exactly nothing. So `sets_varying` is the
ceiling on how often a feature could have mattered at all.

| feature | nonzero_share | sets_varying | std |
|---|---|---|---|
| `vor` | 0.9750 | 1.0000 | 4.009 |
| `dropoff_at_pos` | 0.9692 | 1.0000 | 0.565 |
| `durability` | 0.7840 | 1.0000 | 0.401 |
| `proj_change` | 0.7513 | 1.0000 | 2.552 |
| `slots_left_at_pos` | 0.6648 | 0.9474 | 0.915 |
| `last_of_tier` | 0.0077 | 1.0000 | 0.087 |

Every one of the six had a real chance -- none is constant across a choice
set, and every board-joined column actually joined (the `--league-db` board
supplied `vor` on 97% of candidate rows). The two that were cut were cut on
evidence, not on an empty column. `last_of_tier` is the exception worth
naming: it varies in every choice set, but on 0.77% of rows, which is not the
same thing as having a chance to be measured.

### What happened to `reach`

Rung 3 fits `reach` at -9.05. Rung 4 fits it at **-6.57**, and `vor` at
+0.739. That is the collinearity the design document warned about, playing out
in the direction it did not predict: `vor` and market rank do share signal and
the coefficient did split, but the split ADDED 2.4 points of top-1 rather than
merely redistributing mass. The market's ordering and the board's value-over-
replacement are not the same opinion, and the model is better for carrying
both.

## Where the accuracy moved

Rung 3 against rung 4 (all six), same held-out human picks. Buckets are
`draft_model._round_bucket`: early is rounds 1-3, mid 4-8, late 9-16.

| bucket | n | rung 3 top-1 | rung 4 top-1 | rung 3 top-5 | rung 4 top-5 |
|---|---|---|---|---|---|
| early | 712 | **0.3890** | 0.3820 | 0.8848 | **0.8933** |
| mid | 1,006 | 0.2376 | **0.2505** | 0.7058 | **0.7157** |
| late | 1,171 | 0.1332 | **0.1827** | 0.4321 | **0.4594** |

**Most of the win is in the late rounds**, where top-1 goes from 0.1332 to
0.1827 -- 58 of the roughly 66 extra correct picks come from rounds 9-16 -- and
the early rounds get very slightly WORSE (-0.70pp, 5 picks). That is the same
shape the 2026-08-11 finding flagged and it deserves the same discount: rounds
9-16 are where kickers and defenses have to be taken and where the market's
ordering is thinnest, so a value-over-replacement column has the most room to
beat ADP there. If what you care about is who your opponents take in round 2,
this rung bought you nothing. If it is who is gone before your round 11 pick,
it bought a lot.

## The provenance caveat, which the headline rests on

**`vor` is not in the corpus.** `draft_log_pool` stores `adp_rank` and
`proj_points` per row and nothing else, so `vor`, `durability`, `proj_change`
and the `tier` behind `last_of_tier` are all joined by
`fit_prior.board_signals_by_player_id` from the board as it stands TODAY,
under each draft's own settings, rather than the board as it stood when each
mock was drafted.

**Two of the four columns that carry this caveat shipped, and one of them is
the entire result.** So, stated plainly: **rung 4's win rests on a join the
drafter did not necessarily see.** `last_of_tier` carries the same caveat and
the same statement, though it is not doing measurable work either way.

What limits the damage, and it is worth being precise rather than reassuring:

* Pick order, `market_rank`, `reach` and `fall` are stored per draft and are
  completely unaffected. The choice sets and the market the model is scored
  against are the real ones.
* All four joined columns are PRESEASON quantities. `vor` is computed off the
  projection and the replacement level; `durability` and `proj_change` off
  finished seasons in `weekly`. None of them can see a 2026 game, because no
  2026 game has been played. They move with a projection refresh, not with a
  result.
* The drafts span a few weeks of one preseason, so the board has drifted but
  not turned over.

What is NOT established is that the drift is harmless. A projection refresh
between a draft and today changes `vor` for exactly the players whose stock
moved, and if the room was reacting to the same news, then part of the +0.0187
is the feature being told what the room already knew. **The honest claim is
"value over replacement predicts human picks, measured with a board that is
weeks newer than the picks", and the way to settle it is to record `vor` per
draft in `draft_log_pool` and re-run.** That is a corpus change, not a
modelling one, and it is the single highest-value follow-up from this rung.

`dropoff_at_pos` and `slots_left_at_pos` are exempt: they compute from the
stored pool and the draft's own `settings` alone. `dropoff_at_pos` is also
deliberately NOT the survival-weighted version, which would have come out of
`draft_sim`'s rollouts and been fitted on the output of the model being fitted.

## What this does not establish

* **Not that the model drafts better.** It predicts opponents better on one
  population. Nothing here measured a roster, a win, or a points total.
* **Not that five of the six features matter.** One does. Three more are
  positive by less than two standard errors and ship on a rule, not on
  evidence. Two are cut. If someone re-runs this on twice the corpus and
  `slots_left_at_pos` comes out negative, nothing in this document is
  contradicted.
* **Not that `vor` is worth +0.0187 to a drafter who has today's board.** It
  is worth that to a model scored against picks made weeks before the board it
  was given. See above.
* **Not that a CUT feature is worthless.** `durability` and `proj_change` were
  measured on this population, at this sample size, in the presence of the
  other 28 columns, and did not pay. `durability` in particular is a
  within-position percentile that averages 50 by construction; a corpus with
  more injury churn in it might find something.
* **Not that the other eight `UNMEASURED_FEATURES` are settled on humans.**
  The roster-shape and stat-profile columns were measured by `make fit-prior`
  on the MIXED corpus and have never been measured on human picks only. This
  rung deliberately did not re-litigate them -- rung 3 is pinned to all 23 by
  name, so cutting one here would confound "an older column stopped paying"
  with "the pool signals paid". That is a separate rung for a separate run.
* **Not that it is true of the picks it was measured WITHOUT.** 1,920 of the
  5,497 picks in these drafts were excluded, 478 of them because nobody
  recorded whether a person or the engine made them. Those picks were excluded
  from the measurement, not shown to be anything.
* **Not that it generalises past ESPN's public mock lobby**, which is the only
  population that is both labelled and large enough to hold out by draft.

## What ships

`scoring/human_prior.py`, regenerated. 27 fitted coefficients: rung 3's 23
plus `dropoff_at_pos`, `vor`, `last_of_tier` and `slots_left_at_pos`.
`durability` and `proj_change` carry 0.0 annotated `measured on humans, cut on
delta_top1` -- a different fact from the 0.0 they carried yesterday, which
meant not yet measured.

**Nothing imports it yet, and that is deliberate.** `scoring.draft_model` still
binds `COLD_START_PRIOR` to `mock_prior.PRIOR`. The serving swap is one line
and it belongs at the END of the ladder: rung 5's mixture changes the SHAPE of
what a seat is simulated with (K vectors plus a per-seat posterior, threaded
through `_run_draft`, `survival`, `search_pick` and `predict_board`), and
`human_prior.py` is where those vectors land. Serving a single vector from
there first would mean changing the serving path twice.
