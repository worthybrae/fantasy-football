# A richer opponent model: pool signals and drafter archetypes

**Design, 2026-08-23.** Two tiers. Tier 1 feeds the model signals the board
already computes and the model cannot see. Tier 2 lets different opponents
behave differently, which the model currently cannot represent at all.

## Why

The shipped prior's coefficients say what the model is:

| feature | coefficient |
|---|---|
| `reach` | **-11.24** |
| `first_at_pos_round` | +3.66 |
| `qb_early` | +3.53 |
| `fall` | +2.60 |
| ... | |
| everything describing the player himself | ~1.6 combined |

It is roughly "who is next on the board, with small corrections". Two
measurements bound what that costs:

- **It saturates.** Fitting on 4 drafts of human picks reaches 0.2268 top-1;
  fitting on 25 reaches 0.2298. Tripling the data past ~455 human picks buys
  nothing. More data is not the lever.
- **It is better at the machine than at people.** 0.3010 top-1 on picks made
  by ESPN's engine, 0.2308 on picks made by a person. It learned an
  engine-heavy corpus and it shows.

And structurally, every opponent in a mock draft shares ONE coefficient
vector. The model cannot express "seat 3 reaches for running backs, seat 6
waits on quarterback". That is the largest unexploited axis.

## What is NOT being built, and why

A model reading variable-length picks and pool as sequences -- a set or
attention encoder -- is the right long-term architecture and is deliberately
out of scope. A 23-parameter linear model saturates at 455 human picks; a
learned set-encoder has thousands of parameters. With 2,612 human picks it
would fit noise. Revisit when the corpus is an order of magnitude larger.

Note also that the model ALREADY conditions on who is available: it is a
conditional logit and the softmax runs over exactly the players still on the
board. What is missing is not the list, it is features that SUMMARIZE the
pool. Tier 1's `pos_dropoff` is that.

## Tier 1: signals that exist and are not wired in

Six features, all candidate-specific so they survive the softmax. A column
constant across a choice set cancels out entirely and cannot matter.

| feature | definition | source |
|---|---|---|
| `pos_dropoff` | his projected points per game minus the next-best AVAILABLE player at his position, right now | `draft_log_pool.proj_points`, pool state only |
| `vor` | value over replacement | board |
| `durability` | health | board |
| `proj_change` | growth | board |
| `last_of_tier` | he is the last available player in his market tier | `tier`, as scarcity |
| `slots_left_at_pos` | starters + flex still unfilled at his position | `settings` + roster state |

**`pos_dropoff` must not be circular.** The tempting definition -- points over
the best player expected to SURVIVE to your next turn -- depends on the
survival estimate, which depends on the model being fitted. Fitting a feature
on the model's own output is unreasonable to backtest. Use the next-best
available player at that position at that moment: pure pool state, no model.

`durability`, `proj_change`, `tier` and `vor` are not in the corpus, which
stores only `adp_rank` and `proj_points` per pool row. They join from the
board at fit time, exactly as `fit_prior.attributes_by_player_id` already
joins the stat profile. `pos_dropoff` and `slots_left_at_pos` compute from the
corpus alone.

**Expect most of these to lose.** Five of eight lost last time. `vor` and
`pos_dropoff` are both correlated with market rank, and collinear additions to
a model with a -11.24 coefficient tend to split a coefficient rather than add
signal. If `pos_dropoff` survives and the rest do not, that is a good outcome.

## Tier 2: latent-class conditional logit

K components, each with coefficients `beta_k`, plus population weights `pi_k`.
The clustering unit is a SEAT WITHIN ONE DRAFT, not a person: mock opponents
are strangers with no identity that survives the session, which
`draft_log.ANONYMOUS_PREFIX` already encodes.

Fitted by EM:

- **E-step**: responsibility of class k for seat s is proportional to `pi_k`
  times the product of that seat's pick likelihoods under `beta_k`. **In log
  space** -- sixteen multiplied probabilities underflow.
- **M-step**: `pi_k` is the mean responsibility; each `beta_k` is a
  responsibility-WEIGHTED conditional logit fit.

Three things keep it upright:

1. **Shrinkage toward the pooled fit.** `draft_model.fit(X, chosen,
   prior=..., lam=...)` already fits toward a prior with a ridge penalty --
   the same mechanism per-manager fits use. Each `beta_k` shrinks toward the
   pooled beta. Without it, K=3 on this much data yields one sensible class
   and two overfitted ones.
2. **K chosen by held-out likelihood**, leave-one-draft-out, K in {1,2,3,4}.
   Not AIC or BIC. **K=1 must reproduce the pooled fit exactly** -- that is
   the correctness check on the EM machinery.
3. **Multiple random restarts**, best likelihood kept. EM finds local optima,
   and a bad initialization is indistinguishable from "there are no
   archetypes" -- which must be ruled out before reporting K=1.

Roughly 163 human seat-drafts and 2,612 picks. K=3 is 69 parameters, about 38
picks each: thin, workable with shrinkage. **K=1 winning is a real possible
outcome** and would be a finding, not a failure.

## Live inference

Each opponent seat carries a posterior over the K archetypes, seeded at `pi`.
After every pick that seat makes, Bayes updates it. At pick 1 the posterior IS
`pi`, so predictions equal today's pooled behavior -- the cold start cannot be
worse than what ships, which matters because most opponent picks happen before
a seat has revealed much.

**Rollouts sample; the single-pick answer averages.**

For "who does this team take next", predict with the mixture -- the
posterior-weighted average of the K probability vectors.

Inside `_run_draft`'s rollouts, **sample one archetype per seat per rollout
and hold it for the whole draft**. Averaging would let a seat behave zero-RB
on one pick and QB-early on the next, which is not a drafter. Sampling
preserves within-draft consistency, and the spread across rollouts is a
genuine representation of not yet knowing who a seat is -- which is exactly
what `survival()` should widen.

**Cost.** `_live_features` builds the feature matrix once per pick; only the
dot product repeats per class, so K=3 triples a matvec and not the feature
construction (measured at 98 microseconds per call). Measure rather than
assume.

**The integration risk is the interface.** `cold_start_fits()` returns
`{"__pooled__": beta}` and `slot_managers` maps a slot to a single fit;
`_run_draft`, `survival`, `search_pick` and `predict_board` all thread it
through. Carrying a mixture plus a per-seat posterior changes that shape at
five call sites. `feature_matrix` and `_live_features` already share
`roster_shape_features`; the mixture must not reintroduce a fit/serve split.

## Evaluation

**Everything is measured on HUMAN picks only, held out by draft.** The
existing backtest mixes engine picks, and that inflates top-1 by about 7
points -- measuring this work the old way would reward getting better at
predicting a bot.

Leave-one-draft-out across all labelled drafts, standard errors clustered by
draft. Not a single split; the 8-draft split used during design carried about
+/-1.5 points of slop.

**A ladder. Each rung must beat the one below on `delta_top1` or it does not
ship:**

| | held-out human top-1 |
|---|---|
| ADP baseline | ~0.19 |
| shipped prior (engine-heavy) | 0.2055 |
| human-only refit, current 23 features | ~0.230 |
| + Tier 1 features, K=1 | to measure |
| + mixture, best K | to measure |

The ladder exists to separate three effects that would otherwise be
confounded: fitting on humans, the new features, and the archetypes. The last
refit's headline gain turned out to be one third a bug repair, and that only
surfaced because it was decomposed.

No `--force`, matching `fit-prior`. A rung that loses is written up and
discarded. Snapshot the draft ids used, since the corpus grows while a fit
runs.

Findings doc: the full ladder, every feature cut, the chosen K and why, and an
explicit "what this does not establish".

## Order of work

1. **The human-only evaluation harness, and the human-only refit.** Everything
   downstream is judged by this, so it lands first and establishes rung 3.
2. **Tier 1 features.** Independent of the harness to implement, dependent on
   it to judge.
3. **Tier 2 mixture**, on top of whichever Tier 1 features survived.
