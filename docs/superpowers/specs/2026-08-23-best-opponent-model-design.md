# The best opponent model the data can support

**Design, 2026-08-23.** Supersedes the Tier 1 / Tier 2 plan as the direction
of travel; those land as evidence. Goal stated by the owner: the most accurate
predictor of which player a team takes next, using everything available.
One input is excluded by instruction: the room's `experienceType`.

## What the day's measurements established

1. **The model reads the wrong board.** Drafters in an ESPN room read ESPN's
   list, sorted by ESPN's rank, on screen. Against that list, 23.1% of human
   picks are the top name, 47.4% are in the top 3, 59.8% in the top 5. Against
   `market_rank` -- what `reach`/`fall` are computed on, with a -11.24
   coefficient -- those numbers are 15.9 / 36.8 / 49.8. The fitted model
   (0.236 top-1) barely beats "take the top of ESPN's list" (0.231) because
   the heuristic is using better information than the model.
2. **Drafter identity is a weak signal.** Per-seat ICC of reach depth is
   0.027; of every player attribute tested, at most 0.04 and mostly ~0. Seats
   do not show stable preferences for durable, young, high-usage or any other
   kind of player. What little consistency exists is in how closely a seat
   follows the list.
3. **The linear model saturates at ~455 human picks.** Capacity, not data, is
   what limits it.
4. **Data is no longer scarce.** 284 human picks per session-hour measured;
   three sessions yield ~20k/day, ~143k/week.
5. **Top-1 is bounded by the task.** 47.4% of picks are inside ESPN's top 3;
   the remainder are reaches that no board feature predicts. Realistic best
   case is high-20s to low-30s. The product consumes the full distribution
   (survival probabilities in rollouts), so calibration matters at least as
   much as the argmax.

## The design

### 1. Reference board: ESPN's rank

`espn_ppr_rank` and `espn_proj` from the board's `espn_adp` table become the
primary reference. `reach`/`fall` and list position are computed against it;
consensus `adp_rank` stays as a secondary input, because disagreement between
the two boards is itself informative.

Corpus: `draft_log_pool` gains `espn_rank DOUBLE`, `espn_proj DOUBLE`, `bye
INTEGER`. Additive `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`; the corpus is
never rebuilt. The farm fills them at record time from the board it already
builds. Existing drafts are backfilled from today's `espn_adp` -- a preseason
quantity, stable across the season, which the measurement above already
joined at 95% coverage.

### 2. Capture what the stream already carries and we discard

- **Time to pick.** `SELECTING <team> <clock_ms>` opens a turn; `SELECTED`
  closes it. The listener's events carry no wall-clock today. Add
  `received_at` to every event, and `seconds_to_pick` plus `clock_seconds` to
  `draft_log_pick`. A three-second pick is a list-follower; a sixty-second
  pick deliberated. It is the most behavioral signal in the stream and it is
  available at serve time too -- the live socket sees the same frames.
  Unrecoverable for drafts already recorded: NULL, never a default.
- **Roster identity.** Not new columns -- the pick list already has it -- but
  the features in section 3 read it.

### 3. Features

All candidate-specific; a column constant across the choice set cancels in
the softmax. The existing 29 stay as candidates. New:

| feature | definition |
|---|---|
| `espn_list_pos` | log rank among AVAILABLE players by ESPN rank -- the thing on screen |
| `espn_reach`, `espn_fall` | `_log_rank_features` against ESPN rank instead of market rank |
| `espn_proj` | ESPN's displayed projection, and its drop to the next available at the position |
| `board_disagreement` | `log1p(adp_rank) - log1p(espn_rank)`: where the room's list and the market disagree |
| `team_stack` | candidate shares an NFL team with a rostered QB (for WR/TE) or rostered WR/TE (for QB) |
| `homer_share` | fraction of the roster on the candidate's NFL team |
| `bye_conflict` | rostered starters sharing the candidate's bye |
| `rookie` | board flag |
| `injury` | `player_status`, as a severity ordinal |
| `news_recency` | days since the candidate's last `player_news` item before the draft |
| `seat_follow_rate` | the seat's mean reach depth over its picks so far, vs ESPN rank |
| `seat_pace` | the seat's mean seconds-to-pick so far |
| `seat_pos_counts` | per-position counts of the seat's picks so far, interacted with candidate position |

The `seat_*` features are the history encoder. Given finding 2 they are
expected to contribute modestly; they are included because the effect is
nonzero and because the neural scorer can use them in interaction.

### 3b. The decomposition the data revealed -- this reshapes section 4

Asked against ESPN's board, 23.1% of human picks are the top name overall.
Asked a different way, **53.0% are the top name AT THEIR POSITION**, 72.9%
are top-2 and 82.6% top-3 -- and late rounds hold at 48.1%, where the overall
number collapses to 20%. The "unpredictable reach tail" is mostly people
choosing a position and taking the best player there.

So the choice factorizes:

    P(player) = P(position | roster, round, run, scarcity, seat history)
              x P(player | position, ESPN list, stack, homer, bye, injury, news)

The second factor is at 53% with a trivial rule. A crude held-out lookup on
(round, roster shape) alone predicts the position 46% of the time. Arithmetic:
0.65 x 0.58 = 0.38; 0.78 x 0.64 = 0.50. Mid-30s to 40 is a concrete target;
50 requires a position model near 80% and is the stretch goal.

### 4. The scorer: a nested logit with positions as nests

Positions are the nests. Two scorers: a POSITION scorer over the six nests,
fed the roster/round/run/scarcity/seat-history features, and a WITHIN-POSITION
scorer over the players in the chosen nest, fed the list/player features.
The nested-logit likelihood ties them through the standard inclusive-value
term, so the whole thing is still one softmax over the available set and
still fits the existing machinery. Each scorer is a two-hidden-layer MLP (32 units, tanh) applied identically to every candidate,
softmax over the available set. Roughly 2k parameters. This supplies the
interactions a main-effects model cannot express -- a reach costs differently
when a team is desperate at a position than when it is deep there.

Implemented in numpy with hand-derived backprop and L-BFGS or Adam. **A
finite-difference gradient check is a required test.** No new dependency: a
torch model would need a numpy re-implementation to serve inside rollouts at
~200 microseconds per call, and two implementations of one scorer is the
fit/serve drift this project has already been bitten by twice today. If numpy
proves impractical the implementer says so with the measurement rather than
quietly adding a dependency.

Regularization: L2 plus early stopping on held-out drafts. The features are
standardized on training folds only.

### 5. Serving

`_live_features` already builds the feature matrix once per pick; the MLP
forward is one extra matmul pair. Measure it. The live listener has the same
`SELECTING`/`SELECTED` frames, so `seat_pace` and `seat_follow_rate` are
computable for real opponents mid-draft. Parity tests between fit and serve
extend to every new column and to the scorer itself.

### 6. Evaluation and the decision rule

Human picks only, held out by draft, standard errors clustered by draft.
Five folds grouped by draft rather than leave-one-out, because a neural fit
per fold is slow; the ladder's existing rungs are recomputed under the same
folds so nothing is compared across fold schemes.

**The decision rule changes, and this is the place that change is made in
the open.** The shipping metric becomes **held-out log-loss on human picks**,
with **top-1 as a must-not-regress guard**. Reason: the product consumes
probabilities, not an argmax -- `survival()` runs rollouts that draw from the
predicted distribution, and a better-calibrated model at the same top-1 is a
better draft assistant. The findings report top-1, top-3, top-5 and log-loss
regardless. No force flag.

The ladder:

| rung | |
|---|---|
| ADP baseline | |
| "top of ESPN's list" heuristic | the new naive bar, 0.231 today |
| human-only refit, 23 features, market board | 0.2358 today |
| same, ESPN board | isolates the board fix |
| + new features, linear | isolates the features |
| + MLP scorer | isolates capacity |

Each step is decomposed so that a gain is attributable. The last refit's
headline was one-third a bug repair, and that only surfaced because it was
decomposed.

### 7. Data

Three sessions, as the owner chose. The 8-team PPR room filter stays unless
the owner relaxes it; if it is relaxed, `teams` and scoring format become
inputs and evaluation reports the 8-team PPR subset separately.

## Order of work

**Phase A -- capture.** Schema, listener timestamps, farm recording of ESPN
rank/proj/bye and time-to-pick, backfill of ESPN rank onto existing drafts.
Independent of the model and needs to be running as early as possible, since
every draft farmed before it lands has NULL for time-to-pick forever.

**Phase B -- model.** Board switch, features, MLP scorer, fit/serve parity,
the ladder. Starts once the in-flight Tier 1 measurement and Tier 2 mixture
have reported, because they share `scoring/draft_model.py`,
`scoring/draft_sim.py` and `pipeline/fit_prior.py`.

## What this does not promise

50% top-1. The data says 47.4% of human picks sit inside ESPN's top three and
the rest are reaches that show no stable per-seat pattern. A model cannot
predict what people do not do consistently. The target is the best
calibrated distribution the data supports, with top-1 improving as a
consequence rather than as the objective.
