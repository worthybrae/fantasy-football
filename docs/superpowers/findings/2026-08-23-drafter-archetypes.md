# Do drafters come in kinds? A latent-class conditional logit, halted mid-sweep

**Measured 2026-08-23 on the cross-league mock corpus. Nothing shipped, and
nothing should: the leave-one-draft-out K sweep was stopped at 23 of 43 folds
and its per-fold results were lost, so the number that would have decided
between one drafter and several was never obtained. The evidence that does
exist points at K=1, weakly. The model itself has since been superseded by a
nested logit (`docs/superpowers/specs/2026-08-23-best-opponent-model-design.md`,
section 4), so this is a record of a question left open, not a result.**

Read the "What this does NOT establish" section before quoting anything here.
It is longer than usual because more than usual is missing.

## The question

Every opponent in a mock draft is scored by ONE coefficient vector. The model
cannot express "seat 3 reaches for running backs, seat 6 waits on quarterback"
-- not poorly, at all. The design in
`docs/superpowers/specs/2026-08-23-richer-opponent-model-design.md` called that
the largest unexploited axis in the opponent model and proposed the smallest
thing that could say otherwise: K coefficient vectors `beta_k`, population
weights `pi_k`, and a per-seat posterior over which of the K a given opponent
is, fitted by EM.

**The clustering unit is a SEAT WITHIN ONE DRAFT, not a person.** Mock
opponents are strangers with no identity that survives the session --
`pipeline/draft_log.py` keys them `anon:{draft_id}:{slot}` -- so there is no
manager to pool across drafts, and asking for one would invent an entity the
data does not contain. A seat is 2 to 16 picks by whoever sat there for an
hour. `PickObservation.manager` already carries exactly that key, which is why
nothing upstream had to change to fit this.

## What was built

| file | what it is |
|---|---|
| `scoring/mixture.py` | the model: EM, log-space E-step, responsibility-weighted M-step, shrinkage toward the pooled fit, random restarts, and the live-inference primitives |
| `pipeline/fit_archetypes.py` | the measurement: the K sweep, the ladder rung, and a writer that refuses to write a loser |
| `tests/test_mixture.py`, `tests/test_fit_archetypes.py` | 40 tests, including the K=1 identity below |

`draft_model.fit` and `neg_log_likelihood` gained two arguments. `weights` is
one non-negative number per choice set and is the M-step: a latent-class fit
maximizes the ordinary log-likelihood with each pick counted by the
responsibility its seat's posterior puts on the class being fitted, and a
fractional count cannot be expressed by duplicating rows. `start` separates
where L-BFGS begins from what the ridge penalty pulls toward, so the EM can
resume each class from its own previous coefficients while every class is
shrunk toward the same pooled vector. The objective is convex, so `start`
changes the cost and not the answer: a cold solve on this corpus is 15 seconds
and a warm one is 0.1.

Both default to exactly what was there before.

## The one thing that IS established: the machinery is right

**K=1 reproduces the pooled fit exactly.** This is the correctness check the
design asked for, and it is exact by construction rather than by luck. At K=1
every responsibility is 1.0, so the M-step maximizes
`LL(beta) - lam*||beta - pooled||^2`, and `pooled` is the argmax of both terms
separately, for any lambda. Measured on the real corpus: the pooled fit and the
K=1 mixture agreed to 2.8e-5, which is L-BFGS-B's own stopping tolerance and
three orders of magnitude below the smallest coefficient in the vector. Both
scored log-likelihood -7615.12 on the same 2,745 picks.

`tests/test_mixture.py` asserts it at lambda 0, 1 and 1000, and asserts the
matching identity on the serving side: the prequential scorer at K=1 returns
the same top-1 and log-loss as `fit_prior.score` on the pooled vector, because
a one-class posterior cannot move.

Two more checks that make a K=1 answer readable rather than ambiguous:

- **EM finds archetypes when they exist.** On synthetic seats drawn from two
  drafters who differ by 6.0 on one coefficient, the fit separates them and
  assigns >85% of seats to the right class.
- **EM does not invent archetypes when they do not.** On synthetic seats drawn
  from ONE drafter, held-out likelihood prefers K=1 -- but only once the
  decision rule accounts for noise. With a bare `>`, K=2 still edges K=1 by
  about 4e-5 nats per pick on single-drafter data, because a second class can
  always absorb a little of the folds' own sampling noise. `choose_k` therefore
  requires a gain larger than its own paired standard error across drafts.

Without the second and third of those, a K=1 result would have been unreadable:
"there are no archetypes here" and "this EM cannot find archetypes" would look
identical.

## The rungs below, which DID finish

Re-measured on this run's own snapshot rather than quoted from the previous
run, because the corpus grows while a fit runs. Leave-one-draft-out, human
picks only (`autodrafted IS FALSE`, at a seat that is not ours), standard
errors clustered by draft.

**Snapshot.** 66 drafts carry a pool; 43 carry at least one known-human pick.
5,497 picks in those 43, of which **2,889** are known human picks at a seat
that is not ours, across **251 seats**. Excluded: 688 at our own slot, 1,920
not known to be human, 0 not in the pool, 0 already taken.

| rung | held-out human top-1 | top-5 | log-loss |
|---|---|---|---|
| rung 3, the 23 features already shipping | 0.2326 +/-0.0083 | 0.6390 | 2.7835 |
| rung 4, all 29 (rung 3 plus the six Tier 1 pool signals) | **0.2565** +/-0.0113 | 0.6490 | 2.7028 |

Rung 4 is +2.4pp over rung 3 on this snapshot, so **the bar for the mixture was
0.2565** and the mixture was fitted on all 29 columns. That is the ladder's own
discipline: the archetypes go on top of the feature set that won its own rung,
not one that lost, or the two effects are confounded in the other direction.

(Rung 4 here is the plain all-29 fit with no feature selection. The rung-4
agent's own run additionally ablates each column and may ship a subset; a
selected subset would score at or above this row and would be optimistic by
whatever the selection is worth. Either way 0.2565 is the honest, selection-free
bar and it is the one the mixture had to clear.)

## The K sweep: what happened and what is missing

Leave-one-draft-out over the 43 drafts, K in {1,2,3,4}, shrinkage lambda 10.0,
three random restarts per fit, eight worker processes. K chosen by held-out
MARGINAL log-likelihood per pick -- the class memberships integrated out, which
by the chain rule is exactly the probability a live posterior-updating
predictor assigns to those picks -- and not by AIC or BIC.

**It was stopped at 23 of 43 folds, after 108 minutes.** Two reasons, and the
first is the one that matters: eight workers at capacity for two hours starved
the live mock farm of CPU, the corpus stopped growing, and a schema migration
the farm needed was blocked. The farm is the higher priority. Second, while it
ran, the opponent-model design moved on: the flat mixture is superseded by a
nested logit with positions as nests.

**The 23 completed folds produced nothing readable.** `sweep` aggregates its
per-fold reports at the end and prints nothing until it finishes, so stopping
it discarded every fold it had already done. That is a design fault in
`pipeline/fit_archetypes.py`, it is now written at the top of that module, and
it is the first thing to fix if anyone picks this up.

**Cost, since it is the reason there is no answer.** One fold -- one pooled fit
plus four EM fits at three restarts each, over 2,889 picks and 29 features --
took 25 to 40 minutes. Eight workers cleared 23 folds in 108 minutes and the
full sweep was heading for roughly four hours. The design predicted this would
be affordable; it is not, on this machine, alongside a live farm.

## What the in-sample ladder says, which is all there is

These numbers are from a completed, smaller run: the same EM on the 23-feature
design over an earlier snapshot of the same corpus (41 labelled drafts, 2,745
human picks, 238 seats), two restarts, lambda 10.0. **They are TRAINING
log-likelihoods on the picks they were fitted on**, which is precisely the
quantity the held-out sweep existed to discount.

| K | training log-likelihood | gain over K-1 | class weights |
|---|---|---|---|
| 1 | -7615.12 | — | 1.00 |
| 2 | -7559.25 | +55.87 | 0.62 / 0.38 |
| 3 | -7551.22 | +8.03 | 0.50 / 0.26 / 0.24 |

Each additional class costs 24 free parameters here: 23 coefficients and one
mixing weight. In-sample log-likelihood cannot go down when parameters are
added, and under a null of no real extra class the expected in-sample gain is
roughly half the parameter count, about 12 nats.

- **The third class is worth less than noise.** +8.0 nats for 24 parameters is
  below what fitting noise alone would deliver. K=3 is not a finding; the
  weights suggest it is K=2's smaller class cut in half.
- **The second class is worth more than noise in-sample, and not much more
  out of sample.** +55.9 nats for 24 parameters, net about +32 after the
  crudest possible parameter correction, is **0.012 nats per pick** against a
  log-loss of 2.70. That is 0.4%. The held-out number would be smaller than
  that, not larger, and the decision rule requires it to clear its own paired
  standard error across drafts.
- **The split is not degenerate.** 0.62/0.38 is two populated classes, not one
  populated class and one empty one, so this is not the restart-collapse
  failure mode that would make a K=1 report untrustworthy.

An independent measurement done alongside this one puts the per-seat
intraclass correlation at 0.027 for reach depth and at or below 0.04 for every
player attribute. That is the same answer from a different direction: almost
none of the variation in how a pick is made sits between seats. **The two
agree, and the reading is K=1.** But "the two agree" is not the leave-one-draft-
out held-out likelihood the design asked for, and it is not what would have
been reported.

## What each class's coefficients look like

**Not obtained.** The K=2 fit above recorded its log-likelihood and its class
weights but not its coefficient vectors, and re-running it would have meant
starting another fit after being told to stop. `scoring/mixture.describe_class`
is written and tested and reads a class against `pooled` rather than against
zero -- `reach` is about -9 for everybody, so a description written against
zero would say the same thing about every class -- but it has never been run
on a real fit.

This is the single largest gap in the document. A weight of 0.62/0.38 says a
split exists in the likelihood; only the coefficients would say what the split
is about, and without them there is no honest way to say whether the two
classes differ on something a person would recognise or on the fourteenth
decimal of `hype`.

## The serving design, built and then reverted

The live-inference half was written, tested and then taken back out, because
the mixture never earned the right to serve and the design has moved on.
`scoring/draft_sim.py` is byte-for-byte what it was. What was reverted:

- `archetypes=` threaded through `_run_draft`, `rollout`, `survival`,
  `search_pick` and `predict_board`, replacing only the COLD-START fallback so
  a league with per-manager fits keeps them;
- `seat_beliefs()`, which replays the picks already made into one posterior per
  slot, so a seat that has revealed itself is drafted as the class that
  explains it;
- the sampling rule: **rollouts sample, the single-pick answer averages.** One
  archetype is drawn per seat per rollout and held for the whole simulated
  draft, because averaging inside a rollout would let a seat go zero-RB on one
  pick and QB-early on the next, which is not a drafter.

One measurement from that work is worth keeping even though the code is gone,
because it corrects the design's own expectation. Sampling was expected to
widen `survival`. **It does not, measurably.** With eight seats drawn
independently, any one rollout still contains roughly the population mixture,
so an individual player's marginal survival barely moves. What does widen, a
lot, is the spread of what a WHOLE ROLLOUT looks like: the number of running
backs off the board in two rounds went from a standard deviation of 1.9 under
one blended vector to 3.5 under sampled archetypes. The uncertainty about who
these people are is real and it lives in the joint behaviour of the room, not
in any one player's availability.

## Cost, measured

`_live_features` on the real 250-player board with 29 features, under load:

| | microseconds per call |
|---|---|
| `_live_features` | 290.3 |
| one dot product (today) | 2.5 |
| K=3 dot product | 5.9 |
| `mixture_probs`, K=3 | 24.3 |
| `update_posterior`, K=3 | 36.7 |

The design's worry was that K=3 would triple the per-pick cost. It would not
have: a rollout pays nothing at all, because one archetype is sampled per seat
per rollout and the inner loop is still a single 2.5us dot product. Only the
single-pick answer pays `mixture_probs`, which is +22us on a 290us call, about
7%. The 290us baseline is measured on a loaded machine and is high; the number
to compare it against is the 228us recorded after Tier 1, and the honest
statement is that `_live_features` itself was not touched and did not change.

## What this does NOT establish

1. **Not whether drafter archetypes exist.** This is the headline and it is a
   non-answer. The leave-one-draft-out held-out likelihood for K in {1,2,3,4}
   -- the number the design specified, the number `choose_k` reads, the number
   that would have decided it -- was never obtained. 23 of 43 folds ran and
   their results were discarded by the module's own aggregation.
2. **The in-sample table is in-sample.** Training log-likelihood rises with K
   by construction. The parameter correction applied above is the crudest
   available and is not a substitute for held-out folds; it is an order of
   magnitude, not a measurement.
3. **Not that the mixture would have lost the ladder rung.** No mixture was
   ever scored on held-out human top-1 against the 0.2565 bar. It is plausible
   from the in-sample gain that it would have lost. It was not measured.
4. **Nothing about what the classes are.** No class coefficients were recorded.
   See the section above.
5. **The ICC agreement is corroboration, not confirmation.** A per-seat ICC of
   0.027 says little variance sits between seats under a linear decomposition
   of one feature. A latent-class model asks a different question -- whether
   seats fall into DISCRETE groups -- and a mixture can find structure a
   variance decomposition misses. The two pointing the same way is worth
   something; it is not the sweep.
6. **Not that the classes would have been people.** A seat is 2 to 16 picks by
   whoever sat there for an hour, and the corpus cannot follow a drafter across
   two rooms. Even a clean K=2 would have been a statement about seats.
7. **One format, one lobby, one season.** Every draft is an 8-team, 16-round
   PPR room from ESPN's public mock lobby, drafting the 2026 board.
8. **The corpus grows.** These rung numbers belong to the 43 draft ids the run
   snapshotted and printed. A rerun tomorrow measures a different corpus.
9. **Lambda was never cross-validated.** The shrinkage strength was fixed at
   10.0, the middle of `draft_model.LAMBDA_GRID`. The sweep that would have
   swept it was the same sweep that did not finish, and lambda directly
   controls how different two classes are allowed to be -- so "the classes came
   out similar" is partly a statement about a number nobody measured.
10. **The model is superseded.** The flat latent-class mixture is replaced in
    the design by a nested logit with positions as nests. Anyone re-asking this
    question should re-ask it about that model, and should read the cost note
    at the top of `pipeline/fit_archetypes.py` first.

## If someone picks this up

Three things to fix before running it again, in order of how much they cost:

1. **Stream the per-fold results to disk.** One line per fold to a JSON file.
   The 108 minutes this run spent are gone entirely for want of it, and a
   partial sweep over 23 of 43 folds would have answered the question.
2. **Make one fold affordable.** 25-40 minutes per fold is the whole problem.
   The M-step runs L-BFGS to full convergence on every EM iteration; a
   generalized EM that takes a fixed small number of steps would be a large
   saving for no change in the fixed point. Vectorizing the objective across
   choice sets was tried and measured at only 1.9x, which is not enough to
   matter and is not worth the second implementation of the likelihood.
3. **Do not run it beside the farm.** It saturates ten cores and the farm's
   throughput is the thing the corpus is made of.

The pieces are importable and need no database:

```python
from pipeline import fit_archetypes as fa, fit_prior as fp, score_ladder as sl
from scoring import mixture as mx

corpus, league = fp.open_corpus(), fp.open_league("data/leagues/X.duckdb")
obs = sl.human_observations(corpus, league,
                            sl.labelled_draft_ids(corpus,
                                                  fp.snapshot_draft_ids(corpus)))
X, chosen, groups, boards, buckets = fp.design(obs)
seats = [o.manager for o in obs.observations]          # anon:{draft_id}:{slot}

model = mx.fit_mixture(X, chosen, seats, k=2, lam=10.0, restarts=3)
print(mx.describe_class(model.betas[0], model.pooled, fa.feature_set("all")))
```
