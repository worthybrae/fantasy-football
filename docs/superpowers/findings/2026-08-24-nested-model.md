# Nested (position-then-player) opponent model

**Nested BEATS flat champion on top-1.**

Held out by draft, human picks only, standard errors clustered by draft, head-to-head delta paired by draft. The nested model is refit leave-one-draft-out; the flat champion (`scoring/human_prior.py`, `COLD_START_PRIOR`) is scored full-width on the identical picks, which if anything flatters the champion since it has already seen some of them.

Model: `scoring/nested_model.py` -- P(player) = P(position | team state) x P(player | position, board), the position factor the committed `position_model.MultinomialLogistic` (the 0.53 classifier), the within-position factor a shared conditional logit over same-position candidates. Fit separately, which is exact under the factorized (lambda=1) nested likelihood.

```
drafts 69, human picks 4938

champion  top1 0.2315 +/-0.0074   top3 0.4751   top5 0.6205   logloss  2.8985
nested    top1 0.2511 +/-0.0086   top3 0.5419   top5 0.6893   logloss  2.6463

delta top1 (nested - champion): +0.0196 +/-0.0082   delta logloss: -0.2522 (negative = nested better calibrated)

by round bucket (top1 | top5):
  bucket     n    champ t1   nest t1    champ t5   nest t5      d_t1
  early   1216      0.3816    0.3380      0.8758    0.8676   -0.0436
  mid     1715      0.2262    0.2420      0.6764    0.7300   +0.0157
  late    2007      0.1450    0.2063      0.4180    0.5466   +0.0613

VERDICT: the nested model BEATS the flat champion on top-1 (0.2511 vs 0.2315, +0.0196). The position-then-player decomposition is a genuine advance.
(champion of record: human_prior top-1 0.2595)
```

This is a measurement and a recommendation. It ships nothing: wiring the nested model into `cold_start_fits`/`_run_draft`/`survival` is a separate integration with fit/serve parity across five call sites, deliberately out of scope here.

## What the numbers say

The nested model wins on every aggregate metric on this snapshot: top-1
0.2511 vs 0.2315 (+0.0196, paired SE 0.0082, so about 2.4 standard errors),
top-3 0.5419 vs 0.4751, top-5 0.6893 vs 0.6205, and log-loss 2.6463 vs 2.8985.
The top-1 gain is significant and the log-loss gain is large. This is the
advance the position-ceiling measurement predicted: position is predictable at
~0.53, and asking "which position" first and "best player there" second beats
one flat softmax over the whole board.

## The snapshot caveat, stated plainly

The champion's number of record is 0.2595. On THIS snapshot it scores 0.2315,
not 0.2595, because the corpus grew: `human_prior.py` was fitted on 57 drafts /
3,935 human picks, and this run replays 69 human-labelled drafts / 4,938 picks
(the farm has been adding drafts). The 0.2595 was never the comparison. The
comparison is head to head on the identical 4,938 picks, where both models see
exactly the same choice sets, and there the nested model is ahead by every
metric. Scoring the champion full-width (no fold) on picks it was partly fitted
on only flatters it, so the +0.0196 is if anything a floor.

## Where the gain comes from, and where it costs

The win is entirely in the mid and late rounds, exactly where the flat model
collapsed:

  * early rounds: nested LOSES, top-1 0.3380 vs 0.3816 (-0.0436). Early, the
    best player overall and the best player at his position are usually the
    same name, and the flat model's single sharp board read wins. Splitting the
    call into two factors can only dilute a nearly deterministic early pick.
  * mid rounds: nested wins +0.0157.
  * late rounds: nested wins +0.0613 (top-1 0.2063 vs 0.1450, a 42% relative
    lift; top-5 0.5466 vs 0.4180). This is the whole thesis. Late, people pick
    a position off roster need, scarcity and positional runs, then take the
    best name there. The flat model reads one board and cannot see that; the
    nested model's position factor does.

So the decomposition does not make a uniformly better model. It trades a little
early-round accuracy, where the flat board is already near its ceiling, for a
large late-round gain, where the flat board had collapsed to 0.20 and below.
Net across the draft, and on log-loss (which the survival rollouts actually
consume), it is clearly ahead.

## Recommendation

This is a genuine advance and worth integrating. It is NOT integrated here:
wiring it into `cold_start_fits` / `_run_draft` / `survival` / `predict_board`
is a separate task with fit/serve parity risk across five call sites, and the
spec calls that out as its own phase. Recommend a follow-up that (a) serves the
two factors from one place with a parity test, as the ESPN-board rung did, and
(b) looks at the early-round regression before shipping -- a hybrid that leans
on the flat board in rounds 1-3 and the nested model afterward, or simply
letting the position factor sharpen early, may recover the early loss and keep
the late win. Either way the flat champion should not be the model past round 3.

## Reproduce

```
.venv/bin/python -m pipeline.measure_nested --league-db data/leagues/1251381776.duckdb
```

Sequential, about 11 minutes on 92 corpus drafts. A `--workers 2..7` fork pool
exists but stalled at full scale on this machine (fork + BLAS); use sequential
for a background run. `--limit N` scores the first N drafts as a smoke check.
