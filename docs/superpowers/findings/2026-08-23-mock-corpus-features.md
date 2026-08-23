# Does a prior fitted on real mock drafts beat the one fitted on one league?

**Measured 2026-08-23 on 26 ESPN mock drafts (3,267 fitted picks). Answer:
yes, and it shipped. Of the eight features Task 3 added, three earned a place
and five were cut. Read the caveats before quoting the headline: a third of
the top-1 gain and three quarters of the log-loss gain are repairs to a known
data bug in the incumbent, not better modelling.**

`scoring.draft_model.COLD_START_PRIOR` is the whole model for a league with
no draft history, which is every mock draft. Until today it was fifteen
coefficients fitted on ONE league's 696 picks by the same eight people, plus
eight zeros for columns nothing had ever measured. This is the refit.

## What was fitted, and on what

`pipeline/fit_prior.py`, run as
`make fit-prior LEAGUE_DB=data/leagues/1070645838.duckdb`. About 35 minutes;
almost all of it is 11 leave-one-draft-out backtests at 26 pooled fits each.

**The draft_id snapshot.** The corpus is being written to while a fit runs --
`make farm-mocks` joins live ESPN lobbies and adds roughly one draft every
fifteen minutes -- so the run fixes the set of drafts at its first query and
uses only those. These 26, and no others:

```
mock:0028e1dc89c88eb5  mock:00687860f7454916  mock:0233ebe6ece98629
mock:2dc424ddc7cc0f14  mock:32a44ca29679ec24  mock:3737b470da8def51
mock:4f20a6596a6a69bd  mock:564f6decbbc0047c  mock:58054006f9406c11
mock:58bc3b8ffb26c236  mock:5ad16823220a9f72  mock:5c68221be6e260f1
mock:69022319b7b17b01  mock:6ecbec89a8784261  mock:7394533b30b8e0f1
mock:8e90348faa08ec66  mock:9609995b231d3a10  mock:b4ad423e39dbd5c6
mock:cc20e63568f7f952  mock:d237677dfb1ed04c  mock:e0263539b8ad194e
mock:e12c7b7831d435d8  mock:e23a7228a859bcbb  mock:e7ada8d32b328964
mock:e7f2cb895fec713a  mock:ed5d3690611d75bd
```

23 are `pipeline.mock_backfill`'s harvest of completed mocks already on disk
(`my_slot` NULL); 3 were farmed live by `pipeline.mock_farm` (`my_slot` set).
All are 8-team, 16-round, PPR, season 2026. The corpus held 26 drafts when
this started and more since.

**Picks.** 3,328 in those drafts, 3,267 fitted. Every exclusion, counted:

| excluded | n | why |
|---|---|---|
| `my_slot` | 48 | our own seat, in the 3 farmed drafts. Part policy, part deliberate epsilon-greedy randomness, so not anyone's choice |
| `autodrafted IS TRUE` | 13 | ESPN's engine took it because a clock expired |
| player not in the pool | 0 | would be a join failure in the recording |
| player already taken | 0 | would be a duplicate pick |

`autodrafted IS NULL` is KEPT. NULL means unknown, not True: 3,200 of the
3,328 picks carry NULL because a backfilled draft comes from a `drafted`
table that never had the column, and excluding NULL would discard the corpus.
Across the snapshot the flag reads 3,200 NULL / 115 False / 13 True.

Position mix of the 3,267 fitted picks: WR 1,211, RB 935, QB 391, TE 318,
DST 207, K 205.

**The fold is the DRAFT.** The existing backtest is leave-one-season-out,
which is degenerate here -- every mock is 2026, so it would be one fold with
nothing to train on. The unit of independence is the room: one board, eight
strangers, 128 picks that all see each other. Each draft is held out in turn
and the pooled fit is rebuilt on the other 25.

The incumbent is not refitted in any fold. It was fitted on a different
league's six seasons and has never seen a pick of this corpus, so every pick
is already held out for it, and both sides score exactly the same 3,267
picks.

**The board.** `market_rank` is the pool snapshot each draft was recorded
with, renamed and not recomputed. `draft_sim.build_pool` wrote it as the same
dense cheat-sheet/FFC blend (`FFC_BLEND_WEIGHT = 0.25`) that
`draft_model._enrich_pool` builds for the historical fit, so the incumbent's
`reach`/`fall` coefficients are read against the board they were fitted
against. Re-deriving a board today would score every pick against a market
that did not exist when the pick was made.

**Player attributes** come from `scoring.player_history.attributes_as_of`,
re-keyed onto `player_id` (a corpus pool stores no names). Read from
`data/leagues/1070645838.duckdb` rather than `data/nfl.duckdb`, which a
running API process held locked; `weekly` and `players` are universal tables
that every league file carries a copy of.

## The result

Leave-one-draft-out, n = 3,267. `+/-` is the standard error of that row's own
top-1, clustered by draft.

| model | top-1 | top-5 | log-loss |
|---|---|---|---|
| ADP baseline (softmax over the board) | 0.1827 | — | 8.4174 |
| **incumbent** `COLD_START_PRIOR`, 15 fitted + 8 zeros | **0.2436** +/-0.0077 | 0.6385 | 3.5463 |
| refit, legacy 15 only | 0.2715 +/-0.0098 | 0.7098 | 2.6527 |
| refit, all 23 | 0.2792 +/-0.0090 | 0.7107 | 2.6193 |
| **refit, the 18 that shipped** | **0.2834** +/-0.0089 | 0.7086 | 2.6221 |

**`delta_top1` = +0.0398**, shipped refit against incumbent. Paired by draft
-- both vectors score the same picks in the same rooms, so what makes one room
harder than another cancels -- the standard error of that difference is
0.0077, so the gap is 5.1 standard errors, and the refit wins in 21 of the 26
drafts. That is the number the decision rule reads, and it is why
`scoring/mock_prior.py` was rewritten.

Top-5 and log-loss agree and did not decide. They are here because they were
measured.

## Where the incumbent's deficit actually comes from

**Read this before quoting the headline.** `COLD_START_PRIOR`'s `pos_DST` was
-11.14, and that number was never evidence about anybody's behavior: the fit
behind it ran on a `draft_picks` table from which `espn_league._is_real_pick`
had silently dropped every defense ever drafted (ESPN's D/ST ids are negative
by design). The model saw a defense in every choice set, saw one chosen zero
times, and drove the coefficient as negative as the fit allowed. The import
bug is fixed; the coefficient was never refitted.

This corpus is full of drafted defenses -- 207 of the 3,267 fitted picks. So
how much of the refit's win is repairing that one number rather than modelling
anything? Scoring the incumbent with only `pos_DST` changed, no refitting:

| incumbent variant | top-1 | top-5 | log-loss |
|---|---|---|---|
| as shipped, `pos_DST` -11.14 | 0.2436 | 0.6385 | 3.5463 |
| `pos_DST` -> 0.0 | 0.2436 | 0.6385 | 2.8558 |
| `pos_DST` -> +2.81 (the reconstructed estimate) | 0.2550 | 0.6382 | 2.8519 |

So of the +3.98pp of top-1, about **1.14pp is that one coefficient**, and of
the 0.92 nats of log-loss, about **0.69 nats is that one coefficient**. The
honest summary of the log-loss column is "the incumbent was catastrophically
overconfident that nobody drafts a defense", not "the new model is better
calibrated". The remaining ~2.8pp of top-1 is real refitting.

The refit puts `pos_DST` at **+0.21**, which confirms the sign flip the old
comment predicted and puts the magnitude an order of magnitude below the
+2.81 that comment guessed at.

## Per-feature: which of the eight earn a place

Each row drops one column from the 23-feature refit and re-runs the whole
leave-one-draft-out backtest. `delta_top1` is what the full model has that
the model without that feature does not; positive earns a place, at or below
zero is cut. `delta_top5` is reported for every row, always, so that reaching
for it is visible.

| dropped | top-1 | top-5 | log-loss | delta_top1 | picks | delta_top5 |
|---|---|---|---|---|---|---|
| none (all 23) | 0.2792 | 0.7107 | 2.6193 | — | — | — |
| `usage` | 0.2715 | 0.7107 | 2.6217 | **+0.0077** | +25 | 0.0000 |
| `held_at_pos` | 0.2782 | 0.7104 | 2.6194 | **+0.0009** | +3 | +0.0003 |
| `first_at_pos_round` | 0.2788 | 0.7065 | 2.6388 | **+0.0003** | +1 | +0.0043 |
| `rounds_since_pos` | 0.2795 | 0.7117 | 2.6190 | -0.0003 | -1 | -0.0009 |
| `played_share` | 0.2804 | 0.7101 | 2.6193 | -0.0012 | -4 | +0.0006 |
| `first_at_pos` | 0.2810 | 0.7101 | 2.6197 | -0.0018 | -6 | +0.0006 |
| `efficiency` | 0.2816 | 0.7095 | 2.6202 | -0.0024 | -8 | +0.0012 |
| `peak_gap` | 0.2819 | 0.7062 | 2.6215 | -0.0028 | -9 | +0.0046 |

**Ships:** `usage`, `held_at_pos`, `first_at_pos_round`.
**Cut:** `first_at_pos`, `rounds_since_pos`, `efficiency`, `played_share`,
`peak_gap`.

The rule was applied exactly as written, and it should be read with the
resolution in mind: the clustered standard error on top-1 is about 0.9pp,
which is 29 picks. Only `usage` at 25 picks is anywhere near that, and it does
not clear it either. **`held_at_pos` at 3 picks and `first_at_pos_round` at 1
pick are not distinguishable from zero.** They ship because the rule is
"positive earns a place" and bending it upward for a feature I happened to
doubt is the same failure as bending it downward for one I liked. They are the
first two to re-measure when the corpus doubles.

Two things the table settled that were flagged as entanglement in Task 3, and
settled the way that note predicted:

- `first_at_pos` is a nonlinear transform of `held_at_pos`, and only one of
  the pair survived -- the count, not the flag.
- `rounds_since_pos` is 0.0 on exactly the candidates where `first_at_pos` is
  1.0, and it was cut too. What survived of the "empty position" idea is
  `first_at_pos_round`, the flag crossed with how far through the draft it is,
  which is the only one of the three that says WHEN the empty slot starts to
  matter. Its coefficient is +3.66 and its `delta_top5` (+0.0043) is the
  second largest in the table, so it is doing real work in the ranking even
  though its top-1 contribution is one pick.

The stat profile splits: `usage` is the single most valuable of the eight,
while `efficiency`, `played_share` and `peak_gap` are cut -- the same three
that lost at n=696 in
`docs/superpowers/findings/2026-08-11-stat-profile-vs-position-dummies.md`.
That finding rejected all four against the position dummies. This one does
not overturn it: the dummies were never removed here, and what is being
measured is whether the profile adds anything ON TOP of them. On four times
the sample, one of the four does.

## Where the accuracy moved

Incumbent against the shipped refit, same held-out picks. Buckets are
`draft_model._round_bucket`: early is rounds 1-3, mid 4-8, late 9-16.

| bucket | n | incumbent top-1 | refit top-1 | incumbent top-5 | refit top-5 |
|---|---|---|---|---|---|
| early | 615 | 0.4276 | **0.4358** | 0.8878 | **0.9171** |
| mid | 1,025 | **0.3454** | 0.3239 | 0.7805 | **0.8146** |
| late | 1,627 | 0.1100 | **0.2004** | 0.4548 | **0.5630** |

Essentially the whole top-1 gain is late-round, where half the picks are and
where kickers, defenses and roster-filling live. **The refit is worse than the
incumbent at mid-round top-1**, by 2.15pp over 1,025 picks, while being better
at mid-round top-5. That is not hidden by the headline so much as averaged
away by it, and it is the honest counterpart of the 2026-08-11 finding's
observation that late rounds are where positional structure pays.

## The vector that shipped

`scoring/mock_prior.py`, generated. Cut features carry exactly 0.0 -- they
were not in the fit, so their coefficients are absent rather than small.

| feature | incumbent | refit |
|---|---|---|
| `reach` | -8.088 | -11.239 |
| `fall` | +3.801 | +2.600 |
| `pos_RB` | +0.641 | -0.142 |
| `pos_WR` | +0.341 | -0.658 |
| `pos_TE` | +0.897 | +0.139 |
| `pos_K` | +3.418 | +0.603 |
| `pos_DST` | -11.141 | +0.214 |
| `qb_early` | +1.482 | +3.526 |
| `te_early` | +0.324 | +1.764 |
| `need` | +2.246 | +0.630 |
| `run` | +0.596 | +1.783 |
| `age` | +0.050 | -0.037 |
| `no_track_record` | -0.331 | -0.529 |
| `hype` | -0.063 | -0.020 |
| `trend` | -0.004 | -0.065 |
| `held_at_pos` | 0.0 (unmeasured) | -0.049 |
| `first_at_pos` | 0.0 (unmeasured) | 0.0 (cut) |
| `rounds_since_pos` | 0.0 (unmeasured) | 0.0 (cut) |
| `first_at_pos_round` | 0.0 (unmeasured) | +3.659 |
| `usage` | 0.0 (unmeasured) | -0.035 |
| `efficiency` | 0.0 (unmeasured) | 0.0 (cut) |
| `played_share` | 0.0 (unmeasured) | 0.0 (cut) |
| `peak_gap` | 0.0 (unmeasured) | 0.0 (cut) |

Two readings worth writing down rather than leaving for someone to infer.
`need` fell from +2.25 to +0.63 while `first_at_pos_round` came in at +3.66:
the two describe the same instinct, and the corpus prefers the version that
knows what round it is. And `usage` is NEGATIVE -- these rooms take the
lower-volume player among comparable ones at the same position, which is the
opposite of what the feature was proposed on. It earns its place by predicting
picks, not by being the story anyone expected.

## What this does NOT establish

1. **Not that these are people.** The corpus gate finding established only
   that the picks are not a ranking-follower: they deviate further from ESPN's
   own board than from consensus ADP, in both the backfilled and farmed
   populations. A need-aware drafting engine with positional logic would
   produce that too. The `autodrafted` flag cannot settle it either -- ESPN
   emits no AUTODRAFT frame for the computer seats it uses to fill a thin
   room, so 115 explicit Falses are not 115 humans.
2. **The 18-feature set was selected on the same folds it is reported on.**
   The clean, selection-free number is the all-23 row, 0.2792, and it also
   beats the incumbent by 3.56pp -- so the decision to ship does not rest on
   the selection. But 0.2834 is optimistic by whatever the selection is worth,
   and a nested selection inside each fold (the `_nested_subset_gain`
   discipline) was NOT run: it is 26 x 9 inner backtests, hours of compute for
   a correction smaller than the gap being measured.
3. **A third of the top-1 win and three quarters of the log-loss win is a
   data-bug repair**, quantified in the `pos_DST` table above. Anyone
   comparing this model to the previous one on log-loss is mostly comparing it
   to a coefficient that was fitted to an event the training data made
   unobservable.
4. **The refit is worse at mid-round top-1.**
5. **One format, one lobby, one season.** Every draft is an 8-team, 16-round
   PPR room from ESPN's public mock lobby, drafting the 2026 board. Nothing
   here says a 10- or 12-team league, a half-PPR league, or a real league with
   money on it drafts this way. The prior is applied to all of them.
6. **The corpus grows.** These numbers belong to the 26 draft_ids listed
   above. A rerun tomorrow measures a different corpus and may reach a
   different feature set.
7. **Not comparable to the 2026-08-11 numbers.** That finding's 0.2356 was six
   seasons of one league, leave-one-season-out, ~200-300 candidates per choice
   set. This is 26 mock drafts, leave-one-draft-out, ~250 candidates falling to
   ~125. Compare rows within a table, never across the two documents.
8. **Nothing about per-manager fits.** `fit_all`, `write_profiles`,
   `manager_tendencies` and `make fit-managers` are untouched. This vector is
   the fallback when a league has no history, and the anchor `select_lambda`
   shrinks a personal fit toward. A league with six seasons of its own picks
   is unaffected except through that anchor.
9. **The attribute join was read from one league file.** `weekly` and
   `players` are universal tables and every league file holds the same copy,
   but "every copy is identical" was not verified across all 47 files.

## Reproducing

```bash
# The full run. ~35 minutes. Writes scoring/mock_prior.py only on a top-1 win.
make fit-prior LEAGUE_DB=data/leagues/1070645838.duckdb
```

There is no `--force`. A losing refit prints its numbers, writes nothing and
exits 1; shipping one anyway means editing the rule in `fit_prior.main`, which
is a diff somebody has to justify.

To measure a different feature set without shipping anything, the pieces are
importable:

```python
from pipeline import fit_prior as fp
corpus, league = fp.open_corpus(), fp.open_league("data/leagues/1070645838.duckdb")
obs = fp.build_corpus_observations(corpus, league, fp.snapshot_draft_ids(corpus))
X, chosen, groups, boards, buckets = fp.design(obs)
fp.leave_one_draft_out(X, chosen, groups, buckets=buckets,
                       keep_idx=fp.feature_indices([...]))
fp.ablate(X, chosen, groups)          # defaults to UNMEASURED_FEATURES
```

`ablate`'s default is `UNMEASURED_FEATURES`, deliberately -- `draft_model`'s
own `ablation()` defaults to the four older features, and calling it bare here
would print a clean-looking table that tested none of these eight.
