# Does a player's stat profile describe a manager better than his position?

**Measured 2026-08-11. Answer: no, at this league's sample size. The position
dummies stay. Do not re-litigate this without re-measuring.**

## The question

The five position dummies (`pos_RB`, `pos_WR`, `pos_TE`, `pos_K`, `pos_DST`)
were the model's entire account of what kind of player a manager likes, and
three of them could only say "leans RB" or "fades WR". The proposal was to
replace those three with the *stat profile* of the players each manager takes,
computed from what the player had actually done going into that draft:

| column | definition | units |
|---|---|---|
| `usage` | opportunities (carries + targets + pass attempts) per game | per game |
| `efficiency` | PPR points per opportunity | points |
| `played_share` | games played / games his team played | 0-1 |
| `peak_gap` | best career PPG minus last season's PPG | PPG |

All four read only seasons strictly before the draft, and all four are centred
within position before reaching the fit, so a coefficient reads as a preference
between comparable players rather than a preference between positions.

They are computed in `scoring/player_history.attributes_as_of` and carried on
every pool. **Nothing reads them.** They are kept so that re-measuring on a
seventh season is a change to `FEATURE_NAMES`, not a rebuild of the join.

## Result 1: it predicts picks slightly worse

Six-season leave-one-season-out, n=696 picks.

| feature set | top-1 | top-5 | log-loss |
|---|---|---|---|
| **shipped: 15 features, all five dummies** | **0.2356** | 0.5848 | 2.8681 |
| dummies stripped, nothing added | 0.2227 | 0.5733 | 2.8806 |
| RB/WR/TE replaced by the profile (16) | 0.2198 | **0.5876** | **2.8636** |
| both, dummies and profile (19) | 0.2241 | 0.5833 | 2.8638 |
| profile only, no dummies at all (14) | 0.2313 | 0.5862 | 2.8883 |

The stat profile is better calibrated (log-loss 2.8636 against 2.8681) and
slightly better at top-5, but worse at naming the exact pick. `FEATURE_NAMES`
is decided by `delta_top1` — the rule stated in `draft_model.py`, written
precisely so a metric cannot be reached for only when it defends the
conclusion — and on that metric the incumbent wins.

Keeping both is not a compromise worth making: 19 features scores 0.2241, below
the dummies alone, and buys no log-loss over the profile alone.

The loss is not spread evenly. By round bucket:

| bucket | n | shipped top-1 | profile top-1 | shipped top-5 | profile top-5 |
|---|---|---|---|---|---|
| early (rounds 1-3) | 144 | 0.4792 | 0.4653 | 0.8681 | **0.8750** |
| mid | 237 | 0.2068 | 0.2068 | **0.6034** | 0.5949 |
| late | 315 | **0.1460** | 0.1175 | 0.4413 | **0.4508** |

Nine of the eleven lost picks are late-round, where the dummies carry the
kicker and defense structure that `market_rank` cannot: ESPN's cheat sheet
never ranks a defense, so a DST's rank is wherever the FFC fallback put it.
Early and mid are a wash. That is a real argument for shipping the profile
anyway if late-round precision is worthless to you — it was considered and
declined, because the decision metric is the decision metric.

## Result 2: it does not separate these managers

The stronger claim was that the stat profile would be *more telling* about who
drafts what. It is not, and this is the more important half of the finding,
because it does not depend on which model consumes the columns.

For every pick, take the chosen player's profile value centred against the
other players at his position still on the board, then average per manager and
subtract the league mean. Deviation ± one standard error, n ≈ 86 picks each:

| manager | usage | efficiency | played_share | peak_gap |
|---|---|---|---|---|
| ESPNFAN3676948926 | -0.028 ± 0.324 | -0.012 ± 0.020 | +0.013 ± 0.015 | -0.382 ± 0.226 |
| John Titolo | -0.010 ± 0.312 | -0.028 ± 0.022 | -0.032 ± 0.022 | -0.058 ± 0.308 |
| Lane Bohman | +0.016 ± 0.282 | +0.022 ± 0.023 | **+0.048 ± 0.013** | +0.148 ± 0.274 |
| MaxMandia | +0.334 ± 0.318 | +0.010 ± 0.024 | +0.024 ± 0.018 | +0.306 ± 0.333 |
| Reed1998 | -0.671 ± 0.460 | -0.013 ± 0.018 | -0.031 ± 0.019 | +0.228 ± 0.348 |
| espn26772796 | +0.411 ± 0.342 | +0.017 ± 0.022 | -0.007 ± 0.018 | -0.263 ± 0.323 |
| espn45456832 | -0.066 ± 0.309 | +0.002 ± 0.022 | -0.020 ± 0.021 | +0.162 ± 0.344 |
| jtague99 | +0.022 ± 0.318 | +0.001 ± 0.022 | +0.005 ± 0.019 | -0.118 ± 0.263 |

**One cell of 32 clears two standard errors** — Lane Bohman on `played_share`,
at 3.7 se. Testing 32 cells, roughly 1.5 hits at 2 se is what chance produces,
so this is one marginal result and not a pattern. Reed1998's low-usage lean,
the most interesting number in the table, is 1.5 se. espn26772796's high-usage
lean is 1.2 se.

A `profile_lean` metric was written for `manager_tendencies` and then removed
rather than shipped: putting 32 numbers on eight cards when one of them is
distinguishable from zero would be presenting noise as a scouting report.

## A correction to an earlier reading

The per-manager held-out gains do improve under the profile — mean -0.0118 to
-0.0030, with two managers above zero instead of one. That is **not** evidence
of signal. Most of the improvement is the ridge shrinking all the way to
pooled: a gain of -0.00002 means the personal fit is numerically identical to
the pooled one, which scores better than distinguishing badly. It is graceful
degradation, not information. `_heldout_gain` near zero should always be read
this way before it is read as a finding.

The one number that is not shrinkage is ESPNFAN3676948926 at +0.0083 nats/pick
on a profile-only fit — 0.73 nats total across 88 picks, and his four
descriptive cells above are all inside 2 se.

Under the position dummies, John Titolo was the standout (+0.0527 nested,
`reach+pos_skill` selected in 6 of 6 folds). Under the profile his selection
scatters across `reach+fall` / `reach+run` and lands at +0.0018. **Titolo's
signal was positional**, which is direct evidence against the premise that
position is the less informative description.

## What actually binds

Not the feature set. Six drafts is ~86 picks per manager, and no choice of
columns changes that. What does separate these managers is already on their
card and needs no fit at all: first-pick position counts, reach against the
board by round bucket, and the round they first take each position — all
descriptive statistics over real picks, in `manager_tendencies`.

## Reproducing

The columns are live, so the model comparison is a `features=` mask away:

```python
from scoring.draft_model import backtest, FEATURE_NAMES
backtest(conn)                                    # shipped
backtest(conn, features=[f for f in FEATURE_NAMES
                         if f not in ("pos_RB", "pos_WR", "pos_TE")])
```

Adding the profile back to `FEATURE_NAMES` additionally requires the matching
columns in `draft_sim._live_features` and `SimPool` — `feature_matrix` and
`_live_features` are two implementations of one definition, and a coefficient
fitted against one and applied to the other is silently wrong.
