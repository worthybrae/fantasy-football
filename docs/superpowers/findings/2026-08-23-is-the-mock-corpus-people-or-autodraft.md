# Are the harvested mock drafts people, or ESPN's autodraft?

**Measured 2026-08-23 on the 23 backfilled drafts (2,944 picks). Answer: not
autodraft. The prior refit is worth doing.**

This is the question the spec's phase 0 was built to answer by sitting in a
lobby and counting seats. It is answerable from data already on disk instead.

## The number

Mean `|pick_no - adp_rank|` across all 2,944 mock picks: **12.19**
(median 6.00). ESPN's autodraft follows a ranking, so a corpus dominated by
it would sit near zero.

By round bucket:

| bucket | n | mean deviation | median |
|---|---|---|---|
| early (rounds 1-3) | 552 | 2.61 | 1.00 |
| mid (4-10) | 1,288 | 7.47 | 6.00 |
| late (11-16) | 1,104 | 22.49 | 13.00 |

## Why the shape of that table is the real evidence

A single average could be explained away as a bad ADP join — if `adp_rank`
were matched to the wrong players, every pick would look like a reach.

The early-round rows rule that out. At rounds 1-3 the median deviation is
**1.00**: the join is tight exactly where consensus is tight and everyone
takes the same fifteen players. A broken join cannot produce that. So the
late-round spread of 22.49 is real behavior, not noise in the match.

That is also the correct shape for human drafting. The top of the board is
consensus; the bottom is where preference, reaching for a sleeper, and
position runs live.

## Corroborating structure

First round across 23 drafts: **92 WR, 91 RB, 1 QB**. A near-even RB/WR split
with QB essentially absent is a real strategic distribution, not a ranking
read off in order.

Position share by bucket:

| bucket | RB | WR | QB | TE | DST | K |
|---|---|---|---|---|---|---|
| early | 55.6% | 36.4% | 1.1% | 6.9% | — | — |
| mid | 26.1% | 49.6% | 13.8% | 10.2% | 0.2% | 0.1% |
| late | 18.3% | 22.3% | 15.2% | 10.8% | 16.8% | 16.7% |

Kickers and defenses appear only in the late bucket, at ~17% each. That is
the late-round structure the existing position dummies carry and that
`market_rank` cannot represent — and it is the exact structure the
2026-08-11 stat-profile finding lost nine of its eleven picks on.

## Consequence

Task 4 (refit the cold-start prior on this corpus) is GO. The gate the
controller set before starting — "if the corpus is autodraft-dominated, do
not ship a refitted prior" — is not tripped.

One caveat to carry into Task 4: ESPN's autodraft follows ESPN's own
rankings, which are not the `adp_rank` on the corpus pool, so some part of
the 12.19 is source disagreement rather than choice. The early-round median
of 1.00 bounds how large that part can be. It does not eliminate it.

---

# Follow-up: is the room-filling engine just reading a ranking?

**Measured 2026-08-23 06:30, after the first live-farmed draft landed.**

The first farm run reported that every 8-team PPR room in the lobby overnight
was 0/8 or 1/8 joined, so ESPN's own engine filled most seats. It also found
that ESPN emits no `AUTODRAFT` frame for those seats (32 False, 96 unknown,
0 True across 128 picks), so the `autodrafted` flag cannot separate them.
That puts the whole corpus in question, including the 23 backfilled drafts,
which came from the same lobby.

Deviation from consensus ADP cannot settle it. The farmed draft measured 11.67
against the backfilled 12.19 — nearly identical — but that agreement has two
readings: the engine drafts realistically, OR both populations are the same
engine. The metric cannot tell them apart, because ESPN's engine would follow
ESPN's ranking, not consensus.

## The test that does separate them

Measure each pick against **ESPN's own board** (`espn_adp.espn_ppr_rank`).
A seat reading that ranking down the page sits near zero.

| population | vs ESPN's own board | vs consensus ADP |
|---|---|---|
| backfilled, 23 drafts, n=2,757 | mean **17.66**, median 7.0 | mean 10.24 |
| farmed, 1 draft, n=120 | mean **16.81**, median 6.0 | mean 9.72 |

Both populations deviate MORE from ESPN's ranking than from consensus. That is
the opposite of what a ranking-follower produces, and it rules out the failure
this whole exercise was most at risk of.

Per seat in the farmed draft, against ESPN's board: slots 2-8 range 12.87 to
16.13 (medians 2 to 8). Our own seat is 28.93, which is the epsilon-greedy
policy's random picks showing up exactly where they should.

## What this does and does not establish

**Does:** the picks are not a trivial function of either board. There is
structure for the model to learn, and the backfilled and farmed populations
are behaviorally the same, so they can be pooled.

**Does not:** prove these are humans. A need-aware drafting engine with
positional logic would also produce this. The honest claim is "not a
ranking-follower", not "people".

The practical consequence is the same either way: Task 4's refit is worth
running, and its `delta_top1` backtest against the incumbent prior remains the
real decision. This measurement only establishes that the corpus is not
degenerate.

**Caveat on n.** Only 225 players carry an `espn_ppr_rank`, so 94% of picks
join and the late rounds thin out. The farmed sample is one draft, 15 picks
per seat. The backfilled figure is the solid one.
