# Draft Model Calibration

Date: 2026-08-10

## Goal

Make the simulator's predictions match how this league actually drafts. Three
defects, found by reading a real run against real data, currently make the
predicted board wrong in ways a user spots immediately.

## The evidence

A 50-rollout run at slot 4, against the league's own imported history:

**The model cannot tell elite players apart.** Pick 1 came back Puka Nacua at
12%, Jahmyr Gibbs at 16%, Amon-Ra St. Brown at 10%, Ja'Marr Chase at 8% — a
near coin flip across the top of the board. ESPN's own 2026 ranking has Gibbs
at ADP 1.7, the number one player overall. A real first pick is not a coin
flip.

**The board then shows the wrong name anyway.** Gibbs was the most likely
player at pick 1 (16% against Nacua's 12%) and still did not appear there. He
was placed at pick 13, where he scored 40%. Three of the first sixteen cells
show a primary that the hover reports as less likely than one of its own
alternates.

**The simulated version of the user drafts a quarterback first.** The slot-4
column opened with Josh Allen, market rank 24, at pick 4.

## Root causes

**Linear-in-rank features.** `reach` is `max(0, adp_rank - pick) / teams`.
The gap between the first and fifth ranked players is 0.5 feature units; the
gap between rank 100 and rank 140 is 5.0. With a fitted coefficient near
-0.83, the entire top ten sits within roughly 2x of each other in probability,
while deep-bench noise dominates the feature's range. Draft behavior is
roughly linear in log rank, not rank.

This is the calibration residual parked during the previous branch's review,
now confirmed: the model was fitted on `historic_adp` at 157-249 players per
season, while `build_pool` ranks 525 market-known players out of 702 board
rows.

**A global assignment objective.** `_assign_primaries` minimizes total
`-log(prob)` across all cells, maximizing the joint likelihood of the whole
board. That is a coherent statistical object and the wrong one for a draft
board: it spends a high-confidence player on the single cell where he scores
best and backfills earlier cells with less likely names.

**Raw points in the self-policy.** `roster_value` sums the projected points of
the best starting lineup. A quarterback outscores every running back and
receiver in raw points, so a one-ply greedy takes the top quarterback early.
This is the error value over replacement exists to prevent, and the board
already computes replacement levels it was not using. It matters beyond the
grid: `search_pick` runs this policy in every rollout, so the ΔEV column is
computed against a version of the user that drafts badly.

## Decisions

| Question | Decision |
|---|---|
| Market reference | ESPN, because the league drafts on ESPN off ESPN's board |
| Rank scale | Log rank, replacing linear |
| New player features | Age, volatility, hype, trend — each kept only if it earns its place |
| League positional bias | A pooled term, not eight per-manager coefficients |
| Dedupe objective | Greedy in draft order, earliest pick wins |
| Self-policy value | Points above replacement, not raw points |
| Validation | Leave-one-season-out over all six seasons, per-round accuracy, ESPN baseline |
| Scope | Model only. The grid's order editor is a separate spec. |

---

## Part 1: The market reference moves to ESPN

The league drafts on ESPN, off ESPN's rankings. Fitting against Fantasy
Football Calculator — a 12-team consensus from another site — measures reach
against a board the managers never saw.

`pipeline/sources.py::fetch_espn_adp(year)` is already parameterized by year
and returns usable data for 2020 through 2025. `pipeline/import_league.py`
gains a per-season ESPN pull alongside the existing FFC one, writing a
`historic_espn` table: `season, espn_id, espn_name, position, espn_adp,
espn_ppr_rank`.

**`espn_ppr_rank` is the primary signal, not `espn_adp`.** Verified per
season: 2026 and 2024 return sane ADP, 2023 and 2021 return sane ADP, and
**2025 returns 170.0 for every player** — a reset value, not a ranking. Its
`espn_ppr_rank` that year is fine (Barkley 4, Gibbs 5, Lamb 7). So the
importer ranks on `espn_ppr_rank`, uses `espn_adp` only as a tiebreak, and
validates each season before accepting it: a season whose ADP column is
constant, or whose rank column is missing, is reported and its ADP ignored.

The existing `historic_adp` (FFC) stays. Two independent rankings per season
is what makes the hype feature below measurable, and keeping FFC means the
change in reference is itself testable rather than assumed.

Name matching goes through `scoring.board.adp_match_key`, the same join key
the rest of the pipeline uses.

---

## Part 2: The features

### Log rank replaces linear rank

`reach` and `fall` become log-scale:

```
reach = max(0, log1p(market_rank) - log1p(pick_no))
fall  = max(0, log1p(pick_no) - log1p(market_rank))
```

`log1p` rather than `log` so rank 1 and pick 1 are defined and finite. The
division by `teams` goes away — it was a rounds-based scaling that no longer
means anything on a log axis, and the coefficient absorbs the units.

This inverts the pathology: the distance from rank 1 to rank 5 becomes larger
than the distance from rank 100 to rank 140, which is how drafts behave.

It also makes the fitted-versus-simulated population mismatch far less
damaging. A log scale compresses the tail, so ranking 525 players where the
fit saw 200 shifts the deep end by a bounded amount instead of multiplying it.

### Four new player-at-pick-time features

Every one is restricted to what was knowable on that draft day. No lookahead:
a feature for a 2022 pick may use 2016-2021 data and the player's age in 2022,
nothing later.

- **`age`** — the player's age at that season's draft, from
  `players.birth_date`, already ingested for stat twins. Centered within
  position so the coefficient reads as "younger than typical for the position"
  rather than tracking that tight ends last longer than running backs.
- **`volatility`** — the standard deviation of the player's points per game
  across his prior seasons, plus his games-missed rate. A player with no prior
  seasons at all gets a separate `no_track_record` indicator rather than a
  fabricated zero, because "unknown" and "steady" are opposite things.
- **`hype`** — market rank minus a rank derived from the player's own prior
  production. When the market is far ahead of the stats, taking him is a leap
  of faith. This is the "projected points way above historical" idea, using
  the market as the projection, since it is the only projection that existed
  at the time.
- **`trend`** — the slope of the player's points per game across his prior
  seasons. Rising, flat, or falling.

### One league-wide positional bias term

Across 712 picks, measure how far this league drafts each position ahead of or
behind ESPN's ranking, by round. If the league takes quarterbacks two rounds
early, every manager inherits that, and forcing it through eight separate
per-manager coefficients spends scarce data on a shared effect.

This is a pooled term, fitted once over all managers, and it is worth showing
directly regardless of the model: "your league drafts QBs 18 picks ahead of
ESPN, TEs 12 picks behind" is a fact about the league worth knowing on draft
night.

### The parameter budget is the real constraint

Each manager currently carries 12 coefficients against roughly 105 observed
picks. The additions above would take that to about 16. That ratio is already
thin, and a feature that does not pay for itself is worse than absent: it fits
noise and degrades every other coefficient.

So each new feature is added, measured, and kept only if held-out accuracy
improves. Part 5 is the mechanism. The expected outcome is that at least one
of the four is cut, and reporting which is part of the deliverable.

---

## Part 3: The dedupe objective

`_assign_primaries` stops solving a global assignment and instead walks the
picks in draft order, giving each cell its most likely player not already
claimed by an earlier cell.

The reasoning is about what the artifact is for. A draft board is read top to
bottom, and the earliest picks are the ones that must be right — a wrong name
at pick 1 discredits the whole grid in a way a wrong name at pick 90 does not.
The global objective treats all 120 cells as equally worth optimizing and
happily trades pick 1 away to improve pick 13.

Greedy in draft order is not the maximum-likelihood board, and that is the
correct trade. `scipy.optimize.linear_sum_assignment` is no longer needed;
`ASSIGNMENT_MISS_COST` and the claimed-aware fallback go with it.

A player who could genuinely go at several picks still appears in the hover
alternates for all of them, which is what they were always for.

---

## Part 4: The self-policy values replacement, not raw points

`_greedy_choice` currently ranks candidates by the increase in
`roster_value`, which sums raw projected points. It gains a
positional-replacement baseline: a candidate's value is his projected points
minus the replacement level at his position.

`LeagueSettings.replacement_ranks` already derives those levels from the
league's roster shape, and the board's VOR column already uses them. The
simulator simply was not.

For the 2026 league that means a 380-point quarterback against a replacement
near 300 is worth 80, while a 300-point running back against a replacement
near 150 is worth 150 — so the running back goes first, which is what real
drafters do and what the board's own ranking already says.

`roster_value` itself is unchanged. It is the objective the search reports,
and starting-lineup points remain the right thing to report; only the
in-rollout choice policy changes.

---

## Part 5: Validation is the gate

Every change above is a hypothesis. None of them ship on argument.

**Leave-one-season-out over all six seasons.** The current backtest holds out
2025 alone — 112 picks. Rotating through 2020-2025 gives 712 evaluations,
which is the difference between a number that moves on noise and one worth
acting on.

**The baseline moves to ESPN as well.** Changing what the model sees to ESPN
while leaving the baseline on FFC would confound a better model with a worse
yardstick. Same reference both sides.

**Accuracy reported per round.** A model that nails round 1 and is random by
round 6 scores the same top-1 as one that is mediocre throughout, and they are
different tools on draft night. Per-round numbers also say where the remaining
error lives.

**A feature ablation table.** Each new feature is measured with and without,
and the report states which were kept and which were cut, with the numbers.

**The gate.** The current model scores top-1 16%, top-5 53%, log-loss 3.360
against an ADP baseline of 9.495. The reworked model must beat 16% / 53% on
the six-season rotation. If it does not, it does not ship, and the finding is
that this league drafts less predictably than it appears — which is worth
knowing and worth saying, rather than shipping a prettier grid over the same
flat probabilities.

---

## Testing

All tests offline, seeding DuckDB under pytest's `tmp_path` with literal
fixtures, matching the existing suite.

- ESPN historical parsing, including the corrupt-ADP season: a season whose
  `espn_adp` is constant must be detected, reported, and its ADP ignored while
  its ranks are still used.
- Log-rank features: hand-computed values at the boundaries, and an explicit
  assertion that the rank-1-to-5 distance now exceeds the rank-100-to-140
  distance, which is the whole point.
- Each new feature: computed from a fixture with known history, and a test
  that it uses no data from after the draft it describes.
- `no_track_record` is set for a player with no prior seasons, and
  `volatility` is not silently zero for him.
- The dedupe: pick 1 gets the most likely player at pick 1, and a player
  displaced from a later cell still appears in that cell's alternates.
- The self-policy: given a quarterback and a running back whose raw points
  favor the quarterback but whose replacement-adjusted values favor the
  running back, the greedy takes the running back.
- The validation harness: leave-one-season-out covers every season, per-round
  numbers sum to the overall number, and the ablation table has a row per
  feature.

## Out of scope

- The grid's draft-order editor. Separate spec, built after this one lands.
- Scoring managers by how their picks actually performed. Considered and
  explicitly set aside in favour of validating against the drafts themselves.
- Any model of in-draft trades, keepers, or auction formats.
- Changing `roster_value`'s definition. Only the in-rollout choice policy
  changes.

## Risks

**The features may not survive.** Roughly 105 picks per manager against 16
parameters is thin, and the honest outcome may be that several additions are
cut. The ablation table is designed to make that visible rather than
embarrassing.

**The league may be less predictable than hoped.** If leave-one-season-out
accuracy stays near today's numbers after all of this, the conclusion is about
the league, not the code. The gate in Part 5 exists so that conclusion gets
stated instead of buried.

**Changing the market reference changes every downstream number.** `Avail%`,
`ΔEV`, and the grid all move. That is intended, but it means the before-and-
after cannot be compared cell by cell — only through the backtest.

**ESPN historical data may drift.** The 2025 ADP corruption was found by
inspection, not by a guard. The per-season validation in Part 1 exists so the
next such surprise is reported rather than silently fitted.
