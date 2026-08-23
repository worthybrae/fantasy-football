# Predicting the next pick from a corpus of mock drafts

**Design, 2026-08-23.** Two phases. Phase 1 gets real mock drafts into the
corpus, refits the model's cold-start prior on them, extends the feature set,
and pulses the available list red by how likely a player is to be gone. Phase
2 builds the path map on top of the refitted prior.

## The problem

Every opponent in a mock draft is a stranger. `draft_model.fit_all` falls back
to `cold_start_fits()` when a league has no history, and that returns a single
hardcoded vector, `COLD_START_PRIOR` -- fifteen coefficients fitted once on one
league's six seasons, 696 picks. In a mock draft, which is what the landing
page sells as the free tier, that one vector *is* the model. Every prediction
the room shows, every survival number, every recommendation, traces back to it.

We have no way to make it better, because we have no data on how strangers
draft. The corpus that would hold that data already exists and is empty of
mocks.

## What already exists

Naming this precisely, because most of the machinery for this feature is
already built and the work is smaller than the ask sounds.

- **`scoring/draft_model.py`** -- a conditional-logit choice model. Each
  historical pick is one choice from the pool available at that moment.
  `FEATURE_NAMES` has fifteen entries: `reach`/`fall` (log market rank against
  pick number), five position dummies, `qb_early`/`te_early`, `need`, `run`,
  `age`, `no_track_record`, `hype`, `trend`. Fitted per manager, shrunk toward
  a pooled fit. Backtested leave-one-season-out, and `FEATURE_NAMES` is decided
  by `delta_top1`.
- **`scoring/draft_sim.py`** -- `_run_draft` rolls a whole draft forward from
  any state. `survival()` counts, over many rollouts, how often each available
  player is still there at a target pick; `predict_board()` counts who goes at
  each pick with alternates.
- **`pipeline/draft_log.py`** -- the cross-league corpus, in its own DuckDB
  file so it accumulates rather than being copied per league. It already
  defines `SOURCE_MOCK = "mock"`, and nothing writes it. Its docstring states
  the intent outright: "the answer to most of its open questions is more
  drafts, from anywhere."
- **`pipeline/draft_socket.py`** -- connects to an ESPN draft websocket with
  nothing but cookies, no browser. `/api/live/select` sends a `SELECT` frame
  and waits for ESPN's own `SELECTED` to confirm. `/api/live/autodraft`
  toggles autodraft the same way. `DraftListener` already hears ESPN's
  `AUTODRAFT` frames for every team in the room.
- **`web/src/components/draft/AvailableList.tsx`** -- renders `survive_pct`
  as a column, colored by `riskTone`. `survive_pct` is 0-100, and is `null`
  (not zero) when there is no roster to survive for.

So: the model, the rollouts, the corpus schema, the socket, the pick
submission and the survival number are all in place. What is missing is mock
drafts in the corpus, a prior fitted on them, four features, and an animation.

## Phase 0: the spike, which gates everything

`draft_socket.socket_url` mints a URL from `leagueId`, `teamId` and a
`draftSecurity` nonce. All three presuppose a team that already exists. The
mock draft lobby hands you one by some path this codebase has never touched.

**The spike:** drive `https://fantasy.espn.com/football/mockdraftlobby` by
hand with a proxy recording traffic, find the create-or-join call, and confirm
it replays from Python using only the cookies in `STATE_PATH`.

**Also count the humans.** Sit in one full mock and count how many seats are
autodrafted. If ESPN's mock lobbies are mostly empty seats on autodraft, then
a corpus farmed from them measures ESPN's autodraft algorithm, which follows
ADP, and refitting the prior on it would teach the model that everyone follows
the market. That is worse than the prior we have. This number decides whether
phase 1 is worth building, so it is measured before phase 1 is built, not
after.

**If the join does not replay:** the bot dies and the corpus fills only from
mocks played by hand. Everything else in this design is unchanged; it just
accrues one draft at a time instead of fifty a night.

## Phase 1

### 1.1 Recording a mock into the corpus

`draft_log.record()` already takes a `DraftRecord` and writes it idempotently
by `draft_id`, and `draft_id_for` already gives a mock its start time so two
mocks in the same lobby on the same day are two drafts. `_PICK_COLUMNS`
already carries `owner_key` and `is_anonymous`, and `anonymous_key` already
gives a seat nobody can be identified in a key unique to the draft.

One schema change: **`draft_log_pick` gains an `autodrafted BOOLEAN` column.**
A seat on autodraft is not a manager making choices, and a fit that treats
those picks as human choices is fitting ADP to itself. `DraftListener` already
records ESPN's per-team `AUTODRAFT` frames, so the flag is available; it just
has to reach the corpus.

`ensure_schema` uses `CREATE TABLE IF NOT EXISTS` and has no migration path,
by design -- these tables accumulate and must never be replaced. Adding the
column means an `ALTER TABLE draft_log_pick ADD COLUMN IF NOT EXISTS
autodrafted BOOLEAN` alongside the existing DDL, which DuckDB supports and
which is safe to run against a corpus that already has rows.

Our own seat is identified by `DraftRecord.my_slot`, which already exists. Its
picks are part policy and part randomness and are not a human's choices, so
they must be excludable from the fit -- while remaining in the corpus, because
every other seat's picks are conditioned on the board our picks shaped.

### 1.2 The farm loop

New module `pipeline/mock_farm.py`, run as `make farm-mocks N=50`. One
iteration is one draft:

1. **Join** a mock lobby, yielding `league_id`, `team_id`, `season`. This is
   the only genuinely new integration; phase 0 establishes whether it exists.
2. **Connect** through `draft_socket.run_socket_listener` with cookies from
   `STATE_PATH`. Unchanged from the live path.
3. **Play**, epsilon-greedy, on our turns: with probability `1 - epsilon` the
   tool's own top-ranked candidate, otherwise a uniformly random legal pick.
   Submitted as a `SELECT` frame and confirmed on `SELECTED`, the same round
   trip `/api/live/select` makes.
4. **Record** on completion, as `DraftRecord(source=SOURCE_MOCK, ...)`, with
   the accumulated picks, the pool snapshot, `my_slot`, and each pick's
   `autodrafted` flag.
5. **Repeat**, with a delay between drafts.

Epsilon is a knob with a real trade-off in both directions, so it is exposed
rather than baked in. Higher epsilon reaches roster states an ADP-follower
never reaches, which is the coverage the fit wants. Lower epsilon keeps the
board our opponents are reacting to closer to a real draft, which is the data
the fit is actually collecting. `epsilon = 0.2` is the starting value; nothing
about the design depends on it being right.

Recording happens on completion rather than per pick, but `record()` is
idempotent by `draft_id`, so recording a partial draft and later the whole one
leaves the whole one. A crashed run costs at most the draft in flight.

### 1.3 Refitting the prior

The coefficient vector moves out of `draft_model.py` into a generated module,
`scoring/mock_prior.py`, which `draft_model` imports and binds to the name
`COLD_START_PRIOR`. Every existing reader -- `cold_start_fits()`, the length
assertion, the tests -- keeps referring to the same name and does not change.
`make fit-prior` regenerates that module, and its output is checked in.

Checked in rather than read from the corpus at runtime, for two reasons. The
fifteen numbers stay in git, so a prior that regressed is a reviewable diff
rather than an invisible change under a running app. And the existing
`assert len(COLD_START_PRIOR) == len(FEATURE_NAMES)` -- which exists precisely
to catch a feature added without extending the prior -- keeps working against
the generated file exactly as it works against the literal today.

`make fit-prior` reads the corpus, excludes our own slot and every autodrafted
pick, fits one pooled beta, and backtests it against the incumbent prior. It
writes the new file only if the new beta wins on `delta_top1`. A prior that
loses is reported and discarded, and there is no flag to force it: the way to
ship a losing prior is to change the decision rule in the open, where the
change is reviewable, not to override it at the command line.

### 1.4 New features

The model is a conditional logit: the softmax runs over the candidates in one
choice set, so any feature taking the same value for every candidate cancels
out entirely. "It is round 6" and "this team holds three running backs" are
facts about the pick, not about the player, and neither can enter alone. Both
enter as interactions with the candidate under consideration.

Four additions, each measured independently:

| feature | definition |
|---|---|
| stat profile | `usage`, `efficiency`, `played_share`, `peak_gap` -- already computed by `player_history.attributes_as_of`, already carried on every pool, currently read by nothing |
| `pos_count` | how many the picking team already holds at the candidate's position. Generalizes `need`, which is only a binary "below starter count" |
| `first_at_pos` | the candidate would be that team's first at his position |
| `pos_gap` | rounds since that team last took the candidate's position, plus `first_at_pos` interacted with round number |

The stat profile is a **re-measurement, not a new idea**.
`docs/superpowers/findings/2026-08-11-stat-profile-vs-position-dummies.md`
measured it and rejected it: 0.2198 top-1 against the incumbent 0.2356, at
n=696 picks. That finding says "do not re-litigate this without re-measuring",
and a mock corpus is exactly the re-measurement it names -- the loss was
concentrated in late rounds, where the position dummies carry kicker and
defense structure that `market_rank` cannot, and a larger corpus is where that
structure would become learnable from the profile instead.

Each feature ships only if it beats the incumbent set on `delta_top1` in the
leave-one-out backtest. `draft_model.py` states that rule and the finding above
explains why it is stated: so that a metric cannot be reached for only when it
flatters the conclusion. Some of these will lose. Losing is a result, and it
gets written up as one.

### 1.5 The pulse

Every row in the available list pulses red, with intensity scaled by
`1 - survive_pct`. Nothing to tune in the ramp itself: a player certain to
survive does not visibly move, a player certain to be gone pulses hard, and
everything between is a ramp. The one threshold is a floor, and it is a cost
control rather than a design choice -- see the constraint below.

`survive_pct` is already on `LiveCandidate` and already rendered. The change
is to bind it to a CSS custom property on the `<tr>` and animate from that.

Three constraints:

- **Animate `opacity` and `box-shadow` only, never `background-color` or
  anything that reflows.** Note that `box-shadow` is NOT compositor-friendly:
  it is a paint property, so unlike `opacity` and `transform` every frame of
  it re-rasterizes the row on the main thread. It is used here anyway because
  an inset shadow is the only way to tint a `<tr>` without touching
  `background-color`, but the cost is real -- which is why rows whose
  intensity is below a small floor (~0.02, i.e. already invisible) must get
  no animation at all rather than an invisible one. This is a two-hundred-row
  table with no virtualization, and most of it sits at that floor.
- **Honor `prefers-reduced-motion`.** The fallback is the static tint, which
  carries the same information without moving.
- **`survive_pct` of `null` gets no pulse at all.** Null means there is no
  roster to survive for, not zero chance of surviving. `riskTone` already
  refuses to color a dash for the same reason; the animation follows it.

## Phase 2: the path map

Once the prior is fitted on real mock behavior, the simulator can answer "what
are all the iterations my draft could take" far better than a bot could.

The reason the map comes from the simulator and not from farmed drafts is
arithmetic. A fourteen-round path from one seat has an astronomical branch
count. A few hundred farmed drafts put roughly one sample on each distinct
path, and an average over one sample is noise. `_run_draft`, by contrast, runs
thousands of rollouts per candidate opening pick in seconds, all from our seat,
and `roster_value` scores every ending.

So the split is: **the bot farms opponent behavior, which cannot be simulated;
the simulator maps paths, which cannot be farmed.**

Shape: for each of the top candidates available now, run rollouts pinned to
taking that player, and report the average final roster value along with the
most common positional sequence that follows. Endpoint and panel to be
specified once phase 1's backtest lands, because the numbers are not worth
rendering until the prior behind them has been measured.

## Testing

- **Corpus** -- schema migration against a corpus that already has rows;
  `record()` idempotency for a mock recorded partially and then completely;
  `autodrafted` and `my_slot` round-tripping.
- **Farm loop** -- the epsilon-greedy policy against a fake pool (fixed seed,
  both branches); the record built from a replayed frame log. The socket path
  itself is already covered by `tests/test_espn_live.py`, which replays 161
  frames captured from a real ESPN mock draft.
- **Model** -- each new feature's column construction against a hand-built
  choice set, including the cancellation property: a feature constant across a
  choice set must leave the likelihood unchanged.
- **Prior** -- `make fit-prior` refuses to write a prior that loses on
  `delta_top1`.
- **Pulse** -- null `survive_pct` renders no animation; reduced-motion renders
  the static tint.

## Risks

- **The lobby join may not exist as a replayable call.** Phase 0 gate. Kills
  the bot, not the design.
- **ESPN mock lobbies may be mostly autodrafted seats.** Also phase 0. This is
  the one that would make the corpus worthless rather than merely slow, which
  is why it is counted before anything is built.
- **Automating an ESPN account against their mock lobbies is against ESPN's
  terms of service.** Stated once, accepted by the owner, recorded here so it
  is not rediscovered later as a surprise.
- **Mock drafters may not draft like league managers.** A prior fitted on
  strangers is the right prior for a mock draft, which is the case it governs.
  It is not obviously the right prior for the first season of a real league.
  If the backtest shows them diverging, the prior becomes two priors selected
  by context rather than one.

## Decisions taken during design

- Data comes from mocks played by hand plus a bot farming them, not from
  scraping a third-party mock corpus and not from simulator output.
- The bot plays epsilon-greedy, not uniformly random and not pure policy.
- The mock corpus refits the cold-start prior only. It is not pooled into the
  fits that a real league's per-manager models shrink toward.
- The prior is a checked-in generated constant, not a runtime read.
- The pulse is intensity-scaled across every row, not a threshold on a subset.
