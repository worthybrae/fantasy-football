# Fantasy Football Draft Tool

A personal draft-night tool for an 8-team PPR league (roster: QB / 2 RB / 2 WR
/ TE / 2 FLEX (W-R-T) / K / DST, 5 bench). It pulls a few years of NFL stats
and market consensus rankings, scores every player on a handful of tunable
factors, and serves an interactive draft board you can run live during your
draft.

The pipeline stages are:

1. **refresh** — pull player stats, snap counts, depth charts, schedules, and
   market consensus rankings (Fantasy Football Calculator ADP, ESPN ADP,
   FantasyPros ECR) into a local DuckDB file.
2. **api** — compute PPR points, per-player scoring factors, value over
   replacement (VOR), tiers, and market consensus; serve it all over HTTP.
3. **web** — a React draft board: sortable table with market consensus and
   edge columns, live weight sliders, position filters, search, keyboard
   navigation, and drafted-player tracking that persists to disk.

## Player profiles

Clicking a row (or pressing `Enter` on the keyboard-selected row) opens that
player's profile drawer: header chips (rank, tier, VOR, composite, Mkt,
edge), the five scoring factors as bars, a weekly PPR-points chart across up
to three seasons with a per-season average line, full season-by-season stat
totals, an expandable game log, next-season outlook (depth slot, implied
points, strength of schedule, bye), a market-consensus breakdown by source,
and a list of similar players. For skill positions with stat history,
"similar players" are cross-year **stat twins** — other player-seasons
nearest by a weighted z-score distance over per-game production,
target/carry share, and efficiency — shown next to what that twin's *next*
season's PPG turned out to be, a quick gut check on what a comparable stat
line tends to become; rookies and K/DST (no stat history) instead get
similar-value neighbors from the board.

## Search and keyboard shortcuts

Type into the search box to narrow the board by player name or team — it
composes with the position tabs and "hide drafted" checkbox rather than
replacing them. The board also has a keyboard cursor, independent of the
mouse, that moves over whatever rows are currently visible (i.e. after
search/tab/hide-drafted filtering, in the current sort order):

| Key     | Action                                                   |
| ------- | --------------------------------------------------------- |
| `/`     | Focus the search box                                     |
| `↑` `↓` | Move the board cursor, scrolling it into view as needed  |
| `Enter` | Open the selected player's profile drawer                |
| `D`     | Toggle the selected player's drafted status               |
| `Esc`   | Close the profile drawer if it's open; else clear search |

The cursor/toggle/open shortcuts are inert while an input has focus (the
search box, a weight slider) or while the profile drawer is open — but not
after clicking a button (a position tab, the rail toggle, a row's
drafted-toggle ✓), which stays live for the very next keypress. Tabbing to
one of those buttons and pressing `Enter` activates the button itself
instead (drafts/undrafts the row, switches the tab) rather than also
opening the drawer or toggling drafted on whatever row the board cursor
happens to be on. `Esc` mostly ignores the input/drawer rule: with the
search box focused, it always just clears and blurs search, drawer or no;
with focus anywhere else, it closes the drawer if one's open, else clears
search if it has
text, else does nothing.

## Project structure

- `pipeline/` — data ingestion and refresh workflow (`sources.py`, `db.py`,
  `refresh.py`), plus ESPN draft history import (`espn_league.py`,
  `import_league.py`), manager fitting (`fit_managers.py`), and the
  simulator CLI (`run_sim.py`)
- `scoring/` — PPR calculations, per-factor scoring, composite/VOR/tiers,
  league config (`scoring/config.py`), ESPN-derived league structure
  (`league.py`), the per-manager pick model (`draft_model.py`), and the
  draft simulator (`draft_sim.py`)
- `api/` — FastAPI backend serving the draft board, drafted-player state,
  and draft simulation endpoints
- `web/` — Vite + React + TypeScript frontend
- `tests/` — pytest suite for the Python side

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium

cd web
npm install
```

The Playwright browser install is a one-time step. It's only needed for
`make espn-import` (below), which drives a real Chromium window to log in to
ESPN. Everything else in this repo works without it.

## Usage

Run these three, in order, each time you want a fresh draft board (steps 2
and 3 can be left running throughout your draft):

```bash
# 1. Pull the latest data into data/nfl.duckdb
.venv/bin/python -m pipeline.refresh

# 2. Start the API (from the repo root)
.venv/bin/uvicorn api.main:app --port 8000

# 3. Start the frontend (in another terminal)
cd web && npm run dev
```

Vite proxies `/api` requests to the backend, so just open the `npm run dev`
URL and go. Re-run `pipeline.refresh` any time you want newer stats/ADP —
the API reads straight from the DuckDB file, so a restart isn't required for
the underlying data, only for picking up schema changes.

## How scoring works

Every player gets five raw factors, each normalized to a 0–100 percentile
within their position (missing data defaults to 50, i.e. neutral):

- **production** — recency-weighted PPG over the last three seasons
  (`scoring/config.py: RECENCY_WEIGHTS`, currently 50% most recent year, 30%
  the year before, 20% the year before that)
- **durability** — games played vs. games possible since the player entered
  the league
- **role** — blend of depth-chart position and share of team
  targets+carries
- **environment** — team implied point total for the season (from
  Vegas total/spread lines)
- **schedule** — strength of schedule, i.e. average fantasy points allowed
  by upcoming opponents at the player's position

Those five factors combine into a **composite** score using a weighted
average. Default weights live in `scoring/config.py: DEFAULT_WEIGHTS`
(production 0.35, role 0.25, environment 0.20, schedule 0.10, durability
0.10) but every request to `/api/players` can override them — the web UI's
weight sliders do exactly this, live, no restart needed. Editing
`DEFAULT_WEIGHTS` in `scoring/config.py` just changes what the sliders start
at.

The composite score converts to **VOR** (value over replacement) by
subtracting, per position, the composite score of the last starter-caliber
player at that position (`scoring/config.py: REPLACEMENT_RANK`, calibrated
to this league's 8-team starting lineup). Players are then bucketed into
**tiers** per position, breaking wherever the VOR gap to the next player is
unusually large (mean + one standard deviation of that position's gaps).
The board is ranked by VOR overall; when sorted by rank on a single
position, the table also bands rows by tier (a faint alternating
background) so a tier boundary is visible at a glance.

League size and roster shape (`LEAGUE_TEAMS`, `REPLACEMENT_RANK`) also live
in `scoring/config.py` — update them there if the league format changes.

## Draft simulation

On top of the season-long scouting board, the tool can import your own
league's ESPN draft history, learn how each manager in your league actually
drafts, and simulate the upcoming draft from any slot. Think of it as a
pre-draft study you run in the days before your draft, not a live
draft-room client — it doesn't poll an active ESPN draft room, and it
doesn't cover auctions, keeper leagues, or in-draft trades.

### Importing draft history

```bash
make espn-import LEAGUE=<your-league-url-or-id>
```

This pulls your league's draft picks, team rosters, league settings, and
that season's market data for every season it can find, walking seasons
back from the current one and stopping after two consecutive misses (so one
gap year in ESPN's history won't cut the walk short). It writes five tables:
`draft_picks`, `draft_teams`, `league`, `historic_adp` (Fantasy Football
Calculator's ADP), and `historic_espn_cs` (ESPN's preseason cheat sheets,
the pick model's actual fitting reference; more on that below). It also
pulls a cheat sheet for the current season, even
though there's no draft to import for it yet, because the simulator needs
this year's board ranked the same way the model was fitted. The ADP table
and the cheat sheets are what make "reach" measurable at all: a pick only
means something relative to where the market had that player *that* year.

The first run opens a real Chromium window at the ESPN login page and waits
for you to sign in. Disney SSO's 2FA and bot checks need a human, and a
visible browser is the only place to answer them. Once you're signed in, it
saves the session to `data/espn_state.json` — gitignored, since it's
effectively a login token — and every run after that is headless.

The import prints a validation summary per season: pick count against
`teams x rounds`, manager count, and ADP match rate. A season under 80% ADP
match gets flagged in the output. Picks that don't match an ADP row stay in
`draft_picks` but are excluded from model fitting, and a bad match rate
usually means a name-matching problem worth checking before trusting
anything built on top of it.

### League structure becomes derived

Once a league is imported, team count, starting lineup, FLEX slots, bench
size, round count, and scoring rules all come from ESPN's own settings
(`scoring/league.py`) instead of the hardcoded constants in
`scoring/config.py`. ESPN scoring items that don't map to an nflverse stat
column are reported at import time by name, not silently dropped, so a
league running TE premium or a stat this tool doesn't recognize yet won't
produce a quietly wrong board. Replacement ranks (used for VOR) are
recomputed from the derived roster shape the same way. With no league
imported, everything falls back to `scoring/config.py`'s constants and
reproduces today's 8-team board exactly.

### Fitting manager models

```bash
make fit-managers
```

This fits a conditional logit to each manager's picks: given who was
available at that moment, which player did they take, and does that follow
a consistent pattern of reaching for need, chasing players who fell,
favoring a position early, or following a run. Each manager's model is
ridge-shrunk toward a fit pooled across the whole league, so a manager with
few or noisy picks lands close to league average instead of overfitting to
a handful of decisions.

**The market reference is ESPN's own preseason cheat sheets.** This league
drafts on ESPN, off ESPN's board, so "reach" and "fall" are measured
against how far a pick landed from where ESPN's own printable PPR top-300
ranked that player before the season, not against some other site's
consensus. The importer downloads that PDF for every season it can
(`NFLDK{YYYY}_CS_PPR300.pdf` through 2022, `NFL{YY}_CS_PPR300.pdf` from
2023, both under `espncdn.com/s/ffldraftkit`) and parses it with `pypdf`
into `historic_espn_cs`, 300 players a season. Fantasy Football Calculator's
ADP fills in anyone the sheet doesn't rank; that's the only fallback.

ESPN's own live rankings API was tried first and rejected as a historical
reference, which is worth knowing before you go looking for a shortcut
there. It serves no preseason rank at all for 2020-2022, and the 2023 rank
it does serve isn't preseason either: it correlates 0.905 with the
following year's board (2024) against just 0.678 with 2023's own, and has a
running back who was a late-round flier that August sitting inside the top
ten, a rank he only earned by breaking out during the season. The cheat
sheets don't have that problem. Every season checked opens with the right
players for that year. Those API ranks used to be imported per season into
a `historic_espn` table; that table is gone, because once the cheat sheets
became the reference nothing read it. The board's Mkt column never came
from it and doesn't now — that's `espn_adp`, which `make refresh` pulls for
the current season only, where "where does ESPN have this player right now"
is a fair question to ask.

**`reach` and `fall` are on a log scale**, not linear. Linear rank put a
bigger gap between rank 100 and rank 140 than between rank 1 and rank 5, so
the model couldn't tell elite players apart: pick 1 of a real draft came
back close to a coin flip. On a log scale the gap from 1 to 5 is bigger than
the gap from 100 to 140, which is closer to how a draft actually goes.

**Four features beyond reach/fall/position/need survived a held-out test**:
`age` (centered within position, so the coefficient reads as "younger than
typical for a QB" rather than picking up that QBs simply last longer than
RBs), `no_track_record` (no prior seasons at all, kept separate from a
fabricated zero), `hype` (how far the market's rank sits ahead of a rank
derived from the player's own prior production, i.e. whether the pick is a
leap of faith), and `trend` (the slope of his points per game across prior
seasons). A fifth, `volatility` (points-per-game variance plus a
games-missed rate), was measured and cut: it made the backtest slightly
worse, not better. All five were decided the same way, by whether they
improved held-out accuracy, not by whether they sounded plausible.

**Read the printed backtest line before trusting anything downstream.** It's
leave-one-season-out over all six imported seasons: each season takes a
turn as the held-out test while the other five train the model, and results
accumulate across all six rotations, 696 evaluations rather than the ~112
you'd get from holding out only the newest season. It prints top-1 and
top-5 accuracy and log-loss overall, broken down by round (early/mid/late),
against a pure-market baseline, and a feature-ablation table showing what
each of the five candidate features above did to held-out top-1. On this
league's real history the fitted model scores top-1 24.4%, top-5 59.6%,
log-loss 2.871, against a gate of 16% / 53% and a pre-recalibration
baseline of 19.4% / 51.1% / 3.163 measured the same way. If a future refit
doesn't clear that baseline out of sample, the command prints a warning,
and it means what it says: treat the simulator's output as indicative only,
not as a real prediction, until the fit improves. The result is also saved
and served from `/api/model`, so the same warning appears in the draft rail
rather than only in the terminal you happened to run `make fit-managers` in.

The same fit also measures one league-wide **positional bias**: how far
ahead of or behind the market this league takes each position, by round.
`make fit-managers` prints that table too, and it's worth reading on its
own regardless of the pick model. Right now, for instance, this league
takes kickers roughly 58 ranks ahead of the market in the late rounds and
tight ends roughly 9 ahead in the middle rounds. That's a fact about the
league, not a guess the model is making.

**No manager currently earns a personal model.** A manager needs enough
picks, and needs their own fit to actually beat the pooled fit on held-out
seasons; if it doesn't, the simulator falls back to the pooled model for
that manager, and their card says "league average, not enough signal"
rather than pretending otherwise. Right now that's every manager: at
roughly 87 fitted picks each (six seasons, one league), none of the eight
beats the pooled fit on held-out data. That's expected, not a bug: 87 picks
isn't enough to fit 15 coefficients on its own, and even with shrinkage
pulling a thin fit toward the pooled one, none of the eight personal fits
comes out ahead. It's why the manager cards on `/draft-board`'s "By
manager" tab show the league-average read next to each manager's actual
draft history rather than a personalized forecast: right now the real
history is a better guide to what a specific manager will do than the
model's coefficients are.

### Running a simulation

```bash
make sim SLOT=4 ROLLOUTS=300
```

or click "Run simulation" in the draft rail on the board itself, after
setting your slot and (optionally) editing the draft order — it's seeded
from ESPN's published order but you can override any slot, so "what if I'm
picking third instead" is answerable before the real order is out.

A real run at the default 300 rollouts takes roughly a minute: about 10
seconds fitting manager models and about 50 seconds searching. At each of
your own picks, the simulator forces roughly a dozen plausible candidates
in turn, rolls the rest of the draft forward against the fitted opponent
models N times per candidate, and scores each one by the projected points
of your best legal starting lineup at the end (with an insurance term so
picks past the first several rounds still matter instead of scoring as
noise). `ROLLOUTS` trades runtime for resolution — more rollouts, tighter
standard error on each candidate's score.

Every pick inside a rollout that isn't one of those forced candidates
(every opponent's pick, and your own picks past the one being evaluated) is
chosen by a one-ply greedy policy: take whichever available player raises
roster value the most, where a candidate's points are valued above his
position's replacement level rather than counted raw. Raw points used to
be the rule, and a raw-points greedy takes the best quarterback in the
first round every time, because a quarterback outscores every running back
and receiver on the board in raw points, even though a replacement-level
quarterback is easy to find and a replacement-level running back isn't.
Value over replacement is what the board's own VOR ranking already uses;
the simulator's in-rollout policy just wasn't using it before.
`roster_value` itself, the number a finished rollout reports, is
unchanged; only the policy that picks candidates during a rollout changed.

`make fit-managers` has to have been run first. Without fitted opponent
models every opponent would pick uniformly at random over the whole pool,
which makes the consensus number one look about 100% likely to still be
available at any slot — a confidently wrong answer rather than a rough one —
so the run fails with that message instead, and the rail says so before you
click.

If you've marked players drafted, the simulator reads `drafted.pick_no` to
work out which team took each of them and resumes with real rosters. Rows
marked before that column existed have no pick number and can't be
attributed to anyone, so a run refuses rather than guess; un-mark and re-mark
them in draft order to fix it.

The board picks up two new columns once a sim has run, and the header strip
above it shows your slot, your next pick, and how old the last run is (on
page load too, since those columns come from whatever run last finished):

- **Avail%** — the probability the player is still on the board at your
  next pick.
- **ΔEV** — expected end-of-draft starting-lineup points relative to the
  best available candidate at your current pick. Populated only for the
  players the simulator actually evaluated; blank means that player wasn't
  in contention, and both columns blank together just means no sim has run
  yet. A player who almost certainly won't last until your pick isn't
  evaluated: forcing him mostly wouldn't happen, so his "expected value"
  would really be the value of whatever you'd have taken instead.

### The predicted draft board

A simulation run also fills in a full board, not just your own picks: open
**Grid** in the header, or go to `/draft-board`, for a grid with rounds down
the side and managers across the top, one predicted player in every cell for
the whole draft. It's built from the same rollouts as the `Avail%` and `ΔEV`
columns, `predict_board` in `scoring/draft_sim.py`, so it doesn't form a
second opinion; it adds roughly 4 seconds to a run that already takes about
50, and the results are written to a new `sim_board` table and served at
`GET /api/sim/board`.

Each cell shows the model's single most likely player at that pick. Hover a
cell to see the second and third most likely players and their
probabilities.

**The name in the cell is a consistency choice, not just "the highest raw
number."** If four adjacent picks each have the same player as their
top individual choice, showing him in all four cells would just look broken.
So the raw per-pick frequencies get resolved into one board by walking the
picks in draft order and giving each pick its most likely player, as long
as an earlier pick hasn't already claimed him. Earliest pick wins any
contest over a shared name: a board is read top to bottom, and a wrong name
at pick 1 discredits the whole grid in a way a wrong name at pick 90
doesn't, so the early picks are the ones that have to be right. That's a
deliberate trade, not the assignment that maximizes the whole board's joint
likelihood, and it has a visible cost: on a recent 120-cell run, 79 cells
showed a primary that the same cell's own hover ranks *below* one of its
alternates, because an earlier pick had already claimed the alternate. The
hover is titled "raw odds for this pick" for exactly this reason. It's the
undeduped picture, and it's worth checking on any cell that looks
surprising, since nothing about the real distribution is hidden, only
resolved into one story.

**Your own column is a plan, not a prediction.** The rollouts fill your picks
with the same greedy, best-marginal-value-right-now policy the simulator
uses to search candidates, not with a fitted model of your own behavior.
That policy is close to deterministic given the board state, so its
cell frequencies would sit near 100% almost everywhere, which would read as
confidence about the future that the model doesn't have. Those cells get
their own background shade and show no probability at all — that missing
percentage is the tell, since every predicted cell has one. Read that column
as "what the current plan does at each of your picks," not as a forecast of
what will happen.

A player who was already marked drafted when the run happened gets a third
background shade, again with no probability. It isn't a prediction anymore.
All three states keep the same position colour on the left edge, so the
shade and the presence or absence of a percentage are what tell them apart.

The grid shows the last simulation run, including the drafted state as of
that run — a player you mark drafted afterwards still has predictions sitting
in his cell, because `certain` was computed inside `predict_board` from the
picks that existed when it ran and written into `sim_board` then. The grid
does not refresh itself as picks come in; re-running is a deliberate step,
same as for the board's `Avail%`/`ΔEV` columns. A run made
before this feature existed (or one that otherwise wrote no per-pick rows)
produces an empty grid, and the page says so and tells you to re-run rather
than showing a silently blank board.

Clicking a cell opens the **player card**: a condensed, draft-night version
of the full profile, not the profile itself. It leads with projected
points per game next to the player's recent actual PPG (a wide gap between
the two is flagged, since a projection well ahead of recent real production
is a bet, not a fact), a row of Rank/Tier/Mkt/Edge plus that player's
`Avail%` and `ΔEV` from the same run (`ΔEV` only for the handful of players
scored at your next pick), a row of critical numbers specific to his
position (finish, role, snap share, games played, strength of schedule,
whichever apply), and an 18-cell strength-of-schedule strip, one cell per
week, coloured soft to tough with the bye week marked. A link to the full
profile page covers anything the card leaves out, and a mark-drafted button
means a pick doesn't send you back to the board page.

Because the grid is built on the same fitted manager models as the rest of
the simulator, the backtest line `make fit-managers` prints matters even
more here than for `Avail%`/`ΔEV`: a full, confident-looking board is easy to
over-trust, and it's only as good as that line says the model is.

### Viewing the board by manager

`/draft-board` has two tabs. **Grid** is the round-by-manager table above.
**By manager** shows the same run as one card per team instead of one row
of cells per round. Each card pairs two different kinds of information and
keeps them visually apart on purpose: a quiet, monochrome panel of that
manager's real draft history (their actual first-round pick every season
it's imported, and their overall positional shape by round bucket: rounds
1-3, 4-8, 9+), and below it, in colour with probabilities attached, their
tendency in one line and their next few predicted picks from this run. Fact
and forecast never share a color, so a real pick can't be mistaken for one
more guess. The history comes from `GET /api/managers/history`, a plain
read of `draft_picks` and `draft_teams`, no model involved.

Since no manager currently earns a personal model (above), that real
history is, for now, more informative about a specific manager than the
forecast next to it is, which is exactly why the card leads with it rather
than hiding it behind the coefficients.

## Data sources and quirks

- **Weekly stats / snap counts / depth charts / schedules** come from
  `nfl_data_py` / nflverse. History covers the three prior seasons
  (`scoring/config.py: HISTORY_SEASONS`), used for production and durability;
  the current season's schedule feeds environment, schedule strength, and bye
  weeks.
- **Market consensus** blends up to three independent sources — Fantasy
  Football Calculator (FFC) ADP, ESPN ADP, and FantasyPros' expert consensus
  rankings (ECR) — averaged per player into the board's **Mkt** rank (the
  drawer's Market section and the board's Mkt-column tooltip both show each
  source's raw rank so you can see where they agree or don't, and how many
  actually had the player — a Mkt rank can come from just one source with no
  warning on the board itself beyond that tooltip). FFC is explicitly a
  **12-team** consensus feed, not 8-team (`pipeline/sources.py:
  fetch_adp`); ESPN and FantasyPros don't publish a team-count parameter to
  check against, but treat all of Mkt as a rough market-consensus signal
  rather than a literal pick-order prediction for this league either way.
  When at least two sources have a player, the board also tracks their
  **spread** (max rank − min rank); a muted `±N` (half the spread) appears
  next to Mkt whenever it's 12 or more, flagging players the market itself
  doesn't agree on. Separately, the **edge** column (Mkt rank minus VOR
  rank) compares the market to this board specifically: positive means the
  market is undervaluing the player relative to this board's VOR ranking (a
  potential value), negative means the opposite; `|edge| < 3` isn't a
  meaningful signal either way and renders neutral in the table.
- **K/DST** aren't scored on production, durability, role, or schedule —
  the PPR formula doesn't score kicking or defensive stats, so those factors
  would just be noise. Only **environment** (team implied points) drives
  their ranking; the other four factors are forced to neutral (50).
- **Rookies and K/DST** enter the board from the ADP feed even though they
  have no weekly stat history (rookies) or no player-level stats at all
  (K/DST, which is scored at the team level via `team`, not name matching).
- The `/api/meta` endpoint (surfaced as dots in the UI header) reports
  whether each source's last refresh succeeded and how many rows it pulled —
  useful for confirming a refresh actually got fresh ADP before your draft
  starts. A failed **adp** or **schedules** refresh triggers a visible
  warning banner, since those two feed the board most directly.

## Tests

```bash
.venv/bin/pytest
```
