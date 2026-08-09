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
that season's ADP for every season it can find, walking seasons back from
the current one and stopping after two consecutive misses (so one gap year
in ESPN's history won't cut the walk short). It writes four tables:
`draft_picks`, `draft_teams`, `league`, and `historic_adp`. Historic ADP is
what makes "reach" measurable at all — a pick only means something relative
to where the market had that player *that* year.

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

**Read the printed backtest line before trusting anything downstream.** It
holds out the most recent season and reports top-1 and top-5 pick accuracy
and log-loss against a pure-ADP baseline. If the fitted model doesn't beat
that baseline out of sample, the command prints a warning, and it means
what it says: treat the simulator's output as indicative only, not as a
real prediction, until more seasons are imported. The result is also saved
and served from `/api/model`, so the same warning appears in the draft rail
rather than only in the terminal you happened to run `make fit-managers` in.

Not every manager gets a personal model. A manager needs enough picks, and
needs their own fit to actually beat the pooled fit on held-out seasons —
if it doesn't, the simulator falls back to the pooled model for that
manager, and their card in the rail says "league average, not enough
signal" rather than pretending otherwise. That's expected, not a bug: with
roughly 105 picks per manager (7 seasons of 15 rounds), there's enough
signal for around a dozen coefficients with shrinkage, and no more. It's
nowhere near enough to learn player-level preferences, like a manager's
favorite NFL team, which is why the model doesn't attempt that.

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

### A known open question

The simulator dense-ranks the board's "market-known" players 1..k to match
the scale the pick model was fitted on (`historic_adp`'s per-season dense
rank over that season's ADP pool). Whether k, for your league, actually
lines up with the population size that scale assumes hasn't been confirmed
against a real import yet — only against test fixtures. If reach/fall
behavior looks off once you run this against your own league's real
history, that's the first thing to check.

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
