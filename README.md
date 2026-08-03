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
  `refresh.py`)
- `scoring/` — PPR calculations, per-factor scoring, composite/VOR/tiers, and
  league config (`scoring/config.py`)
- `api/` — FastAPI backend serving the draft board and drafted-player state
- `web/` — Vite + React + TypeScript frontend
- `tests/` — pytest suite for the Python side

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd web
npm install
```

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
- The `/api/meta` endpoint (surfaced as chips in the UI header) reports
  whether each source's last refresh succeeded and how many rows it pulled —
  useful for confirming a refresh actually got fresh ADP before your draft
  starts. A failed **adp** or **schedules** refresh triggers a visible
  warning banner, since those two feed the board most directly.

## Tests

```bash
.venv/bin/pytest
```
