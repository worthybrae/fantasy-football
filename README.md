# Fantasy Football Draft Tool

A personal draft-night tool for an 8-team PPR league (roster: QB / 2 RB / 2 WR
/ TE / 2 FLEX (W-R-T) / K / DST, 5 bench). It pulls a few years of NFL stats
and ADP data, scores every player on a handful of tunable factors, and serves
an interactive draft board you can run live during your draft.

The pipeline stages are:

1. **refresh** — pull player stats, snap counts, depth charts, schedules, and
   ADP into a local DuckDB file.
2. **api** — compute PPR points, per-player scoring factors, value over
   replacement (VOR), and tiers; serve it all over HTTP.
3. **web** — a React draft board: sortable table, live weight sliders,
   position filters, and drafted-player tracking that persists to disk.

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
The board is ranked by VOR overall, and the table shows a horizontal rule
between tiers when sorted by rank.

League size and roster shape (`LEAGUE_TEAMS`, `REPLACEMENT_RANK`) also live
in `scoring/config.py` — update them there if the league format changes.

## Data sources and quirks

- **Weekly stats / snap counts / depth charts / schedules** come from
  `nfl_data_py` / nflverse. History covers the three prior seasons
  (`scoring/config.py: HISTORY_SEASONS`), used for production and durability;
  the current season's schedule feeds environment, schedule strength, and bye
  weeks.
- **ADP** comes from Fantasy Football Calculator's public API and is a
  **12-team** consensus, not 8-team — treat it as a rough market-consensus
  signal (the `edge` column, ADP-rank minus VOR-rank — positive means the
  market is undervaluing the player relative to this board) rather than a
  literal pick-order prediction for this league.
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
