# Scale-out: many live drafts, bounded memory, Supabase, Cloudflare

Date: 2026-08-28. Target: 50-200 concurrent live drafts on one Railway
service by end of day, with the storage shape that lets a second replica
follow later.

## What is wrong today

- `api/live.py` keeps ONE `state` dict and one lock. `POST /api/live/connect-token`
  stops whatever listener is running, so the second person to connect ends
  the first person's draft. Every `/api/live/*` route serves that one session
  to any caller.
- A connect costs 8-35 s of CPU in the request thread: `build_board` (1.6-1.9 s),
  `build_pool` (2.2-3.6 s), `fit_all` (up to 30 s for a league with history).
  Under the GIL that stalls every other request.
- `build_board` reads the full `weekly` table (174k rows x 150 columns, 244 MB in
  pandas) three times per build, and `build_profile` reads `weekly` and
  `snap_counts` whole on top of it. One cold build peaks at 1.67 GB. Concurrent
  builds on different cache keys are unbounded; eight of them killed the
  container on 2026-08-25.
- The board cache key includes the drafted set, so every pick rebuilds the whole
  board to recompute one boolean column.
- Mutable per-user state (custody, billing, the live session record) lives in
  DuckDB files and a JSON file on the volume. Single-writer locks pin the
  service to one replica.
- No gzip, no cache headers, one 440 KB JS chunk, no route splitting.

## Design

### 1. Multi-session live drafts (`api/live.py`)

- `LiveSession`: one object per draft holding what the `state` dict holds today,
  plus its own `threading.Lock`, its two threads (socket pump, recompute
  worker), its `league_conn`, its `ConnectProgress`, and `last_activity`.
- Registry: `dict[str, LiveSession]` guarded by a registry lock. Keyed by a
  session id (`sid`).
- `sid` is an httpOnly, `SameSite=Lax`, 12-hour cookie named `espn_live`,
  minted by the connect endpoints (`connect-token`, `connect`) when the request
  has none. The custody cookie is untouched and still names the account.
- Every `/api/live/*` route resolves its session from the cookie. No cookie or
  no session: `state` answers `active: false`, `connect-progress` answers
  `idle`, `board` answers 404, `select`/`autodraft`/`stop` answer 409.
- A second connect on the same `sid` replaces only that session, with today's
  stop/supersede semantics.
- `app.state.live_settings` becomes `live_settings(request)`; the board and
  profile endpoints in `api/main.py` pass their request through.
- SSE `/api/live/events` polls its own session's revision at 0.25 s.
- Starlette's threadpool is raised to 200 tokens at startup.
- Per-league DuckDB connections set `threads=2` and `memory_limit='256MB'`.
- Reaper thread: a session with no socket activity for 3 hours is stopped and
  dropped, its record deleted.
- Session record: table `live_session` in Postgres (see 3), one row per `sid`,
  the draft token encrypted with the current custody key. Startup restores every
  row younger than 12 hours, at most three at a time.

### 2. Connect and build cost

- `provision_league` + `build_session` run in a `concurrent.futures.ProcessPoolExecutor`
  (spawn context, 3 workers). The worker opens the league file, builds, closes,
  and returns the pickled `DraftSession`; the parent opens `league_conn` only
  after the future resolves. Progress is reported per stage from the parent
  (submitted, building, done) rather than per manager.
- `scoring/board.py`: read `weekly` once per build with a column projection
  (`pipeline/db.read_table(conn, name, columns=...)`), pass the frame to
  `consistency` and `expected_change`.
- `scoring/board_cache.py`: drop `_drafted_key`; apply `isin(drafted)` to the
  cached frame on the way out. A `threading.Semaphore(2)` bounds cold builds.
  `api/live.py` uses `cached_build_board`.
- Warm `cached_build_board`, `cached_profile_frames`, `cached_game_points` on
  startup and at the end of `pipeline.refresh.main()`.
- Recompute stays in-process. A global `threading.Semaphore(os.cpu_count())`
  bounds concurrent recomputes; `SURVIVAL_ROLLOUTS` drops from 400 to 150 while
  more than 50 sessions are active.

### 3. Supabase Postgres for mutable state

- Env `SUPABASE_DB_URL` (session pooler, `sslmode=require`). Absent: every
  store keeps its DuckDB backend, so local and tests change nothing.
- `pipeline/credentials.py`: `CredentialStore(path=..., dsn=...)`. A `_Backend`
  seam with `duckdb` and `postgres` implementations. `_compact`, `_flush`,
  `_lock_down` are DuckDB-only. Same two tables, same ids, same crypto.
- `api/custody.py`: catch a backend-neutral `cred.StoreError` instead of
  `duckdb.Error`.
- `api/billing.py`: `_db()` returns a thin adapter; same three tables in
  Postgres; `_first_time` uses `INSERT ... ON CONFLICT DO NOTHING RETURNING`.
- `live_session(sid PK, account_id, league_id, team_id, season, swid, token_blob,
  my_slot, saved_at)`.
- A 30-second in-process cache in front of `CredentialStore.resolve` keyed by
  the cookie hash, so polls do not pay a cross-region round trip.
- psycopg 3 with a pool of 2-8 connections. Timestamps stored as `timestamptz`.

### 4. Edge and frontend

- `GZipMiddleware` in `api/main.py`.
- `api/static.py`: `/assets` gets `public, max-age=31536000, immutable`; the SPA
  index gets `no-cache`.
- Public GETs (`/api/landing/*`, `/api/lobby`, `/api/market/*`, `/api/mocks/*`,
  `/api/demo/*` except events, SEO pages) get `Cache-Control: public, s-maxage=N`
  with N from 15 to 60 so Cloudflare serves them from the edge. SSE responses
  get `no-store`.
- `web/src/App.tsx`: `React.lazy` per route with a `Suspense` fallback;
  `vite.config.ts` gets a vendor `manualChunks` split.
- `main.tsx`: Inter latin subset only; `index.html` preloads the two fonts.
- `DraftRoom.tsx`: `Promise.all` for state and board. `/api/mocks/drafts`
  inlines the first board.
- Cloudflare: Free plan, proxied DNS, cache rule respecting origin headers.
  Runbook in `docs/superpowers/reference/cloudflare-runbook.md`. The
  nameserver change at Namecheap is the owner's step.

### 5. Verification

- The existing suite stays green. Fixtures in `tests/test_live_api.py` gain a
  `sid` cookie; `tests/test_credentials.py`, `tests/test_custody_api.py` and
  `tests/test_billing.py` keep their DuckDB backend untouched.
- New `tests/test_live_sessions.py`: two connects on different cookies run
  side by side; a connect on the same cookie supersedes; no cookie answers
  inactive; the reaper drops an idle session.
- New `scripts/load_live.py`: N simulated sessions replaying
  `data/draft_room_trace.jsonl` against a local server; reports p95 of
  `/api/live/state` and peak RSS. Acceptance: 100 sessions, p95 under 300 ms,
  RSS under 6 GB.
- Deploy is the owner's `railway up`; nothing here deploys.

## Out of scope today

- Splitting listeners into a worker tier and running more than one web
  replica. The `live_session` table and per-request session routing are the
  prerequisites; the split is the next step.
- Moving per-league DuckDB files or the stats tables anywhere.
- AWS.
