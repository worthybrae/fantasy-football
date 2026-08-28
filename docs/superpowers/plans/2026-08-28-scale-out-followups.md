# Scale-out follow-ups (carried out of the 2026-08-28 reviews)

These are the deferred / parked findings the final whole-branch review should triage. Each line names the file and the change.

## Owner requests (must land)
- "Steady" → "Reliable" wherever the label is shown to a user (web/src/components/PlayerProfile.tsx SteadyPanel heading and any column header / tooltip / App.css comment that spells the label; keep identifiers such as `steadyCols`, `.steady-meter` unchanged).

## Test-suite hygiene
- api/main.py loads `.env` at import even under pytest collection (`PYTEST_CURRENT_TEST` is unset at import time), so a Stripe key in `.env` turns billing on for tests collected in the same process. Guard on `"pytest" in sys.modules` as well.
- tests/test_profile.py:782 and :739 are vacuous on their fixture (the real-DB identity check is out of band); add a note or a fixture with an extra column that the projection drops.
- tests/test_jobs.py::test_a_failed_refresh_still_releases_the_farm runs a real `refresh.main()` (~35 s); inject `run=`.

## Storage (Tasks 1-4)
- api/billing.py `_seed_mocks_from_corpus` re-inserts ~854 rows on every store rebuild.
- Postgres `CREATE TABLE IF NOT EXISTS` race across workers on a cold boot (one 503 then self-heals) — comment or retry.
- `live_session` JSONB keeps swid/league/team in plaintext; only the token is encrypted — follow-up to encrypt the whole document.
- `live_session.saved_at` column written, never read; malformed rows never age out.
- Three near-identical `_run` helpers across the stores.

## Build memory (Task 5)
- scoring/profile.py:1111 stale comment (per-request → per-build).
- README env table lacks WARM_ON_BOOT (and SEO_WARM / DEMO_WARM).
- warm() warms the stored-league entry only; the parent's board cache never sees worker-built boards.

## Live sessions (Tasks 6-7)
- `_evict_seat` is not atomic across 2+ rooms on one seat.
- A failed cookieless connect leaves one inactive registry entry (reaper collects after `idle`).
- `on_event("startup"/"shutdown")` is deprecated in the pinned FastAPI; move to `lifespan`.
- A sequential reaper pass can exceed its interval.
- Residual after a build-timeout 503: a retry can hand the league file to a new worker while the abandoned worker still holds it; tracking the in-flight worker via `add_done_callback` would close it.
- `league_conn` leaks if `_launch_listener` raises after `_provision_and_build`.
- Stale comments in api/live_build.py:50-55 and the `build_in_worker` docstring (only true when `cancel()` succeeds) — may already be fixed in the perf round.

## Edge / frontend (Tasks 9-10)
- The landing cookie test cannot catch a valid-session divergence (needs a live session stood up).
- `/api` exact path (no trailing slash) falls outside `API_PREFIX`; a 500 from ServerErrorMiddleware is never stamped.

## Load tooling (Task 8)
- scripts/load_server.py: refuse to start when `RAILWAY_ENVIRONMENT` is set; make `--db` required.
- scripts/load_live.py: make `--pid` required for a PASS verdict (or print FAIL "unsampled").
- Docstring callout: load rooms' league files land in the real data/leagues directory (exact-id cleanup).
- Empty-trace busy loop in run_fake_socket_listener.

## Docs (Task 11)
- README deploy section: SUPABASE_DB_URL, LIVE_BUILD_WORKERS, WARM_ON_BOOT, LIVE_DEFAULT_ROOM, LIVE_FAKE_SOCKET (never in production), the `espn_live` cookie, the `live_session` table, "one replica still required until the listener tier is split".
- Dockerfile ENV lines for LIVE_BUILD_WORKERS (already) — confirm.

## Task 7 round-4 re-review minors
- api/live.py `close_later`: `try/finally` so the closer's claim token is released even if `close()` raises; log the failure.
- api/live_build.py: delete the unread `CONN_THREADS`/`CONN_MEMORY_LIMIT` aliases and their comment.
- scoring/board_cache.py `cached_build_pool` key: add `_weights_key` and `_sim_key` (hardening; not reachable today).
- scoring/board_cache.py module docstring: describe the identity key (meta fingerprint, path only when meta is empty).

## Controller-added performance levers (from the 100-room run)
- api/live.py `live_state`: cache the per-poll derivations (`_drafted_state` replay, roster, VOR fallback ranking) per room keyed on (`generation`, `picks_seen`, `as_of_pick`) so 80 polls/s do not each replay `drafted`.
- web/src/pages/DraftRoom.tsx: `FALLBACK_POLL_MS` 5000 → 15000 when the SSE stream is attached (SSE already wakes the poll on every state change).

## Recorded after the final wave (follow-ups, not blocking)
- Sid-keyed side effects across `registry.replace()`: claims (`claim_path`/`release_path`) and the restore record are keyed on `s.sid`; a connect landing mid-retirement can (i) orphan the old room on the stop-timeout path, (ii) un-claim the new room's file / delete its record on the success path. Key them on a per-object token; re-register the orphan on the False path.
- `LiveSession.touch()` writes `last_activity` without the lock; `run_once` overreports `gone` when a drop is skipped.
- The capacity 503 drops the `Set-Cookie`, so each retry at capacity mints another empty room until the reaper.
- Build-timeout retry residual (`add_done_callback` on the abandoned future).
- Snapshot copy at `import api.main` under pytest; billing `_seed_mocks_from_corpus` under the lock on first call; token restore across a custody key rotation.
- Next performance levers (100 rooms: state p95 1.27 s): rankings into the worker pool; cache `live_state`'s per-poll derivations on `picks_seen`.
