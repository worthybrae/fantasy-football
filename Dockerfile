# The deployed image: one process serving the API and the built frontend.
#
# TWO STAGES BECAUSE THIS IS TWO LANGUAGES. Node builds `web/dist` and is
# then thrown away -- the runtime has no use for npm, and carrying a node
# toolchain into the final image would roughly double it for nothing. Railway
# builds a Dockerfile natively; its Nixpacks autodetection picks ONE language
# per service and would silently ship a backend with no frontend in it.
#
# WHAT IS NOT IN HERE: the databases. `data/` is gitignored and runs to 80 MB
# of DuckDB, so it lives on a mounted volume instead (see railway.toml and
# the README's deploy section). An image that baked them would have to be
# rebuilt to refresh a stat line.

# ---- the frontend -----------------------------------------------------------
FROM node:20-slim AS web

WORKDIR /build
# Manifests first, so a change to application code does not re-resolve every
# dependency: this layer is cached until the lockfile itself moves.
COPY web/package.json web/package-lock.json* ./
RUN npm ci

COPY web/ ./
# `npm run build` is `tsc -b && vite build`, so a type error fails the image
# rather than shipping. That is deliberate: the frontend is not separately
# gated anywhere else in this deployment.
RUN npm run build

# ---- the runtime ------------------------------------------------------------
FROM python:3.10-slim

# Kept in step with the venv this project is developed against (python3.10,
# see requirements.txt for the pins that matter).
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Requirements before source, same caching reason as npm above. pandas, scipy
# and duckdb are the slow ones and they move rarely.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The application. `data/` is absent by gitignore and by .dockerignore, and
# is supposed to be: it arrives on the volume.
COPY api/ ./api/
COPY pipeline/ ./pipeline/
COPY scoring/ ./scoring/
COPY bookmarklet/ ./bookmarklet/

# The built frontend, from the stage above. `WEB_DIST_PATH` points the app at
# it rather than at the `web/dist` a checkout would have -- the image has no
# `web/` tree at all.
COPY --from=web /build/dist ./web/dist
ENV WEB_DIST_PATH=/app/web/dist

# THE VOLUME MOUNTS AT /app/data, NOT /data, AND THAT IS THE WHOLE POINT.
#
# This started as /data with two env vars pointing at it, which moved the two
# databases and quietly left everything else behind. The farm keeps FOUR more
# things under `data/`, and two of them cannot be redirected at all:
#
#   data/espn_state.json   the farm's ESPN login. NOT overridable
#                          (espn_league.STATE_PATH is a constant). Written at
#                          boot from FARM_ESPN_STATE_B64 (api/seed_state.py),
#                          but onto the volume, so it survives a redeploy
#                          even if the variable is later removed.
#   data/leagues/          per-draft databases. NOT overridable
#                          (leagues.LEAGUES_ROOT is a constant).
#   data/draft_corpus.duckdb, data/farm-claims/, data/farm-live/
#                          overridable, but there is no reason to.
#
# Mounting the volume where the code already looks makes every one of them
# persistent with no env var and no code change. The two paths below are then
# only restating the defaults, kept explicit so `docker run` without a volume
# is obvious rather than mysterious.
ENV DRAFT_DB_PATH=/app/data/nfl.duckdb \
    ESPN_CUSTODY_DB_PATH=/app/data/custody/custody.duckdb \
    BILLING_DB_PATH=/app/data/billing.duckdb

# Railway (and every other proxy-fronted host) terminates TLS at the edge, so
# the app itself sees http:// and the credential guard would refuse every
# connect as plaintext. `pipeline/credentials.py` keeps this OFF by default
# because the header is forgeable when nothing overwrites it; a platform
# whose proxy DOES overwrite it is exactly the case the switch is for.
ENV ESPN_CUSTODY_TRUST_FORWARDED_PROTO=1

# A deployment refreshes its own data: nobody is going to type `make refresh`
# at it, and its volume starts empty. The farm is left OFF here and switched
# on per-environment instead -- it needs an ESPN login on the volume, and an
# instance without one should not spend its life retrying. See api/jobs.py.
ENV RUN_REFRESH_ON_BOOT=1

# Draft-session builds run in this many worker processes instead of on the
# request thread (api/live_build.py): a connect is up to 30 s of CPU, and
# under the GIL every poll in the process would wait behind it. Three is
# sized to the memory a cold build takes (~1.7 GB peak each, before the
# column projection in scoring/board.py brought that down) on an 8 GB box.
# `LIVE_FAKE_SOCKET` (replay ESPN's draft socket from a recording, for
# load tests) and `LIVE_DEFAULT_ROOM` (a cookieless room, for the tests
# and a single-user machine) are never set in production.
ENV LIVE_BUILD_WORKERS=3
# One more knob api/live.py reads, left at its default here: `LIVE_MAX_ROOMS`
# (rooms drafting at once before a new connect answers 503 "at capacity";
# default 150). See the README's deploy table.

# `$PORT` is assigned by the platform, so a shell has to expand it -- exec
# form alone would hand uvicorn the literal string. `exec` then replaces that
# shell with uvicorn, which matters on every redeploy: without it the shell is
# PID 1, SIGTERM stops at the shell, and the platform kills the container
# after its grace period instead of letting uvicorn close its connections.
#
# ONE WORKER, NOT A POOL. DuckDB takes a single-writer lock per file: a
# second worker would fail to open the database the first one holds. The same
# constraint caps this service at one replica, which railway.toml states.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
