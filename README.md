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
3. **web** — a React front end in three screens: a landing page that hands
   over the bookmarklet and reports whether this machine is ready to draft,
   the live board that follows your ESPN draft pick by pick, and a full
   player profile behind any name on it.

## The landing page

`/` is the front door. It opens with what the tool does, then proves it: the
top of your own board, drawn from your own DuckDB, under a demo pick clock.
Below that is the bookmarklet, **🏈 Draft Assistant** — drag it to your
bookmarks bar once and every draft after that is one click — and the three steps for using it.

The last band is a readiness strip: when the data last refreshed, whether a
league and its draft history are imported, whether manager models are fitted,
whether a sim has ever run. Draft night is the one night this has to work,
and every way it can fail is something that did or didn't happen days
earlier, so each row that isn't ready names the command that fixes it.

Two endpoints feed it, split by what they cost. `GET /api/landing/status` is
table reads only and paints immediately. `GET /api/landing/preview?limit=N`
builds the board (seconds) and returns just the eight columns the preview
renders, so the page isn't downloading the full 185KB `/api/players` payload
to show twelve rows.

## Player profiles

Any player on the live board opens their profile at `/players/<slug>`: a
compact popup over the board, drawn in the board's own chart language.
Header (name, position chip, team, age, NFL year, bye) with four figures
beside it — your board rank, consensus ADP, tier, VOR. Then a status line for
the two facts that can make everything under them irrelevant, the injury
designation and where he actually stands on his own depth chart, with the gap
to consensus on its right end. Then four season panels — Health, Finish,
Reliable, Per game — and last season by week with that season's game log under
it. Then six cards: schedule softness, his room on the depth chart, the
market's sources, his per-game usage, the line blocking for him, and the
board's picks around his own, with comparable seasons and his news closing it
out. The last thing in the popup is a Draft button, which opens the same
confirm dialog the board's own Draft buttons do.

Every chart in it is the draft board's hover-panel chart
(`draft/Chart.tsx`), fed by the board's own column builders
(`draft/panels.ts`), and that is the whole point of the redesign it replaced:
one picture — a column per season, taller is better, one five-step ramp —
learned once on the board and read everywhere. The profile computes no tone
and no fill of its own, so a season that is green in a hover panel cannot
come out olive three inches away in the popup.

A panel with nothing to draw is dropped rather than drawn as a frame of
dashes, and a payload with no seasons at all says so in a line under the
status: which parts are empty, why — a rookie has not played, a defense has
no per-player weeks, a kicker's weeks are blank in a league that prices no
kicking — and which parts are real. A popup that silently drops half its
cards reads as broken.

"Near you" is the run of board picks around his own, filtered to players who
are still available. It comes from the payload for anyone it has no stat line
to match — a rookie, a kicker, a defense, whose `similar` is board neighbors
by value — and from the room's own ranked board for everyone else, because
for them `similar` is cross-year **stat twins**: other player-seasons nearest
by a weighted z-score distance over per-game production, target/carry share,
and efficiency. Twins are seasons rather than prices, so they get their own
card, "Comparable seasons", each shown next to what that player's *next*
season's PPG turned out to be — a quick gut check on what a comparable stat
line tends to become, with the wider cohort's record ("13 of 19 declined")
in the card's head.

Three things the endpoint serves are drawn nowhere: the game log's per-game
snap share, the per-season positional rank by points per game and its pool,
and each season row's own age, NFL year and median coefficient of variation.
They are cheap to carry and the popup has no room left to spend on them.

It also carries the player's recent news and his injury status, both filled
by `make refresh` (`pipeline/news.py`) so no profile click ever waits on a
network call. `status` sits beside `header` rather than down with the feed,
because "Questionable" is the one field on the page that changes a pick and
it should be one lookup from the name: Sleeper's injury status, body part,
notes and depth-chart slot, or `null` for a player Sleeper does not carry
(every defense). It is never the word "Healthy" — Sleeper does not publish
that, so a player with nothing wrong with him has a null status.

`news` is up to 8 headlines, newest first, each with its link, publication
and published date. Every item also carries an `attribution` marker, and the
card is meant to show the two apart: `espn_athlete_id` means ESPN tagged the
article with this player's athlete id, and `name_team_query` means it came
back from a search for his name and team, which measured about 93-96%
relevant. A guess and a fact are not the same claim. A player nobody wrote
about gets an empty list; so does every defense, because a search for
"Denver Defense" returns whatever the newspaper wrote about the Broncos.

## The league report card

`/leagues/<league id>/report/<season>` is the morning-after page: power
rankings, a graded report card per team, and a profile per manager, written
up in one Haiku call over numbers computed in Python. It is built once and
stored in that league's own database (one row per season, in
`league_reports`), so a link shared with the league costs nothing after the
first build and everybody reads the same words. The room orders it itself
when the last pick of a real draft lands; the league's owner can also build
a past season, or rebuild this one, with `POST
/api/leagues/<id>/report/<season>` — the button on the dashboard card.
Reading is public, building is the owner's. With no `ANTHROPIC_API_KEY` the
page is complete and quiet: every number, no prose (`status:
"numbers_only"`).

## Project structure

- `pipeline/` — data ingestion and refresh workflow (`sources.py`, `db.py`,
  `refresh.py`), plus ESPN draft history import (`espn_league.py`,
  `import_league.py`), the same import for any connected account's league
  (`league_history.py`), manager fitting (`fit_managers.py`), and the
  simulator CLI (`run_sim.py`)
- `scoring/` — PPR calculations, per-factor scoring, composite/VOR/tiers,
  league config (`scoring/config.py`), ESPN-derived league structure
  (`league.py`), the per-manager pick model (`draft_model.py`), the draft
  simulator (`draft_sim.py`), and the report card's numbers
  (`league_report.py`) and prose (`blurbs.py`)
- `api/` — FastAPI backend serving the draft board, drafted-player state,
  draft simulation endpoints, and the league report card (`reports.py`)
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

## Holding other people's ESPN sessions

Only a deployment that other people connect their ESPN accounts to needs this
section. A single-user local setup does not. Nothing is stored until a request
sends an `espn_s2`, and your own bookmarklet does not send one.

What such a deployment ends up holding is a full ESPN account session. It stays
valid for months, and ESPN offers no per-application revocation. ESPN's fan API
returns no email address either, so a user cannot be contacted, cannot recover,
and cannot be warned. The design is
`docs/superpowers/specs/2026-08-23-espn-credential-custody-design.md`; the code
and the reasoning behind each choice are in `pipeline/credentials.py`.

The one property the whole thing exists to give you: a stolen database file is
inert. That holds only while the key stays out of it.

**CAUTION: Keep the key out of the database and out of this repository. A
stolen database with the key in it gives up every stored session.**

**CAUTION: Do not lose the key. Stored sessions become unreadable, and you
cannot tell the users, because ESPN gives you no address for them.**

To set up the store:

1. Generate a key.
   ```bash
   .venv/bin/python -c "from pipeline.credentials import generate_key; print(generate_key())"
   ```
2. Set `ESPN_CUSTODY_KEYS` to that key in the server environment.
3. Serve the API over HTTPS.
4. Start the API.

Copy `.env.example` for the full list. Nothing loads that file for you; export
the values, or give them to your process manager.

| Variable | Default | What it does |
|---|---|---|
| `ESPN_CUSTODY_KEYS` | none | The key, or a versioned list for rotation. Required. |
| `ESPN_CUSTODY_DB_PATH` | `data/custody/custody.duckdb` | Where the two tables live. |
| `ESPN_CUSTODY_TTL_DAYS` | `30` | Idle days before a stored session is deleted. |
| `ESPN_CUSTODY_ALLOW_PLAINTEXT_HTTP` | off | Local development only. Allows plain HTTP. |
| `ESPN_CUSTODY_TRUST_FORWARDED_PROTO` | off | Set only behind a proxy that ends TLS. |

The last two accept `1`, `true`, `yes`, or `on`. Every other value, `0` and
`false` included, leaves the protection on.

If `ESPN_CUSTODY_KEYS` is not set, the server does not fall back to plaintext.
It answers 503 to every request that carries a credential, and it does not
create the database file.

**CAUTION: Run one writer. DuckDB locks the file, so a second worker or a
second replica answers 503 to every custody request.**

To rotate the key:

1. Generate a second key.
2. Set `ESPN_CUSTODY_KEYS="1:<old key>,2:<new key>"`.
3. Restart the API.
4. Wait. Each row moves to the new key when its owner next reconnects.
5. Delete the old key from the list.

Step 4 has no deadline, and step 5 does not wait for it. A row that never moved
becomes unreadable when you delete its key. The server logs a line that names
the version, and the reaper deletes the row inside `ESPN_CUSTODY_TTL_DAYS`.

Live draft records follow the same rule. A `live_session` row is written under
the newest key and read under any key in the list, so a draft that started
before the rotation still restores after it. Those rows expire twelve hours
after they are written, so step 5 waits one draft night for them and not
`ESPN_CUSTODY_TTL_DAYS`.

## Usage

```bash
# 1. Pull the latest data into data/nfl.duckdb
make refresh

# 2. Start the API and the frontend together (Ctrl-C stops both)
make up
```

Open the printed Vite URL. The landing page tells you what's ready and hands
you the bookmarklet — a button labelled **🏈 Draft Assistant** — drag it to
your bookmarks bar once. On draft night, open
your ESPN draft room and click it — your board opens in a new window and
follows the draft from there. A mock draft works the same way, which is the
cheapest way to check the whole path before it matters.

Vite proxies `/api` to the backend. Re-run `make refresh` any time you want
newer stats/ADP — the API reads straight from the DuckDB file, so a restart
isn't required for the underlying data, only for picking up schema changes.

## Deploying it

Everything above assumes one person on one machine. This section is for the
other case: a hosted deployment other people connect their ESPN accounts to.

**One service, not two.** The API process serves the built frontend as well
(`api/static.py`). That is not a packaging preference — the credential cookie
is `SameSite=Lax`, so a frontend on its own domain would stop sending it, and
every route that reads a stored ESPN session would quietly see an anonymous
visitor. One origin keeps the cookie as written.

**One replica, not more.** DuckDB takes a single-writer lock per file, so a
second instance cannot open the database the first one holds. This is the same
lock that stops `make fit-prior` running while the API is up. With
`SUPABASE_DB_URL` set the mutable tables (custody, billing, the live session
records) already live in Postgres; what still pins the service to one replica
is the live draft rooms themselves -- each holds an ESPN socket and its state
in the process that opened it. Splitting those listeners into their own tier
is the step that lifts the limit.

**The volume grows.** Beside `data/nfl.duckdb` (215 MB) the app writes a
read-only snapshot of it for provisioning (another 215 MB, rewritten after
every refresh) and one file per league anybody has ever drafted in under
`data/leagues/` (about 32 MB each). A hundred leagues is 3.2 GB; the 5 GB
volume Railway attaches by default is enough for a season, not for several.

### First deploy (Railway)

1. Create a service from this repository. `railway.toml` selects the
   Dockerfile; no build configuration is needed.
2. Attach a volume mounted at **`/app/data`** — not `/data`. The code looks
   for `data/` relative to its working directory, and two of the things it
   keeps there cannot be redirected by configuration at all: the farm's ESPN
   login (`espn_state.json`) and the per-draft databases in `leagues/`.
   Mounting where the code already looks makes all of it persistent with no
   environment variable and no code change.
3. Set the variables below.
4. Deploy. The healthcheck (`/api/landing/status`) answers 200 against an
   empty database, so the service goes healthy before any data is uploaded.
   The site will be up and honestly empty.
5. Put the data on the volume — see below.

| Variable | Value | Why |
| --- | --- | --- |
| `ESPN_CUSTODY_KEYS` | generated, see above | Without it, nothing can store or read a credential |
| `ESPN_CUSTODY_TRUST_FORWARDED_PROTO` | `1` | **Required behind Railway.** See below |
| `DRAFT_DB_PATH` | `/app/data/nfl.duckdb` | Set by the image; override only to move it |
| `ESPN_CUSTODY_DB_PATH` | `/app/data/custody/custody.duckdb` | Same |
| `RUN_REFRESH_ON_BOOT` | `1` | Set by the image. The server refreshes its own data |
| `REFRESH_MAX_AGE_HOURS` | `24` | How stale the oldest source may get first |
| `RUN_FARM` | `1` | **Off by default.** Needs the login variable below |
| `FARM_CONCURRENCY` | `1` | How many drafts at once. Six matches a full local setup; capped at 8 |
| `FARM_ESPN_STATE_B64` | `make farm-secret` | The farm's ESPN login. A live session — host's variable store only |
| `ANTHROPIC_API_KEY` | `sk-ant-…` | **Off by default.** Absent, every report card is `numbers_only` — see The league report card |
| `STRIPE_SECRET_KEY` | `rk_live_…` | **Off by default.** Absent, every draft is free — see Charging for it |
| `STRIPE_PRICE_ID` | `price_…` | The $9.99 price, made in the Stripe Dashboard |
| `STRIPE_WEBHOOK_SECRET` | `whsec_…` | Required with the key. Without it the webhook refuses everything |
| `PUBLIC_BASE_URL` | `https://…` | Where Stripe sends a buyer back to. Behind a proxy the app cannot work this out itself |
| `SUPABASE_DB_URL` | `postgresql://postgres.<ref>:…@aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require` | **Off by default.** A Supabase Postgres DSN via the **session** pooler, port 5432, `sslmode=require`. Not the 6543 transaction pooler: psycopg's prepared statements break there. Absent, custody, billing and the live session records stay in DuckDB files on the volume |
| `LIVE_BUILD_WORKERS` | `3` | Process-pool size for draft-session builds. Set by the image; 3 is the measured value for an 8 GB box (5 costs 8 GB and slows the polls). `0` runs them inline in the request thread, which is what a checkout and the test suite do |
| `LIVE_MAX_ROOMS` | `150` | Rooms drafting at once before a new connect answers 503 "at capacity". A room already drafting may always reconnect |
| `WARM_ON_BOOT` | `1` | Build the board, profile and game-points caches at boot so the first reader does not pay for them. `0` in the test suite |
| `SEO_WARM`, `DEMO_WARM` | `1` | The ADP pages' aggregation and the demo room's board, built at boot for the same reason |
| `LIVE_DEFAULT_ROOM` | unset | **Never in production.** Lets a request with no room cookie use a shared default room; the tests and a single-user machine that wants the pre-cookie behaviour |
| `LIVE_FAKE_SOCKET` | unset | **Never in production.** Replays a recorded draft instead of ESPN's socket; `make load-test` sets it |

**Rooms and records.** A browser gets one live draft room, keyed by an
`httpOnly` cookie named `espn_live` (twelve hours, renewed on every connect)
that the page asks for before it starts a connect. Every room has its own
listener, its own per-league database connection and its own ranking thread;
a second browser is a second room, and two leaguemates are two rooms on one
league file. What a restart needs to reopen a room -- the league, the team,
the draft token -- is one row per room in the `live_session` table when
`SUPABASE_DB_URL` is set (the whole record encrypted under the custody key,
so the row is a session id, one ciphertext and a timestamp), a JSON file per
room beside the database otherwise; every row younger than twelve
hours is rebuilt at boot, one after another. A room nobody has touched for
three hours is stopped and dropped.

`ESPN_CUSTODY_TRUST_FORWARDED_PROTO` is the one that will waste an evening if
it is missed. Railway terminates TLS at its edge and forwards plain HTTP to
the container, so the credential guard sees `http://` and refuses every
connect as insecure transport — a 400 with a message about plaintext, on a
site that is plainly served over HTTPS. The switch is off by default because
`X-Forwarded-Proto` is forgeable when nothing overwrites it; a proxy that does
overwrite it is exactly the case it exists for.

### Putting a CDN in front

Not set up yet, and it is a nameserver switch rather than a code change. The
app already labels every response with what a cache may do with it — the
hashed bundle is immutable, the document is never cached, the public reads
carry an `s-maxage`, and everything else under `/api` is `private, no-store`
(see `api/http_cache.py`). Cloudflare Free in front of Railway turns those
labels into traffic the container never sees. The steps, and what each one
is for, are in `docs/superpowers/reference/cloudflare-runbook.md`.

### Charging for it

Off unless `STRIPE_SECRET_KEY` is set. Without it every draft is free, the
gate is a no-op and no billing database is ever created — which is what every
local checkout and the whole test suite runs as.

With it, one real league's draft costs $9.99 for the season. Mock drafts stay
free: they are the trial, and charging for the trial is charging for the sales
pitch. `api/billing.py` has the reasoning; the shape is:

1. **Make the price.** Stripe Dashboard → Products → one product, one
   **one-time** price of $9.99 USD. Copy the `price_…` id into
   `STRIPE_PRICE_ID`. Nothing in this codebase creates a price — a price has
   tax and reporting attached to it, and a program that can make one can make
   a second one by accident.
2. **Make the webhook.** Dashboard → Webhooks → point it at
   `https://<your host>/api/billing/webhook` and send exactly four events:
   `checkout.session.completed`, `checkout.session.async_payment_succeeded`,
   `checkout.session.async_payment_failed`, `charge.refunded`. Copy the
   `whsec_…` into `STRIPE_WEBHOOK_SECRET`.
3. **Use a restricted key.** `rk_` with write access to Checkout Sessions and
   nothing else. This integration creates sessions and reads no customer data,
   so a full `sk_` here is a key worth far more to whoever steals it than it
   is to us.

Access is granted by the **webhook**, never by the success page: somebody
whose browser dies during the redirect has still paid, and a success page that
grants access is a success page anybody can visit. Refunds revoke it.

Test it before pointing it at anything live:

```bash
stripe listen --forward-to localhost:8000/api/billing/webhook
# then use the test key + test price, and card 4242 4242 4242 4242
```

**What a purchase is attached to.** There are no accounts here. The
entitlement is keyed by the same HMAC of the ESPN SWID that indexes the
credential store (`CredentialStore.account_id`), so the row identifies nobody
without the custody key, and the email lives on Stripe's side where a receipt
can actually be sent from. The practical consequence: an ESPN account has to
be connected before anything can be bought, because otherwise there is nothing
to attach the payment to.

**Which drafts are free.** Every room ESPN's mock lobby has listed while this
deployment has been running, plus every room we seated somebody in ourselves.
Both are written to `data/billing.duckdb` and survive a redeploy. If ESPN's
directory cannot be read at all, the draft goes through free — blocking
somebody out of a free mock on draft night because a third party is down is a
worse failure than a missed sale.

### Putting the data there

```bash
make deploy-data     # writes deploy/*.gz — stop `make up` first
```

Two databases, and they are not the same kind of thing:

- **`nfl.duckdb`** is rebuildable. Running `python -m pipeline.refresh` on the
  server is usually better than uploading, because it pulls current stats
  rather than shipping this morning's.
- **`draft_corpus.duckdb`** is not. It is hundreds of mock drafts harvested
  over hours by `make farm-mocks`, and the archive page is empty without it.
  Upload it once, then again after any farm run worth keeping.

Both files gzip to about a quarter of their size, which is worth doing over a
volume connection.

### The server keeps itself current

`RUN_REFRESH_ON_BOOT=1` is set in the image, so a fresh deployment pulls its
own data rather than waiting for an upload. It is not a refresh on every
start: the loop reads `meta` and runs only when the **oldest** source is past
`REFRESH_MAX_AGE_HOURS`, so pushing a CSS fix does not cost 223 news queries.
The first run takes minutes, and the site is up and honestly empty for the
duration — the healthcheck answers throughout, so the deploy does not roll
back.

### Farming from the server

`RUN_FARM=1` plays ESPN mock drafts into the corpus from the deployment
itself, one at a time, forever. It needs one thing the server cannot make for
itself: `data/espn_state.json`, the farm account's ESPN login, written by an
actual browser doing an actual sign-in.

That does not get uploaded. It travels as a variable:

```bash
make farm-secret     # prints the value — a live ESPN session, treat it as one
```

Paste the output into `FARM_ESPN_STATE_B64` and set `RUN_FARM=1`. The server
writes the file to the volume at 0600 on boot, before the farm starts.

Rotating an expired login is the same two steps — new value, redeploy — with
no shell and no file transfer. The variable deliberately **overwrites** any
file already on the volume, because a dead session sitting there is exactly
the case this exists to fix. A bad or truncated value costs the farm and
nothing else: the site still comes up, and the boot log says what was wrong.

**How many at once.** `farm(n)` plays its drafts one after another, so being
in six drafts at the same time means six processes, not a bigger `n` — which
is exactly how a laptop does it, with several terminals. `FARM_CONCURRENCY`
spawns that many supervised children; `pipeline/farm_claims.py` keeps them
out of each other's rooms with an atomic claim file per league. It defaults
to 1 so switching the farm on never silently multiplies this deployment's
traffic to ESPN, and is capped at 8 — the cap is about how a single address
holding a dozen public mock seats looks, not about memory (each farm is a
~60 MB process).

Two things to know before switching it on:

- **ESPN sees a datacenter IP.** The farm has only ever run from a home
  connection. Whether ESPN's mock lobby minds is not something this codebase
  can predict — you find out within a draft or two.
- **The volume grows.** Every farmed draft leaves a database in
  `data/leagues/`. Nothing prunes them yet.

The auto-refit daemon stays local. It is a research loop, not something a
web host should be spending CPU on.

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
0.10) but every request to `/api/players` can override them per call, no
restart needed. Editing `DEFAULT_WEIGHTS` in `scoring/config.py` changes the
weights every screen uses.

The composite score converts to **VOR** (value over replacement) by
subtracting, per position, the composite score of the last starter-caliber
player at that position (`scoring/config.py: REPLACEMENT_RANK`, calibrated
to this league's 8-team starting lineup). Kickers and defenses are the one
exception: nobody holds them past the week they use them, so their
replacement is not the last rostered one but whatever is free on waivers,
and both are pinned to a flat rank instead
(`scoring/config.py: STREAMED_REPLACEMENT_RANK` — a calibration, argued in
the comment there, not derived from roster shape). Players are then bucketed into
**tiers** per position, breaking wherever the VOR gap to the next player is
unusually large (mean + one standard deviation of that position's gaps).
The board is ranked by VOR overall.

League size and roster shape (`LEAGUE_TEAMS`, `REPLACEMENT_RANK`) also live
in `scoring/config.py` — update them there if the league format changes.

## Draft simulation

On top of the season-long scouting board, the tool can import your own
league's ESPN draft history, learn how each manager in your league actually
drafts, and simulate the upcoming draft from any slot. This is the study you
run in the days before the draft; the live board is what you use during it.
Neither covers auctions, keeper leagues, or in-draft trades.

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
each of the five candidate features above did to held-out top-1 *and* top-5.
On this league's real history the fitted model scores top-1 23.6%, top-5
58.5%, log-loss 2.868, against a gate of 16% / 53% and a pre-recalibration
baseline of 19.4% / 51.1% / 3.163 measured the same way.

Those first two numbers used to read 24.4% and 59.6%, and the drop is a real
cost of un-truncating `LAMBDA_GRID`: the grid feeds the per-manager fits
inside the backtest, so letting cross-validation shrink harder costs 6 picks
of top-1 and 8 of top-5 while improving log-loss from 2.871 to 2.868. Better
calibrated, very slightly less accurate at the top of the list. If a future refit
doesn't clear that baseline out of sample, the command prints a warning,
and it means what it says: treat the simulator's output as indicative only,
not as a real prediction, until the fit improves. The result is also saved
and served from `/api/model`, so the warning outlives the terminal you
happened to run `make fit-managers` in.

The same fit also measures one league-wide **positional bias**: how far
ahead of or behind the market this league takes each position, by round.
`make fit-managers` prints that table too, and it's worth reading on its
own regardless of the pick model. Right now, for instance, this league
takes kickers roughly 58 ranks ahead of the market in the late rounds and
tight ends roughly 9 ahead in the middle rounds. That's a fact about the
league, not a guess the model is making.

**These eight managers demonstrably differ from each other.** Their fitted
`reach` coefficients run from -5.8 to -10.9, a between-manager spread of
1.92. Under a parametric bootstrap where every manager is made to draft from
the pooled model on their own real choice sets, that spread has a null median
of 0.85 and a maximum of 1.77 across 40 replicates — none of them reaches
1.92, so p is at most 0.024 at that many replicates. The differences are
real. What follows is about whether a per-manager *fit* can convert a real
difference into better prediction, which is a different question.

**Nearly no manager earns a personal model, and the one that clears the bar
clears it by nothing.** A manager needs enough picks and needs their own fit
to beat the pooled fit on held-out seasons; otherwise the simulator falls
back to pooled. `make fit-managers` prints the per-pick log-likelihood gain
that decides it: -0.0000 (Lane Bohman), -0.0001, +0.0002 (Reed1998), -0.0013,
-0.0065, -0.0102, -0.0343, -0.0421 (jtague99). Only Reed1998 is positive, by
0.0002 nats/pick — 0.018 nats across 88 picks. Cross-validation then shrinks
that personal fit onto pooled so hard (lambda 1e5) that its coefficients
differ from the league's by 0.0001 at most and its summary line reads "drafts
close to league average". The chip changes; the simulator's behaviour does
not. **Do not read `uses_personal` as evidence of anything at this margin** —
a bare `gain > 0` test cannot tell +0.0002 from zero, and tightening it into
a real significance test is the obvious next piece of work.

These numbers are much closer to zero than the ones this README carried
before, because the baseline used to cheat: the pooled fit a personal model
was scored against had been trained on all six seasons, including the one
being held out. On this history that head start is worth 0.028 to 0.079
nats/pick — larger than seven of the eight gains it was used to judge. It is
fixed (see `PooledFits`), and the mean gain moved from -0.0259 to -0.0118.

**A smaller personal model works for one manager, and that is a finding, not
a shipping decision.** `make fit-managers REDUCED=1` fits a named handful of
features per manager (`reach`, `fall`, `run`, the position dummies) with
everything else held at pooled, on the same leave-one-season-out yardstick.
The row to read is `nested`: it chooses the subset inside each training fold,
with "stay pooled" among the options, then scores that choice on a season the
choice never saw — so it cannot be gamed by picking whichever subset happens
to fit best. Two of eight come out positive. John Titolo reaches **+0.0527**,
and the inner selection picks `reach` plus the RB/WR/TE dummies in all six
folds, which is not a coin landing the same way six times by accident.
jtague99 is +0.0047, small enough to ignore.

Titolo's number is still not a licence to ship him a personal model. One
manager out of eight clearing a bar, with seven candidate subsets in play, is
exactly the shape a multiplicity argument has to answer and this has not
answered it; and 62% of his total advantage comes from a single season (2023,
+0.186 against a per-fold spread of -0.029 to +0.186). So the gate stays on
the full 15-feature fit, Titolo stays pooled, and the evidence is written
down instead of acted on.

**The position dummies survived a challenge.** The obvious complaint about
them is that "leans RB" is a thin description of a person, and that what a
manager drafts would be better described by the *stat profile* of the players
he takes — usage, efficiency, availability, distance from a career year, all
computed from seasons strictly before each draft. That was built and measured.
Replacing the RB/WR/TE dummies with those four columns costs top-1 accuracy
(23.6% to 22.0%, with nine of the eleven lost picks in the late rounds) while
improving log-loss and top-5, and the descriptive version separates nobody:
one of 32 manager-by-stat cells clears two standard errors, which is what
chance produces at 32 cells. Titolo's advantage, notably, disappears — his
signal was positional. The columns are still computed and carried on every
pool so a seventh season can re-measure cheaply; nothing reads them. Full
numbers, including the by-round split and the per-manager table:
[docs/superpowers/findings/2026-08-11-stat-profile-vs-position-dummies.md](docs/superpowers/findings/2026-08-11-stat-profile-vs-position-dummies.md).

The defensible summary is narrower than "six seasons isn't enough for a
personal model of any size", which is what this README used to say and which
the evidence does not support: **at ~86 picks a per-manager fit does not
reliably convert a real difference into better held-out log-likelihood.**
Sometimes it does. Not dependably, and not in a way that survives asking how
many chances it had.

**What `GET /api/managers/history` reports instead is measured, not
fitted.** It carries statistics counted straight
from each manager's real picks: what they open a draft with and in how many
of their drafts, how many picks ahead of or behind the market board they
take players and which rounds that's strongest in, which positions they jump
the board hardest for, and the typical round of their first QB, TE, K and
DST. The reach numbers exclude kickers and defenses on purpose: a kicker is
ranked around 130th and taken in the double-digit rounds because every roster
needs exactly one, so every kicker pick scores a huge "reach" by construction
— left in, it flipped two managers' headline number from negative to positive
and printed the same league-wide "K +49..+67" on all eight cards. None of it
depends on a coefficient generalizing, so all of it stays
true while the forecast underneath runs on league-average coefficients —
which is what the summary line now says, instead of calling a manager with
six drafts on record "not enough signal". `make fit-managers` precomputes
this into a `manager_tendencies` table and prints it, and `/api/managers/
history` serves it.

### Running a simulation

```bash
make sim SLOT=4 ROLLOUTS=300
```

Pass any slot you like: the draft order is seeded from ESPN's published one
(`GET /api/draft-order`) but a run takes the slot you give it, so "what if
I'm picking third instead" is answerable before the real order is out.

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
so the run fails with that message instead.

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

A simulation run also fills in a full board, not just your own picks: one
predicted player at every pick of the whole draft, rounds down the side and
managers across the top. It's built from the same rollouts as the `Avail%`
and `ΔEV` columns, `predict_board` in `scoring/draft_sim.py`, so it doesn't
form a second opinion; it adds roughly 4 seconds to a run that already takes
about 50, and the results are written to a `sim_board` table and served at
`GET /api/sim/board`.

Nothing in the web app renders this today — the grid page that did was
removed along with the rest of the standalone research tool. The table and
the endpoint stayed because the run produces them either way, and because
the reasoning below is the part worth keeping.

Each cell is the model's single most likely player at that pick, with the
second and third most likely carried alongside it.

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

Any player on the live board opens the **player card**: a condensed,
draft-night version of the full profile, not the profile itself. It leads with projected
points per game next to the player's recent actual PPG (a wide gap between
the two is flagged, since a projection well ahead of recent real production
is a bet, not a fact), a row of Rank/Tier/Mkt/Edge plus that player's
`Avail%` and `ΔEV` from the same run (`ΔEV` only for the handful of players
scored at your next pick), a row of critical numbers specific to his
position (finish, role, snap share, games played, strength of schedule,
whichever apply), and an 18-cell strength-of-schedule strip, one cell per
week, coloured soft to tough with the bye week marked. A link to the full
profile page covers anything the card leaves out, and a mark-drafted button
means a pick doesn't send you anywhere else.

Because the grid is built on the same fitted manager models as the rest of
the simulator, the backtest line `make fit-managers` prints matters even
more here than for `Avail%`/`ΔEV`: a full, confident-looking board is easy to
over-trust, and it's only as good as that line says the model is.

## Data sources and quirks

- **Weekly stats / snap counts / depth charts / schedules** come from
  `nfl_data_py` / nflverse. History covers the three prior seasons
  (`scoring/config.py: HISTORY_SEASONS`), used for production and durability;
  the current season's schedule feeds environment, schedule strength, and bye
  weeks.
- **Market consensus** blends up to three independent sources — Fantasy
  Football Calculator (FFC) ADP, ESPN ADP, and FantasyPros' expert consensus
  rankings (ECR) — averaged per player into the board's **Mkt** rank (the
  profile page's Market section shows each source's raw rank so you can see
  where they agree or don't, and how many actually had the player — a Mkt
  rank can come from just one source, with no warning anywhere else). FFC is explicitly a
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
