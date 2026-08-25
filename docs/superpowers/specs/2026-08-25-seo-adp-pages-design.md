# Search presence: page hygiene and server-rendered ADP pages

**Date:** 2026-08-25
**Status:** approved in conversation, awaiting spec review

## The problem

espnfantasydraft.com is one client-rendered page. As Google sees it today:

- `index.html` carries a `<title>` and nothing else. No description, no
  canonical, no Open Graph, no structured data.
- `/robots.txt` and `/sitemap.xml` do not exist. The SPA catch-all
  (`api/static.py`) answers both with `index.html` as `text/html`, status
  200, which Google reads as a broken sitemap.
- The landing page's first screen is a live mock draft. The only prose is
  the `Benefits` section. `/archive` is gated behind an account; `/mocks`
  is a data console. There is nothing to rank for except the brand name.

What the site has that nothing else does is the draft corpus:
`data/draft_corpus.duckdb`, 663 real ESPN mock drafts and 84,769 picks as
of today, growing while the farm runs, all one shape (8 teams, 16 rounds,
PPR). Per-player, per-round and per-position pages built from it are
unique, refreshed daily, and answer long-tail queries ("D'Andre Swift ADP
ESPN", "who goes in round 3 of an ESPN mock") that nobody competes for.

## Decision

Two parts, shipped together. A blog is explicitly deferred.

1. **Hygiene** in the SPA build: meta, Open Graph, structured data,
   `robots.txt`, per-route titles, a real text section with an FAQ.
2. **ADP pages** rendered by FastAPI as plain HTML from the corpus, with a
   generated `sitemap.xml`. No JavaScript on these pages.

The archive split, decided with the owner: corpus **aggregates** (ADP,
ranges, round and position tables) are public. The per-seat and per-turn
analysis on `/archive`, and the live room, stay behind the account.

## Part 1: hygiene

### `web/index.html`

Add to `<head>`:

- `<meta name="description">` -- one sentence: a live draft assistant for
  ESPN fantasy football leagues that ranks the board from real mock drafts.
- `<link rel="canonical" href="https://espnfantasydraft.com/">`
- Open Graph and Twitter card tags: `og:title`, `og:description`,
  `og:url`, `og:image` = `https://espnfantasydraft.com/demo-poster.jpg`
  (already in `web/public`), `og:type=website`, `twitter:card=summary_large_image`.
- `<meta name="theme-color">` in the app's background colour.
- JSON-LD `SoftwareApplication`: name "ESPN Draft Assist", `applicationCategory`
  "SportsApplication", `operatingSystem` "Web", `offers` price 9.99 USD,
  `url`.

The SPA rewrites `<title>` per route (below); the static head is the
landing page's.

### `web/public/robots.txt`

    User-agent: *
    Allow: /
    Disallow: /api/
    Disallow: /draft
    Disallow: /live
    Disallow: /room/
    Sitemap: https://espnfantasydraft.com/sitemap.xml

Served as a file by the existing catch-all (files before the fallback).

### Per-route document meta

A hook `useDocumentMeta({ title, description, noindex })` in
`web/src/lib/documentMeta.ts`. Sets `document.title`, upserts
`<meta name="description">` and `<link rel="canonical">`, and adds or
removes `<meta name="robots" content="noindex">`. Used by:

| Route | Title | noindex |
|---|---|---|
| `/` | ESPN Draft Assist -- a live draft assistant for ESPN fantasy football | no |
| `/mocks` | Recorded ESPN mock drafts -- ESPN Draft Assist | no |
| `/archive`, `/archive/data` | The draft archive -- ESPN Draft Assist | no |
| `/draft`, `/live`, `/room/:id` | (unchanged) | yes |

### Landing page text section

A new `web/src/components/Explainer.tsx`, rendered on the landing page
between `Benefits` and the footer, visible to a signed-out visitor only
(same branch as `Welcome`). Plain HTML, no data fetches:

- An `<h1>` (the page has none today): "A draft assistant for ESPN
  fantasy football, trained on real ESPN mock drafts."
- Three short paragraphs: what it does, how the bookmark connects it to
  the ESPN draft room, what it costs.
- An FAQ of six questions in `<details>` elements, with matching
  `FAQPage` JSON-LD injected by the component. Questions: does it work
  with my league; is it allowed; what does the bookmark do; does it see
  my password; what does it cost; what are the mock drafts.

Copy is drafted in the implementation and the owner edits it.

## Part 2: ADP pages

### Module and registration

`api/seo.py`, `register_seo_routes(app, conn)`. Registered in
`api/main.py` immediately before `register_spa(app)`. `conn` is the
universal connection the market routes already receive (names come from
`players`; ESPN's published ADP from `espn_adp`).

Templates in `api/templates/` rendered with Jinja2 (new entry in
`requirements.txt`; already a FastAPI extra). One base template with the
inline stylesheet, header (mark, "ESPN Draft Assist", link to `/`, CTA
"Try it on your draft"), and footer. No JavaScript anywhere on these pages.

### Data

One aggregation, `corpus_adp()`, cached ten minutes through
`api.market._cached` with the corpus connection from `api.market._corpus`
and the shape from `api.market._shape`. It reads `draft_log_pick` joined to
`draft_log` for the dominant shape and returns, per `player_id`:

- `adp` (mean `pick_no`), `median`, `p10`, `p90`, `min`, `max`
- `taken_share` = drafts he was taken in / drafts total
- `round_mode` = the round he goes in most often
- `position`, `pos_rank` (by adp within position)
- `picks` = histogram of `pick_no` for the chart

Drafts total and the latest `recorded_at` are returned alongside for the
"from N drafts, updated D" line and the sitemap's `lastmod`.

Names, headshots and teams: `api.market._names` and `_board_names` for
names (already handle `adp_*` synthetic ids); team from `depth_charts`
(latest `dt` per `gsis_id`) with `espn_adp.team` as fallback. ESPN's
published ADP from `espn_adp` via the sleeper crosswalk, as
`scoring/profile._espn_projected_usage` already does.

Only players taken in at least 2% of drafts get a page. Below that a
player is noise, and a page with "taken in 1 of 663 drafts" is thin
content Google penalises the whole site for.

### Slugs

`slug(name)`: lowercase, ASCII-fold, strip everything but `[a-z0-9]` and
spaces, spaces to hyphens. "D'Andre Swift" -> `dandre-swift`. Defenses:
"Seattle Seahawks D/ST" -> `seattle-seahawks-dst`. Collisions (two players
with the same name) get `-2`, `-3` in `player_id` order, so a slug never
moves once assigned within a corpus snapshot. The slug -> id map is built
with the aggregation and cached with it.

### Pages

| Path | Title | Body |
|---|---|---|
| `/adp` | ESPN Mock Draft ADP 2026 (8-team PPR) | Provenance line. Table of every qualifying player: rank, name (link), pos, team, ADP, range (p10-p90), % drafted, pos rank. Links to round and position pages. |
| `/adp/<slug>` | `<Name> ADP -- ESPN mock drafts 2026` | Headshot, name, pos/team. Figures: ADP, range, most common round, % drafted, position rank, ESPN's published ADP beside the mock ADP. Pick-distribution chart as inline SVG (one bar per pick 1..128). "Goes around" -- the four players either side by ADP, linked. Breadcrumb `ADP > Name`. CTA. |
| `/adp/round/<n>` (1-16) | Round `n` of an ESPN mock draft -- who goes there | Players whose `round_mode` is `n`, plus everyone with p10-p90 crossing the round, ordered by ADP. Previous/next round links. |
| `/adp/<pos>` (qb, rb, wr, te, k, dst) | `<POS>` ADP -- ESPN mock drafts 2026 | The index table filtered to the position. |
| `/sitemap.xml` | -- | `/`, `/mocks`, `/adp`, six position pages, sixteen round pages, every player page. `lastmod` = latest `recorded_at`, date only. |

Every page: unique `<title>`, `<meta name="description">` with the
player's own figures in it ("ADP 46.2, taken in round 6 in 71% of 663
ESPN mock drafts"), canonical, `BreadcrumbList` JSON-LD. Position tables
and the index share one template with a filter.

Unknown slug, round outside 1-16, unknown position: 404 with the base
template's small "not drafted in any recorded mock" page, `noindex`.

An empty corpus (a fresh deployment, no farm run yet): `/adp` renders the
provenance line as "no drafts recorded yet" and an empty table, 200;
player pages 404; the sitemap lists only the static pages.

### Error handling

`_corpus()` raising 503 (farm mid-write) propagates as the same 503 the
market routes give; a crawler retries. Anything the name resolution cannot
answer prints the id, as the archive does.

### Performance

The aggregation is one DuckDB query over ~85k rows (~50 ms) plus name
resolution, cached ten minutes. Pages render from the cached dict; no page
hits the database on its own. Template rendering is microseconds.

## Testing

`tests/test_seo.py`, with a fixture corpus of a few drafts written through
`pipeline.draft_log` (as `tests/test_market.py` does) and a `players`
table:

- `/adp` is 200, `text/html`, contains each fixture player's name and the
  drafts count.
- A player page is 200 with the ADP figure and the breadcrumb; slug
  helper: `D'Andre Swift` -> `dandre-swift`, collision -> `-2`.
- Round and position pages 200 and filtered correctly; round 17 and
  `/adp/ol` are 404 with `noindex`.
- `/sitemap.xml` is `application/xml`, parses, lists every page once,
  `lastmod` is a date.
- Below-threshold player has no page and is not in the sitemap.
- Empty corpus: `/adp` 200, sitemap has only the static pages.
- Registration order: with the SPA mounted, `/mocks` still returns
  `index.html` and `/adp` returns the server page.

Frontend: `npm run build` (tsc) and a headless screenshot of `/`
(explainer + FAQ visible, signed out) and of `/adp` and one player page.

## Out of scope

- A blog. When wanted: `content/*.md` -> `/blog/<slug>` through the same
  base template, linked from the ADP pages. Not now.
- Pre-rendering the SPA routes themselves.
- Format variants (10/12-team, standard). The corpus has one shape; the
  pages say so. If the farm ever plays other shapes, `_shape` picks the
  dominant one and the rest are excluded, same as the archive.
- Search Console: verification and sitemap submission are the owner's
  actions after deploy. A verification `<meta>` goes into `index.html`
  when the owner has the token.
