# Search Presence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make espnfantasydraft.com indexable -- proper page metadata, robots and sitemap, a text section on the landing page -- and publish server-rendered ADP pages built from the mock-draft corpus.

**Architecture:** Part 1 is static: `web/index.html` metadata, `web/public/robots.txt`, a `useDocumentMeta` hook for per-route titles, and an `Explainer` component with an FAQ on the landing page. Part 2 is a new FastAPI module `api/seo.py` that aggregates `draft_corpus.duckdb` once (cached ten minutes through `api.market._cached`), renders Jinja2 templates in `api/templates/` with no JavaScript, and generates `/sitemap.xml`. It is registered immediately before the SPA catch-all in `api/main.py`.

**Tech Stack:** FastAPI, Jinja2 (new dependency), DuckDB, pytest + `fastapi.testclient`; React 18 + TypeScript + Vite on the frontend.

**Spec:** `docs/superpowers/specs/2026-08-25-seo-adp-pages-design.md`

## Global Constraints

- Production origin is `https://espnfantasydraft.com` -- every canonical, OG url and sitemap entry uses it verbatim.
- The corpus is one shape (8 teams, 16 rounds, PPR); pages read the shape from `api.market._shape` and state it, never assume it.
- A player gets a page only when taken in at least 2% of drafts (`MIN_SHARE = 0.02`).
- ADP pages carry **no JavaScript**. Styling is one inline stylesheet using the app's tokens: `--bg-0 #0a0b0e`, `--bg-1 #121319`, `--bg-2 #1b1d25`, `--border #262932`, `--text-1 #e8eaef`, `--text-2 #98a0b0`, `--text-3 #565b69`, `--accent #ffb454`, `--ok #0ca30c`, `--fail #d03b3b`, radius 3px, mono font stack `'JetBrains Mono', ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace`.
- Public: corpus aggregates. Still gated: `/archive` per-seat analysis and the live room. Nothing in this plan reads a session.
- `/draft`, `/live`, `/room/:id` get `noindex`. `/api/` is disallowed in robots.
- Commit after every task. Commit messages: normal prose, no caveman.
- The product's name in copy is "ESPN Draft Assist".

---

## File Structure

| File | Responsibility |
|---|---|
| `web/index.html` (modify) | Static head: description, canonical, OG/Twitter, theme-color, `SoftwareApplication` JSON-LD |
| `web/public/robots.txt` (create) | Crawl rules and sitemap pointer |
| `web/src/lib/documentMeta.ts` (create) | `useDocumentMeta` hook: title, description, canonical, noindex per route |
| `web/src/pages/Landing.tsx`, `MockDrafts.tsx`, `Market.tsx`, `ArchiveData.tsx`, `DraftRoom.tsx`, `Live.tsx`, `WaitingRoomPage.tsx` (modify) | One `useDocumentMeta` call each |
| `web/src/components/Explainer.tsx` (create) | Landing text section: h1, three paragraphs, FAQ with `FAQPage` JSON-LD |
| `web/src/landing.css` (modify) | `.xp-*` rules for the explainer |
| `tests/test_seo_static.py` (create) | Guards the static head and robots.txt |
| `requirements.txt` (modify) | `jinja2>=3.1` |
| `api/seo.py` (create) | `slug`, `build_adp`, `register_seo_routes`, sitemap |
| `api/templates/base.html`, `adp_index.html`, `adp_player.html`, `adp_round.html`, `adp_missing.html` (create) | Jinja2 templates |
| `api/main.py` (modify, ~line 849) | Register SEO routes before `register_spa` |
| `tests/test_seo.py` (create) | Data layer, pages, sitemap, registration order |

---

### Task 1: Static head and robots.txt

**Files:**
- Modify: `web/index.html`
- Create: `web/public/robots.txt`
- Test: `tests/test_seo_static.py`

**Interfaces:**
- Produces: nothing programmatic. Later tasks rely on `robots.txt` naming `/sitemap.xml`, which Task 6 serves.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seo_static.py
"""The parts of the site's search presence that are files, not code.

Read straight off the source tree: a build step cannot lose what was never
in `web/index.html`, and a test that asserts on the built `dist` would be
skipped on every checkout without one."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
ROBOTS = ROOT / "web" / "public" / "robots.txt"


def test_the_document_head_describes_the_page():
    assert '<meta name="description"' in INDEX
    assert '<link rel="canonical" href="https://espnfantasydraft.com/"' in INDEX
    assert 'property="og:title"' in INDEX
    assert 'property="og:image" content="https://espnfantasydraft.com/demo-poster.jpg"' in INDEX
    assert 'name="twitter:card" content="summary_large_image"' in INDEX
    assert '<meta name="theme-color" content="#0a0b0e"' in INDEX


def test_the_document_carries_software_application_structured_data():
    assert 'application/ld+json' in INDEX
    assert '"@type": "SoftwareApplication"' in INDEX
    assert '"name": "ESPN Draft Assist"' in INDEX
    assert '"price": "9.99"' in INDEX


def test_robots_allows_the_site_and_hides_the_app_surfaces():
    text = ROBOTS.read_text(encoding="utf-8")
    assert "User-agent: *" in text
    for path in ("/api/", "/draft", "/live", "/room/"):
        assert f"Disallow: {path}" in text
    assert "Sitemap: https://espnfantasydraft.com/sitemap.xml" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_seo_static.py -q`
Expected: 3 failed (`assert '<meta name="description"' in INDEX`, then the robots file not found).

- [ ] **Step 3: Write `web/index.html`**

Replace the file with:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#0a0b0e" />
    <title>ESPN Draft Assist</title>
    <meta name="description" content="A live draft assistant for ESPN fantasy football. It sits beside your ESPN draft room and ranks the board from real recorded ESPN mock drafts: who is still there at your next pick, and who is not." />
    <link rel="canonical" href="https://espnfantasydraft.com/" />
    <meta property="og:type" content="website" />
    <meta property="og:site_name" content="ESPN Draft Assist" />
    <meta property="og:title" content="ESPN Draft Assist" />
    <meta property="og:description" content="A live draft assistant for ESPN fantasy football, ranked from real recorded ESPN mock drafts." />
    <meta property="og:url" content="https://espnfantasydraft.com/" />
    <meta property="og:image" content="https://espnfantasydraft.com/demo-poster.jpg" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="ESPN Draft Assist" />
    <meta name="twitter:description" content="A live draft assistant for ESPN fantasy football, ranked from real recorded ESPN mock drafts." />
    <meta name="twitter:image" content="https://espnfantasydraft.com/demo-poster.jpg" />
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "SoftwareApplication",
      "name": "ESPN Draft Assist",
      "url": "https://espnfantasydraft.com/",
      "applicationCategory": "SportsApplication",
      "operatingSystem": "Web",
      "description": "A live draft assistant for ESPN fantasy football leagues, ranked from real recorded ESPN mock drafts.",
      "offers": {
        "@type": "Offer",
        "price": "9.99",
        "priceCurrency": "USD"
      }
    }
    </script>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 4: Write `web/public/robots.txt`**

```
User-agent: *
Allow: /
Disallow: /api/
Disallow: /draft
Disallow: /live
Disallow: /room/

Sitemap: https://espnfantasydraft.com/sitemap.xml
```

- [ ] **Step 5: Run tests and the web build**

Run: `.venv/bin/python -m pytest tests/test_seo_static.py -q && (cd web && npm run build 2>&1 | tail -2)`
Expected: `3 passed`; build ends `✓ built`.

- [ ] **Step 6: Commit**

```bash
git add web/index.html web/public/robots.txt tests/test_seo_static.py
git commit -m "The document head describes the page, and robots.txt exists"
```

---

### Task 2: Per-route document metadata

**Files:**
- Create: `web/src/lib/documentMeta.ts`
- Modify: `web/src/pages/Landing.tsx` (inside the component, before the first early return), `web/src/pages/MockDrafts.tsx`, `web/src/pages/Market.tsx`, `web/src/pages/ArchiveData.tsx`, `web/src/pages/DraftRoom.tsx`, `web/src/pages/Live.tsx`, `web/src/pages/WaitingRoomPage.tsx`

**Interfaces:**
- Produces: `useDocumentMeta(meta: { title: string; description?: string; canonical?: string; noindex?: boolean }): void`

No unit-test runner exists for the frontend (`web/package.json` has no `test` script); `tsc` through `npm run build` is the gate, and the headless check in Task 7 reads the rendered head.

- [ ] **Step 1: Write the hook**

```ts
// web/src/lib/documentMeta.ts
import { useEffect } from 'react'

// One hook, one place, for what a route says about itself to a crawler
// and a browser tab: the title, the description, the canonical URL, and
// whether the page is a page at all. The draft room and the live board are
// application surfaces -- a search result that landed somebody inside a
// draft with no draft would be worse than no result -- so they ask not to
// be indexed, and everything else asks to be.
//
// `web/index.html` carries the landing page's static head for the crawler
// that never runs JavaScript. This hook is what the SPA does once it has.

const ORIGIN = 'https://espnfantasydraft.com'

function upsert(selector: string, create: () => HTMLElement): HTMLElement {
  const found = document.head.querySelector<HTMLElement>(selector)
  if (found) return found
  const made = create()
  document.head.appendChild(made)
  return made
}

function meta(name: string): HTMLMetaElement {
  return upsert(`meta[name="${name}"]`, () => {
    const el = document.createElement('meta')
    el.setAttribute('name', name)
    return el
  }) as HTMLMetaElement
}

export function useDocumentMeta({ title, description, canonical, noindex = false }: {
  title: string
  description?: string
  canonical?: string
  noindex?: boolean
}): void {
  useEffect(() => {
    document.title = title
    if (description !== undefined) meta('description').setAttribute('content', description)
    const link = upsert('link[rel="canonical"]', () => {
      const el = document.createElement('link')
      el.setAttribute('rel', 'canonical')
      return el
    })
    link.setAttribute('href', canonical ?? ORIGIN + window.location.pathname)
    const robots = document.head.querySelector('meta[name="robots"]')
    if (noindex) {
      meta('robots').setAttribute('content', 'noindex')
    } else if (robots) {
      robots.remove()
    }
  }, [title, description, canonical, noindex])
}
```

- [ ] **Step 2: Call it from each route**

In each page component, add the import and one call at the top of the component body (before any early `return`; hooks must run unconditionally).

`web/src/pages/Landing.tsx` -- the import goes with the other `../lib` imports; the call goes right after the `useNavigate()` line (`const navigate = useNavigate()`):

```ts
import { useDocumentMeta } from '../lib/documentMeta'
// ...
  useDocumentMeta({
    title: 'ESPN Draft Assist – a live draft assistant for ESPN fantasy football',
    description: 'A live draft assistant for ESPN fantasy football. It sits beside your ESPN draft room and ranks the board from real recorded ESPN mock drafts.',
    canonical: 'https://espnfantasydraft.com/',
  })
```

`web/src/pages/MockDrafts.tsx`:

```ts
  useDocumentMeta({
    title: 'Recorded ESPN mock drafts – ESPN Draft Assist',
    description: 'Every recorded ESPN mock draft, pick by pick, with the board as it stood at each turn.',
  })
```

`web/src/pages/Market.tsx` and `web/src/pages/ArchiveData.tsx`:

```ts
  useDocumentMeta({
    title: 'The draft archive – ESPN Draft Assist',
    description: 'What hundreds of recorded ESPN mock drafts do from each seat: who is taken at your turn, and who is still there at the next one.',
  })
```

`web/src/pages/DraftRoom.tsx`, `web/src/pages/Live.tsx`, `web/src/pages/WaitingRoomPage.tsx`:

```ts
  useDocumentMeta({ title: 'ESPN Draft Assist', noindex: true })
```

Find the component function in each file with `grep -n "^export default function\|^function [A-Z]" <file>` and put the call as the first statement inside it (after any destructuring of props).

- [ ] **Step 3: Build**

Run: `cd web && npm run build 2>&1 | tail -3`
Expected: `✓ built`. A `tsc` error about hooks order means the call landed after an early return -- move it up.

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/documentMeta.ts web/src/pages
git commit -m "Every route names itself, and the app surfaces ask not to be indexed"
```

---

### Task 3: The explainer and FAQ on the landing page

**Files:**
- Create: `web/src/components/Explainer.tsx`
- Modify: `web/src/pages/Landing.tsx` (render between `<Benefits .../>` and `<footer className="lp-foot">`)
- Modify: `web/src/landing.css` (append)

**Interfaces:**
- Produces: `Explainer` default export, props `{ onStart: () => void }`.
- Consumes: `toSetup` from `Landing.tsx` (the existing handler `Welcome` receives as `onStart`).

- [ ] **Step 1: Write the component**

```tsx
// web/src/components/Explainer.tsx
import { useEffect } from 'react'

// The page in words, for the reader who wants them and the crawler that
// needs them. The first screen is a live draft and the second is drawn
// figures; neither says in plain text what this is. This does, once, below
// both, and it is the only <h1> on the page.
//
// The FAQ is real <details> elements and also FAQPage structured data, so
// the same six answers show under the result in a search as well as here.

const FAQ: { q: string; a: string }[] = [
  {
    q: 'Does it work with my ESPN league?',
    a: 'Yes, any ESPN fantasy football league whose draft room you can open in a browser — public or private, snake or auction-free formats, any roster settings. It reads your league’s scoring and roster slots and ranks for those.',
  },
  {
    q: 'Is this allowed?',
    a: 'It is a second browser tab beside your draft. It never picks for you, never touches the draft room, and never logs in as you anywhere. ESPN’s own draft tools do the same job with worse numbers.',
  },
  {
    q: 'What does the bookmark do?',
    a: 'One click on any ESPN fantasy page hands this site the draft session ESPN already gave your browser, so the board here follows every pick in your room live. Nothing is installed.',
  },
  {
    q: 'Does it see my ESPN password?',
    a: 'No. It never sees a password. It holds the same session cookie ESPN gave your browser, encrypted, and only for as long as your draft needs it. You can revoke it from the dashboard at any time.',
  },
  {
    q: 'What does it cost?',
    a: 'Mock drafts are free, as many as you like. A real draft is $9.99, once, paid when you connect the league that is about to draft.',
  },
  {
    q: 'Where do the rankings come from?',
    a: 'From hundreds of real ESPN mock drafts this site records every day, plus consensus ADP and projections. It knows who actually goes at pick 41 in an ESPN room, not just who a list says should.',
  },
]

export default function Explainer({ onStart }: { onStart: () => void }) {
  useEffect(() => {
    const script = document.createElement('script')
    script.type = 'application/ld+json'
    script.text = JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'FAQPage',
      mainEntity: FAQ.map(({ q, a }) => ({
        '@type': 'Question',
        name: q,
        acceptedAnswer: { '@type': 'Answer', text: a },
      })),
    })
    document.head.appendChild(script)
    return () => { script.remove() }
  }, [])

  return (
    <section className="xp" aria-labelledby="xp-h1">
      <h1 id="xp-h1" className="xp-h1">
        A draft assistant for ESPN fantasy football, trained on real ESPN mock drafts.
      </h1>
      <div className="xp-body">
        <p>
          ESPN Draft Assist is a live board that sits beside your ESPN draft room.
          Every pick in your room lands here as it happens, and the board re-ranks
          around it: who is worth taking now, who will still be there at your next
          turn, and who will not.
        </p>
        <p>
          It is ranked from what ESPN rooms actually do. This site records hundreds
          of real ESPN mock drafts every day, so when it says a player usually goes
          at pick 41, that is where he has been going this week, in rooms like yours.
        </p>
        <p>
          Setup is a bookmark. Drag it to your bar, click it once on ESPN, and your
          leagues appear here with a button to open the board when your draft starts.
          Mock drafts are free; a real draft is $9.99, once.
        </p>
      </div>
      <h2 className="xp-h2">Questions</h2>
      <div className="xp-faq">
        {FAQ.map(({ q, a }) => (
          <details key={q} className="xp-q">
            <summary>{q}</summary>
            <p>{a}</p>
          </details>
        ))}
      </div>
      <button type="button" className="lp-cta xp-cta" onClick={onStart}>
        Try a mock draft
      </button>
    </section>
  )
}
```

- [ ] **Step 2: Render it on the landing page**

In `web/src/pages/Landing.tsx`, add the import beside the other components:

```ts
import Explainer from '../components/Explainer'
```

and render it between the `Benefits` line and the footer:

```tsx
      <Benefits onNeedAccount={() => setAskedAt((n) => n + 1)} />

      {/* The page in plain words, for readers and crawlers alike: the only
          <h1>, and the FAQ (also FAQPage structured data). Below the room
          and the drawn figures, which make the case faster for anyone who
          watched them. */}
      <Explainer onStart={toSetup} />

      <footer className="lp-foot">
```

This branch of `Landing` renders only for a signed-out visitor -- the connected case returned `<Dashboard>` above -- so no further condition is needed.

- [ ] **Step 3: Style it**

Append to `web/src/landing.css`:

```css
/* -- the explainer: the page in words, and the FAQ ------------------------ */

.xp {
  max-width: 720px;
  margin: 0 auto;
  padding: 48px 24px 32px;
}

.xp-h1 {
  margin: 0 0 20px;
  font-size: 26px;
  line-height: 1.25;
  font-weight: 600;
  color: var(--text-1);
  letter-spacing: -0.01em;
}

.xp-body p {
  margin: 0 0 14px;
  font-size: 15px;
  line-height: 1.55;
  color: var(--text-2);
}

.xp-h2 {
  margin: 32px 0 10px;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-3);
}

.xp-faq {
  margin: 0;
  border-top: 1px solid var(--border);
}

.xp-q {
  border-bottom: 1px solid var(--border);
}

.xp-q > summary {
  cursor: pointer;
  padding: 12px 0;
  font-size: 15px;
  color: var(--text-1);
  list-style: none;
}

.xp-q > summary::-webkit-details-marker { display: none; }

.xp-q > summary::before {
  content: '+';
  display: inline-block;
  width: 18px;
  color: var(--accent);
  font-family: var(--font-mono);
}

.xp-q[open] > summary::before { content: '–'; }

.xp-q > p {
  margin: 0 0 14px 18px;
  font-size: 14px;
  line-height: 1.55;
  color: var(--text-2);
}

.xp-cta {
  margin-top: 28px;
}
```

- [ ] **Step 4: Build**

Run: `cd web && npm run build 2>&1 | tail -3`
Expected: `✓ built`.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/Explainer.tsx web/src/pages/Landing.tsx web/src/landing.css
git commit -m "The landing page says what it is, in words, with an FAQ"
```

---

### Task 4: The corpus aggregation and slugs (`api/seo.py`, data layer)

**Files:**
- Modify: `requirements.txt` (add `jinja2>=3.1` -- installed here so the module can import it in Task 5)
- Create: `api/seo.py`
- Test: `tests/test_seo.py`

**Interfaces:**
- Consumes: `api.market._corpus()`, `api.market._shape(conn) -> (teams, rounds)`, `api.market._names(conn) -> {pid: {"name", "headshot"}}`, `api.market._board_names(conn, missing: set) -> same shape`, `api.market._cached(key, build)`, `pipeline.db.read_table(conn, name) -> DataFrame`, `scoring.profile._norm_name(name) -> str`.
- Produces:
  - `slug(name: str) -> str`
  - `build_adp(conn) -> dict` with keys `players: list[dict]`, `by_slug: dict[str, dict]`, `drafts: int`, `teams: int`, `rounds: int`, `updated: date | None`. Each player dict: `player_id, slug, name, headshot, position, team, adp, median, p10, p90, low, high, taken, share, round_mode, rank, pos_rank, espn_adp, hist` (`hist` is a list of `teams*rounds` ints, index = pick_no - 1).
  - `adp_data(conn) -> dict` -- `build_adp` through `market._cached("seo-adp", ...)`.
  - `MIN_SHARE = 0.02`, `POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")`.

- [ ] **Step 1: Add the dependency**

Append to `requirements.txt`:

```
# Jinja2, for api/seo.py -- the server-rendered ADP pages a crawler reads
# without running the SPA. FastAPI's own templating extra; pinned to the
# major so a template syntax change cannot arrive with a redeploy.
jinja2>=3.1,<4
```

Run: `.venv/bin/pip install "jinja2>=3.1,<4" -q && .venv/bin/python -c "import jinja2; print(jinja2.__version__)"`
Expected: a `3.1.x` version printed.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_seo.py
"""The pages a search engine reads: ADP from the corpus, rendered as HTML."""
import xml.etree.ElementTree as ET

import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import market, seo
from pipeline import draft_log as dl


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Ten drafts, 4 teams x 2 rounds (8 picks each), so shares are round
    numbers. `star` goes first in every draft; `mid` goes 2nd, 3rd or 4th
    (picks 2,3,4,2,3,4,2,3,4,2 -- mean 2.9); `late` is taken in two drafts
    (20%); `once` in one (10%, still above MIN_SHARE's 2%). `twin_a` and
    `twin_b` share a display name to exercise slug collisions."""
    path = tmp_path / "corpus.duckdb"
    conn = dl.corpus_conn(str(path))
    for i in range(10):
        conn.execute(
            "INSERT INTO draft_log (draft_id, source, league_id, season,"
            " recorded_at, teams, rounds, my_slot, scoring_json, settings_json,"
            " human_seats) VALUES (?, 'mock', '1', 2026, ?::TIMESTAMP, 4, 2,"
            " NULL, NULL, NULL, 3)", [f"d{i}", f"2026-08-24 {i:02d}:00:00"])
        picks = [("star", 1), ("mid", 2 + (i % 3)), ("twin_a", 5), ("twin_b", 6)]
        if i < 2:
            picks.append(("late", 7))
        if i == 0:
            picks.append(("once", 8))
        for pid, pick_no in picks:
            conn.execute(
                "INSERT INTO draft_log_pick (draft_id, pick_no, round, slot,"
                " owner_key, is_anonymous, player_id, position, adp_rank,"
                " proj_points, autodrafted, had_owner, seconds_to_pick,"
                " clock_seconds) VALUES (?, ?, ?, ?, 'o', FALSE, ?, ?, 1.0,"
                " 100.0, FALSE, TRUE, 5.0, 30.0)",
                [f"d{i}", pick_no, (pick_no - 1) // 4 + 1, (pick_no - 1) % 4 + 1,
                 pid, "RB" if pid in ("star", "mid") else "WR"])
    conn.close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    yield path
    market._CACHE.clear()


@pytest.fixture
def board():
    """The universal tables the pages read names, teams and ESPN's ADP from."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE players (gsis_id VARCHAR, display_name VARCHAR,"
                 " headshot VARCHAR)")
    conn.executemany("INSERT INTO players VALUES (?, ?, ?)", [
        ("star", "D'Andre Swift", "https://img.test/swift.png"),
        ("mid", "Amon-Ra St. Brown", None),
        ("late", "Late Guy", None),
        ("once", "Once Guy", None),
        ("twin_a", "Josh Allen", None),
        ("twin_b", "Josh Allen", None),
    ])
    conn.execute("CREATE TABLE depth_charts (dt DATE, gsis_id VARCHAR, team VARCHAR)")
    conn.executemany("INSERT INTO depth_charts VALUES (?, ?, ?)", [
        ("2026-08-01", "star", "PHI"), ("2026-08-20", "star", "CHI"),
        ("2026-08-20", "mid", "DET"),
    ])
    conn.execute("CREATE TABLE sleeper_ids (gsis_id VARCHAR, espn_id BIGINT)")
    conn.execute("INSERT INTO sleeper_ids VALUES ('star', 4259545)")
    conn.execute("CREATE TABLE espn_adp (espn_id BIGINT, espn_name VARCHAR,"
                 " position VARCHAR, team VARCHAR, espn_adp DOUBLE)")
    conn.executemany("INSERT INTO espn_adp VALUES (?, ?, ?, ?, ?)", [
        (4259545, "D'Andre Swift", "RB", "CHI", 61.0),
        (999, "Amon-Ra St. Brown", "RB", "DET", 5.5),   # RB in the fixture corpus
    ])
    yield conn
    conn.close()


# -- slugs -------------------------------------------------------------------

def test_slugs_are_ascii_lowercase_and_stable():
    assert seo.slug("D'Andre Swift") == "dandre-swift"
    assert seo.slug("Amon-Ra St. Brown") == "amon-ra-st-brown"
    assert seo.slug("Seattle Seahawks D/ST") == "seattle-seahawks-dst"
    assert seo.slug("Ka'imi Fairbairn") == "kaimi-fairbairn"
    assert seo.slug("  ") == "player"


# -- the aggregation ---------------------------------------------------------

def test_build_adp_places_every_player_by_average_pick(corpus, board):
    data = seo.build_adp(board)
    assert data["drafts"] == 10 and data["teams"] == 4 and data["rounds"] == 2
    assert str(data["updated"]) == "2026-08-24"
    ranked = [p["player_id"] for p in data["players"]]
    assert ranked[:2] == ["star", "mid"]
    star = data["by_slug"]["dandre-swift"]
    assert star["adp"] == 1.0 and star["taken"] == 10 and star["share"] == 1.0
    assert star["round_mode"] == 1 and star["rank"] == 1 and star["pos_rank"] == 1
    assert star["hist"][0] == 10 and len(star["hist"]) == 8
    mid = data["by_slug"]["amon-ra-st-brown"]
    assert mid["adp"] == 2.9 and mid["p10"] == 2 and mid["p90"] == 4
    assert mid["low"] == 2 and mid["high"] == 4


def test_build_adp_resolves_name_team_and_espn_adp(corpus, board):
    star = seo.build_adp(board)["by_slug"]["dandre-swift"]
    assert star["name"] == "D'Andre Swift"
    assert star["headshot"] == "https://img.test/swift.png"
    assert star["team"] == "CHI"            # latest depth chart, not the older one
    assert star["espn_adp"] == 61.0         # through the sleeper crosswalk
    mid = seo.build_adp(board)["by_slug"]["amon-ra-st-brown"]
    assert mid["espn_adp"] == 5.5           # by name, no crosswalk row


def test_build_adp_gives_colliding_names_distinct_slugs(corpus, board):
    data = seo.build_adp(board)
    assert data["by_slug"]["josh-allen"]["player_id"] == "twin_a"
    assert data["by_slug"]["josh-allen-2"]["player_id"] == "twin_b"


def test_build_adp_drops_a_player_below_the_share_floor(corpus, board, monkeypatch):
    monkeypatch.setattr(seo, "MIN_SHARE", 0.15)
    data = seo.build_adp(board)
    ids = {p["player_id"] for p in data["players"]}
    assert "once" not in ids and "late" in ids     # 10% out, 20% in


def test_build_adp_with_no_drafts_is_empty_not_an_error(tmp_path, monkeypatch, board):
    path = tmp_path / "empty.duckdb"
    dl.corpus_conn(str(path)).close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    data = seo.build_adp(board)
    assert data["players"] == [] and data["drafts"] == 0 and data["updated"] is None


def test_build_adp_survives_a_missing_board(corpus):
    data = seo.build_adp(None)
    star = data["by_slug"]["star"]          # the id itself, as the archive does
    assert star["name"] == "star" and star["team"] is None and star["espn_adp"] is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_seo.py -q`
Expected: `ImportError: cannot import name 'seo' from 'api'`.

- [ ] **Step 4: Write the data layer**

```python
# api/seo.py
"""The pages a search engine reads.

The SPA is one document with a title, and every route in it is drawn by
JavaScript from live data. That is fine for a reader and nearly invisible
to a crawler: nothing on the site says, in text it can index, what a
player's draft position is or what this product does.

What the site has that nothing else does is the corpus -- hundreds of
real ESPN mock drafts, recorded pick by pick and growing while the farm
runs. These routes turn it into plain HTML: one page per player, per round
and per position, plus an index and a sitemap. No JavaScript on any of
them. A crawler gets the finished page, and so does a reader on a slow
connection.

WHAT IS PUBLIC AND WHAT IS NOT. These pages carry corpus AGGREGATES: where
a player goes, how often, in which round. The per-seat and per-turn
analysis on /archive stays behind an account, and nothing here reads a
session or a cookie.

REGISTERED BEFORE THE SPA. `api/static.py`'s fallback matches every path,
so `register_seo_routes` has to run before `register_spa` or these pages
would be served the index document instead.
"""
from __future__ import annotations

import re
import threading
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from fastapi import HTTPException

from api import market
from pipeline.db import read_table

SITE = "https://espnfantasydraft.com"
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

# A player taken in fewer drafts than this share has no page. "Taken in 1
# of 663 drafts" is not a draft position, it is one room's mistake, and a
# site of such pages is thin content that costs the pages worth having.
MIN_SHARE = 0.02

TEMPLATES = Path(__file__).resolve().parent / "templates"


def slug(name: str) -> str:
    """`D'Andre Swift` -> `dandre-swift`; `Seattle Seahawks D/ST` ->
    `seattle-seahawks-dst`. ASCII, lowercase, hyphens; apostrophes, dots and
    slashes vanish rather than becoming separators, so the abbreviation and
    the contraction read as one word the way they are said."""
    folded = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    folded = re.sub(r"['./]", "", folded.lower())
    folded = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    return folded or "player"


def _percentile(sorted_picks: list, q: float) -> int:
    return sorted_picks[int(round(q * (len(sorted_picks) - 1)))]


def _teams_by_player(conn) -> dict:
    """gsis_id -> team, from the newest depth chart row per player."""
    if conn is None:
        return {}
    try:
        dc = read_table(conn, "depth_charts")
    except Exception:      # noqa: BLE001 -- a board that cannot be read costs teams
        return {}
    if dc.empty or not {"gsis_id", "team", "dt"}.issubset(dc.columns):
        return {}
    latest = dc.dropna(subset=["gsis_id"]).sort_values("dt").drop_duplicates("gsis_id", keep="last")
    return {str(r.gsis_id): str(r.team) for r in latest.itertuples() if r.team}


def _espn_adp_by_player(conn, ids: dict) -> dict:
    """gsis_id -> ESPN's published ADP. Crosswalk first, then name and
    position, the two-step every ESPN join in this codebase makes.
    `ids` maps player_id -> (name, position)."""
    if conn is None:
        return {}
    try:
        espn = read_table(conn, "espn_adp")
        sleeper = read_table(conn, "sleeper_ids")
    except Exception:      # noqa: BLE001
        return {}
    if espn.empty or "espn_adp" not in espn.columns:
        return {}
    from scoring.profile import _norm_name
    out: dict = {}
    if not sleeper.empty and {"gsis_id", "espn_id"}.issubset(sleeper.columns):
        xwalk = sleeper.dropna(subset=["gsis_id", "espn_id"]).drop_duplicates("espn_id")
        joined = espn.merge(xwalk[["gsis_id", "espn_id"]], on="espn_id")
        for r in joined.itertuples():
            if r.espn_adp == r.espn_adp:
                out[str(r.gsis_id)] = float(r.espn_adp)
    by_name = {}
    if {"espn_name", "position"}.issubset(espn.columns):
        for r in espn.itertuples():
            if r.espn_adp == r.espn_adp:
                by_name.setdefault((_norm_name(str(r.espn_name)), str(r.position)), float(r.espn_adp))
    for pid, (name, pos) in ids.items():
        if pid not in out:
            hit = by_name.get((_norm_name(name), pos))
            if hit is not None:
                out[pid] = hit
    return out


def _team_fallback_from_espn(conn) -> dict:
    """gsis_id -> team from `espn_adp` through the crosswalk, for players the
    depth charts do not carry (a rookie in August)."""
    if conn is None:
        return {}
    try:
        espn = read_table(conn, "espn_adp")
        sleeper = read_table(conn, "sleeper_ids")
    except Exception:      # noqa: BLE001
        return {}
    if espn.empty or sleeper.empty or "team" not in espn.columns:
        return {}
    joined = espn.merge(sleeper.dropna(subset=["gsis_id", "espn_id"])[["gsis_id", "espn_id"]],
                        on="espn_id")
    return {str(r.gsis_id): str(r.team) for r in joined.itertuples() if r.team == r.team and r.team}


def _empty() -> dict:
    return {"players": [], "by_slug": {}, "drafts": 0, "teams": 0, "rounds": 0, "updated": None}


def build_adp(conn) -> dict:
    """Every player's draft position out of the corpus, in one pass.

    Every pick counts -- the autodrafter's and the farm's own seat's
    included -- because the question is where a player GOES, not what
    people think. The archive filters those out for its behavioural
    questions; ADP is the other kind of question.
    """
    corpus = market._corpus()
    try:
        try:
            teams, rounds = market._shape(corpus)
        except HTTPException:
            return _empty()
        total, latest = corpus.execute(
            "SELECT count(*), max(recorded_at) FROM draft_log"
            " WHERE teams = ? AND rounds = ?", [teams, rounds]).fetchone()
        rows = corpus.execute(
            "SELECT pk.player_id, pk.position, pk.pick_no"
            " FROM draft_log_pick pk JOIN draft_log d USING (draft_id)"
            " WHERE d.teams = ? AND d.rounds = ? AND pk.player_id IS NOT NULL"
            " AND pk.pick_no BETWEEN 1 AND ?",
            [teams, rounds, teams * rounds]).fetchall()
    finally:
        corpus.close()
    total = int(total or 0)
    if total == 0:
        return _empty()

    picks: dict = {}
    positions: dict = {}
    for pid, pos, pick_no in rows:
        picks.setdefault(str(pid), []).append(int(pick_no))
        positions.setdefault(str(pid), str(pos or "").upper())

    names = market._names(conn)
    missing = {pid for pid in picks if pid not in names}
    if missing:
        names.update(market._board_names(conn, missing))
    teams_by = _teams_by_player(conn)
    espn_team = _team_fallback_from_espn(conn)

    players = []
    for pid, taken in picks.items():
        share = len(taken) / total
        if share < MIN_SHARE:
            continue
        ordered = sorted(taken)
        entry = names.get(pid) or {}
        name = entry.get("name") or pid
        hist = [0] * (teams * rounds)
        for p in ordered:
            hist[p - 1] += 1
        players.append({
            "player_id": pid,
            "name": name,
            "headshot": entry.get("headshot"),
            "position": positions.get(pid) or "",
            "team": teams_by.get(pid) or espn_team.get(pid),
            "adp": round(mean(ordered), 1),
            "median": float(median(ordered)),
            "p10": _percentile(ordered, 0.10),
            "p90": _percentile(ordered, 0.90),
            "low": ordered[0],
            "high": ordered[-1],
            "taken": len(ordered),
            "share": round(share, 3),
            "round_mode": Counter((p - 1) // teams + 1 for p in ordered).most_common(1)[0][0],
            "hist": hist,
        })

    players.sort(key=lambda p: (p["adp"], -p["taken"], p["player_id"]))
    espn = _espn_adp_by_player(conn, {p["player_id"]: (p["name"], p["position"]) for p in players})
    pos_seen: Counter = Counter()
    for i, p in enumerate(players, start=1):
        p["rank"] = i
        pos_seen[p["position"]] += 1
        p["pos_rank"] = pos_seen[p["position"]]
        p["espn_adp"] = espn.get(p["player_id"])

    # Slugs: a collision takes -2, -3 in player_id order, so two players
    # with one name keep the same addresses from one snapshot to the next.
    by_base: dict = {}
    for p in sorted(players, key=lambda p: p["player_id"]):
        by_base.setdefault(slug(p["name"]), []).append(p)
    by_slug = {}
    for base, group in by_base.items():
        for n, p in enumerate(group, start=1):
            p["slug"] = base if n == 1 else f"{base}-{n}"
            by_slug[p["slug"]] = p

    updated = latest.date() if isinstance(latest, datetime) else None
    return {"players": players, "by_slug": by_slug, "drafts": total,
            "teams": teams, "rounds": rounds, "updated": updated}


def adp_data(conn) -> dict:
    """`build_adp`, once per CACHE_SECONDS, shared by every page."""
    return market._cached("seo-adp", lambda: build_adp(conn))
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_seo.py -q`
Expected: `7 passed`.

If `test_build_adp_places_every_player_by_average_pick` fails on `updated`, check what DuckDB returns for `max(recorded_at)` -- it is a `datetime.datetime`; the fixture's `+ INTERVAL (?) HOUR` keeps every draft on 2026-08-24.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt api/seo.py tests/test_seo.py
git commit -m "ADP out of the corpus, per player, with stable slugs"
```

---

### Task 5: The pages

**Files:**
- Create: `api/templates/base.html`, `api/templates/adp_index.html`, `api/templates/adp_player.html`, `api/templates/adp_round.html`, `api/templates/adp_missing.html`
- Modify: `api/seo.py` (append rendering and routes)
- Test: `tests/test_seo.py` (append)

**Interfaces:**
- Consumes: `adp_data(conn)`, `POSITIONS`, `SITE`, `TEMPLATES` from Task 4.
- Produces: `register_seo_routes(app, conn=None)` serving `GET /adp`, `GET /adp/round/{n}`, `GET /adp/{key}`; helper `render(name, **ctx) -> str`; `round_players(data, n) -> list`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_seo.py`:

```python
# -- the pages ---------------------------------------------------------------

def _client(board):
    app = FastAPI()
    seo.register_seo_routes(app, conn=board)
    return TestClient(app)


def test_the_index_lists_every_player_with_provenance(corpus, board):
    r = _client(board).get("/adp")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "<title>ESPN Mock Draft ADP 2026 (4-team PPR)" in body
    assert "10 real ESPN mock drafts" in body and "Aug 24, 2026" in body
    assert 'href="/adp/dandre-swift"' in body and "Amon-Ra St. Brown" in body
    assert '<link rel="canonical" href="https://espnfantasydraft.com/adp"' in body
    assert "<script" not in body.replace('<script type="application/ld+json">', "")


def test_a_player_page_states_his_figures(corpus, board):
    body = _client(board).get("/adp/amon-ra-st-brown").text
    assert "<title>Amon-Ra St. Brown ADP" in body
    assert "2.9" in body                       # the ADP
    assert "picks 2" in body and "4" in body   # the range
    assert "round 1" in body.lower()
    assert "5.5" in body                       # ESPN's own ADP beside it
    assert '"@type": "BreadcrumbList"' in body
    assert "<svg" in body                       # the distribution, drawn
    assert 'href="/adp/dandre-swift"' in body   # a neighbour by ADP


def test_a_player_page_is_the_same_for_a_slug_collision(corpus, board):
    c = _client(board)
    assert "twin_a" not in c.get("/adp/josh-allen").text
    assert c.get("/adp/josh-allen-2").status_code == 200


def test_round_and_position_pages_filter(corpus, board):
    c = _client(board)
    r1 = c.get("/adp/round/1").text
    assert "D'Andre Swift" in r1 and "Once Guy" not in r1
    assert 'href="/adp/round/2"' in r1
    rb = c.get("/adp/rb").text
    assert "D'Andre Swift" in rb and "Josh Allen" not in rb
    assert "<title>RB ADP" in rb


def test_unknown_pages_are_404_and_noindex(corpus, board):
    c = _client(board)
    for path in ("/adp/nobody-here", "/adp/round/17", "/adp/round/0", "/adp/ol"):
        r = c.get(path)
        assert r.status_code == 404, path
        assert '<meta name="robots" content="noindex">' in r.text


def test_an_empty_corpus_still_serves_the_index(tmp_path, monkeypatch, board):
    path = tmp_path / "empty.duckdb"
    dl.corpus_conn(str(path)).close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    c = _client(board)
    r = c.get("/adp")
    assert r.status_code == 200 and "No drafts recorded yet" in r.text
    assert c.get("/adp/dandre-swift").status_code == 404
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_seo.py -q -k "index or page or round or unknown or empty_corpus"`
Expected: failures with `AttributeError: module 'api.seo' has no attribute 'register_seo_routes'`.

- [ ] **Step 3: Write the base template**

```html
{# api/templates/base.html #}
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<meta name="description" content="{{ description }}">
{% if noindex %}<meta name="robots" content="noindex">{% else %}<link rel="canonical" href="{{ site }}{{ path }}">{% endif %}
<meta name="theme-color" content="#0a0b0e">
<meta property="og:type" content="website">
<meta property="og:site_name" content="ESPN Draft Assist">
<meta property="og:title" content="{{ title }}">
<meta property="og:description" content="{{ description }}">
<meta property="og:url" content="{{ site }}{{ path }}">
<meta property="og:image" content="{{ headshot or (site ~ '/demo-poster.jpg') }}">
<meta name="twitter:card" content="summary">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
{% if breadcrumbs %}<script type="application/ld+json">{{ breadcrumbs | tojson }}</script>{% endif %}
<style>
:root{--bg-0:#0a0b0e;--bg-1:#121319;--bg-2:#1b1d25;--border:#262932;--text-1:#e8eaef;--text-2:#98a0b0;--text-3:#565b69;--accent:#ffb454;--ok:#0ca30c;--fail:#d03b3b;--radius:3px;--mono:'JetBrains Mono',ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg-0);color:var(--text-2);font:15px/1.5 -apple-system,BlinkMacSystemFont,'Inter','Segoe UI',Helvetica,Arial,sans-serif}
a{color:var(--text-1);text-decoration:none}a:hover{text-decoration:underline}
.bar{display:flex;align-items:center;gap:16px;padding:10px 20px;background:var(--bg-1);border-bottom:1px solid var(--border)}
.bar .mark{font-family:var(--mono);font-weight:600;letter-spacing:.06em;color:var(--text-1)}
.bar .mark span{color:var(--accent)}
.bar nav{display:flex;gap:14px;font-size:13px;color:var(--text-2)}
.bar .cta{margin-left:auto;background:var(--accent);color:#1a1200;font-weight:600;font-size:13px;padding:7px 12px;border-radius:var(--radius)}
main{max-width:960px;margin:0 auto;padding:28px 20px 48px}
h1{margin:0 0 6px;font-size:26px;line-height:1.2;color:var(--text-1);letter-spacing:-.01em}
h2{margin:32px 0 10px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-3)}
.prov{margin:0 0 22px;font-family:var(--mono);font-size:12px;color:var(--text-3)}
.crumbs{font-family:var(--mono);font-size:12px;color:var(--text-3);margin-bottom:14px}
.crumbs a{color:var(--text-2)}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;font-family:var(--mono);font-weight:500;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-3);padding:6px 8px;border-bottom:1px solid var(--border)}
td{padding:7px 8px;border-bottom:1px solid var(--border);vertical-align:middle}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
.pos{display:inline-block;min-width:34px;text-align:center;font-family:var(--mono);font-size:11px;padding:1px 5px;border-radius:var(--radius);background:var(--bg-2);color:var(--text-2)}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:18px 0}
.fig{background:var(--bg-1);border:1px solid var(--border);border-radius:var(--radius);padding:12px 14px}
.fig .k{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-3)}
.fig .v{font-family:var(--mono);font-size:24px;color:var(--text-1);margin-top:2px}
.fig .v small{font-size:13px;color:var(--text-2)}
.head{display:flex;align-items:center;gap:16px}
.head img{width:72px;height:72px;border-radius:50%;object-fit:cover;background:var(--bg-2)}
.chart{background:var(--bg-1);border:1px solid var(--border);border-radius:var(--radius);padding:12px}
.chart svg{display:block;width:100%;height:auto}
.near{display:flex;flex-wrap:wrap;gap:8px}
.near a{background:var(--bg-1);border:1px solid var(--border);border-radius:var(--radius);padding:6px 10px;font-size:13px}
.near a b{font-family:var(--mono);font-weight:500;color:var(--accent);margin-right:6px}
.links{display:flex;flex-wrap:wrap;gap:8px 14px;font-size:13px;margin-top:8px}
.links a{color:var(--text-2)}
.ask{margin-top:40px;padding:20px;background:var(--bg-1);border:1px solid var(--border);border-radius:var(--radius)}
.ask p{margin:0 0 12px;color:var(--text-2)}
footer{max-width:960px;margin:0 auto;padding:0 20px 32px;font-size:12px;color:var(--text-3)}
footer a{color:var(--text-2)}
</style>
</head>
<body>
<header class="bar">
  <a class="mark" href="/">ESPN <span>DRAFT ASSIST</span></a>
  <nav><a href="/adp">ADP</a><a href="/mocks">Mock drafts</a></nav>
  <a class="cta" href="/">Try it on your draft</a>
</header>
<main>
{% block body %}{% endblock %}
<section class="ask">
  <p>ESPN Draft Assist is a live board that sits beside your ESPN draft room and ranks every pick from drafts like these. Mock drafts are free; a real draft is $9.99, once.</p>
  <a class="cta" href="/">Try it on your draft</a>
</section>
</main>
<footer>Not affiliated with ESPN. Figures are from ESPN mock drafts this site recorded; see <a href="/mocks">every draft</a>.</footer>
</body>
</html>
```

- [ ] **Step 4: Write the index / position template**

```html
{# api/templates/adp_index.html #}
{% extends "base.html" %}
{% block body %}
{% if position %}<div class="crumbs"><a href="/adp">ADP</a> › {{ position }}</div>{% endif %}
<h1>{{ heading }}</h1>
<p class="prov">{{ provenance }}</p>
{% if not players %}
<p>No drafts recorded yet. The farm records ESPN mock drafts every day; check back tomorrow.</p>
{% else %}
<table>
<thead><tr><th class="n">#</th><th>Player</th><th>Pos</th><th>Team</th><th class="n">ADP</th><th class="n">Range</th><th class="n">Drafted</th><th class="n">{{ 'Pos rank' }}</th></tr></thead>
<tbody>
{% for p in players %}
<tr>
  <td class="n">{{ p.rank }}</td>
  <td><a href="/adp/{{ p.slug }}">{{ p.name }}</a></td>
  <td><span class="pos">{{ p.position }}</span></td>
  <td>{{ p.team or '—' }}</td>
  <td class="n">{{ '%.1f' % p.adp }}</td>
  <td class="n">{{ p.p10 }}–{{ p.p90 }}</td>
  <td class="n">{{ (p.share * 100) | round | int }}%</td>
  <td class="n">{{ p.position }}{{ p.pos_rank }}</td>
</tr>
{% endfor %}
</tbody>
</table>
{% endif %}
<h2>By round</h2>
<div class="links">{% for n in range(1, rounds + 1) %}<a href="/adp/round/{{ n }}">Round {{ n }}</a>{% endfor %}</div>
<h2>By position</h2>
<div class="links">{% for pos in positions %}<a href="/adp/{{ pos | lower }}">{{ pos }}</a>{% endfor %}</div>
{% endblock %}
```

- [ ] **Step 5: Write the player template**

```html
{# api/templates/adp_player.html #}
{% extends "base.html" %}
{% block body %}
<div class="crumbs"><a href="/adp">ADP</a> › <a href="/adp/{{ p.position | lower }}">{{ p.position }}</a> › {{ p.name }}</div>
<div class="head">
  {% if p.headshot %}<img src="{{ p.headshot }}" alt="" width="72" height="72">{% endif %}
  <div>
    <h1>{{ p.name }} ADP in ESPN mock drafts</h1>
    <p class="prov"><span class="pos">{{ p.position }}</span> {{ p.team or '' }} · {{ provenance }}</p>
  </div>
</div>
<div class="figs">
  <div class="fig"><div class="k">ADP</div><div class="v">{{ '%.1f' % p.adp }}</div></div>
  <div class="fig"><div class="k">Usual range</div><div class="v">picks {{ p.p10 }}<small>–</small>{{ p.p90 }}</div></div>
  <div class="fig"><div class="k">Most often</div><div class="v">round {{ p.round_mode }}</div></div>
  <div class="fig"><div class="k">Drafted in</div><div class="v">{{ (p.share * 100) | round | int }}%<small> of drafts</small></div></div>
  <div class="fig"><div class="k">Position rank</div><div class="v">{{ p.position }}{{ p.pos_rank }}</div></div>
  <div class="fig"><div class="k">ESPN's own ADP</div><div class="v">{% if p.espn_adp is not none %}{{ '%.1f' % p.espn_adp }}{% else %}—{% endif %}</div></div>
</div>
<p>In {{ drafts }} recorded ESPN mock drafts, {{ p.name }} was taken {{ p.taken }} times, on average at pick {{ '%.1f' % p.adp }} (overall rank {{ p.rank }}, {{ p.position }}{{ p.pos_rank }}). Eight drafts in ten took him between picks {{ p.p10 }} and {{ p.p90 }}; the earliest was pick {{ p.low }} and the latest pick {{ p.high }}.{% if p.espn_adp is not none %} ESPN's published ADP has him at {{ '%.1f' % p.espn_adp }}.{% endif %}</p>
<h2>Where he goes, pick by pick</h2>
<div class="chart">
<svg viewBox="0 0 {{ picks_total * 6 }} 120" role="img" aria-label="How many drafts took {{ p.name }} at each overall pick">
{% for count in p.hist %}{% if count %}<rect x="{{ loop.index0 * 6 }}" y="{{ 100 - (100 * count / peak) | round }}" width="5" height="{{ (100 * count / peak) | round }}" fill="#ffb454"><title>Pick {{ loop.index }}: {{ count }}</title></rect>{% endif %}{% endfor %}
{% for n in range(1, rounds + 1) %}<text x="{{ (n - 1) * teams * 6 + 2 }}" y="116" font-size="10" fill="#565b69" font-family="ui-monospace,monospace">R{{ n }}</text>{% endfor %}
</svg>
</div>
{% if near %}
<h2>Goes around</h2>
<div class="near">{% for q in near %}<a href="/adp/{{ q.slug }}"><b>{{ '%.1f' % q.adp }}</b>{{ q.name }} <span class="pos">{{ q.position }}</span></a>{% endfor %}</div>
{% endif %}
{% endblock %}
```

- [ ] **Step 6: Write the round template and the 404 page**

```html
{# api/templates/adp_round.html #}
{% extends "base.html" %}
{% block body %}
<div class="crumbs"><a href="/adp">ADP</a> › Round {{ n }}</div>
<h1>Round {{ n }} of an ESPN mock draft</h1>
<p class="prov">Picks {{ first }}–{{ last }} · {{ provenance }}</p>
{% if not players %}<p>No player has gone in round {{ n }} often enough to say.</p>{% else %}
<table>
<thead><tr><th>Player</th><th>Pos</th><th class="n">ADP</th><th class="n">Range</th><th>In this round</th></tr></thead>
<tbody>
{% for p in players %}
<tr>
  <td><a href="/adp/{{ p.slug }}">{{ p.name }}</a></td>
  <td><span class="pos">{{ p.position }}</span></td>
  <td class="n">{{ '%.1f' % p.adp }}</td>
  <td class="n">{{ p.p10 }}–{{ p.p90 }}</td>
  <td>{% if p.round_mode == n %}usually{% else %}sometimes{% endif %}</td>
</tr>
{% endfor %}
</tbody>
</table>
{% endif %}
<div class="links">{% if n > 1 %}<a href="/adp/round/{{ n - 1 }}">← Round {{ n - 1 }}</a>{% endif %}{% if n < rounds %}<a href="/adp/round/{{ n + 1 }}">Round {{ n + 1 }} →</a>{% endif %}</div>
{% endblock %}
```

```html
{# api/templates/adp_missing.html #}
{% extends "base.html" %}
{% block body %}
<div class="crumbs"><a href="/adp">ADP</a></div>
<h1>Not drafted in any recorded mock</h1>
<p>There is no page at this address. Every player who goes in ESPN mock drafts often enough to have a draft position is on <a href="/adp">the ADP index</a>.</p>
{% endblock %}
```

- [ ] **Step 7: Append the rendering and routes to `api/seo.py`**

```python
# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_env = None
_env_lock = threading.Lock()


def _templates():
    global _env
    with _env_lock:
        if _env is None:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
            _env = Environment(loader=FileSystemLoader(str(TEMPLATES)),
                               autoescape=select_autoescape(["html"]))
        return _env


def render(name: str, **ctx) -> str:
    ctx.setdefault("site", SITE)
    ctx.setdefault("positions", POSITIONS)
    return _templates().get_template(name).render(**ctx)


def _provenance(data: dict) -> str:
    if not data["drafts"]:
        return "No drafts recorded yet."
    when = data["updated"].strftime("%b %-d, %Y") if data["updated"] else "today"
    return (f"From {data['drafts']} real ESPN mock drafts "
            f"({data['teams']}-team PPR, {data['rounds']} rounds), updated {when}.")


def _crumbs(*items) -> dict:
    """`BreadcrumbList` structured data from (name, path) pairs."""
    return {"@context": "https://schema.org", "@type": "BreadcrumbList",
            "itemListElement": [
                {"@type": "ListItem", "position": i, "name": name, "item": SITE + path}
                for i, (name, path) in enumerate(items, start=1)]}


def round_players(data: dict, n: int) -> list:
    """Who goes in round `n`: everyone whose usual range crosses it, most
    often first, then by ADP."""
    teams = data["teams"]
    first, last = (n - 1) * teams + 1, n * teams
    hits = [p for p in data["players"]
            if p["round_mode"] == n or (p["p10"] <= last and p["p90"] >= first)]
    return sorted(hits, key=lambda p: (p["round_mode"] != n, p["adp"]))


def _missing(path: str):
    from fastapi.responses import HTMLResponse
    return HTMLResponse(render("adp_missing.html", title="Not found – ESPN Draft Assist",
                               description="No such page.", path=path, noindex=True),
                        status_code=404)


def register_seo_routes(app, conn=None):
    """The crawlable site: `/adp`, its player, round and position pages, and
    `/sitemap.xml`. Register before `api/static.register_spa`."""
    from fastapi.responses import HTMLResponse, Response

    def data():
        return adp_data(conn)

    def index_page(position: str | None):
        d = data()
        players = d["players"] if position is None else [
            p for p in d["players"] if p["position"] == position]
        season = d["updated"].year if d["updated"] else datetime.now().year
        shape = f"{d['teams']}-team PPR" if d["drafts"] else "PPR"
        if position is None:
            heading = f"ESPN Mock Draft ADP {season} ({shape})"
            path = "/adp"
            desc = (f"Average draft position of every player in {d['drafts']} real ESPN "
                    f"mock drafts ({shape}): ADP, typical range, and how often each is taken.")
            crumbs = _crumbs(("ADP", "/adp"))
        else:
            heading = f"{position} ADP – ESPN mock drafts {season}"
            path = f"/adp/{position.lower()}"
            desc = (f"Where every {position} goes in {d['drafts']} real ESPN mock drafts "
                    f"({shape}): ADP, range, and position rank.")
            crumbs = _crumbs(("ADP", "/adp"), (position, path))
        return HTMLResponse(render(
            "adp_index.html", title=f"{heading} – ESPN Draft Assist", description=desc,
            path=path, heading=heading, provenance=_provenance(d), players=players,
            rounds=d["rounds"], position=position, breadcrumbs=crumbs))

    @app.get("/adp", response_class=HTMLResponse)
    def adp_index():
        return index_page(None)

    @app.get("/adp/round/{n}", response_class=HTMLResponse)
    def adp_round(n: int):
        d = data()
        if not d["drafts"] or n < 1 or n > d["rounds"]:
            return _missing(f"/adp/round/{n}")
        teams = d["teams"]
        first, last = (n - 1) * teams + 1, n * teams
        players = round_players(d, n)
        usual = [p["name"] for p in players if p["round_mode"] == n][:4]
        desc = (f"Who goes in round {n} (picks {first}–{last}) of an ESPN mock draft, "
                f"from {d['drafts']} recorded drafts"
                + (f": {', '.join(usual)}." if usual else "."))
        return HTMLResponse(render(
            "adp_round.html", title=f"Round {n} of an ESPN mock draft – who goes there – ESPN Draft Assist",
            description=desc, path=f"/adp/round/{n}", n=n, first=first, last=last,
            players=players, rounds=d["rounds"], provenance=_provenance(d),
            breadcrumbs=_crumbs(("ADP", "/adp"), (f"Round {n}", f"/adp/round/{n}"))))

    @app.get("/adp/{key}", response_class=HTMLResponse)
    def adp_player(key: str):
        if key.upper() in POSITIONS:
            return index_page(key.upper())
        d = data()
        p = d["by_slug"].get(key)
        if p is None:
            return _missing(f"/adp/{key}")
        i = p["rank"] - 1
        near = [q for q in d["players"][max(0, i - 4): i + 5] if q is not p]
        pct = int(round(p["share"] * 100))
        desc = (f"{p['name']} ADP {p['adp']:.1f} in {d['drafts']} real ESPN mock drafts: "
                f"usually picks {p['p10']}–{p['p90']}, round {p['round_mode']}, "
                f"taken in {pct}% of drafts, {p['position']}{p['pos_rank']}.")
        return HTMLResponse(render(
            "adp_player.html", title=f"{p['name']} ADP – ESPN mock drafts {d['updated'].year if d['updated'] else ''} – ESPN Draft Assist",
            description=desc, path=f"/adp/{p['slug']}", p=p, drafts=d["drafts"],
            teams=d["teams"], rounds=d["rounds"], picks_total=d["teams"] * d["rounds"],
            peak=max(p["hist"]) or 1, near=near, headshot=p["headshot"],
            provenance=_provenance(d),
            breadcrumbs=_crumbs(("ADP", "/adp"), (p["position"], f"/adp/{p['position'].lower()}"),
                                (p["name"], f"/adp/{p['slug']}"))))
```

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_seo.py -q`
Expected: `13 passed`.

If `test_a_player_page_states_his_figures` fails on `"picks 2"`: the figure reads `picks {{ p.p10 }}<small>–</small>{{ p.p90 }}` -- the test looks for `picks 2`, which is present. If it fails on `round 1`: `round {{ p.round_mode }}` renders `round 1`; `.lower()` in the test covers the sentence case. If Jinja raises `TemplateNotFound`, `TEMPLATES` resolved to the wrong directory -- it must be `api/templates`.

- [ ] **Step 9: Commit**

```bash
git add api/seo.py api/templates tests/test_seo.py
git commit -m "ADP pages: an index, a page per player, per round and per position"
```

---

### Task 6: The sitemap, and registration before the SPA

**Files:**
- Modify: `api/seo.py` (append sitemap route inside `register_seo_routes`)
- Modify: `api/main.py` (lines ~837-850: register before `register_spa`)
- Test: `tests/test_seo.py` (append)

**Interfaces:**
- Consumes: `register_seo_routes`, `adp_data` from Tasks 4-5; `api.static.register_spa(app, dist)`.
- Produces: `GET /sitemap.xml` (`application/xml`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_seo.py`:

```python
# -- the sitemap and the registration order ----------------------------------

def test_the_sitemap_lists_every_page_once(corpus, board):
    r = _client(board).get("/sitemap.xml")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/xml")
    root = ET.fromstring(r.text)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [u.find("s:loc", ns).text for u in root.findall("s:url", ns)]
    assert len(locs) == len(set(locs))
    assert "https://espnfantasydraft.com/" in locs
    assert "https://espnfantasydraft.com/adp" in locs
    assert "https://espnfantasydraft.com/adp/dandre-swift" in locs
    assert "https://espnfantasydraft.com/adp/josh-allen-2" in locs
    assert "https://espnfantasydraft.com/adp/round/2" in locs
    assert "https://espnfantasydraft.com/adp/round/3" not in locs   # two rounds
    assert "https://espnfantasydraft.com/adp/rb" in locs
    mods = {u.find("s:lastmod", ns).text for u in root.findall("s:url", ns)}
    assert mods == {"2026-08-24"}


def test_an_empty_corpus_sitemap_has_only_the_static_pages(tmp_path, monkeypatch, board):
    path = tmp_path / "empty.duckdb"
    dl.corpus_conn(str(path)).close()
    monkeypatch.setattr(market.dl, "CORPUS_PATH", str(path))
    market._CACHE.clear()
    root = ET.fromstring(_client(board).get("/sitemap.xml").text)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [u.find("s:loc", ns).text for u in root.findall("s:url", ns)]
    assert locs == ["https://espnfantasydraft.com/", "https://espnfantasydraft.com/mocks",
                    "https://espnfantasydraft.com/adp"]


def test_the_pages_are_reachable_with_the_spa_mounted(corpus, board, tmp_path):
    """`register_spa`'s fallback matches every path. These routes must be
    registered first, and the SPA must still answer everything else."""
    from api.static import register_spa
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    app = FastAPI()
    seo.register_seo_routes(app, conn=board)
    assert register_spa(app, dist)
    c = TestClient(app)
    assert "D'Andre Swift" in c.get("/adp").text
    assert c.get("/sitemap.xml").headers["content-type"].startswith("application/xml")
    assert c.get("/mocks").text == "<html>spa</html>"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_seo.py -q -k sitemap`
Expected: 2 failed -- `/sitemap.xml` is 404.

- [ ] **Step 3: Add the sitemap route**

Inside `register_seo_routes` in `api/seo.py`, after `adp_player`:

```python
    @app.get("/sitemap.xml")
    def sitemap():
        d = data()
        stamp = d["updated"].isoformat() if d["updated"] else None
        urls = ["/", "/mocks", "/adp"]
        if d["drafts"]:
            urls += [f"/adp/{pos.lower()}" for pos in POSITIONS]
            urls += [f"/adp/round/{n}" for n in range(1, d["rounds"] + 1)]
            urls += [f"/adp/{p['slug']}" for p in d["players"]]
        body = ['<?xml version="1.0" encoding="UTF-8"?>',
                '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
        for path in urls:
            body.append("<url><loc>%s%s</loc>%s</url>" % (
                SITE, path, f"<lastmod>{stamp}</lastmod>" if stamp else ""))
        body.append("</urlset>")
        return Response(content="\n".join(body), media_type="application/xml")
```

- [ ] **Step 4: Register in `api/main.py`**

Between `register_market_routes(app, conn)` and the `register_spa` block:

```python
    # The crawlable site: ADP pages rendered from the corpus as plain HTML,
    # and the sitemap. Before the SPA, whose fallback would otherwise answer
    # every one of these paths with index.html -- see api/seo.py.
    from api.seo import register_seo_routes
    register_seo_routes(app, conn)
```

- [ ] **Step 5: Run the whole SEO suite and the app's own smoke tests**

Run: `.venv/bin/python -m pytest tests/test_seo.py tests/test_seo_static.py tests/test_market_api.py -q`
Expected: all pass (`16 passed` in `test_seo.py`).

Run: `.venv/bin/python -c "from api.main import create_app; app = create_app(); print([r.path for r in app.routes if r.path.startswith('/adp') or r.path == '/sitemap.xml'])"`
Expected: `['/adp', '/adp/round/{n}', '/adp/{key}', '/sitemap.xml']`.

- [ ] **Step 6: Commit**

```bash
git add api/seo.py api/main.py tests/test_seo.py
git commit -m "A sitemap, and the ADP pages registered ahead of the SPA"
```

---

### Task 7: See it, then ship it

**Files:** none new. Verification and deploy.

- [ ] **Step 1: Start the API against the real data and look at the pages**

Run (from the repo root; the scratchpad path is this session's):

```bash
S=/private/tmp/claude-501/-Users-worthy-Test-fantasy-football/93996a2e-bfac-469f-a3ab-9fe8f7af368b/scratchpad
(cd web && npm run build 2>&1 | tail -1)
(nohup .venv/bin/uvicorn api.main:app --port 8000 > $S/api.log 2>&1 & echo $! > $S/api.pid); sleep 6
curl -s -o /dev/null -w "adp %{http_code} %{content_type}\n" http://localhost:8000/adp
curl -s http://localhost:8000/sitemap.xml | grep -c "<loc>"
curl -s http://localhost:8000/adp | grep -o "<title>[^<]*" | head -1
curl -s http://localhost:8000/robots.txt | head -3
```

Expected: `adp 200 text/html; charset=utf-8`; a `<loc>` count in the hundreds (3 static + 6 positions + 16 rounds + qualifying players); the title with `8-team PPR`; robots served as text.

- [ ] **Step 2: Screenshot the index, one player, and the landing explainer**

```bash
cat > $S/seo_shot.py <<'EOF'
import time
from playwright.sync_api import sync_playwright
S="/private/tmp/claude-501/-Users-worthy-Test-fantasy-football/93996a2e-bfac-469f-a3ab-9fe8f7af368b/scratchpad"
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width":1200,"height":900}, device_scale_factor=2)
    pg.goto("http://localhost:8000/adp"); pg.screenshot(path=f"{S}/adp_index.png")
    pg.goto("http://localhost:8000/adp/dandre-swift"); pg.screenshot(path=f"{S}/adp_player.png", full_page=True)
    pg.goto("http://localhost:8000/?signedout=1", wait_until="networkidle"); time.sleep(3)
    print("title:", pg.title())
    print("description:", pg.locator('meta[name="description"]').get_attribute("content")[:60])
    pg.locator(".xp").scroll_into_view_if_needed(); time.sleep(0.5)
    pg.locator(".xp").screenshot(path=f"{S}/explainer.png")
    print("h1:", pg.locator("h1.xp-h1").text_content()[:50])
    print("faq json-ld:", pg.locator('script[type="application/ld+json"]').count())
    b.close()
EOF
.venv/bin/python $S/seo_shot.py
kill $(cat $S/api.pid)
```

Open the three PNGs with the Read tool. Check: the index table is aligned and readable; the player page's chart has bars under round labels; the explainer's FAQ opens; the title printed is the landing title from Task 2 and the JSON-LD count is 2 (SoftwareApplication + FAQPage).

- [ ] **Step 3: Run the wider suite once**

Run: `.venv/bin/python -m pytest tests/test_seo.py tests/test_seo_static.py tests/test_market_api.py tests/test_drafts_api.py tests/test_billing.py -q`
Expected: all pass. (`tests/test_live_api.py` carries 10 pre-existing failures about `wk1_points` NaN serialisation unrelated to this work.)

- [ ] **Step 4: Push**

```bash
git push origin main
```

Railway builds from `main` (~20 minutes). After the deploy: `curl -s https://espnfantasydraft.com/sitemap.xml | head -5` and `curl -s -o /dev/null -w "%{http_code}\n" https://espnfantasydraft.com/adp/dandre-swift`.

- [ ] **Step 5: Hand-off to the owner**

Tell the owner: verify the domain in Google Search Console (DNS TXT record or the HTML meta tag -- paste the token and it goes into `web/index.html` next to the other meta), then submit `https://espnfantasydraft.com/sitemap.xml`. Indexing of ~300 new pages on a domain with no history takes days to weeks; the sitemap's `lastmod` moves with every farm run, which is what keeps Google coming back.
