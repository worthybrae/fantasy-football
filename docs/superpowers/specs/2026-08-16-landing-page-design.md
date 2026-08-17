# Landing page, and cutting the legacy research pages

Date: 2026-08-16

## Why

The site's front door is an install card. It hands over a bookmarklet and says
nothing about what the tool does, what it knows, or whether it will actually
work when the draft starts. Everything good in this project is one click past a
page that gives you no reason to click.

At the same time `/legacy` carries a whole second application — the research
board, the sim rail, the draft grid, the manager forecast — that the live draft
path never touches. It is a maintenance surface with no user.

This spec covers two changes that belong together: the landing page becomes a
real product page backed by real data, and the legacy boards are deleted.

Authentication and paid tiers are explicitly **out of scope**. They only mean
something once this is hosted, and today the whole stack is local. They get
their own spec when that decision is made.

## What the landing page becomes

One page, four bands, top to bottom:

1. **Hero.** What the tool is, in a sentence a stranger understands, with the
   draggable `⚓ Draft Helper` bookmarklet as the primary call to action. The
   drag target keeps its current mechanics exactly — the `javascript:` href is
   injected as raw HTML so it is present at first paint and a drag copies the
   bookmarklet rather than the page URL.

2. **Board preview.** Twelve real rows from the user's own DuckDB: rank, player,
   position, team, tier, VOR, market rank, edge. Not a screenshot. This is the
   product demonstrating itself, and because the data is local it is always the
   viewer's own board.

3. **How it works.** The three steps that are on the page today (show the
   bookmarks bar, open the draft, click the button), demoted below the proof
   rather than standing in for it.

4. **Readiness strip.** Whether this machine is ready for draft night: when each
   data source last refreshed, whether draft history is imported, whether
   manager models are fitted, whether a sim has ever run, what league is
   configured. Draft night is the one night the tool has to work; this is where
   you find out it will not.

The token gate that already exists is unchanged in behaviour. Arriving with a
token in the URL hash still connects and redirects to `/draft`; a connect that
is already running still offers the board instead of the install guide.

## Data

Two endpoints, split by cost. `build_board` takes ~3.5s per call (no cache), and
the readiness strip must not wait behind it.

### `GET /api/landing/status`

Raw table reads only — no board build. Answers "is this machine ready".

```json
{
  "sources":  [{"source": "adp", "ok": true, "rows": 300, "refreshed_at": "..."}],
  "league":   {"season": 2026, "teams": 8, "rounds": 15, "derived": true},
  "history":  {"picks": 712, "seasons": [2023, 2024, 2025], "teams": 8},
  "managers": {"fitted": 8, "personal": 3},
  "sim":      {"run_id": "...", "my_slot": 4, "created_at": "..."}
}
```

`sim` is `null` when no sim has run. `history.picks` is 0 with an empty
`seasons` list when no draft history is imported. Every field is derived from a
single table read, so the whole response is milliseconds.

### `GET /api/landing/preview?limit=12`

Runs `build_board` and returns a slim slice — the eight fields the preview
renders, not the twenty-seven `/api/players` carries. ~2KB instead of 185KB.

```json
{"pool": 249, "players": [{"rank": 1, "name": "...", "position": "RB",
                           "team": "DET", "tier": 1, "vor": 22.7,
                           "market_rank": 1, "edge": 0.0}]}
```

`limit` is clamped to 1..50. `pool` is the full board size, which the preview
band uses to say how many players were scored — it is free here because the
board was built anyway, and it is the one readiness fact that cannot be had
cheaply.

### Client behaviour

The status request fires and paints immediately. The preview request runs in
parallel behind a skeleton of the right height, so the page does not reflow when
it lands. Either failing degrades to that band being absent — a landing page
that cannot reach the API still shows the hero and the bookmarklet, because
installing the bookmarklet is exactly what a user with no running helper needs
to do next.

## What gets deleted

Routes `/legacy` and `/legacy/draft-board`, and everything only they used:

- `App.tsx`'s `Board` (the research board, its keyboard cursor, its filters)
- `PlayerTable`, `PositionTabs`, `DraftRail`, `TopBar`, `FreshnessBadge`
- `DraftBoardPage`, `DraftGrid`, `ManagerForecast`
- `BoardSkeleton`, `boardColumnsFor`, and the `api.ts` helpers left with no
  caller (`fetchManagers`, `fetchManagerHistory`, `fetchSimBoard`, `fetchMeta`,
  the sim endpoints)
- the CSS rules for all of the above

Player profiles survive and move to `/players/:slug`. They are the deepest
content in the project, and the live board's `PlayerCard` already links to them
— cutting them would break the live path's one link out. `PlayerProfile` and its
nine leaf components are untouched apart from the route change.

`App.tsx` ends as a routes table.

Backend endpoints stay as they are. `/api/sim`, `/api/managers`, and the rest
still serve the pipeline and the live board; only the client code that read them
for the deleted pages goes.

## Testing

- `landing_status` on a seeded database: reports sources, league, history,
  manager counts, and the latest sim.
- `landing_status` on an empty database: `sim` is null, `history.picks` is 0,
  `managers.fitted` is 0 — the readiness strip's "not ready" states are the ones
  most likely to be wrong, and they are what a new user sees first.
- `landing_preview` returns exactly `limit` rows in rank order, carries `pool`
  as the full board size, and clamps a `limit` outside 1..50.
- `landing_status` does not build the board — asserted by monkeypatching
  `build_board` to raise.
- Frontend: `tsc -b` and `oxlint` clean after the deletions, which is what
  catches a component or helper that turned out to still have a caller.

## Not doing

- Caching `build_board`. The 3.5s cost hits the live board too and deserves its
  own change; the split above means the landing page does not need it.
- Auth, accounts, payment. Separate spec, once hosting is decided.
- Rewriting the README's research-tool sections beyond removing what no longer
  exists.
