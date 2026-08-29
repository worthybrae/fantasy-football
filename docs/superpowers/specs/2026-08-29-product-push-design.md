# Product push: a plan people trust, more shapes of draft, founders, a clear front door

Date: 2026-08-29. Owner's brief: the paywall is off for now; the first 100
users draft free for good; the mock corpus should cover 10- and 12-team
leagues and standard scoring, not only 8-team PPR; the draft plan still
reaches (Josh Allen in round 2); the home page and "Your guys" are confusing
about what the product does. Decisions delegated to the implementer.

## 1. Plan quality — `scoring/plan.py`

The previous rule (ESPN priority, a 12-rank window decided by raw-point
edge) let a QB's raw-point cliff win inside the window. The missing idea is
**can you wait?** A pick is a good pick when the player is ranked well AND is
unlikely to be there at your next turn.

For each eligible candidate at turn T with next turn T′:

- `gone_next = 1 − P(still there at T′ | now)` from the availability table.
- `reach = max(0, espn_adp − pick_T)` (picks earlier than ESPN's ADP; 0 when
  ADP is blank).
- `priority = espn_rank + need_penalty − favourite_bonus + waitable + reach_penalty`
  - `waitable = 15` when `gone_next < 0.5` (you can probably wait), else 0;
    favourites use `gone_next < 0.35`.
  - `reach_penalty = 0.5 × max(0, reach − 6)` (a six-pick reach is free; a
    ten-pick reach costs two ranks; a thirty-pick reach costs twelve).
  - `need_penalty` and `favourite_bonus` unchanged (starter 0, flex +4,
    bench +20, deferred +200, capped ineligible; favourite −8).
- Target = lowest priority. Alternates = next two. **No cross-position
  point comparison decides the target.** `edge_pts` is still computed and
  shown, and among candidates whose priorities tie within 2 ranks the larger
  edge wins.
- Reasons gain: "likely gone before your next pick (72 %)" as a pro when
  `gone_next ≥ 0.5`; "you could probably wait — 81 % still there at pick 27"
  as a con when waitable; "reach: ADP 21 at pick 11" as a con when
  `reach > 6`.

Test cases (8-team PPR, seat 6, turns 6/11/22/27): Allen (QB 26, ADP 21,
favourite, lasts 90 % to 11) is not the pick-11 target while an RB/WR ranked
≤ 20 that is < 50 % to last is available; at pick 22 with Allen 40 % to last
to 27 he is a legitimate target; Gibbs at pick 6 never; a favourite ranked
15 who will be gone beats a stranger ranked 10 who will last.

## 2. Corpus shapes — farm + availability + ADP pages

- `pipeline/mock_farm.py` joins rooms in a rotation over shapes
  `{8, 10, 12} teams × {ppr, standard}`, choosing the shape with the fewest
  recorded drafts that has a joinable room; `FARM_SHAPES` env (default
  `8:ppr,10:ppr,12:ppr,10:std,12:std`) and per-shape counts logged. The
  corpus already stores `teams` and `scoring_json` per draft.
- `scoring/availability.py`: `AvailabilityTable` keeps per-shape counts
  (`taken_by` per `(teams, format)` plus the pooled total). `availability_at`
  gains `teams=None, fmt=None`; when the shape has ≥ `MIN_SHAPE_DRAFTS = 60`
  drafts the shape-conditioned counts are used, otherwise pooled. The live
  room passes its league's `teams` and format; the favourites outlook passes
  the selected league size.
- `api/seo.py`: the index and player pages gain a shape switch (tabs:
  8-team PPR, 10-team PPR, 12-team PPR, …) for shapes with ≥ 60 drafts;
  `/adp` stays the most-recorded shape; `/adp/10-team-ppr` etc. are separate
  indexable pages (sitemap entries) once a shape qualifies.
- Until the farm has recorded other shapes, nothing changes on screen.

## 3. Founders — `api/billing.py`, `api/account.py`, Dashboard

- Paywall off: `require_paid` returns free while `FOUNDERS_OPEN = True`
  (env `FOUNDERS_LIMIT`, default 100).
- Table `founder(account_id PK, ordinal INTEGER, granted_at)`. On any
  authenticated request (`_account_ids` non-empty) the newest account id is
  inserted if the table holds fewer than `FOUNDERS_LIMIT` rows; ordinal is
  the count at insert. Reads span key versions like `favorites`.
- `require_paid` grants free when the account is a founder, forever — even
  after billing turns on.
- `GET /api/account/me` → `{founder: bool, ordinal: int|null, founders_left:
  int}`; the Dashboard shows "Founding member #37 — every draft free" or
  "N founder spots left — connect ESPN to claim one" on the landing.

## 4. The front door — landing + Dashboard

Landing (signed out), top to bottom:
1. One sentence: "Draft with what real ESPN drafts do: who lasts to your
   pick, and a plan for every round." Sub-line: N recorded drafts, updated
   daily. CTA "Connect ESPN — free for the first 100".
2. Three steps with one visual each: connect ESPN (bookmarklet or join a
   mock) → pick your guys → open the room on draft night (the live room,
   your plan, your guys' odds).
3. The live proof (the demo room, already there), then "what you get"
   cards: Lasts %, the Plan, Your guys by pick, ADP pages.
4. FAQ.

Dashboard (signed in), top to bottom: next draft countdown + join; **Your
guys** (names, Edit) and **Your guys by pick** side by side under one heading
"Your guys"; **Your plan** for the next draft's shape and seat: a pre-draft
strategy line from the corpus for that seat ("Seat 6 of 10 usually opens
RB-WR-WR-RB; QBs go in round 6") plus the first three turns' targets from
`build_plan` with no picks made; upcoming drafts; mock lobby.

No new dependencies; the frontend-design skill guides the visual pass.

## Out of scope

Stripe changes, a marketing site, the live room's layout beyond copy, ADP
shape pages before a second shape reaches 60 drafts.
