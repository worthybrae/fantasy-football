# Home experience: built for a normal fantasy player

Date: 2026-08-29. Owner: "the whole home experience is confusing; design it
for a normal fantasy football player who wants a better drafting experience."
Approved direction below.

## The person

Drafts once a year on ESPN, has done a couple of mocks, thinks in rankings,
sleepers, "my guys", "don't get sniped". Wants, in order: (1) on draft night,
who to take right now and why, in three seconds; (2) before draft night,
whether their guys will be there and a round-by-round game plan; (3) proof it
works before trusting it; (4) nothing to learn.

## Vocabulary (user-facing, everywhere)

| ours | theirs |
|---|---|
| Lasts % / still there | **Will he be there?** (% at your pick) |
| Edge pts | **Worth grabbing now** (+N pts over waiting) |
| plan / targets | **Your draft plan** / **Take** |
| favourites | **My guys** |
| corpus / recorded drafts | "real ESPN drafts" (count in a tooltip) |
| founder | "free for the first 100" |

Numbers appear only next to a plain sentence; the model's internals live in
tooltips and the Cheat sheet.

## Signed-out landing (`/`)

1. Hero: "**Your ESPN draft, with a plan.**" One line under it: "Who to take
   at every pick, and whether your guys will still be there — from N real
   ESPN drafts." Two buttons: **Try a mock draft — free** (opens the mock
   lobby flow) and **Connect your ESPN league** (bookmarklet walkthrough).
2. **Try it now**: league size (8/10/12) and seat selects → the round-by-round
   plan for that seat renders instantly from `/api/plan/preview`, first four
   rounds, each "Round r · pick p → Name (POS) · one reason · if he's gone:
   A, B". No login.
3. **Live right now**: the demo room (keep), captioned "A real ESPN mock draft
   in progress — this is what the draft room looks like."
4. Three short "what you get" lines with one number each (will-he-be-there,
   the plan, my guys). FAQ (three questions). Founder line.

## Signed-in home (Dashboard)

1. **Next draft hero**: league name, date/time and countdown, "seat 6 of 8 ·
   from ESPN" (or the seat picker), one primary button **Open the draft
   room** (join flow unchanged). No draft connected: the hero is **Connect
   your ESPN league** with the walkthrough.
2. **Ready for draft night** checklist, three rows with ticks: Connect ESPN ·
   Pick my guys (5-25) · Do a one-minute dry run (opens a mock room or the
   replay). Each row is a button when not done. The section collapses to one
   line "You're ready" when all three tick.
3. **Your draft plan** as rounds: for each of the seat's first 8 turns "Round
   r · pick p → **Name** POS · reason sentence · if he's gone: A, B" with a
   "why" disclosure showing the plan's pros/cons. Header: "For C6 Fantasy
   Football · seat 6 of 8". Data: `/api/plan/preview` (extend `turns` from 3
   to 8).
4. **My guys**: rows name · pos · **When to take him** (the plan's round for
   him, or "later than pick N") · **Will he be there?** at your next pick (%
   chip). Edit button opens the existing picker modal. Empty state: one
   sentence + **Pick my guys**.
5. Other leagues / mock rooms in a compact list; the mock lobby below.
Everything else (consensus, edge tables, corpus counts) is removed from the
home page.

## Draft room default view

- Top: **Take now** card — one name, one reason sentence, two backups
  ("if he's gone: …"), the Draft button. Off the clock: "Up next: pick 11 ·
  likely there: A, B, C".
- List: name · pos · **Will he be there?** · **Worth grabbing now** ·
  Draft. Search and position chips stay.
- **Cheat sheet** toggle (persisted) reveals today's 14-column table.
- Plan rail unchanged; roster unchanged.

## Components (web/src)

`NextDraftHero`, `ReadinessChecklist`, `DraftPlanRounds`, `MyGuysTable`,
`TryItNow`, `RoomSimpleList` + `TakeNowCard`, `CheatSheetToggle`. Existing
data hooks reused; `/api/plan/preview` gains `turns=8` and `reason` (one
sentence) per target; `/api/account/favorites/outlook` gains `plan_round`.

## Out of scope

The ADP pages, the market/data pages, billing.
