# The player profile popup

**Status:** approved design, not yet built.
**Canvas:** https://claude.ai/code/artifact/988e9406-682b-4dbc-9b41-a9cd94d7818f
(artboards: `Main` the deliverable, `Sparse` the thin-payload state, and two
earlier sketches on a second page.)

## What is wrong with the profile today

The profile is already a popup. `PlayerOverlay.tsx` renders `PlayerProfile`
over the draft room, and the `/players/<slug>` route was deleted on purpose:
navigating to it unmounted the room, the board, the tab, the scroll position
and the 2.5s poll, then stalled on a ~3.5s fetch with the pick clock running.
(The README still describes the deleted route. That is a separate fix.)

What it is not is a popup-shaped thing. It is a full page's worth of cards —
fifteen components, its own charts for every one of them — rendered inside an
overlay. Under a pick clock a reader has to scroll it, and each card teaches
its own way of being read: `SeasonFinish` draws seasons left to right,
`ConsistencyTable` draws a coefficient on a track, `WeekByWeek` draws weeks,
and none of them agree on what a colour means or which direction is good.

Meanwhile the board's own hover panels solved that. `CellTip.tsx` has one
`Chart`: a fixed 68px slot, one column per period, a value above and a label
below, taller is better, and one five-step ramp whose cut points every panel
shares. Five panels are drawn with it and a reader learns the picture once.

The profile should be drawn in that language, and it should fit.

## What we are building

One popup, every card on screen, nothing behind a tab. Top to bottom:

1. **Header** — name, position chip, team, age, NFL year, bye; and four
   figures on the right: your board rank, consensus ADP, tier, VOR.
2. **Status line** — injury designation, depth-chart slot, and the edge over
   consensus. `Questionable` is the one field that changes a pick, so it sits
   one lookup from the name.
3. **Four season panels** — Health, Finish, Steady, Per game. The shared
   `Chart`, one column per season, with Per game carrying the hollow
   projected column behind its dashed rule.
4. **2025 by week, with its game log underneath** — the week chart and the
   log share one card, separated by a hairline, because the log is that
   chart's detail: same weeks, with what actually happened in each. Columns
   are week, opponent, the real stat line, and PPR points coloured on the
   same ramp as the bar above it.
5. **Schedule · Room · Market** — the softness strip, the depth chart, and
   the ranking sources.
6. **Usage · Blocking · Near you** — per-game carries and targets, the
   offensive line's rank out of 32, and the board's neighbours by rank.
7. **Comparable seasons · News** — other players whose season looked like
   this one with what their next year did, beside the headlines.
8. **Draft** — a real action. See below.

## Decisions

**Every card stays.** The first pass at this design dropped Usage, Blocking,
Value neighbours and the missing-data notice for want of room. They are all
in. Compact here means denser cards, not fewer of them.

**You can draft from the popup.** This reverses the rule the overlay
currently holds — `onToggleDrafted` is deliberately not passed to
`PlayerProfile`, and the on-the-clock banner merged in `fee5551` says "Picks
are made on the board, not here." Both change: the popup gets a real draft
action routed through the same confirm flow the board uses, and the banner
drops its second sentence to become a plain "You are on the clock."

**Colour keeps meaning one thing.** Every panel, the schedule strip and the
game log's points column all take the same five-step ramp with the same cut
points. A colour that meant one thing on the board and another here would
undo the reason for doing this at all.

**The sparse state is designed, not discovered.** A rookie has no seasons, so
Health, Finish, Steady and the week chart have nothing to draw. They render
as baseline marks with a notice saying which parts are empty and which are
real — a popup that silently drops half its cards reads as broken.

## Architecture

**`Chart` has to come out of `CellTip.tsx`.** Today `Chart` and its `Col`
type are private to that file, which is why the profile has its own charts at
all. They move to their own module and `CellTip` imports them like everyone
else. Nothing about the drawing changes; this is the extraction that makes
one language possible.

**The panels are computed once, not twice.** `barTone`, `finishTone` /
`finishPosition` and the steadiness percentile already exist in `weeks.ts`,
`finish.ts` and `CellTip.tsx`. The profile's panels call the same functions,
so a season that is green in the hover panel cannot be olive in the popup.

**Every field is already in the payload.** No endpoint changes. `cv_rank` /
`cv_rank_n`, `pos_finish`, `games`, `ppg`, `proj_ppg`, `stat_line`, `stats`,
`similar.players`, `cohort`, `schedule[].pct`, `oline`, `depth_chart`,
`status` and `news` all ship today; several are the fields the README lists
as "carried but not rendered".

**Component fates.** `WeekByWeek` is reworked to the shared chart and grows
the log. `SeasonFinish` and `ConsistencyTable` are replaced by panels.
`VerdictStrip` and `RoomGap` fold into the header and status line.
`ScheduleRanks`, `DepthChartCard`, `MarketRow`, `NewsPanel`, `InjuryStatus`,
`UsageLine`, `LineQuality`, `ValueNeighbors`, `MissingData`, `SimilarPlayers`
and `CohortNext` keep their jobs in denser frames.

## Testing

The existing Python suite is untouched by this — it is front-end only. The
checks that matter are `npm run build` run bare (a pipe returns the pipe's
exit code and hides a failing `tsc -b`), the popup rendered against a real
profile payload at both states, and the shared `Chart` still drawing the five
hover panels exactly as it does now.

## Out of scope

The README's stale `/players/<slug>` description, and the `cohort` payload
that returns 19 rows while `CohortNext` renders nothing.
