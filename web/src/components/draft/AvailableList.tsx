import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { LiveCandidate, Player } from '../../api'
import { CellTip, loadProfile, type CellTipKind } from './CellTip'
import { FINISH_STARTERS, finishHeight, finishTone } from './finish'
import { BAR_CEILING, barThresholds, barTone } from './weeks'
import { riskTone } from './tone'

// duplicated from RosterPanel.tsx/DraftBoardGrid.tsx (unexported in both):
// a four-line pure function isn't worth a shared module between four views
// now.
function posBadge(position: string): ReactNode {
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

// Same shape as profile/payload.ts's fmtRank -- one decimal only when the
// aggregate ADP isn't a whole number, dash when the player has no market
// coverage at all. Used for both market ranks in this table: `market_rank`
// (the five-source median, scoring/market.py) and `espn_ppr_rank` (a single
// source, integral in practice, but formatted the same way so the two
// columns line up digit for digit).
function fmtRank(n: number | null): string {
  if (n === null) return '—'
  return Number.isInteger(n) ? String(n) : n.toFixed(1)
}

const POSITIONS = ['ALL', 'QB', 'RB', 'WR', 'TE', 'K', 'DST']

// -- last season's per-game bars -------------------------------------------
//
// One bar per week of the last complete season, in week order, coloured on
// thresholds that depend on POSITION. `Player.game_points`
// (scoring/game_points.py) is already priced in the league's own scoring, so
// these are the same points every other number in this row is in.
//
// THE THRESHOLDS ARE NOW PER-POSITION, AND THAT IS DELIBERATE -- the owner
// asked for it directly. This block used to argue the opposite ("do not make
// these relative"), and the numbers that old argument cited are exactly why
// it flipped. Measured on the real 252-player board (data/nfl.duckdb, 2025,
// the owner's stored PPR scoring, kickers excluded because this league
// prices none of their stats): on the OLD fixed 10/15 split, quarterbacks
// came out 24/19/57 around a median of 16.6 -- 57% of every quarterback's
// games rendered green, which is not a signal at that position, it is the
// column's default state. Tight ends came out 63/20/16 around a median of
// 7.5, the opposite failure: a tight end who clears 15 is rare and worth
// noticing, and on a scale built for quarterbacks he almost never could. A
// tight end's 15-point week and a quarterback's were never the same
// afternoon; per-position cut points are how the column now says so, where
// refusing to have any used to just paint the QB half of the table green.
//
// RB/WR/TE keep the original 10/15 split, and the original reasoning for it
// (a good tight end week is rarer and worth more than the same number from a
// running back) still holds unchanged for them. QB is 10/20, set by the
// owner: a quarterback's measured median week (16.6, cited above) lands in
// amber rather than green, so green means a genuinely good start instead of
// an ordinary one, and the red floor stays where a truly bad quarterback
// week is. K and DST move to
// 5/10 -- both score on a much smaller scale than every other position (a
// good defensive or kicking week is worth a fraction of a good receiver's),
// so the split has to be smaller too; 5/10 keeps the same 1:2 ratio, sized
// down rather than re-derived from scratch.
// How long a drafted player stays on screen on his way out. Long enough to
// read a name under a pick clock, short enough that back-to-back picks do
// not stack up on each other.
// How long a drafted row stays before it is removed. MUST match the fade in
// App.css: a row that reaches full transparency before it is unmounted is
// still occupying a full row of space, and that window IS the gap -- the
// longer the fade, the longer the table sits with a hole in it. At 200ms the
// invisible-but-present window is a couple of frames rather than a beat.
const TAKEN_MS = 200
// The rows below closing up. Deliberately shorter than the fade: by the time
// it starts, the interesting thing has already happened, and a slow slide
// just reads as the table being sluggish.
const SLIDE_MS = 170
// More than this vanishing at once is a resync -- a restored session, a
// reconnect mid-draft -- not picks. Animating that would be a screenful of
// motion describing something that did not just happen.
const TAKEN_BURST_LIMIT = 5

// The bars share ONE vertical scale across every row, so a bad player's best
// week cannot draw as tall as a stud's. 30 points is the ceiling (anything
// above it draws full height): on the same 252-player board it clips 3.0% of
// games, while the 95th percentile is 27.3 and the median 9.2 -- so almost
// every bar lands inside the scale and the ones that clip are already the
// unmistakable ones. A per-row maximum was rejected for exactly the reason
// the fixed colour thresholds were.
// Ceiling in pixels. The row is 32px tall (`th, td { height: 32px }`) with
// 6px of vertical padding, so 18px is the tallest chart that CANNOT make the
// row taller -- and a taller row means fewer players on screen under a
// thirty-second clock, which is the one thing this column is not allowed to
// cost. Measured after the fact: the row is 32px with the column and 32px
// without it.
const BAR_MAX_PX = 18
// A game that was played always draws something. Without a floor, a 0.0 game
// (a receiver held catchless, a kicker who missed everything) would be
// indistinguishable from a week he did not play -- which is a different claim
// and gets its own, greyer, 1px mark.
const BAR_MIN_PX = 2

function barHeight(points: number): number {
  const scaled = Math.round((Math.min(points, BAR_CEILING) / BAR_CEILING) * BAR_MAX_PX)
  // Math.max also catches the negative weeks -- a quarterback can finish
  // under zero, and 70 of 2025's real games did. They are floored to the
  // minimum bar rather than drawn downward: this is a 18px strip in a table
  // row, not a chart with an axis, and half of it cannot be spent on the 0.4%
  // of games that go below the line.
  return Math.max(BAR_MIN_PX, scaled)
}

// Memoized on the array identity: `players` is fetched once and never
// mutated, so a row's bars are built exactly once no matter how often the
// table re-filters and re-sorts. Without this, every keystroke in the search
// box would rebuild up to 250 x 18 = 4,500 spans.
const GameBars = memo(function GameBars({ points, season, position }: {
  points: (number | null)[]
  season: number | null
  position: string
}): ReactNode {
  const { amberFrom, greenFrom } = barThresholds(position)
  const played = points.filter((p): p is number => p !== null)
  const bad = played.filter((p) => p < amberFrom).length
  const mid = played.filter((p) => p >= amberFrom && p < greenFrom).length
  const good = played.filter((p) => p >= greenFrom).length
  return (
    <span
      className="gamebars"
      role="img"
      /* A bar chart is nothing to a screen reader, so it gets the summary a
         sighted reader takes from the colours instead of a shape it cannot
         see. Thresholds are per-position (barThresholds above), so the
         summary names the actual cut points for THIS row rather than a
         number that might not be the one its colours used. */
      aria-label={`${season ?? 'Last season'}: ${played.length} games, `
        + `${good} at ${greenFrom} or more, ${mid} from ${amberFrom} to ${greenFrom}, `
        + `${bad} under ${amberFrom}`}
    >
      {points.map((p, i) => (p === null ? (
        <span key={i} className="gamebar is-none" />
      ) : (
        <span
          key={i}
          className={`gamebar ${barTone(p, position)}`}
          style={{ height: `${barHeight(p)}px` }}
        />
      )))}
    </span>
  )
})

// -- sorting ---------------------------------------------------------------
//
// WHY THIS TABLE NO LONGER SHOWS `gain_now`, `vor_points` OR `fills`:
// the model is unchanged. `gain_now` still ranks this list server-side (see
// scoring/gain.py and api/live.py's _recompute), it is still what `c.rank`
// -- the `#` column and the default sort here -- counts down, and it still
// picks and orders the three recommendation cards above this table, which
// explain the pick in a sentence. What changed is that the table stopped
// showing the working: three columns nobody could read without a paragraph
// of explanation, so they were deliberately deleted at the owner's request
// ("im not sure what they even are"). They were NOT lost in a refactor --
// do not "restore" them. If a number here ever needs defending again, the
// place for it is TopThree's sentence, not a fourth column of jargon.
//
// The same three figures DO still appear on the surfaces this table opens
// -- the recommendation cards, the confirm dialog, the profile overlay's
// seed row -- and the same complaint applied to their captions there, so
// they were renamed rather than abbreviated further. One label per
// quantity, everywhere:
//   `gain_now`    -> "Gain vs waiting"  (what taking him now is worth
//                    against the best survivor at his position at the
//                    measured horizon -- signed, and honestly negative when
//                    waiting is the better play, which "Cost to wait" would
//                    have inverted)
//   `vor_points`  -> "Over replacement" (the app already says "value over
//                    replacement" in prose on the landing page, the board
//                    preview and TopThree's own fallback hint; this is just
//                    that phrase unabbreviated)
//   `fills`       -> "Roster slot"      (not bare "Slot": this room already
//                    uses "slot" for the user's own DRAFT slot)
// Renaming any of them in one place only is the exact inconsistency the
// rename existed to remove.
// (Removing `fills` also removed the accent treatment that marked
// an open starter slot; that signal lives on in RosterPanel, which is where
// a reader looks for "what do I still need" anyway. It is deliberately not
// re-drawn here.)

type SortKey = 'rank' | 'pos' | 'player' | 'finish' | 'health' | 'steady' | 'change' | 'proj' | 'lasts' | 'adp' | 'espn'
type SortDir = 'asc' | 'desc'

// The direction a column gets on its FIRST click -- "best first" for that
// particular column, which is not the same arrow everywhere: rank/ADP/ESPN
// are ranks (1 is best, so ascending), proj/lasts are quantities (bigger is
// better, so descending). HEALTH follows proj/lasts's logic, not
// rank/ADP's -- five bars is the best outcome a row can have, so its
// first click is ascending too, same as a rank where 1 is best. Clicking an
// already-sorted header flips it, so both directions stay reachable on every
// column; this only decides which one you land on without having to click
// twice.
const NATURAL_DIR: Record<SortKey, SortDir> = {
  rank: 'asc', pos: 'asc', player: 'asc', finish: 'asc', health: 'desc', steady: 'desc', change: 'desc', proj: 'desc', lasts: 'desc', adp: 'asc',
  espn: 'asc',
}

// Position sorts in the pill row's order (QB, RB, WR, TE, K, DST), not
// alphabetically -- alphabetical would open with DST and K, which is the
// order nobody thinks about a draft in, and it would disagree with the
// filter pills sitting directly above the header. Anything the server sends
// that isn't in the pill list lands after everything that is.
const POS_ORDER = POSITIONS.slice(1)
function posIndex(position: string): number {
  const i = POS_ORDER.indexOf(position)
  return i === -1 ? POS_ORDER.length : i
}

// -- header tooltips: shared type + positioning -----------------------------
//
// Every sortable header (SortKey) plus the two non-sortable ones that still
// carry an explanation -- the games sparkline's header and the draft
// button's blank one -- share ONE floating tooltip panel rather than each
// header owning its own. Only one can ever be open (a pointer or a focus
// ring is only ever in one place), so this is simpler than ten independent
// popovers and is what a native `title` already behaved like from the
// reader's side.
type TipId = SortKey | 'games' | 'draft'

const TIP_DELAY_MS = 130
const TIP_MARGIN = 8

// Mirrors DraftBoardGrid.tsx's own `popoverStyle` -- `position: fixed` off
// the trigger's own rect, clamped inside the viewport so the panel never
// runs off-screen -- with one change: `size` is the tooltip's own MEASURED
// `getBoundingClientRect` (see the `useLayoutEffect` in AvailableList
// itself) rather than a guessed constant. That file's popover content is
// fairly uniform card text and a rough estimate is "close enough to decide
// which side has room" by its own admission; this file's tooltip copy
// ranges from one short sentence (posTitle) to a genuinely long paragraph
// (gamesTitle), where a single guessed height would either clip the long
// ones or leave the short ones floating with acres of empty space.
function positionTip(
  rect: DOMRect, size: { width: number; height: number },
): { left: number; top: number } {
  const left = Math.max(
    TIP_MARGIN,
    Math.min(rect.left, window.innerWidth - size.width - TIP_MARGIN),
  )
  const below = rect.bottom + TIP_MARGIN
  const top = below + size.height <= window.innerHeight - TIP_MARGIN
    ? below
    : Math.max(TIP_MARGIN, rect.top - size.height - TIP_MARGIN)
  return { left, top }
}

// Career availability, drawn as a five-bar meter.
//
// The number behind it is `career_games_pg` from the board: average games
// played per season across a player's WHOLE career, counting the seasons he
// missed entirely rather than skipping them. That distinction is the reason
// the column is worth having -- a player who sits out a year has no rows at
// all in the weekly data, so the obvious average silently drops his worst
// seasons and reports the fragile as durable.
//
// Cut points measured on the real board (184 of 252 players carry a value;
// the rest are defenses and players with no NFL season yet): 9% land on one
// bar, 15% on two, 35% on three, 28% on four, 14% on five. Wide middle,
// rare extremes -- which is what makes a five-bar meter readable at all.
// -- expected change, as a five-bar meter ------------------------------------
//
// A SIGNED number in a column that has to sit beside two fill meters. The
// first version made it a bar growing up or down from a centre axis, which is
// the honest shape for signed data and the wrong one here: the axis spanned
// the whole cell while the fill was a short stub, so on screen the column was
// a flat line and nothing else. Direction lives in the COLOUR instead --
// green for a projected gain, red for a loss -- and magnitude in the number
// of bars, so Health, Steady and Change all read the same way.
//
// One colour per level rather than a gradient: the meter answers "how many
// bars" first and the colour reinforces it, so five discrete steps read
// faster than a continuous ramp that makes four and five nearly identical.
//
// Shared by all three meters -- Health, Steady and Change -- which is what
// makes a glance across them mean one thing: in every column one bar is the
// worst outcome and five the best, on the same red-to-green ramp.
const METER_CLASS = ['', 'is-1', 'is-2', 'is-3', 'is-4', 'is-5'] as const

// Bands in points per game, chosen to spread the real board rather than to be
// round numbers: they put 43/55/53/37/19 of the 207 measurable players in the
// five levels. Outliers are pinned rather than allowed to set the scale --
// the extremes are quarterbacks who lost a starting job, at -15, and letting
// one of those define "a big fall" would flatten every real difference into
// the middle band.
const CHANGE_CUTS = [-2, -0.5, 0.5, 2] as const

// MORE BARS IS BETTER, which is the rule Health and Steady beside it already
// follow. An earlier version made the count the SIZE of the change and the
// colour its direction, so the biggest faller on the board wore a full
// five-bar meter -- and against two neighbours where full means good, that
// reads as a top rating that happens to be red.
function changeLevel(change: number): number {
  let level = 1
  for (const cut of CHANGE_CUTS) if (change >= cut) level += 1
  return level
}

const ChangeMeter = memo(function ChangeMeter({ change }: { change: number }): ReactNode {
  const level = changeLevel(change)
  const label = `${change > 0 ? '+' : ''}${change.toFixed(1)} points per game vs last season`
  return (
    <span className={`change-meter ${METER_CLASS[level]}`} role="img"
          title={label} aria-label={`${label} (${level} of 5)`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className={`change-mbar${i <= level ? ' is-on' : ''}`} />
      ))}
    </span>
  )
})

// -- positional finish, season by season -----------------------------------
//
// The same five bands SeasonFinish uses on the profile, on the same cut
// points, so the small version in the table and the big one in the card
// cannot disagree about whether a season was good. Fractions of the number
// of players a 12-team league starts at the position, which is why one rule
// reads a quarterback and a running back correctly.
// Careers run from one season to ten, so the arc is RIGHT-ALIGNED: the most
// recent season sits in the same place on every row. Left-aligned, a rookie's
// only bar and a ten-year veteran's latest one landed nine slots apart, and
// the bar a reader actually compares between two players was never in the
// same position twice.
const FINISH_MAX = 10

const FinishArc = memo(function FinishArc(
  { arc, position }: { arc: [number, number][]; position: string },
): ReactNode {
  const starters = FINISH_STARTERS[position] ?? 24
  const shown = arc.slice(-FINISH_MAX)
  // No `title`: the hover panel for this cell (CellTip) lists every season as
  // a row, and the native box rendered ON TOP of it -- two answers to one
  // hover, the worse one covering the better. `aria-label` stays, because a
  // screen reader gets nothing from the panel.
  const label = shown.map(([yr, f]) => `'${String(yr).slice(2)} ${position}${f}`).join('  ')
  return (
    <span className="finish-arc" role="img" aria-label={label}>
      {shown.map(([season, finish]) => (
        <span key={season} className="finish-arc-slot">
          <span className={`finish-arc-bar ${finishTone(finish, starters)}`}
                style={{ height: `${finishHeight(finish, starters)}%` }} />
        </span>
      ))}
    </span>
  )
})

const HEALTH_CUTS = [10, 13, 15, 16.3] as const

function healthLevel(gamesPg: number | null | undefined): number | null {
  if (gamesPg === null || gamesPg === undefined || Number.isNaN(gamesPg)) return null
  let level = 1
  for (const cut of HEALTH_CUTS) if (gamesPg >= cut) level += 1
  return level
}


const HealthMeter = memo(function HealthMeter(
  { level, gamesPg }: { level: number; gamesPg: number },
): ReactNode {
  // The bars are the glance; this is the number behind them. Without it the
  // meter is unfalsifiable -- three bars means nothing a reader can check,
  // and "durability 3 of 5" told a screen reader even less than the picture
  // told everyone else.
  const label = `${gamesPg.toFixed(1)} games per season across his career`
  // No `title`, for the same reason FinishArc has none: this cell opens a
  // panel and the native box would land on top of it.
  return (
    <span className={`health-meter ${METER_CLASS[level]}`} role="img"
          aria-label={`${label} (${level} of 5)`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className={`health-bar${i <= level ? ' is-on' : ''}`} />
      ))}
    </span>
  )
})

// Already a percentile, so the quintiles are the meter: an even fifth of the
// board's players at each position lands on each bar by construction, which
// is the whole reason the percentile is taken against the BOARD and not the
// weekly universe (see scoring/board.py's `consistency`).
function steadyLevel(pct: number | null | undefined): number | null {
  if (pct === null || pct === undefined || Number.isNaN(pct)) return null
  return Math.min(5, Math.max(1, Math.ceil(pct * 5)))
}

const SteadyMeter = memo(function SteadyMeter(
  { level, cv }: { level: number; cv: number | null },
): ReactNode {
  // Same falsifiability as HealthMeter: the bars are the glance, the
  // coefficient is the number a reader can check them against.
  const label = cv === null
    ? `Steadier than ${(level - 1) * 20}-${level * 20}% of his position`
    : `Week-to-week swing of ${cv.toFixed(2)} (sigma over mean) -- steadier `
      + `than ${(level - 1) * 20}-${level * 20}% of his position on this board`
  return (
    <span className={`steady-meter ${METER_CLASS[level]}`} role="img"
          title={label} aria-label={`${label} (${level} of 5)`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className={`steady-bar${i <= level ? ' is-on' : ''}`} />
      ))}
    </span>
  )
})

// The one value a column sorts on. `null` means "this player has no such
// number" and is handled by the comparator, never coerced to 0 -- a player
// no market source covers is not ADP 0, i.e. the best pick on the board.
// Every key reads from exactly the same place the cell renders from, so
// what you see sorted is what you see printed.
function sortValue(
  key: SortKey, c: LiveCandidate, player: Player | undefined,
): number | string | null {
  switch (key) {
    case 'rank': return c.rank
    case 'pos': return posIndex(c.position)
    case 'player': return (player?.name ?? c.player_id).toLowerCase()
    // The most recent finish, because an arc has no single value and the
    // last season is the one being reasoned from. Ascending is natural:
    // RB1 is the best thing a row can say.
    case 'finish': {
      const arc = player?.season_finishes
      return arc && arc.length ? arc[arc.length - 1][1] : null
    }
    case 'health': return player?.career_games_pg ?? null
    case 'steady': return player?.consistency_pct ?? null
    case 'change': return player?.proj_change ?? null
    case 'proj': return c.proj_points
    case 'lasts': return c.survive_pct
    case 'adp': return player?.market_rank ?? null
    case 'espn': return player?.espn_ppr_rank ?? null
  }
}

// NULLS ALWAYS LAST, in both directions -- deliberately not "smallest" or
// "largest". A missing ESPN rank is an absence, not a value: treating it as
// -Infinity would put every uncovered rookie above Ja'Marr Chase on one
// click and below him on the next, and either way the dashes would be
// interleaved through the rows you were actually trying to compare.
// Sinking them keeps the comparable rows contiguous at the top and makes
// the flip button do one predictable thing.
//
// Ties (and null-vs-null) fall back to the server's own `rank`, ascending,
// which is never null and is unique per row -- so the sort is total and
// stable-looking regardless of the engine, and sorting by POS, say, leaves
// each position group in board order rather than in arrival order. The
// tie-break is NOT flipped with `dir`: within one position, or one ADP
// value, board order is the right order either way.
function compareRows(
  a: LiveCandidate, b: LiveCandidate,
  key: SortKey, dir: SortDir, players: Record<string, Player>,
): number {
  const av = sortValue(key, a, players[a.player_id])
  const bv = sortValue(key, b, players[b.player_id])
  if (av === null || bv === null) {
    if (av !== null) return -1
    if (bv !== null) return 1
    return a.rank - b.rank
  }
  const d = typeof av === 'string' && typeof bv === 'string'
    ? av.localeCompare(bv)
    : (av as number) - (bv as number)
  if (d !== 0) return dir === 'asc' ? d : -d
  return a.rank - b.rank
}

interface AvailableListProps {
  candidates: LiveCandidate[]
  players: Record<string, Player>
  onDraft: (c: LiveCandidate) => void
  // Not in the task brief's own signature for this component -- added
  // because "the draft button is disabled unless it is your turn" (the
  // brief's own rule) has nowhere else to come from: this component has no
  // access to `state`, only to the ranked list and the join table. DraftRoom
  // computes it once (on_the_clock === my_slot, the same test ClockPanel's
  // youAreUp already uses) and hands it down, so both this list and
  // TopThree gate their buttons on the exact same boolean rather than each
  // re-deriving "is it my turn" from state fields they don't have.
  isMyTurn: boolean
  // The pick the server measured this list against ("pick 18", "the end of
  // the draft"), or null when there is no gain-ranked list yet. Same value
  // TopThree's hint names. AvailableList only reads its NULLNESS now (is
  // there a horizon yet, to gate the toolbar note) -- it used to interpolate
  // the pick number into "Lasts = chance he's still there at pick 18", which
  // reads as a promise that pick 18 is the user's own turn. Verified against
  // the real model it never is (see lastsTitle's own comment below); the
  // toolbar note and the tooltip now describe the horizon's DISTANCE
  // instead, which needs only whether one exists, not this string's actual
  // content.
  horizonLabel: string | null
  // Opens the player's profile over the room (DraftRoom's PlayerOverlay).
  // The whole row is not the target -- only the name -- because every other
  // cell in this row is a number the eye is comparing down a column, and the
  // last cell is the Draft button. One deliberate target, nowhere near the
  // irreversible one.
  onOpenPlayer: (c: LiveCandidate) => void
  // Player ids the BOARD reports as drafted. Current the moment a pick
  // lands, unlike `candidates`, which only moves when a ranking finishes.
  draftedIds: Set<string>
}

// The ranked available pool: search + position filter above a table that
// opens in the server's own `gain_now` order (`#`) and can be re-sorted by
// any column from its header.
//
// Client-side sorting is confined to THIS component's own copy of the list
// on purpose. `candidates` is DraftRoom's `state.candidates`, handed to
// TopThree as well; the three recommendation cards must always be the
// server's top three in the server's order no matter what this table is
// sorted by, so nothing here may reorder the prop itself. `.filter()`
// already returns a fresh array and `.sort()` below only ever touches that
// -- the shared array is never mutated.
//
// One row, memoized. The table is ~250 rows of SVG sparklines and pip
// strips, and every recompute hands down a brand-new `candidates` array --
// so without this, a ranking landing re-rendered the entire table and the
// main-thread work stalled the take-out animation mid-flight. That is what
// "the recalculating blocks the animations" was.
//
// The props are deliberately primitives and stable references: `player` and
// `isTaken` are resolved by the parent rather than passed as the `players`
// map and the `taken` map, both of which get a new identity on every poll
// and would defeat the comparison entirely.
const AvailableRow = memo(function AvailableRow({
  c, player, isTaken, season, isMyTurn, onOpenPlayer, onDraft,
  onCellEnter, onCellLeave,
}: {
  c: LiveCandidate
  player: Player | undefined
  isTaken: boolean
  season: number | null
  isMyTurn: boolean
  onOpenPlayer: (c: LiveCandidate) => void
  onDraft: (c: LiveCandidate) => void
  // Stable identities from the parent (useCallback), because this row is
  // memoized: a fresh closure per render would defeat that on every tick.
  onCellEnter: (kind: CellTipKind, playerId: string, el: HTMLElement) => void
  onCellLeave: () => void
}): ReactNode {
  const level = healthLevel(player?.career_games_pg)
  const steady = steadyLevel(player?.consistency_pct)
  const arc = player?.season_finishes ?? null
  const change = player?.proj_change ?? null
  return (
              <tr key={c.player_id} data-pid={c.player_id}
                  className={isTaken ? 'avail-row-taken' : undefined}
                  aria-hidden={isTaken || undefined}>
                <td className="avail-col-rank mono">{c.rank}</td>
                <td className="avail-col-pos">{posBadge(c.position)}</td>
                <td className="avail-col-name">
                  {/* A button, not a link: this opens an overlay over the
                      room, and an <a href> here would offer a navigation
                      that no longer happens on click. The board grid keeps
                      its real href for exactly the opposite reason -- see
                      DraftBoardGrid.tsx. */}
                  <button
                    type="button"
                    className="avail-name-btn"
                    onClick={() => onOpenPlayer(c)}
                    title="Open profile"
                  >
                    <span className="avail-name">{player?.name ?? c.player_id}</span>
                  </button>
                  {player && (
                    <span className="avail-meta mono">
                      {player.team} · BYE {player.bye ?? '—'}
                    </span>
                  )}
                </td>
                <td
                  className="avail-col-games"
                  onMouseEnter={(e) => onCellEnter('games', c.player_id, e.currentTarget)}
                  onMouseLeave={onCellLeave}
                >
                  {/* An explicit empty state, never a blank cell. 45 of the
                      252 players on the real board have no games in the last
                      complete season -- 25 defenses (nflverse carries no
                      team-defense weekly rows at all), 17 rookies, and the
                      kickers of a league that prices no kicking. A blank cell
                      reads as "did not score"; a dash reads as "nothing to
                      show", which is the true one. */}
                  {player?.game_points
                    ? <GameBars points={player.game_points} season={season} position={c.position} />
                    : (
                      <span
                        className="gamebars-none"
                        title={`No games in ${season ?? 'the last complete season'}`}
                      >
                        —
                      </span>
                    )}
                </td>
                {/* Unknown (no `game_points` at all -- see missedGames above)
                    renders the same em-dash the sparkline's own empty state
                    uses, never a 0 or an empty strip: a defense or an
                    unpriced rookie has no counted season, not a clean one. */}
                <td
                  className="avail-col-finish"
                  onMouseEnter={(e) => onCellEnter('finish', c.player_id, e.currentTarget)}
                  onMouseLeave={onCellLeave}
                >
                  {arc === null || arc.length === 0
                    ? <span className="gamebars-none">—</span>
                    : <FinishArc arc={arc} position={c.position} />}
                </td>
                <td
                  className="avail-col-health"
                  onMouseEnter={(e) => onCellEnter('health', c.player_id, e.currentTarget)}
                  onMouseLeave={onCellLeave}
                >
                  {level === null
                    ? <span className="gamebars-none">—</span>
                    : <HealthMeter level={level}
                                    gamesPg={player?.career_games_pg ?? 0} />}
                </td>
                {/* Directly after Health: the two meters read as a pair
                    -- was he on the field, and was he worth starting when he
                    was -- and sharing the five-bar shape makes that pairing
                    the point rather than a coincidence. */}
                <td className="avail-col-health">
                  {steady === null
                    ? <span className="gamebars-none">—</span>
                    : <SteadyMeter level={steady}
                                   cv={player?.consistency_cv ?? null} />}
                </td>
                <td className="avail-col-change">
                  {change === null
                    ? <span className="gamebars-none">—</span>
                    : <ChangeMeter change={change} />}
                </td>
                <td className="avail-col-num mono avail-proj">{Math.round(c.proj_points)}</td>
                {/* null survive_pct (no roster to survive FOR yet) gets no
                    riskTone color at all -- riskTone's red/amber/green ramp
                    is a claim about a real probability, and coloring a dash
                    would imply one exists. */}
                <td
                  className="avail-col-num mono"
                  style={c.survive_pct === null ? undefined : { color: riskTone(c.survive_pct) }}
                >
                  {c.survive_pct === null ? '—' : `${Math.round(c.survive_pct)}%`}
                </td>
                <td className="avail-col-num mono avail-adp">{fmtRank(player?.market_rank ?? null)}</td>
                <td className="avail-col-num mono avail-adp">{fmtRank(player?.espn_ppr_rank ?? null)}</td>
                <td className="avail-col-btn">
                  <button
                    type="button"
                    className="avail-draft-btn"
                    disabled={!isMyTurn}
                    title={isMyTurn ? undefined : 'Not your turn yet'}
                    onClick={() => onDraft(c)}
                  >
                    Draft
                  </button>
                </td>
              </tr>
  )
})

// Both filters are client-side per the task brief ("the server sends the
// whole ranked list") -- the pool tops out in the low hundreds, cheap
// enough to filter AND sort on every keystroke without debouncing or memos.
export default function AvailableList({
  candidates, players, onDraft, isMyTurn, horizonLabel, onOpenPlayer,
  draftedIds,
}: AvailableListProps) {
  const [search, setSearch] = useState('')
  const [pos, setPos] = useState('ALL')
  // Opens on the server's ranking, which is the whole point of the list --
  // any other default would hide the model's answer behind a click.
  const [sort, setSort] = useState<{ key: SortKey; dir: SortDir }>({ key: 'rank', dir: 'asc' })

  // -- header tooltips: state, positioning, dismissal --
  //
  // One tooltip for the whole table, not one per header -- only one can
  // ever be open at a time (a pointer or a focus ring is only ever in one
  // place), so a single floating panel driven by "which header, and where"
  // is simpler than ten independent popovers and is exactly what a native
  // `title` already behaved like. `rect` is the trigger's own
  // `getBoundingClientRect`, captured at hover/focus time -- the same
  // approach `.board-pop`'s `HoverInfo` (DraftBoardGrid.tsx) already uses
  // for the identical clipping problem.
  const [tip, setTip] = useState<{ id: TipId; rect: DOMRect } | null>(null)
  // The tooltip's own measured position, filled in by the layout effect
  // below once its real size is known -- null between "a tip just opened"
  // and "its size got measured", which is one synchronous tick, never a
  // visible state (see that effect's own comment).
  const [tipPos, setTipPos] = useState<{ left: number; top: number } | null>(null)
  const tipRef = useRef<HTMLDivElement>(null)

  // The cell panels (Health, the season sparkline, Finish) are a SECOND
  // floating layer, not the header's. They share `positionTip` and the
  // measure-then-place effect, and nothing else: a header tip is one line of
  // static copy, a cell panel is a table fetched per player, and one panel
  // serving both would have to be empty while the other's data loaded.
  const [cellTip, setCellTip] =
    useState<{ kind: CellTipKind; playerId: string; rect: DOMRect } | null>(null)
  const [cellTipPos, setCellTipPos] = useState<{ left: number; top: number } | null>(null)
  const cellTipRef = useRef<HTMLDivElement>(null)
  const cellTimer = useRef<number | null>(null)

  const hideCellTip = useCallback(() => {
    if (cellTimer.current !== null) window.clearTimeout(cellTimer.current)
    cellTimer.current = null
    setCellTip(null)
  }, [])

  // Stable identity, because AvailableRow is memoized and a fresh closure per
  // render would re-render all 252 rows on every tick of the pick clock.
  const showCellTip = useCallback(
    (kind: CellTipKind, playerId: string, el: HTMLElement) => {
      if (cellTimer.current !== null) window.clearTimeout(cellTimer.current)
      const rect = el.getBoundingClientRect()
      // The fetch starts on the FIRST hover, before the panel is due to
      // appear, so the request and the delay overlap rather than queue. By
      // the time the panel opens the data is usually already cached.
      void loadProfile(playerId).catch(() => {})
      cellTimer.current = window.setTimeout(
        () => setCellTip({ kind, playerId, rect }), TIP_DELAY_MS)
    }, [])

  useLayoutEffect(() => {
    if (!cellTip || !cellTipRef.current) {
      setCellTipPos(null)
      return
    }
    const { width, height } = cellTipRef.current.getBoundingClientRect()
    setCellTipPos(positionTip(cellTip.rect, { width, height }))
    // `cellTip.playerId` is in the deps because the panel RESIZES when the
    // fetch lands -- a loading panel is one line and a game log is eighteen
    // rows, so a position measured against the small one would leave the
    // full panel hanging off the bottom of the window.
  }, [cellTip])

  useEffect(() => {
    if (!cellTip) return
    window.addEventListener('scroll', hideCellTip, true)
    return () => window.removeEventListener('scroll', hideCellTip, true)
  }, [cellTip, hideCellTip])
  const tipTimer = useRef<number | null>(null)

  function clearTipTimer(): void {
    if (tipTimer.current !== null) {
      window.clearTimeout(tipTimer.current)
      tipTimer.current = null
    }
  }
  // Hover gets a short delay -- ~130ms, not the ~1s a native `title` makes a
  // reader wait, but enough that sweeping the pointer across the header row
  // doesn't flash a tooltip per column it passes over.
  function scheduleTip(id: TipId, target: HTMLElement): void {
    clearTipTimer()
    const rect = target.getBoundingClientRect()
    tipTimer.current = window.setTimeout(() => setTip({ id, rect }), TIP_DELAY_MS)
  }
  // Keyboard focus skips the delay: a Tab press is already one deliberate,
  // discrete move, and making its result wait would read as the control
  // lagging rather than as considerate pacing.
  function showTipNow(id: TipId, target: HTMLElement): void {
    clearTipTimer()
    setTip({ id, rect: target.getBoundingClientRect() })
  }
  function hideTip(): void {
    clearTipTimer()
    setTip(null)
  }

  // Positions the tooltip AFTER it renders and BEFORE the browser paints --
  // `useLayoutEffect`, not `useEffect` -- so the panel's own measured size
  // can be used instead of a guessed constant. A newly-opened tip's first
  // render is invisible at (0, 0) (see the JSX below); this effect measures
  // it and computes where it actually belongs, and that second render,
  // still inside the same paint, is the only one anyone sees.
  useLayoutEffect(() => {
    if (!tip || !tipRef.current) {
      setTipPos(null)
      return
    }
    const { width, height } = tipRef.current.getBoundingClientRect()
    setTipPos(positionTip(tip.rect, { width, height }))
  }, [tip])

  // Same guard `.board-pop`'s own hover effect uses: any scroll -- this
  // table's own, or the page's -- invalidates the captured trigger rect, so
  // the tooltip is dropped rather than left hanging over whatever used to
  // be under it.
  useEffect(() => {
    if (!tip) return
    window.addEventListener('scroll', hideTip, true)
    return () => window.removeEventListener('scroll', hideTip, true)
  }, [tip])

  // The season the bars describe, read off the same board rows they ride in
  // on. `stats.season` and `game_points` are both cut from `max(season)` in
  // `weekly` (scoring/board.py's _latest_season_stats and
  // scoring/game_points.py), so they cannot name different years -- and a
  // header claiming the wrong year would be worse than one claiming none.
  // Null only until /api/players resolves, or for a board where nobody has a
  // season at all.
  const season = useMemo(() => {
    for (const p of Object.values(players)) {
      if (p.stats) return p.stats.season
    }
    return null
  }, [players])

  // A player leaving the board is the single most informative event in a
  // draft, and until now it was also the least visible: the row simply was
  // not there on the next render. `taken` keeps a departed row on screen
  // long enough to read, in the position it already occupied, then collapses
  // it out. Keyed by player so a second pick landing mid-animation queues
  // its own row rather than restarting a shared timer.
  // A row leaving the board is triggered by the BOARD saying so, not by the
  // player falling out of `candidates`. Those are a full ranking apart --
  // roughly a second -- and driving the animation off the candidate list is
  // what made every pick look like it was waiting for the recompute, because
  // it was. The board reads the drafted rows directly, so this fires as the
  // pick lands and the recompute happens alongside it rather than in front.
  const [taken, setTaken] = useState<Map<string, LiveCandidate>>(new Map())
  const seenDraftedRef = useRef<Set<string> | null>(null)
  const timersRef = useRef<Set<number>>(new Set())

  useEffect(() => {
    const before = seenDraftedRef.current
    seenDraftedRef.current = new Set(draftedIds)
    if (before === null) return          // first render: nothing "just" went

    const fresh = [...draftedIds].filter((id) => !before.has(id))
    // A burst is a resync, not picks -- restoring a session or reconnecting
    // reveals the whole board at once, and animating that describes
    // something that did not just happen.
    if (fresh.length === 0 || fresh.length > TAKEN_BURST_LIMIT) return

    // Held from the CANDIDATE list because that is the only place the row's
    // rendered values live; the board's own player object carries a
    // different, thinner shape.
    const departed = new Map<string, LiveCandidate>()
    for (const c of candidates) if (fresh.includes(c.player_id)) departed.set(c.player_id, c)
    if (departed.size === 0) return

    setTaken((prev) => new Map([...prev, ...departed]))
    // Deliberately NOT cleaned up when this effect re-runs: cancelling on
    // the next pick is what previously left rows stranded on screen forever.
    const timer = window.setTimeout(() => {
      timersRef.current.delete(timer)
      setTaken((prev) => {
        const next = new Map(prev)
        for (const id of departed.keys()) next.delete(id)
        return next
      })
    }, TAKEN_MS)
    timersRef.current.add(timer)
  }, [draftedIds, candidates])

  useEffect(() => {
    const timers = timersRef.current
    return () => { for (const t of timers) window.clearTimeout(t); timers.clear() }
  }, [])

  const q = search.trim().toLowerCase()
  const visible = candidates.filter((c) => {
    // A drafted player stays only while he is ANIMATING. Once his fade is
    // done he leaves `taken`, but the board still calls him drafted and the
    // candidate list still carries him until the next ranking lands about a
    // second later -- so without this he sat there at opacity 0, occupying a
    // full row, and the hole the short fade was meant to close reappeared
    // one step further on.
    if (draftedIds.has(c.player_id) && !taken.has(c.player_id)) return false
    if (pos !== 'ALL' && c.position !== pos) return false
    if (!q) return true
    const player = players[c.player_id]
    const haystack = `${player?.name ?? c.player_id} ${player?.team ?? ''}`.toLowerCase()
    return haystack.includes(q)
  })
  // `visible` is already a fresh array from `.filter()`; sorting it in place
  // cannot reach `candidates`. Written as a copy anyway so that stays true
  // if the filter is ever short-circuited away for the unfiltered case.
  // Departing rows are merged back in and sorted with everyone else, so a
  // taken player animates out from where he actually sat rather than jumping
  // to the end of the list on his way off it.
  // A held row is only ADDED BACK if the candidate list has already dropped
  // him. Since the animation fires off the board, a player is normally still
  // in `candidates` for the second or so it takes the next ranking to land --
  // and appending the held copy on top of the live one rendered him twice,
  // two identical rows at the same rank, both animating out, sharing a React
  // key. The held copy exists to outlive the recompute, not to duplicate it.
  const present = new Set(visible.map((c) => c.player_id))
  const held = [...taken.values()].filter((c) => {
    if (present.has(c.player_id)) return false
    if (pos !== 'ALL' && c.position !== pos) return false
    if (!q) return true
    const player = players[c.player_id]
    return `${player?.name ?? c.player_id} ${player?.team ?? ''}`
      .toLowerCase().includes(q)
  })
  const withTaken = held.length === 0 ? visible : [...visible, ...held]
  const rows = withTaken.slice().sort((a, b) => compareRows(a, b, sort.key, sort.dir, players))

  // FLIP, because a table row cannot be collapsed. Animating a `td` to height
  // zero does not shrink the row while its sparkline still has intrinsic
  // height, so the row either stands there empty or vanishes and everything
  // below it jumps. Instead the leaving row fades, is removed, and the
  // survivors are put back at their OLD offsets and released -- they slide
  // into place under a transform, which costs no layout and cannot fight the
  // table's own sizing.
  const bodyRef = useRef<HTMLTableSectionElement | null>(null)
  const offsetsRef = useRef<Map<string, number>>(new Map())

  // Which rows are on screen, in order. Reading `offsetTop` forces a layout,
  // and doing that for ~250 rows on EVERY render would hand back the cost the
  // row memoization just saved -- on a poll where nothing moved, most of all.
  // Row positions cannot change unless this string does.
  const rowKey = rows.map((r) => r.player_id).join(',')
  const rowKeyRef = useRef('')

  useLayoutEffect(() => {
    const body = bodyRef.current
    if (!body) return
    if (rowKey === rowKeyRef.current) return
    rowKeyRef.current = rowKey
    const previous = offsetsRef.current
    const next = new Map<string, number>()
    const moved: Array<[HTMLElement, number]> = []

    for (const el of Array.from(body.children) as HTMLElement[]) {
      const pid = el.dataset.pid
      if (!pid) continue
      const top = el.offsetTop
      next.set(pid, top)
      const was = previous.get(pid)
      // Only rows that were already on screen AND actually moved. A row
      // appearing for the first time has nowhere to slide from.
      if (was !== undefined && was !== top) moved.push([el, was - top])
    }
    offsetsRef.current = next
    if (moved.length === 0) return

    // Respect the same preference the fade does: with reduced motion the
    // rows simply appear in their new places rather than travelling there.
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
    for (const [el, delta] of moved) {
      el.style.transition = 'none'
      el.style.transform = `translateY(${delta}px)`
    }
    // One frame at the old position, then release: without the double rAF the
    // browser coalesces both styles into a single paint and nothing animates.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      for (const [el] of moved) {
        el.style.transition = `transform ${SLIDE_MS}ms cubic-bezier(0.4, 0, 0.2, 1)`
        el.style.transform = ''
      }
    }))
  })


  function toggleSort(key: SortKey): void {
    setSort((prev) => (prev.key === key
      ? { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' }
      : { key, dir: NATURAL_DIR[key] }))
  }

  // A sortable header. The clickable thing is a real <button> inside the
  // <th>, not a click handler on the cell: this table is driven under a
  // pick clock and has to stay tabbable, and `aria-sort` on the header is
  // how a screen reader gets the same "sorted by, this way" the caret gives
  // everyone else.
  //
  // Hover/focus land on the <th> itself, not the button -- `onMouseEnter`
  // only fires for pointer movement into the element it is attached to (it
  // does not bubble the way a plain DOM `mouseenter` listener on a parent
  // would need to), so putting it here makes the whole cell the trigger,
  // not just the button's own box, matching how the removed `title` used to
  // fire no matter where in the cell the pointer sat. `onFocus`/`onBlur`
  // land here too and still catch the button's own focus/blur: React
  // implements them on `focusin`/`focusout` under the hood, which DO
  // bubble, so a listener on the <th> sees the <button> inside it gain and
  // lose focus without needing its own pair of handlers.
  //
  // WHY NOT THE NATIVE `title` THIS USED TO BE: ~1s of stationary hover to
  // appear, no hint beforehand that anything is there, and an unstyled OS
  // box that can cover the table. See the `.avail-th-tip` section of
  // App.css and `positionTip` above for the replacement -- one shared
  // floating panel, positioned off this header's own `getBoundingClientRect`
  // the same way `.board-pop` (DraftBoardGrid.tsx) already solves the exact
  // same "must not clip inside a scroll region, must not shift the row"
  // problem for the main board's cells.
  function sortableTh(key: SortKey, label: ReactNode, className?: string): ReactNode {
    const active = sort.key === key
    return (
      <th
        className={`${className ?? ''}${active ? ' is-sorted' : ''}`.trim() || undefined}
        aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}
        onMouseEnter={(e) => scheduleTip(key, e.currentTarget)}
        onMouseLeave={hideTip}
        onFocus={(e) => showTipNow(key, e.currentTarget)}
        onBlur={hideTip}
        onKeyDown={(e) => { if (e.key === 'Escape') hideTip() }}
      >
        <button
          type="button"
          className="avail-th-btn"
          onClick={() => toggleSort(key)}
          aria-describedby="avail-th-tip"
        >
          {/* The dotted underline is the "there is an explanation here"
              affordance a reader can see before ever hovering -- quiet on
              purpose, the same weight across all eight sortable headers
              rather than an icon that would compete with the sort caret
              right next to it. */}
          <span className="avail-th-hint">{label}</span>
          {/* The caret is ALWAYS in the markup, transparent until the column
              is the sorted one (or hovered) -- rendering it only when active
              made every header jump sideways by its own width on each sort
              click, which under a pick clock reads as the table twitching.
              Idle, it points the way that column's first click will sort
              (NATURAL_DIR), so hovering PROJ shows ▼ and hovering ADP ▲ --
              the affordance and the promise in one glyph. */}
          <span
            className={`avail-sort${active ? '' : ' is-idle'}`}
            aria-hidden="true"
          >
            {(active ? sort.dir : NATURAL_DIR[key]) === 'asc' ? '▲' : '▼'}
          </span>
        </button>
      </th>
    )
  }

  // Every header's tooltip text, gathered here rather than written inline
  // ten times over so the wording stays consistent with itself and with the
  // rest of the room's own voice (ESPN's own comment on the column above).
  // The owner asked for these because "I don't always know what the columns
  // mean" -- so each one states the actual meaning, not a paraphrase of it.
  // Delivery is `tipCopy`/the shared `.avail-th-tip` panel below, not a
  // native `title` any more (see sortableTh's own comment) -- only that
  // mechanism changed; this is the same copy that was already reviewed and
  // confirmed accurate, except `healthTitle` (the meter replaced a bare
  // number, so its own tooltip has to describe the strip) and `lastsTitle`
  // (see the comment above that one).
  const rankTitle = "This board's own rank of who to take now. Not ADP and "
    + 'not projected points -- it ranks by how much you gain by taking this '
    + 'player now versus waiting until your next pick, weighted by whether '
    + 'your roster can actually start him.'
  const posTitle = "The player's position."
  const playerTitle = 'Name, NFL team, and bye week.'
  // Position-dependent colour cut points (barThresholds above) mean a flat
  // "red under 10, green 15+" claim would be wrong for a QB, K, or DST row --
  // this spells out all three splits rather than one that only fits 3 of 6
  // positions.
  const gamesTitle = `Fantasy points scored in each week of the ${season ?? 'last complete'} `
    + "season, under this league's own scoring, in week order. A gap (faint "
    + 'mark) is a week with no game. A short bar down at the floor is a game '
    + 'he played and scored little or nothing in -- the two are drawn '
    + 'differently on purpose. Colour depends on position: quarterbacks red '
    + 'under 15 / amber 15-25 / green 25+, kickers and defenses red under 5 / '
    + 'amber 5-10 / green 10+, everyone else red under 10 / amber 10-15 / '
    + 'green 15+. Not sortable.'
  // Rewritten for the pip strip (MissedPips above): explains what the three
  // marks mean AND is honest that the strip cannot place the bye in its
  // real week -- `Player.bye` is next season's, `game_points` is last
  // season's, and they only happen to agree 157 of 207 times on the real
  // board, which is exactly what two different seasons' schedules colliding
  // by chance would produce. See missedGames's own comment for the count
  // math, which this strip does not change.
  const finishTitle = 'Where he finished at his position in each of the last '
    + 'five seasons, oldest bar on the left. Taller is better, and the colour '
    + 'is how useful that finish was: green was the best quarter of a '
    + "league's starters at the position, red was outside fantasy relevance "
    + 'entirely. Ranked on season points, so a year cut short by injury shows '
    + 'as the bad finish it was. Sorts on the most recent season.'
  const changeTitle = 'How many more (or fewer) points per game he is '
    + 'PROJECTED for than he actually averaged LAST SEASON. Per game on both '
    + 'sides, so a year cut short by injury does not read as decline. Five '
    + 'bars is the biggest projected gain and one bar the biggest fall, the '
    + 'same direction as Health and Steady. One caveat: a large fall is '
    + 'usually a lost starting job rather than a player getting worse -- '
    + 'ESPN projects a full season for almost everyone, so a backup shows as '
    + 'a huge per-game fall. This is what the projection expects, not a '
    + 'forecast of our own.'
  const healthTitle = "How available this player has been across his whole "
    + 'career: average games played per season, counting seasons he missed '
    + 'entirely rather than skipping them. Five bars is close to a full '
    + 'season every year; one bar is a player who has missed a lot of '
    + 'football. Blank for a defense, or anyone with no NFL season yet.'
  const steadyTitle = 'How steady his scoring has been week to week, ranked '
    + 'against the other players at his position ON THIS BOARD. Five bars is '
    + 'the steadiest fifth, one bar the spikiest. Measured as swing relative '
    + 'to his own average, not raw swing -- a 20-point-a-week player moves in '
    + 'bigger absolute points than an 8-point one without being less '
    + 'reliable. Steady is not the same as good: a spiky player can be worth '
    + 'more if his ceiling is why you want him. Blank for anyone without a '
    + 'full-enough recent season to measure.'
  const projTitle = "Projected fantasy points for the full upcoming season, "
    + "under this league's own scoring."
  // NOT "chance he's still there at pick N" any more -- verified against the
  // real model for an 8-team draft at slot 2 (own turns 2, 15, 18, 31, 34,
  // 47): the pick this number is measured against came back 13, 27, 27, 29,
  // 43, every single one somebody else's turn, never the user's own. That
  // is not a bug in the number -- scoring/draft_sim.horizon_picks/
  // horizon_ceiling measure a fixed distance (roughly a round to a round
  // and a half of opponent picks) on purpose, because the user's own next
  // turn is usually much further off, and at that distance survival reads
  // ~0% for every row and the ranking loses its signal. The number was
  // right; the old label just asserted a pick the user was never actually
  // making. This one names the HORIZON instead of a pick, and says why it
  // stops there rather than at the user's own turn.
  const lastsTitle = "Chance he's still on the board roughly a round to a "
    + "round and a half from now -- not at your own next pick, which is "
    + 'almost always further off than that. Measured any nearer and nearly '
    + 'every player would read close to 100% (nothing left to rank by); any '
    + 'further and nearly every player would read close to 0% (same '
    + 'problem, the other way) -- this is the furthest point out that still '
    + 'tells the rows apart. A low percentage is the argument for taking '
    + 'him now.'
  const adpTitle = 'Average draft position across the consensus of public '
    + 'sources -- where the market as a whole takes him.'
  const espnTitle = "ESPN's own ranking, shown so you can see where this "
    + "board disagrees with the platform you're drafting on."
  const draftColTitle = "Draft this player onto your roster. Only enabled on "
    + 'your turn.'

  // One shared floating panel (below, `.avail-th-tip`) reads from this
  // instead of each header carrying its own `title` -- see sortableTh's
  // comment for why.
  const tipCopy: Record<TipId, string> = {
    rank: rankTitle,
    pos: posTitle,
    player: playerTitle,
    games: gamesTitle,
    finish: finishTitle,
    health: healthTitle,
    steady: steadyTitle,
    change: changeTitle,
    proj: projTitle,
    lasts: lastsTitle,
    adp: adpTitle,
    espn: espnTitle,
    draft: draftColTitle,
  }

  return (
    <div className="avail">
      <div className="avail-toolbar">
        <input
          type="text"
          className="avail-search"
          placeholder="Search players"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search available players"
        />
        <div className="avail-pills">
          {POSITIONS.map((p) => (
            <button
              key={p}
              type="button"
              className={`avail-pill${pos === p ? ' is-active' : ''}`}
              onClick={() => setPos(p)}
            >
              {p}
            </button>
          ))}
        </div>
        {horizonLabel !== null && (
          <div className="avail-horizon-note">
            <span className="avail-horizon-key">Lasts</span>
            {" = chance he's still on the board roughly a round to a round "
              + 'and a half from now'}
          </div>
        )}
      </div>

      <table className="avail-table">
        <thead>
          <tr>
            {sortableTh('rank', '#', 'avail-col-rank')}
            {sortableTh('pos', 'Pos', 'avail-col-pos')}
            {sortableTh('player', 'Player', 'avail-col-name')}
            {/* The one header in this table that is NOT a control, and it
                says so rather than sitting there looking like the seven that
                are. There is no honest single number to sort a distribution
                by: "most 15-point games" and "fewest under 10" and "highest
                median" are three different questions, and picking one would
                make the column quietly answer a question nobody asked. The
                quantity that DOES summarise a season is already sortable two
                columns over -- Proj -- and `stats.ppg` is on the profile the
                name opens. So: no button, no caret, default cursor, and the
                word in the header. Still gets the same tooltip treatment as
                every sortable one (mouse only -- nothing here is focusable,
                same as it was under the native `title`). */}
            <th
              className="avail-col-games"
              onMouseEnter={(e) => scheduleTip('games', e.currentTarget)}
              onMouseLeave={hideTip}
            >
              <span className="avail-th-hint">{season ?? 'Last'}</span>
              <span className="avail-nosort">no sort</span>
            </th>
            {/* Next to the sparkline deliberately: one column is how he
                scored week to week, the next is whether he was there to do
                it. Wider than the numeric columns (see `.avail-col-health`)
                to fit five bars without growing the 32px row. */}
            {sortableTh('finish', 'Finish', 'avail-col-finish')}
            {sortableTh('health', 'Health', 'avail-col-health')}
            {sortableTh('steady', 'Steady', 'avail-col-health')}
            {sortableTh('change', 'Change', 'avail-col-change')}
            {sortableTh('proj', 'Proj', 'avail-col-num')}
            {/* One word. A header naming the horizon at all ("Lasts to pick
                13") reads as a promise that pick 13 is the user's own turn,
                which it verifiably never is (see lastsTitle's own comment)
                -- so the header stays bare and the horizon lives in the
                toolbar note and the tooltip instead, as a DISTANCE rather
                than a pick number. */}
            {sortableTh('lasts', 'Lasts', 'avail-col-num')}
            {sortableTh('adp', 'ADP', 'avail-col-num')}
            {/* ESPN's own PPR rank, always on screen next to this board's
                `#` and the market's ADP -- the owner asked to be able to see
                where ESPN has a player against where this board has him,
                without going into the profile for it. */}
            {sortableTh('espn', 'ESPN', 'avail-col-num')}
            <th
              className="avail-col-btn"
              onMouseEnter={(e) => scheduleTip('draft', e.currentTarget)}
              onMouseLeave={hideTip}
            />
          </tr>
        </thead>
        <tbody ref={bodyRef}>
          {rows.map((c) => (
            <AvailableRow
              key={c.player_id}
              c={c}
              player={players[c.player_id]}
              isTaken={taken.has(c.player_id) || draftedIds.has(c.player_id)}
              season={season}
              isMyTurn={isMyTurn}
              onOpenPlayer={onOpenPlayer}
              onDraft={onDraft}
              onCellEnter={showCellTip}
              onCellLeave={hideCellTip}
            />
          ))}
          {rows.length === 0 && (
            <tr>
              {/* 10 = the nine columns above (rank, pos, player, games,
                  missed, proj, lasts, adp, espn) plus the draft button's. */}
              <td colSpan={10} className="avail-empty">
                {candidates.length === 0 ? 'No candidates yet.' : 'No players match this filter.'}
              </td>
            </tr>
          )}
        </tbody>
      </table>

      {/* The one floating tooltip, shared by every header above -- see the
          state/effects near the top of this component and `.avail-th-tip`
          in App.css. Only mounted while something is actually hovered or
          focused; visible only once `tipPos` is known (the
          `useLayoutEffect` above measures it in the same tick this mounts,
          so nobody ever sees the (0, 0) frame in between). Fixed
          positioning is the load-bearing part -- it escapes `.draft-main`'s
          own scroll region entirely rather than being clipped inside it,
          the same reason `.board-pop` (DraftBoardGrid.tsx) is fixed too. */}
      {tip && (
        <div
          id="avail-th-tip"
          ref={tipRef}
          role="tooltip"
          className="avail-th-tip"
          style={tipPos
            ? { left: tipPos.left, top: tipPos.top, visibility: 'visible' }
            : { left: 0, top: 0, visibility: 'hidden' }}
        >
          {tipCopy[tip.id]}
        </div>
      )}
      {cellTip && (
        <div
          ref={cellTipRef}
          role="tooltip"
          className="avail-cell-tip"
          style={cellTipPos
            ? { left: cellTipPos.left, top: cellTipPos.top, visibility: 'visible' }
            : { left: 0, top: 0, visibility: 'hidden' }}
        >
          <CellTip kind={cellTip.kind} playerId={cellTip.playerId} />
        </div>
      )}
    </div>
  )
}
