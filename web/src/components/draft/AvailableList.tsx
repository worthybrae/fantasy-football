import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { LiveCandidate, Player } from '../../api'
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
const BAR_THRESHOLDS: Record<string, { amberFrom: number; greenFrom: number }> = {
  QB: { amberFrom: 10, greenFrom: 20 },
  K: { amberFrom: 5, greenFrom: 10 },
  DST: { amberFrom: 5, greenFrom: 10 },
}
// RB, WR, TE, and anything the server ever sends that isn't one of the three
// positions above -- the original 10/15 split.
// How long a drafted player stays on screen on his way out. Long enough to
// read a name under a pick clock, short enough that back-to-back picks do
// not stack up on each other.
const TAKEN_MS = 1250
// More than this vanishing at once is a resync -- a restored session, a
// reconnect mid-draft -- not picks. Animating that would be a screenful of
// motion describing something that did not just happen.
const TAKEN_BURST_LIMIT = 5

const BAR_THRESHOLDS_DEFAULT = { amberFrom: 10, greenFrom: 15 }
function barThresholds(position: string): { amberFrom: number; greenFrom: number } {
  return BAR_THRESHOLDS[position] ?? BAR_THRESHOLDS_DEFAULT
}

// The bars share ONE vertical scale across every row, so a bad player's best
// week cannot draw as tall as a stud's. 30 points is the ceiling (anything
// above it draws full height): on the same 252-player board it clips 3.0% of
// games, while the 95th percentile is 27.3 and the median 9.2 -- so almost
// every bar lands inside the scale and the ones that clip are already the
// unmistakable ones. A per-row maximum was rejected for exactly the reason
// the fixed colour thresholds were.
const BAR_CEILING = 30
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

function barTone(points: number, position: string): string {
  const { amberFrom, greenFrom } = barThresholds(position)
  if (points >= greenFrom) return 'is-good'
  if (points >= amberFrom) return 'is-mid'
  return 'is-bad'
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
        <span key={i} className="gamebar is-none" title={`Week ${i + 1}: no game`} />
      ) : (
        <span
          key={i}
          className={`gamebar ${barTone(p, position)}`}
          style={{ height: `${barHeight(p)}px` }}
          title={`Week ${i + 1}: ${p.toFixed(1)}`}
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

type SortKey = 'rank' | 'pos' | 'player' | 'missed' | 'proj' | 'lasts' | 'adp' | 'espn'
type SortDir = 'asc' | 'desc'

// The direction a column gets on its FIRST click -- "best first" for that
// particular column, which is not the same arrow everywhere: rank/ADP/ESPN
// are ranks (1 is best, so ascending), proj/lasts are quantities (bigger is
// better, so descending). MISSED follows the rank/ADP/ESPN logic, not
// proj/lasts's -- 0 missed games is the best outcome a row can have, so its
// first click is ascending too, same as a rank where 1 is best. Clicking an
// already-sorted header flips it, so both directions stay reachable on every
// column; this only decides which one you land on without having to click
// twice.
const NATURAL_DIR: Record<SortKey, SortDir> = {
  rank: 'asc', pos: 'asc', player: 'asc', missed: 'asc', proj: 'desc', lasts: 'desc', adp: 'asc',
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

// -- weeks missed -----------------------------------------------------------
//
// How many of last season's 18 weeks were neither the bye nor a game he
// played -- injury, inactivity, or (for a backup) simply not being the
// starter yet. `game_points` always carries exactly 18 entries when it
// carries any at all (scoring/game_points.py, index 0 = week 1), and every
// player has exactly one bye among his nulls, so missed = (null count) - 1.
// Floored at 0 rather than allowed negative: a player traded mid-season can
// have ZERO null entries (he played 18 game-weeks across two teams with no
// single bye week of his own to subtract), and that is a real 0, not a sign
// something else went wrong.
//
// `null` here (never 0) means UNKNOWN, not "missed nothing" -- `game_points`
// itself is null for 45 of 252 players on the real board (every D/ST, plus
// rookies and kickers with no priced weekly history), who have no season to
// count absences out of at all. A pure function so the sort column and the
// rendered cell read off the literal same number, the same discipline
// `sortValue` below already keeps for every other column.
function missedGames(points: (number | null)[] | null | undefined): number | null {
  if (!points) return null
  const nulls = points.filter((p) => p === null).length
  return Math.max(0, nulls - 1)
}

// Colour ramp for the MISSED cell: recedes to --text-3 at 0, where most of
// the board sits, and pops toward --fail as the count climbs -- the same
// color-mix idiom PlanTab's cliffTone uses for its own recede-vs-pop cells
// (that file's comment explains the technique), reused rather than
// reinvented because it is already this app's answer to "most values here
// are small and unremarkable, a few are large and are the entire point of
// the column."
//
// A 10-game absence and a 14-game one (Najee Harris's real 2025 number) do
// not need to read as two different shades of bad -- both clamp to full
// --fail, because past a point the column's only job is "this player was
// not on the field".
//
// Where the colour ramp saturates. MEASURED, not guessed, on the real
// 2025 board (184 of 252 players carry weekly rows; the rest -- every
// defense, rookies, unpriced kickers -- have no number at all and render as
// an em-dash): the median player missed 1 game, p75 is 5, p90 is 9, p95 is
// 12 and the worst is 16. A ceiling of 10 puts 7.6% of the board at full
// intensity, so the top of the ramp stays the genuinely notable group. An
// earlier ceiling of 6 saturated 21.7% of players -- over a fifth of the
// column rendering identical maximum red, which is exactly the contrast
// this column exists to provide.
const MISSED_CEILING = 10
function missedTone(n: number): string {
  const k = Math.max(0, Math.min(1, n / MISSED_CEILING))
  return `color-mix(in srgb, var(--fail) ${Math.round(k * 100)}%, var(--text-3))`
}

// The same "big number gets heavier" idiom `.plan-cliff-value.is-loud` and
// `.plan-avail-value.is-drop` already use, so a real absence pops on shape
// as well as colour -- not colour alone, which a colourblind reader or a
// dim/greyscale screenshot loses entirely. Set below the colour ramp's own
// ceiling on purpose: by the time a cell reads fully red it should already
// be bold too, not still waiting on one more missed game to earn it. Six of
// a 17-game season is a third of the year missed and covers 21.7% of the
// board -- frequent enough to be worth flagging, rare enough that the bold
// still means something.
const MISSED_LOUD_FROM = 6

// -- weeks missed pip strip -------------------------------------------------
//
// A pip per week (18, always), grouped by STATE rather than drawn in real
// week order -- deliberately. `Player.bye` is the bye for the UPCOMING
// season; `game_points` is LAST season's weekly rows. They only agree by
// coincidence, and on the real board they do not agree often: 157 of 207
// players "played through" their own `bye` index in `game_points`, which is
// exactly what comparing two different seasons' schedules would produce.
// There is nothing in this data that says which of a player's null weeks
// was actually his bye, so this strip does not pretend to place one -- real
// week order already lives one column to the left, in the per-week
// sparkline (GameBars above).
//
// What IS knowable, and what this draws: how many weeks he played, that
// exactly one of the rest is being counted as the bye (missedGames' own
// `- 1`), and how many are left over as genuinely missed. Grouped, that is
// three honest counts instead of one dishonest calendar. Memoized on
// `points`'s identity for the same reason GameBars is: without it every
// re-filter/re-sort would rebuild up to 250 x 18 = 4,500 spans.
// The season drawn as 17 marks -- one per game a player could have played
// -- with the ones he missed in `--fail` at the right end and the rest
// receding. Seventeen, not eighteen, because the bye is excluded outright:
// it is a week nobody is penalised for, and the numeral beside this strip
// already excludes it (missedGames is nulls - 1). Excluding it here too
// keeps the picture and the number telling the same story, and drops a
// third pip state that was costing width without earning it.
//
// NOT drawn in true week order, and it cannot be. One of the gaps in
// `game_points` is the bye and nothing available says which: `Player.bye`
// is the UPCOMING season's bye while `game_points` is last season's, and
// they disagree for 157 of the 207 players who have both. So the marks are
// grouped by state rather than sequenced by week. The real week ordering is
// already on screen -- it is the `2025` sparkline immediately to the left,
// which draws its gaps where they actually fell.
const MISSED_PIP_COUNT = 17

const MissedPips = memo(function MissedPips({ missed }: { missed: number }): ReactNode {
  const bad = Math.max(0, Math.min(MISSED_PIP_COUNT, missed))
  const ok = MISSED_PIP_COUNT - bad
  return (
    <span
      className="missed-pips"
      role="img"
      aria-label={bad === 0
        ? 'played every game outside the bye'
        : `${bad} of ${MISSED_PIP_COUNT} games missed`}
    >
      {Array.from({ length: ok }, (_, i) => (
        <span key={`ok-${i}`} className="missed-pip is-played" />
      ))}
      {Array.from({ length: bad }, (_, i) => (
        <span key={`bad-${i}`} className="missed-pip is-missed" />
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
    case 'missed': return missedGames(player?.game_points)
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
}: {
  c: LiveCandidate
  player: Player | undefined
  isTaken: boolean
  season: number | null
  isMyTurn: boolean
  onOpenPlayer: (c: LiveCandidate) => void
  onDraft: (c: LiveCandidate) => void
}): ReactNode {
  const missed = missedGames(player?.game_points)
  return (
              <tr key={c.player_id} className={isTaken ? 'avail-row-taken' : undefined}
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
                <td className="avail-col-games">
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
                <td className="avail-col-missed">
                  {player?.game_points
                    ? (
                      <span className="avail-missed-wrap">
                        <MissedPips missed={missed ?? 0} />
                        <span
                          className={`avail-missed-num mono${
                            missed !== null && missed >= MISSED_LOUD_FROM ? ' is-loud' : ''}`}
                          style={missed === null ? undefined : { color: missedTone(missed) }}
                        >
                          {missed}
                        </span>
                      </span>
                    )
                    : <span className="gamebars-none">—</span>}
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
  const [taken, setTaken] = useState<Map<string, LiveCandidate>>(new Map())
  // The PREVIOUS candidate list, held whole rather than as a set of ids. By
  // the time a player is missing from `candidates` his row data has gone
  // with him, and this is the only copy left to draw the departing row from.
  // One ref, updated at the end of the same effect that reads it, so nothing
  // depends on the order two effects happen to run in.
  const prevRef = useRef<LiveCandidate[] | null>(null)
  const timersRef = useRef<Set<number>>(new Set())

  useEffect(() => {
    const before = prevRef.current
    prevRef.current = candidates
    if (before === null) return          // first render: nothing has "gone"

    const now = new Set(candidates.map((c) => c.player_id))
    const departed = new Map<string, LiveCandidate>()
    for (const c of before) if (!now.has(c.player_id)) departed.set(c.player_id, c)
    // A burst is a resync, not a draft. Restoring a session or reconnecting
    // mid-draft drops dozens of players at once, and animating that is a
    // screenful of motion describing something that did not just happen.
    if (departed.size === 0 || departed.size > TAKEN_BURST_LIMIT) return

    setTaken((prev) => new Map([...prev, ...departed]))
    // The timer is NOT cleaned up when this effect re-runs, and that is the
    // point. Returning a clearTimeout here cancels the removal of rows that
    // are already mid-animation the moment the NEXT pick lands -- which in a
    // draft is constantly -- so those rows never leave `taken` and sit as a
    // permanent blank gap in the table. Each batch owns its own timer and is
    // allowed to finish; the ref exists only so unmount can sweep them.
    const timer = window.setTimeout(() => {
      timersRef.current.delete(timer)
      setTaken((prev) => {
        const next = new Map(prev)
        for (const id of departed.keys()) next.delete(id)
        return next
      })
    }, TAKEN_MS)
    timersRef.current.add(timer)
  }, [candidates])

  // Sweep on unmount only: leaving a draft mid-animation should not leave
  // timers firing setState against a component that is gone.
  useEffect(() => {
    const timers = timersRef.current
    return () => { for (const t of timers) window.clearTimeout(t); timers.clear() }
  }, [])

  const q = search.trim().toLowerCase()
  const visible = candidates.filter((c) => {
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
  const withTaken = taken.size === 0
    ? visible
    : [...visible, ...[...taken.values()].filter((c) => {
        if (pos !== 'ALL' && c.position !== pos) return false
        if (!q) return true
        const player = players[c.player_id]
        return `${player?.name ?? c.player_id} ${player?.team ?? ''}`
          .toLowerCase().includes(q)
      })]
  const rows = withTaken.slice().sort((a, b) => compareRows(a, b, sort.key, sort.dir, players))


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
  // confirmed accurate, except `missedTitle` (the pip strip replaced a bare
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
  const missedTitle = 'Weeks without a game last season, drawn as 18 marks: '
    + 'games played, then one mark for the week counted as the bye, then '
    + "the rest -- genuinely missed (injury, inactive, or a backup who "
    + "wasn't starting yet). Grouped by state, not real week order: which "
    + "gap was actually the bye can't be told apart from the others in this "
    + 'data, so it is always drawn last rather than in its true spot -- real '
    + 'week order is already the sparkline immediately to the left. The '
    + 'number counts only the missed weeks, same as it always has.'
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
    missed: missedTitle,
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
            {/* Placed immediately after the sparkline it is drawn from --
                absence sits next to the per-week chart that shows it. Wider
                than the other numeric columns on purpose -- see
                `.avail-col-missed` in App.css -- to fit the pip strip
                (MissedPips above) next to its number without growing the
                32px row. */}
            {sortableTh('missed', 'Missed', 'avail-col-missed')}
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
        <tbody>
          {rows.map((c) => (
            <AvailableRow
              key={c.player_id}
              c={c}
              player={players[c.player_id]}
              isTaken={taken.has(c.player_id)}
              season={season}
              isMyTurn={isMyTurn}
              onOpenPlayer={onOpenPlayer}
              onDraft={onDraft}
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
    </div>
  )
}
