import { Fragment, useEffect, useState, type CSSProperties, type FocusEvent, type MouseEvent, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import BoardPeek from './draft/BoardPeek'
import type { BoardCell, BoardPlayer, LiveBoard, PickMaker } from '../api'

// duplicated from LiveDraft.tsx (unexported there): a four-line pure
// function isn't worth a shared module between the board's two views. The
// same call is made in RosterPanel/AvailableList/TopThree, which carry their
// own copies for the same reason.
function posBadge(position: string | undefined): ReactNode {
  if (!position) return null
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

// -- who made the pick --------------------------------------------------
//
// Only a recorded mock draft carries these labels (/api/mocks/{id}/board);
// the live room's own board never does, so `made_by` is undefined there and
// every helper below reads that exactly as 'unknown' -- no label, no
// shading, the cell the /draft board has always drawn.
//
// The vocabulary is one ramp: how much of a PERSON is behind the seat.
// On the board it is drawn as light -- a person's pick at full strength, a
// seat that emptied turned down, a seat nobody took turned down further
// (see App.css) -- and said in each tile's aria-label and hover title. The
// glyphs below are no longer painted on tiles; they survive for the /mocks
// key, where they are hidden from sight and read only by screen readers.
export const MAKER_MARK: Record<Exclude<PickMaker, 'unknown'>, string> = {
  human: '\u25cf',
  auto: '\u25d0',
  engine: '\u25cb',
  us: '\u25c6',
}

// Read aloud in the cell's own aria-label, and shown on hover. Written as
// what happened, not as the label's name -- "auto" means nothing to someone
// who has not read the legend.
export const MAKER_LABEL: Record<Exclude<PickMaker, 'unknown'>, string> = {
  human: 'picked by a person',
  auto: 'autopicked -- this seat had a person, but ESPN was picking for them',
  engine: 'picked by ESPN -- this seat never had a person',
  us: 'picked by our bot',
}

function isLabelled(made_by: PickMaker | undefined): made_by is Exclude<PickMaker, 'unknown'> {
  return made_by !== undefined && made_by !== 'unknown'
}

interface HoverInfo {
  cell: BoardCell
  rect: DOMRect
}

const POPOVER_WIDTH = 340
const POPOVER_GAP = 8
// Rough popover height -- there's no real box to measure until it's already
// painted, and this only has to be close enough to decide which side of the
// cell has room, not pixel-exact.
const POPOVER_EST_HEIGHT = 320

function popoverStyle(rect: DOMRect): CSSProperties {
  const left = Math.max(8, Math.min(rect.left, window.innerWidth - POPOVER_WIDTH - 8))
  const below = rect.bottom + POPOVER_GAP
  const top = below + POPOVER_EST_HEIGHT <= window.innerHeight
    ? below
    : Math.max(8, rect.top - POPOVER_EST_HEIGHT - POPOVER_GAP)
  return { left, top }
}

// The +/- vs ADP: where the pick landed relative to its PPR consensus ADP
// (value = overall pick - ADP). Green and signed "+" when the player fell past
// his ADP (a value); red when he went ahead of it (a reach). Null/zero shows
// nothing rather than a bare "0". Every ADP source feeding this is PPR-native
// (see scoring/market.py), so the delta is a PPR delta.
function adpDelta(value: number | null): ReactNode {
  if (value === null || value === 0) return null
  const steal = value > 0
  return (
    <span className={`board-cell-adp ${steal ? 'is-steal' : 'is-reach'}`}>
      {steal ? `+${value.toFixed(0)}` : value.toFixed(0)}
    </span>
  )
}

interface DraftBoardGridProps {
  board: LiveBoard
  // Opens the pick's profile as a popup overlay over the room. Optional so
  // the grid still stands on its own without a room around it -- a cell
  // simply does nothing on click without a handler -- though DraftRoom
  // always passes it.
  onOpenPlayer?: (player: BoardPlayer) => void
}

// The main-zone canvas: rounds x teams, filling live from `board.cells`.
// Every cell trusts the server's own `round`/`slot` for where it lands
// rather than this component re-deriving a snake order from `overall` and
// the team count -- per the task brief, that keeps the grid correct
// regardless of this league's snake variant (standard, third-round
// reversal, etc). The one exception is the on-the-clock ring, which the
// brief hands over pre-computed as a (round, slot) pair for the same
// reason.
export default function DraftBoardGrid({ board, onOpenPlayer }: DraftBoardGridProps) {
  const [hover, setHover] = useState<HoverInfo | null>(null)

  // Any scroll -- the grid's own horizontal one, or the page's vertical one
  // -- invalidates the popover's captured `rect`, so drop it rather than
  // let it hang in place over the wrong cell. Capture phase: scroll events
  // don't bubble, so a listener on `window` only sees them this way.
  useEffect(() => {
    if (!hover) return
    function dismiss() {
      setHover(null)
    }
    window.addEventListener('scroll', dismiss, true)
    return () => window.removeEventListener('scroll', dismiss, true)
  }, [hover])

  if (!board.active) return null

  const { teams, rounds, columns, cells, on_the_clock, picks_made } = board
  const byCell = new Map<string, BoardCell>()
  for (const c of cells) byCell.set(`${c.round}-${c.slot}`, c)

  // See the goal brief: the next pick is the empty cell at this exact
  // (round, slot), not something derived by walking the snake ourselves.
  const clockRound = on_the_clock !== null ? Math.ceil((picks_made + 1) / teams) : null

  // An empty cell has no pick number from the server -- nothing has happened
  // there yet -- so the only way to label one is to walk the snake, which is
  // the derivation this component otherwise refuses to do. It does it here
  // under a guard rather than on faith: the numbers are the standard snake
  // (odd rounds left to right, even rounds back), which is the same rule the
  // round arrows beside them already draw, and every FILLED cell is checked
  // against it. One disagreement -- a third-round reversal, or any variant
  // this league runs -- and no empty cell gets a number at all. A board that
  // labels a pick wrong is worse than one that labels nothing: these are the
  // numbers someone plans two rounds ahead on.
  const overallAt = (round: number, colIndex: number) =>
    (round - 1) * teams + (round % 2 === 1 ? colIndex + 1 : teams - colIndex)
  const snakeHolds = cells.every((c) => {
    const colIndex = columns.findIndex((col) => col.slot === c.slot)
    return colIndex < 0 || overallAt(c.round, colIndex) === c.overall
  })

  function showPopover(cell: BoardCell, e: MouseEvent<HTMLButtonElement> | FocusEvent<HTMLButtonElement>) {
    setHover({ cell, rect: e.currentTarget.getBoundingClientRect() })
  }

  // A plain <button>, not a link -- there is nowhere left for a cell to
  // navigate to (the standalone /players/:slug page is gone), so there is no
  // href to protect from a swallowed modifier-click and no reason for this
  // to be an anchor any more. Opening the popup is the only thing a click
  // does.
  function handleCellClick(cell: BoardCell) {
    if (!onOpenPlayer) return
    // A pick the board has no row for -- see api/live.py's `identify_players`.
    // ESPN's rooms draft outside this board's pool (it drops anyone ESPN
    // itself does not rank), so such a pick is named, positioned and given a
    // face from the player tables, but there is no profile behind it: the
    // popup would open on "Failed to load player profile". `overall_rank` is
    // the board-row marker -- every player ON the board has one.
    if (cell.player.overall_rank === null) return
    // The hover popover is z-index 50 -- above the overlay's own backdrop --
    // so it would otherwise hang over the dimmed board with no way to
    // dismiss it (the pointer is about to leave without a mouseleave the
    // grid can see).
    setHover(null)
    onOpenPlayer(cell.player)
  }

  return (
    <div className="board-wrap">
      <div className="board-grid" style={{ gridTemplateColumns: `40px repeat(${teams}, minmax(0, 1fr))` }}>
        <div className="board-corner" />
        {columns.map((col) => (
          <div
            key={col.slot}
            className={[
              'board-col-header',
              col.is_me ? 'board-col-mine' : '',
              // Only ever set on a labelled board: `had_owner === false` is
              // "the farm watched this seat and nobody was ever in it",
              // which is worth knocking the header back for. Undefined
              // (the live room) and null (an unlabelled recording) both
              // leave the header exactly as it was.
              col.had_owner === false ? 'board-col-vacant' : '',
            ].filter(Boolean).join(' ')}
          >
            <span className="board-col-name">{col.team_name}</span>
            {/* Whether a person ever sat here is said, not drawn: the
                column's own tiles carry it as light (see App.css), and a
                dot in every header was one more mark on a board that had
                too many. */}
            {col.had_owner !== undefined && col.had_owner !== null && (
              <span className="sr-only">
                {col.had_owner ? 'a person sat in this seat' : 'nobody ever sat in this seat'}
              </span>
            )}
            {col.is_me && <span className="board-you-chip">YOU</span>}
          </div>
        ))}

        {Array.from({ length: rounds }, (_, i) => i + 1).map((round) => {
          const leftToRight = round % 2 === 1
          const rowClass = round % 2 === 0 ? ' board-row-even' : ''
          return (
          <Fragment key={round}>
            <div className={`board-round-label${rowClass}`}>
              <span className="board-round-n mono">{round}</span>
              <span className="board-snake-dir" aria-hidden="true">{leftToRight ? '→' : '←'}</span>
            </div>
            {columns.map((col, colIndex) => {
              const cell = byCell.get(`${round}-${col.slot}`)
              if (!cell) {
                const isClock = clockRound === round && on_the_clock === col.slot
                const overall = snakeHolds ? overallAt(round, colIndex) : null
                return (
                  <div
                    key={col.slot}
                    className={[
                      'board-cell board-cell-empty',
                      col.is_me ? 'board-cell-mine' : '',
                      isClock ? 'board-cell-clock' : '',
                      rowClass.trim(),
                    ].filter(Boolean).join(' ')}
                  >
                    {overall !== null && (
                      // Quiet: an empty cell is a slot, and the number is
                      // there to be counted to rather than read. Brighter in
                      // your own column, which is the only one anybody counts.
                      <span className="board-cell-pick mono">{overall}</span>
                    )}
                  </div>
                )
              }
              const pickInRound = cell.overall - (cell.round - 1) * teams
              return (
                <button
                  key={col.slot}
                  type="button"
                  className={['board-cell board-cell-filled',
                    // The cell's own position hue, for the tint and the left
                    // edge. See `.board-cell-pos-*`.
                    `board-cell-pos-${(cell.player.position || 'na').toLowerCase()}`,
                    col.is_me ? 'board-cell-mine' : '',
                    // Named, but with no board row behind it, so there is no
                    // profile to open (see handleCellClick). The class takes
                    // the pointer's promise back off it.
                    cell.player.overall_rank === null ? 'board-cell-flat' : '',
                    // Absent on the live room's board (no label, no class),
                    // so /draft keeps drawing exactly the cell it always has.
                    isLabelled(cell.made_by) ? `board-cell-by-${cell.made_by}` : '',
                    rowClass.trim()]
                    .filter(Boolean).join(' ')}
                  aria-label={`${cell.player.name}, ${cell.player.position || 'unknown position'}, pick ${cell.round}.${pickInRound}`
                    + (isLabelled(cell.made_by) ? `, ${MAKER_LABEL[cell.made_by]}` : '')}
                  title={isLabelled(cell.made_by) ? MAKER_LABEL[cell.made_by] : undefined}
                  onClick={() => handleCellClick(cell)}
                  onMouseEnter={(e) => showPopover(cell, e)}
                  onMouseLeave={() => setHover(null)}
                  onFocus={(e) => showPopover(cell, e)}
                  onBlur={() => setHover(null)}
                >
                  <div className="board-cell-top">
                    {posBadge(cell.player.position)}
                    <span className="board-cell-meta">
                      {adpDelta(cell.player.value)}
                      <span className="board-cell-team mono">{cell.player.team ?? ''}</span>
                    </span>
                  </div>
                  <div className="board-cell-namewrap">
                    <span className="board-cell-name">{cell.player.name}</span>
                  </div>
                </button>
              )
            })}
          </Fragment>
          )
        })}
      </div>

      {/* A CONDENSED PROFILE, not a list of fields. What was here read
          "#41 overall · Tier 4 · VOR 12.3" -- internal quantities a drafter
          has no feel for, and none of them the question somebody hovering a
          pick is asking. See BoardPeek. */}
      {/* Portalled to the body: the peek is `position: fixed` in viewport
          coordinates, but any ancestor that masks, transforms or scrolls
          (the room's main column, the landing page's faded demo) would
          otherwise clip it at its own edge -- which is exactly where the
          rightmost column's peeks land. */}
      {hover && createPortal(
        <BoardPeek player={hover.cell.player} pick={hover.cell.overall}
                   style={popoverStyle(hover.rect)} />,
        document.body,
      )}
    </div>
  )
}
