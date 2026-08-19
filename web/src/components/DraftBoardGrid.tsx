import { Fragment, useEffect, useState, type CSSProperties, type FocusEvent, type MouseEvent, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { playerSlug, type BoardCell, type BoardPlayer, type LiveBoard } from '../api'

// duplicated from LiveDraft.tsx (unexported there) -- same precedent as
// PlayerCard.tsx's depthSlotLabel/sosLabel: a four-line pure function isn't
// worth a shared module between the board's two views.
function posBadge(position: string | undefined): ReactNode {
  if (!position) return null
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

interface HoverInfo {
  cell: BoardCell
  rect: DOMRect
}

const POPOVER_WIDTH = 220
const POPOVER_GAP = 8
// Rough popover height -- there's no real box to measure until it's already
// painted, and this only has to be close enough to decide which side of the
// cell has room, not pixel-exact.
const POPOVER_EST_HEIGHT = 190

function popoverStyle(rect: DOMRect): CSSProperties {
  const left = Math.max(8, Math.min(rect.left, window.innerWidth - POPOVER_WIDTH - 8))
  const below = rect.bottom + POPOVER_GAP
  const top = below + POPOVER_EST_HEIGHT <= window.innerHeight
    ? below
    : Math.max(8, rect.top - POPOVER_EST_HEIGHT - POPOVER_GAP)
  return { left, top }
}

// value > 0: fell past ADP (a steal). value < 0: went early (a reach).
// value === 0 is neither -- it's dropped rather than tagged, same as every
// other popover field being skipped when it has nothing to say.
function valueTag(value: number | null): { label: string; tone: 'steal' | 'reach' } | null {
  if (value === null || value === 0) return null
  return value > 0
    ? { label: `steal +${value.toFixed(0)}`, tone: 'steal' }
    : { label: `reach ${Math.abs(value).toFixed(0)}`, tone: 'reach' }
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

// The hover/focus detail card: everything ESPN-clean cell content leaves
// out. Every row is conditional on its own field being non-null -- a player
// with no market coverage (no ADP, no edge) still gets a popover, just a
// shorter one.
function BoardPopover({ player, style }: { player: BoardPlayer; style: CSSProperties }) {
  const tag = valueTag(player.value)
  return (
    <div className="board-pop" style={style} role="tooltip">
      <div className="board-pop-name">{player.name}</div>
      <div className="board-pop-meta">
        {posBadge(player.position)}
        <span>{player.team ?? '—'}</span>
        {player.bye !== null && <span>Bye {player.bye}</span>}
      </div>
      <ul className="board-pop-stats">
        {player.overall_rank !== null && <li>#{player.overall_rank} overall</li>}
        {player.tier !== null && <li>Tier {player.tier}</li>}
        {player.market_rank !== null && <li>ADP {player.market_rank.toFixed(1)}</li>}
        {tag && <li className={`board-pop-value is-${tag.tone}`}>{tag.label}</li>}
        {player.vor !== null && <li>VOR {player.vor.toFixed(1)}</li>}
        {player.last_ppg !== null && <li>{player.last_ppg.toFixed(1)} ppg last yr</li>}
      </ul>
    </div>
  )
}

interface DraftBoardGridProps {
  board: LiveBoard
  // Opens the pick's profile over the room instead of navigating to
  // /players/:slug. Optional so the grid still stands on its own (and still
  // navigates) without a room around it; DraftRoom always passes it.
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

  function showPopover(cell: BoardCell, e: MouseEvent<HTMLAnchorElement> | FocusEvent<HTMLAnchorElement>) {
    setHover({ cell, rect: e.currentTarget.getBoundingClientRect() })
  }

  // A cell stays a real <a href="/players/:slug"> and this intercepts the
  // plain left click: the overlay is what the room wants (it keeps the
  // board, the tab and the clock on screen), but cmd/ctrl-click, middle
  // click, shift-click and "open in new tab" all still reach the standalone
  // page, and the browser still shows the target in the status bar. A
  // <button> here would have thrown all of that away to gain nothing.
  //
  // Modifier check before preventDefault, never after: swallowing a
  // cmd-click would silently break new-tab, which is the one thing the href
  // is still here for. `e.button !== 0` covers the middle-click that arrives
  // as a click event in some browsers.
  function handleCellClick(cell: BoardCell, e: MouseEvent<HTMLAnchorElement>) {
    if (!onOpenPlayer) return
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return
    e.preventDefault()
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
          <div key={col.slot} className={`board-col-header${col.is_me ? ' board-col-mine' : ''}`}>
            <span className="board-col-name">{col.team_name}</span>
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
            {columns.map((col) => {
              const cell = byCell.get(`${round}-${col.slot}`)
              if (!cell) {
                const isClock = clockRound === round && on_the_clock === col.slot
                return (
                  <div
                    key={col.slot}
                    className={[
                      'board-cell board-cell-empty',
                      col.is_me ? 'board-cell-mine' : '',
                      isClock ? 'board-cell-clock' : '',
                      rowClass.trim(),
                    ].filter(Boolean).join(' ')}
                  />
                )
              }
              const pickInRound = cell.overall - (cell.round - 1) * teams
              return (
                <Link
                  key={col.slot}
                  to={`/players/${playerSlug(cell.player.name)}`}
                  className={['board-cell board-cell-filled',
                    col.is_me ? 'board-cell-mine' : '', rowClass.trim()]
                    .filter(Boolean).join(' ')}
                  aria-label={`${cell.player.name}, ${cell.player.position}, pick ${cell.round}.${pickInRound}`}
                  onClick={(e) => handleCellClick(cell, e)}
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
                </Link>
              )
            })}
          </Fragment>
          )
        })}
      </div>

      {hover && <BoardPopover player={hover.cell.player} style={popoverStyle(hover.rect)} />}
    </div>
  )
}
