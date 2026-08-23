import type { ReactNode } from 'react'
import type { BoardCell, BoardPlayer, LiveBoard } from '../../api'

// How far past his ADP a pick landed: >0 he FELL that many slots (a steal),
// <0 he went that many early (a reach). Same number, same words and the same
// two colours the snake board's cells already use for it -- `value` is
// computed once in api/live.py and both views read it, so the rail and the
// grid cannot disagree about whether a pick was a bargain.
//
// Nothing at all when there is no ADP to compare against, or when a pick
// landed exactly on it: a bare "0" is a fact about arithmetic rather than
// about the draft.
function adpDelta(value: number | null | undefined): ReactNode {
  if (value === null || value === undefined || Math.round(value) === 0) return null
  const steal = value > 0
  return (
    <span className={`pick-ticker-adp board-cell-adp ${steal ? 'is-steal' : 'is-reach'}`}
          title={steal ? `Fell ${Math.round(value)} picks past his ADP`
            : `Taken ${Math.abs(Math.round(value))} picks early`}>
      {steal ? `+${Math.round(value)}` : Math.round(value)}
    </span>
  )
}

// duplicated from DraftBoardGrid.tsx / RosterPanel.tsx (unexported in both):
// a four-line pure function isn't worth a shared module between four views.
function posBadge(position: string | undefined): ReactNode {
  if (!position) return null
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

// How many landed picks the strip carries. Eight was what fit on one row
// without the names truncating; the rail scrolls now, so the cap is a
// reading limit rather than a width one. Twenty-four is three rounds of an
// eight-team league -- far enough back to answer "who went while I was
// reading this profile" without turning the rail into the draft history the
// snake board above already is.
const TICKER_LENGTH = 24

interface PickTickerProps {
  board: LiveBoard | null
  // Opens the pick's profile over the room, same handler the snake board's
  // cells use. Optional for the same reason DraftBoardGrid's is: the strip
  // still renders (as plain text, not buttons) without a room around it.
  onOpenPlayer?: (player: BoardPlayer) => void
}

// The room's bottom rail: the last eight picks the league made, newest at
// the left. Deliberately NOT a marquee -- nothing here scrolls or animates
// on its own. It re-renders when DraftRoom's existing 2.5s /api/live/board
// poll lands a new pick and is otherwise completely still, because it sits
// directly under the ranked list a user is reading under a 30-second clock
// and motion in the corner of that eye is a cost, not a feature.
//
// `board.cells` is the source rather than a new endpoint: DraftRoom already
// polls it for the snake board, every cell already carries its resolved
// player and slot, and `overall` is a total order over the picks that have
// actually landed -- so "the last eight" is a sort, not a fetch.
export default function PickTicker({ board, onOpenPlayer }: PickTickerProps) {
  if (!board?.active) return null

  const recent: BoardCell[] = [...board.cells]
    .sort((a, b) => b.overall - a.overall)
    .slice(0, TICKER_LENGTH)

  // Column headers keyed by slot, so each pick can name the team that made
  // it. The board serves exactly one column per slot, but a missing entry
  // falls back to the slot number rather than rendering "undefined".
  const teamBySlot = new Map(board.columns.map((c) => [c.slot, c]))

  return (
    <footer className="pick-ticker" aria-label="Last drafted">
      <span className="pick-ticker-label draft-cap">Last drafted</span>
      {recent.length === 0 ? (
        <span className="pick-ticker-empty">No picks yet</span>
      ) : (
        <ol className="pick-ticker-list">
          {recent.map((cell, i) => {
            const column = teamBySlot.get(cell.slot)
            const team = column?.team_name ?? `Team ${cell.slot}`
            const label = `Pick ${cell.overall}, ${cell.player.name}, ${team}`
            return (
              <li
                key={cell.overall}
                // The freshest pick only. A static tint, not a flash: the
                // point is "this one is new", which a colour states just as
                // well as a transition does and without moving anything.
                className={`pick-ticker-item${i === 0 ? ' is-newest' : ''}${
                  column?.is_me ? ' is-mine' : ''}`}
              >
                {onOpenPlayer ? (
                  <button
                    type="button"
                    className="pick-ticker-pick"
                    onClick={() => onOpenPlayer(cell.player)}
                    aria-label={label}
                  >
                    {cell.player.headshot && (
                      // `alt=""` -- the name is right beside it and the
                      // button already carries the whole pick as its label,
                      // so a described image would be the third telling.
                      <img className="pick-ticker-face" src={cell.player.headshot}
                           alt="" width={30} height={30} loading="lazy" />
                    )}
                    <span className="pick-ticker-no mono">{cell.overall}</span>
                    {posBadge(cell.player.position)}
                    {/* Stacked, not in a row. Sharing one line, the name and
                        the team each got half the width and both ellipsised
                        -- "Bijan R..." next to "Tristan's Tal..." names
                        neither the player nor the manager. Two lines give
                        each of them the whole entry to use. */}
                    <span className="pick-ticker-who">
                      <span className="pick-ticker-name">{cell.player.name}</span>
                      <span className="pick-ticker-team">{team}</span>
                    </span>
                    {adpDelta(cell.player.value)}
                  </button>
                ) : (
                  <span className="pick-ticker-pick" aria-label={label}>
                    {cell.player.headshot && (
                      // `alt=""` -- the name is right beside it and the
                      // button already carries the whole pick as its label,
                      // so a described image would be the third telling.
                      <img className="pick-ticker-face" src={cell.player.headshot}
                           alt="" width={30} height={30} loading="lazy" />
                    )}
                    <span className="pick-ticker-no mono">{cell.overall}</span>
                    {posBadge(cell.player.position)}
                    {/* Stacked, not in a row. Sharing one line, the name and
                        the team each got half the width and both ellipsised
                        -- "Bijan R..." next to "Tristan's Tal..." names
                        neither the player nor the manager. Two lines give
                        each of them the whole entry to use. */}
                    <span className="pick-ticker-who">
                      <span className="pick-ticker-name">{cell.player.name}</span>
                      <span className="pick-ticker-team">{team}</span>
                    </span>
                    {adpDelta(cell.player.value)}
                  </span>
                )}
              </li>
            )
          })}
        </ol>
      )}
    </footer>
  )
}
