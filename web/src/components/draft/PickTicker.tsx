import type { ReactNode } from 'react'
import type { BoardCell, BoardPlayer, LiveBoard } from '../../api'

// The one number in this strip worth raising a voice about.
//
// `value` is where the pick landed against the market: >0 he FELL that many
// slots, <0 somebody jumped him. Computed once in api/live.py, so the rail and
// the snake board's cells cannot disagree about whether a pick was a bargain.
//
// UNDER TEN SLOTS IT DRAWS NOTHING. Most picks land about where the market
// said, and a strip that marks every one of them is a strip that marks
// nothing -- six entries of coloured numbers read as an alarm rather than as
// news, which is exactly what it did. Ten either way is the move you would
// say out loud at a real table.
//
// Past sixty the magnitude has stopped meaning anything -- a rookie
// quarterback whose consensus rank is near 500 comes back as -215 -- so it
// clamps and the arrow carries it. One number like that in the row otherwise
// sets the width and the volume of every entry beside it.
const CRAZY_SLOTS = 10
const CLAMP_SLOTS = 60

function adpDelta(value: number | null | undefined): ReactNode {
  if (value === null || value === undefined) return null
  const slots = Math.round(value)
  if (Math.abs(slots) < CRAZY_SLOTS) return null
  const steal = slots > 0
  const size = Math.abs(slots)
  const shown = size <= CLAMP_SLOTS ? String(size) : `${CLAMP_SLOTS}+`
  return (
    <span
      className={`pick-ticker-adp ${steal ? 'is-steal' : 'is-reach'}`}
      title={steal ? `Fell ${shown} picks past his ADP`
        : `Taken ${shown} picks before his ADP`}
    >
      {/* The glyph is the direction and the number is the size, in that
          order, because at the edge of an eye reading a ranked list the
          direction is the whole message. */}
      <span aria-hidden="true">{steal ? '\u25b2' : '\u25bc'}</span>
      {shown}
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
                    {/* The first line is the name and nothing else, so a
                        scan down the rail reads names. Everything that
                        qualifies it -- which pick, which position, whose
                        team, and how far off the market it landed -- sits on
                        a quieter second line underneath. */}
                    <span className="pick-ticker-who">
                      <span className="pick-ticker-name">{cell.player.name}</span>
                      <span className="pick-ticker-detail">
                        <span className="mono pick-ticker-no">{cell.overall}</span>
                        {posBadge(cell.player.position)}
                        <span className="pick-ticker-team">{team}</span>
                        {adpDelta(cell.player.value)}
                      </span>
                    </span>
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
                    {/* The first line is the name and nothing else, so a
                        scan down the rail reads names. Everything that
                        qualifies it -- which pick, which position, whose
                        team, and how far off the market it landed -- sits on
                        a quieter second line underneath. */}
                    <span className="pick-ticker-who">
                      <span className="pick-ticker-name">{cell.player.name}</span>
                      <span className="pick-ticker-detail">
                        <span className="mono pick-ticker-no">{cell.overall}</span>
                        {posBadge(cell.player.position)}
                        <span className="pick-ticker-team">{team}</span>
                        {adpDelta(cell.player.value)}
                      </span>
                    </span>
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
