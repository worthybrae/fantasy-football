import { Fragment, type ReactNode } from 'react'
import type { BoardCell, BoardPlayer, LiveBoard } from '../../api'

// Where the pick landed against the market: >0 he FELL that many slots, <0
// somebody jumped him. `value` is computed once in api/live.py, so the rail
// and the snake board's cells cannot disagree about whether a pick was a
// bargain.
//
// ALWAYS drawn -- direction, size, colour -- because "about where the market
// said" is itself worth seeing on every pick. Nothing only when there is no
// ADP to compare against, or when a pick landed exactly on it: a bare zero is
// a fact about arithmetic rather than about the draft.
function adpDelta(value: number | null | undefined): ReactNode {
  if (value === null || value === undefined) return null
  const slots = Math.round(value)
  if (slots === 0) return null
  const steal = slots > 0
  return (
    <span
      className={`pick-ticker-adp ${steal ? 'is-steal' : 'is-reach'}`}
      title={steal ? `Fell ${Math.abs(slots)} picks past his ADP`
        : `Taken ${Math.abs(slots)} picks before his ADP`}
    >
      {/* Direction then size: at the edge of an eye reading a ranked list,
          which way is the message and how far is the detail. */}
      <span aria-hidden="true">{steal ? '\u25b2' : '\u25bc'}</span>
      {Math.abs(slots)}
    </span>
  )
}

// duplicated from DraftBoardGrid.tsx / RosterPanel.tsx (unexported in both):
// a four-line pure function isn't worth a shared module between four views.
//
// `rank` is how many at that position had gone by the time this one did --
// the 7 in QB7. It rides inside the tag rather than beside it because that
// is the unit people say out loud, and because a loose number on this line
// would read as another stat next to the pick number and the ADP move.
function posBadge(position: string | undefined, rank: number | undefined): ReactNode {
  if (!position) return null
  return (
    <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>
      {position}
      {rank !== undefined && <span className="pos-badge-rank">{rank}</span>}
    </span>
  )
}

// EVERY pick, not the last few. The strip started at eight because it could
// not scroll, and stayed capped at twenty-four out of habit once it could --
// but "who went in the third round" is a question this rail can answer for
// free, and the alternative is the snake board, which costs a tab switch and
// a scroll. Draft night is 120-odd picks in this league; the images are lazy
// and the rows are cheap.

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

  const recent: BoardCell[] = [...board.cells].sort((a, b) => b.overall - a.overall)

  // QB7: the seventh quarterback off the board, keyed by the pick that was
  // him. Counted here off the same `board.cells` the rail already has --
  // walking it oldest-first -- rather than asked of the API, because it is a
  // running total over picks that have landed and nothing else knows it.
  const posRank = new Map<number, number>()
  const takenAtPos = new Map<string, number>()
  for (let i = recent.length - 1; i >= 0; i--) {
    const pos = recent[i].player.position
    if (!pos) continue
    const n = (takenAtPos.get(pos) ?? 0) + 1
    takenAtPos.set(pos, n)
    posRank.set(recent[i].overall, n)
  }

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
            // Newest first, so the rounds run backwards -- a marker goes in
            // front of the first pick of each one, which reading right to
            // left is where that round began.
            const startsRound = i === 0 || recent[i - 1].round !== cell.round
            const column = teamBySlot.get(cell.slot)
            const team = column?.team_name ?? `Team ${cell.slot}`
            const rank = posRank.get(cell.overall)
            // The tag is decoration to a screen reader elsewhere on this
            // line, so the spoken label is where "QB7" has to be said.
            const posSaid = cell.player.position
              ? `${cell.player.position}${rank ?? ''}, `
              : ''
            const label = `Pick ${cell.overall}, ${cell.player.name}, ${posSaid}${team}`
            return (
              <Fragment key={cell.overall}>
              {startsRound && (
                <li className="pick-ticker-round" aria-hidden="true">
                  <span>R{cell.round}</span>
                </li>
              )}
              <li
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
                      {/* The move sits with the name, because they are the
                          two things a reader is here for: who went, and
                          whether it was a bargain. The line under it is the
                          bookkeeping -- which pick, which position, whose
                          team. */}
                      <span className="pick-ticker-nameline">
                        {/* The pick number leads the line it belongs to: it is
                            what the entry IS -- pick 102 -- and reading it
                            before the name matches how anyone says it out
                            loud. */}
                        <span className="mono pick-ticker-no">{cell.overall}</span>
                        <span className="pick-ticker-name">{cell.player.name}</span>
                        {adpDelta(cell.player.value)}
                      </span>
                      <span className="pick-ticker-detail">
                        {posBadge(cell.player.position, posRank.get(cell.overall))}
                        <span className="pick-ticker-team">{team}</span>
                      </span>
                    </span>
                  </button>
                ) : (
                  <span
                    className="pick-ticker-pick"
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
                      {/* The move sits with the name, because they are the
                          two things a reader is here for: who went, and
                          whether it was a bargain. The line under it is the
                          bookkeeping -- which pick, which position, whose
                          team. */}
                      <span className="pick-ticker-nameline">
                        {/* The pick number leads the line it belongs to: it is
                            what the entry IS -- pick 102 -- and reading it
                            before the name matches how anyone says it out
                            loud. */}
                        <span className="mono pick-ticker-no">{cell.overall}</span>
                        <span className="pick-ticker-name">{cell.player.name}</span>
                        {adpDelta(cell.player.value)}
                      </span>
                      <span className="pick-ticker-detail">
                        {posBadge(cell.player.position, posRank.get(cell.overall))}
                        <span className="pick-ticker-team">{team}</span>
                      </span>
                    </span>
                  </span>
                )}
              </li>
              </Fragment>
            )
          })}
        </ol>
      )}
    </footer>
  )
}
