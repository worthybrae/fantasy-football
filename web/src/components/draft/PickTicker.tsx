import { Fragment, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import type { BoardCell, BoardPlayer, LiveBoard } from '../../api'
import { positionTip, TIP_DELAY_MS } from './AvailableList'

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
  // No `title`: the whole entry carries a hover panel (see `verdict` and
  // `.pick-ticker-tip` below) that says the same thing with the two numbers
  // it came from.
  return (
    <span className={`pick-ticker-adp ${steal ? 'is-steal' : 'is-reach'}`}>
      {/* Direction then size: at the edge of an eye reading a ranked list,
          which way is the message and how far is the detail. */}
      <span aria-hidden="true">{steal ? '\u25b2' : '\u25bc'}</span>
      {Math.abs(slots)}
    </span>
  )
}

// What the move against ADP MEANS, in a word, scaled to the room. A pick
// that fell a whole round past its ADP is a steal in any room; the same six
// slots are a round in a six-team draft and half a round in a twelve, so the
// line between "value" and "steal" is the room's own round length rather
// than a fixed number. Within two picks either way is the market itself:
// rankings differ by that much between two sites.
//
// Tone is the pair the inline arrow uses (`is-steal` green, `is-reach` red),
// so the panel never argues with the glyph it explains.
type Verdict = { head: string; note: string; tone: 'is-steal' | 'is-reach' | 'is-flat' }

export function verdict(value: number | null | undefined, teams: number): Verdict {
  if (value === null || value === undefined) {
    return { head: 'No ADP', note: 'no consensus draft position to compare', tone: 'is-flat' }
  }
  const slots = Math.round(value)
  const size = Math.abs(slots)
  const round = Math.max(2, teams)
  if (size <= 2) {
    return { head: 'On the market', note: 'within 2 picks of his ADP', tone: 'is-flat' }
  }
  if (slots > 0) {
    return slots >= round
      ? { head: 'Steal', note: `fell ${size} picks past his ADP`, tone: 'is-steal' }
      : { head: 'Value', note: `${size} picks past his ADP`, tone: 'is-steal' }
  }
  return size >= round
    ? { head: 'Reach', note: `taken ${size} picks before his ADP`, tone: 'is-reach' }
    : { head: 'Early', note: `${size} picks before his ADP`, tone: 'is-reach' }
}

// "64" for a whole number, "64.3" otherwise: the consensus is an average and
// prints as one, but a round one need not carry a decimal to prove it.
function fmtAdp(adp: number): string {
  return Number.isInteger(adp) ? String(adp) : adp.toFixed(1)
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
  // The hover panel: which pick, and where its entry sits on screen. Fixed-
  // positioned through `positionTip` for the reason the available list's
  // tips are -- the strip scrolls sideways and clips anything inside it --
  // and delayed the same TIP_DELAY_MS, so a pointer crossing the rail on its
  // way somewhere else does not flash a panel per pick. Opened by hover and
  // by keyboard focus alike: every entry is a button.
  const [tip, setTip] = useState<{ overall: number; rect: DOMRect } | null>(null)
  const [tipPos, setTipPos] = useState<{ left: number; top: number } | null>(null)
  const tipRef = useRef<HTMLDivElement | null>(null)
  const tipTimer = useRef<number | null>(null)

  function showTip(overall: number, target: HTMLElement): void {
    if (tipTimer.current !== null) window.clearTimeout(tipTimer.current)
    const rect = target.getBoundingClientRect()
    tipTimer.current = window.setTimeout(() => setTip({ overall, rect }), TIP_DELAY_MS)
  }

  function hideTip(): void {
    if (tipTimer.current !== null) window.clearTimeout(tipTimer.current)
    tipTimer.current = null
    setTip(null)
    setTipPos(null)
  }

  // Measured after the panel has rendered, so the placement knows its size;
  // hidden until then (see the `visibility` style) so it never paints at
  // 0,0 for a frame.
  useLayoutEffect(() => {
    if (!tip || !tipRef.current) {
      setTipPos(null)
      return
    }
    const { width, height } = tipRef.current.getBoundingClientRect()
    setTipPos(positionTip(tip.rect, { width, height }))
  }, [tip])

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
                onMouseEnter={(e) => showTip(cell.overall, e.currentTarget)}
                onMouseLeave={hideTip}
              >
                {onOpenPlayer ? (
                  <button
                    type="button"
                    className="pick-ticker-pick"
                    onClick={() => onOpenPlayer(cell.player)}
                    onFocus={(e) => showTip(cell.overall, e.currentTarget)}
                    onBlur={hideTip}
                    aria-label={label}
                    aria-describedby={tip?.overall === cell.overall ? 'pick-ticker-tip' : undefined}
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
      {tip && (() => {
        const cell = recent.find((c) => c.overall === tip.overall)
        if (!cell) return null
        const v = verdict(cell.player.value, board.teams)
        const adp = cell.player.market_rank
        const rank = posRank.get(cell.overall)
        return (
          <div
            ref={tipRef}
            id="pick-ticker-tip"
            role="tooltip"
            className="pick-ticker-tip"
            style={tipPos
              ? { left: tipPos.left, top: tipPos.top, visibility: 'visible' }
              : { left: 0, top: 0, visibility: 'hidden' }}
          >
            {/* The verdict first, in the arrow's own colour: it is the
                answer. Then the two numbers it was computed from, side by
                side -- where he went and where the market had him -- which
                is the comparison a reader hovers to see. */}
            <div className={`pick-ticker-tip-head ${v.tone}`}>
              {v.head}
              <span className="pick-ticker-tip-note"> · {v.note}</span>
            </div>
            <div className="mono pick-ticker-tip-line">
              Pick {cell.overall}
              <span className="pick-ticker-tip-dim"> · R{cell.round}, seat {cell.slot}</span>
              {adp !== null && adp !== undefined && (
                <>
                  <span className="pick-ticker-tip-dim"> — </span>
                  ADP {fmtAdp(adp)}
                </>
              )}
            </div>
            {/* No team here: the entry under the pointer already says whose
                pick it was, and the seat is on the line above. */}
            {cell.player.position && rank !== undefined && (
              <div className="pick-ticker-tip-line pick-ticker-tip-dim">
                {cell.player.position}{rank} off the board
              </div>
            )}
          </div>
        )
      })()}
    </footer>
  )
}
