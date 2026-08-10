import { useMemo, useState } from 'react'
import type { SimBoard, SimBoardCell } from '../api'

interface DraftGridProps {
  board: SimBoard
  onSelectPlayer: (playerId: string) => void
}

/** Cells keyed by the grid coordinate they belong in, primary first.
 *
 *  Keyed on the payload's own `round` and `slot` rather than on a snake
 *  mapping re-derived here from (round, slot, teams). `_board_row` already
 *  derives round/round_pick/slot from `overall_pick` server-side and sends
 *  all three on every cell, so re-deriving the inverse was a second
 *  implementation of the same rule with nothing to keep the two in step --
 *  and the only way for them to ever disagree. It also handles the one input
 *  where they genuinely do: `board.order` carrying a slot outside 1..teams
 *  (nothing validates that -- PUT /api/draft-order checks uniqueness only,
 *  and run_sim's guard rejects missing slots but permits extras) used to
 *  alias onto another cell's pick and render that player twice. Keying on
 *  the cell's own slot leaves the column empty instead. */
function cellKey(round: number, slot: number): string {
  return `${round}|${slot}`
}

function byCell(cells: SimBoardCell[]): Map<string, SimBoardCell[]> {
  const map = new Map<string, SimBoardCell[]>()
  for (const c of cells) {
    const key = cellKey(c.round, c.slot)
    const list = map.get(key) ?? []
    list.push(c)
    map.set(key, list)
  }
  for (const list of map.values()) list.sort((a, b) => a.alt_rank - b.alt_rank)
  return map
}

export default function DraftGrid({ board, onSelectPlayer }: DraftGridProps) {
  const cells = useMemo(() => byCell(board.cells), [board.cells])
  const [hovered, setHovered] = useState<string | null>(null)
  const rounds = Array.from({ length: board.rounds }, (_, i) => i + 1)
  const mySlot = board.run?.my_slot ?? null

  return (
    <div className="grid-wrap">
      <table className="draft-grid">
        <thead>
          <tr>
            <th className="grid-round">Round</th>
            <th className="grid-dir" aria-label="Pick direction" />
            {board.order.map((e) => (
              <th key={e.slot} className={e.slot === mySlot ? 'grid-mine' : undefined}>
                {e.manager}
                {e.slot === mySlot && <span className="grid-you"> (you)</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rounds.map((round) => (
            <tr key={round}>
              <td className="grid-round">{round}</td>
              <td className="grid-dir">{round % 2 === 1 ? '›' : '‹'}</td>
              {board.order.map((e) => {
                const key = cellKey(round, e.slot)
                const list = cells.get(key) ?? []
                const primary = list[0]
                const alts = list.slice(1)
                const mine = e.slot === mySlot
                if (!primary) {
                  return <td key={e.slot} className="grid-cell grid-empty" />
                }
                const state = primary.certain ? 'is-certain' : mine ? 'is-projected' : 'is-predicted'
                const pos = (primary.position ?? '').toLowerCase()
                return (
                  <td
                    key={e.slot}
                    className={`grid-cell ${state}`}
                    style={pos ? { borderLeftColor: `var(--pos-${pos})` } : undefined}
                    onMouseEnter={() => setHovered(key)}
                    onMouseLeave={() => setHovered((h) => (h === key ? null : h))}
                  >
                    <button
                      type="button"
                      className="grid-pick"
                      onClick={() => onSelectPlayer(primary.player_id)}
                    >
                      <span className="grid-name">{primary.name}</span>
                      <span className="grid-meta">
                        {primary.team ?? '—'}
                        {primary.position ? `, ${primary.position}` : ''}
                        {/* A projected cell is what a greedy plan does, not a
                            forecast -- showing its near-1.0 frequency as a
                            probability would read as confidence it hasn't got. */}
                        {!primary.certain && !mine && (
                          <span className="grid-prob">{Math.round(primary.prob * 100)}%</span>
                        )}
                      </span>
                    </button>
                    {hovered === key && alts.length > 0 && (
                      <div className="grid-alts" role="tooltip">
                        {/* Not "Also in play": these are this pick's raw,
                            un-deduped frequencies, and saying so is what
                            makes an alternate outranking the primary read as
                            the dedupe rather than as a bug. */}
                        <div className="grid-alts-title">Raw odds for this pick</div>
                        {alts.map((a) => (
                          <button
                            key={a.player_id}
                            type="button"
                            className="grid-alt"
                            onClick={() => onSelectPlayer(a.player_id)}
                          >
                            <span>{a.name}</span>
                            <span className="grid-prob">{Math.round(a.prob * 100)}%</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
