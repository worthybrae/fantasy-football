import { useMemo, useState } from 'react'
import type { SimBoard, SimBoardCell } from '../api'

interface DraftGridProps {
  board: SimBoard
  onSelectPlayer: (playerId: string) => void
}

/** Cells keyed by overall pick, primary first. */
function byPick(cells: SimBoardCell[]): Map<number, SimBoardCell[]> {
  const map = new Map<number, SimBoardCell[]>()
  for (const c of cells) {
    const list = map.get(c.overall_pick) ?? []
    list.push(c)
    map.set(c.overall_pick, list)
  }
  for (const list of map.values()) list.sort((a, b) => a.alt_rank - b.alt_rank)
  return map
}

/** Overall pick number for a (round, slot) in a snake draft. Round 1 runs
 *  slot 1..teams, round 2 runs teams..1, and so on. */
function pickNumber(round: number, slot: number, teams: number): number {
  const offsetInRound = round % 2 === 1 ? slot - 1 : teams - slot
  return (round - 1) * teams + offsetInRound + 1
}

export default function DraftGrid({ board, onSelectPlayer }: DraftGridProps) {
  const cells = useMemo(() => byPick(board.cells), [board.cells])
  const [hovered, setHovered] = useState<number | null>(null)
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
                const pick = pickNumber(round, e.slot, board.teams)
                const list = cells.get(pick) ?? []
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
                    onMouseEnter={() => setHovered(pick)}
                    onMouseLeave={() => setHovered((h) => (h === pick ? null : h))}
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
                    {hovered === pick && alts.length > 0 && (
                      <div className="grid-alts" role="tooltip">
                        <div className="grid-alts-title">Also in play</div>
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
