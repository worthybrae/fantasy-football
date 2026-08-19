import type { ReactNode } from 'react'
import type { RosterPlayer } from '../../api'

// duplicated from LiveDraft.tsx / DraftBoardGrid.tsx (unexported in both) --
// same precedent as PlayerCard.tsx's depthSlotLabel/sosLabel: a four-line
// pure function isn't worth a shared module between three views.
function posBadge(position: string | undefined): ReactNode {
  if (!position) return null
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

// `player` is `RosterPlayer` (api/live.py's my_roster shape), not the
// board's full `Player` -- DraftRoom builds this list straight off
// /api/live/state now, and my_roster carries only player_id/name/position/
// proj_points, not the whole board row.
export type RosterSlot = { slot: string; player: RosterPlayer | null; urgent: boolean }

export default function RosterPanel({ slots }: { slots: RosterSlot[] }) {
  const filled = slots.filter((s) => s.player !== null).length

  return (
    <div className="roster-panel">
      <div className="roster-panel-head">
        <span className="draft-cap">My roster</span>
        <span className="roster-panel-count mono">{filled} / {slots.length}</span>
      </div>
      <ul className="roster-panel-list">
        {slots.map((s) => {
          const bench = s.slot.startsWith('BN')
          return (
            <li
              key={s.slot}
              className={`roster-row${s.urgent ? ' roster-row-urgent' : ''}`}
            >
              <span className={`roster-row-slot mono${bench ? ' roster-row-slot-bench' : ''}`}>{s.slot}</span>
              {s.player ? (
                <>
                  {posBadge(s.player.position)}
                  <span className="roster-row-name">{s.player.name}</span>
                  <span className="roster-row-proj mono">
                    {s.player.proj_points !== null ? Math.round(s.player.proj_points) : '—'}
                  </span>
                </>
              ) : (
                <span className={`roster-row-open${s.urgent ? ' roster-row-open-urgent' : ''}`}>
                  {s.urgent ? 'needs a starter' : 'empty'}
                </span>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}
