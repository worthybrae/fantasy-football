import type { ReactNode } from 'react'
import type { RosterPlayer } from '../../api'

// duplicated from LiveDraft.tsx / DraftBoardGrid.tsx (unexported in both):
// a four-line pure function isn't worth a shared module between three views.
function posBadge(position: string | undefined): ReactNode {
  if (!position) return null
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

// `player` is `RosterPlayer` (api/live.py's my_roster shape), not the
// board's full `Player` -- DraftRoom builds this list straight off
// /api/live/state now, and my_roster carries only player_id/name/position/
// proj_points, not the whole board row.
export type RosterSlot = { slot: string; player: RosterPlayer | null; urgent: boolean }

interface RosterPanelProps {
  slots: RosterSlot[]
  // Opens the player's profile as the popup overlay over the room -- same
  // handler shape as DraftBoardGrid/PickTicker's own `onOpenPlayer`.
  // Optional so the panel still stands on its own without a room around it;
  // DraftRoom always passes it. Only a slot that HAS a player can be opened
  // -- an empty slot has nothing to seed a profile from and stays inert
  // (see the `s.player &&` check below), which is also why it must not look
  // clickable.
  onOpenPlayer?: (player: RosterPlayer) => void
  // Whose roster this is. Defaults to the room's own wording, because in the
  // room it is always the reader's -- and is passed only by the landing
  // page's live demo, where the roster belongs to whichever seat is on the
  // clock in somebody else's draft. A panel captioned "My roster" over a
  // stranger's team is the kind of small lie that makes a reader doubt every
  // number beside it.
  label?: string
}

// WEEK 1, NOT THE SEASON. A season total is a number nobody has a feel for --
// 323 is good, 297 is fine, and only a reader who already knows the scale can
// tell them apart -- where 18.4 reads against a Sunday every manager in the
// league has watched. It is ESPN's projection for week 1, converted into this
// league's scoring by the board (`scoring/board.week_projections`).
//
// One decimal, because a week's scoring is decided in decimals: rounding 18.4
// and 17.6 to 18 apiece would erase the only difference the column has to
// show. A dash for a player ESPN does not project that week -- a printed 0.0
// would be a claim that he will not score.
function weekProj(player: RosterPlayer): string {
  const value = player.wk1_points
  return value === null || value === undefined ? '—' : value.toFixed(1)
}

export default function RosterPanel({ slots, onOpenPlayer,
                                     label = 'My roster' }: RosterPanelProps) {
  const filled = slots.filter((s) => s.player !== null).length

  return (
    <div className="roster-panel">
      <div className="roster-panel-head">
        <span className="draft-cap">{label}</span>
        {/* What the column of numbers IS. It used to be an unlabelled season
            total, which a reader could only identify by its size; a week's
            points and a season's differ by a factor of seventeen, so the
            caption is the difference between "18.4" meaning something and
            meaning nothing. */}
        <span className="roster-panel-week">wk 1</span>
        <span className="roster-panel-count mono">{filled} / {slots.length}</span>
      </div>
      <ul className="roster-panel-list">
        {slots.map((s) => {
          const bench = s.slot.startsWith('BN')
          const player = s.player
          const rowClass = `roster-row${s.urgent ? ' roster-row-urgent' : ''}`
          return (
            <li key={s.slot}>
              {/* A real <button>, only when there's a player AND a room to
                  open it in -- same "clickable only when both are true"
                  rule DraftBoardGrid's cells and PickTicker's picks follow.
                  An empty slot (or no handler) falls to the plain <div>
                  below instead, which carries the identical `.roster-row`
                  layout but none of the interaction, so it never grows a
                  hover affordance it can't back up. */}
              {player && onOpenPlayer ? (
                <button
                  type="button"
                  className={rowClass}
                  onClick={() => onOpenPlayer(player)}
                >
                  <span className={`roster-row-slot mono${bench ? ' roster-row-slot-bench' : ''}`}>{s.slot}</span>
                  {posBadge(player.position)}
                  <span className="roster-row-name">{player.name}</span>
                  <span className="roster-row-proj mono" title="ESPN's week 1 projection, in this league's scoring">
                    {weekProj(player)}
                  </span>
                </button>
              ) : (
                <div className={rowClass}>
                  <span className={`roster-row-slot mono${bench ? ' roster-row-slot-bench' : ''}`}>{s.slot}</span>
                  {player ? (
                    <>
                      {posBadge(player.position)}
                      <span className="roster-row-name">{player.name}</span>
                      <span className="roster-row-proj mono" title="ESPN's week 1 projection, in this league's scoring">
                        {weekProj(player)}
                      </span>
                    </>
                  ) : (
                    <span className={`roster-row-open${s.urgent ? ' roster-row-open-urgent' : ''}`}>
                      {s.urgent ? 'needs a starter' : 'empty'}
                    </span>
                  )}
                </div>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}
