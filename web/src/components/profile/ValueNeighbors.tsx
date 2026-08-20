import type { SimilarPlayer } from '../../api'
import { fmtRank, fmtSigned } from './payload'

// Who else sits at this price. For a player with no stat line the comps are
// neighbours on the board rather than statistical twins, and the useful
// column is what the room charges for each of them: five defenses your board
// likes and the room is late on is a reason to wait, not to reach.
//
// Only rows the board itself knows are clickable -- the same rule
// SimilarPlayers follows, and for the same reason: opening a profile for a
// player this season's pool has never heard of 404s the fetch and wipes the
// card that is currently open.
export default function ValueNeighbors({ players, onSelectPlayer }: {
  players: SimilarPlayer[]
  onSelectPlayer: (id: string) => void
}) {
  if (players.length === 0) {
    return <p className="pp-note">Nothing else sits at this value on the board.</p>
  }
  return (
    <div className="pp-neighbors">
      {players.map((p, i) => {
        const onBoard = p.player_id !== null && p.rank !== null
        const edge = p.market_rank !== null && p.rank !== null ? p.market_rank - p.rank : null
        return (
          <button
            type="button"
            // index suffix for the same reason SimilarPlayers uses one: the
            // same id can legitimately appear twice.
            key={`${p.player_id ?? p.name}-${i}`}
            className="pp-neighbor"
            disabled={!onBoard}
            onClick={onBoard ? () => onSelectPlayer(p.player_id as string) : undefined}
          >
            <span className="mono pp-neighbor-rank">{p.rank ?? '—'}</span>
            <span className="pp-neighbor-name">{p.name}</span>
            <span className="mono pp-neighbor-adp">ADP {fmtRank(p.market_rank)}</span>
            <span className={`mono pp-neighbor-edge${
              edge !== null && edge > 0 ? ' is-good' : edge !== null && edge < 0 ? ' is-bad' : ''}`}
            >
              {fmtSigned(edge)}
            </span>
          </button>
        )
      })}
    </div>
  )
}
