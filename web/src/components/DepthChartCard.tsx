import type { DepthChartGroup } from '../api'
import PopCard from './profile/PopCard'

// The card used to print the whole offence, four position groups at once. A
// manager holding a pick does not need to know who Detroit's third tight end
// is; he needs to know who is standing between this player and the ball. So
// this is HIS room only -- the group he is charted in, from the top -- and
// the rest of the offence is a click away on those players' own cards.
const ROWS = 3

export default function DepthChartCard({ team, position, groups }: {
  team: string
  position: string
  groups: DepthChartGroup[]
}) {
  // Whichever group actually holds him, before the one his board position
  // names: a player charted somewhere other than where the board ranks him
  // (a receiver taking snaps at running back) belongs in the room he is
  // actually competing in.
  const mine = groups.find((g) => g.players.some((p) => p.is_me))
    ?? groups.find((g) => g.position === position)
  if (!mine || mine.players.length === 0) return null

  const top = mine.players.slice(0, ROWS)
  // He is the reason the card is open, so he is always on it: a fourth-string
  // back cut off under a list of the three men ahead of him would be the one
  // fact the card exists to show, missing.
  const me = mine.players.find((p) => p.is_me)
  const rows = me && !top.includes(me) ? [...top, me] : top

  return (
    <PopCard title="Room" note={team}>
      <div className="pp-pop-list">
        {rows.map((p) => (
          <div
            className={`pp-pop-list-row${p.is_me ? ' is-me' : ''}`}
            key={`${p.rank}-${p.name}`}
          >
            <span className="pp-pop-list-name">{p.name}</span>
            <span className="mono pp-pop-list-tail">{mine.position}{p.rank}</span>
          </div>
        ))}
      </div>
    </PopCard>
  )
}
