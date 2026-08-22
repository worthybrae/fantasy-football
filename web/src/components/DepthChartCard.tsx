import type { DepthChartGroup } from '../api'
import PopCard from './profile/PopCard'
import { depthGroup } from './profile/payload'

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
  const mine = depthGroup(groups, position)
  if (mine === null) return null

  const top = mine.players.slice(0, ROWS)
  // He is the reason the card is open, so he is always on it: a fourth-string
  // back cut off under a list of the three men ahead of him would be the one
  // fact the card exists to show, missing.
  const me = mine.players.find((p) => p.is_me)
  const rows = me && !top.includes(me) ? [...top, me] : top

  return (
    <PopCard title="Room" note={team}>
      {/* An ordered list because a depth chart IS an order -- the whole
          card is who stands where in the room. */}
      <ol className="pp-pop-list">
        {rows.map((p) => (
          <li
            className={`pp-pop-list-row${p.is_me ? ' is-me' : ''}`}
            key={`${p.rank}-${p.name}`}
          >
            <span className="pp-pop-list-name">{p.name}</span>
            <span className="mono pp-pop-list-tail">{mine.position}{p.rank}</span>
          </li>
        ))}
      </ol>
    </PopCard>
  )
}
