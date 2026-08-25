import type { DepthChartGroup } from '../api'
import PopCard from './profile/PopCard'
import { CARD_HINTS } from './profile/hints'
import { depthGroup } from './profile/payload'

// HIS room, whole. Still not the four position groups this once printed -- a
// manager holding a pick does not need Detroit's third tight end -- but no
// longer the top three either: the men behind him are the ones who take his
// carries when a coach changes his mind, and a room cut off at three said
// nothing about how deep it goes. The rest of the offence stays a click away
// on those players' own cards.

export default function DepthChartCard({ team, position, groups }: {
  team: string
  position: string
  groups: DepthChartGroup[]
}) {
  const mine = depthGroup(groups, position)
  if (mine === null) return null

  const top = mine.players
  // He is the reason the card is open, so he is always on it: a fourth-string
  // back cut off under a list of the three men ahead of him would be the one
  // fact the card exists to show, missing.
  const me = mine.players.find((p) => p.is_me)
  const rows = me && !top.includes(me) ? [...top, me] : top

  return (
    <PopCard title="Depth chart" note={team} hint={CARD_HINTS.depth} className="is-wide">
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
