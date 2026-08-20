import { fmtRank, ordinal } from './payload'

// The whole board is 249 players deep, so one fixed axis covers every card
// and two players' gaps are the same size on screen when they are the same
// size in picks. A per-player axis would draw a 12-slot gap and an 81-slot
// gap identically.
const AXIS = 260

function at(rank: number): string {
  return `${Math.max(0, Math.min(100, (rank / AXIS) * 100))}%`
}

// Where this card disagrees with the room, and by how much. For a player
// with no history to argue from -- a defense, a kicker in a league that
// scores no kicking, a rookie -- this is the ONE claim the data actually
// supports, so it gets the weight rather than being a chip beside six
// panels of em-dashes.
export default function RoomGap({ rank, marketRank, edge }: {
  rank: number
  marketRank: number | null
  edge: number | null
}) {
  if (marketRank === null || edge === null) {
    return (
      <p className="pp-note">
        No market source ranks him at all, so there is no consensus for the
        board to disagree with. He is {ordinal(rank)} here on projection alone.
      </p>
    )
  }
  const slots = Math.abs(Math.round(edge))
  const earlier = edge > 0
  const lo = Math.min(rank, marketRank)
  const hi = Math.max(rank, marketRank)

  return (
    <div className="pp-gap">
      <div className="pp-gap-claim">
        <div className={`mono pp-gap-number${earlier ? ' is-accent' : ' is-bad'}`}>{slots}</div>
        <p className="pp-gap-text">
          {slots === 0
            ? <>slots apart — your board and the market agree, both at {fmtRank(marketRank)}.</>
            : <>slots {earlier ? 'earlier' : 'later'} than the market takes him — your
              board has him <span className="mono pp-strong">{ordinal(rank)}</span>,
              consensus ADP is <span className="mono pp-strong">{fmtRank(marketRank)}</span>.</>}
        </p>
      </div>
      <div className="pp-gap-axis">
        <div className="pp-gap-rail" />
        <div className="pp-gap-span" style={{ left: at(lo), width: at(hi - lo) }} />
        <div className="pp-gap-tick is-accent" style={{ left: at(rank) }} />
        <div className="pp-gap-label is-accent" style={{ left: at(rank) }}>board · {rank}</div>
        <div className="pp-gap-tick" style={{ left: at(marketRank) }} />
        <div className="pp-gap-label" style={{ left: at(marketRank) }}>
          market · {fmtRank(marketRank)}
        </div>
      </div>
    </div>
  )
}
