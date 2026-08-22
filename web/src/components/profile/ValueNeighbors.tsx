import type { SimilarPlayer } from '../../api'
import PopCard from './PopCard'
import { fmtRank } from './payload'

// Four rows, his own among them: who else the board has at this price.
//
// The list only exists for a player the payload sends `value_neighbors` for.
// `stat_twins` is a different question with the same shape -- seasons that
// looked like his, whose `rank` is where those players sit on TODAY's board
// and has nothing to do with his own -- and drawing them here would put
// Alvin Kamara at 273 under a card that claims to be about the picks around
// this one. `ComparableSeasons` is the card those belong on.
const NEIGHBOURS = 4

interface NeighbourRow {
  id: string | null
  name: string
  rank: number
  marketRank: number | null
  me: boolean
}

export default function ValueNeighbors({ mode, players, me, onSelectPlayer }: {
  mode: 'stat_twins' | 'value_neighbors'
  players: SimilarPlayer[]
  me: { name: string; rank: number; market_rank: number | null }
  onSelectPlayer: (id: string) => void
}) {
  if (mode !== 'value_neighbors') return null

  const rows: NeighbourRow[] = players
    .filter((p): p is SimilarPlayer & { rank: number } => p.rank !== null)
    .map((p) => ({
      id: p.player_id, name: p.name, rank: p.rank, marketRank: p.market_rank, me: false,
    }))
  if (rows.length === 0) return null
  // His own row goes in the list rather than beside it: the card is about
  // where he sits in a run of picks, and a run with him left out of it is
  // just a list of other players.
  rows.push({ id: null, name: me.name, rank: me.rank, marketRank: me.market_rank, me: true })
  rows.sort((a, b) => a.rank - b.rank)

  // Windowed on him, two above where the board is deep enough: the picks
  // ahead of him are the ones you would have to pass on to take him.
  const mine = rows.findIndex((r) => r.me)
  const start = Math.max(0, Math.min(mine - 2, rows.length - NEIGHBOURS))
  const shown = rows.slice(start, start + NEIGHBOURS)

  return (
    <PopCard title="Near you" note="by board rank">
      {/* Ordered, because board rank is the order: the rows are a run of
          picks and the list markup says which run. */}
      <ol className="pp-pop-list is-board">
        {shown.map((r) => {
          // A `const` rather than `r.id` at the click site: narrowing an
          // object's property does not survive into a closure, and this is
          // the difference between the guard below meaning something and a
          // cast that says it does.
          const id = r.id
          const cells = (
            <>
              <span className="mono pp-pop-list-rank">{r.rank}</span>
              <span className="pp-pop-list-name">{r.name}</span>
              <span className="mono pp-pop-list-tail" title="consensus ADP">
                {fmtRank(r.marketRank)}
              </span>
            </>
          )
          const key = `${id ?? r.name}-${r.rank}`
          // His own row, and any neighbour the board has no id for, is not a
          // control: opening the profile that is already open does nothing,
          // and fetching a player this season's pool has never heard of 404s
          // and wipes the card. Same rule ComparableSeasons follows.
          if (r.me || id === null) {
            return (
              <li className={`pp-pop-list-row${r.me ? ' is-me' : ''}`} key={key}>{cells}</li>
            )
          }
          return (
            <li key={key}>
              <button
                type="button"
                className="pp-pop-list-row"
                onClick={() => onSelectPlayer(id)}
              >
                {cells}
              </button>
            </li>
          )
        })}
      </ol>
    </PopCard>
  )
}
