import type { SimilarPlayer } from '../../api'
import PopCard from './PopCard'
import { fmtRank, type RankedPlayer } from './payload'

// Four rows, his own among them: who else the board has at this price.
//
// TWO SOURCES, one card, because the payload only answers for some players.
// `similar` carries board neighbours (`value_neighbors`: his own position,
// nearest by value) only for a player it has no stat line to match -- a
// rookie, a kicker, a defense. For everyone else it carries stat twins,
// seasons that looked like his, whose `rank` is where those players sit on
// TODAY's board: drawing those here would put Alvin Kamara at 273 under a
// card that claims to be about the picks around this one. `ComparableSeasons`
// is where the twins belong.
//
// So a veteran's neighbours come from the room's own ranked board instead
// (`ranked`, threaded down from DraftRoom). Same claim either way -- the run
// of picks this one sits in -- which is why the two paths end in the same
// four rows rather than in two cards.
const NEIGHBOURS = 4

interface NeighbourRow {
  id: string | null
  name: string
  rank: number
  marketRank: number | null
  me: boolean
}

// Only what a row needs, so the payload's `SimilarPlayer` and the board's
// `RankedPlayer` can both be read against it.
interface Me {
  player_id: string
  name: string
  rank: number
  market_rank: number | null
}

// Windowed on him, two above where the board is deep enough: the picks ahead
// of him are the ones you would have to pass on to take him.
function windowOn<T>(rows: T[], mine: number): T[] {
  const start = Math.max(0, Math.min(mine - 2, rows.length - NEIGHBOURS))
  return rows.slice(start, start + NEIGHBOURS)
}

// The payload's own neighbours, already scored against him server-side
// (scoring/similarity.value_neighbors). Sorted here because the payload
// orders them by distance in value, and this card is a run of picks.
function fromPayload(players: SimilarPlayer[], me: Me): NeighbourRow[] {
  const rows: NeighbourRow[] = players
    .filter((p): p is SimilarPlayer & { rank: number } => p.rank !== null)
    .map((p) => ({
      id: p.player_id, name: p.name, rank: p.rank, marketRank: p.market_rank, me: false,
    }))
  if (rows.length === 0) return []
  // His own row goes in the list rather than beside it: the card is about
  // where he sits in a run of picks, and a run with him left out of it is
  // just a list of other players.
  rows.push({ id: null, name: me.name, rank: me.rank, marketRank: me.market_rank, me: true })
  rows.sort((a, b) => a.rank - b.rank)
  return windowOn(rows, rows.findIndex((r) => r.me))
}

// The room's board, in rank order and already down to who is still
// available (DraftRoom filters it by the drafted set), which holds his own
// row -- so this one windows first and shapes four rows rather than five
// hundred.
//
// Matched by id, not by rank: the two agree (both are the same board build),
// but an id is the thing that identifies him and a rank is a position two
// players could be argued into sharing. An id the list has never heard of --
// a roster player seeded from ESPN's feed, a player already picked, or a
// board that has not loaded -- leaves the card undrawn rather than drawing a
// run he is not in.
function fromBoard(ranked: RankedPlayer[], me: Me): NeighbourRow[] {
  const mine = ranked.findIndex((p) => p.player_id === me.player_id)
  if (mine === -1) return []
  return windowOn(ranked, mine).map((p) => {
    const isMe = p.player_id === me.player_id
    return {
      id: isMe ? null : p.player_id,
      name: p.name,
      rank: p.rank,
      marketRank: p.market_rank,
      me: isMe,
    }
  })
}

export default function ValueNeighbors({ mode, players, ranked, me, onSelectPlayer }: {
  mode: 'stat_twins' | 'value_neighbors'
  players: SimilarPlayer[]
  ranked: RankedPlayer[]
  me: Me
  onSelectPlayer: (id: string) => void
}) {
  // The payload first, when it is the kind of payload that can answer: it is
  // the same board, scored at the same moment as the rest of this popup.
  const payload = mode === 'value_neighbors' ? fromPayload(players, me) : []
  const rows = payload.length > 0 ? payload : fromBoard(ranked, me)
  if (rows.length === 0) return null

  return (
    <PopCard title="Near you" note="by board rank">
      {/* Ordered, because board rank is the order: the rows are a run of
          picks and the list markup says which run. */}
      <ol className="pp-pop-list is-board">
        {rows.map((r) => {
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
