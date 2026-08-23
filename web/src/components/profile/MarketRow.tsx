import type { MarketSources } from '../../api'
import PopCard, { PopRows, type PopRow } from './PopCard'
import { fmtRank } from './payload'

// The five sources, labels and order of `scoring/market.py`'s consensus --
// this list exists to agree with that one, not to choose its own membership.
const SOURCES: { key: keyof Omit<MarketSources, 'fp_tier'>; label: string }[] = [
  { key: 'ffc', label: 'FFC' },
  { key: 'espn', label: 'ESPN' },
  { key: 'fp', label: 'FantasyPros' },
  { key: 'mfl', label: 'MFL' },
  { key: 'cbs', label: 'CBS' },
]

// Where the market has him, in four lines: your board, the consensus, and the
// two sources the consensus is stretched between.
//
// The card used to print all five ranks as tiles. Five is more than this
// column can hold and four of them usually say the same thing -- what the
// note states is the SPREAD, and the two rows under it are who is at each end
// of it. A player five sites rank 1st is a different pick from one they rank
// 3rd and 40th, and those two rows are the shortest way to see which this is.
export default function MarketRow({
  rank, marketRank, marketSpread, sources, position, posRanks = {},
  impliedPoints = null,
}: {
  rank: number
  marketRank: number | null
  marketSpread: number | null
  sources: MarketSources
  /** His position, for the positional places below. */
  position: string
  /** Each source's number as a place among his own position, keyed the same
   *  way `sources` is plus `board` and `consensus`. A drafter fills a roster
   *  by position, so "WR7" is the number the row is really about and the
   *  overall is the scale it sits on. */
  posRanks?: Record<string, number | undefined>
  /** Points a game the betting lines imply for his own offence. Passed only
   *  for a player with no seasons (see PlayerProfile), which is the one
   *  state where the card has room for it and the reader has nothing else
   *  forward-looking to hold it against -- Sparse.dc.html puts it on this
   *  card for exactly that player and Main.dc.html does not. */
  impliedPoints?: number | null
}) {
  const covered = SOURCES
    .map(({ key, label }) => ({ key, label, rank: sources[key] }))
    .filter((s): s is { key: typeof SOURCES[number]['key']; label: string; rank: number } =>
      s.rank !== null && s.rank !== undefined)

  // "19" is a fact about 250 players; "WR7" is the one that decides a pick,
  // because a roster is filled by position. Both, with the overall leading:
  // it is the scale the row sits on and the positional place is what it
  // means.
  const withPos = (key: string, value: string) => {
    const place = posRanks[key]
    return place === undefined ? value : `${value}  ${position}${place}`
  }

  const rows: PopRow[] = [
    // First and in accent, the one number on this card the app computes
    // rather than reports -- the other three are what it is read against.
    { label: 'Your board', value: withPos('board', String(rank)), tone: 'accent' },
  ]
  if (marketRank !== null) {
    rows.push({ label: 'Consensus', value: withPos('consensus', fmtRank(marketRank)) })
  }
  if (covered.length > 0) {
    const low = covered.reduce((a, b) => (b.rank < a.rank ? b : a))
    const high = covered.reduce((a, b) => (b.rank > a.rank ? b : a))
    rows.push({ label: low.label, value: withPos(low.key, fmtRank(low.rank)) })
    // Ties are broken by `market.py`'s own order, so one source can hold both
    // ends -- when it does, that is one row, not the same row twice.
    if (high.label !== low.label) {
      rows.push({ label: high.label, value: withPos(high.key, fmtRank(high.rank)) })
    }
  }
  // Last, because it is the one row here in a different unit: everything
  // above it is a place in a list of players and this is points a game.
  if (impliedPoints !== null) {
    rows.push({ label: 'Implied pts', value: impliedPoints.toFixed(1) })
  }

  return (
    <PopCard
      title="Market"
      note={marketSpread !== null
        ? `spread ${fmtRank(marketSpread)}`
        : covered.length > 0
          ? `${covered.length} of ${SOURCES.length} sources`
          : 'unranked'}
    >
      <PopRows rows={rows} />
    </PopCard>
  )
}
