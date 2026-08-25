import type { MarketSources } from '../../api'
import PopCard, { PopRows, type PopRow } from './PopCard'
import { edgeTone, fmtRank, fmtSigned } from './payload'
import { CARD_HINTS } from './hints'

// The five sources, labels and order of `scoring/market.py`'s consensus --
// this list exists to agree with that one, not to choose its own membership.
const SOURCES: { key: keyof Omit<MarketSources, 'fp_tier'>; label: string }[] = [
  { key: 'ffc', label: 'FFC' },
  { key: 'espn', label: 'ESPN' },
  { key: 'fp', label: 'FantasyPros' },
  { key: 'mfl', label: 'MFL' },
  { key: 'cbs', label: 'CBS' },
]

// Where the market has him: your board, the consensus, and then every source
// that ranks him at all, best opinion first.
//
// It showed only the two ends of the spread for a while, on the argument that
// five rows say what two can. What that missed is the SHAPE between them: two
// endpoints cannot tell a player four sites agree on with one outlier from a
// player the five are evenly spread across, and those are different picks.
// Ordering by rank is what makes the extra rows cost nothing to read -- the
// first row is the highest anyone has him and the last is the lowest, so the
// spread the note states is the distance down the column, and nothing has to
// be labelled as an end of it.
export default function MarketRow({
  rank, marketRank, marketSpread, sources, position, posRanks = {},
  edge = null, impliedPoints = null,
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
  /** How many slots the market's consensus is off this room's own rank,
   *  positive where the market takes him LATER than the room does -- the
   *  direction a drafter shops in. It rode on the status band under the
   *  header until that band became injuries-only; this is the card the
   *  number was always about. */
  edge?: number | null
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
  // Sorted, not in `market.py`'s own order: that order is an implementation
  // fact about how the consensus is blended, and this column is read as a
  // range. Ties keep the source order behind them (`sort` is stable), so two
  // sites that agree exactly always stack the same way rather than swapping
  // places between renders.
  //
  // ONE COLUMN. These were paired into two for a revision, to use the width
  // the card had spare -- which broke the only thing that makes seven rows
  // readable at all: sorted, a single column IS the spread, top row to
  // bottom. Across two the order runs left-right-left-right, the left
  // column's numbers land in the middle of the card, and a reader has to
  // work out which of four numbers on a line belongs to which of two names.
  // The spare width is dealt with by making the card narrower (`is-narrow`),
  // not by folding the list.
  for (const s of [...covered].sort((a, b) => a.rank - b.rank)) {
    rows.push({ label: s.label, value: withPos(s.key, fmtRank(s.rank)) })
  }
  // The gap between the first two rows, stated. It is arithmetic a reader
  // could do -- board 99, consensus 54 -- but doing it is the whole question
  // the card exists to answer, and a coloured number answers it at a glance
  // where two ranks several lines apart do not.
  const footRows: PopRow[] = []
  const slots = edge === null || edge === undefined ? null : Math.round(edge)
  if (slots !== null && marketRank !== null) {
    footRows.push({
      label: 'vs consensus',
      value: (
        <span className={`delta-tone ${edgeTone(slots)}`}>
          {`${fmtSigned(edge)} ${Math.abs(slots) === 1 ? 'slot' : 'slots'}`}
        </span>
      ),
    })
  }
  // Last, because it is the one row here in a different unit: everything
  // above it is a place in a list of players and this is points a game.
  if (impliedPoints !== null) {
    footRows.push({ label: 'Implied pts', value: impliedPoints.toFixed(1) })
  }

  return (
    <PopCard
      className="is-narrow"
      title="Market"
      hint={CARD_HINTS.market}
      note={marketSpread !== null
        ? `spread ${fmtRank(marketSpread)}`
        : covered.length > 0
          ? `${covered.length} of ${SOURCES.length} sources`
          : 'unranked'}
    >
      <PopRows rows={rows} />
      {/* The verdict, in its own block: everything above it is what somebody
          says, and this is the difference between the first two of them. */}
      {footRows.length > 0 && <PopRows rows={footRows} />}
    </PopCard>
  )
}
