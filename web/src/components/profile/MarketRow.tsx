import type { MarketSources } from '../../api'
import { fmtRank } from './payload'

// Same five sources, same labels, same order as RankingsPanel's rows -- two
// views of one consensus that disagreed about which sites were in it would
// be worse than either.
const SOURCES: { key: keyof Omit<MarketSources, 'fp_tier'>; label: string }[] = [
  { key: 'ffc', label: 'FFC' },
  { key: 'espn', label: 'ESPN' },
  { key: 'fp', label: 'FPros' },
  { key: 'mfl', label: 'MFL' },
  { key: 'cbs', label: 'CBS' },
]

// Where the market has him: the five source ranks the board's consensus is
// built from, side by side. The consensus alone cannot tell you whether the
// room agrees with itself, and a player five sites rank 1st is a different
// pick from one they rank 1st, 3rd, 40th, 42nd and nowhere.
export default function MarketRow({ marketRank, marketSpread, sources }: {
  marketRank: number | null
  marketSpread: number | null
  sources: MarketSources
}) {
  const covered = SOURCES.filter(({ key }) => sources[key] !== null).length
  return (
    <div className="pp-market">
      <div className="pp-section-meta mono">
        {marketRank === null
          ? 'no source covers him'
          : <>avg <span className="pp-strong">{fmtRank(marketRank)}</span>
            {marketSpread !== null && <> · they disagree by {Math.round(marketSpread)}</>}</>}
      </div>
      <div className="pp-market-cells">
        {SOURCES.map(({ key, label }) => (
          <div className={`pp-market-cell${sources[key] === null ? ' is-empty' : ''}`} key={key}>
            <div className="pp-cap">{label}</div>
            <div className="mono pp-market-rank">{fmtRank(sources[key])}</div>
          </div>
        ))}
      </div>
      <p className="pp-note">
        {covered === 0
          ? 'Undrafted everywhere the board looks — the consensus rank on this card is his board rank, not a market one.'
          : covered < SOURCES.length
            ? `Only ${covered} of the five sources rank him at all; the consensus is an average of those.`
            : 'All five sources rank him, so the consensus is a real average rather than one site speaking for the room.'}
        {sources.fp_tier !== null && ` FantasyPros has him in tier ${sources.fp_tier}.`}
      </p>
    </div>
  )
}
