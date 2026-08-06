import type { MarketSources } from '../api'

interface RankingsPanelProps {
  marketRank: number | null
  marketSpread: number | null
  sources: MarketSources
}

const SOURCE_ROWS: { key: keyof Omit<MarketSources, 'fp_tier'>; label: string }[] = [
  { key: 'ffc', label: 'FFC ADP' },
  { key: 'espn', label: 'ESPN PPR' },
  { key: 'fp', label: 'FantasyPros' },
  { key: 'mfl', label: 'MFL ADP' },
  { key: 'cbs', label: 'CBS' },
]

// Bars encode "how early does this site take him": full bar = 1st overall,
// empty = rank 200+. Log-ish compression keeps the top-50 range readable
// (where all the draft decisions actually happen).
const BAR_MAX = 200
function barWidth(rank: number): string {
  const clamped = Math.min(rank, BAR_MAX)
  const frac = 1 - Math.log(clamped) / Math.log(BAR_MAX)
  return `${Math.max(frac * 100, 2).toFixed(1)}%`
}

const fmtRank = (n: number | null) => (n === null ? '—' : Number.isInteger(n) ? String(n) : n.toFixed(1))

export default function RankingsPanel({ marketRank, marketSpread, sources }: RankingsPanelProps) {
  return (
    <div className="rankings-panel">
      <div className="rankings-agg">
        <span className="rankings-agg-value mono">{marketRank !== null ? marketRank.toFixed(1) : '—'}</span>
        <span className="rankings-agg-label">
          Aggregate rank
          {marketSpread !== null && marketSpread >= 12 && (
            <span className="rankings-spread"> · sites disagree by {Math.round(marketSpread)}</span>
          )}
        </span>
      </div>
      <ul className="rankings-rows">
        {SOURCE_ROWS.map(({ key, label }) => {
          const rank = sources[key]
          return (
            <li className="rankings-row" key={key}>
              <span className="rankings-row-label">{label}</span>
              <span className="rankings-row-bar">
                {rank !== null && <span className="rankings-row-fill" style={{ width: barWidth(rank) }} />}
              </span>
              <span className="rankings-row-value mono">{fmtRank(rank)}</span>
            </li>
          )
        })}
      </ul>
      {sources.fp_tier !== null && (
        <p className="rankings-footnote">FantasyPros tier {sources.fp_tier}</p>
      )}
    </div>
  )
}
