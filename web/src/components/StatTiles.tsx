import type { Outlook, ProfileSummary } from '../api'

interface StatTilesProps {
  summary: ProfileSummary
  outlook: Outlook
  position: string
}

// Position-relevant per-game stats from summary.w_stats (keys match
// scoring/profile.py::_SUMMARY_STATS).
const STAT_ROWS: Record<string, [string, string][]> = {
  QB: [['Cmp/g', 'completions'], ['Att/g', 'attempts'], ['Pass Yds/g', 'pass_yards'],
       ['Pass TD/g', 'pass_tds'], ['INT/g', 'interceptions'], ['Rush Yds/g', 'rush_yards']],
  RB: [['Car/g', 'carries'], ['Rush Yds/g', 'rush_yards'], ['Tgt/g', 'targets'],
       ['Rec/g', 'receptions'], ['Rec Yds/g', 'rec_yards'], ['TD/g', 'tds']],
  WR: [['Tgt/g', 'targets'], ['Rec/g', 'receptions'], ['Rec Yds/g', 'rec_yards'],
       ['TD/g', 'tds'], ['Car/g', 'carries'], ['Rush Yds/g', 'rush_yards']],
  TE: [['Tgt/g', 'targets'], ['Rec/g', 'receptions'], ['Rec Yds/g', 'rec_yards'],
       ['TD/g', 'tds'], ['Car/g', 'carries'], ['Rush Yds/g', 'rush_yards']],
}

// Hero stat tiles: the recency-weighted production line, the market's
// projection, and the gap between them. Big numbers, small labels.
export default function StatTiles({ summary, outlook, position }: StatTilesProps) {
  const { w_ppg, w_stats, proj_ppg, proj_delta } = summary
  const statRow = (STAT_ROWS[position] ?? []).filter(([, key]) => key in w_stats)

  return (
    <div className="stat-tiles">
      <div className="stat-tiles-hero">
        <div className="stat-tile stat-tile-hero">
          <span className="stat-tile-value mono">{w_ppg !== null ? w_ppg.toFixed(1) : '—'}</span>
          <span className="stat-tile-label">Weighted PPG</span>
        </div>
        <div className="stat-tile stat-tile-hero">
          <span className="stat-tile-value mono">
            {proj_ppg !== null ? proj_ppg.toFixed(1) : '—'}
            {proj_delta !== null && proj_delta !== 0 && (
              <span className={proj_delta > 0 ? 'stat-tile-delta up' : 'stat-tile-delta down'}>
                {proj_delta > 0 ? `▲ +${proj_delta.toFixed(1)}` : `▼ ${proj_delta.toFixed(1)}`}
              </span>
            )}
          </span>
          <span className="stat-tile-label">ESPN projection</span>
        </div>
        <div className="stat-tile stat-tile-hero">
          <span className="stat-tile-value mono">
            {outlook.implied_points !== null ? outlook.implied_points.toFixed(1) : '—'}
          </span>
          <span className="stat-tile-label">Team implied</span>
        </div>
      </div>
      {statRow.length > 0 && (
        <div className="stat-tiles-row">
          {statRow.map(([label, key]) => (
            <div className="stat-tile" key={key}>
              <span className="stat-tile-value mono">{w_stats[key].toFixed(1)}</span>
              <span className="stat-tile-label">{label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
