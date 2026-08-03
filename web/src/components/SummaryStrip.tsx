import type { ProfileSummary } from '../api'

interface SummaryStripProps {
  summary: ProfileSummary
  position: string
}

// Position-relevant per-game stats to surface from summary.w_stats. Keys
// must match scoring/profile.py::_SUMMARY_STATS (season aggregates carry a
// combined `tds`, unlike the game log's split rush/rec TDs).
const STAT_ROWS: Record<string, [string, string][]> = {
  QB: [['Cmp/g', 'completions'], ['Att/g', 'attempts'], ['Pass Yds/g', 'pass_yards'],
       ['Pass TD/g', 'pass_tds'], ['INT/g', 'interceptions'], ['Rush Yds/g', 'rush_yards']],
  RB: [['Car/g', 'carries'], ['Rush Yds/g', 'rush_yards'], ['Tgt/g', 'targets'],
       ['Rec/g', 'receptions'], ['Rec Yds/g', 'rec_yards'], ['TD/g', 'tds']],
  WR: [['Tgt/g', 'targets'], ['Rec/g', 'receptions'], ['Rec Yds/g', 'rec_yards'],
       ['TD/g', 'tds'], ['Car/g', 'carries']],
  TE: [['Tgt/g', 'targets'], ['Rec/g', 'receptions'], ['Rec Yds/g', 'rec_yards'],
       ['TD/g', 'tds']],
}

export default function SummaryStrip({ summary, position }: SummaryStripProps) {
  const { w_ppg, w_stats, proj_ppg, proj_delta } = summary
  if (w_ppg === null && proj_ppg === null) return null

  const statRow = (STAT_ROWS[position] ?? [])
    .filter(([, key]) => key in w_stats)

  return (
    <div className="summary-strip">
      <div className="drawer-chips">
        {w_ppg !== null && <span className="chip">W-avg {w_ppg.toFixed(1)} ppg</span>}
        {proj_ppg !== null && <span className="chip">ESPN proj {proj_ppg.toFixed(1)} ppg</span>}
        {proj_delta !== null && proj_delta > 0 && (
          <span className="chip chip-up">↑ growth expected +{proj_delta.toFixed(1)}</span>
        )}
        {proj_delta !== null && proj_delta < 0 && (
          <span className="chip chip-down">↓ decline {proj_delta.toFixed(1)}</span>
        )}
        {proj_delta === 0 && <span className="chip">→ flat</span>}
      </div>
      {statRow.length > 0 && (
        <div className="drawer-chips summary-stat-row mono">
          {statRow.map(([label, key]) => (
            <span className="chip" key={key}>
              {label} {w_stats[key].toFixed(1)}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
