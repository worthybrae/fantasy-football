import type { SeasonSummary } from '../api'

interface SeasonTableProps {
  seasons: SeasonSummary[]
}

const fmtPct = (n: number | null) => (n === null ? '—' : `${(n * 100).toFixed(1)}%`)
const fmt1 = (n: number | null) => (n === null ? '—' : n.toFixed(1))

export default function SeasonTable({ seasons }: SeasonTableProps) {
  if (seasons.length === 0) {
    return <p className="drawer-placeholder">No season history available.</p>
  }

  const rows = [...seasons].sort((a, b) => b.season - a.season)

  return (
    <div className="season-table-wrap">
      <table className="season-table mono">
        <thead>
          <tr>
            <th>Season</th>
            <th>G</th>
            <th>PPG</th>
            <th>Tgt</th>
            <th>Tgt%</th>
            <th>Car</th>
            <th>Rec Yds</th>
            <th>Rush Yds</th>
            <th>TD</th>
            <th>Rec</th>
            <th>Yds/Opp</th>
            <th>Snap%</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => (
            <tr key={s.season}>
              <td>{s.season}</td>
              <td>{s.games}</td>
              <td>{s.ppg.toFixed(1)}</td>
              <td>{s.targets}</td>
              <td>{fmtPct(s.target_share)}</td>
              <td>{s.carries}</td>
              <td>{s.rec_yards}</td>
              <td>{s.rush_yards}</td>
              <td>{s.tds}</td>
              <td>{s.receptions}</td>
              <td>{fmt1(s.yards_per_opp)}</td>
              <td>{fmtPct(s.snap_share)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
