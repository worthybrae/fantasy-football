import type { SeasonSummary } from '../api'
import { ptsShadeClass, seasonColumnsFor } from '../statColumns'

interface SeasonTableProps {
  seasons: SeasonSummary[]
  position: string
}

export default function SeasonTable({ seasons, position }: SeasonTableProps) {
  if (seasons.length === 0) {
    return <p className="drawer-placeholder">No season history available.</p>
  }

  const columns = seasonColumnsFor(position)
  const rows = [...seasons].sort((a, b) => b.season - a.season)

  // Games-weighted career PPG, so a 3-game injury season doesn't drag the
  // reference point the way a plain mean of season PPGs would.
  const totalGames = rows.reduce((sum, s) => sum + s.games, 0)
  const careerPpg = totalGames > 0
    ? rows.reduce((sum, s) => sum + s.ppg * s.games, 0) / totalGames
    : 0

  return (
    <div className="season-table-wrap">
      <table className="season-table mono">
        <thead>
          <tr>
            <th>Season</th>
            <th>G</th>
            <th>PPG</th>
            {columns.map((c) => (
              <th key={c.label}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => (
            <tr key={s.season}>
              <td>{s.season}</td>
              <td>{s.games}</td>
              <td className={ptsShadeClass(s.ppg, careerPpg)}>{s.ppg.toFixed(1)}</td>
              {columns.map((c) => (
                <td key={c.label}>{c.value(s)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
