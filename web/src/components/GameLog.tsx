import type { GameLogRow } from '../api'
import { gameColumnsFor, ptsShadeClass } from '../statColumns'

interface GameLogProps {
  gameLog: GameLogRow[]
  position: string
}

export default function GameLog({ gameLog, position }: GameLogProps) {
  if (gameLog.length === 0) {
    return <p className="drawer-placeholder">No game log available.</p>
  }

  const columns = gameColumnsFor(position)

  // Group into season blocks without re-sorting -- the API already returns
  // newest-first (season desc, week desc), and grouping by first-encounter
  // order preserves that even if a future caller passes a differently
  // ordered list.
  const groups: { season: number; games: GameLogRow[] }[] = []
  for (const game of gameLog) {
    const current = groups[groups.length - 1]
    if (current && current.season === game.season) {
      current.games.push(game)
    } else {
      groups.push({ season: game.season, games: [game] })
    }
  }

  // One table across all seasons keeps every column aligned vertically --
  // that's the whole point of the log -- with a full-width season row
  // separating the blocks. Shading compares each game to its own season's
  // average, not the career's, so a breakout year doesn't paint every
  // earlier season cold.
  return (
    <div className="game-log-wrap">
      <table className="game-log-table mono">
        <thead>
          <tr>
            <th>Wk</th>
            <th className="game-log-opp">Opp</th>
            {columns.map((c) => (
              <th key={c.label}>{c.label}</th>
            ))}
            <th>Pts</th>
          </tr>
        </thead>
        {groups.map(({ season, games }) => {
          const avg = games.reduce((sum, g) => sum + g.ppr_points, 0) / games.length
          return (
            <tbody key={season}>
              <tr className="game-log-season-row">
                <td colSpan={columns.length + 3}>
                  {season} · avg {avg.toFixed(1)} pts
                </td>
              </tr>
              {games.map((g) => (
                <tr key={`${g.season}-${g.week}`}>
                  <td>{g.week}</td>
                  <td className="game-log-opp">{g.opponent ?? '—'}</td>
                  {columns.map((c) => (
                    <td key={c.label}>{c.value(g.stats)}</td>
                  ))}
                  <td className={ptsShadeClass(g.ppr_points, avg)}>
                    {g.ppr_points.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          )
        })}
      </table>
    </div>
  )
}
