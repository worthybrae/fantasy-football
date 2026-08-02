import type { GameLogRow } from '../api'

interface GameLogProps {
  gameLog: GameLogRow[]
}

const seasonTag = (season: number) => `'${String(season).slice(-2)}`

export default function GameLog({ gameLog }: GameLogProps) {
  if (gameLog.length === 0) {
    return <p className="drawer-placeholder">No game log available.</p>
  }

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

  return (
    <details className="game-log">
      <summary className="game-log-summary">
        Show game log ({gameLog.length} game{gameLog.length === 1 ? '' : 's'})
      </summary>
      {groups.map(({ season, games }) => (
        <div className="game-log-season" key={season}>
          <p className="game-log-season-header">{season}</p>
          <ul className="game-log-list">
            {games.map((g) => (
              <li className="game-log-row mono" key={`${g.season}-${g.week}`}>
                {seasonTag(g.season)} wk {g.week} · vs {g.opponent} · {g.stat_line} · {g.ppr_points.toFixed(1)}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </details>
  )
}
