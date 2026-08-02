import type { SimilarPlayer } from '../api'

interface SimilarPlayersProps {
  mode: 'stat_twins' | 'value_neighbors'
  players: SimilarPlayer[]
  onSelectPlayer: (id: string) => void
}

function seasonTag(season: number | null): string {
  if (season === null) return ''
  return `'${String(season).slice(-2)}`
}

function BoardChips({ p }: { p: SimilarPlayer }) {
  if (p.rank === null && p.adp === null) return null
  return (
    <span className="similar-chips">
      {p.rank !== null && <span className="chip">#{p.rank}</span>}
      {p.adp !== null && <span className="chip">ADP {p.adp.toFixed(1)}</span>}
    </span>
  )
}

export default function SimilarPlayers({ mode, players, onSelectPlayer }: SimilarPlayersProps) {
  if (players.length === 0) {
    return (
      <div className="similar-players">
        <p className="similar-empty">No comparable players found.</p>
      </div>
    )
  }

  if (mode === 'value_neighbors') {
    return (
      <div className="similar-players">
        <p className="similar-heading">Similar draft value (no stat history)</p>
        <ul className="similar-list">
          {players.map((p, i) => (
            <li
              // index suffix guarantees uniqueness even if the same
              // player_id appears more than once in the list
              key={`${p.player_id ?? p.name}-${i}`}
              className={p.player_id ? 'similar-row clickable' : 'similar-row'}
              onClick={p.player_id ? () => onSelectPlayer(p.player_id as string) : undefined}
            >
              <span className="similar-name">{p.name}</span>
              <BoardChips p={p} />
            </li>
          ))}
        </ul>
      </div>
    )
  }

  return (
    <div className="similar-players">
      <ul className="similar-list">
        {players.map((p, i) => (
          <li
            // index suffix guarantees uniqueness even if the same
            // player_id appears more than once in the list (e.g. a twin
            // matched from two different seasons)
            key={`${p.player_id ?? p.name}-${p.season}-${i}`}
            className={p.player_id ? 'similar-row clickable' : 'similar-row'}
            onClick={p.player_id ? () => onSelectPlayer(p.player_id as string) : undefined}
          >
            <span className="similar-name">
              {p.name} {seasonTag(p.season)}
            </span>
            <span className="similar-stats mono">
              {p.similarity !== null ? `${p.similarity.toFixed(1)} sim` : '— sim'} ·{' '}
              {p.ppg !== null ? `${p.ppg.toFixed(1)} ppg` : '— ppg'} →{' '}
              {p.next_ppg !== null ? `${p.next_ppg.toFixed(1)} next` : '—'}
            </span>
            <BoardChips p={p} />
          </li>
        ))}
      </ul>
    </div>
  )
}
