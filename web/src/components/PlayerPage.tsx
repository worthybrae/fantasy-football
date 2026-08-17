import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { fetchPlayers, playerSlug, setDrafted, type Player } from '../api'
import PlayerProfile from './PlayerProfile'
import PageSkeleton from './PageSkeleton'

// Full-page player view at /players/:slug. The slug is the kebab-cased
// player name; it's resolved against the board list, which also gives the
// similar-players links their id -> slug mapping.
export default function PlayerPage() {
  const { slug } = useParams<{ slug: string }>()
  const navigate = useNavigate()
  const [players, setPlayers] = useState<Player[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchPlayers()
      .then(setPlayers)
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load players'))
  }, [])

  if (error) {
    return (
      <div className="player-page">
        <p className="error">{error}</p>
        <Link to="/draft">← Back to the draft board</Link>
      </div>
    )
  }
  if (players === null) {
    return <PageSkeleton />
  }

  const player = players.find((p) => playerSlug(p.name) === slug)
  if (!player) {
    return (
      <div className="player-page">
        <p>No player found for “{slug}”.</p>
        <Link to="/draft">← Back to the draft board</Link>
      </div>
    )
  }

  return (
    <PlayerProfile
      playerId={player.player_id}
      onClose={() => navigate('/draft')}
      onToggleDrafted={async (p) => {
        await setDrafted(p.player_id, !p.drafted)
      }}
      onSelectPlayer={(id) => {
        const target = players.find((p) => p.player_id === id)
        if (target) navigate(`/players/${playerSlug(target.name)}`)
      }}
    />
  )
}
