import { useEffect, useState } from 'react'
import { DEFAULT_WEIGHTS, fetchPlayers, setDrafted, type Player } from './api'
import PlayerTable from './components/PlayerTable'
import './App.css'

function App() {
  const [players, setPlayers] = useState<Player[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    load()
  }, [])

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchPlayers(DEFAULT_WEIGHTS)
      setPlayers(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load players')
    } finally {
      setLoading(false)
    }
  }

  async function handleToggleDrafted(p: Player) {
    const next = !p.drafted
    setPlayers((prev) =>
      prev.map((pl) => (pl.player_id === p.player_id ? { ...pl, drafted: next } : pl))
    )
    await setDrafted(p.player_id, next)
    await load()
  }

  return (
    <div className="app">
      <h1>Draft Board</h1>
      {loading && <p>Loading players…</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && (
        <PlayerTable players={players} onToggleDrafted={handleToggleDrafted} />
      )}
    </div>
  )
}

export default App
