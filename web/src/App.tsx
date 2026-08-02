import { useCallback, useEffect, useMemo, useState } from 'react'
import { DEFAULT_WEIGHTS, fetchPlayers, setDrafted, type Player, type Weights } from './api'
import PlayerTable from './components/PlayerTable'
import PlayerProfile from './components/PlayerProfile'
import WeightSliders from './components/WeightSliders'
import PositionTabs from './components/PositionTabs'
import FreshnessBadge from './components/FreshnessBadge'
import './App.css'

const FLEX_POSITIONS = new Set(['RB', 'WR', 'TE'])

function App() {
  const [players, setPlayers] = useState<Player[]>([])
  const [weights, setWeights] = useState<Weights>(DEFAULT_WEIGHTS)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [hideDrafted, setHideDrafted] = useState(false)
  const [positionFilter, setPositionFilter] = useState('ALL')
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null)

  async function loadPlayers(w: Weights) {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchPlayers(w)
      setPlayers(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load players')
    } finally {
      setLoading(false)
    }
  }

  // Debounce refetches while the user is dragging a slider; also covers the
  // initial load since effects run once on mount regardless of deps.
  useEffect(() => {
    const timer = setTimeout(() => {
      loadPlayers(weights)
    }, 300)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [weights])

  async function handleToggleDrafted(p: Player) {
    const next = !p.drafted
    setPlayers((prev) =>
      prev.map((pl) => (pl.player_id === p.player_id ? { ...pl, drafted: next } : pl))
    )
    await setDrafted(p.player_id, next)
    await loadPlayers(weights)
  }

  // Memoized so PlayerProfile's Esc-listener effect (keyed on this prop)
  // doesn't tear down and re-add its keydown listener on every App
  // re-render (e.g. every players refetch while the drawer is open).
  const handleCloseProfile = useCallback(() => setSelectedPlayerId(null), [])

  const filteredPlayers = useMemo(() => {
    return players.filter((p) => {
      if (hideDrafted && p.drafted) return false
      if (positionFilter === 'ALL') return true
      if (positionFilter === 'FLEX') return FLEX_POSITIONS.has(p.position)
      return p.position === positionFilter
    })
  }, [players, hideDrafted, positionFilter])

  return (
    <div className="app">
      <header className="app-header">
        <h1>Draft Board</h1>
        <FreshnessBadge />
      </header>
      <div className="app-body">
        <aside className="sidebar">
          <WeightSliders weights={weights} onChange={setWeights} />
          <label className="hide-drafted">
            <input
              type="checkbox"
              checked={hideDrafted}
              onChange={(e) => setHideDrafted(e.target.checked)}
            />
            Hide drafted
          </label>
        </aside>
        <main className="main">
          <PositionTabs value={positionFilter} onChange={setPositionFilter} />
          {/* First load / fatal error with nothing to show yet: no table to
              keep mounted, so a full-page message is the only option. */}
          {players.length === 0 && loading && <p>Loading players…</p>}
          {players.length === 0 && !loading && error && <p className="error">{error}</p>}
          {/* Once we have data, keep PlayerTable mounted across every
              refetch (slider debounce, drafted toggle) so its internal sort
              state survives -- swapping it for a loading message would
              remount the table and reset sorting back to rank. */}
          {players.length > 0 && (
            <>
              {error && <p className="error">{error}</p>}
              {loading && <p className="refreshing-hint">Refreshing…</p>}
              <div className={loading ? 'table-wrap is-loading' : 'table-wrap'}>
                <PlayerTable
                  players={filteredPlayers}
                  onToggleDrafted={handleToggleDrafted}
                  onSelectPlayer={(p) => setSelectedPlayerId(p.player_id)}
                  showTierBreaks={positionFilter !== 'ALL' && positionFilter !== 'FLEX'}
                />
              </div>
            </>
          )}
        </main>
      </div>
      {/* Overlay, not a route -- mounting/unmounting it never touches
          PlayerTable, so the board's sort/filter/scroll state survives
          opening, swapping, or closing the drawer. */}
      {selectedPlayerId && (
        <PlayerProfile
          playerId={selectedPlayerId}
          weights={weights}
          onClose={handleCloseProfile}
          onToggleDrafted={handleToggleDrafted}
          onSelectPlayer={setSelectedPlayerId}
        />
      )}
    </div>
  )
}

export default App
