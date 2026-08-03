import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchPlayers, setDrafted, type Player } from './api'
import PlayerTable from './components/PlayerTable'
import PlayerProfile from './components/PlayerProfile'
import PositionTabs from './components/PositionTabs'
import FreshnessBadge from './components/FreshnessBadge'
import TopBar from './components/TopBar'
import './App.css'

const FLEX_POSITIONS = new Set(['RB', 'WR', 'TE'])

function App() {
  const [players, setPlayers] = useState<Player[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [hideDrafted, setHideDrafted] = useState(false)
  const [positionFilter, setPositionFilter] = useState('ALL')
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  // Keyboard cursor over the board's VISIBLE (filtered+sorted) row order --
  // an index into `visibleIds`, not into `players`, since sorting lives
  // inside PlayerTable. `visibleIds` is that order, reported up by
  // PlayerTable every time it changes (filter, sort, or data reload).
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const [visibleIds, setVisibleIds] = useState<string[]>([])

  async function loadPlayers() {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchPlayers()
      setPlayers(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load players')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadPlayers()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Stable identity (deps: []) -- the keyboard effect below reads it via
  // closure every render regardless (its own deps churn on nearly every
  // keystroke anyway), but PlayerTable's `columns` memo is keyed on this
  // prop, so a stable identity here is what keeps that memo from
  // recomputing on every App render.
  const handleToggleDrafted = useCallback(
    async (p: Player) => {
      const next = !p.drafted
      setPlayers((prev) =>
        prev.map((pl) => (pl.player_id === p.player_id ? { ...pl, drafted: next } : pl))
      )
      await setDrafted(p.player_id, next)
      await loadPlayers()
    },
    []
  )

  // Memoized so PlayerProfile's Esc-listener effect (keyed on this prop)
  // doesn't tear down and re-add its keydown listener on every App
  // re-render (e.g. every players refetch while the drawer is open).
  const handleCloseProfile = useCallback(() => setSelectedPlayerId(null), [])

  // Passed to PlayerTable; called whenever the sorted+filtered row order
  // changes so the keyboard cursor always maps to what's actually on
  // screen. `[selectedIndex === i]` setState calls below are no-ops when
  // the value doesn't change, so a stable identity here isn't load-bearing
  // for render count -- it's memoized simply because there's no reason not
  // to be.
  const handleVisibleRowsChange = useCallback((ids: string[]) => setVisibleIds(ids), [])

  // Any path that points the drawer at a player -- a row click, a keyboard
  // Enter, or clicking a "similar player" inside the drawer itself -- also
  // moves the keyboard cursor there, so arrowing right afterward continues
  // from that row instead of jumping back to wherever the cursor last was.
  const handleSelectPlayerId = useCallback(
    (id: string) => {
      setSelectedPlayerId(id)
      setSelectedIndex((i) => {
        const idx = visibleIds.indexOf(id)
        return idx === -1 ? i : idx
      })
    },
    [visibleIds]
  )
  const handleSelectPlayer = useCallback(
    (p: Player) => handleSelectPlayerId(p.player_id),
    [handleSelectPlayerId]
  )

  // Clamp (not preserve) the cursor whenever the visible set changes shape --
  // a filter/sort/tab change can shrink the list out from under the current
  // index, or make it stale in a way that isn't worth chasing the same
  // player across; landing on the nearest valid row is simpler and matches
  // how every other keyboard-driven list (mail, file pickers) behaves.
  useEffect(() => {
    setSelectedIndex((i) => {
      if (i === null) return null
      if (visibleIds.length === 0) return null
      return Math.min(i, visibleIds.length - 1)
    })
  }, [visibleIds])

  // Board keyboard shortcuts: ↑/↓ move the cursor, Enter opens its profile,
  // D toggles drafted. All inert while typing in an input (search) or while
  // the drawer is open -- Enter/D acting on a row you can't see behind the
  // drawer would be surprising.
  //
  // Deliberately NOT gated on "a button/link has focus" at the top level --
  // that blanket guard used to make the whole board go dead after clicking
  // any button (a tab, a row's drafted-toggle ✓) since focus lingers on the
  // clicked element with no visual cue that keyboard nav had gone inert. Two
  // narrower rules replace it, split across mouse and keyboard activation
  // since they leave focus in different states:
  //   - Mouse clicks on those two buttons blur themselves at the end of
  //     their onClick (pointer-only -- `e.detail > 0` -- so this doesn't
  //     regress Tab order for keyboard users), so this listener never sees
  //     them as `e.target` on a subsequent keypress.
  //   - Keyboard activation of a focused button (Tab, then Enter/Space)
  //     keeps focus there by design -- instead, the Enter and D branches
  //     below early-return when `document.activeElement` is itself a
  //     button/link/select, since a keyboard user pressing Enter on a
  //     focused control means to activate *that control*, never the board
  //     cursor underneath it.
  //
  // Esc is the one exception: it's always live (regardless of focus, as
  // long as it's not already inside the search box -- TopBar's own scoped
  // Esc handler stops propagation before this listener sees the event in
  // that case), with a fixed priority: close the drawer if it's open, else
  // clear search if it has text, else do nothing. The drawer's own close
  // (PlayerProfile's window-level Esc listener, mounted only while it's
  // open) handles the first case; this effect only ever needs to handle the
  // second, and skips entirely while the drawer is open so the two don't
  // both fire off one keypress.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      // Never hijack a browser/OS chord (Cmd+D bookmark, Ctrl+D, Alt+D, ...)
      // -- every shortcut below is a bare, unmodified key.
      if (e.metaKey || e.ctrlKey || e.altKey) return

      const target = e.target as HTMLElement | null
      const isTyping = target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.isContentEditable

      if (e.key === 'Escape') {
        if (selectedPlayerId || isTyping) return
        if (search.trim()) setSearch('')
        return
      }

      if (isTyping || selectedPlayerId) return

      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        if (visibleIds.length === 0) return
        e.preventDefault()
        const delta = e.key === 'ArrowDown' ? 1 : -1
        setSelectedIndex((i) => {
          const base = i === null ? (delta > 0 ? -1 : 0) : i
          return Math.min(Math.max(base + delta, 0), visibleIds.length - 1)
        })
        return
      }

      // A Tab-focused button/link/select intercepting Enter (or Space) is
      // the browser's own activation gesture for that control -- e.g.
      // Tabbing to a position tab or a row's drafted-toggle ✓ and pressing
      // Enter should activate *that control*, not also fire the board
      // cursor's Enter/D action underneath it. Scoped to just these two
      // branches (not arrows) since arrow keys have no competing default
      // behavior on a focused button.
      const isControlActivation = !!document.activeElement?.matches?.('button, a, select')

      if (e.key === 'Enter') {
        if (isControlActivation) return
        if (selectedIndex === null) return
        const id = visibleIds[selectedIndex]
        if (id) setSelectedPlayerId(id)
        return
      }

      // Ignore key-repeat -- this one persists to disk (a POST per
      // keydown), unlike the arrow keys above, which only ever move local
      // state and are fine to auto-repeat.
      if (e.key.toLowerCase() === 'd' && !e.repeat) {
        if (isControlActivation) return
        if (selectedIndex === null) return
        const id = visibleIds[selectedIndex]
        const player = players.find((p) => p.player_id === id)
        if (player) handleToggleDrafted(player)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selectedPlayerId, search, selectedIndex, visibleIds, players, handleToggleDrafted])

  // Screen-reader-only announcement of the keyboard cursor, since its
  // on-screen indicator (PlayerTable's `.row-selected` outline) is purely
  // visual and the table isn't (and, given only the cursor moving, isn't
  // worth becoming) a full ARIA grid. `aria-live` on a plain `<div>` is
  // valid in any context, unlike `aria-selected` on a `<tr>` outside
  // `role="grid"`/`row`, which several screen readers ignore or misreport.
  const selectedAnnouncement = useMemo(() => {
    if (selectedIndex === null) return ''
    const id = visibleIds[selectedIndex]
    const p = players.find((pl) => pl.player_id === id)
    return p ? `${p.name}, ${p.position} ${p.team}, ESPN rank ${p.espn_ppr_rank ?? 'unranked'}` : ''
  }, [selectedIndex, visibleIds, players])

  const filteredPlayers = useMemo(() => {
    const query = search.trim().toLowerCase()
    return players.filter((p) => {
      if (query && !p.name.toLowerCase().includes(query) && !p.team.toLowerCase().includes(query)) {
        return false
      }
      if (hideDrafted && p.drafted) return false
      if (positionFilter === 'ALL') return true
      if (positionFilter === 'FLEX') return FLEX_POSITIONS.has(p.position)
      return p.position === positionFilter
    })
  }, [players, search, hideDrafted, positionFilter])

  return (
    <div className="app">
      <div className="sr-only" aria-live="polite">
        {selectedAnnouncement}
      </div>
      <TopBar
        search={search}
        onSearch={setSearch}
        meta={<FreshnessBadge />}
        searchShortcutDisabled={selectedPlayerId !== null}
      />
      <div className="app-body">
        <main className="main">
          <div className="board-controls">
            <PositionTabs value={positionFilter} onChange={setPositionFilter} />
            <label className="hide-drafted">
              <input
                type="checkbox"
                checked={hideDrafted}
                onChange={(e) => setHideDrafted(e.target.checked)}
              />
              Hide drafted
            </label>
          </div>
          {/* First load / fatal error with nothing to show yet: no table to
              keep mounted, so a full-page message is the only option. */}
          {players.length === 0 && loading && <p>Loading players…</p>}
          {players.length === 0 && !loading && error && <p className="error">{error}</p>}
          {/* Once we have data, keep PlayerTable mounted across every
              refetch (drafted toggle) so its internal sort state survives --
              swapping it for a loading message would remount the table and
              reset sorting back to rank. */}
          {players.length > 0 && (
            <>
              {error && <p className="error">{error}</p>}
              {loading && <p className="refreshing-hint">Refreshing…</p>}
              <div className={loading ? 'table-wrap is-loading' : 'table-wrap'}>
                <PlayerTable
                  players={filteredPlayers}
                  onToggleDrafted={handleToggleDrafted}
                  onSelectPlayer={handleSelectPlayer}
                  positionFilter={positionFilter}
                  selectedIndex={selectedIndex}
                  onVisibleRowsChange={handleVisibleRowsChange}
                />
                {/* PlayerTable renders header + zero rows on its own when
                    filteredPlayers is empty; this sits right below it so the
                    empty state still reads as part of the same table.
                    "Esc to clear" only makes sense when search is why the
                    board is empty -- tab/hide-drafted filters have no
                    keyboard shortcut to point at. */}
                {filteredPlayers.length === 0 && (
                  <p className="empty-state-row">
                    {search.trim()
                      ? 'No players match — Esc to clear'
                      : 'No players match the current filters.'}
                  </p>
                )}
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
          onClose={handleCloseProfile}
          onToggleDrafted={handleToggleDrafted}
          onSelectPlayer={handleSelectPlayerId}
        />
      )}
    </div>
  )
}

export default App
