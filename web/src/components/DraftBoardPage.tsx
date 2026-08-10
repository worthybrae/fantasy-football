import { useEffect, useMemo, useState } from 'react'
import {
  ageLabel, bestEv as boardBestEv, fetchManagerHistory, fetchManagers, fetchPlayers, fetchSimBoard,
  type Manager, type ManagerHistory, type Player, type SimBoard,
} from '../api'
import DraftGrid from './DraftGrid'
import ManagerForecast from './ManagerForecast'
import PlayerCard from './PlayerCard'
import TopBar from './TopBar'
import FreshnessBadge from './FreshnessBadge'

type BoardView = 'grid' | 'forecast'

export default function DraftBoardPage() {
  const [board, setBoard] = useState<SimBoard | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [view, setView] = useState<BoardView>('grid')
  // Avail%/ΔEV live on the board row, not on /api/sim/board's cells, and ΔEV
  // is defined against the best EV across the WHOLE board -- the cells cover
  // only 120 players, so a cell-local baseline would disagree with the number
  // the board page shows for the same player. Same list PlayerPage fetches
  // for its slug lookup.
  const [players, setPlayers] = useState<Player[]>([])
  // Per-manager tendency (summary, uses_personal) for the forecast tab's
  // cards. Fetched alongside the board rather than only when that tab is
  // selected, so switching tabs never shows a loading flash.
  const [managers, setManagers] = useState<Manager[]>([])
  // Each manager's real draft history (first-rounders, positional shape) --
  // the evidence the forecast cards rest on. Same failure treatment as
  // managers above: a fetch failure just leaves the history section empty
  // per card, it never blocks the grid tab.
  const [history, setHistory] = useState<ManagerHistory[]>([])

  useEffect(() => {
    fetchSimBoard()
      .then(setBoard)
      .catch((e) => setError(e instanceof Error ? e.message : 'Load failed'))
  }, [])

  // Deliberately not surfaced as a page error: the grid is fully usable
  // without it, and only the card's two extra numbers go missing.
  useEffect(() => {
    fetchPlayers().then(setPlayers).catch(() => setPlayers([]))
  }, [])

  // Same treatment: a manager fetch failing just means the forecast cards
  // fall back to "No model fitted for this manager yet." per card instead of
  // taking down a page the grid tab doesn't need this data for at all.
  useEffect(() => {
    fetchManagers().then(setManagers).catch(() => setManagers([]))
  }, [])

  useEffect(() => {
    fetchManagerHistory().then(setHistory).catch(() => setHistory([]))
  }, [])

  const bestEv = useMemo(() => boardBestEv(players), [players])
  const sel = selected ? players.find((p) => p.player_id === selected) : undefined

  return (
    <div className="app">
      <TopBar meta={<FreshnessBadge />} />
      <div className="app-body">
        <main className="main">
          {error && <p className="error">{error}</p>}
          {!board && !error && <p className="refreshing-hint">Loading…</p>}
          {board && board.order.length === 0 && (
            <p className="rail-empty">
              Run `make espn-import` to load your league, then set the draft
              order on the board page.
            </p>
          )}
          {board && board.order.length > 0 && (
            <>
              {!board.run && (
                <p className="rail-empty">
                  No simulation has run yet. Set your slot on the board page and
                  press Run — this grid fills in with the prediction.
                </p>
              )}
              {board.run && board.cells.length === 0 && (
                <p className="rail-empty">
                  The last simulation predates this board and left no per-pick
                  predictions. Run a new simulation on the board page to fill
                  it in.
                </p>
              )}
              {/* The tab switcher (and whichever view it's on) renders
                  whenever there's a league to show columns/cards for, same
                  as DraftGrid always did before this tab existed --
                  regardless of whether a sim has ever run. With no cells,
                  DraftGrid renders its own full skeleton (every cell
                  `grid-empty`) and ManagerForecast still has something
                  honest to show per card (identity, real draft history,
                  tendency text) even with an all-empty picks list, so
                  neither view needs cells to be worth switching to. */}
              <div className="board-view-tabs">
                <button
                  type="button"
                  className={view === 'grid' ? 'active' : undefined}
                  onClick={() => setView('grid')}
                >
                  Grid
                </button>
                <button
                  type="button"
                  className={view === 'forecast' ? 'active' : undefined}
                  onClick={() => setView('forecast')}
                >
                  By manager
                </button>
              </div>
              {/* Only meaningful once there's a real run with per-pick cells
                  to describe -- an empty grid/forecast has no dedup
                  behavior or staleness to explain yet. */}
              {board.run && board.cells.length > 0 && (
                <p className="grid-legend">
                  {view === 'grid' ? (
                    <>
                      {/* 18-44% of cells show a name their own hover
                          disagrees with, depending on the fitted reach
                          coefficient. The README explains why at length;
                          without a line here the grid just reads as broken
                          at the moment of confusion. */}
                      Names are deduplicated across the whole board, so a
                      cell can show its second-most-likely player. Hover for
                      that pick's raw odds.{' '}
                    </>
                  ) : (
                    <>
                      One card per manager, in draft-slot order — the same
                      deduplicated prediction the grid's cells show, without
                      the alternates. Switch to Grid to see a pick's raw
                      odds against its alternates.{' '}
                    </>
                  )}
                  {/* The rail shows staleness for the same run; the screen
                      that renders 120 predictions at once showed nothing. */}
                  <span className="grid-legend-age">
                    Simulated {ageLabel(board.run.created_at)}.
                  </span>
                </p>
              )}
              {view === 'grid' ? (
                <DraftGrid board={board} onSelectPlayer={setSelected} />
              ) : (
                <ManagerForecast board={board} managers={managers} history={history} onSelectPlayer={setSelected} />
              )}
            </>
          )}
        </main>
      </div>
      {selected && (
        <PlayerCard
          playerId={selected}
          onClose={() => setSelected(null)}
          avail_pct={sel?.avail_pct ?? null}
          // Board-relative, per PlayerCard's prop contract -- the card has no
          // board list of its own to subtract a baseline from. `ev` is null
          // for nearly every player (search_pick scores about twelve
          // candidates), so this renders only around your next pick.
          evDelta={sel?.ev != null && bestEv != null ? sel.ev - bestEv : null}
        />
      )}
    </div>
  )
}
