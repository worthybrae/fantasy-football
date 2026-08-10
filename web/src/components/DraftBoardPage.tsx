import { useEffect, useMemo, useState } from 'react'
import {
  ageLabel, bestEv as boardBestEv, fetchPlayers, fetchSimBoard,
  type Player, type SimBoard,
} from '../api'
import DraftGrid from './DraftGrid'
import PlayerCard from './PlayerCard'
import TopBar from './TopBar'
import FreshnessBadge from './FreshnessBadge'

export default function DraftBoardPage() {
  const [board, setBoard] = useState<SimBoard | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  // Avail%/ΔEV live on the board row, not on /api/sim/board's cells, and ΔEV
  // is defined against the best EV across the WHOLE board -- the cells cover
  // only 120 players, so a cell-local baseline would disagree with the number
  // the board page shows for the same player. Same list PlayerPage fetches
  // for its slug lookup.
  const [players, setPlayers] = useState<Player[]>([])

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
              {board.run && board.cells.length > 0 && (
                <p className="grid-legend">
                  {/* 18-44% of cells show a name their own hover disagrees
                      with, depending on the fitted reach coefficient. The
                      README explains why at length; without a line here the
                      grid just reads as broken at the moment of confusion. */}
                  Names are deduplicated across the whole board, so a cell can
                  show its second-most-likely player. Hover for that pick's raw
                  odds.{' '}
                  {/* The rail shows staleness for the same run; the screen
                      that renders 120 predictions at once showed nothing. */}
                  <span className="grid-legend-age">
                    Simulated {ageLabel(board.run.created_at)}.
                  </span>
                </p>
              )}
              <DraftGrid board={board} onSelectPlayer={setSelected} />
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
