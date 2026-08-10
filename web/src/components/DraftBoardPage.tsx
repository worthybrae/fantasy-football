import { useEffect, useState } from 'react'
import { fetchSimBoard, type SimBoard } from '../api'
import DraftGrid from './DraftGrid'
import PlayerCard from './PlayerCard'
import TopBar from './TopBar'
import FreshnessBadge from './FreshnessBadge'

export default function DraftBoardPage() {
  const [board, setBoard] = useState<SimBoard | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)

  useEffect(() => {
    fetchSimBoard()
      .then(setBoard)
      .catch((e) => setError(e instanceof Error ? e.message : 'Load failed'))
  }, [])

  return (
    <div className="app">
      <TopBar search="" onSearch={() => {}} meta={<FreshnessBadge />} searchShortcutDisabled />
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
              <DraftGrid board={board} onSelectPlayer={setSelected} />
            </>
          )}
        </main>
      </div>
      {selected && <PlayerCard playerId={selected} onClose={() => setSelected(null)} />}
    </div>
  )
}
