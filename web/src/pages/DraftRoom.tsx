import { useEffect, useState } from 'react'
import { fetchLiveState, fetchPlayers, type LiveState, type Player } from '../api'
import ClockPanel from '../components/draft/ClockPanel'
import RosterPanel, { type RosterSlot } from '../components/draft/RosterPanel'

const POLL_MS = 2500

type Tab = 'available' | 'board'

// The one roster shape this personal tool drafts: scoring/league.py's
// default_settings() (1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX, 1 K, 1 DST, 5 bench --
// 15 rounds total, matching ClockPanel's LEAGUE_TEAMS=8 * 15 = 120 picks).
// Pinned here rather than fetched, same reasoning as ClockPanel's team
// count: no endpoint this room's Task 7 scope consumes returns a session's
// actual starter/flex/bench breakdown.
const STARTER_SLOTS = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'FLEX1', 'FLEX2', 'K', 'DST']
const BENCH_SLOTS = ['BN1', 'BN2', 'BN3', 'BN4', 'BN5']
const ALL_ROSTER_SLOTS = [...STARTER_SLOTS, ...BENCH_SLOTS]

// Same 8 as ClockPanel's own LEAGUE_TEAMS (scoring/config.py's
// LEAGUE_TEAMS) -- duplicated rather than imported so ClockPanel.tsx stays a
// single component export (a second export there would hit the same
// react/only-export-components fast-refresh warning RosterPanel.tsx's
// ALL_ROSTER_SLOTS used to trigger, before it moved here). This is the top
// bar's only use of it, for "PICK n / total".
const LEAGUE_TEAMS = 8
const TOTAL_PICKS = ALL_ROSTER_SLOTS.length * LEAGUE_TEAMS

export default function DraftRoom() {
  const [state, setState] = useState<LiveState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [players, setPlayers] = useState<Record<string, Player>>({})
  // Task 7 owns this state and the toggle handlers, since it renders the tab
  // strip -- Task 9 adds only the auto-switch (jumping to Available the
  // moment you come on the clock) on top of what's here.
  const [tab, setTab] = useState<Tab>('available')

  // Player identity (name/position/team) is a one-time join table -- it does
  // not change mid-draft. Lifted unchanged from the deleted LiveDraft.tsx.
  useEffect(() => {
    let cancelled = false
    fetchPlayers()
      .then((list) => {
        if (cancelled) return
        const map: Record<string, Player> = {}
        for (const p of list) map[p.player_id] = p
        setPlayers(map)
      })
      .catch(() => {
        // Candidates still render by player_id; the join table just stays
        // empty and names fall back to that id.
      })
    return () => {
      cancelled = true
    }
  }, [])

  // 2.5s poll of /api/live/state, lifted unchanged from the deleted
  // LiveDraft.tsx. The board's own poll (fetchBoard/LiveBoard) is Task 9's
  // to add, alongside the Snake Board tab it feeds -- this room does not
  // fetch it, so RosterPanel below has no way yet to know which of my
  // roster slots are actually filled (see the comment at `slots`).
  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const data = await fetchLiveState()
        if (!cancelled) {
          setState(data)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load live state')
      }
    }

    poll()
    const id = window.setInterval(poll, POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  // The tab strip's right-aligned hint (see the mock) -- Task 7's, since it
  // owns the strip itself. This is also `players`' only read site in this
  // task: the join table is fetched here for Task 8's AvailableList/TopThree
  // to consume once they exist, but nothing else in this shell needs a
  // player's name or team yet.
  const playerCount = Object.keys(players).length
  const tabHint = playerCount > 0 ? `${playerCount} players in the pool` : ''

  const listenerOk = !!state?.active && state.listener_alive && state.listener_error === null
  const connectionLabel = !state?.active ? 'NOT CONNECTED' : listenerOk ? 'ESPN LIVE' : 'LISTENER DOWN'
  const connectionTone = !state?.active ? 'off' : listenerOk ? 'ok' : 'fail'

  // No source for "which players are on my roster" in this task's scope --
  // that lives in /api/live/board's cells (BoardCell.slot vs. my_slot),
  // which Task 7 does not fetch (see the poll effect's comment above). Every
  // slot therefore renders open. `urgent` stays false rather than flagging
  // every starter slot as a need: with no fill data at all, marking all ten
  // "needs a starter" would overstate the gap on slots that may well already
  // be filled -- an honest "empty" beats a confident-looking wrong one, in
  // the same spirit as ClockPanel never fabricating a countdown.
  const slots: RosterSlot[] = ALL_ROSTER_SLOTS.map((slot) => ({ slot, player: null, urgent: false }))

  return (
    <div className="draft-room">
      <div className="sr-only" aria-live="polite">
        {!state ? 'Loading' : !state.active ? 'Not connected' : connectionLabel}.
      </div>

      <header className="draft-topbar">
        <span className="draft-topbar-title">Draft Helper</span>
        <span className="draft-topbar-sep" aria-hidden="true" />
        {state?.active && (
          <span className="draft-topbar-pick mono">
            PICK <strong>{state.picks_made + 1}</strong> / {TOTAL_PICKS}
          </span>
        )}
        <span className="draft-topbar-spacer" />
        <span className={`draft-status-pill draft-status-pill-${connectionTone}`}>
          <span className="draft-status-dot" aria-hidden="true" />
          {connectionLabel}
        </span>
      </header>

      {error && <p className="error draft-error-banner">{error}</p>}

      <div className="draft-body">
        <div className="draft-main-col">
          <div className="draft-tabs">
            <button
              type="button"
              className={`draft-tab${tab === 'available' ? ' is-active' : ''}`}
              onClick={() => setTab('available')}
            >
              Available
            </button>
            <button
              type="button"
              className={`draft-tab${tab === 'board' ? ' is-active' : ''}`}
              onClick={() => setTab('board')}
            >
              Snake Board
            </button>
            <span className="draft-tab-spacer" />
            <span className="draft-tab-hint">{tabHint}</span>
          </div>
          <div className="draft-main">
            {tab === 'available'
              ? <div className="draft-main-placeholder">Available list — Task 8</div>
              : <div className="draft-main-placeholder">Snake board — Task 9</div>}
          </div>
        </div>

        <aside className="draft-rail">
          {state ? (
            <>
              <ClockPanel state={state} secondsLeft={null} />
              <RosterPanel slots={slots} />
            </>
          ) : (
            <p className="rail-empty draft-rail-loading">{error ?? 'Loading…'}</p>
          )}
        </aside>
      </div>
    </div>
  )
}
