import { useEffect, useState } from 'react'
import { fetchLiveState, fetchPlayers, type LiveSettings, type LiveState, type Player,
         type RosterPlayer } from '../api'
import ClockPanel from '../components/draft/ClockPanel'
import RosterPanel, { type RosterSlot } from '../components/draft/RosterPanel'

const POLL_MS = 2500

type Tab = 'available' | 'board'

// Canonical display order for the starter positions ESPN's own settings
// dict (session.settings.starters, api/live.py's `settings` key) carries no
// guaranteed order for -- fixes one so the roster panel always reads QB
// before RB before WR, matching the design mock and ordinary fantasy
// convention, regardless of the order the server's dict happened to build
// its keys in.
const POSITION_ORDER = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']

// FLEX-eligible positions, matching scoring/draft_sim.py's FLEX_POSITIONS
// exactly. Not a new rule: a flex slot can only ever be filled by one of
// these three server-side (see LeagueSettings.replacement_ranks and
// _lineup_assignment); mirrored here because /api/live/state's my_roster
// carries only a flat, pick-order list of drafted players, not the
// session's own lineup assignment, so the rail has to make the same call
// itself to decide which open slot a given pick fills.
const FLEX_ELIGIBLE = new Set(['RB', 'WR', 'TE'])

// This session's real roster shape, as slot labels: `QB`, `RB1`/`RB2` for a
// 2-RB league, `FLEX1..FLEXn`, then `BN1..BNn`. Replaces the two hardcoded
// 8-team/15-round shapes this file and ClockPanel.tsx used to each carry
// (STARTER_SLOTS/BENCH_SLOTS here, LEAGUE_TEAMS in both) -- both were wrong
// the moment a real ESPN league's settings differed from the cold-start
// default, which is the whole defect this task closes. Any starters
// position outside POSITION_ORDER (an exotic ESPN slot this room doesn't
// specifically name) still gets its own labelled slot, appended after the
// ones this list does know about, so the roster's total slot count always
// matches settings.rounds -- an unlabelled position is a display gap, not a
// missing roster spot.
function rosterSlotLabels(settings: LiveSettings): string[] {
  const order = [...POSITION_ORDER,
    ...Object.keys(settings.starters).filter((p) => !POSITION_ORDER.includes(p))]
  const starters: string[] = []
  for (const pos of order) {
    const n = settings.starters[pos] ?? 0
    for (let i = 1; i <= n; i++) starters.push(n > 1 ? `${pos}${i}` : pos)
  }
  const flex = Array.from({ length: settings.flex_slots ?? 0 }, (_, i) => `FLEX${i + 1}`)
  const bench = Array.from({ length: settings.bench ?? 0 }, (_, i) => `BN${i + 1}`)
  return [...starters, ...flex, ...bench]
}

// Fills each drafted player (my_roster, already in pick order per
// api/live.py's _my_roster) into the first open slot his position can take:
// his own position's dedicated slot(s) first, then a FLEX slot if he is
// FLEX-eligible and none of his own remain, then the bench. This is a
// display-only greedy assignment for the rail, not the optimal starting
// lineup scoring/draft_sim.py's _lineup_assignment computes server-side
// (which assigns by highest points, not draft order) -- the rail only needs
// to show which slots are spoken for, not which lineup scores best.
function assignRoster(labels: string[], myRoster: RosterPlayer[]): RosterSlot[] {
  const filled: (RosterPlayer | null)[] = labels.map(() => null)
  const openIndexWhere = (test: (label: string) => boolean) =>
    labels.findIndex((label, i) => filled[i] === null && test(label))

  for (const player of myRoster) {
    let target = openIndexWhere((label) => label.replace(/\d+$/, '') === player.position)
    if (target === -1 && FLEX_ELIGIBLE.has(player.position)) {
      target = openIndexWhere((label) => label.startsWith('FLEX'))
    }
    if (target === -1) {
      target = openIndexWhere((label) => label.startsWith('BN'))
    }
    // No open slot left at all: my_roster and this session's own roster
    // shape disagree (should not happen -- both are read off the same
    // session), skipped rather than crashing the rail over a display gap.
    if (target !== -1) filled[target] = player
  }

  return labels.map((label, i) => {
    const bench = label.startsWith('BN')
    // A slot the roster still needs a starter for reads in accent (see the
    // design mock) -- bench slots never do, whether empty or not, since an
    // empty bench spot is not a gap in anyone's starting lineup.
    return { slot: label, player: filled[i], urgent: filled[i] === null && !bench }
  })
}

const SCORING_LABEL: Record<'ppr' | 'half' | 'std', string> = {
  ppr: 'PPR', half: 'Half PPR', std: 'Standard',
}

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
  // to add, alongside the Snake Board tab it feeds.
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

  // ms_remaining straight off state -- never decayed or interpolated
  // between polls, see ClockPanel.tsx's own comment on the ticker for why
  // not. null (state not loaded yet, session inactive, or no CLOCK/SELECTING
  // frame seen) reaches ClockPanel unchanged, which renders it as `--:--`.
  const secondsLeft = state?.ms_remaining != null ? Math.round(state.ms_remaining / 1000) : null

  const totalPicks = state?.active
    ? (state.settings.teams as number) * (state.settings.rounds as number)
    : null

  const slots: RosterSlot[] = state?.active
    ? assignRoster(rosterSlotLabels(state.settings), state.my_roster)
    : []

  return (
    <div className="draft-room">
      <div className="sr-only" aria-live="polite">
        {!state ? 'Loading' : !state.active ? 'Not connected' : connectionLabel}.
      </div>

      <header className="draft-topbar">
        <span className="draft-topbar-title">Draft Helper</span>
        <span className="draft-topbar-sep" aria-hidden="true" />
        {state?.active && (
          <span className="draft-topbar-league">
            {state.settings.teams} teams
            {state.settings.scoring_format && ` · ${SCORING_LABEL[state.settings.scoring_format]}`}
          </span>
        )}
        <span className="draft-topbar-spacer" />
        {state?.active && totalPicks !== null && (
          <span className="draft-topbar-pick mono">
            PICK <strong>{state.picks_made + 1}</strong> / {totalPicks}
          </span>
        )}
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
              <ClockPanel state={state} secondsLeft={secondsLeft} />
              {state.active ? (
                <RosterPanel slots={slots} />
              ) : (
                <p className="rail-empty draft-rail-loading">No roster to show.</p>
              )}
            </>
          ) : (
            <p className="rail-empty draft-rail-loading">{error ?? 'Loading…'}</p>
          )}
        </aside>
      </div>
    </div>
  )
}
