import { useEffect, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { fetchBoard, fetchLiveState, fetchPlayers, type LiveBoard, type LiveCandidate, type LiveState, type Player } from '../api'
import DraftBoardGrid from '../components/DraftBoardGrid'
import PlayerCard from '../components/PlayerCard'

const POLL_MS = 2500

// "Xs ago" against a live-ticking clock, distinct from api.ts's `ageLabel`
// (which rounds to whole minutes -- right for a sim run's age, but the
// staleness call here is judged in *seconds*, api/live.py's 15s
// STALE_AFTER_SECONDS, so the display needs that resolution).
function pollAgeLabel(iso: string | null, nowMs: number): string {
  if (iso === null) return 'never'
  const ms = nowMs - new Date(iso).getTime()
  if (!Number.isFinite(ms) || ms < 0) return 'just now'
  const seconds = Math.floor(ms / 1000)
  if (seconds < 2) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  return `${Math.floor(seconds / 60)}m ago`
}

function posBadge(position: string | undefined): ReactNode {
  if (!position) return null
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

const fmtEvDelta = (n: number) => (n === 0 ? '—' : n.toFixed(1))

// The alt bar's fill color: a continuous red -> amber -> green ramp keyed to
// applied_pct, same color-mix technique as PlayerCard's sosTone (two-stop
// interpolation between the design system's fixed --ok/--fail tokens) but
// extended to three stops through --accent -- this app's amber -- at the
// midpoint, since the brief calls for amber as its own readable tier, not
// just a blend implied by two endpoints. Low pct (gone before your next
// pick) reads --fail; high pct (he'll last, you can wait) reads --ok.
function riskTone(pct: number): string {
  const t = Math.max(0, Math.min(100, pct))
  if (t >= 50) {
    const k = ((t - 50) / 50) * 100
    return `color-mix(in srgb, var(--ok) ${k}%, var(--accent) ${100 - k}%)`
  }
  const k = (t / 50) * 100
  return `color-mix(in srgb, var(--accent) ${k}%, var(--fail) ${100 - k}%)`
}

// The reason sentence: candidates[0] (the call) vs. candidates[1] (the
// runner-up). The argument this renders, per the task brief: the runner-up
// is nearly as good AND likely to survive to your next pick, so take the
// one who won't last.
function buildReason(
  leader: LiveCandidate,
  leaderName: string,
  runnerUp: LiveCandidate | undefined,
  runnerUpName: string
): string {
  if (!runnerUp) {
    return `Only real option in range -- ${Math.round(leader.applied_pct)}% chance he lasts to your next pick.`
  }
  const gap = leader.ev - runnerUp.ev
  return `${runnerUpName} is only ${gap.toFixed(1)} points behind and ${Math.round(runnerUp.applied_pct)}% likely to ` +
    `still be there at your next pick -- ${leaderName} is only ${Math.round(leader.applied_pct)}% likely. ` +
    `Take the one who won't last.`
}

export default function LiveDraft() {
  const [state, setState] = useState<LiveState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [board, setBoard] = useState<LiveBoard | null>(null)
  const [boardError, setBoardError] = useState<string | null>(null)
  const [players, setPlayers] = useState<Record<string, Player>>({})
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null)
  // Forces a re-render every second purely so pollAgeLabel's text ticks
  // forward between polls -- no new data is fetched for this, it just
  // reformats the elapsed time against `last_poll_at`, which is real.
  const [nowMs, setNowMs] = useState(() => Date.now())

  // Player identity (name/position/team/market_rank/market_spread) is a
  // one-time join table -- per the task brief, it does not change mid-draft.
  // The grid needs no such join (its cells carry full player data already);
  // this table exists only for the sidebar's candidate list, which -- like
  // the live search loop upstream -- is cheap by design and carries
  // `player_id` alone.
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
        // empty and names fall back to that id (see `playerName` below).
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Both live endpoints share one poll cadence: `/api/live/state` for the
  // sidebar's recommendation, `/api/live/board` for the grid. Each has its
  // own error slot so a hiccup in one doesn't blank out the other -- the
  // board can still be read while the recommendation banner explains a
  // stale search, and vice versa.
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
      try {
        const data = await fetchBoard()
        if (!cancelled) {
          setBoard(data)
          setBoardError(null)
        }
      } catch (e) {
        if (!cancelled) setBoardError(e instanceof Error ? e.message : 'Failed to load the draft board')
      }
    }

    poll()
    const id = window.setInterval(poll, POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  useEffect(() => {
    const id = window.setInterval(() => setNowMs(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [])

  const candidates = state?.candidates ?? []
  const leader = candidates[0]
  const runnerUp = candidates[1]
  const alternatives = candidates.slice(1)

  function playerName(id: string): string {
    return players[id]?.name ?? id
  }

  function findCandidate(id: string): LiveCandidate | undefined {
    return candidates.find((c) => c.player_id === id)
  }

  const selectedCandidate = selectedPlayerId ? findCandidate(selectedPlayerId) : undefined

  const youAreUp = !!state?.active && state.on_the_clock !== null && state.on_the_clock === state.my_slot
  const draftDone = !!state?.active && state.on_the_clock === null
  const isRecomputing = !!state?.active && state.candidates_as_of_pick !== null
    && state.candidates_as_of_pick < state.picks_made
  const isUncertain = !!state?.stale || isRecomputing

  // Whoever's on the clock, named by their board column when the grid has
  // loaded -- falls back to "Slot N" (the only thing `/api/live/state`
  // alone can offer) until it has.
  const clockTeam = state?.active && state.on_the_clock !== null
    ? (board?.active ? board.columns.find((c) => c.slot === state.on_the_clock)?.team_name : undefined)
      ?? `Slot ${state.on_the_clock}`
    : null

  const turnText = !state
    ? 'Loading…'
    : !state.active
    ? 'Not connected'
    : draftDone
    ? 'Draft complete'
    : youAreUp
    ? "You're up"
    : `${clockTeam} on the clock`

  return (
    <div className="live-page">
      <div className="sr-only" aria-live="polite">{turnText}.</div>

      <header className={`live-strip${youAreUp ? ' live-strip-you' : ''}`}>
        <div className="live-strip-turn">{turnText}</div>
        {state?.active && (
          <div className="live-strip-meta mono">
            <span>Pick {state.picks_made + 1}</span>
            {state.my_slot !== null && <span>Your slot {state.my_slot}</span>}
            <span title={state.last_poll_at ?? undefined}>updated {pollAgeLabel(state.last_poll_at, nowMs)}</span>
          </div>
        )}
      </header>

      {error && <p className="error live-banner">{error}</p>}
      {boardError && <p className="error live-banner">{boardError}</p>}

      {!state?.active && !error && (
        <p className="rail-empty live-banner">
          No live draft connected. <Link to="/">Start one from the home page</Link>.
        </p>
      )}

      {state?.active && (
        <>
          {state.stale && (
            <p className="live-banner live-banner-stale" role="status">
              The board may be behind -- no successful update in the last 5 seconds. Numbers below are not current.
            </p>
          )}
          {isRecomputing && (
            <p className="live-banner live-banner-recompute" role="status">
              Recomputing for pick {state.picks_made} -- the list below is still for pick {state.candidates_as_of_pick}.
            </p>
          )}
          {state.unmapped_picks.length > 0 && (
            <div className="live-banner live-banner-unmapped" role="alert">
              <strong>
                {state.unmapped_picks.length} pick{state.unmapped_picks.length === 1 ? '' : 's'} could not be matched
                to a player
              </strong>
              {' '}-- they were drafted but may still be shown as available below.
              <ul className="live-unmapped-list">
                {state.unmapped_picks.map((u) => (
                  <li key={u.overall_pick} className="mono">
                    pick {u.overall_pick} · espn id {u.espn_player_id}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="live-layout">
            <section className="live-main">
              {board?.active ? (
                <DraftBoardGrid board={board} />
              ) : (
                <p className="rail-empty">Waiting for the draft board…</p>
              )}
            </section>

            <aside className="live-sidebar">
              <div className="live-sidebar-head">
                <span className="live-sidebar-turn">{turnText}</span>
                {state.my_slot !== null && <span className="live-sidebar-slot mono">Slot {state.my_slot}</span>}
              </div>

              <div className={`live-decision${isUncertain ? ' is-uncertain' : ''}`}>
                <section className="live-call">
                  {leader ? (
                    <>
                      <button
                        type="button"
                        className="live-call-name"
                        onClick={() => setSelectedPlayerId(leader.player_id)}
                      >
                        {playerName(leader.player_id)}
                      </button>
                      <div className="live-call-meta">
                        {posBadge(players[leader.player_id]?.position)}
                        <span>{players[leader.player_id]?.team ?? '—'}</span>
                      </div>
                      <p className="live-call-reason">
                        {buildReason(leader, playerName(leader.player_id), runnerUp,
                          runnerUp ? playerName(runnerUp.player_id) : '')}
                      </p>
                    </>
                  ) : (
                    <p className="rail-empty">No recommendation yet.</p>
                  )}
                </section>

                <section className="live-alts">
                  <h2>Alternatives</h2>
                  {alternatives.length === 0 ? (
                    <p className="rail-empty">Nobody else in range.</p>
                  ) : (
                    <ul className="live-alts-list">
                      {alternatives.map((c) => (
                        <li className="live-alt" key={c.player_id}>
                          <button
                            type="button"
                            className="live-alt-btn"
                            onClick={() => setSelectedPlayerId(c.player_id)}
                          >
                            {posBadge(players[c.player_id]?.position)}
                            <span className="live-alt-name">{playerName(c.player_id)}</span>
                            <span className="live-alt-team">{players[c.player_id]?.team ?? '—'}</span>
                          </button>
                          <span className="live-alt-cost mono" title="EV cost against the call">
                            {fmtEvDelta(c.ev - (leader?.ev ?? c.ev))}
                          </span>
                          <span className="live-alt-bar-track" aria-hidden="true">
                            <span
                              className="live-alt-bar-fill"
                              style={{
                                width: `${Math.max(0, Math.min(100, c.applied_pct))}%`,
                                background: riskTone(c.applied_pct),
                              }}
                            />
                          </span>
                          <span className="live-alt-pct mono">{Math.round(c.applied_pct)}%</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </section>
              </div>
            </aside>
          </div>
        </>
      )}

      {selectedPlayerId && (
        <PlayerCard
          playerId={selectedPlayerId}
          onClose={() => setSelectedPlayerId(null)}
          avail_pct={selectedCandidate ? selectedCandidate.applied_pct / 100 : null}
          evDelta={selectedCandidate && leader ? selectedCandidate.ev - leader.ev : null}
        />
      )}
    </div>
  )
}
