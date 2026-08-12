import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { fetchLiveState, fetchPlayers, type LiveCandidate, type LiveState, type Player } from '../api'
import PlayerCard from '../components/PlayerCard'

const POLL_MS = 2500
// Cap on the "off the board" feed -- see the component doc comment for how
// it's built. Eight is enough to cover the last round or so without the
// aside outgrowing the decision panel next to it.
const MAX_RECENT_PICKS = 8

// One drafted player this client has observed since mount, joined against
// the one-time player fetch at the moment its `drafted` flag flipped. This
// is the only honest way to build a "recent picks" feed: `/api/live/state`
// carries no pick history (only `picks_made`, a count) and `/api/players`
// carries no timestamp -- see the module doc comment below for the full
// rationale.
interface RecentPick {
  player_id: string
  name: string
  position: string
  team: string
  detectedAt: number
}

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
  const [players, setPlayers] = useState<Record<string, Player>>({})
  const [recentPicks, setRecentPicks] = useState<RecentPick[]>([])
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null)
  // Forces a re-render every second purely so pollAgeLabel's text ticks
  // forward between polls -- no new data is fetched for this, it just
  // reformats the elapsed time against `last_poll_at`, which is real.
  const [nowMs, setNowMs] = useState(() => Date.now())

  // `drafted` truth as of the last players fetch, checked by identity
  // (`.has`) rather than trusting a derived boolean in `players` state,
  // since the diff that builds `recentPicks` needs "was this NOT drafted a
  // moment ago" -- a question a plain state snapshot can't answer inside
  // the same async callback that's about to replace it.
  const draftedIdsRef = useRef<Set<string>>(new Set())
  const picksMadeRef = useRef<number | null>(null)
  const playersLoadedRef = useRef(false)

  // Player identity (name/position/team/market_rank/market_spread) is a
  // one-time join table -- per the task brief, it does not change mid-draft.
  // `drafted` does change, but only in response to a real pick landing (see
  // the poll effect below), never on a fixed timer of its own.
  useEffect(() => {
    let cancelled = false
    fetchPlayers()
      .then((list) => {
        if (cancelled) return
        const map: Record<string, Player> = {}
        const drafted = new Set<string>()
        for (const p of list) {
          map[p.player_id] = p
          if (p.drafted) drafted.add(p.player_id)
        }
        setPlayers(map)
        draftedIdsRef.current = drafted
        playersLoadedRef.current = true
      })
      .catch(() => {
        // Candidates still render by player_id; the join table just stays
        // empty and names fall back to that id (see `playerName` below).
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function refreshPlayersAndDiff() {
      if (!playersLoadedRef.current) return
      try {
        const list = await fetchPlayers()
        if (cancelled) return
        const map: Record<string, Player> = {}
        const nowDrafted = new Set<string>()
        const newlyDrafted: Player[] = []
        for (const p of list) {
          map[p.player_id] = p
          if (p.drafted) {
            nowDrafted.add(p.player_id)
            if (!draftedIdsRef.current.has(p.player_id)) newlyDrafted.push(p)
          }
        }
        setPlayers(map)
        draftedIdsRef.current = nowDrafted
        if (newlyDrafted.length > 0) {
          const detectedAt = Date.now()
          setRecentPicks((prev) =>
            [
              ...newlyDrafted.map((p) => ({
                player_id: p.player_id, name: p.name, position: p.position,
                team: p.team, detectedAt,
              })),
              ...prev,
            ].slice(0, MAX_RECENT_PICKS)
          )
        }
      } catch {
        // Non-fatal: the join table goes stale until the next successful
        // poll picks the diff back up -- candidates and status still render.
      }
    }

    async function poll() {
      try {
        const data = await fetchLiveState()
        if (cancelled) return
        setState(data)
        setError(null)
        // Only re-fetch the player join table when a pick has actually
        // landed -- never on the bare 2500ms cadence. `picksMadeRef` starts
        // at null so the very first poll after mount never counts as "a
        // pick landed" on its own.
        if (picksMadeRef.current !== null && data.picks_made > picksMadeRef.current) {
          await refreshPlayersAndDiff()
        }
        picksMadeRef.current = data.picks_made
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

  const turnAnnouncement = !state?.active
    ? 'Not connected to a draft.'
    : draftDone
    ? 'Draft complete.'
    : youAreUp
    ? "You're up."
    : `Slot ${state.on_the_clock} is on the clock.`

  return (
    <div className="live-page">
      <div className="sr-only" aria-live="polite">{turnAnnouncement}</div>

      <header className={`live-strip${youAreUp ? ' live-strip-you' : ''}`}>
        <div className="live-strip-turn">
          {!state ? 'Loading…' : !state.active ? 'Not connected' : draftDone ? 'Draft complete' : youAreUp ? "You're up" : `Slot ${state.on_the_clock} on the clock`}
        </div>
        {state?.active && (
          <div className="live-strip-meta mono">
            <span>Pick {state.picks_made + 1}</span>
            <span>{state.picks_made} made</span>
            <span title={state.last_poll_at ?? undefined}>updated {pollAgeLabel(state.last_poll_at, nowMs)}</span>
          </div>
        )}
      </header>

      {error && <p className="error live-banner">{error}</p>}

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

          <div className="live-body">
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
                          <span className="live-alt-bar-fill" style={{ width: `${Math.max(0, Math.min(100, c.applied_pct))}%` }} />
                        </span>
                        <span className="live-alt-pct mono">{Math.round(c.applied_pct)}%</span>
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </div>

            <aside className="live-offboard">
              <h2>Off the board</h2>
              {recentPicks.length === 0 ? (
                <p className="rail-empty">No picks observed yet.</p>
              ) : (
                <ul className="live-offboard-list">
                  {recentPicks.map((p) => (
                    <li className="live-offboard-item" key={`${p.player_id}-${p.detectedAt}`}>
                      {posBadge(p.position)}
                      <span className="live-offboard-name">{p.name}</span>
                      <span className="live-offboard-team">{p.team}</span>
                    </li>
                  ))}
                </ul>
              )}
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
