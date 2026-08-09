import { useEffect, useRef, useState } from 'react'
import {
  fetchDraftOrder, fetchLeague, fetchManagers, pollSim, saveDraftOrder, startSim,
  type DraftOrderEntry, type LeagueInfo, type Manager,
} from '../api'

// Coefficients worth showing on a card. The rest are position dummies that
// only make sense relative to each other, which a three-bar summary cannot
// convey honestly.
const SHOWN_FEATURES = ['reach', 'fall', 'need', 'run', 'qb_early', 'te_early']

// A real run_sim takes ~10s fitting manager models plus ~50s searching --
// comfortably under a minute, but nowhere near instant. Poll patiently, but
// not forever: if the run_id never resolves (server restart mid-run, a
// worker thread that silently died), give up rather than spin quietly for
// the rest of the session.
const POLL_INTERVAL_MS = 1000
const POLL_TIMEOUT_MS = 5 * 60 * 1000
// A single failed poll is more likely a network blip than a dead run --
// retry a few times before surfacing an error instead of aborting on the
// first hiccup.
const MAX_CONSECUTIVE_POLL_FAILURES = 4

// Round-and-pick notation for the next turn, from the snake order: pick
// `n` (1-indexed) in a `teams`-team league is round floor((n-1)/teams)+1,
// pick ((n-1)%teams)+1 -- matches scoring/draft_sim.py's snake_slots, e.g.
// for 8 teams pick 9 is round 2 slot 8, pick 16 is round 2 slot 1.
function nextPickLabel(slot: number, drafted: number, teams: number, rounds: number): string {
  for (let offset = drafted; offset < teams * rounds; offset++) {
    const round = Math.floor(offset / teams)
    const onClock = round % 2 === 0 ? (offset % teams) + 1 : teams - (offset % teams)
    if (onClock === slot) return `${round + 1}.${(offset % teams) + 1}`
  }
  return '—'
}

interface DraftRailProps {
  onSimComplete: () => void
  /** Header status text ("Slot 4 · next pick 2.13 · sim just now"). Never
   *  called until a run has actually started, so the header stays
   *  unchanged for a user who hasn't run a sim yet. */
  onStatus: (text: string) => void
  /** Number of players already marked drafted on the board -- owned by App
   *  (it has the player list), needed here only to find *this* slot's next
   *  turn in the snake order. */
  draftedCount: number
}

export default function DraftRail({ onSimComplete, onStatus, draftedCount }: DraftRailProps) {
  const [managers, setManagers] = useState<Manager[]>([])
  const [order, setOrder] = useState<DraftOrderEntry[]>([])
  const [mySlot, setMySlot] = useState<number | null>(null)
  const [league, setLeague] = useState<LeagueInfo | null>(null)
  const [rollouts, setRollouts] = useState(300)
  const [running, setRunning] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)
  // handleRun is a ~60s-running closure created at click time; reading
  // draftedCount through a ref (kept current on every render) rather than
  // the prop directly means the post-run label reflects however many picks
  // have been made by the time the run *finishes*, not however many had
  // been made when the button was clicked.
  const draftedCountRef = useRef(draftedCount)
  draftedCountRef.current = draftedCount

  useEffect(() => {
    Promise.all([fetchManagers(), fetchDraftOrder(), fetchLeague()])
      .then(([m, o, l]) => {
        setManagers(m)
        setOrder(o.order)
        setMySlot(o.my_slot)
        setLeague(l)
      })
      .catch((e) => setError(e instanceof Error ? e.message : 'Load failed'))
  }, [])

  function label(slot: number): string {
    return league ? nextPickLabel(slot, draftedCountRef.current, league.teams, league.rounds) : '—'
  }

  async function handleRun() {
    if (mySlot === null) return
    const slot = mySlot
    setRunning(true)
    setError(null)
    setElapsed(0)
    onStatus(`Slot ${slot} · simulating… 0s`)
    const tick = window.setInterval(() => {
      setElapsed((s) => {
        const next = s + 1
        onStatus(`Slot ${slot} · simulating… ${next}s`)
        return next
      })
    }, 1000)
    try {
      await saveDraftOrder(order, slot)
      const runId = await startSim(slot, rollouts)
      // The sim runs in a background thread server-side (~60s for a real
      // run: ~10s fitting manager models, ~50s searching) -- poll rather
      // than hold a request open for the length of the run.
      const deadline = Date.now() + POLL_TIMEOUT_MS
      let consecutiveFailures = 0
      for (;;) {
        if (Date.now() > deadline) {
          throw new Error('Simulation is taking longer than expected (5+ minutes) -- giving up on this poll, but it may still finish server-side.')
        }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS))
        let status
        try {
          status = await pollSim(runId)
        } catch (pollErr) {
          consecutiveFailures += 1
          if (consecutiveFailures >= MAX_CONSECUTIVE_POLL_FAILURES) throw pollErr
          continue // transient network blip -- retry rather than abort
        }
        consecutiveFailures = 0
        if (status.status === 'done') break
        if (status.status === 'error') throw new Error(status.detail ?? 'Simulation failed')
      }
      onSimComplete()
      onStatus(`Slot ${slot} · next pick ${label(slot)} · sim just now`)
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Simulation failed'
      setError(message)
      onStatus(`Slot ${slot} · sim failed`)
    } finally {
      window.clearInterval(tick)
      setRunning(false)
    }
  }

  const names = managers.map((m) => m.manager)
  const byName = new Map(managers.map((m) => [m.manager, m]))

  return (
    <aside className="draft-rail">
      <section className="rail-section">
        <h2>Draft order</h2>
        {order.length === 0 && <p className="rail-empty">Run `make espn-import` to load your league.</p>}
        <ol className="order-list">
          {order.map((entry, i) => (
            <li key={entry.slot}>
              <span className="slot-no">{entry.slot}</span>
              <select
                value={entry.manager}
                onChange={(e) => {
                  const next = [...order]
                  next[i] = { ...entry, manager: e.target.value }
                  setOrder(next)
                }}
              >
                {names.map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
              <label className="is-me">
                <input
                  type="radio"
                  name="my-slot"
                  checked={mySlot === entry.slot}
                  onChange={() => setMySlot(entry.slot)}
                />
                me
              </label>
            </li>
          ))}
        </ol>
        <div className="rail-run">
          <label>
            Rollouts{' '}
            <input
              type="number"
              min={10}
              step={50}
              value={rollouts}
              onChange={(e) => setRollouts(Number(e.target.value))}
            />
          </label>
          <button onClick={handleRun} disabled={running || mySlot === null}>
            {running ? `Simulating… ${elapsed}s` : 'Run simulation'}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="rail-section">
        <h2>Managers</h2>
        {order.length === 0 && <p className="rail-empty">Nothing to show yet.</p>}
        {order.map((entry) => {
          const m = byName.get(entry.manager)
          if (!m) return null
          const shown = m.coefficients
            .filter((c) => SHOWN_FEATURES.includes(c.feature))
            .sort((a, b) => Math.abs(b.value - b.pooled_value) - Math.abs(a.value - a.pooled_value))
            .slice(0, 3)
          return (
            <article key={entry.slot} className="manager-card">
              <header>
                <span className="slot-no">{entry.slot}</span>
                <strong>{m.manager}</strong>
                <span className={m.uses_personal ? 'rail-chip rail-chip-personal' : 'rail-chip rail-chip-pooled'}>
                  {m.uses_personal ? 'personal model' : 'league average'}
                </span>
              </header>
              <p className="manager-summary">{m.summary}</p>
              <ul className="coef-bars">
                {shown.map((c) => {
                  const delta = c.value - c.pooled_value
                  const width = Math.min(100, Math.abs(delta) * 40)
                  return (
                    <li key={c.feature}>
                      <span className="coef-name">{c.feature}</span>
                      <span className="coef-track">
                        <span
                          className={delta >= 0 ? 'coef-fill pos' : 'coef-fill neg'}
                          style={{ width: `${width}%` }}
                        />
                      </span>
                    </li>
                  )
                })}
              </ul>
              <footer>{m.n_picks} picks</footer>
            </article>
          )
        })}
      </section>
    </aside>
  )
}
