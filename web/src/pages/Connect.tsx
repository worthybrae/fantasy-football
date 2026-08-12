import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { connectDraft, fetchLiveState } from '../api'

type Phase = 'form' | 'connecting' | 'connected'

// Landing screen: the first thing anyone sees. One input (the draft URL),
// one optional fallback (slot), one button. The backend accepts any ESPN
// URL that carries a leagueId -- draft room, waiting room, lobby, wherever
// the drafter happens to be -- and navigates to exactly that page itself
// (see api/live.py's DraftListener), so this screen deliberately doesn't
// steer anyone toward one ESPN page over another.
export default function Connect() {
  const [url, setUrl] = useState('')
  const [slotInput, setSlotInput] = useState('')
  const [phase, setPhase] = useState<Phase>('form')
  const [error, setError] = useState<string | null>(null)
  const [leagueId, setLeagueId] = useState<string | null>(null)
  const [mySlot, setMySlot] = useState<number | null>(null)
  const [slotUnknown, setSlotUnknown] = useState(false)
  const navigate = useNavigate()

  const busy = phase === 'connecting'

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const trimmed = url.trim()
    if (!trimmed || busy) return
    setError(null)
    setPhase('connecting')
    const slot = slotInput.trim() ? Number(slotInput) : null
    try {
      const result = await connectDraft(trimmed, slot)
      setLeagueId(result.league_id)
      // POST /api/live/connect resolves my_slot server-side but does not
      // hand it back -- it's only on GET /api/live/state, which is safe to
      // read immediately after because the session is stored before connect
      // responds. This is the one number a re-randomized draft order could
      // get silently wrong (see task-2 dispatch), so it's worth a second
      // round trip to show it rather than skip straight to the draft room.
      try {
        const state = await fetchLiveState()
        setMySlot(state.my_slot)
        setSlotUnknown(state.my_slot === null)
      } catch {
        setMySlot(null)
        setSlotUnknown(true)
      }
      setPhase('connected')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not connect to that draft')
      setPhase('form')
    }
  }

  function reconnect() {
    setPhase('form')
    setError(null)
  }

  if (phase === 'connected') {
    return (
      <main className="connect">
        <div className="connect-card connect-confirm">
          <p className="connect-eyebrow">Connected · league {leagueId}</p>
          {slotUnknown ? (
            <p className="connect-slot-unknown">
              Couldn't confirm a draft slot for this connection. You can still
              enter the draft room, or reconnect and set a slot by hand below.
            </p>
          ) : (
            <p className="connect-slot-line">
              You're drafting from slot <strong className="connect-slot-value">{mySlot}</strong>.
            </p>
          )}
          <p className="connect-slot-hint">
            Not your slot? Reconnect and enter it directly -- the auto-detect
            reads it off last season's draft, and a league that re-shuffled
            its order this year would get this wrong.
          </p>
          <div className="connect-actions">
            <button type="button" className="connect-submit" onClick={() => navigate('/draft')}>
              Enter draft room
            </button>
            <button type="button" className="connect-secondary" onClick={reconnect}>
              Reconnect
            </button>
          </div>
        </div>
      </main>
    )
  }

  return (
    <main className="connect">
      <div className="connect-card">
        <p className="connect-eyebrow">
          <span className="connect-dot" aria-hidden="true" />
          Draft helper
        </p>
        <h1>Sync to your draft</h1>
        <p className="connect-lede">
          Paste the address of any ESPN page for your draft. As long as it
          carries a leagueId, this connects and takes you there.
        </p>
        <form onSubmit={handleSubmit}>
          <label htmlFor="draft-url">Draft URL</label>
          <input
            id="draft-url"
            value={url}
            autoFocus
            autoComplete="off"
            inputMode="url"
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://fantasy.espn.com/football/draft?leagueId=…"
            disabled={busy}
          />
          <label htmlFor="slot" className="connect-slot-label">
            Draft slot <span className="connect-optional">(only needed if the URL has no teamId)</span>
          </label>
          <input
            id="slot"
            className="connect-slot-input"
            type="number"
            min={1}
            max={20}
            value={slotInput}
            onChange={(e) => setSlotInput(e.target.value)}
            placeholder="auto"
            disabled={busy}
          />
          <button type="submit" className="connect-submit" disabled={busy || !url.trim()}>
            {busy ? 'Connecting…' : 'Sync to draft'}
          </button>
        </form>
        {error && (
          <p className="connect-error" role="alert">
            {error}
          </p>
        )}
        <p className="connect-legacy">
          Looking for the research tool? It's at <Link to="/legacy">/legacy</Link>.
        </p>
      </div>
    </main>
  )
}
