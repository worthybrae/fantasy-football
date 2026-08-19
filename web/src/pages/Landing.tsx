import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { connectWithToken, fetchConnectProgress, fetchLiveState,
         type ConnectProgress, type LiveState, type TokenConnectParams } from '../api'
import { BOOKMARKLET } from '../lib/bookmarklet'
import BoardPreview from '../components/BoardPreview'
import ConnectScreen from '../components/ConnectScreen'
import ReadinessStrip from '../components/ReadinessStrip'

// The front door, and the bookmarklet onboarding.
//
// A bookmarklet cannot be detected -- browsers hide the bookmarks bar from
// pages by design -- so there is nothing to poll for and no "is it installed?"
// gate. Instead the page always shows itself, and flips to "live" the moment
// the bookmarklet is actually used: clicking it on the ESPN draft page mints a
// token there and opens THIS page in a new window with the token in the URL
// hash. That hash is the only signal that matters (did it work), and it is
// self-announcing.
//
//   connecting -> arrived from the bookmarklet with a token: open the socket
//   idle       -> the default: pitch, prove, hand over the bookmarklet
//   live       -> a connect is already running: offer the board
//
// There is no separate `error` gate any more. The connect screen renders its
// own failure (ConnectScreen's Failed), because a failure that names the
// stage it died on and how far it got is worth more than a one-line band on
// a page the user was not looking at.
type Gate = 'connecting' | 'idle' | 'live'

// How often the connect screen asks the helper what it is doing. Far faster
// than the room's own 2.5s poll, and it can afford to be: GET
// /api/live/connect-progress touches no database at all (see its docstring),
// unlike /api/live/state, which runs a COUNT(*), a pick replay and the vor
// fallback ranking on every call. Fast enough that a stage lasting a couple
// of hundred milliseconds is still seen; slow enough to be nothing next to
// the work being reported on.
const PROGRESS_POLL_MS = 400

// The token params the bookmarklet packs into the hash of the window it opens.
// Returns null unless all four load-bearing fields are present -- season is
// optional-ish (the bookmarklet always sends it, but a connect can proceed on
// the ids alone) so it is read separately.
function tokenFromHash(hash: string) {
  const p = new URLSearchParams(hash.replace(/^#/, ''))
  const leagueId = p.get('leagueId')
  const teamId = p.get('teamId')
  const swid = p.get('swid')
  const token = p.get('token')
  if (!leagueId || !teamId || !swid || !token) return null
  return { leagueId, teamId, swid, token, season: p.get('season') || '' }
}

export default function Landing() {
  const [gate, setGate] = useState<Gate>('idle')
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<ConnectProgress | null>(null)
  const [live, setLive] = useState<LiveState | null>(null)
  const [sessionReady, setSessionReady] = useState(false)
  // Bumped by Retry. Every connect-owned piece of state is keyed off it, so a
  // retry starts from a blank screen rather than from the last one's rows.
  const [attempt, setAttempt] = useState(0)
  const navigate = useNavigate()

  // The token the bookmarklet delivered, kept for the whole session. It has
  // to outlive the hash (which is wiped on arrival, see below) because Retry
  // re-runs this exact connect -- there is no other copy of it anywhere, and
  // making the user click the bookmark again just to retry would be asking
  // them to fix something they cannot see.
  const paramsRef = useRef<TokenConnectParams | null>(null)
  // Which attempt has already been POSTed. A ref, not state: React's
  // StrictMode runs every effect twice in development, and without this the
  // second run fires a second connect that supersedes the first mid-build.
  const postedRef = useRef(-1)
  // Whether the record now on /api/live/connect-progress is OURS.
  //
  // The helper keeps the last connect's record after it finishes, and this
  // page starts polling before its own POST has reached the server -- so the
  // first read can be a FINISHED record belonging to an earlier attempt.
  // Caught in a real run: the second of two connects rendered the first
  // one's failed record, stopped polling on it, and showed a stage list that
  // was missing a row its own connect actually ran. A terminal record is
  // only believed once this page has seen its own connect in flight (a
  // `connecting` phase, which every connect spends seconds in -- 4-35s
  // measured) or its own POST has settled.
  const ownRecordRef = useRef(false)

  useEffect(() => {
    // Arrived from the bookmarklet? Keep the token and open the socket.
    const params = tokenFromHash(window.location.hash)
    if (params) {
      paramsRef.current = params
      setGate('connecting')
      // Drop the token out of the address bar immediately -- it should not sit
      // there, be bookmarked, or survive a refresh into a duplicate connect.
      window.history.replaceState(null, '', window.location.pathname)
    }
  }, [])

  // The connect itself: one POST per attempt, and it stays blocked for as
  // long as the work takes (4-35s measured, depending on the league). What
  // makes that bearable is the poll below, which runs concurrently against a
  // different endpoint -- the POST's own response still arrives only at the
  // end, and it is still the thing that decides success or failure.
  useEffect(() => {
    if (gate !== 'connecting' || !paramsRef.current) return
    if (postedRef.current === attempt) return
    postedRef.current = attempt
    connectWithToken(paramsRef.current)
      .then(() => { ownRecordRef.current = true; setSessionReady(true) })
      .catch((e) => {
        ownRecordRef.current = true
        setError(e instanceof Error ? e.message : String(e))
      })
  }, [gate, attempt])

  // What the connect is doing, while it does it. Stops the moment the record
  // settles (every stage terminal, or one of them failed) -- and then reads
  // the live state once, for the only fact on the handoff screen that no
  // connect stage discovers: the pick clock.
  useEffect(() => {
    if (gate !== 'connecting') return
    let cancelled = false
    let timer = 0
    let settledAt = 0

    const poll = async () => {
      try {
        const body = await fetchConnectProgress()
        if (cancelled) return
        const settled = body.phase === 'ready' || body.phase === 'failed'
        if (body.phase === 'connecting') ownRecordRef.current = true
        if (settled && !ownRecordRef.current) {
          // A finished record from an earlier connect, read before ours has
          // started. Not ours to show, and not a reason to stop.
          timer = window.setTimeout(poll, PROGRESS_POLL_MS)
          return
        }
        setProgress(body)
        if (settled && !settledAt) {
          settledAt = Date.now()
          fetchLiveState().then((s) => { if (!cancelled) setLive(s) }).catch(() => {})
        }
        // A short grace period rather than stopping dead on the first
        // terminal phase. `failed` is set the instant one stage fails, and
        // another can still be running at that moment -- the ranking pass
        // landed 50ms after a socket failure in a real run, and stopping on
        // the first terminal read left it drawn as a spinner that would
        // never resolve. Every stage terminal ends it immediately.
        const open = body.stages.some(
          (s) => s.status === 'pending' || s.status === 'running')
        if (settled && (!open || Date.now() - settledAt > 2000)) return
      } catch {
        // The helper is momentarily unreachable. Keep polling: the connect
        // itself is a separate request and is still running, and the POST is
        // what reports a real failure.
      }
      if (!cancelled) timer = window.setTimeout(poll, PROGRESS_POLL_MS)
    }
    poll()
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [gate, attempt])

  // A draft that is already running does not wait for anyone to read a
  // summary: if the socket says the draft has started, the pick clock is
  // ticking and the room is where the user needs to be. This is not a timer
  // or a minimum display time -- it fires only on a fact off the socket, and
  // only once everything is done.
  useEffect(() => {
    if (progress?.phase === 'ready' && live?.draft_started) navigate('/draft')
  }, [progress?.phase, live?.draft_started, navigate])

  // No token in the hash: this is the landing view. Still poll in case a
  // connect is already running from a click in another window -- then this
  // page can offer the board rather than pitch a tool already in use.
  useEffect(() => {
    if (gate !== 'idle') return
    let cancelled = false
    const poll = async () => {
      try {
        const state = await fetchLiveState()
        if (!cancelled && (state.active || state.token_received)) setGate('live')
      } catch {
        /* helper not up yet; the page simply stays on the landing view */
      }
    }
    poll()
    const id = setInterval(poll, 2500)
    return () => { cancelled = true; clearInterval(id) }
  }, [gate])

  const retry = useCallback(() => {
    setError(null)
    setProgress(null)
    setLive(null)
    setSessionReady(false)
    // The record on the server is still the failed/degraded one this retry is
    // replacing, so it has to be disowned too -- otherwise the new attempt
    // renders the old attempt's ending for its first few hundred ms.
    ownRecordRef.current = false
    setAttempt((n) => n + 1)
  }, [])

  // The screen this whole task is about. It replaces the page rather than
  // sitting above it, and it stays up until the work is genuinely finished --
  // including the two stages that land after the connect returns (the socket
  // handshake and the first ranking pass).
  if (gate === 'connecting') {
    return (
      <ConnectScreen
        progress={progress ?? {
          // Before the first poll lands there is genuinely nothing to report.
          // An empty list draws the header and an empty rule, not invented rows.
          phase: 'connecting', stages: [], facts: {}, error: null, elapsed_ms: 0,
        }}
        live={live}
        sessionReady={sessionReady}
        error={error}
        onEnter={() => navigate('/draft')}
        onRetry={retry}
        onBack={() => { paramsRef.current = null; setGate('idle') }}
      />
    )
  }

  return (
    <main className="landing">
      <header className="landing-bar">
        <span className="landing-mark">
          <span className="landing-pip" aria-hidden="true" />
          Draft Helper
        </span>
        <span className="landing-where mono">running on this machine</span>
      </header>

      {gate === 'live' && (
        <div className="landing-band landing-band-live">
          <span>Your draft is synced.</span>
          <button className="landing-band-action" onClick={() => navigate('/draft')}>
            Go to the board
          </button>
        </div>
      )}

      {/* A failed connect no longer lands here as a one-line band. It keeps
          the connect screen, which can say which stage stopped, how far it
          got, and what to do about it -- see ConnectScreen's Failed. "Back to
          setup" is the way to this page, and it clears the token first so
          this view is the pitch again rather than a half-dead connect. */}
      {error !== null && gate === 'idle' && (
        <div className="landing-band landing-band-error">
          <span>Couldn’t connect: {error}</span>
          <button className="landing-band-action" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      <section className="landing-hero">
        <h1 className="landing-title">
          Your draft board,
          <br />
          on the clock.
        </h1>
        <p className="landing-lede">
          Click one bookmark inside your ESPN draft room and the board follows
          every pick — tiers, value over replacement, and who will not last
          until your next turn. Your ESPN password never leaves ESPN.
        </p>
      </section>

      <BoardPreview />

      <section className="landing-install">
        <div className="landing-install-cta">
          {/* The real javascript: link, present from first paint so a drag to
              the bookmarks bar copies IT, not this page's URL (setting the href
              after mount was too late -- the drag grabbed localhost instead).
              React strips javascript: from href props, so it is injected as raw
              HTML. The string has no single quotes (verified in
              bookmarklet.ts), so a single-quoted href is safe; onclick returns
              false so a stray click here does nothing -- it is a drag target,
              not a button. */}
          <span
            className="landing-bookmark-wrap"
            dangerouslySetInnerHTML={{
              __html:
                "<a class='landing-bookmark' title='Drag me to your bookmarks bar' "
                + "onclick='return false' href='" + BOOKMARKLET + "'>"
                + "<span aria-hidden='true'>⚓</span>&nbsp;Draft&nbsp;Helper</a>",
            }}
          />
          <span className="landing-bookmark-hint">← drag this to your bookmarks bar</span>
        </div>

        {/* Numbered because this genuinely is a sequence: each step is only
            possible once the one above it is done. */}
        <ol className="landing-steps">
          <li>
            <span className="landing-step-n mono">1</span>
            <span>
              Show your bookmarks bar if it’s hidden (<kbd>⌘⇧B</kbd> /{' '}
              <kbd>Ctrl⇧B</kbd>), then drag the button up to it.
            </span>
          </li>
          <li>
            <span className="landing-step-n mono">2</span>
            <span>Open your ESPN draft room. A mock draft works too.</span>
          </li>
          <li>
            <span className="landing-step-n mono">3</span>
            <span>
              Click <strong>⚓ Draft&nbsp;Helper</strong> there. Your board opens
              in a new window and starts following the draft.
            </span>
          </li>
        </ol>
      </section>

      <ReadinessStrip />
    </main>
  )
}
