import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { connectWithToken, fetchLiveState } from '../api'
import { BOOKMARKLET } from '../lib/bookmarklet'
import BoardPreview from '../components/BoardPreview'
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
//   error      -> the token connect failed; show why, let them retry
type Gate = 'connecting' | 'idle' | 'live' | 'error'

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
  const navigate = useNavigate()

  useEffect(() => {
    let cancelled = false

    // Arrived from the bookmarklet? Open the socket from the delivered token.
    const params = tokenFromHash(window.location.hash)
    if (params) {
      setGate('connecting')
      // Drop the token out of the address bar immediately -- it should not sit
      // there, be bookmarked, or survive a refresh into a duplicate connect.
      window.history.replaceState(null, '', window.location.pathname)
      connectWithToken(params)
        .then(() => { if (!cancelled) navigate('/draft') })
        .catch((e) => {
          if (cancelled) return
          setError(e instanceof Error ? e.message : String(e))
          setGate('error')
        })
      return () => { cancelled = true }
    }

    // No token in the hash: this is the landing view. Still poll in case a
    // connect is already running from a click in another window -- then this
    // page can offer the board rather than pitch a tool already in use.
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
  }, [navigate])

  // The one screen that replaces the page rather than sitting above it: it
  // lasts about a second and ends in a redirect, so there is nothing to read.
  if (gate === 'connecting') {
    return (
      <main className="landing landing-centered">
        <div className="landing-connecting">
          <span className="landing-spinner" aria-hidden="true" />
          <h1 className="landing-connecting-title">Connecting to your draft</h1>
          <p className="landing-lede">
            Opening the board from the token your browser just minted on ESPN.
          </p>
        </div>
      </main>
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

      {gate === 'error' && (
        <div className="landing-band landing-band-error">
          <span>Couldn’t connect: {error}</span>
          <button className="landing-band-action" onClick={() => setGate('idle')}>
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
