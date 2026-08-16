import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { connectWithToken, fetchLiveState } from '../api'
import { BOOKMARKLET } from '../lib/bookmarklet'

// The bookmarklet onboarding. A bookmarklet cannot be detected -- browsers
// hide the bookmarks bar from pages by design -- so there is nothing to poll
// for and no "is it installed?" gate. Instead the page shows the install-and-
// use guide by default and flips to "live" the moment the bookmarklet is
// actually used: clicking it on the ESPN draft page mints a token there and
// opens THIS page in a new window with the token in the URL hash. That hash is
// the only signal that matters (did it work), and it is self-announcing.
//
//   connecting -> arrived from the bookmarklet with a token: open the socket
//   install    -> the default: drag the button, open your draft, click it
//   live       -> a previous connect is already running: offer the board
//   error      -> the token connect failed; show why, let them retry
type Gate = 'connecting' | 'install' | 'live' | 'error'

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

export default function Connect() {
  const [gate, setGate] = useState<Gate>('install')
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

    // No token in the hash: this is the install/landing view. Still poll once
    // in case a connect is already running from a click in another window --
    // then this page can just offer the board rather than re-installing.
    const poll = async () => {
      try {
        const state = await fetchLiveState()
        if (!cancelled && (state.active || state.token_received)) setGate('live')
      } catch {
        /* helper not up yet; the gate simply stays on the install guide */
      }
    }
    poll()
    const id = setInterval(poll, 2500)
    return () => { cancelled = true; clearInterval(id) }
  }, [navigate])

  return (
    <main className="connect">
      <div className="connect-card">
        <h1 className="connect-kicker">
          <span className="connect-pip" aria-hidden="true" /> Draft Helper
        </h1>

        {gate === 'connecting' && (
          <>
            <h2 className="connect-title">Connecting to your draft…</h2>
            <p className="connect-lede">
              Opening the board from the token your browser just minted on ESPN.
            </p>
            <div className="connect-waiting">
              <span className="connect-spinner" aria-hidden="true" />
              One moment
            </div>
          </>
        )}

        {gate === 'install' && (
          <>
            <h2 className="connect-title">Add the Draft Helper button</h2>
            <p className="connect-lede">
              One drag, then every draft is a single click — no install, no
              login, nothing to paste. Your ESPN password never leaves ESPN.
            </p>

            <div className="connect-bookmark-row">
              {/* The real javascript: link, present from first paint so a drag
                  to the bookmarks bar copies IT, not this page's URL (setting
                  the href after mount was too late -- the drag grabbed
                  localhost instead). React strips javascript: from href props,
                  so it is injected as raw HTML. The string has no single quotes
                  (verified in bookmarklet.ts), so a single-quoted href is safe;
                  onclick returns false so a stray click here does nothing --
                  it is a drag target, not a button. */}
              <span
                className="connect-bookmark-wrap"
                dangerouslySetInnerHTML={{
                  __html:
                    "<a class='connect-bookmark' title='Drag me to your bookmarks bar' "
                    + "onclick='return false' href='" + BOOKMARKLET + "'>"
                    + "<span aria-hidden='true'>⚓</span>&nbsp;Draft&nbsp;Helper</a>",
                }}
              />
              <span className="connect-bookmark-hint">
                ← drag this to your bookmarks bar
              </span>
            </div>

            <ol className="connect-steps">
              <li>
                Show your bookmarks bar if it’s hidden
                (<kbd>⌘⇧B</kbd> / <kbd>Ctrl⇧B</kbd>), then drag the button up
                to it.
              </li>
              <li>Open your ESPN draft room (a mock draft works too).</li>
              <li>
                Click <strong>⚓ Draft&nbsp;Helper</strong> in your bookmarks
                bar. Your board opens in a new window.
              </li>
            </ol>

            <p className="connect-note">
              Nothing to detect and nothing to confirm — this page goes live on
              its own the moment you click the button in a draft.
            </p>
          </>
        )}

        {gate === 'live' && (
          <>
            <h2 className="connect-title">Draft synced</h2>
            <p className="connect-lede">Your board is live.</p>
            <button className="connect-submit" onClick={() => navigate('/draft')}>
              Go to the board
            </button>
          </>
        )}

        {gate === 'error' && (
          <>
            <h2 className="connect-title">Couldn’t connect</h2>
            <p className="connect-lede">{error}</p>
            <button className="connect-submit" onClick={() => setGate('install')}>
              Back to setup
            </button>
          </>
        )}

        <p className="connect-legacy">
          Looking for the research tool? It’s at <Link to="/legacy">/legacy</Link>.
        </p>
      </div>
    </main>
  )
}
