import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { fetchLiveState } from '../api'
import { detectExtension, STORE_URL } from '../lib/extension'

// Onboarding gate. With the browser extension delivering the draft token,
// there is no URL to type and no form to fill -- the page only has to work
// out which of three states the user is in and show the one next step.
//
//   missing  -> install the extension (a button to the store)
//   ready    -> installed, but no draft synced yet: open ESPN, click the icon
//   live     -> a token has arrived; go to the board
//
// The URL field the old connect screen had is gone on purpose: the extension
// reads leagueId/teamId off the ESPN tab and posts them with the token, so
// the server already knows the draft. The field only ever existed because the
// browser-window path had no other way to learn it.
type Gate = 'checking' | 'missing' | 'ready' | 'live'

export default function Connect() {
  const [gate, setGate] = useState<Gate>('checking')
  const [extVersion, setExtVersion] = useState<string | null>(null)
  const navigate = useNavigate()

  // Detect the extension once, then poll live state for a delivered token.
  useEffect(() => {
    let cancelled = false

    detectExtension().then((version) => {
      if (cancelled) return
      setExtVersion(version)
      setGate(version ? 'ready' : 'missing')
    })

    // Only meaningful once the extension is present, but harmless before:
    // token_received stays false until a sync lands.
    const poll = async () => {
      try {
        const state = await fetchLiveState()
        if (!cancelled && state.token_received) setGate('live')
      } catch {
        /* helper not up yet; the gate simply stays where it is */
      }
    }
    poll()
    const id = setInterval(poll, 2500)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  return (
    <main className="connect">
      <div className="connect-card">
        <h1 className="connect-kicker">
          <span className="connect-pip" aria-hidden="true" /> Draft Helper
        </h1>

        {gate === 'checking' && (
          <>
            <h2 className="connect-title">Getting ready…</h2>
            <p className="connect-lede">Checking for the browser extension.</p>
          </>
        )}

        {gate === 'missing' && (
          <>
            <h2 className="connect-title">Add the extension</h2>
            <p className="connect-lede">
              One install, then every draft is a single click — no login, no
              password, nothing to paste. The extension only ever reads a
              token specific to each draft.
            </p>
            <a className="connect-submit" href={STORE_URL} target="_blank" rel="noreferrer">
              Add to Chrome
            </a>
            <p className="connect-note">
              After installing, this page picks it up automatically. Using
              Chrome, Edge, Brave, or another Chromium browser.
            </p>
          </>
        )}

        {gate === 'ready' && (
          <>
            <h2 className="connect-title">You're set</h2>
            <p className="connect-lede">
              Open your ESPN draft, then click the{' '}
              <strong className="connect-anchor">⚓</strong> Draft Helper icon in
              your toolbar. Your board goes live here the moment you do.
            </p>
            <div className="connect-waiting">
              <span className="connect-spinner" aria-hidden="true" />
              Waiting for a draft to sync…
            </div>
            {extVersion && (
              <p className="connect-note">Extension v{extVersion} detected.</p>
            )}
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

        <p className="connect-legacy">
          Looking for the research tool? It's at <Link to="/legacy">/legacy</Link>.
        </p>
      </div>
    </main>
  )
}
