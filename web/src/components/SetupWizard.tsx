import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchUpcomingDrafts, type UpcomingDrafts } from '../api'
import { bookmarkAnchor, BookmarkCopy, barKeys, Keys, NO_DRAG_LINE, useCanDrag } from './BookmarkChip'
import { previewingSignedOut } from './Dashboard'

// The guided setup: the bookmarklet install walked one step at a time, drawn
// in the product's own language. The steps rail reads like a pick order --
// the step you are on is ON THE CLOCK, finished steps get the board's check
// -- because the page underneath is a draft room and an onboarding that
// looks like a different product is the templated thing this replaces.
//
// The one thing that makes this "premium" rather than a paged brochure is
// step three: it does not ask the user to report back. The bookmarklet's
// click connects the account server-side (POST /api/espn/connect, from the
// popup it opens), the custody cookie it sets is browser-wide, and that POST
// pops the drafts cache -- so this window simply polls GET /api/espn/drafts
// until `connected` flips, and the wizard finishes ITSELF the moment the
// click lands. A BroadcastChannel ping from the popup's own tab makes the
// common case (same browser, both tabs alive) instant instead of one poll
// late; the poll stays because the channel is an optimisation, not the
// mechanism.
//
// The poll runs for the wizard's whole life, not only on step three:
// somebody who installed the button last week and clicks it while reading
// step one has completed the setup, whatever page of it they were looking
// at, and holding them to the script would be the wizard refusing to notice
// its own success.

// The channel Landing's session-connect posts on when a bookmarklet popup
// lands an account. One name, both ends -- see the post in pages/Landing.tsx.
export const ACCOUNT_CHANNEL = 'draft-assistant-account'

const POLL_MS = 2500

type Step = 'bar' | 'install' | 'espn' | 'done'

const STEP_ORDER: Step[] = ['bar', 'install', 'espn', 'done']

const RAIL_LABELS: Record<Step, string> = {
  bar: 'Show the bookmarks bar',
  install: 'Put the button on it',
  espn: 'Click it on ESPN',
  done: 'Draft ready',
}

// Which browser is reading, for the one line that differs: where the
// bookmarks-bar setting lives in its menus. The keys themselves come from
// barKeys() in BookmarkChip. A wrong guess costs nothing -- every branch
// also names the menu path -- so this is allowed to be a userAgent sniff.
function barMenu(): string {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : ''
  if (/Safari\//.test(ua) && !/Chrome|Chromium|Edg\//.test(ua)) {
    return 'View → Always Show Favorites Bar'
  }
  if (/Firefox\//.test(ua)) {
    return /Mac/.test(ua)
      ? 'View → Toolbars → Bookmarks Toolbar'
      : 'Menu → Bookmarks → Show Bookmarks Toolbar'
  }
  return 'Bookmarks → Show Bookmarks Bar'
}

// -- the rail's glyphs: the board's check for a finished step, a plain mono
// number for one still to come. Shape, never colour alone, same rule as
// ConnectScreen's. --

function RailCheck() {
  return (
    <svg className="sw-check" viewBox="0 0 24 24" role="img" aria-label="done">
      <path d="M20 6L9 17l-5-5" />
    </svg>
  )
}

function Spinner() {
  return (
    <svg className="sw-spin" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3a9 9 0 0 1 9 9" />
    </svg>
  )
}

export interface SetupWizardProps {
  /** Close without finishing. Reopening starts over; every step is cheap. */
  onClose: () => void
  /** The poll saw `connected: true`: the drafts answer, for the page to
   *  remember (it becomes the dashboard the moment this closes). */
  onConnected: (body: UpcomingDrafts) => void
  /** The finish button: close and let the dashboard through. */
  onDone: () => void
}

export default function SetupWizard({ onClose, onConnected, onDone }: SetupWizardProps) {
  const [step, setStep] = useState<Step>('bar')
  const [leagues, setLeagues] = useState<number>(0)
  const canDrag = useCanDrag()
  const menu = barMenu()
  // The primary action of whichever step is showing. Focused on every step
  // change so the keyboard route is Enter, Enter, Enter rather than a tab
  // hunt through a dialog.
  const primaryRef = useRef<HTMLButtonElement | null>(null)

  // One poll loop for the wizard's whole life -- see the header comment for
  // why it is not gated on the ESPN step. `stepRef` keeps the loop from
  // being torn down and restarted on every step change.
  const doneRef = useRef(false)
  useEffect(() => {
    let cancelled = false
    let timer = 0
    const probe = async () => {
      try {
        const body = await fetchUpcomingDrafts()
        if (cancelled) return
        // `?signedout=1` promises the owner the page exactly as a visitor
        // gets it, and a visitor's poll answers false -- so the preview's
        // must be treated as false too, or the wizard finishes itself off
        // the owner's own local login the instant it opens.
        if (body.connected && !previewingSignedOut()) {
          doneRef.current = true
          setLeagues(body.leagues.length)
          setStep('done')
          onConnected(body)
          return
        }
      } catch {
        // The helper being briefly unreachable is not a failed setup; the
        // next tick asks again.
      }
      if (!cancelled && !doneRef.current) timer = window.setTimeout(probe, POLL_MS)
    }
    probe()
    // The popup's tab announcing the connect it just landed. Answer by
    // probing NOW rather than trusting the message with the state itself:
    // the server is the authority on whether this browser holds the cookie.
    let channel: BroadcastChannel | null = null
    if ('BroadcastChannel' in window) {
      channel = new BroadcastChannel(ACCOUNT_CHANNEL)
      channel.onmessage = () => {
        window.clearTimeout(timer)
        if (!doneRef.current) probe()
      }
    }
    return () => { cancelled = true; window.clearTimeout(timer); channel?.close() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Escape closes, the way every dialog does.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // The page does not move under the dialog; scrollbar width handed back as
  // padding so the fixed layout behind does not twitch. Same treatment, same
  // reasons, as Welcome's -- see the long comment there.
  useEffect(() => {
    const { body } = document
    const overflow = body.style.overflow
    const padding = body.style.paddingRight
    const bar = window.innerWidth - document.documentElement.clientWidth
    body.style.overflow = 'hidden'
    if (bar > 0) body.style.paddingRight = `${bar}px`
    return () => {
      body.style.overflow = overflow
      body.style.paddingRight = padding
    }
  }, [])

  useEffect(() => {
    primaryRef.current?.focus()
  }, [step])

  const back = useCallback(() => {
    setStep((s) => STEP_ORDER[Math.max(0, STEP_ORDER.indexOf(s) - 1)])
  }, [])

  const at = STEP_ORDER.indexOf(step)

  return (
    <div className="sw-scrim" role="dialog" aria-modal="true" aria-labelledby="sw-title"
         onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="sw-card">
        <button type="button" className="sw-x" onClick={onClose} aria-label="Close setup">
          ×
        </button>
        <p className="sw-eyebrow">
          Guided setup <span className="sw-eyebrow-dim">· about two minutes</span>
        </p>

        <div className="sw-body">
          {/* The pick order of the setup. aria-current marks the row the
              pane is showing; the pane itself is the live region, not this. */}
          <ol className="sw-rail">
            {STEP_ORDER.map((s, i) => {
              const state = i < at ? 'is-done' : i === at ? 'is-now' : 'is-todo'
              return (
                <li key={s} className={`sw-rail-row ${state}`}
                    aria-current={i === at ? 'step' : undefined}>
                  <span className="sw-rail-mark">
                    {i < at || (s === 'done' && step === 'done')
                      ? <RailCheck />
                      : <span className="sw-rail-num mono">{i + 1}</span>}
                  </span>
                  <span className="sw-rail-label">{RAIL_LABELS[s]}</span>
                </li>
              )
            })}
          </ol>

          <div className="sw-pane">
            {step === 'bar' && (
              <>
                <p className="sw-cap">Step 1 of 3</p>
                <h2 className="sw-title" id="sw-title">Show your bookmarks bar.</h2>
                <p className="sw-copy">
                  The button installs on your bookmarks bar, the strip under
                  the address bar. If the bar is hidden, press{' '}
                  <Keys keys={barKeys()} /> or use <em>{menu}</em>.
                </p>
                <p className="sw-copy sw-copy-dim">
                  It is only a bookmark. Nothing runs until you click it.
                </p>
                <div className="sw-actions">
                  <button ref={primaryRef} type="button" className="sw-next"
                          onClick={() => setStep('install')}>
                    The bar is visible
                  </button>
                </div>
              </>
            )}

            {step === 'install' && (
              <>
                <p className="sw-cap">Step 2 of 3</p>
                <h2 className="sw-title" id="sw-title">Drag the button to the bar.</h2>
                {!canDrag && <p className="sw-copy">{NO_DRAG_LINE}</p>}
                <div className="sw-chip-row">
                  <span dangerouslySetInnerHTML={{ __html: bookmarkAnchor('sw-copy-hint') }} />
                  {canDrag && <span className="sw-chip-hint">← drag this to your bookmarks bar</span>}
                </div>
                <BookmarkCopy idPrefix="sw" />
                <div className="sw-actions">
                  <button type="button" className="sw-back" onClick={back}>Back</button>
                  <button ref={primaryRef} type="button" className="sw-next"
                          onClick={() => setStep('espn')}>
                    It’s on my bar
                  </button>
                </div>
              </>
            )}

            {step === 'espn' && (
              <>
                <p className="sw-cap">Step 3 of 3</p>
                <h2 className="sw-title" id="sw-title">Click the bookmark on ESPN.</h2>
                <p className="sw-copy">
                  Open ESPN fantasy and sign in. Any page works. Click{' '}
                  <span className="sw-chip-mini">🏈 Draft Assistant</span> on
                  your bookmarks bar. Keep this window open.
                </p>
                <a className="sw-espn" href="https://fantasy.espn.com/"
                   target="_blank" rel="noopener noreferrer">
                  Open ESPN fantasy ↗
                </a>
                {/* The wizard's whole promise, in one honest row: it is
                    watching, and it will finish itself. */}
                <p className="sw-listen" role="status" aria-live="polite">
                  <Spinner />
                  Waiting for your click. This step completes on its own.
                </p>
                <div className="sw-actions">
                  <button ref={primaryRef} type="button" className="sw-back" onClick={back}>
                    Back
                  </button>
                </div>
              </>
            )}

            {step === 'done' && (
              <>
                <p className="sw-cap sw-cap-ok">Setup complete</p>
                <h2 className="sw-title" id="sw-title">Connected.</h2>
                <p className="sw-copy" role="status" aria-live="polite">
                  {leagues > 0
                    ? <>Your account is linked. It has <strong>{leagues} league{leagues === 1 ? '' : 's'}</strong>. Each league gets a Join button when its draft room opens.</>
                    : <>Your account is linked. It has no upcoming drafts yet. New drafts appear here with a Join button.</>}
                </p>
                <p className="sw-copy sw-copy-dim">
                  Setup is done. One click on the bookmark from any ESPN page
                  opens this app.
                </p>
                <div className="sw-actions">
                  <button ref={primaryRef} type="button" className="sw-next" onClick={onDone}>
                    See your drafts
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
