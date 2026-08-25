import { useEffect, useRef, useState } from 'react'
import { fetchBillingStatus, startCheckout } from '../api'

// THE ONE THING THAT COSTS MONEY, asked for at the only moment it makes
// sense to ask: the draft is about to open, the room is real, and nobody has
// paid for it. Not a pricing page, not a banner, not a modal on arrival.
//
// WHY A POPUP AND NOT A REDIRECT. The bookmarklet delivers a per-draft token
// in the URL hash, and Landing wipes that hash the moment it reads it (it can
// carry an account session, so it must not sit in an address bar). The only
// copy left is in memory on this page. Navigating to Stripe would throw it
// away and cost the user a second trip through the bookmark after paying, on
// a clock, with their draft running. A popup leaves this page standing.
//
// The parent then has to learn that the payment landed. Two ways, because
// neither is reliable alone: the success page posts a message to its opener
// (instant, but blocked if the popup was reparented or the user finished on
// their phone), and this polls the server (slower, but true regardless of
// what any browser did). Whichever arrives first wins.

/** How often to ask the server whether the payment has landed. Two seconds:
 *  the webhook usually beats the buyer back to the page, so this is mostly
 *  one poll, and a draft clock is running. */
const POLL_MS = 2000

/** How long to keep asking. Ten minutes is a card, a bank app, and a
 *  three-digit code typed in wrong twice -- past that, the tab has been
 *  abandoned and polling is just noise. */
const GIVE_UP_MS = 600_000

export default function Paywall({ leagueId, season, onPaid, onBack }: {
  leagueId: string
  season: number
  /** The payment landed. The caller retries whatever was refused. */
  onPaid: () => void
  /** Give up and go back, without paying. */
  onBack: () => void
}) {
  const [waiting, setWaiting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const popup = useRef<Window | null>(null)

  // Only while a checkout is actually open. A page that polls a billing
  // endpoint every two seconds forever is a page that will do it on somebody's
  // phone, in the background, for an hour.
  useEffect(() => {
    if (!waiting) return
    let cancelled = false
    const started = Date.now()
    let timer = 0

    const ask = async () => {
      if (cancelled) return
      const status = await fetchBillingStatus(leagueId, season)
      if (cancelled) return
      if (status.entitled) {
        popup.current?.close()
        onPaid()
        return
      }
      if (Date.now() - started > GIVE_UP_MS) {
        setWaiting(false)
        return
      }
      timer = window.setTimeout(ask, POLL_MS)
    }
    timer = window.setTimeout(ask, POLL_MS)

    // The fast path: the success page (see Landing) tells its opener the
    // moment it loads. Same origin, and checked, because a message handler
    // that trusts any sender is a message handler anybody can call.
    const onMessage = (e: MessageEvent) => {
      if (e.origin !== window.location.origin) return
      if (e.data === 'billing:paid') void ask()
    }
    window.addEventListener('message', onMessage)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
      window.removeEventListener('message', onMessage)
    }
  }, [waiting, leagueId, season, onPaid])

  async function pay() {
    setError(null)
    // OPENED SYNCHRONOUSLY, before the await. A window.open that happens
    // after a network round trip is not attributable to the click any more
    // and every popup blocker on earth stops it.
    const win = window.open('', 'stripe-checkout', 'width=480,height=720')
    popup.current = win
    try {
      const url = await startCheckout(leagueId, season, '/?paid=1')
      if (win) win.location.href = url
      // No popup (a blocker, or a browser that refuses them outright): this
      // tab goes to Stripe instead. It costs the bookmarklet token, which is
      // why it is the fallback rather than the plan.
      else window.location.href = url
      setWaiting(true)
    } catch (e) {
      win?.close()
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="pw">
      <div className="pw-card">
        <p className="draft-cap pw-eyebrow">This one is a real league</p>
        <h1 className="pw-title">Unlock this draft</h1>
        <p className="pw-body">
          Mock drafts are free, and always will be. A real league&rsquo;s draft is
          a one-off <strong>$9.99</strong> for the season: the live board, the
          survival model, and every panel behind it, for as long as this draft
          is running.
        </p>
        <p className="mono pw-league">
          league {leagueId} · {season}
        </p>

        {error && <p className="pw-error" role="alert">{error}</p>}

        {waiting ? (
          <>
            <p className="pw-waiting">
              <span className="db-dot" aria-hidden="true" />
              Waiting for Stripe. Finish in the other window; this page picks it
              up on its own.
            </p>
            <button type="button" className="pw-quiet" onClick={() => setWaiting(false)}>
              Cancel
            </button>
          </>
        ) : (
          <>
            <button type="button" className="pw-pay" onClick={pay}>
              Pay $9.99 and connect
            </button>
            <button type="button" className="pw-quiet" onClick={onBack}>
              Not now
            </button>
          </>
        )}

        <p className="pw-fine">
          Card handled by Stripe. Nothing about your card reaches this server.
        </p>
      </div>
    </div>
  )
}
