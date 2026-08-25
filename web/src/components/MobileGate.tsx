import { useEffect, useState, useSyncExternalStore, type ReactNode } from 'react'
import { Logo } from './Logo'
import { fetchMarketOverview, type MarketOverview } from '../api'
import './mobileGate.css'

// The whole app is desktop only, and this says so instead of breaking.
//
// The assistant runs beside ESPN's draft room in a desktop browser -- the
// bookmarklet is clicked there, the room is drafted there -- so a phone has
// no path through the product at all. Left alone, the landing page at 390px
// laid its live draft board out as a six-thousand-pixel column of pick
// numbers and every other route spilled off the right edge. A visitor from a
// Reddit thread would read that as "broken", not "desktop".
//
// So under the page's own phone breakpoint every route yields to this:
// the name, one line saying where it does work, a recording of the real
// room so they can see what they are being sent to, and a way to get the
// link onto their computer.
//
// WHY 719px. landing.css already treats that width as "phone" (the setup
// wizard's rail lies down there); one number to move if the line moves.
//
// WHY A RECORDING, WHEN DemoRoom.tsx argues against one. That argument is
// for the desktop page, where the live room itself can be shown. Here the
// live room cannot be laid out at all, so a recording of it is the honest
// fallback -- it just has to be re-recorded when the room changes
// (scripts/record_demo.py does it against the dev server).

const QUERY = '(max-width: 719px)'

function subscribe(onChange: () => void) {
  const mql = window.matchMedia(QUERY)
  mql.addEventListener('change', onChange)
  return () => mql.removeEventListener('change', onChange)
}
const getSnapshot = () => window.matchMedia(QUERY).matches
const getServerSnapshot = () => false

function usePhone(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
}

export default function MobileGate({ children }: { children: ReactNode }) {
  const phone = usePhone()
  if (!phone) return <>{children}</>
  return <GatePage />
}

// The count is the archive's own, the same number the desktop page prints
// exactly, floored to the thousand and given a plus: it grows on its own and
// never says more than is true. No count, no line -- a placeholder figure
// would be the one thing on this page a reader could check and find wrong.
function corpusLine(o: MarketOverview): string | null {
  if (o.picks < 1000) return null
  const picks = Math.floor(o.picks / 1000) * 1000
  return `${picks.toLocaleString()}+ picks watched across ${o.drafts.toLocaleString()} real ESPN drafts`
}

function GatePage() {
  const url = `${window.location.origin}/`
  const [sent, setSent] = useState<'idle' | 'copied' | 'shown'>('idle')
  const [overview, setOverview] = useState<MarketOverview | null>(null)
  useEffect(() => {
    let live = true
    fetchMarketOverview().then((o) => { if (live) setOverview(o) }, () => {})
    return () => { live = false }
  }, [])
  const corpus = overview ? corpusLine(overview) : null

  async function send() {
    // The share sheet is the phone's own way of getting something to a
    // computer (AirDrop, Notes, mail to self); the clipboard is the fallback,
    // and the URL printed underneath is the fallback for that.
    if (navigator.share) {
      try {
        await navigator.share({ title: 'ESPN Draft Assist', url })
      } catch {
        /* the sheet was closed; nothing to do */
      }
      return
    }
    try {
      await navigator.clipboard.writeText(url)
      setSent('copied')
    } catch {
      setSent('shown')
    }
  }

  return (
    <main className="mg">
      <header className="mg-brand">
        <Logo size={18} />
        <span className="mg-name">ESPN Draft Assist</span>
      </header>

      <h1 className="mg-title">Built for a desktop browser.</h1>
      <p className="mg-lede">
        It runs beside ESPN&rsquo;s draft room on your computer, live, while you
        draft. There is nothing it can do from a phone &mdash; but here is what
        it looks like there.
      </p>
      {corpus && <p className="mg-corpus">{corpus}</p>}

      <figure className="mg-figure">
        <video
          className="mg-video"
          src="/demo.webm"
          poster="/demo-poster.jpg"
          autoPlay
          muted
          loop
          playsInline
          preload="metadata"
        />
        <figcaption className="mg-caption">
          A real ESPN mock draft, the room&rsquo;s own pick priced by
          simulating the rest of the draft.
        </figcaption>
      </figure>

      <button type="button" className="mg-send" onClick={send}>
        {sent === 'copied' ? 'Link copied' : 'Send yourself the link'}
      </button>
      <p className="mg-url mono" aria-live="polite">
        {sent === 'shown' ? url : url.replace(/^https?:\/\//, '')}
      </p>
    </main>
  )
}
