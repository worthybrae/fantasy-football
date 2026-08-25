import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchCustody, fetchLiveMock, fetchMarketOverview, fetchMarketSlot } from '../api'
import {
  VETERAN_SEASONS, benefits, findVeteran, type ShowcaseData,
} from './benefitFigures'
import { loadProfile } from './draft/CellTip'
import type { LiveMock, MarketSlot } from '../api'
import type { ProfilePayload } from './profile/payload'
import { Logo } from './Logo'

// The one piece of branding on a page that is otherwise entirely product.
//
// A visitor lands inside a live draft room. That is the right first
// impression and it has one flaw: nothing on screen says what this IS or what
// to do about it -- the room is somebody else's draft, and the only way in is
// a setup flow further down the page. So the page opens with a card that
// names the thing and offers the one action, over the room rather than
// instead of it.
//
// IT SHOWS EVERY VISIT, TO EVERY VISITOR WHO IS NOT CONNECTED. An earlier
// version remembered dismissal in `localStorage` and never introduced itself
// twice, which is the right instinct for a cookie banner and the wrong one
// here: somebody who closed it to watch a draft and came back a week later
// arrived at an unexplained draft room belonging to strangers. The one thing
// that suppresses it is having a session already -- there is nothing to get
// started with when you are started.
//
// AND IT NEVER TRAPS ANYBODY. Dismissing does not hide the offer, it MOVES
// it: the card shrinks to the bottom-right corner and stays there for the
// rest of the visit, so somebody who wanted to watch the draft first can
// start whenever they decide to. A modal that closes to nothing makes the
// reader hunt for the way in a second time.

/** How long a slide holds. Long enough to read three lines without hurrying,
 *  short enough that a reader who is not reading them still sees two or three
 *  before deciding. */
const SLIDE_MS = 4200

/** The six claims, one at a time.
 *
 *  AUTO-ADVANCING, WITH A BRAKE. It stops while the pointer is over it and
 *  while any of its controls has focus -- a slide that moved out from under
 *  somebody mid-sentence would be the most annoying thing on the page -- and
 *  it does not advance at all for a reader who has asked for less motion,
 *  who gets the dots and nothing that moves on its own.
 */
function Showcase({ data }: { data: ShowcaseData }) {
  const panels = benefits(data)
  const [at, setAt] = useState(0)
  const [held, setHeld] = useState(false)

  useEffect(() => {
    if (held) return
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
    const id = window.setInterval(
      () => setAt((i) => (i + 1) % panels.length), SLIDE_MS)
    return () => window.clearInterval(id)
  }, [held, panels.length])

  const panel = panels[at]
  return (
    <div
      className="wc-show"
      onMouseEnter={() => setHeld(true)}
      onMouseLeave={() => setHeld(false)}
      onFocus={() => setHeld(true)}
      onBlur={() => setHeld(false)}
    >
      {/* `key` on the slide, so React remounts it on every change -- the
          slide's own entrance replays, and the figure inside starts its loop
          from a clean phase rather than inheriting the last one's. */}
      <div className="wc-slide" key={panel.key}>
        <div className="wc-slide-figure">{panel.figure}</div>
        <p className="wc-slide-eyebrow">{panel.eyebrow}</p>
        <p className="wc-slide-body">{panel.body}</p>
      </div>
      <div className="wc-dots" role="tablist" aria-label="What it knows">
        {panels.map((p, i) => (
          <button
            key={p.key}
            type="button"
            role="tab"
            aria-selected={i === at}
            aria-label={p.title}
            className={`wc-dot${i === at ? ' is-on' : ''}`}
            onClick={() => setAt(i)}
          />
        ))}
      </div>
    </div>
  )
}

export default function Welcome({ onStart, live = false, askedAt = 0 }: {
  /** Takes the reader into setup. The page owns that flow -- this card only
   *  decides when to ask. */
  onStart: () => void
  /** Whether the room behind the card is a draft happening right now, or one
   *  from the archive being replayed because none is. It changes one clause
   *  of one sentence, and it has to: "running now" is the only claim on this
   *  page a reader could catch out, and they would catch it out on the
   *  timestamp of the picks in front of them. */
  live?: boolean
  /** Bumped by the page when something else on it needs an account -- the
   *  archive button under the corpus band, today. Raising the card again is
   *  the whole mechanism: it already says what an account is for, so the
   *  only thing added is the sentence naming what was just refused. */
  askedAt?: number
}) {
  const [corner, setCorner] = useState(false)
  const [drafts, setDrafts] = useState<number | null>(null)
  const [profile, setProfile] = useState<ProfilePayload | null>(null)
  const [veteran, setVeteran] = useState<ProfilePayload | null>(null)
  const [room, setRoom] = useState<LiveMock | null>(null)
  const [slot, setSlot] = useState<MarketSlot | null>(null)
  // Null until the probe answers. The card waits rather than guessing: shown
  // optimistically it would flash in the face of somebody already connected,
  // and `/api/espn/custody` is a local lookup with no ESPN call behind it, so
  // the wait is a millisecond rather than a request to a sports website.
  const [connected, setConnected] = useState<boolean | null>(null)
  // How many of the three data loads below have SETTLED -- landed or failed.
  // The card holds off until all three: it used to appear the moment the
  // custody probe answered and then fill in around the reader, which read as
  // a laggy card rather than a loading one. Settled, not succeeded: every
  // slide is written to play with fields absent, and a failed fetch must not
  // hide the only way in.
  const [loads, setLoads] = useState(0)
  const LOADS_NEEDED = 3
  const startRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchCustody()
      .then((body) => { if (!cancelled) setConnected(body.connected) })
      // A probe that fails is not a session. Showing the introduction to
      // somebody who has one costs a click; hiding it from somebody who does
      // not costs them the only way in.
      .catch(() => { if (!cancelled) setConnected(false) })
    return () => { cancelled = true }
  }, [])

  // The archive's own size, for the first slide. Fetched here rather than in
  // the slide so the number is in hand before it is its turn -- a figure that
  // counted up from nothing the moment it became visible would look like it
  // was still loading.
  useEffect(() => {
    let cancelled = false
    fetchMarketOverview()
      .then((body) => { if (!cancelled) setDrafts(body.drafts) })
      .catch(() => { /* the slide falls back to a round number */ })
      .finally(() => { if (!cancelled) setLoads((n) => n + 1) })
    return () => { cancelled = true }
  }, [])

  // THE PANELS ARE REAL, ON A REAL PLAYER. Three of the six claims are about
  // cards that live inside the player profile, which a visitor to this page
  // never opens: they can watch the room behind this card all afternoon and
  // never see the Vegas line, the graded schedule, or who a player is
  // comparable to. So the card shows the actual components, on whoever the
  // live room ranks first at this moment.
  //
  // Two hops, both cached and both cheap: the demo endpoint is already being
  // polled by the room behind this card, and `loadProfile` is the same cache
  // its hover panels use -- so a reader who later opens that player pays
  // nothing for it. Failure is silent by design: the fields play alone, and
  // the sentences still say what the product does.
  useEffect(() => {
    let cancelled = false
    let shortlist: { player_id: string }[] = []
    fetchLiveMock()
      .then((live) => {
        if (cancelled) return null
        setRoom(live)
        shortlist = live.shortlist ?? []
        const first = shortlist[0]?.player_id
        return first ? loadProfile(first) : null
      })
      .then(async (body) => {
        if (cancelled || !body) return
        const main = body as unknown as ProfilePayload
        setProfile(main)
        // A CAREER for the health panel. The room's top pick is regularly a
        // rookie, and a games-played chart with one column in it is a fair
        // reading of him and a poor showing for the panel. Walk down the
        // board until somebody has several seasons on record, at most a few
        // players deep, and stop at the first one. AWAITED, so the card's
        // hold-until-ready gate covers it: the health slide opening on one
        // player and swapping to another mid-read was the exact lag the
        // gate exists to remove.
        if (main.seasons.length >= VETERAN_SEASONS) {
          setVeteran(main)
          return
        }
        const found = await findVeteran(shortlist)
        if (!cancelled && found) setVeteran(found)
      })
      .catch(() => { /* the fields carry the slides on their own */ })
      .finally(() => { if (!cancelled) setLoads((n) => n + 1) })
    fetchMarketSlot(1)
      .then((body) => { if (!cancelled) setSlot(body) })
      .catch(() => { /* the archive slide then shows its field alone */ })
      .finally(() => { if (!cancelled) setLoads((n) => n + 1) })
    return () => { cancelled = true }
  }, [])

  // Raised again on request. `askedAt` is a counter rather than a flag so a
  // second ask after a second dismissal still lands.
  useEffect(() => {
    if (askedAt > 0) setCorner(false)
  }, [askedAt])

  const dismiss = useCallback(() => setCorner(true), [])

  // Escape closes it, the way every dialog on the web does. Bound only while
  // the card is centred -- once it is a corner chip it is not a dialog and
  // must not swallow the key from anything else on the page.
  useEffect(() => {
    if (corner) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') dismiss()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [corner, dismiss])

  // THE PAGE DOES NOT MOVE UNDER THE CARD. The scrim covers the viewport but
  // a wheel over it still scrolled the document, so the pitch, the setup
  // steps and the footer slid past behind a dialog that was still asking the
  // reader to decide -- the room the card is talking about scrolled away and
  // the card stayed. While it is centred there is one thing to do; the page
  // is there once it is dismissed.
  //
  // Only the DOCUMENT is frozen. The room behind keeps its own scroll
  // containers (`.draft-main`, the ranked list), so a reader who wants to look
  // at the board before deciding still can -- which is exactly what the card's
  // second button offers.
  //
  // The scrollbar's own width is handed back as padding. Hiding the bar reflows
  // the whole page by however wide it is, and on a fixed layout like the room
  // that reads as everything twitching sideways the instant the card appears.
  // Only on a viewport that HAS an overlay-free scrollbar: the measurement is
  // 0 on macOS overlay bars and on touch, where padding would be the shift.
  //
  // Both properties are restored to whatever they were rather than cleared,
  // so nothing here can quietly take ownership of a style the page set itself.
  useEffect(() => {
    if (corner || connected === null || connected || loads < LOADS_NEEDED) return
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [corner, connected, loads])

  // The primary action takes focus, so the keyboard route is one key rather
  // than a tab through the whole draft room behind it. `loads` is a dep for
  // the same reason it gates the render: the button does not exist until the
  // card does.
  useEffect(() => {
    if (!corner) startRef.current?.focus()
  }, [corner, loads])

  // Already connected: no introduction and no offer, because there is
  // nothing to start. The room is just a room.
  if (connected === null || connected) return null
  // Everything or nothing: the card arrives complete (see `loads`), never as
  // a frame that fills in while it is being read.
  if (loads < LOADS_NEEDED) return null

  if (corner) {
    return (
      // ONE BUTTON, not a label beside a small one. The whole pill is the
      // target -- mark, name and call to action -- so the hit area is the
      // thing a reader can see rather than the small rectangle inside it, and
      // there is no dead strip in the middle of a control that looks
      // clickable.
      <button type="button" className="wc-corner" onClick={onStart}>
        <span className="wc-corner-mark">
          <Logo size={23} />
          ESPN Draft Assist
        </span>
        {/* Styled as the accent button it replaced, but it is a span: a button
            inside a button is invalid, and the outer one is what takes the
            click from anywhere on the pill. */}
        <span className="wc-corner-go">Get started free</span>
      </button>
    )
  }

  return (
    // `aria-modal` and the scrim are honest here: while this is centred it is
    // the only thing to interact with. The room underneath keeps running,
    // which is the point -- it is a live draft, not a screenshot behind a
    // dialog.
    <div className="wc-scrim" role="dialog" aria-modal="true"
         aria-labelledby="wc-title" onMouseDown={(e) => {
           if (e.target === e.currentTarget) dismiss()
         }}>
      <div className="wc-card">
        <button type="button" className="wc-x" onClick={dismiss}
                aria-label="Close and keep watching the draft">
          ×
        </button>
        <span className="wc-mark">
          <Logo size={22} />
        </span>
        <h2 className="wc-title" id="wc-title">ESPN Draft Assist</h2>
        {askedAt > 0 && (
          // Why they are looking at this card, said before the pitch: they
          // reached for the archive and were sent here instead.
          <p className="wc-why">
            The draft data is free. It needs an account so it knows which
            seat to read it from.
          </p>
        )}
        {/* The claim, then the proof, in one breath: what this is relative
            to the room every other ESPN drafter is sitting in, and the board
            behind the card as evidence. "running now" is the one clause a
            reader could check against the pick timestamps, so it is only
            said when it is true. */}
        <p className="wc-sub">
          This is a real ESPN mock draft{live ? ', running now' : ''}.
          {' '}Everyone in it is drafting off a rankings list. You'd be drafting
          off this.
        </p>
        {/* WHAT IS UNDER THAT BOARD, three lines at a time. The card is the
            only thing a reader who never scrolls will see, so the argument
            has to be inside it -- and six panels stacked in a dialog is a
            page, not a card. One at a time, advancing on its own, is the
            shape that fits: each slide draws its own claim, and the whole
            set is a stop away for anybody who wants to hold on one. */}
        <Showcase data={{ drafts, slot, room, profile, veteran }} />
        {/* Starting also dismisses: the setup flow is further down the
            page, and a card left centred over it would cover the thing it
            just sent the reader to. It moves to the corner, where it goes on
            being the way in. */}
        <button ref={startRef} type="button" className="wc-start"
                onClick={() => { dismiss(); onStart() }}>
          Get started for free
        </button>
        {/* The price, under the button rather than after the fact. "Free" on
            its own would be true of the mock drafts and a surprise at the
            first real one, which is the kind of small dishonesty a reader
            finds out about at exactly the wrong moment. */}
        <p className="wc-price">
          Free in mock drafts · $9.99 when you draft for real
        </p>
        {/* The alternative, said plainly rather than left to the ×. Somebody
            who wants to look before they commit should be told that is a
            choice, not made to guess that closing is safe. */}
        <button type="button" className="wc-watch" onClick={dismiss}>
          or watch the draft first
        </button>
      </div>
    </div>
  )
}
