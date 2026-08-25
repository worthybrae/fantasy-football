import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  fetchCustody, fetchLiveMock, fetchMarketOverview, fetchMarketSlot,
  type MarketOverview,
} from '../api'
import {
  VETERAN_SEASONS, benefits, findVeteran, type Benefit,
} from './benefitFigures'
import { loadProfile } from './draft/CellTip'
import type { LiveMock, MarketSlot } from '../api'
import type { ProfilePayload } from './profile/payload'

// WHAT IT KNOWS, AT LENGTH. The welcome card makes the same six claims three
// lines at a time; this is the version for a reader who scrolled, with every
// panel drawing itself as it arrives. Figures and copy both come from
// `benefitFigures`, so the two surfaces cannot drift apart.
//
// THE ANIMATION IS A REVEAL, NOT A LOOP. Panels draw themselves once, in
// order, when they come into view, and then hold still. A page permanently
// in motion under a reader trying to read is the failure mode this is one
// step away from; the stagger exists to walk the eye down the column, and it
// stops as soon as it has.

function Panel({ panel, delay }: { panel: Benefit; delay: number }) {
  return (
    <article className="ben-panel" style={{ transitionDelay: `${delay}ms` }}>
      <div className="ben-figure">{panel.figure}</div>
      <p className="ben-eyebrow">{panel.eyebrow}</p>
      <h3 className="ben-title">{panel.title}</h3>
      <p className="ben-body">{panel.body}</p>
    </article>
  )
}

/** HOW MUCH OF THIS THERE IS, drawn as itself.
 *
 *  One mark per draft the farm has sat in and recorded, end to end. It is a
 *  scale nobody takes from a sentence -- "hundreds of drafts" is a phrase,
 *  four hundred marks is a wall -- and it is the honest answer to the
 *  question the rest of this section raises: where does any of this come
 *  from.
 *
 *  Live, and it grows while the page is open: the farm is in a mock draft
 *  right now, and the count came from the same endpoint the archive page
 *  reads.
 */
function Corpus({ drafts, overview, connected, onNeedAccount }: {
  drafts: number | null
  overview: MarketOverview | null
  /** Whether this browser holds an ESPN session. Null while the probe is in
   *  flight, which reads as "not yet" -- the button offers the account card,
   *  which is the harmless answer to guess. */
  connected: boolean | null
  onNeedAccount: () => void
}) {
  if (drafts === null || overview === null) return null
  // A cap, said out loud when it bites. The wall is about scale, and past a
  // few hundred marks another row of dots adds nothing but nodes.
  const shown = Math.min(drafts, 480)
  return (
    <section className="ben-corpus">
      <div className="ben-corpus-head">
        <p className="lp-cap">Listening in</p>
        <h3 className="ben-corpus-h3">
          Every one of these is a real draft we sat in and wrote down.
        </h3>
      </div>

      <div className="ben-corpus-wall" role="img"
           aria-label={`${drafts} recorded drafts`}>
        {Array.from({ length: shown }, (_, i) => (
          <span key={i} className="ben-mark"
                // Twelve per second, so the wall fills in about half a
                // minute of reading rather than appearing all at once.
                style={{ animationDelay: `${(i % 120) * 26}ms` }} />
        ))}
      </div>

      <dl className="ben-corpus-facts">
        <div>
          <dt>Drafts recorded</dt>
          <dd className="mono">{drafts.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Picks watched</dt>
          <dd className="mono">{overview.picks.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Made by people</dt>
          <dd className="mono">{overview.human_picks.toLocaleString()}</dd>
        </div>
      </dl>

      {/* THE WAY IN, GATED AT THE DOOR RATHER THAN AT THE PAGE. Somebody who
          has connected an account goes straight to the archive; somebody who
          has not is shown the card that gets them one, with the reason. The
          alternative -- letting them through and refusing at the far end --
          spends a click to say no. */}
      {connected
        ? <Link className="lp-cta ben-corpus-cta" to="/archive">Open the draft data</Link>
        : (
          <button type="button" className="lp-cta ben-corpus-cta"
                  onClick={onNeedAccount}>
            Open the draft data
          </button>
        )}
    </section>
  )
}

export default function Benefits({ onNeedAccount }: {
  /** Asked for when somebody without an account reaches for the archive.
   *  The page owns the welcome card; this only says when to raise it. */
  onNeedAccount: () => void
}) {
  const [drafts, setDrafts] = useState<number | null>(null)
  const [overview, setOverview] = useState<MarketOverview | null>(null)
  const [connected, setConnected] = useState<boolean | null>(null)
  const [profile, setProfile] = useState<ProfilePayload | null>(null)
  const [veteran, setVeteran] = useState<ProfilePayload | null>(null)
  const [room, setRoom] = useState<LiveMock | null>(null)
  const [slot, setSlot] = useState<MarketSlot | null>(null)
  const rootRef = useRef<HTMLElement>(null)

  useEffect(() => {
    let cancelled = false
    fetchMarketOverview()
      .then((body) => {
        if (cancelled) return
        setDrafts(body.drafts)
        setOverview(body)
      })
      // The count falls back to a round number rather than a blank: the
      // archive exists whether or not this fetch does.
      .catch(() => { /* keep the fallback */ })
    return () => { cancelled = true }
  }, [])

  // Whether this browser has an account, for the archive button. A local
  // lookup with no ESPN call behind it (see `/api/espn/custody`).
  useEffect(() => {
    let cancelled = false
    fetchCustody()
      .then((body) => { if (!cancelled) setConnected(body.connected) })
      .catch(() => { if (!cancelled) setConnected(false) })
    return () => { cancelled = true }
  }, [])

  // The same real player the card above uses, through the same two caches:
  // whoever the live room ranks first, and his profile.
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
      .then((body) => {
        if (cancelled || !body) return
        const main = body as unknown as ProfilePayload
        setProfile(main)
        // A CAREER for the health panel. The room's top pick is regularly a
        // rookie, and a games-played chart with one column in it is a fair
        // reading of him and a poor showing for the panel. Walk down the
        // board until somebody has several seasons on record, at most a few
        // players deep, and stop at the first one.
        if (main.seasons.length >= VETERAN_SEASONS) {
          setVeteran(main)
          return
        }
        void findVeteran(shortlist)
          .then((found) => { if (!cancelled && found) setVeteran(found) })
      })
      .catch(() => { /* the fields carry the panels on their own */ })
    fetchMarketSlot(1)
      .then((body) => { if (!cancelled) setSlot(body) })
      .catch(() => { /* the archive panel then shows its field alone */ })
    return () => { cancelled = true }
  }, [])

  // Reveal on arrival, once. `IntersectionObserver` rather than a scroll
  // handler, and the class is added and never removed -- a panel that
  // re-animated every time it scrolled past would be a page that will not
  // sit still.
  useEffect(() => {
    const root = rootRef.current
    if (!root) return
    const panels = Array.from(root.querySelectorAll('.ben-panel'))
    if (!('IntersectionObserver' in window)) {
      panels.forEach((p) => p.classList.add('is-in'))
      return
    }
    const seen = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue
        entry.target.classList.add('is-in')
        seen.unobserve(entry.target)
      }
    }, { rootMargin: '0px 0px -12% 0px', threshold: 0.2 })
    panels.forEach((p) => seen.observe(p))
    return () => seen.disconnect()
  }, [])

  const panels = benefits({ drafts, slot, room, profile, veteran })

  return (
    <section className="ben" ref={rootRef}>
      <div className="ben-lead">
        <p className="lp-cap">Under the board</p>
        <h2 className="ben-h2">
          Six things a rankings list structurally cannot tell you.
        </h2>
      </div>

      {/* TWO COLUMNS, DEALT RATHER THAN FLOWED. `columns: 2` fills the first
          column to half the total height before starting the second, and the
          six panels are ordered by argument rather than by size -- so the
          three tall ones (the recommendation card, comparable seasons, the
          Vegas card) all landed on the left and the three short ones stacked
          on the right. Dealing them explicitly puts one tall panel and two
          shorter ones in each column, and leaves the ORDER alone: it is the
          same array the welcome card rotates through, and the argument runs
          in the sequence it was written in. */}
      <div className="ben-grid">
        <div className="ben-col">
          {[panels[0], panels[3], panels[4]].map((panel, i) => (
            <Panel key={panel.key} panel={panel} delay={i * 60} />
          ))}
        </div>
        <div className="ben-col">
          {[panels[1], panels[2], panels[5]].map((panel, i) => (
            <Panel key={panel.key} panel={panel} delay={40 + i * 60} />
          ))}
        </div>
      </div>

      <Corpus drafts={drafts} overview={overview} connected={connected}
              onNeedAccount={onNeedAccount} />

    </section>
  )
}
