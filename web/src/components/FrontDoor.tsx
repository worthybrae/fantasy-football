import { useEffect, useState } from 'react'
import { fetchMarketOverview, fetchPlanPreview,
         type MarketOverview, type PlanPreview } from '../api'
import DraftPlanRounds from './DraftPlanRounds'
import FounderBadge from './FounderBadge'
import { Logo } from './Logo'
import { ordinal, SIZES } from '../lib/seat'

// THE FRONT DOOR, for somebody who has connected nothing.
//
// WHO IT IS FOR. Somebody who drafts once a year on ESPN, has done a couple
// of mocks, and thinks in rankings, sleepers and "my guys". They want, in
// order: who to take right now and why; whether their guys will still be
// there; proof it works before they trust it; and nothing to learn.
//
// So the page is that, in that order:
//
//   the sentence   -- what this is, in the words a drafter would use
//   TRY IT NOW     -- their league size, their seat, four rounds of a plan
//   the live room  -- the proof, running, with nothing over it
//   three lines    -- what you get, one number each
//   the FAQ        -- three questions, and the price
//
// TRY IT NOW IS THE THESIS, NOT A DEMO. Every other front door in this
// category asks you to sign up to find out whether the thing is any good.
// Two selects here and the real planner answers, for the real seat, with the
// real board -- no account, no email, nothing installed. If the plan is not
// worth having, that is the fastest possible way to find out.
//
// EVERY NUMBER IS READ, NOT WRITTEN. Two endpoints supply all of them: the
// archive's own size (`/api/market/overview`) and one seat's plan
// (`/api/plan/preview`), which is the same planner the draft room runs. A
// figure that has quietly gone stale is worse than a weaker claim that is
// still true, so nothing on this page is a literal.

// The seat the page opens on: the most common league size and a middle seat,
// which is the one a visitor is most likely to recognise as theirs.
const DEMO_TEAMS = 10
const DEMO_SLOT = 6
// The sizes the "try it now" select offers. Fewer than the seat store's full
// menu on purpose: this is a taste of the product, and three options is a
// choice while five is a form.
const TRY_SIZES = SIZES.slice(0, 3)
// Four rounds. Long enough to be a plan rather than a pick, short enough to
// read before deciding whether to bother connecting anything.
const TRY_TURNS = 4

/** The archive's own size, for the one number in the hero. */
function useOverview() {
  const [overview, setOverview] = useState<MarketOverview | null>(null)
  useEffect(() => {
    let cancelled = false
    fetchMarketOverview()
      .then((body) => { if (!cancelled) setOverview(body) })
      .catch(() => { /* the line says less, and still says something true */ })
    return () => { cancelled = true }
  }, [])
  return overview
}

/** THE SENTENCE, THE COUNT AND THE TWO WAYS IN.
 *
 *  No stat tiles and no gradient. The one piece of ornament is the wordmark,
 *  because the page has to say whose tool this is before it says what it
 *  does. */
export function Hero({ onStart }: { onStart: () => void }) {
  const overview = useOverview()
  return (
    <section className="fd-hero">
      <p className="fd-mark">
        <Logo size={22} />
        ESPN Draft Assist
      </p>
      <h1 className="fd-h1">Your ESPN draft, with a plan.</h1>
      <p className="fd-sub">
        Who to take at every pick, and whether your guys will still be there —
        {overview === null
          ? ' from real ESPN drafts, recorded every day.'
          : (
            <>
              {' '}from{' '}
              <strong className="mono">{overview.drafts.toLocaleString()}</strong>
              {' '}real ESPN drafts.
            </>
          )}
      </p>
      <div className="fd-act">
        <button type="button" className="lp-cta fd-cta" onClick={onStart}>
          Try a mock draft — free
        </button>
        <button type="button" className="lp-cta fd-cta fd-cta-2"
                onClick={onStart}>
          Connect your ESPN league
        </button>
      </div>
      {/* THE TWO BUTTONS SHARE ONE MECHANISM, and saying so is better than
          hiding it. They are two intentions -- practise, or draft for real --
          and one setup, which takes a click. A reader who works that out for
          themselves after clicking has been mildly tricked. */}
      <p className="fd-alt">
        Both start the same way: drag one bookmark to your bar and click it on
        ESPN. No password, nothing installed.
      </p>
    </section>
  )
}

/** TRY IT NOW: their league, their seat, four rounds, no login. */
export function TryItNow() {
  const [teams, setTeams] = useState(DEMO_TEAMS)
  const [slot, setSlot] = useState(DEMO_SLOT)

  return (
    <section className="fd-try" aria-labelledby="fd-try-h">
      <h2 className="fd-h2" id="fd-try-h">Try it now</h2>
      <p className="fd-try-say">
        Pick your league size and where you sit. This is the same plan the
        draft room builds, for that seat, right now.
      </p>
      <div className="fd-try-ctl">
        <label className="fd-try-lab">
          League
          <select className="fd-try-select" value={teams}
                  aria-label="League size"
                  onChange={(e) => {
                    const size = Number(e.target.value)
                    setTeams(size)
                    setSlot((s) => Math.min(s, size))
                  }}>
            {TRY_SIZES.map((n) => (
              <option key={n} value={n}>{n} teams</option>
            ))}
          </select>
        </label>
        <label className="fd-try-lab">
          My seat
          <select className="fd-try-select" value={slot}
                  aria-label="Your seat"
                  onChange={(e) => setSlot(Number(e.target.value))}>
            {Array.from({ length: teams }, (_, i) => i + 1).map((n) => (
              <option key={n} value={n}>{ordinal(n)}</option>
            ))}
          </select>
        </label>
      </div>
      <DraftPlanRounds teams={teams} slot={slot} turns={TRY_TURNS}
                       heading="Your first four rounds"
                       note={`Seat ${slot} of ${teams}`} />
    </section>
  )
}

/** WHAT YOU GET: three lines, one number each.
 *
 *  The number leads and the sentence explains it, rather than the other way
 *  round: a line whose figure is an illustration of its heading is a line the
 *  reader can skip. Three, not four -- the ADP pages are a side route and
 *  they were the one card here that was not about draft night. */
export function WhatYouGet() {
  const [plan, setPlan] = useState<PlanPreview | null>(null)
  useEffect(() => {
    let cancelled = false
    fetchPlanPreview(DEMO_TEAMS, DEMO_SLOT, '', TRY_TURNS)
      .then((body) => { if (!cancelled) setPlan(body) })
      .catch(() => { /* the lines fall back to what they can say without it */ })
    return () => { cancelled = true }
  }, [])

  const first = plan?.targets[0]?.target ?? null
  const second = plan?.targets[1] ?? null
  const lasts = second?.target?.lasts_pct ?? null
  const seat = plan ? `${plan.slot} of ${plan.teams}` : null

  return (
    <section className="fd-get" aria-labelledby="fd-get-h">
      <h2 className="fd-h2" id="fd-get-h">What you get</h2>
      <ul className="fd-get-list">
        <li className="fd-get-card">
          <p className="fd-get-num mono">
            {lasts === null ? '—' : `${Math.round(lasts)}%`}
          </p>
          <h3 className="fd-get-h3">Will he be there?</h3>
          <p className="fd-get-p">
            {second?.target && lasts !== null
              ? `${second.target.name}'s chance of reaching pick ${second.pick_no} from seat ${seat} — counted in real ESPN drafts, not guessed from a ranking.`
              : 'Every player’s chance of reaching each of your turns — counted in real ESPN drafts, not guessed from a ranking.'}
          </p>
        </li>

        <li className="fd-get-card">
          <p className="fd-get-num mono">
            {plan ? `Pick ${plan.picks[0]}` : '—'}
          </p>
          <h3 className="fd-get-h3">One name a round, and why</h3>
          <p className="fd-get-p">
            {first
              ? `From seat ${seat} the plan opens with ${first.name}${first.position ? ` (${first.position})` : ''}, one sentence saying why, and two backups if he goes.`
              : 'A name for every turn you own, one sentence saying why, and two backups if he goes.'}
          </p>
        </li>

        <li className="fd-get-card">
          <p className="fd-get-num mono">5–25</p>
          <h3 className="fd-get-h3">My guys</h3>
          <p className="fd-get-p">
            {plan
              ? `Star the players you want and the plan takes them a round early — and says which of them reach the turns seat ${seat} owns: ${plan.picks.slice(0, 4).join(', ')}.`
              : 'Star the players you want and the plan takes them a round early — and says which of them reach your turns.'}
          </p>
        </li>
      </ul>
    </section>
  )
}

/** The price, at the bottom, where somebody has finished deciding. */
export function FounderLine({ onStart }: { onStart: () => void }) {
  return (
    <section className="fd-founder">
      <p className="fd-founder-say">
        Free for the first 100 accounts, for good. After that a real draft is
        $9.99 a season; mock drafts stay free.
      </p>
      <FounderBadge variant="open" />
      <button type="button" className="lp-cta" onClick={onStart}>
        Connect your ESPN league
      </button>
    </section>
  )
}
