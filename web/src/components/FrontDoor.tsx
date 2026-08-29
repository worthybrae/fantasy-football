import { useEffect, useState } from 'react'
import { fetchMarketOverview, fetchPlanPreview,
         type MarketOverview, type PlanPreview } from '../api'
import FounderBadge from './FounderBadge'
import { Logo } from './Logo'
import { PickRule } from './PlanPreview'

// THE FRONT DOOR, for somebody who has connected nothing.
//
// WHAT THIS REPLACES, AND WHY. The page used to open with a live draft room
// and then argue with itself: a welcome card floated over the room saying what
// the product was, a six-panel "under the board" section said it again at
// length, and the explainer below said it a third time in prose. Three
// components, one argument, and a scrim over the top that swallowed the first
// click a visitor made on the very board it was pointing at.
//
// So the argument is made once, in order, and each part does one job:
//
//   the sentence   -- what this is, in the words a drafter would use
//   three steps    -- what you would actually do, in the order you do it
//   the live room  -- the proof, running, with nothing over it
//   four cards     -- what you get, each with a number off this deployment
//   the FAQ        -- the long answers, for whoever wants them
//
// EVERY NUMBER IS READ, NOT WRITTEN. Two endpoints supply all of them: the
// archive's own size (`/api/market/overview`) and one seat's plan
// (`/api/plan/preview`), which is the same planner the draft room runs. A
// figure that has quietly gone stale is worse than a weaker claim that is
// still true, so nothing on this page is a literal.
//
// THE SEAT THE PAGE READS FROM. Six of ten: the most common league size and a
// middle seat, which is the one a visitor is most likely to recognise as
// theirs. Named here rather than buried in three call sites.
const DEMO_TEAMS = 10
const DEMO_SLOT = 6

/** The shared read. Both sections below want the same two answers and mount
 *  together, so they ask through the same deduplicated cache (see
 *  `cachedGet`) and neither owns the other's state. */
function useFrontDoorFigures() {
  const [overview, setOverview] = useState<MarketOverview | null>(null)
  const [plan, setPlan] = useState<PlanPreview | null>(null)
  useEffect(() => {
    let cancelled = false
    fetchMarketOverview()
      .then((body) => { if (!cancelled) setOverview(body) })
      .catch(() => { /* the line says less, and still says something true */ })
    fetchPlanPreview(DEMO_TEAMS, DEMO_SLOT)
      .then((body) => { if (!cancelled) setPlan(body) })
      .catch(() => { /* the cards fall back to what they can say without it */ })
    return () => { cancelled = true }
  }, [])
  return { overview, plan }
}

/** THE SENTENCE, THE COUNT AND THE WAY IN.
 *
 *  No stat tiles and no gradient. The one piece of ornament is the wordmark,
 *  because the page has to say whose tool this is before it says what it does.
 */
export function Hero({ onStart }: { onStart: () => void }) {
  const { overview, plan } = useFrontDoorFigures()
  const opening = plan?.opening[0] ?? null
  return (
    <section className="fd-hero">
      <div className="fd-hero-say">
      <p className="fd-mark">
        <Logo size={22} />
        ESPN Draft Assist
      </p>
      <h1 className="fd-h1">
        Draft with what real ESPN drafts do: who lasts to your pick, and a
        plan for every round.
      </h1>
      <p className="fd-sub">
        {overview === null
          ? 'Read from real ESPN drafts this site records every day.'
          : (
            <>
              Read from{' '}
              <strong className="mono">{overview.drafts.toLocaleString()}</strong>
              {' '}recorded ESPN drafts —{' '}
              <strong className="mono">{overview.picks.toLocaleString()}</strong>
              {' '}picks, counted. Updated daily.
            </>
          )}
      </p>
      <div className="fd-act">
        <button type="button" className="lp-cta fd-cta" onClick={onStart}>
          Connect ESPN — free for the first 100
        </button>
        <FounderBadge variant="open" />
      </div>
      <p className="fd-alt">
        {/* A real anchor to the room below, so the answer to "what IS this"
            is one keystroke away and not a scroll somebody has to guess at. */}
        or <a className="lp-bar-link" href="#demo">watch a draft happening now</a>
      </p>
      </div>

      {/* THE THESIS, DRAWN. One seat of one draft, with its four turns at
          their real spacing and the positions people take at each. It is the
          sentence above restated as the thing itself, and it is the only
          picture on this page that could not belong to another product.

          Absent rather than faked when the plan cannot be read: a rule with
          invented picks on it would be the one dishonest thing here. */}
      {plan !== null && opening !== null && plan.corpus !== null && (
        <aside className="fd-seat" aria-label="One seat, drawn">
          <p className="lp-cap">
            Seat {plan.slot} of {plan.teams} · every turn you own
          </p>
          <PickRule picks={plan.picks} path={opening.path} />
          <p className="fd-seat-say">
            <strong className="mono">{Math.round(opening.share * 100)}%</strong>
            {' '}of {plan.corpus.drafts.toLocaleString()} recorded
            {' '}{plan.corpus.teams}-team drafts open this way from here.
          </p>
        </aside>
      )}
    </section>
  )
}

/** The three steps, with a drawing each.
 *
 *  NUMBERED, because this genuinely is a sequence: you cannot pick your guys
 *  before the site knows who you are, and the room is no use until both. The
 *  drawings are diagrams of the action rather than screenshots of it -- a
 *  screenshot at this size is a grey rectangle, and the point of each one is
 *  the single gesture it takes.
 */
export function Steps({ onStart }: { onStart: () => void }) {
  return (
    <section className="fd-steps" aria-labelledby="fd-steps-h">
      <h2 className="fd-h2" id="fd-steps-h">Three things, once.</h2>
      <ol className="fd-step-list">
        <li className="fd-step">
          <span className="fd-step-no mono" aria-hidden="true">1</span>
          <div className="fd-step-fig">
            <svg viewBox="0 0 120 64" role="img"
                 aria-label="A bookmark dragged to the browser's bar">
              <rect x="2" y="4" width="116" height="13" rx="3"
                    className="fd-svg-bar" />
              <rect x="7" y="8" width="26" height="5" rx="2"
                    className="fd-svg-dim" />
              <rect x="38" y="8" width="18" height="5" rx="2"
                    className="fd-svg-dim" />
              <rect x="62" y="7" width="34" height="7" rx="2"
                    className="fd-svg-chip" />
              <path d="M79 24 L79 44" className="fd-svg-drag" />
              <path d="M75 39 L79 45 L83 39" className="fd-svg-drag" />
              <rect x="2" y="50" width="116" height="12" rx="3"
                    className="fd-svg-panel" />
            </svg>
          </div>
          <h3 className="fd-step-h">Connect ESPN</h3>
          <p className="fd-step-p">
            Drag one bookmark to your bar and click it on any ESPN fantasy
            page. Your leagues appear here. Nothing is installed, and it never
            sees your password.
          </p>
        </li>

        <li className="fd-step">
          <span className="fd-step-no mono" aria-hidden="true">2</span>
          <div className="fd-step-fig">
            <svg viewBox="0 0 120 64" role="img"
                 aria-label="A short list of players, three of them starred">
              {[0, 1, 2, 3].map((i) => (
                <g key={i}>
                  <rect x="4" y={6 + i * 14} width="9" height="9" rx="2"
                        className={i < 3 ? 'fd-svg-pos' : 'fd-svg-dim'} />
                  <rect x="18" y={9 + i * 14} width={62 - i * 9} height="4"
                        rx="2" className="fd-svg-dim" />
                  {i < 3 && (
                    <path d="M0 -4 L1.2 -1.2 L4 -1.2 L1.8 0.6 L2.6 3.4 L0 1.8 L-2.6 3.4 L-1.8 0.6 L-4 -1.2 L-1.2 -1.2 Z"
                          transform={`translate(104 ${11 + i * 14})`}
                          className="fd-svg-star" />
                  )}
                </g>
              ))}
            </svg>
          </div>
          <h3 className="fd-step-h">Pick your guys</h3>
          <p className="fd-step-p">
            Star five to twenty-five players you want this year. The board
            stars them in the room, and the plan reaches for them a round
            earlier than it would for anybody else.
          </p>
        </li>

        <li className="fd-step">
          <span className="fd-step-no mono" aria-hidden="true">3</span>
          <div className="fd-step-fig">
            <svg viewBox="0 0 120 64" role="img"
                 aria-label="A draft board with one pick marked as yours">
              {Array.from({ length: 24 }, (_, i) => (
                <rect key={i} x={4 + (i % 8) * 14.5} y={6 + Math.floor(i / 8) * 18}
                      width="12" height="14" rx="2"
                      className={i === 13 ? 'fd-svg-mine' : 'fd-svg-cell'} />
              ))}
              <path d="M4 58 H116" className="fd-svg-rule" />
            </svg>
          </div>
          <h3 className="fd-step-h">Open the room</h3>
          <p className="fd-step-p">
            On draft night the board sits beside your ESPN room and follows
            every pick. It says who to take, who will still be there next
            round, and who will not.
          </p>
        </li>
      </ol>
      <button type="button" className="lp-cta fd-steps-cta" onClick={onStart}>
        Start with a free mock draft
      </button>
    </section>
  )
}

/** WHAT YOU GET, four cards, each carrying one reading off this deployment.
 *
 *  The number leads and the sentence explains it, rather than the other way
 *  round: a card whose figure is an illustration of its heading is a card the
 *  reader can skip.
 */
export function WhatYouGet() {
  const { overview, plan } = useFrontDoorFigures()
  const first = plan?.targets[0]?.target ?? null
  const second = plan?.targets[1] ?? null
  const lasts = second?.target?.lasts_pct ?? null
  const seat = plan ? `${plan.slot} of ${plan.teams}` : null

  return (
    <section className="fd-get" aria-labelledby="fd-get-h">
      <h2 className="fd-h2" id="fd-get-h">What you get</h2>
      <ul className="fd-get-list">
        <li className="fd-get-card">
          <p className="lp-cap">Lasts %</p>
          <p className="fd-get-num mono">
            {lasts === null ? '—' : `${Math.round(lasts)}%`}
          </p>
          <h3 className="fd-get-h3">Who is still there at your pick</h3>
          <p className="fd-get-p">
            {second?.target && lasts !== null
              ? `${second.target.name}'s chance of reaching pick ${second.pick_no} from seat ${seat}, counted in recorded drafts rather than guessed from a ranking.`
              : 'Every player\u2019s chance of reaching each of your turns, counted in recorded drafts rather than guessed from a ranking.'}
          </p>
        </li>

        <li className="fd-get-card">
          <p className="lp-cap">The plan</p>
          <p className="fd-get-num mono">
            {plan ? `Pick ${plan.picks[0]}` : '—'}
          </p>
          <h3 className="fd-get-h3">One name a round, and why</h3>
          <p className="fd-get-p">
            {first
              ? `From seat ${seat} it opens with ${first.name}${first.position ? ` (${first.position})` : ''}, with two alternates and the reasons for and against each.`
              : 'A target and two alternates for every turn you own, with the reasons for and against each.'}
          </p>
        </li>

        <li className="fd-get-card">
          <p className="lp-cap">Your guys by pick</p>
          {/* The seat, not its picks: the rule in the hero already draws
              those, and printing them twice would spend the page's one
              memorable figure on a repeat. */}
          <p className="fd-get-num mono">
            {plan ? `${plan.slot} of ${plan.teams}` : '—'}
          </p>
          <h3 className="fd-get-h3">Which of your players you can have</h3>
          <p className="fd-get-p">
            {plan
              ? `Star five to twenty-five players and the grid says which of them reach each of seat ${seat}'s turns — ${plan.picks.slice(0, 4).join(', ')}.`
              : 'Star the players you want and the grid says which of them reach each of your turns.'}
          </p>
        </li>

        <li className="fd-get-card">
          <p className="lp-cap">ADP pages</p>
          <p className="fd-get-num mono">
            {overview ? overview.human_picks.toLocaleString() : '—'}
          </p>
          <h3 className="fd-get-h3">Where players actually go</h3>
          <p className="fd-get-p">
            {overview
              ? `Picks made by people, not by ESPN's autodrafter, out of ${overview.picks.toLocaleString()} recorded.`
              : 'Every pick a person made in the drafts this site recorded.'}
            {' '}
            {/* A plain anchor: /adp is rendered by FastAPI from the corpus and
                is not a route this app's router knows. */}
            <a className="lp-bar-link" href="/adp">See ADP for every player</a>.
          </p>
        </li>
      </ul>
    </section>
  )
}
