import { useEffect, useState, type ReactNode } from 'react'
import { fetchPlanPreview, type PlanPreview, type PreviewPlayer } from '../api'

// YOUR DRAFT PLAN, ROUND BY ROUND.
//
// One row per turn you own: the round, the pick, one name, one sentence
// saying why, and the two players to take instead if he is gone. Eight of
// them on the signed-in home, four on the front door. Everything else the
// planner knows is behind the "why" on the row it belongs to.
//
// WHAT IT REPLACES. This was three cards side by side with a percentage, a
// points figure and a bare reason fragment on each, under a rule drawing the
// seat, under a sentence about what other people usually do from it. Five
// things competing on one screen, four of them about the model rather than
// about Thursday.
//
// THE WAIT IS THE PICTURE. A snake seat is not a list of turns, it is a list
// of WAITS: from the sixth seat of ten you pick, wait nine, pick, wait
// eleven, pick. That asymmetry is the whole thing a drafter plans around and
// it is the one shape in this product that could not belong to anything else
// -- so it is drawn between the rungs, in picks, at a length that grows with
// the number. Nothing else on the page is decorated.
//
// EVERY NUMBER IS THE SERVER'S. The rounds, the picks, the names, the
// sentence and the backups all arrive from `/api/plan/preview`, which runs
// `scoring/plan.py` -- the draft room's own planner against an empty board.
// Nothing is ranked or re-derived here: a second opinion that disagreed with
// the room by one name would make both of them worth less.

function posBadge(position: string | null) {
  if (!position) return null
  return (
    <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>
      {position}
    </span>
  )
}

/** The reasons, opened on request.
 *
 *  A DISCLOSURE AND NOT A CARD. The row already carries the plan's own
 *  sentence; this is the rest of what it weighed, for the reader who wants to
 *  argue with it. Closed by default because eight rounds of five pros and
 *  four cons is a spreadsheet, and the row above it is the answer. */
function Why({ player }: { player: PreviewPlayer }) {
  if (player.pros.length === 0 && player.cons.length === 0) return null
  return (
    <details className="dpr-why">
      <summary className="dpr-why-sum">Why him</summary>
      <div className="dpr-why-in">
        {player.pros.length > 0 && (
          <div className="dpr-why-col">
            <p className="lp-cap">For</p>
            <ul className="dpr-why-list">
              {player.pros.map((pro) => <li key={pro}>{pro}</li>)}
            </ul>
          </div>
        )}
        {player.cons.length > 0 && (
          <div className="dpr-why-col">
            <p className="lp-cap">Against</p>
            <ul className="dpr-why-list dpr-why-against">
              {player.cons.map((con) => <li key={con}>{con}</li>)}
            </ul>
          </div>
        )}
      </div>
    </details>
  )
}

/** One turn. */
function Round({ turn, wait }: {
  turn: PlanPreview['targets'][number]
  /** How many picks pass before this turn comes round -- null on the first,
   *  which nothing precedes. Drawn as the gap above the row, so a seat's own
   *  rhythm is visible before a word of it is read. */
  wait: number | null
}) {
  const target = turn.target
  return (
    <li className="dpr-turn">
      {wait !== null && (
        <p className="dpr-wait" style={{ '--wait': wait } as React.CSSProperties}>
          <span className="dpr-wait-line" aria-hidden="true" />
          <span className="dpr-wait-say mono">{wait} picks later</span>
        </p>
      )}
      <div className="dpr-row">
        <p className="dpr-when">
          <span className="dpr-round">Round {turn.round}</span>
          <span className="dpr-pick mono">pick {turn.pick_no}</span>
        </p>
        {target === null ? (
          <p className="dpr-none">
            Nobody the plan is confident about this far out. Take the best
            player left when you get here.
          </p>
        ) : (
          <div className="dpr-take">
            <p className="dpr-name">
              <span className="dpr-verb">Take</span>
              {target.name}
              {posBadge(target.position)}
              {target.team && <span className="dpr-team mono">{target.team}</span>}
              {target.favourite && (
                <span className="dpr-star" title="One of your guys"
                      aria-label="One of your guys">★</span>
              )}
            </p>
            {target.reason && <p className="dpr-reason">{target.reason}</p>}
            {turn.backups.length > 0 && (
              <p className="dpr-backups">
                If he&rsquo;s gone: <span className="dpr-backup-names">
                  {turn.backups.join(', ')}
                </span>
              </p>
            )}
            <Why player={target} />
          </div>
        )}
      </div>
    </li>
  )
}

export default function DraftPlanRounds({
  teams, slot, turns = 8, tag = '', heading = 'Your draft plan',
  note = null, control = null, lede = null,
}: {
  teams: number
  slot: number
  /** How many of the seat's turns to plan. Eight on the home page, four on
   *  the front door. */
  turns?: number
  /** Bumped by the caller when something might have changed the answer -- a
   *  save to "my guys", above all. Not sent anywhere; it discriminates the
   *  request cache. */
  tag?: string
  heading?: string
  /** Which draft this is, said out loud under the heading. A plan for the
   *  wrong league is a plan for somebody else's draft, and the reader is the
   *  only one who can catch that. */
  note?: string | null
  /** The seat control, when the caller has one. It belongs in this section's
   *  head: the seat is what the whole section is about, and a reader who
   *  disagrees with the seat we chose should be able to fix it where they
   *  read it. */
  control?: ReactNode
  lede?: ReactNode
}) {
  const [plan, setPlan] = useState<PlanPreview | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setPlan(null)
    setError(null)
    fetchPlanPreview(teams, slot, tag, turns)
      .then((body) => { if (!cancelled) setPlan(body) })
      .catch((err) => { if (!cancelled) setError(String(err?.message || err)) })
    return () => { cancelled = true }
  }, [teams, slot, tag, turns])

  return (
    <section className="db-sec dpr" aria-labelledby="dpr-h">
      <div className="db-sec-head">
        <h2 className="db-sec-title" id="dpr-h">{heading}</h2>
        {control}
        {note && <span className="db-sec-note">{note}</span>}
      </div>

      {lede}
      {error !== null && <p className="db-error">{error}</p>}

      {/* THE SKELETON KEEPS THE SECTION'S HEIGHT, so the list under it does
          not get shoved down the page while somebody is reading it. */}
      {plan === null && error === null && (
        <div className="dpr-wait-skel" aria-hidden="true">
          <div className="dpr-skel-row" />
          <div className="dpr-skel-row" />
          <div className="dpr-skel-row" />
        </div>
      )}

      {plan !== null && (
        <ol className="dpr-list">
          {plan.targets.map((turn, i) => (
            <Round key={turn.pick_no} turn={turn}
                   wait={i === 0 ? null
                     : turn.pick_no - plan.targets[i - 1].pick_no} />
          ))}
        </ol>
      )}
    </section>
  )
}
