import { useEffect, useState } from 'react'
import { fetchPlanPreview, type PlanPreview as Preview, type PreviewPlayer } from '../api'

// YOUR PLAN, BEFORE THE DRAFT.
//
// The room's plan is the best thing this product does and the one thing a
// reader could not see until they were already in a draft. This is that plan
// run against an empty board -- the first three turns of a seat, with nobody
// taken and nothing on the roster -- so somebody whose draft is on Thursday
// can read it on Sunday.
//
// EVERY NUMBER IS THE SERVER'S. The picks, the paths, the medians, the target
// and its two alternates all arrive from `/api/plan/preview`, which runs
// `scoring/plan.py`. Nothing is ranked, rounded into a different band, or
// re-derived here: a second opinion that disagreed with the draft room by one
// name would make both of them worth less.
//
// THE SIGNATURE IS THE RULE. A snake seat is a measuring stick -- four fixed
// points down a 128-pick draft, spaced by where you sit -- and that is the one
// picture in this product that could not belong to anything else. So the
// section opens with the picks laid along a rule at their real spacing, the
// opening path's positions hung under them, and everything else on the page
// stays quiet around it.

/** "6th", for the seat line. */
function ordinal(n: number): string {
  if (n % 100 >= 11 && n % 100 <= 13) return `${n}th`
  return `${n}${({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th'}`
}

function pct(share: number): string {
  return `${Math.round(share * 100)}%`
}

function posBadge(position: string | null) {
  if (!position) return null
  return (
    <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>
      {position}
    </span>
  )
}

/** The round a pick falls in, in a league of this size. Stated against the
 *  shape the number was COUNTED in, never against the reader's own: pick 41 is
 *  round 6 of an eight-team draft and round 5 of a ten-team one, and quoting
 *  the wrong one is how a true number becomes a false sentence. */
function roundOf(pick: number, teams: number): number {
  return Math.floor((pick - 1) / teams) + 1
}

/** WHERE THE PICKS FALL, drawn at their real spacing.
 *
 *  The four turns of a seat, positioned along the rule by pick number rather
 *  than spread evenly: seat 1 gets two picks almost together and then a long
 *  wait, seat 10 gets the reverse, and that asymmetry IS the thing a drafter
 *  is planning around. Evenly spaced dots would draw every seat the same and
 *  say nothing.
 */
export function PickRule({ picks, path }: { picks: number[]; path: string[] }) {
  if (picks.length === 0) return null
  const last = picks[picks.length - 1]
  return (
    <div className="pr" role="img"
         aria-label={`Picks ${picks.join(', ')}`}>
      <div className="pr-line" aria-hidden="true" />
      {picks.map((pick, i) => (
        <div key={pick} className="pr-mark"
             // Scaled into the middle 92% so the last mark, which is centred
             // on its own position, does not hang half off the right edge.
             style={{ left: `${2 + (pick / last) * 92}%` }}>
          <span className="pr-dot" aria-hidden="true" />
          <span className="pr-no mono">{pick}</span>
          {path[i] && (
            <span className={`pr-pos pos-badge pos-badge-${path[i].toLowerCase()}`}>
              {path[i]}
            </span>
          )}
        </div>
      ))}
    </div>
  )
}

/** The word a reader would use for a corpus format. Matches the three values
 *  `pipeline/draft_log.draft_format` returns; anything else prints as itself
 *  rather than crashing on a shape this build has not been taught the name
 *  of yet. */
const FORMAT_LABELS: Record<string, string> = {
  ppr: 'PPR', half: 'half-PPR', std: 'standard',
}

function formatLabel(format: string): string {
  return FORMAT_LABELS[format] ?? format
}

/** "8-team PPR", "10-team standard": the one phrase used everywhere a
 *  corpus's shape is named, so a reader is told which scoring the numbers
 *  were counted in and not just how many teams. */
export function shapeLabel(teams: number, format: string): string {
  return `${teams}-team ${formatLabel(format)}`
}

/** WHICH DRAFTS THE PICTURE WAS COUNTED IN -- named every time, because the
 *  farm now rotates over more than one shape and a caller that only spoke up
 *  on a mismatch would go quiet the day its own two happened to agree on
 *  team count alone while disagreeing on format.
 *
 *  One sentence, in one place, because both surfaces need it and two
 *  wordings of the same caveat is how one of them quietly goes stale. The
 *  landing page draws the corpus's own shape and stops there; the dashboard
 *  is drawing the reader's league with the corpus's paths on it, so it adds
 *  the half that matters there -- the pick numbers are still theirs.
 */
export function CorpusNote({ corpus, readerTeams, ownPicks = false }: {
  corpus: { teams: number; format: string }
  /** The team count the reader actually asked about -- the seat's own
   *  league on the dashboard, the seat the page opened with on the landing
   *  page. Compared against `corpus.teams` to say whether this is the
   *  reader's own shape or a stand-in for it. */
  readerTeams: number
  ownPicks?: boolean
}) {
  const shape = shapeLabel(corpus.teams, corpus.format)
  const sameShape = corpus.teams === readerTeams
  return (
    <p className="pp-note">
      {sameShape
        ? `Counted in ${shape} drafts.`
        : `Counted in ${shape} drafts — the shape with the most on record; `
          + `your ${readerTeams}-team league will get its own numbers once `
          + 'enough are recorded.'}
      {ownPicks && <> Your pick numbers above are your own league&rsquo;s.</>}
    </p>
  )
}

/** The opening, said in one sentence, with the drafts it was counted from
 *  named in it. */
function Opening({ plan }: { plan: Preview }) {
  const top = plan.opening[0]
  const corpus = plan.corpus
  if (!top || !corpus) {
    return (
      <p className="pp-say">
        Nobody has drafted from this seat in the recorded archive yet. The
        plan below is still this board&rsquo;s answer for it.
      </p>
    )
  }
  const sameShape = corpus.teams === plan.teams
  const runs = ['QB', 'TE'].map((pos) => {
    const pick = plan.position_runs[pos]
    return pick === null || pick === undefined
      ? null
      : `the first ${pos} at pick ${Math.round(pick)}, round ${roundOf(Math.round(pick), corpus.teams)}`
  }).filter(Boolean)

  return (
    <>
      <p className="pp-say">
        {sameShape
          ? `The ${ordinal(plan.slot)} seat of ${plan.teams} usually opens `
          : `The ${ordinal(plan.slot)} seat usually opens `}
        <strong className="pp-path">{top.path.join('–')}</strong>
        {' — '}
        {pct(top.share)} of {corpus.drafts.toLocaleString()} recorded
        {' '}{shapeLabel(corpus.teams, corpus.format)} ESPN drafts.
      </p>
      {runs.length > 0 && (
        <p className="pp-say pp-say-2">
          In those drafts the board empties in this order: {runs.join(', and ')}.
        </p>
      )}
      {!sameShape && (
        <CorpusNote corpus={corpus} readerTeams={plan.teams} ownPicks />
      )}
    </>
  )
}

/** One player, as the plan describes him. `lead` gets the face and the
 *  reasons; an alternate gets a line. */
function PlayerRow({ player, lead = false }: {
  player: PreviewPlayer
  lead?: boolean
}) {
  return (
    <div className={`pp-who${lead ? ' pp-who-lead' : ''}`}>
      {lead && player.headshot && (
        <img className="pp-face" src={player.headshot} alt=""
             width={40} height={40} loading="lazy" />
      )}
      <span className="pp-who-in">
        <span className="pp-name">
          {player.name}
          {player.favourite && (
            <span className="pp-star" title="One of your guys"
                  aria-label="One of your guys">★</span>
          )}
        </span>
        <span className="pp-meta">
          {posBadge(player.position)}
          {player.team && <span className="pp-team mono">{player.team}</span>}
        </span>
      </span>
    </div>
  )
}

/** One turn: what to take, what else, and why. */
function Turn({ turn }: {
  turn: Preview['targets'][number]
}) {
  const target = turn.target
  if (!target) return null
  // The plan's own reason, at most one. The room prints the whole list beside
  // a live board; here it is a card in a column and the first reason is the
  // one the plan led with.
  const why = target.pros[0] ?? target.cons[0] ?? null
  return (
    <article className="pp-turn">
      <header className="pp-turn-head">
        <span className="pp-pick mono">Pick {turn.pick_no}</span>
        <span className="pp-round">Round {turn.round}</span>
      </header>
      <PlayerRow player={target} lead />
      {(target.edge_pts !== null || target.lasts_pct !== null) && (
        <dl className="pp-figs">
          {target.lasts_pct !== null && (
            <div>
              <dt>Still there</dt>
              <dd className="mono">{Math.round(target.lasts_pct)}%</dd>
            </div>
          )}
          {target.edge_pts !== null && target.edge_at_pick !== null && (
            <div>
              <dt>Over waiting to {target.edge_at_pick}</dt>
              <dd className="mono">
                {target.edge_pts > 0 ? '+' : ''}{Math.round(target.edge_pts)} pts
              </dd>
            </div>
          )}
        </dl>
      )}
      {why && <p className="pp-why">{why}</p>}
      {turn.alternates.length > 0 && (
        <div className="pp-alts">
          <p className="pp-alts-cap">Or</p>
          {turn.alternates.map((alt) => (
            <PlayerRow key={alt.player_id} player={alt} />
          ))}
        </div>
      )}
    </article>
  )
}

export default function PlanPreview({ teams, slot, tag = '', heading = 'Your plan',
                                      note = null }: {
  teams: number
  slot: number
  /** Bumped by the caller when something might have changed the answer -- a
   *  favourites save, above all. Not sent anywhere; it discriminates the
   *  request cache, the same way the outlook's does. */
  tag?: string
  heading?: string
  /** A line under the heading saying where the seat came from. The dashboard
   *  knows whether it read the seat off a real upcoming draft or off the
   *  reader's own controls, and that is worth saying. */
  note?: string | null
}) {
  const [plan, setPlan] = useState<Preview | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setPlan(null)
    setError(null)
    fetchPlanPreview(teams, slot, tag)
      .then((body) => { if (!cancelled) setPlan(body) })
      .catch((err) => { if (!cancelled) setError(String(err?.message || err)) })
    return () => { cancelled = true }
  }, [teams, slot, tag])

  return (
    <section className="db-sec pp">
      <div className="db-sec-head">
        <h2 className="db-sec-title">{heading}</h2>
        {note && <span className="db-sec-note">{note}</span>}
      </div>

      <div className="db-card pp-card">
        {error !== null && <p className="db-error">{error}</p>}

        {/* THE SKELETON KEEPS THE CARD'S HEIGHT, for the same reason the
            outlook's does: this section sits above the league list, and a
            card that grows when its answer lands shoves everything under it
            down the page while somebody is reading. */}
        {plan === null && error === null && (
          <div className="pp-wait" aria-hidden="true">
            <div className="pp-wait-rule" />
            <div className="pp-wait-row" />
            <div className="pp-wait-row" />
          </div>
        )}

        {plan !== null && (
          <>
            <PickRule picks={plan.picks} path={plan.opening[0]?.path ?? []} />
            <Opening plan={plan} />
            <div className="pp-turns">
              {plan.targets.map((turn) => (
                <Turn key={turn.pick_no} turn={turn} />
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  )
}
