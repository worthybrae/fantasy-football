import { useEffect, useState, type ReactNode } from 'react'
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
// section opens with the picks laid along a rule at their real spacing, THE
// PLAN'S OWN positions hung under them, and everything else on the page stays
// quiet around it.
//
// THE RULE DRAWS THE PLAN, NOT THE CORPUS. It used to hang the most-walked
// opening path under the picks while the cards below named somebody else --
// a timeline reading WR-RB-RB-WR over cards reading RB-QB-TE, which is two
// answers to one question on one screen. The badges are the cards' own
// positions now, and what rooms in this seat USUALLY do is a muted sentence
// underneath, where a piece of background belongs.

/** "31 %". Spaced, the way `scoring/plan.py` writes the percentages it puts
 *  in its own reasons, so the two halves of a card agree about typography. */
function pct(share: number): string {
  return `${Math.round(share * 100)} %`
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
export function PickRule({ picks, positions }: {
  picks: number[]
  /** One position per pick, in the same order -- what to expect AT that
   *  turn. The dashboard hands it the plan's own targets and the landing
   *  page hands it the corpus's opening path; either way the badge under a
   *  mark has to be about the pick it is drawn on, which is why this is
   *  indexed against `picks` rather than being a path of its own. Short
   *  entries and nulls simply draw no badge. */
  positions: (string | null)[]
}) {
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
          {positions[i] && (
            <span className={
              `pr-pos pos-badge pos-badge-${positions[i]!.toLowerCase()}`}>
              {positions[i]}
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

/** WHAT ROOMS IN THIS SEAT USUALLY DO -- background, and printed as such.
 *
 *  This was the loudest sentence in the section and it is not the answer: it
 *  is what OTHER people did from this seat, in drafts that are probably not
 *  the reader's shape, and it was being read as the plan. The plan is the
 *  cards. So this is a muted line under them, and it names the corpus it
 *  counted in the same breath rather than in a caveat below.
 */
function Opening({ plan }: { plan: Preview }) {
  const top = plan.opening[0]
  const corpus = plan.corpus
  if (!top || !corpus) {
    return (
      <p className="pp-note">
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
      <p className="pp-note">
        Rooms in this seat usually open{' '}
        <strong className="pp-path">{top.path.join('–')}</strong>
        {' ('}{pct(top.share)} of {corpus.drafts.toLocaleString()} recorded
        {' '}{shapeLabel(corpus.teams, corpus.format)} drafts).
      </p>
      {runs.length > 0 && (
        <p className="pp-note pp-say-2">
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

/** One turn: what to take, what else, and why.
 *
 *  THE TWO FIGURES ARE ONLY TRUE ABOUT A PICK, so each says which one. "Still
 *  there · 85 %" is a claim about pick 5 and "+56 pts" is a comparison with
 *  pick 12, and printed as bare labels they read as facts about the player --
 *  which is how somebody comes to plan on 85 % at a turn where it is 40.
 *
 *  AND A NEGATIVE EDGE IS NOT A NUMBER TO PRINT. "−3 pts" against a name is
 *  read as a penalty, or as a mistake in the plan; what it actually says is
 *  that nobody better at this position is expected to be gone by the next
 *  turn, which is a sentence about waiting. So it is said as one.
 */
function Turn({ turn }: {
  turn: Preview['targets'][number]
}) {
  const target = turn.target
  if (!target) return null
  // The plan's own reason, at most one -- ESPN's rank on him, most often.
  // The room prints the whole list beside a live board; here it is a card in
  // a column and the first reason is the one the plan led with.
  const why = target.pros[0] ?? target.cons[0] ?? null
  const priced = target.edge_pts !== null && target.edge_at_pick !== null
  // Worth taking early, and by how much. At or below zero the plan is saying
  // "he keeps", which the sentence under the card says in words.
  const gains = priced && (target.edge_pts as number) > 0
  const keeps = priced && !gains
  return (
    <article className="pp-turn">
      <header className="pp-turn-head">
        <span className="pp-pick mono">Pick {turn.pick_no}</span>
        <span className="pp-round">Round {turn.round}</span>
      </header>
      <PlayerRow player={target} lead />
      {(gains || target.lasts_pct !== null) && (
        <dl className="pp-figs">
          {target.lasts_pct !== null && (
            <div>
              <dt>Still there at pick {turn.pick_no}</dt>
              <dd className="mono">{Math.round(target.lasts_pct)} %</dd>
            </div>
          )}
          {gains && (
            <div>
              <dt>vs waiting to {target.edge_at_pick}</dt>
              <dd className="mono">+{Math.round(target.edge_pts as number)} pts</dd>
            </div>
          )}
        </dl>
      )}
      {keeps && (
        <p className="pp-keeps">
          No better {target.position ?? 'player'} is likely gone by
          {' '}{target.edge_at_pick} — take him only if nothing above falls
        </p>
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

/** THE BADGE UNDER EACH MARK IS THE CARD AT THAT PICK.
 *
 *  Indexed by pick number rather than by position in the list, because the
 *  plan names three turns and the rule draws four: the fourth mark has no
 *  card and must be left bare rather than borrowing the third one's badge.
 */
function planPositions(plan: Preview): (string | null)[] {
  const atPick = new Map(
    plan.targets.map((turn) => [turn.pick_no, turn.target?.position ?? null]))
  return plan.picks.map((pick) => atPick.get(pick) ?? null)
}

export default function PlanPreview({ teams, slot, tag = '', heading = 'Your plan',
                                      note = null, control = null }: {
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
  /** The seat control, when the caller has one. It belongs in this section's
   *  head rather than in a card of its own: the seat is what the whole
   *  section is about, and a reader who disagrees with the seat we chose
   *  should be able to fix it where they read it. */
  control?: ReactNode
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
        {control}
        {note && <span className="db-sec-note">{note}</span>}
      </div>

      <div className="db-card pp-card">
        {/* WHAT THE READER IS LOOKING AT, in one line. Everything below is
            two things folded together and neither is obvious from a picture
            of it: ESPN's own ranking, and how long each of those players is
            expected to last against the seat's real turns. Said once, at the
            top, so the numbers underneath are read as an answer rather than
            as a scoreboard. */}
        <p className="pp-lede">
          ESPN&rsquo;s order, adjusted for who lasts to your next pick — and
          for your guys.
        </p>
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
            <PickRule picks={plan.picks} positions={planPositions(plan)} />
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
