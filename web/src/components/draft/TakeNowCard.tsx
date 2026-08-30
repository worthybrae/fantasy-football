import type { ReactNode } from 'react'
import type { LiveCandidate, LivePlanTurn, PlanPlayer, Player } from '../../api'
import { draftHint, type DraftGate } from './draftGate'
import { turnFor } from './TargetCards'
import { lastsBand } from './tone'

// ONE ANSWER, IN A SENTENCE.
//
// The three target cards this replaces as the room's default are not wrong --
// they are the plan's own answer for a turn, with its own reasoning under
// each name -- they are simply more than the question deserves. A person
// drafting once a year, thirty seconds on the clock, is asking "who do I take
// and why", and three cards of five bar meters and six figures each is a
// research tool answering it.
//
// So: one name at reading size, one sentence, two names to fall back on, and
// the button. Nothing on this card is computed here -- the target, the
// alternates and the reasons all come down in `/api/live/state`'s `plan`
// (scoring/plan.py), the same payload TargetCards draws, so the simple view
// and the cheat sheet can never name two different players for one pick.
//
// Off the clock the card stops recommending and starts forecasting, because
// "take him" is a sentence about a pick you are holding. What is useful while
// you wait is which of your plan's names are likely to survive to your turn,
// which is what the payload's own percentages say.

/** The plan's word for the target's position, when the card needs one and the
 *  join table has not landed. */
function posBadge(position: string): ReactNode {
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

function posHue(position: string): string {
  const key = position.toLowerCase()
  return ['qb', 'rb', 'wr', 'te', 'k', 'dst'].includes(key)
    ? `var(--pos-${key})` : 'var(--text-3)'
}

// -- the reason, in the reader's words --------------------------------------
//
// Every string the plan produces is rule-derived (scoring/plan.py's
// `reasons_for`) and written for a card that has room for five of them. This
// card has room for one, and it has to read like a person said it -- "RBs dry
// up before pick 11", not "biggest drop-off at RB before pick 11".
//
// So the pros are MATCHED, not paraphrased: each pattern below is one of that
// function's own f-strings, and a pro that matches none of them is left out
// rather than half-translated. Nothing is invented -- if the plan did not give
// a reason this card cannot match, the sentence is the availability clause
// alone, and if there is no number for that either there is no sentence at
// all. A made-up reason under a name is the one failure this card cannot
// have.
//
// THE ORDER IS THE STRENGTH OF THE ARGUMENT, not the plan's own print order.
// The plan prints the star and ESPN's rank first because a card listing five
// reasons should open with who he is; a card with room for ONE has to lead
// with the reason the pick is being made now rather than later, which is the
// scarcity, the points, or the hole on the roster.
const LEADS: [RegExp, (m: RegExpMatchArray) => string][] = [
  // "biggest drop-off at RB before pick 11"
  [/^biggest drop-off at (\w+) before pick (\d+)$/,
    (m) => `${m[1]}s dry up before pick ${m[2]}`],
  // "+4.2 pts over the next RB you'd get at pick 11"
  [/^\+([\d.]+) pts over the next (\w+) you'd get at pick (\d+)$/,
    (m) => `he's worth about ${Math.round(Number(m[1]))} more points than the `
      + `next ${m[2]} you would get at pick ${m[3]}`],
  // "fills RB2" / "flex" / "bench" -- the bench is not an argument for
  // taking somebody now, so it is not one of these.
  [/^fills (.+)$/, (m) => `he fills your ${m[1]} spot`],
  [/^flex$/, () => 'he fills your flex spot'],
  // "ESPN's #4 overall"
  [/^ESPN's #(\d+) overall$/, (m) => `ESPN has him ${m[1]} overall`],
  [/^★ favourite$/, () => "he's one of your guys"],
  // "ADP 14 vs ESPN 22 -- may last" (an em dash in the payload)
  [/^ADP (\d+) vs ESPN (\d+) — may last$/,
    (m) => `the wider market has him at ${m[1]} where ESPN has him ${m[2]}`],
]

/** The strongest reason this card can say in plain words, or null when the
 *  plan gave none it can. */
function leadClause(pros: string[]): string | null {
  for (const [pattern, say] of LEADS) {
    for (const pro of pros) {
      const m = pro.match(pattern)
      if (m) return say(m)
    }
  }
  return null
}

/** WILL HE BE THERE, as the second half of the sentence.
 *
 *  The pick is named only when the lead has not already named it -- "RBs dry
 *  up before pick 11 - he's 65% to be there at pick 11" says pick 11 twice in
 *  a sentence a reader has three seconds for. Matched on the WHOLE token: a
 *  substring test reads "pick 1" inside "pick 11" and drops the one number
 *  the second half of the sentence exists to give. */
function availClause(pct: number, atPick: number | null, lead: string | null): string {
  const shown = Math.round(pct)
  const named = atPick !== null
    && new RegExp(`\\bpick ${atPick}\\b`).test(lead ?? '')
  const at = atPick !== null && !named ? ` at pick ${atPick}` : ''
  return lastsBand(shown) === 'low'
    ? `only ${shown}% to be there${at}`
    : `he's ${shown}% to be there${at}`
}

/** The whole sentence, or null when there is nothing honest to say.
 *
 *  Exported for its own test: this is the one piece of the room that turns the
 *  model's words into the reader's, and it has to be checkable without a
 *  rendered card around it. */
export function reasonSentence(source: {
  pros: string[]
  lastsPct: number | null
  lastsAtPick: number | null
  edgePts: number | null
  edgeAtPick: number | null
  position: string
}): string | null {
  let lead = leadClause(source.pros)
  // NO PLAN, BUT STILL NUMBERS. A room whose seat has not resolved, or one
  // watching somebody else draft, has candidate rows and no reasoning at all
  // -- and the edge on those rows is the same figure the plan would have
  // argued from, so the sentence is built from it directly rather than
  // leaving the card silent.
  if (lead === null && source.pros.length === 0 && source.edgePts !== null
      && Math.round(source.edgePts) > 0) {
    const at = source.edgeAtPick === null ? '' : ` at pick ${source.edgeAtPick}`
    lead = `he's worth about ${Math.round(source.edgePts)} more points than the `
      + `next ${source.position || 'player'} you would get${at}`
  }
  const avail = source.lastsPct === null
    ? null : availClause(source.lastsPct, source.lastsAtPick, lead)
  const parts = [lead, avail].filter((p): p is string => p !== null)
  if (parts.length === 0) return null
  const said = parts.join(' — ')
  return `${said.charAt(0).toUpperCase()}${said.slice(1)}.`
}

/** A candidate row read as a plan row, for a room with no plan -- the same
 *  fallback TargetCards makes, and for the same reason: the top of ESPN's
 *  board is still worth naming, with no reasoning attached to it. */
function fromCandidate(c: LiveCandidate): PlanPlayer {
  return {
    player_id: c.player_id,
    lasts_pct: c.lasts_pct,
    edge_pts: c.edge_pts,
    edge_at_pick: c.edge_at_pick,
    pros: [],
    cons: [],
  }
}

interface TakeNowCardProps {
  /** The reader's remaining turns, straight off `/api/live/state`. */
  plan: LivePlanTurn[]
  /** The list the table below renders, used to join a planned player back to
   *  his position and his star, and to hand the Draft button a real row. */
  candidates: LiveCandidate[]
  players: Record<string, Player>
  onDraft: (c: LiveCandidate) => void
  /** Whether the reader may actually send a pick right now: his turn, on a
   *  live socket, in a room he has paid for. The SAME gate the cheat sheet's
   *  cards and table use, computed once by DraftRoom -- the two views must
   *  never disagree about whether a pick can be sent. */
  isMyTurn: boolean
  /** Whether the pick on the clock is HIS, which is true in a room he cannot
   *  draft from too. Decides whether this card recommends or forecasts; see
   *  `turnFor`, whose comment carries the whole argument for keeping these
   *  two booleans apart. */
  onTheClock?: boolean
  pickNo?: number | null
  onOpenPlayer: (c: LiveCandidate) => void
  /** Player ids the BOARD reports as drafted. Current the moment a pick
   *  lands; the plan is not, and does not catch up until the next ranking
   *  finishes about a second later. Without this the card spent that second
   *  offering a player who was already gone, with a live Draft button on him
   *  -- the list below had dropped him on the same frame. */
  draftedIds?: Set<string>
  /** Why the button is off, when it is (see draftGate.ts). Absent is the
   *  plain "not your turn", which is what a room with no billing and a
   *  healthy socket has. */
  gate?: DraftGate
}

// Module-level, so the default identity never changes between renders.
const NOBODY = new Set<string>()

export default function TakeNowCard({
  plan, candidates, players, onDraft, isMyTurn, onTheClock = false,
  pickNo = null, onOpenPlayer, draftedIds = NOBODY, gate,
}: TakeNowCardProps) {
  const turn = turnFor(plan, pickNo, onTheClock)
  const stillThere = (id: string) => !draftedIds.has(id)
  // The plan's names for this turn, minus anyone the board says has just
  // gone. If the target is one of them the first surviving alternate leads,
  // which is exactly what "if he's gone" promised.
  const planned: PlanPlayer[] = turn
    ? [...(turn.target ? [turn.target] : []), ...turn.alternates]
      .filter((row) => stillThere(row.player_id))
    : []
  // EVERY NAME THE PLAN HAD IS GONE, and the recompute that will name new
  // ones is a second away. The top of the board (already drafted-filtered) is
  // a better answer for that second than an empty column under a clock -- and
  // it carries no reasoning, so the card cannot claim the plan chose him.
  const planEmptied = turn !== null && planned.length === 0
    && (turn.target !== null || turn.alternates.length > 0)
  const rows: PlanPlayer[] = turn !== null && !planEmptied
    ? planned.slice(0, 3)
    : candidates.filter((c) => stillThere(c.player_id)).slice(0, 3).map(fromCandidate)
  const noClearTarget = turn !== null && turn.target === null

  if (rows.length === 0) {
    // The plan's honest "nobody cleared the bar for this turn". Said in the
    // card's own slot so the reader sees the gap rather than a room that
    // quietly stops recommending.
    if (!noClearTarget) return null
    return (
      <section className="takenow">
        <div className="takenow-head">
          <span className="draft-cap takenow-cap">Take now</span>
        </div>
        <p className="takenow-none" role="status">
          No standout — take the best name below (ESPN&apos;s order).
        </p>
      </section>
    )
  }

  const byId = new Map(candidates.map((c) => [c.player_id, c]))
  // Same rule TargetCards follows: the card is a recommendation only when the
  // turn it describes is the pick actually on the clock.
  // With no plan at all (a seat that has not resolved, a room being watched
  // from outside) there is no turn to match against, so the clock itself
  // decides: on it, the card recommends the top of ESPN's board; off it, it
  // forecasts the same three names.
  const onClockTurn = turn === null
    ? onTheClock
    : pickNo !== null && turn.pick_no === pickNo

  const lead = rows[0]
  const leadCandidate = byId.get(lead.player_id)
  const leadPlayer = players[lead.player_id]
  const position = leadCandidate?.position ?? leadPlayer?.position ?? ''
  const alts = rows.slice(1)

  function name(row: PlanPlayer): string {
    return players[row.player_id]?.name ?? row.player_id
  }

  // A name opens the profile, here as everywhere else in the room -- but only
  // for a player the current list still carries, since that row is the seed
  // the popup paints from.
  function nameButton(row: PlanPlayer, className: string): ReactNode {
    const c = byId.get(row.player_id)
    return (
      <button
        type="button"
        className={className}
        onClick={() => c && onOpenPlayer(c)}
        disabled={!c}
        title="Open profile"
      >
        {name(row)}
      </button>
    )
  }

  if (!onClockTurn) {
    // WAITING. No recommendation, because the pick is not his: the useful
    // answer is which of the plan's names for his next turn are likely to
    // survive to it, which is exactly what the plan's own percentages say.
    return (
      <section className="takenow is-forecast">
        <div className="takenow-head">
          <span className="draft-cap takenow-cap">Up next</span>
          {turn !== null && (
            <span className="takenow-hint">
              pick {turn.pick_no} · round {turn.round}
            </span>
          )}
        </div>
        <p className="takenow-likely">
          {/* NOT "likely there", which is what this said while sitting over
              a chip reading 30% in red. The caption is neutral and the chips
              carry the odds, because they are the only thing here that
              actually knows them. */}
          <span className="takenow-likely-cap">In the running:</span>
          {rows.map((row) => (
            <span key={row.player_id} className="takenow-likely-row">
              {nameButton(row, 'takenow-alt')}
              <LastsChip pct={row.lasts_pct} />
            </span>
          ))}
        </p>
      </section>
    )
  }

  // ON THE CLOCK THE FIGURES COME OFF THE CANDIDATE ROW, not the plan's --
  // both are measured to the reader's next turn in that case, but only the
  // candidate row is measured to it by contract (see TargetCards' own note
  // where it makes the identical choice).
  //
  // ONE OBJECT, chosen once. Reading each figure with `??` mixed the two
  // sources whenever the candidate row had a null in it: a percentage from
  // the board and a pick number from the plan are measured to different
  // turns, and a sentence built from both is a sentence about no turn at all.
  const figures = leadCandidate
    ? {
        lastsPct: leadCandidate.lasts_pct, lastsAtPick: leadCandidate.lasts_at_pick,
        edgePts: leadCandidate.edge_pts, edgeAtPick: leadCandidate.edge_at_pick,
      }
    : {
        lastsPct: lead.lasts_pct, lastsAtPick: null,
        edgePts: lead.edge_pts, edgeAtPick: lead.edge_at_pick,
      }
  const sentence = reasonSentence({ pros: lead.pros, position, ...figures })

  return (
    <section className="takenow">
      <div className="takenow-head">
        <span className="draft-cap takenow-cap">Take now</span>
        {turn !== null && (
          <span className="takenow-hint">
            pick {turn.pick_no} · round {turn.round}
          </span>
        )}
      </div>

      <div className="takenow-body">
        {leadPlayer?.headshot ? (
          <img className="takenow-shot" src={leadPlayer.headshot} alt=""
               loading="lazy" style={{ borderColor: posHue(position) }} />
        ) : (
          <span className="takenow-shot is-none">{posBadge(position || '?')}</span>
        )}
        <span className="takenow-id">
          <span className="takenow-name-line">
            {leadCandidate?.favourite && (
              <span className="takenow-star" role="img"
                    aria-label="One of your guys">★</span>
            )}
            {nameButton(lead, 'takenow-name-btn')}
          </span>
          <span className="takenow-sub mono">
            {position}
            {leadPlayer?.team ? ` · ${leadPlayer.team}` : ''}
            {leadPlayer?.bye ? ` · BYE ${leadPlayer.bye}` : ''}
          </span>
        </span>
        <button
          type="button"
          className="avail-draft-btn takenow-draft"
          disabled={!isMyTurn || !leadCandidate}
          title={gate ? draftHint(gate) : (isMyTurn ? undefined : 'Not your turn yet')}
          onClick={() => leadCandidate && onDraft(leadCandidate)}
        >
          Draft
        </button>
      </div>

      {sentence && <p className="takenow-why">{sentence}</p>}

      {/* THE ONE THING ARGUING AGAINST HIM, if the plan named one. First
          only: a card with a paragraph of doubt under a thirty-second clock
          is a card that gets skipped, and the plan sorts its cons so that
          the one that argues against the PICK leads. In the plan's own
          words, like everything else on this card. */}
      {lead.cons.length > 0 && (
        <p className="takenow-watch">
          <span className="takenow-watch-cap">Watch:</span> {lead.cons[0]}
        </p>
      )}

      {alts.length > 0 && (
        <p className="takenow-backups">
          <span className="takenow-backups-cap">If he&apos;s gone:</span>{' '}
          {alts.map((row, i) => (
            <span key={row.player_id}>
              {i > 0 && <span className="takenow-sep">, </span>}
              {nameButton(row, 'takenow-alt')}
            </span>
          ))}
        </p>
      )}
    </section>
  )
}

/** The percentage, in the three bands the whole simple room reads it in.
 *  A dash carries no tint: `null` means there is no next turn to survive to,
 *  and colouring it would claim a probability nobody computed. */
export function LastsChip({ pct }: { pct: number | null }) {
  if (pct === null) return <span className="lasts-chip is-none">—</span>
  const shown = Math.round(pct)
  return (
    <span className={`lasts-chip is-${lastsBand(shown)} mono`}>{shown}%</span>
  )
}
