import { Link } from 'react-router-dom'
import type { UpcomingDraft } from '../api'
import { calendarLabel } from '../lib/countdown'
import { ordinal } from '../lib/seat'
import { Countdown } from './Countdown'

// WHEN IS MY DRAFT, AND HOW DO I GET IN.
//
// The two questions a signed-in reader opened the page with, answered above
// everything else and with nothing around them. No card frame: this is not
// one item in a list, it is the answer, and a border would make it one of
// several.
//
// THE PAGE IS A CLOCK. Everything in this product is -- 30 seconds on a pick,
// 128 picks in a draft, a player priced by how long he lasts -- so the hero
// is the room's own monospaced digits at display size, ticking, rather than a
// greeting or a row of stat tiles.

/** WHICH SEAT THE PLAN IS FOR, AND WHERE THAT CAME FROM.
 *
 *  THE BUG THIS EXISTS TO END. The plan used to be built for whatever seat
 *  the reader had last set in a select -- a number this page invented a
 *  default for and then never revisited -- so an owner ESPN has sitting sixth
 *  was reading a plan for the fifth seat, under a heading that named their
 *  real league. Every pick number in it was wrong by one turn and nothing on
 *  the page said where the number had come from.
 *
 *  So the seat is attributed, always. ESPN's own published order is the
 *  answer when there is one and it says so; before the commissioner sets one
 *  there is no answer, and the honest thing is to say THAT and let the reader
 *  choose. The control stays either way -- a reader who is between leagues,
 *  or curious about the seat next to theirs, is entitled to ask -- and when
 *  they do, the page keeps saying what ESPN's answer was.
 */
export function SeatLine({ teams, slot, espnSlot, onPick }: {
  teams: number
  slot: number
  /** ESPN's own answer, or null before the draft order is published. */
  espnSlot: number | null
  onPick: (slot: number) => void
}) {
  const overridden = espnSlot !== null && slot !== espnSlot
  return (
    <p className="ndh-seat">
      <label className="ndh-seat-lab" htmlFor="ndh-seat">Seat</label>
      <select id="ndh-seat" className="ndh-seat-select"
              aria-label="Which seat to plan for" value={slot}
              onChange={(event) => onPick(Number(event.target.value))}>
        {Array.from({ length: teams }, (_, i) => i + 1).map((n) => (
          <option key={n} value={n}>{ordinal(n)}</option>
        ))}
      </select>
      <span className="ndh-seat-say">
        {espnSlot !== null && !overridden
          && `seat ${espnSlot} of ${teams} · from ESPN`}
        {overridden
          && `seat ${slot} of ${teams} · ESPN has you at ${espnSlot}`}
        {espnSlot === null
          && `seat ${slot} of ${teams} · ESPN hasn’t set the order yet`}
      </span>
    </p>
  )
}

export default function NextDraftHero({
  next, teams, slot, espnSlot, joining, onJoin, onPickSeat, onConnect,
}: {
  /** The draft this page is about: the one drafting now if there is one,
   *  otherwise the soonest with a date. Null is a connected account with
   *  nothing to be late for. */
  next: UpcomingDraft | null
  /** The seat the whole page is drawn for. Null for a league of a shape
   *  neither the planner nor the archive answers about. */
  teams: number | null
  slot: number | null
  espnSlot: number | null
  /** Which league a draft-token mint is in flight for, if any. */
  joining: string | null
  onJoin: (league: UpcomingDraft) => void
  onPickSeat: (slot: number) => void
  /** Opens the bookmarklet walkthrough. */
  onConnect: () => void
}) {
  // NO DRAFT TO BE READY FOR. A connected account whose leagues have no date
  // set yet, or none at all. The hero is then the thing that WOULD change
  // that: connect the league whose draft is coming.
  if (next === null) {
    return (
      <section className="ndh ndh-none" aria-labelledby="ndh-h">
        <p className="lp-cap" id="ndh-h">No draft date yet</p>
        <h1 className="ndh-title">Connect your ESPN league</h1>
        <p className="ndh-say">
          Click the bookmark once on any ESPN fantasy page and the league whose
          draft is coming shows up here, with a countdown and a way in.
        </p>
        <div className="ndh-acts">
          <button type="button" className="lp-cta" onClick={onConnect}>
            Connect your ESPN league
          </button>
          <a className="lp-bar-link" href="#mock-lobby">
            or try a mock draft — free
          </a>
        </div>
      </section>
    )
  }

  return (
    <section className="ndh" aria-labelledby="ndh-h">
      <p className="lp-cap" id="ndh-h">
        {next.live ? 'Drafting now' : 'Your next draft'}
      </p>
      <p className="ndh-clock mono">
        {next.live
          ? <><span className="db-dot" aria-hidden="true" />live</>
          : <Countdown at={next.draft_at} />}
      </p>
      <h1 className="ndh-title">
        <Link className="db-league-link"
              to={`/league/${encodeURIComponent(next.league_id)}`}>
          {next.name ?? `League ${next.league_id}`}
        </Link>
      </h1>
      <p className="ndh-when mono">
        {next.live ? 'in progress' : calendarLabel(next.draft_at)}
        {next.team_name && <span className="ndh-team"> · {next.team_name}</span>}
      </p>
      {teams !== null && slot !== null && (
        <SeatLine teams={teams} slot={slot} espnSlot={espnSlot}
                  onPick={onPickSeat} />
      )}
      {next.team_id !== null && (
        <div className="ndh-acts">
          <button type="button" className="lp-cta"
                  disabled={joining !== null}
                  onClick={() => onJoin(next)}>
            {joining === next.league_id ? 'Opening…' : 'Open the draft room'}
          </button>
        </div>
      )}
    </section>
  )
}
