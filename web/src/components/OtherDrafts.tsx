import { memo } from 'react'
import { Link } from 'react-router-dom'
import type { UpcomingDraft } from '../api'
import { calendarLabel } from '../lib/countdown'
import { Countdown } from './Countdown'

// THE REST OF THE ACCOUNT, IN ONE LINE EACH.
//
// The next draft is the hero. This is everything else that could be drafting:
// soonest first, live ones pinned to the front, one row apiece. It used to be
// a grid of 300px cards with a format line, a team line, a report-card link
// and a button on each -- a filing cabinet under the two answers the reader
// came for. A league you are not drafting in tonight needs a clock, a name
// and a door.
//
// MEMOIZED, and its props are stable by construction (arrays memoized by the
// page, callbacks from `useCallback`) -- it is the longest list on the page
// and it has nothing to say about the passing of time. The countdowns tick
// themselves (Countdown.tsx), so this only re-renders when the LEAGUES
// change.

export default memo(function OtherDrafts({ live, upcoming, joining, onJoin }: {
  /** Drafting right now, pinned to the front. */
  live: UpcomingDraft[]
  /** The rest, soonest first. */
  upcoming: UpcomingDraft[]
  /** Which league a draft-token mint is in flight for, if any. */
  joining: string | null
  onJoin: (league: UpcomingDraft) => void
}) {
  const rows = [...live, ...upcoming]
  if (rows.length === 0) return null

  return (
    <section className="db-sec od" aria-labelledby="od-h">
      <div className="db-sec-head">
        <h2 className="db-sec-title" id="od-h">Your other leagues</h2>
        <span className="mono db-sec-count">{rows.length}</span>
      </div>
      <ul className="od-rows">
        {rows.map((league) => (
          <li key={league.league_id} className="od-row">
            <span className="od-when mono">
              {league.live
                ? <><span className="db-dot" aria-hidden="true" />live</>
                : <Countdown at={league.draft_at} empty="no date" />}
            </span>
            <span className="od-name">
              <Link className="db-league-link"
                    to={`/league/${encodeURIComponent(league.league_id)}`}>
                {league.name ?? `League ${league.league_id}`}
              </Link>
              <span className="od-meta">
                {league.live ? 'in progress' : calendarLabel(league.draft_at)}
                {league.teams !== null && ` · ${league.teams} teams`}
              </span>
            </span>
            {league.team_id !== null && (
              <button type="button" className="db-go od-go"
                      disabled={joining !== null}
                      onClick={() => onJoin(league)}>
                {joining === league.league_id ? 'Opening…' : 'Open'}
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
})
