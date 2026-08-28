import { memo } from 'react'
import { Link } from 'react-router-dom'
import type { ReportSummary, UpcomingDraft } from '../api'
import { calendarLabel } from '../lib/countdown'
import { Countdown } from './Countdown'

// THE LEAGUES ROW, MEMOIZED.
//
// Lifted out of Dashboard whole, for one reason: it is the biggest list on the
// signed-in page and it has nothing to say about the passing of time. It used
// to be rebuilt every second because the page held a clock -- the countdown on
// each card read a `now` prop threaded down from a `setInterval` at the top of
// the page. The countdown now ticks itself (see Countdown.tsx), so this list
// only re-renders when something about the LEAGUES changes: a join starting, a
// report landing, ESPN reporting a different set.
//
// Its props are therefore all stable by construction -- arrays memoized by the
// page, callbacks from `useCallback` -- or the memo would be a comment rather
// than a guard.

/** The league's shape -- "8 teams · Snake" -- for the card's top-right. The
 *  team name is not here: it is yours, and it goes under the league's name
 *  where a possessive belongs, not in the corner with the format. */
function leagueFormat(league: UpcomingDraft): string {
  return [league.teams ? `${league.teams} teams` : null, league.draft_type]
    .filter(Boolean).join(' · ')
}

function isMock(league: UpcomingDraft): boolean {
  return (league.name ?? '').toLowerCase().includes('mock')
}

export default memo(function LeagueCards({
  real, live, upcoming, reports, joining, buildingFor, onJoin, onBuild,
}: {
  /** Every non-mock league on the account, for the count in the heading. */
  real: UpcomingDraft[]
  /** The ones drafting right now, pinned to the front of the row. */
  live: UpcomingDraft[]
  /** The rest, soonest first. */
  upcoming: UpcomingDraft[]
  /** league_id -> its stored report cards, newest first. */
  reports: Record<string, ReportSummary[]>
  /** Which league a draft-token mint is in flight for, if any. */
  joining: string | null
  /** Which league a report build is in flight for, if any. */
  buildingFor: string | null
  onJoin: (league: UpcomingDraft) => void
  onBuild: (league: UpcomingDraft) => void
}) {
  return (
    <section className="db-sec">
      <div className="db-sec-head">
        <h2 className="db-sec-title">
          {live.length > 0 && <span className="db-dot" aria-hidden="true" />}
          Leagues
        </h2>
        {real.length > 0 && <span className="mono db-sec-count">{real.length}</span>}
      </div>

      {real.length === 0 ? (
        <p className="db-empty">No leagues on this account.</p>
      ) : (
        <ul className="db-cards">
          {live.map((league) => (
            <li className="db-card db-league is-live" key={league.league_id}>
              <div className="db-league-top">
                <span className="db-league-when is-live">
                  <span className="db-dot" aria-hidden="true" />
                  Drafting now
                </span>
                <span className="db-league-format">{leagueFormat(league)}</span>
              </div>
              <p className="db-card-name db-league-name">
                <Link className="db-league-link" to={`/league/${encodeURIComponent(league.league_id)}`}>
                  {league.name ?? `League ${league.league_id}`}
                </Link>
              </p>
              {league.team_name && (
                <p className="db-league-team">{league.team_name}</p>
              )}
              {/* The only filled control on the page, and only ever on a
                  card with a clock running in it: accent here means "you
                  can walk into this right now" and nothing else. */}
              <div className="db-league-foot">
                <button
                  type="button"
                  className="db-go db-league-go"
                  onClick={() => onJoin(league)}
                  disabled={joining !== null || !league.team_id}
                >
                  {joining === league.league_id ? 'Joining…' : 'Enter the room'}
                  <span className="db-league-go-arrow" aria-hidden="true">→</span>
                </button>
                <Link className="db-league-view" to={`/league/${encodeURIComponent(league.league_id)}`}>
                  View league
                </Link>
              </div>
            </li>
          ))}
          {upcoming.map((league) => {
            // A REPORT THAT EXISTS, not merely a row that exists. A
            // pre-draft press of the button below builds the upcoming
            // season, which has no picks yet, and stores a `failed` row
            // saying so -- and on the next poll a truthiness test on the
            // list would swap the button for a link to a page that only
            // says it could not be built, permanently. The page itself
            // still gets every row (a reader may want to see the
            // failure); the card links only to one worth opening, and
            // otherwise keeps offering the build.
            const built = reports[league.league_id]?.find((r) => r.status !== 'failed')
            return (
              <li className="db-card db-league" key={league.league_id}>
                <div className="db-league-top">
                  {/* The countdown leads the card. A draft with no date
                      says so where the digits would be. */}
                  <Countdown at={league.draft_at} className="mono db-league-when"
                                 empty="No date" unsetClass=" is-unset" />
                  <span className="db-league-format">{leagueFormat(league)}</span>
                </div>
                <p className="db-card-name db-league-name">
                  {/* THE LEAGUE IS A PLACE, and the name is the door to
                      it: seasons past, the draft, the team -- see
                      pages/LeaguePage.tsx. The button below is only the
                      draft room's door, which is locked until ESPN opens
                      it; the league itself is open any time. */}
                  <Link className="db-league-link" to={`/league/${encodeURIComponent(league.league_id)}`}>
                    {league.name ?? `League ${league.league_id}`}
                  </Link>
                </p>
                <p className="db-league-team">
                  {league.team_name}
                  {league.team_name && league.draft_at && (
                    <span className="db-league-team-sep" aria-hidden="true">·</span>
                  )}
                  {league.draft_at && calendarLabel(league.draft_at)}
                  {!league.team_name && !league.draft_at && 'No date set'}
                </p>
                {/* THE BUTTON IS LOCKED. ESPN mints a draft token only
                    once the room is actually open -- a join on a draft
                    two days out comes back 502 (api/drafts.py's
                    draft_token) -- so the control is shown but never
                    clickable here, and the moment ESPN opens the room
                    the poll moves the card to the front of the row,
                    where the live button is.

                    A REAL league's draft is the paid feature, so its
                    button wears the gold and the dollar mark -- the one
                    thing on this page allowed to look like a prize. The
                    paywall lives INSIDE the room (DraftRoom): a reader
                    gets the board, the clock and the ranking before
                    being asked for anything, which is a better place to
                    ask than a list of leagues they have not seen this
                    tool work on yet. */}
                <div className="db-league-foot">
                  <button
                    type="button"
                    className="db-go db-league-go db-go-paid"
                    disabled
                    title="The room opens when ESPN starts the draft."
                  >
                    <span className="db-paid-coin" aria-hidden="true">
                      <svg viewBox="0 0 24 24">
                        <line x1="12" y1="3" x2="12" y2="21" />
                        <path d="M16.5 6.5H10a3 3 0 0 0 0 6h4a3 3 0 0 1 0 6H7" />
                      </svg>
                    </span>
                    Enter the room
                  </button>
                  <Link className="db-league-view" to={`/league/${encodeURIComponent(league.league_id)}`}>
                    View league
                  </Link>
                  {!isMock(league) && (
                    built ? (
                      <Link
                        className="db-go db-league-go db-go-view"
                        to={`/leagues/${league.league_id}/report/${built.season}`}
                      >
                        Report card
                      </Link>
                    ) : (
                      <button
                        type="button"
                        className="db-go db-league-go db-go-view"
                        disabled={buildingFor === league.league_id}
                        onClick={() => onBuild(league)}
                      >
                        {buildingFor === league.league_id ? 'Building…' : 'Build report card'}
                      </button>
                    )
                  )}
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
})
