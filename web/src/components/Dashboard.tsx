import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { disconnectEspn, mintDraftToken,
         type TokenConnectParams, type UpcomingDraft } from '../api'
import { forgetAccount } from '../lib/accountCache'
import { calendarLabel, countdownTo, secondsUntil } from '../lib/countdown'
import { Logo } from './Logo'
import MockLobby from './MockLobby'

// THE PAGE A SIGNED-IN READER GETS. Not the pitch with a card bolted on top:
// somebody who has connected an account has already been sold, and what they
// came back for is one question -- when do I draft, and how do I get in.
//
// THE PAGE IS A CLOCK. Everything in this product is: 30 seconds on a pick,
// 128 picks in a draft, a player priced by how long he lasts. So the page
// opens with the next draft as a clock face -- the room's own monospaced
// digits at display size, ticking -- rather than with a greeting, a logo wall
// or a row of stat tiles. Under it, two columns of everything else you could
// be drafting in, each row led by the same countdown in the same face, so the
// whole page reads as one timeline sorted by "how soon".
//
// The rest of the type is deliberately quiet. In a product whose entire
// subject is numbers, the numbers ARE the display face; giving them a
// decorative headline to sit under would be two things competing to be the
// loud one.
//
// Both lists end in the same place: a join mints a draft token server-side and
// hands the page the four values the bookmarklet's hash used to carry, so the
// connect screen, the retry and the board are one path with two doors into it.

// Seeing the page as a stranger sees it, without destroying anything to do
// it. A connected account can simply be disconnected -- that is a real control
// with a real endpoint behind it. THIS MACHINE'S OWN ESPN LOGIN cannot: it is
// a file the importer wrote (`data/espn_state.json`), no cookie reaches it,
// and "log out" would have to mean deleting a login the user may well want on
// their next refresh.
//
// So the signed-out view is a URL, not a state: `?signedout=1` renders the
// landing page exactly as a visitor gets it, and removing it puts everything
// back. Nothing is stored, nothing is deleted, and the thing being previewed
// is the real page rather than a mock of it.
const PREVIEW_PARAM = 'signedout'

export function previewingSignedOut(): boolean {
  if (typeof window === 'undefined') return false
  return new URLSearchParams(window.location.search).has(PREVIEW_PARAM)
}

// The hero ticks. One interval for the whole page, at the rate the largest
// visible unit actually changes -- a countdown reading `2d 14h` that re-renders
// every second is a second of work per second to redraw the same characters.
// Inside an hour the seconds are the story, so it goes to 1s; outside one, 20s
// is enough to keep the minutes honest.
function useNow(fast: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), fast ? 1000 : 20_000)
    return () => clearInterval(id)
  }, [fast])
  return now
}

// Soonest first, and the leagues whose commissioner has set no date at all
// last -- there is nothing to be late for. Only ever applied to the upcoming
// column: a room already drafting has no date left to sort on.
function byUrgency(a: UpcomingDraft, b: UpcomingDraft): number {
  if (a.draft_at === null || b.draft_at === null) {
    return a.draft_at === b.draft_at ? 0 : (a.draft_at === null ? 1 : -1)
  }
  return Date.parse(a.draft_at) - Date.parse(b.draft_at)
}

/** The league's shape -- "8 teams · Snake" -- for the card's top-right. The
 *  team name is not here: it is yours, and it goes under the league's name
 *  where a possessive belongs, not in the corner with the format. */
function leagueFormat(league: UpcomingDraft): string {
  return [league.teams ? `${league.teams} teams` : null, league.draft_type]
    .filter(Boolean).join(' · ')
}

export default function Dashboard({ leagues, source, onJoin, onOpenRoom }: {
  leagues: UpcomingDraft[]
  /** Where the session came from: an account this browser connected, or this
   *  machine's own saved login. It decides what the sign-out control can
   *  honestly offer. */
  source: 'connected' | 'local' | null
  onJoin: (params: TokenConnectParams) => void
  /** Opens a mock room's waiting room: seats, countdown, who is in. */
  onOpenRoom: (leagueId: string) => void
}) {
  const [joining, setJoining] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [gone, setGone] = useState(false)
  // How many rooms the lobby is offering, reported UP by the list that reads
  // it. The bar states it because the room's own bar states what the room
  // holds in that position -- and a second fetch of the same endpoint, purely
  // so a header could count what the column below it already counted, is two
  // requests that can disagree. Null until the first read lands, which is why
  // the bar says nothing rather than "0 mock rooms open".
  const [roomCount, setRoomCount] = useState<number | null>(null)

  // TWO LISTS, NOT ONE SORTED ONE. "Drafting now" and "drafting on Sunday"
  // are different questions -- one is something to do this minute, the other
  // is something to plan around -- and a single column ordered by urgency
  // made the reader find the boundary themselves every time it moved.
  const live = leagues.filter((l) => l.live)
  const upcoming = leagues.filter((l) => !l.live).sort(byUrgency)
  const soonest = upcoming[0] ?? null
  const soonestSeconds = soonest === null
    ? null : secondsUntil(soonest.draft_at, Date.now())
  // Tick per second only when a second matters -- see useNow. A room already
  // drafting has nothing to count down, so it is the NEXT one that decides.
  const now = useNow(soonestSeconds !== null && soonestSeconds < 3600)

  // Disconnect. Only offered on a CONNECTED account (see the pill below):
  // the cookie is cleared server-side and the stored row with it. A local
  // login gets no button at all -- it is not this page's to delete.
  const signOut = useCallback(async () => {
    try {
      await disconnectEspn()
    } catch {
      // The endpoint clears the cookie even when it refuses, so there is
      // nothing left for this browser either way.
    }
    forgetAccount()
    setGone(true)
  }, [])

  const join = useCallback(async (league: UpcomingDraft) => {
    if (!league.team_id) return
    setJoining(league.league_id)
    setError(null)
    try {
      // Two calls, and the split is the point: this one mints, the page's own
      // connect opens the socket. A token minted here and handed over is the
      // same token the bookmarklet would have delivered, so everything
      // downstream -- the progress screen, the retry, the board -- is the path
      // that already exists and is already tested.
      onJoin(await mintDraftToken(league.league_id, league.team_id, league.season))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setJoining(null)
    }
  }, [onJoin])

  // A disconnect that has already happened: the session is gone, so what
  // belongs on screen is the page a visitor gets. Reloading is how it gets
  // there -- Landing's own probe is what decides which page this is, and it
  // runs on load.
  if (gone) {
    window.location.assign(window.location.pathname)
    return null
  }

  return (
    <main className="db">
      {/* THE ROOM'S OWN BAR, not a second one that resembles it. Same
          `.draft-topbar` classes, same 44px height, same flush-left
          uppercase wordmark, same separator/spacer rhythm, and the session
          state sits exactly where the room's connection pill does -- so
          walking from this page into a draft does not change the furniture,
          it only changes what is under it.

          What differs is only what there is to say: no tabs (there is one
          view), no pick counter (nothing is drafting yet), and the counts
          this page actually holds in the league line's place. */}
      <header className="draft-topbar">
        <span className="draft-topbar-title"><Logo /> ESPN Draft Assist</span>
        <span className="draft-topbar-sep" aria-hidden="true" />
        {/* THE SAME TAB STRIP THE ROOM HAS, in the same slot, carrying this
            page's two views. Drafts is where a signed-in reader lands and is
            marked active on arrival -- the strip states which view you are in
            rather than offering one nameless page and a link off it, which is
            what "Archive on its own" was.

            Drafts links to `/` rather than doing nothing, so the active tab
            behaves like a tab: clicking it reloads the view you are in, and
            arriving here from the archive lands on it already selected. */}
        <nav className="draft-topbar-tabs" aria-label="Views">
          <Link className="draft-tab is-active" to="/" aria-current="page">Drafts</Link>
          <Link className="draft-tab" to="/archive">Archive</Link>
          <Link className="draft-tab" to="/live">Live</Link>
        </nav>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <span className="draft-topbar-league">
          {leagues.length === 1 ? '1 league' : `${leagues.length} leagues`}
          {roomCount !== null && ` · ${roomCount} mock rooms open`}
        </span>
        <span className="draft-topbar-spacer" />
        {/* The session, in the room's own status-pill vocabulary: a dot and
            two words, `--ok` toned, in the exact position the draft room puts
            "listening". A reader learns one shape for "the connection behind
            this page is good" and reads it in both places.

            It is a BUTTON, because unlike the room's pill this one does
            something -- the dot is the state and the click is the way out of
            it, which is one control rather than a label with a second control
            beside it repeating what it refers to.

            Only for a CONNECTED account. The machine's own local login used
            to show a "THIS COMPUTER / preview" pill here; it was operator
            chrome on a page every screenshot goes out from, and the preview
            is still reachable as `?signedout=1` for whoever knows to want
            it. */}
        {source === 'connected' && (
          <button
            type="button"
            className="draft-status-pill draft-status-pill-ok db-session"
            onClick={signOut}
            title="Signed in to ESPN. Click to disconnect this browser."
          >
            <span className="draft-status-dot" aria-hidden="true" />
            SIGNED IN
            <span className="db-session-out" aria-hidden="true">disconnect</span>
          </button>
        )}
      </header>

      {/* The one failure this page can hit: ESPN refused to mint a token for
          a row somebody clicked. Full width above the columns rather than
          inside one, because it is about the click, not about the list -- and
          a reader who has just been refused should not have to work out which
          column their error belongs to. */}
      {error !== null && <p className="db-error">{error}</p>}

      {/* THREE SECTIONS, STACKED, EACH A ROW OF CARDS. They were three fixed
          columns, which is the wrong shape for what they hold: a reader has
          two or three live rooms, a handful of scheduled drafts and twenty-odd
          open mock rooms, so two columns ran dry a fifth of the way down while
          the third scrolled past the fold. Stacked, each section is exactly as
          tall as it needs to be and every card gets the full width of the page
          to be legible in -- which is also what stops a league name being cut
          off mid-word.

          One card shape throughout: a clock, what the draft is, and the way
          in. What changes between sections is what the clock says. */}
      <div className="db-secs">
        <section className="db-sec">
          <div className="db-sec-head">
            <h2 className="db-sec-title">
              {live.length > 0 && <span className="db-dot" aria-hidden="true" />}
              Drafting now
            </h2>
            {live.length > 0 && <span className="mono db-sec-count">{live.length}</span>}
          </div>

          {live.length === 0 ? (
            <p className="db-empty">No room of yours is picking right now.</p>
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
                    {league.name ?? `League ${league.league_id}`}
                  </p>
                  {league.team_name && (
                    <p className="db-league-team">{league.team_name}</p>
                  )}
                  {/* The only filled control on the page, and only ever on a
                      card with a clock running in it: accent here means "you
                      can walk into this right now" and nothing else. */}
                  <button
                    type="button"
                    className="db-go db-league-go"
                    onClick={() => join(league)}
                    disabled={joining !== null || !league.team_id}
                  >
                    {joining === league.league_id ? 'Joining…' : 'Enter the room'}
                    <span className="db-league-go-arrow" aria-hidden="true">→</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="db-sec">
          <div className="db-sec-head">
            <h2 className="db-sec-title">Upcoming drafts</h2>
            {upcoming.length > 0 && (
              <span className="mono db-sec-count">{upcoming.length}</span>
            )}
          </div>

          {upcoming.length === 0 ? (
            <p className="db-empty">Nothing on your ESPN calendar.</p>
          ) : (
            <ul className="db-cards">
              {upcoming.map((league) => {
                const count = countdownTo(secondsUntil(league.draft_at, now))
                return (
                  <li className="db-card db-league" key={league.league_id}>
                    <div className="db-league-top">
                      {/* The countdown leads the card. A draft with no date
                          says so where the digits would be. */}
                      <span className={`mono db-league-when${count?.imminent ? ' is-soon' : ''}${count ? '' : ' is-unset'}`}>
                        {count?.text ?? 'No date'}
                      </span>
                      <span className="db-league-format">{leagueFormat(league)}</span>
                    </div>
                    <p className="db-card-name db-league-name">
                      {league.name ?? `League ${league.league_id}`}
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
                        the poll moves the card up to Drafting now, where the
                        live button is.

                        Two locks, told apart by money. A REAL league's draft
                        is the paid feature, so its button wears the gold and
                        the dollar mark -- the one thing on this page allowed
                        to look like a prize. A lobby mock is free, so its
                        button is the same lock without the price. Mock is
                        read off the name because ESPN's payload carries no
                        flag (its lobby leagues are all named "... Mock");
                        a real league that happens to have mock in its name
                        loses the gold, nothing more. */}
                    <div className="db-league-foot">
                      {(league.name ?? '').toLowerCase().includes('mock') ? (
                        // A mock's waiting room already exists before the
                        // draft does -- seats, countdown, who is in -- so
                        // the card opens it rather than sitting locked. The
                        // draft itself still starts on ESPN's clock; the
                        // waiting room hands the reader in when it does.
                        <button
                          type="button"
                          className="db-go db-league-go db-go-view"
                          onClick={() => onOpenRoom(league.league_id)}
                        >
                          View the room
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="db-go db-league-go db-go-paid"
                          disabled
                          title="Draft night is the paid feature. The room opens when ESPN starts the draft."
                        >
                          <span className="db-paid-coin" aria-hidden="true">
                            <svg viewBox="0 0 24 24">
                              <line x1="12" y1="3" x2="12" y2="21" />
                              <path d="M16.5 6.5H10a3 3 0 0 0 0 6h4a3 3 0 0 1 0 6H7" />
                            </svg>
                          </span>
                          Enter the room
                        </button>
                      )}
                    </div>
                  </li>
                )
              })}
            </ul>
          )}
        </section>

        <MockLobby onOpen={onOpenRoom} now={now} onCount={setRoomCount} />
      </div>
    </main>
  )
}
