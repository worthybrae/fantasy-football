import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  buildLeagueReport, fetchFavorites, fetchLeagueReports, fetchPlayers,
  mintDraftToken,
  type Player, type ReportSummary, type TokenConnectParams, type UpcomingDraft,
} from '../api'
import { calendarLabel, countdownTo, secondsUntil } from '../lib/countdown'
import FavoritesPicker from './FavoritesPicker'
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
// it. There is no sign-out control on this page: a connected account's
// session could be cleared, but THIS MACHINE'S OWN ESPN LOGIN cannot -- it is
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

function isMock(league: UpcomingDraft): boolean {
  return (league.name ?? '').toLowerCase().includes('mock')
}

/** The league's shape -- "8 teams · Snake" -- for the card's top-right. The
 *  team name is not here: it is yours, and it goes under the league's name
 *  where a possessive belongs, not in the corner with the format. */
function leagueFormat(league: UpcomingDraft): string {
  return [league.teams ? `${league.teams} teams` : null, league.draft_type]
    .filter(Boolean).join(' · ')
}

export default function Dashboard({ leagues, onJoin, onOpenRoom }: {
  leagues: UpcomingDraft[]
  onJoin: (params: TokenConnectParams) => void
  /** Opens a mock room's waiting room: seats, countdown, who is in. */
  onOpenRoom: (leagueId: string) => void
}) {
  const [joining, setJoining] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  // How many rooms the lobby is offering, reported UP by the list that reads
  // it. The bar states it because the room's own bar states what the room
  // holds in that position -- and a second fetch of the same endpoint, purely
  // so a header could count what the column below it already counted, is two
  // requests that can disagree. Null until the first read lands, which is why
  // the bar says nothing rather than "0 mock rooms open".
  const [roomCount, setRoomCount] = useState<number | null>(null)

  const navigate = useNavigate()
  // league_id -> its stored reports, newest first. One read per non-mock
  // league, repeated whenever the league list is re-read -- about once a
  // minute, alongside Landing's own account poll -- rather than once at
  // mount: a report the room finishes building at the end of a draft, or
  // one started from another tab, should get its "Report card" link here
  // without a reload. The GET is a cheap read of one stored table, so
  // paying it once a minute per league costs nothing worth guarding.
  // Undefined until the first read lands, so the card shows no report link
  // rather than a wrong one.
  // YOUR GUYS. Three states, and they are three different panels:
  //   null   -- there is no account session to read them from (the server's
  //             own 401), or the read failed. No panel at all: this page
  //             cannot offer to save something it cannot load.
  //   []     -- signed in, nothing picked yet. The onboarding picker.
  //   [...]  -- picked. The compact card, with a way back into the picker.
  // Read once per mount rather than on the league poll: a favourites list
  // changes when somebody changes it, which is on this page, in front of us.
  const [favorites, setFavorites] = useState<string[] | null>(null)
  const [editing, setEditing] = useState(false)
  // The board, fetched only once there is a favourites panel to spend it on
  // -- it is a 250-row payload and a dashboard with no session for it has no
  // use for a single row.
  const [board, setBoard] = useState<Player[]>([])
  // A board that never arrives is a state, not a spinner. Without this the
  // section sat on "Loading the board…" for as long as the page was open,
  // which is a promise it had already stopped keeping.
  const [boardError, setBoardError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchFavorites()
      .then((ids) => { if (!cancelled) setFavorites(ids) })
      .catch(() => { if (!cancelled) setFavorites(null) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (favorites === null || board.length > 0) return
    let cancelled = false
    fetchPlayers()
      .then((rows) => { if (!cancelled) setBoard(rows) })
      .catch((e) => {
        if (!cancelled) setBoardError(e instanceof Error ? e.message : String(e))
      })
    return () => { cancelled = true }
  }, [favorites, board.length])

  const [reports, setReports] = useState<Record<string, ReportSummary[]>>({})
  const [buildingFor, setBuildingFor] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    for (const league of leagues) {
      if (isMock(league)) continue
      fetchLeagueReports(league.league_id)
        .then((rows) => { if (!cancelled) setReports((r) => ({ ...r, [league.league_id]: rows })) })
        .catch(() => { /* no link, then */ })
    }
    return () => { cancelled = true }
  }, [leagues])

  const build = useCallback(async (league: UpcomingDraft) => {
    const season = league.season ?? new Date().getFullYear()
    setBuildingFor(league.league_id)
    try {
      await buildLeagueReport(league.league_id, season)
      navigate(`/leagues/${league.league_id}/report/${season}#building`)
    } catch (err) {
      setError(String((err as Error).message || err))
      setBuildingFor(null)
    }
  }, [navigate])

  // TWO ROWS: LEAGUES, THEN MOCKS. A mock room you have taken a seat in
  // arrives in the same ESPN list as your real leagues, but it is a different
  // kind of thing -- free, disposable, one of a hundred like it -- and it
  // belongs with the lobby it came from, in the lobby's own card, rather
  // than between your leagues wearing a league's card. Mock is read off the
  // name because ESPN's payload carries no flag (its lobby leagues are all
  // named "... Mock"); a real league that happens to have mock in its name
  // lands in the mock row, nothing worse.
  const real = leagues.filter((l) => !isMock(l))
  const mocks = leagues.filter(isMock)
  // Live first, then soonest: one list, because a reader has two or three
  // of these and the boundary between "now" and "Sunday" is the live dot.
  const live = real.filter((l) => l.live)
  const upcoming = real.filter((l) => !l.live).sort(byUrgency)
  const soonest = upcoming[0] ?? null
  const soonestSeconds = soonest === null
    ? null : secondsUntil(soonest.draft_at, Date.now())
  // Tick per second only when a second matters -- see useNow. A room already
  // drafting has nothing to count down, so it is the NEXT one that decides.
  const now = useNow(soonestSeconds !== null && soonestSeconds < 3600)

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

  return (
    <main className="db">
      {/* THE ROOM'S OWN BAR, not a second one that resembles it. Same
          `.draft-topbar` classes, same 44px height, same flush-left
          uppercase wordmark, same separator/spacer rhythm -- so walking from
          this page into a draft does not change the furniture, it only
          changes what is under it.

          What differs is only what there is to say: no pick counter (nothing
          is drafting yet), no connection pill (nothing is listening), and the
          counts this page actually holds in the league line's place. */}
      <header className="draft-topbar">
        <span className="draft-topbar-title"><Logo /> ESPN Draft Assist</span>
        <span className="draft-topbar-sep" aria-hidden="true" />
        {/* THE SAME TAB STRIP THE ROOM HAS, in the same slot, carrying this
            page's views. Home is where a signed-in reader lands and is
            marked active on arrival -- the strip states which view you are in
            rather than offering one nameless page and a link off it, which is
            what "Archive on its own" was.

            Home links to `/` rather than doing nothing, so the active tab
            behaves like a tab: clicking it reloads the view you are in, and
            arriving here from the archive lands on it already selected. */}
        <nav className="draft-topbar-tabs" aria-label="Views">
          <Link className="draft-tab is-active" to="/" aria-current="page">Home</Link>
          <Link className="draft-tab" to="/archive">Data</Link>
          <Link className="draft-tab" to="/live">Live</Link>
        </nav>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <span className="draft-topbar-league">
          {real.length === 1 ? '1 league' : `${real.length} leagues`}
          {roomCount !== null && ` · ${roomCount} mock rooms open`}
        </span>
        <span className="draft-topbar-spacer" />
      </header>

      {/* The one failure this page can hit: ESPN refused to mint a token for
          a row somebody clicked. Full width above the columns rather than
          inside one, because it is about the click, not about the list -- and
          a reader who has just been refused should not have to work out which
          column their error belongs to. */}
      {error !== null && <p className="db-error">{error}</p>}

      {/* TWO SECTIONS, STACKED, EACH A ROW OF CARDS. They were three fixed
          columns, which is the wrong shape for what they hold: a reader has
          a handful of leagues and twenty-odd open mock rooms, so two columns
          ran dry a fifth of the way down while the third scrolled past the
          fold. Stacked, each section is exactly as tall as it needs to be
          and every card gets the full width of the page to be legible in --
          which is also what stops a league name being cut off mid-word.

          One card shape per row: a clock, what the draft is, and the way in.
          A league that is drafting now leads the row with the live dot where
          its clock would be; the rest follow by how soon. */}
      <div className="db-secs">
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
                      onClick={() => join(league)}
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
                const count = countdownTo(secondsUntil(league.draft_at, now))
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
                      <span className={`mono db-league-when${count?.imminent ? ' is-soon' : ''}${count ? '' : ' is-unset'}`}>
                        {count?.text ?? 'No date'}
                      </span>
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
                            onClick={() => build(league)}
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

        {/* YOUR GUYS, between the leagues and the lobby. Above the mock
            rooms deliberately: this is a thing to do once, before a draft,
            and a reader who has not done it should meet it before he meets
            twenty rooms he could join instead. */}
        {favorites !== null && (
          <section className="db-sec">
            <div className="db-sec-head">
              <h2 className="db-sec-title">Your guys</h2>
              {favorites.length > 0 && !editing && (
                <span className="mono db-sec-count">{favorites.length}</span>
              )}
            </div>
            {board.length === 0 ? (
              boardError !== null ? (
                <p className="db-error db-fav-error" role="alert">
                  The player list did not load ({boardError}), so there is
                  nothing to pick from. Reload to try again.
                </p>
              ) : <p className="db-empty">Loading the board…</p>
            ) : editing || favorites.length === 0 ? (
              <FavoritesPicker
                players={board}
                initial={favorites}
                onSaved={(ids) => { setFavorites(ids); setEditing(false) }}
                onCancel={favorites.length > 0 ? () => setEditing(false) : undefined}
              />
            ) : (
              <div className="db-card db-fav-card">
                <ul className="db-fav-list">
                  {favorites.map((id, i) => {
                    const player = board.find((p) => p.player_id === id)
                    return (
                      <li key={id} className="db-fav-row">
                        <span className="db-fav-ord mono">{i + 1}</span>
                        {player && (
                          <span className={`pos-badge pos-badge-${player.position.toLowerCase()}`}>
                            {player.position}
                          </span>
                        )}
                        <span className="db-fav-name">{player?.name ?? id}</span>
                      </li>
                    )
                  })}
                </ul>
                <div className="db-fav-foot">
                  <p className="db-fav-note">
                    The draft room stars them, and the plan reaches for them a
                    round earlier than it would reach for anybody else.
                  </p>
                  <button type="button" className="db-go db-league-go db-go-view"
                          onClick={() => setEditing(true)}>
                    Edit
                  </button>
                </div>
              </div>
            )}
          </section>
        )}

        <MockLobby
          onOpen={onOpenRoom}
          seated={mocks}
          onEnter={join}
          joining={joining}
          now={now}
          onCount={setRoomCount}
        />
      </div>
    </main>
  )
}
