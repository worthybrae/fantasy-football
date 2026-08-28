import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  buildLeagueReport, fetchFavorites, fetchLeagueReports, mintDraftToken,
  type ReportSummary, type TokenConnectParams, type UpcomingDraft,
} from '../api'
import LeagueCards from './LeagueCards'
import { Logo } from './Logo'
import MockLobby from './MockLobby'
import { PickerBoundary, PickerFallback } from './PickerBoundary'
import YourGuys from './YourGuys'

// THE PICKER IS NOT IN THIS PAGE'S BUNDLE. It is a 250-row board with a photo
// per row, opened by a fraction of the readers who load this page and never
// during a draft -- so it arrives when somebody asks for it, not before.
const FavoritesModal = lazy(() => import('./FavoritesModal'))

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

// THE PAGE NO LONGER HOLDS A CLOCK. It used to: one `setInterval` here, a
// `now` in state, and a tick that re-rendered this whole component -- every
// league card, the lobby's twenty cards and their seat pips, and (while the
// favourites picker was inline) 252 player rows with 252 photographs. Measured
// with React's Profiler in jsdom that was ~19ms of render work every second on
// an idle page, which is what "I was scrolling and the screen wasn't moving"
// is made of.
//
// The clock is a store now (lib/clock.ts) and the components that show time
// subscribe to it one by one (Countdown.tsx), so a tick re-renders a span of
// digits and nothing else. Do not reintroduce a `now` here: a page-level clock
// is a page-level re-render, whatever it is passed to.

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
  // YOUR GUYS. Three states, and they are three different panels:
  //   null   -- there is no account session to read them from (the server's
  //             own 401), or the read failed. No panel at all: this page
  //             cannot offer to save something it cannot load.
  //   []     -- signed in, nothing picked yet. The card is an invitation.
  //   [...]  -- picked. The card is a count and a way back in.
  // Read once per mount rather than on the league poll: a favourites list
  // changes when somebody changes it, which is in the dialog this page opens,
  // in front of us -- and that hands the new list straight back.
  const [favorites, setFavorites] = useState<string[] | null>(null)
  // Whether the dialog is open. The board it needs is fetched by the dialog
  // itself, the first time it is opened -- this page never loads it.
  const [picking, setPicking] = useState(false)
  // Bumped whenever something might have taught this tab a name -- see
  // `closePicker`. It is a cache-busting counter, not a piece of state
  // anything reads for its value.
  const [namesAt, setNamesAt] = useState(0)

  useEffect(() => {
    let cancelled = false
    fetchFavorites()
      .then((ids) => { if (!cancelled) setFavorites(ids) })
      .catch(() => { if (!cancelled) setFavorites(null) })
    return () => { cancelled = true }
  }, [])

  // league_id -> its stored reports, newest first. One read per non-mock
  // league, repeated whenever the league list is re-read -- about once a
  // minute, alongside Landing's own account poll -- rather than once at
  // mount: a report the room finishes building at the end of a draft, or
  // one started from another tab, should get its "Report card" link here
  // without a reload. The GET is a cheap read of one stored table, so
  // paying it once a minute per league costs nothing worth guarding.
  // Undefined until the first read lands, so the card shows no report link
  // rather than a wrong one.
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
  // Memoized, all four: they are the props of a memoized list (LeagueCards),
  // and an array rebuilt on every render is a prop that changes on every
  // render. `leagues` gets a new identity about once a minute, when the page
  // above re-reads the account, and that is exactly how often these should be
  // rebuilt.
  const real = useMemo(() => leagues.filter((l) => !isMock(l)), [leagues])
  const mocks = useMemo(() => leagues.filter(isMock), [leagues])
  // Live first, then soonest: one list, because a reader has two or three
  // of these and the boundary between "now" and "Sunday" is the live dot.
  const live = useMemo(() => real.filter((l) => l.live), [real])
  const upcoming = useMemo(
    () => real.filter((l) => !l.live).sort(byUrgency), [real])

  // Stable identities, because YourGuys is memoized and a fresh closure per
  // render would make that memo a comment.
  const openPicker = useCallback(() => setPicking(true), [])
  // Closing bumps a counter the card reads. Saving already changes the ids,
  // so the card redraws on its own; closing WITHOUT saving changes nothing it
  // can see -- and yet the picker has just fetched the board, so the names it
  // could not print a moment ago are now in the tab (lib/playerNames.ts).
  // Without this nudge a reader who opens the picker and presses Escape is
  // left looking at "7 players saved" over a list the page now knows.
  const closePicker = useCallback(() => {
    setPicking(false)
    setNamesAt((n) => n + 1)
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
        <LeagueCards
          real={real}
          live={live}
          upcoming={upcoming}
          reports={reports}
          joining={joining}
          buildingFor={buildingFor}
          onJoin={join}
          onBuild={build}
        />

        {/* YOUR GUYS, between the leagues and the lobby. Above the mock
            rooms deliberately: this is a thing to do once, before a draft,
            and a reader who has not done it should meet it before he meets
            twenty rooms he could join instead.

            A card and a button, never the list itself -- see YourGuys.tsx for
            why the names are not here. Absent entirely when there is no
            account session to read favourites from (`favorites === null`):
            this page cannot offer to save something it cannot load. */}
        {favorites !== null && (
          <YourGuys players={favorites} onOpen={openPicker} refresh={namesAt} />
        )}

        <MockLobby
          onOpen={onOpenRoom}
          seated={mocks}
          onEnter={join}
          joining={joining}
          onCount={setRoomCount}
        />
      </div>

      {/* Mounted only while it is open, so the fetch, the 252 rows and the
          photographs exist only while somebody is looking at them.

          Both the waiting state and the failed one are somebody else's
          problem by design -- see PickerBoundary.tsx. The short version:
          the chunk can 404 after a deploy, and an uncaught error inside
          `Suspense` blanks everything above it, so a 5 KB file would take
          the dashboard with it. Both stand-ins take the same two exits as
          the dialog itself. */}
      {picking && favorites !== null && (
        <PickerBoundary onClose={closePicker}>
          <Suspense fallback={<PickerFallback onClose={closePicker} />}>
            <FavoritesModal
              initial={favorites}
              onSaved={(ids) => { setFavorites(ids); setPicking(false) }}
              onClose={closePicker}
            />
          </Suspense>
        </PickerBoundary>
      )}
    </main>
  )
}
