import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  buildLeagueReport, fetchFavorites, fetchLeagueReports, mintDraftToken,
  type ReportSummary, type TokenConnectParams, type UpcomingDraft,
} from '../api'
import { Countdown } from './Countdown'
import FounderBadge from './FounderBadge'
import LeagueCards from './LeagueCards'
import { Logo } from './Logo'
import MockLobby from './MockLobby'
import PlanPreview from './PlanPreview'
import { PickerBoundary, PickerFallback } from './PickerBoundary'
import YourGuys from './YourGuys'
import YourGuysByPick, { MAX_TEAMS, MIN_TEAMS, readSeat, type Seat }
  from './YourGuysByPick'

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

/** THE NEXT DRAFT, AND NOTHING ELSE. The one drafting now if there is one,
 *  otherwise the soonest with a date. A league whose commissioner has set no
 *  date is never "next": there is nothing to be next before. */
function nextDraft(live: UpcomingDraft[], upcoming: UpcomingDraft[]) {
  return live[0] ?? upcoming.find((l) => l.draft_at !== null) ?? null
}

/** THE PLAN'S SECTION WITH NO PLAN IN IT, for a league of a shape neither the
 *  planner nor the archive answers about -- a twenty-team keeper dynasty, say.
 *
 *  It keeps the section, because the reader's page should not lose a heading
 *  because of what their league is, and it says the one true thing there is
 *  to say. What it must never do is print the endpoint's 422: "slot 12 is not
 *  in a 10-team league" is a sentence about a request nobody made. */
function PlanUnavailable({ teams }: { teams: number }) {
  return (
    <section className="db-sec pp">
      <div className="db-sec-head">
        <h2 className="db-sec-title">Your plan</h2>
      </div>
      <div className="db-card pp-card">
        <p className="pp-note">
          A {teams}-team draft is not a shape this plan covers yet — it reads
          leagues of {MIN_TEAMS} to {MAX_TEAMS} teams. The draft room itself is
          unaffected.
        </p>
      </div>
    </section>
  )
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
  // WHICH SEAT THE PLAN IS FOR. One value, reported up by the availability
  // grid, so the plan and the grid cannot describe two different drafts. The
  // grid owns the controls (it is where a reader is already looking when they
  // think about their seat); this page owns the answer they both read.
  //
  // Seeded from the same store the grid seeds from, so the plan is requested
  // once with the right seat rather than once with a default and again a beat
  // later. The league SIZE is overridden below when the account actually names
  // one.
  const [seat, setSeat] = useState<Seat>(readSeat)

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

  // THE ONE DRAFT THIS PAGE IS ABOUT. Everything above the fold answers two
  // questions -- when, and how do I get in -- and both of them are about this
  // league. The rest of the account is a list further down.
  const next = useMemo(() => nextDraft(live, upcoming), [live, upcoming])
  // The plan's shape, from the draft when the account states one. A league
  // that has not said how many teams it holds leaves the reader's own setting
  // alone rather than guessing at ten.
  const planTeams = next?.teams ?? null

  // THE SEAT THE PLAN IS ACTUALLY ASKED ABOUT, assembled in the one place
  // that holds both halves of it.
  //
  // The size comes from the league and the slot comes from the reader's own
  // stored setting, and those are answers to two different questions: a
  // reader who last set the twelfth seat, whose next draft holds ten teams,
  // asks `/api/plan/preview?teams=10&slot=12`. That is a 422, and because
  // nothing on this page ever asks again, it is a 422 sitting in the card for
  // as long as the page is open. The grid clamps its own copy the same way,
  // but the grid is only mounted when there are favourites -- so the clamp
  // cannot live there.
  //
  // Null is "not a shape we can ask about": ESPN will not host more than
  // sixteen teams and neither endpoint answers about one. The card says that
  // in its own words below rather than printing the server's refusal at a
  // reader who did not ask the question.
  const planSeat = useMemo<Seat | null>(() => {
    const teams = planTeams ?? seat.teams
    if (!Number.isInteger(teams) || teams < MIN_TEAMS || teams > MAX_TEAMS) {
      return null
    }
    return { teams, slot: Math.min(Math.max(1, Math.round(seat.slot)), teams) }
  }, [planTeams, seat])

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
        {/* Only ever drawn for somebody who holds one of the first hundred
            accounts; every other reader gets nothing here. See
            FounderBadge.tsx, which is a placeholder until the founders branch
            lands its own. */}
        <FounderBadge variant="member" />
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

      {/* THE PAGE IN THE ORDER SOMEBODY READS IT.
          
          when is my next draft, and how do I get in
          who do I want, and can I have them
          what should I take, round by round
          everything else I could be drafting in
          twenty mock rooms, if none of the above

          It used to open with the whole league list, which is a filing
          cabinet: a reader with three leagues had to find the one that
          matters among the ones that do not, and the two things that would
          actually help them before Thursday were below the fold under it. */}

      {/* THE CLOCK, ALONE. The soonest draft on the account at display size,
          with the way in beside it. No card frame around it: this is not one
          item in a list, it is the answer to the question the reader opened
          the page with, and a border would make it one of several. */}
      {next !== null && (
        <section className="db-next" aria-labelledby="db-next-h">
          <p className="lp-cap" id="db-next-h">
            {next.live ? 'Drafting now' : 'Your next draft'}
          </p>
          <p className="db-next-clock mono">
            {next.live
              ? <><span className="db-dot" aria-hidden="true" />live</>
              : <Countdown at={next.draft_at} />}
          </p>
          <p className="db-next-name">
            <Link className="db-league-link"
                  to={`/league/${encodeURIComponent(next.league_id)}`}>
              {next.name ?? `League ${next.league_id}`}
            </Link>
            {next.team_name && (
              <span className="db-next-team"> · {next.team_name}</span>
            )}
            {next.teams !== null && (
              <span className="db-next-shape mono"> · {next.teams} teams</span>
            )}
          </p>
          {next.team_id !== null && (
            <button type="button" className="db-join db-next-join"
                    disabled={joining !== null}
                    onClick={() => join(next)}>
              {joining === next.league_id ? 'Opening…' : 'Open the board'}
            </button>
          )}
        </section>
      )}

      <div className="db-secs">
        {/* YOUR GUYS: who you want, and whether you can have them, under one
            heading. They were two sections stacked, which read as two
            unrelated cards -- and the second one only appeared once the first
            had a list, so the connection between them was never on screen at
            the moment it would have explained itself.

            Absent entirely when there is no account session to read
            favourites from (`favorites === null`): this page cannot offer to
            save something it cannot load. */}
        {favorites !== null && (
          <section className="db-sec">
            <div className="db-sec-head">
              <h2 className="db-sec-title">Your guys</h2>
              <span className="db-sec-note">
                Starred in the draft room, and the plan reaches for them a
                round early.
              </span>
            </div>
            <div className={`db-guys${favorites.length === 0 ? ' is-empty' : ''}`}>
              <YourGuys players={favorites} onOpen={openPicker}
                        refresh={namesAt} />
              {/* The grid is an answer ABOUT a list, so with no list saved it
                  would be an empty table asking a question nobody had. The
                  invitation beside it is the whole section then. */}
              {favorites.length > 0 && (
                <YourGuysByPick players={favorites} teams={planTeams}
                                onSeat={setSeat} />
              )}
            </div>
          </section>
        )}

        {/* YOUR PLAN. The room's own planner, run against an empty board for
            this seat -- so the best thing this product does is readable on
            the Sunday before the draft rather than only during it. */}
        {planSeat === null ? (
          <PlanUnavailable teams={planTeams ?? seat.teams} />
        ) : (
          <PlanPreview
            teams={planSeat.teams}
            slot={planSeat.slot}
            tag={(favorites ?? []).join(',')}
            // WHERE THE SEAT CAME FROM, said out loud. A plan for the wrong
            // seat is a plan for somebody else's draft, and the reader is the
            // only one who can catch that.
            note={next !== null && planTeams !== null
              ? `For ${next.name ?? 'your next draft'} — ${planTeams} teams, seat ${planSeat.slot}`
              : (favorites !== null && favorites.length > 0
                ? `Seat ${planSeat.slot} of ${planSeat.teams} — set it under Your guys`
                : `Seat ${planSeat.slot} of ${planSeat.teams}`)}
          />
        )}

        {/* EVERYTHING ELSE ON THE ACCOUNT. The next draft is above; this is
            the rest, soonest first, with its report cards.

            One card shape per row: a clock, what the draft is, and the way in.
            A league that is drafting now leads the row with the live dot where
            its clock would be. */}
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
