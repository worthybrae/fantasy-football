import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  fetchFavorites, mintDraftToken,
  type TokenConnectParams, type UpcomingDraft,
} from '../api'
import DraftPlanRounds from './DraftPlanRounds'
import FounderBadge from './FounderBadge'
import { Logo } from './Logo'
import MockLobby from './MockLobby'
import MyGuysTable from './MyGuysTable'
import NextDraftHero from './NextDraftHero'
import OtherDrafts from './OtherDrafts'
import { PickerBoundary, PickerFallback } from './PickerBoundary'
import ReadinessChecklist, { markDryRun, readDryRun } from './ReadinessChecklist'
import { MAX_TEAMS, MIN_TEAMS, readSeat, writeSeat, type Seat } from '../lib/seat'

// THE PICKER IS NOT IN THIS PAGE'S BUNDLE. It is a 250-row board with a photo
// per row, opened by a fraction of the readers who load this page and never
// during a draft -- so it arrives when somebody asks for it, not before.
const FavoritesModal = lazy(() => import('./FavoritesModal'))

// THE PAGE A SIGNED-IN READER GETS, IN THE ORDER SOMEBODY ASKS IT.
//
//   when is my draft, and how do I get in     -- NextDraftHero
//   what is left to do before it               -- ReadinessChecklist
//   what do I take, round by round             -- DraftPlanRounds
//   which of my guys can I actually have       -- MyGuysTable
//   what else could I be drafting in           -- OtherDrafts, MockLobby
//
// It is not the pitch with a card bolted on top: somebody who has connected
// an account has already been sold, and what they came back for is draft
// night. Everything that was about the MODEL rather than about Thursday --
// the consensus figures, the edge tables, the counts of recorded drafts, the
// eight-column availability grid -- is gone from this page. The corpus is
// still what every number is counted in; it says so where a number is
// explained, not in a headline.
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

// How many turns the plan covers here. Eight rounds is a draft's plan; three
// is its opening, which is what the front door shows a stranger.
const PLAN_TURNS = 8

// THE PAGE NO LONGER HOLDS A CLOCK. It used to: one `setInterval` here, a
// `now` in state, and a tick that re-rendered this whole component. Measured
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

/** THE SECTION WITH NO PLAN IN IT, for a league of a shape neither the
 *  planner nor the archive answers about -- a twenty-team keeper dynasty,
 *  say.
 *
 *  It keeps the heading, because the reader's page should not lose a section
 *  because of what their league is, and it says the one true thing there is
 *  to say. What it must never do is print the endpoint's 422: "slot 12 is not
 *  in a 10-team league" is a sentence about a request nobody made. */
function PlanUnavailable({ teams }: { teams: number }) {
  return (
    <section className="db-sec">
      <div className="db-sec-head">
        <h2 className="db-sec-title">Your draft plan</h2>
      </div>
      <p className="db-sec-say">
        A {teams}-team draft is not a shape this plan covers yet — it reads
        leagues of {MIN_TEAMS} to {MAX_TEAMS} teams. The draft room itself is
        unaffected.
      </p>
    </section>
  )
}

/** THE NEXT DRAFT, AND NOTHING ELSE. The one drafting now if there is one,
 *  otherwise the soonest with a date. A league whose commissioner has set no
 *  date is never "next": there is nothing to be next before. */
function nextDraft(live: UpcomingDraft[], upcoming: UpcomingDraft[]) {
  return live[0] ?? upcoming.find((l) => l.draft_at !== null) ?? null
}

export default function Dashboard({ leagues, onJoin, onOpenRoom, onConnect }: {
  leagues: UpcomingDraft[]
  onJoin: (params: TokenConnectParams) => void
  /** Opens a mock room's waiting room: seats, countdown, who is in. */
  onOpenRoom: (leagueId: string) => void
  /** Opens the bookmarklet walkthrough, for a reader with no draft to be
   *  ready for and for the checklist's first row. */
  onConnect: () => void
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

  // MY GUYS. Three states, and they are three different panels:
  //   null   -- there is no account session to read them from (the server's
  //             own 401), or the read failed. No panel at all: this page
  //             cannot offer to save something it cannot load.
  //   []     -- signed in, nothing picked yet. The card is an invitation.
  //   [...]  -- picked. The table is the answer.
  // Read once per mount rather than on the league poll: a saved list changes
  // when somebody changes it, which is in the dialog this page opens, in
  // front of us -- and that hands the new list straight back.
  const [favorites, setFavorites] = useState<string[] | null>(null)
  // Whether the dialog is open. The board it needs is fetched by the dialog
  // itself, the first time it is opened -- this page never loads it.
  const [picking, setPicking] = useState(false)
  // Whether this reader has been in a room before it counts. See
  // ReadinessChecklist: an account holding a mock seat is the server's
  // answer, and this browser's own memory is the other half.
  const [dryRun, setDryRun] = useState(readDryRun)
  // THE SEAT THIS BROWSER REMEMBERS, which is the LAST of the three answers
  // to "which seat" and not the first -- see `planSeat`.
  const [seat, setSeat] = useState<Seat>(readSeat)
  // THE SEAT THIS READER ASKED FOR, as opposed to the one they were given.
  // Null until somebody actually moves the control in this session, which is
  // what keeps a seat remembered from last August from quietly outranking
  // the order ESPN has published since.
  const [chosen, setChosen] = useState<number | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchFavorites()
      .then((ids) => { if (!cancelled) setFavorites(ids) })
      .catch(() => { if (!cancelled) setFavorites(null) })
    return () => { cancelled = true }
  }, [])

  // TWO KINDS OF ROW: LEAGUES, THEN MOCKS. A mock room you have taken a seat
  // in arrives in the same ESPN list as your real leagues, but it is a
  // different kind of thing -- free, disposable, one of a hundred like it --
  // and it belongs with the lobby it came from. Mock is read off the name
  // because ESPN's payload carries no flag (its lobby leagues are all named
  // "... Mock"); a real league that happens to have mock in its name lands in
  // the mock row, nothing worse.
  const real = useMemo(() => leagues.filter((l) => !isMock(l)), [leagues])
  const mocks = useMemo(() => leagues.filter(isMock), [leagues])
  const live = useMemo(() => real.filter((l) => l.live), [real])
  const upcoming = useMemo(
    () => real.filter((l) => !l.live).sort(byUrgency), [real])

  // THE ONE DRAFT THIS PAGE IS ABOUT. Everything above the fold answers two
  // questions -- when, and how do I get in -- and both of them are about this
  // league. The rest of the account is a list further down.
  const next = useMemo(() => nextDraft(live, upcoming), [live, upcoming])
  const others = useMemo(
    () => upcoming.filter((l) => l !== next), [upcoming, next])
  const liveOthers = useMemo(
    () => live.filter((l) => l !== next), [live, next])
  // The plan's shape, from the draft when the account states one. A league
  // that has not said how many teams it holds leaves the reader's own setting
  // alone rather than guessing at ten.
  const planTeams = next?.teams ?? null

  // ESPN'S OWN ANSWER ABOUT THE SEAT, when it has one. `my_slot` is this
  // account's place in the league's published `draftSettings.pickOrder`
  // (api/drafts.py), so it is the seat rather than a guess at one -- and it
  // outranks anything this browser remembers. Null before the commissioner
  // sets an order, which is the ordinary state in August.
  //
  // Checked against the league's own size as well: an order that disagreed
  // with the team count is two ESPN answers contradicting each other, and
  // the seat is the one to drop.
  const espnSlot = useMemo<number | null>(() => {
    const mine = next?.my_slot ?? null
    if (mine === null || !Number.isInteger(mine) || mine < 1) return null
    if (planTeams !== null && mine > planTeams) return null
    return mine
  }, [next, planTeams])

  // THE SEAT EVERY CARD ON THIS PAGE IS DRAWN FOR, assembled in the one place
  // that holds every half of it.
  //
  // THE SIZE IS THE LEAGUE'S AND THE SLOT IS WHOEVER ANSWERED LAST, and those
  // are answers to different questions: a reader who last set the twelfth
  // seat, whose next draft holds ten teams, would ask
  // `/api/plan/preview?teams=10&slot=12`. That is a 422, and because nothing
  // on this page asks again, it is a 422 sitting in the section for as long
  // as the page is open.
  //
  // Null is "not a shape we can ask about": ESPN will not host more than
  // sixteen teams and neither endpoint answers about one.
  const planSeat = useMemo<Seat | null>(() => {
    const teams = planTeams ?? seat.teams
    if (!Number.isInteger(teams) || teams < MIN_TEAMS || teams > MAX_TEAMS) {
      return null
    }
    // The order of precedence, and it is the whole fix: what the reader just
    // asked for, then what ESPN says, then what this browser remembers.
    const slot = chosen ?? espnSlot ?? seat.slot
    return { teams, slot: Math.min(Math.max(1, Math.round(slot)), teams) }
  }, [planTeams, seat, espnSlot, chosen])

  // The seat the page is currently drawn for, readable by the callback below
  // without making it depend on the value -- it is the prop of a memoized
  // child, and a handler rebuilt on every render is a memo written in vain.
  const shown = useRef<Seat | null>(null)
  shown.current = planSeat

  const pickSeat = useCallback((slot: number) => {
    setChosen(slot)
    setSeat((s) => {
      const picked = { teams: shown.current?.teams ?? s.teams, slot }
      writeSeat(picked)
      return picked
    })
  }, [])

  // Stable identities, because the children are memoized and a fresh closure
  // per render would make that memo a comment.
  const openPicker = useCallback(() => setPicking(true), [])
  const closePicker = useCallback(() => setPicking(false), [])

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

  // Entering a mock room IS the dry run, so the row ticks when somebody takes
  // one -- not when they promise to.
  const enterMock = useCallback(async (league: UpcomingDraft) => {
    markDryRun()
    setDryRun(true)
    await join(league)
  }, [join])
  const openMockRoom = useCallback((leagueId: string) => {
    markDryRun()
    setDryRun(true)
    onOpenRoom(leagueId)
  }, [onOpenRoom])
  // The checklist's own button: no room in mind, so it scrolls to the lobby
  // and lets the reader choose one. The tick waits for the room.
  const toLobby = useCallback(() => {
    document.getElementById('mock-lobby')?.scrollIntoView({ block: 'start' })
  }, [])

  return (
    <main className="db">
      {/* THE ROOM'S OWN BAR, not a second one that resembles it. Same
          `.draft-topbar` classes, same 44px height, same flush-left
          uppercase wordmark -- so walking from this page into a draft does
          not change the furniture, it only changes what is under it. */}
      <header className="draft-topbar">
        <span className="draft-topbar-title"><Logo /> ESPN Draft Assist</span>
        {/* Only ever drawn for somebody who holds one of the first hundred
            accounts; every other reader gets nothing here. */}
        <FounderBadge variant="member" />
        <span className="draft-topbar-sep" aria-hidden="true" />
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
          a row somebody clicked. Full width above everything rather than
          inside a section, because it is about the click. */}
      {error !== null && <p className="db-error">{error}</p>}

      <NextDraftHero
        next={next}
        teams={planSeat?.teams ?? null}
        slot={planSeat?.slot ?? null}
        espnSlot={espnSlot}
        joining={joining}
        onJoin={join}
        onPickSeat={pickSeat}
        onConnect={onConnect}
      />

      <div className="db-secs">
        {/* Absent entirely when there is no account session to read the saved
            list from: a checklist cannot ask somebody to pick players it
            cannot count. */}
        {favorites !== null && (
          <ReadinessChecklist
            connected
            guys={favorites.length}
            dryRun={dryRun || mocks.length > 0}
            onConnect={onConnect}
            onPickGuys={openPicker}
            onDryRun={toLobby}
          />
        )}

        {/* YOUR DRAFT PLAN. The room's own planner, run against an empty
            board for this seat -- so the best thing this product does is
            readable on the Sunday before the draft rather than only during
            it. */}
        {planSeat === null ? (
          <PlanUnavailable teams={planTeams ?? seat.teams} />
        ) : (
          <DraftPlanRounds
            teams={planSeat.teams}
            slot={planSeat.slot}
            turns={PLAN_TURNS}
            tag={(favorites ?? []).join(',')}
            note={next !== null
              ? `For ${next.name ?? 'your next draft'} · seat ${planSeat.slot} of ${planSeat.teams}`
              : `Seat ${planSeat.slot} of ${planSeat.teams}`}
          />
        )}


        {favorites !== null && planSeat !== null && (
          <MyGuysTable players={favorites} teams={planSeat.teams}
                       slot={planSeat.slot} onEdit={openPicker} />
        )}

        <OtherDrafts live={liveOthers} upcoming={others} joining={joining}
                     onJoin={join} />

        <div id="mock-lobby">
          <MockLobby
            onOpen={openMockRoom}
            seated={mocks}
            onEnter={enterMock}
            joining={joining}
            onCount={setRoomCount}
          />
        </div>
      </div>

      {/* Mounted only while it is open, so the fetch, the 252 rows and the
          photographs exist only while somebody is looking at them.

          Both the waiting state and the failed one are somebody else's
          problem by design -- see PickerBoundary.tsx. The short version:
          the chunk can 404 after a deploy, and an uncaught error inside
          `Suspense` blanks everything above it, so a 5 KB file would take
          the dashboard with it. */}
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
