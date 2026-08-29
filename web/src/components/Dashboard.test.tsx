import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { Player, UpcomingDraft } from '../api'
import Dashboard from './Dashboard'
import { forgetBoard } from '../lib/board'
import { forgetNames, rememberNames } from '../lib/playerNames'

// WHAT THIS FILE IS FOR. Three properties.
//
// The first is the whole reason this page was rebuilt: the dashboard must not
// fetch the board. It is 250 rows with a photograph each, it was fetched on
// every visit for a panel almost nobody opened, and it is now the dialog's own
// business. The second is that the three states of "Your guys" are three
// different cards.
//
// The third is the ORDER. The page answers four questions and they have to
// arrive in the order somebody asks them -- when is my draft, who do I want,
// what should I take, what else could I be drafting in -- because the page
// used to open with the whole league list, which is a filing cabinet rather
// than an answer.

const { fetchFavorites, fetchPlayers, fetchLeagueReports, fetchMockRooms,
        fetchRoomProgress, fetchFavoritesOutlook, fetchPlanPreview } = vi.hoisted(() => ({
  fetchFavorites: vi.fn(),
  fetchPlayers: vi.fn(),
  fetchLeagueReports: vi.fn(),
  fetchMockRooms: vi.fn(),
  fetchRoomProgress: vi.fn(),
  fetchFavoritesOutlook: vi.fn(),
  fetchPlanPreview: vi.fn(),
}))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchFavorites,
  fetchPlayers,
  fetchLeagueReports,
  fetchMockRooms,
  fetchRoomProgress,
  fetchFavoritesOutlook,
  fetchPlanPreview,
}))

function player(i: number): Player {
  const id = `p${i}`
  return {
    player_id: id, name: `Player ${i}`, position: 'RB', team: 'DET', bye: 9,
    market_rank: i, market_spread: null,
    market_sources: { ffc: null, espn: null, fp: null, mfl: null, cbs: null, fp_tier: null },
    espn_ppr_rank: i, stats: null, career_games_pg: null, consistency_cv: null,
    consistency_pct: null, season_finishes: null, proj_change: null,
    game_points: null, headshot: null, rookie: false, drafted: false,
    avail_pct: null, ev: null, ev_se: null, rank: i, tier: 1, edge: null,
    proj_points: 100,
  }
}
const BOARD = Array.from({ length: 30 }, (_, i) => player(i + 1))

const PLAN = {
  teams: 10, slot: 5, picks: [5, 16, 25, 36],
  opening: [{ path: ['RB', 'WR', 'WR', 'RB', 'TE'], count: 265, share: 0.31 }],
  opening_rounds: 5, opening_observed: 854,
  position_runs: { QB: 41, TE: 33, K: null, DST: null },
  corpus: { teams: 8, rounds: 16, format: 'ppr', drafts: 854 },
  targets: [{
    pick_no: 5, round: 1,
    target: {
      player_id: 'plan-1', name: 'A Plan Target', position: 'RB', team: 'DET',
      headshot: null, lasts_pct: 88.2, edge_pts: 12.1, edge_at_pick: 16,
      favourite: false, pros: ['a reason'], cons: [],
    },
    alternates: [],
  }],
}

/** A league with a draft an hour out, joinable. */
const SOON: UpcomingDraft = {
  league_id: 'L1', name: 'Sunday Money', team_id: '7', team_name: 'My Team',
  season: 2026, teams: 12, draft_type: 'Snake', live: false,
  draft_at: new Date(Date.now() + 3_600_000).toISOString(),
  // ESPN has published no draft order for it, which is the ordinary state
  // in August. The seat-aware tests at the bottom of this file hand it one.
  my_slot: null,
}

beforeEach(() => {
  // Call counts, not implementations: several tests here assert that the
  // board was never asked for, and one test earlier in the file opens the
  // dialog, which asks for it.
  vi.clearAllMocks()
  // The board cache is module-level and outlives a test's mocks, which is the
  // point of it in a browser and a trap in a file like this one. So is the
  // stored seat, which this page and the availability grid both write.
  forgetBoard()
  window.localStorage.clear()
  fetchLeagueReports.mockResolvedValue([])
  fetchMockRooms.mockResolvedValue({ rooms: [], next: null })
  fetchRoomProgress.mockResolvedValue([])
  fetchPlayers.mockImplementation(async () => { rememberNames(BOARD); return BOARD })
  fetchFavoritesOutlook.mockResolvedValue(
    { teams: 10, slot: 5, picks: [5, 16], players: [] })
  fetchPlanPreview.mockResolvedValue(PLAN)
  // FounderBadge reads /api/account/me directly; absent until the founders
  // branch lands its own route.
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
    { ok: false, status: 404, json: async () => ({}) }))
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  // Both caches are module-level and one of them lives in sessionStorage:
  // without this, one test's board names the next test's ids.
  forgetBoard()
  forgetNames()
})

function draw(leagues: UpcomingDraft[] = []) {
  return render(
    <MemoryRouter>
      <Dashboard leagues={leagues} onJoin={() => {}} onOpenRoom={() => {}} />
    </MemoryRouter>,
  )
}

test('no session for favourites means no card at all', async () => {
  fetchFavorites.mockResolvedValue(null)
  draw()
  await waitFor(() => expect(fetchFavorites).toHaveBeenCalled())
  expect(screen.queryByText('Your guys')).toBeNull()
  expect(fetchPlayers).not.toHaveBeenCalled()
})

test('nothing picked yet is an invitation, and it costs no board', async () => {
  fetchFavorites.mockResolvedValue([])
  draw()
  expect(await screen.findByRole('button', { name: 'Pick your guys' })).toBeTruthy()
  expect(screen.getByText('Your guys')).toBeTruthy()
  // THE POINT OF THE WHOLE TASK: nothing has asked for /api/players.
  expect(fetchPlayers).not.toHaveBeenCalled()
})

test('a saved list is a count and a way back in, still with no board', async () => {
  fetchFavorites.mockResolvedValue(BOARD.slice(0, 7).map((p) => p.player_id))
  draw()
  expect(await screen.findByRole('button', { name: 'Edit' })).toBeTruthy()
  expect(screen.getByText('7')).toBeTruthy()
  expect(fetchPlayers).not.toHaveBeenCalled()
})

test('the board is fetched when the picker is opened, and not before', async () => {
  fetchFavorites.mockResolvedValue([])
  draw()
  const open = await screen.findByRole('button', { name: 'Pick your guys' })
  expect(fetchPlayers).not.toHaveBeenCalled()

  fireEvent.click(open)
  // Lazily loaded, so the dialog arrives a tick later.
  const dialog = await screen.findByRole('dialog')
  expect(dialog.getAttribute('aria-modal')).toBe('true')
  await waitFor(() => expect(fetchPlayers).toHaveBeenCalledTimes(1))
  expect(await screen.findByText('Player 1')).toBeTruthy()
})

// THE CARD IS A LIST OF PEOPLE, NOT OF IDENTIFIERS. The owner asked to see
// the players they picked; the ids are storage. This page never fetches the
// board -- that is the point of the dialog -- so the names come from whatever
// the last board fetch left behind in this tab, which in practice is the
// first time somebody opened the picker.
test('the saved list is named from the tab\'s own cache, with no board fetch', async () => {
  rememberNames([player(1), player(2)])
  fetchFavorites.mockResolvedValue(['p1', 'p2'])
  draw()
  expect(await screen.findByText('Player 1')).toBeTruthy()
  expect(screen.getByText('Player 2')).toBeTruthy()
  expect(fetchPlayers).not.toHaveBeenCalled()
})

test('with nobody named, it says how many rather than printing ids', async () => {
  fetchFavorites.mockResolvedValue(['zz1', 'zz2', 'zz3'])
  draw()
  expect(await screen.findByText('3 players saved.')).toBeTruthy()
  expect(screen.queryByText('zz1')).toBeNull()
  expect(fetchPlayers).not.toHaveBeenCalled()
})

// OPENING THE PICKER IS WHAT TEACHES THIS TAB THE NAMES -- it is the only
// thing on this page that fetches the board. So a reader who opens it and
// changes nothing must not be left looking at a count over a list the page
// now knows how to print.
test('names appear after the picker has been opened and dismissed', async () => {
  fetchFavorites.mockResolvedValue(['p1', 'p2'])
  draw()
  // Nothing in this tab has seen a board yet.
  expect(await screen.findByText('2 players saved.')).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: 'Edit' }))
  await screen.findByRole('dialog')
  await waitFor(() => expect(fetchPlayers).toHaveBeenCalled())

  fireEvent.keyDown(document, { key: 'Escape' })
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())

  expect(await screen.findByText('Player 1')).toBeTruthy()
  expect(screen.getByText('Player 2')).toBeTruthy()
  expect(screen.queryByText('2 players saved.')).toBeNull()
})


// -- the order the page answers in --------------------------------------------

/** The page's headings, top to bottom. The h2s are the sections; the next
 *  draft leads with an eyebrow rather than a heading because it is the answer
 *  and not a category. */
function sections(): string[] {
  return Array.from(document.querySelectorAll('h2'))
    .map((h) => (h.textContent ?? '').trim())
}

test('the next draft is the first thing on the page, with the way in',
     async () => {
       fetchFavorites.mockResolvedValue([])
       draw([SOON])

       expect(await screen.findByText('Your next draft')).toBeTruthy()
       const clock = document.querySelector('.db-next-clock')
       expect(clock?.textContent).toMatch(/^\d+:\d\d$/)
       expect(screen.getByRole('button', { name: 'Open the board' })).toBeTruthy()
       // And it is above everything else the page has to say.
       const next = document.querySelector('.db-next')
       const first = document.querySelector('.db-secs')
       expect(next!.compareDocumentPosition(first!)
         & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
     })

test('a league with no date set is never "next"', async () => {
  fetchFavorites.mockResolvedValue([])
  draw([{ ...SOON, draft_at: null }])

  await waitFor(() => expect(fetchFavorites).toHaveBeenCalled())
  expect(screen.queryByText('Your next draft')).toBeNull()
  // The league is still on the page, in the list where the account lives.
  expect(screen.getByText('Sunday Money')).toBeTruthy()
})

test('the sections run: your guys, your plan, leagues, mocks', async () => {
  fetchFavorites.mockResolvedValue(BOARD.slice(0, 7).map((p) => p.player_id))
  draw([SOON])

  await screen.findByRole('button', { name: 'Edit' })
  expect(sections()).toEqual(['Your guys', 'Your plan', 'Leagues', 'Mock drafts'])
})

test('with no favourites the plan is still there, and the leagues after it',
     async () => {
       fetchFavorites.mockResolvedValue([])
       draw([SOON])

       await screen.findByRole('button', { name: 'Pick your guys' })
       expect(sections()).toEqual(['Your guys', 'Your plan', 'Leagues', 'Mock drafts'])
     })

test('no account session drops "Your guys" and keeps the rest', async () => {
  fetchFavorites.mockResolvedValue(null)
  draw([SOON])

  await waitFor(() => expect(fetchFavorites).toHaveBeenCalled())
  expect(sections()).toEqual(['Your plan', 'Leagues', 'Mock drafts'])
})

// -- your guys, both halves under one heading ---------------------------------

test('nothing saved is the invitation alone, with no grid beside it',
     async () => {
       fetchFavorites.mockResolvedValue([])
       draw()

       await screen.findByRole('button', { name: 'Pick your guys' })
       expect(document.querySelector('.db-guys.is-empty')).toBeTruthy()
       expect(fetchFavoritesOutlook).not.toHaveBeenCalled()
     })

test('a saved list puts the names and the grid side by side', async () => {
  fetchFavorites.mockResolvedValue(BOARD.slice(0, 7).map((p) => p.player_id))
  draw()

  await screen.findByRole('button', { name: 'Edit' })
  expect(document.querySelectorAll('.db-guys .db-guys-col')).toHaveLength(2)
  expect(screen.getByRole('heading', { level: 3, name: 'In your order' })).toBeTruthy()
  expect(screen.getByRole('heading', { level: 3, name: 'By pick' })).toBeTruthy()
})

// -- the plan's seat ----------------------------------------------------------

test('the plan takes the league size from the draft that is coming', async () => {
  fetchFavorites.mockResolvedValue([])
  draw([SOON])

  await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalled())
  // Twelve, from the league, rather than whatever the reader last set.
  expect(fetchPlanPreview.mock.calls.at(-1)?.[0]).toBe(12)
  expect(screen.getByText(/For Sunday Money — 12 teams/)).toBeTruthy()
})

test('with no draft to read a size off, the plan uses the saved seat',
     async () => {
       window.localStorage.setItem('guys-outlook',
                                   JSON.stringify({ teams: 14, slot: 3 }))
       fetchFavorites.mockResolvedValue([])
       draw()

       await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalled())
       expect(fetchPlanPreview.mock.calls.at(-1)?.slice(0, 2)).toEqual([14, 3])
       expect(screen.getByText(/Seat 3 of 14/)).toBeTruthy()
       window.localStorage.clear()
     })

test('a saved seat the next draft does not have is pulled inside it',
     async () => {
       // The seat is the reader's own setting and the size is the league's,
       // and unpaired they ask about seat 12 of a ten-team draft -- a 422
       // that would sit in the card for as long as the page stayed open.
       // Nothing else clamps it here: with no favourites the grid, which
       // clamps its own copy, is not on the page at all.
       window.localStorage.setItem('guys-outlook',
                                   JSON.stringify({ teams: 12, slot: 12 }))
       fetchFavorites.mockResolvedValue([])
       draw([{ ...SOON, teams: 10 }])

       await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalled())
       expect(fetchPlanPreview.mock.calls.at(-1)?.slice(0, 2)).toEqual([10, 10])
       expect(screen.getByText(/For Sunday Money — 10 teams/)).toBeTruthy()
       expect(seatPicker().value).toBe('10')
     })

test('a league of a shape the plan cannot read says so in its own words',
     async () => {
       fetchFavorites.mockResolvedValue([])
       draw([{ ...SOON, teams: 24 }])

       await waitFor(() => expect(fetchFavorites).toHaveBeenCalled())
       // The section stays -- a reader does not lose a heading because of
       // what their league is -- and the endpoint is never asked, so there
       // is no refusal to print.
       expect(sections()).toContain('Your plan')
       expect(screen.getByText(
         /A 24-team draft is not a shape this plan covers yet/)).toBeTruthy()
       expect(fetchPlanPreview).not.toHaveBeenCalled()
     })


// -- whose seat the plan is for -----------------------------------------------
//
// THE BUG. The owner sits sixth in an eight-team league and the dashboard was
// planning for the fifth seat -- a number this page had defaulted to once and
// then remembered forever -- under a heading naming their real league. Every
// pick number in that plan was a turn out, and nothing on the page said where
// the seat had come from, so there was nothing for the reader to disagree
// with. ESPN publishes the order; when it has, that is the answer.

function seatPicker(): HTMLSelectElement {
  return screen.getByLabelText('Seat to preview') as HTMLSelectElement
}

test('ESPN’s own seat beats the one this browser remembers', async () => {
  window.localStorage.setItem('guys-outlook',
                              JSON.stringify({ teams: 12, slot: 3 }))
  fetchFavorites.mockResolvedValue([])
  draw([{ ...SOON, my_slot: 6 }])

  await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalled())
  expect(fetchPlanPreview.mock.calls.at(-1)?.slice(0, 2)).toEqual([12, 6])
  // And it says so, because a seat is the one thing in this section only the
  // reader can check.
  expect(screen.getByText('seat 6 · from ESPN')).toBeTruthy()
  expect(seatPicker().value).toBe('6')
})

test('no published order is said plainly, with the seat left to the reader',
     async () => {
       fetchFavorites.mockResolvedValue([])
       draw([SOON])

       expect(await screen.findByText(
         /ESPN hasn’t set the draft order yet — choose a seat to preview/))
         .toBeTruthy()
       expect(screen.queryByText(/from ESPN/)).toBeNull()
     })

test('a reader may preview another seat, and is told whose it is not',
     async () => {
       fetchFavorites.mockResolvedValue([])
       draw([{ ...SOON, my_slot: 6 }])
       await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalled())

       fireEvent.change(seatPicker(), { target: { value: '3' } })

       await waitFor(() => expect(fetchPlanPreview.mock.calls.at(-1)?.slice(0, 2))
         .toEqual([12, 3]))
       expect(screen.getByText('previewing seat 3 (ESPN has you at 6)'))
         .toBeTruthy()
       expect(screen.queryByText('seat 6 · from ESPN')).toBeNull()
     })

test('the availability grid is asked about the same seat as the plan',
     async () => {
       // Two cards about "your next draft" that disagreed about which seat it
       // was would be worse than one, and the grid is the page's other reader
       // of the stored seat.
       window.localStorage.setItem('guys-outlook',
                                   JSON.stringify({ teams: 12, slot: 3 }))
       fetchFavorites.mockResolvedValue(['p1', 'p2'])
       draw([{ ...SOON, my_slot: 6 }])

       await waitFor(() => expect(fetchFavoritesOutlook).toHaveBeenCalled())
       expect(fetchFavoritesOutlook.mock.calls.at(-1)?.slice(0, 2))
         .toEqual([12, 6])

       fireEvent.change(seatPicker(), { target: { value: '9' } })

       await waitFor(() => expect(fetchFavoritesOutlook.mock.calls.at(-1)
         ?.slice(0, 2)).toEqual([12, 9]))
       expect(fetchPlanPreview.mock.calls.at(-1)?.slice(0, 2)).toEqual([12, 9])
     })
