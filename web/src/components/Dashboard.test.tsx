import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { UpcomingDraft } from '../api'
import Dashboard from './Dashboard'
import { forgetBoard } from '../lib/board'
import { forgetNames } from '../lib/playerNames'

// WHAT THIS FILE IS FOR. Four properties.
//
// The first is the whole reason this page was rebuilt once already: the
// dashboard must not fetch the board. It is 250 rows with a photograph each,
// and it is now the picker dialog's own business.
//
// The second is the SEAT, which is the bug this hero exists to end. The plan
// used to be built for whatever seat the reader last set in a select, so an
// owner ESPN has sitting sixth read a plan for the fifth seat under a heading
// naming their real league. ESPN's published order outranks this browser's
// memory, the page says where the number came from, and the control stays.
//
// The third is the JOIN: one button, one mint, and the params handed straight
// up -- the same path the bookmarklet's token takes.
//
// The fourth is the ORDER. The page answers five questions and they have to
// arrive in the order somebody asks them: when is my draft, what is left to
// do, what do I take, which of my guys can I have, what else is there.

const { fetchFavorites, fetchPlayers, fetchMockRooms, fetchRoomProgress,
        fetchFavoritesOutlook, fetchPlanPreview, mintDraftToken } = vi.hoisted(() => ({
  fetchFavorites: vi.fn(),
  fetchPlayers: vi.fn(),
  fetchMockRooms: vi.fn(),
  fetchRoomProgress: vi.fn(),
  fetchFavoritesOutlook: vi.fn(),
  fetchPlanPreview: vi.fn(),
  mintDraftToken: vi.fn(),
}))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchFavorites,
  fetchPlayers,
  fetchMockRooms,
  fetchRoomProgress,
  fetchFavoritesOutlook,
  fetchPlanPreview,
  mintDraftToken,
}))

const PLAN = {
  teams: 12, slot: 6, turns: 8, picks: [6, 19, 30, 43],
  opening: [], opening_rounds: 5, opening_observed: 0,
  position_runs: {}, corpus: null,
  targets: [{
    pick_no: 6, round: 1,
    target: {
      player_id: 'plan-1', name: 'A Plan Target', position: 'RB', team: 'DET',
      headshot: null, lasts_pct: 88.2, edge_pts: 12.1, edge_at_pick: 19,
      favourite: false, pros: ['a reason'], cons: [],
      reason: 'A reason, and another one.',
    },
    alternates: [],
    backups: [],
  }],
}

/** A league with a draft an hour out, joinable. */
const SOON: UpcomingDraft = {
  league_id: 'L1', name: 'Sunday Money', team_id: '7', team_name: 'My Team',
  season: 2026, teams: 12, draft_type: 'Snake', live: false,
  draft_at: new Date(Date.now() + 3_600_000).toISOString(),
  // ESPN has published no draft order for it, which is the ordinary state
  // in August. The seat-aware tests below hand it one.
  my_slot: null,
}

/** A second league, further out: the compact list under everything. */
const LATER: UpcomingDraft = {
  ...SOON, league_id: 'L2', name: 'The Other One', team_id: '8',
  draft_at: new Date(Date.now() + 9 * 3_600_000).toISOString(),
}

beforeEach(() => {
  vi.clearAllMocks()
  // The board cache is module-level and outlives a test's mocks, which is the
  // point of it in a browser and a trap in a file like this one. So is the
  // stored seat.
  forgetBoard()
  window.localStorage.clear()
  fetchMockRooms.mockResolvedValue({ rooms: [], next: null })
  fetchRoomProgress.mockResolvedValue([])
  fetchPlayers.mockResolvedValue([])
  fetchFavorites.mockResolvedValue([])
  fetchFavoritesOutlook.mockResolvedValue(
    { teams: 12, slot: 6, picks: [6, 19], players: [] })
  fetchPlanPreview.mockResolvedValue(PLAN)
  mintDraftToken.mockResolvedValue(
    { leagueId: 'L1', teamId: '7', swid: 'S', token: 'T', season: '2026' })
  // FounderBadge reads /api/account/me directly.
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
    { ok: false, status: 404, json: async () => ({}) }))
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  forgetBoard()
  forgetNames()
})

function draw(leagues: UpcomingDraft[] = [], props = {}) {
  return render(
    <MemoryRouter>
      <Dashboard leagues={leagues} onJoin={() => {}} onOpenRoom={() => {}}
                 onConnect={() => {}} {...props} />
    </MemoryRouter>,
  )
}

// -- the hero -----------------------------------------------------------------

test('the next draft is the page’s first answer, with its clock', async () => {
  draw([LATER, SOON])

  expect(await screen.findByText('Your next draft')).toBeTruthy()
  const h1 = screen.getByRole('heading', { level: 1 })
  expect(h1.textContent).toBe('Sunday Money')
})

test('ESPN’s published seat is the one the page plans for, and it says so',
     async () => {
       draw([{ ...SOON, my_slot: 6 }])

       expect(await screen.findByText('seat 6 of 12 · from ESPN')).toBeTruthy()
       await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalled())
       expect(fetchPlanPreview.mock.calls.at(-1)?.slice(0, 2)).toEqual([12, 6])
     })

test('a seat this browser remembers does not outrank ESPN’s order', async () => {
  window.localStorage.setItem('guys-outlook', JSON.stringify({ teams: 10, slot: 2 }))
  draw([{ ...SOON, my_slot: 6 }])

  expect(await screen.findByText('seat 6 of 12 · from ESPN')).toBeTruthy()
})

test('no published order says so and leaves the reader to choose', async () => {
  draw([SOON])

  expect(await screen.findByText(/ESPN hasn’t set the order yet/)).toBeTruthy()
  expect(screen.getByLabelText('Which seat to plan for')).toBeTruthy()
})

test('choosing a seat keeps saying what ESPN’s answer was', async () => {
  draw([{ ...SOON, my_slot: 6 }])

  const select = await screen.findByLabelText('Which seat to plan for')
  fireEvent.change(select, { target: { value: '9' } })

  expect(screen.getByText('seat 9 of 12 · ESPN has you at 6')).toBeTruthy()
  await waitFor(() => expect(fetchPlanPreview.mock.calls.at(-1)?.slice(0, 2))
    .toEqual([12, 9]))
})

test('an ESPN order that contradicts the league’s size is dropped', async () => {
  draw([{ ...SOON, teams: 10, my_slot: 14 }])

  expect(await screen.findByText(/ESPN hasn’t set the order yet/)).toBeTruthy()
})

test('the one button mints a token and hands it straight up', async () => {
  const onJoin = vi.fn()
  draw([SOON], { onJoin })

  fireEvent.click(await screen.findByRole('button', { name: 'Open the draft room' }))

  await waitFor(() => expect(onJoin).toHaveBeenCalled())
  expect(mintDraftToken).toHaveBeenCalledWith('L1', '7', 2026)
  expect(onJoin.mock.calls[0][0]).toMatchObject({ leagueId: 'L1', token: 'T' })
})

test('an account with no draft to be ready for is offered the walkthrough',
     async () => {
       const onConnect = vi.fn()
       draw([], { onConnect })

       fireEvent.click(
         await screen.findByRole('button', { name: 'Connect your ESPN league' }))
       expect(onConnect).toHaveBeenCalled()
     })

// -- the page's order ---------------------------------------------------------

test('the sections arrive in the order somebody asks them', async () => {
  fetchFavorites.mockResolvedValue(['p1', 'p2'])
  const { container } = draw([SOON, LATER])

  await screen.findByText('Your draft plan')
  const headings = Array.from(
    container.querySelectorAll('h1, .db-sec-title'))
    .map((node) => node.textContent?.trim())
  expect(headings).toEqual([
    'Sunday Money', 'Ready for draft night', 'Your draft plan', 'My guys',
    'Your other leagues', 'Mock drafts',
  ])
})

test('the league the hero is about is not repeated in the list below',
     async () => {
       draw([SOON, LATER])

       await screen.findByText('Your other leagues')
       expect(screen.getAllByText('Sunday Money')).toHaveLength(1)
       expect(screen.getByText('The Other One')).toBeTruthy()
     })

// -- the board ----------------------------------------------------------------

test('the page never fetches the board', async () => {
  fetchFavorites.mockResolvedValue(['p1', 'p2', 'p3'])
  draw([SOON])

  await screen.findByText('My guys')
  expect(fetchPlayers).not.toHaveBeenCalled()
})

test('no session for the saved list means no checklist and no table', async () => {
  fetchFavorites.mockResolvedValue(null)
  draw([SOON])

  await waitFor(() => expect(fetchFavorites).toHaveBeenCalled())
  expect(screen.queryByText('My guys')).toBeNull()
  expect(screen.queryByText('Ready for draft night')).toBeNull()
})

// -- the shape nothing answers about -----------------------------------------

test('a league neither endpoint reads keeps its section and says why',
     async () => {
       draw([{ ...SOON, teams: 24 }])

       expect(await screen.findByText(/not a shape this plan covers yet/))
         .toBeTruthy()
       expect(fetchPlanPreview).not.toHaveBeenCalled()
     })
