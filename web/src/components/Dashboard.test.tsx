import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { Player } from '../api'
import Dashboard from './Dashboard'
import { forgetBoard } from '../lib/board'
import { forgetNames, rememberNames } from '../lib/playerNames'

// WHAT THIS FILE IS FOR. Two properties, and the first is the whole reason
// this page was rebuilt: the dashboard must not fetch the board. It is 250
// rows with a photograph each, it was fetched on every visit for a panel
// almost nobody opened, and it is now the dialog's own business. The second is
// that the three states of "Your guys" are three different cards.

const { fetchFavorites, fetchPlayers, fetchLeagueReports, fetchMockRooms,
        fetchRoomProgress } = vi.hoisted(() => ({
  fetchFavorites: vi.fn(),
  fetchPlayers: vi.fn(),
  fetchLeagueReports: vi.fn(),
  fetchMockRooms: vi.fn(),
  fetchRoomProgress: vi.fn(),
}))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchFavorites,
  fetchPlayers,
  fetchLeagueReports,
  fetchMockRooms,
  fetchRoomProgress,
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

beforeEach(() => {
  // Call counts, not implementations: several tests here assert that the
  // board was never asked for, and one test earlier in the file opens the
  // dialog, which asks for it.
  vi.clearAllMocks()
  // The board cache is module-level and outlives a test's mocks, which is the
  // point of it in a browser and a trap in a file like this one.
  forgetBoard()
  fetchLeagueReports.mockResolvedValue([])
  fetchMockRooms.mockResolvedValue({ rooms: [], next: null })
  fetchRoomProgress.mockResolvedValue([])
  fetchPlayers.mockImplementation(async () => { rememberNames(BOARD); return BOARD })
})

afterEach(() => {
  cleanup()
  // Both caches are module-level and one of them lives in sessionStorage:
  // without this, one test's board names the next test's ids.
  forgetBoard()
  forgetNames()
})

function draw() {
  return render(
    <MemoryRouter>
      <Dashboard leagues={[]} onJoin={() => {}} onOpenRoom={() => {}} />
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
