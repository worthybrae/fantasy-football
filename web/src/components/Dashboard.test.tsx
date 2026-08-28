import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, test, vi } from 'vitest'
import type { Player } from '../api'
import Dashboard from './Dashboard'

// WHAT THIS FILE IS FOR. "Your guys" has three states and they are three
// different panels -- no session, nothing picked yet, and a saved list -- and
// the one that matters is the first: a dashboard on a server without the
// favourites routes (or for a reader with no custody session) must show no
// panel at all rather than an empty picker offering to save something the
// server will refuse.

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

function draw(board: 'ok' | 'fails' = 'ok') {
  fetchLeagueReports.mockResolvedValue([])
  fetchMockRooms.mockResolvedValue({ rooms: [], next: null })
  fetchRoomProgress.mockResolvedValue([])
  if (board === 'fails') fetchPlayers.mockRejectedValue(new Error('players 503'))
  else fetchPlayers.mockResolvedValue([player(1), player(2), player(3)])
  return render(
    <MemoryRouter>
      <Dashboard leagues={[]} onJoin={() => {}} onOpenRoom={() => {}} />
    </MemoryRouter>,
  )
}

afterEach(cleanup)

test('no session for favourites means no panel at all', async () => {
  fetchFavorites.mockResolvedValue(null)
  draw()
  await waitFor(() => expect(fetchFavorites).toHaveBeenCalled())
  expect(screen.queryByText('Your guys')).toBeNull()
  expect(fetchPlayers).not.toHaveBeenCalled()
})

test('nothing picked yet opens the picker', async () => {
  fetchFavorites.mockResolvedValue([])
  draw()
  expect(await screen.findByText('Pick your guys')).toBeTruthy()
  expect(screen.getByText('0 / 5-25')).toBeTruthy()
})

test('a board that will not load says so, rather than loading forever', async () => {
  fetchFavorites.mockResolvedValue([])
  draw('fails')
  expect(await screen.findByRole('alert')).toBeTruthy()
  expect(screen.getByRole('alert').textContent).toContain('players 503')
  expect(screen.queryByText('Loading the board…')).toBeNull()
})

test('a saved list is a card, not the picker', async () => {
  fetchFavorites.mockResolvedValue(['p2', 'p1'])
  draw()
  expect(await screen.findByText('Player 2')).toBeTruthy()
  expect(screen.queryByText('Pick your guys')).toBeNull()
  expect(screen.getByRole('button', { name: 'Edit' })).toBeTruthy()
})
