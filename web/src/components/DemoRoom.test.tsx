import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import type { LiveMock, Player } from '../api'
import DemoRoom from './DemoRoom'

// WHAT THIS FILE IS FOR. The landing page's live room is the SAME components
// the draft room runs -- that is its whole claim, and the reason it cannot
// drift from the product. Which also means it breaks whenever those
// components change and nobody looked: it renders TargetCards and
// AvailableList against a payload with no plan, no seat and no favourites,
// which is a shape the room itself never has. This holds it to rendering a
// real board, with every Draft button disabled, because nobody here holds a
// seat.

const { fetchLiveMock, fetchPlayers } = vi.hoisted(() => ({
  fetchLiveMock: vi.fn(), fetchPlayers: vi.fn(),
}))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchLiveMock,
  fetchPlayers,
}))

function player(id: string, name: string): Player {
  return {
    player_id: id, name, position: 'RB', team: 'DET', bye: 9,
    market_rank: 1, market_spread: null,
    market_sources: { ffc: null, espn: null, fp: null, mfl: null, cbs: null, fp_tier: null },
    espn_ppr_rank: 1, stats: null, career_games_pg: null, consistency_cv: null,
    consistency_pct: null, season_finishes: null, proj_change: null,
    game_points: null, headshot: null, rookie: false, drafted: false,
    avail_pct: null, ev: null, ev_se: null, rank: 1, tier: 1, edge: null,
    proj_points: 200,
  }
}

const ROOM: LiveMock = {
  live: true,
  mode: 'live',
  teams: 8,
  rounds: 16,
  picks_made: 5,
  picks_total: 128,
  round: 1,
  on_the_clock: 6,
  humans: 3,
  clock_seconds: null,
  turn_started_at: null,
  server_now: null,
  settings: {
    teams: 8, rounds: 16, starters: { QB: 1, RB: 2, WR: 2, TE: 1 },
    flex_slots: 1, bench: 7, scoring_format: 'ppr',
  },
  shortlist: [
    { player_id: 'a', position: 'RB', proj_points: 240, rank: 1, name: 'Ashton',
      team: 'DET', bye: 9, adp: 3, vs_adp: null, board_rank: 1,
      espn_rank: 1, espn_pos_rank: 1, espn_adp: 3, lasts_pct: 40,
      lasts_at_pick: 11, edge_pts: 5, edge_at_pick: 11, need: 'starter' },
    { player_id: 'b', position: 'WR', proj_points: 220, rank: 2, name: 'Bijan',
      team: 'ATL', bye: 12, adp: 5, vs_adp: null, board_rank: 2,
      espn_rank: 2, espn_pos_rank: 1, espn_adp: 5, lasts_pct: 80,
      lasts_at_pick: 11, edge_pts: -2, edge_at_pick: 11, need: 'starter' },
  ],
}

beforeEach(() => {
  fetchLiveMock.mockResolvedValue(ROOM)
  fetchPlayers.mockResolvedValue([player('a', 'Ashton'), player('b', 'Bijan')])
})

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

test('the landing room draws a real board, and nothing on it can be drafted', async () => {
  render(<MemoryRouter><DemoRoom /></MemoryRouter>)

  // The shortlist reaches the room's own list, by name.
  await waitFor(() => expect(screen.getAllByText('Ashton').length).toBeGreaterThan(0))
  expect(screen.getAllByText('Bijan').length).toBeGreaterThan(0)
  expect(screen.getByText('Live mock draft')).toBeTruthy()

  // Nobody holds a seat here, so every Draft button in the room is off --
  // the same disabled state the real room shows on somebody else's turn.
  const draft = screen.getAllByRole('button', { name: 'Draft' }) as HTMLButtonElement[]
  expect(draft.length).toBeGreaterThan(0)
  expect(draft.every((b) => b.disabled)).toBe(true)
})

test('the landing room opens on the Take-now card, says why its buttons are off, and keeps the cheat sheet behind the same toggle', async () => {
  window.localStorage.clear()
  render(<MemoryRouter><DemoRoom /></MemoryRouter>)
  await waitFor(() => expect(screen.getAllByText('Ashton').length).toBeGreaterThan(0))

  // The simple view first: one card, one list, and no fourteen-column table.
  expect(screen.getByText('Take now')).toBeTruthy()
  expect(screen.queryByText('Take one of these')).toBeNull()
  const toggle = screen.getByRole('group', { name: 'Draft room view' })
  expect(toggle).toBeTruthy()

  // Off, and honest about why: nobody here has a seat, and nothing an
  // unlock or a reconnect could do would change that.
  const draft = screen.getAllByRole('button', { name: 'Draft' }) as HTMLButtonElement[]
  expect(draft.length).toBeGreaterThan(0)
  expect(draft.every((b) => b.disabled)).toBe(true)
  expect(draft.every((b) => b.title === "Watching — this is someone else's draft")).toBe(true)

  // The cheat sheet is one click away, and the choice is the room's own
  // remembered one, so draft night opens on whatever the visitor picked.
  fireEvent.click(screen.getByRole('button', { name: 'Cheat sheet' }))
  expect(screen.getByText('Take one of these')).toBeTruthy()
  expect(screen.queryByText('Take now')).toBeNull()
  expect(window.localStorage.getItem('room-view')).toBe('cheat')
})
