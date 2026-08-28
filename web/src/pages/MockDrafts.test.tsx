import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { BoardPlayer, LiveBoard, MockDraft } from '../api'
import MockDrafts from './MockDrafts'

// WHAT THIS FILE IS FOR. /api/mocks carries the board of its own first row,
// and the first row is what this page opens. The saving is only real if the
// page stops asking the board endpoint for that room -- inlining a board the
// page then re-fetches five seconds later would have doubled the work on the
// server to save one request on the very first paint. So the property held
// here is a call count, not a rendering.

const { fetchMockDrafts, fetchMockBoard } = vi.hoisted(() => ({
  fetchMockDrafts: vi.fn(),
  fetchMockBoard: vi.fn(),
}))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchMockDrafts,
  fetchMockBoard,
}))

const POLL_MS = 5000

function player(name: string): BoardPlayer {
  return {
    player_id: name, name, position: 'WR', team: 'DET', bye: 9,
    overall_rank: 1, tier: 1, market_rank: 1, espn_ppr_rank: 1,
    vor: 1, last_ppg: 1, last_points: 1, proj_ppg: 1, value: 0,
  }
}

function board(name: string): LiveBoard {
  return {
    active: true, teams: 2, rounds: 1, my_slot: 1, on_the_clock: null,
    picks_made: 1,
    columns: [
      { slot: 1, team_name: 'One', is_me: true },
      { slot: 2, team_name: 'Two', is_me: false },
    ],
    cells: [{ overall: 1, round: 1, slot: 1, player: player(name) }],
  }
}

const live: MockDraft = {
  id: 'd1', league_id: '1', status: 'live', teams: 2, rounds: 1,
  picks_made: 1, human_seats: 1, my_slot: 1,
  started_at: '2026-08-28T00:00:00Z', recorded_at: null,
}

beforeEach(() => {
  // `shouldAdvanceTime` so testing-library's own waiting still works while
  // the page's five-second poll is under our control.
  vi.useFakeTimers({ shouldAdvanceTime: true })
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

test('the board comes with the listing, poll after poll, and is never fetched', async () => {
  // A different board every time the listing is asked, so what is on screen
  // says WHICH listing response drew it.
  let tick = 0
  fetchMockDrafts.mockImplementation(async () => ({
    drafts: [live], first_board: board(`Pick ${++tick}`),
  }))

  render(<MemoryRouter><MockDrafts /></MemoryRouter>)

  // First paint: a board, drawn without a request of its own.
  expect(await screen.findByText(/^Pick \d+$/)).toBeTruthy()
  expect(fetchMockBoard).not.toHaveBeenCalled()

  // And every window after it, which is the half that makes the inlining pay
  // for itself rather than cost. Several windows and not one: the board poll
  // this replaced ran on a timer of its own, started at a different moment,
  // so a single window can elapse before it would have come round -- and a
  // test that allowed for only one would go on passing after somebody put
  // that second timer back.
  await act(async () => { await vi.advanceTimersByTimeAsync(POLL_MS) })
  await act(async () => { await vi.advanceTimersByTimeAsync(POLL_MS * 3) })

  // The listing really is still polling. Otherwise "nothing fetched the
  // board" would also be true of a page that had quietly stopped.
  expect(tick).toBeGreaterThan(3)
  // The NEWEST listing's board is the one on screen -- not the one that
  // happened to arrive before the page had chosen a room.
  expect(screen.getByText(`Pick ${tick}`)).toBeTruthy()
  expect(fetchMockBoard).not.toHaveBeenCalled()
})


test('a listing with no board of its own falls back to fetching one', async () => {
  // The server builds `first_board` best effort. When it could not, the page
  // has to ask for the board itself or the panel would sit empty forever
  // waiting for a listing to bring one.
  fetchMockDrafts.mockResolvedValue({ drafts: [live], first_board: null })
  fetchMockBoard.mockResolvedValue(board('Fetched Pick'))

  render(<MemoryRouter><MockDrafts /></MemoryRouter>)

  expect(await screen.findByText('Fetched Pick')).toBeTruthy()
  expect(fetchMockBoard).toHaveBeenCalledWith('d1')
})
