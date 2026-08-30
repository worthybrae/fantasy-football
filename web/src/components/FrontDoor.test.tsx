import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { MarketOverview, PlanPreview } from '../api'
import { FounderLine, Hero, TryItNow, WhatYouGet } from './FrontDoor'
import { forgetRequests } from '../api'

// WHAT THIS FILE IS FOR. The front door's job is to be believed, and what
// makes it believable is that everything on it is a reading off this
// deployment rather than a number somebody typed. So these tests are about
// provenance and about the one flow that matters:
//
//   * the hero says what the tool does, counts the drafts it read, and offers
//     two doors -- and both of them go to the same setup, which the page says
//     out loud rather than letting a reader discover after clicking;
//   * TRY IT NOW is the whole pitch: two selects, no login, and the real
//     planner answers for the seat the reader picked. Changing either select
//     asks again, for that seat;
//   * every "what you get" line carries a reading, and falls back to a
//     weaker sentence that is still true when the endpoint is down.

const { fetchMarketOverview, fetchPlanPreview } = vi.hoisted(() => ({
  fetchMarketOverview: vi.fn(),
  fetchPlanPreview: vi.fn(),
}))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchMarketOverview,
  fetchPlanPreview,
}))

const OVERVIEW: MarketOverview = {
  drafts: 854, picks: 109312, human_picks: 71204,
  teams: 8, rounds: 16, from: '2026-07-01', to: '2026-08-28',
  median_humans: 5,
}

function player(name: string, position: string, extra = {}) {
  return {
    player_id: name.toLowerCase().replace(/\W/g, ''), name, position,
    team: 'DET', headshot: null, lasts_pct: null, edge_pts: null,
    edge_at_pick: null, favourite: false, pros: [], cons: [],
    reason: `${name} is the pick.`, ...extra,
  }
}

function planFor(teams: number, slot: number): PlanPreview {
  const first = slot
  const second = teams * 2 - slot + 1
  return {
    teams, slot, turns: 4, picks: [first, second, first + teams * 2,
                                   second + teams * 2],
    opening: [], opening_rounds: 5, opening_observed: 0,
    position_runs: {}, corpus: null,
    targets: [
      { pick_no: first, round: 1, target: player('Bijan Robinson', 'RB'),
        alternates: [player('Ja’Marr Chase', 'WR')],
        backups: ['Ja’Marr Chase'] },
      { pick_no: second, round: 2,
        target: player('Puka Nacua', 'WR', { lasts_pct: 38.4 }),
        alternates: [], backups: [] },
    ],
  }
}

beforeEach(() => {
  forgetRequests()
  vi.clearAllMocks()
  fetchMarketOverview.mockResolvedValue(OVERVIEW)
  fetchPlanPreview.mockImplementation(
    async (teams: number, slot: number) => planFor(teams, slot))
  // FounderBadge asks for /api/account/me directly. Absent on a deployment
  // that predates the founders work, which is the case these tests run in.
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
    { ok: false, status: 404, json: async () => ({}) }))
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

// -- the hero -----------------------------------------------------------------

test('the page opens with one sentence saying what the tool is', () => {
  render(<Hero onStart={() => {}} />)

  expect(screen.getByRole('heading', { level: 1 }).textContent)
    .toBe('Your ESPN draft, with a plan.')
})

test('the sub-line counts the drafts it read, in the reader’s words',
     async () => {
       render(<Hero onStart={() => {}} />)

       await waitFor(() => expect(fetchMarketOverview).toHaveBeenCalled())
       const said = await screen.findByText(/real ESPN drafts/)
       expect(said.textContent).toMatch(/854/)
       expect(said.textContent).toMatch(/whether your guys will still be there/)
     })

test('an archive that cannot be read still says something true', async () => {
  fetchMarketOverview.mockRejectedValue(new Error('no corpus'))
  render(<Hero onStart={() => {}} />)

  await waitFor(() => expect(fetchMarketOverview).toHaveBeenCalled())
  expect(screen.getByText(/from real ESPN drafts, recorded every day/)).toBeTruthy()
  expect(screen.queryByText(/854/)).toBeNull()
})

test('two doors, one setup, and the page says so', () => {
  const onStart = vi.fn()
  render(<Hero onStart={onStart} />)

  fireEvent.click(screen.getByRole('button', { name: 'Try a mock draft — free' }))
  fireEvent.click(screen.getByRole('button', { name: 'Connect your ESPN league' }))
  expect(onStart).toHaveBeenCalledTimes(2)
  expect(screen.getByText(/Both start the same way/)).toBeTruthy()
  expect(screen.getByText(/No password, nothing installed/)).toBeTruthy()
})

// -- try it now ---------------------------------------------------------------

test('try it now opens on a seat and draws the real plan for it', async () => {
  render(<TryItNow />)

  await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalled())
  // Ten teams, sixth seat, four rounds, no login.
  expect(fetchPlanPreview).toHaveBeenCalledWith(10, 6, '', 4)
  expect(await screen.findByText('Bijan Robinson')).toBeTruthy()
  expect(screen.getByText('Round 1')).toBeTruthy()
  expect(screen.getByText('Bijan Robinson is the pick.')).toBeTruthy()
})

test('changing the league size asks again, for that size', async () => {
  render(<TryItNow />)

  await screen.findByText('Bijan Robinson')
  fireEvent.change(screen.getByLabelText('League size'), { target: { value: '12' } })

  await waitFor(() => expect(fetchPlanPreview.mock.calls.at(-1)?.slice(0, 2))
    .toEqual([12, 6]))
  expect(screen.getByText('Seat 6 of 12')).toBeTruthy()
})

test('changing the seat asks again, for that seat', async () => {
  render(<TryItNow />)

  await screen.findByText('Bijan Robinson')
  fireEvent.change(screen.getByLabelText('Your seat'), { target: { value: '2' } })

  await waitFor(() => expect(fetchPlanPreview.mock.calls.at(-1)?.slice(0, 2))
    .toEqual([10, 2]))
  // The seat's own picks, not the ones the page opened with.
  expect(await screen.findByText('pick 2')).toBeTruthy()
})

test('a seat that does not exist in a smaller league is pulled inside it',
     async () => {
       render(<TryItNow />)

       await screen.findByText('Bijan Robinson')
       fireEvent.change(screen.getByLabelText('Your seat'), { target: { value: '9' } })
       await waitFor(() => expect(fetchPlanPreview.mock.calls.at(-1)?.slice(0, 2))
         .toEqual([10, 9]))

       fireEvent.change(screen.getByLabelText('League size'), { target: { value: '8' } })
       await waitFor(() => expect(fetchPlanPreview.mock.calls.at(-1)?.slice(0, 2))
         .toEqual([8, 8]))
     })

// -- what you get -------------------------------------------------------------

test('three lines, each carrying a reading off this deployment', async () => {
  render(<WhatYouGet />)

  expect(await screen.findByText('Will he be there?')).toBeTruthy()
  expect(screen.getByText('38%')).toBeTruthy()
  expect(screen.getByText('One name a round, and why')).toBeTruthy()
  expect(screen.getByText('Pick 6')).toBeTruthy()
  expect(screen.getByText('My guys')).toBeTruthy()
  expect(screen.getAllByRole('listitem')).toHaveLength(3)
})

test('a plan that cannot be read leaves the lines weaker but true', async () => {
  fetchPlanPreview.mockRejectedValue(new Error('no board'))
  render(<WhatYouGet />)

  await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalled())
  expect(screen.getAllByText('—')).toHaveLength(2)
  expect(screen.getByText(/counted in real ESPN drafts/)).toBeTruthy()
})

// -- the price ----------------------------------------------------------------

test('the founder line names the offer in the words the product uses', () => {
  const onStart = vi.fn()
  render(<FounderLine onStart={onStart} />)

  expect(screen.getByText(/Free for the first 100 accounts, for good/)).toBeTruthy()
  expect(screen.getByText(/\$9\.99 a season/)).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Connect your ESPN league' }))
  expect(onStart).toHaveBeenCalled()
})
