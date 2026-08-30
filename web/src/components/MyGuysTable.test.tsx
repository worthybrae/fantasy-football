import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { FavoritesOutlook } from '../api'
import MyGuysTable, { tintFor } from './MyGuysTable'

// WHAT THIS FILE IS FOR. The saved list says who somebody wants; this table
// says whether they can have them, and when. Three properties:
//
//   * two columns, in the reader's own words -- when to take him, and will he
//     be there. It was an eight-column heat map, which is a table to study;
//   * a favourite the plan never reaches for still gets an answer. "Later
//     than pick 75" is a sentence; a dash is our bookkeeping;
//   * nothing saved is an invitation with a button, not an empty table under
//     a heading -- and it costs no request at all.

const { fetchFavoritesOutlook } = vi.hoisted(
  () => ({ fetchFavoritesOutlook: vi.fn() }))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchFavoritesOutlook,
}))

const IDS = ['p1', 'p2', 'p3']

function row(id: string, name: string, extra = {}) {
  return {
    player_id: id, name, position: 'WR', team: 'DET', headshot: null,
    espn_rank: 4, espn_adp: 5.2, market_rank: 4,
    avail: [88.4, 41.2, 12.0], best_pick: 15, plan_round: null, ...extra,
  }
}

const OUTLOOK: FavoritesOutlook = {
  teams: 10, slot: 6, picks: [6, 15, 75],
  players: [
    row('p1', 'Planned Man', { plan_round: 2 }),
    row('p2', 'Unplanned Man', { avail: [39.4, 8.0, 1.0] }),
    row('p3', 'Unknown Man', { name: null, avail: [null, null, null],
                               best_pick: null }),
  ],
}

beforeEach(() => {
  vi.clearAllMocks()
  fetchFavoritesOutlook.mockResolvedValue(OUTLOOK)
})

afterEach(cleanup)

test('the columns are the two questions, and the pick is named', async () => {
  render(<MyGuysTable players={IDS} teams={10} slot={6} onEdit={() => {}} />)

  await screen.findByText('Planned Man')
  expect(screen.getByRole('columnheader', { name: 'Player' })).toBeTruthy()
  expect(screen.getByRole('columnheader', { name: 'When to take him' })).toBeTruthy()
  // The chance is only true ABOUT a pick, so the header says which one.
  const will = screen.getAllByRole('columnheader').at(-1)
  expect(will?.textContent).toBe('Will he be there? at pick 6')
})

test('a favourite the plan wants is named by round', async () => {
  render(<MyGuysTable players={IDS} teams={10} slot={6} onEdit={() => {}} />)

  expect(await screen.findByText('Round 2')).toBeTruthy()
})

test('a favourite the plan never reaches for is a sentence, not a dash',
     async () => {
       render(<MyGuysTable players={IDS} teams={10} slot={6} onEdit={() => {}} />)

       await screen.findByText('Unplanned Man')
       // The last turn the plan covers is 75, so "after all of them" is the
       // honest answer and it is said in picks.
       expect(screen.getAllByText('Later than pick 75').length).toBe(2)
     })

test('the chance is the one at the reader’s next pick, rounded once', async () => {
  render(<MyGuysTable players={IDS} teams={10} slot={6} onEdit={() => {}} />)

  await screen.findByText('Planned Man')
  expect(screen.getByText('88%')).toBeTruthy()
  expect(screen.getByText('39%')).toBeTruthy()
  // A player the board can no longer name carries a dash rather than a zero:
  // "gone" and "we don't know him" are different answers.
  expect(screen.getByText('—')).toBeTruthy()
})

test('the tint is banded off the number the chip prints', () => {
  expect(tintFor(70)).toBe('gbp-likely')
  expect(tintFor(69)).toBe('gbp-maybe')
  expect(tintFor(40)).toBe('gbp-maybe')
  expect(tintFor(39)).toBe('gbp-thin')
  expect(tintFor(null)).toBe('gbp-none')
})

test('nothing saved is an invitation, and it costs no request', async () => {
  const onEdit = vi.fn()
  render(<MyGuysTable players={[]} teams={10} slot={6} onEdit={onEdit} />)

  const go = screen.getByRole('button', { name: 'Pick my guys' })
  expect(screen.getByText(/Star five to twenty-five players/)).toBeTruthy()
  expect(screen.queryByRole('table')).toBeNull()
  expect(fetchFavoritesOutlook).not.toHaveBeenCalled()

  fireEvent.click(go)
  expect(onEdit).toHaveBeenCalled()
})

test('the seat it is asked about is the page’s, not one of its own', async () => {
  render(<MyGuysTable players={IDS} teams={12} slot={9} onEdit={() => {}} />)

  await waitFor(() => expect(fetchFavoritesOutlook).toHaveBeenCalled())
  expect(fetchFavoritesOutlook).toHaveBeenCalledWith(12, 9, 'p1,p2,p3')
  // No controls of its own: two answers about "your next draft" that
  // disagreed about which seat it was would be worse than one.
  expect(screen.queryByRole('combobox')).toBeNull()
})
