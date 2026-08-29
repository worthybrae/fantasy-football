import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { FavoritesOutlook, OutlookPlayer } from '../api'
import YourGuysByPick from './YourGuysByPick'

// WHAT THIS FILE IS FOR. Every number in this card is the server's, so there
// is nothing here worth testing about the arithmetic. What IS this
// component's own, and what breaks silently, is the mapping: a row per saved
// favourite, a column per pick, the right tint for the right band, and a
// control change that actually asks a new question rather than re-rendering
// the old answer.

const { fetchFavoritesOutlook } = vi.hoisted(() => ({
  fetchFavoritesOutlook: vi.fn(),
}))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchFavoritesOutlook,
}))

const PICKS = [5, 16, 25, 36]

function player(name: string, avail: (number | null)[],
                best: number | null = null): OutlookPlayer {
  return {
    player_id: name.toLowerCase(), name, position: 'RB', team: 'DET',
    headshot: `https://example.test/${name}.png`,
    espn_rank: 4, espn_adp: 5.2, market_rank: 4, avail, best_pick: best,
  }
}

// One player per band, so a single render states all four thresholds.
const OUTLOOK: FavoritesOutlook = {
  teams: 10,
  slot: 5,
  picks: PICKS,
  players: [
    player('Likely', [100, 88, 70, 69.9], 25),
    player('Maybe', [40, 39.9, 12, 0]),
    player('Ghost', [null, null, null, null]),
  ],
}

const IDS = OUTLOOK.players.map((p) => p.player_id)

beforeEach(() => {
  window.localStorage.clear()
  fetchFavoritesOutlook.mockReset()
  fetchFavoritesOutlook.mockResolvedValue(OUTLOOK)
})

afterEach(cleanup)

async function draw(ids: string[] = IDS) {
  const view = render(<YourGuysByPick players={ids} />)
  await screen.findByRole('table')
  return view
}

function chip(name: string, pick: number): HTMLElement {
  const row = screen.getByRole('row', { name: new RegExp(name) })
  // +1: the first cell of the row is the player, not a pick.
  return within(row).getAllByRole('cell')[PICKS.indexOf(pick)]
    .firstElementChild as HTMLElement
}

test('a row per favourite and a column per pick', async () => {
  await draw()

  for (const pick of PICKS) {
    expect(screen.getByRole('columnheader', { name: `Pick ${pick}` })).toBeTruthy()
  }
  // The three players, in the order the server sent them -- which is the
  // saved order, and is deliberately not alphabetical here.
  const rows = screen.getAllByRole('row').slice(1)
  expect(rows.map((row) => within(row).getByRole('rowheader').textContent))
    .toEqual(['LikelyRB', 'MaybeRB', 'GhostRB'])
  // A headshot per row, decorative (the name beside it is the label).
  const faces = screen.getAllByAltText('')
  expect(faces).toHaveLength(3)
  expect(faces[0].getAttribute('src')).toBe('https://example.test/Likely.png')
})

test('the tint says which band the chance is in', async () => {
  await draw()

  // 70 is likely and 69.9 is not: the boundary is the claim.
  expect(chip('Likely', 25).className).toContain('gbp-likely')
  expect(chip('Likely', 36).className).toContain('gbp-maybe')
  // 40 is maybe and 39.9 is thin.
  expect(chip('Maybe', 5).className).toContain('gbp-maybe')
  expect(chip('Maybe', 16).className).toContain('gbp-thin')
  // A player the board cannot name is a dash, not a zero.
  expect(chip('Ghost', 5).className).toContain('gbp-none')
  expect(chip('Ghost', 5).textContent).toBe('—')
})

test('the best pick is marked, and only that one', async () => {
  await draw()

  expect(chip('Likely', 25).className).toContain('gbp-best')
  expect(chip('Likely', 16).className).not.toContain('gbp-best')
  // Nobody's best pick, so nothing on the row is marked.
  expect(chip('Maybe', 5).className).not.toContain('gbp-best')
})

test('changing a control asks the server a new question', async () => {
  await draw()
  expect(fetchFavoritesOutlook).toHaveBeenCalledWith(10, 5, IDS.join(','))

  fireEvent.change(screen.getByLabelText('League'), { target: { value: '12' } })

  await waitFor(() => {
    expect(fetchFavoritesOutlook).toHaveBeenCalledWith(12, 5, IDS.join(','))
  })

  fireEvent.change(screen.getByLabelText('Seat'), { target: { value: '11' } })

  await waitFor(() => {
    expect(fetchFavoritesOutlook).toHaveBeenCalledWith(12, 11, IDS.join(','))
  })
  // And the seat survives a remount, which is the whole point of storing it.
  expect(JSON.parse(window.localStorage.getItem('guys-outlook') as string))
    .toEqual({ teams: 12, slot: 11 })
})

test('a slot that the new league size does not have is pulled back inside it',
     async () => {
  window.localStorage.setItem('guys-outlook',
                              JSON.stringify({ teams: 14, slot: 13 }))
  await draw()
  expect(fetchFavoritesOutlook).toHaveBeenCalledWith(14, 13, IDS.join(','))

  fireEvent.change(screen.getByLabelText('League'), { target: { value: '8' } })

  await waitFor(() => {
    expect(fetchFavoritesOutlook).toHaveBeenCalledWith(8, 5, IDS.join(','))
  })
})

test('a refusal is said in the card rather than thrown at the page', async () => {
  fetchFavoritesOutlook.mockRejectedValue(new Error('Slot 11 does not exist'))

  render(<YourGuysByPick players={IDS} />)

  expect(await screen.findByText('Slot 11 does not exist')).toBeTruthy()
  expect(screen.queryByRole('table')).toBeNull()
})
