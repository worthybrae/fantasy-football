import { useState } from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { Player } from '../api'
import FavoritesModal from './FavoritesModal'
import { forgetBoard } from '../lib/board'

// WHAT THIS FILE IS FOR. The dialog is the only way into the favourites list
// now, so the three things a dialog owes a reader are properties of the
// product rather than polish: it closes on Escape, it closes on the backdrop,
// and it holds the page still while it is open. The 5-25 rule is tested
// through it as well, because that is where a reader meets it.

const { fetchPlayers, saveFavorites } = vi.hoisted(() => ({
  fetchPlayers: vi.fn(),
  saveFavorites: vi.fn(),
}))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchPlayers,
  saveFavorites,
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
  forgetBoard()
  fetchPlayers.mockResolvedValue(BOARD)
  saveFavorites.mockImplementation(async (ids: string[]) => ids)
})

afterEach(cleanup)

async function open(initial: string[] = [], handlers: {
  onSaved?: (ids: string[]) => void; onClose?: () => void
} = {}) {
  const view = render(
    <FavoritesModal
      initial={initial}
      onSaved={handlers.onSaved ?? (() => {})}
      onClose={handlers.onClose ?? (() => {})}
    />,
  )
  // The search box exists only once the board has landed, and unlike a
  // player's name it cannot appear twice (a chosen player is drawn both as a
  // chip and as a board row).
  await screen.findByLabelText('Search players')
  return view
}

test('the board is fetched once, however many times it is opened', async () => {
  const first = await open()
  expect(fetchPlayers).toHaveBeenCalledTimes(1)
  first.unmount()
  await open()
  expect(fetchPlayers).toHaveBeenCalledTimes(1)
})

test('five is the floor and Save sends the list', async () => {
  const onSaved = vi.fn()
  await open([], { onSaved })
  const save = () => screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement
  expect(screen.getByText('0 / 5-25')).toBeTruthy()
  expect(save().disabled).toBe(true)

  for (let i = 1; i <= 4; i += 1) {
    fireEvent.click(screen.getByRole('button', { name: `Player ${i}` }))
  }
  expect(screen.getByText('4 / 5-25')).toBeTruthy()
  expect(save().disabled).toBe(true)

  fireEvent.click(screen.getByRole('button', { name: 'Player 5' }))
  expect(screen.getByText('5 / 5-25')).toBeTruthy()
  expect(save().disabled).toBe(false)

  fireEvent.click(save())
  await waitFor(() => expect(saveFavorites).toHaveBeenCalledWith(
    ['p1', 'p2', 'p3', 'p4', 'p5']))
  expect(onSaved).toHaveBeenCalledWith(['p1', 'p2', 'p3', 'p4', 'p5'])
})

test('twenty-five is the ceiling', async () => {
  await open(BOARD.slice(0, 25).map((p) => p.player_id))
  expect(screen.getByText('25 / 5-25')).toBeTruthy()
  const extra = screen.getByRole('button', { name: 'Player 26' }) as HTMLButtonElement
  expect(extra.disabled).toBe(true)
})

test('Escape closes it', async () => {
  const onClose = vi.fn()
  await open([], { onClose })
  fireEvent.keyDown(document, { key: 'Escape' })
  expect(onClose).toHaveBeenCalledTimes(1)
})

test('the backdrop closes it, and the panel does not', async () => {
  const onClose = vi.fn()
  const { container } = await open([], { onClose })
  fireEvent.mouseDown(screen.getByRole('dialog'))
  expect(onClose).not.toHaveBeenCalled()
  fireEvent.mouseDown(container.querySelector('.fav-modal-backdrop') as HTMLElement)
  expect(onClose).toHaveBeenCalledTimes(1)
})

test('the page behind it does not scroll, and gets its scrolling back', async () => {
  const view = await open()
  expect(document.body.style.overflow).toBe('hidden')
  view.unmount()
  expect(document.body.style.overflow).toBe('')
})

test('focus starts inside, and Tab cannot leave', async () => {
  const { container } = await open(BOARD.slice(0, 5).map((p) => p.player_id))
  const dialog = screen.getByRole('dialog')
  expect(dialog.contains(document.activeElement)).toBe(true)
  const stops = Array.from(container.querySelectorAll<HTMLElement>(
    'button:not([disabled]), input:not([disabled])'))
  const last = stops[stops.length - 1]
  last.focus()
  fireEvent.keyDown(document, { key: 'Tab' })
  expect(document.activeElement).toBe(stops[0])
})

// FOCUS GOES BACK WHERE IT CAME FROM. A dialog that takes focus and returns
// it to `document.body` leaves the next Tab at the top of the page, and a
// reader who opened this from the card's own Edit button has to walk the
// whole dashboard to find that button again.
function Opener() {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>Edit</button>
      {open && (
        <FavoritesModal initial={[]} onSaved={() => {}} onClose={() => setOpen(false)} />
      )}
    </>
  )
}

test('closing hands focus back to whatever opened it', async () => {
  render(<Opener />)
  const edit = screen.getByRole('button', { name: 'Edit' })
  edit.focus()
  expect(document.activeElement).toBe(edit)

  fireEvent.click(edit)
  await screen.findByLabelText('Search players')
  // Inside while it is open...
  expect(screen.getByRole('dialog').contains(document.activeElement)).toBe(true)

  fireEvent.keyDown(document, { key: 'Escape' })
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  // ...and back on the button afterwards, not on the body.
  expect(document.activeElement).toBe(edit)
})
