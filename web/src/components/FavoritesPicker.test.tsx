import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import type { Player } from '../api'
import FavoritesPicker from './FavoritesPicker'

// WHAT THIS FILE IS FOR. The server answers 422 for a list outside 5-25
// (api/account.py), and the whole point of this screen is that nobody ever
// meets that refusal: the count is on screen and Save is not a live control
// until the list is one the server would take. The property held here is
// therefore the BUTTON'S STATE against the count, at both ends of the range.

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

function save(): HTMLButtonElement {
  return screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement
}

function pick(name: string): void {
  fireEvent.click(screen.getByRole('button', { name }))
}

afterEach(cleanup)

test('four is not enough, five is', () => {
  render(<FavoritesPicker players={BOARD} onSaved={() => {}} />)
  expect(screen.getByText('0 / 5-25')).toBeTruthy()
  expect(save().disabled).toBe(true)

  for (let i = 1; i <= 4; i += 1) pick(`Player ${i}`)
  expect(screen.getByText('4 / 5-25')).toBeTruthy()
  expect(save().disabled).toBe(true)

  pick('Player 5')
  expect(screen.getByText('5 / 5-25')).toBeTruthy()
  expect(save().disabled).toBe(false)
})

test('twenty-five is the ceiling, and the board stops offering more', () => {
  const twentyFive = BOARD.slice(0, 25).map((p) => p.player_id)
  render(<FavoritesPicker players={BOARD} initial={twentyFive} onSaved={() => {}} />)
  expect(screen.getByText('25 / 5-25')).toBeTruthy()
  expect(save().disabled).toBe(false)
  // A twenty-sixth cannot be added: the row is disabled rather than
  // clickable-and-refused.
  const extra = screen.getByRole('button', { name: 'Player 26' }) as HTMLButtonElement
  expect(extra.disabled).toBe(true)
  fireEvent.click(extra)
  expect(screen.getByText('25 / 5-25')).toBeTruthy()
})

test('a saved list already over the ceiling cannot be saved again', () => {
  const twentySix = BOARD.slice(0, 26).map((p) => p.player_id)
  render(<FavoritesPicker players={BOARD} initial={twentySix} onSaved={() => {}} />)
  expect(screen.getByText('26 / 5-25')).toBeTruthy()
  expect(save().disabled).toBe(true)
})

test('a chosen player can be dropped again from the list at the top', () => {
  render(<FavoritesPicker players={BOARD} initial={['p1', 'p2', 'p3', 'p4', 'p5']}
                          onSaved={() => {}} />)
  expect(save().disabled).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: 'Remove Player 1' }))
  expect(screen.getByText('4 / 5-25')).toBeTruthy()
  expect(save().disabled).toBe(true)
})

// A FRESH SCREEN. The chosen list, the search and the pill are all state
// taken on mount, so a rerender with different props is the same screen with
// the same filters -- these tests want a new one.
function open(props: { players?: Player[]; initial?: string[] } = {}): void {
  cleanup()
  render(<FavoritesPicker players={props.players ?? BOARD} initial={props.initial}
                          onSaved={() => {}} />)
}

// THE COUNTER IS A SENTENCE, not a ratio. "3 / 5-25" left the arithmetic to
// the reader; these two strings do it for them, and they are the only thing
// standing between somebody and a Save button that refuses to work. Both
// ends of the range, and the gap in between where the ratio is the whole
// story and neither sentence should appear.
test('the counter says how many more, or that the list is full', () => {
  open()
  expect(screen.getByText('5 more to save')).toBeTruthy()

  open({ initial: ['p1', 'p2', 'p3'] })
  expect(screen.getByText('2 more to save')).toBeTruthy()

  // Enough to save and room for more: the ratio is the whole story.
  open({ initial: ['p1', 'p2', 'p3', 'p4', 'p5'] })
  expect(screen.queryByText(/more to save/)).toBeNull()
  expect(screen.queryByText(/^Full/)).toBeNull()

  open({ initial: BOARD.slice(0, 25).map((p) => p.player_id) })
  expect(screen.getByText('Full — drop one to add another')).toBeTruthy()

  // And it keeps up with the list rather than being read off the props:
  // dropping one turns the sentence back into a count.
  fireEvent.click(screen.getByRole('button', { name: 'Remove Player 1' }))
  expect(screen.queryByText(/^Full/)).toBeNull()
})

// "No players match that" was true of the search, true of the pills, and
// true of both at once, so it never said which one to undo. Three strings
// now, one per way the board can empty.
test('the empty board names the filter that emptied it', () => {
  open()
  fireEvent.change(screen.getByLabelText('Search players'), { target: { value: 'zzz' } })
  expect(screen.getByText('Nobody on the board matches \u201Czzz\u201D.')).toBeTruthy()

  // Both filters on: the search is named, and so is the pill narrowing it.
  fireEvent.click(screen.getByRole('button', { name: 'QB' }))
  expect(screen.getByText('Nobody on the board matches \u201Czzz\u201D at QB.')).toBeTruthy()

  // The pill on its own -- every fixture player is an RB.
  fireEvent.change(screen.getByLabelText('Search players'), { target: { value: '' } })
  expect(screen.getByText('No QB on the board.')).toBeTruthy()

  // And nothing to filter in the first place.
  open({ players: [] })
  expect(screen.getByText('The board is empty.')).toBeTruthy()
})
