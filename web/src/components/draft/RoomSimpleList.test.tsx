import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { LiveCandidate, Player } from '../../api'
import RoomSimpleList from './RoomSimpleList'

// WHAT THIS FILE IS FOR. Four facts on a row, and every one of them can lie
// in the same way: by drawing a number nobody computed. A null "will he be
// there?" is not 0% and a null "worth grabbing now" is not "+0 pts" -- both
// mean there is no next turn to measure to, and both have to read as a dash
// with no tint on it. The rest is the filters, which are the only way a
// reader gets to a player who is not in the top thirty rows.

const { fetchMarketOverview } = vi.hoisted(() => ({ fetchMarketOverview: vi.fn() }))

vi.mock('../../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api')>()),
  fetchMarketOverview,
}))

function candidate(over: Partial<LiveCandidate> & { player_id: string }): LiveCandidate {
  return {
    position: 'RB',
    proj_points: 200,
    espn_rank: 1,
    espn_pos_rank: 1,
    espn_adp: 1,
    market_rank: 1,
    lasts_pct: 50,
    lasts_at_pick: 20,
    edge_pts: 6,
    edge_at_pick: 20,
    need: 'starter',
    favourite: false,
    rank: 1,
    ...over,
  }
}

function player(id: string, name: string, team = 'DET'): Player {
  return {
    player_id: id, name, position: 'RB', team, bye: 9,
    market_rank: 1, market_spread: null,
    market_sources: { ffc: null, espn: null, fp: null, mfl: null, cbs: null, fp_tier: null },
    espn_ppr_rank: 1, stats: null, career_games_pg: null, consistency_cv: null,
    consistency_pct: null, season_finishes: null, proj_change: null,
    game_points: null, headshot: null, rookie: false, drafted: false,
    avail_pct: null, ev: null, ev_se: null, rank: 1, tier: 1, edge: null,
    proj_points: 200,
  }
}

// Handed over out of order on purpose: the list's order is the server's own
// (`rank`, ESPN's order), not the order the array happened to arrive in.
const CANDIDATES = [
  candidate({ player_id: 'c', rank: 3, position: 'TE', lasts_pct: 30, edge_pts: -4 }),
  candidate({ player_id: 'a', rank: 1, lasts_pct: 85, edge_pts: 12, favourite: true }),
  candidate({ player_id: 'b', rank: 2, position: 'WR', lasts_pct: 55, edge_pts: 0 }),
  candidate({ player_id: 'd', rank: 4, position: 'WR', lasts_pct: null,
              lasts_at_pick: null, edge_pts: null, edge_at_pick: null }),
]

const PLAYERS: Record<string, Player> = {
  a: player('a', 'Ashton'), b: player('b', 'Bijan', 'ATL'),
  c: player('c', 'Chase'), d: player('d', 'Dell'),
}

function draw(over: Partial<Parameters<typeof RoomSimpleList>[0]> = {}) {
  return render(
    <RoomSimpleList
      candidates={CANDIDATES}
      players={PLAYERS}
      onDraft={() => {}}
      isMyTurn={false}
      onOpenPlayer={() => {}}
      draftedIds={new Set<string>()}
      {...over}
    />,
  )
}

function names(container: HTMLElement): (string | null)[] {
  return Array.from(container.querySelectorAll('tbody .rsl-name-btn'))
    .map((el) => el.textContent)
}

beforeEach(() => {
  // The mock is module-level and outlives each test; without this, "did
  // anything fetch?" is answered by an earlier test's hover.
  fetchMarketOverview.mockClear()
  fetchMarketOverview.mockResolvedValue({ drafts: 854, picks: 109_000 })
})
afterEach(cleanup)

test('opens in ESPN order, whatever order the payload arrived in', () => {
  const { container } = draw()
  expect(names(container)).toEqual(['Ashton', 'Bijan', 'Chase', 'Dell'])
})

test('the chance he lasts is tinted in three bands, and a dash is not tinted', () => {
  const { container } = draw()
  const chips = Array.from(container.querySelectorAll('tbody .lasts-chip'))
    .map((chip) => [chip.className, chip.textContent])
  expect(chips).toEqual([
    [expect.stringContaining('is-high'), '85%'],
    [expect.stringContaining('is-mid'), '55%'],
    [expect.stringContaining('is-low'), '30%'],
    // No band at all: there is no next turn to survive to.
    [expect.stringContaining('is-none'), '—'],
  ])
})

test('worth grabbing now is muted at or below zero, and a dash when unpriced', () => {
  const { container } = draw()
  const cells = Array.from(container.querySelectorAll('tbody .rsl-col-edge'))
    .map((cell) => [cell.textContent, cell.className.includes('is-flat')])
  expect(cells).toEqual([
    ['+12 pts', false],
    // Zero is not a reason to take him now, so it neither reads as one nor
    // wears a plus sign -- the same rounding rule the cheat sheet's Edge
    // column uses.
    ['0 pts', true],
    ['-4 pts', true],
    ['—', true],
  ])
})

test('stars the reader\'s own guys, and only them', () => {
  const { container } = draw()
  const stars = screen.getAllByLabelText('One of your guys')
  expect(stars).toHaveLength(1)
  expect(container.querySelector('tbody tr[data-pid="a"]')?.contains(stars[0])).toBe(true)
})

test('search and the position chips still filter', () => {
  const { container } = draw()
  fireEvent.change(screen.getByLabelText('Search available players'),
                   { target: { value: 'bij' } })
  expect(names(container)).toEqual(['Bijan'])

  // A team matches too, the same haystack the cheat sheet searches.
  fireEvent.change(screen.getByLabelText('Search available players'),
                   { target: { value: 'atl' } })
  expect(names(container)).toEqual(['Bijan'])

  fireEvent.change(screen.getByLabelText('Search available players'),
                   { target: { value: '' } })
  fireEvent.click(screen.getByRole('button', { name: 'WR' }))
  expect(names(container)).toEqual(['Bijan', 'Dell'])
  fireEvent.click(screen.getByRole('button', { name: 'ALL' }))
  expect(names(container)).toHaveLength(4)
})

// The one filter that is a question rather than a category: show me only the
// players I lose by waiting. Its cut point is the same 40 that tints a chip
// red, so what it selects is what the reader can already see.
test('the "likely gone" chip filters to the rows tinted red', () => {
  const { container } = draw()
  const chip = screen.getByRole('button', { name: 'Likely gone by your next pick' })
  expect(chip.getAttribute('aria-pressed')).toBe('false')
  fireEvent.click(chip)
  expect(chip.getAttribute('aria-pressed')).toBe('true')
  expect(names(container)).toEqual(['Chase'])
  fireEvent.click(chip)
  expect(names(container)).toHaveLength(4)
})

test('a player the board says is drafted leaves the list', () => {
  const { container } = draw({ draftedIds: new Set(['a']) })
  expect(names(container)).toEqual(['Bijan', 'Chase', 'Dell'])
})

test('the Draft button carries the room\'s one gate', () => {
  const onDraft = vi.fn()
  const { rerender } = draw({ onDraft })
  const buttons = () => screen.getAllByRole('button', { name: 'Draft' }) as HTMLButtonElement[]
  expect(buttons().every((b) => b.disabled)).toBe(true)
  expect(buttons()[0].getAttribute('title')).toBe('Not your turn yet')

  rerender(
    <RoomSimpleList
      candidates={CANDIDATES}
      players={PLAYERS}
      onDraft={onDraft}
      isMyTurn
      onOpenPlayer={() => {}}
      draftedIds={new Set<string>()}
    />,
  )
  expect(buttons().every((b) => b.disabled)).toBe(false)
  fireEvent.click(buttons()[0])
  expect(onDraft.mock.calls[0][0].player_id).toBe('a')
})

// A disabled button that says "not your turn" while the clock is on your seat
// is the one wording that is flatly untrue. Same three answers as the card,
// from the same helper.
test('and says which of the three things is wrong when it is not the turn', () => {
  const title = () => screen.getAllByRole('button', { name: 'Draft' })[0].getAttribute('title')

  draw({ gate: { isMyTurn: false, youAreUp: true, socketAlive: false, locked: false } })
  expect(title()).toBe('Reconnecting to ESPN…')
  cleanup()

  draw({ gate: { isMyTurn: false, youAreUp: true, socketAlive: true, locked: true } })
  expect(title()).toBe('Unlock to draft')
  cleanup()

  draw({ isMyTurn: true,
         gate: { isMyTurn: true, youAreUp: true, socketAlive: true, locked: false } })
  expect(title()).toBeNull()
})

test('a row opens the profile, and its Draft button does not', () => {
  const onOpenPlayer = vi.fn()
  const { container } = draw({ onOpenPlayer })
  const row = container.querySelector('tbody tr[data-pid="b"]') as HTMLElement
  fireEvent.click(row)
  expect(onOpenPlayer.mock.calls[0][0].player_id).toBe('b')

  onOpenPlayer.mockClear()
  fireEvent.click(row.querySelector('.avail-draft-btn') as HTMLElement)
  expect(onOpenPlayer).not.toHaveBeenCalled()
})

// THE NUMBER IS ONLY WORTH SOMETHING IF THE READER CAN WEIGH IT. "The share
// of drafts" is a sentence; "the share of 854 real ESPN drafts" is a claim.
// The count is fetched the first time somebody asks, and never at mount.
test('the header explains the number, with the count of real drafts behind it', async () => {
  draw()
  expect(fetchMarketOverview).not.toHaveBeenCalled()

  fireEvent.focus(screen.getByText('Will he be there?'))
  const tip = screen.getByRole('tooltip')
  // THE CONDITIONING, not just the number: this is counted over the drafts
  // where he was still there at THIS pick, which is the question being asked.
  expect(tip.textContent).toContain('still available at this pick')
  expect(tip.textContent).toContain('still there at pick 20')
  await waitFor(() => expect(tip.textContent).toContain('854 real ESPN drafts'))
  expect(fetchMarketOverview).toHaveBeenCalledTimes(1)

  fireEvent.blur(screen.getByText('Will he be there?'))
  expect(screen.queryByRole('tooltip')).toBeNull()
})

test('a count that never arrives leaves the sentence standing', async () => {
  fetchMarketOverview.mockRejectedValue(new Error('offline'))
  draw()
  fireEvent.focus(screen.getByText('Worth grabbing now'))
  const tip = screen.getByRole('tooltip')
  await waitFor(() => expect(fetchMarketOverview).toHaveBeenCalled())
  expect(tip.textContent).toContain('real ESPN drafts')
  expect(tip.textContent).not.toContain('null')
  // And it says which way a negative number reads, which is the one thing a
  // points figure cannot say for itself.
  expect(tip.textContent).toContain('waiting is the better play')
})

// The archive is only asked about by the two sentences that print its size.
test('a tooltip that does not use the count does not fetch one', () => {
  draw()
  fireEvent.focus(screen.getByText('My guys'))
  expect(screen.getByRole('tooltip').textContent).toContain('my guys')
  fireEvent.focus(screen.getByText('Player'))
  expect(fetchMarketOverview).not.toHaveBeenCalled()
})
