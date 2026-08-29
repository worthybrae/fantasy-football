import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import type { LiveCandidate, Player } from '../../api'
import AvailableList from './AvailableList'

// WHAT THIS FILE IS FOR. The redesign's first decision is that the room's
// order is ESPN's order -- no re-ranking of our own anywhere in the live
// path. The property that holds is therefore about the ORDER OF THE ROWS,
// not about a number in a cell: a list handed to this component in any order
// must come out in ESPN's. The second is the star, which is the only thing
// on a row that comes from the reader's own account.

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
    edge_pts: 10,
    edge_at_pick: 20,
    need: 'starter',
    favourite: false,
    rank: 1,
    ...over,
  }
}

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

// Deliberately handed to the component out of order, and with `rank` (the
// server's own list position) disagreeing with `espn_rank`: a component that
// simply rendered the array it was given, or that fell back to `rank`, would
// pass a test built on a pre-sorted list.
const CANDIDATES = [
  candidate({ player_id: 'c', espn_rank: 30, rank: 1 }),
  candidate({ player_id: 'a', espn_rank: 4, rank: 3, favourite: true }),
  candidate({ player_id: 'b', espn_rank: 11, rank: 2 }),
  candidate({ player_id: 'd', espn_rank: null, rank: 4 }),
]

// Six rows that would all clear the pulse floor on their own.
const AT_RISK = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6'].map((id, i) => candidate({
  player_id: id, espn_rank: i + 1, rank: i + 1, lasts_pct: 5 + i * 3,
}))

const PLAYERS: Record<string, Player> = {
  a: player('a', 'Ashton'), b: player('b', 'Bijan'),
  c: player('c', 'Chase'), d: player('d', 'Dell'),
}

function draw() {
  return render(
    <AvailableList
      candidates={CANDIDATES}
      players={PLAYERS}
      onDraft={() => {}}
      isMyTurn={false}
      onOpenPlayer={() => {}}
      draftedIds={new Set<string>()}
      settings={null}
    />,
  )
}

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

test('opens in ESPN rank order, with the unranked last', () => {
  const { container } = draw()
  const names = Array.from(container.querySelectorAll('tbody .avail-name'))
    .map((el) => el.textContent)
  expect(names).toEqual(['Ashton', 'Bijan', 'Chase', 'Dell'])
})

// A QA pass counted seven rows breathing at once: seven infinite background
// animations on the main thread, and a table where so much moves that nothing
// stands out. Three is the cap, and they are the three in most danger rather
// than the first three on screen.
test('at most three rows pulse, and they are the three least likely to last', () => {
  const players: Record<string, Player> = {}
  for (const c of AT_RISK) players[c.player_id] = player(c.player_id, c.player_id)
  const { container } = render(
    <AvailableList
      candidates={AT_RISK}
      players={players}
      onDraft={() => {}}
      isMyTurn={false}
      onOpenPlayer={() => {}}
      draftedIds={new Set<string>()}
      settings={null}
    />,
  )
  const pulsing = Array.from(container.querySelectorAll('tr.avail-row-pulse'))
    .map((row) => row.getAttribute('data-pid'))
  expect(pulsing).toHaveLength(3)
  expect(new Set(pulsing)).toEqual(new Set(['r1', 'r2', 'r3']))
})

test('stars the reader\'s own guys, and only them', () => {
  const { container } = draw()
  const stars = screen.getAllByLabelText('One of your guys')
  expect(stars).toHaveLength(1)
  // In HIS row, not merely somewhere on the page.
  const row = container.querySelector('tbody tr[data-pid="a"]')
  expect(row?.contains(stars[0])).toBe(true)
})

// THE HEADER ROW AND THE BODY ROW ARE ONE TABLE. Every column class in this
// file is also a width in App.css and, for four of them, a `display: none`
// in a compact tier -- so a class that is on the <th> and not on the <td>
// (or the other way round) is a column the browser hides on one row and
// keeps on the other. That is what happened: `Reliable`'s header was
// `avail-col-steady` and its cell was `avail-col-health`, so under 1360px
// the header row lost a cell the body still had, every label from Growth
// rightward printed over the column to its left, and the Draft button's
// header fell off the end of the row. jsdom applies no media queries, so
// what is checked here is the thing the media queries key off: the two rows
// agree, column for column, about what they are.
const DROPS_AT: Record<string, number> = {
  'avail-col-cons': 1500,
  'avail-col-change': 1500,
  'avail-col-steady': 1360,
  'avail-col-finish': 1200,
  'avail-col-games': 1120,
}

function colClasses(cell: Element): string[] {
  return Array.from(cell.classList).filter((c) => c.startsWith('avail-col-')).sort()
}

test('every column class on a header is on its cell, and the reverse', () => {
  const { container } = draw()
  const heads = Array.from(container.querySelectorAll('thead th'))
  const cells = Array.from(container.querySelectorAll('tbody tr:first-child td'))
  expect(heads).toHaveLength(cells.length)

  // Column for column, the classes that decide whether the column is on
  // screen at this width have to match.
  heads.forEach((th, i) => {
    const dropped = (cell: Element) => colClasses(cell).filter((c) => c in DROPS_AT)
    expect([i, dropped(th)]).toEqual([i, dropped(cells[i])])
  })

  // And no column class exists on one row alone.
  const set = (cellList: Element[]) => new Set(cellList.flatMap(colClasses))
  expect([...set(heads)].sort()).toEqual([...set(cells)].sort())
})

// The window, as the table asks about it: `matchMedia` with the max-width
// out of the query, which is what the compact tiers are written in.
function atWidth(width: number): void {
  vi.stubGlobal('matchMedia', (query: string) => {
    const max = Number(/max-width:\s*(\d+)px/.exec(query)?.[1] ?? Number.MAX_SAFE_INTEGER)
    return {
      matches: width <= max, media: query,
      addEventListener: () => {}, removeEventListener: () => {},
      addListener: () => {}, removeListener: () => {}, onchange: null,
      dispatchEvent: () => false,
    }
  })
}

function sortedHeader(container: HTMLElement): string | null {
  const th = container.querySelector('thead th[aria-sort]:not([aria-sort="none"])')
  return th?.className.split(' ').find((c) => c.startsWith('avail-col-')) ?? null
}

// A SORT NEEDS A COLUMN TO POINT AT. Reliable is gone under 1360px, so a
// table sorted by it had no caret anywhere on the header row and no header
// left to click to put it back -- the rows were simply in an order nothing
// on screen explained. Under that width the order falls back to ESPN's.
test('a sort by a column the width has dropped falls back to ESPN', () => {
  atWidth(1300)
  const { container } = draw()
  fireEvent.click(screen.getByRole('button', { name: 'Reliable' }))
  expect(sortedHeader(container)).toBe('avail-col-rank')
  expect(Array.from(container.querySelectorAll('tbody .avail-name')).map((el) => el.textContent))
    .toEqual(['Ashton', 'Bijan', 'Chase', 'Dell'])
})

test('and takes it, on a window wide enough to show the column', () => {
  atWidth(1600)
  const { container } = draw()
  fireEvent.click(screen.getByRole('button', { name: 'Reliable' }))
  expect(sortedHeader(container)).toBe('avail-col-steady')
})
