import { Profiler, type ProfilerOnRenderCallback, type ComponentProps } from 'react'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { UpcomingDraft } from '../api'
import Dashboard from './Dashboard'

// WHAT THIS FILE IS FOR, AND WHAT IT MEASURED.
//
// The owner's report was "I was scrolling and the screen wasn't moving". The
// cause was one line: the dashboard held a `setInterval` that set state every
// second, so every second React re-rendered the whole signed-in page. Measured
// here with React's Profiler before the fix, in jsdom, with nothing at all
// happening on the page:
//
//   saved list  -- 1 commit per second, 2.8ms, 25 rows each searching a
//                  252-row board for a name
//   empty state -- 1 commit per second, 19.4ms, a tree of 1,835 nodes
//                  including 252 player rows and 252 <img> elements
//
// In a browser that is a main-thread pause every second on a page somebody is
// trying to scroll. The fix is that the clock is a store (lib/clock.ts) and
// only the components that print time subscribe to it, so the test below is
// the property that matters: A TICK MUST NOT RENDER THE PAGE. The two heavy
// children are counted through mock wrappers -- each one counts how many times
// the page asked for that child -- and the countdown is checked to be still
// moving, because a clock that ticks nothing is also a fix nobody wants.

const renders = { guys: 0, leagues: 0 }

vi.mock('./MyGuysTable', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./MyGuysTable')>()
  return {
    ...actual,
    default: (props: ComponentProps<typeof actual.default>) => {
      renders.guys += 1
      return <actual.default {...props} />
    },
  }
})

vi.mock('./OtherDrafts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./OtherDrafts')>()
  return {
    default: (props: ComponentProps<typeof actual.default>) => {
      renders.leagues += 1
      return <actual.default {...props} />
    },
  }
})

const { fetchFavorites, fetchPlayers, fetchMockRooms,
        fetchRoomProgress, fetchFavoritesOutlook, fetchPlanPreview } = vi.hoisted(() => ({
  fetchFavorites: vi.fn(), fetchPlayers: vi.fn(),
  fetchMockRooms: vi.fn(), fetchRoomProgress: vi.fn(),
  fetchFavoritesOutlook: vi.fn(), fetchPlanPreview: vi.fn(),
}))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchFavorites, fetchPlayers, fetchMockRooms,
  fetchRoomProgress, fetchFavoritesOutlook, fetchPlanPreview,
}))

// A draft half an hour out: inside the hour, which is the case the old page
// put on a one-second interval.
const LEAGUES: UpcomingDraft[] = [{
  league_id: '1', name: 'A League', team_id: '2', team_name: 'Mine',
  season: 2026, teams: 10, draft_type: 'Snake', live: false,
  draft_at: new Date(Date.now() + 30 * 60_000).toISOString(),
} as UpcomingDraft, {
  // A second league, so the compact list below the hero has a row to draw --
  // the hero's own draft is not repeated in it.
  league_id: '2', name: 'Another League', team_id: '3', team_name: 'Mine too',
  season: 2026, teams: 10, draft_type: 'Snake', live: false,
  draft_at: new Date(Date.now() + 90 * 60_000).toISOString(),
} as UpcomingDraft]

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  renders.guys = 0
  renders.leagues = 0
  fetchMockRooms.mockResolvedValue({ rooms: [], next: null })
  fetchRoomProgress.mockResolvedValue([])
  fetchFavorites.mockResolvedValue(['p1', 'p2', 'p3', 'p4', 'p5'])
  // Never settling, both of them: this file counts COMMITS, and a payload
  // landing mid-measurement is a commit the clock did not cause.
  fetchFavoritesOutlook.mockReturnValue(new Promise(() => {}))
  fetchPlanPreview.mockReturnValue(new Promise(() => {}))
})

afterEach(() => { cleanup(); vi.useRealTimers() })

test('a second passing renders the countdown and nothing else', async () => {
  const commits: number[] = []
  const onRender: ProfilerOnRenderCallback = (_id, _phase, actual) => { commits.push(actual) }
  render(
    <Profiler id="db" onRender={onRender}>
      <MemoryRouter>
        <Dashboard leagues={LEAGUES} onJoin={() => {}} onOpenRoom={() => {}}
                   onConnect={() => {}} />
      </MemoryRouter>
    </Profiler>,
  )
  await waitFor(() => expect(fetchFavorites).toHaveBeenCalled())
  await act(async () => { await Promise.resolve() })

  // EVERY clock on the page, not one of them. The next draft is stated twice
  // now -- once at the top as the page's own answer, once in the league list
  // as one row among the account's -- and both are the same subscription.
  const clock = () => screen.getAllByText(/^\d+:\d\d$/)
    .map((node) => node.textContent).join(' | ')
  const before = clock()
  const guys = renders.guys
  const leagues = renders.leagues
  const settled = commits.length
  // The wrappers are really in the tree: without this, a mock that silently
  // failed to apply would make every assertion below trivially true.
  expect(guys).toBeGreaterThan(0)
  expect(leagues).toBeGreaterThan(0)

  await act(async () => { vi.advanceTimersByTime(5000) })

  // The page did not ask for either heavy child again...
  expect(renders.guys).toBe(guys)
  expect(renders.leagues).toBe(leagues)
  // ...the commits that did happen are the countdown's own, and cheap...
  expect(commits.length - settled).toBeLessThanOrEqual(5)
  // THE MIDDLE TICK, NOT THE WORST ONE. What is being measured is React's
  // work for one tick of a clock, which is a fraction of a millisecond on a
  // quiet machine. This file does not run on a quiet machine: under the full
  // suite a single commit that lost the CPU mid-render measures the
  // scheduler, not the page, and has been seen at 3.5 ms while every other
  // tick stayed under one. The median cannot be moved by one unlucky commit,
  // and the thing this guards against -- a tick that re-renders the tables --
  // is not one slow commit but every commit slow, which five milliseconds
  // catches comfortably.
  const ticks = commits.slice(settled).sort((a, b) => a - b)
  expect(ticks.length).toBeGreaterThan(0)
  expect(ticks[Math.floor(ticks.length / 2)]).toBeLessThan(5)
  // ...and the clock is still a clock.
  expect(clock()).not.toBe(before)
})

test('the board is never fetched by the page itself', async () => {
  render(
    <MemoryRouter>
      <Dashboard leagues={LEAGUES} onJoin={() => {}} onOpenRoom={() => {}}
                   onConnect={() => {}} />
    </MemoryRouter>,
  )
  await waitFor(() => expect(fetchFavorites).toHaveBeenCalled())
  await act(async () => { vi.advanceTimersByTime(30_000) })
  expect(fetchPlayers).not.toHaveBeenCalled()
})
