import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { PlayerProfileData } from '../../api'
import { cachedProfile, loadProfile } from './CellTip'
import { forgetRequests } from '../../api'

// WHAT THIS FILE IS FOR. `cachedProfile` is what four readers paint from on
// the frame they open -- the hover tip, the board peek, the target cards and
// the profile card. Painting is the privilege it has to earn: it may not grow
// without limit in a room that is open all afternoon, and it may not hand a
// reader an hour-old injury status as though it were current.

const { fetchProfile } = vi.hoisted(() => ({ fetchProfile: vi.fn() }))

vi.mock('../../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api')>()),
  fetchProfile,
}))

function payload(id: string): PlayerProfileData {
  return { header: { player_id: id } } as unknown as PlayerProfileData
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  forgetRequests()
  fetchProfile.mockImplementation(async (id: string) => payload(id))
})

afterEach(() => { vi.useRealTimers(); vi.clearAllMocks() })

test('it holds the last 64 players and lets the rest go', async () => {
  for (let i = 1; i <= 65; i += 1) await loadProfile(`p${i}`)
  // The one asked for longest ago is gone; the recent ones stand.
  expect(cachedProfile('p1')).toBeNull()
  expect(cachedProfile('p65')).not.toBeNull()
  expect(cachedProfile('p64')).not.toBeNull()
})

test('a payload old enough to be wrong is not painted', async () => {
  await loadProfile('x')
  expect(cachedProfile('x')).not.toBeNull()

  // Ten minutes is the line: a profile carries news and an injury status,
  // and past it the reader gets a skeleton and a fresh answer instead.
  vi.advanceTimersByTime(10 * 60_000 + 1)
  expect(cachedProfile('x')).toBeNull()
})

test('a re-read refreshes its place in the queue', async () => {
  for (let i = 1; i <= 64; i += 1) await loadProfile(`q${i}`)
  // Touch the oldest, then push one more in: the entry that leaves should be
  // the next-oldest rather than the one just used.
  forgetRequests()          // past the request window, so this really re-reads
  await loadProfile('q1')
  await loadProfile('q65')
  expect(cachedProfile('q1')).not.toBeNull()
  expect(cachedProfile('q2')).toBeNull()
})
