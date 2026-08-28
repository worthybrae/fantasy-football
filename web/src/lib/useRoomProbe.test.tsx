import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { LiveState } from '../api'
import { ROOM_PROBE_BUSY_MS, ROOM_PROBE_IDLE_MS, useRoomProbe } from './useRoomProbe'

// WHAT THIS FILE IS FOR. The landing page asked `/api/live/state` every 2.5
// seconds forever, including for a signed-out visitor who cannot have a room
// at all -- 1,440 requests an hour to be told "no". The cadence must follow
// the answer.

const { fetchLiveState } = vi.hoisted(() => ({ fetchLiveState: vi.fn() }))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchLiveState,
}))

function state(over: Partial<LiveState>): LiveState {
  return {
    active: false, picks_made: 0, on_the_clock: null, my_slot: null,
    draft_started: false, candidates: [], candidates_as_of_pick: null,
    last_poll_at: null, stale: false, unmapped_picks: [],
    listener_error: null, listener_alive: false, recompute_error: null,
    socket_alive: false, autodraft: null, report_url: null, ms_remaining: null,
    settings: { teams: null, rounds: null, starters: {}, flex_slots: null,
                bench: null, scoring_format: null },
    my_roster: [], ...over,
  } as LiveState
}

function Probe({ busy = false, onLive = () => {} }: { busy?: boolean; onLive?: () => void }) {
  useRoomProbe({ enabled: true, busy, onLive })
  return null
}

beforeEach(() => { vi.useFakeTimers({ shouldAdvanceTime: true }) })
afterEach(() => { cleanup(); vi.useRealTimers(); fetchLiveState.mockReset() })

async function settle() {
  await act(async () => { await Promise.resolve(); await Promise.resolve() })
}

test('a visitor with no session is asked about once every thirty seconds', async () => {
  fetchLiveState.mockResolvedValue(state({}))
  render(<Probe />)
  await settle()
  expect(fetchLiveState).toHaveBeenCalledTimes(1)

  // The old cadence would have made eleven more calls in this window.
  await act(async () => { vi.advanceTimersByTime(ROOM_PROBE_IDLE_MS - 1000) })
  await settle()
  expect(fetchLiveState).toHaveBeenCalledTimes(1)

  await act(async () => { vi.advanceTimersByTime(1000) })
  await settle()
  expect(fetchLiveState).toHaveBeenCalledTimes(2)
})

test('a session in the process moves it to the quick cadence', async () => {
  // Active, but nothing alive behind it: a reason to look often, never a
  // reason to send anybody to a board.
  fetchLiveState.mockResolvedValue(state({ active: true }))
  const onLive = vi.fn()
  render(<Probe onLive={onLive} />)
  await settle()
  expect(fetchLiveState).toHaveBeenCalledTimes(1)
  expect(onLive).not.toHaveBeenCalled()

  await act(async () => { vi.advanceTimersByTime(ROOM_PROBE_BUSY_MS) })
  await settle()
  expect(fetchLiveState).toHaveBeenCalledTimes(2)
})

test('a connect running on the page starts quick', async () => {
  fetchLiveState.mockResolvedValue(state({}))
  render(<Probe busy />)
  await settle()
  await act(async () => { vi.advanceTimersByTime(ROOM_PROBE_BUSY_MS) })
  await settle()
  expect(fetchLiveState).toHaveBeenCalledTimes(2)
})

test('a room that is really alive stops the probe and reports it', async () => {
  fetchLiveState.mockResolvedValue(state({ active: true, listener_alive: true }))
  const onLive = vi.fn()
  render(<Probe onLive={onLive} />)
  await settle()
  expect(onLive).toHaveBeenCalledTimes(1)

  await act(async () => { vi.advanceTimersByTime(ROOM_PROBE_IDLE_MS * 2) })
  await settle()
  // Nothing further: the page has somewhere to send them now.
  expect(fetchLiveState).toHaveBeenCalledTimes(1)
})
