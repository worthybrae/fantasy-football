import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import {
  cachedGet, fetchCustody, fetchLiveMock, fetchMarketOverview, forgetRequests,
} from './api'

// WHAT THIS FILE IS FOR. One load of the landing page fired `/api/demo/live`
// five times, `/api/market/overview` twice, `/api/market/slot/1` twice and
// `/api/espn/custody` twice -- three components each asking for the same
// room. The property that fixes it is small and worth pinning: two callers,
// one request.

beforeEach(() => {
  forgetRequests()
  vi.useFakeTimers({ shouldAdvanceTime: true })
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

function answer(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200, headers: { 'Content-Type': 'application/json' },
  })
}

test('two callers in the same tick make one request', async () => {
  const fetcher = vi.fn(async () => 'once')
  const [a, b] = await Promise.all([
    cachedGet('k', fetcher, 5_000),
    cachedGet('k', fetcher, 5_000),
  ])
  expect(fetcher).toHaveBeenCalledTimes(1)
  expect([a, b]).toEqual(['once', 'once'])
})

test('a caller after the window asks again', async () => {
  const fetcher = vi.fn(async () => 'x')
  await cachedGet('k', fetcher, 5_000)
  await cachedGet('k', fetcher, 5_000)
  expect(fetcher).toHaveBeenCalledTimes(1)
  vi.advanceTimersByTime(5_001)
  await cachedGet('k', fetcher, 5_000)
  expect(fetcher).toHaveBeenCalledTimes(2)
})

test('a slow request is shared however long it takes', async () => {
  let release: (v: string) => void = () => {}
  const fetcher = vi.fn(() => new Promise<string>((r) => { release = r }))
  const first = cachedGet('slow', fetcher, 10)
  vi.advanceTimersByTime(1_000)
  const second = cachedGet('slow', fetcher, 10)
  release('done')
  expect(await first).toBe('done')
  expect(await second).toBe('done')
  expect(fetcher).toHaveBeenCalledTimes(1)
})

test('a failure is not remembered', async () => {
  const fetcher = vi.fn(async () => { throw new Error('nope') })
  await expect(cachedGet('bad', fetcher, 5_000)).rejects.toThrow('nope')
  await expect(cachedGet('bad', fetcher, 5_000)).rejects.toThrow('nope')
  expect(fetcher).toHaveBeenCalledTimes(2)
})

test('the landing page\'s three shared endpoints go out once each', async () => {
  const calls: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    calls.push(url)
    return answer(url.includes('overview') ? { drafts: 1, picks: 2 } : { connected: false })
  }))
  await Promise.all([
    fetchMarketOverview(), fetchMarketOverview(),
    fetchCustody(), fetchCustody(),
    fetchLiveMock(), fetchLiveMock(), fetchLiveMock(),
  ])
  expect(calls.filter((u) => u.includes('/api/market/overview'))).toHaveLength(1)
  expect(calls.filter((u) => u.includes('/api/espn/custody'))).toHaveLength(1)
  expect(calls.filter((u) => u.includes('/api/demo/live'))).toHaveLength(1)
})

test('the demo room\'s own read is never answered from the cache', async () => {
  const calls: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    calls.push(url)
    return answer({ live: true })
  }))
  await fetchLiveMock()
  await fetchLiveMock({ fresh: true })
  expect(calls).toHaveLength(2)
})
