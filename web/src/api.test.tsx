import { afterEach, describe, expect, it, vi } from 'vitest'
import { connectWithToken } from './api'

const params = { leagueId: '1', teamId: '2', swid: '{X}', token: 't', season: '2026' }

function answer(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status, headers: { 'Content-Type': 'application/json' },
  })
}

describe('connectWithToken', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('surfaces the room cap\'s own sentence', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => answer(503, {
      detail: { error: 'at capacity', active: 150, cap: 150,
                message: 'This server is following as many drafts as it can right now.' },
    })))
    await expect(connectWithToken(params)).rejects.toThrow(
      'This server is following as many drafts as it can right now.')
  })

  it('still reports a plain 503', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => answer(503, {
      detail: 'the previous listener did not stop in time -- try again',
    })))
    await expect(connectWithToken(params)).rejects.toThrow('did not stop in time')
  })
})
