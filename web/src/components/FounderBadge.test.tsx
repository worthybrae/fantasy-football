import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import FounderBadge from './FounderBadge'

// The badge makes a promise ("every draft free") and an offer ("connect ESPN
// to claim one"), and the failure that matters is showing the wrong one of
// them to the wrong reader. Each row below is one reader.

const { fetchMe } = vi.hoisted(() => ({ fetchMe: vi.fn() }))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchMe,
}))

const NOBODY = { connected: false, founder: false, ordinal: null,
                 founders_left: 63 }

beforeEach(() => {
  fetchMe.mockReset()
  fetchMe.mockResolvedValue(NOBODY)
})

afterEach(cleanup)

test('a founder is told which one they are', async () => {
  fetchMe.mockResolvedValue(
    { connected: true, founder: true, ordinal: 37, founders_left: 63 })

  render(<FounderBadge />)

  expect(await screen.findByText('Founding member #37 — every draft free'))
    .toBeTruthy()
})

test('a signed-out reader is told what is left and how to take it', async () => {
  render(<FounderBadge />)

  expect(await screen.findByText(
    '63 founder spots left — connect ESPN to claim one')).toBeTruthy()
})

test('the last seat is a spot, not spots', async () => {
  fetchMe.mockResolvedValue({ ...NOBODY, founders_left: 1 })

  render(<FounderBadge />)

  expect(await screen.findByText(
    '1 founder spot left — connect ESPN to claim one')).toBeTruthy()
})

test('a signed-out reader with no seats left is offered nothing', async () => {
  fetchMe.mockResolvedValue({ ...NOBODY, founders_left: 0 })

  const { container } = render(<FounderBadge />)
  await vi.waitFor(() => expect(fetchMe).toHaveBeenCalled())

  expect(container.querySelector('.founder-badge')).toBeNull()
})

test('somebody already connected is not told to connect', async () => {
  // The request that would have claimed a seat is the one that just answered,
  // so a connected non-founder cannot take one however many are left.
  fetchMe.mockResolvedValue(
    { connected: true, founder: false, ordinal: null, founders_left: 63 })

  const { container } = render(<FounderBadge />)
  await vi.waitFor(() => expect(fetchMe).toHaveBeenCalled())

  expect(container.textContent).toBe('')
})

test('a probe that fails draws nothing rather than a wrong claim', async () => {
  fetchMe.mockRejectedValue(new Error('the entitlement store is not usable'))

  const { container } = render(<FounderBadge />)
  await vi.waitFor(() => expect(fetchMe).toHaveBeenCalled())

  expect(container.textContent).toBe('')
})

test('a page that already has the answer is not asked to fetch it again',
     async () => {
  render(<FounderBadge me={{ connected: true, founder: true, ordinal: 2,
                             founders_left: 98 }} />)

  expect(await screen.findByText('Founding member #2 — every draft free'))
    .toBeTruthy()
  expect(fetchMe).not.toHaveBeenCalled()
})
