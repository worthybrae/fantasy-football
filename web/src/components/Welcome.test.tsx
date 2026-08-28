import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import Welcome from './Welcome'

// WHAT THIS FILE IS FOR. A QA pass found the welcome card doing two things a
// card should not: it was a full-page scrim, so a visitor's first click on a
// player row hit the overlay and merely dismissed it, and the dismissal was
// forgotten on every navigation, so the card came back on each page of one
// visit. Both are properties, not polish -- the first click a visitor makes
// is the one that decides whether this product does anything.

const { fetchCustody, fetchMarketOverview, fetchMarketSlot, fetchLiveMock } =
  vi.hoisted(() => ({
    fetchCustody: vi.fn(), fetchMarketOverview: vi.fn(),
    fetchMarketSlot: vi.fn(), fetchLiveMock: vi.fn(),
  }))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchCustody, fetchMarketOverview, fetchMarketSlot, fetchLiveMock,
}))

beforeEach(() => {
  window.sessionStorage.clear()
  fetchCustody.mockResolvedValue({ connected: false })
  fetchMarketOverview.mockResolvedValue({ drafts: 900, picks: 100_000 })
  fetchMarketSlot.mockResolvedValue(null)
  // No shortlist: the card's figures fall back, which every slide is written
  // to do, and the three loads still settle.
  fetchLiveMock.mockResolvedValue({ live: false, shortlist: [] })
})

afterEach(cleanup)

async function show() {
  const view = render(<Welcome onStart={() => {}} />)
  await screen.findByText('ESPN Draft Assist')
  return view
}

test('it is a card over the room, not a modal in front of it', async () => {
  const { container } = await show()
  const wrapper = container.querySelector('.wc-scrim') as HTMLElement
  // Not modal: the draft behind it is live, readable and clickable.
  expect(wrapper.getAttribute('aria-modal')).toBeNull()
  // And the page it sits over can still be scrolled.
  expect(document.body.style.overflow).toBe('')
})

test('a click outside the card no longer dismisses it', async () => {
  const { container } = await show()
  fireEvent.mouseDown(container.querySelector('.wc-scrim') as HTMLElement)
  fireEvent.click(container.querySelector('.wc-scrim') as HTMLElement)
  // Still the full card, not the corner pill it shrinks to.
  expect(container.querySelector('.wc-card')).toBeTruthy()
  expect(container.querySelector('.wc-corner')).toBeNull()
})

test('dismissing it is remembered for the rest of the tab', async () => {
  const first = await show()
  fireEvent.click(screen.getByLabelText('Close and keep watching the draft'))
  await waitFor(() => expect(first.container.querySelector('.wc-corner')).toBeTruthy())
  first.unmount()

  // A second page in the same visit: the corner pill, never the card again.
  const again = await show()
  expect(again.container.querySelector('.wc-corner')).toBeTruthy()
  expect(again.container.querySelector('.wc-card')).toBeNull()
})

test('a new tab gets the introduction', async () => {
  const first = await show()
  fireEvent.click(screen.getByLabelText('Close and keep watching the draft'))
  first.unmount()
  window.sessionStorage.clear()   // what a new tab starts with
  const fresh = await show()
  expect(fresh.container.querySelector('.wc-card')).toBeTruthy()
})
