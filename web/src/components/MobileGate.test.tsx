import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import MobileGate from './MobileGate'

// WHAT THIS FILE IS FOR. This page renders for EVERY route on a small screen,
// and it carried a 2.56 MB clip with `preload="metadata"` -- which the phones
// it exists for read as "fetch the file". The property is simple and worth
// pinning: nothing is downloaded until somebody asks for it.

const { fetchMarketOverview } = vi.hoisted(() => ({ fetchMarketOverview: vi.fn() }))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchMarketOverview,
}))

beforeEach(() => {
  fetchMarketOverview.mockResolvedValue({ drafts: 900, picks: 100_000 })
  // jsdom has no matchMedia; the gate is a media query.
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: true, media: query,
    addEventListener: () => {}, removeEventListener: () => {},
    addListener: () => {}, removeListener: () => {}, onchange: null,
    dispatchEvent: () => false,
  }))
})

afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers() })

test('the clip is a poster until somebody taps it', () => {
  const { container } = render(<MobileGate><span>desktop</span></MobileGate>)
  const video = container.querySelector('video') as HTMLVideoElement
  // NOTHING TO FETCH. No src, and preload off, so the element costs the
  // poster and nothing else.
  expect(video.getAttribute('src')).toBeNull()
  expect(video.getAttribute('preload')).toBe('none')
  expect(video.getAttribute('poster')).toBe('/demo-poster.jpg')
  expect(video.hasAttribute('autoplay')).toBe(false)

  fireEvent.click(screen.getByRole('button', { name: 'Play the draft room clip' }))

  expect(video.getAttribute('src')).toBe('/demo.webm')
  expect(video.hasAttribute('autoplay')).toBe(true)
  expect(screen.queryByRole('button', { name: 'Play the draft room clip' })).toBeNull()
})

test('a desktop reader gets the app, not the gate', () => {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: false, media: query,
    addEventListener: () => {}, removeEventListener: () => {},
    addListener: () => {}, removeListener: () => {}, onchange: null,
    dispatchEvent: () => false,
  }))
  render(<MobileGate><span>desktop</span></MobileGate>)
  expect(screen.getByText('desktop')).toBeTruthy()
})

// The confirmation is not a new name for the control. Left standing, "Link
// copied" WAS the label -- a reader who came back to the tab a minute later
// found a button that no longer said what it does -- so it reverts, and the
// timer is the thing worth pinning: a revert that never fires is the old bug
// back again, and one that fires immediately is a confirmation nobody sees.
test('the copied label goes back to being the button’s name', async () => {
  vi.useFakeTimers()
  const writeText = vi.fn().mockResolvedValue(undefined)
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })

  render(<MobileGate><span>desktop</span></MobileGate>)
  const send = screen.getByRole('button', { name: 'Send yourself the link' })
  // Awaited: `send()` writes to the clipboard before it sets the state, and
  // that is a microtask fake timers do not touch.
  await act(async () => { fireEvent.click(send) })
  expect(writeText).toHaveBeenCalledWith(`${window.location.origin}/`)
  expect(screen.getByRole('button', { name: 'Link copied' })).toBeTruthy()

  // Long enough to be read, and gone after that.
  act(() => { vi.advanceTimersByTime(2500) })
  expect(screen.getByRole('button', { name: 'Link copied' })).toBeTruthy()
  act(() => { vi.advanceTimersByTime(200) })
  expect(screen.getByRole('button', { name: 'Send yourself the link' })).toBeTruthy()
})
