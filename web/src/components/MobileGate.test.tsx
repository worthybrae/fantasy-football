import { cleanup, fireEvent, render, screen } from '@testing-library/react'
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

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

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
