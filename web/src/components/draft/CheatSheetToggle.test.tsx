import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import CheatSheetToggle, { useRoomView } from './CheatSheetToggle'

// WHAT THIS FILE IS FOR. Two properties, and both are about a reader who is
// not in the room right now. A FRESH browser opens on the simple view --
// that is the whole redesign, and a default that quietly flipped back to
// fourteen columns would undo it. And a reader who asked for the cheat sheet
// gets it again next time, including after the reload that follows a dropped
// socket, which is the moment he would least like to be handed a different
// room.

function Harness() {
  const [view, setView] = useRoomView()
  return (
    <>
      <CheatSheetToggle view={view} onChange={setView} />
      <span data-testid="view">{view}</span>
    </>
  )
}

const shown = () => screen.getByTestId('view').textContent

beforeEach(() => { window.localStorage.clear() })
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

test('a fresh browser opens on the simple view', () => {
  render(<Harness />)
  expect(shown()).toBe('simple')
  expect(screen.getByRole('button', { name: 'Simple' }).getAttribute('aria-pressed'))
    .toBe('true')
  expect(screen.getByRole('button', { name: 'Cheat sheet' }).getAttribute('aria-pressed'))
    .toBe('false')
})

test('the switch switches, and the browser remembers which', () => {
  const first = render(<Harness />)
  fireEvent.click(screen.getByRole('button', { name: 'Cheat sheet' }))
  expect(shown()).toBe('cheat')
  expect(window.localStorage.getItem('room-view')).toBe('cheat')

  // A reload, which is what a dropped socket costs a reader mid-draft.
  first.unmount()
  render(<Harness />)
  expect(shown()).toBe('cheat')

  fireEvent.click(screen.getByRole('button', { name: 'Simple' }))
  expect(shown()).toBe('simple')
  expect(window.localStorage.getItem('room-view')).toBe('simple')
})

test('a value this build does not know is the simple view, not a broken room', () => {
  window.localStorage.setItem('room-view', 'whatever-an-older-build-wrote')
  render(<Harness />)
  expect(shown()).toBe('simple')
})

// Safari's private mode throws on the storage accessor itself. The choice is
// then worth one session rather than nothing at all.
test('a browser that refuses storage still switches', () => {
  vi.stubGlobal('localStorage', {
    getItem: () => { throw new Error('denied') },
    setItem: () => { throw new Error('denied') },
  })
  render(<Harness />)
  expect(shown()).toBe('simple')
  fireEvent.click(screen.getByRole('button', { name: 'Cheat sheet' }))
  expect(shown()).toBe('cheat')
})
