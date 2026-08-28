import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { PickerBoundary, PickerFallback } from './PickerBoundary'

// WHAT THIS FILE IS FOR. The picker is a lazy chunk, and a chunk request is
// the one part of this page that fails on its own -- most often minutes after
// a deploy, when the browser holds an index naming a file the server has
// replaced. React's default for an uncaught render error is to blank the
// tree, so without this a 5 KB file takes the whole dashboard with it. And
// neither stand-in may trap anybody behind a dim page.

function Boom(): never {
  throw new Error('Failed to fetch dynamically imported module')
}

beforeEach(() => {
  // React logs the caught error, and so does the boundary. Neither is a test
  // failure; both are noise in the run.
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => { cleanup(); vi.restoreAllMocks() })

test('a chunk that will not load says so instead of blanking the page', () => {
  const onClose = vi.fn()
  render(
    <PickerBoundary onClose={onClose}>
      <Boom />
    </PickerBoundary>,
  )
  expect(screen.getByRole('alert').textContent).toContain('reload the page')
  fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  expect(onClose).toHaveBeenCalledTimes(1)
})

test('the failed state takes the same two exits as the dialog', () => {
  const onClose = vi.fn()
  const { container } = render(
    <PickerBoundary onClose={onClose}><Boom /></PickerBoundary>,
  )
  fireEvent.keyDown(document, { key: 'Escape' })
  expect(onClose).toHaveBeenCalledTimes(1)
  fireEvent.mouseDown(container.querySelector('.fav-modal-backdrop') as HTMLElement)
  expect(onClose).toHaveBeenCalledTimes(2)
})

test('so does the waiting state', () => {
  const onClose = vi.fn()
  const { container } = render(<PickerFallback onClose={onClose} />)
  const scrim = container.querySelector('.fav-modal-backdrop') as HTMLElement
  expect(scrim.getAttribute('aria-busy')).toBe('true')
  fireEvent.keyDown(document, { key: 'Escape' })
  expect(onClose).toHaveBeenCalledTimes(1)
  fireEvent.mouseDown(scrim)
  expect(onClose).toHaveBeenCalledTimes(2)
})

test('a child that renders is left alone', () => {
  render(
    <PickerBoundary onClose={() => {}}>
      <p>the picker</p>
    </PickerBoundary>,
  )
  expect(screen.getByText('the picker')).toBeTruthy()
  expect(screen.queryByRole('alert')).toBeNull()
})
