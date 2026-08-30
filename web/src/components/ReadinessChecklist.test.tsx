import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import ReadinessChecklist from './ReadinessChecklist'

// WHAT THIS FILE IS FOR. Three things make the difference between a draft
// night that works and one that does not, and all three happen days earlier.
// So: every row that is not done is a BUTTON, because a checklist that only
// reports is a page telling somebody what they have failed to do -- and the
// whole section collapses to one line once there is nothing left to ask for.

afterEach(cleanup)

function draw(props = {}) {
  return render(
    <ReadinessChecklist
      connected
      guys={0}
      dryRun={false}
      onConnect={() => {}}
      onPickGuys={() => {}}
      onDryRun={() => {}}
      {...props}
    />,
  )
}

test('a row that is not done is a button that does it', () => {
  const onPickGuys = vi.fn()
  const onDryRun = vi.fn()
  draw({ onPickGuys, onDryRun })

  fireEvent.click(screen.getByRole('button', { name: 'Pick my guys' }))
  fireEvent.click(screen.getByRole('button', { name: 'Open a mock room' }))
  expect(onPickGuys).toHaveBeenCalled()
  expect(onDryRun).toHaveBeenCalled()
})

test('a row that is done states the fact and offers nothing', () => {
  // One row still open, so the section is still a section: with all three
  // done it collapses, which the test below is about.
  draw({ guys: 7, dryRun: false })

  expect(screen.getByText('7 saved')).toBeTruthy()
  expect(screen.getByText('your leagues are here')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Pick my guys' })).toBeNull()
  expect(screen.getByRole('button', { name: 'Open a mock room' })).toBeTruthy()
})

test('a half-picked list asks for the rest by number', () => {
  draw({ guys: 3 })

  expect(screen.getByRole('button', { name: 'Pick 2 more' })).toBeTruthy()
})

test('the heading counts what is left, not what is done', () => {
  draw({ guys: 7 })

  // Connected and picked; only the dry run is open.
  expect(screen.getByText('1 to do')).toBeTruthy()
})

test('all three done collapses the whole section to one line', () => {
  draw({ guys: 5, dryRun: true })

  expect(screen.getByText(/You’re ready for draft night/)).toBeTruthy()
  expect(screen.queryByRole('heading', { name: 'Ready for draft night' })).toBeNull()
  expect(screen.queryByRole('list')).toBeNull()
})

test('five is the floor, because a plan built from four names is not a plan',
     () => {
       draw({ guys: 4, dryRun: true })

       expect(screen.getByRole('button', { name: 'Pick 1 more' })).toBeTruthy()
       expect(screen.queryByText(/You’re ready/)).toBeNull()
     })
