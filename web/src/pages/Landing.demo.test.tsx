import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { DemoSection } from './Landing'

// WHAT THIS FILE IS FOR. The demo section carries a caption that makes a
// claim -- "a real ESPN mock draft in progress" -- above a room that draws
// "No mock draft is running, and there is none on file to replay" whenever
// the farm is between drafts. The two must never be on screen together: this
// is the one section of the front door whose whole argument is that what you
// are looking at is real, and a sentence contradicting the board under it
// costs more than the sentence is worth.
//
// DemoRoom itself is stubbed. It polls, opens an EventSource and fetches a
// board; none of that is what is under test, and the contract between the two
// is one callback wide.

const { modes } = vi.hoisted(() => ({
  modes: [] as Array<'live' | 'replay' | 'none'>,
}))

vi.mock('../components/DemoRoom', () => ({
  default: ({ onMode }: { onMode?: (m: 'live' | 'replay' | 'none') => void }) => {
    // Announced from an effect, the way the real room announces it: from
    // inside the poll it starts after mounting.
    const mode = modes.shift()
    if (mode) queueMicrotask(() => onMode?.(mode))
    return <div data-testid="room" />
  },
}))

afterEach(() => { cleanup(); modes.length = 0 })

const caption = () => screen.queryByText(/this is what the draft room looks like/)

test('the caption waits for a room before claiming there is one', async () => {
  render(<DemoSection live={false} />)

  expect(await screen.findByTestId('room')).toBeTruthy()
  expect(caption()).toBeNull()
})

test('a room the farm is between drafts on gets no caption', async () => {
  modes.push('none')
  render(<DemoSection live={false} />)

  await waitFor(() => expect(screen.getByTestId('room')).toBeTruthy())
  await new Promise((done) => queueMicrotask(() => done(null)))
  expect(caption()).toBeNull()
})

test('a live room is called a draft in progress', async () => {
  modes.push('live')
  render(<DemoSection live={false} />)

  const said = await screen.findByText(/this is what the draft room looks like/)
  expect(said.textContent).toContain('in progress')
})

test('a replay says it is a replay rather than claiming a live draft', async () => {
  modes.push('replay')
  render(<DemoSection live={false} />)

  const said = await screen.findByText(/this is what the draft room looks like/)
  expect(said.textContent).toContain('played back')
  expect(said.textContent).not.toContain('in progress')
})
