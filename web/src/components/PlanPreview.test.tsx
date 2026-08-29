import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { PlanPreview as Preview } from '../api'
import PlanPreview from './PlanPreview'
import { forgetRequests } from '../api'

// WHAT THIS FILE IS FOR. Every number in this card is the server's, so there
// is nothing here worth testing about the arithmetic. What is this component's
// own, and what breaks quietly, is the ATTRIBUTION: the corpus records one
// league size and a reader may be asking about another, so the sentence has to
// say which drafts it counted and the round has to be the round of THOSE
// drafts. A true number in a false sentence is the failure this card is one
// careless edit away from.

const { fetchPlanPreview } = vi.hoisted(() => ({ fetchPlanPreview: vi.fn() }))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchPlanPreview,
}))

function player(name: string, position: string, extra = {}) {
  return {
    player_id: name.toLowerCase().replace(/\W/g, ''), name, position,
    team: 'DET', headshot: null, lasts_pct: null, edge_pts: null,
    edge_at_pick: null, favourite: false, pros: [], cons: [], ...extra,
  }
}

const PLAN: Preview = {
  teams: 8, slot: 6,
  picks: [6, 11, 22, 27],
  opening: [
    { path: ['RB', 'WR', 'WR', 'RB', 'TE'], count: 265, share: 0.31 },
    { path: ['WR', 'WR', 'RB', 'RB', 'TE'], count: 120, share: 0.14 },
  ],
  opening_rounds: 5,
  opening_observed: 854,
  position_runs: { QB: 41, TE: 33, K: null, DST: null },
  corpus: { teams: 8, rounds: 16, drafts: 854 },
  targets: [
    {
      pick_no: 6, round: 1,
      target: player('Bijan Robinson', 'RB', {
        lasts_pct: 41.2, edge_pts: 18.6, edge_at_pick: 11,
        pros: ['likely gone before your next pick (72 %)'],
      }),
      alternates: [player('Ja’Marr Chase', 'WR'), player('Jahmyr Gibbs', 'RB')],
    },
    {
      pick_no: 11, round: 2,
      target: player('Josh Allen', 'QB', { favourite: true, lasts_pct: 90.1 }),
      alternates: [player('Puka Nacua', 'WR')],
    },
    {
      pick_no: 22, round: 3, target: player('Trey McBride', 'TE'),
      alternates: [],
    },
  ],
}

beforeEach(() => {
  forgetRequests()
  fetchPlanPreview.mockResolvedValue(PLAN)
})

afterEach(cleanup)

async function draw(props = {}) {
  const view = render(<PlanPreview teams={8} slot={6} {...props} />)
  await screen.findByText(/usually opens/)
  return view
}

test('the seat’s own picks are drawn on the rule', async () => {
  await draw()

  const rule = screen.getByRole('img', { name: 'Picks 6, 11, 22, 27' })
  for (const pick of ['6', '11', '22', '27']) {
    expect(within(rule).getByText(pick)).toBeTruthy()
  }
})

test('the rule hangs the opening path under the picks it belongs to',
     async () => {
       await draw()

       const rule = screen.getByRole('img', { name: /^Picks/ })
       // Four marks, four positions -- the fifth round of the path has no
       // pick on this rule and must not be drawn against one that is not its.
       expect(within(rule).getAllByText(/^(RB|WR|TE|QB|K|DST)$/)
         .map((el) => el.textContent)).toEqual(['RB', 'WR', 'WR', 'RB'])
     })

test('the opening sentence names the share and the drafts it counted',
     async () => {
       await draw()

       const said = screen.getByText(/usually opens/)
       expect(said.textContent).toMatch(/6th seat of 8/)
       expect(said.textContent).toMatch(/RB–WR–WR–RB–TE/)
       expect(said.textContent).toMatch(/31%/)
       expect(said.textContent).toMatch(/854 recorded 8-team ESPN drafts/)
     })

test('the runs are quoted in the round of the drafts they were counted in',
     async () => {
       await draw()

       // Pick 41 is round 6 of an eight-team draft. Quoting it as round 5 --
       // which is what the reader's own ten-team league would make it -- would
       // be a true number inside a false sentence.
       expect(screen.getByText(/the first QB at pick 41, round 6/)).toBeTruthy()
       expect(screen.getByText(/the first TE at pick 33, round 5/)).toBeTruthy()
     })

test('a seat asking about a shape the archive has not recorded is told so',
     async () => {
       fetchPlanPreview.mockResolvedValue({ ...PLAN, teams: 12, slot: 6 })
       render(<PlanPreview teams={12} slot={6} />)

       await screen.findByText(/usually opens/)
       expect(screen.getByText(/Counted in 8-team drafts, the only shape/))
         .toBeTruthy()
       // And the sentence stops claiming the seat is "of 12", since the path
       // it is about was not walked in a twelve-team room.
       expect(screen.getByText(/usually opens/).textContent)
         .not.toMatch(/seat of 12/)
     })

test('an empty archive costs the opening and not the plan', async () => {
  fetchPlanPreview.mockResolvedValue({
    ...PLAN, opening: [], opening_observed: 0, corpus: null,
    position_runs: { QB: null, TE: null, K: null, DST: null },
  })
  render(<PlanPreview teams={8} slot={6} />)

  expect(await screen.findByText(/Nobody has drafted from this seat/))
    .toBeTruthy()
  expect(screen.getByText('Bijan Robinson')).toBeTruthy()
})

// -- the turn cards -----------------------------------------------------------

test('one card a turn, each naming its pick and round', async () => {
  await draw()

  expect(screen.getByText('Pick 6')).toBeTruthy()
  expect(screen.getByText('Round 1')).toBeTruthy()
  expect(screen.getByText('Pick 11')).toBeTruthy()
  expect(screen.getByText('Round 2')).toBeTruthy()
  expect(screen.getByText('Pick 22')).toBeTruthy()
})

test('the target carries the plan’s two numbers and its first reason',
     async () => {
       await draw()

       expect(screen.getByText('41%')).toBeTruthy()
       expect(screen.getByText('+19 pts')).toBeTruthy()
       expect(screen.getByText('Over waiting to 11')).toBeTruthy()
       expect(screen.getByText('likely gone before your next pick (72 %)'))
         .toBeTruthy()
     })

test('the alternates are named under the target, not ranked beside it',
     async () => {
       await draw()

       expect(screen.getByText('Ja’Marr Chase')).toBeTruthy()
       expect(screen.getByText('Jahmyr Gibbs')).toBeTruthy()
       expect(screen.getAllByText('Or').length).toBeGreaterThan(0)
     })

test('a player the reader starred is marked as theirs', async () => {
  await draw()

  const star = screen.getByLabelText('One of your guys')
  expect(star.closest('.pp-who')?.textContent).toMatch(/Josh Allen/)
})

// -- the states around it -----------------------------------------------------

test('the card holds its height while the answer is in flight', () => {
  fetchPlanPreview.mockReturnValue(new Promise(() => {}))
  const { container } = render(<PlanPreview teams={8} slot={6} />)

  expect(container.querySelector('.pp-wait')).toBeTruthy()
})

test('a refusal is said in the card rather than thrown at the page', async () => {
  fetchPlanPreview.mockRejectedValue(new Error('Slot 11 does not exist'))
  render(<PlanPreview teams={8} slot={11} />)

  expect(await screen.findByText('Slot 11 does not exist')).toBeTruthy()
})

test('a new seat asks a new question', async () => {
  const { rerender } = await draw()
  expect(fetchPlanPreview).toHaveBeenCalledWith(8, 6, '')

  rerender(<PlanPreview teams={12} slot={3} />)

  await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalledWith(12, 3, ''))
})

test('the heading and the note are the caller’s to write', async () => {
  await draw({ heading: 'Your plan', note: 'For Sunday Money — 8 teams, seat 6' })

  expect(screen.getByRole('heading', { name: 'Your plan' })).toBeTruthy()
  expect(screen.getByText('For Sunday Money — 8 teams, seat 6')).toBeTruthy()
})
