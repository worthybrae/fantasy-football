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
//
// The same fault in a picture, which is the other half of this file: the rule
// at the top used to hang the CORPUS's opening path under the picks while the
// cards below named the plan's own players, so a reader saw WR-RB-RB-WR over
// three cards reading RB-QB-TE. Two answers to one question, and the numbers
// on both were true.

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
  corpus: { teams: 8, rounds: 16, format: 'ppr', drafts: 854 },
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
  await screen.findByText(/usually open/)
  return view
}

test('the seat’s own picks are drawn on the rule', async () => {
  await draw()

  const rule = screen.getByRole('img', { name: 'Picks 6, 11, 22, 27' })
  for (const pick of ['6', '11', '22', '27']) {
    expect(within(rule).getByText(pick)).toBeTruthy()
  }
})

test('the rule badges each pick with the card that is drawn for it',
     async () => {
       await draw()

       const rule = screen.getByRole('img', { name: /^Picks/ })
       // The plan's own three targets -- RB, QB, TE -- and NOT the corpus's
       // opening path (RB-WR-WR-RB), which is what this used to draw over
       // cards naming somebody else. The fourth mark has no card, so it
       // carries no badge rather than borrowing the third one's.
       expect(within(rule).getAllByText(/^(RB|WR|TE|QB|K|DST)$/)
         .map((el) => el.textContent)).toEqual(['RB', 'QB', 'TE'])
     })

test('the section says what its numbers are before showing any', async () => {
  await draw()

  expect(screen.getByText(
    /ESPN’s order, adjusted for who lasts to your next pick/)).toBeTruthy()
})

test('what rooms usually do is one muted sentence, with its drafts named',
     async () => {
       await draw()

       const said = screen.getByText(/usually open/)
       expect(said.textContent).toMatch(/RB–WR–WR–RB–TE/)
       expect(said.textContent).toMatch(/31 % of 854 recorded 8-team PPR drafts/)
       // Background, not the answer: it is set in the note register rather
       // than as the card's own claim, which is what it was read as.
       expect(said.className).toContain('pp-note')
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

       await screen.findByText(/usually open/)
       expect(screen.getByText(
         /Counted in 8-team PPR drafts — the shape with the most on record; your 12-team league will get its own numbers once enough are recorded/))
         .toBeTruthy()
       // And the sentence itself names the shape it counted, so the caveat
       // is a second reading of the same fact rather than the only one.
       expect(screen.getByText(/usually open/).textContent)
         .toMatch(/8-team PPR drafts/)
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

test('the target’s two numbers each say which pick they are about',
     async () => {
       await draw()

       // "Still there · 41 %" is a claim about pick 6 and nothing else, and
       // read as a fact about the player it is how somebody comes to plan on
       // 41 % at a turn where it is 90.
       expect(screen.getByText('Still there at pick 6')).toBeTruthy()
       expect(screen.getByText('41 %')).toBeTruthy()
       expect(screen.getByText('vs waiting to 11')).toBeTruthy()
       expect(screen.getByText('+19 pts')).toBeTruthy()
       // ESPN's own rank on him, which is the plan's first reason.
       expect(screen.getByText('likely gone before your next pick (72 %)'))
         .toBeTruthy()
     })

test('an edge of nothing is said in words rather than as a minus sign',
     async () => {
       // The plan pricing a turn at or below zero is not a penalty on the
       // player -- it is "nobody better is going anywhere, so he keeps". A
       // bare "−3 pts" beside a name says the opposite.
       fetchPlanPreview.mockResolvedValue({
         ...PLAN,
         targets: [{
           ...PLAN.targets[0],
           target: player('Bijan Robinson', 'RB',
                          { lasts_pct: 41.2, edge_pts: -3.4, edge_at_pick: 21 }),
         }],
       })
       render(<PlanPreview teams={8} slot={6} />)

       expect(await screen.findByText(
         /No better RB is likely gone by 21 — take him only if nothing above falls/))
         .toBeTruthy()
       expect(screen.queryByText(/-3 pts|−3 pts/)).toBeNull()
       expect(screen.queryByText(/vs waiting to 21/)).toBeNull()
       // The turn's other number is untouched: it is about a different thing.
       expect(screen.getByText('41 %')).toBeTruthy()
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
  await draw({ heading: 'Your plan', note: 'For Sunday Money — 8 teams' })

  expect(screen.getByRole('heading', { name: 'Your plan' })).toBeTruthy()
  expect(screen.getByText('For Sunday Money — 8 teams')).toBeTruthy()
})

test('the caller’s seat control sits in the section’s head', async () => {
  // The seat is what the section is about, so the control for it belongs
  // beside the heading rather than in a card somewhere below the answer.
  await draw({ control: <button type="button">Seat</button> })

  const head = document.querySelector('.db-sec-head')
  expect(within(head as HTMLElement).getByRole('button', { name: 'Seat' }))
    .toBeTruthy()
})
