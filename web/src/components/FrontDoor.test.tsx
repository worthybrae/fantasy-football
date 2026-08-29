import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { MarketOverview, PlanPreview } from '../api'
import { Hero, Steps, WhatYouGet } from './FrontDoor'
import { forgetRequests } from '../api'

// WHAT THIS FILE IS FOR. The landing page's whole job is to be believed, and
// what makes it believable is that every figure on it is a reading off this
// deployment rather than a number somebody typed. So these tests are about
// provenance, not layout: given a payload, does the page print THAT number,
// and does it say where it came from.
//
// The second thing they guard is the failure: both endpoints can be down on a
// deployment whose archive is mid-write, and a front door that renders an
// error where a claim should be is worse than one that says less.

const { fetchMarketOverview, fetchPlanPreview } = vi.hoisted(() => ({
  fetchMarketOverview: vi.fn(),
  fetchPlanPreview: vi.fn(),
}))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchMarketOverview,
  fetchPlanPreview,
}))

const OVERVIEW: MarketOverview = {
  drafts: 854, picks: 109312, human_picks: 71204,
  teams: 8, rounds: 16, from: '2026-07-01', to: '2026-08-28',
  median_humans: 5,
}

function player(name: string, position: string, extra = {}) {
  return {
    player_id: name.toLowerCase().replace(/\W/g, ''), name, position,
    team: 'DET', headshot: null, lasts_pct: null, edge_pts: null,
    edge_at_pick: null, favourite: false, pros: [], cons: [], ...extra,
  }
}

// TWO ANSWERS, BECAUSE THE PAGE ASKS TWICE. It opens by asking about the seat
// a visitor is most likely to recognise -- six of ten -- and the endpoint
// answers with ten-team pick numbers over paths counted in the only shape the
// farm has recorded, which is eight. That answer is a mismatch, and the page
// throws it away: it asks again for the corpus's own shape and draws THAT.
const ASKED: PlanPreview = {
  teams: 10, slot: 6,
  picks: [6, 15, 26, 35],
  opening: [{ path: ['RB', 'WR', 'WR', 'RB', 'TE'], count: 265, share: 0.31 }],
  opening_rounds: 5,
  opening_observed: 854,
  position_runs: { QB: 41, TE: 33, K: 120, DST: 118 },
  corpus: { teams: 8, rounds: 16, format: 'ppr', drafts: 854 },
  targets: [
    { pick_no: 6, round: 1, target: player('Bijan Robinson', 'RB'),
      alternates: [player('Ja’Marr Chase', 'WR')] },
    { pick_no: 15, round: 2,
      target: player('Puka Nacua', 'WR', { lasts_pct: 38.4 }),
      alternates: [] },
    { pick_no: 26, round: 3, target: player('Trey McBride', 'TE'),
      alternates: [] },
  ],
}

/** The one the page ends up with: seat 6 of an eight-team draft, whose turns
 *  really are 6, 11, 22 and 27 -- the same drafts the paths were counted in,
 *  so every number on the page belongs to one league shape. */
const PLAN: PlanPreview = {
  ...ASKED,
  teams: 8,
  picks: [6, 11, 22, 27],
  targets: [
    { pick_no: 6, round: 1, target: player('Bijan Robinson', 'RB'),
      alternates: [player('Ja’Marr Chase', 'WR')] },
    { pick_no: 11, round: 2,
      target: player('Puka Nacua', 'WR', { lasts_pct: 38.4 }),
      alternates: [] },
    { pick_no: 22, round: 3, target: player('Trey McBride', 'TE'),
      alternates: [] },
  ],
}

beforeEach(() => {
  forgetRequests()
  // Call counts, not just implementations: the hero asks the plan endpoint
  // twice on purpose, and a history that carried over from the test above
  // would make "twice" unreadable.
  vi.clearAllMocks()
  fetchMarketOverview.mockResolvedValue(OVERVIEW)
  fetchPlanPreview.mockImplementation(
    async (teams: number) => (teams === PLAN.teams ? PLAN : ASKED))
  // FounderBadge asks for /api/account/me directly. Absent on a deployment
  // that predates the founders work, which is the case these tests run in.
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
    { ok: false, status: 404, json: async () => ({}) }))
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

// -- the hero -----------------------------------------------------------------

test('the page opens with one sentence saying what the tool does', () => {
  render(<Hero onStart={() => {}} />)

  const h1 = screen.getByRole('heading', { level: 1 })
  expect(h1.textContent).toMatch(/who lasts to your pick/)
  expect(h1.textContent).toMatch(/plan for every round/)
})

test('the hero draws the seat it is talking about', async () => {
  render(<Hero onStart={() => {}} />)

  const seat = await screen.findByLabelText('One seat, drawn')
  expect(within(seat).getByText('Seat 6 of 8 · every turn you own')).toBeTruthy()
  const rule = within(seat).getByRole('img', { name: 'Picks 6, 11, 22, 27' })
  expect(within(rule).getByText('27')).toBeTruthy()
  expect(within(seat).getByText(/854 recorded 8-team PPR drafts open this way/))
    .toBeTruthy()
})

test('the seat drawn is the shape the paths were counted in, not the one asked',
     async () => {
       // The bug this replaces: an eight-team room's opening hung under a
       // ten-team seat's pick numbers, drawn as though it were one draft.
       render(<Hero onStart={() => {}} />)

       await screen.findByLabelText('One seat, drawn')
       expect(fetchPlanPreview.mock.calls.map((c) => c.slice(0, 2)))
         .toEqual([[10, 6], [8, 6]])
     })

test('the seat drawn names the drafts it came out of, being nobody\u2019s league',
     async () => {
       render(<Hero onStart={() => {}} />)

       const seat = await screen.findByLabelText('One seat, drawn')
       expect(within(seat).getByText(
         /Counted in 8-team PPR drafts — the shape with the most on record; your 10-team league will get its own numbers once enough are recorded/))
         .toBeTruthy()
     })

test('an archive already of the shape asked about is asked once, and unqualified',
     async () => {
       fetchPlanPreview.mockResolvedValue(
         { ...ASKED, corpus: { teams: 10, rounds: 16, format: 'ppr', drafts: 412 } })
       render(<Hero onStart={() => {}} />)

       const seat = await screen.findByLabelText('One seat, drawn')
       expect(within(seat).getByText('Seat 6 of 10 · every turn you own'))
         .toBeTruthy()
       expect(fetchPlanPreview).toHaveBeenCalledTimes(1)
       expect(screen.queryByText(/Counted in/)).toBeNull()
     })

test('a second read that fails draws no seat rather than the mismatched one',
     async () => {
       fetchPlanPreview.mockImplementation(async (teams: number) => {
         if (teams === 10) return ASKED
         throw new Error('the corpus went away')
       })
       render(<Hero onStart={() => {}} />)

       await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalledTimes(2))
       expect(screen.queryByLabelText('One seat, drawn')).toBeNull()
     })

test('with no plan to read, the hero draws no seat rather than a made-up one',
     async () => {
       fetchPlanPreview.mockRejectedValue(new Error('no board'))
       render(<Hero onStart={() => {}} />)

       await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalled())
       expect(screen.queryByLabelText('One seat, drawn')).toBeNull()
       expect(screen.getByRole('heading', { level: 1 })).toBeTruthy()
     })

test('the sub-line counts the archive it is read from', async () => {
  render(<Hero onStart={() => {}} />)

  expect(await screen.findByText('854')).toBeTruthy()
  expect(screen.getByText('109,312')).toBeTruthy()
})

test('the call to action names the offer', () => {
  render(<Hero onStart={() => {}} />)

  expect(screen.getByRole('button',
    { name: /Connect ESPN — free for the first 100/ })).toBeTruthy()
})

test('an archive that will not answer costs the count, not the page', async () => {
  fetchMarketOverview.mockRejectedValue(new Error('busy'))
  render(<Hero onStart={() => {}} />)

  await waitFor(() => expect(fetchMarketOverview).toHaveBeenCalled())
  expect(screen.getByRole('heading', { level: 1 })).toBeTruthy()
  expect(screen.getByRole('button', { name: /Connect ESPN/ })).toBeTruthy()
  expect(screen.queryByText('854')).toBeNull()
})

// -- the three steps ----------------------------------------------------------

test('three steps, in the order somebody does them', () => {
  render(<Steps onStart={() => {}} />)

  const steps = screen.getAllByRole('listitem')
  expect(steps).toHaveLength(3)
  expect(steps.map((s) => within(s).getByRole('heading').textContent))
    .toEqual(['Connect ESPN', 'Pick your guys', 'Open the room'])
})

test('each step carries one drawing of the thing it asks for', () => {
  render(<Steps onStart={() => {}} />)

  const figures = screen.getAllByRole('img')
  expect(figures).toHaveLength(3)
  // Named, not decorative: the drawings carry the instruction as much as the
  // sentences do, and a screen reader that got three empty boxes would be
  // reading a shorter page than everybody else.
  for (const figure of figures) {
    expect(figure.getAttribute('aria-label')).toBeTruthy()
  }
})

// -- what you get -------------------------------------------------------------

test('four cards, each leading with a number off this deployment', async () => {
  render(<WhatYouGet />)

  await screen.findByText('38%')
  const cards = screen.getAllByRole('listitem')
  expect(cards).toHaveLength(4)
  expect(cards.map((c) => within(c).getByRole('heading').textContent)).toEqual([
    'Who is still there at your pick',
    'One name a round, and why',
    'Which of your players you can have',
    'Where players actually go',
  ])
})

test('the lasts figure is the plan’s own, for a named player and pick',
     async () => {
       render(<WhatYouGet />)

       expect(await screen.findByText('38%')).toBeTruthy()
       expect(screen.getByText(/Puka Nacua.s chance of reaching pick 11/)).toBeTruthy()
     })

test('the plan card names the seat’s first pick and who it opens with',
     async () => {
       render(<WhatYouGet />)

       expect(await screen.findByText('Pick 6')).toBeTruthy()
       expect(screen.getByText(/opens with Bijan Robinson \(RB\)/)).toBeTruthy()
     })

test('the by-pick card names the seat, and its turns in the sentence',
     async () => {
       render(<WhatYouGet />)

       // The seat, not its picks: the hero's rule already draws those, and
       // the page's one memorable figure should not be printed twice.
       expect(await screen.findByText('6 of 8')).toBeTruthy()
       expect(screen.getByText(/the turns seat 6 of 8 owns — 6, 11, 22, 27/))
         .toBeTruthy()
     })

test('the ADP card counts the picks people made, and links to the pages',
     async () => {
       render(<WhatYouGet />)

       expect(await screen.findByText('71,204')).toBeTruthy()
       expect(screen.getByText(/out of 109,312 recorded/)).toBeTruthy()
       expect(screen.getByRole('link', { name: /See ADP for every player/ })
         .getAttribute('href')).toBe('/adp')
     })

test('with no plan to read, the cards say the general thing rather than break',
     async () => {
       fetchPlanPreview.mockRejectedValue(new Error('no board'))
       render(<WhatYouGet />)

       await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalled())
       expect(screen.getAllByRole('listitem')).toHaveLength(4)
       expect(screen.getAllByText('—').length).toBeGreaterThan(0)
     })
