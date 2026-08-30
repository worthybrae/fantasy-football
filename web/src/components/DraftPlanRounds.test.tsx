import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { PlanPreview, PreviewPlayer } from '../api'
import DraftPlanRounds from './DraftPlanRounds'

// WHAT THIS FILE IS FOR. The plan is the best thing this product does and the
// one thing a reader could not see until they were already drafting. Four
// properties make it readable on the Sunday before:
//
//   * eight rounds arrive as eight rows, in order, each naming its own round
//     and its own pick. A ladder that skipped a turn would be a plan for a
//     draft nobody is in;
//   * every row carries the plan's own sentence. The card used to print a
//     bare fragment ("ESPN's #4 overall") against a name, which reads as a
//     label rather than as a reason;
//   * the backups are on the row, by name, so a reader scanning eight rounds
//     never has to open anything to find them;
//   * everything else the planner weighed is behind a disclosure, closed.
//     Eight rounds of five pros and four cons is a spreadsheet.

const { fetchPlanPreview } = vi.hoisted(() => ({ fetchPlanPreview: vi.fn() }))

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../api')>()),
  fetchPlanPreview,
}))

function player(name: string, extra: Partial<PreviewPlayer> = {}): PreviewPlayer {
  return {
    player_id: name.toLowerCase().replace(/\W/g, ''), name, position: 'RB',
    team: 'DET', headshot: null, lasts_pct: 88.2, edge_pts: 4.1,
    edge_at_pick: 15, favourite: false,
    pros: ['ESPN’s #4 overall', '88% still there at pick 6'], cons: [],
    reason: 'ESPN’s #4 overall, and 88% still there at pick 6.', ...extra,
  }
}

/** Seat 6 of ten: turns at 6, 15, 26, 35, 46, 55, 66, 75 -- alternating waits
 *  of nine and eleven picks, which is the shape the ladder draws. */
const PICKS = [6, 15, 26, 35, 46, 55, 66, 75]

const PLAN: PlanPreview = {
  teams: 10, slot: 6, turns: 8, picks: PICKS,
  opening: [], opening_rounds: 5, opening_observed: 0,
  position_runs: {}, corpus: null,
  targets: PICKS.map((pick, i) => ({
    pick_no: pick,
    round: i + 1,
    target: player(`Target ${i + 1}`, {
      reason: `Reason for round ${i + 1}.`,
      pros: [`Pro ${i + 1}a`, `Pro ${i + 1}b`],
      cons: i === 0 ? ['bye week 9 stacks with Somebody'] : [],
    }),
    alternates: [player(`Backup ${i + 1}A`), player(`Backup ${i + 1}B`)],
    backups: [`Backup ${i + 1}A`, `Backup ${i + 1}B`],
  })),
}

beforeEach(() => {
  vi.clearAllMocks()
  fetchPlanPreview.mockResolvedValue(PLAN)
})

afterEach(cleanup)

test('eight turns are eight rounds, in order, each with its own pick', async () => {
  render(<DraftPlanRounds teams={10} slot={6} turns={8} />)

  await screen.findByText('Round 1')
  for (let round = 1; round <= 8; round += 1) {
    expect(screen.getByText(`Round ${round}`)).toBeTruthy()
  }
  expect(screen.getAllByText(/^pick \d+$/).map((n) => n.textContent))
    .toEqual(PICKS.map((pick) => `pick ${pick}`))
})

test('the ask is for as many turns as the caller wants', async () => {
  render(<DraftPlanRounds teams={10} slot={6} turns={8} tag="a,b" />)

  await waitFor(() => expect(fetchPlanPreview).toHaveBeenCalled())
  expect(fetchPlanPreview).toHaveBeenCalledWith(10, 6, 'a,b', 8)
})

test('every row says who to take and why, in one sentence', async () => {
  render(<DraftPlanRounds teams={10} slot={6} turns={8} />)

  expect(await screen.findByText('Target 1')).toBeTruthy()
  expect(screen.getByText('Reason for round 1.')).toBeTruthy()
  expect(screen.getByText('Reason for round 8.')).toBeTruthy()
  // The instruction, not a heading over a list of names.
  expect(screen.getAllByText('Take')).toHaveLength(8)
})

test('the two backups are on the row, by name', async () => {
  render(<DraftPlanRounds teams={10} slot={6} turns={8} />)

  await screen.findByText('Target 1')
  expect(screen.getByText('Backup 1A, Backup 1B')).toBeTruthy()
  expect(screen.getAllByText(/^If he’s gone:/)).toHaveLength(8)
})

test('the wait between turns is drawn, in picks', async () => {
  render(<DraftPlanRounds teams={10} slot={6} turns={8} />)

  await screen.findByText('Target 1')
  // Seven gaps for eight turns, and the first row has none: there is nothing
  // before it to have waited through.
  const waits = screen.getAllByText(/picks later$/).map((n) => n.textContent)
  expect(waits).toEqual(
    ['9 picks later', '11 picks later', '9 picks later', '11 picks later',
     '9 picks later', '11 picks later', '9 picks later'])
})

test('the rest of the reasons are behind a disclosure, closed', async () => {
  render(<DraftPlanRounds teams={10} slot={6} turns={8} />)

  await screen.findByText('Target 1')
  const why = screen.getAllByText('Why him')
  expect(why).toHaveLength(8)
  const details = why[0].closest('details') as HTMLDetailsElement
  expect(details.open).toBe(false)

  fireEvent.click(why[0])
  expect(screen.getByText('Pro 1a')).toBeTruthy()
  expect(screen.getByText('bye week 9 stacks with Somebody')).toBeTruthy()
  expect(screen.getAllByText('For').length).toBeGreaterThan(0)
  expect(screen.getByText('Against')).toBeTruthy()
})

test('a turn the plan could not fill says so rather than printing nothing',
     async () => {
       fetchPlanPreview.mockResolvedValue({
         ...PLAN,
         targets: [{ pick_no: 6, round: 1, target: null, alternates: [],
                     backups: [] }],
       })
       render(<DraftPlanRounds teams={10} slot={6} turns={8} />)

       expect(await screen.findByText(/Take the best player left/)).toBeTruthy()
     })

test('a refused seat prints the server’s sentence and no rows', async () => {
  fetchPlanPreview.mockRejectedValue(new Error('Slot 11 does not exist'))
  render(<DraftPlanRounds teams={8} slot={11} turns={8} />)

  expect(await screen.findByText(/Slot 11 does not exist/)).toBeTruthy()
  expect(screen.queryByText('Round 1')).toBeNull()
})

test('the section keeps its height while the answer is in flight', () => {
  fetchPlanPreview.mockReturnValue(new Promise(() => {}))
  const { container } = render(<DraftPlanRounds teams={10} slot={6} />)

  expect(container.querySelectorAll('.dpr-skel-row').length).toBeGreaterThan(0)
})
