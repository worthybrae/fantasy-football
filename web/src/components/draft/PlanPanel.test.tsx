import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import type { LivePlanTurn, Player } from '../../api'
import PlanPanel from './PlanPanel'

// WHAT THIS FILE IS FOR. The panel's whole job is to print what the plan
// says, verbatim: the turn, the target, the two figures, and the reasons for
// and against. Every string here comes from scoring/plan.py, so the failure
// this guards against is a panel that quietly drops the reasoning -- which
// is the only part of a plan that can be argued with.

function player(id: string, name: string): Player {
  return {
    player_id: id, name, position: 'WR', team: 'CIN', bye: 10,
    market_rank: 3, market_spread: null,
    market_sources: { ffc: null, espn: null, fp: null, mfl: null, cbs: null, fp_tier: null },
    espn_ppr_rank: 3, stats: null, career_games_pg: null, consistency_cv: null,
    consistency_pct: null, season_finishes: null, proj_change: null,
    game_points: null, headshot: null, rookie: false, drafted: false,
    avail_pct: null, ev: null, ev_se: null, rank: 3, tier: 1, edge: null,
    proj_points: 260,
  }
}

const TURN: LivePlanTurn = {
  pick_no: 21,
  round: 2,
  target: {
    player_id: 'p1',
    lasts_pct: 62,
    edge_pts: 18,
    pros: ['★ favourite', '62% still there at pick 21'],
    cons: ['ADP 14 vs ESPN 22 — may go earlier'],
  },
  alternates: [
    { player_id: 'p2', lasts_pct: 40, edge_pts: 4, pros: [], cons: [] },
    { player_id: 'p3', lasts_pct: 88, edge_pts: -2, pros: [], cons: [] },
  ],
}

const PLAYERS: Record<string, Player> = {
  p1: player('p1', 'Puka'), p2: player('p2', 'Pittman'), p3: player('p3', 'Pickens'),
}

afterEach(cleanup)

test('renders the turn, its target and the plan\'s own reasons', () => {
  render(<PlanPanel plan={[TURN]} players={PLAYERS}
                    favourites={new Set(['p1'])} />)
  expect(screen.getByText('Pick 21')).toBeTruthy()
  expect(screen.getByText('Round 2')).toBeTruthy()
  expect(screen.getByText('Puka')).toBeTruthy()
  expect(screen.getByText('62%')).toBeTruthy()
  expect(screen.getByText('+18')).toBeTruthy()
  // The reasoning, both ways round.
  expect(screen.getByText('★ favourite')).toBeTruthy()
  expect(screen.getByText('62% still there at pick 21')).toBeTruthy()
  expect(screen.getByText('ADP 14 vs ESPN 22 — may go earlier')).toBeTruthy()
  // And the two players it would settle for instead.
  expect(screen.getByText('Pittman')).toBeTruthy()
  expect(screen.getByText('Pickens')).toBeTruthy()
  expect(screen.getByLabelText('One of your guys')).toBeTruthy()
})

// EVERY REMAINING TURN, not a quiet first few: the header counts the whole
// plan, and a list that stopped short of its own count would be wrong about
// itself. The panel scrolls instead.
test('draws every turn the plan carries', () => {
  const turns = Array.from({ length: 9 }, (_, i) => (
    { ...TURN, pick_no: 21 + i * 12, round: 2 + i }))
  const { container } = render(<PlanPanel plan={turns} players={PLAYERS} />)
  expect(container.querySelectorAll('.plan-turn')).toHaveLength(9)
  expect(screen.getByText('9 turns left')).toBeTruthy()
})

test('a reader with no turns left gets no panel, not an empty one', () => {
  const { container } = render(<PlanPanel plan={[]} players={PLAYERS} />)
  expect(container.querySelector('.plan-panel')).toBeNull()
})


test('a turn nobody cleared reads "no clear target" and keeps its place', () => {
  const empty: LivePlanTurn = { pick_no: 36, round: 3, target: null, alternates: [] }
  render(<PlanPanel plan={[TURN, empty]} players={PLAYERS} />)
  expect(screen.getByText('Pick 36')).toBeTruthy()
  expect(screen.getByText('No clear target')).toBeTruthy()
  expect(screen.getByText('Puka')).toBeTruthy()
})
