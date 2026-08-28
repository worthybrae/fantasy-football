import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import type { LiveCandidate, LivePlanTurn, Player } from '../../api'
import TargetCards from './TargetCards'

// WHAT THIS FILE IS FOR. These cards are the plan's answer for ONE turn, and
// which turn that is depends on whether the reader is holding the pick. On
// the clock they are a recommendation for this very pick; waiting, they are
// the forecast for his next one. Showing the wrong turn's three names would
// be the old room's mistake in a new place -- numbers captioned as an answer
// to a question they were not asked.

function candidate(id: string, over: Partial<LiveCandidate> = {}): LiveCandidate {
  return {
    player_id: id,
    position: 'RB',
    proj_points: 200,
    espn_rank: 5,
    espn_pos_rank: 2,
    espn_adp: 6,
    market_rank: 5,
    lasts_pct: 40,
    lasts_at_pick: 21,
    edge_pts: 12,
    need: 'starter',
    favourite: false,
    rank: 1,
    ...over,
  }
}

function player(id: string, name: string): Player {
  return {
    player_id: id, name, position: 'RB', team: 'DET', bye: 9,
    market_rank: 5, market_spread: null,
    market_sources: { ffc: null, espn: null, fp: null, mfl: null, cbs: null, fp_tier: null },
    espn_ppr_rank: 5, stats: null, career_games_pg: null, consistency_cv: null,
    consistency_pct: null, season_finishes: null, proj_change: null,
    game_points: null, headshot: null, rookie: false, drafted: false,
    avail_pct: null, ev: null, ev_se: null, rank: 1, tier: 1, edge: null,
    proj_points: 200,
  }
}

const IDS = ['now1', 'now2', 'now3', 'next1', 'next2', 'next3']
const NAMES = ['Now One', 'Now Two', 'Now Three', 'Next One', 'Next Two', 'Next Three']
const PLAYERS: Record<string, Player> = Object.fromEntries(
  IDS.map((id, i) => [id, player(id, NAMES[i])]))
const CANDIDATES = IDS.map((id) => candidate(id))

function turn(pickNo: number, round: number, ids: string[]): LivePlanTurn {
  return {
    pick_no: pickNo,
    round,
    target: { player_id: ids[0], lasts_pct: 100, edge_pts: 14, pros: ['★ favourite'], cons: [] },
    alternates: ids.slice(1).map((id) => (
      { player_id: id, lasts_pct: 60, edge_pts: 3, pros: [], cons: [] })),
  }
}

const PLAN = [
  turn(12, 1, ['now1', 'now2', 'now3']),
  turn(21, 2, ['next1', 'next2', 'next3']),
]

function draw(isMyTurn: boolean) {
  return render(
    <TargetCards
      plan={PLAN}
      candidates={CANDIDATES}
      players={PLAYERS}
      onDraft={() => {}}
      isMyTurn={isMyTurn}
      pickNo={12}
      settings={null}
      recompute={null}
      onOpenPlayer={() => {}}
    />,
  )
}

afterEach(cleanup)

test('on the clock, the three cards are this turn\'s', () => {
  const { container } = draw(true)
  expect(container.querySelectorAll('.target-card')).toHaveLength(3)
  expect(screen.getByText('Now One')).toBeTruthy()
  expect(screen.getByText('Now Three')).toBeTruthy()
  expect(screen.queryByText('Next One')).toBeNull()
  expect(screen.getByText('Take one of these')).toBeTruthy()
  expect(screen.getByText('pick 12 · round 1')).toBeTruthy()
})

test('waiting, they are the next turn\'s', () => {
  const { container } = draw(false)
  expect(container.querySelectorAll('.target-card')).toHaveLength(3)
  expect(screen.getByText('Next One')).toBeTruthy()
  expect(screen.queryByText('Now One')).toBeNull()
  expect(screen.getByText('Your next turn')).toBeTruthy()
  expect(screen.getByText('pick 21 · round 2')).toBeTruthy()
})

test('with no plan at all, the top of the board stands in', () => {
  render(
    <TargetCards
      plan={[]}
      candidates={CANDIDATES}
      players={PLAYERS}
      onDraft={() => {}}
      isMyTurn={false}
      pickNo={12}
      settings={null}
      recompute={null}
      onOpenPlayer={() => {}}
    />,
  )
  expect(screen.getByText('Now One')).toBeTruthy()
  expect(screen.getByText("the top of ESPN's board")).toBeTruthy()
})
