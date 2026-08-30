import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import type { LiveCandidate, LivePlanTurn, Player } from '../../api'
import TakeNowCard, { reasonSentence } from './TakeNowCard'

// WHAT THIS FILE IS FOR. This card is the room's whole answer for a reader
// who wants one: a name, a reason, two backups and the button. Three things
// can go wrong with it and each is guarded here.
//
// (1) It could recommend during somebody else's pick, or forecast during
//     your own -- the card describes a TURN, and which turn it is comes off
//     the clock, not off the plan's array order.
// (2) It could offer a pick that cannot be sent. The Draft button's gate is
//     the same single boolean the cheat sheet's is, and it must stay that.
// (3) It could say something the plan did not. Every reason on this card is
//     matched from scoring/plan.py's own strings; a pro this card cannot
//     match is left out rather than half-said.

function candidate(over: Partial<LiveCandidate> & { player_id: string }): LiveCandidate {
  return {
    position: 'RB',
    proj_points: 240,
    espn_rank: 4,
    espn_pos_rank: 2,
    espn_adp: 6,
    market_rank: 5,
    lasts_pct: 65,
    lasts_at_pick: 11,
    edge_pts: 8,
    edge_at_pick: 11,
    need: 'starter',
    favourite: false,
    rank: 1,
    ...over,
  }
}

function player(id: string, name: string): Player {
  return {
    player_id: id, name, position: 'RB', team: 'DET', bye: 9,
    market_rank: 1, market_spread: null,
    market_sources: { ffc: null, espn: null, fp: null, mfl: null, cbs: null, fp_tier: null },
    espn_ppr_rank: 1, stats: null, career_games_pg: null, consistency_cv: null,
    consistency_pct: null, season_finishes: null, proj_change: null,
    game_points: null, headshot: null, rookie: false, drafted: false,
    avail_pct: null, ev: null, ev_se: null, rank: 1, tier: 1, edge: null,
    proj_points: 240,
  }
}

const CANDIDATES = [
  candidate({ player_id: 'a', rank: 1, favourite: true }),
  candidate({ player_id: 'b', rank: 2, lasts_pct: 30, position: 'WR' }),
  candidate({ player_id: 'c', rank: 3, lasts_pct: 85, position: 'TE' }),
]

const PLAYERS: Record<string, Player> = {
  a: player('a', 'Ashton'), b: player('b', 'Bijan'), c: player('c', 'Chase'),
}

// The turn on the clock (pick 6), with the plan's own reasons on its target.
const TURN: LivePlanTurn = {
  pick_no: 6,
  round: 1,
  target: {
    player_id: 'a',
    lasts_pct: 65,
    edge_pts: 8,
    edge_at_pick: 11,
    pros: ['★ favourite', "ESPN's #4 overall", '65% still there at pick 11',
           'biggest drop-off at RB before pick 11', 'fills RB1'],
    cons: ['health 2/5'],
  },
  alternates: [
    { player_id: 'b', lasts_pct: 30, edge_pts: 3, edge_at_pick: 11, pros: [], cons: [] },
    { player_id: 'c', lasts_pct: 85, edge_pts: -2, edge_at_pick: 11, pros: [], cons: [] },
  ],
}

// His next turn, which is what the card describes while somebody else picks.
const NEXT: LivePlanTurn = { ...TURN, pick_no: 11, round: 2 }

function draw(over: Partial<Parameters<typeof TakeNowCard>[0]> = {}) {
  return render(
    <TakeNowCard
      plan={[TURN]}
      candidates={CANDIDATES}
      players={PLAYERS}
      onDraft={() => {}}
      isMyTurn={false}
      onTheClock
      pickNo={6}
      onOpenPlayer={() => {}}
      {...over}
    />,
  )
}

afterEach(cleanup)

test('on the clock: one name, one reason, two backups and the button', () => {
  const { container } = draw()
  expect(screen.getByText('Take now')).toBeTruthy()
  expect(screen.getByText('pick 6 · round 1')).toBeTruthy()

  // The lead, at the size of the decision, with his own line under it.
  expect(container.querySelector('.takenow-name-btn')?.textContent).toBe('Ashton')
  expect(container.querySelector('.takenow-sub')?.textContent).toBe('RB · DET · BYE 9')
  expect(screen.getByLabelText('One of your guys')).toBeTruthy()

  // ONE sentence, and it is the plan's strongest reason plus the chance he
  // lasts -- not the plan's own print order, which opens with the star.
  expect(container.querySelector('.takenow-why')?.textContent)
    .toBe('RBs dry up before pick 11 — he\'s 65% to be there.')

  // The two alternates, named, in the plan's order.
  const backups = container.querySelector('.takenow-backups')
  expect(backups?.textContent).toBe("If he's gone: Bijan, Chase")

  expect(screen.getByRole('button', { name: 'Draft' })).toBeTruthy()
})

test('off the clock: the next turn, and who is likely to be there for it', () => {
  const { container } = render(
    <TakeNowCard
      plan={[NEXT]}
      candidates={CANDIDATES}
      players={PLAYERS}
      onDraft={() => {}}
      isMyTurn={false}
      onTheClock={false}
      pickNo={6}
      onOpenPlayer={() => {}}
    />,
  )
  expect(screen.getByText('Up next')).toBeTruthy()
  expect(screen.getByText('pick 11 · round 2')).toBeTruthy()
  // No pick to send while you are not holding one.
  expect(screen.queryByRole('button', { name: 'Draft' })).toBeNull()
  expect(container.querySelector('.takenow-why')).toBeNull()

  // The target and both alternates, each with the chance HE is there at
  // that turn -- the plan's own figures, in the plan's own order.
  const rows = Array.from(container.querySelectorAll('.takenow-likely-row'))
    .map((row) => row.textContent)
  expect(rows).toEqual(['Ashton65%', 'Bijan30%', 'Chase85%'])

  // The three bands, tinted the same way everywhere the number appears.
  const bands = Array.from(container.querySelectorAll('.lasts-chip'))
    .map((chip) => chip.className)
  expect(bands[0]).toContain('is-mid')
  expect(bands[1]).toContain('is-low')
  expect(bands[2]).toContain('is-high')
})

// THE GATE IS UNCHANGED. DraftRoom computes one boolean -- his turn, on a
// live socket, in a room he has paid for -- and every Draft button in the
// room reads it. A card that let a pick through on a different test would
// send picks ESPN refuses.
test('the Draft button is disabled off your turn, and sends the pick on it', () => {
  draw()
  const off = screen.getByRole('button', { name: 'Draft' }) as HTMLButtonElement
  expect(off.disabled).toBe(true)
  expect(off.getAttribute('title')).toBe('Not your turn yet')
  cleanup()

  const onDraft = vi.fn()
  draw({ isMyTurn: true, onDraft })
  const on = screen.getByRole('button', { name: 'Draft' }) as HTMLButtonElement
  expect(on.disabled).toBe(false)
  fireEvent.click(on)
  expect(onDraft).toHaveBeenCalledTimes(1)
  expect(onDraft.mock.calls[0][0].player_id).toBe('a')
})

test('a player the list no longer carries cannot be drafted from the card', () => {
  const onDraft = vi.fn()
  draw({ isMyTurn: true, onDraft, candidates: [] })
  const button = screen.getByRole('button', { name: 'Draft' }) as HTMLButtonElement
  expect(button.disabled).toBe(true)
  fireEvent.click(button)
  expect(onDraft).not.toHaveBeenCalled()
})

test('a turn with no clear target says so instead of naming somebody', () => {
  render(
    <TakeNowCard
      plan={[{ pick_no: 6, round: 1, target: null, alternates: [] }]}
      candidates={CANDIDATES}
      players={PLAYERS}
      onDraft={() => {}}
      isMyTurn
      onTheClock
      pickNo={6}
      onOpenPlayer={() => {}}
    />,
  )
  expect(screen.getByRole('status').textContent).toContain('No clear pick for pick 6')
  expect(screen.queryByRole('button', { name: 'Draft' })).toBeNull()
})

// The sentence is the one place the model's words become the reader's, so it
// is checked without a card around it.
test('the reason is matched from the plan, never invented', () => {
  const base = {
    lastsPct: 65, lastsAtPick: 11, edgePts: 8, edgeAtPick: 11, position: 'RB',
  }
  // The pick is named once: the lead already said pick 11.
  expect(reasonSentence({ ...base, pros: ['biggest drop-off at RB before pick 11'] }))
    .toBe('RBs dry up before pick 11 — he\'s 65% to be there.')
  // No lead the card can match: the availability clause stands alone, and
  // names the pick itself.
  expect(reasonSentence({ ...base, pros: ['bench'] }))
    .toBe("He's 65% to be there at pick 11.")
  // Under 40% the sentence stops sounding like a reason to wait.
  expect(reasonSentence({ ...base, pros: ['bench'], lastsPct: 22 }))
    .toBe('Only 22% to be there at pick 11.')
  // Nothing measured at all: no sentence, rather than a made-up one.
  expect(reasonSentence({ ...base, pros: [], lastsPct: null, edgePts: null }))
    .toBeNull()
  // A room with no plan still has the edge on its candidate rows.
  expect(reasonSentence({ ...base, pros: [], lastsPct: null }))
    .toBe("He's worth about 8 more points than the next RB you would get at pick 11.")
})
