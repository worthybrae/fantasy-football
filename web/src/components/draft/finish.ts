// Positional finish: how good a season was, and how to colour it.
//
// Shared by the board's `FinishArc` and the hover panel behind it. They draw
// different shapes -- a bar per season, a row of dots per season -- and they
// must agree on what "RB12" means, or the small picture and the big one would
// tell a reader different things about the same year.

// How many of a position START in this league, which is what makes a finish
// mean anything: the last startable back is a different rank in an 8-team
// league than a 12-team one, and every tier boundary moves with it.
//
// DERIVED FROM THE LEAGUE, not assumed. This was hardcoded to a 12-team
// league -- RB24, TE12 -- which in the owner's own 8-team league overstated
// the startable line by eight ranks at running back and shifted every colour
// on the board's finish column with it.
const FALLBACK_STARTERS: Record<string, number> = {
  QB: 12, TE: 12, RB: 24, WR: 24, K: 12, DST: 12,
}

export type StarterCounts = { teams: number | null; starters: Record<string, number> }

export function startersAt(position: string, league?: StarterCounts | null): number {
  const teams = league?.teams
  const slots = league?.starters?.[position]
  // Both, or neither: a real team count against a missing slot count would
  // silently make every position a one-starter position.
  if (teams && slots) return teams * slots
  // Before a session is connected there is no league to ask, and a
  // twelve-team shape is the least surprising thing to draw until there is.
  return FALLBACK_STARTERS[position] ?? 24
}

export function finishTone(finish: number, starters: number): string {
  if (finish <= starters / 4) return 'is-elite'
  if (finish <= starters / 2) return 'is-strong'
  if (finish <= starters) return 'is-starter'
  if (finish <= starters * 2) return 'is-fringe'
  return 'is-out'
}

// Taller is better, because rank runs the other way. Capped at three tiers
// of starters: stretching the axis to reach a TE40 would squash every
// meaningful season into the top of a 16px strip.
export function finishHeight(finish: number, starters: number): number {
  const floor = starters * 3
  return 12 + (1 - Math.min(finish, floor) / floor) * 88
}

// A finish is a PLACE, not a quantity. Ten filled dots read as "how much of
// something", which is the wrong question for RB8 -- the answer a reader
// wants is where on the position's ladder he came down. So the panel draws
// the ladder itself, with the tiers marked on it, and puts him on it.
//
// Log-spaced, because the ladder is not evenly interesting: RB1 to RB6 is
// the difference between rounds, and RB50 to RB70 is the difference between
// two players nobody drafts. On a linear axis the entire startable range
// would sit in the left third.
export const FINISH_FLOOR_TIERS = 3

function floorFor(starters: number): number {
  return starters * FINISH_FLOOR_TIERS
}

/** Where a finish sits on the track, 0 (best) to 1 (off the bottom). */
export function finishPosition(finish: number, starters: number): number {
  const floor = floorFor(starters)
  const rank = Math.min(Math.max(finish, 1), floor)
  return Math.log(rank) / Math.log(floor)
}


// Most recent season first. Mirrors scoring/config.py's RECENCY_WEIGHTS
// (0.5 / 0.3 / 0.2), which is the weighting the rest of the project already
// uses whenever it asks "what is this player now" -- so the column and the
// board's own production factor cannot disagree about how much last year
// counts against three years ago.
export const FINISH_WEIGHTS = [0.5, 0.3, 0.2] as const

/** One number for a career: average finish, weighted toward recent seasons.
 *
 *  The column used to sort on the LATEST season alone, which made one bad
 *  year look like a collapse and one good year like an arrival. Weighting
 *  says what a run of seasons says: a back who went RB4, RB8, RB4 outranks
 *  one who went RB30, RB30, RB6, even though the second finished better last
 *  year.
 *
 *  Renormalized over the seasons a player actually has, so a rookie is judged
 *  on his one season rather than having two missing ones count as zero
 *  weight -- and lower is better, because it is still a rank.
 */
export function weightedFinish(arc: [number, number][] | null | undefined): number | null {
  if (!arc || !arc.length) return null
  const recent = arc.slice().sort((a, b) => b[0] - a[0]).slice(0, FINISH_WEIGHTS.length)
  let sum = 0
  let weight = 0
  recent.forEach(([, finish], i) => {
    sum += finish * FINISH_WEIGHTS[i]
    weight += FINISH_WEIGHTS[i]
  })
  return weight ? sum / weight : null
}
