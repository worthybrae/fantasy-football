// Positional finish: how good a season was, and how to colour it.
//
// Shared by the board's `FinishArc` and the hover panel behind it. They draw
// different shapes -- a bar per season, a row of dots per season -- and they
// must agree on what "RB12" means, or the small picture and the big one would
// tell a reader different things about the same year.

// What a 12-team league starts at each position, which is what makes a finish
// mean anything: RB24 is the last startable back and TE12 the last startable
// tight end, so the same number is a different season at each.
export const FINISH_STARTERS: Record<string, number> = {
  QB: 12, TE: 12, RB: 24, WR: 24, K: 12, DST: 12,
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
