// Snake pick arithmetic, one copy. ClockPanel and DraftRoom each carried
// their own -- the same two functions, byte for byte, with a comment in
// DraftRoom pointing at ClockPanel's as the precedent. That was tolerable
// while both copies only ever labelled a figure; it stopped being tolerable
// once the two answered the same question ("which pick is `gain_now`
// measured against?") in two places at once, because a divergence between
// them is then a wrong number on screen, not a style nit.

// Overall pick number (1-based) for `slot` (1-based) in round `round`
// (0-based), snake order: forward in even rounds, reversed in odd. Mirrors
// scoring/draft_sim.snake_slots exactly -- verified pick-by-pick against it
// (e.g. teams=4: round 1 gives slot 1 pick 8, slot 4 pick 5) rather than
// assumed from the task brief's formula, which turned out to already be
// correct.
export function pickNumberFor(round: number, slot: number, teams: number): number {
  return round * teams + (round % 2 === 0 ? slot : teams - slot + 1)
}

// `mySlot`'s first pick STRICTLY AFTER `fromPickNo`. Every slot picks
// exactly once per round, so the answer is always in the round `fromPickNo`
// falls in, or the very next one -- no need for the league's total round
// count, which nothing this room fetches carries either.
//
// Strictly after, not "at or after": both call sites pass the pick that is
// on the clock right now and mean "the one I have to wait for". The
// previous `>=` returned that same pick whenever it was already mine, so
// the room rendered "ranked by what you gain now vs. waiting until pick 4"
// while the user was picking at 4, and "This pick 4 / Next pick 4 / Gap 0
// picks". This is the display half of the same off-by-one
// scoring/draft_sim.survival carried (see its `on_the_clock` docstring):
// there too, "my pick is now" and "my pick is next" were indistinguishable
// and both answered "now".
export function nextPickFor(fromPickNo: number, mySlot: number, teams: number): number {
  const round = Math.floor((fromPickNo - 1) / teams)
  const thisRound = pickNumberFor(round, mySlot, teams)
  return thisRound > fromPickNo ? thisRound : pickNumberFor(round + 1, mySlot, teams)
}
