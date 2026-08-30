// Shared color logic for the draft room -- one copy so a "he lasts" number
// always reads the same color wherever it appears, rather than two views
// drifting into slightly different ramps over time.

// A continuous red -> amber -> green ramp keyed to `lasts_pct` (the measured
// chance a candidate is still there at your next turn),
// same color-mix technique as the deleted PlayerCard.tsx's sosTone (two-stop
// interpolation between the fixed --ok/--fail tokens) but extended to
// three stops through --accent -- this app's amber -- at the midpoint,
// since the brief calls for amber as its own readable tier, not just a
// blend implied by two endpoints. Low pct (gone before your next pick)
// reads --fail; high pct (he'll last, you can wait) reads --ok.
//
// Moved here from the deleted LiveDraft.tsx (where it was `riskTone`,
// keyed to `applied_pct`) so AvailableList and TargetCards both import the
// one copy instead of each carrying their own.
export function riskTone(pct: number): string {
  const t = Math.max(0, Math.min(100, pct))
  if (t >= 50) {
    const k = ((t - 50) / 50) * 100
    return `color-mix(in srgb, var(--ok) ${k}%, var(--accent) ${100 - k}%)`
  }
  const k = (t / 50) * 100
  return `color-mix(in srgb, var(--accent) ${k}%, var(--fail) ${100 - k}%)`
}

// -- the same number, read as three answers ---------------------------------
//
// `riskTone` above is a continuous ramp, which is right for a column of ~250
// cells the eye compares against each other. The simple room asks a smaller
// question of the same number -- "will he be there?" -- and answers it one
// row at a time, where a continuous ramp gives a reader nothing to hold: the
// difference between 61% and 67% is two shades of amber nobody can name.
//
// Three bands, and the cut points are the ones the owner set: 70 and up is a
// player you can plan around, 40 to 69 is a real coin toss, under 40 is a
// player who will most likely be gone. They are used for the chip's tint in
// both surfaces that print the percentage, so "amber" means the same thing on
// the card as it does in the list.
export type LastsBand = 'high' | 'mid' | 'low'

export function lastsBand(pct: number): LastsBand {
  if (pct >= 70) return 'high'
  return pct >= 40 ? 'mid' : 'low'
}
