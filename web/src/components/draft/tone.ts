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
