import type { LineQualityData } from './payload'
import { ordinal, rankTone } from './payload'

// Mean seasons in the league, drawn against a ten-year line. A fixed scale
// rather than a percentile because the payload serves the RAW component and
// not its rank among the 32 -- inventing a distribution to normalise
// against would be drawing a comparison nobody computed.
const EXPERIENCE_SCALE = 10

// The line the player's own production runs through. The composite alone is
// an opaque 0-100, so the four parts scoring/oline.py built it from are
// drawn beside it: a line that is 21st because its starters keep getting
// hurt is a different bet from one that is 21st because it was rebuilt in
// March, and only the parts tell them apart.
export default function LineQuality({ oline }: { oline: LineQualityData }) {
  const tone = rankTone(oline.rank, oline.teams)
  // Only the three share-of-one measures are comparable to each other;
  // `experience` is in seasons and is deliberately left out of the "what
  // drags it" sentence rather than scaled into looking comparable.
  const shares = [
    { label: 'continuity', value: oline.continuity },
    { label: 'availability', value: oline.availability },
    { label: 'returning starters', value: oline.returning },
  ].filter((s): s is { label: string; value: number } => s.value !== null)
  const weakest = shares.length
    ? shares.reduce((a, b) => (b.value < a.value ? b : a))
    : null

  const bars: { label: string; pct: number; value: string }[] = []
  if (oline.continuity !== null) {
    bars.push({ label: 'continuity', pct: oline.continuity * 100, value: oline.continuity.toFixed(2) })
  }
  if (oline.availability !== null) {
    bars.push({ label: 'availability', pct: oline.availability * 100, value: oline.availability.toFixed(2) })
  }
  if (oline.returning !== null) {
    // The raw share is what the payload carries; "3 of 5" is the same fact
    // in the unit the depth chart is written in.
    bars.push({ label: 'returning', pct: oline.returning * 100, value: `${Math.round(oline.returning * 5)} of 5` })
  }
  if (oline.experience !== null) {
    bars.push({
      label: 'experience',
      pct: Math.min(100, (oline.experience / EXPERIENCE_SCALE) * 100),
      value: `${oline.experience.toFixed(1)} yr`,
    })
  }

  return (
    <div className="pp-oline">
      <div className="pp-section-meta mono">
        <span className={`pp-strong is-${tone}`}>
          {ordinal(oline.rank)} of {oline.teams}
        </span>
        {oline.line_quality !== null && <> · {oline.line_quality.toFixed(1)} / 100</>}
      </div>
      <div className="pp-oline-rows">
        {bars.map((b) => (
          <div className="pp-oline-row" key={b.label}>
            <div className="pp-oline-label">{b.label}</div>
            <div className="pp-oline-track">
              <div className="pp-oline-fill" style={{ width: `${Math.max(0, Math.min(100, b.pct))}%` }} />
            </div>
            <div className="mono pp-oline-value">{b.value}</div>
          </div>
        ))}
      </div>
      <p className="pp-note">
        {oline.team}&apos;s {oline.season} line, weighted continuity 0.35,
        availability 0.30, returning 0.20, experience 0.15 and normalised
        across the 32 teams.
        {weakest && ` What drags it is ${weakest.label}, at ${weakest.value.toFixed(2)} of 1.`}
        {' '}The bars are the raw parts, not their league ranks — the rank
        above is the only comparison the payload carries.
      </p>
    </div>
  )
}
