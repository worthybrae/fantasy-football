import type { LineQualityData } from './payload'
import PopCard, { PopRows, type PopRow } from './PopCard'
import { ordinal } from './payload'

// The line the player's own production runs through, led by the only number
// on it that is a comparison: 20th of 32. The composite it is ranked on is an
// opaque 0-100 and is left off -- a reader who wants to know why the line is
// 20th is served by the two parts under the rank, not by the score they were
// averaged into.
//
// Continuity and returning, of the four parts `scoring/oline.py` builds, for
// the same reason: they are the two that say whether this is the same line as
// last year. Both are shares of one, printed raw -- the card used to spell
// returning as "3 of 5", which reads better alone but not stacked over a
// continuity of 0.81, where the eye takes the two as one unit.
//
// The rank is plain rather than toned. Every other colour in this popup is
// carrying a season of the player's own; his team's line is the context those
// seasons happened in, and a red 24th beside them would compete with them.
export default function LineQuality({ oline }: { oline: LineQualityData }) {
  const rows: PopRow[] = []
  if (oline.continuity !== null) {
    rows.push({ label: 'Continuity', value: oline.continuity.toFixed(2) })
  }
  if (oline.returning !== null) {
    rows.push({ label: 'Returning', value: oline.returning.toFixed(1) })
  }

  return (
    <PopCard title="Blocking" note={oline.team}>
      <div className="pp-pop-lead">
        <span className="mono pp-pop-lead-value">{ordinal(oline.rank)}</span>
        <span className="mono pp-pop-lead-of">of {oline.teams}</span>
      </div>
      {rows.length > 0 && <PopRows rows={rows} />}
    </PopCard>
  )
}
