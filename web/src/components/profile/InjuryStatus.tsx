import { ageLabel } from '../../api'
import type { PlayerStatus } from './payload'

// Sleeper's designations, worst first. Anything not in this list still
// renders -- an unknown designation is shown as it came rather than
// swallowed -- it just gets the neutral tone.
const SEVERE = ['out', 'ir', 'pup', 'nfi', 'susp', 'doubtful']
const WATCH = ['questionable', 'probable', 'dtd', 'day-to-day']

function tone(status: string): string {
  const s = status.toLowerCase()
  if (SEVERE.some((k) => s.includes(k))) return 'is-bad'
  if (WATCH.some((k) => s.includes(k))) return 'is-accent'
  return ''
}

// The injury line, beside the name because that is where it changes a pick:
// "Questionable — hamstring" is the one fact on this card that can make
// every number above it irrelevant.
//
// NOTHING HERE EVER SAYS HEALTHY. A null `injury_status` means Sleeper
// published no designation, which is not the same claim as a clean bill of
// health, so this renders nothing at all rather than a reassuring pill
// nobody can source.
export default function InjuryStatus({ status }: { status: PlayerStatus }) {
  if (status.injury_status === null) return null
  const detail = [status.injury_body_part, status.injury_notes]
    .filter((s): s is string => typeof s === 'string' && s.trim() !== '')
    .join(' — ')
  return (
    <span className="pp-injury">
      <span className={`pp-injury-pill ${tone(status.injury_status)}`}>
        {status.injury_status}
      </span>
      {detail !== '' && <span className="pp-injury-detail">{detail}</span>}
      {status.news_updated !== null && (
        <span className="pp-injury-age mono">updated {ageLabel(status.news_updated)}</span>
      )}
    </span>
  )
}
