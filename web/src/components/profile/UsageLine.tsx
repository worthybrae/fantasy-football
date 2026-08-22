import type { ProfileSummary } from '../../api'
import PopCard, { PopRows, type PopRow } from './PopCard'

// Three lines per position, off `scoring/profile.py::_SUMMARY_STATS` -- the
// keys are that list's, not one invented here. Three because volume is what
// this card is for: the points are already drawn season by season in the Per
// game panel above, and what a reader cannot see there is whether they came
// from carries or from targets.
//
// Yards are the sum of both halves for the same reason: a back who gains 73
// on the ground and 31 through the air had a 105-yard game, and splitting
// that across two of the three rows would spend the card on one stat.
//
// THERE IS NO `K` OR `DST` ENTRY AND THAT IS DELIBERATE. `_SUMMARY_STATS`
// carries the eleven skill columns only, so `w_stats` has no kicking key to
// read even in a league that prices kicking -- a K entry here would render
// three zeros. The lookup falls through and the card renders nothing, which
// is the whole of what the payload can say about a kicker's usage.
const USAGE: Record<string, [string, string[]][]> = {
  QB: [['Attempts / g', ['attempts']], ['Pass yds / g', ['pass_yards']],
    ['Rush yds / g', ['rush_yards']]],
  RB: [['Carries / g', ['carries']], ['Targets / g', ['targets']],
    ['Yards / g', ['rush_yards', 'rec_yards']]],
  WR: [['Targets / g', ['targets']], ['Catches / g', ['receptions']],
    ['Yards / g', ['rec_yards', 'rush_yards']]],
  TE: [['Targets / g', ['targets']], ['Catches / g', ['receptions']],
    ['Yards / g', ['rec_yards', 'rush_yards']]],
}

// What the points were made of, per game, recency-weighted across his
// seasons. `w_ppg` and `w_stats` are the same weighted average
// (scoring/profile.py's `career_summary`), which is what the note says in one
// word: these are not last season's numbers, they are his.
export default function UsageLine({ summary, position }: {
  summary: ProfileSummary
  position: string
}) {
  const rows: PopRow[] = (USAGE[position] ?? [])
    .filter(([, keys]) => keys.some((k) => k in summary.w_stats))
    .map(([label, keys]) => ({
      label,
      value: keys.reduce((sum, k) => sum + (summary.w_stats[k] ?? 0), 0).toFixed(1),
      tone: 'strong' as const,
    }))
  // A player with no games behind him has an empty `w_stats`, and a card of
  // dashes would be three ways of saying he is a rookie. The header already
  // says it once.
  if (rows.length === 0) return null

  return (
    <PopCard title="Usage" note="weighted">
      <PopRows rows={rows} />
    </PopCard>
  )
}
