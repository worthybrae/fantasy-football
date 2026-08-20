import type { ProfileSummary } from '../../api'

// The keys and their order are `scoring/profile.py::_SUMMARY_STATS`, which is
// the thing this table has to agree with -- not a list invented here. What
// changed is the shape, not the numbers: this used to be a grid of nine tiles
// above the fold, and the card now leads with the verdict instead, so the
// usage they carried is one dim line under the seasons they came from.
//
// THERE IS NO `K` ROW AND THAT IS DELIBERATE. `_SUMMARY_STATS` carries the
// eleven skill columns only, so `w_stats` has no kicking key to read even in a
// league that prices kicking -- a K entry here would render six zeros. The
// lookup falls through to `[]` and the line degrades to the points alone,
// which is the whole of what the payload can say about a kicker's usage.
const STATS: Record<string, [string, string][]> = {
  QB: [['cmp', 'completions'], ['att', 'attempts'], ['pass yds', 'pass_yards'],
    ['pass TD', 'pass_tds'], ['int', 'interceptions'], ['rush yds', 'rush_yards']],
  RB: [['car', 'carries'], ['rush yds', 'rush_yards'], ['tgt', 'targets'],
    ['rec', 'receptions'], ['rec yds', 'rec_yards'], ['TD', 'tds']],
  WR: [['tgt', 'targets'], ['rec', 'receptions'], ['rec yds', 'rec_yards'],
    ['TD', 'tds'], ['car', 'carries'], ['rush yds', 'rush_yards']],
  TE: [['tgt', 'targets'], ['rec', 'receptions'], ['rec yds', 'rec_yards'],
    ['TD', 'tds'], ['car', 'carries'], ['rush yds', 'rush_yards']],
}

// What the points were made of, per game, recency-weighted across his
// seasons -- the line that turns "21.6 a game" into "fourteen carries and
// five targets a game". `w_ppg` and `w_stats` are the same weighted average
// (scoring/profile.py's `career_summary`), so they belong on one line rather
// than in two places that could be read as different periods.
export default function UsageLine({ summary, position }: {
  summary: ProfileSummary
  position: string
}) {
  const parts = (STATS[position] ?? [])
    .filter(([, key]) => key in summary.w_stats)
    .map(([label, key]) => `${summary.w_stats[key].toFixed(1)} ${label}`)
  if (summary.w_ppg === null && parts.length === 0) return null
  return (
    <p className="mono pp-usage">
      <span className="pp-cap">Weighted per game</span>
      {summary.w_ppg !== null && <> {summary.w_ppg.toFixed(1)} pts</>}
      {parts.length > 0 && <> · {parts.join(' · ')}</>}
      {summary.proj_delta !== null && (
        <> · ESPN projects{' '}
          <span className={summary.proj_delta < 0 ? 'is-bad' : 'is-good'}>
            {summary.proj_delta > 0 ? '+' : summary.proj_delta < 0 ? '−' : ''}
            {Math.abs(summary.proj_delta).toFixed(1)}
          </span>
          {' '}on that
        </>
      )}
    </p>
  )
}
