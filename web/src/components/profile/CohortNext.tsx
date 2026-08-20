import type { Cohort } from './payload'
import { fmtSigned, ordinal } from './payload'

const BIN = 2 // ppg per bucket -- fine enough to show a shape, coarse
// enough that a 19-season cohort is not 19 bars of height one.

const WORDS = ['no', 'one', 'two', 'three', 'four', 'five']

function years(n: number): string {
  return n === 1 ? 'a year' : `${WORDS[n] ?? n} years`
}

// What seasons like this one became. Not a projection and not an average of
// one: the distribution of what actually happened next to a tight cohort of
// comparable seasons, drawn as CHANGE in points per game so zero is the
// interesting line rather than a number you have to subtract in your head.
//
// The cohort's recipe is stated, not asserted, because a cohort is only as
// honest as its band: the payload serves the ppg band, the experience band
// and the minimum games that produced it, so the caption reads them back
// instead of describing a recipe someone typed once and nobody re-checked
// against the query.
export default function CohortNext({ cohort }: { cohort: Cohort }) {
  const changes = cohort.players.map((p) => p.change)
  if (changes.length === 0) {
    return (
      <p className="pp-note">
        No comparable season matched the band — no other {cohort.position} has
        put up {cohort.ppg.toFixed(1)} per game this early in a career and
        then played another season.
      </p>
    )
  }

  const lo = Math.min(Math.floor(Math.min(...changes) / BIN) * BIN, 0)
  const hi = Math.max(Math.ceil(Math.max(...changes) / BIN) * BIN, 0)
  const span = Math.max(hi - lo, BIN)
  const bins = new Array(Math.round(span / BIN)).fill(0) as number[]
  for (const c of changes) {
    const i = Math.min(bins.length - 1, Math.max(0, Math.floor((c - lo) / BIN)))
    bins[i] += 1
  }
  const tallest = Math.max(...bins, 1)
  const at = (v: number) => `${((v - lo) / span) * 100}%`

  return (
    <div className="pp-cohort">
      <p className="pp-sub">
        {cohort.n} {cohort.position} season{cohort.n === 1 ? '' : 's'} within
        {' '}±{cohort.ppg_band} ppg of his {cohort.ppg.toFixed(1)} average
        {cohort.exp_band !== null && cohort.nfl_season !== null
          && <> and within {years(cohort.exp_band)} of his {ordinal(cohort.nfl_season)} year</>}
        , each with at least {cohort.min_games} games. Change in points per
        game the season after.
      </p>

      <div
        className="pp-cohort-plot"
        role="img"
        aria-label={`Distribution of the change in points per game the following season across ${
          cohort.n} comparable seasons, in ${BIN}-point buckets from ${lo} to ${hi}, with zero marked.`}
      >
        <div className="pp-cohort-bars">
          {bins.map((count, i) => (
            <div className="pp-cohort-slot" key={lo + i * BIN}>
              <div
                className="pp-cohort-bar"
                style={{ height: `${(count / tallest) * 100}%` }}
                title={`${count} season${count === 1 ? '' : 's'} between ${
                  fmtSigned(lo + i * BIN, 0)} and ${fmtSigned(lo + (i + 1) * BIN, 0)} ppg`}
              />
            </div>
          ))}
        </div>
        <div className="pp-cohort-zero" style={{ left: at(0) }} />
        {cohort.median_change !== null && (
          <div
            className="pp-cohort-median"
            style={{ left: at(cohort.median_change) }}
            title={`median ${fmtSigned(cohort.median_change, 1)} ppg`}
          />
        )}
      </div>
      <div className="pp-cohort-axis">
        <span className="mono is-bad">{fmtSigned(lo, 0)}</span>
        <span className="mono pp-dim">no change</span>
        <span className="mono is-good">{fmtSigned(hi, 0)}</span>
      </div>

      <div className="pp-figures">
        <div>
          <div className="pp-cap">Seasons</div>
          <div className="mono pp-figure">{cohort.n}</div>
        </div>
        <div>
          <div className="pp-cap">Median change</div>
          <div className={`mono pp-figure${
            cohort.median_change !== null && cohort.median_change < 0 ? ' is-bad'
              : cohort.median_change !== null && cohort.median_change > 0 ? ' is-good' : ''}`}
          >
            {fmtSigned(cohort.median_change, 1)}
          </div>
        </div>
        <div>
          <div className="pp-cap">Declined</div>
          <div className={`mono pp-figure${cohort.declined > cohort.n / 2 ? ' is-bad' : ''}`}>
            {cohort.declined} / {cohort.n}
          </div>
        </div>
      </div>

      <p className="pp-note">
        {cohort.declined} of {cohort.n} came back worse
        {cohort.improved > 0 && `, ${cohort.improved} better`}.
        {cohort.n < 10 && ' A cohort this thin is closer to an anecdote than'
          + ' a distribution — read the spread, not the median.'}
      </p>
    </div>
  )
}
