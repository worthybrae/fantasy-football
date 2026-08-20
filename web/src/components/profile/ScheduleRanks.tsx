import type { ScheduleRankWeek } from './payload'
import { ordinal, rankTone } from './payload'

// The season's matchups as league ranks, not percentiles. "20th of 32" is a
// fact you can act on at a glance; "41st percentile" is one you have to
// convert first, and the owner asked for it in exactly those words.
//
// THE DIRECTION IS INVERTED FROM INTUITION AND THE LABELS SAY SO: rank 1 is
// the SOFTEST defence -- the one that gave up the most to this position last
// season -- because a high fantasy-points-allowed is a good matchup. Reading
// it as a difficulty rank turns the best week of the season into the worst.
export default function ScheduleRanks({ weeks }: { weeks: ScheduleRankWeek[] }) {
  const ranked = weeks.filter((w) => w.rank !== null)
  if (weeks.length === 0) return null

  const sorted = [...ranked].map((w) => w.rank as number).sort((a, b) => a - b)
  const median = sorted.length
    ? sorted.length % 2
      ? sorted[(sorted.length - 1) / 2]
      : Math.round((sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2)
    : null
  const of = ranked.length ? ranked[0].rank_n : null

  return (
    <div className="pp-sched">
      <div className="pp-section-meta mono">
        {median !== null && of !== null
          ? <>median matchup <span className={`pp-strong is-${rankTone(median, of)}`}>
            {ordinal(median)} of {of}</span> · 1 = softest</>
          : 'no matchup ranks for this position'}
      </div>
      <div className="pp-sched-weeks">
        {weeks.map((w) => {
          const bye = w.opponent === null
          return (
            <div className={`pp-sched-week${bye ? ' is-bye' : ''}`} key={w.week}>
              <div className="mono pp-sched-wk">{w.week}</div>
              <div className="mono pp-sched-opp">
                {bye ? 'BYE' : `${w.home === false ? '@' : ''}${w.opponent}`}
              </div>
              <div
                className={`mono pp-sched-rank is-${rankTone(w.rank, w.rank_n)}`}
                title={w.rank !== null && w.fpa_pg !== null
                  ? `${ordinal(w.rank)} softest of ${w.rank_n} — allowed ${w.fpa_pg.toFixed(1)} a game to this position last season`
                  : 'bye week'}
              >
                {w.rank ?? '—'}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
