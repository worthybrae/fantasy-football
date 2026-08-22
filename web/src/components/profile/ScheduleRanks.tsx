import type { ScheduleRankWeek } from './payload'
import PopCard from './PopCard'
import { ordinal } from './payload'

// The season as eighteen bars: one per week, tall where the defence in front
// of him gave up the most to his position last year.
//
// THE DIRECTION IS INVERTED FROM INTUITION AND THE COPY SAYS SO: a tall green
// week is a SOFT one, and the card's note calls the season figure "softest"
// for the same reason. Reading it as a difficulty rank turns the best week of
// the season into the worst.
//
// `pct` rather than `rank`, because the strip is a picture: a rank is a place
// in a list and would draw 1st and 32nd the same distance apart as 15th and
// 16th. The exact rank stays on every bar's tooltip, where a reader who wants
// the number can still have it.

// The five-step ramp, cut on the softness percentile, so one colour means one
// thing in every picture in this popup -- these cut points are the strip's
// own, since a percentile is not a finish and has no starters to be graded
// against.
//
// The names are the ramp's OTHER spelling, not the season panels'. Three of
// its five steps answer to two class names apiece -- `is-good`/`is-elite`,
// `is-mid`/`is-starter`, `is-bad`/`is-out` -- aliased onto the same rule in
// App.css (see the ".ctip-col-bar" ramp), which is why the two middle steps
// here read the same as the panels'. A sixth step added in one spelling only
// would come out unstyled, so add it to both.
const TONES: [number, string][] = [
  [75, 'is-good'], [60, 'is-strong'], [40, 'is-mid'], [25, 'is-fringe'],
]

function softnessTone(pct: number): string {
  return TONES.find(([floor]) => pct >= floor)?.[1] ?? 'is-bad'
}

/** The season figure, as a place among the league rather than as the
 *  percentile the payload carries. `sos_pct` is a percentile RANK over the
 *  32 teams at this position (`profile.py` takes pandas' `rank(pct=True)`),
 *  so the place it came from is exact rather than reconstructed -- 84.4 of
 *  32 teams is the 27th hardest, which is the 6th softest. Ties come back
 *  averaged and round to the nearer of the two places they share. */
function softestRank(pct: number, teams: number): number {
  const ascending = Math.round((pct / 100) * teams)
  return Math.min(teams, Math.max(1, teams - ascending + 1))
}

export default function ScheduleRanks({ weeks, sosPct }: {
  weeks: ScheduleRankWeek[]
  sosPct: number | null
}) {
  // Nothing to draw is nothing to say: a position the fantasy-points-allowed
  // table has no column for gets no card, not eighteen grey stubs.
  if (!weeks.some((w) => w.pct !== null)) return null
  const teams = weeks.find((w) => w.rank_n !== null)?.rank_n ?? null
  const note = sosPct !== null && teams !== null
    ? `${ordinal(softestRank(sosPct, teams))} softest of ${teams}`
    : undefined

  return (
    // Wider than its third of the row: "6th softest of 32" is the whole
    // point of the card and does not survive being wrapped or cut.
    <PopCard title="Schedule" note={note} className="is-wide">
      <div className="pp-pop-strip">
        {weeks.map((w) => {
          const opponent = w.opponent === null
            ? 'bye'
            : `wk ${w.week} ${w.home === false ? 'at ' : 'vs '}${w.opponent}`
          if (w.pct === null) {
            // A bye, or a week whose opponent this position is not ranked
            // against: the chart's own mark for "nothing happened here",
            // drawn on the baseline rather than left as a hole in the row.
            return <div className="ctip-col-none" key={w.week} title={opponent} />
          }
          return (
            <div
              key={w.week}
              className={`ctip-col-bar ${softnessTone(w.pct)}`}
              style={{ height: `${w.pct}%` }}
              title={w.rank !== null && w.rank_n !== null && w.fpa_pg !== null
                ? `${opponent} — ${ordinal(w.rank)} softest of ${w.rank_n}, `
                  + `allowed ${w.fpa_pg.toFixed(1)} a game to this position last season`
                : opponent}
            />
          )
        })}
      </div>
    </PopCard>
  )
}
