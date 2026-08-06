import type { ScheduleWeek } from '../api'

interface ScheduleCalendarProps {
  weeks: ScheduleWeek[]
  position: string
  bye: number | null
}

// Percentile -> tint class. High percentile = the opponent allowed a lot of
// PPR points to this position last season = a soft matchup. Same ok/fail
// tint families as the game log's boom/bust shading, so green/red reads
// identically across the app.
function cellClass(w: ScheduleWeek, bye: number | null): string {
  if (w.opponent === null) return w.week === bye ? 'sched-cell sched-bye' : 'sched-cell sched-empty'
  if (w.pct === null) return 'sched-cell'
  if (w.pct >= 80) return 'sched-cell pts-hot-2'
  if (w.pct >= 60) return 'sched-cell pts-hot-1'
  if (w.pct <= 20) return 'sched-cell pts-cold-2'
  if (w.pct <= 40) return 'sched-cell pts-cold-1'
  return 'sched-cell'
}

export default function ScheduleCalendar({ weeks, position, bye }: ScheduleCalendarProps) {
  if (weeks.length === 0) return null
  return (
    <div className="sched-calendar">
      <div className="sched-grid">
        {weeks.map((w) => {
          const title = w.opponent === null
            ? `Week ${w.week} — ${w.week === bye ? 'bye' : 'no game'}`
            : `Week ${w.week} — ${w.home ? 'vs' : 'at'} ${w.opponent}` +
              (w.fpa_pg !== null ? ` · allowed ${w.fpa_pg} PPR/g to ${position}s (${w.pct}th pct)` : '')
          return (
            <div className={cellClass(w, bye)} key={w.week} title={title}>
              <span className="sched-week mono">{w.week}</span>
              <span className="sched-opp mono">
                {w.opponent === null ? (w.week === bye ? 'BYE' : '—') : `${w.home ? '' : '@'}${w.opponent}`}
              </span>
            </div>
          )
        })}
      </div>
      <p className="card-footnote">
        <span className="sched-key pts-hot-2" /> softer for {position}s ·{' '}
        <span className="sched-key pts-cold-2" /> tougher · last season's points allowed
      </p>
    </div>
  )
}
