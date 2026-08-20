import type { GameRow } from './payload'
import { posHueClass } from './payload'

// The snap line's own coordinate space. Width is arbitrary (the svg
// stretches to the column); height matches the bar area in CSS pixels so
// the two layers agree about where "the top" is.
const VIEW_W = 680
const VIEW_H = 150
// The line never touches the ceiling: a 100% snap week would otherwise sit
// exactly on the top edge and read as clipped rather than as maximal.
const LINE_HEADROOM = 0.88

// Last season, week by week: what he scored, who he played, and how much of
// the offence he was on the field for. The season average a card usually
// shows hides both halves of this -- "0.66 of the snaps" is a different
// claim from a role that started at 0.50 and finished at 0.86, and the
// weekly bars are where a boom-or-bust reputation is either earned or
// disproved.
export default function WeekByWeek({ games, position }: {
  games: GameRow[]
  position: string
}) {
  if (games.length === 0) return null
  const season = Math.max(...games.map((g) => g.season))
  const weeks = games.filter((g) => g.season === season).sort((a, b) => a.week - b.week)
  if (weeks.length === 0) return null

  const played = weeks.filter((g) => !g.dnp)
  const points = played.map((g) => g.ppr_points)
  const avg = points.length ? points.reduce((a, b) => a + b, 0) / points.length : 0
  // His own week-to-week spread is the unit a big week is measured in, not
  // a fixed number of points: "seven above average" means something
  // different for a 21-ppg back and an 8-ppg tight end, and this card is
  // shown for both.
  const variance = points.length > 1
    ? points.reduce((a, b) => a + (b - avg) ** 2, 0) / (points.length - 1)
    : 0
  const std = Math.sqrt(variance)
  const maxPts = Math.max(...points, 1)

  const snapWeeks = weeks.filter((g) => g.snap_pct !== null)
  const snapLo = snapWeeks.length ? Math.min(...snapWeeks.map((g) => g.snap_pct as number)) : null
  const snapHi = snapWeeks.length ? Math.max(...snapWeeks.map((g) => g.snap_pct as number)) : null

  // One polyline per unbroken run of weeks with a snap share. A single
  // polyline would draw a straight line across a bye or a missed game,
  // which is a claim about weeks he was not on the field for.
  const step = VIEW_W / weeks.length
  const segments: string[] = []
  let run: string[] = []
  weeks.forEach((g, i) => {
    if (g.snap_pct === null) {
      if (run.length > 1) segments.push(run.join(' '))
      run = []
      return
    }
    const x = (i + 0.5) * step
    const y = VIEW_H - g.snap_pct * VIEW_H * LINE_HEADROOM
    run.push(`${x.toFixed(1)},${y.toFixed(1)}`)
  })
  if (run.length > 1) segments.push(run.join(' '))

  return (
    <div className={`pp-weeks ${posHueClass(position)}`}>
      <div className="pp-weeks-legend">
        <span className="pp-legend-item">
          <span className="pp-legend-bar" /> points
        </span>
        {snapWeeks.length > 0 && (
          <span className="pp-legend-item">
            <span className="pp-legend-line" /> snap share
          </span>
        )}
        <span className="pp-weeks-spacer" />
        {snapLo !== null && snapHi !== null && (
          <span className="mono pp-dim">
            snaps {Math.round(snapLo * 100)}–{Math.round(snapHi * 100)}%
          </span>
        )}
      </div>

      <div
        className="pp-weeks-plot"
        role="img"
        aria-label={`${season} points per week as bars, with offensive snap share as a line over the same weeks. ${
          played.length} games played at ${avg.toFixed(1)} points a game.`}
      >
        <div className="pp-weeks-bars">
          {weeks.map((g) => {
            const tone = g.dnp ? 'dnp'
              : std > 0 && g.ppr_points >= avg + std ? 'good'
                : std > 0 && g.ppr_points <= avg - std ? 'bad'
                  : 'mid'
            // 85% of the column, not 100: the points label sits above the
            // bar in the same flex column, and a tallest-week bar at full
            // height would push its own label out of the plot.
            const height = g.dnp ? 0 : Math.max(2, (g.ppr_points / maxPts) * 85)
            return (
              <div
                className="pp-week"
                key={g.week}
                title={g.dnp
                  ? `Week ${g.week}: did not play`
                  : `Week ${g.week} ${g.opponent ?? ''} — ${g.ppr_points.toFixed(1)} pts${
                    g.snap_pct !== null ? `, ${Math.round(g.snap_pct * 100)}% of snaps` : ''}\n${g.stat_line}`}
              >
                <div className={`pp-week-val mono is-${tone}`}>
                  {g.dnp ? '' : Math.round(g.ppr_points)}
                </div>
                {g.dnp
                  ? <div className="pp-week-dnp" />
                  : <div className={`pp-week-bar is-${tone}`} style={{ height: `${height}%` }} />}
              </div>
            )
          })}
        </div>
        {segments.length > 0 && (
          <svg
            className="pp-weeks-line"
            viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
            preserveAspectRatio="none"
            aria-hidden="true"
          >
            {segments.map((pts) => (
              <polyline key={pts} points={pts} vectorEffect="non-scaling-stroke" />
            ))}
          </svg>
        )}
      </div>

      <div className="pp-weeks-opps">
        {weeks.map((g) => (
          <div className="mono pp-week-opp" key={g.week}>{g.opponent ?? '—'}</div>
        ))}
      </div>
      <p className="pp-note">
        {season} regular season, {played.length} game{played.length === 1 ? '' : 's'} played,
        {' '}{avg.toFixed(1)} per game. Green and red are the weeks more than
        his own spread (±{std.toFixed(1)}) away from that average
        {weeks.length > played.length && ', hatched weeks are ones he missed'}.
      </p>
    </div>
  )
}
