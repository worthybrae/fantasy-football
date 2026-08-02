import type { GameLogRow } from '../api'

interface WeeklyChartProps {
  gameLog: GameLogRow[]
}

// Fixed hue order, dark-mode steps from the dataviz skill's validated
// categorical palette (slots 1/2/3 -- the only three that clear the
// all-pairs CVD/normal-vision floors together, which is exactly the "3
// seasons max" this chart ever shows). Assigned by recency -- oldest of the
// (up to three) seasons shown gets slot 1, newest gets the last slot -- so
// every profile reads the same "older = cooler, newer = warmer" convention.
const SEASON_COLORS = ['#3987e5', '#d95926', '#199e70']

const BAR_WIDTH = 14
const BAR_GAP = 2
const SEASON_GAP = 18
const BAR_RADIUS = 3
const PLOT_HEIGHT = 140
const MARGIN = { top: 10, right: 12, bottom: 30, left: 38 }

function seasonTag(season: number): string {
  return `'${String(season).slice(-2)}`
}

// Heckbert's "nice numbers" step -- picks a round step (1/2/5 x 10^n) so
// axis ticks land on clean values instead of awkward fractions.
function niceNum(range: number, round: boolean): number {
  if (range <= 0) return 1
  const exponent = Math.floor(Math.log10(range))
  const fraction = range / 10 ** exponent
  let niceFraction: number
  if (round) {
    if (fraction < 1.5) niceFraction = 1
    else if (fraction < 3) niceFraction = 2
    else if (fraction < 7) niceFraction = 5
    else niceFraction = 10
  } else {
    if (fraction <= 1) niceFraction = 1
    else if (fraction <= 2) niceFraction = 2
    else if (fraction <= 5) niceFraction = 5
    else niceFraction = 10
  }
  return niceFraction * 10 ** exponent
}

function computeTicks(values: number[], targetCount = 5) {
  const dataMin = Math.min(0, ...values)
  const dataMax = Math.max(0, ...values, 1) // guard against an all-zero series
  const range = niceNum(dataMax - dataMin, false)
  const step = niceNum(range / (targetCount - 1), true)
  const niceMin = Math.floor(dataMin / step) * step
  const niceMax = Math.ceil(dataMax / step) * step
  const ticks: number[] = []
  for (let v = niceMin; v <= niceMax + step / 2; v += step) {
    ticks.push(Math.round(v * 100) / 100)
  }
  return { niceMin, niceMax, ticks }
}

// Draws a bar from the zero baseline to `value`, rounding only the end away
// from the baseline (top for a positive bar) -- square where it meets the
// axis, per the mark spec.
function barPath(x: number, width: number, top: number, bottom: number, positive: boolean): string {
  const r = Math.min(BAR_RADIUS, width / 2, Math.abs(bottom - top))
  if (positive) {
    return `M${x},${bottom} L${x},${top + r} Q${x},${top} ${x + r},${top} ` +
      `L${x + width - r},${top} Q${x + width},${top} ${x + width},${top + r} ` +
      `L${x + width},${bottom} Z`
  }
  return `M${x},${top} L${x},${bottom - r} Q${x},${bottom} ${x + r},${bottom} ` +
    `L${x + width - r},${bottom} Q${x + width},${bottom} ${x + width},${bottom - r} ` +
    `L${x + width},${top} Z`
}

export default function WeeklyChart({ gameLog }: WeeklyChartProps) {
  if (gameLog.length === 0) return null

  // Chronological order (chart reads left-to-right, oldest first); the API
  // returns newest-first, so sort a copy rather than mutating the prop.
  const chronological = [...gameLog].sort((a, b) => a.season - b.season || a.week - b.week)

  const seasonNumbers = [...new Set(chronological.map((g) => g.season))]
  // Defensive cap -- the pipeline only ever loads 3 seasons of weekly data,
  // so this shouldn't trigger, but a 4th season would collide with the
  // palette's validated 3-slot cap, so keep only the most recent 3.
  const shownSeasons = new Set(seasonNumbers.slice(-3))

  const groups = seasonNumbers
    .filter((s) => shownSeasons.has(s))
    .map((season) => chronological.filter((g) => g.season === season))

  const values = chronological
    .filter((g) => shownSeasons.has(g.season))
    .map((g) => g.ppr_points)
  const { niceMin, niceMax, ticks } = computeTicks(values)

  const plotBottom = MARGIN.top + PLOT_HEIGHT
  const yScale = (v: number) => plotBottom - ((v - niceMin) / (niceMax - niceMin)) * PLOT_HEIGHT
  const baselineY = yScale(0)

  let cursor = MARGIN.left
  const bars: { x: number; game: GameLogRow; color: string }[] = []
  const seasonMeta: { season: number; startX: number; endX: number; avg: number; color: string }[] = []

  groups.forEach((games, i) => {
    const color = SEASON_COLORS[i % SEASON_COLORS.length]
    const startX = cursor
    games.forEach((game) => {
      bars.push({ x: cursor, game, color })
      cursor += BAR_WIDTH + BAR_GAP
    })
    const endX = cursor - BAR_GAP
    const avg = games.reduce((sum, g) => sum + g.ppr_points, 0) / games.length
    seasonMeta.push({ season: games[0].season, startX, endX, avg, color })
    cursor += SEASON_GAP - BAR_GAP
  })

  const width = cursor - SEASON_GAP + MARGIN.right
  const height = plotBottom + MARGIN.bottom

  return (
    <div className="weekly-chart">
      <div className="weekly-chart-legend">
        {seasonMeta.map((s) => (
          <span className="weekly-chart-legend-item" key={s.season}>
            <span className="weekly-chart-swatch" style={{ background: s.color }} />
            {seasonTag(s.season)}
          </span>
        ))}
      </div>
      <div className="weekly-chart-scroll">
        <svg
          className="weekly-chart-svg"
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`Weekly PPR points across ${seasonMeta.length} season${seasonMeta.length === 1 ? '' : 's'}`}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line
                x1={MARGIN.left}
                x2={width - MARGIN.right}
                y1={yScale(t)}
                y2={yScale(t)}
                className={t === 0 ? 'weekly-chart-axis' : 'weekly-chart-gridline'}
              />
              <text x={MARGIN.left - 6} y={yScale(t)} className="weekly-chart-tick mono" textAnchor="end" dominantBaseline="middle">
                {t}
              </text>
            </g>
          ))}

          {seasonMeta.map((s) => (
            <line
              key={`avg-${s.season}`}
              x1={s.startX}
              x2={s.endX}
              y1={yScale(s.avg)}
              y2={yScale(s.avg)}
              stroke={s.color}
              strokeOpacity={0.6}
              strokeWidth={1.5}
              strokeDasharray="4 3"
            />
          ))}

          {bars.map(({ x, game, color }) => {
            const positive = game.ppr_points >= 0
            const top = positive ? yScale(game.ppr_points) : baselineY
            const bottom = positive ? baselineY : yScale(game.ppr_points)
            const title = `${seasonTag(game.season)} wk ${game.week} — ${game.ppr_points.toFixed(1)} pts vs ${game.opponent}`
            return (
              <path
                key={`${game.season}-${game.week}`}
                d={barPath(x, BAR_WIDTH, top, bottom, positive)}
                fill={color}
                className="weekly-bar"
                tabIndex={0}
                aria-label={title}
              >
                <title>{title}</title>
              </path>
            )
          })}

          {seasonMeta.map((s) => (
            <text
              key={`label-${s.season}`}
              x={(s.startX + s.endX) / 2}
              y={plotBottom + 18}
              className="weekly-chart-season-label"
              textAnchor="middle"
            >
              {seasonTag(s.season)}
            </text>
          ))}
        </svg>
      </div>
    </div>
  )
}
