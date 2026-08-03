import type { SeasonSummary } from '../api'

interface SeasonRangeChartProps {
  seasons: SeasonSummary[]
  position: string
}

// Single-series magnitude chart: one categorical hue (slot 1 of the
// validated palette, same as the weekly chart's oldest recent season).
const POINT_COLOR = '#3987e5'

const SLOT_WIDTH = 56
// Padding the slot count keeps a short career (rookie with one season)
// from scaling its few marks and labels up to poster size when the svg
// stretches to the full drawer width.
const MIN_SLOTS = 6
const PLOT_HEIGHT = 200
// Extra top room for the positional-finish labels above the whiskers.
const MARGIN = { top: 28, right: 12, bottom: 26, left: 38 }
const POINT_RADIUS = 5
const WHISKER_CAP = 10

function seasonTag(season: number): string {
  return `'${String(season).slice(-2)}`
}

// Avg PPR per game played, with ±1 std dev whiskers, per season. DNP weeks
// never reach these numbers: the backend computes both stats from games
// actually played.
export default function SeasonRangeChart({ seasons, position }: SeasonRangeChartProps) {
  const rows = [...seasons]
    .filter((s) => s.ppg !== null)
    .sort((a, b) => a.season - b.season)
  if (rows.length === 0) return null

  const highs = rows.map((s) => s.ppg + (s.ppg_std ?? 0))
  const lows = rows.map((s) => Math.min(0, s.ppg - (s.ppg_std ?? 0)))
  const yMax = Math.ceil(Math.max(...highs, 1) / 5) * 5
  const yMin = Math.floor(Math.min(...lows, 0) / 5) * 5

  const width = MARGIN.left + Math.max(rows.length, MIN_SLOTS) * SLOT_WIDTH + MARGIN.right
  const plotBottom = MARGIN.top + PLOT_HEIGHT
  const height = plotBottom + MARGIN.bottom
  const yScale = (v: number) =>
    plotBottom - ((v - yMin) / (yMax - yMin)) * PLOT_HEIGHT

  const ticks: number[] = []
  for (let t = yMin; t <= yMax; t += 5) ticks.push(t)

  return (
    <div className="weekly-chart">
      <div className="weekly-chart-scroll">
        <svg
          className="weekly-chart-svg season-chart-wide"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`Average PPR points per game played with one standard deviation of week-to-week volatility, labeled with the ${position} positional finish by total points, for ${rows.length} season${rows.length === 1 ? '' : 's'}.`}
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

          {rows.map((s, i) => {
            const x = MARGIN.left + i * SLOT_WIDTH + SLOT_WIDTH / 2
            const std = s.ppg_std
            const finish = `${position}${s.pos_finish}`
            const title = std !== null
              ? `${seasonTag(s.season)} — ${finish} · ${s.ppg.toFixed(1)} avg ± ${std.toFixed(1)} · ${s.games} gp`
              : `${seasonTag(s.season)} — ${finish} · ${s.ppg.toFixed(1)} avg · ${s.games} gp`
            const labelY = yScale(s.ppg + (std ?? 0)) - 10
            return (
              <g key={s.season}>
                {std !== null && (
                  <>
                    <line x1={x} x2={x} y1={yScale(s.ppg + std)} y2={yScale(s.ppg - std)}
                      stroke={POINT_COLOR} strokeOpacity={0.5} strokeWidth={2} />
                    <line x1={x - WHISKER_CAP / 2} x2={x + WHISKER_CAP / 2}
                      y1={yScale(s.ppg + std)} y2={yScale(s.ppg + std)}
                      stroke={POINT_COLOR} strokeOpacity={0.5} strokeWidth={2} />
                    <line x1={x - WHISKER_CAP / 2} x2={x + WHISKER_CAP / 2}
                      y1={yScale(s.ppg - std)} y2={yScale(s.ppg - std)}
                      stroke={POINT_COLOR} strokeOpacity={0.5} strokeWidth={2} />
                  </>
                )}
                {/* invisible widened hit target so the tooltip is easy to reach */}
                <rect x={x - SLOT_WIDTH / 2} y={MARGIN.top} width={SLOT_WIDTH}
                  height={PLOT_HEIGHT} fill="transparent">
                  <title>{title}</title>
                </rect>
                <circle cx={x} cy={yScale(s.ppg)} r={POINT_RADIUS} fill={POINT_COLOR}>
                  <title>{title}</title>
                </circle>
                <text x={x} y={labelY} className="weekly-chart-tick mono" textAnchor="middle">
                  {finish}
                </text>
                <text x={x} y={plotBottom + 16} className="weekly-chart-season-label" textAnchor="middle">
                  {seasonTag(s.season)}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}
