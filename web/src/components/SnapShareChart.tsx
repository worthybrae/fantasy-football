import type { SeasonSummary } from '../api'

interface SnapShareChartProps {
  seasons: SeasonSummary[]
}

const BAR_COLOR = '#3987e5'
const SLOT_WIDTH = 48
const MIN_SLOTS = 6
const BAR_WIDTH = 22
const PLOT_HEIGHT = 100
const MARGIN = { top: 10, right: 10, bottom: 24, left: 34 }

function seasonTag(season: number): string {
  return `'${String(season).slice(-2)}`
}

// Average offensive snap share per season (0-100%). Seasons without snap
// data (pre-2012, or no snap-count match) are skipped rather than drawn as
// misleading zero bars.
export default function SnapShareChart({ seasons }: SnapShareChartProps) {
  const rows = [...seasons]
    .filter((s) => s.snap_share !== null)
    .sort((a, b) => a.season - b.season)
  if (rows.length === 0) return null

  const slots = Math.max(rows.length, MIN_SLOTS)
  const width = MARGIN.left + slots * SLOT_WIDTH + MARGIN.right
  const plotBottom = MARGIN.top + PLOT_HEIGHT
  const height = plotBottom + MARGIN.bottom
  const yScale = (pct: number) => plotBottom - (pct / 100) * PLOT_HEIGHT

  return (
    <div className="weekly-chart">
      <div className="weekly-chart-scroll">
        <svg
          className="weekly-chart-svg season-chart-wide"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`Average offensive snap share per season, for ${rows.length} season${rows.length === 1 ? '' : 's'}.`}
        >
          {[0, 25, 50, 75, 100].map((t) => (
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
            const pct = (s.snap_share as number) * 100
            const x = MARGIN.left + i * SLOT_WIDTH + (SLOT_WIDTH - BAR_WIDTH) / 2
            const top = yScale(pct)
            const title = `${seasonTag(s.season)} — ${pct.toFixed(0)}% snaps · ${s.games} gp`
            return (
              <g key={s.season}>
                <rect x={x} y={top} width={BAR_WIDTH} height={Math.max(plotBottom - top, 2)}
                  rx={3} fill={BAR_COLOR}>
                  <title>{title}</title>
                </rect>
                <text x={x + BAR_WIDTH / 2} y={plotBottom + 16} className="weekly-chart-season-label" textAnchor="middle">
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
