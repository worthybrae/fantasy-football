import { Chart } from '../draft/Chart'
import { weekCols } from '../draft/panels'
import { barTone } from '../draft/weeks'
import PopCard from './PopCard'
import type { GameRow, SeasonRow } from './payload'

// The season as a shape, and under it the games it is made of.
//
// The chart is the board's own picture drawn by the board's own builder
// (weekCols): one column a week, the opponent under it, and a bye or a
// missed game as a mark on the baseline rather than a bar of no height --
// "did not play" and "played and scored nothing" are different claims.
//
// The log is the same weeks in words. Newest first, because form is what
// someone opening this mid-draft is reading for, and only weeks he PLAYED
// get a row: a missed week is already above as a baseline mark, and a row
// saying nothing happened would cost a reader a line to learn nothing. How
// many rows are visible is CSS's (`.pp-pop-log-body` caps it at six and
// scrolls the rest) -- the component hands over the whole season, so the
// scroll reaches week 1.
export default function WeekByWeek({ games, seasons, position }: {
  games: GameRow[]
  seasons: SeasonRow[]
  position: string
}) {
  if (games.length === 0) return null
  const season = Math.max(...games.map((g) => g.season))
  const log = games
    .filter((g) => g.season === season && !g.dnp)
    .sort((a, b) => b.week - a.week)
  // A season of nothing but DNPs draws no card at all rather than a chart of
  // baseline marks over an empty log.
  if (log.length === 0) return null

  // The season's own row, so the head states the average the rest of the
  // popup states -- a mean recomputed here off the same games would round
  // differently from the Per game panel two inches above it. The fallback is
  // for a game log whose season has no summary row, which is the only case
  // where there is no shared number to agree with.
  const summary = seasons.find((s) => s.season === season) ?? null
  const avg = summary === null
    ? log.reduce((total, g) => total + g.ppr_points, 0) / log.length
    : summary.ppg

  return (
    // The same card as the four season panels above and the dense ones
    // below (PopCard), one row wide: it is the last of those seasons in
    // detail, not a different kind of object.
    <PopCard
      title={`${season} by week`}
      note={(
        <>
          {`avg ${avg.toFixed(1)}`}
          {summary !== null && ` · finished ${position}${summary.pos_finish}`}
        </>
      )}
      className="pp-pop-panel pp-pop-weeks"
    >
      <Chart cols={weekCols(games, season, position)} />

      {/* Same weeks, twice: the hairline is where the picture stops and the
          words for it start. */}
      <div className="pp-pop-rule" />

      <div className="pp-pop-log">
        <div className="pp-pop-log-head">
          <span className="pp-pop-log-wk">Wk</span>
          <span className="pp-pop-log-opp">Opp</span>
          <span className="pp-pop-log-line">Line</span>
          <span className="pp-pop-log-pts">Pts</span>
        </div>
        <div className="pp-pop-log-body">
          {log.map((g) => (
            <div className="pp-pop-log-row" key={g.week}>
              <span className="mono pp-pop-log-wk">{g.week}</span>
              <span className="mono pp-pop-log-opp">{g.opponent ?? '—'}</span>
              <span className="pp-pop-log-line">{g.stat_line}</span>
              {/* The tone the bar above it took, off the same function: a
                  week cannot be green in the chart and amber in its row. */}
              <span className={`mono pp-pop-log-pts ${barTone(g.ppr_points, position)}`}>
                {g.ppr_points.toFixed(1)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </PopCard>
  )
}
