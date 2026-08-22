import { useState } from 'react'

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
  // Every season he has a PLAYED game in, newest first. A season of nothing
  // but DNPs is not offered: it would be a chart of baseline marks over an
  // empty log, and an arrow that lands on one is an arrow that appears
  // broken.
  const played = [...new Set(games.filter((g) => !g.dnp).map((g) => g.season))]
    .sort((a, b) => b - a)
  // Newest by default -- form is what someone opening this mid-draft is
  // reading for. Held as the season itself rather than an index so a payload
  // swapping under it (a comp clicked inside the profile) cannot land on
  // another player's third-newest year.
  const [shown, setShown] = useState<number | null>(null)
  if (played.length === 0) return null
  const season = shown !== null && played.includes(shown) ? shown : played[0]
  const at = played.indexOf(season)
  const log = games
    .filter((g) => g.season === season && !g.dnp)
    .sort((a, b) => b.week - a.week)

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
      {/* An arrow on each side of the picture, pointing the way time runs:
          left is the season before this one, right is the season after. Both
          are real buttons -- the popup is keyboard-reachable, and a clickable
          div here would be a hole in that. Disabled rather than hidden at
          each end, so the chart does not shift sideways as you walk the
          career. */}
      <div className="pp-pop-weeks-nav">
        <button
          type="button"
          className="pp-pop-weeks-arrow"
          onClick={() => setShown(played[at + 1])}
          disabled={at >= played.length - 1}
          aria-label={at >= played.length - 1 ? 'No earlier season'
            : `Show ${played[at + 1]}`}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"
               strokeLinejoin="round" aria-hidden="true">
            <path d="M15 5 8 12l7 7" />
          </svg>
        </button>
        <div className="pp-pop-weeks-chart">
          <Chart cols={weekCols(games, season, position)} />
        </div>
        <button
          type="button"
          className="pp-pop-weeks-arrow"
          onClick={() => setShown(played[at - 1])}
          disabled={at <= 0}
          aria-label={at <= 0 ? 'No later season' : `Show ${played[at - 1]}`}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"
               strokeLinejoin="round" aria-hidden="true">
            <path d="m9 5 7 7-7 7" />
          </svg>
        </button>
      </div>

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
