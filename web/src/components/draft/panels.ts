// What each hover-panel column looks like, built from the profile payload
// alone -- no React, no DOM, so the exact same function can sit behind the
// board's hover panels and the player profile popup. A builder duplicated
// into both trees is how they drift: one earns a fix and the other quietly
// keeps drawing the old picture.
import type { GameLogRow, SeasonSummary } from '../../api'
import type { Col } from './Chart'
import { finishPosition, finishTone } from './finish'
import { BAR_CEILING, SEASON_WEEKS, barTone } from './weeks'

// Exported: the popup's Per game panel puts baseline marks for the seasons a
// rookie has not played beside the columns these builders make, in the same
// chart. Two spellings of a season would show up as two axes in one picture.
export function year(season: number): string {
  return `’${String(season).slice(2)}`
}

// Seasons ran 16 games before 2021 and 17 from it, so "games missed" is only
// meaningful against the right denominator -- a 16-game 2019 is a full year,
// and drawing it as one short would invent an injury.
//
// Exported: HealthBody's head note ("of 17") reads off the same rule the
// column fills were cut against, so the two cannot say different things
// about the same season.
export function seasonLength(season: number): number {
  return season >= 2021 ? 17 : 16
}

/** Games played per season, INCLUDING seasons with none.
 *
 *  A player who missed a whole year has no row in the profile's `seasons`, and
 *  leaving that year out is precisely the mistake the Health meter exists not
 *  to make: `career_availability` counts absent seasons as zeros, so a panel
 *  that skipped them would explain a number by showing different data. Spans
 *  from his first season with any games through the most recent one measured.
 */
function availabilityRows(seasons: SeasonSummary[]): { season: number; games: number }[] {
  if (!seasons.length) return []
  const played = new Map(seasons.map((s) => [s.season, s.games]))
  const first = Math.min(...played.keys())
  const last = Math.max(...played.keys())
  const rows = []
  for (let season = first; season <= last; season += 1) {
    rows.push({ season, games: played.get(season) ?? 0 })
  }
  return rows
}

// Health keeps its circles -- they say "16 of 17" in a way a bar cannot, and
// it was the panel that already read well -- stacked into a column so the
// season axis runs the same way Finish's does.
const HEALTH_TONES = ['is-out', 'is-fringe', 'is-starter', 'is-strong', 'is-elite']

export function healthCols(seasons: SeasonSummary[]): Col[] {
  return availabilityRows(seasons).map((r) => {
    const of = seasonLength(r.season)
    const share = r.games / of
    return {
      key: r.season,
      label: year(r.season),
      value: String(r.games),
      // The same five steps the other panels use, cut on how much of the
      // season he was actually there for.
      tone: HEALTH_TONES[Math.min(4, Math.floor(share * 5))],
      fill: share,
      units: { filled: Math.min(r.games, of), of },
      empty: r.games === 0,
    }
  })
}

export function finishCols(seasons: SeasonSummary[], starters: number): Col[] {
  return seasons.slice().reverse().map((s) => ({
    key: s.season,
    label: year(s.season),
    value: String(s.pos_finish),
    tone: finishTone(s.pos_finish, starters),
    // `1 -` because rank runs backwards. A chart where the best season was
    // the shortest column is the one thing a reader cannot be asked to hold
    // in their head while comparing it to the two panels beside it.
    fill: 1 - finishPosition(s.pos_finish, starters),
  }))
}

// Steadiness, season by season, as a PLACE among the position rather than as
// the coefficient it is computed from.
//
// The panel used to draw `1 - cv/1.2` with the raw coefficient printed above
// it -- an abstract ratio on a clamped axis, inverted twice (a lower cv is
// better, a taller bar is better), and coloured off the bar's own height
// rather than off anything. Nothing on it answered the only question a reader
// brings: is 0.63 good? `cv_rank` has been in the payload the whole time and
// answers exactly that, so the panel now reads like Finish beside it: a rank,
// against a stated pool, taller is better.
//
// Percentile of the REAL pool, not the `starters` yardstick Finish uses.
// Steadiness is not scarce the way RB1 production is -- grading 28th of 95
// against an eight-team league's sixteen startable backs would paint a
// genuinely steady season red.
//
// Linear, not log-spaced like Finish: RB1 to RB6 is the difference between
// rounds, which is why that ladder is log, but consistency has no equivalent
// tier structure to stretch.
export function steadyTone(rank: number, pool: number): string {
  const pct = rank / pool
  if (pct <= 0.25) return 'is-elite'
  if (pct <= 0.5) return 'is-strong'
  if (pct <= 0.75) return 'is-starter'
  if (pct <= 0.9) return 'is-fringe'
  return 'is-out'
}

export function steadyCols(seasons: SeasonSummary[]): Col[] {
  // `cv_rank_n` counts only the seasons that HAVE a coefficient: a season
  // whose mean is zero or negative gets none, ranks nowhere, and would draw
  // as a column with no place on the ladder it is being plotted against.
  const rows = seasons.filter(
    (s) => s.cv_rank !== null && s.cv_rank !== undefined
      && s.cv_rank_n !== null && s.cv_rank_n !== undefined && s.cv_rank_n > 0,
  )
  return rows.slice().reverse().map((r) => {
    const rank = r.cv_rank as number
    const pool = r.cv_rank_n as number
    return {
      key: r.season,
      label: year(r.season),
      value: String(rank),
      tone: steadyTone(rank, pool),
      // `1 -` because rank runs backwards, same as Finish: the steadiest
      // season has to be the tallest column, or this panel and the one next
      // to it would read in opposite directions.
      fill: 1 - rank / pool,
    }
  })
}

// Scoring by season with the PROJECTION as the final column, because the
// Change column is the gap between the last of these and that one -- and a
// gap is the one thing a single number cannot show you the size of.
export function perGameCols(
  seasons: SeasonSummary[], projPpg: number | null, position: string,
): Col[] {
  const rows = seasons.filter((s) => s.games > 0)
  const ceiling = Math.max(...rows.map((r) => r.ppg), projPpg ?? 0, 1)
  const cols: Col[] = rows.slice().reverse().map((r) => ({
    key: r.season,
    label: year(r.season),
    value: r.ppg.toFixed(1),
    tone: barTone(r.ppg, position),
    fill: r.ppg / ceiling,
  }))
  if (projPpg !== null) {
    cols.push({
      key: 'proj',
      // Not a year: this column is the only one that has not happened.
      label: 'proj',
      value: projPpg.toFixed(1),
      tone: barTone(projPpg, position),
      fill: projPpg / ceiling,
      // Drawn hollow, behind a rule: everything left of it is banked, this
      // is the only column still owed. The tone stays, so it is still read
      // against the same good/mid/bad cut points as the seasons beside it.
      projected: true,
    })
  }
  return cols
}

export function weekCols(gameLog: GameLogRow[], season: number, position: string): Col[] {
  const played = new Map(
    gameLog.filter((g) => g.season === season && !g.dnp).map((g) => [g.week, g]))
  return Array.from({ length: SEASON_WEEKS }, (_, i) => {
    const week = i + 1
    const game = played.get(week)
    const pts = game?.ppr_points ?? null
    return {
      key: week,
      // The opponent, not the week number: a reader knows where week 9 is from
      // its place in the row, and "DEN" is the fact that explains a bad one.
      // The number would be the axis restating itself.
      label: game?.opponent ?? '—',
      value: pts === null ? '·' : String(Math.round(pts)),
      tone: pts === null ? '' : barTone(pts, position),
      fill: pts === null ? 0 : Math.min(1, pts / BAR_CEILING),
      empty: pts === null,
    }
  })
}
