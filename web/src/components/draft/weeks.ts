// What a week was worth, and what colour that makes it.
//
// Shared by the board's inline sparkline and the hover panel behind it. They
// draw the same season at different sizes, and a week that is green in one
// and amber in the other would be two answers to one question.

const BAR_THRESHOLDS: Record<string, { amberFrom: number; greenFrom: number }> = {
  QB: { amberFrom: 10, greenFrom: 20 },
  K: { amberFrom: 5, greenFrom: 10 },
  DST: { amberFrom: 5, greenFrom: 10 },
}
// RB, WR, TE, and anything the server ever sends that isn't one of the three
// positions above -- the original 10/15 split.
const BAR_THRESHOLDS_DEFAULT = { amberFrom: 10, greenFrom: 15 }

export function barThresholds(position: string): { amberFrom: number; greenFrom: number } {
  return BAR_THRESHOLDS[position] ?? BAR_THRESHOLDS_DEFAULT
}

export function barTone(points: number, position: string): string {
  const { amberFrom, greenFrom } = barThresholds(position)
  if (points >= greenFrom) return 'is-good'
  if (points >= amberFrom) return 'is-mid'
  return 'is-bad'
}

// One scale for every player, so a bad player's best week cannot draw as tall
// as a stud's. 30 clips about 3% of games on the real board against a 95th
// percentile of 27.3 -- and the weeks it clips are the unmistakable ones.
export const BAR_CEILING = 30

// A full NFL season, so the chart's width is the season rather than however
// many weeks this player happened to appear in. A row of 12 bars and a row of
// 18 sitting in the same panel would make six missed games invisible.
export const SEASON_WEEKS = 18

// GAMES a team plays, which is SEASON_WEEKS minus the bye -- deliberately a
// separate number, because the two are used for different things and a single
// constant serving both would be wrong for one of them.
//
// Must equal `GAMES` in scoring/board.py: that is the divisor behind the
// Change column, so a projection shown per game here and a change measured
// per game there would otherwise be per-game in two different senses.
export const SEASON_GAMES = 17
