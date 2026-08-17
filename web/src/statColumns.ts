import type { GameStats, SeasonSummary } from './api'

// One position -> columns definition drives both the season table and the
// game log, so the two read consistently. Add a position here and both
// tables pick it up.
export interface StatColumn<T> {
  label: string
  value: (row: T) => string
}

const n = (v: number) => String(v)

const GAME_QB: StatColumn<GameStats>[] = [
  { label: 'Cmp/Att', value: (s) => `${s.completions}/${s.attempts}` },
  { label: 'Pass Yds', value: (s) => n(s.pass_yards) },
  { label: 'Pass TD', value: (s) => n(s.pass_tds) },
  { label: 'INT', value: (s) => n(s.interceptions) },
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
]
const GAME_RB: StatColumn<GameStats>[] = [
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
  { label: 'Rush TD', value: (s) => n(s.rush_tds) },
  { label: 'Tgt', value: (s) => n(s.targets) },
  { label: 'Rec', value: (s) => n(s.receptions) },
  { label: 'Rec Yds', value: (s) => n(s.rec_yards) },
]
const GAME_WRTE: StatColumn<GameStats>[] = [
  { label: 'Tgt', value: (s) => n(s.targets) },
  { label: 'Rec', value: (s) => n(s.receptions) },
  { label: 'Rec Yds', value: (s) => n(s.rec_yards) },
  { label: 'Rec TD', value: (s) => n(s.rec_tds) },
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
]

export function gameColumnsFor(position: string): StatColumn<GameStats>[] {
  if (position === 'QB') return GAME_QB
  if (position === 'RB') return GAME_RB
  return GAME_WRTE // WR/TE and any fallback
}

const fmtPct = (v: number | null) => (v === null ? '—' : `${(v * 100).toFixed(1)}%`)
const fmt1 = (v: number | null) => (v === null ? '—' : v.toFixed(1))

// Shared season-table tail (skill/usage rates); QB gets only Snap% -- target
// share and yards-per-opportunity are receiving/rushing constructs.
const SEASON_QB: StatColumn<SeasonSummary>[] = [
  { label: 'Cmp/Att', value: (s) => `${s.completions}/${s.attempts}` },
  { label: 'Pass Yds', value: (s) => n(s.pass_yards) },
  { label: 'Pass TD', value: (s) => n(s.pass_tds) },
  { label: 'INT', value: (s) => n(s.interceptions) },
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
  { label: 'Snap%', value: (s) => fmtPct(s.snap_share) },
]
const SEASON_RB: StatColumn<SeasonSummary>[] = [
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
  { label: 'Tgt', value: (s) => n(s.targets) },
  { label: 'Rec', value: (s) => n(s.receptions) },
  { label: 'Rec Yds', value: (s) => n(s.rec_yards) },
  { label: 'TD', value: (s) => n(s.tds) },
  { label: 'Tgt%', value: (s) => fmtPct(s.target_share) },
  { label: 'Yds/Opp', value: (s) => fmt1(s.yards_per_opp) },
  { label: 'Snap%', value: (s) => fmtPct(s.snap_share) },
]
const SEASON_WRTE: StatColumn<SeasonSummary>[] = [
  { label: 'Tgt', value: (s) => n(s.targets) },
  { label: 'Rec', value: (s) => n(s.receptions) },
  { label: 'Rec Yds', value: (s) => n(s.rec_yards) },
  { label: 'Car', value: (s) => n(s.carries) },
  { label: 'Rush Yds', value: (s) => n(s.rush_yards) },
  { label: 'TD', value: (s) => n(s.tds) },
  { label: 'Tgt%', value: (s) => fmtPct(s.target_share) },
  { label: 'Yds/Opp', value: (s) => fmt1(s.yards_per_opp) },
  { label: 'Snap%', value: (s) => fmtPct(s.snap_share) },
]

export function seasonColumnsFor(position: string): StatColumn<SeasonSummary>[] {
  if (position === 'QB') return SEASON_QB
  if (position === 'RB') return SEASON_RB
  return SEASON_WRTE
}

// Boom/bust shading relative to the player's own average: a game at 150%+
// of average reads clearly "boom", under 50% clearly "bust", with a lighter
// step either side and a neutral band around average so ordinary games stay
// unshaded.
export function ptsShadeClass(points: number, avg: number): string {
  if (avg <= 0) return ''
  const ratio = points / avg
  if (ratio >= 1.5) return 'pts-hot-2'
  if (ratio >= 1.15) return 'pts-hot-1'
  if (ratio <= 0.5) return 'pts-cold-2'
  if (ratio <= 0.85) return 'pts-cold-1'
  return ''
}

