import type { BoardStats, GameStats, SeasonSummary } from './api'

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

// Board (landing page) stat columns. `cell` renders the display string,
// `sortValue` the number TanStack sorts on -- split because Cmp/Att renders
// two numbers but sorts on attempts.
export interface BoardStatColumn {
  id: string
  label: string
  cell: (s: BoardStats) => string
  sortValue: (s: BoardStats) => number
}

export const BOARD_SUMMARY: BoardStatColumn[] = [
  { id: 'g', label: 'G', cell: (s) => n(s.games), sortValue: (s) => s.games },
  { id: 'ppg', label: 'PPG', cell: (s) => s.ppg.toFixed(1), sortValue: (s) => s.ppg },
  { id: 'pts', label: 'Pts', cell: (s) => s.points.toFixed(1), sortValue: (s) => s.points },
]

const BOARD_QB: BoardStatColumn[] = [
  { id: 'cmp_att', label: 'Cmp/Att', cell: (s) => `${s.completions}/${s.attempts}`, sortValue: (s) => s.attempts },
  { id: 'pass_yards', label: 'Pass Yds', cell: (s) => n(s.pass_yards), sortValue: (s) => s.pass_yards },
  { id: 'pass_tds', label: 'Pass TD', cell: (s) => n(s.pass_tds), sortValue: (s) => s.pass_tds },
  { id: 'interceptions', label: 'INT', cell: (s) => n(s.interceptions), sortValue: (s) => s.interceptions },
  { id: 'carries', label: 'Car', cell: (s) => n(s.carries), sortValue: (s) => s.carries },
  { id: 'rush_yards', label: 'Rush Yds', cell: (s) => n(s.rush_yards), sortValue: (s) => s.rush_yards },
]
const BOARD_RB: BoardStatColumn[] = [
  { id: 'carries', label: 'Car', cell: (s) => n(s.carries), sortValue: (s) => s.carries },
  { id: 'rush_yards', label: 'Rush Yds', cell: (s) => n(s.rush_yards), sortValue: (s) => s.rush_yards },
  { id: 'targets', label: 'Tgt', cell: (s) => n(s.targets), sortValue: (s) => s.targets },
  { id: 'receptions', label: 'Rec', cell: (s) => n(s.receptions), sortValue: (s) => s.receptions },
  { id: 'rec_yards', label: 'Rec Yds', cell: (s) => n(s.rec_yards), sortValue: (s) => s.rec_yards },
  { id: 'tds', label: 'TD', cell: (s) => n(s.tds), sortValue: (s) => s.tds },
]
const BOARD_WRTE: BoardStatColumn[] = [
  { id: 'targets', label: 'Tgt', cell: (s) => n(s.targets), sortValue: (s) => s.targets },
  { id: 'receptions', label: 'Rec', cell: (s) => n(s.receptions), sortValue: (s) => s.receptions },
  { id: 'rec_yards', label: 'Rec Yds', cell: (s) => n(s.rec_yards), sortValue: (s) => s.rec_yards },
  { id: 'carries', label: 'Car', cell: (s) => n(s.carries), sortValue: (s) => s.carries },
  { id: 'rush_yards', label: 'Rush Yds', cell: (s) => n(s.rush_yards), sortValue: (s) => s.rush_yards },
  { id: 'tds', label: 'TD', cell: (s) => n(s.tds), sortValue: (s) => s.tds },
]

// Summary always; a single-position tab appends that position's counting
// stats. ALL/FLEX/K/DST get summary only (mixed positions / no weekly stats).
export function boardColumnsFor(positionFilter: string): BoardStatColumn[] {
  if (positionFilter === 'QB') return [...BOARD_SUMMARY, ...BOARD_QB]
  if (positionFilter === 'RB') return [...BOARD_SUMMARY, ...BOARD_RB]
  if (positionFilter === 'WR' || positionFilter === 'TE') return [...BOARD_SUMMARY, ...BOARD_WRTE]
  return BOARD_SUMMARY
}
