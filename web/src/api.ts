export interface MarketSources {
  ffc: number | null; espn: number | null; fp: number | null;
  mfl: number | null; cbs: number | null; fp_tier: number | null;
}
export interface BoardStats {
  season: number; games: number; ppg: number; points: number;
  carries: number; rush_yards: number; targets: number; receptions: number;
  rec_yards: number; tds: number; completions: number; attempts: number;
  pass_yards: number; pass_tds: number; interceptions: number;
}
export interface Player {
  player_id: string; name: string; position: string;
  team: string; bye: number | null;
  market_rank: number | null; market_spread: number | null; market_sources: MarketSources;
  espn_ppr_rank: number | null;
  stats: BoardStats | null;
  rookie: boolean; drafted: boolean;
  avail_pct: number | null; ev: number | null; ev_se: number | null;
  // Model-native rank (VOR order across the whole board), tier (gap-based,
  // within position), and market_rank - rank ("edge": positive = the model
  // likes this player more than the market does). Dropped from the board
  // table's columns in 1f8ba18 as too dense for a sortable grid, but still
  // computed on every board row -- PlayerCard's decision row surfaces them
  // again for the condensed, single-player view. edge is nullable because
  // it's undefined whenever market_rank is (no market source covers the
  // player at all).
  rank: number; tier: number; edge: number | null;
}

// The baseline every ΔEV on screen is measured against: the highest `ev` on
// the whole board, so the top sim candidate reads as a dash and everything
// else reads as what it costs you to take instead. Shared rather than
// recomputed per view -- two views disagreeing about the baseline would show
// two different ΔEVs for the same player. Null when no sim has run (`ev` is
// null for every player until then), and note the baseline is over the board
// list, never over a subset: `search_pick` only scores about twelve
// candidates, so a subset's maximum is not the board's.
export function bestEv(players: Player[]): number | null {
  const values = players.map((p) => p.ev).filter((v): v is number => v !== null)
  return values.length ? Math.max(...values) : null
}

// URL slug for a player page: accent-folded kebab-case name
// ("Amon-Ra St. Brown" -> "amon-ra-st-brown"). Resolved back to a player by
// scanning the board list, so it must be a pure function of the name.
export function playerSlug(name: string): string {
  return name
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

async function detailText(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (body && typeof body.detail === 'string') return body.detail
  } catch {
    // body wasn't JSON (or was empty) -- fall through to a generic message
  }
  return res.statusText || 'request failed'
}

export async function fetchPlayers(): Promise<Player[]> {
  const res = await fetch('/api/players')
  if (!res.ok) {
    throw new Error(`Failed to load players (${res.status}): ${await detailText(res)}`)
  }
  return (await res.json()).players
}

export async function setDrafted(playerId: string, drafted: boolean): Promise<void> {
  await fetch(`/api/drafted/${playerId}`, { method: drafted ? 'POST' : 'DELETE' })
}

export async function fetchMeta(): Promise<{ sources: { source: string; ok: boolean; rows: number; refreshed_at: string }[] }> {
  const res = await fetch('/api/meta')
  if (!res.ok) {
    throw new Error(`Failed to load meta (${res.status}): ${await detailText(res)}`)
  }
  return res.json()
}

export interface SeasonSummary {
  season: number; games: number; ppg: number; ppg_std: number | null;
  pos_finish: number; targets: number;
  target_share: number | null; carries: number; rec_yards: number;
  rush_yards: number; tds: number; receptions: number;
  yards_per_opp: number | null; snap_share: number | null;
  completions: number; attempts: number; pass_yards: number;
  pass_tds: number; interceptions: number;
}
export interface GameStats {
  completions: number; attempts: number; pass_yards: number;
  pass_tds: number; interceptions: number; carries: number;
  rush_yards: number; rush_tds: number; targets: number;
  receptions: number; rec_yards: number; rec_tds: number;
}
export interface GameLogRow {
  season: number; week: number; opponent: string | null; stat_line: string;
  stats: GameStats; ppr_points: number; dnp: boolean;
}
export interface SimilarPlayer {
  player_id: string | null; name: string; season: number | null;
  similarity: number | null; ppg: number | null; next_ppg: number | null;
  age?: number | null;
  rank: number | null; market_rank: number | null;
}
export interface Outlook {
  depth_slot: number | null; implied_points: number | null;
  sos_raw: number | null; sos_pct: number | null; bye: number | null;
}
export interface ProfileSummary {
  w_ppg: number | null;
  w_stats: Record<string, number>;
  proj_ppg: number | null;
  proj_delta: number | null;
}
export interface DepthChartGroup {
  position: string;
  players: { name: string; rank: number; is_me: boolean }[];
}
export interface ScheduleWeek {
  week: number; opponent: string | null; home: boolean | null;
  fpa_pg: number | null; pct: number | null;
}
export interface PlayerProfileData {
  header: Player;
  factors: { production: number; durability: number; role: number;
             environment: number; schedule: number };
  summary: ProfileSummary;
  seasons: SeasonSummary[];
  game_log: GameLogRow[];
  outlook: Outlook;
  depth_chart: DepthChartGroup[];
  schedule: ScheduleWeek[];
  similar: { mode: 'stat_twins' | 'value_neighbors'; target_age?: number | null;
             players: SimilarPlayer[] };
}
export async function fetchProfile(playerId: string): Promise<PlayerProfileData> {
  const res = await fetch(`/api/players/${playerId}/profile`)
  if (!res.ok) throw new Error(`profile ${res.status}: ${await detailText(res)}`)
  return res.json()
}

export interface ManagerCoefficient {
  feature: string
  value: number
  pooled_value: number
}

export interface Manager {
  manager: string
  summary: string
  n_picks: number
  uses_personal: boolean
  heldout_gain: number | null
  coefficients: ManagerCoefficient[]
}

export interface DraftOrderEntry {
  slot: number
  manager: string
}

export interface DraftOrder {
  order: DraftOrderEntry[]
  my_slot: number | null
  // 'unpublished' means ESPN has the league's managers but has not set a
  // draft order yet, so the slots below are a placeholder to rearrange.
  source: 'espn' | 'manual' | 'none' | 'unpublished'
}

// Just the fields DraftRail needs (teams/rounds, for turning a pick count
// into round.pick notation) -- /api/league returns more (season, starters,
// flex_slots, bench, derived, unmapped_scoring) that nothing on this board
// consumes yet.
export interface LeagueInfo {
  teams: number
  rounds: number
}

export async function fetchManagers(): Promise<Manager[]> {
  const res = await fetch('/api/managers')
  if (!res.ok) throw new Error('Failed to load managers')
  return (await res.json()).managers
}

// A manager's real round-1 pick in one season. player_name/position/
// nfl_team are null when that pick's ESPN id was missing from that season's
// player directory (a real gap in ESPN's own data, not a bug) -- render as
// an explicit "unidentified pick" rather than the literal string "null".
export interface ManagerFirstRounder {
  season: number
  player_name: string | null
  position: string | null
  nfl_team: string | null
  keeper: boolean
}

// Pick counts by position within a round bucket ("early" = rounds 1-3,
// "mid" = 4-8, "late" = 9+ -- the same cutoffs the model trains against, see
// api/main.py's _history_round_bucket). Keyed loosely rather than by a fixed
// position union so an unexpected position from the backend degrades to
// "not shown" instead of a type error.
export type ManagerShapeBucket = Record<string, number>

export interface ManagerHistory {
  manager: string
  // Distinct seasons this manager appears in draft_teams for -- "how many
  // drafts they appear in."
  seasons: number
  total_picks: number
  // Most recent season first.
  first_rounders: ManagerFirstRounder[]
  shape: { early: ManagerShapeBucket; mid: ManagerShapeBucket; late: ManagerShapeBucket }
}

export async function fetchManagerHistory(): Promise<ManagerHistory[]> {
  const res = await fetch('/api/managers/history')
  if (!res.ok) throw new Error('Failed to load manager history')
  return (await res.json()).managers
}

export async function fetchDraftOrder(): Promise<DraftOrder> {
  const res = await fetch('/api/draft-order')
  if (!res.ok) throw new Error('Failed to load draft order')
  return res.json()
}

export async function fetchLeague(): Promise<LeagueInfo> {
  const res = await fetch('/api/league')
  if (!res.ok) throw new Error('Failed to load league settings')
  return res.json()
}

export async function saveDraftOrder(order: DraftOrderEntry[], mySlot: number) {
  const res = await fetch('/api/draft-order', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ order, my_slot: mySlot }),
  })
  if (!res.ok) throw new Error('Failed to save draft order')
}

export async function startSim(mySlot: number, rollouts: number): Promise<string> {
  const res = await fetch('/api/sim', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ my_slot: mySlot, rollouts }),
  })
  if (!res.ok) throw new Error('Failed to start simulation')
  return (await res.json()).run_id
}

export async function pollSim(runId: string): Promise<{ status: string; detail: string | null }> {
  const res = await fetch(`/api/sim/${runId}`)
  if (!res.ok) throw new Error('Failed to read simulation status')
  return res.json()
}

// Provenance for whatever sim the board's Avail%/ΔEV columns currently come
// from. Those columns are merged into every /api/players response and each
// run replaces the tables wholesale, so on a fresh page load they are some
// run -- possibly for a different slot, possibly hours old.
export interface SimRun {
  run_id: string
  my_slot: number | null
  /** Overall pick number the run was computed for. */
  pick_no: number | null
  created_at: string
}

/** How stale a sim run is. Shared: the rail and the predicted grid both
 *  render the age of the SAME run, and two spellings of "12m ago" on two
 *  screens for one number would read as two different numbers. */
export function ageLabel(createdAt: string): string {
  const ms = Date.now() - new Date(createdAt).getTime()
  if (!Number.isFinite(ms) || ms < 60_000) return 'just now'
  const minutes = Math.floor(ms / 60_000)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export async function fetchSimLatest(): Promise<SimRun | null> {
  const res = await fetch('/api/sim/latest')
  if (!res.ok) throw new Error('Failed to read the last simulation')
  return (await res.json()).run
}

export interface Backtest {
  /** Every season rotated through as the LOSO holdout, not a single one. */
  seasons: number[] | null
  top1: number | null
  top5: number | null
  logloss: number | null
  adp_top1: number | null
  adp_logloss: number | null
  beats_adp: boolean
}

export interface ModelStatus {
  /** Whether any manager models have been fitted at all. */
  fitted: boolean
  n_managers: number
  backtest: Backtest | null
}

export async function fetchModel(): Promise<ModelStatus> {
  const res = await fetch('/api/model')
  if (!res.ok) throw new Error('Failed to load model status')
  return res.json()
}

export interface SimBoardCell {
  overall_pick: number
  round: number
  round_pick: number
  slot: number
  alt_rank: number
  player_id: string
  name: string
  position: string | null
  team: string | null
  prob: number
  certain: boolean
}

export interface SimBoard {
  // The same SimRun /api/sim/latest serves -- the board handler delegates to
  // it, so my_slot/pick_no are nullable here too (a run that predates those
  // columns has no slot on record). DraftGrid degrades to "no column marked
  // (you)" rather than assuming one.
  run: SimRun | null
  teams: number
  rounds: number
  order: DraftOrderEntry[]
  cells: SimBoardCell[]
}

export async function fetchSimBoard(): Promise<SimBoard> {
  const res = await fetch('/api/sim/board')
  if (!res.ok) throw new Error('Failed to load the predicted board')
  return res.json()
}
