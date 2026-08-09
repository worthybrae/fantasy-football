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
  source: 'espn' | 'manual' | 'none'
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

export async function fetchSimLatest(): Promise<SimRun | null> {
  const res = await fetch('/api/sim/latest')
  if (!res.ok) throw new Error('Failed to read the last simulation')
  return (await res.json()).run
}

export interface Backtest {
  holdout_season: number | null
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
