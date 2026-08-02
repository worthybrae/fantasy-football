export interface MarketSources {
  ffc: number | null; espn: number | null; fp: number | null; fp_tier: number | null;
}
export interface Player {
  rank: number; player_id: string; name: string; position: string;
  team: string; bye: number | null; tier: number;
  production: number; durability: number; role: number;
  environment: number; schedule: number;
  composite: number; vor: number; edge: number | null;
  market_rank: number | null; market_spread: number | null; market_sources: MarketSources;
  rookie: boolean; drafted: boolean;
}

export interface Weights {
  production: number; role: number; environment: number;
  schedule: number; durability: number;
}

export const DEFAULT_WEIGHTS: Weights = {
  production: 0.35, role: 0.25, environment: 0.2, schedule: 0.1, durability: 0.1,
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

export async function fetchPlayers(w: Weights): Promise<Player[]> {
  const params = new URLSearchParams(
    Object.entries(w).map(([k, v]) => [`w_${k}`, String(v)]))
  const res = await fetch(`/api/players?${params}`)
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
  season: number; games: number; ppg: number; targets: number;
  target_share: number | null; carries: number; rec_yards: number;
  rush_yards: number; tds: number; receptions: number;
  yards_per_opp: number | null; snap_share: number | null;
}
export interface GameLogRow {
  season: number; week: number; opponent: string | null; stat_line: string;
  ppr_points: number;
}
export interface SimilarPlayer {
  player_id: string | null; name: string; season: number | null;
  similarity: number | null; ppg: number | null; next_ppg: number | null;
  rank: number | null; market_rank: number | null;
}
export interface Outlook {
  depth_slot: number | null; implied_points: number | null;
  sos_raw: number | null; sos_pct: number | null; bye: number | null;
}
export interface PlayerProfileData {
  header: Player;
  factors: { production: number; durability: number; role: number;
             environment: number; schedule: number };
  seasons: SeasonSummary[];
  game_log: GameLogRow[];
  outlook: Outlook;
  similar: { mode: 'stat_twins' | 'value_neighbors'; players: SimilarPlayer[] };
}
export async function fetchProfile(playerId: string, weights: Weights): Promise<PlayerProfileData> {
  const params = new URLSearchParams(
    Object.entries(weights).map(([k, v]) => [`w_${k}`, String(v)]))
  const res = await fetch(`/api/players/${playerId}/profile?${params}`)
  if (!res.ok) throw new Error(`profile ${res.status}: ${await detailText(res)}`)
  return res.json()
}
