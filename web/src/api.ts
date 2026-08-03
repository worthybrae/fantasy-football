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
  espn_ppr_rank: number | null;
  rookie: boolean; drafted: boolean;
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
export interface PlayerProfileData {
  header: Player;
  factors: { production: number; durability: number; role: number;
             environment: number; schedule: number };
  summary: ProfileSummary;
  seasons: SeasonSummary[];
  game_log: GameLogRow[];
  outlook: Outlook;
  similar: { mode: 'stat_twins' | 'value_neighbors'; target_age?: number | null;
             players: SimilarPlayer[] };
}
export async function fetchProfile(playerId: string): Promise<PlayerProfileData> {
  const res = await fetch(`/api/players/${playerId}/profile`)
  if (!res.ok) throw new Error(`profile ${res.status}: ${await detailText(res)}`)
  return res.json()
}
