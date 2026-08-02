export interface Player {
  rank: number; player_id: string; name: string; position: string;
  team: string; bye: number | null; tier: number;
  production: number; durability: number; role: number;
  environment: number; schedule: number;
  composite: number; vor: number; adp: number | null; edge: number | null;
  rookie: boolean; drafted: boolean;
}

export interface Weights {
  production: number; role: number; environment: number;
  schedule: number; durability: number;
}

export const DEFAULT_WEIGHTS: Weights = {
  production: 0.35, role: 0.25, environment: 0.2, schedule: 0.1, durability: 0.1,
}

export async function fetchPlayers(w: Weights): Promise<Player[]> {
  const params = new URLSearchParams(
    Object.entries(w).map(([k, v]) => [`w_${k}`, String(v)]))
  const res = await fetch(`/api/players?${params}`)
  return (await res.json()).players
}

export async function setDrafted(playerId: string, drafted: boolean): Promise<void> {
  await fetch(`/api/drafted/${playerId}`, { method: drafted ? 'POST' : 'DELETE' })
}

export async function fetchMeta(): Promise<{ sources: { source: string; ok: boolean; rows: number; refreshed_at: string }[] }> {
  return (await fetch('/api/meta')).json()
}
