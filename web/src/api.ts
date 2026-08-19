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

/** How old a timestamp is, in the one spelling the whole app uses. Shared by
 *  the live board's poll age and the landing page's readiness strip -- two
 *  spellings of "12m ago" would read as two different numbers. */
export function ageLabel(createdAt: string): string {
  const ms = Date.now() - new Date(createdAt).getTime()
  if (!Number.isFinite(ms) || ms < 60_000) return 'just now'
  const minutes = Math.floor(ms / 60_000)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

// -- live draft mode --

// One row of the ranked available list. Sorted by `gain_now` descending on
// the server (scoring/gain.py) -- the value this pick gains over the best
// player at the same position expected to survive to your next pick,
// weighted by whether your roster can start him. `vor_points` is the raw
// value over replacement it is derived from; the two differ most exactly
// where the old EV ranking used to reach. No name/position/team here beyond
// `position` itself: the live loop is cheap by design (see api/live.py's
// DraftSession docstring) and joining the rest against the full board is
// the caller's job, via `player_id` against `fetchPlayers()`'s
// one-time-fetched list.
export type LiveCandidate = {
  player_id: string
  position: string
  proj_points: number
  vor_points: number
  gain_now: number
  survive_pct: number
  fills: string
  rank: number
}

// A pick the socket reported but the crosswalk could not resolve to a
// `player_id` -- so it is still sitting on the board as "available" even
// though it was actually taken. `espn_player_id` is the raw id from ESPN's
// event, not a board id, since resolving it is exactly what failed.
export interface UnmappedPick {
  espn_player_id: number
  overall_pick: number
}

// The session's real league shape, straight off session.settings
// (scoring.league.LeagueSettings) -- see api/live.py's _league_settings_payload.
// Present with the same five keys on the inactive response too (same
// convention as listener_error/listener_alive below): null scalars and an
// empty `starters`, never an omitted key, so the rail never has to branch
// on whether `settings` exists, only on whether its values are null.
export interface LiveSettings {
  teams: number | null
  rounds: number | null
  starters: Record<string, number>
  flex_slots: number | null
  bench: number | null
  // scoring.league.scoring_format's own three values. Null only on the
  // inactive response, where there is no session's settings to derive it
  // from -- never a guessed default.
  scoring_format: 'ppr' | 'half' | 'std' | null
}

// One player on this session's own roster, in the order it drafted them --
// straight off /api/live/state's `my_roster` (api/live.py's _my_roster).
// `proj_points`, not `ev`: this is a plain replay of what was actually
// drafted, not a rerun of the simulator's end-of-draft valuation, so there
// is no `ev` here to report honestly.
export interface RosterPlayer {
  player_id: string
  name: string
  position: string
  proj_points: number | null
}

export interface LiveState {
  active: boolean
  picks_made: number
  on_the_clock: number | null
  // Absent (not merely null) on the `active: false` response -- see
  // api/live.py's `live_state`, whose no-session branch omits the key
  // entirely. Typed nullable rather than optional since every read site
  // already treats "no slot" and "not this slot" the same way.
  my_slot: number | null
  candidates: LiveCandidate[]
  // The pick count `candidates` was computed against. Compare against
  // `picks_made` to tell a current list from one a newer pick has already
  // outrun -- see `isRecomputing` in LiveDraft.tsx.
  candidates_as_of_pick: number | null
  last_poll_at: string | null
  // True whenever the listener hasn't successfully polled in the last 15s
  // (api/live.py's STALE_AFTER_SECONDS) -- including "never polled."
  stale: boolean
  unmapped_picks: UnmappedPick[]
  // A dead listener is the worst failure mode this system has -- the board
  // looks current and simply stops updating. `listener_error` carries the
  // exception that killed the thread (null if it never had one), and
  // `listener_alive` is the thread's live status, so a hang with no
  // exception is still visible even though it sets no error. Present (never
  // optional) in both the active and inactive responses -- see
  // api/live.py's `live_state`, which sets `listener_alive: false` on the
  // no-session branch rather than omitting the key.
  listener_error: string | null
  listener_alive: boolean
  // The bookmarklet has delivered a draft token. The onboarding gate flips
  // from "open your draft and click Draft Helper" to the live board on this.
  token_received?: boolean
  // The live pick clock, straight off DraftListener.ms_remaining
  // (pipeline/draft_listener.py) -- null until the first CLOCK or SELECTING
  // frame has been seen. Never decayed or interpolated client-side between
  // polls -- see ClockPanel.tsx's own comment on why not.
  ms_remaining: number | null
  settings: LiveSettings
  my_roster: RosterPlayer[]
}

export async function fetchLiveState(): Promise<LiveState> {
  const res = await fetch('/api/live/state')
  if (!res.ok) {
    throw new Error(`Failed to load live state (${res.status}): ${await detailText(res)}`)
  }
  return res.json()
}

export type SelectResult = { player_id: string; espn_id: number; pick_no: number }

// Makes the pick on ESPN. Resolves only once ESPN echoed it back, so a
// resolved promise means the pick is real -- there is no optimistic state
// anywhere above this. Relative path, matching every other call in this
// file: there is no base-URL constant here (the dev server proxies /api to
// the backend -- see vite.config.ts -- and the build serves both from the
// same origin), so there is no `API` to substitute for the task brief's
// `${API}` snippet.
export async function selectPlayer(playerId: string): Promise<SelectResult> {
  const res = await fetch('/api/live/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ player_id: playerId }),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    throw new Error(detail?.detail ?? `Pick failed (${res.status})`)
  }
  return res.json()
}

// -- live draft board (the round x team grid, GET /api/live/board) --

// A board cell's player: the same shape whether the pick was a steal or a
// reach, unlike `LiveCandidate` which carries no player identity at all --
// the grid renders straight off this, no join against `fetchPlayers()`
// needed. `value` is signed relative to ADP: positive means the player fell
// past where the market had him (a steal), negative means he went early (a
// reach).
export interface BoardPlayer {
  player_id: string
  name: string
  position: string
  team: string | null
  bye: number | null
  overall_rank: number | null
  tier: number | null
  market_rank: number | null
  espn_ppr_rank: number | null
  vor: number | null
  last_ppg: number | null
  last_points: number | null
  proj_ppg: number | null
  value: number | null
}

// One landed pick, already placed at its `round`/`slot` by the server --
// the grid trusts these rather than re-deriving them from `overall` and the
// team count, so it never has to know this league's snake variant.
export interface BoardCell {
  overall: number
  round: number
  slot: number
  player: BoardPlayer
}

// One column header. `is_me` marks the viewer's own team -- there is
// exactly one `true` among `teams` columns whenever `my_slot` is non-null.
export interface BoardColumn {
  slot: number
  team_name: string
  is_me: boolean
}

// Mirrors `LiveState`'s own convention (see its comment above): every field
// below is typed non-optional even though the `active: false` response
// omits all but `active` itself, since every read site already checks
// `board.active` before touching the rest.
export interface LiveBoard {
  active: boolean
  teams: number
  rounds: number
  my_slot: number | null
  on_the_clock: number | null
  picks_made: number
  columns: BoardColumn[]
  cells: BoardCell[]
}

export async function fetchBoard(): Promise<LiveBoard> {
  const res = await fetch('/api/live/board')
  if (!res.ok) {
    throw new Error(`Failed to load the draft board (${res.status}): ${await detailText(res)}`)
  }
  return res.json()
}

// The connect screen's only call. `board_fingerprint` identifies the pool
// build the session locked in, not shown to the user -- what the connect
// screen actually shows is a slot number, but that comes from a follow-up
// GET /api/live/state, not from this response: the resolved my_slot lives
// on the session state, not the connect endpoint's own return value.
export interface ConnectResult {
  connected: boolean
  league_id: string
  board_fingerprint: string
  // Resolved server-side from the URL's teamId. Echoed back so the connect
  // screen can show it for a sanity check -- a league that re-randomised its
  // draft order would make this silently wrong and nothing else would catch it.
  my_slot: number | null
}

// What the bookmarklet mints on the ESPN page and hands to this window in the
// URL hash: the throwaway per-draft token plus the public ids. The account
// session (espn_s2) is never among them -- it stays in the user's ESPN tab,
// where the bookmarklet used it only to fetch this token.
export interface TokenConnectParams {
  leagueId: string
  teamId: string
  swid: string
  token: string
  season: string
}

// The bookmarklet path's connect. The server opens ESPN's draft socket
// directly from these values (no browser window on the server), so this both
// starts the listener and returns the resolved slot -- one call, not a
// separate "store the token" step ahead of it.
export async function connectWithToken(
  params: TokenConnectParams,
): Promise<ConnectResult> {
  const res = await fetch('/api/live/connect-token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) {
    throw new Error(await detailText(res))
  }
  return res.json()
}

// -- landing page --------------------------------------------------------
//
// Two calls, split by what they cost. `status` is table reads (milliseconds)
// and paints the readiness strip on the first frame; `preview` pays for a
// board build (seconds) and lands behind a skeleton. Merging them would put
// the whole page behind the slow half.

export interface LandingSource {
  source: string
  ok: boolean
  rows: number
  refreshed_at: string | null
}

export interface LandingStatus {
  sources: LandingSource[]
  /** `derived` false means the built-in default shape, not a configured league. */
  league: { season: number; teams: number; rounds: number; derived: boolean }
  history: { picks: number; seasons: number[]; teams: number }
  /** `personal` is the subset of `fitted` whose own model beat the pooled one. */
  managers: { fitted: number; personal: number }
  sim: { run_id: string; my_slot: number | null; created_at: string } | null
}

export async function fetchLandingStatus(): Promise<LandingStatus> {
  const res = await fetch('/api/landing/status')
  if (!res.ok) throw new Error(`Failed to read setup status (${res.status})`)
  return res.json()
}

/** Eight fields, not the board's twenty-seven -- see PREVIEW_COLUMNS in api/main.py. */
export interface LandingPlayer {
  rank: number
  name: string
  position: string
  team: string | null
  tier: number | null
  vor: number | null
  market_rank: number | null
  edge: number | null
}

export async function fetchLandingPreview(limit = 12): Promise<{ pool: number; players: LandingPlayer[] }> {
  const res = await fetch(`/api/landing/preview?limit=${limit}`)
  if (!res.ok) throw new Error(`Failed to load the board preview (${res.status})`)
  return res.json()
}
