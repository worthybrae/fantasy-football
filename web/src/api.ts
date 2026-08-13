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
  /** Whether a three-bar card may summarize this coefficient on its own.
   *  Comes from `draft_model.SUMMARY_FEATURES`; false for the position
   *  dummies, which only mean anything relative to each other. Server-side
   *  so adding a feature to the model doesn't need a matching edit here. */
  shown: boolean
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

// Measured statistics over a manager's real picks -- counted, never fitted,
// so every field here stays true whether or not their coefficients
// generalize (see scoring/draft_model.py's manager_tendencies). `mean_gap` is
// market_rank - overall_pick: positive means they take players earlier than
// the board ranks them, negative means they let players slide.
export interface ManagerTendencies {
  // Positions they've opened a draft with, most-used first. `drafts` is how
  // many of `of` drafts started that way.
  first_pick: { position: string; drafts: number; of: number }[]
  // Across every pick with a market rank to compare against. Null when none
  // of their picks matched that season's ADP board.
  reach: { mean_gap: number; n: number } | null
  // Same number split by round bucket, always in early -> late order.
  reach_by_bucket: { bucket: string; mean_gap: number; n: number }[]
  // Same number split by position, strongest reach first, only for positions
  // with enough picks behind them to mean anything.
  reach_by_position: { position: string; mean_gap: number; n: number }[]
  // Typical round of their first QB/TE/K/DST, earliest first. `drafts` is how
  // many drafts they took that position at all.
  first_at_position: { position: string; mean_round: number; drafts: number }[]
}

export interface ManagerHistory {
  manager: string
  // Distinct seasons this manager appears in draft_teams for -- "how many
  // drafts they appear in."
  seasons: number
  total_picks: number
  // Most recent season first.
  first_rounders: ManagerFirstRounder[]
  shape: { early: ManagerShapeBucket; mid: ManagerShapeBucket; late: ManagerShapeBucket }
  // Null on a database where `make fit-managers` has not written the
  // manager_tendencies table -- the card omits the block rather than
  // rendering blanks.
  tendencies: ManagerTendencies | null
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
  /** Null when the stored backtest row predates the column -- no comparison
   *  was made, which is not the same as the model having lost one. */
  beats_adp: boolean | null
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
  market_spread: number | null
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

// -- live draft mode --

// One recommendation from the running search. Sorted by `ev` descending on
// arrival -- `candidates[0]` is the call. No name/position/team here: the
// live loop is cheap by design (see api/live.py's DraftSession docstring)
// and joining against the full board is the caller's job, via `player_id`
// against `fetchPlayers()`'s one-time-fetched list.
export interface LiveCandidate {
  player_id: string
  ev: number
  se: number
  applied_pct: number
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
  // The extension has delivered a draft token. The onboarding gate flips from
  // "open your draft and click the extension" to the live board on this.
  token_received?: boolean
}

export async function fetchLiveState(): Promise<LiveState> {
  const res = await fetch('/api/live/state')
  if (!res.ok) {
    throw new Error(`Failed to load live state (${res.status}): ${await detailText(res)}`)
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

// No slot argument: the draft slot is not something a human should have to
// know. ESPN's socket announces the team on connect, and the backend
// translates it through draft_order. Until it does, `my_slot` is null --
// which is honest, and better than a guess that would attribute picks to
// the wrong manager.
export async function connectDraft(url: string): Promise<ConnectResult> {
  const res = await fetch('/api/live/connect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
  if (!res.ok) {
    // The backend writes this detail for a human to read, verbatim -- no
    // prefix, no re-wording (see detailText).
    throw new Error(await detailText(res))
  }
  return res.json()
}
