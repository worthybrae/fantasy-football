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
  // Average games played per season across this player's WHOLE career, with
  // seasons he missed entirely counted as zero rather than skipped (see
  // scoring/board.career_availability -- the naive version reports a median
  // of 7.2 games against a true 3.0). `null` for a defense, or anyone with
  // no NFL season behind him. The available table draws it as a five-bar
  // meter; it is a RATE, not the board's `durability` percentile.
  career_games_pg: number | null;
  // Week-to-week steadiness. `_cv` is the raw coefficient of variation
  // (sigma over mean, lower is steadier); `_pct` is its percentile among the
  // players at his position ON THIS BOARD, steadiest highest.
  consistency_cv: number | null;
  consistency_pct: number | null;
  // Up to five [season, positional finish] pairs, oldest first -- "he
  // finished RB12", ranked on season points, the same framing the profile
  // uses so the two views cannot disagree about what RB12 means. `null` for
  // a defense or anyone with no NFL season yet.
  season_finishes: [number, number][] | null;
  // Projected points per game minus his recency-weighted actual. Per-game on
  // both sides, so a short season is not read as decline -- see
  // scoring/board.expected_change, including the one thing it cannot
  // separate: a quarterback who lost his job reads as a huge per-game fall.
  proj_change: number | null;
  // Points in each week of the last COMPLETE season -- index 0 is week 1 --
  // scored under the league's own rules, not always PPR (scoring/game_points.py).
  // The available table draws it as an inline bar chart.
  //   * `null` for a player with no rows in that season at all: a rookie,
  //     every defense (nflverse has no team-defense weekly rows), and a
  //     kicker in a league that prices no kicking, whose season would
  //     otherwise be a row of noughts claiming he never scored. Rendered as
  //     an empty state, never as a flat chart.
  //   * a `null` ENTRY is a week with no game -- a bye, an injury, a season
  //     that started late. Different from 0.0, which is a game he played and
  //     scored nothing in, and the chart draws the two differently.
  // Same season as `stats.season`: both come from `max(season)` in `weekly`.
  game_points: (number | null)[] | null;
  rookie: boolean; drafted: boolean;
  avail_pct: number | null; ev: number | null; ev_se: number | null;
  // Model-native rank (VOR order across the whole board), tier (gap-based,
  // within position), and market_rank - rank ("edge": positive = the model
  // likes this player more than the market does). Dropped from the board
  // table's columns in 1f8ba18 as too dense for a sortable grid, but still
  // computed on every board row -- the profile popup's header figures and
  // status line, and DraftBoardGrid's popover, surface them. edge is
  // nullable because
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
  // Week-to-week swing: sigma over mean, and its rank within the season's
  // positional pool. Null for a season under scoring/profile_cache.py's
  // RANK_MIN_GAMES -- a coefficient off four appearances is not a
  // measurement, and null says so where a zero would lie.
  cv: number | null; cv_rank: number | null; cv_rank_n: number | null;
  cv_pos_median: number | null;
  pos_rank_ppg: number | null; pos_rank_ppg_n: number | null;
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
// player at the same position expected to survive to `horizon_pick` below
// (the first turn of yours far enough away for the comparison to mean
// anything, NOT necessarily your immediately-next one), weighted by whether
// your roster can start him, and `survive_pct` is measured to that same
// pick. `vor_points` is the raw value over replacement it is derived from;
// the two differ most exactly where the old EV ranking used to reach. No
// name/position/team here beyond
// `position` itself: the live loop is cheap by design (see api/live.py's
// DraftSession docstring) and joining the rest against the full board is
// the caller's job, via `player_id` against `fetchPlayers()`'s
// one-time-fetched list.
//
// `gain_now`/`survive_pct`/`fills` are `null` together, never individually,
// whenever `/api/live/state`'s `my_slot` is itself null: there is no roster
// to rank a pick FOR yet (see scoring/gain.available_by_vor's own
// docstring), so the list is sorted by `vor_points` alone instead. One
// shape either way -- the frontend never branches on "which payload is
// this," only on whether a given row's own fields are null (AvailableList
// renders `—`; TopThree does not render its cards at all, see its own
// comment).
export type LiveCandidate = {
  player_id: string
  position: string
  proj_points: number
  vor_points: number
  gain_now: number | null
  survive_pct: number | null
  fills: string | null
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
  // DraftListener.started, straight off the socket's own STATE frame
  // (pipeline/draft_listener.py) -- present (never omitted) on both the
  // active and inactive responses, same convention as listener_alive below.
  // Lets ClockPanel tell "the draft has not started yet" apart from
  // "started, waiting on someone else's pick" -- both used to render as the
  // identical "Waiting on the room" heading.
  draft_started: boolean
  candidates: LiveCandidate[]
  // The pick COUNT `candidates` was computed against, so the pick they are
  // for is that + 1 -- the same +1 the room applies to `picks_made`. Less
  // than `picks_made` means a newer pick has already outrun this list and
  // the recompute for it has not landed yet; DraftRoom says so rather than
  // presenting a superseded list as current. (The old pointer here was to
  // `isRecomputing` in LiveDraft.tsx, a file this branch deleted -- and
  // with it, for a while, the indicator itself.)
  candidates_as_of_pick: number | null
  // The pick `gain_now` and `survive_pct` were actually measured against:
  // the first turn of yours at least a full round of opponent picks away
  // (scoring/draft_sim.horizon_picks), which at the wheel and at short gaps
  // is NOT your immediately-next pick. Stored server-side alongside
  // `candidates` under the same lock, so it always describes the list that
  // came with it rather than the poll that fetched it.
  //
  // null when no gain-ranked list exists yet (no session, or my_slot
  // unresolved so the list is vor_points-only) AND when the horizon is the
  // end of the draft -- `horizon_is_end_of_draft` is what tells those apart,
  // and it exists because the alternative was serving pick 121 of a
  // 120-pick draft. Never both: a number here means a pick that exists.
  horizon_pick: number | null
  horizon_is_end_of_draft: boolean
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
  // The recompute worker's own last failure, separate from the listener's
  // because they fail independently: the listener can be perfectly healthy
  // -- frames arriving, the board filling, `listener_alive` true -- while
  // ranking has stopped dead and `candidates` is frozen at an old pick.
  // Present (never optional) on both responses, same as the two above.
  recompute_error: string | null
  // Whether a SELECT actually has somewhere to go -- exactly the condition
  // POST /api/live/select's 503 gates on. Whose turn it is is not enough on
  // its own: during a socket reconnect the handle is detached while the
  // listener thread is alive and `stale` has not tripped, so the draft
  // buttons have to read this too or every click 503s (spec section 6).
  socket_alive: boolean
  // ESPN's own autodraft flag for THIS session's team -- true means ESPN is
  // making the picks itself, which is what happens the moment you miss a
  // turn. Straight off DraftListener.my_autodraft (pipeline/
  // draft_listener.py), which reads it from the `AUTODRAFT <teamId>
  // <true|false>` frames ESPN broadcasts for every team in the room.
  //
  // Three values, not two. `null` means ESPN has not said yet -- no
  // AUTODRAFT frame for our team, or the socket has not named our team --
  // and it is a different fact from `false`. Anything that renders this must
  // keep them apart: showing "off" for a state nobody has confirmed is the
  // exact failure the flag exists to prevent. In practice null lasts about
  // as long as the first frame of a session (ESPN states autodraft in its
  // JOIN replay before anything else) and is permanent only where there is
  // no session at all.
  autodraft: boolean | null
  // The bookmarklet has delivered a draft token. The onboarding gate flips
  // from "open your draft and click Draft Assistant" to the live board on this.
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
//
// Aborted client-side at 15s. The server's own bound (api/live.py's
// SELECT_TIMEOUT_SECONDS, 8s) covers a slow ESPN, not a hung connection --
// a fetch that never settles leaves ConfirmPick stuck on 'sending', where
// Escape and Cancel are both correctly disabled (the SELECT has already
// left the tab, so there is nothing left to cancel), for the rest of the
// pick clock with no way out. 15s sits comfortably past the server's 8s
// plus its round trip, so a genuinely slow confirmation still lands here
// rather than being cut off; the only thing this catches is a request that
// is never coming back.
const SELECT_ABORT_MS = 15000

export async function selectPlayer(playerId: string): Promise<SelectResult> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), SELECT_ABORT_MS)
  try {
    const res = await fetch('/api/live/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_id: playerId }),
      signal: controller.signal,
    })
    if (!res.ok) {
      const detail = await res.json().catch(() => null)
      throw new Error(detail?.detail ?? `Pick failed (${res.status})`)
    }
    return await res.json()
  } catch (e) {
    // Deliberately the server's own 504 wording: an aborted request is the
    // same situation as ESPN not answering -- the SELECT may or may not have
    // landed, and the only thing that can resolve it is looking at ESPN.
    // Anything else (a real HTTP error, a network failure) is rethrown
    // untouched so ConfirmPick still renders the server's own message.
    if (controller.signal.aborted) {
      throw new Error('ESPN did not confirm the pick -- check the ESPN draft '
        + 'room before picking again')
    }
    throw e
  } finally {
    clearTimeout(timer)
  }
}

// Turns ESPN's autodraft on or off. Resolves only once ESPN echoed the
// change back for our own team, exactly like `selectPlayer` above -- a
// resolved promise means ESPN confirmed it, and nothing anywhere writes the
// flag optimistically. The next poll of /api/live/state is what moves the
// switch; this call only ever decides whether that poll will find it moved.
//
// Aborted client-side on the same reasoning and the same margin as
// selectPlayer's (see SELECT_ABORT_MS): the server's own bound is 8s
// (api/live.py's AUTODRAFT_TIMEOUT_SECONDS), and this catches only a request
// that is never coming back, which would otherwise leave the switch stuck
// pending -- and therefore un-clickable -- for the rest of the draft.
const AUTODRAFT_ABORT_MS = 15000

export async function setAutodraft(on: boolean): Promise<{ autodraft: boolean; changed: boolean }> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), AUTODRAFT_ABORT_MS)
  try {
    const res = await fetch('/api/live/autodraft', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ on }),
      signal: controller.signal,
    })
    if (!res.ok) {
      const detail = await res.json().catch(() => null)
      throw new Error(detail?.detail ?? `Could not change autodraft (${res.status})`)
    }
    return await res.json()
  } catch (e) {
    // The server's own 504 wording, for the same reason selectPlayer borrows
    // it: an aborted request is the same situation as ESPN not answering --
    // the command may or may not have landed, and only the ESPN draft room
    // can settle it.
    if (controller.signal.aborted) {
      throw new Error('ESPN did not confirm the autodraft change -- check the '
        + 'ESPN draft room')
    }
    throw e
  } finally {
    clearTimeout(timer)
  }
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

// -- live draft plan (the "Plan" tab, GET /api/live/plan) -------------------
//
// Answers "given my draft slot, which positions should I take in which
// round, and why" -- the round-by-round prediction plus the positional
// cliffs (points of projected value lost by waiting one more round) that
// explain it. Server cost is ~4s (120 simulated drafts) and the plan only
// changes when a pick lands, so DraftRoom refetches it on `picks_made` changing rather than on
// the 2.5s state/board timer -- see its own poll effect.

export interface PlanPositionShare {
  position: string
  pct: number
}

// One round of the round-by-round plan. `positions` is sorted descending by
// `pct` and only carries positions above a small server-side threshold, so
// its length runs 1 to 4 -- QB/RB/WR/TE **and** K/DST all legitimately
// appear here (unlike `cliffs`/`best_available` below): a real roster needs
// one of each late, and the plan predicts them the same as any other
// position.
//
// `is_past`/`actual` is the one distinction the whole tab exists to make
// obvious: `is_past: true` means this pick already happened in the live
// draft and `actual` names what was actually taken -- settled fact, not
// prediction. `actual` is null exactly when `is_past` is false.
export interface PlanRound {
  round: number
  pick: number
  is_past: boolean
  actual: string | null
  positions: PlanPositionShare[]
}

// Points of projected value lost by waiting one more round at that
// position, keyed by QB/RB/WR/TE only -- K and DST never carry a cliff.
// `0.0` is common and meaningful ("costs nothing to wait"), never a
// placeholder for "unknown".
export interface PlanCliffRound {
  round: number
  pick: number
  by_position: Record<string, number>
}

// The best projected points still on the board at that position/round --
// what `PlanCliffRound.by_position` is the round-over-round drop of. Same
// QB/RB/WR/TE-only keying as the cliff it backs.
export interface PlanBestAvailableRound {
  round: number
  pick: number
  by_position: Record<string, number>
}

// `cliffs`/`best_available` carry one entry per round, same `round`/`pick`
// keys as `rounds_plan`, but can run SHORTER than it: the last round has no
// "next round" to wait for, so there is nothing to measure a cliff against.
// Always join these against `rounds_plan` by `round` -- never by array
// index -- and render a round with no matching cliff entry as an empty
// cliff row rather than dropping it.
//
// Every field here is required: a plan missing its slot or its rounds is
// exactly the `pending` case below, not a `ready` one with holes in it.
export interface LivePlan {
  my_slot: number
  teams: number
  rounds: number
  n_drafts: number
  as_of_pick: number
  rounds_plan: PlanRound[]
  cliffs: PlanCliffRound[]
  best_available: PlanBestAvailableRound[]
}

// GET /api/live/plan always answers 200, in one of three shapes:
//   1. no draft session at all --
//      {"active": false, "pending": false, "plan": null}
//   2. a session, but no plan yet -- the slot isn't resolved, or the first
//      ~4s of simulation is still running (indistinguishable from here, and
//      meant to be: both are a quiet loading state, never an error) --
//      {"active": true, "pending": true, "plan": null, "error": null}
//   3. a ready plan -- its fields ride at the TOP LEVEL of the body, not
//      nested under `plan` (which is absent in this shape entirely) --
//      {"active": true, "pending": false, "error": null,
//       "my_slot": 2, "teams": 8, ..., "rounds_plan": [...], ...}
// Normalized here into one discriminated union so nothing downstream has to
// juggle `active`/`pending`/the top-level-vs-nested split itself.
export type PlanFetchResult =
  | { status: 'inactive' }
  | { status: 'pending'; error: string | null }
  // `error` is NOT "the plan failed" here -- it is the last REBUILD
  // attempt's own error, and can be set even on a `ready` result when a
  // previous good plan is still being served underneath it. Render it as a
  // small non-blocking notice beside the tab's content, never in place of
  // it.
  | { status: 'ready'; plan: LivePlan; error: string | null }

export async function fetchPlan(): Promise<PlanFetchResult> {
  const res = await fetch('/api/live/plan')
  if (!res.ok) {
    throw new Error(`Failed to load the draft plan (${res.status}): ${await detailText(res)}`)
  }
  const body = await res.json()
  const error = typeof body.error === 'string' ? body.error : null
  if (!body.active) return { status: 'inactive' }
  if (!Array.isArray(body.rounds_plan)) return { status: 'pending', error }
  return {
    status: 'ready',
    error,
    plan: {
      my_slot: body.my_slot,
      teams: body.teams,
      rounds: body.rounds,
      n_drafts: body.n_drafts,
      as_of_pick: body.as_of_pick,
      rounds_plan: body.rounds_plan,
      cliffs: Array.isArray(body.cliffs) ? body.cliffs : [],
      best_available: Array.isArray(body.best_available) ? body.best_available : [],
    },
  }
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

// -- the connect screen ---------------------------------------------------
//
// GET /api/live/connect-progress, polled while POST /api/live/connect-token
// is still blocked. The two are concurrent on purpose: the connect endpoint
// is a plain sync handler, so Starlette runs it in the threadpool and this
// one answers throughout (api/live.py's live_connect_progress explains the
// choice and what was measured). Nothing here is a client-side estimate --
// every value below is a value the server actually discovered.

// `warn` is "done, but not the way it was meant to be" and the connect
// carried on -- ESPN's settings unavailable so the saved roster was used, no
// team names, no slot yet. `failed` stopped the connect.
export type ConnectStageStatus = 'pending' | 'running' | 'ok' | 'warn' | 'failed'

export interface ConnectStage {
  key: string
  label: string
  status: ConnectStageStatus
  /** What this step discovered ("249 players", "you pick 2nd of 8"). Null
   *  until it has discovered it. */
  value: string | null
  /** Real measured duration, null while the stage is still pending/running. */
  ms: number | null
}

// Everything the connect learned, for the handoff screen's fact grid. Every
// field is optional because every one of them is only present once something
// actually discovered it -- a missing key means "not known", and the screen
// drops that cell rather than rendering a zero it made up.
export interface ConnectFacts {
  league_id?: string
  teams?: number
  rounds?: number
  scoring_format?: 'ppr' | 'half' | 'std'
  starters?: Record<string, number>
  flex_slots?: number
  bench?: number
  players?: number
  /** Managers fitted from this league's imported draft history; 0 means the
   *  cold start (the market prior), which is a different model, not a worse
   *  count. */
  managers?: number
  seasons?: number
  my_slot?: number
  team_names?: (string | null)[]
  my_team?: string | null
  /** Where the roster/scoring the board was built for came from, and the
   *  two fallbacks are NOT the same thing: 'saved' is this league's own
   *  imported settings, 'default' is the built-in cold-start league (8 teams,
   *  PPR, 15 rounds) that any league provisioned by this app falls back to,
   *  because `league` is a per-league table provisioning does not copy. Both
   *  decide the round count and every replacement level. */
  settings_source?: 'espn' | 'saved' | 'default'
}

export interface ConnectProgress {
  /** 'idle' = this helper has not connected since it started. */
  phase: 'idle' | 'connecting' | 'ready' | 'failed'
  stages: ConnectStage[]
  facts: ConnectFacts
  error: { stage: string; label: string; detail: string; hint: string | null } | null
  elapsed_ms: number
}

export async function fetchConnectProgress(): Promise<ConnectProgress> {
  const res = await fetch('/api/live/connect-progress')
  if (!res.ok) {
    throw new Error(`Failed to read connect progress (${res.status})`)
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
