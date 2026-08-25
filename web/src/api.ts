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
  // (sigma over mean, lower is steadier); `_pct` is its percentile among
  // EVERY measured player at his position, steadiest highest.
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
  /** nflverse's NFL.com CDN photo, carried on the board (scoring/board.py
   *  merges it) and served by /api/players like every other column. Null for
   *  a defense and for anyone the players table has no photo of -- 205 of
   *  250 rows have one. */
  headshot: string | null;
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
  /** This season's projection, in POINTS FOR THE SEASON under the league's
   *  own scoring -- the same column the candidate rows carry. Served by
   *  /api/players like every other board column; declared here because the
   *  recommendation cards rank positions by it. Nullable to stay compatible
   *  with `ProfileHeader`, which extends this shape and has a row for every
   *  player the profile can open, projected or not. */
  proj_points: number | null;
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
/** One player on THIS year's board scored against the player whose profile
 *  is open -- the Similar players card. Distinct from `SimilarPlayer` above,
 *  which is a SEASON from any year: these are people you can still draft, so
 *  every one of them carries a board rank and can be opened. `similarity` is
 *  a percentage on `scoring.similarity`'s own curve, the same one the stat
 *  twins are scored on. */
export interface SimilarPeer {
  player_id: string; name: string; similarity: number;
  /** nflverse's own NFL.com CDN url, the same one the popup's header photo
   *  comes from. Null for anyone the players table has no photo of. */
  headshot: string | null;
  proj_points: number | null; ppg: number | null;
  age: number | null; height: number | null; weight: number | null;
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
  /** The stat line behind `proj_ppg`, per game -- ESPN's projected carries,
   *  targets and yards for the season being drafted. Null for a player it
   *  does not project. Shares are deliberately absent; see
   *  scoring/profile._espn_projected_usage. */
  /** Same shape `scoring/profile._espn_projected_usage` returns, per game.
   *  `yards` is rushing plus receiving and `rush_yards` is the rushing half
   *  on its own (a quarterback's row); `tds` never folds a throw in, which
   *  is what `pass_tds` is for. */
  proj_usage: {
    games: number
    carries: number | null
    targets: number | null
    receptions: number | null
    yards: number | null
    rush_yards: number | null
    tds: number | null
    attempts: number | null
    pass_yards: number | null
    pass_tds: number | null
    interceptions: number | null
  } | null;
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
  // The board row this player sits on, plus the one thing only the profile
  // endpoint computes: where his projection places him among his position,
  // best first. Same shape as a played season's `pos_finish`, so the Finish
  // panel can put it on the same ladder. Null when he has no projection.
  header: Player & { proj_pos_finish: number | null };
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
  /** The players on this year's board most like him, best first. Null for a
   *  player the board does not carry at all, and ABSENT from a server that
   *  predates the card -- read it as optional, the same way `news` and
   *  `status` are read, so a running server one deploy behind renders a
   *  popup without the card rather than a popup that throws. */
  similar_players?: { season: number | null; position: string;
                      players: SimilarPeer[] } | null;
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
  /** What waiting costs in POINTS, priced at your own next turn: his value
   *  over the best player at his position expected to still be there when
   *  you pick again. Unweighted by roster need, unlike `gain_now`, which is
   *  the ranking quantity and is measured against a turn a full round out.
   *  Null exactly when `gain_now` is. */
  gain_next: number | null
  /** His points over the best OTHER player at his position expected at your
   *  next turn. `gain_next` counts the player himself, so on the clock a
   *  sure survivor reads 0 no matter how far ahead of his position he is --
   *  this is the figure the cards switch to while it is your pick. Null
   *  exactly when `gain_now` is (and on servers predating the field). */
  edge_next: number | null
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
  /** The season total this board ranks on. Not what the rail prints -- see
   *  `wk1_points`. */
  proj_points: number | null
  /** ESPN's week-1 projection, already in this league's scoring. Null for a
   *  player ESPN does not project that week, which the rail draws as a dash:
   *  a projected zero would be a claim that he will not score. */
  wk1_points?: number | null
}

export interface LiveState {
  active: boolean
  /** Whether this room can be drafted from, and what it costs if not. Comes
   *  down with every poll (api/live.py's `_billing_state`), so the moment a
   *  payment lands the buttons come alive on the next one. `enabled: false`
   *  is every instance nobody has put a Stripe key on -- the room behaves
   *  exactly as it always has. */
  billing?: BillingStatus & { league_id?: string; season?: number }
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
  /** nflverse's photo url, or null -- a defense, a player it has no
   *  biography for, or a database refreshed before the column existed. */
  headshot?: string | null
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

// Who put a pick on the board. Only a recorded mock draft carries this
// (see fetchMockBoard) -- ESPN's own draft socket says who is on the clock,
// not whether a person is behind the seat, so the labels are recorded by
// the farm as it watches a room, and no room it did not watch can ever be
// labelled.
//   human   -- a person chose this
//   auto    -- the seat HAD a person, but ESPN was picking for them
//   engine  -- the seat never had a person at all
//   us      -- our own bot's seat
//   unknown -- not recorded (every draft from before the farm labelled them)
export type PickMaker = 'human' | 'auto' | 'engine' | 'us' | 'unknown'

// One landed pick, already placed at its `round`/`slot` by the server --
// the grid trusts these rather than re-deriving them from `overall` and the
// team count, so it never has to know this league's snake variant.
export interface BoardCell {
  overall: number
  round: number
  slot: number
  player: BoardPlayer
  // ABSENT from /api/live/board, which never labels a pick. Optional rather
  // than `| undefined` so the live room's own board still typechecks
  // untouched, and read everywhere as "missing means `unknown`".
  made_by?: PickMaker
}

// One column header. `is_me` marks the viewer's own team -- there is
// exactly one `true` among `teams` columns whenever `my_slot` is non-null.
export interface BoardColumn {
  slot: number
  team_name: string
  is_me: boolean
  // Whether a person ever sat in this seat, over the whole draft -- true
  // even if they wandered off and ESPN finished the column for them.
  // `null` for a draft with no labels at all; ABSENT from /api/live/board,
  // same as `made_by` above.
  had_owner?: boolean | null
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

/** The connect was refused because this draft has not been paid for.
 *
 *  Its own class, not a message: the page has to draw a price and a button
 *  rather than an error, and it needs the league and season to offer checkout
 *  for the draft that was actually refused. See api/billing.py. */
export class PaymentRequired extends Error {
  leagueId: string
  season: number
  constructor(leagueId: string, season: number, message: string) {
    super(message)
    this.name = 'PaymentRequired'
    this.leagueId = leagueId
    this.season = season
  }
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
  // 402 is the one refusal that is not a failure: the request was fine and
  // the draft costs money. It arrives with a structured body (the league and
  // the season) because the page's next move is a checkout for that draft.
  if (res.status === 402) {
    const body = await res.json().catch(() => null)
    const detail = body?.detail ?? {}
    throw new PaymentRequired(
      String(detail.league_id ?? params.leagueId),
      Number(detail.season ?? params.season),
      detail.message ?? 'This draft has not been paid for.')
  }
  if (!res.ok) {
    throw new Error(await detailText(res))
  }
  return res.json()
}

// -- billing ---------------------------------------------------------------
//
// Mock drafts are free and real ones are $9.99 for the season (api/billing.py).
// An instance with no Stripe key sells nothing, and every one of these
// answers says so rather than pretending: `enabled: false` is the local
// checkout, the test suite, and anybody running this themselves.

export interface BillingStatus {
  /** Whether this instance sells anything at all. */
  enabled: boolean
  /** Whether THIS draft has to be paid for before it can be connected. */
  required: boolean
  /** Whether it already has been. True for every mock. */
  entitled: boolean
  reason?: string
}

export async function fetchBillingStatus(
  leagueId: string, season?: string | number | null,
): Promise<BillingStatus> {
  const query = new URLSearchParams({ leagueId })
  if (season) query.set('season', String(season))
  const res = await fetch(`/api/billing/status?${query}`)
  // Unreadable means "do not put a paywall in front of anybody": the gate on
  // the server is the thing that actually decides, and this endpoint only
  // decides what to draw.
  if (!res.ok) return { enabled: false, required: false, entitled: true }
  return res.json()
}

/** Open a Stripe Checkout for one draft. Returns the URL to send them to.
 *
 *  IN A POPUP, at the call site, rather than by navigating this tab. The
 *  bookmarklet's token lives in memory on the page that received it (it is
 *  wiped from the address bar on arrival, deliberately), so navigating away
 *  to Stripe would destroy the one copy of it and cost the user a second trip
 *  through the bookmark after paying. */
export async function startCheckout(
  leagueId: string, season?: string | number | null, returnTo?: string,
): Promise<string> {
  const res = await fetch('/api/billing/checkout', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      leagueId,
      season: season ? String(season) : null,
      returnTo: returnTo ?? '/',
    }),
  })
  if (!res.ok) throw new Error(await detailText(res))
  const body = await res.json()
  return String(body.url)
}

// -- the live mock draft the landing page opens with -----------------------
//
// `pipeline/mock_farm.py` sits in real ESPN mock drafts to build the corpus
// the model is fitted on, and those rooms are running while somebody reads
// the page. So the hero is the product against picks that landed seconds ago
// -- not a screenshot of it. See api/demo.py.

/** One roster slot as the panel reads it. Declared here rather than imported
 *  from the component so `api.ts` stays the one description of what the
 *  server sends. */
export interface RosterSlotPayload {
  slot: string
  player: RosterPlayer | null
  urgent: boolean
}

export interface LiveMockPick {
  pick_no: number
  slot: number
  name: string | null
  position: string | null
  team: string | null
  /** Nobody chose him; the clock did. Worth showing, because a room where
   *  every pick is autodrafted is weaker evidence than one with seven people
   *  in it. */
  autodrafted: boolean
}

/** One row of the room's ranked list, exactly as `/api/live/state` serves it
 *  -- `LiveCandidate`'s fields, plus the few the landing page shows before
 *  the join table lands. Same shape on purpose: the page hands these to the
 *  real `AvailableList`, and a second shape would be a second product. */
export interface LiveMockCandidate {
  player_id: string
  position: string | null
  proj_points: number | null
  vor_points: number | null
  gain_now: number | null
  gain_next: number | null
  edge_next: number | null
  survive_pct: number | null
  fills: string | null
  rank: number | null
  name: string | null
  team: string | null
  bye: number | null
  adp: number | null
  vs_adp: number | null
  board_rank: number | null
}

export interface LiveMock {
  /** Whether there is a room to draw at all -- NOT whether it is happening
   *  this second. That is `mode`. */
  live: boolean
  /** `live` while a draft is going on right now; `replay` while one from
   *  the archive is played back at the pace its picks were really made (the
   *  farm's rooms end, and the next is minutes away); `none` when there is
   *  neither. The room's status pill says which. */
  mode?: 'live' | 'replay' | 'none'
  /** The server has not finished its first build yet -- seconds, not
   *  minutes, and a different thing from "no draft is running". The room
   *  says which, because one of them is worth waiting through. */
  warming?: boolean
  /** The grid and the on-clock seat's roster, in `/api/live/board`'s and the
   *  roster panel's own shapes -- so the landing page runs the room's real
   *  components against them rather than a second rendering. */
  board?: LiveBoard | null
  roster?: RosterSlotPayload[] | null
  /** Whether the picks are where that room is RIGHT NOW. False means the same
   *  real draft, rewound to a round where the board had something to say --
   *  a room in its fifteenth round is four defences and a kicker. */
  live_moment?: boolean
  rooms?: number
  teams?: number
  rounds?: number
  humans?: number | null
  picks_made?: number
  picks_total?: number
  round?: number | null
  on_the_clock?: number | null
  /** The open turn's pick clock, in seconds, exactly as ESPN stated it when
   *  it opened the turn -- never assumed. Null when this room's clock is not
   *  knowable (a farm that never saw a turn open, or a rewound room), in
   *  which case the page shows no timer rather than a wrong one. */
  clock_seconds?: number | null
  /** When that turn opened, epoch seconds on the SERVER's clock: the moment
   *  the last pick landed. Read against `server_now`, never against the
   *  browser's own clock, which is routinely minutes out. */
  turn_started_at?: number | null
  /** The server's clock when this answer was built. The page subtracts to
   *  learn how far into the turn the room already was, then counts from
   *  there with its own monotonic elapsed time. */
  server_now?: number | null
  /** The room's league shape, so the list draws finish tiers against the
   *  league these prices were made for. */
  settings?: LiveSettings | null
  recent?: LiveMockPick[]
  shortlist?: LiveMockCandidate[]
}

export async function fetchLiveMock(): Promise<LiveMock> {
  const res = await fetch('/api/demo/live')
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}

// -- drafts you can join without the bookmarklet ---------------------------
//
// The bookmarklet exists because ESPN's `draftSecurity` endpoint needs the
// account session cookie, and only a browser sitting on ESPN's own origin has
// it. When the helper HOLDS a session -- this machine's saved login, or an
// account connected through custody -- the server can make that same call
// itself, which turns "find the bookmark, click it, click it again when the
// token expires" into a list with a button on it. See api/drafts.py.

/** One league this account is in whose draft has not finished. */
export interface UpcomingDraft {
  league_id: string
  /** This account's own team in that league -- what makes the row joinable
   *  rather than merely informative. */
  team_id: string | null
  team_name: string | null
  season: number | null
  name: string | null
  teams: number | null
  draft_type: string | null
  /** ISO 8601, or null for a league whose commissioner has not set a date. */
  draft_at: string | null
  /** The room is open now: ESPN says the draft is in progress, or its hour
   *  has passed and it has not finished. */
  live: boolean
}

/** `connected: false` is the ordinary answer, not an error -- most visitors
 *  have connected nothing. `expired` distinguishes "never connected" from
 *  "connected, and ESPN has since rejected it", which are different sentences
 *  to show a user. */
export interface UpcomingDrafts {
  connected: boolean
  source?: 'connected' | 'local'
  season?: number
  expired?: boolean
  leagues: UpcomingDraft[]
}

/** Hand the helper an ESPN account session and get this browser's key.
 *
 *  The session comes from the bookmarklet, which is the only code that can
 *  read it: it runs on ESPN's origin, and `sameSite=Lax` keeps the cookie off
 *  every fetch this page could make. What comes back is the league list, so
 *  connecting and knowing what to do next are one round trip -- and an
 *  `httpOnly` cookie the browser stores and this code never sees. */
export async function connectEspnAccount(
  swid: string, espnS2: string,
): Promise<UpcomingDrafts> {
  const res = await fetch('/api/espn/connect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ swid, espn_s2: espnS2 }),
  })
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}

/** Log this browser out of the stored ESPN session.
 *
 *  The credential stays for any other browser holding it -- "everywhere" is a
 *  separate control (`/api/espn/custody/disconnect-everywhere`), because one
 *  browser deciding for all of them is how somebody loses a draft they have
 *  open on another machine.
 *
 *  Answers 401 when there was nothing to disconnect, which is not an error
 *  worth showing: either way this browser now holds no session, and the
 *  server clears the cookie on both paths. */
export async function disconnectEspn(): Promise<void> {
  await fetch('/api/espn/custody/disconnect', { method: 'POST' })
}

/** Whether THIS browser holds a stored ESPN session.
 *
 *  A local lookup -- no ESPN call, no board work -- so it is the right probe
 *  for anything that only needs to know whether to offer a way in. It answers
 *  200 either way; `connected: false` is the ordinary case. */
export async function fetchCustody(): Promise<{ connected: boolean }> {
  const res = await fetch('/api/espn/custody')
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}

export async function fetchUpcomingDrafts(): Promise<UpcomingDrafts> {
  const res = await fetch('/api/espn/drafts')
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}

/** Mint one draft token server-side, the way the bookmarklet would.
 *
 *  Returns the same four values the bookmarklet puts in the URL hash, so the
 *  caller hands them straight to `connectWithToken` -- there is one connect
 *  path in this app and this only changes where the token came from. */
export async function mintDraftToken(
  leagueId: string, teamId: string, season?: number | null,
): Promise<TokenConnectParams> {
  const res = await fetch('/api/espn/draft-token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ leagueId, teamId, season: season ? String(season) : null }),
  })
  if (!res.ok) throw new Error(await detailText(res))
  const body = await res.json()
  return {
    leagueId: String(body.leagueId), teamId: String(body.teamId),
    swid: String(body.swid), token: String(body.token),
    season: String(body.season),
  }
}

// -- open ESPN mock rooms, and taking a seat in one ------------------------
//
// ESPN runs a public mock-draft lobby: a new room every few minutes, all
// night, free. `api/lobby.py` reads the directory (cookie-free -- it is
// public), and `POST /api/espn/mock-join` takes a seat as the signed-in
// account and mints for it. See api/drafts.py's mock_join.

/** One open room in ESPN's mock lobby. No name: ESPN's directory does not
 *  give rooms one, so a row is identified by its shape and its clock. */
export interface MockRoom {
  league_id: string
  league_size: number | null
  teams_joined: number
  /** ESPN's own words: `PPR`, `STANDARD`. */
  scoring: string | null
  draft_type: string | null
  /** `PRO` / `EXPERT` / `BEGINNER`, or null on a row that does not say. */
  experience: string | null
  /** Seconds until picking starts. Never negative -- the server clamps. */
  starts_in_seconds: number | null
  /** How this room's seat count last MOVED, signed: +2 is two seats taken
   *  between two reads of ESPN's directory, -1 is somebody leaving. Zero for
   *  a room that has not been seen to change. */
  seats_delta: number
  /** Seconds since that move, or null for a room never seen to move. Null is
   *  NOT zero: a room first read at 6/10 did not fill six seats in front of
   *  anybody (api/lobby.py's `_track_seats`). */
  seats_changed_seconds: number | null
  /** How long the server has been watching this room. It is what makes
   *  silence readable -- no change over eight seconds is nothing, over eight
   *  minutes it is a room nobody is joining. */
  watched_seconds: number
}

export interface MockRooms {
  rooms: MockRoom[]
  /** How many rooms the lobby held, before the server's cap. A page showing
   *  120 of 137 can say so instead of implying it has them all. */
  total: number
}

/** `available: false` is ESPN being unreadable, and it is not an error worth
 *  showing: the section draws nothing. Same rule as the lobby summary. */
export async function fetchMockRooms(): Promise<MockRooms> {
  const res = await fetch('/api/lobby/rooms')
  if (!res.ok) return { rooms: [], total: 0 }
  const body = await res.json()
  return body.available && Array.isArray(body.rooms)
    ? { rooms: body.rooms, total: Number(body.total ?? body.rooms.length) }
    : { rooms: [], total: 0 }
}

/** Take a seat in one room and mint the token for it.
 *
 *  Returns exactly what `mintDraftToken` does, because the seat is the only
 *  difference: one connect path in this app, whether the draft is a real
 *  league's or a mock room's. */
export async function joinMockRoom(
  leagueId: string, season?: number | null, teamId?: string | null,
): Promise<TokenConnectParams> {
  const res = await fetch('/api/espn/mock-join', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      leagueId,
      season: season ? String(season) : null,
      // Absent means "any open seat" -- ESPN's own sentinel, applied server
      // side. A seat id here is one ESPN honours (probed): the reader gets the
      // seat they picked or a refusal, never a different one.
      teamId: teamId ?? null,
    }),
  })
  if (!res.ok) throw new Error(await detailText(res))
  const body = await res.json()
  return {
    leagueId: String(body.leagueId), teamId: String(body.teamId),
    swid: String(body.swid), token: String(body.token),
    season: String(body.season),
  }
}

// -- one mock room, while it fills up -------------------------------------
//
// The waiting room reads this: who is in each seat, when picking starts, and
// whether it already has. Public (ESPN answers a room read with no cookies),
// so the only thing the session decides is which seat is marked as yours.

export interface RoomSeat {
  team_id: string
  /** Position in the draft order -- NOT the team id, which only happens to
   *  match it in a fresh room. What every pick number is computed from. */
  slot: number
  name: string
  taken: boolean
  mine: boolean
}

export interface MockRoomState {
  league_id: string
  teams: number
  draft_type: string | null
  /** Seconds per pick once it starts. */
  clock_seconds: number | null
  /** Rounds this room drafts, from the pick list ESPN pre-populates before a
   *  ball is drafted -- not a roster count, which includes an IR slot nobody
   *  drafts into. */
  rounds: number | null
  draft_at: string | null
  /** Counted by the server, so a browser clock that is minutes off does not
   *  make the countdown wrong. */
  starts_in_seconds: number | null
  in_progress: boolean
  drafted: boolean
  /** The seat this session owns, or null while it owns none. */
  my_team_id: string | null
  seats: RoomSeat[]
}

/** How far along one room is -- what the Home page's live cards say. */
export interface RoomProgress {
  /** Null when nobody on this side is in the room to hear the picks land --
   *  ESPN's public read names no player until the draft is over. */
  picks_made: number | null
  picks_total: number | null
  teams: number
  rounds: number | null
  /** The pick that is UP, in round terms: 37 made means round 5, pick 6.
   *  Null with `picks_made`. */
  round: number | null
  pick_in_round: number | null
  in_progress: boolean
  drafted: boolean
}

/** One request for every room the reader is drafting in. A room ESPN would
 *  not read is simply absent from the answer. Never throws: a card that
 *  cannot say its round keeps saying "drafting". */
export async function fetchRoomProgress(
  ids: string[],
): Promise<Record<string, RoomProgress>> {
  if (ids.length === 0) return {}
  const res = await fetch(`/api/espn/rooms/progress?ids=${encodeURIComponent(ids.join(','))}`)
  if (!res.ok) return {}
  const body = await res.json()
  return body && typeof body.rooms === 'object' ? body.rooms : {}
}

export async function fetchMockRoom(
  leagueId: string, season?: number | null,
): Promise<MockRoomState> {
  const query = season ? `?season=${season}` : ''
  const res = await fetch(`/api/espn/mock-room/${encodeURIComponent(leagueId)}${query}`)
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}

// -- who usually goes at a pick -------------------------------------------
//
// The waiting room's question: a seat is a list of pick numbers, and what
// separates one seat from another is who is on the board when they come
// round. Answered from the recorded mock corpus (api/market.py), human picks
// only -- an autodrafted seat is ESPN's ADP list read aloud.

export interface PickName {
  player_id: string
  name: string | null
  headshot: string | null
  position: string
  count: number
  /** Of the picks observed at this pick number, the fraction that were him. */
  share: number
  /** The board's projected season total for him -- divide by SEASON_GAMES for
   *  the per-game figure the rest of the app prints. Null where the board
   *  could not be built, which costs this number and nothing around it. */
  proj_points: number | null
  /** Per-game delta on his recent seasons (board's proj_change). Null where
   *  the board could not answer, or he has no seasons to compare. */
  proj_change: number | null
}

export interface PickHistory {
  pick_no: number
  /** How many recorded drafts had a person make this pick. 0 means the
   *  archive never reached it -- a bigger room's late picks run past its own
   *  last pick -- and the page says "no record" rather than inventing one. */
  observed: number
  top: PickName[]
}

export interface PicksHistory {
  picks: PickHistory[]
  /** What it was counted from, so the page can attribute it: the archive is
   *  8-team mocks, and a 14-team room is not its own history. */
  drafts: number
  teams: number
  rounds: number
}

export async function fetchPicksHistory(picks: number[]): Promise<PicksHistory> {
  const res = await fetch(`/api/market/at-picks?picks=${picks.join(',')}`)
  if (!res.ok) throw new Error(await detailText(res))
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

// -- the live mock-draft lobby (proof, next to the CTA) --
//
// ESPN's own public mock-draft lobby, proxied and cached by api/lobby.py
// (~60s TTL server-side; see that module's docstring for why a cache sits
// in front of it at all). No cookies flow on either leg of this call.

/** One upcoming, joinable room. `league_size`/`scoring` are `null` only if
 *  ESPN ever omits the field this server reads them from -- ordinary
 *  optional-chaining territory for LobbyStrip, not a sign of a bad row. */
export interface LobbyRoom {
  league_size: number | null
  scoring: string | null
  teams_joined: number
  starts_in_seconds: number | null
}

export interface LobbySummary {
  /** False whenever ESPN couldn't be read or answered with something this
   *  server didn't recognise -- every other field is a placeholder in that
   *  case. LobbyStrip renders nothing at all when this is false; see its
   *  own module comment for why that's a hard requirement. */
  available: boolean
  total_open: number
  joinable: number
  /** Soonest-starting joinable rooms first, already capped server-side. */
  upcoming: LobbyRoom[]
}

export async function fetchLobby(): Promise<LobbySummary> {
  const res = await fetch('/api/lobby')
  if (!res.ok) throw new Error(`Failed to load the mock lobby (${res.status})`)
  return res.json()
}

// -- the mock draft farm (/mocks) --

// One room the farm has joined: still playing (`live`) or finished
// (`complete`). `human_seats` is how many of the `teams` seats ever held a
// person, and is null for a draft recorded before the farm tracked that --
// the same "not recorded" that BoardCell.made_by spells `unknown`, which is
// why it is null and not 0: nobody home and nobody counted are different
// facts. `my_slot` is our own bot's seat.
export interface MockDraft {
  id: string
  // Null for a room whose league the farm never resolved -- see api/mocks.py.
  league_id: string | null
  status: 'live' | 'complete'
  teams: number
  rounds: number
  picks_made: number
  human_seats: number | null
  my_slot: number | null
  // Set only while the room is still playing (`status: 'live'`).
  started_at: string | null
  // Set only once the draft is done and the farm has written it out
  // (`status: 'complete'`) -- when that happened, not when the draft did.
  // `started_at` and `recorded_at` are never both set on the same row.
  recorded_at: string | null
}

// Live rooms first, then finished ones, newest first -- the server's own
// order, kept as it arrives.
export async function fetchMockDrafts(): Promise<MockDraft[]> {
  const res = await fetch('/api/mocks')
  if (!res.ok) {
    throw new Error(`Failed to load the mock drafts (${res.status}): ${await detailText(res)}`)
  }
  const body = await res.json()
  return Array.isArray(body.drafts) ? body.drafts : []
}

// The same shape the live room's board comes in, so one grid draws both --
// plus the `made_by`/`had_owner` labels declared as optional above, which
// only this endpoint fills in.
export async function fetchMockBoard(id: string): Promise<LiveBoard> {
  const res = await fetch(`/api/mocks/${encodeURIComponent(id)}/board`)
  if (!res.ok) {
    throw new Error(`Failed to load that draft board (${res.status}): ${await detailText(res)}`)
  }
  return res.json()
}

// -- the draft archive ------------------------------------------------------
//
// Hundreds of recorded ESPN drafts, asked the questions an average draft
// position cannot answer: what the room does at each of YOUR turns, how
// decided each of those turns is, the position paths people actually walk
// from your seat, and the SHAPE of where a player goes rather than his mean.
// See api/market.py for whose picks count and why. Signed in only -- a 403
// here is the ordinary answer for a browser with no account, not a fault.

export interface MarketOverview {
  drafts: number
  picks: number
  /** Picks made by a person: ESPN's autodrafter and this project's own farm
   *  seat are both excluded from every behaviour question. */
  human_picks: number
  teams: number
  rounds: number
  from: string | null
  to: string | null
  median_humans: number | null
}

export interface MarketTurnPlayer {
  player_id: string
  name: string | null
  headshot: string | null
  position: string
  count: number
  /** Of the times a person held this turn, the share who took him. */
  share: number
  /** The board's projected season total -- divide by SEASON_GAMES for the
   *  per-game number this app prints everywhere. Null when the board cannot
   *  price him. */
  proj_points: number | null
  proj_change: number | null
}

export interface MarketTurn {
  pick_no: number
  round: number
  /** How many times a person actually held this turn. Thins out in the late
   *  rounds, where seats go to the clock, and the page says so. */
  observed: number
  distinct: number
  players: MarketTurnPlayer[]
  positions: { position: string; share: number }[]
  /** The share of the single most common pick: how spoken-for the turn is. */
  top_share: number
}

export interface MarketSlot {
  slot: number
  teams: number
  rounds: number
  turns: MarketTurn[]
  sequences: { path: string[]; count: number; share: number }[]
  sequence_rounds: number
  sequences_observed: number
}

export interface MarketPlayer {
  player_id: string
  name: string | null
  headshot: string | null
  position: string
  team: string | null
  /** Drafts he was on the board for -- the denominator for everything else. */
  drafts: number
  taken: number
  taken_share: number
  earliest: number | null
  p10: number | null
  median: number | null
  p90: number | null
  latest: number | null
  espn_rank: number | null
  adp: number | null
  /** Share of drafts he went in each bin of the pick axis, one bin a round. */
  hist: number[]
  /** Share of drafts he was still on the board at each of this seat's turns,
   *  counted rather than simulated -- the archive's answer to the room's
   *  LASTS column. */
  survive: number[]
}

export interface MarketPlayers {
  slot: number
  teams: number
  rounds: number
  /** This seat's own pick numbers, in order. */
  turns: number[]
  bins: number
  picks_total: number
  players: MarketPlayer[]
}

// Open to anybody: the archive holds public mock drafts played by strangers,
// aggregated, with no credential and no account attached to a seat. See
// api/market.py.
async function archive<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(await detailText(res))
  return res.json()
}

export const fetchMarketOverview = () =>
  archive<MarketOverview>('/api/market/overview')

export const fetchMarketSlot = (slot: number) =>
  archive<MarketSlot>(`/api/market/slot/${slot}`)

export const fetchMarketPlayers = (slot: number) =>
  archive<MarketPlayers>(`/api/market/players?slot=${slot}`)
