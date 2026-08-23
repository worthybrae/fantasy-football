import type {
  DepthChartGroup, GameLogRow, Player, PlayerProfileData, ScheduleWeek, SeasonSummary,
} from '../../api'

// The player-card payload as `scoring/profile.py::build_profile` actually
// serves it today, which is more than `api.ts`'s `PlayerProfileData`
// describes: commit 7d7e7d1 added `bio`, `cohort`, `oline`, six keys on each
// season row, `snap_pct` on each game-log row and `rank`/`rank_n` on each
// schedule week, all of them additively and none of them typed on the way
// in.
//
// Declared HERE rather than merged into `api.ts` on purpose: that file is
// the shared contract for every view in the app and the payload is being
// extended by a second, concurrent change (player news and injury status).
// Two changes editing the same forty lines of interface for two unrelated
// reasons is a merge conflict for no benefit -- the card that reads these
// keys is the only thing that needs them named. If the card outlives that
// concurrency, folding this file into `api.ts` is a copy-paste.
//
// EVERY FIELD BELOW WAS READ OFF A REAL PAYLOAD, not off the design, and
// the two disagree in places (the design's o-line figures are a season old
// and its cohort caption is hand-written). Nothing here is defaulted or
// invented: a value the payload nulls is rendered as missing, not as zero.

/** One season row: `SeasonSummary` plus the six the redesign asked for. */
/** ESPN's projected stat line for the season being drafted, per game. No
 *  shares: the projection table has no trustworthy team total to divide by
 *  and does not project snaps at all -- see
 *  scoring/profile._espn_projected_usage. Null for a player it does not
 *  project, and any single rate is null where his position has no such stat. */
export interface ProjectedUsage {
  games: number
  carries: number | null
  targets: number | null
  receptions: number | null
  yards: number | null
  attempts: number | null
  pass_yards: number | null
}

/** Where each usage number places among the same position that season, 0-1,
 *  higher better. Null for a season the pool could not rank -- a snap table
 *  with no row for him, or a rate his position does not have. */
export interface SeasonPercentiles {
  snap_share: number | null
  target_share: number | null
  carries_pg: number | null
  targets_pg: number | null
  receptions_pg: number | null
  yards_pg: number | null
}

export interface SeasonRow extends SeasonSummary {
  pcts?: SeasonPercentiles
  /** Age on September 1 of THAT season (not today) -- see `player_bio`. */
  age: number | null
  /** 1-based: a rookie year is his 1st NFL season. */
  nfl_season: number | null
  /** Positional finish by POINTS PER GAME, and the pool it is out of.
   *  `pos_finish` (already on `SeasonSummary`) is by total points -- the two
   *  are different facts and a 12-game season separates them sharply. */
  pos_rank_ppg: number | null
  pos_rank_ppg_n: number | null
  /** Coefficient of variation: week-to-week spread divided by the average.
   *  The Steady panel ranks on this rather than on `ppg_std` because
   *  volatility is scored per point -- a bigger scorer is not punished for
   *  scoring. */
  cv: number | null
  cv_rank: number | null
  cv_rank_n: number | null
  /** The same position-season's median CV, so "0.63" has something to be
   *  read against. */
  cv_pos_median: number | null
}

/** One game-log row plus the two shares that say how much of the offence he
 *  was that week. Both null on a DNP -- he took no snaps and saw no targets
 *  because he did not play, which is not a share of zero -- and
 *  `target_pct` is null too where the team's total for that week cannot be
 *  built at all. */
export interface GameRow extends GameLogRow {
  snap_pct: number | null
  target_pct: number | null
}

/** One schedule week plus the league rank the owner asked for.
 *  DIRECTION, because it inverts the intuition: rank 1 is the SOFTEST
 *  defence -- the one that gave up the most to this position last season --
 *  out of `rank_n` (32 here). Every label that renders it says so. */
export interface ScheduleRankWeek extends ScheduleWeek {
  rank: number | null
  rank_n: number | null
}

/** Age and service time as of the season being drafted. All-null for a
 *  player the `players` table has no row for: every defense, and any rookie
 *  nflverse has no biography for yet. */
export interface Bio {
  /** nflverse's photo url, or null -- for a defense, a player it has no
   *  biography for, or a database that predates the column. The card draws
   *  no image rather than a broken one. */
  headshot?: string | null
  season: number
  birth_date: string | null
  rookie_season: number | null
  age: number | null
  nfl_season: number | null
}

export interface CohortSeason {
  player_id: string
  name: string
  season: number
  nfl_season: number | null
  ppg: number
  next_ppg: number
  /** next_ppg - ppg. Negative is a decline. */
  change: number
}

/** Seasons like this one, and what they became. The band is served with the
 *  cohort so the caption can state the recipe instead of asserting it. */
export interface Cohort {
  season: number
  position: string
  ppg: number
  nfl_season: number | null
  ppg_band: number
  exp_band: number | null
  min_games: number
  n: number
  median_change: number | null
  declined: number
  improved: number
  players: CohortSeason[]
}

/** The team's offensive line. `rank` is 1 = best, out of `teams`.
 *  Null for a defense by construction (`team_line_quality`): the o-line is a
 *  fact about the eleven players who leave the field when the defense comes
 *  on. Kickers keep it -- their attempts come from their own offence moving
 *  the ball. */
export interface LineQualityData {
  season: number
  team: string
  rank: number
  teams: number
  line_quality: number | null
  /** All three of these are shares of one; `experience` is mean seasons in
   *  the league, a different unit, which is why the card never ranks them
   *  against each other. */
  continuity: number | null
  availability: number | null
  returning: number | null
  experience: number | null
}

/** The board row the profile is built from. `api.ts`'s `Player` is the
 *  subset /api/players serves to the board table; the profile header is the
 *  whole row, which also carries the projection and the value over
 *  replacement the popup's four figures lead with (plus `composite`,
 *  `proj_scale`, `espn_id`, `ffc_rank` and the five raw factors, none of
 *  which this card reads). */
export interface ProfileHeader extends Player {
  proj_points: number | null
  /** Where the projection places him among his own position, best first --
   *  the same shape `pos_finish` gives a season he has played, so the Finish
   *  panel can draw it as one more column on the ladder the played seasons
   *  are already on. Null when the board carries no projection for him. */
  proj_pos_finish: number | null
  vor: number | null
}

/** One row of the room's own ranked board, as `/api/players` serves it --
 *  `Player` satisfies it by construction, so the room hands its join table
 *  straight over without building a second shape.
 *
 *  The popup needs the WHOLE list because the payload can only name a
 *  player's board neighbours when it has no stat line to match him on
 *  (`similar.mode === 'value_neighbors'`). For everyone else `similar` is
 *  stat twins, whose `rank` is where those players sit on TODAY's board --
 *  Alvin Kamara at 273 -- and has nothing to do with where this one does.
 *  Read by nothing today: the card that windowed the board around a
 *  player was removed. Kept because the room still computes it. */
export type RankedPlayer = Pick<Player, 'player_id' | 'name' | 'rank' | 'market_rank'>

/** One headline. `attribution` is the whole reason this is a record and not
 *  a (headline, url) pair: 'espn_athlete_id' means ESPN tagged the article
 *  with this player's athlete id, 'name_team_query' means a name+team search
 *  returned it and we believe it is about him. Those are different claims
 *  and the card says which is which rather than flattening a good guess into
 *  a fact. Timestamps are ISO-8601 strings, or null. */
export interface NewsItem {
  headline: string
  url: string
  published_at: string | null
  source: string | null
  attribution: string
  fetched_at: string | null
}

/** Sleeper's injury and depth signals, or null for a player it has no row
 *  for (every defense). NOTHING HERE EVER SAYS HEALTHY: `injury_status` is
 *  null for a fit player, so the absence of a designation is the absence of
 *  a claim, not a clean bill of health. */
export interface PlayerStatus {
  injury_status: string | null
  injury_body_part: string | null
  injury_notes: string | null
  depth_chart_position: string | null
  depth_chart_order: number | null
  news_updated: string | null
  fetched_at: string | null
  source: string
}

/** The attribution value that means ESPN tagged the article itself
 *  (pipeline/news.py's ATTR_EXACT). Anything else is a name match. */
export const ATTR_EXACT = 'espn_athlete_id'

export type ProfilePayload =
  Omit<PlayerProfileData, 'header' | 'seasons' | 'game_log' | 'schedule'> & {
    header: ProfileHeader
    seasons: SeasonRow[]
    game_log: GameRow[]
    schedule: ScheduleRankWeek[]
    bio: Bio
    cohort: Cohort | null
    oline: LineQualityData | null
    // OPTIONAL, and read as such everywhere: `news` and `status` are being
    // added to this payload by a change landing alongside this card. A card
    // that renders them when they are there and says nothing when they are
    // not works against both versions of the server -- and against a
    // database where `make refresh` has not written the two tables yet,
    // which is the state data/nfl.duckdb is in as this is written. `news` is
    // an empty list for a player nobody wrote about (and for every defense);
    // `status` is null for a player Sleeper has no row for.
    news?: NewsItem[]
    status?: PlayerStatus | null
  }

/** The card has a history to show at all. Deliberately a test of the DATA,
 *  never of the position: a defense has no weekly rows ever, but a kicker
 *  has them exactly when the league prices kicking (see `build_profile`'s
 *  `prices_kicking` branch), and a rookie has none yet whatever he plays.
 *  Branching on position instead would show a kicker six empty panels in a
 *  league that scores field goals, and would have to be found and changed
 *  again the next time the scoring rules move. */
export function hasHistory(p: ProfilePayload): boolean {
  return p.seasons.length > 0 || p.game_log.length > 0
}

/** "+3" / "-3" / "—" -- the app's one spelling of a signed whole number
 *  (see playerSeed.ts, AvailableList.tsx, TopThree.tsx for the others). */
export function fmtSigned(n: number | null | undefined, digits = 0): string {
  if (n === null || n === undefined) return '—'
  const r = digits === 0 ? Math.round(n) : Number(n.toFixed(digits))
  const body = digits === 0 ? String(Math.abs(r)) : Math.abs(r).toFixed(digits)
  if (r > 0) return `+${body}`
  if (r < 0) return `−${body}`
  return digits === 0 ? '0' : (0).toFixed(digits)
}

/** One decimal only when the number isn't whole -- ADP consensus is 1.8 but
 *  a single source is 1. Same rule as AvailableList's own fmtRank. */
export function fmtRank(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—'
  return Number.isInteger(n) ? String(n) : n.toFixed(1)
}

/** 21 -> "21st". Used wherever a rank is stated in prose ("21st of 32"). */
export function ordinal(n: number): string {
  const rem100 = n % 100
  if (rem100 >= 11 && rem100 <= 13) return `${n}th`
  switch (n % 10) {
    case 1: return `${n}st`
    case 2: return `${n}nd`
    case 3: return `${n}rd`
    default: return `${n}th`
  }
}

/** His room, or null if the chart has none for him. Whichever group actually
 *  holds him comes before the one his board position names: a player charted
 *  somewhere other than where the board ranks him (a receiver taking snaps at
 *  running back) belongs in the room he is actually competing in. The
 *  fallback is not a nicety -- `is_me` is false for every row of a player the
 *  chart could not be joined to by id, which is every rookie who is on the
 *  board under an `adp_` id, and Sleeper still charts him by name.
 *
 *  Here rather than in DepthChartCard because the thin-payload notice
 *  (MissingData) tells the reader the room is one of the real things on his
 *  popup, and the card and the notice have to give the same answer. */
export function depthGroup(groups: DepthChartGroup[], position: string): DepthChartGroup | null {
  const mine = groups.find((g) => g.players.some((p) => p.is_me))
    ?? groups.find((g) => g.position === position)
  return mine && mine.players.length > 0 ? mine : null
}
