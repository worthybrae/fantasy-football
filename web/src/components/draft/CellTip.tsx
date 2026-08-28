// Hover panels for the three columns whose meter is a summary of something a
// reader may want the detail of: Health, the 2025 sparkline, and Finish. Each
// meter answers "how much"; each panel answers "out of what".
//
// Each shows ONE thing. The first version put a full stat line on every row of
// the 2025 and Finish panels -- true, and unreadable at a glance, which is the
// only way a panel that opens on hover is ever read. The stat lines live on
// the player card, which is where someone who wants them has gone looking.
//
// All three read from ONE profile fetch per player, cached for the session.
// The board carries 252 players and none of this detail -- embedding every
// player's game log in the board payload would multiply it by the length of a
// season for data that is only ever looked at a row at a time.
import { memo, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

import { cachedGet, fetchProfile, forgetRequest } from '../../api'
import type { LiveSettings, PlayerProfileData } from '../../api'
import { Chart } from './Chart'
import { startersAt, weightedFinish } from './finish'
import {
  finishCols, healthCols, perGameCols, perGameDelta, playedSeasons, ratedSeasons,
  seasonLength, steadyCols,
  weekCols,
} from './panels'

export type CellTipKind = 'health' | 'games' | 'finish' | 'steady' | 'change'

// One in-flight request per player, and one cached result, shared across all
// three columns: hovering a row's Health and then its Finish is one fetch.
/** How long one player's profile stands. It is a 3.5s request and the room
 *  opens it from four places -- a hover, a card, the overlay, the board's
 *  peek -- so the point of the window is that a reader poking at one player
 *  pays for him once. A minute rather than forever because a profile carries
 *  news and an injury status, and a room that is open for an hour should not
 *  still be showing what those said when it opened. */
const PROFILE_MS = 60_000

export function loadProfile(playerId: string): Promise<PlayerProfileData> {
  // api.ts's own cache, not a second one here: the player overlay used to
  // call `fetchProfile` directly and so paid for a profile this cache had
  // already fetched -- two requests for the same player, one click apart.
  return cachedGet(`profile/${playerId}`, () => fetchProfile(playerId), PROFILE_MS)
    .then((data) => { remember(playerId, data); return data })
}

// ONE CACHE, AND A SYNCHRONOUS VIEW OF IT.
//
// The store is api.ts's `cachedGet` (above): one in-flight request per
// player and one window, shared by every reader of a profile -- the hover
// tip, the board-grid peek, the target cards and the profile card. This map
// is not a second one; it never asks for anything and is never consulted
// about freshness. It exists because those readers open on a POINTER, and a
// payload they can paint in the frame they open is the difference between a
// card and a skeleton that flashes over an answer we already hold.
//
// A stale entry here is painted and then replaced: every reader that paints
// from it also calls `loadProfile`, which is a lookup inside the shared
// window and a real request past it.
//
// BOUNDED, AND STAMPED. Two limits, both about what a long session does to
// it. A draft room is open for hours and a pointer crosses a lot of rows, so
// without a cap this grows to the whole board -- 250 profiles at tens of
// kilobytes each, held for a card that shows one. And a payload that arrived
// two hours ago carries news and an injury status from two hours ago, which
// is fine to fetch again behind a drawn card and not fine to PAINT as though
// it were current.
const SETTLED_MAX = 64
const SETTLED_FRESH_MS = 10 * 60_000

const settled = new Map<string, { at: number; data: PlayerProfileData }>()

// Oldest write out first. `Map` iterates in insertion order and a re-write
// deletes before it sets, so the entry that leaves is the one nobody has
// looked at for longest -- which on this page is the row furthest back in the
// sweep that filled the map.
function remember(playerId: string, data: PlayerProfileData): void {
  settled.delete(playerId)
  settled.set(playerId, { at: Date.now(), data })
  if (settled.size > SETTLED_MAX) {
    const oldest = settled.keys().next()
    if (!oldest.done) settled.delete(oldest.value)
  }
}

/** What has already arrived for this player, or null. Synchronous, for a
 *  reader that would otherwise draw a spinner over an answer in hand.
 *
 *  Null for an entry past `SETTLED_FRESH_MS`, which sends that reader to
 *  `loadProfile` and a skeleton instead -- an honest wait for a current
 *  answer beats an instant one that is an hour old. */
export function cachedProfile(playerId: string): PlayerProfileData | null {
  const hit = settled.get(playerId)
  if (!hit) return null
  return Date.now() - hit.at > SETTLED_FRESH_MS ? null : hit.data
}

/** Drops one player from both the view and the store, so the next read
 *  really asks. For the one case where something on screen has CHANGED what
 *  the payload reports -- the card's own drafted toggle -- and every other
 *  reader would otherwise go on serving the answer from before it. */
export function forgetProfile(playerId: string): void {
  settled.delete(playerId)
  forgetRequest(`profile/${playerId}`)
}

// Health, the 2025 sparkline, and Finish are all the same picture -- time
// along the bottom, one column per period, something growing from a shared
// baseline -- built here from each panel's own data and drawn by the shared
// `Chart` component so a reader only has to learn the picture once.
type BodyProps = { data: PlayerProfileData; settings?: LiveSettings | null }

function HealthBody({ data }: BodyProps): ReactNode {
  const cols = healthCols(data.seasons)
  if (!cols.length) return <div className="ctip-empty">No NFL seasons yet.</div>
  // The last column's key is that season: reading it back off the built
  // columns keeps this note from re-deriving "which season is most recent"
  // by a second route than `healthCols` used.
  const lastSeason = cols[cols.length - 1].key as number
  return (
    <>
      <div className="ctip-head">
        <span>Games played</span>
        <span className="ctip-head-note">
          of {seasonLength(lastSeason)}
        </span>
      </div>
      <Chart cols={cols} />
    </>
  )
}

function GamesBody({ data }: BodyProps): ReactNode {
  const seasons = data.game_log.map((g) => g.season)
  if (!seasons.length) return <div className="ctip-empty">No games on record.</div>
  const latest = Math.max(...seasons)
  const played = new Map(
    data.game_log.filter((g) => g.season === latest && !g.dnp).map((g) => [g.week, g]))
  const scored = [...played.values()].map((g) => g.ppr_points)
  const avg = scored.length ? scored.reduce((a, b) => a + b, 0) / scored.length : 0
  const cols = weekCols(data.game_log, latest, data.header.position)
  return (
    <>
      <div className="ctip-head">
        <span>{latest} by week</span>
        <span className="ctip-head-note">avg {avg.toFixed(1)}</span>
      </div>
      <Chart cols={cols} />
    </>
  )
}

function FinishBody({ data, settings }: BodyProps): ReactNode {
  if (!data.seasons.length) return <div className="ctip-empty">No NFL seasons yet.</div>
  const pos = data.header.position
  const starters = startersAt(pos, settings)
  const avg = weightedFinish(data.seasons.map((s) => [s.season, s.pos_finish]))
  const cols = finishCols(data.seasons, starters, data.header.proj_pos_finish)
  return (
    <>
      <div className="ctip-head">
        <span>Positional finish</span>
        {/* Was "startable to RB24", which said neither whose league nor what
            it had to do with this player -- and was hardcoded to a twelve-
            team one at that. His own recency-weighted average finish is the
            summary the column is actually sorted on. */}
        <span className="ctip-head-note">
          {avg === null ? '—' : `avg ${pos}${Math.round(avg)}`}
        </span>
      </div>
      <Chart cols={cols} />
    </>
  )
}

// Steadiness, season by season, as a PLACE among the position rather than as
// the coefficient it is computed from.
//
// The panel used to draw `1 - cv/1.2` with the raw coefficient printed above
// it -- an abstract ratio on a clamped axis, inverted twice (a lower cv is
// better, a taller bar is better), and coloured off the bar's own height
// rather than off anything. Nothing on it answered the only question a reader
// brings: is 0.63 good? `cv_rank` has been in the payload the whole time and
// answers exactly that, so the panel now reads like Finish beside it: a rank,
// against a stated pool, taller is better.
//
// Percentile of the REAL pool, not the `starters` yardstick Finish uses.
// Steadiness is not scarce the way RB1 production is -- grading 28th of 95
// against an eight-team league's sixteen startable backs would paint a
// genuinely steady season red.
//
// Linear, not log-spaced like Finish: RB1 to RB6 is the difference between
// rounds, which is why that ladder is log, but consistency has no equivalent
// tier structure to stretch. (Tone cut points live with the builder, in
// `panels.ts`, alongside the fill they colour.)
function SteadyBody({ data }: BodyProps): ReactNode {
  // The builder's own row filter (`panels.ts`), not a second copy of it: the
  // note below quotes a pool off `rows[0]`, and a filter that disagreed with
  // the one the columns were cut from would quote it from a season this
  // panel never drew.
  const rows = ratedSeasons(data.seasons)
  if (!rows.length) {
    return <div className="ctip-empty">No season long enough to measure.</div>
  }
  const pos = data.header.position
  const cols = steadyCols(data.seasons)
  // The most recent pool, because a denominator moves year to year and the
  // note is there to make the latest column legible, not to average them.
  const pool = rows[0].cv_rank_n as number
  return (
    <>
      <div className="ctip-head">
        <span>Reliability rank</span>
        {/* Which end is good is the one thing a rank cannot say for itself,
            and it is the exact thing the old panel left a reader guessing. */}
        <span className="ctip-head-note">{`1 = most reliable of ${pool} ${pos}s`}</span>
      </div>
      <Chart cols={cols} />
    </>
  )
}

// Scoring by season with the PROJECTION as the final column, because the
// Change column is the gap between the last of these and that one -- and a
// gap is the one thing a single number cannot show you the size of.
function ChangeBody({ data }: BodyProps): ReactNode {
  // Same rule the builder cuts its season columns on, imported rather than
  // repeated: `last` is read off `rows[0]`, so this list and the chart have
  // to agree about which season is the most recent one played.
  const rows = playedSeasons(data.seasons)
  const proj = data.summary?.proj_ppg ?? null
  if (!rows.length && proj === null) {
    return <div className="ctip-empty">Nothing to compare yet.</div>
  }
  const cols = perGameCols(data.seasons, proj, data.header.position)
  const delta = perGameDelta(data.seasons, proj)
  return (
    <>
      <div className="ctip-head">
        <span>Points per game</span>
        <span className="ctip-head-note">
          {delta === null ? 'projection vs history' : (
            <>
              {/* Green up, red down -- the one number on this panel that is a
                  direction rather than a level, and the direction is the
                  point of it. */}
              <span className={`delta-tone ${delta.tone}`}>{delta.label}</span>
              {' vs last season'}
            </>
          )}
        </span>
      </div>
      <Chart cols={cols} />
    </>
  )
}

const BODIES = { health: HealthBody, games: GamesBody, finish: FinishBody,
                 steady: SteadyBody, change: ChangeBody }

export const CellTip = memo(function CellTip(
  { kind, playerId, settings }: {
    kind: CellTipKind; playerId: string; settings?: LiveSettings | null
  },
): ReactNode {
  const [data, setData] = useState<PlayerProfileData | null>(
    () => cachedProfile(playerId))
  const [failed, setFailed] = useState(false)
  // Guards a resolve arriving after the pointer has moved to another row --
  // without it the panel would briefly show the previous player's seasons
  // under the current player's name.
  const wanted = useRef(playerId)

  useEffect(() => {
    wanted.current = playerId
    const hit = cachedProfile(playerId)
    if (hit) { setData(hit); setFailed(false) } else { setData(null); setFailed(false) }
    loadProfile(playerId)
      .then((d) => { if (wanted.current === playerId) setData(d) })
      .catch(() => { if (wanted.current === playerId) setFailed(true) })
  }, [playerId])

  if (failed) return <div className="ctip-empty">Could not load this player.</div>
  if (!data) {
    // A SKELETON OF THE THING BEING FETCHED, not the word "Loading". Every
    // panel this tip can open is a row of bars over a baseline (see
    // `.ctip-chart`), so five bars filling in place say "the chart is coming"
    // in the shape it is coming in -- and, because they move, say it without
    // a reader having to read anything. The static word sat in an otherwise
    // motionless box and was indistinguishable from a tooltip that had given
    // up. The sentence still exists for a screen reader, which cannot see the
    // bars at all.
    return (
      <div className="ctip-loading" role="status">
        <span className="sr-only">Loading this player…</span>
        {[0, 1, 2, 3, 4].map((i) => (
          <span className="ctip-skel" key={i} aria-hidden="true" />
        ))}
      </div>
    )
  }
  const Body = BODIES[kind]
  return <Body data={data} settings={settings} />
})
