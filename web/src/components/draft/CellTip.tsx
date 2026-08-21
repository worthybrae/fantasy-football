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

import { fetchProfile } from '../../api'
import type { LiveSettings, PlayerProfileData, SeasonSummary } from '../../api'
import { Chart, type Col } from './Chart'
import { finishPosition, finishTone, startersAt, weightedFinish } from './finish'
import { BAR_CEILING, SEASON_WEEKS, barTone } from './weeks'

export type CellTipKind = 'health' | 'games' | 'finish' | 'steady' | 'change'

// One in-flight request per player, and one cached result, shared across all
// three columns: hovering a row's Health and then its Finish is one fetch.
const cache = new Map<string, PlayerProfileData>()
const inflight = new Map<string, Promise<PlayerProfileData>>()

export function loadProfile(playerId: string): Promise<PlayerProfileData> {
  const hit = cache.get(playerId)
  if (hit) return Promise.resolve(hit)
  const running = inflight.get(playerId)
  if (running) return running
  const p = fetchProfile(playerId)
    .then((data) => { cache.set(playerId, data); inflight.delete(playerId); return data })
    .catch((e) => { inflight.delete(playerId); throw e })
  inflight.set(playerId, p)
  return p
}

// Seasons ran 16 games before 2021 and 17 from it, so "games missed" is only
// meaningful against the right denominator -- a 16-game 2019 is a full year,
// and drawing it as one short would invent an injury.
function seasonLength(season: number): number {
  return season >= 2021 ? 17 : 16
}


// Health, the 2025 sparkline, and Finish are all the same picture -- time
// along the bottom, one column per period, something growing from a shared
// baseline -- built here from each panel's own data and drawn by the shared
// `Chart` component so a reader only has to learn the picture once.
type BodyProps = { data: PlayerProfileData; settings?: LiveSettings | null }

function year(season: number): string {
  return `\u2019${String(season).slice(2)}`
}

/** Games played per season, INCLUDING seasons with none.
 *
 *  A player who missed a whole year has no row in the profile's `seasons`, and
 *  leaving that year out is precisely the mistake the Health meter exists not
 *  to make: `career_availability` counts absent seasons as zeros, so a panel
 *  that skipped them would explain a number by showing different data. Spans
 *  from his first season with any games through the most recent one measured.
 */
function availabilityRows(seasons: SeasonSummary[]): { season: number; games: number }[] {
  if (!seasons.length) return []
  const played = new Map(seasons.map((s) => [s.season, s.games]))
  const first = Math.min(...played.keys())
  const last = Math.max(...played.keys())
  const rows = []
  for (let season = first; season <= last; season += 1) {
    rows.push({ season, games: played.get(season) ?? 0 })
  }
  return rows
}

// Health keeps its circles -- they say "16 of 17" in a way a bar cannot, and
// it was the panel that already read well -- stacked into a column so the
// season axis runs the same way Finish's does.
const HEALTH_TONES = ['is-out', 'is-fringe', 'is-starter', 'is-strong', 'is-elite']

function HealthBody({ data }: BodyProps): ReactNode {
  const rows = availabilityRows(data.seasons)
  if (!rows.length) return <div className="ctip-empty">No NFL seasons yet.</div>
  const cols: Col[] = rows.map((r) => {
    const of = seasonLength(r.season)
    const share = r.games / of
    return {
      key: r.season,
      label: year(r.season),
      value: String(r.games),
      // The same five steps the other panels use, cut on how much of the
      // season he was actually there for.
      tone: HEALTH_TONES[Math.min(4, Math.floor(share * 5))],
      fill: share,
      units: { filled: Math.min(r.games, of), of },
      empty: r.games === 0,
    }
  })
  return (
    <>
      <div className="ctip-head">
        <span>Games played</span>
        <span className="ctip-head-note">
          of {seasonLength(rows[rows.length - 1].season)}
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
  const position = data.header.position

  const cols: Col[] = Array.from({ length: SEASON_WEEKS }, (_, i) => {
    const week = i + 1
    const game = played.get(week)
    const pts = game?.ppr_points ?? null
    return {
      key: week,
      // The opponent, not the week number: a reader knows where week 9 is from
      // its place in the row, and "DEN" is the fact that explains a bad one.
      // The number would be the axis restating itself.
      label: game?.opponent ?? '—',
      value: pts === null ? '·' : String(Math.round(pts)),
      tone: pts === null ? '' : barTone(pts, position),
      fill: pts === null ? 0 : Math.min(1, pts / BAR_CEILING),
      empty: pts === null,
    }
  })
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
  const cols: Col[] = data.seasons.slice().reverse().map((s) => ({
    key: s.season,
    label: year(s.season),
    value: String(s.pos_finish),
    tone: finishTone(s.pos_finish, starters),
    // `1 -` because rank runs backwards. A chart where the best season was
    // the shortest column is the one thing a reader cannot be asked to hold
    // in their head while comparing it to the two panels beside it.
    fill: 1 - finishPosition(s.pos_finish, starters),
  }))
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
// tier structure to stretch.
function steadyTone(rank: number, pool: number): string {
  const pct = rank / pool
  if (pct <= 0.25) return 'is-elite'
  if (pct <= 0.5) return 'is-strong'
  if (pct <= 0.75) return 'is-starter'
  if (pct <= 0.9) return 'is-fringe'
  return 'is-out'
}

function SteadyBody({ data }: BodyProps): ReactNode {
  // `cv_rank_n` counts only the seasons that HAVE a coefficient: a season
  // whose mean is zero or negative gets none, ranks nowhere, and would draw
  // as a column with no place on the ladder it is being plotted against.
  const rows = data.seasons.filter(
    (s) => s.cv_rank !== null && s.cv_rank !== undefined
      && s.cv_rank_n !== null && s.cv_rank_n !== undefined && s.cv_rank_n > 0,
  )
  if (!rows.length) {
    return <div className="ctip-empty">No season long enough to measure.</div>
  }
  const pos = data.header.position
  const cols: Col[] = rows.slice().reverse().map((r) => {
    const rank = r.cv_rank as number
    const pool = r.cv_rank_n as number
    return {
      key: r.season,
      label: year(r.season),
      value: String(rank),
      tone: steadyTone(rank, pool),
      // `1 -` because rank runs backwards, same as Finish: the steadiest
      // season has to be the tallest column, or this panel and the one next
      // to it would read in opposite directions.
      fill: 1 - rank / pool,
    }
  })
  // The most recent pool, because a denominator moves year to year and the
  // note is there to make the latest column legible, not to average them.
  const pool = rows[0].cv_rank_n as number
  return (
    <>
      <div className="ctip-head">
        <span>Steadiness rank</span>
        {/* Which end is good is the one thing a rank cannot say for itself,
            and it is the exact thing the old panel left a reader guessing. */}
        <span className="ctip-head-note">{`1 = steadiest of ${pool} ${pos}s`}</span>
      </div>
      <Chart cols={cols} />
    </>
  )
}

// Scoring by season with the PROJECTION as the final column, because the
// Change column is the gap between the last of these and that one -- and a
// gap is the one thing a single number cannot show you the size of.
function ChangeBody({ data }: BodyProps): ReactNode {
  const rows = data.seasons.filter((s) => s.games > 0)
  const proj = data.summary?.proj_ppg ?? null
  if (!rows.length && proj === null) {
    return <div className="ctip-empty">Nothing to compare yet.</div>
  }
  const ceiling = Math.max(
    ...rows.map((r) => r.ppg), proj ?? 0, 1)
  const position = data.header.position
  const cols: Col[] = rows.slice().reverse().map((r) => ({
    key: r.season,
    label: year(r.season),
    value: r.ppg.toFixed(1),
    tone: barTone(r.ppg, position),
    fill: r.ppg / ceiling,
  }))
  if (proj !== null) {
    cols.push({
      key: 'proj',
      // Not a year: this column is the only one that has not happened.
      label: 'proj',
      value: proj.toFixed(1),
      tone: barTone(proj, position),
      fill: proj / ceiling,
      // Drawn hollow, behind a rule: everything left of it is banked, this
      // is the only column still owed. The tone stays, so it is still read
      // against the same good/mid/bad cut points as the seasons beside it.
      projected: true,
    })
  }
  const last = rows.length ? rows[0].ppg : null
  const delta = last !== null && proj !== null ? proj - last : null
  return (
    <>
      <div className="ctip-head">
        <span>Points per game</span>
        <span className="ctip-head-note">
          {delta === null
            ? 'projection vs history'
            : `${delta >= 0 ? '+' : ''}${delta.toFixed(1)} vs last season`}
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
    () => cache.get(playerId) ?? null)
  const [failed, setFailed] = useState(false)
  // Guards a resolve arriving after the pointer has moved to another row --
  // without it the panel would briefly show the previous player's seasons
  // under the current player's name.
  const wanted = useRef(playerId)

  useEffect(() => {
    wanted.current = playerId
    const hit = cache.get(playerId)
    if (hit) { setData(hit); setFailed(false); return }
    setData(null); setFailed(false)
    loadProfile(playerId)
      .then((d) => { if (wanted.current === playerId) setData(d) })
      .catch(() => { if (wanted.current === playerId) setFailed(true) })
  }, [playerId])

  if (failed) return <div className="ctip-empty">Could not load this player.</div>
  if (!data) return <div className="ctip-empty">Loading…</div>
  const Body = BODIES[kind]
  return <Body data={data} settings={settings} />
})
