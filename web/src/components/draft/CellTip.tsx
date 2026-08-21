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


// -- one chart, drawn three times ------------------------------------------
//
// Every panel is the same picture: time along the bottom, one column per
// period, something growing from a shared baseline, the value above it and
// the period below. Health and Finish are both about seasons and used to be
// read in opposite directions -- one a stack of rows, the other a left-to-
// right axis -- so a reader had to learn the panel again on each column.
//
// Taller is better in all three and the colour means the same thing in all
// three, which leaves one thing to know per panel: what a column is.
type BodyProps = { data: PlayerProfileData; settings?: LiveSettings | null }

type Col = {
  key: string | number
  label: string                           // under the baseline: a year, a team
  value: string                           // above the bar
  tone: string                            // the shared five-step colour
  fill: number                            // 0..1 of the slot
  units?: { filled: number; of: number }  // circles instead of a solid bar
  empty?: boolean                         // did not play: a baseline mark
}

function Chart({ cols }: { cols: Col[] }): ReactNode {
  return (
    <div className="ctip-chart">
      {cols.map((c) => (
        <span key={c.key} className="ctip-col">
          <span className={`ctip-col-val ${c.empty ? 'is-off' : c.tone}`}>
            {c.value}
          </span>
          <span className="ctip-col-slot">
            {c.empty
              // Not a zero-height bar: "did not play" and "played and scored
              // nothing" are different claims, and the second already draws
              // as the 6% floor below.
              ? <span className="ctip-col-none" />
              : c.units
                ? <span className="ctip-units">
                    {Array.from({ length: c.units.of }, (_, i) => (
                      <span key={i} className={`ctip-unit${
                        i < (c.units?.filled ?? 0) ? ` is-on ${c.tone}` : ''}`} />
                    ))}
                  </span>
                : <span className={`ctip-col-bar ${c.tone}`}
                        style={{ height: `${Math.max(6, c.fill * 100)}%` }} />}
          </span>
          <span className="ctip-col-label">{c.label}</span>
        </span>
      ))}
    </div>
  )
}

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

// Steadiness, season by season. The column ranks a player against the board
// on one recency-weighted number; this is where that number came from.
//
// The bar is 1 - cv, so taller is steadier and the panel obeys the same
// "taller is better" rule as the three beside it. The value printed is the
// coefficient itself, which is what the column's own tooltip quotes.
function SteadyBody({ data }: BodyProps): ReactNode {
  const rows = data.seasons.filter((s) => s.cv !== null && s.cv !== undefined)
  if (!rows.length) {
    return <div className="ctip-empty">No season long enough to measure.</div>
  }
  // A coefficient above 1 means a player whose week-to-week swing exceeds his
  // own average -- real, and rare enough that letting it set the axis would
  // flatten everyone else. Clamped, like every other scale here.
  const CV_MAX = 1.2
  const cols: Col[] = rows.slice().reverse().map((r) => {
    const cv = r.cv as number
    const share = 1 - Math.min(1, cv / CV_MAX)
    return {
      key: r.season,
      label: year(r.season),
      value: cv.toFixed(2),
      tone: HEALTH_TONES[Math.min(4, Math.floor(share * 5))],
      fill: share,
    }
  })
  const median = rows[0].cv_pos_median
  return (
    <>
      <div className="ctip-head">
        <span>Week-to-week swing</span>
        <span className="ctip-head-note">
          {median ? `${data.header.position} median ${median.toFixed(2)}` : 'lower is steadier'}
        </span>
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
