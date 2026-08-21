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
import type { PlayerProfileData, SeasonSummary } from '../../api'
import { FINISH_STARTERS, finishPosition, finishTone } from './finish'
import { BAR_CEILING, SEASON_WEEKS, barTone } from './weeks'

export type CellTipKind = 'health' | 'games' | 'finish'

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
  for (let season = last; season >= first; season -= 1) {
    rows.push({ season, games: played.get(season) ?? 0 })
  }
  return rows
}

function Dots({ season, games }: { season: number; games: number }): ReactNode {
  const total = seasonLength(season)
  const played = Math.min(games, total)
  return (
    <span className="ctip-dots">
      {Array.from({ length: total }, (_, i) => (
        <span key={i} className={`ctip-dot${i < played ? ' is-on' : ''}`} />
      ))}
    </span>
  )
}

function HealthBody({ data }: { data: PlayerProfileData }): ReactNode {
  const rows = availabilityRows(data.seasons)
  if (!rows.length) return <div className="ctip-empty">No NFL seasons yet.</div>
  return (
    <>
      <div className="ctip-head">Games played</div>
      <table className="ctip-table">
        <tbody>
          {rows.map((r) => (
            <tr key={r.season}>
              <td className="ctip-yr">{r.season}</td>
              <td><Dots season={r.season} games={r.games} /></td>
              <td className="ctip-num">{r.games}/{seasonLength(r.season)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}

function GamesBody({ data }: { data: PlayerProfileData }): ReactNode {
  const seasons = data.game_log.map((g) => g.season)
  if (!seasons.length) return <div className="ctip-empty">No games on record.</div>
  const latest = Math.max(...seasons)
  const played = new Map(
    data.game_log.filter((g) => g.season === latest && !g.dnp)
      .map((g) => [g.week, g]))
  const scored = [...played.values()].map((g) => g.ppr_points)
  const avg = scored.length
    ? scored.reduce((a, b) => a + b, 0) / scored.length : 0
  const position = data.header.position

  return (
    <>
      <div className="ctip-head">
        <span>{latest} by week</span>
        {/* The one number the chart cannot state about itself. The best week
            used to sit here too, to recover any value the shared 30-point
            ceiling clipped -- redundant since the points row landed, which
            prints every week's actual value including the clipped ones. */}
        <span className="ctip-head-note">avg {avg.toFixed(1)}</span>
      </div>
      <div className="ctip-weeks">
        {Array.from({ length: SEASON_WEEKS }, (_, i) => {
          const week = i + 1
          const game = played.get(week)
          const pts = game?.ppr_points ?? null
          const height = pts === null
            ? 0 : Math.max(6, Math.min(1, pts / BAR_CEILING) * 100)
          return (
            <span key={week} className="ctip-week">
              {/* Above the bar in a FIXED row rather than riding on top of
                  it: labels that sat on variable-height bars scattered across
                  the chart and stopped being a row of numbers you could read
                  across. Rounded, because one decimal on eighteen columns
                  makes each one wide enough to push the panel past its cap --
                  the header carries the exact average and best. */}
              <span className={`ctip-week-pts${pts === null ? ' is-off' : ''}`}>
                {pts === null ? '·' : Math.round(pts)}
              </span>
              <span className="ctip-week-slot">
                {pts === null
                  // Not a zero-height bar: a week he did not play and a week
                  // he scored nothing are different claims, and the second
                  // one already draws as the 6px floor above.
                  ? <span className="ctip-week-off" />
                  : <span className={`ctip-week-bar ${barTone(pts, position)}`}
                          style={{ height: `${height}%` }} />}
              </span>
              <span className="ctip-week-no">{week}</span>
              <span className="ctip-week-opp">{game?.opponent ?? '—'}</span>
            </span>
          )
        })}
      </div>
    </>
  )
}

// One column per season, oldest to newest, on a vertical rank axis.
//
// The first version painted all five tiers as bands. At the opacity that kept
// them from drowning the dots they blended into one green-to-red gradient,
// and a reader could not see where "startable" ended -- which is the only
// boundary most seasons are judged against. Two labelled lines say it
// exactly: the elite cut and the last startable rank. The dots keep the full
// five-tier colouring, so the finer grades are still there to be read.
const FIN_COL = 26        // per season
const FIN_H = 58          // the rank axis
const FIN_TOP = 11        // the finish numbers above it
const FIN_BOTTOM = 12     // the years below it
const FIN_RIGHT = 30      // room for the two axis labels

function FinishChart(
  { seasons, position }: { seasons: SeasonSummary[]; position: string },
): ReactNode {
  const starters = FINISH_STARTERS[position] ?? 24
  // Oldest first: a career reads left to right, and the payload is newest
  // first because a table wants the latest season at the top.
  const rows = seasons.slice().reverse()
  const plotW = Math.max(FIN_COL, rows.length * FIN_COL)
  const width = plotW + FIN_RIGHT
  const height = FIN_TOP + FIN_H + FIN_BOTTOM
  const x = (i: number) => i * FIN_COL + FIN_COL / 2
  const y = (finish: number) => FIN_TOP + finishPosition(finish, starters) * FIN_H

  const marks = [
    { rank: Math.round(starters / 4), label: `${position}${Math.round(starters / 4)}` },
    { rank: starters, label: `${position}${starters}` },
  ]
  const line = rows.map((r, i) => `${x(i)},${y(r.pos_finish)}`).join(' ')

  return (
    <svg className="ctip-fchart" width={width} height={height}
         viewBox={`0 0 ${width} ${height}`}>
      {/* One tint, for "startable or better". Five of them was four too
          many; one says where the useful half of the chart is. */}
      <rect className="ctip-fzone" x="0" y={FIN_TOP}
            width={plotW} height={y(starters) - FIN_TOP} />
      {marks.map((m) => (
        <g key={m.label}>
          <line className="ctip-frule" x1="0" x2={plotW}
                y1={y(m.rank)} y2={y(m.rank)} />
          <text className="ctip-frule-label" x={plotW + 4} y={y(m.rank) + 3}>
            {m.label}
          </text>
        </g>
      ))}
      {/* Only meaningful with two seasons to join, and drawn under the dots so
          a marker is never half-covered by the path leaving it. */}
      {rows.length > 1 && (
        <polyline className="ctip-fline" points={line} fill="none" />
      )}
      {rows.map((r, i) => (
        <g key={r.season}>
          <text className={`ctip-flabel ${finishTone(r.pos_finish, starters)}`}
                x={x(i)} y={FIN_TOP - 3} textAnchor="middle">{r.pos_finish}</text>
          <circle className={`ctip-fdot ${finishTone(r.pos_finish, starters)}`}
                  cx={x(i)} cy={y(r.pos_finish)} r="3.5" />
          <text className="ctip-fyear" x={x(i)} y={height - 2}
                textAnchor="middle">&rsquo;{String(r.season).slice(2)}</text>
        </g>
      ))}
    </svg>
  )
}

function FinishBody({ data }: { data: PlayerProfileData }): ReactNode {
  if (!data.seasons.length) return <div className="ctip-empty">No NFL seasons yet.</div>
  const pos = data.header.position
  return (
    <>
      {/* No axis note here: the two rules on the chart carry it, and this
          heading was wider than the chart itself -- which, in a panel sized
          to its widest child, is what left the empty strip down the side. */}
      <div className="ctip-head"><span>Positional finish</span></div>
      <FinishChart seasons={data.seasons} position={pos} />
    </>
  )
}

const BODIES = { health: HealthBody, games: GamesBody, finish: FinishBody }

export const CellTip = memo(function CellTip(
  { kind, playerId }: { kind: CellTipKind; playerId: string },
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
  return <Body data={data} />
})
