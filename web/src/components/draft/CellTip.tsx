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
import { FINISH_STARTERS, finishBands, finishPosition, finishTone } from './finish'

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
      <div className="ctip-head">Games played, season by season</div>
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
  const rows = data.game_log
    .filter((g) => g.season === latest)
    .slice()
    .sort((a, b) => a.week - b.week)
  return (
    <>
      <div className="ctip-head">{latest} points per game</div>
      <table className="ctip-table">
        <tbody>
          {rows.map((g) => (
            <tr key={g.week} className={g.dnp ? 'is-dnp' : undefined}>
              <td className="ctip-yr">W{g.week}</td>
              <td className="ctip-opp">{g.opponent ?? '—'}</td>
              <td className="ctip-num">{g.dnp ? '—' : g.ppr_points.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}

function FinishTrack(
  { finish, starters }: { finish: number; starters: number },
): ReactNode {
  const bands = finishBands(starters)
  const at = finishPosition(finish, starters)
  const tone = finishTone(finish, starters)
  let left = 0
  return (
    <span className="ctip-track">
      {bands.map((b) => {
        const width = b.end - left
        left = b.end
        return <span key={b.tone} className={`ctip-band ${b.tone}`}
                     style={{ width: `${width * 100}%` }} />
      })}
      {/* Positioned, not sized: the marker's job is to say WHERE on the
          ladder he came down, and the tiers behind it say what that place is
          called. `translateX(-50%)` so the dot is centred on its rank rather
          than starting at it -- at RB1 that keeps it on the track. */}
      <span className={`ctip-mark ${tone}`} style={{ left: `${at * 100}%` }} />
    </span>
  )
}

function FinishBody({ data }: { data: PlayerProfileData }): ReactNode {
  if (!data.seasons.length) return <div className="ctip-empty">No NFL seasons yet.</div>
  const pos = data.header.position
  const starters = FINISH_STARTERS[pos] ?? 24
  return (
    <>
      <div className="ctip-head">Where he finished, season by season</div>
      <table className="ctip-table">
        <tbody>
          {data.seasons.map((s) => (
            <tr key={s.season}>
              <td className="ctip-yr">{s.season}</td>
              <td className="ctip-trackcell">
                <FinishTrack finish={s.pos_finish} starters={starters} />
              </td>
              <td className={`ctip-fin ${finishTone(s.pos_finish, starters)}`}>
                {pos}{s.pos_finish}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {/* The axis, stated once at the bottom rather than repeated per row. */}
      <div className="ctip-legend">
        <span>{pos}1</span>
        <span className="ctip-legend-mid">startable to {pos}{starters}</span>
        <span>{pos}{starters * 3}+</span>
      </div>
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
