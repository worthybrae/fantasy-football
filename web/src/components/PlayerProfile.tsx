import { useEffect, useRef, useState } from 'react'
import { fetchProfile, type Player, type PlayerProfileData } from '../api'
import SeasonRangeChart from './SeasonRangeChart'
import GameLog from './GameLog'
import SeasonTable from './SeasonTable'
import SimilarPlayers from './SimilarPlayers'
import SnapShareChart from './SnapShareChart'
import StatTiles from './StatTiles'
import RankingsPanel from './RankingsPanel'
import DepthChartCard from './DepthChartCard'
import ScheduleCalendar from './ScheduleCalendar'
import PageSkeleton from './PageSkeleton'

interface PlayerProfileProps {
  playerId: string
  onClose: () => void
  onToggleDrafted: (p: Player) => Promise<void>
  onSelectPlayer: (id: string) => void
}

function depthSlotLabel(position: string, depthSlot: number | null): string | null {
  if (depthSlot === null) return null
  return `${position}${depthSlot}`
}

// Higher sos_raw/sos_pct = opponents allow more fantasy points at this
// position = an easier ("softer") schedule; lower = a tougher one.
function sosLabel(sosRaw: number | null, sosPct: number | null): string {
  if (sosRaw === null || sosPct === null) return 'SoS —'
  const direction = sosPct >= 50 ? 'softer' : 'tougher'
  return `SoS ${sosRaw.toFixed(1)} FPA/g (${sosPct.toFixed(0)}th pct — ${direction})`
}

export default function PlayerProfile({ playerId, onClose, onToggleDrafted, onSelectPlayer }: PlayerProfileProps) {
  const [profile, setProfile] = useState<PlayerProfileData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Always mirrors the current `playerId` prop, updated synchronously every
  // render (not via an effect) so it's current *before* any in-flight
  // fetch's `.then` runs. Every fetch this drawer issues -- the
  // playerId-change effect below, and the post-toggle refetch in
  // handleToggleDraftedClick, whose closure can still be holding the id of
  // a player the user has since swapped away from -- checks this ref before
  // committing state, so a stale response can never clobber what's
  // currently on screen (e.g. open A, immediately click a comp for B: A's
  // slower response is dropped instead of overwriting B's already-rendered
  // profile).
  const playerIdRef = useRef(playerId)
  playerIdRef.current = playerId

  async function loadProfile(forPlayerId: string) {
    // Guard at the top too, not just before each state commit below: a
    // stale invocation (e.g. handleToggleDraftedClick's post-await refetch
    // for a player the user has since swapped away from) must be a
    // complete no-op. Without this, setLoading(true)/setError(null) would
    // still fire for a stale id -- wiping a legitimate error for the
    // *current* player and flipping loading true -- and since every commit
    // below is itself guarded, setLoading(false) would never run for that
    // stale id, leaving the drawer stuck on "Loading profile..." forever.
    if (playerIdRef.current !== forPlayerId) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchProfile(forPlayerId)
      if (playerIdRef.current === forPlayerId) setProfile(data)
    } catch (e) {
      if (playerIdRef.current === forPlayerId) {
        setError(e instanceof Error ? e.message : 'Failed to load player profile')
      }
    } finally {
      if (playerIdRef.current === forPlayerId) setLoading(false)
    }
  }

  // Reset and refetch whenever the drawer is pointed at a new player (initial
  // open, or a comp/row click swapping the id while the drawer stays
  // mounted).
  useEffect(() => {
    setProfile(null)
    loadProfile(playerId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playerId])

  // Scoped to the drawer being mounted at all -- avoids a global listener
  // (and stray Esc-closes) while the board is the only thing on screen.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  async function handleToggleDraftedClick() {
    if (!profile) return
    // Use the id this specific profile was fetched for (not the possibly
    // already-swapped `playerId` prop) -- loadProfile's own playerIdRef
    // check still guards the commit, but passing the right id means a
    // still-current toggle isn't needlessly dropped too.
    const forPlayerId = profile.header.player_id
    await onToggleDrafted(profile.header)
    await loadProfile(forPlayerId)
  }

  const header = profile?.header
  const depthSlot = header ? depthSlotLabel(header.position, profile.outlook.depth_slot) : null

  if (loading && !profile) return <PageSkeleton />

  return (
    <div className="player-page">
      <button type="button" className="player-page-back" onClick={onClose}>
        ← Back to board
      </button>
      {error && <p className="error">{error}</p>}
      {header && profile && (
        <>
          <div className="pp-header">
            <div>
              <h2 className="pp-name">
                {header.name}
                {header.rookie && <span className="rookie-badge">R</span>}
              </h2>
              <p className="drawer-subhead">
                <span className={`pos-badge pos-badge-${header.position.toLowerCase()}`}>
                  {header.position}
                </span>{' '}
                · {header.team} · Bye {header.bye ?? '—'}
                {depthSlot && <> · {depthSlot}</>}
                {' · '}
                {sosLabel(profile.outlook.sos_raw, profile.outlook.sos_pct)}
              </p>
            </div>
            <button type="button" className="drawer-draft-btn" onClick={handleToggleDraftedClick}>
              {header.drafted ? 'Undo draft' : 'Mark drafted'}
            </button>
          </div>

          <div className="pp-grid">
            <section className="pp-card pp-span7">
              <h3>Production</h3>
              <StatTiles summary={profile.summary} outlook={profile.outlook} position={header.position} />
            </section>

            <section className="pp-card pp-span5">
              <h3>Rankings</h3>
              <RankingsPanel
                marketRank={header.market_rank}
                marketSpread={header.market_spread}
                sources={header.market_sources}
              />
            </section>

            {profile.seasons.length > 0 && (
              <section className="pp-card pp-span7">
                <h3>Avg &amp; volatility by season</h3>
                <SeasonRangeChart seasons={profile.seasons} position={header.position} />
              </section>
            )}

            {['RB', 'WR', 'TE'].includes(header.position) &&
              profile.seasons.some((s) => s.snap_share !== null) && (
              <section className="pp-card pp-span5">
                <h3>Snap share</h3>
                <SnapShareChart seasons={profile.seasons} />
              </section>
            )}

            {profile.depth_chart.length > 0 && (
              <section className="pp-card pp-span5">
                <h3>Depth chart</h3>
                <DepthChartCard team={header.team} groups={profile.depth_chart} />
              </section>
            )}

            {profile.schedule.length > 0 && (
              <section className="pp-card pp-span7">
                <h3>2026 matchups</h3>
                <ScheduleCalendar weeks={profile.schedule} position={header.position} bye={profile.outlook.bye} />
              </section>
            )}

            <section className="pp-card pp-span12">
              <h3>Season history</h3>
              <SeasonTable seasons={profile.seasons} position={header.position} />
            </section>

            <section className="pp-card pp-span12">
              <h3>Game log</h3>
              <GameLog gameLog={profile.game_log} position={header.position} />
            </section>

            <section className="pp-card pp-span12">
              <h3>Similar players</h3>
              <SimilarPlayers
                mode={profile.similar.mode}
                players={profile.similar.players}
                targetAge={profile.similar.target_age ?? null}
                onSelectPlayer={onSelectPlayer}
              />
            </section>
          </div>
        </>
      )}
    </div>
  )
}
