import { useEffect, useState } from 'react'
import { fetchProfile, type Player, type PlayerProfileData } from '../api'
import FactorBars from './FactorBars'
import SimilarPlayers from './SimilarPlayers'

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

const fmt1 = (n: number | null) => (n === null ? '—' : n.toFixed(1))

export default function PlayerProfile({ playerId, onClose, onToggleDrafted, onSelectPlayer }: PlayerProfileProps) {
  const [profile, setProfile] = useState<PlayerProfileData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function loadProfile() {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchProfile(playerId)
      setProfile(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load player profile')
    } finally {
      setLoading(false)
    }
  }

  // Reset and refetch whenever the drawer is pointed at a new player (initial
  // open, or a comp/row click swapping the id while the drawer stays mounted).
  useEffect(() => {
    setProfile(null)
    loadProfile()
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
    await onToggleDrafted(profile.header)
    await loadProfile()
  }

  const header = profile?.header

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="player-drawer">
        <button type="button" className="drawer-close" aria-label="Close" onClick={onClose}>
          ✕
        </button>
        {loading && !profile && <p>Loading profile…</p>}
        {error && <p className="error">{error}</p>}
        {header && profile && (
          <>
            <div className="drawer-header">
              <h2>
                {header.name}
                {header.rookie && <span className="rookie-badge">R</span>}
              </h2>
              <p className="drawer-subhead">
                {header.position} · {header.team} · Bye {header.bye ?? '—'}
              </p>
              <div className="drawer-chips">
                <span className="chip">Rank #{header.rank}</span>
                <span className="chip">Tier {header.tier}</span>
                <span className="chip">VOR {header.vor.toFixed(1)}</span>
                <span className="chip">Composite {header.composite.toFixed(1)}</span>
                <span className="chip">ADP {fmt1(header.adp)}</span>
                <span className="chip">Edge {fmt1(header.edge)}</span>
              </div>
              <button type="button" className="drawer-draft-btn" onClick={handleToggleDraftedClick}>
                {header.drafted ? 'Undo draft' : 'Mark drafted'}
              </button>
            </div>

            <section className="drawer-section">
              <h3>Factors</h3>
              <FactorBars factors={profile.factors} />
            </section>

            <section id="weekly-chart" className="drawer-section" />

            <section className="drawer-section">
              <h3>Season history</h3>
              <p className="drawer-placeholder">Season-by-season stats coming in next task.</p>
            </section>

            <section className="drawer-section">
              <h3>Game log</h3>
              <p className="drawer-placeholder">Weekly game log coming in next task.</p>
            </section>

            <section className="drawer-section">
              <h3>Outlook</h3>
              <div className="drawer-chips">
                {depthSlotLabel(header.position, profile.outlook.depth_slot) && (
                  <span className="chip">{depthSlotLabel(header.position, profile.outlook.depth_slot)}</span>
                )}
                <span className="chip">Implied {fmt1(profile.outlook.implied_points)} pts</span>
                <span className="chip">
                  SoS {profile.outlook.sos_pct !== null ? `${profile.outlook.sos_pct.toFixed(0)}th pct` : '—'}
                </span>
                <span className="chip">Bye {profile.outlook.bye ?? '—'}</span>
              </div>
            </section>

            <section className="drawer-section">
              <h3>Similar players</h3>
              <SimilarPlayers
                mode={profile.similar.mode}
                players={profile.similar.players}
                onSelectPlayer={onSelectPlayer}
              />
            </section>
          </>
        )}
      </aside>
    </>
  )
}
