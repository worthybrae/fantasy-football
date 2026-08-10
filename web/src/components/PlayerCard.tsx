import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchProfile, playerSlug, setDrafted, type PlayerProfileData } from '../api'
import SeasonRangeChart from './SeasonRangeChart'

interface PlayerCardProps {
  playerId: string
  onClose: () => void
  // Avail%/ΔEV live on the board row, not the profile payload -- the draft
  // grid (Task 5) already has the full board list in hand when it opens
  // this card, so it passes the one row's values down instead of this card
  // re-fetching the whole board just to pick one out. Both default to null
  // so the card renders correctly when mounted with only a playerId -- e.g.
  // the grid's own data source, GET /api/sim/board, carries no per-player
  // `ev` at all, so it will mount this card with neither prop set.
  avail_pct?: number | null
  // Already relative to the board's best available option -- the same
  // `delta = p.ev - bestEv` PlayerTable.tsx computes for its ΔEV column, not
  // the raw `Player.ev`. The caller must do that subtraction before passing
  // it down; this card has no board list to compute bestEv from itself.
  // Because of that contract, 0 specifically means "tied for the board's
  // best," which is what justifies rendering it as a dash below -- a raw
  // absolute ev of 0 would have no such meaning and dashing it would make a
  // real (low) value indistinguishable from "no sim data."
  evDelta?: number | null
}

// duplicated from PlayerProfile.tsx (both are unexported there)
function depthSlotLabel(position: string, depthSlot: number | null): string | null {
  if (depthSlot === null) return null
  return `${position}${depthSlot}`
}

// duplicated from PlayerProfile.tsx -- higher sos_raw/sos_pct = opponents
// allow more fantasy points at this position = an easier ("softer")
// schedule; lower = a tougher one. Copied rather than shared -- two
// three-line consumers of a pure function don't earn a module of their own.
function sosLabel(sosRaw: number | null, sosPct: number | null): string {
  if (sosRaw === null || sosPct === null) return 'SoS —'
  const direction = sosPct >= 50 ? 'softer' : 'tougher'
  return `SoS ${sosRaw.toFixed(1)} FPA/g (${sosPct.toFixed(0)}th pct — ${direction})`
}

const fmt1 = (n: number | null) => (n === null ? '—' : n.toFixed(1))
// Edge (market_rank - rank) reads in both directions -- an explicit sign
// makes "the model likes them more than the market" vs. the reverse
// unambiguous at a glance, the same way StatTiles signs proj_delta.
const fmtSigned1 = (n: number | null) => (n === null ? '—' : `${n > 0 ? '+' : ''}${n.toFixed(1)}`)

const FACTOR_ROWS: [string, keyof PlayerProfileData['factors']][] = [
  ['Production', 'production'],
  ['Role', 'role'],
  ['Environment', 'environment'],
  ['Schedule', 'schedule'],
  ['Durability', 'durability'],
]

// The three-second, draft-night version of PlayerProfile: one board row's
// worth of decision-making context in a modal, instead of the full research
// page. Opened by clicking a grid cell (Task 5); closed by Escape, a
// backdrop click, or the close button.
export default function PlayerCard({ playerId, onClose, avail_pct = null, evDelta = null }: PlayerCardProps) {
  const [profile, setProfile] = useState<PlayerProfileData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // Optimistic override for the drafted toggle, null while it agrees with
  // whatever the profile fetch returned. PlayerProfile re-fetches the whole
  // profile after toggling; this card is the draft-night view, where a
  // round-trip between "he's gone" and the button changing is the cost the
  // toggle exists to avoid.
  const [draftedOverride, setDraftedOverride] = useState<boolean | null>(null)

  // Same guard PlayerProfile.tsx uses (see its playerIdRef comment for the
  // full rationale): mirrors the current `playerId` synchronously every
  // render, so a fetch that resolves after the id has already moved on
  // (open A, immediately click a comp/cell for B) can't clobber what's
  // already on screen for the new id.
  const playerIdRef = useRef(playerId)
  playerIdRef.current = playerId

  useEffect(() => {
    const forPlayerId = playerId
    setProfile(null)
    setError(null)
    setLoading(true)
    setDraftedOverride(null)
    fetchProfile(forPlayerId)
      .then((data) => {
        if (playerIdRef.current === forPlayerId) setProfile(data)
      })
      .catch((e) => {
        if (playerIdRef.current === forPlayerId) {
          setError(e instanceof Error ? e.message : 'Failed to load player profile')
        }
      })
      .finally(() => {
        if (playerIdRef.current === forPlayerId) setLoading(false)
      })
  }, [playerId])

  // Scoped to this card being mounted at all, and removed on unmount --
  // the grid page has no board-level keyboard handling of its own, but a
  // stray listener left behind after the card closes would still be a leak.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  // Named `player`, not `header`, so it doesn't shadow the literal <header>
  // JSX tag below (the brief's CSS targets `.player-card header`).
  const player = profile?.header
  const depthSlot = player ? depthSlotLabel(player.position, profile.outlook.depth_slot) : null
  const showSim = avail_pct !== null || evDelta !== null
  const drafted = draftedOverride ?? player?.drafted ?? false

  async function handleToggleDrafted() {
    if (!player) return
    const next = !drafted
    setDraftedOverride(next)
    try {
      await setDrafted(player.player_id, next)
    } catch {
      // The board is the source of truth; if the write never landed, don't
      // leave the card claiming a pick that didn't happen.
      setDraftedOverride(!next)
    }
  }

  return (
    <div className="card-backdrop" onClick={onClose}>
      <div className="player-card" onClick={(e) => e.stopPropagation()}>
        {loading && !profile && <p>Loading…</p>}
        {error && <p className="error">{error}</p>}
        {player && profile && (
          <>
            <header>
              <h2>{player.name}</h2>
              <span className="card-sub">
                <span className={`pos-badge pos-badge-${player.position.toLowerCase()}`}>
                  {player.position}
                </span>{' '}
                {player.team} · Bye {player.bye ?? '—'}
              </span>
              <button type="button" className="card-close" onClick={onClose} aria-label="Close">
                ✕
              </button>
            </header>

            <div className="card-row">
              <div>Rank <strong className="mono">{player.rank}</strong></div>
              <div>Tier <strong className="mono">{player.tier}</strong></div>
              <div>Mkt <strong className="mono">{fmt1(player.market_rank)}</strong></div>
              <div>Edge <strong className="mono">{fmtSigned1(player.edge)}</strong></div>
              <div>Proj <strong className="mono">{fmt1(profile.summary.proj_ppg)}</strong></div>
            </div>

            {showSim && (
              <div className="card-row card-sim">
                {avail_pct !== null && (
                  <div>Avail% <strong className="mono">{Math.round(avail_pct * 100)}%</strong></div>
                )}
                {evDelta !== null && (
                  <div>ΔEV <strong className="mono">{evDelta === 0 ? '—' : evDelta.toFixed(1)}</strong></div>
                )}
              </div>
            )}

            <ul className="card-factors">
              {FACTOR_ROWS.map(([label, key]) => {
                const value = profile.factors[key]
                return (
                  <li key={key} title={`${label}: ${Math.round(value)}/100`}>
                    <span className="card-factor-name">{label}</span>
                    <span className="card-factor-track">
                      <span
                        className="card-factor-fill"
                        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
                      />
                    </span>
                  </li>
                )
              })}
            </ul>

            {profile.seasons.length > 0 && (
              <SeasonRangeChart seasons={profile.seasons} position={player.position} />
            )}

            <p className="card-outlook">
              {depthSlot ?? '—'}
              {' · '}
              {profile.outlook.implied_points !== null
                ? `${profile.outlook.implied_points.toFixed(1)} team pts`
                : 'Team pts —'}
              {' · '}
              {sosLabel(profile.outlook.sos_raw, profile.outlook.sos_pct)}
            </p>

            <div className="card-actions">
              <button
                type="button"
                className="drawer-draft-btn"
                onClick={handleToggleDrafted}
              >
                {drafted ? 'Undo draft' : 'Mark drafted'}
              </button>
              <Link to={`/players/${playerSlug(player.name)}`} className="card-full">
                Full profile →
              </Link>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
