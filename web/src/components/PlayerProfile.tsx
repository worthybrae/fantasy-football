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

// Everything the opener already knew about this player, so the profile can
// paint on the frame it opens instead of behind a skeleton.
//
// This exists because GET /api/players/{id}/profile is slow -- measured warm
// against the live server, twice, at 3.50s and 3.51s -- and the draft room
// opens this over a running 30-second pick clock. Waiting three and a half
// seconds to learn a name the room was already displaying is not a load, it
// is a stall. The route at /players/:slug passes no seed and still gets the
// skeleton it always did.
//
// `figures` is pre-formatted, in display order, by whoever opened the
// profile. Deliberately not raw numbers: the live room's figures (gain vs
// waiting, survival, which roster slot he fills) come from
// /api/live/state's ranked list, carry rounding and sign rules this
// component has no business knowing, and are not even defined outside a
// draft. A label/value pair is the widest contract that stays honest.
// `accent` marks the one figure worth highlighting (an open starter slot,
// same rule as `.top3-figure.is-open`).
export interface ProfileSeed {
  name: string
  position: string
  team: string | null
  bye: number | null
  rookie: boolean
  figures: { label: string; value: string; accent?: boolean }[]
}

interface PlayerProfileProps {
  playerId: string
  onClose: () => void
  // Optional: the in-draft overlay deliberately passes nothing. /api/drafted
  // is the manual research-board flag, and during a live draft the drafted
  // set is owned by ESPN's own feed -- offering a button that writes it by
  // hand mid-draft would let a stray click desync the board from the room.
  onToggleDrafted?: (p: Player) => Promise<void>
  onSelectPlayer: (id: string) => void
  // Instant paint (see ProfileSeed). Null/absent restores the original
  // behaviour exactly: skeleton until the request lands.
  seed?: ProfileSeed | null
  // Rendered inside PlayerOverlay rather than as the /players/:slug page:
  // drop the chrome the overlay supplies itself (its own close control, its
  // own Escape handler) and stop claiming the full viewport height.
  embedded?: boolean
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

export default function PlayerProfile({
  playerId, onClose, onToggleDrafted, onSelectPlayer, seed = null, embedded = false,
}: PlayerProfileProps) {
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
  // Skipped when embedded: PlayerOverlay already owns Escape for the dialog
  // it draws around this, and two listeners calling the same onClose is a
  // second way for the same key to do the same thing -- one place to change
  // it, not two.
  useEffect(() => {
    if (embedded) return
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose, embedded])

  async function handleToggleDraftedClick() {
    if (!profile || !onToggleDrafted) return
    // Use the id this specific profile was fetched for (not the possibly
    // already-swapped `playerId` prop) -- loadProfile's own playerIdRef
    // check still guards the commit, but passing the right id means a
    // still-current toggle isn't needlessly dropped too.
    const forPlayerId = profile.header.player_id
    await onToggleDrafted(profile.header)
    await loadProfile(forPlayerId)
  }

  const header = profile?.header
  const depthSlot = header && profile ? depthSlotLabel(header.position, profile.outlook.depth_slot) : null

  // Who this page is about, from whichever source knows first: the loaded
  // profile's own header row, or the seed the opener handed over. The two
  // never disagree about identity -- the seed is read off the same board row
  // the profile is built from -- so the swap when the request lands is
  // invisible.
  const ident = header ?? seed

  // A seed is a painted header, so there is nothing for a skeleton to stand
  // in for. Without one this is unchanged: skeleton until the request lands.
  if (loading && !profile && !seed) return <PageSkeleton />

  return (
    <div className={`player-page${embedded ? ' player-page-embedded' : ''}`}>
      {!embedded && (
        <button type="button" className="player-page-back" onClick={onClose}>
          ← Back to board
        </button>
      )}
      {ident && (
        <div className="pp-header">
          <div>
            <h2 className="pp-name">
              {ident.name}
              {ident.rookie && <span className="rookie-badge">R</span>}
            </h2>
            <p className="drawer-subhead">
              <span className={`pos-badge pos-badge-${ident.position.toLowerCase()}`}>
                {ident.position}
              </span>{' '}
              · {ident.team ?? '—'} · Bye {ident.bye ?? '—'}
              {depthSlot && <> · {depthSlot}</>}
              {/* Depth slot and strength of schedule are the two things
                  only the request can answer, so this tail of the line is
                  where the wait is admitted -- inline, in a line that is
                  already on screen, rather than as a block that appears and
                  then vanishes and moves everything under it. */}
              {profile
                ? <>{' · '}{sosLabel(profile.outlook.sos_raw, profile.outlook.sos_pct)}</>
                : loading && (
                  <span className="pp-loading-tail">
                    {' · '}
                    <span className="pp-loading-dot" aria-hidden="true" />
                    <span role="status">loading full profile…</span>
                  </span>
                )}
            </p>
          </div>
          {onToggleDrafted && header && (
            <button type="button" className="drawer-draft-btn" onClick={handleToggleDraftedClick}>
              {header.drafted ? 'Undo draft' : 'Mark drafted'}
            </button>
          )}
        </div>
      )}

      {/* The opener's own numbers, rendered before anything has been
          fetched and never re-rendered from the response -- see ProfileSeed.
          Same geometry as the confirm dialog's figure row, because it is
          the same kind of thing: the handful of values the decision is
          actually made on. */}
      {seed && seed.figures.length > 0 && (
        <div className="pp-seed-figures">
          {seed.figures.map((f) => (
            <div key={f.label}>
              <div className="draft-cap">{f.label}</div>
              <div className={`pp-seed-figure mono${f.accent ? ' is-open' : ''}`}>{f.value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Below the header and the seed's figures, not above them: when this
          arrives it arrives late (the request has to fail first), and a
          block inserted above a header that is already painted would shove
          the whole profile down under the reader's eyes. */}
      {error && <p className="error">{error}</p>}

      {header && profile && (
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
      )}
    </div>
  )
}
