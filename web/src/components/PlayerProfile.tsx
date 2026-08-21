import { useEffect, useRef, useState } from 'react'
import { fetchProfile, type Player } from '../api'
import DepthChartCard from './DepthChartCard'
import PageSkeleton from './PageSkeleton'
import SimilarPlayers from './SimilarPlayers'
import CohortNext from './profile/CohortNext'
import InjuryStatus from './profile/InjuryStatus'
import NewsPanel from './profile/NewsPanel'
import ConsistencyTable from './profile/ConsistencyTable'
import SeasonFinish from './profile/SeasonFinish'
import LineQuality from './profile/LineQuality'
import MarketRow from './profile/MarketRow'
import MissingData from './profile/MissingData'
import RoomGap from './profile/RoomGap'
import ScheduleRanks from './profile/ScheduleRanks'
import UsageLine from './profile/UsageLine'
import ValueNeighbors from './profile/ValueNeighbors'
import WeekByWeek from './profile/WeekByWeek'
import VerdictStrip, { type VerdictFigure } from './profile/VerdictStrip'
import { fmtRank, fmtSigned, hasHistory, ordinal, type ProfileHeader, type ProfilePayload } from './profile/payload'

// Everything the opener already knew about this player, so the profile can
// paint on the frame it opens instead of behind a skeleton.
//
// This exists because GET /api/players/{id}/profile is slow -- measured warm
// against the live server, twice, at 3.50s and 3.51s -- and the draft room
// opens this over a running 30-second pick clock. Waiting three and a half
// seconds to learn a name the room was already displaying is not a load, it
// is a stall. The un-embedded fallback path (see `embedded` below) passes no
// seed and still gets the skeleton it always did.
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
  // Rendered inside PlayerOverlay's popup (the only caller left -- the
  // standalone /players/:slug page this used to also back is gone): drop the
  // chrome the overlay supplies itself (its own close control, its own
  // Escape handler) and stop claiming the full viewport height.
  embedded?: boolean
}

function depthSlotLabel(position: string, depthSlot: number | null): string | null {
  if (depthSlot === null) return null
  return `${position}${depthSlot}`
}

// The verdict, read off the board row -- the un-embedded fallback's version
// of the figure strip (see `embedded` above), where there is no seed because
// nothing opened this from a list it had already ranked. Same five facts the
// sparse artboard leads with, in the same order.
function headerFigures(h: ProfileHeader): VerdictFigure[] {
  return [
    { label: 'Board rank', value: `#${h.rank}`, accent: true },
    { label: 'Tier', value: h.tier === null || h.tier === undefined ? '—' : `T${h.tier}` },
    { label: 'Projected', value: h.proj_points === null ? '—' : String(Math.round(h.proj_points)) },
    { label: 'Over replacement', value: fmtSigned(h.vor) },
    { label: 'ADP', value: fmtRank(h.market_rank) },
  ]
}

// The o-line section's heading, which is about where the player stands
// relative to the five men in front of him -- a running back runs behind
// them, a quarterback is protected by them, a kicker only needs them to get
// the offence close enough.
function lineHeading(position: string): string {
  if (position === 'RB') return 'The line he runs behind'
  if (position === 'QB') return 'The line protecting him'
  if (position === 'K') return 'The line that gets him in range'
  return 'The line in front of him'
}

export default function PlayerProfile({
  playerId, onClose, onToggleDrafted, onSelectPlayer, seed = null, embedded = false,
}: PlayerProfileProps) {
  const [profile, setProfile] = useState<ProfilePayload | null>(null)
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
      // `fetchProfile` is typed against api.ts's PlayerProfileData, which
      // describes the payload as it was before the card redesign; the seven
      // keys the card actually reads are named in profile/payload.ts. See
      // that file for why they live there and not in api.ts.
      const data = await fetchProfile(forPlayerId) as unknown as ProfilePayload
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

  // Who this page is about, from whichever source knows first: the loaded
  // profile's own header row, or the seed the opener handed over. The two
  // never disagree about identity -- the seed is read off the same board row
  // the profile is built from -- so the swap when the request lands is
  // invisible.
  const ident = header ?? seed

  // A seed is a painted header, so there is nothing for a skeleton to stand
  // in for. Without one this is unchanged: skeleton until the request lands.
  if (loading && !profile && !seed) return <PageSkeleton />

  // The strip never re-renders from the response when a seed exists. The
  // room's figures are the ones the pick is being made on and the payload
  // cannot reproduce three of them, so swapping them out for the board's
  // five when the request lands would replace the better numbers with worse
  // ones and move the reader's eye while it was on them.
  const figures = seed ? seed.figures : header ? headerFigures(header) : []

  // Meta line, in the design's order: team, depth slot, age, service year,
  // bye. Age and service year are the payload's alone (`bio`), so the line
  // grows by two facts when the request lands -- inline, in a line that is
  // already on screen, rather than as a block that appears and shoves
  // everything under it down.
  const meta: string[] = []
  if (ident) meta.push(ident.team ?? '—')
  const depthSlot = header && profile
    ? depthSlotLabel(header.position, profile.outlook.depth_slot) : null
  if (depthSlot) meta.push(depthSlot)
  if (profile?.bio.age !== null && profile?.bio.age !== undefined) {
    meta.push(`age ${profile.bio.age}`)
  }
  if (profile?.bio.nfl_season !== null && profile?.bio.nfl_season !== undefined) {
    meta.push(`${ordinal(profile.bio.nfl_season)} NFL season`)
  }
  const bye = profile?.outlook.bye ?? ident?.bye ?? null
  meta.push(`bye ${bye ?? '—'}`)

  const sparse = profile !== null && !hasHistory(profile)
  const lastSeason = profile && profile.game_log.length > 0
    ? Math.max(...profile.game_log.map((g) => g.season))
    : null

  return (
    <div className={`player-page${embedded ? ' player-page-embedded' : ''}`}>
      {!embedded && (
        <button type="button" className="player-page-back" onClick={onClose}>
          ← Back to board
        </button>
      )}
      {ident && (
        <div className="pp-header">
          <div className="pp-ident">
            <div className="pp-name-row">
              <span className={`pos-badge pos-badge-${ident.position.toLowerCase()}`}>
                {ident.position}
              </span>
              <h2 className="pp-name">
                {ident.name}
                {ident.rookie && <span className="rookie-badge">R</span>}
              </h2>
            </div>
            <p className="mono pp-meta">
              {meta.join(' · ')}
              {!profile && loading && (
                <span className="pp-loading-tail">
                  {' · '}
                  <span className="pp-loading-dot" aria-hidden="true" />
                  <span role="status">loading full profile…</span>
                </span>
              )}
            </p>
            {/* The injury line sits in the header because that is where it
                changes a pick: a designation can make every number below it
                irrelevant. Absent from the payload until the change that
                adds it lands, and absent for every defense after that, so
                it is read defensively rather than assumed. */}
            {profile?.status && <InjuryStatus status={profile.status} />}
          </div>
          <VerdictStrip figures={figures} />
          {onToggleDrafted && header && (
            <button type="button" className="drawer-draft-btn" onClick={handleToggleDraftedClick}>
              {header.drafted ? 'Undo draft' : 'Mark drafted'}
            </button>
          )}
        </div>
      )}

      {/* Below the header and the verdict strip, not above them: when this
          arrives it arrives late (the request has to fail first), and a
          block inserted above a header that is already painted would shove
          the whole profile down under the reader's eyes. */}
      {error && <p className="error">{error}</p>}

      {header && profile && (
        <div className="pp-grid">
          {sparse ? (
            <>
              {/* A card with no history makes ONE claim -- the gap between
                  what the board thinks and what the room charges -- and then
                  says what is missing and why. Six panels of em-dashes would
                  be worse than the admission. */}
              <section className="pp-card pp-span12 pp-card-lead">
                <h3 className="is-accent">Where this disagrees with the room</h3>
                <RoomGap rank={header.rank} marketRank={header.market_rank} edge={header.edge} />
              </section>

              <section className="pp-card pp-span7">
                <h3>What this card can’t show, and why</h3>
                <MissingData payload={profile} />
                {profile.outlook.implied_points !== null && (
                  <p className="pp-implied">
                    <span className="mono pp-implied-value">
                      {profile.outlook.implied_points.toFixed(1)}
                    </span>
                    <span>
                      points a game implied for {header.team}&apos;s own offence by
                      this season&apos;s betting lines — the one forward-looking
                      number that still applies
                      {header.position === 'DST' && ', though it describes the'
                        + ' side of the ball that leaves the field when this unit'
                        + ' comes on'}.
                    </span>
                  </p>
                )}
              </section>

              <section className="pp-card pp-span5">
                <h3>Others at this value</h3>
                <p className="pp-sub">nearest by board rank, not by stat line</p>
                <ValueNeighbors
                  players={profile.similar.players}
                  onSelectPlayer={onSelectPlayer}
                />
              </section>
            </>
          ) : (
            <>
              <section className="pp-card pp-span7">
                <h3>Season by season</h3>
                <p className="pp-sub">
                  volatility is scored per point — spread divided by average, so a
                  bigger scorer isn’t punished for scoring
                </p>
                {/* The chart before the table: a career's shape is the one
                    thing a column of numbers is worst at showing, and
                    "TE40 -> TE7 -> TE2 -> TE1" is the whole story of a
                    breakout in four bars. The table underneath is where you
                    go once the shape has made you curious. */}
                <SeasonFinish seasons={profile.seasons} position={header.position} />
                <ConsistencyTable seasons={profile.seasons} position={header.position} />
                {/* The old card led with nine stat tiles; the redesign leads
                    with the verdict instead. The usage they carried is not
                    lost, it is one line under the seasons it was averaged
                    from. */}
                <UsageLine summary={profile.summary} position={header.position} />
              </section>

              {profile.cohort && (
                <section className="pp-card pp-span5">
                  <h3 className="is-accent">What players like him did next</h3>
                  <CohortNext cohort={profile.cohort} />
                </section>
              )}

              {lastSeason !== null && (
                <section className="pp-card pp-span7">
                  <h3>{lastSeason} week by week</h3>
                  <WeekByWeek games={profile.game_log} position={header.position} />
                </section>
              )}

              <section className="pp-card pp-span5">
                <h3>Where the market has him</h3>
                <MarketRow
                  marketRank={header.market_rank}
                  marketSpread={header.market_spread}
                  sources={header.market_sources}
                />
              </section>
            </>
          )}

          {/* Everything below here is common to both cards: a defense has a
              schedule of its own, a rookie has a line in front of him, and
              neither has to be a special case to say so. Each section is
              omitted when its own data is empty rather than drawn as a row
              of dashes. */}
          {profile.schedule.length > 0 && (
            <section className="pp-card pp-span7">
              <h3>{profile.bio.season} schedule</h3>
              <ScheduleRanks weeks={profile.schedule} />
            </section>
          )}

          {profile.oline && (
            <section className="pp-card pp-span5">
              <h3>{lineHeading(header.position)}</h3>
              <LineQuality oline={profile.oline} />
            </section>
          )}

          {!sparse && (
            <section className="pp-card pp-span7">
              <h3>Comparable players</h3>
              <SimilarPlayers
                mode={profile.similar.mode}
                players={profile.similar.players}
                targetAge={profile.similar.target_age ?? null}
                onSelectPlayer={onSelectPlayer}
              />
            </section>
          )}

          {profile.depth_chart.length > 0 && (
            <section className="pp-card pp-span5">
              <h3>Depth chart</h3>
              <DepthChartCard team={header.team} groups={profile.depth_chart} />
            </section>
          )}

          {/* News, last in the grid: it is the only section whose length is
              unbounded, and it is the one a manager reads after the numbers
              rather than instead of them. Rendered only when the payload
              actually carries items -- the `news` key arrives with a change
              landing alongside this card, and is an empty list both for a
              player nobody wrote about and for every defense. An empty
              "News" card with a dash in it would be a promise the payload
              cannot keep. */}
          {profile.news && profile.news.length > 0 && (
            <section className="pp-card pp-span12">
              <h3>Recent news</h3>
              <NewsPanel items={profile.news} />
            </section>
          )}
        </div>
      )}
    </div>
  )
}
