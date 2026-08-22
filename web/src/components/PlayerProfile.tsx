import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { fetchProfile, type LiveSettings, type Player } from '../api'
import { Chart, type Col } from './draft/Chart'
import { startersAt } from './draft/finish'
import { finishCols, healthCols, perGameCols, seasonLength, steadyCols, year } from './draft/panels'
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
import { fmtRank, fmtSigned, hasHistory, type PlayerStatus, type ProfileHeader, type ProfilePayload } from './profile/payload'

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
  // The league's own shape, straight through from the room (DraftRoom ->
  // PlayerOverlay). The Finish panel grades a season against how many of a
  // position START here, so a popup left to the twelve-team fallback would
  // colour the same season differently from the board column three inches
  // behind it. Optional and defaulted for the same reason `startersAt` has a
  // fallback at all: before a session is connected there is no league to ask.
  settings?: LiveSettings | null
  // Rendered inside PlayerOverlay's popup (the only caller left -- the
  // standalone /players/:slug page this used to also back is gone): drop the
  // chrome the overlay supplies itself (its own close control, its own
  // Escape handler) and stop claiming the full viewport height.
  embedded?: boolean
}

// -- the popup, top to bottom ----------------------------------------------
//
// Header, status line, then the four season panels. The panels are drawn by
// the board's own `Chart` off the board's own column builders
// (draft/panels.ts), never by a second copy of either: someone who has spent
// a draft learning what a tall green column means in a hover panel should not
// have to learn it again the moment the same seasons open in a popup.

// The depth slot, from whichever source actually has one. Sleeper's chart
// (`status`) knows a rookie's place before the board's `outlook` does --
// Jeremiyah Love is charted RB1 in Arizona with `depth_slot` still null --
// and it carries the position he is charted AT, which is not always the
// position the board ranks him at.
function depthSlotLabel(
  position: string, status: PlayerStatus | null, depthSlot: number | null,
): string | null {
  const order = status?.depth_chart_order ?? depthSlot
  if (!order) return null
  return `${status?.depth_chart_position ?? position}${order}`
}

// How loud the disagreement with the room is allowed to be. Under ten slots
// your board and the market take him in the same round of any league this
// tool supports (8 to 14 teams), so there is no decision in the gap and it
// reads as a note. Ten or more is a round he would fall past, which is the
// version of this number worth colouring as a bargain.
const ROUND = 10

function edgeTone(slots: number): string {
  if (slots >= ROUND) return 'is-good'
  if (slots > 0) return 'is-accent'
  // Zero is agreement, not a failure -- it takes the row's own body colour.
  if (slots === 0) return ''
  return 'is-bad'
}

// A rookie's Per game panel is one column, the projection, with nothing to
// be read against. The seasons he did not play go back on the axis as
// baseline marks: the panel is the shape of a career, and "there is not one
// yet" is a shape. Three, because three is what the veterans beside him show.
const BLANK_SEASONS = 3

function blankSeasonCols(season: number): Col[] {
  return Array.from({ length: BLANK_SEASONS }, (_, i) => {
    const s = season - BLANK_SEASONS + i
    return {
      key: s,
      label: year(s),
      value: '·',
      tone: '',
      fill: 0,
      // Not a zero: he did not play, which `Chart` draws as a mark on the
      // baseline rather than as a column of no height.
      empty: true,
    }
  })
}

// The four figures the pick is made on, right-aligned across from the name.
// All four are the loaded payload's own. The seed the room paints from
// carries a different set per source -- a live candidate has no board rank
// on it at all -- and a strip that swapped its columns when the request
// landed would move the reader's eye off the number it was already on.
function PopFigures({ header }: { header: ProfileHeader }): ReactNode {
  const figures = [
    // First and in accent: the only one of the four this app computes rather
    // than reports, and the one the other three are read against.
    { label: 'Your board', value: String(header.rank), accent: true },
    { label: 'ADP', value: fmtRank(header.market_rank) },
    {
      label: 'Tier',
      value: header.tier === null || header.tier === undefined ? '—' : String(header.tier),
    },
    // Whole points: a value over replacement is a season total, and its
    // decimals are noise beside three ranks.
    { label: 'VOR', value: header.vor === null ? '—' : String(Math.round(header.vor)) },
  ]
  return (
    <div className="pp-pop-figures">
      {figures.map((f) => (
        <div key={f.label} className="pp-pop-figure">
          <span className="pp-pop-figure-label">{f.label}</span>
          <span className={`mono pp-pop-figure-value${f.accent ? ' is-accent' : ''}`}>
            {f.value}
          </span>
        </div>
      ))}
    </div>
  )
}

// One line for the two facts that can make every panel under it irrelevant --
// a designation, and where he actually stands on his own depth chart -- with
// the edge over the room on its right end. That edge had a card of its own
// (RoomGap); it is one number, and this is the line it belongs on.
function StatusLine({ profile }: { profile: ProfilePayload }): ReactNode {
  const status = profile.status ?? null
  // NEVER "healthy": a null designation is the absence of a claim, not a
  // clean bill of health. InjuryStatus renders nothing for it (see its own
  // comment), so this row says the thing it can actually source instead.
  const flagged = status !== null && status.injury_status !== null
  const slot = depthSlotLabel(profile.header.position, status, profile.outlook.depth_slot)
  const { edge, market_rank: marketRank, team } = profile.header
  const slots = edge === null ? null : Math.round(edge)
  return (
    <div className={`pp-pop-status${flagged ? ' is-flagged' : ''}`}>
      {/* `currentColor`, so the row's own state colours the mark once. */}
      <svg
        className="pp-pop-status-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
      >
        {flagged ? (
          <>
            <path d="M12 9v5" />
            <path d="M12 17.5v.01" />
            <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
          </>
        ) : (
          <path d="M20 6 9 17l-5-5" />
        )}
      </svg>
      <p className="pp-pop-status-text">
        {flagged && status
          ? <InjuryStatus status={status} />
          : 'No injury designation'}
        {slot !== null && (
          <>
            {' · '}
            <span className="pp-pop-slot">{slot}</span>
            {` on ${team === null ? 'the' : `${team}’s`} depth chart`}
          </>
        )}
      </p>
      {slots !== null && marketRank !== null && (
        <span className={`mono pp-pop-edge ${edgeTone(slots)}`}>
          {`${fmtSigned(edge)} ${Math.abs(slots) === 1 ? 'slot' : 'slots'} vs consensus`}
        </span>
      )}
    </div>
  )
}

function PopPanel({ title, note, cols, wide = false }: {
  title: string; note: string; cols: Col[]; wide?: boolean
}): ReactNode {
  return (
    <section className={`pp-pop-panel${wide ? ' is-wide' : ''}`}>
      {/* The hover panel's own head, one step smaller (see App.css): a title
          and the yardstick the numbers under it are out of. */}
      <div className="ctip-head">
        <span>{title}</span>
        <span className="ctip-head-note">{note}</span>
      </div>
      <Chart cols={cols} />
    </section>
  )
}

// Health, Finish, Steady, Per game -- the same four the board sorts on, in
// the same order the board's columns run. A panel is DROPPED rather than
// drawn empty when its own data is missing: a rookie has no games to count,
// no finish to place and no coefficient to rank, and three cards of baseline
// marks would be three ways of saying the same nothing. Per game survives
// because a projection is not history.
function PopPanels({ profile, settings }: {
  profile: ProfilePayload; settings?: LiveSettings | null
}): ReactNode {
  const { header, seasons } = profile
  const health = healthCols(seasons)
  const finish = finishCols(seasons, startersAt(header.position, settings))
  const steady = steadyCols(seasons)
  // Newest first in the payload, so the first rated season carries the pool
  // the last column was ranked in. A denominator that moves year to year,
  // and this note is here to make that column legible, not to average them.
  const rated = seasons.filter((s) => s.cv_rank_n)
  const played = seasons.filter((s) => s.games > 0)
  const proj = profile.summary?.proj_ppg ?? null
  const scoring = perGameCols(seasons, proj, header.position)
  const perGame = played.length
    ? scoring : [...blankSeasonCols(profile.bio.season), ...scoring]
  const delta = played.length && proj !== null ? proj - played[0].ppg : null
  // A defense has none of these: no games counted, no positional finish, no
  // coefficient, and no per-game projection either. An empty row would still
  // cost the gap above the section under it.
  if (!health.length && !finish.length && !steady.length && !scoring.length) return null
  return (
    <div className="pp-pop-panels">
      {health.length > 0 && (
        <PopPanel
          title="Health"
          // Off the last column's own season, not off today: a 16-game 2019
          // is a full year, and "of 17" against it would invent an injury.
          note={`of ${seasonLength(health[health.length - 1].key as number)}`}
          cols={health}
        />
      )}
      {finish.length > 0 && (
        <PopPanel title="Finish" note={header.position} cols={finish} />
      )}
      {steady.length > 0 && (
        <PopPanel title="Steady" note={`of ${rated[0].cv_rank_n}`} cols={steady} />
      )}
      {scoring.length > 0 && (
        <PopPanel
          title="Per game"
          note={delta === null
            ? 'projection only'
            : `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}`}
          cols={perGame}
          wide
        />
      )}
    </div>
  )
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
  playerId, onClose, onToggleDrafted, onSelectPlayer, seed = null, settings = null,
  embedded = false,
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

  // Meta line, in the artboard's order: team, age, service year, bye. The
  // depth slot left it for the status line, which is where a designation and
  // a depth chart now argue with each other in one sentence.
  //
  // Age and service year are the payload's alone (`bio`), so the line grows
  // by two facts when the request lands -- inline, in a line that is already
  // on screen, rather than as a block that appears and shoves everything
  // under it down.
  const meta: string[] = []
  if (ident) meta.push(ident.team ?? '—')
  const age = profile?.bio.age ?? null
  // "rookie" where a veteran carries an age: `bio` is all-null for a player
  // nflverse has no biography for, which is every rookie until he has
  // played. His first NFL year is the one fact about him that needs none.
  if (age !== null) meta.push(`age ${age}`)
  else if (ident?.rookie) meta.push('rookie')
  const nflSeason = profile?.bio.nfl_season ?? (ident?.rookie ? 1 : null)
  if (nflSeason !== null) meta.push(`NFL yr ${nflSeason}`)
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
        <div className="pp-pop-head">
          <div className="pp-pop-ident">
            {/* Name first, chip second: the artboard reads it as a sentence,
                and the position is the qualifier on the name rather than a
                label the eye has to step over to reach it. */}
            <div className="pp-pop-name-row">
              <h2 className="pp-pop-name">{ident.name}</h2>
              <span className={`pos-badge pos-badge-${ident.position.toLowerCase()} pp-pop-pos`}>
                {ident.position}
              </span>
            </div>
            <p className="mono pp-pop-meta">
              {meta.join(' · ')}
              {!profile && loading && (
                <span className="pp-loading-tail">
                  {' · '}
                  <span className="pp-loading-dot" aria-hidden="true" />
                  <span role="status">loading full profile…</span>
                </span>
              )}
            </p>
          </div>
          {header && <PopFigures header={header} />}
          {onToggleDrafted && header && (
            <button type="button" className="drawer-draft-btn" onClick={handleToggleDraftedClick}>
              {header.drafted ? 'Undo draft' : 'Mark drafted'}
            </button>
          )}
        </div>
      )}

      {/* Below the header, not above it: when this arrives it arrives late
          (the request has to fail first), and a block inserted above a
          header that is already painted would shove the whole profile down
          under the reader's eyes. */}
      {error && <p className="error">{error}</p>}

      {profile && <StatusLine profile={profile} />}
      {profile && <PopPanels profile={profile} settings={settings} />}

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
