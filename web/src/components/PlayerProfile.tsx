import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import type { LiveSettings, Player } from '../api'
import { Chart, type Col } from './draft/Chart'
import { startersAt } from './draft/finish'
import { SEASON_GAMES } from './draft/weeks'
import {
  finishCols, healthCols, perGameCols, perGameDelta, playedSeasons, ratedSeasons,
  seasonLength, steadyCols, year,
} from './draft/panels'
import DepthChartCard from './DepthChartCard'
import PageSkeleton from './PageSkeleton'
import { cachedProfile, forgetProfile, loadProfile as fetchSharedProfile } from './draft/CellTip'
import ComparableSeasons from './profile/ComparableSeasons'
import InjuryStatus, { injuryTone } from './profile/InjuryStatus'
import NewsPanel from './profile/NewsPanel'
import PopCard from './profile/PopCard'
import { CARD_HINTS, finishHint } from './profile/hints'
import LineQuality from './profile/LineQuality'
import MarketRow from './profile/MarketRow'
import MissingData from './profile/MissingData'
import ScheduleRanks from './profile/ScheduleRanks'
import SimilarPlayers from './profile/SimilarPlayers'
import UsageLine from './profile/UsageLine'
import VegasCard from './profile/VegasCard'
import WeekByWeek from './profile/WeekByWeek'
import { hasHistory, type PlayerStatus, type ProfileHeader, type ProfilePayload } from './profile/payload'

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
// waiting, whether he lasts, which roster slot he fills) come from
// /api/live/state's ranked list, carry rounding and sign rules this
// component has no business knowing, and are not even defined outside a
// draft. A label/value pair is the widest contract that stays honest.
// `accent` marks the one figure worth highlighting (an open starter slot,
// same rule as `.confirm-figure.is-open`).
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
  // Takes the player this profile is describing. Present ONLY while the room
  // could actually send that pick -- see DraftRoom, which owns that question
  // -- so the footer can offer the board back instead of a button that would
  // be refused. It opens the confirm dialog; it does not pick.
  //
  // Not to be confused with `onToggleDrafted` above, which is the manual
  // research-board flag. This one is a real pick, on ESPN's socket.
  onDraftPlayer?: (playerId: string) => void
  // Rendered inside PlayerOverlay's popup (the only caller left -- the
  // standalone /players/:slug page this used to also back is gone): drop the
  // chrome the overlay supplies itself (its own close control, its own
  // Escape handler) and stop claiming the full viewport height.
  embedded?: boolean
}

// -- the popup, top to bottom ----------------------------------------------
//
// Header, status line, the notice a thin payload earns, the four season
// panels, last season by week, and then three rows of dense cards. The
// panels are drawn by the board's own `Chart` off the board's own column
// builders (draft/panels.ts), never by a second copy of either: someone who
// has spent a draft learning what a tall green column means in a hover panel
// should not have to learn it again the moment the same seasons open in a
// popup.

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

// Generational suffixes, which are not surnames: ten of the 252 players on
// this board carry one, so "DRAFT III" is not a hypothetical.
const NAME_SUFFIXES = new Set(['jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'v'])

// What the footer button calls him: the surname alone, as the artboard has
// it ("DRAFT GIBBS"). One word, because the button is read under a pick
// clock by someone who is already looking at his full name in the header.
function draftName(name: string, position: string): string {
  // A defense has no surname, and every one of them ends in the same word --
  // "Houston Defense" would read as DRAFT DEFENSE. The city is the name.
  if (position === 'DST') return name.replace(/\s+Defense$/, '')
  const parts = name.trim().split(/\s+/)
  while (parts.length > 1 && NAME_SUFFIXES.has(parts[parts.length - 1].toLowerCase())) {
    parts.pop()
  }
  return parts[parts.length - 1] || name
}

// The four figures the pick is made on, right-aligned across from the name.
// All four are the loaded payload's own. The seed the room paints from
// carries a different set per source -- a live candidate has no board rank
// on it at all -- and a strip that swapped its columns when the request
// landed would move the reader's eye off the number it was already on.
function PopFigures({ header, profile }: {
  header: ProfileHeader
  profile: ProfilePayload | null
}): ReactNode {
  // The projection this header row carries, per game -- the number both
  // figures below are read off, so the pair cannot disagree with each other
  // the way two separately-sourced projections would.
  const projPpg = header.proj_points === null
    ? null : header.proj_points / SEASON_GAMES
  // Null for the thin state (header, no payload yet) and for a player with
  // no season played to move from -- see perGameDelta.
  const delta = profile === null ? null : perGameDelta(profile.seasons, projPpg)
  // Where last season actually finished him among his own position, which is
  // what the projected place is a move FROM. `pos_finish` rather than
  // `pos_rank_ppg` because `proj_pos_finish` ranks on season points too --
  // comparing a per-game place against a total-points one would print a move
  // nobody made.
  const played = profile === null ? [] : playedSeasons(profile.seasons)
  const lastFinish = played.length ? played[0].pos_finish : null
  // Rank runs backwards, so the subtraction is the other way round from the
  // points one: a projection of WR7 against a WR10 season is three places
  // BETTER, and the arrow has to say up.
  const rankMove = lastFinish === null || header.proj_pos_finish === null
    ? null : lastFinish - header.proj_pos_finish
  const figures: {
    label: string; value: string; accent?: boolean
    delta?: { label: string; tone: string }
  }[] = [
    // Per game, not for the season, for the reason the board's own Proj/G
    // column gives: 313 is a number nobody has a feel for and 18.4 is a
    // Sunday. Same divisor as that column (SEASON_GAMES), so the two cannot
    // disagree about what "per game" means.
    //
    // The move against last season's average rides with it: a projection is
    // a claim, and how far it is from what he actually did is the part of
    // the claim worth checking. Nothing when it rounds to nothing -- "+0.0"
    // is a delta that says the number beside it already stands.
    {
      label: 'Proj/G',
      value: projPpg === null ? '—' : projPpg.toFixed(1),
      accent: true,
      delta: delta === null || delta.zero
        ? undefined : { label: delta.label, tone: delta.tone },
    },
    // The same projection as a PLACE among his own position -- WR7 -- which
    // is the unit a drafter thinks in, with the places he moved since last
    // season beside it. An arrow rather than a sign: on a rank, "+3" reads
    // as a bigger number and a bigger number is worse.
    {
      label: 'Pos rank',
      value: header.proj_pos_finish === null
        ? '—' : `${header.position}${header.proj_pos_finish}`,
      delta: rankMove === null || rankMove === 0 ? undefined : {
        label: `${rankMove > 0 ? '\u25b2' : '\u25bc'}${Math.abs(rankMove)}`,
        tone: rankMove > 0 ? 'is-up' : 'is-down',
      },
    },
    // And where the room puts him overall, which is the number every other
    // figure here is a component of and none of them state: "WR50" is a
    // place among receivers, and a draft is one list. To the right of the
    // positional place, because that is what a roster is filled by and this
    // is the scale it sits on.
    { label: 'Rank', value: String(header.rank) },
  ]
  return (
    <div className="pp-pop-figures">
      {figures.map((f) => (
        <div key={f.label} className="pp-pop-figure">
          <span className="pp-pop-figure-label">{f.label}</span>
          <span className="pp-pop-figure-line">
            <span className={`mono pp-pop-figure-value${f.accent ? ' is-accent' : ''}`}>
              {f.value}
            </span>
            {f.delta && (
              <span className={`mono pp-pop-figure-delta delta-tone ${f.delta.tone}`}>
                {f.delta.label}
              </span>
            )}
          </span>
        </div>
      ))}
    </div>
  )
}

// Height and weight as one item on the meta line. Feet and inches because 74
// is a number nobody carries around and 6'2" is how a height is said;
// pounds because the payload's own unit is pounds.
//
// Either half alone still prints. A payload that knows his weight and not
// his height should say the weight -- the alternative is a card that drops a
// fact it holds because it is missing a different one.
function fmtBuild(height: number | null, weight: number | null): string | null {
  const parts: string[] = []
  if (height !== null) parts.push(`${Math.floor(height / 12)}'${height % 12}"`)
  if (weight !== null) parts.push(`${weight} lb`)
  return parts.length === 0 ? null : parts.join(' ')
}

// The mark on the corner of his photo. A designation is the one fact on this
// card that can make every number under it irrelevant, so it belongs where
// the eye lands first rather than in a band below the header -- and as a
// mark, so a reader who is scanning several profiles sees "there is
// something here" before he reads what.
//
// On the photo rather than after the name, which is where it used to sit. A
// triangle in the name line is read as punctuation on the name -- it took
// its place in the reading order between "Malik Nabers" and the meta under
// it, and a name line that can grow a third element is a name line whose
// length depends on a fact about the player's knee. Pinned to a corner of a
// fixed 44px square it is the same mark in the same place on every profile,
// which is what makes it scannable across several.
//
// It falls BACK to the name line for a player with no photo (see the head
// below): a defense, or anyone nflverse has no picture of. The corner of a
// photo that is not there is nowhere, and this is the one mark on the card
// that must not go missing.
//
// Severity is the pill's own (`InjuryStatus`'s SEVERE/WATCH lists), read off
// the same strings, so a red triangle here and a red pill under it cannot
// disagree. The full designation and its detail ride on the tooltip; the
// band below carries them in text for anyone who does not hover, which is
// why this is an icon and not a second copy of the sentence.
function InjuryMark({ status }: { status: PlayerStatus | null }): ReactNode {
  if (status === null || status.injury_status === null) return null
  const said = [status.injury_status, status.injury_body_part, status.injury_notes]
    .filter((s): s is string => typeof s === 'string' && s.trim() !== '')
    .join(' — ')
  return (
    <span className={`pp-pop-injury-mark ${injuryTone(status.injury_status)}`}
          title={said} role="img" aria-label={said}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4"
           strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M12 9v5" />
        <path d="M12 17.5v.01" />
        <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
      </svg>
    </span>
  )
}

// One line for the two facts that can make every panel under it irrelevant --
// a designation, and where he actually stands on his own depth chart -- with
// the edge over the room on its right end. That edge used to have a card of
// its own; it is one number, and this is the line it belongs on.
function StatusLine({ profile }: { profile: ProfilePayload }): ReactNode {
  const status = profile.status ?? null
  // NEVER "healthy": a null designation is the absence of a claim, not a
  // clean bill of health. So the row is drawn ONLY when there is a
  // designation to draw -- it used to render either way and spend a full
  // band of the card saying "No injury designation", which is a sentence
  // about the absence of news. The two facts it carried for everyone else
  // have moved to where they are always true: the depth slot into the
  // header's own meta line, and the edge onto the Market card, which is the
  // card about disagreeing with the market.
  if (status === null || status.injury_status === null) return null
  const slot = depthSlotLabel(profile.header.position, status, profile.outlook.depth_slot)
  const { team } = profile.header
  return (
    <div className="pp-pop-status is-flagged">
      {/* `currentColor`, so the row's own state colours the mark once. */}
      <svg
        className="pp-pop-status-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
      >
        <path d="M12 9v5" />
        <path d="M12 17.5v.01" />
        <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
      </svg>
      <p className="pp-pop-status-text">
        <InjuryStatus status={status} />
        {slot !== null && (
          <>
            {' · '}
            <span className="pp-pop-slot">{slot}</span>
            {` on ${team === null ? 'the' : `${team}’s`} depth chart`}
          </>
        )}
      </p>
    </div>
  )
}

function PopPanel({ title, note, hint, cols, wide = false }: {
  // ReactNode, not string: Per game's note is a signed number that is
  // coloured by its own sign, so it arrives as an element rather than text.
  title: string; note: ReactNode; hint?: string; cols: Col[]; wide?: boolean
}): ReactNode {
  return (
    // The same card as the six below (PopCard), plus the class that says
    // this one holds a chart: how a long career scrolls inside it, and how
    // much wider Per game is allowed to be, are the only two things a panel
    // decides for itself.
    <PopCard title={title} note={note} hint={hint}
             className={`pp-pop-panel${wide ? ' is-wide' : ''}`}>
      <Chart cols={cols} />
    </PopCard>
  )
}

// The most recent seasons the popup's panels draw, newest first in the
// payload. See PopPanels for why the popup caps a career and the board's own
// hover panels do not.
const POPUP_SEASONS = 5

// Health, Finish, Steady, Per game -- the same four the board sorts on, in
// the same order the board's columns run. A panel is DROPPED rather than
// drawn empty when its own data is missing: a rookie has no games to count,
// no finish to place and no coefficient to rank, and three cards of baseline
// marks would be three ways of saying the same nothing. Per game survives
// because a projection is not history.
function PopPanels({ profile, settings }: {
  profile: ProfilePayload; settings?: LiveSettings | null
}): ReactNode {
  const { header } = profile
  // The popup's own cap, applied here and nowhere else: `panels.ts` still
  // builds a whole career for the board's hover panels, where there is room
  // for one. Four cards a quarter of a 562px popup wide hold about five
  // columns before they start scrolling, and a nine-season back
  // (McCaffrey) opened on 2017 -- the seasons that decide a 2026 pick were
  // off the left edge of every panel. Five, weighted toward the recent, is
  // already how this project answers "what is he now": see RECENCY_WEIGHTS
  // in draft/finish.ts.
  const seasons = profile.seasons.slice(0, POPUP_SEASONS)
  const health = healthCols(seasons)
  const finish = finishCols(seasons, startersAt(header.position, settings),
                            header.proj_pos_finish, header.position)
  // The builder's own filter (`panels.ts`), never a local copy: the board's
  // hover panel draws the same seasons for the same player three inches
  // behind this popup, and a looser test here would draw one neither chart
  // agrees on.
  const played = playedSeasons(seasons)
  const proj = profile.summary?.proj_ppg ?? null
  const scoring = perGameCols(seasons, proj, header.position)
  const perGame = played.length
    ? scoring : [...blankSeasonCols(profile.bio.season), ...scoring]
  const delta = perGameDelta(seasons, proj)
  // A defense has none of these: no games counted, no positional finish and
  // no per-game projection either. An empty row would still cost the gap
  // above the section under it. (Steady is not in this row any more -- see
  // SteadyPanel, which lives beside Schedule and Room.)
  if (!health.length && !finish.length && !scoring.length) return null
  return (
    <div className="pp-pop-row">
      {health.length > 0 && (
        <PopPanel
          title="Health"
          // Off the last column's own season, not off today: a 16-game 2019
          // is a full year, and "of 17" against it would invent an injury.
          note={`of ${seasonLength(health[health.length - 1].key as number)}`}
          hint={CARD_HINTS.health}
          cols={health}
        />
      )}
      {finish.length > 0 && (
        // "WR rank", and then bare numbers -- except the projection, which
        // names the position itself (see `finishCols`). The position was on
        // every column for a revision ("WR43 WR12 WR4") and said the same
        // word five times to qualify five numbers in the same unit, which is
        // what a title is for. On the forecast alone it is not repetition:
        // that is the column a reader quotes elsewhere.
        <PopPanel title={`${header.position} rank`} note={undefined}
                  hint={finishHint(header.position)} cols={finish} />
      )}
      {scoring.length > 0 && (
        <PopPanel
          title="Points per game"
          // Green up, red down: the only number in the four panels that is a
          // direction rather than a level, and the direction is why it is
          // there. The panel's own columns already say the levels.
          note={delta === null ? 'projection only'
            : <span className={`delta-tone ${delta.tone}`}>{delta.label}</span>}
          hint={CARD_HINTS.perGame}
          cols={perGame}
          wide
        />
      )}
    </div>
  )
}

// Steady, drawn with the CARDS rather than with the season panels above.
//
// It is the same object -- a PopPanel, same frame, same chart -- and it was
// the fourth of four up there, which made the top row read as "here are four
// things about his scoring". Three of them are: games, finish, points a game.
// Steady is not a level at all, it is the variance around one, and a reader
// running along that row had to change what he was reading for on the last
// panel. Down here it sits with Schedule and Room, which are the other two
// cards that qualify a projection rather than state one -- how hard the
// season is, who else is in the room, and how reliable he is week to week.
//
// It carries its own slice of the payload rather than being handed one:
// PopPanels applies the same POPUP_SEASONS cap for its own three, and two
// callers sharing one derived list would be one more thing to keep in step
// for no gain -- `steadyCols` is a pure function of the seasons.
function SteadyPanel({ profile }: { profile: ProfilePayload }): ReactNode {
  const seasons = profile.seasons.slice(0, POPUP_SEASONS)
  const cols = steadyCols(seasons)
  // A rookie has no coefficient to rank and a defense never will have one.
  if (cols.length === 0) return null
  // Newest first in the payload, so the first rated season carries the pool
  // the last column was ranked in. A denominator that moves year to year,
  // and the note is here to make that column legible, not to average them.
  const rated = ratedSeasons(seasons)
  return (
    <PopPanel title="Reliable"
              note={rated.length ? `of ${rated[0].cv_rank_n}` : undefined}
              hint={CARD_HINTS.steady}
              cols={cols} />
  )
}

// Eight cards in three rows, under the seasons: where he plays, who he plays
// with, what the room charges for him, and what happened to the players his
// season looks like. Each is a card the same size as the panels above, and
// each decides for itself whether it has anything to say -- a rookie has no
// usage to average, a defense has no line in front of it, and a card with
// nothing in it is not drawn at all rather than drawn as a frame of dashes.
// That is why the rows carry no conditions of their own: see
// `.pp-pop-row:empty` in App.css for the one case that needs handling, a row
// where every card opted out.
function PopCards({ profile, onSelectPlayer }: {
  profile: ProfilePayload; onSelectPlayer: (id: string) => void
}): ReactNode {
  const { header } = profile
  return (
    <>
      <div className="pp-pop-row">
        <ScheduleRanks weeks={profile.schedule} sosPct={profile.outlook.sos_pct} />
        {/* The wider of the three: it lists a room, and a name cut short is a
            name you cannot recognise. */}
        <DepthChartCard
          team={header.team}
          position={header.position}
          groups={profile.depth_chart}
        />
        <SteadyPanel profile={profile} />
      </div>
      <div className="pp-pop-row">
        <UsageLine seasons={profile.seasons} position={header.position}
                   projected={profile.summary?.proj_usage} />
        {/* Beside Usage: both are the shape of his role, one as rates and one
            as the line in front of him. Null for every defense by
            construction (see LineQualityData) -- the o-line is a fact about
            the eleven who leave the field when that unit comes on. */}
        {profile.oline && <LineQuality oline={profile.oline} />}
      </div>
      {/* The two market cards, on their own line. One is what the betting
          market prices his offence at, the other what the fantasy market
          prices him at -- both are outside opinions, and reading them next
          to each other is the comparison worth making.
          Smallest first here, and only here: it puts the row's seam on the
          opposite side to the rows above and below, so the three big cards
          -- Usage, Vegas, Comparable seasons -- are the same width without
          the popup becoming one unbroken column of identical edges. */}
      <div className="pp-pop-row">
        <MarketRow
          rank={header.rank}
          marketRank={header.market_rank}
          marketSpread={header.market_spread}
          sources={header.market_sources}
          position={header.position}
          posRanks={header.market_pos}
          edge={header.edge}
          // Only for the thin state, where the whole popup carries one
          // forward-looking number and this is it -- for everyone else the
          // Per game panel and the week chart are already that, and a fifth
          // row here would be the market card answering a question nobody
          // asked it. Sparse.dc.html draws it exactly here.
          impliedPoints={hasHistory(profile) ? null : profile.outlook.implied_points}
        />
        <VegasCard vegas={profile.vegas} position={header.position} />
      </div>
      {/* Last row, and the only one that looks backwards: everything above
          it is this player now, and these are the seasons that already went
          where his might go, beside what has been written about him this
          week. Comparable seasons is the `stat_twins` half of `similar`, so
          a player with no stat line to match leaves News here on its own,
          which is what Sparse.dc.html draws. (Near you, one row up, draws
          for him either way now: the payload's own neighbours when it has
          them, the room's ranked board when it has not.) */}
      <div className="pp-pop-row">
        {/* Null for every defense by construction (see LineQualityData): the
            o-line is a fact about the eleven who leave the field when this
            unit comes on. Down here rather than beside Usage because it is
            the narrowest card in the popup -- a rank and two ratios -- and
            it was taking width from two cards that needed it. */}
        <ComparableSeasons
          mode={profile.similar.mode}
          players={profile.similar.players}
          targetAge={profile.similar.target_age ?? null}
          cohort={profile.cohort}
          onSelectPlayer={onSelectPlayer}
        />
        <NewsPanel items={profile.news ?? []} />
      </div>
    </>
  )
}

export default function PlayerProfile({
  playerId, onClose, onToggleDrafted, onSelectPlayer, seed = null, settings = null,
  onDraftPlayer, embedded = false,
}: PlayerProfileProps) {
  // Seeded from the room's shared profile cache, which is very often already
  // holding this player: the hover tip, the board-grid peek and this card are
  // three readers of ONE payload, and before this the card was the one that
  // refused to use it -- hovering a cell and then clicking it fetched the
  // same 48 KB twice, the second time behind a skeleton, for an endpoint that
  // costs 239 ms warm on a fast machine and multiples of that in production.
  // A hit paints the finished card on the frame it opens.
  const [profile, setProfile] = useState<ProfilePayload | null>(
    () => cachedProfile(playerId) as unknown as ProfilePayload | null)
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
    // A HIT IS NOT A LOAD. Showing a spinner over an answer already in hand
    // is the flicker this path exists to remove, and the hover that usually
    // precedes this click means the answer is usually in hand: the tip, the
    // board peek, the target cards and this card are four readers of ONE
    // payload (draft/CellTip's `cachedProfile`).
    //
    // The read below still runs on a hit. Inside the shared window it is a
    // lookup and nothing happens; past it, it is a real request and the card
    // quietly catches up. What it never does is blank in between -- which is
    // why the cache in front of it is a VIEW of the one store rather than a
    // second store with a longer memory.
    const hit = cachedProfile(forPlayerId) as unknown as ProfilePayload | null
    if (hit) {
      setProfile(hit)
      setError(null)
      setLoading(false)
    } else {
      setLoading(true)
      setError(null)
    }
    try {
      // `fetchSharedProfile` (draft/CellTip's `loadProfile`) rather than
      // api.ts's `fetchProfile` directly: it is the same request behind the
      // app's one request cache and its in-flight table, so the card, the
      // hover tip and the board peek asking for one player at once is ONE
      // request rather than three -- this card used to be the reader that
      // paid for the payload twice, once on hover and again on the click
      // that followed it. It is typed against api.ts's PlayerProfileData,
      // which describes the payload as it was before the card redesign; the
      // seven keys the card actually reads are named in profile/payload.ts.
      // See that file for why they live there and not in api.ts.
      const data = await fetchSharedProfile(forPlayerId) as unknown as ProfilePayload
      if (playerIdRef.current === forPlayerId) setProfile(data)
    } catch (e) {
      // A refresh that fails behind a card that is already drawn is not an
      // error a reader can do anything with: the card is still true, it is
      // just not newer. Only a card with nothing on it reports the failure.
      if (playerIdRef.current === forPlayerId && !hit) {
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
    // Cleared only when there is nothing cached to show. Blanking a hit and
    // setting it again one statement later is a frame of empty card.
    if (!cachedProfile(playerId)) setProfile(null)
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
    // The shared cache is holding a payload whose `header.drafted` this
    // click has just made wrong, and every other reader of it (the hover
    // tip, the board peek) would go on serving it. Drop it so the refetch
    // below is a real request and everybody gets the new answer.
    forgetProfile(forPlayerId)
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

  // Meta line, in the artboard's order: team, age, build, service year, bye.
  //
  // Build sits next to age because they are the same kind of fact -- what he
  // is, rather than where he plays or when he is off -- and because the two
  // of them together are what a reader is checking when they look at a
  // 31-year-old back at all.
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
  const build = fmtBuild(profile?.bio.height ?? null, profile?.bio.weight ?? null)
  if (build !== null) meta.push(build)
  const nflSeason = profile?.bio.nfl_season ?? (ident?.rookie ? 1 : null)
  if (nflSeason !== null) meta.push(`NFL yr ${nflSeason}`)
  const bye = profile?.outlook.bye ?? ident?.bye ?? null
  meta.push(`bye ${bye ?? '—'}`)
  // Where the room has him overall, which is the number this whole card is
  // an argument about -- the header's own figures are per game and per
  // position, and neither says "he is the 181st player on your board".
  //
  // The depth slot was here for a revision and is gone: "RWR2 on depth
  // chart" is his place in a room of four, in a line whose other four facts
  // are about him, and the Room card lists that room in full a few inches
  // down. A designation still carries it, because there the slot is part of
  // the news (see StatusLine).


  return (
    <div className={`player-page${embedded ? ' player-page-embedded' : ''}`}>
      {!embedded && (
        <button type="button" className="player-page-back" onClick={onClose}>
          ← Back to board
        </button>
      )}
      {ident && (
        <div className="pp-pop-head">
          {/* The photo, when the payload has one. Absent for every defense,
              for a player nflverse has no biography for, and for a database
              refreshed before the column existed -- so it is rendered only
              when it is really there rather than behind a placeholder, which
              would put a grey box beside a third of the board. `alt=""`
              because the name is right beside it: a screen reader that reads
              both says his name twice. */}
          {profile?.bio?.headshot && (
            /* The photo and the mark on its corner are one object: the mark
               is positioned against this box, not against the header row. */
            <div className="pp-pop-face-wrap">
              <img
                className="pp-pop-face"
                src={profile.bio.headshot}
                alt=""
                width={44}
                height={44}
                loading="lazy"
              />
              <InjuryMark status={profile?.status ?? null} />
            </div>
          )}
          <div className="pp-pop-ident">
            {/* Name first, chip second: the artboard reads it as a sentence,
                and the position is the qualifier on the name rather than a
                label the eye has to step over to reach it. */}
            <div className="pp-pop-name-row">
              <h2 className="pp-pop-name">{ident.name}</h2>
              <span className={`pos-badge pos-badge-${ident.position.toLowerCase()} pp-pop-pos`}>
                {ident.position}
              </span>
              {/* Only where there is no photo to pin it to. */}
              {!profile?.bio?.headshot && <InjuryMark status={profile?.status ?? null} />}
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
          {header && <PopFigures header={header} profile={profile} />}
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
      {/* Under the status line and above the panels it is about: a reader
          who is told what is empty BEFORE he looks at it reads three cards
          of baseline marks as a fact about this player. Told afterwards, he
          has already read them as a broken popup. Draws nothing at all when
          nothing is missing. */}
      {profile && <MissingData payload={profile} />}
      {profile && <PopPanels profile={profile} settings={settings} />}
      {/* Full width, under the four: the season panels are a career at a
          glance and this is the last year of it in detail. It draws nothing
          for a player with no game log, so a rookie gets no empty frame. */}
      {profile && (
        <WeekByWeek
          games={profile.game_log}
          seasons={profile.seasons}
          position={profile.header.position}
        />
      )}
      {profile && <PopCards profile={profile} onSelectPlayer={onSelectPlayer} />}
      {/* LAST among the content, just above the draft footer: a row of other
          players' faces is the one card on the popup that is about somebody
          else, and every card above it is about him. Reading order follows:
          who he is, what he does, THEN who else to consider -- which is also
          the moment a reader who got this far actually wants the exits. */}
      {profile && (
        <div className="pp-pop-row">
          <SimilarPlayers data={profile.similar_players ?? null}
                          projPpg={profile.summary.proj_ppg}
                          onSelectPlayer={onSelectPlayer} />
        </div>
      )}

      {/* The last thing in the popup, and the whole point of it: the profile
          takes the player it is describing. It opens the confirm dialog and
          stops there -- every pick this room sends goes through the same
          confirmation, whether it started on the board or in here.

          Drawn off `ident`, so it is on screen in the frame the popup opens
          rather than three and a half seconds later when the payload lands.
          A person on the clock who has already decided should not have to
          wait for a dossier he is not going to read.

          `onDraftPlayer` absent means the room has no pick to make for him
          -- he is already gone, he was never on the ranked list, or it is
          not your turn (see DraftRoom, which owns that test). The footer
          then offers the way out that IS available, because a button that
          cannot do what it says is worse than one that admits it. */}
      {onDraftPlayer && ident ? (
        <button
          type="button"
          className="pp-pop-foot is-draft"
          onClick={() => onDraftPlayer(playerId)}
        >
          Draft {draftName(ident.name, ident.position)}
        </button>
      ) : (
        <button type="button" className="pp-pop-foot" onClick={onClose}>
          Back to the board
        </button>
      )}
    </div>
  )
}
