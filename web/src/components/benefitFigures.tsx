import type { ReactNode } from 'react'
import type { LiveCandidate, LiveMock, LiveSettings, MarketSlot, Player } from '../api'
import { TurnBar } from './TurnBar'
import { CellTip, loadProfile } from './draft/CellTip'
import TargetCards from './draft/TargetCards'
import ScheduleRanks from './profile/ScheduleRanks'
import ComparableSeasons from './profile/ComparableSeasons'
import VegasCard from './profile/VegasCard'
// The WIDENED payload, not `PlayerProfileData`: the profile endpoint serves
// `vegas`, the graded schedule and the o-line, and only this type says so.
// The popup reads the same one.
import type { ProfilePayload } from './profile/payload'

// WHAT IT KNOWS, DRAWN ONCE FOR BOTH PLACES THAT SHOW IT.
//
// The landing page makes this argument twice: in the card that opens over the
// live room (one claim at a time, rotating) and in the section below it (all
// six, revealed as they are scrolled to). Same art, same sentences, one
// source, so the two cannot come to promise different things.
//
// SHOW THE PRODUCT, NOT A PICTURE OF IT. Three passes tried to draw these
// claims: miniatures of the product's own widgets, then parametric point
// fields, then drifting colour. All of them were standing in for components
// that already exist and are better than any drawing of them -- and most of
// them live inside the player profile or the archive, which a visitor to
// this page never opens.
//
// So every slide renders the real component, on whoever the live room ranks
// first at that moment. Nothing here is a mock-up and nothing is scaled.

/** Comparable seasons to show on the landing page. The popup pages at
 *  twelve; a dialog over a live draft has room for five. */
const LANDING_COMPS = 5

/** How many recorded seasons make a player worth putting on the health
 *  panel. The chart is about a career, and a career needs enough columns to
 *  have a shape: three left it thin, and the board's top picks are routinely
 *  first- and second-year players. */
export const VETERAN_SEASONS = 5

/** How far down the board to look for one. Past this it is not worth more
 *  round trips on a landing page: the slide shows nothing rather than
 *  spending a reader's bandwidth hunting. */
const VETERAN_DEPTH = 6

/** The first player on the board with a career behind him. Sequential and
 *  short-circuiting: usually one extra request, often none, because the
 *  room's top pick is frequently a veteran himself. */
export async function findVeteran(
  shortlist: { player_id: string }[],
): Promise<ProfilePayload | null> {
  for (const row of shortlist.slice(0, VETERAN_DEPTH)) {
    try {
      const body = await loadProfile(row.player_id) as unknown as ProfilePayload
      if (body.seasons.length >= VETERAN_SEASONS) return body
    } catch {
      // A player whose profile will not load is not a reason to stop
      // looking; the next one down usually loads.
    }
  }
  return null
}

/** Everything the six slides draw from. All of it optional: a slide with no
 *  data renders nothing rather than an empty frame. */
export type ShowcaseData = {
  /** The archive's own size, for the caption over its turn. */
  drafts: number | null
  /** One seat of the archive: who the room takes at each of its picks. */
  slot: MarketSlot | null
  /** The live room behind the card, for its own top recommendation. */
  room: LiveMock | null
  /** That player's profile, which is where four of these panels live. */
  profile: ProfilePayload | null
  /** A player with a CAREER behind him, for the health panel.
   *
   *  The room's top pick is regularly a rookie -- it was Tyler Warren when
   *  this was written -- and a games-played chart with one column in it is a
   *  fair reading of that player and a terrible advertisement for the panel.
   *  So the health slide takes the first candidate on the board who has
   *  several seasons on record; everything else stays on the room's own top
   *  pick, which is the honest live example. */
  veteran: ProfilePayload | null
}

/** THE REAL COMPONENT, AND NOTHING BEHIND IT.
 *
 *  Every one of the six claims is about something the product actually
 *  draws, and most of them live inside the player profile or the archive --
 *  surfaces a visitor to this page never opens. They can watch the room
 *  behind this card all afternoon and never see the Vegas line, the graded
 *  schedule, or who a player is comparable to.
 *
 *  So each slide shows the component itself, unmodified and at its own size,
 *  on whoever the live room ranks first at this moment. Not a scaled copy
 *  and not a redrawing: a card re-tuned for this slot would be a second
 *  version of it to keep in step, and the reader would be shown something
 *  they will not get.
 *
 *  There was a drifting field of coloured orbs behind these for a while,
 *  from when the slides had nothing real to show. A blurred blob wandering
 *  under a table of numbers is noise over signal; with the real panels here
 *  it had no job left.
 */
function Stage({ children }: { children?: ReactNode }) {
  // `null` and `false` both mean "this slide has no panel yet". Rendering the
  // box anyway left a tall empty well above the copy while a fetch was in
  // flight, or forever if that player has no such card.
  if (children === null || children === undefined || children === false) return null
  return (
    // BOTH WRAPPERS, ALWAYS. Every override that fits these panels into a
    // 468px dialog is scoped to `.bf-real` -- the single-column grid, the
    // hidden section header, the compacted card, the tone colours, the
    // hover panel's position. A rewrite of this function dropped the inner
    // div and took all of them with it silently: the recommendation card
    // went on rendering, but into the room's three-column grid at a third
    // of the width, wrapping into a column half the dialog tall.
    <div className="bf-stage">
      <div className="bf-real">{children}</div>
    </div>
  )
}

/** One recommendation card, exactly as the room draws it: the same component,
 *  the same figures, the same meters. Fed the room's own top candidate and
 *  the one player row it needs to name him. */
function Recommendation({ room, profile }: {
  room: LiveMock | null
  profile: ProfilePayload | null
}) {
  const row = room?.shortlist?.[0]
  if (!row || !profile) return null
  const candidate: LiveCandidate = {
    player_id: row.player_id,
    position: row.position ?? profile.header.position,
    proj_points: row.proj_points ?? 0,
    espn_rank: row.espn_rank ?? null,
    espn_pos_rank: row.espn_pos_rank ?? null,
    espn_adp: row.espn_adp ?? null,
    // See DemoRoom's own copy of this mapping: `adp` is the market's
    // answer, so it lands in the consensus field and nowhere else.
    market_rank: row.market_rank ?? row.adp ?? null,
    lasts_pct: row.lasts_pct ?? null,
    lasts_at_pick: row.lasts_at_pick ?? null,
    edge_at_pick: row.edge_at_pick ?? null,
    edge_pts: row.edge_pts ?? null,
    need: row.need ?? null,
    favourite: false,
    rank: row.rank ?? 1,
  }
  const pick = (room?.picks_made ?? 0) + 1
  return (
    <TargetCards
      plan={[]}
      candidates={[candidate]}
      // One entry is all it reads: `players[id]`, for the name, the face,
      // the meters and the market ranks. `ProfileHeader` IS a board row
      // plus the profile's own extras, so it satisfies that lookup exactly.
      players={{ [row.player_id]: profile.header as unknown as Player }}
      onDraft={() => { /* nobody drafts from a landing page */ }}
      isMyTurn={false}
      pickNo={pick}
      settings={(room?.settings ?? null) as LiveSettings | null}
      recompute={null}
      onOpenPlayer={() => { /* the profile is a click away in the room */ }}
    />
  )
}

/** One turn of the archive: who the room takes at this pick, across every
 *  recorded draft. The bar is the archive page's own. */
function ArchiveTurn({ slot, drafts }: {
  slot: MarketSlot | null
  drafts: number | null
}) {
  // A middle-round turn: round one is the same three names in every draft
  // and says nothing about a market, and the late rounds are noise.
  const turn = slot?.turns?.find((t) => t.round === 4 && t.observed > 20)
    ?? slot?.turns?.find((t) => t.observed > 20)
  if (!turn) return null
  return (
    <div className="bf-turn">
      <p className="draft-cap">
        {/* The live count, from the archive's own endpoint, beside the one
            turn being drawn. It moves: the farm is in a mock draft while
            this is on screen. */}
        Pick {turn.pick_no} · {turn.distinct} different players
        {drafts === null ? '' : ` · across ${drafts.toLocaleString()} drafts`}
      </p>
      <TurnBar turn={turn} />
      <ol className="bf-turn-names">
        {turn.players.slice(0, 3).map((p) => (
          <li key={p.player_id}>
            <span className="bf-turn-name">{p.name ?? p.player_id}</span>
            <span className="bf-turn-share mono">{Math.round(p.share * 100)}%</span>
          </li>
        ))}
      </ol>
    </div>
  )
}

export type Benefit = {
  key: string
  eyebrow: string
  title: string
  body: string
  figure: ReactNode
}

/** The six, in the order they are argued: the thing nobody else has, what is
 *  done with it, then the depth underneath. `live` is the archive's own count
 *  once the page has it.
 *
 *  ONE SENTENCE, TWO AT MOST. These are read in four seconds, over a draft
 *  still running behind them. Say the thing; the room is the proof. */
export function benefits(data: ShowcaseData): Benefit[] {
  const { slot, room, profile, veteran } = data
  const player = profile?.header.name ?? null
  const on = (what: string) => (player ? `${what}, on ${player} right now` : what)
  // The archive's own size, said in the copy rather than left to a caption:
  // it is the one number on this page that grows while somebody reads it.
  const drafts = data.drafts === null ? 'Hundreds of' : data.drafts.toLocaleString()
  return [
    // THE ORDER IS THE ARGUMENT. What the tool does that nothing else does
    // comes first; the depth underneath it follows; the thing it is all
    // counted from -- the archive -- lands last, as the answer to "how do you
    // know any of this".
    {
      key: 'survival',
      eyebrow: 'The wait, counted',
      title: 'Who will still be there',
      // NOT "simulated" any more, and the word matters: this figure is
      // counted from the recorded drafts the slide two below is about, not
      // played out by a model. The old copy sold an opponent model the room
      // no longer runs.
      body: `Out of ${drafts.toLowerCase()} recorded ESPN drafts, how often `
        + 'a player was still on the board when your next turn came round, '
        + 'and what taking him now is worth in points against waiting.',
      figure: (
        <Stage>
          {room?.shortlist?.[0] && profile
            ? <Recommendation room={room} profile={profile} />
            : null}
        </Stage>
      ),
    },
    {
      key: 'peers',
      eyebrow: 'Comparable players',
      title: 'Who he looks like',
      body: on('Every comparable season on record, back through the whole '
        + 'era rather than last year alone, scored for closeness'),
      figure: (
        <Stage>
          {profile && (
            <ComparableSeasons
              mode={profile.similar.mode}
              players={profile.similar.players
                .filter((p) => p.season !== null && p.ppg !== null)
                .slice(0, LANDING_COMPS)}
              targetAge={profile.similar.target_age ?? null}
              cohort={profile.cohort}
              onSelectPlayer={() => { /* not a popup */ }}
            />
          )}
        </Stage>
      ),
    },
    {
      key: 'vegas',
      eyebrow: 'Vegas',
      title: 'What his offence is priced at',
      body: on('Implied team totals and spreads, week by week'),
      figure: (
        <Stage>
          {profile && (
            <VegasCard vegas={profile.vegas} position={profile.header.position} />
          )}
        </Stage>
      ),
    },
    {
      key: 'meters',
      eyebrow: 'Health & reliable',
      title: 'Two different ways to lose',
      body: veteran
        ? `Every season he has played, not just the last one: ${
            veteran.seasons.length} of them for ${veteran.header.name}, `
          + 'counting the games he missed'
        : 'Every season he has played, not just the last one, counting the '
          + 'games he missed',
      figure: (
        <Stage>
          {veteran && (
            <div className="avail-cell-tip bf-tip">
              <CellTip kind="health" playerId={veteran.header.player_id} />
            </div>
          )}
        </Stage>
      ),
    },
    {
      key: 'schedule',
      eyebrow: 'Strength of schedule',
      title: 'When his season gets hard',
      body: on('Every week graded against the defence he faces, including '
        + 'the ones your league is decided in'),
      figure: (
        <Stage>
          {profile && (
            <ScheduleRanks weeks={profile.schedule} sosPct={profile.outlook.sos_pct} />
          )}
        </Stage>
      ),
    },
    {
      key: 'archive',
      eyebrow: 'Real drafts, recorded',
      title: 'Where players actually go',
      body: `${drafts} real drafts, every pick kept, and more of them while `
        + 'you read this. A pick is a spread of names, not one average.',
      figure: (
        <Stage>{slot ? <ArchiveTurn slot={slot} drafts={data.drafts} /> : null}</Stage>
      ),
    },
  ]
}
