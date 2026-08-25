import {
  useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode,
} from 'react'
import type {
  LiveCandidate, LiveSettings, Player, PlayerProfileData,
} from '../../api'
import { riskTone } from './tone'
import {
  ChangeMeter, FinishArc, HealthMeter, SteadyMeter, TIP_DELAY_MS, healthLevel,
  positionTip, steadyLevel,
} from './AvailableList'
import { startersAt } from './finish'
import { CellTip, loadProfile, type CellTipKind } from './CellTip'
import { Chart } from './Chart'
import { weekCols } from './panels'

// duplicated from AvailableList.tsx (which duplicated it from
// RosterPanel.tsx/DraftBoardGrid.tsx in turn) -- see either for why a
// four-line pure function is copied rather than shared.
/** The position's own hue, the token the badges and the board already use.
 *  Unknown positions fall to the muted text colour rather than to a colour
 *  that would claim a position they do not have. */
/** How many rollouts the survival number is counted from -- api/live.py's
 *  `SURVIVAL_ROLLOUTS` and api/demo.py's, which are the same number for the
 *  same reason. Named here only so the tooltip can say what it is counting;
 *  if the server ever runs a different count this reads high or low and the
 *  sentence is the thing to fix. */
const ROLLOUTS = 400

function posHue(position: string): string {
  const key = position.toLowerCase()
  return ['qb', 'rb', 'wr', 'te', 'k', 'dst'].includes(key)
    ? `var(--pos-${key})` : 'var(--text-3)'
}

function posBadge(position: string): ReactNode {
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

// See AvailableList.tsx's own copy of these two for the full rationale --
// duplicated rather than shared (both are three-line pure functions, the
// same call this codebase already makes for posBadge).
function fmtSigned(n: number): string {
  const r = Math.round(n)
  return r > 0 ? `+${r}` : `${r}`
}


// The one sentence each card owes the user, built strictly from this row's
// own four numbers (fills, gain_now, vor_points, survive_pct) -- never a
// template with a fixed "take him now" conclusion. That distinction is the
// whole point of this task: the old UI ranked on raw value-over-replacement
// and reached for whoever had the biggest one (Josh Allen's 91-point vor
// over the next QB's 82 -- see scoring/gain.py's module docstring). A
// player can still show up in this top three with a big vor_points and a
// tiny gain_now, because the backup at his position is nearly as good --
// and the sentence has to say exactly that instead of manufacturing
// urgency that isn't there.
//
// `gain_now = weight * (vor_points - next_best[pos])` (scoring/gain.py) can
// itself be negative or zero -- not just small -- whenever this player's
// own vor_points sits below the survivor-weighted expectation for his
// position (two QBs at vor 90/70 with survival .85/.90 give the weaker one
// roughly gain_now = -16). That is reachable at any weight, not only a
// capped one: NEED_WEIGHTS["starter"] = 1.0 (scoring/config.py) applies no
// dampening at all for a starter need. A second reviewed-in bug: neither
// the "nearly as good" ratio branch nor the plain take-now default checked
// the SIGN of `gain_now` before this fix -- `gain_now / vor_points < 0.15`
// is trivially true for any negative `gain_now` (a QB at vor=70, gain=-16
// used to read as "nearly as good," when the honest reading is "worse"),
// and the take-now default asserted "this is the one to take now" for a
// literally negative gain with no check at all. Both were sentences
// asserting the opposite of their own numbers -- the exact failure this
// task exists to eliminate.
//
// Fixed by handling `gain_now <= 0` as its own branch, checked before
// either of those: it is real, useful information for a top-three slot to
// carry ("none of these is worth taking for this position yet" is a
// legitimate thing to tell the user, not a failure to recommend someone).
// Every branch below it can now assume `gain_now > 0` reached it, so
// neither "nearly as good" nor "take him now" can fire for a player the
// model rates as a net-negative pick right now.
//
// The "next-best option is nearly as good" branch further requires
// `vor_points > 0`, since it reads `gain_now` as a *fraction of his own
// vor_points* -- a fraction with a non-positive denominator has no
// coherent reading. A non-positive `vor_points` (with `gain_now > 0`,
// having already passed the branch above -- e.g. vor_points=-5,
// gain_now=+25 for a thin, weak position whose replacement level is even
// worse) instead falls through to the `survive_pct`/default branches,
// neither of which mentions `vor_points` at all.
// A candidate row once there is a roster to rank it against: gain_now,
// survive_pct and fills arrive together as real values or together as null
// (see LiveCandidate's own comment in api.ts -- api/live.py serves one
// payload or the other for the whole list, never a mix per row). Narrowing
// once, via isRanked below, is what lets this file's own
// fmtSigned/fillsIsOpenSlot keep treating these three as plain values
// instead of every line growing a null check that can never actually fire
// once the guard below has run.
/** THE TWO NUMBERS THE PICK TURNS ON, at the size of the decision they
 *  carry: will he be there next time, and what it costs if he is not.
 *
 *  Everything else on this card is evidence for or against those two. They
 *  were a dial and a caption; a dial is a nice picture of one number and a
 *  reader comparing three cards is comparing figures, so the figures are the
 *  figures now. The thin rule under the percentage keeps the proportion
 *  visible without spending a 62px circle on it. */
function Heroes({ candidate, at, isMyTurn, player, settings, onArcEnter, onArcLeave }: {
  candidate: RankedCandidate; at: string | null; isMyTurn: boolean
  player: Player | undefined; settings: LiveSettings | null
  onArcEnter: (el: HTMLElement) => void; onArcLeave: () => void
}) {
  const pct = Math.round(candidate.survive_pct)
  // ON THE CLOCK, BOTH FIGURES ARE DEGENERATE. Survival to "your next pick"
  // is 100% for everyone the moment that pick is yours-after-this-one with
  // nobody between (every wheel turn), and `gain_next` counts the player
  // himself in "the best expected", so a sure survivor reads 0 pts however
  // far ahead of his position he is. So while it is your pick the card
  // answers the two questions you actually hold: how good has he been
  // (the table's own RANK arc, season by season) and what taking him nets
  // over the best OTHER player at his position expected at your next turn
  // (`edge_next`, which excludes him -- see scoring/gain.rank_available).
  if (isMyTurn) {
    const arc = player?.season_finishes ?? null
    const edge = candidate.edge_next
    return (
      <div className="top3-heroes">
        <div className="top3-hero">
          {arc !== null && arc.length > 0 ? (
            /* The table's RANK cell, hover panel included: the arc opens the
               same finish panel a hover on that column opens, because five
               bars cannot say which seasons they are. */
            <span
              className="top3-hero-arc top3-hoverable"
              tabIndex={0}
              onMouseEnter={(e) => onArcEnter(e.currentTarget)}
              onMouseLeave={onArcLeave}
              onFocus={(e) => onArcEnter(e.currentTarget)}
              onBlur={onArcLeave}
            >
              <FinishArc arc={arc} position={candidate.position}
                         starters={startersAt(candidate.position, settings)} />
            </span>
          ) : (
            <span className="top3-hero-arc top3-hero-arc-none mono">—</span>
          )}
          <span className="top3-hero-cap">position rank by season</span>
        </div>
        <div className="top3-hero">
          <span className={`top3-hero-num mono ${edge == null || edge >= 0 ? 'is-up' : 'is-down'}`}>
            {edge == null ? '—' : fmtSigned(edge)}
            <span className="top3-hero-unit">pts</span>
          </span>
          <span className="top3-hero-cap">
            over the next-best {candidate.position} expected at{' '}
            {at === null ? 'your next pick' : at.replace('pick ', '')}
          </span>
        </div>
      </div>
    )
  }
  const tone = riskTone(candidate.survive_pct)
  const gap = candidate.gain_next
  const where = at === null ? 'your next pick' : at.replace('pick ', '')
  return (
    <div className="top3-heroes">
      <div className="top3-hero">
        <span className="top3-hero-num mono" style={{ color: tone }}>
          {pct}<span className="top3-hero-unit">%</span>
        </span>
        <Tip text={`Out of ${ROLLOUTS} simulated drafts from this board, he was `
          + `still on it at ${where} in ${pct}% of them.`}>
          <span className="top3-hero-rule" aria-hidden="true">
            <span style={{ width: `${pct}%`, background: tone }} />
          </span>
        </Tip>
        <span className="top3-hero-cap">still there at {where}</span>
      </div>
      <div className="top3-hero">
        <span className={`top3-hero-num mono ${gap == null || gap >= 0 ? 'is-up' : 'is-down'}`}>
          {gap == null ? '—' : fmtSigned(gap)}
          <span className="top3-hero-unit">pts</span>
        </span>
        <span className="top3-hero-cap">
          over the best {candidate.position} expected at {where}
        </span>
      </div>
    </div>
  )
}

/** HIS SEASON, WEEK BY WEEK -- the same picture the table's hover panel
 *  draws, from the same fetch and through the same `Chart`.
 *
 *  This band has been four things now: a finish arc, a row of numbers, a
 *  plotted career, and a line of typed finishes. All of them answered "how
 *  has he ranked", which is a summary of a summary. What a drafter actually
 *  argues about is the shape of the year -- eight quiet weeks and four huge
 *  ones is a different player from twelve steady ones at the same average,
 *  and the row of bars says that in a glance.
 *
 *  The panel's own component, not a copy of it: one picture to learn, and a
 *  fix to the chart cannot land on the hover and miss the card. The profile
 *  is fetched per player and cached by `loadProfile`, which is the same
 *  request a hover over the meters below would make -- so on a card whose
 *  panels have been opened it is already in hand.
 */
function SeasonWeeks({ playerId, position }: { playerId: string; position: string }) {
  const [data, setData] = useState<PlayerProfileData | null>(null)
  const [failed, setFailed] = useState(false)
  const wanted = useRef(playerId)

  useEffect(() => {
    wanted.current = playerId
    setData(null)
    setFailed(false)
    loadProfile(playerId)
      .then((d) => { if (wanted.current === playerId) setData(d) })
      .catch(() => { if (wanted.current === playerId) setFailed(true) })
  }, [playerId])

  // A fixed-height placeholder either way: this band is 90px of the card, and
  // three cards resizing as their fetches land would move the table under
  // them twice a pick.
  if (failed) return <div className="top3-season is-empty">No game log.</div>
  if (!data) return <div className="top3-season is-empty" aria-hidden="true" />

  const seasons = data.game_log.map((g) => g.season)
  if (!seasons.length) {
    return (
      <div className="top3-season is-empty">
        No NFL games yet. The projection above is his first.
      </div>
    )
  }
  const latest = Math.max(...seasons)
  const played = data.game_log.filter((g) => g.season === latest && !g.dnp)
  const avg = played.length
    ? played.reduce((sum, g) => sum + g.ppr_points, 0) / played.length
    : 0
  return (
    <div className="top3-season">
      <span className="draft-cap">
        {latest} by week
        <span className="top3-season-avg">
          {played.length} games · avg {avg.toFixed(1)}
        </span>
      </span>
      <Chart cols={weekCols(data.game_log, latest, position)} />
    </div>
  )
}

/** A meter under its own caption, with the table's own panel behind it.
 *
 *  FIVE BARS ARE UNFALSIFIABLE ON THEIR OWN. In the table these columns open
 *  a hover panel that gives the measurement -- which is why `HealthMeter`
 *  and the rest carry no `title`, since the native box rendered on top of
 *  that panel. On a card there was no panel at all, so the bars were four
 *  greens with no way to ask what they meant.
 *
 *  The three of them read as one row of ratings, which is the only way
 *  five-bar meters mean anything: same direction, same ramp, same size. */
function Meter({ caption, children, onEnter, onLeave }: {
  caption: string
  children: ReactNode
  /** Opens the table's own panel for this meter. Null for a meter with
   *  nothing measured behind it, so a dash never grows an affordance it
   *  cannot honour. */
  onEnter: ((el: HTMLElement) => void) | null
  onLeave: () => void
}) {
  return (
    <span className="top3-meter">
      <span className="draft-cap">{caption}</span>
      <span
        className={onEnter ? 'top3-hoverable' : undefined}
        tabIndex={onEnter ? 0 : undefined}
        onMouseEnter={onEnter ? (e) => onEnter(e.currentTarget) : undefined}
        onMouseLeave={onEnter ? onLeave : undefined}
        onFocus={onEnter ? (e) => onEnter(e.currentTarget) : undefined}
        onBlur={onEnter ? onLeave : undefined}
      >
        {children}
      </span>
    </span>
  )
}

/** One market's answer: where it ranks him, and how far past that he is. */
function Market({ caption, rank, pickNo }: {
  caption: string; rank: number | null; pickNo: number | null
}) {
  return (
    <span className="top3-meter">
      <span className="draft-cap">{caption}</span>
      <span className="top3-market-val mono">
        {rank ?? '—'}
        <MarketMove pickNo={pickNo} rank={rank} />
      </span>
    </span>
  )
}

/** HOW FAR PAST HIS PRICE HE IS, the same figure the pick ticker draws under
 *  the room and the snake board draws in its cells: the pick number minus his
 *  market rank. Positive means he has already outlasted the market's answer
 *  and is a bargain sitting here; negative means taking him now is that many
 *  picks early.
 *
 *  Arrow then size, matching the ticker exactly -- which way is the message
 *  and how far is the detail -- and the same two tone classes, so a green
 *  triangle means the same thing everywhere in the room. Nothing at all when
 *  there is no rank to compare against, or when the pick landed on it: a
 *  bare zero is a fact about arithmetic rather than about the draft.
 */
function MarketMove({ pickNo, rank }: { pickNo: number | null; rank: number | null }) {
  if (pickNo === null || rank === null) return null
  const slots = Math.round(pickNo - rank)
  if (slots === 0) return null
  const steal = slots > 0
  return (
    <span className={`top3-market-move ${steal ? 'is-steal' : 'is-reach'}`}
          title={steal
            ? `Still here ${Math.abs(slots)} picks past his ADP`
            : `Taking him now is ${Math.abs(slots)} picks before his ADP`}>
      <span aria-hidden="true">{steal ? '\u25b2' : '\u25bc'}</span>
      {Math.abs(slots)}
    </span>
  )
}

/** Anything with a number behind it, and the number. Renders its child bare
 *  when there is nothing to say, so a missing measurement never grows a
 *  hover affordance it cannot honour. */
function Tip({ text, children }: { text: string | null; children: ReactNode }) {
  if (text === null) return <>{children}</>
  return (
    <span className="top3-tip" tabIndex={0}>
      {children}
      <span className="top3-tip-box" role="tooltip">{text}</span>
    </span>
  )
}

type RankedCandidate = LiveCandidate & {
  gain_now: number; survive_pct: number; fills: string
}

// `!= null`, which catches undefined as well as null, and `gain_next` is
// deliberately NOT part of the test. A server that predates that field
// serves rows without it, and `Math.round(undefined)` prints "NaN" -- which
// is exactly what appeared on the cards. A missing figure now renders as a
// dash and the other five stand; a missing RANKING still renders no cards at
// all, which is what this guard is for.
function isRanked(c: LiveCandidate): c is RankedCandidate {
  return c.gain_now != null && c.survive_pct != null && c.fills != null
}

// The list on screen was ranked for an older pick than the one on the clock
// -- `forPick` is the pick being recomputed for (the one on the clock),
// `listedForPick` is the pick the cards and table below were actually
// ranked for. Both spelled by DraftRoom off `candidates_as_of_pick`; null
// whenever the two agree, which is the overwhelming majority of polls.
export type RecomputeState = { forPick: number; listedForPick: number }

interface TopThreeProps {
  candidates: LiveCandidate[]
  players: Record<string, Player>
  onDraft: (c: LiveCandidate) => void
  // See AvailableList.tsx's own field comment -- not in the task brief's
  // literal signature, added for the same reason: the draft button has to
  // gate on whose turn it is, and this component has no other way to know.
  isMyTurn: boolean
  // The pick `gain_now` and `survive_pct` are actually measured against,
  // already phrased ("pick 18", "the end of the draft") by DraftRoom off
  // The server's horizon -- the turn the RANKING is priced against, which is
  // not a pick the reader owns (see scoring/draft_sim.horizon_picks). Kept in
  // the props because DraftRoom and the landing room both pass it and
  // AvailableList still reads it; nothing on these cards names it any more,
  // since a figure whose caption has to explain what it is measured against
  // is a figure nobody reads.
  horizonLabel?: string | null
  // The reader's own next turn ("pick 49"), which is what the LASTS figure
  // is measured to. Null falls back to the words "your next pick", which is
  // the same claim without the number.
  nextPickLabel?: string | null
  /** The pick on the clock, so the market figures can say how far past his
   *  ADP he has already fallen. Null leaves them bare numbers. */
  pickNo?: number | null
  /** The league, for `startersAt` -- the finish arc bands a season against
   *  how many players at that position a league of this size starts, so
   *  "WR12" is a strong year in one league and a weak one in another. */
  settings?: LiveSettings | null
  // Non-null only while a recompute is outstanding -- see RecomputeState.
  // This used to be a `.draft-notice-banner` paragraph in DraftRoom, mounted
  // and unmounted between polls: it added a strip of vertical height above
  // `.draft-body` and took it away again a poll or two later, so the cards
  // and the whole table under them jumped up and down mid-pick. The Draft
  // button moving out from under a cursor on a 30-second clock, for a pick
  // that cannot be undone, is a misdraft risk -- so the same information
  // now lives on this header row, which is painted whether or not a
  // recompute is running.
  recompute: RecomputeState | null
  // Opens the profile over the room -- same handler and same reasoning as
  // AvailableList's own field comment. The name only, never the card: the
  // card's own control is Draft, and that one cannot be undone.
  onOpenPlayer: (c: LiveCandidate) => void
}

// Three cards above the ranked table, the same `candidates` prop
// unfiltered -- AvailableList's own search/position filter is local to
// that component and never touches what shows up here. This always names
// the three best picks on the board by gain_now, regardless of what the
// user happens to be searching for below.
export default function TopThree({
  candidates, players, onDraft, isMyTurn, nextPickLabel = null,
  pickNo = null, settings = null, recompute, onOpenPlayer,
}: TopThreeProps) {
  // THE TABLE'S OWN HOVER PANELS, on the cards. `CellTip` is the same
  // component the available list opens under its Health, Steady, Growth
  // and Rank columns -- a fetched game log, a season-by-season chart -- and
  // the meters up here were drawing the identical bars with no way to ask
  // what was behind them.
  //
  // The machinery is the table's too, down to the 130ms delay and
  // `positionTip`: a panel that opened instantly would flash three times as
  // the pointer crossed a row of meters, and one placed by this file would
  // sooner or later disagree with the other about which edge of the window
  // to avoid.
  const [panel, setPanel] =
    useState<{ kind: CellTipKind; playerId: string; rect: DOMRect } | null>(null)
  const [panelPos, setPanelPos] = useState<{ left: number; top: number } | null>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const panelTimer = useRef<number | null>(null)

  const hidePanel = useCallback(() => {
    if (panelTimer.current !== null) window.clearTimeout(panelTimer.current)
    panelTimer.current = null
    setPanel(null)
  }, [])

  const showPanel = useCallback(
    (kind: CellTipKind, playerId: string, el: HTMLElement) => {
      if (panelTimer.current !== null) window.clearTimeout(panelTimer.current)
      const rect = el.getBoundingClientRect()
      // The fetch starts on the first hover, before the panel is due, so the
      // request and the delay overlap rather than queue.
      void loadProfile(playerId).catch(() => {})
      panelTimer.current = window.setTimeout(
        () => setPanel({ kind, playerId, rect }), TIP_DELAY_MS)
    }, [])

  // Measured after it renders and before the paint, because the panel
  // RESIZES when its fetch lands -- a loading skeleton is one line and a game
  // log is eighteen rows, and a position measured against the small one
  // leaves the full panel hanging off the bottom of the window.
  useLayoutEffect(() => {
    if (!panel || !panelRef.current) {
      setPanelPos(null)
      return
    }
    const { width, height } = panelRef.current.getBoundingClientRect()
    setPanelPos(positionTip(panel.rect, { width, height }))
  }, [panel])

  useEffect(() => {
    if (!panel) return
    window.addEventListener('scroll', hidePanel, true)
    return () => window.removeEventListener('scroll', hidePanel, true)
  }, [panel, hidePanel])

  const top3 = candidates.slice(0, 3)
  // No cards means no header row to hang the recompute indicator on either.
  // Deliberately still `null` rather than growing a header just for the
  // indicator: a header that appears only while recomputing would be the
  // exact layout shift this indicator replaced. Unreachable in practice --
  // `recompute` is derived from `candidates_as_of_pick`, which is only ever
  // set by a recompute that produced a list.
  if (top3.length === 0) return null

  // Defect 2: gain_now is all-or-nothing across the whole list (see
  // RankedCandidate's own comment above) -- api/live.py serves either a
  // slot-ranked payload or the vor-only fallback for my_slot === null,
  // never a mix. Checking the lead row is therefore enough to know which
  // this is. Three cards of dashes plus a "take one of these" caption would
  // imply a real recommendation exists when none has been computed yet --
  // so this renders nothing card-shaped at all, just the one quiet line
  // explaining why, in the same slot the cards would otherwise occupy.
  if (!isRanked(top3[0])) {
    return (
      <div className="top3">
        <p className="top3-hint top3-hint-standalone">
          Ranked by value over replacement until your draft slot is known --
          personalized recommendations start once it resolves.
        </p>
      </div>
    )
  }
  const ranked = top3.filter(isRanked)

  return (
    <div className="top3">
      <div className="top3-head">
        <span className="draft-cap top3-cap">Take one of these</span>
        {/* Plain words for what the order means. It used to read "ranked by
            what you gain now vs. waiting", which is the phrase the figures
            below stopped using for the reason the owner gave: nobody should
            have to be told what a gain is measured against to read a card. */}
        <span className="top3-hint">
          most to lose by waiting{nextPickLabel !== null ? `, priced at your ${nextPickLabel}` : ''}
        </span>
        {/* Always mounted, even with nothing to say -- two reasons, both
            load-bearing. (1) Layout: an element that comes and goes could
            still change this row, and this row sits directly above the
            Draft buttons. Empty, it is a zero-width flex item pushed to the
            far right by `margin-left: auto`, so neither its arrival nor its
            departure moves a pixel of anything. (2) Assistive tech: a
            `role="status"` region announces text that CHANGES inside an
            already-live region -- mounting the region and its text in the
            same commit is not reliably announced, which is what mounting
            the old banner did. The text still names the pick the list below
            was actually ranked for, because that honesty is the entire
            reason this indicator exists. */}
        <span className="top3-recompute" role="status">
          {recompute !== null && (
            <>
              <span className="top3-recompute-dot" aria-hidden="true" />
              Recomputing for pick {recompute.forPick}, and the list below is
              {' '}still for pick {recompute.listedForPick}
            </>
          )}
        </span>
      </div>
      <div className="top3-grid">
        {ranked.map((c, i) => {
          const player = players[c.player_id]
          return (
            <div key={c.player_id} className={`top3-card${i === 0 ? ' top3-card-lead' : ''}`}>
              <div className="top3-card-head">
                <span className="top3-rank mono">#{i + 1}</span>
                {/* The face, bigger than a badge because it is the fastest
                    way anybody tells three cards apart under a clock. The
                    ring is his position's own hue, which is why the badge
                    only appears for the players who have no photograph. */}
                {player?.headshot ? (
                  <img className="top3-shot" src={player.headshot} alt=""
                       loading="lazy"
                       style={{ borderColor: posHue(c.position) }} />
                ) : posBadge(c.position)}
                <span className="top3-id">
                  <button
                    type="button"
                    className="top3-name-btn"
                    onClick={() => onOpenPlayer(c)}
                    title="Open profile"
                  >
                    <span className="top3-name">{player?.name ?? c.player_id}</span>
                  </button>
                  <span className="top3-sub mono">
                    {c.position}
                    {player?.team ? ` · ${player.team}` : ''}
                    {player?.bye ? ` · BYE ${player.bye}` : ''}
                  </span>
                </span>
                <button
                  type="button"
                  className="avail-draft-btn"
                  disabled={!isMyTurn}
                  title={isMyTurn ? undefined : 'Not your turn yet'}
                  onClick={() => onDraft(c)}
                >
                  Draft
                </button>
              </div>

              <Heroes candidate={c} at={nextPickLabel} isMyTurn={isMyTurn}
                      player={player} settings={settings}
                      onArcEnter={(el) => showPanel('finish', c.player_id, el)}
                      onArcLeave={hidePanel} />

              {/* NO PANEL ON THIS ONE. The meters below open one because
                  five bars cannot say what they measure; this IS the panel's
                  own picture, so a hover would redraw what is already on
                  screen, over the top of it. */}
              <SeasonWeeks playerId={c.player_id} position={c.position} />

              <div className="top3-meters">
                <Meter caption="Health"
                       onEnter={player?.career_games_pg == null ? null
                         : (el) => showPanel('health', c.player_id, el)}
                       onLeave={hidePanel}>
                  {healthLevel(player?.career_games_pg) !== null
                    ? <HealthMeter level={healthLevel(player?.career_games_pg) as number}
                                   gamesPg={player?.career_games_pg as number} />
                    : <span className="top3-meter-none">—</span>}
                </Meter>
                <Meter caption="Steady"
                       onEnter={player?.consistency_pct == null ? null
                         : (el) => showPanel('steady', c.player_id, el)}
                       onLeave={hidePanel}>
                  {steadyLevel(player?.consistency_pct) !== null
                    ? <SteadyMeter level={steadyLevel(player?.consistency_pct) as number}
                                   cv={player?.consistency_cv ?? null} />
                    : <span className="top3-meter-none">—</span>}
                </Meter>
                <Meter caption="Growth"
                       onEnter={player?.proj_change == null ? null
                         : (el) => showPanel('change', c.player_id, el)}
                       onLeave={hidePanel}>
                  {player?.proj_change != null
                    ? <ChangeMeter change={player.proj_change} />
                    : <span className="top3-meter-none">—</span>}
                </Meter>
                {/* THE SAME GRAMMAR AS THE METERS BESIDE THEM: a caption
                    over a value. These were one inline run -- "ADP 83 ▲12 ·
                    ESPN 80 ▲15" -- which put five things at one weight and
                    made the row's right half read as a sentence next to
                    three charts. As cells they line up with Health,
                    Steady and Growth, and the eye reads one row of
                    labelled facts instead of two kinds of thing. */}
                <span className="top3-market">
                  <Market caption="ADP" rank={player?.market_rank ?? null}
                          pickNo={pickNo} />
                  <Market caption="ESPN" rank={player?.espn_ppr_rank ?? null}
                          pickNo={pickNo} />
                </span>
              </div>
            </div>
          )
        })}
      </div>
      {panel && (
        <div
          ref={panelRef}
          role="tooltip"
          className="avail-cell-tip"
          style={panelPos
            ? { left: panelPos.left, top: panelPos.top, visibility: 'visible' }
            : { left: 0, top: 0, visibility: 'hidden' }}
        >
          <CellTip kind={panel.kind} playerId={panel.playerId} settings={settings} />
        </div>
      )}
    </div>
  )
}
