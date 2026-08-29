import {
  useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode,
} from 'react'
import type {
  LiveCandidate, LivePlanTurn, LiveSettings, PlanPlayer, Player,
} from '../../api'
import { riskTone } from './tone'
import {
  ChangeMeter, HealthMeter, SteadyMeter, TIP_DELAY_MS, healthLevel,
  positionTip, steadyLevel,
} from './AvailableList'
import { CellTip, loadProfile, type CellTipKind } from './CellTip'
import { needIsOpenSlot, needLabel } from './need'

// THE PLAN'S ANSWER FOR ONE TURN, as three cards.
//
// This replaces TopThree, and the difference is not cosmetic. Those cards
// were the top three rows of a list this app ranked itself, captioned with a
// number (`gain_now`) priced against a pick that was not the reader's. These
// are the target and two alternates the plan named for a SPECIFIC turn --
// the one on the clock while it is the reader's, his next one while it is
// not -- with the plan's own reasons for and against printed under them.
// One rule picks them, the same rule that built the rest of the plan (see
// scoring/plan.py); nothing here re-derives a recommendation of its own.

// duplicated from AvailableList.tsx (which duplicated it from
// RosterPanel.tsx/DraftBoardGrid.tsx in turn) -- see either for why a
// four-line pure function is copied rather than shared.
function posBadge(position: string): ReactNode {
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

/** The position's own hue, the token the badges and the board already use.
 *  Unknown positions fall to the muted text colour rather than to a colour
 *  that would claim a position they do not have. */
function posHue(position: string): string {
  const key = position.toLowerCase()
  return ['qb', 'rb', 'wr', 'te', 'k', 'dst'].includes(key)
    ? `var(--pos-${key})` : 'var(--text-3)'
}

function fmtSigned(n: number): string {
  const r = Math.round(n)
  return r > 0 ? `+${r}` : `${r}`
}

// Rounded before the sign is read, and null is not a zero: a figure nobody
// computed carries no direction, so it carries no colour either.
function edgeTone(edge: number | null): string {
  if (edge === null) return ''
  const shown = Math.round(edge)
  return shown > 0 ? 'is-up' : shown < 0 ? 'is-down' : ''
}

// Same shape as AvailableList's own fmtRank.
function fmtRank(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—'
  return Number.isInteger(n) ? String(n) : n.toFixed(1)
}

// The list on screen was ranked for an older pick than the one on the clock
// -- `forPick` is the pick being recomputed for (the one on the clock),
// `listedForPick` is the pick the cards and table below were actually
// ranked for. Both spelled by DraftRoom off `candidates_as_of_pick`; null
// whenever the two agree, which is the overwhelming majority of polls.
export type RecomputeState = { forPick: number; listedForPick: number }

/** WHICH TURN THESE CARDS ARE FOR.
 *
 *  The plan runs over the reader's REMAINING turns, so while he is on the
 *  clock its first entry is this very pick and while he is waiting its first
 *  entry is the next one he owns. That would make `plan[0]` right nearly
 *  always -- but "nearly" is doing real work there: a plan computed one pick
 *  ago and a clock that has since moved disagree for a poll or two, and the
 *  cards would then be captioned with a turn that has already happened. So
 *  the turn is chosen by PICK NUMBER against the clock rather than by
 *  position in the list, and `plan[0]` is only the fallback when no entry
 *  matches at all.
 *
 *  `onTheClock`, NOT the boolean that drives the Draft button. Whether the
 *  reader may send a pick and whether the pick is his are two different
 *  facts: an unpaid room and a socket mid-reconnect both disable the button
 *  while the clock is still running on his seat. Choosing the turn off the
 *  button's boolean showed him next turn's three names, headed "Your next
 *  turn", during the pick he was actually holding -- the worst moment in the
 *  draft to be describing a different one.
 */
export function turnFor(
  plan: LivePlanTurn[], pickNo: number | null, onTheClock: boolean,
): LivePlanTurn | null {
  if (plan.length === 0) return null
  if (pickNo === null) return plan[0]
  if (onTheClock) return plan.find((t) => t.pick_no === pickNo) ?? plan[0]
  return plan.find((t) => t.pick_no > pickNo) ?? plan[0]
}

/** A row on a card, whether or not a plan produced it.
 *
 *  With no plan at all -- the landing page's spectator room, a session whose
 *  seat has not resolved -- the top of the board is still worth showing, and
 *  the candidate rows carry the same two numbers the plan's targets do. What
 *  they cannot carry is reasoning, so those cards render without any: an
 *  empty pros list draws nothing, rather than a made-up sentence. */
function fromCandidate(c: LiveCandidate): PlanPlayer {
  return {
    player_id: c.player_id,
    lasts_pct: c.lasts_pct,
    edge_pts: c.edge_pts,
    edge_at_pick: c.edge_at_pick,
    pros: [],
    cons: [],
  }
}

interface TargetCardsProps {
  /** The reader's remaining turns, straight off `/api/live/state`. Empty is
   *  a legitimate state, not an error -- see `fromCandidate`. */
  plan: LivePlanTurn[]
  /** The same list the table below renders, used to join a planned player
   *  back to his position, his need tag and the market's ranks. */
  candidates: LiveCandidate[]
  players: Record<string, Player>
  onDraft: (c: LiveCandidate) => void
  /** Whether the reader may actually send a pick right now: his turn, on a
   *  live socket, in a room he has paid for. Gates the Draft button and
   *  nothing else. */
  isMyTurn: boolean
  /** Whether the pick on the clock is HIS -- which is true in a room he
   *  cannot draft from too. This is what decides which turn these cards
   *  describe; see `turnFor`. */
  onTheClock?: boolean
  /** The pick on the clock. Decides which turn these cards are for, and
   *  lets the market figures say how far past his ADP a player has fallen. */
  pickNo?: number | null
  /** The league, for `startersAt` and the hover panels. */
  settings?: LiveSettings | null
  // Non-null only while a recompute is outstanding -- see RecomputeState.
  // Rendered on this header row rather than as a banner of its own: a strip
  // that mounts and unmounts between polls moves the Draft button under a
  // resting cursor, which on a thirty-second clock is a misdraft risk.
  recompute: RecomputeState | null
  // Opens the profile over the room -- the name only, never the card: the
  // card's own control is Draft, and that one cannot be undone.
  onOpenPlayer: (c: LiveCandidate) => void
}

export default function TargetCards({
  plan, candidates, players, onDraft, isMyTurn, onTheClock = false,
  pickNo = null, settings = null, recompute, onOpenPlayer,
}: TargetCardsProps) {
  // THE TABLE'S OWN HOVER PANELS, on the cards. `CellTip` is the same
  // component the available list opens under its Health, Reliable and Growth
  // columns -- a fetched game log, a season-by-season chart -- and the meters
  // up here would otherwise draw the identical bars with no way to ask what
  // is behind them. The machinery is the table's too, down to the 130ms
  // delay and `positionTip`.
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
      // Behind the delay, not ahead of it -- see the same change in
      // AvailableList's `showCellTip` for why a pointer merely passing over
      // these cards must not fire a profile request per card it touches.
      panelTimer.current = window.setTimeout(
        () => {
          void loadProfile(playerId).catch(() => {})
          setPanel({ kind, playerId, rect })
        }, TIP_DELAY_MS)
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

  const turn = turnFor(plan, pickNo, onTheClock)
  // A turn with no clear target (the plan's honest null) still shows its
  // alternates, if any; with none at all the caption says so rather than
  // the cards falling back to the top of the board as if there were no plan.
  const rows: PlanPlayer[] = turn
    ? [...(turn.target ? [turn.target] : []), ...turn.alternates].slice(0, 3)
    : candidates.slice(0, 3).map(fromCandidate)
  const noClearTarget = turn !== null && turn.target === null
  if (rows.length === 0 && !noClearTarget) return null

  const byId = new Map(candidates.map((c) => [c.player_id, c]))
  // ON THE CLOCK the plan's first turn IS this pick, so the cards are a
  // recommendation; waiting, they are a forecast, and the caption has to say
  // which. Getting this backwards would be the old room's mistake in a new
  // place: a number captioned as though it answered a question it does not.
  const onClockTurn = turn !== null && pickNo !== null && turn.pick_no === pickNo

  return (
    <div className="target">
      <div className="target-head">
        <span className="draft-cap target-cap">
          {onClockTurn || turn === null ? 'Take one of these' : 'Your next turn'}
        </span>
        <span className="target-hint">
          {turn === null
            ? "the top of ESPN's board"
            : `pick ${turn.pick_no} · round ${turn.round}`}
        </span>
        {/* Always mounted, even with nothing to say -- two reasons, both
            load-bearing. (1) Layout: an element that comes and goes could
            still change this row, and this row sits directly above the
            Draft buttons. Empty, it is a zero-width flex item pushed to the
            far right by `margin-left: auto`, so neither its arrival nor its
            departure moves a pixel of anything. (2) Assistive tech: a
            `role="status"` region announces text that CHANGES inside an
            already-live region -- mounting the region and its text in the
            same commit is not reliably announced. */}
        <span className="target-recompute" role="status">
          {recompute !== null && (
            <>
              <span className="target-recompute-dot" aria-hidden="true" />
              Recomputing for pick {recompute.forPick}, and the list below is
              {' '}still for pick {recompute.listedForPick}
            </>
          )}
        </span>
      </div>
      {noClearTarget && rows.length === 0 && (
        <p className="target-none" role="status">
          No clear target for pick {turn?.pick_no}: nobody is likely enough to
          be there, or worth waiting for. The list below is in ESPN&apos;s order.
        </p>
      )}
      <div className="target-grid">
        {rows.map((row, i) => {
          const c = byId.get(row.player_id)
          const player = players[row.player_id]
          const position = c?.position ?? player?.position ?? ''
          // The pick the two figures are measured to. On a future turn that
          // is the turn itself; on the clock it is the reader's NEXT turn,
          // which is the only pick "will he still be there" can mean while
          // he is holding this one.
          const atPick = onClockTurn || turn === null
            ? c?.lasts_at_pick ?? null
            : turn.pick_no
          const at = atPick === null ? 'your next turn' : `pick ${atPick}`
          // ON THE CLOCK THE FIGURES COME OFF THE CANDIDATE ROW, not the
          // plan's. Both are measured to the same pick in that case, but
          // only the candidate row is measured to it BY CONTRACT: its
          // `lasts_pct` is always the chance at `lasts_at_pick`, which is
          // the pick the caption names. A plan turn's figures belong to
          // that turn, and for the turn you are already holding the honest
          // reading of "will he last" is the one about waiting for the
          // next one. On a future turn the plan's are the only ones that
          // describe it, so those are used.
          const source = onClockTurn || turn === null ? c ?? row : row
          const lasts = source.lasts_pct
          const edge = source.edge_pts
          // THE EDGE IS NOT PRICED AT THE PICK ABOVE IT. It is what taking
          // him at this turn beats, so it is measured to the turn AFTER the
          // one the card is for -- pick 11 on a card headed "pick 6". The
          // caption used to reuse `at` and name the card's own pick, which
          // made the big number and the reason under it disagree by a whole
          // round. The server says which pick it priced; "later" is the
          // honest fallback when it says nothing.
          const edgeAtPick = source.edge_at_pick ?? null
          const edgeAt = edgeAtPick === null ? 'later' : `at pick ${edgeAtPick}`
          return (
            <div key={row.player_id}
                 className={`target-card${i === 0 ? ' target-card-lead' : ''}`}>
              <div className="target-card-head">
                <span className="target-rank mono">
                  {i === 0 ? 'TARGET' : `ALT ${i}`}
                </span>
                {/* The face, bigger than a badge because it is the fastest
                    way anybody tells three cards apart under a clock. The
                    ring is his position's own hue, which is why the badge
                    only appears for the players who have no photograph. */}
                {player?.headshot ? (
                  <img className="target-shot" src={player.headshot} alt=""
                       loading="lazy"
                       style={{ borderColor: posHue(position) }} />
                ) : posBadge(position || '?')}
                <span className="target-id">
                  <button
                    type="button"
                    className="target-name-btn"
                    onClick={() => c && onOpenPlayer(c)}
                    disabled={!c}
                    title="Open profile"
                  >
                    {c?.favourite && (
                      <span className="target-star" role="img"
                            aria-label="One of your guys">★</span>
                    )}
                    <span className="target-name">
                      {player?.name ?? row.player_id}
                    </span>
                  </button>
                  <span className="target-sub mono">
                    {position}
                    {player?.team ? ` · ${player.team}` : ''}
                    {player?.bye ? ` · BYE ${player.bye}` : ''}
                    {c?.need ? (
                      <span className={`target-need${needIsOpenSlot(c.need) ? ' is-open' : ''}`}>
                        {needLabel(c.need)}
                      </span>
                    ) : null}
                  </span>
                </span>
                <button
                  type="button"
                  className="avail-draft-btn"
                  disabled={!isMyTurn || !c}
                  title={isMyTurn ? undefined : 'Not your turn yet'}
                  onClick={() => c && onDraft(c)}
                >
                  Draft
                </button>
              </div>

              {/* THE TWO NUMBERS THE PICK TURNS ON, at the size of the
                  decision they carry: will he be there, and what it costs
                  if he is not. Everything else on the card is evidence for
                  or against those two. */}
              <div className="target-figs">
                <div className="target-fig">
                  <span className="target-fig-num mono"
                        style={lasts === null ? undefined : { color: riskTone(lasts) }}>
                    {lasts === null ? '—' : Math.round(lasts)}
                    {lasts !== null && <span className="target-fig-unit">%</span>}
                  </span>
                  <span className="target-fig-cap">still there at {at}</span>
                </div>
                {/* A missing edge is toned like a missing anything else --
                    not green. `is-up` on a dash claimed a gain nobody
                    computed. An edge that rounds to nothing gets no tone
                    either, the same rule panels.ts's `signedChange` uses. */}
                <div className="target-fig">
                  <span className={`target-fig-num mono ${edgeTone(edge)}`}>
                    {edge === null ? '—' : fmtSigned(edge)}
                    {edge !== null && <span className="target-fig-unit">pts</span>}
                  </span>
                  <span className="target-fig-cap">
                    {edge !== null && Math.round(edge) < 0
                      ? <>vs waiting for {position || 'him'} {edgeAt}</>
                      : <>over the next {position || 'player'} you would get {edgeAt}</>}
                  </span>
                </div>
              </div>

              {/* THE PLAN'S OWN REASONING, in its own words. Rule-derived,
                  never written here: every string in these two lists comes
                  from scoring/plan.py, so a card cannot argue for a pick the
                  plan did not make. An alternate carries none, and draws
                  nothing rather than an empty box. */}
              {(row.pros.length > 0 || row.cons.length > 0) && (
                <ul className="target-reasons">
                  {row.pros.map((pro) => (
                    <li key={pro} className="target-reason is-pro">
                      <span className="target-reason-mark" aria-hidden="true">+</span>
                      {pro}
                    </li>
                  ))}
                  {row.cons.map((con) => (
                    <li key={con} className="target-reason is-con">
                      <span className="target-reason-mark" aria-hidden="true">−</span>
                      {con}
                    </li>
                  ))}
                </ul>
              )}

              <div className="target-meters">
                <Meter caption="Health"
                       onEnter={player?.career_games_pg == null ? null
                         : (el) => showPanel('health', row.player_id, el)}
                       onLeave={hidePanel}>
                  {healthLevel(player?.career_games_pg) !== null
                    ? <HealthMeter level={healthLevel(player?.career_games_pg) as number}
                                   gamesPg={player?.career_games_pg as number} />
                    : <span className="target-meter-none">—</span>}
                </Meter>
                <Meter caption="Reliable"
                       onEnter={player?.consistency_pct == null ? null
                         : (el) => showPanel('steady', row.player_id, el)}
                       onLeave={hidePanel}>
                  {steadyLevel(player?.consistency_pct) !== null
                    ? <SteadyMeter level={steadyLevel(player?.consistency_pct) as number}
                                   cv={player?.consistency_cv ?? null} />
                    : <span className="target-meter-none">—</span>}
                </Meter>
                <Meter caption="Growth"
                       onEnter={player?.proj_change == null ? null
                         : (el) => showPanel('change', row.player_id, el)}
                       onLeave={hidePanel}>
                  {player?.proj_change != null
                    ? <ChangeMeter change={player.proj_change} />
                    : <span className="target-meter-none">—</span>}
                </Meter>
                {/* THE SAME GRAMMAR AS THE METERS BESIDE THEM: a caption
                    over a value, so the eye reads one row of labelled facts
                    rather than three charts and a sentence. */}
                <span className="target-market">
                  <Market caption="ESPN" basis="ESPN's rank for him"
                          rank={c?.espn_rank ?? player?.espn_ppr_rank ?? null}
                          pickNo={pickNo} />
                  <Market caption="ADP" basis="his ADP"
                          rank={c?.espn_adp ?? null} pickNo={pickNo} />
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

/** A meter under its own caption, with the table's own panel behind it.
 *
 *  FIVE BARS ARE UNFALSIFIABLE ON THEIR OWN. In the table these columns open
 *  a hover panel that gives the measurement -- which is why `HealthMeter`
 *  and the rest carry no `title`, since the native box rendered on top of
 *  that panel. On a card there would be no panel at all, so the bars would
 *  be four greens with no way to ask what they meant. */
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
    <span className="target-meter">
      <span className="draft-cap">{caption}</span>
      <span
        className={onEnter ? 'target-hoverable' : undefined}
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

/** One market's answer: where it ranks him, and how far past that he is.
 *
 *  `basis` is what the distance is measured against, in words, because the
 *  two cells here are NOT the same kind of number: one is ESPN's rank and
 *  one is ESPN's ADP, and a tooltip that said "past his ADP" under a rank
 *  would be describing a figure that is not on screen. */
function Market({ caption, basis, rank, pickNo }: {
  caption: string; basis: string; rank: number | null; pickNo: number | null
}) {
  return (
    <span className="target-meter">
      <span className="draft-cap">{caption}</span>
      <span className="target-market-val mono">
        {fmtRank(rank)}
        <MarketMove pickNo={pickNo} rank={rank} basis={basis} />
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
 *  bare zero is a fact about arithmetic rather than about the draft. */
function MarketMove({ pickNo, rank, basis }: {
  pickNo: number | null; rank: number | null; basis: string
}) {
  if (pickNo === null || rank === null) return null
  const slots = Math.round(pickNo - rank)
  if (slots === 0) return null
  const steal = slots > 0
  return (
    <span className={`target-market-move ${steal ? 'is-steal' : 'is-reach'}`}
          title={steal
            ? `Still here ${Math.abs(slots)} picks past ${basis}`
            : `Taking him now is ${Math.abs(slots)} picks before ${basis}`}>
      <span aria-hidden="true">{steal ? '▲' : '▼'}</span>
      {Math.abs(slots)}
    </span>
  )
}
