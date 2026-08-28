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
 */
export function turnFor(
  plan: LivePlanTurn[], pickNo: number | null, isMyTurn: boolean,
): LivePlanTurn | null {
  if (plan.length === 0) return null
  if (pickNo === null) return plan[0]
  if (isMyTurn) return plan.find((t) => t.pick_no === pickNo) ?? plan[0]
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
  // See AvailableList.tsx's own field comment -- the draft button has to
  // gate on whose turn it is, and this component has no other way to know.
  isMyTurn: boolean
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
  plan, candidates, players, onDraft, isMyTurn, pickNo = null,
  settings = null, recompute, onOpenPlayer,
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

  const turn = turnFor(plan, pickNo, isMyTurn)
  const rows: PlanPlayer[] = turn
    ? [turn.target, ...turn.alternates].slice(0, 3)
    : candidates.slice(0, 3).map(fromCandidate)
  if (rows.length === 0) return null

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
          const lasts = row.lasts_pct
          const edge = row.edge_pts
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
                <div className="target-fig">
                  <span className={`target-fig-num mono ${edge === null || Math.round(edge) >= 0 ? 'is-up' : 'is-down'}`}>
                    {edge === null ? '—' : fmtSigned(edge)}
                    {edge !== null && <span className="target-fig-unit">pts</span>}
                  </span>
                  <span className="target-fig-cap">
                    over the next {position || 'player'} you would get at {at}
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
                  <Market caption="ESPN" rank={c?.espn_rank ?? player?.espn_ppr_rank ?? null}
                          pickNo={pickNo} />
                  <Market caption="ADP" rank={c?.espn_adp ?? null} pickNo={pickNo} />
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

/** One market's answer: where it ranks him, and how far past that he is. */
function Market({ caption, rank, pickNo }: {
  caption: string; rank: number | null; pickNo: number | null
}) {
  return (
    <span className="target-meter">
      <span className="draft-cap">{caption}</span>
      <span className="target-market-val mono">
        {fmtRank(rank)}
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
 *  bare zero is a fact about arithmetic rather than about the draft. */
function MarketMove({ pickNo, rank }: { pickNo: number | null; rank: number | null }) {
  if (pickNo === null || rank === null) return null
  const slots = Math.round(pickNo - rank)
  if (slots === 0) return null
  const steal = slots > 0
  return (
    <span className={`target-market-move ${steal ? 'is-steal' : 'is-reach'}`}
          title={steal
            ? `Still here ${Math.abs(slots)} picks past his ADP`
            : `Taking him now is ${Math.abs(slots)} picks before his ADP`}>
      <span aria-hidden="true">{steal ? '▲' : '▼'}</span>
      {Math.abs(slots)}
    </span>
  )
}
