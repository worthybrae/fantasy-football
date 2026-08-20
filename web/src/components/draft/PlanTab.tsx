import { useState, type ReactNode } from 'react'
import type { PlanBestAvailableRound, PlanCliffRound, PlanFetchResult, PlanRound } from '../../api'

// duplicated from AvailableList.tsx/RosterPanel.tsx (unexported in both): a
// four-line pure function isn't worth a shared module between five views now.
function posBadge(position: string): ReactNode {
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

// The four positions the WHY table names, in EITHER of its two views (the
// cliff view and the best-available view both key off this same list).
// `cliffs`/`best_available` never carry K or DST (see api.ts's own comment
// on PlanCliffRound) -- unlike the round-by-round plan above it, which
// legitimately predicts both. Fixed column order, matching the mock's own
// QB/RB/WR/TE header.
const WHY_POSITIONS = ['QB', 'RB', 'WR', 'TE']

// A round's prediction reads as a bar only when the top position is
// genuinely uncertain -- 60% or better and the field has all but decided
// the round, so the remaining shares underneath it would just be noise (per
// the task brief: "secondary positions shown inline when the top one is
// under ~60%").
const SECONDARY_THRESHOLD = 60

// Below this the single worst cost in a round is small enough that calling
// it out as "a cliff" would be crying wolf -- most rows in a real plan are
// all zeros or single digits, and those get no annotation at all. Above it,
// every position within CLIFF_ANNOTATION_RATIO of the round's worst cost
// joins the callout. Checked against both worked rows in the task brief:
// R1's 86.4 RB / 85.1 WR both clear 60% of 86.4 and neither QB nor TE's 0.0
// does; R3's 47 QB / 31 TE both clear 60% of 47 while its own 14 WR / 4 RB
// do not -- both reproduce the brief's own "← RB/WR cliff" / "← QB/TE
// cliff" examples exactly.
const CLIFF_ANNOTATION_FLOOR = 20
const CLIFF_ANNOTATION_RATIO = 0.6

// Points-lost color ramp for the WHY table's cells: recedes to --text-3 at
// 0, pops to --accent at CLIFF_CEILING and above. Fixed, not scaled to each
// row's own max -- same reasoning as AvailableList's game-point bars (a
// fixed scale is what lets one row's "big" mean the same thing as
// another's). 90 comfortably covers the task brief's own worked numbers
// (60-90 called out there as "the actionable signal").
const CLIFF_CEILING = 90
function cliffTone(v: number): string {
  const k = Math.max(0, Math.min(1, v / CLIFF_CEILING))
  return `color-mix(in srgb, var(--accent) ${Math.round(k * 100)}%, var(--text-3))`
}

// Which of the four cliff positions stand out enough in this round to
// annotate -- see CLIFF_ANNOTATION_FLOOR/_RATIO above. Returns positions in
// WHY_POSITIONS' own order (QB, RB, WR, TE), matching the table's column
// order and the mock's own "← RB/WR cliff" phrasing.
function standoutPositions(byPosition: Record<string, number> | undefined): string[] {
  if (!byPosition) return []
  const entries = WHY_POSITIONS.map((p): [string, number] => [p, byPosition[p] ?? 0])
  const max = Math.max(...entries.map(([, v]) => v))
  if (max < CLIFF_ANNOTATION_FLOOR) return []
  return entries.filter(([, v]) => v >= max * CLIFF_ANNOTATION_RATIO).map(([p]) => p)
}

// The best-available view's own visual language -- deliberately NOT
// `cliffTone`/`CLIFF_CEILING` above. Those numbers are points LOST (0 is
// common, 90 is a big deal); these are points still ON THE BOARD (200-370,
// never near 0), so painting them on the same 0-90 ramp would peg every
// cell near the top of the scale and the whole row would read as "all
// maximum, all bright" -- exactly the failure the task calls out for a row
// like QB 369.7 / RB 364.9 / WR 356.3 / TE 242.0.
//
// What the reader actually wants from a LEVEL table is where it moved, not
// how big it is in isolation. Since the pool only shrinks pick over pick,
// `best_available` is non-increasing round over round, so there are only
// two states worth telling apart:
//   * unchanged from the round above it -- waiting cost nothing here, so it
//     recedes to `--text-3` and the eye skips past it (this is what keeps
//     QB/TE quiet across rounds 1-3 in the worked example, even though
//     their numbers are themselves large).
//   * dropped from the round above it -- the position's ceiling actually
//     fell between these two picks, so it renders in the position's own
//     color and picks up weight, the same "big number gets heavier" idiom
//     `.plan-cliff-value.is-loud` already uses for the cliff view.
// The first row has nothing above it to compare against and renders in the
// position's own color at normal weight: "here's what's on the board right
// now," not yet a claim about change.
function bestAvailTone(position: string, v: number, prevV: number | undefined): { cls: string; color: string } {
  const posColor = `var(--pos-${position.toLowerCase()})`
  if (prevV === undefined) return { cls: 'plan-avail-value', color: posColor }
  if (v < prevV) return { cls: 'plan-avail-value is-drop', color: posColor }
  return { cls: 'plan-avail-value', color: 'var(--text-3)' }
}

// Same spinner idiom as PlayerProfile's `.pp-loading-tail`/`.pp-loading-dot`
// and TopThree's `.top3-recompute` -- an indicator riding the tail of a line
// that is already on screen, never a block of its own, so a refetch landing
// (or not) never moves anything below it. Used both while the very first
// plan is loading and on every quiet refetch after a pick lands; either
// way, whatever `plan` already holds (possibly nothing yet) keeps rendering
// underneath it unchanged.
function LoadingTail() {
  return (
    <span className="plan-loading-tail" role="status">
      <span className="plan-loading-dot" aria-hidden="true" />
      updating…
    </span>
  )
}

interface PlanTabProps {
  // The last successfully normalized /api/live/plan response, or null before
  // the very first one has landed. Its own `status` carries the
  // active/pending/ready distinction -- see PlanFetchResult's own comment in
  // api.ts for what each of the three means and why they collapse into one
  // union.
  result: PlanFetchResult | null
  // True while a fetch is in flight -- both a `pending` retry (see
  // DraftRoom's poll effect) and a `ready` refetch after a pick lands.
  // Whatever `result` already holds keeps rendering underneath it unchanged
  // the whole time (nothing here ever shows a blocking spinner over content
  // that already exists).
  loading: boolean
  // The request itself failed (network error, non-2xx) -- distinct from
  // `result.error`, which is the server's own report that its last PLAN
  // REBUILD attempt threw while still possibly serving a good `result` out
  // the other side. Rendered alongside whatever `result` still holds rather
  // than replacing it, the same "one hiccup must not blank an otherwise-fine
  // view" rule `boardError` follows in DraftRoom.
  fetchError: string | null
}

export default function PlanTab({ result, loading, fetchError }: PlanTabProps) {
  // Which of the WHY table's two views is showing -- purely local UI state,
  // never threaded up to DraftRoom and never persisted: closing the tab (or
  // a refetch landing) is allowed to reset it back to the default, same as
  // AvailableList's own `search`/`pos`/`sort` state.
  const [whyView, setWhyView] = useState<'cliff' | 'available'>('cliff')

  if (result === null || result.status !== 'ready') {
    return (
      <div className="plan-tab">
        <div className="plan-head">
          <span className="draft-cap">Your plan</span>
          {loading && <LoadingTail />}
        </div>
        {fetchError && <p className="error draft-error-banner">{fetchError}</p>}
        {result?.status === 'pending' && result.error && (
          <p className="error draft-error-banner">{result.error}</p>
        )}
        <p className="rail-empty plan-empty">
          {result?.status === 'inactive'
            ? 'Your plan will appear once you’re connected to a live draft.'
            : 'Loading your plan…'}
        </p>
      </div>
    )
  }

  const { plan } = result
  const picksLine = plan.rounds_plan.map((r) => r.pick).join(' · ')
  // The first round not yet drafted -- "where am I", the one thing the task
  // brief calls out as needing to be instantly findable. `undefined` when
  // every round in the plan is already past (the draft is over for this
  // slot, or its very last pick just landed).
  const currentRound = plan.rounds_plan.find((r) => !r.is_past)?.round
  const cliffsByRound = new Map(plan.cliffs.map((c): [number, PlanCliffRound] => [c.round, c]))
  // Built from `best_available`, its OWN array -- never derived from
  // `cliffsByRound` above. The two datasets run different lengths (see
  // LivePlan's own comment in api.ts: cliffs is one shorter, since the last
  // round has no next round to measure a drop against), so each view's rows
  // must come from its own join against `rounds_plan`, not from reusing the
  // other view's already-joined rows.
  const bestAvailByRound = new Map(
    plan.best_available.map((b): [number, PlanBestAvailableRound] => [b.round, b]),
  )

  return (
    <div className="plan-tab">
      <div className="plan-head">
        <span className="draft-cap">Your plan</span>
        <span className="plan-head-line">
          Slot <strong className="mono">{plan.my_slot}</strong>
          <span className="plan-head-sep" aria-hidden="true">·</span>
          <span className="mono">picks {picksLine}</span>
        </span>
        {/* `as_of_pick` is the plan's OWN pick count, computed on a separate,
            slower worker than the live recommendation list -- it legitimately
            lags the draft by a pick or two and that is not a fault to flag,
            just a fact worth captioning. */}
        <span className="plan-head-meta">as of pick {plan.as_of_pick}</span>
        {loading && <LoadingTail />}
      </div>

      {fetchError && <p className="error draft-error-banner">{fetchError}</p>}
      {/* The last rebuild attempt threw, but a good plan (the one rendered
          below) is still on hand -- a small notice, not a replacement for
          the tab's own content. See PlanFetchResult's own comment. */}
      {result.error && (
        <p className="draft-alert-banner" role="status">
          The plan couldn't be refreshed ({result.error}) -- showing the last
          {' '}good one.
        </p>
      )}

      <section className="plan-section">
        <h3 className="draft-cap plan-section-title">Round-by-round plan</h3>
        <ul className="plan-rounds">
          {plan.rounds_plan.map((r) => (
            <PlanRoundRow key={r.round} round={r} isCurrent={r.round === currentRound} />
          ))}
        </ul>
      </section>

      <section className="plan-section">
        <div className="plan-section-head">
          <h3 className="draft-cap plan-section-title">
            {whyView === 'cliff'
              ? 'Why — points lost by waiting a round'
              : 'Best available — projected points still on the board at that pick'}
          </h3>
          {/* Local UI state only (see `whyView` above) -- toggles which
              dataset the table below reads from, nothing else on the tab. */}
          <div className="plan-why-toggle" role="group" aria-label="WHY table view">
            <button
              type="button"
              className={`plan-why-toggle-btn${whyView === 'cliff' ? ' is-active' : ''}`}
              aria-pressed={whyView === 'cliff'}
              onClick={() => setWhyView('cliff')}
            >
              Points lost by waiting
            </button>
            <button
              type="button"
              className={`plan-why-toggle-btn${whyView === 'available' ? ' is-active' : ''}`}
              aria-pressed={whyView === 'available'}
              onClick={() => setWhyView('available')}
            >
              Best available
            </button>
          </div>
        </div>
        <div className="plan-cliff-wrap">
          <table className="plan-cliff-table">
            <thead>
              <tr>
                <th className="plan-cliff-col-round">Round</th>
                {WHY_POSITIONS.map((p) => (
                  <th key={p} className="plan-cliff-col-num">{p}</th>
                ))}
                <th className="plan-cliff-col-note" />
              </tr>
            </thead>
            {whyView === 'cliff' ? (
              <tbody>
                {plan.rounds_plan.map((r) => {
                  const cliff = cliffsByRound.get(r.round)
                  const standout = standoutPositions(cliff?.by_position)
                  return (
                    <tr key={r.round} className={r.round === currentRound ? 'is-current' : undefined}>
                      <td className="plan-cliff-col-round mono">
                        R{r.round}
                        <span className="plan-cliff-pick">p{r.pick}</span>
                      </td>
                      {WHY_POSITIONS.map((p) => {
                        const v = cliff?.by_position[p]
                        return (
                          <td key={p} className="plan-cliff-col-num mono">
                            {v === undefined ? (
                              <span className="plan-cliff-dash">—</span>
                            ) : (
                              <span
                                className={`plan-cliff-value${v >= CLIFF_ANNOTATION_FLOOR ? ' is-loud' : ''}`}
                                style={{ color: cliffTone(v) }}
                              >
                                {Math.round(v)}
                              </span>
                            )}
                          </td>
                        )
                      })}
                      <td className="plan-cliff-col-note">
                        {standout.length > 0 && (
                          <span className="plan-cliff-annotation">← {standout.join('/')} cliff</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            ) : (
              <tbody>
                {plan.rounds_plan.map((r, i) => {
                  // The previous ROUND in this same table, not the previous
                  // entry in `best_available` -- rounds_plan is what decides
                  // which rows the table shows at all (see the comment on
                  // `bestAvailByRound` above), so "the row above this one" is
                  // defined the same way here.
                  const prevRound = i > 0 ? plan.rounds_plan[i - 1] : undefined
                  const avail = bestAvailByRound.get(r.round)
                  const prevAvail = prevRound ? bestAvailByRound.get(prevRound.round) : undefined
                  return (
                    <tr key={r.round} className={r.round === currentRound ? 'is-current' : undefined}>
                      <td className="plan-cliff-col-round mono">
                        R{r.round}
                        <span className="plan-cliff-pick">p{r.pick}</span>
                      </td>
                      {WHY_POSITIONS.map((p) => {
                        const v = avail?.by_position[p]
                        if (v === undefined) {
                          return (
                            <td key={p} className="plan-cliff-col-num mono">
                              <span className="plan-cliff-dash">—</span>
                            </td>
                          )
                        }
                        const tone = bestAvailTone(p, v, prevAvail?.by_position[p])
                        return (
                          <td key={p} className="plan-cliff-col-num mono">
                            <span className={tone.cls} style={{ color: tone.color }}>
                              {Math.round(v)}
                            </span>
                          </td>
                        )
                      })}
                      {/* No annotation column here -- the cliff view's "←
                          RB/WR cliff" callout is calibrated for points LOST
                          (CLIFF_ANNOTATION_FLOOR/_RATIO) and means nothing
                          against a level that runs 200-370. The empty cell is
                          kept, not dropped, so both views share one table
                          shape and the columns never shift under the toggle. */}
                      <td className="plan-cliff-col-note" />
                    </tr>
                  )
                })}
              </tbody>
            )}
          </table>
        </div>
      </section>
    </div>
  )
}

function PlanRoundRow({ round, isCurrent }: { round: PlanRound; isCurrent: boolean }) {
  const primary = round.positions[0]
  const secondary = primary && primary.pct < SECONDARY_THRESHOLD ? round.positions.slice(1) : []
  const fullBreakdown = round.positions.length > 0
    ? round.positions.map((p) => `${p.position} ${p.pct}%`).join(', ')
    : undefined

  return (
    <li className={`plan-round${round.is_past ? ' is-past' : ' is-upcoming'}${isCurrent ? ' is-current' : ''}`}>
      <span className="plan-round-label mono">
        R{round.round}
        <span className="plan-round-pick">p{round.pick}</span>
      </span>
      {isCurrent && <span className="plan-round-now">NOW</span>}

      {round.is_past ? (
        // Settled fact, not prediction -- this pick already happened. No
        // bar, no percentage: a checkmark and the position actually taken,
        // with the plan's own prior prediction (if any) folded into the
        // hover rather than competing with it on the row itself.
        <span className="plan-round-actual" title={fullBreakdown && `Predicted: ${fullBreakdown}`}>
          <span className="plan-round-check" aria-hidden="true">✓</span>
          {round.actual ? posBadge(round.actual) : null}
          <span className="plan-round-actual-label">
            {round.actual ? 'taken' : 'pick made'}
          </span>
        </span>
      ) : primary ? (
        <span className="plan-round-predict" title={fullBreakdown}>
          {posBadge(primary.position)}
          <span className="plan-bar-track">
            <span
              className="plan-bar-fill"
              style={{ width: `${primary.pct}%`, background: `var(--pos-${primary.position.toLowerCase()})` }}
            />
          </span>
          <span className="plan-bar-pct mono">{primary.pct}%</span>
          {secondary.map((s) => (
            <span key={s.position} className="plan-round-secondary">
              {posBadge(s.position)}
              <span className="mono">{s.pct}%</span>
            </span>
          ))}
        </span>
      ) : (
        <span className="plan-round-predict plan-round-nodata">no projection</span>
      )}
    </li>
  )
}
