import type { ReactNode } from 'react'
import type { LiveCandidate, Player } from '../../api'
import { riskTone } from './tone'

// duplicated from AvailableList.tsx (which duplicated it from
// RosterPanel.tsx/DraftBoardGrid.tsx in turn) -- same precedent as
// PlayerCard.tsx's depthSlotLabel/sosLabel.
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

function fillsIsOpenSlot(fills: string): boolean {
  return fills !== 'BENCH' && fills !== '—'
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
// The "next-best option is nearly as good" framing below only makes sense
// as a *fraction of his own vor_points* -- and a fraction is only a
// coherent idea when that denominator is positive. Reviewed-in bug: the
// first version of this function defaulted the ratio to 0 whenever
// `vor_points` was not positive, which routed EVERY such row into that
// same "nearly as good" branch regardless of how large `gain_now` actually
// was -- the exact failure this task exists to eliminate, a sentence
// asserting the opposite of what the numbers say. This is reachable, not
// hypothetical: NEED_WEIGHTS["starter"] = 1.0 (scoring/config.py), so for a
// starter need `gain_now` is literally `vor_points - next_best` with no
// dampening -- a thin, weak position can hand a real starter slot
// `vor_points = -5` (below replacement in an absolute sense) with
// `next_best = -30` (the position's replacement level is even worse),
// giving `gain_now = +25`: a genuine, sizeable gain that the old ratio
// branch would have called "nearly as good" anyway.
//
// Fixed by gating the whole comparison on `vor_points > 0`, not just the
// division -- a non-positive `vor_points` now falls straight through to
// the `survive_pct`/default branches below, neither of which mentions
// `vor_points` at all, only `gain_now` (always literally true: it IS what
// taking him now gains you, whatever sign it carries) and `survive_pct`
// (his own odds, independent of vor entirely).
function reasonFor(c: LiveCandidate): string {
  const survive = Math.round(c.survive_pct)

  const slot =
    c.fills === '—' ? 'has no roster spot open right now'
      : c.fills === 'BENCH' ? 'would only add bench depth'
        : `fills your open ${c.fills} slot`

  if (c.fills === '—') {
    return `He ${slot} -- this pick would not start no matter what, whatever his ${survive}% odds of lasting to your next one are worth.`
  }
  if (c.vor_points > 0 && c.gain_now / c.vor_points < 0.15) {
    return `He ${slot}, but the next-best option there is nearly as good -- only ${fmtSigned(c.gain_now)} is actually at stake, well short of the ${fmtSigned(c.vor_points)} over replacement he shows on the board.`
  }
  if (survive >= 55) {
    return `He ${slot} and is worth ${fmtSigned(c.gain_now)} now, but he is a ${survive}% bet to still be there at your next pick -- fine to wait if you want someone else first.`
  }
  return `He ${slot} and is worth ${fmtSigned(c.gain_now)} over the best replacement there -- only a ${survive}% chance he lasts to your next pick, so this is the one to take now.`
}

interface TopThreeProps {
  candidates: LiveCandidate[]
  players: Record<string, Player>
  onDraft: (c: LiveCandidate) => void
  // See AvailableList.tsx's own field comment -- not in the task brief's
  // literal signature, added for the same reason: the draft button has to
  // gate on whose turn it is, and this component has no other way to know.
  isMyTurn: boolean
  // The overall pick number `gain_now` is actually measured against --
  // "what you gain by taking him now instead of waiting until pick N."
  // DraftRoom derives it the same way ClockPanel already does (a
  // duplicated copy of ClockPanel.tsx's own nextPickFor -- see its
  // comment there). null whenever DraftRoom can't derive it honestly (no
  // active session, my_slot not resolved yet, or the draft is over) --
  // the hint drops the clause entirely rather than guessing a number.
  nextPickNo: number | null
}

// Three cards above the ranked table, the same `candidates` prop
// unfiltered -- AvailableList's own search/position filter is local to
// that component and never touches what shows up here. This always names
// the three best picks on the board by gain_now, regardless of what the
// user happens to be searching for below.
export default function TopThree({ candidates, players, onDraft, isMyTurn, nextPickNo }: TopThreeProps) {
  const top3 = candidates.slice(0, 3)
  if (top3.length === 0) return null

  return (
    <div className="top3">
      <div className="top3-head">
        <span className="draft-cap top3-cap">Take one of these</span>
        <span className="top3-hint">
          ranked by what you gain now vs. waiting{nextPickNo !== null ? ` until pick ${nextPickNo}` : ''}
        </span>
      </div>
      <div className="top3-grid">
        {top3.map((c, i) => {
          const player = players[c.player_id]
          return (
            <div key={c.player_id} className={`top3-card${i === 0 ? ' top3-card-lead' : ''}`}>
              <div className="top3-card-head">
                <span className="top3-rank mono">#{i + 1}</span>
                {posBadge(c.position)}
                <span className="top3-name">{player?.name ?? c.player_id}</span>
                <span className="top3-team mono">{player?.team ?? ''}</span>
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
              <div className="top3-figures">
                <div>
                  <div className="draft-cap">Gain now</div>
                  <div className="top3-figure mono">{fmtSigned(c.gain_now)}</div>
                </div>
                <div>
                  <div className="draft-cap">He lasts</div>
                  <div className="top3-figure mono" style={{ color: riskTone(c.survive_pct) }}>
                    {Math.round(c.survive_pct)}%
                  </div>
                </div>
                <div>
                  <div className="draft-cap">Fills</div>
                  <div className={`top3-figure mono${fillsIsOpenSlot(c.fills) ? ' is-open' : ''}`}>
                    {c.fills}
                  </div>
                </div>
              </div>
              <p className="top3-reason">{reasonFor(c)}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
