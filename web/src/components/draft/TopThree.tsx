import type { ReactNode } from 'react'
import type { LiveCandidate, Player } from '../../api'
import { riskTone } from './tone'

// duplicated from AvailableList.tsx (which duplicated it from
// RosterPanel.tsx/DraftBoardGrid.tsx in turn) -- see either for why a
// four-line pure function is copied rather than shared.
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
// once, via isRanked below, is what lets reasonFor and this file's own
// fmtSigned/fillsIsOpenSlot keep treating these three as plain values
// instead of every line growing a null check that can never actually fire
// once the guard below has run.
type RankedCandidate = LiveCandidate & { gain_now: number; survive_pct: number; fills: string }

function isRanked(c: LiveCandidate): c is RankedCandidate {
  return c.gain_now !== null
}

// `horizonLabel` is the pick gain_now and survive_pct were measured
// against, already phrased by DraftRoom ("pick 18", "the end of the
// draft"). These sentences used to say "your next pick" and that is now
// wrong: the horizon deliberately skips turns too close to measure anything
// (see scoring/draft_sim.horizon_picks), so at the wheel it is two turns
// away, not one. Naming it is the difference between "he is a 33% bet to
// last to pick 31" -- true, and the number the ranking is built on -- and
// the same sentence about a pick nobody measured. The fallback phrase names
// no pick at all rather than guessing one; it is only reachable if a ranked
// list ever arrives without its horizon, which api/live.py writes under one
// lock precisely so it cannot.
function reasonFor(c: RankedCandidate, horizonLabel: string | null): string {
  const survive = Math.round(c.survive_pct)
  const at = horizonLabel ?? 'the turn this list is measured against'

  const slot =
    c.fills === '—' ? 'has no roster slot open right now'
      : c.fills === 'BENCH' ? 'would only add bench depth'
        : `fills your open ${c.fills} slot`

  if (c.fills === '—') {
    return `He ${slot} -- this pick would not start no matter what, whatever his ${survive}% odds of lasting to ${at} are worth.`
  }
  if (c.gain_now <= 0) {
    return `He ${slot}, but the model expects an option at least as good at his position to still be there at ${at} -- taking him now nets ${fmtSigned(c.gain_now)} against that, whatever his own ${survive}% survival odds are worth, so this is not the pick to make yet.`
  }
  if (c.vor_points > 0 && c.gain_now / c.vor_points < 0.15) {
    return `He ${slot}, but the next-best option there is nearly as good -- only ${fmtSigned(c.gain_now)} is actually at stake, well short of the ${fmtSigned(c.vor_points)} over replacement he shows on the board.`
  }
  if (survive >= 55) {
    return `He ${slot} and is worth ${fmtSigned(c.gain_now)} now, but he is a ${survive}% bet to still be there at ${at} -- fine to wait if you want someone else first.`
  }
  return `He ${slot} and is worth ${fmtSigned(c.gain_now)} more than the best option likely to still be there at ${at} -- only a ${survive}% chance he lasts that long, so this is the one to take now.`
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
  // /api/live/state's own `horizon_pick`/`horizon_is_end_of_draft`. Not a
  // number and not derived here: it is the server's horizon (the first turn
  // far enough away to carry signal, see scoring/draft_sim.horizon_picks),
  // and the end-of-draft case has no pick number to print at all. null
  // whenever no gain-ranked list exists yet -- the hint drops the clause
  // entirely rather than guessing.
  horizonLabel: string | null
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
  candidates, players, onDraft, isMyTurn, horizonLabel, recompute, onOpenPlayer,
}: TopThreeProps) {
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
        <span className="top3-hint">
          ranked by what you gain now vs. waiting{horizonLabel !== null ? ` until ${horizonLabel}` : ''}
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
              Recomputing for pick {recompute.forPick} — the list below is
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
                {posBadge(c.position)}
                <button
                  type="button"
                  className="top3-name-btn"
                  onClick={() => onOpenPlayer(c)}
                  title="Open profile"
                >
                  <span className="top3-name">{player?.name ?? c.player_id}</span>
                </button>
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
              {/* Caption wording is shared verbatim with ConfirmPick's
                  figure row and playerSeed's profile seed -- see
                  AvailableList.tsx's comment for what each one means and
                  why it reads the way it does. Three surfaces, one label
                  per quantity: rename them together or not at all. */}
              <div className="top3-figures">
                <div>
                  <div className="draft-cap">Gain vs waiting</div>
                  <div className="top3-figure mono">{fmtSigned(c.gain_now)}</div>
                </div>
                <div>
                  {/* The caption names the horizon, because the figure
                      under it is a probability of lasting to THAT pick --
                      "He lasts 0%" reads as "he will not survive your next
                      pick" and would be flatly wrong at a wheel, where the
                      measured turn is two turns out. */}
                  <div className="draft-cap">{horizonLabel !== null ? `Lasts to ${horizonLabel}` : 'He lasts'}</div>
                  <div className="top3-figure mono" style={{ color: riskTone(c.survive_pct) }}>
                    {Math.round(c.survive_pct)}%
                  </div>
                </div>
                <div>
                  <div className="draft-cap">Roster slot</div>
                  <div className={`top3-figure mono${fillsIsOpenSlot(c.fills) ? ' is-open' : ''}`}>
                    {c.fills}
                  </div>
                </div>
              </div>
              <p className="top3-reason">{reasonFor(c, horizonLabel)}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
