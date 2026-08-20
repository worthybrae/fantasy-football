import { useEffect } from 'react'
import type { LiveCandidate, Player } from '../../api'

function posBadge(position: string) {
  return <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>{position}</span>
}

// `null` (see AvailableList.tsx's own copy of this function for the full
// rationale) renders as a dash. In practice a candidate never reaches this
// dialog with a null gain_now -- ConfirmPick only ever mounts for a pick
// made while it is actually your turn, which requires my_slot to already be
// resolved -- but the type is honest about every LiveCandidate, not just
// the ones this call site happens to see, so this handles it the same way
// every other reader of the type does rather than asserting it away.
function fmtSigned(n: number | null): string {
  if (n === null) return '—'
  const r = Math.round(n)
  return r > 0 ? `+${r}` : `${r}`
}

// duplicated from AvailableList.tsx/TopThree.tsx -- same rule, same reason
// to keep it in sync: an open starter/flex slot reads in accent, `BENCH`
// and gain.py's `—` read muted. Folded-in review finding: this dialog's
// own Roster slot figure was left plain while both other views already
// applied the rule -- one meaning, drawn the same way everywhere it
// appears.
function fillsIsOpenSlot(fills: string | null): boolean {
  return fills !== null && fills !== 'BENCH' && fills !== '—'
}

export type PickStatus = 'idle' | 'sending' | 'done' | 'failed'

interface ConfirmPickProps {
  candidate: LiveCandidate
  // null when the join table (fetchPlayers, DraftRoom's one-time load)
  // hasn't resolved this id yet -- falls back to the raw player_id, same
  // convention as AvailableList/TopThree.
  player: Player | null
  status: PickStatus
  error: string | null
  // The overall pick number this confirmation is for (DraftRoom's own
  // `thisPickNo`) -- named in the heading ("Confirm pick 30") since this is
  // an action that can't be undone once ESPN confirms it. null only when
  // DraftRoom itself has no current pick to name (draft not active, or
  // already over); the heading falls back to the bare "Confirm pick"
  // rather than a fabricated number.
  pickNo: number | null
  // "Roster after": my_roster's count plus this pick, over the room's total
  // slot count -- both numbers DraftRoom already has (my_roster.length and
  // the assigned `slots` array it builds for RosterPanel), computed there
  // rather than re-derived here since this component has no access to
  // `state` or league settings. null only when the room isn't in an active
  // session at all (should not happen in practice -- there is no candidate
  // to confirm without one -- but typed honestly rather than assumed).
  rosterAfter: { filled: number; total: number } | null
  onConfirm: () => void
  onCancel: () => void
}

// The modal DraftRoom mounts only while `confirming` is non-null -- see its
// own wiring comment. `candidate` is therefore never null here; the parent
// owns the "is there anything to confirm at all" question, not this
// component.
export default function ConfirmPick({
  candidate, player, status, error, pickNo, rosterAfter, onConfirm, onCancel,
}: ConfirmPickProps) {
  const sending = status === 'sending'

  // Escape cancels, same precedent as PlayerProfile.tsx's drawer -- but
  // only while cancel itself is enabled. A SELECT already in flight has
  // left this tab and reached ESPN's socket; Escape must not read as
  // "never mind," since there is no undoing it from here (see the task
  // brief: a 504 means "could not confirm," not "did not happen").
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape' && !sending) onCancel()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [sending, onCancel])

  // DraftRoom clears `confirming` in the same state update that sets
  // status to 'done' (see its handleConfirm), so this branch should never
  // actually paint -- kept as a defensive no-render rather than assuming
  // the parent always unmounts synchronously, since "done closes the
  // modal" is this component's own contract per the task brief, not just a
  // side effect of how the parent happens to be written today.
  if (status === 'done') return null

  return (
    <div className="confirm-overlay">
      <div className="confirm-dialog" role="dialog" aria-modal="true" aria-label="Confirm pick">
        <div className="draft-cap confirm-cap">Confirm pick{pickNo !== null ? ` ${pickNo}` : ''}</div>
        <div className="confirm-player">
          {posBadge(candidate.position)}
          <span className="confirm-player-name">{player?.name ?? candidate.player_id}</span>
          {player && <span className="confirm-player-team mono">{player.team}</span>}
        </div>
        {/* Same captions as TopThree's cards and playerSeed's seed row,
            deliberately word-for-word -- see AvailableList.tsx's comment. */}
        <div className="confirm-figures">
          <div>
            <div className="draft-cap">Roster slot</div>
            <div className={`confirm-figure mono${fillsIsOpenSlot(candidate.fills) ? ' is-open' : ''}`}>
              {candidate.fills ?? '—'}
            </div>
          </div>
          <div>
            <div className="draft-cap">Gain vs waiting</div>
            <div className="confirm-figure mono">{fmtSigned(candidate.gain_now)}</div>
          </div>
          {rosterAfter && (
            <div>
              <div className="draft-cap">Roster after</div>
              <div className="confirm-figure mono">{rosterAfter.filled} / {rosterAfter.total}</div>
            </div>
          )}
        </div>
        {status === 'failed' && error ? (
          // Rendered verbatim -- a 504 here is the API's own "ESPN did not
          // confirm the pick -- check the ESPN draft room before picking
          // again" (api/live.py's live_select), which already says "we
          // could not confirm it," not "it failed." Substituting a shorter
          // message of our own would lose exactly that distinction.
          <p className="confirm-error" role="alert">{error}</p>
        ) : (
          <p className="confirm-note">
            {sending
              ? 'Sending to ESPN and waiting for its confirmation — this can take a few seconds. Do not close this tab.'
              : 'Sends the pick straight to ESPN and waits for its confirmation. Nothing here is final until ESPN answers.'}
          </p>
        )}
        <div className="confirm-actions">
          <button type="button" className="confirm-btn-cancel" disabled={sending} onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="confirm-btn-confirm" disabled={sending} onClick={onConfirm}>
            {sending ? 'Sending…' : status === 'failed' ? 'Try again' : 'Draft him'}
          </button>
        </div>
      </div>
    </div>
  )
}
