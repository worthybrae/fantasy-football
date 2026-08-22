import { useEffect } from 'react'
import type { LiveSettings } from '../../api'
import PlayerProfile, { type ProfileSeed } from '../PlayerProfile'
import type { RankedPlayer } from '../profile/payload'

// What the room hands over to open a profile: the id to fetch, and
// everything it already knew about that player so the overlay can paint
// before the fetch is even issued (see ProfileSeed, and playerSeed.ts for
// the three builders).
export interface OverlayTarget {
  playerId: string
  seed: ProfileSeed
}

interface PlayerOverlayProps {
  target: OverlayTarget
  onClose: () => void
  // A comp clicked inside the profile: the room resolves the id against its
  // join table and swaps this overlay's own target, rather than routing.
  // Nothing here navigates.
  onSelectPlayer: (id: string) => void
  // Whose pick it is, decided by the room (DraftRoom's `youAreUp`) off the
  // 2.5s /api/live/state poll it already runs. A prop rather than a fetch of
  // this overlay's own: a second poller for one boolean is a second thing
  // that can disagree with the clock panel three inches to the left, and the
  // profile is opened over a running pick clock -- the last place to spend a
  // request. Defaulted so a caller that has no live session to speak of (or
  // has not been updated) still type-checks, the same defaulting precedent
  // ClockPanel's `board` and `onSetAutodraft` set.
  onTheClock?: boolean
  // Straight through to the profile, which grades a season's finish against
  // how many of a position start in THIS league (see PlayerProfile's own
  // `settings` comment). Passed rather than fetched for the same reason
  // `onTheClock` is: the room already holds it.
  settings?: LiveSettings | null
  // The room's ranked board, threaded exactly as `settings` is and for the
  // same reason: the room has it, the popup would otherwise fetch it under a
  // pick clock. The profile's "Near you" card is a run of picks around his
  // own, and the payload can only name one for a player it has no stat line
  // to match (see PlayerProfile's `ranked`).
  ranked?: RankedPlayer[]
  // Takes the player whose profile this is. The room decides whether there
  // is a pick to make at all and passes nothing when there is not -- see
  // DraftRoom, where this is wired to the same confirm dialog the board's
  // own Draft buttons open.
  onDraftPlayer?: (playerId: string) => void
}

// The in-draft player profile: over the board, never instead of it.
//
// Two rules this exists to keep, both of them about a 30-second clock on an
// unrecallable pick:
//
// 1. It never leaves the room. Clicking a player used to navigate to
//    /players/:slug, which unmounts the whole draft room -- board, tab,
//    scroll position, the 2.5s poll -- and then waits on a 3.5s request
//    while the clock runs. That route is gone now (App.tsx); this popup is
//    the only way a player profile opens, in or out of a draft. Because the
//    room stays mounted, closing this restores nothing: the board behind was
//    never taken down, so it comes back on the same tab, the same scroll
//    offset, the same filter and the same selection by construction rather
//    than by re-derivation.
//
// 2. It never moves the room. `position: fixed` over the page, no scroll
//    lock (locking body overflow would remove a scrollbar and reflow the
//    exact board this is drawn over -- and the room's own scroll container
//    is `.draft-main`, not the document, so there is nothing to lock
//    anyway).
//
// Escape handling, backdrop dismissal, `role="dialog" aria-modal`, and the
// z-index tier are ConfirmPick.tsx's, deliberately: one modal pattern in
// this room, not two. Like ConfirmPick this does not trap focus -- see the
// task report; matching the established pattern beat inventing a second one
// here.
export default function PlayerOverlay({
  target, onClose, onSelectPlayer, onTheClock = false, settings = null,
  ranked = [], onDraftPlayer,
}: PlayerOverlayProps) {
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  return (
    <div
      className="player-overlay"
      // mousedown, not click: a click fires on the element the press and the
      // release share, so a text selection that starts inside the panel and
      // ends over the dim would otherwise close the profile mid-drag. The
      // target check keeps this to the backdrop itself -- events bubbling up
      // from the panel are not a backdrop press.
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className="player-overlay-panel"
        role="dialog"
        aria-modal="true"
        aria-label={`${target.seed.name} — player profile`}
      >
        <button
          type="button"
          className="player-overlay-close"
          onClick={onClose}
          aria-label="Close player profile"
        >
          ✕
        </button>
        {onTheClock && (
          // Your pick, and you are reading a dossier with the board behind
          // it. The popup can take him now (the footer button, see
          // `onDraftPlayer`), so this is no longer a warning that you are in
          // the wrong place -- it is the clock itself, which nothing else in
          // here shows and which is running whether you are reading or not.
          //
          // Inside the panel rather than beside it: the panel is the
          // `aria-modal` dialog, and assistive tech ignores everything
          // outside an open modal -- an alert about a clock running out is
          // exactly the one a person who is not looking at the screen has to
          // hear. It is still pinned to the top centre of the viewport, not
          // to the panel: `position: fixed` resolves against `.player-overlay`
          // (its `backdrop-filter` makes it the containing block), which is
          // `inset: 0` -- the same rectangle the viewport is either way, so
          // it neither scrolls with the profile nor gets clipped by it.
          //
          // Not dismissible, on purpose. It clears itself the moment either
          // half of its own condition stops being true -- the pick lands, or
          // you close the profile -- and a dismiss control on a thirty-second
          // warning is a way to turn it off and then forget.
          <div className="player-overlay-clock" role="alert">
            <strong>You are on the clock.</strong>
            {/* Only against a footer that drafts. It is here because this
                player is not always the one you want and the board is where
                the other two hundred are -- but with no pick to make for him
                the footer IS "Back to the board" (see PlayerProfile), and
                two identical buttons on one popup make the reader choose
                between them for no reason. The footer owns that way out: it
                is drawn in every state, including the ones where no clock is
                running and this alert is not on screen at all. */}
            {onDraftPlayer && (
              <button
                type="button"
                className="player-overlay-clock-back"
                onClick={onClose}
              >
                Back to the board
              </button>
            )}
          </div>
        )}
        {/* Keyed on the player: a comp click swaps the target, and the key
            forces a fresh mount so the new seed paints instantly instead of
            the previous player's header sitting there while
            PlayerProfile's own playerId effect refetches. Its stale-response
            guard still does its job either way -- this just means the
            header never lies for those few hundred milliseconds. */}
        <PlayerProfile
          key={target.playerId}
          playerId={target.playerId}
          seed={target.seed}
          settings={settings}
          ranked={ranked}
          onDraftPlayer={onDraftPlayer}
          embedded
          onClose={onClose}
          onSelectPlayer={onSelectPlayer}
        />
      </div>
    </div>
  )
}
