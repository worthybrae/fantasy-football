import { useEffect } from 'react'
import PlayerProfile, { type ProfileSeed } from '../PlayerProfile'

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
}

// The in-draft player profile: over the board, never instead of it.
//
// Two rules this exists to keep, both of them about a 30-second clock on an
// unrecallable pick:
//
// 1. It never leaves the room. Clicking a player used to navigate to
//    /players/:slug, which unmounts the whole draft room -- board, tab,
//    scroll position, the 2.5s poll -- and then waits on a 3.5s request
//    while the clock runs. That route still exists and still works for a
//    direct link (App.tsx, PlayerPage.tsx); this is the in-draft path, not
//    a replacement for it. Because the room stays mounted, closing this
//    restores nothing: the board behind was never taken down, so it comes
//    back on the same tab, the same scroll offset, the same filter and the
//    same selection by construction rather than by re-derivation.
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
export default function PlayerOverlay({ target, onClose, onSelectPlayer }: PlayerOverlayProps) {
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
          embedded
          onClose={onClose}
          onSelectPlayer={onSelectPlayer}
        />
      </div>
    </div>
  )
}
