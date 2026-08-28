import { useEffect, useRef, useState } from 'react'
import type { Player } from '../api'
import { loadBoard } from '../lib/board'
import FavoritesPicker from './FavoritesPicker'

// YOUR GUYS, IN A DIALOG OVER THE PAGE.
//
// This used to be a panel sitting open on the dashboard, which cost the page
// two things it should never have paid: `/api/players` (250 rows, ~185 KB) on
// every visit whether or not anybody was picking, and 252 rows with 252
// photographs in the tree of a page that re-rendered on a clock. Both are
// gone: the board is fetched the first time somebody opens this, and the rows
// exist only while it is open.
//
// The module is loaded with `React.lazy` from the dashboard, so none of this
// code -- nor the picker's -- is in the bundle a reader downloads to look at
// their leagues. The board itself is cached for the life of the page (see
// lib/board.ts), so opening this a second time costs nothing.

// Everything a keyboard can land on inside the dialog. `:not([disabled])`
// matters here: Save is disabled until the list is savable, and a trap that
// counted it would park focus on a dead control.
const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), '
  + 'select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

export default function FavoritesModal({ initial, onSaved, onClose }: {
  /** The saved list, so the dialog opens on what is already chosen. */
  initial: string[]
  /** The saved order, once the server has taken it. */
  onSaved: (players: string[]) => void
  onClose: () => void
}) {
  const [board, setBoard] = useState<Player[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const dialog = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    loadBoard()
      .then((rows) => { if (!cancelled) setBoard(rows) })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      })
    return () => { cancelled = true }
  }, [])

  // ESCAPE, AND TAB THAT CANNOT LEAVE. A dialog a screen reader can walk out
  // of the back of is a dialog that is modal only to people who can see it.
  // The listener is on the document rather than the panel so Escape works
  // before anything inside has been focused.
  useEffect(() => {
    const node = dialog.current
    node?.focus()
    function onKey(e: KeyboardEvent): void {
      if (e.key === 'Escape') {
        onClose()
        return
      }
      if (e.key !== 'Tab' || node === null) return
      const stops = Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE))
      if (stops.length === 0) return
      const first = stops[0]
      const last = stops[stops.length - 1]
      const here = document.activeElement
      // Backwards off the front, or forwards off the back: the two ways out,
      // and the only two this has to close.
      if (e.shiftKey && (here === first || here === node)) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && here === last) {
        e.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  // FOCUS GOES BACK WHERE IT CAME FROM. The dialog takes focus when it
  // opens, which is right; what was wrong is where focus landed afterwards.
  // Closing left it on `document.body`, so the next Tab started at the top of
  // the page and a reader who opened the picker from the "Edit" button had to
  // walk the whole dashboard to find that button again -- and a screen reader
  // was told nothing at all about where it now was.
  //
  // Captured on the way in rather than passed down: the control that opened
  // this is whatever had focus, and this component does not need to know
  // which one that was. Guarded on `isConnected` because the button may be
  // gone by the time this unmounts (the card redraws when the list is
  // saved), and focusing a detached node silently sends focus to the body,
  // which is the state this exists to avoid.
  //
  // CAPTURED DURING THE FIRST RENDER, not in an effect: the effect that traps
  // focus runs before any effect declared after it and moves focus to the
  // panel, so an effect here would faithfully capture the dialog itself and
  // "restore" focus to a node that no longer exists.
  const opener = useRef<HTMLElement | null>(null)
  if (opener.current === null) {
    opener.current = document.activeElement as HTMLElement | null
  }
  useEffect(() => () => {
    const back = opener.current
    if (back && back.isConnected && typeof back.focus === 'function') back.focus()
  }, [])

  // The page underneath does not scroll while this is open. Restored to
  // whatever it was rather than to `''`: this page is not the only thing that
  // can lock the body, and putting back a value we did not take is how two
  // overlays leave a page permanently unscrollable.
  useEffect(() => {
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = previous }
  }, [])

  return (
    // `onMouseDown`, not `onClick`: a click whose press started INSIDE the
    // panel and finished on the backdrop is somebody dragging a selection
    // across a player's name, and closing on it would throw away a list they
    // are still building.
    <div
      className="fav-modal-backdrop"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        className="fav-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Your guys"
        ref={dialog}
        tabIndex={-1}
      >
        <button type="button" className="fav-modal-close" onClick={onClose}
                aria-label="Close">
          ×
        </button>
        {board !== null ? (
          <FavoritesPicker
            players={board}
            initial={initial}
            onSaved={onSaved}
            onCancel={onClose}
          />
        ) : error !== null ? (
          <p className="fav-error" role="alert">
            The player list did not load ({error}), so there is nothing to pick
            from. Close this and try again.
          </p>
        ) : (
          <p className="fav-modal-loading">Loading the board…</p>
        )}
      </div>
    </div>
  )
}
