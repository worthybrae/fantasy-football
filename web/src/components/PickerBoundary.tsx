import { Component, useEffect, type ErrorInfo, type ReactNode } from 'react'

// THE TWO STATES A LAZY DIALOG HAS THAT AN EAGER ONE DOES NOT: it is still
// arriving, and it never arrived.
//
// Both are real. The picker is a `React.lazy` chunk, and a chunk request is
// the one part of this page that can fail on its own -- most often minutes
// after a deploy, when the browser is holding an index that names a file the
// server has already replaced. Unhandled, that error unmounts everything
// above it: React's default for an uncaught render error is to blank the
// tree, so a 404 on a 5 KB chunk would take the whole dashboard with it.
//
// Neither state may trap anybody either. The scrim is over the page, so both
// take the same two exits as the dialog they stand in for -- Escape, and a
// click on the backdrop.

/** Escape, on whichever of these is on screen. Not a `keydown` on the div:
 *  nothing here is focused, so the listener has to be the document's. */
function useEscape(onClose: () => void): void {
  useEffect(() => {
    function onKey(e: KeyboardEvent): void {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])
}

/** While the chunk is in the air. Deliberately just the dim: it is a small
 *  local file, and a spinner that flashes for one frame is worse than a
 *  moment of shade. */
export function PickerFallback({ onClose }: { onClose: () => void }) {
  useEscape(onClose)
  return (
    <div
      className="fav-modal-backdrop"
      aria-busy="true"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
    />
  )
}

/** When it did not arrive. Says the one thing that fixes it -- a reload gets
 *  the current index and therefore the current chunk name -- and offers the
 *  way out rather than leaving a dim page nobody can dismiss. */
function PickerFailed({ onClose }: { onClose: () => void }) {
  useEscape(onClose)
  return (
    <div
      className="fav-modal-backdrop"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="fav-modal" role="alertdialog" aria-label="The picker did not load">
        <button type="button" className="fav-modal-close" onClick={onClose}
                aria-label="Close">
          ×
        </button>
        <p className="fav-error" role="alert">
          The picker didn&rsquo;t load — reload the page and try again.
        </p>
      </div>
    </div>
  )
}

// A class, because catching a render error is the one thing hooks cannot do.
export class PickerBoundary extends Component<
  { onClose: () => void; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false }

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Logged rather than swallowed: a chunk that stops loading after a deploy
    // is worth seeing in a console session, and this is the only place that
    // ever sees it.
    console.error('The favourites picker failed to load', error, info.componentStack)
  }

  render(): ReactNode {
    if (this.state.failed) return <PickerFailed onClose={this.props.onClose} />
    return this.props.children
  }
}
