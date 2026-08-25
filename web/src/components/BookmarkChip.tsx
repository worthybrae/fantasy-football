import { useCallback, useEffect, useState } from 'react'
import { bookmarkletFor } from '../lib/bookmarklet'

// The bookmarklet chip and everything that decides how it is offered: the
// drag-capability test, the raw-HTML anchor, the copy fallback and the line
// shown to a browser that cannot drag. Moved out of pages/Landing.tsx
// verbatim when the setup wizard became a second consumer -- two copies of
// the anchor builder would be two chances for the "no single quotes" rule
// below to be broken in only one of them.

// ---------------------------------------------------------------------------
// Can this browser actually drag a link onto the bookmarks bar?
// ---------------------------------------------------------------------------
//
// The hero's primary action is a drag target, and a drag target on a phone
// is a button that does nothing: there is no bookmarks bar to drop it on and
// no gesture that would reach one. So the page has to know, and it has to
// know before first paint (a hero that swaps its main action a moment after
// it renders is a layout shift on the most important block of the page).
//
// DELIBERATELY NOT A WIDTH TEST. Width answers a different question than the
// one being asked. A desktop window dragged down to 700px still drags
// perfectly; a 1366px-wide tablet cannot drag at all. Two capability facts
// answer the real one:
//
//   'draggable' in a <div>  -- the HTML drag-and-drop API exists here.
//   (hover: hover) and (pointer: fine) -- the browser's PRIMARY input is a
//   precise pointer that can hover, which is what dragging something out of
//   the page and onto the browser's own chrome requires. iPadOS reports
//   (hover: none) and (pointer: coarse) no matter how wide the window is; a
//   laptop reports fine/hover at 320px.
//
// A hybrid -- a touchscreen laptop, a tablet with a trackpad attached --
// reports its primary input as fine/hover and gets the drag target, which is
// the right answer: it has a mouse. Detaching that trackpad fires the change
// event below and the hero switches over.
//
// When there is no matchMedia at all, the answer is "no". The fallback is a
// working page with an honest sentence; the drag target is the branch that
// breaks if it is shown to the wrong browser, so it is the branch that has
// to be earned.
const DRAG_QUERY = '(hover: hover) and (pointer: fine)'

function canDragNow(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return false
  if (!('draggable' in document.createElement('div'))) return false
  return window.matchMedia(DRAG_QUERY).matches
}

export function useCanDrag(): boolean {
  // Computed in the initialiser, not in an effect, so the first paint is
  // already the right hero.
  const [canDrag, setCanDrag] = useState(canDragNow)
  useEffect(() => {
    if (!window.matchMedia) return
    const mq = window.matchMedia(DRAG_QUERY)
    const sync = () => setCanDrag(canDragNow())
    mq.addEventListener('change', sync)
    return () => mq.removeEventListener('change', sync)
  }, [])
  return canDrag
}

// The bookmarklet as a real `javascript:` anchor.
//
// Present from first paint so a drag to the bookmarks bar copies IT, not
// this page's URL -- setting the href after mount was too late, and the drag
// grabbed localhost instead. React strips `javascript:` from href props, so
// it is injected as raw HTML. The string has no single quotes (verified in
// bookmarklet.ts, which also explains why it is comment-free and must not be
// reformatted), so a single-quoted href is safe; onclick returns false so a
// stray click does nothing -- it is a drag target, not a button.
//
// The football rides in the anchor's TEXT, as an emoji rather than the SVG
// mark: a bookmark's title is the anchor's textContent, so an inline <svg>
// would be dropped on the way to the bar and the label would arrive naked.
// Nothing else may be added inside the anchor for the same reason.
//
// `describedBy` points at the copy hint beside it. The chip is the hero's
// first tab stop and, by design, does nothing on Enter -- a click would run
// the bookmarklet against this page rather than a draft. A keyboard reader
// who is told only "Draft Assistant" and then gets silence has hit a dead
// end; described by the hint, they are told the same thing the sighted
// reader is, that the route for them is the copy button next along.
//
// An attribute, not content: `textContent` is what becomes the bookmark's
// title, so this cannot disturb it the way an inner element would.
export function bookmarkAnchor(describedBy?: string): string {
  const described = describedBy ? "aria-describedby='" + describedBy + "' " : ""
  return "<a class='lp-bookmark' title='Drag me to your bookmarks bar' "
    + described
    + "onclick='return false' href='" + bookmarkletFor(window.location.origin) + "'>"
    + "<span aria-hidden='true'>🏈</span>&nbsp;ESPN&nbsp;Draft&nbsp;Assist</a>"
}

// ---------------------------------------------------------------------------
// The keyboard path for the chip above: not dragging it, copying it.
// ---------------------------------------------------------------------------
//
// The anchor bookmarkAnchor() renders is a drag target and, deliberately,
// nothing else -- its onclick returns false so neither a stray click nor an
// Enter/Space on a focused chip does anything (see the comment on
// bookmarkAnchor). That is correct for a mouse, where the chip is never
// clicked, only dragged. It is a dead end for a keyboard: the chip is the
// hero's first tab stop, Enter does nothing, and nothing on the page says
// another route exists. Dragging is inherently mouse-only and that part
// cannot be fixed -- but the dead end is not inherent, so this is the
// button a keyboard user reaches instead. It copies the exact same
// `javascript:` string, verbatim, so it can be pasted as a hand-made
// bookmark's address.
export async function copyBookmarklet(): Promise<boolean> {
  // The Clipboard API needs a secure context (https, or localhost -- both
  // covered here) and can still reject (permission denied, an iframe
  // without the right allow policy). Either way, fall through to the
  // legacy path rather than surface that as the only route.
  if (typeof navigator !== 'undefined' && navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(bookmarkletFor(window.location.origin))
      return true
    } catch {
      /* fall through */
    }
  }
  // document.execCommand('copy') needs no permission and no secure context,
  // which is what makes it worth keeping as a fallback rather than just
  // reporting failure. An off-screen, unfocusable-by-tab textarea holds the
  // text just long enough to select and copy it, then is removed.
  try {
    const ta = document.createElement('textarea')
    ta.value = bookmarkletFor(window.location.origin)
    ta.setAttribute('readonly', '')
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    ta.style.left = '-9999px'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}

/** Which modifier family this browser's shortcuts use. A wrong guess costs
 *  nothing but a wrong-looking keycap, so a userAgent sniff is fine. */
export function isMac(): boolean {
  return typeof navigator !== 'undefined' && /Mac/.test(navigator.userAgent)
}

/** A keyboard shortcut as one keycap per key, not a run-together string:
 *  "⌘⇧B" reads as a glyph soup, ⌘ ⇧ B reads as three keys to press. The
 *  wrapping span keeps the caps on one line and lets them share a gap. */
export function Keys({ keys }: { keys: string[] }) {
  return (
    <span className="kb-keys">
      {keys.map((k) => <kbd key={k}>{k}</kbd>)}
    </span>
  )
}

/** The show-the-bookmarks-bar shortcut for this platform. Words, not the
 *  ⌘/⇧ glyphs: "Cmd" and "Shift" are readable by everyone, while the Mac
 *  symbols are only obvious to people who already know them. */
export function barKeys(): string[] {
  return isMac() ? ['Cmd', 'Shift', 'B'] : ['Ctrl', 'Shift', 'B']
}

// Rendered beside the chip everywhere it appears (the hero, the setup block,
// and the wizard's install step). `idPrefix` keeps the instances' ids from
// colliding when more than one is on the page at once. The result is
// announced through role="status"/aria-live so a screen reader hears it
// without focus ever leaving the button -- the button's own label never
// changes.
export function BookmarkCopy({ idPrefix }: { idPrefix: string }) {
  const [status, setStatus] = useState<'idle' | 'ok' | 'fail'>('idle')
  const onCopy = useCallback(() => {
    copyBookmarklet().then((ok) => setStatus(ok ? 'ok' : 'fail'))
  }, [])
  // The hint tells the user to bookmark THIS page first and then swap the
  // address, not to create a blank bookmark. Same number of steps, one
  // difference: a bookmark made from a real page keeps that page's favicon
  // in Chromium (and usually Firefox) after the address is edited, so the
  // hand-made bookmark gets the football icon instead of the blank globe. A
  // `javascript:` URL can never earn an icon of its own; inheriting one is
  // the only way it gets one.
  const bookmarkKeys = isMac() ? ['Cmd', 'D'] : ['Ctrl', 'D']
  // The copy action is a link INSIDE the sentence, not a button above it: it
  // is the first step of the fallback the sentence describes, and a grey
  // button floating over its own instructions read as a separate control.
  return (
    <div className="lp-copy">
      <p id={`${idPrefix}-copy-hint`} className="lp-copy-hint">
        Can’t drag it?{' '}
        <button type="button" className="lp-copy-link" onClick={onCopy}>
          Copy the link
        </button>
        . Press <Keys keys={bookmarkKeys} /> to bookmark this page. Paste the
        link as that bookmark’s address. The bookmark keeps the football icon.
      </p>
      <span
        className={
          status === 'idle' ? 'lp-copy-status'
            : status === 'ok' ? 'lp-copy-status is-ok' : 'lp-copy-status is-fail'
        }
        role="status"
        aria-live="polite"
      >
        {status === 'ok' && 'Copied. Paste it as the bookmark’s address.'}
        {status === 'fail' && 'Copy failed. Drag the chip instead.'}
      </span>
    </div>
  )
}

// The one sentence a visitor who cannot drag gets instead of the chip. Said
// once, in the hero, and again above the setup block's own drag target so
// that scrolling down does not land them back on the thing they were just
// told they cannot use.
export const NO_DRAG_LINE = (
  <>
    ESPN Draft Assist is a <strong>bookmark that you drag to your browser’s
    bookmarks bar</strong>. Setup needs a desktop browser.
  </>
)
