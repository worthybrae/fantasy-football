import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { TIP_DELAY_MS, positionTip } from '../draft/AvailableList'

// WHAT THIS CARD IS, on hover of its own title.
//
// Fourteen cards open with this popup and every one of them is a different
// kind of measurement: a count of games, a place among a position, a rank
// among 32 offences, a share of an offence, a spread of names. The titles
// are one word each because a card three inches wide has room for one word,
// and a reader who has not been told what "Steady" counts cannot get it from
// the word.
//
// The mechanism is the available table's header tooltip, on purpose (see
// `sortableTh` in AvailableList.tsx for the full argument against the native
// `title` this would otherwise be): a dotted underline that says an
// explanation exists before anyone hovers, ~130ms of delay so sweeping the
// pointer across a row of cards does not strobe, no delay at all on keyboard
// focus, and one floating panel positioned off the trigger's own rect.
//
// `position: fixed`, which is what lets the panel out of the popup's own
// scroller. Inside the draft overlay the nearest containing block is
// `.player-overlay` -- its `backdrop-filter` makes it one -- and that box is
// `inset: 0`, so fixed coordinates are still viewport coordinates and
// `positionTip` needs no special case. The clipping that matters is the
// card's own `overflow: hidden` on the title, which a fixed child escapes
// because its containing block is further up.
export default function CardHint({ text, children }: {
  /** The explanation. One or two sentences, in the room's own voice. */
  text: string
  /** The title being explained, which becomes the trigger. */
  children: ReactNode
}) {
  // The trigger's rect at the moment it was hovered, which is both the "is
  // it open" flag and the input to `positionTip`.
  const [rect, setRect] = useState<DOMRect | null>(null)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)
  const timer = useRef<number | null>(null)
  const tip = useRef<HTMLDivElement | null>(null)
  const id = useId()

  function clearTimer(): void {
    if (timer.current !== null) {
      window.clearTimeout(timer.current)
      timer.current = null
    }
  }
  function schedule(el: HTMLElement): void {
    clearTimer()
    const at = el.getBoundingClientRect()
    timer.current = window.setTimeout(() => setRect(at), TIP_DELAY_MS)
  }
  // Keyboard focus skips the delay: a Tab press is already one deliberate
  // move, and making its answer wait reads as lag.
  function showNow(el: HTMLElement): void {
    clearTimer()
    setRect(el.getBoundingClientRect())
  }
  function hide(): void {
    clearTimer()
    setRect(null)
  }

  // A card can be unmounted by the pointer that opened it: hovering a comp
  // swaps the whole profile, and a timer left running would call setState on
  // a component that is gone.
  useEffect(() => clearTimer, [])

  // Measured, not guessed, and before the browser paints: this copy runs
  // from one line to five, and a single assumed height would either clip the
  // long ones or float the short ones. The first render is invisible at
  // (0, 0); this effect gives it its real place inside the same paint.
  useLayoutEffect(() => {
    if (!rect || !tip.current) {
      setPos(null)
      return
    }
    const { width, height } = tip.current.getBoundingClientRect()
    setPos(positionTip(rect, { width, height }))
  }, [rect])

  // Any scroll invalidates the captured rect -- the popup has its own
  // scroller and the page has another -- so the panel is dropped rather than
  // left hanging over whatever moved under it.
  useEffect(() => {
    if (!rect) return
    window.addEventListener('scroll', hide, true)
    return () => window.removeEventListener('scroll', hide, true)
  }, [rect])

  return (
    <>
      {/* Focusable, because a tooltip only a mouse can open is a tooltip half
          the room cannot read. A span rather than a button: there is nothing
          to press here, and a button would promise an action. */}
      <span
        className="pp-pop-hint"
        tabIndex={0}
        aria-describedby={rect ? id : undefined}
        onMouseEnter={(e) => schedule(e.currentTarget)}
        onMouseLeave={hide}
        onFocus={(e) => showNow(e.currentTarget)}
        onBlur={hide}
        onKeyDown={(e) => { if (e.key === 'Escape') hide() }}
      >
        {children}
      </span>
      {rect && (
        <div
          id={id}
          ref={tip}
          role="tooltip"
          className="pp-pop-tip"
          style={pos
            ? { left: pos.left, top: pos.top, visibility: 'visible' }
            : { left: 0, top: 0, visibility: 'hidden' }}
        >
          {text}
        </div>
      )}
    </>
  )
}
