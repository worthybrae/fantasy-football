import { useSyncExternalStore } from 'react'

// ONE INTERVAL FOR THE WHOLE TAB, AND ONE RE-RENDER PER SUBSCRIBER.
//
// Every clock in this product used to be a `setNow` in a page component: the
// dashboard held one, and a tick re-rendered the entire signed-in page --
// every league card, the mock lobby's twenty cards and their seat pips, and
// (while the favourites picker was inline) 252 player rows with 252 photos.
// Measured in jsdom, that was 19ms of React work every second with nothing
// happening on the page at all, which is what "I was scrolling and the screen
// wasn't moving" is made of: the main thread was busy redrawing a page whose
// only change was one line of digits.
//
// So the clock is a store, not a page state. A component that shows time
// subscribes to it and re-renders alone; everything around it is untouched,
// because nothing around it ever learns that a second passed.
//
// `useSyncExternalStore` rather than a context + `useState`: a context value
// that changes every second re-renders every consumer's whole subtree, which
// is the thing being fixed. This re-renders exactly the components that read
// the store.

const listeners = new Set<() => void>()
// THE SNAPSHOT MUST BE STABLE BETWEEN TICKS. `getSnapshot` returning
// `Date.now()` would hand React a new value on every call and it would
// re-render forever looking for a fixed point. This is set once per tick and
// read many times.
let now = Date.now()
let timer: ReturnType<typeof setInterval> | null = null

function tick(): void {
  now = Date.now()
  for (const listener of listeners) listener()
}

// The interval exists only while something is watching it -- a dashboard with
// no countdown on screen (every draft in the past, say) pays nothing, and a
// page that navigates away stops the timer rather than leaving it running
// against unmounted components.
function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  if (timer === null) {
    now = Date.now()
    timer = setInterval(tick, 1000)
  }
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0 && timer !== null) {
      clearInterval(timer)
      timer = null
    }
  }
}

function snapshot(): number {
  return now
}

/** The current second, shared. Every caller gets the SAME number until the
 *  next tick, which is also why two countdowns on one page can never disagree
 *  about what time it is by a few hundred milliseconds. */
export function useSecond(): number {
  return useSyncExternalStore(subscribe, snapshot, snapshot)
}
