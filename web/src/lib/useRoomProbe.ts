import { useEffect } from 'react'
import { fetchLiveState } from '../api'

// IS THERE A ROOM TO WALK INTO?
//
// The landing page asks this because a connect can be running in another
// window: somebody who started one there should be offered the board here
// rather than pitched a tool they are already using.
//
// IT USED TO ASK EVERY 2.5 SECONDS, FOREVER. On a signed-out visitor -- who
// has no session at all and therefore cannot have a room until they make one
// -- that is 1,440 requests an hour to be told "no" 1,440 times, on a page
// whose whole job is a first impression. The cadence now follows the answer:
// slow until something is actually happening, quick once it is.
//
// "Something is happening" is `active` on the state payload: a session exists
// in the server process, which means a connect is under way or a room is up.
// That is deliberately WEAKER than the test for offering the board (below) --
// a session that was minted and abandoned reads as active for as long as the
// process lives, so it is a reason to look more often, never a reason to send
// somebody to a board with no draft behind it.

/** While nothing is happening. Long enough that a signed-out visitor's page
 *  is quiet, short enough that a connect started in another window is picked
 *  up before they wonder why. */
export const ROOM_PROBE_IDLE_MS = 30_000
/** Once a session exists, or a connect is running on this page. */
export const ROOM_PROBE_BUSY_MS = 2_500

export function useRoomProbe({ enabled, busy, onLive }: {
  /** False once the page has moved on -- there is nothing left to watch for. */
  enabled: boolean
  /** The page already knows a connect is running: poll quickly from the
   *  first tick rather than waiting to discover it. */
  busy: boolean
  /** A room that is genuinely alive. Must be stable, or the probe restarts
   *  on every render of the page. */
  onLive: () => void
}): void {
  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    let timer = 0
    // Local, not state: the cadence changing must not re-render the page or
    // restart this effect -- it only decides when the next timer fires.
    let quick = busy

    const poll = async () => {
      try {
        const state = await fetchLiveState()
        if (cancelled) return
        if (state.active) quick = true
        // A SESSION IS NOT A DRAFT. `active` and `token_received` only say a
        // connect once happened in this server process: a token minted, used
        // for nothing and left behind reads as both for as long as the
        // process lives. Measured on this deployment -- active,
        // token_received, listener_alive false, stale, zero picks -- and the
        // page was offering "your draft is synced, go to the board" over a
        // session with no draft behind it. So the test is whether the thing
        // is ALIVE: a listener on the socket, or picks that actually landed.
        if (state.active
            && (state.listener_alive || state.draft_started || state.picks_made > 0)) {
          onLive()
          return
        }
      } catch {
        /* helper not up yet; the page simply stays on the landing view */
      }
      if (!cancelled) {
        timer = window.setTimeout(
          () => { void poll() }, quick ? ROOM_PROBE_BUSY_MS : ROOM_PROBE_IDLE_MS)
      }
    }

    void poll()
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [enabled, busy, onLive])
}
