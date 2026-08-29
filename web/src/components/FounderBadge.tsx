import { useEffect, useState } from 'react'

// THE FIRST HUNDRED PEOPLE, COUNTED.
//
// The paywall is off and the first hundred accounts to connect draft free for
// good (spec section 3). That is the only reason a stranger has to act today
// rather than in September, so it is on the page twice: as a count of what is
// left, over the landing page's call to action, and as a membership number in
// the dashboard's own bar once somebody has claimed one.
//
// THIS FILE IS A PLACEHOLDER AND IS MEANT TO BE REPLACED. The founders work
// (`api/account.py`'s `/api/account/me`, the `founder` table, the free grant
// in `require_paid`) is on its own branch, and this component is the shape the
// front door needs from it. It is written to the payload that branch will
// serve and to survive its absence: a 404, a 500 or a network failure all draw
// nothing at all, which is the right answer on a deployment where founders do
// not exist yet. When the real component lands at this path, git will say so
// -- take theirs, and check only that these two call sites still get a badge
// for a signed-out reader and one for a founder.

/** What the founders endpoint answers. Every field optional in practice: this
 *  reads a route that may not be deployed, so nothing here may be assumed. */
export interface FounderState {
  /** Whether THIS account is one of the first hundred. */
  founder: boolean
  /** Which one, 1-based. Null for a reader who is not one. */
  ordinal: number | null
  /** How many of the hundred are unclaimed. */
  founders_left: number
  /** Whether this browser holds an ESPN session at all. */
  connected: boolean
}

/** Null while the read is in flight, and null forever if it fails.
 *
 *  Failure is silent by design. This is a badge, not a feature: a deployment
 *  whose account routes predate the founders work should show one fewer thing
 *  on the page, not an error where a reassurance was meant to go.
 */
export function useFounderState(): FounderState | null {
  const [state, setState] = useState<FounderState | null>(null)
  useEffect(() => {
    let cancelled = false
    fetch('/api/account/me')
      .then((res) => (res.ok ? res.json() : null))
      .then((body) => {
        if (cancelled || !body) return
        setState({
          founder: Boolean(body.founder),
          ordinal: typeof body.ordinal === 'number' ? body.ordinal : null,
          founders_left: Number(body.founders_left ?? 0),
          connected: Boolean(body.connected),
        })
      })
      .catch(() => { /* no badge, then */ })
    return () => { cancelled = true }
  }, [])
  return state
}

export default function FounderBadge({ variant = 'open' }: {
  /** `open` counts what is left, for a reader who has not connected. `member`
   *  states the number somebody already holds. Each draws nothing when it has
   *  nothing true to say, so a page can mount both and let the answer decide
   *  which one appears. */
  variant?: 'open' | 'member'
}) {
  const state = useFounderState()
  if (state === null) return null

  if (variant === 'member') {
    if (!state.founder || state.ordinal === null) return null
    return (
      <span className="fb fb-member">
        <span className="fb-mark mono">#{state.ordinal}</span>
        Founding member — every draft free
      </span>
    )
  }

  // Nothing left to claim, or this reader has already claimed one: either way
  // a countdown of spots is not the sentence they need.
  if (state.founder || state.founders_left <= 0) return null
  return (
    <span className="fb fb-open">
      <span className="fb-mark mono">{state.founders_left}</span>
      founder {state.founders_left === 1 ? 'spot' : 'spots'} left
    </span>
  )
}
