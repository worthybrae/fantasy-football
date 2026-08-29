import { useEffect, useState } from 'react'
import { fetchMe, type Me } from '../api'

// WHAT THIS SAYS AND WHO TO. The first hundred accounts to connect draft free
// forever (api/billing.py). That is two sentences, not one, and which of them
// is true of a reader is the whole logic here:
//
//   * a founder is told which one they are. "#37" is not decoration -- it is
//     the ordinal in the table, it never changes, and it is the difference
//     between a promise and a marketing line;
//   * a reader who has connected nothing is told how many seats are left and
//     what to do about it. This is an OFFER, and it is the only reason
//     `GET /api/account/me` answers a signed-out browser at all;
//   * everybody else is told nothing. Somebody already connected who is not a
//     founder cannot claim one -- the request that would have claimed it is
//     the one that just answered -- so "connect ESPN to claim one" would be
//     an instruction they have already followed.
//
// IT RENDERS NOTHING UNTIL IT KNOWS, and nothing at all unless a well-formed
// 200 came back. The badge is a claim about what somebody has been given; a
// placeholder that resolves into "0 spots left" is worse than a line that was
// never there, a store that is briefly away is not a reason to break the page
// around it, and a deployment whose account route predates the founders work
// should show one fewer thing rather than an error where a reassurance goes.
//
// WHERE IT GOES. Two places, and they want different halves of the logic:
// the landing page's call to action (`variant="open"`, the default, which is
// the offer -- and the membership line for a founder who is reading it signed
// in) and the dashboard's top bar (`variant="member"`, which is only ever the
// membership line: a reader looking at their own drafts has connected, so the
// offer cannot apply to them). It draws one line and takes its styling from
// `.fb` in landing.css, which the two pages size for their own row.

/** A 200 is not enough on its own: this reads a route that other deployments
 *  answer differently, and half a payload would print half a promise. */
function wellFormed(body: unknown): body is Me {
  if (!body || typeof body !== 'object') return false
  const me = body as Record<string, unknown>
  return typeof me.founder === 'boolean'
    && typeof me.connected === 'boolean'
    && typeof me.founders_left === 'number'
    && (me.ordinal === null || typeof me.ordinal === 'number')
}

export default function FounderBadge({ me: given, variant = 'open' }: {
  /** An answer the page already has, for a caller that reads `/api/account/me`
   *  for its own reasons. Absent, the badge asks for itself -- the request is
   *  shared for five seconds either way, so two of these on one page is one
   *  round trip. */
  me?: Me | null
  /** `open` is the whole offer, for a page a stranger can reach. `member` is
   *  the membership line alone. Each draws nothing when it has nothing true
   *  to say, so a page can mount either and let the answer decide. */
  variant?: 'open' | 'member'
} = {}) {
  const [me, setMe] = useState<Me | null>(
    given !== undefined && wellFormed(given) ? given : null)

  useEffect(() => {
    if (given !== undefined) {
      setMe(wellFormed(given) ? given : null)
      return
    }
    let cancelled = false
    fetchMe()
      .then((body) => {
        if (!cancelled) setMe(wellFormed(body) ? body : null)
      })
      // Silent. See the note above: no badge is the honest answer when we
      // cannot say what somebody holds.
      .catch(() => { if (!cancelled) setMe(null) })
    return () => { cancelled = true }
  }, [given])

  if (!me) return null

  if (me.founder && me.ordinal !== null) {
    return (
      <p className="fb fb-member founder-badge" data-founder="member">
        Founding member #{me.ordinal} — every draft free
      </p>
    )
  }

  // The dashboard's bar states a membership or says nothing; it never makes
  // the offer, because everybody reading it has already connected.
  if (variant === 'member') return null

  // The offer, and only to somebody who can still take it.
  if (!me.connected && me.founders_left > 0) {
    const seats = me.founders_left
    return (
      <p className="fb fb-open founder-badge" data-founder="offer">
        {seats} founder {seats === 1 ? 'spot' : 'spots'} left — connect ESPN to
        claim one
      </p>
    )
  }

  return null
}
