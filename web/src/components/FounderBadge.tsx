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
// IT RENDERS NOTHING UNTIL IT KNOWS, and nothing at all if the probe fails.
// The badge is a claim about what somebody has been given; a placeholder that
// resolves into "0 spots left" is worse than a line that was never there, and
// a store that is briefly away is not a reason to break the page around it.
//
// NOT PLACED ANYWHERE YET, deliberately. The landing page and the dashboard
// are being redesigned; this is the piece they will each drop in, so it draws
// one line, carries its own data, and takes its styling from whatever class
// the page that adopts it defines.

export default function FounderBadge({ me: given }: {
  /** An answer the page already has, for a caller that reads `/api/account/me`
   *  for its own reasons. Absent, the badge asks for itself -- the request is
   *  shared for five seconds either way, so two of these on one page is one
   *  round trip. */
  me?: Me | null
} = {}) {
  const [me, setMe] = useState<Me | null>(given ?? null)

  useEffect(() => {
    if (given !== undefined) {
      setMe(given)
      return
    }
    let cancelled = false
    fetchMe()
      .then((body) => { if (!cancelled) setMe(body) })
      // Silent. See the note above: no badge is the honest answer when we
      // cannot say what somebody holds.
      .catch(() => { if (!cancelled) setMe(null) })
    return () => { cancelled = true }
  }, [given])

  if (!me) return null

  if (me.founder && me.ordinal !== null) {
    return (
      <p className="founder-badge" data-founder="member">
        Founding member #{me.ordinal} — every draft free
      </p>
    )
  }

  // The offer, and only to somebody who can still take it.
  if (!me.connected && me.founders_left > 0) {
    const seats = me.founders_left
    return (
      <p className="founder-badge" data-founder="offer">
        {seats} founder {seats === 1 ? 'spot' : 'spots'} left — connect ESPN to
        claim one
      </p>
    )
  }

  return null
}
