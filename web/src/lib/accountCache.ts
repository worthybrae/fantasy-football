import type { UpcomingDrafts } from '../api'

// The last answer `/api/espn/drafts` gave this tab.
//
// The landing page is two different pages -- the pitch for a stranger, the
// dashboard for a signed-in reader -- and which one it is cannot be known
// until that probe answers. Without a remembered answer, every visit to `/`
// (an Archive -> Drafts click, most of all) would either paint the wrong page
// and swap it, or paint nothing for a round trip. This is the remembered
// answer: it seeds the page's first frame, and the live probe overwrites it
// a moment later. `sessionStorage`, not `localStorage`: a snapshot that dies
// with the tab cannot be a week stale, and it is the same tab's own
// navigation this exists to smooth.
const KEY = 'espn-account'

export function readAccount(): UpcomingDrafts | null {
  try {
    const raw = window.sessionStorage.getItem(KEY)
    if (!raw) return null
    const body = JSON.parse(raw) as UpcomingDrafts
    return typeof body.connected === 'boolean' && Array.isArray(body.leagues) ? body : null
  } catch {
    return null
  }
}

export function rememberAccount(body: UpcomingDrafts): void {
  try { window.sessionStorage.setItem(KEY, JSON.stringify(body)) } catch { /* private window */ }
}

/** After a disconnect: the next visit must not open on a dashboard the
 *  probe is about to take away. */
export function forgetAccount(): void {
  try { window.sessionStorage.removeItem(KEY) } catch { /* private window */ }
}
