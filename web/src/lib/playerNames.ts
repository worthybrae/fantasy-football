import type { Player } from '../api'

// WHO A PLAYER ID IS, WITHOUT FETCHING THE BOARD AGAIN.
//
// `/api/players` is 250 rows with a photograph each -- about 185 KB -- and
// there are places that hold a handful of player ids and need nothing from it
// but a name: the "Your guys" card is a list of up to twenty-five ids, and
// fetching a board to print them is the tail wagging the dog.
//
// So every board fetch leaves its names behind here on the way past
// (`fetchPlayers` calls `rememberNames`), and a caller that only needs a name
// asks this instead. Two consequences worth naming:
//
//   - It is a CACHE, not a source. A name it has not seen is `null` and the
//     caller shows what it can (a count, an id) rather than waiting; nothing
//     here ever issues a request.
//   - It lives in `sessionStorage`, so it survives a navigation inside one
//     visit and dies with the tab. A name is not private, but a stale roster
//     of somebody else's favourites sitting in a shared browser for weeks is
//     the kind of thing that only ever surprises people.

const KEY = 'player-names'

export interface KnownPlayer {
  name: string
  position: string
}

// Read once per page, then kept in memory: this is read on every render of
// the card that uses it, and JSON.parse of 250 entries per render is a
// silly way to print a name.
let memory: Record<string, KnownPlayer> | null = null

function load(): Record<string, KnownPlayer> {
  if (memory !== null) return memory
  try {
    const raw = window.sessionStorage.getItem(KEY)
    memory = raw ? JSON.parse(raw) as Record<string, KnownPlayer> : {}
  } catch {
    // A private window, or something else's key under this name.
    memory = {}
  }
  return memory
}

/** Everything a board fetch just told us, remembered for the tab. */
export function rememberNames(players: Player[]): void {
  const known = load()
  for (const p of players) known[p.player_id] = { name: p.name, position: p.position }
  memory = known
  try {
    window.sessionStorage.setItem(KEY, JSON.stringify(known))
  } catch {
    /* private window, or the quota: the names still stand for this page */
  }
}

/** In the order asked for, `null` where this has never seen the id. */
export function namesFor(ids: string[]): (KnownPlayer | null)[] {
  const known = load()
  return ids.map((id) => known[id] ?? null)
}

/** Test seam: a module-level cache outlives a test file's mocks. */
export function forgetNames(): void {
  memory = null
  try { window.sessionStorage.removeItem(KEY) } catch { /* nothing to forget */ }
}
