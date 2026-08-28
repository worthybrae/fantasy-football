import { fetchPlayers, type Player } from '../api'

// THE BOARD, FETCHED AT MOST ONCE PER PAGE.
//
// `/api/players` is about 250 rows -- every market rank, last season's weekly
// scores, a photograph -- and it is the payload the favourites picker is built
// from. It used to be fetched on the signed-in dashboard whether or not
// anybody opened the picker; now only the picker asks for it, and this is what
// stops a reader who opens it twice paying for it twice.
//
// A Map rather than a bare variable because it says what it is -- a cache
// keyed by what it holds -- and because the promise, not its result, is what
// is stored: two opens in the same tick share one request instead of starting
// two. A failed request is evicted, since the point of this is to save a round
// trip, not to remember that the server was down once.
const CACHE = new Map<string, Promise<Player[]>>()
const KEY = 'players'

export function loadBoard(): Promise<Player[]> {
  const cached = CACHE.get(KEY)
  if (cached) return cached
  const request = fetchPlayers().catch((e) => {
    CACHE.delete(KEY)
    throw e
  })
  CACHE.set(KEY, request)
  return request
}

/** Test seam, and nothing else uses it: a module-level cache outlives a test
 *  file's mocks, which would make the second test in a file assert against the
 *  first one's board. */
export function forgetBoard(): void {
  CACHE.delete(KEY)
}
