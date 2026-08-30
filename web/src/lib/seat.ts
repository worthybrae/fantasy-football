// WHICH SEAT, AND HOW BIG THE LEAGUE IS.
//
// Three cards on the home page are answers about one seat -- the plan, "my
// guys", and the hero that names it -- and before a draft is connected there
// is no seat to read off anything. The reader tells us once and we remember
// it, so this is the one place that knows what "your seat" means: one store
// key, one clamp, one word for a seat number.
//
// It lives in lib/ rather than in whichever card happened to need it first.
// Two stores of "which seat am I" would be two answers, and the one that lost
// would be whichever card the reader touched last.

/** The seat, remembered between visits. One key, one small object: the two
 *  values are only ever meaningful together (slot 12 is nonsense in an
 *  eight-team league). */
const STORE_KEY = 'guys-outlook'

/** What the endpoints will answer about: below four a snake is not a snake,
 *  above sixteen ESPN will not host it. `api/plan_preview.py` and
 *  `api/account.py`'s outlook both refuse outside this, so a value outside it
 *  is not a seat to ask about -- it is a question to not ask. */
export const MIN_TEAMS = 4
export const MAX_TEAMS = 16
export const DEFAULT_TEAMS = 10
export const DEFAULT_SLOT = 5

/** The league sizes worth OFFERING, which is not the same as the sizes worth
 *  accepting. Almost every real redraft league is one of these, and a select
 *  of thirteen options would be a worse way to say "twelve" -- but a league
 *  that says it holds sixteen holds sixteen, and no card is entitled to round
 *  that down to the nearest option in its own menu. */
export const SIZES = [8, 10, 12, 14]

export interface Seat { teams: number; slot: number }

/** "5th", for a seat select. The controls carry no bare numbers on purpose --
 *  a select of ten options labelled 1 to 10 puts ten loose digits into a page
 *  whose other cards are counting things. */
export function ordinal(n: number): string {
  if (n % 100 >= 11 && n % 100 <= 13) return `${n}th`
  return `${n}${({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th'}`
}

/** A seat both endpoints will answer about, from whatever we were handed.
 *
 *  THE SIZE IS BOUNDED, NOT ROUNDED. A twelve-team league is twelve and a
 *  sixteen-team league is sixteen; snapping either to the nearest option in
 *  the select is how a card came to print "10 teams" over a plan that said
 *  sixteen. Only a size the endpoints would refuse falls back to the default,
 *  because there is nothing truthful to ask about it. */
export function clampSeat(teams: unknown, slot: unknown): Seat {
  const asked = Number(teams)
  const size = Number.isInteger(asked) && asked >= MIN_TEAMS && asked <= MAX_TEAMS
    ? asked : DEFAULT_TEAMS
  const seat = Number(slot)
  return {
    teams: size,
    slot: Number.isInteger(seat) && seat >= 1 && seat <= size
      ? seat : Math.min(DEFAULT_SLOT, size),
  }
}

export function writeSeat(seat: Seat): void {
  try {
    window.localStorage.setItem(STORE_KEY, JSON.stringify(seat))
  } catch { /* private window: the seat is just not remembered */ }
}

export function readSeat(): Seat {
  try {
    const raw = window.localStorage.getItem(STORE_KEY)
    if (raw === null) return clampSeat(DEFAULT_TEAMS, DEFAULT_SLOT)
    const saved = JSON.parse(raw)
    return clampSeat(saved?.teams, saved?.slot)
  } catch {
    // A private window, or something that is not our JSON any more. Either
    // way the default seat is a better answer than a blank card.
    return clampSeat(DEFAULT_TEAMS, DEFAULT_SLOT)
  }
}
