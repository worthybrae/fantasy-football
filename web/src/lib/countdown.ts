// Time until a draft, in the one shape this product says it.
//
// EVERY SCREEN IN THIS APP IS A CLOCK. The pick clock counts down 30 seconds,
// the ticker counts up 128 picks, the ranked list prices a player against how
// long he lasts. So the signed-in page states time the way the room does --
// monospaced digits, largest unit first, never a sentence -- and this is the
// one function that decides what those digits say.
//
// The unit shrinks as the moment approaches, which is the whole point: three
// days out, seconds are noise and "2d 14h" is the answer; inside the hour, the
// seconds are the story and it becomes a clock face that visibly moves. A
// reader glancing at the page can tell how close a draft is from the SHAPE of
// the number before reading a digit of it.

export interface Countdown {
  /** The digits themselves: `2d 14h`, `14h 09m`, `9:05`, `0:42`. */
  text: string
  /** Seconds remaining, negative once the moment has passed. */
  seconds: number
  /** Inside the last five minutes: close enough that a reader should move. */
  imminent: boolean
}

function pad(n: number): string {
  return String(Math.floor(n)).padStart(2, '0')
}

export function countdownTo(seconds: number | null): Countdown | null {
  if (seconds === null || !Number.isFinite(seconds)) return null
  const left = Math.max(0, Math.floor(seconds))
  const imminent = left < 300
  if (left >= 86_400) {
    const days = Math.floor(left / 86_400)
    return { text: `${days}d ${pad((left % 86_400) / 3600)}h`, seconds: left, imminent }
  }
  if (left >= 3600) {
    return { text: `${Math.floor(left / 3600)}h ${pad((left % 3600) / 60)}m`, seconds: left, imminent }
  }
  return { text: `${Math.floor(left / 60)}:${pad(left % 60)}`, seconds: left, imminent }
}

/** Seconds until an ISO timestamp, or null when there is no timestamp. */
export function secondsUntil(at: string | null, now: number): number | null {
  if (at === null) return null
  const when = Date.parse(at)
  return Number.isNaN(when) ? null : (when - now) / 1000
}

/** The day and time itself, for a draft too far off for a countdown to mean
 *  anything -- nobody plans around "in 73 hours". */
export function calendarLabel(at: string | null): string {
  if (at === null) return 'no date set'
  const when = new Date(at)
  return when.toLocaleString(undefined, {
    weekday: 'short', month: 'short', day: 'numeric',
    hour: 'numeric', minute: '2-digit',
  })
}
