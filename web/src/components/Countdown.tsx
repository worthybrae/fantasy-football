import { countdownTo, secondsUntil } from '../lib/countdown'
import { useSecond } from '../lib/clock'

// THE ONLY THINGS ON THE SIGNED-IN PAGE THAT RE-RENDER EVERY SECOND.
//
// Each of these subscribes to the shared clock (lib/clock.ts) on its own, so a
// tick re-renders one span of digits and nothing else. They exist because the
// page used to hold the clock instead: `now` was a state on Dashboard, handed
// down as a prop, and every second the whole page was rebuilt to move four
// characters.
//
// The tone classes live in here rather than on the parent for the same reason
// the text does: `is-soon` is a fact about the number, it changes when the
// number does, and a parent that owned it would have to re-render to apply it.

/** Time until a fixed moment -- a draft's own start time. */
export function Countdown({ at, className = '', empty = '—', unsetClass = '' }: {
  /** ISO timestamp, or null for a draft with no date set. */
  at: string | null
  className?: string
  /** What to print when there is no date at all. */
  empty?: string
  /** An extra class for that same case, when the caller draws it differently. */
  unsetClass?: string
}) {
  const count = countdownTo(secondsUntil(at, useSecond()))
  return (
    <span className={`${className}${count?.imminent ? ' is-soon' : ''}${count ? '' : unsetClass}`}>
      {count?.text ?? empty}
    </span>
  )
}

/** Time until a moment stated as "N seconds from when we read it" -- ESPN's
 *  lobby reports its rooms that way, so the countdown has to age the number
 *  against the read rather than against a timestamp it never sent. */
export function CountdownIn({ seconds, since, className = '', empty = '—' }: {
  seconds: number | null
  /** `Date.now()` at the read that reported `seconds`. */
  since: number
  className?: string
  empty?: string
}) {
  const now = useSecond()
  const count = countdownTo(seconds === null ? null : seconds - (now - since) / 1000)
  return (
    <span className={`${className}${count?.imminent ? ' is-soon' : ''}`}>
      {count?.text ?? empty}
    </span>
  )
}

/** How long ago something happened, in the caller's own words. `format` is a
 *  prop because "3m" and "3 minutes ago" are different registers and the two
 *  callers of this want different ones. */
export function Ago({ since, format }: {
  since: number
  format: (seconds: number) => string
}) {
  return <>{format(Math.max(0, (useSecond() - since) / 1000))}</>
}
