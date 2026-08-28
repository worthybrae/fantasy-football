// WHICH ROSTER SLOT A PICK WOULD FILL, in one spelling.
//
// The server tags every candidate with `need` -- scoring/gain.need_kind's own
// five words, unchanged by the redesign -- and three surfaces print it: the
// confirm dialog, the profile overlay's seed row, and the target cards. They
// used to each carry their own copy of the rule, which is how "FLEX" ended up
// accented in two of them and plain in the third.
//
// The words themselves are the server's, only cased for reading: inventing
// friendlier ones here would mean a reader could not match what the room says
// to what the API says when something looks wrong.
const NEED_WORDS: Record<string, string> = {
  starter: 'Starter',
  flex: 'Flex',
  deferred: 'Later',
  bench: 'Bench',
  capped: 'Full',
}

export function needLabel(need: string | null | undefined): string {
  if (need === null || need === undefined || need === '') return '—'
  return NEED_WORDS[need] ?? need
}

/** Whether this pick fills a hole in the starting lineup -- the one case
 *  worth an accent. A flex slot counts; a bench spot, a position already
 *  full, and a need the plan is deliberately deferring do not. */
export function needIsOpenSlot(need: string | null | undefined): boolean {
  return need === 'starter' || need === 'flex'
}
