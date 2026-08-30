// READY FOR DRAFT NIGHT.
//
// Three things make the difference between a draft night that works and one
// that does not, and all three happen days earlier: the league is connected,
// the guys are picked, and the reader has been in the room once before it
// matters. So the list is those three, each with a tick or a button, and it
// gets out of the way the moment they are all done -- a checklist of ticks is
// a page telling somebody how well they have followed instructions.
//
// EVERY ROW IS A FACT, NOT A FEELING. Connected is the session this page was
// rendered for. Picked is the saved list against the product's own floor of
// five. The dry run is a mock room this account holds a seat in, or one
// opened from this page -- and that second half is remembered in this browser
// rather than on the server, because a mock draft played somewhere else is
// still a mock draft played.

/** Where a dry run is remembered. Per browser, and deliberately: nothing on
 *  the server records "this person watched a room", and a checklist that
 *  waited for one would never tick. */
const DRY_RUN_KEY = 'home-dry-run'

export function markDryRun(): void {
  try {
    window.localStorage.setItem(DRY_RUN_KEY, '1')
  } catch { /* private window: the row simply stays open */ }
}

export function readDryRun(): boolean {
  try {
    return window.localStorage.getItem(DRY_RUN_KEY) === '1'
  } catch {
    return false
  }
}

/** The product's own floor: a plan built from two names is not a plan. */
export const ENOUGH_GUYS = 5

interface Row {
  key: string
  label: string
  /** What the row says once it is done -- the fact, not a congratulation. */
  done: string
  /** What the button says while it is not. */
  todo: string
  ready: boolean
  act: () => void
}

export default function ReadinessChecklist({
  connected, guys, dryRun, onConnect, onPickGuys, onDryRun,
}: {
  connected: boolean
  /** How many players are saved. */
  guys: number
  /** Whether this account or this browser has been in a mock room. */
  dryRun: boolean
  onConnect: () => void
  onPickGuys: () => void
  onDryRun: () => void
}) {
  const rows: Row[] = [
    {
      key: 'espn',
      label: 'Connect ESPN',
      done: 'your leagues are here',
      todo: 'Connect ESPN',
      ready: connected,
      act: onConnect,
    },
    {
      key: 'guys',
      label: 'Pick my guys',
      done: `${guys} saved`,
      todo: guys === 0 ? 'Pick my guys' : `Pick ${ENOUGH_GUYS - guys} more`,
      ready: guys >= ENOUGH_GUYS,
      act: onPickGuys,
    },
    {
      key: 'mock',
      label: 'Do a one-minute dry run',
      done: 'you have been in the room',
      todo: 'Open a mock room',
      ready: dryRun,
      act: onDryRun,
    },
  ]
  const left = rows.filter((row) => !row.ready)

  // ALL THREE DONE IS ONE LINE. The checklist has nothing left to ask for,
  // and a page that keeps a section open to show three ticks is spending the
  // reader's screen on our own bookkeeping.
  if (left.length === 0) {
    return (
      <p className="rc rc-done">
        <span className="rc-tick" aria-hidden="true">✓</span>
        You&rsquo;re ready for draft night.
      </p>
    )
  }

  return (
    <section className="db-sec rc" aria-labelledby="rc-h">
      <div className="db-sec-head">
        <h2 className="db-sec-title" id="rc-h">Ready for draft night</h2>
        <span className="mono db-sec-count">
          {left.length} to do
        </span>
      </div>
      <ul className="rc-rows">
        {rows.map((row) => (
          <li key={row.key}
              className={`rc-row${row.ready ? ' is-ready' : ''}`}>
            <span className="rc-tick" aria-hidden="true">
              {row.ready ? '✓' : '○'}
            </span>
            <span className="rc-label">{row.label}</span>
            {row.ready ? (
              <span className="rc-said">{row.done}</span>
            ) : (
              <button type="button" className="db-go db-go-view rc-go"
                      onClick={row.act}>
                {row.todo}
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}
