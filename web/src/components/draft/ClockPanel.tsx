import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import type { LiveBoard, LiveState } from '../../api'
import { nextPickFor } from './pickOrder'

const roundOf = (pickNo: number, teams: number) => Math.ceil(pickNo / teams)

// Defect 4: the rail names whose pick it is by the team's real name when it
// can, straight off /api/live/board's own `columns` (team_name, keyed by
// slot) -- the room already polls that endpoint every 2.5s (DraftRoom.tsx),
// so this fetches nothing new. Null (falling back to the bare slot number
// at the call site) whenever the board hasn't loaded yet, isn't active, or
// -- defensively, should not happen while the board IS active -- doesn't
// carry a column for this slot.
function teamNameForSlot(board: LiveBoard | null, slot: number | null): string | null {
  if (slot === null || !board?.active) return null
  return board.columns.find((c) => c.slot === slot)?.team_name ?? null
}

// "Xs ago" against a live-ticking clock, distinct from api.ts's `ageLabel`
// (which rounds to whole minutes -- right for a sim run's age, but the
// staleness call here is judged in *seconds*, api/live.py's 15s
// STALE_AFTER_SECONDS, so the display needs that resolution). Moved here
// from the deleted LiveDraft.tsx unchanged.
function pollAgeLabel(iso: string | null, nowMs: number): string {
  if (iso === null) return 'never'
  const ms = nowMs - new Date(iso).getTime()
  if (!Number.isFinite(ms) || ms < 0) return 'just now'
  const seconds = Math.floor(ms / 1000)
  if (seconds < 2) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  return `${Math.floor(seconds / 60)}m ago`
}

// mm:ss, floor-rounded. `null` (no real countdown source -- see below)
// renders as `--:--` rather than `0:00`, which would read as "the clock
// just ran out" instead of "there is no clock feed."
function formatCountdown(seconds: number | null): string {
  if (seconds === null) return '--:--'
  const s = Math.max(0, Math.round(seconds))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

// `secondsLeft`'s progress bar has no way to know the pick clock's real
// FULL duration -- ESPN's CLOCK/SELECTING frames only ever carry how much
// is left (DraftListener.ms_remaining), never the length the clock started
// at, and /api/live/state does not invent one. 30 is a reasonable placeholder
// ceiling for scaling the bar's width; it never appears as a number on
// screen, only as a fraction, so a league whose real clock runs longer or
// shorter just reads a bar that never quite empties or empties early --
// cosmetic, not a claim about a value the tool does not know.
const ASSUMED_CLOCK_SECONDS = 30

// Defect 3 (post-merge fix): ESPN broadcasts CLOCK roughly every 5s, not
// every second -- measured in the fixture (tests/fixtures/
// espn_draft_socket.jsonl): consecutive CLOCK frames land about 5.0s apart
// (63696 -> 59762 -> 54755 -> 49747 -> ..., a steady ~5007ms). If this
// component has gone materially longer than that with no FRESH
// `secondsLeft` value at all, the feed itself has likely stalled (a dropped
// socket, a reconnect in progress) and continuing to count down locally
// would show a number silently drifting away from ESPN's own clock -- worse
// than admitting the countdown does not know. 12s is a little over 2x the
// observed ~5s cadence: comfortably past one merely-late broadcast (network
// jitter, or this room's own 2.5s poll landing awkwardly against ESPN's 5s
// one) while still catching a real stall well inside a single pick's clock.
const COUNTDOWN_STALE_AFTER_MS = 12_000

export default function ClockPanel({
  state, secondsLeft, board = null,
  autodraftPending = false, autodraftError = null, onSetAutodraft = null,
}: {
  state: LiveState
  secondsLeft: number | null
  // Defect 4's team-name lookup only (see teamNameForSlot), defaulted to
  // null so an existing caller that hasn't been updated still type-checks.
  // DraftRoom already holds this from its own /api/live/board poll.
  board?: LiveBoard | null
  // The autodraft round trip, owned by DraftRoom for the same reason the
  // pick dialog's status is (see its own `pickStatus` comment): a 2.5s poll
  // landing mid-request must not lose track of a request in flight, and the
  // request itself has to outlive whatever re-renders around it. The
  // CONFIRMED state is not among these -- that is `state.autodraft`, off the
  // poll, so this panel can never show a value ESPN did not send.
  autodraftPending?: boolean
  autodraftError?: string | null
  // null (the default) makes the control read-only: it still shows what ESPN
  // says, it just has nowhere to send a change. Same defaulting precedent as
  // `board` above.
  onSetAutodraft?: ((on: boolean) => void) | null
}) {
  // Own ticker, not a prop from DraftRoom: this component's signature is
  // {state, secondsLeft} (Task 8/9 were written against it, and Defect 3
  // does not need to change it -- everything the interpolation below needs
  // beyond `secondsLeft` itself is purely local: the wall-clock moment each
  // new value arrived, and a per-second re-render to count down between
  // arrivals). Also drives `pollAgeLabel`'s "Xs ago" text below.
  const [nowMs, setNowMs] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNowMs(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [])

  // The interpolation anchor: the last `secondsLeft` value that actually
  // arrived (a fresh poll's ms_remaining, floor-seconds already applied by
  // DraftRoom), paired with `Date.now()` at the moment THIS component saw
  // it change. Re-set only when `secondsLeft` itself changes -- React's own
  // dependency-array equality check is exactly "a new value arrived," so a
  // poll that repeats the same rounded second (common: this room polls
  // every 2.5s, ESPN broadcasts every ~5s) leaves the anchor alone rather
  // than restarting the countdown from the same number. `secondsLeft` is
  // already only single-second resolution (Math.round(ms_remaining/1000)
  // in DraftRoom), so the anchor's own precision is exactly what this
  // component was ever promised -- it does not assume any finer-grained
  // clock agreement between browser and server than that.
  const [anchor, setAnchor] = useState<{ value: number; atMs: number } | null>(
    secondsLeft !== null ? { value: secondsLeft, atMs: Date.now() } : null,
  )
  useEffect(() => {
    setAnchor(secondsLeft !== null ? { value: secondsLeft, atMs: Date.now() } : null)
  }, [secondsLeft])

  // The actual per-second countdown: anchor value minus whole seconds
  // elapsed since it was recorded, never below zero. Null (nothing to
  // interpolate from) and "stale" (see COUNTDOWN_STALE_AFTER_MS above) both
  // fall back to `--:--` via formatCountdown, rather than freezing on a
  // number that may no longer be true -- an explicit "unknown" reads more
  // honestly on a personal tool than a clock that quietly stopped moving.
  const staleAnchor = anchor !== null && nowMs - anchor.atMs > COUNTDOWN_STALE_AFTER_MS
  const displaySeconds = anchor === null || staleAnchor
    ? null
    : Math.max(0, anchor.value - Math.floor((nowMs - anchor.atMs) / 1000))

  if (!state.active) {
    return (
      <div className="clock-panel">
        <p className="rail-empty">No live draft connected.</p>
      </div>
    )
  }

  // Draft night's worst failure: a board that looks current and has simply
  // stopped updating. This replaces the countdown outright rather than
  // sharing space with it -- a stopped listener means everything below is
  // suspect, not just late.
  const listenerDown = state.listener_error !== null || !state.listener_alive
  if (listenerDown) {
    return (
      <div className="clock-panel clock-panel-down" role="alert">
        <div className="draft-cap">Listener down</div>
        <p className="clock-down-message">
          {state.listener_error ?? 'The draft listener has stopped responding -- picks made in ESPN will not appear here.'}
        </p>
        {/* Spec section 6 wants the recovery instruction shown whenever the
            room is down, not only when it happens to be embedded in the
            error text. It only ever was: the words "click the Draft Assistant
            bookmark again" live inside two RuntimeError strings in
            pipeline/draft_socket.py (the two give-up-after-N-empty-
            reconnects paths), so every other way this thread dies -- and
            every hang that sets no error at all, which is exactly the case
            `listener_alive` exists to catch -- reached this panel with a
            message and no action. Unconditional now, and it links back to
            the page that actually hands the bookmarklet over, since App.tsx
            routes one way and nothing else here leaves the room. */}
        <p className="clock-down-message">
          Click the Draft Assistant bookmarklet again in your ESPN draft tab to
          reconnect, or <Link to="/">go back to setup</Link>.
        </p>
      </div>
    )
  }

  // state.settings.teams is guaranteed a real number once a session is
  // active -- build_session always attaches real LeagueSettings (the
  // league's own ESPN import, or the cold-start default), never the
  // inactive response's null shape; only that null shape reaches this
  // component before the `!state.active` return above already bailed out.
  // Replaces LEAGUE_TEAMS, the 8-team constant this file and DraftRoom.tsx
  // used to each hardcode (Task 6+7 review finding #2).
  const teams = state.settings.teams as number

  const youAreUp = state.on_the_clock !== null && state.on_the_clock === state.my_slot
  const draftDone = state.on_the_clock === null
  // null, not picks_made + 1, once the draft is over -- a 120-pick, 15-round
  // league otherwise reads "121 · RD 16," naming a round that does not
  // exist (Task 6+7 review finding #4). Renders as the same em dash '—'
  // "Next pick"/"Gap" already fall back to.
  const thisPickNo = draftDone ? null : state.picks_made + 1

  const nextPickNo = state.my_slot !== null && !draftDone && thisPickNo !== null
    ? nextPickFor(thisPickNo, state.my_slot, teams)
    : null
  const gap = nextPickNo !== null && thisPickNo !== null ? nextPickNo - thisPickNo : null

  // Defect 4: "the draft hasn't started" and "waiting on someone's pick"
  // used to render as the identical "Waiting on the room" -- draft_started
  // (DraftListener.started, off the socket's own STATE frame) tells them
  // apart. Whose pick it is prefers the team's real name (via `board`,
  // already polled -- no new fetch) over a bare slot number, falling back
  // to the slot only when the board hasn't named that column yet.
  const waitingOnName = teamNameForSlot(board, state.on_the_clock)

  // `draft_started` alone is not evidence that the draft has NOT started.
  // It is DraftListener.started, set from the socket's STATE frame, and
  // STATE is a one-time transition broadcast rather than a per-connection
  // handshake: tests/fixtures/espn_draft_socket.jsonl carries a single
  // `STATE 1` in 161 frames, and data/draft_room_trace.jsonl exactly two
  // STATE frames in the whole session (draft start, draft complete). A
  // fresh DraftListener is constructed on every connect, so any listener
  // created AFTER the draft began never sees one and reports
  // `started == False` for the rest of the draft -- and the bookmarklet
  // click this very panel's recovery text asks for (see the listener-down
  // branch above) is exactly what constructs one. Picks already on the
  // board are the second, independent witness that it started: `picks_made`
  // is counted from the `drafted` table, not from any frame this listener
  // has to have been alive for.
  const started = state.draft_started || state.picks_made > 0

  // `youAreUp` first, deliberately. It used to sit BELOW the not-started
  // test, so a listener that reconnected mid-draft (started == false
  // forever, see above) read "Draft has not started yet" over a live
  // ticking countdown while the owner was on the clock -- the one heading
  // that must never be wrong. Now the not-started case can only ever paint
  // when nothing else is true: nobody's pick is ours, and no pick has
  // landed.
  const heading = draftDone
    ? 'Draft complete'
    : youAreUp
      ? 'You are on the clock'
      : !started
        ? 'Draft has not started yet'
        : waitingOnName !== null
          ? `Waiting on ${waitingOnName}`
          : `Waiting on slot ${state.on_the_clock}`
  // Bar and text both driven off the same interpolated value -- displaySeconds,
  // not the raw (~5s-granular) secondsLeft prop -- so the two never visibly
  // disagree (a bar frozen in 5s jumps under a number ticking every second).
  const pct = displaySeconds !== null
    ? Math.max(0, Math.min(100, (displaySeconds / ASSUMED_CLOCK_SECONDS) * 100))
    : null

  // ESPN's own flag, never this component's guess -- `state.autodraft` is
  // whatever ESPN last broadcast for our team (api/live.py serves
  // DraftListener.my_autodraft). Three values: true, false, and null for
  // "ESPN has not said", which must not render as "off" (see api.ts's own
  // comment on the field). null shows a dash and no switch: there is no
  // honest position to draw a two-state control in, and it lasts about one
  // frame of a real session.
  const autodraftOn = state.autodraft === true
  // A switch that cannot send is worse than no switch -- it would accept a
  // click and silently do nothing. The socket is the only thing that can
  // carry the command (the browser-observer path publishes none at all, and
  // a reconnect detaches the one it has), and a request already in flight
  // must not take a second click.
  const canToggle = onSetAutodraft !== null && state.socket_alive && !autodraftPending
  const autodraftTitle = autodraftPending
    ? 'Waiting for ESPN to confirm…'
    : onSetAutodraft === null
      ? 'Autodraft, as ESPN last reported it'
      : !state.socket_alive
        ? 'The draft socket is not connected — change this in ESPN itself'
        : autodraftOn
          ? 'ESPN is drafting for you. Switch off to take your picks back.'
          : 'Switch on to let ESPN draft for you.'

  return (
    <div className={`clock-panel${youAreUp ? ' clock-panel-up' : ''}${autodraftOn ? ' clock-panel-auto' : ''}`}>
      <div className="draft-cap">{heading}</div>
      {/* The countdown keeps the whole left side; the autodraft control sits
          beside it, top right, where the owner asked for it. It is state
          first and control second: when autodraft is off this is a quiet
          grey switch that does not compete with a 48px clock, and when ESPN
          has flipped it on the whole panel turns and the alert below spells
          out what is happening. */}
      <div className="clock-headline">
        <div className="clock-countdown mono">{formatCountdown(displaySeconds)}</div>
        <div className="autodraft">
          <span className="draft-cap autodraft-cap">Autodraft</span>
          {state.autodraft === null ? (
            <span className="autodraft-unknown mono" title="ESPN has not reported autodraft for your team yet">—</span>
          ) : (
            <button
              type="button"
              role="switch"
              aria-checked={autodraftOn}
              aria-busy={autodraftPending}
              aria-label="ESPN autodraft"
              title={autodraftTitle}
              className={`autodraft-switch${autodraftOn ? ' is-on' : ''}${autodraftPending ? ' is-pending' : ''}`}
              disabled={!canToggle}
              onClick={() => onSetAutodraft?.(!autodraftOn)}
            >
              <span className="autodraft-knob" />
            </button>
          )}
        </div>
      </div>
      {autodraftOn && (
        // The alarm, not a status line. ESPN flips a team to autodraft when
        // it misses a pick and then just carries on drafting for it -- the
        // person it is happening to may have no idea, which is the whole
        // reason this feature exists. role="alert" so it is announced the
        // moment the poll brings it in, not only when someone looks.
        <div className="autodraft-alert" role="alert">
          <strong>ESPN is drafting for you.</strong> A pick ran out of time, so
          ESPN is making your picks itself. Switch autodraft off to take them
          back.
        </div>
      )}
      {autodraftError !== null && (
        // The server's own words, verbatim, same rule ConfirmPick follows: a
        // 504 here says "ESPN did not confirm", which is not "it failed" --
        // and the difference decides whether the owner needs to go look at
        // the ESPN room.
        <p className="clock-stale" role="alert">{autodraftError}</p>
      )}
      {pct !== null && (
        // Rendered only when there IS a real clock value -- a `pct: 0` bar
        // sitting under `--:--` used to read as "the clock just ran out,"
        // the exact opposite of what `--:--` itself is there to say (Task
        // 6+7 review finding #1). No clock feed now means no bar at all.
        <div className="clock-bar-track" aria-hidden="true">
          <div className="clock-bar-fill" style={{ width: `${pct}%` }} />
        </div>
      )}
      {state.stale && (
        <p className="clock-stale" role="status">
          No successful update in the last 15 seconds -- the numbers below may be behind.
        </p>
      )}
      {/* Spec section 6's first case, said where it is being felt: "socket
          dropped while on the clock -- the clock panel says so and the draft
          buttons disable." Only while `youAreUp`, deliberately. The
          browser-observer path (/api/live/connect) never publishes a
          SocketHandle at all, so `socket_alive` is permanently false there
          and an unconditional notice would be a permanent one for a session
          that is working exactly as designed; on the clock is the one moment
          the difference actually costs the user something. */}
      {youAreUp && !state.socket_alive && (
        <p className="clock-stale" role="alert">
          The draft socket is not connected -- picks cannot be sent from here
          right now. Make this pick in ESPN.
        </p>
      )}
      <div className="clock-figures">
        <div>
          <div className="draft-cap">This pick</div>
          <div className="clock-figure mono">
            {thisPickNo !== null ? `${thisPickNo} · RD ${roundOf(thisPickNo, teams)}` : '—'}
          </div>
        </div>
        <div>
          <div className="draft-cap">Next pick</div>
          <div className="clock-figure mono">
            {nextPickNo !== null ? `${nextPickNo} · RD ${roundOf(nextPickNo, teams)}` : '—'}
          </div>
        </div>
        <div>
          <div className="draft-cap">Gap</div>
          <div className="clock-figure mono">{gap !== null ? `${gap} picks` : '—'}</div>
        </div>
      </div>
      <div className="clock-updated mono" title={state.last_poll_at ?? undefined}>
        updated {pollAgeLabel(state.last_poll_at, nowMs)}
      </div>
    </div>
  )
}
