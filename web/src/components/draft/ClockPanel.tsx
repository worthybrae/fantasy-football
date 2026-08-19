import { useEffect, useState } from 'react'
import type { LiveState } from '../../api'
import { nextPickFor } from './pickOrder'

const roundOf = (pickNo: number, teams: number) => Math.ceil(pickNo / teams)

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

export default function ClockPanel({ state, secondsLeft }: { state: LiveState; secondsLeft: number | null }) {
  // Own ticker, not a prop from DraftRoom: this component's signature is
  // fixed to {state, secondsLeft} (Task 8/9 are written against it). Its
  // only real job is `pollAgeLabel`'s "Xs ago" text below, which needs a
  // per-second re-render to stay current between polls. It does NOT smooth
  // the countdown itself: `secondsLeft` (sourced from state.ms_remaining by
  // DraftRoom, see its own comment) only changes when ESPN's own CLOCK
  // frame updates DraftListener.ms_remaining -- and the real limit there is
  // ESPN's broadcast interval, not this app's 2.5s poll: observed in the
  // fixture (tests/fixtures/espn_draft_socket.jsonl), consecutive CLOCK
  // frames land about 5.0s apart (63696 -> 59762 -> 54755 -> 49747 -> ...,
  // a steady ~5007ms). So the displayed clock steps down in ~5s jumps, not
  // ~2.5s ones -- lowering POLL_MS would not smooth it, since the poll was
  // never the bottleneck. A true 1Hz countdown would need to interpolate
  // from ms_remaining plus the wall-clock moment it was read -- deliberately
  // not done here: it would require assuming the browser's clock and the
  // server's agree (true enough for this tool, which only ever runs on one
  // machine, but a real assumption worth naming rather than baking in
  // silently), for a smoothness gain on a personal tool where a ~5s-granular
  // number over a 30s clock is already legible and honest, and nothing
  // upstream promises finer resolution than ESPN's own broadcast cadence.
  const [nowMs, setNowMs] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNowMs(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [])

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

  const heading = draftDone ? 'Draft complete' : youAreUp ? 'You are on the clock' : 'Waiting on the room'
  const pct = secondsLeft !== null ? Math.max(0, Math.min(100, (secondsLeft / ASSUMED_CLOCK_SECONDS) * 100)) : null

  return (
    <div className={`clock-panel${youAreUp ? ' clock-panel-up' : ''}`}>
      <div className="draft-cap">{heading}</div>
      <div className="clock-countdown mono">{formatCountdown(secondsLeft)}</div>
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
