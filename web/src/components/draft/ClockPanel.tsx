import { useEffect, useState } from 'react'
import type { LiveState } from '../../api'

// This tool's one real league: 8 teams (scoring/config.py's LEAGUE_TEAMS,
// the same default LeagueSettings.teams every session falls back to when no
// ESPN import overrides it -- see scoring/league.py's default_settings).
// Neither /api/live/state nor anything else this room fetches carries the
// connected session's own team count, so -- same call as RosterPanel's
// fixed roster shape -- it is pinned here rather than fetched. A session
// whose ESPN league actually has a different team count would need this
// wired from the server instead; it does not exist as an API today.
const LEAGUE_TEAMS = 8

// Overall pick number (1-based) for `slot` (1-based) in round `round`
// (0-based), snake order: forward in even rounds, reversed in odd. Mirrors
// scoring/draft_sim.snake_slots exactly -- verified pick-by-pick against it
// (e.g. teams=4: round 1 gives slot 1 pick 8, slot 4 pick 5) rather than
// assumed from the task brief's formula, which turned out to already be
// correct.
function pickNumberFor(round: number, slot: number, teams: number): number {
  return round * teams + (round % 2 === 0 ? slot : teams - slot + 1)
}

// The next pick number >= `fromPickNo` belonging to `mySlot`. Every slot
// picks exactly once per round, so the answer is always in the round
// `fromPickNo` falls in, or the very next one -- no need for the league's
// total round count, which nothing this room fetches carries either.
function nextPickFor(fromPickNo: number, mySlot: number, teams: number): number {
  const round = Math.floor((fromPickNo - 1) / teams)
  const thisRound = pickNumberFor(round, mySlot, teams)
  return thisRound >= fromPickNo ? thisRound : pickNumberFor(round + 1, mySlot, teams)
}

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

// The full pick clock a countdown's progress bar would need is whatever
// ESPN's own CLOCK frames carry: pipeline/draft_listener.py already tracks
// it as `ms_remaining`, but api/live.py's live_state never returns it, so
// there is no way for this component to know how much of the clock is
// left. 30 is a reasonable placeholder ceiling for *when* a real value
// does arrive; until then `secondsLeft` is null and the bar renders empty.
const ASSUMED_CLOCK_SECONDS = 30

export default function ClockPanel({ state, secondsLeft }: { state: LiveState; secondsLeft: number | null }) {
  // Own ticker, not a prop from DraftRoom: this component's signature is
  // fixed to {state, secondsLeft} (Task 8/9 are written against it), and it
  // is the only consumer of a per-second tick -- pollAgeLabel's text today,
  // and a real countdown once secondsLeft has a source. A tick kept in
  // DraftRoom instead would sit unused.
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

  const youAreUp = state.on_the_clock !== null && state.on_the_clock === state.my_slot
  const draftDone = state.on_the_clock === null
  const thisPickNo = state.picks_made + 1

  const nextPickNo = state.my_slot !== null && !draftDone
    ? nextPickFor(thisPickNo, state.my_slot, LEAGUE_TEAMS)
    : null
  const gap = nextPickNo !== null ? nextPickNo - thisPickNo : null

  const heading = draftDone ? 'Draft complete' : youAreUp ? 'You are on the clock' : 'Waiting on the room'
  const pct = secondsLeft !== null ? Math.max(0, Math.min(100, (secondsLeft / ASSUMED_CLOCK_SECONDS) * 100)) : 0

  return (
    <div className={`clock-panel${youAreUp ? ' clock-panel-up' : ''}`}>
      <div className="draft-cap">{heading}</div>
      <div className="clock-countdown mono">{formatCountdown(secondsLeft)}</div>
      <div className="clock-bar-track" aria-hidden="true">
        <div className="clock-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      {state.stale && (
        <p className="clock-stale" role="status">
          No successful update in the last 15 seconds -- the numbers below may be behind.
        </p>
      )}
      <div className="clock-figures">
        <div>
          <div className="draft-cap">This pick</div>
          <div className="clock-figure mono">{thisPickNo} · RD {roundOf(thisPickNo, LEAGUE_TEAMS)}</div>
        </div>
        <div>
          <div className="draft-cap">Next pick</div>
          <div className="clock-figure mono">
            {nextPickNo !== null ? `${nextPickNo} · RD ${roundOf(nextPickNo, LEAGUE_TEAMS)}` : '—'}
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
