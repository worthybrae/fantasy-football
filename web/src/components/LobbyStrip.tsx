import { useEffect, useState } from 'react'
import { fetchLobby, type LobbyRoom, type LobbySummary } from '../api'

// Live proof, not a claim. The CTA next to this used to just assert "try it
// in a mock draft" -- this strip backs that up with ESPN's own numbers,
// read live off its public mock-lobby directory (api/lobby.py proxies it,
// cached ~60s server-side; no auth on either leg).
//
// RENDERS NOTHING ON ANYTHING BUT A GOOD, AVAILABLE PAYLOAD. Same rule
// ReadinessStrip already follows one component over (see its own comment):
// a landing page visitor never sees an error state for a proof widget,
// because a broken box next to the headline undermines the pitch more than
// an absent one ever could. `available: false` (ESPN unreachable, or this
// server didn't recognise what it got back) and a failed fetch are handled
// identically -- both just mean "show nothing."

// Inside the server's own ~60s cache window, so most polls land on a cache
// hit rather than a fresh upstream call -- see api/lobby.py's module
// docstring for why that matters (one visitor's poll interval times a
// thousand visitors must not become a thousand requests to ESPN). Picked
// off the 45s point in the brief's 30-60s range rather than lining up with
// the server's own TTL, so many browsers polling independently don't all
// land on the same side of a refresh at once.
const POLL_MS = 45_000

// How often the on-screen countdown re-renders BETWEEN polls, ticking the
// last fetched `starts_in_seconds` down against the wall clock rather than
// waiting a full 45s to show a stale number counting down in big jumps.
// Purely a text update -- see the reduced-motion note on `.lp-lobby-dot`
// below for the one piece of this strip that IS an animation.
const TICK_MS = 1_000

// "in 45s" below a minute (a room that starts inside this poll's own
// interval is exactly where a visitor benefits from seeing the number move
// at all), "in 4 min" up to an hour, "in 1h 12m" past that -- ESPN's lobby
// never seems to list anything further out than ~15 minutes (see the
// reference doc), so the hour branch is a defensive fallback, not the
// common case.
function formatStartsIn(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  if (s < 60) return `${s}s`
  const minutes = Math.round(s / 60)
  if (minutes < 60) return `${minutes} min`
  const hours = Math.floor(minutes / 60)
  const remMinutes = minutes % 60
  return remMinutes ? `${hours}h ${remMinutes}m` : `${hours}h`
}

// A room's live-ticking remaining time, given how long ago the summary that
// named it was fetched. Floors at 0 -- a room whose clock ran out between
// polls should read "starting", not count into negative seconds.
function remainingSeconds(room: LobbyRoom, elapsedMs: number): number | null {
  if (room.starts_in_seconds === null) return null
  return Math.max(0, room.starts_in_seconds - elapsedMs / 1000)
}

function roomLabel(room: LobbyRoom): string {
  const size = room.league_size !== null ? `${room.league_size}-team` : null
  const scoring = room.scoring ?? null
  return [size, scoring].filter(Boolean).join(' ') || 'mock draft'
}

// How many upcoming rooms this strip lists beneath the headline. The server
// already caps `upcoming` at a handful; this keeps it to a tight row that
// fits the hero column (max-width 620px) rather than reproducing the whole
// list ESPN sent.
const ROOMS_SHOWN = 3

export default function LobbyStrip() {
  const [summary, setSummary] = useState<LobbySummary | null>(null)
  const [fetchedAtMs, setFetchedAtMs] = useState<number | null>(null)
  const [nowMs, setNowMs] = useState<number>(() => Date.now())

  useEffect(() => {
    let cancelled = false
    const poll = () => {
      fetchLobby()
        .then((s) => {
          if (cancelled) return
          setSummary(s)
          setFetchedAtMs(Date.now())
        })
        // No error state -- see the module comment above. A failed poll
        // just leaves the strip showing its last good numbers (or nothing,
        // on the very first load) until the next one succeeds.
        .catch(() => undefined)
    }
    poll()
    const id = window.setInterval(poll, POLL_MS)
    return () => { cancelled = true; window.clearInterval(id) }
  }, [])

  // The countdown tick. Only runs once there is something to count down --
  // no point waking up every second before the first response has landed.
  useEffect(() => {
    if (!summary?.available) return undefined
    const id = window.setInterval(() => setNowMs(Date.now()), TICK_MS)
    return () => window.clearInterval(id)
  }, [summary?.available])

  if (!summary || !summary.available) return null

  const elapsedMs = fetchedAtMs !== null ? Math.max(0, nowMs - fetchedAtMs) : 0
  const rooms = summary.upcoming.slice(0, ROOMS_SHOWN)
  const [next] = summary.upcoming
  const nextRemaining = next ? remainingSeconds(next, elapsedMs) : null

  return (
    <div className="lp-lobby" aria-live="polite">
      <p className="lp-lobby-headline">
        {/* Decorative only -- the text alone already says "right now"; a
            screen reader doesn't need a second announcement of a dot. The
            pulse is the one piece of motion in this strip and is the thing
            prefers-reduced-motion below actually targets. */}
        <span className="lp-lobby-dot" aria-hidden="true" />
        <span className="mono">{summary.total_open.toLocaleString()}</span>{' '}
        mock draft{summary.total_open === 1 ? '' : 's'} open right now
        {next && nextRemaining !== null && (
          <>
            {' — next '}
            <strong>{roomLabel(next)}</strong>
            {' starts in '}
            <span className="mono">{formatStartsIn(nextRemaining)}</span>
          </>
        )}
      </p>
      {rooms.length > 0 && (
        <ul className="lp-lobby-rooms">
          {rooms.map((room, i) => {
            const remaining = remainingSeconds(room, elapsedMs)
            return (
              // No stable id on a directory row this strip is handed --
              // ESPN's leagueId isn't part of the reduced payload
              // (api/lobby.py never forwards it, on purpose: this widget
              // only ever links to setup, never to a specific room) -- so
              // position in an already-sorted, capped list is the key.
              <li className="lp-lobby-room" key={i}>
                <span className="lp-lobby-room-fmt">{roomLabel(room)}</span>
                <span className="lp-lobby-room-joined">
                  {room.teams_joined} joined
                </span>
                <span className="lp-lobby-room-time mono">
                  {remaining !== null ? formatStartsIn(remaining) : '—'}
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
