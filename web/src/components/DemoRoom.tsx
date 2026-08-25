import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchLiveMock, fetchPlayers,
         type LiveCandidate, type LiveMock, type LiveState, type Player } from '../api'
import { Logo } from './Logo'
import AvailableList from './draft/AvailableList'
import DraftBoardGrid from './DraftBoardGrid'
import PickTicker from './draft/PickTicker'
import PlayerOverlay, { type OverlayTarget } from './draft/PlayerOverlay'
import RosterPanel from './draft/RosterPanel'
import TopThree from './draft/TopThree'
import { nextPickFor } from './draft/pickOrder'
import { seedFromBoardPlayer, seedFromCandidate, seedFromPlayer } from './draft/playerSeed'

// THE LANDING PAGE IS THE DRAFT ROOM -- the whole window, not a panel of it.
//
// This renders the room's own shell (`.draft-room`, its top bar, its two
// tabs, its main column, its rail, its pick ticker) around the room's own
// components, against a real ESPN mock draft the farm is sitting in while the
// page is open. A visitor lands inside the product with a live draft running
// in it, and the only thing they cannot do is pick: `isMyTurn` is false, so
// every Draft button is disabled exactly as it is in the real room when it is
// somebody else's turn. Nothing is faked to produce that state.
//
// WHY NOT A SCREENSHOT, A VIDEO, OR A REBUILT LOOKALIKE. All three drift from
// the product the day after they are made, and none of them can be checked.
// This cannot drift: it imports the same components, is fed by the same
// board, and its numbers are wrong the moment the product's are.
//
// WHAT IS SYNTHESISED, AND WHY IT IS STILL HONEST. `ClockPanel` reads a
// `LiveState`, and this page has no session to produce one -- so one is built
// out of the room's own facts (picks made, who is on the clock, the league
// shape) with `my_slot` null. That is the browser-observer case the room
// already handles: watching a draft nobody here holds a seat in. No component
// needed a demo branch, and none was added.

// THE ROOM IS PUSHED, NOT POLLED. `/api/demo/events` is an SSE stream that
// says one line when the draft moves; this fetches the board when it hears
// one. What was here before was a 30-second timer, and a draft on a timer
// reads as a slideshow: a pick lands, the ticker sits unchanged for another
// twenty seconds, and the page's whole claim -- that this is happening right
// now -- is undermined by the only evidence a reader has for it.
//
// The timers below survive as a SAFETY NET, not the mechanism. A stream can
// be refused by a proxy, killed by a sleeping laptop, or simply never
// supported; a page whose liveness depends on one working would be a blank
// room when it does not. So: slow while the stream is up (a stream that has
// silently died looks exactly like a quiet room, and this is what tells the
// difference), and the original cadence whenever it is down.
const POLL_MS = 30_000
const EMPTY_POLL_MS = 6_000
const PUSHED_POLL_MS = 120_000

// How far past its clock a turn may run before the countdown stops claiming
// to know. ESPN autodrafts a seat the moment its clock expires, so the next
// pick normally lands within a second or two of zero; a minute past it means
// the farm's socket dropped, the room stalled, or the write we timed from
// never happened -- and a timer counting into the negative would be the one
// thing on this page inventing a fact. It disappears instead.
const STALLED_AFTER = 60

/** The open turn, measured once when a payload lands and counted from there.
 *  `elapsed` is how far in the room already was on the SERVER's clock;
 *  `since` is the browser's own timestamp for that instant, so everything
 *  after it is a local elapsed-time measurement and no comparison is ever
 *  made between two machines' clocks. */
type Turn = { length: number; elapsed: number; since: number }

function turnOf(room: LiveMock): Turn | null {
  const { clock_seconds: length, turn_started_at: opened, server_now: now } = room
  if (!room.live || !length || !opened || !now) return null
  return { length, elapsed: Math.max(0, now - opened), since: Date.now() }
}

/** m:ss, floor-rounded -- the room's own `formatCountdown` rule. */
function mmss(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds))
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`
}

export default function DemoRoom({ onMode, live = false, site = false }: {
  /** Standing as its own page (/live) rather than as the landing page's
   *  first screen. The bar then carries the site's own tab strip -- Drafts,
   *  Archive, Live -- ahead of the room's view tabs, the title routes home
   *  the way every other page's does, and the landing-only chrome (the
   *  viewport break-out, the faded scrollers) is left off: this is the room
   *  at full size, scrolling its own lists, exactly as /draft does. */
  site?: boolean
  /** Told whenever the room's mode changes, because the page's welcome card
   *  says "running now" and that is only true of one of them. Optional: the
   *  room stands on its own without a page around it. */
  onMode?: (mode: 'live' | 'replay' | 'none') => void
  /** Whether this browser has a draft of its own actually running. Puts the
   *  way back to it in this bar rather than in a band above the page, which
   *  pushed the room down and read as an alarm. */
  live?: boolean
} = {}) {
  const [room, setRoom] = useState<LiveMock | null>(null)
  const [players, setPlayers] = useState<Record<string, Player>>({})
  const [open, setOpen] = useState<OverlayTarget | null>(null)
  const [tab, setTab] = useState<'available' | 'board'>('available')
  const [turn, setTurn] = useState<Turn | null>(null)
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    let cancelled = false
    let timer = 0
    // Whether the stream is carrying us. Held in the closure rather than in
    // state because it decides the next timeout and nothing renders from it;
    // as state it would re-run this effect and tear down the stream it just
    // reported was working.
    let pushed = false
    let live = false

    const read = async () => {
      try {
        const body = await fetchLiveMock()
        if (cancelled) return
        live = Boolean(body.live)
        if (body.live) {
          setRoom(body)
        } else {
          // A room already on screen is not removed for a moment of nothing:
          // the farm's rooms rotate, so "no draft in progress" is routinely a
          // few seconds long, and a page that blanked every time one ended
          // would look broken far more often than it looked empty.
          setRoom((current) => (current?.live ? current : body))
        }
        setTurn(turnOf(body))
        onMode?.(body.mode ?? (body.live ? 'live' : 'none'))
      } catch {
        if (!cancelled) setRoom((current) => current ?? { live: false })
      }
    }

    // Always exactly one timer pending. `loop` is called again whenever the
    // stream's state changes, and clearing first is what stops a stream that
    // connects and drops repeatedly from leaving a fleet of timers behind,
    // each of them fetching.
    const loop = () => {
      window.clearTimeout(timer)
      // The slow cadence is for a room we already HAVE. With no room on
      // screen the stream cannot help: it speaks when the draft moves, and
      // "the server finished its first build" is not the draft moving -- so
      // a page that had gone quiet waited for the next pick to hear anything
      // at all, which after a restart meant a minute or two of "reading the
      // room" over a server that was ready in seconds.
      const wait = !live ? EMPTY_POLL_MS : (pushed ? PUSHED_POLL_MS : POLL_MS)
      timer = window.setTimeout(() => { void read().then(loop) }, wait)
    }

    void read().then(loop)

    // The stream carries no board -- only "the room moved", which is the cue
    // to fetch one. See api/demo.py's own note on why the payload is not
    // pushed down it.
    let events: EventSource | null = null
    try {
      events = new EventSource('/api/demo/events')
      events.onopen = () => { pushed = true; loop() }
      events.onmessage = () => { void read() }
      // EventSource reconnects by itself, so this is not a retry -- it is the
      // moment to go back to fetching on a timer until it succeeds.
      events.onerror = () => { pushed = false; loop() }
    } catch {
      // No EventSource at all. The timers above are then the whole
      // mechanism, at the cadence this page ran at before the stream existed.
    }

    return () => {
      cancelled = true
      window.clearTimeout(timer)
      events?.close()
    }
    // `onMode` deliberately out: the page passes a fresh closure every
    // render, and depending on it would tear down the stream on each one.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // The countdown's own heartbeat. One interval, running only while there is
  // a clock to draw -- a rewound room and a farm too old to publish one both
  // land here as `turn === null`, and neither should be paying for a timer.
  useEffect(() => {
    if (!turn) return
    setNow(Date.now())
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [turn])

  useEffect(() => {
    let cancelled = false
    fetchPlayers()
      .then((rows) => {
        if (cancelled) return
        const map: Record<string, Player> = {}
        for (const row of rows) map[row.player_id] = row
        setPlayers(map)
      })
      .catch(() => { /* names degrade to ids; the list still ranks */ })
    return () => { cancelled = true }
  }, [])

  const candidates: LiveCandidate[] = useMemo(
    () => (room?.shortlist ?? []).map((row) => ({
      player_id: row.player_id,
      position: row.position ?? '',
      proj_points: row.proj_points ?? 0,
      vor_points: row.vor_points ?? 0,
      gain_now: row.gain_now,
      gain_next: row.gain_next,
    edge_next: row.edge_next,
      survive_pct: row.survive_pct,
      fills: row.fills,
      rank: row.rank ?? 0,
    })),
    [room],
  )

  // The `LiveState` the rail's clock panel reads, built from the room's own
  // facts rather than invented: `my_slot` null (nobody here has a seat),
  // `draft_started` true (picks have landed), and no clock at all, because
  // the farm records picks rather than the seconds between them -- ClockPanel
  // already draws "waiting on the room" when there is no time to show.
  const state: LiveState | null = useMemo(() => {
    if (!room?.live || !room.settings) return null
    return {
      active: true,
      picks_made: room.picks_made ?? 0,
      on_the_clock: room.on_the_clock ?? null,
      my_slot: null,
      draft_started: true,
      candidates,
      candidates_as_of_pick: room.picks_made ?? 0,
      horizon_pick: null,
      horizon_is_end_of_draft: false,
      last_poll_at: null,
      stale: false,
      unmapped_picks: [],
      listener_error: null,
      listener_alive: true,
      recompute_error: null,
      socket_alive: true,
      autodraft: null,
      ms_remaining: null,
      settings: room.settings,
      my_roster: [],
    } as LiveState
  }, [room, candidates])

  const openCandidate = useCallback((candidate: LiveCandidate) => {
    setOpen({ playerId: candidate.player_id,
              seed: seedFromCandidate(candidate, players[candidate.player_id]) })
  }, [players])

  // Seconds left on the seat's clock, or null when the page does not know --
  // no clock published, a rewound room, or a turn that has run so far past
  // its clock that the room has plainly stalled. Null draws no timer at all.
  const secondsLeft = useMemo(() => {
    if (!turn) return null
    const left = turn.length - (turn.elapsed + (now - turn.since) / 1000)
    if (left < -STALLED_AFTER) return null
    return Math.max(0, left)
  }, [turn, now])

  // A draft that already happened, played back at its own pace, because no
  // room is drafting this minute. Everything below draws it identically --
  // it is the same product against the same kind of room -- and the only
  // difference on screen is the pill saying so.
  const replaying = room?.mode === 'replay'

  // The seat on the clock, and when it comes round again -- the room's own
  // snake arithmetic (`nextPickFor`), which is as true for a seat somebody
  // else holds as for one you do.
  const thisPick = (room?.picks_made ?? 0) + 1
  const nextPick = room?.on_the_clock && room?.teams
    ? nextPickFor(thisPick, room.on_the_clock, room.teams)
    : null

  const waiting = !room?.live || candidates.length === 0 || !state

  return (
    <div className={`draft-room${site ? '' : ' dr-demo'}`}>
      <header className="draft-topbar">
        {/* On the landing page this is not a link: the title routes home,
            and that page IS home. On /live it routes home like everywhere
            else. */}
        {site
          ? <Link to="/" className="draft-topbar-title"><Logo /> ESPN Draft Assist</Link>
          : <span className="draft-topbar-title"><Logo /> ESPN Draft Assist</span>}
        <span className="draft-topbar-sep" aria-hidden="true" />
        {site && (
          <>
            <nav className="draft-topbar-tabs" aria-label="Views">
              <Link className="draft-tab" to="/">Drafts</Link>
              <Link className="draft-tab" to="/archive">Archive</Link>
              <Link className="draft-tab is-active" to="/live" aria-current="page">Live</Link>
            </nav>
            <span className="draft-topbar-sep" aria-hidden="true" />
          </>
        )}
        {/* `draft-topbar-tabs`, the room's own class. An earlier
            `draft-tabs` was the OLD wrapper this bar replaced -- it has no
            rules any more, so the buttons fell back to stacking. */}
        {/* `aria-current`: same reason as the real room -- the active
            tab is colour alone on screen. */}
        <nav className="draft-topbar-tabs" aria-label="Draft views">
          <button
            type="button"
            className={`draft-tab${tab === 'available' ? ' is-active' : ''}`}
            aria-current={tab === 'available'}
            onClick={() => setTab('available')}
          >
            Available
          </button>
          <button
            type="button"
            className={`draft-tab${tab === 'board' ? ' is-active' : ''}`}
            aria-current={tab === 'board'}
            onClick={() => setTab('board')}
          >
            Snake Board
          </button>
        </nav>
        {room?.live && (
          <span className="draft-topbar-league">
            {room.teams} teams
            {room.settings?.scoring_format === 'ppr' ? ' · PPR' : ''}
            {room.humans ? ` · ${room.humans} of ${room.teams} seats human` : ''}
          </span>
        )}
        <span className="draft-topbar-spacer" />
        {live && (
          <Link className="draft-topbar-link" to="/draft">Go to your board</Link>
        )}
        {room?.live && (
          <span className="draft-topbar-pick mono">
            PICK <strong>{(room.picks_made ?? 0) + 1}</strong> / {room.picks_total}
          </span>
        )}
        {/* The room's own connection pill, saying the one true thing about
            this page: it is watching a draft, not holding a seat in one. */}
        <span className={`draft-status-pill${replaying ? '' : ' draft-status-pill-ok'}`}>
          <span className="draft-status-dot" aria-hidden="true" />
          {room?.warming ? 'Reading the room'
            : !room?.live ? 'Between drafts'
            : replaying ? 'Replay · real ESPN mock'
            : 'Live mock draft'}
        </span>
      </header>

      <div className="draft-body">
        <div className="draft-main">
          {waiting ? (
            <div className="draft-board-building" role="status">
              <span className="draft-board-building-dot" aria-hidden="true" />
              {room === null || room.warming
                ? 'Reading a live ESPN mock draft…'
                : 'No mock draft is running, and there is none on file to replay.'}
            </div>
          ) : tab === 'available' ? (
            <>
              <TopThree
                candidates={candidates}
                players={players}
                onDraft={() => { /* not your room */ }}
                isMyTurn={false}
                horizonLabel={null}
                nextPickLabel={nextPick === null ? null : `pick ${nextPick}`}
                pickNo={thisPick}
                settings={room?.settings ?? null}
                recompute={null}
                onOpenPlayer={openCandidate}
              />
              <AvailableList
                candidates={candidates}
                players={players}
                onDraft={() => { /* not your room */ }}
                isMyTurn={false}
                onOpenPlayer={openCandidate}
                draftedIds={new Set<string>()}
                settings={room?.settings ?? null}
              />
            </>
          ) : room?.board ? (
            <div className="board-tab">
              <DraftBoardGrid
                board={room.board}
                onOpenPlayer={(player) => setOpen({
                  playerId: player.player_id,
                  seed: seedFromBoardPlayer(player, players[player.player_id]),
                })}
              />
            </div>
          ) : (
            <p className="rail-empty draft-main-placeholder">Waiting for the board…</p>
          )}
        </div>

        <aside className="draft-rail">
          {state ? (
            <>
              {/* The room's clock panel, minus the half of it that is about
                  holding a seat. `ClockPanel` itself is not reused because
                  three of the things it draws are facts about YOUR session --
                  an autodraft switch, a socket warning, how long ago your
                  browser last polled -- and for somebody watching strangers
                  draft they are blank or false. What is left is true for an
                  observer and is drawn in the panel's own classes, so the
                  rail is the rail: the same 48px countdown, the same bar,
                  the same three figures underneath.

                  `--:--` rather than a hidden clock when the countdown is
                  unknown, which is the room's own convention for "no clock
                  feed" -- a zeroed bar under it would read as "time just ran
                  out", the opposite of what it means. */}
              <div className="clock-panel">
                <div className="draft-cap">Waiting on seat {room?.on_the_clock}</div>
                <div className="clock-headline">
                  <div className="clock-countdown mono">
                    {secondsLeft === null ? '--:--' : mmss(secondsLeft)}
                  </div>
                </div>
                {secondsLeft !== null && (
                  <div className="clock-bar-track" aria-hidden="true">
                    <div
                      className="clock-bar-fill"
                      style={{ width: `${Math.min(100, (secondsLeft / (turn?.length || 1)) * 100)}%` }}
                    />
                  </div>
                )}
                <div className="clock-figures">
                  <div>
                    <div className="draft-cap">This pick</div>
                    <div className="clock-figure mono">
                      {thisPick} · RD {room?.round}
                    </div>
                  </div>
                  <div>
                    <div className="draft-cap">Next pick</div>
                    <div className="clock-figure mono">
                      {nextPick === null ? '—'
                        : `${nextPick} · RD ${Math.ceil(nextPick / (room?.teams || 1))}`}
                    </div>
                  </div>
                  <div>
                    <div className="draft-cap">Gap</div>
                    <div className="clock-figure mono">
                      {nextPick === null ? '—' : `${nextPick - thisPick} picks`}
                    </div>
                  </div>
                </div>
              </div>
              {room?.roster && room.roster.length > 0 && (
                <RosterPanel
                  // Whose roster it is. The panel says "My roster" in the
                  // room because there it is; here it is seat 5's, and a
                  // label claiming otherwise would be the one dishonest
                  // thing on this page.
                  label={`Seat ${room.on_the_clock} roster`}
                  slots={room.roster}
                  onOpenPlayer={(player) => setOpen({
                    playerId: player.player_id,
                    seed: players[player.player_id]
                      ? seedFromPlayer(players[player.player_id])
                      : { name: player.name, position: player.position,
                          team: null, bye: null, rookie: false, figures: [] },
                  })}
                />
              )}
            </>
          ) : (
            <p className="rail-empty draft-rail-loading">Loading…</p>
          )}
        </aside>
      </div>

      {room?.board && (
        <PickTicker
          board={room.board}
          onOpenPlayer={(player) => setOpen({
            playerId: player.player_id,
            seed: seedFromBoardPlayer(player, players[player.player_id]),
          })}
        />
      )}

      {open && (
        <PlayerOverlay
          target={open}
          settings={room?.settings ?? null}
          onTheClock={false}
          onClose={() => setOpen(null)}
          onSelectPlayer={(id) => setOpen({
            playerId: id,
            seed: players[id] ? seedFromPlayer(players[id]) : open.seed,
          })}
        />
      )}
    </div>
  )
}
