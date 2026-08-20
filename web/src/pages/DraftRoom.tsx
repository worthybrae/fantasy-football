import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchBoard, fetchLiveState, fetchPlayers, selectPlayer, setAutodraft,
         type BoardPlayer, type LiveBoard,
         type LiveCandidate, type LiveSettings, type LiveState, type Player,
         type RosterPlayer } from '../api'
import ClockPanel from '../components/draft/ClockPanel'
import RosterPanel, { type RosterSlot } from '../components/draft/RosterPanel'
import TopThree from '../components/draft/TopThree'
import AvailableList from '../components/draft/AvailableList'
import ConfirmPick, { type PickStatus } from '../components/draft/ConfirmPick'
import PickTicker from '../components/draft/PickTicker'
import PlayerOverlay, { type OverlayTarget } from '../components/draft/PlayerOverlay'
import { seedFromBoardPlayer, seedFromCandidate, seedFromPlayer } from '../components/draft/playerSeed'
import DraftBoardGrid from '../components/DraftBoardGrid'

const POLL_MS = 2500

type Tab = 'available' | 'board'

// Canonical display order for the starter positions ESPN's own settings
// dict (session.settings.starters, api/live.py's `settings` key) carries no
// guaranteed order for -- fixes one so the roster panel always reads QB
// before RB before WR, matching the design mock and ordinary fantasy
// convention, regardless of the order the server's dict happened to build
// its keys in.
const POSITION_ORDER = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']

// FLEX-eligible positions, matching scoring/draft_sim.py's FLEX_POSITIONS
// exactly. Not a new rule: a flex slot can only ever be filled by one of
// these three server-side (see LeagueSettings.replacement_ranks and
// _lineup_assignment); mirrored here because /api/live/state's my_roster
// carries only a flat, pick-order list of drafted players, not the
// session's own lineup assignment, so the rail has to make the same call
// itself to decide which open slot a given pick fills.
const FLEX_ELIGIBLE = new Set(['RB', 'WR', 'TE'])

// This session's real roster shape, as slot labels: `QB`, `RB1`/`RB2` for a
// 2-RB league, `FLEX1..FLEXn`, then `BN1..BNn`. Replaces the two hardcoded
// 8-team/15-round shapes this file and ClockPanel.tsx used to each carry
// (STARTER_SLOTS/BENCH_SLOTS here, LEAGUE_TEAMS in both) -- both were wrong
// the moment a real ESPN league's settings differed from the cold-start
// default, which is the whole defect this task closes. Any starters
// position outside POSITION_ORDER (an exotic ESPN slot this room doesn't
// specifically name) still gets its own labelled slot, appended after the
// ones this list does know about, so the roster's total slot count always
// matches settings.rounds -- an unlabelled position is a display gap, not a
// missing roster spot.
function rosterSlotLabels(settings: LiveSettings): string[] {
  const order = [...POSITION_ORDER,
    ...Object.keys(settings.starters).filter((p) => !POSITION_ORDER.includes(p))]
  const starters: string[] = []
  for (const pos of order) {
    const n = settings.starters[pos] ?? 0
    for (let i = 1; i <= n; i++) starters.push(n > 1 ? `${pos}${i}` : pos)
  }
  const flex = Array.from({ length: settings.flex_slots ?? 0 }, (_, i) => `FLEX${i + 1}`)
  const bench = Array.from({ length: settings.bench ?? 0 }, (_, i) => `BN${i + 1}`)
  return [...starters, ...flex, ...bench]
}

// Fills each drafted player (my_roster, already in pick order per
// api/live.py's _my_roster) into the first open slot his position can take:
// his own position's dedicated slot(s) first, then a FLEX slot if he is
// FLEX-eligible and none of his own remain, then the bench. This is a
// display-only greedy assignment for the rail, not the optimal starting
// lineup scoring/draft_sim.py's _lineup_assignment computes server-side
// (which assigns by highest points, not draft order) -- the rail only needs
// to show which slots are spoken for, not which lineup scores best.
function assignRoster(labels: string[], myRoster: RosterPlayer[]): RosterSlot[] {
  const filled: (RosterPlayer | null)[] = labels.map(() => null)
  const openIndexWhere = (test: (label: string) => boolean) =>
    labels.findIndex((label, i) => filled[i] === null && test(label))

  for (const player of myRoster) {
    let target = openIndexWhere((label) => label.replace(/\d+$/, '') === player.position)
    if (target === -1 && FLEX_ELIGIBLE.has(player.position)) {
      target = openIndexWhere((label) => label.startsWith('FLEX'))
    }
    if (target === -1) {
      target = openIndexWhere((label) => label.startsWith('BN'))
    }
    // No open slot left at all: my_roster and this session's own roster
    // shape disagree (should not happen -- both are read off the same
    // session), skipped rather than crashing the rail over a display gap.
    if (target !== -1) filled[target] = player
  }

  return labels.map((label, i) => {
    const bench = label.startsWith('BN')
    // A slot the roster still needs a starter for reads in accent (see the
    // design mock) -- bench slots never do, whether empty or not, since an
    // empty bench spot is not a gap in anyone's starting lineup.
    return { slot: label, player: filled[i], urgent: filled[i] === null && !bench }
  })
}

// The order the rail READS in, which is deliberately not the order
// `rosterSlotLabels` builds: the owner wants FLEX sitting with the RB/WR
// group it is actually filled from (QB, RB1, RB2, WR1, WR2, FLEX1..n, TE,
// K, DST, BN1..n), not stranded after DST where the build order leaves it.
// Note this is also a deliberate departure from .design/Main.dc.html, whose
// own mock roster lists TE before FLEX -- the owner's stated reading
// preference wins over the mock here.
//
// Kept strictly separate from the build order rather than reordering
// POSITION_ORDER/rosterSlotLabels, because that list is what assignRoster
// walks: `openIndexWhere` takes the FIRST open slot matching each
// predicate, so the array's order decides which slot a player lands in
// (RB1 before RB2, FLEX1 before FLEX2). Reordering it to taste would be
// changing an assignment rule to change a display. Sorting the ASSIGNED
// slots afterwards cannot: the players are already placed by then.
const SLOT_DISPLAY_ORDER = ['QB', 'RB', 'WR', 'FLEX', 'TE', 'K', 'DST']

// Rank for the sort above. Anything this list doesn't name -- an exotic
// ESPN starter slot rosterSlotLabels appended for itself -- sorts after the
// named starters and before the bench, which is where the build order
// already put it; the bench always sorts last.
function displayRank(slot: string): number {
  const base = slot.replace(/\d+$/, '')
  if (base === 'BN') return SLOT_DISPLAY_ORDER.length + 1
  const i = SLOT_DISPLAY_ORDER.indexOf(base)
  return i === -1 ? SLOT_DISPLAY_ORDER.length : i
}

// Display order only -- see SLOT_DISPLAY_ORDER. Array.prototype.sort is
// stable (guaranteed since ES2019), so slots that share a rank keep the
// order assignRoster gave them: RB1 stays above RB2, FLEX1 above FLEX2.
function orderForDisplay(slots: RosterSlot[]): RosterSlot[] {
  return [...slots].sort((a, b) => displayRank(a.slot) - displayRank(b.slot))
}

const SCORING_LABEL: Record<'ppr' | 'half' | 'std', string> = {
  ppr: 'PPR', half: 'Half PPR', std: 'Standard',
}

export default function DraftRoom() {
  const [state, setState] = useState<LiveState | null>(null)
  const [error, setError] = useState<string | null>(null)
  // The board grid's own poll and error slot, kept apart from `state`/`error`
  // above -- lifted from the deleted LiveDraft.tsx's own two-slot poll (see
  // the effect below). A hiccup fetching /api/live/board must not blank the
  // recommendation (`state`/`candidates`, still fine), and a hiccup fetching
  // /api/live/state must not blank the board (`board` is only ever written
  // on a *successful* fetchBoard, never cleared on a failed one -- see the
  // catch below) -- one slow endpoint should never make the other tab look
  // broken.
  const [board, setBoard] = useState<LiveBoard | null>(null)
  const [boardError, setBoardError] = useState<string | null>(null)
  const [players, setPlayers] = useState<Record<string, Player>>({})
  // Which of the two main-column views is showing. The toggle for it lives
  // in the top bar (see the header below); the auto-switch that jumps back
  // to Available the moment you come on the clock is in the poll.
  const [tab, setTab] = useState<Tab>('available')

  // The on_the_clock slot seen on the *previous* successful poll, so the
  // auto-switch below can fire on the transition into your turn rather than
  // on the condition "it's your turn" -- the latter would re-fire on every
  // 2.5s poll for as long as you stay on the clock and force you back to
  // Available even if you deliberately switched to the board mid-pick to
  // check something, which is the worst possible moment to yank the view
  // away. `undefined` (not `null`) marks "no poll observed yet in this
  // session" -- distinct from `null`, which is a real value meaning "nobody
  // is on the clock" -- so the very first poll of a session that happens to
  // load with you already on the clock (a reload or reconnect mid-turn)
  // is recorded, not treated as a transition; `tab` already defaults to
  // 'available' on mount, so that case needs no help from here. Reset to
  // `undefined` whenever the session goes inactive (see the poll below) so a
  // slot left over from a *previous* draft can never masquerade as a real
  // transition in the next one.
  const prevOnClockRef = useRef<number | null | undefined>(undefined)

  // The pick this session is confirming, plus how that confirmation is
  // going -- held here (not inside ConfirmPick) because a poll landing
  // mid-confirm must not lose track of what's being sent, and because the
  // 'sending' request itself (selectPlayer) has to survive whatever
  // AvailableList/TopThree re-render around it. null means no dialog is
  // open at all; ConfirmPick only ever mounts while this is non-null (see
  // the render below), so it never has to handle a null candidate itself.
  const [confirming, setConfirming] = useState<LiveCandidate | null>(null)
  const [pickStatus, setPickStatus] = useState<PickStatus>('idle')
  const [pickError, setPickError] = useState<string | null>(null)

  // The autodraft round trip, held here for the same reasons the pick
  // confirmation above is: the 2.5s poll must not lose a request in flight,
  // and the request has to outlive the re-renders around it. Note what is
  // NOT here -- the flag itself. That lives in `state.autodraft`, straight
  // off ESPN via the poll, so nothing in this room can show an autodraft
  // state ESPN did not confirm. Same rule the pick dialog follows, and it
  // matters more here: an owner who wrongly believes they took autodraft
  // back off will stop watching the clock.
  const [autodraftPending, setAutodraftPending] = useState(false)
  const [autodraftError, setAutodraftError] = useState<string | null>(null)

  // The player profile open over the room, or null. Held here rather than
  // in the URL on purpose: a route swap unmounts this whole component --
  // the board, the tab, the scroll position, the 2.5s poll -- and clicking
  // a name mid-draft must cost none of those. Nothing below this line
  // navigates; the /players/:slug page stays for direct links (see
  // PlayerOverlay's own comment, and DraftBoardGrid, which keeps the href
  // so cmd-click still opens it).
  const [openPlayer, setOpenPlayer] = useState<OverlayTarget | null>(null)

  // Player identity (name/position/team) is a one-time join table -- it does
  // not change mid-draft. Lifted unchanged from the deleted LiveDraft.tsx.
  useEffect(() => {
    let cancelled = false
    fetchPlayers()
      .then((list) => {
        if (cancelled) return
        const map: Record<string, Player> = {}
        for (const p of list) map[p.player_id] = p
        setPlayers(map)
      })
      .catch(() => {
        // Candidates still render by player_id; the join table just stays
        // empty and names fall back to that id.
      })
    return () => {
      cancelled = true
    }
  }, [])

  // 2.5s poll of /api/live/state, lifted from the deleted LiveDraft.tsx --
  // now joined by that same file's /api/live/board poll, in the one loop
  // (own try/catch per fetch, own state/error slot per fetch, see LiveDraft
  // git history) so a hiccup in either endpoint can't blank the other's
  // tab.
  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const data = await fetchLiveState()
        if (!cancelled) {
          setState(data)
          setError(null)
          if (!data.active) {
            // No session: whatever slot was on the clock belonged to a
            // *previous* draft, if any -- must not be compared against next
            // session's own on_the_clock as though it were a real edge.
            prevOnClockRef.current = undefined
          } else {
            const prevOnClock = prevOnClockRef.current
            prevOnClockRef.current = data.on_the_clock
            if (prevOnClock !== undefined && data.my_slot !== null &&
                data.on_the_clock === data.my_slot && prevOnClock !== data.my_slot) {
              setTab((t) => (t === 'board' ? 'available' : t))
            }
          }
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load live state')
      }
      try {
        const data = await fetchBoard()
        if (!cancelled) {
          setBoard(data)
          setBoardError(null)
        }
      } catch (e) {
        if (!cancelled) {
          setBoardError(e instanceof Error ? e.message : 'Failed to load the draft board')
        }
      }
    }

    poll()
    const id = window.setInterval(poll, POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  // The tab group's hint (see the mock), now riding in the top bar beside
  // the tabs themselves. This is also `players`' only read site in this
  // task: the join table is fetched here for Task 8's AvailableList/TopThree
  // to consume once they exist, but nothing else in this shell needs a
  // player's name or team yet.
  const playerCount = Object.keys(players).length
  const tabHint = playerCount > 0 ? `${playerCount} players in the pool` : ''

  const listenerOk = !!state?.active && state.listener_alive && state.listener_error === null
  // Two separate failures, named separately. A detached send socket
  // (`socket_alive` false) is not a dead listener: frames keep arriving, the
  // board keeps filling, and the only thing that stops working is picking --
  // which is precisely the failure the old single `listenerOk` pill hid
  // behind a green "ESPN LIVE" while every draft click 503'd. It is also the
  // permanent state of the browser-observer path (/api/live/connect
  // publishes no SocketHandle), where "you cannot draft from here" is the
  // honest label rather than a fault.
  const socketOk = !!state?.active && state.socket_alive
  const connectionLabel = !state?.active
    ? 'NOT CONNECTED'
    : !listenerOk ? 'LISTENER DOWN' : !socketOk ? 'SOCKET DOWN' : 'ESPN LIVE'
  const connectionTone = !state?.active ? 'off' : (listenerOk && socketOk) ? 'ok' : 'fail'

  // ms_remaining straight off state -- never decayed or interpolated
  // between polls, see ClockPanel.tsx's own comment on the ticker for why
  // not. null (state not loaded yet, session inactive, or no CLOCK/SELECTING
  // frame seen) reaches ClockPanel unchanged, which renders it as `--:--`.
  const secondsLeft = state?.ms_remaining != null ? Math.round(state.ms_remaining / 1000) : null

  const totalPicks = state?.active
    ? (state.settings.teams as number) * (state.settings.rounds as number)
    : null

  // Same terminal-state rule as ClockPanel's thisPickNo (Task 6+7 review
  // finding #4): once the draft is over (on_the_clock === null) there is
  // no "current pick" to name. This sibling counter was left unguarded in
  // the first pass at this task -- picks_made + 1 rendered unconditionally,
  // so a finished 120-pick league read "PICK 121 / 120." null here falls
  // back to the same em dash ClockPanel's own "This pick" figure uses.
  const thisPickNo = state?.active && state.on_the_clock !== null
    ? state.picks_made + 1
    : null

  // assignRoster first (build order -- it decides WHICH slot each player
  // fills), then orderForDisplay (reading order -- it only decides which
  // row each already-assigned slot appears in). Never the other way round.
  const slots: RosterSlot[] = state?.active
    ? orderForDisplay(assignRoster(rosterSlotLabels(state.settings), state.my_roster))
    : []

  // Gates both the top-three and the table's draft buttons. Computed once
  // here rather than inside either component, which have no access to
  // `state` at all (see AvailableList.tsx's own comment on why this prop
  // exists beyond what the task brief's original signature listed).
  //
  // `socket_alive` is part of the test, not just whose turn it is: POST
  // /api/live/select answers 503 with no live handle, and during a
  // run_socket_listener reconnect the handle is detached while the listener
  // thread is alive and `stale` has not tripped (CLOCK frames stamped
  // last_poll_at a second ago), so "on turn, socket down" was
  // indistinguishable from "on turn, socket up" and every click 503'd. Spec
  // section 6: the buttons disable while the socket is down and re-enable
  // when the handle reports alive again -- which happens on its own here,
  // since this is recomputed from every 2.5s poll. Deliberately NOT the same
  // test as ClockPanel's `youAreUp`, which is about whose turn it is and
  // stays true while the socket is down.
  const isMyTurn = !!state?.active && state.on_the_clock !== null
    && state.on_the_clock === state.my_slot && state.socket_alive

  // The list on screen was computed for an older pick than the one on the
  // clock: a pick has landed and its recompute has not finished (0.1-0.8s of
  // ranking plus up to 2.5s of poll lag). Presenting it as current would
  // recommend a player who may already be gone -- exactly what
  // _recompute's own as_of_pick guard exists to prevent server-side. Both
  // counts are pick COUNTS, so the pick each is FOR is that + 1, the same +1
  // `thisPickNo` applies. Restored from the deleted LiveDraft.tsx, whose
  // `isRecomputing` banner went with it.
  const rankedForPickNo = state?.active && state.candidates_as_of_pick !== null
    && state.candidates_as_of_pick < state.picks_made
    ? state.candidates_as_of_pick + 1
    : null

  // The same two pick numbers the old banner named, handed to TopThree to
  // render on its own header row instead of as a strip of its own (see the
  // comment where that banner used to be). `state.picks_made + 1` rather
  // than `thisPickNo`, deliberately: `thisPickNo` is null once the draft is
  // over, and a recompute outstanding at that moment is still recomputing
  // for a real pick number.
  const recompute = rankedForPickNo !== null && state?.active
    ? { forPick: state.picks_made + 1, listedForPick: rankedForPickNo }
    : null

  // The pick the ranked list was actually measured against, named exactly
  // as the server reports it. NOT derived here from `nextPickFor` any more:
  // that answers "which pick do I take next", and since the horizon skips
  // turns too close to carry any signal (a wheel, or the 1-pick gap that
  // put a kicker 6th at pick 1 -- see scoring/draft_sim.horizon_picks) the
  // two genuinely differ. Deriving it client-side would caption the list
  // with a pick it was not measured against, which is worse than saying
  // nothing. ClockPanel keeps its own `nextPickFor` because "when do I pick
  // next" really is that question.
  //
  // null -> TopThree drops the clause entirely rather than guessing:
  // `horizon_pick` is null both before any gain-ranked list exists and when
  // the horizon ran off the end of the draft, and the second case is the
  // one `horizon_is_end_of_draft` names instead of inventing a pick number.
  const horizonLabel = state?.active
    ? state.horizon_is_end_of_draft
      ? 'the end of the draft'
      : state.horizon_pick !== null ? `pick ${state.horizon_pick}` : null
    : null

  // "Roster after" for the confirm dialog: my_roster's current length plus
  // this pick, over the room's own total slot count (`slots`, already built
  // above for RosterPanel) -- both read off state that's already in scope
  // here, so ConfirmPick never has to reconstruct them itself.
  const rosterAfter = state?.active
    ? { filled: state.my_roster.length + 1, total: slots.length }
    : null

  // Three ways in, one overlay. Each seeds from the row that was actually
  // clicked (see playerSeed.ts) so the profile paints on this frame rather
  // than after the 3.5s /profile request -- the request is still issued by
  // PlayerProfile itself and fills the rest in when it lands.
  function handleOpenCandidate(c: LiveCandidate) {
    setOpenPlayer({ playerId: c.player_id, seed: seedFromCandidate(c, players[c.player_id]) })
  }

  function handleOpenBoardPlayer(p: BoardPlayer) {
    setOpenPlayer({ playerId: p.player_id, seed: seedFromBoardPlayer(p, players[p.player_id]) })
  }

  // A comp clicked inside the profile. Only swaps for a player this room can
  // actually name -- the same rule PlayerPage's own onSelectPlayer follows
  // (it navigates only when the id resolves against the board list), and it
  // matters more here: SimilarPlayers legitimately lists players who are not
  // in this season's pool at all, and there is no seed to paint for one. No
  // seed, no instant open, so the click does nothing rather than opening an
  // empty box with a spinner in it.
  function handleSelectPlayer(id: string) {
    const p = players[id]
    if (p) setOpenPlayer({ playerId: id, seed: seedFromPlayer(p) })
  }

  function handleDraftClick(c: LiveCandidate) {
    setConfirming(c)
    setPickStatus('idle')
    setPickError(null)
  }

  function handleCancelConfirm() {
    setConfirming(null)
    setPickStatus('idle')
    setPickError(null)
  }

  async function handleConfirmPick() {
    if (!confirming) return
    setPickStatus('sending')
    setPickError(null)
    try {
      await selectPlayer(confirming.player_id)
      // Confirmed by ESPN. No optimistic mutation of `state.candidates`
      // anywhere in this room (per the task brief) -- clear the dialog
      // outright and let the next 2.5s poll of /api/live/state bring the
      // real pick back, the same path a pick made directly in ESPN's own
      // UI would take. Both state updates land together so ConfirmPick's
      // own 'done' branch (see its comment) never actually has a frame to
      // paint.
      setPickStatus('done')
      setConfirming(null)
    } catch (e) {
      // A 504 here is api/live.py's own "ESPN did not confirm the pick --
      // check the ESPN draft room before picking again" -- it means "we
      // could not confirm it," not "it failed," and the pick may or may not
      // have actually landed. selectPlayer's Error already carries that
      // exact message (or a 409's "already drafted" / "not your turn", or a
      // 503's "draft socket is not connected") -- rendered verbatim by
      // ConfirmPick rather than replaced with a generic message here.
      setPickStatus('failed')
      setPickError(e instanceof Error ? e.message : 'Pick failed')
    }
  }

  async function handleSetAutodraft(on: boolean) {
    // Guarded here as well as by the switch's own `disabled`: a second
    // request would race the first one's confirmation, and whichever
    // resolved last would decide what the error slot said about a state
    // neither of them owns any more.
    if (autodraftPending) return
    setAutodraftPending(true)
    setAutodraftError(null)
    try {
      await setAutodraft(on)
      // ESPN confirmed. Nothing is written into `state.autodraft` here --
      // that value only ever comes from the poll. Pulling the state endpoint
      // once immediately is not optimism, it is the opposite: it fetches
      // ESPN's confirmed value rather than waiting up to POLL_MS to see it,
      // so the switch moves on the same fact the request returned on instead
      // of sitting in its old position for two and a half seconds after a
      // click that worked. Best effort -- a failed refetch changes nothing,
      // since the interval poll is still running.
      try {
        setState(await fetchLiveState())
      } catch {
        // The 2.5s poll will bring it.
      }
    } catch (e) {
      // The server's own message, rendered verbatim by ClockPanel: a 504
      // here is api/live.py's "ESPN did not confirm the autodraft change",
      // which means the command may or may not have landed -- replacing it
      // with a generic "failed" would lose exactly that.
      setAutodraftError(e instanceof Error ? e.message : 'Could not change autodraft')
    } finally {
      setAutodraftPending(false)
    }
  }

  return (
    <div className="draft-room">
      <div className="sr-only" aria-live="polite">
        {!state ? 'Loading' : !state.active ? 'Not connected' : connectionLabel}.
      </div>

      <header className="draft-topbar">
        {/* The room's only way out. App.tsx routes one way (/ -> /draft) and
            nothing here linked back, so a user whose listener died had no
            route to the page that hands over the bookmarklet -- the one
            thing that reconnects them -- short of editing the URL bar under
            a pick clock. The title carries it rather than adding a control:
            same layout, same words, now clickable. ClockPanel's listener-down
            panel links here too, with the instruction. */}
        <Link to="/" className="draft-topbar-title">Draft Helper</Link>
        <span className="draft-topbar-sep" aria-hidden="true" />
        {/* The two views, in the top bar rather than on a strip of their own
            below it. They were a 38px `.draft-tabs` row spanning the main
            column, which cost the ranked list a row of height and put a
            second horizontal rule directly under the first one for the sake
            of two buttons. Same buttons, same handlers, same `.draft-tab`
            styling -- they are full-bar height here so the active tab's
            underline lands on the top bar's own bottom border, which is what
            makes them still read as tabs over the region they switch. */}
        <nav className="draft-topbar-tabs" aria-label="Draft views">
          <button
            type="button"
            className={`draft-tab${tab === 'available' ? ' is-active' : ''}`}
            onClick={() => setTab('available')}
          >
            Available
          </button>
          <button
            type="button"
            className={`draft-tab${tab === 'board' ? ' is-active' : ''}`}
            onClick={() => setTab('board')}
          >
            Snake Board
          </button>
        </nav>
        {/* Rendered on both tabs, not just Available. The pool size does not
            change mid-draft, so a hint that came and went with the tab would
            shift the league line and everything after it sideways every time
            the view was switched -- under a clock, in the bar that carries
            the pick counter. */}
        {tabHint && <span className="draft-topbar-hint">{tabHint}</span>}
        {/* Gated on the same condition as the league line it divides off --
            an unconditional rule here would leave a stray tick of border
            hanging after the tabs on a room with no session behind it. */}
        {state?.active && <span className="draft-topbar-sep" aria-hidden="true" />}
        {state?.active && (
          <span className="draft-topbar-league">
            {state.settings.teams} teams
            {state.settings.scoring_format && ` · ${SCORING_LABEL[state.settings.scoring_format]}`}
          </span>
        )}
        {/* Narrowed, not removed. This used to read "ranked on full-PPR
            points regardless", which was true: projections() priced every
            league in full PPR, so proj_points -- and therefore vor, and
            therefore every number this room ranks on -- ignored the league's
            scoring. That is fixed; both rungs of the ladder now follow
            settings.scoring.

            What is left is narrower and still worth saying. ESPN publishes
            one season projection per player, computed under its own PPR
            default, and espn_adp stores only the total -- no projected
            receptions or yards to re-price. So for a non-PPR league that rung
            is CONVERTED (scoring/board.projection_scale: the ratio of the
            player's last season under this league's rules to the same season
            in full PPR), which is exact only if his projected stat mix
            matches last season's. The chip beside this says "Half PPR"; this
            says how far to trust it. scoring/board.py carries the same
            warning to server stderr, where nobody drafting in a browser will
            ever see it. */}
        {state?.active && state.settings.scoring_format !== null
          && state.settings.scoring_format !== 'ppr' && (
          <span className="draft-topbar-note">
            ESPN projections converted from PPR
          </span>
        )}
        <span className="draft-topbar-spacer" />
        {state?.active && totalPicks !== null && (
          <span className="draft-topbar-pick mono">
            PICK <strong>{thisPickNo ?? '—'}</strong> / {totalPicks}
          </span>
        )}
        <span className={`draft-status-pill draft-status-pill-${connectionTone}`}>
          <span className="draft-status-dot" aria-hidden="true" />
          {connectionLabel}
        </span>
      </header>

      {error && <p className="error draft-error-banner">{error}</p>}

      {/* Ranking has stopped while everything else keeps working. The
          listener is fine, the clock ticks, the board fills -- and the list
          below is frozen at whatever pick it last reached. Nothing said so
          before api/live.py grew `recompute_error`; `listener_alive` tracks
          the other thread entirely. role="alert" for the same reason the
          unmapped banner has one: this changes what the numbers below mean. */}
      {state?.active && state.recompute_error !== null && (
        <p className="draft-error-banner" role="alert">
          Ranking has stopped ({state.recompute_error}). The list below is
          {' '}frozen{state.candidates_as_of_pick !== null
            ? ` at pick ${state.candidates_as_of_pick + 1}`
            : ' and was never computed'} -- picks are still being recorded,
          {' '}but nothing below is being re-ranked.
        </p>
      )}

      {/* A pick the socket confirmed that the crosswalk could not resolve to
          a board player. He is off the board in ESPN and still sitting in the
          ranked list here, recommendable, with `survive_pct` counting him as
          available. api/live.py has served these since Task 5 and the room
          rendered them nowhere -- a regression against the deleted
          LiveDraft.tsx, which had exactly this banner. Spec section 6 asks
          for it by name as the surface a crosswalk-gap 400 should also
          appear in. */}
      {state?.active && state.unmapped_picks.length > 0 && (
        <div className="draft-alert-banner" role="alert">
          <strong>
            {state.unmapped_picks.length} pick
            {state.unmapped_picks.length === 1 ? '' : 's'} could not be matched
            to a player
          </strong>
          {' '}-- they are off the board in ESPN but may still be listed as
          available below, and ranked as though nobody had taken them.
          <ul className="draft-unmapped-list">
            {state.unmapped_picks.map((u) => (
              <li key={u.overall_pick} className="mono">
                pick {u.overall_pick} · espn id {u.espn_player_id}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* The "recomputing for pick N -- the list below is still for pick M"
          banner used to render here, between the alert strip and
          `.draft-body`. It is now TopThree's own header row (see `recompute`
          below and TopThree's prop comment): a banner that mounts and
          unmounts between polls adds a strip of height and takes it away
          again, and everything below it -- including the Draft button the
          cursor is already resting on -- moves with it. Same information,
          same role="status", no reflow. */}

      <div className="draft-body">
        <div className="draft-main">
          {tab === 'available'
            ? (
              // TopThree ABOVE AvailableList's own filter row/table --
              // the mock draws the filter row first (search+pills, then
              // top three, then the table). Deliberately not matched:
              // under a 30-second clock the recommendation is what the
              // eye needs first, and the search/position filter is a
              // browsing tool that belongs attached to the table it
              // filters, which is where AvailableList already puts it.
              // Reviewed and confirmed as a conscious deviation from the
              // mock, not an oversight -- do not "fix" this back to
              // match the mock's own ordering.
              <>
                <TopThree
                  candidates={state?.candidates ?? []}
                  players={players}
                  onDraft={handleDraftClick}
                  isMyTurn={isMyTurn}
                  horizonLabel={horizonLabel}
                  recompute={recompute}
                  onOpenPlayer={handleOpenCandidate}
                />
                <AvailableList
                  candidates={state?.candidates ?? []}
                  players={players}
                  onDraft={handleDraftClick}
                  isMyTurn={isMyTurn}
                  horizonLabel={horizonLabel}
                  onOpenPlayer={handleOpenCandidate}
                />
              </>
            )
            : board?.active
              ? (
                // `.board-tab` only pads the grid, same padding the mock's
                // own snake-board panel uses -- `.board-wrap` itself is
                // DraftBoardGrid's own horizontal scroll container (per
                // its own CSS comment), and it flows straight in
                // `.draft-main`'s existing overflow-y: auto region rather
                // than opening a second, nested vertical scroller -- the
                // same "no second scroll container" call the ranked
                // available table already makes (see App.css, the comment
                // above `.avail-col-rank`) so the page itself never
                // scrolls and there is exactly one vertical scroll
                // container per tab.
                <div className="board-tab">
                  {boardError && <p className="error draft-error-banner">{boardError}</p>}
                  <DraftBoardGrid board={board} onOpenPlayer={handleOpenBoardPlayer} />
                </div>
              )
              // No board yet: either still loading (boardError null) or
              // the fetch itself failed before a first board ever landed
              // (nothing stale to fall back to, unlike the branch above).
              : (
                <p className="rail-empty draft-main-placeholder">
                  {boardError ?? 'Waiting for the draft board…'}
                </p>
              )}
        </div>

        <aside className="draft-rail">
          {state ? (
            <>
              {/* board: Defect 4's "waiting on {team name}" lookup -- this
                  room already polls /api/live/board every 2.5s (see the
                  effect above), so ClockPanel is handed the same state
                  rather than fetching its own. */}
              <ClockPanel
                state={state}
                secondsLeft={secondsLeft}
                board={board}
                autodraftPending={autodraftPending}
                autodraftError={autodraftError}
                onSetAutodraft={handleSetAutodraft}
              />
              {state.active ? (
                <RosterPanel slots={slots} />
              ) : (
                <p className="rail-empty draft-rail-loading">No roster to show.</p>
              )}
            </>
          ) : (
            <p className="rail-empty draft-rail-loading">{error ?? 'Loading…'}</p>
          )}
        </aside>
      </div>

      {/* The room's bottom rail, outside `.draft-body` so it spans the main
          column AND the roster rail and stays put while both scroll inside
          themselves. Fed by the board poll that was already running for the
          snake board -- no second endpoint, no second interval. */}
      <PickTicker board={board} onOpenPlayer={handleOpenBoardPlayer} />

      {/* Before ConfirmPick, deliberately. Both sit on the same z-index tier
          (`.player-overlay` and `.confirm-overlay` are both 40 -- one modal
          pattern in this room, per PlayerOverlay's comment), so DOM order is
          what decides which paints on top, and the answer has to be the
          dialog that sends an irreversible pick. */}
      {openPlayer && (
        <PlayerOverlay
          target={openPlayer}
          onClose={() => setOpenPlayer(null)}
          onSelectPlayer={handleSelectPlayer}
        />
      )}

      {confirming && (
        <ConfirmPick
          candidate={confirming}
          player={players[confirming.player_id] ?? null}
          status={pickStatus}
          error={pickError}
          pickNo={thisPickNo}
          rosterAfter={rosterAfter}
          onConfirm={handleConfirmPick}
          onCancel={handleCancelConfirm}
        />
      )}
    </div>
  )
}
