import { useEffect, useRef, useState } from 'react'
import { fetchBoard, fetchLiveState, fetchPlayers, selectPlayer, type LiveBoard, type LiveCandidate,
         type LiveSettings, type LiveState, type Player, type RosterPlayer } from '../api'
import ClockPanel from '../components/draft/ClockPanel'
import RosterPanel, { type RosterSlot } from '../components/draft/RosterPanel'
import TopThree from '../components/draft/TopThree'
import AvailableList from '../components/draft/AvailableList'
import ConfirmPick, { type PickStatus } from '../components/draft/ConfirmPick'
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

const SCORING_LABEL: Record<'ppr' | 'half' | 'std', string> = {
  ppr: 'PPR', half: 'Half PPR', std: 'Standard',
}

// Duplicated from ClockPanel.tsx (unexported there too, same precedent as
// posBadge's own multi-file duplication) -- needed here only so TopThree's
// hint can name the actual pick number `gain_now` is measured against
// ("...vs. waiting until pick N"). See ClockPanel.tsx's own comment on
// pickNumberFor for the verification against scoring/draft_sim.snake_slots.
function pickNumberFor(round: number, slot: number, teams: number): number {
  return round * teams + (round % 2 === 0 ? slot : teams - slot + 1)
}

function nextPickFor(fromPickNo: number, mySlot: number, teams: number): number {
  const round = Math.floor((fromPickNo - 1) / teams)
  const thisRound = pickNumberFor(round, mySlot, teams)
  return thisRound >= fromPickNo ? thisRound : pickNumberFor(round + 1, mySlot, teams)
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
  // Task 7 owns this state and the toggle handlers, since it renders the tab
  // strip -- Task 9 adds only the auto-switch (jumping to Available the
  // moment you come on the clock) on top of what's here.
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

  // The tab strip's right-aligned hint (see the mock) -- Task 7's, since it
  // owns the strip itself. This is also `players`' only read site in this
  // task: the join table is fetched here for Task 8's AvailableList/TopThree
  // to consume once they exist, but nothing else in this shell needs a
  // player's name or team yet.
  const playerCount = Object.keys(players).length
  const tabHint = playerCount > 0 ? `${playerCount} players in the pool` : ''

  const listenerOk = !!state?.active && state.listener_alive && state.listener_error === null
  const connectionLabel = !state?.active ? 'NOT CONNECTED' : listenerOk ? 'ESPN LIVE' : 'LISTENER DOWN'
  const connectionTone = !state?.active ? 'off' : listenerOk ? 'ok' : 'fail'

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

  const slots: RosterSlot[] = state?.active
    ? assignRoster(rosterSlotLabels(state.settings), state.my_roster)
    : []

  // Same test as ClockPanel's own `youAreUp` -- gates both the top-three and
  // the table's draft buttons. Computed once here rather than inside either
  // component, which have no access to `state` at all (see
  // AvailableList.tsx's own comment on why this prop exists beyond what the
  // task brief's original signature listed).
  const isMyTurn = !!state?.active && state.on_the_clock !== null && state.on_the_clock === state.my_slot

  // The pick TopThree's hint names ("...vs. waiting until pick N") --
  // exactly ClockPanel's own `nextPickNo` (see its comment), recomputed
  // here since ClockPanel keeps that value private to its own render. null
  // whenever `thisPickNo` itself is null (draft inactive/over) or my_slot
  // isn't resolved yet -- TopThree drops the clause rather than guessing.
  const nextPickNo = state?.active && state.my_slot !== null && thisPickNo !== null
    ? nextPickFor(thisPickNo, state.my_slot, state.settings.teams as number)
    : null

  // "Roster after" for the confirm dialog: my_roster's current length plus
  // this pick, over the room's own total slot count (`slots`, already built
  // above for RosterPanel) -- both read off state that's already in scope
  // here, so ConfirmPick never has to reconstruct them itself.
  const rosterAfter = state?.active
    ? { filled: state.my_roster.length + 1, total: slots.length }
    : null

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

  return (
    <div className="draft-room">
      <div className="sr-only" aria-live="polite">
        {!state ? 'Loading' : !state.active ? 'Not connected' : connectionLabel}.
      </div>

      <header className="draft-topbar">
        <span className="draft-topbar-title">Draft Helper</span>
        <span className="draft-topbar-sep" aria-hidden="true" />
        {state?.active && (
          <span className="draft-topbar-league">
            {state.settings.teams} teams
            {state.settings.scoring_format && ` · ${SCORING_LABEL[state.settings.scoring_format]}`}
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

      <div className="draft-body">
        <div className="draft-main-col">
          <div className="draft-tabs">
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
            <span className="draft-tab-spacer" />
            <span className="draft-tab-hint">{tabHint}</span>
          </div>
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
                    nextPickNo={nextPickNo}
                  />
                  <AvailableList
                    candidates={state?.candidates ?? []}
                    players={players}
                    onDraft={handleDraftClick}
                    isMyTurn={isMyTurn}
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
                    <DraftBoardGrid board={board} />
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
        </div>

        <aside className="draft-rail">
          {state ? (
            <>
              <ClockPanel state={state} secondsLeft={secondsLeft} />
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
