import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { ageLabel, fetchMockBoard, fetchMockDrafts,
         type LiveBoard, type MockDraft, type PickMaker } from '../api'
import DraftBoardGrid, { MAKER_MARK } from '../components/DraftBoardGrid'
import { Logo } from '../components/Logo'
import { useDocumentMeta } from '../lib/documentMeta'

// Slow on purpose. A mock room picks every few seconds at most, and the two
// endpoints behind this page read recorded state rather than driving
// anything -- there is no clock to be late for here, unlike /draft's own
// 2.5s poll. A FINISHED draft is never polled at all (see the effects
// below): its board cannot change again.
const POLL_MS = 5000

// Reading order for the tally and the key: people first, machines after,
// unlabelled last. It is the order the question is asked in -- "was anyone
// actually in this room?" -- not the alphabet.
const MAKER_ORDER: PickMaker[] = ['human', 'us', 'auto', 'engine', 'unknown']

const MAKER_NAME: Record<PickMaker, string> = {
  human: 'a person',
  us: 'our bot',
  auto: 'autopicked',
  engine: 'ESPN team',
  unknown: 'not recorded',
}

// The one line of explanation each state gets in the key. DraftBoardGrid's
// own MAKER_LABEL (a cell's hover and aria text) says what happened to a
// single pick; these say what the state MEANS for the room, which is the
// thing the page exists to answer. The GLYPHS are imported from the grid
// rather than restated here -- a key that disagreed with the board it
// explains would be worse than no key.
const MAKER_NOTE: Record<PickMaker, string> = {
  human: 'chose the player themselves',
  us: 'our own seat, which drafts semi-randomly on purpose',
  auto: 'the seat had a person, but they had wandered off and ESPN picked',
  engine: 'the seat never had a person in it at all',
  unknown: 'recorded before the farm labelled picks, so nobody knows',
}

type Tally = Record<PickMaker, number>

// Counted off the board's own cells rather than trusted from a summary
// field, so the numbers over the grid and the shading in it can never
// disagree. A cell with no `made_by` at all (an older recording, or a live
// board that carries no labels) counts as `unknown` -- the same reading
// DraftBoardGrid gives it.
function tallyMakers(board: LiveBoard | null): Tally {
  const tally: Tally = { human: 0, us: 0, auto: 0, engine: 0, unknown: 0 }
  if (!board) return tally
  for (const cell of board.cells) {
    const made = cell.made_by
    tally[made !== undefined && made in tally ? made : 'unknown'] += 1
  }
  return tally
}

function errorText(e: unknown, fallback: string): string {
  return e instanceof Error ? e.message : fallback
}

// "3 of 8 seats human". Null is not zero and must not read as it: a draft
// the farm recorded before it counted seats knows nothing about who was in
// the room, which is a different sentence from "nobody was".
function humanSeatsLabel(draft: MockDraft): string {
  if (draft.human_seats === null) return 'humans not recorded'
  if (draft.human_seats === 0) return `no humans in ${draft.teams} seats`
  return `${draft.human_seats} of ${draft.teams} seats human`
}

// When this draft happened, in the shape that answers "how long ago": a live
// room reads by when it started, a finished one by when the farm recorded it
// -- the two never both carry a value, so this is the one place that
// decides which timestamp the rest of the page shows, and both the row and
// the board head read it from here rather than each picking their own.
function draftWhen(draft: MockDraft): { at: string; label: 'started' | 'recorded' } | null {
  if (draft.started_at) return { at: draft.started_at, label: 'started' }
  if (draft.recorded_at) return { at: draft.recorded_at, label: 'recorded' }
  return null
}

// The headline over the board, in the shape of the question the page is
// asked: was this room worth anything? A machine pick is a machine pick
// whether ESPN was covering for someone who left or filling a seat nobody
// ever took, so the two are summed here and separated in the key below.
// Our own seat is excluded from both sides -- our bot is neither a person
// worth studying nor an ESPN team, and counting it either way would move
// the number by a whole column.
function verdict(tally: Tally): { machine: number; human: number; labelled: number } {
  const machine = tally.auto + tally.engine
  return { machine, human: tally.human, labelled: machine + tally.human + tally.us }
}

function DraftRow({ draft, active, onSelect }: {
  draft: MockDraft
  active: boolean
  onSelect: (id: string) => void
}) {
  const total = draft.teams * draft.rounds
  const pct = total > 0 ? Math.min(100, (draft.picks_made / total) * 100) : 0
  const live = draft.status === 'live'
  const hasHumans = draft.human_seats !== null && draft.human_seats > 0
  const when = draftWhen(draft)
  return (
    <button
      type="button"
      className={`mocks-row${active ? ' is-active' : ''}`}
      aria-current={active ? 'true' : undefined}
      onClick={() => onSelect(draft.id)}
    >
      <span className="mocks-row-top">
        <span className="mocks-row-id mono">{draft.league_id}</span>
        <span className={`mocks-pill ${live ? 'is-live' : 'is-done'}`}>
          {live && <span className="mocks-pill-dot" aria-hidden="true" />}
          {live ? 'LIVE' : 'DONE'}
        </span>
      </span>
      <span className="mocks-row-meta">
        <span className="mono">{draft.picks_made}/{total}</span> picks
        <span className="mocks-row-sep" aria-hidden="true">·</span>
        <span className="mono">{draft.teams}×{draft.rounds}</span>
      </span>
      {/* Progress, not decoration: a live room three rounds in and a live
          room on its last pick are the same word ("LIVE") and want telling
          apart from across the list. */}
      <span className="mocks-row-bar" aria-hidden="true">
        <span className="mocks-row-bar-fill" style={{ width: `${pct}%` }} />
      </span>
      <span className="mocks-row-foot">
        <span className={`mocks-row-humans${hasHumans ? ' has-humans' : ''}`}>
          {humanSeatsLabel(draft)}
        </span>
        <span
          className="mocks-row-when"
          title={when ? new Date(when.at).toLocaleString() : undefined}
        >
          {when ? ageLabel(when.at) : 'no time recorded'}
        </span>
      </span>
    </button>
  )
}

// The proportion bar plus its key. The bar is the answer at a glance and the
// key is what makes the board underneath readable -- same glyphs, same
// fills, so a cell in the grid can be matched back to a line here without
// hunting.
function MakerTally({ tally }: { tally: Tally }) {
  const { machine, human, labelled } = verdict(tally)
  const shown = MAKER_ORDER.filter((m) => tally[m] > 0)
  const total = MAKER_ORDER.reduce((n, m) => n + tally[m], 0)
  if (total === 0) return null
  return (
    <div className="mocks-tally">
      <p className="mocks-verdict">
        {labelled === 0
          ? 'Who made these picks was never recorded.'
          : (
            <>
              <strong className="mocks-verdict-n">{machine}</strong> of {labelled} picks
              {' '}made by a machine
              <span className="mocks-verdict-sep" aria-hidden="true">·</span>
              <strong className={`mocks-verdict-n${human === 0 ? ' is-none' : ''}`}>{human}</strong>
              {' '}by a person
            </>
          )}
      </p>
      <div className="mocks-tally-bar" role="img"
           aria-label={shown.map((m) => `${tally[m]} ${MAKER_NAME[m]}`).join(', ')}>
        {shown.map((m) => (
          <span
            key={m}
            className={`mocks-tally-seg by-fill is-${m}`}
            style={{ flexGrow: tally[m] }}
            title={`${tally[m]} ${MAKER_NAME[m]}`}
          />
        ))}
      </div>
      <ul className="mocks-key">
        {MAKER_ORDER.map((m) => (
          <li
            key={m}
            className={`mocks-key-item${tally[m] === 0 ? ' is-none' : ''}`}
            // The note is clipped to one line so the key stays a key rather
            // than a paragraph; hover gives it back in full.
            title={`${MAKER_NAME[m]} -- ${MAKER_NOTE[m]}`}
          >
            <span className={`mocks-key-swatch by-fill is-${m}`} aria-hidden="true" />
            <span className={`mocks-key-mark is-${m}`} aria-hidden="true">
              {m === 'unknown' ? '' : MAKER_MARK[m]}
            </span>
            <span className="mocks-key-name">{MAKER_NAME[m]}</span>
            <span className="mocks-key-n mono">{tally[m]}</span>
            <span className="mocks-key-note">{MAKER_NOTE[m]}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

// Every mock draft the farm has joined, and the board of whichever one is
// selected -- shaded by who actually made each pick, which is the whole
// reason the page exists: most picks in a public ESPN mock turn out to be
// ESPN's own computer rather than a person, and a board that does not say
// so is worth studying far less than it looks.
export default function MockDrafts() {
  useDocumentMeta({
    title: 'Recorded ESPN mock drafts – ESPN Draft Assist',
    description: 'Every recorded ESPN mock draft, pick by pick, with the board as it stood at each turn.',
  })
  const [drafts, setDrafts] = useState<MockDraft[] | null>(null)
  const [listError, setListError] = useState<string | null>(null)
  // Seeded from `?draft=<id>` so another page (the archive's draft list) can
  // link straight to one board. Null when the URL names nothing, and then
  // the effect below opens the first room in the list as before.
  const [selectedId, setSelectedId] = useState<string | null>(() => {
    try {
      return new URLSearchParams(window.location.search).get('draft')
    } catch {
      return null
    }
  })
  // Kept in its own slot beside the list's, the same split /draft makes: a
  // hiccup fetching one board must not blank the list of rooms, and a
  // hiccup fetching the list must not blank the board being read.
  const [board, setBoard] = useState<LiveBoard | null>(null)
  const [boardError, setBoardError] = useState<string | null>(null)

  // ONE REQUEST PER POLL, NOT TWO. The listing carries the board of its own
  // first row (`first_board`, see api/mocks.py), and the first row is what
  // this page opens and what nearly everybody keeps reading. So while that
  // is the selection, the board endpoint is not polled at all: every listing
  // response brings the board with it, and the page's whole five-second tick
  // is a single request.
  //
  //   `firstBoardId` is the draft whose board the LAST listing response
  //   actually carried, or null if it carried none. Null matters: a board
  //   the server could not build has to fall back to the fetch below, or the
  //   panel would sit empty forever waiting for a listing to bring one.
  //
  //   `seededIdRef` names the draft whose board is already in state and has
  //   not been accounted for by the board effect yet. That effect consumes
  //   it and clears it, so re-selecting the same room after looking at
  //   another one still fetches.
  //
  //   `selectedIdRef` is the current selection read from inside
  //   `loadDrafts`, which is memoised with no dependencies (its identity
  //   gates the polling effect) and would otherwise close over the selection
  //   as it was on mount. Seeding only when the selection is the first row
  //   -- or nothing yet -- keeps a poll from throwing that board over one
  //   somebody chose, and keeps `?draft=<id>` pointing where it says.
  const [firstBoardId, setFirstBoardId] = useState<string | null>(null)
  const seededIdRef = useRef<string | null>(null)
  const selectedIdRef = useRef(selectedId)
  useEffect(() => { selectedIdRef.current = selectedId }, [selectedId])

  const loadDrafts = useCallback(async (signal: { cancelled: boolean }) => {
    try {
      const list = await fetchMockDrafts()
      if (signal.cancelled) return
      setDrafts(list.drafts)
      setListError(null)
      const first = list.drafts.length > 0 ? list.drafts[0] : null
      setFirstBoardId(list.first_board && first ? first.id : null)
      if (list.first_board && first &&
          (selectedIdRef.current === null || selectedIdRef.current === first.id)) {
        seededIdRef.current = first.id
        setBoard(list.first_board)
        setBoardError(null)
      }
    } catch (e) {
      if (signal.cancelled) return
      // `drafts` is deliberately left as it was: a list already on screen is
      // better than an empty page when one poll misses, and it stays null
      // only if the FIRST fetch failed -- which is the state the empty
      // panel below reads.
      setListError(errorText(e, 'Failed to load the mock drafts'))
    }
  }, [])

  const anyLive = drafts?.some((d) => d.status === 'live') ?? false

  // The list: fetched once on mount, then polled only while a room is
  // actually playing. With nothing live there is nothing for a poll to
  // find, so the page goes quiet and the Refresh button in the top bar is
  // how a room that started since then gets picked up.
  useEffect(() => {
    const signal = { cancelled: false }
    loadDrafts(signal)
    if (!anyLive) return () => { signal.cancelled = true }
    const id = window.setInterval(() => loadDrafts(signal), POLL_MS)
    return () => {
      signal.cancelled = true
      window.clearInterval(id)
    }
  }, [loadDrafts, anyLive])

  // Open the first room in the list -- the server sorts live first, then
  // newest -- so the page lands on a board rather than on an instruction.
  // Only ever fires when nothing is selected, so a poll can never yank the
  // board out from under someone reading it.
  useEffect(() => {
    if (selectedId !== null || !drafts || drafts.length === 0) return
    setSelectedId(drafts[0].id)
  }, [drafts, selectedId])

  const selected = useMemo(
    () => drafts?.find((d) => d.id === selectedId) ?? null,
    [drafts, selectedId],
  )
  const selectedLive = selected?.status === 'live'
  const selectedWhen = selected ? draftWhen(selected) : null

  // Blanked on the way to a DIFFERENT room only, never on a re-poll of the
  // same one -- a board that flickers to "loading" every five seconds is
  // unreadable. The effect below therefore leaves `board` alone and simply
  // overwrites it when the next one lands.
  const handleSelect = useCallback((id: string) => {
    if (id === selectedId) return
    setBoard(null)
    setBoardError(null)
    setSelectedId(id)
  }, [selectedId])

  // Guards the first paint of a room against a poll of the PREVIOUS one
  // landing late and painting the wrong board under the right heading.
  const wantedRef = useRef<string | null>(null)

  useEffect(() => {
    if (selectedId === null) return
    wantedRef.current = selectedId
    let cancelled = false

    async function load(id: string) {
      try {
        const data = await fetchMockBoard(id)
        if (cancelled || wantedRef.current !== id) return
        setBoard(data)
        setBoardError(null)
      } catch (e) {
        if (cancelled || wantedRef.current !== id) return
        setBoardError(errorText(e, 'Failed to load that draft board'))
      }
    }

    if (seededIdRef.current === selectedId) {
      // Already on screen, inlined with the listing. Fetching it again on
      // the same tick is the request this whole arrangement exists to avoid.
      seededIdRef.current = null
    } else {
      load(selectedId)
    }
    // A finished draft is fetched exactly once. Nothing about it can change
    // again, so polling it would be five seconds of noise a minute for a
    // board that is already final.
    if (!selectedLive) return () => { cancelled = true }
    // Nor is the first row polled here, ever: the listing poll above is
    // already running for exactly as long as this board can change, and it
    // brings the board with it. Two timers asking two endpoints for the same
    // five seconds of a draft is the thing `first_board` exists to stop.
    if (selectedId === firstBoardId) return () => { cancelled = true }
    const id = window.setInterval(() => load(selectedId), POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [selectedId, selectedLive, firstBoardId])

  const tally = useMemo(() => tallyMakers(board), [board])

  // DraftBoardGrid draws nothing when `active` is false -- that guard is
  // there for the live room, where an inactive session means there is no
  // draft to show. A RECORDED draft is on the page because somebody asked
  // for it, finished or not, so anything with a shape gets drawn.
  //
  // `on_the_clock` is dropped on a finished draft for the same reason: the
  // grid paints a pulsing "on the clock" cell wherever it is set, and a
  // draft that ended has nobody on the clock however the recording left the
  // field.
  const gridBoard = useMemo(
    () => (board && board.teams > 0 && board.rounds > 0
      ? { ...board, active: true, on_the_clock: selectedLive ? board.on_the_clock : null }
      : null),
    [board, selectedLive],
  )

  const liveCount = drafts?.filter((d) => d.status === 'live').length ?? 0

  return (
    <div className="mocks-page">
      <header className="draft-topbar">
        <Link to="/" className="draft-topbar-title"><Logo /> ESPN Draft Assist</Link>
        <span className="draft-topbar-sep" aria-hidden="true" />
        {/* The same three tabs every page carries, so the bar does not slim
            down on the way from the archive to one of its boards. Archive
            is the one lit: this is its reading room. */}
        <nav className="draft-topbar-tabs" aria-label="Views">
          <Link className="draft-tab" to="/">Home</Link>
          <Link className="draft-tab is-active" to="/archive" aria-current="page">Data</Link>
          <Link className="draft-tab" to="/live">Live</Link>
        </nav>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <span className="mocks-title">Mock drafts</span>
        {drafts !== null && drafts.length > 0 && (
          <span className="draft-topbar-hint">
            {liveCount > 0 ? `${liveCount} live · ` : ''}{drafts.length} room{drafts.length === 1 ? '' : 's'}
          </span>
        )}
        {/* Where these same drafts end up once they are counted. A plain
            anchor, not Link: /adp is rendered by FastAPI from the corpus and
            is not a route this app's router knows. */}
        <span className="draft-topbar-sep" aria-hidden="true" />
        <a className="draft-tab" href="/adp">ADP</a>
        <span className="draft-topbar-spacer" />
        {/* The only control on the page. It exists because the list stops
            polling once nothing is live (see the effect above): without it
            a room that starts while this tab is open would need a reload. */}
        <button
          type="button"
          className="mocks-refresh"
          onClick={() => void loadDrafts({ cancelled: false })}
        >
          Refresh
        </button>
      </header>

      {listError && <p className="error draft-error-banner">{listError}</p>}

      <div className="mocks-body">
        <aside className="mocks-list" aria-label="Mock drafts">
          {drafts === null && !listError && <p className="rail-empty mocks-empty">Loading…</p>}
          {drafts === null && listError && (
            <p className="rail-empty mocks-empty">
              No list to show. The farm's API has to be up for this page to have
              anything in it.
            </p>
          )}
          {drafts !== null && drafts.length === 0 && (
            <p className="rail-empty mocks-empty">
              The farm has not joined a mock draft yet. Rooms appear here as soon
              as it does.
            </p>
          )}
          {drafts?.map((d) => (
            <DraftRow
              key={d.id}
              draft={d}
              active={d.id === selectedId}
              onSelect={handleSelect}
            />
          ))}
        </aside>

        <section className="mocks-board" aria-label="Draft board">
          {selected && (
            <div className="mocks-board-head">
              <h2 className="mocks-board-title mono">{selected.league_id}</h2>
              <span className="mocks-board-sub">
                {selected.teams} teams · {selected.rounds} rounds
                {selected.my_slot !== null && ` · we drafted from slot ${selected.my_slot}`}
                {selectedWhen && ` · ${selectedWhen.label} ${ageLabel(selectedWhen.at)}`}
              </span>
            </div>
          )}
          {boardError && <p className="error mocks-board-error">{boardError}</p>}
          {!selectedId && !boardError && (
            <p className="rail-empty mocks-empty">Pick a room on the left to see its board.</p>
          )}
          {selectedId && !gridBoard && !boardError && (
            <p className="rail-empty mocks-empty">Loading the board…</p>
          )}
          {gridBoard && (
            <>
              <MakerTally tally={tally} />
              <div className="board-tab mocks-grid">
                <DraftBoardGrid board={gridBoard} />
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  )
}
