import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchMockRooms, fetchRoomProgress, type MockRoom, type RoomProgress,
         type UpcomingDraft } from '../api'
import { Ago, Countdown, CountdownIn } from './Countdown'

// ESPN's public mock-draft lobby, as rooms you can take a seat in.
//
// WHY THIS IS HERE AND NOT A LINK TO ESPN. A mock draft is how somebody tries
// this tool without waiting for August: a room starts every few minutes, all
// night, free. ESPN's own directory is public (api/lobby.py reads it with no
// cookies at all), so the rooms can be listed to anybody -- and for a reader
// who is signed in, the seat can be taken in the waiting room a click away,
// which is the whole difference between "go and do this on ESPN" and a button.
//
// NOT the same thing as /mocks, which is the boards the farm has already
// played and recorded. These are rooms nobody has drafted in yet -- and, at
// the front of the same grid, the ones YOU have a seat in: drafting now
// first, then waiting to start, then everything open. One card for all
// three, because they are the same kind of thing at three moments; what
// changes is what the clock slot says (a countdown, or the round and pick),
// the corner mark, and the button.

// The lobby turns over fast -- rooms fill, rooms start, new ones appear every
// few minutes -- and the seat counts on these cards move while somebody is
// reading them. 45s was a list that quietly lied: a room drawn at 4/10 could
// be 8/10 by the time it was clicked, and nothing on screen ever showed a
// seat being taken.
//
// This costs the server nothing and ESPN nothing. `api/lobby.py` serves one
// cached directory read to every visitor at once, and re-reads it on its own
// 12s clock while anybody is looking -- so a poll here is a read of memory
// that has already been refreshed behind it, and a thousand readers are
// still one request upstream.
const POLL_MS = 12_000

// A page of cards. Twelve is three rows of four on a monitor, and small enough
// that a reader gets through a page before the list reshuffles under them.
const PAGE_SIZE = 12

// How recent a seat change has to be, on first sight of a room, for the
// page to play it: the server says "+2, forty seconds ago" and the two pips
// that just filled come up green. Past this the change is history and the
// pips are simply full.
const FRESH_SECONDS = 90

/** One seat change on one room, for the pips to play: `from` seats taken
 *  became `to`. `at` keys the animation, so a second change on the same room
 *  replays it rather than being swallowed by the first. */
interface SeatChange {
  from: number
  to: number
  at: number
}

// A room's seats, drawn rather than counted. THE ONE THING A READER IS
// CHOOSING BETWEEN is how full these rooms are: a room at 9/10 is nine people
// waiting to draft and one seat left, and a room at 1/10 is ESPN's autodraft
// engine reading ADP back at you (the reasoning is `pipeline/espn_mock_lobby`'s
// own, and it is why the farm ranks by this number too). Pips say that in the
// shape of the thing being described -- a row of seats, filling.
//
// Capped because ESPN runs rooms up to 20 seats and twenty pips is a barcode.
// Past the cap the fraction is printed instead, which is the honest fallback:
// a shape nobody can count is worse than the number it stood for.
const PIP_CAP = 14

function Seats({ room, change }: {
  room: MockRoom
  /** The room's last seat change, if the page saw one (or the server says
   *  one just happened). The pips that changed play it: a seat taken comes up
   *  green and settles to the row's own colour; a seat given up goes red and
   *  fades to empty. The count alone cannot show movement -- a room redrawn
   *  from 4 to 6 is two more grey marks in a row of grey marks. */
  change: SeatChange | null
}): React.ReactNode {
  const size = room.league_size ?? 0
  const label = `${room.teams_joined} of ${size || '?'} seats taken`
  if (size === 0 || size > PIP_CAP) {
    return <span className="ml-seats mono" title={label}>{room.teams_joined}/{size || '?'}</span>
  }
  const joined = change !== null && change.to > change.from
  const left = change !== null && change.to < change.from
  return (
    <span className="ml-pips" title={label} aria-label={label}>
      {Array.from({ length: size }, (_, i) => {
        const taken = i < room.teams_joined
        // The row fills left to right, so the seats that just changed are
        // the ones between the old count and the new.
        const lo = change === null ? 0 : Math.min(change.from, change.to)
        const hi = change === null ? 0 : Math.max(change.from, change.to)
        const moved = change !== null && i >= lo && i < hi
        return (
          <span
            // Keyed on the change so a new one remounts the pip and its
            // animation runs again from the top.
            key={moved ? `${i}-${change.at}` : i}
            className={`ml-pip${taken ? ' is-taken' : ''}`
              + (moved && joined ? ' is-joined' : '')
              + (moved && left ? ' is-left' : '')}
          />
        )
      })}
    </span>
  )
}

/** A duration, as short as it can be said. Seats move on the scale of
 *  seconds to minutes, so nothing here needs a unit bigger than an hour. */
function shortAgo(seconds: number): string {
  if (seconds < 60) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m`
  return `${Math.round(minutes / 60)}h`
}

// "10-team PPR snake", from three fields that are each sometimes missing. Any
// one of them absent is left out rather than dashed in -- a row that says
// "10-team snake" is honest about knowing two things, where "10-team — snake"
// spends a character claiming ESPN told us something it did not.
function shapeLabel(room: MockRoom): string {
  return [room.league_size ? `${room.league_size}-team` : null,
          room.scoring,
          room.draft_type ? room.draft_type.toLowerCase() : null]
    .filter(Boolean).join(' ')
}

/** The same label for a room you are seated in, from ESPN's league record
 *  (teams, draft type) plus the lobby's own row for scoring when the lobby
 *  still lists it. A room the lobby has dropped -- full, or already
 *  drafting -- knows two things about itself and says two. */
function seatedLabel(league: UpcomingDraft, room: MockRoom | undefined): string {
  return [league.teams ? `${league.teams}-team` : null,
          room?.scoring ?? null,
          league.draft_type ? league.draft_type.toLowerCase() : null]
    .filter(Boolean).join(' ') || league.name || 'Mock draft'
}

// Live first, then soonest to start, then the ones with no date at all.
function bySeatUrgency(a: UpcomingDraft, b: UpcomingDraft): number {
  if (a.live !== b.live) return a.live ? -1 : 1
  if (a.draft_at === null || b.draft_at === null) {
    return a.draft_at === b.draft_at ? 0 : (a.draft_at === null ? 1 : -1)
  }
  return Date.parse(a.draft_at) - Date.parse(b.draft_at)
}

// THE FILTERS ARE BUILT FROM WHAT IS OPEN, not from a list of everything ESPN
// has ever served. The lobby at 3am is a different lobby from the one at 8pm --
// sizes come and go, auctions appear in batches -- and a pill for a shape no
// open room has is a control that can only ever return nothing.
type Facet = 'size' | 'scoring' | 'type'

const FACETS: Facet[] = ['size', 'scoring', 'type']

function facetValue(room: MockRoom, facet: Facet): string | null {
  if (facet === 'size') return room.league_size ? String(room.league_size) : null
  if (facet === 'scoring') return room.scoring
  return room.draft_type
}

function facetLabel(facet: Facet, value: string): string {
  return facet === 'size' ? `${value}-team` : value.toLowerCase()
}

function optionsFor(rooms: MockRoom[], facet: Facet): string[] {
  const seen = new Set<string>()
  for (const room of rooms) {
    const value = facetValue(room, facet)
    if (value) seen.add(value)
  }
  const values = [...seen]
  // Sizes sort as numbers (8, 10, 12, 20); everything else alphabetically --
  // "10" before "8" is the one ordering a reader would call broken.
  return facet === 'size'
    ? values.sort((a, b) => Number(a) - Number(b))
    : values.sort()
}

export default function MockLobby({ onOpen, seated, onEnter, joining, onCount }: {
  /** Opens the room's waiting room -- seats, countdown, who is in. Nothing is
   *  committed by looking: the seat is taken there, by clicking one. */
  onOpen: (leagueId: string) => void
  /** The mock rooms this account already holds a seat in, from the same
   *  ESPN league list the leagues row is drawn from. Pinned ahead of the
   *  lobby, and never filtered out: a room you are in is not a room you
   *  are choosing between. */
  seated: UpcomingDraft[]
  /** Walk into a seated room that is drafting: mints the token and opens
   *  the board, exactly as a league's "Enter the room" does. */
  onEnter: (league: UpcomingDraft) => void
  /** Which room a mint is in flight for, so its button says so and the
   *  others wait. Owned by the page: one join at a time across both rows. */
  joining: string | null
  /** How many rooms this read found, for the top bar to state. Reported up
   *  rather than fetched twice: two reads of one endpoint are two numbers that
   *  can disagree on the same screen. */
  onCount?: (n: number) => void
}) {
  const [rooms, setRooms] = useState<MockRoom[] | null>(null)
  const [total, setTotal] = useState(0)
  // When the countdowns were read, so the seconds run down between polls
  // instead of standing still for 45 seconds and then jumping.
  const [readAt, setReadAt] = useState(() => Date.now())
  const [filters, setFilters] = useState<Record<Facet, string | null>>({
    size: null, scoring: null, type: null,
  })
  const [page, setPage] = useState(0)

  // Never throws (see fetchMockRooms): an unreadable lobby is an empty list,
  // which this section states in one line rather than as an error. Stable
  // because the poll below depends on it and a fresh closure per render would
  // restart the interval every time.
  // WHAT MOVED, for the pips to play. Counts from the last read, by room, so
  // a poll that comes back different is a change the page saw and can show.
  // A room seen for the first time takes the server's own last change if it
  // was recent (`seats_delta` within FRESH_SECONDS), so the page's first
  // paint is not the one paint on which nothing ever moves.
  const lastCounts = useRef<Map<string, number>>(new Map())
  const [changes, setChanges] = useState<Map<string, SeatChange>>(() => new Map())
  // Round and pick for the rooms you are drafting in, read on the same poll
  // as the lobby. Keyed by room; a room the server could not read is simply
  // absent and its card says "drafting" without a number.
  const [progress, setProgress] = useState<Record<string, RoomProgress>>({})
  const liveIds = useMemo(
    () => seated.filter((l) => l.live).map((l) => l.league_id).sort().join(','),
    [seated])

  const read = useCallback(async () => {
    const ids = liveIds === '' ? [] : liveIds.split(',')
    const [next, rows] = await Promise.all([fetchMockRooms(), fetchRoomProgress(ids)])
    if (ids.length > 0) setProgress(rows)
    const now = Date.now()
    const seen = lastCounts.current
    const moved = new Map<string, SeatChange>()
    for (const room of next.rooms) {
      const before = seen.get(room.league_id)
      if (before === undefined) {
        const recent = room.seats_changed_seconds !== null
          && room.seats_changed_seconds <= FRESH_SECONDS && room.seats_delta !== 0
        if (recent) {
          moved.set(room.league_id, {
            from: room.teams_joined - room.seats_delta, to: room.teams_joined, at: now,
          })
        }
      } else if (before !== room.teams_joined) {
        moved.set(room.league_id, { from: before, to: room.teams_joined, at: now })
      }
      seen.set(room.league_id, room.teams_joined)
    }
    if (moved.size > 0) {
      setChanges((current) => {
        const merged = new Map(current)
        for (const [id, change] of moved) merged.set(id, change)
        return merged
      })
    }
    setRooms(next.rooms)
    setTotal(next.total)
    setReadAt(now)
    onCount?.(next.total)
  }, [onCount, liveIds])

  useEffect(() => {
    read()
    const id = setInterval(read, POLL_MS)
    return () => clearInterval(id)
  }, [read])

  // The lobby's row for each seated room, if it still has one: that is where
  // the seat count lives, and ESPN's league record carries none.
  const byId = useMemo(() => {
    const map = new Map<string, MockRoom>()
    for (const room of rooms ?? []) map.set(room.league_id, room)
    return map
  }, [rooms])
  const pinned = useMemo(() => [...seated].sort(bySeatUrgency), [seated])
  const seatedIds = useMemo(() => new Set(seated.map((l) => l.league_id)), [seated])

  // A room you are seated in is drawn once, at the front -- not again in the
  // lobby's own order as a room to consider joining.
  const others = useMemo(() => (rooms ?? []).filter((room) =>
    !seatedIds.has(room.league_id)), [rooms, seatedIds])
  const shown = useMemo(() => others.filter((room) =>
    FACETS.every((facet) => {
      const want = filters[facet]
      return want === null || facetValue(room, facet) === want
    })), [others, filters])

  const pages = Math.max(1, Math.ceil(shown.length / PAGE_SIZE))
  // A filter that shortens the list can leave a reader on a page that no
  // longer exists. Clamped at render rather than reset on every click, so
  // narrowing from page 3 to a two-page result lands on the last page rather
  // than throwing them back to the first.
  const current = Math.min(page, pages - 1)
  const visible = shown.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE)

  const setFacet = useCallback((facet: Facet, value: string) => {
    setFilters((prev) => ({ ...prev, [facet]: prev[facet] === value ? null : value }))
    setPage(0)
  }, [])

  const narrowed = rooms !== null && shown.length !== others.length

  return (
    <section className="db-sec">
      <div className="db-sec-head">
        <h2 className="db-sec-title">
          {pinned.some((l) => l.live) && <span className="db-dot" aria-hidden="true" />}
          Mock drafts
        </h2>
        {rooms !== null && (
          <span className="mono db-sec-count">
            {narrowed ? `${shown.length} of ${others.length}`
              : total > rooms.length ? `${others.length} of ${total}`
                : shown.length}
          </span>
        )}
        {/* The page is reading ESPN on a clock, and the seat counts on these
            cards are the reason. Saying when the last read landed is what
            makes a count somebody is about to act on trustworthy -- and the
            figure moves every second, which is the other half of the claim. */}
        <span className="db-sec-note">
          {/* The page's own live dot, the same one the Drafting-now heading
              wears: one mark for "this is being watched right now". */}
          <span className="db-dot" aria-hidden="true" />
          {rooms === null
            ? 'reading seats…'
            : <>seats checked <Ago since={readAt} format={shortAgo} /></>}
          {' · ESPN opens a new room every few minutes · free'}
        </span>
      </div>

      {/* One row of pills per thing worth narrowing by, each built from the
          rooms actually open right now (see optionsFor). A pill toggles: a
          second click on the active one is how a reader clears it, which is
          also why there is no "All" pill to hunt for. */}
      {others.length > 0 && (
        <div className="ml-filters">
          {FACETS.map((facet) => {
            const options = optionsFor(others, facet)
            // Nothing to choose between is not a filter: one value means every
            // open room shares it, and a pill that can only be on is furniture.
            if (options.length < 2) return null
            return (
              <div className="ml-filter" key={facet}>
                {options.map((value) => (
                  <button
                    key={value}
                    type="button"
                    className={`ml-pill${filters[facet] === value ? ' is-active' : ''}`}
                    onClick={() => setFacet(facet, value)}
                    aria-pressed={filters[facet] === value}
                  >
                    {facetLabel(facet, value)}
                  </button>
                ))}
              </div>
            )
          })}
        </div>
      )}

      {rooms === null && pinned.length === 0 ? (
        <p className="db-empty">Reading ESPN’s lobby…</p>
      ) : rooms !== null && others.length === 0 && pinned.length === 0 ? (
        // Both cases at once, and deliberately: ESPN unreachable and a lobby
        // between batches look identical to a reader, and both are fixed by
        // the same thing, which is waiting a minute.
        <p className="db-empty">No rooms open right now. ESPN starts more every few minutes.</p>
      ) : rooms !== null && shown.length === 0 && pinned.length === 0 ? (
        <p className="db-empty">No open room matches that. Rooms turn over every few minutes.</p>
      ) : (
        <>
          <ul className="db-cards">
            {/* YOUR ROOMS FIRST, in the same grid. The lobby card, with the
                corner saying what the lobby's cards leave blank: this one is
                yours, and whether it is picking yet. A room that is drafting
                puts its round and pick where the countdown was, lights the
                card, and takes the filled button -- it is the one card here
                you can walk into this minute. A seat still waiting keeps
                the countdown, the pips if the lobby still lists the room,
                and the quiet button that opens the waiting room. Ahead of
                the pager's first page rather than paged with the lobby:
                neither the filters nor "starting later" are ways of choosing
                between rooms you are already in. */}
            {current === 0 && pinned.map((league) => {
              const room = byId.get(league.league_id)
              const at = progress[league.league_id]
              const open = room && room.league_size !== null
                ? Math.max(0, room.league_size - room.teams_joined) : null
              return (
                <li className={`db-card${league.live ? ' is-live' : ''}`} key={league.league_id}>
                  <div className="db-card-top">
                    {/* Drafting rooms print a pick, not a clock -- and a
                        room with a clock draws its own, so this row is not
                        rebuilt every second to move four characters. */}
                    {league.live ? (
                      <span className="mono db-card-when is-live">
                        {at?.round ? `R${at.round} · P${at.pick_in_round}` : 'Drafting'}
                      </span>
                    ) : (
                      <Countdown at={league.draft_at} className="mono db-card-when" />
                    )}
                    {league.live ? (
                      <span className="ml-mark is-live">
                        <span className="db-dot" aria-hidden="true" />
                        Drafting now
                      </span>
                    ) : (
                      <span className="ml-mark">Your seat</span>
                    )}
                  </div>
                  <p className="db-card-name">{seatedLabel(league, room)}</p>
                  <div className="db-card-foot">
                    {league.live && at?.picks_made !== null && at?.picks_made !== undefined ? (
                      // The room's progress in the seat line's own face:
                      // how much of the draft is done. The team's name is
                      // not repeated here -- the button needs the width
                      // more than the card needs to say "yours" twice.
                      <span className="ml-seat-line">
                        <span className="mono ml-count">
                          <span className="ml-count-in">
                            {at.picks_made} of {at.picks_total ?? '?'} picks
                          </span>
                        </span>
                      </span>
                    ) : room ? (
                      <span className="ml-seat-line">
                        <Seats room={room} change={changes.get(room.league_id) ?? null} />
                        <span className="mono ml-count">
                          <span className="ml-count-in">{room.teams_joined} in</span>
                          {open !== null && (
                            <>
                              <span className="ml-count-sep" aria-hidden="true">·</span>
                              <span className={`ml-count-open${open === 0 ? ' is-full' : ''}`}>
                                {open === 0 ? 'full' : `${open} open`}
                              </span>
                            </>
                          )}
                        </span>
                      </span>
                    ) : (
                      // Off the lobby's list -- full, or already picking --
                      // so the seats are not knowable from here. Your own
                      // team's name is what there is to say instead.
                      <span className="ml-seat-line ml-team">{league.team_name}</span>
                    )}
                    {league.live ? (
                      <button
                        type="button"
                        className="db-go"
                        onClick={() => onEnter(league)}
                        disabled={joining !== null || !league.team_id}
                      >
                        {joining === league.league_id ? 'Joining…' : 'Enter draft'}
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="db-join"
                        onClick={() => onOpen(league.league_id)}
                      >
                        View room
                      </button>
                    )}
                  </div>
                </li>
              )
            })}
            {visible.map((room) => {
              const open = room.league_size === null
                ? null : Math.max(0, room.league_size - room.teams_joined)
              return (
                <li className="db-card" key={room.league_id}>
                  <div className="db-card-top">
                    <CountdownIn seconds={room.starts_in_seconds} since={readAt}
                                 className="mono db-card-when" />
                  </div>
                  <p className="db-card-name">{shapeLabel(room) || 'Mock draft'}</p>
                  <div className="db-card-foot">
                    <span className="ml-seat-line">
                      <Seats room={room} change={changes.get(room.league_id) ?? null} />
                      {/* The two numbers a reader is choosing between, said
                          outright: who is already waiting, and whether there
                          is a seat left. "7 in · 3 open" -- short enough to
                          share a line with the pips and the button at every
                          card width the grid produces. */}
                      <span className="mono ml-count">
                        <span className="ml-count-in">{room.teams_joined} in</span>
                        {open !== null && (
                          <>
                            <span className="ml-count-sep" aria-hidden="true">·</span>
                            <span className={`ml-count-open${open === 0 ? ' is-full' : ''}`}>
                              {open === 0 ? 'full' : `${open} open`}
                            </span>
                          </>
                        )}
                      </span>
                    </span>
                    <button
                      type="button"
                      className="db-join"
                      onClick={() => onOpen(room.league_id)}
                    >
                      View room
                    </button>
                  </div>
                </li>
              )
            })}
          </ul>

          {/* Only when there is a second page: a pager under a single screen of
              cards says "there is more" when there is not. Labelled by what the
              next page HOLDS rather than by its number -- the list is ordered
              by how soon each room starts, so "starting later" is what the
              button actually does. */}
          {pages > 1 && (
            <div className="ml-pager">
              <button
                type="button"
                className="ml-page-btn"
                onClick={() => setPage(current - 1)}
                disabled={current === 0}
              >
                ← Starting sooner
              </button>
              <span className="mono ml-page-at">
                {current * PAGE_SIZE + 1}–{current * PAGE_SIZE + visible.length} of {shown.length}
              </span>
              <button
                type="button"
                className="ml-page-btn"
                onClick={() => setPage(current + 1)}
                disabled={current >= pages - 1}
              >
                Starting later →
              </button>
            </div>
          )}
        </>
      )}
    </section>
  )
}
