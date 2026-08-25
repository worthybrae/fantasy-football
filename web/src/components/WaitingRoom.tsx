import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchMockRoom, fetchPicksHistory, joinMockRoom, mintDraftToken,
         type MockRoomState, type PickHistory,
         type TokenConnectParams } from '../api'
import { countdownTo } from '../lib/countdown'
import { Logo } from './Logo'
import PickOption from './PickOption'

// THE ROOM BEFORE THE ROOM. A mock draft opens a few minutes before it picks,
// and the one decision in those minutes is which seat to take. In a snake that
// is not a preference, it is a schedule: the seat fixes every pick number you
// will ever own. So the page is the room down the left and the seat under
// consideration at full size on the right.
//
// AND THE SEAT IS PRICED, not just listed. Beside each of its picks are the
// three players who most often went at that exact pick across every recorded
// mock, each with its share of that pick and what the board projects him for
// (api/market.py's at-picks, human picks only -- an autodrafted seat is ESPN's
// ADP read back to us). That turns "seat 6 picks 6th and 23rd" into the thing
// a drafter actually wants to know: who is usually there when it is your turn,
// and whether that turn is spoken for or wide open.
// The archive is 8-team mocks and says so on the page; a pick number is
// comparable across room sizes in the one way that matters here (at pick 23,
// twenty-two players are gone), and a pick the archive never reached says "no
// record" rather than showing an invented name.
//
// SEAT CHOICE IS ONE-WAY, AND THE UI SAYS SO. Probed live against ESPN
// (2026-08-24): a specific seat can be requested and is honoured, but a second
// invite for the same league is refused -- "You may not manage more than one
// team in a standard league" -- and no endpoint that gives a seat back
// answered (`isDeleted`: 409, DELETE: 405). So the click that CHOOSES is free
// and repeatable; a separate, labelled action commits.
//
// WHEN IT STARTS, IT LEAVES. The poll watches `in_progress`; the moment ESPN
// says picking has begun, the token is minted for the seat this reader holds
// and handed to the connect the bookmarklet path already uses.

// Fast, because both things it watches move in seconds: a seat taken by
// somebody else (which invalidates a selection), and the draft starting.
const POLL_MS = 4000

// The pick this seat owns in this round. `slot` is 1-based within `teams` and
// the direction alternates -- the same snake arithmetic
// `scoring/draft_sim.snake_slots` does server-side and `pickOrder.ts` does for
// the room's clock panel.
function pickAt(slot: number, teams: number, round: number): number {
  const within = round % 2 === 1 ? slot : teams - slot + 1
  return (round - 1) * teams + within
}

function picksFor(slot: number, teams: number, rounds: number): number[] {
  return Array.from({ length: rounds }, (_, i) => pickAt(slot, teams, i + 1))
}

export default function WaitingRoom({ leagueId, season, onJoin }: {
  leagueId: string
  season?: number | null
  /** Called with the minted token when the draft starts. The page that
   *  routes here decides where that token goes. */
  onJoin: (params: TokenConnectParams) => void
}) {
  const [room, setRoom] = useState<MockRoomState | null>(null)
  const [error, setError] = useState<string | null>(null)
  // The seat under consideration -- defaulted to the first open one by the
  // poll below. Free to change, and nothing has been said to ESPN about it:
  // see the file's own note on why committing is separate.
  const [picked, setPicked] = useState<string | null>(null)
  const [taking, setTaking] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const [history, setHistory] = useState<PickHistory[] | null>(null)
  const [archive, setArchive] = useState<{ drafts: number; teams: number } | null>(null)
  // When the server's countdown was read, so the seconds run down between
  // polls instead of standing still and then jumping.
  const readAt = useRef(Date.now())
  // Guards the hand-off: `in_progress` stays true for the whole draft, and the
  // poll that spots it can fire again before this page has unmounted.
  const entering = useRef(false)

  const read = useCallback(async () => {
    try {
      const next = await fetchMockRoom(leagueId, season)
      setRoom(next)
      readAt.current = Date.now()
      setError(null)
      // The seat under consideration is always a real, open one: the first
      // open seat on arrival, so the page lands on a schedule rather than
      // on an instruction, and the next open seat when the one being read
      // is filled by somebody else -- settled here rather than at click
      // time, so the list never shows a selected seat with another
      // manager's name in it. Nothing about this is a claim on the seat:
      // the chip says "Selected", and only "Take seat N" speaks to ESPN.
      setPicked((current) => {
        const seat = next.seats.find((s) => s.team_id === current)
        if (seat && !seat.taken) return current
        return next.seats.find((s) => !s.taken)?.team_id ?? null
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [leagueId, season])

  useEffect(() => {
    read()
    const id = setInterval(read, POLL_MS)
    return () => clearInterval(id)
  }, [read])

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    if (!room?.in_progress || room.my_team_id === null || entering.current) return
    entering.current = true
    mintDraftToken(leagueId, room.my_team_id, season ?? null)
      .then(onJoin)
      .catch((e) => {
        entering.current = false
        setError(e instanceof Error ? e.message : String(e))
      })
  }, [room?.in_progress, room?.my_team_id, leagueId, season, onJoin, room])

  const mine = room?.seats.find((s) => s.mine) ?? null
  const teams = room?.teams ?? 0
  // ESPN's own count when it gives one, sixteen when it does not: every mock
  // room measured drafts 16 rounds, and a board with no rows is worse than one
  // built on that fact.
  const rounds = room?.rounds ?? 16
  // The seat being read: the one held, else the one selected. Both are shown
  // the same way -- what changes is whether the page is still offering it.
  const shown = mine ?? room?.seats.find((s) => s.team_id === picked) ?? null
  const shownPicks = useMemo(
    () => (shown === null || teams === 0 ? [] : picksFor(shown.slot, teams, rounds)),
    [shown, teams, rounds])

  // The archive's answer for exactly this seat's picks. Re-read when the seat
  // changes and not otherwise: it is a ten-minute-cached aggregate over
  // hundreds of drafts and it does not move while somebody reads it.
  useEffect(() => {
    if (shownPicks.length === 0) { setHistory(null); return }
    let cancelled = false
    fetchPicksHistory(shownPicks)
      .then((body) => {
        if (cancelled) return
        setHistory(body.picks)
        setArchive({ drafts: body.drafts, teams: body.teams })
      })
      // The archive being unreadable costs this column, not the page: the
      // seat, its picks and the way in all still work.
      .catch(() => { if (!cancelled) setHistory(null) })
    return () => { cancelled = true }
  }, [shownPicks])

  const take = useCallback(async () => {
    if (picked === null) return
    setTaking(true)
    setError(null)
    try {
      await joinMockRoom(leagueId, season, picked)
      await read()
    } catch (e) {
      // Somebody took it in the seconds between the read and the click. ESPN
      // refuses rather than seating them elsewhere, so this says so plainly
      // and re-reads rather than leaving them in a seat they did not choose.
      setError(e instanceof Error ? e.message : String(e))
      await read()
    } finally {
      setTaking(false)
    }
  }, [leagueId, picked, read, season])

  const seconds = room === null || room.starts_in_seconds === null
    ? null : room.starts_in_seconds - (now - readAt.current) / 1000
  const count = countdownTo(seconds)
  const taken = room?.seats.filter((s) => s.taken).length ?? 0
  const byPick = new Map((history ?? []).map((h) => [h.pick_no, h]))

  return (
    <main className="wr">
      <header className="draft-topbar">
        <Link to="/" className="draft-topbar-title"><Logo /> ESPN Draft Assist</Link>
        <span className="draft-topbar-sep" aria-hidden="true" />
        {/* THE SITE'S OWN TAB STRIP, unchanged. This page used to swap in a
            "Waiting room" tab and drop the others, which made the room read
            as a different product; the room already announces itself in the
            meta and the countdown pill on this same bar. Drafts stays the
            active tab -- a waiting room is part of getting into a draft. */}
        <nav className="draft-topbar-tabs" aria-label="Views">
          <Link className="draft-tab is-active" to="/" aria-current="page">Drafts</Link>
          <Link className="draft-tab" to="/archive">Archive</Link>
          <Link className="draft-tab" to="/live">Live</Link>
        </nav>
        <span className="draft-topbar-sep" aria-hidden="true" />
        <span className="draft-topbar-league">
          {room === null ? 'Reading the room…' : (
            `${room.teams} teams${room.draft_type ? ` · ${room.draft_type.toLowerCase()}` : ''}`
            + ` · ${rounds} rounds`
            + `${room.clock_seconds ? ` · ${room.clock_seconds}s a pick` : ''}`
          )}
        </span>
        <span className="draft-topbar-spacer" />
        <span className={`draft-status-pill wr-pill${count?.imminent || room?.in_progress ? ' is-soon' : ''}`}>
          <span className="draft-status-dot" aria-hidden="true" />
          {room?.in_progress ? 'DRAFTING' : count?.text ?? '—'}
        </span>
      </header>

      {error !== null && <p className="wr-error" role="alert">{error}</p>}

      <div className="wr-body">

        {/* THE ROOM, down the left in draft order. A list rather than a grid:
            fourteen seats read faster stacked than wrapped, and the thing a
            reader scans for -- which are still open -- is a column. */}
        <section className="wr-list">
          <div className="wr-list-head">
            <h2 className="wr-list-title">The room</h2>
            <span className="mono wr-list-count">{taken} / {room?.teams ?? '—'} seats</span>
          </div>
          <ul className="wr-seats">
            {(room?.seats ?? []).map((seat) => {
              const isPicked = seat.team_id === picked
              const openSeat = !seat.taken && mine === null && !room?.in_progress
              const Tag = openSeat ? 'button' : 'div'
              return (
                <li key={seat.team_id}>
                  <Tag
                    {...(openSeat
                      ? { type: 'button' as const, onClick: () => setPicked(seat.team_id) }
                      : {})}
                    className={`wr-seat${seat.mine ? ' is-mine' : ''}`
                      + (isPicked ? ' is-picked' : '')
                      + (seat.taken ? ' is-taken' : ' is-open')}
                    aria-pressed={openSeat ? isPicked : undefined}
                  >
                    <span className="mono wr-seat-slot">{seat.slot}</span>
                    <span className="wr-seat-name">
                      {seat.mine ? 'Your seat' : seat.taken ? seat.name : 'Open'}
                    </span>
                    <span className="mono wr-seat-picks">
                      #{pickAt(seat.slot, teams, 1)}
                    </span>
                    {seat.mine ? <span className="wr-seat-chip is-mine">Yours</span>
                      : isPicked ? <span className="wr-seat-chip is-picked">Selected</span>
                        : openSeat ? <span className="wr-seat-chip">Choose</span> : null}
                  </Tag>
                </li>
              )
            })}
          </ul>
        </section>

        {/* THE SEAT, AT FULL SIZE. What it picks, and who the archive says is
            usually there when it does. */}
        <section className="wr-detail">
          {shown === null ? (
            <div className="wr-empty">
              <p className="wr-empty-lead">Choose a seat</p>
              <p className="wr-empty-note">
                In a snake the seat is the schedule: seat 1 picks first and
                waits longest, the last seat picks twice at every turn. Pick one
                on the left to see who usually goes at its turns.
              </p>
            </div>
          ) : (
            <>
              <div className="wr-detail-head">
                <div>
                  <p className="wr-eyebrow">
                    {room?.in_progress ? 'Drafting now'
                      : mine ? 'You are in · picking starts in' : 'Picking starts in'}
                  </p>
                  <p className="mono wr-clock">
                    {room?.in_progress ? 'LIVE' : count?.text ?? '—'}
                  </p>
                </div>
                <div className="wr-detail-seat">
                  <p className="mono wr-seat-big">Seat {shown.slot}</p>
                  <p className="wr-seat-sub">
                    of {teams} · picks {shownPicks[0] ?? '—'}
                    {shownPicks[1] !== undefined ? `, then ${shownPicks[1]}` : ''}
                  </p>
                </div>
              </div>

              <div className="wr-picks-head">
                <h2 className="wr-list-title">Every pick this seat owns</h2>
                {/* ATTRIBUTED, ALWAYS. These names are 8-team mocks answering
                    a question about a room that may not be one, and a number
                    without its source is a number a reader cannot weigh. */}
                <span className="wr-source">
                  {archive === null ? 'reading the archive…'
                    : `who went there in ${archive.drafts} recorded ${archive.teams}-team mocks · people's picks only`}
                </span>
              </div>

              {/* EVERY PICK IS A SHORTLIST, NOT A NAME. This list used to
                  print the single most common pick, which reads as a
                  prediction the archive is not making: at pick 5 the modal
                  player went a third of the time, so two times in three the
                  name on the row was not who a reader would face. Three
                  names, each with its own share, say the honest thing --
                  here is the pool this turn draws from, and here is how
                  spoken-for it is. One name at 78% still looks like one
                  name, because the two beside it are tiny.

                  And each one is priced. A shortlist a reader cannot weigh
                  is a shortlist he has to go and look up somewhere else, so
                  the projection rides with the face and the share. */}
              <ul className="wr-picks">
                {shownPicks.map((pick, i) => {
                  const row = byPick.get(pick)
                  const options = row?.top ?? []
                  return (
                    <li className="wr-pick" key={pick}>
                      <span className="mono wr-pick-round">R{i + 1}</span>
                      <span className="mono wr-pick-no">#{pick}</span>
                      {options.length === 0 ? (
                        <span className="wr-pick-none">
                          {row === undefined ? '—'
                            : 'no record — the archive stops before this pick'}
                        </span>
                      ) : (
                        <div className="wr-pick-options">
                          {/* Nothing to open: the draft has not started, and
                              the profile popup is the room's own. */}
                          {options.map((opt) => (
                            <PickOption
                              key={opt.player_id}
                              name={opt.name ?? opt.player_id}
                              position={opt.position}
                              headshot={opt.headshot}
                              share={opt.share}
                              projPoints={opt.proj_points}
                              projChange={opt.proj_change}
                              shareTitle={`${opt.count} of ${row?.observed} recorded picks here`}
                            />
                          ))}
                        </div>
                      )}
                    </li>
                  )
                })}
              </ul>

              <div className="wr-act">
                {mine !== null ? (
                  <p className="wr-act-note">
                    <strong>Seat {mine.slot} is yours.</strong> ESPN allows one
                    team per manager, so it cannot be swapped — the board opens
                    here the moment picking starts.
                  </p>
                ) : (
                  <>
                    <button type="button" className="wr-take" onClick={take}
                            disabled={taking || room?.in_progress}>
                      {taking ? 'Taking the seat…' : `Take seat ${shown.slot}`}
                    </button>
                    <p className="wr-act-note">
                      Final — ESPN allows one team per manager, so a seat cannot
                      be swapped once taken.
                    </p>
                  </>
                )}
              </div>
            </>
          )}
        </section>
      </div>
    </main>
  )
}
