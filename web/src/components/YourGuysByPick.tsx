import { useEffect, useMemo, useState } from 'react'
import { fetchFavoritesOutlook, type FavoritesOutlook } from '../api'

// YOUR GUYS, BY PICK.
//
// The card above this one says who somebody wants. This one says whether they
// can have them, which is the question the list has never answered: from the
// eighth seat of a twelve-team league, half of anybody's five favourites are
// gone before their first turn, and nothing on this page said so.
//
// Every number is the server's (`GET /api/account/favorites/outlook`), read
// off the same counted draft corpus the live room reads. Nothing is computed
// here -- not the picks, not the thresholds' underlying values, not the best
// pick -- because a second implementation of "will he last" that disagreed
// with the room by two points would be worse than no card at all.
//
// TWO CONTROLS AND NOTHING ELSE. Availability is a function of the seat, and
// before a draft is connected there is no seat to read -- so the reader tells
// us, once, and we remember it. They are the only interactive thing here on
// purpose: this is a card to glance at on the way to the picker, not a tool.

/** The seat, remembered between visits. One key, one small object: the two
 *  values are only ever meaningful together (slot 12 is nonsense in an
 *  eight-team league). */
const STORE_KEY = 'guys-outlook'

/** The league sizes worth offering. Every real redraft league is one of
 *  these; the endpoint accepts 4 to 16, and a select of thirteen options
 *  would be a worse way to say "twelve". */
const SIZES = [8, 10, 12, 14]
const DEFAULT_TEAMS = 10
const DEFAULT_SLOT = 5

/** Where the tint changes. Deliberately not a gradient: three bands is a
 *  decision ("plan on him", "maybe", "no"), and a continuous ramp over
 *  twenty-five rows is a heat map nobody reads a row out of. */
const LIKELY = 70
const MAYBE = 40

/** "5th", for the seat select. The controls carry no bare numbers on
 *  purpose -- a select of ten options labelled 1 to 10 puts ten loose digits
 *  into a page whose other cards are counting things. */
function ordinal(n: number): string {
  if (n % 100 >= 11 && n % 100 <= 13) return `${n}th`
  return `${n}${({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th'}`
}

export interface Seat { teams: number; slot: number }

function clampSeat(teams: unknown, slot: unknown): Seat {
  const size = SIZES.includes(teams as number) ? (teams as number) : DEFAULT_TEAMS
  const seat = Number(slot)
  return {
    teams: size,
    slot: Number.isInteger(seat) && seat >= 1 && seat <= size
      ? seat : Math.min(DEFAULT_SLOT, size),
  }
}

function readSeat(): Seat {
  try {
    const raw = window.localStorage.getItem(STORE_KEY)
    if (raw === null) return clampSeat(DEFAULT_TEAMS, DEFAULT_SLOT)
    const saved = JSON.parse(raw)
    return clampSeat(saved?.teams, saved?.slot)
  } catch {
    // A private window, or something that is not our JSON any more. Either
    // way the default seat is a better answer than a blank card.
    return clampSeat(DEFAULT_TEAMS, DEFAULT_SLOT)
  }
}

/** The chip's tint. Exported because the thresholds are the card's whole
 *  claim and a test should be able to state them without a DOM.
 *
 *  TAKES THE ROUNDED NUMBER, the one the chip actually prints. Banding the
 *  raw value instead put 69.6% in a green chip reading 70% and 39.5% in an
 *  amber one reading 40%, which is a card disagreeing with itself. */
export function tintFor(chance: number | null): string {
  if (chance === null) return 'gbp-none'
  if (chance >= LIKELY) return 'gbp-likely'
  if (chance >= MAYBE) return 'gbp-maybe'
  return 'gbp-thin'
}

export default function YourGuysByPick({ players }: {
  /** The saved ids. Not rendered -- the server sends its own rows -- but a
   *  changed list is a changed answer, and it is what keeps the five-second
   *  request cache from serving the pre-save one. */
  players: string[]
}) {
  const [seat, setSeat] = useState<Seat>(readSeat)
  const [outlook, setOutlook] = useState<FavoritesOutlook | null>(null)
  const [error, setError] = useState<string | null>(null)
  const tag = useMemo(() => players.join(','), [players])

  useEffect(() => {
    try {
      window.localStorage.setItem(STORE_KEY, JSON.stringify(seat))
    } catch { /* private window: the seat is just not remembered */ }
  }, [seat])

  useEffect(() => {
    let cancelled = false
    setOutlook(null)
    setError(null)
    fetchFavoritesOutlook(seat.teams, seat.slot, tag)
      .then((body) => { if (!cancelled) setOutlook(body) })
      .catch((err) => { if (!cancelled) setError(String(err?.message || err)) })
    return () => { cancelled = true }
  }, [seat, tag])

  const picks = outlook?.picks ?? []

  return (
    <section className="db-sec">
      <div className="db-sec-head">
        <h2 className="db-sec-title">By pick</h2>
        <div className="gbp-controls">
          <label className="gbp-ctl">
            League
            <select
              className="gbp-select"
              value={seat.teams}
              onChange={(e) => setSeat(clampSeat(Number(e.target.value), seat.slot))}
            >
              {SIZES.map((n) => <option key={n} value={n}>{n} teams</option>)}
            </select>
          </label>
          <label className="gbp-ctl">
            Seat
            <select
              className="gbp-select"
              value={seat.slot}
              onChange={(e) => setSeat(clampSeat(seat.teams, Number(e.target.value)))}
            >
              {Array.from({ length: seat.teams }, (_, i) => i + 1)
                .map((n) => <option key={n} value={n}>{ordinal(n)}</option>)}
            </select>
          </label>
        </div>
      </div>

      <div className="db-card gbp-card">
        {error !== null && <p className="db-error gbp-error">{error}</p>}

        {/* THE SKELETON KEEPS THE CARD'S HEIGHT. Without it the lobby below
            jumps every time somebody changes a select, which on a page whose
            other rows are also loading reads as the page breaking. */}
        {outlook === null && error === null && (
          <div className="gbp-wait" aria-hidden="true">
            {players.slice(0, 6).map((id) => (
              <div key={id} className="gbp-wait-row" />
            ))}
          </div>
        )}

        {outlook !== null && outlook.players.length > 0 && (
          <div className="gbp-scroll">
            <table className="gbp-grid">
              <caption className="gbp-caption">
                How likely each of your guys is to still be on the board at
                each of your turns, counted from real recorded drafts.
              </caption>
              <thead>
                <tr>
                  <th scope="col" className="gbp-who">Player</th>
                  {picks.map((pick) => (
                    <th scope="col" key={pick} className="gbp-head mono">
                      Pick {pick}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {outlook.players.map((row) => (
                  <tr key={row.player_id}>
                    <th scope="row" className="gbp-who">
                      <span className="gbp-who-in">
                        {row.headshot !== null && (
                          <img className="gbp-face" src={row.headshot} alt=""
                               width={24} height={24} loading="lazy" />
                        )}
                        <span className="gbp-name">{row.name ?? row.player_id}</span>
                        {row.position !== null && (
                          <span className={`pos-badge pos-badge-${row.position.toLowerCase()}`}>
                            {row.position}
                          </span>
                        )}
                      </span>
                    </th>
                    {picks.map((pick, i) => {
                      // Rounded ONCE, then banded and printed from the same
                      // number, so the tint can never contradict the text.
                      const raw = row.avail[i]
                      const chance = raw === null || raw === undefined
                        ? null : Math.round(raw)
                      // THE MARKER, not a second colour: the last turn he is
                      // still better than even to reach is the one to plan
                      // on, and it is often an amber cell rather than a green
                      // one.
                      const best = row.best_pick === pick
                      return (
                        <td key={pick} className="gbp-cell">
                          <span
                            className={`gbp-chip mono ${tintFor(chance)}${best ? ' gbp-best' : ''}`}
                            title={best ? `Plan on him at pick ${pick}` : undefined}
                          >
                            {chance === null ? '—' : `${chance}%`}
                          </span>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  )
}
