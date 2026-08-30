import { useEffect, useMemo, useState } from 'react'
import { fetchFavoritesOutlook, type FavoritesOutlook } from '../api'

// MY GUYS: WHEN TO TAKE HIM, AND WILL HE BE THERE.
//
// The starred list says who somebody wants. It never said whether they can
// have them, and from the eighth seat of a twelve-team league the honest
// answer for half the list is no.
//
// TWO COLUMNS AND NOT A GRID. This used to be an eight-column heat map: a
// percentage in every cell of every row, banded into three tints, with the
// one cell that mattered marked. That is a table to study, and a reader has
// two questions -- when do I take him, and will he be there when I do. So it
// is those two, in words, with the number beside them.
//
// EVERY NUMBER IS THE SERVER'S (`GET /api/account/favorites/outlook`), read
// off the same counted draft corpus the live room reads, and the round is the
// same walk the plan above is drawn from. Nothing is computed here: a second
// answer to "when do I take him" that disagreed with the plan by two rounds
// would be worse than no card at all.

/** Where the tint changes. Deliberately not a gradient: three bands is a
 *  decision ("plan on him", "maybe", "no"), and a continuous ramp over
 *  twenty-five rows is a heat map nobody reads a row out of. */
const LIKELY = 70
const MAYBE = 40

/** The chip's tint. Exported because the thresholds are the card's whole
 *  claim and a test should be able to state them without a DOM.
 *
 *  TAKES THE ROUNDED NUMBER, the one the chip actually prints. Banding the
 *  raw value instead put 69.6% in a green chip reading 70% and 39.5% in an
 *  amber one reading 40%, which is a card disagreeing with itself. */
export function tintFor(chance: number | null): string {
  if (chance === null) return 'myg-chip-none'
  if (chance >= LIKELY) return 'myg-chip-hi'
  if (chance >= MAYBE) return 'myg-chip-mid'
  return 'myg-chip-lo'
}

export default function MyGuysTable({ players, teams, slot, onEdit }: {
  /** The saved ids. Not rendered -- the server sends its own rows -- but a
   *  changed list is a changed answer, and it is what keeps the request cache
   *  from serving the pre-save one. */
  players: string[]
  /** The seat this whole page is about. Handed down rather than picked here:
   *  the plan above reads the same seat, and two answers about "your next
   *  draft" that disagreed about which seat it was would be worse than one.
   *
   *  NULL IS A LEAGUE THE PLAN CANNOT READ -- fewer than four teams or more
   *  than sixteen. The list is still theirs and this is still where it is
   *  edited, so the section stays: what goes is the two columns that need a
   *  seat to be true, which print a dash. Deleting the whole card instead
   *  took the Edit button and the picker with it, which is a reader locked
   *  out of their own list by the size of their league. */
  teams: number | null
  slot: number | null
  /** Opens the picker. */
  onEdit: () => void
}) {
  const [outlook, setOutlook] = useState<FavoritesOutlook | null>(null)
  const [error, setError] = useState<string | null>(null)
  const tag = useMemo(() => players.join(','), [players])
  // Whether there is a seat to answer "when" and "will he" for. The request
  // still goes out without one -- a name and a face are the same at every
  // seat -- and everything that is not is left undrawn below.
  const seated = teams !== null && slot !== null

  useEffect(() => {
    if (players.length === 0) return
    let cancelled = false
    setOutlook(null)
    setError(null)
    fetchFavoritesOutlook(teams, slot, tag)
      .then((body) => { if (!cancelled) setOutlook(body) })
      .catch((err) => { if (!cancelled) setError(String(err?.message || err)) })
    return () => { cancelled = true }
  }, [teams, slot, tag, players.length])

  // The turn "will he be there?" is asked about. Before a draft your next
  // pick is your first one, and naming it in the header is what keeps the
  // column from reading as a fact about the player.
  const nextPick = seated ? outlook?.picks[0] ?? null : null
  // The last turn the plan covers. A favourite it never reaches for is one
  // to take after all of them, which is a sentence rather than a dash.
  const lastPick = seated
    ? outlook?.picks[outlook.picks.length - 1] ?? null : null

  return (
    <section className="db-sec myg" aria-labelledby="myg-h">
      <div className="db-sec-head">
        <h2 className="db-sec-title" id="myg-h">My guys</h2>
        {players.length > 0 && (
          <span className="mono db-sec-count">{players.length}</span>
        )}
        <span className="db-sec-note">
          Starred in the draft room, and the plan reaches for them a round
          early.
        </span>
        <button type="button" className="db-go db-go-view myg-edit"
                onClick={onEdit}>
          {players.length === 0 ? 'Pick my guys' : 'Edit'}
        </button>
      </div>

      {/* Said once, above the dashes, rather than left for a reader to work
          out from two columns of nothing. */}
      {!seated && players.length > 0 && (
        <p className="db-sec-say">
          Your league is not a size the plan reads yet, so there is no seat to
          say when to take these players or whether they will last. The list
          itself is unaffected, and so is the draft room.
        </p>
      )}

      {/* AN EMPTY SCREEN IS AN INVITATION. One sentence saying what starring
          does for them, and the button that does it -- not a table of nothing
          with a heading over it. */}
      {players.length === 0 ? (
        <p className="myg-empty">
          Star five to twenty-five players you want this year. The draft room
          stars them, and the plan takes them a round earlier than it would
          take anybody else.
        </p>
      ) : (
        <div className="db-card myg-card">
          {error !== null && <p className="db-error">{error}</p>}

          {outlook === null && error === null && (
            <div className="myg-skel" aria-hidden="true">
              {players.slice(0, 6).map((id) => (
                <div key={id} className="myg-skel-row" />
              ))}
            </div>
          )}

          {outlook !== null && (
            <table className="myg-table">
              <thead>
                <tr>
                  <th scope="col" className="myg-who">Player</th>
                  <th scope="col" className="myg-when">When to take him</th>
                  <th scope="col" className="myg-will">
                    Will he be there?
                    {nextPick !== null && (
                      <span className="myg-at mono"> at pick {nextPick}</span>
                    )}
                  </th>
                </tr>
              </thead>
              <tbody>
                {outlook.players.map((row) => {
                  // Rounded ONCE, then banded and printed from the same
                  // number, so the tint can never contradict the text.
                  const raw = seated ? row.avail[0] : null
                  const chance = raw === null || raw === undefined
                    ? null : Math.round(raw)
                  // The server answered for ITS default seat when this page
                  // had none to send; the round it named is about that seat
                  // and not this reader's, so it is not printed.
                  const round = seated ? row.plan_round : null
                  return (
                    <tr key={row.player_id}>
                      <th scope="row" className="myg-who">
                        <span className="myg-who-in">
                          {row.headshot !== null && (
                            <img className="myg-face" src={row.headshot} alt=""
                                 width={24} height={24} loading="lazy" />
                          )}
                          <span className="myg-name">
                            {row.name ?? row.player_id}
                          </span>
                          {row.position !== null && (
                            <span className={`pos-badge pos-badge-${row.position.toLowerCase()}`}>
                              {row.position}
                            </span>
                          )}
                        </span>
                      </th>
                      <td className="myg-when">
                        {round !== null
                          ? <span className="myg-round">Round {round}</span>
                          : (
                            <span className="myg-late">
                              {lastPick === null
                                ? '—'
                                : `Later than pick ${lastPick}`}
                            </span>
                          )}
                      </td>
                      <td className="myg-will">
                        <span className={`myg-chip mono ${tintFor(chance)}`}>
                          {chance === null ? '—' : `${chance}%`}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </section>
  )
}
