import { memo, useMemo } from 'react'
import { namesFor } from '../lib/playerNames'

// THE CARD THAT OPENS THE PICKER.
//
// It names the players it can and it never fetches the board to do it.
// `/api/players` is 250 rows with a photograph each -- about 185 KB -- and a
// card that wanted twenty-five names used to be the reason the whole signed-in
// page paid for it on every visit. The names come from whatever the last board
// fetch left behind in this tab instead (lib/playerNames.ts), which in
// practice is the first time the picker below is opened, and from then on for
// the rest of the visit.
//
// So there are three states, and each of them is honest about what it knows:
// an invitation when nothing is saved, the list when the names are known, and
// a count when they are not -- because a column of player ids is a fact about
// our storage rather than about somebody's team.
//
// IT NO LONGER OWNS A HEADING. This card and the availability grid beside it
// are two halves of one answer -- who you want, and whether you can have them
// -- so the dashboard puts them under one "Your guys" heading and each of them
// draws a column under it. See Dashboard.tsx.
export default memo(function YourGuys({ players, onOpen, refresh = 0 }: {
  /** The saved ids, in their saved order. */
  players: string[]
  /** Opens the picker. Stable (`useCallback` in Dashboard), or the memo
   *  above is decoration. */
  onOpen: () => void
  /** A counter the page bumps when something might have taught this tab a
   *  name -- closing the picker, most of all, since opening it is what
   *  fetches the board these names come from. The names live in a module,
   *  so nothing else would tell this card to look again. */
  refresh?: number
}) {
  const named = useMemo(() => {
    const known = namesFor(players)
    return players.map((id, i) => ({
      id, name: known[i]?.name ?? null, position: known[i]?.position ?? null,
    }))
    // `refresh` is a dependency on purpose: it is the whole mechanism.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [players, refresh])
  const anyNamed = named.some((row) => row.name !== null)

  return (
    <div className="db-guys-col">
      <div className="db-guys-head">
        <h3 className="db-guys-title">In your order</h3>
        {players.length > 0 && (
          <span className="mono db-sec-count">{players.length}</span>
        )}
      </div>
      <div className="db-card db-fav-card">
        {players.length > 0 && (anyNamed ? (
          <ul className="db-fav-list">
            {named.map((row, i) => (
              <li key={row.id} className="db-fav-row">
                <span className="db-fav-ord mono">{i + 1}</span>
                {row.position && (
                  <span className={`pos-badge pos-badge-${row.position.toLowerCase()}`}>
                    {row.position}
                  </span>
                )}
                <span className="db-fav-name">{row.name ?? '—'}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="db-fav-count">
            {players.length} {players.length === 1 ? 'player' : 'players'} saved.
          </p>
        ))}
        <div className="db-fav-foot">
          <p className="db-fav-note">
            {players.length === 0
              ? 'Nobody saved yet. Pick five to twenty-five players you want '
                + 'this year — the draft room stars them, and the plan reaches '
                + 'for them a round earlier than it would for anybody else.'
              : 'Starred in the draft room, and the plan reaches for them a '
                + 'round earlier than it would for anybody else. In your '
                + 'order: the plan starts at the top.'}
          </p>
          {/* `db-fav-go`, not `db-league-go`. It wore the league card's own
              button, which carries `flex: 1 1 auto` so it can stretch across
              the foot of a card that has nothing else in it -- in this card
              it has a paragraph beside it, and the result was a 500px slab
              reading "Edit". Same quiet fill, sized to its own label, and
              the same button in both of this card's states. */}
          <button type="button" className="db-go db-go-view db-fav-go"
                  onClick={onOpen}>
            {players.length === 0 ? 'Pick your guys' : 'Edit'}
          </button>
        </div>
      </div>
    </div>
  )
})
