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
export default memo(function YourGuys({ players, onOpen }: {
  /** The saved ids, in their saved order. */
  players: string[]
  /** Opens the picker. Stable (`useCallback` in Dashboard), or the memo
   *  above is decoration. */
  onOpen: () => void
}) {
  const named = useMemo(() => {
    const known = namesFor(players)
    return players.map((id, i) => ({
      id, name: known[i]?.name ?? null, position: known[i]?.position ?? null,
    }))
  }, [players])
  const anyNamed = named.some((row) => row.name !== null)

  return (
    <section className="db-sec">
      <div className="db-sec-head">
        <h2 className="db-sec-title">Your guys</h2>
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
              ? 'The players you want on your team this year. Pick five to '
                + 'twenty-five, and the draft room stars them while the plan '
                + 'reaches for them a round earlier than it would reach for '
                + 'anybody else.'
              : 'The draft room stars them, and the plan reaches for them a '
                + 'round earlier than it would reach for anybody else.'}
          </p>
          <button type="button" className="db-go db-league-go db-go-view"
                  onClick={onOpen}>
            {players.length === 0 ? 'Pick your guys' : 'Edit'}
          </button>
        </div>
      </div>
    </section>
  )
})
