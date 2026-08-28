import { memo } from 'react'

// THE CARD THAT OPENS THE PICKER, and the whole of what the dashboard knows
// about favourites: how many there are.
//
// It deliberately does NOT list them. Names would mean the board -- 250 rows,
// ~185 KB -- on every visit to the signed-in page, for a card nobody came here
// to read; the list a reader actually wants to see is the one in the dialog
// this button opens, where the board has a job to do. Two states, one card:
// an invitation when there is nothing saved, a count and a way back in when
// there is.
export default memo(function YourGuys({ count, onOpen }: {
  count: number
  /** Opens the picker. Stable (`useCallback` in Dashboard), or the memo
   *  above is decoration. */
  onOpen: () => void
}) {
  return (
    <section className="db-sec">
      <div className="db-sec-head">
        <h2 className="db-sec-title">Your guys</h2>
        {count > 0 && <span className="mono db-sec-count">{count}</span>}
      </div>
      <div className="db-card db-fav-card">
        <p className="db-fav-note">
          {count === 0
            ? 'The players you want on your team this year. Pick five to '
              + 'twenty-five, and the draft room stars them while the plan '
              + 'reaches for them a round earlier than it would reach for '
              + 'anybody else.'
            : `${count} saved. The draft room stars them, and the plan reaches `
              + 'for them a round earlier than it would reach for anybody else.'}
        </p>
        <button type="button" className="db-go db-league-go db-go-view"
                onClick={onOpen}>
          {count === 0 ? 'Pick your guys' : 'Edit'}
        </button>
      </div>
    </section>
  )
})
