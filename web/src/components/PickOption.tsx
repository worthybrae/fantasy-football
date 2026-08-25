import { SEASON_GAMES } from './draft/weeks'
import '../pickOption.css'

// ONE PLAYER AS THE ARCHIVE KNOWS HIM: a face, a position, a name, how often
// the room takes him at the turn in question, and what he is worth if it
// does.
//
// This lives here rather than in either page because both pages ask the
// corpus the same question and used to answer it in two different shapes:
// the waiting room drew a 34px face with the position badged and the share
// and projection stacked under the name (`/api/market/at-picks`), and the
// archive drew a flat row with a coloured dot standing in for the badge and
// everything on one line (`/api/market/slot/{n}`). Same payload fields, same
// meaning, two looks -- so a reader who walked from one page to the other
// had to re-learn the row. The waiting room's shape won: the badge says the
// position outright where the dot needed the legend two panels over, and two
// short lines fit a name that a single line has to clip.
//
// It is a card, not a link, unless a caller gives it somewhere to go. The
// waiting room has nothing to open -- the draft has not started and the
// profile is the room's own popup -- and the archive opens the profile over
// the page.

/** Per game, the way every projection in this app is printed. The board
 *  holds a season total; a drafter compares a per-game number in his head,
 *  and 238.5 here beside a 14.0 in the room would be one projection in two
 *  currencies. Null stays null -- see the meta line, which prints nothing
 *  rather than a dash. */
function ppg(points: number | null): string | null {
  return points === null ? null : (points / SEASON_GAMES).toFixed(1)
}

export interface PickOptionProps {
  name: string
  position: string
  headshot: string | null
  /** Of the times a person held this turn, the share who took him. */
  share: number
  /** The board's projected season total, divided here. */
  projPoints: number | null
  /** Per-game delta on his recent seasons (the board's proj_change): how far
   *  above or below what he just did this projection prices him. Omitted or
   *  null shows nothing. */
  projChange?: number | null
  /** The count behind the share -- "43 of 321 recorded picks here". On the
   *  title rather than in the row: the share IS the claim, and the sample
   *  behind it is what somebody checks, not what he reads. */
  shareTitle?: string
  /** Given, the whole card becomes a button. Absent, it is a plain tile. */
  onOpen?: () => void
}

export default function PickOption({
  name, position, headshot, share, projPoints, projChange, shareTitle, onOpen,
}: PickOptionProps) {
  const perGame = ppg(projPoints)
  // Hidden when it rounds to nothing: a grey ▲0.0 is noise on every row.
  const change = projChange == null || Math.abs(projChange) < 0.05 ? null : projChange
  const body = (
    <>
      {/* `alt=""` -- the name is printed beside it, and a described face
          would be the second telling of the same fact. A player the board
          has no photo of (every ADP-only rookie and defense, which have no
          roster row to carry one) gets the empty disc rather than a broken
          image, so the tiles in a row stay aligned. */}
      {headshot ? (
        <img className="pick-opt-face" src={headshot} alt=""
             width={34} height={34} loading="lazy" />
      ) : (
        <span className="pick-opt-face is-blank" aria-hidden="true" />
      )}
      <span className="pick-opt-who">
        <span className="pick-opt-nameline">
          <span className={`pos-badge pos-badge-${position.toLowerCase()}`}>
            {position}
          </span>
          <span className="pick-opt-name">{name}</span>
        </span>
        <span className="mono pick-opt-meta">
          <span className="pick-opt-share" title={shareTitle}>
            {Math.round(share * 100)}%
          </span>
          {/* Silent rather than dashed when the board cannot answer: a column
              of em dashes would be the page insisting on a figure it has
              nothing to put in. */}
          {perGame !== null && <span className="pick-opt-ppg">{perGame} ppg</span>}
          {perGame !== null && change !== null && (
            <span className={`pick-opt-move ${change > 0 ? 'is-up' : 'is-down'}`}
                  title={`Projected ${change > 0 ? 'up' : 'down'} ${Math.abs(change).toFixed(1)} points a game on last season`}>
              {change > 0 ? '▲' : '▼'}{Math.abs(change).toFixed(1)}
            </span>
          )}
        </span>
      </span>
    </>
  )

  if (onOpen === undefined) return <div className="pick-opt">{body}</div>
  return (
    <button type="button" className="pick-opt" onClick={onOpen}>{body}</button>
  )
}
