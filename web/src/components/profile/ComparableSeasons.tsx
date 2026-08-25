import { useState, type ReactNode } from 'react'
import type { SimilarPlayer } from '../../api'
import { year } from '../draft/panels'
import PopCard from './PopCard'
import { CARD_HINTS } from './hints'
import { fmtSigned, type Cohort } from './payload'

// Seasons that looked like his, and what each of them did the year after.
//
// The card is one claim in four rows: this has happened before, and here is
// where it went. The MOVE is the column that matters -- a comparable season
// is only interesting because of what followed it -- so the other three
// columns are the smallest identification of the season that carries it.
//
// Twelve at a time, out of the thirty `build_profile` asks `find_twins` for,
// with the rest a page away. Four was set when the popup was 562px wide and
// this card sat beside two headlines; the tail is worth reaching -- the point
// of the card is what happened NEXT to seasons like his, and four outcomes is
// a sample nobody would draw a line through. The count under the list says
// how deep it goes.
//
// The number is set by the card BESIDE it, not by this one. News shares the
// row, `.pp-pop-row` stretches both to the taller, and News runs about 58px
// per story once a headline wraps to a second line -- five of those is
// twelve rows of this table. Ten left a band of empty panel above the pager.
//
// It cannot be exact and is not meant to be. A player whose headlines all
// fit on one line makes News the shorter card and this the taller one, and
// the slack changes sides. Twelve is the count that is wrong by the least
// across the players, not the count that is right for one of them.
//
// `stat_twins` only. `value_neighbors` is the payload's fallback for a
// player with no stat line to match, and those rows are not seasons at all:
// they are today's board neighbours, which the popup no longer draws as
// "Near you". Two neighbour cards on one popup is the bug this guard exists
// to prevent.
const COMPS = 12

// Hooks must run on every render, so the early returns below cannot come
// before `useState`. The page is reset by React itself rather than by an
// effect: the popup remounts this card when the profile changes (a new
// `playerId` remounts PlayerProfile's whole subtree), so page 2 of one
// player's comps can never be page 2 of the next one's.

// The note: what the whole cohort did, not what the rows below did. The rows
// are the shape of the thing; the note is what happened on average to every
// season like it -- and it comes from `cohort`, a wider band than `similar`
// (166 seasons against these twelve) and the only reason the card can say
// anything general at all. With no cohort the age the payload matched on is
// the next-best yardstick, and with neither the head says nothing rather
// than something empty.
//
// A MEAN, not a count. "91 of 166 declined" is a true sentence that answers
// the wrong question: it says a coin came up tails slightly more than half
// the time, which is nearly every cohort in fantasy football, and says
// nothing about SIZE. Whether the average season like his lost half a point
// a game or four is the thing that changes a pick, and it is one number.
//
// Averaged over the seasons themselves rather than read off `median_change`
// beside them: the median is the middle season's fall, and a tail of players
// who fell off a cliff is exactly what a drafter is being warned about here.
function avgChange(cohort: Cohort): number | null {
  const changes = cohort.players.map((p) => p.change).filter((c) => Number.isFinite(c))
  if (changes.length === 0) return cohort.median_change
  return changes.reduce((sum, c) => sum + c, 0) / changes.length
}

function verdict(cohort: Cohort | null, targetAge: number | null): ReactNode {
  if (cohort !== null && cohort.n > 0) {
    const avg = avgChange(cohort)
    if (avg !== null) {
      const shown = Math.round(avg * 10) / 10
      return (
        <>
          {'avg '}
          <span className={`delta-tone ${shown > 0 ? 'is-up' : shown < 0 ? 'is-down' : ''}`}>
            {`${shown > 0 ? '+' : ''}${shown.toFixed(1)}`}
          </span>
          {' / g'}
        </>
      )
    }
  }
  if (targetAge !== null) return `age ${targetAge}`
  return undefined
}

export default function ComparableSeasons({
  mode, players, targetAge, cohort, onSelectPlayer,
}: {
  mode: 'stat_twins' | 'value_neighbors'
  players: SimilarPlayer[]
  targetAge: number | null
  cohort: Cohort | null
  onSelectPlayer: (id: string) => void
}) {
  const [page, setPage] = useState(0)
  if (mode !== 'stat_twins') return null
  // A twin the payload cannot date or score is not a season anyone can read
  // against his: the row would be a name and two dashes.
  const all = players.filter(
    (p): p is SimilarPlayer & { season: number; ppg: number } => (
      p.season !== null && p.ppg !== null
    ))
  if (all.length === 0) return null
  // Clamped rather than trusted: the payload can be shorter on a refetch
  // (a league's scoring changed, and with it which seasons match), and a
  // page number left pointing past the end would draw an empty card.
  const pages = Math.max(1, Math.ceil(all.length / COMPS))
  const at = Math.min(page, pages - 1)
  const rows = all.slice(at * COMPS, at * COMPS + COMPS)

  return (
    // Wider than an even half: this card holds four columns and a sentence
    // of a heading, and News beside it holds two headlines that wrap.
    <PopCard
      title="Comparable seasons"
      note={verdict(cohort, targetAge)}
      hint={CARD_HINTS.comps}
      className="is-widest"
    >
      <div className="pp-pop-comps">
        {/* Outside the list, as the game log's head is: it names the columns
            rather than being one of the seasons in them. */}
        <div className="pp-pop-comps-head">
          <span className="pp-pop-comp-name">Season that looked like this</span>
          <span className="pp-pop-comp-yr">Yr</span>
          <span className="pp-pop-comp-ppg">PPG</span>
          <span className="pp-pop-comp-next">Next</span>
          {/* The distance behind the ordering, said out loud. The rows were
              already sorted by it; a reader could see that row 1 came first
              but not whether it was barely ahead of row 4 or in a different
              league from it. */}
          <span className="pp-pop-comp-sim">Match</span>
        </div>
        {/* Ordered: the payload sorts these by similarity, so the first row
            is the closest season to his and the list markup says so. */}
        <ol className="pp-pop-comp-list">
          {rows.map((p) => {
            const move = p.next_ppg === null ? null : p.next_ppg - p.ppg
            const cells = (
              <>
                <span className="pp-pop-comp-name">{p.name}</span>
                <span className="mono pp-pop-comp-yr">{year(p.season)}</span>
                <span className="mono pp-pop-comp-ppg">{p.ppg.toFixed(1)}</span>
                {/* Green up, red down -- the same two tokens the rest of the
                    popup spends on "better" and "worse". */}
                <span className={`mono pp-pop-comp-next${
                  move === null || move === 0 ? '' : move > 0 ? ' is-good' : ' is-bad'}`}
                >
                  {fmtSigned(move, 1)}
                </span>
                <span className="mono pp-pop-comp-sim">
                  {p.similarity === null ? '—' : `${Math.round(p.similarity)}%`}
                </span>
              </>
            )
            const key = `${p.player_id ?? p.name}-${p.season}`
            // A twin can predate this year's board -- Todd Gurley's 2017 is
            // not a player anyone can draft in 2026 -- and the payload marks
            // those with a null rank. Opening one would 404 the fetch and
            // wipe the popup that is currently up, so only a twin the board
            // still knows is a control.
            const id = p.rank === null ? null : p.player_id
            if (id === null) return <li className="pp-pop-comp" key={key}>{cells}</li>
            return (
              <li key={key}>
                <button
                  type="button"
                  className="pp-pop-comp"
                  onClick={() => onSelectPlayer(id)}
                  title={`Open ${p.name}`}
                >
                  {cells}
                </button>
              </li>
            )
          })}
        </ol>
        {/* Only when there IS a rest. One page of comps is the common case
            for a player the age filter leaves few matches for, and a pager
            under it would be furniture over a control that does nothing. */}
        {pages > 1 && (
          <div className="pp-pop-comps-pager">
            <span className="mono pp-pop-comps-count">
              {at * COMPS + 1}–{at * COMPS + rows.length} of {all.length}
            </span>
            <button
              type="button"
              className="pp-pop-comps-page"
              onClick={() => setPage(at - 1)}
              disabled={at === 0}
              aria-label="Previous comparable seasons"
            >
              ‹
            </button>
            <button
              type="button"
              className="pp-pop-comps-page"
              onClick={() => setPage(at + 1)}
              disabled={at >= pages - 1}
              aria-label="More comparable seasons"
            >
              ›
            </button>
          </div>
        )}
      </div>
    </PopCard>
  )
}
