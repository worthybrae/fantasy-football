import type { SimilarPlayer } from '../../api'
import PopCard from './PopCard'
import { fmtSigned, type Cohort } from './payload'

// Seasons that looked like his, and what each of them did the year after.
//
// The card is one claim in four rows: this has happened before, and here is
// where it went. The MOVE is the column that matters -- a comparable season
// is only interesting because of what followed it -- so the other three
// columns are the smallest identification of the season that carries it.
//
// `stat_twins` only. `value_neighbors` is the payload's fallback for a
// player with no stat line to match, and those rows are not seasons at all:
// they are today's board neighbours, which `ValueNeighbors` already draws as
// "Near you". Two neighbour cards on one popup is the bug this guard exists
// to prevent.
const COMPS = 4

function seasonTag(season: number): string {
  return `’${String(season).slice(-2)}`
}

// The note: what the whole cohort did, not what these four did. The four
// rows are the shape of the thing; "13 of 19 declined" is whether the shape
// is common -- and it comes from `cohort`, which is a wider band than
// `similar` (19 seasons against these four) and is the reason the card can
// say anything about frequency at all. With no cohort the age the payload
// matched on is the next-best yardstick, and with neither the head says
// nothing rather than something empty.
function verdict(cohort: Cohort | null, targetAge: number | null): string | undefined {
  if (cohort !== null && cohort.n > 0) return `${cohort.declined} of ${cohort.n} declined`
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
  if (mode !== 'stat_twins') return null
  // A twin the payload cannot date or score is not a season anyone can read
  // against his: the row would be a name and two dashes.
  const rows = players
    .filter((p): p is SimilarPlayer & { season: number; ppg: number } => (
      p.season !== null && p.ppg !== null
    ))
    .slice(0, COMPS)
  if (rows.length === 0) return null

  return (
    // Wider than an even half: this card holds four columns and a sentence
    // of a heading, and News beside it holds two headlines that wrap.
    <PopCard title="Comparable seasons" note={verdict(cohort, targetAge)} className="is-wide">
      <div className="pp-pop-comps">
        {/* Outside the list, as the game log's head is: it names the columns
            rather than being one of the seasons in them. */}
        <div className="pp-pop-comps-head">
          <span className="pp-pop-comp-name">Season that looked like this</span>
          <span className="pp-pop-comp-yr">Yr</span>
          <span className="pp-pop-comp-ppg">PPG</span>
          <span className="pp-pop-comp-next">Next</span>
        </div>
        {/* Ordered: the payload sorts these by similarity, so the first row
            is the closest season to his and the list markup says so. */}
        <ol className="pp-pop-comp-list">
          {rows.map((p) => {
            const move = p.next_ppg === null ? null : p.next_ppg - p.ppg
            const cells = (
              <>
                <span className="pp-pop-comp-name">{p.name}</span>
                <span className="mono pp-pop-comp-yr">{seasonTag(p.season)}</span>
                <span className="mono pp-pop-comp-ppg">{p.ppg.toFixed(1)}</span>
                {/* Green up, red down -- the same two tokens the rest of the
                    popup spends on "better" and "worse". */}
                <span className={`mono pp-pop-comp-next${
                  move === null || move === 0 ? '' : move > 0 ? ' is-good' : ' is-bad'}`}
                >
                  {fmtSigned(move, 1)}
                </span>
              </>
            )
            const key = `${p.player_id ?? p.name}-${p.season}`
            // A twin can predate this year's board -- Todd Gurley's 2017 is
            // not a player anyone can draft in 2026 -- and the payload marks
            // those with a null rank. Opening one would 404 the fetch and
            // wipe the popup that is currently up, so only a twin the board
            // still knows is a control. Same rule ValueNeighbors follows.
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
      </div>
    </PopCard>
  )
}
