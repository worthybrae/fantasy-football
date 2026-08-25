import type { Vegas } from './payload'
import PopCard from './PopCard'
import { ordinal } from './payload'
import { CARD_HINTS } from './hints'

// What the betting market prices this player's offence at, and what it
// prices him at.
//
// The number the rank is computed from -- an implied team total, (total +
// spread) / 2 for the home side, the same arithmetic behind the board's
// `environment` factor -- IS NOT PRINTED. "26.6 implied team pts / game"
// needed a sentence of explaining, could be misread as a projection for the
// player rather than his team, and after all that said nothing the rank
// beneath it did not say better. The placing is the fact; the points were
// the arithmetic that produced it.
//
// THERE IS NO WEEK STRIP ANY MORE. This card used to draw one bar per week
// of the season under the total -- eighteen of them, most greyed out because
// books price the front of a year and a scattering beyond it. It cost 33
// vertical pixels, which is exactly what made this card taller than the
// Market card it shares a row with, and it was answering a question the
// Schedule card two rows up already answers better: which weeks are hard.
// What survives is the provenance -- how many weeks the average is over --
// which is the only thing the strip said that nothing else does.

// How far the two ranks have to part before the row says so in colour.
// Three places: inside that, two rankings over a field of sixty are simply
// agreeing, and painting every row would make the card look like an argument
// nobody is having.
//
// Green means the BOOKS rank him higher than the draft room does -- a lower
// number, RB4 against RB9 -- which is the direction a drafter is shopping
// in, the same sense the board's own edge carries: a player the room is late
// on is the one worth taking. Red is the room ahead of the book.
const RANK_GAP = 3

function gapTone(place: number | null, adpPlace: number | null): string {
  if (place === null || adpPlace === null) return ''
  if (adpPlace - place >= RANK_GAP) return 'delta-tone is-up'
  if (place - adpPlace >= RANK_GAP) return 'delta-tone is-down'
  return ''
}

// How many of his own markets the card shows. Four, which is what the card
// has room for now that the implied total and the week strip are gone -- it
// renders a shade under the Market card it shares a row with, and an empty
// bottom third would be worse than a fourth market. They arrive
// shortest-price first, so the four shown are the four the market likes him
// most for.
const SHOWN_MARKETS = 4

export default function VegasCard({ vegas, position }: {
  vegas: Vegas | null
  position: string
}) {
  const futures = vegas?.futures ?? []
  // Nothing priced about his offence AND nothing priced about him: no card.
  const hasTeam = vegas && vegas.implied !== null && vegas.weeks.length > 0
  if (!vegas || (!hasTeam && futures.length === 0)) return null

  // "RB4", the way the Market card writes a positional rank, or an em dash
  // where the field prices too few of his position to place him in.
  const byPos = (place: number | null) =>
    place === null ? '—' : `${position}${place}`

  const ranked = hasTeam && vegas.rank !== null && vegas.teams !== null

  return (
    // No `is-widest` any more. This card was given the row's largest share
    // when it carried an implied total, a strip of eighteen weeks and four
    // markets; it carries a rank and one to four rows now, while the Market
    // card beside it grew from four lines to seven. The width belongs to
    // whichever card has something to put in it.
    <PopCard title="Vegas" hint={CARD_HINTS.vegas}>
      {hasTeam && (
      <>
      {ranked && (
        // The rank, in the O-line card's own lead treatment -- `pp-pop-lead`
        // is that card's, not a copy of it -- because the two cards are
        // making the same kind of statement: one number, out of the same
        // thirty-two teams, that a reader places instantly. A popup that
        // wrote "13th of 32" two different ways on two cards would be asking
        // to be read twice.
        //
        // The track that used to sit beside it is gone with the second
        // spelling. A bar drawn against thirty-two teams was a second
        // rendering of a number already in plain words.
        <div className="pp-pop-lead">
          <span className="mono pp-pop-lead-value">{ordinal(vegas.rank!)}</span>
          <span className="mono pp-pop-lead-of">of {vegas.teams} offences</span>
        </div>
      )}
      {/* Where the rank comes from, and what the percentages below it are.
          Both are provenance rather than findings, which is why they share
          one 9px line instead of costing two. */}
      {/* Where the rank comes from: the number itself, how many weeks it is
          an average over, and -- when there are markets below -- what those
          percentages are. All three are provenance rather than findings,
          which is why they share one line instead of costing three.

          The implied total is BACK on this line, and only on this line. It
          was a headline figure once, needed a sentence to explain, and was
          misread as a projection for the player rather than for his whole
          offence; as the small print under a rank that has already made the
          point, it is what it always should have been -- the arithmetic,
          available to anyone who wants to check the placing. */}
      <div className="pp-pop-vegas-foot mono">
        {vegas.implied !== null && `${vegas.implied.toFixed(1)} team pts / g · `}
        over {vegas.priced} priced weeks{futures.length > 0 && ' · book prices'}
      </div>
      </>
      )}
      {futures.length > 0 && (
        <div className="pp-pop-futures">
          {/* The columns, named. Two ranks side by side need saying which is
              which, and nine-pixel headings cost one line to save a reader
              guessing at every row. */}
          <div className="pp-pop-futures-head">
            <span />
            <span />
            <span>chance</span>
            <span>vegas</span>
            <span>adp</span>
          </div>
          {futures.slice(0, SHOWN_MARKETS).map((f) => {
            const pct = f.implied_pct
            // The favourite's own price is the top of the track. A market
            // where the leader sits at 18% and one where the leader sits at
            // 45% are different fields, and a bar drawn to a flat 100% would
            // make the same 14% look identical in both.
            const top = Math.max(f.top_pct ?? 0, pct ?? 0)
            const share = pct === null || !top ? 0 : (pct / top) * 100
            return (
              <div
                className="pp-pop-futures-row"
                key={f.market}
                // The one thing the row does not print, kept where a reader
                // who wants it can still find it: the price itself, and the
                // sizes of the two fields the ranks are out of.
                title={`${f.american ? `${f.american} · ` : ''}`
                  + `${ordinal(f.place)} of ${f.field} priced`
                  + (f.adp_place === null ? ''
                    : ` · ${ordinal(f.adp_place)} of ${f.adp_field} by adp`)}
              >
                <span className="pp-pop-futures-label">{f.label}</span>
                <span className="pp-pop-futures-track">
                  <span
                    className="pp-pop-futures-fill"
                    style={{ width: `${Math.min(100, share)}%` }}
                  />
                </span>
                {/* The chance, in the one unit everybody already reads.
                    "+700" is a price; this is what the price MEANS. Under
                    one per cent says so rather than rounding up to 1%:
                    +20000 is a lottery ticket and printing it as the same
                    number as +10000 would flatter both. */}
                <span className="mono pp-pop-futures-pct">
                  {pct === null ? '—'
                    : pct < 1 ? '<1%' : `${Math.round(pct)}%`}
                </span>
                {/* Two ranks over the same position: where the book puts
                    him, and where the room drafts him. Both printed plainly,
                    and the comparison carried by the colour of the first
                    rather than by a third number.

                    This column was a signed difference for a while -- "+10",
                    "-4" -- and it read as an argument with itself. A reader
                    had to hold which way round the subtraction went before
                    the sign meant anything, on every row, and the two ranks
                    it was computed from were right there beside it. Now the
                    vegas rank is green when the book likes him better than
                    the room does (a LOWER number: QB4 against QB9) and red
                    when the room is ahead of the book -- Jalen Hurts, QB15
                    for MVP and QB5 by ADP, is red without needing to say
                    +10. The direction is the whole finding, and colour says
                    a direction without being read.

                    Still on a three-place deadband (RANK_GAP): two rankings
                    over a field of sixty that differ by one are agreeing,
                    and a card painted top to bottom claims a disagreement on
                    every line. */}
                <span className={`mono pp-pop-futures-rank ${gapTone(f.pos_place, f.pos_adp_place)}`}>
                  {byPos(f.pos_place)}
                </span>
                <span className="mono pp-pop-futures-rank">{byPos(f.pos_adp_place)}</span>
              </div>
            )
          })}
          {/* Said once, quietly, and not negotiable: these are book prices,
              so a field's percentages add to well over a hundred. A card
              that printed them as probabilities without saying so would be
              overstating every player on it. It rides on the weeks-priced
              foot above wherever there is one, and only costs its own line
              on a card that has no team lines to carry it. */}
          {!hasTeam && <div className="pp-pop-futures-note">book prices</div>}
        </div>
      )}
    </PopCard>
  )
}
