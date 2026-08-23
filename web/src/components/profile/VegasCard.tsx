import type { Vegas } from './payload'
import PopCard from './PopCard'
import { ordinal } from './payload'

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

// How many of his own markets the card shows. Four, which is what the card
// has room for now that the implied total and the week strip are gone -- it
// renders a shade under the Market card it shares a row with, and an empty
// bottom third would be worse than a fourth market. They arrive
// shortest-price first, so the four shown are the four the market likes him
// most for.
const SHOWN_MARKETS = 4

export default function VegasCard({ vegas }: { vegas: Vegas | null }) {
  const futures = vegas?.futures ?? []
  // Nothing priced about his offence AND nothing priced about him: no card.
  const hasTeam = vegas && vegas.implied !== null && vegas.weeks.length > 0
  if (!vegas || (!hasTeam && futures.length === 0)) return null

  const ranked = hasTeam && vegas.rank !== null && vegas.teams !== null
    && vegas.teams > 1
  // Better is fuller, the way every other bar in this popup runs: the best
  // offence in the league fills the track and the worst empties it. A rank
  // printed on its own is a number a reader has to place; a rank drawn
  // against its own field places itself.
  const standing = ranked
    ? ((vegas.teams! - vegas.rank!) / (vegas.teams! - 1)) * 100 : 0

  return (
    <PopCard title="Vegas" className="is-widest">
      {hasTeam && (
      <>
      {ranked && (
        // The rank, and now the card's opening line rather than a note under
        // a number: of everything Vegas says about a player's offence it is
        // the one fact that needs no explaining and survives being the only
        // thing a reader takes away.
        <div className="pp-pop-vegas-rank">
          <span className="mono pp-pop-vegas-rank-place">{ordinal(vegas.rank!)}</span>
          <span className="pp-pop-vegas-rank-of">of {vegas.teams} offences</span>
          <span className="pp-pop-vegas-rank-track">
            <span className="pp-pop-vegas-rank-fill" style={{ width: `${standing}%` }} />
          </span>
        </div>
      )}
      {/* Where the rank comes from, and what the percentages below it are.
          Both are provenance rather than findings, which is why they share
          one 9px line instead of costing two. */}
      <div className="pp-pop-vegas-foot mono">
        over {vegas.priced} priced weeks{futures.length > 0 && ' · book prices'}
      </div>
      </>
      )}
      {futures.length > 0 && (
        <div className="pp-pop-futures">
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
                // Everything the row no longer prints, kept where a reader
                // who wants it can still find it: the price itself, and
                // where he sits in the field.
                title={`${f.american ? `${f.american} · ` : ''}`
                  + `${ordinal(f.place)} of ${f.field} priced`}
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
