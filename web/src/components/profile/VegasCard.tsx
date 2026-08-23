import type { Vegas } from './payload'
import PopCard from './PopCard'
import { ordinal } from './payload'

// What the betting market prices this player's offence at.
//
// An implied team total is the half of a line that is about scoring:
// (total + spread) / 2 for the home side. It is the same arithmetic behind the
// board's `environment` factor, so this card and that percentile cannot
// disagree about whose offence the market likes.
//
// A TEAM total, and the note says so. Nothing here claims how many of
// Detroit's 27.8 points go to one back -- a card that let a reader take it for
// a player projection would be worse than no card.
//
// The bars are the weeks a line is actually posted for. Books price the front
// of a season and a scattering beyond it, so an unpriced week draws the same
// baseline mark every other chart in this popup uses for "nothing here yet" --
// never a zero-height bar, which would read as a game nobody expects points
// in.

// The band the bars are drawn against. Implied totals live between about 14
// and 33 in a normal season; anchoring to a fixed band rather than to this
// team's own range means two players' cards can be compared, which is the
// whole point of a number with a rank next to it.
const FLOOR = 14
const CEILING = 33

function height(implied: number): number {
  const share = (implied - FLOOR) / (CEILING - FLOOR)
  return Math.max(6, Math.min(1, share) * 100)
}

// How many of his own markets the card shows. Three is what fits under the
// week bars, and they arrive shortest-price first, so the three shown are the
// three the market likes him most for.
const SHOWN_MARKETS = 3

export default function VegasCard({ vegas }: { vegas: Vegas | null }) {
  const futures = vegas?.futures ?? []
  // Nothing priced about his offence AND nothing priced about him: no card.
  const hasTeam = vegas && vegas.implied !== null && vegas.weeks.length > 0
  if (!vegas || (!hasTeam && futures.length === 0)) return null

  const byWeek = new Map(vegas.weeks.map((w) => [w.week, w]))
  const weeks = vegas.weeks_total ?? 18
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
      <div className="pp-pop-vegas-head">
        <span className="mono pp-pop-vegas-figure">{vegas.implied}</span>
        {/* The unit, spelled out. "26.8" beside a rank could be read as
            anything; this is the team's points, not his. */}
        <span className="pp-pop-vegas-unit">implied team pts / game</span>
      </div>
      {ranked && (
        // The rank, promoted out of the card's header note and given the
        // size the number deserves: of everything on this card it is the one
        // fact that needs no explaining and survives being the only thing a
        // reader takes away.
        <div className="pp-pop-vegas-rank">
          <span className="mono pp-pop-vegas-rank-place">{ordinal(vegas.rank!)}</span>
          <span className="pp-pop-vegas-rank-of">of {vegas.teams} offences</span>
          <span className="pp-pop-vegas-rank-track">
            <span className="pp-pop-vegas-rank-fill" style={{ width: `${standing}%` }} />
          </span>
        </div>
      )}
      <div className="pp-pop-strip">
        {Array.from({ length: weeks }, (_, i) => i + 1).map((week) => {
          const game = byWeek.get(week)
          if (!game || game.implied === null) {
            return <div className="ctip-col-none" key={week} title={`wk ${week} · no line yet`} />
          }
          return (
            <div
              className="pp-pop-vegas-bar"
              key={week}
              style={{ height: `${height(game.implied)}%` }}
              title={`wk ${week} ${game.home ? 'vs ' : 'at '}${game.opponent ?? '—'} · ${game.implied} implied`}
            />
          )
        })}
      </div>
      <div className="pp-pop-vegas-foot mono">
        {vegas.priced}/{weeks} weeks priced
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
                // The price itself is still here for anyone who reads odds,
                // it is just no longer the thing the row is made of.
                title={f.american ? `${f.american} at the book` : undefined}
              >
                <div className="pp-pop-futures-line">
                  <span className="pp-pop-futures-label">{f.label}</span>
                  {/* The chance, in the one unit everybody already reads.
                      "+700" is a price; this is what the price MEANS.
                      Under one per cent says so rather than rounding up to
                      1%: +20000 is a lottery ticket and printing it as the
                      same number as +10000 would flatter both. */}
                  <span className="mono pp-pop-futures-pct">
                    {pct === null ? '—'
                      : pct < 1 ? '<1%' : `${Math.round(pct)}%`}
                  </span>
                  <span className="mono pp-pop-futures-place">
                    {ordinal(f.place)} of {f.field}
                  </span>
                </div>
                <span className="pp-pop-futures-track">
                  <span
                    className="pp-pop-futures-fill"
                    style={{ width: `${Math.min(100, share)}%` }}
                  />
                </span>
              </div>
            )
          })}
          {/* Said once, quietly, and not negotiable: these are book prices,
              so the field's percentages add to well over a hundred. A card
              that printed them as probabilities without saying so would be
              overstating every player on it. */}
          <div className="pp-pop-futures-note">
            book prices · bar is his share of the favourite&rsquo;s
          </div>
        </div>
      )}
    </PopCard>
  )
}
