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
  const note = hasTeam && vegas.rank !== null && vegas.teams !== null
    ? `${ordinal(vegas.rank)} of ${vegas.teams}` : undefined

  return (
    <PopCard title="Vegas" note={note}>
      {hasTeam && (
      <>
      <div className="pp-pop-vegas-head">
        <span className="mono pp-pop-vegas-figure">{vegas.implied}</span>
        {/* The unit, spelled out. "26.8" beside a rank could be read as
            anything; this is the team's points, not his. */}
        <span className="pp-pop-vegas-unit">implied team points a game</span>
      </div>
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
        {vegas.priced} of {weeks} weeks priced
      </div>
      </>
      )}
      {futures.length > 0 && (
        <div className="pp-pop-futures">
          {futures.slice(0, SHOWN_MARKETS).map((f) => (
            <div className="pp-pop-futures-row" key={f.market}>
              <span className="pp-pop-futures-label">{f.label}</span>
              {/* The price, then what it is worth knowing about the price:
                  "+700" means nothing to most readers, "4th of 61" means he
                  is near the front of a big field. */}
              <span className="mono pp-pop-futures-price">{f.american}</span>
              <span className="mono pp-pop-futures-place">
                {ordinal(f.place)} of {f.field}
              </span>
            </div>
          ))}
        </div>
      )}
    </PopCard>
  )
}
