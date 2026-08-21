import type { ReactNode } from 'react'
import type { SeasonRow } from './payload'

// Where a positional finish stops being useful. Cut points are the number of
// players a 12-team league starts at that position -- one QB and one TE each,
// two RB and two WR plus the flex -- so "inside the first band" means "he was
// a startable player that year" rather than an abstract percentile.
//
// This league runs eight teams, which is MORE forgiving than these bands, so
// they read slightly harsh here on purpose: a finish that would start in any
// league is the honest bar, and a chart that flattered an eight-team roster
// would say a player was fine in a context the reader may not be drafting in
// next year.
const STARTERS: Record<string, number> = { QB: 12, TE: 12, RB: 24, WR: 24, K: 12, DST: 12 }

// Beyond three tiers of starters a finish is off the map for fantasy, and
// stretching the axis to reach a TE40 would squash every meaningful season
// into the top inch of the chart.
const TIERS = 3

// Five bands, as fractions of the league's starters at that position, so the
// same code reads a QB and an RB correctly without a table of magic numbers.
//
// The top two exist because three bands did not discriminate where it
// mattered: an RB10, an RB3 and an RB1 all landed in one colour, which made
// a chart of a good player's career a single flat shade and said nothing
// about his best year. Quartering the starter count separates "the best at
// the position" from "comfortably startable", which is the distinction a
// drafter is actually reading for.
function tone(finish: number, starters: number): string {
  if (finish <= starters / 4) return 'is-elite'
  if (finish <= starters / 2) return 'is-strong'
  if (finish <= starters) return 'is-starter'
  if (finish <= starters * 2) return 'is-fringe'
  return 'is-out'
}

// Taller is better, which needs saying because rank runs the other way: a
// finish of 1 is the best season and has to be the tallest bar, so the scale
// is inverted rather than plotted raw.
function height(finish: number, starters: number): number {
  const floor = starters * TIERS
  const scaled = 1 - Math.min(finish, floor) / floor
  // A floor of 6% so the worst season is still a visible mark rather than a
  // gap the eye reads as missing data.
  return 6 + scaled * 94
}

/** Positional finish, season by season -- the shape of a career in one row.
 *
 * Charted on `pos_finish`, which ranks by TOTAL points, because that is the
 * number people mean by "he finished RB3". `pos_rank_ppg` ranks the same
 * season by points per game and the two come apart whenever a player misses
 * time, so the per-game figure rides along in the hover text instead of
 * quietly replacing it. A season with neither is drawn as an empty slot: the
 * year happened, we just cannot rank it. */
export default function SeasonFinish(
  { seasons, position }: { seasons: SeasonRow[]; position: string },
): ReactNode {
  const rows = [...seasons].sort((a, b) => a.season - b.season)
  if (rows.length === 0) return null
  const starters = STARTERS[position] ?? 24

  return (
    <div className="finish-chart">
      {rows.map((s) => {
        const finish = s.pos_finish
        const label = finish === null ? '—' : `${position}${finish}`
        const ppg = s.pos_rank_ppg === null
          ? ''
          : ` · ${position}${s.pos_rank_ppg} by points per game`
        return (
          <div key={s.season} className="finish-col"
               title={finish === null
                 ? `${s.season}: not ranked at ${position}`
                 : `${s.season}: finished ${position}${finish} on total points${ppg}`
                   + ` · ${s.games ?? '?'} games`}>
            <span className="finish-label">{label}</span>
            <div className="finish-track">
              {finish !== null && (
                <div className={`finish-bar ${tone(finish, starters)}`}
                     style={{ height: `${height(finish, starters)}%` }} />
              )}
            </div>
            <span className="finish-year mono">{`'${String(s.season).slice(2)}`}</span>
          </div>
        )
      })}
    </div>
  )
}
