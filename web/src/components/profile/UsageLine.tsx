import PopCard from './PopCard'
import type { SeasonRow } from './payload'

// What the points were made of, season by season, and how big a share of the
// offence they came off.
//
// This card used to be three recency-weighted per-game rates -- his numbers,
// but flattened into one column, which is the one shape that cannot answer
// the question a reader actually brings to usage: is his role GROWING? A
// back at 54% of snaps three years running and a back who went 54 / 56 / 67
// have the same weighted average and are not the same pick.
//
// Share first, rates under it. A rate is what he did; a share is how much of
// his offence he was, which is the half that survives a coaching change and
// the half a projection is really a bet on.
//
// Three seasons, not the five the panels above draw: a row of five numbers
// at this width stops being readable, and usage older than three years is
// describing a different team.
const SHOWN = 3

// Per-game rates by position, off `scoring/profile.py::_SUMMARY_STATS` -- the
// keys are that list's, not one invented here. Two per position, because the
// share rows above them already spend two lines of the card.
//
// `car` / `tgt` / `yds` are the game log's own words two cards up, not
// abbreviations invented for this table -- four season columns leave the
// label about fifty pixels, and "Carries / g" was being ellipsised into
// "Carri...", which is worse than a short word a reader has already met.
//
// Yards are the sum of both halves: a back who gains 73 on the ground and 31
// through the air had a 105-yard game, and splitting that across two rows
// would spend the card on one stat.
//
// THERE IS NO `K` OR `DST` ENTRY AND THAT IS DELIBERATE. A kicker's season
// row carries no skill columns to read, so the lookup falls through, no rate
// row is built, and -- with no snap or target share either -- the card
// renders nothing, which is the whole of what the payload can say about a
// kicker's usage.
const RATES: Record<string, [string, (keyof SeasonRow)[]][]> = {
  QB: [['Att / g', ['attempts']], ['Pass yds / g', ['pass_yards']]],
  RB: [['Car / g', ['carries']], ['Yds / g', ['rush_yards', 'rec_yards']]],
  WR: [['Tgt / g', ['targets']], ['Yds / g', ['rec_yards', 'rush_yards']]],
  TE: [['Tgt / g', ['targets']], ['Yds / g', ['rec_yards', 'rush_yards']]],
}

function pct(share: number | null | undefined): string | null {
  return share === null || share === undefined ? null : `${Math.round(share * 100)}%`
}

function perGame(season: SeasonRow, keys: (keyof SeasonRow)[]): string | null {
  if (!season.games) return null
  const total = keys.reduce((sum, k) => sum + ((season[k] as number | null) ?? 0), 0)
  return (total / season.games).toFixed(1)
}

export default function UsageLine({ seasons, position }: {
  seasons: SeasonRow[]
  position: string
}) {
  // Oldest on the left, so the row reads the way the panels above it do and
  // the way time does. The payload is newest first.
  const shown = seasons.slice(0, SHOWN).reverse()
  if (shown.length === 0) return null

  const candidates: [string, (string | null)[]][] = [
    ['Snap %', shown.map((s) => pct(s.snap_share))],
    ['Target %', shown.map((s) => pct(s.target_share))],
    ...(RATES[position] ?? []).map(([label, keys]): [string, (string | null)[]] =>
      [label, shown.map((s) => perGame(s, keys))]),
  ]
  // A row nothing can answer is dropped rather than dashed across: a
  // quarterback has no target share, a rookie has no seasons to read, and
  // three em-dashes in a line is a row that costs a reader a line to learn
  // nothing. A row with SOME seasons missing keeps its dashes -- there the
  // gap is the fact.
  const rows = candidates.filter(([, values]) => values.some((v) => v !== null))
  if (rows.length === 0) return null

  return (
    <PopCard title="Usage" note="by season" className="is-wide">
      <div className="pp-pop-seasons">
        <div className="pp-pop-seasons-head">
          <span />
          {shown.map((s) => (
            <span className="mono" key={s.season}>&rsquo;{String(s.season).slice(2)}</span>
          ))}
        </div>
        {rows.map(([label, values]) => (
          <div className="pp-pop-seasons-row" key={label}>
            <span>{label}</span>
            {values.map((v, i) => (
              <span className="mono" key={shown[i].season}>{v ?? '—'}</span>
            ))}
          </div>
        ))}
      </div>
    </PopCard>
  )
}
