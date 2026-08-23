import PopCard from './PopCard'
import type { ProjectedUsage, SeasonRow } from './payload'

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
// Four. Five reached back to a season most of these players were not the
// same player in -- and the columns are what pay for it: at four they get
// their width back, which is what keeps a percentage legible. Older usage is
// a different team; seeing WHEN a role changed is the reason to look at all,
// and four seasons still shows a change.
const SHOWN = 4

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
// The third element is which projected rate answers this row -- the card's
// last column is the season being drafted, drawn the way the panels above
// draw theirs. Null where ESPN projects nothing that fits the row.
type Rate = [string, (keyof SeasonRow)[], keyof ProjectedUsage | null]

const RATES: Record<string, Rate[]> = {
  QB: [['Att / g', ['attempts'], 'attempts'],
    ['Pass yds / g', ['pass_yards'], 'pass_yards']],
  RB: [['Car / g', ['carries'], 'carries'],
    ['Yds / g', ['rush_yards', 'rec_yards'], 'yards']],
  WR: [['Tgt / g', ['targets'], 'targets'],
    ['Yds / g', ['rec_yards', 'rush_yards'], 'yards']],
  TE: [['Tgt / g', ['targets'], 'targets'],
    ['Yds / g', ['rec_yards', 'rush_yards'], 'yards']],
}

// Colour is a MOVE, not a placing. The card used to tint each cell by where
// it fell among that position's spread, which meant twelve filled boxes in a
// card whose whole point is reading across a row -- and the placing was
// already the honest answer to a different question, one the Per game panel
// above it answers with a ranked bar.
//
// So: green if the number is up on the season before it, red if it is down,
// and the ink is the number itself. Two colours over four columns say the
// one thing a share is worth reading back four years for -- which way the
// role is going.
//
// The step a move has to clear to earn a colour, as a fraction of what it
// moved FROM. Relative, not absolute: three points of snap share is a real
// change at 20% and rounding at 90%, and a card that painted the second one
// green would be colouring noise. Below the step the number keeps its
// ordinary ink, which is the card saying the role held.
const MOVE = 0.05

function tone(now: number | null, prev: number | null): string {
  if (now === null || prev === null || prev === 0) return ''
  const move = (now - prev) / Math.abs(prev)
  if (move >= MOVE) return 'delta-tone is-up'
  if (move <= -MOVE) return 'delta-tone is-down'
  return ''
}

// The most recent season before this one that HAS a number. A missing season
// is compared past, not treated as a zero: a player who missed a year and
// came back at his old workload did not go up, and the arithmetic of
// dividing by zero would have said he went up infinitely.
function before(values: (number | null)[], i: number): number | null {
  for (let j = i - 1; j >= 0; j -= 1) if (values[j] !== null) return values[j]
  return null
}

function pct(share: number): string {
  return `${Math.round(share * 100)}%`
}

function rate(value: number): string {
  return value.toFixed(1)
}

function perGame(season: SeasonRow, keys: (keyof SeasonRow)[]): number | null {
  if (!season.games) return null
  const total = keys.reduce((sum, k) => sum + ((season[k] as number | null) ?? 0), 0)
  return total / season.games
}

export default function UsageLine({ seasons, position, projected }: {
  seasons: SeasonRow[]
  position: string
  projected?: ProjectedUsage | null
}) {
  // Oldest on the left, so the row reads the way the panels above it do and
  // the way time does. The payload is newest first.
  const shown = seasons.slice(0, SHOWN).reverse()
  if (shown.length === 0) return null

  // A row is its numbers, the way it prints one, and its projection. The
  // numbers stay numbers until the cell is drawn, because the colour is
  // arithmetic between two of them -- formatting first would leave the row
  // comparing "54%" against "57%" as strings.
  type Row = [string, (number | null)[], (v: number) => string, number | null]

  const candidates: Row[] = [
    // The two shares have no projected column and the em dash says so: the
    // projection table carries no team total worth dividing by and does not
    // project snaps at all. A blank cell here is the card declining to guess,
    // in the same place it would otherwise be guessing hardest.
    ['Snap %', shown.map((s) => s.snap_share ?? null), pct, null],
    ['Target %', shown.map((s) => s.target_share ?? null), pct, null],
    ...(RATES[position] ?? []).map(([label, keys, projKey]): Row => [
      label,
      shown.map((s) => perGame(s, keys)),
      rate,
      projKey === null || !projected ? null
        : (projected[projKey] as number | null) ?? null,
    ]),
  ]
  // A row nothing can answer is dropped rather than dashed across: a
  // quarterback has no target share, a rookie has no seasons to read, and
  // three em-dashes in a line is a row that costs a reader a line to learn
  // nothing. A row with SOME seasons missing keeps its dashes -- there the
  // gap is the fact.
  const rows = candidates.filter(([, values, , proj]) =>
    values.some((v) => v !== null) || proj !== null)
  if (rows.length === 0) return null

  return (
    <PopCard title="Usage" note="by season" className="is-widest">
      <div className="pp-pop-seasons">
        <div className={`pp-pop-seasons-head${projected ? ' has-proj' : ''}`}>
          <span />
          {shown.map((s) => (
            <span className="mono" key={s.season}>&rsquo;{String(s.season).slice(2)}</span>
          ))}
          {projected && <span className="mono pp-pop-seasons-proj">proj</span>}
        </div>
        {rows.map(([label, values, fmt, proj]) => (
          <div className={`pp-pop-seasons-row${projected ? ' has-proj' : ''}`} key={label}>
            <span>{label}</span>
            {values.map((v, i) => (
              <span
                className={`mono ${v === null ? 'is-blank' : tone(v, before(values, i))}`}
                key={shown[i].season}
              >
                {v === null ? '—' : fmt(v)}
              </span>
            ))}
            {projected && (
              // The projection is coloured by the same rule as every column
              // before it, against the last season that happened. It is the
              // one comparison a drafter came to this card for -- is the
              // number being projected a step up on the year, or a step down
              // -- and it is a move, not a ranking, so nothing here claims
              // the projection has been placed against anybody.
              <span
                className={`mono pp-pop-seasons-proj ${proj === null ? 'is-blank'
                  : tone(proj, before(values, values.length))}`}
              >
                {proj === null ? '—' : fmt(proj)}
              </span>
            )}
          </div>
        ))}
      </div>
    </PopCard>
  )
}
