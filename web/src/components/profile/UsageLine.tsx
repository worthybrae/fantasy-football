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

// A row is a number pulled out of a season, a number pulled out of the
// projection, and the way both are printed. Functions rather than column
// names, because half these rows are ratios -- catch rate is receptions over
// targets, not a column anybody stores -- and a row that had to name a
// stored column would have left the card at four rates.
type Getter = (s: SeasonRow) => number | null
type ProjGetter = (p: ProjectedUsage) => number | null

interface Row {
  label: string
  of: Getter
  // Null where the projection cannot answer this row: ESPN projects no
  // completions, so there is no projected completion rate, and it projects
  // neither a team total nor snaps, so neither share has one either. The
  // cell prints an em dash, which is the card declining rather than
  // guessing.
  proj: ProjGetter | null
  fmt: (v: number) => string
  // Up is worse for exactly one row on this card. Interceptions rising is a
  // quarterback getting worse, and without this the row would paint that
  // green along with everything else.
  invert?: boolean
}

function total(s: SeasonRow, keys: (keyof SeasonRow)[]): number {
  return keys.reduce((sum, k) => sum + ((s[k] as number | null) ?? 0), 0)
}

// Per game, which is how every row on this card is stated: 325 carries and
// 19.1 a game are the same fact, but only one of them survives being set
// beside a season the player missed five weeks of.
function pg(...keys: (keyof SeasonRow)[]): Getter {
  return (s) => (s.games ? total(s, keys) / s.games : null)
}

// A rate that divides one column by another rather than by games. A zero
// denominator is null, not zero: a back with no targets has no catch rate,
// and 0% would say he dropped everything.
function ratio(num: (keyof SeasonRow)[], den: (keyof SeasonRow)[]): Getter {
  return (s) => {
    const bottom = total(s, den)
    return bottom ? total(s, num) / bottom : null
  }
}

// The projection's own figures are already per game -- see
// `_espn_projected_usage`, which divides by projected games before it
// answers -- so these only add and divide. A row where EVERY key is null is
// null: a projection that says nothing must not print a zero.
function projTotal(p: ProjectedUsage, keys: (keyof ProjectedUsage)[]): number | null {
  const values = keys.map((k) => p[k] as number | null)
  return values.every((v) => v === null || v === undefined)
    ? null : values.reduce((sum: number, v) => sum + (v ?? 0), 0)
}

function projPg(...keys: (keyof ProjectedUsage)[]): ProjGetter {
  return (p) => projTotal(p, keys)
}

function projRatio(num: (keyof ProjectedUsage)[],
                   den: (keyof ProjectedUsage)[]): ProjGetter {
  return (p) => {
    const top = projTotal(p, num)
    const bottom = projTotal(p, den)
    return top === null || !bottom ? null : top / bottom
  }
}

function rate(value: number): string {
  return value.toFixed(1)
}

function pct(share: number): string {
  return `${Math.round(share * 100)}%`
}

// Games are counted, not measured: seventeen is "17", and only the
// projection's fractional 16.4 spends a decimal place.
function count(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
}

// Both shares, offered to every position and dropped by the filter below for
// the ones that cannot answer them -- a quarterback has no target share, and
// nothing outside the skill positions has either.
const SHARES: Row[] = [
  { label: 'Snap %', of: (s) => s.snap_share ?? null, proj: null, fmt: pct },
  { label: 'Target %', of: (s) => s.target_share ?? null, proj: null, fmt: pct },
]

// Availability, last because it is the row that qualifies every row above
// it: a rate is per game, so a career of 11-game seasons reads identically
// to a career of 17-game ones until this row says otherwise.
const GAMES: Row = {
  label: 'Games',
  of: (s) => s.games || null,
  proj: (p) => p.games,
  fmt: count,
}

// Per-position rows, in reading order. `car` / `tgt` / `rec` / `yds` are the
// game log's own words two cards up, not abbreviations invented here -- four
// season columns leave the label about fifty pixels, and "Carries / g" was
// being ellipsised into "Carri...", which is worse than a short word a
// reader has already met.
//
// Yards are the sum of both halves: a back who gains 73 on the ground and 31
// through the air had a 105-yard game, and splitting that across two rows
// would spend the card on one stat. The quarterback's rushing row is the one
// exception, and it is rushing ONLY -- his receiving yards are noise.
//
// `tds` is rushing plus receiving and never a throw (it is built that way in
// similarity.py), which is why the quarterback reads `pass_tds` instead and
// the skill positions read `tds`.
//
// THERE IS NO `K` OR `DST` ENTRY AND THAT IS DELIBERATE. A kicker's season
// row carries no skill columns to read, so the lookup falls through, no rate
// row is built, and -- with no snap or target share either -- the card
// renders nothing, which is the whole of what the payload can say about a
// kicker's usage.
const ROWS: Record<string, Row[]> = {
  QB: [
    { label: 'Att / g', of: pg('attempts'), proj: projPg('attempts'), fmt: rate },
    { label: 'Comp %', of: ratio(['completions'], ['attempts']), proj: null, fmt: pct },
    { label: 'Pass yds / g', of: pg('pass_yards'), proj: projPg('pass_yards'), fmt: rate },
    { label: 'Pass TD / g', of: pg('pass_tds'), proj: projPg('pass_tds'), fmt: rate },
    { label: 'INT / g', of: pg('interceptions'), proj: projPg('interceptions'),
      fmt: rate, invert: true },
    { label: 'Rush yds / g', of: pg('rush_yards'), proj: projPg('rush_yards'), fmt: rate },
    // `tds` is rushing plus receiving and never a throw, so for a
    // quarterback it is his legs and nothing else -- which is the half of
    // his scoring the four rows above cannot show. ESPN projects rushing
    // touchdowns (`proj_rush_tds`), so this row is projected like the rest
    // rather than being history with a blank column: Jalen Hurts scored 8 on
    // the ground in 2025 and is projected 0.5 a game, and a rushing
    // quarterback's floor is mostly that number.
    { label: 'Rush TD / g', of: pg('tds'), proj: projPg('tds'), fmt: rate },
    GAMES,
  ],
  RB: [
    { label: 'Car / g', of: pg('carries'), proj: projPg('carries'), fmt: rate },
    { label: 'Rec / g', of: pg('receptions'), proj: projPg('receptions'), fmt: rate },
    { label: 'Yds / g', of: pg('rush_yards', 'rec_yards'), proj: projPg('yards'), fmt: rate },
    // Yards per opportunity: what one touch or one look was worth. The rate
    // that separates a back given 300 carries from a back who earned them.
    { label: 'Yds / opp', of: ratio(['rush_yards', 'rec_yards'], ['carries', 'targets']),
      proj: projRatio(['yards'], ['carries', 'targets']), fmt: rate },
    { label: 'TD / g', of: pg('tds'), proj: projPg('tds'), fmt: rate },
    GAMES,
  ],
  WR: [
    { label: 'Tgt / g', of: pg('targets'), proj: projPg('targets'), fmt: rate },
    { label: 'Rec / g', of: pg('receptions'), proj: projPg('receptions'), fmt: rate },
    { label: 'Catch %', of: ratio(['receptions'], ['targets']),
      proj: projRatio(['receptions'], ['targets']), fmt: pct },
    { label: 'Yds / g', of: pg('rec_yards', 'rush_yards'), proj: projPg('yards'), fmt: rate },
    { label: 'Yds / opp', of: ratio(['rec_yards', 'rush_yards'], ['targets', 'carries']),
      proj: projRatio(['yards'], ['targets', 'carries']), fmt: rate },
    { label: 'TD / g', of: pg('tds'), proj: projPg('tds'), fmt: rate },
    GAMES,
  ],
}
ROWS.TE = ROWS.WR

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

function tone(now: number | null, prev: number | null, invert = false): string {
  if (now === null || prev === null || prev === 0) return ''
  const move = ((now - prev) / Math.abs(prev)) * (invert ? -1 : 1)
  if (move >= MOVE) return 'delta-tone is-up'
  if (move <= -MOVE) return 'delta-tone is-down'
  return ''
}

// The most recent season before this one that HAS a number. A missing season
// is compared past, not treated as a zero: a player who missed a year and
// came back at his old workload did not go up, and dividing by that zero
// would have said he went up infinitely.
function before(values: (number | null)[], i: number): number | null {
  for (let j = i - 1; j >= 0; j -= 1) if (values[j] !== null) return values[j]
  return null
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

  // Numbers stay numbers until the cell is drawn, because the colour is
  // arithmetic between two of them -- formatting first would leave the row
  // comparing "54%" against "57%" as strings.
  const built = [...SHARES, ...(ROWS[position] ?? [])].map((row) => ({
    row,
    values: shown.map(row.of),
    proj: row.proj && projected ? row.proj(projected) : null,
  }))

  // A row nothing can answer is dropped rather than dashed across: a rookie
  // has no seasons to read, and four em-dashes in a line is a row that costs
  // a reader a line to learn nothing. A row with SOME seasons missing keeps
  // its dashes -- there the gap is the fact.
  //
  // A row of ZEROS goes the same way, and that is what drops "Target %" off
  // a quarterback: his target share is not missing, it is genuinely 0.000
  // every season, and four 0% cells are four facts that never happened. The
  // rule is the same one the dashes follow -- print a row only if some
  // season, or the projection, has something in it.
  const has = (v: number | null) => v !== null && v !== 0
  const rows = built.filter(({ values, proj }) => values.some(has) || has(proj))
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
        {rows.map(({ row, values, proj }) => (
          <div className={`pp-pop-seasons-row${projected ? ' has-proj' : ''}`} key={row.label}>
            <span>{row.label}</span>
            {values.map((v, i) => (
              <span
                className={`mono ${v === null ? 'is-blank'
                  : tone(v, before(values, i), row.invert)}`}
                key={shown[i].season}
              >
                {v === null ? '—' : row.fmt(v)}
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
                  : tone(proj, before(values, values.length), row.invert)}`}
              >
                {proj === null ? '—' : row.fmt(proj)}
              </span>
            )}
          </div>
        ))}
      </div>
    </PopCard>
  )
}
