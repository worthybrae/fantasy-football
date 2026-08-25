import PopCard from './PopCard'
import type { CSSProperties } from 'react'
import type { ProjectedUsage, SeasonRow } from './payload'
import { CARD_HINTS } from './hints'

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
// Three columns of figures, and the line behind them for everything older.
//
// The count used to be the whole argument -- four was a compromise between
// reaching back far enough to show a role changing and leaving a percentage
// enough width to stay legible -- and the line settled it. Reach is no
// longer what the columns are for: the line carries the whole career, so
// dropping a column costs no history at all. What it buys is width, and
// width is what the columns ARE for. Three figures with room around them
// read as three numbers; four cut to fit read as one long one.
//
// Three is also the span a drafter actually argues about. Last year, the
// year before, and the year before that is the window a projection is
// defended in; anything further back is context, which is exactly what a
// line is better at than a column.
const SHOWN = 3

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
// game log's own words two cards up, not abbreviations invented here -- the
// season columns leave the label about seventy pixels, and "Carries / g" was
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
// and the ink is the number itself. Two colours over three columns say the
// one thing a share is worth reading a career for -- which way the
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


// The last season that HAS a number, which is what the card calls "now".
// Not `values[values.length - 1]`: a player who sat out this season still has
// a most recent real one, and printing his blank as his current usage would
// be the card saying he does nothing.
function latest(values: (number | null)[]): number | null {
  for (let i = values.length - 1; i >= 0; i -= 1) if (values[i] !== null) return values[i]
  return null
}

// The sparkline's own box. Its height is real pixels; its width is a
// hundred arbitrary units stretched to whatever the column turns out to be
// (`preserveAspectRatio="none"`), because the column is the card's flexible
// one -- it takes the slack that used to sit as dead air between a label and
// its numbers. Every row in the card is the same grid, so every line is
// stretched by the same factor and a shape in one row stays comparable to
// the shape in the row under it.
//
// Stretching the x axis would also stretch the ink, so nothing here is drawn
// with a filled shape: the line takes `vector-effect="non-scaling-stroke"`
// and every dot is a zero-length segment with a round cap, which SVG renders
// as a circle of the stroke's width -- a width the stretch does not touch.
const SPARK_W = 100
const SPARK_H = 18
// Room for the stroke and for the dot on the last point, which would
// otherwise be clipped in half at the top and bottom of the box.
const SPARK_PAD = 3
// How much of the box the projected step takes, whatever the career behind
// it is: a fifth. See the note in `Spark`.
const PROJ_SPAN = 0.2

// A dot that survives a stretched x axis: a segment of no length, capped
// round, painted at the stroke's own width. A `<circle>` here would come out
// an ellipse, wider or narrower per card, and the head of the line would be
// the one mark on the card whose size meant nothing.
function Dot({ x, y, size, className = '' }: {
  x: number, y: number, size: number, className?: string
}) {
  return (
    <line className={className} x1={x} y1={y} x2={x} y2={y}
          stroke="currentColor" strokeWidth={size} strokeLinecap="round"
          vectorEffect="non-scaling-stroke" />
  )
}

// A monotone cubic through the points, returned a segment at a time.
//
// Smooth, but not a spline that swings: a plain Catmull-Rom curve overshoots
// -- three seasons at 6.8, 7.9, 11.5 bulge BELOW 6.8 on the way in, and a
// usage line that dips under a number the player never fell to is drawing a
// season that did not happen. Fritsch-Carlson tangents (the limiter is the
// `sum > 9` clamp) are the standard fix: the curve stays inside the data it
// was built from, so every low point on the line is a real low point.
//
// A segment at a time because the last one is drawn differently from the
// rest -- the projection is dashed -- and it still has to be part of the
// SAME fit. Fitting the played seasons and then striking a separate line out
// to the projection put a visible corner on the card at exactly the point a
// reader is looking at. Tangents computed over the whole run and the run cut
// afterwards means the dashes leave the solid line pointing the way the
// solid line was already going.
//
// The two-point case is a straight segment on purpose. There is nothing to
// interpolate between two seasons, and a curve there would be invention with
// no data to shape it.
function fit(pts: { x: number, y: number }[]): { head: string, segs: string[] } {
  const n = pts.length
  const head = `M${pts[0].x},${pts[0].y}`
  if (n === 2) return { head, segs: [`L${pts[1].x},${pts[1].y}`] }

  // Secant slope of each segment, and a tangent at each point: the average
  // of the two secants meeting there, or FLAT where they disagree in sign --
  // a peak or a trough is a turn, and a tangent through it would carry the
  // curve past the season it turns on.
  const d = pts.slice(0, -1).map((p, i) =>
    (pts[i + 1].y - p.y) / (pts[i + 1].x - p.x))
  const m = pts.map((_, i) => {
    if (i === 0) return d[0]
    if (i === n - 1) return d[n - 2]
    return d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2
  })
  d.forEach((slope, i) => {
    if (slope === 0) { m[i] = 0; m[i + 1] = 0; return }
    const a = m[i] / slope
    const b = m[i + 1] / slope
    const sum = a * a + b * b
    if (sum > 9) {
      const t = 3 / Math.sqrt(sum)
      m[i] = t * a * slope
      m[i + 1] = t * b * slope
    }
  })

  return {
    head,
    segs: pts.slice(1).map((p, i) => {
      const q = pts[i]
      const third = (p.x - q.x) / 3
      return `C${q.x + third},${q.y + m[i] * third} `
           + `${p.x - third},${p.y - m[i + 1] * third} ${p.x},${p.y}`
    }),
  }
}

// One row's WHOLE career as a line, scaled to that row's own range rather
// than to anything shared. A sparkline is a shape, not a measurement: the
// question it answers is which way the role went, and a snap share and a
// yards-per-game figure have no common axis to be drawn against anyway. The
// columns beside it are the measurement.
//
// The career, not the three seasons the columns print. The columns are the
// detail a drafter reads; the line is the arc they read it inside, and an
// arc cut to three years is the one shape that cannot show a player who has
// been declining since 2019. This is the whole reason the line and the
// columns are two different things rather than one drawn twice.
//
// Missing seasons break the line rather than interpolating across them. A
// year a player missed is not a year he did something in between his other
// two, and a line drawn straight through it would invent the season.
function Spark({ values, proj, tone: cls }: {
  values: (number | null)[]
  proj: number | null
  tone: string
}) {
  // One season is not a trend. A career with a single number in it drew a
  // dot and -- where there was a projection -- a dashed step out to it,
  // which is a two-point line whose entire shape is the forecast: it slopes
  // because ESPN said so, not because the player has done anything yet. Two
  // played seasons is the least a direction can be read from, so below that
  // the card draws no line at all and the columns speak alone.
  const real = values.filter((v): v is number => v !== null)
  if (real.length < 2) return <span />

  // The projection is inside the scale, not on top of it: left out, a
  // projected number above every season the player has played would be drawn
  // off the top of the box.
  const scale = proj === null ? real : [...real, proj]
  const min = Math.min(...scale)
  const max = Math.max(...scale)
  const span = max - min
  const inner = SPARK_H - SPARK_PAD * 2

  // The projection gets a fixed share of the width rather than one more
  // even step. Even steps meant the forecast on a nine-season quarterback
  // was an eighth the length of the one on a two-season rookie -- and it is
  // the same forecast, drawn on the same card, being read for the same
  // reason. A fifth of the box, always.
  const floor = SPARK_PAD
  const ceil = SPARK_W - SPARK_PAD
  const stop = proj === null ? ceil : floor + (ceil - floor) * (1 - PROJ_SPAN)
  // Two real seasons at minimum, so there are always two slots to spread
  // between: the guard at the top is what makes this divisor safe.
  const x = (i: number) => floor + (i * (stop - floor)) / (values.length - 1)
  // A flat row sits on the middle line, not on the floor: every value equal
  // means the range is zero, and dividing by it would put the line nowhere.
  const y = (v: number) => (span
    ? SPARK_H - SPARK_PAD - ((v - min) / span) * inner
    : SPARK_H / 2)

  // Runs of consecutive seasons that both exist. A season the player missed
  // breaks the line rather than being interpolated across: a year he did not
  // play is not a year he did something in between his other two, and a line
  // drawn straight through it would invent the season.
  const runs: { x: number, y: number }[][] = []
  values.forEach((v, i) => {
    if (v === null) { runs.push([]); return }
    if (runs.length === 0) runs.push([])
    runs[runs.length - 1].push({ x: x(i), y: y(v) })
  })
  const drawn = runs.filter((r) => r.length > 0)

  // The projection joins the last run rather than being a shape of its own,
  // so the fit that smooths the played seasons smooths the step out to it
  // too. Where the last thing on the card is a season the player missed,
  // there is nothing to join and the projection stands alone as a dot.
  const tail = drawn.length > 0 ? drawn[drawn.length - 1] : null
  const projPoint = proj === null ? null : { x: ceil, y: y(proj) }
  if (projPoint && tail) tail.push(projPoint)

  const lastReal = tail && projPoint ? tail[tail.length - 2] : null

  return (
    <svg className={`pp-pop-usage-spark ${cls}`} width="100%" height={SPARK_H}
         viewBox={`0 0 ${SPARK_W} ${SPARK_H}`} preserveAspectRatio="none"
         aria-hidden="true">
      {drawn.map((run, i) => {
        // A run of one is a lone season with gaps either side: a path of one
        // point draws nothing at all, so it gets a dot.
        if (run.length < 2) return <Dot key={run[0].x} x={run[0].x} y={run[0].y} size={3} />
        const { head, segs } = fit(run)
        const isTail = projPoint !== null && i === drawn.length - 1
        // Everything but the last segment of the last run is a season that
        // happened. The one segment that is only expected is drawn from the
        // same fit and then dashed, so it leaves the solid line pointing the
        // way the solid line was already going.
        const solid = isTail ? segs.slice(0, -1) : segs
        return (
          <g key={run[0].x}>
            {solid.length > 0 && (
              <path d={head + solid.join('')} fill="none" stroke="currentColor"
                    strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                    vectorEffect="non-scaling-stroke" />
            )}
            {isTail && lastReal && (
              <path className="pp-pop-usage-spark-proj"
                    d={`M${lastReal.x},${lastReal.y}${segs[segs.length - 1]}`}
                    fill="none" stroke="currentColor" strokeWidth="1.5"
                    strokeLinecap="round" strokeDasharray="2.5 2.5"
                    vectorEffect="non-scaling-stroke" />
            )}
          </g>
        )
      })}
      {projPoint && (
        <Dot className="pp-pop-usage-spark-proj" x={projPoint.x} y={projPoint.y} size={3} />
      )}
      {/* The head of the line: the last season that happened, which is the
          number in the last column. The figures are printed beside it, so
          this is not what tells a reader the value -- it is the seam, the
          mark that says everything right of here is a guess. */}
      {tail && tail.length > 0 && (
        <Dot className="pp-pop-usage-spark-head"
             x={(lastReal ?? tail[tail.length - 1]).x}
             y={(lastReal ?? tail[tail.length - 1]).y} size={4} />
      )}
    </svg>
  )
}

export default function UsageLine({ seasons, position, projected }: {
  seasons: SeasonRow[]
  position: string
  projected?: ProjectedUsage | null
}) {
  // Oldest on the left, so the line reads the way the panels above it do and
  // the way time does. The payload is newest first.
  //
  // Two spans, deliberately: `career` is every season the payload carries and
  // is what the line is drawn from, `shown` is the three the columns print.
  // The columns are held at three because a percentage needs its width to stay
  // legible and nine of them would not have any -- but the line costs no
  // width per season, so there is no reason for IT to forget 2019.
  const career = seasons.slice().reverse()
  const shown = seasons.slice(0, SHOWN).reverse()
  if (shown.length === 0) return null

  // Numbers stay numbers until the cell is drawn, because both the colour
  // and the shape of the line are arithmetic between them -- formatting
  // first would leave the row comparing "54%" against "57%" as strings.
  const built = [...SHARES, ...(ROWS[position] ?? [])].map((row) => ({
    row,
    share: SHARES.includes(row),
    values: shown.map(row.of),
    line: career.map(row.of),
    proj: row.proj && projected ? row.proj(projected) : null,
  }))

  // A row nothing can answer is dropped rather than dashed across: a rookie
  // has no seasons to read, and a blank line beside a blank number is a row
  // that costs a reader a line to learn nothing. A row with SOME seasons
  // missing keeps its gaps -- there the gap is the fact.
  //
  // A row of ZEROS goes the same way, and that is what drops "Target %" off
  // a quarterback: his target share is not missing, it is genuinely 0.000
  // every season, and a flat line on the floor is a career of facts that never
  // happened. The rule is the same one the gaps follow -- print a row only
  // if some season, or the projection, has something in it.
  const has = (v: number | null) => v !== null && v !== 0
  const rows = built.filter(({ values, proj }) => values.some(has) || has(proj))
  if (rows.length === 0) return null

  // The seam between the two shares and the per-game rates under them: one
  // half is how much of the offence he was, the other is what he did with
  // it. The hairline is drawn on the first row of the second half, and only
  // when there is a first half above it to be separated from.
  const firstRate = rows.findIndex((r) => !r.share)
  const seam = firstRate > 0 ? firstRate : -1

  const head = (s: number) => `’${String(s).slice(2)}`

  // The year the line starts from, stated only when the line reaches back
  // further than the columns do. When the career IS the three columns the
  // line spans exactly what they span, and printing the year twice would be
  // the card labelling one stretch of time in two places.
  const from = career.length > shown.length ? head(career[0].season) : null

  return (
    <PopCard title="Usage" note="by season" hint={CARD_HINTS.usage} className="is-widest">
      {/* The grid is sized to the seasons this player actually has, not to
          SHOWN. With three fixed tracks and two seasons in them, the third
          track still took its width -- so a two-season card ended a column
          short of its own right edge, and the projection sat adrift of the
          card wall it should be flush with. */}
      <div className={`pp-pop-usage${projected ? ' has-proj' : ''}`}
           style={{ '--season-cols': shown.length } as CSSProperties}>
        <div className="pp-pop-usage-head">
          <span />
          <span className="mono pp-pop-usage-from">{from}</span>
          {shown.map((s) => (
            <span className="mono" key={s.season}>{head(s.season)}</span>
          ))}
          {projected && <span className="mono pp-pop-usage-proj">proj</span>}
        </div>
        {rows.map(({ row, values, line, proj }, i) => {
          const now = latest(values)
          // The line takes ONE direction -- the last season against the one
          // before it -- while the cells each take their own. The shape is
          // already carrying the longer arc, so colouring it by anything
          // else would be a second claim laid over the first. Read off the
          // career rather than the columns for the player whose last three
          // seasons are all blank: the line still has something to say
          // about him, and it should say it in the right colour.
          const move = tone(latest(line), before(line, line.length), row.invert)
          return (
            <div className={`pp-pop-usage-row${i === seam ? ' is-seam' : ''}`}
                 key={row.label}>
              <span className="pp-pop-usage-label">{row.label}</span>
              <Spark values={line} proj={proj} tone={move} />
              {values.map((v, j) => (
                // The last column is the season everything else on this card
                // is read against -- the projection is a step off it, the
                // line ends on it -- so it is set in the card's own ink and
                // a weight up, and the seasons behind it are history.
                <span
                  className={`mono${j === values.length - 1 ? ' is-now' : ''} ${
                    v === null ? 'is-blank' : tone(v, before(values, j), row.invert)}`}
                  key={shown[j].season}
                >
                  {v === null ? '—' : row.fmt(v)}
                </span>
              ))}
              {projected && (
                // The projection is coloured by the same rule as every
                // column before it, against the last season that happened.
                // It is the one comparison a drafter came to this card for
                // -- is the number being projected a step up on the year, or
                // a step down -- and it is a move, not a ranking, so nothing
                // here claims the projection has been placed against
                // anybody.
                <span
                  className={`mono pp-pop-usage-proj ${proj === null ? 'is-blank'
                    : tone(proj, now, row.invert)}`}
                >
                  {proj === null ? '—' : row.fmt(proj)}
                </span>
              )}
            </div>
          )
        })}
      </div>
    </PopCard>
  )
}
