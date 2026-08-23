import PopCard from './PopCard'
import type { ProjectedUsage, SeasonPercentiles, SeasonRow } from './payload'

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
// their width back, which is what keeps a percentage and its chip legible.
// Older usage is a different team; seeing WHEN a role changed is the reason
// to look at all, and four seasons still shows a change.
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
// The third element is the percentile that colours the row -- the pool the
// number is good or bad against. A row with none (a quarterback's passing
// rates, and snap share) prints its number uncoloured rather than borrowing
// a neighbour's pool.
// The fourth element is which projected rate answers this row -- the card's
// last column is the season being drafted, drawn the way the panels above
// draw theirs. Null where ESPN projects nothing that fits the row.
type Rate = [string, (keyof SeasonRow)[], keyof SeasonPercentiles | null,
             keyof ProjectedUsage | null]

const RATES: Record<string, Rate[]> = {
  QB: [['Att / g', ['attempts'], null, 'attempts'],
    ['Pass yds / g', ['pass_yards'], null, 'pass_yards']],
  RB: [['Car / g', ['carries'], 'carries_pg', 'carries'],
    ['Yds / g', ['rush_yards', 'rec_yards'], 'yards_pg', 'yards']],
  WR: [['Tgt / g', ['targets'], 'targets_pg', 'targets'],
    ['Yds / g', ['rec_yards', 'rush_yards'], 'yards_pg', 'yards']],
  TE: [['Tgt / g', ['targets'], 'targets_pg', 'targets'],
    ['Yds / g', ['rec_yards', 'rush_yards'], 'yards_pg', 'yards']],
}

// The five steps, cut on the percentile. Even fifths: unlike a positional
// finish there is no starter count to cut against here -- the question is
// simply where in his position's spread this number falls.
function chipTone(pctl: number | null | undefined): string {
  if (pctl === null || pctl === undefined) return 'is-unrated'
  if (pctl >= 0.8) return 'is-elite'
  if (pctl >= 0.6) return 'is-strong'
  if (pctl >= 0.4) return 'is-starter'
  if (pctl >= 0.2) return 'is-fringe'
  return 'is-out'
}

function pct(share: number | null | undefined): string | null {
  return share === null || share === undefined ? null : `${Math.round(share * 100)}%`
}

function perGame(season: SeasonRow, keys: (keyof SeasonRow)[]): string | null {
  if (!season.games) return null
  const total = keys.reduce((sum, k) => sum + ((season[k] as number | null) ?? 0), 0)
  return (total / season.games).toFixed(1)
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

  type Cell = { text: string | null; tone: string }
  const rate = (v: number | null | undefined) =>
    v === null || v === undefined ? null : v.toFixed(1)
  const cells = (values: (string | null)[],
                 key: keyof SeasonPercentiles | null): Cell[] =>
    values.map((text, i) => ({
      text,
      tone: chipTone(key === null ? null : shown[i].pcts?.[key]),
    }))

  const candidates: [string, Cell[], string | null][] = [
    // The two shares have no projected column and the em dash says so: the
    // projection table carries no team total worth dividing by and does not
    // project snaps at all. A blank cell here is the card declining to guess,
    // in the same place it would otherwise be guessing hardest.
    ['Snap %', cells(shown.map((s) => pct(s.snap_share)), 'snap_share'), null],
    ['Target %', cells(shown.map((s) => pct(s.target_share)), 'target_share'), null],
    ...(RATES[position] ?? []).map(([label, keys, key, projKey]):
    [string, Cell[], string | null] => [
      label,
      cells(shown.map((s) => perGame(s, keys)), key),
      projKey === null || !projected ? null
        : rate(projected[projKey] as number | null),
    ]),
  ]
  // A row nothing can answer is dropped rather than dashed across: a
  // quarterback has no target share, a rookie has no seasons to read, and
  // three em-dashes in a line is a row that costs a reader a line to learn
  // nothing. A row with SOME seasons missing keeps its dashes -- there the
  // gap is the fact.
  const rows = candidates.filter(([, cs, proj]) =>
    cs.some((c) => c.text !== null) || proj !== null)
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
        {rows.map(([label, values, proj]) => (
          <div className={`pp-pop-seasons-row${projected ? ' has-proj' : ''}`} key={label}>
            <span>{label}</span>
            {values.map((c, i) => (
              <span className="mono" key={shown[i].season}>
                <span className={`pp-pop-chip ${c.text === null ? 'is-unrated' : c.tone}`}>
                  {c.text ?? '—'}
                </span>
              </span>
            ))}
            {projected && (
              // Behind a rule and uncoloured, exactly as the Per game panel
              // draws its own projected column: it has not happened, so it is
              // not ranked against anyone.
              <span className="mono pp-pop-seasons-proj">{proj ?? '—'}</span>
            )}
          </div>
        ))}
      </div>
    </PopCard>
  )
}
