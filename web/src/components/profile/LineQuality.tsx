import type { LineQualityData, LineStarter } from './payload'
import PopCard from './PopCard'
import { ordinal } from './payload'
import { CARD_HINTS } from './hints'

// The three parts of the rating, as meters that run the width of the card.
//
// They were vertical bars for one revision, borrowed from the season panels
// above -- and borrowed wrong. Those panels draw a SERIES: five seasons along
// an axis, where height is the only free variable and the eye compares
// neighbours. These three are not a series, they are three separate shares
// with nothing running between them, and drawn as columns they read as a
// chart of something that was never measured over time. Horizontally they
// read as what they are: three gauges, each full at 100%, stacked so the
// short one is obvious.
//
// It also puts the card's own width to work. A three-column chart left two
// thirds of the row empty and pushed the line itself down; a meter uses the
// space it is given and costs 14px a part.
//
// EXPERIENCE is the fourth, and it is drawn differently from the three above
// it on purpose. Those are shares: each is already a fraction of one, so a
// full meter means "all of it". Experience is mean SEASONS, which has no full
// mark -- eight years is a lot for a line and nothing at all for a
// quarterback -- so the meter fills to where the line PLACES among the 32
// (`experience_pct`, the same percentile the composite is built from) while
// the number beside it stays in the unit anyone thinks in.
//
// It was off the card entirely for a revision, because the join behind it
// landed on the wrong man for any rookie who shares a name with a player
// from the nineties -- Detroit's 2026 right tackle read as 34 seasons and
// dragged that line's mean to nine. `scoring/oline.py` now returns NaN past
// a plausible career instead, which is what makes this row trustworthy
// enough to draw: Detroit reads 2.8 years now, and dropped from 21st to 29th
// once it stopped being flattered.
const PARTS: { key: 'continuity' | 'availability' | 'returning'
               label: string; hint: string }[] = [
  // Reading order is the order they weigh in `scoring/oline.py`'s WEIGHTS.
  { key: 'continuity', label: 'Same five',
    hint: 'Share of last season\u2019s O-line snaps taken by its five most-used linemen' },
  { key: 'availability', label: 'Available',
    hint: 'Share of their teams\u2019 games these five have been active for, career' },
  { key: 'returning', label: 'Returning',
    hint: 'Share of this line that started on it last season' },
]

// The popup's own five-step ramp, cut for these three shares specifically.
// Same class names the season panels use, so a green here and a green three
// inches up are the same claim about the same scale.
//
// The cuts are high because the DATA is high. An even 0.4/0.55/0.7/0.85 ramp
// was tried first and painted almost every line the same colour: continuity
// runs about 0.7-0.9 across the league, availability about 0.65-0.85, and a
// ramp whose bottom two steps nothing ever reaches is a ramp with three
// steps. These cuts put the middle of the real spread in the middle of the
// colours, which is the only way the fills say anything by being different.
function shareTone(share: number): string {
  if (share >= 0.9) return 'is-good'
  if (share >= 0.8) return 'is-strong'
  if (share >= 0.7) return 'is-mid'
  if (share >= 0.6) return 'is-fringe'
  return 'is-bad'
}

// One drawn row: a label, how far the bar goes (0-1), what the number beside
// it says, and which step of the ramp it takes. The three shares fill to
// themselves and Experience fills to its percentile, which is exactly why the
// fill and the value are separate fields rather than one number formatted
// twice.
interface Meter {
  key: string; label: string; hint: string
  fill: number; value: string; tone: string
}

function pct(share: number): string {
  return `${Math.round(share * 100)}%`
}

// The same five steps, cut on where a place falls in its own field rather
// than on a share. A rank is only ever as good as the size of what it is a
// rank of, so the cuts are fractions of the field: top fifth, top two
// fifths, and so on down.
function placeTone(rank: number, of: number): string {
  if (of <= 0) return ''
  const p = rank / of
  if (p <= 0.2) return 'is-good'
  if (p <= 0.4) return 'is-strong'
  if (p <= 0.6) return 'is-mid'
  if (p <= 0.8) return 'is-fringe'
  return 'is-bad'
}

// The line the player's own production runs through: where it ranks, what the
// rank is made of, and who is on it.
//
// The composite 0-100 stays off, as it always has: it is opaque, and the
// place -- 13th of 32 -- is the same claim in a unit that means something.
//
// The rank itself is plain rather than toned. Every other colour in this
// popup is carrying a season of the player's own; his team's line is the
// context those seasons happened in, and a red 24th beside them would
// compete with them. The meters under it are coloured because they are read
// against a full mark rather than against his career.
export default function LineQuality({ oline }: { oline: LineQualityData }) {
  const starters = (oline.starters ?? []).filter((s) => s.name)
  const parts: Meter[] = PARTS.flatMap(({ key, label, hint }) => {
    const share = oline[key]
    // A part the database could not compute is left out rather than drawn
    // empty, which would read as "none of it".
    if (share === null || share === undefined) return []
    return [{ key, label, hint, fill: share, value: pct(share),
              tone: shareTone(share) }]
  })
  const years = oline.experience
  const yearsPct = oline.experience_pct
  if (years !== null && years !== undefined
      && yearsPct !== null && yearsPct !== undefined) {
    parts.push({
      key: 'experience',
      label: 'Experience',
      hint: 'Mean seasons in the league across the five \u2014 the bar is where '
        + 'that places among the 32 lines',
      fill: yearsPct / 100,
      // "4yr", not "4.2 yr": the value column is sized for "100%", and a
      // decimal plus a space plus a unit wrapped onto a second line, which
      // pushed the meter out of line with the three above it. The tenth was
      // never worth reading anyway -- this is a mean over five men, and
      // nobody drafts differently against 4.2 years than against 4.
      value: `${Math.round(years)}yr`,
      // A percentile, so an even ramp: top fifth, top two fifths, and so on.
      // The share cuts above are high because those numbers are; a place is
      // spread evenly by construction.
      tone: yearsPct >= 80 ? 'is-good' : yearsPct >= 60 ? 'is-strong'
        : yearsPct >= 40 ? 'is-mid' : yearsPct >= 20 ? 'is-fringe' : 'is-bad',
    })
  }

  return (
    <PopCard title="O-line" note={oline.team} hint={CARD_HINTS.oline}>
      <div className="pp-pop-lead">
        <span className="mono pp-pop-lead-value">{ordinal(oline.rank)}</span>
        <span className="mono pp-pop-lead-of">of {oline.teams}</span>
      </div>
      {parts.length > 0 && (
        <div className="pp-pop-meters">
          {parts.map((p) => (
            <div className="pp-pop-meter" key={p.key} title={p.hint}>
              <span className="pp-pop-meter-label">{p.label}</span>
              <span className="pp-pop-meter-track">
                <span className={`pp-pop-meter-fill ${p.tone}`}
                      style={{ width: `${Math.max(2, Math.min(1, p.fill) * 100)}%` }} />
              </span>
              <span className="mono pp-pop-meter-value">{p.value}</span>
            </div>
          ))}
        </div>
      )}
      {starters.length > 0 && <Starters starters={starters} />}
    </PopCard>
  )
}

// Where he places among the league's other starters at the same slot -- 14th
// of 29 left tackles -- on availability, which is the figure the Available
// meter above averages over the five of them.
//
// A place, not the share it comes from. "68%" is a true number that means
// nothing on its own: a reader cannot tell whether 68% is an ordinary tackle
// or the third-worst in football without knowing the spread, and nothing on
// this card was telling him. The share and the games behind it stay on the
// row's tooltip for anyone who wants them.
//
// The denominators differ between rows (29 tackles, 27 guards) because a
// starter the database cannot price is left out of both the placing and the
// field, rather than being ranked last on missing data.
function Starters({ starters }: { starters: LineStarter[] }) {
  return (
    <ul className="pp-pop-line">
      {starters.map((s) => {
        const rank = s.avail_rank ?? null
        const of = s.avail_rank_of ?? null
        const detail = [
          s.availability === null ? null : `${pct(s.availability)} available`,
          s.games === null || s.games_possible === null
            ? null : `${s.games} of ${s.games_possible} games`,
        ].filter(Boolean).join(', ')
        return (
          <li className="pp-pop-line-row" key={`${s.position}-${s.name}`}
              title={detail ? `${s.name}: ${detail}` : undefined}>
            <span className="mono pp-pop-line-pos">{s.position ?? '\u2014'}</span>
            <span className="pp-pop-line-name">{s.name}</span>
            <span className={`mono pp-pop-line-avail${
              rank === null || of === null ? '' : ` ${placeTone(rank, of)}`}`}>
              {rank === null || of === null ? '\u2014' : `${rank}/${of}`}
            </span>
          </li>
        )
      })}
    </ul>
  )
}
