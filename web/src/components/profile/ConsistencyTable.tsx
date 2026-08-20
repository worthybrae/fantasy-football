import type { SeasonRow } from './payload'
import { ordinal, posHueClass } from './payload'

// The track runs 0.2 -> 1.2 CV, left is steadier. Fixed rather than fitted
// to the player: the position median has to land in the same place on every
// card or the eye cannot carry a sense of "normal" from one player to the
// next. Values outside it clamp to the ends -- the numeric rank beside the
// track is what carries a genuine outlier.
const CV_MIN = 0.2
const CV_MAX = 1.2

function trackPct(cv: number): number {
  return Math.max(0, Math.min(100, ((cv - CV_MIN) / (CV_MAX - CV_MIN)) * 100))
}

// Steadier third / middle / shakiest third of the position, on the rank the
// payload already computed against every qualifying season at the position.
function cvTone(rank: number | null, of: number | null): 'good' | 'mid' | 'bad' {
  if (rank === null || of === null || of <= 0) return 'mid'
  if (rank <= of / 3) return 'good'
  if (rank > (of * 2) / 3) return 'bad'
  return 'mid'
}

// Season by season, with the two things a table of averages cannot say:
// how wide the weeks actually were (the whisker), and how that width
// compares to everyone else at the position once his scoring is divided out
// (the track).
//
// VOLATILITY IS RANKED ON THE COEFFICIENT OF VARIATION -- spread divided by
// average -- AND NEVER ON THE RAW STANDARD DEVIATION. This is not a
// cosmetic choice and it must not be "simplified" back: sigma correlates
// with points per game at r = 0.85 across the 2025 running backs, so
// ranking on it re-ranks the position by scoring and calls the result
// consistency. On raw sigma Jahmyr Gibbs' 2025 comes out 97th of 97, the
// most volatile back in the league, which is the exact opposite of the
// truth; on CV the same season is 28th of 95 against a position median of
// 0.767, i.e. the steadier third. Both numbers are in the payload
// (`ppg_std` and `cv`), one field name apart, which is precisely why this
// comment is here. The +- figure beside the track is still the raw spread,
// because that is the thing a manager feels week to week -- it is just not
// what the ranking is made of.
export default function ConsistencyTable({ seasons, position }: {
  seasons: SeasonRow[]
  position: string
}) {
  const rows = [...seasons].sort((a, b) => b.season - a.season)
  if (rows.length === 0) return null

  // One scale across every season of his career, so the bars are comparable
  // down the column -- a per-row scale would draw a 9-ppg rookie year the
  // same length as a 21-ppg peak.
  const spreadMax = Math.max(
    ...rows.map((s) => s.ppg + (s.ppg_std ?? 0)),
    1,
  )
  const pct = (v: number) => `${Math.max(0, Math.min(100, (v / spreadMax) * 100))}%`

  const ranked = rows.filter((s) => s.cv_rank !== null && s.cv_rank_n !== null)
  const steady = ranked.filter((s) => cvTone(s.cv_rank, s.cv_rank_n) === 'good').length

  return (
    <div className={`pp-seasons ${posHueClass(position)}`}>
      <div className="pp-seasons-row pp-seasons-head">
        <div className="pp-cap">Year</div>
        <div className="pp-cap pp-r">GP</div>
        <div className="pp-cap pp-r">PPG</div>
        <div className="pp-cap pp-r">Fin</div>
        <div className="pp-cap">Average, and the weeks around it</div>
        <div className="pp-cap pp-r">Steadiness (spread ÷ avg)</div>
      </div>

      {rows.map((s) => {
        const std = s.ppg_std ?? 0
        const lo = Math.max(0, s.ppg - std)
        const hi = s.ppg + std
        const tone = cvTone(s.cv_rank, s.cv_rank_n)
        const finish = s.pos_rank_ppg !== null
          ? `${position}${s.pos_rank_ppg}`
          : `${position}${s.pos_finish}`
        return (
          <div className="pp-seasons-row" key={s.season}>
            <div className="pp-season-year">
              <span className="mono">{s.season}</span>
              {s.age !== null && <span className="pp-season-age">age {s.age}</span>}
            </div>
            <div className="mono pp-r pp-dim">{s.games}</div>
            <div className="mono pp-r pp-season-ppg">{s.ppg.toFixed(1)}</div>
            <div
              className={`mono pp-r pp-season-fin${
                s.pos_rank_ppg !== null && s.pos_rank_ppg <= 5 ? ' is-good' : ''}`}
              title={s.pos_rank_ppg !== null && s.pos_rank_ppg_n !== null
                ? `${ordinal(s.pos_rank_ppg)} of ${s.pos_rank_ppg_n} ${position}s by points per game`
                : `${ordinal(s.pos_finish)} at the position by total points`}
            >
              {finish}
            </div>

            {/* Average as a bar, the week-to-week spread as the whisker
                across it. Not a box plot: +-1 standard deviation is what the
                payload carries, so that is what is drawn and what the number
                beside it says. */}
            <div className="pp-spread">
              <div
                className="pp-spread-track"
                title={`${s.ppg.toFixed(1)} average, most weeks between ${lo.toFixed(1)} and ${hi.toFixed(1)}`}
              >
                <div className="pp-spread-bar" style={{ width: pct(s.ppg) }} />
                {std > 0 && (
                  <div
                    className="pp-spread-whisker"
                    style={{ left: pct(lo), width: pct(hi - lo) }}
                  />
                )}
              </div>
              <div className="mono pp-spread-num">±{std.toFixed(1)}</div>
            </div>

            <div className="pp-cv">
              {s.cv === null ? (
                <div className="pp-dim pp-r pp-cv-empty">—</div>
              ) : (
                <>
                  <div className="pp-cv-track" title={`Coefficient of variation ${s.cv.toFixed(2)}${
                    s.cv_pos_median !== null ? `, position median ${s.cv_pos_median.toFixed(2)}` : ''}`}
                  >
                    {s.cv_pos_median !== null && (
                      <div className="pp-cv-median" style={{ left: `${trackPct(s.cv_pos_median)}%` }} />
                    )}
                    <div className={`pp-cv-dot is-${tone}`} style={{ left: `${trackPct(s.cv)}%` }} />
                  </div>
                  <div className={`mono pp-cv-rank is-${tone}`}>
                    {s.cv_rank !== null && s.cv_rank_n !== null
                      ? `${s.cv_rank}/${s.cv_rank_n}`
                      : s.cv.toFixed(2)}
                  </div>
                </>
              )}
            </div>
          </div>
        )
      })}

      <p className="pp-note">
        The tick on each track is the position median.{' '}
        {ranked.length === 0
          ? 'No season here carries a ranked coefficient.'
          : steady === ranked.length
            ? `Once his scoring is divided out he is in the steadier third of the position ${
              ranked.length === 1 ? 'in the one season ranked here' : 'in every season'}.`
            : steady === 0
              ? `Once his scoring is divided out he is outside the steadier third in all ${ranked.length}.`
              : `Once his scoring is divided out he is in the steadier third in ${steady} of ${ranked.length}.`}
        {' '}Rank the same seasons on the raw ± instead and the biggest
        scorers come out the wildest every time — which measures how much
        they score, not how reliably.
      </p>
    </div>
  )
}
